// SPDX-License-Identifier: GPL-3.0-or-later
//
// sovmr — sovereign media re-encoder
// Copyright (C) 2026  AdmiralGDR
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! HEVC (H.265) bitstream facts for the sovereign muxer: Annex B NAL
//! walking, VPS/SPS/PPS harvesting, a strict SPS reader, and the
//! `HEVCDecoderConfigurationRecord` (`hvcC`) builder per ISO/IEC
//! 14496-15 §8.3.3.1.
//!
//! Everything here is byte-exact and allocation-light; the SPS reader
//! parses only up to the fields `hvcC` needs (profile/tier/level,
//! chroma format, bit depths) and refuses malformed input with a typed
//! error instead of guessing (negative trust).

/// NAL unit types carried in the `hvcC` parameter-set arrays.
pub const NAL_VPS: u8 = 32;
/// Sequence parameter set.
pub const NAL_SPS: u8 = 33;
/// Picture parameter set.
pub const NAL_PPS: u8 = 34;

/// `lengthSizeMinusOne` the muxer writes: 4-byte NAL length prefixes.
pub const LENGTH_SIZE_MINUS_ONE: u8 = 3;
/// Bytes per NAL length prefix in muxed samples.
pub const NAL_PREFIX_LEN: usize = LENGTH_SIZE_MINUS_ONE as usize + 1;

/// Why HEVC parsing failed. Each variant names the lying structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcError {
    /// The SPS ended before a mandatory field.
    Truncated(&'static str),
    /// A field carries a value the spec forbids.
    Malformed(&'static str),
}

impl std::fmt::Display for HevcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HevcError::Truncated(what) => write!(f, "SPS truncated at {what}"),
            HevcError::Malformed(what) => write!(f, "SPS malformed: {what}"),
        }
    }
}

/// The NAL type of an HEVC NAL unit (bits 1..7 of the first header byte).
pub fn nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| (b >> 1) & 0x3F)
}

/// Iterator over the NAL units of an Annex B stream (start-code
/// separated). Yields payload slices *without* start codes; a trailing
/// zero that belongs to a 4-byte start code is trimmed.
#[derive(Debug)]
pub struct AnnexBNals<'a> {
    data: &'a [u8],
    /// Byte position just past the previous start code.
    pos: usize,
}

/// Walk the NAL units of `annex_b`. Data before the first start code is
/// skipped (there should be none in a well-formed stream).
pub fn annex_b_nals(annex_b: &[u8]) -> AnnexBNals<'_> {
    AnnexBNals {
        pos: find_start_code(annex_b, 0).unwrap_or(annex_b.len()),
        data: annex_b,
    }
}

/// Position just past the next `00 00 01` at or after `from`.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some(i + 3);
        }
        i += 1;
    }
    None
}

impl<'a> Iterator for AnnexBNals<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let start = self.pos;
        let (mut end, next) = match find_start_code(self.data, start) {
            Some(past_code) => (past_code - 3, past_code),
            None => (self.data.len(), self.data.len()),
        };
        // A 4-byte start code (00 00 00 01) leaves its leading zero on
        // the previous NAL; give it back.
        if end > start && self.data[end - 1] == 0 {
            end -= 1;
        }
        self.pos = next;
        (end > start).then(|| &self.data[start..end])
    }
}

/// The three parameter-set NALs an `hvc1` track stores out-of-band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcParameterSets {
    /// Video parameter set (NAL type 32), without start code.
    pub vps: Vec<u8>,
    /// Sequence parameter set (NAL type 33).
    pub sps: Vec<u8>,
    /// Picture parameter set (NAL type 34).
    pub pps: Vec<u8>,
}

/// Harvest VPS/SPS/PPS from an Annex B access unit (NVENC attaches all
/// three to IDR output). `None` until every one of the three is present.
pub fn extract_parameter_sets(annex_b: &[u8]) -> Option<HevcParameterSets> {
    let (mut vps, mut sps, mut pps) = (None, None, None);
    for nal in annex_b_nals(annex_b) {
        match nal_type(nal) {
            Some(NAL_VPS) => vps.get_or_insert_with(|| nal.to_vec()),
            Some(NAL_SPS) => sps.get_or_insert_with(|| nal.to_vec()),
            Some(NAL_PPS) => pps.get_or_insert_with(|| nal.to_vec()),
            _ => continue,
        };
    }
    Some(HevcParameterSets {
        vps: vps?,
        sps: sps?,
        pps: pps?,
    })
}

/// Convert one Annex B access unit into a length-prefixed `hvc1` sample
/// in `out` (cleared, capacity reused). Parameter-set NALs are dropped:
/// an `hvc1` track carries them exclusively in the `hvcC` record.
pub fn length_prefixed_sample(annex_b: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for nal in annex_b_nals(annex_b) {
        if matches!(nal_type(nal), Some(NAL_VPS..=NAL_PPS)) {
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
}

// ---- SPS reading ------------------------------------------------------------

/// Strip emulation-prevention bytes: every `00 00 03` becomes `00 00`.
fn strip_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len());
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue; // the escape byte itself is not RBSP data
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// MSB-first bit reader over an RBSP with checked reads.
struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position.
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Read `n` bits (n <= 32) as an unsigned value.
    fn u(&mut self, n: u32, what: &'static str) -> Result<u32, HevcError> {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self
                .data
                .get(self.pos / 8)
                .ok_or(HevcError::Truncated(what))?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | u32::from(bit);
            self.pos += 1;
        }
        Ok(v)
    }

    /// Skip `n` bits, still bounds-checked.
    fn skip(&mut self, n: u32, what: &'static str) -> Result<(), HevcError> {
        let end = self.pos + n as usize;
        if end > self.data.len() * 8 {
            return Err(HevcError::Truncated(what));
        }
        self.pos = end;
        Ok(())
    }

    /// Unsigned Exp-Golomb (`ue(v)`).
    fn ue(&mut self, what: &'static str) -> Result<u32, HevcError> {
        let mut leading = 0u32;
        while self.u(1, what)? == 0 {
            leading += 1;
            if leading > 31 {
                return Err(HevcError::Malformed(what));
            }
        }
        let tail = self.u(leading, what)?;
        Ok((1u32 << leading) - 1 + tail)
    }
}

/// The SPS facts `hvcC` republishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpsFacts {
    /// `general_profile_space` (2 bits).
    pub profile_space: u8,
    /// `general_tier_flag`.
    pub tier: u8,
    /// `general_profile_idc` (1 = Main, 2 = Main 10, ...).
    pub profile_idc: u8,
    /// `general_profile_compatibility_flags` (32 bits, verbatim).
    pub compat_flags: u32,
    /// `general_constraint_indicator_flags` (48 bits, verbatim).
    pub constraint_flags: [u8; 6],
    /// `general_level_idc` (level × 30, e.g. 120 = level 4.0).
    pub level_idc: u8,
    /// `sps_max_sub_layers_minus1`.
    pub max_sub_layers_minus1: u8,
    /// `sps_temporal_id_nesting_flag`.
    pub temporal_id_nesting: u8,
    /// `chroma_format_idc` (1 = 4:2:0).
    pub chroma_format_idc: u8,
    /// `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8`.
    pub bit_depth_chroma_minus8: u8,
}

/// SPS NAL header length (2 bytes).
const NAL_HEADER_LEN: usize = 2;
/// Bits of one general/sub-layer profile block inside profile_tier_level:
/// space(2) + tier(1) + idc(5) + compat(32) + constraints(48).
const PTL_PROFILE_BITS: u32 = 88;

/// Parse an SPS NAL (with its 2-byte header, emulation bytes intact) up
/// to the bit-depth fields — everything `hvcC` needs.
///
/// # Errors
///
/// [`HevcError`] naming the field where the SPS ended or contradicted
/// the spec.
pub fn parse_sps(sps_nal: &[u8]) -> Result<SpsFacts, HevcError> {
    if nal_type(sps_nal) != Some(NAL_SPS) {
        return Err(HevcError::Malformed("not an SPS NAL"));
    }
    let rbsp = strip_emulation_prevention(
        sps_nal
            .get(NAL_HEADER_LEN..)
            .ok_or(HevcError::Truncated("NAL header"))?,
    );
    let mut br = BitReader::new(&rbsp);

    br.skip(4, "sps_video_parameter_set_id")?;
    let max_sub_layers_minus1 = br.u(3, "sps_max_sub_layers_minus1")? as u8;
    let temporal_id_nesting = br.u(1, "sps_temporal_id_nesting_flag")? as u8;

    // profile_tier_level(1, sps_max_sub_layers_minus1)
    let profile_space = br.u(2, "general_profile_space")? as u8;
    let tier = br.u(1, "general_tier_flag")? as u8;
    let profile_idc = br.u(5, "general_profile_idc")? as u8;
    let compat_flags = br.u(32, "general_profile_compatibility_flags")?;
    let mut constraint_flags = [0u8; 6];
    for byte in &mut constraint_flags {
        *byte = br.u(8, "general_constraint_indicator_flags")? as u8;
    }
    let level_idc = br.u(8, "general_level_idc")? as u8;
    if max_sub_layers_minus1 > 0 {
        let mut profile_present = [false; 7];
        let mut level_present = [false; 7];
        for i in 0..usize::from(max_sub_layers_minus1) {
            profile_present[i] = br.u(1, "sub_layer_profile_present_flag")? == 1;
            level_present[i] = br.u(1, "sub_layer_level_present_flag")? == 1;
        }
        // reserved_zero_2bits alignment up to 8 sub-layer slots.
        br.skip(
            2 * (8 - u32::from(max_sub_layers_minus1)),
            "reserved_zero_2bits",
        )?;
        for i in 0..usize::from(max_sub_layers_minus1) {
            if profile_present[i] {
                br.skip(PTL_PROFILE_BITS, "sub_layer profile")?;
            }
            if level_present[i] {
                br.skip(8, "sub_layer_level_idc")?;
            }
        }
    }

    br.ue("sps_seq_parameter_set_id")?;
    let chroma_format_idc = br.ue("chroma_format_idc")?;
    if chroma_format_idc > 3 {
        return Err(HevcError::Malformed("chroma_format_idc > 3"));
    }
    if chroma_format_idc == 3 {
        br.skip(1, "separate_colour_plane_flag")?;
    }
    br.ue("pic_width_in_luma_samples")?;
    br.ue("pic_height_in_luma_samples")?;
    if br.u(1, "conformance_window_flag")? == 1 {
        for what in [
            "conf_win_left_offset",
            "conf_win_right_offset",
            "conf_win_top_offset",
            "conf_win_bottom_offset",
        ] {
            br.ue(what)?;
        }
    }
    let bit_depth_luma_minus8 = br.ue("bit_depth_luma_minus8")?;
    let bit_depth_chroma_minus8 = br.ue("bit_depth_chroma_minus8")?;
    if bit_depth_luma_minus8 > 8 || bit_depth_chroma_minus8 > 8 {
        return Err(HevcError::Malformed("bit depth exceeds 16 bits"));
    }

    Ok(SpsFacts {
        profile_space,
        tier,
        profile_idc,
        compat_flags,
        constraint_flags,
        level_idc,
        max_sub_layers_minus1,
        temporal_id_nesting,
        chroma_format_idc: chroma_format_idc as u8,
        bit_depth_luma_minus8: bit_depth_luma_minus8 as u8,
        bit_depth_chroma_minus8: bit_depth_chroma_minus8 as u8,
    })
}

/// Build the `hvcC` box payload (`HEVCDecoderConfigurationRecord`,
/// ISO/IEC 14496-15 §8.3.3.1) from harvested parameter sets. All
/// profile/tier/level facts are read from the SPS itself, never assumed.
///
/// # Errors
///
/// [`HevcError`] when the SPS cannot be parsed.
pub fn hvcc_record(ps: &HevcParameterSets) -> Result<Vec<u8>, HevcError> {
    let sps = parse_sps(&ps.sps)?;
    let mut v = Vec::with_capacity(38 + ps.vps.len() + ps.sps.len() + ps.pps.len());
    v.push(1); // configurationVersion
    v.push((sps.profile_space << 6) | (sps.tier << 5) | sps.profile_idc);
    v.extend_from_slice(&sps.compat_flags.to_be_bytes());
    v.extend_from_slice(&sps.constraint_flags);
    v.push(sps.level_idc);
    // reserved(4) = 1111 + min_spatial_segmentation_idc(12) = 0 (unknown).
    v.extend_from_slice(&0xF000u16.to_be_bytes());
    v.push(0xFC); // reserved(6) + parallelismType(2) = 0 (unknown)
    v.push(0xFC | sps.chroma_format_idc);
    v.push(0xF8 | sps.bit_depth_luma_minus8);
    v.push(0xF8 | sps.bit_depth_chroma_minus8);
    v.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate = 0 (unspecified)
                                              // constantFrameRate(2)=0 | numTemporalLayers(3) | temporalIdNested(1)
                                              // | lengthSizeMinusOne(2)
    v.push(
        ((sps.max_sub_layers_minus1 + 1) << 3)
            | (sps.temporal_id_nesting << 2)
            | LENGTH_SIZE_MINUS_ONE,
    );
    let arrays: [(u8, &[u8]); 3] = [(NAL_VPS, &ps.vps), (NAL_SPS, &ps.sps), (NAL_PPS, &ps.pps)];
    v.push(arrays.len() as u8); // numOfArrays
    for (nal_ty, nal) in arrays {
        v.push(0x80 | nal_ty); // array_completeness=1 | reserved(1)=0 | type
        v.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        v.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        v.extend_from_slice(nal);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSB-first bit writer for handcrafting SPS fixtures.
    struct BitWriter {
        bytes: Vec<u8>,
        used: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter {
                bytes: Vec::new(),
                used: 0,
            }
        }
        fn u(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                if self.used.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = (value >> i) & 1;
                let byte = self.bytes.last_mut().unwrap();
                *byte |= (bit as u8) << (7 - (self.used % 8));
                self.used += 1;
            }
        }
        fn ue(&mut self, value: u32) {
            let coded = value + 1;
            let bits = 32 - coded.leading_zeros();
            self.u(0, bits - 1);
            self.u(coded, bits);
        }
        fn finish(mut self) -> Vec<u8> {
            self.u(1, 1); // rbsp_stop_one_bit
            self.bytes
        }
    }

    /// A minimal Main-profile 4:2:0 8-bit SPS: exactly the fields
    /// [`parse_sps`] consumes, with realistic values.
    fn synthetic_sps() -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(0, 4); // sps_video_parameter_set_id
        w.u(0, 3); // sps_max_sub_layers_minus1
        w.u(1, 1); // sps_temporal_id_nesting_flag
        w.u(0, 2); // general_profile_space
        w.u(0, 1); // general_tier_flag
        w.u(1, 5); // general_profile_idc = Main
        w.u(0x6000_0000, 32); // compatibility: Main + Main 10
        w.u(0x90, 8); // progressive + frame-only constraint flags
        for _ in 0..5 {
            w.u(0, 8); // remaining constraint bits
        }
        w.u(93, 8); // general_level_idc = 3.1
        w.ue(0); // sps_seq_parameter_set_id
        w.ue(1); // chroma_format_idc = 4:2:0
        w.ue(640); // pic_width_in_luma_samples
        w.ue(360); // pic_height_in_luma_samples
        w.u(0, 1); // conformance_window_flag
        w.ue(0); // bit_depth_luma_minus8
        w.ue(0); // bit_depth_chroma_minus8
        let mut nal = vec![NAL_SPS << 1, 0x01]; // NAL header
        nal.extend(w.finish());
        nal
    }

    fn annex_b_of(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, nal) in nals.iter().enumerate() {
            if i == 0 {
                out.extend_from_slice(&[0, 0, 0, 1]); // 4-byte first code
            } else {
                out.extend_from_slice(&[0, 0, 1]);
            }
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn annex_b_walker_handles_both_start_code_lengths() {
        let a = [NAL_VPS << 1, 0x01, 0xAA];
        let b = [NAL_SPS << 1, 0x01, 0xBB, 0xCC];
        let stream = annex_b_of(&[&a, &b]);
        let nals: Vec<&[u8]> = annex_b_nals(&stream).collect();
        assert_eq!(nals, vec![&a[..], &b[..]]);
    }

    #[test]
    fn four_byte_code_zero_is_not_glued_to_previous_nal() {
        // a | 00 00 00 01 | b — the zero belongs to the start code.
        let mut stream = vec![0, 0, 1, 0x40, 0x01, 0xAA];
        stream.extend_from_slice(&[0, 0, 0, 1, 0x42, 0x01, 0xBB]);
        let nals: Vec<&[u8]> = annex_b_nals(&stream).collect();
        assert_eq!(nals[0], &[0x40, 0x01, 0xAA]);
        assert_eq!(nals[1], &[0x42, 0x01, 0xBB]);
    }

    #[test]
    fn garbage_without_start_codes_yields_nothing() {
        assert_eq!(annex_b_nals(&[0xFF; 16]).count(), 0);
        assert_eq!(annex_b_nals(&[]).count(), 0);
    }

    #[test]
    fn parameter_sets_are_harvested_regardless_of_order() {
        let vps = [NAL_VPS << 1, 0x01, 0x0C];
        let sps = synthetic_sps();
        let pps = [NAL_PPS << 1, 0x01, 0xC0];
        let idr = [19 << 1, 0x01, 0x11, 0x22]; // IDR_W_RADL
        let stream = annex_b_of(&[&pps, &vps, &sps, &idr]);
        let ps = extract_parameter_sets(&stream).expect("all three present");
        assert_eq!(ps.vps, vps);
        assert_eq!(ps.sps, sps);
        assert_eq!(ps.pps, pps);
    }

    #[test]
    fn missing_any_parameter_set_yields_none() {
        let vps = [NAL_VPS << 1, 0x01];
        let pps = [NAL_PPS << 1, 0x01];
        let stream = annex_b_of(&[&vps, &pps]); // no SPS
        assert_eq!(extract_parameter_sets(&stream), None);
    }

    #[test]
    fn length_prefixing_drops_parameter_sets_and_keeps_slices() {
        let sps = synthetic_sps();
        let idr: &[u8] = &[19 << 1, 0x01, 0x11, 0x22, 0x33];
        let sei: &[u8] = &[39 << 1, 0x01, 0x05];
        let stream = annex_b_of(&[&sps, sei, idr]);
        let mut out = Vec::new();
        length_prefixed_sample(&stream, &mut out);
        let mut expect = Vec::new();
        expect.extend_from_slice(&(sei.len() as u32).to_be_bytes());
        expect.extend_from_slice(sei);
        expect.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        expect.extend_from_slice(idr);
        assert_eq!(out, expect, "SPS dropped, SEI+slice length-prefixed");
    }

    #[test]
    fn sps_facts_survive_the_round_trip() {
        let facts = parse_sps(&synthetic_sps()).expect("valid SPS");
        assert_eq!(facts.profile_idc, 1, "Main profile");
        assert_eq!(facts.tier, 0);
        assert_eq!(facts.level_idc, 93);
        assert_eq!(facts.compat_flags, 0x6000_0000);
        assert_eq!(facts.constraint_flags, [0x90, 0, 0, 0, 0, 0]);
        assert_eq!(facts.chroma_format_idc, 1, "4:2:0");
        assert_eq!(facts.bit_depth_luma_minus8, 0);
        assert_eq!(facts.temporal_id_nesting, 1);
    }

    #[test]
    fn emulation_prevention_bytes_are_stripped() {
        assert_eq!(
            strip_emulation_prevention(&[0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x03]),
            vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x03]
        );
        // 03 not preceded by two zeros is data, not an escape.
        assert_eq!(
            strip_emulation_prevention(&[0x01, 0x03, 0x02]),
            vec![0x01, 0x03, 0x02]
        );
    }

    #[test]
    fn truncated_sps_names_the_missing_field() {
        let sps = synthetic_sps();
        let cut = &sps[..6]; // ends inside the compatibility flags
        match parse_sps(cut) {
            Err(HevcError::Truncated(what)) => {
                assert!(what.contains("compatibility"), "{what}");
            }
            other => panic!("expected truncation error, got {other:?}"),
        }
        assert!(parse_sps(&[]).is_err());
        assert!(
            parse_sps(&[NAL_VPS << 1, 0x01, 0xFF]).is_err(),
            "wrong NAL type must be rejected"
        );
    }

    #[test]
    fn hvcc_record_matches_the_spec_layout() {
        let ps = HevcParameterSets {
            vps: vec![NAL_VPS << 1, 0x01, 0x0C],
            sps: synthetic_sps(),
            pps: vec![NAL_PPS << 1, 0x01, 0xC0],
        };
        let rec = hvcc_record(&ps).expect("valid record");
        assert_eq!(rec[0], 1, "configurationVersion");
        assert_eq!(rec[1], 0x01, "space 0 / tier 0 / profile Main");
        assert_eq!(&rec[2..6], &0x6000_0000u32.to_be_bytes());
        assert_eq!(&rec[6..12], &[0x90, 0, 0, 0, 0, 0]);
        assert_eq!(rec[12], 93, "level_idc");
        assert_eq!(&rec[13..15], &[0xF0, 0x00], "min_spatial_segmentation");
        assert_eq!(rec[15], 0xFC, "parallelismType");
        assert_eq!(rec[16], 0xFC | 1, "chromaFormat 4:2:0");
        assert_eq!(rec[17], 0xF8, "bitDepthLumaMinus8");
        assert_eq!(rec[18], 0xF8, "bitDepthChromaMinus8");
        assert_eq!(&rec[19..21], &[0, 0], "avgFrameRate");
        assert_eq!(
            rec[21],
            (1 << 3) | (1 << 2) | LENGTH_SIZE_MINUS_ONE,
            "1 temporal layer, nested, 4-byte lengths"
        );
        assert_eq!(rec[22], 3, "numOfArrays");
        // First array: completeness | VPS type, one NAL, its bytes.
        assert_eq!(rec[23], 0x80 | NAL_VPS);
        assert_eq!(&rec[24..26], &1u16.to_be_bytes());
        assert_eq!(&rec[26..28], &(ps.vps.len() as u16).to_be_bytes());
        assert_eq!(&rec[28..28 + ps.vps.len()], &ps.vps[..]);
    }
}
