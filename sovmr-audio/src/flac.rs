// SPDX-License-Identifier: AGPL-3.0-or-later
//
// sovmr — sovereign media re-encoder
// Copyright (C) 2026  sovmr contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! FLAC: lossless, and therefore exact or broken.
//!
//! # How it compresses
//!
//! A predictor guesses each sample from the ones before it, and only the
//! *error* is stored. Music is smooth enough that the error is small, and
//! small numbers code shorter — that is the whole idea.
//!
//! Three predictors appear here. `CONSTANT` says the block is one
//! repeated value. `FIXED` uses one of five hard-coded difference
//! formulas that need no coefficients transmitted. `LPC` sends its own
//! coefficients and suits material the fixed set predicts badly.
//! `VERBATIM` gives up and stores the samples, which is what an encoder
//! does with noise.
//!
//! The errors are then Rice-coded: a value's magnitude decides its
//! length, so quiet passages cost almost nothing.
//!
//! # Nothing here may be approximate
//!
//! Every arithmetic step below is integer and exact. A rounding anywhere
//! would make the output *nearly* the original, which for lossless
//! material is simply wrong — and wrong in a way that only shows up on
//! the one recording somebody cares about.

use crate::bits::BitReader;

/// Largest LPC order FLAC allows.
const MAX_LPC_ORDER: usize = 32;
/// Largest block size FLAC allows, in samples per channel.
const MAX_BLOCK_SIZE: usize = 65_535;
/// Largest channel count FLAC allows.
const MAX_CHANNELS: usize = 8;

/// Why a FLAC stream could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlacError {
    /// The `STREAMINFO` block was absent or too short.
    BadStreamInfo,
    /// A frame did not begin with the sync code.
    NoSync,
    /// The buffer ended inside a frame.
    Truncated,
    /// A field held a value the format reserves or forbids.
    Invalid(&'static str),
    /// A declared size exceeds what the format allows.
    TooLarge(&'static str),
}

impl std::fmt::Display for FlacError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlacError::BadStreamInfo => f.write_str("STREAMINFO missing or truncated"),
            FlacError::NoSync => f.write_str("frame does not start with the sync code"),
            FlacError::Truncated => f.write_str("buffer ended inside a frame"),
            FlacError::Invalid(what) => write!(f, "reserved or invalid {what}"),
            FlacError::TooLarge(what) => write!(f, "{what} exceeds what FLAC allows"),
        }
    }
}

impl std::error::Error for FlacError {}

/// What a stream declares about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Significant bits per sample, 4 to 32.
    pub bits_per_sample: u16,
}

impl StreamInfo {
    /// Read a `STREAMINFO` block.
    ///
    /// Accepts either the bare block — which is what Matroska puts in
    /// `CodecPrivate` — or a whole `fLaC` header with its metadata
    /// blocks, since MP4 and raw files carry that instead. Telling them
    /// apart by the magic is cheaper than asking the caller which it has.
    ///
    /// # Errors
    ///
    /// [`FlacError::BadStreamInfo`] when the block is missing or shorter
    /// than the fields it must contain.
    pub fn parse(data: &[u8]) -> Result<StreamInfo, FlacError> {
        // A full header opens with "fLaC", then metadata blocks whose
        // first is always STREAMINFO: a 4-byte block header, then 34
        // bytes of content.
        let block = if data.starts_with(b"fLaC") {
            data.get(8..).ok_or(FlacError::BadStreamInfo)?
        } else {
            data
        };
        if block.len() < 18 {
            return Err(FlacError::BadStreamInfo);
        }
        // Layout after the min/max block and frame sizes (10 bytes):
        // 20 bits sample rate, 3 bits channels-1, 5 bits depth-1.
        let packed = u32::from_be_bytes([block[10], block[11], block[12], block[13]]);
        let sample_rate = packed >> 12;
        let channels = ((packed >> 9) & 0x7) as u16 + 1;
        let bits_per_sample = ((packed >> 4) & 0x1F) as u16 + 1;
        if channels == 0 || channels as usize > MAX_CHANNELS {
            return Err(FlacError::BadStreamInfo);
        }
        if !(4..=32).contains(&bits_per_sample) {
            return Err(FlacError::BadStreamInfo);
        }
        Ok(StreamInfo {
            sample_rate,
            channels,
            bits_per_sample,
        })
    }
}

/// How the channels of a frame relate to each other.
///
/// Stereo correlates strongly, so FLAC often stores a difference channel
/// instead of one of the pair — the difference is quieter, and quiet
/// codes shorter. The side channel carries one extra bit, because a
/// difference of two n-bit numbers needs n+1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelLayout {
    /// Each channel stands alone.
    Independent(usize),
    /// Left, then (left − right).
    LeftSide,
    /// (left − right), then right.
    RightSide,
    /// (left + right) / 2, then (left − right).
    MidSide,
}

impl ChannelLayout {
    fn from_code(code: u32) -> Result<ChannelLayout, FlacError> {
        Ok(match code {
            0..=7 => ChannelLayout::Independent(code as usize + 1),
            8 => ChannelLayout::LeftSide,
            9 => ChannelLayout::RightSide,
            10 => ChannelLayout::MidSide,
            _ => return Err(FlacError::Invalid("channel assignment")),
        })
    }

    fn channel_count(self) -> usize {
        match self {
            ChannelLayout::Independent(n) => n,
            _ => 2,
        }
    }

    /// Whether channel `i` is a difference channel, and so one bit wider.
    fn is_side(self, i: usize) -> bool {
        matches!(
            (self, i),
            (ChannelLayout::LeftSide, 1)
                | (ChannelLayout::RightSide, 0)
                | (ChannelLayout::MidSide, 1)
        )
    }
}

/// One decoded frame: samples per channel, not yet interleaved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// One vector per channel, each `block_size` long.
    pub channels: Vec<Vec<i32>>,
    /// Bits each sample actually resolves.
    pub bits_per_sample: u16,
    /// Bytes of the input this frame occupied.
    pub consumed: usize,
}

/// Decode one FLAC frame from the start of `data`.
///
/// # Errors
///
/// [`FlacError`] naming what did not add up. A frame is never returned
/// partially decoded: either every channel came out, or the caller is
/// told why not.
pub fn decode_frame(data: &[u8], info: &StreamInfo) -> Result<Frame, FlacError> {
    let mut r = BitReader::new(data);

    // ---- frame header ------------------------------------------------
    if r.u32(14).ok_or(FlacError::Truncated)? != 0b11_1111_1111_1110 {
        return Err(FlacError::NoSync);
    }
    // Reserved bit, then the blocking strategy — which changes only how
    // the frame is numbered, not how it decodes.
    r.u32(1).ok_or(FlacError::Truncated)?;
    let variable_blocking = r.u32(1).ok_or(FlacError::Truncated)? == 1;

    let block_code = r.u32(4).ok_or(FlacError::Truncated)?;
    let rate_code = r.u32(4).ok_or(FlacError::Truncated)?;
    let channel_code = r.u32(4).ok_or(FlacError::Truncated)?;
    let depth_code = r.u32(3).ok_or(FlacError::Truncated)?;
    r.u32(1).ok_or(FlacError::Truncated)?; // reserved

    let layout = ChannelLayout::from_code(channel_code)?;
    let bits_per_sample = match depth_code {
        0 => info.bits_per_sample,
        1 => 8,
        2 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 32,
        _ => return Err(FlacError::Invalid("sample size")),
    };

    // The coded frame or sample number, in a UTF-8-like escape. Its
    // value is not needed to decode, only its length, but skipping it
    // blindly would desynchronise the reader.
    skip_coded_number(&mut r, variable_blocking)?;

    let block_size = match block_code {
        0 => return Err(FlacError::Invalid("block size")),
        1 => 192,
        2..=5 => 576 << (block_code - 2),
        6 => r.u32(8).ok_or(FlacError::Truncated)? as usize + 1,
        7 => r.u32(16).ok_or(FlacError::Truncated)? as usize + 1,
        _ => 256 << (block_code - 8),
    };
    if block_size > MAX_BLOCK_SIZE {
        return Err(FlacError::TooLarge("block size"));
    }
    // Sample-rate escapes carry their value here; the value itself is
    // the stream's business, but the bits must be consumed.
    match rate_code {
        12 => {
            r.u32(8).ok_or(FlacError::Truncated)?;
        }
        13 | 14 => {
            r.u32(16).ok_or(FlacError::Truncated)?;
        }
        15 => return Err(FlacError::Invalid("sample rate")),
        _ => {}
    }
    r.u32(8).ok_or(FlacError::Truncated)?; // header CRC-8

    // ---- subframes ---------------------------------------------------
    let count = layout.channel_count();
    let mut channels = Vec::with_capacity(count);
    for i in 0..count {
        let width = bits_per_sample + u16::from(layout.is_side(i));
        channels.push(decode_subframe(&mut r, block_size, width)?);
    }

    // ---- undo the channel difference ---------------------------------
    restore_stereo(&mut channels, layout);

    r.align();
    r.u32(16).ok_or(FlacError::Truncated)?; // frame CRC-16
    Ok(Frame {
        channels,
        bits_per_sample,
        consumed: r.byte_position(),
    })
}

/// Skip the UTF-8-like coded frame or sample number.
fn skip_coded_number(r: &mut BitReader<'_>, variable: bool) -> Result<(), FlacError> {
    let first = r.u32(8).ok_or(FlacError::Truncated)? as u8;
    // The leading ones say how many continuation bytes follow, exactly
    // as in UTF-8 — extended to seven bytes so a 36-bit sample number
    // fits when the blocking strategy is variable.
    let extra = match first {
        0x00..=0x7F => 0,
        0xC0..=0xDF => 1,
        0xE0..=0xEF => 2,
        0xF0..=0xF7 => 3,
        0xF8..=0xFB => 4,
        0xFC..=0xFD => 5,
        0xFE if variable => 6,
        _ => return Err(FlacError::Invalid("coded number")),
    };
    for _ in 0..extra {
        r.u32(8).ok_or(FlacError::Truncated)?;
    }
    Ok(())
}

/// Decode one channel's samples.
fn decode_subframe(
    r: &mut BitReader<'_>,
    block_size: usize,
    width: u16,
) -> Result<Vec<i32>, FlacError> {
    if r.u32(1).ok_or(FlacError::Truncated)? != 0 {
        return Err(FlacError::Invalid("subframe padding"));
    }
    let kind = r.u32(6).ok_or(FlacError::Truncated)?;
    // Some encoders find the low bits of every sample are zero — quiet
    // 24-bit material recorded at 16, say — and shift them out. They
    // must be shifted back, or the whole channel plays 2^n too quiet.
    let wasted = if r.u32(1).ok_or(FlacError::Truncated)? == 1 {
        r.unary().ok_or(FlacError::Truncated)? + 1
    } else {
        0
    };
    let effective = u32::from(width)
        .checked_sub(wasted)
        .ok_or(FlacError::Invalid("wasted bits"))?;
    if effective == 0 || effective > 32 {
        return Err(FlacError::Invalid("sample width"));
    }

    let mut samples = match kind {
        0 => {
            // CONSTANT: one value, repeated.
            let v = r.i32(effective).ok_or(FlacError::Truncated)?;
            vec![v; block_size]
        }
        1 => {
            // VERBATIM: no prediction worked, so the samples are stored.
            let mut out = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                out.push(r.i32(effective).ok_or(FlacError::Truncated)?);
            }
            out
        }
        8..=12 => {
            let order = (kind - 8) as usize;
            decode_fixed(r, block_size, effective, order)?
        }
        32..=63 => {
            let order = (kind - 31) as usize;
            decode_lpc(r, block_size, effective, order)?
        }
        _ => return Err(FlacError::Invalid("subframe type")),
    };

    if wasted > 0 {
        for s in &mut samples {
            *s = ((*s as i64) << wasted) as i32;
        }
    }
    Ok(samples)
}

/// FIXED prediction: five hard-coded difference formulas.
///
/// Order 0 stores the signal, order 1 its first difference, and so on —
/// each order predicts a polynomial one degree higher, which is why
/// smooth material needs no transmitted coefficients at all.
fn decode_fixed(
    r: &mut BitReader<'_>,
    block_size: usize,
    width: u32,
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    if order > block_size {
        return Err(FlacError::Invalid("fixed order"));
    }
    let mut out = Vec::with_capacity(block_size);
    for _ in 0..order {
        out.push(r.i32(width).ok_or(FlacError::Truncated)?);
    }
    let residual = decode_residual(r, block_size, order)?;
    for (i, e) in residual.into_iter().enumerate() {
        let n = order + i;
        // Widened to i64 throughout: order 4 sums four samples with
        // coefficients up to 6, which overflows i32 on 32-bit material.
        let p = |k: usize| i64::from(out[n - k]);
        let predicted = match order {
            0 => 0,
            1 => p(1),
            2 => 2 * p(1) - p(2),
            3 => 3 * p(1) - 3 * p(2) + p(3),
            _ => 4 * p(1) - 6 * p(2) + 4 * p(3) - p(4),
        };
        out.push((predicted + i64::from(e)) as i32);
    }
    Ok(out)
}

/// LPC prediction: the encoder sends its own coefficients.
fn decode_lpc(
    r: &mut BitReader<'_>,
    block_size: usize,
    width: u32,
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    if order > MAX_LPC_ORDER || order > block_size {
        return Err(FlacError::Invalid("lpc order"));
    }
    let mut out = Vec::with_capacity(block_size);
    for _ in 0..order {
        out.push(r.i32(width).ok_or(FlacError::Truncated)?);
    }
    let precision = r.u32(4).ok_or(FlacError::Truncated)? + 1;
    if precision > 15 {
        return Err(FlacError::Invalid("lpc precision"));
    }
    // The shift is signed in the format but must not be negative in any
    // real stream; a negative one would mean multiplying up, which the
    // reference encoder never emits.
    let shift = r.i32(5).ok_or(FlacError::Truncated)?;
    if shift < 0 {
        return Err(FlacError::Invalid("lpc shift"));
    }
    let mut coefficients = Vec::with_capacity(order);
    for _ in 0..order {
        coefficients.push(i64::from(r.i32(precision).ok_or(FlacError::Truncated)?));
    }

    let residual = decode_residual(r, block_size, order)?;
    for (i, e) in residual.into_iter().enumerate() {
        let n = order + i;
        // i64 again, and for a sharper reason: 32 coefficients of up to
        // 15 bits against 32-bit samples is far outside i32.
        let mut sum = 0i64;
        for (k, c) in coefficients.iter().enumerate() {
            sum += c * i64::from(out[n - 1 - k]);
        }
        out.push(((sum >> shift) + i64::from(e)) as i32);
    }
    Ok(out)
}

/// Decode the Rice-coded prediction errors of one subframe.
fn decode_residual(
    r: &mut BitReader<'_>,
    block_size: usize,
    order: usize,
) -> Result<Vec<i32>, FlacError> {
    let method = r.u32(2).ok_or(FlacError::Truncated)?;
    let (param_bits, escape) = match method {
        0 => (4u32, 0b1111u32),
        1 => (5, 0b11111),
        _ => return Err(FlacError::Invalid("residual method")),
    };
    let partition_order = r.u32(4).ok_or(FlacError::Truncated)? as usize;
    let partitions = 1usize << partition_order;
    if !block_size.is_multiple_of(partitions) || block_size >> partition_order < order {
        return Err(FlacError::Invalid("partition order"));
    }

    let mut out = Vec::with_capacity(block_size - order);
    for p in 0..partitions {
        // The first partition is short by the warm-up samples, which
        // were transmitted raw and are not predicted.
        let count = (block_size >> partition_order) - if p == 0 { order } else { 0 };
        let param = r.u32(param_bits).ok_or(FlacError::Truncated)?;
        if param == escape {
            // Escape: this partition gave up on Rice coding and stores
            // its errors at a fixed width, which may be zero.
            let raw = r.u32(5).ok_or(FlacError::Truncated)?;
            for _ in 0..count {
                out.push(r.i32(raw).ok_or(FlacError::Truncated)?);
            }
        } else {
            for _ in 0..count {
                let quotient = r.unary().ok_or(FlacError::Truncated)?;
                let remainder = r.u32(param).ok_or(FlacError::Truncated)?;
                let folded = (quotient << param) | remainder;
                // Zig-zag: the low bit is the sign, so small negatives
                // stay small. Reading it as plain unsigned would make
                // every other sample enormous.
                out.push(if folded & 1 == 1 {
                    -((folded >> 1) as i32) - 1
                } else {
                    (folded >> 1) as i32
                });
            }
        }
    }
    Ok(out)
}

/// Undo the left/right/mid/side difference so every channel stands alone.
fn restore_stereo(channels: &mut [Vec<i32>], layout: ChannelLayout) {
    if channels.len() != 2 {
        return;
    }
    let (a, b) = channels.split_at_mut(1);
    let (first, second) = (&mut a[0], &mut b[0]);
    match layout {
        ChannelLayout::LeftSide => {
            // first is left, second is (left − right).
            for i in 0..second.len().min(first.len()) {
                second[i] = first[i].wrapping_sub(second[i]);
            }
        }
        ChannelLayout::RightSide => {
            // first is (left − right), second is right.
            for i in 0..first.len().min(second.len()) {
                first[i] = first[i].wrapping_add(second[i]);
            }
        }
        ChannelLayout::MidSide => {
            for i in 0..first.len().min(second.len()) {
                let side = i64::from(second[i]);
                // The mid channel dropped its lowest bit on the way in;
                // the side channel's parity is what puts it back, and
                // forgetting that leaves every sample one unit low.
                let mid = (i64::from(first[i]) << 1) | (side & 1);
                first[i] = ((mid + side) >> 1) as i32;
                second[i] = ((mid - side) >> 1) as i32;
            }
        }
        ChannelLayout::Independent(_) => {}
    }
}

/// Decode a whole FLAC stream into interleaved samples.
///
/// Samples come back left-aligned in `i32`, matching
/// [`crate::pcm::decode`], so material of different depths mixes without
/// each source being asked how wide it was.
///
/// # Errors
///
/// [`FlacError`] from the first frame that does not decode. Frames
/// already decoded are lost with it: a caller that wants partial output
/// should decode frame by frame.
pub fn decode(data: &[u8], info: &StreamInfo) -> Result<Vec<i32>, FlacError> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < data.len() {
        // A stream ends when what remains is too short to be a frame
        // header rather than at a marker.
        if data.len() - at < 5 {
            break;
        }
        let frame = decode_frame(&data[at..], info)?;
        if frame.consumed == 0 {
            return Err(FlacError::Invalid("frame consumed nothing"));
        }
        let shift = 32 - u32::from(frame.bits_per_sample);
        let len = frame.channels.first().map(Vec::len).unwrap_or(0);
        for i in 0..len {
            for c in &frame.channels {
                out.push(((c[i] as i64) << shift) as i32);
            }
        }
        at += frame.consumed;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `STREAMINFO` block declaring the given shape.
    fn stream_info_block(rate: u32, channels: u16, depth: u16) -> Vec<u8> {
        let mut b = vec![0u8; 34];
        let packed = (rate << 12) | (u32::from(channels - 1) << 9) | (u32::from(depth - 1) << 4);
        b[10..14].copy_from_slice(&packed.to_be_bytes());
        b
    }

    #[test]
    fn stream_info_reads_the_shape_a_stream_declares() {
        let b = stream_info_block(44_100, 2, 16);
        let info = StreamInfo::parse(&b).unwrap();
        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn a_full_flac_header_is_accepted_as_well_as_a_bare_block() {
        // Matroska hands over the bare block; MP4 and raw files carry
        // the whole header. Telling them apart by the magic costs less
        // than asking the caller which they have.
        let bare = stream_info_block(48_000, 6, 24);
        let mut full = b"fLaC".to_vec();
        full.extend_from_slice(&[0x80, 0, 0, 34]); // last block, STREAMINFO
        full.extend_from_slice(&bare);
        assert_eq!(StreamInfo::parse(&full), StreamInfo::parse(&bare));
        assert_eq!(StreamInfo::parse(&full).unwrap().channels, 6);
    }

    #[test]
    fn an_impossible_shape_is_refused_rather_than_carried() {
        // A depth of one bit or a truncated block describes no audio;
        // continuing would allocate against a number nobody wrote.
        assert_eq!(StreamInfo::parse(&[]), Err(FlacError::BadStreamInfo));
        assert_eq!(
            StreamInfo::parse(&stream_info_block(44_100, 2, 1)),
            Err(FlacError::BadStreamInfo)
        );
        assert_eq!(StreamInfo::parse(&[0u8; 12]), Err(FlacError::BadStreamInfo));
    }

    #[test]
    fn a_frame_without_the_sync_code_is_refused_not_searched_for() {
        // Resynchronising by scanning is how a decoder turns a corrupt
        // file into confident garbage.
        let info = StreamInfo::parse(&stream_info_block(44_100, 1, 16)).unwrap();
        assert_eq!(
            decode_frame(&[0x00, 0x00, 0x00, 0x00], &info),
            Err(FlacError::NoSync)
        );
    }

    #[test]
    fn side_channels_are_one_bit_wider_than_the_rest() {
        // A difference of two n-bit numbers needs n+1. Reading it at n
        // clips the loudest moments of every stereo frame.
        assert!(ChannelLayout::LeftSide.is_side(1));
        assert!(!ChannelLayout::LeftSide.is_side(0));
        assert!(ChannelLayout::RightSide.is_side(0));
        assert!(!ChannelLayout::RightSide.is_side(1));
        assert!(ChannelLayout::MidSide.is_side(1));
        assert!(!ChannelLayout::Independent(2).is_side(0));
    }

    #[test]
    fn mid_side_recovers_the_low_bit_from_the_side_parity() {
        // The mid channel dropped its lowest bit on the way in. The side
        // channel's parity is what puts it back; without that, every
        // sample of every mid/side frame lands one unit low.
        for (l, rr) in [(100i32, 40i32), (-7, 9), (0, 0), (32767, -32768)] {
            let mid = (l + rr) >> 1;
            let side = l - rr;
            let mut ch = vec![vec![mid], vec![side]];
            restore_stereo(&mut ch, ChannelLayout::MidSide);
            assert_eq!((ch[0][0], ch[1][0]), (l, rr), "mid/side of ({l}, {rr})");
        }
    }

    #[test]
    fn left_and_right_side_layouts_round_trip() {
        for (l, rr) in [(100i32, 40i32), (-7, 9), (0, 0)] {
            let mut ch = vec![vec![l], vec![l - rr]];
            restore_stereo(&mut ch, ChannelLayout::LeftSide);
            assert_eq!((ch[0][0], ch[1][0]), (l, rr), "left/side");

            let mut ch = vec![vec![l - rr], vec![rr]];
            restore_stereo(&mut ch, ChannelLayout::RightSide);
            assert_eq!((ch[0][0], ch[1][0]), (l, rr), "right/side");
        }
    }

    #[test]
    fn independent_channels_are_left_exactly_as_they_came() {
        let mut ch = vec![vec![1, 2, 3], vec![4, 5, 6]];
        restore_stereo(&mut ch, ChannelLayout::Independent(2));
        assert_eq!(ch, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn a_reserved_channel_assignment_is_refused() {
        for code in 11..16 {
            assert_eq!(
                ChannelLayout::from_code(code),
                Err(FlacError::Invalid("channel assignment"))
            );
        }
        assert_eq!(
            ChannelLayout::from_code(0),
            Ok(ChannelLayout::Independent(1))
        );
    }
}
