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

//! Demuxing proper: video *and* audio tracks with full timing.
//!
//! [`find_video_track`] / [`find_audio_tracks`] wire up a track's sample
//! tables — `stsz`, `stco`/`co64`, `stsc`, `stss`, and the timing pair
//! `stts` (decode deltas → DTS) + `ctts` (composition offsets → PTS) —
//! plus the `mdhd` timescale, so every yielded [`Sample`] carries exact
//! DTS/PTS/duration in track units. That timing is what the muxer uses
//! to interleave audio and video for hardware-friendly playback.
//!
//! Zero-copy throughout: samples, parameter sets, and the audio sample
//! entry are `&[u8]` sub-slices of the input buffer. Defensive
//! throughout: every table is validated against its own
//! byte length up front; lying counts are typed [`DemuxError`]s.

use crate::{boxes, find_path, ParseError};

/// Start code separating NAL units in an Annex B stream.
const START_CODE: [u8; 4] = [0, 0, 0, 1];
/// Fixed VisualSampleEntry prefix before child boxes (`avcC`/`hvcC`).
const VISUAL_ENTRY_LEN: usize = 78;
/// Offsets of width/height inside a VisualSampleEntry payload.
const ENTRY_WIDTH_AT: usize = 24;
const ENTRY_HEIGHT_AT: usize = 26;
/// AudioSampleEntry: fixed prefix, channel count and sample rate offsets.
const AUDIO_ENTRY_LEN: usize = 28;
const ENTRY_CHANNELS_AT: usize = 16;
const ENTRY_SAMPLE_RATE_AT: usize = 24;
/// Full boxes (stsd, stsz, ...) start with a 4-byte version/flags word.
const FULL_BOX_HEADER: usize = 4;
/// `hvcC`: offset of the byte carrying lengthSizeMinusOne, then the
/// array count; parameter-set arrays follow.
const HVCC_LENGTH_SIZE_AT: usize = 21;
const HVCC_ARRAYS_AT: usize = 23;

/// Video codec of a track, in container terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// AVC / H.264 (`avc1` + `avcC`).
    H264,
    /// HEVC / H.265 (`hvc1`/`hev1` + `hvcC`).
    Hevc,
    /// AV1 (`av01` + `av1C`) — samples are raw OBUs, no NAL prefixes.
    Av1,
}

/// Why demuxing failed. Every variant names the exact structure that lied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemuxError {
    /// The container itself failed to parse.
    Parse(ParseError),
    /// A required box is absent.
    MissingBox(&'static str),
    /// A box is present but too short for its mandatory fields.
    Truncated(&'static str),
    /// The `stsd` entry is a codec we do not demux.
    UnsupportedCodec([u8; 4]),
    /// The `avcC`/`hvcC` record contradicts itself.
    BadConfigRecord(&'static str),
    /// A sample table's declared count exceeds its actual bytes.
    BadSampleTable(&'static str),
    /// A sample's chunk offset + size runs outside the file.
    SampleOutOfFile {
        /// 0-based sample index.
        index: u32,
    },
    /// A NAL length prefix runs outside its sample.
    BadNalLength {
        /// 0-based sample index.
        sample: u32,
    },
    /// The operation does not apply to this codec (e.g. Annex B
    /// conversion of AV1 OBU samples).
    WrongCodec,
}

impl From<ParseError> for DemuxError {
    fn from(e: ParseError) -> Self {
        DemuxError::Parse(e)
    }
}

impl std::fmt::Display for DemuxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemuxError::Parse(e) => write!(f, "container parse error: {e}"),
            DemuxError::MissingBox(b) => write!(f, "required box missing: {b}"),
            DemuxError::Truncated(b) => write!(f, "box too short for its fields: {b}"),
            DemuxError::UnsupportedCodec(k) => {
                write!(f, "unsupported codec entry: {}", String::from_utf8_lossy(k))
            }
            DemuxError::BadConfigRecord(m) => write!(f, "bad decoder config record: {m}"),
            DemuxError::BadSampleTable(t) => write!(f, "sample table lies about its size: {t}"),
            DemuxError::SampleOutOfFile { index } => {
                write!(f, "sample {index} points outside the file")
            }
            DemuxError::BadNalLength { sample } => {
                write!(f, "NAL length in sample {sample} exceeds the sample")
            }
            DemuxError::WrongCodec => write!(f, "operation does not apply to this codec"),
        }
    }
}

// ---- byte readers (checked, zero-copy) ------------------------------------

fn be_u16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?]))
}

fn be_u32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *d.get(at)?,
        *d.get(at + 1)?,
        *d.get(at + 2)?,
        *d.get(at + 3)?,
    ]))
}

fn be_u64(d: &[u8], at: usize) -> Option<u64> {
    let hi = be_u32(d, at)?;
    let lo = be_u32(d, at + 4)?;
    Some((u64::from(hi) << 32) | u64::from(lo))
}

/// Strip a full box's version/flags word.
fn full_box<'a>(payload: &'a [u8], name: &'static str) -> Result<&'a [u8], DemuxError> {
    payload
        .get(FULL_BOX_HEADER..)
        .ok_or(DemuxError::Truncated(name))
}

// ---- sample tables ---------------------------------------------------------

/// `stsz`: uniform size or per-sample table (validated up front).
#[derive(Debug, Clone, Copy)]
struct SampleSizes<'a> {
    uniform: u32,
    /// Bits per entry in `table` when `uniform` is zero: 32 for `stsz`,
    /// 4, 8 or 16 for the compact `stz2` form.
    field_bits: u8,
    count: u32,
    table: &'a [u8],
}

impl<'a> SampleSizes<'a> {
    /// Entry width of the `stsz` table form.
    const STSZ_FIELD_BITS: u8 = 32;

    fn parse(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "stsz")?;
        let uniform = be_u32(d, 0).ok_or(DemuxError::Truncated("stsz"))?;
        let count = be_u32(d, 4).ok_or(DemuxError::Truncated("stsz"))?;
        let table = &d[8.min(d.len())..];
        if uniform == 0 && (table.len() as u64) < u64::from(count) * 4 {
            return Err(DemuxError::BadSampleTable("stsz"));
        }
        Ok(SampleSizes {
            uniform,
            field_bits: Self::STSZ_FIELD_BITS,
            count,
            table,
        })
    }

    /// `stz2`: the same table with 4-, 8- or 16-bit entries.
    ///
    /// Nothing we write emits it, which is exactly why it must be read:
    /// a muxer that chose the compact form produced a perfectly valid
    /// file, and rejecting it would be our defect, not theirs. Negative
    /// trust cuts both ways — be strict about what a box may contain,
    /// permissive about which legal box a stranger chose.
    fn parse_compact(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "stz2")?;
        // Three reserved bytes, then the field width in bits.
        let field_bits = *d.get(3).ok_or(DemuxError::Truncated("stz2"))?;
        if !matches!(field_bits, 4 | 8 | 16) {
            return Err(DemuxError::BadSampleTable("stz2"));
        }
        let count = be_u32(d, 4).ok_or(DemuxError::Truncated("stz2"))?;
        let table = &d[8.min(d.len())..];
        // 4-bit entries pack two per byte, so an odd count rounds up and
        // the final low nibble is padding.
        let needed = (u64::from(count) * u64::from(field_bits)).div_ceil(8);
        if (table.len() as u64) < needed {
            return Err(DemuxError::BadSampleTable("stz2"));
        }
        Ok(SampleSizes {
            uniform: 0,
            field_bits,
            count,
            table,
        })
    }

    fn get(&self, index: u32) -> u32 {
        if self.uniform != 0 {
            return self.uniform;
        }
        let i = index as usize;
        match self.field_bits {
            4 => {
                // High nibble first, per the spec's bit order.
                let byte = self.table.get(i / 2).copied().unwrap_or(0);
                if i.is_multiple_of(2) {
                    u32::from(byte >> 4)
                } else {
                    u32::from(byte & 0x0F)
                }
            }
            8 => u32::from(self.table.get(i).copied().unwrap_or(0)),
            16 => be_u16(self.table, i * 2).map(u32::from).unwrap_or(0),
            _ => be_u32(self.table, i * 4).unwrap_or(0),
        }
    }
}

/// `stco` (u32) or `co64` (u64) chunk offsets.
#[derive(Debug, Clone, Copy)]
struct ChunkOffsets<'a> {
    wide: bool,
    count: u32,
    table: &'a [u8],
}

impl<'a> ChunkOffsets<'a> {
    fn parse(payload: &'a [u8], wide: bool, name: &'static str) -> Result<Self, DemuxError> {
        let d = full_box(payload, name)?;
        let count = be_u32(d, 0).ok_or(DemuxError::Truncated(name))?;
        let table = &d[4.min(d.len())..];
        let entry = if wide { 8 } else { 4 };
        if (table.len() as u64) < u64::from(count) * entry {
            return Err(DemuxError::BadSampleTable(name));
        }
        Ok(ChunkOffsets { wide, count, table })
    }

    fn get(&self, chunk: u32) -> Option<u64> {
        if chunk >= self.count {
            return None;
        }
        if self.wide {
            be_u64(self.table, chunk as usize * 8)
        } else {
            be_u32(self.table, chunk as usize * 4).map(u64::from)
        }
    }
}

/// `stsc` run-length chunk map (first_chunk is 1-based per spec).
#[derive(Debug, Clone, Copy)]
struct SampleToChunk<'a> {
    count: u32,
    table: &'a [u8],
}

impl<'a> SampleToChunk<'a> {
    const ENTRY: usize = 12;

    fn parse(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "stsc")?;
        let count = be_u32(d, 0).ok_or(DemuxError::Truncated("stsc"))?;
        let table = &d[4.min(d.len())..];
        if (table.len() as u64) < u64::from(count) * Self::ENTRY as u64 || count == 0 {
            return Err(DemuxError::BadSampleTable("stsc"));
        }
        Ok(SampleToChunk { count, table })
    }

    /// Samples-per-chunk for a 0-based chunk index.
    fn samples_per_chunk(&self, chunk: u32) -> u32 {
        let mut current = 1;
        for i in 0..self.count {
            let at = i as usize * Self::ENTRY;
            let first = be_u32(self.table, at).unwrap_or(u32::MAX);
            if first > chunk + 1 {
                break;
            }
            current = be_u32(self.table, at + 4).unwrap_or(1);
        }
        current.max(1)
    }
}

/// `stss` sync-sample numbers (1-based). Absent box = all keyframes.
#[derive(Debug, Clone, Copy)]
struct SyncSamples<'a> {
    count: u32,
    table: &'a [u8],
}

impl<'a> SyncSamples<'a> {
    fn parse(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "stss")?;
        let count = be_u32(d, 0).ok_or(DemuxError::Truncated("stss"))?;
        let table = &d[4.min(d.len())..];
        if (table.len() as u64) < u64::from(count) * 4 {
            return Err(DemuxError::BadSampleTable("stss"));
        }
        Ok(SyncSamples { count, table })
    }

    fn contains(&self, sample_number_1based: u32) -> bool {
        (0..self.count)
            .filter_map(|i| be_u32(self.table, i as usize * 4))
            .any(|n| n == sample_number_1based)
    }
}

/// `stts` decode-delta runs: (sample_count, sample_delta) pairs.
#[derive(Debug, Clone, Copy)]
struct TimeToSample<'a> {
    count: u32,
    table: &'a [u8],
}

impl<'a> TimeToSample<'a> {
    const ENTRY: usize = 8;

    fn parse(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "stts")?;
        let count = be_u32(d, 0).ok_or(DemuxError::Truncated("stts"))?;
        let table = &d[4.min(d.len())..];
        if (table.len() as u64) < u64::from(count) * Self::ENTRY as u64 {
            return Err(DemuxError::BadSampleTable("stts"));
        }
        Ok(TimeToSample { count, table })
    }

    fn run(&self, i: u32) -> Option<(u32, u32)> {
        if i >= self.count {
            return None;
        }
        let at = i as usize * Self::ENTRY;
        Some((be_u32(self.table, at)?, be_u32(self.table, at + 4)?))
    }
}

/// `ctts` composition offsets: (sample_count, offset) runs; offsets are
/// read as i32 (v1 semantics; v0 files never exceed i32 in practice).
#[derive(Debug, Clone, Copy)]
struct CompositionOffsets<'a> {
    count: u32,
    table: &'a [u8],
}

impl<'a> CompositionOffsets<'a> {
    const ENTRY: usize = 8;

    fn parse(payload: &'a [u8]) -> Result<Self, DemuxError> {
        let d = full_box(payload, "ctts")?;
        let count = be_u32(d, 0).ok_or(DemuxError::Truncated("ctts"))?;
        let table = &d[4.min(d.len())..];
        if (table.len() as u64) < u64::from(count) * Self::ENTRY as u64 {
            return Err(DemuxError::BadSampleTable("ctts"));
        }
        Ok(CompositionOffsets { count, table })
    }

    fn run(&self, i: u32) -> Option<(u32, i32)> {
        if i >= self.count {
            return None;
        }
        let at = i as usize * Self::ENTRY;
        Some((be_u32(self.table, at)?, be_u32(self.table, at + 4)? as i32))
    }
}

/// The full set of sample tables one track needs to be walked.
#[derive(Debug, Clone, Copy)]
struct SampleTables<'a> {
    sizes: SampleSizes<'a>,
    offsets: ChunkOffsets<'a>,
    chunk_map: SampleToChunk<'a>,
    sync: Option<SyncSamples<'a>>,
    stts: TimeToSample<'a>,
    ctts: Option<CompositionOffsets<'a>>,
}

impl<'a> SampleTables<'a> {
    fn parse(stbl: &'a [u8]) -> Result<Self, DemuxError> {
        let sizes = match find_path(stbl, &[*b"stsz"]) {
            Some(p) => SampleSizes::parse(p)?,
            None => SampleSizes::parse_compact(
                find_path(stbl, &[*b"stz2"]).ok_or(DemuxError::MissingBox("stsz/stz2"))?,
            )?,
        };
        let offsets = match find_path(stbl, &[*b"stco"]) {
            Some(p) => ChunkOffsets::parse(p, false, "stco")?,
            None => ChunkOffsets::parse(
                find_path(stbl, &[*b"co64"]).ok_or(DemuxError::MissingBox("stco/co64"))?,
                true,
                "co64",
            )?,
        };
        let chunk_map = SampleToChunk::parse(
            find_path(stbl, &[*b"stsc"]).ok_or(DemuxError::MissingBox("stsc"))?,
        )?;
        let sync = match find_path(stbl, &[*b"stss"]) {
            Some(p) => Some(SyncSamples::parse(p)?),
            None => None,
        };
        let stts = TimeToSample::parse(
            find_path(stbl, &[*b"stts"]).ok_or(DemuxError::MissingBox("stts"))?,
        )?;
        let ctts = match find_path(stbl, &[*b"ctts"]) {
            Some(p) => Some(CompositionOffsets::parse(p)?),
            None => None,
        };
        Ok(SampleTables {
            sizes,
            offsets,
            chunk_map,
            sync,
            stts,
            ctts,
        })
    }
}

// ---- track identity --------------------------------------------------------

/// Offsets inside an `mdhd` payload (version/flags word included), for
/// both box versions. ISO/IEC 14496-12 §8.4.2.
const MDHD_V0_LANGUAGE_AT: usize = 20;
const MDHD_V1_LANGUAGE_AT: usize = 32;
/// Offsets inside a `tkhd` payload (version/flags word included), for
/// both box versions. ISO/IEC 14496-12 §8.3.2.
const TKHD_V0_TRACK_ID_AT: usize = 12;
const TKHD_V1_TRACK_ID_AT: usize = 20;
const TKHD_V0_ALTERNATE_GROUP_AT: usize = 34;
const TKHD_V1_ALTERNATE_GROUP_AT: usize = 46;
/// `tkhd` flag bit 0: the track is enabled.
const TKHD_FLAG_ENABLED: u32 = 0x1;
/// Packed ISO-639-2/T: 3 x 5 bits, each biased by this, high bit padding.
const LANGUAGE_CHAR_BIAS: u8 = 0x60;
const LANGUAGE_CHAR_BITS: u16 = 5;
const LANGUAGE_CHAR_MASK: u16 = 0x1F;

/// A three-letter ISO-639-2/T language code, as carried by `mdhd`.
///
/// Kept as bytes rather than a `String` because it is fixed width, needs
/// no allocation, and is written straight back out by the muxer. An
/// absent or nonsensical code becomes [`Language::UNDETERMINED`] — the
/// value the standard itself defines for "unknown", so a caller never
/// has to distinguish "missing" from "explicitly unknown".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language(pub [u8; 3]);

impl Language {
    /// ISO-639-2 `und`: the standard's own "not stated" value.
    pub const UNDETERMINED: Language = Language(*b"und");

    /// Unpack the 16-bit `mdhd` encoding.
    ///
    /// Negative trust: the bit layout can express bytes that are not
    /// letters at all, and real files do contain zeroed language fields.
    /// Anything outside `a`–`z` yields [`Language::UNDETERMINED`] rather
    /// than a code made of control characters.
    fn from_packed(packed: u16) -> Language {
        let mut out = [0u8; 3];
        for (i, slot) in out.iter_mut().enumerate() {
            let shift = LANGUAGE_CHAR_BITS * (2 - i as u16);
            let c = ((packed >> shift) & LANGUAGE_CHAR_MASK) as u8 + LANGUAGE_CHAR_BIAS;
            if !c.is_ascii_lowercase() {
                return Language::UNDETERMINED;
            }
            *slot = c;
        }
        Language(out)
    }

    /// The code as text; always three ASCII letters.
    pub fn as_str(&self) -> &str {
        // SAFETY-free: the constructor only ever admits ASCII lowercase.
        std::str::from_utf8(&self.0).unwrap_or("und")
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Who a track *is*, as opposed to what it contains.
///
/// A twelve-hour source with six audio tracks is unusable if they come
/// out the far end anonymous: the whole point of six tracks is that a
/// viewer picks one. So identity travels with the track and is written
/// back out verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackMeta<'a> {
    /// `tkhd` track identifier as it appeared in the source.
    pub track_id: u32,
    /// Language from `mdhd`.
    pub language: Language,
    /// `tkhd` enabled flag — a disabled track is present but not played.
    pub enabled: bool,
    /// `tkhd` alternate group: tracks sharing a non-zero group are
    /// alternatives to each other (the six-language case), and players
    /// use it to offer a choice rather than play them all at once.
    pub alternate_group: u16,
    /// Human-readable title from `udta/name`, when the source carried one.
    pub name: Option<&'a str>,
}

impl Default for TrackMeta<'_> {
    fn default() -> Self {
        TrackMeta {
            track_id: 0,
            language: Language::UNDETERMINED,
            enabled: true,
            alternate_group: 0,
            name: None,
        }
    }
}

/// Read identity out of one `trak`. Never fails: a track whose headers
/// are unreadable still has usable media, so identity degrades to
/// defaults instead of failing the whole file (graceful degradation).
fn parse_track_meta<'a>(trak: &'a [u8], mdia: &'a [u8]) -> TrackMeta<'a> {
    let mut meta = TrackMeta::default();
    if let Some(mdhd) = find_path(mdia, &[*b"mdhd"]) {
        let version = mdhd.first().copied().unwrap_or(0);
        let at = if version == 1 {
            MDHD_V1_LANGUAGE_AT
        } else {
            MDHD_V0_LANGUAGE_AT
        };
        if let Some(packed) = be_u16(mdhd, at) {
            meta.language = Language::from_packed(packed);
        }
    }
    if let Some(tkhd) = find_path(trak, &[*b"tkhd"]) {
        let version = tkhd.first().copied().unwrap_or(0);
        let (id_at, group_at) = if version == 1 {
            (TKHD_V1_TRACK_ID_AT, TKHD_V1_ALTERNATE_GROUP_AT)
        } else {
            (TKHD_V0_TRACK_ID_AT, TKHD_V0_ALTERNATE_GROUP_AT)
        };
        meta.track_id = be_u32(tkhd, id_at).unwrap_or(0);
        meta.alternate_group = be_u16(tkhd, group_at).unwrap_or(0);
        // Flags are the low 24 bits of the full-box word.
        let flags = be_u32(tkhd, 0).unwrap_or(TKHD_FLAG_ENABLED) & 0x00FF_FFFF;
        meta.enabled = flags & TKHD_FLAG_ENABLED != 0;
    }
    meta.name = find_path(trak, &[*b"udta", *b"name"]).and_then(parse_track_name);
    meta
}

/// A `udta/name` payload is a bare string. Trailing NULs are common and
/// non-UTF-8 bytes are possible; both yield `None` rather than a title
/// made of replacement characters.
fn parse_track_name(payload: &[u8]) -> Option<&str> {
    let trimmed = payload.split(|b| *b == 0).next().unwrap_or(payload);
    let text = std::str::from_utf8(trimmed).ok()?.trim();
    (!text.is_empty()).then_some(text)
}

/// Parse `mdhd`: (timescale, duration in timescale units).
fn parse_mdhd(payload: &[u8]) -> Result<(u32, u64), DemuxError> {
    let version = *payload.first().ok_or(DemuxError::Truncated("mdhd"))?;
    if version == 1 {
        let ts = be_u32(payload, 20).ok_or(DemuxError::Truncated("mdhd"))?;
        let dur = be_u64(payload, 24).ok_or(DemuxError::Truncated("mdhd"))?;
        Ok((ts, dur))
    } else {
        let ts = be_u32(payload, 12).ok_or(DemuxError::Truncated("mdhd"))?;
        let dur = be_u32(payload, 16).ok_or(DemuxError::Truncated("mdhd"))?;
        Ok((ts, u64::from(dur)))
    }
}

// ---- tracks -----------------------------------------------------------------

/// A demuxable video track: codec facts, timing, and wired sample tables.
#[derive(Debug, Clone, Copy)]
pub struct VideoTrack<'a> {
    /// Track codec.
    pub codec: VideoCodec,
    /// Coded width in pixels.
    pub width: u16,
    /// Coded height in pixels.
    pub height: u16,
    /// Bytes per NAL length prefix (H.264/HEVC; 0 for AV1 OBU samples).
    pub nal_length_size: u8,
    /// `mdhd` timescale: DTS/PTS/duration units per second.
    pub timescale: u32,
    /// `mdhd` duration in timescale units (0 = unknown).
    pub media_duration: u64,
    /// What the source said its colours mean, when it said anything.
    ///
    /// `None` covers both "no `colr` box" and "a `colr` carrying an ICC
    /// profile", which this does not model. Either way nothing is
    /// claimed that the file did not state.
    pub colour: Option<crate::colour::Colour>,
    /// The raw `avcC`/`hvcC`/`av1C` payload.
    config: &'a [u8],
    tables: SampleTables<'a>,
}

/// A demuxable audio track (`mp4a`), packets passed through bit-exact.
#[derive(Debug, Clone, Copy)]
pub struct AudioTrack<'a> {
    /// `mdhd` timescale (usually the sample rate).
    pub timescale: u32,
    /// `mdhd` duration in timescale units.
    pub media_duration: u64,
    /// Channel count from the sample entry.
    pub channels: u16,
    /// Sample rate from the sample entry (16.16 fixed → integer part).
    pub sample_rate: u32,
    /// The complete `mp4a` sample-entry payload (including the `esds`
    /// decoder config child) — re-emitted verbatim by the muxer so the
    /// decoder configuration survives bit-exactly.
    pub entry_payload: &'a [u8],
    /// Language, title and grouping — what makes one of six tracks
    /// choosable rather than anonymous.
    pub meta: TrackMeta<'a>,
    tables: SampleTables<'a>,
}

/// Locate the first H.264/HEVC/AV1 video track and validate its tables.
///
/// # Errors
///
/// [`DemuxError::MissingBox`] when moov/track plumbing is absent; other
/// variants when a table or config record is malformed.
pub fn find_video_track(file: &[u8]) -> Result<VideoTrack<'_>, DemuxError> {
    let moov = find_path(file, &[*b"moov"]).ok_or(DemuxError::MissingBox("moov"))?;
    for trak in boxes(moov) {
        let trak = trak?;
        if trak.kind != *b"trak" {
            continue;
        }
        let Some(mdia) = find_path(trak.payload, &[*b"mdia"]) else {
            continue;
        };
        let Some(stbl) = find_path(mdia, &[*b"minf", *b"stbl"]) else {
            continue;
        };
        let (timescale, media_duration) =
            parse_mdhd(find_path(mdia, &[*b"mdhd"]).ok_or(DemuxError::MissingBox("mdhd"))?)?;
        if let Some(track) = try_video_track(stbl, timescale, media_duration)? {
            return Ok(track);
        }
    }
    Err(DemuxError::MissingBox("trak with avc1/hvc1/hev1/av01"))
}

/// `hdlr`: offset of the handler_type fourcc inside the full-box payload
/// (pre_defined word first).
const HDLR_TYPE_AT: usize = 4;
/// Handler type marking a media track as audio.
const HANDLER_AUDIO: [u8; 4] = *b"soun";
/// Handler types marking a media track as subtitles or timed text.
///
/// Three of them because three eras of the format disagreed: `subt` is
/// the ISOBMFF subtitle handler, `sbtl` what QuickTime and most modern
/// muxers emit, and `text` the older timed-text track. A file may use
/// any of them, so all three have to count.
const HANDLER_SUBTITLE: [[u8; 4]; 3] = [*b"subt", *b"sbtl", *b"text"];

/// Ceiling on `trak` boxes examined while surveying a file.
///
/// Bounded execution again: the survey walks every track, not just the
/// audio ones, so it needs its own limit rather than borrowing the audio
/// ceiling. Generous enough that no real file reaches it.
const MAX_SURVEYED_TRACKS: usize = 256;

/// What a source carries besides its video and audio — the material the
/// user asked to keep and the engine cannot yet re-emit into MP4.
///
/// This type exists to make a silent loss loud. Subtitles, chapters and
/// tags are dropped by the MP4 output path today; saying so is the
/// difference between a known limitation and a user discovering months
/// later that their chapter marks are gone. Same contract the
/// unsupported-audio path already honours.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Extras {
    /// Subtitle or timed-text tracks present in the source.
    pub subtitles: usize,
    /// Whether the source carries chapter marks, by either convention:
    /// a Nero `udta/chpl` list or a QuickTime chapter track referenced
    /// through `tref/chap`.
    pub chapters: bool,
    /// Whether the source carries a metadata island (`udta/meta`) —
    /// tags, cover art, and the rest of the descriptive payload.
    pub tags: bool,
}

impl Extras {
    /// Whether there is anything here at all.
    pub fn is_empty(&self) -> bool {
        self.subtitles == 0 && !self.chapters && !self.tags
    }

    /// Human phrase naming what will not survive, or `None` when
    /// nothing would be lost.
    ///
    /// # Example
    ///
    /// ```
    /// use sovmr_mp4::demux::Extras;
    /// let e = Extras { subtitles: 2, chapters: true, tags: false };
    /// assert_eq!(
    ///     e.lost_description().as_deref(),
    ///     Some("2 subtitle track(s), chapters")
    /// );
    /// assert!(Extras::default().lost_description().is_none());
    /// ```
    pub fn lost_description(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut parts = Vec::new();
        if self.subtitles > 0 {
            parts.push(format!("{} subtitle track(s)", self.subtitles));
        }
        if self.chapters {
            parts.push("chapters".to_string());
        }
        if self.tags {
            parts.push("tags/cover art".to_string());
        }
        Some(parts.join(", "))
    }
}

/// Survey everything the source carries beyond video and audio.
///
/// Deliberately total: a malformed or unreadable region yields "not
/// present" rather than failing the job. This answer only drives a
/// warning, and refusing to encode a twelve-hour master because its
/// `udta` was odd would be the tail wagging the dog.
///
/// # Errors
///
/// [`DemuxError::MissingBox`] when the file has no `moov` at all.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), sovmr_mp4::demux::DemuxError> {
/// # let bytes = std::fs::read("/dev/null").unwrap_or_default();
/// if let Ok(extras) = sovmr_mp4::demux::survey_extras(&bytes) {
///     if let Some(lost) = extras.lost_description() {
///         eprintln!("will not survive: {lost}");
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub fn survey_extras(file: &[u8]) -> Result<Extras, DemuxError> {
    let moov = find_path(file, &[*b"moov"]).ok_or(DemuxError::MissingBox("moov"))?;
    let mut extras = Extras {
        tags: has_meaningful_tags(moov),
        chapters: find_path(moov, &[*b"udta", *b"chpl"]).is_some(),
        ..Extras::default()
    };
    // A QuickTime chapter track *is* a text track — the only thing
    // distinguishing it from a subtitle track is that some other track
    // points at it through `tref/chap`. Collect those referents first,
    // or the chapter track gets counted a second time as a subtitle and
    // the notice claims two where the user can see one.
    let mut chapter_tracks: Vec<u32> = Vec::new();
    for trak in boxes(moov).take(MAX_SURVEYED_TRACKS) {
        let Ok(trak) = trak else { break };
        if trak.kind != *b"trak" {
            continue;
        }
        let Some(chap) = find_path(trak.payload, &[*b"tref", *b"chap"]) else {
            continue;
        };
        extras.chapters = true;
        // The payload is a bare list of referenced track IDs.
        for id in chap.chunks_exact(4).take(MAX_SURVEYED_TRACKS) {
            chapter_tracks.push(u32::from_be_bytes([id[0], id[1], id[2], id[3]]));
        }
    }

    for trak in boxes(moov).take(MAX_SURVEYED_TRACKS) {
        let Ok(trak) = trak else { break };
        if trak.kind != *b"trak" {
            continue;
        }
        let Some(mdia) = find_path(trak.payload, &[*b"mdia"]) else {
            continue;
        };
        if !is_subtitle_handler(mdia) {
            continue;
        }
        if chapter_tracks.contains(&track_id_of(trak.payload)) {
            continue; // this text track carries the chapter titles
        }
        extras.subtitles += 1;
    }
    Ok(extras)
}

/// iTunes-style tag naming the encoding tool (`©too`).
///
/// Practically every muxer writes it — a plain `ffmpeg` output carries
/// this and nothing else — and sovmr replaces it with its own on the way
/// out. It is therefore not user data, and counting it would make the
/// "tags will not survive" notice fire on every ordinary file. A warning
/// that fires always is a warning nobody reads, which is how a real one
/// gets missed later.
const TAG_ENCODER: [u8; 4] = [0xA9, b't', b'o', b'o'];

/// Whether `udta/meta/ilst` holds anything a user would actually miss.
fn has_meaningful_tags(moov: &[u8]) -> bool {
    let Some(meta) = find_path(moov, &[*b"udta", *b"meta"]) else {
        return false;
    };
    // `meta` is a FullBox, so its children begin after the version and
    // flags word. A few writers emit it as a plain box instead, so try
    // the standard layout first and fall back rather than declaring a
    // file we can plainly read to be malformed.
    let ilst = meta
        .get(4..)
        .and_then(|body| find_path(body, &[*b"ilst"]))
        .or_else(|| find_path(meta, &[*b"ilst"]));
    let Some(ilst) = ilst else {
        return false;
    };
    boxes(ilst)
        .take(MAX_SURVEYED_TRACKS)
        .filter_map(|b| b.ok())
        .any(|b| b.kind != TAG_ENCODER)
}

/// The `tkhd` track ID of a `trak` payload, or 0 when unreadable.
fn track_id_of(trak: &[u8]) -> u32 {
    find_path(trak, &[*b"tkhd"])
        .and_then(|tkhd| {
            let at = if tkhd.first().copied().unwrap_or(0) == 1 {
                TKHD_V1_TRACK_ID_AT
            } else {
                TKHD_V0_TRACK_ID_AT
            };
            be_u32(tkhd, at)
        })
        .unwrap_or(0)
}

/// Whether this `mdia` belongs to a subtitle or timed-text track.
fn is_subtitle_handler(mdia: &[u8]) -> bool {
    find_path(mdia, &[*b"hdlr"])
        .and_then(|h| full_box(h, "hdlr").ok())
        .and_then(|h| h.get(HDLR_TYPE_AT..HDLR_TYPE_AT + 4))
        .map(|t| HANDLER_SUBTITLE.iter().any(|k| k == t))
        .unwrap_or(false)
}

/// Hard ceiling on audio tracks taken from one file.
///
/// Bounded execution: every loop over externally supplied
/// data needs a statically provable limit, and a crafted `moov` can
/// declare arbitrarily many `trak` boxes. Thirty-two is far above any
/// real multi-language master — six to twelve is the usual range — while
/// keeping the work a hostile file can demand finite.
pub const MAX_AUDIO_TRACKS: usize = 32;

/// Every audio track in a container, plus what had to be left behind.
///
/// The engine's contract is that six tracks in means six tracks out, so
/// "which ones could not come along" is as much a part of the answer as
/// the tracks themselves — it is what the user-facing warning is built
/// from.
#[derive(Debug, Clone)]
pub struct AudioTracks<'a> {
    /// Tracks that can be carried through bit-exactly, in file order.
    pub tracks: Vec<AudioTrack<'a>>,
    /// Sample-entry fourccs of audio tracks that cannot be carried, one
    /// entry per rejected track, in file order.
    pub unsupported: Vec<[u8; 4]>,
    /// What those rejected tracks weigh, in bits per second.
    ///
    /// Kept because the heuristic engine has to weigh **the file**, not
    /// the part of it this project happens to be able to copy. A PCM
    /// track in an MP4 cannot be carried, and it is also the heaviest
    /// thing in the container — reporting it as nothing understated
    /// exactly the files whose sound weighs most.
    pub unsupported_bitrate: u64,
    /// Set when the file declared more audio tracks than
    /// [`MAX_AUDIO_TRACKS`] and the scan stopped early. Silence here
    /// would be a silent data loss, which is the one thing this whole
    /// milestone exists to prevent.
    pub truncated: bool,
}

impl AudioTracks<'_> {
    /// Whether the container had no audio at all — distinct from having
    /// audio that could not be carried.
    pub fn is_silent(&self) -> bool {
        self.tracks.is_empty() && self.unsupported.is_empty()
    }

    /// Total bits per second across every carried track.
    pub fn carried_bitrate(&self) -> u64 {
        self.tracks.iter().map(AudioTrack::bitrate).sum()
    }

    /// Total bits per second of **all** the file's audio, carried or
    /// not.
    ///
    /// This is what the heuristic engine must weigh: with six
    /// passthrough tracks the audio can outweigh the compressed video,
    /// and a track this project cannot copy still occupies the file the
    /// user is looking at. Counting only what we can carry reported
    /// nought per cent for PCM in an MP4 — the very case where the sound
    /// weighs most.
    pub fn total_bitrate(&self) -> u64 {
        self.carried_bitrate() + self.unsupported_bitrate
    }
}

/// Locate **every** supported (`mp4a`) audio track, with identity intact.
///
/// The engine carries every one of them. Its predecessor returned only
/// the first, which silently discarded five of six tracks on a
/// multi-language source — the defect this milestone exists to end.
///
/// # Errors
///
/// Same classes as [`find_video_track`] for malformed audio tables.
pub fn find_audio_tracks(file: &[u8]) -> Result<AudioTracks<'_>, DemuxError> {
    let moov = find_path(file, &[*b"moov"]).ok_or(DemuxError::MissingBox("moov"))?;
    let mut found = AudioTracks {
        tracks: Vec::new(),
        unsupported: Vec::new(),
        unsupported_bitrate: 0,
        truncated: false,
    };
    for trak in boxes(moov) {
        let trak = trak?;
        if trak.kind != *b"trak" {
            continue;
        }
        let Some(mdia) = find_path(trak.payload, &[*b"mdia"]) else {
            continue;
        };
        if !is_audio_handler(mdia) {
            continue;
        }
        if found.tracks.len() + found.unsupported.len() >= MAX_AUDIO_TRACKS {
            found.truncated = true;
            break;
        }
        let Some(stbl) = find_path(mdia, &[*b"minf", *b"stbl"]) else {
            continue;
        };
        match audio_entry(stbl)? {
            Some(entry) => {
                let (timescale, media_duration) = parse_mdhd(
                    find_path(mdia, &[*b"mdhd"]).ok_or(DemuxError::MissingBox("mdhd"))?,
                )?;
                let channels = be_u16(entry, ENTRY_CHANNELS_AT)
                    .ok_or(DemuxError::Truncated("AudioSampleEntry"))?;
                let sample_rate = be_u32(entry, ENTRY_SAMPLE_RATE_AT)
                    .ok_or(DemuxError::Truncated("AudioSampleEntry"))?
                    >> 16;
                found.tracks.push(AudioTrack {
                    timescale,
                    media_duration,
                    channels,
                    sample_rate,
                    entry_payload: entry,
                    meta: parse_track_meta(trak.payload, mdia),
                    tables: SampleTables::parse(stbl)?,
                });
            }
            None => {
                // An audio track whose codec we cannot carry: record the
                // fourcc so the warning can name it.
                if let Some(kind) = first_sample_entry_kind(stbl)? {
                    found.unsupported.push(kind);
                    // What it weighs, so the engine can account for a
                    // track it cannot copy. The sample table is the same
                    // shape whatever the codec, so this costs nothing
                    // beyond reading it.
                    let clock = find_path(mdia, &[*b"mdhd"]).map(parse_mdhd);
                    if let (Some(Ok((timescale, duration))), Ok(tables)) =
                        (clock, SampleTables::parse(stbl))
                    {
                        let seconds = if timescale > 0 {
                            duration as f64 / f64::from(timescale)
                        } else {
                            0.0
                        };
                        if seconds > 0.0 {
                            let sizes = tables.sizes;
                            let bytes = if sizes.uniform != 0 {
                                u64::from(sizes.uniform) * u64::from(sizes.count)
                            } else {
                                (0..sizes.count).map(|i| u64::from(sizes.get(i))).sum()
                            };
                            let bits = bytes as f64 * 8.0;
                            found.unsupported_bitrate += (bits / seconds) as u64;
                        }
                    }
                }
            }
        }
    }
    Ok(found)
}

/// Whether this `mdia` belongs to a sound track.
fn is_audio_handler(mdia: &[u8]) -> bool {
    find_path(mdia, &[*b"hdlr"])
        .and_then(|h| full_box(h, "hdlr").ok())
        .and_then(|h| h.get(HDLR_TYPE_AT..HDLR_TYPE_AT + 4))
        .map(|t| t == HANDLER_AUDIO)
        .unwrap_or(false)
}

/// The `mp4a` sample-entry payload of this `stbl`, if it has one.
fn audio_entry(stbl: &[u8]) -> Result<Option<&[u8]>, DemuxError> {
    for entry in sample_entries(stbl)? {
        let entry = entry?;
        if entry.kind == *b"mp4a" {
            if entry.payload.len() < AUDIO_ENTRY_LEN {
                return Err(DemuxError::Truncated("AudioSampleEntry"));
            }
            return Ok(Some(entry.payload));
        }
    }
    Ok(None)
}

/// The fourcc of this `stbl`'s first sample entry — how an unsupported
/// track names itself to the user.
fn first_sample_entry_kind(stbl: &[u8]) -> Result<Option<[u8; 4]>, DemuxError> {
    match sample_entries(stbl)?.next() {
        Some(entry) => Ok(Some(entry?.kind)),
        None => Ok(None),
    }
}

/// Iterate the sample entries inside a `stbl`'s `stsd`.
fn sample_entries(stbl: &[u8]) -> Result<crate::BoxIter<'_>, DemuxError> {
    let stsd = find_path(stbl, &[*b"stsd"]).ok_or(DemuxError::MissingBox("stsd"))?;
    let entries = full_box(stsd, "stsd")?
        .get(4..)
        .ok_or(DemuxError::Truncated("stsd"))?;
    Ok(boxes(entries))
}

/// Inspect one `stbl` for a video sample entry.
fn try_video_track(
    stbl: &[u8],
    timescale: u32,
    media_duration: u64,
) -> Result<Option<VideoTrack<'_>>, DemuxError> {
    let stsd = find_path(stbl, &[*b"stsd"]).ok_or(DemuxError::MissingBox("stsd"))?;
    let entries = full_box(stsd, "stsd")?
        .get(4..)
        .ok_or(DemuxError::Truncated("stsd"))?;
    for entry in boxes(entries) {
        let entry = entry?;
        let (codec, config_kind) = match &entry.kind {
            b"avc1" | b"avc3" => (VideoCodec::H264, *b"avcC"),
            b"hvc1" | b"hev1" => (VideoCodec::Hevc, *b"hvcC"),
            b"av01" => (VideoCodec::Av1, *b"av1C"),
            _ => continue,
        };
        let width = be_u16(entry.payload, ENTRY_WIDTH_AT)
            .ok_or(DemuxError::Truncated("VisualSampleEntry"))?;
        let height = be_u16(entry.payload, ENTRY_HEIGHT_AT)
            .ok_or(DemuxError::Truncated("VisualSampleEntry"))?;
        let children = entry
            .payload
            .get(VISUAL_ENTRY_LEN..)
            .ok_or(DemuxError::Truncated("VisualSampleEntry"))?;
        let config =
            find_path(children, &[config_kind]).ok_or(DemuxError::MissingBox("avcC/hvcC/av1C"))?;
        let nal_length_size = match codec {
            VideoCodec::Av1 => 0,
            _ => nal_length_size(codec, config)?,
        };
        // A sibling of the codec configuration, not a child of it.
        let colour = find_path(children, &[*b"colr"]).and_then(crate::colour::Colour::from_colr);
        return Ok(Some(VideoTrack {
            codec,
            width,
            height,
            colour,
            nal_length_size,
            timescale,
            media_duration,
            config,
            tables: SampleTables::parse(stbl)?,
        }));
    }
    Ok(None)
}

/// Extract lengthSizeMinusOne from the H.264/HEVC config record.
/// NAL length-prefix width declared by an `avcC` or `hvcC` record.
///
/// Public because Matroska embeds these very records verbatim in
/// `CodecPrivate`: they are ISOBMFF structures wherever they are stored,
/// so the ISOBMFF reader is the right place to read them from, and
/// `sovmr-mkv` borrows it rather than growing a second copy.
///
/// # Errors
///
/// [`DemuxError::BadConfigRecord`] when the record is truncated or does
/// not declare `configurationVersion == 1`.
pub fn nal_length_size(codec: VideoCodec, config: &[u8]) -> Result<u8, DemuxError> {
    let byte = match codec {
        VideoCodec::H264 => *config
            .get(4)
            .ok_or(DemuxError::BadConfigRecord("avcC too short"))?,
        VideoCodec::Hevc => *config
            .get(HVCC_LENGTH_SIZE_AT)
            .ok_or(DemuxError::BadConfigRecord("hvcC too short"))?,
        VideoCodec::Av1 => return Ok(0),
    };
    if config.first() != Some(&1) {
        return Err(DemuxError::BadConfigRecord("configurationVersion != 1"));
    }
    Ok((byte & 0x3) + 1)
}

/// Iterate the parameter sets inside an `avcC` or `hvcC` record.
///
/// Public for the same reason as [`nal_length_size`]: Matroska carries
/// these records verbatim in `CodecPrivate`, and a keyframe there needs
/// its SPS/PPS prepended exactly as an MP4 sync sample does.
pub fn config_parameter_sets(codec: VideoCodec, config: &[u8]) -> ParameterSets<'_> {
    ParameterSets::new(codec, config)
}

/// Rewrite one length-prefixed H.264/HEVC frame as Annex B, appending
/// to `out`.
///
/// Length prefixes become four-byte start codes. `out` is appended to,
/// not cleared, so a caller can prepend parameter sets first.
///
/// Public for the same reason as [`nal_length_size`]: Matroska stores
/// its H.264 and HEVC frames in exactly this length-prefixed layout,
/// because it stores the `avcC`/`hvcC` that defines it.
///
/// # Errors
///
/// [`DemuxError::BadNalLength`] when a prefix describes more bytes than
/// the frame holds, or claims a zero-length NAL — either means the
/// frame is not what it says, and a decoder fed the remainder would
/// produce silent garbage.
pub fn annex_b_from_length_prefixed(
    data: &[u8],
    nal_length_size: u8,
    sample_index: u32,
    out: &mut Vec<u8>,
) -> Result<(), DemuxError> {
    let prefix = usize::from(nal_length_size);
    let mut pos = 0usize;
    while pos < data.len() {
        let mut len = 0usize;
        let Some(prefix_bytes) = data.get(pos..pos + prefix) else {
            return Err(DemuxError::BadNalLength {
                sample: sample_index,
            });
        };
        for b in prefix_bytes {
            len = (len << 8) | usize::from(*b);
        }
        pos += prefix;
        let Some(nal) = data.get(pos..pos + len) else {
            return Err(DemuxError::BadNalLength {
                sample: sample_index,
            });
        };
        if len == 0 {
            return Err(DemuxError::BadNalLength {
                sample: sample_index,
            });
        }
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(nal);
        pos += len;
    }
    Ok(())
}

impl<'a> VideoTrack<'a> {
    /// Number of samples (frames) in the track.
    pub fn sample_count(&self) -> u32 {
        self.tables.sizes.count
    }

    /// Total coded bytes of the video track — the sum of its `stsz`
    /// sizes, exactly.
    ///
    /// The audio side has had this since long-form: what the heuristic
    /// engine weighs the video leg against. The video leg itself had
    /// nothing, so the engine used the **whole file length** as "the
    /// source bitrate" — audio, subtitles and container overhead
    /// included. A film with six lossless tracks therefore looked far
    /// denser than its pictures are, and every ceiling derived from that
    /// density was set for a video that does not exist.
    pub fn total_bytes(&self) -> u64 {
        let sizes = self.tables.sizes;
        if sizes.uniform != 0 {
            return u64::from(sizes.uniform) * u64::from(sizes.count);
        }
        (0..sizes.count).map(|i| u64::from(sizes.get(i))).sum()
    }

    /// Average video bitrate in bits per second, or zero when the track
    /// carries no usable duration.
    pub fn bitrate(&self) -> u64 {
        match self.duration() {
            Some(d) if d.as_secs_f64() > 0.0 => {
                (self.total_bytes() as f64 * 8.0 / d.as_secs_f64()) as u64
            }
            _ => 0,
        }
    }

    /// Frames per second from the whole track.
    ///
    /// Not from the first sample's duration, which is what the provider
    /// used: on variable-rate material the first frame is not the film,
    /// and a wrong frame rate propagates into every bits-per-pixel
    /// judgement the engine makes.
    ///
    /// Returns `None` when the track states no usable duration.
    pub fn frame_rate(&self) -> Option<f64> {
        let seconds = self.duration()?.as_secs_f64();
        if seconds <= 0.0 || self.sample_count() == 0 {
            return None;
        }
        Some(f64::from(self.sample_count()) / seconds)
    }

    /// Media duration as wall time, when `mdhd` carries one.
    pub fn duration(&self) -> Option<std::time::Duration> {
        duration_of(self.media_duration, self.timescale)
    }

    /// Iterate the track's parameter sets (H.264/HEVC only).
    pub fn parameter_sets(&self) -> ParameterSets<'a> {
        ParameterSets::new(self.codec, self.config)
    }

    /// Iterate samples in decode order with full timing.
    pub fn samples(&self, file: &'a [u8]) -> Samples<'a> {
        Samples::new(self.tables, file)
    }

    /// Convert one H.264/HEVC sample to Annex B into `out` (cleared,
    /// capacity reused): keyframes get parameter sets prepended, NAL
    /// length prefixes become start codes.
    ///
    /// # Errors
    ///
    /// [`DemuxError::WrongCodec`] for AV1 tracks (their samples are
    /// already raw OBUs); [`DemuxError::BadNalLength`] on lying prefixes.
    pub fn write_frame(&self, sample: &Sample<'a>, out: &mut Vec<u8>) -> Result<(), DemuxError> {
        if self.codec == VideoCodec::Av1 {
            // AV1 samples are already the elementary form a decoder
            // wants: raw OBUs, each carrying its own size. There is no
            // start-code framing to add and no parameter sets to
            // prepend — a keyframe's sequence header is inside the
            // sample.
            out.clear();
            out.extend_from_slice(sample.data);
            return Ok(());
        }
        out.clear();
        if sample.keyframe {
            for ps in self.parameter_sets() {
                let ps = ps?;
                out.extend_from_slice(&START_CODE);
                out.extend_from_slice(ps);
            }
        }
        annex_b_from_length_prefixed(sample.data, self.nal_length_size, sample.index, out)
    }
}

impl<'a> AudioTrack<'a> {
    /// Number of audio packets in the track.
    pub fn sample_count(&self) -> u32 {
        self.tables.sizes.count
    }

    /// Number of `stco`/`co64` chunks the track's packets are grouped
    /// into — the observable form of interleaving granularity.
    ///
    /// A muxer that emits one chunk per packet still produces a valid
    /// file, so nothing *fails*; it just buries a twelve-hour, seven-track
    /// movie under millions of chunk-table entries. The number is only
    /// visible from the outside here, which is what lets a test hold the
    /// muxer to its slab size instead of trusting it.
    pub fn chunk_count(&self) -> u32 {
        self.tables.offsets.count
    }

    /// Total payload bytes of the track (sum of the `stsz` sizes) — the
    /// exact number of bytes bit-exact passthrough will copy into the
    /// output, and hence what the heuristic engine weighs against the
    /// video leg.
    pub fn total_bytes(&self) -> u64 {
        let sizes = self.tables.sizes;
        if sizes.uniform != 0 {
            return u64::from(sizes.uniform) * u64::from(sizes.count);
        }
        (0..sizes.count).map(|i| u64::from(sizes.get(i))).sum()
    }

    /// Average bitrate in bits/second (0 when the track carries no
    /// usable duration).
    pub fn bitrate(&self) -> u64 {
        match self.duration() {
            Some(d) if d.as_secs_f64() > 0.0 => {
                (self.total_bytes() as f64 * 8.0 / d.as_secs_f64()) as u64
            }
            _ => 0,
        }
    }

    /// Media duration as wall time, when `mdhd` carries one.
    pub fn duration(&self) -> Option<std::time::Duration> {
        duration_of(self.media_duration, self.timescale)
    }

    /// Iterate audio packets in decode order with full timing.
    pub fn samples(&self, file: &'a [u8]) -> Samples<'a> {
        Samples::new(self.tables, file)
    }
}

fn duration_of(units: u64, timescale: u32) -> Option<std::time::Duration> {
    if units == 0 || timescale == 0 {
        return None;
    }
    Some(std::time::Duration::from_secs_f64(
        units as f64 / f64::from(timescale),
    ))
}

/// One sample (video frame or audio packet) with exact timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample<'a> {
    /// 0-based decode-order index.
    pub index: u32,
    /// Sync sample (always `true` for audio).
    pub keyframe: bool,
    /// Raw bytes exactly as stored in `mdat` (zero-copy).
    pub data: &'a [u8],
    /// Decode timestamp in track timescale units (from `stts`).
    pub dts: u64,
    /// Presentation timestamp: DTS plus the `ctts` composition offset.
    pub pts: i64,
    /// Sample duration in track timescale units.
    pub duration: u32,
}

/// See [`VideoTrack::samples`] / [`AudioTrack::samples`].
#[derive(Debug)]
pub struct Samples<'a> {
    tables: SampleTables<'a>,
    file: &'a [u8],
    sample: u32,
    chunk: u32,
    in_chunk: u32,
    chunk_pos: u64,
    // stts cursor.
    stts_run: u32,
    stts_left: u32,
    stts_delta: u32,
    dts: u64,
    // ctts cursor.
    ctts_run: u32,
    ctts_left: u32,
    ctts_offset: i32,
}

impl<'a> Samples<'a> {
    fn new(tables: SampleTables<'a>, file: &'a [u8]) -> Self {
        Samples {
            tables,
            file,
            sample: 0,
            chunk: 0,
            in_chunk: 0,
            chunk_pos: 0,
            stts_run: 0,
            stts_left: 0,
            stts_delta: 0,
            dts: 0,
            ctts_run: 0,
            ctts_left: 0,
            ctts_offset: 0,
        }
    }

    /// Advance the stts cursor and return this sample's delta.
    fn next_delta(&mut self) -> u32 {
        while self.stts_left == 0 {
            match self.tables.stts.run(self.stts_run) {
                Some((count, delta)) => {
                    self.stts_run += 1;
                    self.stts_left = count;
                    self.stts_delta = delta;
                }
                None => return self.stts_delta, // table exhausted: reuse last
            }
        }
        self.stts_left -= 1;
        self.stts_delta
    }

    /// Advance the ctts cursor and return this sample's offset.
    fn next_ctts(&mut self) -> i32 {
        let Some(ctts) = self.tables.ctts else {
            return 0;
        };
        while self.ctts_left == 0 {
            match ctts.run(self.ctts_run) {
                Some((count, offset)) => {
                    self.ctts_run += 1;
                    self.ctts_left = count;
                    self.ctts_offset = offset;
                }
                None => return 0,
            }
        }
        self.ctts_left -= 1;
        self.ctts_offset
    }
}

impl<'a> Iterator for Samples<'a> {
    type Item = Result<Sample<'a>, DemuxError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.sample >= self.tables.sizes.count {
            return None;
        }
        let index = self.sample;
        let size = self.tables.sizes.get(index);
        let Some(chunk_base) = self.tables.offsets.get(self.chunk) else {
            return Some(Err(DemuxError::SampleOutOfFile { index }));
        };
        let start = chunk_base.checked_add(self.chunk_pos);
        let end = start.and_then(|s| s.checked_add(u64::from(size)));
        let (Some(start), Some(end)) = (start, end) else {
            return Some(Err(DemuxError::SampleOutOfFile { index }));
        };
        if end > self.file.len() as u64 {
            return Some(Err(DemuxError::SampleOutOfFile { index }));
        }
        let data = &self.file[start as usize..end as usize];
        let keyframe = match &self.tables.sync {
            Some(stss) => stss.contains(index + 1),
            None => true,
        };
        let duration = self.next_delta();
        let dts = self.dts;
        let pts = dts as i64 + i64::from(self.next_ctts());

        self.sample += 1;
        self.in_chunk += 1;
        self.chunk_pos += u64::from(size);
        self.dts += u64::from(duration);
        if self.in_chunk >= self.tables.chunk_map.samples_per_chunk(self.chunk) {
            self.chunk += 1;
            self.in_chunk = 0;
            self.chunk_pos = 0;
        }
        Some(Ok(Sample {
            index,
            keyframe,
            data,
            dts,
            pts,
            duration,
        }))
    }
}

// ---- parameter sets ---------------------------------------------------------

/// Iterator over SPS/PPS (H.264) or VPS/SPS/PPS array NALs (HEVC).
#[derive(Debug)]
pub struct ParameterSets<'a> {
    data: &'a [u8],
    pos: usize,
    state: PsState,
    poisoned: bool,
}

#[derive(Debug, Clone, Copy)]
enum PsState {
    Avc { remaining: u8, pps_pending: bool },
    Hevc { nals: u16, arrays: u8 },
    None,
}

impl<'a> ParameterSets<'a> {
    fn new(codec: VideoCodec, config: &'a [u8]) -> Self {
        let state = match codec {
            VideoCodec::H264 => PsState::Avc {
                remaining: config.get(5).map(|b| b & 0x1F).unwrap_or(0),
                pps_pending: true,
            },
            VideoCodec::Hevc => PsState::Hevc {
                nals: 0,
                arrays: config.get(HVCC_ARRAYS_AT - 1).copied().unwrap_or(0),
            },
            VideoCodec::Av1 => PsState::None,
        };
        let pos = match codec {
            VideoCodec::H264 => 6,
            VideoCodec::Hevc => HVCC_ARRAYS_AT,
            VideoCodec::Av1 => 0,
        };
        ParameterSets {
            data: config,
            pos,
            state,
            poisoned: false,
        }
    }

    fn take_nal(&mut self) -> Result<&'a [u8], DemuxError> {
        let len = be_u16(self.data, self.pos)
            .ok_or(DemuxError::BadConfigRecord("truncated NAL length"))?;
        self.pos += 2;
        let nal = self
            .data
            .get(self.pos..self.pos + usize::from(len))
            .ok_or(DemuxError::BadConfigRecord("truncated NAL body"))?;
        self.pos += usize::from(len);
        Ok(nal)
    }
}

impl<'a> Iterator for ParameterSets<'a> {
    type Item = Result<&'a [u8], DemuxError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.poisoned {
            return None;
        }
        let step = |s: &mut Self| -> Option<Result<&'a [u8], DemuxError>> {
            loop {
                match s.state {
                    PsState::None => return None,
                    PsState::Avc {
                        remaining: 0,
                        pps_pending: false,
                    } => return None,
                    PsState::Avc {
                        remaining: 0,
                        pps_pending: true,
                    } => {
                        let Some(&count) = s.data.get(s.pos) else {
                            return Some(Err(DemuxError::BadConfigRecord("missing PPS count")));
                        };
                        s.pos += 1;
                        s.state = PsState::Avc {
                            remaining: count,
                            pps_pending: false,
                        };
                    }
                    PsState::Avc {
                        remaining,
                        pps_pending,
                    } => {
                        s.state = PsState::Avc {
                            remaining: remaining - 1,
                            pps_pending,
                        };
                        return Some(s.take_nal());
                    }
                    PsState::Hevc { nals: 0, arrays: 0 } => return None,
                    PsState::Hevc { nals: 0, arrays } => {
                        if s.data.get(s.pos).is_none() {
                            return Some(Err(DemuxError::BadConfigRecord("truncated array")));
                        }
                        let Some(count) = be_u16(s.data, s.pos + 1) else {
                            return Some(Err(DemuxError::BadConfigRecord("truncated array")));
                        };
                        s.pos += 3;
                        s.state = PsState::Hevc {
                            nals: count,
                            arrays: arrays - 1,
                        };
                    }
                    PsState::Hevc { nals, arrays } => {
                        s.state = PsState::Hevc {
                            nals: nals - 1,
                            arrays,
                        };
                        return Some(s.take_nal());
                    }
                }
            }
        };
        let item = step(self);
        if matches!(item, Some(Err(_))) {
            self.poisoned = true;
        }
        item
    }
}
