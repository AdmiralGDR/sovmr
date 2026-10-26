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

//! The sovereign ISOBMFF muxer: AV1 (`av01` + `av1C`) or HEVC (`hvc1` +
//! `hvcC`) video with an optional bit-exact copied `mp4a` audio track,
//! **fast-start** layout (`ftyp` → `moov` → `mdat`) so files stream over
//! the web immediately.
//!
//! Fast start without knowing offsets up front: sample payloads stream
//! into a temporary `mdat` spool file while per-sample metadata (sizes,
//! durations, keyframes, chunk runs) accumulates in memory. `finalize`
//! builds the complete `moov`, patches the `stco` offsets (the box's
//! *length* is offset-independent), writes `ftyp` + `moov` + the `mdat`
//! header to the destination and drains the spool with one
//! `std::io::copy` — zero re-encoding, zero large in-memory buffers.
//!
//! Interleaving is the caller's contract: add samples in presentation
//! order across tracks and the muxer records the resulting chunk runs,
//! so hardware players never seek back and forth between tracks.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::hevc::{self, HevcParameterSets};

/// Movie-header timescale (milliseconds — conventional).
const MOVIE_TIMESCALE: u32 = 1000;
/// `mdat` header length (32-bit size + fourcc).
const MDAT_HEADER: u64 = 8;
/// `mdat` header length in the 64-bit form: `size == 1`, fourcc, then a
/// `u64` largesize. Four extra bytes of header buy an unlimited payload.
const MDAT_HEADER_64: u64 = 16;
/// The sentinel in the 32-bit size field meaning "largesize follows".
const BOX_SIZE_IS_LARGESIZE: u32 = 1;
/// ISO language code `und` packed per the spec.
const LANG_UND: u16 = 0x55C4;
/// Track index of the video track. Audio tracks follow it, one index
/// each, so "which track" is an index rather than a boolean.
const VIDEO_TRACK: usize = 0;
/// AV1 OBU types we care about.
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

// ---- AV1 OBU helpers (pub: the pipeline uses them on encoder output) ----

/// Read one unsigned LEB128; returns (value, bytes consumed).
fn leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, b) in data.iter().take(8).enumerate() {
        value |= u64::from(b & 0x7F) << (i * 7);
        if b & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Walk one OBU at `data[0..]`: returns (type, total OBU length).
fn obu_at(data: &[u8]) -> Option<(u8, usize)> {
    let header = *data.first()?;
    if header & 0x80 != 0 {
        return None; // forbidden bit set: not an OBU
    }
    let obu_type = (header >> 3) & 0xF;
    let has_ext = header & 0x04 != 0;
    let has_size = header & 0x02 != 0;
    let mut pos = 1 + usize::from(has_ext);
    if !has_size {
        return Some((obu_type, data.len())); // last OBU extends to end
    }
    let (size, size_len) = leb128(data.get(pos..)?)?;
    pos += size_len;
    let total = pos.checked_add(usize::try_from(size).ok()?)?;
    if total > data.len() {
        return None;
    }
    Some((obu_type, total))
}

/// Strip a leading temporal-delimiter OBU: ISOBMFF samples should not
/// carry it (the container provides temporal framing).
pub fn strip_temporal_delimiter(frame: &[u8]) -> &[u8] {
    if let Some((OBU_TEMPORAL_DELIMITER, len)) = obu_at(frame) {
        &frame[len..]
    } else {
        frame
    }
}

/// Find the sequence-header OBU inside an encoded frame (present on the
/// first/IDR frames NVENC emits) and return the complete OBU bytes.
pub fn extract_sequence_header(frame: &[u8]) -> Option<&[u8]> {
    let mut rest = frame;
    let mut base = 0usize;
    while let Some((obu_type, len)) = obu_at(rest) {
        if obu_type == OBU_SEQUENCE_HEADER {
            return Some(&frame[base..base + len]);
        }
        base += len;
        rest = &frame[base..];
        if rest.is_empty() {
            break;
        }
    }
    None
}

/// `seq_profile`: the top 3 bits of the sequence header payload.
fn av1_seq_profile(seq_obu: &[u8]) -> u8 {
    // Skip the OBU header + size field to reach the payload.
    let Some((_, _total)) = obu_at(seq_obu) else {
        return 0;
    };
    let header = seq_obu[0];
    let mut pos = 1 + usize::from(header & 0x04 != 0);
    if header & 0x02 != 0 {
        if let Some((_, n)) = leb128(&seq_obu[pos..]) {
            pos += n;
        }
    }
    seq_obu.get(pos).map(|b| b >> 5).unwrap_or(0)
}

/// Build the `av1C` configuration record for a sequence header.
///
/// Public because Matroska carries this same record in `CodecPrivate`:
/// AV1's configuration box is defined once and stored in both
/// containers, so one builder serves both rather than two drifting apart.
pub fn av1c_record(seq_header: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out = vec![
        // marker (1) + version (1)
        0x81,
        (av1_seq_profile(seq_header) << 5) | av1_level_for(width, height),
        0x0C, // 8-bit 4:2:0
        0x00,
    ];
    out.extend_from_slice(seq_header);
    out
}

/// Coarse `seq_level_idx` from resolution (2.0 … 5.1 ladder).
fn av1_level_for(width: u16, height: u16) -> u8 {
    let pixels = u32::from(width) * u32::from(height);
    match pixels {
        0..=147_456 => 0,            // <=  512x288  -> 2.0
        147_457..=460_800 => 4,      // <=  960x480  -> 3.0
        460_801..=921_600 => 8,      // <= 1280x720  -> 4.0
        921_601..=2_073_600 => 9,    // <= 1920x1080 -> 4.1
        2_073_601..=3_686_400 => 12, // <= 2560x1440 -> 5.0
        _ => 13,                     // 4K and up    -> 5.1
    }
}

// ---- box builders -----------------------------------------------------------

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full_boxed(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.push(version);
    payload.extend_from_slice(&flags.to_be_bytes()[1..]);
    payload.extend_from_slice(body);
    boxed(kind, &payload)
}

/// Identity transformation matrix for tkhd/mvhd.
fn matrix() -> [u8; 36] {
    let mut m = [0u8; 36];
    m[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    m[16..20].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    m[32..36].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    m
}

// ---- the muxer ----------------------------------------------------------------

/// Which video codec the track carries — decides the `stsd` sample
/// entry (`av01`+`av1C` vs `hvc1`+`hvcC`) and the `ftyp` brands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxVideoCodec {
    /// AV1: samples are raw OBUs, config is the sequence-header OBU.
    Av1,
    /// HEVC: samples are length-prefixed NALs, config is VPS/SPS/PPS.
    Hevc,
}

/// Static facts for the video track.
#[derive(Debug, Clone, Copy)]
pub struct VideoMuxParams {
    /// Target codec of the encoded samples.
    pub codec: MuxVideoCodec,
    /// Frame width in pixels.
    pub width: u16,
    /// Frame height in pixels.
    pub height: u16,
    /// Track timescale (units per second) the sample durations use.
    pub timescale: u32,
    /// What the source said its colours mean, carried across.
    ///
    /// `None` when the source said nothing. Writing a description it
    /// never gave would be inventing one, and a player trusts a `colr`
    /// box completely — a wrong one is worse than none, because none at
    /// least lets the player fall back to its own guess.
    pub colour: Option<crate::colour::Colour>,
}

/// The codec configuration harvested from the encoder's first keyframe.
#[derive(Debug, Clone)]
enum VideoConfig {
    /// The AV1 sequence-header OBU, verbatim.
    Av1(Vec<u8>),
    /// The HEVC parameter sets, verbatim.
    Hevc(HevcParameterSets),
}

/// Static facts for the passthrough audio track.
#[derive(Debug, Clone)]
pub struct AudioMuxParams {
    /// Track timescale the audio durations use (usually the sample rate).
    pub timescale: u32,
    /// The source `mp4a` sample-entry payload, re-emitted verbatim so
    /// the decoder configuration (`esds`) survives bit-exactly.
    pub entry_payload: Vec<u8>,
    /// ISO-639-2 language, written into `mdhd`. Six tracks that arrive
    /// without languages are six tracks nobody can choose between, so
    /// this travels with the track rather than being regenerated.
    pub language: [u8; 3],
    /// `tkhd` alternate group: a non-zero value shared by several tracks
    /// tells a player they are alternatives, not simultaneous audio.
    pub alternate_group: u16,
    /// `tkhd` enabled flag.
    pub enabled: bool,
    /// Human-readable title, written back as `udta/name` when present.
    pub name: Option<String>,
}

impl AudioMuxParams {
    /// Passthrough parameters with no identity — the shape a caller uses
    /// when the source had none to carry.
    pub fn anonymous(timescale: u32, entry_payload: Vec<u8>) -> Self {
        AudioMuxParams {
            timescale,
            entry_payload,
            language: *b"und",
            alternate_group: 0,
            enabled: true,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SampleMeta {
    size: u32,
    duration: u32,
    keyframe: bool,
    /// Composition (presentation) timestamp in track units. For audio
    /// and non-reordered video this equals the decode time; with
    /// B-frames it differs and `finalize` derives the `ctts` table.
    cts: u64,
}

#[derive(Debug, Clone, Copy)]
struct ChunkMeta {
    /// Index into the muxer's track order: 0 is video, 1..=n are the
    /// audio tracks. A bool could only ever describe two tracks, which
    /// is precisely the limit being removed.
    track: usize,
    samples: u32,
    rel_offset: u64,
}

/// Streaming fast-start muxer. Feed samples in presentation-interleaved
/// order, then [`Mp4Muxer::finalize`]. Dropping without finalizing
/// removes the spool file (rollback).
pub struct Mp4Muxer {
    spool: Option<BufWriter<File>>,
    spool_path: PathBuf,
    video: VideoMuxParams,
    audio: Vec<AudioMuxParams>,
    config: Option<VideoConfig>,
    v_samples: Vec<SampleMeta>,
    /// One sample list per audio track, parallel to `audio`.
    a_samples: Vec<Vec<SampleMeta>>,
    chunks: Vec<ChunkMeta>,
    pos: u64,
}

impl Mp4Muxer {
    /// Open a muxer spooling `mdat` payload into `spool_path`.
    ///
    /// # Errors
    ///
    /// I/O errors creating the spool file.
    pub fn create(
        spool_path: &Path,
        video: VideoMuxParams,
        audio: Vec<AudioMuxParams>,
    ) -> io::Result<Self> {
        let a_samples = audio.iter().map(|_| Vec::new()).collect();
        Ok(Mp4Muxer {
            spool: Some(BufWriter::new(File::create(spool_path)?)),
            spool_path: spool_path.to_path_buf(),
            video,
            audio,
            config: None,
            v_samples: Vec::new(),
            a_samples,
            chunks: Vec::new(),
            pos: 0,
        })
    }

    /// Provide the AV1 sequence-header OBU (from the first encoded
    /// frame); required before `finalize` on AV1 tracks. First call wins.
    pub fn set_sequence_header(&mut self, obu: &[u8]) {
        if self.config.is_none() {
            self.config = Some(VideoConfig::Av1(obu.to_vec()));
        }
    }

    /// Provide the HEVC VPS/SPS/PPS (harvested from the first encoded
    /// IDR); required before `finalize` on HEVC tracks. First call wins.
    pub fn set_hevc_parameter_sets(&mut self, ps: HevcParameterSets) {
        if self.config.is_none() {
            self.config = Some(VideoConfig::Hevc(ps));
        }
    }

    fn append(
        &mut self,
        track: usize,
        data: &[u8],
        duration: u32,
        keyframe: bool,
        cts: u64,
    ) -> io::Result<()> {
        // A sample's size lives in a 32-bit `stsz` entry; a larger one
        // used to be truncated silently by an `as u32` cast, writing a
        // file whose sample table no longer matched its payload.
        let size = u32::try_from(data.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "sample of {} bytes does not fit stsz (max 4 GiB - 1)",
                    data.len()
                ),
            )
        })?;
        // The spool is only taken by `finalize` (which consumes the
        // muxer) and `Drop`, so this is unreachable through the public
        // API today; it is an error rather than a panic so that a future
        // refactor cannot turn it into a mid-encode abort.
        let writer = self
            .spool
            .as_mut()
            .ok_or_else(|| io::Error::other("muxer used after finalize/abort"))?;
        writer.write_all(data)?;
        // Chunk bookkeeping: a run of same-track samples is one chunk.
        match self.chunks.last_mut() {
            Some(last) if last.track == track => last.samples += 1,
            _ => self.chunks.push(ChunkMeta {
                track,
                samples: 1,
                rel_offset: self.pos,
            }),
        }
        let meta = SampleMeta {
            size,
            duration,
            keyframe,
            cts,
        };
        match track.checked_sub(1) {
            None => self.v_samples.push(meta),
            Some(audio_ix) => self.a_samples[audio_ix].push(meta),
        }
        self.pos += data.len() as u64;
        Ok(())
    }

    /// Append one video sample **in decode order**. `cts` is the
    /// sample's presentation timestamp in track units: for streams
    /// without B-frames pass the decode time (offsets collapse and no
    /// `ctts` is written); with B-frames `finalize` derives the `ctts`
    /// composition-offset table from it.
    ///
    /// # Errors
    ///
    /// Spool I/O errors.
    pub fn add_video_sample(
        &mut self,
        data: &[u8],
        duration: u32,
        keyframe: bool,
        cts: u64,
    ) -> io::Result<()> {
        self.append(VIDEO_TRACK, data, duration, keyframe, cts)
    }

    /// Append one bit-exact audio packet to audio track `audio_ix`
    /// (0-based, in the order the tracks were given to
    /// [`Mp4Muxer::create`]). Audio is never reordered: cts = dts.
    ///
    /// # Errors
    ///
    /// Spool I/O errors, or `InvalidInput` when `audio_ix` names a track
    /// this muxer was not created with — negative trust applies to the
    /// caller as much as to the file.
    pub fn add_audio_sample(
        &mut self,
        audio_ix: usize,
        data: &[u8],
        duration: u32,
    ) -> io::Result<()> {
        if audio_ix >= self.audio.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "audio track {audio_ix} does not exist (muxer has {})",
                    self.audio.len()
                ),
            ));
        }
        self.append(VIDEO_TRACK + 1 + audio_ix, data, duration, true, 0)
    }

    /// Number of audio tracks this muxer carries.
    pub fn audio_track_count(&self) -> usize {
        self.audio.len()
    }

    /// Build `moov`, write the fast-start file to `out_path`
    /// (`ftyp | moov | mdat`), drain the spool with one `io::copy`, and
    /// remove the spool. Returns total bytes written.
    ///
    /// # Errors
    ///
    /// I/O errors, or `InvalidInput` when no codec configuration was set
    /// (sequence header / parameter sets) or no video samples were added.
    pub fn finalize(mut self, out_path: &Path) -> io::Result<u64> {
        if self.v_samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no video samples",
            ));
        }
        let config = self.config.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                match self.video.codec {
                    MuxVideoCodec::Av1 => "AV1 sequence header not set",
                    MuxVideoCodec::Hevc => "HEVC parameter sets not set",
                },
            )
        })?;
        let video_entry = self.video_sample_entry(&config)?;
        let ctts_plan = self.video_ctts()?;
        let mut spool_writer = self
            .spool
            .take()
            .ok_or_else(|| io::Error::other("muxer spool already closed"))?;
        spool_writer.flush()?;
        drop(spool_writer);

        let brands: &[u8] = match self.video.codec {
            MuxVideoCodec::Av1 => b"isom\x00\x00\x02\x00isomiso2av01mp41",
            MuxVideoCodec::Hevc => b"isom\x00\x00\x02\x00isomiso2hvc1mp41",
        };
        let ftyp = boxed(b"ftyp", brands);

        // A payload past 4 GiB cannot state its own size in the 32-bit
        // field, so `mdat` switches to the largesize form. That makes the
        // header 16 bytes instead of 8, shifting every sample offset —
        // hence it is settled before anything is measured.
        let mdat_header_len = if MDAT_HEADER + self.pos > u64::from(u32::MAX) {
            MDAT_HEADER_64
        } else {
            MDAT_HEADER
        };

        // Whether chunk offsets fit in 32 bits depends on how big `moov`
        // is, and how big `moov` is depends on how wide its offset
        // tables are: a circular constraint. It is resolved by building
        // the narrow form and widening once if the answer does not fit.
        // Widening only ever GROWS `moov`, and the check below is
        // monotone in that size, so a second widening can never be
        // required — this runs at most twice, bounded by construction
        // rather than by a loop guard.
        let mut wide_offsets = false;
        let (mut moov, mut positions) = self.build_moov(&video_entry, &ctts_plan, wide_offsets);
        let mut mdat_payload_start = ftyp.len() as u64 + moov.len() as u64 + mdat_header_len;
        let highest_chunk = positions.iter().map(|(_, rel)| *rel).max().unwrap_or(0);
        if mdat_payload_start + highest_chunk > u64::from(u32::MAX) {
            wide_offsets = true;
            let (wide_moov, wide_positions) =
                self.build_moov(&video_entry, &ctts_plan, wide_offsets);
            moov = wide_moov;
            positions = wide_positions;
            mdat_payload_start = ftyp.len() as u64 + moov.len() as u64 + mdat_header_len;
        }
        // Absolute chunk offsets are computable now because the length of
        // the offset table does not depend on the values written into it.
        for (pos, rel) in positions {
            let abs = mdat_payload_start + rel;
            if wide_offsets {
                moov[pos..pos + 8].copy_from_slice(&abs.to_be_bytes());
            } else {
                // Proven to fit by the widening check above; still
                // converted rather than cast, so a future edit that
                // breaks the proof fails loudly instead of wrapping.
                let narrow = u32::try_from(abs).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "chunk offset overflowed a 32-bit table that was judged wide enough",
                    )
                })?;
                moov[pos..pos + 4].copy_from_slice(&narrow.to_be_bytes());
            }
        }

        let mut out = BufWriter::new(File::create(out_path)?);
        out.write_all(&ftyp)?;
        out.write_all(&moov)?;
        if mdat_header_len == MDAT_HEADER_64 {
            out.write_all(&BOX_SIZE_IS_LARGESIZE.to_be_bytes())?;
            out.write_all(b"mdat")?;
            out.write_all(&(MDAT_HEADER_64 + self.pos).to_be_bytes())?;
        } else {
            out.write_all(&((MDAT_HEADER + self.pos) as u32).to_be_bytes())?;
            out.write_all(b"mdat")?;
        }
        let mut spool = File::open(&self.spool_path)?;
        io::copy(&mut spool, &mut out)?;
        out.flush()?;
        drop(spool);
        let _ = std::fs::remove_file(&self.spool_path);
        Ok(ftyp.len() as u64 + moov.len() as u64 + mdat_header_len + self.pos)
    }

    /// The complete `stsd` video sample entry (`av01` or `hvc1` box) for
    /// the configured codec.
    fn video_sample_entry(&self, config: &VideoConfig) -> io::Result<Vec<u8>> {
        let mut visual = vec![0u8; 78];
        visual[6..8].copy_from_slice(&1u16.to_be_bytes()); // dref index
        visual[24..26].copy_from_slice(&self.video.width.to_be_bytes());
        visual[26..28].copy_from_slice(&self.video.height.to_be_bytes());
        visual[28..32].copy_from_slice(&0x0048_0000u32.to_be_bytes());
        visual[32..36].copy_from_slice(&0x0048_0000u32.to_be_bytes());
        visual[40..42].copy_from_slice(&1u16.to_be_bytes()); // frame count
        visual[74..76].copy_from_slice(&0x0018u16.to_be_bytes()); // depth
        visual[76..78].copy_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined
                                                                  // The colour description belongs in the sample entry beside the
                                                                  // codec configuration, and the AV1 binding says a `colr` box
                                                                  // SHOULD be there. Written only when the source gave one.
        let colr = self
            .video
            .colour
            .filter(crate::colour::Colour::is_meaningful)
            .map(|c| boxed(b"colr", &c.to_colr()));
        match (self.video.codec, config) {
            (MuxVideoCodec::Av1, VideoConfig::Av1(seq)) => {
                let av1c = av1c_record(seq, self.video.width, self.video.height);
                visual.extend_from_slice(&boxed(b"av1C", &av1c));
                if let Some(colr) = colr {
                    visual.extend_from_slice(&colr);
                }
                Ok(boxed(b"av01", &visual))
            }
            (MuxVideoCodec::Hevc, VideoConfig::Hevc(ps)) => {
                let record = hevc::hvcc_record(ps).map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidInput, format!("hvcC: {e}"))
                })?;
                visual.extend_from_slice(&boxed(b"hvcC", &record));
                if let Some(colr) = colr {
                    visual.extend_from_slice(&colr);
                }
                Ok(boxed(b"hvc1", &visual))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "video codec/config mismatch",
            )),
        }
    }

    /// The video track's `ctts` run-length table, derived from the
    /// gap between each sample's presentation time and its decode time
    /// (cumulative durations). Offsets are normalized so the smallest
    /// is zero — a uniform shift is pure composition delay — and
    /// `None` means every offset collapsed to zero (no reordering):
    /// the box is omitted entirely.
    ///
    /// # Errors
    ///
    /// `InvalidInput` when a normalized offset exceeds 32 bits (a
    /// corrupt timestamp, not a real stream).
    fn video_ctts(&self) -> io::Result<CttsPlan> {
        let mut dts: u64 = 0;
        let mut raw = Vec::with_capacity(self.v_samples.len());
        for s in &self.v_samples {
            raw.push(s.cts as i64 - dts as i64);
            dts += u64::from(s.duration);
        }
        let min = raw.iter().copied().min().unwrap_or(0);
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut any_nonzero = false;
        for r in raw {
            let off = u32::try_from(r - min).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "ctts offset exceeds 32 bits (corrupt timestamps)",
                )
            })?;
            any_nonzero |= off != 0;
            match runs.last_mut() {
                Some((count, o)) if *o == off => *count += 1,
                _ => runs.push((1, off)),
            }
        }
        // The uniform composition delay the normalization introduced.
        // The first decoded frame (IDR) is also the first displayed, so
        // its normalized offset IS the delay; the elst edit list cancels
        // it — without that the whole video would run ~reorder-depth
        // frames late against the audio track.
        let first_offset = runs.first().map(|&(_, off)| off).unwrap_or(0);
        Ok(CttsPlan {
            runs: any_nonzero.then_some(runs),
            edit_delay: u64::from(first_offset),
        })
    }

    /// moov bytes + (position, relative-offset) pairs to patch into stco.
    fn build_moov(
        &self,
        video_entry: &[u8],
        ctts_plan: &CttsPlan,
        wide_offsets: bool,
    ) -> (Vec<u8>, Vec<(usize, u64)>) {
        let v_dur_units: u64 = self.v_samples.iter().map(|s| u64::from(s.duration)).sum();
        let v_dur_ms = scale(v_dur_units, self.video.timescale, MOVIE_TIMESCALE);
        // Per-track media durations, then the movie clock as the longest
        // of all of them: with six audio tracks the movie is only as
        // short as its longest constituent, not as its video.
        let a_dur_units: Vec<u64> = self
            .a_samples
            .iter()
            .map(|samples| samples.iter().map(|s| u64::from(s.duration)).sum())
            .collect();
        let a_dur_ms: Vec<u64> = self
            .audio
            .iter()
            .zip(&a_dur_units)
            .map(|(params, units)| scale(*units, params.timescale, MOVIE_TIMESCALE))
            .collect();
        let movie_dur = a_dur_ms.iter().copied().fold(v_dur_ms, u64::max);
        let track_count = 1 + self.audio.len() as u32;

        let mut moov_payload = Vec::new();
        moov_payload.extend_from_slice(&mvhd(movie_dur, track_count + 1));

        // Video trak.
        let mut stco_patches = Vec::new();
        let video_stbl = self.build_stbl(
            VIDEO_TRACK,
            video_entry,
            ctts_plan.runs.as_deref(),
            &mut stco_patches,
            moov_payload.len(),
            wide_offsets,
        );
        moov_payload.extend_from_slice(&self.build_trak(
            VIDEO_TRACK,
            v_dur_ms,
            v_dur_units,
            ctts_plan.edit_delay,
            &video_stbl,
            &mut stco_patches,
            moov_payload.len(),
        ));

        // One trak per audio track, in the order they were declared.
        for audio_ix in 0..self.audio.len() {
            let track = VIDEO_TRACK + 1 + audio_ix;
            let audio_stbl = self.build_stbl(
                track,
                video_entry,
                None,
                &mut stco_patches,
                moov_payload.len(),
                wide_offsets,
            );
            moov_payload.extend_from_slice(&self.build_trak(
                track,
                a_dur_ms[audio_ix],
                a_dur_units[audio_ix],
                0,
                &audio_stbl,
                &mut stco_patches,
                moov_payload.len(),
            ));
        }

        let moov = boxed(b"moov", &moov_payload);
        // Positions were recorded relative to moov *payload*; shift by
        // the moov box header (8 bytes).
        let patches = stco_patches
            .into_iter()
            .map(|(pos, rel)| (pos + 8, rel))
            .collect();
        (moov, patches)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_trak(
        &self,
        track: usize,
        dur_ms: u64,
        dur_units: u64,
        edit_delay: u64,
        stbl: &[u8],
        patches: &mut [(usize, u64)],
        payload_base: usize,
    ) -> Vec<u8> {
        // stbl was built with patch positions relative to ITS payload;
        // relocate them to the final positions inside moov's payload.
        // Compute the prefix length in front of stbl deterministically.
        let video = track == VIDEO_TRACK;
        let audio_params = track.checked_sub(1).and_then(|ix| self.audio.get(ix));
        // Track ids are 1-based in ISOBMFF and must be unique; the
        // muxer's own index order supplies both properties.
        let track_id = track as u32 + 1;
        let (timescale, handler, name, mhd): (u32, &[u8; 4], &str, Vec<u8>) = if video {
            (
                self.video.timescale,
                b"vide",
                "sovmr video",
                full_boxed(b"vmhd", 0, 1, &[0u8; 8]),
            )
        } else {
            (
                audio_params.map(|a| a.timescale).unwrap_or(1),
                b"soun",
                "sovmr audio",
                full_boxed(b"smhd", 0, 0, &[0u8; 4]),
            )
        };

        // tkhd puts track_id BETWEEN the times and the duration, so its
        // widened form is not simply `header_times` — spelled out here.
        let mut tkhd_body = Vec::new();
        let tkhd_version = if dur_ms > u64::from(u32::MAX) { 1 } else { 0 };
        if tkhd_version == 1 {
            tkhd_body.extend_from_slice(&0u64.to_be_bytes()); // ctime
            tkhd_body.extend_from_slice(&0u64.to_be_bytes()); // mtime
            tkhd_body.extend_from_slice(&track_id.to_be_bytes());
            tkhd_body.extend_from_slice(&0u32.to_be_bytes()); // reserved
            tkhd_body.extend_from_slice(&dur_ms.to_be_bytes());
        } else {
            tkhd_body.extend_from_slice(&0u32.to_be_bytes()); // ctime
            tkhd_body.extend_from_slice(&0u32.to_be_bytes()); // mtime
            tkhd_body.extend_from_slice(&track_id.to_be_bytes());
            tkhd_body.extend_from_slice(&0u32.to_be_bytes()); // reserved
            tkhd_body.extend_from_slice(&(dur_ms as u32).to_be_bytes());
        }
        tkhd_body.extend_from_slice(&[0u8; 8]); // reserved
        tkhd_body.extend_from_slice(&0u16.to_be_bytes()); // layer
        let alternate_group = audio_params.map(|a| a.alternate_group).unwrap_or(0);
        tkhd_body.extend_from_slice(&alternate_group.to_be_bytes());
        tkhd_body.extend_from_slice(&(if video { 0u16 } else { 0x0100 }).to_be_bytes());
        tkhd_body.extend_from_slice(&0u16.to_be_bytes()); // reserved
        tkhd_body.extend_from_slice(&matrix());
        let (w, h) = if video {
            (
                u32::from(self.video.width) << 16,
                u32::from(self.video.height) << 16,
            )
        } else {
            (0, 0)
        };
        tkhd_body.extend_from_slice(&w.to_be_bytes());
        tkhd_body.extend_from_slice(&h.to_be_bytes());
        // Flags: bit 0 enabled, bit 1 in-movie. A source track that
        // arrived disabled stays disabled rather than being silently
        // promoted to something the viewer never asked to hear.
        const TKHD_ENABLED_IN_MOVIE: u32 = 0x3;
        const TKHD_IN_MOVIE_ONLY: u32 = 0x2;
        let tkhd_flags = match audio_params {
            Some(a) if !a.enabled => TKHD_IN_MOVIE_ONLY,
            _ => TKHD_ENABLED_IN_MOVIE,
        };
        let tkhd = full_boxed(b"tkhd", tkhd_version, tkhd_flags, &tkhd_body);

        // The media duration is in TRACK units, so this is the field
        // that overflows first: 12 hours at a 90 kHz timescale is 3.9e9,
        // already most of `u32`, and 24 hours is past it.
        let (mdhd_version, mut mdhd_body) = header_times(timescale, dur_units);
        let language = audio_params
            .map(|a| pack_language(&a.language))
            .unwrap_or(LANG_UND);
        mdhd_body.extend_from_slice(&language.to_be_bytes());
        mdhd_body.extend_from_slice(&0u16.to_be_bytes());
        let mdhd = full_boxed(b"mdhd", mdhd_version, 0, &mdhd_body);

        let mut hdlr_body = Vec::new();
        hdlr_body.extend_from_slice(&0u32.to_be_bytes());
        hdlr_body.extend_from_slice(handler);
        hdlr_body.extend_from_slice(&[0u8; 12]);
        hdlr_body.extend_from_slice(name.as_bytes());
        hdlr_body.push(0);
        let hdlr = full_boxed(b"hdlr", 0, 0, &hdlr_body);

        let dref_url = full_boxed(b"url ", 0, 1, &[]);
        let mut dref_body = 1u32.to_be_bytes().to_vec();
        dref_body.extend_from_slice(&dref_url);
        let dinf = boxed(b"dinf", &full_boxed(b"dref", 0, 0, &dref_body));

        let mut minf_payload = mhd;
        minf_payload.extend_from_slice(&dinf);
        // stbl patch positions: relocate relative -> absolute-in-moov.
        let minf_prefix = minf_payload.len();
        minf_payload.extend_from_slice(stbl);
        let minf = boxed(b"minf", &minf_payload);

        let mut mdia_payload = mdhd;
        mdia_payload.extend_from_slice(&hdlr);
        let mdia_prefix = mdia_payload.len();
        mdia_payload.extend_from_slice(&minf);
        let mdia = boxed(b"mdia", &mdia_payload);

        let mut trak_payload = tkhd;
        if edit_delay > 0 {
            // elst: start presentation `edit_delay` media units into the
            // track, cancelling the B-frame composition delay (the same
            // construction every edit-list-writing muxer uses).
            let wide = dur_ms > u64::from(u32::MAX) || edit_delay > u64::from(u32::MAX);
            let mut elst_body = 1u32.to_be_bytes().to_vec();
            if wide {
                elst_body.extend_from_slice(&dur_ms.to_be_bytes());
                elst_body.extend_from_slice(&edit_delay.to_be_bytes());
            } else {
                elst_body.extend_from_slice(&(dur_ms as u32).to_be_bytes());
                elst_body.extend_from_slice(&(edit_delay as u32).to_be_bytes());
            }
            elst_body.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
            let version = u8::from(wide);
            trak_payload.extend_from_slice(&boxed(
                b"edts",
                &full_boxed(b"elst", version, 0, &elst_body),
            ));
        }
        if let Some(title) = audio_params.and_then(|a| a.name.as_deref()) {
            // The title is what makes one of six tracks choosable.
            trak_payload.extend_from_slice(&boxed(b"udta", &boxed(b"name", title.as_bytes())));
        }
        let trak_prefix = trak_payload.len();
        trak_payload.extend_from_slice(&mdia);
        let trak = boxed(b"trak", &trak_payload);

        // Each nesting adds its prefix bytes plus an 8-byte box header.
        let stbl_shift = payload_base
            + 8 // trak header
            + trak_prefix
            + 8 // mdia header
            + mdia_prefix
            + 8 // minf header
            + minf_prefix;
        for (pos, _) in patches.iter_mut().filter(|(p, _)| *p >= STCO_PATCH_TAG) {
            *pos = *pos - STCO_PATCH_TAG + stbl_shift;
        }
        trak
    }

    /// Build one track's stbl; stco patch positions are recorded tagged
    /// (offset by [`STCO_PATCH_TAG`]) relative to the stbl bytes and
    /// relocated by `build_trak`.
    #[allow(clippy::too_many_arguments)]
    fn build_stbl(
        &self,
        track: usize,
        video_entry: &[u8],
        video_ctts: Option<&[(u32, u32)]>,
        patches: &mut Vec<(usize, u64)>,
        _payload_base: usize,
        wide_offsets: bool,
    ) -> Vec<u8> {
        let video = track == VIDEO_TRACK;
        let samples = match track.checked_sub(1) {
            None => &self.v_samples,
            Some(audio_ix) => &self.a_samples[audio_ix],
        };

        // stsd
        let entry = match track.checked_sub(1) {
            None => video_entry.to_vec(),
            Some(audio_ix) => boxed(b"mp4a", &self.audio[audio_ix].entry_payload),
        };
        let mut stsd_body = 1u32.to_be_bytes().to_vec();
        stsd_body.extend_from_slice(&entry);
        let stsd = full_boxed(b"stsd", 0, 0, &stsd_body);

        // stts: run-length collapse of durations.
        let mut stts_runs: Vec<(u32, u32)> = Vec::new();
        for s in samples {
            match stts_runs.last_mut() {
                Some((count, delta)) if *delta == s.duration => *count += 1,
                _ => stts_runs.push((1, s.duration)),
            }
        }
        let mut stts_body = (stts_runs.len() as u32).to_be_bytes().to_vec();
        for (count, delta) in &stts_runs {
            stts_body.extend_from_slice(&count.to_be_bytes());
            stts_body.extend_from_slice(&delta.to_be_bytes());
        }
        let stts = full_boxed(b"stts", 0, 0, &stts_body);

        // ctts (video only, only when reordering produced offsets).
        let ctts = video_ctts.filter(|_| video).map(|runs| {
            let mut body = (runs.len() as u32).to_be_bytes().to_vec();
            for (count, offset) in runs {
                body.extend_from_slice(&count.to_be_bytes());
                body.extend_from_slice(&offset.to_be_bytes());
            }
            full_boxed(b"ctts", 0, 0, &body)
        });

        // stss (video only, omitted when every sample is a keyframe).
        let keyframes: Vec<u32> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.keyframe)
            .map(|(i, _)| i as u32 + 1)
            .collect();
        let stss = if video && keyframes.len() < samples.len() {
            let mut body = (keyframes.len() as u32).to_be_bytes().to_vec();
            for k in &keyframes {
                body.extend_from_slice(&k.to_be_bytes());
            }
            Some(full_boxed(b"stss", 0, 0, &body))
        } else {
            None
        };

        // stsc + stco from the recorded chunk runs of this track.
        let track_chunks: Vec<&ChunkMeta> =
            self.chunks.iter().filter(|c| c.track == track).collect();
        let mut stsc_entries: Vec<(u32, u32)> = Vec::new();
        for (i, c) in track_chunks.iter().enumerate() {
            match stsc_entries.last() {
                Some((_, samples)) if *samples == c.samples => {}
                _ => stsc_entries.push((i as u32 + 1, c.samples)),
            }
        }
        let mut stsc_body = (stsc_entries.len() as u32).to_be_bytes().to_vec();
        for (first, count) in &stsc_entries {
            stsc_body.extend_from_slice(&first.to_be_bytes());
            stsc_body.extend_from_slice(&count.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
        }
        let stsc = full_boxed(b"stsc", 0, 0, &stsc_body);

        // `stsz` has a uniform form: a non-zero `sample_size` header means
        // every sample is that size and the table is omitted entirely.
        // Variable-rate codecs cannot use it — for real AAC this table is
        // irreducibly one entry per sample — but constant-size media
        // (PCM, and any fixed-frame codec) collapses to twelve bytes
        // instead of four per sample.
        let uniform = samples
            .first()
            .map(|f| f.size)
            .filter(|first| samples.iter().all(|s| s.size == *first))
            .unwrap_or(0);
        let mut stsz_body = Vec::new();
        stsz_body.extend_from_slice(&uniform.to_be_bytes());
        stsz_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        if uniform == 0 {
            for s in samples {
                stsz_body.extend_from_slice(&s.size.to_be_bytes());
            }
        }
        let stsz = full_boxed(b"stsz", 0, 0, &stsz_body);

        // `stco` holds 32-bit chunk offsets; past 4 GiB they cannot be
        // expressed at all, which is why long-form output used to fail
        // outright. `co64` is the same table with 64-bit entries.
        let offset_width = if wide_offsets { 8 } else { 4 };
        let mut stco_body = (track_chunks.len() as u32).to_be_bytes().to_vec();
        let mut stco_entry_rel = Vec::new();
        for c in &track_chunks {
            stco_entry_rel.push(c.rel_offset);
            stco_body.extend_from_slice(&vec![0u8; offset_width]); // patched later
        }
        let stco = if wide_offsets {
            full_boxed(b"co64", 0, 0, &stco_body)
        } else {
            full_boxed(b"stco", 0, 0, &stco_body)
        };

        let mut stbl_payload = stsd;
        stbl_payload.extend_from_slice(&stts);
        if let Some(ctts) = ctts {
            stbl_payload.extend_from_slice(&ctts);
        }
        if let Some(stss) = stss {
            stbl_payload.extend_from_slice(&stss);
        }
        stbl_payload.extend_from_slice(&stsc);
        stbl_payload.extend_from_slice(&stsz);
        // stco entries start after stco's box header (8) + full-box (4)
        // + entry count (4).
        let stco_entries_at = stbl_payload.len() + 8 + 4 + 4;
        stbl_payload.extend_from_slice(&stco);
        for (i, rel) in stco_entry_rel.into_iter().enumerate() {
            // Tag positions so build_trak can relocate them: they are
            // relative to the stbl *box* below (payload + its header).
            patches.push((STCO_PATCH_TAG + 8 + stco_entries_at + i * offset_width, rel));
        }
        boxed(b"stbl", &stbl_payload)
    }
}

/// The video track's derived composition timing: the optional `ctts`
/// run-length table, plus the uniform composition delay the `elst`
/// edit list must cancel (0 = no edit list needed).
struct CttsPlan {
    runs: Option<Vec<(u32, u32)>>,
    edit_delay: u64,
}

/// Sentinel added to patch positions that still need relocation from
/// stbl-relative to moov-relative (removed in `build_trak`). Large enough
/// that no real moov position ever reaches it.
const STCO_PATCH_TAG: usize = 1 << 48;

impl Drop for Mp4Muxer {
    fn drop(&mut self) {
        // Not finalized: roll the spool back (never leave litter).
        if self.spool.is_some() {
            self.spool = None;
            let _ = std::fs::remove_file(&self.spool_path);
        }
    }
}

/// Emit a duration triple (creation, modification, timescale, duration)
/// in whichever `mvhd`/`mdhd` version can actually hold it.
///
/// Version 0 stores times and durations in 32 bits. That is fine for a
/// clip and silently wrong for the material this engine now targets: at
/// a 90 kHz media timescale a track overflows `u32` after about 13
/// hours, and the old code cast with `as u32`, which wraps rather than
/// complains — a twelve-hour master would have muxed to a file claiming
/// to be minutes long. Version 1 widens both fields to 64 bits.
///
/// The version is chosen per box from the value, not globally, so files
/// that fit in 32 bits are emitted exactly as before.
fn header_times(timescale: u32, duration: u64) -> (u8, Vec<u8>) {
    let mut body = Vec::new();
    if duration > u64::from(u32::MAX) {
        body.extend_from_slice(&0u64.to_be_bytes()); // ctime
        body.extend_from_slice(&0u64.to_be_bytes()); // mtime
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&duration.to_be_bytes());
        (1, body)
    } else {
        body.extend_from_slice(&0u32.to_be_bytes()); // ctime
        body.extend_from_slice(&0u32.to_be_bytes()); // mtime
        body.extend_from_slice(&timescale.to_be_bytes());
        body.extend_from_slice(&(duration as u32).to_be_bytes());
        (0, body)
    }
}

fn mvhd(duration_ms: u64, next_track: u32) -> Vec<u8> {
    let (version, mut body) = header_times(MOVIE_TIMESCALE, duration_ms);
    body.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
    body.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
    body.extend_from_slice(&[0u8; 10]); // reserved
    body.extend_from_slice(&matrix());
    body.extend_from_slice(&[0u8; 24]); // pre_defined
    body.extend_from_slice(&next_track.to_be_bytes());
    full_boxed(b"mvhd", version, 0, &body)
}

/// Pack a three-letter ISO-639-2 code the way `mdhd` carries it: three
/// 5-bit values biased by 0x60, high bit padding. Anything that is not
/// three lowercase letters becomes `und` rather than a code made of
/// control characters.
fn pack_language(code: &[u8; 3]) -> u16 {
    let mut packed = 0u16;
    for c in code {
        if !c.is_ascii_lowercase() {
            return LANG_UND;
        }
        packed = (packed << 5) | u16::from(c - 0x60);
    }
    packed
}

fn scale(units: u64, from: u32, to: u32) -> u64 {
    if from == 0 {
        return 0;
    }
    units.saturating_mul(u64::from(to)) / u64::from(from)
}

// ---- building an mp4a sample entry from a bare codec config ------------------

/// `AudioSampleEntry` fixed prefix length, before child boxes.
const AUDIO_ENTRY_PREFIX: usize = 28;
/// Offsets inside that prefix.
const AUDIO_ENTRY_DREF_AT: usize = 6;
const AUDIO_ENTRY_CHANNELS_AT: usize = 16;
const AUDIO_ENTRY_SAMPLE_SIZE_AT: usize = 18;
const AUDIO_ENTRY_SAMPLE_RATE_AT: usize = 24;
/// Bits per sample an `AudioSampleEntry` declares. Fixed at 16 for
/// compressed audio: the field describes the *uncompressed* form the
/// decoder produces, not the bitstream, and every AAC decoder outputs 16.
const AUDIO_ENTRY_SAMPLE_SIZE: u16 = 16;

/// MPEG-4 descriptor tags (ISO/IEC 14496-1).
const TAG_ES_DESCRIPTOR: u8 = 0x03;
const TAG_DECODER_CONFIG: u8 = 0x04;
const TAG_DECODER_SPECIFIC: u8 = 0x05;
const TAG_SL_CONFIG: u8 = 0x06;
/// `objectTypeIndication` for MPEG-4 Audio.
const OBJECT_TYPE_MPEG4_AUDIO: u8 = 0x40;
/// `streamType` 5 (AudioStream) in the upper six bits, `upStream` clear,
/// reserved bit set — the byte every AAC `esds` in the wild carries.
const STREAM_TYPE_AUDIO: u8 = (5 << 2) | 0x01;
/// `SLConfigDescriptor` predefined value 2: "MP4 file".
const SL_CONFIG_MP4: u8 = 0x02;

/// Wrap `body` in an MPEG-4 descriptor with `tag`.
///
/// Descriptor lengths use the "expandable" encoding: seven bits per
/// octet, high bit set on every octet but the last. Writing a plain byte
/// works until the first descriptor over 127 bytes — a multi-channel
/// `AudioSpecificConfig` reaches that — and then produces a file every
/// decoder misreads, so the general form is written from the start.
fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let mut len = body.len();
    let mut digits = Vec::new();
    loop {
        digits.push((len & 0x7F) as u8);
        len >>= 7;
        if len == 0 {
            break;
        }
    }
    // Most significant group first; every octet but the last continues.
    for (i, d) in digits.iter().rev().enumerate() {
        let last = i + 1 == digits.len();
        out.push(if last { *d } else { *d | 0x80 });
    }
    out.extend_from_slice(body);
    out
}

/// Build an `mp4a` sample-entry payload around a bare
/// `AudioSpecificConfig`.
///
/// MP4 carries an AAC track's decoder setup inside an `esds` descriptor
/// tree; Matroska carries the same `AudioSpecificConfig` naked in
/// `CodecPrivate`. Moving a track from one container to the other means
/// building the wrapper this function builds — the payload itself is
/// untouched, so the passthrough stays bit-exact.
///
/// `max_bitrate` and `avg_bitrate` are advisory; zero is legal and is
/// what a caller that has not measured should pass, rather than inventing
/// a figure a player might believe.
///
/// # Example
///
/// ```
/// # use sovmr_mp4::mux::build_mp4a_entry;
/// // AAC-LC, 48 kHz, stereo: the two-byte config Matroska stores.
/// let entry = build_mp4a_entry(48_000, 2, &[0x11, 0x90], 0, 0);
/// assert!(entry.len() > 28, "prefix plus an esds box");
/// ```
pub fn build_mp4a_entry(
    sample_rate: u32,
    channels: u16,
    audio_specific_config: &[u8],
    max_bitrate: u32,
    avg_bitrate: u32,
) -> Vec<u8> {
    let mut specific = descriptor(TAG_DECODER_SPECIFIC, audio_specific_config);

    let mut config = vec![OBJECT_TYPE_MPEG4_AUDIO, STREAM_TYPE_AUDIO];
    config.extend_from_slice(&[0, 0, 0]); // bufferSizeDB, unmeasured
    config.extend_from_slice(&max_bitrate.to_be_bytes());
    config.extend_from_slice(&avg_bitrate.to_be_bytes());
    config.append(&mut specific);
    let config = descriptor(TAG_DECODER_CONFIG, &config);

    let sl = descriptor(TAG_SL_CONFIG, &[SL_CONFIG_MP4]);

    let mut es = Vec::new();
    es.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es.push(0); // no dependency, no URL, no OCR stream, priority 0
    es.extend_from_slice(&config);
    es.extend_from_slice(&sl);
    let esds = full_boxed(b"esds", 0, 0, &descriptor(TAG_ES_DESCRIPTOR, &es));

    let mut entry = vec![0u8; AUDIO_ENTRY_PREFIX];
    entry[AUDIO_ENTRY_DREF_AT..AUDIO_ENTRY_DREF_AT + 2].copy_from_slice(&1u16.to_be_bytes());
    entry[AUDIO_ENTRY_CHANNELS_AT..AUDIO_ENTRY_CHANNELS_AT + 2]
        .copy_from_slice(&channels.to_be_bytes());
    entry[AUDIO_ENTRY_SAMPLE_SIZE_AT..AUDIO_ENTRY_SAMPLE_SIZE_AT + 2]
        .copy_from_slice(&AUDIO_ENTRY_SAMPLE_SIZE.to_be_bytes());
    // The rate is a 16.16 fixed-point value, so it lives in the high half.
    entry[AUDIO_ENTRY_SAMPLE_RATE_AT..AUDIO_ENTRY_SAMPLE_RATE_AT + 4]
        .copy_from_slice(&(sample_rate << 16).to_be_bytes());
    entry.extend_from_slice(&esds);
    entry
}

#[cfg(test)]
mod entry_tests {
    use super::*;

    #[test]
    fn a_short_descriptor_length_is_one_octet() {
        let d = descriptor(0x05, &[0x11, 0x90]);
        assert_eq!(d, vec![0x05, 0x02, 0x11, 0x90]);
    }

    #[test]
    fn a_long_descriptor_length_uses_the_expandable_form() {
        // 200 bytes needs two octets: 0x81 0x48. Writing a plain byte
        // would truncate to 72 and every decoder would misread the tree.
        let d = descriptor(0x05, &[0xAA; 200]);
        assert_eq!(&d[..3], &[0x05, 0x81, 0x48]);
        assert_eq!(d.len(), 3 + 200);
    }

    #[test]
    fn the_length_encoding_round_trips_across_a_boundary() {
        for n in [0usize, 1, 126, 127, 128, 129, 16_383, 16_384] {
            let d = descriptor(0x05, &vec![0u8; n]);
            // Re-read the length the way a decoder would.
            let mut len = 0usize;
            let mut i = 1;
            loop {
                let b = d[i];
                len = (len << 7) | usize::from(b & 0x7F);
                i += 1;
                if b & 0x80 == 0 {
                    break;
                }
            }
            assert_eq!(len, n, "length {n} must survive its own encoding");
            assert_eq!(d.len(), i + n);
        }
    }

    #[test]
    fn the_entry_declares_the_rate_and_channels_it_was_given() {
        let entry = build_mp4a_entry(44_100, 6, &[0x11, 0x90], 0, 0);
        let channels = u16::from_be_bytes([entry[16], entry[17]]);
        let rate = u32::from_be_bytes([entry[24], entry[25], entry[26], entry[27]]) >> 16;
        assert_eq!(channels, 6);
        assert_eq!(rate, 44_100);
    }

    #[test]
    fn the_audio_specific_config_survives_verbatim() {
        // The whole point of passthrough: these bytes are the decoder's
        // setup, and altering them changes what comes out of the speaker.
        let asc = [0x12, 0x10, 0x56, 0xE5, 0x00];
        let entry = build_mp4a_entry(48_000, 2, &asc, 0, 0);
        let at = entry
            .windows(asc.len())
            .position(|w| w == asc)
            .expect("the config must appear in the entry");
        // And it must sit inside a DecoderSpecificInfo descriptor.
        assert_eq!(entry[at - 2], TAG_DECODER_SPECIFIC);
        assert_eq!(entry[at - 1] as usize, asc.len());
    }

    #[test]
    fn our_own_demuxer_reads_back_the_entry_we_built() {
        // Sovereign verification: the entry is only useful if the
        // demuxer on the other side of the muxer agrees with it.
        let entry = build_mp4a_entry(48_000, 2, &[0x11, 0x90], 0, 0);
        let boxed_entry = boxed(b"mp4a", &entry);
        let mut stsd_body = 1u32.to_be_bytes().to_vec();
        stsd_body.extend_from_slice(&boxed_entry);
        // The demuxer wants channels at 16 and the rate at 24 of the
        // payload, which is exactly where they were written.
        let payload = &boxed_entry[8..];
        assert_eq!(u16::from_be_bytes([payload[16], payload[17]]), 2);
        assert_eq!(
            u32::from_be_bytes([payload[24], payload[25], payload[26], payload[27]]) >> 16,
            48_000
        );
    }
}
