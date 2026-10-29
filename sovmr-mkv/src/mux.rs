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

//! The sovereign Matroska muxer.
//!
//! # Why this is so much smaller than the MP4 muxer
//!
//! ISOBMFF puts its index in a header, so writing a fast-start file
//! means spooling every payload to disk, building `moov` once the sizes
//! are known, and copying the spool in behind it. Matroska interleaves
//! its index with the media: clusters are written as they arrive and the
//! seek table goes at the end. Nothing needs to be written twice.
//!
//! Two fields still have to be patched afterwards — the segment's length
//! and its duration — so they are written at a fixed width up front and
//! overwritten in place at the end. That is a seek of a few bytes, not a
//! second pass over the media.
//!
//! # ACID
//!
//! The caller writes to a temporary path and renames on success, exactly
//! as with MP4. A muxer that failed halfway leaves a file the rename
//! never happened to, rather than a plausible-looking ruin.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::ids;

/// Element ids used only when writing.
const EBML_HEADER: u32 = 0x1A45_DFA3;
const EBML_VERSION: u32 = 0x4286;
const EBML_READ_VERSION: u32 = 0x42F7;
const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
const DOC_TYPE: u32 = 0x4282;
const DOC_TYPE_VERSION: u32 = 0x4287;
const DOC_TYPE_READ_VERSION: u32 = 0x4285;
const MUXING_APP: u32 = 0x4D80;
const WRITING_APP: u32 = 0x5741;
const CLUSTER: u32 = 0x1F43_B675;
const CLUSTER_TIMESTAMP: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;
const CUES: u32 = 0x1C53_BB6B;
const CUE_POINT: u32 = 0xBB;
const CUE_TIME: u32 = 0xB3;
const CUE_TRACK_POSITIONS: u32 = 0xB7;
const CUE_TRACK: u32 = 0xF7;
const CUE_CLUSTER_POSITION: u32 = 0xF1;

/// Nanoseconds per timestamp tick in the files this muxer writes.
///
/// One millisecond, the universal convention. It costs no accumulating
/// drift, because a block's time is absolute (its cluster's timestamp
/// plus its own offset) rather than a delta added to the frame before.
pub const TIMESTAMP_SCALE_NANOS: u64 = 1_000_000;

/// Milliseconds a cluster may span before a new one is started.
///
/// A block's offset from its cluster is a *signed 16-bit* value, so
/// ±32.767 s is the hard limit the format imposes. Staying far below it
/// also keeps seeking responsive: a player entering at a cue reads from
/// the cluster start, so a long cluster is a long wait.
const CLUSTER_SPAN_MS: i64 = 2_000;

/// Width of the patched segment-size field, in octets.
///
/// Eight is the maximum a size may use, and writing the maximum from the
/// start is what makes the field patchable: a shorter encoding would
/// need the file moved when the real length turned out larger.
const SEGMENT_SIZE_OCTETS: usize = 8;

/// Encode a length as an EBML variable-size integer, shortest first.
fn vint(value: u64) -> Vec<u8> {
    for octets in 1..=8u32 {
        let bits = octets * 7;
        let max = (1u64 << bits) - 1;
        // The all-ones pattern is the "unknown size" sentinel, so a
        // length that would land on it must use one more octet.
        if value < max {
            let marker = 1u64 << bits;
            let raw = marker | value;
            return raw.to_be_bytes()[8 - octets as usize..].to_vec();
        }
    }
    // Unreachable for any length a file can hold, but a panic here would
    // be a crash on a large file rather than an error.
    vec![0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]
}

/// The bytes of an element id, marker included.
fn id_bytes(id: u32) -> Vec<u8> {
    let b = id.to_be_bytes();
    let start = b.iter().position(|x| *x != 0).unwrap_or(3);
    b[start..].to_vec()
}

/// Serialize one element.
pub fn element(id: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend_from_slice(&vint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

/// Serialize an unsigned integer element in as few octets as it needs.
fn uint_element(id: u32, value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let start = bytes.iter().position(|b| *b != 0).unwrap_or(7);
    element(id, &bytes[start..])
}

/// Serialize a string element.
fn string_element(id: u32, value: &str) -> Vec<u8> {
    element(id, value.as_bytes())
}

/// One track to write, described the way this muxer needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackSpec {
    /// One-based track number blocks refer to.
    pub number: u64,
    /// Matroska codec identifier, e.g. `V_AV1`.
    pub codec_id: String,
    /// Codec setup bytes, verbatim.
    pub codec_private: Vec<u8>,
    /// Whether this is the video track (exactly one must be).
    pub video: bool,
    /// Whether this is a subtitle track. Matroska carries subtitles as
    /// ordinary tracks with their own type, so nothing else differs.
    pub subtitle: bool,
    /// Coded pixel size, for video.
    pub width: u32,
    /// Coded pixel height, for video.
    pub height: u32,
    /// Sampling frequency, for audio.
    pub sample_rate: u32,
    /// Channel count, for audio.
    pub channels: u16,
    /// ISO 639-2 or BCP 47 language.
    pub language: String,
    /// Human title, when the source named the track.
    pub name: Option<String>,
    /// Whether the track is enabled.
    pub enabled: bool,
    /// What the source said its colours mean, for video tracks.
    pub colour: Option<sovmr_mp4::colour::Colour>,
}

/// A cue entry: where a keyframe starts, so a player can seek to it.
struct Cue {
    time_ms: i64,
    track: u64,
    cluster_at: u64,
}

/// One chapter mark to write.
///
/// The demuxer's [`crate::demux::Chapter`] carries borrowed text; this
/// owns it, because a muxer outlives the file it read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterSpec {
    /// The source file's own identifier, carried through rather than
    /// invented — a player that remembered where a viewer stopped finds
    /// the same chapter after a re-mux.
    pub uid: u64,
    /// Where it starts, in nanoseconds.
    pub start_nanos: u64,
    /// Where it ends. Not greater than the start means the chapter runs
    /// until the next one, which is how a file that omitted the element
    /// said it.
    pub end_nanos: u64,
    /// What it is called.
    pub title: String,
    /// The language that name is in.
    pub language: String,
}

/// One descriptive entry to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagSpec {
    /// The track this is about, or `None` for the whole file.
    pub track_uid: Option<u64>,
    /// The entry's name, as it will be written.
    pub name: String,
    /// Its value.
    pub value: String,
}

/// Writes a Matroska file, one frame at a time.
pub struct MkvMuxer {
    out: BufWriter<File>,
    path: PathBuf,
    /// Byte offset of the segment's patchable size field.
    segment_size_at: u64,
    /// Byte offset just past that field — where the segment's content
    /// begins, and the origin cue positions are relative to.
    segment_body_at: u64,
    /// Byte offset of the duration's fixed-width float.
    duration_at: u64,
    /// The video track's number, for cues and cluster boundaries.
    video_track: u64,
    /// Buffer for the cluster being assembled.
    cluster: Vec<u8>,
    /// The open cluster's timestamp in milliseconds, if any.
    cluster_start_ms: Option<i64>,
    /// Where the open cluster will be written.
    cluster_at: u64,
    /// Cue points collected as keyframes go by.
    cues: Vec<Cue>,
    /// Chapters and tags, written in front of the media.
    chapters: Vec<ChapterSpec>,
    tags: Vec<TagSpec>,
    /// Where the file stood when it was created, so a late
    /// [`set_marks`](MkvMuxer::set_marks) can be refused rather than
    /// silently placing the marks where a reader will not look.
    first_frame_would_be: u64,
    /// Highest end time seen, for the duration.
    last_ms: i64,
    /// Bytes written so far, for cue positions.
    written: u64,
    /// Set once `finalize` has run, so `Drop` does not delete the file.
    committed: bool,
}

impl MkvMuxer {
    /// Begin a Matroska file at `path`, declaring `tracks`.
    ///
    /// # Errors
    ///
    /// Any I/O error creating or writing the file.
    pub fn create(path: &Path, tracks: &[TrackSpec]) -> io::Result<Self> {
        Self::create_titled(path, tracks, None)
    }

    /// The same, carrying the segment's title.
    ///
    /// A separate constructor because the title lives in `Info`, which
    /// is written at creation and patched in place afterwards — it
    /// cannot arrive later the way chapters and tags do.
    ///
    /// Measured, and the reason this exists: ffmpeg puts a file's name
    /// in `Info/Title` and *not* in `Tags`, so a muxer that carried only
    /// the tag list dropped the one piece of description a viewer is
    /// most likely to notice. The live gate caught it.
    ///
    /// # Errors
    ///
    /// Any write that fails.
    pub fn create_titled(
        path: &Path,
        tracks: &[TrackSpec],
        title: Option<&str>,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut out = BufWriter::new(file);
        let mut written = 0u64;

        // ---- EBML header ------------------------------------------
        let mut header = uint_element(EBML_VERSION, 1);
        header.extend(uint_element(EBML_READ_VERSION, 1));
        header.extend(uint_element(EBML_MAX_ID_LENGTH, 4));
        header.extend(uint_element(EBML_MAX_SIZE_LENGTH, 8));
        header.extend(string_element(DOC_TYPE, "matroska"));
        header.extend(uint_element(DOC_TYPE_VERSION, 4));
        header.extend(uint_element(DOC_TYPE_READ_VERSION, 2));
        let header = element(EBML_HEADER, &header);
        out.write_all(&header)?;
        written += header.len() as u64;

        // ---- Segment, with a size we come back and fill in ---------
        let seg_id = id_bytes(ids::SEGMENT);
        out.write_all(&seg_id)?;
        written += seg_id.len() as u64;
        let segment_size_at = written;
        // A full-width size, so the real length always fits where the
        // placeholder was and the file never has to be moved.
        out.write_all(&[0x01, 0, 0, 0, 0, 0, 0, 0])?;
        written += SEGMENT_SIZE_OCTETS as u64;
        let segment_body_at = written;

        // ---- Info -------------------------------------------------
        let mut info = uint_element(ids::TIMESTAMP_SCALE, TIMESTAMP_SCALE_NANOS);
        if let Some(title) = title.filter(|t| !t.is_empty()) {
            info.extend(string_element(ids::TITLE, title));
        }
        info.extend(string_element(
            MUXING_APP,
            concat!("sovmr ", env!("CARGO_PKG_VERSION")),
        ));
        info.extend(string_element(
            WRITING_APP,
            concat!("sovmr ", env!("CARGO_PKG_VERSION")),
        ));
        // Duration is a float, written at a fixed eight octets so it can
        // be patched once the last frame is known.
        let duration_offset_in_info = info.len();
        info.extend(element(ids::DURATION, &0f64.to_be_bytes()));
        let info_el = element(ids::INFO, &info);
        // Where the float landed in the file: element header, then the
        // offset within the payload, then that element's own header.
        let info_header = info_el.len() - info.len();
        let duration_at = written
            + info_header as u64
            + duration_offset_in_info as u64
            + (id_bytes(ids::DURATION).len() + 1) as u64;
        out.write_all(&info_el)?;
        written += info_el.len() as u64;

        // ---- Tracks -----------------------------------------------
        let mut body = Vec::new();
        for t in tracks {
            body.extend(track_entry(t));
        }
        let tracks_el = element(ids::TRACKS, &body);
        out.write_all(&tracks_el)?;
        written += tracks_el.len() as u64;

        let video_track = tracks
            .iter()
            .find(|t| t.video)
            .map(|t| t.number)
            .unwrap_or(1);
        Ok(MkvMuxer {
            chapters: Vec::new(),
            tags: Vec::new(),
            first_frame_would_be: written,
            out,
            path: path.to_path_buf(),
            segment_size_at,
            segment_body_at,
            duration_at,
            video_track,
            cluster: Vec::new(),
            cluster_start_ms: None,
            cluster_at: written,
            cues: Vec::new(),
            last_ms: 0,
            written,
            committed: false,
        })
    }

    /// Append one frame.
    ///
    /// `timestamp_ms` and `duration_ms` are absolute milliseconds.
    ///
    /// # Errors
    ///
    /// Any I/O error writing a completed cluster.
    pub fn add_frame(
        &mut self,
        track: u64,
        timestamp_ms: i64,
        duration_ms: i64,
        keyframe: bool,
        data: &[u8],
    ) -> io::Result<()> {
        // A new cluster on a video keyframe, or when the open one has
        // run as long as it may. Both matter: cues point at cluster
        // starts, so a keyframe that is not one cannot be sought to.
        let needs_new = match self.cluster_start_ms {
            None => true,
            Some(start) => {
                (keyframe && track == self.video_track) || timestamp_ms - start >= CLUSTER_SPAN_MS
            }
        };
        if needs_new {
            self.flush_cluster()?;
            self.cluster_start_ms = Some(timestamp_ms);
            self.cluster_at = self.written;
        }
        let start = self.cluster_start_ms.unwrap_or(timestamp_ms);
        // The offset is a signed 16-bit field; the cluster rule above
        // keeps it in range, and clamping is the safe reading if a
        // caller ever hands over timestamps that jump.
        let relative =
            (timestamp_ms - start).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;

        let mut block = vint(track);
        block.extend_from_slice(&relative.to_be_bytes());
        block.push(if keyframe { 0x80 } else { 0x00 });
        block.extend_from_slice(data);
        self.cluster.extend(element(SIMPLE_BLOCK, &block));

        if keyframe && track == self.video_track {
            self.cues.push(Cue {
                time_ms: timestamp_ms,
                track,
                cluster_at: self.cluster_at,
            });
        }
        self.last_ms = self.last_ms.max(timestamp_ms + duration_ms.max(0));
        Ok(())
    }

    /// Write the cluster under assembly, if it holds anything.
    fn flush_cluster(&mut self) -> io::Result<()> {
        let Some(start) = self.cluster_start_ms.take() else {
            return Ok(());
        };
        if self.cluster.is_empty() {
            return Ok(());
        }
        let mut body = uint_element(CLUSTER_TIMESTAMP, start.max(0) as u64);
        body.extend_from_slice(&self.cluster);
        let el = element(CLUSTER, &body);
        self.out.write_all(&el)?;
        self.written += el.len() as u64;
        self.cluster.clear();
        Ok(())
    }

    /// Close the file: flush the last cluster, write the seek index, and
    /// patch the two fields that could only be known at the end.
    ///
    /// # Errors
    ///
    /// Any I/O error writing or patching.
    /// Carry chapters and tags into the file being written.
    ///
    /// # Why this must come before the first frame
    ///
    /// Matroska permits these anywhere inside the segment, and the first
    /// attempt here wrote them at the end alongside the cues — where our
    /// own reader found them and **ffprobe did not**. A reader that has
    /// no seek head to follow stops looking once the clusters begin, and
    /// a file whose chapters only this project can see is a file that
    /// has lost them.
    ///
    /// So they go in front of the media, which means they have to be
    /// known before it. That is no hardship: they come from the source
    /// demuxer, which read them before a frame was ever encoded.
    ///
    /// # Errors
    ///
    /// [`io::ErrorKind::InvalidInput`] if a frame has already been
    /// written. Silently dropping them would be the loss this whole
    /// method exists to prevent.
    pub fn set_marks(&mut self, chapters: Vec<ChapterSpec>, tags: Vec<TagSpec>) -> io::Result<()> {
        if self.cluster_start_ms.is_some() || self.written > self.first_frame_would_be {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "chapters and tags must be set before the first frame",
            ));
        }
        self.chapters = chapters;
        self.tags = tags;
        self.write_marks()
    }

    /// Write the two elements, in front of the media.
    fn write_marks(&mut self) -> io::Result<()> {
        for el in [self.chapters_element(), self.tags_element()]
            .into_iter()
            .flatten()
        {
            self.out.write_all(&el)?;
            self.written += el.len() as u64;
        }
        Ok(())
    }

    /// The `Chapters` element, or nothing when there are none.
    ///
    /// An empty chapter list is not the same as no chapter list: writing
    /// an edition with no atoms in it makes some readers show a film
    /// with one nameless chapter, which is worse than showing none.
    fn chapters_element(&self) -> Option<Vec<u8>> {
        if self.chapters.is_empty() {
            return None;
        }
        let mut edition = uint_element(ids::EDITION_FLAG_DEFAULT, 1);
        for chapter in &self.chapters {
            let mut atom = uint_element(ids::CHAPTER_UID, chapter.uid);
            atom.extend(uint_element(ids::CHAPTER_TIME_START, chapter.start_nanos));
            if chapter.end_nanos > chapter.start_nanos {
                atom.extend(uint_element(ids::CHAPTER_TIME_END, chapter.end_nanos));
            }
            if !chapter.title.is_empty() {
                let mut display = string_element(ids::CHAP_STRING, &chapter.title);
                display.extend(string_element(ids::CHAP_LANGUAGE, &chapter.language));
                atom.extend(element(ids::CHAPTER_DISPLAY, &display));
            }
            edition.extend(element(ids::CHAPTER_ATOM, &atom));
        }
        Some(element(
            ids::CHAPTERS,
            &element(ids::EDITION_ENTRY, &edition),
        ))
    }

    /// The `Tags` element, or nothing when there are none.
    ///
    /// One `Tag` per target rather than one per entry: the nesting is
    /// how the format says "these all describe the same thing", and
    /// flattening it into one target apiece would be larger and would
    /// lose that statement.
    fn tags_element(&self) -> Option<Vec<u8>> {
        if self.tags.is_empty() {
            return None;
        }
        let mut targets: Vec<Option<u64>> = Vec::new();
        for tag in &self.tags {
            if !targets.contains(&tag.track_uid) {
                targets.push(tag.track_uid);
            }
        }
        let mut body = Vec::new();
        for target in targets {
            let mut tag = match target {
                Some(uid) => element(ids::TARGETS, &uint_element(ids::TAG_TRACK_UID, uid)),
                // An empty `Targets` is how a file says "the whole
                // file", and it is what ffmpeg writes.
                None => element(ids::TARGETS, &[]),
            };
            for entry in self.tags.iter().filter(|t| t.track_uid == target) {
                let mut simple = string_element(ids::TAG_NAME, &entry.name);
                simple.extend(string_element(ids::TAG_STRING, &entry.value));
                tag.extend(element(ids::SIMPLE_TAG, &simple));
            }
            body.extend(element(ids::TAG, &tag));
        }
        Some(element(ids::TAGS, &body))
    }

    /// Close the file: flush the last cluster, write the chapters, tags
    /// and cues, and patch the two fields written blind.
    ///
    /// # Errors
    ///
    /// Any write or seek that fails.
    pub fn finalize(mut self) -> io::Result<u64> {
        self.flush_cluster()?;

        // ---- Cues -------------------------------------------------
        if !self.cues.is_empty() {
            let mut body = Vec::new();
            for cue in &self.cues {
                let mut positions = uint_element(CUE_TRACK, cue.track);
                // Cue positions are measured from the start of the
                // segment's content, not the start of the file.
                positions.extend(uint_element(
                    CUE_CLUSTER_POSITION,
                    cue.cluster_at.saturating_sub(self.segment_body_at),
                ));
                let mut point = uint_element(CUE_TIME, cue.time_ms.max(0) as u64);
                point.extend(element(CUE_TRACK_POSITIONS, &positions));
                body.extend(element(CUE_POINT, &point));
            }
            let el = element(CUES, &body);
            self.out.write_all(&el)?;
            self.written += el.len() as u64;
        }

        // ---- Patch the two fields written blind --------------------
        self.out.flush()?;
        let segment_len = self.written - self.segment_body_at;
        let mut size = [0x01u8, 0, 0, 0, 0, 0, 0, 0];
        size[1..].copy_from_slice(&segment_len.to_be_bytes()[1..]);
        self.out.seek(SeekFrom::Start(self.segment_size_at))?;
        self.out.write_all(&size)?;

        let duration_ticks = self.last_ms as f64;
        self.out.seek(SeekFrom::Start(self.duration_at))?;
        self.out.write_all(&duration_ticks.to_be_bytes())?;
        self.out.flush()?;

        self.committed = true;
        Ok(self.written)
    }
}

impl Drop for MkvMuxer {
    fn drop(&mut self) {
        // A muxer dropped without finalizing wrote a file no reader
        // should ever see. Removing it is the same contract the MP4
        // spool honours.
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Serialize one `TrackEntry`.
fn track_entry(t: &TrackSpec) -> Vec<u8> {
    let mut e = uint_element(ids::TRACK_NUMBER, t.number);
    // TrackUID must be non-zero and unique; the number serves, being
    // both.
    e.extend(uint_element(ids::TRACK_UID, t.number));
    e.extend(uint_element(
        ids::TRACK_TYPE,
        match (t.video, t.subtitle) {
            (true, _) => ids::TYPE_VIDEO,
            (_, true) => ids::TYPE_SUBTITLE,
            _ => ids::TYPE_AUDIO,
        },
    ));
    e.extend(uint_element(ids::FLAG_ENABLED, u64::from(t.enabled)));
    e.extend(string_element(ids::CODEC_ID, &t.codec_id));
    if !t.codec_private.is_empty() {
        e.extend(element(ids::CODEC_PRIVATE, &t.codec_private));
    }
    if !t.language.is_empty() {
        e.extend(string_element(ids::LANGUAGE, &t.language));
    }
    if let Some(name) = &t.name {
        e.extend(string_element(ids::NAME, name));
    }
    if t.video {
        let mut v = uint_element(ids::PIXEL_WIDTH, u64::from(t.width));
        v.extend(uint_element(ids::PIXEL_HEIGHT, u64::from(t.height)));
        // What the source said its colours mean, carried across. Written
        // only when it said something: a `Colour` full of "unspecified"
        // is a claim to have described nothing, which is what its
        // absence already says.
        if let Some(colour) = t.colour.filter(sovmr_mp4::colour::Colour::is_meaningful) {
            let mut c = uint_element(ids::COLOUR_PRIMARIES, u64::from(colour.primaries));
            c.extend(uint_element(
                ids::COLOUR_TRANSFER,
                u64::from(colour.transfer),
            ));
            c.extend(uint_element(ids::COLOUR_MATRIX, u64::from(colour.matrix)));
            c.extend(uint_element(
                ids::COLOUR_RANGE,
                if colour.full_range {
                    ids::RANGE_FULL
                } else {
                    ids::RANGE_LIMITED
                },
            ));
            v.extend(element(ids::COLOUR, &c));
        }
        e.extend(element(ids::VIDEO, &v));
    } else if t.subtitle {
        // Subtitles carry no `Audio` or `Video` element at all; the
        // codec name and its private data are the whole description.
    } else {
        // SamplingFrequency is a float by specification, even though
        // every real rate is a whole number.
        let mut a = element(
            ids::SAMPLING_FREQUENCY,
            &f64::from(t.sample_rate).to_be_bytes(),
        );
        a.extend(uint_element(ids::CHANNELS, u64::from(t.channels)));
        e.extend(element(ids::AUDIO, &a));
    }
    element(ids::TRACK_ENTRY, &e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_encode_in_the_fewest_octets_that_fit() {
        assert_eq!(vint(0), vec![0x80]);
        assert_eq!(vint(1), vec![0x81]);
        assert_eq!(vint(126), vec![0xFE]);
        // 127 is all-ones in one octet — the unknown-size sentinel — so
        // it must widen rather than claim an unbounded element.
        assert_eq!(vint(127), vec![0x40, 0x7F]);
        assert_eq!(vint(128), vec![0x40, 0x80]);
        assert_eq!(vint(16_382), vec![0x7F, 0xFE]);
        assert_eq!(vint(16_383), vec![0x20, 0x3F, 0xFF]);
    }

    #[test]
    fn what_we_write_our_own_reader_reads_back() {
        // Sovereign verification at the smallest scale: the writer and
        // the reader must agree about every length they exchange.
        for n in [0usize, 1, 126, 127, 128, 5000] {
            let payload = vec![0xABu8; n];
            let bytes = element(0x80, &payload);
            let e = crate::ebml::elements(&bytes).next().unwrap().unwrap();
            assert_eq!(e.id, 0x80);
            assert_eq!(e.payload.len(), n, "length {n} must survive");
            assert!(!e.unknown_size, "a real length is never 'unknown'");
        }
    }

    #[test]
    fn integers_drop_their_leading_zeros_but_never_vanish() {
        assert_eq!(uint_element(0x80, 0), vec![0x80, 0x81, 0x00]);
        assert_eq!(uint_element(0x80, 1), vec![0x80, 0x81, 0x01]);
        assert_eq!(uint_element(0x80, 256), vec![0x80, 0x82, 0x01, 0x00]);
    }
}
