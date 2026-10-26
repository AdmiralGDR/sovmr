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

//! Blocks: where the actual frames live.
//!
//! A `Cluster` carries a timestamp and a run of blocks; each block names
//! its track, an offset from that timestamp, and one or more frames.
//!
//! # Lacing, and why it exists
//!
//! Audio frames are tiny — an AAC packet is a few hundred bytes — and a
//! block header costs several bytes per frame. Matroska therefore lets
//! one block carry several frames, with their sizes encoded in one of
//! three ways. All three are in the wild, so all three are read here.
//!
//! | Lacing | Sizes are… |
//! |---|---|
//! | none | one frame, the rest of the block |
//! | Xiph | run-length bytes: 255s accumulate, a value below 255 ends it |
//! | fixed | not stored at all; every frame is the same size |
//! | EBML | first an unsigned VINT, then *signed* differences |
//!
//! The last frame's size is never stored in any scheme: it is whatever
//! remains. That is the rule that makes a malformed size list dangerous
//! rather than merely wrong, so every scheme here is checked against the
//! bytes actually present before a frame is handed out.
//!
//! Nothing allocates: a laced block's sizes land in a fixed array on the
//! stack, because [`MAX_FRAMES_PER_BLOCK`] is a hard limit of the format
//! rather than a guess.

use crate::ebml::{self, EbmlError};

// These ids are declared here rather than in `crate::ids` because
// nothing outside this module reads a cluster; keeping them local means
// the shared table holds only what more than one caller needs.

/// `Cluster`, and the elements inside it that carry frames.
const CLUSTER: u32 = 0x1F43_B675;
const CLUSTER_TIMESTAMP: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
/// Present in a `BlockGroup` when its `Block` depends on another frame —
/// its *absence* is how a plain `Block` says "keyframe".
const REFERENCE_BLOCK: u32 = 0xFB;

/// Flag bits in a block header.
const FLAG_KEYFRAME: u8 = 0x80;
const FLAG_LACING_MASK: u8 = 0x06;
const LACING_NONE: u8 = 0x00;
const LACING_XIPH: u8 = 0x02;
const LACING_FIXED: u8 = 0x04;
const LACING_EBML: u8 = 0x06;

/// Frames one block may carry.
///
/// The format stores the count as "frames minus one" in a single octet,
/// so 256 is the ceiling the encoding itself imposes — not a limit
/// invented here. It is what lets the size list live on the stack.
pub const MAX_FRAMES_PER_BLOCK: usize = 256;

/// A single coded frame, borrowed from the file.
///
/// # Why the fields are laid out this way
///
/// Matroska has no index, so every frame in the file has to be walked
/// and remembered before the first one can be handed over. Measured, on
/// a real file: **288 frames per second** for one video track and six
/// audio tracks, which over twelve hours is twelve and a half million
/// of them. The size of this struct is therefore multiplied by twelve
/// and a half million, and it was **40 bytes** — 498 MB of index for a
/// film.
///
/// Three of those fields were far wider than anything they can hold. A
/// `TrackNumber` past sixty-five thousand is not a file this project
/// will ever open; a `bool` in its own word costs eight bytes to say
/// one bit. Narrowing them, and letting the keyframe flag share the
/// track's word, takes the struct to **32 bytes** — measured, by the
/// same instrument, not asserted.
///
/// The coded bytes stay a borrow. Copying them would multiply the
/// *file*, not the index, and the file is mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    /// The coded bytes.
    ///
    /// First, because a slice is two words and putting it first lets
    /// the rest pack against it without padding.
    pub data: &'a [u8],
    /// Presentation timestamp in segment ticks: the cluster's timestamp
    /// plus the block's signed offset.
    pub timestamp: i64,
    /// `TrackNumber` this frame belongs to.
    ///
    /// Sixteen bits, not sixty-four. Matroska allows more; this reader
    /// refuses a file that uses them rather than spending six bytes per
    /// frame on a case that does not exist — and says so, because a
    /// silent truncation would put every frame on the wrong track.
    pub track: u16,
    /// Whether the frame can be decoded without any other.
    pub keyframe: bool,
}

/// The largest `TrackNumber` this reader will carry.
///
/// A file naming a higher one is refused rather than truncated: the
/// number is what every frame is filed under, and getting it wrong is
/// not a small error.
pub const MAX_TRACK_NUMBER: u64 = u16::MAX as u64;

/// Why a block could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// The EBML framing around the block was malformed.
    Ebml(EbmlError),
    /// The block ended before its header did.
    ShortHeader,
    /// A lacing size list did not fit the bytes actually present.
    BadLacing,
    /// A `TrackNumber` past what the frame index carries.
    ///
    /// Matroska allows more; this reader files every frame under a
    /// sixteen-bit number so that a twelve-hour film's index fits in a
    /// third of a gigabyte rather than half of one. A file that uses a
    /// larger number is refused by name — truncating it would scatter a
    /// track's frames across another.
    TrackTooLarge(u64),
    /// Fixed lacing whose frame count does not divide the payload.
    UnevenFixedLacing,
}

impl std::fmt::Display for BlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockError::Ebml(e) => write!(f, "ebml: {e}"),
            BlockError::ShortHeader => f.write_str("block ended inside its header"),
            BlockError::BadLacing => f.write_str("lacing sizes do not fit the block"),
            BlockError::UnevenFixedLacing => {
                f.write_str("fixed lacing frame count does not divide the payload")
            }
            BlockError::TrackTooLarge(n) => {
                write!(
                    f,
                    "track number {n} is past the {MAX_TRACK_NUMBER} this reads"
                )
            }
        }
    }
}

impl std::error::Error for BlockError {}

/// Read the unsigned VINT a block header starts with (the track number).
fn track_vint(data: &[u8]) -> Result<(u64, usize), BlockError> {
    let first = *data.first().ok_or(BlockError::ShortHeader)?;
    if first == 0 {
        return Err(BlockError::ShortHeader);
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return Err(BlockError::ShortHeader);
    }
    let bytes = data.get(..len).ok_or(BlockError::ShortHeader)?;
    let mut raw = 0u64;
    for b in bytes {
        raw = (raw << 8) | u64::from(*b);
    }
    Ok((raw & ((1u64 << (len * 7)) - 1), len))
}

/// Read one EBML-lacing size: unsigned for the first, signed after.
///
/// The signed form is the unsigned value less `2^(7n-1) - 1`, where `n`
/// is the VINT's octet count — the offset that recentres an unsigned
/// range on zero so a size difference can go either way.
fn lace_vint(data: &[u8], signed: bool) -> Result<(i64, usize), BlockError> {
    let first = *data.first().ok_or(BlockError::BadLacing)?;
    if first == 0 {
        return Err(BlockError::BadLacing);
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return Err(BlockError::BadLacing);
    }
    let bytes = data.get(..len).ok_or(BlockError::BadLacing)?;
    let mut raw = 0u64;
    for b in bytes {
        raw = (raw << 8) | u64::from(*b);
    }
    let value = (raw & ((1u64 << (len * 7)) - 1)) as i64;
    if signed {
        Ok((value - ((1i64 << (len * 7 - 1)) - 1), len))
    } else {
        Ok((value, len))
    }
}

/// Split a block's payload into its frames.
///
/// `out` receives one slice per frame. Returns [`BlockError`] rather
/// than a partial answer whenever the declared sizes and the present
/// bytes disagree.
fn split_frames<'a>(body: &'a [u8], lacing: u8, out: &mut Vec<&'a [u8]>) -> Result<(), BlockError> {
    if lacing == LACING_NONE {
        out.push(body);
        return Ok(());
    }
    // "Frames minus one", so a lone octet of 0 means one frame.
    let count = usize::from(*body.first().ok_or(BlockError::ShortHeader)?) + 1;
    let rest = &body[1..];

    if lacing == LACING_FIXED {
        // Nothing is stored: every frame is the same size, so the
        // payload must divide exactly. A remainder means the block is
        // not what it claims, and guessing would hand the decoder a
        // frame cut in half.
        if count == 0 || !rest.len().is_multiple_of(count) {
            return Err(BlockError::UnevenFixedLacing);
        }
        let each = rest.len() / count;
        for i in 0..count {
            out.push(&rest[i * each..(i + 1) * each]);
        }
        return Ok(());
    }

    // Xiph and EBML both store count-1 sizes; the last frame takes what
    // is left. The list lives on the stack — the format caps the count.
    let mut sizes = [0usize; MAX_FRAMES_PER_BLOCK];
    let mut at = 0usize;
    let mut previous = 0i64;
    for (index, size) in sizes.iter_mut().take(count - 1).enumerate() {
        let n = match lacing {
            LACING_XIPH => {
                // 255s accumulate; the first octet below 255 ends the
                // run. A run that never ends is a truncated block.
                let mut total = 0usize;
                loop {
                    let b = *rest.get(at).ok_or(BlockError::BadLacing)?;
                    at += 1;
                    total += usize::from(b);
                    if b != 0xFF {
                        break total;
                    }
                }
            }
            LACING_EBML => {
                // Only the first entry is unsigned. Keying this off the
                // running size instead would misread every entry after
                // a zero-length frame.
                let signed = index > 0;
                let (v, len) = lace_vint(rest.get(at..).ok_or(BlockError::BadLacing)?, signed)?;
                at += len;
                // The first entry is the size itself; each later one is
                // a difference from the frame before it.
                let this = if signed { previous + v } else { v };
                if this < 0 {
                    return Err(BlockError::BadLacing);
                }
                previous = this;
                this as usize
            }
            _ => return Err(BlockError::BadLacing),
        };
        *size = n;
    }

    let payload = rest.get(at..).ok_or(BlockError::BadLacing)?;
    let mut cursor = 0usize;
    for size in sizes.iter().take(count - 1) {
        let end = cursor.checked_add(*size).ok_or(BlockError::BadLacing)?;
        out.push(payload.get(cursor..end).ok_or(BlockError::BadLacing)?);
        cursor = end;
    }
    out.push(payload.get(cursor..).ok_or(BlockError::BadLacing)?);
    Ok(())
}

/// Read one `SimpleBlock` or `Block` into `out`.
///
/// `keyframe_default` is what a plain `Block` reports, since only a
/// `SimpleBlock` carries the flag; a `BlockGroup` says the same thing by
/// leaving out `ReferenceBlock`.
fn read_block<'a>(
    payload: &'a [u8],
    cluster_timestamp: i64,
    simple: bool,
    keyframe_default: bool,
    out: &mut Vec<Frame<'a>>,
) -> Result<(), BlockError> {
    let (track, track_len) = track_vint(payload)?;
    // Refused rather than truncated: the track number is what every
    // frame is filed under, and a silent wrap would put a whole track's
    // frames on somebody else's.
    let track = u16::try_from(track).map_err(|_| BlockError::TrackTooLarge(track))?;
    let rest = payload.get(track_len..).ok_or(BlockError::ShortHeader)?;
    let ts_bytes = rest.get(..2).ok_or(BlockError::ShortHeader)?;
    // Signed, and genuinely negative in files with B-frames: a block may
    // present before the cluster it is stored in.
    let relative = i16::from_be_bytes([ts_bytes[0], ts_bytes[1]]);
    let flags = *rest.get(2).ok_or(BlockError::ShortHeader)?;
    let body = rest.get(3..).ok_or(BlockError::ShortHeader)?;

    let keyframe = if simple {
        flags & FLAG_KEYFRAME != 0
    } else {
        keyframe_default
    };
    let timestamp = cluster_timestamp.saturating_add(i64::from(relative));

    let mut frames = Vec::new();
    split_frames(body, flags & FLAG_LACING_MASK, &mut frames)?;
    for data in frames {
        out.push(Frame {
            track,
            timestamp,
            keyframe,
            data,
        });
    }
    Ok(())
}

/// Every frame in a segment, in stored order.
///
/// Frames are collected cluster by cluster. A malformed cluster ends the
/// walk with its error, so a caller sees the frames that were readable
/// and is told the rest was not — the same contract the track reader
/// honours.
///
/// # Errors
///
/// [`BlockError`] naming what did not add up.
pub fn read_frames(segment_payload: &[u8]) -> (Vec<Frame<'_>>, Option<BlockError>) {
    let mut out = Vec::new();
    for element in ebml::elements(segment_payload) {
        let element = match element {
            Ok(e) => e,
            Err(e) => return (out, Some(BlockError::Ebml(e))),
        };
        if element.id != CLUSTER {
            continue;
        }
        let cluster_timestamp = ebml::find(element.payload, CLUSTER_TIMESTAMP)
            .map(|e| ebml::uint(e.payload) as i64)
            .unwrap_or(0);
        for child in ebml::elements(element.payload) {
            let child = match child {
                Ok(c) => c,
                Err(e) => return (out, Some(BlockError::Ebml(e))),
            };
            let result = match child.id {
                SIMPLE_BLOCK => read_block(child.payload, cluster_timestamp, true, false, &mut out),
                BLOCK_GROUP => {
                    // A Block is a keyframe exactly when its group names
                    // no reference.
                    let keyframe = ebml::find(child.payload, REFERENCE_BLOCK).is_none();
                    match ebml::find(child.payload, BLOCK) {
                        Some(b) => {
                            read_block(b.payload, cluster_timestamp, false, keyframe, &mut out)
                        }
                        None => Ok(()),
                    }
                }
                _ => Ok(()),
            };
            if let Err(e) = result {
                return (out, Some(e));
            }
        }
    }
    (out, None)
}

/// Frames of one track, in stored order.
pub fn track_frames<'a>(
    frames: &'a [Frame<'a>],
    track: u64,
) -> impl Iterator<Item = &'a Frame<'a>> {
    frames.iter().filter(move |f| u64::from(f.track) == track)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        let n = payload.len();
        if n < 0x7F {
            out.push(0x80 | n as u8);
        } else {
            out.push(0x40 | (n >> 8) as u8);
            out.push((n & 0xFF) as u8);
        }
        out.extend_from_slice(payload);
        out
    }

    /// A block header for track 1 at the given offset, with `flags`.
    fn header(track: u8, relative: i16, flags: u8) -> Vec<u8> {
        let mut h = vec![0x80 | track];
        h.extend_from_slice(&relative.to_be_bytes());
        h.push(flags);
        h
    }

    fn cluster(timestamp: u8, blocks: &[Vec<u8>]) -> Vec<u8> {
        let mut c = el(&[0xE7], &[timestamp]);
        for b in blocks {
            c.extend_from_slice(b);
        }
        el(&[0x1F, 0x43, 0xB6, 0x75], &c)
    }

    #[test]
    fn an_unlaced_block_yields_one_frame_with_absolute_timing() {
        let mut b = header(1, 40, FLAG_KEYFRAME);
        b.extend_from_slice(b"frame");
        let seg = cluster(100, &[el(&[0xA3], &b)]);

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].track, 1);
        assert_eq!(frames[0].data, b"frame");
        assert!(frames[0].keyframe);
        assert_eq!(frames[0].timestamp, 140, "cluster 100 + offset 40");
    }

    #[test]
    fn a_negative_offset_is_signed_not_huge() {
        // B-frames genuinely present before the cluster they sit in.
        // Reading the field as unsigned would put them 65 000 ticks late.
        let mut b = header(1, -25, FLAG_KEYFRAME);
        b.extend_from_slice(b"x");
        let seg = cluster(100, &[el(&[0xA3], &b)]);
        let (frames, _) = read_frames(&seg);
        assert_eq!(frames[0].timestamp, 75);
    }

    #[test]
    fn xiph_lacing_accumulates_runs_of_255() {
        // Three frames of 300, 1 and 2 bytes. 300 encodes as 255 + 45.
        let mut b = header(1, 0, LACING_XIPH);
        b.push(2); // three frames, minus one
        b.extend_from_slice(&[0xFF, 45]); // first size: 300
        b.push(1); // second size: 1
        b.extend(std::iter::repeat_n(b'a', 300));
        b.push(b'b');
        b.extend_from_slice(b"cd");
        let seg = cluster(0, &[el(&[0xA3], &b)]);

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].data.len(), 300);
        assert_eq!(frames[1].data, b"b");
        assert_eq!(frames[2].data, b"cd", "the last frame takes the remainder");
    }

    #[test]
    fn fixed_lacing_divides_the_payload_evenly() {
        let mut b = header(1, 0, LACING_FIXED);
        b.push(3); // four frames
        b.extend_from_slice(b"aabbccdd");
        let seg = cluster(0, &[el(&[0xA3], &b)]);

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0].data, b"aa");
        assert_eq!(frames[3].data, b"dd");
    }

    #[test]
    fn fixed_lacing_that_does_not_divide_is_refused() {
        // Handing a decoder a frame cut in half is worse than refusing.
        let mut b = header(1, 0, LACING_FIXED);
        b.push(2); // three frames
        b.extend_from_slice(b"abcdefg"); // 7 does not divide by 3
        let seg = cluster(0, &[el(&[0xA3], &b)]);
        let (_, err) = read_frames(&seg);
        assert_eq!(err, Some(BlockError::UnevenFixedLacing));
    }

    #[test]
    fn ebml_lacing_reads_the_first_size_then_signed_differences() {
        // Sizes 5, 3, 4: the first plain, then -2, then +1. The signed
        // form subtracts 2^(7n-1)-1, so for a one-octet VINT that is 63:
        // -2 is stored as 61 (0x80|61), +1 as 64 (0x80|64).
        let mut b = header(1, 0, LACING_EBML);
        b.push(2); // three frames
        b.push(0x80 | 5); // first size, unsigned
        b.push(0x80 | 61); // 61 - 63 = -2  -> 3
        b.extend_from_slice(b"aaaaa");
        b.extend_from_slice(b"bbb");
        b.extend_from_slice(b"cccc");
        let seg = cluster(0, &[el(&[0xA3], &b)]);

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].data, b"aaaaa");
        assert_eq!(frames[1].data, b"bbb", "5 + (-2) = 3");
        assert_eq!(frames[2].data, b"cccc", "the remainder");
    }

    #[test]
    fn ebml_lacing_after_a_zero_length_frame_is_still_signed() {
        // Sizes 0, 2, 3: the second entry is a difference of +2, stored
        // as 63 + 2 = 65. Read as unsigned it would be 65 bytes and the
        // block would be refused.
        let mut b = header(1, 0, LACING_EBML);
        b.push(2); // three frames
        b.push(0x80); // first size: 0
        b.push(0x80 | 65); // +2 -> 2
        b.extend_from_slice(b"bb");
        b.extend_from_slice(b"ccc");
        let seg = cluster(0, &[el(&[0xA3], &b)]);

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].data, b"");
        assert_eq!(frames[1].data, b"bb");
        assert_eq!(frames[2].data, b"ccc");
    }

    #[test]
    fn a_lacing_list_longer_than_the_block_is_refused() {
        // Negative trust: the sizes are the writer's claim about bytes
        // that may simply not be there.
        let mut b = header(1, 0, LACING_XIPH);
        b.push(1); // two frames
        b.push(200); // first frame claims 200 bytes
        b.extend_from_slice(b"short");
        let seg = cluster(0, &[el(&[0xA3], &b)]);
        let (_, err) = read_frames(&seg);
        assert_eq!(err, Some(BlockError::BadLacing));
    }

    #[test]
    fn a_xiph_run_that_never_ends_is_refused_rather_than_read_past() {
        let mut b = header(1, 0, LACING_XIPH);
        b.push(1);
        b.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // never terminates
        let seg = cluster(0, &[el(&[0xA3], &b)]);
        let (_, err) = read_frames(&seg);
        assert_eq!(err, Some(BlockError::BadLacing));
    }

    #[test]
    fn a_block_group_takes_its_keyframe_flag_from_the_absence_of_a_reference() {
        // A plain Block has no keyframe bit; the group says it by NOT
        // carrying a ReferenceBlock. Reading the flags byte here would
        // mark ordinary P-frames as seekable.
        let mut inner = header(1, 0, 0);
        inner.extend_from_slice(b"key");
        let group_key = el(&[0xA0], &el(&[0xA1], &inner));

        let mut inner = header(1, 10, 0);
        inner.extend_from_slice(b"dep");
        let mut g = el(&[0xA1], &inner);
        g.extend(el(&[0xFB], &[0x01])); // ReferenceBlock present
        let group_dep = el(&[0xA0], &g);

        let seg = cluster(0, &[group_key, group_dep]);
        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        assert_eq!(frames.len(), 2);
        assert!(frames[0].keyframe, "no reference means keyframe");
        assert!(!frames[1].keyframe, "a reference means it depends");
    }

    #[test]
    fn frames_of_several_tracks_interleave_and_can_be_selected() {
        let mut a = header(1, 0, FLAG_KEYFRAME);
        a.extend_from_slice(b"v");
        let mut b = header(2, 0, FLAG_KEYFRAME);
        b.extend_from_slice(b"a1");
        let mut c = header(2, 20, FLAG_KEYFRAME);
        c.extend_from_slice(b"a2");
        let seg = cluster(0, &[el(&[0xA3], &a), el(&[0xA3], &b), el(&[0xA3], &c)]);

        let (frames, _) = read_frames(&seg);
        assert_eq!(frames.len(), 3);
        let audio: Vec<&[u8]> = track_frames(&frames, 2).map(|f| f.data).collect();
        assert_eq!(audio, vec![b"a1".as_slice(), b"a2".as_slice()]);
        assert_eq!(track_frames(&frames, 1).count(), 1);
        assert_eq!(track_frames(&frames, 9).count(), 0);
    }

    #[test]
    fn several_clusters_carry_their_own_timestamps() {
        let mut a = header(1, 5, FLAG_KEYFRAME);
        a.extend_from_slice(b"x");
        let mut b = header(1, 5, FLAG_KEYFRAME);
        b.extend_from_slice(b"y");
        let mut seg = cluster(0, &[el(&[0xA3], &a)]);
        seg.extend(cluster(100, &[el(&[0xA3], &b)]));

        let (frames, err) = read_frames(&seg);
        assert_eq!(err, None);
        let stamps: Vec<i64> = frames.iter().map(|f| f.timestamp).collect();
        assert_eq!(stamps, vec![5, 105]);
    }

    #[test]
    fn a_truncated_block_reports_instead_of_yielding_half_a_frame() {
        let seg = cluster(0, &[el(&[0xA3], &[0x81, 0x00])]); // header cut short
        let (frames, err) = read_frames(&seg);
        assert!(frames.is_empty());
        assert_eq!(err, Some(BlockError::ShortHeader));
    }

    #[test]
    fn readable_frames_survive_a_failure_later_in_the_file() {
        // Graceful degradation: what was readable comes back, and the
        // damage is named rather than swallowed.
        let mut good = header(1, 0, FLAG_KEYFRAME);
        good.extend_from_slice(b"ok");
        let mut seg = cluster(0, &[el(&[0xA3], &good)]);
        seg.extend(cluster(10, &[el(&[0xA3], &[0x81])]));

        let (frames, err) = read_frames(&seg);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, b"ok");
        assert_eq!(err, Some(BlockError::ShortHeader));
    }
}
