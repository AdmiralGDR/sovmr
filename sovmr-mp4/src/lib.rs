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

//! # sovmr-mp4 — strict zero-allocation ISOBMFF box parser
//!
//! The sovereign answer to "NVDEC doesn't eat `.mp4`": walk the container,
//! find the sample tables and the media payload, and hand raw video-track
//! bytes to `sovmr-nvdec`. FFmpeg is banished, so this parser trusts
//! nothing: every size field is bounds-checked with
//! `u64`/`usize` checked arithmetic, malformed input yields a typed
//! [`ParseError`] — never a panic, never an over-read.
//!
//! Zero-allocation: parsing operates on `&[u8]` and yields sub-slices of
//! the same buffer. The only heap use anywhere is inside your own choice
//! of how to load the file (mmap, read, chunk).
//!
//! Current scaffold scope:
//! * [`boxes`] — iterator over top-level or child boxes (32-bit,
//!   64-bit `largesize`, and `size == 0` "to end of buffer" forms).
//! * [`Mp4Box::children`] — descend into container boxes.
//! * [`find_path`] — locate e.g. `moov.trak.mdia` payloads.
//! * [`file_kind`] — cheap ftyp sanity check.
//!
//! Next milestone (the demux proper): parse `stsd` (avc1/hev1 →
//! `avcC`/`hvcC` decoder-config records), `stsz`/`stco`/`stsc` sample
//! tables, and repacketize length-prefixed samples into the Annex B
//! start-code stream NVDEC's parser expects.
//!
//! # Example
//!
//! ```
//! use sovmr_mp4::{boxes, find_path};
//!
//! // ftyp("isom") + moov( mvhd(empty) ) built by hand:
//! let mut file = Vec::new();
//! file.extend_from_slice(&[0, 0, 0, 12]); file.extend_from_slice(b"ftypisom");
//! file.extend_from_slice(&[0, 0, 0, 16]); file.extend_from_slice(b"moov");
//! file.extend_from_slice(&[0, 0, 0, 8]);  file.extend_from_slice(b"mvhd");
//!
//! let kinds: Vec<[u8; 4]> = boxes(&file).map(|b| b.unwrap().kind).collect();
//! assert_eq!(kinds, vec![*b"ftyp", *b"moov"]);
//! assert_eq!(find_path(&file, &[*b"moov", *b"mvhd"]).unwrap(), b"");
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod colour;
pub mod demux;
pub mod hevc;
pub mod mux;

/// Shortest legal box: 4-byte size + 4-byte type.
const HEADER_LEN: usize = 8;
/// Header length when `size == 1` (64-bit `largesize` follows the type).
const LARGE_HEADER_LEN: usize = 16;
/// `size` sentinel: box extends to the end of the enclosing buffer.
const SIZE_TO_END: u32 = 0;
/// `size` sentinel: 64-bit `largesize` field follows the box type.
const SIZE_IS_LARGE: u32 = 1;

/// Container boxes worth descending into on the way to the sample tables.
/// (`meta` is excluded on purpose: it is a *full* box whose payload starts
/// with a version/flags word, handled at the demux milestone.)
pub const CONTAINER_KINDS: [[u8; 4]; 6] =
    [*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"edts"];

/// Why a byte range refused to parse as a box. Offsets are relative to
/// the buffer handed to the iterator, so callers can point at the exact
/// corruption site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Fewer than 8 bytes left where a box header must start.
    TruncatedHeader {
        /// Offset of the incomplete header.
        at: usize,
    },
    /// The header's declared size runs past the end of the buffer.
    TruncatedBox {
        /// Offset of the box whose payload is cut short.
        at: usize,
        /// Bytes the header claims.
        declared: u64,
        /// Bytes actually available from `at`.
        available: u64,
    },
    /// Declared size is smaller than its own header (2..=7, or a
    /// `largesize` below 16) — structurally impossible.
    ImpossibleSize {
        /// Offset of the offending box.
        at: usize,
        /// The impossible declared size.
        declared: u64,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::TruncatedHeader { at } => {
                write!(f, "truncated box header at offset {at}")
            }
            ParseError::TruncatedBox {
                at,
                declared,
                available,
            } => write!(
                f,
                "box at offset {at} declares {declared} bytes but only {available} remain"
            ),
            ParseError::ImpossibleSize { at, declared } => {
                write!(f, "box at offset {at} declares impossible size {declared}")
            }
        }
    }
}

/// One parsed box: its four-byte type and its payload sub-slice
/// (everything after the header, zero copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp4Box<'a> {
    /// Four-character box type, e.g. `*b"moov"`.
    pub kind: [u8; 4],
    /// Offset of this box's header inside the parsed buffer.
    pub offset: usize,
    /// The box payload (header excluded).
    pub payload: &'a [u8],
}

impl<'a> Mp4Box<'a> {
    /// Iterate this box's children. Meaningful for the well-known
    /// container kinds ([`CONTAINER_KINDS`]); calling it on a leaf just
    /// yields that leaf's bytes reinterpreted (and typically a
    /// [`ParseError`]), so check `kind` first when walking blindly.
    pub fn children(&self) -> BoxIter<'a> {
        boxes(self.payload)
    }
}

/// Iterator over consecutive boxes in `data`. Yields `Err` once at the
/// first malformed header and then stops — a hostile tail can not spin
/// the loop.
pub fn boxes(data: &[u8]) -> BoxIter<'_> {
    BoxIter {
        data,
        pos: 0,
        poisoned: false,
    }
}

/// See [`boxes`].
#[derive(Debug, Clone)]
pub struct BoxIter<'a> {
    data: &'a [u8],
    pos: usize,
    poisoned: bool,
}

impl<'a> Iterator for BoxIter<'a> {
    type Item = Result<Mp4Box<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.poisoned || self.pos >= self.data.len() {
            return None;
        }
        match parse_box_at(self.data, self.pos) {
            Ok((mp4box, next_pos)) => {
                self.pos = next_pos;
                Some(Ok(mp4box))
            }
            Err(e) => {
                self.poisoned = true;
                Some(Err(e))
            }
        }
    }
}

/// Parse one box header at `at`; return the box and the offset of the
/// next sibling. All arithmetic is checked — a 2^63 size field cannot
/// wrap into a bogus "valid" range.
fn parse_box_at(data: &[u8], at: usize) -> Result<(Mp4Box<'_>, usize), ParseError> {
    let remaining = &data[at..];
    if remaining.len() < HEADER_LEN {
        return Err(ParseError::TruncatedHeader { at });
    }
    let size32 = u32::from_be_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
    let kind = [remaining[4], remaining[5], remaining[6], remaining[7]];

    let (declared, header_len): (u64, usize) = match size32 {
        SIZE_TO_END => (remaining.len() as u64, HEADER_LEN),
        SIZE_IS_LARGE => {
            if remaining.len() < LARGE_HEADER_LEN {
                return Err(ParseError::TruncatedHeader { at });
            }
            let large = u64::from_be_bytes([
                remaining[8],
                remaining[9],
                remaining[10],
                remaining[11],
                remaining[12],
                remaining[13],
                remaining[14],
                remaining[15],
            ]);
            (large, LARGE_HEADER_LEN)
        }
        _ => (u64::from(size32), HEADER_LEN),
    };

    if declared < header_len as u64 {
        return Err(ParseError::ImpossibleSize { at, declared });
    }
    if declared > remaining.len() as u64 {
        return Err(ParseError::TruncatedBox {
            at,
            declared,
            available: remaining.len() as u64,
        });
    }
    // In-bounds by the checks above; usize cast is safe because declared
    // <= remaining.len() which is a usize.
    let end = declared as usize;
    let payload = &remaining[header_len..end];
    Ok((
        Mp4Box {
            kind,
            offset: at,
            payload,
        },
        at + end,
    ))
}

/// Walk a path of nested box types (e.g. `[moov, trak, mdia]`) and return
/// the payload of the first box matching the full path. Malformed regions
/// terminate that branch of the walk silently — hostile bytes cannot
/// abort an otherwise-valid file, and a missing path is just `None`.
pub fn find_path<'a>(data: &'a [u8], path: &[[u8; 4]]) -> Option<&'a [u8]> {
    let (first, rest) = path.split_first()?;
    for item in boxes(data) {
        let Ok(b) = item else { break };
        if b.kind == *first {
            if rest.is_empty() {
                return Some(b.payload);
            }
            if let Some(found) = find_path(b.payload, rest) {
                return Some(found);
            }
        }
    }
    None
}

/// The major brand from the leading `ftyp` box (`b"isom"`, `b"mp42"`,
/// ...), or `None` when the buffer does not start with a valid `ftyp` —
/// the cheap "is this even MP4" gate before deeper work.
pub fn file_kind(data: &[u8]) -> Option<[u8; 4]> {
    let first = boxes(data).next()?.ok()?;
    if first.kind != *b"ftyp" || first.payload.len() < 4 {
        return None;
    }
    Some([
        first.payload[0],
        first.payload[1],
        first.payload[2],
        first.payload[3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize one box (test helper; production code never builds boxes).
    fn make_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(HEADER_LEN as u32 + payload.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn sample_file() -> Vec<u8> {
        let stbl = make_box(b"stbl", &make_box(b"stsz", &[0xAA; 12]));
        let minf = make_box(b"minf", &stbl);
        let mdia = make_box(b"mdia", &minf);
        let trak = make_box(b"trak", &mdia);
        let moov = make_box(b"moov", &trak);
        let mut file = make_box(b"ftyp", b"isom\0\0\0\0isom");
        file.extend_from_slice(&moov);
        file.extend_from_slice(&make_box(b"mdat", &[0x42; 32]));
        file
    }

    #[test]
    fn walks_top_level_boxes_in_order() {
        let file = sample_file();
        let kinds: Vec<[u8; 4]> = boxes(&file).map(|b| b.unwrap().kind).collect();
        assert_eq!(kinds, vec![*b"ftyp", *b"moov", *b"mdat"]);
    }

    #[test]
    fn find_path_descends_to_the_sample_table() {
        let file = sample_file();
        let stsz = find_path(
            &file,
            &[*b"moov", *b"trak", *b"mdia", *b"minf", *b"stbl", *b"stsz"],
        )
        .expect("full path exists");
        assert_eq!(stsz, &[0xAA; 12]);
        assert!(find_path(&file, &[*b"moov", *b"free"]).is_none());
    }

    #[test]
    fn mdat_payload_is_a_zero_copy_subslice() {
        let file = sample_file();
        let mdat = find_path(&file, &[*b"mdat"]).unwrap();
        assert_eq!(mdat, &[0x42; 32]);
        // Same allocation: the payload points inside the original buffer.
        let file_range = file.as_ptr() as usize..file.as_ptr() as usize + file.len();
        assert!(file_range.contains(&(mdat.as_ptr() as usize)));
    }

    #[test]
    fn largesize_and_to_end_forms_parse() {
        // largesize box: size=1, type, 64-bit size, payload.
        let mut file = Vec::new();
        file.extend_from_slice(&1u32.to_be_bytes());
        file.extend_from_slice(b"mdat");
        file.extend_from_slice(&(LARGE_HEADER_LEN as u64 + 4).to_be_bytes());
        file.extend_from_slice(&[7; 4]);
        // to-end box: size=0.
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(b"free");
        file.extend_from_slice(&[9; 6]);

        let parsed: Vec<Mp4Box> = boxes(&file).map(|b| b.unwrap()).collect();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].payload, &[7; 4]);
        assert_eq!(parsed[1].kind, *b"free");
        assert_eq!(parsed[1].payload, &[9; 6]);
    }

    #[test]
    fn truncated_and_impossible_sizes_are_typed_errors_not_panics() {
        // Declares 100 bytes, provides 20.
        let mut lying = make_box(b"moov", &[0; 12]);
        lying[3] = 100;
        let err = boxes(&lying).next().unwrap().unwrap_err();
        assert!(matches!(
            err,
            ParseError::TruncatedBox { declared: 100, .. }
        ));

        // size = 3: smaller than its own header.
        let mut impossible = Vec::new();
        impossible.extend_from_slice(&3u32.to_be_bytes());
        impossible.extend_from_slice(b"free");
        let err = boxes(&impossible).next().unwrap().unwrap_err();
        assert_eq!(err, ParseError::ImpossibleSize { at: 0, declared: 3 });

        // 5 stray bytes: not even a header.
        let err = boxes(&[0, 0, 0, 9, 0xFF]).next().unwrap().unwrap_err();
        assert_eq!(err, ParseError::TruncatedHeader { at: 0 });
    }

    #[test]
    fn iterator_poisons_after_first_error() {
        let mut file = sample_file();
        file.extend_from_slice(&[0, 0, 0, 9]); // hostile tail: truncated box
        let results: Vec<_> = boxes(&file).collect();
        assert_eq!(results.len(), 4, "3 good boxes, then exactly one error");
        assert!(results[3].is_err());
    }

    #[test]
    fn file_kind_reads_the_ftyp_brand_and_rejects_non_mp4() {
        assert_eq!(file_kind(&sample_file()), Some(*b"isom"));
        assert_eq!(file_kind(b"\x7fELF garbage here padding"), None);
        assert_eq!(file_kind(&[]), None);
    }

    #[test]
    fn giant_size_fields_cannot_overflow() {
        // largesize u64::MAX must not wrap the arithmetic.
        let mut evil = Vec::new();
        evil.extend_from_slice(&1u32.to_be_bytes());
        evil.extend_from_slice(b"mdat");
        evil.extend_from_slice(&u64::MAX.to_be_bytes());
        evil.extend_from_slice(&[0; 16]);
        let err = boxes(&evil).next().unwrap().unwrap_err();
        assert!(matches!(err, ParseError::TruncatedBox { .. }));
    }
}
