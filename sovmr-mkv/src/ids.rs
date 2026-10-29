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

//! Matroska Element IDs, quoted as the specification writes them —
//! marker bit included, so they compare directly against what
//! [`crate::ebml`] reads.
//!
//! Only the elements sovmr actually reads are named. A table of every ID
//! in the format would be a liability: entries nothing consults still
//! have to be kept correct.

#![allow(missing_docs)] // each constant's name is its documentation

// ---- top level --------------------------------------------------------------

pub const SEGMENT: u32 = 0x1853_8067;

// ---- \Segment\Info ----------------------------------------------------------

pub const INFO: u32 = 0x1549_A966;
/// Nanoseconds per timestamp tick; every block timestamp is in these.
pub const TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
/// Total duration, in timestamp ticks, as a float.
pub const DURATION: u32 = 0x4489;
pub const TITLE: u32 = 0x7BA9;

// ---- \Segment\Tracks --------------------------------------------------------

pub const TRACKS: u32 = 0x1654_AE6B;
pub const TRACK_ENTRY: u32 = 0xAE;
pub const TRACK_NUMBER: u32 = 0xD7;
pub const TRACK_UID: u32 = 0x73C5;
pub const TRACK_TYPE: u32 = 0x83;
pub const FLAG_ENABLED: u32 = 0xB9;
pub const FLAG_DEFAULT: u32 = 0x88;
pub const FLAG_FORCED: u32 = 0x55AA;
pub const NAME: u32 = 0x536E;
/// ISO 639-2 language. Defaults to `eng` when absent — a rule of the
/// format, not a guess.
pub const LANGUAGE: u32 = 0x0022_B59C;
/// BCP 47 language, which supersedes [`LANGUAGE`] when both are present.
pub const LANGUAGE_BCP47: u32 = 0x0022_B59D;
pub const CODEC_ID: u32 = 0x86;
/// Codec setup bytes: `avcC`, `hvcC`, `av1C`, or an `AudioSpecificConfig`
/// depending on the codec.
pub const CODEC_PRIVATE: u32 = 0x63A2;
/// Nanoseconds of encoder delay to trim from the start of the track.
pub const CODEC_DELAY: u32 = 0x56AA;
/// Nominal duration of one frame, in nanoseconds.
pub const DEFAULT_DURATION: u32 = 0x0023_E383;

pub const VIDEO: u32 = 0xE0;
pub const PIXEL_WIDTH: u32 = 0xB0;

/// `Colour`: what a video's samples mean, the same code points ISOBMFF
/// puts in a `colr` box. Matroska spells each out as its own element
/// rather than packing them into one payload.
pub const COLOUR: u32 = 0x55B0;
/// `MatrixCoefficients`.
pub const COLOUR_MATRIX: u32 = 0x55B1;
/// `Range`: 0 unspecified, 1 broadcast, 2 full, 3 defined by the matrix.
pub const COLOUR_RANGE: u32 = 0x55B9;
/// `TransferCharacteristics`.
pub const COLOUR_TRANSFER: u32 = 0x55BA;
/// `Primaries`.
pub const COLOUR_PRIMARIES: u32 = 0x55BB;

/// [`COLOUR_RANGE`] for the broadcast 16-235.
pub const RANGE_LIMITED: u64 = 1;
/// [`COLOUR_RANGE`] for the full 0-255.
pub const RANGE_FULL: u64 = 2;
pub const PIXEL_HEIGHT: u32 = 0xBA;

pub const AUDIO: u32 = 0xE1;
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
pub const CHANNELS: u32 = 0x9F;
pub const BIT_DEPTH: u32 = 0x6264;

// ---- track types ------------------------------------------------------------

pub const TYPE_VIDEO: u64 = 1;
pub const TYPE_AUDIO: u64 = 2;
pub const TYPE_SUBTITLE: u64 = 17;

// ---------------------------------------------------------------------
// Chapters and tags.
//
// Every identifier below was read out of a file ffmpeg wrote, by this
// project's own EBML reader — see `examples/walk.rs`, which is the
// instrument. None of them came from memory or from a web page, and
// stage 3 is why: a published answer about EBML lengths was wrong, and
// only the file settled it.
//
// The same probe settled the units. A chapter's start and end are plain
// **nanoseconds**, not segment ticks: the probe's second chapter began
// at 2.0 s and its `ChapterTimeStart` held 2 000 000 000 with a
// `TimestampScale` of a million. Scaling them would have placed every
// mark a thousand times late, and nothing but a decoder would have said
// so.
// ---------------------------------------------------------------------

/// Top-level chapter list.
pub const CHAPTERS: u32 = 0x1043_A770;
/// One edition — an alternative ordering of the same film.
pub const EDITION_ENTRY: u32 = 0x45B9;
/// Whether an edition is the one to use unless told otherwise.
pub const EDITION_FLAG_DEFAULT: u32 = 0x45DB;
/// One chapter.
pub const CHAPTER_ATOM: u32 = 0xB6;
/// A chapter's unique identifier.
pub const CHAPTER_UID: u32 = 0x73C4;
/// Start and end, in nanoseconds.
pub const CHAPTER_TIME_START: u32 = 0x91;
pub const CHAPTER_TIME_END: u32 = 0x92;
/// A chapter's name in one language.
pub const CHAPTER_DISPLAY: u32 = 0x80;
/// The name itself, and the language it is in.
pub const CHAP_STRING: u32 = 0x85;
pub const CHAP_LANGUAGE: u32 = 0x437C;

/// Top-level tag list.
pub const TAGS: u32 = 0x1254_C367;
/// One tag: what it applies to, and the entries themselves.
pub const TAG: u32 = 0x7373;
/// What a tag applies to. Empty means the whole file.
pub const TARGETS: u32 = 0x63C0;
/// A target naming one track by its UID.
pub const TAG_TRACK_UID: u32 = 0x63C5;
/// One name-and-value pair.
pub const SIMPLE_TAG: u32 = 0x67C8;
pub const TAG_NAME: u32 = 0x45A3;
pub const TAG_STRING: u32 = 0x4487;
