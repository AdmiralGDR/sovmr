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

//! # sovmr-audio — the sovereign audio codecs
//!
//! Audio is where a compressor can still win honestly. A track already
//! in AAC or Opus is carried through bit-exactly, because re-encoding it
//! would trade a real generation of quality for a speculative saving.
//! But *uncompressed* audio — PCM, and the lossless FLAC that stands in
//! for it — is a different case entirely: the source is mathematically
//! exact, so compressing it costs no generation at all and saves five to
//! ten times its size.
//!
//! That is the whole remit of this crate: decode what is lossless, and
//! encode it into something small.
//!
//! * [`pcm`] — uncompressed samples, in the several shapes containers
//!   store them;
//! * [`flac`] — lossless compression, and therefore exact or broken;
//! * [`bits`] — reading fields that do not fall on byte boundaries;
//! * [`mdct`] — the transform lossy coding is built on;
//! * [`fft`] — what makes that transform affordable;
//! * [`quant`] — where the bits are actually thrown away;
//! * [`asc`] — the two bytes that say what an AAC stream is.
//!
//! Zero dependencies, like every other crate here.

#![deny(missing_docs)]

pub mod asc;
pub mod bands;
pub mod bits;
pub mod codebooks;
pub mod decode;
pub mod decode_tables;
pub mod encode;
pub mod fft;
pub mod flac;
pub mod frame;
pub mod mdct;
pub mod opus;
pub mod pcm;
pub mod quant;
