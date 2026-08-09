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

//! # sovmr-native — the fully sovereign zero-copy pipeline
//!
//! `sovmr-mp4` demuxes Annex B → `sovmr-nvdec` decodes into VRAM →
//! the `CUdeviceptr` is registered/mapped straight into the
//! `sovmr-nvenc` session → `nvEncLockBitstream` drains compressed
//! AV1/HEVC back to the host. Pixels never touch system RAM; no FFmpeg
//! anywhere in the path.
//!
//! [`provider::NvencNativeProvider`] wraps [`pipeline::transcode`] behind
//! `sovmr_core::encoder::EncoderProvider`, so the whole engine swap is
//! one line in the composition root.
//!
//! AV1 targets are muxed on the fly into a **fast-start MP4** (`moov`
//! before `mdat`) with the source audio passed through bit-exactly,
//! interleaved by timestamp — the file streams over the web natively.
//! HEVC targets still emit raw Annex B until the muxer grows `hvc1`.

#![deny(missing_docs)]

pub mod audio;
pub mod pipeline;
pub mod provider;
mod source;
pub mod verify;

pub use pipeline::{transcode, NativeError, NativeSettings, TranscodeOutcome, TranscodeStats};
pub use provider::NvencNativeProvider;
