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

//! # sovmr-nvenc — Phase-2 native NVENC backend (skeleton)
//!
//! Direct bridge to the NVIDIA Video Codec runtime, bypassing FFmpeg for
//! standard video tracks. Absolute sovereignty: **zero
//! dependencies, not even `libc`** — the `dlopen` family is declared by
//! hand ([`dl`]) and `libnvidia-encode.so.1` is loaded at runtime from
//! the driver stack that is guaranteed present on any machine with an
//! RTX GPU. Nothing links at build time, so this crate builds anywhere.
//!
//! Current scope:
//! * [`api::NvEncApi`] — loads the driver library (via `sovmr-os`),
//!   queries `NvEncodeAPIGetMaxSupportedVersion`, and calls
//!   `NvEncodeAPICreateInstance` to obtain the versioned function table.
//! * [`api::NvEncSession`] — a real hardware encode session opened with
//!   `nvEncOpenEncodeSessionEx` on a sovereign `sovmr-cuda` context;
//!   RAII teardown in reverse order (session, then context).
//!
//! `sovmr-native` drives this crate's frame loop
//! (`nvEncInitializeEncoder`, `nvEncEncodePicture`, `nvEncLockBitstream`)
//! behind `sovmr_core::encoder::EncoderProvider` — the sole production
//! engine since the FFmpeg backend was purged.
//!
//! # Example
//!
//! ```no_run
//! use sovmr_nvenc::NvEncApi;
//!
//! match NvEncApi::load() {
//!     Ok(api) => println!(
//!         "NVENC driver API {}.{}, {} entry points",
//!         api.driver_version().0,
//!         api.driver_version().1,
//!         api.entry_point_count(),
//!     ),
//!     Err(e) => eprintln!("no NVENC runtime here: {e}"),
//! }
//! ```

#![deny(missing_docs)]

pub mod api;
pub mod encoder;

pub use api::{
    status_name, EncodeSupport, MappedInput, NvEncApi, NvEncSession, NvencError,
    RegisteredResource, VramFrame,
};
pub use encoder::{BitstreamBuffer, BitstreamLock, EncodeCodec, EncodePreset, EncoderInit};
