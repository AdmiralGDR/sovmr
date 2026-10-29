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

//! Every tunable in one place. No magic numbers anywhere else in the tree.

use std::time::Duration;

/// NVENC constant-quality target for the "Visually Lossless" one-click preset.
/// CQ operates on a 0–51 scale (lower = better); 25 is the project-specified
/// visually-lossless sweet spot for AV1/HEVC on Ada-generation NVENC.
pub const VISUALLY_LOSSLESS_CQ: u8 = 25;

/// Upper bound of the NVENC CQ scale; values above this are invalid input.
pub const CQ_MAX: u8 = 51;

/// Hard ceiling on concurrent NVENC sessions. GeForce drivers >= 551.23
/// (2024-01) allow 8 concurrent sessions on consumer GPUs; older drivers
/// allow fewer, so this is a cap, not a promise.
pub const NVENC_SESSION_CAP: usize = 8;

/// Conservative VRAM budget per concurrent encode session. Covers decoder
/// surfaces + encoder lookahead for 4K content with headroom; used only to
/// derive the *default* worker count, which the user can override.
pub const VRAM_PER_SESSION_MIB: u64 = 1536;

/// Default worker slots when VRAM cannot be queried.
pub const DEFAULT_WORKER_SLOTS: usize = 2;

/// How often an encode session polls its child process for exit/cancel.
/// 100 ms keeps cancellation snappy without measurable CPU cost.
pub const SESSION_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Suffix appended to the final output path while an encode is in flight.
/// A crash or cancel can only ever leave `*.part` litter, never a truncated
/// file masquerading as a finished artifact (rollback).
pub const PARTIAL_SUFFIX: &str = ".part";

/// Suffix inserted before the extension of produced files
/// (`movie.mp4` -> `movie.sovmr.mp4`) so a batch never overwrites its input.
pub const OUTPUT_TAG: &str = "sovmr";

/// How many bytes of a file we sniff to validate its container signature.
pub const SNIFF_LEN: usize = 16;

/// Smart-Skip threshold: average bits per pixel per second below which a
/// file that is *already* in the target codec counts as "already highly
/// compressed". Rationale: visually-lossless NVENC AV1/HEVC at
/// [`VISUALLY_LOSSLESS_CQ`] typically lands between 0.5 and 1.5 b/(px·s)
/// for real-world content, so re-encoding anything at or below this
/// density can only burn GPU cycles and stack generation loss — it cannot
/// meaningfully shrink the file.
pub const SKIP_MAX_BITS_PER_PIXEL_SECOND: f64 = 1.5;

/// Cross-codec Smart Skip floor in bits/(pixel·second). Below this
/// density the source is already leaner than the encoder's own output
/// ceiling ([`MAX bitrate` clamp ≈ 0.08 b/(px·s)] and typical VBR-CQ
/// results), so a re-encode cannot shrink it — regardless of codec —
/// and would only stack generation loss. Deliberately conservative:
/// half the encoder's bitrate cap.
pub const SKIP_FLOOR_BITS_PER_PIXEL_SECOND: f64 = 0.04;

/// Derive the default number of parallel encode workers from available VRAM,
/// clamped to the driver session cap. Never returns 0.
///
/// # Example
///
/// ```
/// use sovmr_core::config::default_worker_slots;
/// assert_eq!(default_worker_slots(Some(8188)), 5);  // RTX 4070 Mobile
/// assert_eq!(default_worker_slots(Some(512)), 1);   // tiny VRAM -> still 1
/// assert_eq!(default_worker_slots(None), 2);        // unknown -> safe default
/// ```
pub fn default_worker_slots(vram_mib: Option<u64>) -> usize {
    match vram_mib {
        Some(mib) => {
            let by_vram = (mib / VRAM_PER_SESSION_MIB) as usize;
            by_vram.clamp(1, NVENC_SESSION_CAP)
        }
        None => DEFAULT_WORKER_SLOTS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_slots_scale_with_vram() {
        assert_eq!(default_worker_slots(Some(8188)), 5); // RTX 4070 Mobile
        assert_eq!(default_worker_slots(Some(24_576)), NVENC_SESSION_CAP);
    }

    #[test]
    fn worker_slots_never_zero() {
        assert_eq!(default_worker_slots(Some(0)), 1);
        assert_eq!(default_worker_slots(Some(512)), 1);
    }

    #[test]
    fn worker_slots_fallback_without_vram_info() {
        assert_eq!(default_worker_slots(None), DEFAULT_WORKER_SLOTS);
    }
}
