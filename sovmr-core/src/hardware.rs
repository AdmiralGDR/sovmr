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

//! Hardware configurator: discovers the NVIDIA GPU, intersects what the GPU
//! generation supports with what the encoder backend reports it can do, and
//! produces a ready-to-use encode plan (codec + CQ + worker slots).
//!
//! The decision is defensive on both axes: a GPU name alone never enables a
//! codec (the backend must confirm it too), and a missing `nvidia-smi`
//! degrades to a clear error instead of a guess.

use std::process::Command;

use crate::config::default_worker_slots;
use crate::encoder::{EncoderCapabilities, EncoderProvider};
use crate::job::{Codec, EncodeSettings};

/// `nvidia-smi` invocation that yields one CSV line per GPU:
/// `<name>, <vram MiB>, <driver>`.
const NVIDIA_SMI: &str = "nvidia-smi";
const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
];

/// Facts about the detected GPU, as reported by the driver.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    /// Marketing name, e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`.
    pub name: String,
    /// Total VRAM in MiB; `None` if the driver would not report it.
    pub vram_mib: Option<u64>,
    /// Driver version string, e.g. `"580.159.03"`.
    pub driver_version: String,
}

/// Why no encode plan could be produced for this machine.
#[derive(Debug)]
pub enum HardwareError {
    /// `nvidia-smi` missing or failed — no NVIDIA driver stack.
    NoNvidiaDriver(String),
    /// Driver present, but backend reports no usable NVENC codec.
    NoUsableEncoder(String),
}

impl std::fmt::Display for HardwareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardwareError::NoNvidiaDriver(m) => write!(f, "no NVIDIA driver: {m}"),
            HardwareError::NoUsableEncoder(m) => write!(f, "no usable encoder: {m}"),
        }
    }
}

/// The configurator's output: everything the queue needs to start encoding
/// with sensible defaults. All fields are user-overridable in the UI.
#[derive(Debug, Clone)]
pub struct EncodePlan {
    /// The GPU the plan was derived for.
    pub gpu: GpuInfo,
    /// Recommended one-click settings (visually lossless, best codec).
    pub settings: EncodeSettings,
    /// Recommended number of concurrent encode sessions (VRAM-derived,
    /// clamped to the driver session cap).
    pub worker_slots: usize,
}

/// Stateless namespace for the two detection entry points: [`detect`]
/// (against the real machine) and [`plan`] (pure, for tests and headless
/// embedders that already know their hardware).
///
/// # Example
///
/// ```
/// use sovmr_core::encoder::EncoderCapabilities;
/// use sovmr_core::hardware::{parse_smi_line, HardwareConfigurator};
/// use sovmr_core::job::Codec;
///
/// let gpu = parse_smi_line("NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 580.159.03").unwrap();
/// let caps = EncoderCapabilities { av1: true, hevc: true };
/// let plan = HardwareConfigurator::plan(gpu, caps).unwrap();
/// assert_eq!(plan.settings.codec, Codec::Av1); // Ada -> AV1 preferred
/// assert_eq!(plan.worker_slots, 5);            // 8 GiB VRAM -> 5 sessions
/// ```
///
/// [`detect`]: HardwareConfigurator::detect
/// [`plan`]: HardwareConfigurator::plan
pub struct HardwareConfigurator;

impl HardwareConfigurator {
    /// Full detection pipeline against the real machine: queries
    /// `nvidia-smi`, asks the backend for its capabilities, and combines
    /// both via [`HardwareConfigurator::plan`].
    ///
    /// # Errors
    ///
    /// [`HardwareError::NoNvidiaDriver`] when `nvidia-smi` is missing or
    /// reports no GPU; [`HardwareError::NoUsableEncoder`] when the backend
    /// offers neither AV1 nor HEVC.
    pub fn detect(provider: &dyn EncoderProvider) -> Result<EncodePlan, HardwareError> {
        let gpu = query_nvidia_smi()?;
        let caps = provider
            .capabilities()
            .map_err(|e| HardwareError::NoUsableEncoder(e.to_string()))?;
        Self::plan(gpu, caps)
    }

    /// Pure decision logic, unit-testable without hardware: intersect GPU
    /// generation with backend capabilities and pick codec + worker count.
    ///
    /// # Errors
    ///
    /// [`HardwareError::NoUsableEncoder`] when the capability intersection
    /// is empty (backend can encode neither AV1 nor HEVC).
    pub fn plan(gpu: GpuInfo, caps: EncoderCapabilities) -> Result<EncodePlan, HardwareError> {
        // `caps` comes from `nvEncGetEncodeGUIDs` — the driver's own
        // list of what this encoder can produce. There used to be a
        // second gate here, classifying the GPU by its marketing name
        // ("RTX 30" implies no AV1 encode) and AND-ing the two. That was
        // a guess standing in for a fact, and once the fact is available
        // the guess can only do harm: an Ada-class card whose name the
        // table had never seen classified as Unknown and was downgraded
        // to HEVC even though its driver offered AV1.
        let codec = if caps.av1 {
            Codec::Av1
        } else if caps.hevc {
            Codec::Hevc
        } else {
            return Err(HardwareError::NoUsableEncoder(format!(
                "backend offers neither AV1 nor HEVC for '{}'",
                gpu.name
            )));
        };
        let worker_slots = default_worker_slots(gpu.vram_mib);
        Ok(EncodePlan {
            settings: EncodeSettings::visually_lossless(codec),
            worker_slots,
            gpu,
        })
    }
}

/// Parse one CSV line of `nvidia-smi --query-gpu=name,memory.total,driver_version`
/// output into a [`GpuInfo`]. Returns `None` for malformed lines — never
/// panics on driver garbage. Public so embedders with their own GPU
/// enumeration can still reuse the classification logic.
pub fn parse_smi_line(line: &str) -> Option<GpuInfo> {
    let mut fields = line.split(',').map(str::trim);
    let name = fields.next()?.to_string();
    if name.is_empty() {
        return None;
    }
    let vram_mib = fields.next()?.parse::<u64>().ok();
    let driver_version = fields.next().unwrap_or("unknown").to_string();
    Some(GpuInfo {
        name,
        vram_mib,
        driver_version,
    })
}

fn query_nvidia_smi() -> Result<GpuInfo, HardwareError> {
    let output = Command::new(NVIDIA_SMI)
        .args(NVIDIA_SMI_ARGS)
        .output()
        .map_err(|e| HardwareError::NoNvidiaDriver(format!("{NVIDIA_SMI}: {e}")))?;
    if !output.status.success() {
        return Err(HardwareError::NoNvidiaDriver(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(parse_smi_line)
        .ok_or_else(|| HardwareError::NoNvidiaDriver("no GPU rows in nvidia-smi output".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NVENC_SESSION_CAP;

    fn rtx4070m() -> GpuInfo {
        parse_smi_line("NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 550.120").unwrap()
    }

    #[test]
    fn parses_smi_csv_line() {
        let gpu = rtx4070m();
        assert_eq!(gpu.name, "NVIDIA GeForce RTX 4070 Laptop GPU");
        assert_eq!(gpu.vram_mib, Some(8188));
        assert_eq!(gpu.driver_version, "550.120");
    }

    #[test]
    fn rejects_garbage_smi_output() {
        assert!(parse_smi_line("").is_none());
        // Missing fields -> no GpuInfo, never a panic.
        assert!(parse_smi_line("Lone Name Without Fields").is_none());
    }

    #[test]
    fn ada_with_av1_backend_selects_av1() {
        let caps = EncoderCapabilities {
            av1: true,
            hevc: true,
        };
        let plan = HardwareConfigurator::plan(rtx4070m(), caps).unwrap();
        assert_eq!(plan.settings.codec, Codec::Av1);
        assert!(plan.worker_slots >= 1 && plan.worker_slots <= NVENC_SESSION_CAP);
    }

    #[test]
    fn ada_without_av1_backend_falls_back_to_hevc() {
        // GPU name suggests AV1, but the backend reports it cannot encode
        // it: trust the backend, not the GPU name.
        let caps = EncoderCapabilities {
            av1: false,
            hevc: true,
        };
        let plan = HardwareConfigurator::plan(rtx4070m(), caps).unwrap();
        assert_eq!(plan.settings.codec, Codec::Hevc);
    }

    #[test]
    fn a_gpu_whose_driver_offers_no_av1_gets_hevc_whatever_it_is_called() {
        // An Ampere card. Its name is irrelevant now — what decides is
        // that `nvEncGetEncodeGUIDs` did not list AV1, which is the
        // truth about the silicon rather than a guess about the label.
        let gpu = parse_smi_line("NVIDIA GeForce RTX 3080, 10240, 550.120").unwrap();
        let caps = EncoderCapabilities {
            av1: false,
            hevc: true,
        };
        let plan = HardwareConfigurator::plan(gpu, caps).unwrap();
        assert_eq!(plan.settings.codec, Codec::Hevc);
    }

    #[test]
    fn an_unrecognised_name_no_longer_costs_a_capable_card_its_av1() {
        // The regression the name table would have caused: a future card
        // the string list has never heard of, whose driver plainly says
        // it encodes AV1. Guessing conservatively here would silently
        // hand the user HEVC on hardware that could do better.
        let gpu = parse_smi_line("NVIDIA GeForce RTX 6090 Ti, 32768, 700.10").unwrap();
        let caps = EncoderCapabilities {
            av1: true,
            hevc: true,
        };
        let plan = HardwareConfigurator::plan(gpu, caps).unwrap();
        assert_eq!(plan.settings.codec, Codec::Av1);
    }

    #[test]
    fn no_codecs_is_a_hard_error() {
        let caps = EncoderCapabilities {
            av1: false,
            hevc: false,
        };
        assert!(HardwareConfigurator::plan(rtx4070m(), caps).is_err());
    }
}
