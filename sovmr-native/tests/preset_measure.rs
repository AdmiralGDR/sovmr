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

//! Does the slowest preset (P7) earn its extra time? A lever is adopted
//! only when it raises quality without growing the file, so this measures
//! both sides for the same source at the same target quality and prints
//! them. ffmpeg is only the fixture generator; the quality number is our
//! own decode-and-compare.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use sovmr_native::verify::verify;
use sovmr_native::{transcode, NativeSettings, TranscodeOutcome};
use sovmr_nvenc::encoder::{EncodeCodec, EncodePreset};

/// The codec this GPU can actually encode with (AV1 on Ada, else HEVC).
fn best_codec() -> Option<EncodeCodec> {
    use sovmr_core::encoder::EncoderProvider;
    let caps = sovmr_native::NvencNativeProvider.capabilities().ok()?;
    if caps.av1 {
        Some(EncodeCodec::Av1)
    } else if caps.hevc {
        Some(EncodeCodec::Hevc)
    } else {
        None
    }
}

/// A 720p clip with motion and detail — enough that the encoder's effort
/// has something to work on. ffmpeg builds it; nothing here ships.
fn make_fixture(dir: &Path) -> std::path::PathBuf {
    let clip = dir.join("preset_fixture.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=10:size=1280x720:rate=30",
            "-f",
            "lavfi",
            "-i",
            "nullsrc=size=1280x720:duration=10:rate=30",
            "-filter_complex",
            "[0:v]noise=alls=18:allf=t+u[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
        ])
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success(), "fixture generation failed");
    clip
}

/// Encode `input` at `preset` and return (PSNR dB, SSIM, output bytes).
fn measure(codec: EncodeCodec, preset: EncodePreset, dir: &Path, input: &Path) -> (f64, f64, u64) {
    let name = format!("{preset:?}").to_lowercase();
    let output = dir.join(format!("out_{name}.mp4"));
    let outcome = transcode(
        input,
        &output,
        &NativeSettings {
            codec,
            cq: 25,
            max_bitrate: 0,
            preset,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcode must complete");
    assert!(
        matches!(outcome, TranscodeOutcome::Completed(_)),
        "unexpected cancel"
    );
    let size = std::fs::metadata(&output).expect("output present").len();
    let v = verify(input, &output).expect("verify must decode both sides");
    (v.psnr, v.ssim, size)
}

#[test]
#[ignore = "requires NVIDIA GPU + driver + ffmpeg (fixture) — measurement, not a gate"]
fn p7_vs_p6_is_only_worth_it_if_quality_rises_at_no_extra_size() {
    let Some(codec) = best_codec() else {
        println!("SKIPPED: this GPU has no usable encoder");
        return;
    };
    let dir = std::env::temp_dir().join(format!("sovmr-preset-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir);

    let (p6_psnr, p6_ssim, p6_size) = measure(codec, EncodePreset::Quality, &dir, &input);
    let (p7_psnr, p7_ssim, p7_size) = measure(codec, EncodePreset::Slowest, &dir, &input);

    println!("--- preset measurement ({codec:?}, cq=25) ---");
    println!("P6 (Quality): PSNR {p6_psnr:.3} dB  SSIM {p6_ssim:.5}  {p6_size} bytes");
    println!("P7 (Maximum): PSNR {p7_psnr:.3} dB  SSIM {p7_ssim:.5}  {p7_size} bytes");
    let dssim = p7_ssim - p6_ssim;
    let dsize_pct = (p7_size as f64 - p6_size as f64) / p6_size as f64 * 100.0;
    println!("delta: SSIM {dssim:+.5}  size {dsize_pct:+.2}%");
    let verdict = if dssim > 0.0 && p7_size <= p6_size {
        "worth adopting"
    } else if dssim.abs() < 1e-5 && dsize_pct.abs() < 0.5 {
        "a wash"
    } else {
        "a trade, not a free win"
    };
    println!("verdict: P7 is {verdict} — adopt only if quality rose at no larger a file");

    let _ = std::fs::remove_dir_all(&dir);
}
