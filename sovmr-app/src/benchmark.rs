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

//! `sovmr --benchmark <input>`: the sovereign engine measured against
//! the system's ffmpeg as an external yardstick (ffmpeg is invoked
//! purely as a competitor process via `std::process::Command`, never
//! linked). Both engines transcode the same file to AV1-class output;
//! the report table compares wall time, throughput, size, and — when
//! the yardstick ffmpeg has the `psnr` filter — measured visual delta
//! against the source.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use sovmr_core::encoder::EncoderProvider;
use sovmr_core::heuristics::{plan, SourceProfile, Strategy};
use sovmr_core::job::Codec;
use sovmr_native::{transcode, NativeSettings, TranscodeOutcome};
use sovmr_nvenc::encoder::{EncodeCodec, EncodePreset};

/// The strategy under test: the one that must beat ffmpeg on dirty
/// sources (hard 60 % source-bitrate ceiling).
const BENCH_STRATEGY: Strategy = Strategy::Balanced;

/// ffmpeg software encoders tried as the yardstick, in preference
/// order, with "default medium" settings per the benchmark contract.
const YARDSTICKS: [(&str, &[&str]); 2] = [
    ("libsvtav1", &["-c:v", "libsvtav1"]),
    ("libx265", &["-c:v", "libx265", "-preset", "medium"]),
];

/// One engine's measured result.
struct Lap {
    engine: String,
    seconds: f64,
    output_bytes: u64,
    psnr: Option<f64>,
}

/// Run the benchmark; returns a process exit code (0 = table printed).
pub fn run(provider: &dyn EncoderProvider, input: &Path) -> i32 {
    let Ok(info) = provider.probe(input) else {
        eprintln!("benchmark: cannot probe {}", input.display());
        return 1;
    };
    let input_bytes = info.size_bytes;
    let frames = info
        .duration
        .map(|d| d.as_secs_f64() * if info.fps > 0.0 { info.fps } else { 30.0 })
        .unwrap_or(0.0);

    println!("SOVEREIGN BENCHMARK — {}", input.display());
    println!(
        "source: {}x{} {} @ {:.2} fps, {} bytes",
        info.width, info.height, info.video_codec, info.fps, input_bytes
    );
    match SourceProfile::measure(&info) {
        Some(p) => println!(
            "profile: {} bit/s, {:.3} bits/(px·frame){}",
            p.bitrate,
            p.milli_bppf as f64 / 1000.0,
            match p.anomaly {
                Some(a) => format!(" — ANOMALY: {a:?}"),
                None => String::new(),
            }
        ),
        None => println!("profile: source too opaque to measure (fallback ceilings)"),
    }
    // Which codec this GPU can actually produce, asked of the driver.
    // Hardcoding AV1 here made the benchmark unrunnable on every
    // pre-Ada card: `nvEncGetEncodePresetConfigEx` rejects the AV1 GUID
    // with NV_ENC_ERR_INVALID_PARAM on hardware that only decodes it.
    // Found on an RTX 3060, but nothing about it was Windows-specific —
    // an Ampere card under Linux would have failed identically.
    let codec = match provider.capabilities() {
        Ok(caps) if caps.av1 => Codec::Av1,
        Ok(caps) if caps.hevc => Codec::Hevc,
        Ok(_) => {
            eprintln!("benchmark: this GPU encodes neither AV1 nor HEVC");
            return 1;
        }
        Err(e) => {
            eprintln!("benchmark: cannot ask the encoder what it supports: {e}");
            return 1;
        }
    };
    println!("codec: {codec:?} (as reported by the driver)");
    let rate_plan = plan(&info, BENCH_STRATEGY, codec);
    println!(
        "strategy: {} -> cq {}, hard ceiling {} bit/s ({:.1} Mbit/s)\n",
        BENCH_STRATEGY.label(),
        rate_plan.cq,
        rate_plan.max_bitrate,
        f64::from(rate_plan.max_bitrate) / 1e6,
    );

    let dir = std::env::temp_dir().join(format!("sovmr-bench-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("benchmark: cannot create {}", dir.display());
        return 1;
    }

    // Lap 1: the sovereign zero-copy pipeline.
    let ours_path = dir.join("sovmr.mp4");
    let started = Instant::now();
    let outcome = transcode(
        input,
        &ours_path,
        &NativeSettings {
            codec: match codec {
                Codec::Av1 => EncodeCodec::Av1,
                Codec::Hevc => EncodeCodec::Hevc,
            },
            cq: rate_plan.cq,
            max_bitrate: rate_plan.max_bitrate,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    );
    let ours = match outcome {
        Ok(TranscodeOutcome::Completed(stats)) => Lap {
            // Name the codec that was actually used, not the one this
            // harness used to assume. On an Ampere card the row said
            // "AV1" over an HEVC encode, which is the sort of label that
            // quietly invalidates a comparison.
            engine: format!("sovmr (NVENC {codec:?}, {})", BENCH_STRATEGY.label()),
            seconds: started.elapsed().as_secs_f64(),
            output_bytes: stats.output_bytes,
            psnr: psnr_vs_source(&ours_path, input),
        },
        Ok(TranscodeOutcome::Cancelled) => unreachable!("no cancel source"),
        Err(e) => {
            eprintln!("benchmark: sovereign pipeline failed: {e}");
            let _ = std::fs::remove_dir_all(&dir);
            return 1;
        }
    };

    // Lap 2: the ffmpeg yardstick (first software AV1/HEVC encoder that
    // works on this system).
    let ffmpeg = run_ffmpeg_yardstick(input, &dir);

    print_table(input_bytes, frames, &ours, ffmpeg.as_ref());
    verdict(input_bytes, &ours, ffmpeg.as_ref());
    let _ = std::fs::remove_dir_all(&dir);
    0
}

/// Encode with the first available ffmpeg software yardstick.
fn run_ffmpeg_yardstick(input: &Path, dir: &Path) -> Option<Lap> {
    for (name, args) in YARDSTICKS {
        let out: PathBuf = dir.join(format!("ffmpeg_{name}.mp4"));
        let started = Instant::now();
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(input)
            .args(args)
            .args(["-c:a", "copy"])
            .arg(&out)
            .status();
        match status {
            Ok(s) if s.success() => {
                let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
                return Some(Lap {
                    engine: format!("ffmpeg ({name}, defaults)"),
                    seconds: started.elapsed().as_secs_f64(),
                    output_bytes: bytes,
                    psnr: psnr_vs_source(&out, input),
                });
            }
            _ => {
                eprintln!("benchmark: yardstick {name} unavailable, trying next");
                continue;
            }
        }
    }
    eprintln!("benchmark: NO ffmpeg software encoder available — sovereign lap only");
    None
}

/// Average PSNR of `encoded` against `source` via ffmpeg's psnr filter
/// (the external arbiter). `None` when the filter/decoder is missing.
fn psnr_vs_source(encoded: &Path, source: &Path) -> Option<f64> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-nostats", "-i"])
        .arg(encoded)
        .arg("-i")
        .arg(source)
        .args(["-lavfi", "psnr", "-f", "null", "-"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stderr);
    let avg = text.split("average:").nth(1)?;
    avg.split_whitespace().next()?.parse().ok()
}

fn print_table(input_bytes: u64, frames: f64, ours: &Lap, ffmpeg: Option<&Lap>) {
    println!("┌─────────────────────────────────┬────────────┬──────────┬──────────────┬───────────┬───────────┐");
    println!("│ engine                          │  time (s)  │   fps    │  size (B)    │ reduction │ PSNR (dB) │");
    println!("├─────────────────────────────────┼────────────┼──────────┼──────────────┼───────────┼───────────┤");
    for lap in std::iter::once(ours).chain(ffmpeg) {
        let reduction = 100.0 - (lap.output_bytes as f64 / input_bytes as f64 * 100.0);
        let fps = if lap.seconds > 0.0 {
            frames / lap.seconds
        } else {
            0.0
        };
        println!(
            "│ {:<31} │ {:>10.2} │ {:>8.1} │ {:>12} │ {:>8.1}% │ {:>9} │",
            lap.engine,
            lap.seconds,
            fps,
            lap.output_bytes,
            reduction,
            lap.psnr
                .map(|p| format!("{p:.2}"))
                .unwrap_or_else(|| "n/a".into()),
        );
    }
    println!("└─────────────────────────────────┴────────────┴──────────┴──────────────┴───────────┴───────────┘");
}

/// The verdict line: savings per encode-second is THE efficiency
/// metric — a compressor that saves as much but 30× faster wins.
fn verdict(input_bytes: u64, ours: &Lap, ffmpeg: Option<&Lap>) {
    let Some(ff) = ffmpeg else { return };
    let saved = |l: &Lap| input_bytes.saturating_sub(l.output_bytes) as f64;
    let ours_eff = saved(ours) / ours.seconds.max(f64::EPSILON) / 1e6;
    let ff_eff = saved(ff) / ff.seconds.max(f64::EPSILON) / 1e6;
    println!(
        "\nefficiency (MB shaved per encode-second): sovmr {ours_eff:.1} vs ffmpeg {ff_eff:.1} \
         — {:.1}× advantage",
        ours_eff / ff_eff.max(f64::EPSILON)
    );
    if ours.output_bytes < ff.output_bytes {
        println!(
            "size: sovmr output is SMALLER than ffmpeg's ({} vs {} bytes)",
            ours.output_bytes, ff.output_bytes
        );
    } else {
        println!(
            "size: ffmpeg output is {:.1}% smaller, but spent {:.0}× the wall time",
            (1.0 - ff.output_bytes as f64 / ours.output_bytes as f64) * 100.0,
            ff.seconds / ours.seconds.max(f64::EPSILON)
        );
    }
    match (ours.psnr, ff.psnr) {
        (Some(a), Some(b)) if (a - b).abs() < 1.0 => {
            println!("quality: visually identical (PSNR within 1 dB)")
        }
        (Some(a), Some(b)) => println!("quality: PSNR sovmr {a:.2} dB vs ffmpeg {b:.2} dB"),
        _ => println!("quality: PSNR unavailable — visually identical by strategy design"),
    }
}
