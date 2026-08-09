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

//! The Sovereign Heuristic Engine **v2**: deterministic rate-control
//! math from the *physical reality* of the source — non-linear, not a
//! flat percentage.
//!
//! # Why v1's flat percentages were wrong
//!
//! v1 capped the output at a fixed fraction of the source bitrate
//! (90/60/40 %). On a **fat** source that is fine. On a source that is
//! *already lean* it is nearly useless: a 1080p H.264 file at 5.2 Mbit/s
//! (0.084 bits per pixel per frame) got a 4.7 Mbit/s ceiling — a 10 %
//! cut, when AV1 can hold that exact picture at roughly **half** the
//! bitrate. The bits we can remove do not come from "a percentage"; they
//! come from two physically distinct places:
//!
//! 1. **Codec generation.** AV1 needs ~55 % of H.264's bitrate (~80 % of
//!    HEVC's) for the same perceived quality. This gain exists no matter
//!    how lean the source already is — it is the compression *format*
//!    being better, not the source being wasteful.
//! 2. **Source redundancy.** A source spending absurd bits per pixel is
//!    also storing noise, grain, or an old encoder's inefficiency. That
//!    surplus *scales with density*: the fatter the source, the more
//!    there is to throw away — and the leaner it is, the less.
//!
//! v2 multiplies the two, so the ceiling is
//! `source × codec_gain(source→target) × redundancy(bppf) × strategy`,
//! floored by the bitrate the resolution physically needs and capped by
//! the theoretical sanity ceiling. Every step is integer arithmetic with
//! unit-tested knots (no floats where integers do, no magic the
//! tests cannot pin).

use crate::job::{Codec, MediaInfo};

/// Theoretical sane ceiling in milli-bits per pixel per second: no
/// output may exceed 0.08 b/(px·s) (≈ 5 Mbit/s at 1080p30) regardless
/// of how fat the source is — beyond this, modern codecs store noise.
pub const THEORETICAL_MAX_MILLI_BPPS: u64 = 80;

/// The bitrate a resolution physically needs to stay *held* — below this
/// density even AV1 starts shedding structure, not just noise. In
/// milli-bits per pixel per frame: 0.012 bppf ≈ 750 kbit/s at 1080p30.
/// The curve may never push the ceiling below this (unless the source
/// itself already lives under it).
pub const RESOLUTION_FLOOR_MILLI_BPPF: u64 = 12;

/// Audio that eats at least this share of the file's bits makes the
/// video work moot — bit-exact passthrough will dominate the output.
pub const AUDIO_HEAVY_SHARE_PCT: u64 = 20;

// ---- codec generational gain ------------------------------------------------

/// Bitrate (in percent) the *target* codec needs to reproduce what the
/// *source* codec stored, at equal perceived quality. These generational
/// facts are what the whole v2 curve rests on: AV1 ≈ 55 % of H.264,
/// ≈ 80 % of HEVC; HEVC ≈ 65 % of H.264. Re-encoding a codec into itself
/// buys nothing generationally (100 %).
fn codec_gain_pct(source_codec: &str, target: Codec) -> u64 {
    let lower = source_codec.to_ascii_lowercase();
    let source = lower.split_whitespace().next().unwrap_or("");
    match (source, target) {
        ("h264" | "avc" | "avc1", Codec::Av1) => 55,
        ("h264" | "avc" | "avc1", Codec::Hevc) => 65,
        ("hevc" | "h265", Codec::Av1) => 80,
        ("hevc" | "h265", Codec::Hevc) => 100,
        ("av1", Codec::Av1) => 100,
        ("av1", Codec::Hevc) => 125, // a downgrade; Smart Skip normally stops this
        // Unknown source codec: assume a modern one and claim only a
        // conservative gain — never over-promise.
        (_, Codec::Av1) => 75,
        (_, Codec::Hevc) => 85,
    }
}

/// The **non-linear redundancy curve**: how much of what remains (after
/// the codec gain) must be *kept*, as a percentage, keyed on the source's
/// density in milli-bits/(pixel·frame).
///
/// Knots (piecewise-linear between them, flat outside):
///
/// | bppf | keep | meaning |
/// |------|------|---------|
/// | ≤ 0.020 | 100 % | starving already — take nothing beyond the codec gain |
/// | 0.050 | 95 % | lean, well-encoded source |
/// | 0.150 | 75 % | comfortable consumer bitrate: real surplus exists |
/// | ≥ 0.300 | 55 % | noise, grain, or an ancient encoder: cut hard |
///
/// So a *fat* source gets `codec_gain × 0.55` (H.264→AV1: 30 % of source
/// — the aggressive-cut regime), while an *already lean* source gets
/// `codec_gain × ~1.0` (≈ 55 % of source): driven by codec superiority
/// alone, exactly as much as physics actually allows.
fn redundancy_keep_pct(milli_bppf: u64) -> u64 {
    /// (density, keep %) knots of the curve.
    const KNOTS: [(u64, u64); 4] = [(20, 100), (50, 95), (150, 75), (300, 55)];
    if milli_bppf <= KNOTS[0].0 {
        return KNOTS[0].1;
    }
    for w in KNOTS.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if milli_bppf <= x1 {
            // Linear interpolation between knots (integer, y decreasing).
            return y0 - (milli_bppf - x0) * (y0 - y1) / (x1 - x0);
        }
    }
    KNOTS[KNOTS.len() - 1].1
}

/// Strategy multipliers, in percent: the curve says what the *format*
/// physically allows, the strategy says how much of that headroom the
/// human wants to spend.
const STRATEGY_LOSSLESS_PCT: u64 = 100;
const STRATEGY_BALANCED_PCT: u64 = 80;
const STRATEGY_AGGRESSIVE_PCT: u64 = 62;

/// CQ targets per strategy (NVENC constant-quality scale, lower = better
/// fidelity). CQ drives quality *under* the ceiling; the ceiling is the
/// hard physical limit.
const CQ_VISUALLY_LOSSLESS: u8 = 25;
const CQ_BALANCED: u8 = 32;
const CQ_AGGRESSIVE: u8 = 38;

/// Anomaly thresholds in milli-bppf (see [`SourceAnomaly`]).
const NOISY_BPPF_MILLI: u64 = 300;
const STARVED_BPPF_MILLI: u64 = 15;

/// Semantic encode intent — everything the UI exposes. The engine, not
/// the human, owns the translation into CQ and bitrate ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Indistinguishable from the source: spends the full headroom the
    /// codec generation gives us, and never inflates.
    #[default]
    VisuallyLossless,
    /// The sweet spot: 80 % of the physically-allowed headroom.
    Balanced,
    /// Maximum shrink: 62 % of it.
    Aggressive,
}

impl Strategy {
    /// UI label.
    pub fn label(self) -> &'static str {
        match self {
            Strategy::VisuallyLossless => "Visually Lossless",
            Strategy::Balanced => "Balanced",
            Strategy::Aggressive => "Aggressive (smallest)",
        }
    }

    /// How much of the curve's headroom this strategy spends, in percent.
    fn headroom_pct(self) -> u64 {
        match self {
            Strategy::VisuallyLossless => STRATEGY_LOSSLESS_PCT,
            Strategy::Balanced => STRATEGY_BALANCED_PCT,
            Strategy::Aggressive => STRATEGY_AGGRESSIVE_PCT,
        }
    }

    /// The constant-quality target driving VBR under the ceiling.
    pub fn cq(self) -> u8 {
        match self {
            Strategy::VisuallyLossless => CQ_VISUALLY_LOSSLESS,
            Strategy::Balanced => CQ_BALANCED,
            Strategy::Aggressive => CQ_AGGRESSIVE,
        }
    }
}

/// What deep profiling concluded about one source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceAnomaly {
    /// Absurdly high bits/pixel/frame: noise, heavy grain, or an ancient
    /// encoder — the redundancy curve cuts hardest here.
    NoiseInflated,
    /// Starved bits/pixel/frame: an over-compressed rip or a fake-HD/4K
    /// upscale — there is no detail worth extra bitrate.
    BitStarved,
}

/// The measured physical reality of a source, derived purely from probe
/// facts (never from the file name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceProfile {
    /// Exact average bitrate of the whole file in bits/second.
    pub bitrate: u64,
    /// Video density × 1000: integer milli-bits per pixel per frame.
    pub milli_bppf: u64,
    /// Audio bitrate in bits/second (0 = no audio / unknown).
    pub audio_bitrate: u64,
    /// Audio's share of the file's bits, in percent.
    pub audio_share_pct: u64,
    /// Audio eats at least [`AUDIO_HEAVY_SHARE_PCT`] of the file: the
    /// bit-exact passthrough caps the achievable compression ratio no
    /// matter how well the video leg performs.
    pub audio_heavy: bool,
    /// Detected density anomaly, when off the clean-source map.
    pub anomaly: Option<SourceAnomaly>,
}

impl SourceProfile {
    /// Profile a probed source. `None` when the probe facts are too
    /// incomplete for honest math — the caller then falls back to the
    /// resolution-only ceiling.
    pub fn measure(info: &MediaInfo) -> Option<SourceProfile> {
        let seconds = info.duration?.as_secs_f64();
        if seconds <= 0.0 || info.width == 0 || info.height == 0 || info.size_bytes == 0 {
            return None;
        }
        // The video's own bytes when the container can say, and the
        // whole file only when it cannot. Every judgement below is about
        // how densely the *pictures* are coded, and the file length
        // answers a different question.
        let video_bytes = if info.video_bytes > 0 {
            info.video_bytes
        } else {
            info.size_bytes
        };
        let bitrate = ((video_bytes as f64) * 8.0 / seconds) as u64;
        let fps = if info.fps > 0.0 { info.fps } else { 30.0 };
        let pixels = u64::from(info.width) * u64::from(info.height);
        let milli_bppf = ((bitrate as f64 / fps) * 1000.0 / pixels as f64) as u64;
        // Against the whole file, which is the only denominator that
        // contains the audio. Dividing by the video bitrate would ask
        // what share of the pictures the sound is, which is not a
        // question.
        // Against the **file's** bitrate, which is the only denominator
        // that contains the audio. Dividing by the video's own — as this
        // did the moment the video got its own number — asks what share
        // of the pictures the sound is, and answers "more than all of
        // them" on any film with six lossless tracks.
        let file_bitrate = ((info.size_bytes as f64) * 8.0 / seconds) as u64;
        let audio_share_pct = (info.audio_bitrate * 100)
            .checked_div(file_bitrate)
            .unwrap_or(0)
            .min(100);
        let anomaly = if milli_bppf >= NOISY_BPPF_MILLI {
            Some(SourceAnomaly::NoiseInflated)
        } else if milli_bppf <= STARVED_BPPF_MILLI {
            Some(SourceAnomaly::BitStarved)
        } else {
            None
        };
        Some(SourceProfile {
            bitrate,
            milli_bppf,
            audio_bitrate: info.audio_bitrate,
            audio_share_pct,
            audio_heavy: audio_share_pct >= AUDIO_HEAVY_SHARE_PCT,
            anomaly,
        })
    }
}

/// The engine's verdict: the exact numbers NVENC gets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RatePlan {
    /// Constant-quality target (drives VBR quality below the ceiling).
    pub cq: u8,
    /// Hard `maxBitRate` ceiling in bits/second — the hardware is
    /// physically forbidden from exceeding it.
    pub max_bitrate: u32,
}

/// The theoretical ceiling for a geometry, independent of the source:
/// pixels/s × [`THEORETICAL_MAX_MILLI_BPPS`].
pub fn theoretical_max_bitrate(width: u32, height: u32, fps_num: u32, fps_den: u32) -> u32 {
    u32::try_from(
        u64::from(width) * u64::from(height) * u64::from(fps_num) * THEORETICAL_MAX_MILLI_BPPS
            / (u64::from(fps_den.max(1)) * 1000),
    )
    .unwrap_or(u32::MAX)
}

/// The bitrate this geometry physically needs to hold its resolution
/// ([`RESOLUTION_FLOOR_MILLI_BPPF`] × pixels × fps).
fn resolution_floor_bitrate(info: &MediaInfo, fps: f64) -> u64 {
    let pixels = u64::from(info.width) * u64::from(info.height);
    (pixels as f64 * fps * RESOLUTION_FLOOR_MILLI_BPPF as f64 / 1000.0) as u64
}

/// Translate intent + measured reality into NVENC numbers (**v2**).
///
/// `ceiling = source × codec_gain × redundancy(bppf) × strategy`, then
/// clamped: never above the theoretical sanity cap, never below the
/// bitrate the resolution needs (unless the source is already under it),
/// and never at/above the source itself — a re-encode must shrink.
///
/// An unmeasurable source falls back to the theoretical ceiling alone:
/// still capped, never unbounded.
pub fn plan(info: &MediaInfo, strategy: Strategy, target: Codec) -> RatePlan {
    let fps = if info.fps > 0.0 { info.fps } else { 30.0 };
    let fps_milli = (fps * 1000.0) as u32;
    let theoretical = u64::from(theoretical_max_bitrate(
        info.width,
        info.height,
        fps_milli,
        1000,
    ));

    let Some(profile) = SourceProfile::measure(info) else {
        return RatePlan {
            cq: strategy.cq(),
            max_bitrate: u32::try_from(theoretical.max(1)).unwrap_or(u32::MAX),
        };
    };

    // The two physically distinct sources of savings, multiplied.
    let gain = codec_gain_pct(&info.video_codec, target);
    let keep = redundancy_keep_pct(profile.milli_bppf);
    let headroom = strategy.headroom_pct();
    let mut ceiling = profile.bitrate * gain / 100 * keep / 100 * headroom / 100;

    // Floor: never starve the resolution below what it needs to hold —
    // unless the source already lives below that floor, in which case
    // the source itself bounds us (we must never inflate).
    let floor = resolution_floor_bitrate(info, fps).min(profile.bitrate);
    ceiling = ceiling.max(floor);

    // Caps: the theoretical sanity ceiling, and the source itself (a
    // re-encode that grows the file is a bug, not a strategy).
    ceiling = ceiling.min(theoretical).min(profile.bitrate * 95 / 100);

    RatePlan {
        cq: strategy.cq(),
        max_bitrate: u32::try_from(ceiling.max(1)).unwrap_or(u32::MAX),
    }
}

/// One-line human summary of what the engine *measured* on this source
/// and what it *decided* — the string the dashboard puts on the job
/// card, e.g.:
///
/// `Source: Lean · 0.08 bpp · 5.2 Mbit/s → ceiling 2.5 Mbit/s (Visually Lossless)`
///
/// A heavy audio track appends its own warning: passthrough audio caps
/// the achievable ratio no matter how well the video compresses.
///
/// Formatting lives here, next to the math it describes: the UI
/// renders the sentence, it never re-derives the numbers. `None` when
/// the probe facts are too incomplete to say anything honest.
pub fn insight(info: &MediaInfo, strategy: Strategy, target: Codec) -> Option<String> {
    let profile = SourceProfile::measure(info)?;
    let plan = plan(info, strategy, target);
    let character = match profile.anomaly {
        Some(SourceAnomaly::NoiseInflated) => "Noisy",
        Some(SourceAnomaly::BitStarved) => "Bit-starved",
        None if profile.milli_bppf < 100 => "Lean",
        None => "Clean",
    };
    let mut line = format!(
        "Source: {character} · {:.2} bpp · {} → ceiling {} ({})",
        profile.milli_bppf as f64 / 1000.0,
        mbits(profile.bitrate),
        mbits(u64::from(plan.max_bitrate)),
        strategy.label(),
    );
    if let Some(composition) = composition(info) {
        line.push_str(&composition);
    }
    if profile.audio_heavy {
        line.push_str(&format!(
            " · ⚠ Audio is heavy ({}, {} % of the file) — passthrough limits compression",
            mbits(profile.audio_bitrate),
            profile.audio_share_pct,
        ));
    }
    Some(line)
}

/// What the source is made of, as a trailing clause for [`insight`].
///
/// A single audio track is the unremarkable case and stays unsaid; the
/// point of this line is that someone feeding a six-language master can
/// see it was understood *before* the job runs, rather than checking the
/// output afterwards. Subtitles and chapters are shown because the MP4
/// path cannot carry them yet, so seeing them here is the early half of
/// the warning the job will emit.
fn composition(info: &MediaInfo) -> Option<String> {
    let mut parts = Vec::new();
    if info.audio_tracks > 1 {
        parts.push(format!("{} audio tracks", info.audio_tracks));
    }
    if info.subtitle_tracks > 0 {
        parts.push(format!("{} subtitles", info.subtitle_tracks));
    }
    if info.chapters {
        parts.push("chapters".to_string());
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!(" · {}", parts.join(" · ")))
}

/// Render bits/second as a human "12.3 Mbit/s" (or kbit/s when tiny).
fn mbits(bits_per_second: u64) -> String {
    if bits_per_second < 1_000_000 {
        format!("{:.0} kbit/s", bits_per_second as f64 / 1_000.0)
    } else {
        format!("{:.1} Mbit/s", bits_per_second as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod video_bytes_tests {
    use super::*;
    use crate::job::MediaInfo;

    /// A film whose sound is most of its file.
    fn film(video_bytes: u64, size_bytes: u64) -> MediaInfo {
        MediaInfo {
            duration: Some(std::time::Duration::from_secs(3600)),
            fps: 24.0,
            width: 1920,
            height: 1080,
            video_codec: "h264".into(),
            size_bytes,
            video_bytes,
            audio_bitrate: 6 * 640_000,
            audio_tracks: 6,
            subtitle_tracks: 0,
            chapters: false,
        }
    }

    #[test]
    fn six_lossless_audio_tracks_do_not_make_the_video_look_denser() {
        // The defect, stated as a test. The same pictures in two files —
        // one with a single track, one with six — were judged to have
        // different densities, because the density was measured from the
        // file rather than the video.
        let pictures = 4_000_000_000u64;
        let lean =
            SourceProfile::measure(&film(pictures, pictures + 100_000_000)).expect("measurable");
        let heavy =
            SourceProfile::measure(&film(pictures, pictures + 3_000_000_000)).expect("measurable");
        assert_eq!(
            lean.milli_bppf, heavy.milli_bppf,
            "the same pictures were judged at {} and {} milli-bits per pixel \
             because of the sound around them",
            lean.milli_bppf, heavy.milli_bppf
        );
        assert_eq!(lean.bitrate, heavy.bitrate);
    }

    #[test]
    fn a_container_that_cannot_say_falls_back_rather_than_reporting_nothing() {
        // Matroska has no sample-size table. Zero means "not known", and
        // the file length is the old behaviour — now confined to the
        // container that genuinely cannot answer.
        let unknown = SourceProfile::measure(&film(0, 4_000_000_000)).expect("measurable");
        let known =
            SourceProfile::measure(&film(4_000_000_000, 9_000_000_000)).expect("measurable");
        assert_eq!(unknown.bitrate, known.bitrate);
    }

    #[test]
    fn the_audio_share_is_still_measured_against_the_whole_file() {
        // The share asks what part of the *file* is sound. Dividing by
        // the video bitrate would ask what share of the pictures the
        // sound is, which is not a question anyone has.
        let info = film(1_000_000_000, 2_000_000_000);
        let profile = SourceProfile::measure(&info).expect("measurable");
        assert!(
            profile.audio_share_pct > 0 && profile.audio_share_pct < 100,
            "share was {}%",
            profile.audio_share_pct
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn info(w: u32, h: u32, secs: u64, bytes: u64, fps: f64, codec: &str) -> MediaInfo {
        MediaInfo {
            duration: Some(Duration::from_secs(secs)),
            width: w,
            height: h,
            video_codec: codec.into(),
            size_bytes: bytes,
            video_bytes: 0,
            fps,
            audio_bitrate: 0,
            ..Default::default()
        }
    }

    /// The user's actual regression: 1080p30 H.264 at 5.2 Mbit/s.
    fn lean_1080p() -> MediaInfo {
        info(1920, 1080, 100, 65_000_000, 30.0, "h264")
    }

    #[test]
    fn a_multi_track_source_says_what_it_is_made_of() {
        // Someone feeding a six-language master with subtitles and
        // chapters should see that it was understood BEFORE the job
        // runs. Discovering it from the output is too late to fix.
        let mut src = lean_1080p();
        src.audio_tracks = 6;
        src.subtitle_tracks = 2;
        src.chapters = true;
        let line = insight(&src, Strategy::Balanced, Codec::Av1).expect("insight");
        assert!(line.contains("6 audio tracks"), "{line}");
        assert!(line.contains("2 subtitles"), "{line}");
        assert!(line.contains("chapters"), "{line}");
    }

    #[test]
    fn an_ordinary_single_track_file_adds_no_noise_to_the_line() {
        // One audio track, no subtitles, no chapters is the common case
        // and the unremarkable one. Announcing "1 audio track" on every
        // file would bury the signal the line exists to carry.
        let mut src = lean_1080p();
        src.audio_tracks = 1;
        let line = insight(&src, Strategy::Balanced, Codec::Av1).expect("insight");
        assert!(!line.contains("audio track"), "{line}");
        assert!(!line.contains("subtitle"), "{line}");
        assert!(!line.contains("chapters"), "{line}");
    }

    #[test]
    fn subtitles_alone_are_reported_even_with_one_audio_track() {
        // The parts are independent: a normal stereo film with subtitles
        // must still say so, since the MP4 path cannot carry them.
        let mut src = lean_1080p();
        src.audio_tracks = 1;
        src.subtitle_tracks = 3;
        let line = insight(&src, Strategy::Balanced, Codec::Av1).expect("insight");
        assert!(line.contains("3 subtitles"), "{line}");
        assert!(!line.contains("audio track"), "{line}");
    }

    fn plan_of(src: &MediaInfo, s: Strategy) -> u32 {
        plan(src, s, Codec::Av1).max_bitrate
    }

    #[test]
    fn profile_computes_exact_bitrate_and_density() {
        let p = SourceProfile::measure(&info(1920, 1080, 100, 125_000_000, 30.0, "h264")).unwrap();
        assert_eq!(p.bitrate, 10_000_000);
        assert_eq!(p.milli_bppf, 160); // 10e6/30/2 073 600 = 0.1608 bppf
        assert_eq!(p.anomaly, None);
    }

    #[test]
    fn the_redundancy_curve_is_non_linear_and_monotone() {
        // Knots exactly.
        assert_eq!(redundancy_keep_pct(20), 100);
        assert_eq!(redundancy_keep_pct(50), 95);
        assert_eq!(redundancy_keep_pct(150), 75);
        assert_eq!(redundancy_keep_pct(300), 55);
        // Flat outside the knots — never extrapolates into nonsense.
        assert_eq!(redundancy_keep_pct(0), 100);
        assert_eq!(redundancy_keep_pct(5_000), 55);
        // Interpolated midpoints prove the curve actually bends.
        assert_eq!(redundancy_keep_pct(100), 85); // halfway 50 -> 150
        assert_eq!(redundancy_keep_pct(225), 65); // halfway 150 -> 300
                                                  // Monotone decreasing across the whole domain.
        let mut prev = 101;
        for d in (0..400).step_by(7) {
            let keep = redundancy_keep_pct(d);
            assert!(keep <= prev, "curve must not rise at {d}");
            prev = keep;
        }
    }

    #[test]
    fn codec_generation_is_the_engine_of_lean_source_savings() {
        assert_eq!(codec_gain_pct("h264", Codec::Av1), 55);
        assert_eq!(codec_gain_pct("h264", Codec::Hevc), 65);
        assert_eq!(codec_gain_pct("hevc", Codec::Av1), 80);
        assert_eq!(codec_gain_pct("hevc", Codec::Hevc), 100);
        assert_eq!(codec_gain_pct("av1", Codec::Av1), 100);
        // Unknown sources get a conservative, never over-promised gain.
        assert_eq!(codec_gain_pct("vp9", Codec::Av1), 75);
    }

    #[test]
    fn v2_beats_v1_on_the_users_lean_1080p_regression() {
        // v1 gave 90 % of source (4.68 Mbit/s) — a 10 % cut, useless.
        // v2: 0.084 bppf -> keep ~88 %, H.264->AV1 gain 55 % -> ~48 %.
        let source = 5_200_000u32;
        let plan = plan(&lean_1080p(), Strategy::VisuallyLossless, Codec::Av1);
        assert!(
            plan.max_bitrate < source / 2,
            "a lean H.264 source must be targeted under HALF its bitrate — \
             codec generation alone allows it; got {}",
            plan.max_bitrate
        );
        assert!(
            plan.max_bitrate > source / 3,
            "...but never starved below what the picture needs; got {}",
            plan.max_bitrate
        );
        // Strictly better than the v1 flat-90 % ceiling it replaces.
        assert!(plan.max_bitrate < source * 9 / 10);
    }

    #[test]
    fn fat_noisy_sources_get_the_aggressive_cut() {
        // 1080p30 at 25 Mbit/s: 0.40 bppf -> noise regime, keep 55 %.
        let fat = info(1920, 1080, 100, 312_500_000, 30.0, "h264");
        let p = SourceProfile::measure(&fat).unwrap();
        assert_eq!(p.anomaly, Some(SourceAnomaly::NoiseInflated));
        // 25e6 x 0.55 (codec) x 0.55 (redundancy) = 7.6 Mbit/s, so the
        // theoretical cap (4.98 Mbit/s at 1080p30) binds for the two
        // gentler strategies...
        assert_eq!(plan_of(&fat, Strategy::VisuallyLossless), 4_976_640);
        assert_eq!(plan_of(&fat, Strategy::Balanced), 4_976_640);
        // ...while Aggressive (x0.62) undercuts it on the curve alone.
        let aggressive = plan_of(&fat, Strategy::Aggressive);
        assert!(aggressive < 4_976_640, "got {aggressive}");
        // Either way a fat source is cut to a fraction of itself.
        assert!(aggressive < 25_000_000 / 5, "got {aggressive}");
    }

    #[test]
    fn strategies_stay_ordered_and_never_inflate() {
        let src = lean_1080p();
        let vl = plan_of(&src, Strategy::VisuallyLossless);
        let ba = plan_of(&src, Strategy::Balanced);
        let ag = plan_of(&src, Strategy::Aggressive);
        assert!(vl > ba && ba > ag, "{vl} {ba} {ag}");
        assert!(vl < 5_200_000, "every strategy stays under the source");
    }

    #[test]
    fn the_resolution_floor_stops_the_curve_from_starving_the_picture() {
        // 1080p30 needs 0.012 bppf = 746 496 bit/s to hold its structure;
        // not even Aggressive may push below that.
        let floor = resolution_floor_bitrate(&lean_1080p(), 30.0);
        assert_eq!(floor, 746_496);
        assert!(u64::from(plan_of(&lean_1080p(), Strategy::Aggressive)) >= floor);

        // A fake-4K upscale (2 Mbit/s at 2160p) is already below its own
        // floor: the source itself bounds the ceiling — never inflate.
        let upscale = info(3840, 2160, 100, 25_000_000, 30.0, "h264");
        assert_eq!(
            SourceProfile::measure(&upscale).unwrap().anomaly,
            Some(SourceAnomaly::BitStarved)
        );
        assert!(plan_of(&upscale, Strategy::Balanced) <= 2_000_000 * 95 / 100);
    }

    #[test]
    fn same_codec_reencodes_gain_nothing_generationally() {
        // AV1 -> AV1 at a lean density: no codec gain, little redundancy,
        // so the ceiling sits near the source. Smart Skip is what stops
        // this job; the engine simply refuses to promise savings it
        // cannot physically deliver.
        let av1 = info(1920, 1080, 100, 65_000_000, 30.0, "av1");
        let ceiling = plan_of(&av1, Strategy::VisuallyLossless);
        assert!(ceiling >= 4_000_000, "no generational gain to take");
        assert!(ceiling <= 5_200_000 * 95 / 100, "still never inflates");
    }

    #[test]
    fn heavy_audio_is_measured_flagged_and_surfaced() {
        // A 5.2 Mbit/s file whose audio is 1.5 Mbit/s = 28 % of the bits.
        let mut src = lean_1080p();
        src.audio_bitrate = 1_500_000;
        let p = SourceProfile::measure(&src).unwrap();
        assert_eq!(p.audio_share_pct, 28);
        assert!(p.audio_heavy);

        let line = insight(&src, Strategy::VisuallyLossless, Codec::Av1).unwrap();
        assert!(line.contains("Audio is heavy"), "{line}");
        assert!(line.contains("1.5 Mbit/s"), "{line}");
        assert!(line.contains("28 %"), "{line}");

        // A normal 128 kbit/s AAC track (2 % of the file) stays silent.
        let mut light = lean_1080p();
        light.audio_bitrate = 128_000;
        assert!(!SourceProfile::measure(&light).unwrap().audio_heavy);
        assert!(!insight(&light, Strategy::Balanced, Codec::Av1)
            .unwrap()
            .contains("Audio is heavy"));
    }

    #[test]
    fn insight_names_the_character_the_density_and_the_ceiling() {
        let noisy = insight(
            &info(640, 360, 10, 31_250_000, 30.0, "h264"),
            Strategy::Balanced,
            Codec::Av1,
        )
        .unwrap();
        assert!(noisy.starts_with("Source: Noisy"), "{noisy}");
        assert!(
            noisy.contains("bpp") && noisy.contains("ceiling"),
            "{noisy}"
        );

        let lean = insight(&lean_1080p(), Strategy::VisuallyLossless, Codec::Av1).unwrap();
        assert!(lean.starts_with("Source: Lean"), "{lean}");
        assert!(lean.contains("5.2 Mbit/s"), "{lean}");

        // Unmeasurable sources say nothing rather than lie.
        assert_eq!(
            insight(&MediaInfo::default(), Strategy::Balanced, Codec::Av1),
            None
        );
    }

    #[test]
    fn opaque_sources_fall_back_to_the_theoretical_ceiling_only() {
        let unknown = MediaInfo::default();
        assert!(SourceProfile::measure(&unknown).is_none());
        assert!(
            plan_of(&unknown, Strategy::Balanced) >= 1,
            "never unbounded"
        );
    }
}
