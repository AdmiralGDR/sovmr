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

//! The unit of work. Deliberately backend-neutral: a `MediaJob` describes
//! *what* to produce, never *how* — no FFmpeg flags, no NVENC preset strings.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::config::{
    CQ_MAX, OUTPUT_TAG, SKIP_FLOOR_BITS_PER_PIXEL_SECOND, SKIP_MAX_BITS_PER_PIXEL_SECOND,
    VISUALLY_LOSSLESS_CQ,
};

/// Process-unique, monotonically increasing job identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(u64);

impl JobId {
    /// Mint the next identifier. Thread-safe; ids are unique within one
    /// process lifetime and ordered by creation.
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        JobId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "job#{}", self.0)
    }
}

/// Target codec, named by *format*, not by encoder implementation.
/// The adapter decides that `Av1` means `av1_nvenc` today and a native
/// Video Codec SDK session in Phase 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// AV1 — preferred on Ada-generation (RTX 40+) NVENC.
    Av1,
    /// HEVC (H.265) — the fallback for older NVENC generations.
    Hevc,
}

/// Encoder effort. Adapters map this onto their own scale
/// (FFmpeg/NVENC: p2 / p4 / p6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    /// Fastest encode, lowest compression efficiency.
    Speed,
    /// Middle ground for time-pressed batches.
    Balanced,
    /// Best quality-per-bit at a sensible speed (P6) — the batch default.
    Quality,
    /// The slowest preset the hardware offers (P7). Kept apart from
    /// [`Effort::Quality`] until a measurement says the extra time buys
    /// anything: adopted only if `--verify` shows higher quality at no
    /// larger a file. (`ULTRA_HIGH_QUALITY` tuning is deliberately not
    /// reachable here — it segfaults the encode on driver 580.173.)
    Maximum,
}

/// Validated constant-quality target (0–51, lower = higher fidelity).
/// Construction is the only way to obtain one, so an out-of-range CQ
/// cannot exist anywhere downstream (negative trust at the type level).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityTarget(u8);

impl QualityTarget {
    /// Validate and wrap a constant-quality value.
    ///
    /// # Errors
    ///
    /// Returns a human-readable message when `cq` exceeds
    /// [`CQ_MAX`](crate::config::CQ_MAX) — suitable for direct display next
    /// to the input field that produced it.
    ///
    /// # Example
    ///
    /// ```
    /// use sovmr_core::job::QualityTarget;
    /// assert_eq!(QualityTarget::new(25).unwrap().value(), 25);
    /// assert!(QualityTarget::new(52).is_err());
    /// ```
    pub fn new(cq: u8) -> Result<Self, String> {
        if cq > CQ_MAX {
            return Err(format!("cq {cq} out of range 0..={CQ_MAX}"));
        }
        Ok(QualityTarget(cq))
    }

    /// The project-defined visually lossless target
    /// ([`VISUALLY_LOSSLESS_CQ`](crate::config::VISUALLY_LOSSLESS_CQ)).
    pub fn visually_lossless() -> Self {
        QualityTarget(VISUALLY_LOSSLESS_CQ)
    }

    /// The validated CQ value, for adapters to pass to their encoder.
    pub fn value(&self) -> u8 {
        self.0
    }
}

/// What to do with non-video streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPolicy {
    /// Bit-exact passthrough (`-c:a copy` in the FFmpeg adapter).
    Copy,
}

/// Complete, backend-neutral description of one encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeSettings {
    /// Target video codec.
    pub codec: Codec,
    /// Speed/quality trade-off.
    pub effort: Effort,
    /// Semantic quality intent; the heuristic engine
    /// ([`crate::heuristics`]) translates it into CQ + a hard bitrate
    /// ceiling from the measured source.
    pub strategy: crate::heuristics::Strategy,
    /// Handling of audio streams.
    pub audio: AudioPolicy,
    /// Generation-loss protection: when `true` (the default), sources
    /// already in the target codec and already compact are marked
    /// `Skipped` instead of re-encoded (see
    /// [`already_optimal`]). Power users disable it to force a re-encode
    /// regardless — ultimate control stays with the human.
    pub smart_skip: bool,
}

impl EncodeSettings {
    /// The one-click "Visually Lossless (AV1 GPU)" preset.
    pub fn visually_lossless(codec: Codec) -> Self {
        EncodeSettings {
            codec,
            effort: Effort::Quality,
            strategy: crate::heuristics::Strategy::VisuallyLossless,
            audio: AudioPolicy::Copy,
            smart_skip: true,
        }
    }
}

/// Facts about a source file established by probing (never trusted from the
/// filename). `duration` drives percentage/ETA math and may legitimately be
/// unknown for damaged files that are still decodable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaInfo {
    /// Playback duration; `None` when the container would not reveal it.
    pub duration: Option<Duration>,
    /// Video width in pixels (0 when unknown).
    pub width: u32,
    /// Video height in pixels (0 when unknown).
    pub height: u32,
    /// Source video codec name as reported by the prober (e.g. `"h264"`).
    pub video_codec: String,
    /// Source file size in bytes, for compression-ratio reporting.
    pub size_bytes: u64,
    /// Coded bytes of the **video track alone**, or zero when the
    /// container cannot say.
    ///
    /// [`size_bytes`](Self::size_bytes) is the whole file: audio,
    /// subtitles, chapters and container overhead included. Judging
    /// video density by it makes a film with six lossless audio tracks
    /// look far denser than its pictures are, and every ceiling derived
    /// from that density is set for a video that does not exist.
    pub video_bytes: u64,
    /// Frames per second the container reports (0.0 = unknown) — one
    /// input to the heuristic engine's bits/pixel/frame math.
    pub fps: f64,
    /// Audio bitrate in bits/second (0 = no audio track, or unknown).
    /// Bit-exact passthrough means this bitrate lands in the output
    /// untouched, so a fat audio track caps the achievable ratio — the
    /// heuristic engine flags it (`SourceProfile::audio_heavy`).
    pub audio_bitrate: u64,
    /// Audio tracks that will be carried through bit-exactly.
    ///
    /// Shown to the user before the job runs, because "6 audio tracks"
    /// on the card is how someone confirms their six-language master
    /// was understood — and how they would notice if it were not.
    pub audio_tracks: usize,
    /// Subtitle tracks present in the source.
    pub subtitle_tracks: usize,
    /// Whether the source carries chapter marks.
    pub chapters: bool,
}

/// Smart Skip (generation-loss protection): decide whether re-encoding
/// this source into `target` would be a net loss. Returns a UI-ready
/// reason when the input is *already in the target codec* **and** its
/// average bit density is at or below
/// [`SKIP_MAX_BITS_PER_PIXEL_SECOND`](crate::config::SKIP_MAX_BITS_PER_PIXEL_SECOND);
/// `None` means "encode normally".
///
/// Conservative by design: any missing probe fact (unknown duration,
/// zero dimensions or size) disables the skip — when in doubt, encode.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use sovmr_core::job::{already_optimal, Codec, MediaInfo};
///
/// // 1080p AV1 averaging ~0.8 Mbit/s: re-encoding to AV1 is pure loss.
/// let compact_av1 = MediaInfo {
///     duration: Some(Duration::from_secs(100)),
///     width: 1920,
///     height: 1080,
///     video_codec: "av1".into(),
///     size_bytes: 10_000_000,
///     fps: 30.0,
///     audio_bitrate: 0,
///     ..Default::default()
/// };
/// assert!(already_optimal(&compact_av1, Codec::Av1).is_some());
/// // The same file targeted at HEVC is a codec change — never skipped.
/// assert!(already_optimal(&compact_av1, Codec::Hevc).is_none());
/// ```
pub fn already_optimal(info: &MediaInfo, target: Codec) -> Option<String> {
    let seconds = info.duration?.as_secs_f64();
    if seconds <= 0.0 || info.width == 0 || info.height == 0 || info.size_bytes == 0 {
        return None;
    }
    let pixels = f64::from(info.width) * f64::from(info.height);
    let bits_per_pixel_second = (info.size_bytes as f64 * 8.0) / seconds / pixels;

    // Cross-codec floor: a source already leaner than anything our
    // quality-driven encode could produce cannot shrink — re-encoding
    // it would only bloat the file and stack generation loss.
    if bits_per_pixel_second <= SKIP_FLOOR_BITS_PER_PIXEL_SECOND {
        return Some(format!(
            "already extremely compact at {bits_per_pixel_second:.3} b/(px·s) — \
             a re-encode could not shrink it"
        ));
    }

    let name = info.video_codec.to_ascii_lowercase();
    let same_codec = match target {
        Codec::Av1 => name == "av1",
        Codec::Hevc => name == "hevc" || name == "h265",
    };
    if !same_codec {
        return None;
    }
    (bits_per_pixel_second <= SKIP_MAX_BITS_PER_PIXEL_SECOND).then(|| {
        format!(
            "already {} at {bits_per_pixel_second:.2} b/(px·s) — \
             re-encoding would only lose quality",
            info.video_codec
        )
    })
}

/// One entry in the batch queue.
#[derive(Debug, Clone)]
pub struct MediaJob {
    /// Unique identity, referenced by every event about this job.
    pub id: JobId,
    /// Source file (already validated and probed).
    pub input: PathBuf,
    /// Final destination; backends write `<output>.part` then rename.
    pub output: PathBuf,
    /// How to encode it.
    pub settings: EncodeSettings,
    /// Probed facts about the source, used for progress/ETA math.
    pub info: MediaInfo,
}

impl MediaJob {
    /// Derive the output path next to the input: `clip.mp4` -> `clip.sovmr.mp4`.
    /// Guarantees the output never collides with the input file.
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use sovmr_core::job::MediaJob;
    /// let out = MediaJob::derive_output(Path::new("/media/movie.mp4"));
    /// assert_eq!(out, Path::new("/media/movie.sovmr.mp4"));
    /// ```
    pub fn derive_output(input: &Path) -> PathBuf {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("mp4");
        input.with_file_name(format!("{stem}.{OUTPUT_TAG}.{ext}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_ids_are_unique_and_ordered() {
        let a = JobId::next();
        let b = JobId::next();
        assert!(b > a);
    }

    #[test]
    fn quality_target_rejects_out_of_range() {
        assert!(QualityTarget::new(52).is_err());
        assert_eq!(QualityTarget::new(25).unwrap().value(), 25);
        assert_eq!(QualityTarget::visually_lossless().value(), 25);
    }

    fn probe(codec: &str, mbytes: u64, secs: u64) -> MediaInfo {
        MediaInfo {
            duration: Some(Duration::from_secs(secs)),
            width: 1920,
            height: 1080,
            video_codec: codec.to_string(),
            size_bytes: mbytes * 1_000_000,
            fps: 30.0,
            audio_bitrate: 0,
            ..Default::default()
        }
    }

    #[test]
    fn skips_compact_file_already_in_target_codec() {
        // 10 MB / 100 s / 2.07 Mpx = 0.39 b/(px·s) — deep below threshold.
        let reason = already_optimal(&probe("av1", 10, 100), Codec::Av1);
        assert!(reason.as_deref().is_some_and(|r| r.contains("already av1")));
        assert!(already_optimal(&probe("hevc", 10, 100), Codec::Hevc).is_some());
    }

    #[test]
    fn never_skips_codec_changes_or_fat_files() {
        // Different codec: h264 -> AV1 is the whole point of the app.
        assert!(already_optimal(&probe("h264", 10, 100), Codec::Av1).is_none());
        // AV1 -> HEVC is a codec change even though the source is AV1.
        assert!(already_optimal(&probe("av1", 10, 100), Codec::Hevc).is_none());
        // Same codec but fat: 500 MB / 100 s = 19 b/(px·s) — re-encode it.
        assert!(already_optimal(&probe("av1", 500, 100), Codec::Av1).is_none());
    }

    #[test]
    fn missing_probe_facts_disable_the_skip() {
        let mut info = probe("av1", 10, 100);
        info.duration = None;
        assert!(already_optimal(&info, Codec::Av1).is_none());
        let mut info = probe("av1", 10, 100);
        info.width = 0;
        assert!(already_optimal(&info, Codec::Av1).is_none());
        let mut info = probe("av1", 0, 100);
        info.size_bytes = 0;
        assert!(already_optimal(&info, Codec::Av1).is_none());
    }

    #[test]
    fn output_path_never_equals_input() {
        let input = Path::new("/media/movie.mp4");
        let out = MediaJob::derive_output(input);
        assert_eq!(out, Path::new("/media/movie.sovmr.mp4"));
        assert_ne!(out, input);
    }

    #[test]
    fn output_path_handles_missing_extension() {
        let out = MediaJob::derive_output(Path::new("/media/rawclip"));
        assert_eq!(out, Path::new("/media/rawclip.sovmr.mp4"));
    }
}
