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

//! A summary of a Matroska file, cheap enough to run on a queued job.
//!
//! # Why the audio figure is sampled rather than counted
//!
//! MP4 stores a per-sample size table, so the exact byte total of a
//! track is arithmetic on the header. Matroska stores no such thing: the
//! only way to *count* audio bytes is to walk every cluster in the file.
//!
//! On a fifty-gigabyte source that turns a queue-time probe into a full
//! read. So this walks a bounded prefix — [`SAMPLE_SECONDS`] of media —
//! and scales the result by the declared duration. The number feeds the
//! heuristic engine's "audio is heavy" warning, which is a threshold,
//! not an invoice; being a few percent out changes nothing, and taking
//! minutes to be exact would change the whole feel of adding a file.
//!
//! It is labelled an estimate in the type, so nothing downstream can
//! mistake it for a measurement.

use crate::blocks;
use crate::codec;
use crate::demux::{self, MkvError, TrackKind};

/// Media seconds examined when estimating audio weight.
///
/// Long enough to average over a few hundred packets of every track,
/// short enough that the walk touches a few megabytes rather than the
/// whole file.
pub const SAMPLE_SECONDS: f64 = 30.0;

/// Nanoseconds in a second, for turning Matroska's clock into wall time.
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// What a queued Matroska job looks like from the outside.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Wall-clock duration, when the file declares one.
    pub duration_secs: f64,
    /// Coded width and height of the first video track.
    pub width: u32,
    /// Coded height.
    pub height: u32,
    /// Video codec in the engine's spelling (`h264`, `hevc`, `av1`), or
    /// empty when there is no video track this engine can decode.
    pub video_codec: &'static str,
    /// Frames per second, from the track's declared frame duration.
    pub fps: f64,
    /// Audio tracks the MP4 muxer could carry through bit-exactly.
    pub audio_tracks: usize,
    /// Audio tracks present whose codec cannot be re-emitted, by name.
    pub unsupported_audio: Vec<String>,
    /// Subtitle tracks present.
    pub subtitle_tracks: usize,
    /// Combined audio bitrate in bits per second — **an estimate**, see
    /// the module note.
    pub estimated_audio_bitrate: u64,
    /// Whether the file carries chapter marks.
    ///
    /// Read, not guessed. Until the reader parsed the `Chapters`
    /// element this reported a flat `false`, which was honest while
    /// nothing looked and became a lie the moment something did.
    pub chapters: bool,
}

/// Summarise a Matroska file.
///
/// # Errors
///
/// As [`demux::read_segment`].
pub fn summarise(file: &[u8]) -> Result<Summary, MkvError> {
    let segment = demux::read_segment(file)?;
    let duration_secs = segment.duration_ticks * segment.timestamp_scale as f64 / NANOS_PER_SECOND;

    let video = segment
        .of_kind(TrackKind::Video)
        .find(|t| codec::video_codec(t.codec_id).is_some());
    let (width, height, video_codec, fps) = match video {
        Some(t) => {
            let name = match codec::video_codec(t.codec_id) {
                Some(sovmr_mp4::demux::VideoCodec::H264) => "h264",
                Some(sovmr_mp4::demux::VideoCodec::Hevc) => "hevc",
                Some(sovmr_mp4::demux::VideoCodec::Av1) => "av1",
                None => "",
            };
            // `DefaultDuration` is nanoseconds per frame. A file that
            // omits it leaves fps unknown rather than inviting a guess —
            // the heuristic engine has its own fallback.
            let fps = t
                .default_duration
                .filter(|d| *d > 0)
                .map(|d| NANOS_PER_SECOND / d as f64)
                .unwrap_or(0.0);
            (t.width, t.height, name, fps)
        }
        None => (0, 0, "", 0.0),
    };

    let mut audio_tracks = 0usize;
    let mut unsupported_audio = Vec::new();
    for t in segment.of_kind(TrackKind::Audio) {
        if codec::is_passthrough_audio(t.codec_id) {
            audio_tracks += 1;
        } else {
            unsupported_audio.push(t.codec_id.to_string());
        }
    }

    Ok(Summary {
        duration_secs,
        width,
        height,
        video_codec,
        fps,
        audio_tracks,
        unsupported_audio,
        subtitle_tracks: segment.of_kind(TrackKind::Subtitle).count(),
        estimated_audio_bitrate: estimate_audio_bitrate(&segment, duration_secs),
        chapters: !segment.chapters.is_empty(),
    })
}

/// Estimate the combined audio bitrate from a bounded prefix.
///
/// Returns zero when the file declares no duration, rather than
/// extrapolating from nothing.
fn estimate_audio_bitrate(segment: &demux::Segment<'_>, duration_secs: f64) -> u64 {
    if duration_secs <= 0.0 {
        return 0;
    }
    let audio: Vec<u64> = segment
        .of_kind(TrackKind::Audio)
        .map(|t| t.number)
        .collect();
    if audio.is_empty() {
        return 0;
    }
    // Ticks covered by the sample window, in the segment's own clock.
    let window_ticks = (SAMPLE_SECONDS * NANOS_PER_SECOND / segment.timestamp_scale as f64) as i64;

    let (frames, _) = blocks::read_frames(segment.payload);
    let mut bytes = 0u64;
    let mut last = 0i64;
    for f in &frames {
        if f.timestamp > window_ticks {
            break;
        }
        if audio.contains(&u64::from(f.track)) {
            bytes += f.data.len() as u64;
            last = last.max(f.timestamp);
        }
    }
    // How much media the sample actually covered — a file shorter than
    // the window is measured in full, not scaled up from a fraction.
    let covered = (last as f64 * segment.timestamp_scale as f64 / NANOS_PER_SECOND)
        .min(duration_secs)
        .max(f64::EPSILON);
    if bytes == 0 {
        return 0;
    }
    (bytes as f64 * 8.0 / covered) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_that_declares_no_duration_estimates_nothing() {
        // Extrapolating from an unknown span would invent a number the
        // heuristic engine would then act on.
        let seg = demux::Segment {
            timestamp_scale: 1_000_000,
            duration_ticks: 0.0,
            title: None,
            tracks: Vec::new(),
            chapters: Vec::new(),
            tags: Vec::new(),
            truncated: false,
            payload: &[],
        };
        assert_eq!(estimate_audio_bitrate(&seg, 0.0), 0);
    }

    #[test]
    fn a_file_with_no_audio_estimates_nothing() {
        let seg = demux::Segment {
            timestamp_scale: 1_000_000,
            duration_ticks: 1000.0,
            title: None,
            tracks: Vec::new(),
            chapters: Vec::new(),
            tags: Vec::new(),
            truncated: false,
            payload: &[],
        };
        assert_eq!(estimate_audio_bitrate(&seg, 1.0), 0);
    }
}
