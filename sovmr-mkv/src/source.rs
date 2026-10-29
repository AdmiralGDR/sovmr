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

//! Matroska tracks, restated in the terms the output muxer takes.
//!
//! The engine's output path is built around `sovmr_mp4::mux`, whose audio
//! tracks are described by an `mp4a` sample entry, a timescale, and an
//! identity. A Matroska track carries the same facts in different
//! shapes: its decoder setup is a bare `AudioSpecificConfig` rather than
//! an `esds`, and its language is a variable-length string rather than
//! three packed characters.
//!
//! This module does that restating and nothing else. It deliberately
//! stops short of touching the pipeline: converting the description is a
//! separate, testable job from rewiring the transcode loop, and the two
//! fail in different ways.
//!
//! # What can be carried depends on where it is going
//!
//! MP4 describes an audio track through an `esds`, which only speaks
//! MPEG-4 Audio — so only AAC survives that journey. Matroska describes
//! a track by a codec name and a blob, and therefore carries anything:
//! Opus, FLAC, AC-3, DTS, TrueHD, and subtitles besides.
//!
//! That asymmetry is the whole of it. Copying bytes into Matroska needs
//! no encoder, no decoder, and no judgement — and it saves tracks that
//! were being dropped with an honest warning, which is still a track the
//! user no longer has.

use crate::codec;
use crate::demux::{Segment, Track, TrackKind};

/// Language written when a track's own is not three ASCII letters.
///
/// MP4's `mdhd` has room for exactly three characters, while Matroska
/// allows BCP 47 tags like `de-AT`. Truncating `de-AT` to `de-` would
/// produce a language code that means nothing; `und` at least says
/// honestly that the answer did not fit.
pub const UNDETERMINED: [u8; 3] = *b"und";

/// Where a track is going, which decides what may go there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// ISOBMFF: only AAC, and it needs an `esds` built around it.
    Mp4,
    /// Matroska: any codec, copied verbatim.
    Mkv,
}

/// One source track, described the way the output muxer wants it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPassthrough {
    /// Matroska `TrackNumber`, for matching frames back to this track.
    pub track_number: u64,
    /// Track timescale for the output — the sampling rate, so a packet's
    /// duration is a whole number of samples.
    pub timescale: u32,
    /// A complete `mp4a` sample-entry payload built around the source's
    /// `AudioSpecificConfig` — `None` for a Matroska target, and for any
    /// codec MP4 could not have described anyway.
    pub entry_payload: Option<Vec<u8>>,
    /// The source's Matroska codec identifier, copied for a Matroska
    /// target.
    pub codec_id: String,
    /// Whether this is a subtitle track rather than audio. Matroska
    /// carries both the same way; MP4 carries neither but AAC.
    pub subtitle: bool,
    /// The bare `AudioSpecificConfig`, for a Matroska output.
    ///
    /// Carried alongside rather than unwrapped back out of
    /// `entry_payload` when needed: the two containers want the same
    /// bytes in different wrappers, and parsing our own wrapper to undo
    /// our own work would be a descriptor reader existing for no reason.
    pub codec_private: Vec<u8>,
    /// Channel count, which Matroska states directly.
    pub channels: u16,
    /// ISO 639-2 language, or [`UNDETERMINED`] when the source's tag
    /// cannot be expressed in three characters.
    pub language: [u8; 3],
    /// Human title, when the source names the track.
    pub name: Option<String>,
    /// Whether the track is enabled for playback.
    pub enabled: bool,
}

impl AudioPassthrough {
    /// The sampling rate to declare in a Matroska output.
    ///
    /// Zero for subtitles, which have none — writing a rate there would
    /// describe audio that does not exist.
    pub fn sample_rate_for_output(&self) -> u32 {
        if self.subtitle {
            0
        } else {
            self.timescale
        }
    }
}

/// Every audio track of a segment, split into what can be carried and
/// what cannot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AudioPlan {
    /// Tracks the MP4 muxer can re-emit bit-exactly, in file order.
    pub carried: Vec<AudioPassthrough>,
    /// Codec ids of audio tracks that cannot come along, one entry each.
    ///
    /// Named rather than dropped in silence, so the engine's existing
    /// warning path can tell the user exactly what was left behind.
    pub unsupported: Vec<String>,
}

/// Squeeze a Matroska language tag into `mdhd`'s three characters.
///
/// Only a plain three-letter ISO 639-2 code fits. A BCP 47 tag with a
/// region (`de-AT`) or a two-letter code does not, and inventing a
/// mapping would put a language on the track that the source never
/// claimed.
fn packed_language(tag: &str) -> [u8; 3] {
    let bytes = tag.as_bytes();
    if bytes.len() == 3 && bytes.iter().all(|b| b.is_ascii_lowercase()) {
        [bytes[0], bytes[1], bytes[2]]
    } else {
        UNDETERMINED
    }
}

/// Describe one track for the output muxer, or say why it cannot go.
fn passthrough_for(track: &Track<'_>, target: Target) -> Option<AudioPassthrough> {
    let subtitle = track.kind == TrackKind::Subtitle;
    match target {
        Target::Mp4 => {
            // `esds` describes MPEG-4 Audio and nothing else, so only
            // AAC can make this journey — and it needs its own
            // `CodecPrivate` to build one from.
            if subtitle || !codec::is_passthrough_audio(track.codec_id) {
                return None;
            }
            if track.codec_private.is_empty() {
                return None;
            }
            // A track declaring neither rate nor channels describes no
            // audio anyone could decode; refusing beats writing a header
            // full of zeros.
            if track.sample_rate == 0 || track.channels == 0 {
                return None;
            }
        }
        Target::Mkv => {
            // Matroska needs only a name and a blob, so anything the
            // source could describe, the output can too. A track with no
            // codec name at all is the one thing that cannot be copied:
            // there would be nothing to write in `CodecID`.
            if track.codec_id.is_empty() {
                return None;
            }
        }
    }
    Some(AudioPassthrough {
        track_number: track.number,
        // For subtitles there is no sampling rate; their timestamps are
        // the segment's, and the muxer works in milliseconds anyway.
        timescale: if subtitle {
            1_000
        } else {
            track.sample_rate.max(1)
        },
        entry_payload: (target == Target::Mp4).then(|| {
            sovmr_mp4::mux::build_mp4a_entry(
                track.sample_rate,
                track.channels,
                track.codec_private,
                0,
                0,
            )
        }),
        codec_id: track.codec_id.to_string(),
        subtitle,
        codec_private: track.codec_private.to_vec(),
        channels: track.channels,
        language: packed_language(track.language),
        name: track.name.map(str::to_string),
        enabled: track.enabled,
    })
}

/// Plan which of a segment's tracks reach the output.
///
/// Audio first, then subtitles, both in file order — the order a player
/// will offer them in.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), sovmr_mkv::demux::MkvError> {
/// # let bytes: Vec<u8> = Vec::new();
/// # if sovmr_mkv::is_matroska(&bytes) {
/// let segment = sovmr_mkv::demux::read_segment(&bytes)?;
/// let plan = sovmr_mkv::source::plan_audio(&segment, sovmr_mkv::source::Target::Mkv);
/// println!("{} carried, {} left behind", plan.carried.len(), plan.unsupported.len());
/// # }
/// # Ok(())
/// # }
/// ```
pub fn plan_audio(segment: &Segment<'_>, target: Target) -> AudioPlan {
    let mut plan = AudioPlan::default();
    for kind in [TrackKind::Audio, TrackKind::Subtitle] {
        for track in segment.of_kind(kind) {
            match passthrough_for(track, target) {
                Some(p) => plan.carried.push(p),
                None => plan.unsupported.push(track.codec_id.to_string()),
            }
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_track<'a>(
        number: u64,
        codec_id: &'a str,
        private: &'a [u8],
        language: &'a str,
    ) -> Track<'a> {
        Track {
            number,
            kind: TrackKind::Audio,
            codec_id,
            codec_private: private,
            language,
            name: None,
            enabled: true,
            default: true,
            forced: false,
            codec_delay: 0,
            default_duration: None,
            width: 0,
            height: 0,
            sample_rate: 48_000,
            channels: 2,
            colour: None,
        }
    }

    fn segment_of<'a>(tracks: Vec<Track<'a>>) -> Segment<'a> {
        Segment {
            timestamp_scale: 1_000_000,
            duration_ticks: 1000.0,
            title: None,
            tracks,
            chapters: Vec::new(),
            tags: Vec::new(),
            truncated: false,
            payload: &[],
        }
    }

    #[test]
    fn every_aac_track_becomes_a_carryable_description() {
        let asc = [0x11u8, 0x90];
        let seg = segment_of(vec![
            audio_track(1, "A_AAC", &asc, "eng"),
            audio_track(2, "A_AAC/MPEG4/LC", &asc, "rus"),
        ]);
        let plan = plan_audio(&seg, Target::Mp4);
        assert_eq!(plan.carried.len(), 2);
        assert!(plan.unsupported.is_empty());
        assert_eq!(plan.carried[0].language, *b"eng");
        assert_eq!(plan.carried[1].language, *b"rus");
        assert_eq!(plan.carried[1].track_number, 2);
        // The timescale must be the sampling rate, so a packet's
        // duration is a whole number of samples rather than a rounding.
        assert_eq!(plan.carried[0].timescale, 48_000);
        assert!(plan.carried[0]
            .entry_payload
            .as_deref()
            .unwrap()
            .windows(asc.len())
            .any(|w| w == asc));
    }

    #[test]
    fn a_codec_the_muxer_cannot_write_is_named_not_dropped() {
        // The engine already has a warning path for this; it only works
        // if the codec arrives with a name attached.
        let seg = segment_of(vec![
            audio_track(1, "A_OPUS", &[1, 2], "eng"),
            audio_track(2, "A_FLAC", &[1, 2], "eng"),
        ]);
        let plan = plan_audio(&seg, Target::Mp4);
        assert!(plan.carried.is_empty());
        assert_eq!(plan.unsupported, vec!["A_OPUS", "A_FLAC"]);
    }

    #[test]
    fn an_aac_track_without_its_config_cannot_be_carried() {
        // Writing an esds around nothing produces a track that describes
        // no decoder — worse than admitting it cannot come along.
        let seg = segment_of(vec![audio_track(1, "A_AAC", &[], "eng")]);
        let plan = plan_audio(&seg, Target::Mp4);
        assert!(plan.carried.is_empty());
        assert_eq!(plan.unsupported, vec!["A_AAC"]);
    }

    #[test]
    fn a_track_declaring_no_rate_or_channels_is_refused() {
        let asc = [0x11u8, 0x90];
        for (rate, channels) in [(0u32, 2u16), (48_000, 0)] {
            let mut t = audio_track(1, "A_AAC", &asc, "eng");
            t.sample_rate = rate;
            t.channels = channels;
            let plan = plan_audio(&segment_of(vec![t]), Target::Mp4);
            assert!(
                plan.carried.is_empty(),
                "rate {rate}, channels {channels} describes no audio"
            );
        }
    }

    #[test]
    fn a_language_that_does_not_fit_three_characters_becomes_und() {
        // MP4's mdhd holds exactly three; Matroska allows BCP 47.
        // Truncating "de-AT" to "de-" would name a language that does
        // not exist, which is worse than admitting the answer did not fit.
        assert_eq!(packed_language("eng"), *b"eng");
        assert_eq!(packed_language("de-AT"), UNDETERMINED);
        assert_eq!(packed_language("de"), UNDETERMINED);
        assert_eq!(packed_language(""), UNDETERMINED);
        assert_eq!(packed_language("ENG"), UNDETERMINED, "mdhd packs lowercase");
    }

    #[test]
    fn identity_travels_with_the_track() {
        let asc = [0x11u8, 0x90];
        let mut t = audio_track(7, "A_AAC", &asc, "jpn");
        t.name = Some("Director commentary");
        t.enabled = false;
        let plan = plan_audio(&segment_of(vec![t]), Target::Mp4);
        let p = &plan.carried[0];
        assert_eq!(p.track_number, 7);
        assert_eq!(p.name.as_deref(), Some("Director commentary"));
        assert!(!p.enabled, "a disabled track must not be silently promoted");
    }

    #[test]
    fn matroska_output_carries_codecs_mp4_would_have_to_drop() {
        // THE point of asking where a track is going. Opus, FLAC, AC-3
        // and friends were being dropped with an honest warning — which
        // is still a track the user no longer has — purely because an
        // `esds` cannot describe them. Matroska needs no `esds`.
        let seg = segment_of(vec![
            audio_track(1, "A_OPUS", &[1, 2], "eng"),
            audio_track(2, "A_FLAC", &[1, 2], "rus"),
            audio_track(3, "A_AC3", &[1, 2], "deu"),
            audio_track(4, "A_DTS", &[1, 2], "fra"),
            audio_track(5, "A_TRUEHD", &[1, 2], "jpn"),
        ]);
        let to_mkv = plan_audio(&seg, Target::Mkv);
        assert_eq!(to_mkv.carried.len(), 5, "Matroska carries them all");
        assert!(to_mkv.unsupported.is_empty());
        assert_eq!(to_mkv.carried[0].codec_id, "A_OPUS");
        assert!(
            to_mkv.carried[0].entry_payload.is_none(),
            "an mp4a entry would be meaningless here"
        );

        // The same file bound for MP4 loses every one of them, and says
        // so by name — the behaviour that must not regress.
        let to_mp4 = plan_audio(&seg, Target::Mp4);
        assert!(to_mp4.carried.is_empty());
        assert_eq!(to_mp4.unsupported.len(), 5);
    }

    #[test]
    fn subtitles_travel_into_matroska_and_never_into_mp4() {
        let mut sub = audio_track(9, "S_TEXT/UTF8", &[], "eng");
        sub.kind = TrackKind::Subtitle;
        let seg = segment_of(vec![audio_track(1, "A_AAC", &[0x11, 0x90], "eng"), sub]);

        let to_mkv = plan_audio(&seg, Target::Mkv);
        assert_eq!(to_mkv.carried.len(), 2);
        assert!(to_mkv.carried[1].subtitle, "the subtitle came along");
        assert_eq!(to_mkv.carried[1].codec_id, "S_TEXT/UTF8");

        let to_mp4 = plan_audio(&seg, Target::Mp4);
        assert_eq!(to_mp4.carried.len(), 1, "MP4 takes only the AAC");
        assert_eq!(to_mp4.unsupported, vec!["S_TEXT/UTF8"]);
    }

    #[test]
    fn a_track_with_no_codec_name_cannot_be_copied_anywhere() {
        // Matroska needs something to write in `CodecID`; there is no
        // honest placeholder for "we do not know what this is".
        let seg = segment_of(vec![audio_track(1, "", &[1, 2], "eng")]);
        assert!(plan_audio(&seg, Target::Mkv).carried.is_empty());
        assert!(plan_audio(&seg, Target::Mp4).carried.is_empty());
    }

    #[test]
    fn a_codec_matroska_carries_needs_no_rate_or_channels_declared() {
        // Those fields matter to an `mp4a` header. A Matroska copy just
        // needs the blob, so refusing on a missing rate would drop a
        // track the output could hold perfectly well.
        let mut t = audio_track(1, "A_TRUEHD", &[1, 2], "eng");
        t.sample_rate = 0;
        t.channels = 0;
        assert_eq!(
            plan_audio(&segment_of(vec![t]), Target::Mkv).carried.len(),
            1
        );
    }

    #[test]
    fn mixed_sources_carry_what_they_can_and_report_the_rest() {
        let asc = [0x11u8, 0x90];
        let seg = segment_of(vec![
            audio_track(1, "A_AAC", &asc, "eng"),
            audio_track(2, "A_AC3", &[1], "rus"),
            audio_track(3, "A_AAC", &asc, "deu"),
        ]);
        let plan = plan_audio(&seg, Target::Mp4);
        let numbers: Vec<u64> = plan.carried.iter().map(|p| p.track_number).collect();
        assert_eq!(numbers, vec![1, 3], "file order, minus what cannot come");
        assert_eq!(plan.unsupported, vec!["A_AC3"]);
    }
}
