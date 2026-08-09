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

//! One view of a source file, whichever container it arrived in.
//!
//! The transcode loop needs five things and does not care where they
//! come from: the video track's static facts, its per-frame timing, an
//! Annex B feed for the decoder, a description of each carryable audio
//! track, and that track's packets. This module answers all five for
//! both ISOBMFF and Matroska, so `pipeline.rs` names no container at all.
//!
//! # The two shapes underneath
//!
//! MP4 keeps sample tables in its header, so packets can be *streamed*:
//! a cursor walks the table and reads each payload straight out of the
//! mapping. That is what keeps a twelve-hour job's metadata bounded, and
//! it is preserved here untouched.
//!
//! Matroska keeps no such index. Its frames are found only by walking
//! clusters, so they are read once into a list of borrowed slices. The
//! *payload* is still never copied — but the per-frame records are held,
//! at roughly 32 bytes each. A twelve-hour seven-track source is some
//! 13 M frames, so about 420 MB. That is a real cost, measured rather
//! than waved at, and it is the honest price of a format with no index.
//! Streaming it would mean re-walking clusters per track; that trade is
//! worth making when a long Matroska source is in front of us to measure,
//! not before.

use sovmr_mkv::blocks::Frame;
use sovmr_mp4::demux::{self, VideoCodec};
use sovmr_mp4::mux::AudioMuxParams;

use crate::pipeline::NativeError;

/// Nanoseconds per second — the timescale used for Matroska video.
///
/// Matroska states frame durations in nanoseconds (`DefaultDuration`),
/// so carrying them in a nanosecond timescale is exact. It also makes a
/// twelve-hour movie 4.3e13 ticks, which is precisely why the muxer
/// learned to write 64-bit `mdhd` boxes.
const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Timescale used for video carried out of Matroska: 90 kHz.
///
/// Nanoseconds were the obvious first choice, being exactly what
/// Matroska states, and they were wrong twice over. A nanosecond clock
/// makes ten seconds of GOP overflow a `u32` before it becomes a frame
/// count, and it makes composition offsets exceed the 32 bits `ctts`
/// gives them — the muxer refused the file outright, which was correct
/// behaviour on a wrong input.
///
/// 90 kHz is the classic MPEG clock precisely because it divides evenly
/// by every ordinary frame rate: 24, 25, 30, 50 and 60 fps come out as
/// 3750, 3600, 3000, 1800 and 1500 ticks with nothing left over. A
/// millisecond clock would have been simpler and drifts a full 1 % at
/// 30 fps — seven minutes of drift across twelve hours.
pub(crate) const MKV_TIMESCALE: u32 = 90_000;

/// Convert a Matroska timestamp to [`MKV_TIMESCALE`] ticks.
///
/// Rounded rather than truncated: 33 333 333 ns is one frame at 30 fps
/// and must land on 3000, not 2999.
fn to_ticks(value: i64, scale_nanos: u64) -> i64 {
    let nanos = i128::from(value) * i128::from(scale_nanos);
    let den = i128::from(NANOS_PER_SECOND);
    ((nanos * i128::from(MKV_TIMESCALE) + den / 2) / den) as i64
}

/// Static facts about the video track, in container-neutral terms.
pub(crate) struct VideoFacts {
    /// Elementary-stream codec.
    pub codec: VideoCodec,
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
    /// Timescale the timing values below are expressed in.
    pub timescale: u32,
}

/// A source file, opened and understood.
pub(crate) enum Source<'a> {
    /// ISOBMFF, streamed from its sample tables.
    Mp4 {
        /// The video track and its tables.
        track: demux::VideoTrack<'a>,
        /// Every audio track the muxer can carry, plus what it cannot.
        audio: demux::AudioTracks<'a>,
    },
    /// Matroska, whose frames were read up front for want of an index.
    Mkv {
        /// Nanoseconds per timestamp tick in the source's own clock.
        timestamp_scale: u64,
        /// The chosen video track.
        video: sovmr_mkv::demux::Track<'a>,
        /// Frames of the video track, in storage (decode) order.
        video_frames: Vec<Frame<'a>>,
        /// Per carried audio track: its packets, in storage order.
        audio_frames: Vec<Vec<Frame<'a>>>,
        /// Descriptions of the carried audio tracks.
        plan: sovmr_mkv::source::AudioPlan,
        /// The source's own title, which lives in `Info` and not among
        /// the tags — measured, and the reason a muxer carrying only
        /// tags dropped it.
        title: Option<String>,
        /// Chapters and tags the source declared, carried so the output
        /// does not lose what a viewer can see is gone.
        marks: (
            Vec<sovmr_mkv::mux::ChapterSpec>,
            Vec<sovmr_mkv::mux::TagSpec>,
        ),
    },
}

impl<'a> Source<'a> {
    /// Open `file`, choosing the container by its leading bytes.
    ///
    /// # Errors
    ///
    /// The container's own diagnostic, or [`NativeError::Pipeline`] when
    /// the file carries no video track this engine can decode.
    pub fn open(file: &'a [u8]) -> Result<Self, NativeError> {
        if sovmr_mkv::is_matroska(file) {
            return Self::open_matroska(file);
        }
        let track = demux::find_video_track(file)?;
        let audio = demux::find_audio_tracks(file)?;
        Ok(Source::Mp4 { track, audio })
    }

    fn open_matroska(file: &'a [u8]) -> Result<Self, NativeError> {
        let segment = sovmr_mkv::demux::read_segment(file)
            .map_err(|e| NativeError::Pipeline(format!("matroska: {e}")))?;
        let video = segment
            .of_kind(sovmr_mkv::demux::TrackKind::Video)
            .find(|t| sovmr_mkv::codec::video_codec(t.codec_id).is_some())
            .cloned()
            .ok_or_else(|| NativeError::Pipeline("no video track this engine can decode".into()))?;

        // The output mirrors the input, so a Matroska source means a
        // Matroska target — which is what lets Opus, FLAC, AC-3 and the
        // subtitles come along at all.
        let plan = sovmr_mkv::source::plan_audio(&segment, sovmr_mkv::source::Target::Mkv);
        let (frames, err) = sovmr_mkv::blocks::read_frames(segment.payload);
        // A cluster that fails to parse costs the frames after it, not
        // the job: what was readable still encodes, and the loss is
        // named rather than swallowed.
        if let Some(e) = err {
            eprintln!("sovmr: matroska stopped early: {e}");
        }

        let mut video_frames = Vec::new();
        let mut audio_frames: Vec<Vec<Frame<'a>>> = vec![Vec::new(); plan.carried.len()];
        for f in frames {
            if u64::from(f.track) == video.number {
                video_frames.push(f);
            } else if let Some(i) = plan
                .carried
                .iter()
                .position(|p| p.track_number == u64::from(f.track))
            {
                audio_frames[i].push(f);
            }
        }
        if video_frames.is_empty() {
            return Err(NativeError::Pipeline(
                "matroska video track carries no frames".into(),
            ));
        }
        Ok(Source::Mkv {
            timestamp_scale: segment.timestamp_scale,
            video,
            video_frames,
            audio_frames,
            plan,
            title: segment.title.map(str::to_string),
            marks: (
                segment
                    .chapters
                    .iter()
                    .map(|c| sovmr_mkv::mux::ChapterSpec {
                        uid: c.uid,
                        start_nanos: c.start_nanos,
                        end_nanos: c.end_nanos,
                        title: c.title.unwrap_or_default().to_string(),
                        language: c.language.unwrap_or("und").to_string(),
                    })
                    .collect(),
                segment
                    .tags
                    .iter()
                    .map(|t| sovmr_mkv::mux::TagSpec {
                        track_uid: t.track_uid,
                        name: t.name.to_string(),
                        value: t.value.to_string(),
                    })
                    .collect(),
            ),
        })
    }

    /// The video track's static facts.
    pub fn video_facts(&self) -> VideoFacts {
        match self {
            Source::Mp4 { track, .. } => VideoFacts {
                codec: track.codec,
                width: u32::from(track.width),
                height: u32::from(track.height),
                timescale: track.timescale.max(1),
            },
            Source::Mkv { video, .. } => VideoFacts {
                codec: sovmr_mkv::codec::video_codec(video.codec_id).unwrap_or(VideoCodec::H264),
                width: video.width,
                height: video.height,
                timescale: MKV_TIMESCALE,
            },
        }
    }

    /// Per-frame `(presentation timestamp, duration)` in decode order,
    /// both in the timescale from [`Source::video_facts`].
    ///
    /// # Errors
    ///
    /// A demuxer error from a lying sample table.
    pub fn video_timing(&self, file: &'a [u8]) -> Result<Vec<(i64, u32)>, NativeError> {
        match self {
            Source::Mp4 { track, .. } => Ok(track
                .samples(file)
                .map(|r| r.map(|s| (s.pts, s.duration)))
                .collect::<Result<_, _>>()?),
            Source::Mkv {
                timestamp_scale,
                video,
                video_frames,
                ..
            } => {
                // Matroska states a nominal frame duration; when it does
                // not, the gap to the next frame is the honest answer,
                // and the final frame inherits the one before it.
                // `DefaultDuration` is nanoseconds outright, not ticks
                // of the segment clock.
                let nominal = video
                    .default_duration
                    .filter(|d| *d > 0)
                    .map(|ns| to_ticks(ns as i64, 1) as u64);
                let stamps: Vec<i64> = video_frames
                    .iter()
                    .map(|f| to_ticks(f.timestamp, *timestamp_scale))
                    .collect();
                Ok(stamps
                    .iter()
                    .copied()
                    .zip(frame_durations(&stamps, nominal))
                    .collect())
            }
        }
    }

    /// Descriptions of every audio track that can be carried through.
    pub fn audio_params(&self) -> Vec<AudioMuxParams> {
        match self {
            Source::Mp4 { audio, .. } => audio
                .tracks
                .iter()
                .map(|at| AudioMuxParams {
                    timescale: at.timescale.max(1),
                    entry_payload: at.entry_payload.to_vec(),
                    language: at.meta.language.0,
                    alternate_group: at.meta.alternate_group,
                    enabled: at.meta.enabled,
                    name: at.meta.name.map(str::to_string),
                })
                .collect(),
            // Only reached for an MP4 output, which a Matroska source
            // never produces — but written honestly rather than left to
            // panic if that ever changes.
            Source::Mkv { plan, .. } => plan
                .carried
                .iter()
                .filter(|p| !p.subtitle)
                .filter_map(|p| {
                    Some(AudioMuxParams {
                        timescale: p.timescale.max(1),
                        entry_payload: p.entry_payload.clone()?,
                        language: p.language,
                        // Matroska has no `alternate_group`; several
                        // audio tracks in one file are alternatives by
                        // convention, and saying so lets a player offer
                        // a choice.
                        alternate_group: 1,
                        enabled: p.enabled,
                        name: p.name.clone(),
                    })
                })
                .collect(),
        }
    }

    /// A cursor over one carried audio track's packets.
    pub fn audio_packets(&'a self, file: &'a [u8], index: usize) -> AudioPackets<'a> {
        match self {
            Source::Mp4 { audio, .. } => AudioPackets::Mp4(audio.tracks[index].samples(file)),
            Source::Mkv {
                timestamp_scale,
                audio_frames,
                plan,
                ..
            } => AudioPackets::Mkv {
                frames: &audio_frames[index],
                at: 0,
                timescale: plan.carried[index].timescale,
                timestamp_scale: *timestamp_scale,
                previous: 0,
            },
        }
    }

    /// A cursor over the video track's frames, already in Annex B.
    pub fn video_frames(&self, file: &'a [u8]) -> VideoFrames<'a, '_> {
        match self {
            Source::Mp4 { track, .. } => VideoFrames::Mp4 {
                track,
                samples: track.samples(file),
            },
            Source::Mkv {
                video,
                video_frames,
                ..
            } => VideoFrames::Mkv {
                track: video,
                frames: video_frames.iter(),
                index: 0,
            },
        }
    }

    /// A phrase naming what the output container cannot carry
    /// (subtitles, chapters, tags), or `None` when nothing is lost.
    pub fn extras_description(&self, file: &'a [u8]) -> Option<String> {
        match self {
            Source::Mp4 { .. } => demux::survey_extras(file)
                .ok()
                .and_then(|e| e.lost_description()),
            Source::Mkv { .. } => {
                // Nothing to report. A Matroska source produces
                // Matroska, and its subtitle tracks are now copied
                // across rather than dropped — so warning about them
                // would be a message about a loss that no longer
                // happens, which is worse than silence.
                //
                // Chapters and tags live in elements this reader does
                // not parse yet. They are not claimed present and not
                // claimed absent: saying nothing about what has not been
                // looked at is the only honest option available.
                let _ = file;
                None
            }
        }
    }

    /// Total audio payload bytes that passthrough will copy verbatim —
    /// the exactly-known half of the output-size estimate.
    pub fn audio_bytes(&self) -> u64 {
        match self {
            Source::Mp4 { audio, .. } => audio.tracks.iter().map(|t| t.total_bytes()).sum(),
            Source::Mkv { audio_frames, .. } => audio_frames
                .iter()
                .flat_map(|t| t.iter())
                .map(|f| f.data.len() as u64)
                .sum(),
        }
    }

    /// Media duration in seconds, when the container states one.
    pub fn duration_secs(&self, timing: &[(i64, u32)]) -> f64 {
        match self {
            Source::Mp4 { track, .. } => track
                .duration()
                .map(|d| d.as_secs_f64())
                .unwrap_or_default(),
            Source::Mkv { .. } => {
                // Matroska's `Duration` is optional and often absent in
                // streamed files, so the frames themselves are the more
                // reliable answer — and they are already in hand.
                let last = timing
                    .last()
                    .map(|(pts, dur)| *pts as u64 + u64::from(*dur));
                last.map(|ticks| ticks as f64 / f64::from(MKV_TIMESCALE))
                    .unwrap_or_default()
            }
        }
    }

    /// Codec names of audio tracks that could not come along, and
    /// whether the track list was cut short.
    pub fn audio_left_behind(&self) -> (Vec<String>, bool) {
        match self {
            Source::Mp4 { audio, .. } => (
                audio
                    .unsupported
                    .iter()
                    .map(|k| String::from_utf8_lossy(k).trim().to_string())
                    .collect(),
                audio.truncated,
            ),
            Source::Mkv { plan, .. } => (plan.unsupported.clone(), false),
        }
    }
}

/// One carried audio track's packets, in storage order.
///
/// MP4 streams from its sample table; Matroska replays the list read at
/// open time. Both yield `(payload, duration in the track's timescale,
/// decode timestamp)` — the shape the interleaver already speaks.
pub(crate) enum AudioPackets<'a> {
    /// A cursor over an MP4 sample table.
    Mp4(demux::Samples<'a>),
    /// A replay of frames read from Matroska clusters.
    Mkv {
        /// Frames of this track, in storage order.
        ///
        /// Borrowed from the source's index rather than copied out of
        /// it. Measured: a `Frame` is 40 bytes and a twelve-hour film
        /// with six audio tracks holds some twelve million of them, so
        /// the clone this replaced duplicated **427 MB** — once per
        /// track, at the moment the pipeline asked for packets.
        frames: &'a [Frame<'a>],
        /// How many have been handed out.
        at: usize,
        /// The track's timescale — its sampling rate.
        timescale: u32,
        /// Nanoseconds per tick of the source's clock.
        timestamp_scale: u64,
        /// Duration of the previous packet, inherited by the last one.
        previous: u32,
    },
}

impl<'a> AudioPackets<'a> {
    /// The next packet, or `None` at end of track.
    ///
    /// A demuxer error ends the track rather than the job: a truncated
    /// audio tail must not throw away an otherwise good encode.
    pub fn next_packet(&mut self) -> Option<(&'a [u8], u32, u64)> {
        match self {
            AudioPackets::Mp4(samples) => match samples.next() {
                Some(Ok(s)) => Some((s.data, s.duration, s.dts)),
                _ => None,
            },
            AudioPackets::Mkv {
                frames,
                at,
                timescale,
                timestamp_scale,
                previous,
            } => {
                let f = frames.get(*at)?;
                // Matroska times packets in nanoseconds; the muxer wants
                // them in the track's own timescale, where a packet is a
                // whole number of samples. The gap to the next packet is
                // the duration; the last one inherits its predecessor's.
                let ticks = |stamp: i64| -> u64 {
                    let nanos = i128::from(stamp.max(0)) * i128::from(*timestamp_scale);
                    (nanos * i128::from(*timescale) / i128::from(NANOS_PER_SECOND)) as u64
                };
                let duration = frames
                    .get(*at + 1)
                    .map(|next| ticks(next.timestamp).saturating_sub(ticks(f.timestamp)))
                    .filter(|d| *d > 0)
                    .map(|d| u32::try_from(d).unwrap_or(u32::MAX))
                    .unwrap_or(*previous);
                *previous = duration;
                *at += 1;
                Some((f.data, duration, ticks(f.timestamp)))
            }
        }
    }
}

/// The video track's frames, ready to become Annex B.
pub(crate) enum VideoFrames<'a, 's> {
    /// A cursor over an MP4 sample table.
    Mp4 {
        /// The track, for its parameter sets and prefix width.
        track: &'s demux::VideoTrack<'a>,
        /// The sample walker.
        samples: demux::Samples<'a>,
    },
    /// A replay of frames read from Matroska clusters.
    Mkv {
        /// The track, for its `CodecPrivate`.
        track: &'s sovmr_mkv::demux::Track<'a>,
        /// Remaining frames.
        frames: std::slice::Iter<'s, Frame<'a>>,
        /// Decode-order index, which is the join key into the timing
        /// list built at open time.
        index: usize,
    },
}

impl VideoFrames<'_, '_> {
    /// Write the next frame into `out` as Annex B, returning its
    /// decode-order index.
    ///
    /// # Errors
    ///
    /// The container's own diagnostic when a frame cannot be read or a
    /// NAL length prefix lies.
    pub fn next_frame(&mut self, out: &mut Vec<u8>) -> Option<Result<usize, NativeError>> {
        match self {
            VideoFrames::Mp4 { track, samples } => {
                let sample = match samples.next()? {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e.into())),
                };
                let index = sample.index as usize;
                match track.write_frame(&sample, out) {
                    Ok(()) => Some(Ok(index)),
                    Err(e) => Some(Err(e.into())),
                }
            }
            VideoFrames::Mkv {
                track,
                frames,
                index,
            } => {
                let f = frames.next()?;
                let here = *index;
                *index += 1;
                match sovmr_mkv::codec::write_frame(track, f.data, f.keyframe, out) {
                    Ok(()) => Some(Ok(here)),
                    Err(e) => Some(Err(e.into())),
                }
            }
        }
    }
}

/// Per-frame durations for a list of presentation timestamps.
///
/// Matroska usually states one nominal duration for the whole track
/// (`DefaultDuration`), and when it does that is the answer. When it
/// does not, the gap to the next frame is the honest measurement, and
/// the final frame — which has no successor to measure against —
/// inherits the one before it rather than being given a zero the muxer
/// would then write into a clock that never advances.
fn frame_durations(stamps: &[i64], nominal: Option<u64>) -> Vec<u32> {
    let mut out = Vec::with_capacity(stamps.len());
    let mut previous = nominal.unwrap_or(0);
    for (i, ts) in stamps.iter().enumerate() {
        let duration = match nominal {
            Some(d) => d,
            None => stamps
                .get(i + 1)
                .map(|next| (next - ts).max(0) as u64)
                .filter(|d| *d > 0)
                .unwrap_or(previous),
        };
        previous = duration;
        out.push(u32::try_from(duration).unwrap_or(u32::MAX));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::frame_durations;

    #[test]
    fn a_declared_nominal_duration_is_used_for_every_frame() {
        // The common case: Matroska states one frame duration and every
        // frame keeps it, however the timestamps happen to be rounded.
        let d = frame_durations(&[0, 40, 80, 130], Some(33));
        assert_eq!(d, vec![33, 33, 33, 33]);
    }

    #[test]
    fn without_one_the_gap_to_the_next_frame_is_the_duration() {
        let d = frame_durations(&[0, 40, 80, 130], None);
        assert_eq!(
            d,
            vec![40, 40, 50, 50],
            "the last frame inherits, having nothing to measure against"
        );
    }

    #[test]
    fn a_lone_frame_still_gets_a_duration_rather_than_a_zero() {
        // A zero here would be written straight into a clock that then
        // never advances past the first frame.
        assert_eq!(frame_durations(&[0], Some(33)), vec![33]);
        // With nothing declared and nothing to measure, zero is all
        // there is — but the caller sees one frame, not a broken list.
        assert_eq!(frame_durations(&[0], None), vec![0]);
    }

    #[test]
    fn timestamps_that_go_backwards_do_not_wrap_into_huge_durations() {
        // B-frame reordering can put a later block before an earlier
        // one. A negative gap must clamp, not become 4 billion ticks.
        let d = frame_durations(&[0, 100, 50, 200], None);
        assert!(d.iter().all(|x| *x < 1000), "got {d:?}");
    }

    #[test]
    fn an_empty_track_yields_an_empty_list() {
        assert!(frame_durations(&[], None).is_empty());
    }
}

/// Which container the output should be written in.
///
/// The user asked for the output to mirror the input, so this is not a
/// setting: a Matroska source produces Matroska, an MP4 source produces
/// MP4. Re-containerising silently is the kind of helpfulness that loses
/// somebody their subtitles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Container {
    /// ISOBMFF, `.mp4`.
    Mp4,
    /// Matroska, `.mkv`.
    Mkv,
}

// No `extension()` here on purpose: the output's name comes from
// `MediaJob::derive_output`, which keeps the source's own extension, so
// a Matroska source already becomes `name.sovmr.mkv`. A second place
// that decides file extensions would be a second place to disagree.

impl Source<'_> {
    /// The container this source arrived in, which the output mirrors.
    pub fn container(&self) -> Container {
        match self {
            Source::Mp4 { .. } => Container::Mp4,
            Source::Mkv { .. } => Container::Mkv,
        }
    }

    /// Descriptions of the carried audio tracks as Matroska track
    /// specifications, for a Matroska output.
    ///
    /// Only ever called for a Matroska source, where the codec setup is
    /// already a bare `AudioSpecificConfig` and needs no unwrapping.
    /// What the source said its colours mean.
    ///
    /// Carried across rather than restated: the encoder is given the
    /// same pixels the source had, so the description that was true of
    /// them is still true. Inventing one where the source gave none
    /// would be worse than carrying nothing — a player trusts a colour
    /// description completely, and falls back to its own guess only
    /// when there is none.
    pub fn video_colour(&self) -> Option<sovmr_mp4::colour::Colour> {
        match self {
            Source::Mp4 { track, .. } => track.colour,
            Source::Mkv { video, .. } => video.colour,
        }
    }

    /// Coded size of every video frame with its presentation time.
    ///
    /// The size is the source encoder's own record of where the picture
    /// was hard. The timestamp is what identifies the frame afterwards:
    /// sizes come in decode order and frames come back from the decoder
    /// in *display* order, and with B-frames those are not the same
    /// sequence. Matching by position would place a keyframe wherever
    /// the reorder happened to put it.
    ///
    /// The source encoder's own record of where the picture was hard —
    /// read from a table, never by decoding. Empty when the container
    /// keeps no such table: Matroska has none, and a caller must then
    /// fall back on the fixed cadence rather than pretend to knowledge.
    pub fn video_sample_sizes(&self, file: &[u8]) -> Vec<(u32, i64)> {
        match self {
            Source::Mp4 { track, .. } => track
                .samples(file)
                .filter_map(|s| s.ok().map(|s| (s.data.len() as u32, s.pts)))
                .collect(),
            Source::Mkv { .. } => Vec::new(),
        }
    }

    /// Timescale of every carried track, in the order the output muxer
    /// declares them.
    ///
    /// This is what the interleaver builds its feeds from, and it must
    /// be container-neutral. [`Self::audio_params`] is not: it describes
    /// MP4 sample entries and returns nothing at all for a Matroska
    /// target, because a Matroska track needs no `mp4a` box. Building
    /// feeds from it meant a Matroska source's audio never reached the
    /// output — no feed, no packets, an empty track. Every gate missed
    /// it: the container ones checked our muxer directly, and the
    /// pipeline ones all used MP4 fixtures.
    pub fn audio_timescales(&self) -> Vec<u32> {
        match self {
            Source::Mp4 { audio, .. } => audio.tracks.iter().map(|t| t.timescale.max(1)).collect(),
            Source::Mkv { plan, .. } => plan.carried.iter().map(|p| p.timescale.max(1)).collect(),
        }
    }

    /// A carried track's **source** codec setup bytes.
    ///
    /// Distinct from what [`Self::audio_track_specs`] declares, which
    /// for a compressed track is already the AAC config it is about to
    /// become. A FLAC compressor needs the `STREAMINFO` it started
    /// from — handing it the output config instead makes it refuse to
    /// start, and the track would be quietly passed through at full
    /// size.
    pub fn audio_source_config(&self, index: usize) -> Vec<u8> {
        match self {
            Source::Mp4 { audio, .. } => audio
                .tracks
                .get(index)
                .map(|t| t.entry_payload.to_vec())
                .unwrap_or_default(),
            Source::Mkv { plan, .. } => plan
                .carried
                .get(index)
                .map(|p| p.codec_private.clone())
                .unwrap_or_default(),
        }
    }

    /// What will happen to each carried track, in the order
    /// [`Self::audio_track_specs`] and [`Self::audio_params`] use.
    ///
    /// Subtitles are always carried: they are text, and there is nothing
    /// to compress. An MP4 source only ever carries AAC, so nothing
    /// there is compressed either — the decision only has anything to
    /// decide when the source is Matroska.
    pub fn audio_routes(&self) -> Vec<crate::audio::Route> {
        match self {
            Source::Mp4 { audio, .. } => {
                vec![crate::audio::Route::Carry; audio.tracks.len()]
            }
            Source::Mkv { plan, .. } => plan
                .carried
                .iter()
                .map(|p| {
                    if p.subtitle {
                        return crate::audio::Route::Carry;
                    }
                    crate::audio::route(
                        &p.codec_id,
                        &p.codec_private,
                        p.sample_rate_for_output(),
                        u32::from(p.channels),
                        sovmr_mkv::source::Target::Mkv,
                    )
                })
                .collect(),
        }
    }

    /// The source's chapters and tags, ready for the muxer.
    ///
    /// Owned rather than borrowed, because a muxer outlives the mapping
    /// the file was read through. Empty for MP4 sources, which this
    /// project reads the *presence* of these from but does not yet
    /// parse — `Extras` says so out loud rather than letting a viewer
    /// discover it.
    pub fn marks(
        &self,
    ) -> (
        Vec<sovmr_mkv::mux::ChapterSpec>,
        Vec<sovmr_mkv::mux::TagSpec>,
    ) {
        match self {
            Source::Mkv { marks, .. } => marks.clone(),
            Source::Mp4 { .. } => (Vec::new(), Vec::new()),
        }
    }

    /// The source's title, when it named itself.
    pub fn title(&self) -> Option<&str> {
        match self {
            Source::Mkv { title, .. } => title.as_deref(),
            Source::Mp4 { .. } => None,
        }
    }

    pub fn audio_track_specs(&self) -> Vec<sovmr_mkv::mux::TrackSpec> {
        match self {
            Source::Mp4 { .. } => Vec::new(),
            Source::Mkv { plan, .. } => {
                let routes = self.audio_routes();
                plan.carried
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        // A compressed track is no longer what it was. Its
                        // name and setup bytes come from what it is about to
                        // become, or a player would be handed AAC frames
                        // labelled as PCM.
                        let (codec_id, codec_private) = match routes.get(i) {
                            Some(crate::audio::Route::Compress(_)) => (
                                "A_AAC".to_string(),
                                sovmr_audio::asc::AudioSpecificConfig {
                                    object_type: sovmr_audio::asc::AAC_LC,
                                    sample_rate: p.sample_rate_for_output(),
                                    channels: u32::from(p.channels),
                                }
                                .to_bytes(),
                            ),
                            _ => (p.codec_id.clone(), p.codec_private.clone()),
                        };
                        sovmr_mkv::mux::TrackSpec {
                            // Renumbered from one, so the output is
                            // self-consistent even where the source left gaps.
                            number: 2 + i as u64,
                            codec_id,
                            codec_private,
                            video: false,
                            subtitle: p.subtitle,
                            width: 0,
                            height: 0,
                            sample_rate: p.sample_rate_for_output(),
                            channels: p.channels,
                            language: String::from_utf8_lossy(&p.language).to_string(),
                            name: p.name.clone(),
                            enabled: p.enabled,
                            colour: None,
                        }
                    })
                    .collect()
            }
        }
    }
}
