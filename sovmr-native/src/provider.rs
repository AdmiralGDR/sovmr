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

//! [`NvencNativeProvider`] — the sovereign pipeline behind the
//! `EncoderProvider` facade: demux → NVDEC → VRAM → NVENC → fast-start
//! MP4 (AV1 + bit-exact audio passthrough). The composition root swaps
//! engines by constructing this type; core, queue, and UI notice
//! nothing.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use sovmr_core::encoder::{
    CancelHandle, EncodeOutcome, EncodeSession, EncoderCapabilities, EncoderError, EncoderProvider,
    ProgressUpdate,
};
use sovmr_core::job::{Codec, Effort, MediaInfo, MediaJob};
use sovmr_mp4::demux;
use sovmr_nvenc::encoder::{EncodeCodec, EncodePreset};
use sovmr_os::MappedFile;

use crate::pipeline::{self, NativeSettings, TranscodeOutcome};

/// Nominal fps used to synthesize `out_time` progress until the demuxer
/// reads real timescales (matches the pipeline's rate-control nominal).
const NOMINAL_FPS: f64 = 30.0;

/// Summarise a Matroska source into the engine's container-neutral
/// [`MediaInfo`].
///
/// The audio bitrate here is **sampled**, not counted: Matroska keeps no
/// per-track size table, so an exact figure would mean reading every
/// cluster of a fifty-gigabyte file just to answer a queue-time
/// question. It feeds a warning threshold, and is documented as an
/// estimate where it is produced.
fn probe_matroska(bytes: &[u8]) -> Result<MediaInfo, EncoderError> {
    let s =
        sovmr_mkv::probe::summarise(bytes).map_err(|e| EncoderError::ProbeFailed(e.to_string()))?;
    if s.video_codec.is_empty() {
        return Err(EncoderError::ProbeFailed(
            "no video track this engine can decode".into(),
        ));
    }
    Ok(MediaInfo {
        duration: (s.duration_secs > 0.0).then(|| Duration::from_secs_f64(s.duration_secs)),
        width: s.width,
        height: s.height,
        video_codec: s.video_codec.to_string(),
        size_bytes: bytes.len() as u64,
        // Matroska has no sample-size table, so the video's own byte
        // count is not there to be read without walking every cluster.
        // Zero says "not known" and the curve falls back to the file
        // length — the old behaviour, now confined to the container that
        // genuinely cannot answer instead of applied to both.
        video_bytes: 0,
        fps: s.fps,
        audio_bitrate: s.estimated_audio_bitrate,
        audio_tracks: s.audio_tracks,
        subtitle_tracks: s.subtitle_tracks,
        chapters: s.chapters,
    })
}

/// The Phase-2 engine: demux + NVDEC + NVENC, zero copies, zero FFmpeg.
pub struct NvencNativeProvider;

impl EncoderProvider for NvencNativeProvider {
    fn id(&self) -> &'static str {
        "native/nvenc"
    }

    /// Capabilities come from the driver, not from the GPU's name.
    ///
    /// This used to load the runtime and then answer `av1: true, hevc:
    /// true` regardless — a claim with nothing behind it. What kept the
    /// engine honest was a string match on the marketing name elsewhere
    /// ("RTX 30" implies no AV1 encode), which is right for Ampere by
    /// coincidence of naming and wrong for the first card whose name the
    /// table has not seen. Choosing a codec the silicon cannot produce
    /// fails at the first frame, hours into a batch.
    ///
    /// So: open a real session on a real CUDA context and ask
    /// `nvEncGetEncodeGUIDs` what this encoder supports.
    fn capabilities(&self) -> Result<EncoderCapabilities, EncoderError> {
        let api = sovmr_nvenc::NvEncApi::load()
            .map_err(|e| EncoderError::BackendUnavailable(e.to_string()))?;
        let cuda = sovmr_cuda::Cuda::load()
            .map_err(|e| EncoderError::BackendUnavailable(e.to_string()))?;
        let ctx = cuda
            .create_context(0)
            .map_err(|e| EncoderError::BackendUnavailable(e.to_string()))?;
        let session = api
            .open_session(&ctx)
            .map_err(|e| EncoderError::BackendUnavailable(e.to_string()))?;
        let support = session
            .supported_codecs()
            .map_err(|e| EncoderError::BackendUnavailable(e.to_string()))?;
        Ok(EncoderCapabilities {
            av1: support.av1,
            hevc: support.hevc,
        })
    }

    /// Probe via the sovereign demuxer: codec, dimensions, duration
    /// (from `mdhd`), file size.
    fn probe(&self, input: &Path) -> Result<MediaInfo, EncoderError> {
        // Probing reads only the header boxes; mapping keeps a 50 GiB
        // candidate from costing 50 GiB of RAM just to answer "how long
        // is it?".
        let bytes =
            MappedFile::open(input).map_err(|e| EncoderError::ProbeFailed(e.to_string()))?;
        // Which container, by the bytes rather than the name. A file
        // called `.mkv` that is really an MP4 must still open, and a
        // name is a claim rather than a fact.
        if sovmr_mkv::is_matroska(&bytes) {
            return probe_matroska(&bytes);
        }
        let track = demux::find_video_track(&bytes)
            .map_err(|e| EncoderError::ProbeFailed(e.to_string()))?;
        // From the whole track, not its first sample. On variable-rate
        // material the first frame is not the film, and a wrong frame
        // rate propagates into every bits-per-pixel judgement the engine
        // makes after it.
        let fps = track.frame_rate().unwrap_or(0.0);
        // Audio weighs on the ratio: passthrough copies it verbatim, and
        // ALL of it comes along. Six 192 kbit/s tracks across twelve
        // hours are gigabytes the heuristic engine has to account for —
        // measuring only the first would understate the file's audio
        // weight sixfold and make the compression estimate a fiction.
        // A file with no audio parses to an empty list, not an error; an
        // error here means the audio tables are structurally broken while
        // the video track already validated. Degrade to a video-only
        // estimate rather than refuse a usable file — but SAY the failure:
        // a silent 0 would understate the file to the heuristic and
        // hide a real defect in the source from everyone downstream, which
        // is exactly the "no audio" vs "audio unreadable" confusion the
        // rest of this engine is built to avoid.
        let found_audio = match demux::find_audio_tracks(&bytes) {
            Ok(tracks) => Some(tracks),
            Err(e) => {
                eprintln!(
                    "sovmr: probe could not read the audio tables of {}: {e} \
                     — proceeding as if it carried no audio",
                    input.display()
                );
                None
            }
        };
        let audio_bitrate = found_audio.as_ref().map_or(0, |t| t.total_bitrate());
        let audio_tracks = found_audio.as_ref().map_or(0, |t| t.tracks.len());
        // What else the file carries, so the card can say so before the
        // job runs rather than the warning saying so afterwards. A
        // source we cannot survey simply reports nothing extra — this
        // drives a label, not a decision.
        let extras = demux::survey_extras(&bytes).unwrap_or_default();
        Ok(MediaInfo {
            duration: track.duration(),
            fps,
            audio_bitrate,
            audio_tracks,
            subtitle_tracks: extras.subtitles,
            chapters: extras.chapters,
            width: u32::from(track.width),
            height: u32::from(track.height),
            video_codec: match track.codec {
                demux::VideoCodec::H264 => "h264".to_string(),
                demux::VideoCodec::Hevc => "hevc".to_string(),
                demux::VideoCodec::Av1 => "av1".to_string(),
            },
            size_bytes: bytes.len() as u64,
            video_bytes: track.total_bytes(),
        })
    }

    fn spawn(
        &self,
        job: &MediaJob,
        progress: Sender<ProgressUpdate>,
    ) -> Result<Box<dyn EncodeSession>, EncoderError> {
        let cancel = Arc::new(NativeCancel(AtomicBool::new(false)));
        // The heuristic engine translates semantic intent + measured
        // source reality into the exact NVENC numbers.
        let rate_plan =
            sovmr_core::heuristics::plan(&job.info, job.settings.strategy, job.settings.codec);
        let settings = NativeSettings {
            codec: match job.settings.codec {
                Codec::Av1 => EncodeCodec::Av1,
                Codec::Hevc => EncodeCodec::Hevc,
            },
            cq: rate_plan.cq,
            max_bitrate: rate_plan.max_bitrate,
            preset: match job.settings.effort {
                Effort::Speed => EncodePreset::Speed,
                Effort::Balanced => EncodePreset::Balanced,
                Effort::Quality => EncodePreset::Quality,
            },
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        };
        let input = job.input.clone();
        let output = job.output.clone();
        let flag = cancel.clone();
        let worker: JoinHandle<Result<TranscodeOutcome, pipeline::NativeError>> =
            thread::Builder::new()
                .name("sovmr-native-pipeline".into())
                .spawn(move || {
                    let started = Instant::now();
                    pipeline::transcode(
                        &input,
                        &output,
                        &settings,
                        |frames| {
                            let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
                            let fps = frames as f64 / elapsed;
                            let _ = progress.send(ProgressUpdate {
                                frames,
                                fps,
                                out_time: Duration::from_secs_f64(frames as f64 / NOMINAL_FPS),
                                speed: fps / NOMINAL_FPS,
                            });
                        },
                        &flag.0,
                    )
                })
                .map_err(|e| EncoderError::SpawnFailed(e.to_string()))?;
        Ok(Box::new(NativeSession { cancel, worker }))
    }
}

struct NativeCancel(AtomicBool);

impl CancelHandle for NativeCancel {
    fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct NativeSession {
    cancel: Arc<NativeCancel>,
    worker: JoinHandle<Result<TranscodeOutcome, pipeline::NativeError>>,
}

impl EncodeSession for NativeSession {
    fn cancel_handle(&self) -> Arc<dyn CancelHandle> {
        self.cancel.clone()
    }

    fn wait(self: Box<Self>) -> Result<EncodeOutcome, EncoderError> {
        match self.worker.join() {
            Ok(Ok(TranscodeOutcome::Completed(stats))) => Ok(EncodeOutcome::Completed {
                output_bytes: stats.output_bytes,
                warning: stats.warning,
            }),
            Ok(Ok(TranscodeOutcome::Cancelled)) => Ok(EncodeOutcome::Cancelled),
            Ok(Err(e)) => Err(EncoderError::EncodeFailed(e.to_string())),
            Err(_) => Err(EncoderError::EncodeFailed(
                "native pipeline thread panicked".to_string(),
            )),
        }
    }
}
