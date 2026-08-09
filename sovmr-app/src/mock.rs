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

//! Simulation backend (`sovmr --mock`): a full [`EncoderProvider`] that
//! fabricates smooth progress without touching ffmpeg, the GPU, or the
//! disk. Exists to exercise the real scheduler + UI under load — parallel
//! progress bars, ETA math, failure rows, cancellation — on any machine.
//! Jobs whose file name contains [`FAIL_MARKER`] fail at ~60% to prove the
//! per-job failure path.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use sovmr_core::encoder::{
    CancelHandle, EncodeOutcome, EncodeSession, EncoderCapabilities, EncoderError, EncoderProvider,
    ProgressUpdate,
};
use sovmr_core::job::{MediaInfo, MediaJob};

/// Inputs containing this substring simulate a mid-encode failure.
pub const FAIL_MARKER: &str = "corrupt";
/// Inputs containing this substring complete with a degradation warning
/// (exercises the ⚠ badge exactly like an unsupported-audio source).
pub const WARN_MARKER: &str = "screen_capture";

/// Simulated heartbeat period — fast enough to prove bars animate smoothly.
const TICK: Duration = Duration::from_millis(80);
/// How much simulated media time one tick advances (≈12× realtime).
const MEDIA_TIME_PER_TICK: Duration = Duration::from_millis(1000);
/// Fake source duration: base + per-job spread so bars finish staggered.
const FAKE_DURATION_BASE_SECS: u64 = 18;
const FAKE_DURATION_SPREAD_SECS: u64 = 7;
/// Fraction at which `FAIL_MARKER` jobs abort.
const FAIL_AT_FRACTION: f64 = 0.6;
/// Pretend sources are ~1.8 GB and outputs ~38% of that.
const FAKE_INPUT_BYTES: u64 = 1_800_000_000;
const FAKE_OUTPUT_RATIO: f64 = 0.38;
/// Baseline fake encoder throughput.
const FAKE_FPS_BASE: f64 = 150.0;

/// The simulation provider. Stateless; every session is independent.
pub struct MockProvider;

/// Derive a per-job "source duration" from the file name so the demo queue
/// finishes staggered instead of in lockstep.
fn fake_duration(input: &Path) -> Duration {
    let salt: u64 =
        input.to_string_lossy().bytes().map(u64::from).sum::<u64>() % FAKE_DURATION_SPREAD_SECS;
    Duration::from_secs(FAKE_DURATION_BASE_SECS + salt * 2)
}

impl EncoderProvider for MockProvider {
    fn id(&self) -> &'static str {
        "mock/simulation"
    }

    fn capabilities(&self) -> Result<EncoderCapabilities, EncoderError> {
        Ok(EncoderCapabilities {
            av1: true,
            hevc: true,
        })
    }

    fn probe(&self, input: &Path) -> Result<MediaInfo, EncoderError> {
        Ok(MediaInfo {
            duration: Some(fake_duration(input)),
            width: 3840,
            height: 2160,
            video_codec: "h264 (simulated)".to_string(),
            size_bytes: FAKE_INPUT_BYTES,
            video_bytes: 0,
            fps: 30.0,
            audio_bitrate: 1_536_000, // fat PCM-ish track: exercises the ⚠ audio warning
            // A six-language master with subtitles and chapters, so the
            // simulation exercises the composition line on the card.
            audio_tracks: 6,
            subtitle_tracks: 2,
            chapters: true,
        })
    }

    fn spawn(
        &self,
        job: &MediaJob,
        progress: Sender<ProgressUpdate>,
    ) -> Result<Box<dyn EncodeSession>, EncoderError> {
        Ok(Box::new(MockSession {
            cancel: Arc::new(MockCancel(AtomicBool::new(false))),
            total: job
                .info
                .duration
                .unwrap_or(Duration::from_secs(FAKE_DURATION_BASE_SECS)),
            fail: job.input.to_string_lossy().contains(FAIL_MARKER),
            warn: job.input.to_string_lossy().contains(WARN_MARKER),
            progress,
        }))
    }
}

struct MockCancel(AtomicBool);

impl CancelHandle for MockCancel {
    fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct MockSession {
    cancel: Arc<MockCancel>,
    total: Duration,
    fail: bool,
    warn: bool,
    progress: Sender<ProgressUpdate>,
}

impl EncodeSession for MockSession {
    fn cancel_handle(&self) -> Arc<dyn CancelHandle> {
        self.cancel.clone()
    }

    /// Tick simulated media time forward, emitting a heartbeat per tick,
    /// until completion, simulated corruption, or cancellation.
    fn wait(self: Box<Self>) -> Result<EncodeOutcome, EncoderError> {
        let mut out_time = Duration::ZERO;
        let mut frames: u64 = 0;
        while out_time < self.total {
            thread::sleep(TICK);
            if self.cancel.0.load(Ordering::SeqCst) {
                return Ok(EncodeOutcome::Cancelled);
            }
            out_time += MEDIA_TIME_PER_TICK;
            frames += (MEDIA_TIME_PER_TICK.as_secs_f64() * 30.0) as u64;
            let fraction = out_time.as_secs_f64() / self.total.as_secs_f64();
            if self.fail && fraction >= FAIL_AT_FRACTION {
                return Err(EncoderError::EncodeFailed(
                    "simulated bitstream corruption (mock)".to_string(),
                ));
            }
            // Gentle wobble so fps/speed read as live telemetry, not a
            // frozen constant.
            let wobble = 1.0 + 0.15 * (out_time.as_secs_f64() * 1.3).sin();
            let _ = self.progress.send(ProgressUpdate {
                frames,
                fps: FAKE_FPS_BASE * wobble,
                out_time,
                speed: (MEDIA_TIME_PER_TICK.as_secs_f64() / TICK.as_secs_f64()) * wobble,
            });
        }
        Ok(EncodeOutcome::Completed {
            output_bytes: (FAKE_INPUT_BYTES as f64 * FAKE_OUTPUT_RATIO) as u64,
            warning: self.warn.then(|| {
                "audio skipped: unsupported codec 'lpcm' — output is video-only (simulated)"
                    .to_string()
            }),
        })
    }
}
