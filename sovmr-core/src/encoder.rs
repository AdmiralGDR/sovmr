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

//! The encoder facade every backend implements.
//!
//! This crate talks exclusively to these traits. The production backend
//! is `sovmr_native::NvencNativeProvider` — the sovereign zero-copy
//! NVDEC→NVENC pipeline (it replaced the Phase-1 FFmpeg-process backend
//! by swapping one line in the composition root — no change to core or
//! UI was required, which is the entire point of this facade).
//!
//! # Implementing a backend
//!
//! A backend supplies three things: an [`EncoderProvider`] (factory +
//! capability probe), an [`EncodeSession`] (one running encode, consumed by
//! `wait`), and a [`CancelHandle`] (out-of-band, non-blocking kill switch):
//!
//! ```ignore
//! struct MyProvider;
//!
//! impl EncoderProvider for MyProvider {
//!     fn id(&self) -> &'static str { "my/backend" }
//!     fn capabilities(&self) -> Result<EncoderCapabilities, EncoderError> {
//!         Ok(EncoderCapabilities { av1: true, hevc: true })
//!     }
//!     fn probe(&self, input: &Path) -> Result<MediaInfo, EncoderError> {
//!         // read duration, dimensions, codec from the container
//!     }
//!     fn spawn(&self, job: &MediaJob, progress: Sender<ProgressUpdate>)
//!         -> Result<Box<dyn EncodeSession>, EncoderError> {
//!         // start encoding; stream heartbeats into `progress`
//!     }
//! }
//! ```
//!
//! Contract highlights every implementation must honour:
//! * `spawn` returns immediately; progress flows from a backend-owned thread.
//! * `wait` must delete partial output on failure/cancel and atomically
//!   commit it on success (rollback).
//! * Errors cross this boundary as [`EncoderError`] values — never panics.

use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use crate::job::{MediaInfo, MediaJob};

/// Everything that can go wrong inside a backend, in terms the UI can
/// display verbatim. Never panics across this boundary.
#[derive(Debug)]
pub enum EncoderError {
    /// The backend binary/library is missing or unusable.
    BackendUnavailable(String),
    /// The source file could not be probed (corrupt header, not media, ...).
    ProbeFailed(String),
    /// The encode process could not be started.
    SpawnFailed(String),
    /// The encode started but ended in failure; carries the backend's own
    /// diagnostic (e.g. the tail of ffmpeg's stderr).
    EncodeFailed(String),
    /// Underlying I/O problem (temp file, rename, pipe).
    Io(std::io::Error),
}

impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderError::BackendUnavailable(m) => write!(f, "backend unavailable: {m}"),
            EncoderError::ProbeFailed(m) => write!(f, "probe failed: {m}"),
            EncoderError::SpawnFailed(m) => write!(f, "spawn failed: {m}"),
            EncoderError::EncodeFailed(m) => write!(f, "encode failed: {m}"),
            EncoderError::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl From<std::io::Error> for EncoderError {
    fn from(e: std::io::Error) -> Self {
        EncoderError::Io(e)
    }
}

/// Which target codecs this backend can actually produce *on this machine*
/// (build support and hardware support intersected by the provider).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncoderCapabilities {
    /// AV1 encoding is available (backend build supports it).
    pub av1: bool,
    /// HEVC (H.265) encoding is available.
    pub hevc: bool,
}

/// One progress heartbeat from a running encode. Units are backend-neutral;
/// providers translate their native telemetry into this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressUpdate {
    /// Frames emitted so far.
    pub frames: u64,
    /// Instantaneous encode rate in frames per second.
    pub fps: f64,
    /// Position in the output timeline; divide by source duration for %.
    pub out_time: Duration,
    /// Realtime multiple (2.0 = encoding twice as fast as playback).
    pub speed: f64,
}

/// Terminal state of a session that did not error.
#[derive(Debug, PartialEq, Eq)]
pub enum EncodeOutcome {
    /// Output file is fully written and atomically moved into place.
    Completed {
        /// Size of the finished output file, for compression-ratio display.
        output_bytes: u64,
        /// Non-fatal degradation the user must learn about (e.g. "audio
        /// skipped: unsupported codec") — the job still succeeded, but
        /// not at full fidelity (graceful degradation).
        warning: Option<String>,
    },
    /// User cancelled; partial output has been rolled back (deleted).
    Cancelled,
}

/// Thread-safe cancellation trigger for one session. Held by the scheduler
/// while the session's owning thread is parked in [`EncodeSession::wait`].
pub trait CancelHandle: Send + Sync {
    /// Request cancellation. Must be idempotent and must never block; the
    /// session observes the request and reports `Cancelled` from `wait`.
    fn cancel(&self);
}

/// A single in-flight encode. Owned by exactly one worker thread, which
/// consumes it via `wait`; cancellation arrives from outside through the
/// [`CancelHandle`] (mutability control).
pub trait EncodeSession: Send {
    /// Obtain the out-of-band cancellation trigger.
    fn cancel_handle(&self) -> Arc<dyn CancelHandle>;

    /// Block until the encode reaches a terminal state. On success the
    /// output is in its final location; on cancel or error any partial
    /// output has been deleted (rollback guarantee).
    fn wait(self: Box<Self>) -> Result<EncodeOutcome, EncoderError>;
}

/// The Facade. The application core passes a [`MediaJob`] in and consumes
/// [`ProgressUpdate`]s out; it must never learn which engine is behind this.
pub trait EncoderProvider: Send + Sync {
    /// Stable identifier for logs and the UI's "engine" badge
    /// (e.g. `"ffmpeg/nvenc"`, later `"native/nvenc"`).
    fn id(&self) -> &'static str;

    /// What this backend can encode on this machine, probed defensively
    /// (never assumed from the GPU name alone).
    fn capabilities(&self) -> Result<EncoderCapabilities, EncoderError>;

    /// Inspect a source file: duration, dimensions, codec, size. Called
    /// after (never instead of) `core::validate` signature checks.
    fn probe(&self, input: &Path) -> Result<MediaInfo, EncoderError>;

    /// Start encoding. Non-blocking: returns a session immediately; progress
    /// flows through `progress` from a backend-owned thread. A dropped
    /// receiver must not wedge or kill the encode.
    fn spawn(
        &self,
        job: &MediaJob,
        progress: Sender<ProgressUpdate>,
    ) -> Result<Box<dyn EncodeSession>, EncoderError>;
}
