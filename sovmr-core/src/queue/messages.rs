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

//! The complete UI <-> engine contract. These two enums are
//! the *only* things that cross the thread boundary: the UI owns its state,
//! the engine owns the queue, and neither ever locks the other's data.

use std::time::Duration;

use crate::job::{JobId, MediaJob};

/// UI -> engine. Fire-and-forget; results come back as [`EngineEvent`]s.
#[derive(Debug)]
pub enum EngineCommand {
    /// Add a validated, probed job to the tail of the queue.
    Enqueue(MediaJob),
    /// Cancel one job: dequeues it if pending, kills it if running.
    /// Any partial output is rolled back either way.
    Cancel(JobId),
    /// Cancel everything, pending and running.
    CancelAll,
    /// Resize the worker pool. Takes effect as slots free up; running
    /// encodes are never interrupted by a shrink.
    SetWorkerSlots(usize),
    /// Stop taking pending jobs. Running encodes are deliberately left to
    /// finish: killing them would discard partial work that cannot be
    /// resumed (a half-encoded stream is unusable), so the atomic unit is
    /// always one whole job. Idempotent.
    PauseQueue,
    /// Start taking pending jobs again. Idempotent.
    ResumeQueue,
    /// Cancel everything and terminate the engine thread.
    Shutdown,
}

/// Engine -> UI. Everything the dashboard needs to render, precomputed;
/// the UI thread does no math heavier than formatting.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// Job accepted into the pending queue.
    Queued {
        /// The job this event is about.
        id: JobId,
        /// Display name (file name) for the dashboard row.
        input_name: String,
        /// The heuristic engine's one-line verdict on *this* source —
        /// measured density/bitrate and the ceiling it will enforce
        /// (see `sovmr_core::heuristics::insight`). `None` when the
        /// source was too opaque to measure, or for synthetic rows the
        /// intake rejected before probing.
        insight: Option<String>,
    },
    /// A worker slot picked the job up; an encode session is live.
    Started {
        /// The job this event is about.
        id: JobId,
    },
    /// Periodic heartbeat from a running encode.
    Progress {
        /// The job this event is about.
        id: JobId,
        /// 0.0..=1.0 when the source duration is known.
        fraction: Option<f32>,
        /// Encoder throughput, frames per second.
        fps: f64,
        /// Realtime multiple (2.0 = twice realtime).
        speed: f64,
        /// Remaining wall-clock estimate, when computable.
        eta: Option<Duration>,
    },
    /// Non-terminal advisory: the job kept going but degraded gracefully
    /// (e.g. unsupported audio skipped). Shown on the row without
    /// changing its lifecycle state.
    Warning {
        /// The job this event is about.
        id: JobId,
        /// Human-readable degradation notice, shown verbatim in the UI.
        message: String,
    },
    /// Terminal: output committed to its final path.
    Completed {
        /// The job this event is about.
        id: JobId,
        /// Source size, for the compression-ratio badge.
        input_bytes: u64,
        /// Result size.
        output_bytes: u64,
    },
    /// Terminal: this job failed; the rest of the queue continues.
    Failed {
        /// The job this event is about.
        id: JobId,
        /// Human-readable diagnostic, shown verbatim in the UI.
        reason: String,
    },
    /// Terminal: cancelled by the user; partial output rolled back.
    Cancelled {
        /// The job this event is about.
        id: JobId,
    },
    /// Terminal: Smart Skip — the source is already in the target codec
    /// and already compact, so encoding it would waste GPU cycles and
    /// stack generation loss. The input file is untouched.
    Skipped {
        /// The job this event is about.
        id: JobId,
        /// Human-readable justification, shown verbatim in the UI.
        reason: String,
    },
    /// The queue is paused: in-flight encodes keep running to completion,
    /// but no pending job will start until [`EngineEvent::QueueResumed`].
    QueuePaused,
    /// The queue is taking pending jobs again.
    QueueResumed,
    /// The engine thread is exiting (response to `Shutdown`).
    EngineStopped,
    /// Pre-flight failed: this machine cannot run the engine at all
    /// (no NVIDIA driver, no usable GPU). Emitted once, before any job
    /// could exist; the UI replaces the dashboard with a full-window
    /// explanation instead of the process dying in a terminal nobody
    /// launched it from (graceful degradation).
    HardwareUnsupported {
        /// Human-readable diagnostic from the failed probe, shown
        /// verbatim (e.g. the dlopen error for `libcuda.so.1`).
        reason: String,
    },
}
