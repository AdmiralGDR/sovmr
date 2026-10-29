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

//! View state: a pure reducer over [`EngineEvent`]s, deliberately
//! independent of egui so it is fully unit-testable headless and reusable
//! by any renderer (TUI, web frontend, ...).
//!
//! [`DashboardState::apply`] is the only mutation path, the engine owns the
//! queue, and the two sides share nothing but channels.
//! The egui projection of this state lives in [`crate::shell`].
//!
//! # Example
//!
//! ```
//! use sovmr_core::job::JobId;
//! use sovmr_core::queue::messages::EngineEvent;
//! use sovmr_ui::state::{DashboardState, RowStatus};
//!
//! let mut dashboard = DashboardState::default();
//! let id = JobId::next();
//! dashboard.apply(EngineEvent::Queued {
//!     id,
//!     input_name: "clip.mp4".into(),
//!     insight: None,
//! });
//! dashboard.apply(EngineEvent::Completed { id, input_bytes: 100, output_bytes: 40 });
//! assert!(dashboard.rows[0].status.is_terminal());
//! ```

use std::time::Duration;

use sovmr_core::job::JobId;
use sovmr_core::queue::messages::EngineEvent;

/// Repaint cadence for the egui shell while jobs are running. Half the
/// typical progress heartbeat, so updates render without busy-repainting.
pub const UI_REPAINT_INTERVAL: Duration = Duration::from_millis(250);

/// Render-ready lifecycle of one dashboard row.
#[derive(Debug, Clone, PartialEq)]
pub enum RowStatus {
    /// Waiting for a free worker slot.
    Queued,
    /// Encoding right now.
    Running {
        /// 0.0..=1.0 when the source duration is known.
        fraction: Option<f32>,
        /// Encoder throughput, frames per second.
        fps: f64,
        /// Realtime multiple (2.0 = twice realtime).
        speed: f64,
        /// Remaining wall-clock estimate, when computable.
        eta: Option<Duration>,
    },
    /// Finished; output committed to its final path.
    Completed {
        /// Source size in bytes.
        input_bytes: u64,
        /// Result size in bytes.
        output_bytes: u64,
    },
    /// Failed; the queue kept going.
    Failed {
        /// Human-readable diagnostic, displayed verbatim.
        reason: String,
    },
    /// Cancelled by the user; partial output rolled back.
    Cancelled,
    /// Smart Skip: already in the target codec and already compact —
    /// encoding it would only lose quality. Input untouched.
    Skipped {
        /// Why the engine decided the file is already optimal.
        reason: String,
    },
}

impl RowStatus {
    /// `true` for states a row can never leave
    /// (completed/failed/cancelled/skipped).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            RowStatus::Completed { .. }
                | RowStatus::Failed { .. }
                | RowStatus::Cancelled
                | RowStatus::Skipped { .. }
        )
    }
}

/// One row of the queue table.
#[derive(Debug, Clone, PartialEq)]
pub struct JobRow {
    /// Job identity; pass to `EngineCommand::Cancel` from a cancel button.
    pub id: JobId,
    /// Display name (source file name).
    pub name: String,
    /// Current lifecycle state.
    pub status: RowStatus,
    /// Graceful-degradation notice (e.g. "audio skipped"), orthogonal to
    /// the lifecycle: a completed job can still carry a warning.
    pub warning: Option<String>,
    /// The heuristic engine's verdict on this source (density, character,
    /// enforced ceiling), rendered verbatim on the card — proof that the
    /// engine analysed *this* file rather than applying a default.
    pub insight: Option<String>,
}

/// Everything the dashboard renders. Owned exclusively by the UI thread.
#[derive(Debug, Default)]
pub struct DashboardState {
    /// Queue rows in enqueue order.
    pub rows: Vec<JobRow>,
    /// Queue is paused: running jobs finish, pending jobs are held.
    pub paused: bool,
    /// Set when the engine announced its shutdown.
    pub engine_stopped: bool,
    /// Set when pre-flight found no usable GPU stack: the renderer
    /// replaces the dashboard with a full-window explanation.
    pub hardware_error: Option<String>,
}

impl DashboardState {
    /// Fold one engine event into the view state. Unknown job ids are
    /// ignored (stale events after a queue clear), never a panic.
    pub fn apply(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Queued {
                id,
                input_name,
                insight,
            } => {
                self.rows.push(JobRow {
                    id,
                    name: input_name,
                    status: RowStatus::Queued,
                    warning: None,
                    insight,
                });
            }
            EngineEvent::Warning { id, message } => {
                if let Some(row) = self.rows.iter_mut().find(|r| r.id == id) {
                    row.warning = Some(message);
                }
            }
            EngineEvent::Started { id } => self.set_status(
                id,
                RowStatus::Running {
                    fraction: None,
                    fps: 0.0,
                    speed: 0.0,
                    eta: None,
                },
            ),
            EngineEvent::Progress {
                id,
                fraction,
                fps,
                speed,
                eta,
            } => self.set_status(
                id,
                RowStatus::Running {
                    fraction,
                    fps,
                    speed,
                    eta,
                },
            ),
            EngineEvent::Completed {
                id,
                input_bytes,
                output_bytes,
            } => self.set_status(
                id,
                RowStatus::Completed {
                    input_bytes,
                    output_bytes,
                },
            ),
            EngineEvent::Failed { id, reason } => self.set_status(id, RowStatus::Failed { reason }),
            EngineEvent::Cancelled { id } => self.set_status(id, RowStatus::Cancelled),
            EngineEvent::Skipped { id, reason } => {
                self.set_status(id, RowStatus::Skipped { reason })
            }
            EngineEvent::QueuePaused => self.paused = true,
            EngineEvent::QueueResumed => self.paused = false,
            EngineEvent::EngineStopped => self.engine_stopped = true,
            EngineEvent::HardwareUnsupported { reason } => self.hardware_error = Some(reason),
        }
    }

    /// Fraction of the whole batch that is done, for the window-level bar.
    pub fn batch_fraction(&self) -> f32 {
        if self.rows.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .rows
            .iter()
            .map(|r| match &r.status {
                RowStatus::Queued => 0.0,
                RowStatus::Running { fraction, .. } => fraction.unwrap_or(0.0),
                _ => 1.0,
            })
            .sum();
        sum / self.rows.len() as f32
    }

    /// `true` while any row is queued or running — drives the repaint
    /// cadence (idle dashboards schedule no repaints at all).
    pub fn has_active(&self) -> bool {
        self.rows.iter().any(|r| !r.status.is_terminal())
    }

    /// `true` while at least one encode is actually on a worker slot.
    /// Stricter than [`DashboardState::has_active`]: a paused queue full
    /// of pending jobs has active rows but nothing running — and needs no
    /// scheduled repaints.
    pub fn has_running(&self) -> bool {
        self.rows
            .iter()
            .any(|r| matches!(r.status, RowStatus::Running { .. }))
    }

    /// `(finished, total)` for the "3 of 8 done" batch summary.
    pub fn terminal_counts(&self) -> (usize, usize) {
        let done = self.rows.iter().filter(|r| r.status.is_terminal()).count();
        (done, self.rows.len())
    }

    /// `(input, output)` byte totals across all *completed* rows — the
    /// data behind the "Total Space Saved" visual. `(0, 0)` until the
    /// first completion.
    pub fn batch_savings(&self) -> (u64, u64) {
        self.rows.iter().fold((0, 0), |(i, o), r| match r.status {
            RowStatus::Completed {
                input_bytes,
                output_bytes,
            } => (i + input_bytes, o + output_bytes),
            _ => (i, o),
        })
    }

    /// Drop a *terminal* row from the dashboard (the "Remove" button).
    /// Active rows are refused — cancel them first, so a row can never
    /// silently vanish while its encode is still running.
    pub fn remove_row(&mut self, id: JobId) -> bool {
        match self
            .rows
            .iter()
            .position(|r| r.id == id && r.status.is_terminal())
        {
            Some(pos) => {
                self.rows.remove(pos);
                true
            }
            None => false,
        }
    }

    fn set_status(&mut self, id: JobId, status: RowStatus) {
        if let Some(row) = self.rows.iter_mut().find(|r| r.id == id) {
            // Terminal states are final: a straggling Progress heartbeat
            // that raced the Done message must not resurrect the row.
            if row.status.is_terminal() && matches!(status, RowStatus::Running { .. }) {
                return;
            }
            row.status = status;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(state: &mut DashboardState) -> JobId {
        let id = JobId::next();
        state.apply(EngineEvent::Queued {
            id,
            input_name: "clip.mp4".into(),
            insight: Some("Source: Noisy · 1.59 bpp · 99.0 Mbit/s → ceiling 5.0 Mbit/s".into()),
        });
        id
    }

    #[test]
    fn full_lifecycle_updates_one_row() {
        let mut s = DashboardState::default();
        let id = queued(&mut s);
        s.apply(EngineEvent::Started { id });
        s.apply(EngineEvent::Progress {
            id,
            fraction: Some(0.5),
            fps: 120.0,
            speed: 4.0,
            eta: Some(Duration::from_secs(30)),
        });
        s.apply(EngineEvent::Completed {
            id,
            input_bytes: 100,
            output_bytes: 40,
        });
        assert_eq!(s.rows.len(), 1);
        assert_eq!(
            s.rows[0].status,
            RowStatus::Completed {
                input_bytes: 100,
                output_bytes: 40
            }
        );
    }

    #[test]
    fn late_progress_cannot_resurrect_terminal_row() {
        let mut s = DashboardState::default();
        let id = queued(&mut s);
        s.apply(EngineEvent::Cancelled { id });
        s.apply(EngineEvent::Progress {
            id,
            fraction: Some(0.9),
            fps: 1.0,
            speed: 1.0,
            eta: None,
        });
        assert_eq!(s.rows[0].status, RowStatus::Cancelled);
    }

    #[test]
    fn events_for_unknown_jobs_are_ignored() {
        let mut s = DashboardState::default();
        s.apply(EngineEvent::Failed {
            id: JobId::next(),
            reason: "x".into(),
        });
        assert!(s.rows.is_empty());
    }

    #[test]
    fn skipped_rows_are_terminal_and_removable() {
        let mut s = DashboardState::default();
        let id = queued(&mut s);
        s.apply(EngineEvent::Skipped {
            id,
            reason: "already av1".into(),
        });
        assert!(s.rows[0].status.is_terminal());
        assert_eq!(s.terminal_counts(), (1, 1));
        assert!(s.remove_row(id));
    }

    #[test]
    fn warnings_attach_to_the_row_and_survive_completion() {
        let mut s = DashboardState::default();
        let id = queued(&mut s);
        s.apply(EngineEvent::Started { id });
        s.apply(EngineEvent::Warning {
            id,
            message: "audio skipped: unsupported codec 'fLaC' — output is video-only".into(),
        });
        s.apply(EngineEvent::Completed {
            id,
            input_bytes: 100,
            output_bytes: 40,
        });
        assert!(s.rows[0].status.is_terminal());
        assert!(
            s.rows[0]
                .warning
                .as_deref()
                .is_some_and(|w| w.contains("audio skipped")),
            "warning must outlive the state transition"
        );
        // Unknown ids are ignored, never a panic.
        s.apply(EngineEvent::Warning {
            id: JobId::next(),
            message: "stale".into(),
        });
    }

    #[test]
    fn hardware_unsupported_sets_the_error_screen_state() {
        let mut s = DashboardState::default();
        assert_eq!(s.hardware_error, None);
        s.apply(EngineEvent::HardwareUnsupported {
            reason: "backend unavailable: dlopen(libcuda.so.1) failed".into(),
        });
        assert!(s
            .hardware_error
            .as_deref()
            .is_some_and(|r| r.contains("libcuda.so.1")));
    }

    #[test]
    fn pause_events_toggle_the_flag() {
        let mut s = DashboardState::default();
        let id = queued(&mut s);
        s.apply(EngineEvent::QueuePaused);
        assert!(s.paused);
        assert!(s.has_active(), "queued row is active");
        assert!(
            !s.has_running(),
            "paused queue with no started job runs nothing"
        );
        s.apply(EngineEvent::Started { id });
        assert!(
            s.has_running(),
            "in-flight job keeps running through a pause"
        );
        s.apply(EngineEvent::QueueResumed);
        assert!(!s.paused);
    }

    #[test]
    fn remove_only_touches_terminal_rows() {
        let mut s = DashboardState::default();
        let running = queued(&mut s);
        let done = queued(&mut s);
        s.apply(EngineEvent::Started { id: running });
        s.apply(EngineEvent::Completed {
            id: done,
            input_bytes: 2,
            output_bytes: 1,
        });
        assert!(!s.remove_row(running), "active rows must be refused");
        assert!(s.remove_row(done));
        assert_eq!(s.rows.len(), 1);
        assert!(s.has_active());
        assert_eq!(s.terminal_counts(), (0, 1));
    }

    #[test]
    fn batch_savings_sums_only_completed_rows() {
        let mut s = DashboardState::default();
        let a = queued(&mut s);
        let b = queued(&mut s);
        let c = queued(&mut s);
        s.apply(EngineEvent::Completed {
            id: a,
            input_bytes: 1000,
            output_bytes: 400,
        });
        s.apply(EngineEvent::Completed {
            id: b,
            input_bytes: 500,
            output_bytes: 300,
        });
        s.apply(EngineEvent::Failed {
            id: c,
            reason: "x".into(),
        });
        assert_eq!(s.batch_savings(), (1500, 700));
        assert_eq!(DashboardState::default().batch_savings(), (0, 0));
    }

    #[test]
    fn batch_fraction_averages_across_rows() {
        let mut s = DashboardState::default();
        let a = queued(&mut s);
        let _b = queued(&mut s);
        s.apply(EngineEvent::Completed {
            id: a,
            input_bytes: 1,
            output_bytes: 1,
        });
        assert!((s.batch_fraction() - 0.5).abs() < f32::EPSILON);
        assert_eq!(DashboardState::default().batch_fraction(), 0.0);
    }
}
