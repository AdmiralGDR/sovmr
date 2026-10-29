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

//! Sovereign desktop integration (no third-party crates —
//! everything below talks to the OS through `std::process::Command`).
//!
//! Split in two for testability:
//! * [`BatchWatcher`] — a *pure* state machine folding [`EngineEvent`]s
//!   into [`SystemAction`]s (inhibit sleep / release / notify). Fully unit
//!   tested, no I/O.
//! * [`SystemBridge`] — executes those actions: holds a `systemd-inhibit`
//!   child while encodes run, sends a notification when the batch drains.
//!   Every external tool is optional; a missing binary degrades to a
//!   one-line warning, never a crash.

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicI16, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use sovmr_core::job::JobId;
use sovmr_core::queue::messages::EngineEvent;

/// Video-engine utilization polling. `utilization.encoder` is the NVENC
/// block specifically (verified against driver 580.x — the older
/// `utilization.video` field is rejected there).
const GPU_METER_BIN: &str = "nvidia-smi";
const GPU_METER_ARGS: [&str; 2] = [
    "--query-gpu=utilization.encoder",
    "--format=csv,noheader,nounits",
];
/// 1 Hz is plenty for a glanceable meter and costs one short-lived
/// process per second (measured, bounded overhead).
const GPU_POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Sentinel stored while utilization is unknown (meter hidden in the UI).
const GPU_UTIL_UNKNOWN: i16 = -1;
/// Simulation waveform for `--mock`: base ± swing, slow sine.
const GPU_SIM_BASE: f64 = 62.0;
const GPU_SIM_SWING: f64 = 28.0;
const GPU_SIM_HZ: f64 = 0.11;

/// Spawn the 1 Hz utilization poller. Returns the shared cell the UI
/// reads each frame: `0..=100` = live percentage, negative = unknown.
/// In `simulate` mode (mock engine) it synthesizes a plausible waveform
/// instead of spawning `nvidia-smi`. The thread is detached: it holds no
/// locks and dies with the process.
pub fn spawn_gpu_meter(simulate: bool) -> Arc<AtomicI16> {
    let cell = Arc::new(AtomicI16::new(GPU_UTIL_UNKNOWN));
    let writer = cell.clone();
    thread::Builder::new()
        .name("sovmr-gpu-meter".into())
        .spawn(move || {
            let started = Instant::now();
            loop {
                let value = if simulate {
                    let t = started.elapsed().as_secs_f64();
                    let wave = (t * GPU_SIM_HZ * std::f64::consts::TAU).sin();
                    (GPU_SIM_BASE + GPU_SIM_SWING * wave) as i16
                } else {
                    query_video_util().unwrap_or(GPU_UTIL_UNKNOWN)
                };
                writer.store(value, Ordering::Relaxed);
                thread::sleep(GPU_POLL_INTERVAL);
            }
        })
        .expect("spawning gpu meter thread");
    cell
}

/// One `nvidia-smi` sample. `None` on any failure (missing binary, driver
/// hiccup, unparsable output) — the meter simply hides.
fn query_video_util() -> Option<i16> {
    let output = Command::new(GPU_METER_BIN)
        .args(GPU_METER_ARGS)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_video_util(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first CSV line into a clamped percentage. Defensive: driver
/// output is input like any other.
fn parse_video_util(text: &str) -> Option<i16> {
    let value: i16 = text.lines().next()?.trim().parse().ok()?;
    (0..=100).contains(&value).then_some(value)
}

/// What the watcher wants done to the OS, in order.
#[derive(Debug, PartialEq, Eq)]
pub enum SystemAction {
    /// First encode started: take the sleep/idle inhibitor lock.
    InhibitStart,
    /// Last running encode ended: release the lock (even if pending jobs
    /// are held by a pause — a paused app must not keep the OS awake).
    InhibitStop,
    /// The whole queue drained with at least one success: tell the user.
    Notify {
        /// Notification title.
        summary: String,
        /// Notification body (counts + bytes saved).
        body: String,
    },
}

/// Where a tracked job currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Pending,
    Running,
}

/// Pure fold of engine events into system actions. Owns no I/O, so the
/// exact inhibit/notify behaviour is provable in unit tests.
#[derive(Debug, Default)]
pub struct BatchWatcher {
    jobs: HashMap<JobId, Phase>,
    completed: usize,
    failed: usize,
    skipped: usize,
    /// Net bytes saved by completed jobs (can go negative if files grew).
    saved_bytes: i128,
    inhibited: bool,
}

impl BatchWatcher {
    /// Fold one event; returns the actions it triggers (usually none).
    pub fn observe(&mut self, event: &EngineEvent) -> Vec<SystemAction> {
        match event {
            EngineEvent::Queued { id, .. } => {
                self.jobs.insert(*id, Phase::Pending);
                Vec::new()
            }
            EngineEvent::Started { id } => {
                self.jobs.insert(*id, Phase::Running);
                if !self.inhibited {
                    self.inhibited = true;
                    vec![SystemAction::InhibitStart]
                } else {
                    Vec::new()
                }
            }
            EngineEvent::Completed {
                id,
                input_bytes,
                output_bytes,
            } => {
                self.completed += 1;
                self.saved_bytes += *input_bytes as i128 - *output_bytes as i128;
                self.job_ended(*id)
            }
            EngineEvent::Failed { id, .. } => {
                self.failed += 1;
                self.job_ended(*id)
            }
            EngineEvent::Cancelled { id } => self.job_ended(*id),
            EngineEvent::Skipped { id, .. } => {
                self.skipped += 1;
                self.job_ended(*id)
            }
            _ => Vec::new(),
        }
    }

    fn job_ended(&mut self, id: JobId) -> Vec<SystemAction> {
        self.jobs.remove(&id);
        let mut actions = Vec::new();
        let any_running = self.jobs.values().any(|p| *p == Phase::Running);
        if self.inhibited && !any_running {
            self.inhibited = false;
            actions.push(SystemAction::InhibitStop);
        }
        if self.jobs.is_empty() {
            if self.completed > 0 {
                actions.push(SystemAction::Notify {
                    summary: "sovmr — batch complete".to_string(),
                    body: batch_body(self.completed, self.failed, self.skipped, self.saved_bytes),
                });
            }
            self.completed = 0;
            self.failed = 0;
            self.skipped = 0;
            self.saved_bytes = 0;
        }
        actions
    }
}

/// Human notification body:
/// "7 files encoded · 2 already optimal · 1 failed · 3.4 GB saved".
fn batch_body(completed: usize, failed: usize, skipped: usize, saved_bytes: i128) -> String {
    let files = if completed == 1 { "file" } else { "files" };
    let mut body = format!("{completed} {files} encoded");
    if skipped > 0 {
        body.push_str(&format!(" · {skipped} already optimal"));
    }
    if failed > 0 {
        body.push_str(&format!(" · {failed} failed"));
    }
    if saved_bytes > 0 {
        body.push_str(&format!(
            " · {} saved",
            sovmr_ui::fmt::bytes(saved_bytes as u64)
        ));
    }
    body
}

/// Executes [`SystemAction`]s against the real desktop. All tools are
/// best-effort: a system without them just loses the nicety.
#[derive(Default)]
pub struct SystemBridge {
    stay_awake: Option<sovmr_os::desktop::StayAwake>,
}

impl SystemBridge {
    /// Apply one action now.
    pub fn apply(&mut self, action: SystemAction) {
        match action {
            SystemAction::InhibitStart => self.inhibit_start(),
            SystemAction::InhibitStop => self.inhibit_stop(),
            SystemAction::Notify { summary, body } => notify(&summary, &body),
        }
    }

    /// Take the sleep/idle lock by parking a trivial child under
    /// Ask the machine to stay awake while there is work.
    ///
    /// The platform's own mechanism, chosen by [`sovmr_os::desktop`]:
    /// logind on Linux, `SetThreadExecutionState` on Windows. This used
    /// to spawn `systemd-inhibit` directly, which on Windows spawned
    /// nothing and said nothing — a twelve-hour encode could be
    /// interrupted by the machine going to sleep.
    fn inhibit_start(&mut self) {
        if self.stay_awake.is_some() {
            return;
        }
        match sovmr_os::desktop::stay_awake("encoding video") {
            Ok(held) => self.stay_awake = Some(held),
            // Named, not swallowed: a user whose machine sleeps mid-job
            // should be able to find out why.
            Err(e) => eprintln!("sovmr: sleep inhibition unavailable ({e})"),
        }
    }

    /// Release it. Dropping the promise is the whole mechanism.
    fn inhibit_stop(&mut self) {
        self.stay_awake = None;
    }
}

impl Drop for SystemBridge {
    /// The lock must never outlive the app: an orphaned inhibitor would
    /// block the machine's sleep indefinitely.
    fn drop(&mut self) {
        self.inhibit_stop();
    }
}

/// Tell the user a job finished, when they may not be looking.
///
/// Asked of the platform boundary. This used to spawn `notify-send` and
/// fall back to `gdbus`, neither of which exists on Windows — so on one
/// of the two platforms it shipped to, nothing was ever shown and
/// nothing said so.
///
/// Best effort by design: a notice that cannot be shown is not a reason
/// to fail a job that succeeded.
fn notify(summary: &str, body: &str) {
    if !sovmr_os::desktop::notify(summary, body) {
        // Named once rather than silently: a user who expected a popup
        // should be able to find out why there was none.
        eprintln!("sovmr: no desktop notification service accepted '{summary}'");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(w: &mut BatchWatcher) -> JobId {
        let id = JobId::next();
        assert!(w
            .observe(&EngineEvent::Queued {
                id,
                input_name: "x.mp4".into(),
                insight: None
            })
            .is_empty());
        id
    }

    #[test]
    fn single_job_lifecycle_inhibits_then_notifies() {
        let mut w = BatchWatcher::default();
        let id = queued(&mut w);
        assert_eq!(
            w.observe(&EngineEvent::Started { id }),
            vec![SystemAction::InhibitStart]
        );
        let actions = w.observe(&EngineEvent::Completed {
            id,
            input_bytes: 1000,
            output_bytes: 400,
        });
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], SystemAction::InhibitStop);
        assert!(matches!(
            &actions[1],
            SystemAction::Notify { body, .. } if body.contains("1 file encoded")
        ));
    }

    #[test]
    fn parallel_jobs_take_the_lock_exactly_once() {
        let mut w = BatchWatcher::default();
        let a = queued(&mut w);
        let b = queued(&mut w);
        assert_eq!(w.observe(&EngineEvent::Started { id: a }).len(), 1);
        assert!(w.observe(&EngineEvent::Started { id: b }).is_empty());
        // First completion: b still running -> lock stays, no notify.
        assert!(w
            .observe(&EngineEvent::Completed {
                id: a,
                input_bytes: 10,
                output_bytes: 5
            })
            .is_empty());
        let actions = w.observe(&EngineEvent::Completed {
            id: b,
            input_bytes: 10,
            output_bytes: 5,
        });
        assert_eq!(actions[0], SystemAction::InhibitStop);
        assert!(matches!(&actions[1], SystemAction::Notify { body, .. }
            if body.contains("2 files encoded")));
    }

    #[test]
    fn pause_scenario_releases_lock_without_notifying() {
        let mut w = BatchWatcher::default();
        let running = queued(&mut w);
        let held = queued(&mut w); // pending job held by a paused queue
        w.observe(&EngineEvent::Started { id: running });
        // Running job drains while paused: lock released (GPU idle, OS may
        // sleep) but the batch is NOT announced — a job is still pending.
        let actions = w.observe(&EngineEvent::Completed {
            id: running,
            input_bytes: 10,
            output_bytes: 5,
        });
        assert_eq!(actions, vec![SystemAction::InhibitStop]);
        // Resume later: lock re-taken, and the final notification counts
        // every completion since the batch began.
        assert_eq!(w.observe(&EngineEvent::Started { id: held }).len(), 1);
        let actions = w.observe(&EngineEvent::Completed {
            id: held,
            input_bytes: 10,
            output_bytes: 5,
        });
        assert!(matches!(&actions[1], SystemAction::Notify { body, .. }
            if body.contains("2 files encoded")));
    }

    #[test]
    fn cancelled_only_batch_stays_silent() {
        let mut w = BatchWatcher::default();
        let id = queued(&mut w);
        w.observe(&EngineEvent::Started { id });
        let actions = w.observe(&EngineEvent::Cancelled { id });
        assert_eq!(actions, vec![SystemAction::InhibitStop]);
    }

    #[test]
    fn failures_and_savings_reach_the_body() {
        assert_eq!(
            batch_body(7, 1, 2, 3_650_722_201),
            "7 files encoded · 2 already optimal · 1 failed · 3.4 GB saved"
        );
        assert_eq!(batch_body(1, 0, 0, -500), "1 file encoded");
    }

    #[test]
    fn skipped_jobs_end_the_batch_and_reach_the_notification() {
        let mut w = BatchWatcher::default();
        let enc = queued(&mut w);
        let skip = queued(&mut w);
        // The skipped job terminates straight from Pending: no inhibitor
        // action, no notification while the other job is still tracked.
        assert!(w
            .observe(&EngineEvent::Skipped {
                id: skip,
                reason: "already av1".into()
            })
            .is_empty());
        w.observe(&EngineEvent::Started { id: enc });
        let actions = w.observe(&EngineEvent::Completed {
            id: enc,
            input_bytes: 10,
            output_bytes: 5,
        });
        assert!(matches!(&actions[1], SystemAction::Notify { body, .. }
            if body.contains("1 file encoded · 1 already optimal")));
    }

    #[test]
    fn counters_reset_between_batches() {
        let mut w = BatchWatcher::default();
        let a = queued(&mut w);
        w.observe(&EngineEvent::Started { id: a });
        w.observe(&EngineEvent::Completed {
            id: a,
            input_bytes: 10,
            output_bytes: 5,
        });
        // Second batch must not inherit the first batch's counts.
        let b = queued(&mut w);
        w.observe(&EngineEvent::Started { id: b });
        let actions = w.observe(&EngineEvent::Completed {
            id: b,
            input_bytes: 10,
            output_bytes: 5,
        });
        assert!(matches!(&actions[1], SystemAction::Notify { body, .. }
            if body.contains("1 file encoded")));
    }

    #[test]
    fn video_util_parser_is_defensive() {
        assert_eq!(parse_video_util("42\n"), Some(42));
        assert_eq!(parse_video_util(" 0 "), Some(0));
        assert_eq!(parse_video_util("100"), Some(100));
        assert_eq!(parse_video_util("101"), None, "out of range");
        assert_eq!(parse_video_util("-3"), None);
        assert_eq!(parse_video_util("N/A"), None);
        assert_eq!(parse_video_util(""), None);
        assert_eq!(parse_video_util("[Insufficient Permissions]"), None);
    }

    #[test]
    fn progress_and_state_events_are_inert() {
        let mut w = BatchWatcher::default();
        assert!(w.observe(&EngineEvent::QueuePaused).is_empty());
        assert!(w.observe(&EngineEvent::QueueResumed).is_empty());
        assert!(w.observe(&EngineEvent::EngineStopped).is_empty());
    }
}
