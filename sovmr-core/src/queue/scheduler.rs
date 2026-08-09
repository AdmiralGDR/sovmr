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

//! The engine: a single control thread owning the queue, dispatching jobs
//! to up to `worker_slots` concurrent encode sessions (each an NVENC stream
//! in Phase 1's ffmpeg children), and translating raw backend progress into
//! UI-ready events.
//!
//! Concurrency model: the control thread is the sole owner of all queue
//! state — no shared mutable state, no locks around the queue. Workers and
//! progress relays communicate with it exclusively through its inbox
//! channel, and it communicates with sessions only through their
//! `CancelHandle`s. A panic in a worker thread converts to a `Failed` event
//! for that job; it can never take down the control thread or the UI.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::encoder::{CancelHandle, EncodeOutcome, EncoderProvider};
use crate::job::{already_optimal, JobId, MediaJob};
use crate::queue::messages::{EngineCommand, EngineEvent};

/// Everything that can wake the control thread.
enum Inbox {
    Ui(EngineCommand),
    /// Progress relay: precomputed fraction/ETA for one running job.
    Progress {
        id: JobId,
        fraction: Option<f32>,
        fps: f64,
        speed: f64,
        eta: Option<Duration>,
    },
    /// A worker finished (in any terminal state) and freed its slot.
    Done {
        id: JobId,
        result: DoneResult,
    },
}

enum DoneResult {
    Completed {
        input_bytes: u64,
        output_bytes: u64,
        warning: Option<String>,
    },
    Failed {
        reason: String,
    },
    Cancelled,
}

/// Handle held by the UI. Cloneable; dropping all handles does *not* stop
/// the engine — send `Shutdown` (or call
/// [`EngineHandle::shutdown_and_wait`]) for a deterministic stop.
#[derive(Clone)]
pub struct EngineHandle {
    inbox: Sender<Inbox>,
    /// The engine thread, joinable exactly once at shutdown. This Mutex
    /// is cold-path only (teardown); nothing in the encode/dispatch loop
    /// ever touches it.
    thread: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl EngineHandle {
    /// Deliver a command to the engine thread. Fire-and-forget: results
    /// come back on the event channel returned by [`start`].
    pub fn send(&self, cmd: EngineCommand) {
        // A dead engine thread makes sends fail; the UI learns about it
        // from the closed event channel, so the error is ignorable here.
        let _ = self.inbox.send(Inbox::Ui(cmd));
    }

    /// Graceful teardown: send `Shutdown` (cancelling active encodes,
    /// which roll their partial output back) and **block** until the
    /// engine thread has fully exited. Idempotent — later calls (and
    /// calls on other clones) return immediately. This is what makes
    /// "process exit" safe: no encode is ever killed mid-rollback.
    pub fn shutdown_and_wait(&self) {
        self.send(EngineCommand::Shutdown);
        let handle = match self.thread.lock() {
            Ok(mut slot) => slot.take(),
            Err(poison) => poison.into_inner().take(),
        };
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

/// Spawn the engine. Returns the command handle and the event stream —
/// a GUI drains it non-blocking once per frame; a headless embedder may
/// simply block on it.
///
/// # Example
///
/// ```ignore
/// let (engine, events) = scheduler::start(provider, plan.worker_slots);
/// engine.send(EngineCommand::Enqueue(job));
/// while let Ok(event) = events.recv() {
///     match event {
///         EngineEvent::Completed { id, .. } => println!("{id} done"),
///         EngineEvent::Failed { id, reason } => eprintln!("{id}: {reason}"),
///         _ => {}
///     }
/// }
/// ```
pub fn start(
    provider: Arc<dyn EncoderProvider>,
    worker_slots: usize,
) -> (EngineHandle, Receiver<EngineEvent>) {
    let (inbox_tx, inbox_rx) = channel::<Inbox>();
    let (event_tx, event_rx) = channel::<EngineEvent>();
    let thread = thread::Builder::new()
        .name("sovmr-engine".into())
        .spawn({
            let inbox_tx = inbox_tx.clone();
            move || {
                Engine {
                    provider,
                    inbox_tx,
                    events: event_tx,
                    pending: VecDeque::new(),
                    active: HashMap::new(),
                    worker_slots: worker_slots.max(1),
                    paused: false,
                    shutting_down: false,
                }
                .run(inbox_rx)
            }
        })
        .expect("spawning engine thread");
    let handle = EngineHandle {
        inbox: inbox_tx,
        thread: Arc::new(Mutex::new(Some(thread))),
    };
    (handle, event_rx)
}

struct Engine {
    provider: Arc<dyn EncoderProvider>,
    /// Cloned into workers/relays so they can reach the control thread.
    inbox_tx: Sender<Inbox>,
    events: Sender<EngineEvent>,
    pending: VecDeque<MediaJob>,
    active: HashMap<JobId, Arc<dyn CancelHandle>>,
    worker_slots: usize,
    /// While set, `fill_slots` is a no-op: running jobs drain naturally,
    /// pending jobs wait. Never touches in-flight sessions.
    paused: bool,
    shutting_down: bool,
}

impl Engine {
    fn run(mut self, inbox: Receiver<Inbox>) {
        // Blocking recv is correct here: every state change arrives as a
        // message, so there is nothing to poll.
        while let Ok(msg) = inbox.recv() {
            match msg {
                Inbox::Ui(cmd) => self.on_command(cmd),
                Inbox::Progress {
                    id,
                    fraction,
                    fps,
                    speed,
                    eta,
                } => {
                    // Suppress progress from jobs already cancelled/finished.
                    if self.active.contains_key(&id) {
                        self.emit(EngineEvent::Progress {
                            id,
                            fraction,
                            fps,
                            speed,
                            eta,
                        });
                    }
                }
                Inbox::Done { id, result } => self.on_done(id, result),
            }
            self.fill_slots();
            if self.shutting_down && self.active.is_empty() {
                break;
            }
        }
        self.emit(EngineEvent::EngineStopped);
    }

    fn on_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Enqueue(job) => {
                if self.shutting_down {
                    return;
                }
                self.emit(EngineEvent::Queued {
                    id: job.id,
                    input_name: job
                        .input
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| job.input.display().to_string()),
                    // Show, don't tell: the dashboard proves the engine
                    // actually analysed *this* file.
                    insight: crate::heuristics::insight(
                        &job.info,
                        job.settings.strategy,
                        job.settings.codec,
                    ),
                });
                // Smart Skip: sources already compact in the target codec
                // never reach a worker slot (generation-loss protection).
                // User-defeatable per job settings — forcing a re-encode
                // is a legitimate power-user move.
                if job.settings.smart_skip {
                    if let Some(reason) = already_optimal(&job.info, job.settings.codec) {
                        self.emit(EngineEvent::Skipped { id: job.id, reason });
                        return;
                    }
                }
                self.pending.push_back(job);
            }
            EngineCommand::Cancel(id) => self.cancel_one(id),
            EngineCommand::CancelAll => {
                let pending_ids: Vec<JobId> = self.pending.iter().map(|j| j.id).collect();
                let active_ids: Vec<JobId> = self.active.keys().copied().collect();
                for id in pending_ids.into_iter().chain(active_ids) {
                    self.cancel_one(id);
                }
            }
            EngineCommand::SetWorkerSlots(n) => {
                self.worker_slots = n.clamp(1, crate::config::NVENC_SESSION_CAP);
            }
            EngineCommand::PauseQueue => {
                if !self.paused {
                    self.paused = true;
                    self.emit(EngineEvent::QueuePaused);
                }
            }
            EngineCommand::ResumeQueue => {
                if self.paused {
                    self.paused = false;
                    self.emit(EngineEvent::QueueResumed);
                }
            }
            EngineCommand::Shutdown => {
                self.shutting_down = true;
                let ids: Vec<JobId> = self.active.keys().copied().collect();
                for id in ids {
                    self.cancel_one(id);
                }
                self.pending.clear();
            }
        }
    }

    fn cancel_one(&mut self, id: JobId) {
        if let Some(pos) = self.pending.iter().position(|j| j.id == id) {
            self.pending.remove(pos);
            self.emit(EngineEvent::Cancelled { id });
        } else if let Some(handle) = self.active.get(&id) {
            // Asynchronous: the session's worker observes the kill and
            // reports Done::Cancelled, which emits the event and frees the
            // slot. No double-emit here.
            handle.cancel();
        }
    }

    fn on_done(&mut self, id: JobId, result: DoneResult) {
        if self.active.remove(&id).is_none() {
            return; // stale message from an already-forgotten job
        }
        match result {
            DoneResult::Completed {
                input_bytes,
                output_bytes,
                warning,
            } => {
                // The advisory precedes the terminal event so the row
                // carries its warning the moment it flips to Completed.
                if let Some(message) = warning {
                    self.emit(EngineEvent::Warning { id, message });
                }
                self.emit(EngineEvent::Completed {
                    id,
                    input_bytes,
                    output_bytes,
                });
            }
            DoneResult::Failed { reason } => {
                self.emit(EngineEvent::Failed { id, reason });
            }
            DoneResult::Cancelled => self.emit(EngineEvent::Cancelled { id }),
        }
    }

    /// Start pending jobs while worker slots are free. One slot == one
    /// concurrent NVENC session.
    fn fill_slots(&mut self) {
        while !self.shutting_down
            && !self.paused
            && self.active.len() < self.worker_slots
            && !self.pending.is_empty()
        {
            let job = self.pending.pop_front().expect("checked non-empty");
            self.dispatch(job);
        }
    }

    fn dispatch(&mut self, job: MediaJob) {
        let id = job.id;
        let (progress_tx, progress_rx) = channel();
        let session = match self.provider.spawn(&job, progress_tx) {
            Ok(s) => s,
            Err(e) => {
                // Failure to even start is a per-job failure, never fatal.
                self.emit(EngineEvent::Failed {
                    id,
                    reason: e.to_string(),
                });
                return;
            }
        };
        self.active.insert(id, session.cancel_handle());
        self.emit(EngineEvent::Started { id });

        // Progress relay: turns backend heartbeats into UI-ready fractions.
        let relay_inbox = self.inbox_tx.clone();
        let duration = job.info.duration;
        thread::spawn(move || {
            while let Ok(update) = progress_rx.recv() {
                let fraction = duration.map(|total| {
                    (update.out_time.as_secs_f64() / total.as_secs_f64().max(f64::EPSILON))
                        .clamp(0.0, 1.0) as f32
                });
                let eta = duration.and_then(|total| {
                    let remaining = total.checked_sub(update.out_time)?;
                    (update.speed > 0.0)
                        .then(|| Duration::from_secs_f64(remaining.as_secs_f64() / update.speed))
                });
                let msg = Inbox::Progress {
                    id,
                    fraction,
                    fps: update.fps,
                    speed: update.speed,
                    eta,
                };
                if relay_inbox.send(msg).is_err() {
                    break;
                }
            }
        });

        // Worker: parks in `wait` until the session reaches a terminal
        // state, then frees the slot via the inbox.
        let worker_inbox = self.inbox_tx.clone();
        let input_bytes = job.info.size_bytes;
        thread::Builder::new()
            .name(format!("sovmr-worker-{id}"))
            .spawn(move || {
                let result = match session.wait() {
                    Ok(EncodeOutcome::Completed {
                        output_bytes,
                        warning,
                    }) => DoneResult::Completed {
                        input_bytes,
                        output_bytes,
                        warning,
                    },
                    Ok(EncodeOutcome::Cancelled) => DoneResult::Cancelled,
                    Err(e) => DoneResult::Failed {
                        reason: e.to_string(),
                    },
                };
                let _ = worker_inbox.send(Inbox::Done { id, result });
            })
            .expect("spawning worker thread");
    }

    fn emit(&self, event: EngineEvent) {
        // The UI dropping its receiver must never wedge the engine.
        let _ = self.events.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{EncodeSession, EncoderCapabilities, EncoderError, ProgressUpdate};
    use crate::job::{Codec, EncodeSettings, MediaInfo};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Instant;

    /// Deterministic in-memory backend proving the Facade holds: the
    /// scheduler under test runs unmodified against a provider that has
    /// nothing to do with ffmpeg.
    struct MockProvider {
        /// How long each fake encode takes.
        encode_time: Duration,
        /// Inputs whose file name contains this marker "fail" to encode.
        fail_marker: &'static str,
        peak_concurrency: Arc<AtomicUsize>,
        running_now: Arc<AtomicUsize>,
    }

    impl MockProvider {
        fn new(encode_time: Duration) -> Self {
            MockProvider {
                encode_time,
                fail_marker: "corrupt",
                peak_concurrency: Arc::new(AtomicUsize::new(0)),
                running_now: Arc::new(AtomicUsize::new(0)),
            }
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
        deadline: Instant,
        fail: bool,
        running_now: Arc<AtomicUsize>,
    }

    impl EncodeSession for MockSession {
        fn cancel_handle(&self) -> Arc<dyn CancelHandle> {
            self.cancel.clone()
        }
        fn wait(self: Box<Self>) -> Result<EncodeOutcome, EncoderError> {
            let poll = Duration::from_millis(1);
            while Instant::now() < self.deadline {
                if self.cancel.0.load(Ordering::SeqCst) {
                    self.running_now.fetch_sub(1, Ordering::SeqCst);
                    return Ok(EncodeOutcome::Cancelled);
                }
                thread::sleep(poll);
            }
            self.running_now.fetch_sub(1, Ordering::SeqCst);
            if self.fail {
                Err(EncoderError::EncodeFailed("mock: corrupt bitstream".into()))
            } else {
                Ok(EncodeOutcome::Completed {
                    output_bytes: 42,
                    warning: None,
                })
            }
        }
    }

    impl EncoderProvider for MockProvider {
        fn id(&self) -> &'static str {
            "mock"
        }
        fn capabilities(&self) -> Result<EncoderCapabilities, EncoderError> {
            Ok(EncoderCapabilities {
                av1: true,
                hevc: true,
            })
        }
        fn probe(&self, _input: &Path) -> Result<MediaInfo, EncoderError> {
            Ok(MediaInfo::default())
        }
        fn spawn(
            &self,
            job: &MediaJob,
            _progress: Sender<ProgressUpdate>,
        ) -> Result<Box<dyn EncodeSession>, EncoderError> {
            let now = self.running_now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_concurrency.fetch_max(now, Ordering::SeqCst);
            Ok(Box::new(MockSession {
                cancel: Arc::new(MockCancel(AtomicBool::new(false))),
                deadline: Instant::now() + self.encode_time,
                fail: job.input.to_string_lossy().contains(self.fail_marker),
                running_now: self.running_now.clone(),
            }))
        }
    }

    fn mock_job(name: &str) -> MediaJob {
        MediaJob {
            id: JobId::next(),
            input: PathBuf::from(format!("/mock/{name}")),
            output: PathBuf::from(format!("/mock/{name}.out")),
            settings: EncodeSettings::visually_lossless(Codec::Av1),
            info: MediaInfo::default(),
        }
    }

    /// Drain events until the predicate says stop or the timeout expires.
    fn drain_until(
        rx: &Receiver<EngineEvent>,
        mut stop: impl FnMut(&[EngineEvent]) -> bool,
    ) -> Vec<EngineEvent> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut seen = Vec::new();
        while Instant::now() < deadline && !stop(&seen) {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(50)) {
                seen.push(ev);
            }
        }
        seen
    }

    fn terminal_count(events: &[EngineEvent]) -> usize {
        events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineEvent::Completed { .. }
                        | EngineEvent::Failed { .. }
                        | EngineEvent::Cancelled { .. }
                        | EngineEvent::Skipped { .. }
                )
            })
            .count()
    }

    #[test]
    fn disabling_smart_skip_forces_the_encode() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
        let (engine, events) = start(provider, 1);
        // Identical compact-AV1 facts, but the user turned Smart Skip off.
        let mut forced = mock_job("compact_but_forced.mp4");
        forced.info = MediaInfo {
            duration: Some(Duration::from_secs(100)),
            width: 1920,
            height: 1080,
            video_codec: "av1".into(),
            size_bytes: 10_000_000,
            video_bytes: 0,
            fps: 30.0,
            audio_bitrate: 0,
            ..Default::default()
        };
        forced.settings.smart_skip = false;
        let id = forced.id;
        engine.send(EngineCommand::Enqueue(forced));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 1);
        assert!(
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Started { id: s } if *s == id)),
            "with smart_skip off the job must actually encode: {seen:?}"
        );
        assert!(seen
            .iter()
            .any(|e| matches!(e, EngineEvent::Completed { .. })));
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn already_optimal_sources_are_skipped_without_encoding() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
        let (engine, events) = start(provider, 2);
        // A compact AV1 source targeted at AV1: must skip, not encode.
        let mut compact = mock_job("compact_av1.mp4");
        compact.info = MediaInfo {
            duration: Some(Duration::from_secs(100)),
            width: 1920,
            height: 1080,
            video_codec: "av1".into(),
            size_bytes: 10_000_000,
            video_bytes: 0,
            fps: 30.0,
            audio_bitrate: 0,
            ..Default::default()
        };
        let skip_id = compact.id;
        engine.send(EngineCommand::Enqueue(compact));
        engine.send(EngineCommand::Enqueue(mock_job("normal_h264.mp4")));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 2);
        assert!(
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Skipped { id, reason }
                if *id == skip_id && reason.contains("already av1"))),
            "compact AV1 must be skipped: {seen:?}"
        );
        assert!(
            !seen
                .iter()
                .any(|e| matches!(e, EngineEvent::Started { id } if *id == skip_id)),
            "skipped job must never reach a worker slot"
        );
        assert!(
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Completed { .. })),
            "the normal job still encodes: {seen:?}"
        );
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn respects_worker_slot_limit() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(50)));
        let peak = provider.peak_concurrency.clone();
        let (engine, events) = start(provider, 2);
        for i in 0..6 {
            engine.send(EngineCommand::Enqueue(mock_job(&format!("clip{i}.mp4"))));
        }
        let seen = drain_until(&events, |seen| terminal_count(seen) == 6);
        assert_eq!(terminal_count(&seen), 6);
        assert!(peak.load(Ordering::SeqCst) <= 2, "slot limit breached");
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn one_bad_file_does_not_stop_the_batch() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
        let (engine, events) = start(provider, 2);
        engine.send(EngineCommand::Enqueue(mock_job("good1.mp4")));
        engine.send(EngineCommand::Enqueue(mock_job("corrupt.mp4")));
        engine.send(EngineCommand::Enqueue(mock_job("good2.mp4")));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 3);
        let completed = seen
            .iter()
            .filter(|e| matches!(e, EngineEvent::Completed { .. }))
            .count();
        let failed: Vec<_> = seen
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Failed { reason, .. } => Some(reason.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(completed, 2);
        assert_eq!(failed.len(), 1);
        assert!(failed[0].contains("corrupt bitstream"), "{failed:?}");
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn cancel_dequeues_pending_job_before_it_starts() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(100)));
        let (engine, events) = start(provider, 1);
        let running = mock_job("running.mp4");
        let queued = mock_job("queued.mp4");
        let queued_id = queued.id;
        engine.send(EngineCommand::Enqueue(running));
        engine.send(EngineCommand::Enqueue(queued));
        engine.send(EngineCommand::Cancel(queued_id));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 2);
        assert!(
            seen.contains(&EngineEvent::Cancelled { id: queued_id }),
            "queued job should be cancelled: {seen:?}"
        );
        // The cancelled job must never have started.
        assert!(!seen
            .iter()
            .any(|e| matches!(e, EngineEvent::Started { id } if *id == queued_id)));
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn cancel_kills_running_job() {
        let provider = Arc::new(MockProvider::new(Duration::from_secs(30)));
        let (engine, events) = start(provider, 1);
        let job = mock_job("long.mp4");
        let id = job.id;
        engine.send(EngineCommand::Enqueue(job));
        let seen = drain_until(&events, |seen| {
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Started { .. }))
        });
        assert!(!seen.is_empty());
        engine.send(EngineCommand::Cancel(id));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 1);
        assert!(seen.contains(&EngineEvent::Cancelled { id }));
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn pause_lets_running_finish_but_holds_pending_until_resume() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(40)));
        let (engine, events) = start(provider, 1);
        let first = mock_job("first.mp4");
        let second = mock_job("second.mp4");
        let second_id = second.id;
        engine.send(EngineCommand::Enqueue(first));
        engine.send(EngineCommand::Enqueue(second));
        // Pause as soon as the first job is on a worker slot.
        drain_until(&events, |seen| {
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Started { .. }))
        });
        engine.send(EngineCommand::PauseQueue);

        // The running job must complete (never killed by a pause)...
        let seen = drain_until(&events, |seen| terminal_count(seen) == 1);
        assert!(
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Completed { .. })),
            "running job should finish during pause: {seen:?}"
        );
        assert!(seen.contains(&EngineEvent::QueuePaused));

        // ...while the pending one must NOT start during the pause window.
        let deadline = Instant::now() + Duration::from_millis(200);
        let mut leaked = Vec::new();
        while Instant::now() < deadline {
            if let Ok(ev) = events.recv_timeout(Duration::from_millis(50)) {
                leaked.push(ev);
            }
        }
        assert!(
            !leaked
                .iter()
                .any(|e| matches!(e, EngineEvent::Started { id } if *id == second_id)),
            "paused queue must not dispatch pending jobs: {leaked:?}"
        );

        engine.send(EngineCommand::ResumeQueue);
        let seen = drain_until(&events, |seen| terminal_count(seen) == 1);
        assert!(seen.contains(&EngineEvent::QueueResumed));
        assert!(
            seen.iter()
                .any(|e| matches!(e, EngineEvent::Completed { .. })),
            "resume must dispatch the held job: {seen:?}"
        );
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn pause_and_resume_are_idempotent() {
        let provider = Arc::new(MockProvider::new(Duration::from_millis(10)));
        let (engine, events) = start(provider, 1);
        engine.send(EngineCommand::PauseQueue);
        engine.send(EngineCommand::PauseQueue);
        engine.send(EngineCommand::ResumeQueue);
        engine.send(EngineCommand::ResumeQueue);
        engine.send(EngineCommand::Enqueue(mock_job("after.mp4")));
        let seen = drain_until(&events, |seen| terminal_count(seen) == 1);
        // Exactly one state-change event per actual transition.
        let pauses = seen
            .iter()
            .filter(|e| **e == EngineEvent::QueuePaused)
            .count();
        let resumes = seen
            .iter()
            .filter(|e| **e == EngineEvent::QueueResumed)
            .count();
        assert_eq!((pauses, resumes), (1, 1), "{seen:?}");
        engine.send(EngineCommand::Shutdown);
    }

    #[test]
    fn shutdown_and_wait_blocks_until_the_engine_is_gone() {
        let provider = Arc::new(MockProvider::new(Duration::from_secs(30)));
        let (engine, events) = start(provider, 2);
        engine.send(EngineCommand::Enqueue(mock_job("long_a.mp4")));
        engine.send(EngineCommand::Enqueue(mock_job("long_b.mp4")));
        drain_until(&events, |seen| {
            seen.iter()
                .filter(|e| matches!(e, EngineEvent::Started { .. }))
                .count()
                == 2
        });
        // Blocks through cancellation of both running encodes; after it
        // returns the engine thread no longer exists, so the event
        // channel must be closed once drained.
        engine.shutdown_and_wait();
        let mut seen = Vec::new();
        while let Ok(ev) = events.try_recv() {
            seen.push(ev);
        }
        assert!(seen.contains(&EngineEvent::EngineStopped), "{seen:?}");
        assert!(
            matches!(
                events.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Disconnected)
            ),
            "engine thread exited, channel closed"
        );
        // Idempotent, including on clones.
        engine.clone().shutdown_and_wait();
        engine.shutdown_and_wait();
    }

    #[test]
    fn shutdown_cancels_everything_and_stops() {
        let provider = Arc::new(MockProvider::new(Duration::from_secs(30)));
        let (engine, events) = start(provider, 2);
        engine.send(EngineCommand::Enqueue(mock_job("a.mp4")));
        engine.send(EngineCommand::Enqueue(mock_job("b.mp4")));
        engine.send(EngineCommand::Enqueue(mock_job("c.mp4")));
        engine.send(EngineCommand::Shutdown);
        let seen = drain_until(&events, |seen| seen.contains(&EngineEvent::EngineStopped));
        assert!(seen.contains(&EngineEvent::EngineStopped));
    }
}
