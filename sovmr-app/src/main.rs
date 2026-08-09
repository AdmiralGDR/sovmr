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

//! Composition root. The ONLY place where core, an encoder backend, and
//! the egui shell meet. The engine is [`NvencNativeProvider`] — the
//! sovereign zero-copy NVDEC→VRAM→NVENC pipeline; swapping backends is
//! one line here and nothing else in the workspace changes (that is how
//! the FFmpeg era ended).
//!
//! Usage: `sovmr [--version|--report|--mock] [FILE...]`
//! * `FILE...`  — media files to enqueue immediately (this is how the
//!   file-manager "Compress with sovmr" action hands files over).
//! * `--version` — name this build exactly (release, commit, target).
//! * `--report` — print the hardware plan and exit (headless check).
//! * `--mock`   — simulation mode: fake GPU plan, fake staggered encodes
//!   through the *real* scheduler and UI; seeds a demo queue. For UI
//!   verification on any machine.

mod benchmark;
mod integration;
mod logger;
mod mock;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sovmr_core::config::NVENC_SESSION_CAP;
use sovmr_core::encoder::EncoderProvider;
use sovmr_core::hardware::{EncodePlan, GpuInfo, HardwareConfigurator};

use sovmr_core::job::{Codec, EncodeSettings, JobId, MediaJob};
use sovmr_core::queue::messages::{EngineCommand, EngineEvent};
use sovmr_core::queue::scheduler::{self, EngineHandle};
use sovmr_core::validate::validate_media_file;
use sovmr_native::NvencNativeProvider;
use sovmr_ui::bridge::UiBridge;
use sovmr_ui::i18n::Lang;
use sovmr_ui::shell::{self, CodecChoice, ShellConfig, SovmrShell, UiEncoderSettings};

/// CLI flag naming this build exactly and exiting.
///
/// Two machines run this program, and a fix is not delivered until it is
/// on both. Which build is where has to be a question the binary
/// answers, not one inferred from a file date.
const VERSION_FLAG: &str = "--version";
/// CLI flag for the headless hardware report.
const REPORT_FLAG: &str = "--report";
/// CLI flag for the UI simulation mode.
const MOCK_FLAG: &str = "--mock";
/// CLI flag for the sovereign-vs-ffmpeg benchmark harness.
const BENCHMARK_FLAG: &str = "--benchmark";
/// CLI flag forcing the X11/XWayland window backend. Upstream winit has
/// no Wayland drag-and-drop, so on Wayland sessions this is the switch
/// that brings OS drops back (the Add-media picker works everywhere).
const X11_FLAG: &str = "--x11";
/// The environment variable winit keys its backend choice on.
const WAYLAND_ENV: &str = "WAYLAND_DISPLAY";
/// How often the watcher thread polls the (async-signal-safe) flag the
/// SIGINT/SIGTERM handler sets. Cheap: one atomic load per tick.
const SIGNAL_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// The one repaint waker, filled in by the shell at startup and shared
/// by everything that must wake an idle window (engine events, the
/// signal watcher). Cold-path lock: taken per *event*, never per frame.
type SharedWaker = Arc<Mutex<Option<shell::RepaintWaker>>>;

/// Wake the window if the shell has published its waker already.
fn wake(waker: &SharedWaker) {
    if let Ok(guard) = waker.lock() {
        if let Some(wake) = guard.as_ref() {
            wake();
        }
    }
}

/// File names seeded into the demo queue in `--mock` mode. One contains
/// [`mock::FAIL_MARKER`] to exercise the failure row.
const MOCK_DEMO_FILES: [&str; 8] = [
    "holiday_4k.mp4",
    "drone_footage.mp4",
    "family_dinner.mp4",
    "corrupt_stream.mp4",
    "concert_night.mp4",
    "screen_capture.mp4",
    "road_trip_day1.mp4",
    "road_trip_day2.mp4",
];

fn main() {
    // Unix pipe manners first: `sovmr --report | head` must end quietly
    // (kernel default), not with Rust's broken-pipe panic.
    let _ = sovmr_os::signals::default_sigpipe();
    let args: Vec<String> = std::env::args().collect();
    // Before anything else can fail: a build that cannot start must
    // still be able to say which build it is.
    if args.iter().any(|a| a == VERSION_FLAG) {
        println!("{}", sovmr_core::identity::describe_full());
        return;
    }
    let mock_mode = args.iter().any(|a| a == MOCK_FLAG);
    // `--x11`: drop off Wayland *before* any window machinery exists so
    // winit selects the X11/XWayland backend (which has working DnD).
    if args.iter().any(|a| a == X11_FLAG) {
        std::env::remove_var(WAYLAND_ENV);
    }
    let dnd_available = std::env::var_os(WAYLAND_ENV).is_none();
    if let Some(log_path) = logger::init() {
        sovlog!(
            "startup: {} ({} mode, dnd {}) — log at {}",
            sovmr_core::identity::describe(),
            if mock_mode { "simulation" } else { "hardware" },
            if dnd_available {
                "native"
            } else {
                "picker/CLI only (Wayland)"
            },
            log_path.display()
        );
    }
    // Everything that isn't a flag is a file to enqueue (the Freedesktop
    // `%F` field code in sovmr.desktop expands to plain absolute paths).
    let cli_paths: Vec<PathBuf> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();

    let setup: Result<(Arc<dyn EncoderProvider>, EncodePlan, String), String> = if mock_mode {
        Ok(mock_setup())
    } else {
        real_setup()
    };
    let (provider, plan, engine_note) = match setup {
        Ok(parts) => parts,
        Err(reason) => {
            // Pre-flight failed: no NVIDIA stack. Headless callers get
            // the reason on stderr; double-click launches get a window
            // that explains itself instead of a silent crash.
            sovlog!("preflight: hardware unsupported: {reason}");
            eprintln!("sovmr: no usable encoder backend: {reason}");
            if !args.iter().any(|a| a == REPORT_FLAG) {
                eprintln!("hint: run `sovmr {REPORT_FLAG}` for the full diagnosis");
                run_unsupported_window(reason);
            }
            std::process::exit(1);
        }
    };

    if args.iter().any(|a| a == BENCHMARK_FLAG) {
        let Some(input) = cli_paths.first() else {
            eprintln!("usage: sovmr {BENCHMARK_FLAG} <input.mp4>");
            std::process::exit(2);
        };
        std::process::exit(benchmark::run(provider.as_ref(), input));
    }

    if args.iter().any(|a| a == REPORT_FLAG) {
        print_report(provider.as_ref(), &plan, &engine_note);
        return;
    }

    // Live encoder settings: written by the UI's Advanced tab, read by the
    // intake thread for every newly added file.
    let live_settings = Arc::new(Mutex::new(plan.settings.clone()));

    // Engine (background worker pool) — the UI never sees the provider.
    let (engine, engine_events) = scheduler::start(provider.clone(), plan.worker_slots);

    // Event fan-in: the UI drains ONE channel. The forwarder relays engine
    // events (waking the window per event once the RepaintWaker arrives),
    // folds them into the desktop-integration watcher (sleep inhibition
    // while encoding, notification when the batch drains), and the intake
    // thread injects synthetic Queued+Failed pairs for files rejected
    // before they become jobs.
    let (ui_events_tx, ui_events_rx) = channel::<EngineEvent>();
    let waker: SharedWaker = Arc::new(Mutex::new(None));
    let forwarder = {
        let ui_events_tx = ui_events_tx.clone();
        let waker = waker.clone();
        thread::Builder::new()
            .name("sovmr-event-forwarder".into())
            .spawn(move || {
                let mut watcher = integration::BatchWatcher::default();
                // Owns the systemd-inhibit child; Drop releases the lock
                // when the engine stops and this thread unwinds.
                let mut system = integration::SystemBridge::default();
                while let Ok(event) = engine_events.recv() {
                    // Sovereign telemetry: pipeline problems and results
                    // land in ~/.local/state/sovmr/sovmr.log verbatim.
                    match &event {
                        EngineEvent::Failed { id, reason } => {
                            sovlog!("pipeline: job {id} FAILED: {reason}");
                        }
                        EngineEvent::Warning { id, message } => {
                            sovlog!("pipeline: job {id} warning: {message}");
                        }
                        EngineEvent::Completed {
                            id, output_bytes, ..
                        } => {
                            sovlog!("pipeline: job {id} completed ({output_bytes} bytes)");
                        }
                        _ => {}
                    }
                    for action in watcher.observe(&event) {
                        system.apply(action);
                    }
                    if ui_events_tx.send(event).is_err() {
                        break; // UI gone; engine shutdown follows via commands
                    }
                    wake(&waker); // repaint exactly when something changed
                }
            })
            .expect("spawning event forwarder")
    };

    // Sovereign signal handling: SIGINT/SIGTERM flip a flag; this
    // watcher turns it into the same teardown the window close uses —
    // cancel encodes (rolling back .part files), close the window, and
    // let the single exit path below wait for the engine. A second
    // signal exits immediately (see signals.rs).
    let close_flag = Arc::new(AtomicBool::new(false));
    if let Err(signum) = sovmr_os::signals::install() {
        eprintln!("sovmr: cannot install handler for signal {signum}; Ctrl+C will be abrupt");
    }
    {
        let engine = engine.clone();
        let close_flag = close_flag.clone();
        let waker = waker.clone();
        thread::Builder::new()
            .name("sovmr-signal-watcher".into())
            .spawn(move || loop {
                if sovmr_os::signals::termination_requested() {
                    eprintln!(
                        "sovmr: termination signal — cancelling encodes, \
                         rolling back partial output"
                    );
                    close_flag.store(true, Ordering::SeqCst);
                    engine.send(EngineCommand::Shutdown);
                    wake(&waker); // an idle window repaints and sees the flag
                    break;
                }
                thread::sleep(SIGNAL_POLL_INTERVAL);
            })
            .expect("spawning signal watcher");
    }

    // Intake: dropped paths -> validate -> probe -> enqueue, off the UI
    // thread so a slow disk or a huge folder never stalls a frame.
    let (intake_tx, intake_rx) = channel::<Vec<PathBuf>>();
    {
        let engine = engine.clone();
        let provider = provider.clone();
        let live_settings = live_settings.clone();
        thread::Builder::new()
            .name("sovmr-intake".into())
            .spawn(move || {
                while let Ok(paths) = intake_rx.recv() {
                    for path in expand_dirs(paths) {
                        let settings = live_settings
                            .lock()
                            .map(|s| s.clone())
                            .unwrap_or_else(|poison| poison.into_inner().clone());
                        ingest(
                            &path,
                            provider.as_ref(),
                            settings,
                            &engine,
                            &ui_events_tx,
                            mock_mode,
                        );
                    }
                }
            })
            .expect("spawning intake thread");
    }

    if mock_mode {
        seed_demo_queue(provider.as_ref(), &plan, &engine);
    }
    // Files handed over on the command line (file-manager context action)
    // enter the exact same validate -> probe -> enqueue path as drops.
    if !cli_paths.is_empty() {
        let _ = intake_tx.send(cli_paths);
    }

    let bridge = UiBridge::new(ui_events_rx, {
        let engine = engine.clone();
        move |cmd| engine.send(cmd)
    });
    let config = ShellConfig {
        gpu_name: plan.gpu.name.clone(),
        vram_mib: plan.gpu.vram_mib,
        worker_slots: plan.worker_slots,
        session_cap: NVENC_SESSION_CAP,
        engine_id: format!("{} — {engine_note}", provider.id()),
        auto_codec: plan.settings.codec,
        dnd_available,
    };
    let auto_codec = plan.settings.codec;
    let settings_sink = Box::new(move |ui_settings: UiEncoderSettings| {
        if let Ok(mut s) = live_settings.lock() {
            *s = to_encode_settings(&ui_settings, auto_codec);
        }
    });
    // The language chosen on a previous run, if any: a stored code that
    // is not one we know falls back to the default rather than guessing.
    let initial_lang = sovmr_os::desktop::read_preference("language")
        .and_then(|code| Lang::from_code(&code))
        .unwrap_or_default();
    let shell = SovmrShell::new(
        bridge,
        config,
        Box::new(move |paths| {
            let _ = intake_tx.send(paths);
        }),
        settings_sink,
    )
    .with_language(initial_lang)
    .with_language_sink(|lang| {
        // Persist the choice so the next launch opens in it. Best effort,
        // but SAID on failure — a preference silently lost is exactly the
        // silence this project forbids.
        if let Err(e) = sovmr_os::desktop::write_preference("language", lang.code()) {
            eprintln!("sovmr: could not save the language preference: {e}");
        }
    })
    .with_close_request(close_flag.clone())
    .with_raw_preview(move |ui_settings| {
        native_preview(&to_encode_settings(ui_settings, auto_codec))
    })
    .with_gpu_meter({
        // 1 Hz background poller (nvidia-smi, or a waveform in mock mode);
        // the UI reads a lock-free atomic each frame.
        let gpu_util = integration::spawn_gpu_meter(mock_mode);
        Box::new(move || {
            let v = gpu_util.load(Ordering::Relaxed);
            u8::try_from(v).ok().filter(|p| *p <= 100)
        })
    });

    let title = if mock_mode {
        "sovmr (simulation)"
    } else {
        "sovmr"
    };
    let run_result = shell::run(title, shell, {
        let waker = waker.clone();
        move |repaint| {
            if let Ok(mut slot) = waker.lock() {
                *slot = Some(repaint);
            }
        }
    });

    // THE teardown path — window close, Ctrl+C, and SIGTERM all land
    // here: cancel active encodes and *wait* for their rollback (no
    // .part litter, no leaked CUDA contexts), then let the forwarder
    // drain — dropping its SystemBridge releases the systemd-inhibit
    // lock before the process exits.
    engine.shutdown_and_wait();
    let _ = forwarder.join();

    if let Err(e) = run_result {
        eprintln!("fatal: cannot open a window: {e}");
        eprintln!("hint: run `sovmr {REPORT_FLAG}` for a headless hardware check");
        std::process::exit(1);
    }
}

/// Real hardware path: the sovereign engine + nvidia-smi. Capabilities
/// are probed by actually loading the NVENC runtime (dlopen of the
/// driver's own libraries), so a missing/foreign GPU stack fails *here*
/// with a symbolic reason — surfaced as the pre-flight window, never a
/// crash mid-encode.
///
/// # Errors
///
/// A human-readable reason (the dlopen/probe diagnostic verbatim).
fn real_setup() -> Result<(Arc<dyn EncoderProvider>, EncodePlan, String), String> {
    let provider: Arc<dyn EncoderProvider> = Arc::new(NvencNativeProvider);
    provider.capabilities().map_err(|e| e.to_string())?;
    let engine_note = "sovereign zero-copy NVDEC\u{2192}NVENC".to_string();
    let plan = HardwareConfigurator::detect(provider.as_ref())
        .map_err(|e| format!("hardware detection failed: {e}"))?;
    Ok((provider, plan, engine_note))
}

/// Minimal window for machines that cannot run the engine at all: the
/// shell renders its full-window explanation off one synthetic
/// [`EngineEvent::HardwareUnsupported`]. No engine, no intake — there
/// is nothing to queue without the silicon.
fn run_unsupported_window(reason: String) {
    let (event_tx, event_rx) = channel::<EngineEvent>();
    let _ = event_tx.send(EngineEvent::HardwareUnsupported { reason });
    let bridge = UiBridge::new(event_rx, |_cmd| {});
    let config = ShellConfig {
        gpu_name: "no compatible GPU".to_string(),
        vram_mib: None,
        worker_slots: 0,
        session_cap: 0,
        engine_id: "unavailable".to_string(),
        auto_codec: Codec::Hevc,
        dnd_available: false,
    };
    let shell = SovmrShell::new(bridge, config, Box::new(|_| {}), Box::new(|_| {}));
    if let Err(e) = shell::run("sovmr — unsupported hardware", shell, |_waker| {}) {
        // No window *and* no GPU: the stderr text above is all we have.
        eprintln!("sovmr: cannot open the explanation window either: {e}");
    }
}

/// Simulation path: fabricated Ada GPU + the mock provider, still planned
/// through the real configurator logic.
fn mock_setup() -> (Arc<dyn EncoderProvider>, EncodePlan, String) {
    let provider: Arc<dyn EncoderProvider> = Arc::new(mock::MockProvider);
    let gpu = GpuInfo {
        name: "Simulated RTX 4070 Laptop GPU".to_string(),
        vram_mib: Some(8188),
        driver_version: "simulation".to_string(),
    };
    let caps = provider
        .capabilities()
        .expect("mock capabilities are infallible");
    let plan = HardwareConfigurator::plan(gpu, caps).expect("mock plan is always satisfiable");
    (provider, plan, "simulation".to_string())
}

/// Fill the queue with staggered fake jobs so the UI shows parallel bars,
/// a failure, and completions within the first minute.
fn seed_demo_queue(provider: &dyn EncoderProvider, plan: &EncodePlan, engine: &EngineHandle) {
    for name in MOCK_DEMO_FILES {
        let path = PathBuf::from(format!("/simulated/{name}"));
        let info = provider.probe(&path).expect("mock probe is infallible");
        engine.send(EngineCommand::Enqueue(MediaJob {
            id: JobId::next(),
            input: path.clone(),
            output: MediaJob::derive_output(&path),
            settings: plan.settings.clone(),
            info,
        }));
    }
}

/// The Advanced tab's "what will actually run" line — the sovereign
/// pipeline described honestly instead of an ffmpeg command string.
fn native_preview(s: &EncodeSettings) -> String {
    format!(
        "NVDEC \u{2192} VRAM \u{2192} NVENC {:?} \u{b7} {} (CQ {}, ceiling from source) \u{b7} {:?} \u{b7} audio copied bit-exact \u{b7} fast-start MP4",
        s.codec,
        s.strategy.label(),
        s.strategy.cq(),
        s.effort,
    )
}

/// Map the UI's zero-jargon settings onto core encode settings.
fn to_encode_settings(ui: &UiEncoderSettings, auto_codec: Codec) -> EncodeSettings {
    EncodeSettings {
        codec: match ui.codec {
            CodecChoice::Auto => auto_codec,
            CodecChoice::Av1 => Codec::Av1,
            CodecChoice::Hevc => Codec::Hevc,
        },
        effort: ui.effort,
        strategy: ui.strategy,
        audio: sovmr_core::job::AudioPolicy::Copy,
        smart_skip: ui.smart_skip,
    }
}

/// Depth-1 folder expansion: dropping a directory enqueues its files.
/// Deeper nesting stays out of scope until the UI can show a scan state.
fn expand_dirs(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                let mut files: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.is_file())
                    .collect();
                files.sort();
                out.extend(files);
            }
        } else {
            out.push(path);
        }
    }
    out
}

/// Validate + probe one file; enqueue it, or surface the rejection in the
/// dashboard as a synthetic Queued -> Failed pair (per-file failures never
/// abort the batch). Mock mode skips disk validation —
/// simulated paths don't exist.
fn ingest(
    path: &std::path::Path,
    provider: &dyn EncoderProvider,
    settings: EncodeSettings,
    engine: &EngineHandle,
    ui_events: &Sender<EngineEvent>,
    skip_validation: bool,
) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let reject = |reason: String| {
        sovlog!("intake: REJECTED {}: {reason}", path.display());
        let id = JobId::next();
        let _ = ui_events.send(EngineEvent::Queued {
            id,
            input_name: name.clone(),
            insight: None, // rejected before probing: nothing measured
        });
        let _ = ui_events.send(EngineEvent::Failed { id, reason });
    };

    if !skip_validation {
        if let Err(e) = validate_media_file(path) {
            reject(format!("rejected: {e}"));
            return;
        }
    }
    let info = match provider.probe(path) {
        Ok(info) => info,
        Err(e) => {
            reject(e.to_string());
            return;
        }
    };
    let output = MediaJob::derive_output(path);
    if !skip_validation && output.exists() {
        reject("output already exists (won't overwrite)".to_string());
        return;
    }
    sovlog!(
        "intake: enqueued {} ({}x{} {}, {} bytes) -> {}",
        path.display(),
        info.width,
        info.height,
        info.video_codec,
        info.size_bytes,
        output.display()
    );
    engine.send(EngineCommand::Enqueue(MediaJob {
        id: JobId::next(),
        input: path.to_path_buf(),
        output,
        settings,
        info,
    }));
}

fn print_report(provider: &dyn EncoderProvider, plan: &EncodePlan, engine_note: &str) {
    println!("encoder backend : {}", provider.id());
    println!("engine source   : {engine_note}");
    // Engine diagnostics: the NVENC/NVDEC runtimes THE pipeline drives.
    match sovmr_nvenc::NvEncApi::load() {
        Ok(api) => {
            println!(
                "native nvenc    : ready — driver API {}.{}, {} entry points",
                api.driver_version().0,
                api.driver_version().1,
                api.entry_point_count()
            );
            // Deeper probe: sovereign CUDA context + real encode session,
            // opened and torn down cleanly.
            let session_state = match sovmr_cuda::Cuda::load() {
                Ok(cuda) => match cuda.create_context(0) {
                    Ok(ctx) => match api.open_session(&ctx) {
                        Ok(_session) => "opened & closed cleanly on CUDA ctx".to_string(),
                        Err(e) => format!("session failed ({e})"),
                    },
                    Err(e) => format!("no CUDA context ({e})"),
                },
                Err(e) => format!("no CUDA driver ({e})"),
            };
            println!("native session  : {session_state}");
        }
        Err(e) => println!("native nvenc    : unavailable ({e})"),
    }
    match sovmr_nvdec::NvDec::load() {
        Ok(dec) => println!(
            "native nvdec    : ready — {} entry points resolved",
            dec.entry_point_count()
        ),
        Err(e) => println!("native nvdec    : unavailable ({e})"),
    }
    println!("gpu             : {}", plan.gpu.name);
    match plan.gpu.vram_mib {
        Some(mib) => println!("vram            : {mib} MiB"),
        None => println!("vram            : unknown"),
    }
    println!("driver          : {}", plan.gpu.driver_version);
    println!("chosen codec    : {:?}", plan.settings.codec);
    println!(
        "strategy        : {} (cq {})",
        plan.settings.strategy.label(),
        plan.settings.strategy.cq()
    );
    println!("worker slots    : {}", plan.worker_slots);
}
