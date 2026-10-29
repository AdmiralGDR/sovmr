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

//! The egui window: a thin projection of [`DashboardState`].
//!
//! Layout: a hardware-status header with a Dashboard/Advanced tab switch,
//! a large highlighted drop zone (full-bleed while the queue is empty, a
//! slim strip once jobs exist), and the queue as rounded metric cards.
//! Everything here is presentation: the shell never blocks, never touches
//! the filesystem, and never constructs a job — dropped paths go out
//! through [`FileIntake`], settings changes go out through
//! [`SettingsSink`], and results come back as ordinary engine events.
//!
//! Repaint policy (battery-friendly): while jobs are active the shell asks
//! for a repaint every [`UI_REPAINT_INTERVAL`]; when idle it schedules
//! nothing and sleeps until user input — or until the engine wakes it via
//! the [`RepaintWaker`] handed to the embedder by [`run`].

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use eframe::egui::{self, Align, CornerRadius, FontId, Layout, RichText, Sense, Stroke};
use sovmr_core::heuristics::Strategy;
use sovmr_core::job::{Codec, Effort, JobId};
use sovmr_core::queue::messages::EngineCommand;

use crate::bridge::UiBridge;
use crate::i18n::Lang;
use crate::state::{DashboardState, RowStatus, UI_REPAINT_INTERVAL};
use crate::theme;

/// Receives paths the user dropped onto the window. The embedder decides
/// what a path means (validate, probe, enqueue — or reject); the shell
/// only reports the drop. Must not block: heavy work belongs on the
/// embedder's own thread.
pub type FileIntake = Box<dyn Fn(Vec<PathBuf>) + Send>;

/// Receives the power-user settings whenever they change in the Advanced
/// tab. Applies to *future* jobs; the embedder stores them for its intake.
pub type SettingsSink = Box<dyn Fn(UiEncoderSettings) + Send>;

/// Receives the interface language whenever the user switches it, so the
/// embedder can persist the choice for next launch. Like [`SettingsSink`],
/// presentation reports *what* changed; it never does the I/O itself
/// (the disk is the outermost layer's business, not the shell's).
pub type LanguageSink = Box<dyn Fn(Lang) + Send>;

/// Wakes the UI thread from anywhere (wraps `egui::Context::request_repaint`).
/// [`run`] hands one to the embedder so engine events repaint the window
/// immediately instead of waiting for user input.
pub type RepaintWaker = Box<dyn Fn() + Send + Sync>;

/// Supplies the current video-engine utilization (0–100 %), or `None`
/// while unknown. Called once per frame, so it must be a cheap read (an
/// atomic behind a closure) — the actual polling belongs on the
/// embedder's background thread. The shell neither knows nor cares that
/// the number comes from `nvidia-smi`.
pub type GpuMeter = Box<dyn Fn() -> Option<u8> + Send>;

/// Backend-owned formatter turning the current settings into the raw
/// command/parameters that will actually run — the power-user preview in
/// the Advanced tab. Keeps encoder jargon out of this crate.
pub type RawPreview = Box<dyn Fn(&UiEncoderSettings) -> String + Send>;

/// Zero-jargon codec selector: `Auto` defers to the hardware plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecChoice {
    /// Let the hardware configurator pick (AV1 on Ada+, else HEVC).
    Auto,
    /// Force AV1.
    Av1,
    /// Force HEVC.
    Hevc,
}

/// Everything the Advanced tab edits, in backend-neutral terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiEncoderSettings {
    /// Which codec future jobs should use.
    pub codec: CodecChoice,
    /// Semantic quality intent; the heuristic engine owns the math.
    pub strategy: Strategy,
    /// Encoder speed/quality trade-off.
    pub effort: Effort,
    /// Generation-loss protection: skip files that are already compact
    /// in the target codec. Off = force re-encode of everything.
    pub smart_skip: bool,
}

impl Default for UiEncoderSettings {
    fn default() -> Self {
        UiEncoderSettings {
            codec: CodecChoice::Auto,
            strategy: Strategy::VisuallyLossless,
            effort: Effort::Quality,
            smart_skip: true,
        }
    }
}

fn effort_label(lang: Lang, e: Effort) -> &'static str {
    let s = lang.strings();
    match e {
        Effort::Speed => s.effort_fast,
        Effort::Balanced => s.effort_balanced,
        Effort::Quality => s.effort_quality,
        Effort::Maximum => s.effort_maximum,
    }
}

/// Localised label for a codec choice — replaces `CodecChoice::label`,
/// which only ever spoke English.
fn codec_label(lang: Lang, choice: CodecChoice) -> &'static str {
    let s = lang.strings();
    match choice {
        CodecChoice::Auto => s.codec_auto,
        CodecChoice::Av1 => s.codec_av1,
        CodecChoice::Hevc => s.codec_hevc,
    }
}

/// Localised label for a compression strategy. The strategy type lives in
/// `sovmr-core` and carries an English `label()`; the interface names it
/// here instead so the wording can be translated without the engine
/// crate growing a notion of language.
fn strategy_label(lang: Lang, strategy: Strategy) -> &'static str {
    let s = lang.strings();
    match strategy {
        Strategy::VisuallyLossless => s.strategy_vl,
        Strategy::Balanced => s.strategy_balanced,
        Strategy::Aggressive => s.strategy_aggressive,
    }
}

/// Fraction at or above which the frame loop is effectively finished and
/// the muxer is writing its index (`moov` for MP4, the cue/tag front
/// matter for Matroska). On a ten-hour file that write takes real
/// seconds with no engine event behind it, so the row is labelled
/// "Finalizing…" rather than showing a frozen 100 %. Paired with a live
/// spinner, it is the difference between "the app is working" and "the
/// app has hung" — the reading testers reported at both ends of a job.
const FINALIZING_FRACTION: f32 = 0.99;

/// Human progress text for a running row.
///
/// * `None` fraction — the source duration is not known yet (pre-roll,
///   right after Start): "Preparing…".
/// * fraction ≥ [`FINALIZING_FRACTION`] — the encode is done and the
///   container is being finalized: "Finalizing…".
/// * otherwise — the percentage, with the ETA when one is known.
///
/// Kept pure and free of egui so the wording is unit-testable headless.
fn running_progress_text(lang: Lang, fraction: Option<f32>, eta: Option<Duration>) -> String {
    let s = lang.strings();
    match fraction {
        None => s.progress_preparing.to_string(),
        Some(f) if f >= FINALIZING_FRACTION => s.progress_finalizing.to_string(),
        Some(f) => match eta {
            Some(eta) => format!(
                "{:.0}% · {} {}",
                f * 100.0,
                crate::fmt::eta(lang, eta),
                s.progress_left
            ),
            None => format!("{:.0}%", f * 100.0),
        },
    }
}

/// Static facts the shell displays but never interprets.
pub struct ShellConfig {
    /// GPU marketing name, e.g. `"NVIDIA GeForce RTX 4070 Laptop GPU"`.
    pub gpu_name: String,
    /// Total VRAM in MiB, when known.
    pub vram_mib: Option<u64>,
    /// Worker slots currently configured (the Advanced slider edits this).
    pub worker_slots: usize,
    /// Driver-imposed ceiling on concurrent sessions.
    pub session_cap: usize,
    /// Engine badge, e.g. `"ffmpeg/nvenc"`.
    pub engine_id: String,
    /// What the plan chose, for the one-click badge, e.g. `"AV1"`.
    pub auto_codec: Codec,
    /// Whether the window backend can deliver OS drag-and-drop at all
    /// (upstream winit has no Wayland DnD; X11/XWayland works). Decides
    /// the drop zone's copy — the zone must never advertise a gesture
    /// the backend will silently swallow.
    pub dnd_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Dashboard,
    Advanced,
}

/// The desktop window. Construct with [`SovmrShell::new`], optionally add
/// [`SovmrShell::with_raw_preview`], then hand to [`run`].
///
/// # Example
///
/// ```ignore
/// let shell = SovmrShell::new(bridge, config, intake, settings_sink)
///     .with_raw_preview(|s| my_backend::command_preview(s));
/// sovmr_ui::shell::run("sovmr", shell, move |waker| { waker_tx.send(waker).ok(); })?;
/// ```
pub struct SovmrShell {
    bridge: UiBridge,
    dashboard: DashboardState,
    config: ShellConfig,
    intake: FileIntake,
    settings_sink: SettingsSink,
    /// Backend-supplied "what would actually run" line for power users.
    raw_preview: Option<RawPreview>,
    /// Optional live video-engine utilization source for the header meter.
    gpu_meter: Option<GpuMeter>,
    /// External close request (e.g. SIGINT/SIGTERM watcher): when it
    /// flips to `true`, the next frame closes the window so the
    /// embedder's normal teardown path runs.
    close_request: Option<Arc<AtomicBool>>,
    /// Which build this is, short enough for a header.
    build: String,
    /// The same with the roadmap position, for the tooltip.
    build_detail: String,
    /// Native file-picker plumbing: the background dialog thread sends
    /// chosen paths here; `logic` drains them into the intake on the UI
    /// thread. `picker_busy` disables the button while a dialog is up.
    picker_tx: Sender<Vec<PathBuf>>,
    picker_rx: Receiver<Vec<PathBuf>>,
    picker_busy: Arc<AtomicBool>,
    settings: UiEncoderSettings,
    tab: Tab,
    /// Interface language. English by default (the source language), a
    /// header toggle switches it; a pure presentation choice the engine
    /// never sees.
    lang: Lang,
    /// Optional sink notified when the language changes, so the embedder
    /// can persist it. `None` = changes are not persisted.
    language_sink: Option<LanguageSink>,
}

impl SovmrShell {
    /// Assemble a shell from its collaborators: the engine bridge,
    /// display-only configuration, the dropped-file callback, and the
    /// settings callback.
    pub fn new(
        bridge: UiBridge,
        config: ShellConfig,
        intake: FileIntake,
        settings_sink: SettingsSink,
    ) -> Self {
        let (picker_tx, picker_rx) = channel();
        SovmrShell {
            bridge,
            dashboard: DashboardState::default(),
            config,
            intake,
            settings_sink,
            raw_preview: None,
            gpu_meter: None,
            close_request: None,
            build: format!(
                "{} {}",
                sovmr_core::identity::RELEASE,
                sovmr_core::identity::COMMIT
            ),
            build_detail: sovmr_core::identity::describe_full(),
            picker_tx,
            picker_rx,
            picker_busy: Arc::new(AtomicBool::new(false)),
            settings: UiEncoderSettings::default(),
            tab: Tab::Dashboard,
            lang: Lang::default(),
            language_sink: None,
        }
    }

    /// Start in a specific interface language — typically a choice
    /// persisted from a previous run. Without this the shell opens in
    /// [`Lang::default`] (English).
    pub fn with_language(mut self, lang: Lang) -> Self {
        self.lang = lang;
        self
    }

    /// Attach a sink notified whenever the user switches language, so the
    /// embedder can persist it. The shell itself never touches the disk.
    pub fn with_language_sink(mut self, f: impl Fn(Lang) + Send + 'static) -> Self {
        self.language_sink = Some(Box::new(f));
        self
    }

    /// Attach an external close request: while the flag reads `true`,
    /// the shell asks its viewport to close on the next frame. Pair it
    /// with the [`RepaintWaker`] so a signal wakes an idle window.
    pub fn with_close_request(mut self, flag: Arc<AtomicBool>) -> Self {
        self.close_request = Some(flag);
        self
    }

    /// Attach a live utilization source; the header then shows a compact
    /// "NVENC n%" meter fed by it.
    pub fn with_gpu_meter(mut self, meter: GpuMeter) -> Self {
        self.gpu_meter = Some(meter);
        self
    }

    /// Attach a backend-owned formatter that renders the current settings
    /// as the raw command/parameters it will execute — shown (only) in the
    /// Advanced tab, keeping the main screen jargon-free.
    pub fn with_raw_preview(
        mut self,
        f: impl Fn(&UiEncoderSettings) -> String + Send + 'static,
    ) -> Self {
        self.raw_preview = Some(Box::new(f));
        self
    }

    // ---- header ----------------------------------------------------------

    fn header(&mut self, ui: &mut egui::Ui) {
        // `strings()` returns a `&'static` table, so this borrows nothing
        // from `self` and lives freely beside the `&mut self` closure.
        let s = self.lang.strings();
        ui.add_space(theme::CARD_GAP);
        ui.horizontal(|ui| {
            ui.heading(RichText::new("sovmr").color(theme::ACCENT_CYAN).strong());
            ui.add_space(theme::CARD_PADDING);
            ui.selectable_value(&mut self.tab, Tab::Dashboard, s.tab_dashboard);
            ui.selectable_value(&mut self.tab, Tab::Advanced, s.tab_advanced);
            ui.add_space(theme::CARD_PADDING);
            let paused = self.dashboard.paused;
            let toggle = if paused {
                egui::Button::new(RichText::new(s.resume).color(theme::BG_WINDOW))
                    .fill(theme::ACCENT_ORANGE)
            } else {
                egui::Button::new(s.pause)
            };
            if ui.add(toggle).on_hover_text(s.pause_hover).clicked() {
                self.bridge.dispatch(if paused {
                    EngineCommand::ResumeQueue
                } else {
                    EngineCommand::PauseQueue
                });
            }
            if paused {
                ui.label(
                    RichText::new(s.paused_badge)
                        .color(theme::ACCENT_ORANGE)
                        .strong(),
                );
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                // One toggle, naming the language it leads to. A pure
                // presentation flip — the engine is never told.
                if ui
                    .button(self.lang.switch_badge())
                    .on_hover_text(s.language_hover)
                    .clicked()
                {
                    self.lang = self.lang.other();
                    if let Some(sink) = &self.language_sink {
                        sink(self.lang);
                    }
                }
                self.nvenc_meter(ui);
                ui.label(RichText::new(self.hardware_line()).color(theme::FG_MUTED));
                // Which build this is, and what it is for. Two machines
                // run this program and a fix is not delivered until both
                // have it — "which version is where" should not need a
                // terminal to answer, and until now it did.
                ui.add(
                    egui::Label::new(
                        RichText::new(self.build.as_str())
                            .color(theme::FG_MUTED)
                            .small(),
                    )
                    .sense(Sense::hover()),
                )
                .on_hover_text(self.build_detail.as_str());
            });
        });
        ui.add_space(theme::CARD_GAP);
    }

    /// Compact custom-painted utilization meter: `GPU ▮▮▮ 63%`.
    /// Hidden entirely while the source reports unknown (no NVIDIA stack,
    /// first poll pending) — an empty gauge would read as "0% = idle",
    /// which is a lie.
    fn nvenc_meter(&self, ui: &mut egui::Ui) {
        let Some(pct) = self.gpu_meter.as_ref().and_then(|m| m()) else {
            return;
        };
        let pct = pct.min(100);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(Self::METER_WIDTH, Self::METER_HEIGHT),
            Sense::hover(),
        );
        let painter = ui.painter();
        let radius = CornerRadius::same(theme::RADIUS / 2);
        painter.rect_filled(rect, radius, theme::BG_CARD);
        let mut fill = rect;
        fill.set_width(rect.width() * f32::from(pct) / 100.0);
        // Saturated encoder is the goal — flip to orange
        // only when the chip is nearly maxed so users see the ceiling.
        let color = if pct >= Self::METER_HOT_PCT {
            theme::ACCENT_ORANGE
        } else {
            theme::ACCENT_CYAN
        };
        painter.rect_filled(fill, radius, color);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("GPU {pct}%"),
            FontId::proportional(Self::METER_FONT_SIZE),
            theme::FG_TEXT,
        );
        response.on_hover_text(self.lang.strings().gpu_meter_hover);
    }

    /// Header meter geometry.
    const METER_WIDTH: f32 = 92.0;
    const METER_HEIGHT: f32 = 16.0;
    const METER_FONT_SIZE: f32 = 10.0;
    /// At or above this percentage the meter turns orange ("saturated").
    const METER_HOT_PCT: u8 = 90;

    /// "RTX 4070 Laptop · 8 GB · AV1" — the facts a user recognizes;
    /// worker-slot plumbing lives in the Advanced tab where it is edited.
    fn hardware_line(&self) -> String {
        let gpu = self
            .config
            .gpu_name
            .trim_start_matches("NVIDIA ")
            .trim_start_matches("GeForce ");
        let vram = match self.config.vram_mib {
            Some(mib) => crate::fmt::bytes(mib << 20),
            None => "?".to_string(),
        };
        let codec = match self.settings.codec {
            CodecChoice::Auto => match self.config.auto_codec {
                Codec::Av1 => "AV1",
                Codec::Hevc => "HEVC",
            },
            CodecChoice::Av1 => "AV1",
            CodecChoice::Hevc => "HEVC",
        };
        format!("{gpu} · {vram} · {codec}")
    }

    // ---- dashboard -------------------------------------------------------

    fn dashboard_tab(&mut self, ui: &mut egui::Ui) {
        let hovering_files = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
        if self.dashboard.rows.is_empty() {
            self.drop_zone(ui, hovering_files, true);
            return;
        }
        self.batch_summary(ui);
        self.savings_bar(ui);
        ui.add_space(theme::CARD_GAP);
        let list_height = ui.available_height() - Self::DROP_STRIP_HEIGHT;
        egui::ScrollArea::vertical()
            .max_height(list_height)
            .auto_shrink(false)
            .show(ui, |ui| self.queue_cards(ui));
        self.drop_zone(ui, hovering_files, false);
    }

    /// Height reserved for the compact drop strip under a populated queue.
    const DROP_STRIP_HEIGHT: f32 = 56.0;

    /// The drop target. `hero` = full remaining space (empty queue);
    /// otherwise a slim strip. Highlights warm orange while the OS reports
    /// files hovering over the window.
    fn drop_zone(&mut self, ui: &mut egui::Ui, hovering: bool, hero: bool) {
        let s = self.lang.strings();
        let (fill, stroke, text_color) = if hovering {
            (
                theme::ACCENT_ORANGE.gamma_multiply(0.15),
                Stroke::new(2.0 * theme::HAIRLINE, theme::ACCENT_ORANGE),
                theme::ACCENT_ORANGE,
            )
        } else {
            (
                theme::BG_FAINT,
                Stroke::new(theme::HAIRLINE, theme::FG_MUTED.gamma_multiply(0.5)),
                theme::FG_MUTED,
            )
        };
        let height = if hero {
            ui.available_height() - theme::CARD_GAP
        } else {
            Self::DROP_STRIP_HEIGHT - theme::CARD_GAP
        };
        let dnd = self.config.dnd_available;
        egui::Frame::new()
            .fill(fill)
            .stroke(stroke)
            .inner_margin(theme::CARD_PADDING / 2.0)
            .corner_radius(CornerRadius::same(theme::RADIUS))
            .show(ui, |ui| {
                ui.set_min_size(egui::vec2(ui.available_width(), height));
                if hero {
                    ui.vertical_centered(|ui| {
                        ui.add_space((height * 0.5 - 70.0).max(theme::CARD_GAP));
                        let (icon, headline) = if dnd {
                            ("⬇", s.drop_headline_dnd)
                        } else {
                            ("🎬", s.drop_headline_nodnd)
                        };
                        ui.label(RichText::new(icon).size(42.0).color(text_color));
                        ui.label(RichText::new(headline).size(20.0).color(text_color));
                        ui.add_space(theme::CARD_GAP);
                        self.add_media_button(ui, 20.0, egui::vec2(240.0, 52.0));
                        ui.add_space(theme::CARD_GAP);
                        ui.label(RichText::new(s.tagline).color(theme::FG_MUTED));
                        if !dnd {
                            ui.label(RichText::new(s.dnd_unavailable).color(theme::FG_MUTED));
                        }
                    });
                } else {
                    ui.add_space(theme::CARD_GAP / 2.0);
                    ui.horizontal(|ui| {
                        ui.add_space(theme::CARD_PADDING);
                        self.add_media_button(ui, 14.0, egui::vec2(140.0, 30.0));
                        let strip_text = if dnd {
                            s.strip_more_dnd
                        } else {
                            s.strip_more_nodnd
                        };
                        ui.label(RichText::new(strip_text).color(text_color));
                    });
                }
            });
    }

    /// The guaranteed intake path: opens the native file dialog
    /// (zenity/kdialog) on a background thread. Disabled while a dialog
    /// is already up; re-enabled by that thread when it finishes.
    fn add_media_button(&mut self, ui: &mut egui::Ui, font_size: f32, min_size: egui::Vec2) {
        let busy = self.picker_busy.load(Ordering::SeqCst);
        let s = self.lang.strings();
        let label = if busy { s.add_media_busy } else { s.add_media };
        let button = egui::Button::new(
            RichText::new(label)
                .size(font_size)
                .strong()
                .color(theme::BG_WINDOW),
        )
        .fill(theme::ACCENT_CYAN)
        .min_size(min_size)
        .corner_radius(CornerRadius::same(theme::RADIUS));
        if ui.add_enabled(!busy, button).clicked() {
            self.picker_busy.store(true, Ordering::SeqCst);
            let tx = self.picker_tx.clone();
            let busy = self.picker_busy.clone();
            let ctx = ui.ctx().clone();
            let spawned = thread::Builder::new()
                .name("sovmr-file-picker".into())
                .spawn(move || {
                    let paths = crate::picker::pick_files_blocking();
                    if !paths.is_empty() {
                        let _ = tx.send(paths);
                    }
                    busy.store(false, Ordering::SeqCst);
                    ctx.request_repaint(); // idle window: pick up the result now
                });
            if let Err(e) = spawned {
                self.picker_busy.store(false, Ordering::SeqCst);
                eprintln!("sovmr: cannot spawn the file-picker thread: {e}");
            }
        }
    }

    fn batch_summary(&mut self, ui: &mut egui::Ui) {
        let (done, total) = self.dashboard.terminal_counts();
        let running = self
            .dashboard
            .rows
            .iter()
            .filter(|r| matches!(r.status, RowStatus::Running { .. }))
            .count();
        ui.horizontal(|ui| {
            ui.add(
                egui::ProgressBar::new(self.dashboard.batch_fraction())
                    .fill(theme::ACCENT_CYAN)
                    .desired_width(ui.available_width() * 0.6)
                    .corner_radius(CornerRadius::same(theme::RADIUS)),
            );
            let s = self.lang.strings();
            let summary = if self.dashboard.paused {
                format!(
                    "{done} {} {total} {} · {} ({running} {})",
                    s.word_of, s.word_done, s.word_paused, s.word_finishing
                )
            } else {
                format!(
                    "{done} {} {total} {} · {running} {}",
                    s.word_of, s.word_done, s.word_running
                )
            };
            let summary_color = if self.dashboard.paused {
                theme::ACCENT_ORANGE
            } else {
                theme::FG_MUTED
            };
            ui.label(RichText::new(summary).color(summary_color));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button(s.cancel_all).clicked() {
                    self.bridge.dispatch(EngineCommand::CancelAll);
                }
            });
        });
    }

    /// "Total Space Saved" visual: one custom-painted bar where the full
    /// width is the original bytes of every completed job, the cyan
    /// segment is what they shrank to, and the green remainder is the
    /// disk space given back. Appears after the first completion.
    fn savings_bar(&self, ui: &mut egui::Ui) {
        let (input, output) = self.dashboard.batch_savings();
        if input == 0 {
            return;
        }
        ui.add_space(theme::CARD_GAP);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), Self::SAVINGS_BAR_HEIGHT),
            Sense::hover(),
        );
        let painter = ui.painter();
        let radius = CornerRadius::same(theme::RADIUS);
        // Whole bar = original footprint (green: "this is yours again").
        // Kept bright enough that the dark label stays readable on both
        // segments.
        painter.rect_filled(rect, radius, theme::OK_GREEN.gamma_multiply(0.55));
        // Left segment = what the files occupy now.
        let ratio = (output as f32 / input as f32).clamp(0.0, 1.0);
        let mut occupied = rect;
        occupied.set_width(rect.width() * ratio);
        painter.rect_filled(occupied, radius, theme::ACCENT_CYAN.gamma_multiply(0.9));
        let s = self.lang.strings();
        let saved = input.saturating_sub(output);
        let label = if saved > 0 {
            format!(
                "{} {}   ({} → {})",
                s.space_saved_prefix,
                crate::fmt::bytes(saved),
                crate::fmt::bytes(input),
                crate::fmt::bytes(output)
            )
        } else {
            format!(
                "{} ({} → {})",
                s.no_space_saved,
                crate::fmt::bytes(input),
                crate::fmt::bytes(output)
            )
        };
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            FontId::proportional(Self::SAVINGS_FONT_SIZE),
            theme::BG_WINDOW,
        );
        response.on_hover_text(s.savings_hover);
    }

    /// Savings bar geometry.
    const SAVINGS_BAR_HEIGHT: f32 = 20.0;
    const SAVINGS_FONT_SIZE: f32 = 12.0;
    /// Heuristic-insight line on a queue card: small, secondary, cyan.
    const INSIGHT_FONT_SIZE: f32 = 11.0;
    /// Diameter of the liveness spinner shown beside a running row's bar.
    const SPINNER_SIZE: f32 = 14.0;

    fn queue_cards(&mut self, ui: &mut egui::Ui) {
        // Collect actions first: we can't mutate rows/bridge mid-iteration.
        let mut cancel: Option<JobId> = None;
        let mut remove: Option<JobId> = None;
        let lang = self.lang;
        let s = lang.strings();
        for row in &self.dashboard.rows {
            egui::Frame::new()
                .fill(theme::BG_CARD)
                .corner_radius(CornerRadius::same(theme::RADIUS))
                .inner_margin(theme::CARD_PADDING)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&row.name).strong());
                        if let Some(warning) = &row.warning {
                            ui.add(
                                egui::Label::new(RichText::new("⚠").color(theme::ACCENT_ORANGE))
                                    .sense(Sense::hover()),
                            )
                            .on_hover_text(warning);
                        }
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            match &row.status {
                                RowStatus::Queued | RowStatus::Running { .. } => {
                                    if ui.small_button("Cancel").clicked() {
                                        cancel = Some(row.id);
                                    }
                                }
                                _ => {
                                    if ui.small_button("Remove").clicked() {
                                        remove = Some(row.id);
                                    }
                                }
                            }
                            Self::row_metrics(lang, ui, &row.status);
                        });
                    });
                    // Show, don't tell: the engine's measured verdict on
                    // THIS file (character, density, enforced ceiling).
                    if let Some(insight) = &row.insight {
                        ui.add(
                            egui::Label::new(
                                RichText::new(insight)
                                    .size(Self::INSIGHT_FONT_SIZE)
                                    .color(theme::ACCENT_CYAN),
                            )
                            .sense(Sense::hover()),
                        )
                        .on_hover_text(s.insight_hover);
                    }
                    if let RowStatus::Running { fraction, eta, .. } = &row.status {
                        let text = running_progress_text(lang, *fraction, *eta);
                        // A live spinner alongside the bar. It animates on
                        // its own (requesting repaints), so the row keeps
                        // moving even when the percentage does not — during
                        // pre-roll before the first frame, and during the
                        // container finalize after the last one. Both are
                        // windows the engine is busy but silent, and a
                        // static 100 % read as a hang to testers.
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Spinner::new()
                                    .size(Self::SPINNER_SIZE)
                                    .color(theme::ACCENT_CYAN),
                            );
                            ui.add(
                                egui::ProgressBar::new(fraction.unwrap_or(0.0))
                                    .fill(theme::ACCENT_CYAN)
                                    .text(RichText::new(text).color(theme::FG_TEXT))
                                    .corner_radius(CornerRadius::same(theme::RADIUS)),
                            );
                        });
                    }
                });
            ui.add_space(theme::CARD_GAP);
        }
        if let Some(id) = cancel {
            self.bridge.dispatch(EngineCommand::Cancel(id));
        }
        if let Some(id) = remove {
            self.dashboard.remove_row(id);
        }
    }

    /// Right-aligned, per-state metric text ("142 fps · 4.9×", "-62% · …").
    fn row_metrics(lang: Lang, ui: &mut egui::Ui, status: &RowStatus) {
        let s = lang.strings();
        match status {
            RowStatus::Queued => {
                ui.label(RichText::new(s.status_waiting).color(theme::FG_MUTED));
            }
            RowStatus::Running { speed, .. } => {
                ui.label(
                    RichText::new(format!("{speed:.1}{}", s.realtime_suffix))
                        .color(theme::FG_MUTED),
                );
            }
            RowStatus::Completed {
                input_bytes,
                output_bytes,
            } => {
                ui.label(
                    RichText::new(format!(
                        "{} → {}",
                        crate::fmt::bytes(*input_bytes),
                        crate::fmt::bytes(*output_bytes)
                    ))
                    .color(theme::FG_MUTED),
                );
                ui.label(
                    RichText::new(format!(
                        "✓ {}",
                        crate::fmt::reduction(*input_bytes, *output_bytes)
                    ))
                    .color(theme::OK_GREEN),
                );
            }
            RowStatus::Failed { reason } => {
                // A plain `ui.label` does not sense hover, so `on_hover_text`
                // on it never fires — the reason stayed hidden. A `Label`
                // with an explicit hover sense makes the tooltip reachable.
                ui.add(
                    egui::Label::new(RichText::new(s.status_failed_hint).color(theme::ERR_RED))
                        .sense(Sense::hover()),
                )
                .on_hover_text(reason);
            }
            RowStatus::Cancelled => {
                ui.label(RichText::new(s.status_cancelled).color(theme::FG_MUTED));
            }
            RowStatus::Skipped { reason } => {
                ui.add(
                    egui::Label::new(
                        RichText::new(s.status_already_optimal).color(theme::ACCENT_CYAN),
                    )
                    .sense(Sense::hover()),
                )
                .on_hover_text(reason);
            }
        }
    }

    // ---- advanced --------------------------------------------------------

    fn advanced_tab(&mut self, ui: &mut egui::Ui) {
        let before = (self.settings, self.config.worker_slots);
        let lang = self.lang;
        let s = lang.strings();

        ui.label(RichText::new(s.adv_hardware).strong());
        ui.add_space(theme::CARD_GAP);
        ui.add(
            egui::Slider::new(&mut self.config.worker_slots, 1..=self.config.session_cap)
                .text(s.adv_parallel),
        );
        ui.label(
            RichText::new(format!(
                "{} · {} {} {}",
                self.config.gpu_name, s.adv_driver_cap, self.config.session_cap, s.adv_sessions
            ))
            .color(theme::FG_MUTED),
        );

        ui.add_space(theme::CARD_PADDING);
        ui.separator();
        ui.add_space(theme::CARD_PADDING);

        ui.label(RichText::new(s.adv_encoder_section).strong());
        ui.add_space(theme::CARD_GAP);
        egui::ComboBox::from_label(s.adv_codec)
            .selected_text(codec_label(lang, self.settings.codec))
            .show_ui(ui, |ui| {
                for choice in [CodecChoice::Auto, CodecChoice::Av1, CodecChoice::Hevc] {
                    ui.selectable_value(
                        &mut self.settings.codec,
                        choice,
                        codec_label(lang, choice),
                    );
                }
            });
        egui::ComboBox::from_label(s.adv_strategy)
            .selected_text(strategy_label(lang, self.settings.strategy))
            .show_ui(ui, |ui| {
                for strategy in [
                    Strategy::VisuallyLossless,
                    Strategy::Balanced,
                    Strategy::Aggressive,
                ] {
                    ui.selectable_value(
                        &mut self.settings.strategy,
                        strategy,
                        strategy_label(lang, strategy),
                    );
                }
            });
        ui.label(
            RichText::new(match self.settings.strategy {
                Strategy::VisuallyLossless => s.strategy_note_vl,
                Strategy::Balanced => s.strategy_note_balanced,
                Strategy::Aggressive => s.strategy_note_aggressive,
            })
            .color(theme::FG_MUTED),
        );
        egui::ComboBox::from_label(s.adv_preset)
            .selected_text(effort_label(lang, self.settings.effort))
            .show_ui(ui, |ui| {
                for effort in [
                    Effort::Speed,
                    Effort::Balanced,
                    Effort::Quality,
                    Effort::Maximum,
                ] {
                    ui.selectable_value(
                        &mut self.settings.effort,
                        effort,
                        effort_label(lang, effort),
                    );
                }
            });
        ui.add_space(theme::CARD_GAP);
        ui.checkbox(&mut self.settings.smart_skip, s.smart_skip)
            .on_hover_text(s.smart_skip_hover);
        if !self.settings.smart_skip {
            ui.label(RichText::new(s.smart_skip_off).color(theme::ACCENT_ORANGE));
        }

        if let Some(preview) = &self.raw_preview {
            ui.add_space(theme::CARD_PADDING);
            ui.label(RichText::new(s.raw_params).color(theme::FG_MUTED));
            ui.label(
                RichText::new(preview(&self.settings))
                    .monospace()
                    .color(theme::ACCENT_CYAN),
            );
        }
        ui.add_space(theme::CARD_PADDING);
        ui.label(
            RichText::new(format!("{} {}", s.engine_prefix, self.config.engine_id))
                .color(theme::FG_MUTED),
        );

        // Publish changes exactly once per changed frame, not per widget.
        if before.0 != self.settings {
            (self.settings_sink)(self.settings);
        }
        if before.1 != self.config.worker_slots {
            self.bridge
                .dispatch(EngineCommand::SetWorkerSlots(self.config.worker_slots));
        }
    }
}

impl eframe::App for SovmrShell {
    /// Non-painting phase, called before every [`Self::ui`]: fold engine
    /// events, forward dropped files, keep the repaint clock ticking only
    /// while something is actually moving.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // External teardown (termination signal): close the window so
        // `run` returns and the embedder's single shutdown path runs.
        if self
            .close_request
            .as_ref()
            .is_some_and(|f| f.load(Ordering::SeqCst))
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.bridge.pump(&mut self.dashboard);

        // OS drops, hardened: every entry either resolves to an
        // existing absolute path (file:// URIs included) or is rejected
        // LOUDLY on stderr — intake must never fail silently.
        let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped.is_empty() {
            let mut accepted = Vec::new();
            for file in &dropped {
                match resolve_dropped(file) {
                    Ok(path) => accepted.push(path),
                    Err(why) => eprintln!(
                        "sovmr: DROP REJECTED (path={:?}, name={:?}, mime={:?}): {why}",
                        file.path, file.name, file.mime
                    ),
                }
            }
            eprintln!(
                "sovmr: drop received: {} entries, {} accepted",
                dropped.len(),
                accepted.len()
            );
            if !accepted.is_empty() {
                (self.intake)(accepted);
            }
        }

        // Paths chosen in the native picker dialog (background thread).
        while let Ok(paths) = self.picker_rx.try_recv() {
            eprintln!("sovmr: picker returned {} file(s)", paths.len());
            (self.intake)(paths);
        }

        // Battery policy: schedule repaints only while an encode is on a
        // worker slot; an idle (or fully paused) window sleeps until user
        // input or an external RepaintWaker call.
        if self.dashboard.has_running() {
            ctx.request_repaint_after(UI_REPAINT_INTERVAL);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if let Some(reason) = self.dashboard.hardware_error.clone() {
            let lang = self.lang;
            egui::CentralPanel::default().show(ui, |ui| hardware_error_screen(lang, ui, &reason));
            return;
        }
        egui::Panel::top("sovmr-header").show(ui, |ui| self.header(ui));
        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Dashboard => self.dashboard_tab(ui),
            Tab::Advanced => self.advanced_tab(ui),
        });
    }

    /// Window canvas behind the panels — pin it to the theme's midnight.
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::BG_WINDOW.to_normalized_gamma_f32()
    }
}

/// Full-window pre-flight failure screen: what is missing, what the
/// machine needs, and where to look for details. Replaces the entire
/// dashboard — there is nothing to queue on a machine without the
/// silicon this app drives.
fn hardware_error_screen(lang: Lang, ui: &mut egui::Ui, reason: &str) {
    let s = lang.strings();
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.25);
        ui.label(RichText::new("⚠").size(52.0).color(theme::ACCENT_ORANGE));
        ui.add_space(theme::CARD_GAP);
        ui.label(
            RichText::new(s.hw_no_gpu)
                .size(24.0)
                .strong()
                .color(theme::FG_TEXT),
        );
        ui.add_space(theme::CARD_GAP);
        ui.label(RichText::new(s.hw_body).color(theme::FG_TEXT));
        ui.add_space(theme::CARD_PADDING);
        ui.label(
            RichText::new(format!("{} {reason}", s.hw_probe_prefix))
                .monospace()
                .color(theme::ERR_RED),
        );
        ui.add_space(theme::CARD_PADDING);
        ui.label(RichText::new(s.hw_report_hint).color(theme::FG_MUTED));
    });
}

/// Resolve one backend drop entry to an existing absolute path.
///
/// Order of trust: a non-URI `path` is used as-is (no lossy string
/// round-trip — paths are bytes); a `path` or `name` that *looks* like a
/// `file://` URI goes through the percent-decoding resolver; a bare
/// `name` is tried as a path last (some backends only fill `name`).
///
/// # Errors
///
/// [`crate::paths::PathRejection`] naming exactly what was wrong.
fn resolve_dropped(file: &egui::DroppedFile) -> Result<PathBuf, crate::paths::PathRejection> {
    use crate::paths::{resolve_existing, PathRejection};
    if let Some(path) = &file.path {
        let text = path.to_string_lossy();
        if text.starts_with("file://") {
            return resolve_existing(&text);
        }
        if !path.as_os_str().is_empty() {
            if !path.is_absolute() {
                return Err(PathRejection::NotAbsolute(text.into_owned()));
            }
            if !path.exists() {
                return Err(PathRejection::NotFound(text.into_owned()));
            }
            return Ok(path.clone());
        }
    }
    if !file.name.is_empty() {
        return resolve_existing(&file.name);
    }
    Err(PathRejection::Empty)
}

/// Open the native window, install the sovmr theme, and run the shell
/// until the user closes it. `on_start` receives a [`RepaintWaker`] just
/// before the event loop begins — hand it to whatever thread forwards
/// engine events so each event repaints the window immediately.
///
/// # Errors
///
/// Propagates `eframe::Error` when no usable display/GPU surface exists
/// (e.g. headless SSH session) — embedders should degrade gracefully
/// instead of panicking.
pub fn run(
    title: &str,
    shell: SovmrShell,
    on_start: impl FnOnce(RepaintWaker) + 'static,
) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(title)
            .with_inner_size(WINDOW_DEFAULT_SIZE)
            .with_min_inner_size(WINDOW_MIN_SIZE)
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        title,
        options,
        Box::new(move |cc| {
            theme::apply(&cc.egui_ctx);
            let ctx = cc.egui_ctx.clone();
            on_start(Box::new(move || ctx.request_repaint()));
            Ok(Box::new(shell))
        }),
    )
}

/// Comfortable default window size for a queue of ~8 visible jobs.
const WINDOW_DEFAULT_SIZE: [f32; 2] = [760.0, 560.0];
/// Below this the drop zone and cards stop being usable.
const WINDOW_MIN_SIZE: [f32; 2] = [520.0, 380.0];

#[cfg(test)]
mod identity_tests {
    /// The header must name the build, not merely exist.
    ///
    /// "Which version is on which machine" was answerable only from a
    /// terminal, and the person who has to answer it is the person
    /// looking at the window.
    #[test]
    fn the_header_shows_something_that_identifies_the_build() {
        let short = format!(
            "{} {}",
            sovmr_core::identity::RELEASE,
            sovmr_core::identity::COMMIT
        );
        assert!(short.contains(sovmr_core::identity::RELEASE));
        assert!(short.contains(sovmr_core::identity::COMMIT));
        // And the detail behind it says what the build is *for*, which a
        // release number cannot.
        let detail = sovmr_core::identity::describe_full();
        assert!(detail.contains(sovmr_core::identity::MILESTONE));
        assert!(detail.contains(sovmr_core::identity::TARGET));
    }
}

#[cfg(test)]
mod progress_text_tests {
    use super::{running_progress_text, FINALIZING_FRACTION};
    use crate::i18n::Lang;
    use std::time::Duration;

    #[test]
    fn no_fraction_is_preparing_not_a_frozen_zero() {
        // Right after Start, before the source duration is known. What
        // matters is it is never a silent 0 % that reads as stuck.
        assert_eq!(running_progress_text(Lang::En, None, None), "Preparing…");
        assert_eq!(
            running_progress_text(Lang::En, None, Some(Duration::from_secs(5))),
            "Preparing…"
        );
        // And it is translated, not left in English under a Russian UI.
        assert_eq!(running_progress_text(Lang::Ru, None, None), "Подготовка…");
    }

    #[test]
    fn a_full_fraction_reads_as_finalizing_not_a_hung_hundred_percent() {
        // The bug testers hit: the frame loop is done, the muxer is
        // writing its index, and the number sits at 100 % with no motion.
        assert_eq!(
            running_progress_text(Lang::En, Some(1.0), None),
            "Finalizing…"
        );
        assert_eq!(
            running_progress_text(
                Lang::En,
                Some(FINALIZING_FRACTION),
                Some(Duration::from_secs(9))
            ),
            "Finalizing…",
            "the ETA is suppressed once finalizing — there is no frame ETA to give"
        );
        assert_eq!(
            running_progress_text(Lang::Ru, Some(1.0), None),
            "Финализация…"
        );
    }

    #[test]
    fn mid_encode_shows_the_percentage_and_eta() {
        assert_eq!(running_progress_text(Lang::En, Some(0.5), None), "50%");
        let with_eta = running_progress_text(Lang::En, Some(0.5), Some(Duration::from_secs(30)));
        assert!(with_eta.starts_with("50% · "), "got {with_eta:?}");
        assert!(with_eta.ends_with("left"), "got {with_eta:?}");
        // Russian keeps the percentage (a number, language-neutral) but
        // translates the trailing word.
        let ru = running_progress_text(Lang::Ru, Some(0.5), Some(Duration::from_secs(30)));
        assert!(ru.starts_with("50% · "), "got {ru:?}");
        assert!(ru.ends_with("осталось"), "got {ru:?}");
    }

    #[test]
    fn the_finalizing_threshold_is_just_below_one() {
        // Just under the threshold is still a percentage; at or above it
        // flips to Finalizing. Guards the boundary against drift.
        assert_eq!(running_progress_text(Lang::En, Some(0.98), None), "98%");
        assert_eq!(
            running_progress_text(Lang::En, Some(0.999), None),
            "Finalizing…"
        );
    }
}
