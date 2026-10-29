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

//! The sovmr dark theme: deep charcoal surfaces with Pop!_OS-inspired
//! cyan/orange accents, tuned to sit naturally next to COSMIC on Pop!_OS.
//!
//! Every colour and metric is a named constant — the palette is the single
//! source of truth for the whole shell (no inline colour literals anywhere
//! else in this crate). Apply once at startup with [`apply`]; widgets pick
//! the rest up from the egui style system.

use eframe::egui::{self, Color32, CornerRadius, Stroke};

/// Window background — the deepest layer ("midnight").
pub const BG_WINDOW: Color32 = Color32::from_rgb(0x14, 0x14, 0x18);
/// Panel background — one step above the window ("charcoal").
pub const BG_PANEL: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x21);
/// Card / widget resting background.
pub const BG_CARD: Color32 = Color32::from_rgb(0x24, 0x24, 0x2b);
/// Widget background while hovered.
pub const BG_HOVER: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x37);
/// Faint fill for striped/secondary surfaces.
pub const BG_FAINT: Color32 = Color32::from_rgb(0x1f, 0x1f, 0x25);

/// Primary body text.
pub const FG_TEXT: Color32 = Color32::from_rgb(0xe6, 0xe6, 0xea);
/// De-emphasised metrics and hints.
pub const FG_MUTED: Color32 = Color32::from_rgb(0x9a, 0x9a, 0xa5);

/// Pop!_OS teal-cyan — progress bars, selection, the "active" colour.
pub const ACCENT_CYAN: Color32 = Color32::from_rgb(0x48, 0xb9, 0xc7);
/// Pop!_OS warm orange — the drop-zone highlight and attention colour.
pub const ACCENT_ORANGE: Color32 = Color32::from_rgb(0xfa, 0xa4, 0x1a);
/// Success (completed jobs, savings badge).
pub const OK_GREEN: Color32 = Color32::from_rgb(0x73, 0xc4, 0x8f);
/// Failure text.
pub const ERR_RED: Color32 = Color32::from_rgb(0xf0, 0x6a, 0x6a);

/// Uniform corner radius: soft, modern, but not bubbly.
pub const RADIUS: u8 = 8;
/// Hairline stroke width used for card and drop-zone borders.
pub const HAIRLINE: f32 = 1.0;
/// Padding inside queue cards, in points — roomy enough that rows read
/// as calm cards, not table lines.
pub const CARD_PADDING: f32 = 14.0;
/// Vertical gap between queue cards, in points.
pub const CARD_GAP: f32 = 10.0;

/// Install the sovmr dark theme on this egui context. Call once, right
/// after window creation (the shell's `run` does this for you). Forces
/// dark mode regardless of the system preference — the palette is part of
/// the product identity — and styles both egui theme slots so a stray
/// theme switch can never produce unthemed widgets.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.all_styles_mut(|style| style_visuals(&mut style.visuals));
}

fn style_visuals(v: &mut egui::style::Visuals) {
    v.dark_mode = true;
    v.override_text_color = Some(FG_TEXT);
    v.panel_fill = BG_PANEL;
    v.window_fill = BG_PANEL;
    v.extreme_bg_color = BG_WINDOW;
    v.faint_bg_color = BG_FAINT;
    v.hyperlink_color = ACCENT_CYAN;
    v.selection.bg_fill = ACCENT_CYAN.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(HAIRLINE, ACCENT_CYAN);
    v.warn_fg_color = ACCENT_ORANGE;
    v.error_fg_color = ERR_RED;

    v.widgets.noninteractive.bg_fill = BG_PANEL;
    v.widgets.noninteractive.fg_stroke = Stroke::new(HAIRLINE, FG_MUTED);
    v.widgets.inactive.bg_fill = BG_CARD;
    v.widgets.inactive.weak_bg_fill = BG_CARD;
    v.widgets.hovered.bg_fill = BG_HOVER;
    v.widgets.hovered.weak_bg_fill = BG_HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(HAIRLINE, ACCENT_CYAN);
    v.widgets.active.bg_fill = BG_HOVER;
    v.widgets.active.weak_bg_fill = BG_HOVER;
    v.widgets.active.bg_stroke = Stroke::new(HAIRLINE, ACCENT_CYAN);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(RADIUS);
    }
}
