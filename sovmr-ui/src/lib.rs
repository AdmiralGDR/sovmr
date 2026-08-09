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

//! # sovmr-ui — reusable batch-dashboard shell
//!
//! An egui frontend for any engine that speaks the `sovmr-core` queue
//! protocol. This crate deliberately knows **nothing** about FFmpeg, NVENC,
//! or file probing — it consumes [`EngineEvent`]s from a generic channel,
//! renders them, and emits [`EngineCommand`]s / dropped-file paths through
//! callbacks. Rip it out and point it at a completely different engine (or
//! reuse the engine headless without this crate): both directions work by
//! construction.
//!
//! Three layers, outermost optional:
//!
//! | Module     | Depends on egui? | Purpose                                      |
//! |------------|------------------|----------------------------------------------|
//! | [`state`]  | no               | Pure `EngineEvent` -> view-state reducer     |
//! | [`fmt`]    | no               | Human-friendly metric formatting             |
//! | [`bridge`] | no               | Channel plumbing between engine and renderer |
//! | [`theme`]  | yes              | Dark palette + egui style (named constants)  |
//! | [`shell`]  | yes              | The actual egui window                       |
//!
//! # Example (headless embedding — no window needed)
//!
//! ```
//! use std::sync::mpsc::channel;
//! use sovmr_core::queue::messages::{EngineCommand, EngineEvent};
//! use sovmr_ui::{bridge::UiBridge, state::DashboardState};
//!
//! let (event_tx, event_rx) = channel();
//! let (cmd_tx, _cmd_rx) = channel();
//! let bridge = UiBridge::new(event_rx, move |cmd| { let _ = cmd_tx.send(cmd); });
//!
//! event_tx.send(EngineEvent::EngineStopped).unwrap();
//! let mut dashboard = DashboardState::default();
//! bridge.pump(&mut dashboard);           // fold pending events, never blocks
//! assert!(dashboard.engine_stopped);
//! bridge.dispatch(EngineCommand::Shutdown);
//! ```
//!
//! [`EngineEvent`]: sovmr_core::queue::messages::EngineEvent
//! [`EngineCommand`]: sovmr_core::queue::messages::EngineCommand

#![deny(missing_docs)]

pub mod bridge;
pub mod fmt;
pub mod i18n;
pub mod paths;
pub mod picker;
pub mod shell;
pub mod state;
pub mod theme;

pub use bridge::UiBridge;
pub use state::DashboardState;
