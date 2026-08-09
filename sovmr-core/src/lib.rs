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

//! # sovmr-core — the sovereign heart of the sovmr media engine
//!
//! Engine-agnostic domain logic for batch, visually lossless, GPU-accelerated
//! media compression. This crate is **standard-library-only, forever**, and is
//! designed to be embedded anywhere: the sovmr desktop app today, a headless
//! batch web-service tomorrow.
//!
//! ## What lives here
//!
//! | Module       | Responsibility                                              |
//! |--------------|-------------------------------------------------------------|
//! | [`job`]      | The unit of work: [`job::MediaJob`], codecs, quality targets |
//! | [`encoder`]  | The [`encoder::EncoderProvider`] facade all backends implement |
//! | [`queue`]    | Background engine thread + worker pool, message-passing only |
//! | [`hardware`] | GPU detection and automatic encode planning                  |
//! | [`validate`] | Hostile-input gate: container signature sniffing             |
//! | [`config`]   | Every tunable constant, in one place                         |
//!
//! ## What deliberately does NOT live here
//!
//! No encoder flag, no NVENC preset string, no GUI type. Backends
//! implement [`encoder::EncoderProvider`] in sibling crates (today:
//! `sovmr-native`, the zero-copy NVIDIA Video Codec SDK bridge) and
//! frontends consume [`queue::messages::EngineEvent`]s; both are
//! swappable without touching this crate.
//!
//! ## Ten-line tour
//!
//! ```ignore
//! use std::sync::Arc;
//! use sovmr_core::{hardware::HardwareConfigurator, job::{JobId, MediaJob}};
//! use sovmr_core::queue::{messages::EngineCommand, scheduler};
//!
//! let provider = Arc::new(MyBackend::new());          // any EncoderProvider
//! let plan = HardwareConfigurator::detect(provider.as_ref())?;
//! let (engine, events) = scheduler::start(provider, plan.worker_slots);
//! engine.send(EngineCommand::Enqueue(job));           // fire and forget
//! for event in events { /* render progress, completions, failures */ }
//! ```

#![deny(missing_docs)]

pub mod complexity;
pub mod config;
pub mod encoder;
pub mod hardware;
pub mod heuristics;
pub mod identity;
pub mod job;
pub mod quality;
pub mod queue;
pub mod validate;
