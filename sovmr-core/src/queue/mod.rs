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

//! Batch queue: a background engine thread plus a worker pool, decoupled
//! from any frontend by two `mpsc` channels — [`messages::EngineCommand`]
//! in, [`messages::EngineEvent`] out. A GUI drains events per frame; a
//! headless service can just block on the receiver. Neither changes the
//! engine.

pub mod messages;
pub mod scheduler;
