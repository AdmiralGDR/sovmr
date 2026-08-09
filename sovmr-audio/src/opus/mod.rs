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

//! A sovereign Opus encoder, built up in stages against the reference
//! decoder. This module is additive: nothing else in the audio path
//! depends on it yet, and copying audio through untouched stays the
//! default. Each stage lands with its own verification before the next.
//!
//! Stage 1, here: the range coder — Opus's entropy layer (RFC 6716
//! §4.1). It is the foundation everything above it writes through, so it
//! ships first and alone, proven by round-trip: whatever the encoder
//! writes, the decoder reads back exactly.

pub mod rangecoder;
