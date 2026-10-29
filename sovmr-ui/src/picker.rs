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

//! The sovereign file picker: the **guaranteed** intake path when
//! drag-and-drop is unavailable (upstream winit has no Wayland DnD).
//! No `rfd`, no portal crates — the desktop's own dialog binary
//! (`zenity`, then `kdialog`) is spawned via `std::process::Command` on
//! a background thread; its stdout is parsed into absolute paths.
//!
//! Failure discipline: a user pressing *Cancel* is silence; a
//! missing/broken dialog binary is a loud stderr diagnostic naming
//! every attempt.

use std::path::PathBuf;

/// Returns the chosen absolute paths (empty = cancelled). When no
/// dialog binary works, returns an empty list after printing a loud
/// stderr diagnostic naming every attempt (the intake must never fail
/// silently).
pub fn pick_files_blocking() -> Vec<PathBuf> {
    // Asked of the platform boundary. This used to spawn `zenity` and
    // `kdialog` from here, which on Windows spawned nothing and left the
    // button doing nothing at all — the intake must never fail
    // silently, and on one of the two platforms it always did.
    match sovmr_os::desktop::pick_media_files("Add media to sovmr") {
        Ok(paths) => paths,
        Err(sovmr_os::desktop::PickError::Cancelled) => Vec::new(),
        Err(e) => {
            eprintln!(
                "sovmr: FILE PICKER UNAVAILABLE — {e}\n\
                 sovmr: fallback: drag files onto the window (X11/--x11) or pass them on \
                 the command line: sovmr FILE...",
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    /// The intake must never fail silently.
    ///
    /// Spawning the dialog moved to `sovmr_os::desktop`, where both
    /// platforms are served and both selection formats are parsed and
    /// tested. What is left here is the one promise this layer makes:
    /// a refusal is reported, and cancelling is not a refusal.
    #[test]
    fn cancelling_is_not_reported_as_a_failure() {
        assert_eq!(
            sovmr_os::desktop::PickError::Cancelled.to_string(),
            "cancelled"
        );
        assert_ne!(
            sovmr_os::desktop::PickError::Cancelled,
            sovmr_os::desktop::PickError::Unavailable(String::new())
        );
    }
}
