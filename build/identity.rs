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

// Build-time provenance, shared by every build script that needs it.
//
// `include!`d rather than linked: build scripts cannot depend on each
// other, and copying eight lines into two files is how they drift.
//
// # What is emitted, and what deliberately is not
//
// `SOVMR_COMMIT` — the short commit hash, with `-dirty` appended when
// the working tree differs from it. The suffix matters more than the
// hash: a hash alone would claim to identify a binary that was built
// from uncommitted edits, which is a lie told confidently.
//
// There is **no build timestamp**. The goal is that the same source
// produce the same bytes on any machine; a timestamp guarantees it
// never does. A commit hash identifies a build exactly and is a
// function of the source, which is what provenance is supposed to be.

/// Emit `SOVMR_COMMIT` for the crate being built.
///
/// Falls back to `unknown` when git is absent or this is not a
/// repository — a release unpacked from a tarball still builds, it just
/// cannot say which commit it came from, and says so.
fn emit_commit_identity() {
    // Without these, a build script runs once and then reports the
    // commit it happened to see, forever.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");

    let describe = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|hash| !hash.is_empty());

    let identity = match describe {
        None => "unknown".to_string(),
        Some(hash) => {
            let clean = std::process::Command::new("git")
                .args(["status", "--porcelain", "--untracked-files=no"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .map(|out| out.stdout.is_empty())
                .unwrap_or(false);
            if clean {
                hash
            } else {
                format!("{hash}-dirty")
            }
        }
    };
    println!("cargo:rustc-env=SOVMR_COMMIT={identity}");
}
