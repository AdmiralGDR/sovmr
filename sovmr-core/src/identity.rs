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

//! Which build this is, exactly.
//!
//! Two machines run this program — a Linux box with an Ada card and a
//! Windows box with an Ampere one — and a fix is not delivered until it
//! is on both. "Which version is installed where" therefore has to be a
//! question the binary itself answers, not one inferred from a file
//! date.
//!
//! Three facts identify a build completely:
//!
//! * the **release**, which says what was intended;
//! * the **commit**, which says what was actually compiled — with
//!   `-dirty` when the tree had uncommitted edits, because then the
//!   commit does not identify it and pretending otherwise is worse than
//!   saying nothing;
//! * the **target triple**, which says which machine it can run on.
//!
//! Deliberately absent: a build timestamp. The goal is that the same
//! source produce the same bytes anywhere, and a timestamp guarantees it
//! never does.

/// Semantic version of this release, from the workspace manifest.
pub const RELEASE: &str = env!("CARGO_PKG_VERSION");

/// Short commit hash, `-dirty` when the tree differed, `unknown` when
/// built outside a repository.
pub const COMMIT: &str = env!("SOVMR_COMMIT");

/// Target triple this binary was compiled for.
pub const TARGET: &str = env!("SOVMR_TARGET");

/// Where this build sits in the roadmap, in plain words.
///
/// A release number cannot answer "what is finished and what is being
/// worked on": `2.0.0-rc.10` says only that ten increments have
/// happened. Somebody running the program — including the person who
/// asked for it — should be able to see the answer without reading a
/// commit log, so it is carried in the binary and shown in the
/// interface.
///
/// One line to keep current, deliberately: a scheme that derived this
/// from anything else would be more machinery than the fact is worth.
pub const MILESTONE: &str =
    "stage 6.11 of 7 — EN/RU interface, visible progress, and reproducible signed builds";

/// Whether this build came from a tree with uncommitted changes.
///
/// Worth asking separately: a dirty build is fine to run and not fine to
/// report as "the machine has commit X".
pub fn is_dirty() -> bool {
    COMMIT.ends_with("-dirty")
}

/// One line naming this build exactly.
///
/// # Example
///
/// ```
/// let line = sovmr_core::identity::describe();
/// assert!(line.starts_with("sovmr "));
/// ```
pub fn describe() -> String {
    format!("sovmr {RELEASE} ({COMMIT}, {TARGET})")
}

/// The same, with the roadmap position under it.
///
/// What `--version` prints and what the interface shows: the release
/// identifies the build, the milestone says what that build is for.
pub fn describe_full() -> String {
    format!("{}\n{MILESTONE}", describe())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_all_three_facts() {
        // Any one of them missing makes "which build is on that machine"
        // unanswerable, which is the only reason this module exists.
        let line = describe();
        assert!(line.contains(RELEASE), "{line}");
        assert!(line.contains(COMMIT), "{line}");
        assert!(line.contains(TARGET), "{line}");
    }

    #[test]
    fn the_milestone_says_where_this_build_sits() {
        // A release number cannot answer "what is finished": rc.10 says
        // ten increments happened and nothing about which. Somebody
        // running the program should not have to read a commit log.
        assert!(MILESTONE.contains("stage"), "{MILESTONE}");
        assert!(
            MILESTONE.len() > 20,
            "a milestone that says nothing is worse than none: {MILESTONE}"
        );
        assert!(describe_full().contains(MILESTONE));
        assert!(describe_full().contains(RELEASE));
    }

    #[test]
    fn the_target_is_the_one_this_was_built_for() {
        // A cross-built Windows binary reporting the host triple would
        // make the two machines indistinguishable in a log — which is
        // exactly the confusion this exists to prevent.
        assert!(TARGET.contains('-'), "{TARGET} is not a target triple");
        assert!(!TARGET.is_empty());
    }

    #[test]
    fn a_dirty_build_admits_it() {
        // The suffix is load-bearing: without it a hash would claim to
        // identify a binary compiled from edits that hash never saw.
        assert_eq!(is_dirty(), COMMIT.ends_with("-dirty"));
        if is_dirty() {
            assert!(describe().contains("-dirty"));
        }
    }
}
