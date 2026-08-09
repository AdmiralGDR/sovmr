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

//! Turning raw bytes into a path the operating system will accept.
//!
//! File pickers and drag-and-drop hand over `file://` URIs, whose
//! percent-escapes decode to *bytes*. What those bytes mean is the one
//! genuinely platform-specific question in the whole intake path, and
//! the two answers disagree:
//!
//! * Unix paths are arbitrary byte strings that owe nothing to UTF-8, so
//!   the bytes are the path, verbatim. Refusing a filename because it is
//!   not valid UTF-8 would reject files the user can plainly see.
//! * Windows paths are UTF-16. Bytes have to be decoded first, and
//!   [RFC 3986] settles the encoding: percent-escapes in a URI carry
//!   UTF-8. Bytes that are not valid UTF-8 name no Windows file at all.
//!
//! [RFC 3986]: https://www.rfc-editor.org/rfc/rfc3986#section-2.5

use std::ffi::OsString;

/// Interpret decoded URI bytes as a path name for this platform.
///
/// Returns `None` only where the bytes cannot name a file at all — on
/// Windows, when they are not valid UTF-8. On Unix every byte string is
/// a legal name, so the answer is always `Some`.
///
/// Callers are expected to turn `None` into their own user-facing
/// rejection; this layer deliberately has no opinion about how an intake
/// failure should be worded.
///
/// # Example
///
/// ```
/// # use sovmr_os::os_string_from_bytes;
/// // Plain ASCII is a valid name on every platform.
/// assert_eq!(
///     os_string_from_bytes(b"clip.mp4".to_vec()).unwrap(),
///     std::ffi::OsString::from("clip.mp4")
/// );
/// ```
pub fn os_string_from_bytes(bytes: Vec<u8>) -> Option<OsString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Some(OsString::from_vec(bytes))
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(bytes).ok().map(OsString::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_names_survive_on_every_platform() {
        let got = os_string_from_bytes(b"/home/dev/\xd0\xba\xd0\xbb\xd0\xb8\xd0\xbf.mp4".to_vec())
            .expect("valid UTF-8 is a legal name everywhere");
        assert_eq!(got.to_string_lossy(), "/home/dev/клип.mp4");
    }

    #[test]
    fn an_empty_name_is_still_an_answer_not_a_rejection() {
        // Emptiness is the caller's business to judge: paths.rs already
        // has a `NotAbsolute` check that catches it with a better
        // message than this layer could give.
        assert_eq!(os_string_from_bytes(Vec::new()), Some(OsString::new()));
    }

    #[cfg(unix)]
    #[test]
    fn unix_keeps_bytes_that_are_not_utf8() {
        // A real filename can be any byte string. Rejecting this would
        // make files the user can see in their file manager un-openable.
        use std::os::unix::ffi::OsStrExt;
        let got = os_string_from_bytes(vec![b'/', b't', 0xFF, 0xFE]).expect("unix takes any bytes");
        assert_eq!(got.as_os_str().as_bytes(), &[b'/', b't', 0xFF, 0xFE]);
    }

    #[cfg(not(unix))]
    #[test]
    fn windows_refuses_bytes_that_name_no_file() {
        // UTF-16 has no room for these, so there is no file to open and
        // no honest way to guess one.
        assert_eq!(os_string_from_bytes(vec![b'/', b't', 0xFF, 0xFE]), None);
    }
}
