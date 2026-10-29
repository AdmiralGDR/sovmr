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

//! Intake path plumbing, hardened: window backends and
//! external pickers hand us strings in several dialects — plain absolute
//! paths, `file://` URIs (percent-encoded, sometimes with a `localhost`
//! authority), or `text/uri-list` lines with trailing `\r` — and every
//! dialect must resolve to a real absolute [`PathBuf`] or fail with a
//! *named* reason the embedder can print loudly. Paths are treated as
//! **bytes**, not UTF-8 (Linux paths owe us nothing).

use std::ffi::OsString;
use std::path::PathBuf;

/// URI scheme the Freedesktop drag-and-drop dialects use.
const FILE_SCHEME: &str = "file://";
/// Authority some toolkits insert between the scheme and the path.
const LOCALHOST: &str = "localhost";

/// Why a dropped/picked entry could not become a usable path. Rendered
/// verbatim in the loud stderr diagnostics (an intake
/// failure must never be silent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathRejection {
    /// The backend delivered a drop with neither a path nor a name.
    Empty,
    /// A `file://` URI pointed at a remote host we cannot reach.
    RemoteHost(String),
    /// A percent escape was cut off or not hexadecimal.
    BadPercentEscape(String),
    /// The result was not an absolute filesystem path.
    NotAbsolute(String),
    /// The path resolved but nothing exists there (or it is unreadable).
    NotFound(String),
    /// The decoded bytes cannot name a file on this platform at all
    /// (Windows: not valid UTF-8, so there is no UTF-16 name for them).
    NotRepresentable(String),
}

impl std::fmt::Display for PathRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathRejection::Empty => write!(f, "drop carried neither a path nor a name"),
            PathRejection::RemoteHost(h) => {
                write!(
                    f,
                    "file:// URI names a remote host '{h}' — not a local file"
                )
            }
            PathRejection::BadPercentEscape(s) => {
                write!(f, "malformed percent-escape in URI: {s}")
            }
            PathRejection::NotAbsolute(s) => {
                write!(f, "not an absolute filesystem path: {s}")
            }
            PathRejection::NotFound(s) => {
                write!(f, "path does not exist (or is unreadable): {s}")
            }
            PathRejection::NotRepresentable(s) => {
                write!(f, "no filename on this platform matches those bytes: {s}")
            }
        }
    }
}

/// Resolve one raw drop/pick string into an absolute [`PathBuf`].
///
/// Accepts a plain path (`/videos/a.mp4`), a `file://` URI with
/// percent-encoding (`file:///videos/with%20space.mp4`), or the same
/// with a `localhost` authority; trailing `\r`/`\n` from `text/uri-list`
/// payloads is trimmed. Existence is deliberately **not** checked here —
/// that is [`resolve_existing`]'s job — so the pure dialect logic stays
/// unit-testable without a filesystem.
///
/// # Errors
///
/// A [`PathRejection`] naming exactly what was wrong.
pub fn resolve_raw(raw: &str) -> Result<PathBuf, PathRejection> {
    let trimmed = raw.trim_end_matches(['\r', '\n']);
    if trimmed.is_empty() {
        return Err(PathRejection::Empty);
    }
    let path = if let Some(rest) = trimmed.strip_prefix(FILE_SCHEME) {
        // file://[host]/path — only an empty host or "localhost" is us.
        let slash = rest.find('/').unwrap_or(rest.len());
        let (host, path_part) = rest.split_at(slash);
        if !host.is_empty() && host != LOCALHOST {
            return Err(PathRejection::RemoteHost(host.to_string()));
        }
        PathBuf::from(percent_decode(path_part)?)
    } else {
        PathBuf::from(trimmed)
    };
    if !path.is_absolute() {
        return Err(PathRejection::NotAbsolute(path.display().to_string()));
    }
    Ok(path)
}

/// [`resolve_raw`] plus an existence check — the variant the intake
/// actually uses, so a stale URI fails *here* with a clear reason
/// instead of deep inside probing.
///
/// # Errors
///
/// Same as [`resolve_raw`], plus [`PathRejection::NotFound`].
pub fn resolve_existing(raw: &str) -> Result<PathBuf, PathRejection> {
    let path = resolve_raw(raw)?;
    if !path.exists() {
        return Err(PathRejection::NotFound(path.display().to_string()));
    }
    Ok(path)
}

/// Decode `%XX` escapes into raw bytes (Linux paths are bytes, not
/// UTF-8) and return an `OsString` suitable for `PathBuf`.
fn percent_decode(s: &str) -> Result<OsString, PathRejection> {
    let mut bytes = Vec::with_capacity(s.len());
    let mut rest = s.as_bytes();
    while let Some((&b, tail)) = rest.split_first() {
        if b != b'%' {
            bytes.push(b);
            rest = tail;
            continue;
        }
        let (Some(hi), Some(lo)) = (tail.first(), tail.get(1)) else {
            return Err(PathRejection::BadPercentEscape(s.to_string()));
        };
        let (Some(hi), Some(lo)) = (hex_val(*hi), hex_val(*lo)) else {
            return Err(PathRejection::BadPercentEscape(s.to_string()));
        };
        bytes.push((hi << 4) | lo);
        rest = &tail[2..];
    }
    sovmr_os::os_string_from_bytes(bytes)
        .ok_or_else(|| PathRejection::NotRepresentable(s.to_string()))
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Parse a file picker's stdout (`zenity --separator=\n` or `kdialog
/// --separate-output`): one absolute path per line, empty lines and
/// non-absolute noise skipped with a loud stderr note.
pub fn parse_picker_output(stdout: &[u8]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in stdout.split(|b| *b == b'\n') {
        let line = String::from_utf8_lossy(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match resolve_raw(line) {
            Ok(path) => out.push(path),
            Err(why) => eprintln!("sovmr: picker line rejected: {why}"),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_absolute_paths_pass_through() {
        assert_eq!(
            resolve_raw("/videos/clip.mp4").unwrap(),
            PathBuf::from("/videos/clip.mp4")
        );
    }

    #[test]
    fn file_uris_are_decoded_including_spaces_and_utf8() {
        assert_eq!(
            resolve_raw("file:///videos/with%20space.mp4").unwrap(),
            PathBuf::from("/videos/with space.mp4")
        );
        // UTF-8 percent sequences decode byte-wise: "видео".
        assert_eq!(
            resolve_raw("file:///v/%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE.mp4").unwrap(),
            PathBuf::from("/v/видео.mp4")
        );
        // localhost authority is us; uri-list line endings are trimmed.
        assert_eq!(
            resolve_raw("file://localhost/videos/a.mp4\r\n").unwrap(),
            PathBuf::from("/videos/a.mp4")
        );
    }

    /// Only Unix has paths that are bytes.
    ///
    /// Windows paths are UTF-16, so there is no such thing as a
    /// non-UTF-8 one to survive — and the import this needs does not
    /// exist there, which is why `--all-targets` for Windows could not
    /// build the tests at all.
    #[cfg(unix)]
    #[test]
    fn non_utf8_bytes_survive_percent_decoding() {
        use std::os::unix::ffi::OsStrExt;
        let path = resolve_raw("file:///v/%FF%FEclip.mp4").unwrap();
        assert_eq!(
            path.as_os_str().as_bytes(),
            b"/v/\xFF\xFEclip.mp4",
            "paths are bytes, not UTF-8"
        );
    }

    #[test]
    fn every_rejection_is_named() {
        assert_eq!(resolve_raw(""), Err(PathRejection::Empty));
        assert_eq!(resolve_raw("\r\n"), Err(PathRejection::Empty));
        assert!(matches!(
            resolve_raw("file://nas.local/share/clip.mp4"),
            Err(PathRejection::RemoteHost(h)) if h == "nas.local"
        ));
        assert!(matches!(
            resolve_raw("file:///bad%G0.mp4"),
            Err(PathRejection::BadPercentEscape(_))
        ));
        assert!(matches!(
            resolve_raw("file:///cut%2"),
            Err(PathRejection::BadPercentEscape(_))
        ));
        assert!(matches!(
            resolve_raw("relative/clip.mp4"),
            Err(PathRejection::NotAbsolute(_))
        ));
        assert!(matches!(
            resolve_existing("/definitely/not/here.mp4"),
            Err(PathRejection::NotFound(_))
        ));
    }

    #[test]
    fn resolve_existing_accepts_real_files() {
        // This source file itself is a guaranteed-present fixture.
        let me = concat!(env!("CARGO_MANIFEST_DIR"), "/src/paths.rs");
        assert!(resolve_existing(me).is_ok());
    }

    #[test]
    fn picker_output_parses_lines_and_skips_noise() {
        let stdout = b"/videos/a.mp4\n\n/videos/b c.mkv\nnot-absolute\n";
        assert_eq!(
            parse_picker_output(stdout),
            vec![
                PathBuf::from("/videos/a.mp4"),
                PathBuf::from("/videos/b c.mkv")
            ]
        );
        assert!(parse_picker_output(b"").is_empty());
    }
}
