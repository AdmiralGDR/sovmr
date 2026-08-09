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

//! Hostile-input gate. Every file that enters the queue
//! passes through here first. We sniff container magic bytes ourselves rather
//! than trusting extensions, and we do it *before* handing anything to an
//! external process.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::config::SNIFF_LEN;

/// Container formats we accept into the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// ISO BMFF family: MP4, MOV, M4V ("ftyp" box at offset 4).
    IsoBmff,
    /// Matroska / WebM (EBML magic 1A 45 DF A3).
    Matroska,
    /// AVI (RIFF .... "AVI ").
    Avi,
    /// MPEG transport stream (sync byte 0x47).
    MpegTs,
}

/// Why a candidate file was refused entry to the queue. Every variant's
/// `Display` text is written for direct display in the UI.
#[derive(Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// Path exists but is not a regular file (directory, device, socket).
    NotAFile,
    /// Filesystem error while opening or reading; carries the OS message.
    Unreadable(String),
    /// Shorter than the smallest possible media header.
    TooSmall,
    /// Leading bytes match no supported container (see [`Container`]).
    UnknownSignature,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::NotAFile => write!(f, "not a regular file"),
            ValidationError::Unreadable(e) => write!(f, "unreadable: {e}"),
            ValidationError::TooSmall => write!(f, "file too small to be media"),
            ValidationError::UnknownSignature => {
                write!(f, "unrecognised container signature")
            }
        }
    }
}

/// Classify a byte prefix. Pure function, fully unit-testable without I/O.
///
/// # Errors
///
/// [`ValidationError::TooSmall`] for prefixes shorter than
/// [`SNIFF_LEN`](crate::config::SNIFF_LEN);
/// [`ValidationError::UnknownSignature`] when no magic matches.
///
/// # Example
///
/// ```
/// use sovmr_core::validate::{sniff_container, Container};
/// let mp4_header = [0, 0, 0, 0x20, b'f', b't', b'y', b'p',
///                   b'i', b's', b'o', b'm', 0, 0, 0, 0];
/// assert_eq!(sniff_container(&mp4_header), Ok(Container::IsoBmff));
/// assert!(sniff_container(b"#!/bin/sh - not media!!!").is_err());
/// ```
pub fn sniff_container(prefix: &[u8]) -> Result<Container, ValidationError> {
    if prefix.len() < SNIFF_LEN {
        return Err(ValidationError::TooSmall);
    }
    if &prefix[4..8] == b"ftyp" {
        return Ok(Container::IsoBmff);
    }
    if prefix.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Ok(Container::Matroska);
    }
    if prefix.starts_with(b"RIFF") && &prefix[8..12] == b"AVI " {
        return Ok(Container::Avi);
    }
    if prefix[0] == 0x47 {
        return Ok(Container::MpegTs);
    }
    Err(ValidationError::UnknownSignature)
}

/// Validate a path on disk: must be a regular file whose leading bytes carry
/// a known media container signature. Symlinks are resolved by `metadata`,
/// so a link pointing at a directory or device node is rejected here.
///
/// # Errors
///
/// Any [`ValidationError`]; all messages are safe to show verbatim in the UI.
pub fn validate_media_file(path: &Path) -> Result<Container, ValidationError> {
    let meta = std::fs::metadata(path).map_err(|e| ValidationError::Unreadable(e.to_string()))?;
    if !meta.is_file() {
        return Err(ValidationError::NotAFile);
    }
    let mut prefix = [0u8; SNIFF_LEN];
    let mut file = File::open(path).map_err(|e| ValidationError::Unreadable(e.to_string()))?;
    let mut read = 0;
    while read < SNIFF_LEN {
        match file.read(&mut prefix[read..]) {
            Ok(0) => break,
            Ok(n) => read += n,
            Err(e) => return Err(ValidationError::Unreadable(e.to_string())),
        }
    }
    sniff_container(&prefix[..read])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn padded(bytes: &[u8]) -> Vec<u8> {
        let mut v = bytes.to_vec();
        v.resize(SNIFF_LEN, 0);
        v
    }

    #[test]
    fn detects_mp4_ftyp() {
        let sig = padded(&[
            0, 0, 0, 0x20, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm',
        ]);
        assert_eq!(sniff_container(&sig), Ok(Container::IsoBmff));
    }

    #[test]
    fn detects_matroska() {
        let sig = padded(&[0x1A, 0x45, 0xDF, 0xA3]);
        assert_eq!(sniff_container(&sig), Ok(Container::Matroska));
    }

    #[test]
    fn detects_avi() {
        let mut sig = padded(b"RIFF\x00\x00\x00\x00AVI ");
        sig.truncate(SNIFF_LEN);
        assert_eq!(sniff_container(&sig), Ok(Container::Avi));
    }

    #[test]
    fn rejects_executable_masquerading_as_mp4() {
        // ELF magic — the classic "renamed .mp4" attack.
        let sig = padded(&[0x7F, b'E', b'L', b'F']);
        assert_eq!(
            sniff_container(&sig),
            Err(ValidationError::UnknownSignature)
        );
    }

    #[test]
    fn rejects_truncated_input() {
        assert_eq!(sniff_container(&[0, 0]), Err(ValidationError::TooSmall));
    }

    #[test]
    fn rejects_directory_on_disk() {
        let dir = std::env::temp_dir();
        assert_eq!(validate_media_file(&dir), Err(ValidationError::NotAFile));
    }
}
