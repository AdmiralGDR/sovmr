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

//! # sovmr-mkv — sovereign Matroska
//!
//! Matroska is the container long multi-track material actually arrives
//! in: a twelve-hour master with six language tracks, subtitles and
//! chapters is far more often `.mkv` than `.mp4`. sovmr could not open
//! one at all.
//!
//! Like [`sovmr_mp4`](../sovmr_mp4/index.html) next door, this crate
//! works entirely on borrowed `&[u8]`, so a mapped fifty-gigabyte source
//! is walked without copying a byte, and it depends on nothing outside
//! the standard library.
//!
//! * [`ebml`] — the tag-length-value encoding Matroska is built from
//!   ([RFC 8794]);
//! * [`ids`] — the Element IDs sovmr reads;
//! * [`demux`] — a file's clock and its tracks;
//! * [`blocks`] — the coded frames inside its clusters;
//! * [`codec`] — turning its codec names and setup bytes into what the
//!   engine already speaks.
//!
//! # Telling the two containers apart
//!
//! By the bytes, never the extension. A file named `.mkv` that is really
//! an MP4 is a file we should still read correctly, and negative trust
//! says the name is a claim rather than a fact.
//!
//! ```
//! use sovmr_mkv::is_matroska;
//! assert!(is_matroska(&[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x02]));
//! // An ISOBMFF file starts with a size then `ftyp`.
//! assert!(!is_matroska(b"\0\0\0\x20ftypisom"));
//! ```
//!
//! [RFC 8794]: https://www.rfc-editor.org/rfc/rfc8794.html

#![deny(missing_docs)]

pub mod blocks;
pub mod codec;
pub mod demux;
pub mod ebml;
pub mod ids;
pub mod mux;
pub mod probe;
pub mod source;

/// The `EBML` master element every Matroska file opens with — the magic
/// number, in the only sense EBML has one.
pub const EBML_HEADER_ID: u32 = 0x1A45_DFA3;

/// Whether these bytes begin an EBML document (and so, in this project's
/// world, a Matroska file).
///
/// Checked against the leading four octets rather than the file name.
pub fn is_matroska(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == EBML_HEADER_ID.to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ebml_magic_is_recognised() {
        assert!(is_matroska(&[0x1A, 0x45, 0xDF, 0xA3]));
        assert!(is_matroska(&[0x1A, 0x45, 0xDF, 0xA3, 0x93, 0x42]));
    }

    #[test]
    fn other_containers_and_short_buffers_are_not_matroska() {
        assert!(!is_matroska(b"\0\0\0\x20ftypisom"));
        assert!(!is_matroska(&[0x1A, 0x45, 0xDF]), "three bytes is not four");
        assert!(!is_matroska(&[]));
        // One byte off is not a near miss, it is a different file.
        assert!(!is_matroska(&[0x1A, 0x45, 0xDF, 0xA4]));
    }
}
