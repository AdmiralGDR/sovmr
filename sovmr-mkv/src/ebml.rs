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

//! EBML: the tag-length-value encoding Matroska is written in
//! ([RFC 8794]).
//!
//! Everything here works on borrowed `&[u8]`, exactly like the ISOBMFF
//! reader next door, so a mapped twelve-hour file is walked without
//! copying a byte.
//!
//! # The one encoding rule worth stating
//!
//! A Variable-Size Integer begins with `VINT_WIDTH`, a run of zero bits,
//! terminated by `VINT_MARKER`, a single one bit. The number of bits up
//! to and including the marker is the integer's length in octets. What
//! follows is `VINT_DATA`.
//!
//! Element IDs keep their marker bit — that is how Matroska writes them
//! down and how every published table of IDs quotes them (`Segment` is
//! `0x18538067`, not `0x08538067`), so comparing against those constants
//! needs no unmasking. Data sizes have the marker stripped, because there
//! the number means a length.
//!
//! **A data size whose `VINT_DATA` bits are all ones means "unknown".**
//! Read from the RFC rather than recalled: an all-*zero* `VINT_DATA` is
//! an ordinary empty element, and confusing the two would turn every
//! zero-length element into an unbounded one.
//!
//! [RFC 8794]: https://www.rfc-editor.org/rfc/rfc8794.html

use std::fmt;

/// Longest Element ID this reader accepts, in octets.
///
/// RFC 8794's default `EBMLMaxIDLength`. Files may declare a larger one
/// in their header; none in the wild do, and accepting arbitrary lengths
/// would mean trusting a stranger's number to size our own reads.
pub const MAX_ID_OCTETS: usize = 4;
/// Longest Element Data Size this reader accepts, in octets (RFC 8794's
/// default `EBMLMaxSizeLength`).
pub const MAX_SIZE_OCTETS: usize = 8;

/// Why a stretch of bytes is not valid EBML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EbmlError {
    /// The buffer ended in the middle of a header or a payload.
    Truncated,
    /// A Variable-Size Integer's leading octet was zero, which would
    /// describe a length this reader refuses to honour.
    BadVint,
    /// An Element ID longer than [`MAX_ID_OCTETS`].
    IdTooLong,
    /// A data size longer than [`MAX_SIZE_OCTETS`].
    SizeTooLong,
    /// A declared payload runs past the end of its container.
    Overrun,
}

impl fmt::Display for EbmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self {
            EbmlError::Truncated => "buffer ended inside an EBML element",
            EbmlError::BadVint => "malformed variable-size integer",
            EbmlError::IdTooLong => "element id longer than 4 octets",
            EbmlError::SizeTooLong => "element size longer than 8 octets",
            EbmlError::Overrun => "element payload runs past its container",
        };
        f.write_str(what)
    }
}

impl std::error::Error for EbmlError {}

/// A parsed Variable-Size Integer.
struct Vint {
    /// The whole integer including its marker bit — what an Element ID
    /// is quoted as.
    raw: u64,
    /// `VINT_DATA` alone, marker stripped — what a length means.
    value: u64,
    /// Octets consumed.
    len: usize,
    /// Whether `VINT_DATA` was all ones (the "unknown size" sentinel).
    all_ones: bool,
}

/// Read one Variable-Size Integer at the start of `data`.
fn read_vint(data: &[u8], max_octets: usize, too_long: EbmlError) -> Result<Vint, EbmlError> {
    let first = *data.first().ok_or(EbmlError::Truncated)?;
    if first == 0 {
        // A zero leading octet means the width marker is somewhere in a
        // later byte, describing an integer of at least nine octets.
        return Err(EbmlError::BadVint);
    }
    // Bits before the marker, plus the marker itself, count the octets.
    let len = first.leading_zeros() as usize + 1;
    if len > max_octets {
        return Err(too_long);
    }
    let bytes = data.get(..len).ok_or(EbmlError::Truncated)?;
    let mut raw: u64 = 0;
    for b in bytes {
        raw = (raw << 8) | u64::from(*b);
    }
    // The marker sits at bit `len * 7`, so everything below it is data.
    let data_mask = (1u64 << (len * 7)) - 1;
    let value = raw & data_mask;
    Ok(Vint {
        raw,
        value,
        len,
        all_ones: value == data_mask,
    })
}

/// One EBML element: its ID, its payload, and how it was framed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element<'a> {
    /// Element ID **including** its marker bit, matching the published
    /// tables (`Segment` is `0x1853_8067`).
    pub id: u32,
    /// The element's contents.
    pub payload: &'a [u8],
    /// Whether the size was the "unknown" sentinel, in which case
    /// `payload` runs to the end of the enclosing buffer.
    ///
    /// Legal and common for `Segment` and `Cluster` in streamed files,
    /// so it must be handled rather than rejected.
    pub unknown_size: bool,
}

/// Walk the elements laid out in `data`.
///
/// Iteration stops at the first malformed element, yielding the error —
/// a truncated tail is reported, never silently ignored.
///
/// # Example
///
/// ```
/// use sovmr_mkv::ebml::elements;
/// // One element: id 0x80, size 1, payload [0x2a].
/// let bytes = [0x80u8, 0x81, 0x2a];
/// let found: Vec<_> = elements(&bytes).collect::<Result<_, _>>()?;
/// assert_eq!(found.len(), 1);
/// assert_eq!(found[0].id, 0x80);
/// assert_eq!(found[0].payload, &[0x2a]);
/// # Ok::<(), sovmr_mkv::ebml::EbmlError>(())
/// ```
pub fn elements(data: &[u8]) -> Elements<'_> {
    Elements { data, at: 0 }
}

/// Iterator over sibling EBML elements. See [`elements`].
pub struct Elements<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Iterator for Elements<'a> {
    type Item = Result<Element<'a>, EbmlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.data.len() {
            return None;
        }
        let rest = &self.data[self.at..];
        let id = match read_vint(rest, MAX_ID_OCTETS, EbmlError::IdTooLong) {
            Ok(v) => v,
            Err(e) => return Some(Err(self.fail(e))),
        };
        let size = match read_vint(&rest[id.len..], MAX_SIZE_OCTETS, EbmlError::SizeTooLong) {
            Ok(v) => v,
            Err(e) => return Some(Err(self.fail(e))),
        };
        let header = id.len + size.len;
        // Bounded execution: a header is never shorter than two octets,
        // so `at` strictly increases and this iterator always terminates
        // however hostile the input.
        debug_assert!(header >= 2);

        let body = &rest[header..];
        let payload = if size.all_ones {
            // Unknown size: the element owns everything left in its
            // container. Only the last element can, which is exactly how
            // `Segment` and `Cluster` are written when streaming.
            body
        } else {
            let want = match usize::try_from(size.value) {
                Ok(n) => n,
                // A length no pointer could address is an overrun by
                // definition, whatever this architecture is.
                Err(_) => return Some(Err(self.fail(EbmlError::Overrun))),
            };
            match body.get(..want) {
                Some(p) => p,
                None => return Some(Err(self.fail(EbmlError::Overrun))),
            }
        };
        self.at += header + payload.len();
        Some(Ok(Element {
            id: id.raw as u32,
            payload,
            unknown_size: size.all_ones,
        }))
    }
}

impl Elements<'_> {
    /// Park the cursor at the end so a failed parse yields its error
    /// once and then stops, rather than repeating it forever.
    fn fail(&mut self, e: EbmlError) -> EbmlError {
        self.at = self.data.len();
        e
    }
}

/// Find the first child with `id`, ignoring malformed siblings after it.
pub fn find<'a>(data: &'a [u8], id: u32) -> Option<Element<'a>> {
    elements(data).flatten().find(|e| e.id == id)
}

/// Read an unsigned integer element's payload.
///
/// EBML stores integers big-endian in as few octets as the writer felt
/// like using, including none at all — an empty payload is zero, not an
/// error.
pub fn uint(payload: &[u8]) -> u64 {
    payload
        .iter()
        .take(8)
        .fold(0u64, |acc, b| (acc << 8) | u64::from(*b))
}

/// Read a string element's payload, trimming the trailing NULs some
/// writers pad with.
pub fn string(payload: &[u8]) -> &str {
    let end = payload
        .iter()
        .position(|b| *b == 0)
        .unwrap_or(payload.len());
    std::str::from_utf8(&payload[..end]).unwrap_or("")
}

/// Read an IEEE 754 float element's payload (4 or 8 octets per the
/// spec); anything else, including empty, reads as zero.
pub fn float(payload: &[u8]) -> f64 {
    match payload.len() {
        4 => f32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]).into(),
        8 => f64::from_be_bytes([
            payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6],
            payload[7],
        ]),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize an element with the shortest legal ID/size encoding the
    /// test needs, for building fixtures by hand.
    fn elem(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        // One-octet size when it fits in 7 bits, else two.
        if payload.len() < 0x7F {
            out.push(0x80 | payload.len() as u8);
        } else {
            out.push(0x40 | (payload.len() >> 8) as u8);
            out.push((payload.len() & 0xFF) as u8);
        }
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn an_id_keeps_its_marker_bit_so_it_matches_the_published_tables() {
        // Segment is quoted everywhere as 0x18538067. A reader that
        // stripped the marker would look for 0x08538067 and find nothing
        // in any file ever written.
        let mut bytes = vec![0x18, 0x53, 0x80, 0x67];
        bytes.push(0x80); // zero-length payload
        let e = elements(&bytes).next().unwrap().unwrap();
        assert_eq!(e.id, 0x1853_8067);
        assert!(e.payload.is_empty());
    }

    #[test]
    fn every_id_width_from_one_to_four_octets_reads_back() {
        for id in [
            vec![0x80u8],
            vec![0x40, 0x01],
            vec![0x20, 0x00, 0x01],
            vec![0x10, 0x00, 0x00, 0x01],
        ] {
            let bytes = elem(&id, b"x");
            let e = elements(&bytes).next().unwrap().unwrap();
            assert_eq!(e.payload, b"x", "id {id:02x?}");
        }
    }

    #[test]
    fn an_empty_payload_is_empty_not_unknown() {
        // THE distinction the RFC draws: all-zero VINT_DATA is a
        // zero-length element; all-ones is the unknown-size sentinel.
        // Reading a zero as "unknown" would hand every empty element the
        // whole rest of the file.
        let bytes = [0x80u8, 0x80, 0xAA];
        let e = elements(&bytes).next().unwrap().unwrap();
        assert!(!e.unknown_size, "0x80 is a length of zero");
        assert!(e.payload.is_empty());
    }

    #[test]
    fn the_all_ones_size_means_unknown_and_takes_the_rest() {
        // 0xFF is a one-octet size with all seven data bits set.
        let bytes = [0x80u8, 0xFF, 0x01, 0x02, 0x03];
        let e = elements(&bytes).next().unwrap().unwrap();
        assert!(e.unknown_size, "0xFF is the unknown-size sentinel");
        assert_eq!(e.payload, &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn siblings_are_walked_in_order() {
        let mut bytes = elem(&[0x80], b"one");
        bytes.extend(elem(&[0x81], b"two"));
        bytes.extend(elem(&[0x82], b"three"));
        let ids: Vec<u32> = elements(&bytes).map(|e| e.unwrap().id).collect();
        assert_eq!(ids, vec![0x80, 0x81, 0x82]);
    }

    #[test]
    fn a_payload_longer_than_its_container_is_refused() {
        // Negative trust: the declared length is a stranger's number.
        // Believing it would read past the buffer.
        let bytes = [0x80u8, 0x8F, 0x01, 0x02]; // says 15 bytes, has 2
        assert_eq!(
            elements(&bytes).next().unwrap().unwrap_err(),
            EbmlError::Overrun
        );
    }

    #[test]
    fn a_zero_leading_octet_is_refused_rather_than_looped_on() {
        // 0x00 describes a VINT of at least nine octets. Accepting it
        // would mean reading a length from bytes we then have to trust.
        let bytes = [0x00u8, 0x81, 0x2a];
        assert_eq!(
            elements(&bytes).next().unwrap().unwrap_err(),
            EbmlError::BadVint
        );
    }

    #[test]
    fn an_oversized_id_is_named_distinctly_from_an_oversized_size() {
        // 0x08 leads a five-octet ID; the default maximum is four.
        let bytes = [0x08u8, 0, 0, 0, 1, 0x80];
        assert_eq!(
            elements(&bytes).next().unwrap().unwrap_err(),
            EbmlError::IdTooLong
        );
    }

    #[test]
    fn a_header_cut_in_half_is_truncated_not_misread() {
        // ID present, size absent.
        assert_eq!(
            elements(&[0x80u8]).next().unwrap().unwrap_err(),
            EbmlError::Truncated
        );
        // A four-octet ID with only three octets there.
        assert_eq!(
            elements(&[0x10u8, 0, 0]).next().unwrap().unwrap_err(),
            EbmlError::Truncated
        );
    }

    #[test]
    fn a_bad_element_ends_the_walk_instead_of_repeating_forever() {
        // Bounded execution: an iterator that returned the same error
        // every time would hang any `collect`.
        let bytes = [0x00u8, 0x00];
        let mut it = elements(&bytes);
        assert!(it.next().unwrap().is_err());
        assert!(it.next().is_none(), "the walk must stop after a failure");
    }

    #[test]
    fn a_truncated_tail_after_good_elements_is_reported_not_dropped() {
        // The readable prefix is still yielded, and the damage after it
        // is named — silently returning only the good part is how a
        // half-written file passes for a whole one.
        let mut bytes = elem(&[0x80], b"good");
        bytes.extend_from_slice(&[0x81, 0x8F, 0x01]); // claims 15, has 1
        let got: Vec<_> = elements(&bytes).collect();
        assert_eq!(got.len(), 2);
        assert!(got[0].is_ok());
        assert_eq!(got[1].unwrap_err(), EbmlError::Overrun);
    }

    #[test]
    fn integers_read_big_endian_at_any_width() {
        assert_eq!(uint(&[]), 0, "an absent value is zero, not an error");
        assert_eq!(uint(&[0x2a]), 42);
        assert_eq!(uint(&[0x01, 0x00]), 256);
        assert_eq!(uint(&[0xFF; 8]), u64::MAX);
        // Over-long input must not shift off the top and wrap.
        assert_eq!(uint(&[0xFF; 12]), u64::MAX);
    }

    #[test]
    fn strings_stop_at_the_padding_writers_add() {
        assert_eq!(string(b"eng"), "eng");
        assert_eq!(string(b"eng\0\0\0"), "eng");
        assert_eq!(string(b""), "");
        // Invalid UTF-8 degrades to empty rather than panicking: a
        // mislabelled track name must not fail a twelve-hour job.
        assert_eq!(string(&[0xFF, 0xFE]), "");
    }

    #[test]
    fn floats_read_at_both_legal_widths() {
        assert_eq!(float(&1.5f32.to_be_bytes()), 1.5);
        assert_eq!(float(&2.25f64.to_be_bytes()), 2.25);
        // The spec allows only 4 and 8 octets; anything else is not a
        // number we can honour, and zero is the safe reading.
        assert_eq!(float(&[0x01, 0x02]), 0.0);
        assert_eq!(float(&[]), 0.0);
    }

    #[test]
    fn find_returns_the_first_match_and_none_when_absent() {
        let mut bytes = elem(&[0x80], b"a");
        bytes.extend(elem(&[0x81], b"b"));
        bytes.extend(elem(&[0x80], b"c"));
        assert_eq!(find(&bytes, 0x80).unwrap().payload, b"a");
        assert_eq!(find(&bytes, 0x81).unwrap().payload, b"b");
        assert!(find(&bytes, 0x99).is_none());
    }
}
