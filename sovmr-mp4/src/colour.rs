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

//! What the colours in a video actually mean.
//!
//! # Why this is not optional
//!
//! Three code points decide how a decoder turns numbers into light:
//! which primaries the numbers are relative to, what transfer curve was
//! applied, and which matrix took RGB to luma and chroma. A fourth says
//! whether the range is the full 0–255 or the broadcast 16–235.
//!
//! Get them wrong and nothing crashes — the picture simply comes out the
//! wrong colour. A professional editor reads them and trusts them; with
//! nothing to read it guesses, and a guess that happens to be right on
//! one file is wrong on the next.
//!
//! These are the same numbers everywhere. ISO/IEC 23091-2 defines them,
//! H.264 and HEVC put them in their VUI, AV1 puts them in its sequence
//! header, ISOBMFF carries them in a `colr` box and Matroska in a
//! `Colour` element. One type therefore serves every container this
//! project reads or writes.
//!
//! # What is deliberately not modelled
//!
//! ICC profiles. `colr` can also carry `rICC`/`prof` payloads, and
//! Matroska has mastering-display metadata for HDR. Neither is invented
//! here: this carries across what a source stated, and inventing a
//! profile a source never had would be worse than carrying nothing.

/// The code point meaning "the source did not say".
///
/// Two is *unspecified* in every one of the three tables, and it is a
/// legitimate value to write: it says honestly that nothing was
/// declared, which is different from claiming BT.709 because that is
/// what most files are.
pub const UNSPECIFIED: u16 = 2;

/// How a video's samples map to colour.
///
/// Every field is a code point from ISO/IEC 23091-2, carried verbatim
/// from wherever the source stated it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Colour {
    /// `colour_primaries`: 1 is BT.709, 9 is BT.2020.
    pub primaries: u16,
    /// `transfer_characteristics`: 1 is BT.709, 16 is PQ, 18 is HLG.
    pub transfer: u16,
    /// `matrix_coefficients`: 1 is BT.709, 9 is BT.2020 non-constant.
    pub matrix: u16,
    /// Whether the samples use the full range rather than 16–235.
    pub full_range: bool,
}

impl Default for Colour {
    /// Unspecified throughout, which is *not* the same as zero.
    ///
    /// Zero is a real code point: reserved for primaries and transfer,
    /// and identity — RGB straight through, no luma/chroma split — for
    /// the matrix. A derived `Default` would therefore describe every
    /// undescribed video as RGB, which is a confident and wrong answer
    /// where [`UNSPECIFIED`] is an honest one.
    fn default() -> Colour {
        Colour {
            primaries: UNSPECIFIED,
            transfer: UNSPECIFIED,
            matrix: UNSPECIFIED,
            full_range: false,
        }
    }
}

/// Four-character code of the `colr` payload this project writes.
const NCLX: [u8; 4] = *b"nclx";
/// Bytes of a `colr` payload in `nclx` form: the type, three code
/// points, and the byte holding the range flag.
const NCLX_PAYLOAD_LEN: usize = 4 + 2 + 2 + 2 + 1;

impl Colour {
    /// Whether this says anything a decoder could not have guessed.
    ///
    /// A description that is unspecified throughout is worth writing
    /// only if the source wrote one; carrying it into a file that had
    /// none adds a box that says nothing.
    pub fn is_meaningful(&self) -> bool {
        self.primaries != UNSPECIFIED
            || self.transfer != UNSPECIFIED
            || self.matrix != UNSPECIFIED
            || self.full_range
    }

    /// Read a `colr` box payload.
    ///
    /// Returns `None` for `rICC`/`prof` payloads and for anything
    /// truncated. An ICC profile is not this type's business, and
    /// guessing at a short box would invent a colour the file never
    /// claimed.
    pub fn from_colr(payload: &[u8]) -> Option<Colour> {
        if payload.len() < NCLX_PAYLOAD_LEN || payload[..4] != NCLX {
            return None;
        }
        Some(Colour {
            primaries: u16::from_be_bytes([payload[4], payload[5]]),
            transfer: u16::from_be_bytes([payload[6], payload[7]]),
            matrix: u16::from_be_bytes([payload[8], payload[9]]),
            // Top bit; the remaining seven are reserved and zero.
            full_range: payload[10] & 0x80 != 0,
        })
    }

    /// The payload of a `colr` box describing this.
    ///
    /// A plain box, not a full box: `colr` carries no version or flags,
    /// and writing four bytes of them would push every field along by
    /// four and describe something else entirely.
    pub fn to_colr(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(NCLX_PAYLOAD_LEN);
        out.extend_from_slice(&NCLX);
        out.extend_from_slice(&self.primaries.to_be_bytes());
        out.extend_from_slice(&self.transfer.to_be_bytes());
        out.extend_from_slice(&self.matrix.to_be_bytes());
        out.push(if self.full_range { 0x80 } else { 0x00 });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BT.709 limited range — what most of the world's video says.
    const BT709: Colour = Colour {
        primaries: 1,
        transfer: 1,
        matrix: 1,
        full_range: false,
    };

    #[test]
    fn a_description_survives_the_round_trip_it_exists_for() {
        for colour in [
            BT709,
            Colour {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: false,
            },
            Colour {
                primaries: 1,
                transfer: 13,
                matrix: 0,
                full_range: true,
            },
            Colour::default(),
        ] {
            assert_eq!(Colour::from_colr(&colour.to_colr()), Some(colour));
        }
    }

    #[test]
    fn the_payload_is_eleven_bytes_and_starts_with_nclx() {
        // A `colr` is a plain box. Treating it as a full box would put
        // four bytes of version and flags in front of the type, and
        // every field after would describe something else.
        let bytes = BT709.to_colr();
        assert_eq!(bytes.len(), NCLX_PAYLOAD_LEN);
        assert_eq!(&bytes[..4], b"nclx");
        assert_eq!(&bytes[4..10], &[0, 1, 0, 1, 0, 1]);
        assert_eq!(bytes[10], 0x00, "limited range");
    }

    #[test]
    fn the_range_flag_is_the_top_bit_and_the_rest_is_reserved() {
        let full = Colour {
            full_range: true,
            ..BT709
        };
        assert_eq!(full.to_colr()[10], 0x80);
        // A decoder that read the wrong bit would swap full and limited
        // range, which crushes blacks or clips whites on every frame.
        assert_eq!(
            Colour::from_colr(&[b'n', b'c', b'l', b'x', 0, 1, 0, 1, 0, 1, 0x80]),
            Some(full)
        );
        assert_eq!(
            Colour::from_colr(&[b'n', b'c', b'l', b'x', 0, 1, 0, 1, 0, 1, 0x00]),
            Some(BT709)
        );
    }

    #[test]
    fn an_icc_payload_is_not_claimed_as_a_description() {
        // `rICC` and `prof` are colour information this type does not
        // model. Reading their bytes as code points would produce a
        // confident description of nothing.
        assert_eq!(Colour::from_colr(b"rICC\0\0\0\0\0\0\0"), None);
        assert_eq!(Colour::from_colr(b"prof\0\0\0\0\0\0\0"), None);
    }

    #[test]
    fn a_truncated_box_is_refused_rather_than_padded() {
        for len in 0..NCLX_PAYLOAD_LEN {
            let payload = &BT709.to_colr()[..len];
            assert_eq!(Colour::from_colr(payload), None, "{len} bytes");
        }
        assert!(Colour::from_colr(&BT709.to_colr()).is_some());
    }

    #[test]
    fn the_default_is_unspecified_and_not_zero() {
        // Zero is a real code point — identity matrix means RGB with no
        // luma/chroma split — so a derived `Default` would describe
        // every undescribed video as RGB. Confidently, and wrongly.
        let d = Colour::default();
        assert_eq!(d.primaries, UNSPECIFIED);
        assert_eq!(d.transfer, UNSPECIFIED);
        assert_eq!(d.matrix, UNSPECIFIED);
        assert_ne!(d.matrix, 0, "zero is identity, not silence");
    }

    #[test]
    fn unspecified_throughout_is_not_worth_writing() {
        // Two means "not stated" in all three tables. Carrying a box
        // that says nothing into a file that had none is noise; a source
        // that stated it is another matter, and the caller decides.
        assert!(!Colour::default().is_meaningful());
        assert!(!Colour {
            primaries: UNSPECIFIED,
            transfer: UNSPECIFIED,
            matrix: UNSPECIFIED,
            full_range: false,
        }
        .is_meaningful());
        assert!(BT709.is_meaningful());
        assert!(Colour {
            full_range: true,
            ..Colour::default()
        }
        .is_meaningful());
    }
}
