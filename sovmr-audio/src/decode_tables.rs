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

//! Reading the codebooks backwards.
//!
//! # Why this is not just the tables again
//!
//! [`crate::codebooks`] answers "which bits mean this tuple", which is
//! what an encoder asks. A decoder asks the other question — "which
//! tuple do these bits mean" — and cannot ask it directly, because it
//! does not know where the codeword ends until it has recognised it.
//!
//! The property that makes it answerable is the one the tables were
//! already proven to have: no codeword is a prefix of another. So a
//! decoder can read one bit at a time and ask, after each, whether what
//! it holds is a whole codeword. Exactly one length can ever match.
//!
//! # Built, not stored
//!
//! The reverse direction is derived from the forward tables at first
//! use rather than written down again. Four thousand three hundred and
//! sixty-three codewords transcribed a second time is four thousand
//! three hundred and sixty-three chances to differ from the tables that
//! were measured — and a decoder that disagreed with our encoder would
//! be wrong in a way that only shows up on somebody else's file.

use std::collections::HashMap;

/// A codeword's bits and their count, as [`crate::codebooks`] stores it.
pub type Codeword = (u32, u8);

/// One book, arranged for reading rather than writing.
pub struct DecodeBook {
    /// `(length, bits)` to the index the forward table used.
    by_code: HashMap<(u8, u32), usize>,
    /// Codeword lengths present, shortest first — the only lengths worth
    /// testing as bits arrive.
    lengths: Vec<u8>,
}

impl DecodeBook {
    /// Turn a forward table around.
    ///
    /// # Panics
    ///
    /// If two entries share a codeword. The forward tables are proven
    /// prefix-free by their own gate, so this cannot fire for them —
    /// but it would fire loudly for a table edited by hand, which is
    /// the point of checking rather than overwriting.
    pub fn from_forward(book: &[Codeword]) -> DecodeBook {
        let mut by_code = HashMap::with_capacity(book.len());
        let mut lengths: Vec<u8> = Vec::new();
        for (index, (bits, length)) in book.iter().enumerate() {
            let previous = by_code.insert((*length, *bits), index);
            assert!(
                previous.is_none(),
                "two entries share the codeword {bits:b} of {length} bits"
            );
            if !lengths.contains(length) {
                lengths.push(*length);
            }
        }
        lengths.sort_unstable();
        DecodeBook { by_code, lengths }
    }

    /// Shortest and longest codeword in the book.
    pub fn length_range(&self) -> (u8, u8) {
        (
            self.lengths.first().copied().unwrap_or(0),
            self.lengths.last().copied().unwrap_or(0),
        )
    }

    /// Read one codeword, returning the forward table's index for it and
    /// how many bits it took.
    ///
    /// `peek` supplies the next bits, most significant first, and is
    /// asked for whole lengths rather than one bit at a time — the
    /// caller usually has them already and asking bit by bit would make
    /// this the slow part of a decoder for no reason.
    ///
    /// Returns `None` when no codeword matches any length the book uses,
    /// which means the bits are not this book's — a truncated frame, or
    /// a reader that has lost its place. Guessing the nearest match
    /// would turn a detectable fault into audible noise.
    pub fn read<F>(&self, mut peek: F) -> Option<(usize, u8)>
    where
        F: FnMut(u8) -> Option<u32>,
    {
        for length in &self.lengths {
            let bits = peek(*length)?;
            if let Some(index) = self.by_code.get(&(*length, bits)) {
                return Some((*index, *length));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebooks::{
        ESCAPE_11, PAIR_10, PAIR_5, PAIR_6, PAIR_7, PAIR_8, PAIR_9, QUAD_1, QUAD_2, QUAD_3, QUAD_4,
    };

    /// Read a codeword out of a bit string, most significant first.
    fn peeker(bits: u32, available: u8) -> impl FnMut(u8) -> Option<u32> {
        move |want| {
            if want > available {
                return None;
            }
            Some(bits >> (available - want))
        }
    }

    #[test]
    fn every_codeword_of_every_book_reads_back_to_itself() {
        // The whole claim, on all four thousand three hundred and
        // sixty-three. A reverse table that disagreed with the forward
        // one anywhere would be wrong in a way that only shows up on
        // somebody else's file.
        let mut checked = 0usize;
        // All eleven, not a sample: a table that disagreed only in the
        // one book nobody looked at is the table that would ship.
        for book in [
            &QUAD_1[..],
            &QUAD_2[..],
            &QUAD_3[..],
            &QUAD_4[..],
            &PAIR_5[..],
            &PAIR_6[..],
            &PAIR_7[..],
            &PAIR_8[..],
            &PAIR_9[..],
            &PAIR_10[..],
            &ESCAPE_11[..],
        ] {
            let decode = DecodeBook::from_forward(book);
            for (index, (bits, length)) in book.iter().enumerate() {
                let (read, used) = decode
                    .read(peeker(*bits, *length))
                    .unwrap_or_else(|| panic!("entry {index} did not read back"));
                assert_eq!(read, index, "entry {index} read as {read}");
                assert_eq!(used, *length, "entry {index} took {used} bits");
                checked += 1;
            }
        }
        // Every codeword measured out of a decoder in stage 4: 81+81+
        // 625+625 for the quads, 81+81+225+225+625+625 for the pairs,
        // and 1089 for the escape book.
        assert_eq!(checked, 4363, "the books hold 4363 codewords");
    }

    #[test]
    fn a_codeword_is_recognised_with_bits_still_to_come() {
        // A decoder never knows where a codeword ends: the bits after it
        // belong to the next one. So every entry is read again with a
        // full word of trailing rubbish behind it, and must take exactly
        // as many bits as before.
        let decode = DecodeBook::from_forward(&QUAD_1);
        for (index, (bits, length)) in QUAD_1.iter().enumerate() {
            let trailing = 32 - length;
            let padded = (*bits << trailing) | (0x5A5A_5A5A >> length);
            let (read, used) = decode
                .read(peeker(padded, 32))
                .unwrap_or_else(|| panic!("entry {index} vanished behind its neighbours"));
            assert_eq!((read, used), (index, *length), "entry {index}");
        }
    }

    #[test]
    fn bits_that_are_not_a_codeword_are_refused() {
        // A truncated frame, or a reader that has lost its place.
        // Guessing the nearest match turns a detectable fault into
        // audible noise that nothing reports.
        let decode = DecodeBook::from_forward(&QUAD_1);
        // Ask for lengths the peek cannot supply: the book's shortest
        // codeword is one bit, so nothing at all must fail.
        assert_eq!(decode.read(|_| None), None);
    }

    #[test]
    fn the_length_range_is_the_books_own() {
        // What a caller needs to size its lookahead. Codebook 1 runs
        // from one bit to eleven, measured out of a decoder in an
        // earlier milestone.
        let (shortest, longest) = DecodeBook::from_forward(&QUAD_1).length_range();
        assert_eq!(shortest, 1);
        assert_eq!(longest, 11);
        let (shortest, longest) = DecodeBook::from_forward(&PAIR_10).length_range();
        assert!(shortest >= 1 && longest > shortest, "{shortest}..{longest}");
    }

    #[test]
    #[should_panic(expected = "share the codeword")]
    fn a_table_with_a_repeated_codeword_is_refused_rather_than_shadowed() {
        // Not reachable from the shipped tables, which are proven
        // prefix-free. Reachable from a hand edit, and silently keeping
        // the last of two would make one tuple unreachable forever.
        let broken: [Codeword; 2] = [(0b10, 2), (0b10, 2)];
        let _ = DecodeBook::from_forward(&broken);
    }
}
