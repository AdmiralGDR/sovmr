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

//! Reading and writing fields that do not fall on byte boundaries.
//!
//! FLAC packs almost everything at bit granularity — a subframe type is
//! six bits, a Rice parameter is four or five, a residual is however
//! many the encoder chose. Every read here is checked: running off the
//! end returns `None` rather than reading whatever follows in memory,
//! because a truncated frame is exactly what a damaged file looks like.

/// A most-significant-bit-first cursor over borrowed bytes.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    at: usize,
}

impl<'a> BitReader<'a> {
    /// Start reading at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, at: 0 }
    }

    /// Bits consumed so far.
    pub fn bit_position(&self) -> usize {
        self.at
    }

    /// Whole bytes consumed, rounding down.
    pub fn byte_position(&self) -> usize {
        self.at / 8
    }

    /// Whether the cursor sits exactly on a byte boundary.
    pub fn is_aligned(&self) -> bool {
        self.at.is_multiple_of(8)
    }

    /// Discard bits up to the next byte boundary.
    pub fn align(&mut self) {
        self.at = self.at.div_ceil(8) * 8;
    }

    /// Read `n` bits (`n <= 32`) as an unsigned value.
    pub fn u32(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        debug_assert!(n <= 32);
        let mut out = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.at / 8)?;
            let bit = (byte >> (7 - (self.at % 8))) & 1;
            out = (out << 1) | u32::from(bit);
            self.at += 1;
        }
        Some(out)
    }

    /// Read `n` bits without consuming them, or `None` if fewer remain.
    ///
    /// A Huffman decoder cannot know where a codeword ends until it has
    /// recognised it, so it must be able to look and then decide. The
    /// refusal at the end of the buffer matters as much as the value: a
    /// codeword that runs off the end of a frame is a damaged file, and
    /// zeroes invented to complete it would decode as sound.
    pub fn peek(&self, n: u32) -> Option<u32> {
        let mut ahead = BitReader {
            data: self.data,
            at: self.at,
        };
        ahead.u32(n)
    }

    /// Step over `n` bits, or refuse if fewer remain.
    ///
    /// The refusal is the whole value of this over adding to a counter:
    /// a length field in a damaged file can say anything, and a cursor
    /// that walked past the end would report the *next* structure's
    /// failure instead of this one's.
    pub fn skip(&mut self, n: u32) -> Option<()> {
        let end = self.at.checked_add(n as usize)?;
        if end > self.data.len() * 8 {
            return None;
        }
        self.at = end;
        Some(())
    }

    /// Read `n` bits as a two's-complement signed value.
    ///
    /// Sign extension is the point: FLAC's coefficients and residuals
    /// are signed at widths the language has no type for, and reading
    /// them unsigned turns every negative number into a large positive
    /// one — audible as a burst of noise, not as a quiet error.
    pub fn i32(&mut self, n: u32) -> Option<i32> {
        if n == 0 {
            return Some(0);
        }
        let raw = self.u32(n)?;
        if n == 32 {
            return Some(raw as i32);
        }
        let sign = 1u32 << (n - 1);
        Some(if raw & sign != 0 {
            (raw as i64 - (1i64 << n)) as i32
        } else {
            raw as i32
        })
    }

    /// Read a unary value: count zero bits until a one, return the count.
    ///
    /// Bounded by the buffer, so a run of zeros to the end of a damaged
    /// file ends the read instead of spinning.
    pub fn unary(&mut self) -> Option<u32> {
        let mut count = 0u32;
        loop {
            let byte = *self.data.get(self.at / 8)?;
            let bit = (byte >> (7 - (self.at % 8))) & 1;
            self.at += 1;
            if bit == 1 {
                return Some(count);
            }
            count += 1;
        }
    }
}

/// A most-significant-bit-first bit writer.
///
/// The exact counterpart of [`BitReader`], and deliberately so: an
/// encoder and a decoder that disagree about bit order produce a file
/// that only the encoder can read, which is the least useful kind of
/// output there is. The round-trip tests below hold the two together.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits already written into the byte under construction, 0..8.
    partial: u32,
    /// The byte under construction, filled from the top down.
    current: u8,
}

impl BitWriter {
    /// An empty writer.
    pub fn new() -> Self {
        BitWriter::default()
    }

    /// Bits written so far, including any part-filled byte.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.partial as usize
    }

    /// Whether the next bit would start a fresh byte.
    pub fn is_aligned(&self) -> bool {
        self.partial == 0
    }

    /// Append the low `n` bits of `value`, most significant first.
    ///
    /// Bits above `n` are ignored rather than trusted: a caller that
    /// passes a value too wide for its field has made a mistake, and
    /// letting those bits spill into the next field would corrupt
    /// everything after it instead of just this one.
    pub fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.partial += 1;
            if self.partial == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.partial = 0;
            }
        }
    }

    /// Append a signed value in `n` bits, two's complement.
    pub fn write_signed(&mut self, value: i32, n: u32) {
        self.write(value as u32, n);
    }

    /// Append `count` zero bits followed by a one — the unary form
    /// [`BitReader::unary`] reads back.
    pub fn write_unary(&mut self, count: u32) {
        for _ in 0..count {
            self.write(0, 1);
        }
        self.write(1, 1);
    }

    /// Pad with zero bits up to the next byte boundary.
    ///
    /// Zeros rather than anything else because that is what every format
    /// here specifies, and because a reader skipping to alignment must
    /// not find something that looks like data.
    pub fn align(&mut self) {
        while self.partial != 0 {
            self.write(0, 1);
        }
    }

    /// Finish, padding to a byte boundary, and take the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_read_most_significant_bit_first() {
        // 0b1011_0010 read as 3 + 5 bits.
        let mut r = BitReader::new(&[0b1011_0010]);
        assert_eq!(r.u32(3), Some(0b101));
        assert_eq!(r.u32(5), Some(0b10010));
        assert_eq!(r.u32(1), None, "past the end is None, not zero");
    }

    #[test]
    fn a_field_may_straddle_a_byte_boundary() {
        // FLAC's headers do this constantly; a reader that only worked
        // within bytes would be right for the first field and wrong
        // after.
        let mut r = BitReader::new(&[0b0000_1111, 0b0000_0000]);
        assert_eq!(r.u32(4), Some(0));
        assert_eq!(r.u32(8), Some(0b1111_0000));
    }

    #[test]
    fn signed_fields_sign_extend() {
        // All ones in four bits is -1, not 15. Getting this wrong turns
        // every negative coefficient into a large positive one.
        let mut r = BitReader::new(&[0b1111_0111]);
        assert_eq!(r.i32(4), Some(-1));
        assert_eq!(r.i32(4), Some(7), "and a positive stays positive");

        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.i32(1), Some(-1), "one bit of all-ones is -1");
    }

    #[test]
    fn a_full_width_signed_read_does_not_overflow() {
        let bytes = (-5i32).to_be_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.i32(32), Some(-5));
    }

    #[test]
    fn unary_counts_zeros_up_to_the_terminating_one() {
        // 0b0001_0100: three zeros then a one, then one zero then a one.
        let mut r = BitReader::new(&[0b0001_0100]);
        assert_eq!(r.unary(), Some(3));
        assert_eq!(r.unary(), Some(1));
    }

    #[test]
    fn unary_stops_at_the_end_instead_of_spinning() {
        // A damaged file can be all zeros. Bounded execution: the read
        // ends, it does not loop looking for a one that never comes.
        let mut r = BitReader::new(&[0, 0, 0]);
        assert_eq!(r.unary(), None);
    }

    #[test]
    fn alignment_reports_and_skips_correctly() {
        let mut r = BitReader::new(&[0xFF, 0xFF]);
        assert!(r.is_aligned());
        r.u32(3);
        assert!(!r.is_aligned());
        r.align();
        assert!(r.is_aligned());
        assert_eq!(r.byte_position(), 1);
    }

    // ---- the writer, held to the reader ----------------------------

    #[test]
    fn what_the_writer_writes_the_reader_reads() {
        // The property that matters. Neither half is checked against
        // itself: they are checked against each other, across widths
        // that straddle byte boundaries in every way.
        let fields: [(u32, u32); 9] = [
            (0b101, 3),
            (0, 1),
            (0xFF, 8),
            (1, 1),
            (0x1234, 16),
            (0b11, 2),
            (0xDEAD_BEEF, 32),
            (0, 5),
            (0x7F, 7),
        ];
        let mut w = BitWriter::new();
        for (value, n) in fields {
            w.write(value, n);
        }
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        for (value, n) in fields {
            assert_eq!(r.u32(n), Some(value), "field of {n} bits");
        }
    }

    #[test]
    fn signed_fields_survive_the_round_trip_at_every_width() {
        // Sign extension is the reader's job and truncation the
        // writer's; a mismatch shows here and nowhere else.
        for n in 2..=32u32 {
            let limit = if n >= 32 {
                i32::MAX
            } else {
                (1 << (n - 1)) - 1
            };
            for value in [0, 1, -1, limit, -limit - 1] {
                let mut w = BitWriter::new();
                w.write_signed(value, n);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                assert_eq!(r.i32(n), Some(value), "{value} in {n} bits");
            }
        }
    }

    #[test]
    fn unary_round_trips_including_the_zero_case() {
        for count in [0u32, 1, 2, 7, 8, 9, 100] {
            let mut w = BitWriter::new();
            w.write_unary(count);
            let bytes = w.finish();
            assert_eq!(BitReader::new(&bytes).unary(), Some(count));
        }
    }

    #[test]
    fn bits_above_the_field_width_are_dropped_not_spilled() {
        // A caller passing a value too wide has made a mistake. Letting
        // the excess run into the next field would corrupt everything
        // after it rather than just this one, and the damage would
        // surface far from its cause.
        let mut w = BitWriter::new();
        w.write(0xFFFF_FFFF, 4);
        w.write(0b101, 3);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u32(4), Some(0b1111));
        assert_eq!(r.u32(3), Some(0b101), "the next field is untouched");
    }

    #[test]
    fn a_part_filled_byte_is_padded_with_zeros() {
        let mut w = BitWriter::new();
        w.write(0b101, 3);
        assert!(!w.is_aligned());
        assert_eq!(w.bit_len(), 3);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b1010_0000], "padded low, not left dangling");
    }

    #[test]
    fn aligning_mid_stream_leaves_the_reader_in_step() {
        let mut w = BitWriter::new();
        w.write(0b11, 2);
        w.align();
        w.write(0xAB, 8);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u32(2), Some(0b11));
        r.align();
        assert_eq!(r.u32(8), Some(0xAB));
    }

    #[test]
    fn an_empty_writer_produces_nothing() {
        assert!(BitWriter::new().finish().is_empty());
        assert_eq!(BitWriter::new().bit_len(), 0);
        assert!(BitWriter::new().is_aligned());
    }

    #[test]
    fn a_long_mixed_stream_round_trips_exactly() {
        // Deterministic, and long enough that any drift in the bit
        // position accumulates into a visible failure rather than
        // hiding in one lucky field.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let fields: Vec<(u32, u32)> = (0..500)
            .map(|_| {
                let n = (next() % 32) as u32 + 1;
                let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
                ((next() as u32) & mask, n)
            })
            .collect();

        let mut w = BitWriter::new();
        for (v, n) in &fields {
            w.write(*v, *n);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for (i, (v, n)) in fields.iter().enumerate() {
            assert_eq!(r.u32(*n), Some(*v), "field {i} of {n} bits");
        }
    }
}
