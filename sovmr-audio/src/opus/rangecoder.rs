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

//! Opus range coder (RFC 6716 §4.1): the entropy layer every other part
//! of an Opus frame writes through.
//!
//! It is a carry-propagating range coder that packs two streams into one
//! buffer from opposite ends — range-coded symbols grow from the front,
//! and raw (uniform) bits from the back. They meet in the middle, so the
//! whole fixed-size buffer is the packet.
//!
//! Values and layout follow the reference so the output is byte-exact:
//! anything encoded here is meant to decode in any Opus decoder, and the
//! full-frame stage will prove that against one. For now the guarantee is
//! narrower and just as strict — round-trip: what [`RangeEncoder`] writes,
//! [`RangeDecoder`] reads back exactly, symbol for symbol.

/// Bits carried per output byte.
const SYM_BITS: u32 = 8;
/// Largest byte value.
const SYM_MAX: u32 = 0xFF;
/// Bits in the arithmetic window.
const CODE_BITS: u32 = 32;
/// The high bit of the 32-bit code.
const CODE_TOP: u32 = 1 << 31;
/// Renormalise once the range drops to here.
const CODE_BOT: u32 = CODE_TOP >> SYM_BITS;
/// Shift that lifts the top output byte out of `val`.
const CODE_SHIFT: u32 = CODE_BITS - SYM_BITS - 1;
/// Spare low bits the decoder primes its range with.
const CODE_EXTRA: u32 = (CODE_BITS - 2) % SYM_BITS + 1;
/// Bits held in the raw-bit window.
const WINDOW_SIZE: u32 = 32;
/// Split point for `enc_uint`/`dec_uint`: values wider than this carry
/// their low bits as raw bits.
const UINT_BITS: u32 = 8;

/// One more than the position of the highest set bit (0 for 0) — the
/// reference `EC_ILOG`.
#[inline]
fn ilog(x: u32) -> u32 {
    CODE_BITS - x.leading_zeros()
}

/// Writes an Opus range-coded packet into a fixed-size buffer.
///
/// Construct with the exact byte budget the packet must fit
/// ([`RangeEncoder::new`]); range symbols fill from the front and raw
/// bits from the back until [`RangeEncoder::finish`] seals it.
pub struct RangeEncoder {
    buf: Vec<u8>,
    storage: usize,
    offs: usize,
    end_offs: usize,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    rng: u32,
    val: u32,
    rem: i32,
    ext: u32,
    error: bool,
}

impl RangeEncoder {
    /// A coder that must fit `storage` bytes.
    pub fn new(storage: usize) -> Self {
        RangeEncoder {
            buf: vec![0u8; storage],
            storage,
            offs: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1) as i32,
            rng: CODE_TOP,
            val: 0,
            rem: -1,
            ext: 0,
            error: false,
        }
    }

    /// Whether the packet overran its byte budget. A busted packet is not
    /// valid output; the caller must react rather than ship it.
    pub fn is_error(&self) -> bool {
        self.error
    }

    /// Bits committed so far (the reference `ec_tell`).
    pub fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    fn write_byte(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        self.buf[self.offs] = (value & SYM_MAX) as u8;
        self.offs += 1;
    }

    fn write_byte_at_end(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = true;
            return;
        }
        self.end_offs += 1;
        self.buf[self.storage - self.end_offs] = (value & SYM_MAX) as u8;
    }

    fn carry_out(&mut self, c: i32) {
        if c as u32 != SYM_MAX {
            let carry = c >> SYM_BITS;
            if self.rem >= 0 {
                self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                let sym = (SYM_MAX as i32 + carry) as u32 & SYM_MAX;
                while self.ext > 0 {
                    self.write_byte(sym);
                    self.ext -= 1;
                }
            }
            self.rem = c & SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.carry_out((self.val >> CODE_SHIFT) as i32);
            self.val = (self.val << SYM_BITS) & (CODE_TOP - 1);
            self.rng <<= SYM_BITS;
            self.nbits_total += SYM_BITS as i32;
        }
    }

    /// Encode a symbol occupying the cumulative-frequency span
    /// `[fl, fh)` out of a total `ft`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
    }

    /// [`RangeEncoder::encode`] where the total is a power of two, `1<<bits`.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val += self.rng - r * ((1 << bits) - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * ((1 << bits) - fh);
        }
        self.normalize();
    }

    /// Encode one boolean whose `true` probability is `1/(1<<logp)`.
    pub fn enc_bit_logp(&mut self, value: bool, logp: u32) {
        let r = self.rng;
        let l = self.val;
        let s = r >> logp;
        let r = r - s;
        if value {
            self.val = l + r;
        }
        self.rng = if value { s } else { r };
        self.normalize();
    }

    /// Encode symbol `s` against an inverse-cdf table scaled to `1<<ftb`.
    pub fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val += self.rng - r * u32::from(icdf[s - 1]);
            self.rng = r * u32::from(icdf[s - 1] - icdf[s]);
        } else {
            self.rng -= r * u32::from(icdf[s]);
        }
        self.normalize();
    }

    /// Encode a uniformly distributed integer `fl` in `[0, ft)`.
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        let ftm = ft - 1;
        let ftb = ilog(ftm);
        if ftb > UINT_BITS {
            let ftb = ftb - UINT_BITS;
            let ft2 = (ftm >> ftb) + 1;
            let fl2 = fl >> ftb;
            self.encode(fl2, fl2 + 1, ft2);
            self.enc_bits(fl & ((1 << ftb) - 1), ftb);
        } else {
            self.encode(fl, fl + 1, ftm + 1);
        }
    }

    /// Encode `bits` raw (uniform) bits, packed from the back.
    pub fn enc_bits(&mut self, val: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used as u32 + bits > WINDOW_SIZE {
            while used >= SYM_BITS as i32 {
                self.write_byte_at_end(window & SYM_MAX);
                window >>= SYM_BITS;
                used -= SYM_BITS as i32;
            }
        }
        window |= val << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    /// Seal the packet and return its bytes. After this the coder is spent.
    pub fn finish(mut self) -> Vec<u8> {
        let l0 = CODE_BITS - ilog(self.rng);
        let mut l = l0 as i32;
        let mut msk = (CODE_TOP - 1) >> l0;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> CODE_SHIFT) as i32);
            end = (end << SYM_BITS) & (CODE_TOP - 1);
            l -= SYM_BITS as i32;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= SYM_BITS as i32 {
            self.write_byte_at_end(window & SYM_MAX);
            window >>= SYM_BITS;
            used -= SYM_BITS as i32;
        }
        if !self.error {
            for b in self.buf[self.offs..self.storage - self.end_offs].iter_mut() {
                *b = 0;
            }
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = true;
                } else {
                    self.buf[self.storage - self.end_offs - 1] |= (window & SYM_MAX) as u8;
                }
            }
        }
        self.buf
    }
}

/// Reads an Opus range-coded packet back.
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    storage: usize,
    offs: usize,
    end_offs: usize,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    rng: u32,
    val: u32,
    rem: i32,
    ext: u32,
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Start reading `buf`, a complete range-coded packet.
    pub fn new(buf: &'a [u8]) -> Self {
        let storage = buf.len();
        let mut d = RangeDecoder {
            buf,
            storage,
            offs: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (CODE_BITS + 1 - (CODE_BITS - CODE_EXTRA) / SYM_BITS * SYM_BITS) as i32,
            rng: 1 << CODE_EXTRA,
            val: 0,
            rem: 0,
            ext: 0,
            error: false,
        };
        d.rem = d.read_byte() as i32;
        d.val = d.rng - 1 - (d.rem as u32 >> (SYM_BITS - CODE_EXTRA));
        d.normalize();
        d
    }

    /// Whether the packet was malformed (a value ran past its total).
    pub fn is_error(&self) -> bool {
        self.error
    }

    fn read_byte(&mut self) -> u32 {
        if self.offs < self.storage {
            let b = u32::from(self.buf[self.offs]);
            self.offs += 1;
            b
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.storage {
            self.end_offs += 1;
            u32::from(self.buf[self.storage - self.end_offs])
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.rng <= CODE_BOT {
            self.nbits_total += SYM_BITS as i32;
            self.rng <<= SYM_BITS;
            let sym = self.rem;
            self.rem = self.read_byte() as i32;
            let sym = ((sym << SYM_BITS | self.rem) as u32) >> (SYM_BITS - CODE_EXTRA);
            self.val = ((self.val << SYM_BITS) + (SYM_MAX & !sym)) & (CODE_TOP - 1);
        }
    }

    /// The cumulative frequency of the next symbol out of `ft`. Follow it
    /// with [`RangeDecoder::dec_update`] once the span is known.
    pub fn decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// [`RangeDecoder::decode`] where the total is `1<<bits`.
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        (1 << bits) - (s + 1).min(1 << bits)
    }

    /// Commit the span `[fl, fh)` out of `ft` chosen after [`RangeDecoder::decode`].
    pub fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext * (ft - fh);
        self.val -= s;
        self.rng = if fl > 0 {
            self.ext * (fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    /// Decode one boolean whose `true` probability is `1/(1<<logp)`.
    pub fn dec_bit_logp(&mut self, logp: u32) -> bool {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if !ret {
            self.val = d - s;
        }
        self.rng = if ret { s } else { r - s };
        self.normalize();
        ret
    }

    /// Decode a symbol against an inverse-cdf table scaled to `1<<ftb`.
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> usize {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret: isize = -1;
        let mut t;
        loop {
            t = s;
            ret += 1;
            s = r * u32::from(icdf[ret as usize]);
            if d >= s {
                break;
            }
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret as usize
    }

    /// Decode a uniformly distributed integer in `[0, ft)`.
    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        let ftm = ft - 1;
        let ftb = ilog(ftm);
        if ftb > UINT_BITS {
            let ftb = ftb - UINT_BITS;
            let ft2 = (ftm >> ftb) + 1;
            let s = self.decode(ft2);
            self.dec_update(s, s + 1, ft2);
            let t = (s << ftb) | self.dec_bits(ftb);
            if t <= ftm {
                t
            } else {
                self.error = true;
                ftm
            }
        } else {
            let s = self.decode(ftm + 1);
            self.dec_update(s, s + 1, ftm + 1);
            s
        }
    }

    /// Decode `bits` raw (uniform) bits from the back.
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            while available <= (WINDOW_SIZE - SYM_BITS) as i32 {
                window |= self.read_byte_from_end() << available;
                available += SYM_BITS as i32;
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        window >>= bits;
        available -= bits as i32;
        self.end_window = window;
        self.nend_bits = available;
        self.nbits_total -= bits as i32;
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A three-symbol alphabet with a lopsided distribution, as an icdf.
    // icdf[i] = (1<<ftb) - cumfreq(i+1); the last entry is 0.
    const FTB: u32 = 8;
    // icdf[i] = (1<<FTB) - cumfreq(i+1), last entry 0; freqs 200, 50, 6.
    const ICDF: [u8; 3] = [56, 6, 0];

    #[test]
    fn range_symbols_round_trip_exactly() {
        let symbols = [0usize, 1, 2, 0, 0, 2, 1, 0, 2, 2, 1, 0, 0, 0, 1, 2];
        let mut enc = RangeEncoder::new(256);
        for &s in &symbols {
            enc.enc_icdf(s, &ICDF, FTB);
        }
        assert!(!enc.is_error());
        let packet = enc.finish();

        let mut dec = RangeDecoder::new(&packet);
        for &want in &symbols {
            assert_eq!(dec.dec_icdf(&ICDF, FTB), want);
        }
        assert!(!dec.is_error());
    }

    #[test]
    fn raw_bits_and_range_symbols_interleave_and_recover() {
        // Raw bits live at the back, range symbols at the front; a real
        // frame interleaves them, so the round trip must too.
        let mut enc = RangeEncoder::new(512);
        enc.enc_icdf(1, &ICDF, FTB);
        enc.enc_bits(0b1011, 4);
        enc.enc_icdf(2, &ICDF, FTB);
        enc.enc_bits(0x1_2345, 20);
        enc.enc_bit_logp(true, 3);
        enc.enc_icdf(0, &ICDF, FTB);
        let packet = enc.finish();

        let mut dec = RangeDecoder::new(&packet);
        assert_eq!(dec.dec_icdf(&ICDF, FTB), 1);
        assert_eq!(dec.dec_bits(4), 0b1011);
        assert_eq!(dec.dec_icdf(&ICDF, FTB), 2);
        assert_eq!(dec.dec_bits(20), 0x1_2345);
        assert!(dec.dec_bit_logp(3));
        assert_eq!(dec.dec_icdf(&ICDF, FTB), 0);
    }

    #[test]
    fn uints_round_trip_across_the_split() {
        // Values below and above UINT_BITS take different paths; both must
        // recover.
        let values: [(u32, u32); 6] = [
            (0, 2),
            (1, 2),
            (37, 100),
            (255, 256),
            (100_000, 1_000_000),
            (0xFFFF, 0x0001_0000),
        ];
        let mut enc = RangeEncoder::new(256);
        for &(v, ft) in &values {
            enc.enc_uint(v, ft);
        }
        let packet = enc.finish();

        let mut dec = RangeDecoder::new(&packet);
        for &(v, ft) in &values {
            assert_eq!(dec.dec_uint(ft), v);
        }
    }

    #[test]
    fn bit_accounting_advances_with_each_symbol() {
        let mut enc = RangeEncoder::new(64);
        let start = enc.tell();
        assert_eq!(start, 1, "a fresh coder has told exactly one bit");
        enc.enc_bits(0, 10);
        assert_eq!(enc.tell(), start + 10, "raw bits cost exactly their width");
    }
}
