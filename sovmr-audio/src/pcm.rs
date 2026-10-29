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

//! Uncompressed audio, in the several shapes containers store it.
//!
//! There is no cleverness here and there must not be: PCM is bytes in a
//! declared layout, and a decoder that got creative would corrupt
//! material that arrived perfect. Every conversion below is exact.
//!
//! # Why samples come back as `i32`
//!
//! The rest of this crate works on whole-numbered samples at a declared
//! depth. Floats would be convenient for the encoder and would make
//! bit-exactness untestable — `0.1f32` is not a value you can assert on.
//! An integer sample either matches the source or does not, and for
//! lossless material that is the only interesting question.

/// How a container says its uncompressed samples are laid out.
///
/// The names are the fourcc or codec id the source used, because that is
/// what a caller has in hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    /// Signed 16-bit, little-endian (`sowt`, `A_PCM/INT/LIT`).
    S16Le,
    /// Signed 16-bit, big-endian (`twos`).
    S16Be,
    /// Signed 24-bit, big-endian (`in24`).
    S24Be,
    /// Signed 24-bit, little-endian.
    S24Le,
    /// Signed 32-bit, big-endian (`in32`).
    S32Be,
    /// Signed 32-bit, little-endian.
    S32Le,
    /// IEEE 754 single precision, big-endian (`fl32`).
    F32Be,
    /// IEEE 754 single precision, little-endian (`A_PCM/FLOAT/IEEE`).
    F32Le,
    /// Unsigned 8-bit, offset by 128 — the one format that is not
    /// two's complement, and the one most often got wrong.
    U8,
}

impl PcmFormat {
    /// Bytes one sample of one channel occupies.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PcmFormat::U8 => 1,
            PcmFormat::S16Le | PcmFormat::S16Be => 2,
            PcmFormat::S24Be | PcmFormat::S24Le => 3,
            PcmFormat::S32Be | PcmFormat::S32Le | PcmFormat::F32Be | PcmFormat::F32Le => 4,
        }
    }

    /// Significant bits per sample, as the source stores them.
    ///
    /// Floats report 32 because that is their storage width; what they
    /// actually resolve is a different question, and not one a container
    /// asks.
    pub fn bit_depth(self) -> u16 {
        match self {
            PcmFormat::U8 => 8,
            PcmFormat::S16Le | PcmFormat::S16Be => 16,
            PcmFormat::S24Be | PcmFormat::S24Le => 24,
            PcmFormat::S32Be | PcmFormat::S32Le | PcmFormat::F32Be | PcmFormat::F32Le => 32,
        }
    }

    /// Recognise an ISOBMFF sample-entry fourcc.
    ///
    /// Returns `None` for anything not uncompressed — the caller reports
    /// that rather than guessing at a layout.
    pub fn from_fourcc(fourcc: &[u8; 4]) -> Option<PcmFormat> {
        match fourcc {
            b"sowt" => Some(PcmFormat::S16Le),
            b"twos" => Some(PcmFormat::S16Be),
            b"in24" => Some(PcmFormat::S24Be),
            b"in32" => Some(PcmFormat::S32Be),
            b"fl32" => Some(PcmFormat::F32Be),
            // `lpcm` alone does not say: its layout lives in flags the
            // sample entry carries separately, so a caller that has only
            // the fourcc genuinely does not know yet.
            _ => None,
        }
    }

    /// Recognise a Matroska codec id.
    pub fn from_codec_id(codec_id: &str) -> Option<PcmFormat> {
        match codec_id {
            "A_PCM/INT/LIT" => Some(PcmFormat::S16Le),
            "A_PCM/INT/BIG" => Some(PcmFormat::S16Be),
            "A_PCM/FLOAT/IEEE" => Some(PcmFormat::F32Le),
            _ => None,
        }
    }
}

/// Scale a float sample to the 32-bit integer range.
///
/// Clamped, not wrapped: a float stream may legitimately exceed ±1.0
/// after mixing, and wrapping would turn a loud passage into the
/// opposite polarity — an unmistakable crack rather than a clipped peak.
fn float_to_i32(v: f32) -> i32 {
    let scaled = f64::from(v) * f64::from(i32::MAX);
    scaled.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

/// Decode interleaved PCM into whole-numbered samples.
///
/// Samples come back left-aligned in `i32` — that is, scaled up to the
/// full 32-bit range — so a caller can mix depths without asking each
/// one how wide it was. `format.bit_depth()` still says what the source
/// actually resolved.
///
/// A trailing partial sample is ignored rather than padded: inventing
/// the missing bytes would put a sample in the output that the source
/// never contained.
///
/// # Example
///
/// ```
/// use sovmr_audio::pcm::{decode, PcmFormat};
/// // Full-scale positive and negative, 16-bit little-endian.
/// let out = decode(&[0xFF, 0x7F, 0x00, 0x80], PcmFormat::S16Le);
/// assert_eq!(out.len(), 2);
/// assert!(out[0] > 0 && out[1] < 0);
/// ```
pub fn decode(data: &[u8], format: PcmFormat) -> Vec<i32> {
    let width = format.bytes_per_sample();
    let count = data.len() / width;
    let mut out = Vec::with_capacity(count);
    for chunk in data.chunks_exact(width) {
        out.push(match format {
            // 8-bit PCM is *unsigned*, centred on 128. Reading it as
            // two's complement inverts the waveform, which sounds fine
            // in isolation and wrong the moment it meets another track.
            PcmFormat::U8 => (i32::from(chunk[0]) - 128) << 24,
            PcmFormat::S16Le => i32::from(i16::from_le_bytes([chunk[0], chunk[1]])) << 16,
            PcmFormat::S16Be => i32::from(i16::from_be_bytes([chunk[0], chunk[1]])) << 16,
            // 24-bit has no native type: assemble it in the top three
            // bytes, which sign-extends by construction.
            PcmFormat::S24Be => i32::from_be_bytes([chunk[0], chunk[1], chunk[2], 0]),
            PcmFormat::S24Le => i32::from_be_bytes([chunk[2], chunk[1], chunk[0], 0]),
            PcmFormat::S32Be => i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            PcmFormat::S32Le => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            PcmFormat::F32Be => {
                float_to_i32(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
            PcmFormat::F32Le => {
                float_to_i32(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endianness_is_not_guessed() {
        // The same four bytes are two different waveforms depending on
        // which end the container meant. Getting this wrong produces
        // noise, which at least is obvious; getting it wrong on only one
        // format produces a file that is fine until track three.
        let bytes = [0x00, 0x01, 0x00, 0x02];
        let le = decode(&bytes, PcmFormat::S16Le);
        let be = decode(&bytes, PcmFormat::S16Be);
        assert_eq!(le, vec![0x0100 << 16, 0x0200 << 16]);
        assert_eq!(be, vec![0x0001 << 16, 0x0002 << 16]);
    }

    #[test]
    fn eight_bit_is_unsigned_and_centred_on_128() {
        // The one PCM format that is not two's complement. Read as
        // signed, silence (128) would become a full-scale offset.
        let out = decode(&[128, 0, 255], PcmFormat::U8);
        assert_eq!(out[0], 0, "128 is silence");
        assert_eq!(out[1], -128 << 24, "0 is the negative extreme");
        assert_eq!(out[2], 127 << 24, "255 is the positive extreme");
    }

    #[test]
    fn twenty_four_bit_sign_extends_in_both_endiannesses() {
        // 0xFFFFFF is -1 in 24-bit. Assembling it into the low three
        // bytes would read it as +16777215.
        let be = decode(&[0xFF, 0xFF, 0xFF], PcmFormat::S24Be);
        let le = decode(&[0xFF, 0xFF, 0xFF], PcmFormat::S24Le);
        assert!(be[0] < 0 && le[0] < 0, "got {be:?} / {le:?}");
        // And a positive value stays positive with its magnitude intact.
        assert_eq!(decode(&[0x00, 0x00, 0x01], PcmFormat::S24Be)[0], 1 << 8);
        assert_eq!(decode(&[0x01, 0x00, 0x00], PcmFormat::S24Le)[0], 1 << 8);
    }

    #[test]
    fn every_depth_lands_on_the_same_scale() {
        // Full-scale positive in each width must come back as roughly
        // the same number, or mixing tracks of different depths would
        // change their relative loudness.
        let s16 = decode(&0x7FFFi16.to_le_bytes(), PcmFormat::S16Le)[0];
        let s24 = decode(&[0x7F, 0xFF, 0xFF], PcmFormat::S24Be)[0];
        let s32 = decode(&0x7FFF_FFFFi32.to_le_bytes(), PcmFormat::S32Le)[0];
        for v in [s16, s24, s32] {
            assert!(v > i32::MAX - (1 << 17), "full scale drifted: {v}");
        }
    }

    #[test]
    fn floats_clamp_instead_of_wrapping() {
        // A mixed float stream can exceed ±1.0. Wrapping would flip the
        // polarity of the loudest moment — a crack, not a clipped peak.
        let hot = decode(&2.0f32.to_le_bytes(), PcmFormat::F32Le);
        let cold = decode(&(-2.0f32).to_le_bytes(), PcmFormat::F32Le);
        assert_eq!(hot[0], i32::MAX);
        assert_eq!(cold[0], i32::MIN);
        // And an ordinary value keeps its sign and rough magnitude.
        let half = decode(&0.5f32.to_le_bytes(), PcmFormat::F32Le)[0];
        assert!(half > i32::MAX / 3 && half < i32::MAX, "got {half}");
    }

    #[test]
    fn a_trailing_partial_sample_is_dropped_not_padded() {
        // Padding would put a sample in the output the source never
        // held — and for lossless material that is the whole failure.
        let out = decode(&[0x00, 0x01, 0xFF], PcmFormat::S16Le);
        assert_eq!(out.len(), 1);
        assert!(decode(&[0xFF], PcmFormat::S32Le).is_empty());
        assert!(decode(&[], PcmFormat::S16Le).is_empty());
    }

    #[test]
    fn container_names_map_to_layouts_or_to_nothing() {
        assert_eq!(PcmFormat::from_fourcc(b"sowt"), Some(PcmFormat::S16Le));
        assert_eq!(PcmFormat::from_fourcc(b"twos"), Some(PcmFormat::S16Be));
        assert_eq!(PcmFormat::from_fourcc(b"in24"), Some(PcmFormat::S24Be));
        assert_eq!(
            PcmFormat::from_codec_id("A_PCM/FLOAT/IEEE"),
            Some(PcmFormat::F32Le)
        );
        // `lpcm` carries its layout in flags elsewhere, so the fourcc
        // alone genuinely does not say — and guessing would be worse.
        assert_eq!(PcmFormat::from_fourcc(b"lpcm"), None);
        assert_eq!(PcmFormat::from_fourcc(b"mp4a"), None);
        assert_eq!(PcmFormat::from_codec_id("A_AAC"), None);
    }

    #[test]
    fn widths_and_depths_agree_with_each_other() {
        for f in [
            PcmFormat::U8,
            PcmFormat::S16Le,
            PcmFormat::S24Be,
            PcmFormat::S32Le,
            PcmFormat::F32Le,
        ] {
            assert_eq!(f.bytes_per_sample() * 8, usize::from(f.bit_depth()));
        }
    }
}
