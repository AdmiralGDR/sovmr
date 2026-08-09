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

//! Scalefactor bands: where AAC divides the spectrum.
//!
//! A frame's 1024 coefficients are grouped into bands, each carrying one
//! scalefactor and one codebook choice. The boundaries are not
//! calculated — they are a fixed table per sampling rate, chosen to
//! follow the ear's critical bands, and every encoder and decoder must
//! agree on them exactly.
//!
//! # Where these numbers came from
//!
//! Every rate the format can name in four bits — all thirteen of them,
//! seven distinct tables. Not from memory, and not copied. They were
//! **measured**, by
//! `sovmr-audio/tests/oracle.rs`, which builds a frame declaring every
//! band below the one of interest empty, puts a single coefficient in
//! that band, and asks a decoder where it landed. Each reading came back
//! as one coefficient tens of millions of times above everything else in
//! the spectrum, so there is no interpretation involved.
//!
//! The width of a band came out of the same probe: a band is filled by a
//! whole number of codewords, and the number needed is its width divided
//! by four.
//!
//! That gate re-derives the whole table and compares it against what is
//! written here, so these numbers cannot rot silently.
//!
//! # The check that makes them trustworthy
//!
//! The widths must add up to exactly 1024. Nothing enforces that during
//! the measurement — each band is probed on its own, and a single wrong
//! reading anywhere would leave the total short or long. Every one of
//! the thirteen rates comes to 1024, which is a strong statement about
//! all of them at once.
//!
//! Some rates turned out to share a table: 48 kHz with 44.1, 24 with
//! 22.05, 96 with 88.2, 16 with 12 and 11.025, and 8 with 7.35. Nothing
//! required that, and it was not assumed — each was measured separately
//! and the tables compared afterwards.

/// Coefficients in a long frame — what every band table must total.
pub const LONG_FRAME: usize = 1024;

/// Scalefactor band boundaries for 96000 Hz and 88200 Hz: 41 bands.
///
/// These rates share one table — measured separately and found
/// identical, which is worth stating because nothing required it.
///
/// Band `i` covers `LONG_96000_88200[i]..LONG_96000_88200[i + 1]`.
pub const LONG_96000_88200: [usize; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// Scalefactor band boundaries for 64000 Hz: 47 bands.
///
/// Band `i` covers `LONG_64000[i]..LONG_64000[i + 1]`.
pub const LONG_64000: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

/// Scalefactor band boundaries for 48000 Hz and 44100 Hz: 49 bands.
///
/// These rates share one table — measured separately and found
/// identical, which is worth stating because nothing required it.
///
/// Band `i` covers `LONG_48000_44100[i]..LONG_48000_44100[i + 1]`.
pub const LONG_48000_44100: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// Scalefactor band boundaries for 32000 Hz: 51 bands.
///
/// Band `i` covers `LONG_32000[i]..LONG_32000[i + 1]`.
pub const LONG_32000: [usize; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// Scalefactor band boundaries for 24000 Hz and 22050 Hz: 47 bands.
///
/// These rates share one table — measured separately and found
/// identical, which is worth stating because nothing required it.
///
/// Band `i` covers `LONG_24000_22050[i]..LONG_24000_22050[i + 1]`.
pub const LONG_24000_22050: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// Scalefactor band boundaries for 16000 Hz and 12000 Hz and 11025 Hz: 43 bands.
///
/// These rates share one table — measured separately and found
/// identical, which is worth stating because nothing required it.
///
/// Band `i` covers `LONG_16000_12000_11025[i]..LONG_16000_12000_11025[i + 1]`.
pub const LONG_16000_12000_11025: [usize; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

/// Scalefactor band boundaries for 8000 Hz and 7350 Hz: 40 bands.
///
/// These rates share one table — measured separately and found
/// identical, which is worth stating because nothing required it.
///
/// Band `i` covers `LONG_8000_7350[i]..LONG_8000_7350[i + 1]`.
pub const LONG_8000_7350: [usize; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

// ---------------------------------------------------------------------
// Short windows.
//
// A frame that would smear an attack is coded as eight short blocks of
// 128 coefficients instead of one long block of 1024, and those 128 have
// scalefactor bands of their own — a different table, at every rate.
//
// Walked out of a decoder the same way as the long ones and by the same
// gate: every band below the one of interest declared empty, the band
// itself filled with codebook-1 codewords, and the count that fits taken
// as its width. Too few leaves the decoder mid-band, too many makes it
// read the terminator as data, so the fit is exact from both sides.
//
// Every one of the thirteen rates totals exactly 128. Nothing enforced
// that during the walk — each band was probed on its own — which is what
// makes the total a statement about all of them at once.
//
// **The rates group differently from the long tables**, and that is
// worth stating because reusing the long grouping is the obvious
// mistake: 64 kHz has a long table of its own but shares the short one
// with 96 and 88.2; 32 kHz likewise has its own long table but shares
// the short one with 48 and 44.1. Five distinct short tables against the
// long side's seven.
// ---------------------------------------------------------------------

/// Coefficients in one short window — what every short band table must
/// total.
pub const SHORT_FRAME: usize = 128;
/// Short windows in an eight-short frame. Eight of them cover the same
/// span of time as one long window.
pub const SHORT_WINDOWS: usize = 8;

/// Short-window bands for 96000 Hz, 88200 Hz and 64000 Hz: 12 bands.
pub const SHORT_96000_88200_64000: [usize; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

/// Short-window bands for 48000 Hz, 44100 Hz and 32000 Hz: 14 bands.
pub const SHORT_48000_44100_32000: [usize; 15] =
    [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

/// Short-window bands for 24000 Hz and 22050 Hz: 15 bands.
pub const SHORT_24000_22050: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

/// Short-window bands for 16000 Hz, 12000 Hz and 11025 Hz: 15 bands.
pub const SHORT_16000_12000_11025: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

/// Short-window bands for 8000 Hz and 7350 Hz: 15 bands.
pub const SHORT_8000_7350: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

/// The short-window band table for a sampling rate, or `None` for one
/// not measured.
///
/// Separate from [`long_bands`] on purpose: the two are different tables
/// with a different grouping of rates, and a reader that reached for the
/// long one because it had the rate to hand would place every band of a
/// short frame somewhere else.
pub fn short_bands(sample_rate: u32) -> Option<&'static [usize]> {
    match sample_rate {
        96_000 | 88_200 | 64_000 => Some(&SHORT_96000_88200_64000),
        48_000 | 44_100 | 32_000 => Some(&SHORT_48000_44100_32000),
        24_000 | 22_050 => Some(&SHORT_24000_22050),
        16_000 | 12_000 | 11_025 => Some(&SHORT_16000_12000_11025),
        8_000 | 7_350 => Some(&SHORT_8000_7350),
        _ => None,
    }
}

/// Every rate this project has measured a band table for.
///
/// All thirteen the format can name in four bits. Each was measured on
/// its own, and each totals exactly 1024 coefficients.
pub const MEASURED_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// The band table for a sampling rate, or `None` for one not measured.
///
/// Every rate has its own table and they are not related by any formula.
/// Returning `None` rather than the nearest one is the whole point: a
/// stream encoded against the wrong boundaries is not slightly wrong, it
/// is unlistenable, and it would announce itself as fine.
pub fn long_bands(sample_rate: u32) -> Option<&'static [usize]> {
    match sample_rate {
        96_000 => Some(&LONG_96000_88200),
        88_200 => Some(&LONG_96000_88200),
        64_000 => Some(&LONG_64000),
        48_000 => Some(&LONG_48000_44100),
        44_100 => Some(&LONG_48000_44100),
        32_000 => Some(&LONG_32000),
        24_000 => Some(&LONG_24000_22050),
        22_050 => Some(&LONG_24000_22050),
        16_000 => Some(&LONG_16000_12000_11025),
        12_000 => Some(&LONG_16000_12000_11025),
        11_025 => Some(&LONG_16000_12000_11025),
        8_000 => Some(&LONG_8000_7350),
        7_350 => Some(&LONG_8000_7350),
        _ => None,
    }
}

/// The band a coefficient falls in, or `None` past the end.
pub fn band_of(bands: &[usize], coefficient: usize) -> Option<usize> {
    bands
        .windows(2)
        .position(|pair| coefficient >= pair[0] && coefficient < pair[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_measured_rate_covers_a_short_window_exactly() {
        // The same check the long tables get, and it earns its keep the
        // same way: each short band was probed on its own, so the total
        // is the one statement about all of them together.
        for rate in MEASURED_RATES {
            let bands = short_bands(rate).unwrap_or_else(|| panic!("{rate} is listed as measured"));
            assert_eq!(bands[0], 0, "{rate}: the first band starts at zero");
            assert_eq!(
                *bands.last().expect("not empty"),
                SHORT_FRAME,
                "{rate}: the bands must end where a short window does"
            );
            let total: usize = bands.windows(2).map(|p| p[1] - p[0]).sum();
            assert_eq!(total, SHORT_FRAME, "{rate}: widths total {total}");
            // Every width is a whole number of codebook-1 codewords,
            // which is how the walk could count them at all.
            for pair in bands.windows(2) {
                assert!(pair[1] > pair[0], "{rate}: a band of no width");
                assert_eq!((pair[1] - pair[0]) % 4, 0, "{rate}: a width off the grid");
            }
        }
    }

    #[test]
    fn eight_short_windows_hold_what_one_long_one_does() {
        // Not a coincidence worth relying on blindly, but a real
        // constraint: the two layouts describe the same span of time, so
        // a frame is the same number of coefficients either way.
        assert_eq!(SHORT_FRAME * SHORT_WINDOWS, LONG_FRAME);
    }

    #[test]
    fn the_short_tables_group_the_rates_differently_from_the_long_ones() {
        // The trap this records. Reaching for the long table because the
        // rate is to hand would place every band of a short frame
        // somewhere else — and 64 kHz and 32 kHz are exactly where that
        // would happen without any other symptom.
        assert_eq!(short_bands(64_000), short_bands(96_000));
        assert_ne!(long_bands(64_000), long_bands(96_000));
        assert_eq!(short_bands(32_000), short_bands(48_000));
        assert_ne!(long_bands(32_000), long_bands(48_000));
        let distinct: std::collections::BTreeSet<&[usize]> = MEASURED_RATES
            .iter()
            .filter_map(|r| short_bands(*r))
            .collect();
        assert_eq!(distinct.len(), 5, "five distinct short tables");
    }

    #[test]
    fn a_rate_without_a_short_table_is_refused() {
        // The same refusal as the long side, for the same reason: a
        // stream read against the wrong boundaries is not slightly
        // wrong, it is unlistenable, and it announces itself as fine.
        assert_eq!(short_bands(47_999), None);
        assert_eq!(short_bands(0), None);
    }

    #[test]
    fn every_measured_rate_covers_the_frame_exactly() {
        // The check no measurement could make for itself. Each band was
        // probed alone; that they add up to the frame length is a
        // statement about all of them together, and it has to hold for
        // every rate independently.
        for rate in MEASURED_RATES {
            let bands = long_bands(rate).unwrap_or_else(|| panic!("{rate} is listed as measured"));
            assert_eq!(bands[0], 0, "{rate}: the first band starts at zero");
            assert_eq!(
                *bands.last().expect("not empty"),
                LONG_FRAME,
                "{rate}: the bands must end where the frame does"
            );
            let total: usize = bands.windows(2).map(|p| p[1] - p[0]).sum();
            assert_eq!(total, LONG_FRAME, "{rate}: widths total {total}");
        }
    }

    #[test]
    fn every_band_is_a_whole_number_of_codewords_wide() {
        // A band is coded by whole codewords, and the widest books take
        // four coefficients at a time. A width not divisible by four
        // could not be filled — a property of the format rather than of
        // these tables, so worth checking the tables have it.
        for rate in MEASURED_RATES {
            let bands = long_bands(rate).expect("measured");
            for (index, pair) in bands.windows(2).enumerate() {
                let width = pair[1] - pair[0];
                assert!(width > 0, "{rate}: band {index} is empty");
                assert_eq!(width % 4, 0, "{rate}: band {index} is {width} wide");
            }
        }
    }

    #[test]
    fn the_bands_only_ever_widen() {
        // The tables follow the ear, which resolves pitch finely at the
        // bottom and coarsely at the top. A band narrower than the one
        // below it would be a transcription slip, and this is the shape
        // that catches one.
        for rate in MEASURED_RATES {
            let bands = long_bands(rate).expect("measured");
            let widths: Vec<usize> = bands.windows(2).map(|p| p[1] - p[0]).collect();
            for (index, pair) in widths.windows(2).enumerate() {
                assert!(
                    pair[1] >= pair[0],
                    "{rate}: band {} is {} wide but band {index} is {}",
                    index + 1,
                    pair[1],
                    pair[0]
                );
            }
        }
    }

    #[test]
    fn a_rate_the_format_cannot_name_is_refused_rather_than_approximated() {
        // Every rate has its own boundaries and no formula relates them.
        // Handing back a neighbour's table would produce a stream that
        // claims to be fine and is not.
        for rate in MEASURED_RATES {
            assert!(long_bands(rate).is_some(), "{rate}");
        }
        for rate in [0u32, 1, 47_999, 48_001, 50_000, 192_000] {
            assert!(long_bands(rate).is_none(), "{rate} was never measured");
        }
    }

    #[test]
    fn the_measured_rates_are_the_ones_a_config_can_name() {
        // The two lists have to agree. A rate an `AudioSpecificConfig`
        // can name but no table covers would be encodable in principle
        // and not in practice; the reverse would be a table nothing can
        // ask for.
        let mut measured = MEASURED_RATES;
        measured.sort_unstable();
        let mut nameable = crate::asc::SAMPLE_RATES;
        nameable.sort_unstable();
        assert_eq!(
            measured.as_slice(),
            nameable.as_slice(),
            "the band tables and the config's rate list have drifted apart"
        );
    }

    #[test]
    fn a_coefficient_falls_in_exactly_one_band() {
        for rate in MEASURED_RATES {
            let bands = long_bands(rate).expect("measured");
            for coefficient in 0..LONG_FRAME {
                let band = band_of(bands, coefficient)
                    .unwrap_or_else(|| panic!("{rate}: coefficient {coefficient} is in no band"));
                assert!(coefficient >= bands[band] && coefficient < bands[band + 1]);
            }
            assert_eq!(band_of(bands, LONG_FRAME), None, "{rate}: past the end");
        }
    }
}
