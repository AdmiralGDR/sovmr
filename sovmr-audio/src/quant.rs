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

//! Quantisation: where the bits are actually thrown away.
//!
//! Everything before this is exact. The transform is reversible, the
//! decoders are lossless — this is the step that decides what the file
//! will and will not remember, and it is the only place in the audio
//! path where "close enough" is the correct answer.
//!
//! # Why the law is a power law, not a straight line
//!
//! Rounding a coefficient to a fixed step spends the same precision on a
//! whisper as on a cymbal, and the ear does not work that way: what it
//! notices is *proportional* error. AAC therefore raises magnitudes to
//! the power ¾ before rounding, which spaces the levels further apart as
//! they grow — so the error stays a roughly constant fraction, and far
//! fewer distinct values need coding.
//!
//! That holds only where there are levels to spare. Down at a quantised
//! magnitude of six the neighbouring levels are 8.55 and 10.90, so a
//! coefficient of 10 cannot come back closer than about 9% however
//! carefully it is rounded. The floor is real and it is measured in the
//! tests rather than glossed over — and it is exactly what the
//! scalefactor exists to escape.
//!
//! # The scalefactor
//!
//! One number per band scales it before that rounding. Raising it by
//! four halves the precision of the band and roughly halves what it
//! costs to store. That is the dial an encoder turns to hit a bitrate,
//! and choosing it well is what separates a good encoder from a correct
//! one — this module is the correct one.

/// The value a scalefactor is measured from.
///
/// The format stores scalefactors as small unsigned numbers, so the
/// point of unity gain sits in the middle of the range rather than at
/// zero — a band quieter than the reference gets a scalefactor below
/// this, a louder one above.
pub const SF_OFFSET: i32 = 100;

/// Largest magnitude a quantised coefficient may take.
///
/// The Huffman codebooks that will carry these values cannot express
/// more, so a coefficient beyond it is not a loud sample but an
/// unencodable one: the scalefactor has to come down instead.
pub const MAX_QUANT: i32 = 8191;

/// Rounding bias applied before truncation.
///
/// Truncating alone would pull every coefficient toward zero and drain
/// the signal; a half would round symmetrically but is wrong for a power
/// law, whose steps are not evenly spaced. This is the value the format's
/// reference encoders settled on.
const ROUNDING_BIAS: f64 = 0.4054;

/// The compression exponent, and its inverse for the way back.
const COMPRESS: f64 = 0.75;
const EXPAND: f64 = 4.0 / 3.0;

/// The gain a scalefactor asks for.
fn gain(scalefactor: i32) -> f64 {
    // A step of four in the scalefactor is a factor of two in amplitude,
    // which is what makes the units convenient to reason about.
    2f64.powf(f64::from(scalefactor - SF_OFFSET) / 4.0)
}

/// Quantise one coefficient.
///
/// The magnitude is scaled down by the band's gain, compressed, rounded,
/// and clamped. The sign is carried through untouched — a sign lost here
/// is not a quiet error but a sample pointing the wrong way.
pub fn quantise_one(coefficient: f64, scalefactor: i32) -> i32 {
    if coefficient == 0.0 || !coefficient.is_finite() {
        return 0;
    }
    let magnitude = (coefficient.abs() / gain(scalefactor)).powf(COMPRESS) + ROUNDING_BIAS;
    let q = magnitude as i64;
    let clamped = q.min(i64::from(MAX_QUANT)) as i32;
    if coefficient < 0.0 {
        -clamped
    } else {
        clamped
    }
}

/// Reconstruct one coefficient from its quantised value.
///
/// This is what a decoder does, and having it here is not duplication:
/// an encoder that cannot see what its own decision sounds like is
/// choosing scalefactors blind.
pub fn dequantise_one(quantised: i32, scalefactor: i32) -> f64 {
    if quantised == 0 {
        return 0.0;
    }
    let magnitude = f64::from(quantised.unsigned_abs()).powf(EXPAND) * gain(scalefactor);
    if quantised < 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Quantise a band of coefficients with one scalefactor.
pub fn quantise(coefficients: &[f64], scalefactor: i32, out: &mut Vec<i32>) {
    out.clear();
    out.extend(coefficients.iter().map(|c| quantise_one(*c, scalefactor)));
}

/// Reconstruct a band.
pub fn dequantise(quantised: &[i32], scalefactor: i32, out: &mut Vec<f64>) {
    out.clear();
    out.extend(quantised.iter().map(|q| dequantise_one(*q, scalefactor)));
}

/// The smallest scalefactor that keeps every coefficient in a band
/// representable.
///
/// A band louder than [`MAX_QUANT`] can express does not clip in the
/// pleasant analogue sense — it is simply unencodable, and the only
/// remedy is a coarser scalefactor. Finding the boundary here means the
/// rate-control loop above can start from something valid rather than
/// discovering the problem after it has spent its bits.
///
/// Returns [`SF_OFFSET`] for a silent band, where any scalefactor would
/// do and none is more honest than another.
pub fn smallest_usable_scalefactor(coefficients: &[f64]) -> i32 {
    let peak = coefficients.iter().fold(
        0.0f64,
        |m, c| {
            if c.is_finite() {
                m.max(c.abs())
            } else {
                m
            }
        },
    );
    if peak == 0.0 {
        return SF_OFFSET;
    }
    // Solve for the gain at which the compressed peak reaches the
    // ceiling, then take the scalefactor just above it. Derived rather
    // than searched: a loop would be slower and no more exact.
    let ceiling = f64::from(MAX_QUANT) - ROUNDING_BIAS;
    let needed = peak / ceiling.powf(EXPAND);
    let sf = (4.0 * needed.log2()).ceil() as i32 + SF_OFFSET;
    // A scalefactor is stored in eight bits; nothing outside that range
    // can be written down, however loud the band.
    let sf = sf.clamp(0, 255);
    // The derivation rounds; step up until it is genuinely true, which
    // is at most a step or two and is cheap next to being wrong.
    let mut sf = sf;
    while sf < 255
        && coefficients
            .iter()
            .any(|c| quantise_one(*c, sf).abs() >= MAX_QUANT)
    {
        sf += 1;
    }
    sf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, amplitude: f64) -> Vec<f64> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    #[test]
    fn the_error_is_proportional_once_the_quantised_value_is_not_tiny() {
        // The power law spaces its levels so the error is a roughly
        // constant *fraction* — but only where there are levels to
        // spare. At a quantised magnitude of six the neighbouring levels
        // are 8.55 and 10.90, so a coefficient of 10 cannot come back
        // closer than about 9%, and no quantiser could do better at that
        // resolution.
        //
        // That floor is not a defect; it is the reason a scalefactor
        // exists per band. Rate control scales a band so its
        // coefficients land where the levels are dense enough, and this
        // test states the bargain rather than pretending it is absent.
        for (magnitude, tolerance) in [
            (10.0, 0.10),
            (100.0, 0.02),
            (1_000.0, 0.01),
            (100_000.0, 0.01),
        ] {
            let q = quantise_one(magnitude, SF_OFFSET);
            let back = dequantise_one(q, SF_OFFSET);
            let relative = (back - magnitude).abs() / magnitude;
            assert!(
                relative < tolerance,
                "{magnitude} (quantised to {q}): came back as {back}, off by {:.1}%",
                relative * 100.0
            );
        }
    }

    #[test]
    fn a_scalefactor_buys_back_the_precision_a_quiet_band_lacks() {
        // The remedy for the floor above, demonstrated: the same quiet
        // coefficient, given a scalefactor that puts it where the levels
        // are dense, comes back far closer.
        let quiet = 10.0;
        let coarse = (dequantise_one(quantise_one(quiet, SF_OFFSET), SF_OFFSET) - quiet).abs();
        let sf = SF_OFFSET - 40; // ten octaves of gain
        let fine = (dequantise_one(quantise_one(quiet, sf), sf) - quiet).abs();
        assert!(
            fine < coarse / 10.0,
            "scaling the band should sharpen it: {coarse} then {fine}"
        );
    }

    #[test]
    fn the_sign_always_survives() {
        // A magnitude a little wrong is quantisation. A sign wrong is a
        // sample pointing the other way, which is not the same kind of
        // mistake at all.
        for c in noise(200, 5_000.0) {
            let q = quantise_one(c, SF_OFFSET);
            if q != 0 {
                assert_eq!(
                    q.signum(),
                    if c < 0.0 { -1 } else { 1 },
                    "sign of {c} became {q}"
                );
                assert_eq!(dequantise_one(q, SF_OFFSET).signum(), c.signum());
            }
        }
    }

    #[test]
    fn raising_the_scalefactor_costs_precision_and_saves_range() {
        // This is the dial rate control turns, so its direction had
        // better be certain: four more means half the precision.
        let x = 5_000.0;
        let fine = quantise_one(x, SF_OFFSET);
        let coarse = quantise_one(x, SF_OFFSET + 4);
        assert!(
            coarse < fine,
            "a larger scalefactor must quantise more coarsely: {fine} then {coarse}"
        );
        // And the reconstruction still lands near the original, which is
        // what makes it a trade rather than a loss.
        let back = dequantise_one(coarse, SF_OFFSET + 4);
        assert!((back - x).abs() / x < 0.1, "came back as {back}");
    }

    #[test]
    fn nothing_can_exceed_what_the_codebooks_can_express() {
        // A value past the ceiling is not a loud coefficient, it is an
        // unwritable one. Clamping here keeps the bitstream valid; the
        // scalefactor is what actually fixes it.
        for x in [1e9, -1e9, 1e30, -1e30] {
            let q = quantise_one(x, SF_OFFSET);
            assert!(q.abs() <= MAX_QUANT, "{x} quantised to {q}");
            assert_eq!(q.signum(), if x < 0.0 { -1 } else { 1 });
        }
    }

    #[test]
    fn silence_and_nonsense_quantise_to_zero() {
        // Infinities and NaN reach here only from a bug upstream, and
        // the answer that keeps a file playable is zero rather than a
        // panic or whatever `as i32` happens to produce.
        for x in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(quantise_one(x, SF_OFFSET), 0, "{x}");
        }
        assert_eq!(dequantise_one(0, SF_OFFSET), 0.0);
    }

    #[test]
    fn the_smallest_usable_scalefactor_actually_fits_the_band() {
        // The whole value of computing it: whatever it returns must
        // leave every coefficient representable, or rate control starts
        // from an invalid state and finds out later.
        for amplitude in [1.0, 1e3, 1e6, 1e9, 1e12] {
            let band = noise(64, amplitude);
            let sf = smallest_usable_scalefactor(&band);
            assert!((0..=255).contains(&sf), "scalefactor {sf} is unwritable");
            for c in &band {
                let q = quantise_one(*c, sf);
                assert!(
                    q.abs() < MAX_QUANT,
                    "amplitude {amplitude}: {c} still needs {q} at scalefactor {sf}"
                );
            }
        }
    }

    #[test]
    fn one_step_lower_would_not_have_fitted() {
        // It must be the *smallest* usable one, not merely a safe one —
        // an over-large scalefactor throws away precision nobody asked
        // to lose.
        let band = noise(64, 1e6);
        let sf = smallest_usable_scalefactor(&band);
        assert!(sf > 0);
        let too_fine = sf - 1;
        assert!(
            band.iter()
                .any(|c| quantise_one(*c, too_fine).abs() >= MAX_QUANT),
            "scalefactor {sf} was not the smallest that fits"
        );
    }

    #[test]
    fn a_silent_band_needs_no_particular_scalefactor() {
        assert_eq!(smallest_usable_scalefactor(&[0.0; 32]), SF_OFFSET);
        assert_eq!(smallest_usable_scalefactor(&[]), SF_OFFSET);
    }

    #[test]
    fn quantising_a_band_matches_quantising_its_coefficients() {
        // The batch call is a convenience, and a convenience that
        // disagreed with the thing it wraps would be a trap.
        let band = noise(32, 2_000.0);
        let mut q = Vec::new();
        quantise(&band, SF_OFFSET + 8, &mut q);
        assert_eq!(q.len(), band.len());
        for (i, c) in band.iter().enumerate() {
            assert_eq!(q[i], quantise_one(*c, SF_OFFSET + 8));
        }
        let mut back = Vec::new();
        dequantise(&q, SF_OFFSET + 8, &mut back);
        for (i, v) in back.iter().enumerate() {
            assert_eq!(*v, dequantise_one(q[i], SF_OFFSET + 8));
        }
    }

    #[test]
    fn quantisation_never_increases_a_magnitude_ordering() {
        // Monotonicity: a louder coefficient must not quantise smaller.
        // Without it, the loop that searches for a scalefactor could
        // walk in the wrong direction and never terminate.
        let mut previous = 0i32;
        for step in 0..2_000 {
            let x = f64::from(step) * 7.0;
            let q = quantise_one(x, SF_OFFSET);
            assert!(q >= previous, "at {x}: {q} after {previous}");
            previous = q;
        }
    }
}
