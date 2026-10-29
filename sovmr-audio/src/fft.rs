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

//! A fast Fourier transform, because the direct one is unaffordable.
//!
//! [`crate::mdct`] measures its own direct transform at 0.68 seconds of
//! work per second of audio per channel — some forty-nine hours for a
//! twelve-hour six-track job. That number is why this file exists.
//!
//! Radix-2 decimation-in-time, in place, on power-of-two lengths. There
//! is nothing novel here and there should not be: the transform is
//! sixty years old, and the only thing worth insisting on is that this
//! one is *right*, which is why it is checked against the definition it
//! replaces rather than against itself.

use std::f64::consts::PI;

/// A complex number. Written out rather than pulled in, like everything
/// else in this project.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// A complex number from its parts.
    pub const fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// `e^(i·theta)`, the unit vector at that angle.
    pub fn unit(theta: f64) -> Complex {
        Complex::new(theta.cos(), theta.sin())
    }

    /// Distance from the origin.
    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }
}

// The real operators rather than methods named after them: inherent
// `mul`/`add`/`sub` would shadow the traits at every call site, and a
// reader would have to check which one they were looking at.
impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

/// Reorder `data` so a decimation-in-time pass reads it sequentially.
///
/// Each element moves to the position its index spells backwards in
/// binary. Doing this first is what lets the butterflies below run in
/// place with no scratch buffer at all.
fn bit_reverse(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Transform `data` in place.
///
/// `inverse` conjugates the twiddle factors and scales by `1/n`, so a
/// forward transform followed by an inverse one returns the input.
///
/// # Panics
///
/// If the length is not a power of two. That is a caller's mistake
/// rather than a datum, and silently padding would return a transform of
/// something the caller never asked about.
pub fn transform(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    // Nothing and one thing are already their own transforms, and the
    // check below would reject an empty slice for not being a power of
    // two — which is true and unhelpful.
    if n <= 1 {
        return;
    }
    assert!(
        n.is_power_of_two(),
        "radix-2 needs a power-of-two length, got {n}"
    );
    bit_reverse(data);

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        // The angle between successive twiddles at this stage.
        let step = sign * 2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = Complex::unit(step * k as f64);
                let a = data[start + k];
                let b = data[start + k + len / 2] * w;
                data[start + k] = a + b;
                data[start + k + len / 2] = a - b;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            x.re *= scale;
            x.im *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform as defined, summed term by term.
    ///
    /// Unusably slow and obviously correct — which is exactly what an
    /// oracle is for. The fast version is checked against this, not
    /// against a previous run of itself.
    fn naive(data: &[Complex], inverse: bool) -> Vec<Complex> {
        let n = data.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
        (0..n)
            .map(|k| {
                let mut sum = Complex::default();
                for (j, x) in data.iter().enumerate() {
                    let w = Complex::unit(sign * 2.0 * PI * (j * k) as f64 / n as f64);
                    sum = sum + *x * w;
                }
                Complex::new(sum.re * scale, sum.im * scale)
            })
            .collect()
    }

    fn noise(len: usize) -> Vec<Complex> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        (0..len)
            .map(|_| {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
                };
                Complex::new(next(), next())
            })
            .collect()
    }

    fn assert_close(a: &[Complex], b: &[Complex], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let drift = (*x - *y).magnitude();
            assert!(drift < 1e-9, "{what}: element {i} differs by {drift}");
        }
    }

    #[test]
    fn the_fast_transform_agrees_with_the_definition() {
        // Every size the encoder will actually use, and the small ones
        // besides — a butterfly indexing error often shows only at one
        // stage count.
        for bits in 0..=10 {
            let n = 1usize << bits;
            let x = noise(n);
            for inverse in [false, true] {
                let mut mine = x.clone();
                transform(&mut mine, inverse);
                assert_close(
                    &mine,
                    &naive(&x, inverse),
                    &format!("n={n} inverse={inverse}"),
                );
            }
        }
    }

    #[test]
    fn a_forward_transform_undone_returns_the_input() {
        for bits in 1..=10 {
            let n = 1usize << bits;
            let x = noise(n);
            let mut y = x.clone();
            transform(&mut y, false);
            transform(&mut y, true);
            assert_close(&y, &x, &format!("round trip n={n}"));
        }
    }

    #[test]
    fn a_pure_tone_lands_in_exactly_two_bins() {
        // A real cosine at bin k must appear at k and at n-k, and
        // nowhere else. This is the check that catches a twiddle sign
        // error, which a round trip alone would hide — conjugating twice
        // returns the input however wrong each pass was.
        let n = 64usize;
        let k = 7usize;
        let mut x: Vec<Complex> = (0..n)
            .map(|i| Complex::new((2.0 * PI * (k * i) as f64 / n as f64).cos(), 0.0))
            .collect();
        transform(&mut x, false);
        for (bin, value) in x.iter().enumerate() {
            let expected = if bin == k || bin == n - k {
                n as f64 / 2.0
            } else {
                0.0
            };
            assert!(
                (value.magnitude() - expected).abs() < 1e-9,
                "bin {bin}: {} but expected {expected}",
                value.magnitude()
            );
        }
    }

    #[test]
    fn a_constant_signal_lands_entirely_in_the_first_bin() {
        let n = 32usize;
        let mut x = vec![Complex::new(1.0, 0.0); n];
        transform(&mut x, false);
        assert!((x[0].re - n as f64).abs() < 1e-9);
        assert!(x[1..].iter().all(|v| v.magnitude() < 1e-9));
    }

    #[test]
    fn trivial_lengths_are_handled_rather_than_special_cased_by_the_caller() {
        let mut one = vec![Complex::new(3.0, -4.0)];
        transform(&mut one, false);
        assert_eq!(one[0], Complex::new(3.0, -4.0));
        let mut none: Vec<Complex> = Vec::new();
        transform(&mut none, false);
        assert!(none.is_empty());
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn a_length_this_cannot_transform_is_refused_not_padded() {
        // Padding would return the transform of a different signal, and
        // return it confidently.
        let mut x = vec![Complex::default(); 12];
        transform(&mut x, false);
    }

    #[test]
    fn the_speed_that_justifies_this_file_is_measured() {
        // The direct MDCT was measured at 14.5 ms per 1024-point block.
        // This prints the comparable figure so the next step — wiring
        // the transform into the MDCT — has a target rather than a hope.
        let n = 512usize;
        let mut x = noise(n);
        let started = std::time::Instant::now();
        const ROUNDS: u32 = 200;
        let mut sink = 0.0;
        for r in 0..ROUNDS {
            x[0].re = f64::from(r) * 1e-9;
            transform(std::hint::black_box(&mut x), false);
            sink += x[0].re;
        }
        std::hint::black_box(sink);
        let per = started.elapsed().as_secs_f64() / f64::from(ROUNDS);
        println!("fft: {:.4} ms per {n}-point transform", per * 1000.0);
        assert!(per > 0.0);
    }
}
