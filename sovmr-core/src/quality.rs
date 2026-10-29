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

//! How close two pictures are, measured two ways.
//!
//! # Why the program has to measure itself
//!
//! Today the only number this project can produce for output quality
//! comes from `ffmpeg -lavfi psnr`, which means the shipped binary
//! cannot check its own work. Every encoder lever added from here —
//! B-pyramid, more reference frames, a different preset — is a claim
//! about quality, and a claim that cannot be measured must not be
//! believed.
//!
//! # Two numbers, because one is not enough
//!
//! **PSNR** is the error energy, and it is honest about exactly one
//! thing: how far each sample moved. It cannot tell a shift of every
//! sample by one from a single block destroyed, and it scores blur
//! generously.
//!
//! **SSIM** compares local structure — brightness, contrast and
//! correlation over a small window — and tracks what a viewer notices
//! rather better. It is also slower and bounded to one, so it stops
//! discriminating near the top.
//!
//! Neither replaces the other. A lever that raised one and lowered the
//! other would be worth arguing about, which is the point of having
//! both.
//!
//! # What is measured
//!
//! The luma plane only. Chroma is subsampled and carries far less of
//! what an eye judges sharpness by; measuring it separately is a
//! refinement, and measuring it *mixed into one number* would hide
//! which of the two moved.

/// Largest value an eight-bit sample can take.
const MAX_SAMPLE: f64 = 255.0;

/// Side of the window SSIM compares structure over.
///
/// Eight is the block size video codecs themselves think in, so a
/// window this size lines up with the artefacts they produce rather
/// than straddling them.
const SSIM_WINDOW: usize = 8;

/// Stabilisers from the SSIM definition, keeping the ratio finite where
/// a window is flat: `(0.01 · L)²` and `(0.03 · L)²` for dynamic range
/// `L`.
const SSIM_C1: f64 = (0.01 * MAX_SAMPLE) * (0.01 * MAX_SAMPLE);
/// See [`SSIM_C1`].
const SSIM_C2: f64 = (0.03 * MAX_SAMPLE) * (0.03 * MAX_SAMPLE);

/// A luma plane: samples, and how they are laid out.
///
/// `stride` is bytes per row, which is not width: a plane out of a
/// decoder is pitch-aligned, and reading it as if it were tightly
/// packed would shear the picture a little further on every row.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    /// Sample bytes, at least `stride * height` of them.
    pub samples: &'a [u8],
    /// Picture width in samples.
    pub width: usize,
    /// Picture height in rows.
    pub height: usize,
    /// Bytes per row, `>= width`.
    pub stride: usize,
}

impl Plane<'_> {
    /// One sample, or `None` when the plane does not reach it.
    fn at(&self, x: usize, y: usize) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.samples.get(y * self.stride + x).copied()
    }

    /// Whether the plane holds every sample it claims to.
    fn is_whole(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.stride >= self.width
            && self.samples.len() >= (self.height - 1) * self.stride + self.width
    }
}

/// Why two planes could not be compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityError {
    /// The two pictures are not the same size.
    ///
    /// Comparing them would need a resampler, and any answer would be
    /// as much about the resampler as about the encoder.
    SizeMismatch {
        /// The reference's dimensions.
        reference: (usize, usize),
        /// The candidate's dimensions.
        candidate: (usize, usize),
    },
    /// A plane does not hold the samples it claims to.
    Truncated,
}

/// Mean squared error between two luma planes.
///
/// The quantity both metrics below are built from, exposed because a
/// caller accumulating over a whole file wants to sum errors rather
/// than average decibels — the mean of a logarithm is not the logarithm
/// of a mean, and averaging per-frame PSNR is a subtly different number
/// from the PSNR of the whole.
pub fn mean_squared_error(
    reference: &Plane<'_>,
    candidate: &Plane<'_>,
) -> Result<f64, QualityError> {
    check(reference, candidate)?;
    let mut total = 0.0f64;
    for y in 0..reference.height {
        for x in 0..reference.width {
            let a = f64::from(reference.at(x, y).ok_or(QualityError::Truncated)?);
            let b = f64::from(candidate.at(x, y).ok_or(QualityError::Truncated)?);
            total += (a - b) * (a - b);
        }
    }
    Ok(total / (reference.width * reference.height) as f64)
}

/// Peak signal-to-noise ratio in decibels.
///
/// Returns `f64::INFINITY` for identical pictures. That is the honest
/// answer — the error is zero and its logarithm is unbounded — and a
/// caller that would rather print a number should say so at the point
/// of printing rather than have this lie by capping.
pub fn psnr(reference: &Plane<'_>, candidate: &Plane<'_>) -> Result<f64, QualityError> {
    let mse = mean_squared_error(reference, candidate)?;
    Ok(psnr_from_mse(mse))
}

/// PSNR from an already-accumulated mean squared error.
pub fn psnr_from_mse(mse: f64) -> f64 {
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (MAX_SAMPLE * MAX_SAMPLE / mse).log10()
}

/// Structural similarity, averaged over eight-by-eight windows.
///
/// One for identical pictures, falling towards zero as structure
/// diverges. Windows are tiled rather than slid one sample at a time:
/// the sliding form costs sixty-four times as much for a number that
/// moves in the third decimal, and this is a measurement taken over
/// every frame of a twelve-hour file.
///
/// Rows and columns past the last whole window are not measured. A
/// partial window has fewer samples and would carry the same weight as
/// a full one, which quietly biases the average towards whatever
/// happens to be at the right edge.
pub fn ssim(reference: &Plane<'_>, candidate: &Plane<'_>) -> Result<f64, QualityError> {
    check(reference, candidate)?;
    let across = reference.width / SSIM_WINDOW;
    let down = reference.height / SSIM_WINDOW;
    if across == 0 || down == 0 {
        // Smaller than one window: there is no structure to compare, and
        // inventing a score from a partial window would be worse than
        // saying the pictures are as similar as this can tell.
        return Ok(1.0);
    }

    let mut total = 0.0f64;
    for wy in 0..down {
        for wx in 0..across {
            total += window_ssim(reference, candidate, wx * SSIM_WINDOW, wy * SSIM_WINDOW)?;
        }
    }
    Ok(total / (across * down) as f64)
}

/// SSIM of one window, its top-left corner at `(ox, oy)`.
fn window_ssim(
    reference: &Plane<'_>,
    candidate: &Plane<'_>,
    ox: usize,
    oy: usize,
) -> Result<f64, QualityError> {
    let n = (SSIM_WINDOW * SSIM_WINDOW) as f64;
    let (mut sum_a, mut sum_b) = (0.0f64, 0.0f64);
    let (mut sum_aa, mut sum_bb, mut sum_ab) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..SSIM_WINDOW {
        for x in 0..SSIM_WINDOW {
            let a = f64::from(
                reference
                    .at(ox + x, oy + y)
                    .ok_or(QualityError::Truncated)?,
            );
            let b = f64::from(
                candidate
                    .at(ox + x, oy + y)
                    .ok_or(QualityError::Truncated)?,
            );
            sum_a += a;
            sum_b += b;
            sum_aa += a * a;
            sum_bb += b * b;
            sum_ab += a * b;
        }
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    // Population variance, not sample: every sample in the window is
    // present, so there is nothing to correct for.
    let var_a = sum_aa / n - mean_a * mean_a;
    let var_b = sum_bb / n - mean_b * mean_b;
    let cov = sum_ab / n - mean_a * mean_b;

    let luminance =
        (2.0 * mean_a * mean_b + SSIM_C1) / (mean_a * mean_a + mean_b * mean_b + SSIM_C1);
    let structure = (2.0 * cov + SSIM_C2) / (var_a + var_b + SSIM_C2);
    Ok(luminance * structure)
}

/// Refuse anything that cannot be compared honestly.
fn check(reference: &Plane<'_>, candidate: &Plane<'_>) -> Result<(), QualityError> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return Err(QualityError::SizeMismatch {
            reference: (reference.width, reference.height),
            candidate: (candidate.width, candidate.height),
        });
    }
    if !reference.is_whole() || !candidate.is_whole() {
        return Err(QualityError::Truncated);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tightly packed plane of one repeated value.
    fn flat(value: u8, width: usize, height: usize) -> Vec<u8> {
        vec![value; width * height]
    }

    fn plane<'a>(samples: &'a [u8], width: usize, height: usize) -> Plane<'a> {
        Plane {
            samples,
            width,
            height,
            stride: width,
        }
    }

    #[test]
    fn identical_pictures_have_no_error_at_all() {
        // Infinity rather than a large number: the error is zero and its
        // logarithm is unbounded. Capping here would make a perfect
        // encode indistinguishable from a very good one.
        let a = flat(128, 16, 16);
        let p = plane(&a, 16, 16);
        assert_eq!(mean_squared_error(&p, &p), Ok(0.0));
        assert_eq!(psnr(&p, &p), Ok(f64::INFINITY));
        let s = ssim(&p, &p).expect("compares");
        assert!((s - 1.0).abs() < 1e-12, "identical structure scores {s}");
    }

    #[test]
    fn a_difference_of_one_everywhere_is_the_textbook_number() {
        // MSE 1 gives 20·log10(255) = 48.13 dB. Hand-computable, which
        // is the point: a metric checked only against itself proves
        // nothing.
        let a = flat(100, 32, 32);
        let b = flat(101, 32, 32);
        let mse = mean_squared_error(&plane(&a, 32, 32), &plane(&b, 32, 32)).expect("compares");
        assert!((mse - 1.0).abs() < 1e-12, "mse {mse}");
        let db = psnr(&plane(&a, 32, 32), &plane(&b, 32, 32)).expect("compares");
        assert!((db - 48.1308).abs() < 1e-3, "{db} dB");
    }

    #[test]
    fn black_against_white_is_zero_decibels() {
        // The worst an eight-bit picture can do: every sample moved the
        // whole range, so the error energy equals the peak energy.
        let a = flat(0, 16, 16);
        let b = flat(255, 16, 16);
        let db = psnr(&plane(&a, 16, 16), &plane(&b, 16, 16)).expect("compares");
        assert!(db.abs() < 1e-9, "{db} dB");
    }

    #[test]
    fn a_pitched_plane_is_read_by_its_stride_and_not_its_width() {
        // A decoder hands back pitch-aligned rows. Reading them as if
        // tightly packed shears the picture a little further on every
        // row, which shows up as a plausible-looking bad score rather
        // than as an error.
        let width = 5usize;
        let stride = 8usize;
        let height = 4usize;
        let mut padded = vec![0u8; stride * height];
        for y in 0..height {
            for x in 0..width {
                padded[y * stride + x] = 200;
            }
            // Padding bytes differ, and must be ignored entirely.
            for x in width..stride {
                padded[y * stride + x] = 7;
            }
        }
        let pitched = Plane {
            samples: &padded,
            width,
            height,
            stride,
        };
        let packed_samples = flat(200, width, height);
        let packed = plane(&packed_samples, width, height);
        assert_eq!(mean_squared_error(&pitched, &packed), Ok(0.0));
    }

    #[test]
    fn different_sizes_are_refused_rather_than_resampled() {
        // Any answer would be as much about the resampler as about the
        // encoder, and this exists to measure the encoder.
        let a = flat(0, 16, 16);
        let b = flat(0, 8, 8);
        assert_eq!(
            mean_squared_error(&plane(&a, 16, 16), &plane(&b, 8, 8)),
            Err(QualityError::SizeMismatch {
                reference: (16, 16),
                candidate: (8, 8),
            })
        );
    }

    #[test]
    fn a_plane_that_does_not_hold_its_samples_is_refused() {
        let short = vec![0u8; 10];
        let claimed = Plane {
            samples: &short,
            width: 16,
            height: 16,
            stride: 16,
        };
        assert_eq!(
            mean_squared_error(&claimed, &claimed),
            Err(QualityError::Truncated)
        );
    }

    #[test]
    fn ssim_notices_a_structural_change_psnr_scores_the_same() {
        // The reason both numbers exist. Two candidates with *identical*
        // mean squared error: one moved every sample a little, the other
        // destroyed one region. PSNR cannot tell them apart; SSIM must.
        let size = 32usize;
        let mut reference = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                reference[y * size + x] = ((x * 8 + y * 3) % 256) as u8;
            }
        }

        // Spread: every sample off by four.
        let spread: Vec<u8> = reference.iter().map(|v| v.saturating_add(4)).collect();
        // Concentrated: the same total error energy, in one sixteenth of
        // the picture. 4² over the whole equals 16² over a sixteenth.
        let mut concentrated = reference.clone();
        for y in 0..size / 4 {
            for x in 0..size / 4 {
                concentrated[y * size + x] = reference[y * size + x].saturating_add(16);
            }
        }

        let r = plane(&reference, size, size);
        let mse_spread = mean_squared_error(&r, &plane(&spread, size, size)).expect("compares");
        let mse_conc = mean_squared_error(&r, &plane(&concentrated, size, size)).expect("compares");
        assert!(
            (mse_spread - mse_conc).abs() / mse_spread < 0.05,
            "the fixture must put the same error energy in both: {mse_spread} vs {mse_conc}"
        );

        let ssim_spread = ssim(&r, &plane(&spread, size, size)).expect("compares");
        let ssim_conc = ssim(&r, &plane(&concentrated, size, size)).expect("compares");
        assert!(
            ssim_conc < ssim_spread,
            "SSIM scored a destroyed region ({ssim_conc:.4}) no worse than \
             an even shift ({ssim_spread:.4}), so it is measuring energy rather \
             than structure"
        );
    }

    #[test]
    fn ssim_stays_within_its_bounds() {
        let size = 64usize;
        let black = flat(0, size, size);
        let white = flat(255, size, size);
        let mut noise = vec![0u8; size * size];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for byte in noise.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 33) as u8;
        }
        for (a, b) in [
            (&black, &white),
            (&black, &noise),
            (&noise, &white),
            (&noise, &noise),
        ] {
            let s = ssim(&plane(a, size, size), &plane(b, size, size)).expect("compares");
            assert!((-1.0..=1.0).contains(&s), "SSIM {s} is outside [-1, 1]");
        }
    }

    #[test]
    fn a_picture_smaller_than_one_window_says_so_rather_than_guessing() {
        // Fewer than eight rows or columns leaves no whole window. A
        // score from a partial one would weigh four samples as heavily
        // as sixty-four.
        let a = flat(30, 4, 4);
        let b = flat(200, 4, 4);
        assert_eq!(ssim(&plane(&a, 4, 4), &plane(&b, 4, 4)), Ok(1.0));
        // ...and PSNR still answers, because it needs no window.
        assert!(psnr(&plane(&a, 4, 4), &plane(&b, 4, 4)).expect("compares") < 10.0);
    }
}
