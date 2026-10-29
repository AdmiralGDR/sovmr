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

//! The modified discrete cosine transform, which is where lossy audio
//! coding begins.
//!
//! # The trick it is built on
//!
//! Cutting audio into blocks and transforming each one independently
//! produces audible clicks at every seam. The usual fix is to overlap
//! the blocks and fade between them — but that stores every sample
//! twice, and a compressor cannot afford it.
//!
//! The MDCT overlaps blocks by half and *still* produces only as many
//! coefficients as there are new samples. It manages this by folding the
//! discarded half back into the kept half as a deliberate distortion —
//! time-domain aliasing — arranged so that the aliasing of one block is
//! the exact negative of the next block's. Overlap the two and the error
//! cancels: the original returns exactly.
//!
//! That cancellation is not approximate, and it is the property this
//! module is tested on. A window or a sign error anywhere breaks it, and
//! breaks it loudly.
//!
//! # Windows
//!
//! Cancellation only holds for windows satisfying the Princen–Bradley
//! condition: `w[i]² + w[i + N]² = 1`, where `N` is half the window. It
//! relates a point to the one half a window later — which is where the
//! *next* block's overlap lands — and not, as it is easy to write down,
//! to its own mirror.
//! AAC uses two, chosen per block: a sine window, and a Kaiser–Bessel
//! derived window whose steeper skirts suit material with strong tones.

use std::f64::consts::PI;

use crate::fft::{transform, Complex};

/// Long-block length in samples, per AAC-LC. Half of these are new.
pub const LONG_WINDOW: usize = 2048;
/// Short-block length, used where a long one would smear an attack.
pub const SHORT_WINDOW: usize = 256;

/// Which window shape a block uses.
///
/// AAC lets an encoder switch per block and signals the choice, so a
/// decoder can only reproduce the overlap if both ends agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// The sine window: the simplest shape satisfying the condition.
    Sine,
    /// Kaiser–Bessel derived, alpha 4 for long blocks and 6 for short —
    /// the values the specification fixes.
    Kbd,
}

/// How a frame's window is laid out.
///
/// A long block smears an attack across 23 milliseconds, so AAC lets an
/// encoder split one into eight short ones — and the two frames on
/// either side of the split need asymmetric windows to meet them, which
/// is what the start and stop sequences are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSequence {
    /// One long block.
    OnlyLong,
    /// A long block whose second half narrows to meet short ones.
    LongStart,
    /// Eight short blocks.
    EightShort,
    /// A long block whose first half widens out of short ones.
    LongStop,
}

/// Zeroth-order modified Bessel function of the first kind.
///
/// Summed until the terms stop mattering rather than for a fixed count:
/// the series converges quickly for the arguments a KBD window uses, and
/// a fixed count would be either wasteful or wrong depending on alpha.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..=64 {
        term *= (half / f64::from(k)) * (half / f64::from(k));
        sum += term;
        if term < sum * 1e-18 {
            break;
        }
    }
    sum
}

/// Build the rising half of a window of total length `n`.
///
/// Only half is computed because the other half is its mirror — and
/// because writing it once is the surest way to keep the two halves
/// consistent, which is what the cancellation depends on.
pub fn window_half(n: usize, shape: WindowShape) -> Vec<f64> {
    let half = n / 2;
    match shape {
        WindowShape::Sine => (0..half)
            .map(|i| (PI / n as f64 * (i as f64 + 0.5)).sin())
            .collect(),
        WindowShape::Kbd => {
            // The specification fixes alpha by block length: 4 for long
            // blocks, 6 for short.
            let alpha = if n > SHORT_WINDOW { 4.0 } else { 6.0 };
            let mut kaiser = Vec::with_capacity(half + 1);
            for i in 0..=half {
                let r = 2.0 * i as f64 / half as f64 - 1.0;
                kaiser.push(bessel_i0(PI * alpha * (1.0 - r * r).max(0.0).sqrt()));
            }
            let total: f64 = kaiser.iter().sum();
            // A KBD window is the square root of the running sum of a
            // Kaiser window, normalised — which is precisely what makes
            // its squares sum to one across the overlap.
            let mut running = 0.0;
            (0..half)
                .map(|i| {
                    running += kaiser[i];
                    (running / total).sqrt()
                })
                .collect()
        }
    }
}

/// The full window of length `n`: rising half, then its mirror.
pub fn window(n: usize, shape: WindowShape) -> Vec<f64> {
    let rising = window_half(n, shape);
    let mut out = rising.clone();
    out.extend(rising.into_iter().rev());
    out
}

/// Forward MDCT: `2n` windowed samples in, `n` coefficients out.
///
/// The direct definition, evaluated as written: `n` coefficients each
/// summing `2n` terms.
///
/// That is O(n²), and the cost was measured rather than guessed: **14.5
/// ms per 1024-point block, or 0.68 seconds of work per second of audio
/// per channel** (release build, this host). A twelve-hour six-track job
/// would therefore spend some *forty-nine hours* in this function alone.
///
/// So an FFT-based transform is not an optimisation, it is a
/// requirement — and this stays as the oracle it will be checked
/// against, because a fast transform that is subtly wrong produces audio
/// that sounds almost right.
///
/// A first attempt at that measurement reported zero: the optimiser had
/// removed a loop whose result never changed and was never read. The
/// benchmark disturbs its input and consumes its output now.
///
/// # Panics
///
/// If `input` is not exactly twice `output`'s length.
pub fn mdct(input: &[f64], output: &mut [f64]) {
    let n = output.len();
    assert_eq!(input.len(), 2 * n, "MDCT takes 2n samples and gives n");
    // The n/2 offset is what places the aliasing symmetrically, and is
    // the part most often dropped by mistake — the transform still runs
    // without it, and the overlap simply never cancels.
    let offset = 0.5 + n as f64 / 2.0;
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, x) in input.iter().enumerate() {
            sum += x * (PI / n as f64 * (i as f64 + offset) * (k as f64 + 0.5)).cos();
        }
        *out = sum;
    }
}

/// Inverse MDCT: `n` coefficients in, `2n` samples out.
///
/// The output is *not* the original block — it still carries the
/// time-domain aliasing. Only overlap-adding two consecutive inverse
/// blocks removes it.
///
/// # Panics
///
/// If `output` is not exactly twice `input`'s length.
pub fn imdct(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    assert_eq!(output.len(), 2 * n, "IMDCT takes n coefficients, gives 2n");
    let offset = 0.5 + n as f64 / 2.0;
    let scale = 2.0 / n as f64;
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (k, x) in input.iter().enumerate() {
            sum += x * (PI / n as f64 * (i as f64 + offset) * (k as f64 + 0.5)).cos();
        }
        *out = sum * scale;
    }
}

/// Analyse a signal into overlapping MDCT blocks.
///
/// Uses [`mdct_fast`], so `n/2` must be a power of two — true of every
/// block length AAC defines.
///
/// Blocks advance by `n` samples and each covers `2n`, so consecutive
/// blocks share half their input — the overlap the cancellation needs.
/// The signal is expected to be padded by `n` zeros at each end, which
/// is what lets the first and last blocks reconstruct.
pub fn analyse(signal: &[f64], n: usize, shape: WindowShape) -> Vec<Vec<f64>> {
    let w = window(2 * n, shape);
    let mut blocks = Vec::new();
    let mut windowed = vec![0.0; 2 * n];
    let mut at = 0usize;
    while at + 2 * n <= signal.len() {
        for i in 0..2 * n {
            windowed[i] = signal[at + i] * w[i];
        }
        let mut coefficients = vec![0.0; n];
        // The fast path, which the tests hold to the direct one's answer
        // on every size and signal.
        mdct_fast(&windowed, &mut coefficients);
        blocks.push(coefficients);
        at += n;
    }
    blocks
}

/// Reconstruct a signal from overlapping MDCT blocks.
///
/// The inverse of [`analyse`]: each block is transformed back, windowed
/// again, and added into place. Where two blocks overlap their aliasing
/// cancels, which is the whole reason the transform is usable.
pub fn synthesise(blocks: &[Vec<f64>], n: usize, shape: WindowShape) -> Vec<f64> {
    let w = window(2 * n, shape);
    let mut out = vec![0.0; blocks.len() * n + n];
    let mut block = vec![0.0; 2 * n];
    for (b, coefficients) in blocks.iter().enumerate() {
        imdct(coefficients, &mut block);
        for i in 0..2 * n {
            out[b * n + i] += block[i] * w[i];
        }
    }
    out
}

/// Reconstruct a signal whose blocks do not all share a window shape.
///
/// AAC signals the shape per block, and a decoder that assumes one is
/// not decoding — it is guessing. Measured against ffmpeg's own output:
/// reading real frames with the wrong shape agrees to **16 dB**, and
/// with the right one to over eighty. The sound is recognisable either
/// way, which is exactly why the assumption survived until something
/// measured it.
///
/// The rule the overlap depends on: a block's *rising* half carries the
/// **previous** block's shape, because that is the half overlapping the
/// previous block's falling half, and the aliasing only cancels when the
/// two agree. Its falling half carries its own.
///
/// # Panics
///
/// If there is not exactly one shape per block.
pub fn synthesise_shaped(blocks: &[Vec<f64>], shapes: &[WindowShape], n: usize) -> Vec<f64> {
    assert_eq!(
        blocks.len(),
        shapes.len(),
        "a shape per block, or the overlap is a guess"
    );
    let rising = [
        window_half(2 * n, WindowShape::Sine),
        window_half(2 * n, WindowShape::Kbd),
    ];
    let index = |shape: WindowShape| usize::from(shape == WindowShape::Kbd);
    let mut out = vec![0.0; blocks.len() * n + n];
    let mut block = vec![0.0; 2 * n];
    for (b, coefficients) in blocks.iter().enumerate() {
        imdct(coefficients, &mut block);
        // Nothing precedes the first block, and its rising half is the
        // incomplete end of the overlap-add either way.
        let left = &rising[index(shapes[b.saturating_sub(1)])];
        let right = &rising[index(shapes[b])];
        for i in 0..n {
            out[b * n + i] += block[i] * left[i];
            out[b * n + n + i] += block[n + i] * right[n - 1 - i];
        }
    }
    out
}

/// Short windows in an eight-short frame.
pub const SHORT_WINDOWS: usize = 8;

/// Where the eight short windows sit inside a frame's 2048-sample span.
///
/// Not recalled — it follows from the hop arithmetic and nothing else.
/// Eight short windows advancing by `SHORT_WINDOW / 2` cover
/// `(SHORT_WINDOWS + 1) * SHORT_WINDOW / 2` samples between them, and
/// that run is centred in the long span, because the frame it replaces
/// is centred there too.
///
/// Everything about the two transition windows falls out of this one
/// number: a long-start window must fall exactly where the next frame's
/// first short window rises, and a long-stop window must rise exactly
/// where the previous frame's last short window falls. There is no
/// freedom left to guess at.
pub const SHORT_OFFSET: usize = (LONG_WINDOW - (SHORT_WINDOWS + 1) * SHORT_WINDOW / 2) / 2;

/// The span the short windows occupy, from [`SHORT_OFFSET`].
const SHORT_SPAN: usize = (SHORT_WINDOWS + 1) * SHORT_WINDOW / 2;

/// Build one frame's 2048-sample contribution, already windowed.
///
/// `previous` is the shape of the frame before, because a window's
/// rising half overlaps the previous frame's falling half and the
/// aliasing only cancels when the two agree.
fn windowed_frame(
    coefficients: &[f64],
    sequence: WindowSequence,
    shape: WindowShape,
    previous: WindowShape,
    out: &mut [f64],
) {
    let half = LONG_WINDOW / 2;
    let short_half = SHORT_WINDOW / 2;
    out.fill(0.0);
    if sequence == WindowSequence::EightShort {
        // Eight transforms of their own, overlap-added at the short hop
        // inside the long span. Only the first window's rising half
        // meets the frame before; the rest meet each other.
        let mut block = [0.0f64; SHORT_WINDOW];
        for window in 0..SHORT_WINDOWS {
            let base = SHORT_OFFSET + window * short_half;
            // Each window's coefficients sit one after another, in
            // order — measured, by asking a decoder when the sound of
            // each arrived.
            imdct(
                &coefficients[window * short_half..][..short_half],
                &mut block,
            );
            let rising = window_half(SHORT_WINDOW, if window == 0 { previous } else { shape });
            let falling = window_half(SHORT_WINDOW, shape);
            for i in 0..short_half {
                out[base + i] += block[i] * rising[i];
                out[base + short_half + i] += block[short_half + i] * falling[short_half - 1 - i];
            }
        }
        return;
    }

    let mut block = vec![0.0f64; LONG_WINDOW];
    imdct(coefficients, &mut block);
    // The rising half: a whole long half, or the short one a long-stop
    // window narrows to, with silence before it and unity after.
    match sequence {
        WindowSequence::LongStop => {
            let rising = window_half(SHORT_WINDOW, previous);
            for i in 0..short_half {
                out[SHORT_OFFSET + i] = block[SHORT_OFFSET + i] * rising[i];
            }
            // Unity between the short half and the middle: the window
            // has already fully opened there.
            let flat = SHORT_OFFSET + short_half..half;
            out[flat.clone()].copy_from_slice(&block[flat]);
        }
        _ => {
            let rising = window_half(LONG_WINDOW, previous);
            for i in 0..half {
                out[i] = block[i] * rising[i];
            }
        }
    }
    // The falling half, by the same argument mirrored.
    match sequence {
        WindowSequence::LongStart => {
            let falling = window_half(SHORT_WINDOW, shape);
            // Unity from the middle until the window narrows.
            let flat = half..SHORT_OFFSET + SHORT_SPAN - short_half;
            out[flat.clone()].copy_from_slice(&block[flat]);
            let base = SHORT_OFFSET + SHORT_SPAN - short_half;
            for i in 0..short_half {
                out[base + i] = block[base + i] * falling[short_half - 1 - i];
            }
        }
        _ => {
            let falling = window_half(LONG_WINDOW, shape);
            for i in 0..half {
                out[half + i] = block[half + i] * falling[half - 1 - i];
            }
        }
    }
}

/// Reconstruct a signal from frames that need not all be long.
///
/// # Why the sequences cannot be ignored
///
/// A long block smears an attack across 23 milliseconds, so an encoder
/// splits one into eight short ones — and the frames on either side of
/// the split carry asymmetric windows to meet them. A synthesis that
/// treated every frame as one long block would put the aliasing back
/// instead of cancelling it, at exactly the moments a listener is most
/// able to hear.
///
/// Nothing here is a remembered window table. The one number the layout
/// rests on is [`SHORT_OFFSET`], which is hop arithmetic, and every
/// boundary follows from the requirement that overlapping halves line up
/// exactly. The gate that judges it is the same one that judges
/// everything else in this crate: our reading of somebody else's
/// bitstream against their own decoder's sound.
///
/// # Panics
///
/// If there is not exactly one sequence and one shape per block.
pub fn synthesise_sequenced(
    blocks: &[Vec<f64>],
    sequences: &[WindowSequence],
    shapes: &[WindowShape],
) -> Vec<f64> {
    assert_eq!(
        blocks.len(),
        sequences.len(),
        "a sequence per block, or the layout is a guess"
    );
    assert_eq!(
        blocks.len(),
        shapes.len(),
        "a shape per block, or the overlap is a guess"
    );
    let hop = LONG_WINDOW / 2;
    let mut out = vec![0.0; blocks.len() * hop + hop];
    let mut frame = vec![0.0f64; LONG_WINDOW];
    for (b, coefficients) in blocks.iter().enumerate() {
        assert_eq!(coefficients.len(), hop, "a frame is {hop} coefficients");
        // Nothing precedes the first block, and its rising half is the
        // incomplete end of the overlap-add either way.
        let previous = shapes[b.saturating_sub(1)];
        windowed_frame(coefficients, sequences[b], shapes[b], previous, &mut frame);
        for i in 0..LONG_WINDOW {
            out[b * hop + i] += frame[i];
        }
    }
    out
}

/// Forward MDCT computed through an FFT: the same result as [`mdct`],
/// at a cost the engine can actually pay.
///
/// Two steps. First the `2n` inputs are *folded* into `n` values, which
/// turns the MDCT into a DCT-IV — an exact rearrangement that loses no
/// arithmetic, and where the transform's aliasing is created in plain
/// sight. Then the DCT-IV is evaluated with a complex FFT of length
/// `n/2` between two rotations.
///
/// # The rotation angles, derived rather than recalled
///
/// A first attempt took the post-rotation to be `exp(−iπ(4k+1)/4n)`,
/// symmetric with the pre-rotation, and produced results that were
/// *nearly* right — every coefficient a little off, none of them wildly.
/// Working the phases out settles it. After the pre-rotation and the
/// FFT, term `j` of output `k` carries
///
/// ```text
///   −π(4j + 1 + 16jk) / 4n
/// ```
///
/// while DCT-IV wants `−π(4j+1)(4k+1)/4n`, which expands to
/// `−π(16jk + 4j + 4k + 1)/4n`. The shortfall is `−πk/n` exactly, so
/// that — and not the symmetric-looking angle — is the post-rotation.
/// The difference between the two is a constant `−π/4n`, which is
/// precisely the kind of small phase error that sounds like nothing in
/// particular and is still wrong.
///
/// # Panics
///
/// If `input` is not twice `output`'s length, or if `n/2` is not a power
/// of two — the FFT refuses anything else, and padding would transform a
/// signal the caller never supplied.
pub fn mdct_fast(input: &[f64], output: &mut [f64]) {
    let n = output.len();
    assert_eq!(input.len(), 2 * n, "MDCT takes 2n samples and gives n");
    assert!(
        (n / 2).is_power_of_two(),
        "the FFT needs n/2 to be a power of two, got n={n}"
    );
    let half = n / 2;

    // Fold: MDCT(a,b,c,d) = DCT-IV(−reverse(c)−d, a−reverse(b)).
    let mut u = vec![0.0; n];
    for i in 0..half {
        u[i] = -input[3 * half - 1 - i] - input[3 * half + i];
        u[half + i] = input[i] - input[n - 1 - i];
    }

    // DCT-IV of length n through a complex FFT of length n/2.
    let mut z = Vec::with_capacity(half);
    for j in 0..half {
        // Even values paired with odd ones taken backwards; the rotation
        // is what turns that pairing into DCT-IV's cosine basis.
        let pair = Complex::new(u[2 * j], u[n - 1 - 2 * j]);
        z.push(pair * Complex::unit(-PI * (4.0 * j as f64 + 1.0) / (4.0 * n as f64)));
    }
    transform(&mut z, false);
    for k in 0..half {
        let rotated = z[k] * Complex::unit(-PI * k as f64 / n as f64);
        output[2 * k] = rotated.re;
        output[n - 1 - 2 * k] = -rotated.im;
    }
}

#[cfg(test)]
mod tests {

    /// A signal with structure at several scales, so a windowing fault
    /// cannot hide in a single frequency.
    fn probe(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let t = i as f64;
                (t * 0.031).sin() + 0.4 * (t * 0.29).sin() + 0.1 * (t * 1.13).cos()
            })
            .collect()
    }

    #[test]
    fn a_run_of_long_frames_is_what_the_simpler_synthesis_already_did() {
        // The sequenced path has to be a superset of the shaped one, or
        // every long-only stream this project read before today reads
        // differently now.
        let n = LONG_WINDOW / 2;
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let blocks: Vec<Vec<f64>> = (0..5)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                    })
                    .collect()
            })
            .collect();
        for shape in [WindowShape::Sine, WindowShape::Kbd] {
            let shapes = vec![shape; blocks.len()];
            let plain = synthesise_shaped(&blocks, &shapes, n);
            let sequenced = synthesise_sequenced(
                &blocks,
                &vec![WindowSequence::OnlyLong; blocks.len()],
                &shapes,
            );
            assert_eq!(plain.len(), sequenced.len());
            for (i, (a, b)) in plain.iter().zip(&sequenced).enumerate() {
                assert!((a - b).abs() < 1e-12, "{shape:?} sample {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn a_short_windows_sound_lands_in_its_own_eighth() {
        // The layout the whole short path rests on, and it was measured:
        // a decoder asked when the sound of each window arrived put
        // window w in eighth w. This holds the synthesis to the same
        // answer, and it is what a wrong `SHORT_OFFSET` would break.
        let n = LONG_WINDOW / 2;
        let short = SHORT_WINDOW / 2;
        for window in 0..SHORT_WINDOWS {
            let mut coefficients = vec![0.0f64; n];
            // A middling coefficient, so the burst is a tone rather than
            // an impulse spread over the whole window.
            coefficients[window * short + short / 4] = 1.0;
            let out = synthesise_sequenced(
                &[coefficients],
                &[WindowSequence::EightShort],
                &[WindowShape::Sine],
            );
            // Where each window's energy should be, by the arithmetic:
            // its 256 samples start at the offset plus its own hop.
            let base = SHORT_OFFSET + window * short;
            let inside: f64 = out[base..base + SHORT_WINDOW].iter().map(|s| s * s).sum();
            let total: f64 = out.iter().map(|s| s * s).sum();
            assert!(
                inside > 0.999 * total,
                "window {window}: {:.4} of its energy is outside its own span",
                1.0 - inside / total
            );
        }
    }

    #[test]
    fn the_short_layout_is_the_hop_arithmetic_and_nothing_else() {
        // Eight windows advancing by half a short window, centred in the
        // long span. Everything about the two transition windows follows
        // from this: a long-start falls exactly where the next frame's
        // first short window rises, and a long-stop rises exactly where
        // the previous frame's last one falls.
        assert_eq!(SHORT_OFFSET * 2 + SHORT_SPAN, LONG_WINDOW, "not centred");
        assert_eq!(SHORT_SPAN, (SHORT_WINDOWS + 1) * SHORT_WINDOW / 2);
        // The falling short half of a long-start window, in that frame's
        // own coordinates, and the rising half of the next frame's first
        // short window, in absolute ones. They must be the same samples.
        let hop = LONG_WINDOW / 2;
        let long_start_falls_at = SHORT_OFFSET + SHORT_SPAN - SHORT_WINDOW / 2;
        let next_frames_first_short_rises_at = hop + SHORT_OFFSET;
        assert_eq!(long_start_falls_at, next_frames_first_short_rises_at);
    }

    #[test]
    #[should_panic(expected = "a sequence per block")]
    fn a_missing_sequence_is_refused_rather_than_assumed() {
        // The one thing worse than the wrong layout is a default one: it
        // reconstructs, it sounds nearly right, and nothing reports it.
        let n = LONG_WINDOW / 2;
        synthesise_sequenced(
            &[vec![0.0; n], vec![0.0; n]],
            &[WindowSequence::OnlyLong],
            &[WindowShape::Sine, WindowShape::Sine],
        );
    }

    #[test]
    fn one_shape_throughout_is_what_the_simpler_synthesis_already_did() {
        // The new path has to be the old one's superset, or every stream
        // this project wrote before today reads back differently.
        for shape in [WindowShape::Sine, WindowShape::Kbd] {
            let n = 256;
            let blocks = analyse(&probe(4 * n), n, shape);
            let plain = synthesise(&blocks, n, shape);
            let shaped = synthesise_shaped(&blocks, &vec![shape; blocks.len()], n);
            assert_eq!(plain.len(), shaped.len());
            for (i, (a, b)) in plain.iter().zip(&shaped).enumerate() {
                assert!((a - b).abs() < 1e-12, "{shape:?} sample {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn a_shape_change_still_reconstructs_across_the_join() {
        // The property the overlap rests on: a block's rising half
        // carries the *previous* block's shape. Get that backwards and
        // the aliasing stops cancelling exactly at the change — audible
        // as a click, and invisible to a test that never switches.
        let n = 256;
        let signal = probe(8 * n);
        let count = analyse(&signal, n, WindowShape::Sine).len();
        assert!(count > 4, "not enough blocks to switch inside");

        // Blocks up to the change analysed with one shape, the rest with
        // the other — which is what an encoder that switched would have
        // written, because each block's window is built the same way.
        let change = count / 2;
        let shapes: Vec<WindowShape> = (0..count)
            .map(|b| {
                if b < change {
                    WindowShape::Sine
                } else {
                    WindowShape::Kbd
                }
            })
            .collect();
        let mixed: Vec<Vec<f64>> = (0..count)
            .map(|b| {
                // A block whose halves differ cannot come from `analyse`,
                // which uses one shape for both, so it is built here the
                // way the format defines it.
                let left = window_half(2 * n, shapes[b.saturating_sub(1)]);
                let right = window_half(2 * n, shapes[b]);
                let mut windowed = vec![0.0; 2 * n];
                for i in 0..n {
                    windowed[i] = signal[b * n + i] * left[i];
                    windowed[n + i] = signal[b * n + n + i] * right[n - 1 - i];
                }
                let mut out = vec![0.0; n];
                mdct_fast(&windowed, &mut out);
                out
            })
            .collect();
        let back = synthesise_shaped(&mixed, &shapes, n);

        // Judge the settled middle, including the join. The first and
        // last half-window have one contribution instead of two.
        let (from, to) = (n, signal.len().min(back.len()) - n);
        assert!(to > from, "nothing settled to judge");
        for i in from..to {
            assert!(
                (signal[i] - back[i]).abs() < 1e-9,
                "sample {i} across a shape change: {} vs {}",
                signal[i],
                back[i]
            );
        }
        // And the change really is inside the judged span.
        assert!(
            change * n > from && change * n < to,
            "the join was not judged"
        );
    }

    #[test]
    #[should_panic(expected = "a shape per block")]
    fn a_missing_shape_is_refused_rather_than_assumed() {
        // The one thing worse than the wrong shape is a default one: it
        // decodes, it sounds nearly right, and nothing reports it.
        let n = 256;
        let blocks = analyse(&probe(4 * n), n, WindowShape::Sine);
        synthesise_shaped(&blocks, &[WindowShape::Sine], n);
    }

    use super::*;

    /// A deterministic pseudo-random signal — no dependency, and the
    /// same numbers every run so a failure can be reproduced.
    fn noise(len: usize) -> Vec<f64> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn both_windows_satisfy_the_condition_cancellation_depends_on() {
        // Princen–Bradley, stated correctly: the condition relates a
        // point to the one *half a window later*, because that is where
        // the next block's overlap lands. It is not a symmetry within
        // one window — an easy thing to write down wrongly, and this
        // test did, while the window itself was right all along.
        //
        // Fail this and the overlap does not cancel, which sounds like a
        // rasp on every block boundary, forty-seven times a second.
        for n in [SHORT_WINDOW, LONG_WINDOW] {
            for shape in [WindowShape::Sine, WindowShape::Kbd] {
                let w = window(n, shape);
                assert_eq!(w.len(), n);
                let half = n / 2;
                for i in 0..half {
                    let sum = w[i] * w[i] + w[i + half] * w[i + half];
                    assert!(
                        (sum - 1.0).abs() < 1e-12,
                        "{shape:?}/{n}: at {i} the squares sum to {sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_window_rises_from_near_zero_to_near_one() {
        // A window that did not would still satisfy the condition while
        // being useless: the point is to fade in, not to scale.
        for shape in [WindowShape::Sine, WindowShape::Kbd] {
            let w = window(LONG_WINDOW, shape);
            assert!(w[0] < 0.01, "{shape:?} starts at {}", w[0]);
            assert!(w[LONG_WINDOW / 2 - 1] > 0.99);
            assert!(
                w.windows(2).take(LONG_WINDOW / 2).all(|p| p[1] >= p[0]),
                "{shape:?} must rise monotonically"
            );
        }
    }

    #[test]
    fn overlapping_blocks_reconstruct_the_signal_exactly() {
        // THE property. Time-domain aliasing cancellation is not an
        // approximation: the error here is floating-point noise and
        // nothing else. A sign slip or a missing n/2 offset would show
        // as an obvious wrong answer, not a small one.
        for n in [16usize, 64, SHORT_WINDOW / 2, LONG_WINDOW / 2] {
            for shape in [WindowShape::Sine, WindowShape::Kbd] {
                let mut signal = vec![0.0; n];
                signal.extend(noise(n * 6));
                signal.extend(vec![0.0; n]);

                let blocks = analyse(&signal, n, shape);
                let back = synthesise(&blocks, n, shape);

                // The first and last half-blocks are the fade in and
                // out; everything between must return exactly.
                let mut worst: f64 = 0.0;
                for i in n..signal.len() - n {
                    worst = worst.max((signal[i] - back[i]).abs());
                }
                assert!(
                    worst < 1e-9,
                    "n={n} {shape:?}: reconstruction is off by {worst}"
                );
            }
        }
    }

    #[test]
    fn a_pure_tone_lands_in_few_coefficients() {
        // The transform must actually compact energy, or it buys
        // nothing. A sine at a bin centre should put nearly all of its
        // energy in a couple of coefficients.
        let n = 256;
        let mut signal = vec![0.0; n];
        for i in 0..n * 4 {
            signal.push((2.0 * PI * 20.5 * i as f64 / n as f64).sin());
        }
        signal.extend(vec![0.0; n]);

        let blocks = analyse(&signal, n, WindowShape::Sine);
        let middle = &blocks[2];
        let total: f64 = middle.iter().map(|c| c * c).sum();

        // Where the peak lands is arithmetic, and getting that
        // arithmetic wrong in a test proves nothing about the transform.
        // So the peak is *found*, and what is asserted is the property
        // that matters: nearly all the energy sits beside it.
        let (peak_bin, _) = middle
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        let near: f64 = middle
            .iter()
            .enumerate()
            .filter(|(k, _)| k.abs_diff(peak_bin) <= 2)
            .map(|(_, c)| c * c)
            .sum();
        assert!(
            near / total > 0.98,
            "a tone should compact: {:.3} of the energy is within two bins of {peak_bin}",
            near / total
        );
    }

    #[test]
    fn the_direct_transform_is_measured_rather_than_assumed_slow() {
        // "Metrics before theories": the case for an FFT should rest on
        // a number, not on the shape of the loop. This prints the cost
        // of a long block so the follow-up has something to beat.
        let n = LONG_WINDOW / 2;
        let mut x = noise(2 * n);
        let mut coefficients = vec![0.0; n];
        let started = std::time::Instant::now();
        const ROUNDS: u32 = 20;
        let mut sink = 0.0f64;
        for r in 0..ROUNDS {
            // The input is disturbed and the output consumed, or an
            // optimiser removes the whole loop and the measurement
            // reports zero — which it did, on the first attempt.
            x[0] = f64::from(r) * 1e-9;
            mdct(std::hint::black_box(&x), &mut coefficients);
            sink += coefficients[0];
        }
        std::hint::black_box(sink);
        let per_block = started.elapsed().as_secs_f64() / f64::from(ROUNDS);
        // A long block advances by n samples, so at 48 kHz one second of
        // audio is 48000/n blocks.
        let per_audio_second = per_block * 48_000.0 / n as f64;
        println!(
            "direct MDCT: {:.2} ms per {n}-point block, \
             {:.3} s of work per second of audio per channel",
            per_block * 1000.0,
            per_audio_second
        );
        assert!(per_block > 0.0);
    }

    #[test]
    fn the_fast_transform_agrees_with_the_direct_one() {
        // The whole justification for the fast path, and the reason the
        // direct one is kept: a fast transform that is subtly wrong
        // produces audio that sounds almost right, and nothing further
        // down the pipeline would notice.
        for n in [16usize, 32, 64, 256, LONG_WINDOW / 2] {
            let x = noise(2 * n);
            let mut slow = vec![0.0; n];
            let mut fast = vec![0.0; n];
            mdct(&x, &mut slow);
            mdct_fast(&x, &mut fast);
            let scale = slow.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
            let worst = (0..n).fold(0.0f64, |m, k| m.max((slow[k] - fast[k]).abs() / scale));
            assert!(worst < 1e-9, "n={n}: fast differs from direct by {worst}");
        }
    }

    #[test]
    fn the_fast_path_is_measured_against_the_one_it_replaces() {
        // The direct transform was measured at 15.6 ms per long block.
        // This prints both on the same host in the same run, so the
        // claim is a comparison rather than two remembered numbers.
        let n = LONG_WINDOW / 2;
        let mut x = noise(2 * n);
        let mut out = vec![0.0; n];

        let mut time = |rounds: u32, fast: bool, x: &mut Vec<f64>| -> f64 {
            let started = std::time::Instant::now();
            let mut sink = 0.0;
            for r in 0..rounds {
                x[0] = f64::from(r) * 1e-9;
                if fast {
                    mdct_fast(std::hint::black_box(x), &mut out);
                } else {
                    mdct(std::hint::black_box(x), &mut out);
                }
                sink += out[0];
            }
            std::hint::black_box(sink);
            started.elapsed().as_secs_f64() / f64::from(rounds)
        };

        let slow = time(5, false, &mut x);
        let fast = time(500, true, &mut x);
        println!(
            "MDCT {n}-point: direct {:.3} ms, fast {:.4} ms — {:.0}x; \
             {:.4} s of work per second of audio per channel",
            slow * 1000.0,
            fast * 1000.0,
            slow / fast,
            fast * 48_000.0 / n as f64
        );
        assert!(fast < slow, "the fast path must actually be faster");
    }

    #[test]
    fn silence_transforms_to_silence() {
        let n = 64;
        let blocks = analyse(&vec![0.0; n * 8], n, WindowShape::Sine);
        assert!(blocks.iter().flatten().all(|c| c.abs() < 1e-12));
    }

    #[test]
    fn the_inverse_undoes_the_forward_transform_up_to_the_aliasing() {
        // Without the overlap, a single round trip is *not* the
        // identity — it is the signal plus its folded reflection. This
        // pins that the aliasing is really there, so the cancellation
        // test above is proving something rather than restating it.
        let n = 32;
        let x = noise(2 * n);
        let mut coefficients = vec![0.0; n];
        mdct(&x, &mut coefficients);
        let mut y = vec![0.0; 2 * n];
        imdct(&coefficients, &mut y);
        let differs = (0..2 * n).filter(|i| (x[*i] - y[*i]).abs() > 1e-6).count();
        assert!(
            differs > n,
            "a lone round trip must carry aliasing, not reproduce the input"
        );
    }
}
