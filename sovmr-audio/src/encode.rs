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

//! The AAC-LC encoder: coefficients in, frames out.
//!
//! # What this does
//!
//! Windowed MDCT, one scalefactor for the whole frame, a codebook chosen
//! per band by what it costs, and the frame written out. Every table it
//! leans on — the band boundaries, all eleven codebooks, the escape
//! rule — was measured out of a decoder rather than remembered, and each
//! is re-derived by a gate that can fail.
//!
//! # What this deliberately does not do
//!
//! There is no psychoacoustic model. One scalefactor covers every band,
//! so the quantisation noise is flat across the spectrum rather than
//! hidden under whatever is loud nearby.
//!
//! That was a decision, not an omission. A psychoacoustic model is the
//! part of an encoder whose mistakes are *audible but invisible to
//! tests*: a stream with a bad model is perfectly valid, decodes without
//! complaint, and simply sounds worse. Correctness first, measured
//! against a decoder; the model afterwards, accepted on measurement.
//!
//! What flat noise costs is bitrate, not correctness. The output is a
//! conformant AAC-LC stream at every rate.

use crate::bands::{long_bands, LONG_FRAME};
use crate::bits::BitWriter;
use crate::codebooks::{
    escape_sequence, pair_codeword, quad_codeword, scalefactor_index, Codeword, ESCAPE_11,
    ESCAPE_MAGNITUDE, PAIR_10, PAIR_5, PAIR_6, PAIR_7, PAIR_8, PAIR_9, QUAD_1, QUAD_2, QUAD_3,
    QUAD_4, SCALEFACTOR_DELTA, SCALEFACTOR_DELTA_RANGE,
};
use crate::mdct::{analyse, WindowShape};
use crate::quant::{quantise_one, MAX_QUANT, SF_OFFSET};

/// Syntactic element identifiers.
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_END: u32 = 0x7;
/// `window_sequence` for a plain long block.
const ONLY_LONG_SEQUENCE: u32 = 0;
/// Bits a long window spends on one section's band count.
const SECT_LEN_BITS: u32 = 5;
/// The codebook meaning "this band is entirely zero".
const CODEBOOK_ZERO: u32 = 0;
/// The codebook meaning "this band is noise of a stated energy", and
/// the rules for writing one. All four were walked out of a decoder in
/// stage 6.2 and are re-derived by a gate; this is the writing side of
/// the same measurement.
use crate::decode::{NOISE_BOOK, NOISE_ENERGY_BITS, NOISE_ENERGY_OFFSET, NOISE_ENERGY_RANGE};

/// A flatness threshold no band can meet, so nothing is ever replaced.
///
/// The honest way to hold the two levers apart in a test, and the way a
/// caller turns noise substitution off outright. Flatness is at most
/// one, so anything above that is unreachable.
pub const NEVER_SUBSTITUTE: f64 = 2.0;

/// How flat a band must be before its contents are replaced by noise.
///
/// # What is measured here, and what is not
///
/// The format says exactly how to *write* a noise band, and all of it
/// was walked out of a decoder in stage 6.2: nine bits of energy from
/// `global_gain`, an offset of 346, a range, and a scalefactor codeword
/// for every noise band after the first. The writing side is held to
/// that by a round trip through our own reader, which agrees with the
/// decoder those numbers came from.
///
/// What the format does *not* say is when substituting is a good idea,
/// because that is a judgement about hearing — and this encoder still
/// has no psychoacoustic model. [`flatness_of`] is a measurable proxy
/// for "this band is noise, so nobody can tell one noise from another",
/// and it was swept:
///
/// | flatness | tone @128k        | clicks @320k     | pink noise @320k |
/// |----------|-------------------|------------------|------------------|
/// |      off |  -70.6%, 68.3 dB  |  +0.3%, 61.5 dB  | +191.6%, 57.6 dB |
/// |      0.7 |  -78.1%, 38.5 dB  | -18.7%, 14.5 dB  | +188.7%,  8.0 dB |
/// |      0.3 |  -85.2%, 20.1 dB  | -82.3%,  4.6 dB  |  +66.5%, -0.3 dB |
/// |     0.15 |  -88.3%,  3.7 dB  | -89.3%,  0.8 dB  |  -80.6%, -2.1 dB |
///
/// The size lever is real: pink noise goes from **+191.6% to -80.6%**,
/// which is the material that weighs most in a film.
///
/// # Why it is off by default all the same
///
/// The decibels in that table are *sample* agreement, and for a
/// substituted band they must collapse — the energy is right and the
/// waveform is deliberately different. Judged the only way a noise band
/// can be, by per-band energy, every setting above is within **±0.15 dB**
/// of the encoder that wrote the file.
///
/// So the numbers do not say the tool is wrong. They say this
/// *criterion* is: at a threshold loose enough to catch pink noise it
/// also catches the bands carrying a tone's own harmonics, and no
/// setting gives both. A real encoder has a second condition — it does
/// not substitute below a frequency the ear reads phase at — and that
/// is a claim about hearing this project has not measured and will not
/// guess.
///
/// Shipping it on would trade a size win for damage nothing here has
/// listened for. So it is off, exposed through
/// [`encode_frame_tuned`], and the table above is what the next
/// increment has to improve on.
pub const DEFAULT_NOISE_FLATNESS: f64 = NEVER_SUBSTITUTE;
/// What the loudest coefficient in a frame is asked to quantise to.
///
/// # The one knob this encoder has, and what it is worth
///
/// Until this existed there was no knob at all: the scalefactor was the
/// *smallest* that kept the loudest coefficient inside the quantiser, so
/// every frame was coded at the finest resolution the format allows —
/// maximum bitrate, always. Lowering the target coarsens the step
/// uniformly, and quiet bands fall to zero and cost nothing.
///
/// Priced by re-encoding somebody else's AAC, against the sound we were
/// handed (`tests/oracle.rs`):
///
/// | target | tone @128k        | clicks @320k     | pink noise @320k  |
/// |--------|-------------------|------------------|-------------------|
/// |     63 |  -93.1%, 39.4 dB  | -80.9%, 32.0 dB  |   +6.0%, 27.6 dB  |
/// |    511 |  -84.8%, 56.5 dB  | -40.8%, 49.4 dB  | +108.9%, 45.6 dB  |
/// |   2047 |  -70.6%, 68.3 dB  |  +0.3%, 61.5 dB  | +191.6%, 57.6 dB  |
///
/// The old behaviour was in effect a target of 8191, which measured
/// -8.0% at 79.3 dB on the tone. This default trades eleven decibels for
/// two thirds of the size on tonal material, and it is a measured point
/// on that curve — **not** a claim of transparency, which nothing here
/// has listened for.
///
/// # What this does not fix, and why
///
/// Noise-like material stays larger than the encoder that wrote it, at
/// every target. The reason is not the knob: a real encoder replaces
/// those bands with *noise of a stated energy* and spends nine bits on
/// each, where this one codes every coefficient. That rule was measured
/// in full in stage 6.2 — for reading — and writing it is the next
/// lever, not another turn of this one.
///
/// # A path measured and rejected
///
/// A scalefactor *per band*, each sized so its own loudest coefficient
/// hit the target — constant relative precision — was built and priced
/// first. It was **worse at every target**: the tone went from -70.6% to
/// +608% at matched fidelity. The reason is that a frame-wide step
/// already discards bands that fall under it for nothing, while
/// per-band targets spend full bits on a band a hundred decibels down.
/// The machinery it needed is kept — the delta chain is what a
/// psychoacoustic model will write through — but the policy is gone.
pub const DEFAULT_QUANT_TARGET: i32 = 2047;

/// How a decoder's floating-point samples relate to AAC's coefficients.
///
/// Measured: a quantised one at a scalefactor of 150 comes back as
/// 0.088388, and `2^((150 - 100) / 4) / 0.088388` is 65536 exactly. A
/// decoder hands out samples in minus one to one; the format counts in
/// whole numbers.
///
/// If this were wrong the output would decode at the wrong volume, which
/// is precisely what the round-trip gate measures.
pub(crate) const COEFFICIENT_SCALE: f64 = 65536.0;

/// A codebook an encoder may choose for a band.
///
/// The pairs exist because two books cover each range with different
/// codeword lengths — one is shorter for values clustered near zero, the
/// other for values spread across the range. Which wins is not worth
/// predicting, so the encoder prices both.
struct Book {
    /// What `sect_cb` calls it.
    id: u32,
    /// Largest magnitude it can carry outright.
    range: i32,
    /// Coefficients one codeword covers.
    carries: usize,
    /// The table.
    table: &'static [Codeword],
}

/// Every book, cheapest range first. Order matters: the encoder takes
/// the first whose range fits, then prices the alternatives of that same
/// range against each other.
const BOOKS: [Book; 11] = [
    Book {
        id: 1,
        range: 1,
        carries: 4,
        table: &QUAD_1,
    },
    Book {
        id: 2,
        range: 1,
        carries: 4,
        table: &QUAD_2,
    },
    Book {
        id: 3,
        range: 2,
        carries: 4,
        table: &QUAD_3,
    },
    Book {
        id: 4,
        range: 2,
        carries: 4,
        table: &QUAD_4,
    },
    Book {
        id: 5,
        range: 4,
        carries: 2,
        table: &PAIR_5,
    },
    Book {
        id: 6,
        range: 4,
        carries: 2,
        table: &PAIR_6,
    },
    Book {
        id: 7,
        range: 7,
        carries: 2,
        table: &PAIR_7,
    },
    Book {
        id: 8,
        range: 7,
        carries: 2,
        table: &PAIR_8,
    },
    Book {
        id: 9,
        range: 12,
        carries: 2,
        table: &PAIR_9,
    },
    Book {
        id: 10,
        range: 12,
        carries: 2,
        table: &PAIR_10,
    },
    Book {
        id: 11,
        range: ESCAPE_MAGNITUDE,
        carries: 2,
        table: &ESCAPE_11,
    },
];

/// What a band costs in a given book, or `None` if the book cannot carry
/// it.
///
/// The escape book can carry anything, at a price that grows with the
/// logarithm of the value — so "can it" and "what does it cost" are the
/// same question, asked once.
fn band_cost(book: &Book, values: &[i32]) -> Option<usize> {
    let mut bits = 0usize;
    for group in values.chunks(book.carries) {
        let (_, length) = codeword_for(book, group)?;
        bits += length;
    }
    Some(bits)
}

/// The bits one group of coefficients costs in a book, and their length.
fn codeword_for(book: &Book, group: &[i32]) -> Option<(Vec<Codeword>, usize)> {
    let mut parts = Vec::new();
    let mut length = 0usize;
    if book.carries == 4 {
        let tuple = [group[0], group[1], group[2], group[3]];
        let word = quad_codeword(book.table, tuple, book.range)?;
        length += usize::from(word.1);
        parts.push(word);
    } else {
        let pair = [group[0], group[1]];
        if book.id == 11 {
            // The base carries magnitudes up to fifteen outright and
            // sixteen as a promise; the promise is kept by a sequence
            // whose length depends on the value.
            let clamped = [clamp_for_escape(pair[0]), clamp_for_escape(pair[1])];
            let word = pair_codeword(book.table, clamped, book.range)?;
            length += usize::from(word.1);
            parts.push(word);
            for value in pair {
                if value.abs() >= ESCAPE_MAGNITUDE {
                    let sequence = escape_sequence(value.abs())?;
                    length += usize::from(sequence.1);
                    parts.push(sequence);
                }
            }
        } else {
            let word = pair_codeword(book.table, pair, book.range)?;
            length += usize::from(word.1);
            parts.push(word);
        }
    }
    Some((parts, length))
}

/// A value as the escape book's index sees it: anything at or past the
/// escape magnitude becomes the promise, keeping its sign.
fn clamp_for_escape(value: i32) -> i32 {
    value.clamp(-ESCAPE_MAGNITUDE, ESCAPE_MAGNITUDE)
}

/// The cheapest book for a band, or `None` when the band is all zeros
/// and needs no book at all.
fn choose_book(values: &[i32]) -> Option<(&'static Book, usize)> {
    if values.iter().all(|v| *v == 0) {
        return None;
    }
    BOOKS
        .iter()
        .filter_map(|book| band_cost(book, values).map(|cost| (book, cost)))
        .min_by_key(|(book, cost)| (*cost, book.id))
}

/// The scalefactor that quantises the loudest of `coefficients` to about
/// `target`.
///
/// Inverting the quantiser: `q = (|x| / 2^((sf - 100) / 4))^0.75`, so
/// the scalefactor putting the loudest coefficient at `target` is this.
/// Derived, then stepped up until it truly fits — rounding at the
/// boundary is not worth reasoning about when one extra check settles
/// it, and a coefficient past the ceiling would clip into a different
/// sound entirely.
fn scalefactor_for(coefficients: &[f64], target: i32) -> i32 {
    let loudest = coefficients.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    if loudest <= 0.0 {
        // Nothing to quantise; the frame will code no bands at all and
        // this scalefactor will never be written.
        return SF_OFFSET;
    }
    let aim = f64::from(target.clamp(1, MAX_QUANT));
    let mut scalefactor = SF_OFFSET + (4.0 * (loudest / aim.powf(4.0 / 3.0)).log2()).round() as i32;
    while quantise_one(loudest, scalefactor).abs() > MAX_QUANT {
        scalefactor += 1;
    }
    scalefactor
}

/// Why a frame could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// No band table has been measured for this sampling rate.
    ///
    /// Encoding against another rate's boundaries would produce a stream
    /// that claims to be fine and is not, so it is refused instead.
    UnmeasuredRate(u32),
    /// The scalefactor a frame needs is outside the eight bits
    /// `global_gain` has to carry it.
    UnwritableGain(i32),
}

/// One channel's decisions, made before a bit is written.
///
/// The section list has to know where the runs are before it can be
/// written, and in a stereo element the two channels have to agree on
/// how many bands are coded — so every choice is made first and written
/// second.
struct Plan {
    /// What `global_gain` carries: the first coded band's ordinary
    /// scalefactor, and the base of *both* running chains.
    ///
    /// Kept explicitly rather than read back from the first coded band,
    /// which was wrong the moment that band could be noise — its
    /// scalefactor then holds a stated energy, and every ordinary
    /// difference after it was computed from the wrong base.
    global_gain: i32,
    /// Bands replaced by noise, and the energy each states.
    noise: Vec<Option<i32>>,
    /// The scalefactor of each band, after the delta chain has clamped
    /// what the book can express.
    scalefactors: Vec<i32>,
    /// Quantised coefficients, one per spectral position.
    quantised: Vec<i32>,
    /// The book chosen for each band, or [`CODEBOOK_ZERO`].
    books: Vec<u32>,
    /// Bands up to the last one carrying anything.
    coded_bands: usize,
}

/// Decide how one channel will be coded.
/// How flat a band is: the geometric mean of its squared magnitudes over
/// their arithmetic mean.
///
/// Zero when one coefficient holds everything, one when they are all
/// equal, and — this is the point — **independent of the band's width**.
///
/// The first attempt used peak over root mean square, which is not: it
/// is bounded above by the square root of the width, so a single
/// threshold meant "replace every narrow band" regardless of contents.
/// Measured, that destroyed a pure tone as readily as noise, and the
/// sweep showed it at once as fidelity collapsing on *every* signal
/// rather than only the noisy one.
fn flatness_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut log_sum = 0.0f64;
    let mut sum = 0.0f64;
    for value in values {
        let power = value * value;
        if power <= 0.0 {
            // A zero coefficient sends the geometric mean to zero, which
            // is the right answer: a band with a hole in it is not the
            // flat noise this is looking for.
            return 0.0;
        }
        log_sum += power.ln();
        sum += power;
    }
    let count = values.len() as f64;
    (log_sum / count).exp() / (sum / count)
}

/// The energy a noise band must state to carry `values`, in the units
/// the format counts in, or `None` if it is outside what can be said.
fn noise_energy(values: &[f64]) -> Option<i32> {
    let energy: f64 = values.iter().map(|c| c * c).sum();
    if energy <= 0.0 {
        return None;
    }
    // The measured law: a band's total squared magnitude is two to the
    // half of its stated value.
    Some((2.0 * energy.log2()).round() as i32)
}

fn plan_channel(
    coefficients: &[f64],
    bands: &[usize],
    target: i32,
    flatness: f64,
) -> Result<Plan, EncodeError> {
    assert_eq!(
        coefficients.len(),
        LONG_FRAME,
        "a long frame is {LONG_FRAME} coefficients"
    );
    let scaled: Vec<f64> = coefficients.iter().map(|c| c * COEFFICIENT_SCALE).collect();

    // First pass: what each band would like, and which bands survive it.
    // A band that quantises to nothing is coded as empty and consumes no
    // scalefactor at all, so it must be known before the delta chain is
    // laid down.
    let count = bands.len() - 1;
    let frame = scalefactor_for(&scaled, target);
    let wanted = vec![frame; count];
    let mut quantised = vec![0i32; scaled.len()];
    for (at, value) in quantised.iter_mut().enumerate() {
        *value = quantise_one(scaled[at], frame);
    }
    let mut books: Vec<u32> = (0..count)
        .map(|band| {
            choose_book(&quantised[bands[band]..bands[band + 1]])
                .map_or(CODEBOOK_ZERO, |(book, _)| book.id)
        })
        .collect();

    // Bands flat enough to be noise, and cheaper written as noise than
    // coded. Both halves matter: substituting a tone is audible, and
    // substituting a band that codes in fewer bits than the energy field
    // costs would spend more to say less.
    let mut noise = vec![None; count];
    for band in 0..count {
        if books[band] == CODEBOOK_ZERO {
            continue;
        }
        let (from, to) = (bands[band], bands[band + 1]);
        if flatness_of(&scaled[from..to]) < flatness {
            continue;
        }
        let coded = choose_book(&quantised[from..to]).map_or(0, |(_, cost)| cost);
        // Nine bits for the first, a scalefactor codeword for the rest;
        // the longest that book holds is nineteen, so the cheaper bound
        // is the honest one to compare against.
        if coded <= NOISE_ENERGY_BITS as usize {
            continue;
        }
        let Some(stated) = noise_energy(&scaled[from..to]) else {
            continue;
        };
        let (low, high) = NOISE_ENERGY_RANGE;
        if !(low..=high).contains(&stated) {
            continue;
        }
        noise[band] = Some(stated);
        books[band] = NOISE_BOOK;
    }

    // Second pass: the chain. Each coded band's scalefactor is written
    // as a difference from the one before, and the book measured in
    // stage 6.0 reaches sixty either way. A band that wants more than
    // that gets what the book can say — coarser or finer than ideal, but
    // never something a decoder would read differently from what was
    // meant, which is the only kind of wrong that matters here.
    let first = books.iter().position(|book| *book != CODEBOOK_ZERO);
    let Some(first) = first else {
        // Nothing survived: a silent frame codes no bands at all.
        return Ok(Plan {
            global_gain: SF_OFFSET,
            noise,
            scalefactors: wanted,
            quantised,
            books,
            coded_bands: 0,
        });
    };
    let global_gain = wanted[first];
    if !(0..=255).contains(&global_gain) {
        return Err(EncodeError::UnwritableGain(global_gain));
    }
    let mut scalefactors = wanted.clone();
    let mut running = global_gain;
    // A noise band's energy keeps a running value entirely of its own —
    // measured, and the reason a noise band does not shift the bands
    // after it.
    let mut running_noise = global_gain;
    let mut first_noise = true;
    for band in first..count {
        if books[band] == CODEBOOK_ZERO {
            continue;
        }
        if books[band] == NOISE_BOOK {
            let stated = noise[band].expect("set with the book");
            let sayable = if first_noise {
                // Nine bits from the gain, so anything the field can
                // reach.
                let field = stated - global_gain + NOISE_ENERGY_OFFSET;
                if (0..(1 << NOISE_ENERGY_BITS)).contains(&field) {
                    Some(stated)
                } else {
                    None
                }
            } else {
                let delta = stated - running_noise;
                scalefactor_index(delta).map(|_| stated)
            };
            match sayable {
                Some(stated) => {
                    first_noise = false;
                    running_noise = stated;
                    noise[band] = Some(stated);
                    scalefactors[band] = stated;
                    continue;
                }
                None => {
                    // Outside what the book can say. Code the band the
                    // ordinary way rather than state an energy a decoder
                    // would read as a different one.
                    noise[band] = None;
                    books[band] = choose_book(&quantised[bands[band]..bands[band + 1]])
                        .map_or(CODEBOOK_ZERO, |(b, _)| b.id);
                    if books[band] == CODEBOOK_ZERO {
                        continue;
                    }
                }
            }
        }
        let delta =
            (wanted[band] - running).clamp(-SCALEFACTOR_DELTA_RANGE, SCALEFACTOR_DELTA_RANGE);
        running += delta;
        scalefactors[band] = running;
        if running != wanted[band] {
            // Re-quantise at what the chain could actually say, so the
            // encoder's own idea of the frame is the decoder's.
            let (from, to) = (bands[band], bands[band + 1]);
            for at in from..to {
                quantised[at] = quantise_one(scaled[at], running);
            }
            books[band] = choose_book(&quantised[from..to]).map_or(CODEBOOK_ZERO, |(b, _)| b.id);
        }
    }

    // Trailing silence needs no bands at all: `max_sfb` stops short and
    // a decoder zeroes the rest. On quiet frames that is most of them.
    let coded_bands = books
        .iter()
        .rposition(|book| *book != CODEBOOK_ZERO)
        .map_or(0, |last| last + 1);
    Ok(Plan {
        global_gain,
        noise,
        scalefactors,
        quantised,
        books,
        coded_bands,
    })
}

/// Write an `ics_info` for a long window coding `coded_bands` bands.
fn write_ics_info(w: &mut BitWriter, coded_bands: usize) {
    w.write(0, 1); // ics_reserved_bit
    w.write(ONLY_LONG_SEQUENCE, 2);
    w.write(0, 1); // window_shape: sine, matching the analysis window
    w.write(coded_bands as u32, 6);
    w.write(0, 1); // predictor_data_present — never set for AAC-LC
}

/// Write one `individual_channel_stream`.
///
/// `own_ics_info` is false for the channels of a pair, whose window was
/// already described once for both. Getting that wrong desynchronises
/// the second channel by exactly the width of that structure.
fn write_channel(
    w: &mut BitWriter,
    plan: &Plan,
    bands: &[usize],
    coded_bands: usize,
    own_ics_info: bool,
) {
    let gain = plan.global_gain;
    w.write(gain as u32, 8);
    if own_ics_info {
        write_ics_info(w, coded_bands);
    }

    // section_data: runs of bands sharing a book.
    let mut at = 0usize;
    while at < coded_bands {
        let book = plan.books[at];
        let mut run = 1usize;
        while at + run < coded_bands && plan.books[at + run] == book {
            run += 1;
        }
        w.write(book, 4);
        write_section_length(w, run as u32);
        at += run;
    }

    // scale_factor_data: one Huffman-coded difference per coded band,
    // each from the band before it and the first from `global_gain`.
    let mut running = gain;
    let mut running_noise = gain;
    let mut first_noise = true;
    for (band, book) in plan.books.iter().enumerate().take(coded_bands) {
        if *book == CODEBOOK_ZERO {
            continue;
        }
        if *book == NOISE_BOOK {
            let stated = plan.noise[band].expect("set with the book");
            if first_noise {
                first_noise = false;
                let field = stated - gain + NOISE_ENERGY_OFFSET;
                w.write(field as u32, NOISE_ENERGY_BITS);
            } else {
                let index = scalefactor_index(stated - running_noise)
                    .expect("the plan kept every noise step sayable");
                let (bits, length) = SCALEFACTOR_DELTA[index];
                w.write(bits, u32::from(length));
            }
            running_noise = stated;
            continue;
        }
        let delta = plan.scalefactors[band] - running;
        let index = scalefactor_index(delta).expect("the chain clamped every difference");
        let (bits, length) = SCALEFACTOR_DELTA[index];
        w.write(bits, u32::from(length));
        running = plan.scalefactors[band];
    }

    w.write(0, 1); // pulse_data_present
    w.write(0, 1); // tns_data_present
    w.write(0, 1); // gain_control_data_present

    // spectral_data.
    for (index, book) in plan.books.iter().enumerate().take(coded_bands) {
        if *book == CODEBOOK_ZERO || *book == NOISE_BOOK {
            // A noise band carries no spectral data at all; its energy
            // was the whole message.
            continue;
        }
        let book = BOOKS
            .iter()
            .find(|candidate| candidate.id == *book)
            .expect("chosen from this list");
        let values = &plan.quantised[bands[index]..bands[index + 1]];
        for group in values.chunks(book.carries) {
            let (parts, _) = codeword_for(book, group).expect("priced before it was chosen");
            for (bits, length) in parts {
                w.write(bits, u32::from(length));
            }
        }
    }
}

/// The spectrum this encoder *meant* to write, in the scale a decoder
/// hands back.
///
/// Not what went in: quantising throws away detail on purpose, and
/// trailing bands are dropped on purpose. This is the encoder's own
/// decision, read out of the same code path that writes the bits — so
/// [`crate::decode`] reproducing it exactly is a statement about the two
/// halves agreeing on every measured table, not about quantisation
/// error, which would hide a table fault behind a tolerance.
#[cfg(test)]
pub(crate) fn intended_spectrum(
    coefficients: &[f64],
    sample_rate: u32,
) -> Result<Vec<f64>, EncodeError> {
    let bands = bands_for(sample_rate)?;
    // Substitution off explicitly, not by relying on the default: a
    // noise band's contents are noise, which by construction neither
    // side can predict, and stating an *intended spectrum* for one would
    // be a contradiction.
    let plan = plan_channel(coefficients, bands, DEFAULT_QUANT_TARGET, NEVER_SUBSTITUTE)?;
    let mut out = vec![0.0f64; LONG_FRAME];
    for (index, book) in plan.books.iter().enumerate().take(plan.coded_bands) {
        if *book == CODEBOOK_ZERO {
            continue;
        }
        let (from, to) = (bands[index], bands[index + 1]);
        for (at, coefficient) in out[from..to].iter_mut().enumerate() {
            *coefficient =
                crate::quant::dequantise_one(plan.quantised[from + at], plan.scalefactors[index])
                    / COEFFICIENT_SCALE;
        }
    }
    Ok(out)
}

/// Whether this encoder can produce a stream at this shape.
///
/// A rate needs a measured band table and a channel count needs an
/// element to carry it. Asking before starting is what lets a caller
/// choose a different route rather than discover the refusal halfway
/// through a file.
pub fn can_encode(sample_rate: u32, channels: u32) -> bool {
    long_bands(sample_rate).is_some() && (channels == 1 || channels == 2)
}

/// The band table for a rate, or the error naming the rate.
fn bands_for(sample_rate: u32) -> Result<&'static [usize], EncodeError> {
    long_bands(sample_rate).ok_or(EncodeError::UnmeasuredRate(sample_rate))
}

/// Encode one block of coefficients into a mono `raw_data_block`.
///
/// The coefficients are this project's own MDCT output, in the scale a
/// decoder hands back — the conversion to what the format counts in is
/// [`COEFFICIENT_SCALE`], and it is measured.
pub fn encode_frame(coefficients: &[f64], sample_rate: u32) -> Result<Vec<u8>, EncodeError> {
    encode_frame_at(coefficients, sample_rate, DEFAULT_QUANT_TARGET)
}

/// The same, at a chosen band target.
///
/// The knob a rate loop turns, and the one the sweep in `tests/oracle.rs`
/// prices. Lower is coarser and cheaper; see [`DEFAULT_QUANT_TARGET`].
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode_frame_at(
    coefficients: &[f64],
    sample_rate: u32,
    target: i32,
) -> Result<Vec<u8>, EncodeError> {
    encode_frame_tuned(coefficients, sample_rate, target, DEFAULT_NOISE_FLATNESS)
}

/// The same, with the noise threshold as well.
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode_frame_tuned(
    coefficients: &[f64],
    sample_rate: u32,
    target: i32,
    flatness: f64,
) -> Result<Vec<u8>, EncodeError> {
    let bands = bands_for(sample_rate)?;
    let plan = plan_channel(coefficients, bands, target, flatness)?;

    let mut w = BitWriter::new();
    w.write(ID_SCE, 3);
    w.write(0, 4); // element_instance_tag
    write_channel(&mut w, &plan, bands, plan.coded_bands, true);
    w.write(ID_END, 3);
    Ok(w.finish())
}

/// Encode one block of each channel into a stereo `raw_data_block`.
///
/// The two channels share one `ics_info`, so they must agree on how many
/// bands are coded; the wider of the two wins, and the narrower simply
/// declares its extra bands empty. Sharing costs a few bits on one
/// channel and saves the whole structure on the other.
///
/// Mid/side is not used. It is a coding gain that only pays when
/// something decides which bands benefit, and that decision belongs to
/// the psychoacoustic model this encoder deliberately does not have yet.
pub fn encode_frame_pair(
    left: &[f64],
    right: &[f64],
    sample_rate: u32,
) -> Result<Vec<u8>, EncodeError> {
    encode_frame_pair_tuned(
        left,
        right,
        sample_rate,
        DEFAULT_QUANT_TARGET,
        DEFAULT_NOISE_FLATNESS,
    )
}

/// The same, with both levers exposed.
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode_frame_pair_tuned(
    left: &[f64],
    right: &[f64],
    sample_rate: u32,
    target: i32,
    flatness: f64,
) -> Result<Vec<u8>, EncodeError> {
    let bands = bands_for(sample_rate)?;
    let plans = [
        plan_channel(left, bands, target, flatness)?,
        plan_channel(right, bands, target, flatness)?,
    ];
    let coded_bands = plans[0].coded_bands.max(plans[1].coded_bands);

    let mut w = BitWriter::new();
    w.write(ID_CPE, 3);
    w.write(0, 4); // element_instance_tag
    w.write(1, 1); // common_window
    write_ics_info(&mut w, coded_bands);
    w.write(0, 2); // ms_mask_present: neither band-wise nor whole-frame
    for plan in &plans {
        write_channel(&mut w, plan, bands, coded_bands, false);
    }
    w.write(ID_END, 3);
    Ok(w.finish())
}

/// A section's band count, with the escape the field needs.
///
/// The all-ones value means "and more follows", so a run of exactly that
/// length must be written as all-ones then zero. Writing it plainly
/// would silently join this section to the next.
fn write_section_length(w: &mut BitWriter, bands: u32) {
    let escape = (1u32 << SECT_LEN_BITS) - 1;
    let mut left = bands;
    while left >= escape {
        w.write(escape, SECT_LEN_BITS);
        left -= escape;
    }
    w.write(left, SECT_LEN_BITS);
}

/// Encode a whole mono signal, one frame per 1024 new samples.
pub fn encode_mono(samples: &[f64], sample_rate: u32) -> Result<Vec<Vec<u8>>, EncodeError> {
    blocks_of(samples)
        .iter()
        .map(|block| encode_frame(block, sample_rate))
        .collect()
}

/// Encode two channels, one frame per 1024 new samples of each.
///
/// The channels must be the same length: a stereo frame carries one
/// block of each, and there is no sensible frame to write when one side
/// has run out.
pub fn encode_stereo(
    left: &[f64],
    right: &[f64],
    sample_rate: u32,
) -> Result<Vec<Vec<u8>>, EncodeError> {
    assert_eq!(
        left.len(),
        right.len(),
        "the two channels of a stereo signal must be the same length"
    );
    let (left, right) = (blocks_of(left), blocks_of(right));
    left.iter()
        .zip(&right)
        .map(|(l, r)| encode_frame_pair(l, r, sample_rate))
        .collect()
}

/// Window and transform a signal into the blocks a frame is built from.
///
/// The signal is padded at both ends: the first frame's overlap has
/// nothing before it, and the last needs something after it, so a
/// decoder's output begins and ends with a frame of ramp rather than
/// clipped signal.
fn blocks_of(samples: &[f64]) -> Vec<Vec<f64>> {
    let mut padded = vec![0.0; LONG_FRAME];
    padded.extend_from_slice(samples);
    padded.resize(padded.len().next_multiple_of(LONG_FRAME) + LONG_FRAME, 0.0);
    analyse(&padded, LONG_FRAME, WindowShape::Sine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;

    /// A signal with something in it, reproducibly.
    fn tone(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                0.4 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 3_000.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn a_rate_without_a_measured_band_table_is_refused() {
        // Encoding against another rate's boundaries would put every
        // scalefactor on the wrong coefficients — a stream that claims
        // to be fine and is not.
        let coefficients = vec![0.0; LONG_FRAME];
        for rate in [0u32, 47_999, 50_000, 192_000] {
            assert_eq!(
                encode_frame(&coefficients, rate),
                Err(EncodeError::UnmeasuredRate(rate)),
                "{rate}"
            );
        }
        // And every rate that *is* measured must encode.
        for rate in crate::bands::MEASURED_RATES {
            assert!(encode_frame(&coefficients, rate).is_ok(), "{rate}");
        }
    }

    #[test]
    fn a_silent_frame_codes_no_bands_at_all() {
        // Silence is the commonest thing an encoder sees, and `max_sfb`
        // stopping at zero is what makes it nearly free.
        let frame = encode_frame(&vec![0.0; LONG_FRAME], 48_000).expect("encodes");
        let mut r = BitReader::new(&frame);
        assert_eq!(r.u32(3), Some(ID_SCE));
        r.u32(4);
        r.u32(8);
        r.u32(1);
        assert_eq!(r.u32(2), Some(ONLY_LONG_SEQUENCE));
        r.u32(1);
        assert_eq!(r.u32(6), Some(0), "a silent frame codes no bands");
    }

    #[test]
    fn the_sections_cover_exactly_the_bands_the_frame_declares() {
        // The same check that proved this project could read ffmpeg's
        // frames, turned around onto its own. Sections that covered a
        // different number of bands would desynchronise every decoder.
        let blocks = crate::mdct::analyse(&tone(8192), LONG_FRAME, WindowShape::Sine);
        for (index, block) in blocks.iter().enumerate() {
            let frame = encode_frame(block, 48_000).expect("encodes");
            let mut r = BitReader::new(&frame);
            r.u32(3);
            r.u32(4);
            r.u32(8);
            r.u32(1);
            r.u32(2);
            r.u32(1);
            let max_sfb = r.u32(6).expect("max_sfb");
            r.u32(1);
            let mut covered = 0u32;
            while covered < max_sfb {
                r.u32(4).expect("sect_cb");
                loop {
                    let increment = r.u32(SECT_LEN_BITS).expect("sect_len");
                    covered += increment;
                    if increment != (1 << SECT_LEN_BITS) - 1 {
                        break;
                    }
                }
            }
            assert_eq!(covered, max_sfb, "block {index}");
        }
    }

    #[test]
    fn every_band_is_priced_before_a_book_is_chosen() {
        // The encoder must never pick a book that cannot carry a value,
        // because the write that follows has no way to refuse.
        for values in [
            vec![0, 0, 0, 0],
            vec![1, -1, 0, 1],
            vec![2, -2, 1, 0],
            vec![5, -4, 3, 2],
            vec![12, -12, 0, 7],
            vec![100, -3, 0, 0],
            vec![MAX_QUANT, -MAX_QUANT, 0, 0],
        ] {
            match choose_book(&values) {
                None => assert!(values.iter().all(|v| *v == 0), "{values:?}"),
                Some((book, cost)) => {
                    assert_eq!(band_cost(book, &values), Some(cost), "{values:?}");
                    assert!(cost > 0, "{values:?}");
                }
            }
        }
    }

    #[test]
    fn the_cheapest_book_is_the_one_chosen() {
        // Against every book that could carry the band, not just the
        // pair the encoder happens to look at first.
        for values in [
            vec![1, 0, 0, 1],
            vec![2, 2, -2, 2],
            vec![4, -4, 4, -4],
            vec![9, 0, -9, 0],
        ] {
            let (chosen, cost) = choose_book(&values).expect("not silent");
            for book in &BOOKS {
                if let Some(other) = band_cost(book, &values) {
                    assert!(
                        cost <= other,
                        "{values:?}: book {} costs {cost} but book {} costs {other}",
                        chosen.id,
                        book.id
                    );
                }
            }
        }
    }

    #[test]
    fn the_escape_book_can_carry_anything_the_quantiser_produces() {
        // The last resort has to actually be one. A band the escape book
        // could not carry would leave the encoder with no book at all.
        let escape = BOOKS.iter().find(|b| b.id == 11).expect("present");
        for magnitude in [16, 17, 31, 32, 63, 64, 1000, MAX_QUANT] {
            assert!(
                band_cost(escape, &[magnitude, -magnitude]).is_some(),
                "magnitude {magnitude}"
            );
        }
    }

    #[test]
    fn the_scalefactor_keeps_every_coefficient_inside_the_quantiser() {
        // One too small clips the loudest coefficient; the check is that
        // it never does, across amplitudes spanning ten decades and
        // every target a rate loop could ask for.
        for amplitude in [1e-6, 1e-3, 1.0, 100.0, 1e4, 1e6] {
            for target in [1, 15, DEFAULT_QUANT_TARGET, MAX_QUANT] {
                let coefficients = [amplitude, -amplitude / 3.0, amplitude / 100.0];
                let scalefactor = scalefactor_for(&coefficients, target);
                for coefficient in &coefficients {
                    assert!(
                        quantise_one(*coefficient, scalefactor).abs() <= MAX_QUANT,
                        "amplitude {amplitude} at target {target} clipped at {scalefactor}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_bands_loudest_coefficient_lands_near_the_target() {
        // The whole point of a per-band scalefactor: what a band costs
        // follows its own content, not its neighbour's loudness. If the
        // loudest coefficient did not land near the target, a quiet band
        // would still be quantised at the loud one's resolution.
        for amplitude in [1e-3, 1.0, 1e3, 1e6] {
            for target in [15, 63, DEFAULT_QUANT_TARGET, 2000] {
                let coefficients = [amplitude, amplitude / 8.0];
                let sf = scalefactor_for(&coefficients, target);
                let got = quantise_one(amplitude, sf);
                // A scalefactor is a quarter-power-of-two step, so the
                // best any integer one can do is within that step.
                let ratio = f64::from(got) / f64::from(target);
                assert!(
                    (0.7..=1.45).contains(&ratio),
                    "amplitude {amplitude}, target {target}: landed on {got}"
                );
            }
        }
    }

    #[test]
    fn a_noise_band_we_write_is_read_back_at_the_energy_we_meant() {
        // The writing side of a law walked out of a decoder in stage
        // 6.2, held to that decoder's own reading through ours. Flat
        // bands, so the criterion actually fires.
        let bands = bands_for(48_000).expect("measured");
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut coefficients = vec![0.0f64; LONG_FRAME];
        for value in coefficients.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = 0.05 * ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0);
        }
        let flatness = 0.15;
        let plan =
            plan_channel(&coefficients, bands, DEFAULT_QUANT_TARGET, flatness).expect("plans");
        let substituted = plan.books.iter().filter(|b| **b == NOISE_BOOK).count();
        assert!(substituted > 4, "only {substituted} bands were substituted");

        let frame = encode_frame_tuned(&coefficients, 48_000, DEFAULT_QUANT_TARGET, flatness)
            .expect("encodes");
        let read = crate::decode::decode_frame(&frame, 48_000).expect("decodes");
        assert_eq!(read[0].noise_bands, substituted, "a band changed its mind");

        for band in 0..plan.coded_bands {
            if plan.books[band] != NOISE_BOOK {
                continue;
            }
            let (from, to) = (bands[band], bands[band + 1]);
            let got: f64 = read[0].spectrum[from..to]
                .iter()
                .map(|c| c * c * COEFFICIENT_SCALE * COEFFICIENT_SCALE)
                .sum();
            let meant = 2f64.powf(f64::from(plan.noise[band].expect("stated")) / 2.0);
            // The stated value is a whole number of half-powers of two,
            // so the round trip cannot be exact — but it must be within
            // that step, or the law is being written wrongly.
            let difference = 10.0 * (got / meant).log10();
            assert!(
                difference.abs() < 0.01,
                "band {band}: carries {difference:+.3} dB against what it stated"
            );
        }
    }

    #[test]
    fn a_tone_is_never_mistaken_for_noise() {
        // The one thing the criterion must not do. A band with its
        // energy in one coefficient is a tone, and replacing it would be
        // plainly audible — so it must be refused at every threshold the
        // knob can take.
        let bands = bands_for(48_000).expect("measured");
        let mut coefficients = vec![0.0f64; LONG_FRAME];
        coefficients[bands[8]] = 0.5;
        coefficients[bands[12] + 1] = 0.3;
        for flatness in [0.9, 0.5, 0.3, 0.15, 0.05] {
            let plan =
                plan_channel(&coefficients, bands, DEFAULT_QUANT_TARGET, flatness).expect("plans");
            for band in 0..plan.coded_bands {
                assert_ne!(
                    plan.books[band], NOISE_BOOK,
                    "band {band} of a pure tone was replaced at flatness {flatness}"
                );
            }
        }
    }

    #[test]
    fn flatness_does_not_depend_on_how_wide_a_band_is() {
        // The defect the first attempt had. Peak over root mean square
        // is bounded by the square root of the width, so one threshold
        // meant "replace every narrow band" — measured, it destroyed a
        // pure tone as readily as noise.
        for width in [4usize, 8, 16, 32] {
            let flat = vec![1.0f64; width];
            assert!(
                (flatness_of(&flat) - 1.0).abs() < 1e-12,
                "width {width}: a flat band read {}",
                flatness_of(&flat)
            );
            let mut spike = vec![1e-6f64; width];
            spike[0] = 1.0;
            assert!(
                flatness_of(&spike) < 0.05,
                "width {width}: a spike read {}",
                flatness_of(&spike)
            );
        }
        // And a band with a hole in it is not the flat noise this is
        // looking for, whatever the rest of it does.
        assert_eq!(flatness_of(&[1.0, 1.0, 0.0, 1.0]), 0.0);
    }

    #[test]
    fn a_band_under_the_step_costs_nothing_at_all() {
        // Why one step for the whole frame beats a target per band, in
        // one test. A band sixty decibels down falls under the step and
        // is coded as empty — no scalefactor, no codewords, nothing. Per
        // band targets spend full bits on it, which is what the measured
        // sweep showed as a sixfold blowup.
        let bands = bands_for(48_000).expect("measured");
        let mut coefficients = vec![0.0f64; LONG_FRAME];
        coefficients[bands[2]..bands[3]].fill(0.5);
        coefficients[bands[20]..bands[21]].fill(0.5 / 1_000_000.0);
        let plan = plan_channel(
            &coefficients,
            bands,
            DEFAULT_QUANT_TARGET,
            DEFAULT_NOISE_FLATNESS,
        )
        .expect("plans");
        assert_ne!(plan.books[2], CODEBOOK_ZERO, "the loud band vanished");
        assert_eq!(plan.books[20], CODEBOOK_ZERO, "the quiet band was paid for");
    }

    #[test]
    fn the_target_is_a_knob_and_turning_it_changes_the_size() {
        // The knob this encoder did not have. Coarser must be smaller,
        // at every step, or it is not a rate control.
        let block = crate::mdct::analyse(&tone(4096), LONG_FRAME, WindowShape::Sine)[1].clone();
        // With noise substitution switched off, so this measures the
        // quantiser knob and nothing else: a band replaced by noise
        // costs the same nine bits however finely the rest is coded, and
        // that flattening is the *other* lever's business.
        let mut previous = 0usize;
        for target in [15, 63, 255, 1023, 4095] {
            let size = encode_frame_tuned(&block, 48_000, target, NEVER_SUBSTITUTE)
                .expect("encodes")
                .len();
            assert!(
                size > previous,
                "target {target} cost {size} bytes, no more than {previous}"
            );
            previous = size;
        }
    }

    #[test]
    fn scalefactors_that_differ_survive_the_delta_chain() {
        // The chain writes a difference per band and a decoder adds them
        // back up. Nothing this encoder decides today makes them differ
        // — but the machinery is what a psychoacoustic model will write
        // through, and an unexercised path is a path that rots.
        let bands = bands_for(48_000).expect("measured");
        let mut plan = plan_channel(
            &crate::mdct::analyse(&tone(4096), LONG_FRAME, WindowShape::Sine)[1],
            bands,
            DEFAULT_QUANT_TARGET,
            NEVER_SUBSTITUTE,
        )
        .expect("plans");
        assert!(plan.coded_bands > 3, "the fixture codes too little");
        // Walk the coded bands apart by a step the book can say, and
        // re-quantise so the plan stays self-consistent.
        let mut step = 0i32;
        for band in 0..plan.coded_bands {
            if plan.books[band] == CODEBOOK_ZERO {
                continue;
            }
            step = (step + 3) % 17;
            plan.scalefactors[band] += step - 8;
        }
        let mut w = BitWriter::new();
        w.write(ID_SCE, 3);
        w.write(0, 4);
        write_channel(&mut w, &plan, bands, plan.coded_bands, true);
        w.write(ID_END, 3);
        let frame = w.finish();

        // What a decoder makes of it must be what the plan said.
        let read = crate::decode::decode_frame(&frame, 48_000).expect("decodes");
        for (band, at) in bands.iter().enumerate().take(plan.coded_bands) {
            if plan.books[band] == CODEBOOK_ZERO {
                continue;
            }
            let at = *at;
            let want = crate::quant::dequantise_one(plan.quantised[at], plan.scalefactors[band])
                / COEFFICIENT_SCALE;
            assert_eq!(
                want.to_bits(),
                read[0].spectrum[at].to_bits(),
                "band {band} came back at a different scalefactor"
            );
        }
    }

    #[test]
    fn a_frame_is_whole_bytes_and_ends_with_the_terminator() {
        let blocks = crate::mdct::analyse(&tone(4096), LONG_FRAME, WindowShape::Sine);
        for block in &blocks {
            let frame = encode_frame(block, 48_000).expect("encodes");
            assert!(!frame.is_empty());
            // Walking to the terminator needs the whole parser, so this
            // checks the cheaper property: the frame is byte-aligned, so
            // the next one starts where a container says it does.
            assert_eq!(frame.len() * 8 % 8, 0);
        }
    }

    #[test]
    fn a_stereo_frame_describes_its_window_once_for_both_channels() {
        // With `common_window` set, the second channel has no `ics_info`
        // of its own. Writing one anyway would push everything after it
        // along by exactly that structure's width, and the decoder would
        // read the second channel's sections out of its scalefactors.
        let blocks = crate::mdct::analyse(&tone(8192), LONG_FRAME, WindowShape::Sine);
        let frame = encode_frame_pair(&blocks[2], &blocks[3], 48_000).expect("encodes");
        let mut r = BitReader::new(&frame);
        assert_eq!(r.u32(3), Some(ID_CPE));
        assert_eq!(r.u32(4), Some(0), "element_instance_tag");
        assert_eq!(r.u32(1), Some(1), "common_window");
        r.u32(1); // ics_reserved_bit
        assert_eq!(r.u32(2), Some(ONLY_LONG_SEQUENCE));
        r.u32(1); // window_shape
        let max_sfb = r.u32(6).expect("max_sfb");
        r.u32(1); // predictor_data_present
        assert_eq!(r.u32(2), Some(0), "ms_mask_present");

        // The first channel's sections must cover the shared band count
        // exactly. Reaching that agreement is what proves every field
        // before it is the right width. The second channel cannot be
        // walked without decoding the first one's spectrum, and the
        // round-trip gate covers it by decoding rather than by parsing.
        r.u32(8).expect("global_gain");
        let mut covered = 0u32;
        while covered < max_sfb {
            r.u32(4).expect("sect_cb");
            loop {
                let increment = r.u32(SECT_LEN_BITS).expect("sect_len");
                covered += increment;
                if increment != (1 << SECT_LEN_BITS) - 1 {
                    break;
                }
            }
        }
        assert_eq!(
            covered, max_sfb,
            "the sections do not cover the shared bands"
        );
    }

    #[test]
    fn the_shared_band_count_is_the_wider_of_the_two_channels() {
        // One loud channel and one silent one. The silent side must not
        // drag the shared count down, or the loud side loses its top
        // bands without anything saying so.
        let loud = crate::mdct::analyse(&tone(4096), LONG_FRAME, WindowShape::Sine)[1].clone();
        let silent = vec![0.0; LONG_FRAME];
        let bands = bands_for(48_000).expect("measured");
        let loud_plan = plan_channel(&loud, bands, DEFAULT_QUANT_TARGET, DEFAULT_NOISE_FLATNESS)
            .expect("plans");
        assert!(loud_plan.coded_bands > 0, "the fixture is not loud enough");

        for (left, right) in [(&loud, &silent), (&silent, &loud)] {
            let frame = encode_frame_pair(left, right, 48_000).expect("encodes");
            let mut r = BitReader::new(&frame);
            r.u32(3);
            r.u32(4);
            r.u32(1);
            r.u32(1);
            r.u32(2);
            r.u32(1);
            assert_eq!(
                r.u32(6),
                Some(loud_plan.coded_bands as u32),
                "the shared count followed the silent channel"
            );
        }
    }

    #[test]
    fn a_stereo_frame_is_bigger_than_either_channel_alone_and_smaller_than_both() {
        // Sharing the window costs a few bits on one channel and saves
        // the whole structure on the other. A pair that came out larger
        // than two separate frames would mean the sharing was not
        // happening at all.
        let blocks = crate::mdct::analyse(&tone(8192), LONG_FRAME, WindowShape::Sine);
        let (left, right) = (&blocks[2], &blocks[3]);
        let pair = encode_frame_pair(left, right, 48_000)
            .expect("encodes")
            .len();
        let separate = encode_frame(left, 48_000).expect("encodes").len()
            + encode_frame(right, 48_000).expect("encodes").len();
        assert!(
            pair > encode_frame(left, 48_000).expect("encodes").len(),
            "one channel's worth"
        );
        assert!(
            pair < separate,
            "a pair is {pair} bytes and two separate frames are {separate} — nothing was shared"
        );
    }

    #[test]
    fn stereo_needs_both_channels_to_be_the_same_length() {
        // A frame carries one block of each. There is no sensible frame
        // to write when one side has run out, and silently truncating
        // the longer one would drop audio without saying so.
        let result = std::panic::catch_unwind(|| encode_stereo(&tone(4096), &tone(2048), 48_000));
        assert!(result.is_err(), "mismatched lengths were accepted");
    }

    #[test]
    fn stereo_produces_one_frame_per_block_of_each_channel() {
        let left = tone(24_000);
        let right: Vec<f64> = left.iter().map(|s| -s * 0.5).collect();
        let frames = encode_stereo(&left, &right, 48_000).expect("encodes");
        let mono = encode_mono(&left, 48_000).expect("encodes");
        assert_eq!(
            frames.len(),
            mono.len(),
            "a pair is still one frame per block"
        );
    }

    #[test]
    fn a_signal_produces_a_frame_for_every_1024_samples_and_then_some() {
        // The overlap means the first and last frames carry a ramp
        // rather than signal, so there is always at least one more frame
        // than the sample count alone would suggest.
        let samples = tone(48_000);
        let frames = encode_mono(&samples, 48_000).expect("encodes");
        let expected = samples.len().div_ceil(LONG_FRAME);
        assert!(
            frames.len() > expected,
            "{} frames for {} samples is not enough to carry the overlap",
            frames.len(),
            samples.len()
        );
    }
}
