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

//! Reading an AAC `raw_data_block` back into coefficients.
//!
//! # Why a decoder, in a program that compresses
//!
//! A film with six AAC tracks at 640 kbit/s carries gigabytes of sound
//! that this project copies byte for byte, because re-encoding needs a
//! decoder and there was none. Copying is right for a track already as
//! small as it should be; it is not right for one at four times the
//! bitrate anything needs.
//!
//! # Everything it reads was measured
//!
//! The eleven spectral codebooks, the scalefactor book, the escape rule
//! and the band boundaries for all thirteen rates were probed out of a
//! working decoder rather than transcribed, and each is re-derived by a
//! gate that can fail. This module is the other side of the same
//! tables: it stores no numbers of its own, and even the mapping from a
//! book index back to the values it stands for is *derived* from how the
//! forward tables are indexed rather than written down a second time.
//!
//! # What it refuses, by name
//!
//! Short windows, mid/side stereo, intensity stereo, pulse data,
//! temporal noise shaping, gain control and channel coupling. Each is a
//! real part of AAC-LC that this project's encoder never produces, so
//! there is no file in hand that would prove a reader for it correct.
//! They arrive one at a time, each with a real file to gate it — and
//! until then a stream that needs one is refused by name, so it is a
//! known gap and not a mystery.
//!
//! Measured over real files, what is built reads **98%** of frames; the
//! remainder all want short windows.
//!
//! [`unsupported_element`] exists to find out which of them real files
//! actually need, so the order they are built in is measured rather than
//! assumed.

use std::sync::OnceLock;

use crate::bands::{long_bands, short_bands, LONG_FRAME, SHORT_FRAME, SHORT_WINDOWS};
use crate::bits::BitReader;
use crate::codebooks::{
    Codeword, ESCAPE_11, ESCAPE_MAGNITUDE, PAIR_10, PAIR_5, PAIR_6, PAIR_7, PAIR_8, PAIR_9, QUAD_1,
    QUAD_2, QUAD_3, QUAD_4, SCALEFACTOR_DELTA, SCALEFACTOR_DELTA_RANGE,
};
use crate::decode_tables::DecodeBook;
use crate::encode::COEFFICIENT_SCALE;
pub use crate::mdct::WindowSequence;
use crate::mdct::WindowShape;
use crate::quant::dequantise_one;

/// Element identifiers, as `raw_data_block` tags them.
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_CCE: u32 = 0x2;
const ID_LFE: u32 = 0x3;
const ID_DSE: u32 = 0x4;
const ID_PCE: u32 = 0x5;
const ID_FIL: u32 = 0x6;
const ID_END: u32 = 0x7;
/// The count field of a fill element, and the value meaning "and more".
const FIL_COUNT_BITS: u32 = 4;
const FIL_COUNT_ESCAPE: u32 = 15;
/// The same for a data stream element.
const DSE_COUNT_BITS: u32 = 8;
const DSE_COUNT_ESCAPE: u32 = 255;
/// Channel elements a block may hold before this stops reading.
///
/// The largest arrangement AAC allows is far below this; the bound is
/// here so a damaged block cannot make the loop the long part of a
/// decode.
const MAX_ELEMENTS: usize = 64;
/// Bits a long window spends on one section's band count, and the
/// narrower field an eight-short frame uses. Both measured: a sweep of
/// every width from one to eight found exactly one that let a frame
/// through, on each side.
const SECT_LEN_BITS: u32 = 5;
const SHORT_SECT_LEN_BITS: u32 = 3;
/// Bits an eight-short frame spends on `max_sfb` and on the grouping.
///
/// Measured together and then split. A frame declaring no bands at all
/// has nothing after the header, so the terminator's position gives
/// their *sum*: eleven, uniquely, out of every width from two to twenty.
/// Splitting it took a second sweep over a frame with one empty band,
/// where exactly one combination out of five hundred read cleanly —
/// four and seven, with sections three bits wide and repeated once per
/// group.
///
/// Note what is *absent*: a long frame spends a bit on
/// `predictor_data_present` and a short one does not. Had it, the clean
/// sum would have been twelve.
const SHORT_MAX_SFB_BITS: u32 = 4;
const SHORT_GROUPING_BITS: u32 = 7;
/// The codebook meaning "this band is entirely zero".
const CODEBOOK_ZERO: u32 = 0;
/// Spectral codebook numbers, lowest and highest.
const FIRST_SPECTRAL_BOOK: u32 = 1;
const LAST_SPECTRAL_BOOK: u32 = 11;
/// The codebook meaning "this band is noise of a stated energy".
pub(crate) const NOISE_BOOK: u32 = 13;
/// Bits the first noise band of a channel spends stating its energy.
///
/// Measured, bracketed on both sides: at eight the decoder runs out of
/// frame before the terminator, at ten it finds an element that is not
/// there, and nine is the only width it reads without complaint.
pub(crate) const NOISE_ENERGY_BITS: u32 = 9;
/// What a noise band's energy counts down from.
///
/// The band's total squared magnitude is two to the half of
/// `global_gain + field - NOISE_ENERGY_OFFSET`. The law was walked out
/// of a decoder over twenty steps of gain and forty of the field, and
/// the constant came out at **410.008** in the scale a decoder hands
/// back — flat to three decimals across every combination, which is what
/// a law looks like when it is one. This is that constant in the units
/// the format counts in, which differ by [`COEFFICIENT_SCALE`] squared:
/// `410 - 2 * log2(65536)`.
pub(crate) const NOISE_ENERGY_OFFSET: i32 = 346;
/// The range that energy is held to.
///
/// Measured at both edges: below the bottom every value gives the same
/// quietest noise, above the top the same loudest. And the clamp applies
/// where the energy is *used*, not to the running value a later band
/// steps from — settled by pushing one band twenty past the top and
/// stepping the next sixteen down, which lands where an unclamped
/// running value would and not where a clamped one would.
pub(crate) const NOISE_ENERGY_RANGE: (i32, i32) = (-100, 155);
/// The longest escape prefix worth reading.
///
/// The quantiser tops out at 8191, whose escape carries a prefix of
/// seven — so a stream claiming far more than that has lost its place,
/// and a reader that kept counting would spin through a damaged file
/// instead of reporting it (bounded execution).
const MAX_ESCAPE_PREFIX: u32 = 16;

/// Why a frame could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The bits ran out mid-field, which is what a damaged frame looks
    /// like. Completing it would hand a listener sound the file never
    /// contained.
    Truncated(&'static str),
    /// No band table has been measured for this sampling rate.
    UnmeasuredRate(u32),
    /// A part of the format this reader does not implement, named.
    Unsupported(&'static str),
    /// Bits that are not a codeword of the book the section named.
    ///
    /// A detectable fault. Guessing the nearest match would turn it into
    /// audible noise that nothing reports.
    BadCodeword {
        /// Which book was being read, or zero for the scalefactor book.
        codebook: u32,
        /// Which band it was reading.
        band: usize,
    },
}

/// One channel's spectrum, in the scale a decoder hands back — the same
/// scale [`crate::encode::encode_frame`] takes and [`crate::mdct`] works
/// in, so a decoded frame goes straight back through synthesis.
pub type Spectrum = Vec<f64>;

/// One channel of one frame.
///
/// The window shape travels with the spectrum because synthesis cannot
/// be done without it, and a caller that had to fetch it separately
/// would eventually fetch the wrong one. Measured: reading real frames
/// with the wrong shape agrees with their own decoder to 16 dB, and with
/// the right one to over eighty — a difference that sounds like nothing
/// in particular and is still wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// Dequantised coefficients, one per spectral position.
    pub spectrum: Spectrum,
    /// The shape this block's window ends with.
    pub window: WindowShape,
    /// How the frame's window is laid out.
    pub sequence: WindowSequence,
    /// For an eight-short frame, how many windows each group holds; a
    /// single entry of one for the other three, which are one window.
    ///
    /// The grouping is what says which of the eight windows shared a set
    /// of scalefactors, and a caller re-encoding has to keep it or the
    /// frame it writes will not be the frame it read.
    pub groups: Vec<usize>,
    /// How many of its bands were filled with noise rather than read.
    ///
    /// Those bands carry an energy and nothing else, so two correct
    /// decoders disagree on their contents by design. A caller comparing
    /// against another decoder needs to know that before it draws a
    /// conclusion, and a caller re-encoding needs to know it before it
    /// promises the sound is unchanged.
    pub noise_bands: usize,
}

/// The reverse tables, built once for the life of the process.
///
/// Twelve hash maps rebuilt per frame would put four thousand
/// allocations on the hot path of every 23 milliseconds of audio
///. They are immutable once built, so one copy serves every
/// thread.
struct Books {
    /// The eleven spectral books, in `sect_cb` order from one, each with
    /// the magnitude it carries outright and the coefficients one
    /// codeword covers.
    spectral: Vec<(DecodeBook, i32, usize)>,
    /// The scalefactor delta book.
    scalefactor: DecodeBook,
}

/// Every spectral book's forward table, with its range and tuple width.
const SPECTRAL_TABLES: [(&[Codeword], i32, usize); 11] = [
    (&QUAD_1, 1, 4),
    (&QUAD_2, 1, 4),
    (&QUAD_3, 2, 4),
    (&QUAD_4, 2, 4),
    (&PAIR_5, 4, 2),
    (&PAIR_6, 4, 2),
    (&PAIR_7, 7, 2),
    (&PAIR_8, 7, 2),
    (&PAIR_9, 12, 2),
    (&PAIR_10, 12, 2),
    (&ESCAPE_11, ESCAPE_MAGNITUDE, 2),
];

/// The reverse tables, turned around at first use.
fn books() -> &'static Books {
    static BOOKS: OnceLock<Books> = OnceLock::new();
    BOOKS.get_or_init(|| Books {
        spectral: SPECTRAL_TABLES
            .iter()
            .map(|(table, range, carries)| (DecodeBook::from_forward(table), *range, *carries))
            .collect(),
        scalefactor: DecodeBook::from_forward(&SCALEFACTOR_DELTA),
    })
}

/// Turn a book index back into the values it stands for.
///
/// The forward tables are indexed by tuple, most significant value
/// first, so an index is a mixed-radix number and this is its expansion.
/// Deriving it beats storing a second table that could disagree with the
/// first. Only the leading `carries` entries mean anything.
fn values_of(index: usize, range: i32, carries: usize) -> [i32; 4] {
    let side = (2 * range + 1) as usize;
    let mut out = [0i32; 4];
    let mut left = index;
    for slot in out[..carries].iter_mut().rev() {
        *slot = (left % side) as i32 - range;
        left /= side;
    }
    out
}

/// Read one `raw_data_block` into dequantised spectra, one per channel.
///
/// # Errors
///
/// See [`DecodeError`]. Every variant names what was wrong and where.
pub fn decode_frame(frame: &[u8], sample_rate: u32) -> Result<Vec<Channel>, DecodeError> {
    let mut r = BitReader::new(frame);
    let mut spectra = Vec::new();
    for _ in 0..MAX_ELEMENTS {
        match r.u32(3).ok_or(DecodeError::Truncated("element id"))? {
            ID_END => return Ok(spectra),
            // A single channel, and the low-frequency channel that is
            // syntactically the same thing. Reading them alike is not a
            // guess about the format: the difference is which speaker
            // the sound goes to, which is the container's business.
            ID_SCE | ID_LFE => {
                r.u32(4).ok_or(DecodeError::Truncated("element tag"))?;
                let mut noise = Noise::seeded(frame, spectra.len());
                spectra.push(read_channel(&mut r, sample_rate, None, &mut noise)?);
            }
            ID_CPE => {
                r.u32(4).ok_or(DecodeError::Truncated("element tag"))?;
                // A pair describes its window once, for both channels.
                if r.u32(1).ok_or(DecodeError::Truncated("common_window"))? != 1 {
                    return Err(DecodeError::Unsupported("a pair with two separate windows"));
                }
                let shared = read_ics_info(&mut r)?;
                if r.u32(2).ok_or(DecodeError::Truncated("ms_mask_present"))? != 0 {
                    return Err(DecodeError::Unsupported("mid/side stereo"));
                }
                for _ in 0..2 {
                    let mut noise = Noise::seeded(frame, spectra.len());
                    spectra.push(read_channel(
                        &mut r,
                        sample_rate,
                        Some(shared.clone()),
                        &mut noise,
                    )?);
                }
            }
            // Elements carrying no sound of their own. Skipping them is
            // not ignoring them: their length has to be read exactly, or
            // everything after is misaligned.
            ID_FIL => skip_fill(&mut r)?,
            ID_DSE => skip_data(&mut r)?,
            ID_CCE => return Err(DecodeError::Unsupported("channel coupling")),
            ID_PCE => return Err(DecodeError::Unsupported("an in-band program config")),
            other => unreachable!("three bits cannot hold {other}"),
        }
    }
    Err(DecodeError::Unsupported("more elements than a block holds"))
}

/// Step over a fill element.
///
/// Its payload is a count of bytes, and whatever those bytes describe —
/// an encoder's name, or the spectral band replication this does not
/// read — they are that many. Nothing here interprets them; the length
/// is all that the rest of the block depends on.
fn skip_fill(r: &mut BitReader<'_>) -> Result<(), DecodeError> {
    let mut count = r
        .u32(FIL_COUNT_BITS)
        .ok_or(DecodeError::Truncated("fill count"))?;
    if count == FIL_COUNT_ESCAPE {
        let more = r
            .u32(8)
            .ok_or(DecodeError::Truncated("fill count escape"))?;
        count = count + more - 1;
    }
    r.skip(count * 8)
        .ok_or(DecodeError::Truncated("fill payload"))
}

/// Step over a data stream element, the same way.
fn skip_data(r: &mut BitReader<'_>) -> Result<(), DecodeError> {
    r.u32(4).ok_or(DecodeError::Truncated("data tag"))?;
    let aligned = r.u32(1).ok_or(DecodeError::Truncated("data align flag"))?;
    let mut count = r
        .u32(DSE_COUNT_BITS)
        .ok_or(DecodeError::Truncated("data count"))?;
    if count == DSE_COUNT_ESCAPE {
        let more = r
            .u32(8)
            .ok_or(DecodeError::Truncated("data count escape"))?;
        count += more;
    }
    if aligned == 1 {
        r.align();
    }
    r.skip(count * 8)
        .ok_or(DecodeError::Truncated("data payload"))
}

/// Put a run of decoded frames back together into sound.
///
/// Every window sequence, including the eight-short frames a real file
/// uses at every transient and the two asymmetric ones that meet them.
/// The shapes and the layout come out of the bitstream and the hop
/// arithmetic; nothing is assumed, and the gate that judges the result
/// is our reading of somebody else's file against their own decoder.
pub fn synthesise(channels: &[Channel]) -> Vec<f64> {
    let spectra: Vec<Vec<f64>> = channels.iter().map(|c| c.spectrum.clone()).collect();
    let sequences: Vec<WindowSequence> = channels.iter().map(|c| c.sequence).collect();
    let shapes: Vec<WindowShape> = channels.iter().map(|c| c.window).collect();
    crate::mdct::synthesise_sequenced(&spectra, &sequences, &shapes)
}

/// Which part of the format a frame needs that this reader lacks, if any.
///
/// Pointed at a real file's frames, this says what to build next and in
/// what order — measured from the files that exist rather than guessed
/// from the specification's table of contents. `None` means
/// the frame decoded, or failed for a reason other than a missing
/// feature.
pub fn unsupported_element(frame: &[u8], sample_rate: u32) -> Option<&'static str> {
    match decode_frame(frame, sample_rate) {
        Err(DecodeError::Unsupported(what)) => Some(what),
        _ => None,
    }
}

/// What an `ics_info` says about a frame's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IcsInfo {
    coded_bands: usize,
    window: WindowShape,
    sequence: WindowSequence,
    /// Windows per group, in order; one entry of eight unless the frame
    /// is eight-short.
    groups: Vec<usize>,
}

/// Read an `ics_info`.
fn read_ics_info(r: &mut BitReader<'_>) -> Result<IcsInfo, DecodeError> {
    r.u32(1).ok_or(DecodeError::Truncated("ics_reserved_bit"))?;
    let sequence = match r.u32(2).ok_or(DecodeError::Truncated("window_sequence"))? {
        0 => WindowSequence::OnlyLong,
        1 => WindowSequence::LongStart,
        2 => WindowSequence::EightShort,
        _ => WindowSequence::LongStop,
    };
    let window = match r.u32(1).ok_or(DecodeError::Truncated("window_shape"))? {
        0 => WindowShape::Sine,
        _ => WindowShape::Kbd,
    };
    if sequence == WindowSequence::EightShort {
        let max_sfb = r
            .u32(SHORT_MAX_SFB_BITS)
            .ok_or(DecodeError::Truncated("max_sfb"))? as usize;
        let grouping = r
            .u32(SHORT_GROUPING_BITS)
            .ok_or(DecodeError::Truncated("scale_factor_grouping"))?;
        return Ok(IcsInfo {
            coded_bands: max_sfb,
            window,
            sequence,
            groups: window_groups(grouping),
        });
    }
    let max_sfb = r.u32(6).ok_or(DecodeError::Truncated("max_sfb"))?;
    if r.u32(1).ok_or(DecodeError::Truncated("predictor_data"))? != 0 {
        return Err(DecodeError::Unsupported("backward prediction"));
    }
    Ok(IcsInfo {
        coded_bands: max_sfb as usize,
        window,
        sequence,
        // A long frame is one window covering the whole span.
        groups: vec![1],
    })
}

/// How the seven grouping bits partition eight windows.
///
/// Measured, on all one hundred and twenty-eight values: a decoder
/// demands one section list per group, and the count it demands is
/// always eight less the bits set. So each bit joins a window to the
/// group before it, most significant bit first — the first window always
/// opens a group, which is why seven bits describe eight windows.
fn window_groups(grouping: u32) -> Vec<usize> {
    let mut groups = vec![1usize];
    for window in 1..SHORT_WINDOWS {
        let joins = (grouping >> (SHORT_WINDOWS - 1 - window)) & 1 == 1;
        if joins {
            *groups.last_mut().expect("seeded with one") += 1;
        } else {
            groups.push(1);
        }
    }
    groups
}

/// Read the `section_data`: runs of bands sharing a codebook.
fn read_sections(
    r: &mut BitReader<'_>,
    coded_bands: usize,
    length_bits: u32,
) -> Result<Vec<u32>, DecodeError> {
    let mut used = vec![CODEBOOK_ZERO; coded_bands];
    let escape = (1u32 << length_bits) - 1;
    let mut at = 0usize;
    while at < coded_bands {
        let codebook = r.u32(4).ok_or(DecodeError::Truncated("sect_cb"))?;
        if codebook > LAST_SPECTRAL_BOOK && codebook != NOISE_BOOK {
            // Twelve is reserved; fourteen and fifteen are intensity
            // stereo, which is a channel pair's business and not read
            // here.
            return Err(DecodeError::Unsupported("intensity stereo"));
        }
        // The all-ones length means "and more follows", so a run of
        // exactly that is written as all-ones then zero.
        let mut run = 0usize;
        loop {
            let increment = r
                .u32(length_bits)
                .ok_or(DecodeError::Truncated("sect_len"))?;
            run += increment as usize;
            if increment != escape {
                break;
            }
            if at + run > coded_bands {
                // Bounded: a run of all-ones would otherwise count
                // forever, and it cannot be describing this frame.
                return Err(DecodeError::Unsupported("a section past the last band"));
            }
        }
        if run == 0 {
            return Err(DecodeError::Unsupported("a section covering no bands"));
        }
        for band in used.iter_mut().skip(at).take(run) {
            *band = codebook;
        }
        at += run;
    }
    Ok(used)
}

/// Read the `scale_factor_data`.
///
/// One Huffman-coded delta per coded band, each relative to the band
/// before it and the first to `global_gain` — except for noise bands,
/// which state an energy instead and keep a running value entirely of
/// their own.
///
/// That independence is measured, not assumed: a noise band placed
/// between two ordinary ones leaves both of their magnitudes exactly
/// where they were without it. Had they shared, every band after a noise
/// band would shift — silent corruption of the sort that sounds like
/// somebody's mixing decision.
/// The two running values a channel's scalefactors step from.
///
/// They carry on across the groups of a short frame rather than
/// restarting, which is most of what grouping buys — and they are
/// independent of each other, which was measured.
struct ScalefactorState {
    /// The ordinary running scalefactor.
    running: i32,
    /// The running noise energy. Where it starts and what the first
    /// field subtracts are not separately observable, so they are one
    /// measured constant applied at the end.
    noise: i32,
    /// Whether the next noise band is the channel's first.
    first_noise: bool,
}

impl ScalefactorState {
    fn new(global_gain: i32) -> ScalefactorState {
        ScalefactorState {
            running: global_gain,
            noise: global_gain,
            first_noise: true,
        }
    }
}

fn read_scalefactors(
    r: &mut BitReader<'_>,
    used: &[u32],
    state: &mut ScalefactorState,
) -> Result<Vec<i32>, DecodeError> {
    let book = &books().scalefactor;
    let mut values = vec![state.running; used.len()];
    for (band, codebook) in used.iter().enumerate() {
        if *codebook == CODEBOOK_ZERO {
            continue;
        }
        if *codebook == NOISE_BOOK {
            if state.first_noise {
                state.first_noise = false;
                state.noise +=
                    r.u32(NOISE_ENERGY_BITS)
                        .ok_or(DecodeError::Truncated("noise energy"))? as i32
                        - NOISE_ENERGY_OFFSET;
            } else {
                // Every noise band after the first costs what an
                // ordinary scalefactor costs, stepping from the last
                // noise band — measured at deltas of nought and either
                // sign.
                let (index, length) =
                    book.read(|want| r.peek(u32::from(want)))
                        .ok_or(DecodeError::BadCodeword {
                            codebook: NOISE_BOOK,
                            band,
                        })?;
                r.u32(u32::from(length))
                    .ok_or(DecodeError::Truncated("noise energy delta"))?;
                state.noise += index as i32 - SCALEFACTOR_DELTA_RANGE;
            }
            values[band] = state.noise;
            continue;
        }
        let (index, length) = book
            .read(|want| r.peek(u32::from(want)))
            .ok_or(DecodeError::BadCodeword { codebook: 0, band })?;
        r.u32(u32::from(length))
            .ok_or(DecodeError::Truncated("scalefactor delta"))?;
        state.running += index as i32 - SCALEFACTOR_DELTA_RANGE;
        values[band] = state.running;
    }
    Ok(values)
}

/// A deterministic source of noise for the bands that state only an
/// energy.
///
/// Perceptual noise substitution says what a band's energy is and
/// nothing about its contents, because the point is that nobody can hear
/// the difference. Which noise, then, is the decoder's own business —
/// and the one property worth insisting on is that it is *always the
/// same noise for the same file*, so a re-run of this program produces
/// the byte-identical result the rest of it promises.
///
/// Seeded from the frame's own bytes rather than carried between frames:
/// a decoder that reset per frame would emit the same noise every 23
/// milliseconds, and identical blocks overlap-added at a fixed period
/// are a tone, not noise.
struct Noise(u64);

impl Noise {
    /// Seed from a frame, mixing in which channel of it this is.
    fn seeded(frame: &[u8], channel: usize) -> Noise {
        // Fowler–Noll–Vo, for no reason beyond being short and spreading
        // single-byte changes across the whole word.
        const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut state = BASIS ^ (channel as u64).wrapping_mul(PRIME);
        for byte in frame {
            state = (state ^ u64::from(*byte)).wrapping_mul(PRIME);
        }
        // A zero state is a fixed point of the generator below.
        Noise(state | 1)
    }

    /// The next value, in minus one to one.
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // The top fifty-three bits are the ones a double can hold.
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Fill a band with noise of the stated energy.
///
/// The energy is what the bitstream says; the scaling makes it so
/// exactly, which is the only part a listener could ever check.
fn fill_with_noise(band: &mut [f64], stated: i32, noise: &mut Noise) {
    let (low, high) = NOISE_ENERGY_RANGE;
    let target = 2f64.powf(f64::from(stated.clamp(low, high)) / 2.0);
    let mut energy = 0.0f64;
    for value in band.iter_mut() {
        *value = noise.next();
        energy += *value * *value;
    }
    if energy <= 0.0 {
        return;
    }
    let scale = (target / energy).sqrt() / COEFFICIENT_SCALE;
    for value in band.iter_mut() {
        *value *= scale;
    }
}

/// Read the escape a magnitude of [`ESCAPE_MAGNITUDE`] promised: `n`
/// ones, a zero, then `n + 4` bits holding the value less two to the
/// `n + 4`.
fn read_escape(r: &mut BitReader<'_>) -> Result<i32, DecodeError> {
    let mut ones = 0u32;
    while r.u32(1).ok_or(DecodeError::Truncated("escape prefix"))? == 1 {
        ones += 1;
        if ones > MAX_ESCAPE_PREFIX {
            return Err(DecodeError::Unsupported(
                "an escape longer than the quantiser can reach",
            ));
        }
    }
    let width = ones + 4;
    let payload = r
        .u32(width)
        .ok_or(DecodeError::Truncated("escape payload"))?;
    Ok(((1u32 << width) + payload) as i32)
}

/// Read one `individual_channel_stream`.
fn read_channel(
    r: &mut BitReader<'_>,
    rate: u32,
    shared: Option<IcsInfo>,
    noise: &mut Noise,
) -> Result<Channel, DecodeError> {
    let global_gain = r.u32(8).ok_or(DecodeError::Truncated("global_gain"))? as i32;
    let info = match shared {
        Some(info) => info,
        None => read_ics_info(r)?,
    };
    let short = info.sequence == WindowSequence::EightShort;
    // A short frame's bands are a different table with a different
    // grouping of rates. Reaching for the long one because the rate was
    // to hand would place every band somewhere else.
    let bands = if short {
        short_bands(rate).ok_or(DecodeError::UnmeasuredRate(rate))?
    } else {
        long_bands(rate).ok_or(DecodeError::UnmeasuredRate(rate))?
    };
    let coded_bands = info.coded_bands;
    if coded_bands + 1 > bands.len() {
        return Err(DecodeError::Unsupported("more bands than the rate has"));
    }
    let length_bits = if short {
        SHORT_SECT_LEN_BITS
    } else {
        SECT_LEN_BITS
    };

    // One section list per group, and one scalefactor per group per
    // band — the running values carry on across groups rather than
    // restarting, which is what makes a grouped frame cheaper.
    let mut used = Vec::with_capacity(info.groups.len());
    for _ in &info.groups {
        used.push(read_sections(r, coded_bands, length_bits)?);
    }
    let mut state = ScalefactorState::new(global_gain);
    let mut scalefactors = Vec::with_capacity(info.groups.len());
    for group in &used {
        scalefactors.push(read_scalefactors(r, group, &mut state)?);
    }

    for (field, what) in [
        ("pulse_data_present", "pulse data"),
        ("tns_data_present", "temporal noise shaping"),
        ("gain_control_data_present", "gain control"),
    ] {
        if r.u32(1).ok_or(DecodeError::Truncated(field))? != 0 {
            return Err(DecodeError::Unsupported(what));
        }
    }

    // A short frame's eight windows sit one after another, each holding
    // its own 128 coefficients — measured by placing a unit in each
    // codeword of a single group in turn and asking a decoder *when* the
    // sound arrived. It walked window by window, in order.
    let mut spectrum = vec![0.0f64; LONG_FRAME];
    let mut noise_bands = 0usize;
    let stride = if short { SHORT_FRAME } else { LONG_FRAME };
    let mut first_window = 0usize;
    for (group, width) in info.groups.iter().enumerate() {
        for (band, codebook) in used[group].iter().enumerate() {
            if *codebook == CODEBOOK_ZERO {
                continue;
            }
            let (from, to) = (bands[band], bands[band + 1]);
            let stated = scalefactors[group][band];
            // One run covers this band across every window of the group.
            for window in first_window..first_window + width {
                let base = window * stride;
                if *codebook == NOISE_BOOK {
                    // No spectral data at all: the band is noise at the
                    // energy its scalefactor field stated.
                    fill_with_noise(&mut spectrum[base + from..base + to], stated, noise);
                    noise_bands += 1;
                    continue;
                }
                read_band(
                    r,
                    &mut spectrum[base + from..base + to],
                    *codebook,
                    band,
                    stated,
                )?;
            }
        }
        first_window += width;
    }
    Ok(Channel {
        spectrum,
        window: info.window,
        sequence: info.sequence,
        groups: info.groups,
        noise_bands,
    })
}

/// Read one band's coefficients with the codebook its section named.
fn read_band(
    r: &mut BitReader<'_>,
    out: &mut [f64],
    codebook: u32,
    band: usize,
    scalefactor: i32,
) -> Result<(), DecodeError> {
    let (book, range, carries) = &books().spectral[(codebook - FIRST_SPECTRAL_BOOK) as usize];
    let mut at = 0usize;
    while at < out.len() {
        let (word, length) = book
            .read(|want| r.peek(u32::from(want)))
            .ok_or(DecodeError::BadCodeword { codebook, band })?;
        r.u32(u32::from(length))
            .ok_or(DecodeError::Truncated("spectral codeword"))?;
        let mut values = values_of(word, *range, *carries);
        if codebook == LAST_SPECTRAL_BOOK {
            for value in values[..*carries].iter_mut() {
                if value.abs() < ESCAPE_MAGNITUDE {
                    continue;
                }
                let magnitude = read_escape(r)?;
                *value = if *value < 0 { -magnitude } else { magnitude };
            }
        }
        for value in &values[..*carries] {
            if at >= out.len() {
                // The last codeword of a band whose width the tuple does
                // not divide carries padding past its end.
                break;
            }
            out[at] = dequantise_one(*value, scalefactor) / COEFFICIENT_SCALE;
            at += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{
        encode_frame, encode_frame_tuned, intended_spectrum, DEFAULT_QUANT_TARGET, NEVER_SUBSTITUTE,
    };
    use crate::mdct::{analyse, synthesise, WindowShape};

    const RATE: u32 = 48_000;

    /// A tone, a harmonic and a periodic transient: quiet bands the
    /// encoder will drop, loud ones it will spend the escape book on.
    fn tone(len: usize, hz: f64) -> Vec<f64> {
        (0..len)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                let envelope = if i % 3000 < 40 { 0.9 } else { 0.35 };
                envelope
                    * ((std::f64::consts::TAU * hz * t).sin()
                        + 0.3 * (std::f64::consts::TAU * hz * 7.3 * t).sin())
            })
            .collect()
    }

    fn blocks(hz: f64) -> Vec<Vec<f64>> {
        analyse(&tone(8192, hz), LONG_FRAME, WindowShape::Sine)
    }

    #[test]
    fn a_frame_reads_back_to_exactly_what_the_encoder_meant() {
        // Not "close to the input" — quantisation makes that a
        // tolerance, and a tolerance would hide a wrong table. This is
        // the encoder's own decision, reproduced coefficient for
        // coefficient by walking the same measured books backwards.
        // With substitution off, because a noise band's contents are
        // noise: what the two sides must agree on there is an energy,
        // and that is gated on its own.
        let mut loud = 0usize;
        for block in blocks(440.0) {
            let frame = encode_frame_tuned(&block, RATE, DEFAULT_QUANT_TARGET, NEVER_SUBSTITUTE)
                .expect("encodes");
            let spectra = decode_frame(&frame, RATE).expect("decodes");
            assert_eq!(spectra.len(), 1, "a mono element has one channel");
            let meant = intended_spectrum(&block, RATE).expect("plans");
            assert_eq!(spectra[0].spectrum.len(), meant.len());
            for (i, (want, got)) in meant.iter().zip(&spectra[0].spectrum).enumerate() {
                assert_eq!(want.to_bits(), got.to_bits(), "coefficient {i}");
            }
            loud += meant
                .iter()
                .filter(|c| c.abs() * COEFFICIENT_SCALE > 1000.0)
                .count();
        }
        // A pass over silence would prove nothing: without this the
        // whole test could be zeroes agreeing with zeroes.
        assert!(loud > 0, "the signal never reached the larger books");
    }

    /// Lay out a probe frame by hand: a run of empty bands, then one
    /// section of a chosen codebook, with the scalefactor field supplied
    /// verbatim.
    fn probe(codebook: u32, band: u32, gain: u32, scalefactor: &[bool]) -> Vec<u8> {
        let mut w = crate::bits::BitWriter::new();
        w.write(ID_SCE, 3);
        w.write(0, 4);
        w.write(gain, 8);
        w.write(0, 1);
        w.write(0, 2); // ONLY_LONG_SEQUENCE
        w.write(0, 1);
        w.write(band + 1, 6);
        w.write(0, 1);
        if band > 0 {
            w.write(CODEBOOK_ZERO, 4);
            w.write(band, SECT_LEN_BITS);
        }
        w.write(codebook, 4);
        w.write(1, SECT_LEN_BITS);
        for bit in scalefactor {
            w.write(u32::from(*bit), 1);
        }
        w.write(0, 3); // pulse, tns, gain control
        w.write(ID_END, 3);
        w.finish()
    }

    /// The nine bits of a noise energy field, most significant first.
    fn energy_field(value: u32) -> Vec<bool> {
        (0..NOISE_ENERGY_BITS)
            .map(|i| (value >> (NOISE_ENERGY_BITS - 1 - i)) & 1 == 1)
            .collect()
    }

    fn band_energy(channel: &Channel, band: usize) -> f64 {
        let bands = crate::bands::long_bands(RATE).expect("a measured rate");
        channel.spectrum[bands[band]..bands[band + 1]]
            .iter()
            .map(|c| c * c)
            .sum()
    }

    /// The band this project's noise probes use: high enough to be wide,
    /// clear of the first few.
    const NOISE_PROBE_BAND: u32 = 20;

    #[test]
    fn a_noise_band_holds_exactly_the_energy_the_bitstream_stated() {
        // The law was walked out of a decoder over twenty steps of gain
        // and forty of the field. This holds the implementation to it at
        // both knobs, because a constant fitted to one of them cannot
        // tell which it really depends on.
        for (gain, value) in [
            (100u32, 340u32),
            (100, 360),
            (100, 380),
            (90, 380),
            (110, 380),
        ] {
            let frame = probe(NOISE_BOOK, NOISE_PROBE_BAND, gain, &energy_field(value));
            let channels = decode_frame(&frame, RATE).expect("decodes");
            assert_eq!(channels[0].noise_bands, 1, "the band was not substituted");
            let energy = band_energy(&channels[0], NOISE_PROBE_BAND as usize)
                * COEFFICIENT_SCALE
                * COEFFICIENT_SCALE;
            let stated = f64::from(gain as i32 + value as i32 - NOISE_ENERGY_OFFSET) / 2.0;
            assert!(
                (energy.log2() - stated).abs() < 1e-9,
                "gain {gain}, value {value}: 2^{:.6} where the frame said 2^{stated:.6}",
                energy.log2()
            );
        }
    }

    #[test]
    fn a_stated_energy_past_either_edge_is_held_to_the_edge() {
        // Measured at both ends: past the top every value gives the same
        // loudest noise, and past the bottom the same quietest. A reader
        // that let the exponent run would produce infinities from a file
        // a real decoder plays.
        let (low, high) = NOISE_ENERGY_RANGE;
        let at = |gain: u32, value: u32| {
            let frame = probe(NOISE_BOOK, NOISE_PROBE_BAND, gain, &energy_field(value));
            let channels = decode_frame(&frame, RATE).expect("decodes");
            band_energy(&channels[0], NOISE_PROBE_BAND as usize).log2()
        };
        // The field value that states a given energy at a given gain.
        // At gain 100 the top lands on 401, which is exactly where the
        // decoder that was walked stopped rising.
        const GAIN: u32 = 100;
        let field = |nrg: i32| (nrg + NOISE_ENERGY_OFFSET - GAIN as i32) as u32;
        assert_eq!(field(high), 401, "the top of the range moved");
        assert_eq!(at(GAIN, field(high)), at(GAIN, field(high) + 8), "the top");
        assert_eq!(at(GAIN, field(low)), at(GAIN, field(low) - 8), "the bottom");
        assert!(
            at(GAIN, field(high)) > at(GAIN, field(low)),
            "the edges are the same"
        );
    }

    #[test]
    fn the_same_file_always_decodes_to_the_same_noise() {
        // The one property worth insisting on for content nobody can
        // check: a re-run produces the byte-identical result the rest of
        // this program promises.
        let frame = probe(NOISE_BOOK, NOISE_PROBE_BAND, 100, &energy_field(380));
        let once = decode_frame(&frame, RATE).expect("decodes");
        let twice = decode_frame(&frame, RATE).expect("decodes");
        assert_eq!(once, twice);
    }

    #[test]
    fn different_frames_get_different_noise() {
        // A decoder that reset its generator per frame would emit the
        // same noise every 23 milliseconds, and identical blocks
        // overlap-added at a fixed period are a tone, not noise —
        // audible, and invisible to any energy check.
        let a = probe(NOISE_BOOK, NOISE_PROBE_BAND, 100, &energy_field(380));
        let b = probe(NOISE_BOOK, NOISE_PROBE_BAND, 101, &energy_field(379));
        let one = decode_frame(&a, RATE).expect("decodes").remove(0);
        let two = decode_frame(&b, RATE).expect("decodes").remove(0);
        // The same stated energy, by construction.
        let band = NOISE_PROBE_BAND as usize;
        assert!(
            (band_energy(&one, band).log2() - band_energy(&two, band).log2()).abs() < 1e-9,
            "the probes do not state the same energy"
        );
        assert_ne!(one.spectrum, two.spectrum, "two frames, one noise");
    }

    #[test]
    fn a_noise_band_does_not_disturb_the_scalefactors_around_it() {
        // Measured: a noise band between two ordinary ones leaves both
        // of their magnitudes exactly where they were without it. Had
        // the two running values shared, every band after a noise band
        // would shift — silent corruption that sounds like a mixing
        // decision.
        let mut w = crate::bits::BitWriter::new();
        let mut with_noise = crate::bits::BitWriter::new();
        for (w, noisy) in [(&mut w, false), (&mut with_noise, true)] {
            w.write(ID_SCE, 3);
            w.write(0, 4);
            w.write(150, 8);
            w.write(0, 1);
            w.write(0, 2); // ONLY_LONG_SEQUENCE
            w.write(0, 1);
            w.write(if noisy { 3 } else { 2 }, 6);
            w.write(0, 1);
            // Band 0 ordinary, then optionally noise, then ordinary.
            w.write(1, 4);
            w.write(1, SECT_LEN_BITS);
            if noisy {
                w.write(NOISE_BOOK, 4);
                w.write(1, SECT_LEN_BITS);
            }
            w.write(1, 4);
            w.write(1, SECT_LEN_BITS);
            // Scalefactors: zero delta, the noise field, zero delta.
            w.write(0, 1);
            if noisy {
                for bit in energy_field(253) {
                    w.write(u32::from(bit), 1);
                }
            }
            w.write(0, 1);
            w.write(0, 3);
            // One quantised unit in each ordinary band.
            let word =
                crate::codebooks::quad_codeword(&QUAD_1, [1, 0, 0, 0], 1).expect("a codeword");
            for _ in 0..2 {
                w.write(word.0, u32::from(word.1));
            }
            w.write(ID_END, 3);
        }
        let plain = decode_frame(&w.finish(), RATE).expect("decodes").remove(0);
        let noisy = decode_frame(&with_noise.finish(), RATE)
            .expect("decodes")
            .remove(0);
        assert_eq!(noisy.noise_bands, 1);
        let bands = crate::bands::long_bands(RATE).expect("a measured rate");
        // Band 0 is the first in both; the last ordinary band is 1 in
        // one frame and 2 in the other.
        assert_eq!(
            plain.spectrum[bands[0]], noisy.spectrum[bands[0]],
            "band 0 moved"
        );
        assert_eq!(
            plain.spectrum[bands[1]], noisy.spectrum[bands[2]],
            "the band after the noise moved"
        );
    }

    #[test]
    fn intensity_stereo_is_still_refused_by_name() {
        // Thirteen is noise and is read; fourteen and fifteen are a
        // channel pair's business and are not. A reader that took them
        // for noise would fill real bands with rubbish at plausible
        // energy.
        for codebook in [14u32, 15] {
            let frame = probe(codebook, NOISE_PROBE_BAND, 100, &energy_field(380));
            assert_eq!(
                unsupported_element(&frame, RATE),
                Some("intensity stereo"),
                "codebook {codebook}"
            );
        }
    }

    /// Lay out an eight-short frame by hand, with the measured widths.
    ///
    /// Our encoder writes only long frames, so a short one has to be
    /// built here — which is also the point: every field width and the
    /// grouping rule came out of a decoder, and this is what they say a
    /// frame looks like.
    fn short_probe(grouping: u32, max_sfb: u32, spectral: &[bool], deltas: usize) -> Vec<u8> {
        let groups = window_groups(grouping);
        let mut w = crate::bits::BitWriter::new();
        w.write(ID_SCE, 3);
        w.write(0, 4);
        w.write(150, 8);
        w.write(0, 1); // ics_reserved_bit
        w.write(2, 2); // EIGHT_SHORT_SEQUENCE
        w.write(0, 1); // window_shape
        w.write(max_sfb, SHORT_MAX_SFB_BITS);
        w.write(grouping, SHORT_GROUPING_BITS);
        for _ in &groups {
            w.write(1, 4); // codebook 1
            w.write(max_sfb, SHORT_SECT_LEN_BITS);
        }
        for _ in 0..deltas {
            w.write(0, 1); // the all-zero scalefactor delta
        }
        w.write(0, 3); // pulse, tns, gain control
        for bit in spectral {
            w.write(u32::from(*bit), 1);
        }
        w.write(ID_END, 3);
        w.finish()
    }

    /// A codeword's bits, most significant first.
    fn word_bits(word: crate::codebooks::Codeword) -> Vec<bool> {
        (0..word.1)
            .map(|i| (word.0 >> (word.1 - 1 - i)) & 1 == 1)
            .collect()
    }

    #[test]
    fn eight_short_windows_read_into_eight_windows_in_order() {
        // Measured by placing a unit in each codeword of one group in
        // turn and asking a decoder when the sound arrived: it walked
        // window by window, in order. This holds the reader to that.
        let unit = crate::codebooks::quad_codeword(&QUAD_1, [1, 0, 0, 0], 1).expect("a codeword");
        let zero = crate::codebooks::quad_codeword(&QUAD_1, [0, 0, 0, 0], 1).expect("a codeword");
        // One group of all eight windows: every bit joins its window to
        // the one before it.
        const ONE_GROUP: u32 = 0b111_1111;
        for slot in 0..SHORT_WINDOWS {
            let mut spectral = Vec::new();
            for k in 0..SHORT_WINDOWS {
                spectral.extend(word_bits(if k == slot { unit } else { zero }));
            }
            let frame = short_probe(ONE_GROUP, 1, &spectral, 1);
            let channel = &decode_frame(&frame, RATE).expect("decodes")[0];
            assert_eq!(channel.sequence, WindowSequence::EightShort);
            assert_eq!(channel.groups, vec![SHORT_WINDOWS], "one group of eight");
            for window in 0..SHORT_WINDOWS {
                let loud = channel.spectrum[window * SHORT_FRAME] != 0.0;
                assert_eq!(
                    loud,
                    window == slot,
                    "codeword {slot} landed in window {window}"
                );
            }
        }
    }

    #[test]
    fn the_grouping_rule_is_the_one_a_decoder_demanded() {
        // Measured on all one hundred and twenty-eight values: a decoder
        // demands one section list per group, and the count is always
        // eight less the bits set.
        for grouping in 0..128u32 {
            let groups = window_groups(grouping);
            assert_eq!(
                groups.len(),
                SHORT_WINDOWS - grouping.count_ones() as usize,
                "grouping {grouping:07b}"
            );
            assert_eq!(
                groups.iter().sum::<usize>(),
                SHORT_WINDOWS,
                "grouping {grouping:07b}"
            );
        }
        // The two ends, spelled out.
        assert_eq!(window_groups(0), vec![1; SHORT_WINDOWS]);
        assert_eq!(window_groups(0b111_1111), vec![SHORT_WINDOWS]);
    }

    #[test]
    fn a_grouped_short_frame_shares_one_set_of_scalefactors() {
        // What grouping buys: one section list and one scalefactor for
        // several windows. A reader that restarted the running value per
        // group would read a cheaper frame as a different frame.
        let unit = crate::codebooks::quad_codeword(&QUAD_1, [1, 0, 0, 0], 1).expect("a codeword");
        const TWO_GROUPS: u32 = 0b111_0111;
        let mut spectral = Vec::new();
        for _ in 0..SHORT_WINDOWS {
            spectral.extend(word_bits(unit));
        }
        let frame = short_probe(TWO_GROUPS, 1, &spectral, 2);
        let channel = &decode_frame(&frame, RATE).expect("decodes")[0];
        assert_eq!(channel.groups, vec![4, 4]);
        let first = channel.spectrum[0];
        assert!(first != 0.0, "the probe put nothing anywhere");
        for window in 1..SHORT_WINDOWS {
            assert_eq!(
                channel.spectrum[window * SHORT_FRAME],
                first,
                "window {window}"
            );
        }
    }

    #[test]
    fn a_long_frame_is_one_window_and_says_so() {
        // The two layouts share four bits and nothing else. A reader
        // that treated a long frame as eight windows would walk off the
        // end of its own spectrum.
        let frame = encode_frame(&blocks(440.0)[1], RATE).expect("encodes");
        let channel = &decode_frame(&frame, RATE).expect("decodes")[0];
        assert_eq!(channel.sequence, WindowSequence::OnlyLong);
        assert_eq!(channel.groups, vec![1]);
    }

    #[test]
    fn the_window_shape_is_carried_out_and_not_assumed() {
        // A decoder that assumes the shape is not decoding, it is
        // guessing. Ours writes the sine shape; every frame ffmpeg
        // writes uses the other one, and reading those with the wrong
        // shape agrees with their own decoder to only 16 dB.
        let frame = encode_frame(&blocks(440.0)[1], RATE).expect("encodes");
        assert_eq!(
            decode_frame(&frame, RATE).expect("decodes")[0].window,
            WindowShape::Sine
        );
        // `window_shape` sits at bit 18: three of element id, four of
        // tag, eight of global gain, one reserved, two of sequence.
        let mut kbd = frame.clone();
        kbd[2] |= 0b0010_0000;
        assert_eq!(
            decode_frame(&kbd, RATE).expect("decodes")[0].window,
            WindowShape::Kbd
        );
    }

    #[test]
    fn both_channels_of_a_pair_read_back_to_what_the_encoder_meant() {
        // The second channel of a pair has no `ics_info` of its own, so
        // a reader expecting one would desynchronise by exactly that
        // structure's width — and produce noise, not an error.
        // Substitution stays off here for the same reason as above.
        let (left, right) = (blocks(440.0), blocks(1310.0));
        for (l, r) in left.iter().zip(&right) {
            let frame = crate::encode::encode_frame_pair_tuned(
                l,
                r,
                RATE,
                DEFAULT_QUANT_TARGET,
                NEVER_SUBSTITUTE,
            )
            .expect("encodes");
            let spectra = decode_frame(&frame, RATE).expect("decodes");
            assert_eq!(spectra.len(), 2, "a pair element has two channels");
            for (channel, source) in spectra.iter().zip([l, r]) {
                // The pair shares `max_sfb`, so a channel may carry
                // bands its own plan would have dropped; those decode to
                // zero, which is what dropping them meant.
                let meant = intended_spectrum(source, RATE).expect("plans");
                for (i, (want, got)) in meant.iter().zip(&channel.spectrum).enumerate() {
                    assert_eq!(want.to_bits(), got.to_bits(), "coefficient {i}");
                }
            }
        }
    }

    #[test]
    fn sound_encoded_and_decoded_comes_back_as_the_same_sound() {
        // The end-to-end claim, in the only unit anybody hears. What is
        // left is quantisation noise and nothing else — the tables are
        // held exactly by the tests above.
        let signal = tone(8192, 440.0);
        let coded: Vec<Vec<f64>> = analyse(&signal, LONG_FRAME, WindowShape::Sine)
            .iter()
            .map(|block| {
                // The end-to-end claim is about the quantiser, so the
                // other lever is held still; what noise substitution
                // does to the sound is measured on real material in
                // `tests/oracle.rs`, against the encoder that wrote it.
                let frame = encode_frame_tuned(block, RATE, DEFAULT_QUANT_TARGET, NEVER_SUBSTITUTE)
                    .expect("encodes");
                decode_frame(&frame, RATE)
                    .expect("decodes")
                    .remove(0)
                    .spectrum
            })
            .collect();
        let out = synthesise(&coded, LONG_FRAME, WindowShape::Sine);

        // Overlap-add is only complete away from the ends: the first and
        // last half-window each have one contribution instead of two.
        let (from, to) = (LONG_FRAME, signal.len().min(out.len()) - LONG_FRAME);
        assert!(to > from + LONG_FRAME, "not enough settled signal to judge");
        let mut signal_power = 0.0f64;
        let mut error_power = 0.0f64;
        for i in from..to {
            signal_power += signal[i] * signal[i];
            let error = signal[i] - out[i];
            error_power += error * error;
        }
        let snr = 10.0 * (signal_power / error_power).log10();
        assert!(snr > 60.0, "round trip managed only {snr:.1} dB");
    }

    #[test]
    fn a_rate_without_a_measured_band_table_is_refused() {
        // Reading against another rate's boundaries produces plausible
        // noise, which is worse than a refusal.
        let frame = encode_frame(&vec![0.0; LONG_FRAME], RATE).expect("encodes");
        assert_eq!(
            decode_frame(&frame, RATE - 1),
            Err(DecodeError::UnmeasuredRate(RATE - 1))
        );
    }

    #[test]
    fn a_truncated_frame_is_refused_rather_than_completed() {
        // Half a frame is a damaged file. Inventing the rest hands a
        // listener sound the file never contained.
        let frame = encode_frame(&blocks(440.0)[1], RATE).expect("encodes");
        assert!(frame.len() > 8, "the probe frame is too small to cut");
        for cut in [0usize, 1, frame.len() / 4, frame.len() / 2, frame.len() - 1] {
            assert!(
                decode_frame(&frame[..cut], RATE).is_err(),
                "{cut} bytes of {} decoded",
                frame.len()
            );
        }
        assert!(
            decode_frame(&frame, RATE).is_ok(),
            "and the whole frame does not"
        );
    }

    #[test]
    fn rubbish_is_refused_and_never_spins() {
        // Negative trust: nothing here came from our encoder. Whatever
        // it answers, it must answer — an unbounded read or a panic on a
        // hostile frame is the fault this rules out.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for len in [1usize, 2, 3, 5, 8, 13, 64, 300] {
            for _ in 0..64 {
                let frame: Vec<u8> = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state as u8
                    })
                    .collect();
                let _ = decode_frame(&frame, RATE);
            }
        }
    }

    #[test]
    fn the_index_expansion_is_the_forward_tables_own() {
        // The forward tables are indexed by tuple, most significant
        // first. This is that mixed-radix number expanded; a second
        // stored table would be a second chance to disagree.
        assert_eq!(values_of(0, 1, 4), [-1, -1, -1, -1]);
        assert_eq!(values_of(40, 1, 4), [0, 0, 0, 0]);
        assert_eq!(values_of(80, 1, 4), [1, 1, 1, 1]);
        assert_eq!(values_of(0, 12, 2)[..2], [-12, -12]);
        assert_eq!(values_of(312, 12, 2)[..2], [0, 0]);
        assert_eq!(values_of(624, 12, 2)[..2], [12, 12]);
    }

    #[test]
    fn parts_of_the_format_this_does_not_read_are_named() {
        // A refusal that says which part is a known gap; one that just
        // says "unsupported" is a mystery, and mysteries get guessed at.
        let frame = encode_frame(&blocks(440.0)[1], RATE).expect("encodes");
        assert_eq!(unsupported_element(&frame, RATE), None);
        // Channel coupling is refused by name; the low-frequency
        // element is not, because it is syntactically a single channel
        // and the difference is which speaker it goes to — the
        // container's business, not the bitstream reader's.
        let mut coupled = frame.clone();
        coupled[0] = (coupled[0] & 0x1F) | ((ID_CCE as u8) << 5);
        assert_eq!(
            unsupported_element(&coupled, RATE),
            Some("channel coupling")
        );
        let mut low = frame.clone();
        low[0] = (low[0] & 0x1F) | ((ID_LFE as u8) << 5);
        assert_eq!(
            decode_frame(&low, RATE)
                .expect("reads as one channel")
                .len(),
            1
        );
    }
}
