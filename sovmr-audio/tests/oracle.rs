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

//! Deriving AAC's normative tables by measurement rather than memory.
//!
//! # The problem
//!
//! An AAC-LC encoder needs some thousands of fixed numbers from ISO/IEC
//! 14496-3: eleven spectral Huffman codebooks, the scalefactor codebook,
//! and the scalefactor band boundaries for every sampling rate. None of
//! it is derivable. All of it is exactly the kind of data this project
//! has twice been burned writing from memory.
//!
//! # The way out
//!
//! A decoder is a function from bits to samples. Point it at bits we
//! choose and it tells us what the table says — without our having to
//! know the table first.
//!
//! Concretely: a frame can be filled with an arbitrary bit pattern where
//! the spectral data goes. The decoder reads it as codewords, produces
//! coefficients, and those coefficients come out as audible samples we
//! can measure. Enumerate the patterns and the code falls out.
//!
//! What makes this sound rather than clever is that the tables are then
//! checked by a *different* gate: an encoder built on derived tables
//! must produce streams that decode back to what it intended. Derivation
//! and verification never share an assumption.
//!
//! # This file, so far
//!
//! The first question, before any of that: does the oracle answer at
//! all? A probe frame is a good deal more than the empty one already
//! proven — it carries section data, a scalefactor, and spectral bits.
//! If any of those three is laid out wrongly, nothing further is worth
//! attempting.

use std::path::{Path, PathBuf};
use std::process::Command;

use sovmr_audio::asc::{AudioSpecificConfig, AAC_LC};
use sovmr_audio::bits::BitWriter;

/// Sampling rate every probe uses.
const PROBE_RATE: u32 = 48_000;
/// Coefficients in a long frame.
const FRAME_LENGTH: usize = 1024;
/// How many bits a long window spends on a section's length.
const SECT_LEN_BITS: u32 = 5;

/// One probe: a codebook to read the band with, and the bits to read.
///
/// `scalefactor_bits` and `spectral_bits` are handed over verbatim. That
/// is the whole point — we do not know what they mean yet, and the
/// decoder is being asked.
#[derive(Debug, Clone)]
struct Probe {
    /// `(codebook, bands)` runs, in order, covering `max_sfb` between
    /// them. Placing ZERO runs before a coded one is how a probe reaches
    /// past the low bands without knowing where they end.
    sections: Vec<(u32, u32)>,
    global_gain: u32,
    scalefactor_bits: Vec<bool>,
    spectral_bits: Vec<bool>,
}

impl Probe {
    /// Lay the probe out as a `raw_data_block`.
    ///
    /// Everything outside the two bit strings is the structure already
    /// proven by the silent-frame gate, plus the two fields that gate
    /// deliberately avoided: `section_data` and `scale_factor_data`.
    fn to_frame(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write(0x0, 3); // ID_SCE
        w.write(0, 4); // element_instance_tag
        w.write(self.global_gain, 8);
        // ics_info: one long window, exactly one coded band.
        w.write(0, 1); // ics_reserved_bit
        w.write(0, 2); // ONLY_LONG_SEQUENCE
        w.write(0, 1); // window_shape
        let max_sfb: u32 = self.sections.iter().map(|(_, bands)| bands).sum();
        w.write(max_sfb, 6);
        w.write(0, 1); // predictor_data_present
        for (codebook, bands) in &self.sections {
            w.write(*codebook, 4);
            write_section_length(&mut w, *bands);
        }
        // scale_factor_data: one band, so one delta — Huffman coded with
        // a book we do not have either, which is why these bits are also
        // supplied blind.
        for bit in &self.scalefactor_bits {
            w.write(u32::from(*bit), 1);
        }
        w.write(0, 1); // pulse_data_present
        w.write(0, 1); // tns_data_present
        w.write(0, 1); // gain_control_data_present
        for bit in &self.spectral_bits {
            w.write(u32::from(*bit), 1);
        }
        w.write(0x7, 3); // ID_END
        w.finish()
    }
}

/// A section's band count, with the escape the field needs.
///
/// The all-ones value means "and more follows", so a run of exactly that
/// length has to be written as all-ones then zero. Writing it plainly
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

impl Probe {
    /// A probe reaching band `band`: everything below it declared empty,
    /// and that one band read with `codebook`.
    fn at_band(band: u32, codebook: u32, gain: u32, spectral: Vec<bool>) -> Probe {
        let mut sections = Vec::new();
        if band > 0 {
            sections.push((CODEBOOK_ZERO, band));
        }
        sections.push((codebook, 1));
        Probe {
            sections,
            global_gain: gain,
            scalefactor_bits: vec![false],
            spectral_bits: spectral,
        }
    }
}

/// A run of one repeated bit, for padding a probe out past the longest
/// codeword any book contains.
fn filler(bit: bool, count: usize) -> Vec<bool> {
    vec![bit; count]
}

/// Where probe files live for this process.
fn probe_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sovmr-oracle-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("probe directory");
    dir
}

/// Write frames into a Matroska file this project's own muxer produces.
fn mux_frames_at(path: &Path, frames: &[Vec<u8>], rate: u32) {
    mux_frames_with(path, frames, rate, 1);
}

/// The same, for a stream declaring a channel count of its own.
fn mux_frames_with(path: &Path, frames: &[Vec<u8>], rate: u32, channels: u32) {
    let spec = sovmr_mkv::mux::TrackSpec {
        number: 1,
        codec_id: "A_AAC".to_string(),
        codec_private: AudioSpecificConfig {
            object_type: AAC_LC,
            sample_rate: rate,
            channels,
        }
        .to_bytes(),
        video: false,
        subtitle: false,
        width: 0,
        height: 0,
        sample_rate: rate,
        channels: channels as u16,
        language: "und".to_string(),
        name: None,
        enabled: true,
        colour: None,
    };
    let mut mux = sovmr_mkv::mux::MkvMuxer::create(path, &[spec]).expect("muxer");
    let at = |f: u64| (f * FRAME_LENGTH as u64 * 1000 / u64::from(rate)) as i64;
    for (n, frame) in frames.iter().enumerate() {
        let n = n as u64;
        mux.add_frame(1, at(n), at(n + 1) - at(n), true, frame)
            .expect("frame");
    }
    mux.finalize().expect("finalize");
}

/// Decode a probe file, returning the samples and anything the decoder
/// had to say about them.
///
/// The complaints matter as much as the samples: a probe the decoder
/// rejected has told us something, and treating its output as an answer
/// would be reading noise as data.
fn decode(path: &Path) -> (Vec<f64>, String) {
    let raw = path.with_extension("raw");
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            raw.to_str().unwrap(),
        ])
        .output()
        .expect("ffmpeg runs");
    let complaints = String::from_utf8_lossy(&out.stderr).to_string();
    let bytes = std::fs::read(&raw).unwrap_or_default();
    let samples = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes(*c)))
        .collect();
    (samples, complaints)
}

/// Codebook meaning "this band is entirely zero".
const CODEBOOK_ZERO: u32 = 0;
/// Mid-scale gain: loud enough to measure, far from either end of the
/// scalefactor range where clamping could hide a mistake.
const PROBE_GAIN: u32 = 100;
/// No AAC codeword is anywhere near this long, so a run that still does
/// not fit by here means the probe layout is wrong, not the run.
const LONGEST_RUN: usize = 24;

/// Mux one probe, decode it, and hand back both halves of the answer.
fn run_probe(dir: &Path, name: &str, probe: &Probe) -> (Vec<f64>, String) {
    run_probe_at(dir, name, probe, PROBE_RATE)
}

/// The same, at a chosen sampling rate — which is what makes a band
/// table measurable for any rate rather than only the one.
fn run_probe_at(dir: &Path, name: &str, probe: &Probe, rate: u32) -> (Vec<f64>, String) {
    let path = dir.join(format!("{name}.mkv"));
    // Repeated, because a lone frame's output is half-cancelled by the
    // overlap with its silent neighbours. In a run of identical frames
    // the middle ones are steady state, and that is where a coefficient
    // can be read back cleanly.
    mux_frames_at(&path, &vec![probe.to_frame(); PROBE_FRAMES], rate);
    decode(&path)
}

/// Frames per probe file.
const PROBE_FRAMES: usize = 8;

/// The shortest run of zero bits that lets a frame using `codebook`
/// decode cleanly, or `None` if no run up to [`LONGEST_RUN`] does.
///
/// Shorter runs leave the decoder mid-codeword; longer ones make it read
/// the terminator as data. The fit is exact from both sides, which is
/// what makes the answer a measurement.
fn smallest_fitting_run(dir: &Path, codebook: u32) -> Option<usize> {
    (0..=LONGEST_RUN).find(|run| {
        let probe = Probe {
            sections: vec![(codebook, 1)],
            global_gain: PROBE_GAIN,
            scalefactor_bits: filler(false, *run),
            spectral_bits: vec![],
        };
        let (samples, complaints) = run_probe(dir, &format!("cb{codebook}_run{run}"), &probe);
        complaints.trim().is_empty() && !samples.is_empty()
    })
}

/// `window_sequence` for the eight-short frame, whose `ics_info` is laid
/// out differently and cannot be walked with the long-window reader.
const EIGHT_SHORT_SEQUENCE: u32 = 2;
/// Where the steady state begins: far enough in that the overlap with
/// the decoder's priming is long past.
const STEADY_AT: usize = 3072;
/// Transform length for the readout. Identical frames make the output
/// periodic, and this is a whole number of those periods.
const READOUT_LEN: usize = 2048;
/// A gain that puts the coefficient many decades above float noise
/// without coming anywhere near full scale.
const QUIET_GAIN: u32 = 150;
/// Enough gain steps to be unambiguous: four steps double, so this is a
/// factor of thirty-two.
const GAIN_STEP: u32 = 20;
/// Coefficients one codebook-1 codeword covers, measured: a band of
/// four needs one codeword and a band of eight needs two.
const COEFFICIENTS_PER_CODEWORD: usize = 4;
/// No band needs more filler than this; past it the probe is wrong.
const MAX_FILLER: usize = 40;
/// No codeword in a four-coefficient book runs longer than this. Walking
/// past it would only cost time: the walk stops when every tuple has
/// been accounted for, and fails loudly if it cannot be.
const LONGEST_CODEWORD: u32 = 20;
/// What a quantised one comes back as at [`QUIET_GAIN`], measured.
const UNIT_MAGNITUDE: f64 = 8.8388e-2;
/// The book whose codewords are bases rather than whole answers.
const ESCAPE_BOOK: u32 = 11;
/// Magnitude pairs the escape rule must predict correctly before the
/// walk is allowed to rely on it. Chosen to cross both payload-width
/// boundaries the rule has near the bottom of its range.
const PREDICTED_MAGNITUDES: [(i32, i32); 6] =
    [(16, 16), (17, 16), (16, 17), (31, 16), (32, 16), (16, 31)];

/// The section layout, pinned by a band that needs nothing.
///
/// A band assigned codebook ZERO reads neither a scalefactor nor any
/// spectral data. So a frame containing one must decode with *exactly*
/// zero bits between the section and the terminator — and it does.
///
/// One bit more and the decoder walks off into whatever follows, which
/// is what makes this a measurement rather than an observation: the
/// answer is bracketed on both sides.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn a_zero_band_needs_no_bits_which_pins_the_section_layout() {
    let dir = probe_dir("section");
    let probe = |extra: usize| Probe {
        sections: vec![(CODEBOOK_ZERO, 1)],
        global_gain: PROBE_GAIN,
        scalefactor_bits: filler(false, extra),
        spectral_bits: vec![],
    };

    let (samples, complaints) = run_probe(&dir, "exact", &probe(0));
    assert!(
        complaints.trim().is_empty(),
        "a ZERO band should need no bits, but the decoder said: {complaints}"
    );
    assert!(!samples.is_empty(), "nothing decoded");

    // The bracket. Without it, "zero bits works" would be equally true
    // of a decoder that ignored the field entirely.
    for extra in 1..=2usize {
        let (_, complaints) = run_probe(&dir, &format!("over{extra}"), &probe(extra));
        assert!(
            !complaints.trim().is_empty(),
            "{extra} spare bits went unnoticed, so the exact fit above proves nothing"
        );
    }
}

/// How long a run of zero bits does each book need to fill a band?
///
/// Not, as this once claimed, "how long is the all-zero codeword" — for
/// two of the books a run of zeros selects a codeword meaning one, not
/// zero. What is measured is the run, which is what the parity argument
/// below actually uses.
///
/// Widen a run of zero bits until the frame decodes. Too few and the
/// decoder runs out mid-codeword; too many and it reads the terminator
/// as data. Exactly one width fits, and that width is a number from the
/// specification we never had to know.
///
/// The measurement covers the scalefactor codeword as well, since both
/// are in the same run — which is what the next test separates out.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn the_decoder_tells_us_how_long_an_all_zero_run_has_to_be() {
    let dir = probe_dir("lengths");
    let mut measured = Vec::new();
    for codebook in 1..=11u32 {
        let fit = smallest_fitting_run(&dir, codebook);
        let fit = fit.unwrap_or_else(|| {
            panic!("codebook {codebook}: no run of zero bits decoded — the probe layout is wrong")
        });
        measured.push((codebook, fit));
    }

    // Every book must code the all-zero tuple in *some* number of bits;
    // a book that needed none would mean the band was never read.
    for (codebook, fit) in &measured {
        assert!(
            *fit > 0,
            "codebook {codebook} consumed nothing, so the band was not read"
        );
    }

    // The scalefactor codeword is in every one of these runs, so the
    // shortest run bounds it. It cannot be longer than the smallest
    // total, and it cannot be zero.
    let shortest = measured.iter().map(|(_, f)| *f).min().expect("measured");
    println!("all-zero runs, scalefactor codeword included:");
    for (codebook, fit) in &measured {
        println!("  codebook {codebook:2}: {fit:2} bits");
    }
    println!("  → the scalefactor codeword is at most {shortest} bits");
    assert!(shortest >= 1);
}

/// Separating the scalefactor codeword from the spectral one.
///
/// A ZERO band reads no scalefactor; a coded band reads exactly one. So
/// two frames differing only in that respect differ by exactly the
/// scalefactor codeword plus the band's spectral codewords — and a book
/// whose spectral part is known to be one codeword long isolates it.
///
/// This is the measurement the whole extraction rests on: every later
/// probe holds the scalefactor fixed at this codeword so that what
/// changes is spectrum alone.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn the_scalefactor_codeword_is_separated_from_the_spectral_one() {
    let dir = probe_dir("split");

    // Books measured above, with their total all-zero run lengths.
    let totals: Vec<(u32, usize)> = (1..=11u32)
        .map(|cb| (cb, smallest_fitting_run(&dir, cb).expect("a run fits")))
        .collect();

    // The scalefactor codeword length is the one value that makes every
    // book's remaining bits divide into whole codewords of at least one
    // bit each. Solve for it rather than assume it.
    let feasible: Vec<usize> = (1..=totals.iter().map(|(_, t)| *t).min().unwrap())
        .filter(|sf| totals.iter().all(|(_, total)| total > sf))
        .collect();
    assert!(
        !feasible.is_empty(),
        "no scalefactor length leaves room for a spectral codeword in every book"
    );
    let scalefactor_bits = feasible[0];

    // Confirm it directly: a frame with that many zero bits and a ZERO
    // band must *over*-run, because a ZERO band reads no scalefactor at
    // all. That is the difference being claimed, tested from the side.
    let (_, complaints) = run_probe(
        &dir,
        "zero_with_sf",
        &Probe {
            sections: vec![(CODEBOOK_ZERO, 1)],
            global_gain: PROBE_GAIN,
            scalefactor_bits: filler(false, scalefactor_bits),
            spectral_bits: vec![],
        },
    );
    assert!(
        !complaints.trim().is_empty(),
        "a ZERO band accepted scalefactor bits, so the two are not separated after all"
    );

    println!("SCALEFACTOR CODEWORD: the all-zero delta is {scalefactor_bits} bit(s)");
    println!("spectral all-zero codewords, that much subtracted:");
    for (codebook, total) in &totals {
        println!("  codebook {codebook:2}: {} bits", total - scalefactor_bits);
    }

    // A structural fact falls out of these numbers for free. A book that
    // codes four coefficients at a time fills this band with one
    // codeword; a book that codes two needs two, so its spectral bits
    // must be even. An *odd* count therefore proves a four-at-a-time
    // book — without anyone having to be told which books those are.
    //
    // The converse does not follow: an even count is what a pair book
    // gives, and also what a quad book with an even-length codeword
    // gives. Those stay open until the coefficients themselves can be
    // read back, which is the next thing to build.
    let quads: Vec<u32> = totals
        .iter()
        .filter(|(_, total)| (total - scalefactor_bits) % 2 == 1)
        .map(|(cb, _)| *cb)
        .collect();
    assert!(
        !quads.is_empty(),
        "no book showed an odd spectral length, so the band is not four coefficients wide \
         and every length above needs reinterpreting"
    );
    println!("  → codebooks {quads:?} code four coefficients at a time (odd lengths prove it)");
    println!("  → the rest are even and stay undecided until coefficients can be read back");
}

/// Samples at the start of a signal where the overlap is deliberately
/// incomplete, and comparing would measure the ramp rather than the
/// encoder. The tail margin is a whole frame, taken from the decoded
/// stream — see the comparison itself.
const RAMP: usize = 2048;
/// Coefficients, and output samples, in one long frame.
const LONG_FRAME: usize = sovmr_audio::bands::LONG_FRAME;
/// How far the decoded stream may be shifted from the input before the
/// alignment search gives up.
const MAXIMUM_OFFSET: usize = 4096;
/// Samples the alignment search correlates over.
const ALIGNMENT_WINDOW: usize = 8192;
/// Samples in a round-trip signal, at every rate.
///
/// Long enough for the leading ramp, the alignment window, and a
/// trailing frame, with room left to measure over.
const SIGNAL_SAMPLES: usize = 24_000;
/// The floor a round trip must clear.
///
/// Set well below what is measured — a tone comes back at eighty
/// decibels — so the gate fails on a real regression rather than on the
/// last bit of a noisy signal. Flat quantisation noise with no
/// psychoacoustic model is not transparent, but the signal that comes
/// out must be unmistakably the one that went in.
const MINIMUM_SNR_DB: f64 = 40.0;

/// Two channels, decoded by somebody else, and not swapped.
///
/// A stereo element shares one `ics_info` between its channels, which is
/// the thing most easily got wrong: write one extra structure and the
/// second channel's sections are read out of its scalefactors. A decoder
/// would either complain or produce noise, and the round trip catches
/// both.
///
/// The channels are deliberately *different* — one is a low tone, the
/// other a high one at a different amplitude. Identical channels would
/// pass this gate with the two sides swapped, or with one written twice.
#[test]
#[ignore = "needs ffmpeg to decode what we wrote"]
fn our_stereo_survives_the_round_trip_with_its_channels_the_right_way_round() {
    use sovmr_audio::encode::encode_stereo;

    let dir = probe_dir("stereo");
    let mut checked = 0usize;
    let mut worst = f64::INFINITY;

    for rate in [48_000u32, 44_100, 32_000, 24_000] {
        let left: Vec<f64> = (0..SIGNAL_SAMPLES)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                0.5 * (std::f64::consts::TAU * 300.0 * t).sin()
            })
            .collect();
        let right: Vec<f64> = (0..SIGNAL_SAMPLES)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                0.25 * (std::f64::consts::TAU * 2_500.0 * t).sin()
            })
            .collect();

        let frames = encode_stereo(&left, &right, rate).expect("a measured rate encodes");
        let path = dir.join(format!("stereo_{rate}.mkv"));
        mux_frames_with(&path, &frames, rate, 2);

        let (interleaved, complaints) = decode(&path);
        assert!(
            complaints.trim().is_empty(),
            "{rate}: the decoder complained about our stereo:\n{complaints}"
        );
        // A decoder hands back one stream with the channels interleaved,
        // so pulling them apart is the first thing that has to be right.
        let decoded: [Vec<f64>; 2] = [
            interleaved.iter().step_by(2).copied().collect(),
            interleaved.iter().skip(1).step_by(2).copied().collect(),
        ];
        assert!(
            decoded[0].len() >= left.len(),
            "{rate}: {} samples per channel from {} — frames were lost",
            decoded[0].len(),
            left.len()
        );

        for (index, (want, got)) in [(&left, &decoded[0]), (&right, &decoded[1])]
            .into_iter()
            .enumerate()
        {
            let (offset, correlation) = best_alignment(want, got);
            assert!(
                correlation > 0.9,
                "{rate} channel {index}: best correlation is only {correlation:.3}"
            );
            let from = RAMP;
            let to = (want.len() - RAMP).min(got.len() - offset - LONG_FRAME);
            let mut signal_power = 0.0;
            let mut noise_power = 0.0;
            for i in from..to {
                signal_power += want[i] * want[i];
                noise_power += (want[i] - got[i + offset]).powi(2);
            }
            let snr = 10.0 * (signal_power / noise_power.max(f64::MIN_POSITIVE)).log10();
            assert!(
                snr > MINIMUM_SNR_DB,
                "{rate} channel {index}: {snr:.1} dB, below the {MINIMUM_SNR_DB} dB floor"
            );

            // And it must be *this* channel, not the other one. The two
            // are different tones at different amplitudes, so a swap
            // would show up here and nowhere else.
            let other = if index == 0 { &right } else { &left };
            let mut crossed = 0.0;
            for i in from..to {
                crossed += (other[i] - got[i + offset]).powi(2);
            }
            assert!(
                crossed > noise_power * 100.0,
                "{rate} channel {index} resembles the other channel as much as its own — \
                 the two were swapped or one was written twice"
            );
            println!("  {rate:6} Hz channel {index}: {snr:5.1} dB");
            worst = worst.min(snr);
            checked += 1;
        }
    }

    println!(
        "STEREO ROUND TRIP: {checked} channels, worst {worst:.1} dB, \
         each one matching its own side and not the other"
    );
}

/// Our AAC, decoded by somebody else, compared against what went in.
///
/// Every gate before this proved a piece: the frame layout, the band
/// table, the eleven codebooks, the escape rule. This is the one that
/// proves they work *together* — a real signal encoded by this project's
/// own encoder, decoded by ffmpeg, and measured against the original.
///
/// Two things are required, and they catch different failures.
///
/// **Silence on stderr.** A decoder that complains has found the stream
/// malformed, and no amount of numerical closeness afterwards would make
/// that acceptable.
///
/// **Signal-to-noise.** A stream can be perfectly valid and still decode
/// to something unlike the input — a wrong scale, a wrong band, a
/// codebook entry off by one. Only comparing against the original
/// catches that, and only a ratio catches it at every amplitude.
#[test]
#[ignore = "needs ffmpeg to decode what we wrote"]
fn our_encoder_survives_a_round_trip_through_somebody_elses_decoder() {
    use sovmr_audio::encode::encode_mono;

    let dir = probe_dir("roundtrip");
    let mut checked = 0usize;
    let mut worst = f64::INFINITY;

    // Every rate, not only the one the rest of this file probes at. A
    // band table that was measured correctly and then wired to the
    // wrong rate would pass every other gate in this file and produce
    // an unlistenable stream, and only encoding at that rate finds it.
    for rate in sovmr_audio::bands::MEASURED_RATES {
        for (name, signal) in round_trip_signals(rate) {
            let name = format!("{name}_{rate}");
            let frames = encode_mono(&signal, rate).expect("a measured rate encodes");
            let path = dir.join(format!("{name}.mkv"));
            mux_frames_at(&path, &frames, rate);

            let (decoded, complaints) = decode(&path);
            assert!(
                complaints.trim().is_empty(),
                "{name}: the decoder complained about our stream:\n{complaints}"
            );
            assert!(
                decoded.len() >= signal.len(),
                "{name}: {} samples came back from {} — frames were lost",
                decoded.len(),
                signal.len()
            );

            // Line the two up. A decoder emits a frame of priming before the
            // signal proper, and the encoder pads the front by a frame so
            // the first overlap has something to ramp from. Rather than
            // reason about whose is whose, the offset is *found*: the shift
            // that maximises correlation is the alignment, and if no shift
            // correlates the streams are unrelated and the test says so.
            let (offset, correlation) = best_alignment(&signal, &decoded);
            assert!(
                correlation > 0.9,
                "{name}: the best alignment correlates only {correlation:.3} — \
             the decoded audio does not resemble the input at any offset"
            );

            // Signal to noise, over the part both cover.
            //
            // The end of the comparison is set by the *decoded* stream, not
            // the input. A decoder's last frame is the overlap ramping down
            // into the encoder's trailing pad, and where that begins depends
            // on the alignment. Measuring into it once cost 17.6 dB against
            // 80 dB everywhere else, and the fault was here rather than in
            // anything being measured.
            let from = RAMP;
            let to = (signal.len() - RAMP).min(decoded.len() - offset - LONG_FRAME);
            assert!(
                to > from + LONG_FRAME,
                "{name}: only {} samples overlap, which measures nothing",
                to.saturating_sub(from)
            );
            let mut signal_power = 0.0;
            let mut noise_power = 0.0;
            for i in from..to {
                let want = signal[i];
                let got = decoded[i + offset];
                signal_power += want * want;
                noise_power += (want - got) * (want - got);
            }
            let snr = 10.0 * (signal_power / noise_power.max(f64::MIN_POSITIVE)).log10();
            assert!(
                snr > MINIMUM_SNR_DB,
                "{name}: {snr:.1} dB signal to noise, below the {MINIMUM_SNR_DB} dB floor"
            );
            println!("  {name:>24}: {snr:5.1} dB, {} frames", frames.len());
            worst = worst.min(snr);
            checked += 1;
        }
    }

    println!(
        "ROUND TRIP THROUGH FFMPEG: {checked} signals across {} rates, \
         worst {worst:.1} dB signal to noise, not one complaint",
        sovmr_audio::bands::MEASURED_RATES.len()
    );
}

/// Signals chosen to fail differently.
///
/// A tone finds a wrong band or a wrong scale. A sweep finds a band
/// boundary that is off, because it crosses every one of them. An
/// impulse is flat across the whole spectrum, so it finds a codebook
/// that cannot carry what the quantiser produced. Noise finds anything
/// that only works on tidy input.
fn round_trip_signals(rate: u32) -> Vec<(String, Vec<f64>)> {
    // A fixed number of samples rather than a fixed duration: the
    // alignment window and the two ramps have to fit whatever the rate
    // is, and at eight kilohertz half a second does not reach them.
    let length = SIGNAL_SAMPLES;
    let mut signals = Vec::new();

    let tone: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            0.5 * (std::f64::consts::TAU * 440.0 * t).sin()
        })
        .collect();
    signals.push(("tone_440".to_string(), tone));

    let chord: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            0.3 * (std::f64::consts::TAU * 220.0 * t).sin()
                + 0.2 * (std::f64::consts::TAU * 1_318.5 * t).sin()
                + 0.1 * (std::f64::consts::TAU * 5_000.0 * t).sin()
        })
        .collect();
    signals.push(("three_tones".to_string(), chord));

    // Logarithmic, so it spends comparable time in every band rather
    // than racing through the low ones.
    let sweep: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let span = length as f64 / f64::from(rate);
            // Up to just under Nyquist: sweeping past it would ask the
            // encoder to carry what the rate cannot represent.
            let (low, high) = (50.0f64, f64::from(rate) * 0.45);
            let k = (high / low).powf(t / span);
            let phase = std::f64::consts::TAU * low * span * (k - 1.0) / (high / low).ln();
            0.5 * phase.sin()
        })
        .collect();
    signals.push(("log_sweep".to_string(), sweep));

    let mut impulses = vec![0.0; length];
    for i in (RAMP..length).step_by(4096) {
        impulses[i] = 0.8;
    }
    signals.push(("impulses".to_string(), impulses));

    // Seeded, because a test that cannot be repeated cannot be debugged.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let noise: Vec<f64> = (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.4 * ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
        })
        .collect();
    signals.push(("noise".to_string(), noise));

    // Quiet enough that a scalefactor chosen for a loud frame would
    // quantise it to nothing.
    let quiet: Vec<f64> = (0..length)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            0.001 * (std::f64::consts::TAU * 900.0 * t).sin()
        })
        .collect();
    signals.push(("very_quiet".to_string(), quiet));

    signals
}

/// The offset at which the decoded audio best lines up with the input,
/// and how well it does at that offset.
///
/// Correlation rather than difference, so the answer does not depend on
/// the amplitude — a decoded stream at half volume still correlates
/// perfectly, and the signal-to-noise check afterwards is what would
/// catch the volume.
fn best_alignment(signal: &[f64], decoded: &[f64]) -> (usize, f64) {
    let window = ALIGNMENT_WINDOW.min(signal.len());
    let reference = &signal[RAMP..RAMP + window];
    let energy: f64 = reference.iter().map(|s| s * s).sum::<f64>().sqrt();
    let mut best = (0usize, 0.0f64);
    for offset in 0..=MAXIMUM_OFFSET {
        if offset + RAMP + window > decoded.len() {
            break;
        }
        let candidate = &decoded[offset + RAMP..offset + RAMP + window];
        let dot: f64 = reference.iter().zip(candidate).map(|(a, b)| a * b).sum();
        let candidate_energy: f64 = candidate.iter().map(|s| s * s).sum::<f64>().sqrt();
        let correlation = dot / (energy * candidate_energy).max(f64::MIN_POSITIVE);
        if correlation > best.1 {
            best = (offset, correlation);
        }
    }
    best
}

/// The escape book, re-derived from a decoder — model and all.
///
/// Codebook 11 is the one whose codeword is not the whole story. A
/// magnitude of sixteen promises that the real value follows in a
/// variable-length sequence, so a naive walk finds unboundedly many
/// "codewords" and never finishes. It was allowed ten minutes and did
/// not.
///
/// What makes it tractable is knowing the escape sequence, and what
/// makes knowing it honest is that the knowledge was *predicted before
/// it was used*: six magnitudes were encoded by the rule, handed to a
/// decoder, and all six came back exactly as predicted. That prediction
/// is the first half of this gate.
///
/// The second half walks the book. At each node it appends zero, one and
/// two escape sequences, and accepts only when the number of escaped
/// magnitudes read back equals the number appended. That test is what
/// separates a real escape base from a prefix which, with escape bits
/// stuck on the end, merely parses as something else — the first such
/// accident was a two-bit prefix reading back as (1, 2).
#[test]
#[ignore = "needs ffmpeg to answer, and asks it some thousands of times"]
fn the_escape_book_is_what_a_decoder_says_it_is() {
    use sovmr_audio::codebooks::{
        escape_sequence, pair_index, Codeword, ESCAPE_11, ESCAPE_MAGNITUDE,
    };
    use sovmr_audio::mdct::{mdct, window, WindowShape};

    let dir = probe_dir("escape");
    let n = sovmr_audio::bands::LONG_FRAME;
    let shape = window(2 * n, WindowShape::Sine);

    // Bits of a codeword, most significant first.
    let spell = |(bits, length): Codeword| -> Vec<bool> {
        (0..u32::from(length))
            .map(|i| bits >> (u32::from(length) - 1 - i) & 1 == 1)
            .collect()
    };
    // The filler behind a candidate: half the shortest zero run that
    // fills a band, measured rather than assumed.
    let run = (1..=MAX_FILLER)
        .find(|run| {
            let probe = Probe::at_band(0, ESCAPE_BOOK, QUIET_GAIN, filler(false, *run));
            let (samples, complaints) = run_probe(&dir, "fill", &probe);
            complaints.trim().is_empty() && samples.len() >= 2 * n
        })
        .expect("a run of zeros fills the band");
    assert_eq!(run % 2, 0, "{run} zero bits is not two equal codewords");
    let behind = filler(false, run / 2);

    let read = |samples: &[f64]| -> [i32; 2] {
        let windowed: Vec<f64> = (0..2 * n).map(|i| samples[i] * shape[i]).collect();
        let mut coefficients = vec![0.0; n];
        mdct(&windowed, &mut coefficients);
        let mut pair = [0i32; 2];
        for (k, slot) in pair.iter_mut().enumerate() {
            let magnitude = (coefficients[k].abs() / UNIT_MAGNITUDE).powf(0.75).round();
            *slot = if coefficients[k] < 0.0 {
                -(magnitude as i32)
            } else {
                magnitude as i32
            };
        }
        pair
    };

    // ---- the model, predicted before it is relied on ---------------
    //
    // Take the base for (16, 16) straight from the shipped table, append
    // the sequences the rule gives for a chosen magnitude, and require
    // the decoder to produce that magnitude. If the rule were wrong the
    // shipped table would be built on sand, and this says so first.
    let base = ESCAPE_11
        [pair_index([ESCAPE_MAGNITUDE, ESCAPE_MAGNITUDE], ESCAPE_MAGNITUDE).expect("in range")];
    for (first, second) in PREDICTED_MAGNITUDES {
        let mut bits = spell(base);
        for magnitude in [first, second] {
            bits.extend(spell(escape_sequence(magnitude).expect("escapable")));
        }
        bits.extend_from_slice(&behind);
        let probe = Probe::at_band(0, ESCAPE_BOOK, QUIET_GAIN, bits);
        let (samples, complaints) = run_probe(&dir, "predict", &probe);
        assert!(
            complaints.trim().is_empty(),
            "predicted ({first}, {second}) was rejected: {complaints}"
        );
        assert_eq!(
            read(&samples),
            [first, second],
            "the escape rule predicted ({first}, {second}) and the decoder disagreed"
        );
    }

    // ---- the walk --------------------------------------------------
    let escape_bits = spell(escape_sequence(ESCAPE_MAGNITUDE).expect("escapable"));
    let mut derived: Vec<Option<Codeword>> = vec![None; ESCAPE_11.len()];
    let mut found: Vec<Codeword> = Vec::new();
    let mut probes = 0usize;

    for length in 1..=LONGEST_CODEWORD {
        for value in 0..(1u64 << length) {
            let value = value as u32;
            if found.iter().any(|(bits, len)| {
                u32::from(*len) <= length && value >> (length - u32::from(*len)) == *bits
            }) {
                continue;
            }
            let candidate: Vec<bool> = (0..length)
                .map(|i| value >> (length - 1 - i) & 1 == 1)
                .collect();
            for escapes in 0..=2usize {
                probes += 1;
                let mut bits = candidate.clone();
                for _ in 0..escapes {
                    bits.extend_from_slice(&escape_bits);
                }
                bits.extend_from_slice(&behind);
                let probe = Probe::at_band(0, ESCAPE_BOOK, QUIET_GAIN, bits);
                let (samples, complaints) = run_probe(&dir, "walk", &probe);
                if !complaints.trim().is_empty() || samples.len() < 2 * n {
                    continue;
                }
                let pair = read(&samples);
                // The discriminator. Escape bits appended to a prefix
                // that is not a base usually parse as something else,
                // and that something else does not carry escaped
                // magnitudes it was never given.
                if pair.iter().filter(|v| v.abs() >= ESCAPE_MAGNITUDE).count() != escapes {
                    continue;
                }
                let index = pair_index(pair, ESCAPE_MAGNITUDE).unwrap_or_else(|| {
                    panic!("the decoder produced {pair:?}, outside the book's own range")
                });
                assert!(
                    derived[index].is_none(),
                    "{pair:?} has two codewords, so the walk is wrong"
                );
                derived[index] = Some((value, length as u8));
                found.push((value, length as u8));
                break;
            }
        }
    }

    let missing = derived.iter().filter(|e| e.is_none()).count();
    assert_eq!(
        missing,
        0,
        "{missing} of {} entries never appeared — the walk stopped short",
        ESCAPE_11.len()
    );
    let derived: Vec<Codeword> = derived.into_iter().map(|e| e.expect("filled")).collect();
    assert_eq!(
        derived,
        ESCAPE_11.to_vec(),
        "the decoder disagrees with the escape book this project ships"
    );
    println!(
        "ESCAPE BOOK RE-DERIVED: {} codewords, complete and matching, from {probes} probes, \
         and the escape rule predicted {} magnitudes before being trusted",
        derived.len(),
        PREDICTED_MAGNITUDES.len()
    );
}

/// Walk the scalefactor codebook, reading each delta out of the sound.
#[test]
#[ignore = "diagnostic"]
fn walk_scalefactor_book() {
    use sovmr_audio::mdct::{mdct, window, WindowShape};
    let dir = probe_dir("sfbook");
    let n = sovmr_audio::bands::LONG_FRAME;
    let shape = window(2 * n, WindowShape::Sine);
    let max_len: u32 = std::env::var("LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(19);

    // One band, codebook 1, a spectral pattern measured to set one
    // coefficient to a quantised value of one. What that coefficient
    // comes back as tells us the scalefactor the delta produced.
    let spectral: Vec<bool> = "10010".chars().map(|c| c == '1').collect();
    let mut found: Vec<(u32, u32)> = Vec::new();
    let mut probes = 0usize;
    for length in 1..=max_len {
        for value in 0..(1u64 << length) {
            let value = value as u32;
            if found
                .iter()
                .any(|(b, l)| *l <= length && value >> (length - l) == *b)
            {
                continue;
            }
            probes += 1;
            let bits: Vec<bool> = (0..length)
                .map(|i| value >> (length - 1 - i) & 1 == 1)
                .collect();
            let probe = Probe {
                sections: vec![(1, 1)],
                global_gain: QUIET_GAIN,
                scalefactor_bits: bits,
                spectral_bits: spectral.clone(),
            };
            let (samples, c) = run_probe(&dir, "sf", &probe);
            if !c.trim().is_empty() || samples.len() < 2 * n {
                continue;
            }
            let windowed: Vec<f64> = (0..2 * n).map(|i| samples[i] * shape[i]).collect();
            let mut coeffs = vec![0.0; n];
            mdct(&windowed, &mut coeffs);
            let peak = coeffs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            // magnitude = 2^((sf-100)/4) / 65536, and the probe sets a
            // quantised one, so the exponent is readable directly.
            let delta = if peak <= 0.0 {
                None
            } else {
                let sf = 100.0 + 4.0 * (peak * 65536.0).log2();
                Some((sf - QUIET_GAIN as f64).round() as i32)
            };
            let pattern: String = (0..length)
                .map(|i| {
                    if value >> (length - 1 - i) & 1 == 1 {
                        '1'
                    } else {
                        '0'
                    }
                })
                .collect();
            println!("sf {pattern:>20} -> delta {delta:?} (peak {peak:.4e})");
            found.push((value, length));
        }
    }
    println!("sf: {} codewords, {probes} probes", found.len());
}

/// The ten spectral codebooks, re-derived from a decoder.
///
/// The tables this checks live in [`sovmr_audio::codebooks`] and were
/// produced by exactly this walk. Keeping the gate is the whole
/// argument for trusting them: a table nobody can re-derive is a table
/// nobody can check.
///
/// The walk offers a decoder a bit pattern and watches what it does. Too
/// few bits and the frame under-runs; too many and the terminator is
/// read as data; exactly one codeword and the frame decodes clean. What
/// that codeword stood for is then read out of the decoded audio with
/// this project's own transform.
///
/// A prefix code makes it cheap — nothing with a complete codeword as a
/// proper prefix can be one itself, so whole subtrees are skipped.
///
/// Books coding two coefficients at a time need a second, complete
/// codeword behind the candidate before a band of four will decode at
/// all. Its length is measured rather than assumed, and what it decodes
/// to does not matter: it occupies the coefficients the candidate does
/// not.
#[test]
#[ignore = "needs ffmpeg to answer, and asks it some thousands of times"]
fn the_codebooks_are_what_a_decoder_says_they_are() {
    use sovmr_audio::codebooks::{
        pair_index, quad_index, Codeword, PAIR_10, PAIR_5, PAIR_6, PAIR_7, PAIR_8, PAIR_9, QUAD_1,
        QUAD_2, QUAD_3, QUAD_4,
    };

    /// Every book, what it ships, the range it covers, and how many
    /// coefficients one of its codewords carries.
    const BOOKS_TO_CHECK: [(u32, &[Codeword], i32, usize); 10] = [
        (1, &QUAD_1, 1, 4),
        (2, &QUAD_2, 1, 4),
        (3, &QUAD_3, 2, 4),
        (4, &QUAD_4, 2, 4),
        (5, &PAIR_5, 4, 2),
        (6, &PAIR_6, 4, 2),
        (7, &PAIR_7, 7, 2),
        (8, &PAIR_8, 7, 2),
        (9, &PAIR_9, 12, 2),
        (10, &PAIR_10, 12, 2),
    ];
    use sovmr_audio::mdct::{mdct, window, WindowShape};

    let dir = probe_dir("books");
    let n = sovmr_audio::bands::LONG_FRAME;
    let shape = window(2 * n, WindowShape::Sine);

    for (codebook, shipped, range, carries) in BOOKS_TO_CHECK {
        // A band of four is filled by one quad codeword or two pair
        // ones, so a pair book needs a second, already-complete codeword
        // behind the candidate before the frame will decode at all.
        // Its length is measured, not assumed: the shortest run of zeros
        // that fills the band is exactly two of them.
        let behind = if carries == 4 {
            Vec::new()
        } else {
            let run = (1..=MAX_FILLER)
                .find(|run| {
                    let probe = Probe::at_band(0, codebook, QUIET_GAIN, filler(false, *run));
                    let (samples, complaints) = run_probe(&dir, "fill", &probe);
                    complaints.trim().is_empty() && samples.len() >= 2 * n
                })
                .unwrap_or_else(|| panic!("codebook {codebook}: no run of zeros fills the band"));
            assert_eq!(
                run % 2,
                0,
                "codebook {codebook}: {run} zero bits is not two equal codewords"
            );
            filler(false, run / 2)
        };

        let mut derived: Vec<Option<Codeword>> = vec![None; shipped.len()];
        let mut found: Vec<(u32, u8)> = Vec::new();
        let mut probes = 0usize;

        for length in 1..=LONGEST_CODEWORD {
            for value in 0..(1u64 << length) {
                let value = value as u32;
                // Anything with a complete codeword in front of it would
                // over-run, so the decoder need never be asked.
                if found.iter().any(|(bits, len)| {
                    u32::from(*len) <= length && value >> (length - u32::from(*len)) == *bits
                }) {
                    continue;
                }
                probes += 1;
                let mut bits: Vec<bool> = (0..length)
                    .map(|i| value >> (length - 1 - i) & 1 == 1)
                    .collect();
                bits.extend_from_slice(&behind);
                let probe = Probe::at_band(0, codebook, QUIET_GAIN, bits);
                let (samples, complaints) = run_probe(&dir, "book", &probe);
                if !complaints.trim().is_empty() || samples.len() < 2 * n {
                    continue;
                }

                let windowed: Vec<f64> = (0..2 * n).map(|i| samples[i] * shape[i]).collect();
                let mut coefficients = vec![0.0; n];
                mdct(&windowed, &mut coefficients);
                // Dequantisation is a power law, so reading a magnitude
                // back means inverting it. Rounding the ratio itself
                // would read a quantised two as a three.
                // Only the coefficients this codeword carries; whatever
                // sits behind it is somebody else's codeword.
                let mut tuple = vec![0i32; carries];
                for (k, slot) in tuple.iter_mut().enumerate() {
                    let magnitude = (coefficients[k].abs() / UNIT_MAGNITUDE).powf(0.75);
                    let rounded = magnitude.round();
                    assert!(
                        (magnitude - rounded).abs() < 0.05,
                        "codebook {codebook}, pattern {value:0width$b}: coefficient {k} read back \
                         as {magnitude}, which is not a quantised value",
                        width = length as usize
                    );
                    *slot = if coefficients[k] < 0.0 {
                        -(rounded as i32)
                    } else {
                        rounded as i32
                    };
                }

                let index = if carries == 4 {
                    quad_index([tuple[0], tuple[1], tuple[2], tuple[3]], range)
                } else {
                    pair_index([tuple[0], tuple[1]], range)
                }
                .unwrap_or_else(|| {
                    panic!("codebook {codebook} produced {tuple:?}, outside its own range")
                });
                assert!(
                    derived[index].is_none(),
                    "codebook {codebook}: {tuple:?} has two codewords, so the walk is wrong"
                );
                derived[index] = Some((value, length as u8));
                found.push((value, length as u8));
            }
        }

        let missing = derived.iter().filter(|entry| entry.is_none()).count();
        assert_eq!(
            missing,
            0,
            "codebook {codebook}: {missing} of {} tuples never appeared — the walk stopped short",
            shipped.len()
        );
        let derived: Vec<Codeword> = derived.into_iter().map(|e| e.expect("filled")).collect();
        assert_eq!(
            derived, shipped,
            "codebook {codebook}: the decoder disagrees with the table this project ships"
        );
        println!(
            "CODEBOOK {codebook} RE-DERIVED: {} codewords, complete and matching, from {probes} probes",
            derived.len()
        );
    }
}

/// Every scalefactor band table, re-derived from a decoder.
///
/// The tables this checks live in [`sovmr_audio::bands`] — all thirteen
/// rates the format can name — and they were produced by exactly this
/// measurement. Keeping the gate is what stops
/// it rotting: numbers nobody can re-derive are numbers nobody can
/// trust.
///
/// The method, per band: declare every band below it empty, put a single
/// coefficient in it, and ask where the coefficient landed. Two answers
/// come back at once — the position, read with our own transform, and
/// the width, which is the number of codewords it took to fill.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn the_band_table_is_what_a_decoder_says_it_is() {
    use sovmr_audio::bands::{long_bands, LONG_FRAME};

    let dir = probe_dir("bands");
    let n = LONG_FRAME;
    for rate in sovmr_audio::bands::MEASURED_RATES {
        let known = long_bands(rate).expect("a shipped rate has a table");
        derive_band_table(&dir, rate, known, n);
    }
}

/// Re-derive one rate's table and compare it against what is shipped.
fn derive_band_table(dir: &Path, rate: u32, known: &[usize], n: usize) {
    use sovmr_audio::bands::LONG_FRAME;
    use sovmr_audio::mdct::{mdct, window, WindowShape};
    {
        let shape = window(2 * n, WindowShape::Sine);
        // A pattern measured to set the last coefficient of the first
        // codeword in whichever band it is aimed at.
        let pattern: Vec<bool> = "10010".chars().map(|c| c == '1').collect();

        let mut derived = vec![0usize];
        for band in 0..known.len() - 1 {
            // The filler that exactly fills the band is the width
            // measurement: too little under-runs the frame, too much makes
            // the decoder read the terminator as data.
            let mut filled = None;
            for codewords in 0..MAX_FILLER {
                let mut bits = pattern.clone();
                bits.extend(filler(false, codewords));
                let probe = Probe::at_band(band as u32, 1, QUIET_GAIN, bits);
                let (samples, complaints) = run_probe_at(dir, "band", &probe, rate);
                if complaints.trim().is_empty() && samples.len() >= 2 * n {
                    filled = Some((codewords, samples));
                    break;
                }
            }
            let (codewords, samples) =
                filled.unwrap_or_else(|| panic!("band {band} could not be filled"));

            // Where did it land? Our own transform answers, so the reading
            // owes nothing to any assumption about the decoder's internals.
            let windowed: Vec<f64> = (0..2 * n).map(|i| samples[i] * shape[i]).collect();
            let mut coefficients = vec![0.0; n];
            mdct(&windowed, &mut coefficients);
            let (loudest, value) = coefficients
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .expect("a loudest coefficient");
            let elsewhere = coefficients
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != loudest)
                .map(|(_, v)| v.abs())
                .fold(0.0, f64::max);
            assert!(
            value.abs() > elsewhere * 1e5,
            "band {band}: coefficient {loudest} is only {:.0}x the rest, which is not a reading",
            value.abs() / elsewhere
        );

            // The pattern sets the last of the first codeword's four, so the
            // band starts three below where it landed.
            // `+ 1 -` rather than `- ... + 1`: band zero lands on
            // coefficient three, and the other order underflows.
            let start = loudest + 1 - COEFFICIENTS_PER_CODEWORD;
            assert_eq!(
                start, derived[band],
                "band {band} starts at {start}, but the band below it ended at {}",
                derived[band]
            );
            derived.push(start + (codewords + 1) * COEFFICIENTS_PER_CODEWORD);
        }

        // The check no single probe could make. Each band was measured on
        // its own; that all forty-nine add up to the frame length is a
        // statement about every one of them at once.
        assert_eq!(
            *derived.last().expect("not empty"),
            LONG_FRAME,
            "the measured bands total {} coefficients, not {LONG_FRAME}",
            derived.last().expect("not empty")
        );
        assert_eq!(
            derived, known,
            "at {rate} Hz the decoder disagrees with the table this project ships"
        );
        println!(
            "  {rate:6} Hz: {:2} bands, totalling {LONG_FRAME} coefficients, every one matching",
            derived.len() - 1
        );
    }
}

/// Our reading of a frame, against a frame we did not write.
///
/// The silent-frame gate proved ffmpeg can read what we write. This is
/// the other direction, and it is the one that catches a field width
/// that happens to be self-consistent: ffmpeg encodes a tone, and our
/// own reader walks the result field by field.
///
/// The check that has teeth is the section list. Each section names a
/// codebook and a run of bands, and the runs must add up to `max_sfb`
/// *exactly*. Nothing about that adds up by accident — one bit wrong
/// anywhere earlier and the counts land somewhere else entirely.
#[test]
#[ignore = "needs ffmpeg to make the frame"]
fn our_frame_layout_reads_a_frame_ffmpeg_wrote() {
    use sovmr_audio::bits::BitReader;
    let dir = probe_dir("real");
    let mkv = dir.join("real.mkv");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg("sine=frequency=1000:duration=1:sample_rate=48000")
        .args([
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            mkv.to_str().unwrap(),
        ])
        .status()
        .expect("ffmpeg runs");
    assert!(made.success(), "could not make the fixture");

    let data = std::fs::read(&mkv).expect("read the fixture");
    let seg = sovmr_mkv::demux::read_segment(&data).expect("segment");
    let (frames, err) = sovmr_mkv::blocks::read_frames(seg.payload);
    assert_eq!(err, None, "our own demuxer stumbled");

    let mut walked = 0usize;
    for frame in &frames {
        let mut r = BitReader::new(frame.data);
        // ffmpeg opens with a fill element; anything but a single
        // channel element is not what this gate is about.
        if r.u32(3) != Some(0) {
            continue;
        }
        r.u32(4); // element_instance_tag
        r.u32(8); // global_gain
        r.u32(1); // ics_reserved_bit
        let window_sequence = r.u32(2).expect("window_sequence");
        r.u32(1); // window_shape
                  // A short-window frame spends its next bits differently, and
                  // reading it with the long layout would be measuring nothing.
        if window_sequence == EIGHT_SHORT_SEQUENCE {
            continue;
        }
        let max_sfb = r.u32(6).expect("max_sfb");
        r.u32(1); // predictor_data_present
        if max_sfb == 0 {
            continue;
        }

        let mut covered = 0u32;
        while covered < max_sfb {
            let _codebook = r.u32(4).expect("sect_cb");
            let mut run = 0u32;
            loop {
                let increment = r.u32(SECT_LEN_BITS).expect("sect_len");
                run += increment;
                // An all-ones increment means the run continues, so a
                // run of exactly 31 has to be written as 31 then 0.
                if increment != (1 << SECT_LEN_BITS) - 1 {
                    break;
                }
            }
            assert!(run > 0, "a section covering no bands would never terminate");
            covered += run;
        }
        assert_eq!(
            covered, max_sfb,
            "sections covered {covered} bands but the frame declared {max_sfb} — \
             our layout has drifted somewhere before section_data"
        );
        walked += 1;
    }
    assert!(
        walked >= 8,
        "only {walked} long-window frames were walked; the fixture proves too little"
    );
    println!("OUR READER WALKS FFMPEG'S FRAMES: {walked} long-window frames, sections covering max_sfb exactly");
}

/// One coefficient, read back out of the decoded audio.
///
/// This is the readout every table extraction will rest on, so it is
/// worth pinning on its own. Identical frames make the spectrum constant
/// and the output periodic, which puts a single coefficient in a single
/// bin of a plain transform — no window, no leakage, no interpolation.
///
/// Two things are asserted. That the bin really is isolated, because a
/// readout that smeared across neighbours could not tell two adjacent
/// coefficients apart. And that its magnitude follows `2^(gain/4)`,
/// which is the dequantisation law the encoder will have to invert —
/// measured here rather than assumed.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn one_coefficient_comes_back_isolated_and_scales_by_the_quarter_power() {
    use sovmr_audio::fft::{transform, Complex};
    let dir = probe_dir("readout");

    let read_at = |gain: u32| -> (usize, f64, f64) {
        let probe = Probe::at_band(
            0,
            1,
            gain,
            // A pattern measured to set exactly one coefficient. Which
            // one it sets is a question for the extraction; that it sets
            // one is what this gate needs.
            "10010".chars().map(|c| c == '1').collect(),
        );
        let (samples, complaints) = run_probe(&dir, &format!("g{gain}"), &probe);
        assert!(complaints.trim().is_empty(), "gain {gain}: {complaints}");
        assert!(
            samples.len() >= STEADY_AT + READOUT_LEN,
            "gain {gain}: too short"
        );
        let mut buf: Vec<Complex> = samples[STEADY_AT..STEADY_AT + READOUT_LEN]
            .iter()
            .map(|s| Complex::new(*s, 0.0))
            .collect();
        transform(&mut buf, false);
        let magnitude = |bin: usize| buf[bin].magnitude() * 2.0 / READOUT_LEN as f64;
        let loudest = (1..READOUT_LEN / 2)
            .max_by(|a, b| magnitude(*a).partial_cmp(&magnitude(*b)).unwrap())
            .expect("a loudest bin");
        let elsewhere = (1..READOUT_LEN / 2)
            .filter(|b| *b != loudest)
            .map(magnitude)
            .fold(0.0, f64::max);
        (loudest, magnitude(loudest), elsewhere)
    };

    let (bin, quiet, elsewhere) = read_at(QUIET_GAIN);
    assert!(
        quiet > elsewhere * 1e6,
        "bin {bin} at {quiet:.3e} is only {:.0}x the next at {elsewhere:.3e} — \
         a readout this smeared cannot separate neighbouring coefficients",
        quiet / elsewhere
    );

    // The law. Four steps of gain double the amplitude, so twenty steps
    // must multiply it by thirty-two — and the same bin must carry it,
    // since changing the scale must not change which coefficient it is.
    let (louder_bin, loud, _) = read_at(QUIET_GAIN + GAIN_STEP);
    assert_eq!(
        louder_bin, bin,
        "the coefficient moved when the gain changed"
    );
    let ratio = loud / quiet;
    let expected = 2f64.powf(f64::from(GAIN_STEP) / 4.0);
    assert!(
        (ratio / expected - 1.0).abs() < 1e-6,
        "gain {GAIN_STEP} steps multiplied the coefficient by {ratio}, not {expected}"
    );
    println!(
        "COEFFICIENT READOUT: bin {bin} isolated {:.0e}x above the rest, \
         and {GAIN_STEP} gain steps scale it by exactly {expected}",
        quiet / elsewhere
    );
}

// ---------------------------------------------------------------------
// The decoder, judged by the decoder it was measured from.
//
// Reading our own encoder's frames back proves the two halves agree. It
// does not prove either is right about somebody else's file, and the
// whole point of a decoder here is somebody else's file. So the same
// arrangement as everywhere else in this project: ffmpeg produces the
// bitstream and ffmpeg says what it means; we must arrive at the same
// sound without asking it anything.
// ---------------------------------------------------------------------

/// Frames of a raw ADTS stream, headers stripped.
///
/// ADTS is how ffmpeg writes bare AAC. Inside an MP4 or a Matroska the
/// same blocks arrive already framed, which is the route this project
/// will actually take — the headers are stripped here only because a
/// file on disk needs some framing and this is the simplest that does
/// not drag in a container.
fn adts_frames(bytes: &[u8]) -> Vec<&[u8]> {
    const SYNC: u8 = 0xFF;
    const SHORT_HEADER: usize = 7;
    const HEADER_WITH_CRC: usize = 9;
    let mut frames = Vec::new();
    let mut at = 0usize;
    while at + SHORT_HEADER <= bytes.len() {
        assert_eq!(bytes[at], SYNC, "ADTS sync word at byte {at}");
        assert_eq!(bytes[at + 1] & 0xF0, 0xF0, "ADTS sync word at byte {at}");
        // A thirteen-bit length split across three bytes.
        let length = (usize::from(bytes[at + 3] & 0x03) << 11)
            | (usize::from(bytes[at + 4]) << 3)
            | (usize::from(bytes[at + 5]) >> 5);
        // The protection-absent bit is inverted: set means no CRC.
        let header = if bytes[at + 1] & 1 == 1 {
            SHORT_HEADER
        } else {
            HEADER_WITH_CRC
        };
        assert!(length >= header, "an ADTS frame shorter than its header");
        if at + length > bytes.len() {
            break;
        }
        frames.push(&bytes[at + header..at + length]);
        at += length;
    }
    frames
}

/// Have ffmpeg encode a signal to bare AAC, and decode the same file
/// back to samples. Both directions come from the oracle; ours is what
/// is on trial.
fn aac_probe(name: &str, source: &str, rate: u32, extra: &[&str]) -> (Vec<u8>, Vec<f64>) {
    let dir = probe_dir("decode");
    let coded = dir.join(format!("{name}.aac"));
    let raw = dir.join(format!("{name}.f64"));
    let rate_text = rate.to_string();
    let mut args = vec![
        "-v", "error", "-y", "-f", "lavfi", "-i", source, "-ar", &rate_text, "-ac", "1", "-c:a",
        "aac",
    ];
    args.extend_from_slice(extra);
    let made = Command::new("ffmpeg")
        .args(&args)
        .arg(&coded)
        .output()
        .expect("ffmpeg encodes");
    assert!(
        made.status.success(),
        "ffmpeg refused to encode: {}",
        String::from_utf8_lossy(&made.stderr)
    );
    let back = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&coded)
        .args(["-f", "f64le", "-ac", "1"])
        .arg(&raw)
        .output()
        .expect("ffmpeg decodes");
    assert!(
        back.status.success(),
        "ffmpeg refused to decode its own file: {}",
        String::from_utf8_lossy(&back.stderr)
    );
    let bytes = std::fs::read(&coded).expect("coded probe");
    let samples: Vec<f64> = std::fs::read(&raw)
        .expect("decoded probe")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    (bytes, samples)
}

/// The best whole-sample alignment of two signals, and the ratio of
/// signal to difference at it, in decibels.
///
/// The lag is searched rather than assumed. An earlier gate in this
/// project reported a catastrophic number because it compared the right
/// samples at the wrong offset, and the lesson was to measure the offset
/// instead of reasoning about it.
fn agreement_at_best_lag(
    reference: &[f64],
    ours: &[f64],
    base: usize,
    search: i64,
    span: (usize, usize),
) -> (i64, f64) {
    let (from, to) = span;
    let mut best = (0i64, f64::NEG_INFINITY);
    for lag in -search..=search {
        let mut signal = 0.0f64;
        let mut error = 0.0f64;
        let mut counted = 0usize;
        for i in from..to {
            let j = i as i64 - base as i64 + lag;
            if j < 0 || j as usize >= ours.len() || i >= reference.len() {
                continue;
            }
            let (a, b) = (reference[i], ours[j as usize]);
            signal += a * a;
            error += (a - b) * (a - b);
            counted += 1;
        }
        if counted * 2 < to - from || error <= 0.0 {
            continue;
        }
        let snr = 10.0 * (signal / error).log10();
        if snr > best.1 {
            best = (lag, snr);
        }
    }
    best
}

/// The longest stretch of frames this reader can take, and where in the
/// file it starts.
///
/// A real file switches to short windows at every transient, and those
/// are not built yet. Judging what *is* built needs an unbroken run —
/// a gap would shift everything after it by a frame and turn a correct
/// reading into a failing one for the wrong reason.
fn longest_readable_run<'a>(
    frames: &[&'a [u8]],
    rate: u32,
    reproducible_only: bool,
) -> (usize, Vec<&'a [u8]>) {
    let readable: Vec<bool> = frames
        .iter()
        .map(|f| match sovmr_audio::decode::decode_frame(f, rate) {
            // Both gates below go through synthesis, which this project
            // can only do for one long window so far — the reader takes
            // the other sequences apart correctly and putting them back
            // together is the next thing measured.
            //
            // A frame with noise bands is decoded correctly and still
            // cannot match another decoder sample for sample: those
            // bands carry an energy and nothing else. Judging sound
            // needs frames without them; judging energy does not.
            Ok(channels) => !reproducible_only || channels.iter().all(|c| c.noise_bands == 0),
            Err(_) => false,
        })
        .collect();
    let (mut best_start, mut best_len) = (0usize, 0usize);
    let mut at = 0usize;
    while at < readable.len() {
        if !readable[at] {
            at += 1;
            continue;
        }
        let start = at;
        while at < readable.len() && readable[at] {
            at += 1;
        }
        if at - start > best_len {
            (best_start, best_len) = (start, at - start);
        }
    }
    (
        best_start,
        frames[best_start..best_start + best_len].to_vec(),
    )
}

/// Read a run of frames ourselves and put the sound back together.
fn our_sound(frames: &[&[u8]], rate: u32) -> Vec<f64> {
    let channels: Vec<sovmr_audio::decode::Channel> = frames
        .iter()
        .map(|frame| {
            sovmr_audio::decode::decode_frame(frame, rate)
                .unwrap_or_else(|e| panic!("our decoder refused a real frame: {e:?}"))
                .remove(0)
        })
        .collect();
    // Through the crate's own synthesis, which refuses a sequence it
    // cannot put back together rather than approximating one. The shape
    // it uses comes out of the bitstream, one per block — assuming it
    // cost the first attempt at this gate sixty-five decibels.
    sovmr_audio::decode::synthesise(&channels)
}

/// Judge one real file: our reading of ffmpeg's frames against ffmpeg's
/// own reading of the same frames.
fn agreement_on(
    name: &str,
    source: &str,
    rate: u32,
    bitrate: &str,
    least: usize,
) -> (i64, f64, usize, usize) {
    let (coded, reference) = aac_probe(name, source, rate, &["-b:a", bitrate]);
    let frames = adts_frames(&coded);
    assert!(frames.len() > 100, "only {} frames in {name}", frames.len());
    let (start, run) = longest_readable_run(&frames, rate, true);
    // How long a run there is to judge is a property of the signal, not
    // a knob: noise substitution breaks the run wherever it appears, and
    // a transient signal reaches for it often. What matters is that the
    // run is long enough to hold several of whatever it is testing.
    assert!(
        run.len() >= least,
        "only {} unbroken readable frames in {name}",
        run.len()
    );
    // A gate that never met a short window would pass whatever the
    // synthesis did with them.
    let short = run
        .iter()
        .filter(|f| {
            sovmr_audio::decode::decode_frame(f, rate).expect("decodes")[0].sequence
                != sovmr_audio::decode::WindowSequence::OnlyLong
        })
        .count();
    let ours = our_sound(&run, rate);

    // Where the run sits in the file, plus the half-window at each end
    // of an overlap-add that has one contribution instead of two.
    let frame_length = sovmr_audio::bands::LONG_FRAME;
    let base = start * frame_length;
    let from = base + 2 * frame_length;
    let to = (base + ours.len()).min(reference.len()) - 2 * frame_length;
    assert!(
        to > from + frame_length,
        "not enough settled sound in {name}"
    );
    let (lag, snr) =
        agreement_at_best_lag(&reference, &ours, base, 2 * frame_length as i64, (from, to));
    (lag, snr, run.len(), short)
}

#[test]
fn our_decoder_arrives_at_the_same_sound_as_the_decoder_it_was_measured_from() {
    // The gate the whole AAC effort exists for. Nothing here came from
    // us: ffmpeg chose the codebooks, the scalefactors and the sections,
    // and ffmpeg says what they mean. Our reader must land on the same
    // sound using only tables it measured for itself.
    //
    // Two rates, because their band boundaries are different measured
    // tables, and one of them is the one everything else is compared at.
    for (name, source, rate, bitrate, least, must_switch) in [
        (
            "steady-48",
            "sine=frequency=440:duration=4",
            48_000u32,
            "128k",
            50usize,
            false,
        ),
        (
            "steady-44",
            "sine=frequency=1000:duration=4",
            44_100,
            "256k",
            50,
            true,
        ),
        // A decaying click every 120 ms: an encoder switches to short
        // windows at every one of them, which is the only way to put the
        // transition windows on trial rather than merely past them.
        (
            "clicks-48",
            r"aevalsrc='0.9*sin(6283*t)*exp(-40*mod(t\,0.12))':d=12",
            48_000,
            "320k",
            25,
            true,
        ),
    ] {
        let (lag, snr, frames, short) = agreement_on(name, source, rate, bitrate, least);
        assert!(
            !must_switch || short > 0,
            "{name} was meant to exercise the transition windows and met none"
        );
        println!(
            "{name}: {frames} frames ({short} not one long window), \
             lag {lag}, agreement {snr:.1} dB"
        );
        // What this number is: how closely two independent readings of
        // the same bitstream agree. It is not a quality figure — the
        // bitstream's own quantisation is already inside both, so what
        // is left is arithmetic. Measured at 138 dB on both rates, which
        // is double-precision rounding and nothing else; the bar is set
        // where a real disagreement would have to show.
        assert!(
            snr > 120.0,
            "{name}: our reading of ffmpeg's frames agrees to only {snr:.1} dB at lag {lag}"
        );
    }
}

/// How many frames of the census files are not one long window.
///
/// Counted rather than assumed. It is what makes the sound gates mean
/// anything about short windows: a gate that never met one would pass
/// whatever the synthesis did with them.
/// The files the census asks.
///
/// Across bitrates as well as signals, because an encoder reaches for
/// different tools at different rates. A census at one rate is a census
/// of one, and taking it for the whole format is exactly the overclaim
/// this list exists to prevent: temporal noise shaping never appears at
/// 128 kbit/s on these signals and is common at 320.
fn census_probes() -> Vec<(String, String, u32, &'static str)> {
    let mut probes = Vec::new();
    for (name, source, rate) in [
        ("sine", "sine=frequency=440:duration=4", 48_000u32),
        ("pink", "anoisesrc=duration=4:color=pink:seed=1", 48_000),
        ("white", "anoisesrc=duration=4:color=white:seed=2", 44_100),
        (
            "clicks",
            r"aevalsrc='0.9*sin(6283*t)*exp(-40*mod(t\,0.12))':d=4",
            48_000,
        ),
    ] {
        for bitrate in ["64k", "128k", "320k"] {
            probes.push((
                format!("census-{name}-{bitrate}"),
                source.to_string(),
                rate,
                bitrate,
            ));
        }
    }
    probes
}

fn frames_that_are_not_one_long_window() -> usize {
    let mut count = 0usize;
    for (name, source, rate, bitrate) in census_probes() {
        let (coded, _) = aac_probe(&name, &source, rate, &["-b:a", bitrate]);
        for frame in adts_frames(&coded) {
            let Ok(channels) = sovmr_audio::decode::decode_frame(frame, rate) else {
                continue;
            };
            if channels
                .iter()
                .any(|c| c.sequence != sovmr_audio::decode::WindowSequence::OnlyLong)
            {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn the_census_says_which_parts_of_aac_real_files_still_need() {
    // Which elements matter is not a question for the specification's
    // table of contents. It is a question for the files that exist, and
    // this asks them — so the order the remaining parts get
    // built in is measured rather than guessed.
    let mut census: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut total = 0usize;
    let mut decoded = 0usize;
    for (name, source, rate, bitrate) in census_probes() {
        let (coded, _) = aac_probe(&name, &source, rate, &["-b:a", bitrate]);
        for frame in adts_frames(&coded) {
            total += 1;
            match sovmr_audio::decode::unsupported_element(frame, rate) {
                Some(what) => *census.entry(what).or_default() += 1,
                None => decoded += 1,
            }
        }
    }
    assert!(total > 300, "only {total} frames to census");
    println!("{decoded} of {total} frames read");
    for (what, count) in &census {
        println!("  {count:5} still need {what}");
    }
    // What is claimed, and no more. An earlier version of this gate
    // asserted *every* frame read — true of the probes it had, which
    // were all at one bitrate, and an overclaim about the format.
    // Widening it across bitrates found temporal noise shaping at once.
    let share = decoded * 100 / total;
    assert!(share >= 80, "only {share}% of real frames read: {census:?}");
    // And how many of them are not one long window, which is what makes
    // the sound gates mean anything about short ones.
    let short = frames_that_are_not_one_long_window();
    println!("{short} of those carry a window sequence other than one long block");
    assert!(short > 0, "no probe file exercised a short window at all");
}

// ---------------------------------------------------------------------
// Perceptual noise substitution, walked out of a decoder.
//
// Codebook 13 is the one that carries no spectral data at all: the band
// is filled with noise at an energy the bitstream states. Sixty-two of
// the five hundred and fifty-two frames in the census need it, and it
// is the largest measured gap in the reader.
//
// Nothing about it is recalled here. How wide the energy field is, what
// it counts from and how it scales are all asked of a decoder, and each
// answer is bracketed on both sides.
// ---------------------------------------------------------------------

/// The codebook that means "noise of a stated energy".
const NOISE_BOOK: u32 = 13;
/// A band far enough up to be wide and clear of the first few.
const NOISE_BAND: u32 = 20;
/// How many copies of a probe frame to decode.
///
/// A single frame overlaps only silence, so half its window never
/// completes. Repeating it gives a steady state whose mean square is a
/// property of the frame rather than of where it sits.
const PROBE_REPEATS: usize = 24;

/// Mean square of the settled middle of a repeated-frame decode.
fn mean_square(samples: &[f64]) -> f64 {
    let skip = 4 * FRAME_LENGTH;
    if samples.len() <= 2 * skip {
        return 0.0;
    }
    let settled = &samples[skip..samples.len() - skip];
    settled.iter().map(|s| s * s).sum::<f64>() / settled.len() as f64
}

/// Decode a probe repeated into a steady state, and report its mean
/// square and anything the decoder had to say.
fn steady_state(dir: &Path, name: &str, probe: &Probe) -> (f64, String) {
    let path = dir.join(format!("{name}.mkv"));
    let frames = vec![probe.to_frame(); PROBE_REPEATS];
    mux_frames_at(&path, &frames, PROBE_RATE);
    let (samples, complaints) = decode(&path);
    (mean_square(&samples), complaints)
}

/// A probe whose band is noise, with `bits` handed to the energy field
/// verbatim — we do not know what it means yet, and the decoder is being
/// asked.
fn noise_probe(gain: u32, bits: Vec<bool>) -> Probe {
    Probe {
        sections: vec![(CODEBOOK_ZERO, NOISE_BAND), (NOISE_BOOK, 1)],
        global_gain: gain,
        scalefactor_bits: bits,
        spectral_bits: vec![],
    }
}

/// The bits of `value` in `width`, most significant first.
fn bits_of(value: u32, width: usize) -> Vec<bool> {
    (0..width)
        .map(|i| (value >> (width - 1 - i)) & 1 == 1)
        .collect()
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn walk_the_noise_energy_field() {
    let dir = probe_dir("pns");

    // How wide is the field? The terminator has to land where the
    // decoder expects it, so only the true width leaves a frame the
    // decoder does not complain about — and a band that is silent at
    // every width would mean the probe never reached the noise at all.
    println!("--- width of the first noise band's energy field ---");
    let mut clean = Vec::new();
    for width in 1..=MAX_FILLER {
        let probe = noise_probe(PROBE_GAIN, filler(false, width));
        let (power, complaints) = steady_state(&dir, &format!("w{width}"), &probe);
        if complaints.trim().is_empty() {
            clean.push(width);
        }
        println!(
            "{width:3} bits: mean square {power:.6e}  {}",
            complaints.lines().next().unwrap_or("")
        );
    }
    println!("clean widths: {clean:?}");

    // How the field scales. Both knobs are turned, because a law fitted
    // to one of them cannot tell which of the two it actually depends on.
    let width = clean[0];
    println!("--- energy against the field value, at gain {PROBE_GAIN} ---");
    for value in [300u32, 320, 340, 360, 380, 400, 402, 404, 406, 408] {
        let probe = noise_probe(PROBE_GAIN, bits_of(value, width));
        let (power, complaints) = steady_state(&dir, &format!("v{value}"), &probe);
        println!(
            "value {value:3}: mean square {power:.6e}  {}",
            complaints.lines().next().unwrap_or("")
        );
    }
    println!("--- energy against global_gain, at field value 380 ---");
    for gain in [80u32, 90, 100, 110, 120] {
        let probe = noise_probe(gain, bits_of(380, width));
        let (power, complaints) = steady_state(&dir, &format!("g{gain}"), &probe);
        println!(
            "gain {gain:3}: mean square {power:.6e}  {}",
            complaints.lines().next().unwrap_or("")
        );
    }
    // The constant the two sweeps leave undetermined. Ratios cannot give
    // it: something has to tie the decoder's output level to the units
    // our own coefficients are counted in.
    //
    // The ruler is our own encoder, which is not an assumption — it is
    // held to ffmpeg's own decoder at 138 dB in both directions by the
    // gate above. Noise against noise, so both sides carry the same
    // relation between coefficient energy and output power; a repeated
    // identical block would correlate with itself across the overlap and
    // quietly bias the constant.
    let (k, reference_power) = coefficient_energy_to_output_power(&dir);
    println!(
        "--- output power per unit of coefficient energy: {k:.6e} (from {reference_power:.6e}) ---"
    );
    for (gain, value) in [
        (PROBE_GAIN, 340u32),
        (PROBE_GAIN, 360),
        (PROBE_GAIN, 380),
        (90, 380),
        (110, 380),
    ] {
        let probe = noise_probe(gain, bits_of(value, width));
        let (power, _) = steady_state(&dir, &format!("c{gain}-{value}"), &probe);
        let energy = power / k;
        // If the law is 2^((gain + value - c) / 2), this is c.
        let c = gain as f64 + value as f64 - 2.0 * energy.log2();
        println!("gain {gain}, value {value}: band energy {energy:.6e}, constant {c:.3}");
    }
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn walk_the_rest_of_the_noise_rules() {
    let dir = probe_dir("pns2");
    let (k, _) = coefficient_energy_to_output_power(&dir);
    let energy_of = |dir: &Path, name: &str, probe: &Probe| {
        let (power, complaints) = steady_state(dir, name, probe);
        assert!(complaints.trim().is_empty(), "{name}: {complaints}");
        power / k
    };
    // The stated energy, in the units the band is counted in.
    let stated = |gain: u32, value: u32| gain as f64 + value as f64 - 410.008;

    // Where does it stop rising? A clamp is the kind of edge that a law
    // fitted in the middle of the range never sees.
    println!("--- the top of the range ---");
    for value in 396..=406u32 {
        let e = energy_of(
            &dir,
            &format!("t{value}"),
            &noise_probe(PROBE_GAIN, bits_of(value, 9)),
        );
        println!(
            "value {value}: 2^({:.2}/2), stated 2^({:.2}/2)",
            2.0 * e.log2(),
            stated(PROBE_GAIN, value)
        );
    }
    println!("--- the bottom of the range ---");
    for value in [140u32, 145, 150, 155, 160, 165, 170] {
        let e = energy_of(
            &dir,
            &format!("b{value}"),
            &noise_probe(PROBE_GAIN, bits_of(value, 9)),
        );
        println!(
            "value {value}: 2^({:.2}/2), stated 2^({:.2}/2)",
            2.0 * e.log2(),
            stated(PROBE_GAIN, value)
        );
    }

    // A second noise band cannot be another nine raw bits — that would
    // cost more than the band it replaces. What it costs instead is the
    // question, and the scalefactor book is the obvious suspect, so it
    // is put on trial rather than assumed.
    println!("--- a second noise band ---");
    for (label, delta_bits, expect) in [
        ("delta 0", vec![false], 0i32),
        ("delta +1", vec![true, false, true, false], 1),
        ("delta -1", vec![true, false, false], -1),
    ] {
        let mut bits = bits_of(360, 9);
        bits.extend(delta_bits);
        let probe = Probe {
            sections: vec![(CODEBOOK_ZERO, NOISE_BAND), (NOISE_BOOK, 2)],
            global_gain: PROBE_GAIN,
            scalefactor_bits: bits,
            spectral_bits: vec![],
        };
        let (power, complaints) = steady_state(&dir, &format!("s{expect}"), &probe);
        // Two bands, the first at a known energy; what is left is the
        // second's, and what it says about the delta.
        let one = 2f64.powf(stated(PROBE_GAIN, 360) / 2.0);
        let both = power / k;
        let second = (both - one).max(f64::MIN_POSITIVE);
        println!(
            "{label}: second band 2^({:.2}/2), first 2^({:.2}/2), difference {:.2}  {}",
            2.0 * second.log2(),
            stated(PROBE_GAIN, 360),
            2.0 * second.log2() - stated(PROBE_GAIN, 360),
            complaints.lines().next().unwrap_or("")
        );
        let _ = expect;
    }

    // Does the clamp stick to the running value, or apply only where it
    // is used? Two noise bands settle it: push the first past the top,
    // then step the second well down. A running value that was clamped
    // lands somewhere different from one that was not.
    println!("--- is the clamp remembered ---");
    // Sixteen below, written with the measured book: the delta is the
    // book's index minus sixty.
    let down = sovmr_audio::codebooks::SCALEFACTOR_DELTA
        [sovmr_audio::codebooks::scalefactor_index(-16).expect("in range")];
    let down_bits: Vec<bool> = (0..down.1)
        .map(|i| (down.0 >> (down.1 - 1 - i)) & 1 == 1)
        .collect();
    for over in [0u32, 20, 40] {
        let mut bits = bits_of(401 + over, 9);
        bits.extend(down_bits.iter().copied());
        let probe = Probe {
            sections: vec![(CODEBOOK_ZERO, NOISE_BAND), (NOISE_BOOK, 2)],
            global_gain: PROBE_GAIN,
            scalefactor_bits: bits,
            spectral_bits: vec![],
        };
        let both = energy_of(&dir, &format!("clamp{over}"), &probe);
        let first = 2f64.powf(stated(PROBE_GAIN, 401) / 2.0);
        let second = (both - first).max(f64::MIN_POSITIVE);
        println!(
            "{over} over the top, then -16: second band 2^({:.2}/2)  — clamped would be 2^({:.2}/2), not {:.2}",
            2.0 * second.log2(),
            stated(PROBE_GAIN, 401) - 16.0,
            stated(PROBE_GAIN, 401 + over) - 16.0
        );
    }

    // Do noise energies and ordinary scalefactors share a running value?
    // If they do, a loud noise band shifts every band after it — silent
    // corruption of the kind that sounds like a mixing decision.
    println!("--- do the two running values interfere ---");
    let word =
        sovmr_audio::codebooks::quad_codeword(&sovmr_audio::codebooks::QUAD_1, [1, 0, 0, 0], 1)
            .expect("a codeword for one unit");
    let spectral: Vec<bool> = (0..word.1)
        .map(|i| (word.0 >> (word.1 - 1 - i)) & 1 == 1)
        .collect();
    for (label, sections, scalefactors) in [
        (
            "plain, plain",
            vec![(1u32, 1u32), (1, 1)],
            vec![vec![false], vec![false]],
        ),
        (
            "plain, noise, plain",
            vec![(1, 1), (NOISE_BOOK, 1), (1, 1)],
            // An energy near the plain bands' own: a colossal one would
            // swamp the decoder's output and the small coefficients
            // would come back as rounding, not as an answer.
            vec![vec![false], bits_of(253, 9), vec![false]],
        ),
    ] {
        let mut bits = Vec::new();
        for part in &scalefactors {
            bits.extend(part.iter().copied());
        }
        let mut spectral_bits = Vec::new();
        for (cb, _) in &sections {
            if *cb != NOISE_BOOK {
                spectral_bits.extend(spectral.iter().copied());
            }
        }
        let probe = Probe {
            sections,
            global_gain: QUIET_GAIN,
            scalefactor_bits: bits,
            spectral_bits,
        };
        let path = dir.join(format!("{}.mkv", label.replace([' ', ','], "_")));
        mux_frames_at(&path, &vec![probe.to_frame(); PROBE_REPEATS], PROBE_RATE);
        let (samples, complaints) = decode(&path);
        // Read the coefficients back with our own transform, which the
        // gate above holds to this decoder at 138 dB.
        let blocks = sovmr_audio::mdct::analyse(
            &samples[4 * FRAME_LENGTH..samples.len() - 4 * FRAME_LENGTH],
            FRAME_LENGTH,
            sovmr_audio::mdct::WindowShape::Sine,
        );
        let bands = sovmr_audio::bands::long_bands(PROBE_RATE).expect("48 kHz");
        let peak = |block: &[f64], band: usize| {
            block[bands[band]..bands[band + 1]]
                .iter()
                .fold(0.0f64, |m, c| m.max(c.abs()))
        };
        let middle = &blocks[blocks.len() / 2];
        println!(
            "{label}: band 0 peak {:.6e}, last plain band peak {:.6e}  {}",
            peak(middle, 0),
            peak(middle, if label.contains("noise") { 2 } else { 1 }),
            complaints.lines().next().unwrap_or("")
        );
    }
}

/// How much output power one unit of coefficient energy makes.
///
/// Measured, not derived: the relation runs through the inverse
/// transform, the window and the overlap, and a constant assembled from
/// three remembered normalisations is three chances to be wrong.
fn coefficient_energy_to_output_power(dir: &Path) -> (f64, f64) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let signal: Vec<f64> = (0..40 * FRAME_LENGTH)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.1 * ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
        })
        .collect();
    let blocks =
        sovmr_audio::mdct::analyse(&signal, FRAME_LENGTH, sovmr_audio::mdct::WindowShape::Sine);
    let frames: Vec<Vec<u8>> = blocks
        .iter()
        .map(|b| sovmr_audio::encode::encode_frame(b, PROBE_RATE).expect("encodes"))
        .collect();
    // What the bitstream says, read back by us — not what went in, which
    // quantisation has already changed.
    let energy: f64 = frames
        .iter()
        .map(|f| {
            sovmr_audio::decode::decode_frame(f, PROBE_RATE).expect("decodes")[0]
                .spectrum
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
        })
        .sum::<f64>()
        / frames.len() as f64;
    let path = dir.join("ruler.mkv");
    mux_frames_at(&path, &frames, PROBE_RATE);
    let (samples, complaints) = decode(&path);
    assert!(
        complaints.trim().is_empty(),
        "the ruler itself: {complaints}"
    );
    let power = mean_square(&samples);
    (power / energy, power)
}

#[test]
fn noise_bands_carry_the_energy_the_bitstream_stated() {
    // Perceptual noise substitution says what a band's energy is and
    // nothing about its contents. Two correct decoders therefore
    // disagree on those bands sample for sample, by design — so the
    // sample gate above steps around them and this one takes them on
    // the only terms they have.
    //
    // Both signals are analysed the same way and compared band by band.
    // Nothing here asks our transform to invert theirs; it asks two
    // recordings of the same sound whether they hold the same energy
    // where the bitstream said they would.
    const RATE: u32 = 48_000;
    // The seed is not decoration. Without it this source draws fresh
    // noise every run, the encoder makes different choices, and the gate
    // judges a different file each time — which showed as three failures
    // in six before it was pinned. A probe that is not the same probe
    // twice is not a measurement.
    let (coded, reference) = aac_probe(
        "noisy",
        "anoisesrc=duration=6:color=pink:seed=3",
        RATE,
        &["-b:a", "96k"],
    );
    let frames = adts_frames(&coded);
    let (start, run) = longest_readable_run(&frames, RATE, false);
    assert!(run.len() > 100, "only {} readable frames", run.len());

    // The run has to actually exercise what this is about.
    let substituted: usize = run
        .iter()
        .map(|f| sovmr_audio::decode::decode_frame(f, RATE).expect("decodes")[0].noise_bands)
        .sum();
    assert!(
        substituted > 50,
        "only {substituted} noise bands in the run — this gate proved nothing"
    );

    let frame_length = sovmr_audio::bands::LONG_FRAME;
    let ours = our_sound(&run, RATE);
    let base = start * frame_length;
    // Our synthesis runs one window past the last frame's new samples;
    // the reference stops at the file's end. Judge what both hold.
    let together = ours.len().min(reference.len() - base);
    let theirs = &reference[base..base + together];
    let ours = &ours[..together];

    // Real frames declare the KBD window, which the sample gate above
    // confirms by agreeing to 138 dB when it is honoured.
    let shape = sovmr_audio::mdct::WindowShape::Kbd;
    let mine = sovmr_audio::mdct::analyse(ours, frame_length, shape);
    let yours = sovmr_audio::mdct::analyse(theirs, frame_length, shape);
    let blocks = mine.len().min(yours.len());
    assert!(blocks > 100, "only {blocks} blocks to compare");

    let bands = sovmr_audio::bands::long_bands(RATE).expect("a measured rate");
    let energy = |set: &[Vec<f64>], from: usize, to: usize| {
        // The ends of an overlap-add are incomplete either way.
        set[2..blocks - 2]
            .iter()
            .map(|b| b[from..to].iter().map(|c| c * c).sum::<f64>())
            .sum::<f64>()
    };
    let per_band: Vec<(f64, f64)> = (0..bands.len() - 1)
        .map(|band| {
            let (from, to) = (bands[band], bands[band + 1]);
            (energy(&mine, from, to), energy(&yours, from, to))
        })
        .collect();

    // Bands the encoder coded nothing into: ours are exactly zero, and
    // theirs hold whatever the reference's own 32-bit samples round to.
    // Measured, that residue sits 130 dB under the loudest band — below
    // what those samples can represent, so comparing it would measure
    // the probe file's format and not the decoder.
    const AUDIBLE_FLOOR: f64 = 1e-12;
    let loudest = per_band.iter().fold(0.0f64, |m, (a, b)| m.max(a.max(*b)));
    let mut worst = (0usize, 0.0f64);
    let mut judged = 0usize;
    for (band, (a, b)) in per_band.iter().enumerate() {
        if a.max(*b) < loudest * AUDIBLE_FLOOR {
            continue;
        }
        judged += 1;
        let difference = 10.0 * (a / b).log10();
        if difference.abs() > worst.1.abs() {
            worst = (band, difference);
        }
    }
    println!(
        "{judged} bands judged over {blocks} blocks, {substituted} noise bands; \
         worst is band {} at {:+.2} dB",
        worst.0, worst.1
    );
    assert!(judged > 30, "only {judged} bands carried anything");
    // Measured at 0.00 dB on every band carrying real energy, noise
    // bands included; the top band sits near the floor, holds almost
    // nothing, and comes in at a quarter of a decibel. The bar is set
    // where a real disagreement would have to show.
    assert!(
        worst.1.abs() < 0.5,
        "band {} holds {:+.2} dB more energy than the decoder that wrote it said",
        worst.0,
        worst.1
    );
}

// ---------------------------------------------------------------------
// Short windows, from nothing.
//
// The last gap the census names, and the only one where not a single
// number is known: not the field widths, not what the grouping bits
// mean, not the band boundaries. A long block's `ics_info` was measured
// in stage 4; the short one shares only its first four bits.
//
// So the walk starts where every other one in this file started — with a
// frame the decoder is asked to read, and a sweep wide enough that the
// right answer has to be inside it.
// ---------------------------------------------------------------------

/// A short-window probe with every unknown width as a parameter.
///
/// Nothing here is believed. `max_sfb_bits`, `grouping_bits` and
/// `sect_len_bits` are swept, and only the combination the decoder reads
/// as sound can be right.
#[derive(Debug, Clone)]
struct ShortProbe {
    max_sfb_bits: u32,
    grouping_bits: u32,
    sect_len_bits: u32,
    grouping: u32,
    max_sfb: u32,
    /// `(codebook, bands)` per section, repeated for each group.
    sections: Vec<(u32, u32)>,
    groups: usize,
    /// Scalefactor deltas and spectral bits, verbatim.
    scalefactor_bits: Vec<bool>,
    spectral_bits: Vec<bool>,
    global_gain: u32,
}

impl ShortProbe {
    fn to_frame(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write(0x0, 3); // ID_SCE
        w.write(0, 4); // element_instance_tag
        w.write(self.global_gain, 8);
        w.write(0, 1); // ics_reserved_bit
        w.write(EIGHT_SHORT_SEQUENCE, 2);
        w.write(0, 1); // window_shape
        w.write(self.max_sfb, self.max_sfb_bits);
        w.write(self.grouping, self.grouping_bits);
        // section_data, once per group.
        for _ in 0..self.groups {
            for (codebook, bands) in &self.sections {
                w.write(*codebook, 4);
                let escape = (1u32 << self.sect_len_bits) - 1;
                let mut left = *bands;
                while left >= escape {
                    w.write(escape, self.sect_len_bits);
                    left -= escape;
                }
                w.write(left, self.sect_len_bits);
            }
        }
        for bit in &self.scalefactor_bits {
            w.write(u32::from(*bit), 1);
        }
        w.write(0, 1); // pulse_data_present
        w.write(0, 1); // tns_data_present
        w.write(0, 1); // gain_control_data_present
        for bit in &self.spectral_bits {
            w.write(u32::from(*bit), 1);
        }
        w.write(0x7, 3); // ID_END
        w.finish()
    }
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn walk_the_short_window_header() {
    let dir = probe_dir("short");

    // Step one: how many bits the header spends after `window_shape`.
    //
    // A frame declaring no bands at all has no section data, no
    // scalefactors and no spectral data — so the only thing that can
    // put the terminator anywhere but where the decoder looks for it is
    // the width of the two fields left. Their *sum* is what this reads;
    // splitting it takes a second probe, because zeros are zeros.
    println!("--- bits after window_shape, from a frame declaring no bands ---");
    let mut sums = Vec::new();
    for total in 2..=20u32 {
        let mut w = BitWriter::new();
        w.write(0x0, 3);
        w.write(0, 4);
        w.write(QUIET_GAIN, 8);
        w.write(0, 1);
        w.write(EIGHT_SHORT_SEQUENCE, 2);
        w.write(0, 1);
        w.write(0, total);
        w.write(0, 3);
        w.write(0x7, 3);
        let path = dir.join(format!("empty{total}.mkv"));
        mux_frames_at(&path, &vec![w.finish(); PROBE_FRAMES], PROBE_RATE);
        let (_, complaints) = decode(&path);
        if complaints.trim().is_empty() {
            sums.push(total);
        }
        println!(
            "{total:2} bits: {}",
            complaints.lines().next().unwrap_or("clean")
        );
    }
    println!("clean sums: {sums:?}");
    let Some(total) = sums.first().copied() else {
        panic!("no width let an empty short frame through");
    };

    // Step two: split that sum, and find how many times the section
    // data repeats. One band coded with the empty codebook needs
    // section data and nothing else, so the only unknowns left are
    // where `max_sfb` ends, how wide a section length is, and how many
    // groups the decoder thinks eight windows fall into.
    println!("--- splitting the sum, and counting groups ---");
    let mut clean = Vec::new();
    for max_sfb_bits in 1..total {
        for sect_len_bits in 1..=8u32 {
            for groups in 1..=8usize {
                let probe = ShortProbe {
                    max_sfb_bits,
                    grouping_bits: total - max_sfb_bits,
                    sect_len_bits,
                    grouping: 0,
                    max_sfb: 1,
                    sections: vec![(CODEBOOK_ZERO, 1)],
                    groups,
                    scalefactor_bits: vec![],
                    spectral_bits: vec![],
                    global_gain: QUIET_GAIN,
                };
                let path = dir.join(format!("s{max_sfb_bits}-{sect_len_bits}-{groups}.mkv"));
                mux_frames_at(&path, &vec![probe.to_frame(); PROBE_FRAMES], PROBE_RATE);
                let (_, complaints) = decode(&path);
                if complaints.trim().is_empty() {
                    clean.push((max_sfb_bits, sect_len_bits, groups));
                }
            }
        }
    }
    for (m, l, g) in &clean {
        println!(
            "max_sfb {m} bits, grouping {} bits, sect_len {l} bits, {g} groups",
            total - m
        );
    }
    println!("{} combinations read cleanly", clean.len());
}

/// The header widths the sweep above settled, used by every probe below.
const SHORT_MAX_SFB_BITS: u32 = 4;
const SHORT_GROUPING_BITS: u32 = 7;
const SHORT_SECT_LEN_BITS: u32 = 3;
/// Windows in an eight-short frame.
const SHORT_WINDOWS: usize = 8;
/// Coefficients one short window holds — what every short band table
/// must total.
const SHORT_COEFFICIENTS: usize = 128;

/// A short probe with the measured header widths and a chosen grouping.
fn short_probe(
    grouping: u32,
    groups: usize,
    max_sfb: u32,
    sections: Vec<(u32, u32)>,
) -> ShortProbe {
    ShortProbe {
        max_sfb_bits: SHORT_MAX_SFB_BITS,
        grouping_bits: SHORT_GROUPING_BITS,
        sect_len_bits: SHORT_SECT_LEN_BITS,
        grouping,
        max_sfb,
        sections,
        groups,
        scalefactor_bits: vec![],
        spectral_bits: vec![],
        global_gain: QUIET_GAIN,
    }
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn walk_what_the_grouping_bits_mean() {
    // Seven bits, eight windows: each bit plainly speaks about one
    // boundary between neighbours. Which way round is the question, and
    // the section data answers it — a decoder reads one section list per
    // group, so the number of lists it demands *is* the group count.
    let dir = probe_dir("grouping");
    println!("--- groups demanded, by grouping value ---");
    let mut rule_holds = true;
    for grouping in 0..128u32 {
        let found = (1..=SHORT_WINDOWS).find(|groups| {
            let probe = short_probe(grouping, *groups, 1, vec![(CODEBOOK_ZERO, 1)]);
            let path = dir.join(format!("g{grouping}-{groups}.mkv"));
            mux_frames_at(&path, &vec![probe.to_frame(); PROBE_FRAMES], PROBE_RATE);
            decode(&path).1.trim().is_empty()
        });
        // The reading under test: a set bit joins a window to the group
        // before it, so the count is eight less the bits set.
        let expected = SHORT_WINDOWS - grouping.count_ones() as usize;
        if found != Some(expected) {
            rule_holds = false;
            println!("grouping {grouping:3} ({grouping:07b}): {found:?} groups, not {expected}");
        }
    }
    println!(
        "{}",
        if rule_holds {
            "every one of the 128 groupings demands eight lists less its set bits"
        } else {
            "the rule above does not hold — see the lines printed"
        }
    );
}

/// Bits of a codeword, most significant first.
fn codeword_bits(word: sovmr_audio::codebooks::Codeword) -> Vec<bool> {
    (0..word.1)
        .map(|i| (word.0 >> (word.1 - 1 - i)) & 1 == 1)
        .collect()
}

/// Ask a decoder how wide one short band is, by counting the codewords
/// it takes to fill.
///
/// Eight groups of one window each, every band below the one of interest
/// declared empty, and the band itself filled with codebook-1 codewords.
/// Too few leaves the decoder mid-band; too many makes it read the
/// terminator as data. The fit is exact from both sides, which is what
/// makes the answer a measurement and not a guess.
fn short_band_width(dir: &Path, rate: u32, band: u32, longest: usize) -> Option<usize> {
    let unit = codeword_bits(
        sovmr_audio::codebooks::quad_codeword(&sovmr_audio::codebooks::QUAD_1, [1, 0, 0, 0], 1)
            .expect("a codeword for one unit"),
    );
    let zero = codeword_bits(
        sovmr_audio::codebooks::quad_codeword(&sovmr_audio::codebooks::QUAD_1, [0, 0, 0, 0], 1)
            .expect("a codeword for silence"),
    );
    let mut sections = Vec::new();
    if band > 0 {
        sections.push((CODEBOOK_ZERO, band));
    }
    sections.push((1, 1));
    for words in 1..=longest {
        let mut probe = short_probe(0, SHORT_WINDOWS, band + 1, sections.clone());
        // One delta per group for the one coded band, all zero.
        probe.scalefactor_bits = filler(false, SHORT_WINDOWS);
        // The first codeword of each group carries a unit, so a frame
        // that reads is also a frame that sounds — silence alone could
        // be silence for any number of reasons.
        let per_group = words;
        let mut bits = Vec::new();
        for _ in 0..SHORT_WINDOWS {
            bits.extend(unit.iter().copied());
            for _ in 1..per_group {
                bits.extend(zero.iter().copied());
            }
        }
        probe.spectral_bits = bits;
        let path = dir.join(format!("w{rate}-{band}-{words}.mkv"));
        mux_frames_at(&path, &vec![probe.to_frame(); PROBE_FRAMES], rate);
        let (samples, complaints) = decode(&path);
        let peak = samples.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        if complaints.trim().is_empty() && peak > 1e-9 {
            return Some(words * COEFFICIENTS_PER_CODEWORD);
        }
    }
    None
}

/// The short band tables, re-derived from a decoder and compared
/// against what is shipped.
///
/// The same arrangement that keeps the long tables honest: numbers
/// nobody can re-derive are numbers nobody can trust. Each band is
/// probed on its own, and the total of 128 is a check nothing during the
/// walk enforced.
#[test]
#[ignore = "needs ffmpeg to answer"]
fn the_short_band_table_is_what_a_decoder_says_it_is() {
    let dir = probe_dir("shortbands");
    let mut tables: Vec<(u32, Vec<usize>)> = Vec::new();
    for rate in sovmr_audio::bands::MEASURED_RATES {
        let known = sovmr_audio::bands::short_bands(rate).expect("a shipped rate has a table");
        let mut boundaries = vec![0usize];
        for band in 0..32u32 {
            let Some(width) = short_band_width(&dir, rate, band, 32) else {
                panic!("{rate} Hz band {band}: no codeword count fits");
            };
            let last = *boundaries.last().unwrap();
            boundaries.push(last + width);
            if last + width >= SHORT_COEFFICIENTS {
                break;
            }
        }
        assert_eq!(
            *boundaries.last().unwrap(),
            SHORT_COEFFICIENTS,
            "{rate} Hz totals {} coefficients, not {SHORT_COEFFICIENTS}",
            boundaries.last().unwrap()
        );
        assert_eq!(boundaries, known, "{rate} Hz");
        println!(
            "{rate:6} Hz: {} bands, re-derived and identical",
            boundaries.len() - 1
        );
        tables.push((rate, boundaries));
    }

    // Which rates share a table was not assumed either — each was walked
    // on its own and they are compared here. The grouping differs from
    // the long side's, which is the trap this records.
    let distinct: std::collections::BTreeSet<&Vec<usize>> = tables.iter().map(|(_, t)| t).collect();
    assert_eq!(distinct.len(), 5, "five distinct short tables");
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn walk_how_a_group_interleaves_its_windows() {
    // A group holding several windows stores their coefficients in one
    // run. Whether that run walks window by window or coefficient by
    // coefficient is a fork with no symptom other than the sound coming
    // out in the wrong order — so it is asked, not assumed.
    //
    // One group of all eight windows, one band four coefficients wide,
    // eight codewords. A unit is placed in each in turn and the decoder
    // is asked *when* the sound arrives.
    let dir = probe_dir("interleave");
    let unit = codeword_bits(
        sovmr_audio::codebooks::quad_codeword(&sovmr_audio::codebooks::QUAD_1, [1, 0, 0, 0], 1)
            .expect("a codeword for one unit"),
    );
    let zero = codeword_bits(
        sovmr_audio::codebooks::quad_codeword(&sovmr_audio::codebooks::QUAD_1, [0, 0, 0, 0], 1)
            .expect("a codeword for silence"),
    );
    // All seven bits set: every window joins the one before it, so there
    // is one group of eight — the rule the sweep above established.
    const ONE_GROUP: u32 = 0b111_1111;
    println!("--- where in the frame each codeword's sound lands ---");
    for slot in 0..SHORT_WINDOWS {
        let mut probe = short_probe(ONE_GROUP, 1, 1, vec![(1, 1)]);
        probe.scalefactor_bits = vec![false];
        let mut bits = Vec::new();
        for k in 0..SHORT_WINDOWS {
            bits.extend(if k == slot { &unit } else { &zero }.iter().copied());
        }
        probe.spectral_bits = bits;
        let path = dir.join(format!("i{slot}.mkv"));
        mux_frames_at(&path, &vec![probe.to_frame(); 12], PROBE_RATE);
        let (samples, complaints) = decode(&path);
        assert!(
            complaints.trim().is_empty(),
            "codeword {slot}: {complaints}"
        );
        // One steady period. Identical frames make the output periodic
        // at the frame length, so this is one whole period well past the
        // decoder's priming.
        let period = &samples[STEADY_AT..STEADY_AT + FRAME_LENGTH];
        // The energy of each short window's slice of that period.
        let hop = FRAME_LENGTH / SHORT_WINDOWS;
        let energies: Vec<f64> = (0..SHORT_WINDOWS)
            .map(|w| period[w * hop..(w + 1) * hop].iter().map(|s| s * s).sum())
            .collect();
        let loudest = energies
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let total: f64 = energies.iter().sum();
        println!(
            "codeword {slot}: loudest eighth is {loudest}, holding {:.1}% of the frame",
            100.0 * energies[loudest] / total.max(f64::MIN_POSITIVE)
        );
    }
}

// ---------------------------------------------------------------------
// What re-encoding is actually worth.
//
// The decoder was built to turn a 640 kbit/s track into a smaller one.
// Before any of that is plumbed into the pipeline, the lever itself gets
// measured: decode somebody else's AAC, re-encode it with ours, and see
// what comes out — in bytes and in decibels.
//
// Our encoder has no rate control at all. It picks the smallest
// scalefactor that keeps a frame inside the quantiser and spends
// whatever that costs. Whether that is smaller or larger than what a
// real encoder spent is not a matter of opinion.
// ---------------------------------------------------------------------

/// The largest per-band energy difference between two recordings of the
/// same sound, in decibels.
///
/// The only fair judgement of a band replaced by noise: its contents are
/// meant to differ, and its energy is not.
fn worst_band_energy_difference(a: &[f64], b: &[f64], rate: u32) -> f64 {
    let n = sovmr_audio::bands::LONG_FRAME;
    let shape = sovmr_audio::mdct::WindowShape::Sine;
    let (mine, theirs) = (
        sovmr_audio::mdct::analyse(a, n, shape),
        sovmr_audio::mdct::analyse(b, n, shape),
    );
    let blocks = mine.len().min(theirs.len());
    if blocks < 6 {
        return f64::INFINITY;
    }
    let bands = sovmr_audio::bands::long_bands(rate).expect("a measured rate");
    let energy = |set: &[Vec<f64>], from: usize, to: usize| {
        set[2..blocks - 2]
            .iter()
            .map(|block| block[from..to].iter().map(|c| c * c).sum::<f64>())
            .sum::<f64>()
    };
    let per_band: Vec<(f64, f64)> = bands
        .windows(2)
        .map(|pair| {
            (
                energy(&mine, pair[0], pair[1]),
                energy(&theirs, pair[0], pair[1]),
            )
        })
        .collect();
    // Bands near the floor say nothing about either side; the same
    // relative floor the reading gate uses.
    let loudest = per_band.iter().fold(0.0f64, |m, (x, y)| m.max(x.max(*y)));
    let mut worst = 0.0f64;
    for (x, y) in &per_band {
        if x.max(*y) < loudest * 1e-12 {
            continue;
        }
        let difference = 10.0 * (y / x).log10();
        if difference.abs() > worst.abs() {
            worst = difference;
        }
    }
    worst
}

#[test]
#[ignore = "needs ffmpeg to answer"]
fn what_re_encoding_costs_today() {
    println!("--- source, ours, and the difference ---");
    // The knob, swept. Coarser is smaller; what it costs in decibels
    // is the other half of the answer, and the reason a default cannot
    // be chosen by taste.
    for (target, flatness) in [
        (2047i32, sovmr_audio::encode::NEVER_SUBSTITUTE),
        (2047, 0.7),
        (2047, 0.5),
        (2047, 0.3),
        (2047, 0.15),
        (511, 0.5),
    ] {
        println!("target {target}, flatness {flatness}:");
        for (name, source, rate, bitrate) in [
            (
                "re-tone",
                "sine=frequency=440:duration=6",
                48_000u32,
                "128k",
            ),
            ("re-rich", "sine=frequency=440:duration=6", 48_000, "320k"),
            (
                "re-noise",
                "anoisesrc=duration=6:color=pink:seed=4",
                48_000,
                "320k",
            ),
            (
                "re-clicks",
                r"aevalsrc='0.9*sin(6283*t)*exp(-40*mod(t\,0.12))':d=6",
                48_000,
                "320k",
            ),
        ] {
            let (coded, _) = aac_probe(name, source, rate, &["-b:a", bitrate]);
            let all = adts_frames(&coded);
            // The longest unbroken stretch this reader takes: a gap would
            // shift the sound after it and turn a fair comparison into an
            // unfair one.
            let (_, frames) = longest_readable_run(&all, rate, false);
            let channels: Vec<sovmr_audio::decode::Channel> = frames
                .iter()
                .map(|f| {
                    sovmr_audio::decode::decode_frame(f, rate)
                        .expect("reads")
                        .remove(0)
                })
                .collect();
            let sound = sovmr_audio::decode::synthesise(&channels);

            // Round it back through our own analysis and encoder, which is
            // the route a re-encode would actually take.
            let blocks = sovmr_audio::mdct::analyse(
                &sound,
                sovmr_audio::bands::LONG_FRAME,
                sovmr_audio::mdct::WindowShape::Sine,
            );
            let ours: Vec<Vec<u8>> = blocks
                .iter()
                .map(|b| {
                    sovmr_audio::encode::encode_frame_tuned(b, rate, target, flatness)
                        .expect("encodes")
                })
                .collect();
            let theirs_bytes: usize = frames.iter().map(|f| f.len()).sum();
            let ours_bytes: usize = ours.iter().map(|f| f.len()).sum();
            let seconds =
                frames.len() as f64 * sovmr_audio::bands::LONG_FRAME as f64 / f64::from(rate);
            let kbit = |bytes: usize| bytes as f64 * 8.0 / seconds / 1000.0;

            // And what it costs in sound: our re-encode against the sound we
            // were handed, which is the only thing a re-encode can be judged
            // against — the original is already gone.
            let back: Vec<Vec<f64>> = ours
                .iter()
                .map(|f| {
                    sovmr_audio::decode::decode_frame(f, rate)
                        .expect("reads")
                        .remove(0)
                        .spectrum
                })
                .collect();
            let again = sovmr_audio::mdct::synthesise_shaped(
                &back,
                &vec![sovmr_audio::mdct::WindowShape::Sine; back.len()],
                sovmr_audio::bands::LONG_FRAME,
            );
            let n = sovmr_audio::bands::LONG_FRAME;
            let (from, to) = (2 * n, sound.len().min(again.len()) - 2 * n);
            let mut signal = 0.0f64;
            let mut error = 0.0f64;
            for i in from..to {
                signal += sound[i] * sound[i];
                let d = sound[i] - again[i];
                error += d * d;
            }
            // Two numbers, because one cannot say it.
            //
            // Sample-level agreement is meaningless for a band replaced
            // by noise: the energy is right and the waveform is, by
            // design, different. It is still reported, because it
            // catches a quantiser fault the other number would not.
            //
            // What a substituted band *can* be judged on is what the
            // reading side judges it on — does it carry the energy it
            // was meant to? So the second number is the worst per-band
            // energy difference across the whole spectrum.
            let snr = 10.0 * (signal / error).log10();
            let worst = worst_band_energy_difference(&sound[from..to], &again[from..to], rate);
            println!(
                "{name:10} {bitrate:>5}: theirs {:6.1}, ours {:6.1} kbit/s ({:+6.1}%), \
                 {snr:6.1} dB samples, {worst:+6.2} dB worst band",
                kbit(theirs_bytes),
                kbit(ours_bytes),
                100.0 * (ours_bytes as f64 / theirs_bytes as f64 - 1.0),
            );
        }
    }
}
