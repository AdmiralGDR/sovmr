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

//! Lossless decoders against the only standard that means anything.
//!
//! A lossy decoder can be judged by ear or by PSNR. A lossless one
//! cannot: it is either exact or it is broken, and "close" is a
//! euphemism. So every sample this crate produces is compared with the
//! same sample produced by ffmpeg, on real signals rather than on
//! fixtures written to agree with us.
//!
//! ```sh
//! cargo test -p sovmr-audio --test bit_exact -- --ignored --nocapture
//! ```

use std::path::Path;
use std::process::Command;

use sovmr_audio::pcm::{decode, PcmFormat};

/// Seconds of audio per case. Long enough to cross many blocks, short
/// enough that a mismatch is findable by hand.
const SECONDS: u32 = 3;
/// Sampling rate the fixtures are generated at.
const RATE: u32 = 48_000;

/// Render a test signal to a raw PCM file in `layout`, via ffmpeg.
///
/// Every signal must be *reproducible*: the same filter rendered twice
/// has to give the same samples, or the comparison is between two
/// different recordings. That is why the noise source is seeded.
fn render(dir: &Path, name: &str, filter: &str, layout: &str) -> Vec<u8> {
    let path = dir.join(name);
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg(filter)
        .args(["-f", layout, "-ar", &RATE.to_string(), "-ac", "1"])
        .arg(&path)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success(), "ffmpeg could not render {name}");
    std::fs::read(&path).unwrap()
}

/// The signals worth testing: each stresses something different.
fn signals() -> Vec<(&'static str, String)> {
    vec![
        // A tone visits a known waveform, so a sample-order error shows.
        (
            "tone",
            format!("sine=frequency=997:duration={SECONDS}:sample_rate={RATE}"),
        ),
        // Noise has no structure to accidentally agree with — and a
        // fixed seed, because every render must be the same signal.
        // Without one, ffmpeg produces fresh noise per invocation and
        // the test compares two different recordings, which fails for a
        // reason that has nothing to do with the decoder.
        (
            "noise",
            format!("anoisesrc=duration={SECONDS}:sample_rate={RATE}:amplitude=0.9:seed=20260803"),
        ),
        // Silence catches an offset error a loud signal hides — reading
        // 8-bit as signed, say, turns silence into full scale.
        (
            "silence",
            format!("anullsrc=duration={SECONDS}:sample_rate={RATE}"),
        ),
    ]
}

#[test]
#[ignore = "needs ffmpeg to render the fixtures"]
fn every_pcm_layout_decodes_to_the_same_samples_ffmpeg_produces() {
    let dir = std::env::temp_dir().join(format!("sovmr-pcm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // (ffmpeg's raw format name, our layout, bits it actually resolves)
    let layouts = [
        ("u8", PcmFormat::U8, 8u32),
        ("s16le", PcmFormat::S16Le, 16),
        ("s16be", PcmFormat::S16Be, 16),
        ("s24le", PcmFormat::S24Le, 24),
        ("s24be", PcmFormat::S24Be, 24),
        ("s32le", PcmFormat::S32Le, 32),
        ("s32be", PcmFormat::S32Be, 32),
        ("f32le", PcmFormat::F32Le, 24),
        ("f32be", PcmFormat::F32Be, 24),
    ];

    let mut checked = 0usize;
    for (name, filter) in signals() {
        // The reference: the same signal at full integer width, decoded
        // by ffmpeg. Every layout must agree with it once scaled.
        let reference_bytes = render(&dir, &format!("{name}.ref"), &filter, "s32le");
        let reference = decode(&reference_bytes, PcmFormat::S32Le);
        assert!(!reference.is_empty(), "{name}: ffmpeg rendered nothing");

        for (ff_name, format, bits) in layouts {
            let bytes = render(&dir, &format!("{name}.{ff_name}"), &filter, ff_name);
            let mine = decode(&bytes, format);
            assert_eq!(
                mine.len(),
                reference.len(),
                "{name}/{ff_name}: sample count"
            );

            // A narrower layout has thrown bits away, and how ffmpeg
            // rounded is its own business — so the two are compared at
            // the width the narrow one actually resolves. A float
            // carries 24 bits of mantissa, so it is judged there too.
            let shift = 32 - bits;
            let mut worst = 0i64;
            for (i, (a, b)) in mine.iter().zip(&reference).enumerate() {
                let (a, b) = (i64::from(*a >> shift), i64::from(*b >> shift));
                let drift = (a - b).abs();
                worst = worst.max(drift);
                assert!(
                    drift <= 1,
                    "{name}/{ff_name}: sample {i} is {a}, ffmpeg says {b}"
                );
            }
            // Reported rather than merely asserted: a run where every
            // layout drifts by exactly one is a rounding convention, and
            // a run where they are all zero says something stronger.
            println!("  {name:8} {ff_name:6} worst drift {worst} at {bits} bits");
            // Silence must be *exactly* silence in every layout — this
            // is where an 8-bit sign error or a stray offset shows.
            if name == "silence" {
                assert!(
                    mine.iter().all(|s| *s == 0),
                    "{ff_name}: silence must decode to zero, not to an offset"
                );
            }
            checked += 1;
        }
    }

    // The exact half of the claim. Comparing across widths can only
    // ever be approximate — ffmpeg rounded twice from one source, and
    // that residual is its arithmetic, not ours. But a big-endian file
    // and a little-endian file of the *same* width hold the same numbers
    // in reversed bytes, so decoding them must agree **exactly**. No
    // rounding can hide a mistake here.
    let mut exact = 0usize;
    for (name, filter) in signals() {
        for (le, be, width) in [
            ("s16le", "s16be", PcmFormat::S16Le),
            ("s24le", "s24be", PcmFormat::S24Le),
            ("s32le", "s32be", PcmFormat::S32Le),
        ] {
            let big = match width {
                PcmFormat::S16Le => PcmFormat::S16Be,
                PcmFormat::S24Le => PcmFormat::S24Be,
                _ => PcmFormat::S32Be,
            };
            let a = decode(&render(&dir, &format!("{name}.x{le}"), &filter, le), width);
            let b = decode(&render(&dir, &format!("{name}.x{be}"), &filter, be), big);
            assert_eq!(
                a, b,
                "{name}: {le} and {be} hold the same numbers; decoding them must agree exactly"
            );
            exact += 1;
        }
    }

    println!(
        "PCM vs FFMPEG: {checked} layout/signal combinations within a unit, \
         {exact} endian pairs bit-for-bit identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Bit depths and channel counts FLAC is actually used at.
const FLAC_SHAPES: [(u32, u32); 4] = [(16, 1), (16, 2), (24, 1), (24, 2)];

#[test]
#[ignore = "needs ffmpeg to render the fixtures"]
fn flac_decodes_to_exactly_the_samples_ffmpeg_produces() {
    // The whole point of a lossless decoder. Not "close", not "within a
    // unit" — every sample, or the decoder is broken.
    let dir = std::env::temp_dir().join(format!("sovmr-flac-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Every signal above duplicates one channel, which makes the side
    // channel identically zero — mid/side would then "work" while being
    // wrong. These two carry genuinely different channels, so the
    // difference channel has real content and the parity trick that
    // recovers the mid channel's low bit is actually exercised.
    let mut cases = signals();
    cases.push((
        "decorrelated",
        format!(
            "sine=frequency=440:duration={SECONDS}:sample_rate={RATE}[a];\
             sine=frequency=997:duration={SECONDS}:sample_rate={RATE}[b];\
             [a][b]amerge=inputs=2"
        ),
    ));
    cases.push((
        "stereo-noise",
        format!(
            "anoisesrc=duration={SECONDS}:sample_rate={RATE}:amplitude=0.8:seed=11[a];\
             anoisesrc=duration={SECONDS}:sample_rate={RATE}:amplitude=0.8:seed=99[b];\
             [a][b]amerge=inputs=2"
        ),
    ));

    let mut checked = 0usize;
    for (name, filter) in cases {
        for (depth, channels) in FLAC_SHAPES {
            let tag = format!("{name}-{depth}b-{channels}c");
            let flac = dir.join(format!("{tag}.flac"));
            let raw_fmt = if depth == 16 { "s16le" } else { "s32le" };

            // Encode to FLAC, and separately decode that same FLAC back
            // to raw with ffmpeg. Comparing against ffmpeg's decode of
            // OUR file is the honest test: both sides are looking at
            // identical bytes, so any difference is ours.
            let ok = Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                ])
                .arg(&filter)
                .args([
                    "-ac",
                    &channels.to_string(),
                    "-sample_fmt",
                    if depth == 16 { "s16" } else { "s32" },
                    "-c:a",
                    "flac",
                ])
                .arg(&flac)
                .status()
                .expect("ffmpeg present");
            assert!(ok.success(), "{tag}: ffmpeg could not encode");

            let raw = dir.join(format!("{tag}.raw"));
            let ok = Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
                .arg(&flac)
                .args(["-f", raw_fmt])
                .arg(&raw)
                .status()
                .expect("ffmpeg present");
            assert!(ok.success(), "{tag}: ffmpeg could not decode its own FLAC");

            let expected = decode(
                &std::fs::read(&raw).unwrap(),
                if depth == 16 {
                    PcmFormat::S16Le
                } else {
                    PcmFormat::S32Le
                },
            );

            // Our turn, from the container-level bytes.
            let bytes = std::fs::read(&flac).unwrap();
            let info = sovmr_audio::flac::StreamInfo::parse(&bytes)
                .unwrap_or_else(|e| panic!("{tag}: STREAMINFO: {e}"));
            assert_eq!(info.channels as u32, channels, "{tag}: channels");
            assert_eq!(u32::from(info.bits_per_sample), depth, "{tag}: depth");

            // Frames begin after the metadata blocks; find where.
            let start = flac_audio_start(&bytes).expect("metadata blocks end somewhere");
            let mine = sovmr_audio::flac::decode(&bytes[start..], &info)
                .unwrap_or_else(|e| panic!("{tag}: decode: {e}"));

            assert_eq!(mine.len(), expected.len(), "{tag}: sample count");
            // Both sides are left-aligned in i32, so this is exact.
            let shift = 32 - depth;
            for (i, (a, b)) in mine.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a >> shift,
                    b >> shift,
                    "{tag}: sample {i} differs — lossless has no tolerance"
                );
            }
            checked += 1;
        }
    }

    println!("FLAC BIT-EXACT: {checked} shape/signal combinations, every sample");
    std::fs::remove_dir_all(&dir).ok();
}

/// Byte offset where FLAC audio frames begin, past the metadata blocks.
fn flac_audio_start(data: &[u8]) -> Option<usize> {
    if !data.starts_with(b"fLaC") {
        return None;
    }
    let mut at = 4usize;
    loop {
        let header = data.get(at..at + 4)?;
        let last = header[0] & 0x80 != 0;
        let len = u32::from_be_bytes([0, header[1], header[2], header[3]]) as usize;
        at += 4 + len;
        if last {
            return Some(at);
        }
    }
}

/// Rate and channel combinations to check the config against.
const ASC_SHAPES: [(u32, u32); 6] = [
    (48_000, 1),
    (48_000, 2),
    (44_100, 1),
    (44_100, 2),
    (32_000, 2),
    (22_050, 2),
];

#[test]
#[ignore = "needs ffmpeg to write the fixtures"]
fn our_audio_specific_config_is_the_one_ffmpeg_writes() {
    // The two bytes a decoder reads before anything else. Our own tests
    // check them against byte pairs I believe to be canonical, which is
    // still a belief; this checks them against what a real encoder
    // actually put in a real file.
    let dir = std::env::temp_dir().join(format!("sovmr-asc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut checked = 0usize;
    for (rate, channels) in ASC_SHAPES {
        let mkv = dir.join(format!("aac-{rate}-{channels}.mkv"));
        let ok = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg(format!("sine=frequency=440:duration=1:sample_rate={rate}"))
            .args([
                "-ac",
                &channels.to_string(),
                "-c:a",
                "aac",
                "-ar",
                &rate.to_string(),
            ])
            .arg(&mkv)
            .status()
            .expect("ffmpeg present");
        assert!(ok.success(), "{rate}/{channels}: ffmpeg could not encode");

        // Matroska stores the config bare in CodecPrivate, which is
        // exactly the shape this module speaks.
        let bytes = std::fs::read(&mkv).unwrap();
        let segment = sovmr_mkv::demux::read_segment(&bytes).expect("segment");
        let track = segment
            .of_kind(sovmr_mkv::demux::TrackKind::Audio)
            .next()
            .expect("an audio track");
        let theirs = track.codec_private;
        assert!(!theirs.is_empty(), "{rate}/{channels}: no CodecPrivate");

        // Our parser must read ffmpeg's bytes correctly...
        let parsed = sovmr_audio::asc::AudioSpecificConfig::parse(theirs)
            .unwrap_or_else(|| panic!("{rate}/{channels}: could not parse {theirs:02x?}"));
        assert_eq!(
            parsed.object_type,
            sovmr_audio::asc::AAC_LC,
            "{rate}/{channels}"
        );
        assert_eq!(parsed.sample_rate, rate, "{rate}/{channels}: rate");
        assert_eq!(parsed.channels, channels, "{rate}/{channels}: channels");

        // ...and our writer must produce the same description. This is
        // the half that matters for an encoder: reading correctly proves
        // nothing about what we would emit.
        //
        // ffmpeg appends three further bytes carrying an optional
        // backward-compatible signal — eleven bits of `0x2B7`, then
        // extension type 5 (SBR) and a flag saying SBR is absent. It is
        // not required, and a config without it is what most encoders
        // write and every decoder understands. So the assertion is that
        // ours is ffmpeg's *prefix*, not that the two are equal: equal
        // would mean copying an extension we have no use for.
        let ours = parsed.to_bytes();
        assert!(
            theirs.starts_with(&ours),
            "{rate}/{channels}: ours {ours:02x?} is not a prefix of ffmpeg's {theirs:02x?}"
        );
        // And whatever ffmpeg added, re-reading it must not change the
        // answer — an extension that altered the rate or channel count
        // would mean we had misread the file.
        assert_eq!(
            sovmr_audio::asc::AudioSpecificConfig::parse(&ours),
            Some(parsed),
            "{rate}/{channels}: our own bytes must read back the same"
        );
        checked += 1;
    }

    println!(
        "AAC CONFIG vs FFMPEG: {checked} rate/channel combinations — \
         same fields, and our bytes are its exact prefix"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A stream we built ourselves, decoded by somebody else's decoder.
///
/// Every band of every frame is declared empty, so the audio is silence.
/// That is not the point. The point is everything *around* the silence:
/// the `AudioSpecificConfig`, the element framing, the width of every
/// field, the terminator, the byte alignment, and the container wiring
/// that carries them. All of it is this project's own code, and none of
/// it has ever been read by anything but this project's own code.
///
/// If ffmpeg reads this file, reports AAC-LC at the rate and channel
/// count we asked for, decodes every frame without one complaint, and
/// produces exactly the number of samples the frame count implies, then
/// the only thing left unproven in an AAC encoder is the spectrum
/// itself.
///
/// If it does not, no amount of correct Huffman coding would have
/// helped — which is precisely why this gate comes before the codebooks
/// rather than after them.
#[test]
#[ignore = "needs ffmpeg to decode what we wrote"]
fn a_stream_we_wrote_ourselves_decodes_in_somebody_elses_decoder() {
    use sovmr_audio::asc::{AudioSpecificConfig, AAC_LC};
    use sovmr_audio::frame::{silent_frame, ChannelElement, FRAME_LENGTH};

    let dir = std::env::temp_dir().join(format!("sovmr-aac-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut checked = 0;

    for (rate, channels) in [(48_000u32, 1u32), (48_000, 2), (44_100, 2), (32_000, 1)] {
        let element = ChannelElement::for_channels(channels).expect("mono and stereo are laid out");
        let asc = AudioSpecificConfig {
            object_type: AAC_LC,
            sample_rate: rate,
            channels,
        };
        let frame = silent_frame(element);

        // Long enough that a decoder desynchronising anywhere would run
        // off the end rather than merely stumble.
        const FRAMES: u64 = 100;
        let path = dir.join(format!("ours_{rate}_{channels}.mkv"));
        let spec = sovmr_mkv::mux::TrackSpec {
            number: 1,
            codec_id: "A_AAC".to_string(),
            codec_private: asc.to_bytes(),
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
        let mut mux = sovmr_mkv::mux::MkvMuxer::create(&path, &[spec]).expect("muxer");
        // Frame n covers samples [n·1024, (n+1)·1024). Deriving the
        // timestamp from that rather than accumulating a rounded
        // duration keeps a hundred frames from drifting.
        for n in 0..FRAMES {
            let at = |f: u64| (f * FRAME_LENGTH as u64 * 1000 / u64::from(rate)) as i64;
            mux.add_frame(1, at(n), at(n + 1) - at(n), true, &frame)
                .expect("frame");
        }
        mux.finalize().expect("finalize");

        // What does an independent demuxer think this file is?
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,profile,sample_rate,channels",
                "-of",
                "default=nw=1",
                path.to_str().unwrap(),
            ])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .expect("ffprobe runs");
        let field = |name: &str| {
            probe
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{name}=")))
                .unwrap_or_else(|| panic!("{rate}/{channels}: no {name} in:\n{probe}"))
                .trim()
                .to_string()
        };
        assert_eq!(field("codec_name"), "aac", "{rate}/{channels}");
        assert_eq!(field("profile"), "LC", "{rate}/{channels}: not AAC-LC");
        assert_eq!(
            field("sample_rate"),
            rate.to_string(),
            "{rate}/{channels}: our config named a different rate"
        );
        assert_eq!(
            field("channels"),
            channels.to_string(),
            "{rate}/{channels}: our config named a different channel count"
        );

        // Now decode it. `-v error` means anything printed at all is a
        // complaint, and a decoder complaining about our frames is the
        // failure this whole gate exists to catch.
        let raw = dir.join(format!("out_{rate}_{channels}.raw"));
        let out = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-i",
                path.to_str().unwrap(),
                "-f",
                "s32le",
                "-acodec",
                "pcm_s32le",
                raw.to_str().unwrap(),
            ])
            .output()
            .expect("ffmpeg runs");
        let complaints = String::from_utf8_lossy(&out.stderr);
        assert!(
            complaints.trim().is_empty(),
            "{rate}/{channels}: the decoder complained about our frames:\n{complaints}"
        );

        // Every frame must have produced its samples. A decoder that
        // silently dropped frames would otherwise pass everything above.
        let produced = std::fs::metadata(&raw).expect("decoded").len();
        let per_frame = FRAME_LENGTH as u64 * u64::from(channels) * 4;
        let frames_out = produced / per_frame;
        assert!(
            frames_out >= FRAMES,
            "{rate}/{channels}: {frames_out} frames came out of {FRAMES} — frames were dropped"
        );

        // And it must be silence, because that is what we encoded. A
        // decoder producing *something* from all-zero bands would mean
        // we had written bits that say more than we intended.
        let bytes = std::fs::read(&raw).expect("decoded");
        let loudest = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            loudest, 0,
            "{rate}/{channels}: empty bands decoded to something audible"
        );
        checked += 1;
    }

    println!(
        "OUR OWN AAC DECODES ELSEWHERE: {checked} rate/channel shapes — \
         config, framing and alignment all read correctly by ffmpeg"
    );
}

/// The negative control for the gate above.
///
/// That test asserts ffmpeg says nothing. An assertion that something
/// does not happen is worth exactly as much as the proof that it *can*
/// happen — otherwise a decoder that had quietly stopped reporting
/// anything would pass it forever.
///
/// So this builds a frame that is wrong in a specific, minimal way: it
/// declares one coded scalefactor band and then supplies none of the
/// data that band requires. Everything else is byte-for-byte the frame
/// the gate above writes. A decoder must notice.
#[test]
#[ignore = "needs ffmpeg to decode what we wrote"]
fn a_frame_we_broke_on_purpose_is_caught_by_that_same_decoder() {
    use sovmr_audio::asc::{AudioSpecificConfig, AAC_LC};
    use sovmr_audio::bits::BitWriter;

    let dir = std::env::temp_dir().join(format!("sovmr-aac-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut w = BitWriter::new();
    w.write(0x0, 3); // ID_SCE
    w.write(0, 4); // element_instance_tag
    w.write(0, 8); // global_gain
    w.write(0, 1); // ics_reserved_bit
    w.write(0, 2); // ONLY_LONG_SEQUENCE
    w.write(0, 1); // window_shape
    w.write(1, 6); // max_sfb: one band — and this is the lie
    w.write(0, 1); // predictor_data_present
    w.write(0, 1); // pulse_data_present
    w.write(0, 1); // tns_data_present
    w.write(0, 1); // gain_control_data_present
    w.write(0x7, 3); // ID_END, with the promised band never delivered
    let broken = w.finish();

    let rate = 48_000u32;
    let path = dir.join("broken.mkv");
    let spec = sovmr_mkv::mux::TrackSpec {
        number: 1,
        codec_id: "A_AAC".to_string(),
        codec_private: AudioSpecificConfig {
            object_type: AAC_LC,
            sample_rate: rate,
            channels: 1,
        }
        .to_bytes(),
        video: false,
        subtitle: false,
        width: 0,
        height: 0,
        sample_rate: rate,
        channels: 1,
        language: "und".to_string(),
        name: None,
        enabled: true,
        colour: None,
    };
    let mut mux = sovmr_mkv::mux::MkvMuxer::create(&path, &[spec]).expect("muxer");
    for n in 0..20u64 {
        let at = |f: u64| (f * 1024 * 1000 / u64::from(rate)) as i64;
        mux.add_frame(1, at(n), at(n + 1) - at(n), true, &broken)
            .expect("frame");
    }
    mux.finalize().expect("finalize");

    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s32le",
            "-acodec",
            "pcm_s32le",
            dir.join("broken.raw").to_str().unwrap(),
        ])
        .output()
        .expect("ffmpeg runs");
    let complaints = String::from_utf8_lossy(&out.stderr);
    assert!(
        !complaints.trim().is_empty(),
        "the decoder accepted a frame that promises a band and delivers none — \
         which means the silence of the positive gate proves nothing"
    );
    println!(
        "NEGATIVE CONTROL HOLDS: a deliberately malformed frame is rejected — \
         {}",
        complaints.lines().next().unwrap_or("").trim()
    );
}
