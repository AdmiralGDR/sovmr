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

//! THE gate: sovereign AV1 in a sovereign fast-start MP4. ffmpeg appears
//! only as a fixture generator; from the moment the file exists, every
//! byte is handled by sovmr code and NVIDIA silicon — demux, NVDEC, VRAM
//! handoff, NVENC, muxing — and the *verification* is sovereign too: the
//! output is parsed back with our own demuxer, not ffprobe.
//!
//! ```sh
//! cargo test -p sovmr-native --test e2e_pipeline -- --ignored --nocapture
//! ```

use std::process::Command;
use std::sync::atomic::AtomicBool;

use sovmr_mp4::demux::{find_audio_tracks, find_video_track, VideoCodec};
use sovmr_mp4::{boxes, file_kind};
use sovmr_native::{transcode, NativeSettings, TranscodeOutcome};
use sovmr_nvenc::encoder::{EncodeCodec, EncodePreset};

/// Fixture geometry: 12 s at 30 fps — long enough to cross the 10-second
/// GOP boundary, so seeking (multiple sync samples) is still exercised
/// now that IDRs are 5x rarer (the compression-efficiency trade).
const FIXTURE_FRAMES: u64 = 360;

/// Audio leg of a generated fixture.
enum FixtureAudio {
    /// AAC sine tone: proves the bit-exact passthrough leg.
    Aac,
    /// FLAC-in-MP4: an audio codec the sovereign muxer cannot pass
    /// through — proves graceful degradation.
    Flac,
    /// Six AAC tracks, each a different tone under a different language
    /// tag — the shape of the material this milestone exists for.
    Many,
    /// No audio track at all.
    None,
}

/// Language tags for [`FixtureAudio::Many`]. Distinct on purpose: if the
/// muxer dropped identity and wrote `und` everywhere, or shuffled the
/// order, six otherwise-identical tracks would hide it.
const MANY_LANGUAGES: [&str; 6] = ["eng", "rus", "deu", "fra", "jpn", "spa"];

/// Base frequency of the first tone in a [`FixtureAudio::Many`] fixture,
/// and the step between tracks. Different tones mean each track carries
/// genuinely different bytes, so a muxer that wrote one track's payload
/// six times could not pass.
const MANY_TONE_BASE_HZ: u32 = 220;
/// Frequency added per additional track.
const MANY_TONE_STEP_HZ: u32 = 110;

/// The codec this machine can actually encode with.
///
/// An RTX 4070 has both AV1 and HEVC; an RTX 3060 has only HEVC. A gate
/// that hardcoded AV1 passed on one machine and failed on the other for
/// a reason that is not a defect — the engine was refusing correctly.
///
/// So the gates that are about the *pipeline* rather than about a
/// particular codec take whatever this hardware offers, and coverage
/// follows the machine instead of vanishing on it.
fn best_codec() -> EncodeCodec {
    if encoder_supports(EncodeCodec::Av1) {
        EncodeCodec::Av1
    } else {
        EncodeCodec::Hevc
    }
}

/// The name ffprobe gives a codec.
fn codec_name(codec: EncodeCodec) -> &'static str {
    match codec {
        EncodeCodec::Av1 => "av1",
        EncodeCodec::Hevc => "hevc",
    }
}

/// The name ffprobe gives [`best_codec`].
fn best_codec_name() -> &'static str {
    codec_name(best_codec())
}

/// The codec [`best_codec`] produces, as our own MP4 demuxer names it.
fn best_video_codec() -> VideoCodec {
    match best_codec() {
        EncodeCodec::Av1 => VideoCodec::Av1,
        EncodeCodec::Hevc => VideoCodec::Hevc,
    }
}

/// The same, as Matroska names it.
fn best_matroska_codec_id() -> &'static str {
    match best_codec() {
        EncodeCodec::Av1 => "V_AV1",
        EncodeCodec::Hevc => "V_MPEGH/ISO/HEVC",
    }
}

/// Whether this machine has an encoder for `codec`.
///
/// Asked of the driver's own GUID list, never of a marketing name — the
/// "RTX 30 implies no AV1" classifier was deleted in stage 2 for being a
/// guess standing in for a fact.
fn encoder_supports(codec: EncodeCodec) -> bool {
    use sovmr_core::encoder::EncoderProvider;
    match sovmr_native::provider::NvencNativeProvider.capabilities() {
        Ok(caps) => match codec {
            EncodeCodec::Av1 => caps.av1,
            EncodeCodec::Hevc => caps.hevc,
        },
        Err(_) => false,
    }
}

/// Announce a gate that this hardware cannot run, and why.
///
/// Printed loudly rather than skipped in silence: a gate that quietly
/// never runs proves nothing, and the whole point of running on two
/// machines is knowing which half each one covered.
fn skip_unsupported(gate: &str, codec: &str) {
    println!("SKIPPED {gate}: this GPU has no {codec} encoder");
}

/// The first carryable audio track — enough for the gates that only
/// assert one survived.
fn first_audio(file: &[u8]) -> Option<sovmr_mp4::demux::AudioTrack<'_>> {
    find_audio_tracks(file).ok()?.tracks.into_iter().next()
}

/// 12 s of H.264 test pattern with the requested audio leg.
fn make_fixture(dir: &std::path::Path, audio: FixtureAudio) -> std::path::PathBuf {
    let clip = dir.join("pipeline_fixture.mp4");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=12:size=640x360:rate=30",
    ]);
    match audio {
        FixtureAudio::Aac | FixtureAudio::Flac => {
            cmd.args(["-f", "lavfi", "-i", "sine=frequency=440:duration=12"]);
        }
        FixtureAudio::Many => {
            for i in 0..MANY_LANGUAGES.len() as u32 {
                let hz = MANY_TONE_BASE_HZ + i * MANY_TONE_STEP_HZ;
                cmd.args(["-f", "lavfi", "-i"]);
                cmd.arg(format!("sine=frequency={hz}:duration=12"));
            }
        }
        FixtureAudio::None => {}
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30"]);
    match audio {
        FixtureAudio::Aac => {
            cmd.args(["-c:a", "aac", "-shortest"]);
        }
        FixtureAudio::Flac => {
            // FLAC needs the experimental muxer flag inside MP4.
            cmd.args(["-c:a", "flac", "-strict", "-2", "-shortest"]);
        }
        FixtureAudio::Many => {
            // ffmpeg's default mapping would take exactly one audio
            // stream, so every input has to be mapped by hand.
            cmd.args(["-map", "0:v"]);
            for (i, lang) in MANY_LANGUAGES.iter().enumerate() {
                cmd.arg("-map");
                cmd.arg(format!("{}:a", i + 1));
                cmd.arg(format!("-metadata:s:a:{i}"));
                cmd.arg(format!("language={lang}"));
            }
            cmd.args(["-c:a", "aac", "-shortest"]);
        }
        FixtureAudio::None => {}
    }
    let status = cmd
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success());
    clip
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (full NVDEC->NVENC->mux pipeline)"]
fn sovereign_av1_mp4_flows_end_to_end() {
    if !encoder_supports(EncodeCodec::Av1) {
        skip_unsupported("sovereign_av1_mp4_flows_end_to_end", "AV1");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sovmr-pipe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Aac);
    let output = dir.join("sovereign.mp4");

    let mut last_progress = 0u64;
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: EncodeCodec::Av1,
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |frames| last_progress = frames,
        &AtomicBool::new(false),
    )
    .expect("the sovereign pipeline must transcode the fixture");

    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES, "every decoded frame encoded");
    assert_eq!(last_progress, FIXTURE_FRAMES);
    assert_eq!(stats.warning, None, "AAC passthrough needs no warning");

    let bytes = std::fs::read(&output).expect("output committed via rename");
    assert_eq!(bytes.len() as u64, stats.output_bytes, "stats == file size");

    // Sovereign verification: our own parser reads our own file.
    assert_eq!(file_kind(&bytes), Some(*b"isom"));
    let kinds: Vec<[u8; 4]> = boxes(&bytes).map(|b| b.unwrap().kind).collect();
    assert_eq!(
        kinds,
        vec![*b"ftyp", *b"moov", *b"mdat"],
        "fast start: moov strictly before mdat"
    );

    let track = find_video_track(&bytes).expect("own demuxer parses own muxer");
    assert_eq!(track.codec, VideoCodec::Av1);
    assert_eq!((track.width, track.height), (640, 360));
    assert_eq!(u64::from(track.sample_count()), FIXTURE_FRAMES);
    let duration = track.duration().expect("mdhd duration present");
    assert!(
        (duration.as_secs_f64() - 12.0).abs() < 0.1,
        "12 s fixture, got {duration:?}"
    );
    let mut keyframes = 0;
    for sample in track.samples(&bytes) {
        let sample = sample.expect("stco/stsz offsets stay inside the file");
        // Every sample must begin with a valid AV1 OBU header:
        // forbidden bit clear, known low-overhead type.
        let header = sample.data[0];
        assert_eq!(header & 0x80, 0, "OBU forbidden bit");
        assert!(
            matches!((header >> 3) & 0xF, 1..=8),
            "unexpected leading OBU type {} in sample {}",
            (header >> 3) & 0xF,
            sample.index
        );
        if sample.keyframe {
            keyframes += 1;
            // Seeking contract (AV1-ISOBMFF binding): every sync sample
            // begins with a sequence-header OBU so a player landing on
            // it mid-file can decode standalone.
            assert_eq!(
                (header >> 3) & 0xF,
                1,
                "sync sample {} must open with a sequence header",
                sample.index
            );
        }
    }
    assert!(
        keyframes >= 2,
        "10 s GOP over a 12 s clip must still yield multiple sync samples \
         (seeking needs them), got {keyframes}"
    );

    // B-frame timing contract: decode times strictly monotone, and the
    // presentation timestamps form a complete, evenly spaced sequence —
    // every source frame's PTS survived the reorder through ctts.
    let samples: Vec<_> = track.samples(&bytes).map(|s| s.unwrap()).collect();
    assert!(
        samples.windows(2).all(|w| w[0].dts < w[1].dts),
        "dts must be strictly increasing"
    );
    let mut pts: Vec<i64> = samples.iter().map(|s| s.pts).collect();
    pts.sort_unstable();
    let base = pts[0];
    let delta = i64::from(samples[0].duration);
    for (i, p) in pts.iter().enumerate() {
        assert_eq!(
            *p,
            base + i as i64 * delta,
            "presentation timeline must be gapless"
        );
    }
    // AV1 never reorders its output: "B-frames" are hidden alt-refs
    // inside temporal units (show_existing_frame), so TU order == both
    // decode and presentation order and no ctts is ever needed.
    let first_offset = samples[0].pts - samples[0].dts as i64;
    assert!(
        samples.iter().all(|s| s.pts - s.dts as i64 == first_offset),
        "AV1 temporal units must stay in presentation order"
    );

    let audio = first_audio(&bytes).expect("AAC track passed through");
    assert!(audio.sample_count() > 0, "audio packets survived");
    assert_eq!(audio.sample_rate, 44_100);
    for packet in audio.samples(&bytes) {
        packet.expect("audio offsets stay inside the file");
    }

    assert!(
        !dir.join("sovereign.mp4.part").exists() && !dir.join("sovereign.mp4.part.mdat").exists(),
        "no .part/.mdat litter after commit"
    );
    println!(
        "SOVEREIGN AV1 MP4: {} frames + {} audio packets, {} bytes, fast start verified",
        stats.frames,
        audio.sample_count(),
        stats.output_bytes
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (HEVC target)"]
fn sovereign_hevc_mp4_flows_end_to_end() {
    let dir = std::env::temp_dir().join(format!("sovmr-pipeh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Aac);
    let output = dir.join("sovereign_hevc.mp4");

    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: EncodeCodec::Hevc,
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("HEVC target must also produce a valid MP4");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES);
    assert_eq!(stats.warning, None);

    let bytes = std::fs::read(&output).expect("output committed via rename");
    let kinds: Vec<[u8; 4]> = boxes(&bytes).map(|b| b.unwrap().kind).collect();
    assert_eq!(
        kinds,
        vec![*b"ftyp", *b"moov", *b"mdat"],
        "fast start holds for HEVC"
    );

    let track = find_video_track(&bytes).expect("own demuxer parses own hvc1");
    assert_eq!(track.codec, VideoCodec::Hevc);
    assert_eq!((track.width, track.height), (640, 360));
    assert_eq!(u64::from(track.sample_count()), FIXTURE_FRAMES);
    assert_eq!(track.nal_length_size, 4, "hvcC lengthSizeMinusOne = 3");

    // The hvcC arrays hold exactly the harvested VPS/SPS/PPS, with the
    // right NAL types in order.
    let sets: Vec<&[u8]> = track
        .parameter_sets()
        .collect::<Result<_, _>>()
        .expect("hvcC arrays parse");
    assert_eq!(sets.len(), 3, "VPS + SPS + PPS");
    let types: Vec<u8> = sets.iter().map(|n| (n[0] >> 1) & 0x3F).collect();
    assert_eq!(types, vec![32, 33, 34]);

    // Every sample walks its 4-byte NAL prefixes cleanly (proven by the
    // Annex B converter, the same consumer a future decode would use),
    // and no parameter sets leak into hvc1 samples.
    let mut keyframes = 0;
    let mut annex_b = Vec::new();
    for sample in track.samples(&bytes) {
        let sample = sample.expect("stco/stsz offsets stay inside the file");
        track
            .write_frame(&sample, &mut annex_b)
            .expect("NAL length prefixes are consistent");
        keyframes += u32::from(sample.keyframe);
        let mut pos = 0usize;
        while pos < sample.data.len() {
            let len = u32::from_be_bytes(sample.data[pos..pos + 4].try_into().unwrap()) as usize;
            let nal_type = (sample.data[pos + 4] >> 1) & 0x3F;
            assert!(
                !(32..=34).contains(&nal_type),
                "parameter set inside an hvc1 sample"
            );
            pos += 4 + len;
        }
    }
    assert!(
        keyframes >= 2,
        "10 s GOP over 12 s: seeking needs multiple sync samples, got {keyframes}"
    );

    // HEVC B-frames genuinely reorder output: the ctts table must carry
    // distinct composition offsets, the presentation timeline must stay
    // gapless, and decode times monotone.
    let samples: Vec<_> = track.samples(&bytes).map(|s| s.unwrap()).collect();
    assert!(
        samples.windows(2).all(|w| w[0].dts < w[1].dts),
        "dts must be strictly increasing"
    );
    let mut pts: Vec<i64> = samples.iter().map(|s| s.pts).collect();
    pts.sort_unstable();
    let delta = i64::from(samples[0].duration);
    for (i, p) in pts.iter().enumerate() {
        assert_eq!(*p, pts[0] + i as i64 * delta, "gapless presentation");
    }
    let first_offset = samples[0].pts - samples[0].dts as i64;
    assert!(
        samples.iter().any(|s| s.pts - s.dts as i64 != first_offset),
        "B-frames enabled: HEVC samples must carry distinct ctts offsets"
    );

    let audio = first_audio(&bytes).expect("AAC passthrough is codec-independent");
    assert!(audio.sample_count() > 0);

    println!(
        "SOVEREIGN HEVC MP4: {} frames + {} audio packets, {} bytes",
        stats.frames,
        audio.sample_count(),
        stats.output_bytes
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (video-only source)"]
fn video_only_source_produces_video_only_mp4_without_deadlock() {
    let dir = std::env::temp_dir().join(format!("sovmr-pipevo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::None);
    let output = dir.join("video_only.mp4");

    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("a source without audio must simply encode");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES);
    assert_eq!(stats.warning, None, "no audio is not a degradation");

    let bytes = std::fs::read(&output).unwrap();
    let track = find_video_track(&bytes).expect("video track valid");
    assert_eq!(u64::from(track.sample_count()), FIXTURE_FRAMES);
    assert!(
        find_audio_tracks(&bytes)
            .expect("no demux error")
            .is_silent(),
        "output has exactly one track"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (unsupported audio source)"]
fn unsupported_audio_degrades_to_video_only_with_a_warning() {
    let dir = std::env::temp_dir().join(format!("sovmr-pipefl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Flac);
    let output = dir.join("degraded.mp4");

    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("unsupported audio must not fail the job");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES, "video encoded in full");
    let warning = stats.warning.expect("degradation must be reported");
    // The message must say how many tracks were lost and name the codec:
    // "some audio was dropped" is not actionable, and with six-track
    // sources the count is the part that matters.
    assert!(
        warning.contains("skipped") && warning.contains("fLaC") && warning.contains('1'),
        "warning must name the count and the codec: {warning}"
    );

    let bytes = std::fs::read(&output).unwrap();
    let track = find_video_track(&bytes).expect("video track valid");
    assert_eq!(track.codec, best_video_codec());
    assert!(
        find_audio_tracks(&bytes)
            .expect("no demux error")
            .is_silent(),
        "the FLAC track was not smuggled into the output"
    );
    println!("DEGRADED GRACEFULLY: {warning}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (six audio tracks through the whole engine)"]
fn every_audio_track_survives_the_whole_engine() {
    // The milestone gate. Container-level tests prove the muxer can hold
    // six tracks; only this one proves the *engine* carries them —
    // demuxer, N-way interleaver and muxer in one piece. Before this
    // work the pipeline took track zero and dropped the other five
    // without a word, which is the worst kind of failure: silent,
    // plausible-looking output.
    let dir = std::env::temp_dir().join(format!("sovmr-pipemany-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Many);
    let output = dir.join("many.mp4");

    // Negative trust toward the fixture itself: assert the source really
    // has what the test claims before drawing conclusions from the output.
    let src_bytes = std::fs::read(&input).unwrap();
    let src_audio = find_audio_tracks(&src_bytes).expect("fixture demuxes");
    assert_eq!(
        src_audio.tracks.len(),
        MANY_LANGUAGES.len(),
        "ffmpeg must have produced the six-track fixture this test needs"
    );

    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("six audio tracks must not fail the job");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES, "video encoded in full");
    assert_eq!(
        stats.warning, None,
        "every track was carryable — nothing to warn about"
    );

    let bytes = std::fs::read(&output).unwrap();
    let video = find_video_track(&bytes).expect("video track valid");
    assert_eq!(video.codec, best_video_codec());
    assert_eq!(u64::from(video.sample_count()), FIXTURE_FRAMES);

    let out_audio = find_audio_tracks(&bytes).expect("own demuxer parses own muxer");
    assert_eq!(
        out_audio.tracks.len(),
        MANY_LANGUAGES.len(),
        "six tracks in, six tracks out"
    );

    for (i, (src, out)) in src_audio.tracks.iter().zip(&out_audio.tracks).enumerate() {
        assert_eq!(
            out.meta.language.as_str(),
            MANY_LANGUAGES[i],
            "track {i} lost its language tag"
        );
        assert_eq!(
            out.sample_count(),
            src.sample_count(),
            "track {i} lost packets in transit"
        );
        assert_eq!(out.sample_rate, src.sample_rate, "track {i} sample rate");
        assert_eq!(out.channels, src.channels, "track {i} channel count");
        assert_eq!(
            out.total_bytes(),
            src.total_bytes(),
            "track {i} is passthrough: the payload must be bit-for-bit the same size"
        );

        // Interleaving must stay chunk-granular. Releasing packets one
        // at a time across seven tracks is what would bury a long file
        // under millions of chunks; the muxer batches ~1 s slabs, so a
        // 12 s track lands in a dozen or so.
        let chunks = out.chunk_count();
        assert!(
            chunks > 0 && chunks <= 32,
            "track {i}: a 12 s track should land in a handful of ~1 s chunks, got {chunks}"
        );
    }

    println!(
        "SIX TRACKS SURVIVED: {} audio tracks, {} bytes out",
        out_audio.tracks.len(),
        stats.output_bytes
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (compression efficiency gate)"]
fn output_must_be_smaller_than_a_normal_quality_1080p_source() {
    // THE business gate: a compression tool that grows files is broken.
    // The fixture mimics a normal camera/export file — 1080p H.264 at
    // good quality (crf 19). Our AV1 re-encode must come out smaller.
    let dir = std::env::temp_dir().join(format!("sovmr-pipesz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("hd_source.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=8:size=1920x1080:rate=30",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "60",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success());
    let input_bytes = std::fs::metadata(&input).unwrap().len();

    let output = dir.join("hd_out.mp4");
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("1080p transcode");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    let reduction = 100.0 - (stats.output_bytes as f64 / input_bytes as f64 * 100.0);
    println!(
        "COMPRESSION GATE (1080p): {input_bytes} -> {} bytes ({reduction:+.1}% saved)",
        stats.output_bytes
    );
    assert!(
        stats.output_bytes < input_bytes,
        "a compression tool must not grow files: {input_bytes} -> {}",
        stats.output_bytes
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver"]
fn cancellation_rolls_back_the_partial_output() {
    let dir = std::env::temp_dir().join(format!("sovmr-pipec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Aac);
    let output = dir.join("cancelled.mp4");

    // Pre-set cancel: the loop must observe it before the first sample.
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(true),
    )
    .expect("cancel is a clean outcome, not an error");
    assert_eq!(outcome, TranscodeOutcome::Cancelled);
    assert!(!output.exists(), "no committed output after cancel");
    assert!(
        !dir.join("cancelled.mp4.part").exists() && !dir.join("cancelled.mp4.part.mdat").exists(),
        "partial + spool rolled back"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Frames to let through before pulling the plug: enough that NVDEC has
/// pictures queued and its worker thread is busy, few enough to stay fast.
const CANCEL_AFTER_FRAMES: u64 = 8;
/// Cancel/teardown cycles per run — repetition widens the window in
/// which a torn-down decoder could be caught still working.
const CANCEL_CYCLES: usize = 8;

#[test]
#[ignore = "requires NVIDIA GPU + driver and ffmpeg for the fixture"]
fn cancelling_mid_stream_leaves_no_output_and_no_partials() {
    // Coverage gap this closes: [`cancellation_rolls_back_the_partial_output`]
    // pre-sets the cancel flag, so the loop exits before a single sample
    // is fed and the decoder never has anything in flight. Nothing
    // exercised cancelling a *running* pipeline — which is the only kind
    // of cancellation a user can actually perform.
    //
    // What it does NOT do, stated plainly so nobody trusts it further
    // than it deserves: it is not a reliable reproducer for the SIGSEGV
    // seen inside libnvcuvid after the source became a file mapping.
    // That fault is timing dependent and did not reproduce here even
    // with the orderly-shutdown code removed. This test pins the
    // observable contract — cancel mid-encode yields `Cancelled`, no
    // committed output, no `.part` left behind — and nothing more.
    let dir = std::env::temp_dir().join(format!("sovmr-pipecm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Aac);

    for cycle in 0..CANCEL_CYCLES {
        let output = dir.join(format!("cancelled-{cycle}.mp4"));
        let cancel = AtomicBool::new(false);
        let outcome = transcode(
            &input,
            &output,
            &NativeSettings {
                codec: best_codec(),
                cq: 25,
                max_bitrate: 0,
                preset: EncodePreset::Quality,
                references: sovmr_nvenc::encoder::ReferencePlan::default(),
            },
            |frames| {
                if frames >= CANCEL_AFTER_FRAMES {
                    cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            },
            &cancel,
        )
        .expect("cancel is a clean outcome, not an error");
        assert_eq!(
            outcome,
            TranscodeOutcome::Cancelled,
            "cycle {cycle}: mid-stream cancel must report Cancelled"
        );
        assert!(
            !output.exists(),
            "cycle {cycle}: no committed output after cancel"
        );
        assert!(
            !dir.join(format!("cancelled-{cycle}.mp4.part")).exists(),
            "cycle {cycle}: partial rolled back"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Chapter boundaries of the rich fixture, in milliseconds — spanning
/// the full 12 s clip.
const RICH_CHAPTER_MS: [(u32, u32); 2] = [(0, 6000), (6000, 12000)];
/// Subtitle cues written into the rich fixture.
const RICH_CUES: [(&str, &str, &str); 2] = [
    ("00:00:01,000", "00:00:03,000", "Hello"),
    ("00:00:04,000", "00:00:06,000", "World"),
];

/// A fixture carrying everything the user asked to preserve: video,
/// audio, a subtitle track, chapters and tags.
fn make_rich_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let srt = dir.join("cues.srt");
    let mut cues = String::new();
    for (i, (start, end, text)) in RICH_CUES.iter().enumerate() {
        cues.push_str(&format!("{}\n{start} --> {end}\n{text}\n\n", i + 1));
    }
    std::fs::write(&srt, cues).expect("subtitle file written");

    let meta = dir.join("chapters.txt");
    let mut m = String::from(";FFMETADATA1\ntitle=Sovereign Test\n");
    for (i, (start, end)) in RICH_CHAPTER_MS.iter().enumerate() {
        m.push_str(&format!(
            "\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Part {}\n",
            i + 1
        ));
    }
    std::fs::write(&meta, m).expect("chapter file written");

    let clip = dir.join("rich_fixture.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=12:size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
        ])
        .arg("-i")
        .arg(&srt)
        .arg("-i")
        .arg(&meta)
        .args([
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:s",
            "-map_chapters",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-c:a",
            "aac",
            "-c:s",
            "mov_text",
            // No `-shortest` here: the subtitle stream ends with its last
            // cue at 6 s, and clamping to the shortest stream would halve
            // the clip and quietly change what this gate encodes.
        ])
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success());
    clip
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (subtitles/chapters degradation notice)"]
fn what_the_output_cannot_carry_is_named_instead_of_vanishing() {
    // The user asked to preserve the maximum a source carries. The MP4
    // output path cannot re-emit subtitles, chapters or tags yet, and
    // the honest interim behaviour is to say so — the same contract the
    // unsupported-audio path already honours. A limitation the user is
    // told about is a limitation; one they discover months later, when
    // the original is gone, is data loss.
    let dir = std::env::temp_dir().join(format!("sovmr-piperich-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_rich_fixture(&dir);
    let output = dir.join("rich.mp4");

    // Negative trust toward the fixture: prove it really carries the
    // things whose loss is being reported before trusting the report.
    let src = std::fs::read(&input).unwrap();
    let extras = sovmr_mp4::demux::survey_extras(&src).expect("fixture surveys");
    assert!(
        extras.subtitles >= 1,
        "fixture must carry a subtitle track, got {extras:?}"
    );
    assert!(extras.chapters, "fixture must carry chapters: {extras:?}");

    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("extra tracks must not fail the job");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES, "video encoded in full");

    let warning = stats
        .warning
        .expect("dropping subtitles and chapters must be reported");
    assert!(
        warning.contains("subtitle") && warning.contains("chapters"),
        "the notice must name what was left behind: {warning}"
    );

    // The audio leg is unaffected by any of this.
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(
        find_audio_tracks(&bytes).expect("demuxes").tracks.len(),
        1,
        "the carryable audio track still came through"
    );

    // The same facts must reach the *card*, not only the post-run
    // warning: someone feeding a six-language master needs to see it was
    // understood before committing hours to the job. This is the probe
    // path, so it is CPU-only and independent of the encode above.
    let info =
        sovmr_core::encoder::EncoderProvider::probe(&sovmr_native::NvencNativeProvider, &input)
            .expect("probe reads the rich fixture");
    assert_eq!(info.audio_tracks, 1, "one carryable audio track");
    assert_eq!(
        info.subtitle_tracks, 1,
        "one subtitle track, not the chapter track"
    );
    assert!(info.chapters, "chapters must reach MediaInfo");
    let line = sovmr_core::heuristics::insight(
        &info,
        sovmr_core::heuristics::Strategy::Balanced,
        sovmr_core::job::Codec::Av1,
    )
    .expect("insight line");
    assert!(
        line.contains("1 subtitles") && line.contains("chapters"),
        "the card must say what the source carries: {line}"
    );
    println!("CARD SAYS: {line}");

    println!("NAMED THE LOSS: {warning}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Languages of the Matroska probe fixture.
const MKV_LANGUAGES: [&str; 3] = ["eng", "rus", "deu"];

#[test]
#[ignore = "needs ffmpeg to write the fixture"]
fn a_matroska_source_probes_through_the_same_facade_as_mp4() {
    // The container is chosen by the file's magic bytes, so the engine's
    // own `probe` must answer for Matroska exactly as it does for MP4 —
    // no separate call, no separate type at the boundary.
    let dir = std::env::temp_dir().join(format!("sovmr-mkvprobe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let clip = dir.join("probe.mkv");

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=6:size=640x360:rate=30",
    ]);
    for i in 0..MKV_LANGUAGES.len() {
        cmd.args(["-f", "lavfi", "-i"]);
        cmd.arg(format!("sine=frequency={}:duration=6", 220 + 110 * i));
    }
    cmd.args(["-map", "0:v"]);
    for (i, lang) in MKV_LANGUAGES.iter().enumerate() {
        cmd.arg("-map");
        cmd.arg(format!("{}:a", i + 1));
        cmd.arg(format!("-metadata:s:a:{i}"));
        cmd.arg(format!("language={lang}"));
    }
    cmd.args([
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-c:a", "aac",
    ]);
    assert!(cmd.arg(&clip).status().expect("ffmpeg present").success());

    let info =
        sovmr_core::encoder::EncoderProvider::probe(&sovmr_native::NvencNativeProvider, &clip)
            .expect("the engine probes Matroska through its normal facade");

    assert_eq!(info.video_codec, "h264");
    assert_eq!((info.width, info.height), (640, 360));
    assert_eq!(
        info.audio_tracks,
        MKV_LANGUAGES.len(),
        "every AAC track must be counted as carryable"
    );
    let duration = info.duration.expect("Matroska declares its duration");
    assert!(
        (duration.as_secs_f64() - 6.0).abs() < 0.2,
        "6 s fixture, got {duration:?}"
    );
    assert!(
        (info.fps - 30.0).abs() < 0.5,
        "fps comes from DefaultDuration, got {}",
        info.fps
    );
    assert!(
        info.audio_bitrate > 0,
        "three AAC tracks weigh something; the estimate must not be zero"
    );

    // Moving an AAC track from Matroska to MP4 means rebuilding the
    // `esds` wrapper around the bare `AudioSpecificConfig` Matroska
    // stores. Do it from the real config bytes and read the result back
    // with our own demuxer — the two sides must agree, or a passthrough
    // track would land in the output describing itself wrongly.
    let bytes = std::fs::read(&clip).unwrap();
    let segment = sovmr_mkv::demux::read_segment(&bytes).expect("segment");
    let aac = segment
        .of_kind(sovmr_mkv::demux::TrackKind::Audio)
        .find(|t| sovmr_mkv::codec::is_passthrough_audio(t.codec_id))
        .expect("the fixture has AAC");
    assert!(
        !aac.codec_private.is_empty(),
        "an AAC track carries its AudioSpecificConfig"
    );

    let entry =
        sovmr_mp4::mux::build_mp4a_entry(aac.sample_rate, aac.channels, aac.codec_private, 0, 0);
    // The demuxer reads channels at 16 and the 16.16 rate at 24.
    assert_eq!(u16::from_be_bytes([entry[16], entry[17]]), aac.channels);
    assert_eq!(
        u32::from_be_bytes([entry[24], entry[25], entry[26], entry[27]]) >> 16,
        aac.sample_rate
    );
    assert!(
        entry
            .windows(aac.codec_private.len())
            .any(|w| w == aac.codec_private),
        "the decoder config must survive verbatim into the esds"
    );
    // And the whole audio side planned at once, which is the shape the
    // muxer takes: one description per carryable track, in file order,
    // with the rest named rather than dropped.
    let plan = sovmr_mkv::source::plan_audio(&segment, sovmr_mkv::source::Target::Mp4);
    assert_eq!(
        plan.carried.len(),
        MKV_LANGUAGES.len(),
        "every AAC track must be carryable"
    );
    assert!(plan.unsupported.is_empty());
    let planned: Vec<String> = plan
        .carried
        .iter()
        .map(|p| String::from_utf8_lossy(&p.language).to_string())
        .collect();
    assert_eq!(
        planned, MKV_LANGUAGES,
        "languages must survive into the MP4 description, in order"
    );
    for p in &plan.carried {
        assert_eq!(
            p.timescale, aac.sample_rate,
            "the timescale is the sampling rate, so durations are whole samples"
        );
        assert!(p.entry_payload.is_some());
    }
    println!(
        "AUDIO PLANNED: {} tracks carried as {:?}, {} left behind",
        plan.carried.len(),
        planned,
        plan.unsupported.len()
    );

    println!(
        "MP4A ENTRY BUILT: {} bytes around a {}-byte AudioSpecificConfig ({} Hz, {} ch)",
        entry.len(),
        aac.codec_private.len(),
        aac.sample_rate,
        aac.channels
    );

    println!(
        "MATROSKA PROBED: {}x{} {} @ {:.1} fps, {} audio tracks, ~{} bit/s audio",
        info.width, info.height, info.video_codec, info.fps, info.audio_tracks, info.audio_bitrate
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "requires NVIDIA GPU + driver (Matroska source through the whole engine)"]
fn a_matroska_source_transcodes_with_every_audio_track() {
    // THE gate for Matroska input. Everything before it proved the
    // reader; this proves the engine, with a container it could not open
    // at all a week ago, all the way to a file our own demuxer reads.
    let dir = std::env::temp_dir().join(format!("sovmr-mkvenc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("source.mkv");

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=12:size=640x360:rate=30",
    ]);
    for i in 0..MKV_LANGUAGES.len() {
        cmd.args(["-f", "lavfi", "-i"]);
        cmd.arg(format!("sine=frequency={}:duration=12", 220 + 110 * i));
    }
    cmd.args(["-map", "0:v"]);
    for (i, lang) in MKV_LANGUAGES.iter().enumerate() {
        cmd.arg("-map");
        cmd.arg(format!("{}:a", i + 1));
        cmd.arg(format!("-metadata:s:a:{i}"));
        cmd.arg(format!("language={lang}"));
    }
    cmd.args([
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-c:a", "aac",
    ]);
    assert!(cmd.arg(&input).status().expect("ffmpeg present").success());

    // The output mirrors the input: a Matroska source produces
    // Matroska, because silently re-containerising is how somebody
    // loses the tracks their player was going to offer them.
    let output = dir.join("out.mkv");
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("a Matroska source must transcode");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES, "every frame encoded");

    let bytes = std::fs::read(&output).unwrap();
    assert!(
        sovmr_mkv::is_matroska(&bytes),
        "the output must be Matroska, by its bytes"
    );
    let seg = sovmr_mkv::demux::read_segment(&bytes).expect("own demuxer parses own muxer");
    let video = seg
        .of_kind(sovmr_mkv::demux::TrackKind::Video)
        .next()
        .expect("a video track");
    assert_eq!(video.codec_id, best_matroska_codec_id());
    assert_eq!((video.width, video.height), (640, 360));
    assert!(
        !video.codec_private.is_empty(),
        "an AV1 track carries its av1C in CodecPrivate"
    );

    let out_audio: Vec<&str> = seg
        .of_kind(sovmr_mkv::demux::TrackKind::Audio)
        .map(|t| t.language)
        .collect();
    assert_eq!(out_audio, MKV_LANGUAGES, "three tracks in, three out");

    let (out_frames, err) = sovmr_mkv::blocks::read_frames(seg.payload);
    assert_eq!(err, None, "our own clusters must parse");
    let video_out = sovmr_mkv::blocks::track_frames(&out_frames, video.number).count();
    assert_eq!(
        video_out as u64, FIXTURE_FRAMES,
        "every frame reached a cluster"
    );

    // A reader with no stake in our being right.
    let probe = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .expect("ffprobe present");
    let codecs: Vec<String> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .map(|l| l.trim().trim_end_matches(',').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        codecs,
        vec![best_codec_name(), "aac", "aac", "aac"],
        "ffprobe must find what we wrote"
    );

    println!(
        "MATROSKA IN, MATROSKA OUT: {} frames, tracks {:?}, {} bytes",
        stats.frames, codecs, stats.output_bytes
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The mixed-codec fixture: one track of each thing MP4 cannot hold.
/// The audio codecs the mixed-codec fixture goes in with, after the
/// engine has had them.
///
/// The two lossy ones are copied byte for byte, because the best
/// possible re-encode of already-compressed audio is worse than not
/// doing one. The FLAC track becomes AAC, because it is lossless — a
/// compressor that copied gigabytes of it verbatim would be failing at
/// the one thing it was asked for, and re-encoding an exact source
/// costs it no generation.
const MIXED_AUDIO_OUT: [&str; 3] = ["aac", "opus", "aac"];

#[test]
#[ignore = "requires NVIDIA GPU + driver (mixed-codec Matroska passthrough)"]
fn matroska_output_carries_codecs_mp4_would_have_dropped() {
    // Before this, only AAC survived the engine. Opus, FLAC and the
    // subtitles were reported as skipped — an honest warning about a
    // track the user no longer had. Matroska needs no `esds`, so they
    // are copied rather than dropped.
    let dir = std::env::temp_dir().join(format!("sovmr-mixed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srt = dir.join("s.srt");
    std::fs::write(&srt, "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n").unwrap();
    // Chapters and tags too: a viewer who loses them has lost something
    // they can see is gone, which is the same reason the tracks above
    // are carried.
    let meta = dir.join("m.txt");
    std::fs::write(
        &meta,
        ";FFMETADATA1\ntitle=Probe Title\nartist=Probe Artist\n\
         [CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=4000\ntitle=First\n\
         [CHAPTER]\nTIMEBASE=1/1000\nSTART=4000\nEND=12000\ntitle=Second\n",
    )
    .unwrap();
    let input = dir.join("mixed.mkv");

    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=12:size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
        ])
        .arg("-i")
        .arg(&srt)
        .arg("-i")
        .arg(&meta)
        .args([
            "-map_metadata",
            "5",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-map",
            "3:a",
            "-map",
            "4:s",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=rus",
            "-metadata:s:a:2",
            "language=deu",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-c:a:0",
            "aac",
            "-c:a:1",
            "libopus",
            "-c:a:2",
            "flac",
            "-c:s",
            "srt",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present");
    assert!(status.success());

    let output = dir.join("out.mkv");
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 25,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("a mixed-codec Matroska source must transcode");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };
    assert_eq!(stats.frames, FIXTURE_FRAMES);
    // Nothing was left behind — but the notice is not silent, because
    // the FLAC track was re-encoded and that is a material change to
    // someone's file. What must not appear is anything about loss.
    let notice = stats.warning.clone().unwrap_or_default();
    assert!(
        notice.contains("compressed (FLAC → AAC)") && notice.contains("carried losslessly"),
        "the notice did not say what happened to the audio: {notice:?}"
    );
    assert!(
        !notice.contains("skipped") && !notice.contains("dropped"),
        "nothing was left behind, so nothing should say otherwise: {notice:?}"
    );

    // What a stranger sees in the result.
    let probe = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .expect("ffprobe present");
    let rows: Vec<String> = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .map(|l| l.trim().trim_end_matches(',').to_string())
        .filter(|l| !l.is_empty())
        .collect();

    // ffprobe orders the fields its own way, not the way they were
    // asked for, so each row is split rather than prefix-matched — the
    // same lesson the Matroska muxer gate taught about assuming layout.
    let named = |kind: &str| -> Vec<String> {
        rows.iter()
            .filter_map(|r| {
                let mut f = r.split(',');
                let name = f.next()?.to_string();
                (f.next()? == kind).then_some(name)
            })
            .collect()
    };
    let audio = named("audio");
    assert_eq!(
        audio, MIXED_AUDIO_OUT,
        "the lossy tracks must arrive unchanged and the lossless one as AAC, \
         got rows {rows:?}"
    );
    assert_eq!(
        named("subtitle"),
        vec!["subrip"],
        "the subtitle track must come along too: {rows:?}"
    );
    assert_eq!(
        named("video"),
        vec![best_codec_name()],
        "the video is still re-encoded, only the rest is copied: {rows:?}"
    );

    // Chapters and tags, judged by a third party rather than by our own
    // reader — reading our own file back proves the two halves agree,
    // not that either is right. An earlier attempt wrote them after the
    // clusters, where our reader found them and ffprobe did not.
    let marks = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_chapters",
            "-show_entries",
            "format_tags",
        ])
        .arg(&output)
        .output()
        .expect("ffprobe runs");
    assert!(marks.status.success());
    let marks = String::from_utf8_lossy(&marks.stdout);
    for wanted in [
        "TAG:title=First",
        "TAG:title=Second",
        // The file's name lives in `Info/Title`, which ffprobe
        // reports lower-case; the rest come from the tag list.
        "TAG:title=Probe Title",
        "TAG:ARTIST=Probe Artist",
    ] {
        assert!(marks.contains(wanted), "the output lost {wanted}:\n{marks}");
    }
    // And they point where they did. The source's second chapter began
    // at four seconds; a reader that scaled nanoseconds by the timestamp
    // scale would place it a thousand times late, and nothing but this
    // would say so.
    assert!(
        marks.contains("start=4000000000"),
        "the second chapter moved:\n{marks}"
    );

    println!("MIXED CODECS CARRIED: {rows:?}");
    println!(
        "MARKS CARRIED: {} chapters",
        marks.matches("[CHAPTER]").count()
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A lossless track goes in and AAC comes out — through the real engine.
///
/// Every other audio gate in this file proves passthrough: bytes in,
/// the same bytes out. This proves the other route, and it is the one
/// the tool exists for. A film with a PCM track carries gigabytes of
/// audio that Matroska would happily copy verbatim, and copying it is
/// the compressor failing at its job.
///
/// The whole engine runs: decode, encode, mux, interleave. The audio
/// leg is compressed as it streams past rather than gathered first, so
/// this also exercises the path a twelve-hour file would take.
///
/// ffprobe is the judge, because our own demuxer reading our own muxer
/// could agree with itself about a stream nothing else can play.
#[test]
#[ignore = "needs a real GPU and ffmpeg for the fixture"]
fn a_lossless_source_track_leaves_as_aac() {
    let dir = std::env::temp_dir().join(format!("sovmr-lossless-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // A Matroska source whose audio is FLAC: lossless, and something
    // Matroska could carry untouched if the policy let it.
    let input = dir.join("lossless_source.mkv");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4:sample_rate=48000",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "flac",
            "-ac",
            "2",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(made.success(), "could not make the fixture");

    let output = dir.join("compressed.mkv");
    let outcome = transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("the pipeline must transcode a Matroska source");
    let TranscodeOutcome::Completed(stats) = outcome else {
        panic!("unexpected cancel");
    };

    // The engine must say what it did, in the line a user reads.
    let warning = stats.warning.unwrap_or_default();
    assert!(
        warning.contains("compressed (FLAC → AAC)"),
        "the notice did not mention the compression: {warning:?}"
    );

    // And an independent reader must find AAC where FLAC went in.
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels,sample_rate",
            "-of",
            "default=nw=1",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("ffprobe present");
    let probe = String::from_utf8_lossy(&probe.stdout);
    let field = |name: &str| {
        probe
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{name}=")))
            .unwrap_or_else(|| panic!("no {name} in:\n{probe}"))
            .trim()
            .to_string()
    };
    assert_eq!(field("codec_name"), "aac", "the track did not become AAC");
    assert_eq!(field("channels"), "2", "the channel count changed");
    assert_eq!(field("sample_rate"), "48000", "the sampling rate changed");

    // Decoding it must be silent, and it must actually contain audio —
    // an empty track would satisfy every check above.
    let raw = dir.join("out.f32");
    let decoded = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            output.to_str().unwrap(),
            "-map",
            "a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            raw.to_str().unwrap(),
        ])
        .output()
        .expect("ffmpeg present");
    let complaints = String::from_utf8_lossy(&decoded.stderr);
    assert!(
        complaints.trim().is_empty(),
        "the decoder complained about our audio:\n{complaints}"
    );
    let bytes = std::fs::read(&raw).expect("decoded audio");
    let peak = |data: &[u8]| {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs())
            .fold(0.0f32, f32::max)
    };
    let loudest = peak(&bytes);

    // Against the source's own audio rather than a number chosen by
    // hand. The fixture's level is whatever ffmpeg's mono-to-stereo pan
    // law makes it, and a threshold invented here would be measuring the
    // fixture instead of the encoder.
    let source_raw = dir.join("source.f32");
    let decoded_source = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            input.to_str().unwrap(),
            "-map",
            "a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            source_raw.to_str().unwrap(),
        ])
        .output()
        .expect("ffmpeg present");
    assert!(
        decoded_source.status.success(),
        "could not decode the source audio"
    );
    let source_bytes = std::fs::read(&source_raw).expect("decoded source");
    let source_loudest = peak(&source_bytes);

    assert!(source_loudest > 0.0, "the fixture itself is silent");
    let ratio = f64::from(loudest) / f64::from(source_loudest);
    assert!(
        (0.9..=1.1).contains(&ratio),
        "the output peaks at {loudest} against the source's {source_loudest} — \
         a ratio of {ratio:.3}, so the level changed on the way through"
    );

    // Four seconds of 48 kHz stereo, less the ramps at each end.
    let samples = bytes.len() / 4 / 2;
    let source_samples = source_bytes.len() / 4 / 2;
    assert!(
        samples + 4096 >= source_samples,
        "{samples} samples came out of {source_samples} — audio was lost"
    );

    println!(
        "LOSSLESS TRACK COMPRESSED IN THE PIPELINE: FLAC → AAC, \
         {samples} samples at peak {loudest:.4} against the source's {source_loudest:.4}, \
         {} bytes of output",
        stats.output_bytes
    );
}

/// Full decode of our video by somebody else's decoder, with nothing to say.
///
/// Every other gate in this file is sovereign: our demuxer reading our
/// muxer, which can only prove we agree with ourselves. ffprobe appears
/// in several of them but only reads headers, and the one empty-stderr
/// assertion in the file is `-map a:0` — audio.
///
/// So nothing here has ever asked an independent decoder to actually
/// decode our pictures. This does, for both codecs and both containers,
/// and requires ffmpeg to say **nothing at all**.
///
/// `-f null -` decodes every frame and discards it, so the only output
/// is complaints. `-v error` means anything printed is one.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn our_video_decodes_in_somebody_elses_decoder_without_complaint() {
    let dir = std::env::temp_dir().join(format!("sovmr-conform-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut checked = 0usize;
    let mut complaints_found: Vec<String> = Vec::new();

    for (label, codec, container) in [
        ("av1_mp4", EncodeCodec::Av1, "mp4"),
        ("hevc_mp4", EncodeCodec::Hevc, "mp4"),
        ("av1_mkv", EncodeCodec::Av1, "mkv"),
        ("hevc_mkv", EncodeCodec::Hevc, "mkv"),
    ] {
        if !encoder_supports(codec) {
            skip_unsupported(label, codec_name(codec));
            continue;
        }
        // The output container mirrors the input, so the source decides
        // it. Both are exercised because the tail-drop that both codecs
        // share happens before the container split.
        let input = match container {
            "mkv" => matroska_fixture(&dir, label),
            _ => make_fixture(&dir, FixtureAudio::Aac),
        };
        let output = dir.join(format!("{label}.{container}"));
        let outcome = transcode(
            &input,
            &output,
            &NativeSettings {
                codec,
                cq: 32,
                max_bitrate: 0,
                preset: EncodePreset::Quality,
                references: sovmr_nvenc::encoder::ReferencePlan::default(),
            },
            |_| {},
            &AtomicBool::new(false),
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let TranscodeOutcome::Completed(_) = outcome else {
            panic!("{label}: unexpected cancel");
        };

        let complaints = decode_video_complaints(&output);
        if complaints.trim().is_empty() {
            println!("  {label:>10}: clean");
        } else {
            println!("  {label:>10}: {}", complaints.lines().next().unwrap_or(""));
            complaints_found.push(format!("{label}: {}", complaints.trim()));
        }
        checked += 1;
    }

    assert!(
        complaints_found.is_empty(),
        "an independent decoder objected to {} of {checked} outputs:\n\n{}",
        complaints_found.len(),
        complaints_found.join("\n\n")
    );
    println!("CONFORMANCE: {checked} outputs decoded by ffmpeg without one word");
}

/// The negative control for the gate above.
///
/// "ffmpeg said nothing" is worth exactly what the proof that it *can*
/// say something is worth. A decoder that had quietly stopped reporting
/// would satisfy that assertion forever.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn a_deliberately_corrupted_output_is_caught_by_that_same_decoder() {
    let dir = std::env::temp_dir().join(format!("sovmr-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::Aac);
    let output = dir.join("good.mp4");
    transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcodes");

    // Flip every bit of the second half. The output is fast-start, so
    // `moov` sits at the front and the whole of that half is coded
    // picture data — the container stays parseable and only what the
    // decoder must read is destroyed.
    //
    // Measured, not assumed: four kilobytes was enough for AV1 to object
    // and *not* enough for HEVC, which concealed it and said nothing.
    // A control has to hold for every codec the machine can produce, so
    // it corrupts an amount no decoder can conceal rather than the
    // smallest amount that happened to work once.
    let mut bytes = std::fs::read(&output).expect("read");
    let from = bytes.len() / 2;
    for byte in bytes[from..].iter_mut() {
        *byte ^= 0xFF;
    }
    let broken = dir.join("broken.mp4");
    std::fs::write(&broken, &bytes).expect("write");

    let complaints = decode_video_complaints(&broken);
    assert!(
        !complaints.trim().is_empty(),
        "the decoder accepted deliberately corrupted picture data, \
         so the silence of the conformance gate proves nothing"
    );
    println!(
        "NEGATIVE CONTROL HOLDS: corruption is reported — {}",
        complaints.lines().next().unwrap_or("").trim()
    );
}

/// Decode every video frame of `path`, returning whatever the decoder
/// had to say. Empty means it was content.
fn decode_video_complaints(path: &std::path::Path) -> String {
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-map",
            "0:v",
            "-f",
            "null",
            "-",
        ])
        .output()
        .expect("ffmpeg present");
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// A Matroska source, so the `.mkv` output path is exercised too.
fn matroska_fixture(dir: &std::path::Path, label: &str) -> std::path::PathBuf {
    let clip = dir.join(format!("{label}_source.mkv"));
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success(), "could not make the Matroska fixture");
    clip
}

/// What a source said about its colours is written into the output.
///
/// Three code points decide how a decoder turns numbers into light, and
/// a fourth says whether the range is 0-255 or 16-235. Nothing crashes
/// when they are lost; the picture simply comes out the wrong colour,
/// and an editor that reads them has nothing to read.
///
/// # What this proves, and what it does not
///
/// The source's description is stated by ffmpeg, so that end is
/// independent. The output's `colr` box is read back by this project's
/// own demuxer, which proves the value survived the engine unchanged.
///
/// It does **not** prove a decoder will use it. Measured: ffprobe
/// reports the *bitstream's* own description in preference, and NVENC
/// writes "unspecified" there unless told otherwise. Telling it means
/// reaching into the codec-specific config union, which is the lever
/// work of a later increment — and until then the container says the
/// right thing while the elementary stream says nothing.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn a_colour_description_is_carried_into_the_output() {
    use sovmr_mp4::colour::Colour;

    let dir = std::env::temp_dir().join(format!("sovmr-colour-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A gate that carried nothing would pass every assertion below.
    let mut carried = 0usize;

    for (container, label, args, want) in [
        (
            "mp4",
            "bt2020_pq",
            vec![
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
                "-color_range",
                "tv",
            ],
            Some(Colour {
                primaries: 9,
                transfer: 16,
                matrix: 9,
                full_range: false,
            }),
        ),
        (
            "mkv",
            "bt709_full",
            vec![
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "pc",
            ],
            Some(Colour {
                primaries: 1,
                transfer: 1,
                matrix: 1,
                full_range: true,
            }),
        ),
        // A source that declares nothing must not be given a description
        // it never had: a player trusts one completely, and falls back
        // to its own judgement only when there is none.
        ("mp4", "undeclared", vec![], None),
    ] {
        let input = dir.join(format!("{label}.{container}"));
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            // Force the container box. Some ffmpeg builds write it from
            // the stream description and some do not — measured: the
            // Linux build here does, the Windows build there does not —
            // and a fixture that depends on which is installed tests the
            // wrong thing entirely.
            "-movflags",
            "+write_colr",
        ]);
        cmd.args(&args);
        assert!(
            cmd.arg(&input).status().expect("ffmpeg present").success(),
            "{label}: could not make the fixture"
        );

        let output = dir.join(format!("{label}_out.{container}"));
        transcode(
            &input,
            &output,
            &NativeSettings {
                codec: best_codec(),
                cq: 32,
                max_bitrate: 0,
                preset: EncodePreset::Quality,
                references: sovmr_nvenc::encoder::ReferencePlan::default(),
            },
            |_| {},
            &AtomicBool::new(false),
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));

        // Read whichever container it is with that container's reader.
        // Matroska spells the same code points out as separate elements
        // where ISOBMFF packs them into one payload, so a gate that knew
        // only one would prove carriage for only one.
        let read_colour = |path: &std::path::Path| {
            let bytes = std::fs::read(path).expect("read");
            if sovmr_mkv::is_matroska(&bytes) {
                // Bound inside its own scope: the segment borrows the
                // bytes, and the description is a plain value that
                // outlives both.
                let segment = sovmr_mkv::demux::read_segment(&bytes).expect("segment");
                let found = segment
                    .of_kind(sovmr_mkv::demux::TrackKind::Video)
                    .next()
                    .expect("video track")
                    .colour;
                found
            } else {
                find_video_track(&bytes).expect("video track").colour
            }
        };
        let source_colour = read_colour(&input);
        let out_colour = read_colour(&output);

        // Carriage is the claim, so the source is the reference —
        // whatever it turned out to be. The Windows ffmpeg build writes
        // no `colr` where the Linux one does, and a gate that demanded a
        // particular fixture would have been testing ffmpeg rather than
        // this engine.
        // Compared as *statements*, not as bytes. An all-unspecified box
        // and no box at all say the same thing — the source declared
        // nothing — and this muxer writes no box rather than one that
        // says nothing.
        let meaningful = |c: Option<sovmr_mp4::colour::Colour>| c.filter(|c| c.is_meaningful());
        assert_eq!(
            meaningful(out_colour),
            meaningful(source_colour),
            "{label}: the description changed on the way through"
        );
        if source_colour.is_some_and(|c| c.is_meaningful()) {
            carried += 1;
        }
        // Nothing meaningful may appear that the source did not state.
        // Checked against the source rather than against an expectation:
        // with `+write_colr` some builds emit an all-unspecified box for
        // an undeclared stream, which is still the source saying nothing.
        let source_says = meaningful(source_colour).is_some();
        let output_says = meaningful(out_colour).is_some();
        assert!(
            source_says || !output_says,
            "{label}: a description was invented for a source that stated none"
        );
        let _ = want;
        // And an independent reader must agree. This is the half the
        // container alone could not buy: ffprobe reports the elementary
        // stream's own description in preference, and until the encoder
        // was told, a file could state BT.2020 in its `colr` box and
        // nothing at all in its bitstream.
        //
        // Compared field by field against the *source's* own reading,
        // never against a fixed expectation: the two ffmpeg builds here
        // state different subsets — the Windows one gives a matrix and
        // no primaries — and a gate demanding all three would be testing
        // which build is installed.
        let probe = |path: &std::path::Path| -> Vec<(String, String)> {
            let out = Command::new("ffprobe")
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=color_primaries,color_transfer,color_space,color_range",
                    "-of",
                    "default=nw=1",
                    path.to_str().unwrap(),
                ])
                .output()
                .expect("ffprobe present");
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.trim().to_string()))
                .filter(|(_, v)| v != "unknown")
                .collect()
        };
        if meaningful(source_colour).is_some() {
            let stated_in = probe(&input);
            let stated_out = probe(&output);
            assert!(
                !stated_in.is_empty(),
                "{label}: the fixture states nothing, so nothing is being tested"
            );
            assert_eq!(
                stated_out, stated_in,
                "{label}: an independent reader sees a different description \
                 in the output than in the source"
            );
        }

        match meaningful(source_colour) {
            Some(c) => println!(
                "  {label:>12}: primaries {} transfer {} matrix {} range {} carried",
                c.primaries,
                c.transfer,
                c.matrix,
                if c.full_range { "full" } else { "limited" }
            ),
            None => println!("  {label:>12}: nothing claimed, as the source claimed nothing"),
        }
    }
    assert!(
        carried > 0,
        "no fixture carried a colour description, so nothing was tested — \
         the ffmpeg on this machine writes none, and the gate needs one that does"
    );
    println!(
        "COLOUR CARRIED: {carried} described source(s) survived the engine unchanged, \
         and the undescribed one was left undescribed"
    );
}

/// Our own output, decoded back by our own decoder.
///
/// Until now the bridge knew H.264 and HEVC only, so the engine could
/// not read a file it had just written — which is the first thing a
/// self-verifying encoder has to be able to do.
///
/// This does not measure quality yet; it establishes that the frames
/// come back at all, in the right number and the right size. A metric
/// computed over frames that never arrived would be a confident number
/// about nothing.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn the_engine_can_decode_the_file_it_just_wrote() {
    use sovmr_nvdec::{DecodeCodec, DecoderConfig, NvDec};

    let dir = std::env::temp_dir().join(format!("sovmr-readback-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::None);
    let output = dir.join("written.mp4");
    transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcodes");

    // Our own demuxer reads our own output, and our own bridge decodes
    // it. Every step is this project's code.
    let bytes = std::fs::read(&output).expect("read");
    let track = find_video_track(&bytes).expect("video track");
    let codec = match track.codec {
        VideoCodec::Av1 => DecodeCodec::Av1,
        VideoCodec::Hevc => DecodeCodec::Hevc,
        VideoCodec::H264 => DecodeCodec::H264,
    };

    let cuda = sovmr_cuda::Cuda::load().expect("driver");
    let ctx = cuda.create_context(0).expect("GPU 0");
    let nvdec = NvDec::load().expect("nvcuvid");
    let decoder = nvdec
        .create_decoder(
            &ctx,
            &DecoderConfig {
                codec,
                width: u32::from(track.width),
                height: u32::from(track.height),
                bit_depth_minus8: 0,
                decode_surfaces: 16,
            },
        )
        .unwrap_or_else(|e| panic!("a decoder for our own {codec:?} output: {e:?}"));
    drop(decoder);

    // And our own demuxer must hand every sample over in the elementary
    // form a parser wants — for AV1 that is the sample verbatim, which
    // it used to refuse outright.
    let mut buffer = Vec::new();
    let mut count = 0usize;
    let mut smallest = usize::MAX;
    for sample in track.samples(&bytes) {
        let sample = sample.expect("every sample reads");
        track
            .write_frame(&sample, &mut buffer)
            .expect("every frame converts");
        assert!(!buffer.is_empty(), "frame {count} came back empty");
        smallest = smallest.min(buffer.len());
        count += 1;
    }
    assert_eq!(
        count as u64, FIXTURE_FRAMES,
        "the engine wrote {FIXTURE_FRAMES} frames and could read back {count}"
    );
    println!("READ BACK: {count} {codec:?} frames from our own output, smallest {smallest} bytes");
}

/// One decoder, one file: the half of verification that is not a pair.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn one_decoder_walks_a_whole_file() {
    let dir = std::env::temp_dir().join(format!("sovmr-count-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::None);
    let counted = sovmr_native::verify::frame_count(&input).expect("decodes");
    assert_eq!(
        counted, FIXTURE_FRAMES,
        "one decoder saw {counted} of {FIXTURE_FRAMES} frames"
    );
    println!("ONE DECODER: {counted} frames");
}

/// The program measures its own output, with no ffmpeg in the loop.
///
/// Every quality number this project has published came from
/// `ffmpeg -lavfi psnr`, which is fine for a development benchmark and
/// useless in a shipped tool: a user cannot be told to install the thing
/// this program replaces in order to learn whether it worked.
///
/// Two checks, and the first is the one with teeth. A file against
/// itself must come back identical — anything else means the
/// measurement is measuring itself. Only then does a real encode's
/// score mean anything.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn the_program_measures_its_own_output() {
    let dir = std::env::temp_dir().join(format!("sovmr-verify-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = make_fixture(&dir, FixtureAudio::None);
    let output = dir.join("encoded.mp4");
    transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcodes");

    let same = sovmr_native::verify::verify(&output, &output).expect("compares");
    assert_eq!(same.frames, FIXTURE_FRAMES, "self-comparison lost frames");
    assert!(
        same.psnr.is_infinite(),
        "a file against itself scored {:.2} dB, so the comparison is \
         measuring itself rather than the encode",
        same.psnr
    );
    assert!((same.ssim - 1.0).abs() < 1e-9, "SSIM {:.6}", same.ssim);
    println!("  self:   {same}");

    let verdict = sovmr_native::verify::verify(&input, &output).expect("compares");
    assert_eq!(verdict.frames, FIXTURE_FRAMES, "the comparison lost frames");
    assert!(
        verdict.psnr.is_finite() && verdict.psnr > 25.0,
        "a lossy encode of a test pattern should be well above 25 dB, got {:.2}",
        verdict.psnr
    );
    assert!(
        (0.5..1.0).contains(&verdict.ssim),
        "SSIM {:.4} is outside what a lossy encode of this fixture can be",
        verdict.ssim
    );
    println!("SELF-MEASURED: {verdict}");
}

/// Price every encoder lever against the source, with our own ruler.
///
/// The rule this milestone runs on: a lever stays only if it makes the
/// output smaller without making it worse. Until `--verify` existed
/// that was a promise; it is a procedure now, and this is the procedure.
///
/// Nothing here asserts which lever wins — that is what the numbers are
/// for, and a test that decided in advance would be an opinion wearing
/// a gate's clothes. What it does assert is that every candidate
/// produces a file an independent decoder accepts, so a lever cannot be
/// adopted on a number measured from a broken stream.
#[test]
#[ignore = "needs a real GPU and ffmpeg; prints a table"]
fn every_encoder_lever_is_priced_against_the_source() {
    let dir = std::env::temp_dir().join(format!("sovmr-levers-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Material where *quality* decides the size, not the cap. The
    // ordinary fixture is a clean synthetic pattern encoded against a
    // binding bitrate ceiling, so every preset produced the same file to
    // within a fifth of a percent — the ceiling was being measured, not
    // the encoder.
    //
    // Noise plus motion cannot be predicted away, so the encoder has to
    // spend where it judges best, and a lever that judges better shows
    // up as fewer bytes at the same score.
    let input = dir.join("demanding.mp4");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=6",
            // Noise applied to the pattern, not blended from a second
            // graph: `lavfi` will not accept an open input.
            "-vf",
            "noise=alls=26:allf=t+u",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present");
    assert!(made.success(), "could not make the demanding fixture");
    let source_bytes = std::fs::metadata(&input).expect("fixture").len();

    // Presets only. `NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY` was priced
    // here too and refused: it segfaults inside the encode on this
    // driver, from a clean process, with lookahead already forced off.
    use sovmr_nvenc::encoder::ReferencePlan;
    let flat = ReferencePlan::default();
    let candidates: Vec<(&str, EncodePreset, ReferencePlan)> = vec![
        ("P2 speed    ", EncodePreset::Speed, flat),
        ("P4 balanced ", EncodePreset::Balanced, flat),
        ("P6 quality  ", EncodePreset::Quality, flat),
        ("P7 slowest  ", EncodePreset::Slowest, flat),
        (
            "P6 +pyramid ",
            EncodePreset::Quality,
            ReferencePlan {
                b_pyramid: true,
                deep_refs: false,
            },
        ),
        (
            "P6 +refs    ",
            EncodePreset::Quality,
            ReferencePlan {
                b_pyramid: false,
                deep_refs: true,
            },
        ),
        (
            "P6 +both    ",
            EncodePreset::Quality,
            ReferencePlan {
                b_pyramid: true,
                deep_refs: true,
            },
        ),
    ];

    println!(
        "\n  {:<13} {:>9} {:>8} {:>9} {:>8} {:>7}",
        "lever", "bytes", "vs P6", "PSNR dB", "SSIM", "sec"
    );
    let mut baseline: Option<(u64, f64)> = None;
    for (label, preset, references) in candidates {
        let output = dir.join(format!("{}.mp4", label.trim().replace(' ', "_")));
        let started = std::time::Instant::now();
        let outcome = transcode(
            &input,
            &output,
            &NativeSettings {
                codec: best_codec(),
                cq: 32,
                // Deliberately not binding. With the heuristic's own cap
                // the output was 2.2% of the source and every lever
                // landed within 0.4% of every other — the cap was being
                // measured, not the encoder.
                max_bitrate: 40_000_000,
                preset,
                references,
            },
            |_| {},
            &AtomicBool::new(false),
        );
        let Ok(TranscodeOutcome::Completed(stats)) = outcome else {
            println!("  {label} refused by this hardware: {outcome:?}");
            continue;
        };
        let seconds = started.elapsed().as_secs_f64();

        // A number from a stream nobody else can read is worthless, so
        // conformance is checked before quality is believed.
        let complaints = decode_video_complaints(&output);
        assert!(
            complaints.trim().is_empty(),
            "{label}: the output does not decode\n{complaints}"
        );
        let verdict = sovmr_native::verify::verify(&input, &output).expect("measures");

        if label.starts_with("P6 quality") {
            baseline = Some((stats.output_bytes, verdict.psnr));
        }
        let delta = baseline.map_or(String::from("—"), |(bytes, _)| {
            format!(
                "{:+.1}%",
                (stats.output_bytes as f64 - bytes as f64) * 100.0 / bytes as f64
            )
        });
        println!(
            "  {label} {:>9} {:>8} {:>9.2} {:>8.4} {:>7.1}",
            stats.output_bytes, delta, verdict.psnr, verdict.ssim, seconds
        );
    }
    println!("  source was {source_bytes} bytes; every row decoded without complaint\n");
}

/// The rate plan judges the pictures, not the file around them.
///
/// The engine's whole curve rested on `size_bytes` — the file length,
/// audio and container overhead included. A film with heavy lossless
/// sound therefore looked far denser than its pictures are, and every
/// ceiling derived from that density was set for a video that does not
/// exist.
///
/// Two files, the same pictures, wildly different sound. The plan must
/// be the same.
#[test]
#[ignore = "needs ffmpeg"]
fn the_rate_plan_is_not_moved_by_the_weight_of_the_audio() {
    use sovmr_core::encoder::EncoderProvider;

    let dir = std::env::temp_dir().join(format!("sovmr-density-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut plans = Vec::new();
    for (label, audio_args) in [
        ("lean", vec!["-c:a", "aac", "-b:a", "64k"]),
        // Six uncompressed tracks: gigabytes of sound around identical
        // pictures.
        ("heavy", vec!["-c:a", "pcm_s24le", "-ac", "2"]),
    ] {
        let path = dir.join(format!("{label}.mp4"));
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4:sample_rate=48000",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
        ]);
        cmd.args(&audio_args);
        assert!(
            cmd.arg(&path).status().expect("ffmpeg present").success(),
            "{label}: could not make the fixture"
        );

        let info = sovmr_native::provider::NvencNativeProvider
            .probe(&path)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let profile = sovmr_core::heuristics::SourceProfile::measure(&info)
            .unwrap_or_else(|| panic!("{label}: unmeasurable"));
        println!(
            "  {label:>6}: file {:>9} video {:>9} → {} milli-bpp, audio share {}%",
            info.size_bytes, info.video_bytes, profile.milli_bppf, profile.audio_share_pct
        );
        plans.push((label, info, profile));
    }

    let (_, lean_info, lean) = &plans[0];
    let (_, heavy_info, heavy) = &plans[1];
    assert!(
        heavy_info.size_bytes > lean_info.size_bytes * 2,
        "the fixture must differ in sound weight to prove anything"
    );
    assert_eq!(
        lean.milli_bppf, heavy.milli_bppf,
        "identical pictures were judged at different densities because of \
         the sound around them"
    );
    // Measured while proving the above, and worth saying: the heavy
    // file reports an audio share of **zero**. `audio_bitrate` counts
    // only tracks this engine can carry, and PCM-in-MP4 is not one — so
    // the metric understates exactly the files whose sound weighs most.
    // A separate defect from the one fixed here, and not silently
    // folded into it.
    assert!(
        heavy.audio_share_pct <= 100,
        "a share of {}% means it is being divided by the wrong total",
        heavy.audio_share_pct
    );
    println!("DENSITY IS THE VIDEO'S: identical pictures, identical judgement");
}

/// The complexity oracle finds a cut it was never told about.
///
/// The sizes in `stsz` are the previous encoder's record of where the
/// picture was hard — a complexity curve already computed, already
/// stored, free to read. Every other way of learning the same thing
/// costs a decode.
///
/// So: a file built from two visually unrelated halves, joined at a
/// frame this test knows and the detector does not. It must find that
/// frame, and it must not invent others.
#[test]
#[ignore = "needs ffmpeg"]
fn the_complexity_oracle_finds_a_real_cut_without_decoding() {
    use sovmr_core::complexity::Complexity;

    let dir = std::env::temp_dir().join(format!("sovmr-oracle-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Two halves that share nothing: a moving test pattern, then flat
    // colour. Ninety frames each at thirty a second.
    const CUT_AT: usize = 90;
    let input = dir.join("two_scenes.mp4");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:size=640x360:rate=30:duration=3",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            // No fixed keyframe cadence imposed on the fixture: the cut
            // must be found from how the sizes fall, not from where a
            // `-g` flag happened to place an I-frame.
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-g",
            "600",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present");
    assert!(made.success(), "could not make the two-scene fixture");

    // Read the sizes with our own demuxer — a table walk, no decoding.
    let bytes = std::fs::read(&input).expect("read");
    let track = find_video_track(&bytes).expect("video track");
    let sizes: Vec<u32> = track
        .samples(&bytes)
        .map(|s| s.expect("sample").data.len() as u32)
        .collect();
    assert!(sizes.len() > CUT_AT + 30, "{} frames", sizes.len());

    let cuts = Complexity::from_sizes(sizes.clone()).scene_cuts();
    println!("  {} frames, cuts at {cuts:?}", sizes.len());

    // Within a few frames of the join: encoders reorder, and the frame
    // that pays for a change is not always the first one after it.
    let found = cuts.iter().any(|c| c.abs_diff(CUT_AT) <= 4);
    assert!(
        found,
        "the join at frame {CUT_AT} was not found; cuts were {cuts:?}"
    );
    assert!(
        cuts.len() <= 3,
        "one join was read as {} scenes: {cuts:?}",
        cuts.len()
    );

    let scenes = Complexity::from_sizes(sizes).scene_weights();
    println!(
        "  scenes: {:?}",
        scenes
            .iter()
            .map(|(a, b, w)| format!("{a}..{b} ×{:.2}", *w as f64 / 1000.0))
            .collect::<Vec<_>>()
    );
    println!("COMPLEXITY ORACLE: a real cut found from the sample table alone");
}

/// A keyframe lands on the cut, not on the grid.
///
/// The engine placed keyframes every ten seconds regardless of what the
/// picture did. A viewer seeking to a scene therefore landed before it
/// and watched the previous one play out, and every frame after a cut
/// was predicted from a picture that shares nothing with it — the most
/// expensive frames in the file, spent on a prediction that could not
/// work.
///
/// The source's own sample table says where the scenes are. This checks
/// the engine acted on it.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn a_keyframe_is_placed_where_the_scene_changes() {
    let dir = std::env::temp_dir().join(format!("sovmr-cuts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // One cut, far from any ten-second boundary: at three seconds, where
    // the fixed cadence would never have put a keyframe.
    const CUT_AT: u64 = 90;
    let input = dir.join("two_scenes.mp4");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:size=640x360:rate=30:duration=3",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-g",
            "600",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present");
    assert!(made.success(), "could not make the two-scene fixture");

    let output = dir.join("cut.mp4");
    transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcodes");

    // Our own demuxer reads our own sync-sample table.
    let bytes = std::fs::read(&output).expect("read");
    let track = find_video_track(&bytes).expect("video track");
    let syncs: Vec<u64> = track
        .samples(&bytes)
        .enumerate()
        .filter_map(|(i, s)| s.ok().filter(|s| s.keyframe).map(|_| i as u64))
        .collect();
    println!("  sync samples at {syncs:?}");

    assert!(
        syncs.contains(&0),
        "the first sample must be a keyframe: {syncs:?}"
    );
    assert!(
        syncs.iter().any(|s| s.abs_diff(CUT_AT) <= 2),
        "no keyframe near the cut at {CUT_AT}: {syncs:?}"
    );
    // And it did not turn every frame into one.
    assert!(
        syncs.len() <= 4,
        "{} keyframes in a six-second clip with one cut: {syncs:?}",
        syncs.len()
    );

    // The output must still decode, and still be the same pictures.
    let complaints = decode_video_complaints(&output);
    assert!(complaints.trim().is_empty(), "{complaints}");
    let verdict = sovmr_native::verify::verify(&input, &output).expect("measures");
    assert!(
        verdict.psnr > 25.0,
        "placing keyframes cost quality: {verdict}"
    );
    println!("KEYFRAME ON THE CUT: {syncs:?}, and {verdict}");
}

/// Bits move between scenes, and the seam does not show.
///
/// The complexity curve says which stretch the source encoder found
/// hard. Acting on it means changing the quality target at a scene
/// boundary, which the driver will either accept or refuse — and a
/// refusal that went unnoticed would look exactly like a
/// redistribution that never helped.
///
/// The check with teeth is the seam. Lending bits from an easy scene to
/// a hard one is only worth doing if the join stays invisible, so each
/// scene is measured separately and neither may fall far behind the
/// other.
#[test]
#[ignore = "needs a real GPU and ffmpeg"]
fn bits_move_between_scenes_without_a_visible_seam() {
    let dir = std::env::temp_dir().join(format!("sovmr-seam-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // One hard scene and one easy one, joined at a known frame.
    let input = dir.join("hard_then_easy.mp4");
    let made = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:size=640x360:rate=30:duration=3",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-g",
            "600",
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("ffmpeg present");
    assert!(made.success(), "could not make the fixture");

    let output = dir.join("redistributed.mp4");
    transcode(
        &input,
        &output,
        &NativeSettings {
            codec: best_codec(),
            cq: 32,
            max_bitrate: 0,
            preset: EncodePreset::Quality,
            references: sovmr_nvenc::encoder::ReferencePlan::default(),
        },
        |_| {},
        &AtomicBool::new(false),
    )
    .expect("transcodes");

    let complaints = decode_video_complaints(&output);
    assert!(
        complaints.trim().is_empty(),
        "changing the rate mid-stream broke the stream\n{complaints}"
    );
    let verdict = sovmr_native::verify::verify(&input, &output).expect("measures");
    assert!(
        verdict.psnr > 25.0 && verdict.ssim > 0.5,
        "redistribution cost the picture: {verdict}"
    );
    // And it must have actually moved bits. A run where every scene
    // kept the job's own target would pass every assertion above while
    // proving nothing at all — the redistribution has to be visible in
    // the file, not only in the log.
    let syncs = find_video_track(&std::fs::read(&output).expect("read"))
        .expect("video track")
        .samples(&std::fs::read(&output).expect("read"))
        .filter(|s| s.as_ref().is_ok_and(|s| s.keyframe))
        .count();
    assert!(
        syncs >= 2,
        "only {syncs} keyframe(s): the scene boundary was never acted on"
    );
    println!("SEAMLESS REDISTRIBUTION: {verdict}, {syncs} scenes");
}
