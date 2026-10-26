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

//! Lossless in, AAC out, checked end to end against the original audio.
//!
//! The unit tests prove the routing decision is the one intended. This
//! proves the route actually works: a real FLAC file, made by ffmpeg,
//! demuxed by this project's Matroska reader, decoded by its FLAC
//! decoder, encoded by its AAC encoder, muxed by its Matroska muxer, and
//! handed back to ffmpeg — which must decode it without complaint into
//! something measurably the same audio.
//!
//! Every step but the first and the last is this project's own code, and
//! nothing in the chain has ever been checked against the others.

use std::path::{Path, PathBuf};
use std::process::Command;

use sovmr_native::audio::{compress, route, Lossless, Route};

/// Where fixtures live for this process.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sovmr-route-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Decode a file to mono f64 samples, with whatever the decoder said.
fn decode_to_samples(path: &Path) -> (Vec<f64>, String) {
    let raw = path.with_extension("f32");
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
    let bytes = std::fs::read(&raw).unwrap_or_default();
    (
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The offset that best lines two signals up, and how well.
fn best_alignment(reference: &[f64], candidate: &[f64]) -> (usize, f64) {
    const RAMP: usize = 2048;
    const WINDOW: usize = 8192;
    let window = WINDOW.min(reference.len().saturating_sub(RAMP));
    let head = &reference[RAMP..RAMP + window];
    let energy: f64 = head.iter().map(|s| s * s).sum::<f64>().sqrt();
    let mut best = (0usize, 0.0f64);
    for offset in 0..=8192usize {
        if offset + RAMP + window > candidate.len() {
            break;
        }
        let slice = &candidate[offset + RAMP..offset + RAMP + window];
        let dot: f64 = head.iter().zip(slice).map(|(a, b)| a * b).sum();
        let slice_energy: f64 = slice.iter().map(|s| s * s).sum::<f64>().sqrt();
        let correlation = dot / (energy * slice_energy).max(f64::MIN_POSITIVE);
        if correlation > best.1 {
            best = (offset, correlation);
        }
    }
    best
}

/// A lossless track, compressed to AAC, still sounding like itself.
#[test]
#[ignore = "needs ffmpeg to make the fixture and read the result"]
fn a_lossless_track_becomes_aac_that_still_sounds_like_the_original() {
    const RATE: u32 = 48_000;
    const RAMP: usize = 2048;
    const FRAME: usize = 1024;
    const FLOOR_DB: f64 = 40.0;

    let dir = scratch("lossless");
    let mut checked = 0usize;
    let mut worst = f64::INFINITY;

    for (label, codec, channels) in [
        ("flac_mono", "flac", 1u32),
        ("flac_stereo", "flac", 2),
        ("pcm_mono", "pcm_s16le", 1),
        ("pcm_stereo", "pcm_s16le", 2),
    ] {
        // A source with content in it, at a rate and layout the encoder
        // can take.
        let source = dir.join(format!("{label}.mkv"));
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
            .arg(format!("sine=frequency=440:duration=1:sample_rate={RATE}"))
            .args([
                "-ac",
                &channels.to_string(),
                "-c:a",
                codec,
                source.to_str().unwrap(),
            ])
            .status()
            .expect("ffmpeg runs");
        assert!(made.success(), "{label}: could not make the fixture");

        // Read it with our own demuxer.
        let bytes = std::fs::read(&source).expect("read the fixture");
        let segment = sovmr_mkv::demux::read_segment(&bytes).expect("segment");
        let track = segment
            .of_kind(sovmr_mkv::demux::TrackKind::Audio)
            .next()
            .expect("an audio track");

        // The routing decision, on the real track. Into MP4, where only
        // AAC can go, a lossless track must choose compression.
        let decision = route(
            track.codec_id,
            track.codec_private,
            track.sample_rate,
            u32::from(track.channels),
            sovmr_mkv::source::Target::Mp4,
        );
        assert!(
            matches!(decision, Route::Compress(_)),
            "{label}: routed to {decision:?} rather than compression"
        );
        let Route::Compress(kind) = decision else {
            unreachable!("just asserted")
        };
        match (codec, kind) {
            ("flac", Lossless::Flac) | ("pcm_s16le", Lossless::Pcm(_)) => {}
            _ => panic!("{label}: routed through the wrong decoder, {kind:?}"),
        }

        // Compress it, from the real packets.
        let (frames, error) = sovmr_mkv::blocks::read_frames(segment.payload);
        assert_eq!(error, None, "{label}: our demuxer stumbled");
        let packets: Vec<&[u8]> = frames
            .iter()
            .filter(|f| u64::from(f.track) == track.number)
            .map(|f| f.data)
            .collect();
        assert!(!packets.is_empty(), "{label}: no packets");
        let compressed = compress(
            kind,
            &packets,
            track.codec_private,
            track.sample_rate,
            u32::from(track.channels),
        )
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(compressed.channels, u32::from(track.channels));

        // Mux our AAC with our own muxer and hand it back to ffmpeg.
        let out = dir.join(format!("{label}_aac.mkv"));
        let spec = sovmr_mkv::mux::TrackSpec {
            number: 1,
            codec_id: "A_AAC".to_string(),
            codec_private: compressed.codec_private.clone(),
            video: false,
            subtitle: false,
            width: 0,
            height: 0,
            sample_rate: track.sample_rate,
            channels: compressed.channels as u16,
            language: "und".to_string(),
            name: None,
            enabled: true,
            colour: None,
        };
        let mut muxer = sovmr_mkv::mux::MkvMuxer::create(&out, &[spec]).expect("muxer");
        let at = |n: u64| (n * FRAME as u64 * 1000 / u64::from(track.sample_rate)) as i64;
        for (n, frame) in compressed.frames.iter().enumerate() {
            let n = n as u64;
            muxer
                .add_frame(1, at(n), at(n + 1) - at(n), true, frame)
                .expect("frame");
        }
        muxer.finalize().expect("finalize");

        let (theirs, complaints) = decode_to_samples(&out);
        assert!(
            complaints.trim().is_empty(),
            "{label}: the decoder complained about our AAC:\n{complaints}"
        );

        // And the original, to compare against. Its own decode, so any
        // difference is ours rather than the fixture's.
        let (original, complaints) = decode_to_samples(&source);
        assert!(complaints.trim().is_empty(), "{label}: {complaints}");

        // Compare channel by channel, de-interleaving both.
        let stride = compressed.channels as usize;
        for channel in 0..stride {
            let want: Vec<f64> = original
                .iter()
                .skip(channel)
                .step_by(stride)
                .copied()
                .collect();
            let got: Vec<f64> = theirs
                .iter()
                .skip(channel)
                .step_by(stride)
                .copied()
                .collect();
            let (offset, correlation) = best_alignment(&want, &got);
            assert!(
                correlation > 0.9,
                "{label} channel {channel}: best correlation is only {correlation:.3} — \
                 the compressed audio does not resemble the source at any offset"
            );
            let to = (want.len() - RAMP).min(got.len().saturating_sub(offset + FRAME));
            assert!(
                to > RAMP + FRAME,
                "{label} channel {channel}: too little overlap"
            );
            let mut signal = 0.0;
            let mut noise = 0.0;
            for i in RAMP..to {
                signal += want[i] * want[i];
                noise += (want[i] - got[i + offset]).powi(2);
            }
            let snr = 10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10();
            assert!(
                snr > FLOOR_DB,
                "{label} channel {channel}: {snr:.1} dB, below the {FLOOR_DB} dB floor"
            );
            println!("  {label:>12} channel {channel}: {snr:5.1} dB");
            worst = worst.min(snr);
            checked += 1;
        }
    }

    println!(
        "LOSSLESS → AAC: {checked} channels across four tracks, worst {worst:.1} dB, \
         every step this project's own code"
    );
}
