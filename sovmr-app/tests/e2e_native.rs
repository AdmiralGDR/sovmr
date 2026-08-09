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

//! Native-pipeline end-to-end: a *real* MP4 (produced by ffmpeg purely as
//! a fixture generator) demuxed by `sovmr-mp4` — no FFmpeg at parse
//! time — its geometry feeding `cuvidCreateDecoder` dynamically, and its
//! first sample converted to Annex B.
//!
//! ```sh
//! cargo test -p sovmr-app --test e2e_native -- --ignored
//! ```

use std::process::Command;

use sovmr_mp4::demux::{find_video_track, VideoCodec};
use sovmr_nvdec::{DecodeCodec, DecoderConfig, NvDec};

/// Annex B start code our converter must emit.
const START_CODE: [u8; 4] = [0, 0, 0, 1];
/// H.264 NAL type mask + SPS type value.
const NAL_TYPE_MASK: u8 = 0x1F;
const NAL_SPS: u8 = 7;

fn make_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let clip = dir.join("native_fixture.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=2:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
        ])
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success());
    clip
}

#[test]
#[ignore = "requires ffmpeg (fixture only) + NVIDIA GPU/driver"]
fn real_mp4_demuxes_and_sizes_a_real_hardware_decoder() {
    let dir = std::env::temp_dir().join(format!("sovmr-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_fixture(&dir);
    let file = std::fs::read(&path).unwrap();

    // 1. Sovereign demux: no external process touches these bytes.
    let track = find_video_track(&file).expect("video track");
    assert_eq!(track.codec, VideoCodec::H264);
    assert_eq!((track.width, track.height), (640, 360));
    assert_eq!(track.sample_count(), 60, "2 s at 30 fps");

    // 2. Every sample resolves in-file; the first is a keyframe.
    let samples: Vec<_> = track
        .samples(&file)
        .collect::<Result<Vec<_>, _>>()
        .expect("all samples resolve");
    assert_eq!(samples.len(), 60);
    assert!(samples[0].keyframe);

    // 3. Annex B conversion injects parameter sets on the keyframe.
    let mut annex_b = Vec::new();
    track.write_frame(&samples[0], &mut annex_b).unwrap();
    assert_eq!(&annex_b[..4], &START_CODE);
    assert_eq!(
        annex_b[4] & NAL_TYPE_MASK,
        NAL_SPS,
        "keyframe stream must open with the SPS"
    );

    // 4. The demuxed geometry sizes a REAL hardware decoder.
    let cuda = sovmr_cuda::Cuda::load().expect("CUDA driver");
    let ctx = cuda.create_context(0).expect("GPU 0");
    let nvdec = NvDec::load().expect("NVDEC runtime");
    let cfg = DecoderConfig {
        codec: match track.codec {
            VideoCodec::H264 => DecodeCodec::H264,
            VideoCodec::Hevc => DecodeCodec::Hevc,
            VideoCodec::Av1 => unreachable!("fixture is H.264"),
        },
        width: u32::from(track.width),
        height: u32::from(track.height),
        bit_depth_minus8: 0,
        decode_surfaces: 16,
    };
    let decoder = nvdec
        .create_decoder(&ctx, &cfg)
        .expect("decoder from demuxed facts");
    assert!(!decoder.raw().is_null());

    std::fs::remove_dir_all(&dir).ok();
}
