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

//! Sovereign verification, distrust turned on our own output: the
//! muxer's binary output is parsed back with **our own demuxer** — no
//! ffprobe in the loop. Asserted invariants:
//!
//! * fast start: `moov` appears strictly before `mdat`;
//! * `stco` offsets point at valid AV1 OBU headers / audio packet bytes
//!   (proven by demuxing every sample back bit-exactly);
//! * DTS/durations survive the round trip;
//! * audio interleaves with video inside `mdat` in feed order;
//! * the `mp4a` sample entry survives bit-exactly;
//! * the spool file never outlives the muxer (finalized or dropped).

use std::path::PathBuf;

use sovmr_mp4::demux::{find_audio_tracks, find_video_track, VideoCodec};
use sovmr_mp4::hevc::{HevcParameterSets, NAL_PPS, NAL_SPS, NAL_VPS};
use sovmr_mp4::mux::{
    extract_sequence_header, strip_temporal_delimiter, AudioMuxParams, Mp4Muxer, MuxVideoCodec,
    VideoMuxParams,
};
use sovmr_mp4::{boxes, file_kind};

/// Video track timescale used by the fixtures (30 fps family).
const V_TIMESCALE: u32 = 15_360;
/// Per-frame duration in that timescale.
const V_DELTA: u32 = 512;
/// AAC frame duration at the audio timescale (samples per packet).
const A_DELTA: u32 = 1024;
/// Audio timescale = sample rate.
const A_TIMESCALE: u32 = 48_000;

/// The first carryable audio track, for the tests that only care that
/// one exists. Written out here rather than kept as a shipped API: the
/// engine wants every track, and a convenience that returned one was
/// exactly how five of six used to go missing.
fn first_audio(file: &[u8]) -> Option<sovmr_mp4::demux::AudioTrack<'_>> {
    find_audio_tracks(file).ok()?.tracks.into_iter().next()
}

/// Serialize one AV1 OBU with the has_size flag and a LEB128 length.
fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() < 128,
        "single-byte LEB128 keeps fixtures simple"
    );
    let mut out = vec![(obu_type << 3) | 0x02, payload.len() as u8];
    out.extend_from_slice(payload);
    out
}

/// Sequence-header OBU (type 1): payload high bits carry seq_profile 0.
fn seq_header_obu() -> Vec<u8> {
    obu(1, &[0x00, 0x00, 0x00, 0x4A])
}

/// Frame OBU (type 6) with recognizable payload bytes.
fn frame_obu(tag: u8) -> Vec<u8> {
    obu(6, &[tag, tag ^ 0xFF, 0x55, tag])
}

/// The `mp4a` sample-entry payload fed in (stereo, 48 kHz, fake esds).
fn mp4a_entry_payload() -> Vec<u8> {
    let mut e = vec![0u8; 28];
    e[6..8].copy_from_slice(&1u16.to_be_bytes());
    e[16..18].copy_from_slice(&2u16.to_be_bytes());
    e[24..28].copy_from_slice(&(A_TIMESCALE << 16).to_be_bytes());
    e.extend_from_slice(&[0, 0, 0, 12, b'e', b's', b'd', b's', 3, 0x19, 0, 2]);
    e
}

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Mux 3 video frames + 2 audio packets interleaved; return the file.
fn mux_fixture(stem: &str) -> Vec<u8> {
    let spool = tmp(&format!("{stem}.spool"));
    let out = tmp(&format!("{stem}.mp4"));
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 1280,
            height: 720,
            timescale: V_TIMESCALE,
            colour: None,
        },
        vec![AudioMuxParams::anonymous(A_TIMESCALE, mp4a_entry_payload())],
    )
    .expect("spool created");

    // Keyframe sample = sequence header + frame (as NVENC emits after
    // the temporal delimiter is stripped).
    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(
        extract_sequence_header(&key).expect("seq header findable in the keyframe"),
    );

    muxer.add_video_sample(&key, V_DELTA, true, 0).unwrap();
    muxer.add_audio_sample(0, &[0x11; 12], A_DELTA).unwrap();
    muxer
        .add_video_sample(&frame_obu(0xB2), V_DELTA, false, u64::from(V_DELTA))
        .unwrap();
    muxer.add_audio_sample(0, &[0x22; 9], A_DELTA).unwrap();
    muxer
        .add_video_sample(&frame_obu(0xC3), V_DELTA, false, 2 * u64::from(V_DELTA))
        .unwrap();

    let written = muxer.finalize(&out).expect("finalize");
    let bytes = std::fs::read(&out).expect("output exists");
    assert_eq!(written, bytes.len() as u64, "reported size == real size");
    assert!(!spool.exists(), "spool removed after finalize");
    let _ = std::fs::remove_file(&out);
    bytes
}

#[test]
fn fast_start_moov_precedes_mdat() {
    let file = mux_fixture("faststart");
    let kinds: Vec<[u8; 4]> = boxes(&file).map(|b| b.unwrap().kind).collect();
    assert_eq!(kinds, vec![*b"ftyp", *b"moov", *b"mdat"]);
    assert_eq!(file_kind(&file), Some(*b"isom"));
}

#[test]
fn own_demuxer_reads_back_every_video_sample_bit_exact() {
    let file = mux_fixture("videoback");
    let track = find_video_track(&file).expect("our own file parses");
    assert_eq!(track.codec, VideoCodec::Av1);
    assert_eq!((track.width, track.height), (1280, 720));
    assert_eq!(track.timescale, V_TIMESCALE);
    assert_eq!(track.sample_count(), 3);

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    let expected: [&[u8]; 3] = [&key, &frame_obu(0xB2), &frame_obu(0xC3)];

    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    for (i, (sample, want)) in samples.iter().zip(expected).enumerate() {
        // stco -> stsz walk lands exactly on the OBU bytes we fed in.
        assert_eq!(sample.data, want, "sample {i} bit-exact");
        // And the first byte is a *valid OBU header*: forbidden bit
        // clear, known type (1 = sequence header, 6 = frame).
        let header = sample.data[0];
        assert_eq!(header & 0x80, 0, "forbidden bit");
        assert!(matches!((header >> 3) & 0xF, 1 | 6), "OBU type");
        assert_eq!(sample.dts, i as u64 * u64::from(V_DELTA));
        assert_eq!(sample.pts, sample.dts as i64, "no ctts -> pts == dts");
        assert_eq!(sample.duration, V_DELTA);
    }
    assert!(samples[0].keyframe);
    assert!(!samples[1].keyframe && !samples[2].keyframe);
}

#[test]
fn own_demuxer_reads_back_audio_and_the_entry_is_bit_exact() {
    let file = mux_fixture("audioback");
    let audio = first_audio(&file).expect("audio track present");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, A_TIMESCALE);
    assert_eq!(audio.timescale, A_TIMESCALE);
    assert_eq!(audio.entry_payload, &mp4a_entry_payload()[..]);
    assert_eq!(audio.sample_count(), 2);

    let samples: Vec<_> = audio.samples(&file).map(|s| s.unwrap()).collect();
    assert_eq!(samples[0].data, &[0x11; 12]);
    assert_eq!(samples[1].data, &[0x22; 9]);
    assert_eq!(samples[0].dts, 0);
    assert_eq!(samples[1].dts, u64::from(A_DELTA));
    assert!(samples.iter().all(|s| s.keyframe), "audio is all sync");
}

#[test]
fn mdat_interleaves_tracks_in_feed_order() {
    let file = mux_fixture("interleave");
    let video = find_video_track(&file).unwrap();
    let audio = first_audio(&file).unwrap();
    let base = file.as_ptr() as usize;
    let v_at: Vec<usize> = video
        .samples(&file)
        .map(|s| s.unwrap().data.as_ptr() as usize - base)
        .collect();
    let a_at: Vec<usize> = audio
        .samples(&file)
        .map(|s| s.unwrap().data.as_ptr() as usize - base)
        .collect();
    // Feed order was v a v a v: offsets must interleave strictly.
    assert!(v_at[0] < a_at[0]);
    assert!(a_at[0] < v_at[1]);
    assert!(v_at[1] < a_at[1]);
    assert!(a_at[1] < v_at[2]);
}

#[test]
fn video_only_files_mux_without_an_audio_track() {
    let spool = tmp("videoonly.spool");
    let out = tmp("videoonly.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 640,
            height: 360,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0x77));
    muxer.set_sequence_header(&seq_header_obu());
    muxer.add_video_sample(&key, V_DELTA, true, 0).unwrap();
    muxer.finalize(&out).unwrap();

    let file = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    assert!(find_audio_tracks(&file).unwrap().is_silent());
    let track = find_video_track(&file).unwrap();
    assert_eq!(track.sample_count(), 1);
    assert_eq!(track.samples(&file).next().unwrap().unwrap().data, &key[..]);
}

#[test]
fn stss_maps_exactly_the_keyframes_one_based() {
    // Feed k P P k P: the sync table must read [1, 4] — 1-based sample
    // numbers per ISO 14496-12 §8.6.2 — and survive our own demuxer.
    let spool = tmp("stss.spool");
    let out = tmp("stss.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 64,
            height: 64,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    muxer.set_sequence_header(&seq_header_obu());
    for (i, key) in [true, false, false, true, false].into_iter().enumerate() {
        muxer
            .add_video_sample(
                &frame_obu(i as u8),
                V_DELTA,
                key,
                i as u64 * u64::from(V_DELTA),
            )
            .unwrap();
    }
    muxer.finalize(&out).unwrap();
    let file = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);

    // Byte-level: locate stss and check the exact table.
    let at = file
        .windows(4)
        .position(|w| w == b"stss")
        .expect("stss box present when not every sample is sync");
    let body = &file[at + 4 + 4..]; // fourcc + version/flags
    let count = u32::from_be_bytes(body[0..4].try_into().unwrap());
    assert_eq!(count, 2, "two sync samples");
    let first = u32::from_be_bytes(body[4..8].try_into().unwrap());
    let second = u32::from_be_bytes(body[8..12].try_into().unwrap());
    assert_eq!((first, second), (1, 4), "1-based sample numbers");

    // Semantic: our demuxer agrees.
    let track = find_video_track(&file).unwrap();
    let keys: Vec<bool> = track.samples(&file).map(|s| s.unwrap().keyframe).collect();
    assert_eq!(keys, vec![true, false, false, true, false]);
}

#[test]
fn dropping_an_unfinalized_muxer_rolls_the_spool_back() {
    let spool = tmp("rollback.spool");
    {
        let mut muxer = Mp4Muxer::create(
            &spool,
            VideoMuxParams {
                codec: MuxVideoCodec::Av1,
                width: 64,
                height: 64,
                timescale: V_TIMESCALE,
                colour: None,
            },
            Vec::new(),
        )
        .unwrap();
        muxer
            .add_video_sample(&frame_obu(0x01), V_DELTA, true, 0)
            .unwrap();
        assert!(spool.exists(), "spool live while muxing");
    }
    assert!(!spool.exists(), "drop without finalize removes the spool");
}

#[test]
fn finalize_without_a_sequence_header_is_a_typed_error() {
    let spool = tmp("noseq.spool");
    let out = tmp("noseq.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 64,
            height: 64,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    muxer
        .add_video_sample(&frame_obu(0x01), V_DELTA, true, 0)
        .unwrap();
    let err = muxer.finalize(&out).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(!out.exists(), "no partial output on failure");
    assert!(!spool.exists(), "spool cleaned up on failed finalize");
}

// ---- HEVC (`hvc1`) round trip ----------------------------------------------

/// MSB-first bit writer for handcrafting a minimal SPS fixture.
struct BitWriter {
    bytes: Vec<u8>,
    used: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            used: 0,
        }
    }
    fn u(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            if self.used.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = ((value >> i) & 1) as u8;
            *self.bytes.last_mut().unwrap() |= bit << (7 - (self.used % 8));
            self.used += 1;
        }
    }
    fn ue(&mut self, value: u32) {
        let coded = value + 1;
        let bits = 32 - coded.leading_zeros();
        self.u(0, bits - 1);
        self.u(coded, bits);
    }
}

/// A minimal Main-profile 4:2:0 8-bit SPS (only the fields the hvcC
/// builder reads, with realistic values).
fn synthetic_hevc_sps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(0, 4); // sps_video_parameter_set_id
    w.u(0, 3); // sps_max_sub_layers_minus1
    w.u(1, 1); // sps_temporal_id_nesting_flag
    w.u(0, 2); // general_profile_space
    w.u(0, 1); // general_tier_flag
    w.u(1, 5); // general_profile_idc = Main
    w.u(0x6000_0000, 32); // compatibility flags
    w.u(0x90, 8); // progressive + frame-only constraints
    for _ in 0..5 {
        w.u(0, 8); // remaining constraint bits
    }
    w.u(93, 8); // general_level_idc = 3.1
    w.ue(0); // sps_seq_parameter_set_id
    w.ue(1); // chroma_format_idc = 4:2:0
    w.ue(1280); // pic_width_in_luma_samples
    w.ue(720); // pic_height_in_luma_samples
    w.u(0, 1); // conformance_window_flag
    w.ue(0); // bit_depth_luma_minus8
    w.ue(0); // bit_depth_chroma_minus8
    w.u(1, 1); // rbsp_stop_one_bit
    let mut nal = vec![NAL_SPS << 1, 0x01];
    nal.extend(w.bytes);
    nal
}

fn hevc_parameter_sets() -> HevcParameterSets {
    HevcParameterSets {
        vps: vec![NAL_VPS << 1, 0x01, 0x0C, 0x11],
        sps: synthetic_hevc_sps(),
        pps: vec![NAL_PPS << 1, 0x01, 0xC0],
    }
}

/// One length-prefixed `hvc1` sample carrying a single slice NAL.
fn hevc_sample(tag: u8) -> Vec<u8> {
    let nal = [19u8 << 1, 0x01, tag, tag ^ 0xFF, 0x55]; // IDR_W_RADL shape
    let mut s = (nal.len() as u32).to_be_bytes().to_vec();
    s.extend_from_slice(&nal);
    s
}

#[test]
fn hvc1_output_round_trips_through_our_own_demuxer() {
    let spool = tmp("hevcback.spool");
    let out = tmp("hevcback.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Hevc,
            width: 1280,
            height: 720,
            timescale: V_TIMESCALE,
            colour: None,
        },
        vec![AudioMuxParams::anonymous(A_TIMESCALE, mp4a_entry_payload())],
    )
    .unwrap();
    muxer.set_hevc_parameter_sets(hevc_parameter_sets());
    muxer
        .add_video_sample(&hevc_sample(0xA1), V_DELTA, true, 0)
        .unwrap();
    muxer.add_audio_sample(0, &[0x33; 7], A_DELTA).unwrap();
    muxer
        .add_video_sample(&hevc_sample(0xB2), V_DELTA, false, u64::from(V_DELTA))
        .unwrap();
    muxer.finalize(&out).unwrap();

    let file = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);

    // Fast start holds for HEVC too.
    let kinds: Vec<[u8; 4]> = boxes(&file).map(|b| b.unwrap().kind).collect();
    assert_eq!(kinds, vec![*b"ftyp", *b"moov", *b"mdat"]);

    // Our demuxer recognizes the track and the hvcC record it just wrote.
    let track = find_video_track(&file).expect("own hvc1 parses");
    assert_eq!(track.codec, VideoCodec::Hevc);
    assert_eq!((track.width, track.height), (1280, 720));
    assert_eq!(track.nal_length_size, 4, "hvcC lengthSizeMinusOne = 3");
    assert_eq!(track.sample_count(), 2);

    // The parameter sets come back bit-exact from the hvcC arrays.
    let ps = hevc_parameter_sets();
    let sets: Vec<&[u8]> = track.parameter_sets().map(|p| p.unwrap()).collect();
    assert_eq!(sets, vec![&ps.vps[..], &ps.sps[..], &ps.pps[..]]);

    // Samples land bit-exactly, and Annex B conversion (the decoder-side
    // consumer) walks the 4-byte length prefixes without complaint.
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    assert_eq!(samples[0].data, &hevc_sample(0xA1)[..]);
    assert_eq!(samples[1].data, &hevc_sample(0xB2)[..]);
    assert!(samples[0].keyframe && !samples[1].keyframe);
    let mut annex_b = Vec::new();
    track.write_frame(&samples[0], &mut annex_b).unwrap();
    assert_eq!(&annex_b[..4], &[0, 0, 0, 1], "starts with a start code");

    // Audio passthrough is codec-independent.
    let audio = first_audio(&file).unwrap();
    assert_eq!(audio.sample_count(), 1);
    assert_eq!(
        audio.samples(&file).next().unwrap().unwrap().data,
        &[0x33; 7]
    );
    assert!(!spool.exists());
}

#[test]
fn hevc_finalize_without_parameter_sets_is_a_typed_error() {
    let spool = tmp("nops.spool");
    let out = tmp("nops.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Hevc,
            width: 64,
            height: 64,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    muxer
        .add_video_sample(&hevc_sample(0x01), V_DELTA, true, 0)
        .unwrap();
    let err = muxer.finalize(&out).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        err.to_string().contains("HEVC parameter sets"),
        "error names the missing config: {err}"
    );
    assert!(!out.exists() && !spool.exists());
}

#[test]
fn codec_config_mismatch_is_refused() {
    let spool = tmp("mismatch.spool");
    let out = tmp("mismatch.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Hevc,
            width: 64,
            height: 64,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    // An AV1 sequence header on an HEVC track is a caller bug, not a
    // silently corrupt file.
    muxer.set_sequence_header(&seq_header_obu());
    muxer
        .add_video_sample(&hevc_sample(0x01), V_DELTA, true, 0)
        .unwrap();
    let err = muxer.finalize(&out).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(!out.exists() && !spool.exists());
}

#[test]
fn b_frame_reordering_round_trips_through_ctts() {
    // Decode order I P B (as an encoder emits): the P presented last,
    // the B between. cts:  I=0, P=1024, B=512; dts by duration: 0/512/1024.
    let spool = tmp("ctts.spool");
    let out = tmp("ctts.mp4");
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 64,
            height: 64,
            timescale: V_TIMESCALE,
            colour: None,
        },
        Vec::new(),
    )
    .unwrap();
    muxer.set_sequence_header(&seq_header_obu());
    let d = u64::from(V_DELTA);
    muxer
        .add_video_sample(&frame_obu(0x11), V_DELTA, true, 0)
        .unwrap();
    muxer
        .add_video_sample(&frame_obu(0x22), V_DELTA, false, 2 * d)
        .unwrap();
    muxer
        .add_video_sample(&frame_obu(0x33), V_DELTA, false, d)
        .unwrap();
    muxer.finalize(&out).unwrap();
    let file = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);

    // ctts must exist; our own demuxer reconstructs the presentation
    // order. Version-0 offsets are unsigned, so the muxer normalizes
    // them (min offset = 0): the whole timeline gains one frame of
    // uniform composition delay — exactly like every muxer that avoids
    // edit lists. Order and spacing are what players care about.
    assert!(
        file.windows(4).any(|w| w == b"ctts"),
        "reordering must produce a ctts box"
    );
    assert!(
        file.windows(4).any(|w| w == b"elst"),
        "the composition delay must be cancelled by an edit list"
    );
    let track = find_video_track(&file).unwrap();
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    let dts: Vec<u64> = samples.iter().map(|s| s.dts).collect();
    let pts: Vec<i64> = samples.iter().map(|s| s.pts).collect();
    assert_eq!(dts, vec![0, d, 2 * d], "decode times stay monotone");
    let d = d as i64;
    assert_eq!(
        pts,
        vec![d, 3 * d, 2 * d],
        "presentation order recovered (cts + one frame of composition delay)"
    );
}

#[test]
fn linear_streams_omit_the_ctts_box() {
    let file = mux_fixture("noctts");
    assert!(
        !file.windows(4).any(|w| w == b"ctts"),
        "cts == dts everywhere: writing ctts would be dead weight"
    );
    assert!(
        !file.windows(4).any(|w| w == b"elst"),
        "no reordering: no edit list either"
    );
}

#[test]
fn temporal_delimiter_stripping_and_seq_extraction() {
    let td = obu(2, &[]);
    let mut packet = td.clone();
    packet.extend_from_slice(&seq_header_obu());
    packet.extend_from_slice(&frame_obu(0x0F));

    let stripped = strip_temporal_delimiter(&packet);
    assert_eq!(stripped.len(), packet.len() - td.len());
    assert_eq!(stripped[0] >> 3 & 0xF, 1, "now starts at the seq header");
    assert_eq!(
        extract_sequence_header(&packet).expect("seq present"),
        &seq_header_obu()[..],
    );
    // A delta packet has no sequence header.
    assert!(extract_sequence_header(&frame_obu(0x0F)).is_none());
}

// ---- long-form output: 64-bit boxes -----------------------------------------

/// Twelve hours is the material this milestone exists for. At a 90 kHz
/// media timescale that is 3.9e9 units — already most of a `u32` — and
/// the old muxer cast the field with `as u32`, so a longer master would
/// have wrapped and produced a file claiming to be minutes long.
const LONG_TIMESCALE: u32 = 90_000;
/// Two samples of this duration put the media clock past `u32::MAX`
/// (8.0e9 units ≈ 24.7 h at 90 kHz), forcing the 64-bit header form.
const HUGE_SAMPLE_DURATION: u32 = 4_000_000_000;

/// Mux two samples with the given track timescale and per-sample
/// duration, returning the finished file.
fn mux_with_clock(stem: &str, timescale: u32, sample_duration: u32) -> Vec<u8> {
    let spool = tmp(&format!("{stem}.spool"));
    let out = tmp(&format!("{stem}.mp4"));
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 1280,
            height: 720,
            timescale,
            colour: None,
        },
        Vec::new(),
    )
    .expect("spool created");

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(extract_sequence_header(&key).expect("seq header"));
    muxer
        .add_video_sample(&key, sample_duration, true, 0)
        .unwrap();
    muxer
        .add_video_sample(
            &frame_obu(0xB2),
            sample_duration,
            false,
            u64::from(sample_duration),
        )
        .unwrap();
    muxer.finalize(&out).expect("finalize");
    let bytes = std::fs::read(&out).expect("output exists");
    let _ = std::fs::remove_file(&out);
    bytes
}

/// The version byte of the first box with this fourcc anywhere in the file.
fn box_version(file: &[u8], kind: &[u8; 4]) -> Option<u8> {
    fn walk(data: &[u8], kind: &[u8; 4]) -> Option<u8> {
        for b in boxes(data) {
            let b = b.ok()?;
            if b.kind == *kind {
                return b.payload.first().copied();
            }
            // Only descend into the container boxes on the path to the
            // headers under test.
            if matches!(&b.kind, b"moov" | b"trak" | b"mdia" | b"edts") {
                if let Some(v) = walk(b.payload, kind) {
                    return Some(v);
                }
            }
        }
        None
    }
    walk(file, kind)
}

#[test]
fn a_media_clock_past_32_bits_is_written_as_a_64_bit_mdhd() {
    let file = mux_with_clock("longclock", LONG_TIMESCALE, HUGE_SAMPLE_DURATION);
    assert_eq!(
        box_version(&file, b"mdhd"),
        Some(1),
        "the media duration no longer fits in 32 bits, so mdhd must be v1"
    );

    // The value must survive the round trip, which is the whole point:
    // wrapping silently was the failure being fixed.
    let track = find_video_track(&file).expect("own demuxer reads own file");
    let expected_units = 2 * u64::from(HUGE_SAMPLE_DURATION);
    let duration = track.duration().expect("mdhd duration present");
    let expected_secs = expected_units as f64 / f64::from(LONG_TIMESCALE);
    assert!(
        (duration.as_secs_f64() - expected_secs).abs() < 1.0,
        "expected ~{expected_secs} s, got {duration:?}"
    );
    assert!(
        duration.as_secs() > 12 * 3600,
        "this fixture must exceed twelve hours to be worth anything"
    );
}

#[test]
fn a_movie_clock_that_fits_keeps_the_32_bit_boxes() {
    // Regression guard in the other direction: ordinary files must not
    // silently gain 64-bit headers, which older players refuse.
    let file = mux_fixture("narrowclock");
    assert_eq!(box_version(&file, b"mvhd"), Some(0));
    assert_eq!(box_version(&file, b"tkhd"), Some(0));
    assert_eq!(box_version(&file, b"mdhd"), Some(0));
    let kinds: Vec<[u8; 4]> = boxes(&file).map(|b| b.unwrap().kind).collect();
    assert_eq!(kinds, vec![*b"ftyp", *b"moov", *b"mdat"]);
}

#[test]
fn a_movie_clock_past_32_bits_widens_mvhd_and_tkhd_too() {
    // The movie clock is in milliseconds, so it only overflows for
    // absurd durations — but "absurd" is exactly what negative trust
    // says to handle, and the fields are cast the same way.
    let file = mux_with_clock("longmovie", 1, HUGE_SAMPLE_DURATION);
    assert_eq!(box_version(&file, b"mvhd"), Some(1));
    assert_eq!(box_version(&file, b"tkhd"), Some(1));
    assert_eq!(box_version(&file, b"mdhd"), Some(1));
}

/// Container boxes worth descending into when looking for a table.
const CONTAINERS: [&[u8; 4]; 6] = [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"];

/// Whether a box with this fourcc exists anywhere in the tree.
fn contains_box(data: &[u8], kind: &[u8; 4]) -> bool {
    for b in boxes(data) {
        let Ok(b) = b else { return false };
        if b.kind == *kind {
            return true;
        }
        if CONTAINERS.contains(&&b.kind) && contains_box(b.payload, kind) {
            return true;
        }
    }
    false
}

/// Sample payload size for the oversize test: big enough that four
/// thousand of them cross 4 GiB, small enough to stay one write each.
const BIG_SAMPLE_BYTES: usize = 2 * 1024 * 1024;
/// Pairs of (video, audio) samples to write. 2200 x 2 x 2 MiB = 8.8 GiB
/// of payload, comfortably past the 4 GiB wall in both the `mdat` size
/// field and the chunk-offset table.
const BIG_SAMPLE_PAIRS: usize = 1100;

#[test]
#[ignore = "writes ~4.6 GiB to disk (twice: spool then output)"]
fn output_past_four_gibibytes_uses_co64_and_a_64_bit_mdat() {
    // THE long-form gate. Before this, `finalize` returned
    // `Err("file exceeds 4 GiB")` — a twelve-hour master could not be
    // written at all, no matter how well it encoded.
    let spool = tmp("oversize.spool");
    let out = tmp("oversize.mp4");
    let _ = std::fs::remove_file(&out);

    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 1280,
            height: 720,
            timescale: V_TIMESCALE,
            colour: None,
        },
        vec![AudioMuxParams::anonymous(A_TIMESCALE, mp4a_entry_payload())],
    )
    .expect("spool created");

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(extract_sequence_header(&key).expect("seq header"));
    muxer.add_video_sample(&key, V_DELTA, true, 0).unwrap();

    // Alternate tracks so every sample starts a new chunk: that is what
    // spreads chunk offsets across the whole file and pushes the last
    // one past what a 32-bit `stco` entry can hold.
    let filler = vec![0x5Au8; BIG_SAMPLE_BYTES];
    for i in 0..BIG_SAMPLE_PAIRS {
        muxer.add_audio_sample(0, &filler, A_DELTA).unwrap();
        let pts = u64::from(V_DELTA) * (i as u64 + 1);
        muxer
            .add_video_sample(&filler, V_DELTA, false, pts)
            .unwrap();
    }

    let written = muxer.finalize(&out).expect("a 4 GiB+ file must finalize");
    assert!(
        written > u64::from(u32::MAX),
        "fixture must actually cross the 4 GiB wall, wrote {written}"
    );

    let bytes = std::fs::read(&out).expect("output exists");
    assert_eq!(written as usize, bytes.len(), "reported size == real size");

    // `mdat` must use the largesize form: 32-bit size field == 1, with
    // the real size in the u64 that follows the fourcc.
    let mdat_at = bytes
        .windows(4)
        .position(|w| w == b"mdat")
        .expect("mdat present");
    let size32 = u32::from_be_bytes(bytes[mdat_at - 4..mdat_at].try_into().unwrap());
    assert_eq!(size32, 1, "oversize mdat must signal largesize");
    let large = u64::from_be_bytes(bytes[mdat_at + 4..mdat_at + 12].try_into().unwrap());
    assert_eq!(
        large as usize,
        bytes.len() - (mdat_at - 4),
        "largesize must cover the whole mdat box"
    );

    assert!(contains_box(&bytes, b"co64"), "offsets must widen to co64");
    assert!(
        !contains_box(&bytes, b"stco"),
        "a widened file must not also carry the 32-bit table"
    );

    // Sovereign verification: our own demuxer resolves the wide offsets
    // and lands on the right bytes at the far end of a 4 GiB+ file.
    let track = find_video_track(&bytes).expect("own demuxer reads own file");
    let samples: Vec<_> = track.samples(&bytes).map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), BIG_SAMPLE_PAIRS + 1);
    let last = samples.last().unwrap();
    assert!(
        last.data.len() == BIG_SAMPLE_BYTES && last.data.iter().all(|b| *b == 0x5A),
        "the final sample past 4 GiB must read back intact"
    );

    let audio = first_audio(&bytes).expect("audio");
    assert_eq!(audio.sample_count() as usize, BIG_SAMPLE_PAIRS);

    let _ = std::fs::remove_file(&out);
}

// ---- twelve-hour scale ------------------------------------------------------

/// The milestone's stated duration, in seconds.
const TWELVE_HOURS_SECS: u64 = 12 * 60 * 60;
/// Video frames per second of media time in the long-form fixture.
const LONG_FPS: u64 = 30;
/// Audio tracks carried alongside the video leg.
const LONG_AUDIO_TRACKS: usize = 6;
/// Payload sizes cycled through by the long-form fixture. Real AAC and
/// AV1 packets vary in size, which matters here: a fixture of identical
/// samples would collapse into `stsz`'s uniform form and quietly hide
/// the one table that genuinely cannot be run-length encoded.
const LONG_SAMPLE_SIZES: [usize; 4] = [3, 5, 4, 6];
/// Keyframe cadence in seconds, matching the encoder's 10 s GOP.
const LONG_GOP_SECS: u64 = 10;
/// Ceiling on peak resident memory for the whole twelve-hour mux,
/// sampled after `finalize` and before the output is read back.
///
/// Not a performance nicety: the muxer holds per-sample metadata for
/// every track until `finalize`, so this is the number that decides
/// whether a twelve-hour job dies of OOM on an ordinary machine.
///
/// Measured baseline is 489 MiB for 13.2 M samples — about 39 B per
/// sample, which is the 24-byte record plus `Vec` growth slack. The
/// budget sits above that with enough room for allocator variance but
/// not enough to absorb a doubling of the per-sample cost, which is
/// exactly the regression worth failing on.
const LONG_FORM_PEAK_RSS_LIMIT: u64 = 768 * 1024 * 1024;

/// Peak resident set size of this process, from `VmHWM` — the kernel's
/// own high-water mark, so it does not matter that the peak happened
/// midway through the run.
fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("procfs available");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .parse()
                .expect("VmHWM is a number of kB");
            return kb * 1024;
        }
    }
    panic!("VmHWM missing from /proc/self/status");
}

#[test]
#[ignore = "builds sample tables for a 12 h, 7-track movie (~13 M samples)"]
fn a_twelve_hour_seven_track_movie_builds_compact_tables() {
    // THE long-form gate for table scale. A twelve-hour six-language
    // master is the material this milestone exists for, and every
    // structure that grows per sample gets its one honest test here:
    // 1.3 M video frames and 6 x ~2 M audio packets in one file.
    //
    // What is being defended is that the *tables* stay run-length
    // collapsed. A muxer that emits one `stts` entry per sample still
    // produces a valid file — it just prepends a hundred megabytes of
    // `moov` that every player must parse before the first frame.
    let spool = tmp("twelve_hour.spool");
    let out = tmp("twelve_hour.mp4");
    let _ = std::fs::remove_file(&out);

    let audio: Vec<AudioMuxParams> = (0..LONG_AUDIO_TRACKS)
        .map(|_| AudioMuxParams::anonymous(A_TIMESCALE, mp4a_entry_payload()))
        .collect();
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 1920,
            height: 1080,
            timescale: V_TIMESCALE,
            colour: None,
        },
        audio,
    )
    .expect("spool created");

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(extract_sequence_header(&key).expect("seq header"));

    // Interleave in one-second slabs, exactly as the engine does: chunk
    // granularity is what keeps `stsc`/`stco` from exploding.
    let filler = vec![0xC5u8; *LONG_SAMPLE_SIZES.iter().max().unwrap()];
    let audio_pkts_per_sec = u64::from(A_TIMESCALE) / u64::from(A_DELTA);
    let mut frame = 0u64;
    let mut a_pkt = 0u64;
    for _ in 0..TWELVE_HOURS_SECS {
        for _ in 0..LONG_FPS {
            let pts = u64::from(V_DELTA) * frame;
            let keyframe = frame.is_multiple_of(LONG_FPS * LONG_GOP_SECS);
            let n = LONG_SAMPLE_SIZES[frame as usize % LONG_SAMPLE_SIZES.len()];
            let data: &[u8] = if frame == 0 { &key } else { &filler[..n] };
            muxer
                .add_video_sample(data, V_DELTA, keyframe, pts)
                .expect("video sample");
            frame += 1;
        }
        for track in 0..LONG_AUDIO_TRACKS {
            for i in 0..audio_pkts_per_sec {
                let n = LONG_SAMPLE_SIZES[(a_pkt + i) as usize % LONG_SAMPLE_SIZES.len()];
                muxer
                    .add_audio_sample(track, &filler[..n], A_DELTA)
                    .expect("audio sample");
            }
        }
        a_pkt += audio_pkts_per_sec;
    }

    let expected_frames = TWELVE_HOURS_SECS * LONG_FPS;
    let expected_pkts = a_pkt;
    let written = muxer
        .finalize(&out)
        .expect("a twelve-hour movie must finalize");

    let peak = peak_rss_bytes();
    let bytes = std::fs::read(&out).expect("output exists");
    assert_eq!(written as usize, bytes.len());

    let moov_len = boxes(&bytes)
        .map(|b| b.unwrap())
        .find(|b| b.kind == *b"moov")
        .expect("moov present")
        .payload
        .len() as u64;
    let total_samples = expected_frames + expected_pkts * LONG_AUDIO_TRACKS as u64;
    let stsz_len = box_total_len(&bytes, b"stsz");
    let stts_len = box_total_len(&bytes, b"stts");
    let stsc_len = box_total_len(&bytes, b"stsc");
    let stco_len = box_total_len(&bytes, b"stco") + box_total_len(&bytes, b"co64");

    // Report before asserting: when this gate trips, the numbers are the
    // diagnosis, and a panic that fires first hides them.
    println!(
        "TWELVE-HOUR MUX: {expected_frames} frames + {} audio packets \
         across {LONG_AUDIO_TRACKS} tracks\n  \
         file {written} B, moov {moov_len} B ({:.2} B/sample)\n  \
         stsz {stsz_len} B, stts {stts_len} B, stsc {stsc_len} B, stco/co64 {stco_len} B\n  \
         peak RSS {} MiB",
        expected_pkts * LONG_AUDIO_TRACKS as u64,
        moov_len as f64 / total_samples as f64,
        peak / (1024 * 1024)
    );

    // `stsz` is the one table ISOBMFF cannot run-length encode when
    // sample sizes vary: four bytes per sample is the format's floor,
    // not a defect. Everything else must collapse, so the whole `moov`
    // is held to a small margin above that floor. A muxer that stopped
    // collapsing `stts` would double this immediately.
    let stsz_floor = total_samples * 4;
    assert!(
        moov_len < stsz_floor + stsz_floor / 8,
        "moov {moov_len} B is more than 12.5% above the {stsz_floor} B \
         stsz floor — some table stopped collapsing"
    );

    // Durations are constant per track, so `stts` is one run per track
    // however long the movie is. This is the assertion that would catch
    // a regression to one entry per sample (~106 MB here).
    assert!(
        stts_len < 1024,
        "stts must stay run-length collapsed across 7 tracks, got {stts_len} B"
    );
    // Same for `stsc`: one-second slabs hold a constant sample count per
    // chunk, so each track needs a single entry.
    assert!(
        stsc_len < 1024,
        "stsc must stay run-length collapsed, got {stsc_len} B"
    );
    // `stco` is inherently one entry per chunk; one-second slabs mean
    // ~43 200 chunks per track, not one chunk per packet (~13 M).
    assert!(
        stco_len < 4 * 1024 * 1024,
        "chunk offsets exploded: {stco_len} B means interleaving lost its slab granularity"
    );

    assert!(
        peak < LONG_FORM_PEAK_RSS_LIMIT,
        "peak RSS {peak} exceeded the {LONG_FORM_PEAK_RSS_LIMIT} byte budget for a 12 h mux"
    );

    // Sovereign verification: our own demuxer reads back every track at
    // full length, including the media clock past 32 bits.
    let track = find_video_track(&bytes).expect("own demuxer reads own file");
    assert_eq!(u64::from(track.sample_count()), expected_frames);
    let duration = track.duration().expect("mdhd duration");
    assert!(
        (duration.as_secs_f64() - TWELVE_HOURS_SECS as f64).abs() < 1.0,
        "twelve hours of media time, got {duration:?}"
    );

    let out_audio = find_audio_tracks(&bytes).expect("audio demuxes");
    assert_eq!(out_audio.tracks.len(), LONG_AUDIO_TRACKS);
    for (i, t) in out_audio.tracks.iter().enumerate() {
        assert_eq!(
            u64::from(t.sample_count()),
            expected_pkts,
            "audio track {i} lost packets at twelve-hour scale"
        );
    }

    let _ = std::fs::remove_file(&out);
}

/// Total byte length of every box with this fourcc, header included.
///
/// A flat scan rather than a tree walk: these tables only ever appear
/// inside `stbl`, and the fourccs are four-byte-aligned box headers, so
/// summing occurrences is enough to size a table across all tracks.
fn box_total_len(bytes: &[u8], kind: &[u8; 4]) -> u64 {
    let mut total = 0u64;
    let mut i = 4usize;
    while i + 4 <= bytes.len() {
        if &bytes[i..i + 4] == kind {
            let size = u32::from_be_bytes(bytes[i - 4..i].try_into().unwrap());
            total += u64::from(size);
        }
        i += 1;
    }
    total
}

// ---- many audio tracks ------------------------------------------------------

/// Languages of the six-track fixture — the shape of a real
/// multi-language master.
const SIX_LANGUAGES: [[u8; 3]; 6] = [*b"eng", *b"rus", *b"deu", *b"fra", *b"jpn", *b"spa"];
/// Alternate group shared by all of them: they are alternatives to each
/// other, not six soundtracks meant to play at once.
const ALTERNATE_GROUP: u16 = 1;
/// Audio packets written per track.
const PACKETS_PER_TRACK: usize = 4;

#[test]
fn six_audio_tracks_in_means_six_audio_tracks_out() {
    // THE contract of this milestone. The muxer could previously hold
    // exactly one audio track, so even a perfect demuxer would have had
    // to throw five of six away.
    let spool = tmp("sixtrack.spool");
    let out = tmp("sixtrack.mp4");
    let _ = std::fs::remove_file(&out);

    let tracks: Vec<AudioMuxParams> = SIX_LANGUAGES
        .iter()
        .enumerate()
        .map(|(i, lang)| AudioMuxParams {
            timescale: A_TIMESCALE,
            entry_payload: mp4a_entry_payload(),
            language: *lang,
            alternate_group: ALTERNATE_GROUP,
            // Make one track disabled to prove the flag is carried and
            // not silently promoted.
            enabled: i != 3,
            name: Some(format!("Commentary {i}")),
        })
        .collect();

    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 1280,
            height: 720,
            timescale: V_TIMESCALE,
            colour: None,
        },
        tracks,
    )
    .expect("spool created");
    assert_eq!(muxer.audio_track_count(), SIX_LANGUAGES.len());

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(extract_sequence_header(&key).expect("seq header"));
    muxer.add_video_sample(&key, V_DELTA, true, 0).unwrap();

    // Feed a slab per track at a time — the interleaving granularity a
    // caller is expected to choose, and the reason each track's samples
    // land in a few chunks rather than thousands.
    for round in 0..PACKETS_PER_TRACK {
        for (ix, _) in SIX_LANGUAGES.iter().enumerate() {
            let payload = [(0x10 + ix) as u8; 8];
            muxer.add_audio_sample(ix, &payload, A_DELTA).unwrap();
        }
        let pts = u64::from(V_DELTA) * (round as u64 + 1);
        muxer
            .add_video_sample(&frame_obu(0xB0 + round as u8), V_DELTA, false, pts)
            .unwrap();
    }

    muxer.finalize(&out).expect("finalize");
    let bytes = std::fs::read(&out).expect("output exists");
    let _ = std::fs::remove_file(&out);

    // Sovereign verification: our own demuxer reads our own file.
    let found = find_audio_tracks(&bytes).expect("output parses");
    assert!(!found.truncated);
    assert!(found.unsupported.is_empty());
    assert_eq!(
        found.tracks.len(),
        SIX_LANGUAGES.len(),
        "every audio track must survive the mux"
    );

    for (ix, track) in found.tracks.iter().enumerate() {
        assert_eq!(
            track.meta.language.0, SIX_LANGUAGES[ix],
            "track {ix} language must be carried, not regenerated"
        );
        assert_eq!(track.meta.alternate_group, ALTERNATE_GROUP);
        assert_eq!(track.meta.name, Some(format!("Commentary {ix}")).as_deref());
        assert_eq!(
            track.meta.enabled,
            ix != 3,
            "track {ix} enabled flag must survive verbatim"
        );
        assert_eq!(track.sample_count() as usize, PACKETS_PER_TRACK);
        assert_eq!(track.channels, 2);
    }

    // Track ids must be unique and 1-based, or players cannot address
    // them at all.
    let mut ids: Vec<u32> = found.tracks.iter().map(|t| t.meta.track_id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), SIX_LANGUAGES.len(), "track ids must be unique");
    assert!(ids.iter().all(|id| *id > 0), "track ids are 1-based");

    // And the payloads must be the ones fed to each track, not another
    // track's bytes read through the wrong offset table.
    for (ix, track) in found.tracks.iter().enumerate() {
        for sample in track.samples(&bytes) {
            let sample = sample.expect("offsets stay inside the file");
            assert!(
                sample.data.iter().all(|b| *b == (0x10 + ix) as u8),
                "track {ix} read back another track's bytes: {:?}",
                &sample.data[..4.min(sample.data.len())]
            );
        }
    }

    let video = find_video_track(&bytes).expect("video survives too");
    assert_eq!(video.sample_count() as usize, PACKETS_PER_TRACK + 1);
}

// ---- stsz uniform form ------------------------------------------------------

/// Packets written by the `stsz` form tests.
const STSZ_PACKETS: usize = 64;
/// Payload size when every packet is the same length.
const STSZ_CONSTANT_BYTES: usize = 32;

/// Mux `STSZ_PACKETS` audio packets whose sizes come from `sizes`,
/// cycled, and return the finished file.
fn mux_audio_sizes(stem: &str, sizes: &[usize]) -> Vec<u8> {
    let spool = tmp(&format!("{stem}.spool"));
    let out = tmp(&format!("{stem}.mp4"));
    let _ = std::fs::remove_file(&out);
    let mut muxer = Mp4Muxer::create(
        &spool,
        VideoMuxParams {
            codec: MuxVideoCodec::Av1,
            width: 320,
            height: 240,
            timescale: V_TIMESCALE,
            colour: None,
        },
        vec![AudioMuxParams::anonymous(A_TIMESCALE, mp4a_entry_payload())],
    )
    .expect("spool created");

    let mut key = seq_header_obu();
    key.extend_from_slice(&frame_obu(0xA1));
    muxer.set_sequence_header(extract_sequence_header(&key).expect("seq header"));
    muxer.add_video_sample(&key, V_DELTA, true, 0).unwrap();

    let filler = vec![0x7Eu8; *sizes.iter().max().unwrap()];
    for i in 0..STSZ_PACKETS {
        let n = sizes[i % sizes.len()];
        muxer.add_audio_sample(0, &filler[..n], A_DELTA).unwrap();
    }
    muxer.finalize(&out).expect("finalize");
    let bytes = std::fs::read(&out).expect("output exists");
    let _ = std::fs::remove_file(&out);
    bytes
}

#[test]
fn constant_size_packets_collapse_into_the_uniform_stsz_form() {
    // ISOBMFF lets a track declare one size for every sample and omit the
    // table entirely. Constant-rate media — PCM above all, which the
    // sovereign audio decoder will produce — otherwise pays four bytes
    // per sample for a number that never changes.
    let bytes = mux_audio_sizes("stsz_uniform", &[STSZ_CONSTANT_BYTES]);
    let audio = first_audio(&bytes).expect("audio track");
    assert_eq!(audio.sample_count() as usize, STSZ_PACKETS);
    assert_eq!(
        audio.total_bytes() as usize,
        STSZ_PACKETS * STSZ_CONSTANT_BYTES,
        "the uniform form must still describe the same payload"
    );

    // The table itself must be gone, not merely redundant: header only.
    let stsz = box_total_len(&bytes, b"stsz");
    assert!(
        stsz < 4 * STSZ_PACKETS as u64,
        "uniform stsz must omit the per-sample table, got {stsz} B"
    );

    // Every packet must still read back at full length through the
    // demuxer's own resolution of the uniform form.
    let track = first_audio(&bytes).expect("audio track");
    let samples: Vec<_> = track.samples(&bytes).map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), STSZ_PACKETS);
    assert!(
        samples
            .iter()
            .all(|s| s.data.len() == STSZ_CONSTANT_BYTES && s.data.iter().all(|b| *b == 0x7E)),
        "uniform-form samples must resolve to the right bytes"
    );
}

#[test]
fn varying_packet_sizes_still_get_a_full_stsz_table() {
    // The other half of the contract, and the one that would break real
    // files: a muxer that assumed uniformity from the first packet would
    // corrupt every variable-rate track ever written. AAC is variable.
    let sizes = [16usize, 48, 24, 33];
    let bytes = mux_audio_sizes("stsz_varying", &sizes);
    let track = first_audio(&bytes).expect("audio track");
    assert_eq!(track.sample_count() as usize, STSZ_PACKETS);

    let expected: usize = (0..STSZ_PACKETS).map(|i| sizes[i % sizes.len()]).sum();
    assert_eq!(track.total_bytes() as usize, expected);

    let samples: Vec<_> = track.samples(&bytes).map(|s| s.unwrap()).collect();
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(
            s.data.len(),
            sizes[i % sizes.len()],
            "packet {i} came back the wrong length"
        );
        assert!(s.data.iter().all(|b| *b == 0x7E), "packet {i} payload");
    }
}
