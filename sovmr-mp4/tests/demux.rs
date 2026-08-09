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

//! Demux tests against a hand-assembled synthetic MP4 (no fixtures on
//! disk, no ffmpeg): two length-prefixed H.264 samples in one chunk,
//! full sample tables with timing (`mdhd`/`stts`/`ctts`), an `mp4a`
//! audio track, plus hostile mutations of each table.

use sovmr_mp4::demux::{
    find_audio_tracks, find_video_track, survey_extras, DemuxError, Language, VideoCodec,
    MAX_AUDIO_TRACKS,
};
use sovmr_mp4::{boxes, find_path};

const SPS: &[u8] = &[0x67, 0x64, 0x00, 0x1F];
const PPS: &[u8] = &[0x68, 0xEB, 0xE3];
const IDR_NAL: &[u8] = &[0x65, 0xAA, 0xBB, 0xCC];
const P_NAL: &[u8] = &[0x41, 0xDD];

/// Track timescale used by the synthetic file (90k-family, 30 fps).
const TIMESCALE: u32 = 15_360;
/// Per-frame decode delta at 30 fps in that timescale.
const FRAME_DELTA: u32 = 512;
/// Constant composition offset applied through `ctts`.
const CTTS_OFFSET: u32 = 512;

/// The first carryable audio track, for the tests that only care that
/// one exists. Written out here rather than kept as a shipped API: the
/// engine wants every track, and a convenience that returned one was
/// exactly how five of six used to go missing.
fn first_audio(file: &[u8]) -> Option<sovmr_mp4::demux::AudioTrack<'_>> {
    find_audio_tracks(file).ok()?.tracks.into_iter().next()
}

fn make_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(8 + payload.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0]; // version + flags
    payload.extend_from_slice(body);
    make_box(kind, &payload)
}

/// Version-0 `mdhd` with the given timescale and duration.
fn mdhd_box(timescale: u32, duration: u32) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes()); // creation
    body.extend_from_slice(&0u32.to_be_bytes()); // modification
    body.extend_from_slice(&timescale.to_be_bytes());
    body.extend_from_slice(&duration.to_be_bytes());
    body.extend_from_slice(&0x55C4u16.to_be_bytes()); // 'und'
    body.extend_from_slice(&0u16.to_be_bytes());
    full_box(b"mdhd", &body)
}

/// `stts` from (count, delta) runs.
fn stts_box(runs: &[(u32, u32)]) -> Vec<u8> {
    let mut body = (runs.len() as u32).to_be_bytes().to_vec();
    for (count, delta) in runs {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&delta.to_be_bytes());
    }
    full_box(b"stts", &body)
}

/// `ctts` from (count, offset) runs.
fn ctts_box(runs: &[(u32, u32)]) -> Vec<u8> {
    let mut body = (runs.len() as u32).to_be_bytes().to_vec();
    for (count, offset) in runs {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full_box(b"ctts", &body)
}

/// AVCDecoderConfigurationRecord with one SPS and one PPS, 4-byte NALs.
fn avcc() -> Vec<u8> {
    let mut c = vec![1, 0x64, 0x00, 0x1F, 0xFF, 0xE1];
    c.extend_from_slice(&(SPS.len() as u16).to_be_bytes());
    c.extend_from_slice(SPS);
    c.push(1); // one PPS
    c.extend_from_slice(&(PPS.len() as u16).to_be_bytes());
    c.extend_from_slice(PPS);
    c
}

/// avc1 VisualSampleEntry (width 640, height 360) wrapping avcC.
fn avc1_entry() -> Vec<u8> {
    let mut e = vec![0u8; 78];
    e[24..26].copy_from_slice(&640u16.to_be_bytes());
    e[26..28].copy_from_slice(&360u16.to_be_bytes());
    e.extend_from_slice(&make_box(b"avcC", &avcc()));
    make_box(b"avc1", &e)
}

/// `mp4a` AudioSampleEntry payload: stereo, 48 kHz, with a stand-in
/// `esds` child appended (its bytes only need to survive verbatim).
fn mp4a_entry_payload() -> Vec<u8> {
    let mut e = vec![0u8; 28];
    e[6..8].copy_from_slice(&1u16.to_be_bytes()); // dref index
    e[16..18].copy_from_slice(&2u16.to_be_bytes()); // channels
    e[24..28].copy_from_slice(&(48_000u32 << 16).to_be_bytes());
    e.extend_from_slice(&full_box(b"esds", &[0x03, 0x19, 0x00, 0x02]));
    e
}

/// A `soun` hdlr box, marking a trak as audio.
fn soun_hdlr() -> Vec<u8> {
    let mut body = 0u32.to_be_bytes().to_vec(); // pre_defined
    body.extend_from_slice(b"soun");
    body.extend_from_slice(&[0u8; 12]);
    body.push(0); // empty name
    full_box(b"hdlr", &body)
}

/// An audio trak with a valid entry but zero samples (entry parsing is
/// under test; audio payload walking shares the video code path).
fn audio_trak() -> Vec<u8> {
    audio_trak_with(b"mp4a")
}

/// Same trak shape with an arbitrary sample-entry fourcc (`fLaC`, ...).
fn audio_trak_with(entry_kind: &[u8; 4]) -> Vec<u8> {
    let mut stsd_body = 1u32.to_be_bytes().to_vec();
    stsd_body.extend_from_slice(&make_box(entry_kind, &mp4a_entry_payload()));
    let mut stbl_payload = full_box(b"stsd", &stsd_body);
    stbl_payload.extend_from_slice(&full_box(b"stsz", &[0u8; 8]));
    let mut stsc_body = 1u32.to_be_bytes().to_vec();
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stbl_payload.extend_from_slice(&full_box(b"stsc", &stsc_body));
    stbl_payload.extend_from_slice(&full_box(b"stco", &0u32.to_be_bytes()));
    stbl_payload.extend_from_slice(&stts_box(&[]));
    let stbl = make_box(b"stbl", &stbl_payload);
    let minf = make_box(b"minf", &stbl);
    let mut mdia_payload = mdhd_box(48_000, 96_000);
    mdia_payload.extend_from_slice(&soun_hdlr());
    mdia_payload.extend_from_slice(&minf);
    make_box(b"trak", &make_box(b"mdia", &mdia_payload))
}

/// Length-prefix a NAL with a 4-byte size.
fn prefixed(nal: &[u8]) -> Vec<u8> {
    let mut s = (nal.len() as u32).to_be_bytes().to_vec();
    s.extend_from_slice(nal);
    s
}

/// Assemble a complete file; returns (bytes, sample1, sample2).
fn synthetic_mp4() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let s1 = prefixed(IDR_NAL);
    let s2 = prefixed(P_NAL);
    let mut mdat_payload = s1.clone();
    mdat_payload.extend_from_slice(&s2);

    let ftyp = make_box(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = make_box(b"mdat", &mdat_payload);
    let s1_offset = (ftyp.len() + 8) as u32; // mdat header is 8 bytes

    let mut stsz_body = Vec::new();
    stsz_body.extend_from_slice(&0u32.to_be_bytes()); // per-sample sizes
    stsz_body.extend_from_slice(&2u32.to_be_bytes());
    stsz_body.extend_from_slice(&(s1.len() as u32).to_be_bytes());
    stsz_body.extend_from_slice(&(s2.len() as u32).to_be_bytes());

    let mut stsc_body = Vec::new();
    stsc_body.extend_from_slice(&1u32.to_be_bytes()); // one run
    stsc_body.extend_from_slice(&1u32.to_be_bytes()); // from chunk 1
    stsc_body.extend_from_slice(&2u32.to_be_bytes()); // 2 samples/chunk
    stsc_body.extend_from_slice(&1u32.to_be_bytes()); // desc index

    let mut stco_body = Vec::new();
    stco_body.extend_from_slice(&1u32.to_be_bytes());
    stco_body.extend_from_slice(&s1_offset.to_be_bytes());

    let mut stss_body = Vec::new();
    stss_body.extend_from_slice(&1u32.to_be_bytes());
    stss_body.extend_from_slice(&1u32.to_be_bytes()); // sample 1 is sync

    let mut stsd_body = Vec::new();
    stsd_body.extend_from_slice(&1u32.to_be_bytes());
    stsd_body.extend_from_slice(&avc1_entry());

    let mut stbl_payload = full_box(b"stsd", &stsd_body);
    stbl_payload.extend_from_slice(&full_box(b"stsz", &stsz_body));
    stbl_payload.extend_from_slice(&full_box(b"stsc", &stsc_body));
    stbl_payload.extend_from_slice(&full_box(b"stco", &stco_body));
    stbl_payload.extend_from_slice(&full_box(b"stss", &stss_body));
    stbl_payload.extend_from_slice(&stts_box(&[(2, FRAME_DELTA)]));
    stbl_payload.extend_from_slice(&ctts_box(&[(2, CTTS_OFFSET)]));

    let stbl = make_box(b"stbl", &stbl_payload);
    let minf = make_box(b"minf", &stbl);
    let mut mdia_payload = mdhd_box(TIMESCALE, 2 * FRAME_DELTA);
    mdia_payload.extend_from_slice(&minf);
    let mdia = make_box(b"mdia", &mdia_payload);
    let trak = make_box(b"trak", &mdia);
    let mut moov_payload = trak;
    moov_payload.extend_from_slice(&audio_trak());
    let moov = make_box(b"moov", &moov_payload);

    let mut file = ftyp;
    file.extend_from_slice(&mdat);
    file.extend_from_slice(&moov);
    (file, s1, s2)
}

#[test]
fn finds_the_track_with_dynamic_dimensions_and_codec() {
    let (file, ..) = synthetic_mp4();
    let track = find_video_track(&file).expect("track present");
    assert_eq!(track.codec, VideoCodec::H264);
    assert_eq!((track.width, track.height), (640, 360));
    assert_eq!(track.nal_length_size, 4);
    assert_eq!(track.sample_count(), 2);
    assert_eq!(track.timescale, TIMESCALE);
    assert_eq!(track.media_duration, u64::from(2 * FRAME_DELTA));
}

#[test]
fn parameter_sets_yield_sps_then_pps() {
    let (file, ..) = synthetic_mp4();
    let track = find_video_track(&file).unwrap();
    let sets: Vec<&[u8]> = track.parameter_sets().map(|p| p.unwrap()).collect();
    assert_eq!(sets, vec![SPS, PPS]);
}

#[test]
fn samples_are_zero_copy_and_keyframes_follow_stss() {
    let (file, s1, s2) = synthetic_mp4();
    let track = find_video_track(&file).unwrap();
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].data, &s1[..]);
    assert!(samples[0].keyframe);
    assert_eq!(samples[1].data, &s2[..]);
    assert!(!samples[1].keyframe);
    // Zero-copy proof: the sample points inside the file buffer.
    let range = file.as_ptr() as usize..file.as_ptr() as usize + file.len();
    assert!(range.contains(&(samples[0].data.as_ptr() as usize)));
}

#[test]
fn timing_follows_stts_and_ctts() {
    let (file, ..) = synthetic_mp4();
    let track = find_video_track(&file).unwrap();
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    assert_eq!(samples[0].dts, 0);
    assert_eq!(samples[1].dts, u64::from(FRAME_DELTA));
    assert_eq!(samples[0].pts, i64::from(CTTS_OFFSET));
    assert_eq!(samples[1].pts, i64::from(FRAME_DELTA + CTTS_OFFSET));
    assert!(samples.iter().all(|s| s.duration == FRAME_DELTA));
    let dur = track.duration().expect("mdhd carries duration");
    let expected = f64::from(2 * FRAME_DELTA) / f64::from(TIMESCALE);
    assert!((dur.as_secs_f64() - expected).abs() < 1e-9);
}

#[test]
fn audio_track_reports_entry_facts_verbatim() {
    let (file, ..) = synthetic_mp4();
    let audio = first_audio(&file).expect("audio trak present");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.timescale, 48_000);
    assert_eq!(audio.media_duration, 96_000);
    assert_eq!(audio.sample_count(), 0);
    // Bit-exact passthrough guarantee for the muxer.
    assert_eq!(audio.entry_payload, &mp4a_entry_payload()[..]);
}

#[test]
fn missing_audio_is_none_not_an_error() {
    // Rebuild the file without the audio trak (and shrink moov's size
    // field accordingly): truly video-only, no `soun` handler anywhere.
    let (orig, ..) = synthetic_mp4();
    let audio = audio_trak();
    let at = orig
        .windows(audio.len())
        .position(|w| w == &audio[..])
        .expect("audio trak embedded verbatim");
    let mut file = orig;
    file.drain(at..at + audio.len());
    let moov_at = file
        .windows(4)
        .position(|w| w == b"moov")
        .expect("moov present")
        - 4;
    let old = u32::from_be_bytes(file[moov_at..moov_at + 4].try_into().unwrap());
    file[moov_at..moov_at + 4].copy_from_slice(&(old - audio.len() as u32).to_be_bytes());
    assert!(find_audio_tracks(&file).unwrap().is_silent());
}

#[test]
fn unsupported_audio_names_the_codec_instead_of_vanishing() {
    // Same file, but the audio entry is FLAC-in-MP4: a `soun` handler
    // trak whose sample entry we cannot pass through.
    let (file, ..) = synthetic_mp4();
    let audio = audio_trak();
    let at = file
        .windows(audio.len())
        .position(|w| w == &audio[..])
        .expect("audio trak embedded verbatim");
    let mut flac = file.clone();
    let entry_at = at
        + audio
            .windows(4)
            .position(|w| w == b"mp4a")
            .expect("entry fourcc");
    flac[entry_at..entry_at + 4].copy_from_slice(b"fLaC");
    match find_audio_tracks(&flac).unwrap() {
        found if !found.unsupported.is_empty() => assert_eq!(&found.unsupported[0], b"fLaC"),
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn annex_b_injects_parameter_sets_on_keyframes_only() {
    let (file, ..) = synthetic_mp4();
    let track = find_video_track(&file).unwrap();
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    let mut out = Vec::new();

    track.write_frame(&samples[0], &mut out).unwrap();
    let mut expected = Vec::new();
    for part in [SPS, PPS, IDR_NAL] {
        expected.extend_from_slice(&[0, 0, 0, 1]);
        expected.extend_from_slice(part);
    }
    assert_eq!(out, expected, "keyframe: SPS+PPS then the IDR NAL");

    track.write_frame(&samples[1], &mut out).unwrap();
    let mut expected = vec![0, 0, 0, 1];
    expected.extend_from_slice(P_NAL);
    assert_eq!(out, expected, "non-keyframe: bare start-coded NAL");
}

#[test]
fn lying_nal_length_is_a_typed_error() {
    let (mut file, s1, _) = synthetic_mp4();
    // Corrupt sample 1's NAL length prefix to claim 200 bytes.
    let mdat_data_at = file.windows(s1.len()).position(|w| w == s1).unwrap();
    file[mdat_data_at + 3] = 200;
    let track = find_video_track(&file).unwrap();
    let sample = track.samples(&file).next().unwrap().unwrap();
    let err = track.write_frame(&sample, &mut Vec::new()).unwrap_err();
    assert_eq!(err, DemuxError::BadNalLength { sample: 0 });
}

#[test]
fn chunk_offset_outside_the_file_is_caught() {
    let (mut file, ..) = synthetic_mp4();
    // The video stco's single entry: overwrite with an offset past EOF.
    let stco_at = file.windows(4).position(|w| w == b"stco").unwrap();
    let entry_at = stco_at + 4 /*kind*/ + 4 /*ver/flags*/ + 4 /*count*/;
    file[entry_at..entry_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    let track = find_video_track(&file).unwrap();
    let err = track.samples(&file).next().unwrap().unwrap_err();
    assert_eq!(err, DemuxError::SampleOutOfFile { index: 0 });
}

#[test]
fn truncated_sample_size_table_is_rejected_up_front() {
    let (mut file, ..) = synthetic_mp4();
    // Video stsz claims 2 samples; slash its count to 200 without adding
    // bytes (the audio stsz comes later in the file, so `position`).
    let stsz_at = file.windows(4).position(|w| w == b"stsz").unwrap();
    let count_at = stsz_at + 4 + 4 + 4; // kind, ver/flags, uniform-size
    file[count_at..count_at + 4].copy_from_slice(&200u32.to_be_bytes());
    assert_eq!(
        find_video_track(&file).unwrap_err(),
        DemuxError::BadSampleTable("stsz")
    );
}

#[test]
fn hevc_config_records_parse_vps_sps_pps_arrays() {
    // Minimal hvcC: 23-byte fixed header + 2 arrays (VPS[1], SPS[1]).
    let vps: &[u8] = &[0x40, 0x01, 0x0C];
    let sps: &[u8] = &[0x42, 0x01, 0x01, 0x02];
    let mut hvcc = vec![0u8; 21]; // fixed header bytes 0..=20
    hvcc[0] = 1; // configurationVersion
    hvcc.push(0x03); // byte 21: lengthSizeMinusOne = 3 -> 4-byte NALs
    hvcc.push(2); // two arrays
    for nal in [vps, sps] {
        hvcc.push(0x20); // array header (type)
        hvcc.extend_from_slice(&1u16.to_be_bytes());
        hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        hvcc.extend_from_slice(nal);
    }

    // Wrap in a minimal hvc1 track (reuse the H.264 scaffolding shape).
    let mut entry = vec![0u8; 78];
    entry[24..26].copy_from_slice(&1920u16.to_be_bytes());
    entry[26..28].copy_from_slice(&1080u16.to_be_bytes());
    entry.extend_from_slice(&make_box(b"hvcC", &hvcc));
    let hvc1 = make_box(b"hvc1", &entry);

    let mut stsd_body = Vec::new();
    stsd_body.extend_from_slice(&1u32.to_be_bytes());
    stsd_body.extend_from_slice(&hvc1);
    let mut stbl_payload = full_box(b"stsd", &stsd_body);
    // Empty-but-valid tables: 0 samples, 0 chunks, one stsc run.
    stbl_payload.extend_from_slice(&full_box(b"stsz", &[0u8; 8]));
    let mut stsc_body = 1u32.to_be_bytes().to_vec();
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stbl_payload.extend_from_slice(&full_box(b"stsc", &stsc_body));
    stbl_payload.extend_from_slice(&full_box(b"stco", &0u32.to_be_bytes()));
    stbl_payload.extend_from_slice(&stts_box(&[]));

    let mut mdia_payload = mdhd_box(90_000, 0);
    mdia_payload.extend_from_slice(&make_box(b"minf", &make_box(b"stbl", &stbl_payload)));
    let moov = make_box(
        b"moov",
        &make_box(b"trak", &make_box(b"mdia", &mdia_payload)),
    );
    let mut file = make_box(b"ftyp", b"isomisom");
    file.extend_from_slice(&moov);

    let track = find_video_track(&file).expect("hevc track");
    assert_eq!(track.codec, VideoCodec::Hevc);
    assert_eq!((track.width, track.height), (1920, 1080));
    assert_eq!(track.nal_length_size, 4);
    let sets: Vec<&[u8]> = track.parameter_sets().map(|p| p.unwrap()).collect();
    assert_eq!(sets, vec![vps, sps]);
}

#[test]
fn audio_only_files_report_missing_video_track() {
    let moov = make_box(b"moov", &audio_trak());
    assert_eq!(
        find_video_track(&moov).unwrap_err(),
        DemuxError::MissingBox("trak with avc1/hvc1/hev1/av01")
    );
}

// ---- multi-track audio ------------------------------------------------------

/// Pack a three-letter code the way `mdhd` carries it: three 5-bit
/// values biased by 0x60, high bit padding.
fn pack_language(code: &[u8; 3]) -> u16 {
    let mut packed = 0u16;
    for c in code {
        packed = (packed << 5) | u16::from(c - 0x60);
    }
    packed
}

/// Version-0 `mdhd` carrying an explicit language.
fn mdhd_box_lang(timescale: u32, duration: u32, language: &[u8; 3]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes()); // creation
    body.extend_from_slice(&0u32.to_be_bytes()); // modification
    body.extend_from_slice(&timescale.to_be_bytes());
    body.extend_from_slice(&duration.to_be_bytes());
    body.extend_from_slice(&pack_language(language).to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    full_box(b"mdhd", &body)
}

/// Version-0 `tkhd` with an explicit id, alternate group and enabled bit.
fn tkhd_box(track_id: u32, alternate_group: u16, enabled: bool) -> Vec<u8> {
    let mut payload = Vec::new();
    let flags: u32 = if enabled { 0x1 } else { 0x0 };
    payload.extend_from_slice(&flags.to_be_bytes()); // version 0 + flags
    payload.extend_from_slice(&0u32.to_be_bytes()); // creation
    payload.extend_from_slice(&0u32.to_be_bytes()); // modification
    payload.extend_from_slice(&track_id.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
    payload.extend_from_slice(&0u32.to_be_bytes()); // duration
    payload.extend_from_slice(&[0u8; 8]); // reserved
    payload.extend_from_slice(&0u16.to_be_bytes()); // layer
    payload.extend_from_slice(&alternate_group.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes()); // volume
    payload.extend_from_slice(&0u16.to_be_bytes()); // reserved
    payload.extend_from_slice(&[0u8; 36]); // matrix
    payload.extend_from_slice(&0u32.to_be_bytes()); // width
    payload.extend_from_slice(&0u32.to_be_bytes()); // height
    make_box(b"tkhd", &payload)
}

/// A fully identified audio trak: codec entry, language, id, group, title.
fn identified_audio_trak(
    entry_kind: &[u8; 4],
    language: &[u8; 3],
    track_id: u32,
    alternate_group: u16,
    name: Option<&str>,
) -> Vec<u8> {
    identified_audio_trak_sized(entry_kind, language, track_id, alternate_group, name, 0, 0)
}

/// As above, but with a uniform-size `stsz` so the track has a real,
/// computable bitrate (`uniform` bytes x `count` samples over the
/// `mdhd` duration).
#[allow(clippy::too_many_arguments)]
fn identified_audio_trak_sized(
    entry_kind: &[u8; 4],
    language: &[u8; 3],
    track_id: u32,
    alternate_group: u16,
    name: Option<&str>,
    uniform: u32,
    count: u32,
) -> Vec<u8> {
    let mut stsd_body = 1u32.to_be_bytes().to_vec();
    stsd_body.extend_from_slice(&make_box(entry_kind, &mp4a_entry_payload()));
    let mut stbl_payload = full_box(b"stsd", &stsd_body);
    let mut stsz_body = uniform.to_be_bytes().to_vec();
    stsz_body.extend_from_slice(&count.to_be_bytes());
    stbl_payload.extend_from_slice(&full_box(b"stsz", &stsz_body));
    let mut stsc_body = 1u32.to_be_bytes().to_vec();
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stbl_payload.extend_from_slice(&full_box(b"stsc", &stsc_body));
    stbl_payload.extend_from_slice(&full_box(b"stco", &0u32.to_be_bytes()));
    stbl_payload.extend_from_slice(&stts_box(&[]));
    let minf = make_box(b"minf", &make_box(b"stbl", &stbl_payload));

    let mut mdia_payload = mdhd_box_lang(48_000, 96_000, language);
    mdia_payload.extend_from_slice(&soun_hdlr());
    mdia_payload.extend_from_slice(&minf);

    let mut trak_payload = tkhd_box(track_id, alternate_group, true);
    trak_payload.extend_from_slice(&make_box(b"mdia", &mdia_payload));
    if let Some(title) = name {
        trak_payload.extend_from_slice(&make_box(b"udta", &make_box(b"name", title.as_bytes())));
    }
    make_box(b"trak", &trak_payload)
}

/// A file with one video track and the given audio traks appended.
fn mp4_with_audio_traks(traks: &[Vec<u8>]) -> Vec<u8> {
    let (base, ..) = synthetic_mp4();
    // Rebuild moov: keep the video trak, drop the stock audio trak, and
    // append the requested ones.
    let moov = find_path(&base, &[*b"moov"]).expect("synthetic file has moov");
    let mut moov_payload = Vec::new();
    for b in boxes(moov) {
        let b = b.unwrap();
        if b.kind != *b"trak" {
            continue;
        }
        // The video trak has no `soun` handler; keep exactly that one.
        let is_audio = find_path(b.payload, &[*b"mdia", *b"hdlr"])
            .map(|h| h[8..12] == *b"soun")
            .unwrap_or(false);
        if !is_audio {
            moov_payload.extend_from_slice(&make_box(b"trak", b.payload));
        }
    }
    for t in traks {
        moov_payload.extend_from_slice(t);
    }
    let new_moov = make_box(b"moov", &moov_payload);

    // Reassemble ftyp | mdat | moov, preserving the original prefix so
    // the video sample offsets stay valid.
    let moov_start = base.len() - moov.len() - 8;
    let mut file = base[..moov_start].to_vec();
    file.extend_from_slice(&new_moov);
    file
}

#[test]
fn every_audio_track_is_found_with_its_identity_intact() {
    // The regression this pins: the old scanner returned the FIRST
    // audio track, so a six-language master silently lost five of six.
    let langs: [&[u8; 3]; 6] = [b"eng", b"rus", b"deu", b"fra", b"jpn", b"spa"];
    let traks: Vec<Vec<u8>> = langs
        .iter()
        .enumerate()
        .map(|(i, lang)| {
            identified_audio_trak(
                b"mp4a",
                lang,
                i as u32 + 2,
                1,
                Some(&format!("Track {}", i + 1)),
            )
        })
        .collect();
    let file = mp4_with_audio_traks(&traks);

    let found = find_audio_tracks(&file).expect("multi-track file parses");
    assert!(!found.truncated);
    assert!(found.unsupported.is_empty());
    assert_eq!(found.tracks.len(), 6, "six in, six out");

    let codes: Vec<String> = found
        .tracks
        .iter()
        .map(|t| t.meta.language.to_string())
        .collect();
    assert_eq!(codes, vec!["eng", "rus", "deu", "fra", "jpn", "spa"]);

    for (i, track) in found.tracks.iter().enumerate() {
        assert_eq!(track.meta.track_id, i as u32 + 2, "track id preserved");
        assert_eq!(track.meta.alternate_group, 1, "they are alternatives");
        assert!(track.meta.enabled);
        assert_eq!(track.meta.name, Some(format!("Track {}", i + 1)).as_deref());
        assert_eq!(track.channels, 2);
        assert_eq!(track.sample_rate, 48_000);
    }
}

#[test]
fn tracks_come_back_in_file_order() {
    // Old callers must keep working while the engine migrates.
    let traks = vec![
        identified_audio_trak(b"mp4a", b"eng", 2, 1, None),
        identified_audio_trak(b"mp4a", b"rus", 3, 1, None),
    ];
    let file = mp4_with_audio_traks(&traks);
    let first = first_audio(&file).expect("a track");
    assert_eq!(first.meta.language.to_string(), "eng");
}

#[test]
fn unsupported_tracks_are_counted_alongside_the_ones_we_can_carry() {
    // Mixed reality: some tracks pass through, some cannot. Both facts
    // must survive, because the warning is built from the second.
    let traks = vec![
        identified_audio_trak(b"mp4a", b"eng", 2, 1, None),
        identified_audio_trak(b"fLaC", b"rus", 3, 1, None),
        identified_audio_trak(b"mp4a", b"deu", 4, 1, None),
        identified_audio_trak(b"ac-3", b"fra", 5, 1, None),
    ];
    let file = mp4_with_audio_traks(&traks);
    let found = find_audio_tracks(&file).unwrap();

    assert_eq!(found.tracks.len(), 2);
    assert_eq!(found.unsupported, vec![*b"fLaC", *b"ac-3"]);
    assert!(!found.is_silent(), "there is audio, just not all carryable");
}

#[test]
fn a_file_with_no_audio_at_all_is_silent_not_unsupported() {
    let file = mp4_with_audio_traks(&[]);
    let found = find_audio_tracks(&file).unwrap();
    assert!(found.is_silent());
    assert!(found.tracks.is_empty() && found.unsupported.is_empty());
}

#[test]
fn more_tracks_than_the_ceiling_stop_the_scan_and_say_so() {
    // Bounded execution: a hostile moov can declare any number of traks.
    // Stopping is correct; stopping *quietly* would be the silent data
    // loss this milestone exists to eliminate.
    let traks: Vec<Vec<u8>> = (0..MAX_AUDIO_TRACKS + 5)
        .map(|i| identified_audio_trak(b"mp4a", b"eng", i as u32 + 2, 1, None))
        .collect();
    let file = mp4_with_audio_traks(&traks);
    let found = find_audio_tracks(&file).unwrap();
    assert_eq!(found.tracks.len(), MAX_AUDIO_TRACKS);
    assert!(found.truncated, "truncation must be reported, never silent");
}

#[test]
fn total_bitrate_sums_every_track_not_just_the_first() {
    // Six passthrough tracks can outweigh the compressed video, so the
    // heuristic engine is only honest if it weighs all of them. Sizes
    // are chosen so the expected answer is arithmetic, not whatever the
    // implementation happens to produce: 1000 B x 120 samples = 120 kB
    // over 2 s (96000 units at 48 kHz) = 480 kbit/s per track.
    const BYTES_PER_SAMPLE: u32 = 1000;
    const SAMPLE_COUNT: u32 = 120;
    const PER_TRACK_BITS_PER_SEC: u64 = 480_000;
    const TRACKS: u32 = 3;

    let traks: Vec<Vec<u8>> = (0..TRACKS)
        .map(|i| {
            identified_audio_trak_sized(
                b"mp4a",
                b"eng",
                i + 2,
                1,
                None,
                BYTES_PER_SAMPLE,
                SAMPLE_COUNT,
            )
        })
        .collect();
    let file = mp4_with_audio_traks(&traks);
    let found = find_audio_tracks(&file).unwrap();

    assert_eq!(found.tracks.len(), TRACKS as usize);
    assert_eq!(found.tracks[0].bitrate(), PER_TRACK_BITS_PER_SEC);
    assert_eq!(
        found.total_bitrate(),
        PER_TRACK_BITS_PER_SEC * u64::from(TRACKS),
        "the total must be every track, not the first one repeated"
    );
}

#[test]
fn a_track_we_cannot_carry_still_counts_towards_what_the_file_weighs() {
    // The defect this closes. `total_bitrate` counted only the tracks
    // this project can copy, so a PCM track in an MP4 — which cannot be
    // carried and is also the heaviest thing in the container —
    // reported as nothing. The heuristic engine then understated
    // exactly the files whose sound weighs most.
    //
    // The engine weighs *the file*, not the part of it we happen to be
    // able to copy.
    const BYTES_PER_SAMPLE: u32 = 1000;
    const SAMPLE_COUNT: u32 = 120;
    const PER_TRACK_BITS_PER_SEC: u64 = 480_000;

    let traks = vec![
        identified_audio_trak_sized(b"mp4a", b"eng", 2, 1, None, BYTES_PER_SAMPLE, SAMPLE_COUNT),
        // Same size, a codec the muxer cannot re-emit.
        identified_audio_trak_sized(b"lpcm", b"eng", 3, 1, None, BYTES_PER_SAMPLE, SAMPLE_COUNT),
    ];
    let file = mp4_with_audio_traks(&traks);
    let found = find_audio_tracks(&file).unwrap();

    assert_eq!(found.tracks.len(), 1, "one carryable track");
    assert_eq!(found.unsupported.len(), 1, "one that is not");
    assert_eq!(
        found.carried_bitrate(),
        PER_TRACK_BITS_PER_SEC,
        "what passthrough will copy"
    );
    assert_eq!(
        found.unsupported_bitrate, PER_TRACK_BITS_PER_SEC,
        "what the file carries that we cannot"
    );
    assert_eq!(
        found.total_bitrate(),
        2 * PER_TRACK_BITS_PER_SEC,
        "the file's audio weight is all of it"
    );
}

#[test]
fn a_garbage_language_field_degrades_to_undetermined() {
    // Negative trust: the packing can express bytes that are not letters,
    // and real files do carry zeroed language fields. A code made of
    // control characters would be worse than admitting we do not know.
    let mut trak = identified_audio_trak(b"mp4a", b"eng", 2, 1, None);
    // Overwrite the packed language with a value whose 5-bit groups fall
    // outside a-z.
    let needle = pack_language(b"eng").to_be_bytes();
    let at = trak
        .windows(2)
        .position(|w| w == needle)
        .expect("language field present");
    trak[at..at + 2].copy_from_slice(&0x0000u16.to_be_bytes());
    let file = mp4_with_audio_traks(&[trak]);
    let found = find_audio_tracks(&file).unwrap();
    assert_eq!(found.tracks[0].meta.language, Language::UNDETERMINED);
}

#[test]
fn a_track_without_headers_still_yields_media_with_default_identity() {
    // Graceful degradation: missing tkhd/udta must not fail the file.
    // `audio_trak()` has no tkhd and no udta at all.
    let file = mp4_with_audio_traks(&[audio_trak()]);
    let found = find_audio_tracks(&file).unwrap();
    assert_eq!(found.tracks.len(), 1);
    let meta = found.tracks[0].meta;
    assert_eq!(meta.track_id, 0);
    assert_eq!(meta.language, Language::UNDETERMINED);
    assert_eq!(meta.name, None);
    assert!(meta.enabled, "absent flags must not mean 'disabled'");
}

// ---- stz2, the compact sample-size table ------------------------------------

/// Build an `stz2` body for `sizes` at the given field width in bits.
fn stz2_box(field_bits: u8, sizes: &[u32]) -> Vec<u8> {
    let mut body = vec![0u8, 0, 0, field_bits];
    body.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    match field_bits {
        4 => {
            // Two entries per byte, high nibble first; an odd count
            // leaves the final low nibble as padding.
            for pair in sizes.chunks(2) {
                let hi = (pair[0] as u8) & 0x0F;
                let lo = pair.get(1).map(|v| (*v as u8) & 0x0F).unwrap_or(0);
                body.push((hi << 4) | lo);
            }
        }
        8 => body.extend(sizes.iter().map(|s| *s as u8)),
        16 => {
            for s in sizes {
                body.extend_from_slice(&(*s as u16).to_be_bytes());
            }
        }
        other => panic!("stz2 field width {other} is not legal"),
    }
    full_box(b"stz2", &body)
}

/// A single-track file whose sample sizes live in `stz2` rather than
/// `stsz`, with `payloads` laid out back to back in one chunk.
fn mp4_with_stz2(field_bits: u8, payloads: &[Vec<u8>]) -> Vec<u8> {
    let mut mdat_payload = Vec::new();
    for p in payloads {
        mdat_payload.extend_from_slice(p);
    }
    let ftyp = make_box(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = make_box(b"mdat", &mdat_payload);
    let first_offset = (ftyp.len() + 8) as u32;

    let sizes: Vec<u32> = payloads.iter().map(|p| p.len() as u32).collect();

    let mut stsc_body = Vec::new();
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());
    stsc_body.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    stsc_body.extend_from_slice(&1u32.to_be_bytes());

    let mut stco_body = Vec::new();
    stco_body.extend_from_slice(&1u32.to_be_bytes());
    stco_body.extend_from_slice(&first_offset.to_be_bytes());

    let mut stss_body = Vec::new();
    stss_body.extend_from_slice(&1u32.to_be_bytes());
    stss_body.extend_from_slice(&1u32.to_be_bytes());

    let mut stsd_body = Vec::new();
    stsd_body.extend_from_slice(&1u32.to_be_bytes());
    stsd_body.extend_from_slice(&avc1_entry());

    let mut stbl_payload = full_box(b"stsd", &stsd_body);
    stbl_payload.extend_from_slice(&stz2_box(field_bits, &sizes));
    stbl_payload.extend_from_slice(&full_box(b"stsc", &stsc_body));
    stbl_payload.extend_from_slice(&full_box(b"stco", &stco_body));
    stbl_payload.extend_from_slice(&full_box(b"stss", &stss_body));
    stbl_payload.extend_from_slice(&stts_box(&[(sizes.len() as u32, FRAME_DELTA)]));

    let stbl = make_box(b"stbl", &stbl_payload);
    let minf = make_box(b"minf", &stbl);
    let mut mdia_payload = mdhd_box(TIMESCALE, sizes.len() as u32 * FRAME_DELTA);
    mdia_payload.extend_from_slice(&minf);
    let mdia = make_box(b"mdia", &mdia_payload);
    let trak = make_box(b"trak", &mdia);
    let moov = make_box(b"moov", &trak);

    let mut file = ftyp;
    file.extend_from_slice(&mdat);
    file.extend_from_slice(&moov);
    file
}

#[test]
fn the_compact_stz2_table_is_read_at_every_legal_field_width() {
    // We never write `stz2`, which is exactly why we must read it: a
    // muxer that chose the compact form produced a perfectly valid file,
    // and rejecting it would be our defect rather than theirs. Before
    // this the box was not recognised at all and the whole job failed
    // with a missing-`stsz` error.
    //
    // Sizes are chosen to be representable at 4 bits (max 15), so the
    // same payloads exercise all three widths and the widths can be
    // compared against each other.
    let payloads: Vec<Vec<u8>> = [5usize, 12, 3, 15, 8]
        .iter()
        .enumerate()
        .map(|(i, n)| vec![0xA0 | i as u8; *n])
        .collect();

    for bits in [4u8, 8, 16] {
        let file = mp4_with_stz2(bits, &payloads);
        let track =
            find_video_track(&file).unwrap_or_else(|e| panic!("stz2/{bits} must parse, got {e:?}"));
        assert_eq!(track.sample_count() as usize, payloads.len());

        let samples: Vec<_> = track
            .samples(&file)
            .map(|s| s.unwrap_or_else(|e| panic!("stz2/{bits} sample: {e:?}")))
            .collect();
        assert_eq!(samples.len(), payloads.len(), "stz2/{bits} sample count");
        for (i, (got, want)) in samples.iter().zip(&payloads).enumerate() {
            assert_eq!(
                got.data,
                &want[..],
                "stz2/{bits} sample {i} must resolve to the right bytes"
            );
        }
    }
}

#[test]
fn an_odd_sample_count_at_four_bits_does_not_read_the_padding_nibble() {
    // The last byte of a 4-bit table holds one real entry and one
    // padding nibble. Reading the padding as a sample would append a
    // phantom zero-length sample to every such file.
    let payloads: Vec<Vec<u8>> = vec![vec![0x11; 7], vec![0x22; 9], vec![0x33; 4]];
    let file = mp4_with_stz2(4, &payloads);
    let track = find_video_track(&file).expect("odd 4-bit stz2 parses");
    assert_eq!(track.sample_count(), 3, "three samples, not four");
    let samples: Vec<_> = track.samples(&file).map(|s| s.unwrap()).collect();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[2].data, &payloads[2][..]);
}

#[test]
fn an_illegal_stz2_field_width_is_refused_rather_than_guessed() {
    // Negative trust: 12 is not one of the three legal widths. Guessing
    // would desynchronise the whole table and hand back plausible
    // garbage — worse than a clean refusal.
    let mut file = mp4_with_stz2(8, &[vec![0x55; 4], vec![0x66; 6]]);
    let at = file.windows(4).position(|w| w == b"stz2").unwrap();
    // kind(4) + version/flags(4) + 3 reserved bytes -> the width byte.
    file[at + 4 + 4 + 3] = 12;
    assert!(
        matches!(
            find_video_track(&file),
            Err(DemuxError::BadSampleTable("stz2"))
        ),
        "an illegal field width must be refused"
    );
}

#[test]
fn a_truncated_stz2_table_is_refused_before_any_sample_is_read() {
    // The count says five entries; the table carries bytes for two.
    // Trusting the count would walk off the end of the box.
    let file = mp4_with_stz2(16, &[vec![0x77; 3], vec![0x88; 5]]);
    let at = file.windows(4).position(|w| w == b"stz2").unwrap();
    let count_at = at + 4 + 4 + 4; // kind, version/flags, reserved+width
    let mut broken = file.clone();
    broken[count_at..count_at + 4].copy_from_slice(&5u32.to_be_bytes());
    assert!(
        matches!(
            find_video_track(&broken),
            Err(DemuxError::BadSampleTable("stz2"))
        ),
        "a table shorter than its own count must be refused"
    );
}

// ---- surveying what else the source carries ---------------------------------

/// A minimal `trak` with the given handler fourcc and nothing else.
///
/// The survey only reads `hdlr` and `tref`, so a track needs no sample
/// tables to be counted — and must not, since a real subtitle track's
/// tables describe a codec we do not parse.
fn handler_trak(handler: &[u8; 4]) -> Vec<u8> {
    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(handler);
    hdlr_body.extend_from_slice(&[0u8; 12]);
    hdlr_body.push(0); // empty name
    let mdia = make_box(b"mdia", &full_box(b"hdlr", &hdlr_body));
    make_box(b"trak", &mdia)
}

/// Rebuild the synthetic file's `moov` with `extra` boxes appended.
fn mp4_with_extra_moov_boxes(extra: &[Vec<u8>]) -> Vec<u8> {
    let (base, ..) = synthetic_mp4();
    let moov = find_path(&base, &[*b"moov"]).expect("synthetic file has moov");
    let mut moov_payload = moov.to_vec();
    for e in extra {
        moov_payload.extend_from_slice(e);
    }
    let new_moov = make_box(b"moov", &moov_payload);

    let mut out = Vec::new();
    for b in boxes(&base) {
        let b = b.unwrap();
        if b.kind == *b"moov" {
            out.extend_from_slice(&new_moov);
        } else {
            out.extend_from_slice(&make_box(&b.kind, b.payload));
        }
    }
    out
}

#[test]
fn a_plain_file_has_nothing_extra_to_lose() {
    // The common case must stay quiet. A warning that fires on every
    // ordinary file is a warning nobody reads.
    let (file, ..) = synthetic_mp4();
    let extras = survey_extras(&file).expect("survey succeeds");
    assert!(extras.is_empty(), "got {extras:?}");
    assert_eq!(extras.lost_description(), None);
}

#[test]
fn every_subtitle_handler_convention_is_counted() {
    // Three handler fourccs mean three eras of the format. A survey that
    // knew only `subt` would under-report on nearly every real file,
    // since `sbtl` is what modern muxers actually write.
    for handler in [b"subt", b"sbtl", b"text"] {
        let file = mp4_with_extra_moov_boxes(&[handler_trak(handler)]);
        let extras = survey_extras(&file).expect("survey succeeds");
        assert_eq!(
            extras.subtitles,
            1,
            "handler {} must count as a subtitle track",
            String::from_utf8_lossy(handler)
        );
    }
}

#[test]
fn several_subtitle_tracks_are_counted_not_merely_noticed() {
    // "Has subtitles" is not actionable; "3 subtitle track(s)" is.
    let traks: Vec<Vec<u8>> = [b"sbtl", b"subt", b"text"]
        .iter()
        .map(|h| handler_trak(h))
        .collect();
    let file = mp4_with_extra_moov_boxes(&traks);
    let extras = survey_extras(&file).expect("survey succeeds");
    assert_eq!(extras.subtitles, 3);
    let lost = extras.lost_description().expect("something is lost");
    assert!(lost.contains('3'), "the count must reach the user: {lost}");
}

#[test]
fn chapters_are_found_by_either_convention() {
    // Nero writes `udta/chpl`; QuickTime instead points at a text track
    // through `tref/chap`. Both are in the wild, so knowing only one
    // would silently drop half the chaptered files there are.
    let nero = make_box(b"udta", &full_box(b"chpl", &[0u8; 8]));
    let file = mp4_with_extra_moov_boxes(&[nero]);
    assert!(
        survey_extras(&file).unwrap().chapters,
        "a Nero chpl list is chapters"
    );

    let tref = make_box(b"tref", &make_box(b"chap", &2u32.to_be_bytes()));
    let qt_trak = make_box(b"trak", &tref);
    let file = mp4_with_extra_moov_boxes(&[qt_trak]);
    assert!(
        survey_extras(&file).unwrap().chapters,
        "a tref/chap reference is chapters"
    );
}

/// Build `udta/meta/ilst` holding the given tag fourccs.
fn tag_island(items: &[[u8; 4]]) -> Vec<u8> {
    let mut ilst_payload = Vec::new();
    for kind in items {
        // The value shape does not matter here; only which tags exist.
        ilst_payload.extend_from_slice(&make_box(kind, b"\0\0\0\x0cdata"));
    }
    let mut meta_payload = vec![0u8; 4]; // meta is a FullBox
    meta_payload.extend_from_slice(&make_box(b"ilst", &ilst_payload));
    make_box(b"udta", &make_box(b"meta", &meta_payload))
}

/// The `©too` encoder tag every muxer writes.
const TAG_ENCODER: [u8; 4] = [0xA9, b't', b'o', b'o'];
/// A tag a user would actually miss: the title.
const TAG_TITLE: [u8; 4] = [0xA9, b'n', b'a', b'm'];

#[test]
fn real_tags_and_cover_art_count_as_something_to_lose() {
    let file = mp4_with_extra_moov_boxes(&[tag_island(&[TAG_TITLE, *b"covr"])]);
    let extras = survey_extras(&file).expect("survey succeeds");
    assert!(extras.tags, "a title and cover art are user data");
    let lost = extras.lost_description().expect("something is lost");
    assert!(lost.contains("tags"), "{lost}");
}

#[test]
fn an_island_holding_only_the_encoder_tag_stays_quiet() {
    // Practically every muxer writes `©too`, and sovmr replaces it on
    // the way out, so it is not user data. Counting it made the notice
    // fire on every ordinary file — caught by the existing gates, which
    // is exactly what they are for. A warning that always fires is a
    // warning nobody reads.
    let file = mp4_with_extra_moov_boxes(&[tag_island(&[TAG_ENCODER])]);
    let extras = survey_extras(&file).expect("survey succeeds");
    assert!(!extras.tags, "the encoder tag alone is not user data");
    assert_eq!(extras.lost_description(), None);
}

#[test]
fn a_real_tag_alongside_the_encoder_tag_still_counts() {
    // The encoder tag must be ignored, not treated as poisoning the
    // whole island: a file with both must still report its title.
    let file = mp4_with_extra_moov_boxes(&[tag_island(&[TAG_ENCODER, TAG_TITLE])]);
    assert!(survey_extras(&file).unwrap().tags);
}

#[test]
fn a_meta_box_written_without_its_version_word_is_still_read() {
    // `meta` is a FullBox, but a few writers emit it as a plain box.
    // Refusing to read those would under-report, which is the
    // silent-loss direction.
    let mut meta_payload = Vec::new(); // no version/flags word
    meta_payload.extend_from_slice(&make_box(b"ilst", &make_box(&TAG_TITLE, b"\0\0\0\x0cdata")));
    let udta = make_box(b"udta", &make_box(b"meta", &meta_payload));
    let file = mp4_with_extra_moov_boxes(&[udta]);
    assert!(survey_extras(&file).unwrap().tags);
}

#[test]
fn a_file_without_a_moov_is_an_error_not_a_silent_empty_answer() {
    // Reporting "nothing extra" for a file we could not read at all
    // would be a lie in the user's favour, which is the worst kind.
    let junk = make_box(b"ftyp", b"isom\0\0\0\0isom");
    assert!(matches!(
        survey_extras(&junk),
        Err(DemuxError::MissingBox("moov"))
    ));
}

#[test]
fn a_malformed_track_does_not_fail_the_survey() {
    // Negative trust, pointed the other way: this answer only drives a
    // warning, so a garbled `trak` must cost the user information, not
    // their twelve-hour encode.
    let truncated = make_box(b"trak", &[0xFFu8; 3]);
    let file = mp4_with_extra_moov_boxes(&[truncated, handler_trak(b"sbtl")]);
    let extras = survey_extras(&file).expect("a bad trak must not fail the survey");
    assert_eq!(extras.subtitles, 1, "the readable track is still counted");
}

/// Track IDs used by the chapter-track disambiguation test.
const CHAPTER_TRACK_ID: u32 = 7;
/// A real subtitle track living alongside it.
const SUBTITLE_TRACK_ID: u32 = 8;

/// A `trak` with a handler, a `tkhd` carrying `track_id`, and nothing
/// else the survey looks at.
fn identified_handler_trak(handler: &[u8; 4], track_id: u32) -> Vec<u8> {
    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(handler);
    hdlr_body.extend_from_slice(&[0u8; 12]);
    hdlr_body.push(0);
    let mdia = make_box(b"mdia", &full_box(b"hdlr", &hdlr_body));
    let mut payload = tkhd_box(track_id, 0, true);
    payload.extend_from_slice(&mdia);
    make_box(b"trak", &payload)
}

#[test]
fn a_chapter_track_is_not_also_counted_as_a_subtitle() {
    // A QuickTime chapter track *is* a text track — the only thing that
    // makes it chapters is another track pointing at it through
    // `tref/chap`. Counting it twice made the notice claim two subtitle
    // tracks where the user could see exactly one, which is the kind of
    // wrong number that teaches people to ignore warnings. Observed on a
    // real ffmpeg-produced file, not hypothesised.
    let chapter_trak = identified_handler_trak(b"text", CHAPTER_TRACK_ID);
    let subtitle_trak = identified_handler_trak(b"sbtl", SUBTITLE_TRACK_ID);
    // The video track is what references the chapter track.
    let referrer = make_box(
        b"trak",
        &make_box(b"tref", &make_box(b"chap", &CHAPTER_TRACK_ID.to_be_bytes())),
    );

    let file = mp4_with_extra_moov_boxes(&[chapter_trak, subtitle_trak, referrer]);
    let extras = survey_extras(&file).expect("survey succeeds");
    assert!(extras.chapters, "the chap reference means chapters exist");
    assert_eq!(
        extras.subtitles, 1,
        "only the genuine subtitle track counts: {extras:?}"
    );
}

#[test]
fn a_text_track_nobody_references_is_still_a_subtitle_track() {
    // The other side of the same rule: text tracks are subtitles by
    // default. Suppressing them all would trade an over-count for an
    // under-count, and under-counting is the silent-loss direction.
    let orphan = identified_handler_trak(b"text", CHAPTER_TRACK_ID);
    let file = mp4_with_extra_moov_boxes(&[orphan]);
    assert_eq!(survey_extras(&file).unwrap().subtitles, 1);
}
