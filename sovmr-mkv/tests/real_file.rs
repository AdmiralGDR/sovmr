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

//! The EBML reader against a file it did not write.
//!
//! The unit tests build their own fixtures, which proves the reader
//! agrees with itself. This one opens Matroska produced by somebody
//! else's muxer — the only way to find out whether the encoding was
//! understood or merely reimplemented consistently.
//!
//! `#[ignore]` because it needs `ffmpeg` present to make the fixture,
//! the same convention the GPU gates follow:
//!
//! ```sh
//! cargo test -p sovmr-mkv --test real_file -- --ignored --nocapture
//! ```

use std::process::Command;

use sovmr_mkv::demux::TrackKind;
use sovmr_mkv::ebml::{elements, find, string, uint};
use sovmr_mkv::{is_matroska, EBML_HEADER_ID};

/// `Segment`, the master element holding everything but the header.
const SEGMENT: u32 = 0x1853_8067;
/// `Info`, and the `TimestampScale` inside it.
const INFO: u32 = 0x1549_A966;
const TIMESTAMP_SCALE: u32 = 0x002A_D7B1;
/// `Tracks` and one `TrackEntry`.
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const LANGUAGE: u32 = 0x0022_B59C;
/// `DocType` inside the EBML header.
const DOC_TYPE: u32 = 0x4282;
/// Track types: 1 = video, 2 = audio.
const TRACK_TYPE_VIDEO: u64 = 1;
const TRACK_TYPE_AUDIO: u64 = 2;
/// Languages the fixture tags its audio tracks with.
const LANGUAGES: [&str; 3] = ["eng", "rus", "deu"];

/// How many video frames ffprobe can actually decode out of a file.
fn ffprobe_frame_count(path: &std::path::Path) -> usize {
    let out = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe present");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .trim_end_matches(',')
        .parse()
        .unwrap_or(0)
}

/// How many packets ffprobe sees on a given Matroska track number.
///
/// ffprobe indexes streams from zero in file order while Matroska numbers
/// tracks from one, so the mapping is `number - 1` for files written this
/// way — which the fixture is, being freshly muxed in track order.
fn ffprobe_packet_count(path: &std::path::Path, track_number: u64) -> usize {
    let out = Command::new("ffprobe")
        .args(["-hide_banner", "-loglevel", "error", "-select_streams"])
        .arg(format!("{}", track_number - 1))
        .args([
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .expect("ffprobe present");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .trim_end_matches(',')
        .parse()
        .unwrap_or_else(|e| {
            panic!(
                "ffprobe said {:?}: {e}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
}

fn fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let clip = dir.join("real.mkv");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=3:size=320x240:rate=30",
    ]);
    for i in 0..LANGUAGES.len() {
        cmd.args(["-f", "lavfi", "-i"]);
        cmd.arg(format!("sine=frequency={}:duration=3", 220 + 110 * i));
    }
    cmd.args(["-map", "0:v"]);
    for (i, lang) in LANGUAGES.iter().enumerate() {
        cmd.arg("-map");
        cmd.arg(format!("{}:a", i + 1));
        cmd.arg(format!("-metadata:s:a:{i}"));
        cmd.arg(format!("language={lang}"));
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libopus"]);
    let status = cmd
        .arg(&clip)
        .status()
        .expect("ffmpeg present (fixture generation only)");
    assert!(status.success());
    clip
}

#[test]
#[ignore = "needs ffmpeg to write the fixture"]
fn a_matroska_file_written_by_somebody_else_reads_back_whole() {
    let dir = std::env::temp_dir().join(format!("sovmr-mkv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input_path = fixture(&dir);
    let bytes = std::fs::read(&input_path).unwrap();

    assert!(is_matroska(&bytes), "ffmpeg wrote an EBML document");

    // Top level: the EBML header, then the Segment. Every element must
    // parse — a reader that quietly skipped what it did not understand
    // would pass this test while being wrong.
    let top: Vec<_> = elements(&bytes)
        .map(|e| e.expect("every top-level element parses"))
        .collect();
    let ids: Vec<u32> = top.iter().map(|e| e.id).collect();
    assert_eq!(
        ids,
        vec![EBML_HEADER_ID, SEGMENT],
        "a Matroska file is a header followed by a segment"
    );

    let header = top[0];
    assert_eq!(
        string(
            find(header.payload, DOC_TYPE)
                .expect("DocType present")
                .payload
        ),
        "matroska",
        "DocType must read as text, not as bytes that look like text"
    );

    // The Segment is written with an unknown size when ffmpeg is
    // streaming and a real one when it can seek back. Either is legal,
    // and both must yield the same children.
    let segment = top[1];
    let scale = uint(
        find(
            find(segment.payload, INFO).expect("Info present").payload,
            TIMESTAMP_SCALE,
        )
        .expect("TimestampScale present")
        .payload,
    );
    assert_eq!(
        scale, 1_000_000,
        "the conventional nanosecond scale, read as a big-endian integer"
    );

    // Four tracks: one video, three audio, each with the language it was
    // tagged with. This is the shape the whole milestone is about.
    let tracks = find(segment.payload, TRACKS).expect("Tracks present");
    let entries: Vec<_> = elements(tracks.payload)
        .map(|e| e.expect("every TrackEntry parses"))
        .filter(|e| e.id == TRACK_ENTRY)
        .collect();
    assert_eq!(entries.len(), 1 + LANGUAGES.len());

    let mut audio_languages = Vec::new();
    let mut video_seen = 0;
    for entry in &entries {
        let kind = uint(find(entry.payload, TRACK_TYPE).expect("TrackType").payload);
        let number = uint(
            find(entry.payload, TRACK_NUMBER)
                .expect("TrackNumber")
                .payload,
        );
        assert!(number > 0, "track numbers are one-based");
        let codec = string(find(entry.payload, CODEC_ID).expect("CodecID").payload);
        match kind {
            TRACK_TYPE_VIDEO => {
                video_seen += 1;
                assert_eq!(codec, "V_MPEG4/ISO/AVC");
            }
            TRACK_TYPE_AUDIO => {
                assert_eq!(codec, "A_OPUS");
                audio_languages.push(
                    find(entry.payload, LANGUAGE)
                        .map(|l| string(l.payload).to_string())
                        .unwrap_or_else(|| "eng".to_string()),
                );
            }
            other => panic!("unexpected track type {other}"),
        }
    }
    assert_eq!(video_seen, 1);
    assert_eq!(
        audio_languages, LANGUAGES,
        "every audio track's language survives, in file order"
    );

    // Now the same file through the demuxer that will actually be used,
    // rather than through element lookups written for this test. The two
    // must agree; if they disagree, one of them is inventing.
    let seg = sovmr_mkv::demux::read_segment(&bytes).expect("read_segment");
    assert_eq!(seg.timestamp_scale, scale);
    assert!(!seg.truncated);
    assert_eq!(seg.tracks.len(), entries.len());
    assert_eq!(seg.of_kind(TrackKind::Video).count(), 1);

    let by_demux: Vec<&str> = seg.of_kind(TrackKind::Audio).map(|t| t.language).collect();
    assert_eq!(
        by_demux, LANGUAGES,
        "the demuxer must read the same languages the raw walk found"
    );
    for t in seg.of_kind(TrackKind::Audio) {
        assert_eq!(t.codec_id, "A_OPUS");
        assert_eq!(t.channels, 1, "the sine fixtures are mono");
        assert_eq!(t.sample_rate, 48_000, "Opus is always 48 kHz");
        assert!(t.number > 0);
    }
    let video = seg.of_kind(TrackKind::Video).next().unwrap();
    assert_eq!((video.width, video.height), (320, 240));
    assert!(
        !video.codec_private.is_empty(),
        "an H.264 track carries its avcC in CodecPrivate"
    );
    assert_eq!(
        video.codec_private[0], 1,
        "an avcC record starts with configurationVersion 1"
    );

    // The frames themselves, counted against an independent authority.
    // Our own reader agreeing with our own reader proves nothing; ffprobe
    // counting the same packets does.
    let (frames, err) = sovmr_mkv::blocks::read_frames(seg.payload);
    assert_eq!(err, None, "a whole file must read without a complaint");
    assert!(!frames.is_empty());

    for t in &seg.tracks {
        let mine = sovmr_mkv::blocks::track_frames(&frames, t.number).count();
        let theirs = ffprobe_packet_count(&input_path, t.number);
        assert_eq!(
            mine, theirs,
            "track {} ({}): our frame count must match ffprobe's packet count",
            t.number, t.codec_id
        );
    }

    // Video keyframes must be found, or seeking would be impossible.
    let video_keys = sovmr_mkv::blocks::track_frames(&frames, video.number)
        .filter(|f| f.keyframe)
        .count();
    assert!(video_keys > 0, "an H.264 track has sync frames");

    // Timestamps must advance. A reader that mislaid the cluster
    // timestamp would pile every frame onto instant zero.
    let stamps: Vec<i64> = sovmr_mkv::blocks::track_frames(&frames, video.number)
        .map(|f| f.timestamp)
        .collect();
    assert!(
        stamps.iter().max() > stamps.iter().min(),
        "timestamps must span the clip, not collapse to one instant"
    );

    // The frames must become something a decoder accepts. Writing Annex
    // B that only our own reader likes would prove nothing, so the
    // result is handed to ffmpeg to decode for real.
    let prefix = sovmr_mkv::codec::nal_length_size(video).expect("avcC declares a prefix width");
    assert!((1..=4).contains(&prefix), "got {prefix}");

    let mut annex_b = Vec::new();
    let mut stream = Vec::new();
    for f in sovmr_mkv::blocks::track_frames(&frames, video.number) {
        sovmr_mkv::codec::write_frame(video, f.data, f.keyframe, &mut annex_b)
            .expect("every frame converts");
        stream.extend_from_slice(&annex_b);
    }
    let raw = dir.join("video.h264");
    std::fs::write(&raw, &stream).unwrap();

    let decoded = Command::new("ffmpeg")
        .args(["-hide_banner", "-v", "error", "-i"])
        .arg(&raw)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg present");
    let complaints = String::from_utf8_lossy(&decoded.stderr);
    assert!(
        complaints.trim().is_empty(),
        "our Annex B must decode without complaint, got: {complaints}"
    );
    let count = ffprobe_frame_count(&raw);
    assert_eq!(
        count,
        sovmr_mkv::blocks::track_frames(&frames, video.number).count(),
        "every frame we wrote must decode back"
    );
    println!(
        "ANNEX B FROM MATROSKA: {} bytes, {count} frames decoded clean",
        stream.len()
    );

    println!(
        "REAL MATROSKA FRAMES: {} total, {} video keyframes, span {}..{} ticks",
        frames.len(),
        video_keys,
        stamps.iter().min().unwrap(),
        stamps.iter().max().unwrap()
    );
    println!(
        "REAL MATROSKA READ: {} bytes, {} tracks, languages {:?}, video {}x{}, avcC {} bytes",
        bytes.len(),
        seg.tracks.len(),
        by_demux,
        video.width,
        video.height,
        video.codec_private.len()
    );
    std::fs::remove_dir_all(&dir).ok();
}
