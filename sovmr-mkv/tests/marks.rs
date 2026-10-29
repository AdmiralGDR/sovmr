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

//! Chapters and tags, in and out.
//!
//! Every element identifier these rely on was read out of a file ffmpeg
//! wrote, by this project's own EBML reader — `examples/walk.rs` is the
//! instrument. So the gate that matters is the one below: what we write
//! must be what a *third party* reads back, not merely what we read.

use std::path::{Path, PathBuf};
use std::process::Command;

use sovmr_mkv::demux::read_segment;
use sovmr_mkv::mux::{ChapterSpec, MkvMuxer, TagSpec, TrackSpec};

/// Scratch directory for this process.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sovmr-marks-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// A minimal track to hang the marks on: Matroska has no file without
/// one.
fn track() -> TrackSpec {
    TrackSpec {
        number: 1,
        codec_id: "A_AAC".to_string(),
        codec_private: vec![0x11, 0x88],
        video: false,
        subtitle: false,
        width: 0,
        height: 0,
        sample_rate: 48_000,
        channels: 1,
        language: "und".to_string(),
        name: None,
        enabled: true,
        colour: None,
    }
}

fn write(path: &Path, chapters: Vec<ChapterSpec>, tags: Vec<TagSpec>) {
    // Real frames, from this project's own encoder. A made-up payload
    // makes ffprobe complain about the *audio*, which would mask what
    // this gate is actually asking about.
    let frame = sovmr_audio::encode::encode_frame(&[0.0; 1024], 48_000).expect("a silent frame");
    let mut mux = MkvMuxer::create(path, &[track()]).expect("a muxer");
    mux.set_marks(chapters, tags).expect("set before any frame");
    // Enough to cover the chapters the probe declares, at the frame
    // length AAC actually uses.
    let per_frame_ms = 1024 * 1000 / 48_000;
    for n in 0..200u64 {
        let at = (n as i64) * per_frame_ms;
        mux.add_frame(1, at, per_frame_ms, true, &frame)
            .expect("a frame");
    }
    mux.finalize().expect("finalize");
}

fn probe_chapters() -> Vec<ChapterSpec> {
    vec![
        ChapterSpec {
            uid: 0x1234_5678,
            start_nanos: 0,
            end_nanos: 2_000_000_000,
            title: "Opening".to_string(),
            language: "eng".to_string(),
        },
        ChapterSpec {
            uid: 0x9ABC_DEF0,
            start_nanos: 2_000_000_000,
            end_nanos: 4_000_000_000,
            title: "Пролог".to_string(),
            language: "rus".to_string(),
        },
    ]
}

fn probe_tags() -> Vec<TagSpec> {
    vec![
        TagSpec {
            track_uid: None,
            name: "TITLE".to_string(),
            value: "A Probe".to_string(),
        },
        TagSpec {
            track_uid: None,
            name: "ARTIST".to_string(),
            value: "Nobody".to_string(),
        },
        TagSpec {
            track_uid: Some(0x4242),
            name: "LANGUAGE".to_string(),
            value: "und".to_string(),
        },
    ]
}

#[test]
fn what_we_write_is_what_we_read() {
    let path = scratch("round").join("marks.mkv");
    write(&path, probe_chapters(), probe_tags());
    let bytes = std::fs::read(&path).expect("the file we just wrote");
    let segment = read_segment(&bytes).expect("a segment");

    assert_eq!(segment.chapters.len(), 2);
    for (wrote, read) in probe_chapters().iter().zip(&segment.chapters) {
        assert_eq!(read.uid, wrote.uid, "the uid was not carried");
        assert_eq!(read.start_nanos, wrote.start_nanos);
        assert_eq!(read.end_nanos, wrote.end_nanos);
        assert_eq!(read.title, Some(wrote.title.as_str()));
        assert_eq!(read.language, Some(wrote.language.as_str()));
    }

    assert_eq!(segment.tags.len(), 3);
    for (wrote, read) in probe_tags().iter().zip(&segment.tags) {
        assert_eq!(read.track_uid, wrote.track_uid);
        assert_eq!(read.name, wrote.name);
        assert_eq!(read.value, wrote.value);
    }
}

#[test]
fn a_file_with_no_marks_declares_none() {
    // An empty chapter list is not the same as no chapter list: an
    // edition with no atoms makes some readers show a film with one
    // nameless chapter, which is worse than showing none.
    let path = scratch("bare").join("bare.mkv");
    write(&path, Vec::new(), Vec::new());
    let bytes = std::fs::read(&path).expect("the file");
    let segment = read_segment(&bytes).expect("a segment");
    assert!(segment.chapters.is_empty());
    assert!(segment.tags.is_empty());
    // And the elements are absent outright, not present and empty.
    for id in [sovmr_mkv::ids::CHAPTERS, sovmr_mkv::ids::TAGS] {
        assert!(
            sovmr_mkv::ebml::find(segment.payload, id).is_none(),
            "0x{id:X} was written empty"
        );
    }
}

#[test]
#[ignore = "needs ffmpeg to judge"]
fn a_third_party_reads_back_what_we_wrote() {
    // The gate that matters. Reading our own file proves the two halves
    // agree; only somebody else's reader proves either is right.
    let path = scratch("third").join("marks.mkv");
    write(&path, probe_chapters(), probe_tags());
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_chapters",
            "-show_entries",
            "format_tags",
        ])
        .arg(&path)
        .output()
        .expect("ffprobe runs");
    assert!(
        out.status.success(),
        "ffprobe refused the file: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "ffprobe complained: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    // The marks themselves, by their own values — including one in a
    // script that would not survive a byte-for-byte assumption about
    // encoding.
    for wanted in [
        "start=0",
        "end=2000000000",
        "start=2000000000",
        "end=4000000000",
        "TAG:title=Opening",
        "TAG:title=Пролог",
        "TAG:TITLE=A Probe",
        "TAG:ARTIST=Nobody",
    ] {
        assert!(
            text.contains(wanted),
            "ffprobe did not report {wanted}:\n{text}"
        );
    }
}

#[test]
fn marks_set_after_the_first_frame_are_refused_rather_than_lost() {
    // The defect this rules out, stated as a test. Written after the
    // clusters, our own reader found the chapters and ffprobe did not —
    // a file whose marks only this project can see is a file that has
    // lost them. So a late call is an error, not a quiet nothing.
    let path = scratch("late").join("late.mkv");
    let frame = sovmr_audio::encode::encode_frame(&[0.0; 1024], 48_000).expect("a frame");
    let mut mux = MkvMuxer::create(&path, &[track()]).expect("a muxer");
    mux.add_frame(1, 0, 21, true, &frame).expect("a frame");
    let refused = mux.set_marks(probe_chapters(), probe_tags());
    assert!(refused.is_err(), "a late call was accepted");
    mux.finalize().expect("finalize");
}

#[test]
fn a_survey_says_whether_a_file_has_chapters() {
    // Until the reader parsed the `Chapters` element the survey reported
    // a flat `false`, which was honest while nothing looked and became a
    // lie the moment something did. It now answers from the file.
    let with = scratch("survey").join("with.mkv");
    write(&with, probe_chapters(), Vec::new());
    let bytes = std::fs::read(&with).expect("the file");
    assert!(
        sovmr_mkv::probe::summarise(&bytes)
            .expect("a summary")
            .chapters
    );

    let without = scratch("survey").join("without.mkv");
    write(&without, Vec::new(), Vec::new());
    let bytes = std::fs::read(&without).expect("the file");
    assert!(
        !sovmr_mkv::probe::summarise(&bytes)
            .expect("a summary")
            .chapters
    );
}
