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

//! The muxer, checked by somebody else's reader.
//!
//! Our demuxer reading our muxer proves the two agree; it cannot prove
//! either is right. These gates take real frames out of a real Matroska
//! file, write them back out with our muxer, and hand the result to
//! `ffprobe` — which has no stake in our being correct.
//!
//! ```sh
//! cargo test -p sovmr-mkv --test write_read -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use sovmr_mkv::demux::TrackKind;
use sovmr_mkv::mux::{MkvMuxer, TrackSpec};

/// Languages the fixture tags its audio with.
const LANGUAGES: [&str; 3] = ["eng", "rus", "deu"];
/// Fixture length in seconds, and its frame rate.
const SECONDS: u32 = 4;
const FPS: u32 = 30;

fn fixture(dir: &Path) -> PathBuf {
    let clip = dir.join("in.mkv");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
    ]);
    cmd.arg(format!(
        "testsrc2=duration={SECONDS}:size=320x240:rate={FPS}"
    ));
    for i in 0..LANGUAGES.len() {
        cmd.args(["-f", "lavfi", "-i"]);
        cmd.arg(format!(
            "sine=frequency={}:duration={SECONDS}",
            220 + 110 * i
        ));
    }
    cmd.args(["-map", "0:v"]);
    for (i, lang) in LANGUAGES.iter().enumerate() {
        cmd.arg("-map");
        cmd.arg(format!("{}:a", i + 1));
        cmd.arg(format!("-metadata:s:a:{i}"));
        cmd.arg(format!("language={lang}"));
    }
    cmd.args([
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-c:a", "aac",
    ]);
    assert!(cmd.arg(&clip).status().expect("ffmpeg present").success());
    clip
}

/// One `ffprobe` field per stream, as CSV rows.
fn probe(path: &Path, entries: &str) -> Vec<String> {
    let out = Command::new("ffprobe")
        .args(["-hide_banner", "-loglevel", "error", "-show_entries"])
        .arg(entries)
        .args(["-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("ffprobe present");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().trim_end_matches(',').to_string())
        .collect()
}

#[test]
#[ignore = "needs ffmpeg to write the fixture and ffprobe to judge the result"]
fn a_matroska_file_we_wrote_is_read_by_a_stranger() {
    let dir = std::env::temp_dir().join(format!("sovmr-mkvmux-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = std::fs::read(fixture(&dir)).unwrap();

    let segment = sovmr_mkv::demux::read_segment(&bytes).expect("fixture reads");
    let (frames, err) = sovmr_mkv::blocks::read_frames(segment.payload);
    assert_eq!(err, None);

    // Restate every source track for the muxer, unchanged: this is a
    // remux, so anything that differs on the far side is our doing.
    let specs: Vec<TrackSpec> = segment
        .tracks
        .iter()
        .filter(|t| matches!(t.kind, TrackKind::Video | TrackKind::Audio))
        .map(|t| TrackSpec {
            number: t.number,
            codec_id: t.codec_id.to_string(),
            codec_private: t.codec_private.to_vec(),
            video: t.kind == TrackKind::Video,
            subtitle: t.kind == TrackKind::Subtitle,
            width: t.width,
            height: t.height,
            sample_rate: t.sample_rate,
            channels: t.channels,
            language: t.language.to_string(),
            name: t.name.map(str::to_string),
            enabled: t.enabled,
            colour: None,
        })
        .collect();
    assert_eq!(specs.len(), 1 + LANGUAGES.len());

    let out_path = dir.join("out.mkv");
    let mut muxer = MkvMuxer::create(&out_path, &specs).expect("muxer starts");
    // Source ticks are milliseconds here; convert through the segment's
    // own scale rather than assuming it.
    let to_ms = |ticks: i64| ticks * segment.timestamp_scale as i64 / 1_000_000;
    let nominal_ms = i64::from(1000 / FPS);
    for f in &frames {
        muxer
            .add_frame(
                u64::from(f.track),
                to_ms(f.timestamp),
                nominal_ms,
                f.keyframe,
                f.data,
            )
            .expect("frame writes");
    }
    let written = muxer.finalize().expect("finalize");
    assert!(written > 0);

    // ---- our own reader first ---------------------------------------
    let out = std::fs::read(&out_path).unwrap();
    assert!(sovmr_mkv::is_matroska(&out));
    let back = sovmr_mkv::demux::read_segment(&out).expect("our own file reads");
    assert_eq!(back.tracks.len(), specs.len());
    let langs: Vec<&str> = back.of_kind(TrackKind::Audio).map(|t| t.language).collect();
    assert_eq!(langs, LANGUAGES, "languages survive the round trip");
    let (back_frames, err) = sovmr_mkv::blocks::read_frames(back.payload);
    assert_eq!(err, None, "our own clusters must parse");
    assert_eq!(
        back_frames.len(),
        frames.len(),
        "every frame written must read back"
    );

    // ---- then a reader with no stake in our being right --------------
    let codecs: Vec<String> = probe(&out_path, "stream=codec_name")
        .into_iter()
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        codecs.len(),
        specs.len(),
        "ffprobe must find every track we wrote, got {codecs:?}"
    );
    assert_eq!(codecs[0], "h264");
    assert!(codecs[1..].iter().all(|c| c == "aac"), "{codecs:?}");

    // Languages, matched to their streams rather than to row numbers.
    // ffmpeg omits a tag that equals the format's own default, so a
    // row-per-stream assumption silently shifts every answer by one —
    // which is exactly how this assertion first failed, on a muxer that
    // was already correct.
    let rows = probe(&out_path, "stream=codec_type:stream_tags=language");
    let audio_langs: Vec<String> = rows
        .iter()
        .filter_map(|r| {
            let mut f = r.split(',');
            let kind = f.next()?;
            (kind == "audio").then(|| f.next().unwrap_or("").to_string())
        })
        .collect();
    assert_eq!(
        audio_langs, LANGUAGES,
        "ffprobe must read the languages we wrote, got rows {rows:?}"
    );

    // Packet counts, per track, from a decoder that parsed our clusters.
    for (i, spec) in specs.iter().enumerate() {
        let counted = Command::new("ffprobe")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-select_streams",
                &i.to_string(),
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
            ])
            .arg(&out_path)
            .output()
            .expect("ffprobe present");
        let theirs: usize = String::from_utf8_lossy(&counted.stdout)
            .trim()
            .trim_end_matches(',')
            .parse()
            .unwrap_or(0);
        let mine = frames
            .iter()
            .filter(|f| u64::from(f.track) == spec.number)
            .count();
        assert_eq!(theirs, mine, "track {} packet count", spec.number);
    }

    // And the video must actually decode out of our container.
    let decoded = Command::new("ffmpeg")
        .args(["-hide_banner", "-v", "error", "-i"])
        .arg(&out_path)
        .args(["-map", "0:v", "-f", "null", "-"])
        .output()
        .expect("ffmpeg present");
    let complaints = String::from_utf8_lossy(&decoded.stderr);
    assert!(
        complaints.trim().is_empty(),
        "video must decode out of our Matroska, got: {complaints}"
    );

    println!(
        "MATROSKA WRITTEN AND READ BY FFPROBE: {written} bytes, {} tracks {:?}, {} frames",
        codecs.len(),
        audio_langs,
        back_frames.len()
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_frame_index_stays_the_size_it_was_measured_at() {
    // Matroska has no index, so every frame in the file has to be
    // walked and remembered before the first is handed over. Measured on
    // a real file by `examples/weigh.rs`: **288 frames per second** for
    // one video and six audio tracks, which over twelve hours is twelve
    // and a half million of them.
    //
    // The size of this struct is therefore multiplied by twelve and a
    // half million. It was 40 bytes — 498 MB of index for a film — and
    // narrowing the track number and letting the keyframe flag share its
    // word took it to 32, which is 100 MB back.
    //
    // This is not a style preference, and it is not free to get wrong in
    // the other direction either: a field added carelessly here costs a
    // hundred megabytes a word. So the number is asserted.
    assert_eq!(
        std::mem::size_of::<sovmr_mkv::blocks::Frame<'_>>(),
        32,
        "the frame index grew; a word here is 100 MB on a twelve-hour film"
    );
}

#[test]
fn a_track_number_past_what_the_index_carries_is_refused() {
    // The price of narrowing it. Truncating a track number would scatter
    // one track's frames across another — silently, and audibly. So a
    // file that uses a larger one is refused by name.
    //
    // A `SimpleBlock` whose track vint is four octets — marker 0x10 —
    // holding 65 537, then a timestamp, flags and one byte of payload.
    let block = [0x10, 0x01, 0x00, 0x01, 0x00, 0x00, 0x80, 0xAA];
    let cluster = sovmr_mkv::mux::element(0xE7, &[0x00]);
    let mut body = cluster;
    body.extend(sovmr_mkv::mux::element(0xA3, &block));
    let segment = sovmr_mkv::mux::element(0x1F43_B675, &body);

    let (frames, error) = sovmr_mkv::blocks::read_frames(&segment);
    assert!(
        frames.is_empty(),
        "a frame was filed under a wrapped number"
    );
    match error {
        Some(sovmr_mkv::blocks::BlockError::TrackTooLarge(n)) => {
            assert!(
                n > sovmr_mkv::blocks::MAX_TRACK_NUMBER,
                "{n} is not too large"
            );
        }
        other => panic!("expected a named refusal, got {other:?}"),
    }
}
