// SPDX-License-Identifier: GPL-3.0-or-later
//! Hostile-input gate for the Matroska reader: a file this crate wrote is
//! truncated at every length and randomly mutated thousands of times, and
//! every public entry point a hostile file can reach must return (with or
//! without an error) instead of panicking. Deterministic: a fixed-seed
//! xorshift PRNG, no extra crates, no fixtures on disk beyond a temp file.

use sovmr_mkv::blocks::read_frames;
use sovmr_mkv::demux::read_segment;
use sovmr_mkv::mux::{MkvMuxer, TrackSpec};
use sovmr_mkv::{ebml, is_matroska, probe};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn spec(number: u64, codec_id: &str, video: bool) -> TrackSpec {
    TrackSpec {
        number,
        codec_id: codec_id.to_string(),
        codec_private: if video { vec![1, 2, 3, 4] } else { Vec::new() },
        video,
        subtitle: false,
        width: if video { 64 } else { 0 },
        height: if video { 48 } else { 0 },
        sample_rate: if video { 0 } else { 48_000 },
        channels: if video { 0 } else { 2 },
        language: "eng".to_string(),
        name: Some(format!("track {number}")),
        enabled: true,
        colour: None,
    }
}

fn seed_file(tag: &str) -> Vec<u8> {
    // One file per caller: tests in this binary run in parallel.
    let path = std::env::temp_dir().join(format!(
        "sovmr-mkv-hostile-{}-{tag}.mkv",
        std::process::id()
    ));
    let tracks = [spec(1, "V_AV1", true), spec(2, "A_OPUS", false)];
    let mut muxer = MkvMuxer::create_titled(&path, &tracks, Some("hostile-input")).expect("create");
    for i in 0..12i64 {
        muxer
            .add_frame(
                1,
                i * 40,
                40,
                i % 6 == 0,
                &[0x12, 0x00, i as u8, 0xAB, 0xCD],
            )
            .expect("video frame");
        muxer
            .add_frame(2, i * 40, 20, true, &[0xFC, i as u8, 0x01])
            .expect("audio frame");
    }
    muxer.finalize().expect("finalize");
    let bytes = std::fs::read(&path).expect("read back");
    let _ = std::fs::remove_file(&path);
    bytes
}

fn exercise(file: &[u8]) {
    let _ = is_matroska(file);
    for el in ebml::elements(file).take(256) {
        let _ = el;
    }
    let _ = probe::summarise(file);
    if let Ok(segment) = read_segment(file) {
        let _ = segment.ticks_to_nanos(i64::MAX);
        let _ = segment.ticks_to_nanos(i64::MIN);
        for track in &segment.tracks {
            let _ = sovmr_mkv::codec::nal_length_size(track);
        }
        let (frames, _err) = read_frames(segment.payload);
        let _ = frames.len();
    }
}

#[test]
fn the_seed_is_a_readable_file() {
    let file = seed_file("seed");
    assert!(is_matroska(&file));
    let segment = read_segment(&file).expect("seed parses");
    assert_eq!(segment.tracks.len(), 2);
    let (frames, err) = read_frames(segment.payload);
    assert!(err.is_none(), "{err:?}");
    assert_eq!(frames.len(), 24);
}

#[test]
fn mutated_and_truncated_files_never_panic() {
    let seed = seed_file("mutate");
    for len in 0..=seed.len() {
        exercise(&seed[..len]);
    }
    let mut rng = Rng(0x0BAD_5EED_1234_5678);
    for _ in 0..20_000 {
        let mut f = seed.clone();
        for _ in 0..=rng.below(8) {
            let i = rng.below(f.len());
            f[i] = match rng.below(4) {
                0 => 0x00,
                1 => 0xFF,
                2 => f[i] ^ (1 << rng.below(8)),
                _ => rng.next() as u8,
            };
        }
        let cut = f.len() - rng.below(f.len() / 4);
        exercise(&f[..cut]);
    }
}
