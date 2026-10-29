// SPDX-License-Identifier: GPL-3.0-or-later
//
// ISOBMFF input the way sovmr-native opens it: video track, audio
// tracks, extras, then every sample read and converted to Annex B.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sovmr_mp4::{boxes, demux, file_kind};

/// Sample tables can claim billions of entries; stop well before that.
const MAX_SAMPLES: usize = 4096;

fuzz_target!(|data: &[u8]| {
    let _ = file_kind(data);
    for b in boxes(data).flatten() {
        for _ in b.children().flatten() {}
    }
    let _ = demux::survey_extras(data);
    if let Ok(audio) = demux::find_audio_tracks(data) {
        let _ = audio.carried_bitrate();
        let _ = audio.total_bitrate();
        for t in audio.tracks.iter() {
            let _ = t.sample_count();
            let _ = t.chunk_count();
            let _ = t.total_bytes();
            let _ = t.bitrate();
            let _ = t.duration();
            for s in t.samples(data).take(MAX_SAMPLES) {
                if s.is_err() {
                    break;
                }
            }
        }
    }
    let Ok(track) = demux::find_video_track(data) else {
        return;
    };
    let _ = track.sample_count();
    let _ = track.total_bytes();
    let _ = track.bitrate();
    let _ = track.frame_rate();
    let _ = track.duration();
    for ps in track.parameter_sets() {
        if ps.is_err() {
            break;
        }
    }
    let mut out = Vec::new();
    for s in track.samples(data).take(MAX_SAMPLES) {
        let Ok(s) = s else { break };
        let _ = track.write_frame(&s, &mut out);
    }
});
