// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Matroska input the way sovmr-native opens it: segment, audio plan,
// clusters, then every video frame converted to Annex B.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sovmr_mkv::{blocks, codec, demux, probe, source};

fuzz_target!(|data: &[u8]| {
    let _ = probe::summarise(data);
    let Ok(segment) = demux::read_segment(data) else {
        return;
    };
    let _ = source::plan_audio(&segment, source::Target::Mkv);
    let _ = source::plan_audio(&segment, source::Target::Mp4);
    let (frames, _err) = blocks::read_frames(segment.payload);
    let mut out = Vec::new();
    for track in segment.tracks.iter() {
        if codec::video_codec(track.codec_id).is_some() {
            let _ = codec::nal_length_size(track);
            for f in blocks::track_frames(&frames, track.number) {
                let _ = segment.ticks_to_nanos(f.timestamp);
                let _ = codec::write_frame(track, f.data, f.keyframe, &mut out);
            }
        }
    }
});
