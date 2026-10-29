// SPDX-License-Identifier: GPL-3.0-or-later
//
// HEVC Annex B helpers and the colr reader. The pipeline runs
// extract_parameter_sets on every encoded packet, and parse_sps on
// whatever SPS it finds there.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sovmr_mp4::{colour::Colour, demux, hevc};

fuzz_target!(|data: &[u8]| {
    for nal in hevc::annex_b_nals(data) {
        let _ = hevc::nal_type(nal);
        let _ = hevc::parse_sps(nal);
    }
    if let Some(ps) = hevc::extract_parameter_sets(data) {
        let _ = hevc::hvcc_record(&ps);
    }
    let mut out = Vec::new();
    hevc::length_prefixed_sample(data, &mut out);
    let _ = hevc::parse_sps(data);
    let _ = Colour::from_colr(data);
    for codec in [demux::VideoCodec::H264, demux::VideoCodec::Hevc] {
        let _ = demux::nal_length_size(codec, data);
        for ps in demux::config_parameter_sets(codec, data) {
            if ps.is_err() {
                break;
            }
        }
        for prefix in [1u8, 2, 4] {
            out.clear();
            let _ = demux::annex_b_from_length_prefixed(data, prefix, 0, &mut out);
        }
    }
});
