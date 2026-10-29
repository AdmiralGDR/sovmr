// SPDX-License-Identifier: GPL-3.0-or-later
//
// The bare EBML reader: walk every element that parses, descending into
// payloads, and read each one as uint, float and string.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sovmr_mkv::ebml;

fn walk(data: &[u8], depth: usize) {
    if depth > 8 {
        return;
    }
    for element in ebml::elements(data).flatten() {
        let _ = ebml::uint(element.payload);
        let _ = ebml::float(element.payload);
        let _ = ebml::string(element.payload);
        walk(element.payload, depth + 1);
    }
}

fuzz_target!(|data: &[u8]| {
    let _ = sovmr_mkv::is_matroska(data);
    let _ = ebml::find(data, sovmr_mkv::ids::SEGMENT);
    walk(data, 0);
});
