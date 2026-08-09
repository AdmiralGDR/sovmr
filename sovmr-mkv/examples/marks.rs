// SPDX-License-Identifier: AGPL-3.0-or-later
//! Print a file's chapters and tags, read by our own demuxer.

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(path).unwrap();
    let seg = sovmr_mkv::demux::read_segment(&data).unwrap();
    println!("title {:?}", seg.title);
    for c in &seg.chapters {
        println!(
            "chapter {} {}..{} {:?} {:?}",
            c.uid, c.start_nanos, c.end_nanos, c.title, c.language
        );
    }
    for t in &seg.tags {
        println!("tag {:?} {} = {}", t.track_uid, t.name, t.value);
    }
}
