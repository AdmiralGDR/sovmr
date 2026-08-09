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

//! Weigh what reading a Matroska file's frames actually costs.
//!
//! The ~420 MB figure this project has been carrying around
//! for a twelve-hour file was an estimate nobody had checked. This is
//! the check — measured on a real file and on the process, not derived
//! from a struct's size and a guess at the frame count.

use std::time::Instant;

/// Resident set size of this process in bytes, from the kernel.
///
/// Linux only, and that is fine: this is a measuring instrument, not a
/// shipped feature. `statm` gives pages resident; the page size comes
/// from the same place rather than being assumed to be four kilobytes.
fn resident_bytes() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|f| f.parse().ok())
        .unwrap_or(0);
    // Pages, times the page size. Four kilobytes on every host this
    // project targets; a measurement that guessed differently would be
    // wrong by a constant factor and say so plainly in the totals.
    const PAGE: u64 = 4096;
    pages * PAGE
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: weigh <file.mkv>");
    let before_read = resident_bytes();
    let data = std::fs::read(&path).expect("a readable file");
    let after_read = resident_bytes();

    let segment = sovmr_mkv::demux::read_segment(&data).expect("a segment");
    let started = Instant::now();
    let (frames, error) = sovmr_mkv::blocks::read_frames(segment.payload);
    let elapsed = started.elapsed();
    let after_frames = resident_bytes();

    let each = std::mem::size_of::<sovmr_mkv::blocks::Frame<'_>>();
    let seconds = segment.duration_ticks * segment.timestamp_scale as f64 / 1e9;
    println!("file          {} MB", data.len() / 1_000_000);
    println!("duration      {seconds:.1} s");
    println!("frames        {}", frames.len());
    println!("Frame is      {each} bytes");
    println!(
        "index         {} MB (frames x size)",
        frames.len() * each / 1_000_000
    );
    println!(
        "capacity      {} MB (what the Vec asked for)",
        frames.capacity() * each / 1_000_000
    );
    println!(
        "rss: file     {} MB",
        (after_read - before_read) / 1_000_000
    );
    println!(
        "rss: index    {} MB",
        (after_frames - after_read) / 1_000_000
    );
    println!("walk took     {:.3} s", elapsed.as_secs_f64());
    if let Some(e) = error {
        println!("stopped at    {e:?}");
    }

    // What this says about the case that matters: a twelve-hour film
    // with a video track and six audio tracks.
    if seconds > 0.0 {
        let per_second = frames.len() as f64 / seconds;
        let twelve_hours = 12.0 * 3600.0;
        println!(
            "extrapolated  {:.0} MB for 12 h at this frame density",
            per_second * twelve_hours * each as f64 / 1e6
        );
    }
}
