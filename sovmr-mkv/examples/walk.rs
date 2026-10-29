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

//! Print a Matroska file's element tree, read by our own EBML reader.
//!
//! The measuring instrument for anything this project has not carried
//! yet: point it at a file some other tool wrote and read the element
//! identifiers out of it, rather than out of memory or a web page. Stage
//! 3 already paid for the alternative — a published answer about EBML
//! lengths was wrong, and only the file settled it.

use sovmr_mkv::ebml::elements;

/// How deep to descend. Clusters nest two or three deep; past this is a
/// payload that merely happens to parse.
const MAX_DEPTH: usize = 6;
/// Bytes of payload to show as text.
const PREVIEW: usize = 32;

fn walk(data: &[u8], depth: usize) {
    if depth > MAX_DEPTH {
        return;
    }
    for element in elements(data).flatten() {
        let preview: String = element
            .payload
            .iter()
            .take(PREVIEW)
            .map(|b| {
                if b.is_ascii_graphic() || *b == b' ' {
                    *b as char
                } else {
                    '.'
                }
            })
            .collect();
        println!(
            "{:indent$}0x{:X} ({} bytes) {preview}",
            "",
            element.id,
            element.payload.len(),
            indent = depth * 2
        );
        // Descend into anything whose payload parses cleanly as
        // children. A leaf occasionally does by chance; the depth bound
        // and the printed sizes make that obvious rather than confusing.
        let children: Vec<_> = elements(element.payload).take(4).collect();
        if !children.is_empty() && children.iter().all(Result::is_ok) {
            walk(element.payload, depth + 1);
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: walk <file.mkv>");
    let data = std::fs::read(path).expect("a readable file");
    walk(&data, 0);
}
