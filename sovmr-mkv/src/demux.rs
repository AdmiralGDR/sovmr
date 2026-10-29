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

//! What a Matroska file declares it contains: the segment's clock, and
//! every track with its identity.
//!
//! This is the counterpart of `sovmr_mp4::demux`'s track discovery, and
//! it answers the same question the milestone cares about — *how many
//! audio tracks are there, and who is each one?* Packet reading comes
//! next; nothing here touches a frame.

use crate::ebml::{self, EbmlError};
use crate::ids;

/// Hard ceiling on tracks taken from one file.
///
/// Bounded execution: a crafted `Tracks` element can declare arbitrarily
/// many entries, and every one of them would cost an allocation. The
/// figure matches [`sovmr_mp4`]'s audio ceiling for the same reason —
/// far above any real multi-language master, finite against a hostile
/// one.
///
/// [`sovmr_mp4`]: ../../sovmr_mp4/index.html
pub const MAX_TRACKS: usize = 64;

/// Language a track claims when it says nothing.
///
/// The format's own default, not an invention: a `TrackEntry` with no
/// `Language` element *is* English by specification.
pub const DEFAULT_LANGUAGE: &str = "eng";

/// Why a Matroska file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MkvError {
    /// The EBML underneath was malformed.
    Ebml(EbmlError),
    /// No `Segment` element — not a Matroska file, whatever it is.
    MissingSegment,
    /// No `Tracks` element, so the file describes no media at all.
    MissingTracks,
}

impl std::fmt::Display for MkvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MkvError::Ebml(e) => write!(f, "ebml: {e}"),
            MkvError::MissingSegment => f.write_str("no Segment element"),
            MkvError::MissingTracks => f.write_str("no Tracks element"),
        }
    }
}

impl std::error::Error for MkvError {}

impl From<EbmlError> for MkvError {
    fn from(e: EbmlError) -> Self {
        MkvError::Ebml(e)
    }
}

/// What a track carries, as far as sovmr distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    /// Pictures.
    Video,
    /// Sound.
    Audio,
    /// Subtitles or timed text.
    Subtitle,
    /// Something else entirely (buttons, control tracks, metadata).
    Other,
}

impl TrackKind {
    fn from_code(code: u64) -> TrackKind {
        match code {
            ids::TYPE_VIDEO => TrackKind::Video,
            ids::TYPE_AUDIO => TrackKind::Audio,
            ids::TYPE_SUBTITLE => TrackKind::Subtitle,
            _ => TrackKind::Other,
        }
    }
}

/// One track's declaration.
///
/// Everything borrows from the mapped file — a name or codec-private
/// blob is a slice, never a copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track<'a> {
    /// `TrackNumber`, the one-based key blocks refer to their track by.
    pub number: u64,
    /// What this track carries.
    pub kind: TrackKind,
    /// Matroska codec identifier, e.g. `V_MPEG4/ISO/AVC`, `A_OPUS`.
    pub codec_id: &'a str,
    /// Codec setup bytes (`avcC` / `hvcC` / `av1C` / `AudioSpecificConfig`),
    /// verbatim. Empty when the codec needs none.
    pub codec_private: &'a [u8],
    /// Language: BCP 47 when the file gives one, else ISO 639-2, else
    /// [`DEFAULT_LANGUAGE`].
    pub language: &'a str,
    /// Human title, when the file names the track.
    pub name: Option<&'a str>,
    /// Whether the track is enabled for playback.
    pub enabled: bool,
    /// Whether it is the default choice for its kind.
    pub default: bool,
    /// Whether a player must show it regardless of preference.
    pub forced: bool,
    /// Nanoseconds of encoder priming to discard at the start.
    pub codec_delay: u64,
    /// Nominal frame duration in nanoseconds, when declared.
    pub default_duration: Option<u64>,
    /// Coded pixel size, for video tracks.
    pub width: u32,
    /// Coded pixel height, for video tracks.
    pub height: u32,
    /// What the track said its colours mean, when it said anything.
    pub colour: Option<sovmr_mp4::colour::Colour>,
    /// Sampling frequency in Hz, for audio tracks.
    pub sample_rate: u32,
    /// Channel count, for audio tracks.
    pub channels: u16,
}

/// One chapter mark: where it starts and what it is called.
///
/// # The units, measured rather than assumed
///
/// `start_nanos` and `end_nanos` are plain nanoseconds, **not** segment
/// ticks. A probe file whose second chapter began at two seconds held
/// 2 000 000 000 with a `TimestampScale` of a million; scaling them
/// would have placed every mark a thousand times late, and nothing
/// short of a player would have said so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chapter<'a> {
    /// The file's own identifier for this chapter, carried through so a
    /// re-mux does not invent new ones.
    pub uid: u64,
    /// Start and end, in nanoseconds from the beginning.
    pub start_nanos: u64,
    /// Zero when the file did not say, which is legal: a chapter runs
    /// until the next one begins.
    pub end_nanos: u64,
    /// What it is called, in the first language the file gives.
    pub title: Option<&'a str>,
    /// That language, as the file spelled it.
    pub language: Option<&'a str>,
}

/// One descriptive entry: a name, a value, and what it is about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag<'a> {
    /// The track this describes, or `None` for the whole file.
    pub track_uid: Option<u64>,
    /// The entry's name, as written — `TITLE`, `ARTIST`, and the rest.
    pub name: &'a str,
    /// Its value.
    pub value: &'a str,
}

/// A Matroska file's declared contents.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment<'a> {
    /// Nanoseconds per timestamp tick — the clock every block is in.
    pub timestamp_scale: u64,
    /// Duration in ticks, when declared. Streamed files often omit it.
    pub duration_ticks: f64,
    /// Segment title, when named.
    pub title: Option<&'a str>,
    /// Every declared track, in file order.
    pub tracks: Vec<Track<'a>>,
    /// Chapter marks, in file order.
    ///
    /// Read because a viewer who loses them has lost something they can
    /// see is gone — the same reason subtitles are carried.
    pub chapters: Vec<Chapter<'a>>,
    /// Descriptive entries, file-wide and per track.
    pub tags: Vec<Tag<'a>>,
    /// Whether [`MAX_TRACKS`] cut the list short.
    pub truncated: bool,
    /// The `Segment` element's payload, for the packet reader to walk.
    pub payload: &'a [u8],
}

impl<'a> Segment<'a> {
    /// Tracks of one kind, in file order.
    pub fn of_kind(&self, kind: TrackKind) -> impl Iterator<Item = &Track<'a>> {
        self.tracks.iter().filter(move |t| t.kind == kind)
    }

    /// Convert a block timestamp to nanoseconds.
    pub fn ticks_to_nanos(&self, ticks: i64) -> i64 {
        ticks.saturating_mul(self.timestamp_scale as i64)
    }
}

/// Matroska's default `TimestampScale`: one millisecond per tick.
///
/// Applied when the file omits the element, which the specification
/// permits — the value is a default, not a missing fact.
const DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;

/// Read a file's `Segment`: its clock and its tracks.
///
/// # Errors
///
/// [`MkvError::MissingSegment`] when there is no segment, and
/// [`MkvError::MissingTracks`] when it declares no tracks. Malformed
/// EBML inside a single `TrackEntry` costs that track, not the file.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), sovmr_mkv::demux::MkvError> {
/// # let bytes: Vec<u8> = Vec::new();
/// if sovmr_mkv::is_matroska(&bytes) {
///     let segment = sovmr_mkv::demux::read_segment(&bytes)?;
///     println!("{} tracks", segment.tracks.len());
/// }
/// # Ok(())
/// # }
/// ```
pub fn read_segment(file: &[u8]) -> Result<Segment<'_>, MkvError> {
    let segment = ebml::elements(file)
        .flatten()
        .find(|e| e.id == ids::SEGMENT)
        .ok_or(MkvError::MissingSegment)?;

    let info = ebml::find(segment.payload, ids::INFO);
    let timestamp_scale = info
        .and_then(|i| ebml::find(i.payload, ids::TIMESTAMP_SCALE))
        .map(|e| ebml::uint(e.payload))
        // A scale of zero would make every timestamp zero; treat a file
        // that says so as one that said nothing.
        .filter(|s| *s > 0)
        .unwrap_or(DEFAULT_TIMESTAMP_SCALE);
    let duration_ticks = info
        .and_then(|i| ebml::find(i.payload, ids::DURATION))
        .map(|e| ebml::float(e.payload))
        .unwrap_or(0.0);
    let title = info
        .and_then(|i| ebml::find(i.payload, ids::TITLE))
        .map(|e| ebml::string(e.payload))
        .filter(|s| !s.is_empty());

    let tracks_el = ebml::find(segment.payload, ids::TRACKS).ok_or(MkvError::MissingTracks)?;
    let mut tracks = Vec::new();
    let mut truncated = false;
    for entry in ebml::elements(tracks_el.payload) {
        // A garbled entry ends the walk; the tracks already read are
        // still worth carrying, and saying nothing about the rest would
        // be the silent loss this project keeps refusing to commit.
        let Ok(entry) = entry else {
            truncated = true;
            break;
        };
        if entry.id != ids::TRACK_ENTRY {
            continue;
        }
        if tracks.len() >= MAX_TRACKS {
            truncated = true;
            break;
        }
        tracks.push(parse_track(entry.payload));
    }

    Ok(Segment {
        timestamp_scale,
        duration_ticks,
        title,
        tracks,
        chapters: read_chapters(segment.payload),
        tags: read_tags(segment.payload),
        truncated,
        payload: segment.payload,
    })
}

/// The most chapters or tags to read from one file.
///
/// Negative trust: a hostile file can declare as many as its length
/// allows, and a reader that believed it would spend the memory saying
/// so. Real films have tens of chapters.
const MAX_MARKS: usize = 10_000;

/// Read the chapter list, if the file has one.
///
/// Total by design, like the track reader: a garbled atom costs that
/// atom, not the file, because losing every chapter over one bad one is
/// the larger loss.
fn read_chapters(segment: &[u8]) -> Vec<Chapter<'_>> {
    let mut out = Vec::new();
    let Some(chapters) = ebml::find(segment, ids::CHAPTERS) else {
        return out;
    };
    for edition in ebml::elements(chapters.payload).flatten() {
        if edition.id != ids::EDITION_ENTRY {
            continue;
        }
        for atom in ebml::elements(edition.payload).flatten() {
            if atom.id != ids::CHAPTER_ATOM || out.len() >= MAX_MARKS {
                continue;
            }
            let display = ebml::find(atom.payload, ids::CHAPTER_DISPLAY);
            out.push(Chapter {
                uid: ebml::find(atom.payload, ids::CHAPTER_UID)
                    .map_or(0, |e| ebml::uint(e.payload)),
                start_nanos: ebml::find(atom.payload, ids::CHAPTER_TIME_START)
                    .map_or(0, |e| ebml::uint(e.payload)),
                end_nanos: ebml::find(atom.payload, ids::CHAPTER_TIME_END)
                    .map_or(0, |e| ebml::uint(e.payload)),
                title: display
                    .and_then(|d| ebml::find(d.payload, ids::CHAP_STRING))
                    .map(|e| ebml::string(e.payload))
                    .filter(|s| !s.is_empty()),
                language: display
                    .and_then(|d| ebml::find(d.payload, ids::CHAP_LANGUAGE))
                    .map(|e| ebml::string(e.payload))
                    .filter(|s| !s.is_empty()),
            });
        }
    }
    out
}

/// Read the tag list, flattening the tree into one entry per pair.
///
/// Matroska nests: a `Tag` says what it is about once and then carries
/// any number of name-and-value pairs. Flattening loses nothing — every
/// entry keeps the target it came from — and spares every caller the
/// walk.
fn read_tags(segment: &[u8]) -> Vec<Tag<'_>> {
    let mut out = Vec::new();
    let Some(tags) = ebml::find(segment, ids::TAGS) else {
        return out;
    };
    for tag in ebml::elements(tags.payload).flatten() {
        if tag.id != ids::TAG {
            continue;
        }
        // An empty `Targets`, or none at all, means the whole file.
        let track_uid = ebml::find(tag.payload, ids::TARGETS)
            .and_then(|t| ebml::find(t.payload, ids::TAG_TRACK_UID))
            .map(|e| ebml::uint(e.payload));
        for simple in ebml::elements(tag.payload).flatten() {
            if simple.id != ids::SIMPLE_TAG || out.len() >= MAX_MARKS {
                continue;
            }
            let name = ebml::find(simple.payload, ids::TAG_NAME).map(|e| ebml::string(e.payload));
            let value =
                ebml::find(simple.payload, ids::TAG_STRING).map(|e| ebml::string(e.payload));
            if let (Some(name), Some(value)) = (name, value) {
                if !name.is_empty() {
                    out.push(Tag {
                        track_uid,
                        name,
                        value,
                    });
                }
            }
        }
    }
    out
}

/// Read one `TrackEntry`.
///
/// Total by design: every field has a defined default, so a sparse or
/// partly unreadable entry yields a usable track rather than failing the
/// file. What cannot be defaulted — the track number — comes back as
/// zero, which no real track uses.
fn parse_track(payload: &[u8]) -> Track<'_> {
    let mut t = Track {
        number: 0,
        kind: TrackKind::Other,
        codec_id: "",
        codec_private: &[],
        language: DEFAULT_LANGUAGE,
        name: None,
        enabled: true,
        default: true,
        forced: false,
        codec_delay: 0,
        default_duration: None,
        width: 0,
        height: 0,
        sample_rate: 0,
        channels: 0,
        colour: None,
    };
    // BCP 47 supersedes ISO 639-2 when both are present, so remember
    // whether the better one has been seen rather than letting element
    // order decide.
    let mut have_bcp47 = false;

    for child in ebml::elements(payload).flatten() {
        match child.id {
            ids::TRACK_NUMBER => t.number = ebml::uint(child.payload),
            ids::TRACK_TYPE => t.kind = TrackKind::from_code(ebml::uint(child.payload)),
            ids::CODEC_ID => t.codec_id = ebml::string(child.payload),
            ids::CODEC_PRIVATE => t.codec_private = child.payload,
            ids::CODEC_DELAY => t.codec_delay = ebml::uint(child.payload),
            ids::DEFAULT_DURATION => {
                t.default_duration = Some(ebml::uint(child.payload)).filter(|d| *d > 0)
            }
            ids::NAME => t.name = Some(ebml::string(child.payload)).filter(|s| !s.is_empty()),
            ids::LANGUAGE if !have_bcp47 => {
                let s = ebml::string(child.payload);
                if !s.is_empty() {
                    t.language = s;
                }
            }
            ids::LANGUAGE_BCP47 => {
                let s = ebml::string(child.payload);
                if !s.is_empty() {
                    t.language = s;
                    have_bcp47 = true;
                }
            }
            ids::FLAG_ENABLED => t.enabled = ebml::uint(child.payload) != 0,
            ids::FLAG_DEFAULT => t.default = ebml::uint(child.payload) != 0,
            ids::FLAG_FORCED => t.forced = ebml::uint(child.payload) != 0,
            ids::VIDEO => {
                for v in ebml::elements(child.payload).flatten() {
                    match v.id {
                        ids::PIXEL_WIDTH => t.width = ebml::uint(v.payload) as u32,
                        ids::PIXEL_HEIGHT => t.height = ebml::uint(v.payload) as u32,
                        // Matroska spells the colour code points out as
                        // separate children where ISOBMFF packs them
                        // into one payload. Same numbers either way.
                        ids::COLOUR => {
                            let mut c = sovmr_mp4::colour::Colour::default();
                            for child in ebml::elements(v.payload).flatten() {
                                let n = ebml::uint(child.payload);
                                match child.id {
                                    ids::COLOUR_PRIMARIES => c.primaries = n as u16,
                                    ids::COLOUR_TRANSFER => c.transfer = n as u16,
                                    ids::COLOUR_MATRIX => c.matrix = n as u16,
                                    ids::COLOUR_RANGE => c.full_range = n == ids::RANGE_FULL,
                                    _ => {}
                                }
                            }
                            t.colour = Some(c);
                        }
                        _ => {}
                    }
                }
            }
            ids::AUDIO => {
                for a in ebml::elements(child.payload).flatten() {
                    match a.id {
                        // Stored as a float, because Matroska allows
                        // fractional rates; every real one is an integer.
                        ids::SAMPLING_FREQUENCY => t.sample_rate = ebml::float(a.payload) as u32,
                        ids::CHANNELS => t.channels = ebml::uint(a.payload) as u16,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize one element, choosing the shortest size encoding that
    /// fits — one octet up to 126 bytes, two up to 16382.
    fn el(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        let n = payload.len();
        if n < 0x7F {
            out.push(0x80 | n as u8);
        } else {
            assert!(n < 0x3FFF, "fixtures stay well inside two octets");
            out.push(0x40 | (n >> 8) as u8);
            out.push((n & 0xFF) as u8);
        }
        out.extend_from_slice(payload);
        out
    }

    /// Serialize an element with a multi-octet ID.
    fn el_id(id: u32, payload: &[u8]) -> Vec<u8> {
        let bytes = id.to_be_bytes();
        let start = bytes.iter().position(|b| *b != 0).unwrap_or(3);
        el(&bytes[start..], payload)
    }

    /// A minimal file: EBML header, Segment holding Info and Tracks.
    fn file_with(info: &[u8], tracks: &[u8]) -> Vec<u8> {
        let mut seg = el_id(ids::INFO, info);
        seg.extend(el_id(ids::TRACKS, tracks));
        let mut out = el_id(crate::EBML_HEADER_ID, &[]);
        out.extend(el_id(ids::SEGMENT, &seg));
        out
    }

    fn audio_entry(number: u8, lang: &str) -> Vec<u8> {
        let mut e = el_id(ids::TRACK_NUMBER, &[number]);
        e.extend(el_id(ids::TRACK_TYPE, &[ids::TYPE_AUDIO as u8]));
        e.extend(el_id(ids::CODEC_ID, b"A_OPUS"));
        e.extend(el_id(ids::LANGUAGE, lang.as_bytes()));
        let mut audio = el_id(ids::CHANNELS, &[2]);
        audio.extend(el_id(ids::SAMPLING_FREQUENCY, &48000.0f64.to_be_bytes()));
        e.extend(el_id(ids::AUDIO, &audio));
        el_id(ids::TRACK_ENTRY, &e)
    }

    #[test]
    fn tracks_come_back_in_file_order_with_their_identity() {
        let mut tracks = audio_entry(1, "eng");
        tracks.extend(audio_entry(2, "rus"));
        tracks.extend(audio_entry(3, "deu"));
        let bytes = file_with(&el_id(ids::TIMESTAMP_SCALE, &[0x0F, 0x42, 0x40]), &tracks);

        let seg = read_segment(&bytes).unwrap();
        assert_eq!(seg.timestamp_scale, 1_000_000);
        assert_eq!(seg.tracks.len(), 3);
        assert!(!seg.truncated);
        let langs: Vec<&str> = seg.tracks.iter().map(|t| t.language).collect();
        assert_eq!(langs, vec!["eng", "rus", "deu"]);
        let numbers: Vec<u64> = seg.tracks.iter().map(|t| t.number).collect();
        assert_eq!(numbers, vec![1, 2, 3]);
        for t in &seg.tracks {
            assert_eq!(t.kind, TrackKind::Audio);
            assert_eq!(t.codec_id, "A_OPUS");
            assert_eq!(t.channels, 2);
            assert_eq!(t.sample_rate, 48_000);
        }
    }

    #[test]
    fn a_track_that_says_nothing_is_english_by_specification() {
        // Not a guess: a TrackEntry with no Language element IS `eng`
        // per the format. Defaulting to "und" would relabel every file
        // that relied on the rule.
        let mut e = el_id(ids::TRACK_NUMBER, &[1]);
        e.extend(el_id(ids::TRACK_TYPE, &[ids::TYPE_AUDIO as u8]));
        let bytes = file_with(&[], &el_id(ids::TRACK_ENTRY, &e));
        let seg = read_segment(&bytes).unwrap();
        assert_eq!(seg.tracks[0].language, "eng");
        assert!(
            seg.tracks[0].enabled,
            "tracks are enabled unless told otherwise"
        );
        assert!(seg.tracks[0].default);
        assert!(!seg.tracks[0].forced);
    }

    #[test]
    fn bcp47_wins_over_iso_639_whichever_order_they_appear_in() {
        // Both spellings are legal and a file may carry both; the newer
        // one is authoritative. Letting element order decide would make
        // the answer depend on the muxer's whim.
        for iso_first in [true, false] {
            let mut e = el_id(ids::TRACK_NUMBER, &[1]);
            e.extend(el_id(ids::TRACK_TYPE, &[ids::TYPE_AUDIO as u8]));
            let iso = el_id(ids::LANGUAGE, b"ger");
            let bcp = el_id(ids::LANGUAGE_BCP47, b"de-AT");
            if iso_first {
                e.extend(iso);
                e.extend(bcp);
            } else {
                e.extend(bcp);
                e.extend(iso);
            }
            let bytes = file_with(&[], &el_id(ids::TRACK_ENTRY, &e));
            let seg = read_segment(&bytes).unwrap();
            assert_eq!(seg.tracks[0].language, "de-AT", "iso_first={iso_first}");
        }
    }

    #[test]
    fn every_track_kind_is_told_apart() {
        let kinds = [
            (ids::TYPE_VIDEO, TrackKind::Video),
            (ids::TYPE_AUDIO, TrackKind::Audio),
            (ids::TYPE_SUBTITLE, TrackKind::Subtitle),
            (99, TrackKind::Other),
        ];
        for (code, want) in kinds {
            let mut e = el_id(ids::TRACK_NUMBER, &[1]);
            e.extend(el_id(ids::TRACK_TYPE, &[code as u8]));
            let bytes = file_with(&[], &el_id(ids::TRACK_ENTRY, &e));
            assert_eq!(read_segment(&bytes).unwrap().tracks[0].kind, want);
        }
    }

    #[test]
    fn codec_private_survives_byte_for_byte() {
        // These bytes go straight to the decoder; a reader that trimmed
        // or re-encoded them would break every track it touched.
        let avcc = [0x01u8, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x04];
        let mut e = el_id(ids::TRACK_NUMBER, &[1]);
        e.extend(el_id(ids::TRACK_TYPE, &[ids::TYPE_VIDEO as u8]));
        e.extend(el_id(ids::CODEC_ID, b"V_MPEG4/ISO/AVC"));
        e.extend(el_id(ids::CODEC_PRIVATE, &avcc));
        let mut video = el_id(ids::PIXEL_WIDTH, &[0x05, 0x00]);
        video.extend(el_id(ids::PIXEL_HEIGHT, &[0x02, 0xD0]));
        e.extend(el_id(ids::VIDEO, &video));
        let bytes = file_with(&[], &el_id(ids::TRACK_ENTRY, &e));

        let seg = read_segment(&bytes).unwrap();
        let t = &seg.tracks[0];
        assert_eq!(t.codec_private, &avcc);
        assert_eq!((t.width, t.height), (1280, 720));
    }

    #[test]
    fn a_missing_timestamp_scale_uses_the_formats_default() {
        let bytes = file_with(&[], &audio_entry(1, "eng"));
        assert_eq!(read_segment(&bytes).unwrap().timestamp_scale, 1_000_000);
    }

    #[test]
    fn a_zero_timestamp_scale_is_treated_as_absent() {
        // Honouring a zero would collapse every timestamp in the file to
        // the same instant — worse than ignoring an impossible claim.
        let bytes = file_with(&el_id(ids::TIMESTAMP_SCALE, &[0]), &audio_entry(1, "eng"));
        assert_eq!(read_segment(&bytes).unwrap().timestamp_scale, 1_000_000);
    }

    #[test]
    fn a_file_with_no_segment_or_no_tracks_says_which() {
        let header = el_id(crate::EBML_HEADER_ID, &[]);
        assert_eq!(read_segment(&header), Err(MkvError::MissingSegment));

        let mut bytes = el_id(crate::EBML_HEADER_ID, &[]);
        bytes.extend(el_id(ids::SEGMENT, &el_id(ids::INFO, &[])));
        assert_eq!(read_segment(&bytes), Err(MkvError::MissingTracks));
    }

    #[test]
    fn more_tracks_than_the_ceiling_stop_the_scan_and_say_so() {
        let mut tracks = Vec::new();
        for i in 0..MAX_TRACKS + 8 {
            tracks.extend(audio_entry((i % 200) as u8 + 1, "eng"));
        }
        let bytes = file_with(&[], &tracks);
        let seg = read_segment(&bytes).unwrap();
        assert_eq!(seg.tracks.len(), MAX_TRACKS);
        assert!(seg.truncated, "the cut must be reported, not silent");
    }

    #[test]
    fn a_garbled_entry_costs_that_track_not_the_whole_file() {
        // Graceful degradation: the readable tracks still come back, and
        // `truncated` says the list is incomplete.
        let mut tracks = audio_entry(1, "eng");
        tracks.extend(audio_entry(2, "rus"));
        tracks.extend_from_slice(&[0xAE, 0x9F, 0x01]); // claims 31 bytes, has 1
        let bytes = file_with(&[], &tracks);
        let seg = read_segment(&bytes).unwrap();
        assert_eq!(seg.tracks.len(), 2);
        assert!(seg.truncated);
    }

    #[test]
    fn of_kind_selects_without_reordering() {
        let mut tracks = audio_entry(1, "eng");
        let mut v = el_id(ids::TRACK_NUMBER, &[2]);
        v.extend(el_id(ids::TRACK_TYPE, &[ids::TYPE_VIDEO as u8]));
        tracks.extend(el_id(ids::TRACK_ENTRY, &v));
        tracks.extend(audio_entry(3, "rus"));
        let bytes = file_with(&[], &tracks);
        let seg = read_segment(&bytes).unwrap();

        let audio: Vec<u64> = seg.of_kind(TrackKind::Audio).map(|t| t.number).collect();
        assert_eq!(audio, vec![1, 3]);
        assert_eq!(seg.of_kind(TrackKind::Video).count(), 1);
        assert_eq!(seg.of_kind(TrackKind::Subtitle).count(), 0);
    }

    #[test]
    fn ticks_become_nanoseconds_without_overflowing() {
        let bytes = file_with(&[], &audio_entry(1, "eng"));
        let seg = read_segment(&bytes).unwrap();
        assert_eq!(seg.ticks_to_nanos(1_000), 1_000_000_000);
        // A twelve-hour file is ~43.2 M ticks; a hostile one may claim
        // far more, and saturating beats wrapping to a negative instant.
        assert_eq!(seg.ticks_to_nanos(i64::MAX), i64::MAX);
    }
}
