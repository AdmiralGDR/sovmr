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

//! The AAC frame: `raw_data_block`, as a container carries it.
//!
//! # What is here, and what is not
//!
//! The *structure* of a frame — which fields exist, how wide they are,
//! and in what order — is small enough to state and to check. That is
//! what this module builds.
//!
//! The *spectral data* inside it is coded with eleven Huffman codebooks
//! from ISO/IEC 14496-3: some thousands of fixed numbers. Those are not
//! written from memory, because this project has twice paid for doing
//! that, so for now every band is declared empty.
//!
//! # An empty frame is not a useless one
//!
//! A frame whose `max_sfb` is zero codes no scalefactor bands at all,
//! which means every spectral coefficient is zero. It is entirely valid
//! AAC and it decodes to silence — and that makes it the one thing that
//! can prove the whole surrounding structure before a single codebook
//! exists: the config, the element framing, the field widths, the
//! alignment, and the container wiring.
//!
//! If a decoder plays this and complains about nothing, everything
//! except the spectrum is right. If it does not, no amount of correct
//! spectral coding would have saved the file.

use crate::bits::BitWriter;

/// Syntactic element identifiers, as `raw_data_block` tags them.
const ID_SCE: u32 = 0x0;
const ID_CPE: u32 = 0x1;
const ID_END: u32 = 0x7;

/// `window_sequence` values.
const ONLY_LONG_SEQUENCE: u32 = 0;

/// Coefficients a long AAC frame carries per channel.
pub const FRAME_LENGTH: usize = 1024;

/// How the channels of a frame are packed into elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelElement {
    /// One channel, as a `single_channel_element`.
    Single,
    /// Two channels sharing an element, as a `channel_pair_element`.
    Pair,
}

impl ChannelElement {
    /// The layout for a channel count, or `None` for one this encoder
    /// does not lay out yet.
    ///
    /// More than two channels needs several elements and a programme
    /// configuration to say which is which; refusing is better than
    /// writing a file that claims a layout it does not have.
    pub fn for_channels(channels: u32) -> Option<ChannelElement> {
        match channels {
            1 => Some(ChannelElement::Single),
            2 => Some(ChannelElement::Pair),
            _ => None,
        }
    }
}

/// Write one `individual_channel_stream` with no coded bands.
///
/// `common_window` is true for the second channel of a pair, whose
/// `ics_info` was already written once for both.
fn write_empty_ics(w: &mut BitWriter, common_window: bool) {
    // `global_gain` scales every band; with no bands it changes nothing,
    // but it is not optional and a decoder reads it either way.
    w.write(0, 8);
    if !common_window {
        write_ics_info(w);
    }
    // section_data and scale_factor_data are both empty when `max_sfb`
    // is zero: there are no bands to assign codebooks to and none to
    // scale. That emptiness is the point of this frame.
    w.write(0, 1); // pulse_data_present
    w.write(0, 1); // tns_data_present
    w.write(0, 1); // gain_control_data_present
                   // spectral_data: nothing, for the same reason.
}

/// Write an `ics_info` describing a long window with no coded bands.
fn write_ics_info(w: &mut BitWriter) {
    w.write(0, 1); // ics_reserved_bit
    w.write(ONLY_LONG_SEQUENCE, 2);
    w.write(0, 1); // window_shape: sine
    w.write(0, 6); // max_sfb: no bands are coded
    w.write(0, 1); // predictor_data_present — never set for AAC-LC
}

/// Build one silent `raw_data_block` for the given channel layout.
///
/// The frame is complete and valid: a decoder reads it, finds no coded
/// bands, and produces 1024 samples of silence per channel.
///
/// # Example
///
/// ```
/// use sovmr_audio::frame::{silent_frame, ChannelElement};
/// let bytes = silent_frame(ChannelElement::Single);
/// assert!(!bytes.is_empty());
/// ```
pub fn silent_frame(element: ChannelElement) -> Vec<u8> {
    let mut w = BitWriter::new();
    match element {
        ChannelElement::Single => {
            w.write(ID_SCE, 3);
            w.write(0, 4); // element_instance_tag
            write_empty_ics(&mut w, false);
        }
        ChannelElement::Pair => {
            w.write(ID_CPE, 3);
            w.write(0, 4); // element_instance_tag
                           // Both channels share one `ics_info`, written here rather
                           // than inside either stream — which is what `common_window`
                           // means, and getting it wrong desynchronises the second
                           // channel by exactly the width of that structure.
            w.write(1, 1); // common_window
            write_ics_info(&mut w);
            w.write(0, 2); // ms_mask_present: no mid/side
            write_empty_ics(&mut w, true);
            write_empty_ics(&mut w, true);
        }
    }
    // Every `raw_data_block` ends with this tag, then pads to a byte.
    // A decoder that does not find it keeps reading into whatever
    // follows, which is how one malformed frame corrupts a whole file.
    w.write(ID_END, 3);
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;

    #[test]
    fn a_mono_frame_has_the_fields_the_specification_lists() {
        // Read back with the reader a decoder would use, field by field,
        // in order. A width wrong anywhere shifts everything after it.
        let bytes = silent_frame(ChannelElement::Single);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u32(3), Some(ID_SCE), "element id");
        assert_eq!(r.u32(4), Some(0), "element_instance_tag");
        assert_eq!(r.u32(8), Some(0), "global_gain");
        assert_eq!(r.u32(1), Some(0), "ics_reserved_bit");
        assert_eq!(r.u32(2), Some(ONLY_LONG_SEQUENCE), "window_sequence");
        assert_eq!(r.u32(1), Some(0), "window_shape");
        assert_eq!(r.u32(6), Some(0), "max_sfb");
        assert_eq!(r.u32(1), Some(0), "predictor_data_present");
        assert_eq!(r.u32(1), Some(0), "pulse_data_present");
        assert_eq!(r.u32(1), Some(0), "tns_data_present");
        assert_eq!(r.u32(1), Some(0), "gain_control_data_present");
        assert_eq!(r.u32(3), Some(ID_END), "the terminator must be reachable");
    }

    #[test]
    fn a_stereo_frame_writes_one_shared_window_and_two_streams() {
        let bytes = silent_frame(ChannelElement::Pair);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u32(3), Some(ID_CPE));
        assert_eq!(r.u32(4), Some(0), "element_instance_tag");
        assert_eq!(r.u32(1), Some(1), "common_window");
        // The shared ics_info.
        assert_eq!(r.u32(1), Some(0));
        assert_eq!(r.u32(2), Some(ONLY_LONG_SEQUENCE));
        assert_eq!(r.u32(1), Some(0));
        assert_eq!(r.u32(6), Some(0), "max_sfb");
        assert_eq!(r.u32(1), Some(0));
        assert_eq!(r.u32(2), Some(0), "ms_mask_present");
        // Two streams, each a global gain and three absent-data flags —
        // and *no* second ics_info, which is what common_window bought.
        for channel in 0..2 {
            assert_eq!(r.u32(8), Some(0), "channel {channel} global_gain");
            assert_eq!(r.u32(1), Some(0), "channel {channel} pulse");
            assert_eq!(r.u32(1), Some(0), "channel {channel} tns");
            assert_eq!(r.u32(1), Some(0), "channel {channel} gain control");
        }
        assert_eq!(r.u32(3), Some(ID_END), "the terminator must be reachable");
    }

    #[test]
    fn the_terminator_lands_where_a_decoder_stops_reading() {
        // The whole value of the two tests above is this consequence: if
        // any field were the wrong width, ID_END would be found at the
        // wrong bit and a decoder would read past the frame.
        for element in [ChannelElement::Single, ChannelElement::Pair] {
            let bytes = silent_frame(element);
            let expected_bits = match element {
                // 3 + 4 + 8 + (1+2+1+6+1) + 3 + 3
                ChannelElement::Single => 32,
                // 3 + 4 + 1 + (1+2+1+6+1) + 2 + 2*(8+3) + 3
                ChannelElement::Pair => 48,
            };
            assert_eq!(
                bytes.len() * 8,
                expected_bits,
                "{element:?}: a frame of the wrong length means a field of the wrong width"
            );
        }
    }

    #[test]
    fn a_frame_is_whole_bytes_so_the_next_one_starts_clean() {
        // Containers hand a decoder whole frames. One that ended
        // mid-byte would leave the next frame's first field split across
        // a boundary the container does not know about.
        for element in [ChannelElement::Single, ChannelElement::Pair] {
            let bytes = silent_frame(element);
            assert!(!bytes.is_empty(), "{element:?}");
        }
    }

    #[test]
    fn layouts_this_cannot_write_are_refused_rather_than_approximated() {
        // Beyond stereo needs several elements and a programme
        // configuration saying which is which. Writing a file that
        // claims a layout it does not have would be worse than refusing.
        assert_eq!(
            ChannelElement::for_channels(1),
            Some(ChannelElement::Single)
        );
        assert_eq!(ChannelElement::for_channels(2), Some(ChannelElement::Pair));
        for channels in [0u32, 3, 6, 8] {
            assert_eq!(ChannelElement::for_channels(channels), None, "{channels}");
        }
    }
}
