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

//! `AudioSpecificConfig` — the two bytes that say what an AAC stream is.
//!
//! Every container carries it somewhere: MP4 wraps it in an `esds`
//! descriptor, Matroska puts it bare in `CodecPrivate`. A decoder reads
//! nothing else before the first frame, so if these bits are wrong the
//! stream is unplayable no matter how good the frames are.
//!
//! It is small enough to state completely. For AAC-LC:
//!
//! | Field | Bits |
//! |---|---|
//! | `audioObjectType` | 5 |
//! | `samplingFrequencyIndex` | 4, or 4 + 24 when the rate is unlisted |
//! | `channelConfiguration` | 4 |
//! | `frameLengthFlag` | 1 |
//! | `dependsOnCoreCoder` | 1 |
//! | `extensionFlag` | 1 |
//!
//! Sixteen bits for an ordinary stream, which is why an AAC
//! `CodecPrivate` is so often exactly two bytes.

use crate::bits::{BitReader, BitWriter};

/// `audioObjectType` for AAC Low Complexity — the only profile this
/// encoder produces, and the one every decoder supports.
pub const AAC_LC: u32 = 2;

/// The sampling rates the format can name in four bits, in order.
///
/// A rate outside this list is legal but must be written out in full,
/// which costs 24 further bits and is why the table is worth having.
pub const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// The escape index meaning "the rate follows in full".
const RATE_ESCAPE: u32 = 0xF;

/// What an `AudioSpecificConfig` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// Profile; [`AAC_LC`] for anything this encoder writes.
    pub object_type: u32,
    /// Sampling rate in Hz.
    pub sample_rate: u32,
    /// Channel configuration: 1 is mono, 2 stereo, up to 7 for 7.1.
    ///
    /// Zero is legal and means "the frames describe their own layout",
    /// which this encoder does not emit — but a file being *read* may
    /// well use it.
    pub channels: u32,
}

impl AudioSpecificConfig {
    /// Index of `sample_rate` in [`SAMPLE_RATES`], or the escape.
    fn rate_index(&self) -> u32 {
        SAMPLE_RATES
            .iter()
            .position(|r| *r == self.sample_rate)
            .map(|i| i as u32)
            .unwrap_or(RATE_ESCAPE)
    }

    /// Serialize to the bytes a container carries.
    ///
    /// # Example
    ///
    /// ```
    /// use sovmr_audio::asc::{AudioSpecificConfig, AAC_LC};
    /// // The two bytes every 48 kHz stereo AAC track begins with.
    /// let asc = AudioSpecificConfig { object_type: AAC_LC, sample_rate: 48_000, channels: 2 };
    /// assert_eq!(asc.to_bytes(), vec![0x11, 0x90]);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write(self.object_type, 5);
        let index = self.rate_index();
        w.write(index, 4);
        if index == RATE_ESCAPE {
            // An unlisted rate is spelled out rather than approximated
            // to a neighbour: a decoder told the wrong rate plays the
            // whole track at the wrong speed.
            w.write(self.sample_rate, 24);
        }
        w.write(self.channels, 4);
        // GASpecificConfig: a 1024-sample frame, no core coder, no
        // extension. All three are zero for every stream this encoder
        // produces, and writing them explicitly beats relying on the
        // padding to mean the same thing.
        //
        // Nothing follows them. ffmpeg appends a further three bytes —
        // eleven bits of `0x2B7`, then extension type 5 and a flag
        // saying SBR is absent — which is an optional backward-compatible
        // signal, not a requirement. Saying "no SBR" explicitly and
        // saying nothing mean the same thing to a decoder, and the
        // shorter config is what most encoders write. Verified against
        // ffmpeg's own bytes: ours is exactly its prefix.
        w.write(0, 1); // frameLengthFlag: 0 = 1024 samples
        w.write(0, 1); // dependsOnCoreCoder
        w.write(0, 1); // extensionFlag
        w.finish()
    }

    /// Read a config a container handed over.
    ///
    /// Returns `None` when the bytes run out mid-field — which is what a
    /// truncated `CodecPrivate` looks like, and is worth distinguishing
    /// from a config that merely says something surprising.
    pub fn parse(data: &[u8]) -> Option<AudioSpecificConfig> {
        let mut r = BitReader::new(data);
        let object_type = r.u32(5)?;
        let index = r.u32(4)?;
        let sample_rate = if index == RATE_ESCAPE {
            r.u32(24)?
        } else {
            *SAMPLE_RATES.get(index as usize)?
        };
        let channels = r.u32(4)?;
        Some(AudioSpecificConfig {
            object_type,
            sample_rate,
            channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_bytes_every_aac_track_is_known_by() {
        // These pairs appear in essentially every AAC file in existence,
        // which makes them the closest thing to an independent check
        // available without decoding something.
        let cases = [
            (48_000u32, 2u32, vec![0x11u8, 0x90]),
            (44_100, 2, vec![0x12, 0x10]),
            (48_000, 1, vec![0x11, 0x88]),
            (44_100, 1, vec![0x12, 0x08]),
        ];
        for (rate, channels, expected) in cases {
            let asc = AudioSpecificConfig {
                object_type: AAC_LC,
                sample_rate: rate,
                channels,
            };
            assert_eq!(asc.to_bytes(), expected, "{rate} Hz, {channels} ch");
        }
    }

    #[test]
    fn every_listed_rate_round_trips_through_its_index() {
        for rate in SAMPLE_RATES {
            for channels in 1..=7 {
                let asc = AudioSpecificConfig {
                    object_type: AAC_LC,
                    sample_rate: rate,
                    channels,
                };
                let back = AudioSpecificConfig::parse(&asc.to_bytes()).expect("parses");
                assert_eq!(back, asc, "{rate} Hz, {channels} ch");
                assert_eq!(asc.to_bytes().len(), 2, "a listed rate fits in two bytes");
            }
        }
    }

    #[test]
    fn an_unlisted_rate_is_written_out_rather_than_rounded() {
        // Snapping 47 999 Hz to 48 000 would play the whole track at the
        // wrong speed — slightly, permanently, and without complaint.
        let asc = AudioSpecificConfig {
            object_type: AAC_LC,
            sample_rate: 47_999,
            channels: 2,
        };
        let bytes = asc.to_bytes();
        assert!(bytes.len() > 2, "the rate needs spelling out");
        assert_eq!(AudioSpecificConfig::parse(&bytes), Some(asc));
    }

    #[test]
    fn a_truncated_config_is_none_rather_than_a_guess() {
        // A short `CodecPrivate` is a damaged file. Filling in the
        // missing bits would hand a decoder a plausible stream
        // description that nothing in the file supports.
        assert_eq!(AudioSpecificConfig::parse(&[]), None);
        assert_eq!(AudioSpecificConfig::parse(&[0x11]), None);
        // Two bytes is exactly enough for a listed rate.
        assert!(AudioSpecificConfig::parse(&[0x11, 0x90]).is_some());
    }

    #[test]
    fn a_reserved_rate_index_is_refused() {
        // Indices 13 and 14 are reserved. Reading one as a rate would
        // invent a number the format does not define.
        for index in [13u32, 14] {
            let mut w = BitWriter::new();
            w.write(AAC_LC, 5);
            w.write(index, 4);
            w.write(2, 4);
            w.write(0, 3);
            assert_eq!(
                AudioSpecificConfig::parse(&w.finish()),
                None,
                "index {index}"
            );
        }
    }

    #[test]
    fn the_channel_count_lands_where_a_decoder_looks_for_it() {
        // Off-by-one here is the classic way to produce a stereo file
        // that announces itself as mono, which decoders honour.
        for channels in 0..=7u32 {
            let asc = AudioSpecificConfig {
                object_type: AAC_LC,
                sample_rate: 44_100,
                channels,
            };
            assert_eq!(
                AudioSpecificConfig::parse(&asc.to_bytes())
                    .unwrap()
                    .channels,
                channels
            );
        }
    }
}
