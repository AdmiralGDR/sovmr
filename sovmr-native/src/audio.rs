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

//! What happens to each audio track, and why.
//!
//! # The decision
//!
//! Three outcomes, and the reasoning is the same every time: *what is
//! the least this track can lose?*
//!
//! **Carried.** Already-compressed audio is copied byte for byte.
//! Decoding and re-encoding it would add a second generation of loss on
//! top of the first, which is strictly worse than doing nothing.
//!
//! **Compressed.** Lossless sources — PCM and FLAC — are decoded and
//! encoded to AAC. There is no generation loss to worry about here: the
//! source is mathematically exact, so this is the *first* time the audio
//! is quantised, and the same thing an encoder would have done to the
//! original recording.
//!
//! **Dropped.** Named, never silent. Losing five of six tracks without a
//! word was the defect an earlier milestone existed to end, and the same
//! rule holds for anything this one cannot carry.
//!
//! # Why the policy lives here
//!
//! `sovmr-mkv` knows containers and `sovmr-audio` knows codecs; neither
//! should know about the other. This crate already depends on both, so
//! the decision that needs both belongs here and nowhere else.

use sovmr_audio::asc::{AudioSpecificConfig, AAC_LC};
use sovmr_audio::bands::LONG_FRAME;
use sovmr_audio::encode::{encode_frame, encode_frame_pair, EncodeError};
use sovmr_audio::flac::{self, StreamInfo};
use sovmr_audio::mdct::{mdct, window, WindowShape};
use sovmr_audio::pcm::{self, PcmFormat};
use sovmr_mkv::source::Target;

/// A lossless source this project can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lossless {
    /// Linear PCM in one of the layouts a container can name.
    Pcm(PcmFormat),
    /// FLAC, whose `CodecPrivate` is its `STREAMINFO`.
    Flac,
}

impl Lossless {
    /// What to call this in a message to a person.
    pub fn name(self) -> &'static str {
        match self {
            Lossless::Pcm(_) => "PCM",
            Lossless::Flac => "FLAC",
        }
    }
}

/// What will happen to one track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// Copied byte for byte, losing nothing.
    Carry,
    /// Decoded and encoded to AAC, because the source is lossless and a
    /// re-encode costs it no generation.
    Compress(Lossless),
    /// Cannot come along; the reason is for the user, not the log.
    Drop(&'static str),
}

/// Decide what happens to one track.
///
/// # Why lossy tracks are still copied, now that we can decode them
///
/// Stage 6 built an AAC decoder so an over-large lossy track could be
/// re-encoded instead of copied byte for byte, and then priced the lever
/// before plumbing it in. Re-encoding somebody else's AAC
/// with ours, judged against the sound we were handed:
///
/// | material          | change  | fidelity |
/// |-------------------|---------|----------|
/// | tone, 128 kbit/s  |  -70.6% |  68.3 dB |
/// | tone, 320 kbit/s  |  -74.2% |  68.3 dB |
/// | clicks, 320       |   +0.3% |  61.5 dB |
/// | pink noise, 320   | +191.6% |  57.6 dB |
///
/// Noise substitution — the lever named here as next — has since been
/// written, and it works: pink noise goes from +191.6% to **-80.6%**,
/// and every substituted band carries the energy it stated to within
/// 0.15 dB. It is off by default all the same, because the criterion
/// deciding *which* bands to replace cannot be settled without a claim
/// about hearing this project has not measured. See
/// `sovmr_audio::encode::DEFAULT_NOISE_FLATNESS`.
///
/// A film soundtrack is far closer to noise and transients than to a
/// pure tone, so the route stays `Carry`. When the table says ours is
/// smaller at fidelity a listener would accept, this decision changes,
/// and not before — and "a listener would accept" is a claim nothing
/// here has yet listened for.
///
/// # The order, and why it is this way round
///
/// Lossless first, carrying second. That looks backwards — surely
/// copying a track is always better than re-encoding it? — and for
/// *lossy* audio it is, which is why nothing already compressed is ever
/// touched.
///
/// But this is a compressor. A twelve-hour film with a PCM track carries
/// four gigabytes of audio at 1.5 Mbit/s per channel pair, and Matroska
/// will happily copy every byte. Copying it is not caution; it is the
/// tool failing to do the one thing it was asked for. The source is
/// mathematically exact, so encoding it costs no generation — it is the
/// first quantisation, the same one the original recording would have
/// had.
///
/// When the encoder cannot take a lossless track — an unmeasured rate,
/// or more channels than an element can lay out — it falls through to
/// carrying rather than being dropped. Losing the track would be a far
/// worse answer than keeping it large.
pub fn route(
    codec_id: &str,
    codec_private: &[u8],
    sample_rate: u32,
    channels: u32,
    target: Target,
) -> Route {
    let lossless = lossless_source(codec_id, codec_private);
    if let Some(source) = lossless {
        if sovmr_audio::encode::can_encode(sample_rate, channels) {
            return Route::Compress(source);
        }
    }
    if can_carry(codec_id, codec_private, sample_rate, channels, target) {
        return Route::Carry;
    }
    // Nothing worked. Say which wall was hit, because the two have
    // different answers and "unsupported" would send someone looking in
    // the wrong place.
    Route::Drop(match lossless {
        Some(_) if channels == 0 || channels > 2 => {
            "more than two channels cannot be laid out yet, and this container cannot carry it"
        }
        Some(_) => {
            "no scalefactor band table for this sampling rate, and this container cannot carry it"
        }
        None => "no decoder for this codec, and the output container cannot carry it",
    })
}

/// Whether a track can be copied verbatim into the target container.
fn can_carry(
    codec_id: &str,
    codec_private: &[u8],
    sample_rate: u32,
    channels: u32,
    target: Target,
) -> bool {
    match target {
        // Matroska needs a name and a blob, and it has both for anything
        // that came out of a container in the first place.
        Target::Mkv => !codec_id.is_empty(),
        // `esds` describes MPEG-4 Audio and nothing else, so only AAC
        // makes this journey — and only when it brought the config a
        // descriptor has to be built around.
        Target::Mp4 => {
            (codec_id == "A_AAC" || codec_id.starts_with("A_AAC/"))
                && !codec_private.is_empty()
                && sample_rate != 0
                && channels != 0
        }
    }
}

/// Which lossless decoder, if any, reads this codec.
fn lossless_source(codec_id: &str, codec_private: &[u8]) -> Option<Lossless> {
    if let Some(format) = PcmFormat::from_codec_id(codec_id) {
        return Some(Lossless::Pcm(format));
    }
    // FLAC without its `STREAMINFO` cannot be decoded at all: the block
    // size, depth and channel count all live there.
    if codec_id == "A_FLAC" && StreamInfo::parse(codec_private).is_ok() {
        return Some(Lossless::Flac);
    }
    None
}

/// Why a track could not be compressed after all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressError {
    /// The source bytes were not what the codec said they were.
    Undecodable(&'static str),
    /// The encoder refused the rate or the layout.
    Unencodable(EncodeError),
    /// A decoded track whose sample count is not a whole number of
    /// frames of every channel — which means the channel count is wrong,
    /// not that the audio is odd.
    Ragged {
        /// Samples decoded in total, across all channels.
        samples: usize,
        /// Channels the source declared.
        channels: u32,
    },
}

/// A lossless track being turned into AAC, one packet at a time.
///
/// # Why this is a machine rather than a function
///
/// A whole-track function is the obvious shape and the wrong one. This
/// project exists to digest files longer than twelve hours; a twelve-hour
/// stereo PCM track decodes to some eight gigabytes of samples, and
/// holding those — or the frames they become — would undo the streaming
/// the rest of the pipeline was built for.
///
/// So samples are consumed as they arrive and frames are handed back as
/// soon as they are complete. What is retained is one block of overlap
/// per channel, which is what the transform needs and nothing more:
/// about sixteen kilobytes for stereo, whatever the length of the file.
pub struct TrackCompressor {
    source: Lossless,
    sample_rate: u32,
    channels: u32,
    /// How many bits of a decoded integer are significant — see
    /// [`Self::significant_bits`].
    significant_bits: u32,
    /// FLAC's stream description, needed for every frame it decodes.
    info: Option<StreamInfo>,
    /// Samples awaiting a transform, per channel, in minus one to one.
    pending: Vec<Vec<f64>>,
    /// The analysis window, built once.
    shape: Vec<f64>,
    /// Set when the trailing pad has been added, so it is added once.
    finished: bool,
}

impl TrackCompressor {
    /// Start compressing a track.
    ///
    /// # Errors
    ///
    /// [`CompressError::Undecodable`] when FLAC arrives without a
    /// readable `STREAMINFO`, and [`CompressError::Unencodable`] when
    /// the encoder will not take the rate or the layout. Both are
    /// refusals to begin rather than failures partway through a file.
    pub fn new(
        source: Lossless,
        codec_private: &[u8],
        sample_rate: u32,
        channels: u32,
    ) -> Result<TrackCompressor, CompressError> {
        if !sovmr_audio::encode::can_encode(sample_rate, channels) {
            return Err(CompressError::Unencodable(EncodeError::UnmeasuredRate(
                sample_rate,
            )));
        }
        let info = match source {
            Lossless::Flac => Some(
                StreamInfo::parse(codec_private)
                    .map_err(|_| CompressError::Undecodable("FLAC STREAMINFO is unreadable"))?,
            ),
            Lossless::Pcm(_) => None,
        };
        let significant_bits = match source {
            // Left-aligned into the top of an `i32` whatever the source
            // depth was, so the whole width is significant.
            Lossless::Pcm(_) => i32::BITS,
            Lossless::Flac => u32::from(info.expect("just parsed").bits_per_sample),
        };
        Ok(TrackCompressor {
            source,
            sample_rate,
            channels,
            significant_bits,
            info,
            // The leading pad: the first block's overlap has nothing
            // before it, so it ramps from silence rather than clipping.
            pending: vec![vec![0.0; LONG_FRAME]; channels as usize],
            shape: window(2 * LONG_FRAME, WindowShape::Sine),
            finished: false,
        })
    }

    /// How many bits of each decoded integer carry signal.
    ///
    /// The two decoders in `sovmr-audio` do not agree, and pretending
    /// they did cost this project an end-to-end failure no unit test
    /// could have seen: [`pcm::decode`] left-aligns every layout into an
    /// `i32`, while [`flac::decode_frame`] returns natural depth.
    /// Neither is wrong — they answer different questions — but dividing
    /// sixteen-bit FLAC by two to the thirty-first produced audio 65536
    /// times too quiet, which decoded perfectly, correlated perfectly,
    /// and could not be heard.
    pub fn significant_bits(&self) -> u32 {
        self.significant_bits
    }

    /// The `AudioSpecificConfig` the output container must carry.
    pub fn codec_private(&self) -> Vec<u8> {
        AudioSpecificConfig {
            object_type: AAC_LC,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
        .to_bytes()
    }

    /// Feed one source packet, taking whatever frames it completed.
    pub fn push(&mut self, packet: &[u8]) -> Result<Vec<Vec<u8>>, CompressError> {
        let interleaved = self.decode(packet)?;
        let stride = self.channels as usize;
        if interleaved.len() % stride != 0 {
            return Err(CompressError::Ragged {
                samples: interleaved.len(),
                channels: self.channels,
            });
        }
        let full_scale = 2f64.powi(self.significant_bits as i32 - 1);
        for (index, sample) in interleaved.iter().enumerate() {
            self.pending[index % stride].push(f64::from(*sample) / full_scale);
        }
        self.drain()
    }

    /// Finish the track: pad the tail and take the last frames.
    ///
    /// The final block's overlap needs something after it, so the tail
    /// ramps into silence rather than being cut off. Calling this twice
    /// yields nothing the second time.
    pub fn finish(&mut self) -> Result<Vec<Vec<u8>>, CompressError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        // Enough to flush every buffered sample through a whole block.
        let held = self.pending[0].len();
        let pad = (held.next_multiple_of(LONG_FRAME) + LONG_FRAME) - held;
        for channel in &mut self.pending {
            channel.resize(held + pad, 0.0);
        }
        self.drain()
    }

    /// Turn every complete block into a frame and forget its samples.
    fn drain(&mut self) -> Result<Vec<Vec<u8>>, CompressError> {
        let mut frames = Vec::new();
        while self.pending[0].len() >= 2 * LONG_FRAME {
            let mut blocks = Vec::with_capacity(self.pending.len());
            for channel in &self.pending {
                let windowed: Vec<f64> = (0..2 * LONG_FRAME)
                    .map(|i| channel[i] * self.shape[i])
                    .collect();
                let mut coefficients = vec![0.0; LONG_FRAME];
                mdct(&windowed, &mut coefficients);
                blocks.push(coefficients);
            }
            let frame = if self.channels == 1 {
                encode_frame(&blocks[0], self.sample_rate)
            } else {
                encode_frame_pair(&blocks[0], &blocks[1], self.sample_rate)
            }
            .map_err(CompressError::Unencodable)?;
            frames.push(frame);
            // Half the block is consumed; the other half is the overlap
            // the next one needs. This is the only memory that persists,
            // and it does not grow with the file.
            for channel in &mut self.pending {
                channel.drain(..LONG_FRAME);
            }
        }
        Ok(frames)
    }

    /// One packet's samples, interleaved.
    fn decode(&self, packet: &[u8]) -> Result<Vec<i32>, CompressError> {
        match self.source {
            Lossless::Pcm(format) => Ok(pcm::decode(packet, format)),
            Lossless::Flac => {
                let info = self.info.as_ref().expect("FLAC always parses one");
                let frame = flac::decode_frame(packet, info)
                    .map_err(|_| CompressError::Undecodable("a FLAC frame did not decode"))?;
                let count = frame.channels.first().map_or(0, Vec::len);
                let mut samples = Vec::with_capacity(count * frame.channels.len());
                for i in 0..count {
                    for channel in &frame.channels {
                        samples.push(channel[i]);
                    }
                }
                Ok(samples)
            }
        }
    }
}

/// A track's audio, re-encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compressed {
    /// One AAC frame per 1024 samples of each channel.
    pub frames: Vec<Vec<u8>>,
    /// The `AudioSpecificConfig` a container must carry beside them.
    pub codec_private: Vec<u8>,
    /// Channels the output declares.
    pub channels: u32,
}

/// Compress a whole track at once.
///
/// A convenience over [`TrackCompressor`] for callers that already hold
/// every packet — tests, and short tracks. Anything long should drive
/// the compressor directly and mux each frame as it appears, because
/// this collects the lot.
pub fn compress(
    source: Lossless,
    packets: &[&[u8]],
    codec_private: &[u8],
    sample_rate: u32,
    channels: u32,
) -> Result<Compressed, CompressError> {
    let mut compressor = TrackCompressor::new(source, codec_private, sample_rate, channels)?;
    let mut frames = Vec::new();
    for packet in packets {
        frames.extend(compressor.push(packet)?);
    }
    frames.extend(compressor.finish()?);
    Ok(Compressed {
        frames,
        codec_private: compressor.codec_private(),
        channels,
    })
}

/// One line saying what happened to a file's audio.
///
/// `None` when there is nothing worth saying — every track carried, and
/// none dropped, is the ordinary case and does not need announcing.
pub fn describe(routes: &[Route]) -> Option<String> {
    let carried = routes.iter().filter(|r| **r == Route::Carry).count();
    let mut compressed: Vec<&'static str> = routes
        .iter()
        .filter_map(|r| match r {
            Route::Compress(source) => Some(source.name()),
            _ => None,
        })
        .collect();
    let dropped: Vec<&&'static str> = routes
        .iter()
        .filter_map(|r| match r {
            Route::Drop(reason) => Some(reason),
            _ => None,
        })
        .collect();
    if compressed.is_empty() && dropped.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    if !compressed.is_empty() {
        compressed.sort_unstable();
        compressed.dedup();
        parts.push(format!(
            "{} track(s) compressed ({} → AAC)",
            routes
                .iter()
                .filter(|r| matches!(r, Route::Compress(_)))
                .count(),
            compressed.join(", ")
        ));
    }
    if carried > 0 {
        parts.push(format!("{carried} carried losslessly"));
    }
    for reason in dropped {
        parts.push(format!("1 dropped: {reason}"));
    }
    Some(format!("audio: {}", parts.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_audio_is_carried_rather_than_re_encoded() {
        // The best possible re-encode of already-lossy audio is still
        // worse than not doing one. Carrying must win even where this
        // project could encode the result perfectly well.
        for (codec, private) in [
            ("A_AAC", &b"\x11\x90"[..]),
            ("A_OPUS", &b"OpusHead"[..]),
            ("A_AC3", &b""[..]),
        ] {
            assert_eq!(
                route(codec, private, 48_000, 2, Target::Mkv),
                Route::Carry,
                "{codec}"
            );
        }
    }

    #[test]
    fn lossless_audio_is_compressed_whichever_container_it_is_going_into() {
        // This is a compressor. A twelve-hour film's PCM track is four
        // gigabytes that Matroska would happily copy byte for byte, and
        // copying it is the tool failing at the one thing it was asked
        // to do. The source is exact, so encoding costs no generation.
        let info = flac_streaminfo(48_000, 2);
        for target in [Target::Mp4, Target::Mkv] {
            assert_eq!(
                route("A_FLAC", &info, 48_000, 2, target),
                Route::Compress(Lossless::Flac),
                "{target:?}"
            );
            assert_eq!(
                route("A_PCM/INT/LIT", &[], 48_000, 1, target),
                Route::Compress(Lossless::Pcm(PcmFormat::S16Le)),
                "{target:?}"
            );
        }
    }

    #[test]
    fn a_lossless_track_the_encoder_cannot_take_is_carried_rather_than_lost() {
        // Six-channel FLAC has no element to go in yet. Keeping it large
        // is a far better answer than dropping it, and Matroska can.
        let info = flac_streaminfo(48_000, 6);
        assert_eq!(route("A_FLAC", &info, 48_000, 6, Target::Mkv), Route::Carry);
        // Into MP4 there is no such fallback — `esds` describes MPEG-4
        // Audio and nothing else — so it is dropped, by name.
        assert!(matches!(
            route("A_FLAC", &info, 48_000, 6, Target::Mp4),
            Route::Drop(_)
        ));
    }

    #[test]
    fn a_limit_is_named_precisely_rather_than_called_unsupported() {
        // The three walls have three different answers — one waits on an
        // element layout, one on a measurement, one on a decoder — so a
        // message that conflated them would send the user looking in the
        // wrong place.
        let info = flac_streaminfo(48_000, 6);
        let reason = |r: Route| match r {
            Route::Drop(reason) => reason,
            other => panic!("expected a drop, got {other:?}"),
        };
        assert!(reason(route("A_FLAC", &info, 48_000, 6, Target::Mp4)).contains("two channels"));
        assert!(reason(route("A_PCM/INT/LIT", &[], 47_999, 2, Target::Mp4))
            .contains("scalefactor band table"));
        assert!(reason(route("A_DTS", &[], 48_000, 6, Target::Mp4)).contains("no decoder"));
    }

    #[test]
    fn flac_without_its_streaminfo_is_not_claimed_as_decodable() {
        // Block size, depth and channel count all live in STREAMINFO.
        // Claiming the track and failing later would turn a clear
        // refusal into a mid-transcode error.
        assert!(matches!(
            route("A_FLAC", &[], 48_000, 2, Target::Mp4),
            Route::Drop(_)
        ));
        assert!(matches!(
            route("A_FLAC", &[0u8; 4], 48_000, 2, Target::Mp4),
            Route::Drop(_)
        ));
    }

    #[test]
    fn a_codec_with_no_decoder_is_dropped_by_name() {
        assert!(matches!(
            route("A_DTS", &[], 48_000, 6, Target::Mp4),
            Route::Drop(_)
        ));
        assert!(matches!(
            route("", &[], 48_000, 2, Target::Mkv),
            Route::Drop(_)
        ));
    }

    #[test]
    fn the_ordinary_case_says_nothing() {
        // Every track carried is what usually happens, and announcing it
        // would train the user to ignore the line that matters.
        assert_eq!(describe(&[Route::Carry, Route::Carry]), None);
        assert_eq!(describe(&[]), None);
    }

    #[test]
    fn a_mixed_file_is_described_in_one_line() {
        let routes = vec![
            Route::Compress(Lossless::Pcm(PcmFormat::S16Le)),
            Route::Compress(Lossless::Pcm(PcmFormat::S24Le)),
            Route::Carry,
            Route::Carry,
            Route::Carry,
            Route::Carry,
        ];
        assert_eq!(
            describe(&routes).as_deref(),
            Some("audio: 2 track(s) compressed (PCM → AAC), 4 carried losslessly")
        );
    }

    #[test]
    fn two_lossless_kinds_are_both_named() {
        let routes = vec![
            Route::Compress(Lossless::Flac),
            Route::Compress(Lossless::Pcm(PcmFormat::S16Le)),
            Route::Drop("more than two channels cannot be laid out yet"),
        ];
        let line = describe(&routes).expect("something happened");
        assert!(line.contains("FLAC, PCM → AAC"), "{line}");
        assert!(line.contains("1 dropped: more than two channels"), "{line}");
    }

    #[test]
    fn a_track_whose_samples_do_not_divide_by_its_channels_is_refused() {
        // An odd sample count with two channels means the channel count
        // is wrong. Encoding it anyway would silently shift every
        // sample of the right channel into the left.
        let packet = vec![0u8; 6]; // three 16-bit samples
        let result = compress(Lossless::Pcm(PcmFormat::S16Le), &[&packet], &[], 48_000, 2);
        assert_eq!(
            result,
            Err(CompressError::Ragged {
                samples: 3,
                channels: 2
            })
        );
    }

    #[test]
    fn a_rate_the_encoder_refuses_comes_back_as_that_refusal() {
        let packet = vec![0u8; 4096];
        assert!(matches!(
            compress(Lossless::Pcm(PcmFormat::S16Le), &[&packet], &[], 47_999, 1),
            Err(CompressError::Unencodable(EncodeError::UnmeasuredRate(
                47_999
            )))
        ));
    }

    #[test]
    fn what_the_compressor_holds_does_not_grow_with_the_file() {
        // The reason this is a machine and not a function. A twelve-hour
        // stereo track decodes to some eight gigabytes of samples, and a
        // compressor that accumulated them would undo the streaming the
        // rest of the pipeline was built for.
        let mut compressor =
            TrackCompressor::new(Lossless::Pcm(PcmFormat::S16Le), &[], 48_000, 2).expect("starts");
        let packet: Vec<u8> = (0..8_192u32)
            .flat_map(|i| ((i as i16).wrapping_mul(97)).to_le_bytes())
            .collect();
        let mut frames = 0usize;
        let mut high_water = 0usize;
        for _ in 0..40 {
            frames += compressor.push(&packet).expect("compresses").len();
            high_water = high_water.max(compressor.pending.iter().map(Vec::len).sum::<usize>());
        }
        frames += compressor.finish().expect("finishes").len();

        // Forty packets is a hundred and sixty thousand samples per
        // channel — a hundred and sixty frames — and the buffer never
        // exceeds two blocks per channel, which is what the transform
        // needs and the only thing it is entitled to keep.
        assert!(
            frames > 100,
            "only {frames} frames, so the fixture proves little"
        );
        assert!(
            high_water <= 2 * 2 * LONG_FRAME,
            "the compressor held {high_water} samples, which grows with the file"
        );
    }

    #[test]
    fn packets_arriving_in_any_chunking_give_the_same_frames() {
        // A container may split PCM however it likes, and AAC's block
        // length has nothing to do with either. One big packet and many
        // small ones must produce byte-identical output, or the frames a
        // file gets would depend on how it happened to be stored.
        let whole: Vec<u8> = (0..20_000u32)
            .flat_map(|i| {
                let phase = f64::from(i) * std::f64::consts::TAU / 91.0;
                ((phase.sin() * 20_000.0) as i16).to_le_bytes()
            })
            .collect();

        let one = compress(Lossless::Pcm(PcmFormat::S16Le), &[&whole], &[], 48_000, 1)
            .expect("compresses");
        let pieces: Vec<&[u8]> = whole.chunks(1_234 * 2).collect();
        let many =
            compress(Lossless::Pcm(PcmFormat::S16Le), &pieces, &[], 48_000, 1).expect("compresses");
        assert_eq!(one.frames, many.frames, "the chunking changed the output");
    }

    #[test]
    fn finishing_twice_yields_nothing_the_second_time() {
        // The trailing pad must be added once. Twice would append a
        // block of silence to every track that was flushed defensively.
        let mut compressor =
            TrackCompressor::new(Lossless::Pcm(PcmFormat::S16Le), &[], 48_000, 1).expect("starts");
        compressor.push(&vec![0u8; 8_192]).expect("compresses");
        let first = compressor.finish().expect("finishes");
        assert!(!first.is_empty(), "the tail produced no frames at all");
        assert!(compressor.finish().expect("finishes").is_empty());
    }

    #[test]
    fn a_track_the_encoder_cannot_take_is_refused_before_a_byte_is_read() {
        // Refusing to begin is worth a great deal more than failing
        // partway through a twelve-hour file.
        assert!(matches!(
            TrackCompressor::new(Lossless::Pcm(PcmFormat::S16Le), &[], 47_999, 1),
            Err(CompressError::Unencodable(_))
        ));
        assert!(matches!(
            TrackCompressor::new(Lossless::Pcm(PcmFormat::S16Le), &[], 48_000, 6),
            Err(CompressError::Unencodable(_))
        ));
        assert!(matches!(
            TrackCompressor::new(Lossless::Flac, &[], 48_000, 2),
            Err(CompressError::Undecodable(_))
        ));
    }

    #[test]
    fn the_two_decoders_disagree_about_scale_and_this_knows_it() {
        // PCM left-aligns into an i32; FLAC returns natural depth.
        // Assuming they matched made sixteen-bit FLAC come out 65536
        // times too quiet — a stream that decoded without complaint,
        // correlated perfectly with the original, and could not be
        // heard. This pins both conventions so neither can drift.
        let full_scale_16 = pcm::decode(&[0xFF, 0x7F], PcmFormat::S16Le)[0];
        assert!(
            full_scale_16 > i32::MAX / 2,
            "PCM stopped left-aligning: full scale came back as {full_scale_16}"
        );
        // And the compressed output of a loud source must be loud. A
        // scale error shows up here and almost nowhere else.
        let packet: Vec<u8> = (0..4_000u32)
            .flat_map(|i| {
                let phase = f64::from(i) * std::f64::consts::TAU / 64.0;
                ((phase.sin() * 30_000.0) as i16).to_le_bytes()
            })
            .collect();
        let out = compress(Lossless::Pcm(PcmFormat::S16Le), &[&packet], &[], 48_000, 1)
            .expect("compresses");
        // A silent frame codes no bands, so its `max_sfb` is zero and it
        // is tiny. Loud frames are not.
        let biggest = out.frames.iter().map(Vec::len).max().unwrap_or(0);
        assert!(
            biggest > 64,
            "the loudest frame is {biggest} bytes, which is what silence looks like"
        );
    }

    #[test]
    fn pcm_compresses_into_one_frame_per_block() {
        // Two thousand samples is more than one AAC block and less than
        // two, so the overlap alone decides the count — and it must be
        // at least three, since the first and last carry only a ramp.
        let packet: Vec<u8> = (0..4_000u32)
            .flat_map(|i| ((i as i16).wrapping_mul(37)).to_le_bytes())
            .collect();
        let out = compress(Lossless::Pcm(PcmFormat::S16Le), &[&packet], &[], 48_000, 1)
            .expect("compresses");
        assert!(out.frames.len() >= 3, "{} frames", out.frames.len());
        assert_eq!(out.channels, 1);
        assert_eq!(out.codec_private, vec![0x11, 0x88], "48 kHz mono config");
    }

    /// A minimal FLAC `STREAMINFO` block: enough for the parser to
    /// accept, which is all the routing decision looks at.
    fn flac_streaminfo(sample_rate: u32, channels: u32) -> Vec<u8> {
        let mut out = vec![0u8; 34];
        // Block sizes: 4096 both ways.
        out[0..2].copy_from_slice(&4096u16.to_be_bytes());
        out[2..4].copy_from_slice(&4096u16.to_be_bytes());
        // 20 bits of rate, 3 of channels less one, 5 of depth less one.
        let packed =
            (u64::from(sample_rate) << 44) | (u64::from(channels - 1) << 41) | (15u64 << 36);
        out[10..18].copy_from_slice(&packed.to_be_bytes());
        out
    }
}
