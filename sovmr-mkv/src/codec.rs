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

//! Turning Matroska's codec names into what the engine already speaks.
//!
//! Matroska identifies codecs by string (`V_MPEG4/ISO/AVC`) where
//! ISOBMFF uses a fourcc (`avc1`), but underneath they carry the *same*
//! bytes: a Matroska H.264 track's `CodecPrivate` is an `avcC` record,
//! and its frames are the same length-prefixed NAL units an MP4 sample
//! holds.
//!
//! That is why this module borrows [`sovmr_mp4`]'s readers instead of
//! growing its own. An `avcC` is an ISOBMFF structure wherever it is
//! stored, and a second implementation would be a second thing to keep
//! correct.

use sovmr_mp4::demux::{self, DemuxError, VideoCodec};

use crate::demux::Track;

/// Matroska codec identifiers this engine can decode.
const V_H264: &str = "V_MPEG4/ISO/AVC";
const V_HEVC: &str = "V_MPEGH/ISO/HEVC";
const V_AV1: &str = "V_AV1";

/// Matroska identifier for AAC, the one audio codec the MP4 muxer can
/// re-emit bit-exactly today.
const A_AAC: &str = "A_AAC";

/// The video codec a Matroska track carries, in the engine's terms.
///
/// `None` for anything sovmr cannot decode — the caller reports that
/// rather than guessing, since feeding NVDEC a stream it cannot parse
/// fails at the first frame.
pub fn video_codec(codec_id: &str) -> Option<VideoCodec> {
    match codec_id {
        V_H264 => Some(VideoCodec::H264),
        V_HEVC => Some(VideoCodec::Hevc),
        V_AV1 => Some(VideoCodec::Av1),
        _ => None,
    }
}

/// Whether an audio track can be carried through into MP4 bit-exactly.
///
/// Matroska spells AAC variants as `A_AAC/MPEG4/LC` and friends, so the
/// test is a prefix rather than equality — the profile suffix does not
/// change that the payload is AAC in an `esds`.
pub fn is_passthrough_audio(codec_id: &str) -> bool {
    codec_id == A_AAC || codec_id.starts_with("A_AAC/")
}

/// How many bytes prefix each NAL unit in this track's frames.
///
/// Read from the track's `CodecPrivate`, which for H.264 and HEVC is an
/// `avcC` or `hvcC` record — the same structure MP4 stores, parsed by
/// the same reader.
///
/// # Errors
///
/// [`DemuxError::BadConfigRecord`] when `CodecPrivate` is missing,
/// truncated, or does not declare `configurationVersion == 1`. AV1
/// tracks answer zero: their frames are raw OBUs with no prefixes.
pub fn nal_length_size(track: &Track<'_>) -> Result<u8, DemuxError> {
    let codec =
        video_codec(track.codec_id).ok_or(DemuxError::BadConfigRecord("unsupported codec id"))?;
    if codec == VideoCodec::Av1 {
        return Ok(0);
    }
    if track.codec_private.is_empty() {
        return Err(DemuxError::BadConfigRecord("CodecPrivate absent"));
    }
    demux::nal_length_size(codec, track.codec_private)
}

/// Rewrite one Matroska frame as Annex B, ready for NVDEC.
///
/// `out` is cleared. On a keyframe the track's parameter sets are
/// prepended, exactly as the MP4 path does: NVDEC needs an SPS/PPS in
/// front of every sync sample, not merely the first.
///
/// # Errors
///
/// [`DemuxError::WrongCodec`] for AV1 (its frames are already raw OBUs
/// and need no rewriting), and [`DemuxError::BadNalLength`] when a
/// length prefix describes more bytes than the frame holds.
pub fn write_frame(
    track: &Track<'_>,
    frame: &[u8],
    keyframe: bool,
    out: &mut Vec<u8>,
) -> Result<(), DemuxError> {
    let codec =
        video_codec(track.codec_id).ok_or(DemuxError::BadConfigRecord("unsupported codec id"))?;
    if codec == VideoCodec::Av1 {
        // Already the elementary form: raw OBUs, each carrying its own
        // size, with a keyframe's sequence header inside the frame.
        // Nothing to frame and nothing to prepend.
        out.clear();
        out.extend_from_slice(frame);
        return Ok(());
    }
    let prefix = nal_length_size(track)?;
    out.clear();
    if keyframe {
        for ps in demux::config_parameter_sets(codec, track.codec_private) {
            let ps = ps?;
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(ps);
        }
    }
    demux::annex_b_from_length_prefixed(frame, prefix, 0, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track_with<'a>(codec_id: &'a str, private: &'a [u8]) -> Track<'a> {
        Track {
            number: 1,
            kind: crate::demux::TrackKind::Video,
            codec_id,
            codec_private: private,
            language: "eng",
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
        }
    }

    /// A minimal `avcC`: version 1, three profile bytes, then the
    /// length-size byte, then one SPS and one PPS.
    fn avcc(length_size_minus_one: u8) -> Vec<u8> {
        let mut c = vec![1, 0x64, 0x00, 0x1F, 0xFC | length_size_minus_one, 0xE1];
        c.extend_from_slice(&[0, 4, 0x67, 0x64, 0x00, 0x1F]); // SPS
        c.push(1);
        c.extend_from_slice(&[0, 3, 0x68, 0xEB, 0xE3]); // PPS
        c
    }

    #[test]
    fn the_codec_names_this_engine_decodes_are_recognised() {
        assert_eq!(video_codec("V_MPEG4/ISO/AVC"), Some(VideoCodec::H264));
        assert_eq!(video_codec("V_MPEGH/ISO/HEVC"), Some(VideoCodec::Hevc));
        assert_eq!(video_codec("V_AV1"), Some(VideoCodec::Av1));
    }

    #[test]
    fn a_codec_we_cannot_decode_is_none_rather_than_a_guess() {
        // Feeding NVDEC a stream it cannot parse fails at the first
        // frame, hours into a batch. Saying so up front is the whole job.
        for id in ["V_VP9", "V_MPEG2", "V_THEORA", "", "v_mpeg4/iso/avc"] {
            assert_eq!(video_codec(id), None, "{id} must not be claimed");
        }
    }

    #[test]
    fn aac_is_recognised_with_or_without_its_profile_suffix() {
        assert!(is_passthrough_audio("A_AAC"));
        assert!(is_passthrough_audio("A_AAC/MPEG4/LC"));
        assert!(is_passthrough_audio("A_AAC/MPEG2/LC"));
        // Everything else must be reported as uncarryable, not smuggled
        // into an MP4 the muxer cannot honestly write.
        for id in ["A_OPUS", "A_FLAC", "A_AC3", "A_VORBIS", "A_AACX"] {
            assert!(!is_passthrough_audio(id), "{id}");
        }
    }

    #[test]
    fn the_nal_prefix_width_comes_from_codec_private() {
        for minus_one in 0..4u8 {
            let record = avcc(minus_one);
            let t = track_with(V_H264, &record);
            assert_eq!(nal_length_size(&t).unwrap(), minus_one + 1);
        }
    }

    #[test]
    fn a_track_without_codec_private_is_an_error_not_a_default() {
        // Assuming four-byte prefixes would misread every frame of a
        // track that used something else, silently.
        let t = track_with(V_H264, &[]);
        assert!(matches!(
            nal_length_size(&t),
            Err(DemuxError::BadConfigRecord(_))
        ));
    }

    #[test]
    fn av1_frames_are_handed_over_exactly_as_stored() {
        // AV1 samples are already the elementary form a decoder wants:
        // raw OBUs, each carrying its own size, with a keyframe's
        // sequence header inside the frame. Adding start codes or
        // prepending parameter sets would corrupt them.
        //
        // This used to be a refusal, which was right while nothing could
        // decode AV1. Reading back what the engine has just written
        // needs it, and the frames were always ready to be read.
        let t = track_with(V_AV1, &[0x81, 0x00]);
        assert_eq!(nal_length_size(&t).unwrap(), 0);
        let frame = [0x12u8, 0x00, 0x0A, 0xFF];
        for keyframe in [true, false] {
            let mut out = vec![0xEE; 3];
            assert_eq!(write_frame(&t, &frame, keyframe, &mut out), Ok(()));
            assert_eq!(
                out,
                frame.to_vec(),
                "keyframe {keyframe}: the frame was altered on the way out"
            );
        }
    }

    #[test]
    fn a_keyframe_gets_its_parameter_sets_prepended() {
        // NVDEC needs an SPS/PPS in front of every sync sample, not just
        // the first — the same rule the MP4 path already honours.
        let record = avcc(3);
        let t = track_with(V_H264, &record);
        let frame = [0, 0, 0, 2, 0x65, 0xAA];
        let mut out = Vec::new();
        write_frame(&t, &frame, true, &mut out).unwrap();

        let starts = out.windows(4).filter(|w| *w == [0, 0, 0, 1]).count();
        assert_eq!(starts, 3, "SPS + PPS + the frame itself");
        assert_eq!(&out[out.len() - 2..], &[0x65, 0xAA]);
    }

    #[test]
    fn a_non_keyframe_carries_only_its_own_nals() {
        let record = avcc(3);
        let t = track_with(V_H264, &record);
        let frame = [0, 0, 0, 2, 0x41, 0xBB];
        let mut out = Vec::new();
        write_frame(&t, &frame, false, &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 0, 1, 0x41, 0xBB]);
    }

    #[test]
    fn a_length_prefix_longer_than_the_frame_is_refused() {
        // Negative trust: the prefix is the writer's claim. A decoder fed
        // the remainder would produce silent garbage.
        let record = avcc(3);
        let t = track_with(V_H264, &record);
        let frame = [0, 0, 0, 99, 0x41];
        let mut out = Vec::new();
        assert!(matches!(
            write_frame(&t, &frame, false, &mut out),
            Err(DemuxError::BadNalLength { .. })
        ));
    }
}
