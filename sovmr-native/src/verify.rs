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

//! Checking an encode against the source it came from.
//!
//! # Why the program has to do this itself
//!
//! Every quality number this project has published so far came from
//! `ffmpeg -lavfi psnr`. That is fine for a development benchmark and
//! useless for a shipped tool: a user cannot be told to install the
//! thing this program exists to replace in order to find out whether it
//! worked.
//!
//! # How the two files are kept in step
//!
//! Both are decoded by this project's own NVDEC bridge, and the parser
//! is push-driven with inline callbacks — feeding one frame yields zero
//! or more decoded pictures, depending on reorder depth. So each side is
//! fed one frame at a time into its own queue, and a comparison happens
//! whenever both queues have something.
//!
//! That bounds memory to the reorder depth rather than the file: a
//! twelve-hour comparison holds a handful of frames, not twelve hours of
//! them.
//!
//! # What is compared
//!
//! Luma only, and by [`sovmr_core::quality`], which is where the
//! reasoning about *what* the numbers mean lives. This module's job is
//! to get two matching pictures into host memory and hand them over.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::Path;

use sovmr_core::quality::{self, Plane};
use sovmr_nvdec::parser::VideoFormat;
use sovmr_nvdec::{DecodeCodec, DecoderConfig, NvDec};

use crate::pipeline::NativeError;
use crate::source::Source;

/// How many decoded frames one side may run ahead of the other.
///
/// Reorder depth plus a little: a stream whose two sides diverge by more
/// than this is not two encodings of one source, and continuing would
/// compare unrelated pictures and report a number for them.
const MAX_QUEUE: usize = 16;

/// What a comparison found.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// Frames compared.
    pub frames: u64,
    /// Peak signal-to-noise ratio over the whole file, in decibels.
    ///
    /// Computed from the summed error rather than by averaging per-frame
    /// decibels: the mean of a logarithm is not the logarithm of a mean,
    /// and the second is a different number wearing the same name.
    pub psnr: f64,
    /// Structural similarity, averaged over frames.
    pub ssim: f64,
    /// Frames compared per second, so the cost of measuring is known
    /// rather than assumed.
    pub fps: f64,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.psnr.is_infinite() {
            write!(f, "{} frames, identical", self.frames)?;
        } else {
            write!(f, "{} frames, PSNR {:.2} dB", self.frames, self.psnr)?;
        }
        write!(f, ", SSIM {:.4}, {:.0} fps", self.ssim, self.fps)
    }
}

/// One decoded luma plane in host memory.
struct HostFrame {
    samples: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
}

impl HostFrame {
    fn plane(&self) -> Plane<'_> {
        Plane {
            samples: &self.samples,
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }
}

/// Everything one side of the comparison needs while it runs.
struct Side<'a> {
    decoder: Option<sovmr_nvdec::Decoder<'a>>,
    /// Coded surface dimensions, from the sequence callback. Not the
    /// display size: NVDEC pads to alignment, and reading a plane by the
    /// display height would shear it.
    surface: Option<(u32, u32)>,
    /// Display crop, which is what actually gets compared.
    display: Option<(u32, u32)>,
    ready: VecDeque<HostFrame>,
    error: Option<String>,
}

impl Side<'_> {
    fn new() -> Self {
        Side {
            decoder: None,
            surface: None,
            display: None,
            ready: VecDeque::new(),
            error: None,
        }
    }
}

/// Decode one file and count the frames that come back.
///
/// The half of [`verify`] that involves one decoder rather than two —
/// worth having on its own, because a fault in the decode loop and a
/// fault in running two of them at once are different problems and this
/// tells them apart.
pub fn frame_count(path: &Path) -> Result<u64, NativeError> {
    let bytes = sovmr_os::MappedFile::open(path)
        .map_err(|e| NativeError::Pipeline(format!("{}: {e}", path.display())))?;
    let cuda = sovmr_cuda::Cuda::load()?;
    let ctx = cuda.create_context(0)?;
    let nvdec = NvDec::load()?;
    let side = RefCell::new(Side::new());
    let source = Source::open(&bytes)?;
    let codec = decode_codec(&source);
    let mut frames = source.video_frames(&bytes);
    let mut buffer = Vec::new();
    let mut parser = build_parser(&nvdec, &ctx, &side, codec)?;
    let mut counted = 0u64;
    let mut fed = 0i64;
    while feed_one(&mut parser, &mut frames, &mut buffer, fed)? {
        fed += 1;
        counted += side.borrow_mut().ready.drain(..).count() as u64;
        if let Some(message) = side.borrow().error.clone() {
            return Err(NativeError::Pipeline(message));
        }
    }
    parser.end_of_stream()?;
    counted += side.borrow_mut().ready.drain(..).count() as u64;
    Ok(counted)
}

/// Compare an encode against its source.
///
/// # Errors
///
/// Anything that stops the comparison being honest: a file that will not
/// open, a codec the bridge cannot decode, pictures of different sizes,
/// or the two sides drifting further apart than [`MAX_QUEUE`].
pub fn verify(source_path: &Path, encoded_path: &Path) -> Result<Verdict, NativeError> {
    let source_bytes = sovmr_os::MappedFile::open(source_path)
        .map_err(|e| NativeError::Pipeline(format!("{}: {e}", source_path.display())))?;
    let encoded_bytes = sovmr_os::MappedFile::open(encoded_path)
        .map_err(|e| NativeError::Pipeline(format!("{}: {e}", encoded_path.display())))?;

    let cuda = sovmr_cuda::Cuda::load()?;
    let ctx = cuda.create_context(0)?;
    let nvdec = NvDec::load()?;

    let started = std::time::Instant::now();
    let mut total_mse = 0.0f64;
    let mut total_ssim = 0.0f64;
    let mut frames = 0u64;

    let left = RefCell::new(Side::new());
    let right = RefCell::new(Side::new());
    // Sources and their cursors live here so the cursors can be advanced
    // across calls. A reader that rebuilt its cursor each time would be
    // quadratic, which on a twelve-hour file is not a detail.
    let left_source = Source::open(&source_bytes)?;
    let right_source = Source::open(&encoded_bytes)?;
    let left_codec = decode_codec(&left_source);
    let right_codec = decode_codec(&right_source);
    let mut left_frames = left_source.video_frames(&source_bytes);
    let mut right_frames = right_source.video_frames(&encoded_bytes);
    let mut left_buffer = Vec::new();
    let mut right_buffer = Vec::new();
    let mut fed = 0i64;

    // Two parsers, each writing into its own side. Callbacks fire inline
    // during `feed`, so nothing here runs concurrently and a plain
    // `RefCell` is the whole synchronisation story.
    let mut left_parser = build_parser(&nvdec, &ctx, &left, left_codec)?;
    let mut right_parser = build_parser(&nvdec, &ctx, &right, right_codec)?;

    let mut left_done = false;
    let mut right_done = false;
    loop {
        if !left_done && left.borrow().ready.len() < MAX_QUEUE {
            left_done = !feed_one(&mut left_parser, &mut left_frames, &mut left_buffer, fed)?;
            if left_done {
                left_parser.end_of_stream()?;
            }
        }
        if !right_done && right.borrow().ready.len() < MAX_QUEUE {
            right_done = !feed_one(&mut right_parser, &mut right_frames, &mut right_buffer, fed)?;
            fed += 1;
            if right_done {
                right_parser.end_of_stream()?;
            }
        }
        if let Some(message) = left.borrow().error.clone().or(right.borrow().error.clone()) {
            return Err(NativeError::Pipeline(message));
        }

        let pair = {
            let mut l = left.borrow_mut();
            let mut r = right.borrow_mut();
            match (l.ready.pop_front(), r.ready.pop_front()) {
                (Some(a), Some(b)) => Some((a, b)),
                // Put back whichever side had one: the other will catch
                // up on the next round, or the loop ends.
                (Some(a), None) => {
                    l.ready.push_front(a);
                    None
                }
                (None, Some(b)) => {
                    r.ready.push_front(b);
                    None
                }
                (None, None) => None,
            }
        };
        match pair {
            Some((a, b)) => {
                total_mse += quality::mean_squared_error(&a.plane(), &b.plane())
                    .map_err(|e| NativeError::Pipeline(format!("frame {frames}: {e:?}")))?;
                total_ssim += quality::ssim(&a.plane(), &b.plane())
                    .map_err(|e| NativeError::Pipeline(format!("frame {frames}: {e:?}")))?;
                frames += 1;
            }
            None => {
                if left_done && right_done {
                    let both_empty =
                        left.borrow().ready.is_empty() && right.borrow().ready.is_empty();
                    if both_empty {
                        break;
                    }
                }
            }
        }
        // Neither side can grow past the queue bound while the other is
        // stuck: that is not two encodings of one source.
        if left.borrow().ready.len() >= MAX_QUEUE && right.borrow().ready.is_empty() && right_done {
            return Err(NativeError::Pipeline(
                "the encode ran out of frames while the source had more".into(),
            ));
        }
        if right.borrow().ready.len() >= MAX_QUEUE && left.borrow().ready.is_empty() && left_done {
            return Err(NativeError::Pipeline(
                "the encode has more frames than the source".into(),
            ));
        }
    }

    if frames == 0 {
        return Err(NativeError::Pipeline(
            "nothing was decoded from either file".into(),
        ));
    }
    let elapsed = started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
    Ok(Verdict {
        frames,
        psnr: quality::psnr_from_mse(total_mse / frames as f64),
        ssim: total_ssim / frames as f64,
        fps: frames as f64 / elapsed,
    })
}

/// Feed the parser one frame. `false` when the file has run out.
fn feed_one(
    parser: &mut sovmr_nvdec::parser::Parser<'_>,
    frames: &mut crate::source::VideoFrames<'_, '_>,
    buffer: &mut Vec<u8>,
    timestamp: i64,
) -> Result<bool, NativeError> {
    match frames.next_frame(buffer) {
        None => Ok(false),
        Some(Err(e)) => Err(e),
        Some(Ok(_)) => {
            parser.feed(buffer, timestamp)?;
            Ok(true)
        }
    }
}

/// Which decoder a source needs.
fn decode_codec(source: &Source<'_>) -> DecodeCodec {
    match source.video_facts().codec {
        sovmr_mp4::demux::VideoCodec::H264 => DecodeCodec::H264,
        sovmr_mp4::demux::VideoCodec::Hevc => DecodeCodec::Hevc,
        sovmr_mp4::demux::VideoCodec::Av1 => DecodeCodec::Av1,
    }
}

/// Wire a parser to one side's queue.
fn build_parser<'a, 'b>(
    nvdec: &'a NvDec,
    ctx: &'a sovmr_cuda::CudaContext<'a>,
    side: &'b RefCell<Side<'a>>,
    codec: DecodeCodec,
) -> Result<sovmr_nvdec::parser::Parser<'b>, NativeError>
where
    'a: 'b,
{
    let on_sequence = move |fmt: &VideoFormat| -> Option<u32> {
        let mut s = side.borrow_mut();
        let cfg = DecoderConfig {
            codec,
            width: fmt.coded_width,
            height: fmt.coded_height,
            bit_depth_minus8: fmt.bit_depth_luma_minus8,
            // The same number returned to the parser below.
            decode_surfaces: u32::from(fmt.min_num_decode_surfaces).max(DECODE_SURFACES),
        };
        match nvdec.create_decoder(ctx, &cfg) {
            Ok(decoder) => {
                s.decoder = Some(decoder);
                s.surface = Some((fmt.coded_width, fmt.coded_height));
                // The crop the source actually shows. Comparing the
                // padded surface would score alignment rows nobody sees.
                let (left, top, right, bottom) = (
                    fmt.display_area[0],
                    fmt.display_area[1],
                    fmt.display_area[2],
                    fmt.display_area[3],
                );
                let width = (right - left).max(0) as u32;
                let height = (bottom - top).max(0) as u32;
                s.display = Some(if width > 0 && height > 0 {
                    (width, height)
                } else {
                    (fmt.coded_width, fmt.coded_height)
                });
                // The driver's own minimum, never a guess. Too few
                // decode surfaces and it writes past the ring, which
                // shows up as a segmentation fault inside libnvcuvid
                // rather than as an error.
                Some(u32::from(fmt.min_num_decode_surfaces).max(DECODE_SURFACES))
            }
            Err(e) => {
                s.error = Some(format!("decoder: {e:?}"));
                None
            }
        }
    };
    let on_decode = move |pic: *mut std::ffi::c_void| -> bool {
        let s = side.borrow();
        match s.decoder.as_ref() {
            // SAFETY: `pic` is the driver's own picture parameters,
            // valid for this callback only.
            // SAFETY: `pic` is the driver's own picture parameters,
            // valid for the duration of this callback.
            Some(decoder) => unsafe { decoder.decode_picture(pic) }.is_ok(),
            None => false,
        }
    };
    let on_display = move |info: &sovmr_nvdec::parser::DisplayInfo| -> bool {
        // Map, copy, unmap — all while holding the decoder — and only
        // then take the queue. Holding both borrows at once is what the
        // borrow checker objects to, and it is right: the callback would
        // be reading a surface it had already handed on.
        let read = {
            let s = side.borrow();
            let (Some((_, surface_h)), Some((width, _)), Some((_, height))) =
                (s.surface, s.display, s.display)
            else {
                return false;
            };
            let Some(decoder) = s.decoder.as_ref() else {
                return false;
            };
            // Bound rather than matched in place: the temporary would
            // otherwise live to the end of the whole `let`, and the
            // mapped frame borrows the decoder borrowed from `s`.
            let mapped = decoder.map_frame(info.picture_index);
            match mapped {
                Err(e) => Err(format!("map: {e:?}")),
                Ok(frame) => {
                    // The mapped frame parks a slot in the decoder's
                    // ring and borrows the decoder, so it must be gone
                    // before this borrow of the side is.
                    // Luma only, and only the rows the picture has: the
                    // chroma plane starts at pitch x coded height, past
                    // everything this reads.
                    let stride = frame.pitch() as usize;
                    let rows = (height as usize).min(surface_h as usize);
                    let mut samples = vec![0u8; stride * rows];
                    // SAFETY: the mapped luma plane is `pitch * coded
                    // height` bytes and `rows <= coded_height`.
                    let out = match unsafe { ctx.copy_to_host(&mut samples, frame.device_ptr()) } {
                        Err(e) => Err(format!("read back: {e:?}")),
                        Ok(()) => Ok(HostFrame {
                            samples,
                            width: width as usize,
                            height: rows,
                            stride,
                        }),
                    };
                    drop(frame);
                    out
                }
            }
        };
        let mut s = side.borrow_mut();
        match read {
            Ok(frame) => {
                s.ready.push_back(frame);
                true
            }
            Err(message) => {
                s.error = Some(message);
                false
            }
        }
    };
    let callbacks = sovmr_nvdec::parser::ParserCallbacks {
        on_sequence: Box::new(on_sequence),
        on_decode: Box::new(on_decode),
        on_display: Box::new(on_display),
    };
    Ok(nvdec.create_parser(codec, DECODE_SURFACES, callbacks)?)
}

/// Decode surfaces the parser is created with.
///
/// The sequence callback may ask for the driver's own
/// `min_num_decode_surfaces`, but it cannot ask for more than the parser
/// was built to hold — exceed it and the driver writes past its ring,
/// which arrives as a fault inside `cuvidDecodePicture` rather than as
/// an error.
///
/// Measured on this hardware: H.264 asks for 5, AV1 for 9. Sixteen
/// leaves room for a codec that asks for more without another crash
/// teaching it.
const DECODE_SURFACES: u32 = 16;
