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

//! NVDEC's push parser (`cuvidCreateVideoParser`) with its three C-ABI
//! callbacks wrapped safely: the trampolines are `extern "C"` functions
//! that recover the Rust closure from the parser's `pUserData`, run it
//! under `catch_unwind` (a panic must never unwind across the C
//! boundary — it becomes a clean "stop parsing" return instead), and
//! translate its result to the driver's 0/1 convention.
//!
//! Struct sizes/offsets pinned by a compiled C probe:
//! `CUVIDPARSERPARAMS` = 136 (callbacks at 48/56/64),
//! `CUVIDSOURCEDATAPACKET` = 32, `CUVIDEOFORMAT` = 64 (coded size at
//! 16/20), `CUVIDPARSERDISPINFO` = 24.

use std::ffi::{c_int, c_uint, c_ulong, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{NvDec, NvdecError, CUDA_SUCCESS};

/// `CUVID_PKT_ENDOFSTREAM`.
const PKT_ENDOFSTREAM: c_ulong = 0x01;
/// `CUVID_PKT_TIMESTAMP`: the packet's `timestamp` field is valid.
/// Without this flag the parser synthesizes its own 10 MHz-clock
/// timestamps and presentation order becomes unrecoverable.
const PKT_TIMESTAMP: c_ulong = 0x02;
/// `CUVID_PKT_ENDOFPICTURE`: the packet contains exactly one frame —
/// true for our feed (one demuxed sample per packet), and it lets the
/// parser emit the picture without waiting for the next packet.
const PKT_ENDOFPICTURE: c_ulong = 0x08;
/// Parser callback convention: nonzero = proceed, zero = stop.
const CB_FAIL: c_int = 0;

/// `CUVIDEOFORMAT`, complete (64 bytes; verified by C probe).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VideoFormat {
    /// `cudaVideoCodec` of the elementary stream.
    pub codec: i32,
    /// Frame rate numerator/denominator.
    pub frame_rate: [c_uint; 2],
    /// 1 = progressive.
    pub progressive_sequence: u8,
    /// 0 for 8-bit, 2 for 10-bit luma.
    pub bit_depth_luma_minus8: u8,
    /// Chroma bit depth minus 8.
    pub bit_depth_chroma_minus8: u8,
    /// Minimum decode surfaces the stream needs (DPB + reorder).
    pub min_num_decode_surfaces: u8,
    /// Coded width in pixels.
    pub coded_width: c_uint,
    /// Coded height in pixels.
    pub coded_height: c_uint,
    /// Visible rectangle (left, top, right, bottom).
    pub display_area: [i32; 4],
    /// `cudaVideoChromaFormat`.
    pub chroma_format: i32,
    /// Bitrate in bps when known, else 0.
    pub bitrate: c_uint,
    /// Display aspect ratio.
    pub display_aspect_ratio: [i32; 2],
    /// Packed video signal description bits.
    pub video_signal_description: u32,
    /// Trailing sequence header bytes count.
    pub seqhdr_data_length: u32,
}

/// `CUVIDPARSERDISPINFO` (24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DisplayInfo {
    /// Index into the decoder's surface ring — feed to `map_frame`.
    pub picture_index: c_int,
    /// 1 = progressive frame.
    pub progressive_frame: c_int,
    /// Interlacing order flag.
    pub top_field_first: c_int,
    /// Interlacing repeat flag.
    pub repeat_first_field: c_int,
    /// Presentation timestamp (10 MHz clock).
    pub timestamp: i64,
}

/// `CUVIDPARSERPARAMS` (136 bytes; pUserData at 40, callbacks at 48+).
#[repr(C)]
pub(crate) struct ParserParams {
    codec_type: i32,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    flags: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    pfn_sequence: Option<SequenceCb>,
    pfn_decode: Option<DecodeCb>,
    pfn_display: Option<DisplayCb>,
    pfn_operating_point: *mut c_void,
    pfn_sei_msg: *mut c_void,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

/// `CUVIDSOURCEDATAPACKET`.
///
/// The first two fields are `tcu_ulong` in the header, i.e. C's
/// `unsigned long` — **8 bytes on LP64 Linux, 4 on Windows' LLP64**.
/// They were declared `u64` here, which is right on Linux and silently
/// wrong on Windows: the driver read `payload_size` out of the padding
/// and the parser produced no frames at all, without ever returning an
/// error. `c_ulong` is the type that means the same thing on both.
///
/// The lesson is worth keeping next to the code: a size assert on our
/// own struct cannot catch this, because our struct was self-consistently
/// 32 bytes on both targets. Only the *type* was wrong, and only a live
/// decode revealed it.
#[repr(C)]
pub(crate) struct SourcePacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const u8,
    timestamp: i64,
}

type SequenceCb = unsafe extern "C" fn(*mut c_void, *mut VideoFormat) -> c_int;
type DecodeCb = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type DisplayCb = unsafe extern "C" fn(*mut c_void, *mut DisplayInfo) -> c_int;

pub(crate) type RawCreateParserFn =
    unsafe extern "C" fn(*mut *mut c_void, *mut ParserParams) -> i32;
pub(crate) type RawParseVideoDataFn = unsafe extern "C" fn(*mut c_void, *mut SourcePacket) -> i32;
pub(crate) type RawDestroyParserFn = unsafe extern "C" fn(*mut c_void) -> i32;

/// Sequence hook: returns the number of decode surfaces to commit
/// (>= `min_num_decode_surfaces`), or `None` to stop the stream.
pub type SequenceHook<'cb> = Box<dyn FnMut(&VideoFormat) -> Option<u32> + 'cb>;
/// Decode hook: forwards the opaque `CUVIDPICPARAMS` pointer; `false`
/// stops the stream.
pub type DecodeHook<'cb> = Box<dyn FnMut(*mut c_void) -> bool + 'cb>;
/// Display hook: a frame is ready in display order; `false` stops.
pub type DisplayHook<'cb> = Box<dyn FnMut(&DisplayInfo) -> bool + 'cb>;

/// The three pipeline hooks. Each returns "continue" or stops the
/// stream; a panic is caught at the ABI boundary and reads as a stop.
pub struct ParserCallbacks<'cb> {
    /// New sequence header: create/size the decoder here.
    pub on_sequence: SequenceHook<'cb>,
    /// A picture is ready to decode: forward the opaque `CUVIDPICPARAMS`
    /// pointer to [`crate::Decoder::decode_picture`].
    pub on_decode: DecodeHook<'cb>,
    /// A picture is ready for display order: map it, hand the VRAM
    /// pointer to the encoder.
    pub on_display: DisplayHook<'cb>,
}

unsafe extern "C" fn sequence_trampoline(user: *mut c_void, fmt: *mut VideoFormat) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user` is the ParserCallbacks the owning Parser keeps
        // alive for its whole lifetime; `fmt` is driver-owned and valid
        // for the duration of the callback.
        let cbs = unsafe { &mut *user.cast::<ParserCallbacks<'_>>() };
        // SAFETY: the driver passes a non-null CUVIDEOFORMAT* valid for
        // the duration of this callback; `VideoFormat` mirrors its
        // layout (size verified by the C probe) and we only read it.
        let fmt = unsafe { &*fmt };
        (cbs.on_sequence)(fmt)
    }));
    match result {
        Ok(Some(surfaces)) => surfaces.clamp(1, i32::MAX as u32) as c_int,
        Ok(None) => CB_FAIL,
        Err(_) => CB_FAIL, // panic contained at the ABI boundary
    }
}

unsafe extern "C" fn decode_trampoline(user: *mut c_void, pic: *mut c_void) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: as in sequence_trampoline.
        let cbs = unsafe { &mut *user.cast::<ParserCallbacks<'_>>() };
        (cbs.on_decode)(pic)
    }));
    matches!(result, Ok(true)) as c_int
}

unsafe extern "C" fn display_trampoline(user: *mut c_void, info: *mut DisplayInfo) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: as in sequence_trampoline.
        let cbs = unsafe { &mut *user.cast::<ParserCallbacks<'_>>() };
        // SAFETY: the driver passes a non-null CUVIDPARSERDISPINFO* valid
        // for the duration of this callback; `DisplayInfo` mirrors its
        // layout and we only read it.
        let info = unsafe { &*info };
        (cbs.on_display)(info)
    }));
    matches!(result, Ok(true)) as c_int
}

/// A live `CUvideoparser` bound to Rust callbacks. Destroyed on drop;
/// borrows the runtime and pins the callbacks box so the trampolines'
/// `user_data` can never dangle.
pub struct Parser<'cb> {
    raw: *mut c_void,
    parse: RawParseVideoDataFn,
    destroy: RawDestroyParserFn,
    /// Keep-alive: the trampolines' user_data points into this box.
    _callbacks: Box<ParserCallbacks<'cb>>,
}

impl NvDec {
    /// Create a push parser for `codec` (`cuvidCreateVideoParser`).
    /// `max_decode_surfaces` sizes the parser's ring hint; the sequence
    /// callback's return value commits the real number.
    ///
    /// # Errors
    ///
    /// [`NvdecError::Status`] with the symbolic `CUresult`.
    pub fn create_parser<'cb>(
        &self,
        codec: crate::DecodeCodec,
        max_decode_surfaces: u32,
        callbacks: ParserCallbacks<'cb>,
    ) -> Result<Parser<'cb>, NvdecError> {
        let mut callbacks = Box::new(callbacks);
        let mut params: Box<ParserParams> = Box::new(ParserParams {
            codec_type: crate::cuda_codec_id(codec),
            max_num_decode_surfaces: max_decode_surfaces.max(1),
            clock_rate: 0,
            error_threshold: 0,
            // Zero display delay: display callbacks fire as soon as
            // decode order allows — lowest latency for transcoding.
            max_display_delay: 0,
            flags: 0,
            reserved1: [0; 4],
            user_data: (callbacks.as_mut() as *mut ParserCallbacks<'_>).cast(),
            pfn_sequence: Some(sequence_trampoline),
            pfn_decode: Some(decode_trampoline),
            pfn_display: Some(display_trampoline),
            pfn_operating_point: std::ptr::null_mut(),
            pfn_sei_msg: std::ptr::null_mut(),
            reserved2: [std::ptr::null_mut(); 5],
            ext_video_info: std::ptr::null_mut(),
        });
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: params fully initialized; callbacks boxed at a stable
        // address that outlives the parser (owned by the returned value).
        let status = unsafe { (self.create_parser)(&mut raw, params.as_mut()) };
        if status != CUDA_SUCCESS || raw.is_null() {
            return Err(NvdecError::Status {
                call: "cuvidCreateVideoParser",
                code: status,
            });
        }
        Ok(Parser {
            raw,
            parse: self.parse_video_data,
            destroy: self.destroy_parser,
            _callbacks: callbacks,
        })
    }
}

impl Parser<'_> {
    /// Push one Annex B chunk into the parser; the callbacks fire
    /// synchronously on this thread before the call returns.
    ///
    /// # Errors
    ///
    /// [`NvdecError::Status`] — note that a callback returning `false`
    /// also surfaces as a parse error from the driver.
    pub fn feed(&mut self, data: &[u8], timestamp: i64) -> Result<(), NvdecError> {
        let mut packet = SourcePacket {
            flags: PKT_TIMESTAMP | PKT_ENDOFPICTURE,
            payload_size: data.len() as c_ulong,
            payload: data.as_ptr(),
            timestamp,
        };
        // SAFETY: packet points at caller-owned bytes valid for the call;
        // parser handle is alive.
        let status = unsafe { (self.parse)(self.raw, &mut packet) };
        if status != CUDA_SUCCESS {
            return Err(NvdecError::Status {
                call: "cuvidParseVideoData",
                code: status,
            });
        }
        Ok(())
    }

    /// Signal end of stream, flushing all pending display callbacks.
    ///
    /// # Errors
    ///
    /// [`NvdecError::Status`] as in [`Parser::feed`].
    pub fn end_of_stream(&mut self) -> Result<(), NvdecError> {
        let mut packet = SourcePacket {
            flags: PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: std::ptr::null(),
            timestamp: 0,
        };
        // SAFETY: EOS packets carry no payload per the API contract.
        let status = unsafe { (self.parse)(self.raw, &mut packet) };
        if status != CUDA_SUCCESS {
            return Err(NvdecError::Status {
                call: "cuvidParseVideoData(EOS)",
                code: status,
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for Parser<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser").field("raw", &self.raw).finish()
    }
}

impl Drop for Parser<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful create; destroyed once; the
        // callbacks box (user_data target) is still alive in self.
        unsafe {
            (self.destroy)(self.raw);
        }
    }
}

/// Layout of every FFI struct above, checked by the **compiler**.
///
/// Each number was printed by a compiled C probe against the SDK
/// header on this toolchain — facts, not estimates. They live in a
/// `const` block rather than a test so that a padding difference
/// between toolchains (MSVC and GCC lay some of these out
/// differently) stops the build on the machine doing the building,
/// instead of corrupting a driver call on frame 40 000 of a
/// twelve-hour encode.
const _: () = {
    assert!(std::mem::size_of::<ParserParams>() == 136);
    assert!(std::mem::offset_of!(ParserParams, user_data) == 40);
    assert!(std::mem::offset_of!(ParserParams, pfn_sequence) == 48);
    assert!(std::mem::offset_of!(ParserParams, pfn_display) == 64);
    assert!(std::mem::offset_of!(ParserParams, ext_video_info) == 128);
    // 32 on LP64, 24 on Windows — two `unsigned long` fields, not a
    // discrepancy. Both measured from the compiler, not derived.
    #[cfg(unix)]
    assert!(std::mem::size_of::<SourcePacket>() == 32);
    #[cfg(windows)]
    assert!(std::mem::size_of::<SourcePacket>() == 24);
    // The payload pointer sits after the two `unsigned long` fields, so
    // it moves with them: 16 on LP64, 8 on Windows.
    #[cfg(unix)]
    assert!(std::mem::offset_of!(SourcePacket, payload) == 16);
    #[cfg(windows)]
    assert!(std::mem::offset_of!(SourcePacket, payload) == 8);
    assert!(std::mem::size_of::<VideoFormat>() == 64);
    assert!(std::mem::offset_of!(VideoFormat, coded_width) == 16);
    assert!(std::mem::offset_of!(VideoFormat, chroma_format) == 40);
    assert!(std::mem::size_of::<DisplayInfo>() == 24);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_layouts_match_the_c_probe() {}

    #[test]
    fn trampolines_contain_panics_at_the_abi_boundary() {
        let mut cbs = ParserCallbacks {
            on_sequence: Box::new(|_| panic!("hostile stream drove a panic")),
            on_decode: Box::new(|_| true),
            on_display: Box::new(|_| true),
        };
        let user = (&mut cbs as *mut ParserCallbacks<'_>).cast::<c_void>();
        // SAFETY: `VideoFormat` is `#[repr(C)]` plain integers and
        // integer arrays; all-zero is a valid value.
        let mut fmt: VideoFormat = unsafe { std::mem::zeroed() };
        // SAFETY: exercising the trampoline exactly as the driver would.
        let rc = unsafe { sequence_trampoline(user, &mut fmt) };
        assert_eq!(rc, CB_FAIL, "panic must become a clean stop, not UB");
    }
}
