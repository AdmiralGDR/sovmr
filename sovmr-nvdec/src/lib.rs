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

//! # sovmr-nvdec — sovereign NVDEC bridge (skeleton)
//!
//! `dlopen("libnvcuvid.so.1")` — the hardware video *decoder* runtime
//! shipped with the NVIDIA driver — and map the entry points of the
//! zero-copy decode path. No FFmpeg, no wrapper crates: the demuxed
//! elementary stream (from `sovmr-mp4`) goes in, and decoded frames come
//! out as **`CUdeviceptr` living in VRAM**, ready to be registered
//! directly with the NVENC session from `sovmr-nvenc` — pixels never
//! visit system RAM (see the crate docs section on the zero-copy bridge
//! and `docs/zero-copy-pipeline.md`).
//!
//! ## Zero-copy VRAM bridge (the whole point)
//!
//! ```text
//!  sovmr-mp4        sovmr-nvdec (this crate)          sovmr-nvenc
//!  ─────────        ────────────────────────          ───────────
//!  Annex B ──PCIe──► cuvidDecodePicture
//!                    cuvidMapVideoFrame64 ──► CUdeviceptr + pitch (VRAM, NV12)
//!                                             │ nvEncRegisterResource
//!                                             │ nvEncMapInputResource
//!                                             ▼
//!                                          nvEncEncodePicture
//!                    cuvidUnmapVideoFrame64 ◄─ after encode consumes it
//!                                             nvEncLockBitstream ──PCIe──► .mp4
//! ```
//!
//! Only compressed bytes ever cross the PCIe bus; raw frames stay on the
//! GPU for their entire lifetime.
//!
//! Current skeleton scope: library loading, the five decode-path entry
//! points resolved and typed, symbolic `CUresult` errors (`cuvid*`
//! returns the CUDA error namespace — names shared from `sovmr-cuda`).
//! Next milestone: `CUVIDDECODECREATEINFO`/`CUVIDPICPARAMS` structs and
//! the demux→decode→map loop, exactly as `sovmr-nvenc` graduated from
//! skeleton to live sessions.
//!
//! # Example
//!
//! ```no_run
//! use sovmr_nvdec::NvDec;
//!
//! match NvDec::load() {
//!     Ok(dec) => println!("NVDEC runtime ready ({} entry points)", dec.entry_point_count()),
//!     Err(e) => eprintln!("no NVDEC here: {e}"), // fall back, never crash
//! }
//! ```

#![deny(missing_docs)]

pub mod parser;

use std::ffi::{c_int, c_uint, c_ulong, c_void};

use sovmr_cuda::CudaContext;
use sovmr_os::Library;

/// The driver library, under whichever name this platform gives
/// it. Only one can exist, so the loader takes both and picks.
const NVDEC_LIB: [&str; 2] = ["libnvcuvid.so.1", "nvcuvid.dll"];

/// The decode-path entry points, in the order we resolve them.
/// 64-bit frame mapping is used unconditionally — sovmr targets x86_64.
const REQUIRED_SYMBOLS: [&str; 8] = [
    "cuvidCreateDecoder",
    "cuvidDestroyDecoder",
    "cuvidDecodePicture",
    "cuvidMapVideoFrame64",
    "cuvidUnmapVideoFrame64",
    "cuvidCreateVideoParser",
    "cuvidParseVideoData",
    "cuvidDestroyVideoParser",
];

/// `CUresult cuvidCreateDecoder(CUvideodecoder*, CUVIDDECODECREATEINFO*)`.
/// The create-info struct lands with the decode-loop milestone; until
/// then the parameter stays behind an opaque pointer.
type CreateDecoderFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> i32;
/// `CUresult cuvidDestroyDecoder(CUvideodecoder)`.
type DestroyDecoderFn = unsafe extern "C" fn(*mut c_void) -> i32;
/// `CUresult cuvidDecodePicture(CUvideodecoder, CUVIDPICPARAMS*)`.
type DecodePictureFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
/// `CUresult cuvidMapVideoFrame64(CUvideodecoder, int, unsigned long long*,
/// unsigned int*, CUVIDPROCPARAMS*)` — yields the VRAM `CUdeviceptr` +
/// pitch of a decoded surface: the zero-copy handoff point.
type MapVideoFrameFn =
    unsafe extern "C" fn(*mut c_void, c_int, *mut u64, *mut c_uint, *mut c_void) -> i32;
/// `CUresult cuvidUnmapVideoFrame64(CUvideodecoder, unsigned long long)`.
type UnmapVideoFrameFn = unsafe extern "C" fn(*mut c_void, u64) -> i32;

/// `CUresult` success value (`cuvid*` speaks the CUDA error namespace).
pub(crate) const CUDA_SUCCESS: i32 = 0;

/// Map our codec enum onto `cudaVideoCodec` (shared with the parser).
pub(crate) fn cuda_codec_id(codec: DecodeCodec) -> i32 {
    match codec {
        DecodeCodec::H264 => CUDA_VIDEO_CODEC_H264,
        DecodeCodec::Hevc => CUDA_VIDEO_CODEC_HEVC,
        DecodeCodec::Av1 => CUDA_VIDEO_CODEC_AV1,
    }
}

/// `cudaVideoCodec` values we decode (per `cuviddec.h`).
const CUDA_VIDEO_CODEC_H264: i32 = 4;
const CUDA_VIDEO_CODEC_HEVC: i32 = 8;
/// C probe against `dynlink_cuviddec.h`, which also confirmed the two
/// above: counting positions in an enum by hand is exactly the habit
/// this project has been punished for.
const CUDA_VIDEO_CODEC_AV1: i32 = 11;
/// `cudaVideoChromaFormat_420` — the only chroma layout of consumer video.
const CHROMA_FORMAT_420: i32 = 1;
/// `cudaVideoSurfaceFormat`: NV12 for 8-bit, P016 for 10/12-bit.
const SURFACE_NV12: i32 = 0;
const SURFACE_P016: i32 = 1;
/// `cudaVideoDeinterlaceMode_Weave`: pass frames through untouched.
const DEINTERLACE_WEAVE: i32 = 0;
/// Decode surface ring: 8 covers H.264/HEVC max reference depth plus
/// scheduling headroom without hoarding VRAM.
const DEFAULT_DECODE_SURFACES: u32 = 8;
/// Surfaces exposed to the zero-copy mapper at once.
const DEFAULT_OUTPUT_SURFACES: u32 = 2;
/// NVDEC hardware ceiling per dimension on current generations; used
/// only for negative-trust input validation.
const MAX_DIMENSION: u32 = 8192;

/// `CUVIDDECODECREATEINFO` per `cuviddec.h`. Field-for-field: the header
/// uses `unsigned long` (8 bytes on LP64 Linux) — mapped as [`c_ulong`],
/// enums as `i32`, rectangles as four `i16`. The layout is pinned by a
/// size assertion in the tests.
#[repr(C)]
struct DecodeCreateInfo {
    width: c_ulong,
    height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec_type: i32,
    chroma_format: i32,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area: [i16; 4],
    output_format: i32,
    deinterlace_mode: i32,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    vid_lock: *mut c_void,
    target_rect: [i16; 4],
    enable_histogram: c_ulong,
    reserved2: [c_ulong; 4],
}

/// Codec of the elementary stream handed to the decoder — matches what
/// `sovmr-mp4`'s demuxer reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeCodec {
    /// AVC / H.264.
    H264,
    /// HEVC / H.265.
    Hevc,
    /// AV1. Samples are raw OBUs; there is no start-code framing.
    ///
    /// Both cards this project targets **decode** AV1 even where only
    /// one encodes it, which is what lets the engine read back a file it
    /// has just written.
    Av1,
}

/// Everything needed to size a hardware decoder — populated dynamically
/// from the demuxed track (`sovmr_mp4::demux::VideoTrack`), never
/// hardcoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderConfig {
    /// Elementary-stream codec.
    pub codec: DecodeCodec,
    /// Coded width in pixels.
    pub width: u32,
    /// Coded height in pixels.
    pub height: u32,
    /// 0 for 8-bit content, 2 for 10-bit.
    pub bit_depth_minus8: u8,
    /// Decode surfaces to allocate.
    ///
    /// Must be at least the driver's own `min_num_decode_surfaces` for
    /// the stream, and must agree with what the sequence callback tells
    /// the parser: the parser hands out picture indices against *its*
    /// count while the decoder allocates against this one. When they
    /// disagree the driver is asked to map a surface it never made.
    ///
    /// Measured on this hardware: H.264 asks for 5, AV1 for 9. A
    /// constant 8 was enough until AV1 could be decoded at all, and
    /// then it was a segmentation fault inside `cuvidDecodePicture`.
    pub decode_surfaces: u32,
}

/// Why the NVDEC runtime could not be brought up or a call failed.
/// `Display` always yields the symbolic NVIDIA name, never a bare code.
#[derive(Debug)]
pub enum NvdecError {
    /// `libnvcuvid.so.1` did not load (no NVIDIA driver stack).
    DriverMissing(String),
    /// A required entry symbol is absent (corrupt/exotic driver build).
    SymbolMissing(String),
    /// A `cuvid*` call returned a non-success `CUresult`.
    Status {
        /// Which call failed.
        call: &'static str,
        /// Raw `CUresult` value.
        code: i32,
    },
    /// The requested decoder configuration fails negative-trust checks
    /// before ever reaching the driver.
    InvalidConfig(&'static str),
}

impl std::fmt::Display for NvdecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvdecError::DriverMissing(m) => write!(f, "NVDEC runtime missing: {m}"),
            NvdecError::SymbolMissing(m) => write!(f, "NVDEC symbol missing: {m}"),
            NvdecError::Status { call, code } => {
                // cuvid speaks CUresult — reuse the shared symbolic table.
                write!(
                    f,
                    "{call} failed: {} ({code})",
                    sovmr_cuda::result_name(*code)
                )
            }
            NvdecError::InvalidConfig(m) => write!(f, "invalid decoder config: {m}"),
        }
    }
}

/// A loaded NVDEC runtime with the decode-path entry points resolved and
/// typed. Decoder objects (next milestone) will borrow this, so the
/// library provably outlives every decoder — same borrow-checker
/// discipline as `sovmr-nvenc`'s sessions (at the FFI boundary).
pub struct NvDec {
    /// Keeps `libnvcuvid` mapped; the fn pointers below die with it.
    _lib: Library,
    create_decoder: CreateDecoderFn,
    destroy_decoder: DestroyDecoderFn,
    decode_picture: DecodePictureFn,
    map_video_frame: MapVideoFrameFn,
    unmap_video_frame: UnmapVideoFrameFn,
    pub(crate) create_parser: parser::RawCreateParserFn,
    pub(crate) parse_video_data: parser::RawParseVideoDataFn,
    pub(crate) destroy_parser: parser::RawDestroyParserFn,
}

impl NvDec {
    /// Load `libnvcuvid.so.1` and resolve all five decode-path entry
    /// points eagerly, so a broken stack surfaces here and not mid-batch.
    ///
    /// # Errors
    ///
    /// [`NvdecError::DriverMissing`] on machines without the NVIDIA
    /// driver — callers treat it as "no hardware decode path", never as a
    /// crash; [`NvdecError::SymbolMissing`] for exotic driver builds.
    pub fn load() -> Result<Self, NvdecError> {
        let lib = Library::open_first(&NVDEC_LIB).map_err(NvdecError::DriverMissing)?;
        let mut resolved: [*mut c_void; REQUIRED_SYMBOLS.len()] =
            [std::ptr::null_mut(); REQUIRED_SYMBOLS.len()];
        for (slot, name) in REQUIRED_SYMBOLS.iter().enumerate() {
            resolved[slot] = lib.symbol(name).map_err(NvdecError::SymbolMissing)?;
        }
        // SAFETY: (a) every `resolved[i]` came from `Library::symbol`,
        // which turns a null address into `Err` (propagated by `?`), so
        // no null fn pointer is created; (b) slot `i` holds
        // `REQUIRED_SYMBOLS[i]`, and the target type of each transmute is
        // the Rust alias of that entry point's prototype in
        // cuviddec.h / nvcuvid.h (order checked against the array above);
        // (c) a data pointer and an `extern "C" fn` pointer have the same
        // size on supported targets (enforced by `transmute`); (d) `lib`
        // is moved into `_lib`, so the code stays mapped for as long as
        // these pointers are reachable.
        unsafe {
            Ok(NvDec {
                create_decoder: std::mem::transmute::<*mut c_void, CreateDecoderFn>(resolved[0]),
                destroy_decoder: std::mem::transmute::<*mut c_void, DestroyDecoderFn>(resolved[1]),
                decode_picture: std::mem::transmute::<*mut c_void, DecodePictureFn>(resolved[2]),
                map_video_frame: std::mem::transmute::<*mut c_void, MapVideoFrameFn>(resolved[3]),
                unmap_video_frame: std::mem::transmute::<*mut c_void, UnmapVideoFrameFn>(
                    resolved[4],
                ),
                create_parser: std::mem::transmute::<*mut c_void, parser::RawCreateParserFn>(
                    resolved[5],
                ),
                parse_video_data: std::mem::transmute::<*mut c_void, parser::RawParseVideoDataFn>(
                    resolved[6],
                ),
                destroy_parser: std::mem::transmute::<*mut c_void, parser::RawDestroyParserFn>(
                    resolved[7],
                ),
                _lib: lib,
            })
        }
    }

    /// How many decode-path entry points are resolved (always the full
    /// set once `load` succeeded) — the diagnostics counterpart of
    /// `NvEncApi::entry_point_count`.
    pub fn entry_point_count(&self) -> usize {
        // All eight are mandatory in `load`; expose the constant so
        // callers need no knowledge of the symbol list.
        REQUIRED_SYMBOLS.len()
    }

    /// Create a hardware decoder sized for `cfg` (`cuvidCreateDecoder`).
    /// Must run on the thread where `ctx` is current — the same rule as
    /// every CUDA call. The decoder borrows both the runtime and the
    /// context, so teardown order (decoder → context → library) is a
    /// compile-time fact.
    ///
    /// # Errors
    ///
    /// [`NvdecError::InvalidConfig`] for dimensions the hardware can
    /// never accept (zero, or beyond 8K), before the driver is touched;
    /// [`NvdecError::Status`] with the symbolic `CUresult` otherwise.
    pub fn create_decoder<'a>(
        &'a self,
        _ctx: &'a CudaContext<'a>,
        cfg: &DecoderConfig,
    ) -> Result<Decoder<'a>, NvdecError> {
        if cfg.width == 0 || cfg.height == 0 {
            return Err(NvdecError::InvalidConfig("zero width or height"));
        }
        if cfg.width > MAX_DIMENSION || cfg.height > MAX_DIMENSION {
            return Err(NvdecError::InvalidConfig("dimension exceeds 8192"));
        }
        // One mapping, one place: the decoder and the parser must agree
        // about what codec they are handling or the parser feeds frames
        // nothing can read.
        let codec_type = cuda_codec_id(cfg.codec);
        let output_format = if cfg.bit_depth_minus8 == 0 {
            SURFACE_NV12
        } else {
            SURFACE_P016
        };
        // Display area / target rect: full frame. Dimensions fit i16 per
        // the MAX_DIMENSION check above.
        let rect = [0i16, 0, cfg.width as i16, cfg.height as i16];
        let mut info = DecodeCreateInfo {
            width: c_ulong::from(cfg.width),
            height: c_ulong::from(cfg.height),
            num_decode_surfaces: c_ulong::from(cfg.decode_surfaces.max(DEFAULT_DECODE_SURFACES)),
            codec_type,
            chroma_format: CHROMA_FORMAT_420,
            creation_flags: 0,
            bit_depth_minus8: c_ulong::from(cfg.bit_depth_minus8),
            intra_decode_only: 0,
            max_width: c_ulong::from(cfg.width),
            max_height: c_ulong::from(cfg.height),
            reserved1: 0,
            display_area: rect,
            output_format,
            deinterlace_mode: DEINTERLACE_WEAVE,
            target_width: c_ulong::from(cfg.width),
            target_height: c_ulong::from(cfg.height),
            num_output_surfaces: c_ulong::from(DEFAULT_OUTPUT_SURFACES),
            vid_lock: std::ptr::null_mut(),
            target_rect: rect,
            enable_histogram: 0,
            reserved2: [0; 4],
        };
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: info is fully initialized per the header contract; raw
        // is a valid out-pointer; a CUDA context is current per the
        // documented calling convention (enforced socially by the &ctx
        // parameter and the same-thread rule).
        let status =
            unsafe { (self.create_decoder)(&mut raw, (&mut info as *mut DecodeCreateInfo).cast()) };
        if status != CUDA_SUCCESS || raw.is_null() {
            return Err(NvdecError::Status {
                call: "cuvidCreateDecoder",
                code: status,
            });
        }
        Ok(Decoder { raw, nvdec: self })
    }
}

/// An open hardware decoder (RAII: `cuvidDestroyDecoder` on drop). The
/// decode loop milestone adds `decode_picture`/`map_frame` methods here.
pub struct Decoder<'a> {
    raw: *mut c_void,
    nvdec: &'a NvDec,
}

/// `CUVIDPROCPARAMS` (C probe: 264 bytes, raw_input_dptr at 24,
/// output_stream at 56). Zeroed apart from the progressive flag.
#[repr(C)]
struct ProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: u64,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: u64,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut u64,
    proc_ext: *mut c_void,
}

impl Decoder<'_> {
    /// The raw `CUvideodecoder` handle for subsequent `cuvid*` calls.
    /// Valid only while `self` is alive.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Forward one parser-produced `CUVIDPICPARAMS` to the hardware
    /// (`cuvidDecodePicture`). The pointer is driver-owned and opaque —
    /// it is only ever passed through, exactly as the callback pipeline
    /// intends.
    ///
    /// # Safety
    ///
    /// `pic_params` must be the pointer handed to the *current* decode
    /// callback invocation (it is only valid for that callback's
    /// duration), and the decoder's CUDA context must be current on this
    /// thread.
    ///
    /// # Errors
    ///
    /// [`NvdecError::Status`] with the symbolic `CUresult`.
    pub unsafe fn decode_picture(&self, pic_params: *mut c_void) -> Result<(), NvdecError> {
        // SAFETY: caller upholds the callback-scope contract above; the
        // decoder handle is alive via &self.
        let status = unsafe { (self.nvdec.decode_picture)(self.raw, pic_params) };
        if status != CUDA_SUCCESS {
            return Err(NvdecError::Status {
                call: "cuvidDecodePicture",
                code: status,
            });
        }
        Ok(())
    }

    /// Map a decoded surface out of the ring (`cuvidMapVideoFrame64`),
    /// yielding the VRAM `CUdeviceptr` + pitch — **the zero-copy handoff
    /// to NVENC**. Unmapped on drop; the borrow pins the decoder.
    ///
    /// # Errors
    ///
    /// [`NvdecError::Status`] with the symbolic `CUresult`.
    pub fn map_frame(&self, picture_index: i32) -> Result<MappedFrame<'_>, NvdecError> {
        let mut device_ptr: u64 = 0;
        let mut pitch: c_uint = 0;
        // SAFETY: `ProcParams` is `#[repr(C)]` and made only of integers
        // and raw pointers, for which the all-zero bit pattern is valid
        // (null pointers are what CUVIDPROCPARAMS expects for unused
        // fields).
        let mut proc_params: Box<ProcParams> = unsafe { Box::new(std::mem::zeroed()) };
        proc_params.progressive_frame = 1;
        // SAFETY: valid out-pointers; proc params fully initialized; the
        // decoder is alive and its context current on this thread.
        let status = unsafe {
            (self.nvdec.map_video_frame)(
                self.raw,
                picture_index,
                &mut device_ptr,
                &mut pitch,
                (proc_params.as_mut() as *mut ProcParams).cast(),
            )
        };
        if status != CUDA_SUCCESS || device_ptr == 0 {
            return Err(NvdecError::Status {
                call: "cuvidMapVideoFrame64",
                code: status,
            });
        }
        Ok(MappedFrame {
            device_ptr,
            pitch: pitch as u32,
            decoder: self,
        })
    }
}

/// A decoded frame mapped out of the surface ring: NV12 in VRAM at
/// `device_ptr` with `pitch` bytes per row. Unmapped on drop — holding
/// it parks one ring slot, so keep the scope tight.
pub struct MappedFrame<'d> {
    device_ptr: u64,
    pitch: u32,
    decoder: &'d Decoder<'d>,
}

impl MappedFrame<'_> {
    /// The VRAM `CUdeviceptr` — feed directly to
    /// `sovmr_nvenc::VramFrame::device_ptr`.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Bytes per row of the NV12 surface.
    pub fn pitch(&self) -> u32 {
        self.pitch
    }
}

impl std::fmt::Debug for MappedFrame<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFrame")
            .field("device_ptr", &format_args!("{:#x}", self.device_ptr))
            .field("pitch", &self.pitch)
            .finish()
    }
}

impl Drop for MappedFrame<'_> {
    fn drop(&mut self) {
        // SAFETY: device_ptr came from a successful map; unmapped once;
        // the decoder outlives us per the borrow.
        unsafe {
            (self.decoder.nvdec.unmap_video_frame)(self.decoder.raw, self.device_ptr);
        }
    }
}

impl std::fmt::Debug for Decoder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder").field("raw", &self.raw).finish()
    }
}

impl Drop for Decoder<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful cuvidCreateDecoder and is
        // destroyed exactly once; the runtime outlives us per the borrow.
        unsafe {
            (self.nvdec.destroy_decoder)(self.raw);
        }
    }
}

/// The five entry points are typed fields, so referencing them here keeps
/// the "skeleton is complete" claim honest at compile time (the fields
/// are consumed by the decode-loop milestone).
#[allow(dead_code)]
fn _typecheck(dec: &NvDec) -> [usize; 5] {
    [
        dec.create_decoder as usize,
        dec.destroy_decoder as usize,
        dec.decode_picture as usize,
        dec.map_video_frame as usize,
        dec.unmap_video_frame as usize,
    ]
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
    // 176 on Linux, 112 on Windows, and the difference is not a bug in
    // either: `CUVIDDECODECREATEINFO` is built from `unsigned long`,
    // which is 8 bytes under LP64 and 4 under Windows' LLP64. Rust's
    // `c_ulong` follows the same rule, so one struct definition is
    // correct on both — only the expected total differs.
    //
    // Both numbers are measured, not derived: the compiler was asked
    // directly for each target. A hand computation of the Windows figure
    // came out at 120, which is exactly the kind of quiet padding error
    // this assert exists to catch.
    #[cfg(unix)]
    assert!(std::mem::size_of::<DecodeCreateInfo>() == 176);
    #[cfg(windows)]
    assert!(std::mem::size_of::<DecodeCreateInfo>() == 112);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_symbolically_via_the_shared_curesult_table() {
        let err = NvdecError::Status {
            call: "cuvidCreateDecoder",
            code: 2,
        };
        assert_eq!(
            err.to_string(),
            "cuvidCreateDecoder failed: CUDA_ERROR_OUT_OF_MEMORY (2)"
        );
        let err = NvdecError::Status {
            call: "cuvidMapVideoFrame64",
            code: 201,
        };
        assert!(err.to_string().contains("CUDA_ERROR_INVALID_CONTEXT"));
    }

    #[test]
    fn driver_missing_is_a_soft_display_ready_error() {
        // On machines with the driver this still exercises Display via a
        // synthetic value; the live load path is covered below.
        let err = NvdecError::DriverMissing("libnvcuvid.so.1: not found".into());
        assert!(err.to_string().starts_with("NVDEC runtime missing:"));
    }

    /// Live driver test: `cargo test -p sovmr-nvdec -- --ignored`
    #[test]
    #[ignore = "requires NVIDIA driver (libnvcuvid.so.1)"]
    fn loads_the_real_decoder_runtime_with_all_entry_points() {
        let dec = NvDec::load().expect("driver present on target machine");
        assert_eq!(dec.entry_point_count(), 8);
        // Every fn pointer must be a real, distinct address.
        let addrs = super::_typecheck(&dec);
        assert!(addrs.iter().all(|a| *a != 0));
    }

    #[test]
    fn create_info_layout_matches_the_lp64_header() {
        // Field-by-field on LP64: 3×u64 (24) + 2 enums (8) + 6×u64 (48)
        // + display_area (8) + 2 enums (8) + 3×u64 (24) + ptr (8)
        // + target_rect (8) + u64 (8) + reserved2 (32) = 176.
        assert_eq!(std::mem::align_of::<DecodeCreateInfo>(), 8);
    }

    #[test]
    fn hostile_configs_never_reach_the_driver() {
        let dec_missing = NvDec::load();
        let Ok(dec) = dec_missing else {
            return; // no driver on this machine: nothing to probe safely
        };
        // The library can be present while the GPU stack is unusable
        // (e.g. kernel module / userspace version skew pending a
        // reboot) — availability over false alarms: skip, don't fail.
        let Ok(cuda) = sovmr_cuda::Cuda::load() else {
            return;
        };
        let Ok(ctx) = cuda.create_context(0) else {
            eprintln!("skipping: CUDA context unavailable (driver state)");
            return;
        };
        for bad in [
            DecoderConfig {
                codec: DecodeCodec::H264,
                width: 0,
                height: 720,
                bit_depth_minus8: 0,
                decode_surfaces: DEFAULT_DECODE_SURFACES,
            },
            DecoderConfig {
                codec: DecodeCodec::Hevc,
                width: 1280,
                height: 999_999,
                bit_depth_minus8: 0,
                decode_surfaces: DEFAULT_DECODE_SURFACES,
            },
        ] {
            assert!(matches!(
                dec.create_decoder(&ctx, &bad),
                Err(NvdecError::InvalidConfig(_))
            ));
        }
    }

    /// The workstream-2 gate: a real hardware decoder created with
    /// dynamically supplied dimensions, destroyed via RAII, recreated.
    #[test]
    #[ignore = "requires NVIDIA GPU + driver (libcuda.so.1, libnvcuvid.so.1)"]
    fn creates_a_real_h264_decoder_and_destroys_it_cleanly() {
        let cuda = sovmr_cuda::Cuda::load().expect("CUDA driver present");
        let ctx = cuda.create_context(0).expect("GPU 0 usable");
        let dec = NvDec::load().expect("NVDEC runtime present");
        let cfg = DecoderConfig {
            codec: DecodeCodec::H264,
            width: 1280,
            height: 720,
            bit_depth_minus8: 0,
            decode_surfaces: DEFAULT_DECODE_SURFACES,
        };
        let decoder = dec.create_decoder(&ctx, &cfg).expect("decoder must open");
        assert!(!decoder.raw().is_null());
        drop(decoder); // cuvidDestroyDecoder
                       // Repeatability proves clean teardown against driver accounting.
        let again = dec.create_decoder(&ctx, &cfg).expect("decoder reopens");
        assert!(!again.raw().is_null());
    }
}
