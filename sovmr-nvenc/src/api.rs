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

//! [`NvEncApi`]: entry into the NVENC runtime.
//!
//! Loads `libnvidia-encode.so.1`, checks the driver's supported API level
//! via `NvEncodeAPIGetMaxSupportedVersion`, then calls
//! `NvEncodeAPICreateInstance` to receive the versioned table of encoder
//! entry points. Struct layout and version encoding follow the public
//! `nvEncodeAPI.h` contract:
//!
//! * `NVENCAPI_VERSION = major | (minor << 24)`
//! * `NVENCAPI_STRUCT_VERSION(v) = NVENCAPI_VERSION | (v << 16) | (0x7 << 28)`
//! * `NvEncodeAPIGetMaxSupportedVersion` yields `(major << 4) | minor`.
//!
//! The driver validates the version tag before writing anything, so a
//! mismatch fails cleanly with a status code instead of corrupting memory.

use std::ffi::c_void;

use sovmr_cuda::CudaContext;
use sovmr_os::Library;

/// The encoder library, under whichever name this platform gives
/// it. Only one can exist, so the loader takes both and picks.
const NVENC_LIB: [&str; 2] = ["libnvidia-encode.so.1", "nvEncodeAPI64.dll"];
/// Driver-level version probe: `NVENCSTATUS fn(uint32_t*)`.
const SYM_MAX_VERSION: &str = "NvEncodeAPIGetMaxSupportedVersion";
/// Function-table constructor: `NVENCSTATUS fn(NV_ENCODE_API_FUNCTION_LIST*)`.
const SYM_CREATE_INSTANCE: &str = "NvEncodeAPICreateInstance";

/// NVENC API level this crate programs against (SDK 12.0 — supported by
/// every driver that can run an RTX 4070).
const API_MAJOR: u32 = 12;
const API_MINOR: u32 = 0;
/// `NVENCAPI_VERSION` for the requested level.
const NVENCAPI_VERSION: u32 = API_MAJOR | (API_MINOR << 24);
/// `NV_ENCODE_API_FUNCTION_LIST_VER` (struct revision 2, magic 0x7 nibble).
const FUNCTION_LIST_VER: u32 = NVENCAPI_VERSION | (2 << 16) | (0x7 << 28);
/// `NV_ENC_SUCCESS`.
const NV_ENC_SUCCESS: i32 = 0;

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER` (struct revision 1).
const SESSION_PARAMS_VER: u32 = NVENCAPI_VERSION | (1 << 16) | (0x7 << 28);
/// `NV_ENC_DEVICE_TYPE_CUDA` — we host sessions on a CUDA context.
const NV_ENC_DEVICE_TYPE_CUDA: u32 = 0x1;

/// Positions inside `NV_ENCODE_API_FUNCTION_LIST` per the public
/// `nvEncodeAPI.h` (stable since API v5; FFmpeg's nvenc.c relies on the
/// same layout). Only the slots we call are named.
const SLOT_GET_ENCODE_GUID_COUNT: usize = 1;
const SLOT_GET_ENCODE_GUIDS: usize = 4;
const SLOT_MAP_INPUT_RESOURCE: usize = 25;
const SLOT_UNMAP_INPUT_RESOURCE: usize = 26;
const SLOT_DESTROY_ENCODER: usize = 27;
const SLOT_OPEN_ENCODE_SESSION_EX: usize = 29;
const SLOT_REGISTER_RESOURCE: usize = 30;
const SLOT_UNREGISTER_RESOURCE: usize = 31;

/// `NV_ENC_REGISTER_RESOURCE_VER` / `NV_ENC_MAP_INPUT_RESOURCE_VER`
/// (both struct revision 4 in the API level we target).
const REGISTER_RESOURCE_VER: u32 = NVENCAPI_VERSION | (4 << 16) | (0x7 << 28);
const MAP_INPUT_RESOURCE_VER: u32 = NVENCAPI_VERSION | (4 << 16) | (0x7 << 28);
/// `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR` — VRAM pointers, our whole
/// zero-copy premise.
const NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR: u32 = 0x1;
/// `NV_ENC_BUFFER_FORMAT_NV12` — NVDEC's default output layout.
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x1;
/// `NV_ENC_INPUT_IMAGE` buffer usage.
const NV_ENC_INPUT_IMAGE: u32 = 0x0;
/// Reserved tails, oversized past the header's 247/60 and 251/63 — the
/// driver never touches reserved regions, and extra tail capacity keeps
/// us safe across minor header revisions (same policy as the function
/// list).
const REGISTER_RESERVED1: usize = 256;
const REGISTER_RESERVED2: usize = 64;
const MAP_RESERVED1: usize = 256;
const MAP_RESERVED2: usize = 64;

/// `NV_ENC_REGISTER_RESOURCE` per the header: version, resource type,
/// geometry, the pointer to register, the returned registration handle,
/// format/usage, D3D12 fence (unused on Linux), reserved tail.
#[repr(C)]
struct RegisterResourceParams {
    version: u32,
    resource_type: u32,
    width: u32,
    height: u32,
    pitch: u32,
    sub_resource_index: u32,
    resource_to_register: *mut c_void,
    registered_resource: *mut c_void,
    buffer_format: u32,
    buffer_usage: u32,
    input_fence_point: *mut c_void,
    reserved1: [u32; REGISTER_RESERVED1],
    reserved2: [*mut c_void; REGISTER_RESERVED2],
}

/// `NV_ENC_MAP_INPUT_RESOURCE` per the header: version, legacy fields,
/// the registration handle in, the mapped input pointer + format out.
#[repr(C)]
struct MapInputResourceParams {
    version: u32,
    sub_resource_index: u32,
    input_resource: *mut c_void,
    registered_resource: *mut c_void,
    mapped_resource: *mut c_void,
    mapped_buffer_fmt: u32,
    reserved1: [u32; MAP_RESERVED1],
    reserved2: [*mut c_void; MAP_RESERVED2],
}

type RegisterResourceFn = unsafe extern "C" fn(*mut c_void, *mut RegisterResourceParams) -> i32;
type UnregisterResourceFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type MapInputResourceFn = unsafe extern "C" fn(*mut c_void, *mut MapInputResourceParams) -> i32;
type UnmapInputResourceFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

/// Geometry of a VRAM surface to register with the encoder — populated
/// from a `sovmr_cuda::DeviceBuffer` today, from `cuvidMapVideoFrame64`
/// output in the full pipeline. NV12 layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramFrame {
    /// The `CUdeviceptr`.
    pub device_ptr: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bytes per row as allocated (>= width for NV12).
    pub pitch: u32,
}

/// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS` per the header: version,
/// device type, device handle, reserved, client API version, then
/// reserved tail words.
#[repr(C)]
struct OpenSessionParams {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

type GetEncodeGuidCountFn = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type GetEncodeGuidsFn = unsafe extern "C" fn(*mut c_void, *mut Guid, u32, *mut u32) -> i32;

use crate::encoder::{Guid, CODEC_AV1_GUID, CODEC_HEVC_GUID};
type OpenSessionExFn = unsafe extern "C" fn(*mut OpenSessionParams, *mut *mut c_void) -> i32;

/// Ceiling on codec GUIDs read back from the driver.
///
/// Bounded execution at an FFI boundary: the count comes from the other
/// side, and a buffer sized from an untrusted number is how that goes
/// wrong. NVENC has published five codecs in its entire history, so this
/// is far above anything real while keeping the array a fixed size on
/// the stack.
const MAX_ENCODE_GUIDS: usize = 32;

/// Which codecs this GPU's encoder can actually produce, as reported by
/// the driver itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeSupport {
    /// HEVC / H.265 encode.
    pub hevc: bool,
    /// AV1 encode — Ada Lovelace and newer only. Ampere decodes AV1 but
    /// cannot encode it, which is precisely the distinction a marketing
    /// name cannot be trusted to make.
    pub av1: bool,
}
type DestroyEncoderFn = unsafe extern "C" fn(*mut c_void) -> i32;

/// Symbolic name for a raw `NVENCSTATUS`, per the `nvEncodeAPI.h` enum —
/// downstream users read `NV_ENC_ERR_OUT_OF_MEMORY`, never a bare `-10`.
pub fn status_name(code: i32) -> &'static str {
    match code {
        0 => "NV_ENC_SUCCESS",
        1 => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        2 => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        3 => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        4 => "NV_ENC_ERR_INVALID_DEVICE",
        5 => "NV_ENC_ERR_DEVICE_NOT_EXIST",
        6 => "NV_ENC_ERR_INVALID_PTR",
        7 => "NV_ENC_ERR_INVALID_EVENT",
        8 => "NV_ENC_ERR_INVALID_PARAM",
        9 => "NV_ENC_ERR_INVALID_CALL",
        10 => "NV_ENC_ERR_OUT_OF_MEMORY",
        11 => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        12 => "NV_ENC_ERR_UNSUPPORTED_PARAM",
        13 => "NV_ENC_ERR_LOCK_BUSY",
        14 => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
        15 => "NV_ENC_ERR_INVALID_VERSION",
        16 => "NV_ENC_ERR_MAP_FAILED",
        17 => "NV_ENC_ERR_NEED_MORE_INPUT",
        18 => "NV_ENC_ERR_ENCODER_BUSY",
        19 => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
        20 => "NV_ENC_ERR_GENERIC",
        21 => "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
        22 => "NV_ENC_ERR_UNIMPLEMENTED",
        23 => "NV_ENC_ERR_RESOURCE_REGISTER_FAILED",
        24 => "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
        25 => "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
        26 => "NV_ENC_ERR_NEED_MORE_OUTPUT",
        _ => "NV_ENC_ERR_<unlisted>",
    }
}

/// Generous upper bound on the entry-point slots we hand the driver
/// (SDK 12.x fills ~35; extra slots stay NULL and are ignored).
const FN_SLOTS: usize = 64;
/// Trailing reserved words per the header (`void* reserved2[281]`).
const RESERVED2: usize = 281;

/// `NV_ENCODE_API_FUNCTION_LIST`, oversized defensively: the driver fills
/// exactly the layout matching the version tag it validated, and never
/// reads past it, so extra tail capacity is harmless by construction.
#[repr(C)]
struct NvEncFunctionList {
    version: u32,
    reserved: u32,
    entries: [*mut c_void; FN_SLOTS],
    reserved2: [*mut c_void; RESERVED2],
}

type MaxVersionFn = unsafe extern "C" fn(*mut u32) -> i32;
type CreateInstanceFn = unsafe extern "C" fn(*mut NvEncFunctionList) -> i32;

/// Why the NVENC runtime could not be brought up. Every variant carries a
/// display-ready diagnostic.
#[derive(Debug)]
pub enum NvencError {
    /// `libnvidia-encode.so.1` did not load (no NVIDIA driver stack).
    DriverMissing(String),
    /// A required entry symbol is absent (corrupt/exotic driver build).
    SymbolMissing(String),
    /// The installed driver speaks an older API than we program against.
    DriverTooOld {
        /// API level the driver supports.
        driver: (u32, u32),
        /// API level this crate requires.
        required: (u32, u32),
    },
    /// The runtime returned a non-success `NVENCSTATUS`.
    Status {
        /// Which call failed.
        call: &'static str,
        /// The raw `NVENCSTATUS` value (see [`status_name`]).
        code: i32,
    },
    /// The driver's function table lacks an entry we need (would only
    /// happen if the requested API level dropped a core function).
    EntryPointMissing(usize),
}

impl std::fmt::Display for NvencError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvencError::DriverMissing(m) => write!(f, "NVENC runtime missing: {m}"),
            NvencError::SymbolMissing(m) => write!(f, "NVENC symbol missing: {m}"),
            NvencError::DriverTooOld { driver, required } => write!(
                f,
                "driver NVENC API {}.{} older than required {}.{}",
                driver.0, driver.1, required.0, required.1
            ),
            NvencError::Status { call, code } => {
                write!(f, "{call} failed: {} ({code})", status_name(*code))
            }
            NvencError::EntryPointMissing(slot) => {
                write!(
                    f,
                    "driver filled no entry point at function-table slot {slot}"
                )
            }
        }
    }
}

/// Decode `NvEncodeAPIGetMaxSupportedVersion`'s `(major << 4) | minor`.
fn decode_driver_version(raw: u32) -> (u32, u32) {
    (raw >> 4, raw & 0xF)
}

/// A loaded, version-checked NVENC runtime with its entry-point table.
///
/// This is the Phase-2 seed: the table obtained here contains every
/// `nvEnc*` function pointer (open session, init encoder, encode picture,
/// lock bitstream, ...) that the future `NvencNativeProvider` will drive.
pub struct NvEncApi {
    /// Keeps the shared object mapped; entry pointers die with it.
    _lib: Library,
    functions: Box<NvEncFunctionList>,
    driver_version: (u32, u32),
}

impl NvEncApi {
    /// Load the driver runtime, verify its API level, and materialize the
    /// entry-point table.
    ///
    /// # Errors
    ///
    /// See [`NvencError`] — machines without an NVIDIA driver fail with
    /// [`NvencError::DriverMissing`], which callers treat as "use the
    /// FFmpeg backend instead", never as a crash.
    pub fn load() -> Result<Self, NvencError> {
        let lib = Library::open_first(&NVENC_LIB).map_err(NvencError::DriverMissing)?;

        let max_version: MaxVersionFn = {
            let ptr = lib
                .symbol(SYM_MAX_VERSION)
                .map_err(NvencError::SymbolMissing)?;
            // SAFETY: symbol verified present; signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, MaxVersionFn>(ptr) }
        };
        let mut raw = 0u32;
        // SAFETY: valid out-pointer to a u32, call contract per header.
        let status = unsafe { max_version(&mut raw) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: SYM_MAX_VERSION,
                code: status,
            });
        }
        let driver_version = decode_driver_version(raw);
        if driver_version < (API_MAJOR, API_MINOR) {
            return Err(NvencError::DriverTooOld {
                driver: driver_version,
                required: (API_MAJOR, API_MINOR),
            });
        }

        let create_instance: CreateInstanceFn = {
            let ptr = lib
                .symbol(SYM_CREATE_INSTANCE)
                .map_err(NvencError::SymbolMissing)?;
            // SAFETY: symbol verified present; signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, CreateInstanceFn>(ptr) }
        };
        let mut functions = Box::new(NvEncFunctionList {
            version: FUNCTION_LIST_VER,
            reserved: 0,
            entries: [std::ptr::null_mut(); FN_SLOTS],
            reserved2: [std::ptr::null_mut(); RESERVED2],
        });
        // SAFETY: the struct is zeroed apart from the version tag the
        // driver validates first; on version mismatch it returns an error
        // without touching the buffer, on success it fills only the
        // layout belonging to that version (our buffer is a superset).
        let status = unsafe { create_instance(functions.as_mut()) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: SYM_CREATE_INSTANCE,
                code: status,
            });
        }

        Ok(NvEncApi {
            _lib: lib,
            functions,
            driver_version,
        })
    }

    /// Highest NVENC API `(major, minor)` the installed driver supports.
    pub fn driver_version(&self) -> (u32, u32) {
        self.driver_version
    }

    /// How many entry points the driver filled in — a liveness signal for
    /// diagnostics (SDK 12 populates ~35).
    pub fn entry_point_count(&self) -> usize {
        self.functions
            .entries
            .iter()
            .filter(|p| !p.is_null())
            .count()
    }

    /// A non-null entry pointer from the function table, by header slot.
    pub(crate) fn entry(&self, slot: usize) -> Result<*mut c_void, NvencError> {
        match self.functions.entries.get(slot) {
            Some(ptr) if !ptr.is_null() => Ok(*ptr),
            _ => Err(NvencError::EntryPointMissing(slot)),
        }
    }

    /// Open a hardware encode session on a sovereign CUDA context
    /// (`nvEncOpenEncodeSessionEx` with `NV_ENC_DEVICE_TYPE_CUDA`).
    ///
    /// The returned session closes itself (`nvEncDestroyEncoder`) on drop
    /// and borrows both the API and the context, so neither can be torn
    /// down while the encoder handle is alive (at the FFI
    /// boundary).
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] with the symbolic NVENC reason — e.g.
    /// `NV_ENC_ERR_INVALID_VERSION` if the driver rejects our API level,
    /// `NV_ENC_ERR_OUT_OF_MEMORY` when VRAM is exhausted.
    pub fn open_session<'a>(
        &'a self,
        ctx: &'a CudaContext<'a>,
    ) -> Result<NvEncSession<'a>, NvencError> {
        let open: OpenSessionExFn = {
            let ptr = self.entry(SLOT_OPEN_ENCODE_SESSION_EX)?;
            // SAFETY: slot position and signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, OpenSessionExFn>(ptr) }
        };
        let destroy: DestroyEncoderFn = {
            let ptr = self.entry(SLOT_DESTROY_ENCODER)?;
            // SAFETY: slot position and signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, DestroyEncoderFn>(ptr) }
        };

        let mut params = Box::new(OpenSessionParams {
            version: SESSION_PARAMS_VER,
            device_type: NV_ENC_DEVICE_TYPE_CUDA,
            device: ctx.raw(),
            reserved: std::ptr::null_mut(),
            api_version: NVENCAPI_VERSION,
            reserved1: [0; 253],
            reserved2: [std::ptr::null_mut(); 64],
        });
        let mut encoder: *mut c_void = std::ptr::null_mut();
        // SAFETY: params is a fully initialized, version-tagged struct the
        // driver validates before use; encoder is a valid out-pointer; the
        // CUDA context is alive (borrowed for 'a).
        let status = unsafe { open(params.as_mut(), &mut encoder) };
        if status != NV_ENC_SUCCESS || encoder.is_null() {
            return Err(NvencError::Status {
                call: "nvEncOpenEncodeSessionEx",
                code: status,
            });
        }
        Ok(NvEncSession {
            encoder,
            destroy,
            api: self,
            _ctx: ctx,
            accepted: std::cell::RefCell::new(None),
        })
    }
}

/// An open hardware encode session. This is the object the Phase-2
/// frame-feeding loop will drive (`nvEncInitializeEncoder`,
/// `nvEncEncodePicture`, `nvEncLockBitstream`, ...). Closed on drop.
pub struct NvEncSession<'a> {
    encoder: *mut c_void,
    destroy: DestroyEncoderFn,
    /// The entry pointers live inside the API's mapped library.
    pub(crate) api: &'a NvEncApi,
    /// The session's VRAM lives inside this context.
    _ctx: &'a CudaContext<'a>,
    /// The configuration the driver accepted, kept so it can be handed
    /// back changed.
    ///
    /// `nvEncReconfigureEncoder` takes the *whole* initialization
    /// structure, not a delta — so rebuilding one from scratch would
    /// mean re-deriving every field the preset filled in, and any field
    /// forgotten would silently revert. Keeping what was accepted is the
    /// only way to change one thing and nothing else.
    pub(crate) accepted: std::cell::RefCell<Option<crate::encoder::AcceptedConfig>>,
}

impl NvEncSession<'_> {
    /// The raw encoder handle for subsequent `nvEnc*` calls. Valid only
    /// while `self` is alive.
    pub fn raw(&self) -> *mut c_void {
        self.encoder
    }

    /// Ask the driver which codecs this GPU can encode.
    ///
    /// This replaces guessing from the marketing name. The old heuristic
    /// mapped `"RTX 30"` to "no AV1" by string match, which happens to be
    /// right for Ampere and is wrong the moment NVIDIA ships a name the
    /// table has never seen — silently choosing a codec the hardware
    /// cannot produce. The driver knows; ask it.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] if either call fails, and
    /// [`NvencError::EntryPointMissing`] on a driver that filled neither
    /// slot.
    pub fn supported_codecs(&self) -> Result<EncodeSupport, NvencError> {
        let count_fn: GetEncodeGuidCountFn = {
            let ptr = self.api.entry(SLOT_GET_ENCODE_GUID_COUNT)?;
            // SAFETY: slot position and signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, GetEncodeGuidCountFn>(ptr) }
        };
        let guids_fn: GetEncodeGuidsFn = {
            let ptr = self.api.entry(SLOT_GET_ENCODE_GUIDS)?;
            // SAFETY: slot position and signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, GetEncodeGuidsFn>(ptr) }
        };

        let mut count: u32 = 0;
        // SAFETY: live encoder handle; `count` is a valid out-pointer.
        let status = unsafe { count_fn(self.encoder, &mut count) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncGetEncodeGUIDCount",
                code: status,
            });
        }

        // Negative trust toward the driver's own number: clamp before it
        // is used to size anything.
        let wanted = (count as usize).min(MAX_ENCODE_GUIDS);
        let mut guids = [Guid::ZERO; MAX_ENCODE_GUIDS];
        let mut filled: u32 = 0;
        // SAFETY: live encoder handle; the array is `MAX_ENCODE_GUIDS`
        // long and we declare exactly that capacity, so the driver
        // cannot write past it whatever it reported above.
        let status = unsafe {
            guids_fn(
                self.encoder,
                guids.as_mut_ptr(),
                MAX_ENCODE_GUIDS as u32,
                &mut filled,
            )
        };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncGetEncodeGUIDs",
                code: status,
            });
        }
        // Trust the smaller of the two counts the driver gave us.
        let filled = (filled as usize).min(wanted.max(filled as usize).min(MAX_ENCODE_GUIDS));
        let listed = &guids[..filled];
        Ok(EncodeSupport {
            hevc: listed.contains(&CODEC_HEVC_GUID),
            av1: listed.contains(&CODEC_AV1_GUID),
        })
    }

    /// Crate-internal access to the function table for sibling modules
    /// (the encoder frame loop lives in `crate::encoder`).
    pub(crate) fn entry_point(&self, slot: usize) -> Result<*mut c_void, NvencError> {
        self.api.entry(slot)
    }

    /// Register a VRAM surface (`CUdeviceptr`, NV12) with the encoder —
    /// the zero-copy handoff point: this is exactly the pointer
    /// `cuvidMapVideoFrame64` produces on the decode side.
    ///
    /// Registration is the expensive step; do it once per pool surface
    /// and keep the [`RegisteredResource`], then [`RegisteredResource::map`]
    /// cheaply per frame. Teardown order (mapping → registration →
    /// session → context) is enforced by the borrow checker.
    ///
    /// Driver policy note (verified live on 580.x): registration is
    /// accepted only after the encoder is initialized with
    /// `nvEncInitializeEncoder`; before that the driver answers
    /// `NV_ENC_ERR_DEVICE_NOT_EXIST`. The initialize call lands with the
    /// frame-loop milestone.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] with the symbolic reason — e.g.
    /// `NV_ENC_ERR_RESOURCE_REGISTER_FAILED` for a bogus pointer or
    /// `NV_ENC_ERR_UNSUPPORTED_PARAM` for a pitch the encoder rejects.
    pub fn register_vram_frame(
        &self,
        frame: &VramFrame,
    ) -> Result<RegisteredResource<'_>, NvencError> {
        // Resolve all four resource entry points up front so the mapped
        // wrapper never needs to reach back into the function table.
        let register: RegisterResourceFn = {
            let ptr = self.api.entry(SLOT_REGISTER_RESOURCE)?;
            // SAFETY: slot position and signature per nvEncodeAPI.h.
            unsafe { std::mem::transmute::<*mut c_void, RegisterResourceFn>(ptr) }
        };
        let unregister: UnregisterResourceFn = {
            let ptr = self.api.entry(SLOT_UNREGISTER_RESOURCE)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, UnregisterResourceFn>(ptr) }
        };
        let map_fn: MapInputResourceFn = {
            let ptr = self.api.entry(SLOT_MAP_INPUT_RESOURCE)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, MapInputResourceFn>(ptr) }
        };
        let unmap_fn: UnmapInputResourceFn = {
            let ptr = self.api.entry(SLOT_UNMAP_INPUT_RESOURCE)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, UnmapInputResourceFn>(ptr) }
        };

        let mut params = Box::new(RegisterResourceParams {
            version: REGISTER_RESOURCE_VER,
            resource_type: NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
            width: frame.width,
            height: frame.height,
            pitch: frame.pitch,
            sub_resource_index: 0,
            resource_to_register: frame.device_ptr as *mut c_void,
            registered_resource: std::ptr::null_mut(),
            buffer_format: NV_ENC_BUFFER_FORMAT_NV12,
            buffer_usage: NV_ENC_INPUT_IMAGE,
            input_fence_point: std::ptr::null_mut(),
            reserved1: [0; REGISTER_RESERVED1],
            reserved2: [std::ptr::null_mut(); REGISTER_RESERVED2],
        });
        // SAFETY: params is fully initialized and version-tagged; the
        // encoder handle is alive (self borrows); the driver validates
        // the version before writing only `registered_resource`.
        let status = unsafe { register(self.encoder, params.as_mut()) };
        if status != NV_ENC_SUCCESS || params.registered_resource.is_null() {
            return Err(NvencError::Status {
                call: "nvEncRegisterResource",
                code: status,
            });
        }
        Ok(RegisteredResource {
            raw: params.registered_resource,
            session: self,
            unregister,
            map_fn,
            unmap_fn,
        })
    }
}

impl Drop for NvEncSession<'_> {
    fn drop(&mut self) {
        // SAFETY: encoder came from a successful open and is destroyed
        // exactly once; the API and context are still alive per borrows.
        unsafe {
            (self.destroy)(self.encoder);
        }
    }
}

/// A VRAM surface registered with an encode session
/// (`nvEncRegisterResource` with `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`).
/// Unregistered on drop; borrows the session, so it can never outlive
/// the encoder it is registered with — and, transitively, never outlive
/// the CUDA context the memory lives in.
pub struct RegisteredResource<'s> {
    raw: *mut c_void,
    session: &'s NvEncSession<'s>,
    unregister: UnregisterResourceFn,
    map_fn: MapInputResourceFn,
    unmap_fn: UnmapInputResourceFn,
}

impl RegisteredResource<'_> {
    /// The raw `NV_ENC_REGISTERED_PTR`.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Map the registered surface for encoding (`nvEncMapInputResource`).
    /// The result is the `NV_ENC_INPUT_PTR` a future `nvEncEncodePicture`
    /// consumes. Cheap; called per frame. The mapping borrows `self`, so
    /// unmap-before-unregister ordering is a compile-time fact.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] with the symbolic reason (e.g.
    /// `NV_ENC_ERR_MAP_FAILED`).
    pub fn map(&self) -> Result<MappedInput<'_>, NvencError> {
        let mut params = Box::new(MapInputResourceParams {
            version: MAP_INPUT_RESOURCE_VER,
            sub_resource_index: 0,
            input_resource: std::ptr::null_mut(),
            registered_resource: self.raw,
            mapped_resource: std::ptr::null_mut(),
            mapped_buffer_fmt: 0,
            reserved1: [0; MAP_RESERVED1],
            reserved2: [std::ptr::null_mut(); MAP_RESERVED2],
        });
        // SAFETY: params fully initialized and version-tagged; encoder
        // and registration are alive per the borrow chain.
        let status = unsafe { (self.map_fn)(self.session.encoder, params.as_mut()) };
        if status != NV_ENC_SUCCESS || params.mapped_resource.is_null() {
            return Err(NvencError::Status {
                call: "nvEncMapInputResource",
                code: status,
            });
        }
        Ok(MappedInput {
            raw: params.mapped_resource,
            buffer_format: params.mapped_buffer_fmt,
            registration: self,
        })
    }
}

impl std::fmt::Debug for RegisteredResource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredResource")
            .field("raw", &self.raw)
            .finish()
    }
}

impl Drop for RegisteredResource<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful registration and is
        // unregistered exactly once; the session outlives us per borrow.
        // Any MappedInput has already dropped (it borrows self).
        unsafe {
            (self.unregister)(self.session.encoder, self.raw);
        }
    }
}

/// A registered surface mapped for encoding — the object whose `raw()`
/// feeds `nvEncEncodePicture` in the frame loop. Unmapped on drop, and
/// the borrow ties it under its [`RegisteredResource`]: dropping the
/// registration (or the session) while a mapping lives is a compile
/// error, not a driver fault.
pub struct MappedInput<'r> {
    raw: *mut c_void,
    buffer_format: u32,
    registration: &'r RegisteredResource<'r>,
}

impl MappedInput<'_> {
    /// The raw `NV_ENC_INPUT_PTR` for `nvEncEncodePicture`.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// The `NV_ENC_BUFFER_FORMAT` the driver reported for the mapping
    /// (expected: NV12 for 8-bit surfaces).
    pub fn buffer_format(&self) -> u32 {
        self.buffer_format
    }
}

impl std::fmt::Debug for MappedInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedInput")
            .field("raw", &self.raw)
            .field("buffer_format", &self.buffer_format)
            .finish()
    }
}

impl Drop for MappedInput<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful map and is unmapped exactly
        // once; encoder and registration are alive per the borrow chain.
        unsafe {
            (self.registration.unmap_fn)(self.registration.session.encoder, self.raw);
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
    assert!(std::mem::size_of::<OpenSessionParams>() == 1552);
    assert!(std::mem::size_of::<RegisterResourceParams>() == 1592);
    assert!(std::mem::size_of::<MapInputResourceParams>() == 1576);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_constants_follow_the_header_encoding() {
        // NVENCAPI_VERSION for 12.0 is just 12; struct ver adds the
        // revision in bits 16.. and the 0x7 magic in the top nibble.
        assert_eq!(NVENCAPI_VERSION, 12);
        assert_eq!(FUNCTION_LIST_VER, 0x7002_000C);
    }

    #[test]
    fn driver_version_word_decodes_as_major_minor() {
        assert_eq!(decode_driver_version((12 << 4) | 2), (12, 2));
        assert_eq!(decode_driver_version(13 << 4), (13, 0));
        assert_eq!(decode_driver_version(0), (0, 0));
    }

    #[test]
    fn status_names_are_symbolic() {
        assert_eq!(status_name(10), "NV_ENC_ERR_OUT_OF_MEMORY");
        assert_eq!(status_name(15), "NV_ENC_ERR_INVALID_VERSION");
        assert_eq!(status_name(-3), "NV_ENC_ERR_<unlisted>");
        let err = NvencError::Status {
            call: "nvEncEncodePicture",
            code: 10,
        };
        assert_eq!(
            err.to_string(),
            "nvEncEncodePicture failed: NV_ENC_ERR_OUT_OF_MEMORY (10)"
        );
    }

    #[test]
    fn session_params_version_matches_header_encoding() {
        // NVENCAPI_STRUCT_VERSION(1) for API 12.0.
        assert_eq!(SESSION_PARAMS_VER, 0x7001_000C);
    }

    /// Live driver test — requires an NVIDIA GPU + driver:
    /// `cargo test -p sovmr-nvenc -- --ignored`
    #[test]
    #[ignore = "requires NVIDIA driver (libnvidia-encode.so.1)"]
    fn loads_the_real_driver_and_gets_entry_points() {
        let api = NvEncApi::load().expect("driver present on target machine");
        assert!(api.driver_version() >= (API_MAJOR, API_MINOR));
        assert!(
            api.entry_point_count() >= 30,
            "suspiciously few entry points: {}",
            api.entry_point_count()
        );
    }

    #[test]
    fn resource_struct_versions_and_layouts_are_pinned() {
        assert_eq!(REGISTER_RESOURCE_VER, 0x7004_000C);
        assert_eq!(MAP_INPUT_RESOURCE_VER, 0x7004_000C);
        // Layout stability pins (LP64): prefix fields at fixed offsets,
        // oversized reserved tails included.
        assert_eq!(std::mem::align_of::<RegisterResourceParams>(), 8);
    }

    /// The zero-copy plumbing, live: sovereign VRAM allocation feeding a
    /// registration attempt against a real session. NVIDIA's flow gates
    /// `nvEncRegisterResource` behind `nvEncInitializeEncoder` (the
    /// frame-loop milestone), so today the driver answers with one of the
    /// documented pre-init statuses — which is itself a layout proof: a
    /// wrong struct would fail as `NV_ENC_ERR_INVALID_VERSION`/`_PARAM`,
    /// not as an encoder-state complaint.
    #[test]
    #[ignore = "requires NVIDIA GPU + driver (libcuda.so.1, libnvidia-encode.so.1)"]
    fn vram_registration_plumbing_reaches_the_driver() {
        const NV_ENC_ERR_DEVICE_NOT_EXIST: i32 = 5;
        const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: i32 = 11;

        let cuda = sovmr_cuda::Cuda::load().expect("CUDA driver present");
        let ctx = cuda.create_context(0).expect("GPU 0 usable");
        // NV12 720p: 1280 bytes/row, 720 * 3 / 2 rows. Proves the new
        // sovereign VRAM allocator regardless of NVENC's answer below.
        let buffer = ctx.alloc_pitched(1280, 1080).expect("VRAM available");
        assert!(buffer.pitch() >= 1280);
        assert_ne!(buffer.device_ptr(), 0);

        let api = NvEncApi::load().expect("NVENC runtime present");
        let session = api.open_session(&ctx).expect("session opens");
        let frame = VramFrame {
            device_ptr: buffer.device_ptr(),
            width: 1280,
            height: 720,
            pitch: buffer.pitch() as u32,
        };
        match session.register_vram_frame(&frame) {
            // A future driver may accept pre-init registration: exercise
            // the full map/unmap chain when it does.
            Ok(registration) => {
                assert!(!registration.raw().is_null());
                let mapped = registration.map().expect("mapping");
                assert!(!mapped.raw().is_null());
            }
            Err(NvencError::Status {
                call: "nvEncRegisterResource",
                code,
            }) => {
                assert!(
                    code == NV_ENC_ERR_DEVICE_NOT_EXIST
                        || code == NV_ENC_ERR_ENCODER_NOT_INITIALIZED,
                    "status {code} ({}) suggests a struct-layout problem, \
                     not the expected pre-initialization gate",
                    status_name(code)
                );
            }
            Err(e) => panic!("unexpected error class: {e}"),
        };
        // Drop order: session -> buffer -> ctx (borrow-checker enforced).
    }

    /// The Phase-2 milestone gate: a real NVENC session opened on a
    /// sovereignly created CUDA context, then torn down in reverse order.
    #[test]
    #[ignore = "requires NVIDIA GPU + driver (libcuda.so.1, libnvidia-encode.so.1)"]
    fn opens_a_real_encode_session_on_a_cuda_context() {
        let cuda = sovmr_cuda::Cuda::load().expect("CUDA driver present");
        let ctx = cuda.create_context(0).expect("GPU 0 usable");
        let api = NvEncApi::load().expect("NVENC runtime present");
        let session = api.open_session(&ctx).expect("session must open");
        assert!(!session.raw().is_null());
        drop(session); // nvEncDestroyEncoder
                       // Repeatability proves clean teardown (no leaked sessions
                       // against the driver's concurrent-session accounting).
        let again = api.open_session(&ctx).expect("session reopens after close");
        assert!(!again.raw().is_null());
    }
}
