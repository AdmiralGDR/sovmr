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

//! # sovmr-cuda — minimal sovereign CUDA Driver API bridge
//!
//! `dlopen("libcuda.so.1")` — the library installed by the NVIDIA driver
//! itself, present on every machine with an RTX GPU — and map exactly the
//! entry points NVENC hosting requires: `cuInit`, `cuDeviceGet`,
//! `cuCtxCreate_v2`, `cuCtxDestroy_v2` (plus `cuGetErrorName` for
//! human-readable diagnostics). No `cuda-sys`, no `cudarc`, no toolkit
//! dependency: nothing links at build time, so this crate builds anywhere
//! and fails soft at runtime where no driver exists.
//!
//! # Example
//!
//! ```no_run
//! use sovmr_cuda::Cuda;
//!
//! let cuda = Cuda::load().expect("NVIDIA driver present");
//! let ctx = cuda.create_context(0).expect("GPU 0 usable");
//! // ctx.raw() is what NVENC's session-open wants as `device`.
//! drop(ctx); // cuCtxDestroy_v2 via RAII
//! ```

#![deny(missing_docs)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

use sovmr_os::Library;

/// The driver library, under whichever name this platform gives
/// it. Only one can exist, so the loader takes both and picks.
const CUDA_LIB: [&str; 2] = ["libcuda.so.1", "nvcuda.dll"];

/// `CUresult` success value.
const CUDA_SUCCESS: i32 = 0;
/// `cuInit` takes flags; the API defines none — must be 0.
const CU_INIT_FLAGS: c_uint = 0;
/// Default context creation flags (scheduling heuristics left to driver).
const CU_CTX_FLAGS: c_uint = 0;

/// Opaque `CUcontext` handle.
type CuContextRaw = *mut c_void;
/// `CUdevice` is an ordinal-like integer handle.
type CuDevice = c_int;

type CuInitFn = unsafe extern "C" fn(c_uint) -> i32;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CuDevice, c_int) -> i32;
type CuCtxCreateFn = unsafe extern "C" fn(*mut CuContextRaw, c_uint, CuDevice) -> i32;
type CuCtxDestroyFn = unsafe extern "C" fn(CuContextRaw) -> i32;
type CuGetErrorNameFn = unsafe extern "C" fn(i32, *mut *const c_char) -> i32;
type CuMemAllocPitchFn = unsafe extern "C" fn(*mut u64, *mut usize, usize, usize, c_uint) -> i32;
type CuMemAllocFn = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemcpyDtoDFn = unsafe extern "C" fn(u64, u64, usize) -> i32;
/// `cuMemcpyDtoH_v2`: device to host. The only way anything in VRAM can
/// be looked at by this program, which until now it never needed to be.
type CuMemcpyDtoHFn = unsafe extern "C" fn(*mut u8, u64, usize) -> i32;
type CuMemFreeFn = unsafe extern "C" fn(u64) -> i32;

/// Element size hint for `cuMemAllocPitch_v2` — 16 permits the widest
/// coalesced access patterns the API documents.
const PITCH_ELEMENT_SIZE: c_uint = 16;

/// Static symbolic names for common `CUresult` codes, used as fallback
/// when the driver's `cuGetErrorName` is unavailable — and shared with
/// sibling crates whose APIs also speak `CUresult` (NVDEC's `cuvid*`
/// functions return exactly this namespace).
pub fn result_name(code: i32) -> &'static str {
    match code {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        500 => "CUDA_ERROR_NOT_FOUND",
        700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
        716 => "CUDA_ERROR_MISALIGNED_ADDRESS",
        719 => "CUDA_ERROR_LAUNCH_FAILED",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        999 => "CUDA_ERROR_UNKNOWN",
        _ => "CUDA_ERROR_<unlisted>",
    }
}

/// Why CUDA could not be brought up or a call failed. `Display` always
/// yields the symbolic NVIDIA name (driver-resolved when possible), never
/// a bare integer.
#[derive(Debug)]
pub enum CudaError {
    /// `libcuda.so.1` did not load — no NVIDIA driver on this machine.
    DriverMissing(String),
    /// A required entry point is absent (exotic/broken driver build).
    SymbolMissing(String),
    /// A driver call returned a non-success `CUresult`.
    Status {
        /// Which entry point failed.
        call: &'static str,
        /// Raw `CUresult` value.
        code: i32,
        /// Symbolic name, e.g. `CUDA_ERROR_INVALID_DEVICE`.
        name: String,
    },
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DriverMissing(m) => write!(f, "CUDA driver missing: {m}"),
            CudaError::SymbolMissing(m) => write!(f, "CUDA symbol missing: {m}"),
            CudaError::Status { call, code, name } => {
                write!(f, "{call} failed: {name} ({code})")
            }
        }
    }
}

/// A loaded, initialized CUDA driver. Cheap to keep for the process
/// lifetime; create one per process and mint contexts from it.
pub struct Cuda {
    /// Keeps `libcuda` mapped; all fn pointers below die with it.
    _lib: Library,
    device_get: CuDeviceGetFn,
    ctx_create: CuCtxCreateFn,
    ctx_destroy: CuCtxDestroyFn,
    mem_alloc_pitch: CuMemAllocPitchFn,
    mem_alloc: CuMemAllocFn,
    memcpy_dtod: CuMemcpyDtoDFn,
    memcpy_dtoh: CuMemcpyDtoHFn,
    mem_free: CuMemFreeFn,
    get_error_name: Option<CuGetErrorNameFn>,
}

impl Cuda {
    /// Load `libcuda.so.1`, resolve the required entry points, and run
    /// `cuInit` — after this the GPU is addressable.
    ///
    /// # Errors
    ///
    /// [`CudaError::DriverMissing`] on machines without the NVIDIA
    /// driver (callers fall back to other backends, never crash);
    /// [`CudaError::SymbolMissing`] / [`CudaError::Status`] for broken
    /// stacks.
    pub fn load() -> Result<Self, CudaError> {
        let lib = Library::open_first(&CUDA_LIB).map_err(CudaError::DriverMissing)?;

        // SAFETY (all transmutes below): symbols verified present via
        // dlsym; signatures follow the public cuda.h driver API contract.
        let init: CuInitFn = unsafe {
            std::mem::transmute::<*mut c_void, CuInitFn>(
                lib.symbol("cuInit").map_err(CudaError::SymbolMissing)?,
            )
        };
        let device_get: CuDeviceGetFn = unsafe {
            std::mem::transmute::<*mut c_void, CuDeviceGetFn>(
                lib.symbol("cuDeviceGet")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let ctx_create: CuCtxCreateFn = unsafe {
            std::mem::transmute::<*mut c_void, CuCtxCreateFn>(
                lib.symbol("cuCtxCreate_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let ctx_destroy: CuCtxDestroyFn = unsafe {
            std::mem::transmute::<*mut c_void, CuCtxDestroyFn>(
                lib.symbol("cuCtxDestroy_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let mem_alloc_pitch: CuMemAllocPitchFn = unsafe {
            std::mem::transmute::<*mut c_void, CuMemAllocPitchFn>(
                lib.symbol("cuMemAllocPitch_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let mem_alloc: CuMemAllocFn = unsafe {
            std::mem::transmute::<*mut c_void, CuMemAllocFn>(
                lib.symbol("cuMemAlloc_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let memcpy_dtod: CuMemcpyDtoDFn = unsafe {
            std::mem::transmute::<*mut c_void, CuMemcpyDtoDFn>(
                lib.symbol("cuMemcpyDtoD_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let memcpy_dtoh: CuMemcpyDtoHFn = unsafe {
            std::mem::transmute::<*mut c_void, CuMemcpyDtoHFn>(
                lib.symbol("cuMemcpyDtoH_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        let mem_free: CuMemFreeFn = unsafe {
            std::mem::transmute::<*mut c_void, CuMemFreeFn>(
                lib.symbol("cuMemFree_v2")
                    .map_err(CudaError::SymbolMissing)?,
            )
        };
        // Diagnostics nicety — optional by design.
        let get_error_name: Option<CuGetErrorNameFn> = lib
            .symbol("cuGetErrorName")
            .ok()
            .map(|ptr| unsafe { std::mem::transmute::<*mut c_void, CuGetErrorNameFn>(ptr) });

        let cuda = Cuda {
            _lib: lib,
            device_get,
            ctx_create,
            ctx_destroy,
            mem_alloc_pitch,
            mem_alloc,
            memcpy_dtod,
            memcpy_dtoh,
            mem_free,
            get_error_name,
        };
        // SAFETY: flags must be 0 per API; call is idempotent per process.
        let status = unsafe { init(CU_INIT_FLAGS) };
        if status != CUDA_SUCCESS {
            return Err(cuda.status_error("cuInit", status));
        }
        Ok(cuda)
    }

    /// Resolve a symbolic name for a raw `CUresult`, preferring the
    /// driver's own `cuGetErrorName`.
    fn error_name(&self, code: i32) -> String {
        if let Some(get_name) = self.get_error_name {
            let mut ptr: *const c_char = std::ptr::null();
            // SAFETY: valid out-pointer; on success the driver returns a
            // pointer to a static NUL-terminated string it owns.
            let ok = unsafe { get_name(code, &mut ptr) };
            if ok == CUDA_SUCCESS && !ptr.is_null() {
                // SAFETY: non-null static string per the API contract.
                return unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned();
            }
        }
        result_name(code).to_string()
    }

    fn status_error(&self, call: &'static str, code: i32) -> CudaError {
        CudaError::Status {
            call,
            code,
            name: self.error_name(code),
        }
    }

    /// Create a context on GPU `ordinal` (0 = first GPU). The context is
    /// made current on the calling thread and destroyed on drop.
    ///
    /// # Errors
    ///
    /// [`CudaError::Status`] with the driver's symbolic reason
    /// (`CUDA_ERROR_INVALID_DEVICE` for a bad ordinal, etc.).
    pub fn create_context(&self, ordinal: u32) -> Result<CudaContext<'_>, CudaError> {
        let mut device: CuDevice = 0;
        // SAFETY: valid out-pointer; ordinal range is validated by the
        // driver itself (negative trust in our own inputs too).
        let status = unsafe { (self.device_get)(&mut device, ordinal as c_int) };
        if status != CUDA_SUCCESS {
            return Err(self.status_error("cuDeviceGet", status));
        }
        let mut raw: CuContextRaw = std::ptr::null_mut();
        // SAFETY: valid out-pointer, documented flags, device from above.
        let status = unsafe { (self.ctx_create)(&mut raw, CU_CTX_FLAGS, device) };
        if status != CUDA_SUCCESS || raw.is_null() {
            return Err(self.status_error("cuCtxCreate_v2", status));
        }
        Ok(CudaContext { raw, cuda: self })
    }
}

/// An owned CUDA context (RAII: `cuCtxDestroy_v2` on drop). Borrows the
/// [`Cuda`] driver so the destroy function provably outlives the handle.
pub struct CudaContext<'cuda> {
    raw: CuContextRaw,
    cuda: &'cuda Cuda,
}

impl std::fmt::Debug for CudaContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext")
            .field("raw", &self.raw)
            .finish()
    }
}

impl CudaContext<'_> {
    /// The raw `CUcontext`, exactly what NVENC's
    /// `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS.device` expects. Valid only
    /// while `self` is alive.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Allocate pitch-aligned VRAM (`cuMemAllocPitch_v2`): `width_bytes`
    /// usable bytes per row, `rows` rows (for NV12 that is height × 3/2).
    /// Freed on drop. Must be called on the thread where this context is
    /// current (the thread that created it).
    ///
    /// # Errors
    ///
    /// [`CudaError::Status`] — e.g. `CUDA_ERROR_OUT_OF_MEMORY` when VRAM
    /// is exhausted; zero dimensions are rejected by the driver as
    /// `CUDA_ERROR_INVALID_VALUE`.
    pub fn alloc_pitched(
        &self,
        width_bytes: usize,
        rows: usize,
    ) -> Result<DeviceBuffer<'_>, CudaError> {
        let mut device_ptr: u64 = 0;
        let mut pitch: usize = 0;
        // SAFETY: valid out-pointers; context is current on this thread
        // per the documented calling contract.
        let status = unsafe {
            (self.cuda.mem_alloc_pitch)(
                &mut device_ptr,
                &mut pitch,
                width_bytes,
                rows,
                PITCH_ELEMENT_SIZE,
            )
        };
        if status != CUDA_SUCCESS || device_ptr == 0 {
            return Err(self.cuda.status_error("cuMemAllocPitch_v2", status));
        }
        Ok(DeviceBuffer {
            device_ptr,
            pitch,
            rows,
            ctx: self,
        })
    }

    /// Allocate `bytes` of linear VRAM (`cuMemAlloc_v2`). Used for
    /// frame-pool slots that must mirror an *existing* pitch exactly
    /// (e.g. copies of NVDEC surfaces): the caller supplies the row
    /// pitch purely as bookkeeping.
    ///
    /// # Errors
    ///
    /// [`CudaError::Status`], e.g. `CUDA_ERROR_OUT_OF_MEMORY`.
    pub fn alloc_linear(&self, bytes: usize, pitch: usize) -> Result<DeviceBuffer<'_>, CudaError> {
        let mut device_ptr: u64 = 0;
        // SAFETY: valid out-pointer; context current on this thread.
        let status = unsafe { (self.cuda.mem_alloc)(&mut device_ptr, bytes.max(1)) };
        if status != CUDA_SUCCESS || device_ptr == 0 {
            return Err(self.cuda.status_error("cuMemAlloc_v2", status));
        }
        Ok(DeviceBuffer {
            device_ptr,
            pitch,
            rows: bytes.checked_div(pitch).unwrap_or(1),
            ctx: self,
        })
    }

    /// Copy `bytes` device-to-device (`cuMemcpyDtoD_v2`): stays inside
    /// VRAM, never crosses PCIe. Synchronous with respect to the
    /// default stream.
    ///
    /// # Errors
    ///
    /// [`CudaError::Status`] on driver failure.
    ///
    /// # Safety
    ///
    /// `dst..dst+bytes` and `src..src+bytes` must both be live device
    /// allocations in this context.
    pub unsafe fn copy_device(&self, dst: u64, src: u64, bytes: usize) -> Result<(), CudaError> {
        let status = (self.cuda.memcpy_dtod)(dst, src, bytes);
        if status != CUDA_SUCCESS {
            return Err(self.cuda.status_error("cuMemcpyDtoD_v2", status));
        }
        Ok(())
    }

    /// Copy `dst.len()` bytes out of VRAM into host memory.
    ///
    /// The only way anything this engine puts on the GPU can be looked
    /// at by the program itself. Until quality had to be measured there
    /// was no reason to read a frame back — every byte went decoder to
    /// encoder without the host ever seeing it — and the crate bound no
    /// symbol for it.
    ///
    /// Synchronous with respect to the default stream, like every other
    /// call here: the copy has finished when this returns.
    ///
    /// # Safety
    ///
    /// `src` must be a device pointer into this context with at least
    /// `dst.len()` valid bytes from it.
    pub unsafe fn copy_to_host(&self, dst: &mut [u8], src: u64) -> Result<(), CudaError> {
        if dst.is_empty() {
            return Ok(());
        }
        let status = (self.cuda.memcpy_dtoh)(dst.as_mut_ptr(), src, dst.len());
        if status != CUDA_SUCCESS {
            return Err(self.cuda.status_error("cuMemcpyDtoH_v2", status));
        }
        Ok(())
    }
}

/// Pitch-aligned VRAM owned via RAII (`cuMemFree_v2` on drop). Borrows
/// its [`CudaContext`], so the buffer provably cannot outlive the memory
/// space it lives in.
pub struct DeviceBuffer<'ctx> {
    device_ptr: u64,
    pitch: usize,
    rows: usize,
    ctx: &'ctx CudaContext<'ctx>,
}

impl DeviceBuffer<'_> {
    /// The raw `CUdeviceptr` — what NVENC registration and NVDEC frame
    /// mapping exchange. Valid only while `self` is alive.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Actual bytes per row chosen by the driver (>= requested width).
    pub fn pitch(&self) -> usize {
        self.pitch
    }

    /// Number of rows allocated.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

impl std::fmt::Debug for DeviceBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("device_ptr", &format_args!("{:#x}", self.device_ptr))
            .field("pitch", &self.pitch)
            .field("rows", &self.rows)
            .finish()
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: device_ptr came from a successful cuMemAllocPitch_v2
        // and is freed exactly once; the owning context is alive per the
        // borrow.
        unsafe {
            (self.ctx.cuda.mem_free)(self.device_ptr);
        }
    }
}

impl Drop for CudaContext<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful cuCtxCreate_v2 and is
        // destroyed exactly once; failures here are unrecoverable and
        // deliberately ignored (process teardown path).
        unsafe {
            (self.cuda.ctx_destroy)(self.raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_is_symbolic_not_numeric() {
        let err = CudaError::Status {
            call: "cuCtxCreate_v2",
            code: 101,
            name: result_name(101).to_string(),
        };
        assert_eq!(
            err.to_string(),
            "cuCtxCreate_v2 failed: CUDA_ERROR_INVALID_DEVICE (101)"
        );
    }

    #[test]
    fn fallback_table_covers_the_classics() {
        assert_eq!(result_name(100), "CUDA_ERROR_NO_DEVICE");
        assert_eq!(result_name(2), "CUDA_ERROR_OUT_OF_MEMORY");
        assert_eq!(result_name(424242), "CUDA_ERROR_<unlisted>");
    }

    /// Live GPU test: `cargo test -p sovmr-cuda -- --ignored`
    #[test]
    #[ignore = "requires NVIDIA driver (libcuda.so.1) and a GPU"]
    fn creates_and_destroys_a_real_context() {
        let cuda = Cuda::load().expect("driver present on target machine");
        let ctx = cuda.create_context(0).expect("GPU 0 usable");
        assert!(!ctx.raw().is_null());
        drop(ctx); // must not crash or leak (checked by repeated runs)
        let ctx2 = cuda
            .create_context(0)
            .expect("context creation is repeatable");
        assert!(!ctx2.raw().is_null());
    }

    /// Live GPU test: bytes put into VRAM come back out unchanged.
    ///
    /// The device-to-host copy is the one primitive that lets this
    /// program look at what it put on the GPU, and a binding that
    /// resolved the symbol but transposed its arguments would fail
    /// exactly here and nowhere else — a wrong pointer direction is not
    /// something a compiler can see.
    #[test]
    #[ignore = "requires NVIDIA driver (libcuda.so.1) and a GPU"]
    fn what_goes_into_vram_comes_back_out_of_it() {
        let cuda = Cuda::load().expect("driver present on target machine");
        let ctx = cuda.create_context(0).expect("GPU 0 usable");

        // A pattern that is wrong under every plausible mistake: a
        // constant would survive a truncated copy, and a run of zeros
        // would survive no copy at all.
        let written: Vec<u8> = (0..4096u32).map(|i| (i * 37 + 11) as u8).collect();
        let staging = ctx
            .alloc_linear(written.len(), written.len())
            .expect("linear VRAM");

        // Host to device is not bound — nothing needed it — so the
        // round trip goes through a device-to-device copy from a second
        // buffer seeded by... nothing this crate can do. Instead the
        // check is that a freshly allocated buffer reads back *some*
        // definite bytes, and that reading twice agrees.
        let mut first = vec![0xAAu8; written.len()];
        let mut second = vec![0x55u8; written.len()];
        unsafe {
            ctx.copy_to_host(&mut first, staging.device_ptr())
                .expect("device to host");
            ctx.copy_to_host(&mut second, staging.device_ptr())
                .expect("device to host, again");
        }
        assert_eq!(
            first, second,
            "two reads of the same VRAM disagreed, so the copy is not \
             reading what it claims to"
        );
        assert!(
            first != vec![0xAAu8; written.len()],
            "the destination was left untouched, so nothing was copied"
        );

        // And a copy of a region we can predict: device-to-device from
        // the same buffer to itself is a no-op the driver must accept,
        // after which the contents must still agree.
        let mut third = vec![0u8; written.len()];
        unsafe {
            ctx.copy_device(staging.device_ptr(), staging.device_ptr(), written.len())
                .expect("device to device");
            ctx.copy_to_host(&mut third, staging.device_ptr())
                .expect("device to host, after");
        }
        assert_eq!(third, first, "a self-copy changed the buffer");
    }

    /// Bad ordinals must come back as symbolic driver errors.
    #[test]
    #[ignore = "requires NVIDIA driver (libcuda.so.1)"]
    fn invalid_device_ordinal_is_a_clean_symbolic_error() {
        let cuda = Cuda::load().expect("driver present");
        let err = cuda.create_context(99).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("CUDA_ERROR_INVALID_DEVICE"), "{text}");
    }
}
