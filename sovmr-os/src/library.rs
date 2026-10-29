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

//! The sovereign dynamic loader.
//!
//! Hand-rolled dynamic loading: the four `dl*` functions declared
//! directly against glibc (no `libc` crate, no `libloading`), wrapped in
//! a small RAII [`Library`] type so downstream crates never touch raw
//! handles. Used by the GPU bridges (`sovmr-cuda`, `sovmr-nvdec`,
//! `sovmr-nvenc`) to reach the driver's libraries at runtime.
//!
//! Loading a driver library is a syscall by another name, so it belongs
//! on the platform boundary with the rest of them rather than in a crate
//! of its own — which is also what lets the Windows backend be a change
//! of one module instead of a change of every caller.
//!
//! # Example
//!
//! ```
//! use sovmr_os::Library;
//!
//! let libm = Library::open("libm.so.6").expect("libm ships with glibc");
//! let cos_addr = libm.symbol("cos").expect("cos is exported");
//! assert!(!cos_addr.is_null());
//! ```

use std::ffi::{c_char, c_void, CString};

#[cfg(unix)]
use std::ffi::{c_int, CStr};

// The dlopen family lives in libc itself on glibc >= 2.34; `libdl` is
// kept as a compatibility stub, so linking it works on every glibc that
// Pop!_OS 22.04+ can ship.
#[cfg(unix)]
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
}

/// Resolve all symbols at load time so failures surface immediately.
#[cfg(unix)]
const RTLD_NOW: c_int = 0x2;
/// Keep the library's symbols out of the global namespace.
#[cfg(unix)]
const RTLD_LOCAL: c_int = 0;

/// Consume and render the thread-local `dlerror` message.
#[cfg(unix)]
fn take_dlerror(context: &str) -> String {
    // SAFETY: dlerror returns either NULL or a pointer to a thread-local
    // NUL-terminated string owned by the loader; we only copy from it.
    let msg = unsafe {
        let ptr = dlerror();
        if ptr.is_null() {
            "unknown dlerror".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    format!("{context}: {msg}")
}

#[cfg(windows)]
extern "system" {
    fn LoadLibraryExW(file_name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

/// `LOAD_LIBRARY_SEARCH_SYSTEM32`: resolve the name against `System32`
/// only, never the current directory or `%PATH%`.
///
/// This is negative trust applied to the loader itself. The default
/// Windows search order consults the directory the executable was
/// started from, so a `nvcuda.dll` dropped next to `sovmr.exe` — by a
/// careless installer or a deliberate one — would be loaded in
/// preference to the driver's. Every library this crate opens is a
/// driver component that lives in `System32`, so nothing legitimate is
/// lost by refusing to look anywhere else.
#[cfg(windows)]
const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

/// Encode a name as a NUL-terminated UTF-16 string for the `W` APIs.
#[cfg(windows)]
fn wide(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(std::iter::once(0)).collect()
}

/// An open shared library. Closed on drop; symbol pointers obtained from
/// it must not outlive it (enforced by keeping the `Library` alive inside
/// whatever struct stores the pointers).
pub struct Library {
    handle: *mut c_void,
    name: &'static str,
}

// SAFETY: the loader handle is process-global state; glibc's dl* API is
// thread-safe (and so are LoadLibrary/GetProcAddress/FreeLibrary).
// Nothing here is tied to the opening thread, so moving a `Library` to
// another thread (and closing it there on drop) is sound.
unsafe impl Send for Library {}
// SAFETY: `&Library` only exposes `symbol()`, which calls dlsym /
// GetProcAddress on a handle that cannot be closed while the shared
// borrow exists (closing needs `Drop`, i.e. ownership). Both lookups are
// thread-safe and do not mutate the handle, so concurrent `symbol()`
// calls from several threads are sound. `dlerror` state read on failure
// is thread-local per POSIX.
unsafe impl Sync for Library {}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The raw handle value is meaningless to readers; the name is not.
        f.debug_struct("Library").field("name", &self.name).finish()
    }
}

impl Library {
    /// `dlopen` the library with immediate binding.
    ///
    /// # Errors
    ///
    /// The loader's own diagnostic (library missing, wrong architecture,
    /// unresolved symbols) as a display-ready string.
    /// Open the first of `names` that exists.
    ///
    /// The same driver is `libcuda.so.1` on one platform and
    /// `nvcuda.dll` on another, and only one of them can be present. So
    /// a caller lists both and asks for whichever is there — which is
    /// shorter than a conditional, and means a third platform is a
    /// third name rather than a third arm in four crates.
    ///
    /// # Errors
    ///
    /// Every failure, named, when none opened. Which name was expected
    /// is exactly what a user debugging a missing driver needs.
    pub fn open_first(names: &[&'static str]) -> Result<Self, String> {
        let mut refusals = Vec::new();
        for name in names {
            match Self::open(name) {
                Ok(lib) => return Ok(lib),
                Err(e) => refusals.push(e),
            }
        }
        Err(refusals.join("; "))
    }

    /// Open one library by name.
    ///
    /// # Errors
    ///
    /// The platform loader's own message, which names the library.
    pub fn open(name: &'static str) -> Result<Self, String> {
        #[cfg(unix)]
        {
            let c_name = CString::new(name).map_err(|e| format!("{name}: {e}"))?;
            // SAFETY: c_name is a valid NUL-terminated string; flags are a
            // documented combination.
            let handle = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
            if handle.is_null() {
                return Err(take_dlerror(name));
            }
            Ok(Library { handle, name })
        }
        #[cfg(windows)]
        {
            let w_name = wide(name);
            // SAFETY: w_name is NUL-terminated UTF-16; a null file handle
            // is required by the API; the flag is a documented value.
            let handle = unsafe {
                LoadLibraryExW(
                    w_name.as_ptr(),
                    std::ptr::null_mut(),
                    LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
            };
            if handle.is_null() {
                return Err(format!("{name}: {}", std::io::Error::last_os_error()));
            }
            Ok(Library { handle, name })
        }
    }

    /// Look up one exported symbol and return its raw address.
    ///
    /// # Errors
    ///
    /// The loader's diagnostic when the symbol is absent.
    ///
    /// # Safety notes for callers
    ///
    /// The returned pointer is only meaningful transmuted to the symbol's
    /// true C signature, and only while `self` is alive.
    pub fn symbol(&self, symbol: &str) -> Result<*mut c_void, String> {
        let c_symbol = CString::new(symbol).map_err(|e| format!("{symbol}: {e}"))?;
        // Symbol names are ASCII in both worlds: `GetProcAddress` takes a
        // narrow string even in a UTF-16 API, because export tables are
        // narrow.
        //
        // SAFETY: `self.handle` is non-null (checked in `open`) and not
        // yet closed (only `Drop` closes it, which cannot run while
        // `&self` is borrowed); `c_symbol` is a valid NUL-terminated
        // string that outlives the call.
        #[cfg(unix)]
        let ptr = unsafe { dlsym(self.handle, c_symbol.as_ptr()) };
        // SAFETY: same invariants as the unix branch, for an HMODULE
        // returned by a successful `LoadLibraryExW`.
        #[cfg(windows)]
        let ptr = unsafe { GetProcAddress(self.handle, c_symbol.as_ptr()) };
        if ptr.is_null() {
            #[cfg(unix)]
            return Err(take_dlerror(&format!("{}!{symbol}", self.name)));
            #[cfg(windows)]
            return Err(format!(
                "{}!{symbol}: {}",
                self.name,
                std::io::Error::last_os_error()
            ));
        }
        Ok(ptr)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: handle came from a successful `dlopen` (non-null) and
        // `drop` runs at most once, so it is released exactly once. Fn
        // pointers resolved from it are stored next to the `Library` in
        // their owning struct and die with it, so none is called after
        // this point.
        #[cfg(unix)]
        unsafe {
            dlclose(self.handle);
        }
        // SAFETY: as above, for an HMODULE from a successful
        // `LoadLibraryExW`.
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_nonexistent_library_reports_the_loader_error() {
        let err = Library::open("libsovmr-definitely-not-real.so.9").unwrap_err();
        assert!(err.contains("libsovmr-definitely-not-real"), "{err}");
    }

    /// A library guaranteed present wherever this test can run, with a
    /// symbol it is guaranteed to export.
    #[cfg(unix)]
    const ALWAYS_PRESENT: (&str, &str) = ("libm.so.6", "cos");
    #[cfg(windows)]
    const ALWAYS_PRESENT: (&str, &str) = ("kernel32.dll", "GetProcAddress");

    #[test]
    fn missing_symbols_are_errors_not_null_pointers() {
        let (lib_name, symbol) = ALWAYS_PRESENT;
        let lib = Library::open(lib_name).expect("the system library always loads");
        assert!(lib.symbol(symbol).is_ok());
        // A null return must become an error, never a null pointer handed
        // to a caller that would then call through it.
        assert!(lib.symbol("sovmr_not_a_symbol").is_err());
    }
}
