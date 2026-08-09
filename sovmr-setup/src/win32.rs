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

//! The Win32 half of the installer: registry, Start-menu shortcut, and
//! the message boxes a double-clicked installer owes the user.
//!
//! Every declaration here is hand-written `extern "system"`, and every
//! COM vtable index was read out of the mingw-w64 headers this build
//! links against (`shobjidl.h`, `objidl.h`) rather than recalled — the
//! same methodology that pinned the NVENC structs.

use std::ffi::{c_void, OsStr};
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

// ---- string helpers ---------------------------------------------------------

/// Encode as a NUL-terminated UTF-16 string for the `W` APIs.
pub fn wide(s: impl AsRef<OsStr>) -> Vec<u16> {
    s.as_ref().encode_wide().chain(std::iter::once(0)).collect()
}

// ---- registry ---------------------------------------------------------------

/// `HKEY_CURRENT_USER`. Per-user install, so no elevation is ever asked
/// for: the whole thing lives under the user's own profile and registry
/// hive. An installer that demands administrator rights to drop one
/// executable in a home directory is asking for trust it does not need.
const HKEY_CURRENT_USER: *mut c_void = 0x8000_0001_usize as *mut c_void;
/// `KEY_ALL_ACCESS`.
const KEY_ALL_ACCESS: u32 = 0x000F_003F;
/// `REG_SZ` / `REG_DWORD`.
const REG_SZ: u32 = 1;
const REG_DWORD: u32 = 4;
/// `ERROR_SUCCESS`.
const ERROR_SUCCESS: i32 = 0;

#[link(name = "advapi32")]
extern "system" {
    fn RegCreateKeyExW(
        key: *mut c_void,
        sub_key: *const u16,
        reserved: u32,
        class: *const u16,
        options: u32,
        desired: u32,
        security: *mut c_void,
        result: *mut *mut c_void,
        disposition: *mut u32,
    ) -> i32;
    fn RegSetValueExW(
        key: *mut c_void,
        value_name: *const u16,
        reserved: u32,
        ty: u32,
        data: *const u8,
        data_len: u32,
    ) -> i32;
    fn RegQueryValueExW(
        key: *mut c_void,
        value_name: *const u16,
        reserved: *mut u32,
        ty: *mut u32,
        data: *mut u8,
        data_len: *mut u32,
    ) -> i32;
    fn RegOpenKeyExW(
        key: *mut c_void,
        sub_key: *const u16,
        options: u32,
        desired: u32,
        result: *mut *mut c_void,
    ) -> i32;
    fn RegCloseKey(key: *mut c_void) -> i32;
    fn RegDeleteTreeW(key: *mut c_void, sub_key: *const u16) -> i32;
}

/// An open registry key, closed on drop.
struct RegKey(*mut c_void);

impl Drop for RegKey {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful open/create call.
        unsafe { RegCloseKey(self.0) };
    }
}

/// Create (or open) `HKCU\{path}` for writing.
fn create_key(path: &str) -> Result<RegKey, String> {
    let sub = wide(path);
    let mut handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: `sub` is NUL-terminated UTF-16; `handle` is a valid out-pointer.
    let rc = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            sub.as_ptr(),
            0,
            std::ptr::null(),
            0,
            KEY_ALL_ACCESS,
            std::ptr::null_mut(),
            &mut handle,
            std::ptr::null_mut(),
        )
    };
    if rc != ERROR_SUCCESS {
        return Err(format!("RegCreateKeyEx({path}) failed with {rc}"));
    }
    Ok(RegKey(handle))
}

/// Write one `REG_SZ` value.
fn set_string(key: &RegKey, name: &str, value: &str) -> Result<(), String> {
    let n = wide(name);
    let v = wide(value);
    let bytes = std::mem::size_of_val(&v[..]) as u32;
    // SAFETY: both strings are NUL-terminated UTF-16; `bytes` is their
    // exact byte length including the terminator, as REG_SZ requires.
    let rc =
        unsafe { RegSetValueExW(key.0, n.as_ptr(), 0, REG_SZ, v.as_ptr().cast::<u8>(), bytes) };
    if rc != ERROR_SUCCESS {
        return Err(format!("RegSetValueEx({name}) failed with {rc}"));
    }
    Ok(())
}

/// Write one `REG_DWORD` value.
fn set_dword(key: &RegKey, name: &str, value: u32) -> Result<(), String> {
    let n = wide(name);
    // SAFETY: name is NUL-terminated; the data is exactly 4 bytes.
    let rc = unsafe {
        RegSetValueExW(
            key.0,
            n.as_ptr(),
            0,
            REG_DWORD,
            std::ptr::addr_of!(value).cast::<u8>(),
            4,
        )
    };
    if rc != ERROR_SUCCESS {
        return Err(format!("RegSetValueEx({name}) failed with {rc}"));
    }
    Ok(())
}

/// Read one `REG_SZ` value, or `None` when the key or value is absent.
pub fn read_string(path: &str, name: &str) -> Option<String> {
    let sub = wide(path);
    let mut handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: NUL-terminated inputs, valid out-pointer.
    let rc = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            sub.as_ptr(),
            0,
            KEY_ALL_ACCESS,
            &mut handle,
        )
    };
    if rc != ERROR_SUCCESS {
        return None;
    }
    let key = RegKey(handle);
    let n = wide(name);
    let mut len: u32 = 0;
    // First call sizes the buffer; negative trust means never assuming
    // a maximum and never writing past what the API reported.
    // SAFETY: null data pointer with a valid length out-pointer is the
    // documented way to ask for the size.
    let rc = unsafe {
        RegQueryValueExW(
            key.0,
            n.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut len,
        )
    };
    if rc != ERROR_SUCCESS || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    // SAFETY: buffer is exactly `len` bytes, which is what the call above
    // reported it needs.
    let rc = unsafe {
        RegQueryValueExW(
            key.0,
            n.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            buf.as_mut_ptr(),
            &mut len,
        )
    };
    if rc != ERROR_SUCCESS {
        return None;
    }
    let units: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .take_while(|u| *u != 0)
        .collect();
    Some(String::from_utf16_lossy(&units))
}

/// Remove a key and everything under it.
pub fn delete_tree(path: &str) -> Result<(), String> {
    let sub = wide(path);
    // SAFETY: NUL-terminated UTF-16 subkey path.
    let rc = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, sub.as_ptr()) };
    // Absent is the desired end state, so "not found" is success.
    const ERROR_FILE_NOT_FOUND: i32 = 2;
    if rc != ERROR_SUCCESS && rc != ERROR_FILE_NOT_FOUND {
        return Err(format!("RegDeleteTree({path}) failed with {rc}"));
    }
    Ok(())
}

/// Register the program in "Apps & features".
///
/// This is what separates an installed application from a folder someone
/// copied: the user can find it, see its version, and remove it the way
/// they remove everything else.
#[allow(clippy::too_many_arguments)]
pub fn write_uninstall_entry(
    key_path: &str,
    display_name: &str,
    version: &str,
    publisher: &str,
    install_dir: &Path,
    uninstall_cmd: &str,
    size_kib: u32,
) -> Result<(), String> {
    let key = create_key(key_path)?;
    set_string(&key, "DisplayName", display_name)?;
    set_string(&key, "DisplayVersion", version)?;
    set_string(&key, "Publisher", publisher)?;
    set_string(&key, "InstallLocation", &install_dir.display().to_string())?;
    set_string(&key, "UninstallString", uninstall_cmd)?;
    set_dword(&key, "EstimatedSize", size_kib)?;
    // Per-user install: say so, rather than letting the shell guess.
    set_dword(&key, "NoModify", 1)?;
    set_dword(&key, "NoRepair", 1)?;
    Ok(())
}

// ---- COM: the Start-menu shortcut -------------------------------------------

/// A COM `GUID`, same shape as everywhere else in this project.
#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

/// `CLSID_ShellLink` — {00021401-0000-0000-C000-000000000046}.
const CLSID_SHELL_LINK: Guid = Guid {
    data1: 0x0002_1401,
    data2: 0,
    data3: 0,
    data4: [0xc0, 0, 0, 0, 0, 0, 0, 0x46],
};
/// `IID_IShellLinkW` — {000214F9-0000-0000-C000-000000000046}, as
/// declared by `__CRT_UUID_DECL(IShellLinkW, ...)` in `shobjidl.h`.
const IID_ISHELL_LINK_W: Guid = Guid {
    data1: 0x0002_14f9,
    data2: 0,
    data3: 0,
    data4: [0xc0, 0, 0, 0, 0, 0, 0, 0x46],
};
/// `IID_IPersistFile` — {0000010B-0000-0000-C000-000000000046}.
const IID_IPERSIST_FILE: Guid = Guid {
    data1: 0x0000_010b,
    data2: 0,
    data3: 0,
    data4: [0xc0, 0, 0, 0, 0, 0, 0, 0x46],
};

/// `COINIT_APARTMENTTHREADED`.
const COINIT_APARTMENTTHREADED: u32 = 0x2;
/// `CLSCTX_INPROC_SERVER`.
const CLSCTX_INPROC_SERVER: u32 = 0x1;

/// Vtable slots, counted from the mingw-w64 headers this build links
/// against — not from memory. `IShellLinkW` (shobjidl.h): QueryInterface,
/// AddRef, Release, GetPath, GetIDList, SetIDList, GetDescription,
/// SetDescription, GetWorkingDirectory, SetWorkingDirectory,
/// GetArguments, SetArguments, GetHotkey, SetHotkey, GetShowCmd,
/// SetShowCmd, GetIconLocation, SetIconLocation, SetRelativePath,
/// Resolve, SetPath.
const SLOT_RELEASE: usize = 2;
const SLOT_QUERY_INTERFACE: usize = 0;
const SLOT_SET_DESCRIPTION: usize = 7;
const SLOT_SET_WORKING_DIRECTORY: usize = 9;
const SLOT_SET_PATH: usize = 20;
/// `IPersistFile` (objidl.h): QueryInterface, AddRef, Release,
/// GetClassID, IsDirty, Load, Save, SaveCompleted, GetCurFile.
const SLOT_PERSIST_SAVE: usize = 6;

#[link(name = "ole32")]
extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, init: u32) -> i32;
    fn CoUninitialize();
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        context: u32,
        iid: *const Guid,
        out: *mut *mut c_void,
    ) -> i32;
}

/// Call vtable slot `n` on a COM object with the given argument tuple.
///
/// COM objects are a pointer to a pointer to an array of function
/// pointers; there is no way to reach them from Rust without this shape.
///
/// # Safety
///
/// `obj` must be a live COM interface pointer whose vtable really has
/// the named slot with the given signature.
unsafe fn vcall2<A, B>(obj: *mut c_void, slot: usize, a: A, b: B) -> i32 {
    let vtable = *obj.cast::<*const *const c_void>();
    let f: unsafe extern "system" fn(*mut c_void, A, B) -> i32 =
        std::mem::transmute(*vtable.add(slot));
    f(obj, a, b)
}

/// As [`vcall2`] with one argument.
///
/// # Safety
///
/// As [`vcall2`].
unsafe fn vcall1<A>(obj: *mut c_void, slot: usize, a: A) -> i32 {
    let vtable = *obj.cast::<*const *const c_void>();
    let f: unsafe extern "system" fn(*mut c_void, A) -> i32 =
        std::mem::transmute(*vtable.add(slot));
    f(obj, a)
}

/// Release a COM interface pointer.
///
/// # Safety
///
/// `obj` must be a live interface pointer not yet released.
unsafe fn release(obj: *mut c_void) {
    let vtable = *obj.cast::<*const *const c_void>();
    let f: unsafe extern "system" fn(*mut c_void) -> u32 =
        std::mem::transmute(*vtable.add(SLOT_RELEASE));
    f(obj);
}

/// Write a `.lnk` shortcut pointing at `target`.
///
/// # Errors
///
/// The failing COM call and its `HRESULT`.
pub fn create_shortcut(
    link_path: &Path,
    target: &Path,
    working_dir: &Path,
    description: &str,
) -> Result<(), String> {
    // SAFETY: standard COM bring-up on this thread.
    let hr = unsafe { CoInitializeEx(std::ptr::null_mut(), COINIT_APARTMENTTHREADED) };
    // S_OK or S_FALSE (already initialised) are both fine; anything else
    // is fatal for this operation.
    if hr < 0 {
        return Err(format!("CoInitializeEx failed with 0x{hr:08x}"));
    }
    let result = create_shortcut_inner(link_path, target, working_dir, description);
    // SAFETY: paired with the successful CoInitializeEx above.
    unsafe { CoUninitialize() };
    result
}

fn create_shortcut_inner(
    link_path: &Path,
    target: &Path,
    working_dir: &Path,
    description: &str,
) -> Result<(), String> {
    let mut link: *mut c_void = std::ptr::null_mut();
    // SAFETY: well-known CLSID/IID pair; `link` is a valid out-pointer.
    let hr = unsafe {
        CoCreateInstance(
            &CLSID_SHELL_LINK,
            std::ptr::null_mut(),
            CLSCTX_INPROC_SERVER,
            &IID_ISHELL_LINK_W,
            &mut link,
        )
    };
    if hr < 0 || link.is_null() {
        return Err(format!(
            "CoCreateInstance(ShellLink) failed with 0x{hr:08x}"
        ));
    }
    // From here on `link` must be released on every path.
    let outcome = (|| -> Result<(), String> {
        let target_w = wide(target);
        let dir_w = wide(working_dir);
        let desc_w = wide(description);

        // SAFETY: live IShellLinkW; each slot's signature is
        // `(this, *const u16) -> HRESULT` per shobjidl.h.
        let hr = unsafe { vcall1(link, SLOT_SET_PATH, target_w.as_ptr()) };
        if hr < 0 {
            return Err(format!("IShellLinkW::SetPath failed with 0x{hr:08x}"));
        }
        // SAFETY: as above.
        let hr = unsafe { vcall1(link, SLOT_SET_WORKING_DIRECTORY, dir_w.as_ptr()) };
        if hr < 0 {
            return Err(format!("SetWorkingDirectory failed with 0x{hr:08x}"));
        }
        // SAFETY: as above.
        let hr = unsafe { vcall1(link, SLOT_SET_DESCRIPTION, desc_w.as_ptr()) };
        if hr < 0 {
            return Err(format!("SetDescription failed with 0x{hr:08x}"));
        }

        let mut persist: *mut c_void = std::ptr::null_mut();
        // SAFETY: QueryInterface has signature (this, *const IID, *mut *mut c_void).
        let hr = unsafe {
            vcall2(
                link,
                SLOT_QUERY_INTERFACE,
                &IID_IPERSIST_FILE as *const Guid,
                &mut persist as *mut *mut c_void,
            )
        };
        if hr < 0 || persist.is_null() {
            return Err(format!("QueryInterface(IPersistFile) failed 0x{hr:08x}"));
        }
        let link_w = wide(link_path);
        // SAFETY: IPersistFile::Save is (this, *const u16, BOOL).
        let hr = unsafe { vcall2(persist, SLOT_PERSIST_SAVE, link_w.as_ptr(), 1i32) };
        // SAFETY: releasing the interface we just obtained, once.
        unsafe { release(persist) };
        if hr < 0 {
            return Err(format!("IPersistFile::Save failed with 0x{hr:08x}"));
        }
        Ok(())
    })();
    // SAFETY: releasing the ShellLink we created, exactly once.
    unsafe { release(link) };
    outcome
}

// ---- user-visible messages --------------------------------------------------

/// `MB_OK | MB_ICONINFORMATION` / `MB_ICONERROR`.
const MB_ICONINFORMATION: u32 = 0x40;
const MB_ICONERROR: u32 = 0x10;

#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(owner: *mut c_void, text: *const u16, caption: *const u16, ty: u32) -> i32;
}

/// Show a message box.
///
/// The installer deliberately has no progress bar. Writing one 13 MB file
/// to a local disk finishes faster than a window can usefully animate,
/// so a progress control would be decoration pretending to be
/// information — and decoration is still code that can go wrong.
pub fn message(caption: &str, text: &str, error: bool) {
    let c = wide(caption);
    let t = wide(text);
    let icon = if error {
        MB_ICONERROR
    } else {
        MB_ICONINFORMATION
    };
    // SAFETY: both strings are NUL-terminated UTF-16; a null owner means
    // "no parent window", which is correct for a standalone installer.
    unsafe { MessageBoxW(std::ptr::null_mut(), t.as_ptr(), c.as_ptr(), icon) };
}

/// Whether anyone is listening on standard output.
///
/// An installer double-clicked from Explorer must talk in message boxes;
/// the same binary driven from a script must talk on stdout, or an
/// unattended install hangs forever behind a dialog nobody can see.
///
/// The test is for a usable stdout **handle**, not for a console
/// *window*. `GetConsoleWindow` was the obvious choice and was wrong:
/// this binary is built with `windows_subsystem = "windows"`, so it
/// never owns a console window even when run over SSH, where stdout is a
/// perfectly good pipe. Measured the hard way — the first remote install
/// completed and then hung until the timeout, waiting on a message box
/// at the far end of an SSH session.
pub fn has_stdout() -> bool {
    /// `STD_OUTPUT_HANDLE`.
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    /// `INVALID_HANDLE_VALUE`.
    const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(std_handle: u32) -> *mut c_void;
    }
    // SAFETY: a documented constant in, a handle out; no state touched.
    let h = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
    !h.is_null() && h != INVALID_HANDLE_VALUE
}
