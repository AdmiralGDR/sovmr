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

//! Free space on the filesystem holding a given path.
//!
//! A twelve-hour encode runs for hours and spools its `mdat` to disk as
//! it goes. Discovering at hour two that the volume is full wastes the
//! whole run, so the engine asks this question *before* committing —
//! a bounded-execution rule applied to storage: a job that
//! cannot possibly finish must not be started.

use std::io;
use std::path::Path;

#[cfg(unix)]
use std::ffi::{c_char, c_int, CString};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

/// `struct statvfs` as laid out by glibc on this target, confirmed with
/// a C probe (`sizeof` 112; `f_bsize` 0, `f_frsize` 8, `f_blocks` 16,
/// `f_bfree` 24, `f_bavail` 32) rather than assumed. Only the leading
/// fields are named; the tail is reserved so the kernel always has the
/// room it expects to write into.
#[cfg(unix)]
#[repr(C)]
#[derive(Default)]
struct StatVfs {
    /// Preferred I/O block size.
    bsize: u64,
    /// Fundamental block size — the unit `bavail` counts in.
    frsize: u64,
    /// Total blocks.
    blocks: u64,
    /// Blocks free.
    bfree: u64,
    /// Blocks available to an unprivileged process.
    bavail: u64,
    /// `f_files` through `f_namemax` plus `__f_spare`: never read, but
    /// the syscall writes here, so the allocation must cover it.
    _reserved: [u64; 9],
}

#[cfg(unix)]
extern "C" {
    fn statvfs(path: *const c_char, buf: *mut StatVfs) -> c_int;
}

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

#[cfg(windows)]
extern "system" {
    /// Bytes available to the calling user on the volume holding a
    /// directory. The first output already accounts for per-user quota,
    /// which is what makes it the counterpart of `f_bavail` rather than
    /// of `f_bfree`.
    fn GetDiskFreeSpaceExW(
        directory_name: *const u16,
        free_bytes_available_to_caller: *mut u64,
        total_number_of_bytes: *mut u64,
        total_number_of_free_bytes: *mut u64,
    ) -> i32;
}

/// Bytes an unprivileged process may still write to the filesystem that
/// holds `path`.
///
/// `path` need not exist itself, but some existing ancestor must — the
/// question is about the *volume*, and a not-yet-created output file
/// still has a directory it will be created in.
///
/// # Errors
///
/// The OS diagnostic when the path cannot be resolved to a mounted
/// filesystem, and [`io::ErrorKind::Unsupported`] on platforms without a
/// backend yet.
///
/// # Example
///
/// ```
/// let free = sovmr_os::free_space(&std::env::temp_dir())?;
/// assert!(free > 0, "a writable temp dir has room");
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn free_space(path: &Path) -> io::Result<u64> {
    #[cfg(unix)]
    {
        // Negative trust: an interior NUL means the caller handed us a
        // path the OS can never accept, so say so instead of truncating.
        // Checked on the path as given, before any ancestor walk, so a
        // malformed argument is never silently answered by its parent.
        for component in path.as_os_str().as_bytes() {
            if *component == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "path contains an interior NUL",
                ));
            }
        }
        let target = nearest_existing(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no existing ancestor to measure", path.display()),
            )
        })?;
        let c_path = CString::new(target.as_os_str().as_bytes())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut buf = StatVfs::default();
        // SAFETY: c_path is NUL-terminated; buf is a live allocation of
        // exactly the size the C declaration describes.
        let rc = unsafe { statvfs(c_path.as_ptr(), &mut buf) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // `f_frsize` is the unit `f_bavail` counts in; a filesystem that
        // leaves it zero means "same as f_bsize".
        let unit = if buf.frsize > 0 {
            buf.frsize
        } else {
            buf.bsize
        };
        Ok(buf.bavail.saturating_mul(unit))
    }
    #[cfg(windows)]
    {
        // Negative trust, and checked on the path as given — before any
        // ancestor walk, exactly as the Unix arm does. An interior NUL
        // would otherwise truncate the string inside the Win32 layer and
        // answer confidently about the wrong volume.
        if path.as_os_str().encode_wide().any(|u| u == 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "path contains an interior NUL",
            ));
        }
        // `GetDiskFreeSpaceExW` answers about the volume holding the
        // named directory, but insists the name exist — the same
        // constraint `statvfs` has, and it gets the same answer: walk up
        // to the nearest existing ancestor, because the caller's real
        // question is about a file that does not exist yet.
        let target = nearest_existing(path).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no existing ancestor to measure", path.display()),
            )
        })?;
        // Unlike `statvfs`, this call wants a *directory*. When the walk
        // stopped on an existing file — asking about a source file's
        // volume is a perfectly reasonable question — step up to the
        // directory holding it.
        let target = if target.is_dir() {
            target
        } else {
            target.parent().unwrap_or(target)
        };
        let mut wide: Vec<u16> = target.as_os_str().encode_wide().collect();
        wide.push(0);
        let mut available: u64 = 0;
        // SAFETY: `wide` is NUL-terminated UTF-16; the two output
        // pointers we do not want are null, which the API permits.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut available,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        // The first output is already "bytes available to the calling
        // user", quota included — the same question `f_bavail` answers.
        Ok(available)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "free_space: no backend for this platform yet",
        ))
    }
}

/// Walk up from `path` to the first component that actually exists.
///
/// `statvfs` answers about a mounted filesystem but insists on being
/// handed a live path, while the caller's real question — "will the
/// output file fit?" — is by definition about a file that does not exist
/// yet. The directory it will be created in is on the same volume, so
/// the honest translation of the question is to ask about the nearest
/// existing ancestor.
///
/// Bounded by the path's own component count, so it always terminates.
#[cfg(any(unix, windows))]
fn nearest_existing(path: &Path) -> Option<&Path> {
    let mut candidate = Some(path);
    while let Some(p) = candidate {
        if p.exists() {
            return Some(p);
        }
        candidate = p.parent();
    }
    None
}

/// Whether the volume holding `path` can still take `needed` bytes,
/// keeping a safety margin so the encode does not fill the disk to the
/// last byte and strand the rest of the system.
///
/// # Errors
///
/// As [`free_space`].
pub fn has_room_for(path: &Path, needed: u64) -> io::Result<bool> {
    /// Never plan to consume the last slice of a volume.
    const RESERVE_BYTES: u64 = 256 * 1024 * 1024;
    let free = free_space(path)?;
    Ok(free >= needed.saturating_add(RESERVE_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_writable_volume_reports_a_plausible_amount_of_room() {
        let free = free_space(&std::env::temp_dir()).unwrap();
        assert!(free > 0, "temp dir reported {free} bytes free");
    }

    #[test]
    fn the_answer_is_about_the_volume_not_the_exact_file() {
        // Two paths on the same filesystem must agree, which is what lets
        // callers ask about an output file that does not exist yet.
        //
        // Compared with a tolerance for the same reason as the test below:
        // free space is live state, and any other process writing to the
        // volume between the two calls would otherwise fail this. An
        // exact-equality version did fail, reproducibly, once something
        // else was writing to the temp volume.
        let dir = std::env::temp_dir();
        let unborn = dir.join(format!("sovmr-space-{}.mp4", std::process::id()));
        let here = free_space(&dir).unwrap();
        let there = free_space(&unborn).unwrap();
        assert!(
            here.abs_diff(there) < here / 100,
            "both paths are on one volume: {here} vs {there}"
        );
    }

    #[test]
    fn a_path_under_an_existing_root_is_answered_by_that_volume() {
        // The output file does not exist yet — that is the normal case,
        // not an error. Reporting "0 bytes free" here would make the
        // engine refuse jobs that were perfectly runnable.
        //
        // Compared with a tolerance, not for equality: free space is
        // live state, and anything else on the machine writing a block
        // between the two calls would otherwise fail this test. An
        // earlier exact-equality version did exactly that.
        // Built from the temp dir rather than a hardcoded "/": the
        // question is "does the walk reach the right volume", and every
        // platform spells its roots differently.
        let base = std::env::temp_dir();
        let deep = base.join("sovmr-no-such-directory/nor-this/out.mp4");
        let answered = free_space(&deep).unwrap();
        let known = free_space(&base).unwrap();
        let drift = answered.abs_diff(known);
        assert!(
            drift < known / 100,
            "the deep path must be answered by its volume: {answered} vs {known}"
        );
    }

    /// A path carrying an interior NUL, built the way each platform can.
    ///
    /// Both C and Win32 take NUL-terminated strings, so on either system
    /// such a path would be silently truncated and answered about the
    /// wrong volume. That is the failure this guards against, and it has
    /// to be guarded on both.
    #[cfg(unix)]
    fn path_with_interior_nul() -> std::ffi::OsString {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        OsStr::from_bytes(b"/tmp/so\0vmr").to_owned()
    }
    #[cfg(windows)]
    fn path_with_interior_nul() -> std::ffi::OsString {
        use std::os::windows::ffi::OsStringExt;
        std::ffi::OsString::from_wide(&[
            'C' as u16,
            ':' as u16,
            '\\' as u16,
            's' as u16,
            'o' as u16,
            0,
            'v' as u16,
            'm' as u16,
            'r' as u16,
        ])
    }

    #[test]
    fn a_path_containing_a_nul_is_rejected_before_the_syscall() {
        let bad = path_with_interior_nul();
        let err = free_space(Path::new(&bad)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn an_absurd_requirement_is_refused_and_a_modest_one_allowed() {
        let dir = std::env::temp_dir();
        assert!(!has_room_for(&dir, u64::MAX / 2).unwrap());
        assert!(has_room_for(&dir, 1).unwrap());
    }

    #[test]
    fn the_reserve_margin_is_actually_enforced() {
        // Asking for exactly everything free must fail: the margin is the
        // difference between a finished encode and an unusable system.
        let dir = std::env::temp_dir();
        let free = free_space(&dir).unwrap();
        assert!(!has_room_for(&dir, free).unwrap());
    }
}
