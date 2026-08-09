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

//! Read-only whole-file memory mapping.
//!
//! # Why this type exists
//!
//! The demuxer is written against `&[u8]` sub-slices of the whole file,
//! which is the right shape — it copies nothing. But feeding it with
//! `std::fs::read` made the *caller* copy everything: a 12-hour source
//! is tens of gigabytes, and reading it into a `Vec` costs that much
//! resident memory before a single frame is decoded.
//!
//! A private read-only mapping gives the same `&[u8]` for free. The
//! kernel pages in what is actually touched and evicts it again under
//! pressure, so peak memory is governed by the working set (sample
//! tables plus the packets in flight), not by the file size.
//!
//! # Negative trust
//!
//! A mapping is a live window onto a file, not a snapshot. If another
//! process truncates the file while it is mapped, touching the vanished
//! pages raises `SIGBUS` — a fault no `Result` can express. The engine
//! therefore treats its inputs as it treats any other external party:
//! the source is expected to sit still for the duration of a job, and
//! callers that cannot assume that must copy instead.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

#[cfg(unix)]
use std::ffi::{c_int, c_void};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;

// Values confirmed against <sys/mman.h> on the target platform rather
// than copied from memory: PROT_READ 1, MAP_PRIVATE 2, MAP_FAILED -1,
// MADV_SEQUENTIAL 2.
#[cfg(unix)]
const PROT_READ: c_int = 1;
#[cfg(unix)]
const MAP_PRIVATE: c_int = 2;
#[cfg(unix)]
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
#[cfg(unix)]
const MADV_SEQUENTIAL: c_int = 2;

#[cfg(unix)]
extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
}

#[cfg(windows)]
use std::ffi::c_void;
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

/// Read-only page protection for the mapping object (`PAGE_READONLY`).
#[cfg(windows)]
const PAGE_READONLY: u32 = 0x02;
/// Read access for the view (`FILE_MAP_READ`).
#[cfg(windows)]
const FILE_MAP_READ: u32 = 0x0004;

#[cfg(windows)]
extern "system" {
    /// Create a file-mapping object. A zero maximum size means "as large
    /// as the file", which is exactly the whole-file mapping wanted here.
    fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *mut c_void,
        protect: u32,
        maximum_size_high: u32,
        maximum_size_low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        mapping: *mut c_void,
        desired_access: u32,
        offset_high: u32,
        offset_low: u32,
        bytes_to_map: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(base_address: *mut c_void) -> i32;
    fn CloseHandle(object: *mut c_void) -> i32;
}

/// A whole file mapped read-only into the address space, usable
/// anywhere a `&[u8]` is expected.
///
/// The mapping outlives the file descriptor that created it (POSIX
/// guarantees this), so no descriptor is held open for the hours a
/// long-form job can run.
///
/// # Example
///
/// ```
/// use sovmr_os::MappedFile;
///
/// # let path = std::env::temp_dir().join(format!("sovmr-map-doc-{}.bin", std::process::id()));
/// # std::fs::write(&path, b"ftypisom").unwrap();
/// let file = MappedFile::open(&path)?;
/// assert_eq!(&file[4..8], b"isom");
/// assert_eq!(file.len(), 8);
/// # std::fs::remove_file(&path).unwrap();
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct MappedFile {
    /// Start of the mapping, or a dangling (but aligned, non-null)
    /// pointer when `len == 0` — `mmap` rejects a zero length, and an
    /// empty file is not an error.
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is read-only and private; no interior mutability
// and no thread affinity. Handing the bytes to another thread is exactly
// as safe as handing over a `&[u8]`.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    /// Map `path` read-only in its entirety.
    ///
    /// # Errors
    ///
    /// Any I/O error opening or measuring the file, and the kernel's own
    /// diagnostic when the mapping itself is refused (out of address
    /// space, unmappable filesystem).
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        // 32-bit hosts cannot address a large file even though the API
        // can describe one: refuse loudly instead of truncating.
        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{}: file is larger than this architecture can map",
                    path.display()
                ),
            )
        })?;
        if len == 0 {
            return Ok(MappedFile {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        Self::map(&file, len)
    }

    #[cfg(unix)]
    fn map(file: &File, len: usize) -> io::Result<Self> {
        // SAFETY: a null hint lets the kernel choose the address; the fd
        // is open for reading; length is non-zero and offset zero, so the
        // whole file is covered.
        let addr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: mmap returned success, so addr is non-null and page
        // aligned (hence aligned for u8).
        let ptr = unsafe { NonNull::new_unchecked(addr.cast::<u8>()) };
        Ok(MappedFile { ptr, len })
    }

    /// Windows maps in two steps — a mapping *object* over the file, then
    /// a *view* of it in this process's address space.
    ///
    /// The object handle is closed as soon as the view exists: Win32
    /// keeps the view alive on its own, which reproduces the POSIX
    /// property this type is documented to have. A twelve-hour job holds
    /// no handle open on either system.
    #[cfg(windows)]
    fn map(file: &File, len: usize) -> io::Result<Self> {
        // SAFETY: the file is open for reading; a zero maximum size means
        // "the whole file", and a null name makes the object unnamed and
        // private to this process.
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle().cast::<c_void>(),
                std::ptr::null_mut(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mapping` is a live mapping object; offset zero and
        // `len` bytes cover exactly the file measured by the caller.
        let addr = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        // Close the object either way: on success the view keeps the
        // pages alive, and on failure it is simply no longer wanted.
        // Capture the error first — CloseHandle would overwrite it.
        let failure = addr.is_null().then(io::Error::last_os_error);
        // SAFETY: `mapping` came from a successful call and is closed once.
        unsafe { CloseHandle(mapping) };
        if let Some(e) = failure {
            return Err(e);
        }
        // SAFETY: MapViewOfFile succeeded, so addr is non-null and
        // allocation-granularity aligned (hence aligned for u8).
        let ptr = unsafe { NonNull::new_unchecked(addr.cast::<u8>()) };
        Ok(MappedFile { ptr, len })
    }

    /// Number of mapped bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Tell the kernel the mapping will be read front to back, so it
    /// reads ahead aggressively and releases pages already passed.
    ///
    /// This is a hint about *memory footprint* more than speed: streaming
    /// a 50 GiB source without it leaves the page cache holding every
    /// byte ever touched until something else needs the memory. Failure
    /// is deliberately ignored — an advice the kernel declines costs
    /// nothing but the advice.
    ///
    /// On Windows this is a no-op rather than a guess. The nearest
    /// equivalent (`PrefetchVirtualMemory`) prefetches rather than
    /// releases, which is not what the Unix arm asks for; and this
    /// function is currently called from nowhere, so inventing a second
    /// unmeasured behaviour would be optimising blind twice over.
    pub fn advise_sequential(&self) {
        #[cfg(unix)]
        if self.len > 0 {
            // SAFETY: the range is exactly our own live mapping.
            unsafe {
                madvise(
                    self.ptr.as_ptr().cast::<c_void>(),
                    self.len,
                    MADV_SEQUENTIAL,
                );
            }
        }
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: ptr is non-null and aligned in both arms (a real
        // mapping of `len` bytes, or dangling with len == 0, which
        // `from_raw_parts` explicitly permits). The mapping is read-only
        // and lives exactly as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The address is noise; the size is the fact worth printing.
        f.debug_struct("MappedFile")
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        if self.len > 0 {
            #[cfg(unix)]
            // SAFETY: unmapping exactly the range this type mapped, once.
            unsafe {
                munmap(self.ptr.as_ptr().cast::<c_void>(), self.len);
            }
            #[cfg(windows)]
            // SAFETY: releasing exactly the view this type mapped, once.
            // Win32 takes only the base address — it knows the length.
            unsafe {
                UnmapViewOfFile(self.ptr.as_ptr().cast::<c_void>());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique scratch path per test, following the workspace convention.
    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("sovmr-map-{}-{}.bin", tag, std::process::id()))
    }

    #[test]
    fn a_mapped_file_reads_back_exactly_what_was_written() {
        let path = scratch("roundtrip");
        let payload: Vec<u8> = (0u8..=255).cycle().take(300_000).collect();
        std::fs::write(&path, &payload).unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        assert_eq!(mapped.len(), payload.len());
        assert_eq!(&mapped[..], &payload[..], "mapping must be byte-identical");
        // Spanning several pages is the point: verify a late offset too.
        assert_eq!(mapped[299_999], payload[299_999]);

        drop(mapped);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn an_empty_file_maps_to_an_empty_slice_not_an_error() {
        // mmap(2) rejects a zero length, but a zero-byte input is a
        // legitimate (if useless) file: it must parse as "no bytes",
        // not as a syscall failure.
        let path = scratch("empty");
        std::fs::write(&path, b"").unwrap();

        let mapped = MappedFile::open(&path).unwrap();
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        assert_eq!(&mapped[..], b"");

        drop(mapped);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_missing_file_reports_the_os_error() {
        let path = scratch("definitely-absent").with_extension("nope");
        let err = MappedFile::open(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_directory_is_refused_rather_than_mapped() {
        // Negative trust: the caller may hand us anything a path can name.
        let err = MappedFile::open(&std::env::temp_dir()).unwrap_err();
        assert!(
            err.raw_os_error().is_some(),
            "expected an OS-level refusal, got {err:?}"
        );
    }

    #[test]
    fn advising_sequential_access_is_harmless_on_both_empty_and_real_files() {
        let path = scratch("advise");
        std::fs::write(&path, vec![7u8; 65_536]).unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        mapped.advise_sequential();
        assert_eq!(mapped[0], 7, "advice must not disturb the contents");
        drop(mapped);

        let empty = scratch("advise-empty");
        std::fs::write(&empty, b"").unwrap();
        let mapped = MappedFile::open(&empty).unwrap();
        mapped.advise_sequential(); // must not call madvise on a dangling ptr
        drop(mapped);

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&empty).unwrap();
    }

    #[test]
    fn the_mapping_survives_the_descriptor_that_created_it() {
        // A 12-hour job must not hold a file descriptor open for hours;
        // `open` drops the `File` before returning, so this is the
        // regression test for that guarantee.
        let path = scratch("fd-closed");
        std::fs::write(&path, b"still here").unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        assert_eq!(&mapped[..], b"still here");
        drop(mapped);
        std::fs::remove_file(&path).unwrap();
    }
}
