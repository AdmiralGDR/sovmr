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

//! # sovmr-os — the sovereign platform boundary
//!
//! Every syscall the project makes lives here. The rule this crate
//! exists to enforce is an *agnostic core*: business logic
//! must not know whether it runs on Linux, Windows, or in orbit. So the
//! layers above deal in [`MappedFile`] and [`free_space`], never in
//! `mmap` or `GetDiskFreeSpaceExW`, and porting means adding a backend
//! here rather than auditing the whole tree for `#[cfg]`.
//!
//! Each module exposes one portable type or function and hides one
//! platform's mechanism behind it:
//!
//! | Module | Linux | Windows (planned) |
//! |---|---|---|
//! | [`library`] | `dlopen`/`dlsym` | `LoadLibraryExW`/`GetProcAddress` |
//! | [`map`] | `mmap`/`munmap` | `CreateFileMappingW`/`MapViewOfFile` |
//! | [`signals`] | `sigaction` | `SetConsoleCtrlHandler` |
//! | [`space`] | `statvfs` | `GetDiskFreeSpaceExW` |
//!
//! # Example
//!
//! ```
//! use sovmr_os::MappedFile;
//!
//! # let path = std::env::temp_dir().join(format!("sovmr-doc-{}.bin", std::process::id()));
//! # std::fs::write(&path, b"sovereign").unwrap();
//! // A 50 GiB file costs no more resident memory than the pages actually read.
//! let file = MappedFile::open(&path)?;
//! assert_eq!(&file[..9], b"sovereign");
//! # std::fs::remove_file(&path).unwrap();
//! # Ok::<(), std::io::Error>(())
//! ```

#![deny(missing_docs)]

pub mod desktop;
pub mod library;
pub mod map;
pub mod path;
pub mod signals;
pub mod space;

pub use library::Library;
pub use map::MappedFile;
pub use path::os_string_from_bytes;
pub use space::{free_space, has_room_for};
