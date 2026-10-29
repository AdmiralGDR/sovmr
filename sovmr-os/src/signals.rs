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

//! Graceful termination: SIGINT/SIGTERM on Unix, and the console
//! control events that stand in for them on Windows.
//!
//! Hand-rolled `sigaction` FFI against the glibc that `std` already
//! links — no `ctrlc`, no `libc` crate.
//!
//! Contract: the first signal flips a process-wide flag that the
//! embedder polls to run a *graceful* teardown (cancel encodes → roll
//! back `.part` files → release the sleep inhibitor → close the
//! window). A **second** signal means the user insists: the handler
//! `_exit`s immediately with the conventional `128 + signum` status.
//! The handler itself is async-signal-safe by construction — one atomic
//! swap, one optional `_exit`, nothing else.
//!
//! Layout note: `struct sigaction` here mirrors glibc on Linux x86_64 —
//! pinned by a C probe (`sizeof = 152`, `sa_mask` at offset 8 spanning
//! 128 bytes, `sa_flags` at 136, `sa_restorer` at 144) and re-asserted
//! in the unit tests below, the same methodology that pinned the
//! NVENC/NVDEC FFI structs.

#[cfg(windows)]
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

/// `SIGINT` (Ctrl+C on the controlling terminal).
pub const SIGINT: i32 = 2;
/// `SIGPIPE` (write to a closed pipe, e.g. `sovmr --report | head`).
pub const SIGPIPE: i32 = 13;
/// `SIGTERM` (polite kill, the desktop/systemd default).
pub const SIGTERM: i32 = 15;
/// `SIG_DFL`: the kernel's default disposition.
#[cfg(unix)]
const SIG_DFL: usize = 0;
/// Restart interrupted syscalls instead of surfacing `EINTR` everywhere.
#[cfg(unix)]
const SA_RESTART: i32 = 0x1000_0000;
/// glibc `sigset_t` is 1024 bits = 16 × u64 words.
#[cfg(unix)]
const SIGSET_WORDS: usize = 16;
/// Conventional exit-status base for "killed by signal N".
const SIGNAL_EXIT_BASE: i32 = 128;

/// Set once by the handler; polled by the embedder's watcher thread.
static TERMINATION_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
extern "system" {
    fn SetConsoleCtrlHandler(
        handler: Option<unsafe extern "system" fn(u32) -> i32>,
        add: i32,
    ) -> i32;
    /// Immediate termination without unwinding — the Windows counterpart
    /// of `_exit`, and the same reason for using it: the *second*
    /// interrupt must not wait on anything.
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
}

/// Console control events that mean "the user wants this to stop".
///
/// `CTRL_C_EVENT` and `CTRL_BREAK_EVENT` are the console equivalents of
/// SIGINT. The other three arrive when the console window is closed or
/// the user logs off or shuts down; Windows gives only a few seconds
/// before killing the process anyway, but flipping the flag still lets
/// the engine roll back a `.part` file in that window rather than
/// leaving a corrupt output behind.
#[cfg(windows)]
const CTRL_EVENTS: [u32; 5] = [0, 1, 2, 5, 6];

/// The installed console handler. Runs on a thread Windows creates for
/// the purpose, so like the Unix handler it does one atomic swap and
/// nothing else.
#[cfg(windows)]
unsafe extern "system" fn on_console_ctrl(event: u32) -> i32 {
    if !CTRL_EVENTS.contains(&event) {
        return 0; // not ours: let the default handling run
    }
    if TERMINATION_REQUESTED.swap(true, Ordering::SeqCst) {
        // Second interrupt: the user insists. Graceful teardown is
        // already running or wedged — leave immediately.
        TerminateProcess(GetCurrentProcess(), (SIGNAL_EXIT_BASE + SIGINT) as u32);
    }
    1 // handled: do not run the default terminator
}

/// glibc's `struct sigaction` (Linux x86_64): handler union, 1024-bit
/// mask, flags, restorer. The handler slot is a `usize` so it can also
/// carry the `SIG_DFL` sentinel, exactly as C's union does.
#[cfg(unix)]
#[repr(C)]
struct SigAction {
    handler: usize,
    mask: [u64; SIGSET_WORDS],
    flags: i32,
    restorer: usize,
}

#[cfg(unix)]
extern "C" {
    /// POSIX `sigaction(2)` from the glibc `std` already links.
    fn sigaction(signum: i32, act: *const SigAction, oldact: *mut SigAction) -> i32;
    /// Async-signal-safe immediate exit (no atexit, no Drop — that is
    /// the point: the *second* signal must not wait for anything).
    fn _exit(status: i32) -> !;
}

/// The installed handler. Async-signal-safe: one atomic op; on repeat
/// delivery, `_exit` (also async-signal-safe per POSIX).
#[cfg(unix)]
extern "C" fn on_termination_signal(signum: i32) {
    if TERMINATION_REQUESTED.swap(true, Ordering::SeqCst) {
        // Second Ctrl+C / SIGTERM: the user insists. Graceful teardown
        // is already running or wedged — leave immediately.
        // SAFETY: `_exit` is async-signal-safe (POSIX.1-2008 list), takes
        // a plain int, and never returns, so calling it from a signal
        // handler cannot observe or corrupt interrupted state.
        unsafe { _exit(SIGNAL_EXIT_BASE + signum) }
    }
}

/// Install the graceful-termination handler for `SIGINT` and `SIGTERM`.
///
/// # Errors
///
/// The signal number that the kernel rejected (it never does for these
/// two on Linux, but negative trust applies to syscalls too).
pub fn install() -> Result<(), i32> {
    #[cfg(unix)]
    {
        for signum in [SIGINT, SIGTERM] {
            set_handler(signum, on_termination_signal as *const () as usize)?;
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        // SAFETY: a valid handler address and TRUE to add rather than
        // remove it. Windows calls it on a thread of its own making,
        // which is why the handler touches nothing but one atomic.
        if unsafe { SetConsoleCtrlHandler(Some(on_console_ctrl), 1) } == 0 {
            return Err(SIGINT);
        }
        Ok(())
    }
}

/// Restore the Unix default for `SIGPIPE`: Rust's runtime ignores it,
/// which turns `sovmr --report | head` into a "failed printing to
/// stdout" panic. A CLI must die silently when its pipe reader leaves —
/// like every other Unix tool.
///
/// # Errors
///
/// The rejected signal number (never happens for SIGPIPE on Linux).
pub fn default_sigpipe() -> Result<(), i32> {
    #[cfg(unix)]
    {
        set_handler(SIGPIPE, SIG_DFL)
    }
    #[cfg(windows)]
    {
        // Nothing to restore: Windows has no SIGPIPE. A write to a
        // closed pipe fails with ERROR_BROKEN_PIPE, which `std` already
        // surfaces as an ordinary `io::Error`, so `sovmr --report |
        // more` ends quietly without any help from us.
        Ok(())
    }
}

#[cfg(unix)]
fn set_handler(signum: i32, handler: usize) -> Result<(), i32> {
    let action = SigAction {
        handler,
        mask: [0; SIGSET_WORDS], // block nothing extra during delivery
        flags: SA_RESTART,
        restorer: 0,
    };
    // SAFETY: `action` is a fully initialized glibc-layout struct
    // (layout pinned by the tests below); a null oldact is allowed.
    if unsafe { sigaction(signum, &action, std::ptr::null_mut()) } != 0 {
        return Err(signum);
    }
    Ok(())
}

/// `true` once a termination signal has been delivered. The embedder's
/// watcher polls this and drives the graceful teardown.
pub fn termination_requested() -> bool {
    TERMINATION_REQUESTED.load(Ordering::SeqCst)
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;

    extern "C" {
        /// POSIX `raise(3)`: deliver a signal to the calling thread.
        fn raise(signum: i32) -> i32;
    }

    #[test]
    fn layout_matches_the_c_probe() {
        // Pinned by the C probe on this host: glibc x86_64.
        assert_eq!(std::mem::size_of::<SigAction>(), 152);
        assert_eq!(std::mem::offset_of!(SigAction, handler), 0);
        assert_eq!(std::mem::offset_of!(SigAction, mask), 8);
        assert_eq!(std::mem::offset_of!(SigAction, flags), 136);
        assert_eq!(std::mem::offset_of!(SigAction, restorer), 144);
    }

    #[test]
    fn first_signal_sets_the_flag_instead_of_killing_the_process() {
        // Install first: with the default disposition this SIGTERM
        // would kill the test harness — surviving it IS the assertion.
        install().expect("sigaction accepted");
        assert!(!termination_requested(), "clean slate before delivery");
        // SAFETY: raise() with our no-op-ish handler installed.
        assert_eq!(unsafe { raise(SIGTERM) }, 0);
        assert!(
            termination_requested(),
            "handler must flip the flag on delivery"
        );
        // NOTE: the second-signal _exit path is deliberately untested
        // in-process (it would take the harness down); its body is one
        // POSIX-guaranteed call.
        TERMINATION_REQUESTED.store(false, Ordering::SeqCst);
    }
}
