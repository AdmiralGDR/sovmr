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

//! The desktop a running program sits on: where it may write, and how it
//! asks the machine to stay awake.
//!
//! # Why these live here
//!
//! Business logic has no right to know where it is running, and
//! every system call is injected at the outermost layer. `sovmr-os` is
//! that layer — the only crate that makes them.
//!
//! Until now the application called `systemd-inhibit` and read
//! `$XDG_STATE_HOME` directly. Both compile on Windows and both do
//! nothing there: the binary is absent and the variable unset. A twelve
//! hour encode could put the machine to sleep, and nothing was written
//! to a log anyone could find.
//!
//! **Silence is the defect.** A backend that is missing must say so; a
//! platform without one must be told apart from a platform where it
//! failed.
//!
//! # One interface, not one implementation per caller
//!
//! Each concern is one function whose signature says nothing about the
//! platform. The arms differ; the callers do not see that they do.

use std::path::PathBuf;

/// Where this program may keep logs and other state it owns.
///
/// Returns `None` when the platform's own answer is unavailable, which
/// is a real possibility on both: a service account with no `HOME`, or a
/// Windows profile without `LOCALAPPDATA`. Inventing a path — the
/// current directory, or `/tmp` — would scatter state somewhere nobody
/// would look for it.
///
/// # Platform
///
/// * Unix: `$XDG_STATE_HOME`, else `$HOME/.local/state`, per the XDG
///   base directory specification.
/// * Windows: `%LOCALAPPDATA%`, which is the roaming-exempt per-user
///   store — a log has no business following a user between machines.
pub fn state_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        /// Where the XDG specification says state goes when unset.
        const XDG_STATE_DEFAULT: &str = ".local/state";
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(XDG_STATE_DEFAULT)))
    }
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

/// Subdirectory of the state directory that holds this program's own
/// small preferences.
const PREFERENCES_SUBDIR: &str = "sovmr";

/// Resolve a preference file path, refusing a name that is not a plain
/// key.
///
/// Negative trust: a preference name is a key, never a path. A
/// value like `../../.ssh/authorized_keys` arriving here must not become
/// a real path — so anything but `[A-Za-z0-9_-]` is refused outright,
/// before the state directory is even consulted.
fn preference_path(name: &str) -> Option<PathBuf> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return None;
    }
    Some(state_dir()?.join(PREFERENCES_SUBDIR).join(name))
}

/// Read a small named preference this program persisted for itself.
///
/// `None` means the preference was never set, the state directory is
/// unavailable, or the name is not a plain key — every case where the
/// caller's own default is the right answer. Reading is allowed to be
/// lossy this way because a missing preference is ordinary; *writing* one
/// is not, and [`write_preference`] reports its failures instead.
pub fn read_preference(name: &str) -> Option<String> {
    let path = preference_path(name)?;
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Persist a small named preference for the next run.
///
/// # Errors
///
/// [`std::io::Error`] when the name is not a plain key, the state
/// directory is unavailable, or the write itself fails — a lost
/// preference is *said*, not swallowed, so the caller can log it rather
/// than silently forget what the user chose.
pub fn write_preference(name: &str, value: &str) -> std::io::Result<()> {
    let path = preference_path(name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "preference name is not a plain key, or no state directory exists",
        )
    })?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, value)
}

/// A promise to the machine that it should stay awake.
///
/// Held for as long as the value lives. Dropping it releases the
/// promise, which is the whole reason it is a value and not a pair of
/// calls: an encode that panics must not leave a machine unable to
/// sleep.
#[derive(Debug)]
pub struct StayAwake {
    inner: Inner,
}

/// Why a machine could not be asked to stay awake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwakeError {
    /// This platform has no implementation here.
    ///
    /// Distinct from a failure: nothing went wrong, there is simply
    /// nothing to call, and a user should be told which.
    Unsupported,
    /// The platform's own mechanism refused or is absent.
    Refused(String),
}

impl std::fmt::Display for AwakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwakeError::Unsupported => write!(f, "not supported on this platform"),
            AwakeError::Refused(why) => write!(f, "{why}"),
        }
    }
}

/// Ask the machine to stay awake until the returned value is dropped.
///
/// `reason` is shown to the user by systems that display one.
///
/// # Errors
///
/// [`AwakeError::Unsupported`] where there is nothing to call, and
/// [`AwakeError::Refused`] where the call was made and did not work.
/// The two are told apart because the answers differ: one is a gap in
/// this program, the other a problem on the machine.
pub fn stay_awake(reason: &str) -> Result<StayAwake, AwakeError> {
    Inner::acquire(reason).map(|inner| StayAwake { inner })
}

#[cfg(unix)]
mod platform {
    use super::AwakeError;
    use std::process::{Child, Command, Stdio};

    /// logind's own tool. It holds the inhibitor for as long as the
    /// child it wraps lives, so releasing is killing the child.
    const INHIBIT_BIN: &str = "systemd-inhibit";

    #[derive(Debug)]
    pub(super) struct Inner {
        child: Option<Child>,
    }

    impl Inner {
        pub(super) fn acquire(reason: &str) -> Result<Inner, AwakeError> {
            let child = Command::new(INHIBIT_BIN)
                .args([
                    "--what=idle:sleep",
                    "--who=sovmr",
                    &format!("--why={reason}"),
                    "--mode=block",
                    // Something that will not exit on its own: the
                    // inhibitor lasts exactly as long as this does.
                    "sleep",
                    "infinity",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| AwakeError::Refused(format!("{INHIBIT_BIN}: {e}")))?;
            Ok(Inner { child: Some(child) })
        }

        pub(super) fn release(&mut self) {
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                // Reaped, never left a zombie.
                let _ = child.wait();
            }
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::AwakeError;

    /// `EXECUTION_STATE SetThreadExecutionState(EXECUTION_STATE)`, from
    /// Kernel32. Verified against the Win32 documentation rather than
    /// remembered.
    type SetThreadExecutionStateFn = unsafe extern "system" fn(u32) -> u32;

    /// The state persists until cleared, rather than merely resetting
    /// the idle timer once.
    const ES_CONTINUOUS: u32 = 0x8000_0000;
    /// Keep the system out of sleep.
    const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;

    /// The call is **per thread**, so the promise lives exactly as long
    /// as the thread that made it. This holds one for its own lifetime
    /// and clears it on drop, on that same thread.
    #[derive(Debug)]
    pub(super) struct Inner {
        held: bool,
    }

    /// Ask Kernel32 for the entry point, once per call.
    ///
    /// Cheap — the library is already loaded into every process — and it
    /// keeps this crate's rule that nothing is linked at build time.
    fn entry() -> Result<SetThreadExecutionStateFn, AwakeError> {
        let lib = crate::library::Library::open("kernel32.dll")
            .map_err(|e| AwakeError::Refused(format!("kernel32: {e}")))?;
        let symbol = lib
            .symbol("SetThreadExecutionState")
            .map_err(|e| AwakeError::Refused(format!("SetThreadExecutionState: {e}")))?;
        // SAFETY: the symbol is the documented Kernel32 export and the
        // signature matches winbase.h.
        let f: SetThreadExecutionStateFn = unsafe { std::mem::transmute(symbol) };
        // The library must outlive the pointer; Kernel32 is never
        // unloaded from a running process, and leaking the handle is
        // cheaper and safer than keeping one alive per promise.
        std::mem::forget(lib);
        Ok(f)
    }

    impl Inner {
        pub(super) fn acquire(reason: &str) -> Result<Inner, AwakeError> {
            // Windows takes no reason; the argument is part of one
            // interface serving both platforms.
            let _ = reason;
            let set = entry()?;
            // SAFETY: documented signature, plain integer argument.
            let previous = unsafe { set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) };
            if previous == 0 {
                return Err(AwakeError::Refused(
                    "SetThreadExecutionState refused the request".into(),
                ));
            }
            Ok(Inner { held: true })
        }

        pub(super) fn release(&mut self) {
            if !self.held {
                return;
            }
            self.held = false;
            if let Ok(set) = entry() {
                // ES_CONTINUOUS alone clears every other flag, which is
                // how the documentation says to let the machine sleep
                // normally again.
                // SAFETY: as above.
                unsafe { set(ES_CONTINUOUS) };
            }
        }
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    use super::AwakeError;

    #[derive(Debug)]
    pub(super) struct Inner;

    impl Inner {
        pub(super) fn acquire(_reason: &str) -> Result<Inner, AwakeError> {
            Err(AwakeError::Unsupported)
        }
        pub(super) fn release(&mut self) {}
    }
}

use platform::Inner;

impl Drop for StayAwake {
    /// The promise must never outlive the work. An orphaned inhibitor
    /// leaves a machine unable to sleep with nothing on screen to
    /// explain why.
    fn drop(&mut self) {
        self.inner.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialises every test that reads or writes the process-global
    /// state-directory environment variable. Without it the default
    /// parallel harness lets one test observe another's temporary value
    /// (a test must be the same test twice). Poison-tolerant so
    /// a panic in one env test does not cascade into the rest.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The env var each platform roots its state directory in.
    fn state_key() -> &'static str {
        if cfg!(windows) {
            "LOCALAPPDATA"
        } else {
            "XDG_STATE_HOME"
        }
    }

    #[test]
    fn the_state_directory_is_absolute_or_absent() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // A relative path would put state wherever the program happened
        // to be started from — findable by nobody, including the person
        // who has to read the log.
        if let Some(dir) = state_dir() {
            assert!(dir.is_absolute(), "{}", dir.display());
        }
    }

    #[test]
    fn a_relative_override_is_refused_rather_than_used() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // A `XDG_STATE_HOME` of `logs` is a configuration mistake, and
        // honouring it scatters state relative to the working directory.
        let key = state_key();
        let restore = std::env::var_os(key);
        std::env::set_var(key, "relative/path");
        let got = state_dir();
        match restore {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        assert!(
            got.as_ref().is_none_or(|p| p.is_absolute()),
            "a relative override was used: {got:?}"
        );
    }

    #[test]
    fn a_preference_round_trips_and_a_hostile_name_is_refused() {
        // A name that is not a plain key is refused *before* the state
        // directory is consulted, so this half needs no environment and
        // is fully deterministic — the security-critical part.
        assert!(
            write_preference("../escape", "x").is_err(),
            "a traversal name must not become a real path"
        );
        assert!(read_preference("../escape").is_none());
        assert!(
            write_preference("", "x").is_err(),
            "an empty key is refused"
        );

        // The round trip needs a real (temporary) state directory.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let key = state_key();
        let restore = std::env::var_os(key);
        let tmp = std::env::temp_dir().join(format!("sovmr-pref-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var(key, &tmp);

        let outcome = (|| {
            assert!(read_preference("language").is_none(), "nothing saved yet");
            write_preference("language", "ru")?;
            assert_eq!(read_preference("language").as_deref(), Some("ru"));
            // Overwrite, and confirm the new value wins.
            write_preference("language", "en")?;
            assert_eq!(read_preference("language").as_deref(), Some("en"));
            std::io::Result::Ok(())
        })();

        match restore {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        let _ = std::fs::remove_dir_all(&tmp);
        outcome.expect("preference I/O against a temp state dir");
    }

    #[test]
    fn an_unsupported_platform_says_so_rather_than_failing_quietly() {
        // The distinction the whole error type exists for: nothing to
        // call is not the same as something refusing, and the two have
        // different answers.
        assert_eq!(
            AwakeError::Unsupported.to_string(),
            "not supported on this platform"
        );
        assert_eq!(
            AwakeError::Refused("logind absent".into()).to_string(),
            "logind absent"
        );
        assert_ne!(AwakeError::Unsupported, AwakeError::Refused(String::new()));
    }
}

/// Why a file could not be chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickError {
    /// The user closed the dialog without choosing.
    Cancelled,
    /// No dialog exists to show.
    ///
    /// On Linux that means neither `zenity` nor `kdialog` is installed;
    /// there is no Win32 equivalent of "absent", since the dialog is
    /// part of the system. Either way it is said rather than shown as a
    /// button that does nothing.
    Unavailable(String),
}

impl std::fmt::Display for PickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PickError::Cancelled => write!(f, "cancelled"),
            PickError::Unavailable(why) => write!(f, "{why}"),
        }
    }
}

/// Ask the user to choose media files.
///
/// Blocking: the dialog owns the thread until it closes, so a caller
/// with a window to keep responsive runs this elsewhere.
///
/// # Platform
///
/// * Unix: the desktop's own chooser, `zenity` then `kdialog`.
/// * Windows: `GetOpenFileNameW`, which is the system's, not a program
///   that may or may not be installed.
///
/// # Errors
///
/// [`PickError::Cancelled`] when the user declined, and
/// [`PickError::Unavailable`] when there was nothing to show — the two
/// mean different things to whoever pressed the button.
pub fn pick_media_files(title: &str) -> Result<Vec<PathBuf>, PickError> {
    picker::pick(title)
}

#[cfg(windows)]
mod picker {
    use super::{PathBuf, PickError};

    /// `BOOL GetOpenFileNameW(LPOPENFILENAMEW)`, from comdlg32.
    type GetOpenFileNameWFn = unsafe extern "system" fn(*mut u8) -> i32;

    /// Probed with the cross-compiler against `commdlg.h`, never counted
    /// by hand: `sizeof(OPENFILENAMEW)` is 152 on this target and every
    /// offset below came from the same probe.
    const OFN_SIZE: usize = 152;
    const OFF_FILTER: usize = 24;
    const OFF_FILTER_INDEX: usize = 44;
    const OFF_FILE: usize = 48;
    const OFF_MAX_FILE: usize = 56;
    const OFF_TITLE: usize = 88;
    const OFF_FLAGS: usize = 96;

    /// Explorer-style dialog, several files at once, and every one must
    /// exist. `OFN_NOCHANGEDIR` keeps the dialog from moving the
    /// process's working directory out from under everything else.
    const FLAGS: u32 = 0x0008_0000 | 0x0000_0200 | 0x0000_1000 | 0x0000_0800 | 0x0000_0008;

    /// Room for a multi-selection: a directory plus many names, each
    /// null-terminated, the whole double-null-terminated.
    const BUFFER_CHARS: usize = 32 * 1024;

    /// A null-terminated UTF-16 string.
    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub(super) fn pick(title: &str) -> Result<Vec<PathBuf>, PickError> {
        let lib = crate::library::Library::open("comdlg32.dll")
            .map_err(|e| PickError::Unavailable(format!("comdlg32: {e}")))?;
        let symbol = lib
            .symbol("GetOpenFileNameW")
            .map_err(|e| PickError::Unavailable(format!("GetOpenFileNameW: {e}")))?;
        // SAFETY: the documented comdlg32 export, signature per commdlg.h.
        let open: GetOpenFileNameWFn = unsafe { std::mem::transmute(symbol) };

        // Pairs of null-terminated strings, the list double-terminated.
        let mut filter: Vec<u16> = Vec::new();
        for part in ["Media files", "*.mp4;*.mkv;*.mov;*.m4v", "All files", "*.*"] {
            filter.extend(wide(part));
        }
        filter.push(0);
        let title = wide(title);
        let mut file = vec![0u16; BUFFER_CHARS];

        let mut ofn = [0u8; OFN_SIZE];
        let put_usize = |ofn: &mut [u8; OFN_SIZE], at: usize, v: u64| {
            ofn[at..at + 8].copy_from_slice(&v.to_le_bytes());
        };
        let put_u32 = |ofn: &mut [u8; OFN_SIZE], at: usize, v: u32| {
            ofn[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        put_u32(&mut ofn, 0, OFN_SIZE as u32);
        put_usize(&mut ofn, OFF_FILTER, filter.as_ptr() as u64);
        put_u32(&mut ofn, OFF_FILTER_INDEX, 1);
        put_usize(&mut ofn, OFF_FILE, file.as_mut_ptr() as u64);
        put_u32(&mut ofn, OFF_MAX_FILE, BUFFER_CHARS as u32);
        put_usize(&mut ofn, OFF_TITLE, title.as_ptr() as u64);
        put_u32(&mut ofn, OFF_FLAGS, FLAGS);

        // SAFETY: the structure is sized and laid out per the probe, and
        // every pointer in it outlives the call.
        let chose = unsafe { open(ofn.as_mut_ptr()) };
        if chose == 0 {
            // Zero is also returned when the user simply closed the
            // dialog, which is not a failure worth a message.
            return Err(PickError::Cancelled);
        }
        Ok(parse_selection(&file))
    }

    /// Read the dialog's answer.
    ///
    /// One file comes back as a single path. Several come back as the
    /// directory, then each bare name, every string null-terminated and
    /// the list ended by an empty one — a format that looks like a
    /// single path right up until somebody selects two files.
    pub(super) fn parse_selection(buffer: &[u16]) -> Vec<PathBuf> {
        let mut parts: Vec<String> = Vec::new();
        let mut start = 0usize;
        for (i, unit) in buffer.iter().enumerate() {
            if *unit == 0 {
                if i == start {
                    break;
                }
                parts.push(String::from_utf16_lossy(&buffer[start..i]));
                start = i + 1;
            }
        }
        match parts.len() {
            0 => Vec::new(),
            1 => vec![PathBuf::from(&parts[0])],
            _ => {
                let dir = PathBuf::from(&parts[0]);
                parts[1..].iter().map(|name| dir.join(name)).collect()
            }
        }
    }
}

#[cfg(unix)]
mod picker {
    use super::{PathBuf, PickError};
    use std::process::Command;

    /// Exit code both tools use for "the user said no".
    const CANCELLED: i32 = 1;

    pub(super) fn pick(title: &str) -> Result<Vec<PathBuf>, PickError> {
        let mut refusals = Vec::new();
        for (bin, args) in [
            (
                "zenity",
                vec![
                    "--file-selection".to_string(),
                    "--multiple".to_string(),
                    // Newline, so a path containing the default `|`
                    // separator survives.
                    "--separator=\n".to_string(),
                    format!("--title={title}"),
                ],
            ),
            (
                "kdialog",
                vec![
                    "--getopenfilename".to_string(),
                    ".".to_string(),
                    "--multiple".to_string(),
                    "--separate-output".to_string(),
                    "--title".to_string(),
                    title.to_string(),
                ],
            ),
        ] {
            match Command::new(bin).args(&args).output() {
                Err(e) => refusals.push(format!("{bin}: {e}")),
                Ok(out) if out.status.code() == Some(CANCELLED) => {
                    return Err(PickError::Cancelled)
                }
                Ok(out) if !out.status.success() => {
                    refusals.push(format!("{bin}: exit {:?}", out.status.code()));
                }
                Ok(out) => {
                    return Ok(parse_selection(&String::from_utf8_lossy(&out.stdout)));
                }
            }
        }
        Err(PickError::Unavailable(format!(
            "no file chooser available ({})",
            refusals.join("; ")
        )))
    }

    /// One path per line, blanks ignored.
    pub(super) fn parse_selection(text: &str) -> Vec<PathBuf> {
        text.lines()
            .map(|l| l.trim_end_matches('\r'))
            .filter(|l| !l.is_empty())
            .map(PathBuf::from)
            .collect()
    }
}

#[cfg(not(any(unix, windows)))]
mod picker {
    use super::{PathBuf, PickError};
    pub(super) fn pick(_title: &str) -> Result<Vec<PathBuf>, PickError> {
        Err(PickError::Unavailable(
            "no file chooser on this platform".into(),
        ))
    }
}

#[cfg(test)]
mod picker_tests {
    use super::*;

    #[test]
    fn cancelling_is_told_apart_from_having_nothing_to_show() {
        // Different answers: one is the user's choice, the other is a
        // machine missing a chooser. A button that did nothing said
        // neither, which is what this exists to end.
        assert_eq!(PickError::Cancelled.to_string(), "cancelled");
        assert_eq!(
            PickError::Unavailable("zenity absent".into()).to_string(),
            "zenity absent"
        );
        assert_ne!(PickError::Cancelled, PickError::Unavailable(String::new()));
    }

    #[cfg(unix)]
    #[test]
    fn a_path_containing_the_default_separator_survives() {
        // zenity separates with `|` unless told otherwise, and `|` is a
        // legal character in a Unix filename. Newlines are asked for,
        // and this is what makes that matter.
        let chosen = picker::parse_selection("/media/a|b.mp4\n/media/c.mkv\n");
        assert_eq!(
            chosen,
            vec![
                PathBuf::from("/media/a|b.mp4"),
                PathBuf::from("/media/c.mkv")
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn blank_lines_and_carriage_returns_are_not_paths() {
        let chosen = picker::parse_selection("\n/media/a.mp4\r\n\n");
        assert_eq!(chosen, vec![PathBuf::from("/media/a.mp4")]);
    }

    #[cfg(windows)]
    #[test]
    fn one_file_and_many_files_come_back_differently() {
        // The Win32 answer is a single path when one file was chosen,
        // and a directory followed by bare names when several were —
        // a format that looks like a single path right up until
        // somebody selects two.
        let one: Vec<u16> = "C:\\media\\a.mp4\0\0".encode_utf16().collect();
        assert_eq!(
            picker::parse_selection(&one),
            vec![PathBuf::from("C:\\media\\a.mp4")]
        );
        let many: Vec<u16> = "C:\\media\0a.mp4\0b.mkv\0\0".encode_utf16().collect();
        assert_eq!(
            picker::parse_selection(&many),
            vec![
                PathBuf::from("C:\\media\\a.mp4"),
                PathBuf::from("C:\\media\\b.mkv")
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn an_empty_answer_is_no_files_rather_than_one_empty_path() {
        assert!(picker::parse_selection(&[0u16, 0]).is_empty());
        assert!(picker::parse_selection(&[]).is_empty());
    }
}

/// Tell the user something finished, when they may not be looking.
///
/// Best effort by design: a notification that cannot be shown is not a
/// reason to fail a job that succeeded. The return says whether it was
/// delivered, so a caller can fall back to its own window rather than
/// assume.
///
/// # Platform
///
/// * Unix: `notify-send`, the desktop's own service.
/// * Windows: a balloon from the notification area, via
///   `Shell_NotifyIconW` — flat Win32, no WinRT and no COM.
pub fn notify(title: &str, body: &str) -> bool {
    notifier::notify(title, body)
}

#[cfg(unix)]
mod notifier {
    use std::process::{Command, Stdio};

    /// From the system theme; nothing is bundled.
    const ICON: &str = "video-x-generic";
    /// Dismiss on its own; a job notice should not need clearing.
    const TIMEOUT_MS: u32 = 8000;

    pub(super) fn notify(title: &str, body: &str) -> bool {
        Command::new("notify-send")
            .args(["--app-name=sovmr", "--urgency=normal"])
            .arg(format!("--icon={ICON}"))
            .arg(format!("--expire-time={TIMEOUT_MS}"))
            .arg(title)
            .arg(body)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .is_ok()
    }
}

#[cfg(windows)]
mod notifier {
    /// `HWND CreateWindowExW(...)` — twelve arguments, from user32.
    type CreateWindowExWFn = unsafe extern "system" fn(
        u32,
        *const u16,
        *const u16,
        u32,
        i32,
        i32,
        i32,
        i32,
        isize,
        isize,
        isize,
        *mut core::ffi::c_void,
    ) -> isize;
    /// `BOOL DestroyWindow(HWND)`.
    type DestroyWindowFn = unsafe extern "system" fn(isize) -> i32;
    /// `BOOL Shell_NotifyIconW(DWORD, PNOTIFYICONDATAW)`.
    type ShellNotifyIconWFn = unsafe extern "system" fn(u32, *mut u8) -> i32;

    /// Parent that makes a window message-only: it never draws, never
    /// appears, and costs nothing — but it is a valid owner, which is
    /// all the notification area asks for.
    const HWND_MESSAGE: isize = -3;

    /// `NOTIFYICONDATAW`, probed with the cross-compiler against
    /// `shellapi.h` rather than counted by hand.
    const NID_SIZE: usize = 976;
    const OFF_HWND: usize = 8;
    const OFF_UID: usize = 16;
    const OFF_FLAGS: usize = 20;
    const OFF_TIP: usize = 40;
    const OFF_INFO: usize = 304;
    const OFF_INFO_TITLE: usize = 820;
    const OFF_INFO_FLAGS: usize = 948;

    /// From the same probe: carry a balloon, carry a tooltip, and give
    /// the balloon the informational glyph.
    const NIF_TIP: u32 = 0x0004;
    const NIF_INFO: u32 = 0x0010;
    const NIIF_INFO: u32 = 0x0001;
    const NIM_ADD: u32 = 0x0000;
    const NIM_DELETE: u32 = 0x0002;

    /// Our icon's identifier within our own window. Any value serves;
    /// it only has to be the same for add and delete.
    const ICON_ID: u32 = 1;

    /// Write a null-terminated UTF-16 string into a fixed field,
    /// truncated to fit and always terminated.
    pub(super) fn put_wide(buffer: &mut [u8], at: usize, chars: usize, text: &str) {
        let mut units: Vec<u16> = text.encode_utf16().take(chars - 1).collect();
        units.push(0);
        for (i, unit) in units.iter().enumerate() {
            let byte = at + i * 2;
            buffer[byte..byte + 2].copy_from_slice(&unit.to_le_bytes());
        }
    }

    fn symbol<T>(library: &'static str, name: &str) -> Option<T> {
        let lib = crate::library::Library::open(library).ok()?;
        let raw = lib.symbol(name).ok()?;
        // The pointer must outlive the handle; user32 and shell32 are
        // never unloaded from a running process.
        std::mem::forget(lib);
        // SAFETY: the caller names a documented export and its signature.
        Some(unsafe { std::mem::transmute_copy::<*mut core::ffi::c_void, T>(&raw) })
    }

    pub(super) fn notify(title: &str, body: &str) -> bool {
        let Some(create) = symbol::<CreateWindowExWFn>("user32.dll", "CreateWindowExW") else {
            return false;
        };
        let Some(destroy) = symbol::<DestroyWindowFn>("user32.dll", "DestroyWindow") else {
            return false;
        };
        let Some(shell_notify) = symbol::<ShellNotifyIconWFn>("shell32.dll", "Shell_NotifyIconW")
        else {
            return false;
        };

        // A built-in class, so nothing has to be registered and nothing
        // has to be unregistered. The window is message-only: invisible,
        // undrawn, and gone again in a moment.
        let class: Vec<u16> = "STATIC".encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: documented signature; the class is a system one.
        let window = unsafe {
            create(
                0,
                class.as_ptr(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                HWND_MESSAGE,
                0,
                0,
                std::ptr::null_mut(),
            )
        };
        if window == 0 {
            return false;
        }

        let mut data = vec![0u8; NID_SIZE];
        data[0..4].copy_from_slice(&(NID_SIZE as u32).to_le_bytes());
        data[OFF_HWND..OFF_HWND + 8].copy_from_slice(&(window as u64).to_le_bytes());
        data[OFF_UID..OFF_UID + 4].copy_from_slice(&ICON_ID.to_le_bytes());
        data[OFF_FLAGS..OFF_FLAGS + 4].copy_from_slice(&(NIF_INFO | NIF_TIP).to_le_bytes());
        put_wide(&mut data, OFF_TIP, 128, "sovmr");
        put_wide(&mut data, OFF_INFO, 256, body);
        put_wide(&mut data, OFF_INFO_TITLE, 64, title);
        data[OFF_INFO_FLAGS..OFF_INFO_FLAGS + 4].copy_from_slice(&NIIF_INFO.to_le_bytes());

        // SAFETY: the structure is sized and laid out per the probe.
        let shown = unsafe { shell_notify(NIM_ADD, data.as_mut_ptr()) } != 0;
        // The balloon is queued by the shell and survives the icon, so
        // both can go straight away rather than leaving a tray entry
        // behind for a message that has already been handed over.
        // SAFETY: as above; the same identity that was added.
        unsafe {
            shell_notify(NIM_DELETE, data.as_mut_ptr());
            destroy(window);
        }
        shown
    }
}

#[cfg(not(any(unix, windows)))]
mod notifier {
    pub(super) fn notify(_title: &str, _body: &str) -> bool {
        false
    }
}

#[cfg(all(test, windows))]
mod notifier_tests {
    use super::notifier;

    #[test]
    fn a_long_message_is_truncated_rather_than_overrunning_its_field() {
        // `NOTIFYICONDATAW` holds fixed arrays. A title longer than its
        // field would write past it and into the next — which is not a
        // truncated notification but a corrupted structure.
        let mut buffer = vec![0u8; 64];
        notifier::put_wide(&mut buffer, 0, 8, "a very long title indeed");
        // Seven characters plus the terminator, and nothing beyond.
        let units: Vec<u16> = buffer[..16]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(units[7], 0, "the field must end in a terminator");
        assert!(buffer[16..].iter().all(|b| *b == 0), "wrote past the field");
    }

    #[test]
    fn an_empty_message_is_still_terminated() {
        let mut buffer = vec![0xFFu8; 16];
        notifier::put_wide(&mut buffer, 0, 8, "");
        assert_eq!(buffer[0], 0);
        assert_eq!(buffer[1], 0);
    }
}
