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

//! Sovereign telemetry: a minimal append-only file logger for
//! `$XDG_STATE_HOME/sovmr/sovmr.log` (default `~/.local/state/sovmr/`)
//! — the Freedesktop-blessed home for logs. No `log`, no `tracing`, no
//! `chrono` ; timestamps come from `SystemTime` plus a ~15-line
//! civil-date conversion (Howard Hinnant's `civil_from_days`).
//!
//! Allocation discipline: callers pass `format_args!` (via the
//! [`crate::sovlog!`] macro), which is formatted **directly into the
//! `BufWriter`** — no intermediate `String` per message. Failure
//! discipline: a missing/unwritable state dir degrades to
//! "logging disabled" once on stderr; it never breaks the app.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Application subdirectory + file name under the state home.
const LOG_SUBPATH: &str = "sovmr/sovmr.log";
/// Days from 0000-03-01 to 1970-01-01 (era-based civil calendar).
const DAYS_TO_UNIX_EPOCH: i64 = 719_468;
/// Days per 400-year era (146 097) — the calendar's full repeat cycle.
const DAYS_PER_ERA: i64 = 146_097;

/// The process-wide sink. `None` inside = logging disabled.
static SINK: OnceLock<Mutex<Option<BufWriter<File>>>> = OnceLock::new();

/// Resolve, create, and open the log file; enables [`log`] for the rest
/// of the process. Returns the path for the startup banner, or `None`
/// (after a single stderr note) when the state dir is unusable.
pub fn init() -> Option<PathBuf> {
    // Asked of the platform boundary rather than read here: this used
    // to consult `$XDG_STATE_HOME` directly, which on Windows is unset,
    // so nothing was logged and nothing said so.
    let state_home = sovmr_os::desktop::state_dir()?;
    let path = state_home.join(LOG_SUBPATH);
    let open = || -> std::io::Result<File> {
        std::fs::create_dir_all(path.parent().expect("LOG_SUBPATH has a parent"))?;
        OpenOptions::new().create(true).append(true).open(&path)
    };
    match open() {
        Ok(file) => {
            let _ = SINK.set(Mutex::new(Some(BufWriter::new(file))));
            Some(path)
        }
        Err(e) => {
            eprintln!("sovmr: logging disabled ({}: {e})", path.display());
            let _ = SINK.set(Mutex::new(None));
            None
        }
    }
}

/// Append one timestamped line, formatting straight into the writer.
/// Prefer the [`crate::sovlog!`] macro. No-op before [`init`] / after
/// an init failure; a write error disables logging (once, on stderr)
/// rather than ever failing the caller.
pub fn log(args: std::fmt::Arguments<'_>) {
    let Some(sink) = SINK.get() else { return };
    let Ok(mut guard) = sink.lock() else { return };
    let Some(writer) = guard.as_mut() else { return };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let (y, mo, d, h, mi, s) = civil(now.as_secs());
    let result = writeln!(
        writer,
        "{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z {args}"
    )
    .and_then(|()| writer.flush());
    if let Err(e) = result {
        eprintln!("sovmr: logging disabled mid-run (write failed: {e})");
        *guard = None;
    }
}

/// Timestamped log line into the sovereign logger (see [`logger`](self)).
#[macro_export]
macro_rules! sovlog {
    ($($t:tt)*) => {
        $crate::logger::log(format_args!($($t)*))
    };
}

/// Unix seconds → (year, month, day, hour, minute, second) in UTC.
/// Civil-from-days per Howard Hinnant's algorithms (exact for the whole
/// proleptic Gregorian calendar; we only ever feed it the present).
fn civil(unix_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = unix_secs as i64;
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400) as u32;

    let z = days + DAYS_TO_UNIX_EPOCH;
    let era = z.div_euclid(DAYS_PER_ERA);
    let doe = z.rem_euclid(DAYS_PER_ERA); // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // March-based month [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, tod / 3600, (tod / 60) % 60, tod % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_conversion_hits_known_instants() {
        assert_eq!(civil(0), (1970, 1, 1, 0, 0, 0));
        // 2000-02-29T12:34:56Z (leap day) = 951827696.
        assert_eq!(civil(951_827_696), (2000, 2, 29, 12, 34, 56));
        // 2026-07-11T00:00:00Z = 1783728000 (verified: date -u -d @…).
        assert_eq!(civil(1_783_728_000), (2026, 7, 11, 0, 0, 0));
        // Year-boundary second: 2023-12-31T23:59:59Z = 1704067199.
        assert_eq!(civil(1_704_067_199), (2023, 12, 31, 23, 59, 59));
    }

    #[test]
    fn logger_writes_timestamped_lines_to_the_state_file() {
        // Redirect the state home into a scratch dir for the test.
        let dir = std::env::temp_dir().join(format!("sovmr-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("XDG_STATE_HOME", &dir);
        let path = init().expect("state dir is writable");
        assert!(path.starts_with(&dir), "honours XDG_STATE_HOME");

        sovlog!("intake: accepted {} file(s)", 2);
        log(format_args!("pipeline: {} failed: {}", "clip.mp4", "demux"));

        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let first = lines.next().expect("first line written");
        assert!(
            first.ends_with("Z intake: accepted 2 file(s)"),
            "got: {first}"
        );
        // Timestamp shape: YYYY-MM-DDTHH:MM:SSZ then the message.
        assert_eq!(first.as_bytes()[4], b'-');
        assert_eq!(first.as_bytes()[10], b'T');
        assert_eq!(first.as_bytes()[19], b'Z');
        assert!(lines.next().unwrap().contains("clip.mp4 failed: demux"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
