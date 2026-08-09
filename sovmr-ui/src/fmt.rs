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

//! Human-friendly formatting of machine metrics ("responsive metrics, not
//! a terminal dump"). Pure functions, reusable by any frontend.

use std::time::Duration;

use crate::i18n::Lang;

/// Binary size thresholds and their unit labels.
const UNITS: [(&str, u64); 4] = [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10), ("B", 1)];

/// Render a byte count with one decimal for GB/MB, none below.
///
/// # Example
///
/// ```
/// use sovmr_ui::fmt::bytes;
/// assert_eq!(bytes(1_288_490_189), "1.2 GB");
/// assert_eq!(bytes(471_040), "460 KB");
/// assert_eq!(bytes(0), "0 B");
/// ```
pub fn bytes(n: u64) -> String {
    for (unit, scale) in UNITS {
        if n >= scale {
            let value = n as f64 / scale as f64;
            return if scale >= 1 << 20 {
                format!("{value:.1} {unit}")
            } else {
                format!("{value:.0} {unit}")
            };
        }
    }
    "0 B".to_string()
}

/// Render a remaining-time estimate: `"2m 14s"`, `"48s"`, `"1h 05m"`.
///
/// The digits and layout are language-neutral; only the unit suffixes
/// (`h`/`m`/`s` vs `ч`/`м`/`с`) come from the catalogue, so a Russian UI
/// does not leave English letters trailing the number.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use sovmr_ui::fmt::eta;
/// use sovmr_ui::i18n::Lang;
/// assert_eq!(eta(Lang::En, Duration::from_secs(134)), "2m 14s");
/// assert_eq!(eta(Lang::En, Duration::from_secs(48)), "48s");
/// assert_eq!(eta(Lang::Ru, Duration::from_secs(3900)), "1ч 05м");
/// ```
pub fn eta(lang: Lang, d: Duration) -> String {
    let s = d.as_secs();
    let t = lang.strings();
    match s {
        0..=59 => format!("{s}{}", t.time_seconds),
        60..=3599 => format!(
            "{}{} {:02}{}",
            s / 60,
            t.time_minutes,
            s % 60,
            t.time_seconds
        ),
        _ => format!(
            "{}{} {:02}{}",
            s / 3600,
            t.time_hours,
            (s % 3600) / 60,
            t.time_minutes
        ),
    }
}

/// Render the size change of a finished job: `"-62%"`; negative savings
/// (output grew) render as `"+8%"` so surprises stay visible.
///
/// # Example
///
/// ```
/// use sovmr_ui::fmt::reduction;
/// assert_eq!(reduction(1000, 380), "-62%");
/// assert_eq!(reduction(1000, 1080), "+8%");
/// assert_eq!(reduction(0, 500), "n/a");
/// ```
pub fn reduction(input_bytes: u64, output_bytes: u64) -> String {
    if input_bytes == 0 {
        return "n/a".to_string();
    }
    let pct = (output_bytes as f64 / input_bytes as f64 - 1.0) * 100.0;
    format!("{pct:+.0}%").replace("+-", "-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_covers_all_magnitudes() {
        assert_eq!(bytes(3), "3 B");
        assert_eq!(bytes(2048), "2 KB");
        assert_eq!(bytes(5 << 20), "5.0 MB");
        assert_eq!(bytes(3_650_722_201), "3.4 GB");
    }

    #[test]
    fn eta_boundaries() {
        assert_eq!(eta(Lang::En, Duration::ZERO), "0s");
        assert_eq!(eta(Lang::En, Duration::from_secs(59)), "59s");
        assert_eq!(eta(Lang::En, Duration::from_secs(60)), "1m 00s");
        assert_eq!(eta(Lang::En, Duration::from_secs(3599)), "59m 59s");
        assert_eq!(eta(Lang::En, Duration::from_secs(3600)), "1h 00m");
    }

    #[test]
    fn eta_units_are_translated() {
        // The layout and digits are identical; only the unit letters
        // change, so a Russian UI shows no trailing English.
        assert_eq!(eta(Lang::Ru, Duration::from_secs(48)), "48с");
        assert_eq!(eta(Lang::Ru, Duration::from_secs(134)), "2м 14с");
        assert_eq!(eta(Lang::Ru, Duration::from_secs(3900)), "1ч 05м");
    }

    #[test]
    fn reduction_signs() {
        assert_eq!(reduction(100, 100), "+0%");
        assert_eq!(reduction(100, 25), "-75%");
        assert_eq!(reduction(100, 250), "+150%");
    }
}
