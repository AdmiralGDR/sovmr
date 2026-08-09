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

//! # sovmr-setup — the sovereign Windows installer
//!
//! One self-contained executable that installs, upgrades and removes
//! sovmr. The program being installed is carried inside this binary by
//! `include_bytes!`, so there is nothing to unpack alongside it and
//! nothing to download.
//!
//! No WiX, no NSIS, no Inno. Each would pull an external toolchain into
//! the build chain for work the Win32 API already does, and the project
//! builds itself from one compiler by rule.
//!
//! # What "installed" means here
//!
//! * the program lands in `%LOCALAPPDATA%\Programs\sovmr`;
//! * a Start-menu shortcut points at it;
//! * an entry appears in *Apps & features* with a version and a working
//!   Remove button.
//!
//! All per-user: no elevation is requested, because none is needed to
//! write inside the user's own profile. An installer that asks for
//! administrator rights to copy one executable is asking for trust it
//! has no use for.
//!
//! # Modes
//!
//! | Argument | Meaning |
//! |---|---|
//! | *(none)* | install, or upgrade in place if already present |
//! | `--upgrade` | refuse unless something is already installed |
//! | `--uninstall` | remove files, shortcut and registry entry |
//!
//! State (`%LOCALAPPDATA%\sovmr`, where the log lives) is never touched
//! by an upgrade, and left alone by an uninstall as well — removing a
//! program should not silently discard the record of what it did.

#![deny(missing_docs)]
#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(windows)]
mod win32;

// The payload, baked in by `build.rs` (the generated file carries the
// doc comment; rustdoc does not read one on a macro invocation). Empty
// when this installer was built without a payload, which is checked
// before anything is written.
include!(concat!(env!("OUT_DIR"), "/payload.rs"));

/// What the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Install, upgrading in place if an installation already exists.
    Install,
    /// Upgrade only: fail if nothing is installed.
    Upgrade,
    /// Remove everything this installer created.
    Uninstall,
}

/// Parse the command line into a [`Mode`].
///
/// Unknown arguments are an error rather than being ignored: an
/// installer that quietly does something other than what it was asked is
/// the worst kind of surprise.
fn parse_mode(args: &[String]) -> Result<Mode, String> {
    match args.first().map(String::as_str) {
        None => Ok(Mode::Install),
        Some("--upgrade") => Ok(Mode::Upgrade),
        Some("--uninstall") => Ok(Mode::Uninstall),
        Some(other) => Err(format!(
            "unknown argument '{other}' (expected --upgrade, --uninstall, or nothing)"
        )),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match parse_mode(&args) {
        Ok(m) => m,
        Err(e) => {
            report(&e, true);
            std::process::exit(2);
        }
    };
    match run(mode) {
        Ok(summary) => report(&summary, false),
        Err(e) => {
            report(&e, true);
            std::process::exit(1);
        }
    }
}

/// Say something to whoever started this: stdout under a console, a
/// message box when double-clicked from Explorer.
fn report(text: &str, error: bool) {
    #[cfg(windows)]
    {
        if win32::has_stdout() {
            if error {
                eprintln!("sovmr-setup: {text}");
            } else {
                println!("{text}");
            }
        } else {
            win32::message("sovmr", text, error);
        }
    }
    #[cfg(not(windows))]
    {
        if error {
            eprintln!("sovmr-setup: {text}");
        } else {
            println!("{text}");
        }
    }
}

#[cfg(not(windows))]
fn run(_mode: Mode) -> Result<String, String> {
    // Compiles everywhere so `cargo build --workspace` and the lints stay
    // whole-workspace on Linux; does nothing, because there is nothing
    // here it could honestly do.
    Err("sovmr-setup installs sovmr on Windows; on Linux use `make install`".to_string())
}

#[cfg(windows)]
fn run(mode: Mode) -> Result<String, String> {
    use std::path::PathBuf;

    /// Where the program is installed, under the user's profile.
    const PROGRAMS_SUBDIR: &str = r"Programs\sovmr";
    /// Registry location Windows reads "Apps & features" from.
    const UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\sovmr";
    /// Start-menu location, relative to `%APPDATA%`.
    const START_MENU_SUBDIR: &str = r"Microsoft\Windows\Start Menu\Programs";
    /// Name of the installed program.
    const EXE_NAME: &str = "sovmr.exe";
    /// Name the installer keeps itself under, so the Remove button in
    /// "Apps & features" has something to invoke.
    const SETUP_NAME: &str = "sovmr-setup.exe";
    /// Shortcut file name.
    const SHORTCUT_NAME: &str = "sovmr.lnk";
    /// Display name in "Apps & features".
    const DISPLAY_NAME: &str = "sovmr — sovereign media re-encoder";
    /// Publisher string.
    const PUBLISHER: &str = "sovmr contributors";
    /// Suffix for the staging directory an install is assembled in.
    const STAGING_SUFFIX: &str = ".staging";

    // Release and commit both: the registry entry a user sees should
    // identify the build exactly, not merely the release it intended.
    let version = concat!(env!("CARGO_PKG_VERSION"), "+", env!("SOVMR_COMMIT"));

    let local = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| "LOCALAPPDATA is not set — cannot locate the user profile".to_string())?;
    let install_dir = local.join(PROGRAMS_SUBDIR);
    let shortcut = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .map(|a| a.join(START_MENU_SUBDIR).join(SHORTCUT_NAME))
        .ok_or_else(|| "APPDATA is not set — cannot locate the Start menu".to_string())?;

    let installed = win32::read_string(UNINSTALL_KEY, "DisplayVersion");

    match mode {
        Mode::Uninstall => {
            win32::delete_tree(UNINSTALL_KEY)?;
            let _ = std::fs::remove_file(&shortcut);
            // Remove the directory last: on failure the registry entry is
            // already gone, so a retry is never blocked by a half state.
            if install_dir.exists() {
                std::fs::remove_dir_all(&install_dir)
                    .map_err(|e| format!("removing {}: {e}", install_dir.display()))?;
            }
            Ok(format!(
                "sovmr removed.\n\nSettings and the log in {} were left in place.",
                local.join("sovmr").display()
            ))
        }
        Mode::Upgrade if installed.is_none() => {
            Err("nothing to upgrade — sovmr is not installed".to_string())
        }
        Mode::Install | Mode::Upgrade => {
            if PAYLOAD.is_empty() {
                return Err(
                    "this installer was built without a payload and cannot install anything"
                        .to_string(),
                );
            }
            // ACID, the same contract the muxer honours: assemble the new
            // installation beside the old one and swap it in with a
            // rename. An install interrupted halfway leaves the previous
            // version working rather than a half-written directory that
            // looks installed.
            let staging = install_dir.with_extension(STAGING_SUFFIX.trim_start_matches('.'));
            let _ = std::fs::remove_dir_all(&staging);
            std::fs::create_dir_all(&staging)
                .map_err(|e| format!("creating {}: {e}", staging.display()))?;

            std::fs::write(staging.join(EXE_NAME), PAYLOAD)
                .map_err(|e| format!("writing {EXE_NAME}: {e}"))?;
            // Keep a copy of ourselves so the Remove button has something
            // to run after the download is long gone.
            let self_exe =
                std::env::current_exe().map_err(|e| format!("locating this installer: {e}"))?;
            std::fs::copy(&self_exe, staging.join(SETUP_NAME))
                .map_err(|e| format!("copying the uninstaller: {e}"))?;

            // The running program cannot be replaced underneath itself.
            if install_dir.exists() {
                let retired = install_dir.with_extension("previous");
                let _ = std::fs::remove_dir_all(&retired);
                std::fs::rename(&install_dir, &retired).map_err(|e| {
                    format!(
                        "cannot replace {} — is sovmr still running? ({e})",
                        install_dir.display()
                    )
                })?;
                std::fs::rename(&staging, &install_dir)
                    .map_err(|e| format!("committing the new version: {e}"))?;
                let _ = std::fs::remove_dir_all(&retired);
            } else {
                if let Some(parent) = install_dir.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("creating {}: {e}", parent.display()))?;
                }
                std::fs::rename(&staging, &install_dir)
                    .map_err(|e| format!("committing the installation: {e}"))?;
            }

            let exe = install_dir.join(EXE_NAME);
            if let Some(parent) = shortcut.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            win32::create_shortcut(&shortcut, &exe, &install_dir, DISPLAY_NAME)?;

            let uninstall_cmd =
                format!("\"{}\" --uninstall", install_dir.join(SETUP_NAME).display());
            let size_kib = (PAYLOAD.len() / 1024).max(1) as u32;
            win32::write_uninstall_entry(
                UNINSTALL_KEY,
                DISPLAY_NAME,
                version,
                PUBLISHER,
                &install_dir,
                &uninstall_cmd,
                size_kib,
            )?;

            let what = match installed {
                Some(old) if old != version => format!("upgraded {old} → {version}"),
                Some(_) => format!("reinstalled {version}"),
                None => format!("installed {version}"),
            };
            Ok(format!(
                "sovmr {what}.\n\nProgram: {}\nStart menu: {}",
                exe.display(),
                shortcut.display()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_means_install() {
        assert_eq!(parse_mode(&[]).unwrap(), Mode::Install);
    }

    #[test]
    fn the_two_maintenance_modes_are_recognised() {
        assert_eq!(
            parse_mode(&["--upgrade".to_string()]).unwrap(),
            Mode::Upgrade
        );
        assert_eq!(
            parse_mode(&["--uninstall".to_string()]).unwrap(),
            Mode::Uninstall
        );
    }

    #[test]
    fn an_unknown_argument_is_refused_rather_than_ignored() {
        // Silently installing when asked to do something else is how a
        // user ends up with software they did not agree to.
        let err = parse_mode(&["--remove".to_string()]).unwrap_err();
        assert!(err.contains("--remove"), "{err}");
        assert!(err.contains("--uninstall"), "must name the real spelling");
    }

    #[test]
    fn a_typo_never_falls_through_to_install() {
        for bad in ["-uninstall", "uninstall", "--Uninstall", "/S"] {
            assert!(
                parse_mode(&[bad.to_string()]).is_err(),
                "'{bad}' must not be accepted"
            );
        }
    }
}
