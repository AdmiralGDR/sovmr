// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emits the build provenance `sovmr_core::identity` reads back.

include!("../build/identity.rs");

fn main() {
    emit_commit_identity();
    // `TARGET` is set by cargo for the build script but is not visible
    // to the crate itself; forwarding it is how a cross-built binary can
    // say which machine it is for.
    println!(
        "cargo:rustc-env=SOVMR_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string())
    );
}
