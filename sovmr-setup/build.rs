// SPDX-License-Identifier: GPL-3.0-or-later
//
// Bakes the payload path in at compile time.
//
// `include_bytes!` needs a literal, so the choice of "real payload" vs
// "none" has to be made before the compiler sees the source. Writing a
// tiny generated file is the way to do that while keeping
// `cargo build --workspace` working on a machine that has never built a
// Windows binary — there, the payload is simply empty and the installer
// refuses to run rather than failing to compile.
use std::path::Path;

include!("../build/identity.rs");

fn main() {
    emit_commit_identity();
    println!("cargo:rerun-if-env-changed=SOVMR_PAYLOAD");
    let out = std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR");
    let dest = Path::new(&out).join("payload.rs");

    let body = match std::env::var("SOVMR_PAYLOAD") {
        Ok(p) if Path::new(&p).is_file() => {
            println!("cargo:rerun-if-changed={p}");
            format!(
                "/// The `sovmr.exe` this installer carries.\n\
                 pub static PAYLOAD: &[u8] = include_bytes!(r\"{p}\");\n"
            )
        }
        _ => "/// No payload was supplied at build time.\n\
              pub static PAYLOAD: &[u8] = &[];\n"
            .to_string(),
    };
    std::fs::write(&dest, body).expect("OUT_DIR is writable");
}
