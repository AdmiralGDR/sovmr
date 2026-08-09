// SPDX-License-Identifier: AGPL-3.0-or-later
// sovmr — provider facade smoke example (Copyright (C) 2026 sovmr contributors,
// GNU AGPL v3-or-later; see <https://www.gnu.org/licenses/>).
// quick provider-facade smoke via the real core scheduler
use std::sync::Arc;
fn main() {
    let s = std::env::args().nth(1).unwrap();
    let input = std::path::PathBuf::from(&s);
    let output = input.with_extension("sovereign.obu");
    let provider: Arc<dyn sovmr_core::encoder::EncoderProvider> =
        Arc::new(sovmr_native::NvencNativeProvider);
    let info = provider.probe(&input).unwrap();
    println!(
        "probe: {}x{} {} {}B",
        info.width, info.height, info.video_codec, info.size_bytes
    );
    let job = sovmr_core::job::MediaJob {
        id: sovmr_core::job::JobId::next(),
        input,
        output: output.clone(),
        settings: sovmr_core::job::EncodeSettings::visually_lossless(sovmr_core::job::Codec::Av1),
        info,
    };
    let (engine, events) = sovmr_core::queue::scheduler::start(provider, 1);
    engine.send(sovmr_core::queue::messages::EngineCommand::Enqueue(job));
    while let Ok(ev) = events.recv() {
        use sovmr_core::queue::messages::EngineEvent as E;
        match ev {
            E::Progress { fps, .. } => println!("progress fps={fps:.0}"),
            E::Completed { output_bytes, .. } => {
                println!("COMPLETED {output_bytes} bytes -> {}", output.display());
                break;
            }
            E::Failed { reason, .. } => {
                println!("FAILED: {reason}");
                break;
            }
            _ => {}
        }
    }
    engine.send(sovmr_core::queue::messages::EngineCommand::Shutdown);
}
