# sovmr — sovereign media re-encoder

*Русская версия: [README.ru.md](README.ru.md).*

Visually lossless, GPU-accelerated **batch video compression** for Linux
and Windows (built for Pop!_OS / COSMIC, cross-built for Windows x86_64).
Drop folders of MP4s or MKVs in, get fast-start AV1 (or HEVC) files out at
a fraction of the size — **each container preserved**: MP4 stays MP4,
Matroska stays Matroska with every audio track, chapter and tag intact —
powered by a fully sovereign zero-copy NVDEC→NVENC pipeline on RTX GPUs
(**no FFmpeg anywhere**), with a queue that survives pauses, protects you
from generation loss, and never leaves half-written files behind.

Licensed **AGPL-3.0-or-later**. Built with zero third-party dependencies
apart from one sanctioned exception (`egui` for the GUI).

## Quick start

```sh
make            # optimized build + launch the GUI
make mock       # simulation mode: full UI demo without a GPU
make report     # headless hardware check
make install    # pin to the launcher + right-click "Compress with sovmr"
make package    # single-file dist/sovmr-x86_64.AppImage (engine included)
make test       # workspace unit + doc tests
make e2e        # real-GPU end-to-end encode test
```

## Adding files (three guaranteed ways)

1. **“➕ Add media…” button** — opens the desktop's own file dialog
   (`zenity`, `kdialog` fallback) via `std::process::Command`. Works on
   every Linux desktop, Wayland included.
2. **Command line / file manager** — `sovmr FILE...` (the right-click
   “Compress with sovmr” action uses this path).
3. **Drag & drop** — works on X11 sessions out of the box. On Wayland,
   upstream `winit` does not implement drag-and-drop yet; launch with
   `sovmr --x11` to run on XWayland where drops work. The drop zone
   tells you this instead of failing silently, and every rejected drop
   prints a detailed reason to stderr.

Intake attempts and pipeline results are also logged to
`~/.local/state/sovmr/sovmr.log` (sovereign logger, no crates).

## Workspace layout

| Crate | Depends on | Purpose |
|---|---|---|
| `sovmr-core` | *nothing* (std only, forever) | Job model, validated settings, hostile-input gate, hardware planner, and the zero-mutex message-passing batch engine. Owns the `EncoderProvider` facade every backend implements. |
| `sovmr-os` | *nothing* (not even `libc`) | The sovereign platform boundary — **every** syscall the project makes lives here: the hand-declared `dlopen`/`dlsym` loader behind an RAII `Library`, memory-mapped files, free-space checks, signal handling, and desktop integration (file picker, sleep inhibition, notifications, persisted preferences) — one platform-neutral interface per concern, Linux and Windows behind the same signatures. |
| `sovmr-cuda` | `sovmr-os` | Minimal CUDA Driver API bridge (`libcuda.so.1`): `cuInit`, device lookup, RAII contexts, driver-resolved symbolic errors. |
| `sovmr-nvenc` | `sovmr-os`, `sovmr-cuda` | Hardware encoding: versioned NVENC function table via `NvEncodeAPICreateInstance`, **AV1 and HEVC** encode sessions on sovereign CUDA contexts, VRAM registration RAII, symbolic `NVENCSTATUS` errors. |
| `sovmr-nvdec` | `sovmr-os`, `sovmr-cuda` | Hardware decoding (`libnvcuvid.so.1`): the input side of the zero-copy VRAM pipeline (see `docs/zero-copy-pipeline.md`). |
| `sovmr-mp4` | *nothing* (`#![forbid(unsafe_code)]`) | Strict zero-copy ISOBMFF toolkit: box parser, A/V demuxer with full PTS/DTS timing, HEVC bitstream facts, and a fast-start **AV1/HEVC** muxer (`moov` before `mdat`, bit-exact `mp4a` passthrough). |
| `sovmr-mkv` | `sovmr-mp4` | Sovereign Matroska: EBML reader, demuxer and muxer — carries video, **every audio track**, chapters, title and tags, zero dependencies. |
| `sovmr-audio` | *nothing* | Sovereign audio: lossless decoders and an AAC-LC encoder, zero dependencies. |
| `sovmr-native` | all GPU bridges + `sovmr-core` + `sovmr-mp4`/`sovmr-mkv`/`sovmr-audio` | **THE engine**: demux → NVDEC → VRAM → NVENC → fast-start **MP4 or Matroska**, zero copies, lock-free VRAM registration cache, as `NvencNativeProvider`. |
| `sovmr-ui` | `sovmr-core`, `sovmr-os`, `egui` | Reusable dashboard shell: pure event reducer, generic channel bridge, themed egui window, **English/Russian interface**. Knows nothing about FFmpeg or NVIDIA. |
| `sovmr-app` | everything above | Composition root (the only crate that sees all pieces): hardware detection, intake pipeline, desktop integration, CLI/file-manager entry. |
| `sovmr-setup` | *the app binary (embedded at build time)* | Sovereign Windows installer: one self-contained executable that installs, upgrades and removes sovmr. |

Dependency arrows only point left; `sovmr-ui` and the backends never see
each other. Swapping the encoding engine is one line in `sovmr-app`.

## Design guarantees

- **ACID queue** — one job is the atomic unit: outputs appear only via
  same-filesystem rename; cancel/failure deletes the `.part`; **Pause**
  never kills a running encode (unresumable work is never discarded).
- **Smart Skip** — sources already in the target codec below
  1.5 b/(px·s) are marked "already optimal" instead of being degraded by
  a pointless re-encode.
- **Hostile input** — container magic bytes are sniffed up front, and
  the sovereign demuxer distrusts every size field and table count
  (typed errors, never panics); driver output and telemetry are parsed
  defensively too.
- **Sovereign desktop integration** — notifications via `notify-send` or
  raw `gdbus`, sleep inhibition via `systemd-inhibit`, GPU meter via
  `nvidia-smi`: `std::process::Command` everywhere, no client crates.
- **Web-ready output** — the muxer writes fast-start MP4s (`moov` before
  `mdat`) with the source audio copied bit-exactly, interleaved by
  timestamp.
- **Battery-aware UI** — repaints are scheduled only while an encode is
  running; otherwise the window sleeps until an engine event wakes it.

## Portability

`make package` produces `dist/sovmr-x86_64.AppImage` — a single small
file containing exactly one binary. The engine needs no external
programs at all: it `dlopen`s the NVIDIA driver's own libraries
(`libcuda.so.1`, `libnvcuvid.so.1`, `libnvidia-encode.so.1`) at runtime,
so the image runs on a bare machine with nothing installed but the
NVIDIA driver.

On Windows the same engine ships through `sovmr-setup` (`make windows`) —
one self-contained installer executable that carries the program inside
itself and resolves the driver's own DLLs (`nvcuda.dll`, `nvcuvid.dll`,
`nvEncodeAPI64.dll`) at runtime, the same way.

## Requirements

- Linux x86_64 with a display (Wayland or X11), or Windows x86_64.
- NVIDIA GPU + driver for hardware encoding (AV1 needs RTX 40-series or
  newer; older RTX/GTX fall back to HEVC). The `--mock` mode runs the
  whole UI on any machine, GPU or not.
