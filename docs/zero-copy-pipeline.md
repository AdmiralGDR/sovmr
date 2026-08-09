<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# The zero-copy VRAM pipeline (Phase 2 target architecture)

*Русская версия: [zero-copy-pipeline.ru.md](zero-copy-pipeline.ru.md).*

The sovereign replacement for FFmpeg is not "our own FFmpeg" — it is a
shorter path. Once `sovmr-mp4` feeds NVDEC directly and NVDEC's output
feeds NVENC directly, **decoded pixels never leave the GPU**:

```
 host (CPU RAM)                     │ device (VRAM, RTX 4070)
 ───────────────────────────────── │ ─────────────────────────────────────
 sovmr-mp4: moov walk, stsz/stco,  │
 avcC → Annex B repacketize        │
      compressed samples (small) ──►│ NVDEC (sovmr-nvdec)
                                    │   cuvidDecodePicture
                                    │   cuvidMapVideoFrame64
                                    │     └─► CUdeviceptr + pitch (NV12 surface)
                                    │           │  nvEncRegisterResource
                                    │           │    (NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR)
                                    │           │  nvEncMapInputResource
                                    │           ▼
                                    │ NVENC (sovmr-nvenc)
                                    │   nvEncEncodePicture (AV1/HEVC, CQ mode)
      encoded packets (small)  ◄────│   nvEncLockBitstream
 sovmr-mp4: mux into fresh mp4     │   cuvidUnmapVideoFrame64 (surface reuse)
```

Only compressed bitstream crosses PCIe — for a typical 4K H.264 → AV1
job that is 1–4 % of the bytes a CPU round-trip would move. The video
engines (NVDEC/NVENC) are dedicated silicon, so the CUDA cores stay free.

## Memory-management strategy

1. **One `Cuda` + one `CudaContext` per worker slot** (sovmr-cuda).
   The context owns every VRAM allocation transitively; its RAII drop is
   the last one to run — enforced by the borrow checker, exactly like
   `NvEncSession<'a>` today.

2. **Decode surface pool, owned by the decoder.** NVDEC decodes into an
   internal ring of `ulNumDecodeSurfaces` frames. `cuvidMapVideoFrame64`
   borrows one surface out of the ring as a `CUdeviceptr`; the planned
   `MappedFrame<'dec>` RAII type unmaps on drop, making "forgot to unmap
   → decoder stalls" unrepresentable.

3. **Registration cache on the encode side.** `nvEncRegisterResource` is
   expensive; each of the pool's device pointers is registered once and
   cached. Per frame only the cheap `nvEncMapInputResource` /
   `nvEncUnmapInputResource` pair runs.

4. **Ordering via CUDA streams, not host waits.** One `CUstream` per
   worker slot handed to both engines (`nvEncSetIOCudaStreams`), so
   decode-complete → encode-start ordering is resolved on-device; the
   host thread only pumps packets and drains bitstream.

5. **ACID at every step** (pattern lifted from the `sovscale` review —
   its `vram.zig` rolls back partially-created buffers with `errdefer`
   chains): each pipeline object (decoder, pool slot, registration,
   mapping) is an RAII type constructed through a fallible builder; a
   failure at step N automatically unwinds steps N-1..1. A crash can
   leak at most driver-reclaimed VRAM, never a wedged encoder session.

## Crate responsibilities

| Stage | Crate | Status |
|---|---|---|
| Demux (moov/stsz/stco → Annex B) | `sovmr-mp4` | **done & battle-tested** in the live loop |
| Hardware decode | `sovmr-nvdec` | **done**: push parser + safe C callbacks, decode/map loop live |
| VRAM handoff | `sovmr-nvdec` ↔ `sovmr-nvenc` | **done**: `MappedFrame` → `register_vram_frame` → `MappedInput`, zero host copies |
| Hardware encode | `sovmr-nvenc` | **done**: `nvEncInitializeEncoder` (preset fill + CQ patch), `EncodePicture`, `LockBitstream` |
| The whole loop | `sovmr-native` | **LIVE**: `NvencNativeProvider` behind the core facade; 60/60 fixture frames → valid AV1, 720p at ~530 fps (~17× realtime) on the RTX 4070M |
| Mux back to MP4 | `sovmr-mp4` | **done**: fast-start MP4 (`moov` before `mdat`) for **both** targets — AV1 (`av01`+`av1C`) and HEVC (`hvc1`+`hvcC`, parameter sets harvested from the SPS itself) — with bit-exact `mp4a` audio passthrough and timestamp interleave; unsupported source audio degrades to video-only with a warning; verified by parsing our own output with our own demuxer |
| VRAM registration cache | `sovmr-native` | **done**: one `nvEncRegisterResource` per NVDEC pool surface, cached in a plain single-thread `HashMap` behind `&mut` — no `Arc<Mutex>` in the hot loop (strategy §3 below) |

When the loop closes, `NvencNativeProvider` implements
`sovmr_core::encoder::EncoderProvider` and the composition root swaps one
line — UI, queue, Smart Skip and OS integration stay untouched.
