<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Русская версия. English: [zero-copy-pipeline.md](zero-copy-pipeline.md). -->
# Конвейер zero-copy через VRAM (целевая архитектура фазы 2)

Суверенная замена FFmpeg — это не «свой FFmpeg», а более короткий путь.
Как только `sovmr-mp4` подаёт данные прямо в NVDEC, а выход NVDEC — прямо
в NVENC, **декодированные пиксели никогда не покидают GPU**:

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

По PCIe пересекает только сжатый битстрим — для типичной задачи
4K H.264 → AV1 это 1–4 % байтов, которые перегонял бы обход через CPU.
Видеодвижки (NVDEC/NVENC) — выделенный кремний, поэтому ядра CUDA остаются
свободны.

## Стратегия управления памятью

1. **Один `Cuda` + один `CudaContext` на рабочий слот** (sovmr-cuda).
   Контекст транзитивно владеет каждой аллокацией VRAM; его RAII-drop
   срабатывает последним — это гарантирует borrow checker, ровно как
   `NvEncSession<'a>` сегодня.

2. **Пул поверхностей декодирования, которым владеет декодер.** NVDEC
   декодирует во внутреннее кольцо из `ulNumDecodeSurfaces` кадров.
   `cuvidMapVideoFrame64` заимствует одну поверхность из кольца как
   `CUdeviceptr`; запланированный RAII-тип `MappedFrame<'dec>` снимает
   отображение при drop, делая «забыл unmap → декодер застревает»
   непредставимым.

3. **Кэш регистраций на стороне кодирования.** `nvEncRegisterResource`
   дорог; каждый указатель устройства из пула регистрируется один раз и
   кэшируется. На кадр выполняется только дешёвая пара
   `nvEncMapInputResource` / `nvEncUnmapInputResource`.

4. **Упорядочивание через CUDA-стримы, а не ожидания на хосте.** Один
   `CUstream` на рабочий слот передаётся обоим движкам
   (`nvEncSetIOCudaStreams`), так что порядок «декод завершён →
   кодирование началось» разрешается на устройстве; поток хоста лишь
   качает пакеты и сливает битстрим.

5. **ACID на каждом шаге** (паттерн взят из обзора `sovscale` — его
   `vram.zig` откатывает частично созданные буферы цепочками `errdefer`):
   каждый объект конвейера (декодер, слот пула, регистрация, отображение)
   — RAII-тип, создаваемый через отказоспособный билдер; сбой на шаге N
   автоматически разматывает шаги N-1..1. Крах может утечь максимум VRAM,
   возвращаемую драйвером, но никогда — заклиненную сессию энкодера.

## Ответственности крейтов

| Этап | Крейт | Статус |
|---|---|---|
| Демукс (moov/stsz/stco → Annex B) | `sovmr-mp4` | **готово и обкатано в бою** в живом цикле |
| Аппаратное декодирование | `sovmr-nvdec` | **готово**: push-парсер + безопасные C-колбэки, цикл декод/map живой |
| Передача через VRAM | `sovmr-nvdec` ↔ `sovmr-nvenc` | **готово**: `MappedFrame` → `register_vram_frame` → `MappedInput`, ноль копий на хосте |
| Аппаратное кодирование | `sovmr-nvenc` | **готово**: `nvEncInitializeEncoder` (заполнение пресета + патч CQ), `EncodePicture`, `LockBitstream` |
| Весь цикл | `sovmr-native` | **ЖИВОЙ**: `NvencNativeProvider` за фасадом ядра; 60/60 кадров фикстуры → валидный AV1, 720p на ~530 fps (~17× реального времени) на RTX 4070M |
| Мукс обратно в MP4 | `sovmr-mp4` | **готово**: fast-start MP4 (`moov` перед `mdat`) для **обеих** целей — AV1 (`av01`+`av1C`) и HEVC (`hvc1`+`hvcC`, наборы параметров извлечены из самого SPS) — с побайтовым passthrough звука `mp4a` и чередованием по временным меткам; неподдерживаемый звук источника деградирует до видео-only с предупреждением; проверено разбором нашего собственного вывода нашим же демуксером |
| Кэш регистраций VRAM | `sovmr-native` | **готово**: одна `nvEncRegisterResource` на поверхность пула NVDEC, кэш в обычном однопоточном `HashMap` за `&mut` — без `Arc<Mutex>` в горячем цикле (стратегия §3 выше) |

Когда цикл замыкается, `NvencNativeProvider` реализует
`sovmr_core::encoder::EncoderProvider`, и корень композиции меняет одну
строку — UI, очередь, Умный пропуск и интеграция с ОС остаются
нетронутыми.
