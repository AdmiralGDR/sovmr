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

//! Sovereign, dependency-free localisation.
//!
//! No `fluent`, no `gettext`, no crates: a language is a `&'static`
//! table of strings, chosen at run time. The design makes a **missing
//! translation unrepresentable** — [`Strings`] is a plain struct of
//! `&'static str` fields, and [`EN`]/[`RU`] are `const` values of it, so
//! a field added without both languages does not compile. There is no
//! run-time "key not found", because there are no keys: every message is
//! a field the type system already checked.
//!
//! Formatting stays in the caller. This module holds words and fixed
//! phrases; anything with a number in it (`"50 %"`, `"3 of 8"`) is
//! composed by the shell from these fragments, so no ad-hoc format
//! language has to be invented or parsed.
//!
//! # Example
//!
//! ```
//! use sovmr_ui::i18n::Lang;
//!
//! assert_eq!(Lang::default(), Lang::En);
//! assert_eq!(Lang::En.strings().tab_dashboard, "Dashboard");
//! assert_eq!(Lang::Ru.strings().tab_dashboard, "Панель");
//! assert_eq!(Lang::En.other(), Lang::Ru); // the toggle target
//! ```

/// Which language the interface is drawn in. English is the default so a
/// machine with no preference set reads the same as the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lang {
    /// English — the default and the language the source strings are in.
    #[default]
    En,
    /// Russian.
    Ru,
}

impl Lang {
    /// The string table for this language.
    pub fn strings(self) -> &'static Strings {
        match self {
            Lang::En => &EN,
            Lang::Ru => &RU,
        }
    }

    /// The other language — what a single toggle switches to.
    pub fn other(self) -> Lang {
        match self {
            Lang::En => Lang::Ru,
            Lang::Ru => Lang::En,
        }
    }

    /// A stable two-letter code for persistence. Distinct from
    /// [`Lang::switch_badge`], which names the *toggle target*: this names
    /// the language itself, and must stay constant across releases so a
    /// saved preference keeps meaning the same thing.
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ru => "ru",
        }
    }

    /// Parse a stored [`Lang::code`] back into a language.
    ///
    /// Returns `None` for anything unrecognised — an old or corrupt
    /// preference falls back to the default rather than guessing, and the
    /// caller decides what the default is.
    pub fn from_code(code: &str) -> Option<Lang> {
        match code.trim() {
            "en" => Some(Lang::En),
            "ru" => Some(Lang::Ru),
            _ => None,
        }
    }

    /// A two-letter badge for the language *switch*, naming where the
    /// toggle leads rather than where it is: shown on the button that
    /// changes the language, so it reads as an action.
    pub fn switch_badge(self) -> &'static str {
        // The badge names the destination, so it is the *other* code.
        match self.other() {
            Lang::En => "EN",
            Lang::Ru => "RU",
        }
    }
}

/// Every fixed piece of interface text, in one language.
///
/// One flat struct on purpose: adding a message is adding a field, and a
/// field cannot be half-added — [`EN`] and [`RU`] both stop compiling
/// until it is filled in on both sides. That compile error is the whole
/// point, and it replaces an entire class of "untranslated string ships
/// to users" bug with something the build refuses.
#[derive(Debug, Clone, Copy)]
pub struct Strings {
    // ---- header -----------------------------------------------------
    /// Dashboard tab label.
    pub tab_dashboard: &'static str,
    /// Advanced tab label.
    pub tab_advanced: &'static str,
    /// Pause button.
    pub pause: &'static str,
    /// Resume button (queue is paused).
    pub resume: &'static str,
    /// The "PAUSED" badge shown beside the toggle.
    pub paused_badge: &'static str,
    /// Hover text explaining what Pause does.
    pub pause_hover: &'static str,
    /// Hover text over the GPU utilisation meter.
    pub gpu_meter_hover: &'static str,
    /// Tooltip on the language toggle (bilingual on purpose, so it is
    /// recognisable whichever language is active).
    pub language_hover: &'static str,

    // ---- empty drop zone --------------------------------------------
    /// Headline when drag-and-drop is available.
    pub drop_headline_dnd: &'static str,
    /// Headline when drag-and-drop is not available.
    pub drop_headline_nodnd: &'static str,
    /// Add-media button, idle.
    pub add_media: &'static str,
    /// Add-media button while the native dialog is open.
    pub add_media_busy: &'static str,
    /// The one-line promise under the button.
    pub tagline: &'static str,
    /// Note shown when drag-and-drop is unavailable (Wayland).
    pub dnd_unavailable: &'static str,
    /// Strip prompt to add more, drag-and-drop available.
    pub strip_more_dnd: &'static str,
    /// Strip prompt to add more, no drag-and-drop.
    pub strip_more_nodnd: &'static str,

    // ---- batch summary ----------------------------------------------
    /// Connector in "3 of 8 done".
    pub word_of: &'static str,
    /// "done" in "3 of 8 done".
    pub word_done: &'static str,
    /// "running" in "2 running".
    pub word_running: &'static str,
    /// "finishing" in "2 finishing" (shown while paused).
    pub word_finishing: &'static str,
    /// "paused" in the summary line.
    pub word_paused: &'static str,
    /// Cancel-all button.
    pub cancel_all: &'static str,

    // ---- savings bar ------------------------------------------------
    /// Prefix of "Space saved: 1.2 GB (…)".
    pub space_saved_prefix: &'static str,
    /// Whole line when nothing was saved yet.
    pub no_space_saved: &'static str,
    /// Hover over the savings bar.
    pub savings_hover: &'static str,

    // ---- per-row status ---------------------------------------------
    /// Queued row.
    pub status_waiting: &'static str,
    /// Suffix of "2.0× realtime".
    pub realtime_suffix: &'static str,
    /// Failed row, inviting the hover that now works.
    pub status_failed_hint: &'static str,
    /// Smart-skip row.
    pub status_already_optimal: &'static str,
    /// Cancelled row.
    pub status_cancelled: &'static str,
    /// Running row before the first frame (fraction unknown).
    pub progress_preparing: &'static str,
    /// Running row once the frame loop is done and the container is
    /// being finalized.
    pub progress_finalizing: &'static str,
    /// "left" in "3m 20s left".
    pub progress_left: &'static str,
    /// Hours suffix in an ETA ("1h 05m").
    pub time_hours: &'static str,
    /// Minutes suffix in an ETA.
    pub time_minutes: &'static str,
    /// Seconds suffix in an ETA.
    pub time_seconds: &'static str,
    /// Hover over the per-file heuristic insight line.
    pub insight_hover: &'static str,

    // ---- advanced tab -----------------------------------------------
    /// "Hardware" section heading.
    pub adv_hardware: &'static str,
    /// Worker-slots slider label.
    pub adv_parallel: &'static str,
    /// "driver cap" in the concurrency note.
    pub adv_driver_cap: &'static str,
    /// "concurrent NVENC sessions" in the concurrency note.
    pub adv_sessions: &'static str,
    /// Encoder-section heading.
    pub adv_encoder_section: &'static str,
    /// Codec combo label.
    pub adv_codec: &'static str,
    /// Compression-strategy combo label.
    pub adv_strategy: &'static str,
    /// Encoder-preset combo label.
    pub adv_preset: &'static str,
    /// Codec choice: automatic.
    pub codec_auto: &'static str,
    /// Codec choice: AV1 (a format name — identical across languages).
    pub codec_av1: &'static str,
    /// Codec choice: HEVC (a format name — identical across languages).
    pub codec_hevc: &'static str,
    /// Strategy: visually lossless.
    pub strategy_vl: &'static str,
    /// Strategy: balanced.
    pub strategy_balanced: &'static str,
    /// Strategy: aggressive.
    pub strategy_aggressive: &'static str,
    /// Note under the visually-lossless strategy.
    pub strategy_note_vl: &'static str,
    /// Note under the balanced strategy.
    pub strategy_note_balanced: &'static str,
    /// Note under the aggressive strategy.
    pub strategy_note_aggressive: &'static str,
    /// Preset: fast.
    pub effort_fast: &'static str,
    /// Preset: balanced.
    pub effort_balanced: &'static str,
    /// Preset: best quality.
    pub effort_quality: &'static str,
    /// Preset: maximum (slowest).
    pub effort_maximum: &'static str,
    /// Smart-skip checkbox label.
    pub smart_skip: &'static str,
    /// Smart-skip checkbox hover.
    pub smart_skip_hover: &'static str,
    /// Warning shown when smart-skip is off.
    pub smart_skip_off: &'static str,
    /// Raw-parameters preview label.
    pub raw_params: &'static str,
    /// "engine:" prefix of the backend badge.
    pub engine_prefix: &'static str,

    // ---- hardware-error screen --------------------------------------
    /// Headline when no usable GPU was found.
    pub hw_no_gpu: &'static str,
    /// Body explaining the hardware requirement.
    pub hw_body: &'static str,
    /// "probe said:" prefix before the raw diagnostic.
    pub hw_probe_prefix: &'static str,
    /// Hint pointing at the terminal diagnosis.
    pub hw_report_hint: &'static str,
}

/// English — the source language.
pub const EN: Strings = Strings {
    tab_dashboard: "Dashboard",
    tab_advanced: "Advanced",
    pause: "Pause",
    resume: "Resume",
    paused_badge: "PAUSED",
    pause_hover: "Pause: running encodes finish, nothing new starts \
                  (frees the GPU without losing the queue)",
    gpu_meter_hover: "How busy the video-encoder chip is right now",
    language_hover: "Язык / Language",

    drop_headline_dnd: "Drop videos or folders here",
    drop_headline_nodnd: "Add videos to compress them",
    add_media: "➕  Add media…",
    add_media_busy: "waiting for the file dialog…",
    tagline: "Visually Lossless (AV1) · audio kept untouched",
    dnd_unavailable: "drag & drop is unavailable on Wayland (upstream \
                      limitation) — use the button, `sovmr FILE...`, or \
                      restart with `sovmr --x11`",
    strip_more_dnd: "…or drop more files to add them to the queue",
    strip_more_nodnd: "…to add more files to the queue",

    word_of: "of",
    word_done: "done",
    word_running: "running",
    word_finishing: "finishing",
    word_paused: "paused",
    cancel_all: "Cancel all",

    space_saved_prefix: "Space saved:",
    no_space_saved: "No space saved yet",
    savings_hover: "Cyan: what the completed files occupy now · Green: \
                    reclaimed disk space",

    status_waiting: "Waiting",
    realtime_suffix: "× realtime",
    status_failed_hint: "Failed — hover for why",
    status_already_optimal: "Already optimal",
    status_cancelled: "Cancelled",
    progress_preparing: "Preparing…",
    progress_finalizing: "Finalizing…",
    progress_left: "left",
    time_hours: "h",
    time_minutes: "m",
    time_seconds: "s",
    insight_hover: "What the heuristic engine measured on this source and the \
                    bitrate ceiling it will enforce — the hardware is physically \
                    forbidden from exceeding it.",

    adv_hardware: "Hardware",
    adv_parallel: "Parallel encodes (worker slots)",
    adv_driver_cap: "driver cap",
    adv_sessions: "concurrent NVENC sessions",
    adv_encoder_section: "Encoder (applies to newly added files)",
    adv_codec: "Codec",
    adv_strategy: "Compression strategy",
    adv_preset: "Encoder preset",
    codec_auto: "Auto (recommended)",
    codec_av1: "AV1",
    codec_hevc: "HEVC (H.265)",
    strategy_vl: "Visually Lossless",
    strategy_balanced: "Balanced",
    strategy_aggressive: "Aggressive",
    strategy_note_vl: "indistinguishable from the source — still never larger than it",
    strategy_note_balanced: "the sweet spot: capped at 60% of the source's bitrate",
    strategy_note_aggressive: "smallest files: capped at 40% of the source's bitrate",
    effort_fast: "Fast",
    effort_balanced: "Balanced",
    effort_quality: "Best quality (default)",
    effort_maximum: "Maximum (slowest)",
    smart_skip: "Smart Skip — protect already compressed files",
    smart_skip_hover: "On: files already in the target codec at a low bitrate are \
                       marked \"already optimal\" instead of being re-encoded \
                       (re-encoding them only loses quality). Off: everything you \
                       add is encoded, no exceptions.",
    smart_skip_off: "off — every file will be re-encoded, including already \
                     compact AV1/HEVC sources",
    raw_params: "Raw parameters (what will actually run)",
    engine_prefix: "engine:",

    hw_no_gpu: "No compatible GPU found",
    hw_body: "sovmr drives NVIDIA's NVDEC/NVENC video silicon directly.\n\
              It needs an NVIDIA GPU (RTX 20-series or newer; RTX 40-series \
              for AV1) with the proprietary driver installed.",
    hw_probe_prefix: "probe said:",
    hw_report_hint: "run `sovmr --report` in a terminal for the full diagnosis",
};

/// Russian.
pub const RU: Strings = Strings {
    tab_dashboard: "Панель",
    tab_advanced: "Дополнительно",
    pause: "Пауза",
    resume: "Продолжить",
    paused_badge: "ПАУЗА",
    pause_hover: "Пауза: текущие кодирования завершатся, новые не начнутся \
                  (освобождает GPU, не теряя очередь)",
    gpu_meter_hover: "Насколько сейчас загружен видеокодер",
    language_hover: "Язык / Language",

    drop_headline_dnd: "Перетащите сюда видео или папки",
    drop_headline_nodnd: "Добавьте видео, чтобы сжать их",
    add_media: "➕  Добавить медиа…",
    add_media_busy: "ожидание диалога выбора файлов…",
    tagline: "Визуально без потерь (AV1) · звук сохраняется как есть",
    dnd_unavailable: "перетаскивание недоступно в Wayland (ограничение \
                      среды) — используйте кнопку, `sovmr FILE...` или \
                      перезапуск с `sovmr --x11`",
    strip_more_dnd: "…или перетащите ещё файлы, чтобы добавить их в очередь",
    strip_more_nodnd: "…чтобы добавить ещё файлы в очередь",

    word_of: "из",
    word_done: "готово",
    word_running: "в работе",
    word_finishing: "завершается",
    word_paused: "пауза",
    cancel_all: "Отменить всё",

    space_saved_prefix: "Сэкономлено:",
    no_space_saved: "Пока ничего не сэкономлено",
    savings_hover: "Голубой: сколько занимают готовые файлы сейчас · \
                    Зелёный: возвращённое место на диске",

    status_waiting: "Ожидание",
    realtime_suffix: "× реального времени",
    status_failed_hint: "Ошибка — наведите, чтобы узнать почему",
    status_already_optimal: "Уже оптимально",
    status_cancelled: "Отменено",
    progress_preparing: "Подготовка…",
    progress_finalizing: "Финализация…",
    progress_left: "осталось",
    time_hours: "ч",
    time_minutes: "м",
    time_seconds: "с",
    insight_hover: "Что движок эвристики измерил на этом источнике и какой \
                    потолок битрейта он применит — оборудованию физически \
                    запрещено его превышать.",

    adv_hardware: "Оборудование",
    adv_parallel: "Параллельных кодирований (рабочих слотов)",
    adv_driver_cap: "лимит драйвера",
    adv_sessions: "одновременных сессий NVENC",
    adv_encoder_section: "Кодировщик (применяется к новым файлам)",
    adv_codec: "Кодек",
    adv_strategy: "Стратегия сжатия",
    adv_preset: "Пресет кодировщика",
    codec_auto: "Авто (рекомендуется)",
    codec_av1: "AV1",
    codec_hevc: "HEVC (H.265)",
    strategy_vl: "Визуально без потерь",
    strategy_balanced: "Сбалансированно",
    strategy_aggressive: "Агрессивно",
    strategy_note_vl: "неотличимо от источника — и всё равно не больше него",
    strategy_note_balanced: "золотая середина: потолок 60% битрейта источника",
    strategy_note_aggressive: "минимальный размер: потолок 40% битрейта источника",
    effort_fast: "Быстро",
    effort_balanced: "Сбалансированно",
    effort_quality: "Лучшее качество (по умолчанию)",
    effort_maximum: "Максимум (самое медленное)",
    smart_skip: "Умный пропуск — беречь уже сжатые файлы",
    smart_skip_hover: "Вкл.: файлы, уже в целевом кодеке с низким битрейтом, \
                       помечаются «уже оптимально» вместо перекодирования \
                       (перекодирование лишь теряет качество). Выкл.: всё, что \
                       вы добавили, кодируется без исключений.",
    smart_skip_off: "выкл. — каждый файл будет перекодирован, включая уже \
                     компактные источники AV1/HEVC",
    raw_params: "Сырые параметры (что действительно запустится)",
    engine_prefix: "движок:",

    hw_no_gpu: "Совместимый GPU не найден",
    hw_body: "sovmr управляет видео-кремнием NVIDIA NVDEC/NVENC напрямую.\n\
              Нужен GPU NVIDIA (RTX 20-й серии или новее; RTX 40-й серии \
              для AV1) с установленным проприетарным драйвером.",
    hw_probe_prefix: "проба сообщила:",
    hw_report_hint: "запустите `sovmr --report` в терминале для полной диагностики",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_is_the_default() {
        // A machine with no preference must read the source language,
        // not surprise a user with a language they did not choose.
        assert_eq!(Lang::default(), Lang::En);
    }

    #[test]
    fn the_toggle_leads_to_the_other_language_and_names_it() {
        assert_eq!(Lang::En.other(), Lang::Ru);
        assert_eq!(Lang::Ru.other(), Lang::En);
        // The badge names where the button leads, so it is the code of
        // the *other* language.
        assert_eq!(Lang::En.switch_badge(), "RU");
        assert_eq!(Lang::Ru.switch_badge(), "EN");
    }

    #[test]
    fn a_saved_code_round_trips_and_junk_falls_back() {
        // Persistence depends on this: what `code()` writes, `from_code()`
        // must read back to the same language, for every language.
        for lang in [Lang::En, Lang::Ru] {
            assert_eq!(Lang::from_code(lang.code()), Some(lang));
        }
        // Surrounding whitespace (a file with a trailing newline) is fine.
        assert_eq!(Lang::from_code("ru\n"), Some(Lang::Ru));
        // Anything else is not a language — the caller uses its default.
        assert_eq!(Lang::from_code("fr"), None);
        assert_eq!(Lang::from_code(""), None);
    }

    #[test]
    fn both_languages_are_actually_filled_in_and_differ() {
        // The struct-of-fields design already forbids a missing field at
        // compile time; this guards the softer failure of a field copied
        // from English and never translated. Language-neutral tokens
        // (`AV1`, `EN`, punctuation) are exempt by construction — they
        // are not in this table.
        let en = Lang::En.strings();
        let ru = Lang::Ru.strings();
        assert_ne!(en.tab_dashboard, ru.tab_dashboard);
        assert_ne!(en.pause, ru.pause);
        assert_ne!(en.add_media, ru.add_media);
        assert_ne!(en.progress_finalizing, ru.progress_finalizing);
        assert_ne!(en.status_failed_hint, ru.status_failed_hint);
        // The one field that is meant to be identical in both: the
        // language toggle's tooltip is bilingual on purpose.
        assert_eq!(en.language_hover, ru.language_hover);
    }

    #[test]
    fn no_message_is_empty() {
        // An empty label is a blank spot in the window — never right.
        for lang in [Lang::En, Lang::Ru] {
            let s = lang.strings();
            for field in [
                s.tab_dashboard,
                s.tab_advanced,
                s.pause,
                s.resume,
                s.paused_badge,
                s.pause_hover,
                s.gpu_meter_hover,
                s.language_hover,
                s.drop_headline_dnd,
                s.drop_headline_nodnd,
                s.add_media,
                s.add_media_busy,
                s.tagline,
                s.dnd_unavailable,
                s.strip_more_dnd,
                s.strip_more_nodnd,
                s.word_of,
                s.word_done,
                s.word_running,
                s.word_finishing,
                s.word_paused,
                s.cancel_all,
                s.space_saved_prefix,
                s.no_space_saved,
                s.savings_hover,
                s.status_waiting,
                s.realtime_suffix,
                s.status_failed_hint,
                s.status_already_optimal,
                s.status_cancelled,
                s.progress_preparing,
                s.progress_finalizing,
                s.progress_left,
                s.time_hours,
                s.time_minutes,
                s.time_seconds,
                s.insight_hover,
                s.adv_hardware,
                s.adv_parallel,
                s.adv_driver_cap,
                s.adv_sessions,
                s.adv_encoder_section,
                s.adv_codec,
                s.adv_strategy,
                s.adv_preset,
                s.codec_auto,
                s.codec_av1,
                s.codec_hevc,
                s.strategy_vl,
                s.strategy_balanced,
                s.strategy_aggressive,
                s.strategy_note_vl,
                s.strategy_note_balanced,
                s.strategy_note_aggressive,
                s.effort_fast,
                s.effort_balanced,
                s.effort_quality,
                s.effort_maximum,
                s.smart_skip,
                s.smart_skip_hover,
                s.smart_skip_off,
                s.raw_params,
                s.engine_prefix,
                s.hw_no_gpu,
                s.hw_body,
                s.hw_probe_prefix,
                s.hw_report_hint,
            ] {
                assert!(!field.trim().is_empty());
            }
        }
    }
}
