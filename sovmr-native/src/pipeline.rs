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

//! The synchronous zero-copy transcode loop. Single-threaded by design:
//! CUDA contexts are thread-bound, and NVDEC's parser fires its
//! callbacks on the feeding thread — so one worker thread owns the whole
//! GPU pipeline end to end (one owner, no locks).
//!
//! **Lock-free VRAM registration cache** (measured cost):
//! `nvEncRegisterResource` is the expensive per-surface step, while
//! NVDEC recycles a small ring of decode surfaces. Each distinct
//! `CUdeviceptr` is registered once and cached in a plain
//! `HashMap<u64, RegisteredResource>` owned by the single pipeline
//! thread — mutation happens through `&mut`, so there is no `Mutex`,
//! no `Arc`, no atomic in the hot loop at all.
//!
//! **Sovereign MP4 output**: both targets are muxed on the fly into a
//! fast-start MP4 (`moov` before `mdat`) — AV1 as `av01`+`av1C`, HEVC as
//! `hvc1`+`hvcC` (NVENC's Annex B output is re-packetized to 4-byte NAL
//! length prefixes, parameter sets harvested into the config record) —
//! with the source's audio track passed through bit-exactly, interleaved
//! by timestamp. A source whose audio codec cannot be passed through
//! degrades to a video-only output with a warning instead of failing the
//! job.
//!
//! ACID output discipline matches the historical FFmpeg backend: bytes
//! stream into `<output>.part` (plus a `.part.mdat` spool for MP4),
//! committed by atomic rename only on success, deleted on cancel/failure.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use sovmr_cuda::CudaError;
use sovmr_mp4::demux::{self, DemuxError, VideoCodec};
use sovmr_mp4::hevc;
use sovmr_mp4::mux::{self, Mp4Muxer, MuxVideoCodec, VideoMuxParams};
use sovmr_nvdec::parser::{ParserCallbacks, VideoFormat};
use sovmr_nvdec::{DecodeCodec, Decoder, DecoderConfig, NvDec, NvdecError};
use sovmr_nvenc::encoder::{EncodeCodec, EncodePreset, EncoderInit};
use sovmr_nvenc::{NvencError, RegisteredResource, VramFrame};
use sovmr_os::MappedFile;

/// Media time each audio track accumulates before its packets are
/// released as one chunk. One second is the conventional interleaving
/// granularity: small enough that a player reading linearly always has
/// every track buffered, large enough that the chunk tables stay a
/// rounding error next to the media (see [`drain_audio_feeds`]).
const INTERLEAVE_CHUNK_SECONDS: u64 = 1;

/// Suffix for in-flight output (same convention as the FFmpeg backend).
const PARTIAL_SUFFIX: &str = ".part";
/// Suffix (appended to the partial path) for the muxer's mdat spool.
const SPOOL_SUFFIX: &str = ".mdat";

/// Peak disk copies of the output that exist at once.
///
/// Fast start means the payload cannot be written in place: samples go
/// to the `.mdat` spool while the sizes needed to build `moov` are still
/// being discovered, and `finalize` then writes `ftyp` + `moov` + the
/// `mdat` header to `.part` and drains the spool into it. Between those
/// two moments both files exist, so the volume must hold twice the
/// output — the number a pre-flight check has to use.
const PEAK_OUTPUT_COPIES: u64 = 2;

/// Safety factor on the estimated output size, in percent.
///
/// Rate control tracks its ceiling but is not bound by it frame to
/// frame, and the estimate ignores container overhead entirely. The
/// margin exists so an honest near-miss does not refuse a runnable job;
/// it is deliberately small, because the failure this guards against —
/// filling the volume two hours into a twelve-hour encode — is far more
/// expensive than refusing early.
const SIZE_ESTIMATE_MARGIN_PCT: u64 = 15;

/// Fallback nominal frame rate when the container carries no usable
/// timing (rate-control pacing + synthesized durations).
const NOMINAL_FPS: u32 = 30;
/// Keyframe (IDR) interval in seconds — the timeline-seek granularity.
/// Ten seconds: the old 2 s GOP forced an IDR every 60 frames, and
/// IDRs are the most expensive frames there are — on static scenes that
/// is pure waste. 10 s still lands a sync sample often enough that
/// seeking feels immediate (players decode forward from the nearest
/// one), while giving the encoder 5x more inter-frame redundancy to
/// exploit.
const GOP_SECONDS: u32 = 10;
/// ...and never fewer than this many frames, whatever the frame rate.
const GOP_MIN_FRAMES: u32 = 250;
/// How far a scene's quality target may move from the job's own.
///
/// Small on purpose. The point is to lend bits from easy scenes to hard
/// ones, not to encode two different films: an unbounded spread starves
/// a scene into visible mush while the average still reads well.
const QUALITY_SPREAD: u8 = 4;
/// Two-pass full-resolution rate control: the first pass measures frame
/// complexity so the second spends the (hard-capped) bits where the eye
/// sees them. Essential once the heuristic ceiling gets tight.
const MULTI_PASS: bool = true;
/// `frameIntervalP`: 4 = up to 3 B-frames between references (
/// measured −46 % HEVC / −58 % AV1 on this pipeline).
const FRAME_INTERVAL_P: u32 = 4;

/// Rate-control lookahead depth: fully implemented end to end —
/// `EncoderInit::lookahead_depth`, the init feature ladder, the staging
/// pool, EOS padding and flush pump sized for a 32-frame window — but
/// PINNED TO 0 on driver 580.x after empirical falsification: in sync
/// mode the driver loses a nondeterministic subset of outputs
/// (73/117/121 of 120 across identical runs) and its `doNotWait`
/// bitstream lock can block forever inside libnvidia-encode (stack-
/// verified). A rate-control optimization that loses frames or hangs is
/// not an optimization. Flip to 32 to re-evaluate on newer drivers; the
/// e2e gates assert exact frame counts and will catch regressions.
const HEVC_LOOKAHEAD_DEPTH: u16 = 0;
const AV1_LOOKAHEAD_DEPTH: u16 = 0;
/// Output bitstream buffers in flight. Must exceed the encoder's
/// reordering + lookahead depth (32 + 3 B-frames); 48 leaves slack.
const BITSTREAM_POOL: usize = 48;
/// Parser surface hint before the sequence callback commits the real
/// number (the callback value overrides this).
const PARSER_SURFACE_HINT: u32 = 8;
/// Pause between EOS flush pump rounds while the hardware drains.
const EOS_PUMP_SLEEP_MS: u64 = 2;
/// Patience budget for the lookahead flush pump (rounds x sleep = 1 s).
const EOS_PUMP_ROUNDS: u32 = 500;

/// Reduce a frame-rate fraction to lowest terms.
fn reduce(num: u32, den: u32) -> (u32, u32) {
    let (mut a, mut b) = (num, den.max(1));
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    let g = a.max(1);
    (num / g, den.max(1) / g)
}

/// Estimated output size in bytes: the encoded video leg at its rate
/// ceiling plus the audio tracks, which passthrough copies byte for
/// byte and are therefore known exactly.
///
/// `max_bitrate` of zero means uncapped, and an uncapped encode has no
/// analytic ceiling — the source size stands in, since re-encoding to a
/// more efficient codec that *grew* the file would be a failure of the
/// tool on its own terms.
fn estimate_output_bytes(
    duration_secs: f64,
    max_bitrate: u32,
    audio_bytes: u64,
    source_bytes: u64,
) -> u64 {
    let video = if max_bitrate == 0 {
        source_bytes
    } else {
        (duration_secs * f64::from(max_bitrate) / 8.0) as u64
    };
    let raw = video.saturating_add(audio_bytes);
    raw.saturating_add(raw / 100 * SIZE_ESTIMATE_MARGIN_PCT)
}

/// What to encode and how — the native mirror of core's `EncodeSettings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSettings {
    /// Target codec.
    pub codec: EncodeCodec,
    /// Constant-quality target (0–51).
    pub cq: u8,
    /// Hard bitrate ceiling in bits/s from the heuristic engine
    /// (0 = derive the theoretical geometry cap here).
    pub max_bitrate: u32,
    /// Encoder effort preset.
    pub preset: EncodePreset,
    /// How B-frames and references are configured.
    ///
    /// Carried so a sweep can vary it: both fields live in the codec
    /// union and neither has ever been set, so neither has ever been
    /// measured.
    pub references: sovmr_nvenc::encoder::ReferencePlan,
}

/// Counters from a finished run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeStats {
    /// Frames encoded.
    pub frames: u64,
    /// Bytes of output written (the complete fast-start MP4).
    pub output_bytes: u64,
    /// What the user should know about this run, when there is
    /// anything: audio re-encoded, a track that could not come along, an
    /// extra the output container cannot hold. The transcode succeeded
    /// either way.
    ///
    /// Not only degradations. A lossless track compressed to AAC is not
    /// a failure — it is what the tool was asked to do — but it is a
    /// material change to somebody's file and saying so is the least
    /// this owes them.
    pub warning: Option<String>,
}

/// Terminal state of a pipeline run.
#[derive(Debug, PartialEq, Eq)]
pub enum TranscodeOutcome {
    /// Output committed to its final path.
    Completed(TranscodeStats),
    /// Cancel flag observed; partial output rolled back.
    Cancelled,
}

/// Everything that can go wrong across the pipeline, with the failing
/// stage named. All variants render symbolically.
#[derive(Debug)]
pub enum NativeError {
    /// Container/demux failure.
    Demux(DemuxError),
    /// CUDA bring-up failure.
    Cuda(CudaError),
    /// Decoder-side failure.
    Nvdec(NvdecError),
    /// Encoder-side failure.
    Nvenc(NvencError),
    /// Output I/O failure.
    Io(std::io::Error),
    /// Logic-level pipeline failure (e.g. stream/decoder mismatch).
    Pipeline(String),
    /// The destination volume cannot hold the estimated output. Refused
    /// before any work starts rather than discovered at hour two.
    NoRoom {
        /// Bytes the job is estimated to need, spool included.
        needed: u64,
        /// Bytes actually available on the destination volume.
        available: u64,
    },
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeError::Demux(e) => write!(f, "demux: {e}"),
            NativeError::Cuda(e) => write!(f, "cuda: {e}"),
            NativeError::Nvdec(e) => write!(f, "nvdec: {e}"),
            NativeError::Nvenc(e) => write!(f, "nvenc: {e}"),
            NativeError::Io(e) => write!(f, "output i/o: {e}"),
            NativeError::Pipeline(m) => write!(f, "pipeline: {m}"),
            NativeError::NoRoom { needed, available } => write!(
                f,
                "not enough free space: the job needs about {} MiB (output plus its spool) \
                 but only {} MiB is available on the destination volume",
                needed / (1024 * 1024),
                available / (1024 * 1024)
            ),
        }
    }
}

impl From<DemuxError> for NativeError {
    fn from(e: DemuxError) -> Self {
        NativeError::Demux(e)
    }
}
impl From<CudaError> for NativeError {
    fn from(e: CudaError) -> Self {
        NativeError::Cuda(e)
    }
}
impl From<NvdecError> for NativeError {
    fn from(e: NvdecError) -> Self {
        NativeError::Nvdec(e)
    }
}
impl From<NvencError> for NativeError {
    fn from(e: NvencError) -> Self {
        NativeError::Nvenc(e)
    }
}
impl From<std::io::Error> for NativeError {
    fn from(e: std::io::Error) -> Self {
        NativeError::Io(e)
    }
}

/// One source audio track, streamed rather than materialised.
///
/// The packets are zero-copy slices of the mapped input, released into
/// the muxer by timestamp as video frames come out of the encoder. The
/// list used to be collected up front, which cost roughly 24 bytes per
/// packet: at twelve hours and six tracks that is about 290 MB of
/// bookkeeping for data already sitting in the mapping. A cursor over
/// the demuxer's own iterator costs one packet.
struct AudioFeed<'a> {
    /// Index of this track in the muxer's audio order.
    index: usize,
    /// The source's packet cursor, pulled one at a time. Which
    /// container it walks is [`crate::source`]'s business, not this
    /// loop's.
    packets: crate::source::AudioPackets<'a>,
    /// The packet read but not yet due — the one-element lookahead a
    /// timestamp comparison needs.
    peeked: Option<(&'a [u8], u32, u64)>,
    /// Set for a track being re-encoded, absent for one passed through.
    compressing: Option<Compressing>,
    timescale: u32,
    /// Media time already released, in this track's units. Packets are
    /// handed over in slabs rather than one at a time; see
    /// [`INTERLEAVE_CHUNK_SECONDS`].
    released_upto: u64,
    /// Set once the track's samples run out.
    drained: bool,
}

/// A track being turned into AAC as it streams past.
struct Compressing {
    compressor: crate::audio::TrackCompressor,
    /// Frames finished but not yet due.
    ready: std::collections::VecDeque<Vec<u8>>,
    /// Decode time of the next frame, in samples.
    next_dts: u64,
    /// Set once the tail has been flushed, so it is flushed once.
    finished: bool,
}

/// A packet on its way to the muxer.
///
/// Passthrough audio is a slice of the mapped source and is never
/// copied; a re-encoded frame is this project's own bytes and has
/// nowhere else to live. Distinguishing them keeps the zero-copy path
/// zero-copy, which is the whole reason the source is mapped.
enum Payload<'a> {
    /// Borrowed from the mapped file.
    Borrowed(&'a [u8]),
    /// Produced by this project's encoder.
    Owned(Vec<u8>),
}

impl Payload<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Payload::Borrowed(data) => data,
            Payload::Owned(data) => data,
        }
    }
}

impl<'a> AudioFeed<'a> {
    /// Decode time of the next packet, or `None` at end of track.
    ///
    /// Errors from the demuxer end the track rather than the job: a
    /// truncated audio tail must not throw away an otherwise good
    /// encode (graceful degradation). A track being
    /// compressed ends the same way — a packet that will not decode
    /// stops that track and leaves the rest of the file alone.
    fn peek_dts(&mut self) -> Option<u64> {
        match &mut self.compressing {
            None => {
                if self.peeked.is_none() && !self.drained {
                    match self.packets.next_packet() {
                        Some(p) => self.peeked = Some(p),
                        None => self.drained = true,
                    }
                }
                self.peeked.map(|(_, _, dts)| dts)
            }
            Some(_) => {
                self.fill_frames();
                let state = self.compressing.as_ref().expect("just matched");
                (!state.ready.is_empty()).then_some(state.next_dts)
            }
        }
    }

    /// Push source packets through the compressor until it yields a
    /// frame, or the track runs out and its tail is flushed.
    fn fill_frames(&mut self) {
        loop {
            let state = self
                .compressing
                .as_mut()
                .expect("only called when compressing");
            if !state.ready.is_empty() || state.finished {
                return;
            }
            match self.packets.next_packet() {
                Some((data, _, _)) => {
                    let state = self.compressing.as_mut().expect("still compressing");
                    match state.compressor.push(data) {
                        Ok(frames) => state.ready.extend(frames),
                        Err(_) => {
                            // Stop this track where it stopped making
                            // sense; the rest of the file is unaffected.
                            state.finished = true;
                            self.drained = true;
                            return;
                        }
                    }
                }
                None => {
                    let state = self.compressing.as_mut().expect("still compressing");
                    state.finished = true;
                    if let Ok(frames) = state.compressor.finish() {
                        state.ready.extend(frames);
                    }
                    self.drained = true;
                    return;
                }
            }
        }
    }

    fn take(&mut self) -> Option<(Payload<'a>, u32, u64)> {
        match &mut self.compressing {
            None => {
                self.peek_dts();
                self.peeked
                    .take()
                    .map(|(data, duration, dts)| (Payload::Borrowed(data), duration, dts))
            }
            Some(_) => {
                self.peek_dts();
                let state = self.compressing.as_mut().expect("just matched");
                let frame = state.ready.pop_front()?;
                let dts = state.next_dts;
                state.next_dts += AAC_FRAME_SAMPLES;
                Some((Payload::Owned(frame), AAC_FRAME_SAMPLES as u32, dts))
            }
        }
    }
}

/// Samples one AAC frame covers, which is also its duration in a track
/// whose timescale is its sampling rate.
const AAC_FRAME_SAMPLES: u64 = 1024;

/// The output muxer, whichever container the source arrived in.
///
/// The two are not interchangeable in shape — MP4 wants durations in a
/// track timescale and a composition time, Matroska wants absolute
/// milliseconds — so this converts rather than pretending they are the
/// same call.
enum OutputMuxer {
    /// ISOBMFF, fast-start.
    Mp4(Mp4Muxer),
    /// Matroska. Its tracks must be declared with their codec setup up
    /// front, and for video that setup only exists once the encoder has
    /// produced its first keyframe — so the muxer is built then, and
    /// audio arriving before it is held.
    Mkv {
        /// `None` until the first keyframe supplies the codec config.
        muxer: Option<sovmr_mkv::mux::MkvMuxer>,
        /// Where the file will be written.
        path: std::path::PathBuf,
        /// Audio tracks to declare, known from the source.
        audio: Vec<sovmr_mkv::mux::TrackSpec>,
        /// Video geometry for the track entry.
        width: u32,
        /// Video geometry for the track entry.
        height: u32,
        /// Ticks per second of the timestamps the sink hands over.
        timescale: u32,
        /// The source's title, which lives in `Info` rather than among
        /// the tags and so has to be handed over at creation.
        title: Option<String>,
        /// Chapters and tags from the source, written in front of the
        /// media — which is where a reader that has no seek head to
        /// follow will look, and nowhere else.
        marks: (
            Vec<sovmr_mkv::mux::ChapterSpec>,
            Vec<sovmr_mkv::mux::TagSpec>,
        ),
        /// Frames held while the muxer does not exist yet.
        pending: Vec<(u64, i64, i64, bool, Vec<u8>)>,
        /// What the source said its colours mean.
        colour: Option<sovmr_mp4::colour::Colour>,
    },
}

impl OutputMuxer {
    /// Milliseconds, from a timestamp in `timescale` ticks.
    fn to_ms(ticks: u64, timescale: u32) -> i64 {
        (u128::from(ticks) * 1000 / u128::from(timescale.max(1))) as i64
    }

    /// Append a video frame.
    ///
    /// `config` is the codec setup harvested from this frame when it is
    /// a keyframe — Matroska needs it before it can declare its tracks.
    fn add_video(
        &mut self,
        data: &[u8],
        duration: u32,
        keyframe: bool,
        pts: u64,
        config: Option<&[u8]>,
        codec_id: &str,
    ) -> std::io::Result<()> {
        match self {
            OutputMuxer::Mp4(m) => m.add_video_sample(data, duration, keyframe, pts),
            OutputMuxer::Mkv {
                muxer,
                title,
                marks,
                path,
                audio,
                width,
                height,
                timescale,
                pending,
                colour,
            } => {
                if muxer.is_none() {
                    let Some(config) = config else {
                        // No configuration yet, so no tracks can be
                        // declared. Hold the frame rather than write one
                        // the file has no way to describe.
                        pending.push((
                            1,
                            Self::to_ms(pts, *timescale),
                            Self::to_ms(u64::from(duration), *timescale),
                            keyframe,
                            data.to_vec(),
                        ));
                        return Ok(());
                    };
                    let mut specs = vec![sovmr_mkv::mux::TrackSpec {
                        number: 1,
                        codec_id: codec_id.to_string(),
                        codec_private: config.to_vec(),
                        video: true,
                        subtitle: false,
                        colour: *colour,
                        width: *width,
                        height: *height,
                        sample_rate: 0,
                        channels: 0,
                        language: "und".to_string(),
                        name: None,
                        enabled: true,
                    }];
                    specs.extend(audio.iter().cloned());
                    let mut m =
                        sovmr_mkv::mux::MkvMuxer::create_titled(path, &specs, title.as_deref())?;
                    // Before the first frame, because that is the only
                    // place a reader without a seek head will find them.
                    m.set_marks(marks.0.clone(), marks.1.clone())?;
                    for (t, ts, dur, key, bytes) in pending.drain(..) {
                        m.add_frame(t, ts, dur, key, &bytes)?;
                    }
                    *muxer = Some(m);
                }
                let m = muxer.as_mut().expect("just created");
                m.add_frame(
                    1,
                    Self::to_ms(pts, *timescale),
                    Self::to_ms(u64::from(duration), *timescale),
                    keyframe,
                    data,
                )
            }
        }
    }

    /// Append an audio packet from track `index`.
    fn add_audio(
        &mut self,
        index: usize,
        data: &[u8],
        duration: u32,
        timestamp: u64,
        timescale: u32,
    ) -> std::io::Result<()> {
        match self {
            OutputMuxer::Mp4(m) => m.add_audio_sample(index, data, duration),
            OutputMuxer::Mkv { muxer, pending, .. } => {
                let number = 2 + index as u64;
                let ts = Self::to_ms(timestamp, timescale);
                let dur = Self::to_ms(u64::from(duration), timescale);
                match muxer {
                    Some(m) => m.add_frame(number, ts, dur, true, data),
                    // Audio can legitimately precede the first video
                    // keyframe; hold it rather than lose it.
                    None => {
                        pending.push((number, ts, dur, true, data.to_vec()));
                        Ok(())
                    }
                }
            }
        }
    }

    /// Close the file.
    fn finalize(self, mp4_path: &std::path::Path) -> std::io::Result<u64> {
        match self {
            OutputMuxer::Mp4(m) => m.finalize(mp4_path),
            OutputMuxer::Mkv { muxer, .. } => match muxer {
                Some(m) => m.finalize(),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "no video keyframe, so no Matroska tracks could be declared",
                )),
            },
        }
    }
}

/// Where encoded frames go: the sovereign fast-start muxer plus the
/// source timing and pending audio it interleaves.
struct Mp4Sink<'a> {
    muxer: OutputMuxer,
    /// Target codec: decides how encoder packets become MP4 samples.
    codec: EncodeCodec,
    audio: Vec<AudioFeed<'a>>,
    /// Source frame duration keyed by (shifted) presentation timestamp:
    /// the encoder reorders output with B-frames, but it echoes each
    /// frame's PTS back on the packet, so PTS is the stable join key.
    durations: HashMap<u64, u32>,
    /// Decode-time cursor of the video track (cumulative durations of
    /// samples written so far) — drives audio interleaving.
    v_dts: u64,
    v_timescale: u32,
    /// Highest legitimate (shifted) source PTS.
    ///
    /// Nothing should now arrive beyond it — the sacrificial EOS padding
    /// that used to is gone. The guard stays because a picture the
    /// container never asked for is exactly what a dangling reference is
    /// made of, and it costs one comparison to refuse.
    pts_ceiling: u64,
    /// PTS values already muxed. A duplicate presentation instant must
    /// never reach the container: two samples claiming one instant give
    /// a player no way to order them.
    muxed_pts: std::collections::HashSet<u64>,
    /// The AV1 sequence-header OBU harvested from the first keyframe.
    /// NVENC only attaches it to the FIRST IDR, but the AV1-in-ISOBMFF
    /// binding requires every sync sample to carry one — without it a
    /// player seeking to a later keyframe feeds the decoder an
    /// unparsable stream (the "seeking freezes" bug). Later keyframes
    /// that arrive bare get it prepended.
    seq_header: Option<Vec<u8>>,
    /// The `hvcC` record harvested from the first IDR, for a Matroska
    /// output's `CodecPrivate`. MP4 keeps its own copy inside the muxer.
    hevc_config: Option<Vec<u8>>,
    /// Coded geometry, needed to build `av1C`.
    width: u16,
    /// Coded geometry, needed to build `av1C`.
    height: u16,
    /// Reused buffer for sample repacketization (HEVC length prefixes,
    /// AV1 sequence-header injection).
    scratch: Vec<u8>,
}

impl Mp4Sink<'_> {
    /// Synthesized per-frame duration when the source table runs dry.
    fn fallback_duration(&self) -> u32 {
        (self.v_timescale / NOMINAL_FPS).max(1)
    }

    /// The sample duration for a packet, by its echoed PTS.
    fn duration_for(&self, pts: u64) -> u32 {
        self.durations
            .get(&pts)
            .copied()
            .unwrap_or_else(|| self.fallback_duration())
    }

    /// Add one encoded packet **in decode order**: harvest the codec
    /// configuration off keyframes, repacketize for the container,
    /// interleave due audio, record the PTS for `ctts`.
    ///
    /// * AV1: strip the temporal delimiter, samples are raw OBUs.
    /// * HEVC: NVENC emits Annex B — parameter sets go to `hvcC`, the
    ///   remaining NALs get 4-byte length prefixes (`hvc1` samples).
    ///
    /// Returns `false` when the packet was refused: past the last real
    /// presentation instant, or a duplicate of one already written.
    fn add_frame(&mut self, packet: &[u8], keyframe: bool, pts: u64) -> std::io::Result<bool> {
        if pts > self.pts_ceiling || !self.muxed_pts.insert(pts) {
            // Nothing legitimate lands here. Anything that does is a
            // picture no source frame asked for, and muxing it would put
            // a reference in the file that points nowhere.
            return Ok(false);
        }
        let data: &[u8] = match self.codec {
            EncodeCodec::Av1 => {
                let data = mux::strip_temporal_delimiter(packet);
                if keyframe {
                    match mux::extract_sequence_header(data) {
                        Some(seq) => {
                            // First IDR (NVENC attaches the header here):
                            // harvest it for av1C and for re-injection.
                            if let OutputMuxer::Mp4(m) = &mut self.muxer {
                                m.set_sequence_header(seq);
                            }
                            if self.seq_header.is_none() {
                                self.seq_header = Some(seq.to_vec());
                            }
                            data
                        }
                        None => match &self.seq_header {
                            // Later IDRs arrive bare: prepend the header
                            // so the sync sample is decodable standalone
                            // (AV1-ISOBMFF binding; seeking depends on it).
                            Some(seq) => {
                                self.scratch.clear();
                                self.scratch.extend_from_slice(seq);
                                self.scratch.extend_from_slice(data);
                                &self.scratch
                            }
                            None => data, // no header seen yet: pass through
                        },
                    }
                } else {
                    data
                }
            }
            EncodeCodec::Hevc => {
                if keyframe {
                    if let Some(ps) = hevc::extract_parameter_sets(packet) {
                        if self.hevc_config.is_none() {
                            self.hevc_config = sovmr_mp4::hevc::hvcc_record(&ps).ok();
                        }
                        if let OutputMuxer::Mp4(m) = &mut self.muxer {
                            m.set_hevc_parameter_sets(ps);
                        }
                    }
                }
                hevc::length_prefixed_sample(packet, &mut self.scratch);
                &self.scratch
            }
        };
        let duration = self.duration_for(pts);
        let dts = self.v_dts;
        let v_timescale = self.v_timescale;
        // The codec setup Matroska needs before it can declare tracks;
        // MP4 harvests the same facts through its own setters above.
        let (config, codec_id): (Option<Vec<u8>>, &str) = match self.codec {
            EncodeCodec::Av1 => (
                self.seq_header
                    .as_deref()
                    .map(|seq| mux::av1c_record(seq, self.width, self.height)),
                "V_AV1",
            ),
            EncodeCodec::Hevc => (self.hevc_config.clone(), "V_MPEGH/ISO/HEVC"),
        };
        let Mp4Sink { muxer, audio, .. } = self;
        drain_audio_feeds(muxer, audio, Some((dts, v_timescale)))?;
        muxer.add_video(data, duration, keyframe, pts, config.as_deref(), codec_id)?;
        self.v_dts = dts + u64::from(duration);
        Ok(true)
    }
}

/// Release due audio into the muxer, one slab per track.
///
/// `until` is the video decode cursor `(v_dts, v_timescale)`; `None`
/// means end of stream, flush everything.
///
/// **Why slabs and not packets.** The muxer starts a new chunk whenever
/// the track changes, so releasing each packet the instant it comes due
/// would interleave seven tracks packet by packet and produce a chunk
/// per packet: a twelve-hour file at 30 fps with six audio tracks would
/// declare on the order of nine million chunks, and `stsc`/`stco` would
/// dwarf the media. Holding each track back until a whole
/// [`INTERLEAVE_CHUNK_SECONDS`] is due turns that into one run per track
/// per second — about 300 000 chunks for the same file — while keeping
/// tracks close enough together that a player never seeks to stay fed.
fn drain_audio_feeds(
    muxer: &mut OutputMuxer,
    feeds: &mut [AudioFeed<'_>],
    until: Option<(u64, u32)>,
) -> std::io::Result<()> {
    for feed in feeds.iter_mut() {
        // The video cursor expressed in this track's units. Cross-
        // timescale without division, as the comparison below needs.
        let horizon = match until {
            None => u64::MAX,
            Some((v_dts, v_ts)) => {
                let v_ts = u64::from(v_ts.max(1));
                let scaled = u128::from(v_dts) * u128::from(feed.timescale) / u128::from(v_ts);
                u64::try_from(scaled).unwrap_or(u64::MAX)
            }
        };
        // Not enough has come due to be worth a chunk yet.
        let slab = u64::from(feed.timescale) * INTERLEAVE_CHUNK_SECONDS;
        if until.is_some() && horizon < feed.released_upto.saturating_add(slab) {
            continue;
        }
        while let Some(dts) = feed.peek_dts() {
            if dts > horizon {
                break;
            }
            let (data, duration, ts) = feed.take().expect("peek just succeeded");
            muxer.add_audio(feed.index, data.bytes(), duration, ts, feed.timescale)?;
        }
        feed.released_upto = horizon;
    }
    Ok(())
}

/// Lock pending output buffers in submission (= decode) order and hand
/// the packets to the muxer sink. Mid-stream (`blocking = false`) the
/// lookahead/reorder queues stay full, so readiness arrives one frame
/// at a time: drain until the encoder says "still mine" and stop. After
/// EOS (`blocking = true`) every submitted frame WILL complete, so
/// blocking locks flush the tail.
fn drain_pending(
    buffers: &[sovmr_nvenc::BitstreamBuffer<'_>],
    pending: &mut VecDeque<usize>,
    free: &mut VecDeque<usize>,
    sink: &mut Mp4Sink<'_>,
    frames: &mut u64,
    output_bytes: &mut u64,
    blocking: bool,
) -> Result<(), NativeError> {
    while let Some(index) = pending.pop_front() {
        let lock = if blocking {
            Some(buffers[index].lock()?)
        } else {
            buffers[index].try_lock()?
        };
        let Some(lock) = lock else {
            pending.push_front(index); // encoder still holds it
            break;
        };
        if lock.is_empty() {
            if blocking {
                // Final EOS drain: an empty buffer is a flush artifact
                // (padding absorbed the tail); nothing more will come.
                drop(lock);
                free.push_back(index);
                continue;
            }
            // Mid-stream, "locked but empty" means NOT READY YET: the
            // packet arrives later. Consuming the buffer now would let
            // a resubmission clobber it — put it back and wait.
            drop(lock);
            pending.push_front(index);
            break;
        }
        if sink.add_frame(&lock, lock.is_keyframe(), lock.output_time_stamp())? {
            *frames += 1;
            *output_bytes += lock.len() as u64;
        }
        drop(lock);
        free.push_back(index);
    }
    Ok(())
}

/// Mutable state shared between the three parser callbacks (single
/// thread — `RefCell` is exactly the right tool, no locks). The VRAM
/// registration cache lives here too: plain owned state behind `&mut`,
/// never behind a `Mutex` (the hot loop pays for a hash
/// lookup, not for synchronization it cannot need on one thread).
struct Shared<'a> {
    decoder: Option<Decoder<'a>>,
    /// NVDEC's *surface* dimensions (coded, alignment-padded — e.g.
    /// 640×368 for a 640×360 stream), set by the sequence callback.
    /// VRAM registration MUST use these: NVENC locates the NV12 chroma
    /// plane at `pitch × registered_height`, so registering the display
    /// height shifts every chroma read by the padding rows — the classic
    /// "colored band" artifact.
    surface: Option<(u32, u32)>,
    /// One `nvEncRegisterResource` per staging slot pointer, reused
    /// across every frame copied into that slot. DECLARED BEFORE the
    /// staging pool on purpose: drop order is declaration order, and
    /// every registration must be unregistered while the VRAM it points
    /// at is still allocated.
    registrations: HashMap<u64, RegisteredResource<'a>>,
    /// Staging frame pool, one slot per bitstream buffer: lookahead
    /// makes the encoder read an input for up to `lookaheadDepth` more
    /// frames after submission, while NVDEC recycles its small surface
    /// ring immediately — so each mapped NVDEC frame is DtoD-copied
    /// (VRAM→VRAM, never across PCIe) into the slot tied to its output
    /// buffer, which by construction stays untouched until that output
    /// is drained. Lazily allocated once the surface pitch is known.
    staging: Vec<Option<sovmr_cuda::DeviceBuffer<'a>>>,
    /// Bitstream-pool bookkeeping: indices of buffers not currently
    /// held by the encoder, and of submitted buffers awaiting output
    /// (the B-frame reorderer holds several at once; sync-mode contract:
    /// when a submit reports ready, every pending buffer is lockable in
    /// submission = decode order).
    free: VecDeque<usize>,
    pending: VecDeque<usize>,
    sink: Mp4Sink<'a>,
    frames: u64,
    output_bytes: u64,
    error: Option<NativeError>,
    /// Set when the job is being cancelled. The callbacks then accept
    /// everything the parser hands them and do nothing with it, so the
    /// parser can drain to a proper end-of-stream instead of being
    /// abandoned mid-picture — see the cancellation branch of the feed
    /// loop for why an orderly decoder shutdown is not optional.
    abandoned: bool,
}

/// Run the full sovereign transcode: `input` (MP4, H.264/HEVC, optional
/// AAC audio) → `output` (fast-start MP4 with AV1 or HEVC video +
/// bit-exact audio passthrough; unsupported source audio degrades to a
/// video-only file with [`TranscodeStats::warning`] set). Blocking; call
/// from a dedicated worker thread. `on_progress` receives the running
/// frame count; `cancel` is polled between samples.
///
/// # Errors
///
/// [`NativeError`] naming the failing stage; partial output (`.part`
/// and the mdat spool) is always rolled back on error or cancel.
pub fn transcode(
    input: &Path,
    output: &Path,
    settings: &NativeSettings,
    mut on_progress: impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<TranscodeOutcome, NativeError> {
    // Mapped, not read: a twelve-hour source is tens of gigabytes, and
    // the demuxer wants a `&[u8]` either way. The kernel pages in what
    // the sample tables and in-flight packets actually touch, so peak
    // memory tracks the working set instead of the file size.
    // `MappedFile::advise_sequential` is deliberately NOT called here:
    // it is an unmeasured guess until there is a genuinely large source
    // to profile it against, and that would be a blind guess.
    //
    // OPEN FAULT, recorded where it would be found. Since the source
    // became a mapping, an intermittent SIGSEGV has been seen inside a
    // libnvcuvid worker thread (four times in this file's history;
    // unmodified pre-mapping code: 0 in 26 runs). It is timing and
    // layout sensitive — measured rates across successive builds of the
    // SAME logic were 3/5, 0/28, 1/12, 0/25 — never reproduces under a
    // debugger, and is NOT diagnosed. Two things are known: the driver
    // is fed heap buffers (`annex_b`), never the mapping itself, so the
    // likely mechanism is a stale driver pointer that only becomes
    // fatal once `munmap` punches a hole in the address space; and
    // declaration order here keeps `file` alive until after every CUDA
    // and NVDEC object is dropped. If it fires again, capture a core
    // (`kernel.core_pattern` must be redirected away from apport, which
    // deletes them) and read the faulting address.
    let file = MappedFile::open(input)?;
    // Which container, decided by the file's leading bytes. From here
    // down this function names no format at all.
    let source = crate::source::Source::open(&file)?;
    let facts = source.video_facts();
    let width = facts.width;
    let height = facts.height;
    let decode_codec = match facts.codec {
        VideoCodec::H264 => DecodeCodec::H264,
        VideoCodec::Hevc => DecodeCodec::Hevc,
        VideoCodec::Av1 => {
            return Err(NativeError::Pipeline(
                "source is already AV1 — Smart Skip should have caught this".into(),
            ))
        }
    };
    let v_timescale = facts.timescale.max(1);

    // Source timing up front (cheap table walk, no payload access).
    // PTS is the join key across the whole pipeline: packets are fed to
    // the decoder with their presentation timestamp, NVDEC displays in
    // presentation order, NVENC echoes the value back on each (decode-
    // ordered) output packet, and the muxer recovers durations + ctts
    // from it. `pts_shift` lifts any negative source PTS to >= 0 once,
    // uniformly — the muxer normalizes the constant away.
    let mut v_timing: Vec<(i64, u32)> = source.video_timing(&file)?;
    let (fps_num, fps_den) = match v_timing.first() {
        // Reduced to lowest terms: a nanosecond timescale states 30 fps
        // as 1000000000/33333333, and handing an encoder a ratio that
        // large invites it to round somewhere we cannot see. 30/1 says
        // the same thing and cannot be misread.
        Some(&(_, dur)) if dur > 0 => reduce(v_timescale, dur),
        _ => (NOMINAL_FPS, 1),
    };
    // Taken before the list is drained below: the pre-flight size check
    // needs it, and by then the timing has been folded into maps.
    let source_duration_secs = source.duration_secs(&v_timing);
    let pts_shift = v_timing
        .iter()
        .map(|&(pts, _)| pts)
        .min()
        .filter(|min| *min < 0)
        .map_or(0, |min| -min);
    let mut durations = HashMap::with_capacity(v_timing.len());
    let mut pts_by_decode_order = Vec::with_capacity(v_timing.len());
    let mut pts_ceiling: u64 = 0;
    for (pts, duration) in v_timing.drain(..) {
        let shifted = (pts + pts_shift) as u64;
        durations.insert(shifted, duration);
        pts_ceiling = pts_ceiling.max(shifted);
        pts_by_decode_order.push(shifted);
    }

    // GPU bring-up, all on this thread (CUDA contexts are thread-bound).
    let cuda = sovmr_cuda::Cuda::load()?;
    let ctx = cuda.create_context(0)?;
    let nvdec = NvDec::load()?;
    let api = sovmr_nvenc::NvEncApi::load()?;
    let session = api.open_session(&ctx)?;
    // Keyframe every GOP_SECONDS: that interval IS the seek granularity
    // (players jump to the nearest stss sync sample).
    // Widened deliberately. `fps_num` is a timescale, and Matroska
    // sources carry nanoseconds — 1e9, which multiplied by ten seconds
    // overflows a u32 long before it becomes a frame count. The old
    // formula was correct only for the modest timescales MP4 happens to
    // use, which is the kind of assumption a second container finds.
    let gop_length =
        u32::try_from(u64::from(GOP_SECONDS) * u64::from(fps_num) / u64::from(fps_den.max(1)))
            .unwrap_or(u32::MAX)
            .max(GOP_MIN_FRAMES);
    // Peak-bitrate guard rail from the heuristic engine; a zero means
    // the caller had no plan — fall back to the pure geometry ceiling.
    let max_bit_rate = if settings.max_bitrate > 0 {
        settings.max_bitrate
    } else {
        sovmr_core::heuristics::theoretical_max_bitrate(width, height, fps_num, fps_den)
    };
    let rc = session.initialize(&EncoderInit {
        codec: settings.codec,
        width,
        height,
        fps_num,
        fps_den,
        cq: settings.cq,
        gop_length,
        max_bit_rate,
        frame_interval_p: FRAME_INTERVAL_P,
        lookahead_depth: match settings.codec {
            EncodeCodec::Av1 => AV1_LOOKAHEAD_DEPTH,
            EncodeCodec::Hevc => HEVC_LOOKAHEAD_DEPTH,
        },
        multi_pass: MULTI_PASS,
        preset: settings.preset,
        references: settings.references,
        // The same description the container will carry. A decoder
        // reads the stream's own first, so a file stating BT.2020 in a
        // box and nothing in its bitstream was still being guessed at.
        colour: source
            .video_colour()
            .filter(sovmr_mp4::colour::Colour::is_meaningful)
            .map(|c| sovmr_nvenc::encoder::StreamColour {
                primaries: c.primaries,
                transfer: c.transfer,
                matrix: c.matrix,
                full_range: c.full_range,
            }),
    })?;
    // The accepted rate-control facts, loudly: a silent driver fallback
    // to bitrate-driven defaults is the classic file-bloat failure.
    eprintln!(
        "sovmr: nvenc init [{}]: {rc} ({}x{} @ {fps_num}/{fps_den} fps)",
        input.display(),
        width,
        height
    );
    // Output pool: the B-frame reorderer holds several buffers before
    // the first packet emerges; they are locked in submission order
    // whenever the encoder reports output ready.
    // Without lookahead the classic sync contract holds (SUCCESS =>
    // every queued buffer lockable, blocking locks safe — the proven
    // pre-lookahead behavior). With lookahead, readiness trickles and
    // only non-blocking locks are safe.
    let midstream_blocking = rc.lookahead_depth == 0;
    let buffers: Vec<_> = (0..BITSTREAM_POOL)
        .map(|_| session.create_bitstream_buffer())
        .collect::<Result<_, _>>()?;

    let partial = {
        let mut p = output.as_os_str().to_owned();
        p.push(PARTIAL_SUFFIX);
        std::path::PathBuf::from(p)
    };
    // Audio: bit-exact passthrough when supported; a source whose audio
    // codec the muxer cannot re-emit degrades to video-only WITH a
    // warning; a file with no audio at all is simply video-only.
    // Degradations accumulate rather than overwrite each other: a file
    // can easily have an uncarryable audio codec *and* chapters, and
    // reporting only whichever check ran last is how a known limitation
    // turns back into a silent loss.
    let mut degradations: Vec<String> = Vec::new();
    // Every carryable track comes along. What cannot be carried is named
    // rather than dropped in silence — losing five of six tracks without
    // a word was the defect this milestone exists to end.
    let (unsupported, truncated) = source.audio_left_behind();
    if !unsupported.is_empty() {
        degradations.push(format!(
            "{} audio track(s) skipped — unsupported codec(s): {}",
            unsupported.len(),
            unsupported.join(", ")
        ));
    }
    if let Some(line) = crate::audio::describe(&source.audio_routes()) {
        degradations.push(line);
    }
    if truncated {
        degradations.push(format!(
            "source declares more than {} audio tracks; the rest were skipped",
            demux::MAX_AUDIO_TRACKS
        ));
    }
    // Subtitles, chapters and tags are not re-emitted into MP4 yet. The
    // user asked to keep the maximum of what a source carries, so the
    // honest interim answer is to name what is being left behind rather
    // than let it disappear quietly.
    if let Some(lost) = source.extras_description(&file) {
        degradations.push(format!("not carried into MP4 output: {lost}"));
    }
    let warning = if degradations.is_empty() {
        None
    } else {
        Some(degradations.join(" · "))
    };
    // Pre-flight: refuse a job the volume cannot hold, before a single
    // byte is written. A twelve-hour encode runs for hours; discovering
    // at hour two that the disk is full throws away the whole run and
    // leaves the machine wedged. Bounded execution applied to storage —
    // a job that cannot possibly finish must not be started.
    //
    // The audio half of the estimate is exact (passthrough copies those
    // bytes verbatim); only the video leg is projected.
    {
        let audio_bytes: u64 = source.audio_bytes();
        let duration_secs = source_duration_secs;
        let estimate = estimate_output_bytes(
            duration_secs,
            settings.max_bitrate,
            audio_bytes,
            file.len() as u64,
        );
        let needed = estimate.saturating_mul(PEAK_OUTPUT_COPIES);
        // A volume we cannot measure is not grounds for refusing to
        // work: graceful degradation means an unreadable `statvfs`
        // costs the user a check, not their job.
        if let Ok(false) = sovmr_os::has_room_for(&partial, needed) {
            let available = sovmr_os::free_space(&partial).unwrap_or(0);
            return Err(NativeError::NoRoom { needed, available });
        }
    }
    let audio_params = source.audio_params();
    let audio_routes = source.audio_routes();
    let audio_specs = source.audio_track_specs();
    let audio: Vec<AudioFeed> = source
        .audio_timescales()
        .into_iter()
        .enumerate()
        .map(|(index, timescale)| {
            // A compressed track's frames are ours, not the source's, so
            // its cursor feeds a compressor rather than the muxer. If the
            // compressor refuses to start — an unmeasured rate, an
            // unreadable FLAC header — the track is passed through
            // instead of lost.
            let compressing = match audio_routes.get(index) {
                Some(crate::audio::Route::Compress(kind)) => {
                    let channels = audio_specs
                        .get(index)
                        .map_or(0, |spec| u32::from(spec.channels));
                    // The *source's* setup bytes, not the output's:
                    // `audio_track_specs` already renamed this track to
                    // AAC, and a FLAC compressor handed an AAC config
                    // would refuse to start.
                    // The *source's* setup bytes, not the output's:
                    // `audio_track_specs` already renamed this track to
                    // AAC, and a FLAC compressor handed an AAC config
                    // would refuse to start.
                    let private = source.audio_source_config(index);
                    crate::audio::TrackCompressor::new(*kind, &private, timescale, channels)
                        .ok()
                        .map(|compressor| Compressing {
                            compressor,
                            ready: std::collections::VecDeque::new(),
                            next_dts: 0,
                            finished: false,
                        })
                }
                _ => None,
            };
            AudioFeed {
                index,
                packets: source.audio_packets(&file, index),
                peeked: None,
                compressing,
                timescale,
                released_upto: 0,
                drained: false,
            }
        })
        .collect();
    let spool = {
        let mut p = partial.as_os_str().to_owned();
        p.push(SPOOL_SUFFIX);
        std::path::PathBuf::from(p)
    };
    let muxer = match source.container() {
        crate::source::Container::Mkv => OutputMuxer::Mkv {
            muxer: None,
            path: partial.clone(),
            audio: source.audio_track_specs(),
            title: source.title().map(str::to_string),
            marks: source.marks(),
            width,
            height,
            timescale: v_timescale,
            pending: Vec::new(),
            colour: source.video_colour(),
        },
        crate::source::Container::Mp4 => OutputMuxer::Mp4(Mp4Muxer::create(
            &spool,
            VideoMuxParams {
                codec: match settings.codec {
                    EncodeCodec::Av1 => MuxVideoCodec::Av1,
                    EncodeCodec::Hevc => MuxVideoCodec::Hevc,
                },
                width: u16::try_from(width).unwrap_or(u16::MAX),
                height: u16::try_from(height).unwrap_or(u16::MAX),
                timescale: v_timescale,
                colour: source.video_colour(),
            },
            // Identity travels with each track: language, title, grouping
            // and the enabled flag are copied from the source rather than
            // regenerated, so a track arrives at the far end still
            // recognisable and still choosable.
            audio_params,
        )?),
    };
    let sink = Mp4Sink {
        muxer,
        hevc_config: None,
        width: u16::try_from(width).unwrap_or(u16::MAX),
        height: u16::try_from(height).unwrap_or(u16::MAX),
        codec: settings.codec,
        audio,
        durations,
        v_dts: 0,
        pts_ceiling,
        muxed_pts: std::collections::HashSet::new(),
        v_timescale,
        seq_header: None,
        scratch: Vec::new(),
    };
    // Where the source's own sample sizes say scenes begin. Sorted by
    // construction, which is what lets the encode loop ask in constant
    // time rather than scanning per frame.
    let sample_shape = source.video_sample_sizes(&file);
    let mut scene_cuts: Vec<u64> = sovmr_core::complexity::Complexity::from_sizes(
        sample_shape.iter().map(|(size, _)| *size).collect(),
    )
    .scene_cuts()
    .into_iter()
    .filter_map(|i| {
        sample_shape
            .get(i)
            .map(|(_, pts)| (*pts + pts_shift) as u64)
    })
    .collect();
    scene_cuts.sort_unstable();
    // What each scene is worth, keyed by the timestamp its first frame
    // carries. A scene the source encoder found hard gets a lower
    // quality target — lower means more bits — and an easy one gives
    // some back. Bounded on both sides: an unbounded redistribution can
    // starve a scene into visible mush while the meter still reads well
    // on average.
    let scene_quality: std::collections::HashMap<u64, u8> = {
        let curve = sovmr_core::complexity::Complexity::from_sizes(
            sample_shape.iter().map(|(size, _)| *size).collect(),
        );
        curve
            .scene_weights()
            .into_iter()
            .filter_map(|(start, _, weight)| {
                let pts = (sample_shape.get(start)?.1 + pts_shift) as u64;
                // A weight of 1000 is exactly average and changes
                // nothing. Twice average moves the target by the full
                // step, and the step is small on purpose.
                let step = (weight as i64 - 1000) * i64::from(QUALITY_SPREAD) / 1000;
                let target = i64::from(settings.cq)
                    - step.clamp(-i64::from(QUALITY_SPREAD), i64::from(QUALITY_SPREAD));
                Some((pts, target.clamp(1, 51) as u8))
            })
            .collect()
    };
    if !scene_cuts.is_empty() {
        // Both numbers, because they answer different questions: how
        // many scenes were found, and how many were judged different
        // enough from the job's own target to be worth changing. A
        // redistribution that changed nothing looks exactly like one
        // that never ran.
        let moved = scene_quality
            .values()
            .filter(|q| **q != settings.cq)
            .count();
        eprintln!(
            "sovmr: {} scene cut(s) read from the source's sample table, \
             {moved} scene(s) given a different quality target",
            scene_cuts.len()
        );
    }

    let shared = RefCell::new(Shared {
        decoder: None,
        surface: None,
        registrations: HashMap::new(),
        staging: (0..BITSTREAM_POOL).map(|_| None).collect(),
        free: (0..BITSTREAM_POOL).collect(),
        pending: VecDeque::new(),
        sink,
        frames: 0,
        output_bytes: 0,
        error: None,
        abandoned: false,
    });

    // The three hooks of the push pipeline. Any error is parked in
    // `shared.error` and stops the parser via `false`/`None`.
    let callbacks = ParserCallbacks {
        on_sequence: Box::new(|fmt: &VideoFormat| {
            let mut s = shared.borrow_mut();
            let cfg = DecoderConfig {
                codec: decode_codec,
                width: fmt.coded_width,
                height: fmt.coded_height,
                bit_depth_minus8: fmt.bit_depth_luma_minus8,
                // The same number the callback is about to
                // promise the parser; two counts is how the driver
                // ends up mapping a surface it never made.
                decode_surfaces: u32::from(fmt.min_num_decode_surfaces).max(PARSER_SURFACE_HINT),
            };
            match nvdec.create_decoder(&ctx, &cfg) {
                Ok(dec) => {
                    s.decoder = Some(dec);
                    s.surface = Some((fmt.coded_width, fmt.coded_height));
                    Some(u32::from(fmt.min_num_decode_surfaces).max(PARSER_SURFACE_HINT))
                }
                Err(e) => {
                    s.error = Some(e.into());
                    None
                }
            }
        }),
        on_decode: Box::new(|pic| {
            let mut s = shared.borrow_mut();
            if s.abandoned {
                // Cancelled: keep the parser draining, spend no GPU on it.
                return true;
            }
            let Some(dec) = s.decoder.as_ref() else {
                s.error = Some(NativeError::Pipeline(
                    "decode callback before sequence callback".into(),
                ));
                return false;
            };
            // SAFETY: `pic` is the pointer of the current decode
            // callback invocation, and the pipeline's CUDA context is
            // current on this (single) pipeline thread.
            match unsafe { dec.decode_picture(pic) } {
                Ok(()) => true,
                Err(e) => {
                    s.error = Some(e.into());
                    false
                }
            }
        }),
        on_display: Box::new(|info| {
            let mut guard = shared.borrow_mut();
            if guard.abandoned {
                // Cancelled: the output is being thrown away anyway, so
                // nothing reaches the encoder or the muxer from here on.
                return true;
            }
            // Disjoint field borrows: decoder read-only; registrations
            // and sink mutable — all owned by this one thread.
            let Shared {
                decoder,
                surface,
                registrations,
                staging,
                free,
                pending,
                sink,
                frames,
                output_bytes,
                error,
                // Already handled by the early return above; named
                // rather than elided so a new field still fails here.
                abandoned: _,
            } = &mut *guard;
            let (Some(dec), Some(&(surf_w, surf_h))) = (decoder.as_ref(), surface.as_ref()) else {
                *error = Some(NativeError::Pipeline(
                    "display callback before sequence callback".into(),
                ));
                return false;
            };
            // Zero-copy handoff: the VRAM pointer from NVDEC's ring is
            // registered once (cached) and mapped into NVENC — never
            // copied to host; only the compressed lock result crosses
            // the bus. Registration describes the SURFACE (coded dims:
            // that is where NV12's chroma plane really starts); the pic
            // params describe the display crop being encoded.
            let result = (|| -> Result<(), NativeError> {
                let buffer = free.pop_front().ok_or_else(|| {
                    NativeError::Pipeline(
                        "bitstream pool exhausted (encoder holding too many frames)".into(),
                    )
                })?;
                let frame = dec.map_frame(info.picture_index)?;
                let pitch = frame.pitch();
                // NV12: luma rows + half-height chroma rows, pitch each.
                let frame_bytes = pitch as usize * (surf_h as usize) * 3 / 2;
                if staging[buffer].is_none() {
                    staging[buffer] = Some(
                        ctx.alloc_linear(frame_bytes, pitch as usize)
                            .map_err(NativeError::Cuda)?,
                    );
                }
                let slot = staging[buffer].as_ref().expect("just filled");
                // SAFETY: both allocations are live in this context; the
                // NVDEC surface is mapped for exactly this scope.
                unsafe { ctx.copy_device(slot.device_ptr(), frame.device_ptr(), frame_bytes) }
                    .map_err(NativeError::Cuda)?;
                drop(frame); // surface back to NVDEC's ring immediately
                let registration = match registrations.entry(slot.device_ptr()) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(v) => v.insert(session.register_vram_frame(&VramFrame {
                        device_ptr: slot.device_ptr(),
                        width: surf_w,
                        height: surf_h,
                        pitch,
                    })?),
                };
                let mapped = registration.map()?;
                // NVDEC displays in presentation order; the timestamp is
                // the (shifted) source PTS we fed the parser. The encoder
                // echoes it back on the reordered output packet.
                let pts = info.timestamp.max(0) as u64;
                let duration = u64::from(sink.duration_for(pts));
                // A keyframe where the source's own sample sizes say a
                // scene begins, rather than wherever a ten-second grid
                // happens to fall. The grid still applies as a ceiling —
                // a film with no cuts must still be seekable.
                let force_idr = scene_cuts.binary_search(&pts).is_ok();
                // Redistribute at the boundary, never inside a scene: a
                // rate change mid-scene is a visible step, and this is
                // the one place the picture changes anyway.
                if force_idr {
                    if let Some(target) = scene_quality.get(&pts) {
                        if *target != settings.cq {
                            // A refusal is not fatal — the encode
                            // continues at the previous setting — but it
                            // is said, because silence would make a
                            // driver that never accepts one look like a
                            // redistribution that never helped.
                            if let Err(e) = session.reconfigure_quality(*target) {
                                eprintln!("sovmr: scene rate change refused ({e:?})");
                            }
                        }
                    }
                }
                let ready = session.encode_frame(
                    &mapped,
                    &buffers[buffer],
                    width,
                    height,
                    pitch,
                    *frames as u32,
                    pts,
                    duration,
                    force_idr,
                )?;
                pending.push_back(buffer);
                if ready {
                    drain_pending(
                        &buffers,
                        pending,
                        free,
                        sink,
                        frames,
                        output_bytes,
                        midstream_blocking,
                    )?;
                }
                Ok(())
            })();
            match result {
                Ok(()) => true,
                Err(e) => {
                    *error = Some(e);
                    false
                }
            }
        }),
    };
    let mut parser = nvdec.create_parser(decode_codec, PARSER_SURFACE_HINT, callbacks)?;

    // Feed loop: demux -> Annex B -> parser (callbacks fire inline).
    let mut annex_b = Vec::new();
    let mut cancelled = false;
    let mut video_frames = source.video_frames(&file);
    while let Some(next) = video_frames.next_frame(&mut annex_b) {
        if cancel.load(Ordering::SeqCst) {
            cancelled = true;
            break;
        }
        let index = next?;
        // Feed the SHIFTED presentation timestamp: NVDEC hands it back
        // on the display callback, keyed to this exact frame.
        let pts = pts_by_decode_order.get(index).copied().unwrap_or(0);
        parser.feed(&annex_b, pts as i64)?;
        let s = shared.borrow();
        if s.error.is_some() {
            break;
        }
        let frames = s.frames;
        drop(s);
        on_progress(frames);
    }
    if cancelled {
        // Cancellation used to `break` here and drop the parser without
        // ever signalling end of stream, leaving libnvcuvid's own worker
        // thread mid-picture while `transcode` returned and tore its
        // frame down around it.
        //
        // Honest status: a SIGSEGV inside that driver thread was
        // observed three times right after the source moved from a heap
        // `Vec` to a file mapping (unmodified HEAD: 0 crashes in 26
        // runs), and it has not reproduced since — but neither does it
        // reproduce with this shutdown removed, because recompiling
        // perturbs the timing. So this is NOT a proven fix, and it is
        // not claimed as one. It is kept because an orderly shutdown is
        // the correct way to stop a decoder regardless: flag the
        // callbacks inert so no further GPU work is queued, then drain
        // the parser to a real end of stream so the driver is quiesced
        // before anything it might still be touching goes away.
        shared.borrow_mut().abandoned = true;
        let _ = parser.end_of_stream();
    } else {
        parser.end_of_stream()?;
        session.send_eos()?;
        let mut s = shared.borrow_mut();
        let Shared {
            free,
            pending,
            sink,
            frames,
            output_bytes,
            ..
        } = &mut *s;
        if midstream_blocking {
            // Classic sync contract (no lookahead): EOS makes every
            // queued output lockable — the proven blocking drain.
            drain_pending(&buffers, pending, free, sink, frames, output_bytes, true)?;
        } else {
            // Lookahead flush: readiness trickles; pump non-blocking
            // rounds while the hardware finishes. Buffers still stuck
            // after the patience budget are abandoned loudly — the e2e
            // gates assert exact frame counts, so a real loss cannot
            // ship silently.
            for _round in 0..EOS_PUMP_ROUNDS {
                if pending.is_empty() {
                    break;
                }
                drain_pending(&buffers, pending, free, sink, frames, output_bytes, false)?;
                if !pending.is_empty() {
                    std::thread::sleep(std::time::Duration::from_millis(EOS_PUMP_SLEEP_MS));
                }
            }
            if !pending.is_empty() {
                eprintln!(
                    "sovmr: nvenc flush left {} buffer(s) unreleased (driver flush quirk)",
                    pending.len()
                );
                pending.clear();
            }
        }
    }
    drop(parser); // callbacks release their borrow of `shared`

    let mut s = shared.into_inner();
    if let Some(err) = s.error.take() {
        let _ = std::fs::remove_file(&partial);
        return Err(err); // an unfinalized Mp4Muxer rolls its spool back
    }
    if cancelled {
        let _ = std::fs::remove_file(&partial);
        return Ok(TranscodeOutcome::Cancelled);
    }
    let mut sink = s.sink;
    // Flush the audio tail past the last video frame.
    drain_audio_feeds(&mut sink.muxer, &mut sink.audio, None)?;
    // Any failure from here on must not leave a `.part` behind (the
    // muxer already rolls its spool back on drop).
    match sink.muxer.finalize(&partial) {
        Ok(bytes) => s.output_bytes = bytes,
        Err(e) => {
            let _ = std::fs::remove_file(&partial);
            return Err(e.into());
        }
    }
    if let Err(e) = std::fs::rename(&partial, output) {
        let _ = std::fs::remove_file(&partial);
        return Err(e.into());
    }
    on_progress(s.frames);
    Ok(TranscodeOutcome::Completed(TranscodeStats {
        frames: s.frames,
        output_bytes: s.output_bytes,
        warning,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A twelve-hour movie, the duration this milestone is built for.
    const TWELVE_HOURS_SECS: f64 = 12.0 * 60.0 * 60.0;
    /// A plausible heuristic ceiling for 1080p, in bits/s.
    const CEILING_BPS: u32 = 4_000_000;

    #[test]
    fn the_estimate_tracks_the_rate_ceiling_over_the_whole_duration() {
        // 4 Mbit/s x 12 h = 21.6 GB of video before margin. An estimate
        // that quietly used minutes, or dropped the duration entirely,
        // would wave through a job that cannot fit.
        let est = estimate_output_bytes(TWELVE_HOURS_SECS, CEILING_BPS, 0, 0);
        let bare = (TWELVE_HOURS_SECS * f64::from(CEILING_BPS) / 8.0) as u64;
        assert!(
            est > bare && est < bare + bare / 5,
            "estimate {est} must sit just above the bare {bare} bytes"
        );
    }

    #[test]
    fn audio_is_counted_exactly_because_passthrough_copies_it_verbatim() {
        // Six tracks of a twelve-hour master are gigabytes on their own.
        // Ignoring them is how a "just fits" job runs out of disk.
        let audio = 6 * 700 * 1024 * 1024;
        let without = estimate_output_bytes(TWELVE_HOURS_SECS, CEILING_BPS, 0, 0);
        let with = estimate_output_bytes(TWELVE_HOURS_SECS, CEILING_BPS, audio, 0);
        assert!(
            with >= without + audio,
            "every audio byte must appear in the estimate: {with} vs {without} + {audio}"
        );
    }

    #[test]
    fn an_uncapped_encode_falls_back_to_the_source_size() {
        // max_bitrate == 0 means "no ceiling", which has no analytic
        // answer. Returning ~0 there would make the check useless
        // precisely on the jobs with the least predictable output.
        let source = 50 * 1024 * 1024 * 1024;
        let est = estimate_output_bytes(TWELVE_HOURS_SECS, 0, 0, source);
        assert!(
            est >= source,
            "an uncapped job must budget at least the source size, got {est}"
        );
    }

    #[test]
    fn a_source_with_no_usable_duration_still_yields_an_answer() {
        // Containers do lie, and a missing `mdhd` duration arrives here
        // as 0.0. That must degrade to a small estimate, not a panic and
        // not a NaN that poisons every comparison downstream.
        let est = estimate_output_bytes(0.0, CEILING_BPS, 1024, 0);
        assert!(est >= 1024, "audio is still known exactly: {est}");
    }

    #[test]
    fn the_no_room_refusal_tells_the_user_both_numbers() {
        // This message is the whole user-facing value of the check: a
        // bare "failed" would leave someone guessing whether to free
        // 200 MB or 2 TB. Both figures must survive into the text.
        let err = NativeError::NoRoom {
            needed: 40 * 1024 * 1024 * 1024,
            available: 3 * 1024 * 1024 * 1024,
        };
        let text = err.to_string();
        assert!(
            text.contains("40960") && text.contains("3072"),
            "the refusal must name what is needed and what is free: {text}"
        );
    }

    #[test]
    fn absurd_inputs_saturate_instead_of_wrapping() {
        // Negative trust: a hostile or corrupt header can claim a
        // duration of centuries. Wrapping would turn a colossal estimate
        // into a tiny one and wave the job straight through.
        let est = estimate_output_bytes(f64::MAX, u32::MAX, u64::MAX, u64::MAX);
        assert!(est > u64::MAX / 2, "must saturate high, not wrap: {est}");
    }
}
