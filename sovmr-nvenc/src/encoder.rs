// SPDX-License-Identifier: GPL-3.0-or-later
//
// sovmr — sovereign media re-encoder
// Copyright (C) 2026  AdmiralGDR
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Encoder bring-up and the frame loop: `nvEncInitializeEncoder`,
//! bitstream buffers, `nvEncEncodePicture`, `nvEncLockBitstream`.
//!
//! Every struct layout below is pinned against the public SDK 12.0
//! header, cross-checked with a compiled C probe on this exact toolchain
//! (`sizeof`/`offsetof` asserted in the tests). Rate control is patched
//! on top of the driver-supplied preset configuration
//! (`nvEncGetEncodePresetConfigEx`), so we never guess codec-specific
//! defaults — the driver fills them, we adjust CQ.

use std::ffi::c_void;

use crate::api::{status_name, MappedInput, NvEncSession, NvencError};

/// Windows-style GUID, byte-compatible with the header's `GUID`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

impl Guid {
    /// An all-zero GUID, for buffers the driver is about to fill.
    ///
    /// It matches no codec, so a slot the driver leaves untouched can
    /// never be mistaken for a supported one.
    pub const ZERO: Guid = Guid {
        data1: 0,
        data2: 0,
        data3: 0,
        data4: [0; 8],
    };
}

/// `NV_ENC_CODEC_AV1_GUID`.
pub const CODEC_AV1_GUID: Guid = Guid {
    data1: 0x0a35_2289,
    data2: 0x0aa7,
    data3: 0x4759,
    data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
};
/// `NV_ENC_CODEC_HEVC_GUID`.
pub const CODEC_HEVC_GUID: Guid = Guid {
    data1: 0x790c_dc88,
    data2: 0x4522,
    data3: 0x4d7b,
    data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
};
/// `NV_ENC_PRESET_P7_GUID` — the slowest and, in principle, the most
/// efficient. Whether it earns its time is a measurement, not a claim:
/// see the sweep that decides it.
pub const PRESET_P7_GUID: Guid = Guid {
    data1: 0x8484_8c12,
    data2: 0x6f71,
    data3: 0x4c13,
    data4: [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
};

/// `NV_ENC_PRESET_P2_GUID` (speed).
pub const PRESET_P2_GUID: Guid = Guid {
    data1: 0xf581_cfb8,
    data2: 0x88d6,
    data3: 0x4381,
    data4: [0x93, 0xf0, 0xdf, 0x13, 0xf9, 0xc2, 0x7d, 0xab],
};
/// `NV_ENC_PRESET_P4_GUID` (balanced).
pub const PRESET_P4_GUID: Guid = Guid {
    data1: 0x90a7_b826,
    data2: 0xdf06,
    data3: 0x4862,
    data4: [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
};
/// `NV_ENC_PRESET_P6_GUID` (quality — the project default).
pub const PRESET_P6_GUID: Guid = Guid {
    data1: 0x8e75_c279,
    data2: 0x6299,
    data3: 0x4ab6,
    data4: [0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5],
};

/// Struct version tags (SDK 12.0; the `1<<31` bit marks extended structs).
const NVENCAPI_VERSION: u32 = 12;
const fn struct_ver(rev: u32) -> u32 {
    NVENCAPI_VERSION | (rev << 16) | (0x7 << 28)
}
const INITIALIZE_PARAMS_VER: u32 = struct_ver(5) | (1 << 31);
const CONFIG_VER: u32 = struct_ver(8) | (1 << 31);
const PRESET_CONFIG_VER: u32 = struct_ver(4) | (1 << 31);
const PIC_PARAMS_VER: u32 = struct_ver(6) | (1 << 31);
const LOCK_BITSTREAM_VER: u32 = struct_ver(2);
const CREATE_BITSTREAM_BUFFER_VER: u32 = struct_ver(1);

/// Function-table slots (0-based, verified against the SDK 12.0 header).
const SLOT_GET_PRESET_CONFIG_EX: usize = 39;
const SLOT_INITIALIZE_ENCODER: usize = 11;
const SLOT_CREATE_BITSTREAM_BUFFER: usize = 14;
const SLOT_DESTROY_BITSTREAM_BUFFER: usize = 15;
const SLOT_ENCODE_PICTURE: usize = 16;
const SLOT_LOCK_BITSTREAM: usize = 17;
const SLOT_UNLOCK_BITSTREAM: usize = 18;

/// Enum values per the header.
const NV_ENC_PARAMS_RC_VBR: u32 = 0x1;
/// Offset of `idrPeriod` inside BOTH `NV_ENC_CONFIG_AV1` and
/// `NV_ENC_CONFIG_HEVC` (C probe against nv-codec-headers sdk/12.0:
/// AV1 size 1552 / HEVC size 1560, idrPeriod at 20 in each).
const IDR_PERIOD_OFFSET: usize = 20;

/// Where the colour description lives inside the codec union.
///
/// The union is [`CodecConfigBlob`], 1792 opaque bytes: the driver's
/// preset fill is authoritative and this project does not model codec
/// internals. Reaching one field costs one probe, and the probe is
/// recorded so the next reader need not guess.
///
/// C probe against nv-codec-headers (NVENCAPI 13.0), agreeing with the
/// 12.0 probe that found `idrPeriod`: `NV_ENC_CONFIG_HEVC` is 1560
/// bytes, `NV_ENC_CONFIG_AV1` 1552, `idrPeriod` at 20 in both under
/// either version. Every field below is a `uint32_t`.
///
/// # Why it matters
///
/// A container can carry a colour description — and this project writes
/// one — but a decoder reads the *elementary stream's* own description
/// in preference. NVENC leaves that unspecified unless told, so a file
/// could say BT.2020 in its `colr` box and nothing in its bitstream.
mod reference_offset {
    /// HEVC `useBFramesAsRef`, AV1 the same field at its own place.
    ///
    /// A B-pyramid lets B-frames reference other B-frames, so a group of
    /// pictures forms a hierarchy rather than a flat run. Whether that
    /// buys anything is a measurement.
    pub const HEVC_B_AS_REF: usize = 180;
    /// AV1 `useBFramesAsRef`.
    pub const AV1_B_AS_REF: usize = 88;
    /// HEVC `numRefL0` and `numRefL1`: how many pictures each list may
    /// hold. More references cost search time and can buy bits back on
    /// material that repeats.
    pub const HEVC_NUM_REF_L0: usize = 184;
    /// See [`HEVC_NUM_REF_L0`].
    pub const HEVC_NUM_REF_L1: usize = 188;
    /// AV1 `numFwdRefs`.
    pub const AV1_FWD_REFS: usize = 104;
    /// AV1 `numBwdRefs`.
    pub const AV1_BWD_REFS: usize = 108;

    /// `NV_ENC_BFRAME_REF_MODE_DISABLED`.
    pub const B_REF_DISABLED: u32 = 0;
    /// `NV_ENC_BFRAME_REF_MODE_MIDDLE`: the middle B of each group
    /// becomes a reference, which is the shallowest useful pyramid.
    pub const B_REF_MIDDLE: u32 = 2;
    /// `NV_ENC_NUM_REF_FRAMES_AUTOSELECT` — let the driver decide.
    pub const REF_AUTOSELECT: u32 = 0;
    /// Four pictures per list, the deepest these enums name.
    pub const REF_FOUR: u32 = 4;
}

mod colour_offset {
    /// `hevcVUIParameters.videoSignalTypePresentFlag`.
    pub const HEVC_SIGNAL_PRESENT: usize = 72;
    /// `hevcVUIParameters.videoFullRangeFlag`.
    pub const HEVC_FULL_RANGE: usize = 80;
    /// `hevcVUIParameters.colourDescriptionPresentFlag`.
    pub const HEVC_DESCRIPTION_PRESENT: usize = 84;
    /// `hevcVUIParameters.colourPrimaries`.
    pub const HEVC_PRIMARIES: usize = 88;
    /// `hevcVUIParameters.transferCharacteristics`.
    pub const HEVC_TRANSFER: usize = 92;
    /// `hevcVUIParameters.colourMatrix`.
    pub const HEVC_MATRIX: usize = 96;
    /// AV1 `colorPrimaries`.
    pub const AV1_PRIMARIES: usize = 68;
    /// AV1 `transferCharacteristics`.
    pub const AV1_TRANSFER: usize = 72;
    /// AV1 `matrixCoefficients`.
    pub const AV1_MATRIX: usize = 76;
    /// AV1 `colorRange`.
    pub const AV1_RANGE: usize = 80;
}

/// A colour description to state in the elementary stream.
///
/// Plain numbers rather than a shared type: this crate binds NVENC and
/// nothing else, and depending on a container crate to name three
/// integers would be the wrong way round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamColour {
    /// `colour_primaries` code point.
    pub primaries: u16,
    /// `transfer_characteristics` code point.
    pub transfer: u16,
    /// `matrix_coefficients` code point.
    pub matrix: u16,
    /// Whether samples use the full range rather than 16-235.
    pub full_range: bool,
}

/// The initialization the driver accepted, kept for reconfiguring.
///
/// Both boxes, because the driver was handed pointers into them and a
/// reconfigure hands back the same shape. The preset lives inside the
/// params by pointer, so the two must move together or the pointer
/// dangles.
pub struct AcceptedConfig {
    params: Box<InitializeParams>,
    preset: Box<PresetConfig>,
    /// Which codec it is, since the union's meaning depends on it.
    codec: EncodeCodec,
}

impl std::fmt::Debug for AcceptedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcceptedConfig")
            .field("codec", &self.codec)
            .finish_non_exhaustive()
    }
}

/// `NV_ENC_RECONFIGURE_PARAMS`, laid out around **our** initialization
/// structure rather than the newest header's.
///
/// The header on this machine is API 13, where `NV_ENC_INITIALIZE_PARAMS`
/// is 1800 bytes and the reconfigure structure is 1816. This crate speaks
/// API 12, where the inner structure is 1808 — so the flags that follow
/// it sit eight bytes further along, and the whole is 1824.
///
/// Deriving that by subtraction is what made the first attempt at this
/// stop: the flags were not where the newer header put them. They are
/// placed here by the same rule the header uses — immediately after the
/// inner structure — with the size asserted at compile time.
#[repr(C)]
struct ReconfigureParams {
    version: u32,
    _reserved: u32,
    init: InitializeParams,
    /// Bit 0 `resetEncoder`, bit 1 `forceIDR`; the rest reserved.
    flags: u32,
    _reserved2: u32,
}

/// `NV_ENC_RECONFIGURE_PARAMS_VER`.
const RECONFIGURE_PARAMS_VER: u32 = struct_ver(1) | (1 << 31);
/// `reInitEncodeParams.forceIDR` — start a fresh sequence at the change.
const RECONFIGURE_FORCE_IDR: u32 = 1 << 1;
/// Function-table slot of `nvEncReconfigureEncoder` (C probe: 32, with
/// `nvEncInitializeEncoder` at 11 in both the header and this crate).
const SLOT_RECONFIGURE_ENCODER: usize = 32;

/// How B-frames and references are configured for one encode.
///
/// Both live in the opaque codec union, so both cost a probe — recorded
/// in [`reference_offset`] rather than guessed. Defaults reproduce what
/// every build so far did, so turning a lever on is the only thing that
/// changes anything.
/// # Measured, and refused
///
/// Both defaults are off, and both stay off, because the sweep priced
/// them on material where quality rather than a cap decided the size
/// (AV1, driver 580.173, 640x360 noise at cq 32):
///
/// ```text
/// P6 quality     1058795 bytes   30.82 dB   SSIM 0.6058
/// P6 +pyramid    1085650  +2.5%  30.88 dB   SSIM 0.6090
/// P6 +refs       1058795   0.0%  30.82 dB   SSIM 0.6058
/// P6 +both       1092833  +3.2%  30.89 dB   SSIM 0.6091
/// ```
///
/// The B-pyramid buys six hundredths of a decibel for two and a half
/// percent more bytes — the wrong direction for a compressor. Four
/// references per list changed the output by **not one byte**: the
/// driver's autoselect had already chosen it.
///
/// Kept rather than deleted so the next reader inherits the numbers
/// instead of the probe, exactly as lookahead is kept pinned at zero
/// with its own falsification recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReferencePlan {
    /// Let B-frames reference other B-frames.
    pub b_pyramid: bool,
    /// Ask for four pictures per reference list instead of the driver's
    /// own choice.
    pub deep_refs: bool,
}

/// Write a [`ReferencePlan`] into the codec-specific config union.
///
/// Stated either way, never merely added: the preset fill is the
/// driver's, and a lever left to whatever a preset chose is a lever
/// nobody is measuring. That lesson cost an undecodable file when
/// `ULTRA_HIGH_QUALITY` switched lookahead on by itself.
fn write_reference_plan(bytes: &mut [u8], codec: EncodeCodec, plan: ReferencePlan) {
    fn put(bytes: &mut [u8], at: usize, value: u32) {
        if let Some(slot) = bytes.get_mut(at..at + 4) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
    let b_mode = if plan.b_pyramid {
        reference_offset::B_REF_MIDDLE
    } else {
        reference_offset::B_REF_DISABLED
    };
    let refs = if plan.deep_refs {
        reference_offset::REF_FOUR
    } else {
        reference_offset::REF_AUTOSELECT
    };
    match codec {
        EncodeCodec::Hevc => {
            put(bytes, reference_offset::HEVC_B_AS_REF, b_mode);
            put(bytes, reference_offset::HEVC_NUM_REF_L0, refs);
            put(bytes, reference_offset::HEVC_NUM_REF_L1, refs);
        }
        EncodeCodec::Av1 => {
            put(bytes, reference_offset::AV1_B_AS_REF, b_mode);
            put(bytes, reference_offset::AV1_FWD_REFS, refs);
            put(bytes, reference_offset::AV1_BWD_REFS, refs);
        }
    }
}

/// State a colour description inside the codec-specific config union.
///
/// A raw splice by measured offset, exactly as `idrPeriod` is written.
///
/// HEVC needs two flags raised before the values mean anything: its VUI
/// carries a colour description only when it says it does. AV1 has no
/// such flags here — the driver raises `color_description_present_flag`
/// itself once the values are set.
fn write_stream_colour(bytes: &mut [u8], codec: EncodeCodec, colour: StreamColour) {
    fn put(bytes: &mut [u8], at: usize, value: u32) {
        if let Some(slot) = bytes.get_mut(at..at + 4) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
    match codec {
        EncodeCodec::Hevc => {
            put(bytes, colour_offset::HEVC_SIGNAL_PRESENT, 1);
            put(bytes, colour_offset::HEVC_DESCRIPTION_PRESENT, 1);
            put(
                bytes,
                colour_offset::HEVC_FULL_RANGE,
                u32::from(colour.full_range),
            );
            put(
                bytes,
                colour_offset::HEVC_PRIMARIES,
                u32::from(colour.primaries),
            );
            put(
                bytes,
                colour_offset::HEVC_TRANSFER,
                u32::from(colour.transfer),
            );
            put(bytes, colour_offset::HEVC_MATRIX, u32::from(colour.matrix));
        }
        EncodeCodec::Av1 => {
            put(
                bytes,
                colour_offset::AV1_PRIMARIES,
                u32::from(colour.primaries),
            );
            put(
                bytes,
                colour_offset::AV1_TRANSFER,
                u32::from(colour.transfer),
            );
            put(bytes, colour_offset::AV1_MATRIX, u32::from(colour.matrix));
            put(
                bytes,
                colour_offset::AV1_RANGE,
                u32::from(colour.full_range),
            );
        }
    }
}
/// `NV_ENC_RC_PARAMS` bitfield word (our `flags` field): `enableAQ`
/// (spatial) is bit 3 and `enableTemporalAQ` bit 8 — counted off the
/// SDK 12.0 header's bitfield declaration order (LSB-first on this ABI).
/// `NV_ENC_MULTI_PASS` (C header, sdk/12.0): there is no
/// `NV_ENC_MULTI_PASS_PIQ` in the API — the real enum is
/// DISABLED=0, TWO_PASS_QUARTER_RESOLUTION=1, TWO_PASS_FULL_RESOLUTION=2.
/// Full-resolution two-pass makes the hardware analyse frame structure
/// before quantizing, which is what buys quality at a hard ceiling.
const NV_ENC_TWO_PASS_FULL_RESOLUTION: u32 = 0x2;
const RC_FLAG_ENABLE_SPATIAL_AQ: u32 = 1 << 3;
const RC_FLAG_ENABLE_LOOKAHEAD: u32 = 1 << 5;
const RC_FLAG_ENABLE_TEMPORAL_AQ: u32 = 1 << 8;
const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;
// `NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY` is 5, probed from the same
// header, and is deliberately not offered: measured on driver 580.173 it
// segfaults inside the encode, from a clean process, with lookahead
// already forced off. A lever that cannot produce a file is refused by
// the same rule that would have adopted a good one.
const NV_ENC_PIC_STRUCT_FRAME: u32 = 0x1;
/// `NV_ENC_PIC_FLAG_FORCEIDR`, probed from the header: start a fresh
/// coded video sequence at this picture.
const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 2;
const NV_ENC_PIC_FLAG_EOS: u32 = 0x8;
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x1;
const NV_ENC_SUCCESS: i32 = 0;
/// `NV_ENC_ERR_NEED_MORE_INPUT`: not an error — the encoder wants more
/// frames before it can emit output (B-frame reordering).
const NV_ENC_ERR_NEED_MORE_INPUT: i32 = 17;
/// `NV_ENC_ERR_INVALID_PARAM` / `NV_ENC_ERR_LOCK_BUSY`: what a
/// non-blocking lock returns for a buffer the encoder still holds
/// (lookahead keeps its window full, so readiness arrives one frame at
/// a time rather than queue-at-once).
const NV_ENC_ERR_INVALID_PARAM: i32 = 8;
const NV_ENC_ERR_LOCK_BUSY: i32 = 21;
/// `NV_ENC_LOCK_BITSTREAM.doNotWait` (bit 0 of the bitfield word).
const LOCK_FLAG_DO_NOT_WAIT: u32 = 1;
/// `NV_ENC_PIC_TYPE_I` / `NV_ENC_PIC_TYPE_IDR` — the two intra picture
/// types `nvEncLockBitstream` reports for sync samples.
const NV_ENC_PIC_TYPE_I: u32 = 0x02;
const NV_ENC_PIC_TYPE_IDR: u32 = 0x03;

/// `NV_ENC_QP` (per-frame-type QP triple).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvEncQp {
    qp_inter_p: u32,
    qp_inter_b: u32,
    qp_intra: u32,
}

/// `NV_ENC_RC_PARAMS`, exact (C probe: size 128, targetQuality at 88).
#[repr(C)]
struct RcParams {
    version: u32,
    rate_control_mode: u32,
    const_qp: NvEncQp,
    average_bit_rate: u32,
    max_bit_rate: u32,
    vbv_buffer_size: u32,
    vbv_initial_delay: u32,
    flags: u32,
    min_qp: NvEncQp,
    max_qp: NvEncQp,
    initial_rc_qp: NvEncQp,
    temporal_layer_idx_mask: u32,
    temporal_layer_qp: [u8; 8],
    target_quality: u8,
    target_quality_lsb: u8,
    lookahead_depth: u16,
    low_delay_key_frame_scale: u8,
    y_dc_qp_index_offset: i8,
    u_dc_qp_index_offset: i8,
    v_dc_qp_index_offset: i8,
    qp_map_mode: u32,
    multi_pass: u32,
    alpha_layer_bitrate_ratio: u32,
    cb_qp_index_offset: i8,
    cr_qp_index_offset: i8,
    reserved2: u16,
    reserved: [u32; 4],
}

/// `NV_ENC_CONFIG`, exact (C probe: size 3584, rcParams at 40,
/// encodeCodecConfig at 168 with a 1792-byte codec union).
#[repr(C)]
struct EncConfig {
    version: u32,
    profile_guid: Guid,
    gop_length: u32,
    frame_interval_p: i32,
    monochrome_encoding: u32,
    frame_field_mode: u32,
    mv_precision: u32,
    rc_params: RcParams,
    codec_config: CodecConfigBlob,
    reserved: [u32; 278],
    reserved2: [*mut c_void; 64],
}

/// The codec-specific union, kept opaque: the driver's preset fill is
/// authoritative and we do not edit codec internals at this milestone.
#[repr(C, align(8))]
struct CodecConfigBlob {
    bytes: [u8; 1792],
}

/// `NV_ENC_PRESET_CONFIG` (C probe: size 5128, presetCfg at 8).
#[repr(C)]
struct PresetConfig {
    version: u32,
    _pad: u32,
    preset_cfg: EncConfig,
    reserved1: [u32; 255],
    reserved2: [*mut c_void; 64],
}

/// `NV_ENC_INITIALIZE_PARAMS` (C probe: size 1808, encodeConfig at 88,
/// tuningInfo at 136).
#[repr(C)]
struct InitializeParams {
    version: u32,
    encode_guid: Guid,
    preset_guid: Guid,
    encode_width: u32,
    encode_height: u32,
    dar_width: u32,
    dar_height: u32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    enable_encode_async: u32,
    enable_ptd: u32,
    flags: u32,
    priv_data_size: u32,
    priv_data: *mut c_void,
    encode_config: *mut EncConfig,
    max_encode_width: u32,
    max_encode_height: u32,
    max_me_hint_counts: [u32; 8],
    tuning_info: u32,
    buffer_format: u32,
    reserved: [u32; 287],
    _pad_tail: u32,
    reserved2: [*mut c_void; 64],
}

/// `NV_ENC_PIC_PARAMS` prefix + opaque tail (C probe: size 3360,
/// inputBuffer at 40, codecPicParams at 80, codec union 1552).
#[repr(C)]
struct PicParams {
    version: u32,
    input_width: u32,
    input_height: u32,
    input_pitch: u32,
    encode_pic_flags: u32,
    frame_idx: u32,
    input_time_stamp: u64,
    input_duration: u64,
    input_buffer: *mut c_void,
    output_bitstream: *mut c_void,
    completion_event: *mut c_void,
    buffer_fmt: u32,
    picture_struct: u32,
    picture_type: u32,
    _pad: u32,
    codec_pic_params: CodecPicBlob,
    tail: PicTail,
}

#[repr(C, align(8))]
struct CodecPicBlob {
    bytes: [u8; 1552],
}

#[repr(C, align(8))]
struct PicTail {
    bytes: [u8; 3360 - 80 - 1552],
}

/// `NV_ENC_LOCK_BITSTREAM` (C probe: size 1544, bitstreamSizeInBytes at
/// 36, bitstreamBufferPtr at 56).
#[repr(C)]
struct LockBitstreamParams {
    version: u32,
    flags: u32,
    output_bitstream: *mut c_void,
    slice_offsets: *mut u32,
    frame_idx: u32,
    hw_encode_status: u32,
    num_slices: u32,
    bitstream_size_in_bytes: u32,
    output_time_stamp: u64,
    output_duration: u64,
    bitstream_buffer_ptr: *mut c_void,
    picture_type: u32,
    picture_struct: u32,
    tail: LockTail,
}

#[repr(C, align(8))]
struct LockTail {
    bytes: [u8; 1544 - 72],
}

/// `NV_ENC_CREATE_BITSTREAM_BUFFER` (C probe: size 776, buffer at 16).
#[repr(C)]
struct CreateBitstreamBufferParams {
    version: u32,
    size: u32,
    memory_heap: u32,
    reserved: u32,
    bitstream_buffer: *mut c_void,
    bitstream_buffer_ptr: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 64],
}

type GetPresetConfigExFn =
    unsafe extern "C" fn(*mut c_void, Guid, Guid, u32, *mut PresetConfig) -> i32;
type InitializeEncoderFn = unsafe extern "C" fn(*mut c_void, *mut InitializeParams) -> i32;
type CreateBitstreamBufferFn =
    unsafe extern "C" fn(*mut c_void, *mut CreateBitstreamBufferParams) -> i32;
type DestroyBitstreamBufferFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
/// `NVENCSTATUS nvEncReconfigureEncoder(void*, NV_ENC_RECONFIGURE_PARAMS*)`.
type ReconfigureEncoderFn = unsafe extern "C" fn(*mut c_void, *mut ReconfigureParams) -> i32;
type EncodePictureFn = unsafe extern "C" fn(*mut c_void, *mut PicParams) -> i32;
type LockBitstreamFn = unsafe extern "C" fn(*mut c_void, *mut LockBitstreamParams) -> i32;
type UnlockBitstreamFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

/// Codec selector for encoder bring-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeCodec {
    /// AV1 — the project's preferred target on Ada.
    Av1,
    /// HEVC fallback.
    Hevc,
}

/// Speed/quality preset (maps to NVENC P2/P4/P6/P7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePreset {
    /// Fast (P2).
    Speed,
    /// Balanced (P4).
    Balanced,
    /// Best quality per bit (P6) — the project default.
    Quality,
    /// P7, the slowest the hardware offers.
    ///
    /// Kept separate from [`EncodePreset::Quality`] until a measurement
    /// says which belongs where: a preset that costs more time and
    /// returns nothing is a lever removed, not a default changed.
    Slowest,
}

/// Everything needed to initialize the hardware encoder — mirrors
/// `sovmr-core`'s backend-neutral `EncodeSettings` without depending on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderInit {
    /// Target codec.
    pub codec: EncodeCodec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate numerator (rate control pacing).
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Constant-quality target (0–51, lower = better) — drives VBR-CQ.
    pub cq: u8,
    /// GOP length in frames: an IDR/keyframe is forced every this many
    /// frames. This is what makes timeline seeking work — players jump
    /// to the nearest sync sample (`stss`), so the interval is the seek
    /// granularity. Never 0: the preset default (250-frame GOPs) would
    /// leave multi-second stretches with no seek point.
    pub gop_length: u32,
    /// Peak bitrate cap in bits/s (0 = uncapped). In quality-driven VBR
    /// this is the guard rail that stops complex/noisy scenes from
    /// spiking to hardware maximums and bloating the file.
    pub max_bit_rate: u32,
    /// `frameIntervalP`: 1 = IPP (no reordering), N = N−1 B-frames
    /// between references. B-frames are the single biggest compression
    /// win; they reorder output, so the caller must mux with PTS/`ctts`.
    pub frame_interval_p: u32,
    /// Rate-control lookahead depth in frames (0 = off). Lookahead lets
    /// the encoder see complexity ahead and shift bits away from noise;
    /// it also deepens output latency by the same amount — the caller's
    /// bitstream pool must cover it.
    pub lookahead_depth: u16,
    /// Two-pass rate control: the first pass measures frame complexity so
    /// the second can distribute the (hard-capped) bits where they matter.
    /// The init ladder degrades this if the driver rejects it.
    pub multi_pass: bool,
    /// Encoder effort preset.
    pub preset: EncodePreset,
    /// How B-frames and references are configured.
    pub references: ReferencePlan,
    /// What the source said its colours mean, to state in the stream.
    ///
    /// `None` leaves the driver's "unspecified", which is the honest
    /// answer when the source declared nothing.
    pub colour: Option<StreamColour>,
}

/// The rate-control facts the driver was actually initialized with —
/// returned by [`NvEncSession::initialize`] so the pipeline can log
/// them (a silent fallback to the driver's CBR default is exactly the
/// failure mode this makes impossible to miss).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitSummary {
    /// `NV_ENC_PARAMS_RC_MODE` value (1 = VBR).
    pub rate_control_mode: u32,
    /// Constant-quality target handed to the driver.
    pub target_quality: u8,
    /// Average bitrate cap (0 = quality-driven, uncapped).
    pub average_bit_rate: u32,
    /// Peak bitrate cap (0 = uncapped).
    pub max_bit_rate: u32,
    /// Keyframe interval in frames (both `gopLength` and `idrPeriod`).
    pub gop_length: u32,
    /// `frameIntervalP` actually accepted (may be degraded to 1).
    pub frame_interval_p: i32,
    /// Spatial adaptive quantization accepted by the driver.
    pub spatial_aq: bool,
    /// Temporal adaptive quantization accepted by the driver.
    pub temporal_aq: bool,
    /// Lookahead depth the preset chose (kept as-is).
    pub lookahead_depth: u16,
    /// Multi-pass mode the preset chose (kept as-is).
    pub multi_pass: u32,
}

impl std::fmt::Display for InitSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = match self.rate_control_mode {
            0 => "CONSTQP",
            1 => "VBR",
            2 => "CBR",
            other => return write!(f, "rc mode {other} (UNEXPECTED)"),
        };
        write!(
            f,
            "rc={mode} targetQuality={} avgBitrate={} maxBitrate={} gop={} bframes={} \
             spatialAQ={} temporalAQ={} lookahead={} multipass={}",
            self.target_quality,
            self.average_bit_rate,
            self.max_bit_rate,
            self.gop_length,
            self.frame_interval_p.max(1) - 1,
            u8::from(self.spatial_aq),
            u8::from(self.temporal_aq),
            self.lookahead_depth,
            self.multi_pass,
        )
    }
}

impl NvEncSession<'_> {
    fn entry_fn(&self, slot: usize) -> Result<*mut c_void, NvencError> {
        self.entry_point(slot)
    }

    /// Initialize the encoder for `init` (`nvEncInitializeEncoder`).
    ///
    /// The configuration starts from the driver's own preset fill
    /// (`nvEncGetEncodePresetConfigEx`, tuning HIGH_QUALITY) and patches
    /// only rate control: VBR with `targetQuality = cq` and uncapped
    /// bitrate — NVENC's constant-quality mode, exactly what
    /// `-cq N -b:v 0` meant in the FFmpeg era. `frameIntervalP` is forced
    /// to 1 (no B-frame reordering) so the synchronous frame loop emits
    /// output for every submitted picture; the lookahead/B-frame path is
    /// a later measured optimization.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] with the symbolic reason (e.g.
    /// `NV_ENC_ERR_UNSUPPORTED_PARAM` if the GPU lacks the codec).
    pub fn initialize(&self, init: &EncoderInit) -> Result<InitSummary, NvencError> {
        let get_preset: GetPresetConfigExFn = {
            let ptr = self.entry_fn(SLOT_GET_PRESET_CONFIG_EX)?;
            // SAFETY: slot + signature per the SDK 12.0 header.
            unsafe { std::mem::transmute::<*mut c_void, GetPresetConfigExFn>(ptr) }
        };
        let initialize: InitializeEncoderFn = {
            let ptr = self.entry_fn(SLOT_INITIALIZE_ENCODER)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, InitializeEncoderFn>(ptr) }
        };

        let encode_guid = match init.codec {
            EncodeCodec::Av1 => CODEC_AV1_GUID,
            EncodeCodec::Hevc => CODEC_HEVC_GUID,
        };
        let preset_guid = match init.preset {
            EncodePreset::Slowest => PRESET_P7_GUID,
            EncodePreset::Speed => PRESET_P2_GUID,
            EncodePreset::Balanced => PRESET_P4_GUID,
            EncodePreset::Quality => PRESET_P6_GUID,
        };

        // Feature ladder: temporal AQ and B-frames vary by codec
        // generation. Ask for everything; on rejection degrade exactly
        // one feature per retry instead of failing the whole job. The
        // accepted combination is visible in the returned summary.
        let wanted_fip = init.frame_interval_p.max(1);
        let la = init.lookahead_depth;
        let mp = init.multi_pass;
        // Degrade exactly one feature per retry, cheapest-to-lose last.
        let attempts: [(u16, bool, bool, u32); 5] = [
            (la, true, mp, wanted_fip),
            (0, true, mp, wanted_fip),
            (0, false, mp, wanted_fip),
            (0, false, false, wanted_fip),
            (0, false, false, 1),
        ];
        let mut last = NvencError::Status {
            call: "nvEncInitializeEncoder",
            code: NV_ENC_SUCCESS,
        };
        for (lookahead, temporal_aq, multi_pass, fip) in attempts {
            match self.try_initialize(
                get_preset,
                initialize,
                init,
                encode_guid,
                preset_guid,
                lookahead,
                temporal_aq,
                multi_pass,
                fip,
            ) {
                Ok(summary) => return Ok(summary),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// One initialization attempt with the given feature combination.
    #[allow(clippy::too_many_arguments)]
    fn try_initialize(
        &self,
        get_preset: GetPresetConfigExFn,
        initialize: InitializeEncoderFn,
        init: &EncoderInit,
        encode_guid: Guid,
        preset_guid: Guid,
        lookahead: u16,
        temporal_aq: bool,
        multi_pass: bool,
        frame_interval_p: u32,
    ) -> Result<InitSummary, NvencError> {
        // 1. Let the driver fill its own preset configuration.
        // SAFETY: zeroed is valid for this plain-old-data struct; the
        // version tags are set before the call.
        let mut preset: Box<PresetConfig> = unsafe { Box::new(std::mem::zeroed()) };
        preset.version = PRESET_CONFIG_VER;
        preset.preset_cfg.version = CONFIG_VER;
        // SAFETY: params fully initialized & version-tagged; GUIDs by
        // value per the typedef; encoder handle alive via &self.
        let status = unsafe {
            get_preset(
                self.raw(),
                encode_guid,
                preset_guid,
                NV_ENC_TUNING_INFO_HIGH_QUALITY,
                preset.as_mut(),
            )
        };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncGetEncodePresetConfigEx",
                code: status,
            });
        }

        // 2. Patch rate control: constant-quality VBR with a peak cap.
        // averageBitRate stays 0 (quality drives the size); maxBitRate +
        // a one-second VBV window stop complex scenes from spiking to
        // hardware maximums (the file-bloat failure mode). Spatial (and
        // when accepted, temporal) AQ shifts bits toward what eyes see.
        let rc = &mut preset.preset_cfg.rc_params;
        rc.rate_control_mode = NV_ENC_PARAMS_RC_VBR;
        rc.average_bit_rate = 0;
        rc.max_bit_rate = init.max_bit_rate;
        rc.vbv_buffer_size = init.max_bit_rate;
        rc.vbv_initial_delay = 0;
        rc.target_quality = init.cq;
        rc.target_quality_lsb = 0;
        rc.flags |= RC_FLAG_ENABLE_SPATIAL_AQ;
        if temporal_aq {
            rc.flags |= RC_FLAG_ENABLE_TEMPORAL_AQ;
        }
        // Stated either way, never assumed. The preset fill is the
        // driver's, and some presets turn lookahead on by themselves —
        // `ULTRA_HIGH_QUALITY` asks for twenty-five. Adding the flag
        // when wanted while never clearing it when not meant the
        // pipeline believed lookahead was off, took its non-blocking
        // drain path, and produced a file no decoder would read.
        if lookahead > 0 {
            rc.flags |= RC_FLAG_ENABLE_LOOKAHEAD;
            rc.lookahead_depth = lookahead;
        } else {
            rc.flags &= !RC_FLAG_ENABLE_LOOKAHEAD;
            rc.lookahead_depth = 0;
        }
        rc.multi_pass = if multi_pass {
            NV_ENC_TWO_PASS_FULL_RESOLUTION
        } else {
            0
        };
        preset.preset_cfg.frame_interval_p = frame_interval_p as i32;
        // Keyframe cadence: gopLength in the shared config AND idrPeriod
        // inside the codec union (byte 20 of both NV_ENC_CONFIG_AV1 and
        // NV_ENC_CONFIG_HEVC — C probe against SDK 12.0). The preset's
        // own 250-frame GOP would leave seek bars nothing to land on.
        let gop = init.gop_length.max(1);
        preset.preset_cfg.gop_length = gop;
        preset.preset_cfg.codec_config.bytes[IDR_PERIOD_OFFSET..IDR_PERIOD_OFFSET + 4]
            .copy_from_slice(&gop.to_ne_bytes());
        // The colour description, in the stream itself. The container
        // carries one too, and a decoder prefers this one.
        write_reference_plan(
            &mut preset.preset_cfg.codec_config.bytes,
            init.codec,
            init.references,
        );
        if let Some(colour) = init.colour {
            write_stream_colour(
                &mut preset.preset_cfg.codec_config.bytes,
                init.codec,
                colour,
            );
        }

        // 3. Initialize.
        // SAFETY: zeroed POD, then every required field set below.
        let mut params: Box<InitializeParams> = unsafe { Box::new(std::mem::zeroed()) };
        params.version = INITIALIZE_PARAMS_VER;
        params.encode_guid = encode_guid;
        params.preset_guid = preset_guid;
        params.encode_width = init.width;
        params.encode_height = init.height;
        params.dar_width = init.width;
        params.dar_height = init.height;
        params.frame_rate_num = init.fps_num.max(1);
        params.frame_rate_den = init.fps_den.max(1);
        params.enable_encode_async = 0; // synchronous: no completion events
        params.enable_ptd = 1; // driver decides picture types
        params.encode_config = &mut preset.preset_cfg;
        params.max_encode_width = init.width;
        params.max_encode_height = init.height;
        params.tuning_info = NV_ENC_TUNING_INFO_HIGH_QUALITY;
        // SAFETY: params + config fully initialized; both boxes outlive
        // the call; the driver copies what it needs during the call.
        let status = unsafe { initialize(self.raw(), params.as_mut()) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncInitializeEncoder",
                code: status,
            });
        }
        // Report the exact rate-control facts the driver accepted, so
        // the caller can log them — never trust silence.
        let rc = &preset.preset_cfg.rc_params;
        let summary = {
            let rc = &preset.preset_cfg.rc_params;
            (
                rc.rate_control_mode,
                rc.target_quality,
                rc.average_bit_rate,
                rc.max_bit_rate,
                rc.flags,
                rc.lookahead_depth,
                rc.multi_pass,
            )
        };
        let _ = summary;
        let accepted_summary = Ok(InitSummary {
            rate_control_mode: rc.rate_control_mode,
            target_quality: rc.target_quality,
            average_bit_rate: rc.average_bit_rate,
            max_bit_rate: rc.max_bit_rate,
            gop_length: preset.preset_cfg.gop_length,
            frame_interval_p: preset.preset_cfg.frame_interval_p,
            spatial_aq: rc.flags & RC_FLAG_ENABLE_SPATIAL_AQ != 0,
            temporal_aq: rc.flags & RC_FLAG_ENABLE_TEMPORAL_AQ != 0,
            lookahead_depth: rc.lookahead_depth,
            multi_pass: rc.multi_pass,
        });
        // Kept so a later reconfigure can change one field and leave
        // every other exactly as the driver accepted it.
        *self.accepted.borrow_mut() = Some(AcceptedConfig {
            params,
            preset,
            codec: init.codec,
        });
        accepted_summary
    }

    /// Change the quality target mid-stream, at a scene boundary.
    ///
    /// `nvEncReconfigureEncoder` takes the whole initialization
    /// structure, so this hands back the one the driver accepted with a
    /// single field changed. Rebuilding it would mean re-deriving
    /// everything the preset filled in, and one field forgotten would
    /// silently revert.
    ///
    /// `forceIDR` is set: a rate change belongs at a fresh sequence, and
    /// this is only called where the picture changes anyway.
    ///
    /// # Errors
    ///
    /// The driver's own status. A refusal is not fatal — the encode
    /// continues at the previous setting — so the caller decides whether
    /// to care.
    pub fn reconfigure_quality(&self, target_quality: u8) -> Result<(), NvencError> {
        let mut accepted = self.accepted.borrow_mut();
        let Some(config) = accepted.as_mut() else {
            return Err(NvencError::Status {
                call: "nvEncReconfigureEncoder (never initialized)",
                code: -1,
            });
        };
        let reconfigure: ReconfigureEncoderFn = {
            let ptr = self.api.entry(SLOT_RECONFIGURE_ENCODER)?;
            // SAFETY: the slot holds the documented entry point.
            unsafe { std::mem::transmute::<*mut c_void, ReconfigureEncoderFn>(ptr) }
        };
        config.preset.preset_cfg.rc_params.target_quality = target_quality;
        let _ = config.codec;

        // SAFETY: `ReconfigureParams` is `#[repr(C)]` and (through
        // `InitializeParams`) contains only integers, `Guid`s, arrays of
        // those and raw pointers; all-zero is a valid value for each, and
        // the version tag is set on the next line before any driver call.
        let mut params: Box<ReconfigureParams> = unsafe { Box::new(std::mem::zeroed()) };
        params.version = RECONFIGURE_PARAMS_VER;
        // A byte copy: the structure is plain data plus pointers that
        // still point into `config`, and copying it is exactly what the
        // driver expects to be handed.
        // SAFETY: both are the same `#[repr(C)]` type and neither
        // overlaps the other.
        unsafe {
            std::ptr::copy_nonoverlapping(
                config.params.as_ref() as *const InitializeParams,
                &mut params.init as *mut InitializeParams,
                1,
            );
        }
        params.flags = RECONFIGURE_FORCE_IDR;
        // SAFETY: the structure is laid out per the probe and every
        // pointer inside it still points into `config`, which outlives
        // this call.
        let status = unsafe { reconfigure(self.raw(), params.as_mut()) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncReconfigureEncoder",
                code: status,
            });
        }
        Ok(())
    }

    /// Allocate one driver-owned output buffer
    /// (`nvEncCreateBitstreamBuffer`). Destroyed on drop.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] — typically `NV_ENC_ERR_ENCODER_NOT_INITIALIZED`
    /// if called before [`NvEncSession::initialize`].
    pub fn create_bitstream_buffer(&self) -> Result<BitstreamBuffer<'_>, NvencError> {
        let create: CreateBitstreamBufferFn = {
            let ptr = self.entry_fn(SLOT_CREATE_BITSTREAM_BUFFER)?;
            // SAFETY: slot + signature per header.
            unsafe { std::mem::transmute::<*mut c_void, CreateBitstreamBufferFn>(ptr) }
        };
        let destroy: DestroyBitstreamBufferFn = {
            let ptr = self.entry_fn(SLOT_DESTROY_BITSTREAM_BUFFER)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, DestroyBitstreamBufferFn>(ptr) }
        };
        let lock: LockBitstreamFn = {
            let ptr = self.entry_fn(SLOT_LOCK_BITSTREAM)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, LockBitstreamFn>(ptr) }
        };
        let unlock: UnlockBitstreamFn = {
            let ptr = self.entry_fn(SLOT_UNLOCK_BITSTREAM)?;
            // SAFETY: as above.
            unsafe { std::mem::transmute::<*mut c_void, UnlockBitstreamFn>(ptr) }
        };
        // SAFETY: zeroed POD + version tag.
        let mut params: Box<CreateBitstreamBufferParams> = unsafe { Box::new(std::mem::zeroed()) };
        params.version = CREATE_BITSTREAM_BUFFER_VER;
        // SAFETY: initialized params; encoder alive via &self.
        let status = unsafe { create(self.raw(), params.as_mut()) };
        if status != NV_ENC_SUCCESS || params.bitstream_buffer.is_null() {
            return Err(NvencError::Status {
                call: "nvEncCreateBitstreamBuffer",
                code: status,
            });
        }
        Ok(BitstreamBuffer {
            raw: params.bitstream_buffer,
            session: self,
            destroy,
            lock,
            unlock,
        })
    }

    /// Submit one mapped VRAM frame (`nvEncEncodePicture`).
    ///
    /// Returns `true` when encoded output is ready: per the sync-mode
    /// contract, **all previously `false`-returning (queued) submissions
    /// plus this one** become lockable, in submission order. `false` is
    /// the `NEED_MORE_INPUT` signal — the B-frame reorderer is holding
    /// this output buffer; do not lock it yet.
    ///
    /// `pts`/`duration` are opaque timestamps echoed back on the encoded
    /// packet (`BitstreamLock::output_time_stamp`) — with B-frames the
    /// output order differs from presentation order, and this echo is
    /// how the muxer recovers each packet's PTS for `ctts`.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] with the symbolic reason.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_frame(
        &self,
        frame: &MappedInput<'_>,
        output: &BitstreamBuffer<'_>,
        width: u32,
        height: u32,
        pitch: u32,
        frame_idx: u32,
        pts: u64,
        duration: u64,
        force_idr: bool,
    ) -> Result<bool, NvencError> {
        let encode: EncodePictureFn = {
            let ptr = self.entry_fn(SLOT_ENCODE_PICTURE)?;
            // SAFETY: slot + signature per header.
            unsafe { std::mem::transmute::<*mut c_void, EncodePictureFn>(ptr) }
        };
        // SAFETY: zeroed POD; all required fields set below.
        let mut pic: Box<PicParams> = unsafe { Box::new(std::mem::zeroed()) };
        pic.version = PIC_PARAMS_VER;
        pic.input_width = width;
        pic.input_height = height;
        pic.input_pitch = pitch;
        pic.frame_idx = frame_idx;
        // A keyframe here rather than wherever the fixed cadence fell.
        // `FORCEIDR` restarts prediction outright, which is what a new
        // scene needs and what makes the picture after it seekable.
        if force_idr {
            pic.encode_pic_flags |= NV_ENC_PIC_FLAG_FORCEIDR;
        }
        pic.input_time_stamp = pts;
        pic.input_duration = duration;
        pic.input_buffer = frame.raw();
        pic.output_bitstream = output.raw;
        pic.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
        pic.picture_struct = NV_ENC_PIC_STRUCT_FRAME;
        // SAFETY: initialized params; mapped input + bitstream buffer are
        // alive per the borrows; encoder alive via &self.
        let status = unsafe { encode(self.raw(), pic.as_mut()) };
        match status {
            NV_ENC_SUCCESS => Ok(true),
            NV_ENC_ERR_NEED_MORE_INPUT => Ok(false),
            code => Err(NvencError::Status {
                call: "nvEncEncodePicture",
                code,
            }),
        }
    }

    /// Flush the encoder (`nvEncEncodePicture` with the EOS flag).
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] on driver failure.
    pub fn send_eos(&self) -> Result<(), NvencError> {
        let encode: EncodePictureFn = {
            let ptr = self.entry_fn(SLOT_ENCODE_PICTURE)?;
            // SAFETY: slot + signature per header.
            unsafe { std::mem::transmute::<*mut c_void, EncodePictureFn>(ptr) }
        };
        // SAFETY: zeroed POD; EOS pictures carry only version + flag.
        let mut pic: Box<PicParams> = unsafe { Box::new(std::mem::zeroed()) };
        pic.version = PIC_PARAMS_VER;
        pic.encode_pic_flags = NV_ENC_PIC_FLAG_EOS;
        // SAFETY: initialized params; encoder alive via &self.
        let status = unsafe { encode(self.raw(), pic.as_mut()) };
        if status != NV_ENC_SUCCESS {
            return Err(NvencError::Status {
                call: "nvEncEncodePicture(EOS)",
                code: status,
            });
        }
        Ok(())
    }
}

/// A driver-owned output buffer; one encoded frame is locked out of it at
/// a time. Destroyed on drop; borrows the session (teardown order is a
/// compile-time fact).
pub struct BitstreamBuffer<'s> {
    raw: *mut c_void,
    session: &'s NvEncSession<'s>,
    destroy: DestroyBitstreamBufferFn,
    lock: LockBitstreamFn,
    unlock: UnlockBitstreamFn,
}

impl BitstreamBuffer<'_> {
    /// Lock the buffer and view the encoded bytes (`nvEncLockBitstream`).
    /// The driver's pointer stays valid until the [`BitstreamLock`]
    /// drops (`nvEncUnlockBitstream`) — write the slice out, then drop.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] (e.g. `NV_ENC_ERR_LOCK_BUSY`).
    pub fn lock(&self) -> Result<BitstreamLock<'_>, NvencError> {
        match self.lock_inner(false)? {
            Some(lock) => Ok(lock),
            None => unreachable!("blocking lock never reports not-ready"),
        }
    }

    /// Non-blocking [`BitstreamBuffer::lock`]: `Ok(None)` when the
    /// encoder still holds this buffer (its frame sits in the lookahead
    /// or reorder queue) — try again after feeding more input, or with
    /// a blocking lock once EOS has flushed the queues.
    ///
    /// # Errors
    ///
    /// [`NvencError::Status`] for real failures.
    pub fn try_lock(&self) -> Result<Option<BitstreamLock<'_>>, NvencError> {
        self.lock_inner(true)
    }

    fn lock_inner(&self, do_not_wait: bool) -> Result<Option<BitstreamLock<'_>>, NvencError> {
        // SAFETY: zeroed POD + version + target buffer.
        let mut params: Box<LockBitstreamParams> = unsafe { Box::new(std::mem::zeroed()) };
        params.version = LOCK_BITSTREAM_VER;
        if do_not_wait {
            params.flags = LOCK_FLAG_DO_NOT_WAIT;
        }
        params.output_bitstream = self.raw;
        // SAFETY: initialized params; buffer + encoder alive per borrows.
        let status = unsafe { (self.lock)(self.session.raw(), params.as_mut()) };
        if do_not_wait && matches!(status, NV_ENC_ERR_LOCK_BUSY | NV_ENC_ERR_INVALID_PARAM) {
            return Ok(None); // the encoder still owns this buffer
        }
        if status != NV_ENC_SUCCESS || params.bitstream_buffer_ptr.is_null() {
            return Err(NvencError::Status {
                call: "nvEncLockBitstream",
                code: status,
            });
        }
        Ok(Some(BitstreamLock {
            data: params.bitstream_buffer_ptr.cast(),
            len: params.bitstream_size_in_bytes as usize,
            picture_type: params.picture_type,
            output_time_stamp: params.output_time_stamp,
            output_duration: params.output_duration,
            buffer: self,
        }))
    }
}

impl std::fmt::Debug for BitstreamBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitstreamBuffer")
            .field("raw", &self.raw)
            .finish()
    }
}

impl Drop for BitstreamBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful create; destroyed once; any
        // BitstreamLock has already dropped (it borrows self).
        unsafe {
            (self.destroy)(self.session.raw(), self.raw);
        }
    }
}

/// A locked view of one encoded frame. Unlocked on drop; dereferences to
/// the compressed bytes (AV1 OBUs / HEVC Annex B).
pub struct BitstreamLock<'b> {
    data: *const u8,
    len: usize,
    picture_type: u32,
    output_time_stamp: u64,
    output_duration: u64,
    buffer: &'b BitstreamBuffer<'b>,
}

impl BitstreamLock<'_> {
    /// The `NV_ENC_PIC_TYPE` the driver reported for this frame.
    pub fn picture_type(&self) -> u32 {
        self.picture_type
    }

    /// Whether this frame is a sync sample (I/IDR) — the fact the muxer
    /// records in `stss` so players can seek.
    pub fn is_keyframe(&self) -> bool {
        matches!(self.picture_type, NV_ENC_PIC_TYPE_I | NV_ENC_PIC_TYPE_IDR)
    }

    /// The presentation timestamp echoed from `encode_frame`'s `pts`.
    /// With B-frames, packets arrive in decode order — this is how the
    /// muxer recovers each packet's PTS for the `ctts` table.
    pub fn output_time_stamp(&self) -> u64 {
        self.output_time_stamp
    }

    /// The duration echoed from `encode_frame`'s `duration`.
    pub fn output_duration(&self) -> u64 {
        self.output_duration
    }
}

impl std::ops::Deref for BitstreamLock<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the driver guarantees `data..data+len` readable while
        // the lock is held; the lock lives exactly as long as `self`.
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

impl std::fmt::Debug for BitstreamLock<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitstreamLock")
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for BitstreamLock<'_> {
    fn drop(&mut self) {
        // SAFETY: lock acquired successfully; unlocked exactly once.
        unsafe {
            (self.buffer.unlock)(self.buffer.session.raw(), self.buffer.raw);
        }
    }
}

/// Human-readable status helper re-exported for pipeline diagnostics.
pub fn describe_status(code: i32) -> &'static str {
    status_name(code)
}

/// Layout of every FFI struct above, checked by the **compiler**.
///
/// Each number was printed by a compiled C probe against the SDK
/// header on this toolchain — facts, not estimates. They live in a
/// `const` block rather than a test so that a padding difference
/// between toolchains (MSVC and GCC lay some of these out
/// differently) stops the build on the machine doing the building,
/// instead of corrupting a driver call on frame 40 000 of a
/// twelve-hour encode.
const _: () = {
    assert!(std::mem::size_of::<RcParams>() == 128);
    assert!(std::mem::offset_of!(RcParams, target_quality) == 88);
    assert!(std::mem::offset_of!(RcParams, average_bit_rate) == 20);
    assert!(std::mem::offset_of!(RcParams, lookahead_depth) == 90);
    assert!(std::mem::offset_of!(RcParams, multi_pass) == 100);
    assert!(std::mem::size_of::<EncConfig>() == 3584);
    assert!(std::mem::offset_of!(EncConfig, rc_params) == 40);
    assert!(std::mem::offset_of!(EncConfig, codec_config) == 168);
    assert!(std::mem::size_of::<PresetConfig>() == 5128);
    assert!(std::mem::offset_of!(PresetConfig, preset_cfg) == 8);
    assert!(std::mem::size_of::<InitializeParams>() == 1808);
    // The reconfigure structure wraps the one above: four bytes of
    // version, four reserved, the whole initialization structure, then
    // the flags. The header on this machine is a newer API where the
    // inner structure is 1800 and the whole is 1816 — deriving our
    // offsets from those numbers is what went wrong the first time, so
    // the compiler is asked instead.
    assert!(std::mem::size_of::<ReconfigureParams>() == 1824);
    assert!(std::mem::offset_of!(ReconfigureParams, init) == 8);
    assert!(std::mem::offset_of!(ReconfigureParams, flags) == 1816);
    assert!(std::mem::offset_of!(InitializeParams, encode_config) == 88);
    assert!(std::mem::offset_of!(InitializeParams, tuning_info) == 136);
    assert!(std::mem::size_of::<PicParams>() == 3360);
    assert!(std::mem::offset_of!(PicParams, input_buffer) == 40);
    assert!(std::mem::offset_of!(PicParams, codec_pic_params) == 80);
    assert!(std::mem::size_of::<LockBitstreamParams>() == 1544);
    assert!(std::mem::offset_of!(LockBitstreamParams, bitstream_size_in_bytes) == 36);
    assert!(std::mem::offset_of!(LockBitstreamParams, bitstream_buffer_ptr) == 56);
    assert!(std::mem::offset_of!(LockBitstreamParams, picture_type) == 64);
    assert!(std::mem::size_of::<CreateBitstreamBufferParams>() == 776);
    assert!(std::mem::offset_of!(CreateBitstreamBufferParams, bitstream_buffer) == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every size/offset below was printed by a compiled C probe against
    /// the SDK 12.0 header on this toolchain — these are facts, not
    /// estimates.
    #[test]
    fn struct_layouts_match_the_c_probe() {}

    #[test]
    fn version_tags_follow_the_sdk12_macros() {
        assert_eq!(INITIALIZE_PARAMS_VER, 0x7005_000C | (1 << 31));
        assert_eq!(CONFIG_VER, 0x7008_000C | (1 << 31));
        assert_eq!(PRESET_CONFIG_VER, 0x7004_000C | (1 << 31));
        assert_eq!(PIC_PARAMS_VER, 0x7006_000C | (1 << 31));
        assert_eq!(LOCK_BITSTREAM_VER, 0x7002_000C);
        assert_eq!(CREATE_BITSTREAM_BUFFER_VER, 0x7001_000C);
    }
}
