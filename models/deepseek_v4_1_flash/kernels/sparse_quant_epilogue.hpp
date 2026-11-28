// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed-cache gather, dual KV rings, and sink-aware sparse attention.
#pragma once
#include <pto/pto-inst.hpp>
#include "intrinsic.h"

namespace sparse_quant {
using namespace pto;
constexpr int HeadHalf = 32;
constexpr int Heads = 64;
constexpr int ChannelWidth = 256;
constexpr int HeadWidth = 2 * ChannelWidth;
constexpr int CodesPerChannel = ChannelWidth / 32;
constexpr int CodesPerToken = 2 * Heads * CodesPerChannel;
constexpr int RopePairs = 32;

using PayloadTile = Tile<TileType::Vec, uint8_t, HeadHalf, ChannelWidth>;
using ScaleTile = Tile<TileType::Vec, uint8_t, 1, HeadHalf * CodesPerChannel>;
using AngleTile = Tile<TileType::Vec, float, 1, RopePairs>;
template <typename T, int Rows, int Cols>
using GM = GlobalTensor<T, TileShape2D<T, Rows, Cols>, BaseShape2D<T, Rows, Cols>>;

constexpr uint32_t PayloadUb = 245760;
constexpr uint32_t ScalesUb = PayloadUb + HeadHalf * ChannelWidth;
constexpr uint32_t CosineUb = ScalesUb + HeadHalf * CodesPerChannel;
constexpr uint32_t SineUb = CosineUb + RopePairs * sizeof(float);
static_assert(SineUb + RopePairs * sizeof(float) <= 262144);

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
using InputChannel = Tile<TileType::Vec, bfloat16_t, HeadHalf, ChannelWidth>;
using QuantizedChannel = Tile<TileType::Vec, int8_t, HeadHalf, ChannelWidth>;
using QuantScratch = Tile<TileType::Vec, bfloat16_t, 1, HeadHalf * CodesPerChannel>;

constexpr uint32_t MaxUb = SineUb + RopePairs * sizeof(float);
constexpr uint32_t ReciprocalUb = MaxUb + HeadHalf * CodesPerChannel * sizeof(bfloat16_t);
// The balanced merge has finished using its left numerator before compaction.
constexpr uint32_t CompactMergeUb = 8192;
static_assert(CompactMergeUb + HeadHalf * ChannelWidth * sizeof(bfloat16_t) <= 73728);
static_assert(ReciprocalUb + HeadHalf * CodesPerChannel * sizeof(bfloat16_t) <= 262144);

template <typename SourceTile>
__tf__ __aicore__ inline void compact_merge_channel(
    InputChannel::TileDType __out__ destination,
    typename SourceTile::TileDType __in__ source, int source_column) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(destination);
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(source);
    __VEC_SCOPE__ {
        mem_bar(VST_VLD);
        const MaskReg full = pset_b16(PAT_ALL);
        for (uint16_t head = 0; head < uint16_t(HeadHalf); ++head) {
            RegTensor<bfloat16_t> low, high;
            const uint32_t input = head * SourceTile::Cols + source_column;
            const uint32_t output = head * ChannelWidth;
            vlds(low, src, input, NORM);
            vlds(high, src, input + 128, NORM);
            vsts(low, dst, output, NORM_B16, full);
            vsts(high, dst, output + 128, NORM_B16, full);
        }
    }
}

__tf__ __aicore__ inline void inverse_rope(
    InputChannel::TileDType __out__ source,
    AngleTile::TileDType __in__ cosine, AngleTile::TileDType __in__ sine) {
    auto input = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(source);
    auto cosines = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sines = (__ubuf__ float*)__cce_get_tile_ptr(sine);
    __VEC_SCOPE__ {
        mem_bar(VST_VLD);
        const MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = RopePairs;
        const MaskReg lower = plt_b32(count, POST_UPDATE);
        RegTensor<float> cosv, sinv;
        vlds(cosv, cosines, 0, NORM);
        vlds(sinv, sines, 0, NORM);
        for (uint16_t head = 0; head < uint16_t(HeadHalf); ++head) {
            const uint32_t offset = head * ChannelWidth + ChannelWidth - 64;
            RegTensor<bfloat16_t> rounded;
            RegTensor<float> value, even, odd, real, imag, cross, unused;
            vlds(rounded, input, offset, UNPK_B16);
            vcvt(value, rounded, full, PART_EVEN);
            vdintlv(even, odd, value, value);
            vmul(real, even, cosv, lower, MODE_ZEROING);
            vmul(cross, odd, sinv, lower, MODE_ZEROING);
            vadd(real, real, cross, lower, MODE_ZEROING);
            vmul(imag, odd, cosv, lower, MODE_ZEROING);
            vmul(cross, even, sinv, lower, MODE_ZEROING);
            vsub(imag, imag, cross, lower);
            vintlv(value, unused, real, imag);
            vcvt(rounded, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(rounded, input, offset, PK_B32, full);
        }
    }
}

__tf__ __aicore__ inline void clear_zero_scale_magnitudes(
    PayloadTile::TileDType __out__ payload, ScaleTile::TileDType __in__ scales) {
    auto words = (__ubuf__ uint32_t*)__cce_get_tile_ptr(payload);
    auto codes = (__ubuf__ uint32_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        mem_bar(VST_VLD);
        const MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        const MaskReg lower = plt_b32(count, POST_UPDATE);
        RegTensor<uint32_t> lanes, shifts, byte_mask, word_mask, sign_mask;
        vci((RegTensor<int32_t>&)lanes, int32_t(0), INC_ORDER);
        vshrs(shifts, lanes, 3, full, MODE_ZEROING);
        vdup(byte_mask, uint32_t(3), full, MODE_ZEROING);
        vand(shifts, shifts, byte_mask, full);
        vshls(shifts, shifts, 3, full, MODE_ZEROING);
        vdup(byte_mask, uint32_t(0xff), full, MODE_ZEROING);
        vdup(word_mask, uint32_t(0xffffffff), full, MODE_ZEROING);
        vdup(sign_mask, uint32_t(0x80808080), full, MODE_ZEROING);
        for (uint16_t head = 0; head < uint16_t(HeadHalf); ++head) {
            RegTensor<uint32_t> low, high, selected, exponent, mask, values;
            MaskReg zero;
            vlds(low, codes, head * 2, BRC_B32);
            vlds(high, codes, head * 2 + 1, BRC_B32);
            vsel(selected, low, high, lower);
            vshr((vector_u32&)exponent, (vector_u32&)selected, (vector_s32&)shifts, full, MODE_ZEROING);
            vand(exponent, exponent, byte_mask, full);
            vcmps_eq(zero, exponent, uint32_t(0), full);
            vsel(mask, sign_mask, word_mask, zero);
            vlds(values, words, head * (ChannelWidth / 4), NORM);
            vand(values, values, mask, full);
            vsts(values, words, head * (ChannelWidth / 4), NORM_B32, full);
        }
    }
}

__aicore__ inline void bulk_quantize(
    PayloadTile& payload, ScaleTile& scales, InputChannel& input,
    AngleTile& cosine, AngleTile& sine, bool rotate_tail) {
    if (rotate_tail) inverse_rope(input.data(), cosine.data(), sine.data());
    QuantizedChannel bytes;
    QuantScratch maximum, reciprocal;
    TASSIGN(bytes, PayloadUb);
    TASSIGN(maximum, MaxUb);
    TASSIGN(reciprocal, ReciprocalUb);
    const uint64_t control = get_ctrl();
    TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>(bytes, input, &scales, &maximum, &reciprocal);
    set_ctrl(control);
    clear_zero_scale_magnitudes(payload.data(), scales.data());
}

template <typename SourceTile>
__aicore__ inline void quantize_channel(
    PayloadTile& payload, ScaleTile& scales, SourceTile& source,
    AngleTile& cosine, AngleTile& sine, int source_column, bool rotate_tail) {
    if constexpr (SourceTile::Cols == ChannelWidth) {
        bulk_quantize(payload, scales, source, cosine, sine, rotate_tail);
    } else {
        InputChannel compact;
        TASSIGN(compact, CompactMergeUb);
        compact_merge_channel<SourceTile>(compact.data(), source.data(), source_column);
        bulk_quantize(payload, scales, compact, cosine, sine, rotate_tail);
    }
}

__aicore__ inline void store_channel(
    PayloadTile& payload, ScaleTile& scales,
    __gm__ uint8_t* output, __gm__ uint8_t* codes, int token, int head, int channel, event_t event) {
    using Shape = TileShape2D<uint8_t, HeadHalf, ChannelWidth>;
    using Stride = BaseShape2D<uint8_t, HeadHalf, HeadWidth>;
    using Output = GlobalTensor<uint8_t, Shape, Stride>;
    Output dst(output + int64_t(token) * Heads * HeadWidth + head * HeadWidth + channel * ChannelWidth);
    GM<uint8_t, 1, HeadHalf * CodesPerChannel> code_destination(
        codes + int64_t(token) * CodesPerToken + channel * Heads * CodesPerChannel + head * CodesPerChannel);
    set_flag(PIPE_V, PIPE_MTE3, event);
    wait_flag(PIPE_V, PIPE_MTE3, event);
    TSTORE(dst, payload);
    TSTORE(code_destination, scales);
    set_flag(PIPE_MTE3, PIPE_V, event);
    wait_flag(PIPE_MTE3, PIPE_V, event);
}
#endif
} // namespace sparse_quant
