// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// MXFP8 output epilogue for the sparse decode gather.
#pragma once

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

constexpr uint32_t MaxUb = SineUb + RopePairs * sizeof(float);
// A split-token merge produces both channels before either one is consumed.
constexpr uint32_t MaxChannelBytes = HeadHalf * CodesPerChannel * sizeof(bfloat16_t);
constexpr uint32_t ScaleFactorUb = MaxUb + 2 * MaxChannelBytes;
static_assert(ScaleFactorUb + MaxChannelBytes <= 262144);

struct Angles {
    RegTensor<float> cosine, sine;
};

PTO_INTERNAL void load_angles(Angles& a) {
    vlds(a.cosine, (__ubuf__ float*)CosineUb, 0, NORM);
    vlds(a.sine, (__ubuf__ float*)SineUb, 0, NORM);
}

// A producer emits 64 BF16 values (two MX groups). Use the rounded values
// for both the data and the maxima; rotating the FP32 accumulator changes ties.
PTO_INTERNAL void finish_fragment(RegTensor<bfloat16_t>& rounded, Angles& a,
                                   uint32_t row, uint32_t col, uint32_t channel) {
    MaskReg full = pset_b32(PAT_ALL);
    uint32_t count = 32;
    MaskReg lower = plt_b32(count, POST_UPDATE), upper;
    pnot(upper, lower, full);
    if (channel == 1 && col == ChannelWidth - 64) {
        RegTensor<float> value, even, odd, real, imag, cross, unused;
        vcvt(value, rounded, full, PART_EVEN);
        vdintlv(even, odd, value, value);
        vmul(real, even, a.cosine, lower, MODE_ZEROING);
        vmul(cross, odd, a.sine, lower, MODE_ZEROING);
        vadd(real, real, cross, lower, MODE_ZEROING);
        vmul(imag, odd, a.cosine, lower, MODE_ZEROING);
        vmul(cross, even, a.sine, lower, MODE_ZEROING);
        vsub(imag, imag, cross, lower);
        vintlv(value, unused, real, imag);
        vcvt(rounded, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
    }
    RegTensor<uint32_t> magnitude, signless, lo, hi;
    vdup(signless, uint32_t(0x7fff), full, MODE_ZEROING);
    vand(magnitude, (RegTensor<uint32_t>&)rounded, signless, full);
    vcmax(lo, magnitude, lower, MODE_ZEROING);
    vcmax(hi, magnitude, upper, MODE_ZEROING);
    auto maxima = (__ubuf__ uint16_t*)(MaxUb + channel * MaxChannelBytes);
    uint32_t group = row * CodesPerChannel + col / 32;
    vsts((vector_u16&)lo, maxima, group, ONEPT_B16, pset_b16(PAT_ALL));
    vsts((vector_u16&)hi, maxima, group + 1, ONEPT_B16, pset_b16(PAT_ALL));
}

// Reuse PTO's OCP exponent/reciprocal calculation. Pack each head once;
// for a zero scale retain the converted byte's sign and clear its magnitude.
template <typename SourceTile>
__tf__ __aicore__ inline void pack_channel(
    PayloadTile::TileDType __out__ output, ScaleTile::TileDType __out__ scales,
    typename SourceTile::TileDType __in__ values, int source_column, int channel) {
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto codes = (__ubuf__ uint8_t*)__cce_get_tile_ptr(scales);
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(values);
    auto maxima = (__ubuf__ bfloat16_t*)(MaxUb + channel * MaxChannelBytes);
    auto factors = (__ubuf__ bfloat16_t*)ScaleFactorUb;
    __VEC_SCOPE__ {
        mem_bar(VST_VLD);
        ExtractB8ExponentAndScaling(maxima, codes, factors, 2, HeadHalf * CodesPerChannel);
        mem_bar(VST_VLD);
        MaskReg bytes = pset_b8(PAT_ALL), halves = pset_b16(PAT_ALL), words = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg lower = plt_b32(count, POST_UPDATE);
        RegTensor<uint32_t> shifts, mask3, mask8, all_bits, signs;
        vci((RegTensor<int32_t>&)shifts, int32_t(0), INC_ORDER);
        vshrs(shifts, shifts, 3, words, MODE_ZEROING);
        vdup(mask3, uint32_t(3), words, MODE_ZEROING);
        vand(shifts, shifts, mask3, words);
        vshls(shifts, shifts, 3, words, MODE_ZEROING);
        vdup(mask8, uint32_t(0xff), words, MODE_ZEROING);
        vdup(all_bits, uint32_t(0xffffffff), words, MODE_ZEROING);
        vdup(signs, uint32_t(0x80808080), words, MODE_ZEROING);
        for (uint16_t head = 0; head < HeadHalf; ++head) {
            RegTensor<bfloat16_t> even, odd, multiplier;
            vlds(even, odd, src, head * SourceTile::Cols + source_column, DINTLV_B16);
            vlds((vector_u16&)multiplier, (__ubuf__ uint16_t*)factors, head * CodesPerChannel, E2B_B16);
            vmul(even, even, multiplier, halves, MODE_ZEROING);
            vmul(odd, odd, multiplier, halves, MODE_ZEROING);
            RegTensor<float> f0, f1, f2, f3;
            vcvt(f0, even, halves, PART_EVEN);
            vcvt(f1, odd, halves, PART_EVEN);
            vcvt(f2, even, halves, PART_ODD);
            vcvt(f3, odd, halves, PART_ODD);
            vector_f8e4m3 q0, q1, q2, q3;
            vcvt(q0, f0, words, ROUND_R, RS_ENABLE, PART_P0);
            vcvt(q1, f1, words, ROUND_R, RS_ENABLE, PART_P1);
            vcvt(q2, f2, words, ROUND_R, RS_ENABLE, PART_P2);
            vcvt(q3, f3, words, ROUND_R, RS_ENABLE, PART_P3);
            vor(q0, q0, q1, bytes);
            vor(q2, q2, q3, bytes);
            vor(q0, q0, q2, bytes);
            RegTensor<uint32_t> low_codes, high_codes, code, keep;
            vlds(low_codes, (__ubuf__ uint32_t*)codes, head * 2, BRC_B32);
            vlds(high_codes, (__ubuf__ uint32_t*)codes, head * 2 + 1, BRC_B32);
            vsel(code, low_codes, high_codes, lower);
            vshr((vector_u32&)code, (vector_u32&)code, (vector_s32&)shifts, words, MODE_ZEROING);
            vand(code, code, mask8, words);
            MaskReg zero;
            vcmps_eq(zero, code, uint32_t(0), words);
            vsel(keep, signs, all_bits, zero);
            vand((vector_u32&)q0, (vector_u32&)q0, (vector_u32&)keep, words);
            vsts((vector_u8&)q0, dst, head * ChannelWidth, NORM_B8, bytes);
        }
    }
}
#endif
} // namespace sparse_quant
