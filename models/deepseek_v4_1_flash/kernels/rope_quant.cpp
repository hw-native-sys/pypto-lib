// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <cstring>
#endif

using namespace pto;

namespace {
template <typename T, int Rows, int Cols>
using Vec = Tile<TileType::Vec, T, Rows, Cols>;

template <typename T>
AICORE inline __gm__ T *Data(__gm__ int64_t *args, int index)
{
    auto tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

AICORE inline int64_t RowStride(__gm__ int64_t *args, int index)
{
    return reinterpret_cast<__gm__ Tensor *>(args[index])->strides[0];
}

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
// Eight group-32 reductions per 256-value window, using full BF16 vector lanes.
template <bool Rotate, typename Output, typename Scales, typename Input, typename Trig>
__tf__ AICORE inline void TransformWide(
    typename Output::TileDType __out__ output, typename Scales::TileDType __out__ scales,
    typename Input::TileDType __out__ input, typename Trig::TileDType __in__ cosine,
    typename Trig::TileDType __in__ sine, uint16_t activeRows)
{
    auto src = (__ubuf__ bfloat16_t *)__cce_get_tile_ptr(input);
    auto cos = (__ubuf__ float *)__cce_get_tile_ptr(cosine);
    auto sin = (__ubuf__ float *)__cce_get_tile_ptr(sine);
    auto dst = (__ubuf__ uint8_t *)__cce_get_tile_ptr(output);
    auto codes = (__ubuf__ uint16_t *)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all32 = pset_b32(PAT_ALL);
        MaskReg all16 = pset_b16(PAT_ALL);
        MaskReg all8 = pset_b8(PAT_ALL);
        if constexpr (Rotate) {
            RegTensor<uint32_t> lane, swapIndex, pairIndex, one32;
            vci((RegTensor<int32_t> &)lane, 0, INC_ORDER);
            vbr(one32, 1u);
            vxor(swapIndex, lane, one32, all32);
            vshrs(pairIndex, lane, 1u, all32, MODE_ZEROING);
            MaskReg even;
            vcmp_gt(even, swapIndex, lane, all32);
            for (uint16_t row = 0; row < activeRows; ++row) {
                RegTensor<bfloat16_t> raw;
                RegTensor<float> x, c, s, swapped, a, b, real, imag;
                vlds(raw, src, row * 512 + 448, UNPK_B16);
                vcvt(x, raw, all16, PART_EVEN);
                vgather2(c, cos + row * 32, pairIndex, all32);
                vgather2(s, sin + row * 32, pairIndex, all32);
                vmuls(s, s, -1.0f, all32, MODE_ZEROING);
                vselr((RegTensor<uint32_t> &)swapped, (RegTensor<uint32_t> &)x, swapIndex);
                vmul(a, x, c, all32, MODE_ZEROING);
                vmul(b, swapped, s, all32, MODE_ZEROING);
                vsub(real, a, b, all32, MODE_ZEROING);
                vadd(imag, b, a, all32, MODE_ZEROING);
                vsel(x, real, imag, even);
                vcvt(raw, x, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(raw, src, row * 512 + 448, PK_B32, all32);
            }
            // The BF16 boundary is materialized in UB, never in GM.
            mem_bar(VST_VLD);
        }
        RegTensor<uint16_t> exponentMask, minimum, bias, zero, lane16, groupIndex, scaleIndex;
        vbr(exponentMask, uint16_t(0x7f80));
        vbr(minimum, uint16_t(8 << 7));
        vbr(bias, uint16_t(254 << 7));
        vbr(zero, uint16_t(0));
        vci((RegTensor<int16_t> &)lane16, 0, INC_ORDER);
        vshrs(groupIndex, lane16, uint16_t(4), all16, MODE_ZEROING);
        vshls(scaleIndex, lane16, uint16_t(4), all16, MODE_ZEROING);
        MaskReg four16 = pset_b16(PAT_VL4);
        RegTensor<uint32_t> lowMask, highMask;
        vbr(lowMask, uint32_t(255));
        vbr(highMask, uint32_t(65280));
        for (uint16_t row = 0; row < 16; ++row) {
            for (uint16_t part = 0; part < 2; ++part) {
                RegTensor<bfloat16_t> even, odd;
                if (row < activeRows) {
                    vlds(even, odd, src, row * 512 + part * 256, DINTLV_B16);
                } else {
                    vbr((RegTensor<uint16_t> &)even, uint16_t(0));
                    vbr((RegTensor<uint16_t> &)odd, uint16_t(0));
                }
                RegTensor<uint16_t> e0, e1, shared, expanded, reciprocal;
                vand(e0, (RegTensor<uint16_t> &)even, exponentMask, all16);
                vand(e1, (RegTensor<uint16_t> &)odd, exponentMask, all16);
                vmax(e0, e0, e1, all16, MODE_ZEROING);
                vcgmax(shared, e0, all16, MODE_ZEROING);
                vmax(shared, shared, minimum, all16, MODE_ZEROING);
                vsub(shared, shared, minimum, all16, MODE_ZEROING);
                vselr(expanded, shared, groupIndex);
                vsub(reciprocal, bias, expanded, all16, MODE_ZEROING);
                MaskReg nonzero;
                vcmps_gt(nonzero, expanded, uint16_t(0), all16);
                vsel(reciprocal, reciprocal, zero, nonzero);
                vmul(even, even, (RegTensor<bfloat16_t> &)reciprocal, all16, MODE_ZEROING);
                vmul(odd, odd, (RegTensor<bfloat16_t> &)reciprocal, all16, MODE_ZEROING);
                RegTensor<float> x0, x1, x2, x3;
                vcvt(x0, even, all16, PART_EVEN);
                vcvt(x1, odd, all16, PART_EVEN);
                vcvt(x2, even, all16, PART_ODD);
                vcvt(x3, odd, all16, PART_ODD);
                vmaxs(x0, x0, -448.0f, all32, MODE_ZEROING);
                vmins(x0, x0, 448.0f, all32, MODE_ZEROING);
                vmaxs(x1, x1, -448.0f, all32, MODE_ZEROING);
                vmins(x1, x1, 448.0f, all32, MODE_ZEROING);
                vmaxs(x2, x2, -448.0f, all32, MODE_ZEROING);
                vmins(x2, x2, 448.0f, all32, MODE_ZEROING);
                vmaxs(x3, x3, -448.0f, all32, MODE_ZEROING);
                vmins(x3, x3, 448.0f, all32, MODE_ZEROING);
                vector_f8e4m3 p0, p1, p2, p3, packed;
                vcvt(p0, x0, all32, ROUND_R, RS_ENABLE, PART_P0);
                vcvt(p1, x1, all32, ROUND_R, RS_ENABLE, PART_P1);
                vcvt(p2, x2, all32, ROUND_R, RS_ENABLE, PART_P2);
                vcvt(p3, x3, all32, ROUND_R, RS_ENABLE, PART_P3);
                vor((RegTensor<uint8_t> &)p0, (RegTensor<uint8_t> &)p0, (RegTensor<uint8_t> &)p1, all8);
                vor((RegTensor<uint8_t> &)p2, (RegTensor<uint8_t> &)p2, (RegTensor<uint8_t> &)p3, all8);
                vor((RegTensor<uint8_t> &)packed, (RegTensor<uint8_t> &)p0, (RegTensor<uint8_t> &)p2, all8);
                vsts((RegTensor<uint8_t> &)packed, dst, row * 512 + part * 256, NORM_B8, all8);
                vshrs(shared, shared, uint16_t(7), all16, MODE_ZEROING);
                RegTensor<uint32_t> lows, highs, pairBits;
                vand(lows, (RegTensor<uint32_t> &)shared, lowMask, all32);
                vshrs(highs, (RegTensor<uint32_t> &)shared, uint32_t(8), all32, MODE_ZEROING);
                vand(highs, highs, highMask, all32);
                vor(pairBits, lows, highs, all32);
                RegTensor<uint16_t> pairs;
                vpack(pairs, pairBits, LOWER);
                vscatter(pairs, codes + part * 64 + row, scaleIndex, four16);
            }
        }
    }
}

template <bool Rotate>
AICORE inline void Prefetch(__gm__ int64_t *args, int work, int slot)
{
    const int heads = args[6] * 8;
    const int head = work % heads;
    const int row = work / heads * 16;
    int rows = args[5] - row;
    rows = rows < 0 ? 0 : (rows > 16 ? 16 : rows);
    using Input = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Trig = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, DYNAMIC, 32>;
    using InputShape = TileShape2D<bfloat16_t, DYNAMIC, 512>;
    using TrigShape = TileShape2D<float, DYNAMIC, 32>;
    using Strides = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    Input input(rows);
    Trig cos(rows), sin(rows);
    TASSIGN(input, slot * 65536);
    TASSIGN(cos, slot * 65536 + 16384);
    TASSIGN(sin, slot * 65536 + 18432);
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
    if (rows > 0) {
        GlobalTensor<bfloat16_t, InputShape, Strides> inputGm(
            Data<bfloat16_t>(args, 0) + row * RowStride(args, 0) + head * 512,
            InputShape(rows, 512), Strides(1, 1, 1, RowStride(args, 0), 1));
        TLOAD(input, inputGm);
        if constexpr (Rotate) {
            GlobalTensor<float, TrigShape, Strides> cosGm(
                Data<float>(args, 1) + row * RowStride(args, 1),
                TrigShape(rows, 32), Strides(1, 1, 1, RowStride(args, 1), 1));
            GlobalTensor<float, TrigShape, Strides> sinGm(
                Data<float>(args, 2) + row * RowStride(args, 2),
                TrigShape(rows, 32), Strides(1, 1, 1, RowStride(args, 2), 1));
            TLOAD(cos, cosGm);
            TLOAD(sin, sinGm);
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(slot));
}

template <bool Rotate, bool Quantize>
AICORE inline void Run(__gm__ int64_t *args)
{
    const int active = args[5];
    const int groups = args[6];
    const int padded = args[7];
    const int heads = groups * 8;
    using Input = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Trig = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, DYNAMIC, 32>;
    using Strides = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using OutType = std::conditional_t<Quantize, float8_e4m3_t, bfloat16_t>;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    const int total = (padded / 16) * heads;
    const int first = get_block_idx(args);
    const int step = get_block_num(args);
    if (first < total) Prefetch<Rotate>(args, first, 0);
    int slot = 0;
    for (int work = first; work < total; work += step) {
        const int head = work % heads;
        const int row = (work / heads) * 16;
        int rows = active - row;
        rows = rows < 0 ? 0 : (rows > 16 ? 16 : rows);
        Input input(rows);
        Trig cos(rows), sin(rows);
        Vec<OutType, 16, 512> output;
        Vec<uint8_t, 1, 256> scales;
        TASSIGN(input, slot * 65536 + 0);
        TASSIGN(cos, slot * 65536 + 16384);
        TASSIGN(sin, slot * 65536 + 18432);
        TASSIGN(output, slot * 65536 + 24576);
        TASSIGN(scales, slot * 65536 + 40960);
        wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(slot));
        // Separate UB slots let MTE2 fetch the next head tile while this VF runs.
        if (work + step < total) Prefetch<Rotate>(args, work + step, slot ^ 1);
        if constexpr (Quantize) {
            TransformWide<Rotate, decltype(output), decltype(scales), Input, Trig>(
                output.data(), scales.data(), input.data(), cos.data(), sin.data(), rows);
        }
        set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(slot));
        wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(slot));
        const int outputRow = Quantize ? (head / 8) * padded + row : row;
        const int outputCol = Quantize ? (head % 8) * 512 : head * 512;
        using OutShape = TileShape2D<OutType, 16, 512>;
        GlobalTensor<OutType, OutShape, Strides> outputGm(
            Data<OutType>(args, 3) + outputRow * RowStride(args, 3) + outputCol,
            OutShape(), Strides(1, 1, 1, RowStride(args, 3), 1));
        TSTORE(outputGm, output);
        if constexpr (Quantize) {
            using ScaleShape = TileShape2D<uint8_t, 1, 256>;
            GlobalTensor<uint8_t, ScaleShape, BaseShape2D<uint8_t, 1, 256>> scaleGm(
                Data<uint8_t>(args, 4) + (head / 8) * RowStride(args, 4) + row * 128 + (head % 8) * 256);
            TSTORE(scaleGm, scales);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
        slot ^= 1;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
AICORE inline void RunCpu(__gm__ int64_t *args)
{
    const int active = args[5], groups = args[6], padded = args[7], heads = groups * 8;
    for (int work = get_block_idx(args); work < padded / 16 * heads; work += get_block_num(args)) {
        const int head = work % heads, row0 = work / heads * 16;
        for (int row = 0; row < 16; ++row) {
            float values[512] = {};
            if (row0 + row < active) {
                auto src = Data<bfloat16_t>(args, 0) + (row0 + row) * RowStride(args, 0) + head * 512;
                for (int col = 0; col < 512; ++col) values[col] = static_cast<float>(src[col]);
                for (int pair = 0; pair < 32; ++pair) {
                    float c = Data<float>(args, 1)[(row0 + row) * RowStride(args, 1) + pair];
                    float s = -Data<float>(args, 2)[(row0 + row) * RowStride(args, 2) + pair];
                    float re = values[448 + 2 * pair], im = values[449 + 2 * pair];
                    volatile float re_cos = re * c, im_sin = im * s;
                    volatile float re_sin = re * s, im_cos = im * c;
                    values[448 + 2 * pair] = static_cast<float>(static_cast<bfloat16_t>(re_cos - im_sin));
                    values[449 + 2 * pair] = static_cast<float>(static_cast<bfloat16_t>(re_sin + im_cos));
                }
            }
            for (int group = 0; group < 16; ++group) {
                float maximum = 0;
                for (int col = 0; col < 32; ++col) maximum = std::max(maximum, std::fabs(values[group * 32 + col]));
                uint32_t bits;
                std::memcpy(&bits, &maximum, sizeof(bits));
                int exponent = std::max(int(bits >> 23) - 8, 0);
                uint32_t divisor_bits = uint32_t(std::max(exponent, 1)) << 23;
                float divisor;
                std::memcpy(&divisor, &divisor_bits, sizeof(divisor));
                for (int col = 0; col < 32; ++col) {
                    float value = values[group * 32 + col] / divisor * float(exponent != 0);
                    value = std::min(std::max(value, -448.0f), 448.0f);
                    int output_row = head / 8 * padded + row0 + row;
                    int output_col = head % 8 * 512 + group * 32 + col;
                    Data<float8_e4m3_t>(args, 3)[output_row * RowStride(args, 3) + output_col] = float8_e4m3_t(value, RoundMode::CAST_RINT, SaturationMode::ON);
                }
                int offset = row0 * 128 + head % 8 * 256 + group / 2 * 32 + row * 2 + group % 2;
                Data<uint8_t>(args, 4)[head / 8 * RowStride(args, 4) + offset] = exponent;
            }
        }
    }
}
#endif
} // namespace

extern "C" AICORE void kernel_entry(__gm__ int64_t *args)
{
#if defined(__CPU_SIM)
    RunCpu(args);
#elif defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    Run<true, true>(args);
#endif
}
