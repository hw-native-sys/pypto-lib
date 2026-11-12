// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Rotary embedding and group-16 cache publication with selectable E2M1 rounding.
// Batch maxima/scale phases and packed cache rows adapted from Huawei Technologies
// kv_compress_epilog_v2_quant_fp4_g16.h, Copyright (c) 2026, under the same license.
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
template <typename T, int C>
using Vec = Tile<TileType::Vec, T, 1, C>;
template <typename T, int C>
using Batch = Tile<TileType::Vec, T, 8, C>;
template <typename T, int C>
using GM = GlobalTensor<T, TileShape2D<T, 1, C>, BaseShape2D<T, 1, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
template <bool RoundEven>
__tf__ __aicore__ inline void quantize(
    Batch<uint8_t, 320>::TileDType __out__ packed,
    Batch<bfloat16_t, 512>::TileDType __in__ raw,
    Batch<float, 32>::TileDType __out__ scratch, uint16_t rows) {
    auto src_base = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(raw);
    auto dst_base = (__ubuf__ uint8_t*)__cce_get_tile_ptr(packed);
    auto rounded_scales = (__ubuf__ float*)__cce_get_tile_ptr(scratch);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        auto src = src_base + row * 512;
        auto dst = dst_base + row * 320;
        auto factors = reinterpret_cast<__ubuf__ bfloat16_t*>(dst + 256);
        for (int group = 0; group < 32; ++group) {
            float maximum = 0.0f;
            for (int i = 0; i < 16; ++i) maximum = std::max(maximum, std::abs(static_cast<float>(src[group * 16 + i])));
            bfloat16_t stored = static_cast<bfloat16_t>(std::max(maximum * (1.0f / 6.0f), 0.001953125f));
            factors[group] = stored;
            for (int pair = 0; pair < 8; ++pair) {
                int byte = 0;
                for (int half = 0; half < 2; ++half) {
                    float value = static_cast<float>(src[group * 16 + pair * 2 + half]) / static_cast<float>(stored);
                    float mag = std::abs(value);
                    int code = (mag > 0.25f) + (RoundEven ? mag >= 0.75f : mag > 0.75f) + (mag > 1.25f) + (RoundEven ? mag >= 1.75f : mag > 1.75f) +
                               (mag > 2.5f) + (RoundEven ? mag >= 3.5f : mag > 3.5f) + (mag > 5.0f);
                    code |= (std::bit_cast<int32_t>(value) >> 28) & 8;
                    byte |= code << (half * 4);
                }
                dst[group * 8 + pair] = byte;
            }
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t half_count = 32;
        MaskReg lower = plt_b32(half_count, POST_UPDATE);
        MaskReg full16 = pset_b16(PAT_ALL);
        RegTensor<uint16_t> abs_mask;
        vdup(abs_mask, (uint16_t)0x7fff, full16, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            auto src = src_base + row * 512;
            auto dst = dst_base + row * 320;
            auto factors = reinterpret_cast<__ubuf__ bfloat16_t*>(dst + 256);
            vector_align max_store;
            for (uint16_t chunk = 0; chunk < 4; ++chunk) {
                RegTensor<bfloat16_t> values;
                RegTensor<uint16_t> magnitude, maximum;
                vlds(values, src, chunk * 128, NORM);
                vand(magnitude, (RegTensor<uint16_t>&)values, abs_mask, full16, MODE_ZEROING);
                vcgmax(maximum, magnitude, full16, MODE_ZEROING);
                vstus(max_store, 8, (RegTensor<bfloat16_t>&)maximum, factors + chunk * 8);
            }
            vstas(max_store, factors, 0);
        }
        mem_bar(VST_VLD);
        for (uint16_t row = 0; row < rows; ++row) {
            auto factors = reinterpret_cast<__ubuf__ bfloat16_t*>(dst_base + row * 320 + 256);
            RegTensor<bfloat16_t> bf_scales;
            RegTensor<float> maxima, rounded;
            vlds(bf_scales, factors, 0, UNPK_B16);
            vcvt(maxima, bf_scales, lower, PART_EVEN);
            vmuls(maxima, maxima, 1.0f / 6.0f, lower, MODE_ZEROING);
            vmaxs(maxima, maxima, 0.001953125f, lower, MODE_ZEROING);
            vcvt(bf_scales, maxima, lower, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(rounded, bf_scales, lower, PART_EVEN);
            vsts(bf_scales, factors, 0, PK_B32, lower);
            vsts(rounded, rounded_scales, row * 32, NORM_B32, lower);
        }
        mem_bar(VST_VLD);
        uint32_t packed_count = 64;
        MaskReg packed16 = plt_b16(packed_count, POST_UPDATE);
        RegTensor<uint32_t> magnitude_mask, sign_mask32, low_mask, jam_mask;
        vdup(magnitude_mask, (uint32_t)0x7fffffff, full, MODE_ZEROING);
        vdup(sign_mask32, (uint32_t)0x80000000, full, MODE_ZEROING);
        vdup(low_mask, (uint32_t)0xffff, full, MODE_ZEROING);
        vdup(jam_mask, (uint32_t)0x10000, full, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            auto src = src_base + row * 512;
            auto dst = dst_base + row * 320;
            RegTensor<uint32_t> indices;
            vci((RegTensor<int32_t>&)indices, 0, INC_ORDER);
            vshrs(indices, indices, (uint32_t)4, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<bfloat16_t> bf_values;
                RegTensor<float> values, scale, normalized;
                vlds(bf_values, src, chunk * 64, UNPK_B16);
                vcvt(values, bf_values, full, PART_EVEN);
                vgather2(scale, rounded_scales + row * 32 + chunk * 4, indices, full);
                vdiv(normalized, values, scale, full, MODE_ZEROING);
                RegTensor<uint32_t> bits, absolute, sign_zero, low, jammed;
                vand(absolute, (RegTensor<uint32_t>&)normalized, magnitude_mask, full, MODE_ZEROING);
                vand(sign_zero, (RegTensor<uint32_t>&)normalized, sign_mask32, full, MODE_ZEROING);
                MaskReg selected;
                // Manual encoding maps NaNs to zero magnitude while retaining the sign.
                vcmps_gt(selected, absolute, (uint32_t)0x7f800000, full);
                vsel(bits, sign_zero, (RegTensor<uint32_t>&)normalized, selected);
                if constexpr (!RoundEven) {
                    // Only these RNE ties round upward; C2A requires the lower magnitude.
                    MaskReg tie, next_tie;
                    vcmps_eq(tie, absolute, (uint32_t)0x3f400000, full);
                    vcmps_eq(next_tie, absolute, (uint32_t)0x3fe00000, full);
                    por(tie, tie, next_tie, full);
                    vcmps_eq(next_tie, absolute, (uint32_t)0x40600000, full);
                    por(tie, tie, next_tie, full);
                    vadds(jammed, bits, (uint32_t)-1, full, MODE_ZEROING);
                    vsel(bits, jammed, bits, tie);
                }
                // Round-to-odd BF16 preserves the side of every FP4 midpoint; plain
                // BF16 rounding would introduce double-rounding at those boundaries.
                vand(low, bits, low_mask, full, MODE_ZEROING);
                vcmps_ne(selected, low, (uint32_t)0, full);
                vor(jammed, bits, jam_mask, full, MODE_ZEROING);
                vsel(bits, jammed, bits, selected);
                RegTensor<bfloat16_t> bf_bridge;
                RegTensor<uint16_t> dense, unused;
                vcvt(bf_bridge, (RegTensor<float>&)bits, full, ROUND_Z, RS_DISABLE, PART_EVEN);
                vdintlv(dense, unused, (RegTensor<uint16_t>&)bf_bridge, (RegTensor<uint16_t>&)bf_bridge);
                vector_f4e2m1x2 fp4;
                vcvt(fp4, (vector_bf16&)dense, packed16, ROUND_R, PART_P0);
                vsts((RegTensor<uint8_t>&)fp4, dst, chunk * 32, PK4_B32, packed16);
            }
        }
    }
#endif
}

__tf__ __aicore__ inline void rotate(
    Batch<bfloat16_t, 512>::TileDType __out__ output,
    Batch<bfloat16_t, 512>::TileDType __in__ row,
    Batch<float, 32>::TileDType __in__ cosine,
    Batch<float, 32>::TileDType __in__ sine, uint16_t rows) {
    auto values = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(row);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row_index = 0; row_index < rows; ++row_index) {
        for (int pair = 0; pair < 32; ++pair) {
            float even = static_cast<float>(values[row_index * 512 + 448 + pair * 2]);
            float odd = static_cast<float>(values[row_index * 512 + 449 + pair * 2]);
            dst[row_index * 512 + 448 + pair * 2] = static_cast<bfloat16_t>(even * cp[row_index * 32 + pair] - odd * sp[row_index * 32 + pair]);
            dst[row_index * 512 + 449 + pair * 2] = static_cast<bfloat16_t>(even * sp[row_index * 32 + pair] + odd * cp[row_index * 32 + pair]);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg half = plt_b32(count, POST_UPDATE);
        for (uint16_t row_index = 0; row_index < rows; ++row_index) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> raw, even, odd, cosv, sinv, a, b, real, imag, result, unused;
            vlds(bf, values, row_index * 512 + 448, UNPK_B16);
            vcvt(raw, bf, full, PART_EVEN);
            vlds(cosv, cp, row_index * 32, NORM);
            vlds(sinv, sp, row_index * 32, NORM);
            vdintlv(even, odd, raw, raw);
            vmul(a, even, cosv, half, MODE_ZEROING);
            vmul(b, odd, sinv, half, MODE_ZEROING);
            vsub(real, a, b, half);
            vmul(a, even, sinv, half, MODE_ZEROING);
            vmul(b, odd, cosv, half, MODE_ZEROING);
            vadd(imag, a, b, half, MODE_ZEROING);
            vintlv(result, unused, real, imag);
            vcvt(bf, result, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row_index * 512 + 448, PK_B32, full);
        }
        mem_bar(VST_VLD);
    }
#endif
}

#endif
} // namespace
