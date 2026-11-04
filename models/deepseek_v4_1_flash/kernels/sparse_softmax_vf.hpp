// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once
#include <pto/pto-inst.hpp>

namespace sparse_softmax {
using namespace pto;

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <typename Weights, typename Scores, typename Indices, typename State>
__tf__ __aicore__ inline void update(
    typename Weights::TileDType __out__ weights, typename State::TileDType __out__ maximum,
    typename State::TileDType __out__ denominator, typename State::TileDType __out__ correction,
    typename Scores::TileDType __in__ scores, typename Indices::TileDType __in__ indices,
    typename State::TileDType __in__ previous_max, typename State::TileDType __in__ previous_den) {
    auto p = (__ubuf__ uint16_t*)__cce_get_tile_ptr(weights);
    auto m = (__ubuf__ float*)__cce_get_tile_ptr(maximum);
    auto l = (__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto c = (__ubuf__ float*)__cce_get_tile_ptr(correction);
    auto qk = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto ids = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto old_m = (__ubuf__ float*)__cce_get_tile_ptr(previous_max);
    auto old_l = (__ubuf__ float*)__cce_get_tile_ptr(previous_den);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        MaskReg all16 = pset_b16(PAT_ALL);
        uint32_t one_count = 1;
        MaskReg one = CreatePredicate<float>(one_count);
        RegTensor<int32_t> i0, i1;
        RegTensor<uint32_t> first;
        RegTensor<float> zero, invalid;
        vdup(first, 0u, all, MODE_ZEROING);
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(invalid, -1e30f, all, MODE_ZEROING);
        vlds(i0, ids, 0, NORM);
        vlds(i1, ids, 64, NORM);
        MaskReg valid0, valid1;
        vcmps_ge(valid0, i0, 0, all);
        vcmps_ge(valid1, i1, 0, all);
        for (uint16_t row = 0; row < Weights::Rows; ++row) {
            RegTensor<float> x0, x1, mx, old, sum0, sum1;
            vlds(x0, qk, row * 128, NORM);
            vlds(x1, qk, row * 128 + 64, NORM);
            vmuls(x0, x0, 0.0441941731f, all, MODE_ZEROING);
            vmuls(x1, x1, 0.0441941731f, all, MODE_ZEROING);
            vsel(x0, x0, invalid, valid0);
            vsel(x1, x1, invalid, valid1);
            vmax(mx, x0, x1, all, MODE_ZEROING);
            vcmax(mx, mx, all, MODE_ZEROING);
            vselr((RegTensor<uint32_t>&)mx, (RegTensor<uint32_t>&)mx, first);
            vlds(old, old_m, row, BRC_B32);
            vmax(mx, mx, old, all, MODE_ZEROING);
            vsub(x0, x0, mx, all, MODE_ZEROING);
            vsub(x1, x1, mx, all, MODE_ZEROING);
            vexp(x0, x0, all, MODE_ZEROING);
            vexp(x1, x1, all, MODE_ZEROING);
            vsel(x0, x0, zero, valid0);
            vsel(x1, x1, zero, valid1);
            vcadd(sum0, x0, all, MODE_ZEROING);
            vcadd(sum1, x1, all, MODE_ZEROING);
            vadd(sum0, sum0, sum1, one, MODE_ZEROING);
            vsts(mx, m, row, ONEPT_B32, one);
            vsts(sum0, c, row, ONEPT_B32, one);
            RegTensor<bfloat16_t> b0, b1, packed, unused;
            vcvt(b0, x0, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(b1, x1, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vdintlv(packed, unused, b0, b1);
            auto target = p + row * 16;
            vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(Weights::Rows << 16), all16, POST_UPDATE);
        }
        // Finish per-head maximum and tile-sum stores before loading the state vectors.
        mem_bar(VST_VLD);
        uint32_t state_count = Weights::Rows;
        MaskReg state_mask = CreatePredicate<float>(state_count);
        RegTensor<float> previous, current, alpha, old_sum, tile_sum;
        vlds(previous, old_m, 0, NORM);
        vlds(current, m, 0, NORM);
        vlds(old_sum, old_l, 0, NORM);
        vlds(tile_sum, c, 0, NORM);
        vsub(alpha, previous, current, state_mask, MODE_ZEROING);
        vexp(alpha, alpha, state_mask, MODE_ZEROING);
        vmul(old_sum, old_sum, alpha, state_mask, MODE_ZEROING);
        vadd(old_sum, old_sum, tile_sum, state_mask, MODE_ZEROING);
        vsts(old_sum, l, 0, NORM_B32, state_mask);
        vsts(alpha, c, 0, NORM_B32, state_mask);
    }
}

template <typename Values, typename State>
__tf__ __aicore__ inline void accumulate(
    typename Values::TileDType __out__ result, typename Values::TileDType __in__ previous,
    typename Values::TileDType __in__ weighted, typename State::TileDType __in__ correction) {
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(result);
    auto old = (__ubuf__ float*)__cce_get_tile_ptr(previous);
    auto pv = (__ubuf__ float*)__cce_get_tile_ptr(weighted);
    auto corr = (__ubuf__ float*)__cce_get_tile_ptr(correction);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < Values::Rows; ++row) {
            RegTensor<float> c;
            vlds(c, corr, row, BRC_B32);
            for (uint16_t col = 0; col < Values::Cols; col += 64) {
                RegTensor<float> old_value, value;
                vlds(old_value, old, row * Values::Cols + col, NORM);
                vlds(value, pv, row * Values::Cols + col, NORM);
                vmul(old_value, old_value, c, all, MODE_ZEROING);
                vadd(value, old_value, value, all, MODE_ZEROING);
                vsts(value, dst, row * Values::Cols + col, NORM_B32, all);
            }
        }
    }
}

template <typename Result, typename Values, typename State>
__tf__ __aicore__ inline void finalize(
    typename Result::TileDType __out__ result, typename State::TileDType __out__ multiplier,
    typename Values::TileDType __in__ numerator, typename State::TileDType __in__ maximum,
    typename State::TileDType __in__ denominator, typename State::TileDType __in__ sinks) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(result);
    auto scale = (__ubuf__ float*)__cce_get_tile_ptr(multiplier);
    auto num = (__ubuf__ float*)__cce_get_tile_ptr(numerator);
    auto max = (__ubuf__ float*)__cce_get_tile_ptr(maximum);
    auto den = (__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto sink = (__ubuf__ float*)__cce_get_tile_ptr(sinks);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        uint32_t head_count = Values::Rows;
        MaskReg heads = CreatePredicate<float>(head_count);
        RegTensor<float> m, l, s, mx, corr;
        vlds(m, max, 0, NORM);
        vlds(l, den, 0, NORM);
        vlds(s, sink, 0, NORM);
        vmax(mx, m, s, heads, MODE_ZEROING);
        vsub(corr, m, mx, heads, MODE_ZEROING);
        vexp(corr, corr, heads, MODE_ZEROING);
        vmul(l, l, corr, heads, MODE_ZEROING);
        vsub(s, s, mx, heads, MODE_ZEROING);
        vexp(s, s, heads, MODE_ZEROING);
        vadd(s, l, s, heads, MODE_ZEROING);
        vdiv(s, corr, s, heads, MODE_ZEROING);
        vsts(s, scale, 0, NORM_B32, heads);
        mem_bar(VST_VLD);
        for (uint16_t row = 0; row < Values::Rows; ++row) {
            RegTensor<float> c;
            vlds(c, scale, row, BRC_B32);
            for (uint16_t col = 0; col < Values::Cols; col += 64) {
                RegTensor<float> value;
                RegTensor<bfloat16_t> converted;
                vlds(value, num, row * Values::Cols + col, NORM);
                vmul(value, value, c, all, MODE_ZEROING);
                vcvt(converted, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
                // Forward BF16 stores stay behind unread FP32 values when the buffers alias.
                vsts(converted, dst, row * Values::Cols + col, PK_B32, all);
            }
        }
    }
}

#endif
} // namespace sparse_softmax
