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
            RegTensor<float> x0, x1, mx, old, den, corr, sum0, sum1;
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
            vsub(corr, old, mx, all, MODE_ZEROING);
            vexp(corr, corr, all, MODE_ZEROING);
            vsub(x0, x0, mx, all, MODE_ZEROING);
            vsub(x1, x1, mx, all, MODE_ZEROING);
            vexp(x0, x0, all, MODE_ZEROING);
            vexp(x1, x1, all, MODE_ZEROING);
            vsel(x0, x0, zero, valid0);
            vsel(x1, x1, zero, valid1);
            vcadd(sum0, x0, all, MODE_ZEROING);
            vcadd(sum1, x1, all, MODE_ZEROING);
            vlds(den, old_l, row, BRC_B32);
            vmul(den, den, corr, one, MODE_ZEROING);
            vadd(sum0, sum0, sum1, one, MODE_ZEROING);
            vadd(den, den, sum0, one, MODE_ZEROING);
            vsts(mx, m, row, ONEPT_B32, one);
            vsts(den, l, row, ONEPT_B32, one);
            vsts(corr, c, row, ONEPT_B32, one);
            RegTensor<bfloat16_t> b0, b1, packed, unused;
            vcvt(b0, x0, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(b1, x1, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vdintlv(packed, unused, b0, b1);
            auto target = p + row * 16;
            vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(Weights::Rows << 16), all16, POST_UPDATE);
        }
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
#endif
} // namespace sparse_softmax
