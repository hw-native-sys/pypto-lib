// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#pragma once
#include <pto/pto-inst.hpp>
namespace o_quant {
using namespace pto;
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <typename Output, typename Scales, typename Input>
__tf__ __aicore__ inline void quant_vf(
    typename Output::TileDType __out__ output, typename Scales::TileDType __out__ scales,
    typename Input::TileDType __in__ input, uint16_t active_rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto scale = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t count64 = 64, count1 = 1;
        MaskReg bytes64 = CreatePredicate<uint8_t>(count64);
        MaskReg half1 = CreatePredicate<uint16_t>(count1);
        MaskReg lower32 = pset_b32(PAT_VL32);
        RegTensor<uint32_t> lane, lo, hi, tmp, upper_index;
        RegTensor<int32_t> izero, ione;
        RegTensor<float> zero, one, low, high;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vdup(lo, 0u, all, MODE_ZEROING);
        vdup(hi, 32u, all, MODE_ZEROING);
        vdup(izero, 0, all, MODE_ZEROING);
        vdup(ione, 1, all, MODE_ZEROING);
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(one, 1.0f, all, MODE_ZEROING);
        vdup(low, -448.0f, all, MODE_ZEROING);
        vdup(high, 448.0f, all, MODE_ZEROING);
        // Upper 32-value reduction group.
        vshrs(tmp, lane, 5u, all, MODE_ZEROING);
        vshls(tmp, tmp, 5u, all, MODE_ZEROING);
        vsub(upper_index, lane, tmp, all, MODE_ZEROING);
        vadds(upper_index, upper_index, 32u, all, MODE_ZEROING);
        RegTensor<uint8_t> byte_index;
        vci((RegTensor<int8_t>&)byte_index, (int8_t)0, INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_index, (RegTensor<int16_t>&)byte_index, (int16_t)4, all8);
        for (uint16_t row = 0; row < Input::Rows; ++row) {
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<float> x, amax, shuffled, divisor, floor_mask;
                RegTensor<int32_t> exponent, nonzero;
                if (row < active_rows) {
                    RegTensor<bfloat16_t> raw;
                    vlds(raw, src, row * 512 + chunk * 64, UNPK_B16);
                    vcvt(x, raw, all16, PART_EVEN);
                } else {
                    vdup(x, 0.0f, all, MODE_ZEROING);
                }
                vabs(amax, x, all, MODE_ZEROING);
                RegTensor<float> lower_max, upper_max;
                // Both 32-value groups use the lower-half mask and lane-zero maxima.
                vcmax(lower_max, amax, lower32);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, upper_index);
                vcmax(upper_max, shuffled, lower32);
                vselr((RegTensor<uint32_t>&)lower_max, (RegTensor<uint32_t>&)lower_max, lo);
                vselr((RegTensor<uint32_t>&)upper_max, (RegTensor<uint32_t>&)upper_max, lo);
                vsel(amax, lower_max, upper_max, lower32);
                vshrs(exponent, (RegTensor<int32_t>&)amax, 23, all, MODE_ZEROING);
                vadds(exponent, exponent, -8, all, MODE_ZEROING);
                vmax(exponent, exponent, izero, all, MODE_ZEROING);
                vmax(nonzero, exponent, ione, all, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)divisor, nonzero, 23, all, MODE_ZEROING);
                vdiv(x, x, divisor, all, MODE_ZEROING);
                MaskReg nonzero_mask;
                vcmps_gt(nonzero_mask, exponent, 0, all);
                vsel(floor_mask, one, zero, nonzero_mask);
                vmul(x, x, floor_mask, all, MODE_ZEROING);
                vmax(x, x, low, all, MODE_ZEROING);
                vmin(x, x, high, all, MODE_ZEROING);
                vector_f8e4m3 payload;
                RegTensor<uint8_t> packed;
                vcvt(payload, x, all, ROUND_R, RS_DISABLE, PART_P0);
                vselr(packed, (RegTensor<uint8_t>&)payload, byte_index);
                vsts(packed, dst, row * 512 + chunk * 64, NORM_B8, bytes64);
                RegTensor<uint32_t> e0, e1;
                vselr(e0, (RegTensor<uint32_t>&)exponent, lo);
                vselr(e1, (RegTensor<uint32_t>&)exponent, hi);
                vshls(e1, e1, 8u, all, MODE_ZEROING);
                vor(e0, e0, e1, all);
                // Scale offsets are in uint16 pairs; each 64D chunk advances 16 pairs.
                vsts((RegTensor<uint16_t>&)e0, scale, row + chunk * 16, ONEPT_B16, half1);
            }
        }
    }
}

#endif
} // namespace o_quant
