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
template <int TrigRowStride, int ScaleRowStride, typename Output, typename Scales, typename Input, typename Trig>
__tf__ __aicore__ inline void rope_quant_vf(
    typename Output::TileDType __out__ output, typename Scales::TileDType __out__ scales,
    typename Input::TileDType __in__ input, typename Trig::TileDType __in__ cos_tile,
    typename Trig::TileDType __in__ sin_tile, uint16_t active_rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto cos = (__ubuf__ float*)__cce_get_tile_ptr(cos_tile);
    auto sin = (__ubuf__ float*)__cce_get_tile_ptr(sin_tile);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto scale = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t count64 = 64, count1 = 1;
        MaskReg bytes64 = CreatePredicate<uint8_t>(count64);
        MaskReg half1 = CreatePredicate<uint16_t>(count1);
        RegTensor<uint32_t> lane, lo, hi, tmp, ix1, ix2, ix4, ix8, ix16, pair_index;
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
        // XOR permutations for a butterfly max restricted to each 32-value group.
        vshrs(tmp, lane, 1u, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vsub(tmp, lane, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix1, lane, 1u, all, MODE_ZEROING); vsub(ix1, ix1, tmp, all, MODE_ZEROING);
        vshrs(ix2, lane, 1u, all, MODE_ZEROING); vshls(ix2, ix2, 1u, all, MODE_ZEROING);
        vshrs(tmp, lane, 2u, all, MODE_ZEROING); vshls(tmp, tmp, 2u, all, MODE_ZEROING);
        vsub(tmp, ix2, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix2, lane, 2u, all, MODE_ZEROING); vsub(ix2, ix2, tmp, all, MODE_ZEROING);
        vshrs(ix4, lane, 2u, all, MODE_ZEROING); vshls(ix4, ix4, 2u, all, MODE_ZEROING);
        vshrs(tmp, lane, 3u, all, MODE_ZEROING); vshls(tmp, tmp, 3u, all, MODE_ZEROING);
        vsub(tmp, ix4, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix4, lane, 4u, all, MODE_ZEROING); vsub(ix4, ix4, tmp, all, MODE_ZEROING);
        vshrs(ix8, lane, 3u, all, MODE_ZEROING); vshls(ix8, ix8, 3u, all, MODE_ZEROING);
        vshrs(tmp, lane, 4u, all, MODE_ZEROING); vshls(tmp, tmp, 4u, all, MODE_ZEROING);
        vsub(tmp, ix8, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix8, lane, 8u, all, MODE_ZEROING); vsub(ix8, ix8, tmp, all, MODE_ZEROING);
        vshrs(ix16, lane, 4u, all, MODE_ZEROING); vshls(ix16, ix16, 4u, all, MODE_ZEROING);
        vshrs(tmp, lane, 5u, all, MODE_ZEROING); vshls(tmp, tmp, 5u, all, MODE_ZEROING);
        vsub(tmp, ix16, tmp, all, MODE_ZEROING); vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix16, lane, 16u, all, MODE_ZEROING); vsub(ix16, ix16, tmp, all, MODE_ZEROING);
        vshrs(pair_index, lane, 1u, all, MODE_ZEROING);
        MaskReg even;
        vcmp_gt(even, ix1, lane, all);
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
                    if (chunk == 7) {
                        RegTensor<float> c, s, swapped, a, b, re, im;
                        RegTensor<bfloat16_t> rounded;
                        vgather2(c, cos + row * TrigRowStride, pair_index, all);
                        vgather2(s, sin + row * TrigRowStride, pair_index, all);
                        vmuls(s, s, -1.0f, all, MODE_ZEROING);
                        vselr((RegTensor<uint32_t>&)swapped, (RegTensor<uint32_t>&)x, ix1);
                        vmul(a, x, c, all, MODE_ZEROING);
                        vmul(b, swapped, s, all, MODE_ZEROING);
                        vsub(re, a, b, all, MODE_ZEROING);
                        vadd(im, b, a, all, MODE_ZEROING);
                        vsel(x, re, im, even);
                        vcvt(rounded, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
                        vcvt(x, rounded, all16, PART_EVEN);
                    }
                } else {
                    vdup(x, 0.0f, all, MODE_ZEROING);
                }
                vabs(amax, x, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix16); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix8); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix4); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix2); vmax(amax, amax, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)amax, ix1); vmax(amax, amax, shuffled, all, MODE_ZEROING);
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
                vsts((RegTensor<uint16_t>&)e0, scale, row * ScaleRowStride + chunk * 16, ONEPT_B16, half1);
            }
        }
    }
}

#endif
} // namespace o_quant
