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

namespace sparse_dequant {
using namespace pto;

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <bool FP4, typename Output, typename Payload, typename Factors, typename Indices>
__tf__ __aicore__ inline void decode_nz(
    typename Output::TileDType __out__ dst, typename Payload::TileDType __in__ src,
    typename Factors::TileDType __in__ factors, typename Indices::TileDType __in__ ids) {
    auto output = (__ubuf__ uint16_t*)__cce_get_tile_ptr(dst);
    auto input = (__ubuf__ uint8_t*)__cce_get_tile_ptr(src);
    auto scale = (__ubuf__ float*)__cce_get_tile_ptr(factors);
    auto indices = (__ubuf__ int32_t*)__cce_get_tile_ptr(ids);
    __VEC_SCOPE__ {
        MaskReg all8 = pset_b8(PAT_ALL);
        MaskReg all16 = pset_b16(PAT_ALL);
        MaskReg all32 = pset_b32(PAT_ALL);
        RegTensor<uint32_t> group;
        RegTensor<float> zero;
        vdup(zero, 0.0f, all32, MODE_ZEROING);
        vci((vector_s32&)group, 0, INC_ORDER);
        vshrs(group, group, (uint32_t)(FP4 ? 3 : 5), all32, MODE_ZEROING);
        for (uint16_t row = 0; row < Output::Rows; ++row) {
            RegTensor<int32_t> index;
            MaskReg valid;
            vlds(index, indices, row, BRC_B32);
            vcmps_ge(valid, index, 0, all32);
            for (uint16_t chunk = 0; chunk < 4; ++chunk) {
                RegTensor<float> x0, x1, s0, s1;
                RegTensor<bfloat16_t> b0, b1, packed, unused;
                if constexpr (FP4) {
                    vector_f4e2m1x2 raw;
                    vlds((RegTensor<uint8_t>&)raw, input, row * 256 + chunk * 64, UNPK4_B8);
                    vcvt(packed, raw, all8, PART_P0);
                    vcvt(x0, packed, all16, PART_EVEN);
                    vcvt(x1, packed, all16, PART_ODD);
                    vgather2(s0, scale + row * 32 + chunk * 8, group, all32);
                    vmul(x0, x0, s0, all32, MODE_ZEROING);
                    vmul(x1, x1, s0, all32, MODE_ZEROING);
                    vsel(x0, x0, zero, valid);
                    vsel(x1, x1, zero, valid);
                    vcvt(b0, x0, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                    vcvt(b1, x1, all32, ROUND_R, RS_DISABLE, PART_ODD);
                    vor(packed, b0, b1, all16);
                } else {
                    vector_f8e4m3 raw0, raw1;
                    vlds((RegTensor<uint8_t>&)raw0, input, row * 512 + chunk * 128, UNPK4_B8);
                    vlds((RegTensor<uint8_t>&)raw1, input, row * 512 + chunk * 128 + 64, UNPK4_B8);
                    vcvt(x0, raw0, all8, PART_P0);
                    vcvt(x1, raw1, all8, PART_P0);
                    vgather2(s0, scale + row * 16 + chunk * 4, group, all32);
                    vgather2(s1, scale + row * 16 + chunk * 4 + 2, group, all32);
                    vmul(x0, x0, s0, all32, MODE_ZEROING);
                    vmul(x1, x1, s1, all32, MODE_ZEROING);
                    vsel(x0, x0, zero, valid);
                    vsel(x1, x1, zero, valid);
                    vcvt(b0, x0, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                    vcvt(b1, x1, all32, ROUND_R, RS_DISABLE, PART_EVEN);
                    vdintlv(packed, unused, b0, b1);
                }
                auto target = output + chunk * Output::Rows * 128 + row * 16;
                vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(Output::Rows << 16), all16, POST_UPDATE);
            }
        }
    }
}
#endif
} // namespace sparse_dequant
