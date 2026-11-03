// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <cstring>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T, int R, int C> using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T> __aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}
__aicore__ inline int64_t stride(__gm__ int64_t* args, int i) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->strides[0];
}
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <typename Output, typename Scales, typename Input, typename Trig>
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
        for (uint16_t row = 0; row < 16; ++row) {
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
                        vgather2(c, cos + row * 32, pair_index, all);
                        vgather2(s, sin + row * 32, pair_index, all);
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
                // MX_A_ZZ: pairs of scale groups, then 16 rows, then two bytes.
                vsts((RegTensor<uint16_t>&)e0, scale, chunk * 16 + row, ONEPT_B16, half1);
            }
        }
    }
}

__aicore__ inline void run_native(__gm__ int64_t* args) {
    int active = args[5], groups = args[6], heads = groups * 8;
    using Input = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Trig = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, DYNAMIC, 32>;
    using InputShape = TileShape2D<bfloat16_t, DYNAMIC, 512>;
    using TrigShape = TileShape2D<float, DYNAMIC, 32>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int work = get_block_idx(args); work < 5 * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * 16;
        int rows = active - t0; rows = rows < 0 ? 0 : (rows > 16 ? 16 : rows);
        Input input(rows); Trig cos(rows), sin(rows);
        Vec<float8_e4m3_t, 16, 512> output;
        Vec<uint8_t, 1, 256> scales;
        TASSIGN(input, 0); TASSIGN(cos, 16384); TASSIGN(sin, 18432);
        TASSIGN(output, 24576); TASSIGN(scales, 32768);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if (rows > 0) {
            GlobalTensor<bfloat16_t, InputShape, DynamicStride> input_gm(data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + h * 512,
                InputShape(rows, 512), DynamicStride(1, 1, 1, stride(args, 0), 1));
            GlobalTensor<float, TrigShape, DynamicStride> cos_gm(data<float>(args, 1) + t0 * stride(args, 1),
                TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 1), 1));
            GlobalTensor<float, TrigShape, DynamicStride> sin_gm(data<float>(args, 2) + t0 * stride(args, 2),
                TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 2), 1));
            TLOAD(input, input_gm); TLOAD(cos, cos_gm); TLOAD(sin, sin_gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        rope_quant_vf<decltype(output), decltype(scales), Input, Trig>(output.data(), scales.data(), input.data(), cos.data(), sin.data(), rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int group = h / 8, local_h = h % 8;
        using OutShape = TileShape2D<float8_e4m3_t, 16, 512>;
        GlobalTensor<float8_e4m3_t, OutShape, DynamicStride> output_gm(data<float8_e4m3_t>(args, 3) + (group * 80 + t0) * stride(args, 3) + local_h * 512,
            OutShape(), DynamicStride(1, 1, 1, stride(args, 3), 1));
        using ScaleShape = TileShape2D<uint8_t, 1, 256>;
        GlobalTensor<uint8_t, ScaleShape, BaseShape2D<uint8_t, 1, 256>> scale_gm(
            data<uint8_t>(args, 4) + group * stride(args, 4) + t0 * 128 + local_h * 256);
        TSTORE(output_gm, output); TSTORE(scale_gm, scales);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
__aicore__ inline void run_cpu(__gm__ int64_t* args) {
    int active = args[5], heads = args[6] * 8;
    for (int work = get_block_idx(args); work < 5 * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * 16;
        for (int r = 0; r < 16; ++r) {
            float value[512] = {};
            if (t0 + r < active) {
                auto src = data<bfloat16_t>(args, 0) + (t0 + r) * stride(args, 0) + h * 512;
                for (int d = 0; d < 512; ++d) value[d] = static_cast<float>(src[d]);
                for (int p = 0; p < 32; ++p) {
                    float c = data<float>(args, 1)[(t0 + r) * stride(args, 1) + p];
                    float s = -data<float>(args, 2)[(t0 + r) * stride(args, 2) + p];
                    float re = value[448 + p * 2], im = value[449 + p * 2];
                    value[448 + p * 2] = static_cast<float>(static_cast<bfloat16_t>(re * c - im * s));
                    value[449 + p * 2] = static_cast<float>(static_cast<bfloat16_t>(re * s + im * c));
                }
            }
            for (int g = 0; g < 16; ++g) {
                float maximum = 0;
                for (int d = 0; d < 32; ++d) maximum = std::max(maximum, std::fabs(value[g * 32 + d]));
                uint32_t bits; std::memcpy(&bits, &maximum, 4);
                int e = std::max(int(bits >> 23) - 8, 0);
                uint32_t dbits = uint32_t(std::max(e, 1)) << 23; float divisor; std::memcpy(&divisor, &dbits, 4);
                for (int d = 0; d < 32; ++d) {
                    float v = value[g * 32 + d] / divisor * float(e != 0);
                    v = std::min(std::max(v, -448.0f), 448.0f);
                    data<float8_e4m3_t>(args, 3)[(h / 8 * 80 + t0 + r) * stride(args, 3) + h % 8 * 512 + g * 32 + d] = static_cast<float8_e4m3_t>(v);
                }
                data<uint8_t>(args, 4)[h / 8 * stride(args, 4) + t0 * 128 + h % 8 * 256 + g / 2 * 32 + r * 2 + g % 2] = e;
            }
        }
    }
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__CPU_SIM)
    run_cpu(args);
#elif defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1); run_native(args);
#endif
}
