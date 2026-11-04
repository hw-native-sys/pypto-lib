// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T> __aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
    auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
    return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}
__aicore__ inline int64_t stride(__gm__ int64_t* args, int i) {
    return reinterpret_cast<__gm__ Tensor*>(args[i])->strides[0];
}
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
using Values = Tile<TileType::Vec, bfloat16_t, 64, 64, BLayout::RowMajor, DYNAMIC, 64>;
using Trig = Tile<TileType::Vec, float, 1, 32>;
__tf__ __aicore__ inline void rotate(
    Values::TileDType __out__ output, Values::TileDType __in__ input,
    Trig::TileDType __in__ cosine, Trig::TileDType __in__ sine, uint16_t heads) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cos = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sin = (__ubuf__ float*)__cce_get_tile_ptr(sine);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL);
        RegTensor<uint32_t> lane, tmp, ix1, pair_index;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(tmp, lane, 1u, all, MODE_ZEROING);
        vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vsub(tmp, lane, tmp, all, MODE_ZEROING);
        vshls(tmp, tmp, 1u, all, MODE_ZEROING);
        vadds(ix1, lane, 1u, all, MODE_ZEROING);
        vsub(ix1, ix1, tmp, all, MODE_ZEROING);
        vshrs(pair_index, lane, 1u, all, MODE_ZEROING);
        MaskReg even;
        vcmp_gt(even, ix1, lane, all);
        RegTensor<float> c, s;
        vgather2(c, cos, pair_index, all);
        vgather2(s, sin, pair_index, all);
        vmuls(s, s, -1.0f, all, MODE_ZEROING);
        for (uint16_t head = 0; head < heads; ++head) {
            RegTensor<bfloat16_t> raw;
            RegTensor<float> x, swapped, a, b, re, im;
            vlds(raw, src, head * 64, UNPK_B16);
            vcvt(x, raw, all16, PART_EVEN);
            vselr((RegTensor<uint32_t>&)swapped, (RegTensor<uint32_t>&)x, ix1);
            vmul(a, x, c, all, MODE_ZEROING);
            vmul(b, swapped, s, all, MODE_ZEROING);
            vsub(re, a, b, all, MODE_ZEROING);
            vadd(im, b, a, all, MODE_ZEROING);
            vsel(x, re, im, even);
            vcvt(raw, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(raw, dst, head * 64, PK_B32, all);
        }
    }
}
__aicore__ inline void run_native(__gm__ int64_t* args) {
    int active = args[4], heads = args[5] * 8;
    using ValueShape = TileShape2D<bfloat16_t, DYNAMIC, 64>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using TrigShape = TileShape2D<float, 1, 32>;
    Values input(heads), output(heads);
    Trig cos, sin;
    TASSIGN(input, 0);
    TASSIGN(output, 8192);
    TASSIGN(cos, 16384);
    TASSIGN(sin, 16512);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int token = get_block_idx(args); token < active; token += get_block_num(args)) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        GlobalTensor<bfloat16_t, ValueShape, DynamicStride> input_gm(
            data<bfloat16_t>(args, 0) + token * stride(args, 0) + 448,
            ValueShape(heads, 64), DynamicStride(1, 1, 1, 512, 1));
        GlobalTensor<float, TrigShape, BaseShape2D<float, 1, 32>> cos_gm(
            data<float>(args, 1) + token * stride(args, 1));
        GlobalTensor<float, TrigShape, BaseShape2D<float, 1, 32>> sin_gm(
            data<float>(args, 2) + token * stride(args, 2));
        TLOAD(input, input_gm);
        TLOAD(cos, cos_gm);
        TLOAD(sin, sin_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        rotate(output.data(), input.data(), cos.data(), sin.data(), heads);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GlobalTensor<bfloat16_t, ValueShape, BaseShape2D<bfloat16_t, 64, 64>> output_gm(
            data<bfloat16_t>(args, 3) + token * stride(args, 3), ValueShape(heads, 64));
        TSTORE(output_gm, output);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
__aicore__ inline void run_cpu(__gm__ int64_t* args) {
    int active = args[4], heads = args[5] * 8;
    for (int work = get_block_idx(args); work < active * heads; work += get_block_num(args)) {
        int h = work % heads, row = work / heads;
        auto src = data<bfloat16_t>(args, 0) + row * stride(args, 0) + h * 512;
        auto dst = data<bfloat16_t>(args, 3) + row * stride(args, 3) + h * 64;
        for (int p = 0; p < 32; ++p) {
            float c = data<float>(args, 1)[row * stride(args, 1) + p];
            float s = -data<float>(args, 2)[row * stride(args, 2) + p];
            float re = static_cast<float>(src[448 + p * 2]);
            float im = static_cast<float>(src[449 + p * 2]);
            dst[p * 2] = static_cast<bfloat16_t>(re * c - im * s);
            dst[p * 2 + 1] = static_cast<bfloat16_t>(re * s + im * c);
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
