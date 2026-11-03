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
#include "o_rope_quant_vf.hpp"
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
        o_quant::rope_quant_vf<32, 1, decltype(output), decltype(scales), Input, Trig>(output.data(), scales.data(), input.data(), cos.data(), sin.data(), rows);
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
