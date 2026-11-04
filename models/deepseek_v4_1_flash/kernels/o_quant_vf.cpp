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
#include "o_quant_vf.hpp"
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
template <int Rows>
__aicore__ inline void run_native(__gm__ int64_t* args) {
    int active = args[4], groups = args[5], heads = groups * 8;
    using Input = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Prefix = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 448>;
    using Tail = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 64>;
    using PrefixShape = TileShape2D<bfloat16_t, DYNAMIC, 448>;
    using TailShape = TileShape2D<bfloat16_t, DYNAMIC, 64>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int work = get_block_idx(args); work < ((80 + Rows - 1) / Rows) * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * Rows;
        int capacity = 80 - t0; capacity = capacity > Rows ? Rows : capacity;
        int rows = active - t0; rows = rows < 0 ? 0 : (rows > capacity ? capacity : rows);
        Input input(rows);
        Tile<TileType::Vec, float8_e4m3_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512> output(capacity);
        Tile<TileType::Vec, uint8_t, Rows / 16, 256, BLayout::RowMajor, DYNAMIC, 256> scales(capacity / 16);
        TASSIGN(input, 0);
        TASSIGN(output, Rows * 1024); TASSIGN(scales, Rows * 1536);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if (rows > 0) {
            Prefix prefix(rows);
            Tail tail(rows);
            TASSIGN(prefix, 0);
            TASSIGN(tail, 896);
            GlobalTensor<bfloat16_t, PrefixShape, DynamicStride> input_gm(
                data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + h * 512,
                PrefixShape(rows, 448), DynamicStride(1, 1, 1, stride(args, 0), 1));
            GlobalTensor<bfloat16_t, TailShape, DynamicStride> tail_gm(
                data<bfloat16_t>(args, 1) + t0 * stride(args, 1) + h * 64,
                TailShape(rows, 64), DynamicStride(1, 1, 1, stride(args, 1), 1));
            TLOAD(prefix, input_gm);
            TLOAD(tail, tail_gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        o_quant::quant_vf<decltype(output), decltype(scales), Input>(output.data(), scales.data(), input.data(), rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int group = h / 8, local_h = h % 8;
        using OutShape = TileShape2D<float8_e4m3_t, DYNAMIC, 512>;
        GlobalTensor<float8_e4m3_t, OutShape, DynamicStride> output_gm(data<float8_e4m3_t>(args, 2) + (group * 80 + t0) * stride(args, 2) + local_h * 512,
            OutShape(capacity, 512), DynamicStride(1, 1, 1, stride(args, 2), 1));
        using ScaleShape = TileShape2D<uint8_t, DYNAMIC, 256>;
        GlobalTensor<uint8_t, ScaleShape, DynamicStride> scale_gm(
            data<uint8_t>(args, 3) + group * stride(args, 3) + t0 * 128 + local_h * 256,
            ScaleShape(capacity / 16, 256), DynamicStride(1, 1, 1, 2048, 1));
        TSTORE(output_gm, output); TSTORE(scale_gm, scales);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
__aicore__ inline void run_cpu(__gm__ int64_t* args) {
    int active = args[4], heads = args[5] * 8;
    for (int work = get_block_idx(args); work < 5 * heads; work += get_block_num(args)) {
        int h = work % heads, t0 = (work / heads) * 16;
        for (int r = 0; r < 16; ++r) {
            float value[512] = {};
            if (t0 + r < active) {
                auto src = data<bfloat16_t>(args, 0) + (t0 + r) * stride(args, 0) + h * 512;
                for (int d = 0; d < 448; ++d) value[d] = static_cast<float>(src[d]);
                auto tail = data<bfloat16_t>(args, 1) + (t0 + r) * stride(args, 1) + h * 64;
                for (int d = 0; d < 64; ++d) value[448 + d] = static_cast<float>(tail[d]);
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
                    data<float8_e4m3_t>(args, 2)[(h / 8 * 80 + t0 + r) * stride(args, 2) + h % 8 * 512 + g * 32 + d] = float8_e4m3_t(v, RoundMode::CAST_RINT);
                }
                data<uint8_t>(args, 3)[h / 8 * stride(args, 3) + t0 * 128 + h % 8 * 256 + g / 2 * 32 + r * 2 + g % 2] = e;
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
    set_mask_norm(); set_vector_mask(-1, -1);
    if (args[5] >= 8) run_native<80>(args);
    else if (args[5] >= 4) run_native<48>(args);
    else if (args[5] >= 2) run_native<32>(args);
    else run_native<16>(args);
#endif
}
