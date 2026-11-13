// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Buffered inverse RoPE and OCP MXFP8 quantization with packed MX_A_ZZ scales.
#include <cstdint>
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
#if defined(__DAV_VEC__)
template <int Rows>
__aicore__ inline void run_native(__gm__ int64_t* args) {
    const int active = args[5], groups = args[6], heads = groups * 8, capacity_rows = args[7];
    // 80-row tiles use one UB slot; smaller tiles use two.
    constexpr int slots = Rows <= 64 ? 2 : 1;
    constexpr int slot_bytes = ((Rows * 1856 + 255) / 256) * 256;
    static_assert(slots * slot_bytes <= 256 * 1024, "quant buffers must fit UB");
    using Input = Tile<TileType::Vec, bfloat16_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512>;
    using Trig = Tile<TileType::Vec, float, Rows, 32, BLayout::RowMajor, DYNAMIC, 32>;
    using InputShape = TileShape2D<bfloat16_t, DYNAMIC, 512>;
    using TrigShape = TileShape2D<float, DYNAMIC, 32>;
    using DynamicStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    const int block = get_block_idx(args), workers = get_block_num(args);
    const int total = ((capacity_rows + Rows - 1) / Rows) * heads;
    const int count = (total - block + workers - 1) / workers;
    for (int slot = 0; slot < slots; ++slot) {
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
    }
    for (int tick = 0; tick < count + slots - 1; ++tick) {
        if (tick < count) {
            const int work = block + tick * workers;
            const int h = work % heads, t0 = (work / heads) * Rows;
            int rows = active - t0;
            rows = rows < 0 ? 0 : (rows > Rows ? Rows : rows);
            const int base = (tick % slots) * slot_bytes;
            const auto event = static_cast<event_t>(tick % slots);
            Input input(rows);
            Trig cosine(rows), sine(rows);
            TASSIGN(input, base);
            TASSIGN(cosine, base + Rows * 1600);
            TASSIGN(sine, base + Rows * 1728);
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            if (rows > 0) {
                GlobalTensor<bfloat16_t, InputShape, DynamicStride> input_gm(
                    data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + h * 512,
                    InputShape(rows, 512), DynamicStride(1, 1, 1, stride(args, 0), 1));
                GlobalTensor<float, TrigShape, DynamicStride> cos_gm(
                    data<float>(args, 1) + t0 * stride(args, 1),
                    TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 1), 1));
                GlobalTensor<float, TrigShape, DynamicStride> sin_gm(
                    data<float>(args, 2) + t0 * stride(args, 2),
                    TrigShape(rows, 32), DynamicStride(1, 1, 1, stride(args, 2), 1));
                TLOAD(cosine, cos_gm);
                TLOAD(sine, sin_gm);
                TLOAD(input, input_gm);
            }
            set_flag(PIPE_MTE2, PIPE_V, event);
        }
        if (tick >= slots - 1) {
            const int ordinal = tick - (slots - 1);
            const int work = block + ordinal * workers;
            const int h = work % heads, t0 = (work / heads) * Rows;
            int capacity = capacity_rows - t0;
            capacity = capacity > Rows ? Rows : capacity;
            int rows = active - t0;
            rows = rows < 0 ? 0 : (rows > capacity ? capacity : rows);
            const int base = (ordinal % slots) * slot_bytes;
            const auto event = static_cast<event_t>(ordinal % slots);
            Input input(rows);
            Trig cosine(rows), sine(rows);
            Tile<TileType::Vec, float8_e4m3_t, Rows, 512, BLayout::RowMajor, DYNAMIC, 512> output(capacity);
            Tile<TileType::Vec, uint8_t, Rows / 16, 256, BLayout::RowMajor, DYNAMIC, 256> scales(capacity / 16);
            TASSIGN(input, base);
            TASSIGN(output, base + Rows * 1024);
            TASSIGN(scales, base + Rows * 1536);
            TASSIGN(cosine, base + Rows * 1600);
            TASSIGN(sine, base + Rows * 1728);
            wait_flag(PIPE_MTE2, PIPE_V, event);
            o_quant::rotate_quant_vf<decltype(output), decltype(scales), Input, Trig>(
                output.data(), scales.data(), input.data(), cosine.data(), sine.data(), rows);
            set_flag(PIPE_V, PIPE_MTE3, event);
            wait_flag(PIPE_V, PIPE_MTE3, event);
            const int group = h / 8, local_h = h % 8;
            using OutShape = TileShape2D<float8_e4m3_t, DYNAMIC, 512>;
            GlobalTensor<float8_e4m3_t, OutShape, DynamicStride> output_gm(
                data<float8_e4m3_t>(args, 3) + (group * capacity_rows + t0) * stride(args, 3) + local_h * 512,
                OutShape(capacity, 512), DynamicStride(1, 1, 1, stride(args, 3), 1));
            using ScaleShape = TileShape2D<uint8_t, DYNAMIC, 256>;
            GlobalTensor<uint8_t, ScaleShape, DynamicStride> scale_gm(
                data<uint8_t>(args, 4) + group * stride(args, 4) + t0 * 128 + local_h * 256,
                ScaleShape(capacity / 16, 256), DynamicStride(1, 1, 1, 2048, 1));
            TSTORE(output_gm, output);
            TSTORE(scale_gm, scales);
            set_flag(PIPE_MTE3, PIPE_MTE2, event);
        }
    }
    for (int slot = 0; slot < slots; ++slot) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    if (args[6] >= 8) run_native<80>(args);
    else if (args[6] >= 4) run_native<64>(args);
    else if (args[6] >= 2) run_native<32>(args);
    else run_native<16>(args);
#endif
}
