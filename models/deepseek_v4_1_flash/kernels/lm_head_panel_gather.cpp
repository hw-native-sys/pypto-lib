// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Restore FP32 row-major logits from BF16 projection-panel receive storage.
#include "bf16_projection.hpp"
using namespace bf16_projection;

constexpr int Rows = 72;
constexpr int InputBytes = Rows * 256 * sizeof(BFloat);
constexpr int OutputBytes = Rows * 256 * sizeof(float);

template <int Width>
struct GatherTiles {
    using Packed = Tile<TileType::Vec, BFloat, Rows, Width, BLayout::RowMajor, DYNAMIC, Width>;
    using Output = Tile<TileType::Vec, float, Rows, Width, BLayout::RowMajor, DYNAMIC, Width>;
    using ShapeB = TileShape2D<BFloat, DYNAMIC, Width>;
    using ShapeF = TileShape2D<float, DYNAMIC, Width>;
    using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;

    __aicore__ static inline void load(__gm__ int64_t* args, int task, int panels, int slot) {
        int row_tiles = (args[3] + Rows - 1) / Rows;
        int shard = task / (row_tiles * panels);
        int row = task / panels % row_tiles * Rows;
        int column = Width == 256 ? task % panels * 256 : args[4] / 256 * 256;
        int rows = args[3] - row;
        rows = rows < Rows ? rows : Rows;
        Packed packed(rows);
        TASSIGN(packed, slot * InputBytes);
        int offset = shard * args[3] * args[4] + column * args[3] + row * Width;
        GlobalTensor<BFloat, ShapeB, Stride> src(data<BFloat>(args, 0) + offset, ShapeB(rows, Width), Stride(1, 1, 1, Width, 1));
        auto event = static_cast<event_t>(slot);
        wait_flag(PIPE_V, PIPE_MTE2, event);
        TLOAD(packed, src);
        set_flag(PIPE_MTE2, PIPE_V, event);
    }

    __aicore__ static inline void run(__gm__ int64_t* args, int panels) {
        int worker = get_block_idx(args), workers = get_block_num(args);
        int row_tiles = (args[3] + Rows - 1) / Rows;
        int tasks = args[2] * row_tiles * panels;
        if (worker >= tasks) return;
        for (int slot = 0; slot < 2; ++slot) {
            auto event = static_cast<event_t>(slot);
            set_flag(PIPE_V, PIPE_MTE2, event);
            set_flag(PIPE_MTE3, PIPE_V, event);
        }
        load(args, worker, panels, 0);
        int sequence = 0;
        for (int task = worker; task < tasks; task += workers, ++sequence) {
            int slot = sequence % 2;
            auto event = static_cast<event_t>(slot);
            int shard = task / (row_tiles * panels);
            int row = task / panels % row_tiles * Rows;
            int column = Width == 256 ? task % panels * 256 : args[4] / 256 * 256;
            int rows = args[3] - row;
            rows = rows < Rows ? rows : Rows;
            Packed packed(rows);
            Output output(rows);
            TASSIGN(packed, slot * InputBytes);
            TASSIGN(output, 2 * InputBytes + slot * OutputBytes);
            wait_flag(PIPE_MTE2, PIPE_V, event);
            wait_flag(PIPE_MTE3, PIPE_V, event);
            TCVT(output, packed, RoundMode::CAST_NONE);
            set_flag(PIPE_V, PIPE_MTE2, event);
            set_flag(PIPE_V, PIPE_MTE3, event);
            wait_flag(PIPE_V, PIPE_MTE3, event);
            GlobalTensor<float, ShapeF, Stride> dst(data<float>(args, 1) + row * args[2] * args[4] + shard * args[4] + column,
                ShapeF(rows, Width), Stride(1, 1, 1, args[2] * args[4], 1));
            TSTORE(dst, output);
            set_flag(PIPE_MTE3, PIPE_V, event);
            if (task + workers < tasks) load(args, task + workers, panels, 1 - slot);
        }
        for (int slot = 0; slot < 2; ++slot) {
            auto event = static_cast<event_t>(slot);
            wait_flag(PIPE_V, PIPE_MTE2, event);
            wait_flag(PIPE_MTE3, PIPE_V, event);
        }
    }
};

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
#if defined(__DAV_VEC__)
    GatherTiles<256>::run(args, args[4] / 256);
    if (args[4] % 256 == 32) GatherTiles<32>::run(args, 1);
    else if (args[4] % 256 == 64) GatherTiles<64>::run(args, 1);
    else if (args[4] % 256 == 128) GatherTiles<128>::run(args, 1);
#endif
    pipe_barrier(PIPE_ALL);
}
