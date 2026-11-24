// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 vocabulary projection with overlapping dual-AIV peer publication.
#include "bf16_projection.hpp"
#include "mixed_entry.hpp"

using namespace bf16_projection;

constexpr int PROJECT_M_TILE = 224;
constexpr int PROJECT_N_TILE = 256;
constexpr int PUBLISH_M_TILE = PROJECT_M_TILE / 2;
constexpr int PUBLISH_ROW_BYTES = PROJECT_N_TILE * sizeof(BFloat);
constexpr int PUBLISH_SLOT_BYTES = PROJECT_M_TILE * PUBLISH_ROW_BYTES;

#if defined(__DAV_CUBE__)
struct PublishPanel {
    static constexpr bool full_rows = false;
    int count = 0;

    template <typename Acc>
    __aicore__ inline void operator()(Acc& acc, int rows, int column, int row) {
        int slot = count % 2;
        wait_intra_block(PIPE_FIX, slot * 2 + 1);
        wait_intra_block(PIPE_FIX, slot * 2 + 17);
        Tile<TileType::Vec, BFloat, PROJECT_M_TILE, PROJECT_N_TILE, BLayout::RowMajor, DYNAMIC, Acc::Cols> raw(rows);
        TASSIGN(raw, slot * PUBLISH_SLOT_BYTES);
        TMOV<decltype(raw), Acc, AccToVecMode::SingleModeVec0>(raw, acc);
        TMOV<decltype(raw), Acc, AccToVecMode::SingleModeVec1>(raw, acc);
        set_intra_block(PIPE_FIX, slot * 2);
        set_intra_block(PIPE_FIX, slot * 2 + 16);
        ++count;
    }

    __aicore__ inline void finish() {
        for (int slot = 0; slot < 2; ++slot) {
            wait_intra_block(PIPE_FIX, slot * 2 + 1);
            wait_intra_block(PIPE_FIX, slot * 2 + 17);
        }
    }
};

template <int Vocab>
__aicore__ inline void cube(__gm__ int64_t* args) {
    auto hidden = reinterpret_cast<__gm__ Tensor*>(args[0]);
    PublishPanel emit;
    project<5120, Vocab, PROJECT_N_TILE, 128, 64, 4, Layout::NZ, PROJECT_M_TILE>(
        data<BFloat>(args, 0), data<BFloat>(args, 1), args[5] * args[6], hidden->shapes[0],
        hidden->strides[0], 0, args[8], get_block_idx(args), get_block_num(args), emit);
    emit.finish();
}
#endif

#if defined(__DAV_VEC__)
template <int Width>
__aicore__ inline void send_panel(__gm__ int64_t* args, int row, int column, int count) {
    int lane = get_sub_block_id(args);
    int slot = count % 2;
    int flag = lane * 16;
    wait_intra_block(PIPE_MTE3, flag + slot * 2);
    int band = args[8] / 2;
    int begin = row + lane * band;
    int end = begin + band;
    int total = args[5] * args[6];
    end = end < total ? end : total;
    auto ctx = reinterpret_cast<__gm__ int64_t*>(args[10]);
    int local_rank = ctx[0];
    auto local_window = data<BFloat>(args, 2);
    for (int first = begin; first < end;) {
        int owner = first / args[9] % args[5];
        int last = (first / args[9] + 1) * args[9];
        last = last < end ? last : end;
        int rows = last - first;
        int peer = args[4] + owner;
        auto remote = reinterpret_cast<__gm__ BFloat*>(reinterpret_cast<uint64_t>(local_window) + ctx[peer + 2] - ctx[local_rank + 2]);
        using Shape = TileShape2D<BFloat, DYNAMIC, Width>;
        using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        Tile<TileType::Vec, BFloat, PUBLISH_M_TILE, PROJECT_N_TILE, BLayout::RowMajor, DYNAMIC, Width> tile(rows);
        TASSIGN(tile, slot * PUBLISH_SLOT_BYTES + (lane * band + first - begin) * PUBLISH_ROW_BYTES);
        int owner_row = first / (args[9] * args[5]) * args[9] + first % args[9];
        int destination = args[3] * args[6] * args[7] + column * args[6] + owner_row * Width;
        GlobalTensor<BFloat, Shape, Stride> dst(remote + destination, Shape(rows, Width), Stride(1, 1, 1, Width, 1));
        TSTORE(dst, tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        dcci(reinterpret_cast<__gm__ void*>(remote + destination), cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        first = last;
    }
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    set_intra_block(PIPE_MTE3, flag + slot * 2 + 1);
}

__aicore__ inline void vector(__gm__ int64_t* args) {
    int lane = get_sub_block_id(args);
    for (int slot = 0; slot < 2; ++slot) set_intra_block(PIPE_MTE3, lane * 16 + slot * 2 + 1);
    int worker = get_block_idx(args), workers = get_block_num(args);
    int count = 0;
    for (int row = 0; row < args[5] * args[6]; row += args[8]) {
        for (int panel = worker; panel < args[7] / 256; panel += workers) send_panel<256>(args, row, panel * 256, count++);
        if (worker == (args[7] / 256) % workers && args[7] % 256 == 32) send_panel<32>(args, row, args[7] / 256 * 256, count++);
        if (worker == (args[7] / 256) % workers && args[7] % 256 == 64) send_panel<64>(args, row, args[7] / 256 * 256, count++);
        if (worker == (args[7] / 256) % workers && args[7] % 256 == 128) send_panel<128>(args, row, args[7] / 256 * 256, count++);
    }
}
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    native_lifecycle::enter_dual_aiv();
    set_atomic_none();
#if defined(__DAV_CUBE__)
    if (args[7] == 16160) cube<16160>(args);
    else if (args[7] == 32320) cube<32320>(args);
    else if (args[7] == 64640) cube<64640>(args);
    else cube<129280>(args);
#elif defined(__DAV_VEC__)
    vector(args);
#endif
}
