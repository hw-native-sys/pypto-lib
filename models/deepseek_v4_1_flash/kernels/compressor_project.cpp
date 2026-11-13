// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Compensated BF16 compressor projection with buffered Cube operands and Vector registers.
#include "bf16_projection.hpp"
using namespace pto;
namespace {
constexpr int M_TILE = 80;
constexpr int K = 5120;
constexpr int N = 512;
template <int N_TILE> using Pipe = TPipe<0, Direction::DIR_C2V, M_TILE * N_TILE * sizeof(float), 2, 2, true>;
template <int N_TILE> using Vec = Tile<TileType::Vec, float, M_TILE, N_TILE>;
using bf16_projection::data;
#if defined(__DAV_VEC__)
template <int N_TILE>
__tf__ __aicore__ inline void accumulate_vf(
    typename Vec<N_TILE>::TileDType __out__ sum_out,
    typename Vec<N_TILE>::TileDType __out__ error_out,
    typename Vec<N_TILE>::TileDType __in__ sum_in,
    typename Vec<N_TILE>::TileDType __in__ error_in,
    typename Vec<N_TILE>::TileDType __in__ partial) {
    auto sum = (__ubuf__ float*)__cce_get_tile_ptr(sum_in);
    auto error = (__ubuf__ float*)__cce_get_tile_ptr(error_in);
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(partial);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(sum_out);
    auto correction = (__ubuf__ float*)__cce_get_tile_ptr(error_out);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        for (uint16_t segment = 0; segment < static_cast<uint16_t>(M_TILE * N_TILE / 64); ++segment) {
            RegTensor<float> previous, compensation, value, delta, updated, roundoff, corrected;
            vlds(previous, sum, segment * 64, NORM);
            vlds(compensation, error, segment * 64, NORM);
            vlds(value, src, segment * 64, NORM);
            vsub(delta, value, compensation, all, MODE_ZEROING);
            vadd(updated, previous, delta, all, MODE_ZEROING);
            vsub(roundoff, updated, previous, all, MODE_ZEROING);
            vsub(corrected, roundoff, delta, all, MODE_ZEROING);
            vsts(updated, dst, segment * 64, NORM_B32, all);
            vsts(corrected, correction, segment * 64, NORM_B32, all);
        }
    }
}
template <int N_TILE>
__aicore__ inline void vector_project(__gm__ int64_t* args) {
    Pipe<N_TILE> pipe(nullptr, 0, 0);
    int tokens = args[5];
    int tasks = (tokens + M_TILE - 1) / M_TILE * (2 * N / N_TILE);
    Vec<N_TILE> sum, error;
    TASSIGN(sum, 2 * M_TILE * N_TILE * sizeof(float));
    TASSIGN(error, 3 * M_TILE * N_TILE * sizeof(float));
    for (int work = get_block_idx(args); work < tasks; work += get_block_num(args)) {
        int row = work / (2 * N / N_TILE) * M_TILE;
        int col = work % (N / N_TILE) * N_TILE;
        int rows = tokens - row < M_TILE ? tokens - row : M_TILE;
        TEXPANDS(sum, 0.0f);
        TEXPANDS(error, 0.0f);
        for (int step = 0; step < K / 256; ++step) {
            Vec<N_TILE> partial;
            TPOP<Pipe<N_TILE>, Vec<N_TILE>, TileSplitAxis::TILE_NO_SPLIT>(pipe, partial);
            accumulate_vf<N_TILE>(sum.data(), error.data(), sum.data(), error.data(), partial.data());
            pipe_barrier(PIPE_ALL);
            TFREE<Pipe<N_TILE>, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
        using Output = Tile<TileType::Vec, float, M_TILE, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
        Output valid(rows);
        TASSIGN(valid, 2 * M_TILE * N_TILE * sizeof(float));
        using Shape = TileShape2D<float, DYNAMIC, N_TILE>;
        using Base = BaseShape2D<float, DYNAMIC, N>;
        int output_index = work % (2 * N / N_TILE) < N / N_TILE ? 3 : 4;
        GlobalTensor<float, Shape, Base> gm(data<float>(args, output_index) + row * N + col, Shape(rows, N_TILE), Base(tokens, N));
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(gm, valid);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__DAV_CUBE__)
template <int N_TILE>
__aicore__ inline void cube_project(__gm__ int64_t* args) {
    using Panel = bf16_projection::PanelPipeline<K, N, N_TILE, 256, 128, 2, Layout::NZ, M_TILE>;
    Pipe<N_TILE> pipe(nullptr, 0, 0);
    int tokens = args[5];
    int tasks = (tokens + M_TILE - 1) / M_TILE * (2 * N / N_TILE);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
    for (int slot = 0; slot < 2; ++slot) set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    for (int work = get_block_idx(args); work < tasks; work += get_block_num(args)) {
        int row = work / (2 * N / N_TILE) * M_TILE;
        int col = work % (N / N_TILE) * N_TILE;
        int rows = tokens - row < M_TILE ? tokens - row : M_TILE;
        auto x = data<bfloat16_t>(args, 0) + row * K;
        auto weight = data<bfloat16_t>(args, work % (2 * N / N_TILE) < N / N_TILE ? 1 : 2);
        typename Panel::AC acc(M_TILE);
        TASSIGN(acc, 0);
        for (int slot = 0; slot < 2; ++slot) Panel::load(x, weight, rows, slot, 0, 1, slot, K, N, col);
        for (int step = 0; step < K / 256; ++step) {
            int slot = step % 2;
            typename Panel::AM am(M_TILE);
            typename Panel::BM bm;
            TASSIGN(am, slot * Panel::SlotBytes);
            TASSIGN(bm, slot * Panel::SlotBytes + M_TILE * 256 * sizeof(bfloat16_t));
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
            for (int inner = 0; inner < 2; ++inner) {
                auto event = static_cast<event_t>(inner);
                typename Panel::AL left(M_TILE);
                typename Panel::BR right;
                TASSIGN(left, inner * 32768);
                TASSIGN(right, inner * 32768);
                wait_flag(PIPE_M, PIPE_MTE1, event);
                TEXTRACT(left, am, 0, inner * 128);
                TEXTRACT(right, bm, inner * 128, 0);
                set_flag(PIPE_MTE1, PIPE_M, event);
                wait_flag(PIPE_MTE1, PIPE_M, event);
                if (inner == 0) {
                    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
                    TMATMUL(acc, left, right);
                } else {
                    TMATMUL_ACC(acc, acc, left, right);
                }
                set_flag(PIPE_M, PIPE_MTE1, event);
            }
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
            TPUSH<Pipe<N_TILE>, typename Panel::AC, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
            if (step + 2 < K / 256) {
                set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                Panel::load(x, weight, rows, step + 2, 0, 1, slot, K, N, col);
            }
        }
        pipe_barrier(PIPE_ALL);
    }
    for (int slot = 0; slot < 2; ++slot) wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
    pipe_barrier(PIPE_ALL);
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    set_atomic_none();
#if defined(__DAV_CUBE__)
    if (args[6] == 128) cube_project<128>(args);
    else cube_project<64>(args);
#elif defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    if (args[6] == 128) vector_project<128>(args);
    else vector_project<64>(args);
#endif
}
