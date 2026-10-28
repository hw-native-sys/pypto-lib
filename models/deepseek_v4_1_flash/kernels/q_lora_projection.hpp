// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Q-latent projections with resident A/scales and buffered K/N transfers.
#pragma once
#include "mxfp8_projection.hpp"
using namespace mxfp8_projection;
using MatB = mxfp8_projection::BM<256>;
using Acc = mxfp8_projection::AC<256>;
using ScaleB = BS<1280, 256>;
using FullA = Tile<TileType::Mat, F, 80, 1280, BLayout::ColMajor, 80, 1280, SLayout::RowMajor, 512>;

template <int N, int WORKERS>
__aicore__ inline void load_b(__gm__ F* b, int step, int block, int slot) {
    int n = (step / 5 * WORKERS + block) * 256;
    int k = step % 5 * 256;
    MatB tile;
    TASSIGN(tile, 196608 + slot * 65536);
    int offset = n * 1280 + k * 32;
    GM<F, 256, 256, 1280, N, Layout::NZ> gm(b + offset);
    TLOAD(tile, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int INNER>
__aicore__ inline void compute_qb(FullA& am, MatB& bm, AS<1280>& sa, ScaleB& sb, Acc& acc, int k, int wave) {
    constexpr int slot = INNER;
    wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    TileLeft<F, 80, 128, 80, 128> a;
    TileRight<F, 128, 256, 128, 256> b;
    TileLeftScale<S, 80, 4, 80, 4> as;
    TileRightScale<S, 4, 256, 4, 256> bs;
    TASSIGN(a, slot * 32768);
    TASSIGN(b, slot * 32768);
    TASSIGN(as, GetScaleAddr(a.data()));
    TASSIGN(bs, GetScaleAddr(b.data()));
    TEXTRACT(a, am, 0, k + INNER * 128);
    TEXTRACT(b, bm, INNER * 128, 0);
    TEXTRACT(as, sa, 0, k / 32 + INNER * 4);
    TEXTRACT(bs, sb, k / 32 + INNER * 4, 0);
    set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(slot));
    if (k == 0 && INNER == 0) {
        wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + wave % 2));
        TMATMUL_MX(acc, a, as, b, bs);
    } else {
        TMATMUL_MX(acc, acc, a, as, b, bs);
    }
    set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int N, int WORKERS = (N / 256 < 32 ? N / 256 : 32), bool OFFSET_INPUT = false>
__aicore__ inline void project_qb(__gm__ int64_t* args) {
    constexpr int WAVES = N / (256 * WORKERS);
    int block = get_block_idx(args);
    int input_row = 0;
    if constexpr (OFFSET_INPUT) input_row = args[8];
    FullA am;
    AS<1280> sa;
    TASSIGN(am, 0);
    TASSIGN(sa, 102400);
    GM<F, 80, 1280, 80, 1280, Layout::ND> ag(data<F>(args, 0) + input_row * 1280);
    GM<S, 80, 40, 80, 40, Layout::MX_A_ZZ> sag(data<S>(args, 2) + input_row * 40);
    TLOAD(am, ag);
    TLOAD(sa, sag);
    for (int wave = 0; wave < WAVES; ++wave) {
        ScaleB sb;
        TASSIGN(sb, 105600 + wave * 10240);
        GM<S, 40, 256, 40, N, Layout::MX_B_NN> sbg(data<S>(args, 3) + (wave * WORKERS + block) * 256 * 40);
        TLOAD(sb, sbg);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    for (int slot = 0; slot < 2; ++slot) {
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
    }
    for (int slot = 0; slot < 4; ++slot) {
        load_b<N, WORKERS>(data<F>(args, 1), slot, block, slot);
    }
    for (int step = 0; step < 5 * WAVES; ++step) {
        int slot = step % 4;
        int wave = step / 5;
        int k = step % 5 * 256;
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
        MatB bm;
        ScaleB sb;
        Acc acc;
        TASSIGN(bm, 196608 + slot * 65536);
        TASSIGN(sb, 105600 + wave * 10240);
        TASSIGN(acc, wave % 2 * 81920);
        acc.SetValidRow(80);
        compute_qb<0>(am, bm, sa, sb, acc, k, wave);
        compute_qb<1>(am, bm, sa, sb, acc, k, wave);
        if (step + 4 < 5 * WAVES) {
            set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            load_b<N, WORKERS>(data<F>(args, 1), step + 4, block, slot);
        }
        if (step % 5 == 4) {
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            acc.SetValidRow(args[5]);
            using OS = TileShape2D<bfloat16_t, DYNAMIC, 256, Layout::ND>;
            using ST = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC, Layout::ND>;
            GlobalTensor<bfloat16_t, OS, ST, Layout::ND> cg(
                data<bfloat16_t>(args, 4) + args[7] * N + (wave * WORKERS + block) * 256,
                OS(args[5], 256), ST(args[5], N));
            TSTORE(cg, acc);
            set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + wave % 2));
        }
    }
    for (int slot = 0; slot < 2; ++slot) {
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
    }
    pipe_barrier(PIPE_ALL);
}
