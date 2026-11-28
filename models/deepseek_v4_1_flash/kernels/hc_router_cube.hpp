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
using namespace pto;

namespace hcpre {
// Logical NT GEMM. The same NZ panel can be extracted as a transposed ZN panel.
template <int Rows, int Cols, int Pitch> __aicore__ inline auto panel(__gm__ float *p) {
    using S = Shape<1, 1, 1, Rows, Cols>;
    using D = pto::Stride<1, 1, 1, Pitch, 1>;
    return GlobalTensor<float, S, D>(p);
}
#if defined(__DAV_CUBE__)
template <int M, int L, int Slot, class A, class B>
__aicore__ inline void load_pair(__gm__ float *x, __gm__ float *f, int p) {
    constexpr int MT = M < 16 ? 16 : M, K = 20480;
    int d = p & 1;
    A a;
    B b;
    TASSIGN(a, d * Slot);
    TASSIGN(b, d * Slot + MT * L * 4);
    wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)d);
    auto gx = panel<M, L, K>(x + p * L);
    auto gf = panel<24, L, K>(f + p * L);
    TLOAD(a, gx);
    TLOAD(b, gf);
    set_flag(PIPE_MTE2, PIPE_MTE1, (event_t)d);
}

template <int M, int Splits = 1, bool Wide = false>
__aicore__ inline void project(__gm__ float *x, __gm__ float *f, __gm__ float *out) {
    constexpr int K = 20480, L = Wide ? (M <= 16 ? 1280 : (M <= 64 ? 640 : 512)) : 512,
                  I = Wide ? (M <= 16 ? 256 : (M <= 64 ? 128 : 64)) : (Splits == 1 ? 128 : 256);
    constexpr int MT = M < 16 ? 16 : M;
    constexpr int Slot = (MT + 32) * L * 4;
    using A = Tile<TileType::Mat, float, MT, L, BLayout::ColMajor, M, L, SLayout::RowMajor>;
    using B = Tile<TileType::Mat, float, 32, L, BLayout::ColMajor, 24, L, SLayout::RowMajor>;
    using BT = Tile<TileType::Mat, float, L, 32, BLayout::RowMajor, L, 24, SLayout::ColMajor>;
    using LA = Tile<TileType::Left, float, MT, I, BLayout::ColMajor, M, I, SLayout::RowMajor>;
    using LB = Tile<TileType::Right, float, I, 32, BLayout::RowMajor, I, 24, SLayout::ColMajor>;
    using C = Tile<TileType::Acc, float, MT, 32, BLayout::ColMajor, M, 24, SLayout::RowMajor, 1024>;
    C acc;
    TASSIGN(acc, 0);
    auto ctrl = get_ctrl();
    set_ctrl(sbitset1(sbitset1(ctrl, 46), 47));
    for (int j = 0; j < 2; j++) {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)j);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)j);
    }
    constexpr int Steps = K / L / Splits;
    static_assert(K % (L * Splits) == 0 && Slot * 2 <= 524288);
    if constexpr (Wide)
        load_pair<M, L, Slot, A, B>(x, f, 0);
    for (int p = 0; p < Steps; p++) {
        int d = p & 1;
        A a;
        BT bt;
        TASSIGN(a, d * Slot);
        TASSIGN(bt, d * Slot + MT * L * 4);
        if constexpr (Wide) {
            if (p + 1 < Steps)
                load_pair<M, L, Slot, A, B>(x, f, p + 1);
        } else
            load_pair<M, L, Slot, A, B>(x, f, p);
        wait_flag(PIPE_MTE2, PIPE_MTE1, (event_t)d);
        for (int k = 0; k < L / I; k++) {
            int s = k & 1;
            LA la;
            LB lb;
            TASSIGN(la, s * 32768);
            TASSIGN(lb, s * 32768);
            wait_flag(PIPE_M, PIPE_MTE1, (event_t)s);
            TEXTRACT(la, a, 0, k * I);
            TEXTRACT(lb, bt, k * I, 0);
            set_flag(PIPE_MTE1, PIPE_M, (event_t)s);
            wait_flag(PIPE_MTE1, PIPE_M, (event_t)s);
            if (p == 0 && k == 0)
                TMATMUL(acc, la, lb);
            else
                TMATMUL_ACC(acc, acc, la, lb);
            set_flag(PIPE_M, PIPE_MTE1, (event_t)s);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)d);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    auto go = panel<M, 24, 32>(out);
    TSTORE(go, acc);
    for (int s = 0; s < 2; s++) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)s);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)s);
    }
    pipe_barrier(PIPE_ALL);
    set_ctrl(ctrl);
}
template <int M, int E, int L, class A, class B>
__aicore__ inline void gate_load(__gm__ bfloat16_t *x, __gm__ bfloat16_t *w, int p) {
    constexpr int MT = (M + 15) / 16 * 16, N = (E + 15) / 16 * 16;
    int s = p & 1, base = s * (MT + N) * L * 2;
    A a;
    B b;
    TASSIGN(a, base);
    TASSIGN(b, base + MT * L * 2);
    GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, L>, pto::Stride<1, 1, 1, 5120, 1>> gx(x + p * L);
    GlobalTensor<bfloat16_t, Shape<1, 1, 1, E, L>, pto::Stride<1, 1, 1, 5120, 1>> gw(w + p * L);
    wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)s);
    TLOAD(a, gx);
    TLOAD(b, gw);
    set_flag(PIPE_MTE2, PIPE_MTE1, (event_t)s);
}
// Gate weights are BF16 NT, unlike the preceding HF32 HC projection.
template <int M, int E, int Pitch = E>
__aicore__ inline void gate_mm(__gm__ bfloat16_t *x, __gm__ bfloat16_t *w, __gm__ float *logits) {
    constexpr int N = (E + 15) / 16 * 16, MT = (M + 15) / 16 * 16;
    constexpr int L = E <= 48 && MT <= 64 && MT + N <= 96 ? 1280 : (E <= 96 ? 640 : 512),
                  I = L == 1280 ? (MT <= 48 && N <= 48 ? 320 : 256) : (E <= 96 ? 128 : 64);
    static_assert(2 * (MT + N) * L * 2 <= 524288 && N * I * 2 <= 32768 && MT * I * 2 <= 32768);
    using A = Tile<TileType::Mat, bfloat16_t, MT, L, BLayout::ColMajor, M, L, SLayout::RowMajor>;
    using B = Tile<TileType::Mat, bfloat16_t, N, L, BLayout::ColMajor, E, L, SLayout::RowMajor>;
    using BT = Tile<TileType::Mat, bfloat16_t, L, N, BLayout::RowMajor, L, E, SLayout::ColMajor>;
    using LA = Tile<TileType::Left, bfloat16_t, MT, I, BLayout::ColMajor, M, I, SLayout::RowMajor>;
    using LB = Tile<TileType::Right, bfloat16_t, I, N, BLayout::RowMajor, I, E, SLayout::ColMajor>;
    using C = Tile<TileType::Acc, float, MT, N, BLayout::ColMajor, M, E, SLayout::RowMajor, 1024>;
    C c;
    TASSIGN(c, 0);
    for (int j = 0; j < 2; ++j) {
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)j);
        set_flag(PIPE_M, PIPE_MTE1, (event_t)j);
    }
    gate_load<M, E, L, A, B>(x, w, 0);
    for (int p = 0; p < 5120 / L; ++p) {
        int s = p & 1, base = s * (MT + N) * L * 2;
        A a;
        BT bt;
        TASSIGN(a, base);
        TASSIGN(bt, base + MT * L * 2);
        if (p + 1 < 5120 / L)
            gate_load<M, E, L, A, B>(x, w, p + 1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, (event_t)s);
        for (int k = 0; k < L / I; k++) {
            int d = k & 1;
            LA la;
            LB lb;
            TASSIGN(la, d * 32768);
            TASSIGN(lb, d * 32768);
            wait_flag(PIPE_M, PIPE_MTE1, (event_t)d);
            TEXTRACT(la, a, 0, k * I);
            TEXTRACT(lb, bt, k * I, 0);
            set_flag(PIPE_MTE1, PIPE_M, (event_t)d);
            wait_flag(PIPE_MTE1, PIPE_M, (event_t)d);
            if (!p && !k)
                TMATMUL(c, la, lb);
            else
                TMATMUL_ACC(c, c, la, lb);
            set_flag(PIPE_M, PIPE_MTE1, (event_t)d);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)s);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    auto dst = panel<M, E, Pitch>(logits);
    TSTORE(dst, c);
    for (int j = 0; j < 2; ++j) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)j);
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)j);
    }
    pipe_barrier(PIPE_ALL);
}

#endif
} // namespace hcpre
