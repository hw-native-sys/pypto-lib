// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include "tensor.h"
#include "intrinsic.h"
#include "hc_router_cube.hpp"
#include "hc_router_vector.hpp"
#include "mixed_entry.hpp"

namespace hcpre {
template <class T> __aicore__ inline __gm__ T *ga(__gm__ int64_t *a, int n) {
    auto t = reinterpret_cast<__gm__ Tensor *>(a[n]);
    return (__gm__ T *)t->buffer.addr + t->start_offset;
}
__aicore__ inline int dim(__gm__ int64_t *a, int t) {
    return reinterpret_cast<__gm__ Tensor *>(a[t])->shapes[0];
}
__aicore__ inline void publish(__gm__ int *p) {
    *reinterpret_cast<volatile __gm__ int *>(p) = 1;
    dcci(p, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
}
__aicore__ inline void await_flag(__gm__ int *p) {
    do {
        dcci(p, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    } while (*reinterpret_cast<volatile __gm__ int *>(p) != 1);
}
#if defined(__DAV_CUBE__)
// Bounded expert panels keep L1/L0 use independent of total gate width.
template <int R, int E, int Begin, int End> __aicore__ inline void columns(__gm__ int64_t *a, int t) {
    if constexpr (Begin < End) {
        constexpr int n = End - Begin < 48 ? End - Begin : 48;
        gate_mm<R, n, E>(ga<bfloat16_t>(a, 7) + t * 5120, ga<bfloat16_t>(a, 11) + Begin * 5120,
                         ga<float>(a, 14) + t * E + Begin);
        columns<R, E, Begin + n, End>(a, t);
    }
}
template <int M, int G, int E, int J = 0>
__aicore__ inline void gate_part(__gm__ int64_t *a, int group_gate, int t) {
    if constexpr (J < G) {
        if (group_gate == J) {
            constexpr int step = ((E + G - 1) / G + 7) / 8 * 8;
            constexpr int first = J * step, last = (J + 1) * step < E ? (J + 1) * step : E;
            columns<M, E, first, last>(a, t);
        } else
            gate_part<M, G, E, J + 1>(a, group_gate, t);
    }
}
#endif
template <int M, int G, int E>
__aicore__ inline void role_tile(__gm__ int64_t *a, int rows, int active, int role, int group, int start) {
    constexpr int S = 4, slots = 12 + G;
    constexpr int vr = M / (S * 2);
    static_assert(G == 4 && E == 384);
    int bias_rows = dim(a, 12) / E;
    auto proj = ga<float>(a, 13);
    auto flags = (__gm__ int *)(proj + S * rows * 32) + group * slots * 32;
#if defined(__DAV_CUBE__)
    if (role < S) {
        int k = role * (20480 / S);
        project<M, S, true>(ga<float>(a, 0) + start * 20480 + k, ga<float>(a, 2) + k,
                            proj + (role * rows + start) * 32);
        publish(flags + role * 32);
    }
    {
        for (int k = 0; k < S; k++)
            await_flag(flags + k * 32);
        if (start < active) {
            for (int v = 0; v < 2 * S; ++v)
                await_flag(flags + (S + v) * 32);
            gate_part<M, G, E>(a, role, start);
        }
        publish(flags + (S + 2 * S + role) * 32);
    }
    set_intra_block(PIPE_FIX, 0);
    set_intra_block(PIPE_FIX, 16);
    wait_intra_block(PIPE_FIX, 1);
    wait_intra_block(PIPE_FIX, 17);
#elif defined(__DAV_VEC__)
    if (role < S) {
        int first = start + (role * 2 + get_sub_block_id(a)) * vr;
        auto inv = (__ubuf__ float *)245760;
        stream_rows<vr>(ga<float>(a, 0) + first * 20480, ga<float>(a, 1) + first * 4, ga<bfloat16_t>(a, 5),
                        ga<bfloat16_t>(a, 6) + first * 5120, ga<bfloat16_t>(a, 7) + first * 5120, inv,
                        active - first);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID6);
        publish(flags + (S + role * 2 + get_sub_block_id(a)) * 32);
        for (int k = 0; k < S; k++)
            await_flag(flags + k * 32);
        constexpr int batch = vr < 4 ? vr : 4;
        for (int j = 0; j + batch <= vr; j += batch)
            coeff_batch<batch, S>(proj + (first + j) * 32, ga<float>(a, 3), ga<float>(a, 4),
                                  ga<float>(a, 8) + (first + j) * 4, ga<float>(a, 9) + (first + j) * 4,
                                  ga<float>(a, 10) + (first + j) * 16, inv + j * 8, rows * 32);
        if constexpr (vr % batch == 2) {
            constexpr int j = vr - 2;
            coeff_batch<2, S>(proj + (first + j) * 32, ga<float>(a, 3), ga<float>(a, 4),
                              ga<float>(a, 8) + (first + j) * 4, ga<float>(a, 9) + (first + j) * 4,
                              ga<float>(a, 10) + (first + j) * 16, inv + j * 8, rows * 32);
        }
    }
    // Each of four gate clusters has two AIVs. All column panels are needed for global top-6.
    if (role < S) {
        constexpr int rr = M / (2 * G);
        int g = role;
        int first = start + (g * 2 + get_sub_block_id(a)) * rr;
        for (int j = 0; j < G; ++j)
            await_flag(flags + (3 * S + j) * 32);
        for (int row = first; row < first + rr; ++row)
            top<E>(ga<float>(a, 14) + row * E, ga<float>(a, 12) + (bias_rows == 1 ? 0 : row) * E,
                   ga<int>(a, 15) + row * 6, ga<float>(a, 16) + row * 6, row < active);
    }
    wait_intra_block(PIPE_MTE2, 0);
    set_intra_block(PIPE_MTE3, 1);
#endif
}
__aicore__ inline void role_group(__gm__ int64_t *a) {
    constexpr int team = 4, experts = 384;
    int rows = dim(a, 0), active = a[17], worker = get_block_idx(a), workers = get_block_num(a);
    if (rows < 160 || rows > 1024 || rows % 16 || workers < team || workers % team ||
        active < 0 || active > rows || dim(a, 11) != experts) {
        trap();
        return;
    }
    set_atomic_none();
    native_lifecycle::enter_dual_aiv();
    const int groups = rows <= 512 ? 8 : 16;
    const int tiles = rows / 16, base = tiles / groups, extra = tiles % groups;
    for (int group = worker / team; group < groups; group += workers / team) {
        const int start = (group * base + (group < extra ? group : extra)) * 16;
        const int width = base + (group < extra);
        switch (width) {
        case 1: role_tile<16, team, experts>(a, rows, active, worker % team, group, start); break;
        case 2: role_tile<32, team, experts>(a, rows, active, worker % team, group, start); break;
        case 3: role_tile<48, team, experts>(a, rows, active, worker % team, group, start); break;
        case 4: role_tile<64, team, experts>(a, rows, active, worker % team, group, start); break;
        default: trap(); return;
        }
    }
    pipe_barrier(PIPE_ALL);
}
} // namespace hcpre
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *a) {
    hcpre::role_group(a);
}
