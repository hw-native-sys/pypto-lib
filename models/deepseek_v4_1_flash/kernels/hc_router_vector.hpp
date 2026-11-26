// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once
#include "rmsnorm.hpp"
namespace hcpre {
#if defined(__DAV_VEC__)
template <class T, int N, int Width = N>
using Row = Tile<TileType::Vec, T, 1, Width, BLayout::RowMajor, 1, N>;
template <class T, int N> __aicore__ inline auto gmrow(__gm__ T *p) {
    return GlobalTensor<T, Shape<1, 1, 1, 1, N>, pto::Stride<1, 1, 1, N, 1>>(p);
}
template <class T, int N, int W> __aicore__ inline void load_row(__gm__ T *p, Row<T, N, W> &t) {
    auto g = gmrow<T, N>(p);
    TLOAD(t, g);
}
template <class T, int N, int W> __aicore__ inline void store_row(__gm__ T *p, Row<T, N, W> &t) {
    auto g = gmrow<T, N>(p);
    TSTORE(g, t);
}
__aicore__ inline void zero_norm(__ubuf__ bfloat16_t *out) {
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<float> z;
        RegTensor<bfloat16_t> bf;
        vdup(z, 0.f, all, MODE_ZEROING);
        vcvt(bf, z, all, ROUND_R, RS_DISABLE, PART_EVEN);
        for (uint16_t i = 0; i < 5120; i += 64)
            vsts(bf, out, i, PK_B32, all);
    }
}
using F = RegTensor<float>;
using U = RegTensor<uint32_t>;
__aicore__ inline void fold(F &v, U &p, U &q, MaskReg &m, bool maximum) {
    F n;
    vselr((U &)n, (U &)v, p);
    if (maximum)
        vmax(v, v, n, m, MODE_ZEROING);
    else
        vadd(v, v, n, m, MODE_ZEROING);
    vselr((U &)n, (U &)v, q);
    if (maximum)
        vmax(v, v, n, m, MODE_ZEROING);
    else
        vadd(v, v, n, m, MODE_ZEROING);
}
__aicore__ inline void renorm(F &v, U &p, U &q, MaskReg &m, float eps) {
    F den = v;
    fold(den, p, q, m, false);
    vadds(den, den, eps, m, MODE_ZEROING);
    vdiv(v, v, den, m, MODE_ZEROING);
}
template <bool Normalize>
__aicore__ inline void mix_registers(__ubuf__ float *xp, __ubuf__ float *mp, __ubuf__ bfloat16_t *gp,
                                     __ubuf__ bfloat16_t *yp, __ubuf__ bfloat16_t *zp, __ubuf__ float *inv) {
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        {
            F a, b, c, d, m0, m1, m2, m3, h0, h1, h2, h3, ys, t, r, yr;
            vlds(m0, mp, 0, BRC_B32);
            vlds(m1, mp, 1, BRC_B32);
            vlds(m2, mp, 2, BRC_B32);
            vlds(m3, mp, 3, BRC_B32);
            vdup(h0, 0.f, all, MODE_ZEROING);
            h1 = h0;
            h2 = h0;
            h3 = h0;
            ys = h0;
            for (uint16_t i = 0; i < 5120; i += 64) {
                vlds(a, xp, i, NORM);
                vlds(b, xp, 5120 + i, NORM);
                vlds(c, xp, 10240 + i, NORM);
                vlds(d, xp, 15360 + i, NORM);
                vmula(h0, a, a, all, MODE_ZEROING);
                vmula(h1, b, b, all, MODE_ZEROING);
                vmula(h2, c, c, all, MODE_ZEROING);
                vmula(h3, d, d, all, MODE_ZEROING);
                vmul(a, a, m0, all, MODE_ZEROING);
                vmul(b, b, m1, all, MODE_ZEROING);
                vmul(c, c, m2, all, MODE_ZEROING);
                vmul(d, d, m3, all, MODE_ZEROING);
                vadd(a, a, b, all, MODE_ZEROING);
                vadd(a, a, c, all, MODE_ZEROING);
                vadd(a, a, d, all, MODE_ZEROING);
                RegTensor<bfloat16_t> bf;
                vcvt(bf, a, all, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(bf, yp, i, PK_B32, all);
                if constexpr (Normalize) {
                    vcvt(a, bf, all, PART_EVEN);
                    vmula(ys, a, a, all, MODE_ZEROING);
                }
            }
            vadd(h0, h0, h1, all, MODE_ZEROING);
            vadd(h2, h2, h3, all, MODE_ZEROING);
            vadd(h0, h0, h2, all, MODE_ZEROING);
            vcadd(t, h0, all, MODE_ZEROING);
            vmuls(t, t, 1.f / 20480, all, MODE_ZEROING);
            recipe_rms::inverse_root(r, t, all);
            vsts(r, inv, 0, ONEPT_B32, all);
            if constexpr (Normalize) {
                vcadd(t, ys, all, MODE_ZEROING);
                vmuls(t, t, 1.f / 5120, all, MODE_ZEROING);
                recipe_rms::inverse_root(yr, t, all);
                vdup(yr, yr, all, POS_LOWEST, MODE_ZEROING);
                // vec store -> load dependency within this VF.
                mem_bar(VST_VLD);
                for (uint16_t i = 0; i < 5120; i += 64) {
                    recipe_rms::load(a, yp, i, all);
                    recipe_rms::load(b, gp, i, all);
                    vmul(a, a, yr, all, MODE_ZEROING);
                    vmul(a, a, b, all, MODE_ZEROING);
                    RegTensor<bfloat16_t> bf;
                    vcvt(bf, a, all, ROUND_R, RS_DISABLE, PART_EVEN);
                    vsts(bf, zp, i, PK_B32, all);
                }
            }
        }
    }
}
__aicore__ inline void load_stream_row(__gm__ float *x, __gm__ float *mix, int row) {
    int slot = row & 1;
    Row<float, 20480> src;
    Row<float, 4, 8> pre;
    TASSIGN(src, slot * 81920);
    TASSIGN(pre, 174080 + slot * 256);
    wait_flag(PIPE_V, PIPE_MTE2, (event_t)slot);
    load_row<float, 20480, 20480>(x + row * 20480, src);
    load_row<float, 4, 8>(mix + row * 4, pre);
    set_flag(PIPE_MTE2, PIPE_V, (event_t)slot);
}
template <int Count>
__aicore__ inline void stream_rows(__gm__ float *x, __gm__ float *mix, __gm__ bfloat16_t *gamma,
                                   __gm__ bfloat16_t *raw, __gm__ bfloat16_t *norm, __ubuf__ float *inv,
                                   int active) {
    Row<bfloat16_t, 5120> g;
    TASSIGN(g, 163840);
    if (active > 0)
        load_row<bfloat16_t, 5120, 5120>(gamma, g);
    for (int s = 0; s < 2; s++) {
        set_flag(PIPE_V, PIPE_MTE2, (event_t)s);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)s);
    }
    load_stream_row(x, mix, 0);
    for (int row = 0; row < Count; row++) {
        int s = row & 1;
        if (row + 1 < Count)
            load_stream_row(x, mix, row + 1);
        Row<float, 20480> src;
        Row<float, 4, 8> pre;
        Row<bfloat16_t, 5120> y, z;
        TASSIGN(src, s * 81920);
        TASSIGN(pre, 174080 + s * 256);
        TASSIGN(y, 174592 + s * 20480);
        TASSIGN(z, 184832 + s * 20480);
        wait_flag(PIPE_MTE2, PIPE_V, (event_t)s);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)s);
        if (row < active) {
            mix_registers<true>(src.data(), pre.data(), g.data(), y.data(), z.data(), inv + row * 8);
        } else {
            mix_registers<false>(src.data(), pre.data(), g.data(), y.data(), z.data(), inv + row * 8);
            zero_norm(z.data());
        }
        set_flag(PIPE_V, PIPE_MTE2, (event_t)s);
        set_flag(PIPE_V, PIPE_MTE3, (event_t)s);
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)s);
        store_row<bfloat16_t, 5120, 5120>(raw + row * 5120, y);
        store_row<bfloat16_t, 5120, 5120>(norm + row * 5120, z);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)s);
    }
    for (int s = 0; s < 2; s++) {
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)s);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)s);
    }
    pipe_barrier(PIPE_ALL);
}
template <int N, int S>
__aicore__ inline void coeff_batch(__gm__ float *projection, __gm__ float *scales, __gm__ float *bias,
                                   __gm__ float *pre, __gm__ float *post, __gm__ float *res,
                                   __ubuf__ float *inv, int pitch) {
    static_assert(N == 2 || N == 4);
    Row<float, 24, 64> bi;
    Row<float, 3, 8> sc;
    Row<float, N * 4, 64> y, z;
    Row<float, N * 16, 64> m;
    TASSIGN(bi, 4096);
    TASSIGN(sc, 4352);
    TASSIGN(y, 4608);
    TASSIGN(z, 4864);
    TASSIGN(m, 5120);
    load_row<float, 24, 64>(bias, bi);
    load_row<float, 3, 8>(scales, sc);
    constexpr int PlaneStride = 128;
    for (int s = 0; s < S; s++) {
        Tile<TileType::Vec, float, N, 32, BLayout::RowMajor, N, 24> v;
        TASSIGN(v, s * 512);
        auto src = panel<N, 24, 32>(projection + s * pitch);
        TLOAD(v, src);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    auto pp = (__ubuf__ float *)0;
    auto bp = bi.data();
    auto sp = sc.data();
    auto yp = y.data();
    auto zp = z.data();
    auto mp = m.data();
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        U lane, token, col, idx, offset, iv, p1, p2, p4, p8, k;
        F v, part, b, r, scale, one, t;
        vci((RegTensor<int32_t> &)lane, 0, INC_ORDER);
        vdup(one, 1.f, all, MODE_ZEROING);
        for (uint16_t g = 0; g < 3; g++) {
            uint32_t count = g < 2 ? N * 4 : N * 16;
            MaskReg mask = plt_b32(count, POST_UPDATE);
            if (g < 2) {
                vshrs(token, lane, 2, all, MODE_ZEROING);
                vdup(k, 3u, all, MODE_ZEROING);
            } else {
                vshrs(token, lane, 4, all, MODE_ZEROING);
                vdup(k, 15u, all, MODE_ZEROING);
            }
            vand(col, lane, k, all);
            vadds(col, col, uint32_t(g * 4), all, MODE_ZEROING);
            vshls(offset, token, 5, all, MODE_ZEROING);
            vadd(idx, col, offset, all, MODE_ZEROING);
            vshls(iv, token, 3, all, MODE_ZEROING);
            vgather2(r, inv, iv, mask);
            vgather2(v, pp, idx, mask);
            for (uint16_t s = 1; s < S; s++) {
                vgather2(part, pp + s * PlaneStride, idx, mask);
                vadd(v, v, part, mask, MODE_ZEROING);
            }
            vgather2(b, bp, col, mask);
            vlds(scale, sp, g, BRC_B32);
            vmul(v, v, r, mask, MODE_ZEROING);
            vmul(v, v, scale, mask, MODE_ZEROING);
            vadd(v, v, b, mask, MODE_ZEROING);
            if (g < 2) {
                vmuls(v, v, -1.f, mask, MODE_ZEROING);
                vexp(v, v, mask, MODE_ZEROING);
                vadds(v, v, 1.f, mask, MODE_ZEROING);
                vdiv(v, one, v, mask, MODE_ZEROING);
                if (g == 0) {
                    vadds(v, v, 1e-6f, mask, MODE_ZEROING);
                    vsts(v, yp, 0, NORM_B32, mask);
                } else {
                    vmuls(v, v, 2.f, mask, MODE_ZEROING);
                    vsts(v, zp, 0, NORM_B32, mask);
                }
            } else {
                vdup(k, 1u, all, MODE_ZEROING);
                vxor(p1, lane, k, all);
                vdup(k, 2u, all, MODE_ZEROING);
                vxor(p2, lane, k, all);
                vdup(k, 4u, all, MODE_ZEROING);
                vxor(p4, lane, k, all);
                vdup(k, 8u, all, MODE_ZEROING);
                vxor(p8, lane, k, all);
                t = v;
                fold(t, p1, p2, all, true);
                vsub(v, v, t, mask, MODE_ZEROING);
                vexp(v, v, mask, MODE_ZEROING);
                renorm(v, p1, p2, mask, 0.f);
                vadds(v, v, 1e-6f, mask, MODE_ZEROING);
                renorm(v, p4, p8, mask, 1e-6f);
                for (uint16_t j = 1; j < 20; j++) {
                    renorm(v, p1, p2, mask, 1e-6f);
                    renorm(v, p4, p8, mask, 1e-6f);
                }
                vsts(v, mp, 0, NORM_B32, mask);
            }
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    store_row<float, N * 4, 64>(pre, y);
    store_row<float, N * 4, 64>(post, z);
    store_row<float, N * 16, 64>(res, m);
    pipe_barrier(PIPE_ALL);
}
template <int E>
__aicore__ inline void top(__gm__ float *logits, __gm__ float *bias, __gm__ int *indices,
                           __gm__ float *weights, bool active) {
    constexpr int N = (E + 63) / 64 * 64;
    Row<float, E, N> l, b;
    Row<float, N> scores, biased;
    Row<int, 6, 8> ids;
    Row<float, 6, 8> ws;
    TASSIGN(l, 0);
    TASSIGN(b, 2048);
    TASSIGN(scores, 4096);
    TASSIGN(biased, 6144);
    TASSIGN(ids, 8192);
    TASSIGN(ws, 8448);
    auto ip = ids.data();
    auto wp = ws.data();
    auto sp = scores.data();
    auto bp = biased.data();
    if (active) {
        load_row<float, E, N>(logits, l);
        load_row<float, E, N>(bias, b);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        auto xp = l.data();
        auto cb = b.data();
        Row<unsigned, N> keys;
        Row<float, 2 * N> pairs;
        TASSIGN(keys, 9216);
        TASSIGN(pairs, 11264);
        auto kp = keys.data();
        auto pair = pairs.data();
        __VEC_SCOPE__ {
            MaskReg all = pset_b32(PAT_ALL);
            for (uint16_t i = 0; i < N; i += 64) {
                F x, t, rel, floor, mask, bv;
                vlds(x, xp, i, NORM);
                vlds(bv, cb, i, NORM);
                vmaxs(rel, x, 0.f, all);
                vabs(t, x, all);
                vmuls(t, t, -1.f, all, MODE_ZEROING);
                vexp(t, t, all, MODE_ZEROING);
                vadds(t, t, 1.f, all, MODE_ZEROING);
                vln(t, t, all, MODE_ZEROING);
                vadd(t, t, rel, all, MODE_ZEROING);
                vmuls(mask, x, -1.f, all, MODE_ZEROING);
                vadds(mask, mask, -10.f, all, MODE_ZEROING);
                vmaxs(mask, mask, 0.f, all);
                vmins(mask, mask, 1.f, all);
                vmins(floor, x, 0.f, all);
                vexp(floor, floor, all, MODE_ZEROING);
                vmul(floor, floor, mask, all, MODE_ZEROING);
                vmax(t, t, floor, all, MODE_ZEROING);
                vsqrt(t, t, all, MODE_ZEROING);
                vsts(t, sp, i, NORM_B32, all);
                vadd(t, t, bv, all, MODE_ZEROING);
                // Emit sort keys and invalidate padded experts alongside the score transform.
                RegTensor<int> key;
                vci(key, i, INC_ORDER);
                vsts(key, (__ubuf__ int *)kp, i, NORM_B32, all);
                if constexpr (E % 64) {
                    uint32_t live = E - i;
                    MaskReg valid = plt_b32(live, POST_UPDATE);
                    F lowest;
                    vdup(lowest, -3.402823466e38f, all, MODE_ZEROING);
                    vsel(t, t, lowest, valid);
                }
                vsts(t, bp, i, NORM_B32, all);
            }
        }
        pipe_barrier(PIPE_ALL);
        TSORT32(pairs, biased, keys);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        constexpr int lists = (E + 31) / 32;
        int heads[lists] = {};
        for (int j = 0; j < 6; j++) {
            int best = 0;
            for (int g = 1; g < lists; g++)
                if (pair[g * 64 + 2 * heads[g]] > pair[best * 64 + 2 * heads[best]])
                    best = g;
            int idx = ((__ubuf__ unsigned *)pair)[best * 64 + 2 * heads[best] + 1];
            ++heads[best];
            ip[j] = idx;
            wp[j] = sp[idx];
        }
        float sum = ((wp[0] + wp[4]) + (wp[2] + 0.f)) + ((wp[1] + wp[5]) + (wp[3] + 0.f));
        for (int j = 0; j < 6; j++)
            wp[j] = wp[j] / sum * 1.5f;
    } else
        for (int j = 0; j < 6; j++) {
            ip[j] = 0;
            wp[j] = 0;
        }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    store_row<int, 6, 8>(indices, ids);
    store_row<float, 6, 8>(weights, ws);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace hcpre
