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

#if defined(__CPU_SIM)
using QBRopePipe = TPipe<0, Direction::DIR_C2V, 20480, 2, 2>;
#endif
using QBRopeAcc = TileAcc<float, 80, 64>;
using QBRopeRaw = Tile<TileType::Vec, float, 80, 64>;
using QBRopeResult = Tile<TileType::Vec, bfloat16_t, 80, 64>;
using QBRopeAngles = Tile<TileType::Vec, float, 80, 32>;

#if defined(__DAV_CUBE__)
template <bool ENABLED>
struct QBRopeTransfer {
    __aicore__ inline void finish() {}
};

template <>
struct QBRopeTransfer<true> {
    int transferred = 0;
#if defined(__CPU_SIM)
    QBRopePipe pipe;
    __aicore__ QBRopeTransfer() : pipe(nullptr, 0, 0) {}
#endif
    __aicore__ inline void push(QBRopeAcc& tail) {
#if defined(__CPU_SIM)
        TPUSH<QBRopePipe, QBRopeAcc, TileSplitAxis::NONE>(pipe, tail);
#else
        int slot = transferred % 2;
        // The vector lane owns both slots before the first transfer.
        wait_intra_block(PIPE_FIX, slot * 2 + 1);
        QBRopeRaw raw;
        TASSIGN(raw, slot * 20480);
        TMOV<QBRopeRaw, QBRopeAcc, AccToVecMode::SingleModeVec0>(raw, tail);
        set_intra_block(PIPE_FIX, slot * 2);
#endif
        ++transferred;
    }
    __aicore__ inline void finish() {
#if !defined(__CPU_SIM)
        if (transferred > 0) {
            // Drain both credits, including an unused second startup slot.
            // Vector may return only after this final Cube acknowledgement.
            wait_intra_block(PIPE_FIX, 1);
            wait_intra_block(PIPE_FIX, 3);
            set_intra_block(PIPE_FIX, 4);
        }
#endif
    }
};

template <int N>
__aicore__ inline void load_b(__gm__ F* b, int step, int block, int workers, int slot) {
    int n = (step / 5 * workers + block) * 256;
    int k = step % 5 * 256;
    MatB tile;
    TASSIGN(tile, 196608 + slot * 65536);
    GM<F, 256, 256, 1280, N, Layout::NZ> gm(b + n * 1280 + k * 32);
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

template <int N, int WORKERS = (N / 256 < 32 ? N / 256 : 32), bool OFFSET_INPUT = false, bool FUSED_ROPE = false>
__aicore__ inline void project_qb(__gm__ int64_t* args) {
    // Preserve constant loop bounds for full-width launches.
    if constexpr (WORKERS > 0) {
        if (get_block_num(args) != WORKERS) {
            project_qb<N, 0, OFFSET_INPUT, FUSED_ROPE>(args);
            return;
        }
    }
    // Runtime ABI groups Tensor descriptors before scalar arguments.
    const int rows = args[FUSED_ROPE ? 7 : 5];
    const int row_start = args[FUSED_ROPE ? 9 : 7];
    QBRopeTransfer<FUSED_ROPE> rope_transfer;
    int block = get_block_idx(args);
    int workers = WORKERS;
    int waves;
    if constexpr (WORKERS > 0) {
        waves = N / (256 * WORKERS);
    } else {
        workers = get_block_num(args);
        waves = (N / 256 - block + workers - 1) / workers;
    }
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
    // At most eight scale tiles fit below the payload buffers at 196608.
    for (int first_wave = 0; first_wave < waves; first_wave += 8) {
        int batch_waves = waves - first_wave < 8 ? waves - first_wave : 8;
        for (int wave = 0; wave < batch_waves; ++wave) {
            ScaleB sb;
            TASSIGN(sb, 105600 + wave * 10240);
            int n = ((first_wave + wave) * workers + block) * 256;
            GM<S, 40, 256, 40, N, Layout::MX_B_NN> sbg(data<S>(args, 3) + n * 40);
            TLOAD(sb, sbg);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        for (int slot = 0; slot < 2; ++slot) {
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        for (int slot = 0; slot < 4; ++slot) {
            load_b<N>(data<F>(args, 1), first_wave * 5 + slot, block, workers, slot);
        }
        for (int step = 0; step < 5 * batch_waves; ++step) {
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
            if (step + 4 < 5 * batch_waves) {
                set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
                load_b<N>(data<F>(args, 1), first_wave * 5 + step + 4, block, workers, slot);
            }
            if (step % 5 == 4) {
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                acc.SetValidRow(rows);
                using OS = TileShape2D<bfloat16_t, DYNAMIC, 256, Layout::ND>;
                using ST = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC, Layout::ND>;
                int n = ((first_wave + wave) * workers + block) * 256;
                GlobalTensor<bfloat16_t, OS, ST, Layout::ND> cg(
                    data<bfloat16_t>(args, 4) + row_start * N + n, OS(rows, 256), ST(rows, N));
                if constexpr (FUSED_ROPE) {
                    if (n % 512 == 256) {
                        TileAcc<float, 80, 256, DYNAMIC, 192> prefix(rows);
                        TASSIGN(prefix, wave % 2 * 81920);
                        using PS = TileShape2D<bfloat16_t, DYNAMIC, 192, Layout::ND>;
                        GlobalTensor<bfloat16_t, PS, ST, Layout::ND> prefix_gm(
                            data<bfloat16_t>(args, 4) + row_start * N + n, PS(rows, 192), ST(rows, N));
                        TSTORE(prefix_gm, prefix);
                        QBRopeAcc tail;
                        TASSIGN(tail, wave % 2 * 81920 + 192 * 80 * sizeof(float));
                        rope_transfer.push(tail);
                    } else {
                        TSTORE(cg, acc);
                    }
                } else {
                    TSTORE(cg, acc);
                }
                set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + wave % 2));
            }
        }
        for (int slot = 0; slot < 2; ++slot) {
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(4 + slot));
        }
        pipe_barrier(PIPE_ALL);
    }
    rope_transfer.finish();
}
#endif

#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void qb_rotate_tail(
    QBRopeResult::TileDType __out__ output, QBRopeRaw::TileDType __in__ input,
    QBRopeAngles::TileDType __in__ cosine, QBRopeAngles::TileDType __in__ sine, uint16_t rows) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        for (int pair = 0; pair < 32; ++pair) {
            float even = static_cast<float>(static_cast<bfloat16_t>(src[row * 64 + pair * 2]));
            float odd = static_cast<float>(static_cast<bfloat16_t>(src[row * 64 + pair * 2 + 1]));
            float ec = even * cp[row * 32 + pair];
            float os = odd * sp[row * 32 + pair];
            float es = even * sp[row * 32 + pair];
            float oc = odd * cp[row * 32 + pair];
            dst[row * 64 + pair * 2] = static_cast<bfloat16_t>(ec - os);
            dst[row * 64 + pair * 2 + 1] = static_cast<bfloat16_t>(es + oc);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg half = plt_b32(count, POST_UPDATE);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> values, even, odd, cosv, sinv, a, b, real, imag, rotated, unused;
            vlds(values, src, row * 64, NORM);
            vcvt(bf, values, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vcvt(values, bf, full, PART_EVEN);
            vlds(cosv, cp, row * 32, NORM);
            vlds(sinv, sp, row * 32, NORM);
            vdintlv(even, odd, values, values);
            vmul(a, even, cosv, half, MODE_ZEROING);
            vmul(b, odd, sinv, half, MODE_ZEROING);
            vsub(real, a, b, half);
            vmul(a, even, sinv, half, MODE_ZEROING);
            vmul(b, odd, cosv, half, MODE_ZEROING);
            vadd(imag, a, b, half, MODE_ZEROING);
            vintlv(rotated, unused, real, imag);
            vcvt(bf, rotated, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row * 64, PK_B32, full);
        }
    }
#endif
}

__aicore__ inline void project_qb_rope(__gm__ int64_t* args) {
    int block = get_block_idx(args);
    int workers = get_block_num(args);
    int width = args[8];
    int waves = (width / 256 - block + workers - 1) / workers;
    int rows = args[7];
    bool have_tail = false;
    for (int wave = 0; wave < waves; ++wave) {
        if (((wave * workers + block) & 1) != 0) have_tail = true;
    }
    if (!have_tail) return;
#if defined(__CPU_SIM)
    QBRopePipe pipe(nullptr, 0, 0);
#endif
    QBRopeAngles cosine, sine;
    QBRopeResult result;
    TASSIGN(cosine, 40960);
    TASSIGN(sine, 51200);
    TASSIGN(result, 61440);
    if (rows > 0) {
        using Shape = TileShape2D<float, DYNAMIC, 32>;
        using Stride = BaseShape2D<float, DYNAMIC, DYNAMIC>;
        auto cos_tensor = reinterpret_cast<__gm__ Tensor*>(args[5]);
        auto sin_tensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
        Tile<TileType::Vec, float, 80, 32, BLayout::RowMajor, DYNAMIC, 32> c(rows), s(rows);
        TASSIGN(c, 40960);
        TASSIGN(s, 51200);
        GlobalTensor<float, Shape, Stride> cg(data<float>(args, 5) + args[9] * cos_tensor->strides[0],
            Shape(rows, 32), Stride(rows, cos_tensor->strides[0]));
        GlobalTensor<float, Shape, Stride> sg(data<float>(args, 6) + args[9] * sin_tensor->strides[0],
            Shape(rows, 32), Stride(rows, sin_tensor->strides[0]));
        TLOAD(c, cg);
        TLOAD(s, sg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }
    int received = 0;
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#if !defined(__CPU_SIM)
    // No-tail lanes returned above. Active lanes lend two 20480-byte slots;
    // the Cube consumes every credit and acknowledges their final return.
    set_intra_block(PIPE_V, 1);
    set_intra_block(PIPE_V, 3);
#endif
    for (int wave = 0; wave < waves; ++wave) {
        int n = (wave * workers + block) * 256;
        if (n % 512 != 256) continue;
        int slot = received % 2;
        QBRopeRaw raw;
#if defined(__CPU_SIM)
        TPOP<QBRopePipe, QBRopeRaw, TileSplitAxis::NONE>(pipe, raw);
#else
        TASSIGN(raw, slot * 20480);
        wait_intra_block(PIPE_V, slot * 2);
#endif
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        qb_rotate_tail(result.data(), raw.data(), cosine.data(), sine.data(), rows);
#if defined(__CPU_SIM)
        TFREE<QBRopePipe, TileSplitAxis::NONE>(pipe);
#else
        set_intra_block(PIPE_V, slot * 2 + 1);
#endif
        if (rows > 0) {
            using Shape = TileShape2D<bfloat16_t, DYNAMIC, 64>;
            using Stride = BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC>;
            Tile<TileType::Vec, bfloat16_t, 80, 64, BLayout::RowMajor, DYNAMIC, 64> active_result(rows);
            TASSIGN(active_result, 61440);
            GlobalTensor<bfloat16_t, Shape, Stride> output(data<bfloat16_t>(args, 4) + args[9] * width + n + 192,
                Shape(rows, 64), Stride(rows, width));
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(output, active_result);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        ++received;
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#if !defined(__CPU_SIM)
    wait_intra_block(PIPE_V, 4);
#endif
    pipe_barrier(PIPE_ALL);
}
#endif
