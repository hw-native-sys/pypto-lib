// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Bounded-worker K64 online attention with paired Cube/Vector ownership.
#include <cstdint>
#if defined(__CPU_SIM)
#error "DSpark native attention requires A5 device execution"
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "sparse_softmax_vf.hpp"
#include "mixed_entry.hpp"

using namespace pto;
namespace {
constexpr int Q_L1 = 0;
constexpr int KV_L1 = 65536;
constexpr int P_L1 = 262144;
constexpr int QK_ACC = 131072;
constexpr int NUM_UB = 0;
constexpr int PV_UB = 65536;
constexpr int SCORE_UB = 131072;
constexpr int P_UB = 139264;
constexpr int MAX_UB = 143360;
constexpr int DEN_UB = 143488;
constexpr int CORR_UB = 143616;
constexpr int SINK_UB = 143744;
constexpr int NORM_UB = 143872;
constexpr int IDS_UB = 144000;
constexpr int SCORE_READY = 0, SCORE_FREE = 1;
constexpr int P_READY = 2, P_FREE = 3;
constexpr int PV_READY = 4, PV_FREE = 5;
constexpr int EXIT_C = 10, EXIT_V = 15;

template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C>
using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
__aicore__ inline int64_t row_stride(__gm__ int64_t* args, int index) {
    return reinterpret_cast<__gm__ Tensor*>(args[index])->strides[0];
}
template <int M>
using Query = Tile<TileType::Mat, bfloat16_t, M, 512, BLayout::ColMajor, M, 512, SLayout::RowMajor>;
using Keys = Tile<TileType::Mat, bfloat16_t, 192, 512, BLayout::ColMajor, 192, 512, SLayout::RowMajor>;
using KeysT = Tile<TileType::Mat, bfloat16_t, 512, 192, BLayout::RowMajor, 512, 192, SLayout::ColMajor>;
template <int M>
using PMat = Tile<TileType::Mat, bfloat16_t, M, 64, BLayout::ColMajor, M, 64, SLayout::RowMajor>;
template <int R>
using Probability = Tile<TileType::Vec, bfloat16_t, R, 64, BLayout::ColMajor, R, 64, SLayout::RowMajor, 512>;

#if defined(__DAV_CUBE__)
template <pipe_t Pipe>
__aicore__ inline void signal_both(int flag) {
    set_intra_block(Pipe, flag); set_intra_block(Pipe, flag + 16);
}
template <pipe_t Pipe>
__aicore__ inline void wait_both(int flag) {
    wait_intra_block(Pipe, flag); wait_intra_block(Pipe, flag + 16);
}

template <int M>
__aicore__ inline void cube(__gm__ int64_t* args) {
    constexpr int V = M / 2;
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID7); wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID7);
    signal_both<PIPE_MTE1>(P_FREE);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID4); set_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    Query<M> q;
    Keys kv; KeysT kv_t; PMat<M> p;
    TASSIGN(q, Q_L1); TASSIGN(kv, KV_L1); TASSIGN(kv_t, KV_L1); TASSIGN(p, P_L1);
    TileAcc<float, M, 64> qk;
    TileAcc<float, M, 512> pv;
    TASSIGN(qk, QK_ACC); TASSIGN(pv, 0);
    Vec<float, V, 64> score;
    Vec<float, V, 512> weighted;
    TASSIGN(score, SCORE_UB); TASSIGN(weighted, PV_UB);
    int tokens = args[5], worker = get_block_idx(args), workers = get_block_num(args);
    for (int token = worker; token < tokens; token += workers) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
        GM<bfloat16_t, M, 512> q_gm(data<bfloat16_t>(args, 0) + token * row_stride(args, 0));
        GM<bfloat16_t, 192, 512> kv_gm(data<bfloat16_t>(args, 1) + token * row_stride(args, 1));
        TLOAD(q, q_gm); TLOAD(kv, kv_gm);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        for (int part = 0; part < 3; ++part) {
            if (part == 0) wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
            for (int chunk = 0; chunk < 2; ++chunk) {
                auto event = static_cast<event_t>(chunk);
                TileLeft<bfloat16_t, M, 256> left;
                TileRight<bfloat16_t, 256, 64> right;
                TASSIGN(left, chunk * 32768); TASSIGN(right, chunk * 32768);
                wait_flag(PIPE_M, PIPE_MTE1, event);
                TEXTRACT(left, q, 0, chunk * 256);
                TEXTRACT(right, kv_t, chunk * 256, part * 64);
                set_flag(PIPE_MTE1, PIPE_M, event); wait_flag(PIPE_MTE1, PIPE_M, event);
                if (chunk == 0) TMATMUL(qk, left, right);
                else TMATMUL_ACC(qk, qk, left, right);
                set_flag(PIPE_M, PIPE_MTE1, event);
            }
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0); wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_both<PIPE_FIX>(SCORE_FREE);
            TMOV<decltype(score), decltype(qk), AccToVecMode::DualModeSplitM>(score, qk);
            signal_both<PIPE_FIX>(SCORE_READY);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
            wait_both<PIPE_MTE1>(P_READY);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TileLeft<bfloat16_t, M, 64> p_left;
            TileRight<bfloat16_t, 64, 512> value;
            TASSIGN(p_left, 0); TASSIGN(value, 0);
            TEXTRACT(p_left, p, 0, 0);
            TEXTRACT(value, kv, part * 64, 0);
            signal_both<PIPE_MTE1>(P_FREE);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2); wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
            TMATMUL(pv, p_left, value);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID1); wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
            wait_both<PIPE_FIX>(PV_FREE);
            TMOV<decltype(weighted), decltype(pv), AccToVecMode::DualModeSplitM>(weighted, pv);
            signal_both<PIPE_FIX>(PV_READY);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4); wait_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
    wait_both<PIPE_FIX>(SCORE_FREE); wait_both<PIPE_FIX>(PV_FREE);
    pipe_barrier(PIPE_ALL);
    signal_both<PIPE_FIX>(EXIT_C);
    wait_both<PIPE_FIX>(EXIT_V);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_VEC__)
template <int R>
__tf__ __aicore__ inline void softmax64(
    typename Probability<R>::TileDType __out__ weights,
    typename Vec<float, 1, R>::TileDType __out__ maximum,
    typename Vec<float, 1, R>::TileDType __out__ denominator,
    typename Vec<float, 1, R>::TileDType __out__ correction,
    typename Vec<float, R, 64>::TileDType __in__ score,
    typename Vec<int32_t, 1, 64>::TileDType __in__ indices,
    typename Vec<float, 1, R>::TileDType __in__ previous_max,
    typename Vec<float, 1, R>::TileDType __in__ previous_den) {
    auto p = (__ubuf__ uint16_t*)__cce_get_tile_ptr(weights);
    auto m = (__ubuf__ float*)__cce_get_tile_ptr(maximum);
    auto l = (__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto c = (__ubuf__ float*)__cce_get_tile_ptr(correction);
    auto old_max = (__ubuf__ float*)__cce_get_tile_ptr(previous_max);
    auto old_den = (__ubuf__ float*)__cce_get_tile_ptr(previous_den);
    auto qk = (__ubuf__ float*)__cce_get_tile_ptr(score);
    auto ids = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    __VEC_SCOPE__ {
        // Drain state initialization and the preceding softmax's state stores.
        mem_bar(VST_VLD);
        MaskReg all = pset_b32(PAT_ALL);
        uint32_t one_count = 1, p_count = 64;
        MaskReg one = CreatePredicate<float>(one_count);
        MaskReg p_mask = CreatePredicate<uint16_t>(p_count);
        RegTensor<int32_t> index;
        RegTensor<float> zero, invalid;
        vlds(index, ids, 0, NORM);
        MaskReg valid;
        vcmps_ge(valid, index, 0, all);
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(invalid, -1e30f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < R; ++row) {
            RegTensor<float> x, mx, old_m, old_l, corr, sum;
            vlds(x, qk, row * 64, NORM);
            vmuls(x, x, 0.0441941731f, all, MODE_ZEROING);
            vsel(x, x, invalid, valid);
            vcmax(mx, x, all, MODE_ZEROING);
            vdup(mx, mx, all, POS_LOWEST, MODE_ZEROING);
            vlds(old_m, old_max, row, BRC_B32);
            vlds(old_l, old_den, row, BRC_B32);
            vmax(mx, mx, old_m, all, MODE_ZEROING);
            vsub(corr, old_m, mx, all, MODE_ZEROING);
            vexp(corr, corr, all, MODE_ZEROING);
            vsub(x, x, mx, all, MODE_ZEROING);
            vexp(x, x, all, MODE_ZEROING);
            vsel(x, x, zero, valid);
            vcadd(sum, x, all, MODE_ZEROING);
            vmul(old_l, old_l, corr, one, MODE_ZEROING);
            vadd(old_l, old_l, sum, one, MODE_ZEROING);
            vsts(mx, m, row, ONEPT_B32, one);
            vsts(old_l, l, row, ONEPT_B32, one);
            vsts(corr, c, row, ONEPT_B32, one);
            RegTensor<bfloat16_t> bf, packed, unused;
            vcvt(bf, x, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vdintlv(packed, unused, bf, bf);
            auto target = p + row * 16;
            vsstb((RegTensor<uint16_t>&)packed, target, (uint32_t)(R << 16), p_mask, POST_UPDATE);
        }
        mem_bar(VST_VLD);
    }
}

template <int M>
__aicore__ inline void vector(__gm__ int64_t* args) {
    constexpr int V = M / 2;
    int heads = args[6], lane = get_sub_block_id(args);
    int valid = heads - lane * V;
    valid = valid < V ? valid : V;
    valid = valid > 0 ? valid : 0;
    Vec<float, V, 512> numerator, weighted;
    Vec<float, V, 64> score;
    Probability<V> weights;
    Vec<float, 1, V> maximum, denominator, correction, sinks, norm;
    Vec<int32_t, 1, 64> indices;
    TASSIGN(numerator, NUM_UB); TASSIGN(weighted, PV_UB); TASSIGN(score, SCORE_UB);
    TASSIGN(weights, P_UB); TASSIGN(maximum, MAX_UB); TASSIGN(denominator, DEN_UB);
    TASSIGN(correction, CORR_UB); TASSIGN(sinks, SINK_UB); TASSIGN(norm, NORM_UB); TASSIGN(indices, IDS_UB);
    TEXPANDS(sinks, 0.0f);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (valid > 0) {
        Tile<TileType::Vec, float, 1, V, BLayout::RowMajor, 1, DYNAMIC> active_sink(valid);
        TASSIGN(active_sink, SINK_UB);
        using S = TileShape2D<float, 1, DYNAMIC>;
        using B = BaseShape2D<float, 1, DYNAMIC>;
        GlobalTensor<float, S, B> input(data<float>(args, 3) + lane * V, S(1, valid), B(1, heads));
        TLOAD(active_sink, input);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_intra_block(PIPE_V, SCORE_FREE); set_intra_block(PIPE_V, PV_FREE);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    int tokens = args[5], worker = get_block_idx(args), workers = get_block_num(args);
    PMat<M> probability_l1;
    TASSIGN(probability_l1, P_L1);
    for (int token = worker; token < tokens; token += workers) {
        TEXPANDS(numerator, 0.0f); TEXPANDS(maximum, -1e30f); TEXPANDS(denominator, 0.0f);
        for (int part = 0; part < 3; ++part) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            GM<int32_t, 1, 64> ids(data<int32_t>(args, 2) + token * row_stride(args, 2) + part * 64);
            TLOAD(indices, ids);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_intra_block(PIPE_V, SCORE_READY);
            softmax64<V>(weights.data(), maximum.data(), denominator.data(), correction.data(), score.data(), indices.data(), maximum.data(), denominator.data());
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            set_intra_block(PIPE_V, SCORE_FREE);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_intra_block(PIPE_MTE3, P_FREE);
            TINSERT(probability_l1, weights, lane * V, 0);
            set_intra_block(PIPE_MTE3, P_READY);
            wait_intra_block(PIPE_V, PV_READY);
            sparse_softmax::accumulate<decltype(numerator), decltype(correction)>(
                numerator.data(), numerator.data(), weighted.data(), correction.data());
            if (part < 2) set_intra_block(PIPE_V, PV_FREE);
        }
        sparse_softmax::normalization<decltype(norm), true>(norm.data(), maximum.data(), denominator.data(), sinks.data());
        Vec<bfloat16_t, V, 512> result;
        TASSIGN(result, PV_UB);
        sparse_softmax::finish<true, decltype(result), decltype(numerator), decltype(norm)>(
            result.data(), numerator.data(), numerator.data(), correction.data(), norm.data());
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        if (valid > 0) {
            Tile<TileType::Vec, bfloat16_t, V, 512, BLayout::RowMajor, DYNAMIC, 512> active_result(valid);
            TASSIGN(active_result, PV_UB);
            using S = TileShape2D<bfloat16_t, DYNAMIC, 512>;
            using B = BaseShape2D<bfloat16_t, DYNAMIC, 512>;
            GlobalTensor<bfloat16_t, S, B> out(data<bfloat16_t>(args, 4) + token * row_stride(args, 4) + lane * V * 512, S(valid, 512), B(valid, 512));
            TSTORE(out, active_result);
        }
        pipe_barrier(PIPE_ALL);
        set_intra_block(PIPE_V, PV_FREE);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_intra_block(PIPE_MTE3, P_FREE);
    pipe_barrier(PIPE_ALL);
    wait_intra_block(PIPE_V, EXIT_C);
    set_intra_block(PIPE_V, EXIT_V);
    pipe_barrier(PIPE_ALL);
}
#endif

}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    native_lifecycle::enter_dual_aiv();
#if defined(__DAV_CUBE__)
    const uint64_t saved = get_ctrl();
    if (args[6] > 32) cube<64>(args); else cube<32>(args);
    set_ctrl(saved);
#elif defined(__DAV_VEC__)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
    if (args[6] > 32) vector<64>(args); else vector<32>(args);
#endif
}
