// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed-cache gather, direct-L1 KV reuse, and sink-aware sparse attention.
#include <cstdint>
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "sparse_dequant_vf.hpp"
#include "sparse_softmax_vf.hpp"

using namespace pto;

namespace {
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C>
using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <event_t Event, typename T, int PayloadColumns, int ScaleColumns>
__aicore__ inline void load_rows(
    __gm__ T* payload, __gm__ bfloat16_t* scales, Vec<int32_t, 1, 32>& ids) {
    if constexpr (PayloadColumns == 512) {
        int first = ids.GetValue(0);
        bool contiguous = first >= 0;
        for (int row = 1; row < 32 && contiguous; ++row) {
            contiguous = (int64_t)ids.GetValue(row) == (int64_t)first + row;
        }
        if (contiguous) {
            Vec<T, 32, PayloadColumns> rows;
            Vec<bfloat16_t, 32, ScaleColumns> factors;
            TASSIGN(rows, 0);
            TASSIGN(factors, 16384);
            GM<T, 32, PayloadColumns> input(payload + (int64_t)first * PayloadColumns);
            GM<bfloat16_t, 32, ScaleColumns> scale(scales + (int64_t)first * ScaleColumns);
            TLOAD(rows, input);
            TLOAD(factors, scale);
            set_flag(PIPE_MTE2, PIPE_V, Event);
            wait_flag(PIPE_MTE2, PIPE_V, Event);
            return;
        }
    }
    for (int lane = 0; lane < 32; lane += 2) {
        int first = ids.GetValue(lane);
        int second = ids.GetValue(lane + 1);
        if (first >= 0 && second >= 0 && first != second) {
            int low = first < second ? first : second;
            int high = first < second ? second : first;
            Vec<T, 2, PayloadColumns> rows;
            Vec<bfloat16_t, 2, ScaleColumns> factors;
            TASSIGN(rows, lane * PayloadColumns * sizeof(T));
            TASSIGN(factors, 16384 + lane * ScaleColumns * sizeof(bfloat16_t));
            using PayloadShape = Shape<1, 1, 1, 2, PayloadColumns>;
            using ScaleShape = Shape<1, 1, 1, 2, ScaleColumns>;
            using PairStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
            GlobalTensor<T, PayloadShape, PairStride> input(
                payload + (int64_t)low * PayloadColumns, PayloadShape(),
                PairStride(1, 1, 1, (int64_t)(high - low) * PayloadColumns, 1));
            GlobalTensor<bfloat16_t, ScaleShape, PairStride> scale(
                scales + (int64_t)low * ScaleColumns, ScaleShape(),
                PairStride(1, 1, 1, (int64_t)(high - low) * ScaleColumns, 1));
            TLOAD(rows, input);
            TLOAD(factors, scale);
        } else {
            for (int pair_row = 0; pair_row < 2; ++pair_row) {
                int physical = pair_row == 0 ? first : second;
                if (physical < 0) physical = 0;
                Vec<T, 1, PayloadColumns> row;
                Vec<bfloat16_t, 1, ScaleColumns> factor;
                TASSIGN(row, (lane + pair_row) * PayloadColumns * sizeof(T));
                TASSIGN(factor, 16384 + (lane + pair_row) * ScaleColumns * sizeof(bfloat16_t));
                GM<T, 1, PayloadColumns> input(payload + (int64_t)physical * PayloadColumns);
                GM<bfloat16_t, 1, ScaleColumns> scale(scales + (int64_t)physical * ScaleColumns);
                TLOAD(row, input);
                TLOAD(factor, scale);
            }
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, Event);
    wait_flag(PIPE_MTE2, PIPE_V, Event);
}


#endif


constexpr int ARG_TOKENS = 9;
constexpr int ARG_HEADS = ARG_TOKENS + 1;
constexpr int ARG_WINDOW_KEYS = ARG_TOKENS + 2;
constexpr int ARG_COMPRESSED_KEYS = ARG_TOKENS + 3;
constexpr int DIM = 512;
// L1: P ring [0,32 KiB), Q [64,128 KiB), KV ring [128,512 KiB).
constexpr int QUERY_L1 = 65536;
constexpr int KV_L1 = 131072;
constexpr int MAXIMUM_UB = 205056;
constexpr int DENOMINATOR_UB = 205184;
constexpr int NUMERATOR_UB = 131072;
constexpr int NEXT_MAX_UB = 205312;
constexpr int CORRECTION_UB = 205440;
// UB: gather [0,64 KiB), PV [64,128 KiB), numerator [128,192 KiB).
// Score occupies [216,232 KiB); state, P and IDs occupy the intervening region.
constexpr int SCORE_UB = 221184;
constexpr int QUERY_STATE_STRIDE = 1024;
constexpr int CORRECTION_STRIDE = 256;
constexpr int KV_SLOTS = 3;
// Intra-block pairs: score 0/1, PV 2/3, P slots 4/5 and 6/7,
// KV slots 8/9, 10/11 and 12/13. Cube uses both AIV flag banks.
constexpr int FINISH_READY = 14;
constexpr int FINISH_ACK = 15;

template <int R, int C>
using Mat = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
template <int R, int C>
using TransposedMat = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::ColMajor>;
template <int R, int C>
using Left = Tile<TileType::Left, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
template <int R, int C>
using Right = Tile<TileType::Right, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::ColMajor>;
template <int R, int C>
using Acc = Tile<TileType::Acc, float, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor, 1024>;
template <int H>
using HeadColumn = Tile<TileType::Vec, float, H, 1, BLayout::ColMajor>;

__aicore__ inline int64_t row_stride(__gm__ int64_t* args, int index) {
    return reinterpret_cast<__gm__ Tensor*>(args[index])->strides[0];
}

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <int Slot, bool WaitInput = true>
__aicore__ inline void gather_fp4_subtile(
    __gm__ int64_t* args, int index_offset, int destination_row, Mat<128, DIM>& kv) {
    constexpr int payload_address = Slot * 8192;
    constexpr int scale_address = 16384 + Slot * 1024;
    int id_address = 24576 + index_offset * sizeof(int32_t);
    constexpr int nz_address = 32768 + Slot * 16384;
    Vec<int32_t, 1, 16> ids;
    Vec<float4_e2m1x2_t, 16, DIM> payload;
    Vec<bfloat16_t, 16, 32> scales;
    Tile<TileType::Vec, bfloat16_t, 16, DIM, BLayout::ColMajor, 16, DIM, SLayout::RowMajor> nz;
    TASSIGN(ids, id_address);
    TASSIGN(payload, payload_address);
    TASSIGN(scales, scale_address);
    TASSIGN(nz, nz_address);
    auto payload_base = data<uint8_t>(args, 5);
    auto scale_base = data<bfloat16_t>(args, 6);

    // Payload/scales are reusable after VF consumes this slot. The 64
    // indices remain immutable throughout the gather; NZ has its own credit.
    if constexpr (WaitInput) wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(5 + Slot));
    for (int row = 0; row < 16; row += 2) {
        int first = ids.GetValue(row);
        int second = ids.GetValue(row + 1);
        if (first >= 0 && second >= 0 && first != second) {
            int low = first < second ? first : second;
            int high = first < second ? second : first;
            Vec<uint8_t, 2, 256> pair;
            Vec<bfloat16_t, 2, 32> factor_pair;
            TASSIGN(pair, payload_address + row * 256);
            TASSIGN(factor_pair, scale_address + row * 64);
            using PairStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
            using PayloadShape = Shape<1, 1, 1, 2, 256>;
            using ScaleShape = Shape<1, 1, 1, 2, 32>;
            GlobalTensor<uint8_t, PayloadShape, PairStride> input(
                payload_base + (int64_t)low * 256, PayloadShape(),
                PairStride(1, 1, 1, (int64_t)(high - low) * 256, 1));
            GlobalTensor<bfloat16_t, ScaleShape, PairStride> scale(
                scale_base + (int64_t)low * 32, ScaleShape(),
                PairStride(1, 1, 1, (int64_t)(high - low) * 32, 1));
            TLOAD(pair, input);
            TLOAD(factor_pair, scale);
        } else {
            for (int pair_row = 0; pair_row < 2; ++pair_row) {
                int physical = pair_row == 0 ? first : second;
                if (physical < 0) physical = 0;
                Vec<uint8_t, 1, 256> value;
                Vec<bfloat16_t, 1, 32> factor;
                TASSIGN(value, payload_address + (row + pair_row) * 256);
                TASSIGN(factor, scale_address + (row + pair_row) * 64);
                GM<uint8_t, 1, 256> input(payload_base + (int64_t)physical * 256);
                GM<bfloat16_t, 1, 32> scale(scale_base + (int64_t)physical * 32);
                TLOAD(value, input);
                TLOAD(factor, scale);
            }
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(5 + Slot));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(5 + Slot));
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(5 + Slot));
    sparse_dequant::decode_nz<true, decltype(nz), decltype(payload), decltype(scales), decltype(ids)>(
        nz.data(), payload.data(), scales.data(), ids.data());
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(5 + Slot));
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(5 + Slot));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(5 + Slot));
    TINSERT(kv, nz, destination_row, 0);
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(5 + Slot));
}

__aicore__ inline void drain_fp4_slots(bool& active) {
    if (active) {
        // FP8 aliases both FP4 input and NZ slots. Consume all outstanding
        // reader/copy credits before changing formats or retiring the kernel.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        active = false;
    }
}

__aicore__ inline void gather_part(
    __gm__ int64_t* args, int token, int part, int lane, int ordinal, bool& fp4_active) {
    wait_intra_block(PIPE_MTE3, 9 + 2 * (ordinal % KV_SLOTS));
    if (part * 128 >= args[ARG_WINDOW_KEYS]) {
        Mat<128, DIM> kv;
        TASSIGN(kv, KV_L1 + (ordinal % KV_SLOTS) * 131072);
        if (!fp4_active) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
            fp4_active = true;
        }
        // Both final VFs read the shared 64-ID array. Acquire their input
        // credits before replacing IDs, then reuse those acquisitions for
        // this part's first payload load in each slot.
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        int source_column = part * 128 + lane * 64 - args[ARG_WINDOW_KEYS];
        Vec<int32_t, 1, 64> all_ids;
        TASSIGN(all_ids, 24576);
        GM<int32_t, 1, 64> source_ids(
            data<int32_t>(args, 8) + token * row_stride(args, 8) + source_column);
        TLOAD(all_ids, source_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        gather_fp4_subtile<0, false>(args, 0, lane * 64, kv);
        gather_fp4_subtile<1, false>(args, 16, lane * 64 + 16, kv);
        gather_fp4_subtile<0>(args, 32, lane * 64 + 32, kv);
        gather_fp4_subtile<1>(args, 48, lane * 64 + 48, kv);
        // Keep NZ copy credits live: following softmax does not touch NZ.
        set_intra_block(PIPE_MTE3, 8 + 2 * (ordinal % KV_SLOTS));
        return;
    }
    drain_fp4_slots(fp4_active);
    Vec<int32_t, 1, 32> ids;
    Tile<TileType::Vec, bfloat16_t, 32, DIM, BLayout::ColMajor, 32, DIM, SLayout::RowMajor> nz;
    Mat<128, DIM> kv;
    TASSIGN(ids, 28672);
    TASSIGN(nz, 32768);
    TASSIGN(kv, KV_L1 + (ordinal % KV_SLOTS) * 131072);
    for (int sub = 0; sub < 2; ++sub) {
        int column = part * 128 + lane * 64 + sub * 32;
        int source = column < args[ARG_WINDOW_KEYS] ? 7 : 8;
        int source_column = column < args[ARG_WINDOW_KEYS] ? column : column - args[ARG_WINDOW_KEYS];
        GM<int32_t, 1, 32> source_ids(
            data<int32_t>(args, source) + token * row_stride(args, source) + source_column);
        TLOAD(ids, source_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        if (column < args[ARG_WINDOW_KEYS]) {
            Vec<float8_e4m3_t, 32, DIM> payload;
            Vec<bfloat16_t, 32, 16> scales;
            TASSIGN(payload, 0);
            TASSIGN(scales, 16384);
            load_rows<EVENT_ID7, float8_e4m3_t, 512, 16>(
                data<float8_e4m3_t>(args, 3), data<bfloat16_t>(args, 4), ids);
            sparse_dequant::decode_nz<false, decltype(nz), decltype(payload), decltype(scales), decltype(ids)>(
                nz.data(), payload.data(), scales.data(), ids.data());
        } else {
            Vec<float4_e2m1x2_t, 32, DIM> payload;
            Vec<bfloat16_t, 32, 32> scales;
            TASSIGN(payload, 0);
            TASSIGN(scales, 16384);
            load_rows<EVENT_ID7, uint8_t, 256, 32>(data<uint8_t>(args, 5), data<bfloat16_t>(args, 6), ids);
            sparse_dequant::decode_nz<true, decltype(nz), decltype(payload), decltype(scales), decltype(ids)>(
                nz.data(), payload.data(), scales.data(), ids.data());
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        TINSERT(kv, nz, lane * 64 + sub * 32, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    }
    set_intra_block(PIPE_MTE3, 8 + 2 * (ordinal % KV_SLOTS));
}
#endif

#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
template <pipe_t Pipe>
__aicore__ inline void signal_vectors(int flag) {
    set_intra_block(Pipe, flag);
    set_intra_block(Pipe, flag + 16);
}

template <pipe_t Pipe>
__aicore__ inline void wait_vectors(int flag) {
    wait_intra_block(Pipe, flag);
    wait_intra_block(Pipe, flag + 16);
}

template <int H, int K, typename Query>
__aicore__ inline void compute_qk(Query& query, int ordinal) {
    constexpr int QK_STEP = 128;
    constexpr int QK_END = DIM;
    constexpr int LEFT_SECOND = H * QK_STEP * 2;
    TransposedMat<DIM, K> key;
    Acc<H, K> scores;
    TASSIGN(key, KV_L1 + (ordinal % KV_SLOTS) * 131072);
    // Score and PV have independent L0C ownership through their FIXPIPE copies.
    TASSIGN(scores, 131072);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_vectors<PIPE_MTE1>(8 + 2 * (ordinal % KV_SLOTS));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int offset = 0; offset < QK_END; offset += 2 * QK_STEP) {
        Left<H, QK_STEP> q0, q1;
        Right<QK_STEP, K> k0, k1;
        TASSIGN(q0, LEFT_SECOND);
        TASSIGN(q1, 0);
        TASSIGN(k0, 0);
        TASSIGN(k1, 32768);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_MTE1);
        TEXTRACT(q0, query, 0, offset);
        TEXTRACT(k0, key, offset, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(q1, query, 0, offset + QK_STEP);
        TEXTRACT(k1, key, offset + QK_STEP, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        pipe_barrier(PIPE_M);
        if (offset == 0) TMATMUL(scores, q0, k0);
        else TMATMUL_ACC(scores, scores, q0, k0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        TMATMUL_ACC(scores, scores, q1, k1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_vectors<PIPE_FIX>(1);
    Vec<float, H / 2, K> received;
    TASSIGN(received, SCORE_UB);
    TMOV<decltype(received), decltype(scores), AccToVecMode::DualModeSplitM>(received, scores);
    signal_vectors<PIPE_FIX>(0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

template <int H, int K>
__aicore__ inline void compute_pv(int ordinal) {
    Mat<K, DIM> kv;
    Mat<H, K> weights;
    Acc<H, DIM> weighted;
    TASSIGN(kv, KV_L1 + (ordinal % KV_SLOTS) * 131072);
    TASSIGN(weights, (ordinal % 2) * H * K * sizeof(bfloat16_t));
    TASSIGN(weighted, 0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_vectors<PIPE_MTE1>(4 + 2 * (ordinal % 2));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    Left<H, K> p;
    TASSIGN(p, 0);
    TEXTRACT(p, weights, 0, 0);
    signal_vectors<PIPE_MTE1>(5 + 2 * (ordinal % 2));
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    for (int offset = 0; offset < DIM; offset += 256) {
        Right<K, 128> v0, v1;
        Acc<H, 128> out0, out1;
        TASSIGN(v0, 0);
        TASSIGN(v1, 32768);
        TASSIGN(out0, offset * H * sizeof(float));
        TASSIGN(out1, (offset + 128) * H * sizeof(float));
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        TEXTRACT(v0, kv, 0, offset);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        TEXTRACT(v1, kv, 0, offset + 128);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
        pipe_barrier(PIPE_M);
        TMATMUL(out0, p, v0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        TMATMUL(out1, p, v1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    }
    // KV remains live through the final PV L1 read, including both AIV halves.
    signal_vectors<PIPE_MTE1>(9 + 2 * (ordinal % KV_SLOTS));
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_vectors<PIPE_FIX>(3);
    Vec<float, H / 2, DIM> received;
    TASSIGN(received, 65536);
    TMOV<decltype(received), decltype(weighted), AccToVecMode::DualModeSplitM>(received, weighted);
    signal_vectors<PIPE_FIX>(2);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
}

template <int H, int K>
__aicore__ inline void attend_cube(__gm__ int64_t* args) {
    const int workers = get_block_num(args);
    const int block = get_block_idx(args);
    if (block >= args[ARG_TOKENS]) return;
    const int queries = (args[ARG_TOKENS] - block + workers - 1) / workers;
    const int parts = (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / K;
    const int tiles = queries * parts;
    if (tiles == 0) return;
    const int valid_heads = args[ARG_HEADS] < H ? args[ARG_HEADS] : H;
    Tile<TileType::Mat, bfloat16_t, H, DIM, BLayout::ColMajor,
         DYNAMIC, DIM, SLayout::RowMajor, 512, PadValue::Zero> query(valid_heads);
    TASSIGN(query, QUERY_L1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    signal_vectors<PIPE_MTE1>(5);
    signal_vectors<PIPE_MTE1>(7);
    for (int slot = 0; slot < KV_SLOTS; ++slot) signal_vectors<PIPE_MTE1>(9 + 2 * slot);
    // Cube issues QK(i) then PV(i-1); Vector advances softmax on its own pipe.
    for (int ordinal = 0; ordinal <= tiles; ++ordinal) {
        if (ordinal < tiles) {
            const int part = ordinal % parts;
            if (part == 0) {
                const int token = block + (ordinal / parts) * workers;
                using QueryShape = TileShape2D<bfloat16_t, DYNAMIC, DIM>;
                GlobalTensor<bfloat16_t, QueryShape, BaseShape2D<bfloat16_t, H, DIM>> query_gm(
                    data<bfloat16_t>(args, 1) + token * row_stride(args, 1), QueryShape(valid_heads, DIM));
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                TLOAD(query, query_gm);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            }
            compute_qk<H, K>(query, ordinal);
            if (part + 1 == parts) set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        }
        if (ordinal > 0) compute_pv<H, K>(ordinal - 1);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    wait_vectors<PIPE_FIX>(1);
    wait_vectors<PIPE_FIX>(3);
    wait_vectors<PIPE_MTE1>(FINISH_ACK);
    pipe_barrier(PIPE_ALL);
    signal_vectors<PIPE_FIX>(FINISH_READY);
}
#endif

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <int H, int K>
__aicore__ inline void update_softmax(__gm__ int64_t* args, int token, int part, int ordinal, int query_slot) {
    Vec<float, H, K> qk;
    Vec<int32_t, 1, K> ids;
    Vec<float, 1, H> maximum, denominator, next_max, correction;
    Tile<TileType::Vec, bfloat16_t, H, K, BLayout::ColMajor, H, K, SLayout::RowMajor> weights;
    Mat<2 * H, K> weights_l1;
    TASSIGN(qk, SCORE_UB);
    TASSIGN(ids, 217088);
    TASSIGN(maximum, MAXIMUM_UB + query_slot * QUERY_STATE_STRIDE);
    TASSIGN(denominator, DENOMINATOR_UB + query_slot * QUERY_STATE_STRIDE);
    TASSIGN(next_max, NEXT_MAX_UB);
    TASSIGN(correction, CORRECTION_UB + (ordinal % 2) * CORRECTION_STRIDE);
    TASSIGN(weights, 208896);
    TASSIGN(weights_l1, (ordinal % 2) * 2 * H * K * sizeof(bfloat16_t));
    if (part == 0) {
        TEXPANDS(maximum, -1e30f);
        TEXPANDS(denominator, 0.0f);
    }
    const int source = part * K < args[ARG_WINDOW_KEYS] ? 7 : 8;
    const int column = part * K < args[ARG_WINDOW_KEYS] ? part * K : part * K - args[ARG_WINDOW_KEYS];
    GM<int32_t, 1, K> ids_gm(data<int32_t>(args, source) + token * row_stride(args, source) + column);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(ids, ids_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_intra_block(PIPE_V, 0);
    // The previous P copy owns only this NZ staging tile, not gather or PV UB.
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    sparse_softmax::update<decltype(weights), decltype(qk), decltype(ids), decltype(maximum)>(
        weights.data(), next_max.data(), denominator.data(), correction.data(),
        qk.data(), ids.data(), maximum.data(), denominator.data());
    TMOV(maximum, next_max);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_intra_block(PIPE_V, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_intra_block(PIPE_MTE3, 5 + 2 * (ordinal % 2));
    TINSERT(weights_l1, weights, get_sub_block_id(args) * H, 0);
    set_intra_block(PIPE_MTE3, 4 + 2 * (ordinal % 2));
    // Return the P staging credit; the next softmax consumes it before writing.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
}

template <int H>
__aicore__ inline void finalize_query(__gm__ int64_t* args, int token, int query_slot, bool single_part) {
    const int head = get_sub_block_id(args) * H;
    const int valid_heads = args[ARG_HEADS] < H ? args[ARG_HEADS] : H;
    Vec<float, 1, H> maximum, denominator, sink, final_max, correction;
    Vec<float, H, DIM> numerator;
    TASSIGN(maximum, MAXIMUM_UB + query_slot * QUERY_STATE_STRIDE);
    TASSIGN(denominator, DENOMINATOR_UB + query_slot * QUERY_STATE_STRIDE);
    TASSIGN(numerator, single_part ? 65536 : NUMERATOR_UB);
    TASSIGN(sink, 196608);
    TASSIGN(final_max, 196736);
    // The other tile's alpha may already be live; finalization uses separate scratch.
    TASSIGN(correction, 196864);
    Tile<TileType::Vec, float, 1, H, BLayout::RowMajor, 1, DYNAMIC> sink_load(valid_heads);
    TASSIGN(sink_load, 196608);
    using SinkShape = TileShape2D<float, 1, DYNAMIC>;
    auto sink_data = data<float>(args, 2);
    const int64_t sink_stride = row_stride(args, 2);
    if (sink_stride == 1) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        GlobalTensor<float, SinkShape, BaseShape2D<float, 1, H>> sink_gm(
            sink_data + head, SinkShape(1, valid_heads));
        TLOAD(sink_load, sink_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    } else {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        for (int lane = 0; lane < H; ++lane) {
            const float value = lane < valid_heads ? sink_data[(head + lane) * sink_stride] : 0.0f;
            sink.SetValue(lane, value);
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }
    TMAX(final_max, maximum, sink);
    TSUB(correction, maximum, final_max);
    TEXP(correction, correction);
    TMUL(maximum, denominator, correction);
    TSUB(sink, sink, final_max);
    TEXP(sink, sink);
    TADD(sink, maximum, sink);
    TDIV(sink, correction, sink);
    HeadColumn<H> multiplier;
    TASSIGN(multiplier, 196608);
    TROWEXPANDMUL(numerator, numerator, multiplier);
    Vec<bfloat16_t, H, DIM> result;
    TASSIGN(result, NUMERATOR_UB);
    TCVT(result, numerator, RoundMode::CAST_RINT, SaturationMode::OFF);
    if (single_part) set_intra_block(PIPE_V, 3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    Tile<TileType::Vec, bfloat16_t, H, DIM, BLayout::RowMajor, DYNAMIC, DIM> active_result(valid_heads);
    TASSIGN(active_result, NUMERATOR_UB);
    using OutputShape = TileShape2D<bfloat16_t, DYNAMIC, DIM>;
    GlobalTensor<bfloat16_t, OutputShape, BaseShape2D<bfloat16_t, H, DIM>> output_gm(
        data<bfloat16_t>(args, 0) + token * row_stride(args, 0) + head * DIM, OutputShape(valid_heads, DIM));
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(output_gm, active_result);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <int H, int K>
__aicore__ inline void attend_vector(__gm__ int64_t* args) {
    const int workers = get_block_num(args);
    const int block = get_block_idx(args);
    if (block >= args[ARG_TOKENS]) return;
    const int queries = (args[ARG_TOKENS] - block + workers - 1) / workers;
    const int parts = (args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS]) / K;
    const int tiles = queries * parts;
    if (tiles == 0) return;
    const bool single_part = parts == 1;
    bool fp4_active = false;
    set_intra_block(PIPE_V, 1);
    set_intra_block(PIPE_V, 3);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    gather_part(args, block, 0, get_sub_block_id(args), 0, fp4_active);
    for (int ordinal = 0; ordinal <= tiles; ++ordinal) {
        if (ordinal + 1 < tiles) {
            const int next = ordinal + 1;
            const int token = block + (next / parts) * workers;
            gather_part(args, token, next % parts, get_sub_block_id(args), next, fp4_active);
        }
        if (ordinal < tiles) {
            const int token = block + (ordinal / parts) * workers;
            update_softmax<H, K>(args, token, ordinal % parts, ordinal, (ordinal / parts) % 2);
        }
        if (ordinal > 0) {
            const int previous = ordinal - 1;
            const int part = previous % parts;
            Vec<float, H, DIM> numerator, weighted;
            Vec<float, 1, H> correction;
            TASSIGN(numerator, NUMERATOR_UB);
            TASSIGN(weighted, 65536);
            TASSIGN(correction, CORRECTION_UB + (previous % 2) * CORRECTION_STRIDE);
            wait_intra_block(PIPE_V, 2);
            if (part == 0) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                if (!single_part) TEXPANDS(numerator, 0.0f);
            }
            if (!single_part) {
                sparse_softmax::accumulate<decltype(numerator), decltype(correction)>(
                    numerator.data(), numerator.data(), weighted.data(), correction.data());
                set_intra_block(PIPE_V, 3);
            }
            if (part + 1 == parts) {
                const int token = block + (previous / parts) * workers;
                finalize_query<H>(args, token, (previous / parts) % 2, single_part);
            }
        }
    }
    drain_fp4_slots(fp4_active);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // Consume all returned and unused startup credits before either core retires.
    wait_intra_block(PIPE_MTE3, 5);
    wait_intra_block(PIPE_MTE3, 7);
    for (int slot = 0; slot < KV_SLOTS; ++slot) wait_intra_block(PIPE_MTE3, 9 + 2 * slot);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, FINISH_ACK);
    wait_intra_block(PIPE_V, FINISH_READY);
    pipe_barrier(PIPE_ALL);
}
#endif
#if defined(__CPU_SIM)
float decode_byte(uint8_t code, bool fp4) {
    if (fp4) {
        const float magnitude[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        return magnitude[code & 7] * (code & 8 ? -1.0f : 1.0f);
    }
    int exponent = (code >> 3) & 15;
    int mantissa = code & 7;
    float value = exponent == 0 ? mantissa / 512.0f : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
    if (exponent == 15 && mantissa == 7) value = std::numeric_limits<float>::quiet_NaN();
    return code & 128 ? -value : value;
}

void simulate_attention(__gm__ int64_t* args) {
    int heads = args[ARG_HEADS];
    int keys = args[ARG_WINDOW_KEYS] + args[ARG_COMPRESSED_KEYS];
    auto query = data<bfloat16_t>(args, 1);
    auto output = data<bfloat16_t>(args, 0);
    auto sink = data<float>(args, 2);
    int64_t sink_stride = row_stride(args, 2);
    for (int token = get_block_idx(args); token < args[ARG_TOKENS]; token += get_block_num(args)) {
        std::vector<float> numerator(heads * 512, 0.0f), maximum(heads, -1e30f), denominator(heads, 0.0f);
        std::vector<float> kv(128 * 512), scores(128);
        std::vector<bool> valid(128);
        for (int part = 0; part < keys; part += 128) {
            bool fp4 = part >= args[ARG_WINDOW_KEYS];
            int payload_arg = fp4 ? 5 : 3;
            int index_arg = fp4 ? 8 : 7;
            int width = fp4 ? 256 : 512;
            int groups = fp4 ? 32 : 16;
            auto payload = data<uint8_t>(args, payload_arg);
            auto scales = data<bfloat16_t>(args, payload_arg + 1);
            auto indices = data<int32_t>(args, index_arg) + token * row_stride(args, index_arg);
            for (int k = 0; k < 128; ++k) {
                int physical = indices[part + k - (fp4 ? args[ARG_WINDOW_KEYS] : 0)];
                valid[k] = physical >= 0;
                for (int d = 0; d < 512; ++d) {
                    float value = 0;
                    if (valid[k]) {
                        uint8_t code = payload[(int64_t)physical * width + (fp4 ? d / 2 : d)];
                        if (fp4) code = (code >> ((d % 2) * 4)) & 15;
                        float factor = static_cast<float>(scales[(int64_t)physical * groups + d / (fp4 ? 16 : 32)]);
                        value = static_cast<float>(static_cast<bfloat16_t>(decode_byte(code, fp4) * factor));
                    }
                    kv[k * 512 + d] = value;
                }
            }
            for (int h = 0; h < heads; ++h) {
                float next = maximum[h];
                for (int k = 0; k < 128; ++k) {
                    float dot = 0;
                    if (valid[k]) {
                        for (int d = 0; d < 512; ++d) {
                            dot += static_cast<float>(query[token * row_stride(args, 1) + h * 512 + d]) * kv[k * 512 + d];
                        }
                    }
                    scores[k] = valid[k] ? dot * 0.0441941731f : -1e30f;
                    next = std::max(next, scores[k]);
                }
                float correction = std::exp(maximum[h] - next);
                denominator[h] *= correction;
                for (int d = 0; d < 512; ++d) numerator[h * 512 + d] *= correction;
                for (int k = 0; k < 128; ++k) {
                    float probability = valid[k] ? std::exp(scores[k] - next) : 0;
                    denominator[h] += probability;
                    float rounded = static_cast<float>(static_cast<bfloat16_t>(probability));
                    for (int d = 0; d < 512; ++d) numerator[h * 512 + d] += rounded * kv[k * 512 + d];
                }
                maximum[h] = next;
            }
        }
        for (int h = 0; h < heads; ++h) {
            float sink_value = sink[h * sink_stride];
            float final_max = std::max(maximum[h], sink_value);
            float correction = std::exp(maximum[h] - final_max);
            float factor = correction / (denominator[h] * correction + std::exp(sink_value - final_max));
            for (int d = 0; d < 512; ++d) {
                output[token * row_stride(args, 0) + h * 512 + d] = static_cast<bfloat16_t>(numerator[h * 512 + d] * factor);
            }
        }
    }
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // ABI: O, Q, sink, W, Ws, C, Cs, Wi, Ci; tokens, heads, Wkeys, Ckeys.
#if defined(__CPU_SIM) && defined(__DAV_VEC__)
    if (get_sub_block_id(args) == 0) {
        simulate_attention(args);
    }
#endif
#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
    attend_cube<64, 128>(args);
#elif defined(__DAV_VEC__) && !defined(__CPU_SIM)
    set_mask_norm();
    set_vector_mask(-1, -1);
    attend_vector<32, 128>(args);
#endif
}
