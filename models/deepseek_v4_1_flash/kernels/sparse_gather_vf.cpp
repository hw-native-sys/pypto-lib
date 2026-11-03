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
#include <type_traits>
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


#if !defined(__CPU_SIM)
// Consume explicit startup credits on older local split pipes: PTO-ISA issue 390.
// https://github.com/hw-native-sys/pto-isa/issues/390
template <typename Pipe, typename = void>
struct HasLocalCreditMethod : std::false_type {};

template <typename Pipe>
struct HasLocalCreditMethod<Pipe, std::void_t<decltype(Pipe::usesLocalNoSplitProtocol())>> : std::true_type {};

template <typename Pipe, typename = void>
struct HasLocalCreditFlag : std::false_type {};

template <typename Pipe>
struct HasLocalCreditFlag<Pipe, std::void_t<decltype(Pipe::uses_local_no_split_credit_protocol)>> : std::true_type {};

template <typename Pipe>
__aicore__ inline bool wait_for_local_credit(uint32_t tile_index) {
    if constexpr (HasLocalCreditMethod<Pipe>::value) {
        if (!Pipe::usesLocalNoSplitProtocol()) return true;
    } else if constexpr (HasLocalCreditFlag<Pipe>::value) {
        if (!Pipe::uses_local_no_split_credit_protocol) return true;
    }
    return Pipe::shouldWaitFree(tile_index);
}

struct AttentionPipe : TPipe<0, Direction::DIR_BOTH, 65536, 1, 2, false> {
    using Base = TPipe<0, Direction::DIR_BOTH, 65536, 1, 2, false>;
    using Base::Base;

    __aicore__ inline static bool shouldWaitFree(uint32_t tile_index) {
        return wait_for_local_credit<Base>(tile_index);
    }
};
#endif
constexpr int DIM = 512;
constexpr int PIPE_BYTES = 65536;
constexpr int QUERY_L1 = 65536;
constexpr int KV_L1 = 131072;
constexpr int MAXIMUM_UB = 205056;
constexpr int DENOMINATOR_UB = 205184;
constexpr int NUMERATOR_UB = 131072;
constexpr int WORK_UB = 65536;
constexpr int REDUCE_UB = 81920;
constexpr int NEXT_MAX_UB = 205312;
constexpr int CORRECTION_UB = 205440;
constexpr int ROW_SUM_UB = 205568;

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
template <int Slot>
__aicore__ inline void gather_fp4_subtile(
    __gm__ int64_t* args, int index_offset, int destination_row, Mat<128, DIM>& kv) {
    constexpr int payload_address = Slot * 8192;
    constexpr int scale_address = 16384 + Slot * 1024;
    constexpr int factor_address = 18432 + Slot * 2048;
    int id_address = 24576 + index_offset * sizeof(int32_t);
    constexpr int nz_address = 32768 + Slot * 16384;
    Vec<int32_t, 1, 16> ids;
    Vec<float4_e2m1x2_t, 16, DIM> payload;
    Vec<bfloat16_t, 16, 32> scales;
    Vec<float, 16, 32> factors;
    Tile<TileType::Vec, bfloat16_t, 16, DIM, BLayout::ColMajor, 16, DIM, SLayout::RowMajor> nz;
    TASSIGN(ids, id_address);
    TASSIGN(payload, payload_address);
    TASSIGN(scales, scale_address);
    TASSIGN(factors, factor_address);
    TASSIGN(nz, nz_address);
    auto payload_base = data<uint8_t>(args, 5);
    auto scale_base = data<bfloat16_t>(args, 6);

    // Payload/scales are reusable after VF consumes this slot. The 64
    // indices remain immutable throughout the gather; NZ has its own credit.
    wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(5 + Slot));
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
    TCVT(factors, scales, RoundMode::CAST_RINT);
    wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(5 + Slot));
    sparse_dequant::decode_nz<true, decltype(nz), decltype(payload), decltype(factors), decltype(ids)>(
        nz.data(), payload.data(), factors.data(), ids.data());
    set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(5 + Slot));
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(5 + Slot));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(5 + Slot));
    TINSERT(kv, nz, destination_row, 0);
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(5 + Slot));
}

__aicore__ inline void gather_part(__gm__ int64_t* args, int token, int part, int lane, int ordinal) {
    if (part * 128 >= args[11]) {
        Mat<128, DIM> kv;
        TASSIGN(kv, KV_L1 + (ordinal % 2) * 131072);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        int source_column = part * 128 + lane * 64 - args[11];
        Vec<int32_t, 1, 64> all_ids;
        TASSIGN(all_ids, 24576);
        GM<int32_t, 1, 64> source_ids(
            data<int32_t>(args, 8) + token * row_stride(args, 8) + source_column);
        TLOAD(all_ids, source_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        gather_fp4_subtile<0>(args, 0, lane * 64, kv);
        gather_fp4_subtile<1>(args, 16, lane * 64 + 16, kv);
        gather_fp4_subtile<0>(args, 32, lane * 64 + 32, kv);
        gather_fp4_subtile<1>(args, 48, lane * 64 + 48, kv);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        set_intra_block(PIPE_MTE3, 8);
        return;
    }
    Vec<int32_t, 1, 32> ids;
    Tile<TileType::Vec, bfloat16_t, 32, DIM, BLayout::ColMajor, 32, DIM, SLayout::RowMajor> nz;
    Mat<128, DIM> kv;
    TASSIGN(ids, 28672);
    TASSIGN(nz, 32768);
    TASSIGN(kv, KV_L1 + (ordinal % 2) * 131072);
    for (int sub = 0; sub < 2; ++sub) {
        int column = part * 128 + lane * 64 + sub * 32;
        int source = column < args[11] ? 7 : 8;
        int source_column = column < args[11] ? column : column - args[11];
        GM<int32_t, 1, 32> source_ids(
            data<int32_t>(args, source) + token * row_stride(args, source) + source_column);
        TLOAD(ids, source_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        if (column < args[11]) {
            Vec<float8_e4m3_t, 32, DIM> payload;
            Vec<bfloat16_t, 32, 16> scales;
            Vec<float, 32, 16> factors;
            TASSIGN(payload, 0);
            TASSIGN(scales, 16384);
            TASSIGN(factors, 24576);
            load_rows<EVENT_ID7, float8_e4m3_t, 512, 16>(
                data<float8_e4m3_t>(args, 3), data<bfloat16_t>(args, 4), ids);
            TCVT(factors, scales, RoundMode::CAST_RINT);
            sparse_dequant::decode_nz<false, decltype(nz), decltype(payload), decltype(factors), decltype(ids)>(
                nz.data(), payload.data(), factors.data(), ids.data());
        } else {
            Vec<float4_e2m1x2_t, 32, DIM> payload;
            Vec<bfloat16_t, 32, 32> scales;
            Vec<float, 32, 32> factors;
            TASSIGN(payload, 0);
            TASSIGN(scales, 16384);
            TASSIGN(factors, 24576);
            load_rows<EVENT_ID7, uint8_t, 256, 32>(data<uint8_t>(args, 5), data<bfloat16_t>(args, 6), ids);
            TCVT(factors, scales, RoundMode::CAST_RINT);
            sparse_dequant::decode_nz<true, decltype(nz), decltype(payload), decltype(factors), decltype(ids)>(
                nz.data(), payload.data(), factors.data(), ids.data());
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        TINSERT(kv, nz, lane * 64 + sub * 32, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    }
    set_intra_block(PIPE_MTE3, 8);
}
#endif

#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
template <int H, int K>
__aicore__ inline void attend_cube(__gm__ int64_t* args) {
    constexpr int QK_STEP = K == 320 ? 48 : 128;
    constexpr int QK_END = DIM / (2 * QK_STEP) * (2 * QK_STEP);
    constexpr int LEFT_SECOND = H * QK_STEP * 2;
    AttentionPipe pipe(nullptr, 65536, 0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    int heads = args[10];
    int valid_heads = heads < H ? heads : H;
    int parts = (args[11] + args[12]) / K;
    int tile_base = 0;
    for (int token = get_block_idx(args); token < args[9]; token += get_block_num(args), tile_base += parts) {
        for (int head = 0; head < heads; head += H) {
            Tile<TileType::Mat, bfloat16_t, H, DIM, BLayout::ColMajor,
                 DYNAMIC, DIM, SLayout::RowMajor, 512, PadValue::Zero> query(valid_heads);
            TASSIGN(query, QUERY_L1);
            using QueryShape = TileShape2D<bfloat16_t, DYNAMIC, DIM>;
            GlobalTensor<bfloat16_t, QueryShape, BaseShape2D<bfloat16_t, H, DIM>> query_gm(
                data<bfloat16_t>(args, 1) + token * row_stride(args, 1) + head * DIM,
                QueryShape(valid_heads, DIM));
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(query, query_gm);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            for (int part = 0; part < (args[11] + args[12]) / K; ++part) {
                Mat<K, DIM> kv;
                TransposedMat<DIM, K> key;
                Acc<H, K> scores;
                TASSIGN(kv, KV_L1 + ((tile_base + part) % 2) * 131072);
                TASSIGN(key, KV_L1 + ((tile_base + part) % 2) * 131072);
                TASSIGN(scores, 0);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_intra_block(PIPE_MTE1, 8);
                wait_intra_block(PIPE_MTE1, 24);
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
                if constexpr (QK_END < DIM) {
                    Left<H, DIM - QK_END> qtail;
                    Right<DIM - QK_END, K> ktail;
                    TASSIGN(qtail, LEFT_SECOND);
                    TASSIGN(ktail, 0);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
                    pipe_barrier(PIPE_MTE1);
                    TEXTRACT(qtail, query, 0, QK_END);
                    TEXTRACT(ktail, key, QK_END, 0);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    pipe_barrier(PIPE_M);
                    TMATMUL_ACC(scores, scores, qtail, ktail);
                }
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                TPUSH<AttentionPipe, Acc<H, K>, TileSplitAxis::TILE_UP_DOWN>(pipe, scores);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                Mat<H, K> weights;
                Acc<H, DIM> weighted;
                TASSIGN(weighted, 0);
                TPOP<AttentionPipe, Mat<H, K>, TileSplitAxis::TILE_UP_DOWN>(pipe, weights);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                Left<H, K> p;
                TASSIGN(p, 0);
                TEXTRACT(p, weights, 0, 0);
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
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

                TFREE<AttentionPipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
                set_intra_block(PIPE_MTE1, 9);
                set_intra_block(PIPE_MTE1, 25);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                TPUSH<AttentionPipe, Acc<H, DIM>, TileSplitAxis::TILE_UP_DOWN>(pipe, weighted);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            }
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        }
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <int H, int K>
__aicore__ inline void attend_vector(__gm__ int64_t* args) {
    AttentionPipe pipe(nullptr, 65536, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    int heads = args[10];
    int valid_heads = heads < H ? heads : H;
    bool single_part = args[11] + args[12] == K;
    int parts = (args[11] + args[12]) / K;
    int tile_base = 0;
    for (int token = get_block_idx(args); token < args[9]; token += get_block_num(args), tile_base += parts) {
        int head = get_sub_block_id(args) * H;
        {
            Vec<float, 1, H> maximum, denominator;
            Vec<float, H, DIM> numerator;
            TASSIGN(maximum, MAXIMUM_UB);
            TASSIGN(denominator, DENOMINATOR_UB);
            TASSIGN(numerator, single_part ? 65536 : NUMERATOR_UB);
            TEXPANDS(maximum, -1e30f);
            TEXPANDS(denominator, 0.0f);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            if (!single_part) TEXPANDS(numerator, 0.0f);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            if (tile_base == 0) gather_part(args, token, 0, get_sub_block_id(args), 0);
            for (int part = 0; part < (args[11] + args[12]) / K; ++part) {
                Vec<float, H, K> qk, scores, reduce;
                Vec<int32_t, 1, K> ids;
                Vec<float, 1, K> penalty, valid;
                Vec<float, 1, H> next_max, correction, previous_denominator, part_sum;
                HeadColumn<H> max_column, correction_column, sum_column;
                TASSIGN(scores, WORK_UB);
                TASSIGN(reduce, REDUCE_UB);
                TASSIGN(ids, 217088);
                TASSIGN(penalty, REDUCE_UB);
                TASSIGN(valid, 136448);
                TASSIGN(next_max, NEXT_MAX_UB);
                TASSIGN(max_column, NEXT_MAX_UB);
                TASSIGN(correction, CORRECTION_UB);
                TASSIGN(correction_column, CORRECTION_UB);
                TASSIGN(previous_denominator, 205696);
                TASSIGN(part_sum, ROW_SUM_UB);
                TASSIGN(sum_column, ROW_SUM_UB);
                TPOP<AttentionPipe, Vec<float, H, K>, TileSplitAxis::TILE_UP_DOWN>(pipe, qk);
                Tile<TileType::Vec, bfloat16_t, H, K, BLayout::ColMajor, H, K, SLayout::RowMajor> weights_nz;
                TASSIGN(weights_nz, 208896);

                int source = part * K < args[11] ? 7 : 8;
                int source_column = part * K < args[11] ? part * K : part * K - args[11];
                GM<int32_t, 1, K> ids_gm(
                    data<int32_t>(args, source) + token * row_stride(args, source) + source_column);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                TLOAD(ids, ids_gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                sparse_softmax::update<decltype(weights_nz), decltype(qk), decltype(ids), decltype(maximum)>(
                    weights_nz.data(), next_max.data(), denominator.data(), correction.data(),
                    qk.data(), ids.data(), maximum.data(), denominator.data());
                TFREE<AttentionPipe, TileSplitAxis::TILE_UP_DOWN>(pipe);

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TPUSH<AttentionPipe, decltype(weights_nz), TileSplitAxis::TILE_UP_DOWN>(pipe, weights_nz);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
                if (part + 1 < parts) {
                    gather_part(args, token, part + 1, get_sub_block_id(args), tile_base + part + 1);
                } else if (token + get_block_num(args) < args[9]) {
                    gather_part(args, token + get_block_num(args), 0, get_sub_block_id(args), tile_base + parts);
                }
                Vec<float, H, DIM> previous_numerator, weighted;
                TASSIGN(previous_numerator, WORK_UB);

                TPOP<AttentionPipe, Vec<float, H, DIM>, TileSplitAxis::TILE_UP_DOWN>(pipe, weighted);
                if (!single_part) {
                    sparse_softmax::accumulate<decltype(numerator), decltype(correction)>(
                        numerator.data(), numerator.data(), weighted.data(), correction.data());
                    TFREE<AttentionPipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
                }
                TMOV(maximum, next_max);
                wait_intra_block(PIPE_MTE3, 9);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            Vec<float, 1, H> sink, final_max, correction;
            TASSIGN(sink, 196608);
            TASSIGN(final_max, 196736);
            TASSIGN(correction, CORRECTION_UB);
            Tile<TileType::Vec, float, 1, H, BLayout::RowMajor, 1, DYNAMIC> sink_load(valid_heads);
            TASSIGN(sink_load, 196608);
            using SinkShape = TileShape2D<float, 1, DYNAMIC>;
            auto sink_data = data<float>(args, 2);
            int64_t sink_stride = row_stride(args, 2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            if (sink_stride == 1) {
                GlobalTensor<float, SinkShape, BaseShape2D<float, 1, H>> sink_gm(
                    sink_data + head, SinkShape(1, valid_heads));
                TLOAD(sink_load, sink_gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            } else {
                set_flag(PIPE_V, PIPE_S, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
                for (int lane = 0; lane < H; ++lane) {
                    float value = lane < valid_heads ? sink_data[(head + lane) * sink_stride] : 0.0f;
                    sink.SetValue(lane, value);
                }
                set_flag(PIPE_S, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            }
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            TMAX(final_max, maximum, sink);
            TSUB(correction, maximum, final_max);
            TEXP(correction, correction);
            TMUL(maximum, denominator, correction);
            TSUB(sink, sink, final_max);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            TEXP(sink, sink);
            TADD(sink, maximum, sink);
            TDIV(sink, correction, sink);
            HeadColumn<H> multiplier;
            TASSIGN(multiplier, 196608);
            TROWEXPANDMUL(numerator, numerator, multiplier);
            Vec<bfloat16_t, H, DIM> result;
            TASSIGN(result, NUMERATOR_UB);
            TCVT(result, numerator, RoundMode::CAST_RINT, SaturationMode::OFF);
            if (single_part) TFREE<AttentionPipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            // Padded Cube rows outside the active head shard are never published.
            Tile<TileType::Vec, bfloat16_t, H, DIM, BLayout::RowMajor, DYNAMIC, DIM> active_result(valid_heads);
            TASSIGN(active_result, NUMERATOR_UB);
            using OutputShape = TileShape2D<bfloat16_t, DYNAMIC, DIM>;
            GlobalTensor<bfloat16_t, OutputShape, BaseShape2D<bfloat16_t, H, DIM>> output_gm(
                data<bfloat16_t>(args, 0) + token * row_stride(args, 0) + head * DIM,
                OutputShape(valid_heads, DIM));
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(output_gm, active_result);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
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
    int heads = args[10];
    int keys = args[11] + args[12];
    auto query = data<bfloat16_t>(args, 1);
    auto output = data<bfloat16_t>(args, 0);
    auto sink = data<float>(args, 2);
    int64_t sink_stride = row_stride(args, 2);
    for (int token = get_block_idx(args); token < args[9]; token += get_block_num(args)) {
        std::vector<float> numerator(heads * 512, 0.0f), maximum(heads, -1e30f), denominator(heads, 0.0f);
        std::vector<float> kv(128 * 512), scores(128);
        std::vector<bool> valid(128);
        for (int part = 0; part < keys; part += 128) {
            bool fp4 = part >= args[11];
            int payload_arg = fp4 ? 5 : 3;
            int index_arg = fp4 ? 8 : 7;
            int width = fp4 ? 256 : 512;
            int groups = fp4 ? 32 : 16;
            auto payload = data<uint8_t>(args, payload_arg);
            auto scales = data<bfloat16_t>(args, payload_arg + 1);
            auto indices = data<int32_t>(args, index_arg) + token * row_stride(args, index_arg);
            for (int k = 0; k < 128; ++k) {
                int physical = indices[part + k - (fp4 ? args[11] : 0)];
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
    if (get_sub_block_id(args) == 0) simulate_attention(args);
#endif
#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
    attend_cube<64, 128>(args);
#elif defined(__DAV_VEC__) && !defined(__CPU_SIM)
    set_mask_norm();
    set_vector_mask(-1, -1);
    attend_vector<32, 128>(args);
#endif
}
