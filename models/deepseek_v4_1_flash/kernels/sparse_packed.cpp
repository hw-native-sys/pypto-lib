// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Dual-AIV packed-cache gather and sink-aware online sparse attention.
#include "packed_cache.hpp"

using namespace packed_cache;

namespace {
using AttentionPipe = TPipe<0, Direction::DIR_BOTH, 65536, 1, 2, true>;
constexpr int DIM = 512;
constexpr int KEYS = 640;
constexpr int PIPE_BYTES = 65536;
constexpr int QUERY_L1 = 65536;
constexpr int KV_L1 = 98304;
constexpr int MAXIMUM_UB = 173696;
constexpr int DENOMINATOR_UB = 173824;
constexpr int NUMERATOR_UB = 173952;
constexpr int WORK_UB = 108160;
constexpr int REDUCE_UB = 66816;
constexpr int NEXT_MAX_UB = 107776;
constexpr int CORRECTION_UB = 107904;
constexpr int ROW_SUM_UB = 108032;

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

#if defined(__DAV_VEC__)
__aicore__ inline void gather_token(__gm__ int64_t* args, int token, int lane) {
    // Gather scratch aliases the drained attention scratch, including its FIFO.
    pipe_barrier(PIPE_ALL);
    Vec<int32_t, 1, 32> ids;
    Vec<bfloat16_t, 32, DIM> decoded;
    Vec<bfloat16_t, 1, DIM> zero;
    TASSIGN(ids, 135168);
    TASSIGN(decoded, 32768);
    TASSIGN(zero, 135424);
    TEXPANDS(zero, (bfloat16_t)0.0f);
    for (int column = lane * 32; column < KEYS; column += 64) {
        int source = column < 128 ? 9 : 10;
        int source_column = column < 128 ? column : column - 128;
        GM<int32_t, 1, 32> source_ids(
            data<int32_t>(args, source) + token * row_stride(args, source) + source_column);
        TLOAD(ids, source_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        if (column < 128) {
            decode_window<EVENT_ID7>(data<float8_e4m3_t>(args, 5), data<bfloat16_t>(args, 6), ids, decoded);
        } else {
            decode_compressed<EVENT_ID7>(data<uint8_t>(args, 7), data<bfloat16_t>(args, 8), ids, decoded);
        }
        for (int row = 0; row < 32; ++row) {
            if (ids.GetValue(row) < 0) TINSERT(decoded, zero, row, 0);
        }
        GM<int32_t, 1, 32> combined(data<int32_t>(args, 3) + token * row_stride(args, 3) + column);
        GM<bfloat16_t, 32, DIM> selected(
            data<bfloat16_t>(args, 2) + token * row_stride(args, 2) + column * DIM);
        // Event 7 is separate from the online consumer's primed local flags.
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        TSTORE(combined, ids);
        TSTORE(selected, decoded);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    }
    set_intra_block(PIPE_MTE3, 8);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_CUBE__)
template <int H, int K>
__aicore__ inline void attend_cube(__gm__ int64_t* args) {
    constexpr int QK_STEP = K == 320 ? 48 : 128;
    constexpr int QK_END = DIM / (2 * QK_STEP) * (2 * QK_STEP);
    constexpr int LEFT_SECOND = H * QK_STEP * 2;
    AttentionPipe pipe(nullptr, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    int heads = args[12];
    int valid_heads = heads < H ? heads : H;
    for (int token = get_block_idx(args); token < args[11]; token += get_block_num(args)) {
        // AIV1's intra-block flags use the Cube-side +16 flag bank.
        wait_intra_block(PIPE_MTE2, 8);
        wait_intra_block(PIPE_MTE2, 24);
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
            for (int part = 0; part < KEYS / K; ++part) {
                Mat<K, DIM> kv;
                TransposedMat<DIM, K> key;
                Acc<H, K> scores;
                TASSIGN(kv, KV_L1);
                TASSIGN(key, KV_L1);
                TASSIGN(scores, 0);
                GM<bfloat16_t, K, DIM> kv_gm(
                    data<bfloat16_t>(args, 2) + token * row_stride(args, 2) + part * K * DIM);
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                TLOAD(kv, kv_gm);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
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
                TPUSH<AttentionPipe, Acc<H, K>, TileSplitAxis::TILE_NO_SPLIT>(pipe, scores);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                Mat<H, K> weights;
                Acc<H, DIM> weighted;
                TASSIGN(weighted, 0);
                TPOP<AttentionPipe, Mat<H, K>, TileSplitAxis::TILE_NO_SPLIT>(pipe, weights);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                for (int offset = 0; offset < K; offset += 64) {
                    Left<H, 32> p0, p1;
                    Right<32, DIM> v0, v1;
                    TASSIGN(p0, LEFT_SECOND);
                    TASSIGN(p1, 0);
                    TASSIGN(v0, 0);
                    TASSIGN(v1, 32768);
                    pipe_barrier(PIPE_MTE1);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
                    TEXTRACT(p0, weights, 0, offset);
                    TEXTRACT(v0, kv, offset, 0);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                    TEXTRACT(p1, weights, 0, offset + 32);
                    TEXTRACT(v1, kv, offset + 32, 0);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
                    pipe_barrier(PIPE_M);
                    if (offset == 0) TMATMUL(weighted, p0, v0);
                    else TMATMUL_ACC(weighted, weighted, p0, v0);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
                    pipe_barrier(PIPE_M);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
                    TMATMUL_ACC(weighted, weighted, p1, v1);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
                }
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
                TFREE<AttentionPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                TPUSH<AttentionPipe, Acc<H, DIM>, TileSplitAxis::TILE_NO_SPLIT>(pipe, weighted);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            }
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        }
        set_intra_block(PIPE_MTE1, 9);
        set_intra_block(PIPE_MTE1, 25);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_VEC__)
template <int H, int K>
__aicore__ inline void attend_vector(__gm__ int64_t* args) {
    AttentionPipe pipe(nullptr, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    int heads = args[12];
    int valid_heads = heads < H ? heads : H;
    for (int token = get_block_idx(args); token < args[11]; token += get_block_num(args)) {
        gather_token(args, token, 0);
        for (int head = 0; head < heads; head += H) {
            Vec<float, 1, H> maximum, denominator;
            Vec<float, H, DIM> numerator;
            TASSIGN(maximum, MAXIMUM_UB);
            TASSIGN(denominator, DENOMINATOR_UB);
            TASSIGN(numerator, NUMERATOR_UB);
            TEXPANDS(maximum, -1e30f);
            TEXPANDS(denominator, 0.0f);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            TEXPANDS(numerator, 0.0f);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            for (int part = 0; part < KEYS / K; ++part) {
                Vec<float, H, K> qk, scores, reduce;
                Vec<int32_t, 1, K> ids;
                Vec<float, 1, K> penalty, valid;
                Vec<float, 1, H> next_max, correction, previous_denominator, part_sum;
                HeadColumn<H> max_column, correction_column, sum_column;
                TASSIGN(scores, WORK_UB);
                TASSIGN(reduce, REDUCE_UB);
                TASSIGN(ids, REDUCE_UB);
                TASSIGN(penalty, REDUCE_UB);
                TASSIGN(valid, PIPE_BYTES);
                TASSIGN(next_max, NEXT_MAX_UB);
                TASSIGN(max_column, NEXT_MAX_UB);
                TASSIGN(correction, CORRECTION_UB);
                TASSIGN(correction_column, CORRECTION_UB);
                TASSIGN(previous_denominator, PIPE_BYTES);
                TASSIGN(part_sum, ROW_SUM_UB);
                TASSIGN(sum_column, ROW_SUM_UB);
                TPOP<AttentionPipe, Vec<float, H, K>, TileSplitAxis::TILE_NO_SPLIT>(pipe, qk);
                TMULS(scores, qk, 0.0441941731f);
                TFREE<AttentionPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
                GM<int32_t, 1, K> ids_gm(data<int32_t>(args, 3) + token * row_stride(args, 3) + part * K);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                TLOAD(ids, ids_gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                TCVT(penalty, ids, RoundMode::CAST_ROUND, SaturationMode::OFF);
                TADDS(penalty, penalty, 1.0f);
                TMAXS(penalty, penalty, 0.0f);
                TMINS(valid, penalty, 1.0f);
                TSUBS(penalty, valid, 1.0f);
                TMULS(penalty, penalty, 1e30f);
                TCOLEXPANDADD(scores, scores, penalty);
                TROWMAX(max_column, scores, reduce);
                TMAX(next_max, maximum, next_max);
                Vec<float, 1, H> delta;
                TASSIGN(delta, REDUCE_UB);
                pipe_barrier(PIPE_ALL);
                TSUB(delta, maximum, next_max);
                TEXP(correction, delta);
                TROWEXPANDSUB(scores, scores, max_column);
                TEXP(scores, scores);
                TCOLEXPANDMUL(scores, scores, valid);
                pipe_barrier(PIPE_ALL);
                TMUL(previous_denominator, denominator, correction);
                TROWSUM(sum_column, scores, reduce);
                TADD(denominator, previous_denominator, part_sum);
                Vec<bfloat16_t, H, K> weights;
                Tile<TileType::Vec, bfloat16_t, H, K, BLayout::ColMajor, H, K, SLayout::RowMajor> weights_nz;
                TASSIGN(weights, WORK_UB);
                TASSIGN(weights_nz, REDUCE_UB);
                TCVT(weights, scores, RoundMode::CAST_RINT, SaturationMode::OFF);
                pipe_barrier(PIPE_ALL);
                TMOV(weights_nz, weights);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TPUSH<AttentionPipe, decltype(weights_nz), TileSplitAxis::TILE_NO_SPLIT>(pipe, weights_nz);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                Vec<float, H, DIM> previous_numerator, weighted;
                TASSIGN(previous_numerator, WORK_UB);
                TROWEXPANDMUL(previous_numerator, numerator, correction_column);
                TPOP<AttentionPipe, Vec<float, H, DIM>, TileSplitAxis::TILE_NO_SPLIT>(pipe, weighted);
                TADD(numerator, previous_numerator, weighted);
                TFREE<AttentionPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
                TMOV(maximum, next_max);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            Vec<float, 1, H> sink, final_max, correction;
            TASSIGN(sink, WORK_UB);
            TASSIGN(final_max, REDUCE_UB);
            TASSIGN(correction, PIPE_BYTES);
            Tile<TileType::Vec, float, 1, H, BLayout::RowMajor, 1, DYNAMIC> sink_load(valid_heads);
            TASSIGN(sink_load, WORK_UB);
            using SinkShape = TileShape2D<float, 1, DYNAMIC>;
            GlobalTensor<float, SinkShape, BaseShape2D<float, 1, H>> sink_gm(
                data<float>(args, 4) + head, SinkShape(1, valid_heads));
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            TLOAD(sink_load, sink_gm);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
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
            TASSIGN(multiplier, WORK_UB);
            TROWEXPANDMUL(numerator, numerator, multiplier);
            Vec<bfloat16_t, H, DIM> result;
            TASSIGN(result, NUMERATOR_UB);
            TCVT(result, numerator, RoundMode::CAST_RINT, SaturationMode::OFF);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            // Padded Cube rows for eight-head shards are never published.
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
        wait_intra_block(PIPE_MTE3, 9);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // ABI: output, query, selected, indices, sink, W/Wscale, C/Cscale, Wi/Ci; tokens, heads.
#if defined(__DAV_CUBE__)
    if (args[12] == 64) attend_cube<32, 320>(args);
    else if (args[12] == 32) attend_cube<32, 128>(args);
    else attend_cube<16, 128>(args);
#elif defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    if (get_sub_block_id(args) == 1) {
        for (int token = get_block_idx(args); token < args[11]; token += get_block_num(args)) {
            gather_token(args, token, 1);
            wait_intra_block(PIPE_MTE3, 9);
        }
        pipe_barrier(PIPE_ALL);
    } else if (args[12] == 64) attend_vector<32, 320>(args);
    else if (args[12] == 32) attend_vector<32, 128>(args);
    else attend_vector<16, 128>(args);
#endif
}
