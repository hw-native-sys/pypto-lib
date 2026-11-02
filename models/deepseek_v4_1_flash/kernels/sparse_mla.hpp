// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed-cache gather, BF16 QK/PV and sink-aware online sparse attention.
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "packed_cache.hpp"

using namespace pto;

namespace sparse_mla {
constexpr int D_TILE = 512;
constexpr int KV_TILE = 128;
constexpr int K_TILE = 128;
constexpr int OUTPUT_TILE = 64;
constexpr int KV_SLOT_BYTES = KV_TILE * D_TILE * sizeof(bfloat16_t);
constexpr int Q_L1 = 3 * KV_SLOT_BYTES;
constexpr int P_L1 = Q_L1 + 64 * D_TILE * sizeof(bfloat16_t);
constexpr float SCALE = 0.04419417382415922f;

using KvPipe = TPipe<0, Direction::DIR_V2C, KV_SLOT_BYTES, 3>;
using KvMat = Tile<TileType::Mat, bfloat16_t, KV_TILE, D_TILE, BLayout::ColMajor,
                   KV_TILE, D_TILE, SLayout::RowMajor>;
using KeyMat = Tile<TileType::Mat, bfloat16_t, D_TILE, KV_TILE, BLayout::RowMajor,
                    D_TILE, KV_TILE, SLayout::ColMajor>;
using KvVec = Tile<TileType::Vec, bfloat16_t, KV_TILE / 2, D_TILE, BLayout::ColMajor,
                   KV_TILE / 2, D_TILE, SLayout::RowMajor>;

template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename Pipe, typename TileData>
__aicore__ inline void pipe_push(Pipe& pipe, TileData& tile, int subblock) {
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    TPUSH<Pipe, TileData, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
#else
    TPUSH<Pipe, TileData, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, subblock);
#endif
}

template <typename Pipe, typename TileData>
__aicore__ inline void pipe_pop(Pipe& pipe, TileData& tile, int subblock) {
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    TPOP<Pipe, TileData, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
#else
    TPOP<Pipe, TileData, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, subblock);
#endif
}

template <int H>
struct Tiles {
    static constexpr int M = H < 32 ? 32 : H;
    static constexpr int V = M / 2;
    using Query = Tile<TileType::Mat, bfloat16_t, M, D_TILE, BLayout::ColMajor,
                       DYNAMIC, D_TILE, SLayout::RowMajor>;
    using Probability = Tile<TileType::Mat, bfloat16_t, M, KV_TILE, BLayout::ColMajor,
                             M, KV_TILE, SLayout::RowMajor>;
    using PVec = Tile<TileType::Vec, bfloat16_t, V, KV_TILE, BLayout::ColMajor,
                      V, KV_TILE, SLayout::RowMajor>;
    using State = Vec<float, 1, V>;
    using Column = Tile<TileType::Vec, float, V, 1, BLayout::ColMajor>;
    using ScorePipe = TPipe<2, Direction::DIR_C2V, V * KV_TILE * sizeof(float), 1>;
    using ProbPipe = TPipe<4, Direction::DIR_V2C, M * KV_TILE * sizeof(bfloat16_t), 1>;
    using OutputPipe = TPipe<6, Direction::DIR_C2V, V * OUTPUT_TILE * sizeof(float), 2>;
};

#if defined(__DAV_CUBE__)
template <int H>
__aicore__ inline void cube_query(__gm__ int64_t* args, int token, typename Tiles<H>::Query& q) {
    auto input = reinterpret_cast<__gm__ Tensor*>(args[0]);
    using Shape = TileShape2D<bfloat16_t, H, D_TILE>;
    using Stride = BaseShape2D<bfloat16_t, H, D_TILE>;
    GlobalTensor<bfloat16_t, Shape, Stride> gm(data<bfloat16_t>(args, 0) + token * input->strides[0]);
    TLOAD(q, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
}

template <int H>
__aicore__ inline void cube_qk(
    typename Tiles<H>::Query& q, KvMat& kv, typename Tiles<H>::ScorePipe& scores) {
    constexpr int M = Tiles<H>::M;
    KeyMat key;
    TASSIGN(key, reinterpret_cast<uint64_t>(kv.data()));
    TileLeft<bfloat16_t, M, K_TILE> left;
    TileRight<bfloat16_t, K_TILE, KV_TILE> right;
    TileAcc<float, M, KV_TILE> dot;
    TASSIGN(left, 0);
    TASSIGN(right, 0);
    TASSIGN(dot, 0);
    for (int k = 0; k < D_TILE; k += K_TILE) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(left, q, 0, k);
        TEXTRACT(right, key, k, 0);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (k == 0) TMATMUL(dot, left, right);
        else TMATMUL_ACC(dot, dot, left, right);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_push(scores, dot, 0);
}

template <int H>
__aicore__ inline void cube_pv(
    typename Tiles<H>::Probability& p, KvMat& kv, typename Tiles<H>::OutputPipe& output) {
    constexpr int M = Tiles<H>::M;
    TileLeft<bfloat16_t, M, KV_TILE> left;
    TileRight<bfloat16_t, KV_TILE, OUTPUT_TILE> right;
    TileAcc<float, M, OUTPUT_TILE> weighted;
    TASSIGN(left, 0);
    TASSIGN(right, 0);
    TASSIGN(weighted, 65536);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(left, p, 0, 0);
    for (int column = 0; column < D_TILE; column += OUTPUT_TILE) {
        TEXTRACT(right, kv, 0, column);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(weighted, left, right);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_push(output, weighted, 0);
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    }
}
#endif

#if defined(__DAV_VEC__)
__aicore__ inline void gather_kv(__gm__ int64_t* args, int token, int part, KvPipe& pipe) {
    int source = part == 0 ? 5 : 6;
    int column = part == 0 ? 0 : (part - 1) * KV_TILE;
    auto index_tensor = reinterpret_cast<__gm__ Tensor*>(args[source]);
    Vec<int32_t, 1, KV_TILE> indices;
    TASSIGN(indices, 22528);
    using IndexGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, KV_TILE>, BaseShape2D<int32_t, 1, KV_TILE>>;
    IndexGM input(data<int32_t>(args, source) + token * index_tensor->strides[0] + column);
    TLOAD(indices, input);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    for (int chunk = 0; chunk < 2; ++chunk) {
        Vec<int32_t, 1, 32> ids;
        Vec<bfloat16_t, 32, D_TILE> decoded;
        KvVec nz;
        int start = get_sub_block_id(args) * 64 + chunk * 32;
        TASSIGN(ids, 22528 + start * sizeof(int32_t));
        TASSIGN(decoded, 32768);
        TASSIGN(nz, 131072 + chunk * 32 * 16 * sizeof(bfloat16_t));
        if (part == 0) {
            packed_cache::decode_window(data<float8_e4m3_t>(args, 1), data<bfloat16_t>(args, 2), ids, decoded);
        } else {
            packed_cache::decode_compressed(data<uint8_t>(args, 3), data<bfloat16_t>(args, 4), ids, decoded);
        }
        for (int row = 0; row < 32; ++row) {
            if (ids.GetValue(row) < 0) {
                Vec<bfloat16_t, 1, D_TILE> zero;
                TASSIGN(zero, 32768 + row * D_TILE * sizeof(bfloat16_t));
                TEXPANDS(zero, (bfloat16_t)0.0f);
            }
        }
        // A 32-row source fills one half of the 64-row NZ pitch.
        TMOV(nz, decoded);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    KvVec kv;
    TASSIGN(kv, 131072);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_push(pipe, kv, get_sub_block_id(args));
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <int H>
__aicore__ inline void softmax(
    typename Tiles<H>::ScorePipe& scores, typename Tiles<H>::ProbPipe& probabilities,
    typename Tiles<H>::State& maximum, typename Tiles<H>::State& denominator,
    typename Tiles<H>::State& correction, int subblock) {
    constexpr int V = Tiles<H>::V;
    Vec<float, V, KV_TILE> score, temporary;
    Vec<float, 1, KV_TILE> valid, penalty;
    Vec<int32_t, 1, KV_TILE> indices;
    typename Tiles<H>::State part_maximum, next_maximum, part_sum;
    typename Tiles<H>::Column part_maximum_column, next_maximum_column, part_sum_column;
    Vec<bfloat16_t, V, KV_TILE> weights;
    typename Tiles<H>::PVec nz;
    TASSIGN(temporary, 0);
    TASSIGN(valid, 16384);
    TASSIGN(penalty, 17408);
    TASSIGN(indices, 22528);
    TASSIGN(part_maximum, 24960);
    TASSIGN(next_maximum, 25088);
    TASSIGN(part_sum, 25216);
    TASSIGN(part_maximum_column, 24960);
    TASSIGN(next_maximum_column, 25088);
    TASSIGN(part_sum_column, 25216);
    TASSIGN(weights, 32768);
    TASSIGN(nz, 40960);
    pipe_pop(scores, score, subblock);
    TCVT(valid, indices, RoundMode::CAST_RINT);
    TADDS(valid, valid, 1.0f);
    TMAXS(valid, valid, 0.0f);
    TMINS(valid, valid, 1.0f);
    TADDS(penalty, valid, -1.0f);
    TMULS(penalty, penalty, 1e30f);
    TMULS(score, score, SCALE);
    TCOLEXPANDADD(score, score, penalty);
    TROWMAX(part_maximum_column, score, temporary);
    TMAX(next_maximum, maximum, part_maximum);
    TSUB(correction, maximum, next_maximum);
    TEXP(correction, correction);
    TROWEXPANDSUB(score, score, next_maximum_column);
    TEXP(score, score);
    TCOLEXPANDMUL(score, score, valid);
    TROWSUM(part_sum_column, score, temporary);
    TMUL(denominator, denominator, correction);
    TADD(denominator, denominator, part_sum);
    TMOV(maximum, next_maximum);
    TCVT(weights, score, RoundMode::CAST_RINT);
    TMOV(nz, weights);
    TFREE<typename Tiles<H>::ScorePipe, TileSplitAxis::TILE_UP_DOWN>(scores);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_push(probabilities, nz, subblock);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <int H>
__tf__ __aicore__ inline void update_output(
    typename Vec<float, Tiles<H>::V, D_TILE>::TileDType __out__ result,
    typename Vec<float, Tiles<H>::V, OUTPUT_TILE>::TileDType __in__ partial,
    typename Tiles<H>::State::TileDType __in__ alpha, int column) {
    __ubuf__ float* output = (__ubuf__ float*)__cce_get_tile_ptr(result);
    __ubuf__ float* input = (__ubuf__ float*)__cce_get_tile_ptr(partial);
    __ubuf__ float* factors = (__ubuf__ float*)__cce_get_tile_ptr(alpha);
#if defined(__CPU_SIM)
    for (int row = 0; row < Tiles<H>::V; ++row) {
        for (int col = 0; col < OUTPUT_TILE; ++col) {
            int offset = row * D_TILE + column + col;
            output[offset] = output[offset] * factors[row] + input[row * OUTPUT_TILE + col];
        }
    }
#else
    __VEC_SCOPE__ {
        RegTensor<float> old, value, factor;
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < static_cast<uint16_t>(Tiles<H>::V); ++row) {
            vlds(old, output, row * D_TILE + column, NORM);
            vlds(value, input, row * OUTPUT_TILE, NORM);
            vlds(factor, factors, row, BRC_B32);
            vmul(old, old, factor, full, MODE_ZEROING);
            vadd(value, old, value, full, MODE_ZEROING);
            vsts(value, output, row * D_TILE + column, NORM_B32, full);
        }
    }
#endif
}

template <int H>
__aicore__ inline void store_output(
    __gm__ int64_t* args, int token, Vec<float, Tiles<H>::V, D_TILE>& result,
    typename Tiles<H>::State& maximum, typename Tiles<H>::State& denominator,
    typename Tiles<H>::State& correction) {
    constexpr int V = Tiles<H>::V;
    typename Tiles<H>::State sink, final_maximum, sink_weight;
    typename Tiles<H>::Column gain;
    Vec<float, 1, V> sink_row;
    TASSIGN(sink, 25344);
    TASSIGN(sink_row, 25344);
    TASSIGN(final_maximum, 24960);
    TASSIGN(sink_weight, 25088);
    TASSIGN(gain, 24832);
    int valid = H - get_sub_block_id(args) * V;
    if (valid > V) valid = V;
    if (valid <= 0) return;
    TEXPANDS(sink_row, 0.0f);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    using SinkShape = TileShape2D<float, 1, DYNAMIC>;
    using SinkStride = BaseShape2D<float, 1, DYNAMIC>;
    GlobalTensor<float, SinkShape, SinkStride> sg(data<float>(args, 7) + get_sub_block_id(args) * V,
        SinkShape(1, valid), SinkStride(1, valid));
    TLOAD(sink_row, sg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMAX(final_maximum, maximum, sink);
    TSUB(correction, maximum, final_maximum);
    TEXP(correction, correction);
    TSUB(sink_weight, sink, final_maximum);
    TEXP(sink_weight, sink_weight);
    TMUL(denominator, denominator, correction);
    TADD(denominator, denominator, sink_weight);
    TDIV(correction, correction, denominator);
    TROWEXPANDMUL(result, result, gain);
    Tile<TileType::Vec, bfloat16_t, V, D_TILE, BLayout::RowMajor, DYNAMIC, D_TILE> output(V);
    TASSIGN(output, 0);
    TCVT(output, result, RoundMode::CAST_RINT);
    output.SetValidRow(valid);
    auto output_tensor = reinterpret_cast<__gm__ Tensor*>(args[8]);
    using Shape = TileShape2D<bfloat16_t, DYNAMIC, D_TILE>;
    using Stride = BaseShape2D<bfloat16_t, DYNAMIC, D_TILE>;
    auto dst = data<bfloat16_t>(args, 8) + token * output_tensor->strides[0] + get_sub_block_id(args) * V * D_TILE;
    GlobalTensor<bfloat16_t, Shape, Stride> gm(dst, Shape(valid, D_TILE), Stride(valid, D_TILE));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm, output);
}
#endif

template <typename Pipe>
__aicore__ inline void drain_completion_credits(Pipe& pipe) {
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    // Consume this task's completion credits; the destructor consumes the startup credits.
    constexpr uint32_t slots = Pipe::RingFiFo::SLOT_NUM;
    constexpr uint32_t period = Pipe::SyncPeriod;
    const uint32_t pushed = pipe.prod.tileIndex;
    const uint32_t waited = pushed > slots ? (pushed - 1) / period - (slots - 1) / period : 0;
    const uint32_t freed = pushed / period;
    for (uint32_t credit = waited; credit < freed; ++credit) {
        pipe.prod.template allocate<TileSplitAxis::TILE_UP_DOWN>();
    }
#endif
}

template <int H>
__aicore__ inline void attention(__gm__ int64_t* args) {
    KvPipe kv_pipe(nullptr, 0, 0);
    typename Tiles<H>::ScorePipe scores(nullptr, 65536, 0);
    typename Tiles<H>::ProbPipe probabilities(nullptr, 0, P_L1);
    typename Tiles<H>::OutputPipe partials(nullptr, 81920, 0);
    int tokens = args[9];
    for (int token = get_block_idx(args); token < tokens; token += get_block_num(args)) {
#if defined(__DAV_CUBE__)
        typename Tiles<H>::Query query(H);
        TASSIGN(query, Q_L1);
        cube_query<H>(args, token, query);
#else
        typename Tiles<H>::State maximum, denominator, correction;
        Vec<float, Tiles<H>::V, D_TILE> result;
        TASSIGN(maximum, 24576);
        TASSIGN(denominator, 24704);
        TASSIGN(correction, 24832);
        TASSIGN(result, 196608);
        TEXPANDS(maximum, -1e30f);
        TEXPANDS(denominator, 0.0f);
        TEXPANDS(result, 0.0f);
#endif
        for (int part = 0; part < 5; ++part) {
#if defined(__DAV_CUBE__)
            KvMat kv;
            pipe_pop(kv_pipe, kv, 0);
            cube_qk<H>(query, kv, scores);
            typename Tiles<H>::Probability probability;
            pipe_pop(probabilities, probability, 0);
            cube_pv<H>(probability, kv, partials);
            TFREE<typename Tiles<H>::ProbPipe, TileSplitAxis::TILE_UP_DOWN>(probabilities);
            TFREE<KvPipe, TileSplitAxis::TILE_UP_DOWN>(kv_pipe);
#else
            gather_kv(args, token, part, kv_pipe);
            softmax<H>(scores, probabilities, maximum, denominator, correction, get_sub_block_id(args));
            for (int column = 0; column < D_TILE; column += OUTPUT_TILE) {
                Vec<float, Tiles<H>::V, OUTPUT_TILE> partial;
                pipe_pop(partials, partial, get_sub_block_id(args));
                update_output<H>(result.data(), partial.data(), correction.data(), column);
                TFREE<typename Tiles<H>::OutputPipe, TileSplitAxis::TILE_UP_DOWN>(partials);
            }
#endif
            pipe_barrier(PIPE_ALL);
        }
#if defined(__DAV_VEC__)
        store_output<H>(args, token, result, maximum, denominator, correction);
#endif
        pipe_barrier(PIPE_ALL);
    }
    drain_completion_credits(kv_pipe);
    drain_completion_credits(scores);
    drain_completion_credits(probabilities);
    drain_completion_credits(partials);
}
template <int H>
__aicore__ inline void run(__gm__ int64_t* args) {
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_atomic_none();
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
#else
    set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
#endif
#endif
    attention<H>(args);
}
} // namespace sparse_mla
