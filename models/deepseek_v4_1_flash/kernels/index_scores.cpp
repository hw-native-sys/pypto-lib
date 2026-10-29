// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Request-group index scores with packed FP4 Cube operands.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
using F4 = float4_e2m1x2_t;
using F8 = float8_e4m3_t;
using S = float8_e8m0_t;
using QueryPipe = TPipe<4, Direction::DIR_V2C, 16384, 2, 2, true>;
using ScorePipe = TPipe<0, Direction::DIR_C2V, 32768, 2, 2, true>;

template <typename T, int R, int C, int BR, int BC, Layout L = Layout::ND>
using GM = GlobalTensor<T, TileShape2D<T, R, C, L>, BaseShape2D<T, BR, BC, L>, L>;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using QueryMat = Tile<TileType::Mat, F4, 192, 128, BLayout::ColMajor, 192, 128, SLayout::RowMajor>;
using QueryVec = Tile<TileType::Vec, F4, 192, 128, BLayout::ColMajor, 192, 128, SLayout::RowMajor>;
using KeyMat = Tile<TileType::Mat, F4, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor>;
using QueryScale = Tile<TileType::Mat, S, 192, 4, BLayout::RowMajor, DYNAMIC, 4, SLayout::RowMajor, 32>;
using KeyScale = Tile<TileType::Mat, S, 4, 128, BLayout::ColMajor, 4, 128, SLayout::ColMajor, 32>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int read(__gm__ int32_t* ptr) {
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}

__aicore__ inline int minimum(int a, int b) { return a < b ? a : b; }

#if defined(__DAV_VEC__)
__aicore__ inline void prepare_query(__gm__ int64_t* args, int begin, int count, QueryPipe& pipe) {
    Tile<TileType::Vec, F8, 192, 128, BLayout::RowMajor, DYNAMIC, 128> input(count * 32);
    Vec<F8, 192, 128> full_input;
    Vec<float, 192, 128> fp32;
    Vec<bfloat16_t, 192, 128> bf16;
    Vec<F4, 192, 128> packed;
    QueryVec nz;
    TASSIGN(input, 131072);
    TASSIGN(full_input, 131072);
    TASSIGN(fp32, 155648);
    TASSIGN(bf16, 65536);
    TASSIGN(packed, 131072);
    TASSIGN(nz, 143360);
    GM<F8, DYNAMIC, 128, DYNAMIC, 128> gm(data<F8>(args, 0) + begin * 4096,
                                        TileShape2D<F8, DYNAMIC, 128>(count * 32, 128),
                                        BaseShape2D<F8, DYNAMIC, 128>(count * 32, 128));
    TLOAD(input, gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(fp32, full_input, RoundMode::CAST_RINT);
    TCVT(bf16, fp32, RoundMode::CAST_RINT);
    TCVT(packed, bf16, RoundMode::CAST_RINT);
    TMOV(nz, packed);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<QueryPipe, QueryVec, TileSplitAxis::TILE_NO_SPLIT>(pipe, nz);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

    Tile<TileType::Vec, bfloat16_t, 6, 32, BLayout::RowMajor, DYNAMIC, 32> weight(count);
    Vec<bfloat16_t, 1, 192> weight_raw;
    Vec<float, 1, 192> scaled;
    Vec<bfloat16_t, 1, 192> rounded;
    Vec<float, 1, 192> result;
    TASSIGN(weight, 65536);
    TASSIGN(weight_raw, 65536);
    TASSIGN(scaled, 66048);
    TASSIGN(rounded, 66816);
    TASSIGN(result, 67328);
    GM<bfloat16_t, DYNAMIC, 32, DYNAMIC, 32> wg(data<bfloat16_t>(args, 2) + begin * 32,
        TileShape2D<bfloat16_t, DYNAMIC, 32>(count, 32), BaseShape2D<bfloat16_t, DYNAMIC, 32>(count, 32));
    // Complete query conversion before the weight DMA reuses its UB storage.
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(weight, wg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(scaled, weight_raw, RoundMode::CAST_RINT);
    TMULS(scaled, scaled, 0.015625f);
    TCVT(rounded, scaled, RoundMode::CAST_RINT);
    TCVT(result, rounded, RoundMode::CAST_RINT);
}

__aicore__ inline void store_score(__gm__ int64_t* args, int token, int group_offset,
                                  int logical, int page, Vec<bfloat16_t, 64, 128>& dot,
                                  int pair_offset, bool have_dot) {
    int width = args[10];
    int output_count = minimum(128, width - logical);
    int visible = minimum(read(data<int32_t>(args, 7) + token), width);
    if (visible <= args[13]) visible = 0;
    int valid = have_dot && page >= 0 ? minimum(128, visible - logical) : 0;
    if (valid < 0) valid = 0;
    Vec<float, 1, 128> total;
    TASSIGN(total, 110592);
    if (valid > 0) {
        Vec<bfloat16_t, 32, 128> heads;
        Vec<float, 32, 128> work;
        Vec<bfloat16_t, 32, 128> rounded;
        Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor> weight;
        Vec<bfloat16_t, 1, 128> reduced;
        TASSIGN(heads, reinterpret_cast<uint64_t>(dot.data()) + pair_offset * 8192);
        TASSIGN(work, 86016);
        TASSIGN(rounded, 102400);
        TASSIGN(weight, 67328 + group_offset * 128);
        TASSIGN(reduced, 111104);
        TCVT(work, heads, RoundMode::CAST_RINT);
        TMAXS(work, work, 0.0f);
        TROWEXPANDMUL(work, work, weight);
        TCVT(rounded, work, RoundMode::CAST_RINT);
        TCVT(work, rounded, RoundMode::CAST_RINT);
        TCOLSUM(total, work);
        TCVT(reduced, total, RoundMode::CAST_RINT);
        TCVT(total, reduced, RoundMode::CAST_RINT);
    } else {
        TEXPANDS(total, -__builtin_inff());
    }
    if (valid < output_count && valid > 0) {
        Vec<int32_t, 1, 128> indices;
        Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, 1, 16> mask;
        Vec<uint8_t, 1, 32> tmp;
        Vec<float, 1, 128> negative;
        TASSIGN(indices, 111360);
        TASSIGN(mask, 111872);
        TASSIGN(tmp, 112128);
        TASSIGN(negative, 112384);
        TEXPANDS(negative, -__builtin_inff());
        TCI<decltype(indices), int32_t, 0>(indices, 0);
        TCMPS(mask, indices, valid, CmpMode::LT);
        TSEL(total, mask, total, negative, tmp);
    }
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, DYNAMIC> output(output_count);
    TASSIGN(output, 110592);
    using Shape = TileShape2D<float, 1, DYNAMIC>;
    using Stride = BaseShape2D<float, DYNAMIC, DYNAMIC>;
    GlobalTensor<float, Shape, Stride> gm(data<float>(args, 8) + token * width + logical,
                                         Shape(1, output_count), Stride(1, width));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm, output);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}
#endif

__aicore__ inline void score(__gm__ int64_t* args) {
    QueryPipe query_pipe(nullptr, 0, 0);
    ScorePipe score_pipe(nullptr, 0, 0);
    int width = args[10];
    int run_stride = args[11];
    int table_width = args[12];
    int macro_tiles = (width + 511) / 512;
    int run_count = read(data<int32_t>(args, 9) + 3 * run_stride);
    int block = get_block_idx(args);
    int workers = get_block_num(args);
    for (int work = block; work < run_count * macro_tiles; work += workers) {
        int run = work / macro_tiles;
        int logical_begin = work % macro_tiles * 512;
        int begin = read(data<int32_t>(args, 9) + run);
        int end = read(data<int32_t>(args, 9) + run_stride + run);
        int maximum = read(data<int32_t>(args, 9) + 2 * run_stride + run);
        int request = read(data<int32_t>(args, 6) + begin);
        for (int group = begin; group < end; group += 6) {
            int count = minimum(6, end - group);
#if defined(__DAV_CUBE__)
            QueryMat query;
            QueryScale query_scale(count * 32);
            TASSIGN(query_scale, 75008);
            TPOP<QueryPipe, QueryMat, TileSplitAxis::TILE_NO_SPLIT>(query_pipe, query);
            using QShape = TileShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
            using QStride = BaseShape2D<S, DYNAMIC, 4, Layout::MX_A_ND>;
            GlobalTensor<S, QShape, QStride, Layout::MX_A_ND> qg(data<S>(args, 1) + group * 128,
                QShape(count * 32, 4), QStride(count * 32, 4));
            TLOAD(query_scale, qg);
#else
            prepare_query(args, group, count, query_pipe);
#endif
            for (int panel = 0; panel < 4; ++panel) {
                int logical = logical_begin + panel * 128;
                if (logical >= width) continue;
                int page = -1;
                if (logical < maximum) page = read(data<int32_t>(args, 5) + request * table_width + logical / 128);
                bool have_dot = logical < maximum && page >= 0;
#if defined(__DAV_CUBE__)
                if (have_dot) {
                    KeyMat key;
                    KeyScale key_scale;
                    TileRight<F4, 128, 128> right;
                    TileRightScale<S, 4, 128> right_scale;
                    TASSIGN(key, 65536);
                    TASSIGN(key_scale, 73728);
                    TASSIGN(right, 0);
                    TASSIGN(right_scale, GetScaleAddr(right.data()));
                    GM<F4, 128, 128, 128, 128, Layout::DN> kg(data<F4>(args, 3) + page * 8192);
                    GM<S, 4, 128, 4, 128, Layout::MX_B_DN> sg(data<S>(args, 4) + page * 512);
                    TLOAD(key, kg);
                    TLOAD(key_scale, sg);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    TMOV(right, key);
                    TMOV(right_scale, key_scale);
                    for (int pair = 0; pair < (count + 1) / 2; ++pair) {
                        TileLeft<F4, 64, 128> left;
                        TileLeftScale<S, 64, 4> left_scale;
                        TileAcc<float, 64, 128> dot;
                        TASSIGN(left, 0);
                        TASSIGN(left_scale, GetScaleAddr(left.data()));
                        TASSIGN(dot, 0);
                        TEXTRACT(left, query, pair * 64, 0);
                        TEXTRACT(left_scale, query_scale, pair * 64, 0);
                        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                        TMATMUL_MX(dot, left, left_scale, right, right_scale);
                        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                        TPUSH<ScorePipe, decltype(dot), TileSplitAxis::TILE_NO_SPLIT>(score_pipe, dot);
                        pipe_barrier(PIPE_ALL);
                    }
                }
#else
                for (int pair = 0; pair < (count + 1) / 2; ++pair) {
                    Vec<bfloat16_t, 64, 128> rounded;
                    TASSIGN(rounded, 69632);
                    if (have_dot) {
                        Vec<float, 64, 128> dot;
                        TPOP<ScorePipe, decltype(dot), TileSplitAxis::TILE_NO_SPLIT>(score_pipe, dot);
                        TCVT(rounded, dot, RoundMode::CAST_RINT);
                        TFREE<ScorePipe, TileSplitAxis::TILE_NO_SPLIT>(score_pipe);
                    }
                    for (int offset = 0; offset < 2 && pair * 2 + offset < count; ++offset) {
                        int group_offset = pair * 2 + offset;
                        store_score(args, group + group_offset, group_offset, logical, page,
                                    rounded, offset, have_dot);
                    }
                }
#endif
            }
#if defined(__DAV_CUBE__)
            TFREE<QueryPipe, TileSplitAxis::TILE_NO_SPLIT>(query_pipe);
#endif
            pipe_barrier(PIPE_ALL);
        }
    }
}
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_atomic_none();
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
#endif
#if defined(__DAV_CUBE__)
    set_ctrl(sbitset0(get_ctrl(), pto::MAD_MODE_BIT));
#endif
#endif
    score(args);
}
