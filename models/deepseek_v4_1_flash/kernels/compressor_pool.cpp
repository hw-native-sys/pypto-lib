// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Ratio-2 pair pooling, BF16 rounding, and weighted RMS normalization.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
template <typename T, int R, int C>
using TileRC = Tile<TileType::Vec, T, R, C>;
template <typename T, int C>
using GM = GlobalTensor<T, TileShape2D<T, 1, C>, BaseShape2D<T, 1, C>>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void pool_rows(
    TileRC<bfloat16_t, 8, 512>::TileDType __out__ output,
    TileRC<float, 8, 512>::TileDType __in__ current_kv,
    TileRC<float, 8, 512>::TileDType __in__ current_score,
    TileRC<float, 8, 512>::TileDType __in__ previous_kv,
    TileRC<float, 8, 512>::TileDType __in__ previous_score,
    TileRC<bfloat16_t, 1, 512>::TileDType __in__ weight,
    TileRC<float, 1, 512>::TileDType __out__ scratch, uint16_t mask, uint16_t rows) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto ck = (__ubuf__ float*)__cce_get_tile_ptr(current_kv);
    auto cs = (__ubuf__ float*)__cce_get_tile_ptr(current_score);
    auto pk = (__ubuf__ float*)__cce_get_tile_ptr(previous_kv);
    auto ps = (__ubuf__ float*)__cce_get_tile_ptr(previous_score);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto rounded = (__ubuf__ float*)__cce_get_tile_ptr(scratch);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float square = 0;
        for (int col = 0; col < 512; ++col) {
            float v = 0;
            if (mask & (1u << row)) {
                int i = row * 512 + col;
                float highest = std::max(cs[i], ps[i]);
                float a = std::exp(ps[i] - highest), b = std::exp(cs[i] - highest);
                v = static_cast<float>(static_cast<bfloat16_t>(pk[i] * (a / (a + b)) + ck[i] * (b / (a + b))));
            }
            rounded[col] = v;
            square += v * v;
        }
        float inv = 1.0f / std::sqrt(square / 512.0f + 1.0e-6f);
        for (int col = 0; col < 512; ++col)
            dst[row * 512 + col] = static_cast<bfloat16_t>(rounded[col] * inv * static_cast<float>(gamma[col]));
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            if (!(mask & (1u << row))) {
                RegTensor<bfloat16_t> zero;
                MaskReg full16 = pset_b16(PAT_ALL);
                vdup(zero, (bfloat16_t)0.0f, full16, MODE_ZEROING);
                for (uint16_t chunk = 0; chunk < 4; ++chunk)
                    vsts(zero, dst, row * 512 + chunk * 128, NORM_B16, full16);
                continue;
            }
            RegTensor<float> sum;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                int offset = row * 512 + chunk * 64;
                RegTensor<float> a, b, high, av, bv, denominator, value, square;
                RegTensor<bfloat16_t> bf;
                vlds(a, ps, offset, NORM);
                vlds(b, cs, offset, NORM);
                vmax(high, a, b, full, MODE_ZEROING);
                vsub(a, a, high, full);
                vsub(b, b, high, full);
                vexp(a, a, full, MODE_ZEROING);
                vexp(b, b, full, MODE_ZEROING);
                vlds(av, pk, offset, NORM);
                vlds(bv, ck, offset, NORM);
                vadd(denominator, a, b, full, MODE_ZEROING);
                vdiv(a, a, denominator, full, MODE_ZEROING);
                vdiv(b, b, denominator, full, MODE_ZEROING);
                vmul(av, av, a, full, MODE_ZEROING);
                vmul(bv, bv, b, full, MODE_ZEROING);
                vadd(value, av, bv, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vcvt(value, bf, full, PART_EVEN);
                vsts(value, rounded, chunk * 64, NORM_B32, full);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(sum, sum, 1.0f / 512.0f, full, MODE_ZEROING);
            vadds(sum, sum, 1.0e-6f, full, MODE_ZEROING);
            RegTensor<float> one, inv, root;
            vdup(one, 1.0f, full, MODE_ZEROING);
            vsqrt(root, sum, full, MODE_ZEROING);
            vdiv(inv, one, root, full, MODE_ZEROING);
            // Broadcast the reduced RMS factor from lane zero.
            vdup(inv, inv, full, POS_LOWEST, MODE_ZEROING);
            mem_bar(VST_VLD);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<float> value, g;
                RegTensor<bfloat16_t> bf;
                vlds(value, rounded, chunk * 64, NORM);
                vlds(bf, gamma, chunk * 64, UNPK_B16);
                vcvt(g, bf, full, PART_EVEN);
                vmul(value, value, inv, full, MODE_ZEROING);
                vmul(value, value, g, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(bf, dst, row * 512 + chunk * 64, PK_B32, full);
            }
        }
    }
#endif
}

__aicore__ inline void run(__gm__ int64_t* args) {
    int tokens = args[9];
    int begin = get_block_idx(args) * 8;
    int rows = tokens - begin < 8 ? tokens - begin : 8;
    if (rows <= 0) return;
    TileRC<float, 8, 512> current_kv, current_score, previous_kv, previous_score;
    TileRC<bfloat16_t, 8, 512> output;
    TileRC<bfloat16_t, 1, 512> weight;
    TileRC<float, 1, 512> scratch;
    TASSIGN(current_kv, 0);
    TASSIGN(current_score, 16384);
    TASSIGN(previous_kv, 32768);
    TASSIGN(previous_score, 49152);
    TASSIGN(output, 65536);
    TASSIGN(weight, 73728);
    TASSIGN(scratch, 74752);
    using Meta = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, DYNAMIC>;
    Meta positions(rows), requests(rows), boundaries(2), state_blocks(1);
    TASSIGN(positions, 76800);
    TASSIGN(requests, 76832);
    TASSIGN(boundaries, 76864);
    TASSIGN(state_blocks, 76896);
    using MetaGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 8>>;
    MetaGM pos_gm(data<int32_t>(args, 3) + begin, TileShape2D<int32_t, 1, DYNAMIC>(1, rows));
    MetaGM req_gm(data<int32_t>(args, 7) + begin, TileShape2D<int32_t, 1, DYNAMIC>(1, rows));
    TLOAD(positions, pos_gm);
    TLOAD(requests, req_gm);
    GM<bfloat16_t, 512> weight_gm(data<bfloat16_t>(args, 4));
    TLOAD(weight, weight_gm);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    int request_count = reinterpret_cast<__gm__ Tensor*>(args[8])->shapes[0];
    int blocks = args[10];
    uint16_t active = 0;
    int last_request = -1, start = 0, end = 0, block = -1;
    for (int row = 0; row < rows; ++row) {
        int token = begin + row;
        int request = requests.GetValue(row), position = positions.GetValue(row);
        if (request < 0 || request >= request_count || position < 0 || !(position % 2)) continue;
        if (request != last_request) {
            MetaGM bounds_gm(data<int32_t>(args, 6) + request, TileShape2D<int32_t, 1, DYNAMIC>(1, 2));
            MetaGM blocks_gm(data<int32_t>(args, 8) + request, TileShape2D<int32_t, 1, DYNAMIC>(1, 1));
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            TLOAD(boundaries, bounds_gm);
            TLOAD(state_blocks, blocks_gm);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            start = boundaries.GetValue(0);
            end = boundaries.GetValue(1);
            block = state_blocks.GetValue(0);
            last_request = request;
        }
        if (block < 0 || block >= blocks || token < start || token >= end) continue;
        active |= 1u << row;
        TileRC<float, 1, 512> ck, cs, pk, ps;
        TASSIGN(ck, row * 2048);
        TASSIGN(cs, 16384 + row * 2048);
        TASSIGN(pk, 32768 + row * 2048);
        TASSIGN(ps, 49152 + row * 2048);
        GM<float, 512> ck_gm(data<float>(args, 0) + token * 512);
        GM<float, 512> cs_gm(data<float>(args, 1) + token * 512);
        __gm__ float* prev_kv;
        __gm__ float* prev_score;
        if (token > start) {
            prev_kv = data<float>(args, 0) + (token - 1) * 512;
            prev_score = data<float>(args, 1) + (token - 1) * 512;
        } else {
            int slot = block * 4 + (position - 1) % 4;
            prev_kv = data<float>(args, 2) + slot * 1024;
            prev_score = prev_kv + 512;
        }
        GM<float, 512> pk_gm(prev_kv), ps_gm(prev_score);
        TLOAD(ck, ck_gm);
        TLOAD(cs, cs_gm);
        TLOAD(pk, pk_gm);
        TLOAD(ps, ps_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pool_rows(output.data(), current_kv.data(), current_score.data(), previous_kv.data(),
              previous_score.data(), weight.data(), scratch.data(), active, rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using Output = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, 1, DYNAMIC>;
    using OutputGM = GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, DYNAMIC>, BaseShape2D<bfloat16_t, 1, 4096>>;
    Output result(rows * 512);
    TASSIGN(result, 65536);
    OutputGM result_gm(data<bfloat16_t>(args, 5) + begin * 512,
                       TileShape2D<bfloat16_t, 1, DYNAMIC>(1, rows * 512));
    TSTORE(result_gm, result);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    run(args);
#endif
}
