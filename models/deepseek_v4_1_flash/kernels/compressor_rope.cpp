// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Interleaved rotary embedding on the last 64 compressor channels.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
template <typename T, int C>
using Vec = Tile<TileType::Vec, T, 8, C, BLayout::RowMajor, DYNAMIC, C>;
template <typename T, int C>
using GM = GlobalTensor<T, TileShape2D<T, DYNAMIC, C>, BaseShape2D<T, DYNAMIC, DYNAMIC>>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void rotate(
    Vec<bfloat16_t, 512>::TileDType __out__ output,
    Vec<bfloat16_t, 512>::TileDType __in__ row,
    Vec<float, 32>::TileDType __in__ cosine,
    Vec<float, 32>::TileDType __in__ sine, uint16_t rows) {
    auto values = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(row);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(cosine);
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(sine);
#if defined(__CPU_SIM)
    for (int row_index = 0; row_index < rows; ++row_index) {
        for (int pair = 0; pair < 32; ++pair) {
            float even = static_cast<float>(values[row_index * 512 + 448 + pair * 2]);
            float odd = static_cast<float>(values[row_index * 512 + 449 + pair * 2]);
            dst[row_index * 512 + 448 + pair * 2] = static_cast<bfloat16_t>(even * cp[row_index * 32 + pair] - odd * sp[row_index * 32 + pair]);
            dst[row_index * 512 + 449 + pair * 2] = static_cast<bfloat16_t>(even * sp[row_index * 32 + pair] + odd * cp[row_index * 32 + pair]);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = 32;
        MaskReg half = plt_b32(count, POST_UPDATE);
        for (uint16_t row_index = 0; row_index < rows; ++row_index) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> raw, even, odd, cosv, sinv, a, b, real, imag, result, unused;
            vlds(bf, values, row_index * 512 + 448, UNPK_B16);
            vcvt(raw, bf, full, PART_EVEN);
            vlds(cosv, cp, row_index * 32, NORM);
            vlds(sinv, sp, row_index * 32, NORM);
            vdintlv(even, odd, raw, raw);
            vmul(a, even, cosv, half, MODE_ZEROING);
            vmul(b, odd, sinv, half, MODE_ZEROING);
            vsub(real, a, b, half);
            vmul(a, even, sinv, half, MODE_ZEROING);
            vmul(b, odd, cosv, half, MODE_ZEROING);
            vadd(imag, a, b, half, MODE_ZEROING);
            vintlv(result, unused, real, imag);
            vcvt(bf, result, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row_index * 512 + 448, PK_B32, full);
        }
    }
#endif
}

__aicore__ inline void run(__gm__ int64_t* args) {
    Vec<bfloat16_t, 512> row(8);
    Vec<float, 32> cosine(8), sine(8);
    TASSIGN(row, 0);
    TASSIGN(cosine, 8192);
    TASSIGN(sine, 9216);
    int tokens = args[4];
    int workers = get_block_num(args);
    for (int token = get_block_idx(args); token < tokens; token += workers * 8) {
        int remaining = (tokens - token + workers - 1) / workers;
        int rows = remaining < 8 ? remaining : 8;
        row.SetValidRow(rows);
        cosine.SetValidRow(rows);
        sine.SetValidRow(rows);
        GM<bfloat16_t, 512> input(data<bfloat16_t>(args, 0) + (int64_t)token * 512,
            TileShape2D<bfloat16_t, DYNAMIC, 512>(rows, 512),
            BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC>(rows, workers * 512));
        GM<float, 32> cos_gm(data<float>(args, 1) + (int64_t)token * 32,
            TileShape2D<float, DYNAMIC, 32>(rows, 32),
            BaseShape2D<float, DYNAMIC, DYNAMIC>(rows, workers * 32));
        GM<float, 32> sin_gm(data<float>(args, 2) + (int64_t)token * 32,
            TileShape2D<float, DYNAMIC, 32>(rows, 32),
            BaseShape2D<float, DYNAMIC, DYNAMIC>(rows, workers * 32));
        TLOAD(row, input);
        TLOAD(cosine, cos_gm);
        TLOAD(sine, sin_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        rotate(row.data(), row.data(), cosine.data(), sine.data(), rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GM<bfloat16_t, 512> output(data<bfloat16_t>(args, 3) + (int64_t)token * 512,
            TileShape2D<bfloat16_t, DYNAMIC, 512>(rows, 512),
            BaseShape2D<bfloat16_t, DYNAMIC, DYNAMIC>(rows, workers * 512));
        TSTORE(output, row);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    run(args);
#endif
}
