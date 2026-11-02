// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed FP8/FP4 cache gather and BF16 dequantization.
#include "packed_cache.hpp"

using namespace packed_cache;

namespace {
#if defined(__DAV_VEC__)
__aicore__ inline void gather(__gm__ int64_t* args) {
    Vec<int32_t, 1, 32> ids;
    Vec<bfloat16_t, 32, 512> output;
    Vec<bfloat16_t, 1, 512> zero;
    TASSIGN(ids, 135168);
    TASSIGN(output, 32768);
    TASSIGN(zero, 135424);
    TEXPANDS(zero, (bfloat16_t)0.0f);
    int tokens = args[8];
    int worker = get_block_idx(args);
    int workers = get_block_num(args);
    for (int block = worker; block < tokens * 20; block += workers) {
        int token = block / 20;
        int column = block % 20 * 32;
        int source = column < 128 ? 4 : 5;
        int source_column = column < 128 ? column : column - 128;
        auto source_tensor = reinterpret_cast<__gm__ Tensor*>(args[source]);
        auto output_tensor = reinterpret_cast<__gm__ Tensor*>(args[6]);
        auto indices_tensor = reinterpret_cast<__gm__ Tensor*>(args[7]);
        GM<int32_t, 1, 32> input_ids(
            data<int32_t>(args, source) + (int64_t)token * source_tensor->strides[0] + source_column);
        TLOAD(ids, input_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        if (column < 128) decode_window<EVENT_ID0>(data<float8_e4m3_t>(args, 0), data<bfloat16_t>(args, 1), ids, output);
        else decode_compressed<EVENT_ID0>(data<uint8_t>(args, 2), data<bfloat16_t>(args, 3), ids, output);
        auto dst = data<bfloat16_t>(args, 6) + (int64_t)token * output_tensor->strides[0] + column * 512;
        auto combined = data<int32_t>(args, 7) + (int64_t)token * indices_tensor->strides[0] + column;
        GM<bfloat16_t, 32, 512> output_gm(dst);
        GM<int32_t, 1, 32> indices_gm(combined);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(output_gm, output);
        TSTORE(indices_gm, ids);
        for (int lane = 0; lane < 32; ++lane) {
            if (ids.GetValue(lane) < 0) {
                GM<bfloat16_t, 1, 512> masked(dst + lane * 512);
                // Order the overlapping full-tile and masked-row stores.
                pipe_barrier(PIPE_MTE3);
                TSTORE(masked, zero);
            }
        }
        // Complete all UB readers before the next tile reuses the buffers.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    gather(args);
#endif
}
