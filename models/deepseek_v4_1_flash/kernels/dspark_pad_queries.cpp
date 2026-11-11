// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Bulk draft query packing and initial HC coefficients.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
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
#if defined(__DAV_VEC__)
__aicore__ inline void pack(__gm__ int64_t* args) {
    auto ids = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto slots = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto indices = reinterpret_cast<__gm__ Tensor*>(args[2]);
    auto cosine = reinterpret_cast<__gm__ Tensor*>(args[3]);
    auto sine = reinterpret_cast<__gm__ Tensor*>(args[4]);
    auto padded_ids = reinterpret_cast<__gm__ Tensor*>(args[5]);
    auto padded_slots = reinterpret_cast<__gm__ Tensor*>(args[6]);
    auto padded_indices = reinterpret_cast<__gm__ Tensor*>(args[7]);
    auto padded_cosine = reinterpret_cast<__gm__ Tensor*>(args[8]);
    auto padded_sine = reinterpret_cast<__gm__ Tensor*>(args[9]);
    auto pre_mix = reinterpret_cast<__gm__ Tensor*>(args[10]);
    int first = args[11], active = args[12], rows = padded_ids->shapes[0];
    int worker = get_block_idx(args), workers = get_block_num(args);
    Tile<TileType::Vec, int64_t, 1, 4, BLayout::RowMajor, 1, 1> id, slot;
    Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4> mix;
    Vec<int32_t, 1, 192> index;
    Vec<float, 1, 32> cos, sin;
    TASSIGN(id, 0); TASSIGN(slot, 32); TASSIGN(mix, 64);
    TASSIGN(index, 128); TASSIGN(cos, 896); TASSIGN(sin, 1024);
    for (int row = worker; row < rows; row += workers) {
        if (row < active) {
            GM<int64_t, 1, 1> source_id(data<int64_t>(args, 0) + (first + row) * ids->strides[0]);
            GM<float, 1, 32> source_cos(data<float>(args, 3) + (first + row) * cosine->strides[0]);
            GM<float, 1, 32> source_sin(data<float>(args, 4) + (first + row) * sine->strides[0]);
            TLOAD(id, source_id); TLOAD(cos, source_cos); TLOAD(sin, source_sin);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        } else {
            TEXPANDS(id, static_cast<int64_t>(0));
            TEXPANDS(cos, 0.0f); TEXPANDS(sin, 0.0f);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
        GM<int64_t, 1, 1> target_id(data<int64_t>(args, 5) + row * padded_ids->strides[0]);
        GM<float, 1, 32> target_cos(data<float>(args, 8) + row * padded_cosine->strides[0]);
        GM<float, 1, 32> target_sin(data<float>(args, 9) + row * padded_sine->strides[0]);
        TSTORE(target_id, id); TSTORE(target_cos, cos); TSTORE(target_sin, sin);
        for (int layer = 0; layer < 3; ++layer) {
            if (row < active) {
                GM<int64_t, 1, 1> source_slot(data<int64_t>(args, 1) + layer * slots->strides[0] + (first + row) * slots->strides[1]);
                GM<int32_t, 1, 192> source_index(data<int32_t>(args, 2) + layer * indices->strides[0] + (first + row) * indices->strides[1]);
                TLOAD(slot, source_slot); TLOAD(index, source_index);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            } else {
                TEXPANDS(slot, static_cast<int64_t>(-1)); TEXPANDS(index, -1);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            }
            GM<int64_t, 1, 1> target_slot(data<int64_t>(args, 6) + layer * padded_slots->strides[0] + row * padded_slots->strides[1]);
            GM<int32_t, 1, 192> target_index(data<int32_t>(args, 7) + layer * padded_indices->strides[0] + row * padded_indices->strides[1]);
            TSTORE(target_slot, slot); TSTORE(target_index, index);
            pipe_barrier(PIPE_ALL);
        }
        for (int lane = 0; lane < 4; ++lane) mix.SetValue(lane, lane == 0 ? 1.0f : 0.0f);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        GM<float, 1, 4> target_mix(data<float>(args, 10) + row * pre_mix->strides[0]);
        TSTORE(target_mix, mix);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    pack(args);
#endif
}
