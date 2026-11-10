// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Paged DSpark query metadata with one worker per request and draft layer.
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
template <typename T>
__aicore__ inline T load_scalar(__gm__ T* address, uint32_t ub_address) {
    Tile<TileType::Vec, T, 1, 32 / sizeof(T), BLayout::RowMajor, 1, 1> value;
    TASSIGN(value, ub_address);
    GM<T, 1, 1> source(address);
    TLOAD(value, source);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    return value.GetValue(0);
}

__aicore__ inline void metadata(__gm__ int64_t* args) {
    auto anchors = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto positions = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto tables = reinterpret_cast<__gm__ Tensor*>(args[2]);
    auto ids_out = reinterpret_cast<__gm__ Tensor*>(args[3]);
    auto pos_out = reinterpret_cast<__gm__ Tensor*>(args[4]);
    auto slots_out = reinterpret_cast<__gm__ Tensor*>(args[5]);
    auto indices_out = reinterpret_cast<__gm__ Tensor*>(args[6]);
    auto lens_out = reinterpret_cast<__gm__ Tensor*>(args[7]);
    int batch = anchors->shapes[0];
    int worker = get_block_idx(args), workers = get_block_num(args);
    for (int job = worker; job < batch * 3; job += workers) {
        int request = job / 3, layer = job % 3;
        int first = load_scalar<int32_t>(data<int32_t>(args, 1) + request * positions->strides[0], 0);
        int active = first > 0;
        int prefix = first > 128 ? first - 128 : 0;
        int context = first - prefix;
        int visible = active ? context + 5 : 0;
        int first_page = prefix / 128;
        int pages = active ? (first + 4) / 128 - first_page + 1 : 0;
        Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, DYNAMIC> table(pages);
        TASSIGN(table, 32);
        if (active) {
            GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, DYNAMIC>> table_gm(
                data<int32_t>(args, 2) + layer * tables->strides[0] + request * tables->strides[1] + first_page,
                TileShape2D<int32_t, 1, DYNAMIC>(1, pages), BaseShape2D<int32_t, 1, DYNAMIC>(1, pages));
            TLOAD(table, table_gm);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        }
        Vec<int32_t, 1, 192> indices;
        Tile<TileType::Vec, int64_t, 1, 8, BLayout::RowMajor, 1, 5> slots;
        Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1> length;
        TASSIGN(indices, 128);
        TASSIGN(slots, 896);
        TASSIGN(length, 960);
        for (int i = 0; i < 192; ++i) {
            int slot = -1;
            if (i < visible) {
                int position = prefix + i;
                int block = table.GetValue(position / 128 - first_page);
                slot = block * 128 + position % 128;
            }
            indices.SetValue(i, slot);
            if (active && i >= context && i < context + 5) slots.SetValue(i - context, static_cast<int64_t>(slot));
        }
        if (!active) for (int i = 0; i < 5; ++i) slots.SetValue(i, static_cast<int64_t>(-1));
        length.SetValue(0, visible);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        for (int query = 0; query < 5; ++query) {
            GM<int32_t, 1, 192> target(data<int32_t>(args, 6) + layer * indices_out->strides[0]
                + (request * 5 + query) * indices_out->strides[1]);
            TSTORE(target, indices);
        }
        GM<int64_t, 1, 5> slots_gm(data<int64_t>(args, 5) + layer * slots_out->strides[0] + request * 5);
        GM<int32_t, 1, 1> length_gm(data<int32_t>(args, 7) + layer * lens_out->strides[0] + request * lens_out->strides[1]);
        TSTORE(slots_gm, slots);
        TSTORE(length_gm, length);
        if (layer == 0) {
            int64_t anchor = active ? load_scalar<int64_t>(data<int64_t>(args, 0) + request * anchors->strides[0], 992) : 0;
            Tile<TileType::Vec, int64_t, 1, 8, BLayout::RowMajor, 1, 5> ids;
            Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 5> query_positions;
            TASSIGN(ids, 1024);
            TASSIGN(query_positions, 1088);
            for (int query = 0; query < 5; ++query) {
                ids.SetValue(query, active ? (query == 0 ? anchor : static_cast<int64_t>(128799)) : static_cast<int64_t>(0));
                query_positions.SetValue(query, active ? first + query : 0);
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            GM<int64_t, 1, 5> ids_gm(data<int64_t>(args, 3) + request * 5 * ids_out->strides[0]);
            GM<int32_t, 1, 5> positions_gm(data<int32_t>(args, 4) + request * 5 * pos_out->strides[0]);
            TSTORE(ids_gm, ids);
            TSTORE(positions_gm, query_positions);
        }
        pipe_barrier(PIPE_ALL);
    }
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    metadata(args);
#endif
}
