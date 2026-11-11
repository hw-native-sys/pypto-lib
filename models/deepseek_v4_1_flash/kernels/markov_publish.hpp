// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

namespace markov_publish {
using namespace pto;
template <typename T> __aicore__ inline __gm__ T* data(__gm__ Tensor* tensor) {
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
#if defined(__DAV_VEC__)
using InputIds = Tile<TileType::Vec, int32_t, 16, 160, BLayout::RowMajor, DYNAMIC, 160>;
using InputConfidence = Tile<TileType::Vec, float, 16, 80, BLayout::RowMajor, DYNAMIC, 80>;
using OutputIds = Tile<TileType::Vec, int32_t, 16, 8, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
using OutputConfidence = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
__tf__ __aicore__ inline void compact(OutputIds::TileDType __out__ ids,
    OutputConfidence::TileDType __out__ conf, InputIds::TileDType __in__ source_ids,
    InputConfidence::TileDType __in__ source_conf, uint16_t rows) {
    auto ip = (__ubuf__ int32_t*)__cce_get_tile_ptr(source_ids);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(source_conf);
    auto op = (__ubuf__ int32_t*)__cce_get_tile_ptr(ids);
    auto pp = (__ubuf__ float*)__cce_get_tile_ptr(conf);
    __VEC_SCOPE__ {
        uint32_t count = 5;
        MaskReg active = CreatePredicate<int32_t>(count);
        RegTensor<int32_t> lane, id_offsets, conf_offsets, values;
        RegTensor<float> probabilities;
        vci(lane, 0, INC_ORDER);
        vmuls(id_offsets, lane, 32, active, MODE_ZEROING);
        vmuls(conf_offsets, lane, 16, active, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            vgather2(values, ip + row * 160, (RegTensor<uint32_t>&)id_offsets, active);
            vgather2(probabilities, cp + row * 80, (RegTensor<uint32_t>&)conf_offsets, active);
            vsts(values, op, row * 8, NORM_B32, active);
            vsts(probabilities, pp, row * 8, NORM_B32, active);
        }
    }
}
__aicore__ inline void publish(__gm__ Tensor* draft_state, __gm__ Tensor* confidence_state,
    __gm__ Tensor* draft_ids, __gm__ Tensor* confidences) {
    int batch = draft_ids->shapes[0];
    // A single writer owns packed output rows, including partial DMA blocks.
    for (int first = 0; first < batch; first += 16) {
        int rows = batch - first; rows = rows < 16 ? rows : 16;
        InputIds source_ids(rows); InputConfidence source_conf(rows);
        OutputIds ids(rows, 5); OutputConfidence conf(rows, 5);
        TASSIGN(source_ids, 0); TASSIGN(source_conf, 10240);
        TASSIGN(ids, 15360); TASSIGN(conf, 15872);
        using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        using IdShape = TileShape2D<int32_t, DYNAMIC, 160>;
        using ConfShape = TileShape2D<float, DYNAMIC, 80>;
        GlobalTensor<int32_t, IdShape, Stride> input_ids(
            reinterpret_cast<__gm__ int32_t*>(data<int64_t>(draft_state) + first * draft_state->strides[0]),
            IdShape(rows, 160), Stride(1, 1, 1, draft_state->strides[0] * 2, 1));
        GlobalTensor<float, ConfShape, Stride> input_conf(data<float>(confidence_state) + first * confidence_state->strides[0],
            ConfShape(rows, 80), Stride(1, 1, 1, confidence_state->strides[0], 1));
        TLOAD(source_ids, input_ids); TLOAD(source_conf, input_conf);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        compact(ids.data(), conf.data(), source_ids.data(), source_conf.data(), rows);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        using OutIdShape = TileShape2D<int32_t, DYNAMIC, DYNAMIC>;
        using OutConfShape = TileShape2D<float, DYNAMIC, DYNAMIC>;
        GlobalTensor<int32_t, OutIdShape, Stride> output_ids(data<int32_t>(draft_ids) + first * draft_ids->strides[0],
            OutIdShape(rows, 5), Stride(1, 1, 1, draft_ids->strides[0], 1));
        GlobalTensor<float, OutConfShape, Stride> output_conf(data<float>(confidences) + first * confidences->strides[0],
            OutConfShape(rows, 5), Stride(1, 1, 1, confidences->strides[0], 1));
        TSTORE(output_ids, ids); TSTORE(output_conf, conf);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
} // namespace markov_publish
