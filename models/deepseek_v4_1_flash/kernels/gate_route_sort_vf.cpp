// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Router Top-K selection and normalized weight publication.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
constexpr int Rows = 8;
constexpr int TopK = 6;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using Indices = Vec<int32_t, Rows, 8>;
using Selected = Vec<float, Rows, 8>;
using Denominator = Tile<TileType::Vec, float, Rows, 1, BLayout::ColMajor>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
#if defined(PTO_NPU_ARCH_A5)
template <int Columns>
__tf__ __aicore__ inline void gather_scores(
    Selected::TileDType __out__ output, Indices::TileDType __in__ indices,
    typename Vec<float, Rows, Columns>::TileDType __in__ scores) {
    auto idx = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), active;
        RegTensor<uint32_t> lane, row, column, bits, index, offset;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(row, lane, 3u, all, MODE_ZEROING);
        vdup(bits, 7u, all, MODE_ZEROING);
        vand(column, lane, bits, all, MODE_ZEROING);
        vcmps_lt(active, column, static_cast<uint32_t>(TopK), all);
        vlds((RegTensor<int32_t>&)index, idx, 0, NORM);
        vmuls(offset, row, static_cast<uint32_t>(Columns), all, MODE_ZEROING);
        vadd(index, index, offset, all, MODE_ZEROING);
        RegTensor<float> value, zero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vgather2(value, src, index, all);
        vsel(value, value, zero, active);
        vsts(value, dst, 0, NORM_B32, all);
    }
}

__tf__ __aicore__ inline void normalize_weights(
    Selected::TileDType __out__ output, Selected::TileDType __in__ selected,
    Denominator::TileDType __in__ denominator, float route_scale) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(selected);
    auto inv = (__ubuf__ float*)__cce_get_tile_ptr(denominator);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<uint32_t> lane, row;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vshrs(row, lane, 3u, all, MODE_ZEROING);
        RegTensor<float> value, divisor;
        vlds(value, src, 0, NORM);
        vgather2(divisor, inv, row, all);
        vdiv(value, value, divisor, all, MODE_ZEROING);
        vmuls(value, value, route_scale, all, MODE_ZEROING);
        vsts(value, dst, 0, NORM_B32, all);
    }
}
#endif

template <int Columns>
__aicore__ inline void run(__gm__ int64_t* args) {
    int capacity = reinterpret_cast<__gm__ Tensor*>(args[2])->shapes[0];
    int active = args[4];
    active = active < 0 ? 0 : (active > capacity ? capacity : active);
    int first = get_block_idx(args) * Rows;
    int valid = active - first;
    valid = valid > Rows ? Rows : valid;
    if (valid <= 0) return;
    float route_scale = *reinterpret_cast<__gm__ float*>(args + 5);
    using Input = Vec<float, Rows, Columns>;
    using Row = Vec<float, 1, Columns>;
    using Sorted = Vec<float, 1, 2 * Columns>;
    using Part = Vec<float, 1, 256>;
    using Range = Vec<uint32_t, 1, Columns>;
    Input biased, scores;
    Range range;
    Sorted sorted32, sorted64, sorted, scratch;
    Indices indices;
    Selected selected, weights;
    Denominator denominator;
    Vec<float, Rows, 128> sum_scratch;
    TASSIGN(biased, 0); TASSIGN(scores, 16384);
    TASSIGN(sorted32, 32768); TASSIGN(sorted64, 36864);
    TASSIGN(sorted, 40960); TASSIGN(scratch, 45056);
    TASSIGN(range, 49152); TASSIGN(indices, 51200);
    TASSIGN(selected, 51456); TASSIGN(weights, 51712);
    TASSIGN(denominator, 51968); TASSIGN(sum_scratch, 52224);
    using Shape = TileShape2D<float, Rows, Columns>;
    using Stride = BaseShape2D<float, Rows, Columns>;
    GlobalTensor<float, Shape, Stride> biased_gm(data<float>(args, 0) + first * Columns);
    GlobalTensor<float, Shape, Stride> scores_gm(data<float>(args, 1) + first * Columns);
    TLOAD(biased, biased_gm); TLOAD(scores, scores_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCI<Range, uint32_t, 0>(range, 0u);
    for (int row = 0; row < Rows; ++row) {
        Row input;
        TASSIGN(input, row * Columns * sizeof(float));
        TSORT32(sorted32, input, range);
        pipe_barrier(PIPE_ALL);
        TMRGSORT(sorted64, sorted32, 64);
        pipe_barrier(PIPE_ALL);
        Part p0, p1, p2;
        TASSIGN(p0, 36864); TASSIGN(p1, 37888); TASSIGN(p2, 38912);
        MrgSortExecutedNumList executed{0, 0, 0, 0};
        if constexpr (Columns == 256) {
            TMRGSORT<Sorted, Sorted, Part, Part, false>(sorted, executed, scratch, p0, p1);
        } else {
            TMRGSORT<Sorted, Sorted, Part, Part, Part, false>(sorted, executed, scratch, p0, p1, p2);
        }
        pipe_barrier(PIPE_ALL);
        Vec<float, 1, 16> prefix;
        Vec<int32_t, 1, 8> row_indices;
        TASSIGN(prefix, 40960); TASSIGN(row_indices, 51200 + row * 32);
        TGATHER<decltype(row_indices), decltype(prefix), MaskPattern::P1010>(row_indices, prefix);
        pipe_barrier(PIPE_ALL);
    }
#if defined(PTO_NPU_ARCH_A5)
    gather_scores<Columns>(selected.data(), indices.data(), scores.data());
#else
    Vec<int32_t, Rows, 8> offsets, row_index;
    Vec<int32_t, 1, 64> linear;
    Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, DYNAMIC, DYNAMIC> gathered(Rows, 8);
    TASSIGN(offsets, 56320); TASSIGN(row_index, 56576); TASSIGN(linear, 56576);
    TASSIGN(gathered, 56832);
    TCI<decltype(linear), int32_t, 0>(linear, 0);
    TDIVS(row_index, row_index, 8);
    TMULS(offsets, row_index, Columns);
    TADD(offsets, offsets, indices);
    Vec<int32_t, Rows, 8> gather_scratch;
    TASSIGN(gather_scratch, 57088);
    TGATHER(gathered, scores, offsets, gather_scratch);
    gathered.SetValidShape(Rows, TopK);
    Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, Rows, 8,
         SLayout::NoneBox, 512, PadValue::Zero> padded;
    TASSIGN(padded, 51456);
    TFILLPAD(padded, gathered);
#endif
    pipe_barrier(PIPE_ALL);
    TROWSUM(denominator, selected, sum_scratch);
    pipe_barrier(PIPE_ALL);
#if defined(PTO_NPU_ARCH_A5)
    normalize_weights(weights.data(), selected.data(), denominator.data(), route_scale);
#else
    TROWEXPANDDIV(weights, selected, denominator);
    TMULS(weights, weights, route_scale);
#endif
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using OutputIndex = Tile<TileType::Vec, int32_t, Rows, 8, BLayout::RowMajor, DYNAMIC, TopK>;
    using OutputWeight = Tile<TileType::Vec, float, Rows, 8, BLayout::RowMajor, DYNAMIC, TopK>;
    OutputIndex output_index(valid);
    OutputWeight output_weight(valid);
    TASSIGN(output_index, 51200); TASSIGN(output_weight, 51712);
    using IndexShape = TileShape2D<int32_t, DYNAMIC, TopK>;
    using WeightShape = TileShape2D<float, DYNAMIC, TopK>;
    GlobalTensor<int32_t, IndexShape, BaseShape2D<int32_t, Rows, TopK>> index_gm(
        data<int32_t>(args, 2) + first * TopK, IndexShape(valid, TopK), BaseShape2D<int32_t, Rows, TopK>());
    GlobalTensor<float, WeightShape, BaseShape2D<float, Rows, TopK>> weight_gm(
        data<float>(args, 3) + first * TopK, WeightShape(valid, TopK), BaseShape2D<float, Rows, TopK>());
    TSTORE(index_gm, output_index); TSTORE(weight_gm, output_weight);
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
#endif
    int columns = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[1];
    if (columns == 256) run<256>(args);
    else run<384>(args);
#endif
}
