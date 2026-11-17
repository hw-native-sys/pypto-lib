// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Merge exact vocabulary-panel maxima and first indices.
#include "bf16_projection.hpp"
using namespace pto;
using namespace bf16_projection;
#if defined(__DAV_VEC__)
using PartialScores = Tile<TileType::Vec, float, 32, 8>;
using PartialIds = Tile<TileType::Vec, int32_t, 32, 8>;
using OutputIds = Tile<TileType::Vec, int32_t, 8, 8, BLayout::RowMajor, DYNAMIC, 8>;

__tf__ __aicore__ inline void merge_values(
    OutputIds::TileDType __out__ output, PartialScores::TileDType __in__ scores,
    PartialIds::TileDType __in__ indices) {
    auto values = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto ids = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto destination = (__ubuf__ int32_t*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<float> best;
        RegTensor<int32_t> best_id;
        vlds(best, values, 0, NORM);
        vlds(best_id, ids, 0, NORM);
        for (uint16_t segment = 1; segment < 4; ++segment) {
            RegTensor<float> candidate;
            RegTensor<int32_t> candidate_id;
            vlds(candidate, values, segment * 64, NORM);
            vlds(candidate_id, ids, segment * 64, NORM);
            MaskReg greater, equal, earlier, choose;
            vcmp_gt(greater, candidate, best, all);
            vcmp_eq(equal, candidate, best, all);
            vcmp_lt(earlier, candidate_id, best_id, all);
            pand(equal, equal, earlier, all);
            por(choose, greater, equal, all);
            vsel(best, candidate, best, choose);
            vsel(best_id, candidate_id, best_id, choose);
        }
        RegTensor<uint32_t> lane, bit, shuffled_index;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        for (uint16_t shift = 8; shift <= 32; shift *= 2) {
            RegTensor<float> candidate;
            RegTensor<int32_t> candidate_id;
            vdup(bit, static_cast<uint32_t>(shift), all, MODE_ZEROING);
            vxor(shuffled_index, lane, bit, all);
            vselr((RegTensor<uint32_t>&)candidate, (RegTensor<uint32_t>&)best, shuffled_index);
            vselr((RegTensor<uint32_t>&)candidate_id, (RegTensor<uint32_t>&)best_id, shuffled_index);
            MaskReg greater, equal, earlier, choose;
            vcmp_gt(greater, candidate, best, all);
            vcmp_eq(equal, candidate, best, all);
            vcmp_lt(earlier, candidate_id, best_id, all);
            pand(equal, equal, earlier, all);
            por(choose, greater, equal, all);
            vsel(best, candidate, best, choose);
            vsel(best_id, candidate_id, best_id, choose);
        }
        RegTensor<uint32_t> row_index, col_index, mask_bits;
        vshrs(row_index, lane, 3, all, MODE_ZEROING);
        RegTensor<int32_t> selected, zeros, result;
        vselr((RegTensor<uint32_t>&)selected, (RegTensor<uint32_t>&)best_id, row_index);
        vdup(mask_bits, 7u, all, MODE_ZEROING);
        vand(col_index, lane, mask_bits, all, MODE_ZEROING);
        MaskReg first_column;
        vcmps_eq(first_column, col_index, 0u, all);
        vdup(zeros, 0, all, MODE_ZEROING);
        vsel(result, selected, zeros, first_column);
        vsts(result, destination, 0, NORM_B32, all);
    }
}

__aicore__ inline void merge_partials(__gm__ int64_t* args) {
    int worker = get_block_idx(args);
    auto output = reinterpret_cast<__gm__ Tensor*>(args[2]);
    int groups = (output->shapes[0] + 7) / 8;
    for (int group = worker; group < groups; group += get_block_num(args)) {
        int row = group * 8;
        int rows = output->shapes[0] - row < 8 ? output->shapes[0] - row : 8;
        PartialScores scores;
        PartialIds indices;
        OutputIds ids(rows);
        TASSIGN(scores, 0);
        TASSIGN(indices, 1024);
        TASSIGN(ids, 2048);
        using S = TileShape2D<float, 32, 8>;
        using I = TileShape2D<int32_t, 32, 8>;
        using B = BaseShape2D<float, 32, 80>;
        using IB = BaseShape2D<int32_t, 32, 80>;
        GlobalTensor<float, S, B> sg(data<float>(args, 0) + row);
        GlobalTensor<int32_t, I, IB> ig(data<int32_t>(args, 1) + row);
        TLOAD(scores, sg);
        TLOAD(indices, ig);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        merge_values(ids.data(), scores.data(), indices.data());
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        using OS = TileShape2D<int32_t, DYNAMIC, 8>;
        using OB = BaseShape2D<int32_t, DYNAMIC, DYNAMIC>;
        GlobalTensor<int32_t, OS, OB> out(
            data<int32_t>(args, 2) + row * output->strides[0], OS(rows, 8), OB(rows, output->strides[0]));
        TSTORE(out, ids);
        pipe_barrier(PIPE_ALL);
    }
}


#endif
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_VEC__)
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(-1, -1);
    merge_partials(args);
#endif
}
