// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 vocabulary projection and exact local maximum selection.
#include "bf16_projection.hpp"
#include "mixed_entry.hpp"

using namespace pto;
using namespace bf16_projection;
template <int NT>
using Raw = Tile<TileType::Vec, float, 40, 512, BLayout::RowMajor, 40, NT>;
using State = Tile<TileType::Vec, float, 1, 64>;
using IndexState = Tile<TileType::Vec, int32_t, 1, 64>;
using Widths = Tile<TileType::Vec, int32_t, 12, 8, BLayout::RowMajor, 12, 1>;

#if defined(__DAV_CUBE__)
struct Transfer {
    static constexpr bool full_rows = true;
    int count = 0;

    template <typename Acc>
    __aicore__ inline void operator()(Acc& acc, int, int, int) {
        int slot = count % 2;
        wait_intra_block(PIPE_FIX, slot * 2 + 1);
        wait_intra_block(PIPE_FIX, slot * 2 + 17);
        Raw<Acc::Cols> raw;
        TASSIGN(raw, slot * 81920);
        TMOV<Raw<Acc::Cols>, Acc, AccToVecMode::DualModeSplitM>(raw, acc);
        set_intra_block(PIPE_FIX, slot * 2);
        set_intra_block(PIPE_FIX, slot * 2 + 16);
        ++count;
    }

    __aicore__ inline void finish() {
        for (int slot = 0; slot < 2; ++slot) {
            wait_intra_block(PIPE_FIX, slot * 2 + 1);
            wait_intra_block(PIPE_FIX, slot * 2 + 17);
        }
        set_intra_block(PIPE_FIX, 4);
        set_intra_block(PIPE_FIX, 20);
        wait_intra_block(PIPE_S, 6);
        wait_intra_block(PIPE_S, 22);
    }
};
#endif

#if defined(__DAV_VEC__)
template <int NT>
__tf__ __aicore__ inline void reduce_panel(
    State::TileDType __out__ state, IndexState::TileDType __out__ indices,
    typename Raw<NT>::TileDType __in__ raw, Widths::TileDType __in__ widths, uint32_t start, uint16_t rows, uint16_t row_base, bool first) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(raw);
    auto width_ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(widths);
    auto values = (__ubuf__ float*)__cce_get_tile_ptr(state);
    auto ids = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        uint32_t count = 1;
        MaskReg one = CreatePredicate<float>(count);
        RegTensor<int32_t> invalid, zero_id;
        RegTensor<float> zero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(zero_id, 0, all, MODE_ZEROING);
        vdup(invalid, 2147483647, all, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> best;
            RegTensor<int32_t> best_id, current_id;
            vdup((RegTensor<int32_t>&)best, -8388608, all, MODE_ZEROING);
            vci(best_id, static_cast<int32_t>(start), INC_ORDER);
            vmov(current_id, best_id, all);
            uint32_t remaining = NT;
            for (uint16_t segment = 0; segment < (NT + 63) / 64; ++segment) {
                MaskReg valid = CreatePredicate<float>(remaining);
                RegTensor<float> current;
                vlds(current, src, row * 512 + segment * 64, NORM);
                MaskReg better;
                vcmp_gt(better, current, best, valid);
                pand(better, better, valid, all);
                vsel(best, current, best, better);
                vsel(best_id, current_id, best_id, better);
                vadds(current_id, current_id, 64, all, MODE_ZEROING);
            }
            RegTensor<float> maximum;
            vcmax(maximum, best, all, MODE_ZEROING);
            vdup(maximum, maximum, all, POS_LOWEST, MODE_ZEROING);
            MaskReg equal;
            vcmp_eq(equal, best, maximum, all);
            RegTensor<int32_t> selected;
            vsel(selected, best_id, invalid, equal);
            vcmin(selected, selected, all, MODE_ZEROING);
            if (!first) {
                RegTensor<float> previous;
                RegTensor<int32_t> previous_id;
                vlds(previous, values, row, BRC_B32);
                vlds(previous_id, ids, row, BRC_B32);
                MaskReg better;
                vcmp_gt(better, maximum, previous, one);
                vsel(maximum, maximum, previous, better);
                vsel(selected, selected, previous_id, better);
            }
            RegTensor<int32_t> width;
            int global_row = row_base + row;
            vlds(width, width_ptr, global_row / 6 * 8, BRC_B32);
            MaskReg active;
            vcmps_gt(active, width, global_row % 6, one);
            vsel(maximum, maximum, zero, active);
            vsel(selected, selected, zero_id, active);
            vsts(maximum, values, row, ONEPT_B32, one);
            vsts(selected, ids, row, ONEPT_B32, one);
        }
    }
}

__tf__ __aicore__ inline void initialize(State::TileDType __out__ state, IndexState::TileDType __out__ indices) {
    auto values = (__ubuf__ float*)__cce_get_tile_ptr(state);
    auto ids = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<float> zero;
        RegTensor<int32_t> izero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(izero, 0, all, MODE_ZEROING);
        vsts(zero, values, 0, NORM_B32, all);
        vsts(izero, ids, 0, NORM_B32, all);
    }
}

__aicore__ inline void vector_sample(__gm__ int64_t* args) {
    int block = get_block_idx(args), workers = get_block_num(args);
    int lane = get_sub_block_id(args);
    int panels = (129280 / 512 + workers - 1 - block) / workers;
    bool tail = block == (129280 / 512) % workers;
    int available = reinterpret_cast<__gm__ Tensor*>(args[0])->shapes[0];
    available = available < args[5] ? available : args[5];
    int rows = available - lane * 40;
    rows = rows < 40 ? rows : 40;
    rows = rows > 0 ? rows : 0;
    State state;
    IndexState indices;
    TASSIGN(state, 163840);
    TASSIGN(indices, 164096);
    Widths widths;
    TASSIGN(widths, 164352);
    auto metadata = reinterpret_cast<__gm__ Tensor*>(args[4]);
    for (int request = 0; request < 12; ++request) {
        Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1> value;
        TASSIGN(value, 164352 + request * 32);
        GlobalTensor<int32_t, TileShape2D<int32_t, 1, 1>, BaseShape2D<int32_t, 1, 1>> gm(
            data<int32_t>(args, 4) + request * metadata->strides[0]);
        TLOAD(value, gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    initialize(state.data(), indices.data());
    for (int slot = 0; slot < 2; ++slot) {
        set_intra_block(PIPE_V, slot * 2 + 1);
    }
    for (int panel = 0; panel < panels + static_cast<int>(tail); ++panel) {
        int slot = panel % 2;
        wait_intra_block(PIPE_V, slot * 2);
        if (panel < panels) {
            Raw<512> raw;
            TASSIGN(raw, slot * 81920);
            reduce_panel<512>(state.data(), indices.data(), raw.data(), widths.data(), (block + panel * workers) * 512, rows, lane * 40, panel == 0);
        } else {
            Raw<256> raw;
            TASSIGN(raw, slot * 81920);
            reduce_panel<256>(state.data(), indices.data(), raw.data(), widths.data(), 129024, rows, lane * 40, panel == 0);
        }
        set_intra_block(PIPE_V, slot * 2 + 1);
    }
    wait_intra_block(PIPE_V, 4);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using Values = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
    using Indices = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
    Values result(40);
    Indices result_ids(40);
    TASSIGN(result, 163840);
    TASSIGN(result_ids, 164096);
    using Shape = TileShape2D<float, 1, DYNAMIC>;
    using IdShape = TileShape2D<int32_t, 1, DYNAMIC>;
    using Stride = pto::Stride<1, 1, 1, 80, 1>;
    GlobalTensor<float, Shape, Stride> output(data<float>(args, 2) + block * 80 + lane * 40, Shape(1, 40));
    GlobalTensor<int32_t, IdShape, Stride> ids(data<int32_t>(args, 3) + block * 80 + lane * 40, IdShape(1, 40));
    TSTORE(output, result);
    TSTORE(ids, result_ids);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, 6);
}
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    native_lifecycle::enter_dual_aiv();
#if defined(__DAV_CUBE__)
    auto hidden = reinterpret_cast<__gm__ Tensor*>(args[0]);
    Transfer transfer;
    project<5120, 129280, 512, 128, 32, 3, Layout::NZ>(
        data<BFloat>(args, 0), data<BFloat>(args, 1), args[5], hidden->shapes[0],
        hidden->strides[0], 0, M_TILE, get_block_idx(args), get_block_num(args), transfer);
    transfer.finish();
#else
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(-1, -1);
    vector_sample(args);
#endif
}
