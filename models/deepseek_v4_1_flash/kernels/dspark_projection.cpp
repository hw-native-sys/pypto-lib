// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// DSpark projection with fused MXFP8 activation quantization.
#include "mx_quant.hpp"
namespace fast_quant {
using namespace pto;
using namespace mx_quant;
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
template <int C, typename Input>
__tf__ __aicore__ inline void quantize_fast(
    typename Vec<uint8_t, 16, C>::TileDType __out__ output,
    typename Vec<uint8_t, 1, 16 * C / 32>::TileDType __out__ scales,
    typename Input::TileDType __in__ input, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(output);
    auto scale = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all16 = pset_b16(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t four_count = 4;
        MaskReg four = CreatePredicate<uint16_t>(four_count);
        RegTensor<uint16_t> half_lane, group_index;
        vci((RegTensor<int16_t>&)half_lane, 0, INC_ORDER);
        vshrs(group_index, half_lane, (uint16_t)4, all16, MODE_ZEROING);
        RegTensor<uint16_t> exponent_mask, exponent_floor, inverse_bias, zero16;
        vdup(exponent_mask, (uint16_t)0x7f80, all16, MODE_ZEROING);
        vdup(exponent_floor, (uint16_t)(8 << 7), all16, MODE_ZEROING);
        vdup(inverse_bias, (uint16_t)(254 << 7), all16, MODE_ZEROING);
        vdup(zero16, (uint16_t)0, all16, MODE_ZEROING);
        RegTensor<uint16_t> scatter_offset;
        vshls(scatter_offset, half_lane, (uint16_t)4, all16, MODE_ZEROING);
        RegTensor<float> zero;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint16_t chunk = 0; chunk < C / 256; ++chunk) {
                RegTensor<float> x0, x1, x2, x3;
                RegTensor<bfloat16_t> even, odd;
                vlds(even, odd, src, row * C + chunk * 256, DINTLV_B16);
                RegTensor<uint16_t> e0_bits, e1_bits, maximum, shared, codes;
                // OCP shared exponents use the maximum BF16 exponent in each group of 32.
                vand(e0_bits, (RegTensor<uint16_t>&)even, exponent_mask, all16);
                vand(e1_bits, (RegTensor<uint16_t>&)odd, exponent_mask, all16);
                vmax(maximum, e0_bits, e1_bits, all16, MODE_ZEROING);
                vcgmax(maximum, maximum, all16, MODE_ZEROING);
                vmax(maximum, maximum, exponent_floor, all16, MODE_ZEROING);
                vsub(shared, maximum, exponent_floor, all16, MODE_ZEROING);
                vshrs(codes, shared, (uint16_t)7, all16, MODE_ZEROING);
                RegTensor<uint16_t> inverse, expanded_inverse;
                vsub(inverse, inverse_bias, shared, all16, MODE_ZEROING);
                MaskReg nonzero;
                vcmps_gt(nonzero, shared, (uint16_t)0, all16);
                vsel(inverse, inverse, zero16, nonzero);
                vselr(expanded_inverse, inverse, group_index);
                vmul(even, even, (RegTensor<bfloat16_t>&)expanded_inverse, all16, MODE_ZEROING);
                vmul(odd, odd, (RegTensor<bfloat16_t>&)expanded_inverse, all16, MODE_ZEROING);
                // Clamp normalized BF16 values to the finite E4M3 range.
                vmaxs(even, even, bfloat16_t(-448), all16, MODE_ZEROING);
                vmins(even, even, bfloat16_t(448), all16, MODE_ZEROING);
                vmaxs(odd, odd, bfloat16_t(-448), all16, MODE_ZEROING);
                vmins(odd, odd, bfloat16_t(448), all16, MODE_ZEROING);
                vcvt(x0, even, all16, PART_EVEN);
                vcvt(x2, even, all16, PART_ODD);
                vcvt(x1, odd, all16, PART_EVEN);
                vcvt(x3, odd, all16, PART_ODD);
                vector_f8e4m3 p0, p1, p2, p3;
                vcvt(p0, x0, all, ROUND_R, RS_DISABLE, PART_P0);
                vcvt(p1, x1, all, ROUND_R, RS_DISABLE, PART_P1);
                vcvt(p2, x2, all, ROUND_R, RS_DISABLE, PART_P2);
                vcvt(p3, x3, all, ROUND_R, RS_DISABLE, PART_P3);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, all8);
                vor((RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, all8);
                vor((RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p2, all8);
                vsts((RegTensor<uint8_t>&)p0, dst, row * C + chunk * 256, NORM_B8, all8);
                RegTensor<uint8_t> packed_codes;
                vpack(packed_codes, codes, LOWER);
                auto scale_base = scale + row + chunk * 64;
                vscatter((RegTensor<uint16_t>&)packed_codes, scale_base, scatter_offset, four);
            }
        }
        for (uint16_t row = rows; row < Input::Rows; ++row) {
            for (uint16_t chunk = 0; chunk < C / 256; ++chunk) {
                vsts((RegTensor<uint8_t>&)zero, dst, row * C + chunk * 256, NORM_B8, all8);
                auto scale_base = scale + row + chunk * 64;
                vscatter((RegTensor<uint16_t>&)zero, scale_base, scatter_offset, four);
            }
        }
    }
}
#endif
}

using namespace mx_quant;
namespace {
#if defined(__DAV_VEC__)
constexpr int width = 15360, columns = 1280;
using Input = Tile<TileType::Vec, bfloat16_t, 16, columns, BLayout::RowMajor, DYNAMIC, columns>;
using Shape = TileShape2D<bfloat16_t, DYNAMIC, columns>;
using GlobalStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
__aicore__ inline int valid_rows(int active, int work) {
    int rows = active - work / (width / columns) * 16;
    return rows < 0 ? 0 : (rows > 16 ? 16 : rows);
}
__aicore__ inline void load_work(__gm__ int64_t* args, int work, int slot) {
    int rows = valid_rows(args[6], work);
    if (rows > 0) {
        Input input(rows);
        TASSIGN(input, slot * 62464);
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        GlobalTensor<bfloat16_t, Shape, GlobalStride> source(data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + k0,
            Shape(rows, columns), GlobalStride(1, 1, 1, stride(args, 0), 1));
        TLOAD(input, source);
    }
}
#endif
}
static __aicore__ void quant_impl(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int first = get_block_idx(args) * 2 + get_sub_block_id(args), workers = get_block_num(args) * 2;
    int total = args[7] / 16 * (width / columns);
    if (first >= total) return;
    load_work(args, first, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // Each UB slot contains input, payload, and packed scales.
    // V-to-MTE2 credits protect input reuse; MTE3-to-V credits protect output reuse.
    int iteration = 0;
    for (int work = first; work < total; work += workers, ++iteration) {
        int slot = iteration % 2;
        event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
        if (work + workers < total) {
            if (iteration > 0) wait_flag(PIPE_V, PIPE_MTE2, next);
            load_work(args, work + workers, 1 - slot);
            set_flag(PIPE_MTE2, PIPE_V, next);
        }
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        int rows = valid_rows(args[6], work);
        Input input(rows);
        Vec<uint8_t, 16, columns> output;
        Vec<uint8_t, 1, 16 * columns / 32> scales;
        TASSIGN(input, slot * 62464);
        TASSIGN(output, slot * 62464 + 40960);
        TASSIGN(scales, slot * 62464 + 61440);
        wait_flag(PIPE_MTE2, PIPE_V, current);
        wait_flag(PIPE_MTE3, PIPE_V, current);
        fast_quant::quantize_fast<columns, Input>(output.data(), scales.data(), input.data(), rows);
        set_flag(PIPE_V, PIPE_MTE2, current);
        set_flag(PIPE_V, PIPE_MTE3, current);
        wait_flag(PIPE_V, PIPE_MTE3, current);
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 16, columns>, GlobalStride> target(
            data<uint8_t>(args, 1) + t0 * width + k0,
            TileShape2D<uint8_t, 16, columns>(), GlobalStride(1, 1, 1, width, 1));
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 1, 640>, BaseShape2D<uint8_t, 1, 640>> scale_target(
            data<uint8_t>(args, 2) + t0 * (width / 32) + k0 / columns * 640);
        TSTORE(target, output);
        TSTORE(scale_target, scales);
        set_flag(PIPE_MTE3, PIPE_V, current);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (iteration > 1) wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
#endif
}

#include "mxfp8_projection.hpp"
#include "mixed_entry.hpp"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    native_lifecycle::enter_dual_aiv();
#if defined(__DAV_VEC__)
    quant_impl(args);
    SYNCALL<SyncCoreType::AIVOnly>();
    set_intra_block(PIPE_V, 10);
#elif defined(__DAV_CUBE__)
    wait_intra_block(PIPE_FIX, 10);
    wait_intra_block(PIPE_FIX, 26);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
    using namespace mxfp8_projection;
    constexpr int input_width = 15360, output_width = 5120, n_tile = 160, m_tile = 80;
    int active = static_cast<int>(args[6]);
    int worker = get_block_idx(args), workers = get_block_num(args);
    for (int n = worker * n_tile; n < output_width; n += workers * n_tile) {
        for (int row = 0; row < active; row += m_tile) {
            int rows = active - row < m_tile ? active - row : m_tile;
            project<input_width, output_width, n_tile, m_tile>(
                mxfp8_projection::data<F>(args, 1) + row * input_width, mxfp8_projection::data<F>(args, 3),
                mxfp8_projection::data<S>(args, 2) + row * (input_width / 32), mxfp8_projection::data<S>(args, 4),
                mxfp8_projection::data<bfloat16_t>(args, 5) + row * output_width, rows, output_width, n);
        }
    }
#endif
}
