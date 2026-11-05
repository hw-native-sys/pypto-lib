// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Two-row RMSNorm with FP32 accumulation and BF16 output.
#include <cmath>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace {
constexpr int Rows = 2;
constexpr int Width = 5120;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using Input = Vec<bfloat16_t, Rows, Width>;
using Weight = Vec<bfloat16_t, 1, Width>;
using Output = Vec<bfloat16_t, Rows, Width>;
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
#if defined(__DAV_VEC__) && (defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM))
__tf__ __aicore__ inline void normalize(Output::TileDType __out__ output,
    Input::TileDType __in__ input, Weight::TileDType __in__ weight, uint16_t valid) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
#if defined(__CPU_SIM)
    for (int row = 0; row < Rows; ++row) {
        // Avoid serial FP32 reduction error crossing a BF16 rounding boundary.
        double sum = 0.0;
        for (int k = 0; k < Width; ++k) {
            float value = static_cast<float>(src[row * Width + k]);
            sum += value * value;
        }
        float inverse = 1.0f / std::sqrt(static_cast<float>(sum) / Width + 1.0e-6f);
        for (int k = 0; k < Width; ++k) {
            float value = static_cast<float>(src[row * Width + k]) * inverse;
            value *= static_cast<float>(gamma[k]);
            dst[row * Width + k] = row < valid ? static_cast<bfloat16_t>(value) : 0.0f;
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < Rows; ++row) {
            RegTensor<float> sum, sum1, sum2, sum3, inverse, one;
            vdup(sum, 0.0f, all, MODE_ZEROING);
            vdup(sum1, 0.0f, all, MODE_ZEROING);
            vdup(sum2, 0.0f, all, MODE_ZEROING);
            vdup(sum3, 0.0f, all, MODE_ZEROING);
            for (uint16_t k = 0; k < Width / 256; ++k) {
                RegTensor<bfloat16_t> packed0;
                RegTensor<float> value0, square0;
                vlds(packed0, src, row * Width + k * 256 + 0, UNPK_B16);
                vcvt(value0, packed0, all, PART_EVEN);
                vmul(square0, value0, value0, all, MODE_ZEROING);
                vadd(sum, sum, square0, all, MODE_ZEROING);
                RegTensor<bfloat16_t> packed1;
                RegTensor<float> value1, square1;
                vlds(packed1, src, row * Width + k * 256 + 64, UNPK_B16);
                vcvt(value1, packed1, all, PART_EVEN);
                vmul(square1, value1, value1, all, MODE_ZEROING);
                vadd(sum1, sum1, square1, all, MODE_ZEROING);
                RegTensor<bfloat16_t> packed2;
                RegTensor<float> value2, square2;
                vlds(packed2, src, row * Width + k * 256 + 128, UNPK_B16);
                vcvt(value2, packed2, all, PART_EVEN);
                vmul(square2, value2, value2, all, MODE_ZEROING);
                vadd(sum2, sum2, square2, all, MODE_ZEROING);
                RegTensor<bfloat16_t> packed3;
                RegTensor<float> value3, square3;
                vlds(packed3, src, row * Width + k * 256 + 192, UNPK_B16);
                vcvt(value3, packed3, all, PART_EVEN);
                vmul(square3, value3, value3, all, MODE_ZEROING);
                vadd(sum3, sum3, square3, all, MODE_ZEROING);
            }
            vadd(sum, sum, sum1, all, MODE_ZEROING);
            vadd(sum2, sum2, sum3, all, MODE_ZEROING);
            vadd(sum, sum, sum2, all, MODE_ZEROING);
            vcadd(sum, sum, all, MODE_ZEROING);
            vmuls(inverse, sum, 1.0f / Width, all, MODE_ZEROING);
            vadds(inverse, inverse, 1.0e-6f, all, MODE_ZEROING);
            vsqrt(inverse, inverse, all, MODE_ZEROING);
            vdup(one, 1.0f, all, MODE_ZEROING);
            vdiv(inverse, one, inverse, all, MODE_ZEROING);
            vdup(inverse, inverse, all, POS_LOWEST, MODE_ZEROING);
            for (uint16_t k = 0; k < Width / 64; ++k) {
                RegTensor<bfloat16_t> packed;
                RegTensor<float> value, gain;
                vlds(packed, src, row * Width + k * 64, UNPK_B16);
                vcvt(value, packed, all, PART_EVEN);
                vlds(packed, gamma, k * 64, UNPK_B16);
                vcvt(gain, packed, all, PART_EVEN);
                vmul(value, value, inverse, all, MODE_ZEROING);
                vmul(value, value, gain, all, MODE_ZEROING);
                if (row >= valid) vdup(value, 0.0f, all, MODE_ZEROING);
                vcvt(packed, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(packed, dst, row * Width + k * 64, PK_B32, all);
            }
        }
    }
#endif
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
#endif
    int first = get_block_idx(args) * Rows;
    int valid = args[3] - first;
    valid = valid < 0 ? 0 : (valid > Rows ? Rows : valid);
    Input input;
    Weight weight;
    Output output;
    TASSIGN(input, 0); TASSIGN(weight, 20480); TASSIGN(output, 30720);
    GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, Rows, Width>, BaseShape2D<bfloat16_t, Rows, Width>> source(data<bfloat16_t>(args, 0) + first * Width);
    GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 1, Width>, BaseShape2D<bfloat16_t, 1, Width>> gamma(data<bfloat16_t>(args, 1));
    GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, Rows, Width>, BaseShape2D<bfloat16_t, Rows, Width>> target(data<bfloat16_t>(args, 2) + first * Width);
    TLOAD(input, source); TLOAD(weight, gamma);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    normalize(output.data(), input.data(), weight.data(), valid);
#else
    Vec<float, Rows, Width> value, square;
    Vec<float, 1, Width> gain;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, DYNAMIC, 1> inverse(Rows);
    Vec<float, Rows, 128> scratch;
    TASSIGN(value, 73728); TASSIGN(square, 114688);
    TASSIGN(gain, 155648); TASSIGN(inverse, 176128); TASSIGN(scratch, 176160);
    TCVT(value, input, RoundMode::CAST_NONE);
    TCVT(gain, weight, RoundMode::CAST_NONE);
    TMUL(square, value, value);
    TROWSUM(inverse, square, scratch);
    TMULS(inverse, inverse, 1.0f / Width);
    TADDS(inverse, inverse, 1.0e-6f);
    TRSQRT(inverse, inverse);
    TROWEXPANDMUL(value, value, inverse);
    TCOLEXPANDMUL(value, value, gain);
    TCVT(input, value, RoundMode::CAST_RINT);
    TMOV(output, input);
    if (valid < Rows) {
        Vec<bfloat16_t, 1, Width> zero;
        TASSIGN(zero, 30720 + valid * Width * sizeof(bfloat16_t));
        for (int row = valid; row < Rows; ++row) {
            TASSIGN(zero, 30720 + row * Width * sizeof(bfloat16_t));
            TEXPANDS(zero, 0.0f);
        }
    }
#endif
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(target, output);
    pipe_barrier(PIPE_ALL);
#endif
}
