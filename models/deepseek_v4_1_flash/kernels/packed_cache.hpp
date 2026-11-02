// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#pragma once

#include <pto/pto-inst.hpp>

namespace packed_cache {
using namespace pto;

template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C>
using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;

#if defined(__DAV_VEC__)
template <int GroupShift, typename Values, typename Scales>
__tf__ __aicore__ inline void multiply_groups(
    typename Values::TileDType __out__ dst, typename Values::TileDType __in__ src,
    typename Scales::TileDType __in__ scales) {
    __ubuf__ float* output = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    __ubuf__ float* input = (__ubuf__ float*)__cce_get_tile_ptr(src);
    __ubuf__ float* factors = (__ubuf__ float*)__cce_get_tile_ptr(scales);
#if defined(__CPU_SIM)
    for (int element = 0; element < Values::Rows * Values::Cols; ++element) {
        output[element] = input[element] * factors[element >> GroupShift];
    }
#else
    __VEC_SCOPE__ {
        RegTensor<uint32_t> index;
        RegTensor<float> value, factor, result;
        MaskReg full = pset_b32(PAT_ALL);
        vci((vector_s32&)index, 0, INC_ORDER);
        vshrs(index, index, (uint32_t)GroupShift, full, MODE_ZEROING);
        for (uint16_t chunk = 0; chunk < 32 * 512 / 64; ++chunk) {
            vlds(value, input, chunk * 64, NORM);
            vgather2(factor, factors + chunk * (64 >> GroupShift), index, full);
            vmul(result, value, factor, full, MODE_ZEROING);
            vsts(result, output, chunk * 64, NORM_B32, full);
        }
    }
#endif
}

template <typename T, int PayloadColumns, int ScaleColumns>
__aicore__ inline void load_rows(
    __gm__ T* payload, __gm__ bfloat16_t* scales, Vec<int32_t, 1, 32>& ids) {
    for (int lane = 0; lane < 32; ++lane) {
        int physical = ids.GetValue(lane);
        if (physical < 0) physical = 0;
        Vec<T, 1, PayloadColumns> row;
        Vec<bfloat16_t, 1, ScaleColumns> factor;
        TASSIGN(row, lane * PayloadColumns * sizeof(T));
        TASSIGN(factor, 16384 + lane * ScaleColumns * sizeof(bfloat16_t));
        GM<T, 1, PayloadColumns> input(payload + (int64_t)physical * PayloadColumns);
        GM<bfloat16_t, 1, ScaleColumns> scale(scales + (int64_t)physical * ScaleColumns);
        TLOAD(row, input);
        TLOAD(factor, scale);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

template <int FactorsAddress = 18432>
__aicore__ inline void decode_window(
    __gm__ float8_e4m3_t* input, __gm__ bfloat16_t* scale, Vec<int32_t, 1, 32>& ids,
    Vec<bfloat16_t, 32, 512>& output) {
    Vec<float8_e4m3_t, 32, 512> payload;
    Vec<bfloat16_t, 32, 16> scales;
    Vec<float, 32, 16> factors;
    Vec<float, 32, 512> values;
    TASSIGN(payload, 0);
    TASSIGN(scales, 16384);
    TASSIGN(factors, FactorsAddress);
    TASSIGN(values, 65536);
    load_rows<float8_e4m3_t, 512, 16>(input, scale, ids);
    TCVT(factors, scales, RoundMode::CAST_RINT);
    TCVT(values, payload, RoundMode::CAST_RINT);
    multiply_groups<5, decltype(values), decltype(factors)>(values.data(), values.data(), factors.data());
    TCVT(output, values, RoundMode::CAST_RINT);
}

template <int FactorsAddress = 18432>
__aicore__ inline void decode_compressed(
    __gm__ uint8_t* input, __gm__ bfloat16_t* scale, Vec<int32_t, 1, 32>& ids,
    Vec<bfloat16_t, 32, 512>& output) {
    Vec<float4_e2m1x2_t, 32, 512> payload;
    Vec<bfloat16_t, 32, 32> scales;
    Vec<float, 32, 32> factors;
    Vec<float, 32, 512> values;
    TASSIGN(payload, 0);
    TASSIGN(scales, 16384);
    TASSIGN(factors, FactorsAddress);
    TASSIGN(values, 65536);
    load_rows<uint8_t, 256, 32>(input, scale, ids);
    TCVT(factors, scales, RoundMode::CAST_RINT);
    TCVT(output, payload, RoundMode::CAST_RINT);
    TCVT(values, output, RoundMode::CAST_RINT);
    multiply_groups<4, decltype(values), decltype(factors)>(values.data(), values.data(), factors.data());
    TCVT(output, values, RoundMode::CAST_RINT);
}
#endif
} // namespace packed_cache
