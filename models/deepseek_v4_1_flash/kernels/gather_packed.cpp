// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Packed FP8/FP4 cache gather and BF16 dequantization.
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

__aicore__ inline void decode_window(__gm__ int64_t* args, Vec<int32_t, 1, 32>& ids,
                                    Vec<bfloat16_t, 32, 512>& output) {
    Vec<float8_e4m3_t, 32, 512> payload;
    Vec<bfloat16_t, 32, 16> scales;
    Vec<float, 32, 16> factors;
    Vec<float, 32, 512> values;
    TASSIGN(payload, 0);
    TASSIGN(scales, 16384);
    TASSIGN(factors, 131072);
    TASSIGN(values, 65536);
    load_rows<float8_e4m3_t, 512, 16>(data<float8_e4m3_t>(args, 0), data<bfloat16_t>(args, 1), ids);
    TCVT(factors, scales, RoundMode::CAST_RINT);
    TCVT(values, payload, RoundMode::CAST_RINT);
    multiply_groups<5, decltype(values), decltype(factors)>(values.data(), values.data(), factors.data());
    TCVT(output, values, RoundMode::CAST_RINT);
}

__aicore__ inline void decode_compressed(__gm__ int64_t* args, Vec<int32_t, 1, 32>& ids,
                                        Vec<bfloat16_t, 32, 512>& output) {
    Vec<float4_e2m1x2_t, 32, 512> payload;
    Vec<bfloat16_t, 32, 32> scales;
    Vec<float, 32, 32> factors;
    Vec<float, 32, 512> values;
    TASSIGN(payload, 0);
    TASSIGN(scales, 16384);
    TASSIGN(factors, 131072);
    TASSIGN(values, 65536);
    load_rows<uint8_t, 256, 32>(data<uint8_t>(args, 2), data<bfloat16_t>(args, 3), ids);
    TCVT(factors, scales, RoundMode::CAST_RINT);
    TCVT(output, payload, RoundMode::CAST_RINT);
    TCVT(values, output, RoundMode::CAST_RINT);
    multiply_groups<4, decltype(values), decltype(factors)>(values.data(), values.data(), factors.data());
    TCVT(output, values, RoundMode::CAST_RINT);
}

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
        if (column < 128) decode_window(args, ids, output);
        else decode_compressed(args, ids, output);
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
