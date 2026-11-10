// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Shared-query FP8 window gather with guarded contiguous-row loads.
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


__aicore__ inline void decode_rows(__gm__ int64_t* args, Vec<int32_t, 1, 32>& ids,
                                   Vec<bfloat16_t, 32, 512>& output) {
    Vec<float8_e4m3_t, 32, 512> payload;
    Vec<uint8_t, 32, 512> payload_bytes;
    Vec<bfloat16_t, 32, 16> scales;
    Vec<float, 32, 16> factors;
    Vec<float, 32, 512> values;
    TASSIGN(payload, 0); TASSIGN(payload_bytes, 0); TASSIGN(scales, 16384);
    TASSIGN(factors, 131072); TASSIGN(values, 65536);
    int first = ids.GetValue(0);
    bool contiguous = first >= 0, any_valid = false;
    for (int lane = 0; lane < 32; ++lane) {
        int physical = ids.GetValue(lane);
        contiguous = contiguous && physical == first + lane;
        any_valid = any_valid || physical >= 0;
    }
    if (!any_valid) {
        TEXPANDS(output, static_cast<bfloat16_t>(0.0f));
        return;
    }
    auto cache = data<uint8_t>(args, 0);
    if (contiguous) {
        GlobalTensor<float8_e4m3_t, TileShape2D<float8_e4m3_t, 32, 512>, BaseShape2D<float8_e4m3_t, 32, 544>> p(
            reinterpret_cast<__gm__ float8_e4m3_t*>(cache) + static_cast<int64_t>(first) * 544);
        GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, 32, 16>, BaseShape2D<bfloat16_t, 32, 272>> s(
            reinterpret_cast<__gm__ bfloat16_t*>(cache + 512) + static_cast<int64_t>(first) * 272);
        TLOAD(payload, p); TLOAD(scales, s);
    } else {
        TEXPANDS(payload_bytes, static_cast<uint8_t>(0));
        TEXPANDS(scales, static_cast<bfloat16_t>(0.0f));
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        for (int lane = 0; lane < 32; ++lane) {
            int physical = ids.GetValue(lane);
            if (physical >= 0) {
                Vec<float8_e4m3_t, 1, 512> p;
                Vec<bfloat16_t, 1, 16> s;
                TASSIGN(p, lane * 512); TASSIGN(s, 16384 + lane * 32);
                GM<float8_e4m3_t, 1, 512> p_gm(
                    reinterpret_cast<__gm__ float8_e4m3_t*>(cache) + static_cast<int64_t>(physical) * 544);
                GM<bfloat16_t, 1, 16> s_gm(
                    reinterpret_cast<__gm__ bfloat16_t*>(cache + 512) + static_cast<int64_t>(physical) * 272);
                TLOAD(p, p_gm); TLOAD(s, s_gm);
            }
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(factors, scales, RoundMode::CAST_RINT);
    TCVT(values, payload, RoundMode::CAST_RINT);
    multiply_groups<5, decltype(values), decltype(factors)>(values.data(), values.data(), factors.data());
    TCVT(output, values, RoundMode::CAST_RINT);
}

__aicore__ inline void gather(__gm__ int64_t* args) {
    Vec<int32_t, 1, 32> ids, other_ids;
    Vec<bfloat16_t, 32, 512> output;
    TASSIGN(ids, 135168); TASSIGN(other_ids, 135296); TASSIGN(output, 32768);
    int tokens = args[3], groups = (tokens + 4) / 5;
    int worker = get_block_idx(args), workers = get_block_num(args);
    auto index_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto output_tensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
    for (int job = worker; job < groups * 6; job += workers) {
        int first = job / 6 * 5, column = job % 6 * 32;
        int count = tokens - first < 5 ? tokens - first : 5;
        GM<int32_t, 1, 32> first_ids(data<int32_t>(args, 1) + first * index_tensor->strides[0] + column);
        TLOAD(ids, first_ids);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        bool shared = true;
        for (int query = 1; query < count; ++query) {
            GM<int32_t, 1, 32> next_ids(data<int32_t>(args, 1) + (first + query) * index_tensor->strides[0] + column);
            TLOAD(other_ids, next_ids);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            for (int lane = 0; lane < 32; ++lane) shared = shared && ids.GetValue(lane) == other_ids.GetValue(lane);
        }
        int decodes = shared ? 1 : count;
        for (int decoded = 0; decoded < decodes; ++decoded) {
            if (decoded > 0) {
                GM<int32_t, 1, 32> next_ids(data<int32_t>(args, 1) + (first + decoded) * index_tensor->strides[0] + column);
                TLOAD(ids, next_ids);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            }
            decode_rows(args, ids, output);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            int copies = shared ? count : 1;
            for (int copy = 0; copy < copies; ++copy) {
                int token = first + (shared ? copy : decoded);
                GM<bfloat16_t, 32, 512> target(data<bfloat16_t>(args, 2) + token * output_tensor->strides[0] + column * 512);
                TSTORE(target, output);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    gather(args);
#endif
}
