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

namespace markov_native {
using namespace pto;
constexpr int D = 5120;
constexpr int K = 256;
template <typename T, int C> using Vec = Tile<TileType::Vec, T, 1, C>;
template <typename T, int C> using GM = GlobalTensor<T, TileShape2D<T, 1, C>, pto::Stride<1, 1, 1, C, 1>>;
#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void finalize_vf(
    Vec<int32_t, 32>::TileDType __out__ output_ids,
    Vec<float, 16>::TileDType __out__ output_confidence,
    Vec<float, 512>::TileDType __in__ scores,
    Vec<int32_t, 512>::TileDType __in__ indices,
    Vec<bfloat16_t, D>::TileDType __in__ hidden,
    Vec<bfloat16_t, K>::TileDType __in__ embedding,
    Vec<float, D + K>::TileDType __in__ weight, int32_t workers) {
    auto sp = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto ip = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto hp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(hidden);
    auto ep = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(embedding);
    auto wp = (__ubuf__ float*)__cce_get_tile_ptr(weight);
    auto op = (__ubuf__ int32_t*)__cce_get_tile_ptr(output_ids);
    auto cp = (__ubuf__ float*)__cce_get_tile_ptr(output_confidence);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), active, tied, even;
        RegTensor<int32_t> lane, offsets, raw_id, chosen, invalid, zero_i, parity, one_i;
        RegTensor<float> values, maximum, negative_inf;
        vci(lane, 0, INC_ORDER);
        vmuls(offsets, lane, 16, all, MODE_ZEROING);
        vcmps_lt(active, lane, workers, all);
        vgather2(values, sp, (RegTensor<uint32_t>&)offsets, active);
        vgather2(raw_id, ip, (RegTensor<uint32_t>&)offsets, active);
        vdup((RegTensor<int32_t>&)negative_inf, -8388608, all, MODE_ZEROING);
        vsel(values, values, negative_inf, active);
        vcmax(maximum, values, all, MODE_ZEROING);
        vdup(maximum, maximum, all, POS_LOWEST, MODE_ZEROING);
        vcmp_eq(tied, values, maximum, all);
        vdup(invalid, 2147483647, all, MODE_ZEROING);
        vsel(raw_id, raw_id, invalid, active);
        vsel(chosen, raw_id, invalid, tied);
        vcmin(chosen, chosen, all, MODE_ZEROING);
        vdup(chosen, chosen, all, POS_LOWEST, MODE_ZEROING);
        vdup(zero_i, 0, all, MODE_ZEROING);
        vdup(one_i, 1, all, MODE_ZEROING);
        vand(parity, lane, one_i, all, MODE_ZEROING);
        vcmps_eq(even, parity, 0, all);
        vsel(chosen, chosen, zero_i, even);
        uint32_t id_count = 32;
        MaskReg id_mask = CreatePredicate<int32_t>(id_count);
        vsts(chosen, op, 0, NORM_B32, id_mask);
        RegTensor<float> sum;
        vdup(sum, 0.0f, all, MODE_ZEROING);
        for (uint16_t offset = 0; offset < static_cast<uint16_t>(D); offset += 64) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> value, scale, product;
            vlds(bf, hp, offset, UNPK_B16);
            vcvt(value, bf, all, PART_EVEN);
            vlds(scale, wp, offset, NORM);
            vmul(product, value, scale, all, MODE_ZEROING);
            vadd(sum, sum, product, all, MODE_ZEROING);
        }
        for (uint16_t offset = 0; offset < static_cast<uint16_t>(K); offset += 64) {
            RegTensor<bfloat16_t> bf;
            RegTensor<float> value, scale, product;
            vlds(bf, ep, offset, UNPK_B16);
            vcvt(value, bf, all, PART_EVEN);
            vlds(scale, wp, D + offset, NORM);
            vmul(product, value, scale, all, MODE_ZEROING);
            vadd(sum, sum, product, all, MODE_ZEROING);
        }
        vcadd(sum, sum, all, MODE_ZEROING);
        vdup(sum, sum, all, POS_LOWEST, MODE_ZEROING);
        vmuls(sum, sum, -1.0f, all, MODE_ZEROING);
        vexp(sum, sum, all, MODE_ZEROING);
        vadds(sum, sum, 1.0f, all, MODE_ZEROING);
        RegTensor<float> one;
        vdup(one, 1.0f, all, MODE_ZEROING);
        vdiv(sum, one, sum, all, MODE_ZEROING);
        uint32_t conf_count = 16;
        MaskReg conf_mask = CreatePredicate<float>(conf_count);
        vsts(sum, cp, 0, NORM_B32, conf_mask);
    }
}

__aicore__ inline int finalize_request(
    __gm__ float* score_ptr, __gm__ int32_t* index_ptr,
    __gm__ bfloat16_t* hidden_ptr, __gm__ bfloat16_t* embedding_ptr,
    __gm__ float* weight_ptr, __gm__ int64_t* output_ids, __gm__ float* output_confidence, int workers) {
    Vec<float, 512> scores;
    Vec<int32_t, 512> indices;
    Vec<bfloat16_t, D> hidden;
    Vec<bfloat16_t, K> embedding;
    Vec<float, D + K> weight;
    Vec<int32_t, 32> ids;
    Vec<float, 16> confidence;
    TASSIGN(scores, 0); TASSIGN(indices, 2048);
    TASSIGN(hidden, 4096); TASSIGN(embedding, 14336); TASSIGN(weight, 16384);
    TASSIGN(ids, 40960); TASSIGN(confidence, 41088);
    GM<float, 512> score_gm(score_ptr); GM<int32_t, 512> index_gm(index_ptr);
    GM<bfloat16_t, D> hidden_gm(hidden_ptr); GM<bfloat16_t, K> embedding_gm(embedding_ptr);
    GM<float, D + K> weight_gm(weight_ptr);
    TLOAD(scores, score_gm); TLOAD(indices, index_gm); TLOAD(hidden, hidden_gm);
    TLOAD(embedding, embedding_gm); TLOAD(weight, weight_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    finalize_vf(ids.data(), confidence.data(), scores.data(), indices.data(), hidden.data(), embedding.data(), weight.data(), workers);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GM<int32_t, 32> ids_gm(reinterpret_cast<__gm__ int32_t*>(output_ids));
    GM<float, 16> confidence_gm(output_confidence);
    TSTORE(ids_gm, ids); TSTORE(confidence_gm, confidence);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    int token = ids.GetValue(0);
    pipe_barrier(PIPE_ALL);
    return token;
}
#endif
} // namespace markov_native
