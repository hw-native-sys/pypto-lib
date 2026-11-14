/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#define ASCENDC_DUMP 0
#define ASCENDC_TILING_KEY_VAR_IMPL 0
#include <utils/debug/asc_printf.h>
#include <utils/debug/asc_assert.h>
#include "moe_pipeline_compute.hpp"

#if !defined(MOE_EP) || !defined(MOE_EXPERTS) || !defined(MOE_TOKENS) || \
    !defined(MOE_TOPK) || !defined(MOE_HIDDEN) || !defined(MOE_INTER)
#error "The immutable native specialization must define all MEGA shape macros."
#endif

namespace v41_official_entry {

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    if constexpr (MOE_TOKENS > 80)
        dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int64_t dimension(__gm__ int64_t* args, uint32_t index, uint32_t axis)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    if constexpr (MOE_TOKENS > 80) {
        dcci(tensor, SINGLE_CACHE_LINE);
        dcci(reinterpret_cast<__gm__ uint8_t*>(tensor) + 64, SINGLE_CACHE_LINE);
    }
    return tensor->shapes[axis];
}

__aicore__ inline void acquire_metadata(__gm__ int64_t* args)
{
    static_assert(alignof(Tensor) >= 64 && sizeof(Tensor) == 128);
    static_assert(offsetof(Tensor, start_offset) + sizeof(uint64_t) <= 64);
    static_assert(offsetof(Tensor, shapes) + sizeof(Tensor::shapes) <= 128);
    #pragma unroll
    for (uint32_t index = 0; index < 17; ++index)
        dcci(reinterpret_cast<__gm__ Tensor*>(args[index]), SINGLE_CACHE_LINE);
    // The symmetric binding acquires argument 17's first metadata line.
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[0]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[1]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[4]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[10]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[16]) + 64, SINGLE_CACHE_LINE);
    dcci(reinterpret_cast<__gm__ uint8_t*>(args[17]) + 64, SINGLE_CACHE_LINE);
}

template <uint64_t workspace_bytes>
__aicore__ inline void validate_shapes(__gm__ int64_t* args)
{
    if (dimension(args, 0, 0) != MOE_TOKENS || dimension(args, 0, 1) != MOE_TOPK ||
        dimension(args, 1, 0) != MOE_TOKENS || dimension(args, 1, 1) != MOE_HIDDEN ||
        dimension(args, 4, 0) != MOE_EXPERTS ||
        dimension(args, 10, 0) != MOE_HIDDEN || dimension(args, 10, 1) != MOE_INTER ||
        dimension(args, 16, 0) != MOE_TOKENS || dimension(args, 16, 1) != MOE_HIDDEN ||
        dimension(args, 17, 0) < static_cast<int64_t>(workspace_bytes)) trap();
}

template <bool kFullTopology>
__aicore__ inline void run(__gm__ int64_t* args, deep_gemm::CoreTopology<kFullTopology> topology)
{
    constexpr auto& workspace = deep_gemm::mega_moe::layout::kWorkspace<
        MOE_EP, MOE_EP * MOE_EXPERTS, 1, MOE_TOKENS,
        MOE_TOPK, MOE_HIDDEN, MOE_INTER>;
    constexpr uint64_t workspace_bytes = workspace.get_num_bytes();
    const auto sym = [&]() __aicore__ {
        if constexpr (MOE_TOKENS <= 80) {
            acquire_metadata(args);
            const auto acquired = bind_pypto_sym_buffer<MOE_EP, workspace_bytes>(args, 17, 19);
            validate_shapes<workspace_bytes>(args);
            return acquired;
        } else {
            validate_shapes<workspace_bytes>(args);
            return bind_pypto_sym_buffer<MOE_EP, workspace_bytes>(args, 17, 19);
        }
    }();
    const uint32_t core = get_block_idx(args);
#ifdef __DAV_VEC__
    const uint32_t lane = get_sub_block_id(args);
#else
    const uint32_t lane = 0;
#endif
    const int32_t requested = static_cast<int32_t>(args[18]);
    const uint32_t active = requested < 0 ? 0 : (requested > MOE_TOKENS ? MOE_TOKENS : requested);
    auto* output = data<bfloat16_t>(args, 16);
    // Both asc_init and the Cube MAD direction update CTRL within this call.
    const uint64_t saved_ctrl = get_ctrl();
    asc_init();
    v41_official_input::publish<workspace>(sym.template get_base_ptr<__gm__ uint8_t*>(),
        data<uint8_t>(args, 1), data<uint8_t>(args, 2), data<int32_t>(args, 0),
        data<float>(args, 3), output, active, 2 * core + lane, topology);
    v41_official_input::publication_fence(lane);
    const auto weights = deep_gemm::mega_moe::layout::make_original_weights(
        data<uint8_t>(args, 4), data<uint8_t>(args, 5), data<uint8_t>(args, 6), data<uint8_t>(args, 7),
        data<uint8_t>(args, 8), data<uint8_t>(args, 9), data<uint8_t>(args, 10), data<uint8_t>(args, 11),
        data<uint8_t>(args, 12), data<uint8_t>(args, 13), data<uint8_t>(args, 14), data<uint8_t>(args, 15));
    deep_gemm::mega_moe::mega_moe_device<MOE_EP, MOE_EP * MOE_EXPERTS, 1,
        MOE_TOKENS, MOE_TOPK, MOE_HIDDEN, MOE_INTER>(
        output, nullptr, weights, active, 10.0f, sym, core, lane, topology);
    // The core drains its event credits and all pipes before returning. Keep
    // the PTO unit-repeat DMA convention and restore the caller's CTRL.
#ifdef __DAV_VEC__
    constexpr uint64_t unit_loop = (uint64_t{1} << 21) | 1;
    set_loop_size_ubtoout(unit_loop);
    set_loop_size_outtoub(unit_loop);
#endif
    set_ctrl(saved_ctrl);
}

} // namespace v41_official_entry

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    const uint32_t cores = get_block_num(args);
    if (cores == 0 || cores > deep_gemm::kNumAICores) trap();
    // Retain compile-time scheduling on a full cohort; smaller cohorts share
    // the same pipeline with runtime work strides and completion counts.
    if (cores == deep_gemm::kNumAICores)
        v41_official_entry::run(args, deep_gemm::CoreTopology<true>{cores});
    else
        v41_official_entry::run(args, deep_gemm::CoreTopology<false>{cores});
}
