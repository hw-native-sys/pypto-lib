/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#ifdef __CPU_SIM
#include <atomic>

#ifndef __aicore__
#define __aicore__
#endif
#endif

#include "tensor.h"

#define PTOAutoSyncTailMode SplitPrePostAutoSyncTailMode
#define ptoas_auto_sync_tail split_pre_post_auto_sync_tail
#define kernel_entry split_pre_post_generated_entry
#include "generated/split_pre_post.cpp.inc"
#undef kernel_entry
#undef ptoas_auto_sync_tail
#undef PTOAutoSyncTailMode

#define PTOAutoSyncTailMode MixXAutoSyncTailMode
#define ptoas_auto_sync_tail mix_x_auto_sync_tail
#define kernel_entry mix_x_generated_entry
#include "generated/mix_x.cpp.inc"
#undef kernel_entry
#undef ptoas_auto_sync_tail
#undef PTOAutoSyncTailMode

#define PTOAutoSyncTailMode FfnNormAutoSyncTailMode
#define ptoas_auto_sync_tail ffn_norm_auto_sync_tail
#define kernel_entry ffn_norm_generated_entry
#include "generated/ffn_norm.cpp.inc"
#undef kernel_entry
#undef ptoas_auto_sync_tail
#undef PTOAutoSyncTailMode

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) float
scalar_fp32(__gm__ int64_t *args, int32_t index) {
    union {
        uint64_t bits;
        float value;
    } scalar;
    scalar.bits = static_cast<uint64_t>(args[index]);
    return scalar.value;
}

static constexpr int32_t kFusedAivCores = 8;
static constexpr int32_t kMixBlocks = 4;
static constexpr int32_t kHiddenSize = 4096;
static constexpr int32_t kPreBytes = 8 * 8 * sizeof(float);
static constexpr int32_t kXMixedRowBytes =
    kHiddenSize * sizeof(bfloat16_t);
static constexpr uint64_t kSoftSyncUbAddr = 176 * 1024;

#ifdef __CPU_SIM
static std::atomic<int32_t> sim_sync_arrivals{0};
static std::atomic<int32_t> sim_sync_generation{0};
#endif

static __aicore__ __attribute__((always_inline)) void
invalidate_gm_range(__gm__ void *ptr, int32_t bytes) {
    __gm__ uint8_t *base = reinterpret_cast<__gm__ uint8_t *>(ptr);
    for (int32_t offset = 0; offset < bytes; offset += 32) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void *>(base + offset), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) void
soft_sync(__gm__ int32_t *workspace, int32_t lane) {
#ifdef __CPU_SIM
    (void)workspace;
    (void)lane;
    int32_t generation =
        sim_sync_generation.load(std::memory_order_acquire);
    if (sim_sync_arrivals.fetch_add(1, std::memory_order_acq_rel) ==
        kFusedAivCores - 1) {
        sim_sync_arrivals.store(0, std::memory_order_relaxed);
        sim_sync_generation.fetch_add(1, std::memory_order_release);
    } else {
        while (sim_sync_generation.load(std::memory_order_acquire) ==
               generation) {
        }
    }
#else
    pto::GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gm_workspace(
        workspace);
    pto::Tile<
        pto::TileType::Vec,
        int32_t,
        1,
        pto::SYNCALL_SOFT_SLOT_INT32
    > sync_tile;
#ifndef __PTO_AUTO__
    sync_tile.data() =
        reinterpret_cast<__ubuf__ int32_t *>(kSoftSyncUbAddr);
#endif
    pipe_barrier(PIPE_ALL);
    pto::SYNCALL_SOFT_AIV_BARRIER(
        gm_workspace.data(),
        sync_tile.data(),
        kFusedAivCores,
        lane);
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    int32_t lane = get_block_idx(args);

    __gm__ Tensor *inv_rms_tensor =
        reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *x_flat_tensor =
        reinterpret_cast<__gm__ Tensor *>(args[6]);
    int64_t t_linear = static_cast<int64_t>(inv_rms_tensor->shapes[0]);
    int64_t t_dim = static_cast<int64_t>(x_flat_tensor->shapes[0]);

    if (lane == 0) {
        split_pre_post(
            tensor_data<float>(args, 1),
            tensor_data<float>(args, 2),
            tensor_data<float>(args, 3),
            tensor_data<float>(args, 4),
            tensor_data<float>(args, 5),
            scalar_fp32(args, 13),
            scalar_fp32(args, 14),
            t_linear,
            0,
            1);
    }
    soft_sync(tensor_data<int32_t>(args, 12), lane);
    invalidate_gm_range(
        static_cast<__gm__ void *>(tensor_data<float>(args, 4)),
        kPreBytes);

    if (lane < kMixBlocks) {
        mix_x(
            tensor_data<float>(args, 4),
            tensor_data<bfloat16_t>(args, 0),
            tensor_data<float>(args, 6),
            t_linear,
            t_dim,
            lane,
            kMixBlocks);
    }
    soft_sync(tensor_data<int32_t>(args, 12), lane);

    int32_t num_tokens = static_cast<int32_t>(args[15]);
    int32_t active_tokens = num_tokens > 0 ? static_cast<int32_t>(t_dim) : 0;
    if (lane < active_tokens) {
        invalidate_gm_range(
            static_cast<__gm__ void *>(
                tensor_data<bfloat16_t>(args, 0) +
                lane * kHiddenSize),
            kXMixedRowBytes);
        ffn_norm(
            tensor_data<bfloat16_t>(args, 0),
            tensor_data<bfloat16_t>(args, 7),
            tensor_data<float>(args, 8),
            tensor_data<float>(args, 9),
            tensor_data<float>(args, 10),
            tensor_data<float>(args, 11),
            lane,
            active_tokens);
    }
}
