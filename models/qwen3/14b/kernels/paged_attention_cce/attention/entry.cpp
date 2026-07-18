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

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include <pto/pto-inst.hpp>

#include "../kernel/fai_body.hpp"

namespace {

static __aicore__ __attribute__((always_inline)) __gm__ int32_t *qwen_fai_barrier_data(GM_ADDR metadata) {
    uint64_t raw_barrier = reinterpret_cast<uint64_t>(metadata + qwen_fai_metadata::kBarrierAlignmentOffset);
    uint64_t aligned_barrier = (raw_barrier + qwen_fai_metadata::kBarrierAlignmentBytes - 1) &
                               ~(static_cast<uint64_t>(qwen_fai_metadata::kBarrierAlignmentBytes) - 1);
    return reinterpret_cast<__gm__ int32_t *>(aligned_barrier);
}

// After the AIV prologue writes q/k/v to GM, synchronize so the AIC attention does
// not read them early. The 48 AIV lanes soft-barrier among themselves over the
// rope-ready slots; each AIC polls those slots until all 48 are non-zero.
// Indexing is args-based (get_block_idx*2 + get_sub_block_id): the hardware
// get_subblockdim() reads 0 in this pypto launch (the dispatch does not set the
// subblockdim register), so pto::SYNCALL's builtin participant model
// (SYNCALL_GET_MIX_PARTICIPANT_*) cannot be used — it under-counts (24 not 72) and
// collides the AIV indices, deadlocking. The vendor attention uses the same
// args-based getters, so this matches its view of the topology. The soft barrier's
// DCCI + dsb(DSB_DDR) gives the cross-core GM visibility the AIC needs.
static __aicore__ __attribute__((always_inline)) void sync_qwen_rope_producers(
    __gm__ int64_t *args,
    __gm__ int32_t *fai_barrier
) {
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(fai_barrier) + qwen_fai_metadata::kBarrierBytes
    );
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    while (true) {
        uint32_t ready_count = 0;
        for (uint32_t i = 0; i < qwen_fai_metadata::kRopeReadySlotCount; ++i) {
            __gm__ int32_t *slot = ready + i * qwen_fai_metadata::kRopeReadySlotWords;
            dcci(reinterpret_cast<__gm__ void *>(slot), SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            if (slot[0] != 0) {
                ++ready_count;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (ready_count >= qwen_fai_metadata::kRopeReadySlotCount) {
            break;
        }
    }
#elif defined(__DAV_VEC__)
    uint32_t aiv_idx = static_cast<uint32_t>(get_block_idx(args)) * 2 +
                       static_cast<uint32_t>(get_sub_block_id(args));
    pto::SYNCALL_SOFT_AIV_BARRIER(ready, reinterpret_cast<__ubuf__ int32_t *>(0x3000),
                                  static_cast<int32_t>(qwen_fai_metadata::kRopeReadySlotCount),
                                  static_cast<int32_t>(aiv_idx));
#endif
    pipe_barrier(PIPE_ALL);
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    GM_ADDR metadata = tensor_data<uint8_t>(args, kMetadataArg);
    acquire_qwen_fai_metadata(metadata);
    run_qwen_rope_qkv(args);
    __gm__ int32_t *fai_barrier = qwen_fai_barrier_data(metadata);
    sync_qwen_rope_producers(args, fai_barrier);
    __gm__ const FAInferTilingData *tiling = reinterpret_cast<__gm__ const FAInferTilingData *>(metadata);
    if (tiling->needCoreNum != 0) {
        run_qwen_fai<true>(args, fai_barrier);
    } else {
        run_qwen_fai<false>(args);
    }
}

#endif
