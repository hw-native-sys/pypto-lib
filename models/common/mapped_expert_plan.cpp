// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

namespace {
__aicore__ inline __gm__ int32_t *words(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ int32_t *>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline void sync_cache() {
#if !defined(__CPU_SIM)
    dcci(reinterpret_cast<__gm__ void *>(0), cache_line_t::ENTIRE_DATA_CACHE);
#endif
}

__aicore__ inline void touch(__gm__ int32_t *state, int32_t row) {
    const int32_t previous = state[6 * row + 2], next = state[6 * row + 3];
    state[6 * previous + 3] = next;
    state[6 * next + 2] = previous;
    const int32_t tail = state[2];
    state[6 * row + 2] = tail;
    state[6 * row + 3] = 0;
    state[6 * tail + 3] = row;
    state[2] = row;
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *counts_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *counts = words(counts_tensor);
    auto *state = words(reinterpret_cast<__gm__ Tensor *>(args[1]));
    auto *slot_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *slot_map = words(slot_tensor);
    auto *misses = words(reinterpret_cast<__gm__ Tensor *>(args[3]));
    const int32_t layer = args[4], first = args[5], stop = args[6];
    sync_cache();
    const int32_t capacity = state[0];
#if defined(MAPPED_EXPERT_PLAN_TOP6)
    const int32_t experts = slot_tensor->shapes[0];
    int32_t requested[6];
    for (int32_t i = first; i < stop; ++i) requested[i - first] = counts[i];
    // Keep the counts planner's expert order, including its LRU tie behavior.
    for (int32_t i = 1; i < stop - first; ++i) {
        const int32_t value = requested[i];
        int32_t j = i;
        while (j > 0 && requested[j - 1] > value) {
            requested[j] = requested[j - 1];
            --j;
        }
        requested[j] = value;
    }
#else
    const int32_t experts = counts_tensor->shapes[0];
#endif
    auto *directory = state + 6 * (capacity + 1);
    // Protect every resident demand before selecting any miss victim.
    for (int32_t position = first; position < stop; ++position) {
#if defined(MAPPED_EXPERT_PLAN_TOP6)
        const int32_t e = requested[position - first];
        const int32_t slot = directory[layer * experts + e];
#else
        const int32_t e = position;
        const int32_t slot = counts[e] > 0 ? directory[layer * experts + e] : -1;
#endif
        slot_map[e] = slot;
        misses[e] = -1;
        if (slot >= 0) {
            touch(state, slot + 1);
            ++reinterpret_cast<__gm__ uint32_t *>(state + 6 * (slot + 1))[4];
        }
    }
    for (int32_t position = first; position < stop; ++position) {
#if defined(MAPPED_EXPERT_PLAN_TOP6)
        const int32_t e = requested[position - first];
        if (slot_map[e] >= 0) continue;
#else
        const int32_t e = position;
        if (counts[e] <= 0 || slot_map[e] >= 0) continue;
#endif
        const int32_t row = state[3], slot = row - 1;
        const int32_t owner_layer = state[6 * row], owner = state[6 * row + 1];
        if (owner_layer >= 0) directory[owner_layer * experts + owner] = -1;
        directory[layer * experts + e] = slot;
        state[6 * row] = layer;
        state[6 * row + 1] = e;
        ++reinterpret_cast<__gm__ uint32_t *>(state + 6 * row)[5];
        slot_map[e] = slot;
        misses[e] = slot;
        touch(state, row);
    }
    sync_cache();
}
