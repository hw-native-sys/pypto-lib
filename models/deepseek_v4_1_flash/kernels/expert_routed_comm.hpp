/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "platform_comm/comm_context.h"

using namespace pto;

namespace {
// Derive the wire offsets from the runtime that compiles this kernel.
constexpr int COMM_RANK_WORD = offsetof(CommContext, rankId) / sizeof(uint64_t);
constexpr int COMM_WINDOWS_WORD = offsetof(CommContext, windowsIn) / sizeof(uint64_t);
static_assert(offsetof(CommContext, rankId) % sizeof(uint64_t) == 0);
static_assert(offsetof(CommContext, windowsIn) % sizeof(uint64_t) == 0);
constexpr int WIDTH = 5120;
constexpr int SLOTS = 8;
constexpr int ROW_BYTES = WIDTH * sizeof(bfloat16_t);
using Row = Tile<TileType::Vec, bfloat16_t, 1, WIDTH>;
using GlobalRow = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, WIDTH>, pto::Stride<WIDTH, WIDTH, WIDTH, WIDTH, 1>>;

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, int index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

template <int CAPACITY>
AICORE inline __ubuf__ int32_t* loadMetadata(__gm__ int32_t* source, int count, int offset)
{
    using Local = Tile<TileType::Vec, int32_t, 1, CAPACITY, BLayout::RowMajor, 1, DYNAMIC>;
    using Global = GlobalTensor<int32_t, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
    Local local;
    local.SetValidCol(count);
    TASSIGN(local, offset);
    Global global(source, Global::Shape(1, 1, 1, 1, count));
    TLOAD(local, global);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID4);
    return local.data();
}

}
