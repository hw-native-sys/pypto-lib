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
#include "intrinsic.h"
#include "pipe_sync.h"

using namespace pto;

namespace {
__aicore__ inline void sync_scalar_reads(__gm__ void *address, uint64_t bytes) {
#if !defined(__CPU_SIM)
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    for (uintptr_t line = start & ~uintptr_t(63); line < start + bytes; line += 64) {
        dcci(reinterpret_cast<__gm__ void *>(line), cache_line_t::SINGLE_CACHE_LINE);
    }
#endif
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    // A scheduled core can retain the previous kernel's atomic write mode.
    set_atomic_none();
#endif
    auto *desc_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *desc = reinterpret_cast<__gm__ int64_t *>(desc_tensor->buffer.addr) + desc_tensor->start_offset;
    auto *miss_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *misses = reinterpret_cast<__gm__ int32_t *>(miss_tensor->buffer.addr) + miss_tensor->start_offset;
    auto *bank = reinterpret_cast<__gm__ Tensor *>(args[2]);
    const int32_t first = args[3];
    const int32_t stop = args[4] == -1 ? first + 1 : args[4];
    const int32_t bank_origin = args[5];
    // A different AIV may have published these inputs.
    sync_scalar_reads(desc, 4 * sizeof(int64_t));
    sync_scalar_reads(misses + first, (stop - first) * sizeof(int32_t));
    auto *src = reinterpret_cast<__gm__ uint8_t *>(desc[0]);
    const uint64_t src_stride = desc[1], field_bytes = desc[2], dst_stride = desc[3];
    // Scale banks flatten the slot and group axes, so strides[0] is not a
    // complete expert. Read the dtype without the runtime's A2-stalling size table.
    // Nonvolatile width selection stalled A2 copies; preserve the individual GM reads.
    const volatile __gm__ DataType *dtype = &bank->dtype;
    const uint64_t element_bytes =
        (*dtype == DataType::FLOAT32 || *dtype == DataType::INT32 || *dtype == DataType::UINT32) ? 4 :
        (*dtype == DataType::FLOAT16 || *dtype == DataType::BFLOAT16 || *dtype == DataType::INT16 || *dtype == DataType::UINT16) ? 2 :
        (*dtype == DataType::INT64 || *dtype == DataType::UINT64) ? 8 : 1;
    auto *dst = reinterpret_cast<__gm__ uint8_t *>(bank->buffer.addr) + bank->start_offset * element_bytes;
    constexpr int kTile = 65536;
    using ByteShape = Shape<1, 1, 1, 1, DYNAMIC>;
    using ByteStride = pto::Stride<1, 1, 1, 1, 1>;
    using ByteRange = GlobalTensor<uint8_t, ByteShape, ByteStride>;
    using ByteTile = Tile<TileType::Vec, uint8_t, 1, kTile, BLayout::RowMajor, 1, DYNAMIC>;
    const uint64_t first_tile = static_cast<uint64_t>(get_block_idx(args)) * kTile;
    const uint64_t tile_stride = static_cast<uint64_t>(get_block_num(args)) * kTile;
    for (int32_t e = first; e < stop; ++e) {
        const int32_t slot = misses[e];
        if (slot < 0) continue;
        for (uint64_t offset = first_tile; offset < field_bytes; offset += tile_stride) {
            const int64_t width = (field_bytes - offset > kTile) ? kTile : field_bytes - offset;
            ByteShape shape(1, 1, 1, 1, width);
            ByteRange source(src + e * src_stride + offset, shape);
            ByteRange target(dst + (slot - bank_origin) * dst_stride + offset, shape);
            ByteTile tile(width);
            TASSIGN(tile, 0);
            TLOAD(tile, source);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(target, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
