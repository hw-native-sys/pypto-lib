// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Gather selected eight-row blocks directly into contiguous output panels.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
template <typename T, int R, int C> using Vec = Tile<TileType::Vec, T, R, C>;
template <typename T, int R, int C> using GM = GlobalTensor<T, TileShape2D<T, R, C>, BaseShape2D<T, R, C>>;
template <typename T> __aicore__ inline __gm__ T *data(__gm__ int64_t *args, int i) {
    auto tensor = reinterpret_cast<__gm__ Tensor *>(args[i]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}
__aicore__ inline int read(__gm__ int32_t *ptr) {
    dcci((__gm__ void *)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}
__aicore__ inline int minimum(int a, int b) {
    return a < b ? a : b;
}
__aicore__ inline int maximum(int a, int b) {
    return a > b ? a : b;
}

__tf__ __aicore__ void make_metadata(__ubuf__ int32_t *bases, __ubuf__ int32_t *counts,
                                     __ubuf__ int32_t *valid, __ubuf__ int32_t *physical) {
    __VEC_SCOPE__ {
        vector_u32 lanes, block, within, base, count, yes, value, zero, invalid, seven;
        vector_bool active;
        auto full = pset_b32(PAT_ALL);
        vbr(yes, uint32_t(1));
        vbr(zero, uint32_t(0));
        vbr(seven, uint32_t(7));
        vbr(invalid, uint32_t(0xffffffff));
        for (uint16_t part = 0; part < 4; ++part) {
            vci((vector_s32 &)lanes, int32_t(part * 64), INC_ORDER);
            vshrs(block, lanes, 3, full, MODE_ZEROING);
            vand(within, lanes, seven, full, MODE_ZEROING);
            vgather2(base, (__ubuf__ uint32_t *)bases, block, full);
            vgather2(count, (__ubuf__ uint32_t *)counts, block, full);
            vcmp_lt(active, within, count, full);
            vsel(value, yes, zero, active);
            vsts(value, (__ubuf__ uint32_t *)valid, part * 64, NORM_B32, full);
            vadd(value, base, within, full, MODE_ZEROING);
            vsel(value, value, invalid, active);
            vsts(value, (__ubuf__ uint32_t *)physical, part * 64, NORM_B32, full);
        }
    }
}
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(-1, -1);
    int tokens = maximum(0, int(args[11]));
    int capacity = args[12], gathered_rows = args[13];
    int panels = gathered_rows / 256;
    int table_width = reinterpret_cast<__gm__ Tensor *>(args[2])->shapes[1];
    int requests = reinterpret_cast<__gm__ Tensor *>(args[2])->shapes[0];
    int cache_pages = reinterpret_cast<__gm__ Tensor *>(args[0])->shapes[0];
    Vec<uint8_t, 256, 64> packed;
    Vec<uint8_t, 1, 1024> scales;
    Vec<int32_t, 1, 256> valid, physical;
    Vec<int32_t, 1, 32> bases, counts;
    TASSIGN(packed, 0);
    TASSIGN(scales, 16384);
    TASSIGN(valid, 17408);
    TASSIGN(physical, 18432);
    TASSIGN(bases, 19584);
    TASSIGN(counts, 19712);
    for (int work = get_block_idx(args); work < tokens * panels; work += get_block_num(args)) {
        int token = work / panels, panel = work % panels;
        int count = maximum(0, minimum(read(data<int32_t>(args, 6) + token), capacity));
        int request = read(data<int32_t>(args, 3) + token);
        int visible = maximum(0, read(data<int32_t>(args, 4) + token));
        int first_slot = panel * 32;
        int id_count = maximum(0, minimum(32, capacity - first_slot));
        Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC> ids(id_count);
        TASSIGN(ids, 19456);
        if (id_count > 0) {
            GM<int32_t, 1, DYNAMIC> gm(data<int32_t>(args, 5) + token * capacity + first_slot,
                                       TileShape2D<int32_t, 1, DYNAMIC>(1, id_count),
                                       BaseShape2D<int32_t, 1, DYNAMIC>(1, id_count));
            TLOAD(ids, gm);
        }
        TEXPANDS(packed, uint8_t(0));
        TEXPANDS(scales, uint8_t(127));
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int previous_page = -1, mapped_page = -1;
        for (int block = 0; block < 32; ++block) {
            int row = -1, remaining = 0;
            if (first_slot + block < count && block < id_count) {
                int logical = ids.data()[block];
                if (logical >= 0 && logical < (visible + 7) / 8 && request >= 0 && request < requests) {
                    int logical_page = logical / 16;
                    if (logical_page < table_width) {
                        if (logical_page != previous_page) {
                            mapped_page = read(data<int32_t>(args, 2) + request * table_width + logical_page);
                            previous_page = logical_page;
                        }
                        if (mapped_page >= 0 && mapped_page < cache_pages) {
                            row = mapped_page * 128 + logical % 16 * 8;
                            remaining = minimum(8, visible - logical * 8);
                            Vec<uint8_t, 8, 64> payload;
                            Vec<uint8_t, 1, 32> scale;
                            TASSIGN(payload, block * 512);
                            TASSIGN(scale, 16384 + block * 32);
                            GM<uint8_t, 8, 64> kg(data<uint8_t>(args, 0) + int64_t(row) * 64);
                            GM<uint8_t, 1, 32> sg(data<uint8_t>(args, 1) + int64_t(row) * 4);
                            TLOAD(payload, kg);
                            TLOAD(scale, sg);
                        }
                    }
                }
            }
            bases.data()[block] = row;
            counts.data()[block] = remaining;
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        make_metadata(bases.data(), counts.data(), valid.data(), physical.data());
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t offset = int64_t(token) * gathered_rows + panel * 256;
        GM<uint8_t, 256, 64> pg(data<uint8_t>(args, 7) + offset * 64);
        GM<uint8_t, 1, 1024> sg(data<uint8_t>(args, 8) + offset * 4);
        GM<int32_t, 1, 256> vg(data<int32_t>(args, 9) + offset);
        GM<int32_t, 1, 256> ig(data<int32_t>(args, 10) + offset);
        TSTORE(pg, packed);
        TSTORE(sg, scales);
        TSTORE(vg, valid);
        TSTORE(ig, physical);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
