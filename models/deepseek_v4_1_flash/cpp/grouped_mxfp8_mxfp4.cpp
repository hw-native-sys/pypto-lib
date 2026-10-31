/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_mxfp4_pipeline.hpp"

using namespace pto;
using namespace grouped_mxfp4;

namespace {

#ifdef __DAV_VEC__
constexpr uint32_t ZERO_UB_BASE = 2 * (UB_SOURCE_BYTES + UB_NZ_BYTES);
static_assert(ZERO_UB_BASE + M_TILE * N_TILE * sizeof(bfloat16_t) <= 240 * 1024);
using PaddingTile = Tile<TileType::Vec, bfloat16_t, M_TILE, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;

AICORE inline void initializePadding()
{
    PaddingTile zero;
    zero.SetValidRow(M_TILE);
    TASSIGN(zero, ZERO_UB_BASE);
    TEXPANDS(zero, static_cast<bfloat16_t>(0.0f));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
}

AICORE inline void clearPadding(__gm__ bfloat16_t* out, int64_t n, Chunk chunk, int64_t n0)
{
    if (get_subblockid() != 0 || chunk.validRows == M_TILE) {
        return;
    }
    using GlobalPadding = GlobalTensor<bfloat16_t, Shape<1, 1, 1, DYNAMIC, N_TILE>,
                                       pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    PaddingTile zero;
    const int64_t tail = M_TILE - chunk.validRows;
    zero.SetValidRow(tail);
    TASSIGN(zero, ZERO_UB_BASE);
    GlobalPadding global(out + (chunk.stageRow + chunk.validRows) * n + n0,
                         GlobalPadding::Shape(1, 1, 1, tail, N_TILE),
                         GlobalPadding::Stride(n * M_TILE, n * M_TILE, n * M_TILE, n, 1));
    TSTORE(global, zero);
}
#endif

#ifdef __DAV_CUBE__
struct StoreProjection {
    __gm__ bfloat16_t* out;
    int64_t n;
    int64_t rows;
    Chunk chunk;
    int64_t n0;

    template <typename Acc>
    AICORE inline void operator()(Acc& acc) const
    {
        using GlobalOut = GlobalTensor<bfloat16_t, Shape<1, 1, 1, DYNAMIC, N_TILE>,
                                        pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
        GlobalOut globalOut(out + chunk.stageRow * n + n0,
                            GlobalOut::Shape(1, 1, 1, chunk.validRows, N_TILE),
                            GlobalOut::Stride(rows * n, rows * n, rows * n, n, 1));
        TSTORE<STPhase::Final>(globalOut, acc);
    }
};
#endif

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    // Tensor slots: x, x_scale, weight, weight_scale, metadata, out. Scalars: K, N, E, C, row_stride.
    auto* chunks = tensorData<int32_t>(args, 4);
    const int64_t k = args[6];
    const int64_t n = args[7];
    const int64_t experts = args[8];
    const int64_t chunkCount = args[9];
    const int64_t rowStride = args[10];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    const int64_t nTiles = n / N_TILE;
#ifdef __DAV_VEC__
    auto* weight = tensorData<uint8_t>(args, 2);
    auto* out = tensorData<bfloat16_t>(args, 5);
#endif
#ifdef __DAV_CUBE__
    auto* x = tensorData<float8_e4m3_t>(args, 0);
    auto* xScale = tensorData<float8_e8m0_t>(args, 1);
    auto* weightScale = tensorData<float8_e8m0_t>(args, 3);
    auto* out = tensorData<bfloat16_t>(args, 5);
#endif
#ifdef __DAV_VEC__
    if (rowStride == 0 && get_subblockid() == 0) {
        initializePadding();
    }
#endif
    beginGemm();
    if (rowStride > 0) {
        // Counts are immutable after input_ready; AIC and both AIV lanes acquire the same work list.
        int64_t workBase = 0;
        for (int64_t expert = 0; expert < experts; ++expert) {
            const int64_t count = acquireCount(chunks, expert);
            const int64_t workCount = ((count + EXPERT_M_TILE - 1) / EXPERT_M_TILE) * nTiles;
            const int64_t firstWork = (block + blocks - workBase % blocks) % blocks;
            for (int64_t work = firstWork; work < workCount; work += blocks) {
                const int64_t expertRow = (work / nTiles) * EXPERT_M_TILE;
                const int64_t remaining = count - expertRow;
                const int64_t validRows = remaining < EXPERT_M_TILE ? remaining : EXPERT_M_TILE;
                const Chunk chunk = {expert, expert * rowStride + expertRow, validRows};
                const int64_t n0 = (work % nTiles) * N_TILE;
#ifdef __DAV_VEC__
                produceWeight(weight, k, n, chunk.expert, n0);
#endif
#ifdef __DAV_CUBE__
                computeProjection<EXPERT_M_TILE>(x, xScale, weightScale, k, n, experts,
                                                experts * rowStride, chunk, n0,
                                                StoreProjection{out, n, experts * rowStride, chunk, n0});
#endif
            }
            workBase += workCount;
        }
        endGemm();
        return;
    }
    for (int64_t work = block; work < chunkCount * nTiles; work += blocks) {
        const Chunk chunk = acquireChunk(chunks, work / nTiles);
        const int64_t n0 = (work % nTiles) * N_TILE;
#ifdef __DAV_VEC__
        clearPadding(out, n, chunk, n0);
        if (chunk.validRows > 0) {
            produceWeight(weight, k, n, chunk.expert, n0);
        }
#endif
#ifdef __DAV_CUBE__
        if (chunk.validRows > 0) {
            computeProjection<M_TILE>(x, xScale, weightScale, k, n, experts, chunkCount * M_TILE, chunk, n0,
                                      StoreProjection{out, n, chunkCount * M_TILE, chunk, n0});
        }
#endif
    }
    endGemm();
}
