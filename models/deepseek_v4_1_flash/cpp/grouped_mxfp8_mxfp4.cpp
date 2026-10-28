/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {

constexpr uint32_t M_TILE = 64;
constexpr uint32_t N_TILE = 128;
constexpr uint32_t K_TILE = 256;
constexpr uint32_t SCALE_K_TILE = K_TILE / 32;
constexpr uint32_t VEC_K_TILE = K_TILE / 2;
constexpr uint32_t PACKED_COLS = N_TILE / 2;
constexpr uint32_t UB_SOURCE_BYTES = VEC_K_TILE * PACKED_COLS;
constexpr uint32_t UB_NZ_BYTES = VEC_K_TILE * N_TILE;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t L1_BASE = 0x10000;
constexpr uint32_t L1_SLOT_BYTES = K_TILE * N_TILE;
constexpr uint32_t L1_SLOT_COUNT = 2;
constexpr uint8_t WORK_DONE_FLAG = 4;
constexpr uint8_t VEC_CORE_ID_OFFSET = 16;

static_assert(UB_SOURCE_BYTES + UB_NZ_BYTES == 24 * 1024);
static_assert(M_TILE * K_TILE <= L0_PINGPONG_BYTES);
static_assert(K_TILE * N_TILE == L0_PINGPONG_BYTES);
static_assert(0x8800 + SCALE_K_TILE * N_TILE <= L1_BASE);
static_assert(L1_BASE + L1_SLOT_COUNT * L1_SLOT_BYTES <= 512 * 1024);

#ifdef __DAV_CUBE__
AICORE inline void releaseSlot(uint32_t slot)
{
    const uint8_t freeFlag = static_cast<uint8_t>(1 + 2 * slot);
    set_intra_block(PIPE_MTE1, freeFlag);
    set_intra_block(PIPE_MTE1, freeFlag + VEC_CORE_ID_OFFSET);
}

AICORE inline void waitSlot(uint32_t slot)
{
    const uint8_t readyFlag = static_cast<uint8_t>(2 * slot);
    wait_intra_block(PIPE_MTE1, readyFlag);
    wait_intra_block(PIPE_MTE1, readyFlag + VEC_CORE_ID_OFFSET);
}
#endif

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

struct Chunk {
    int64_t expert;
    int64_t stageRow;
};

AICORE inline Chunk acquireChunk(__gm__ int32_t* chunks, int64_t index)
{
    auto* descriptor = chunks + index * 4;
    dcci(descriptor, SINGLE_CACHE_LINE);
    dcci(descriptor + 3, SINGLE_CACHE_LINE);
    return {descriptor[0], descriptor[1]};
}

#ifdef __DAV_VEC__
AICORE inline void produceWeight(__gm__ uint8_t* weight, int64_t k, int64_t n,
                                int64_t expert, int64_t n0)
{
    constexpr uint32_t PACKED_ROW_BYTES = 256 / 2;
    constexpr uint32_t PACKED_TILE_BYTES = 256 * 256 / 2;
    using Packed = Tile<TileType::Vec, float4_e2m1x2_t, VEC_K_TILE, N_TILE>;
    using Raw = Tile<TileType::Vec, uint8_t, VEC_K_TILE, PACKED_COLS>;
    using NZ = Tile<TileType::Vec, float8_e4m3_t, VEC_K_TILE, N_TILE, BLayout::ColMajor,
                    VEC_K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using GlobalPacked = GlobalTensor<uint8_t, Shape<1, 1, 1, VEC_K_TILE, PACKED_COLS>,
                                      pto::Stride<1, 1, 1, PACKED_ROW_BYTES, 1>>;
    using MatSlot = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                         K_TILE, N_TILE, SLayout::RowMajor, 512>;
    MatSlot matSlot;
    Packed packed;
    Raw raw;
    NZ nz;
    // Per AIV UB: packed bytes [0, 8192), ordinary NZ [8192, 24576).
    TASSIGN(packed, 0);
    TASSIGN(raw, 0);
    TASSIGN(nz, UB_SOURCE_BYTES);

    const int64_t lane = get_subblockid();
    const int64_t expertBase = expert * k * n / 2;
    const int64_t panelBase = (n0 / 256) * (k / K_TILE) * PACKED_TILE_BYTES;
    const int64_t laneBase = lane * VEC_K_TILE * PACKED_ROW_BYTES + (n0 % 256) / 2;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    for (int64_t k0 = 0; k0 < k; k0 += K_TILE) {
        const int64_t offset = expertBase + panelBase + (k0 / K_TILE) * PACKED_TILE_BYTES + laneBase;
        GlobalPacked globalWeight(weight + offset);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(raw, globalWeight);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        TCVT(nz, packed, RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        const uint32_t slot = (k0 / K_TILE) % L1_SLOT_COUNT;
        wait_intra_block(PIPE_MTE3, static_cast<uint8_t>(1 + 2 * slot));
        TASSIGN(matSlot, L1_BASE + slot * L1_SLOT_BYTES);
        TINSERT(matSlot, nz, static_cast<uint16_t>(lane * VEC_K_TILE), static_cast<uint16_t>(0));
        set_intra_block(PIPE_MTE3, static_cast<uint8_t>(2 * slot));
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // Consume the one returned credit for each slot before the next work item.
    wait_intra_block(PIPE_MTE3, 1);
    wait_intra_block(PIPE_MTE3, 3);
    set_intra_block(PIPE_MTE3, WORK_DONE_FLAG);
    pipe_barrier(PIPE_ALL);
}
#endif

#ifdef __DAV_CUBE__
AICORE inline void consumeWeight(__gm__ float8_e4m3_t* x, __gm__ float8_e8m0_t* xScale,
                                __gm__ float8_e8m0_t* weightScale, __gm__ bfloat16_t* out,
                                int64_t k, int64_t n, int64_t experts, int64_t rows, Chunk chunk, int64_t n0)
{
    using MatA = Tile<TileType::Mat, float8_e4m3_t, M_TILE, K_TILE, BLayout::ColMajor,
                      M_TILE, K_TILE, SLayout::RowMajor, 512>;
    using MatB = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                      K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using MatAScale = Tile<TileType::Mat, float8_e8m0_t, M_TILE, SCALE_K_TILE, BLayout::RowMajor,
                           M_TILE, SCALE_K_TILE, SLayout::RowMajor, 32>;
    using MatBScale = Tile<TileType::Mat, float8_e8m0_t, SCALE_K_TILE, N_TILE, BLayout::ColMajor,
                           SCALE_K_TILE, N_TILE, SLayout::ColMajor, 32>;
    using Left = TileLeft<float8_e4m3_t, M_TILE, K_TILE>;
    using Right = TileRight<float8_e4m3_t, K_TILE, N_TILE>;
    using LeftScale = TileLeftScale<float8_e8m0_t, M_TILE, SCALE_K_TILE>;
    using RightScale = TileRightScale<float8_e8m0_t, SCALE_K_TILE, N_TILE>;
    using Acc = TileAcc<float, M_TILE, N_TILE>;
    using GlobalA = GlobalTensor<float8_e4m3_t, Shape<1, 1, 1, M_TILE, K_TILE>,
                                 pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    using GlobalAScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, M_TILE, SCALE_K_TILE, Layout::MX_A_ZZ>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ZZ>, Layout::MX_A_ZZ>;
    using GlobalBScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, SCALE_K_TILE, N_TILE, Layout::MX_B_NN>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_B_NN>, Layout::MX_B_NN>;
    using GlobalOut = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M_TILE, N_TILE>,
                                   pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;

    MatA aMat[2];
    MatB bMat;
    MatAScale aScaleMat[2];
    MatBScale bScaleMat[2];
    Left a[2];
    Right b[2];
    LeftScale aScale[2];
    RightScale bScale[2];
    Acc acc;
    // L1: A [0, 0x8000), A scales [0x8000, 0x8400), B scales [0x8400, 0x8c00).
    // The two L1 slots occupy [0x10000, 0x20000).
    TASSIGN(aMat[0], 0x0000);
    TASSIGN(aMat[1], 0x4000);
    TASSIGN(aScaleMat[0], 0x8000);
    TASSIGN(aScaleMat[1], 0x8200);
    TASSIGN(bScaleMat[0], 0x8400);
    TASSIGN(bScaleMat[1], 0x8800);
    // L0A/B data use 32 KiB slots; scale tiles bind to their data slot.
    TASSIGN(a[0], 0);
    TASSIGN(a[1], L0_PINGPONG_BYTES);
    TASSIGN(b[0], 0);
    TASSIGN(b[1], L0_PINGPONG_BYTES);
    TASSIGN(aScale[0], GetScaleAddr(a[0].data()));
    TASSIGN(aScale[1], GetScaleAddr(a[1].data()));
    TASSIGN(bScale[0], GetScaleAddr(b[0].data()));
    TASSIGN(bScale[1], GetScaleAddr(b[1].data()));
    TASSIGN(acc, 0);

    releaseSlot(0);
    releaseSlot(1);
    const int64_t q = k / 32;
    const int64_t globalQ = experts * q;
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (int64_t k0 = 0; k0 < k; k0 += K_TILE) {
        const uint32_t slot = (k0 / K_TILE) % 2;
        const auto event = static_cast<event_t>(slot);
        GlobalA globalA(x + chunk.stageRow * k + k0, GlobalA::Shape(),
                        GlobalA::Stride(rows * k, rows * k, rows * k, k, 1));
        // MX_A_ZZ: [row/16, q/2, 16, 2], with q covering all K groups.
        const int64_t aScaleOffset = chunk.stageRow * q + (k0 / 32) * 16;
        GlobalAScale globalAScale(xScale + aScaleOffset, GlobalAScale::Shape(), GlobalAScale::Stride(rows, q));
        // MX_B_NN: [n/16, global_q/2, 16, 2], with experts flattened into global_q.
        const int64_t bScaleOffset = n0 * globalQ + (chunk.expert * q + k0 / 32) * 16;
        GlobalBScale globalBScale(weightScale + bScaleOffset, GlobalBScale::Shape(), GlobalBScale::Stride(globalQ, n));
        wait_flag(PIPE_MTE1, PIPE_MTE2, event);
        TLOAD(aMat[slot], globalA);
        TLOAD(aScaleMat[slot], globalAScale);
        TLOAD(bScaleMat[slot], globalBScale);
        waitSlot(slot);
        TASSIGN(bMat, L1_BASE + slot * L1_SLOT_BYTES);
        set_flag(PIPE_MTE2, PIPE_MTE1, event);
        wait_flag(PIPE_MTE2, PIPE_MTE1, event);
        wait_flag(PIPE_M, PIPE_MTE1, event);
        TMOV(a[slot], aMat[slot]);
        TMOV(b[slot], bMat);
        TMOV(aScale[slot], aScaleMat[slot]);
        TMOV(bScale[slot], bScaleMat[slot]);
        // Slot release follows every L1-to-L0 data and scale read.
        releaseSlot(slot);
        set_flag(PIPE_MTE1, PIPE_MTE2, event);
        set_flag(PIPE_MTE1, PIPE_M, event);
        wait_flag(PIPE_MTE1, PIPE_M, event);
        if (k0 == 0) {
            TMATMUL_MX(acc, a[slot], aScale[slot], b[slot], bScale[slot]);
        } else {
            TMATMUL_MX(acc, acc, a[slot], aScale[slot], b[slot], bScale[slot]);
        }
        set_flag(PIPE_M, PIPE_MTE1, event);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    GlobalOut globalOut(out + chunk.stageRow * n + n0, GlobalOut::Shape(),
                        GlobalOut::Stride(rows * n, rows * n, rows * n, n, 1));
    TSTORE(globalOut, acc);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_intra_block(PIPE_MTE1, WORK_DONE_FLAG);
    wait_intra_block(PIPE_MTE1, WORK_DONE_FLAG + VEC_CORE_ID_OFFSET);
    pipe_barrier(PIPE_ALL);
}
#endif

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    // Tensor slots: x, x_scale, weight, weight_scale, chunks, out. Scalar slots: K, N, G, C.
    auto* chunks = tensorData<int32_t>(args, 4);
    const int64_t k = args[6];
    const int64_t n = args[7];
    const int64_t chunkCount = args[9];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    const int64_t nTiles = n / N_TILE;
#ifdef __DAV_VEC__
    auto* weight = tensorData<uint8_t>(args, 2);
#endif
#ifdef __DAV_CUBE__
    const int64_t experts = args[8];
    auto* x = tensorData<float8_e4m3_t>(args, 0);
    auto* xScale = tensorData<float8_e8m0_t>(args, 1);
    auto* weightScale = tensorData<float8_e8m0_t>(args, 3);
    auto* out = tensorData<bfloat16_t>(args, 5);
#endif
    for (int64_t work = block; work < chunkCount * nTiles; work += blocks) {
        const Chunk chunk = acquireChunk(chunks, work / nTiles);
        const int64_t n0 = (work % nTiles) * N_TILE;
#ifdef __DAV_VEC__
        produceWeight(weight, k, n, chunk.expert, n0);
#endif
#ifdef __DAV_CUBE__
        consumeWeight(x, xScale, weightScale, out, k, n, experts, chunkCount * M_TILE, chunk, n0);
#endif
    }
}
