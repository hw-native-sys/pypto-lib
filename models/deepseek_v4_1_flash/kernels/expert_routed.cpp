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

namespace grouped_mxfp4 {

using namespace pto;

#if defined(__CPU_SIM)
// FIFOs transport data between the simulator's private core memories.
using WeightPipe = TPipe<0, Direction::DIR_V2C, 65536, 2>;
using ProjectionPipe = TPipe<1, Direction::DIR_C2V, 65536, 2>;
using HiddenPipe = TPipe<2, Direction::DIR_V2C, 16384, 2>;
using ScalePipe = TPipe<3, Direction::DIR_V2C, 512, 2>;
inline thread_local WeightPipe weightPipe(nullptr, 0, 0);
inline thread_local ProjectionPipe projectionPipe(nullptr, 0, 0);
inline thread_local HiddenPipe hiddenPipe(nullptr, 0, 0);
inline thread_local ScalePipe scalePipe(nullptr, 0, 0);
#endif

constexpr uint32_t M_TILE = 64;
constexpr uint32_t N_TILE = 256;
constexpr uint32_t K_TILE = 256;
constexpr uint32_t SCALE_K_TILE = K_TILE / 32;
constexpr uint32_t L0_K_TILE = 128;
constexpr uint32_t A_STAGE_K = 1024;
constexpr uint32_t SCALE_STAGE_K = 4096;
constexpr uint32_t SCALE_STAGE_Q = SCALE_STAGE_K / 32;
constexpr uint32_t L0_SCALE_K_TILE = L0_K_TILE / 32;
constexpr uint32_t VEC_K_TILE = K_TILE / 2;
constexpr uint32_t PACKED_COLS = N_TILE / 2;
constexpr uint32_t UB_SOURCE_BYTES = VEC_K_TILE * PACKED_COLS;
constexpr uint32_t UB_NZ_BYTES = ((VEC_K_TILE + 15) / 16 * 16 + 1) * N_TILE;
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;
constexpr uint32_t L1_BASE = 2 * M_TILE * A_STAGE_K + 2 * (M_TILE + N_TILE) * SCALE_STAGE_Q;
constexpr uint32_t L1_SLOT_BYTES = K_TILE * N_TILE;
constexpr uint32_t L1_SLOT_COUNT = 2;
constexpr uint8_t VEC_CORE_ID_OFFSET = 16;

static_assert(UB_SOURCE_BYTES + UB_NZ_BYTES <= 64 * 1024);
static_assert(M_TILE * K_TILE <= L0_PINGPONG_BYTES);
static_assert(L0_K_TILE * N_TILE <= L0_PINGPONG_BYTES);

static_assert(L1_BASE + L1_SLOT_COUNT * L1_SLOT_BYTES <= 512 * 1024);

#ifdef __DAV_CUBE__
AICORE inline void releaseSlot(uint32_t slot)
{
#if defined(__CPU_SIM)
    TFREE<WeightPipe, TileSplitAxis::TILE_UP_DOWN>(weightPipe);
#else
    const uint8_t freeFlag = static_cast<uint8_t>(1 + 2 * slot);
    set_intra_block(PIPE_MTE1, freeFlag);
    set_intra_block(PIPE_MTE1, freeFlag + VEC_CORE_ID_OFFSET);
#endif
}

AICORE inline void waitSlot(uint32_t slot)
{
#if !defined(__CPU_SIM)
    const uint8_t readyFlag = static_cast<uint8_t>(2 * slot);
    wait_intra_block(PIPE_MTE1, readyFlag);
    wait_intra_block(PIPE_MTE1, readyFlag + VEC_CORE_ID_OFFSET);
#endif
}
#endif

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
#if defined(__CPU_SIM)
    dcci(tensor, cache_line_t::SINGLE_CACHE_LINE);
#else
    dcci(tensor, SINGLE_CACHE_LINE);
#endif
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

struct Chunk {
    int64_t expert;
    int64_t stageRow;
    int64_t validRows;
};

AICORE inline int64_t acquireCount(__gm__ int32_t* counts, int64_t expert)
{
    auto* count = counts + expert;
#if defined(__CPU_SIM)
    dcci(count, cache_line_t::SINGLE_CACHE_LINE);
#else
    dcci(count, SINGLE_CACHE_LINE);
#endif
    return *count;
}

#ifdef __DAV_VEC__
AICORE inline void produceWeight(__gm__ uint8_t* weight, int64_t k, int64_t n,
                                int64_t expert, int64_t n0, int64_t kBegin = 0, int64_t kEnd = 0)
{
    constexpr uint32_t PACKED_ROW_BYTES = 256 / 2;
    constexpr uint32_t PACKED_TILE_BYTES = 256 * 256 / 2;
    static_assert(2 * (UB_SOURCE_BYTES + UB_NZ_BYTES) <= 192 * 1024);
    using Packed = Tile<TileType::Vec, float4_e2m1x2_t, VEC_K_TILE, N_TILE>;
    using Raw = Tile<TileType::Vec, uint8_t, VEC_K_TILE, PACKED_COLS>;
    using NZ = Tile<TileType::Vec, float8_e4m3_t, VEC_K_TILE, N_TILE, BLayout::ColMajor,
                    VEC_K_TILE, N_TILE, SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;
    static_assert(NZ::StorageBytes == UB_NZ_BYTES);
    using GlobalPacked = GlobalTensor<uint8_t, Shape<1, 1, 1, VEC_K_TILE, PACKED_COLS>,
                                      pto::Stride<1, 1, 1, PACKED_ROW_BYTES, 1>>;
    using MatSlot = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                         K_TILE, N_TILE, SLayout::RowMajor, 512>;
    MatSlot matSlot;
    Packed packed[2];
    Raw raw[2];
    NZ nz[2];
    for (uint32_t slot = 0; slot < 2; ++slot) {
        TASSIGN(packed[slot], slot * UB_SOURCE_BYTES);
        TASSIGN(raw[slot], slot * UB_SOURCE_BYTES);
        TASSIGN(nz[slot], 2 * UB_SOURCE_BYTES + slot * UB_NZ_BYTES);
    }
    const int64_t lane = get_subblockid();
    const int64_t expertBase = expert * k * n / 2;
    const int64_t panelBase = (n0 / 256) * (k / K_TILE) * PACKED_TILE_BYTES;
    const int64_t sliceBase = (kBegin / K_TILE) * PACKED_TILE_BYTES;
    const int64_t laneBase = lane * VEC_K_TILE * PACKED_ROW_BYTES + (n0 % 256) / 2 + sliceBase;
    const int64_t iterations = ((kEnd == 0 ? k : kEnd) - kBegin) / K_TILE;
    // Seed both input buffers. MTE2 may load slot1 while VF consumes slot0.
    for (int64_t first = 0; first < 2 && first < iterations; ++first) {
        GlobalPacked globalWeight(weight + expertBase + panelBase + first * PACKED_TILE_BYTES + laneBase);
        TLOAD(raw[first], globalWeight);
        set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(first));
    }
    for (int64_t iteration = 0; iteration < iterations; ++iteration) {
        const uint32_t slot = iteration % 2;
        const auto event = static_cast<event_t>(slot);
        wait_flag(PIPE_MTE2, PIPE_V, event);
        wait_flag(PIPE_MTE3, PIPE_V, event);
        TCVT(nz[slot], packed[slot], RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE2, event);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE2, event);
        // Reuse only the input buffer whose conversion has completed.
        // This prefetch overlaps the following iteration's VF on the other slot.
        if (iteration + 2 < iterations) {
            const int64_t offset = expertBase + panelBase + (iteration + 2) * PACKED_TILE_BYTES + laneBase;
            GlobalPacked globalWeight(weight + offset);
            TLOAD(raw[slot], globalWeight);
            set_flag(PIPE_MTE2, PIPE_V, event);
        }
        wait_flag(PIPE_V, PIPE_MTE3, event);
#if defined(__CPU_SIM)
        TPUSH<WeightPipe, NZ, TileSplitAxis::TILE_UP_DOWN>(weightPipe, nz[slot]);
#else
        wait_intra_block(PIPE_MTE3, static_cast<uint8_t>(1 + 2 * slot));
        TASSIGN(matSlot, L1_BASE + slot * L1_SLOT_BYTES);
        TINSERT(matSlot, nz[slot], static_cast<uint16_t>(lane * VEC_K_TILE), static_cast<uint16_t>(0));
        set_intra_block(PIPE_MTE3, static_cast<uint8_t>(2 * slot));
#endif
        set_flag(PIPE_MTE3, PIPE_V, event);
    }
}
#endif

#ifdef __DAV_CUBE__
template <typename AStage, typename GlobalA>
AICORE inline void loadAStage(AStage& target, __gm__ float8_e4m3_t* x,
                             int64_t k, int64_t rows, Chunk chunk, int64_t start, uint32_t slot)
{
    const auto event = static_cast<event_t>(2 + slot);
    const int64_t validK = k - start < A_STAGE_K ? k - start : A_STAGE_K;
    target.SetValidCol(validK);
    GlobalA globalA(x + chunk.stageRow * k + start,
                    typename GlobalA::Shape(1, 1, 1, chunk.validRows, validK),
                    typename GlobalA::Stride(rows * k, rows * k, rows * k, k, 1));
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    TLOAD(target, globalA);
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
}

template <typename AScale, typename BScale, typename GlobalAScale, typename GlobalBScale>
AICORE inline void loadScaleStage(AScale& aScale, BScale& bScale,
                                 __gm__ float8_e8m0_t* xScale, __gm__ float8_e8m0_t* weightScale,
                                 int64_t k, int64_t n, int64_t experts, int64_t rows,
                                 Chunk chunk, int64_t n0, int64_t start, uint32_t slot)
{
    const auto event = static_cast<event_t>(4 + slot);
    const int64_t q = k / 32;
    const int64_t globalQ = experts * q;
    const int64_t validQ = (k - start < SCALE_STAGE_K ? k - start : SCALE_STAGE_K) / 32;
    const int64_t alignedRows = ((chunk.validRows + 15) / 16) * 16;
    aScale.SetValidRow(alignedRows);
    aScale.SetValidCol(validQ);
    bScale.SetValidRow(validQ);
    const int64_t aOffset = chunk.stageRow * q + (start / 32) * 16;
    const int64_t bOffset = n0 * globalQ + (chunk.expert * q + start / 32) * 16;
    GlobalAScale globalA(xScale + aOffset, typename GlobalAScale::Shape(alignedRows, validQ),
                        typename GlobalAScale::Stride(rows, q));
    GlobalBScale globalB(weightScale + bOffset, typename GlobalBScale::Shape(validQ, N_TILE),
                        typename GlobalBScale::Stride(globalQ, n));
    wait_flag(PIPE_MTE1, PIPE_MTE2, event);
    TLOAD(aScale, globalA);
    TLOAD(bScale, globalB);
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
}

template <uint32_t M, typename OutputSink>
AICORE inline void computeProjection(__gm__ float8_e4m3_t* x, __gm__ float8_e8m0_t* xScale,
                                     __gm__ float8_e8m0_t* weightScale, int64_t k, int64_t n,
                                     int64_t experts, int64_t rows, Chunk chunk, int64_t n0, OutputSink sink)
{
    constexpr int VALID_M = DYNAMIC;
    using MatA = Tile<TileType::Mat, float8_e4m3_t, M, A_STAGE_K, BLayout::ColMajor,
                      VALID_M, DYNAMIC, SLayout::RowMajor, 512>;
    using MatB = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                      K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using MatAScale = Tile<TileType::Mat, float8_e8m0_t, M, SCALE_STAGE_Q, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::RowMajor, 32>;
    using MatBScale = Tile<TileType::Mat, float8_e8m0_t, SCALE_STAGE_Q, N_TILE, BLayout::ColMajor,
                           DYNAMIC, N_TILE, SLayout::ColMajor, 32>;
    using Left = TileLeftCompact<float8_e4m3_t, M, L0_K_TILE, VALID_M>;
    using Right = TileRight<float8_e4m3_t, L0_K_TILE, N_TILE>;
    using LeftScale = TileLeftScaleCompact<float8_e8m0_t, M, L0_SCALE_K_TILE, DYNAMIC>;
    using RightScale = TileRightScale<float8_e8m0_t, L0_SCALE_K_TILE, N_TILE>;
    using Acc = TileAccCompact<float, M, N_TILE, VALID_M>;
    using GlobalA = GlobalTensor<float8_e4m3_t, Shape<1, 1, 1, VALID_M, DYNAMIC>,
                                 pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    using GlobalAScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ZZ>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_A_ZZ>, Layout::MX_A_ZZ>;
    using GlobalBScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, N_TILE, Layout::MX_B_NN>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_B_NN>, Layout::MX_B_NN>;
    MatA aMat[2];
    MatB bMat;
    MatAScale aScaleMat[2];
    MatBScale bScaleMat[2];
    Left a[2];
    Right b[2];
    LeftScale aScale[2];
    RightScale bScale[2];
    Acc acc;
    {
        aMat[0].SetValidRow(chunk.validRows);
        aMat[1].SetValidRow(chunk.validRows);
        a[0].SetValidRow(chunk.validRows);
        a[1].SetValidRow(chunk.validRows);
        acc.SetValidRow(chunk.validRows);
        aScale[0].SetValidRow(chunk.validRows);
        aScale[1].SetValidRow(chunk.validRows);
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        TASSIGN(aMat[slot], slot * M_TILE * A_STAGE_K);
        TASSIGN(aScaleMat[slot], 2 * M_TILE * A_STAGE_K + slot * M_TILE * SCALE_STAGE_Q);
        TASSIGN(bScaleMat[slot], 2 * M_TILE * (A_STAGE_K + SCALE_STAGE_Q) + slot * N_TILE * SCALE_STAGE_Q);
        TASSIGN(a[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(b[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(aScale[slot], GetScaleAddr(a[slot].data()));
        TASSIGN(bScale[slot], GetScaleAddr(b[slot].data()));
    }
    TASSIGN(acc, 0);
    loadAStage<MatA, GlobalA>(aMat[0], x, k, rows, chunk, 0, 0);
    loadScaleStage<MatAScale, MatBScale, GlobalAScale, GlobalBScale>(
        aScaleMat[0], bScaleMat[0], xScale, weightScale, k, n, experts, rows, chunk, n0, 0, 0);
    for (int64_t k0 = 0; k0 < k; k0 += K_TILE) {
        const uint32_t slot = (k0 / K_TILE) % 2;
        const uint32_t aSlot = (k0 / A_STAGE_K) % 2;
        const uint32_t scaleSlot = (k0 / SCALE_STAGE_K) % 2;
        const uint16_t aOffset = k0 % A_STAGE_K;
        const uint16_t scaleOffset = (k0 % SCALE_STAGE_K) / 32;
        if (aOffset == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2 + aSlot));
        }
        if (scaleOffset == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(4 + scaleSlot));
        }
#if defined(__CPU_SIM)
        TPOP<WeightPipe, MatB, TileSplitAxis::TILE_UP_DOWN>(weightPipe, bMat);
#else
        waitSlot(slot);
        TASSIGN(bMat, L1_BASE + slot * L1_SLOT_BYTES);
#endif
        for (uint32_t half = 0; half < 2; ++half) {
            const auto event = static_cast<event_t>(half);
            const uint16_t offset = half * L0_K_TILE;
            const uint16_t qOffset = half * L0_SCALE_K_TILE;
            wait_flag(PIPE_M, PIPE_MTE1, event);
            TEXTRACT(a[half], aMat[aSlot], 0, static_cast<uint16_t>(aOffset + offset));
            TEXTRACT(b[half], bMat, offset, 0);
            TEXTRACT(aScale[half], aScaleMat[scaleSlot], 0, static_cast<uint16_t>(scaleOffset + qOffset));
            TEXTRACT(bScale[half], bScaleMat[scaleSlot], static_cast<uint16_t>(scaleOffset + qOffset), 0);
            if (half == 1) {
                releaseSlot(slot);
                if (aOffset + K_TILE == A_STAGE_K || k0 + K_TILE == k) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + aSlot));
                }
                if (k0 % SCALE_STAGE_K + K_TILE == SCALE_STAGE_K || k0 + K_TILE == k) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + scaleSlot));
                }
            }
            set_flag(PIPE_MTE1, PIPE_M, event);
            wait_flag(PIPE_MTE1, PIPE_M, event);
            if (k0 == 0 && half == 0) {
                TMATMUL_MX<AccPhase::Partial>(acc, a[half], aScale[half], b[half], bScale[half]);
            } else if (k0 + K_TILE == k && half == 1) {
                TMATMUL_MX<AccPhase::Final>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            } else {
                TMATMUL_MX<AccPhase::Partial>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            }
            set_flag(PIPE_M, PIPE_MTE1, event);
        }
        if (aOffset == 0 && k0 + A_STAGE_K < k) {
            loadAStage<MatA, GlobalA>(aMat[1-aSlot], x, k, rows, chunk, k0 + A_STAGE_K, 1-aSlot);
        }
        if (scaleOffset == 0 && k0 + SCALE_STAGE_K < k) {
            loadScaleStage<MatAScale, MatBScale, GlobalAScale, GlobalBScale>(
                aScaleMat[1-scaleSlot], bScaleMat[1-scaleSlot], xScale, weightScale,
                k, n, experts, rows, chunk, n0, k0 + SCALE_STAGE_K, 1-scaleSlot);
        }
    }
    sink(acc);
}

#endif

AICORE inline void beginGemm()
{
    for (uint32_t slot = 0; slot < 2; ++slot) {
#ifdef __DAV_CUBE__
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + slot));
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + slot));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));

#if !defined(__CPU_SIM)
        releaseSlot(slot);
#endif
#endif
#ifdef __DAV_VEC__
        set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
#endif
    }
}

AICORE inline void endGemm()
{
    for (uint32_t slot = 0; slot < 2; ++slot) {
#ifdef __DAV_CUBE__
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + slot));
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4 + slot));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
#endif
#ifdef __DAV_VEC__
        wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(slot));
#if !defined(__CPU_SIM)
        wait_intra_block(PIPE_MTE3, static_cast<uint8_t>(1 + 2 * slot));
#endif
#endif
    }
    pipe_barrier(PIPE_ALL);
}

}  // namespace grouped_mxfp4

#if defined(__CPU_SIM)
#include <cmath>
#include <cstring>
#endif

using namespace pto;
using namespace grouped_mxfp4;

namespace {

// Local capacity for either half of the 2304-column routed activation.
constexpr int H_CAP = 1280;
constexpr int GATE_UB = 100352;
constexpr int UP_UB = 133120;
constexpr int ACT_UB = 165888;
constexpr int SCALE_UB = 174080;
constexpr int WEIGHT_UB = 174336;
constexpr int ACT_L1 = L1_BASE + L1_SLOT_COUNT * L1_SLOT_BYTES;
constexpr int SCALE_L1 = ACT_L1 + M_TILE * H_CAP;
static_assert(2 * (UB_SOURCE_BYTES + UB_NZ_BYTES) <= GATE_UB);
static_assert(SCALE_L1 + M_TILE * H_CAP / 32 <= 512 * 1024);
static_assert(WEIGHT_UB + M_TILE * sizeof(float) <= 240 * 1024);

#ifdef __DAV_VEC__
AICORE inline void activateQuantize(__ubuf__ float* gate, __ubuf__ float* up,
                                   __ubuf__ float* weights, __ubuf__ uint8_t* out,
                                   __ubuf__ uint8_t* codes, uint16_t validRows, uint16_t paddedRows)
{
#if defined(__CPU_SIM)
    using Input = Tile<TileType::Vec, float, M_TILE, 128>;
    using Weight = Tile<TileType::Vec, float, 1, M_TILE>;
    using Payload = Tile<TileType::Vec, uint8_t, 1, M_TILE * 128>;
    using Codes = Tile<TileType::Vec, uint8_t, 1, M_TILE * 4>;
    Input gateTile, upTile;
    Weight weightTile;
    Payload payloadTile;
    Codes codeTile;
    TASSIGN(gateTile, reinterpret_cast<uintptr_t>(gate));
    TASSIGN(upTile, reinterpret_cast<uintptr_t>(up));
    TASSIGN(weightTile, reinterpret_cast<uintptr_t>(weights));
    TASSIGN(payloadTile, reinterpret_cast<uintptr_t>(out));
    TASSIGN(codeTile, reinterpret_cast<uintptr_t>(codes));
    gate = gateTile.data();
    up = upTile.data();
    weights = weightTile.data();
    out = payloadTile.data();
    codes = codeTile.data();
    for (uint16_t row = 0; row < paddedRows; ++row) {
        for (uint16_t group = 0; group < 4; ++group) {
            float values[32] = {};
            float maximum = 1e-4f;
            for (uint16_t col = 0; col < 32; ++col) {
                if (row < validRows) {
                    const uint16_t offset = row * 128 + group * 32 + col;
                    const float g = std::fmin(gate[offset], 10.0f);
                    const float u = std::fmax(std::fmin(up[offset], 10.0f), -10.0f);
                    values[col] = (g / (1.0f + std::exp(-g))) * u * weights[row];
                }
                maximum = std::fmax(maximum, std::fabs(values[col]));
            }
            maximum *= 1.0f / 448.0f;
            uint32_t bits;
            std::memcpy(&bits, &maximum, sizeof(bits));
            const uint32_t exponent = (bits + 8388607) >> 23;
            const uint16_t scaleOffset = (row / 16) * 64 + (group / 2) * 32 + (row % 16) * 2 + group % 2;
            codes[scaleOffset] = static_cast<uint8_t>(exponent);
            bits = (254 - exponent) << 23;
            float reciprocal;
            std::memcpy(&reciprocal, &bits, sizeof(reciprocal));
            for (uint16_t col = 0; col < 32; ++col) {
                const float normalized = std::fmax(std::fmin(values[col] * reciprocal, 448.0f), -448.0f);
                const float8_e4m3_t value(normalized, RoundMode::CAST_RINT, SaturationMode::ON);
                out[group * M_TILE * 32 + row * 32 + col] = value.RawData();
            }
        }
    }
#else
    __VEC_SCOPE__ {
        constexpr auto halfStore = static_cast<::DistVST>(pto::DistVST::DIST_ONEPT_B16);
        vector_bool all = pset_b32(PAT_ALL);
        vector_bool lower = pset_b32(PAT_VL32);
        vector_bool upper;
        pnot(upper, lower, all);
        vector_bool four = pset_b32(PAT_VL4);
        vector_bool one = pset_b32(PAT_VL1);
        vector_bool one16 = pset_b16(PAT_VL1);
        vector_s32 maxReciprocalExponent;
        vbr(maxReciprocalExponent, 254);
        for (uint16_t row = 0; row < validRows; ++row) {
            vector_f32 weight;
            vlds(weight, weights, row, BRC_B32);
            vector_f32 g0, g1, u0, u1;
            vlds(g0, gate, row * 128, NORM);
            vlds(g1, gate, row * 128 + 64, NORM);
            vlds(u0, up, row * 128, NORM);
            vlds(u1, up, row * 128 + 64, NORM);
            vmins(g0, g0, 10.0f, all);
            vmins(g1, g1, 10.0f, all);
            vmins(u0, u0, 10.0f, all);
            vmins(u1, u1, 10.0f, all);
            vmaxs(u0, u0, -10.0f, all);
            vmaxs(u1, u1, -10.0f, all);
            vector_f32 sigmoid0, sigmoid1, hiddenLow, hiddenHigh;
            vmuls(sigmoid0, g0, -1.0f, all);
            vmuls(sigmoid1, g1, -1.0f, all);
            vexp(sigmoid0, sigmoid0, all);
            vexp(sigmoid1, sigmoid1, all);
            vadds(sigmoid0, sigmoid0, 1.0f, all);
            vadds(sigmoid1, sigmoid1, 1.0f, all);
            vdiv(hiddenLow, g0, sigmoid0, all);
            vdiv(hiddenHigh, g1, sigmoid1, all);
            vmul(hiddenLow, hiddenLow, u0, all);
            vmul(hiddenHigh, hiddenHigh, u1, all);
            vmul(hiddenLow, hiddenLow, weight, all);
            vmul(hiddenHigh, hiddenHigh, weight, all);
            vector_f32 h0, h1;
            vdintlv(h0, h1, hiddenLow, hiddenHigh);
            vector_f32 magnitude0, magnitude1, maximum, groups, even, odd;
            vabs(magnitude0, h0, all);
            vabs(magnitude1, h1, all);
            vmax(maximum, magnitude0, magnitude1, all);
            vcgmax(groups, maximum, all);
            vdintlv(even, odd, groups, groups);
            vmax(maximum, even, odd, four);
            vmaxs(maximum, maximum, 1e-4f, four);
            vmuls(maximum, maximum, 1.0f / 448.0f, four);
            vector_s32 exponent, reciprocal, expanded, unused;
            vadds(exponent, (vector_s32&)maximum, 8388607, four);
            vshrs(exponent, exponent, 23, four);
            vector_u16 packed16;
            vector_u8 packed8;
            vpack(packed16, (vector_u32&)exponent, LOWER);
            vpack(packed8, packed16, LOWER);
            const uint16_t scaleOffset = (row / 16) * 64 + (row % 16) * 2;
            vsts((vector_u16&)packed8, (__ubuf__ uint16_t*)codes, scaleOffset / 2,
                 std::integral_constant<::DistVST, halfStore>(), one16);
            vector_s32 highExponent;
            vshrs(highExponent, (vector_s32&)packed8, 16, one);
            vsts((vector_u16&)highExponent, (__ubuf__ uint16_t*)codes, (scaleOffset + 32) / 2,
                 std::integral_constant<::DistVST, halfStore>(), one16);
            vsub(reciprocal, maxReciprocalExponent, exponent, four);
            vshls(reciprocal, reciprocal, 23, four);
            vintlv(expanded, unused, reciprocal, reciprocal);
            vintlv(reciprocal, unused, expanded, expanded);
            vintlv(expanded, unused, reciprocal, reciprocal);
            vintlv(reciprocal, unused, expanded, expanded);
            vmul(h0, h0, (vector_f32&)reciprocal, all);
            vmul(h1, h1, (vector_f32&)reciprocal, all);
            vector_f32 low, high;
            vintlv(low, high, h0, h1);
            vector_f8e4m3 payload0, payload1;
            vcvt(payload0, low, all, ROUND_R, RS_ENABLE, PART_P0);
            vcvt(payload1, high, all, ROUND_R, RS_ENABLE, PART_P0);
            vsts((vector_u8&)payload0, out, row * 32, PK4_B32, lower);
            vsts((vector_u8&)payload0, out, M_TILE * 32 + row * 32 - 32, PK4_B32, upper);
            vsts((vector_u8&)payload1, out, 2 * M_TILE * 32 + row * 32, PK4_B32, lower);
            vsts((vector_u8&)payload1, out, 3 * M_TILE * 32 + row * 32 - 32, PK4_B32, upper);
        }
        if (validRows < paddedRows) {
            vector_f32 zero;
            vector_f8e4m3 zeroPayload;
            vector_s32 zeroExponent;
            vbr(zero, 0.0f);
            vcvt(zeroPayload, zero, all, ROUND_R, RS_ENABLE, PART_P0);
            vbr(zeroExponent, 105 | (105 << 8));
            for (uint16_t row = validRows; row < paddedRows; ++row) {
                for (uint16_t col = 0; col < 128; col += 64) {
                    const uint16_t scaleOffset = (row / 16) * 64 + (col / 64) * 32 + (row % 16) * 2;
                    vsts((vector_u16&)zeroExponent, (__ubuf__ uint16_t*)codes, scaleOffset / 2,
                         std::integral_constant<::DistVST, halfStore>(), one16);
                    vsts((vector_u8&)zeroPayload, out, (col / 32) * M_TILE * 32 + row * 32, PK4_B32, lower);
                    vsts((vector_u8&)zeroPayload, out, (col / 32 + 1) * M_TILE * 32 + row * 32 - 32, PK4_B32, upper);
                }
            }
        }
    }
#endif
}


#endif

#ifdef __DAV_CUBE__
struct ActivateProjection {
    uint32_t ubOffset;

    template <typename Acc>
    AICORE inline void operator()(Acc& acc) const
    {
#if defined(__CPU_SIM)
        TPUSH<ProjectionPipe, Acc, TileSplitAxis::TILE_LEFT_RIGHT>(projectionPipe, acc);
#else
        using VecResult = Tile<TileType::Vec, float, M_TILE, N_TILE / 2, BLayout::RowMajor, DYNAMIC, N_TILE / 2>;
        VecResult target;
        target.SetValidRow(acc.GetValidRow());
        TASSIGN(target, ubOffset);
        if (ubOffset == GATE_UB) {
            wait_intra_block(PIPE_FIX, 5);
            wait_intra_block(PIPE_FIX, 5 + VEC_CORE_ID_OFFSET);
        }
        TMOV<STPhase::Final, VecResult, Acc, AccToVecMode::DualModeSplitN>(target, acc);
        if (ubOffset == UP_UB) {
            set_intra_block(PIPE_FIX, 4);
            set_intra_block(PIPE_FIX, 4 + VEC_CORE_ID_OFFSET);
        }
#endif
    }
};
#endif

#ifdef __DAV_CUBE__
AICORE inline void consumeDown(__gm__ float8_e8m0_t* weightScale, __gm__ float* out,
                              int64_t k, int64_t n, int64_t experts, int64_t rowStride,
                              Chunk chunk, int64_t n0, int64_t kBegin, int64_t kEnd, int64_t outputRow)
{
    using MatA = Tile<TileType::Mat, float8_e4m3_t, M_TILE, H_CAP, BLayout::ColMajor,
                      DYNAMIC, H_CAP, SLayout::RowMajor, 512>;
    using MatB = Tile<TileType::Mat, float8_e4m3_t, K_TILE, N_TILE, BLayout::ColMajor,
                      K_TILE, N_TILE, SLayout::RowMajor, 512>;
    using MatAScale = Tile<TileType::Mat, float8_e8m0_t, M_TILE, H_CAP / 32, BLayout::RowMajor,
                           DYNAMIC, H_CAP / 32, SLayout::RowMajor, 32>;
    using MatBScale = Tile<TileType::Mat, float8_e8m0_t, SCALE_STAGE_Q, N_TILE, BLayout::ColMajor,
                           DYNAMIC, N_TILE, SLayout::ColMajor, 32>;
    using Left = TileLeftCompact<float8_e4m3_t, M_TILE, L0_K_TILE, DYNAMIC>;
    using Right = TileRight<float8_e4m3_t, L0_K_TILE, N_TILE>;
    using LeftScale = TileLeftScaleCompact<float8_e8m0_t, M_TILE, L0_SCALE_K_TILE, DYNAMIC>;
    using RightScale = TileRightScale<float8_e8m0_t, L0_SCALE_K_TILE, N_TILE>;
    using Acc = TileAccCompact<float, M_TILE, N_TILE, DYNAMIC>;
    using GlobalBScale = GlobalTensor<float8_e8m0_t, TileShape2D<float8_e8m0_t, DYNAMIC, N_TILE, Layout::MX_B_NN>,
                                      BaseShape2D<float8_e8m0_t, DYNAMIC, DYNAMIC, Layout::MX_B_NN>, Layout::MX_B_NN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, N_TILE>,
                                   pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;
    MatA aMat;
    MatB bMat;
    MatAScale aScaleMat;
    MatBScale bScaleMat;
    Left a[2];
    Right b[2];
    LeftScale aScale[2];
    RightScale bScale[2];
    Acc acc;
    aMat.SetValidRow(chunk.validRows);
    aScaleMat.SetValidRow((chunk.validRows + 15) / 16 * 16);
    bScaleMat.SetValidRow((kEnd - kBegin) / 32);
    acc.SetValidRow(chunk.validRows);
    TASSIGN(aMat, ACT_L1);
    TASSIGN(aScaleMat, SCALE_L1);
    TASSIGN(bScaleMat, 2 * M_TILE * (A_STAGE_K + SCALE_STAGE_Q));
    TASSIGN(acc, 0);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        a[slot].SetValidRow(chunk.validRows);
        aScale[slot].SetValidRow(chunk.validRows);
        TASSIGN(a[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(b[slot], slot * L0_PINGPONG_BYTES);
        TASSIGN(aScale[slot], GetScaleAddr(a[slot].data()));
        TASSIGN(bScale[slot], GetScaleAddr(b[slot].data()));
    }
    const int64_t globalQ = experts * (k / 32);
    const int64_t bOffset = n0 * globalQ + (chunk.expert * (k / 32) + kBegin / 32) * 16;
    GlobalBScale globalScale(weightScale + bOffset,
                             GlobalBScale::Shape((kEnd-kBegin)/32, N_TILE), GlobalBScale::Stride(globalQ, n));
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    TLOAD(bScaleMat, globalScale);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    for (int64_t k0 = 0; k0 < kEnd - kBegin; k0 += K_TILE) {
        const uint32_t slot = (k0 / K_TILE) % 2;
#if defined(__CPU_SIM)
        TPOP<WeightPipe, MatB, TileSplitAxis::TILE_UP_DOWN>(weightPipe, bMat);
#else
        waitSlot(slot);
        TASSIGN(bMat, L1_BASE + slot * L1_SLOT_BYTES);
#endif
        for (uint32_t half = 0; half < 2; ++half) {
            const auto event = static_cast<event_t>(half);
            const uint16_t offset = half * L0_K_TILE;
            const uint16_t qOffset = half * L0_SCALE_K_TILE;
            wait_flag(PIPE_M, PIPE_MTE1, event);
            TEXTRACT(a[half], aMat, 0, static_cast<uint16_t>(k0 + offset));
            TEXTRACT(b[half], bMat, offset, 0);
            TEXTRACT(aScale[half], aScaleMat, 0, static_cast<uint16_t>(k0 / 32 + qOffset));
            TEXTRACT(bScale[half], bScaleMat, static_cast<uint16_t>(k0 / 32 + qOffset), 0);
            if (half == 1) {
                releaseSlot(slot);
                if (k0 + K_TILE == kEnd - kBegin) {
                    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
                }
            }
            set_flag(PIPE_MTE1, PIPE_M, event);
            wait_flag(PIPE_MTE1, PIPE_M, event);
            if (k0 == 0 && half == 0) {
                TMATMUL_MX<AccPhase::Partial>(acc, a[half], aScale[half], b[half], bScale[half]);
            } else if (k0 + K_TILE == kEnd - kBegin && half == 1) {
                TMATMUL_MX<AccPhase::Final>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            } else {
                TMATMUL_MX<AccPhase::Partial>(acc, acc, a[half], aScale[half], b[half], bScale[half]);
            }
            set_flag(PIPE_M, PIPE_MTE1, event);
        }
    }
    GlobalOut globalOut(out + outputRow * n + n0,
                        GlobalOut::Shape(1, 1, 1, chunk.validRows, N_TILE),
                        GlobalOut::Stride(rowStride * n, rowStride * n, rowStride * n, n, 1));
    TSTORE<STPhase::Final>(globalOut, acc);
}
#endif

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    auto* counts = tensorData<int32_t>(args, 3);
    const int64_t experts = args[11];
    const int64_t rowStride = args[12];
    const int64_t width = args[13];
    const int64_t hidden = args[14];
    const int64_t parts = args[15];
    const int64_t partialStride = args[16];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
#ifdef __DAV_CUBE__
    auto* x = tensorData<float8_e4m3_t>(args, 0);
    auto* xScale = tensorData<float8_e8m0_t>(args, 1);
    auto* w1Scale = tensorData<float8_e8m0_t>(args, 5);
    auto* w3Scale = tensorData<float8_e8m0_t>(args, 7);
    auto* w2Scale = tensorData<float8_e8m0_t>(args, 9);
    auto* out = tensorData<float>(args, 10);
#endif
#ifdef __DAV_VEC__
    auto* weights = tensorData<float>(args, 2);
    auto* w1 = tensorData<uint8_t>(args, 4);
    auto* w3 = tensorData<uint8_t>(args, 6);
    auto* w2 = tensorData<uint8_t>(args, 8);
    const int64_t lane = get_subblockid();
    using Weight = Tile<TileType::Vec, float, 1, M_TILE, BLayout::RowMajor, 1, DYNAMIC>;
    using GlobalWeight = GlobalTensor<float, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<64, 64, 64, 64, 1>>;
    using Payload = Tile<TileType::Vec, float8_e4m3_t, M_TILE, N_TILE / 2, BLayout::ColMajor,
                         DYNAMIC, N_TILE / 2, SLayout::RowMajor, 512>;
    using Hidden = Tile<TileType::Mat, float8_e4m3_t, M_TILE, H_CAP, BLayout::ColMajor,
                        DYNAMIC, H_CAP, SLayout::RowMajor, 512>;
    Weight weight;
    Payload payload;
    Hidden hMat;
    TASSIGN(weight, WEIGHT_UB);
    TASSIGN(payload, ACT_UB);
    TASSIGN(hMat, ACT_L1);
#endif
    beginGemm();
    int64_t workBase = 0;
    int64_t compactBase = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
        const int64_t count = acquireCount(counts, expert);
        const int64_t workCount = ((count + M_TILE - 1) / M_TILE) * parts;
        const int64_t first = (block + blocks - workBase % blocks) % blocks;
        for (int64_t work = first; work < workCount; work += blocks) {
            const int64_t part = work % parts;
            const int64_t localRow = work / parts * M_TILE;
            const int64_t valid = count - localRow < M_TILE ? count - localRow : M_TILE;
            const int64_t padded = (valid + 15) / 16 * 16;
            const int64_t row = expert * rowStride + localRow;
            const Chunk chunk{expert, row, valid};
            const int64_t hBegin = (hidden / N_TILE) * part / parts;
            const int64_t hEnd = (hidden / N_TILE) * (part + 1) / parts;
#ifdef __DAV_VEC__
            weight.SetValidCol(padded);
            payload.SetValidRow(padded);
            hMat.SetValidRow(padded);
            GlobalWeight globalWeight(weights + row, GlobalWeight::Shape(1, 1, 1, 1, padded));
            TLOAD(weight, globalWeight);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
#if !defined(__CPU_SIM)
            set_intra_block(PIPE_V, 5);
#endif
#endif
            for (int64_t h = hBegin; h < hEnd; ++h) {
#ifdef __DAV_VEC__
                produceWeight(w1, width, hidden, expert, h * N_TILE);
                produceWeight(w3, width, hidden, expert, h * N_TILE);
#if !defined(__CPU_SIM)
                wait_intra_block(PIPE_V, 4);
#endif
#if defined(__CPU_SIM)
                using Projection = Tile<TileType::Vec, float, M_TILE, N_TILE / 2,
                                        BLayout::RowMajor, DYNAMIC, N_TILE / 2>;
                for (int address : {GATE_UB, UP_UB}) {
                    Projection incoming, local;
                    incoming.SetValidRow(valid);
                    local.SetValidRow(valid);
                    TPOP<ProjectionPipe, Projection, TileSplitAxis::TILE_LEFT_RIGHT>(projectionPipe, incoming);
                    TASSIGN(local, address);
                    TMOV(local, incoming);
                    TFREE<ProjectionPipe, TileSplitAxis::TILE_LEFT_RIGHT>(projectionPipe);
                }
#endif
                activateQuantize((__ubuf__ float*)GATE_UB, (__ubuf__ float*)UP_UB,
                                 (__ubuf__ float*)WEIGHT_UB, (__ubuf__ uint8_t*)ACT_UB,
                                 (__ubuf__ uint8_t*)SCALE_UB, valid, padded);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
#if defined(__CPU_SIM)
                TPUSH<HiddenPipe, Payload, TileSplitAxis::TILE_LEFT_RIGHT>(hiddenPipe, payload);
                using Codes = Tile<TileType::Vec, uint8_t, 1, M_TILE * 4>;
                Codes codes;
                TASSIGN(codes, SCALE_UB);
                TPUSH<ScalePipe, Codes, TileSplitAxis::TILE_LEFT_RIGHT>(scalePipe, codes);
#else
                TINSERT(hMat, payload, 0, (h - hBegin) * N_TILE + lane * (N_TILE / 2));
                auto* scaleDestination = (__cbuf__ uint8_t*)SCALE_L1 + (h - hBegin) * 128 + lane * 64;
                copy_ubuf_to_cbuf(scaleDestination, (__ubuf__ uint8_t*)SCALE_UB, 0, padded / 16,
                                 2, 0, (H_CAP / 32 * 16 - 64) / 32);
#endif
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
#if !defined(__CPU_SIM)
                set_intra_block(PIPE_MTE3, 5);
#endif
#endif
#ifdef __DAV_CUBE__
                computeProjection<M_TILE>(x, xScale, w1Scale, width, hidden, experts,
                                          experts * rowStride, chunk, h * N_TILE, ActivateProjection{GATE_UB});
                computeProjection<M_TILE>(x, xScale, w3Scale, width, hidden, experts,
                                          experts * rowStride, chunk, h * N_TILE, ActivateProjection{UP_UB});
#if defined(__CPU_SIM)
                using HiddenTile = Tile<TileType::Mat, float8_e4m3_t, M_TILE, N_TILE,
                                        BLayout::ColMajor, DYNAMIC, N_TILE, SLayout::RowMajor, 512>;
                using HiddenMat = Tile<TileType::Mat, float8_e4m3_t, M_TILE, H_CAP,
                                       BLayout::ColMajor, DYNAMIC, H_CAP, SLayout::RowMajor, 512>;
                using Codes = Tile<TileType::Mat, uint8_t, 1, M_TILE * 8>;
                HiddenTile incoming;
                HiddenMat local;
                incoming.SetValidRow(padded);
                local.SetValidRow(padded);
                TPOP<HiddenPipe, HiddenTile, TileSplitAxis::TILE_LEFT_RIGHT>(hiddenPipe, incoming);
                TASSIGN(local, ACT_L1);
                TINSERT(local, incoming, 0, (h - hBegin) * N_TILE);
                TFREE<HiddenPipe, TileSplitAxis::TILE_LEFT_RIGHT>(hiddenPipe);
                Codes codes;
                TPOP<ScalePipe, Codes, TileSplitAxis::TILE_LEFT_RIGHT>(scalePipe, codes);
                using ScaleMat = Tile<TileType::Mat, float8_e8m0_t, M_TILE, H_CAP / 32,
                                      BLayout::RowMajor, DYNAMIC, H_CAP / 32, SLayout::RowMajor, 32>;
                ScaleMat scales;
                TASSIGN(scales, SCALE_L1);
                auto* destination = reinterpret_cast<uint8_t*>(scales.data());
                for (int lane = 0; lane < 2; ++lane) {
                    for (int rowBlock = 0; rowBlock < padded / 16; ++rowBlock) {
                        std::memcpy(destination + rowBlock * (H_CAP / 32 * 16) + (h - hBegin) * 128 + lane * 64,
                                    codes.data() + lane * M_TILE * 4 + rowBlock * 64, 64);
                    }
                }
                TFREE<ScalePipe, TileSplitAxis::TILE_LEFT_RIGHT>(scalePipe);
#endif
#endif
            }
#ifdef __DAV_CUBE__
#if !defined(__CPU_SIM)
            wait_intra_block(PIPE_MTE1, 5);
            wait_intra_block(PIPE_MTE1, 5 + VEC_CORE_ID_OFFSET);
#endif
#endif
            for (int64_t n0 = 0; n0 < width; n0 += N_TILE) {
#ifdef __DAV_VEC__
                produceWeight(w2, hidden, width, expert, n0, hBegin * N_TILE, hEnd * N_TILE);
#endif
#ifdef __DAV_CUBE__
                consumeDown(w2Scale, out, hidden, width, experts, rowStride, chunk, n0,
                            hBegin * N_TILE, hEnd * N_TILE, part * partialStride + compactBase + localRow);
#endif
            }
        }
        workBase += workCount;
        compactBase += count;
    }
    endGemm();
}
