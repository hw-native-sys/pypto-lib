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

// Local capacity for either half of the 2304-column routed activation.
constexpr int H_CAP = 1280;
constexpr int GATE_UB = 100352;
constexpr int UP_UB = 133120;
constexpr int ACT_UB = 165888;
constexpr int SCALE_UB = 174080;
constexpr int WEIGHT_UB = 174336;
constexpr int ACT_L1 = L1_BASE + L1_SLOT_COUNT * L1_SLOT_BYTES;
constexpr int SCALE_L1 = ACT_L1 + M_TILE * H_CAP;
static_assert(SCALE_L1 + M_TILE * H_CAP / 32 <= 512 * 1024);
static_assert(WEIGHT_UB + M_TILE * sizeof(float) <= 240 * 1024);

#ifdef __DAV_VEC__
AICORE inline void activateQuantize(__ubuf__ float* gate, __ubuf__ float* up,
                                   __ubuf__ float* weights, __ubuf__ uint8_t* out,
                                   __ubuf__ uint8_t* codes, uint16_t validRows, uint16_t paddedRows)
{
    __VEC_SCOPE__ {
        constexpr auto byteStore = static_cast<::DistVST>(pto::DistVST::DIST_ONEPT_B8);
        vector_bool all = pset_b32(PAT_ALL);
        vector_bool lower = pset_b32(PAT_VL32);
        vector_bool upper;
        pnot(upper, lower, all);
        vector_bool one = pset_b32(PAT_VL1);
        vector_s32 maxReciprocalExponent;
        vbr(maxReciprocalExponent, 254);
        for (uint16_t row = 0; row < paddedRows; ++row) {
            vector_f32 weight;
            vlds(weight, weights, row, BRC_B32);
            for (uint16_t col = 0; col < 128; col += 64) {
                vector_f32 hidden;
                vbr(hidden, 0.0f);
                if (row < validRows) {
                    vector_f32 g, u, sigmoid;
                    vlds(g, gate, row * 128 + col, NORM);
                    vlds(u, up, row * 128 + col, NORM);
                    vmins(g, g, 10.0f, all);
                    vmins(u, u, 10.0f, all);
                    vmaxs(u, u, -10.0f, all);
                    vmuls(sigmoid, g, -1.0f, all);
                    vexp(sigmoid, sigmoid, all);
                    vadds(sigmoid, sigmoid, 1.0f, all);
                    vdiv(hidden, g, sigmoid, all);
                    vmul(hidden, hidden, u, all);
                    vmul(hidden, hidden, weight, all);
                }
                vector_f32 magnitude, max0, max1;
                vabs(magnitude, hidden, all);
                vcmax(max0, magnitude, lower, MODE_ZEROING);
                vcmax(max1, magnitude, upper, MODE_ZEROING);
                vmaxs(max0, max0, 1e-4f, one);
                vmaxs(max1, max1, 1e-4f, one);
                vmuls(max0, max0, 1.0f / 448.0f, one);
                vmuls(max1, max1, 1.0f / 448.0f, one);
                vector_s32 exp0, exp1, exponent, scaleBits;
                vadds(exp0, (vector_s32&)max0, 8388607, one);
                vadds(exp1, (vector_s32&)max1, 8388607, one);
                vshrs(exp0, exp0, 23, one);
                vshrs(exp1, exp1, 23, one);
                const uint16_t scaleOffset = (row / 16) * 64 + (col / 64) * 32 + (row % 16) * 2;
                vsts((vector_u8&)exp0, codes, scaleOffset, std::integral_constant<::DistVST, byteStore>(), one);
                vsts((vector_u8&)exp1, codes, scaleOffset + 1, std::integral_constant<::DistVST, byteStore>(), one);
                vdup(exp0, exp0, all, POS_LOWEST, MODE_ZEROING);
                vdup(exp1, exp1, all, POS_LOWEST, MODE_ZEROING);
                vsel(exponent, exp0, exp1, lower);
                vsub(scaleBits, maxReciprocalExponent, exponent, all);
                vshls(scaleBits, scaleBits, 23, all);
                vmul(hidden, hidden, (vector_f32&)scaleBits, all);
                vmaxs(hidden, hidden, -448.0f, all);
                vmins(hidden, hidden, 448.0f, all);
                vector_f8e4m3 payload;
                vcvt(payload, hidden, all, ROUND_R, RS_ENABLE, PART_P0);
                vsts((vector_u8&)payload, out, (col / 32) * M_TILE * 32 + row * 32, PK4_B32, lower);
                vsts((vector_u8&)payload, out, (col / 32 + 1) * M_TILE * 32 + row * 32 - 32, PK4_B32, upper);
            }
        }

    }
}


#endif

#ifdef __DAV_CUBE__
struct ActivateProjection {
    uint32_t ubOffset;

    template <typename Acc>
    AICORE inline void operator()(Acc& acc) const
    {
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
        waitSlot(slot);
        TASSIGN(bMat, L1_BASE + slot * L1_SLOT_BYTES);
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
            set_intra_block(PIPE_V, 5);
#endif
            for (int64_t h = hBegin; h < hEnd; ++h) {
#ifdef __DAV_VEC__
                produceWeight(w1, width, hidden, expert, h * N_TILE);
                produceWeight(w3, width, hidden, expert, h * N_TILE);
                wait_intra_block(PIPE_V, 4);
                activateQuantize((__ubuf__ float*)GATE_UB, (__ubuf__ float*)UP_UB,
                                 (__ubuf__ float*)WEIGHT_UB, (__ubuf__ uint8_t*)ACT_UB,
                                 (__ubuf__ uint8_t*)SCALE_UB, valid, padded);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
                TINSERT(hMat, payload, 0, (h - hBegin) * N_TILE + lane * (N_TILE / 2));
                auto* scaleDestination = (__cbuf__ uint8_t*)SCALE_L1 + (h - hBegin) * 128 + lane * 64;
                copy_ubuf_to_cbuf(scaleDestination, (__ubuf__ uint8_t*)SCALE_UB, 0, padded / 16,
                                 2, 0, (H_CAP / 32 * 16 - 64) / 32);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
                set_intra_block(PIPE_MTE3, 5);
#endif
#ifdef __DAV_CUBE__
                computeProjection<M_TILE>(x, xScale, w1Scale, width, hidden, experts,
                                          experts * rowStride, chunk, h * N_TILE, ActivateProjection{GATE_UB});
                computeProjection<M_TILE>(x, xScale, w3Scale, width, hidden, experts,
                                          experts * rowStride, chunk, h * N_TILE, ActivateProjection{UP_UB});
#endif
            }
#ifdef __DAV_CUBE__
            wait_intra_block(PIPE_MTE1, 5);
            wait_intra_block(PIPE_MTE1, 5 + VEC_CORE_ID_OFFSET);
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
