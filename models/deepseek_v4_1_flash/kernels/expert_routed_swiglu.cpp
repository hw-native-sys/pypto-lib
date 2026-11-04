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

#include "tensor.h"
#include "intrinsic.h"

#if defined(__CPU_SIM)
#include <cmath>
#include <cstring>
#endif

using namespace pto;

namespace {

constexpr int ROW_TILE = 16;
constexpr int COL_TILE = 768;
constexpr int SCALE_COLS = COL_TILE / 32;

template <typename T>
AICORE inline __gm__ T* tensorData(__gm__ int64_t* args, uint32_t index)
{
    auto* tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    dcci(tensor, SINGLE_CACHE_LINE);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

AICORE inline void activateQuantize(__ubuf__ float* gate, __ubuf__ float* up,
                                   __ubuf__ float* weights, __ubuf__ uint8_t* out,
                                   __ubuf__ uint8_t* codes, uint16_t validRows, uint16_t validCols)
{
#if defined(__CPU_SIM)
    using Input = Tile<TileType::Vec, float, ROW_TILE, COL_TILE>;
    using Weight = Tile<TileType::Vec, float, 1, ROW_TILE>;
    using Output = Tile<TileType::Vec, uint8_t, ROW_TILE, COL_TILE>;
    using Codes = Tile<TileType::Vec, uint8_t, 1, ROW_TILE * SCALE_COLS>;
    Input gateTile, upTile;
    Weight weightTile;
    Output outputTile;
    Codes codeTile;
    TASSIGN(gateTile, reinterpret_cast<uintptr_t>(gate));
    TASSIGN(upTile, reinterpret_cast<uintptr_t>(up));
    TASSIGN(weightTile, reinterpret_cast<uintptr_t>(weights));
    TASSIGN(outputTile, reinterpret_cast<uintptr_t>(out));
    TASSIGN(codeTile, reinterpret_cast<uintptr_t>(codes));
    gate = gateTile.data();
    up = upTile.data();
    weights = weightTile.data();
    out = outputTile.data();
    codes = codeTile.data();
    for (uint16_t row = 0; row < ROW_TILE; ++row) {
        for (uint16_t group = 0; group < validCols / 32; ++group) {
            float values[32] = {};
            float maximum = 1e-4f;
            for (uint16_t col = 0; col < 32; ++col) {
                if (row < validRows) {
                    const uint16_t offset = row * COL_TILE + group * 32 + col;
                    const float g = std::fmin(static_cast<float>(gate[offset]), 10.0f);
                    const float u = std::fmax(std::fmin(static_cast<float>(up[offset]), 10.0f), -10.0f);
                    values[col] = (g / (1.0f + std::exp(-g))) * u * weights[row];
                }
                maximum = std::fmax(maximum, std::fabs(values[col]));
            }
            maximum *= 1.0f / 448.0f;
            uint32_t bits;
            std::memcpy(&bits, &maximum, sizeof(bits));
            const uint32_t exponent = (bits + 8388607) >> 23;
            codes[(group / 2) * 32 + row * 2 + group % 2] = static_cast<uint8_t>(exponent);
            bits = (254 - exponent) << 23;
            float reciprocal;
            std::memcpy(&reciprocal, &bits, sizeof(reciprocal));
            for (uint16_t col = 0; col < 32; ++col) {
                const float normalized = std::fmax(std::fmin(values[col] * reciprocal, 448.0f), -448.0f);
                const float8_e4m3_t value(normalized, RoundMode::CAST_RINT, SaturationMode::ON);
                out[row * COL_TILE + group * 32 + col] = value.RawData();
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
            for (uint16_t col = 0; col < validCols; col += 128) {
                vector_f32 g0, g1, u0, u1;
                vlds(g0, gate, row * COL_TILE + col, NORM);
                vlds(g1, gate, row * COL_TILE + col + 64, NORM);
                vlds(u0, up, row * COL_TILE + col, NORM);
                vlds(u1, up, row * COL_TILE + col + 64, NORM);
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
                const uint16_t scaleOffset = (col / 64) * 32 + row * 2;
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
                vsts((vector_u8&)payload0, out, row * COL_TILE + col, PK4_B32, all);
                vsts((vector_u8&)payload1, out, row * COL_TILE + col + 64, PK4_B32, all);
            }
        }
        if (validRows < ROW_TILE) {
            vector_f32 zero;
            vector_f8e4m3 zeroPayload;
            vector_s32 zeroExponent;
            vbr(zero, 0.0f);
            vcvt(zeroPayload, zero, all, ROUND_R, RS_ENABLE, PART_P0);
            vbr(zeroExponent, 105 | (105 << 8));
            for (uint16_t row = validRows; row < ROW_TILE; ++row) {
                for (uint16_t col = 0; col < validCols; col += 64) {
                    const uint16_t scaleOffset = (col / 64) * 32 + row * 2;
                    vsts((vector_u16&)zeroExponent, (__ubuf__ uint16_t*)codes, scaleOffset / 2,
                         std::integral_constant<::DistVST, halfStore>(), one16);
                    vsts((vector_u8&)zeroPayload, out, row * COL_TILE + col, PK4_B32, all);
                }
            }
        }
    }
#endif
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args)
{
    auto* gate = tensorData<float>(args, 0);
    auto* up = gate;
    auto* weights = tensorData<float>(args, 1);
    auto* counts = tensorData<int32_t>(args, 2);
    auto* out = tensorData<uint8_t>(args, 3);
    auto* scales = tensorData<uint8_t>(args, 4);
    const int64_t experts = args[5];
    const int64_t rowStride = args[6];
    const int64_t width = args[7];
    const int64_t block = get_block_idx(args);
    const int64_t blocks = get_block_num(args);
    const int64_t colTiles = (width + COL_TILE - 1) / COL_TILE;

    using Input = Tile<TileType::Vec, float, ROW_TILE, COL_TILE, BLayout::RowMajor, ROW_TILE, DYNAMIC>;
    using Weight = Tile<TileType::Vec, float, 1, ROW_TILE>;
    using Output = Tile<TileType::Vec, uint8_t, ROW_TILE, COL_TILE, BLayout::RowMajor, ROW_TILE, DYNAMIC>;
    using Codes = Tile<TileType::Vec, uint8_t, 1, ROW_TILE * SCALE_COLS, BLayout::RowMajor, 1, DYNAMIC>;
    Input gateTile, upTile;
    Weight weightTile;
    Output outputTile;
    Codes codes;
    TASSIGN(gateTile, 0);
    TASSIGN(upTile, ROW_TILE * COL_TILE * 4);
    TASSIGN(weightTile, ROW_TILE * COL_TILE * 8);
    TASSIGN(outputTile, ROW_TILE * COL_TILE * 8 + 64);
    TASSIGN(codes, ROW_TILE * COL_TILE * 9 + 64);

    using GlobalInput = GlobalTensor<float, Shape<1, 1, 1, ROW_TILE, DYNAMIC>,
                                    pto::Stride<-1, -1, -1, -1, 1>>;
    using GlobalWeight = GlobalTensor<float, Shape<1, 1, 1, 1, ROW_TILE>,
                                     pto::Stride<ROW_TILE, ROW_TILE, ROW_TILE, ROW_TILE, 1>>;
    using GlobalOutput = GlobalTensor<uint8_t, Shape<1, 1, 1, ROW_TILE, DYNAMIC>,
                                     pto::Stride<-1, -1, -1, -1, 1>>;
    using GlobalCodes = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, DYNAMIC>,
                                    pto::Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, 1>>;

    int64_t workBase = 0;
    int64_t activationBase = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
        dcci(counts + expert, SINGLE_CACHE_LINE);
        const int64_t count = counts[expert];
        const int64_t workCount = ((count + ROW_TILE - 1) / ROW_TILE) * colTiles;
        const int64_t firstWork = (block + blocks - workBase % blocks) % blocks;
        for (int64_t work = firstWork; work < workCount; work += blocks) {
            const int64_t localRow = (work / colTiles) * ROW_TILE;
            const int64_t row = activationBase + localRow;
            const int64_t col = (work % colTiles) * COL_TILE;
            const uint16_t validCols = width - col < COL_TILE ? width - col : COL_TILE;
            gateTile.SetValidCol(validCols);
            upTile.SetValidCol(validCols);
            outputTile.SetValidCol(validCols);
            codes.SetValidCol(ROW_TILE * (validCols / 32));
            const uint16_t validRows = count - localRow < ROW_TILE ? count - localRow : ROW_TILE;
            const pto::Stride<-1, -1, -1, -1, 1> stride(ROW_TILE * width, ROW_TILE * width,
                                                       ROW_TILE * width, width, 1);
            const pto::Stride<-1, -1, -1, -1, 1> inputStride(ROW_TILE * width * 2, ROW_TILE * width * 2,
                                                            ROW_TILE * width * 2, width * 2, 1);
            GlobalInput globalGate(gate + row * width * 2 + col, GlobalInput::Shape(1, 1, 1, ROW_TILE, validCols), inputStride);
            GlobalInput globalUp(up + row * width * 2 + width + col, GlobalInput::Shape(1, 1, 1, ROW_TILE, validCols), inputStride);
            GlobalWeight globalWeight(weights + expert * rowStride + localRow);
            TLOAD(gateTile, globalGate);
            TLOAD(upTile, globalUp);
            TLOAD(weightTile, globalWeight);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            activateQuantize(gateTile.data(), upTile.data(), weightTile.data(), outputTile.data(),
                             codes.data(), validRows, validCols);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            GlobalOutput globalOut(out + row * width + col, GlobalOutput::Shape(1, 1, 1, ROW_TILE, validCols), stride);
            const uint16_t codeBytes = ROW_TILE * (validCols / 32);
            GlobalCodes globalScales(scales + row * (width / 32) + (col / COL_TILE) * ROW_TILE * SCALE_COLS,
                                     GlobalCodes::Shape(1, 1, 1, 1, codeBytes),
                                     GlobalCodes::Stride(codeBytes, codeBytes, codeBytes, codeBytes, 1));
            TSTORE(globalOut, outputTile);
            TSTORE(globalScales, codes);
            pipe_barrier(PIPE_ALL);
        }
        workBase += workCount;
        activationBase += (count + 15) / 16 * 16;
    }
}
