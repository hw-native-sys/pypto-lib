// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Persistent eight-group O-A projection with split-N FP8 requantization.
#include "mxfp8_projection.hpp"
#include "mx_quant.hpp"

using namespace mxfp8_projection;

namespace {
constexpr int MaxM = 256, NT = 256, HalfN = NT / 2;
constexpr int K = 4096, GroupN = 1024, OutN = 8 * GroupN;
constexpr int Ready = 0, Release = 1, VecOffset = 16;
template <int MT>
using Raw = Tile<TileType::Vec, float, MT, HalfN>;
template <int MT>
using Rounded = Tile<TileType::Vec, bfloat16_t, MT, HalfN>;
constexpr int RoundedUB = MaxM * HalfN * sizeof(float);
constexpr int OutUB = RoundedUB + MaxM * HalfN * sizeof(bfloat16_t);
constexpr int ScaleUB = OutUB + MaxM * HalfN;
static_assert(ScaleUB + (MaxM / 16) * (HalfN / 2) <= 245760);
#if defined(__CPU_SIM)
using SimPipe = TPipe<0, Direction::DIR_C2V, MaxM * NT * sizeof(float), 1, 2>;
#endif

#if defined(__DAV_CUBE__)
template <int MT>
struct Emit {
#if defined(__CPU_SIM)
    SimPipe& pipe;
#endif
    __aicore__ inline void operator()(AC<NT, MT>& acc) {
#if defined(__CPU_SIM)
        TPUSH<SimPipe, AC<NT, MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, acc);
#else
        wait_intra_block(PIPE_FIX, Release);
        wait_intra_block(PIPE_FIX, Release + VecOffset);
        Raw<MT> raw;
        TASSIGN(raw, 0);
        TMOV<Raw<MT>, AC<NT, MT>, AccToVecMode::DualModeSplitN>(raw, acc);
        set_intra_block(PIPE_FIX, Ready);
        set_intra_block(PIPE_FIX, Ready + VecOffset);
#endif
    }
};
#endif

struct Channel {
#if defined(__CPU_SIM)
    SimPipe pipe;
    __aicore__ Channel() : pipe(nullptr, 0, 0) {}
#endif
};

template <int MT>
__aicore__ inline void project_tile(__gm__ int64_t* args, Channel& channel, int row, int group, int n) {
    const int padded = args[7];
#if defined(__DAV_CUBE__)
#if defined(__CPU_SIM)
    Emit<MT> emit{channel.pipe};
#else
    Emit<MT> emit;
#endif
    const int m = group * padded + row;
    project_emit<K, GroupN, NT, MT>(
        data<F>(args, 0) + m * K, data<F>(args, 1) + group * K * GroupN,
        data<S>(args, 2) + m * (K / 32), data<S>(args, 3) + group * GroupN * (K / 32),
        n, emit);
#elif defined(__DAV_VEC__)
#if defined(__CPU_SIM)
    auto& pipe = channel.pipe;
#endif
    const int active = args[6];
    const int col = group * GroupN + n + get_sub_block_id(args) * HalfN;
    using Input = mx_quant::Vec<bfloat16_t, 16, HalfN>;
    using Payload = mx_quant::Vec<uint8_t, MT, HalfN>;
    using Scale = mx_quant::Vec<uint8_t, MT / 16, HalfN / 2>;
    Raw<MT> raw;
    Rounded<MT> rounded;
    Payload payload;
    Scale scales;
    TASSIGN(raw, 0);
    TASSIGN(rounded, RoundedUB);
    TASSIGN(payload, OutUB);
    TASSIGN(scales, ScaleUB);
#if defined(__CPU_SIM)
    TPOP<SimPipe, Raw<MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, raw);
#else
    wait_intra_block(PIPE_V, Ready);
#endif
    TCVT(rounded, raw, RoundMode::CAST_RINT);
#if defined(__CPU_SIM)
    TFREE<SimPipe, TileSplitAxis::TILE_LEFT_RIGHT>(pipe);
#else
    if (row + MT < padded) {
        set_intra_block(PIPE_V, Release);
    }
#endif
    if (row + MT > active) {
        TEXPANDS(payload, uint8_t(0));
    }
    for (int block = 0; block < MT / 16; ++block) {
        Input chunk;
        mx_quant::Vec<uint8_t, 16, HalfN> out;
        mx_quant::Vec<uint8_t, 1, HalfN / 2> codes;
        TASSIGN(chunk, RoundedUB + block * 16 * HalfN * sizeof(bfloat16_t));
        TASSIGN(out, OutUB + block * 16 * HalfN);
        TASSIGN(codes, ScaleUB + block * (HalfN / 2));
        int valid = active - row - block * 16;
        valid = valid < 0 ? 0 : (valid > 16 ? 16 : valid);
        mx_quant::quantize<HalfN, Input, true>(out.data(), codes.data(), chunk.data(), valid);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GM<uint8_t, MT, HalfN, MT, OutN, Layout::ND> dst(data<uint8_t>(args, 4) + row * OutN + col);
    GM<uint8_t, MT / 16, HalfN / 2, MT / 16, OutN / 2, Layout::ND> exp(
        data<uint8_t>(args, 5) + (row / 16) * (OutN / 2) + col / 2);
    TSTORE(dst, payload);
    TSTORE(exp, scales);
    pipe_barrier(PIPE_ALL);
#endif
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    const int workers = get_block_num(args);
    const int padded = args[7];
    Channel channel;
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
    set_intra_block(PIPE_V, Release);
#endif
    for (int panel = get_block_idx(args); panel < OutN / NT; panel += workers) {
        const int group = panel / (GroupN / NT);
        const int n = panel % (GroupN / NT) * NT;
        int row = 0;
        for (; row + MaxM <= padded; row += MaxM) {
            project_tile<MaxM>(args, channel, row, group, n);
        }
        switch (padded - row) {
            case 32: project_tile<32>(args, channel, row, group, n); break;
            case 64: project_tile<64>(args, channel, row, group, n); break;
            case 96: project_tile<96>(args, channel, row, group, n); break;
            case 128: project_tile<128>(args, channel, row, group, n); break;
            case 160: project_tile<160>(args, channel, row, group, n); break;
            case 192: project_tile<192>(args, channel, row, group, n); break;
            case 224: project_tile<224>(args, channel, row, group, n); break;
            default: break;
        }
#if defined(__DAV_VEC__) && !defined(__CPU_SIM)
        // Return the last row tile's credit after both panel outputs have been stored.
        set_intra_block(PIPE_MTE3, Release);
#endif
    }
#if defined(__DAV_CUBE__) && !defined(__CPU_SIM)
    wait_intra_block(PIPE_FIX, Release);
    wait_intra_block(PIPE_FIX, Release + VecOffset);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
#endif
}
