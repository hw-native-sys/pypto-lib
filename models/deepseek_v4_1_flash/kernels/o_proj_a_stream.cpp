// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// -----------------------------------------------------------------------------------------------------------
// Persistent eight-group O-A projection with split-N FP8 requantization.
#include "mxfp8_projection.hpp"
#include "mx_quant.hpp"

using namespace mxfp8_projection;
constexpr int MT = 128, NT = 256, HalfN = NT / 2;
constexpr int K = 4096, GroupN = 1024, OutN = 8 * GroupN;
constexpr int Ready = 0, Release = 1, VecOffset = 16;
using Raw = Tile<TileType::Vec, float, MT, HalfN>;
using Rounded = Tile<TileType::Vec, bfloat16_t, MT, HalfN>;
constexpr int RoundedUB = MT * HalfN * sizeof(float);
constexpr int OutUB = RoundedUB + MT * HalfN * sizeof(bfloat16_t);
constexpr int ScaleUB = OutUB + MT * HalfN;
#if defined(__CPU_SIM)
using SimPipe = TPipe<0, Direction::DIR_C2V, MT * NT * sizeof(float), 1, 2>;
#endif

#if defined(__DAV_CUBE__)
struct Emit {
#if defined(__CPU_SIM)
    SimPipe pipe;
    __aicore__ Emit() : pipe(nullptr, 0, 0) {}
#endif
    __aicore__ inline void operator()(AC<NT, MT>& acc) {
#if defined(__CPU_SIM)
        TPUSH<SimPipe, AC<NT, MT>, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, acc);
#else
        wait_intra_block(PIPE_FIX, Release);
        wait_intra_block(PIPE_FIX, Release + VecOffset);
        Raw raw;
        TASSIGN(raw, 0);
        TMOV<Raw, AC<NT, MT>, AccToVecMode::DualModeSplitN>(raw, acc);
        set_intra_block(PIPE_FIX, Ready);
        set_intra_block(PIPE_FIX, Ready + VecOffset);
#endif
    }
};
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    const int group = get_block_idx(args) / 4;
    const int n = (get_block_idx(args) % 4) * NT;
    const int active = args[6], padded = args[7];
#if defined(__DAV_CUBE__)
    Emit emit;
    for (int row = 0; row < padded; row += MT) {
        int m = group * padded + row;
        project_emit<K, GroupN, NT, MT>(data<F>(args, 0) + m * K,
            data<F>(args, 1) + group * K * GroupN,
            data<S>(args, 2) + m * (K / 32), data<S>(args, 3) + group * GroupN * (K / 32), n, emit);
    }
#if !defined(__CPU_SIM)
    wait_intra_block(PIPE_FIX, Release);
    wait_intra_block(PIPE_FIX, Release + VecOffset);
#endif
#elif defined(__DAV_VEC__)
    const int col = group * GroupN + n + get_sub_block_id(args) * HalfN;
    using Input = mx_quant::Vec<bfloat16_t, 16, HalfN>;
    using Payload = mx_quant::Vec<uint8_t, MT, HalfN>;
    using Scale = mx_quant::Vec<uint8_t, MT / 16, HalfN / 2>;
    Raw raw; Rounded rounded; Payload payload; Scale scales;
    TASSIGN(raw, 0);
    TASSIGN(rounded, RoundedUB);
    TASSIGN(payload, OutUB);
    TASSIGN(scales, ScaleUB);
#if defined(__CPU_SIM)
    SimPipe pipe(nullptr, 0, 0);
#else
    set_intra_block(PIPE_V, Release);
#endif
    for (int row = 0; row < padded; row += MT) {
#if defined(__CPU_SIM)
        TPOP<SimPipe, Raw, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, raw);
#else
        wait_intra_block(PIPE_V, Ready);
#endif
        TCVT(rounded, raw, RoundMode::CAST_RINT);
#if defined(__CPU_SIM)
        TFREE<SimPipe, TileSplitAxis::TILE_LEFT_RIGHT>(pipe);
#else
        if (row + MT < padded) set_intra_block(PIPE_V, Release);
#endif
        if (row + MT > active) TEXPANDS(payload, uint8_t(0));
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
        GM<uint8_t, MT/16, HalfN/2, MT/16, OutN/2, Layout::ND> exp(
            data<uint8_t>(args, 5) + (row / 16) * (OutN / 2) + col / 2);
        TSTORE(dst, payload);
        TSTORE(exp, scales);
        pipe_barrier(PIPE_ALL);
    }
#if !defined(__CPU_SIM)
    // Release the last slot only after payload and scale stores are complete.
    set_intra_block(PIPE_MTE3, Release);
#endif
#endif
}
