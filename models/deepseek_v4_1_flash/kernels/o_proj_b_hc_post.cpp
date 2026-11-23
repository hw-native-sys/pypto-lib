// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// -----------------------------------------------------------------------------------------------------------
// Runtime-M O-B with N160 tiles and SplitM HC-post.
#include "mxfp8_projection.hpp"
#if defined(__CPU_SIM)
#include <cmath>
#endif

using namespace mxfp8_projection;
namespace {
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
template <int MT, int NT>
using Raw = Vec<float, MT / 2, NT>;
constexpr int ROW_TILE = 256;
constexpr int PRELOAD_ROWS = 64;
// The four residual streams also hold the mixed output.
template <int NT>
using Residual = Tile<TileType::Vec, float, 4 * PRELOAD_ROWS, NT, BLayout::RowMajor, DYNAMIC, NT>;
using PostMix = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, DYNAMIC, 4>;
using ResMix = Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, DYNAMIC, 16>;
constexpr int RAW_UB = 0;
template <int NT>
constexpr int ResidualUB = (ROW_TILE / 2) * NT * sizeof(float);
constexpr int POST_UB = 245760;
constexpr int RES_MIX_UB = 247808;
constexpr int READY = 0, RELEASE = 1, FINISH = 2, STORE_DONE = 3, VEC_OFFSET = 16;
static_assert(RES_MIX_UB + 64 * 16 * 4 <= 262144, "UB capacity exceeded");
#if defined(__CPU_SIM)
using SimPipe = TPipe<0, Direction::DIR_C2V, ROW_TILE * 160 * sizeof(float), 1, 2>;
#endif

#if defined(__DAV_CUBE__)
#if !defined(__CPU_SIM)
template <int MT, int NT>
__tf__ __aicore__ inline void move_valid_rows(
    typename Raw<MT, NT>::TileDType __out__ dst,
    typename AC<NT, MT>::TileDType __in__ src, uint16_t rows) {
    auto dst_addr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    auto src_addr = (__cc__ float*)__cce_get_tile_ptr(src);
    constexpr auto mode = GetDualDstCtl<Raw<MT, NT>, AC<NT, MT>, AccToVecMode::DualModeSplitM,
                                       QuantMode_t::NoQuant>();
    SetLoop3Para();
    // MT is the physical L0C stride; rows is the even-rounded transfer extent.
    pto_copy_matrix_cc_to_ub(
        dst_addr, src_addr, 0, NT, rows, NT, MT, mode, false, 0, 0,
        QuantMode_t::NoQuant, static_cast<uint8_t>(ReluPreMode::NoRelu), false, true,
        0, 0, false, false, 0, false, false, false, false, false, false);
}
#endif
template <int MT>
struct PostOutput {
#if defined(__CPU_SIM)
    SimPipe pipe;
    __aicore__ PostOutput() : pipe(nullptr, 0, 0) {}
#endif
    int rows;
    template <int NT>
    __aicore__ inline void operator()(AC<NT, MT>& acc) {
#if defined(__CPU_SIM)
        acc.SetValidRow((rows + 1) & ~1);
        TPUSH<SimPipe, AC<NT, MT>, TileSplitAxis::TILE_UP_DOWN_ODD>(pipe, acc);
#else
        wait_intra_block(PIPE_FIX, RELEASE);
        wait_intra_block(PIPE_FIX, RELEASE + VEC_OFFSET);
        Raw<MT, NT> raw;
        TASSIGN(raw, RAW_UB);
        move_valid_rows<MT, NT>(raw.data(), acc.data(), (rows + 1) & ~1);
        set_intra_block(PIPE_FIX, READY);
        set_intra_block(PIPE_FIX, READY + VEC_OFFSET);
#endif
    }
    __aicore__ inline void finish() {
#if !defined(__CPU_SIM)
        wait_intra_block(PIPE_FIX, RELEASE);
        wait_intra_block(PIPE_FIX, RELEASE + VEC_OFFSET);
        wait_intra_block(PIPE_FIX, STORE_DONE);
        wait_intra_block(PIPE_FIX, STORE_DONE + VEC_OFFSET);
        set_intra_block(PIPE_FIX, FINISH);
        set_intra_block(PIPE_FIX, FINISH + VEC_OFFSET);
#endif
    }
};
#endif

#if defined(__DAV_VEC__)
template <int MT, int NT>
__tf__ __aicore__ inline void mix_rows(
    typename Residual<NT>::TileDType __out__ residual_tile, typename Raw<MT, NT>::TileDType __in__ raw_tile,
    PostMix::TileDType __in__ post_tile, ResMix::TileDType __in__ residual_mix_tile,
    uint16_t rows, int raw_offset) {
    auto residual_base = (__ubuf__ float*)__cce_get_tile_ptr(residual_tile);
    auto raw_base = (__ubuf__ float*)__cce_get_tile_ptr(raw_tile) + raw_offset * NT;
    auto post_base = (__ubuf__ float*)__cce_get_tile_ptr(post_tile);
    auto mix_base = (__ubuf__ float*)__cce_get_tile_ptr(residual_mix_tile);
#if defined(__CPU_SIM)
    for (uint16_t row = 0; row < rows; ++row) {
        auto residual_output = residual_base + row * 4 * NT;
        auto raw = raw_base + row * NT;
        auto post = post_base + row * 8;
        auto residual_mix = mix_base + row * 16;
        for (int k = 0; k < NT; ++k) {
            const bfloat16_t rounded = static_cast<bfloat16_t>(raw[k]);
            const float x = static_cast<float>(rounded);
            const float r0 = residual_output[k];
            const float r1 = residual_output[NT + k];
            const float r2 = residual_output[2 * NT + k];
            const float r3 = residual_output[3 * NT + k];
            for (int h = 0; h < 4; ++h) {
                float value = x * post[h];
                value = std::fma(r0, residual_mix[h], value);
                value = std::fma(r1, residual_mix[4 + h], value);
                value = std::fma(r2, residual_mix[8 + h], value);
                value = std::fma(r3, residual_mix[12 + h], value);
                residual_output[h * NT + k] = value;
            }
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL), full16 = pset_b16(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            auto residual_output = residual_base + row * 4 * NT;
            auto raw = raw_base + row * NT;
            auto post = post_base + row * 8;
            auto residual_mix = mix_base + row * 16;
            for (uint16_t k = 0; k < (NT + 63) / 64; ++k) {
                RegTensor<bfloat16_t> rounded;
                RegTensor<float> value, x0, r0, r1, r2, r3, coef, term;
                vlds(x0, raw, k * 64, NORM);
                vcvt(rounded, x0, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vcvt(x0, rounded, full16, PART_EVEN);
                vlds(r0, residual_output, k * 64, NORM);
                vlds(r1, residual_output, NT + k * 64, NORM);
                vlds(r2, residual_output, 2 * NT + k * 64, NORM);
                vlds(r3, residual_output, 3 * NT + k * 64, NORM);
                for (uint16_t h = 0; h < 4; ++h) {
                    vlds(coef, post, h, BRC_B32);
                    vmul(value, x0, coef, full, MODE_ZEROING);
                    vlds(coef, residual_mix, h, BRC_B32);
                    term = r0;
                    vmadd(term, coef, value, full, MODE_ZEROING);
                    value = term;
                    vlds(coef, residual_mix, 4 + h, BRC_B32);
                    term = r1;
                    vmadd(term, coef, value, full, MODE_ZEROING);
                    value = term;
                    vlds(coef, residual_mix, 8 + h, BRC_B32);
                    term = r2;
                    vmadd(term, coef, value, full, MODE_ZEROING);
                    value = term;
                    vlds(coef, residual_mix, 12 + h, BRC_B32);
                    term = r3;
                    vmadd(term, coef, value, full, MODE_ZEROING);
                    value = term;
                    if constexpr (NT % 64 == 0) {
                        vsts(value, residual_output, h * NT + k * 64, NORM_B32, full);
                    } else {
                        uint32_t remaining = NT - k * 64;
                        MaskReg tail = CreatePredicate<float>(remaining);
                        vsts(value, residual_output, h * NT + k * 64, NORM_B32, tail);
                    }
                }
            }
        }
    }
#endif
}

template <int Columns, int Stride>
__aicore__ inline auto row_panel(__gm__ float* ptr, int rows) {
    using Shape = TileShape2D<float, DYNAMIC, Columns, Layout::ND>;
    using Strides = BaseShape2D<float, DYNAMIC, Stride, Layout::ND>;
    return GlobalTensor<float, Shape, Strides>(ptr, Shape(rows, Columns), Strides(rows, Stride));
}

template <int MT, int NT, int NC>
__aicore__ inline void vector_task(__gm__ int64_t* args, int task, PostMix& post, ResMix& res_mix) {
    static_assert(ResidualUB<NT> + 4 * PRELOAD_ROWS * NT * sizeof(float) <= POST_UB);
    const int active = args[8];
    constexpr int COLUMNS = (5120 + NC - 1) / NC;
    const int row_group = task / COLUMNS * MT;
    const int group = active - row_group > MT ? MT : active - row_group;
    const int half = (group + 1) / 2;
    const int row = row_group + get_sub_block_id(args) * half;
    const int n = task % COLUMNS * NC;
    const int valid = active - row < 0 ? 0 : (active - row > half ? half : active - row);
    Raw<MT, NT> raw;
    TASSIGN(raw, RAW_UB);
#if defined(__CPU_SIM)
    SimPipe pipe(nullptr, 0, 0);
#endif
    // Both lanes acknowledge the matrix transfer, including a lane with zero valid rows.
    int offset = 0;
    do {
        const int rows = valid - offset < PRELOAD_ROWS ? valid - offset : PRELOAD_ROWS;
        Residual<NT> residual(rows * 4);
        TASSIGN(residual, ResidualUB<NT>);
        post.SetValidRow(rows);
        res_mix.SetValidRow(rows);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if (rows > 0) {
            auto r = row_panel<NT, 5120>(data<float>(args, 4) + (row + offset) * 20480 + n, rows * 4);
            auto p = row_panel<4, 4>(data<float>(args, 5) + (row + offset) * 4, rows);
            auto m = row_panel<16, 16>(data<float>(args, 6) + (row + offset) * 16, rows);
            TLOAD(residual, r);
            TLOAD(post, p);
            TLOAD(res_mix, m);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (offset == 0) {
#if defined(__CPU_SIM)
            TPOP<SimPipe, Raw<MT, NT>, TileSplitAxis::TILE_UP_DOWN_ODD>(pipe, raw);
#else
            wait_intra_block(PIPE_V, READY);
#endif
        }
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        mix_rows<MT, NT>(residual.data(), raw.data(), post.data(), res_mix.data(), rows, offset);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (rows > 0) {
            auto y = row_panel<NT, 5120>(data<float>(args, 7) + (row + offset) * 20480 + n, rows * 4);
            TSTORE(y, residual);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        offset += PRELOAD_ROWS;
    } while (offset < valid);
#if defined(__CPU_SIM)
    TFREE<SimPipe, TileSplitAxis::TILE_UP_DOWN_ODD>(pipe);
#else
    set_intra_block(PIPE_V, RELEASE);
#endif
}

template <int MT, int NC>
__aicore__ inline void vector_entry(__gm__ int64_t* args) {
    const int active = args[8];
    const int workers = get_block_num(args);
    constexpr int COLUMNS = (5120 + NC - 1) / NC;
    constexpr int FULL_COLUMNS = 5120 / NC;
    const int tasks = ((active + MT - 1) / MT) * COLUMNS;
    PostMix post(PRELOAD_ROWS);
    ResMix res_mix(PRELOAD_ROWS);
    TASSIGN(post, POST_UB);
    TASSIGN(res_mix, RES_MIX_UB);
#if !defined(__CPU_SIM)
    set_intra_block(PIPE_V, RELEASE);
#endif
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (int task = get_block_idx(args); task < tasks; task += workers) {
        if (task % COLUMNS < FULL_COLUMNS) {
            vector_task<MT, NC, NC>(args, task, post, res_mix);
        } else {
            vector_task<MT, 128, NC>(args, task, post, res_mix);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#if !defined(__CPU_SIM)
    set_intra_block(PIPE_MTE3, STORE_DONE);
    wait_intra_block(PIPE_V, FINISH);
#endif
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_CUBE__)
template <int MT, int NT>
__aicore__ inline void project_group(__gm__ int64_t* args, int row, int n, int rows) {
    constexpr int K = 8192;
    PostOutput<MT> emit;
    emit.rows = rows;
    project_emit<K, 5120, NT, MT>(
        data<F>(args, 0) + row * K, data<F>(args, 1),
        data<S>(args, 2) + row * (K / 32), data<S>(args, 3), n, emit);
}

template <int NT>
__aicore__ inline void project_group(__gm__ int64_t* args, int row, int n, int rows) {
    if (rows <= 16) {
        project_group<16, NT>(args, row, n, rows);
    } else if (rows <= 32) {
        project_group<32, NT>(args, row, n, rows);
    } else if (rows <= 48) {
        project_group<48, NT>(args, row, n, rows);
    } else if (rows <= 64) {
        project_group<64, NT>(args, row, n, rows);
    } else if (rows <= 80) {
        project_group<80, NT>(args, row, n, rows);
    } else if (rows <= 96) {
        project_group<96, NT>(args, row, n, rows);
    } else if (rows <= 128) {
        project_group<128, NT>(args, row, n, rows);
    } else {
        project_group<ROW_TILE, NT>(args, row, n, rows);
    }
}
#endif

template <int MT, int NC>
__aicore__ inline void kernel_entry_m(__gm__ int64_t* args) {
    const int active = args[8];
    const int workers = get_block_num(args);
    constexpr int COLUMNS = (5120 + NC - 1) / NC;
    constexpr int FULL_COLUMNS = 5120 / NC;
    const int tasks = ((active + MT - 1) / MT) * COLUMNS;
#if defined(__DAV_CUBE__)
    PostOutput<MT> emit;
    for (int task = get_block_idx(args); task < tasks; task += workers) {
        const int row = task / COLUMNS * MT;
        const int column = task % COLUMNS;
        const int n = column * NC;
        const int rows = active - row > MT ? MT : active - row;
        if (column < FULL_COLUMNS) {
            project_group<NC>(args, row, n, rows);
        } else {
            project_group<128>(args, row, n, rows);
        }
    }
    emit.finish();
#elif defined(__DAV_VEC__)
    vector_entry<MT, NC>(args);
#endif
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    kernel_entry_m<ROW_TILE, 160>(args);
}
