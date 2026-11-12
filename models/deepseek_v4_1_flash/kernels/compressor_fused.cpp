/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Adapted from ops-transformer CompressorV2BlockCube (arch35): paired KV/gate
// weights share each X panel, with buffered L1 operands and 128-wide L0 K steps.
#if defined(__CPU_SIM)
#include <algorithm>
#include <cmath>
#endif
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;
namespace {
constexpr int M_TILE = 256;
constexpr int N_TILE = 64;
constexpr int K_TILE = 256;
constexpr int M_L0_TILE = 128;
constexpr int K_L0_TILE = 128;
constexpr int H = 5120;
constexpr int D = 512;
using AL = TileLeftCompact<bfloat16_t, M_L0_TILE, K_L0_TILE, DYNAMIC, K_L0_TILE>;
using BR = TileRight<bfloat16_t, K_L0_TILE, 2 * N_TILE>;
using AC = TileAccCompact<float, M_L0_TILE, 2 * N_TILE, DYNAMIC, 2 * N_TILE>;
using ACView = TileAccCompact<float, M_L0_TILE, N_TILE, DYNAMIC, N_TILE>;
static_assert(K_L0_TILE * 2 * N_TILE * sizeof(bfloat16_t) <= 32768);
static_assert(M_L0_TILE * 2 * N_TILE * sizeof(float) <= 65536);

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_CUBE__)
template <int PanelRows, int PanelK>
__aicore__ inline void load_panel(__gm__ int64_t* args, int row, int rows, int col, int step, int slot) {
    constexpr int A_BYTES = PanelRows * PanelK * sizeof(bfloat16_t);
    constexpr int B_BYTES = PanelK * N_TILE * sizeof(bfloat16_t);
    constexpr int SLOT_BYTES = A_BYTES + 2 * B_BYTES;
    using AM = Tile<TileType::Mat, bfloat16_t, PanelRows, PanelK, BLayout::ColMajor, DYNAMIC, PanelK,
                    SLayout::RowMajor>;
    using BM = Tile<TileType::Mat, bfloat16_t, PanelK, N_TILE, BLayout::ColMajor, PanelK, N_TILE,
                    SLayout::RowMajor>;
    int k = step * PanelK;
    AM a(rows);
    TASSIGN(a, slot * SLOT_BYTES);
    using AS = TileShape2D<bfloat16_t, DYNAMIC, PanelK>;
    using AB = BaseShape2D<bfloat16_t, DYNAMIC, H>;
    GlobalTensor<bfloat16_t, AS, AB> ag(data<bfloat16_t>(args, 0) + (int64_t)row * H + k,
                                      AS(rows, PanelK), AB(rows, H));
    TLOAD(a, ag);
    for (int w = 0; w < 2; ++w) {
        BM b;
        TASSIGN(b, slot * SLOT_BYTES + A_BYTES + w * B_BYTES);
        using BS = TileShape2D<bfloat16_t, PanelK, N_TILE, Layout::NZ>;
        using BB = BaseShape2D<bfloat16_t, H, D, Layout::NZ>;
        GlobalTensor<bfloat16_t, BS, BB, Layout::NZ> bg(
            data<bfloat16_t>(args, 1 + w) + (int64_t)col * H + k * 16);
        TLOAD(b, bg);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

template <int PanelRows, int PanelK>
__aicore__ inline void project(__gm__ int64_t* args, int row, int rows, int col, int scratch_row) {
    constexpr int A_BYTES = PanelRows * PanelK * sizeof(bfloat16_t);
    constexpr int B_BYTES = PanelK * N_TILE * sizeof(bfloat16_t);
    constexpr int SLOT_BYTES = A_BYTES + 2 * B_BYTES;
    static_assert(2 * SLOT_BYTES <= 512 * 1024);
    using AM = Tile<TileType::Mat, bfloat16_t, PanelRows, PanelK, BLayout::ColMajor, DYNAMIC, PanelK,
                    SLayout::RowMajor>;
    using BM = Tile<TileType::Mat, bfloat16_t, PanelK, 2 * N_TILE, BLayout::ColMajor, PanelK, 2 * N_TILE,
                    SLayout::RowMajor>;
    int halves = (rows + M_L0_TILE - 1) / M_L0_TILE;
    for (int slot = 0; slot < 2; ++slot) {
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
        load_panel<PanelRows, PanelK>(args, row, rows, col, slot, slot);
    }
    int serial = 0;
    for (int step = 0; step < H / PanelK; ++step) {
        int slot = step % 2;
        AM a(rows);
        TASSIGN(a, slot * SLOT_BYTES);
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
        // The two column-major weight panels form one K x (KV | gate) tile.
        BM b;
        TASSIGN(b, slot * SLOT_BYTES + A_BYTES);
        for (int m = 0; m < halves; ++m) {
            int count = rows - m * M_L0_TILE < M_L0_TILE ? rows - m * M_L0_TILE : M_L0_TILE;
            AC acc(count);
            TASSIGN(acc, m * M_L0_TILE * 2 * N_TILE * sizeof(float));
            for (int inner = 0; inner < PanelK / K_L0_TILE; ++inner) {
                auto event = static_cast<event_t>(serial++ % 2);
                AL left(count);
                BR right;
                TASSIGN(left, (serial - 1) % 2 * 32768);
                TASSIGN(right, (serial - 1) % 2 * 32768);
                wait_flag(PIPE_M, PIPE_MTE1, event);
                TEXTRACT(left, a, m * M_L0_TILE, inner * K_L0_TILE);
                TEXTRACT(right, b, inner * K_L0_TILE, 0);
                set_flag(PIPE_MTE1, PIPE_M, event);
                wait_flag(PIPE_MTE1, PIPE_M, event);
                if (step == 0 && inner == 0) TMATMUL(acc, left, right);
                else TMATMUL_ACC(acc, acc, left, right);
                set_flag(PIPE_M, PIPE_MTE1, event);
            }
        }
        if (step + 2 < H / PanelK) {
            set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            load_panel<PanelRows, PanelK>(args, row, rows, col, step + 2, slot);
        }
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    // Complete L1 extracts before the next projection reloads its panels.
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    for (int w = 0; w < 2; ++w) {
        for (int m = 0; m < halves; ++m) {
            int count = rows - m * M_L0_TILE < M_L0_TILE ? rows - m * M_L0_TILE : M_L0_TILE;
            // CPU GetElement uses the static Rows pitch; hardware compact MAD
            // and FIX use ceil16(count). Both views retain the same valid rows.
#if defined(__CPU_SIM)
            int pitch = M_L0_TILE;
#else
            int pitch = (count + 15) / 16 * 16;
#endif
            ACView acc(count);
            TASSIGN(acc, (m * M_L0_TILE * 2 * N_TILE + w * pitch * N_TILE) * sizeof(float));
            using OS = TileShape2D<float, DYNAMIC, N_TILE>;
            using OB = BaseShape2D<float, DYNAMIC, N_TILE>;
            GlobalTensor<float, OS, OB> output(data<float>(args, 11 + w) + (int64_t)(scratch_row + m * M_L0_TILE) * N_TILE,
                                              OS(count, N_TILE), OB(count, N_TILE));
            TSTORE(output, acc);
        }
    }
    // Complete FIX reads before the next projection reuses L0C.
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}
#endif
} // namespace


namespace {
constexpr int SCRATCH_ROWS = M_TILE + 8;
constexpr int READY = 0, RELEASE = 1, FINISH = 2, VEC_OFFSET = 16;

// CPU mode 2 covers both AIV lanes; device events address them separately.
template <pipe_t Pipe, int Event>
__aicore__ inline void signal_pair() {
#if defined(__CPU_SIM)
    ffts_cross_core_sync(Pipe, getFFTSMsg(2, Event));
#else
    set_intra_block(Pipe, Event);
#if defined(__DAV_CUBE__)
    set_intra_block(Pipe, Event + VEC_OFFSET);
#endif
#endif
}

template <pipe_t Pipe, int Event>
__aicore__ inline void wait_pair() {
#if defined(__CPU_SIM)
    wait_flag_dev(Pipe, Event);
#else
    wait_intra_block(Pipe, Event);
#if defined(__DAV_CUBE__)
    wait_intra_block(Pipe, Event + VEC_OFFSET);
#endif
#endif
}

template <typename T>
__aicore__ inline T metadata(__gm__ T* ptr) {
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    return *ptr;
}

__aicore__ inline int request_count(__gm__ int64_t* args) {
    return reinterpret_cast<__gm__ Tensor*>(args[6])->shapes[0];
}

__aicore__ inline int block_count(__gm__ int64_t* args) {
    return reinterpret_cast<__gm__ Tensor*>(args[7])->shapes[0];
}

__aicore__ inline int ring_capacity(__gm__ int64_t* args) {
    return reinterpret_cast<__gm__ Tensor*>(args[7])->shapes[1];
}

__aicore__ inline bool first_odd(__gm__ int64_t* args, int request, int start, int end) {
    if (start < 0 || start >= end || start >= args[13]) return false;
    int position = metadata(data<int32_t>(args, 4) + start);
    return position >= 0 && position % 2 && metadata(data<int32_t>(args, 5) + start) == request;
}

__aicore__ inline int owner_tile(__gm__ int64_t* args, int block) {
    if (args[13] <= M_TILE) return 0;
    int tail = -1;
    for (int request = 0; request < request_count(args); ++request) {
        if (metadata(data<int32_t>(args, 6) + request) != block) continue;
        int start = metadata(data<int32_t>(args, 3) + request);
        int end = metadata(data<int32_t>(args, 3) + request + 1);
        if (start < 0) start = 0;
        if (end > args[13]) end = args[13];
        if (start < end && end - 1 > tail) tail = end - 1;
    }
    return tail < 0 ? -1 : tail / M_TILE;
}

__aicore__ inline bool boundary_needed(__gm__ int64_t* args, int row) {
    if (row == 0) return false;
    int request = metadata(data<int32_t>(args, 5) + row);
    int position = metadata(data<int32_t>(args, 4) + row);
    if (request < 0 || request >= request_count(args) || position < 0 || position % 2 == 0) return false;
    int start = metadata(data<int32_t>(args, 3) + request);
    int end = metadata(data<int32_t>(args, 3) + request + 1);
    int block = metadata(data<int32_t>(args, 6) + request);
    return row > start && row < end && block >= 0 && block < block_count(args);
}

__aicore__ inline void begin_projection(__gm__ int64_t* args, int row, int rows, int col, int scratch_row) {
#if defined(__DAV_CUBE__)
    if (rows <= M_L0_TILE) project<M_L0_TILE, 2 * K_TILE>(args, row, rows, col, scratch_row);
    else project<M_TILE, K_TILE>(args, row, rows, col, scratch_row);
    signal_pair<PIPE_FIX, READY>();
#elif defined(__DAV_VEC__)
    wait_pair<PIPE_MTE2, READY>();
#endif
}

__aicore__ inline void end_projection() {
#if defined(__DAV_CUBE__)
    wait_pair<PIPE_FIX, RELEASE>();
#elif defined(__DAV_VEC__)
    pipe_barrier(PIPE_ALL);
    signal_pair<PIPE_MTE3, RELEASE>();
#endif
}

#if defined(__DAV_VEC__)
template <typename T, int R, int C>
using VT = Tile<TileType::Vec, T, R, C>;
template <typename T, int C>
using Vector = VT<T, 1, C>;
template <typename T, int C>
using Global = GlobalTensor<T, TileShape2D<T, 1, C>, BaseShape2D<T, 1, C>>;

struct RequestMetadata {
    int first = -1;

    __aicore__ inline void load(__gm__ int64_t* args, int request) {
        int base = request / 32 * 32;
        if (base == first) return;
        int count = request_count(args) - base;
        if (count > 32) count = 32;
        using Bounds = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
        using Blocks = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, DYNAMIC>;
        using BoundsGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 64>>;
        using BlocksGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 32>>;
        Bounds starts(count + 1);
        Blocks blocks(count);
        TASSIGN(starts, 108032);
        TASSIGN(blocks, 108288);
        BoundsGM sg(data<int32_t>(args, 3) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count + 1));
        BlocksGM bg(data<int32_t>(args, 6) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
        TLOAD(starts, sg);
        TLOAD(blocks, bg);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID3);
        first = base;
    }

    __aicore__ inline int start(__gm__ int64_t* args, int request, int offset = 0) {
        load(args, request);
        using Bounds = Tile<TileType::Vec, int32_t, 1, 64>;
        Bounds starts;
        TASSIGN(starts, 108032);
        return starts.GetValue(request - first + offset);
    }

    __aicore__ inline int block(__gm__ int64_t* args, int request) {
        load(args, request);
        using Blocks = Tile<TileType::Vec, int32_t, 1, 32>;
        Blocks blocks;
        TASSIGN(blocks, 108288);
        return blocks.GetValue(request - first);
    }
};

constexpr int STATE_KV = 40960;
constexpr int STATE_SCORE = STATE_KV + 128 * N_TILE * sizeof(float);
constexpr int STATE_POSITIONS = STATE_SCORE + 128 * N_TILE * sizeof(float);
constexpr int STATE_REQUESTS = STATE_POSITIONS + 128 * sizeof(int32_t);
static_assert(STATE_REQUESTS + 128 * sizeof(int32_t) <= 108032);
static_assert(108288 + 32 * sizeof(int32_t) <= 256 * 1024);

struct MainState {
    __aicore__ inline void load(__gm__ int64_t* args, int row, int rows, int scratch) {
        using State = Tile<TileType::Vec, float, 128, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
        using StateGM = GlobalTensor<float, TileShape2D<float, DYNAMIC, N_TILE>,
                                    BaseShape2D<float, DYNAMIC, N_TILE>>;
        using Metadata = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, DYNAMIC>;
        using MetadataGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 128>>;
        State kv(rows), score(rows);
        Metadata positions(rows), requests(rows);
        TASSIGN(kv, STATE_KV); TASSIGN(score, STATE_SCORE);
        TASSIGN(positions, STATE_POSITIONS); TASSIGN(requests, STATE_REQUESTS);
        StateGM kg(data<float>(args, 11) + (int64_t)scratch * N_TILE,
                   TileShape2D<float, DYNAMIC, N_TILE>(rows, N_TILE));
        StateGM sg(data<float>(args, 12) + (int64_t)scratch * N_TILE,
                   TileShape2D<float, DYNAMIC, N_TILE>(rows, N_TILE));
        MetadataGM pg(data<int32_t>(args, 4) + row, TileShape2D<int32_t, 1, DYNAMIC>(1, rows));
        MetadataGM rg(data<int32_t>(args, 5) + row, TileShape2D<int32_t, 1, DYNAMIC>(1, rows));
        TLOAD(kv, kg); TLOAD(score, sg);
        TLOAD(positions, pg); TLOAD(requests, rg);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID2); wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    }

    __aicore__ inline int position(int row) {
        Vector<int32_t, 128> positions;
        TASSIGN(positions, STATE_POSITIONS);
        return positions.GetValue(row);
    }

    __aicore__ inline int request(int row) {
        Vector<int32_t, 128> requests;
        TASSIGN(requests, STATE_REQUESTS);
        return requests.GetValue(row);
    }
};

template <int TileRows>
__tf__ __aicore__ inline void pool_rows(
    typename VT<bfloat16_t, TileRows, N_TILE>::TileDType __out__ output,
    typename VT<float, TileRows, N_TILE>::TileDType __in__ current_kv,
    typename VT<float, TileRows, N_TILE>::TileDType __in__ current_score,
    typename VT<float, TileRows, N_TILE>::TileDType __in__ previous_kv,
    typename VT<float, TileRows, N_TILE>::TileDType __in__ previous_score,
    uint64_t mask, uint16_t rows) {
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto ck = (__ubuf__ float*)__cce_get_tile_ptr(current_kv);
    auto cs = (__ubuf__ float*)__cce_get_tile_ptr(current_score);
    auto pk = (__ubuf__ float*)__cce_get_tile_ptr(previous_kv);
    auto ps = (__ubuf__ float*)__cce_get_tile_ptr(previous_score);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < N_TILE; ++col) {
            int i = row * N_TILE + col;
            float value = 0;
            if (mask & (1ULL << row)) {
                float highest = std::max(cs[i], ps[i]);
                // Preserve the original non-finite-score behavior while sharing one exponential.
                float one = 1.0f + (highest - highest);
                float e = std::exp(-std::abs(ps[i] - cs[i]));
                float a = ps[i] >= cs[i] ? one : e;
                float b = ps[i] >= cs[i] ? e : one;
                float inverse = 1.0f / (a + b);
                value = pk[i] * (a * inverse) + ck[i] * (b * inverse);
            }
            dst[i] = static_cast<bfloat16_t>(value);
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> value;
            if (mask & (1ULL << row)) {
                int offset = row * N_TILE;
                RegTensor<float> a, b, high, av, bv, denominator, exponential;
                MaskReg previous_high;
                vlds(a, ps, offset, NORM);
                vlds(b, cs, offset, NORM);
                vmax(high, a, b, full, MODE_ZEROING);
                vsub(high, high, high, full);
                vadds(high, high, 1.0f, full, MODE_ZEROING);
                vsub(exponential, a, b, full);
                vabs(exponential, exponential, full);
                vmuls(exponential, exponential, -1.0f, full, MODE_ZEROING);
                vexp(exponential, exponential, full, MODE_ZEROING);
                vcmp_ge(previous_high, a, b, full);
                vsel(a, high, exponential, previous_high);
                vsel(b, exponential, high, previous_high);
                vlds(av, pk, offset, NORM);
                vlds(bv, ck, offset, NORM);
                vadd(denominator, a, b, full, MODE_ZEROING);
                vdup(high, 1.0f, full, MODE_ZEROING);
                vdiv(denominator, high, denominator, full, MODE_ZEROING);
                vmul(a, a, denominator, full, MODE_ZEROING);
                vmul(b, b, denominator, full, MODE_ZEROING);
                vmul(av, av, a, full, MODE_ZEROING);
                vmul(bv, bv, b, full, MODE_ZEROING);
                vadd(value, av, bv, full, MODE_ZEROING);
            } else {
                vdup(value, 0.0f, full, MODE_ZEROING);
            }
            RegTensor<bfloat16_t> bf;
            vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(bf, dst, row * N_TILE, PK_B32, full);
        }
    }
#endif
}

__aicore__ inline void load_pair(__gm__ int64_t* args, int local_row, int output_row,
                                 __gm__ float* previous_kv, __gm__ float* previous_score) {
    Vector<float, N_TILE> ck, cs, pk, ps;
    TASSIGN(ck, output_row * N_TILE * 4);
    TASSIGN(cs, 8192 + output_row * N_TILE * 4);
    TASSIGN(pk, 16384 + output_row * N_TILE * 4);
    TASSIGN(ps, 24576 + output_row * N_TILE * 4);
    Global<float, N_TILE> ck_gm(data<float>(args, 11) + (int64_t)local_row * N_TILE);
    Global<float, N_TILE> cs_gm(data<float>(args, 12) + (int64_t)local_row * N_TILE);
    Global<float, N_TILE> pk_gm(previous_kv), ps_gm(previous_score);
    TLOAD(ck, ck_gm);
    TLOAD(cs, cs_gm);
    TLOAD(pk, pk_gm);
    TLOAD(ps, ps_gm);
}

__aicore__ inline void ordinary_pool(__gm__ int64_t* args, int row, int rows, int col, int scratch,
                                    RequestMetadata& request_metadata) {
    // These 64-row buffers drain before MainState reuses the upper UB region.
    // Historical pooling keeps its separate 32-row layout below STATE_KV.
    VT<float, 64, N_TILE> ck, cs, pk, ps;
    VT<bfloat16_t, 64, N_TILE> output;
    TASSIGN(ck, 0); TASSIGN(cs, 16384); TASSIGN(pk, 32768); TASSIGN(ps, 49152); TASSIGN(output, 65536);
    using Metadata = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
    using MetadataGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 64>>;
    using Slots = Tile<TileType::Vec, int64_t, 1, 64, BLayout::RowMajor, 1, DYNAMIC>;
    using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 64>>;
    Metadata positions(64), requests(64);
    Slots targets(64);
    TASSIGN(positions, 73728); TASSIGN(requests, 73984); TASSIGN(targets, 74240);
    int cached = -1, start = 0, end = 0, block = -1;
    int lane = get_sub_block_id(args);
    int split = (rows + 15) / 16 * 8;
    int lane_end = lane == 0 ? split : rows;
    if (lane_end > rows) lane_end = rows;
    for (int base = lane * split; base < lane_end; base += 64) {
        int count = lane_end - base < 64 ? lane_end - base : 64;
        positions.SetValidCol(count); requests.SetValidCol(count); targets.SetValidCol(count);
        MetadataGM pg(data<int32_t>(args, 4) + row + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
        MetadataGM rg(data<int32_t>(args, 5) + row + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
        TLOAD(positions, pg); TLOAD(requests, rg);
        if (col == 0) {
            SlotGM slots(data<int64_t>(args, 8) + row + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
            TLOAD(targets, slots);
        }
        using Input = Tile<TileType::Vec, float, 64, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
        using InputGM = GlobalTensor<float, TileShape2D<float, DYNAMIC, N_TILE>,
                                     BaseShape2D<float, DYNAMIC, N_TILE>>;
        Input current_kv(count), current_score(count);
        TASSIGN(current_kv, 0); TASSIGN(current_score, 16384);
        InputGM cg(data<float>(args, 11) + (int64_t)(scratch + base) * N_TILE,
                   TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
        InputGM sg(data<float>(args, 12) + (int64_t)(scratch + base) * N_TILE,
                   TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
        TLOAD(current_kv, cg); TLOAD(current_score, sg);
        if (base > 0) {
            Input previous_kv(count), previous_score(count);
            TASSIGN(previous_kv, 32768); TASSIGN(previous_score, 49152);
            InputGM pg(data<float>(args, 11) + (int64_t)(scratch + base - 1) * N_TILE,
                       TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
            InputGM ps(data<float>(args, 12) + (int64_t)(scratch + base - 1) * N_TILE,
                       TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
            TLOAD(previous_kv, pg); TLOAD(previous_score, ps);
        } else {
            Vector<float, N_TILE> previous_kv, previous_score;
            TASSIGN(previous_kv, 32768); TASSIGN(previous_score, 49152);
            int previous = boundary_needed(args, row) ? scratch + M_TILE : scratch;
            Global<float, N_TILE> pg(data<float>(args, 11) + (int64_t)previous * N_TILE);
            Global<float, N_TILE> ps(data<float>(args, 12) + (int64_t)previous * N_TILE);
            TLOAD(previous_kv, pg); TLOAD(previous_score, ps);
            if (count > 1) {
                Input tail_kv(count - 1), tail_score(count - 1);
                TASSIGN(tail_kv, 32768 + N_TILE * 4); TASSIGN(tail_score, 49152 + N_TILE * 4);
                InputGM tg(data<float>(args, 11) + (int64_t)scratch * N_TILE,
                           TileShape2D<float, DYNAMIC, N_TILE>(count - 1, N_TILE));
                InputGM ts(data<float>(args, 12) + (int64_t)scratch * N_TILE,
                           TileShape2D<float, DYNAMIC, N_TILE>(count - 1, N_TILE));
                TLOAD(tail_kv, tg); TLOAD(tail_score, ts);
            }
        }
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        uint64_t active = 0, historical = 0;
        for (int r = 0; r < count; ++r) {
            int token = row + base + r;
            int request = requests.GetValue(r), position = positions.GetValue(r);
            int64_t target = col == 0 ? targets.GetValue(r) : int64_t(-1);
            targets.SetValue(r, int64_t(-1));
            if (request < 0 || request >= request_count(args) || position < 0 || position % 2 == 0) continue;
            if (request != cached) {
                start = request_metadata.start(args, request);
                end = request_metadata.start(args, request, 1);
                block = request_metadata.block(args, request);
                cached = request;
            }
            if (token < start || token >= end || block < 0 || block >= block_count(args)) continue;
            if (col == 0) targets.SetValue(r, target);
            if (token == start) { historical |= 1ULL << r; continue; }
            active |= 1ULL << r;
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pool_rows<64>(output.data(), ck.data(), cs.data(), pk.data(), ps.data(), active, count);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (historical == 0) {
            using Output = Tile<TileType::Vec, bfloat16_t, 64, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
            using OutputGM = GlobalTensor<bfloat16_t, TileShape2D<bfloat16_t, DYNAMIC, N_TILE>,
                                          BaseShape2D<bfloat16_t, DYNAMIC, D>>;
            Output value(count);
            TASSIGN(value, 65536);
            OutputGM dst(data<bfloat16_t>(args, 9) + (int64_t)(row + base) * D + col,
                         TileShape2D<bfloat16_t, DYNAMIC, N_TILE>(count, N_TILE),
                         BaseShape2D<bfloat16_t, DYNAMIC, D>(count, D));
            TSTORE(dst, value);
        } else {
            for (int r = 0; r < count; ++r) {
                if (historical & (1ULL << r)) continue;
                Vector<bfloat16_t, N_TILE> value;
                TASSIGN(value, 65536 + r * N_TILE * 2);
                Global<bfloat16_t, N_TILE> dst(data<bfloat16_t>(args, 9) + (int64_t)(row + base + r) * D + col);
                TSTORE(dst, value);
            }
        }
        if (col == 0) {
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            SlotGM target_gm(data<int64_t>(args, 10) + row + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
            TSTORE(target_gm, targets);
        }
        pipe_barrier(PIPE_ALL);
    }
}

__aicore__ inline void historical_pool(__gm__ int64_t* args, int token, int block, int col, int local_row, int position) {
    int64_t slot = (int64_t)block * ring_capacity(args) + (position - 1) % ring_capacity(args);
    load_pair(args, local_row, 0, data<float>(args, 7) + slot * 1024 + col,
              data<float>(args, 7) + slot * 1024 + D + col);
    VT<float, 32, N_TILE> ck, cs, pk, ps;
    VT<bfloat16_t, 32, N_TILE> output;
    TASSIGN(ck, 0); TASSIGN(cs, 8192); TASSIGN(pk, 16384); TASSIGN(ps, 24576); TASSIGN(output, 32768);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pool_rows<32>(output.data(), ck.data(), cs.data(), pk.data(), ps.data(), 1ULL, 1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Vector<bfloat16_t, N_TILE> value;
    TASSIGN(value, 32768);
    Global<bfloat16_t, N_TILE> dst(data<bfloat16_t>(args, 9) + (int64_t)token * D + col);
    TSTORE(dst, value);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void write_state(__gm__ int64_t* args, int request, int begin, int end,
                                  int block, int col, int source_start, int row, bool staged, MainState& main_state) {
    if (staged) {
        using State = Tile<TileType::Vec, float, 8, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
        using StateGM = GlobalTensor<float, TileShape2D<float, DYNAMIC, N_TILE>,
                                    BaseShape2D<float, DYNAMIC, 1024>>;
        int capacity = ring_capacity(args);
        for (int token = begin; token < end;) {
            int position = main_state.position(token - row);
            if (position < 0 || main_state.request(token - row) != request) {
                ++token;
                continue;
            }
            int ring_row = position % capacity;
            int count = 1;
            while (count < 8 && token + count < end && ring_row + count < capacity) {
                int next_position = main_state.position(token + count - row);
                if (int64_t(next_position) != int64_t(position) + count ||
                    main_state.request(token + count - row) != request) break;
                ++count;
            }
            State value(count), gate(count);
            TASSIGN(value, STATE_KV + (token - row) * N_TILE * 4);
            TASSIGN(gate, STATE_SCORE + (token - row) * N_TILE * 4);
            int64_t slot = (int64_t)block * capacity + ring_row;
            StateGM kd(data<float>(args, 7) + slot * 1024 + col,
                       TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE),
                       BaseShape2D<float, DYNAMIC, 1024>(count, 1024));
            StateGM sd(data<float>(args, 7) + slot * 1024 + D + col,
                       TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE),
                       BaseShape2D<float, DYNAMIC, 1024>(count, 1024));
            TSTORE(kd, value); TSTORE(sd, gate);
            token += count;
        }
        return;
    }
    using State = Tile<TileType::Vec, float, 8, N_TILE, BLayout::RowMajor, DYNAMIC, N_TILE>;
    using StateGM = GlobalTensor<float, TileShape2D<float, DYNAMIC, N_TILE>,
                                BaseShape2D<float, DYNAMIC, N_TILE>>;
    using Metadata = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, DYNAMIC>;
    using MetadataGM = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC>, BaseShape2D<int32_t, 1, 8>>;
    for (int base = begin; base < end; base += 8) {
        int count = end - base < 8 ? end - base : 8;
        State kv(count), score(count);
        Metadata positions(count), requests(count);
        TASSIGN(kv, 0); TASSIGN(score, 2048);
        TASSIGN(positions, 4096); TASSIGN(requests, 4128);
        int64_t local = (int64_t)(source_start + base - begin) * N_TILE;
        StateGM kg(data<float>(args, 11) + local, TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
        StateGM sg(data<float>(args, 12) + local, TileShape2D<float, DYNAMIC, N_TILE>(count, N_TILE));
        MetadataGM pg(data<int32_t>(args, 4) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
        MetadataGM rg(data<int32_t>(args, 5) + base, TileShape2D<int32_t, 1, DYNAMIC>(1, count));
        TLOAD(kv, kg); TLOAD(score, sg);
        TLOAD(positions, pg); TLOAD(requests, rg);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        for (int r = 0; r < count; ++r) {
            int position = positions.GetValue(r);
            if (position < 0 || requests.GetValue(r) != request) continue;
            Vector<float, N_TILE> value, gate;
            TASSIGN(value, r * N_TILE * 4); TASSIGN(gate, 2048 + r * N_TILE * 4);
            int64_t slot = (int64_t)block * ring_capacity(args) + position % ring_capacity(args);
            Global<float, N_TILE> kd(data<float>(args, 7) + slot * 1024 + col);
            Global<float, N_TILE> sd(data<float>(args, 7) + slot * 1024 + D + col);
            TSTORE(kd, value); TSTORE(sd, gate);
        }
        if (base + count < end) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif

__aicore__ inline void fused(__gm__ int64_t* args) {
    int tasks = (args[13] + M_TILE - 1) / M_TILE * (D / N_TILE);
    int scratch = get_block_idx(args) * SCRATCH_ROWS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int row = task / (D / N_TILE) * M_TILE, col = task % (D / N_TILE) * N_TILE;
        int rows = args[13] - row < M_TILE ? args[13] - row : M_TILE;
        bool boundary = boundary_needed(args, row);
#if defined(__DAV_VEC__)
        // Immutable request bounds can load while Cube projects this tile.
        RequestMetadata request_metadata;
        if (request_count(args) > 0) request_metadata.load(args, 0);
#endif
        begin_projection(args, row, rows, col, scratch);
        if (boundary) {
            end_projection();
            begin_projection(args, row - 1, 1, col, scratch + M_TILE);
        }
#if defined(__DAV_VEC__)
        ordinary_pool(args, row, rows, col, scratch, request_metadata);
#endif
        // One row tile never reprojects support rows. FINISH drains state writes
        // before this worker reuses its scratch for the next logical channel.
        if (args[13] > M_TILE) end_projection();
#if defined(__DAV_VEC__)
        bool staged = args[13] <= 128;
        MainState main_state;
        if (staged) main_state.load(args, row, rows, scratch);
#endif
        // Only the tile containing a block's last request tail may access its ring.
        // Recompute remote support rows locally; other logical tiles never wait.
#if defined(__DAV_CUBE__)
        if (args[13] > M_TILE) {
#endif
        for (int request = 0; request < request_count(args); ++request) {
#if defined(__DAV_VEC__)
            int block = request_metadata.block(args, request);
#else
            int block = metadata(data<int32_t>(args, 6) + request);
#endif
            if (block < 0 || block >= block_count(args) || owner_tile(args, block) != row / M_TILE) continue;
#if defined(__DAV_VEC__)
            if (args[13] <= M_TILE && get_sub_block_id(args) != block % 2) continue;
#endif
#if defined(__DAV_VEC__)
            int start = request_metadata.start(args, request);
            int end = request_metadata.start(args, request, 1);
#else
            int start = metadata(data<int32_t>(args, 3) + request);
            int end = metadata(data<int32_t>(args, 3) + request + 1);
#endif
#if defined(__DAV_VEC__)
            bool historical = staged ? start >= row && start < row + rows && start < end &&
                              main_state.position(start - row) >= 0 && main_state.position(start - row) % 2 &&
                              main_state.request(start - row) == request : first_odd(args, request, start, end);
#else
            bool historical = first_odd(args, request, start, end);
#endif
            if (historical) {
                bool remote = start < row || start >= row + rows;
                if (remote) begin_projection(args, start, 1, col, scratch + M_TILE);
#if defined(__DAV_VEC__)
                if (get_sub_block_id(args) == block % 2) {
                    if (staged) {
                        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
                        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
                    }
                    int position = staged ? main_state.position(start - row) : metadata(data<int32_t>(args, 4) + start);
                    historical_pool(args, start, block, col, remote ? scratch + M_TILE : scratch + start - row, position);
                }
#endif
                if (remote) end_projection();
            }
            if (start < 0) start = 0;
            if (end > args[13]) end = args[13];
            if (start < end - ring_capacity(args)) start = end - ring_capacity(args);
            if (start >= end) continue;
            bool remote = start < row || end > row + rows;
            if (remote) begin_projection(args, start, end - start, col, scratch + M_TILE);
#if defined(__DAV_VEC__)
            if (get_sub_block_id(args) == block % 2)
                write_state(args, request, start, end, block, col,
                            remote ? scratch + M_TILE : scratch + start - row, row, staged, main_state);
#endif
            if (remote) end_projection();
        }
#if defined(__DAV_CUBE__)
        }
        wait_pair<PIPE_FIX, FINISH>();
#elif defined(__DAV_VEC__)
        pipe_barrier(PIPE_ALL);
        signal_pair<PIPE_MTE3, FINISH>();
#endif
    }
    pipe_barrier(PIPE_ALL);
}
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    fused(args);
}
