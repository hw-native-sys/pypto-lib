// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Unified HC projection, raw-input RMS, stream collapse, and coefficients.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
using namespace pto;
namespace stage {
constexpr int DK = 192;
constexpr int K = 20480;
constexpr int D = 5120;
constexpr int OWNERS = 27;
constexpr int RAW_COLS = DK + 8;

template <int MT, int WIDTH = DK>
struct Buffers {
    static constexpr int DK = WIDTH;
    static constexpr int RAW_COLS = DK + 8;
    static constexpr int INNER_K = DK == 64 ? 32 : 64;
    static constexpr int NZ_ROWS = ((MT / 2 + 15) / 16) * 16;
    static constexpr int RAW_BYTES = NZ_ROWS * RAW_COLS * 4;
    static constexpr int NZ_BASE = 4 * RAW_BYTES;
    static constexpr int A_BYTES = MT * DK * 4;
    static constexpr int OUTPUT_BASE = NZ_BASE;
    static constexpr int MIX_BASE = DK == 64 ? NZ_BASE + NZ_ROWS * DK * 4 : OUTPUT_BASE + NZ_ROWS * DK * 2;
    static constexpr int RMS_BASE = MIX_BASE + NZ_ROWS * 8 * 4;
    static_assert(NZ_BASE + NZ_ROWS * DK * 4 <= 253952, "UB conversion capacity");
    static constexpr int SUM_BASE = RMS_BASE + NZ_ROWS * 4;
    static_assert(DK != 64 || SUM_BASE + NZ_ROWS * 4 <= 253952, "UB persistent sum capacity");
    static_assert(DK == 64 || RMS_BASE + 256 <= NZ_BASE + NZ_ROWS * DK * 4, "UB phase alias capacity");
    static_assert(4 * A_BYTES + 4 * DK * 32 * 4 <= 524288, "L1 capacity");
    static_assert((DK == 64 ? 2 : 1) * MT * INNER_K * 4 <= 65536, "L0A capacity");
    using MatA = Tile<TileType::Mat, float, MT, DK, BLayout::ColMajor, MT, DK, SLayout::RowMajor>;
    using MatB = Tile<TileType::Mat, float, DK, 32, BLayout::RowMajor, DYNAMIC, 24, SLayout::ColMajor>;
    using Left = TileLeftCompact<float, MT, INNER_K, MT, INNER_K>;
    using Right = TileRight<float, INNER_K, 32, INNER_K, 24>;
    using Acc = TileAccCompact<float, MT, 32, MT, 24>;
#if defined(__DAV_VEC__)
    using Input = Tile<TileType::Vec, float, NZ_ROWS, RAW_COLS, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using NZ = Tile<TileType::Vec, float, NZ_ROWS, DK, BLayout::ColMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor>;
    using Output = Tile<TileType::Vec, bfloat16_t, NZ_ROWS, DK, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Mix = Tile<TileType::Vec, float, NZ_ROWS, 8, BLayout::RowMajor, DYNAMIC, 4>;
    using Rms = Tile<TileType::Vec, float, 1, DK == 64 ? NZ_ROWS : 64, BLayout::RowMajor, 1, DYNAMIC>;
#endif
};

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

#if defined(__DAV_VEC__)
template <int MT, int WIDTH = DK>
__tf__ __aicore__ inline void padded_nd_to_nz(
    typename Buffers<MT, WIDTH>::NZ::TileDType __out__ output,
    typename Buffers<MT, WIDTH>::Input::TileDType __in__ input,
    uint16_t rows, uint16_t width) {
    constexpr int RAW_COLS = WIDTH + 8;
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        for (uint16_t row = 0; row < rows; row += 8) {
            uint32_t active = (rows - row < 8 ? rows - row : 8) * 8;
            MaskReg mask = plt_b32(active, POST_UPDATE);
            for (uint16_t col = 0; col < width; col += 8) {
                RegTensor<float> values;
                vsldb(values, src + row * RAW_COLS + col, (RAW_COLS / 8) << 16, mask);
                vsts(values, dst + (col / 8) * Buffers<MT, WIDTH>::NZ_ROWS * 8 + row * 8, 0, NORM_B32, mask);
            }
        }
    }
}

template <int MT, int WIDTH = DK>
__tf__ __aicore__ inline void stats(
    typename Buffers<MT, WIDTH>::Output::TileDType __out__ output, typename Buffers<MT, WIDTH>::Rms::TileDType __out__ rms,
    typename Buffers<MT, WIDTH>::Input::TileDType __in__ v0, typename Buffers<MT, WIDTH>::Input::TileDType __in__ v1,
    typename Buffers<MT, WIDTH>::Input::TileDType __in__ v2, typename Buffers<MT, WIDTH>::Input::TileDType __in__ v3,
    typename Buffers<MT, WIDTH>::Mix::TileDType __in__ premix, uint16_t rows, uint16_t width) {
    constexpr int RAW_COLS = WIDTH + 8;
    auto x0 = (__ubuf__ float*)__cce_get_tile_ptr(v0);
    auto x1 = (__ubuf__ float*)__cce_get_tile_ptr(v1);
    auto x2 = (__ubuf__ float*)__cce_get_tile_ptr(v2);
    auto x3 = (__ubuf__ float*)__cce_get_tile_ptr(v3);
    auto mix = (__ubuf__ float*)__cce_get_tile_ptr(premix);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto sq = (__ubuf__ float*)__cce_get_tile_ptr(rms);
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t one = 1;
        MaskReg scalar = plt_b32(one, POST_UPDATE);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> c0, c1, c2, c3, sum;
            vlds(c0, mix, row * 8, BRC_B32);
            vlds(c1, mix, row * 8 + 1, BRC_B32);
            vlds(c2, mix, row * 8 + 2, BRC_B32);
            vlds(c3, mix, row * 8 + 3, BRC_B32);
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t col = 0; col < width; col += 64) {
                RegTensor<float> a, b, c, d, p, q, y, square;
                RegTensor<bfloat16_t> rounded;
                vlds(a, x0, row * RAW_COLS + col, NORM);
                vlds(b, x1, row * RAW_COLS + col, NORM);
                vlds(c, x2, row * RAW_COLS + col, NORM);
                vlds(d, x3, row * RAW_COLS + col, NORM);
                vmul(square, a, a, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
                vmul(square, b, b, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
                vmul(square, c, c, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
                vmul(square, d, d, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
                vmul(a, a, c0, full, MODE_ZEROING);
                vmul(b, b, c1, full, MODE_ZEROING);
                vmul(c, c, c2, full, MODE_ZEROING);
                vmul(d, d, c3, full, MODE_ZEROING);
                vadd(p, a, b, full, MODE_ZEROING);
                vadd(q, c, d, full, MODE_ZEROING);
                vadd(y, p, q, full, MODE_ZEROING);
                vcvt(rounded, y, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(rounded, dst, row * WIDTH + col, PK_B32, full);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            RegTensor<uint32_t> location;
            vbr(location, static_cast<uint32_t>(row));
            vscatter(sum, sq, location, scalar);
        }
    }
}

template <int MT>
__aicore__ inline void vector_stage(__gm__ int64_t* args, int task) {
    using L = Buffers<MT>;
    using Input = typename L::Input;
    using NZ = typename L::NZ;
    using Rms = typename L::Rms;
    using Output = typename L::Output;
    using Mix = typename L::Mix;
    using MatA = typename L::MatA;
    const int tokens = args[11];
    const int padded = args[12];
    const int lane = get_sub_block_id(args);
    const int owner = task % OWNERS;
    const int row0 = task / OWNERS * MT;
    const int active = tokens - row0 < MT ? tokens - row0 : MT;
    const int split = (active + 1) / 2;
    const int first = row0 + (lane == 0 ? 0 : split);
    const int rows = lane == 0 ? split : active - split;
    const int width = owner == OWNERS - 1 ? 128 : DK;
    if (rows > 0) {
        Input in0(rows, width), in1(rows, width), in2(rows, width), in3(rows, width);
        TASSIGN(in0, 0);
        TASSIGN(in1, L::RAW_BYTES);
        TASSIGN(in2, 2 * L::RAW_BYTES);
        TASSIGN(in3, 3 * L::RAW_BYTES);
        using Shape = TileShape2D<float, DYNAMIC, DYNAMIC>;
        using Stride = BaseShape2D<float, DYNAMIC, K>;
        for (int hc = 0; hc < 4; ++hc) {
            Input input(rows, width);
            TASSIGN(input, hc * L::RAW_BYTES);
            GlobalTensor<float, Shape, Stride> gm(data<float>(args, 0) + first * K + hc * D + owner * DK,
                Shape(rows, width), Stride(rows, K));
            TLOAD(input, gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (int hc = 0; hc < 4; ++hc) {
            Input input(rows, width);
            NZ nz(L::NZ_ROWS, width), part(rows, width);
            MatA mat;
            TASSIGN(input, hc * L::RAW_BYTES);
            TASSIGN(nz, L::NZ_BASE);
            TASSIGN(part, L::NZ_BASE);
            TASSIGN(mat, hc * L::A_BYTES);
            if (hc > 0) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
            padded_nd_to_nz<MT>(nz.data(), input.data(), rows, width);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TINSERT(mat, part, first - row0, 0);
            set_intra_block(PIPE_MTE3, hc * 2);
        }
        // Complete the final NZ read before MTE2 reuses the shared arena.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        Mix mix(rows);
        TASSIGN(mix, L::MIX_BASE);
        using MixShape = TileShape2D<float, DYNAMIC, 4>;
        using MixStride = BaseShape2D<float, DYNAMIC, 4>;
        GlobalTensor<float, MixShape, MixStride> mg(data<float>(args, 1) + first * 4,
            MixShape(rows, 4), MixStride(rows, 4));
        TLOAD(mix, mg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        Output output(rows, width);
        Rms rms(rows);
        TASSIGN(output, L::OUTPUT_BASE);
        TASSIGN(rms, L::RMS_BASE);
        stats<MT>(output.data(), rms.data(), in0.data(), in1.data(), in2.data(), in3.data(), mix.data(), rows, width);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        using OutShape = TileShape2D<bfloat16_t, DYNAMIC, DYNAMIC>;
        using OutStride = BaseShape2D<bfloat16_t, DYNAMIC, D>;
        GlobalTensor<bfloat16_t, OutShape, OutStride> og(data<bfloat16_t>(args, 3) + first * D + owner * DK,
            OutShape(rows, width), OutStride(rows, D));
        TSTORE(og, output);
        using RShape = TileShape2D<float, 1, DYNAMIC>;
        using RStride = BaseShape2D<float, 1, DYNAMIC>;
        GlobalTensor<float, RShape, RStride> rg(data<float>(args, 5) + owner * padded + first,
            RShape(1, rows), RStride(1, rows));
        TSTORE(rg, rms);
    } else {
        for (int hc = 0; hc < 4; ++hc) {
            set_intra_block(PIPE_MTE3, hc * 2);
        }
    }
    wait_intra_block(PIPE_MTE3, 8);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_CUBE__)
template <int MT>
__aicore__ inline void cube_stage(__gm__ int64_t* args, int task) {
    using L = Buffers<MT>;
    using MatA = typename L::MatA;
    using MatB = typename L::MatB;
    using Left = typename L::Left;
    using Right = typename L::Right;
    using Acc = typename L::Acc;
    const int padded = args[12];
    const int owner = task % OWNERS;
    const int row0 = task / OWNERS * MT;
    const int width = owner == OWNERS - 1 ? 128 : DK;
    for (int hc = 0; hc < 4; ++hc) {
        MatB b(width);
        TASSIGN(b, 4 * L::A_BYTES + hc * DK * 32 * 4);
        using Shape = TileShape2D<float, DYNAMIC, 24, Layout::DN>;
        using Stride = BaseShape2D<float, K, 24, Layout::DN>;
        GlobalTensor<float, Shape, Stride, Layout::DN> wg(data<float>(args, 2) + hc * D + owner * DK,
            Shape(width, 24), Stride());
        TLOAD(b, wg);
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Acc acc;
    TASSIGN(acc, 0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    for (int hc = 0; hc < 4; ++hc) {
        wait_intra_block(PIPE_MTE1, hc * 2);
        wait_intra_block(PIPE_MTE1, hc * 2 + 16);
        MatA a;
        MatB b(width);
        TASSIGN(a, hc * L::A_BYTES);
        TASSIGN(b, 4 * L::A_BYTES + hc * DK * 32 * 4);
        for (int k = 0; k < width; k += 64) {
            Left left;
            Right right;
            TASSIGN(left, 0);
            TASSIGN(right, 0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            TEXTRACT(left, a, 0, k);
            TEXTRACT(right, b, k, 0);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            if (hc == 0 && k == 0) {
                TMATMUL(acc, left, right);
            } else {
                TMATMUL_ACC(acc, acc, left, right);
            }
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        }
    }
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_intra_block(PIPE_MTE1, 8);
    set_intra_block(PIPE_MTE1, 24);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    using Shape = TileShape2D<float, MT, 24>;
    using Stride = BaseShape2D<float, MT, 32>;
    GlobalTensor<float, Shape, Stride> out(data<float>(args, 4) + (owner * padded + row0) * 32);
    TSTORE(out, acc);
    pipe_barrier(PIPE_ALL);
}
#endif

template <int MT>
__aicore__ inline void project(__gm__ int64_t* args) {
    const int tasks = ((args[11] + MT - 1) / MT) * OWNERS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
#if defined(__DAV_CUBE__)
        cube_stage<MT>(args, task);
#elif defined(__DAV_VEC__)
        vector_stage<MT>(args, task);
#endif
    }
}
}

namespace prefill {
constexpr int DK = 64, K = 20480, D = 5120, OWNERS = 8, OWNER_D = 640;
using stage::data;
#if defined(__DAV_VEC__)
template <int MT>
__aicore__ inline void vector_stage(__gm__ int64_t* args, int task) {
    using L = stage::Buffers<MT, DK>;
    using Input = typename L::Input;
    using NZ = typename L::NZ;
    using Rms = typename L::Rms;
    using Output = typename L::Output;
    using Mix = typename L::Mix;
    using MatA = typename L::MatA;
    const int tokens = args[11], padded = args[12];
    const int lane = get_sub_block_id(args);
    const int owner = task % OWNERS, row0 = task / OWNERS * MT;
    const int active = tokens - row0 < MT ? tokens - row0 : MT;
    const int split = (active + 1) / 2;
    const int first = row0 + (lane == 0 ? 0 : split);
    const int rows = lane == 0 ? split : active - split;
    Mix mix(rows);
    Rms sum(rows), rms(rows);
    TASSIGN(mix, L::MIX_BASE);
    TASSIGN(rms, L::RMS_BASE);
    TASSIGN(sum, L::SUM_BASE);
    if (rows > 0) {
        using Shape = TileShape2D<float, DYNAMIC, 4>;
        using Stride = BaseShape2D<float, DYNAMIC, 4>;
        GlobalTensor<float, Shape, Stride> gm(data<float>(args, 1) + first * 4,
            Shape(rows, 4), Stride(rows, 4));
        TLOAD(mix, gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TEXPANDS(sum, 0.0f);
    }
    for (int d = 0; d < OWNER_D; d += DK) {
        if (rows > 0) {
            Input in0(rows, DK), in1(rows, DK), in2(rows, DK), in3(rows, DK);
            TASSIGN(in0, 0);
            TASSIGN(in1, L::RAW_BYTES);
            TASSIGN(in2, 2 * L::RAW_BYTES);
            TASSIGN(in3, 3 * L::RAW_BYTES);
            for (int hc = 0; hc < 4; ++hc) {
                Input input(rows, DK);
                NZ nz(L::NZ_ROWS, DK), part(rows, DK);
                MatA mat;
                TASSIGN(input, hc * L::RAW_BYTES);
                TASSIGN(nz, L::NZ_BASE);
                TASSIGN(part, L::NZ_BASE);
                TASSIGN(mat, hc * L::A_BYTES);
                using Shape = TileShape2D<float, DYNAMIC, DYNAMIC>;
                using Stride = BaseShape2D<float, DYNAMIC, K>;
                GlobalTensor<float, Shape, Stride> gm(data<float>(args, 0) + first * K + hc * D + owner * OWNER_D + d,
                    Shape(rows, DK), Stride(rows, K));
                TLOAD(input, gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (hc > 0) {
                    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                }
                stage::padded_nd_to_nz<MT, DK>(nz.data(), input.data(), rows, DK);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TINSERT(mat, part, first - row0, 0);
                set_intra_block(PIPE_MTE3, hc * 2);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            Output output(rows, DK);
            TASSIGN(output, L::OUTPUT_BASE);
            stage::stats<MT, DK>(output.data(), rms.data(), in0.data(), in1.data(), in2.data(), in3.data(), mix.data(), rows, DK);
            TADD(sum, sum, rms);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            using Shape = TileShape2D<bfloat16_t, DYNAMIC, DYNAMIC>;
            using Stride = BaseShape2D<bfloat16_t, DYNAMIC, D>;
            GlobalTensor<bfloat16_t, Shape, Stride> out(data<bfloat16_t>(args, 3) + first * D + owner * OWNER_D + d,
                Shape(rows, DK), Stride(rows, D));
            TSTORE(out, output);
            if (d + DK == OWNER_D) {
                using RShape = TileShape2D<float, 1, DYNAMIC>;
                using RStride = BaseShape2D<float, 1, DYNAMIC>;
                GlobalTensor<float, RShape, RStride> rg(data<float>(args, 5) + owner * padded + first,
                    RShape(1, rows), RStride(1, rows));
                TSTORE(rg, sum);
            }
        } else {
            for (int hc = 0; hc < 4; ++hc) {
                set_intra_block(PIPE_MTE3, hc * 2);
            }
        }
        wait_intra_block(PIPE_MTE3, 8);
        pipe_barrier(PIPE_ALL);
    }
}
#endif

#if defined(__DAV_CUBE__)
template <int MT>
__aicore__ inline void cube_stage(__gm__ int64_t* args, int task) {
    using L = stage::Buffers<MT, DK>;
    using MatA = typename L::MatA;
    using MatB = typename L::MatB;
    using Left = typename L::Left;
    using Right = typename L::Right;
    using Acc = typename L::Acc;
    const int padded = args[12];
    const int owner = task % OWNERS, row0 = task / OWNERS * MT;
    Acc acc;
    TASSIGN(acc, 0);
    for (int d = 0; d < OWNER_D; d += DK) {
        for (int hc = 0; hc < 4; ++hc) {
            MatB b(DK);
            TASSIGN(b, 4 * L::A_BYTES + hc * DK * 32 * 4);
            using Shape = TileShape2D<float, DYNAMIC, 24, Layout::DN>;
            using Stride = BaseShape2D<float, K, 24, Layout::DN>;
            GlobalTensor<float, Shape, Stride, Layout::DN> wg(data<float>(args, 2) + hc * D + owner * OWNER_D + d,
                Shape(DK, 24), Stride());
            TLOAD(b, wg);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        for (int hc = 0; hc < 4; ++hc) {
            wait_intra_block(PIPE_MTE1, hc * 2);
            wait_intra_block(PIPE_MTE1, hc * 2 + 16);
            MatA a;
            MatB b(DK);
            TASSIGN(a, hc * L::A_BYTES);
            TASSIGN(b, 4 * L::A_BYTES + hc * DK * 32 * 4);
            for (int k = 0; k < DK; k += 32) {
                const int slot = k / 32;
                const auto event = static_cast<event_t>(slot);
                Left left;
                Right right;
                TASSIGN(left, slot * MT * 32 * 4);
                TASSIGN(right, slot * 32 * 32 * 4);
                wait_flag(PIPE_M, PIPE_MTE1, event);
                TEXTRACT(left, a, 0, k);
                TEXTRACT(right, b, k, 0);
                set_flag(PIPE_MTE1, PIPE_M, event);
                wait_flag(PIPE_MTE1, PIPE_M, event);
                if (d == 0 && hc == 0 && k == 0) {
                    TMATMUL(acc, left, right);
                } else {
                    TMATMUL_ACC(acc, acc, left, right);
                }
                set_flag(PIPE_M, PIPE_MTE1, event);
            }
        }
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_intra_block(PIPE_MTE1, 8);
        set_intra_block(PIPE_MTE1, 24);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    using Shape = TileShape2D<float, MT, 24>;
    using Stride = BaseShape2D<float, MT, 32>;
    GlobalTensor<float, Shape, Stride> out(data<float>(args, 4) + (owner * padded + row0) * 32);
    TSTORE(out, acc);
    pipe_barrier(PIPE_ALL);
}
#endif

template <int MT>
__aicore__ inline void project(__gm__ int64_t* args) {
    const int tasks = ((args[11] + MT - 1) / MT) * OWNERS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
#if defined(__DAV_CUBE__)
        cube_stage<MT>(args, task);
#elif defined(__DAV_VEC__)
        vector_stage<MT>(args, task);
#endif
    }
}
}

#include "hc_coefficients.hpp"
namespace finalize {
template <int OWNERS, int ROWS>
using Project = Tile<TileType::Vec, float, OWNERS, ROWS * 32, BLayout::RowMajor, OWNERS, DYNAMIC>;
template <int OWNERS, int ROWS>
using Squares = Tile<TileType::Vec, float, OWNERS, ((ROWS + 7) / 8) * 8, BLayout::RowMajor, OWNERS, DYNAMIC>;
#if defined(__DAV_VEC__)
template <int OWNERS, int ROWS>
__tf__ __aicore__ inline void reduce(
    typename Input<ROWS>::TileDType __out__ total, Small::TileDType __out__ inverse,
    typename Project<OWNERS, ROWS>::TileDType __in__ partials,
    typename Squares<OWNERS, ROWS>::TileDType __in__ squares, uint16_t rows) {
    constexpr int SQ_COLS = ((ROWS + 7) / 8) * 8;
    auto p = (__ubuf__ float*)__cce_get_tile_ptr(partials);
    auto s = (__ubuf__ float*)__cce_get_tile_ptr(squares);
    auto out = (__ubuf__ float*)__cce_get_tile_ptr(total);
    auto inv = (__ubuf__ float*)__cce_get_tile_ptr(inverse);
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        uint32_t count = rows * 32;
        for (uint16_t c = 0; c < ROWS * 32; c += 64) {
            MaskReg mask = plt_b32(count, POST_UPDATE);
            RegTensor<float> sum, part;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t owner = 0; owner < OWNERS; ++owner) {
                vlds(part, p, owner * ROWS * 32 + c, NORM);
                vadd(sum, sum, part, mask, MODE_ZEROING);
            }
            vsts(sum, out, c, NORM_B32, mask);
        }
        uint32_t active = rows;
        MaskReg mask = plt_b32(active, POST_UPDATE);
        RegTensor<uint32_t> lane;
        vci((vector_s32&)lane, 0, INC_ORDER);
        RegTensor<float> sum, part, one;
        vdup(sum, 0.0f, full, MODE_ZEROING);
        for (uint16_t owner = 0; owner < OWNERS; ++owner) {
            vgather2(part, s + owner * SQ_COLS, lane, mask);
            vadd(sum, sum, part, mask, MODE_ZEROING);
        }
        vmuls(sum, sum, 1.0f / 20480.0f, mask, MODE_ZEROING);
        vadds(sum, sum, 1.0e-6f, mask, MODE_ZEROING);
        vsqrt(sum, sum, mask, MODE_ZEROING);
        vdup(one, 1.0f, full, MODE_ZEROING);
        vdiv(sum, one, sum, mask, MODE_ZEROING);
        vsts(sum, inv, 0, NORM_B32, mask);
    }
}
template <int OWNERS, int ROWS>
__aicore__ inline void finish(__gm__ int64_t* args) {
    constexpr int SQ_COLS = ((ROWS + 7) / 8) * 8;
    constexpr int SQUARES_BASE = OWNERS * ROWS * 32 * 4;
    constexpr int BIAS_BASE = ((SQUARES_BASE + OWNERS * SQ_COLS * 4 + 511) / 512) * 512;
    constexpr int SCALE_BASE = BIAS_BASE + 128;
    constexpr int TOTAL_BASE = SCALE_BASE + 128;
    constexpr int ZERO_BASE = TOTAL_BASE + ROWS * 32 * 4;
    constexpr int INVERSE_BASE = ZERO_BASE + ROWS * 32 * 4;
    constexpr int PRE_BASE = INVERSE_BASE + 128;
    constexpr int POST_BASE = PRE_BASE + ((ROWS * 8 * 4 + 127) / 128) * 128;
    constexpr int COMB_BASE = POST_BASE + ((ROWS * 8 * 4 + 127) / 128) * 128;
    static_assert(COMB_BASE + ROWS * 16 * 4 <= 253952, "UB finalizer capacity");
    const int tokens = args[11], padded = args[12];
    Small bias(24), scales(3);
    TASSIGN(bias, BIAS_BASE);
    TASSIGN(scales, SCALE_BASE);
    using Shape = TileShape2D<float, 1, DYNAMIC>;
    using Stride = BaseShape2D<float, 1, DYNAMIC>;
    GlobalTensor<float, Shape, Stride> bg(data(args, 7), Shape(1, 24), Stride(1, 24));
    GlobalTensor<float, Shape, Stride> sg(data(args, 6), Shape(1, 3), Stride(1, 3));
    TLOAD(bias, bg);
    TLOAD(scales, sg);
    for (int start = (2 * get_block_idx(args) + get_sub_block_id(args)) * ROWS; start < tokens; start += 2 * get_block_num(args) * ROWS) {
        const int rows = tokens - start < ROWS ? tokens - start : ROWS;
        Project<OWNERS, ROWS> projection(rows * 32);
        Squares<OWNERS, ROWS> squares(rows);
        TASSIGN(projection, 0);
        TASSIGN(squares, SQUARES_BASE);
        using PShape = TileShape2D<float, OWNERS, DYNAMIC>;
        using PStride = BaseShape2D<float, DYNAMIC, DYNAMIC>;
        GlobalTensor<float, PShape, PStride> pg(data(args, 4) + start * 32,
            PShape(OWNERS, rows * 32), PStride(OWNERS, padded * 32));
        GlobalTensor<float, PShape, PStride> rg(data(args, 5) + start,
            PShape(OWNERS, rows), PStride(OWNERS, padded));
        TLOAD(projection, pg);
        TLOAD(squares, rg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Input<ROWS> total(rows), zero(rows);
        Small inverse(rows);
        Output<ROWS> pre(rows), post(rows);
        Comb<ROWS> comb(rows);
        TASSIGN(total, TOTAL_BASE);
        TASSIGN(zero, ZERO_BASE);
        TASSIGN(inverse, INVERSE_BASE);
        TASSIGN(pre, PRE_BASE);
        TASSIGN(post, POST_BASE);
        TASSIGN(comb, COMB_BASE);
        TEXPANDS(zero, 0.0f);
        reduce<OWNERS, ROWS>(total.data(), inverse.data(), projection.data(), squares.data(), rows);
        pipe_barrier(PIPE_ALL);
        coefficients<ROWS>(pre.data(), post.data(), comb.data(), total.data(), zero.data(), zero.data(), zero.data(),
            inverse.data(), bias.data(), scales.data(), rows, 1.0e-6f, 20);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        using OShape = TileShape2D<float, DYNAMIC, 4>;
        using OStride = BaseShape2D<float, DYNAMIC, 4>;
        GlobalTensor<float, OShape, OStride> pre_g(data(args, 8) + start * 4, OShape(rows, 4), OStride(rows, 4));
        GlobalTensor<float, OShape, OStride> post_g(data(args, 9) + start * 4, OShape(rows, 4), OStride(rows, 4));
        using CShape = TileShape2D<float, DYNAMIC, 16>;
        using CStride = BaseShape2D<float, DYNAMIC, 16>;
        GlobalTensor<float, CShape, CStride> comb_g(data(args, 10) + start * 16, CShape(rows, 16), CStride(rows, 16));
        TSTORE(pre_g, pre);
        TSTORE(post_g, post);
        TSTORE(comb_g, comb);
        pipe_barrier(PIPE_ALL);
    }
}
#endif
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_CUBE__)
    uint64_t saved_ctrl = get_ctrl();
    stage::Buffers<64>::Acc mode;
    mode.SetMadTF32Mode(RoundMode::CAST_ROUND);
#endif
    const int bucket = args[12];
    if (args[11] > 512) {
        prefill::project<256>(args);
    } else if (args[11] > 128) {
        prefill::project<128>(args);
    } else if (bucket == 64) {
        stage::project<64>(args);
    } else if (bucket == 80) {
        stage::project<80>(args);
    } else if (bucket == 96) {
        stage::project<96>(args);
    } else {
        stage::project<128>(args);
    }
    pipe_barrier(PIPE_ALL);
    SYNCALL<SyncCoreType::Mix>();
#if defined(__DAV_VEC__)
    if (args[11] <= 128) {
        finalize::finish<27, 6>(args);
    } else {
        finalize::finish<8, 16>(args);
    }
#elif defined(__DAV_CUBE__)
    set_ctrl(saved_ctrl);
#endif
}
