// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 router projection with FP32 accumulation, sqrt-softplus, and selection bias.
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"

using namespace pto;

namespace {
constexpr int M = 16;
constexpr int K = 1024;
using Pipe = TPipe<0, Direction::DIR_C2V, M * M * sizeof(float), 2, 2, true>;
using Scores = Tile<TileType::Vec, float, M, M>;
using Bias = Tile<TileType::Vec, float, M, M, BLayout::RowMajor, M, DYNAMIC>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}

__aicore__ inline int dim(__gm__ int64_t* args, int index, int axis) {
    return reinterpret_cast<__gm__ Tensor*>(args[index])->shapes[axis];
}

template <int R, int C>
using Shape = TileShape2D<float, R, C>;
using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;

#if defined(__DAV_VEC__)
#if defined(PTO_NPU_ARCH_A5)
__tf__ __aicore__ inline void score_vf(
    Scores::TileDType __out__ scores, Scores::TileDType __out__ biased,
    Scores::TileDType __in__ logits, Bias::TileDType __in__ bias, uint16_t valid) {
    auto src = (__ubuf__ float*)__cce_get_tile_ptr(logits);
    auto gamma = (__ubuf__ float*)__cce_get_tile_ptr(bias);
    auto dst = (__ubuf__ float*)__cce_get_tile_ptr(scores);
    auto selection = (__ubuf__ float*)__cce_get_tile_ptr(biased);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        RegTensor<uint32_t> lane, column, low_bits;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vdup(low_bits, 15u, all, MODE_ZEROING);
        vand(column, lane, low_bits, all, MODE_ZEROING);
        MaskReg valid_columns;
        vcmps_lt(valid_columns, column, static_cast<uint32_t>(valid), all);
        RegTensor<float> zero, one, minus_inf, b;
        vdup(zero, 0.0f, all, MODE_ZEROING);
        vdup(one, 1.0f, all, MODE_ZEROING);
        vdup(minus_inf, -3.4028234663852886e38f, all, MODE_ZEROING);
        for (uint16_t segment = 0; segment < 4; ++segment) {
            vlds(b, gamma, segment * 64, NORM);
            vsel(b, b, minus_inf, valid_columns);
            RegTensor<float> x, relu, absolute, tail, mask, floor, score;
            vlds(x, src, segment * 64, NORM);
            vmax(relu, x, zero, all, MODE_ZEROING);
            vabs(absolute, x, all, MODE_ZEROING);
            vmuls(tail, absolute, -1.0f, all, MODE_ZEROING);
            vexp(tail, tail, all, MODE_ZEROING);
            vadds(tail, tail, 1.0f, all, MODE_ZEROING);
            vln(tail, tail, all, MODE_ZEROING);
            vadd(tail, relu, tail, all, MODE_ZEROING);
            vmuls(mask, x, -1.0f, all, MODE_ZEROING);
            vadds(mask, mask, -10.0f, all, MODE_ZEROING);
            vmax(mask, mask, zero, all, MODE_ZEROING);
            vmin(mask, mask, one, all, MODE_ZEROING);
            vmin(floor, x, zero, all, MODE_ZEROING);
            vexp(floor, floor, all, MODE_ZEROING);
            vmul(floor, mask, floor, all, MODE_ZEROING);
            vmax(score, tail, floor, all, MODE_ZEROING);
            vsqrt(score, score, all, MODE_ZEROING);
            vsts(score, dst, segment * 64, NORM_B32, all);
            vadd(score, score, b, all, MODE_ZEROING);
            vsts(score, selection, segment * 64, NORM_B32, all);
        }
    }
}
#else
__aicore__ inline void score_tiles(Scores& scores, Scores& biased, Scores& logits, Bias& bias) {
    Scores relu, absolute, tail, mask, floor;
    TASSIGN(relu, 68608); TASSIGN(absolute, 69632); TASSIGN(tail, 70656);
    TASSIGN(mask, 71680); TASSIGN(floor, 72704);
    TMAXS(relu, logits, 0.0f); TABS(absolute, logits); TNEG(tail, absolute);
    TEXP(tail, tail); TADDS(tail, tail, 1.0f); TLOG(tail, tail); TADD(tail, relu, tail);
    TNEG(mask, logits); TSUBS(mask, mask, 10.0f); TMAXS(mask, mask, 0.0f); TMINS(mask, mask, 1.0f);
    TMINS(floor, logits, 0.0f); TEXP(floor, floor); TMUL(floor, mask, floor);
    TMAX(scores, tail, floor); TSQRT(scores, scores); TADD(biased, scores, bias);
}
#endif

__aicore__ inline void vector_gate(__gm__ int64_t* args) {
    int experts = dim(args, 1, 0), columns = (experts + 15) / 16;
    int out_stride = dim(args, 3, 1);
    Pipe pipe(nullptr, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int work = get_block_idx(args); work < args[5]; work += get_block_num(args)) {
        int t0 = work / columns * M, n0 = work % columns * M;
        int valid = experts - n0;
        valid = valid > M ? M : valid;
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        Scores logits, scores, biased;
        Bias bias(valid);
        TASSIGN(scores, 65536); TASSIGN(biased, 66560); TASSIGN(bias, 67584);
        int bias_stride = args[6];
        GlobalTensor<float, Shape<M, DYNAMIC>, Stride> bias_gm(
            data<float>(args, 2) + t0 * bias_stride + n0,
            Shape<M, DYNAMIC>(M, valid), Stride(1, 1, 1, bias_stride, 1));
        TLOAD(bias, bias_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TPOP<Pipe, Scores, TileSplitAxis::TILE_NO_SPLIT>(pipe, logits);
#if defined(PTO_NPU_ARCH_A5)
        score_vf(scores.data(), biased.data(), logits.data(), bias.data(), valid);
#else
        using BiasPad = Tile<TileType::Vec, float, M, M, BLayout::RowMajor, M, M,
                             SLayout::NoneBox, 512, PadValue::Min>;
        BiasPad padded_bias;
        TASSIGN(padded_bias, 67584);
        TFILLPAD<pto::TFillPadMode::InPlace>(padded_bias, bias);
        Bias full_bias(M);
        TASSIGN(full_bias, 67584);
        score_tiles(scores, biased, logits, full_bias);
#endif
        pipe_barrier(PIPE_ALL);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        GlobalTensor<float, Shape<M, M>, Stride> scores_gm(
            data<float>(args, 3) + t0 * out_stride + n0, Shape<M, M>(), Stride(1, 1, 1, out_stride, 1));
        GlobalTensor<float, Shape<M, M>, Stride> biased_gm(
            data<float>(args, 4) + t0 * out_stride + n0, Shape<M, M>(), Stride(1, 1, 1, out_stride, 1));
        TSTORE(scores_gm, scores); TSTORE(biased_gm, biased);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    pipe_barrier(PIPE_ALL);
}
#endif

#if defined(__DAV_CUBE__)
using Mat = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor>;
using MatWeight = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, DYNAMIC, K,
                       SLayout::RowMajor, 512, PadValue::Zero>;
using Transposed = Tile<TileType::Mat, bfloat16_t, K, M, BLayout::RowMajor, K, DYNAMIC,
                        SLayout::ColMajor, 512, PadValue::Zero>;
using Left = Tile<TileType::Left, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor>;
using Right = Tile<TileType::Right, bfloat16_t, K, M, BLayout::RowMajor, K, DYNAMIC,
                   SLayout::ColMajor, 512, PadValue::Zero>;
using Acc = Tile<TileType::Acc, float, M, M, BLayout::ColMajor, M, M, SLayout::RowMajor, 1024>;
constexpr int SLOT_BYTES = M * K * sizeof(bfloat16_t);

__aicore__ inline void load_projection(__gm__ int64_t* args, int t0, int n0, int k0, int slot, int valid) {
    int width = dim(args, 0, 1);
    Mat input;
    MatWeight weight(valid);
    TASSIGN(input, slot * 2 * SLOT_BYTES);
    TASSIGN(weight, (slot * 2 + 1) * SLOT_BYTES);
    using InputShape = TileShape2D<bfloat16_t, M, K>;
    using WeightShape = TileShape2D<bfloat16_t, DYNAMIC, K>;
    GlobalTensor<bfloat16_t, InputShape, Stride> input_gm(
        data<bfloat16_t>(args, 0) + t0 * width + k0, InputShape(), Stride(1, 1, 1, width, 1));
    GlobalTensor<bfloat16_t, WeightShape, Stride> weight_gm(
        data<bfloat16_t>(args, 1) + n0 * width + k0, WeightShape(valid, K), Stride(1, 1, 1, width, 1));
    TLOAD(input, input_gm);
    TLOAD(weight, weight_gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

__aicore__ inline void cube_gate(__gm__ int64_t* args) {
    int experts = dim(args, 1, 0), width = dim(args, 0, 1), columns = (experts + 15) / 16;
    Pipe pipe(nullptr, 0, 0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int slot = 0; slot < 2; ++slot) {
        set_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
        set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    }
    for (int work = get_block_idx(args); work < args[5]; work += get_block_num(args)) {
        int t0 = work / columns * M, n0 = work % columns * M;
        int valid = experts - n0;
        valid = valid > M ? M : valid;
        Acc acc;
        TASSIGN(acc, 0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int slot = 0; slot < 2; ++slot) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
            load_projection(args, t0, n0, slot * K, slot, valid);
        }
        for (int step = 0; step < width / K; ++step) {
            int slot = step % 2;
            auto event = static_cast<event_t>(slot);
            Mat input;
            MatWeight weight(valid);
            Transposed transposed(valid);
            Left left;
            Right right(valid);
            TASSIGN(input, slot * 2 * SLOT_BYTES);
            TASSIGN(weight, (slot * 2 + 1) * SLOT_BYTES);
            TASSIGN(left, slot * SLOT_BYTES);
            TASSIGN(right, slot * SLOT_BYTES);
            wait_flag(PIPE_MTE2, PIPE_MTE1, event);
            wait_flag(PIPE_M, PIPE_MTE1, event);
            TRESHAPE(transposed, weight);
            TMOV(left, input);
            TMOV(right, transposed);
            set_flag(PIPE_MTE1, PIPE_M, event);
            wait_flag(PIPE_MTE1, PIPE_M, event);
            if (step == 0) TMATMUL(acc, left, right);
            else TMATMUL_ACC(acc, acc, left, right);
            set_flag(PIPE_M, PIPE_MTE1, event);
            set_flag(PIPE_MTE1, PIPE_MTE2, event);
            if (step + 2 < width / K) {
                wait_flag(PIPE_MTE1, PIPE_MTE2, event);
                load_projection(args, t0, n0, (step + 2) * K, slot, valid);
            }
        }
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TPUSH<Pipe, Acc, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    }
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int slot = 0; slot < 2; ++slot) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(slot));
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
    }
    pipe_barrier(PIPE_ALL);
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if !defined(__CPU_SIM)
    set_atomic_none();
#endif
#if defined(__DAV_CUBE__)
    cube_gate(args);
#elif defined(__DAV_VEC__)
#if !defined(__CPU_SIM)
    set_mask_norm();
    set_vector_mask(-1, -1);
#endif
    vector_gate(args);
#endif
}
