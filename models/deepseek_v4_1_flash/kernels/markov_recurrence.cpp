// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
#include "markov_finalize.hpp"
#include "markov_publish.hpp"
#include "mixed_entry.hpp"

using namespace pto;
namespace {
constexpr int M = 16;
constexpr int N = 256;
constexpr int K = 256;
constexpr int InnerK = 64;
constexpr int Workers = 32;
constexpr int SlotBytes = M / 2 * N * sizeof(float);
using Raw = Tile<TileType::Vec, float, M / 2, N>;
using State = Tile<TileType::Vec, float, M / 2, 64>;
using IndexState = Tile<TileType::Vec, int32_t, M / 2, 64>;
using Scores = Tile<TileType::Vec, float, M / 2, 16>;
using Indices = Tile<TileType::Vec, int32_t, M / 2, 16>;

template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int index) {
    auto tensor = reinterpret_cast<__gm__ Tensor*>(args[index]);
    return reinterpret_cast<__gm__ T*>(tensor->buffer.addr) + tensor->start_offset;
}
__aicore__ inline __gm__ Tensor* descriptor(__gm__ int64_t* args, int index) {
    return reinterpret_cast<__gm__ Tensor*>(args[index]);
}

#if defined(__DAV_CUBE__)
using MatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, DYNAMIC, K, SLayout::RowMajor, 512, PadValue::Zero>;
using FullA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
using MatB = Tile<TileType::Mat, bfloat16_t, N, K, BLayout::ColMajor, N, K, SLayout::RowMajor, 512>;
using TransposedB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
using Left = TileLeft<bfloat16_t, M, InnerK>;
using Right = TileRight<bfloat16_t, InnerK, N>;
using Acc = TileAcc<float, M, N>;

__aicore__ inline void load_weight(__gm__ int64_t* args, int panel, int slot, int64_t weight_stride) {
    MatB b;
    TASSIGN(b, 16384 + slot * N * K * sizeof(bfloat16_t));
    using Shape = TileShape2D<bfloat16_t, N, K>;
    using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    GlobalTensor<bfloat16_t, Shape, Stride> gm(data<bfloat16_t>(args, 2) + panel * N * weight_stride,
        Shape(), Stride(1, 1, 1, weight_stride, 1));
    TLOAD(b, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(slot));
}

__aicore__ inline void cube(__gm__ int64_t* args) {
    int batch = descriptor(args, 0)->shapes[0];
    int64_t weight_stride = descriptor(args, 2)->strides[0];
    int panels = descriptor(args, 0)->shapes[2] / N;
    int worker = get_block_idx(args), workers = get_block_num(args);
    int waves = (panels + workers - 1 - worker) / workers;
    for (int first_row = 0; first_row < batch; first_row += M) {
        int valid = batch - first_row; valid = valid < M ? valid : M;
        MatA a(valid); FullA full_a;
        TASSIGN(a, 0); TASSIGN(full_a, 0);
        using ShapeA = TileShape2D<bfloat16_t, DYNAMIC, K>;
        using StrideA = pto::Stride<1, 1, 1, DYNAMIC, 1>;
        GlobalTensor<bfloat16_t, ShapeA, StrideA> input(data<bfloat16_t>(args, 1) + first_row * descriptor(args, 1)->strides[0],
            ShapeA(valid, K), StrideA(1, 1, 1, descriptor(args, 1)->strides[0], 1));
        TLOAD(a, input);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4); wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
        for (int slot = 0; slot < 2; ++slot) {
            set_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(slot + 4));
            if (slot < waves) load_weight(args, worker + slot * workers, slot, weight_stride);
        }
        for (int wave = 0; wave < waves; ++wave) {
            int slot = wave % 2;
            auto event = static_cast<event_t>(slot);
            auto acc_event = static_cast<event_t>(slot + 4);
            MatB weight; TransposedB transposed; Acc accum;
            TASSIGN(weight, 16384 + slot * N * K * sizeof(bfloat16_t));
            TASSIGN(transposed, 16384 + slot * N * K * sizeof(bfloat16_t));
            TASSIGN(accum, slot * M * N * sizeof(float));
            wait_flag(PIPE_MTE2, PIPE_MTE1, event);
            wait_flag(PIPE_FIX, PIPE_M, acc_event);
            for (int k = 0; k < K; k += InnerK) {
                int inner = k / InnerK % 2;
                auto inner_event = static_cast<event_t>(inner);
                Left left; Right right;
                TASSIGN(left, inner * M * InnerK * sizeof(bfloat16_t));
                TASSIGN(right, inner * InnerK * N * sizeof(bfloat16_t));
                wait_flag(PIPE_M, PIPE_MTE1, inner_event);
                TEXTRACT(left, full_a, 0, k);
                TEXTRACT(right, transposed, k, 0);
                set_flag(PIPE_MTE1, PIPE_M, inner_event); wait_flag(PIPE_MTE1, PIPE_M, inner_event);
                if (k == 0) TMATMUL(accum, left, right);
                else TMATMUL_ACC(accum, accum, left, right);
                set_flag(PIPE_M, PIPE_MTE1, inner_event);
            }
            if (wave + 2 < waves) {
                set_flag(PIPE_MTE1, PIPE_MTE2, event); wait_flag(PIPE_MTE1, PIPE_MTE2, event);
                load_weight(args, worker + (wave + 2) * workers, slot, weight_stride);
            }
            set_flag(PIPE_M, PIPE_FIX, acc_event); wait_flag(PIPE_M, PIPE_FIX, acc_event);
            wait_intra_block(PIPE_FIX, slot * 2 + 1);
            wait_intra_block(PIPE_FIX, slot * 2 + 17);
            Raw raw;
            TASSIGN(raw, slot * SlotBytes);
            TMOV<Raw, Acc, AccToVecMode::DualModeSplitM>(raw, accum);
            set_intra_block(PIPE_FIX, slot * 2);
            set_intra_block(PIPE_FIX, slot * 2 + 16);
            set_flag(PIPE_FIX, PIPE_M, acc_event);
        }
        for (int slot = 0; slot < 2; ++slot) {
            wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(slot));
            wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(slot + 4));
            wait_intra_block(PIPE_FIX, slot * 2 + 1);
            wait_intra_block(PIPE_FIX, slot * 2 + 17);
        }
        set_intra_block(PIPE_FIX, 4); set_intra_block(PIPE_FIX, 20);
        wait_intra_block(PIPE_FIX, 5); wait_intra_block(PIPE_FIX, 21);
        pipe_barrier(PIPE_ALL);
    }
}
#endif

#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void update(
    State::TileDType __out__ state, IndexState::TileDType __out__ indices,
    Raw::TileDType __in__ base, Raw::TileDType __in__ bias,
    uint32_t start, uint16_t rows, bool initialize) {
    auto state_ptr = (__ubuf__ float*)__cce_get_tile_ptr(state);
    auto index_ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto base_ptr = (__ubuf__ float*)__cce_get_tile_ptr(base);
    auto bias_ptr = (__ubuf__ float*)__cce_get_tile_ptr(bias);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> best;
            RegTensor<int32_t> best_id, next_id;
            vci(next_id, static_cast<int32_t>(start), INC_ORDER);
            if (initialize) {
                vdup((RegTensor<int32_t>&)best, -8388608, all, MODE_ZEROING);
                vmov(best_id, next_id, all);
            } else {
                vlds(best, state_ptr, row * 64, NORM);
                vlds(best_id, index_ptr, row * 64, NORM);
            }
            for (uint16_t segment = 0; segment < N / 64; ++segment) {
                RegTensor<float> x, y, score;
                vlds(x, base_ptr, row * N + segment * 64, NORM);
                vlds(y, bias_ptr, row * N + segment * 64, NORM);
                vadd(score, x, y, all, MODE_ZEROING);
                MaskReg better;
                vcmp_gt(better, score, best, all);
                vsel(best, score, best, better);
                vsel(best_id, next_id, best_id, better);
                vadds(next_id, next_id, 64, all, MODE_ZEROING);
            }
            vsts(best, state_ptr, row * 64, NORM_B32, all);
            vsts(best_id, index_ptr, row * 64, NORM_B32, all);
        }
    }
}

__tf__ __aicore__ inline void finish(
    Scores::TileDType __out__ score, Indices::TileDType __out__ index,
    State::TileDType __in__ state, IndexState::TileDType __in__ indices, uint16_t rows) {
    auto state_ptr = (__ubuf__ float*)__cce_get_tile_ptr(state);
    auto index_ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(indices);
    auto score_ptr = (__ubuf__ float*)__cce_get_tile_ptr(score);
    auto output_ptr = (__ubuf__ int32_t*)__cce_get_tile_ptr(index);
    __VEC_SCOPE__ {
        // The final update has no following MTE2-to-V return dependency.
        mem_bar(VST_VLD);
        MaskReg all = pset_b32(PAT_ALL);
        uint32_t count = 16;
        MaskReg output = CreatePredicate<float>(count);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> values, maximum;
            RegTensor<int32_t> ids, selected, invalid;
            vlds(values, state_ptr, row * 64, NORM);
            vlds(ids, index_ptr, row * 64, NORM);
            vcmax(maximum, values, all, MODE_ZEROING);
            vdup(maximum, maximum, all, POS_LOWEST, MODE_ZEROING);
            MaskReg ties;
            vcmp_eq(ties, values, maximum, all);
            vdup(invalid, 2147483647, all, MODE_ZEROING);
            vsel(selected, ids, invalid, ties);
            vcmin(selected, selected, all, MODE_ZEROING);
            vdup(selected, selected, all, POS_LOWEST, MODE_ZEROING);
            vsts(maximum, score_ptr, row * 16, NORM_B32, output);
            vsts(selected, output_ptr, row * 16, NORM_B32, output);
        }
    }
}

__aicore__ inline void vector(__gm__ int64_t* args, int step) {
    auto logits = descriptor(args, 0);
    int batch = logits->shapes[0], panels = logits->shapes[2] / N;
    int worker = get_block_idx(args), workers = get_block_num(args), lane = get_sub_block_id(args);
    int waves = (panels + workers - 1 - worker) / workers;
    int flag_base = 0;
    State state; IndexState indices; Scores result; Indices result_ids;
    TASSIGN(state, 32768); TASSIGN(indices, 34816);
    TASSIGN(result, 36864); TASSIGN(result_ids, 37376);
    for (int first_row = 0; first_row < batch; first_row += M) {
        int first = first_row + lane * (M / 2);
        int valid = batch - first; valid = valid < M / 2 ? valid : M / 2;
        valid = valid > 0 ? valid : 0;
        set_intra_block(PIPE_V, flag_base + 1); set_intra_block(PIPE_V, flag_base + 3);
        for (int wave = 0; wave < waves; ++wave) {
            int slot = wave % 2, column = (worker + wave * workers) * N;
            Raw raw, base;
            TASSIGN(raw, slot * SlotBytes); TASSIGN(base, 16384);
            if (valid > 0) {
                using Active = Tile<TileType::Vec, float, M / 2, N, BLayout::RowMajor, DYNAMIC, N>;
                using Shape = TileShape2D<float, DYNAMIC, N>;
                using Stride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
                Active input(valid); TASSIGN(input, 16384);
                GlobalTensor<float, Shape, Stride> input_gm(data<float>(args, 0) + first * logits->strides[0] + step * logits->strides[1] + column,
                    Shape(valid, N), Stride(1, 1, 1, logits->strides[0], 1));
                TLOAD(input, input_gm);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            }
            wait_intra_block(PIPE_V, flag_base + slot * 2);
            update(state.data(), indices.data(), base.data(), raw.data(), column, valid, wave == 0);
            set_intra_block(PIPE_V, flag_base + slot * 2 + 1);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        finish(result.data(), result_ids.data(), state.data(), indices.data(), valid);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (valid > 0) {
            using ActiveScore = Tile<TileType::Vec, float, M / 2, 16, BLayout::RowMajor, DYNAMIC, 16>;
            using ActiveIndex = Tile<TileType::Vec, int32_t, M / 2, 16, BLayout::RowMajor, DYNAMIC, 16>;
            using Stride = pto::Stride<1, 1, 1, Workers * 16, 1>;
            ActiveScore scores(valid); ActiveIndex ids(valid);
            TASSIGN(scores, 36864); TASSIGN(ids, 37376);
            GlobalTensor<float, TileShape2D<float, DYNAMIC, 16>, Stride> score_gm(data<float>(args, 3) + (first * Workers + worker) * 16, TileShape2D<float, DYNAMIC, 16>(valid, 16));
            GlobalTensor<int32_t, TileShape2D<int32_t, DYNAMIC, 16>, Stride> ids_gm(data<int32_t>(args, 4) + (first * Workers + worker) * 16, TileShape2D<int32_t, DYNAMIC, 16>(valid, 16));
            TSTORE(score_gm, scores); TSTORE(ids_gm, ids);
        }
        pipe_barrier(PIPE_ALL);
        SYNCALL<SyncCoreType::AIVOnly>();
        for (int local = worker * 2 + lane; local < M && first_row + local < batch; local += workers * 2) {
            int request = first_row + local;
            auto head = descriptor(args, 5);
            int token = markov_native::finalize_request(
                data<float>(args, 3) + request * Workers * 16,
                data<int32_t>(args, 4) + request * Workers * 16,
                data<bfloat16_t>(args, 5) + request * head->strides[0] + step * head->strides[1],
                data<bfloat16_t>(args, 1) + request * K,
                data<float>(args, 7), data<int64_t>(args, 8) + (request * 5 + step) * 16,
                data<float>(args, 9) + (request * 5 + step) * 16, workers);
            if (step + 1 < 5) {
                markov_native::Vec<bfloat16_t, K> next_embedding;
                TASSIGN(next_embedding, 45056);
                markov_native::GM<bfloat16_t, K> w1_gm(data<bfloat16_t>(args, 6) + static_cast<int64_t>(token) * descriptor(args, 6)->strides[0]);
                markov_native::GM<bfloat16_t, K> embedding_gm(data<bfloat16_t>(args, 1) + request * K);
                TLOAD(next_embedding, w1_gm);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
                TSTORE(embedding_gm, next_embedding);
            }
        }
        pipe_barrier(PIPE_ALL);
        SYNCALL<SyncCoreType::AIVOnly>();
        wait_intra_block(PIPE_V, flag_base + 4);
        set_intra_block(PIPE_V, flag_base + 5);
    }
}
#endif
} // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    native_lifecycle::enter_dual_aiv();
#if defined(__DAV_CUBE__)
    set_atomic_none();
    // This is embedding readiness, separate from the task-entry rendezvous.
    wait_intra_block(PIPE_FIX, 10); wait_intra_block(PIPE_FIX, 26);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6); wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID6);
    for (int step = 0; step < 5; ++step) cube(args);
    // The primary lane cannot retire before the packed public stores complete.
    wait_intra_block(PIPE_FIX, 6); wait_intra_block(PIPE_FIX, 22);
    pipe_barrier(PIPE_ALL);
#elif defined(__DAV_VEC__)
    set_atomic_none(); set_mask_norm(); set_vector_mask(-1, -1);
    int batch = descriptor(args, 0)->shapes[0];
    int lane = get_sub_block_id(args), worker = get_block_idx(args);
    int count = get_block_num(args) * 2;
    for (int request = worker * 2 + lane; request < batch; request += count) {
        auto anchor = data<int64_t>(args, 12) + request * descriptor(args, 12)->strides[0];
        dcci(reinterpret_cast<__gm__ void*>(anchor), cache_line_t::SINGLE_CACHE_LINE);
        int64_t token = *anchor;
        markov_native::Vec<bfloat16_t, K> initial;
        TASSIGN(initial, 45056);
        markov_native::GM<bfloat16_t, K> source(data<bfloat16_t>(args, 6) + token * descriptor(args, 6)->strides[0]);
        markov_native::GM<bfloat16_t, K> target(data<bfloat16_t>(args, 1) + request * K);
        TLOAD(initial, source);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID6);
        TSTORE(target, initial);
        pipe_barrier(PIPE_ALL);
    }
    SYNCALL<SyncCoreType::AIVOnly>();
    set_intra_block(PIPE_V, 10);
    for (int step = 0; step < 5; ++step) vector(args, step);
    if (worker == 0 && lane == 0) {
        markov_publish::publish(descriptor(args, 8), descriptor(args, 9), descriptor(args, 10), descriptor(args, 11));
    }
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_V, 6);
#endif
}
