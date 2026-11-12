// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// RMS normalization, latent output, rotary embedding, and compressed-cache publication.
#if defined(__CPU_SIM)
#include <algorithm>
#include <bit>
#include <cmath>
#endif
#include "compressor_publish.h"
namespace {
#if defined(__DAV_VEC__)
__tf__ __aicore__ inline void normalize(
    Batch<bfloat16_t, 512>::TileDType __out__ output,
    Batch<bfloat16_t, 512>::TileDType __out__ retained,
    Batch<bfloat16_t, 512>::TileDType __in__ input,
    Vec<bfloat16_t, 512>::TileDType __in__ weight, uint16_t rows) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(output);
    auto latent = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(retained);
    auto gamma = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(weight);
#if defined(__CPU_SIM)
    for (int row = 0; row < rows; ++row) {
        float square = 0;
        for (int col = 0; col < 512; ++col) {
            float value = static_cast<float>(src[row * 512 + col]);
            square += value * value;
        }
        float inv = 1.0f / std::sqrt(square / 512.0f + 1.0e-6f);
        for (int col = 0; col < 512; ++col) {
            dst[row * 512 + col] = static_cast<bfloat16_t>(static_cast<float>(src[row * 512 + col]) * inv * static_cast<float>(gamma[col]));
            latent[row * 512 + col] = dst[row * 512 + col];
        }
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<float> sum, value, square, inv, one;
            RegTensor<bfloat16_t> bf;
            vdup(sum, 0.0f, full, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                vlds(bf, src, row * 512 + chunk * 64, UNPK_B16);
                vcvt(value, bf, full, PART_EVEN);
                vmul(square, value, value, full, MODE_ZEROING);
                vadd(sum, sum, square, full, MODE_ZEROING);
            }
            vcadd(sum, sum, full, MODE_ZEROING);
            vmuls(inv, sum, 1.0f / 512.0f, full, MODE_ZEROING);
            vadds(inv, inv, 1.0e-6f, full, MODE_ZEROING);
            vsqrt(inv, inv, full, MODE_ZEROING);
            vdup(one, 1.0f, full, MODE_ZEROING);
            vdiv(inv, one, inv, full, MODE_ZEROING);
            vdup(inv, inv, full, POS_LOWEST, MODE_ZEROING);
            for (uint16_t chunk = 0; chunk < 8; ++chunk) {
                RegTensor<float> gain;
                vlds(bf, src, row * 512 + chunk * 64, UNPK_B16);
                vcvt(value, bf, full, PART_EVEN);
                vlds(bf, gamma, chunk * 64, UNPK_B16);
                vcvt(gain, bf, full, PART_EVEN);
                vmul(value, value, inv, full, MODE_ZEROING);
                vmul(value, value, gain, full, MODE_ZEROING);
                vcvt(bf, value, full, ROUND_R, RS_DISABLE, PART_EVEN);
                vsts(bf, dst, row * 512 + chunk * 64, PK_B32, full);
                vsts(bf, latent, row * 512 + chunk * 64, PK_B32, full);
            }
        }
    }
#endif
}

using Slots = Tile<TileType::Vec, int64_t, 1, 1024, BLayout::RowMajor, 1, DYNAMIC>;

// Keep a single owner for every cache slot, but compact its token indices in
// vector registers instead of inspecting all metadata with scalar modulo.
__tf__ __aicore__ inline void select_tokens(
    Vec<int32_t, 1024>::TileDType __out__ selected,
    Vec<int32_t, 32>::TileDType __out__ counts,
    Slots::TileDType __in__ slots, uint16_t count, int32_t base, int32_t worker, int32_t owner_mask) {
    auto metadata = (__ubuf__ int32_t*)__cce_get_tile_ptr(slots);
    auto output = (__ubuf__ int32_t*)__cce_get_tile_ptr(selected);
    auto sizes = (__ubuf__ int32_t*)__cce_get_tile_ptr(counts);
#if defined(__CPU_SIM)
    auto values = (__ubuf__ int64_t*)metadata;
    for (int chunk = 0; chunk < (count + 31) / 32; ++chunk) {
        int n = 0;
        for (int lane = 0; lane < 32 && chunk * 32 + lane < count; ++lane) {
            int token = base + chunk * 32 + lane;
            int64_t slot = values[chunk * 32 + lane];
            uint64_t slot_bits = static_cast<uint64_t>(slot);
            int owner = slot < 0 ? owner_mask ^ ((token / 2) & owner_mask) : (slot_bits ^ (slot_bits >> 7)) & owner_mask;
            if (owner == worker) output[chunk * 32 + n++] = token;
        }
        sizes[chunk] = n;
    }
#else
    __VEC_SCOPE__ {
        MaskReg full = pset_b32(PAT_ALL);
        RegTensor<int32_t> bits, low, high, indices, negative_owner, owner, mask, one, sizes_reg, compact;
        vdup(mask, owner_mask, full, MODE_ZEROING);
        vdup(one, 1, full, MODE_ZEROING);
        for (uint16_t chunk = 0; chunk < (count + 31) / 32; ++chunk) {
            uint32_t valid = count - chunk * 32 < 32 ? count - chunk * 32 : 32;
            MaskReg active = plt_b32(valid, POST_UPDATE);
            vlds(bits, metadata, chunk * 64, NORM);
            vdintlv(low, high, bits, bits);
            vci(indices, base + chunk * 32, INC_ORDER);
            vshrs(negative_owner, indices, 1, active, MODE_ZEROING);
            vand(negative_owner, negative_owner, mask, active, MODE_ZEROING);
            vxor(negative_owner, negative_owner, mask, active, MODE_ZEROING);
            vshrs(owner, low, 7, active, MODE_ZEROING);
            vxor(owner, owner, low, active, MODE_ZEROING);
            vand(owner, owner, mask, active, MODE_ZEROING);
            MaskReg negative, owned;
            vcmps_lt(negative, high, 0, active);
            vsel(owner, negative_owner, owner, negative);
            vcmps_eq(owned, owner, worker, active);
            vsqz(compact, indices, owned, MODE_NO_STORED);
            vsts(compact, output, chunk * 32, NORM_B32, active);
            vdup(sizes_reg, 0, full, MODE_ZEROING);
            vsel(sizes_reg, one, sizes_reg, owned);
            vcadd(sizes_reg, sizes_reg, full, MODE_ZEROING);
            vsts(sizes_reg, sizes, chunk, ONEPT_B32, full);
        }
    }
#endif
}

template <bool RoundEven>
__aicore__ inline void flush(__gm__ int64_t* args, int rows, Vec<int64_t, 8>& targets, Vec<int32_t, 8>& tokens) {
    Batch<bfloat16_t, 512> input, retained;
    Vec<bfloat16_t, 512> weight;
    Batch<float, 32> cosine, sine, scratch;
    Batch<uint8_t, 320> packed;
    TASSIGN(input, 0);
    TASSIGN(retained, 8192);
    TASSIGN(packed, 16384);
    TASSIGN(scratch, 18944);
    TASSIGN(cosine, 20032);
    TASSIGN(sine, 21056);
    TASSIGN(weight, 22080);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    normalize(input.data(), retained.data(), input.data(), weight.data(), rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    for (int row = 0; row < rows; ++row) {
        Vec<bfloat16_t, 512> value;
        TASSIGN(value, 8192 + row * 1024);
        GM<bfloat16_t, 512> latent_gm(data<bfloat16_t>(args, 6) + (int64_t)tokens.GetValue(row) * 512);
        TSTORE(latent_gm, value);
    }
    // RoPE and quantization use a different UB region from latent writeback.
    rotate(input.data(), input.data(), cosine.data(), sine.data(), rows);
    quantize<RoundEven>(packed.data(), input.data(), scratch.data(), rows);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    int64_t cache_stride = reinterpret_cast<__gm__ Tensor*>(args[2])->strides[1];
    for (int row = 0; row < rows; ++row) {
        int64_t slot = targets.GetValue(row);
        if (slot < 0) continue;
        Vec<uint8_t, 320> payload;
        TASSIGN(payload, 16384 + row * 320);
        GM<uint8_t, 320> output_gm(data<uint8_t>(args, 2) + slot * cache_stride);
        // Drain earlier writes before another write to the same 64-bit slot.
        for (int previous = 0; previous < row; ++previous) {
            if (targets.GetValue(previous) == slot) {
                pipe_barrier(PIPE_MTE3);
                break;
            }
        }
        TSTORE(output_gm, payload);
    }
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void run(__gm__ int64_t* args) {
    using SlotGM = GlobalTensor<int64_t, TileShape2D<int64_t, 1, DYNAMIC>, BaseShape2D<int64_t, 1, 1024>>;
    Slots slots(1024);
    Vec<int32_t, 1024> selected;
    Vec<int32_t, 32> counts;
    Vec<int64_t, 8> targets;
    Vec<int32_t, 8> tokens;
    Vec<bfloat16_t, 512> weight;
    TASSIGN(slots, 32768);
    TASSIGN(selected, 40960);
    TASSIGN(counts, 45056);
    TASSIGN(targets, 23104);
    TASSIGN(tokens, 23168);
    TASSIGN(weight, 22080);
    GM<bfloat16_t, 512> weight_gm(data<bfloat16_t>(args, 5));
    TLOAD(weight, weight_gm);
    int count_tokens = args[7];
    int worker = get_block_idx(args);
    int pending = 0;
    for (int base = 0; base < count_tokens; base += 1024) {
        int count = count_tokens - base < 1024 ? count_tokens - base : 1024;
        slots.SetValidCol(count);
        SlotGM slot_gm(data<int64_t>(args, 1) + base, TileShape2D<int64_t, 1, DYNAMIC>(1, count));
        TLOAD(slots, slot_gm);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        select_tokens(selected.data(), counts.data(), slots.data(), count, base, worker, get_block_num(args) - 1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        for (int chunk = 0; chunk < (count + 31) / 32; ++chunk) {
            int owned = counts.GetValue(chunk);
            for (int lane = 0; lane < owned; ++lane) {
                int token = selected.GetValue(chunk * 32 + lane);
                int64_t slot = slots.GetValue(token - base);
                targets.SetValue(pending, slot);
                tokens.SetValue(pending, token);
                Vec<bfloat16_t, 512> input;
                Vec<float, 32> cosine, sine;
                TASSIGN(input, pending * 1024);
                TASSIGN(cosine, 20032 + pending * 128);
                TASSIGN(sine, 21056 + pending * 128);
                GM<bfloat16_t, 512> input_gm(data<bfloat16_t>(args, 0) + (int64_t)token * 512);
                GM<float, 32> cos_gm(data<float>(args, 3) + (int64_t)token * 32);
                GM<float, 32> sin_gm(data<float>(args, 4) + (int64_t)token * 32);
                TLOAD(input, input_gm);
                TLOAD(cosine, cos_gm);
                TLOAD(sine, sin_gm);
                if (++pending == 8) {
                    if (args[8]) flush<true>(args, pending, targets, tokens);
                    else flush<false>(args, pending, targets, tokens);
                    pending = 0;
                }
            }
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
    if (pending > 0) {
        if (args[8]) flush<true>(args, pending, targets, tokens);
        else flush<false>(args, pending, targets, tokens);
    } else {
        pipe_barrier(PIPE_ALL);
    }
}

#endif
} // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    run(args);
#endif
}
