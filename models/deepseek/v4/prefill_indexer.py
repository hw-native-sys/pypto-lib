# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill indexer scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from decode_indexer import *  # noqa: F401,F403
from decode_indexer import _int8_quant_per_row, _quant_w_per_output_channel
from prefill_indexer_compressor import prefill_indexer_compressor

B = 1
S = 128
T = B * S
START_POS = 0
OFFSET = S
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
PREFILL_CACHE_BLOCKS = (PREFILL_COMPRESSED_LEN + CACHE_TILE - 1) // CACHE_TILE
SCORE_B_GROUP = 1
PREFILL_Q_OUT_CHUCK = 128

@pl.jit.inline
def prefill_indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[S, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, PREFILL_COMPRESSED_LEN, INNER_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    start_pos: pl.Scalar[pl.INT32],
    offset: pl.Scalar[pl.INT32],
    inner_rotate: pl.Scalar[pl.BOOL],
):
    qr_i8 = qr
    qr_scale_dq = qr_scale

    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.BF16)
    for o0 in pl.parallel(0, IDX_N_HEADS * IDX_HEAD_DIM, PREFILL_Q_OUT_CHUCK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_proj"):
            qr_acc = pl.create_tensor([T, PREFILL_Q_OUT_CHUCK], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_CHUCK, stage=2):
                q0 = kb * Q_CHUCK
                qr_tile = qr_i8[:, q0 : q0 + Q_CHUCK]
                wq_tile = wq_b[q0 : q0 + Q_CHUCK, o0 : o0 + PREFILL_Q_OUT_CHUCK]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_proj_write"):
            qr_acc_fp32 = pl.cast(qr_acc, target_type=pl.FP32, mode="none")
            wq_scale = pl.reshape(wq_b_scale[o0 : o0 + PREFILL_Q_OUT_CHUCK], [1, PREFILL_Q_OUT_CHUCK])
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(qr_acc_fp32, qr_scale_dq), wq_scale)
            qr_proj[:, o0 : o0 + PREFILL_Q_OUT_CHUCK] = pl.cast(qr_dequant, target_type=pl.BF16, mode="rint")

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_rope_out = pl.create_tensor([T * IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.BF16)
    qr_hadamard_acc_g = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)

    for t in pl.parallel(T):
        row0 = t * IDX_N_HEADS
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_slice"):
            even_acc = pl.create_tensor([IDX_N_HEADS, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
            odd_acc = pl.create_tensor([IDX_N_HEADS, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
            for rb in pl.pipeline(0, ROPE_HEAD_DIM // ROPE_CHUCK, stage=2):
                r0 = rb * ROPE_CHUCK
                qr_proj_rope_tile = qr_proj_flat[row0 : row0 + IDX_N_HEADS, IDX_NOPE_HEAD_DIM + r0 : IDX_NOPE_HEAD_DIM + r0 + ROPE_CHUCK]
                even_select_tile = even_select[r0 : r0 + ROPE_CHUCK, :]
                odd_select_tile = odd_select[r0 : r0 + ROPE_CHUCK, :]
                if r0 == 0:
                    even_acc = pl.matmul(qr_proj_rope_tile, even_select_tile, out_dtype=pl.FP32)
                    odd_acc = pl.matmul(qr_proj_rope_tile, odd_select_tile, out_dtype=pl.FP32)
                else:
                    even_acc = pl.matmul_acc(even_acc, qr_proj_rope_tile, even_select_tile)
                    odd_acc = pl.matmul_acc(odd_acc, qr_proj_rope_tile, odd_select_tile)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_apply"):
            cos_t = cos[t : t + 1, :]
            sin_t = sin[t : t + 1, :]
            rope_even_acc = pl.cast(pl.sub(pl.col_expand_mul(even_acc, cos_t), pl.col_expand_mul(odd_acc, sin_t)), target_type=pl.BF16, mode="rint")
            rope_odd_acc = pl.cast(pl.add(pl.col_expand_mul(even_acc, sin_t), pl.col_expand_mul(odd_acc, cos_t)), target_type=pl.BF16, mode="rint")

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_assemble"):
            rope_acc = pl.create_tensor([IDX_N_HEADS, ROPE_HEAD_DIM], dtype=pl.FP32)
            for ra_b in pl.pipeline(0, (ROPE_HEAD_DIM // 2) // ROPE_CHUCK, stage=2):
                ra_0 = ra_b * ROPE_CHUCK
                rope_even_tile = rope_even_acc[:, ra_0 : ra_0 + ROPE_CHUCK]
                rope_odd_tile = rope_odd_acc[:, ra_0 : ra_0 + ROPE_CHUCK]
                even_select_tile_t = even_select[:, ra_0 : ra_0 + ROPE_CHUCK]
                odd_select_tile_t = odd_select[:, ra_0 : ra_0 + ROPE_CHUCK]
                if ra_0 == 0:
                    rope_acc = pl.matmul(rope_even_tile, even_select_tile_t, out_dtype=pl.FP32, b_trans=True)
                else:
                    rope_acc = pl.matmul_acc(rope_acc, rope_even_tile, even_select_tile_t, b_trans=True)
                rope_acc = pl.matmul_acc(rope_acc, rope_odd_tile, odd_select_tile_t, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_write"):
            qr_rope_out[row0 : row0 + IDX_N_HEADS, :] = pl.cast(rope_acc, target_type=pl.BF16, mode="rint")

    for o0 in pl.parallel(0, T * IDX_N_HEADS, HEAD_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_hadamard"):
            qh_nope = qr_proj_flat[o0 : o0 + HEAD_ROWS, 0 : IDX_NOPE_HEAD_DIM]
            qh_rope = qr_rope_out[o0 : o0 + HEAD_ROWS, :]
            qh_acc = pl.matmul(qh_nope, hadamard[0 : IDX_NOPE_HEAD_DIM, :], out_dtype=pl.FP32)
            qh_hadamard = pl.matmul_acc(qh_acc, qh_rope, hadamard[IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM, :])
            qr_hadamard_acc_g[o0 : o0 + HEAD_ROWS, :] = qh_hadamard

    for o0 in pl.parallel(0, T * IDX_N_HEADS, HEAD_ROWS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_hadamard_quant"):
            qh_amax = pl.full([1, HEAD_ROWS], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                qh_a_f32 = qr_hadamard_acc_g[o0 : o0 + HEAD_ROWS, h0 : h0 + HEAD_DIM_CHUCK]
                qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, HEAD_ROWS])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(pl.full([1, HEAD_ROWS], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax)
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [HEAD_ROWS, 1])
            qr_hadamard_scale_dq[o0 : o0 + HEAD_ROWS, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [HEAD_ROWS, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                qh_q_f32 = qr_hadamard_acc_g[o0 : o0 + HEAD_ROWS, h1 : h1 + HEAD_DIM_CHUCK]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
                qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
                qr_hadamard_i8[o0 : o0 + HEAD_ROWS, h1 : h1 + HEAD_DIM_CHUCK] = qh_i8

    x_flat = pl.reshape(x, [T, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_weights_proj"):
        weights_acc = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_CHUCK, stage=2):
            d0 = db * D_CHUCK
            x_tile = x_flat[:, d0 : d0 + D_CHUCK]
            weights_proj_tile = weights_proj[d0 : d0 + D_CHUCK, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, weights_proj_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile)

    weights_flat = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_weights_write"):
        weights = pl.mul(weights_acc, WEIGHTS_SCALE)
        weights_flat[:, :] = weights

    cmp_cos = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    for c in pl.range(PREFILL_COMPRESSED_LEN):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cmp_rope_gather"):
            src = c * COMPRESS_RATIO
            cmp_cos = pl.assemble(cmp_cos, cos[src : src + 1, :], [c, 0])
            cmp_sin = pl.assemble(cmp_sin, sin[src : src + 1, :], [c, 0])

    inner_kv, inner_kv_state, inner_score_state, idx_kv_cache = prefill_indexer_compressor(
        x,
        inner_kv,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        cmp_cos,
        cmp_sin,
        even_select,
        odd_select,
        hadamard,
        idx_kv_cache,
        start_pos,
        inner_rotate,
    )

    kv_cache_flat = pl.reshape(idx_kv_cache, [B * IDX_KV_LEN, IDX_HEAD_DIM])
    score_kv_scale = pl.create_tensor([B * PREFILL_CACHE_BLOCKS * CACHE_TILE, 1], dtype=pl.FP32)
    score_flat = pl.reshape(score, [T, SCORE_LEN])
    score_pre_relu_flat = pl.create_tensor([T * IDX_N_HEADS, PREFILL_COMPRESSED_LEN], dtype=pl.FP32)
    kv_tile_i8_g = pl.create_tensor([B * PREFILL_CACHE_BLOCKS * CACHE_TILE, IDX_HEAD_DIM], dtype=pl.INT8)
    score_acc_g = pl.create_tensor([B * PREFILL_CACHE_BLOCKS * S * CACHE_TILE, IDX_N_HEADS], dtype=pl.INT32)

    for t in pl.parallel(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_score_init"):
            score_flat[t : t + 1, :] = pl.full([1, SCORE_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    for bg in pl.parallel(0, B, SCORE_B_GROUP):
        for cb in pl.parallel(PREFILL_CACHE_BLOCKS):
            cache0 = cb * CACHE_TILE
            cache_block_len = pl.min(CACHE_TILE, PREFILL_COMPRESSED_LEN - cache0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_score_quant"):
                for bi in pl.range(SCORE_B_GROUP):
                    b = bg + bi
                    kv0 = b * IDX_KV_LEN
                    score_row0 = (b * PREFILL_CACHE_BLOCKS + cb) * CACHE_TILE
                    kv_amax = pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                        kv_a_tile = kv_cache_flat[kv0 + cache0 : kv0 + cache0 + CACHE_TILE, h0 : h0 + HEAD_DIM_CHUCK]
                        kv_a_f32 = pl.cast(kv_a_tile, target_type=pl.FP32)
                        kv_a_abs = pl.maximum(kv_a_f32, pl.neg(kv_a_f32))
                        kv_a_max = pl.reshape(pl.row_max(kv_a_abs), [1, CACHE_TILE])
                        kv_amax = pl.maximum(kv_amax, kv_a_max)
                    kv_scale_quant_row = pl.div(pl.full([1, CACHE_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
                    kv_cache_scale_dq_row = pl.recip(kv_scale_quant_row)
                    kv_cache_scale_dq = pl.reshape(kv_cache_scale_dq_row, [CACHE_TILE, 1])
                    kv_scale_quant = pl.reshape(kv_scale_quant_row, [CACHE_TILE, 1])
                    for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_CHUCK):
                        kv_q_tile = kv_cache_flat[kv0 + cache0 : kv0 + cache0 + CACHE_TILE, h1 : h1 + HEAD_DIM_CHUCK]
                        kv_q_f32 = pl.cast(kv_q_tile, target_type=pl.FP32)
                        kv_q_scaled = pl.row_expand_mul(kv_q_f32, kv_scale_quant)
                        kv_q_i32 = pl.cast(kv_q_scaled, target_type=pl.INT32, mode="rint")
                        kv_q_half = pl.cast(kv_q_i32, target_type=pl.FP16, mode="round")
                        kv_q_i8 = pl.cast(kv_q_half, target_type=pl.INT8, mode="trunc")
                        kv_tile_i8_g[score_row0 : score_row0 + CACHE_TILE, h1 : h1 + HEAD_DIM_CHUCK] = kv_q_i8
                    score_kv_scale[score_row0 : score_row0 + CACHE_TILE, :] = kv_cache_scale_dq

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_score_accum"):
                for bi in pl.range(SCORE_B_GROUP):
                    b = bg + bi
                    q0 = b * S * IDX_N_HEADS
                    score_row0 = (b * PREFILL_CACHE_BLOCKS + cb) * CACHE_TILE
                    for s in pl.range(S):
                        q_s0 = q0 + s * IDX_N_HEADS
                        acc_row0 = ((b * PREFILL_CACHE_BLOCKS + cb) * S + s) * CACHE_TILE
                        qr_hadamard_tile = qr_hadamard_i8[q_s0 : q_s0 + IDX_N_HEADS, :]
                        score_acc_tile = pl.matmul(kv_tile_i8_g[score_row0 : score_row0 + CACHE_TILE, :], qr_hadamard_tile, out_dtype=pl.INT32, b_trans=True)
                        score_acc_g[acc_row0 : acc_row0 + CACHE_TILE, :] = score_acc_tile

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_score_store"):
                for bi in pl.range(SCORE_B_GROUP):
                    b = bg + bi
                    t0 = b * S
                    q0 = b * S * IDX_N_HEADS
                    score_row0 = (b * PREFILL_CACHE_BLOCKS + cb) * CACHE_TILE
                    kv_cache_scale_dq = score_kv_scale[score_row0 : score_row0 + CACHE_TILE, :]

                    for s in pl.range(S):
                        q_s0 = q0 + s * IDX_N_HEADS
                        acc_row0 = ((b * PREFILL_CACHE_BLOCKS + cb) * S + s) * CACHE_TILE
                        qh_scale = pl.reshape(qr_hadamard_scale_dq[q_s0 : q_s0 + IDX_N_HEADS, :], [1, IDX_N_HEADS])
                        score_tile_s = pl.cast(score_acc_g[acc_row0 : acc_row0 + CACHE_TILE, :], target_type=pl.FP32, mode="none")
                        score_kv_scaled_s = pl.row_expand_mul(score_tile_s, kv_cache_scale_dq)
                        score_tile_s = pl.col_expand_mul(score_kv_scaled_s, qh_scale)
                        score_pre_relu_flat[q_s0 : q_s0 + IDX_N_HEADS, cache0 : cache0 + CACHE_TILE] = pl.transpose(score_tile_s, axis1=0, axis2=1)

    for bg in pl.parallel(0, B, SCORE_B_GROUP):
        for cb in pl.parallel(PREFILL_CACHE_BLOCKS):
            cache0 = cb * CACHE_TILE

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_score_reduce"):
                for bi in pl.range(SCORE_B_GROUP):
                    b = bg + bi
                    t0 = b * S
                    q0 = b * S * IDX_N_HEADS

                    for s in pl.range(S):
                        q_s0 = q0 + s * IDX_N_HEADS
                        score_reduce_s = pl.transpose(
                            score_pre_relu_flat[q_s0 : q_s0 + IDX_N_HEADS, cache0 : cache0 + CACHE_TILE],
                            axis1=0,
                            axis2=1,
                        )
                        relu_score_reduce_s = pl.maximum(score_reduce_s, pl.mul(score_reduce_s, 0.0))
                        weights_row_s = weights_flat[t0 + s : t0 + s + 1, :]
                        weighted_score_s_t = pl.col_expand_mul(relu_score_reduce_s, weights_row_s)
                        weighted_score_s = pl.reshape(pl.row_sum(weighted_score_s_t), [1, CACHE_TILE])
                        visible_len_s = (s + 1) // COMPRESS_RATIO
                        if visible_len_s > cache0:
                            valid_len_s = pl.min(CACHE_TILE, visible_len_s - cache0)
                        else:
                            valid_len_s = 0
                        weighted_score_valid_s = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len_s), pad_value=pl.PadValue.min)
                        weighted_score_valid_s = pl.maximum(
                            weighted_score_valid_s,
                            pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF),
                        )
                        score_flat[t0 + s : t0 + s + 1, cache0 : cache0 + CACHE_TILE] = weighted_score_valid_s

    topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])
    for t in pl.parallel(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_topk"):
            invalid_idxs = pl.full([1, SCORE_LEN], dtype=pl.INT32, value=-1)
            topk_idxs_flat[t : t + 1, :] = invalid_idxs
            visible_len = (t + 1) // COMPRESS_RATIO
            if visible_len > 0:
                offset_i32 = pl.cast(offset, target_type=pl.INT32)
                score_row = score_flat[t : t + 1, :]
                idx_init = pl.tensor.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
                sorted_score_tile = pl.tensor.sort32(score_row, idx_init)
                sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=64)
                sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=256)
                sorted_score_tile = pl.tensor.mrgsort(sorted_score_tile, block_len=1024)
                topk_pairs = sorted_score_tile[:, 0 : 2 * IDX_TOPK]
                topk_idxs_tile = pl.tensor.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                valid_topk = pl.min(IDX_TOPK, visible_len)
                topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                topk_idxs_flat[t : t + 1, 0 : IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, idx_kv_cache, topk_idxs

@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[S, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, PREFILL_COMPRESSED_LEN, INNER_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16]],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    start_pos: pl.Scalar[pl.INT32],
    offset: pl.Scalar[pl.INT32],
    inner_rotate: pl.Scalar[pl.BOOL],
):
    score, idx_kv_cache, topk_idxs = prefill_indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos,
        sin,
        even_select,
        odd_select,
        hadamard,
        inner_kv,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        score,
        topk_idxs,
        start_pos,
        offset,
        inner_rotate,
    )
    return score, idx_kv_cache, topk_idxs


def golden_prefill_indexer(tensors):
    """Torch reference for Indexer.forward prefill branch with kernel int8 score path."""
    import torch
    from prefill_indexer_compressor import golden_prefill_indexer_compressor

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    offset = int(tensors["offset"])

    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM
    cache_len = seqlen // ratio

    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(
        bsz, seqlen, IDX_N_HEADS, IDX_HEAD_DIM
    )

    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    q0, q1 = q_pair[..., 0], q_pair[..., 1]
    cos_q = cos[:seqlen].view(1, seqlen, 1, -1)
    sin_q = sin[:seqlen].view(1, seqlen, 1, -1)
    y0 = (q0 * cos_q - q1 * sin_q).to(torch.bfloat16)
    y1 = (q0 * sin_q + q1 * cos_q).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = q @ hadamard

    inner_tensors = {
        "x": tensors["x"],
        "kv": tensors["inner_kv"],
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"][::ratio],
        "sin": tensors["sin"][::ratio],
        "even_select": tensors["even_select"],
        "odd_select": tensors["odd_select"],
        "hadamard": tensors["hadamard"],
        "kv_cache": tensors["idx_kv_cache"],
        "start_pos": tensors["start_pos"],
        "rotate": tensors["inner_rotate"],
    }
    golden_prefill_indexer_compressor(inner_tensors)

    weights = (x @ weights_proj) * WEIGHTS_SCALE

    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)
    if cache_len == 0:
        tensors["score"][:] = score_full
        tensors["topk_idxs"][:] = topk_idxs
        return

    idx_kv_cache = tensors["idx_kv_cache"].float()
    kv_view = idx_kv_cache[:bsz, :cache_len]

    q_i8, q_scale = _int8_quant_per_row(q.reshape(bsz * seqlen * IDX_N_HEADS, IDX_HEAD_DIM))
    kv_i8, kv_scale = _int8_quant_per_row(kv_view.reshape(bsz * cache_len, IDX_HEAD_DIM))
    q_i8 = q_i8.view(bsz, seqlen, IDX_N_HEADS, IDX_HEAD_DIM)
    q_scale = q_scale.view(bsz, seqlen, IDX_N_HEADS, 1)
    kv_i8 = kv_i8.view(bsz, cache_len, IDX_HEAD_DIM)
    kv_scale = kv_scale.view(bsz, cache_len, 1)

    score_i32 = torch.einsum("bshd,btd->bsht", q_i8.to(torch.int32), kv_i8.to(torch.int32))
    score = score_i32.float() * q_scale * kv_scale.view(bsz, 1, 1, cache_len)
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=2)

    valid_counts = torch.arange(1, seqlen + 1, device=score.device).unsqueeze(1) // ratio
    score_pos = torch.arange(cache_len, device=score.device).unsqueeze(0)
    causal_mask = score_pos >= valid_counts
    score = score.masked_fill(causal_mask.view(1, seqlen, cache_len), FP32_NEG_INF)
    score_full[..., :cache_len] = score.to(torch.float32)
    tensors["score"][:] = score_full

    k = min(IDX_TOPK, cache_len)
    _, idx = score.topk(k, dim=-1)
    invalid_topk = idx >= valid_counts.view(1, seqlen, 1)
    idx = torch.where(invalid_topk, torch.full_like(idx, -1), idx.to(torch.int64) + offset)
    topk_idxs[..., :k] = idx.to(torch.int32)
    tensors["topk_idxs"][:] = topk_idxs

def build_tensor_specs(*args, **kwargs):
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.rand(B, S, D)
    def init_qr():
        return torch.rand(T, Q_LORA)
    def init_wq_b():
        return torch.rand(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM)
    def init_weights_proj():
        return torch.rand(D, IDX_N_HEADS)
    def init_cos():
        return torch.rand(S, ROPE_HEAD_DIM // 2)
    def init_sin():
        return torch.rand(S, ROPE_HEAD_DIM // 2)
    def init_odd_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i + 1, i] = 1
        return matrix
    def init_even_select():
        matrix = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            matrix[2 * i, i] = 1
        return matrix
    def init_hadamard():
        return torch.rand(IDX_HEAD_DIM, IDX_HEAD_DIM) * (IDX_HEAD_DIM ** -0.5)
    def init_inner_kv_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)
    def init_inner_score_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)
    def init_inner_wkv():
        return torch.rand(D, INNER_OUT_DIM)
    def init_inner_wgate():
        return torch.rand(D, INNER_OUT_DIM)
    def init_inner_ape():
        return torch.rand(COMPRESS_RATIO, INNER_OUT_DIM)
    def init_inner_norm_w():
        return torch.ones(INNER_HEAD_DIM)
    def init_idx_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, IDX_HEAD_DIM)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    qr_i8, qr_scale = _int8_quant_per_row(init_qr())
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_kv", [B, PREFILL_COMPRESSED_LEN, INNER_HEAD_DIM], torch.float32),
        TensorSpec("inner_kv_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_kv_state),
        TensorSpec("inner_score_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_score_state),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [B, IDX_KV_LEN, IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache, is_output=True),
        TensorSpec("score", [B, S, SCORE_LEN], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [B, S, SCORE_LEN], torch.int32, is_output=True),
        ScalarSpec("start_pos", torch.int32, START_POS),
        ScalarSpec("offset", torch.int32, OFFSET),
        ScalarSpec("inner_rotate", torch.bool, INNER_ROTATE),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        invalid_top = a_top < OFFSET
        a_orig = (a_top.long() - OFFSET).clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        paired = torch.where(invalid_top, torch.full_like(paired, -torch.inf), paired)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    result = run_jit(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_indexer,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "topk_idxs":    topk_idxs_compare,
            "idx_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / IDX_CACHE_BLOCK_NUM * BLOCK_SIZE),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
