# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B A8W8 full-layer prefill forward."""

import pypto.language as pl

from config import (
    QWEN3_14B_DIMS as D,
    QWEN3_14B_TILING as T,
    QWEN3_14B as M,
)

USER_BATCH_DYN = D.user_batch
KV_CACHE_ROWS_DYN = D.kv_cache_rows
BLOCK_TABLE_FLAT_DYN = D.block_table_flat
LAYER_DYN = D.layer
LAYER_HIDDEN_ROWS_DYN = D.layer_hidden_rows
LAYER_INTER_ROWS_DYN = D.layer_inter_rows
PREFILL_TOKENS_DYN = pl.dynamic("PREFILL_TOKENS_DYN")

MAX_SEQ = M.max_seq
NUM_HEADS = M.num_heads
NUM_KV_HEADS = M.num_kv_heads
HEAD_DIM = M.head_dim
HIDDEN = M.hidden
INTERMEDIATE = M.intermediate
KV_HIDDEN = M.kv_hidden
VOCAB = M.vocab
EPS = M.eps
HIDDEN_INV = M.hidden_inv
HEAD_DIM_INV = M.head_dim_inv
ATTN_SCALE = M.attn_scale
HALF_DIM = M.half_dim
Q_PER_KV = M.q_per_kv
SEQ_TILE = T.seq_tile
BLOCK_SIZE = T.block_size
Q_HEAD_BATCH = M.q_head_batch
Q_HEAD_PAD = M.q_head_pad
Q_GROUPS = M.q_groups
TOTAL_Q_GROUPS = M.total_q_groups
INT8_SCALE_MAX = M.int8_scale_max
INT8_AMAX_EPS = M.int8_amax_eps

K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 32
SB_BATCH = 32
MLP_OUT_CHUNK = 128
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK

@pl.jit.inline(auto_scope=False)
def prefill_layer(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.INT8],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.INT8],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.INT8],
    wq_scale: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wk_scale: pl.Tensor[[LAYER_DYN, KV_HIDDEN], pl.FP32],
    wv_scale: pl.Tensor[[LAYER_DYN, KV_HIDDEN], pl.FP32],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.INT8],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.INT8],
    k_cache_scale: pl.Tensor[[KV_CACHE_ROWS_DYN, 8], pl.FP32],
    v_cache_scale: pl.Tensor[[KV_CACHE_ROWS_DYN, 8], pl.FP32],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.INT8],
    wo_scale: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    out: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    out.bind_dynamic(0, PREFILL_TOKENS_DYN)

    user_batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * layer_cache_rows
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch
    q_norm_w = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
    k_norm_w = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
    for b in pl.parallel(0, user_batch, 1):
        token_base = pl.cast(pl.tensor.read(chunk_offsets, [b]), pl.INDEX)
        seq_len_b = pl.tensor.read(seq_lens, [b])
        chunk_len_b = pl.tensor.read(chunk_lens, [b])
        chunk_start = seq_len_b - chunk_len_b
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        for p0_idx in pl.range(tok_blocks):
            with pl.scope():
                p0 = p0_idx * TOK_TILE
                token_p0 = token_base + p0
                valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)

                normed_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                    partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        rms_sum_k0 = kb * K_CHUNK
                        rms_sum_chunk = pl.cast(
                            pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [token_p0, rms_sum_k0],
                                     valid_shape=[valid_tok, K_CHUNK]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(rms_sum_chunk, rms_sum_chunk)), [1, TOK_TILE]),
                        )
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                        [TOK_TILE, 1],
                    )
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.range(HIDDEN_BLOCKS):
                        rms_norm_k0 = kb * K_CHUNK
                        rms_norm_chunk = pl.cast(
                            pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [token_p0, rms_norm_k0],
                                     valid_shape=[valid_tok, K_CHUNK]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [layer_idx, rms_norm_k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(rms_norm_chunk, inv_rms), gamma)
                        normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, rms_norm_k0])

                normed_i8 = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.INT8)
                act_scales = pl.create_tensor([TOK_TILE], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="normed_act_quant"):
                    for act_scale_ti in pl.range(valid_tok):
                        act_amax_value = INT8_AMAX_EPS
                        for kb in pl.range(HIDDEN_BLOCKS):
                            quant_amax_k0 = kb * K_CHUNK
                            quant_amax_row_bf16 = pl.slice(
                                normed_tile, [1, K_CHUNK], [act_scale_ti, quant_amax_k0]
                            )
                            quant_amax_row_f = pl.cast(quant_amax_row_bf16, target_type=pl.FP32)
                            quant_amax_row_abs = pl.maximum(quant_amax_row_f, pl.neg(quant_amax_row_f))
                            for hd in pl.range(K_CHUNK):
                                act_amax_value = pl.max(
                                    act_amax_value,
                                    pl.tensor.read(quant_amax_row_abs, [0, hd]),
                                )
                        act_scale_q_value = INT8_SCALE_MAX / act_amax_value
                        pl.tensor.write(act_scales, [act_scale_ti], act_amax_value / INT8_SCALE_MAX)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            quant_write_k0 = kb * K_CHUNK
                            quant_write_row_bf16 = pl.slice(
                                normed_tile, [1, K_CHUNK], [act_scale_ti, quant_write_k0]
                            )
                            quant_write_row_f = pl.cast(quant_write_row_bf16, target_type=pl.FP32)
                            scaled_row = pl.mul(quant_write_row_f, act_scale_q_value)
                            i32_row = pl.cast(scaled_row, target_type=pl.INT32, mode="rint")
                            i32_row = pl.minimum(
                                pl.maximum(i32_row, pl.full([1, K_CHUNK], dtype=pl.INT32, value=-127)),
                                pl.full([1, K_CHUNK], dtype=pl.INT32, value=127),
                            )
                            i16_row = pl.cast(i32_row, target_type=pl.FP16, mode="round")
                            i8_row = pl.cast(i16_row, target_type=pl.INT8, mode="trunc")
                            normed_i8 = pl.assemble(normed_i8, i8_row, [act_scale_ti, quant_write_k0])

                q_proj_i32 = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.INT32)
                q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                for ob_chunk in pl.parallel(0, Q_OUT_BLOCKS, 4):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj_matmul"):
                        for ob in pl.range(ob_chunk, ob_chunk + 4):
                            q0 = ob * Q_OUT_CHUNK
                            q_tile_a0 = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, 0])
                            q_tile_w0 = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q0])
                            q_acc = pl.matmul(q_tile_a0, q_tile_w0, out_dtype=pl.INT32)
                            for kb in pl.range(1, HIDDEN_BLOCKS):
                                q_k0 = kb * K_CHUNK
                                q_tile_a_i = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, q_k0])
                                q_tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q0])
                                q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_w_i)
                            q_proj_i32 = pl.assemble(q_proj_i32, q_acc, [0, q0])
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj_dequant"):
                        for ob in pl.range(ob_chunk, ob_chunk + 4):
                            q_deq_o0 = ob * Q_OUT_CHUNK
                            q_w_scale_col = pl.reshape(
                                pl.slice(wq_scale, [1, Q_OUT_CHUNK], [layer_idx, q_deq_o0]),
                                [1, Q_OUT_CHUNK],
                            )
                            q_deq_acc = pl.slice(q_proj_i32, [TOK_TILE, Q_OUT_CHUNK], [0, q_deq_o0])
                            q_deq_weighted = pl.col_expand_mul(
                                pl.cast(q_deq_acc, target_type=pl.FP32),
                                q_w_scale_col,
                            )
                            q_deq = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                            for q_deq_ti in pl.range(TOK_TILE):
                                q_deq_row = pl.slice(q_deq_weighted, [1, Q_OUT_CHUNK], [q_deq_ti, 0])
                                q_deq_scale = pl.read(act_scales, q_deq_ti)
                                q_deq = pl.assemble(q_deq, pl.mul(q_deq_row, q_deq_scale), [q_deq_ti, 0])
                            q_proj_tile = pl.assemble(q_proj_tile, q_deq, [0, q_deq_o0])

                k_proj_i32 = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.INT32)
                v_proj_i32 = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.INT32)
                k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                for ob_chunk in pl.parallel(0, KV_OUT_BLOCKS, 4):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_matmul"):
                        for ob in pl.range(ob_chunk, ob_chunk + 4):
                            kv0 = ob * KV_OUT_CHUNK

                            k_proj_tile_a0 = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, 0])
                            k_proj_tile_w0 = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                            k_acc = pl.matmul(k_proj_tile_a0, k_proj_tile_w0, out_dtype=pl.INT32)
                            for kb in pl.range(1, HIDDEN_BLOCKS):
                                k_proj_k0 = kb * K_CHUNK
                                k_proj_tile_a_i = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, k_proj_k0])
                                k_proj_tile_w_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_proj_k0, kv0])
                                k_acc = pl.matmul_acc(k_acc, k_proj_tile_a_i, k_proj_tile_w_i)
                            k_proj_i32 = pl.assemble(k_proj_i32, k_acc, [0, kv0])

                            v_proj_tile_a0 = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, 0])
                            v_proj_tile_w0 = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                            v_acc = pl.matmul(v_proj_tile_a0, v_proj_tile_w0, out_dtype=pl.INT32)
                            for kb in pl.range(1, HIDDEN_BLOCKS):
                                v_proj_k0 = kb * K_CHUNK
                                v_proj_tile_a_i = pl.slice(normed_i8, [TOK_TILE, K_CHUNK], [0, v_proj_k0])
                                v_proj_tile_w_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_proj_k0, kv0])
                                v_acc = pl.matmul_acc(v_acc, v_proj_tile_a_i, v_proj_tile_w_i)
                            v_proj_i32 = pl.assemble(v_proj_i32, v_acc, [0, kv0])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_dequant"):
                        for ob in pl.range(ob_chunk, ob_chunk + 4):
                            kv_deq_o0 = ob * KV_OUT_CHUNK
                            k_deq_acc = pl.slice(k_proj_i32, [TOK_TILE, KV_OUT_CHUNK], [0, kv_deq_o0])
                            wk_scale_col = pl.reshape(
                                pl.slice(wk_scale, [1, KV_OUT_CHUNK], [layer_idx, kv_deq_o0]),
                                [1, KV_OUT_CHUNK],
                            )
                            k_deq_weighted = pl.col_expand_mul(
                                pl.cast(k_deq_acc, target_type=pl.FP32),
                                wk_scale_col,
                            )
                            k_deq = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            for k_deq_ti in pl.range(TOK_TILE):
                                k_deq_row = pl.slice(k_deq_weighted, [1, KV_OUT_CHUNK], [k_deq_ti, 0])
                                k_deq_scale = pl.read(act_scales, k_deq_ti)
                                k_deq = pl.assemble(k_deq, pl.mul(k_deq_row, k_deq_scale), [k_deq_ti, 0])
                            k_proj_tile = pl.assemble(k_proj_tile, k_deq, [0, kv_deq_o0])

                            v_deq_acc = pl.slice(v_proj_i32, [TOK_TILE, KV_OUT_CHUNK], [0, kv_deq_o0])
                            wv_scale_col = pl.reshape(
                                pl.slice(wv_scale, [1, KV_OUT_CHUNK], [layer_idx, kv_deq_o0]),
                                [1, KV_OUT_CHUNK],
                            )
                            v_deq_weighted = pl.col_expand_mul(
                                pl.cast(v_deq_acc, target_type=pl.FP32),
                                wv_scale_col,
                            )
                            v_deq = pl.create_tensor([TOK_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                            for v_deq_ti in pl.range(TOK_TILE):
                                v_deq_row = pl.slice(v_deq_weighted, [1, KV_OUT_CHUNK], [v_deq_ti, 0])
                                v_deq_scale = pl.read(act_scales, v_deq_ti)
                                v_deq = pl.assemble(v_deq, pl.mul(v_deq_row, v_deq_scale), [v_deq_ti, 0])
                            v_proj_tile = pl.assemble(v_proj_tile, v_deq, [0, kv_deq_o0])

                for qh_chunk in pl.parallel(0, NUM_HEADS, NUM_HEADS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_norm"):
                        for qh in pl.range(qh_chunk, qh_chunk + NUM_HEADS):
                            q_col = qh * HEAD_DIM
                            q_head = pl.slice(q_proj_tile, [TOK_TILE, HEAD_DIM], [0, q_col])
                            q_sq = pl.reshape(
                                pl.row_sum(pl.mul(q_head, q_head)),
                                [TOK_TILE, 1],
                            )
                            q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS)))
                            q_normed = pl.col_expand_mul(
                                pl.row_expand_mul(q_head, q_inv_rms),
                                q_norm_w,
                            )
                            q_proj_tile = pl.assemble(q_proj_tile, q_normed, [0, q_col])
                for kh_chunk in pl.parallel(0, NUM_KV_HEADS, NUM_KV_HEADS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_norm"):
                        for kh in pl.range(kh_chunk, kh_chunk + NUM_KV_HEADS):
                            k_col = kh * HEAD_DIM
                            k_head = pl.slice(k_proj_tile, [TOK_TILE, HEAD_DIM], [0, k_col])
                            k_sq = pl.reshape(
                                pl.row_sum(pl.mul(k_head, k_head)),
                                [TOK_TILE, 1],
                            )
                            k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS)))
                            k_normed = pl.col_expand_mul(
                                pl.row_expand_mul(k_head, k_inv_rms),
                                k_norm_w,
                            )
                            k_proj_tile = pl.assemble(k_proj_tile, k_normed, [0, k_col])

                attn_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                for ti in pl.range(valid_tok):
                    with pl.scope():
                        chunk_pos = p0 + ti
                        pos = chunk_start + chunk_pos
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
                        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
                        cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
                        cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
                        sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
                        sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

                        all_q_padded = pl.create_tensor([TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_pad"):
                            for gi in pl.unroll(TOTAL_Q_GROUPS):
                                all_q_padded = pl.assemble(
                                    all_q_padded, pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16), [gi * Q_HEAD_PAD + Q_HEAD_BATCH, 0]
                                )
                        cache_slot_raw = pl.tensor.read(slot_mapping, [token_base + chunk_pos])
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                            for ki in pl.unroll(NUM_KV_HEADS):
                                    kv_col = ki * HEAD_DIM
                                    k_lo = pl.slice(k_proj_tile, [1, HALF_DIM], [ti, kv_col])
                                    k_hi = pl.slice(k_proj_tile, [1, HALF_DIM], [ti, kv_col + HALF_DIM])
                                    rot_lo = pl.sub(
                                        pl.col_expand_mul(k_lo, cos_lo),
                                        pl.col_expand_mul(k_hi, sin_lo),
                                    )
                                    rot_hi = pl.add(
                                        pl.col_expand_mul(k_hi, cos_hi),
                                        pl.col_expand_mul(k_lo, sin_hi),
                                    )
                                    rot_abs_lo = pl.maximum(rot_lo, pl.neg(rot_lo))
                                    rot_abs_hi = pl.maximum(rot_hi, pl.neg(rot_hi))
                                    rot_abs_lo_groups = pl.reshape(rot_abs_lo, [8, 8])
                                    rot_abs_hi_groups = pl.reshape(rot_abs_hi, [8, 8])
                                    k_amax_lo_parts = pl.row_max(rot_abs_lo_groups)
                                    k_amax_hi_parts = pl.row_max(rot_abs_hi_groups)
                                    k_amax_scalar = INT8_AMAX_EPS
                                    for part in pl.range(8):
                                        k_amax_scalar = pl.max(k_amax_scalar, pl.tensor.read(k_amax_lo_parts, [part, 0]))
                                        k_amax_scalar = pl.max(k_amax_scalar, pl.tensor.read(k_amax_hi_parts, [part, 0]))
                                    k_scale_q_value = INT8_SCALE_MAX / k_amax_scalar
                                    kq_lo_i32 = pl.cast(pl.mul(rot_lo, k_scale_q_value), target_type=pl.INT32, mode="rint")
                                    kq_hi_i32 = pl.cast(pl.mul(rot_hi, k_scale_q_value), target_type=pl.INT32, mode="rint")
                                    kq_lo_i32 = pl.minimum(
                                        pl.maximum(kq_lo_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                                        pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
                                    )
                                    kq_hi_i32 = pl.minimum(
                                        pl.maximum(kq_hi_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                                        pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
                                    )
                                    kq_lo = pl.cast(pl.cast(kq_lo_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                                    kq_hi = pl.cast(pl.cast(kq_hi_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                                    k_scale_value = k_amax_scalar / INT8_SCALE_MAX
                                    k_scale_out = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), k_scale_value)
                                    v_row_f = pl.slice(v_proj_tile, [1, HEAD_DIM], [ti, ki * HEAD_DIM])
                                    v_abs = pl.maximum(v_row_f, pl.neg(v_row_f))
                                    v_abs_groups = pl.reshape(v_abs, [16, 8])
                                    v_amax_parts = pl.row_max(v_abs_groups)
                                    v_amax_scalar = INT8_AMAX_EPS
                                    for part in pl.range(16):
                                        v_amax_scalar = pl.max(v_amax_scalar, pl.tensor.read(v_amax_parts, [part, 0]))
                                    v_scale_q_value = INT8_SCALE_MAX / v_amax_scalar
                                    vq_i32 = pl.cast(pl.mul(v_row_f, v_scale_q_value), target_type=pl.INT32, mode="rint")
                                    vq_i32 = pl.minimum(
                                        pl.maximum(vq_i32, pl.full([1, HEAD_DIM], dtype=pl.INT32, value=-127)),
                                        pl.full([1, HEAD_DIM], dtype=pl.INT32, value=127),
                                    )
                                    vq = pl.cast(pl.cast(vq_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
                                    v_scale_value = v_amax_scalar / INT8_SCALE_MAX
                                    v_scale_out = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), v_scale_value)
                                    if cache_slot_raw != -1:
                                        cache_slot = pl.cast(cache_slot_raw, pl.INDEX)
                                        cache_slot_block = cache_slot // BLOCK_SIZE
                                        cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE
                                        cache_row = (
                                            layer_cache_base
                                            + (cache_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE
                                            + cache_slot_offset
                                        )
                                        k_cache = pl.assemble(k_cache, kq_lo, [cache_row, 0])
                                        k_cache = pl.assemble(k_cache, kq_hi, [cache_row, HALF_DIM])
                                        k_cache_scale = pl.assemble(k_cache_scale, k_scale_out, [cache_row, 0])
                                        v_cache = pl.assemble(v_cache, vq, [cache_row, 0])
                                        v_cache_scale = pl.assemble(v_cache_scale, v_scale_out, [cache_row, 0])
                                    q_base = ki * Q_PER_KV
                                    for qi in pl.unroll(Q_HEAD_BATCH):
                                        q_col = (q_base + qi) * HEAD_DIM
                                        q_lo = pl.slice(q_proj_tile, [1, HALF_DIM], [ti, q_col])
                                        q_hi = pl.slice(q_proj_tile, [1, HALF_DIM], [ti, q_col + HALF_DIM])
                                        rot_lo_bf16 = pl.cast(
                                            pl.sub(
                                                pl.col_expand_mul(q_lo, cos_lo),
                                                pl.col_expand_mul(q_hi, sin_lo),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        rot_hi_bf16 = pl.cast(
                                            pl.add(
                                                pl.col_expand_mul(q_hi, cos_hi),
                                                pl.col_expand_mul(q_lo, sin_hi),
                                            ),
                                            target_type=pl.BF16,
                                        )
                                        all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [ki * Q_HEAD_PAD + qi, 0])
                                        all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [ki * Q_HEAD_PAD + qi, HALF_DIM])

                        attn_row = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
                        for gi in pl.unroll(TOTAL_Q_GROUPS):
                            kvh = gi // Q_GROUPS
                            qg = gi - kvh * Q_GROUPS
                            q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH

                            q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [gi * Q_HEAD_PAD, 0])

                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax_init"):
                                oi = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
                                li_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
                                li = pl.reshape(li_flat, [Q_HEAD_PAD, 1])
                                mi_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
                                mi = pl.reshape(mi_flat, [Q_HEAD_PAD, 1])

                            for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                                chunk_k_tiles = pl.create_tensor([SB_BATCH * SEQ_TILE, HEAD_DIM], dtype=pl.BF16)
                                chunk_v_tiles = pl.create_tensor([SB_BATCH * SEQ_TILE, HEAD_DIM], dtype=pl.BF16)
                                chunk_raw_scores = pl.create_tensor([SB_BATCH * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                                chunk_exp_padded = pl.create_tensor([SB_BATCH * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                                chunk_oi_tmp = pl.create_tensor([SB_BATCH * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                                chunk_cur_mi = pl.create_tensor([SB_BATCH * Q_HEAD_PAD, 1], dtype=pl.FP32)
                                chunk_cur_li = pl.create_tensor([SB_BATCH * Q_HEAD_PAD, 1], dtype=pl.FP32)

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_dequant"):
                                    for qk_load_lane in pl.range(SB_BATCH):
                                        qk_load_sb = sb0 + qk_load_lane
                                        if qk_load_sb < ctx_blocks:
                                            qk_block_table_idx = b * max_blocks_per_seq + qk_load_sb
                                            qk_pbid = pl.cast(pl.tensor.read(block_table, [qk_block_table_idx]), pl.INDEX)
                                            qk_cache_row0 = layer_cache_base + (qk_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                                            qk_tile_i8 = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [qk_cache_row0, 0])
                                            qk_tile_bf16 = pl.cast(
                                                pl.full([SEQ_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0),
                                                target_type=pl.BF16,
                                            )
                                            qk_valid_len = pl.min(SEQ_TILE, ctx_len - qk_load_sb * SEQ_TILE)
                                            for qk_scale_ti in pl.range(qk_valid_len):
                                                qk_scale_row = qk_cache_row0 + qk_scale_ti
                                                qk_scale = pl.tensor.read(k_cache_scale, [qk_scale_row, 0])
                                                qk_row_i8 = pl.slice(qk_tile_i8, [1, HEAD_DIM], [qk_scale_ti, 0])
                                                qk_row_fp32 = pl.cast(pl.cast(qk_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                                                qk_tile_bf16 = pl.assemble(qk_tile_bf16, pl.cast(pl.mul(qk_row_fp32, qk_scale), target_type=pl.BF16), [qk_scale_ti, 0])
                                            chunk_k_tiles = pl.assemble(chunk_k_tiles, qk_tile_bf16, [qk_load_lane * SEQ_TILE, 0])

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                                    for qk_sb_lane in pl.range(SB_BATCH):
                                        qk_sb = sb0 + qk_sb_lane
                                        if qk_sb < ctx_blocks:
                                            qk_tile_bf16 = pl.slice(chunk_k_tiles, [SEQ_TILE, HEAD_DIM], [qk_sb_lane * SEQ_TILE, 0])
                                            qk_raw_scores = pl.matmul(q_padded, qk_tile_bf16, b_trans=True, out_dtype=pl.FP32)
                                            chunk_raw_scores = pl.assemble(chunk_raw_scores, qk_raw_scores, [qk_sb_lane * Q_HEAD_PAD, 0])

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                                    for softmax_sb_lane in pl.range(SB_BATCH):
                                        softmax_sb = sb0 + softmax_sb_lane
                                        if softmax_sb < ctx_blocks:
                                            softmax_s0 = softmax_sb * SEQ_TILE
                                            valid_len = pl.min(SEQ_TILE, ctx_len - softmax_s0)
                                            scores_valid = pl.slice(
                                                chunk_raw_scores, [Q_HEAD_PAD, SEQ_TILE],
                                                [softmax_sb_lane * Q_HEAD_PAD, 0],
                                                valid_shape=[Q_HEAD_BATCH, valid_len],
                                            )
                                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                            scores = pl.mul(scores_padded, ATTN_SCALE)
                                            cur_mi = pl.row_max(scores)
                                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                            cur_li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                                            chunk_exp_padded = pl.assemble(chunk_exp_padded, exp_scores_bf16, [softmax_sb_lane * Q_HEAD_PAD, 0])
                                            chunk_cur_mi = pl.assemble(chunk_cur_mi, cur_mi, [softmax_sb_lane * Q_HEAD_PAD, 0])
                                            chunk_cur_li = pl.assemble(chunk_cur_li, cur_li, [softmax_sb_lane * Q_HEAD_PAD, 0])

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_dequant"):
                                    for sv_load_lane in pl.range(SB_BATCH):
                                        sv_load_sb = sb0 + sv_load_lane
                                        if sv_load_sb < ctx_blocks:
                                            sv_block_table_idx = b * max_blocks_per_seq + sv_load_sb
                                            sv_pbid = pl.cast(pl.tensor.read(block_table, [sv_block_table_idx]), pl.INDEX)
                                            sv_cache_row0 = layer_cache_base + (sv_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                                            sv_tile_i8 = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM], [sv_cache_row0, 0])
                                            sv_tile_bf16 = pl.cast(
                                                pl.full([SEQ_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0),
                                                target_type=pl.BF16,
                                            )
                                            sv_valid_len = pl.min(SEQ_TILE, ctx_len - sv_load_sb * SEQ_TILE)
                                            for sv_scale_ti in pl.range(sv_valid_len):
                                                sv_scale_row = sv_cache_row0 + sv_scale_ti
                                                sv_scale = pl.tensor.read(v_cache_scale, [sv_scale_row, 0])
                                                sv_row_i8 = pl.slice(sv_tile_i8, [1, HEAD_DIM], [sv_scale_ti, 0])
                                                sv_row_fp32 = pl.cast(pl.cast(sv_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                                                sv_tile_bf16 = pl.assemble(sv_tile_bf16, pl.cast(pl.mul(sv_row_fp32, sv_scale), target_type=pl.BF16), [sv_scale_ti, 0])
                                            chunk_v_tiles = pl.assemble(chunk_v_tiles, sv_tile_bf16, [sv_load_lane * SEQ_TILE, 0])

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                                    for sv_sb_lane in pl.range(SB_BATCH):
                                        sv_sb = sb0 + sv_sb_lane
                                        if sv_sb < ctx_blocks:
                                            sv_exp_tile = pl.slice(chunk_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sv_sb_lane * Q_HEAD_PAD, 0])
                                            sv_tile_bf16 = pl.slice(chunk_v_tiles, [SEQ_TILE, HEAD_DIM], [sv_sb_lane * SEQ_TILE, 0])
                                            sv_oi_tmp = pl.matmul(sv_exp_tile, sv_tile_bf16, out_dtype=pl.FP32)
                                            chunk_oi_tmp = pl.assemble(chunk_oi_tmp, sv_oi_tmp, [sv_sb_lane * Q_HEAD_PAD, 0])

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                                    for online_sb_lane in pl.range(SB_BATCH):
                                        online_sb = sb0 + online_sb_lane
                                        if online_sb < ctx_blocks:
                                            oi_sb = pl.slice(chunk_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [online_sb_lane * Q_HEAD_PAD, 0])
                                            mi_sb = pl.slice(chunk_cur_mi, [Q_HEAD_PAD, 1], [online_sb_lane * Q_HEAD_PAD, 0])
                                            li_sb = pl.slice(chunk_cur_li, [Q_HEAD_PAD, 1], [online_sb_lane * Q_HEAD_PAD, 0])
                                            if online_sb == 0:
                                                oi = oi_sb
                                                li = li_sb
                                                mi = mi_sb
                                            else:
                                                mi_new = pl.maximum(mi, mi_sb)
                                                alpha = pl.exp(pl.sub(mi, mi_new))
                                                beta = pl.exp(pl.sub(mi_sb, mi_new))
                                                li = pl.add(pl.mul(alpha, li), pl.mul(beta, li_sb))
                                                oi = pl.add(pl.row_expand_mul(oi, alpha),
                                                            pl.row_expand_mul(oi_sb, beta))
                                                mi = mi_new

                            ctx_tmp = pl.create_tensor([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_context"):
                                ctx = pl.row_expand_div(oi, li)
                                ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_writeback"):
                                for qi in pl.unroll(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * HEAD_DIM
                                    row = pl.slice(ctx_tmp, [1, HEAD_DIM], [qi, 0])
                                    row_bf16 = pl.cast(row, target_type=pl.BF16)
                                    attn_row = pl.assemble(attn_row, row_bf16, [0, q_col])

                        attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])

                attn_i8 = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.INT8)
                attn_scales = pl.create_tensor([TOK_TILE, 8], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_act_quant"):
                    for attn_scale_ti in pl.range(valid_tok):
                        attn_amax_value = INT8_AMAX_EPS
                        for kb in pl.range(HIDDEN_BLOCKS):
                            out_quant_amax_k0 = kb * K_CHUNK
                            out_quant_amax_row_bf16 = pl.slice(
                                attn_tile, [1, K_CHUNK], [attn_scale_ti, out_quant_amax_k0]
                            )
                            out_quant_amax_row_f = pl.cast(out_quant_amax_row_bf16, target_type=pl.FP32)
                            out_quant_amax_row_abs = pl.maximum(out_quant_amax_row_f, pl.neg(out_quant_amax_row_f))
                            for hd in pl.range(K_CHUNK):
                                attn_amax_value = pl.max(
                                    attn_amax_value,
                                    pl.tensor.read(out_quant_amax_row_abs, [0, hd]),
                                )
                        attn_scale_out = pl.mul(
                            pl.full([1, 8], dtype=pl.FP32, value=1.0),
                            attn_amax_value / INT8_SCALE_MAX,
                        )
                        attn_scales = pl.assemble(attn_scales, attn_scale_out, [attn_scale_ti, 0])
                        out_scale_value = INT8_SCALE_MAX / attn_amax_value
                        for kb in pl.range(HIDDEN_BLOCKS):
                            out_quant_write_k0 = kb * K_CHUNK
                            out_scale_row_bf16 = pl.slice(
                                attn_tile, [1, K_CHUNK], [attn_scale_ti, out_quant_write_k0]
                            )
                            out_scale_row_f = pl.cast(out_scale_row_bf16, target_type=pl.FP32)
                            out_quant_i32 = pl.cast(
                                pl.mul(out_scale_row_f, out_scale_value),
                                target_type=pl.INT32,
                                mode="rint",
                            )
                            out_quant_i32 = pl.minimum(
                                pl.maximum(out_quant_i32, pl.full([1, K_CHUNK], dtype=pl.INT32, value=-127)),
                                pl.full([1, K_CHUNK], dtype=pl.INT32, value=127),
                            )
                            out_quant_i16 = pl.cast(out_quant_i32, target_type=pl.FP16, mode="round")
                            out_quant_i8 = pl.cast(out_quant_i16, target_type=pl.INT8, mode="trunc")
                            attn_i8 = pl.assemble(attn_i8, out_quant_i8, [attn_scale_ti, out_quant_write_k0])

                resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                for ob in pl.unroll(Q_OUT_BLOCKS):
                    out_proj_o0 = ob * Q_OUT_CHUNK
                    out_proj_partials = pl.create_tensor([HIDDEN_BLOCKS * TOK_TILE, Q_OUT_CHUNK], dtype=pl.INT32)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_matmul"):
                            out_proj_k0 = kb * K_CHUNK
                            out_proj_tile_a_i = pl.slice(attn_i8, [TOK_TILE, K_CHUNK], [0, out_proj_k0])
                            out_proj_tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + out_proj_k0, out_proj_o0])
                            out_proj_partial_i32 = pl.matmul(out_proj_tile_a_i, out_proj_tile_w_i, out_dtype=pl.INT32)
                            out_proj_partials = pl.assemble(
                                out_proj_partials,
                                out_proj_partial_i32,
                                [kb * TOK_TILE, 0],
                            )
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_reduce_dequant"):
                        out_proj_acc_local = pl.full([TOK_TILE, Q_OUT_CHUNK], dtype=pl.INT32, value=0)
                        for kb in pl.unroll(HIDDEN_BLOCKS):
                            out_proj_partial_i = pl.slice(
                                out_proj_partials,
                                [TOK_TILE, Q_OUT_CHUNK],
                                [kb * TOK_TILE, 0],
                            )
                            out_proj_acc_local = pl.add(out_proj_acc_local, out_proj_partial_i)

                        wo_scale_col = pl.reshape(
                            pl.slice(wo_scale, [1, Q_OUT_CHUNK], [layer_idx, out_proj_o0]),
                            [1, Q_OUT_CHUNK],
                        )
                        o_deq_weighted = pl.col_expand_mul(
                            pl.cast(out_proj_acc_local, target_type=pl.FP32),
                            wo_scale_col,
                        )
                        o_deq = pl.create_tensor([TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        for o_deq_ti in pl.unroll(TOK_TILE):
                            o_deq_row = pl.slice(o_deq_weighted, [1, Q_OUT_CHUNK], [o_deq_ti, 0])
                            o_deq_scale = pl.tensor.read(attn_scales, [o_deq_ti, 0])
                            o_deq = pl.assemble(o_deq, pl.mul(o_deq_row, o_deq_scale), [o_deq_ti, 0])
                        resid_chunk = pl.cast(
                            pl.slice(hidden_states, [TOK_TILE, Q_OUT_CHUNK], [token_p0, out_proj_o0],
                                     valid_shape=[valid_tok, Q_OUT_CHUNK]),
                            target_type=pl.FP32,
                        )
                        resid1_chunk = pl.add(o_deq, resid_chunk)
                        resid1_chunk_valid = pl.tensor.set_validshape(resid1_chunk, valid_tok, Q_OUT_CHUNK)
                        resid1_tile = pl.assemble(resid1_tile, resid1_chunk_valid, [0, out_proj_o0])

                post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                    sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        post_rms_sum_k0 = kb * K_CHUNK
                        post_rms_sum_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, post_rms_sum_k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(post_rms_sum_chunk, post_rms_sum_chunk)), [1, TOK_TILE]),
                        )
                    post_inv_rms = pl.recip(pl.sqrt(pl.reshape(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                        [TOK_TILE, 1],
                    )))

                    for kb in pl.range(HIDDEN_BLOCKS):
                        post_rms_norm_k0 = kb * K_CHUNK
                        post_rms_norm_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, post_rms_norm_k0])
                        post_rms_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, post_rms_norm_k0])
                        post_rms_normed = pl.col_expand_mul(
                            pl.row_expand_mul(post_rms_norm_chunk, post_inv_rms),
                            post_rms_gamma,
                        )
                        post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_rms_normed, target_type=pl.BF16), [0, post_rms_norm_k0])

                mlp_silu_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.BF16)
                for ob in pl.range(MLP_OUT_BLOCKS):
                    mlp_out_o0 = ob * MLP_OUT_CHUNK

                    gate_acc = pl.create_tensor([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    gate_partials = pl.create_tensor([HIDDEN_BLOCKS * TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    gate_ready = pl.create_tensor([HIDDEN_BLOCKS], dtype=pl.INT32)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj_matmul"):
                            gate_proj_k0 = kb * K_CHUNK
                            gate_proj_tile_a_i = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, gate_proj_k0])
                            gate_proj_tile_w_i = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + gate_proj_k0, mlp_out_o0])
                            gate_partial = pl.matmul(gate_proj_tile_a_i, gate_proj_tile_w_i, out_dtype=pl.FP32)
                            gate_partials = pl.assemble(gate_partials, gate_partial, [kb * TOK_TILE, 0])
                            pl.tensor.write(gate_ready, [kb], pl.cast(kb * 0 + 1, target_type=pl.INT32))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj_reduce"):
                        gate_acc_local = pl.full([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.unroll(HIDDEN_BLOCKS):
                            gate_partial_i = pl.slice(gate_partials, [TOK_TILE, MLP_OUT_CHUNK], [kb * TOK_TILE, 0])
                            gate_ready_i = pl.cast(pl.tensor.read(gate_ready, [kb]), target_type=pl.FP32)
                            gate_partial_i = pl.mul(gate_partial_i, gate_ready_i)
                            gate_acc_local = pl.add(gate_acc_local, gate_partial_i)
                        gate_acc = pl.assemble(gate_acc, gate_acc_local, [0, 0])

                    up_acc = pl.create_tensor([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    up_partials = pl.create_tensor([HIDDEN_BLOCKS * TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    up_ready = pl.create_tensor([HIDDEN_BLOCKS], dtype=pl.INT32)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj_matmul"):
                            up_proj_k0 = kb * K_CHUNK
                            up_proj_tile_a_i = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, up_proj_k0])
                            up_proj_tile_w_i = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + up_proj_k0, mlp_out_o0])
                            up_partial = pl.matmul(up_proj_tile_a_i, up_proj_tile_w_i, out_dtype=pl.FP32)
                            up_partials = pl.assemble(up_partials, up_partial, [kb * TOK_TILE, 0])
                            pl.tensor.write(up_ready, [kb], pl.cast(kb * 0 + 1, target_type=pl.INT32))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj_reduce"):
                        up_acc_local = pl.full([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.unroll(HIDDEN_BLOCKS):
                            up_partial_i = pl.slice(up_partials, [TOK_TILE, MLP_OUT_CHUNK], [kb * TOK_TILE, 0])
                            up_ready_i = pl.cast(pl.tensor.read(up_ready, [kb]), target_type=pl.FP32)
                            up_partial_i = pl.mul(up_partial_i, up_ready_i)
                            up_acc_local = pl.add(up_acc_local, up_partial_i)
                        up_acc = pl.assemble(up_acc, up_acc_local, [0, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                        mlp_zero = pl.cast(
                            pl.full([TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        )
                        mlp_silu_tile = pl.assemble(mlp_silu_tile, mlp_zero, [0, mlp_out_o0])
                        for mlp_ti in pl.range(valid_tok):
                            gate_row = pl.slice(gate_acc, [1, MLP_OUT_CHUNK], [mlp_ti, 0])
                            up_row = pl.slice(up_acc, [1, MLP_OUT_CHUNK], [mlp_ti, 0])
                            mlp_silu_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_row)), 1.0))
                            mlp_silu_chunk = pl.mul(pl.mul(gate_row, mlp_silu_sigmoid), up_row)
                            mlp_silu_tile = pl.assemble(
                                mlp_silu_tile,
                                pl.cast(mlp_silu_chunk, target_type=pl.BF16),
                                [mlp_ti, mlp_out_o0],
                            )

                mlp_down_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.BF16)
                for ob in pl.range(MLP_OUT_BLOCKS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mlp_down_materialize"):
                        mlp_down_o0 = ob * MLP_OUT_CHUNK
                        mlp_down_chunk = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, mlp_down_o0])
                        mlp_down_tile = pl.assemble(mlp_down_tile, mlp_down_chunk, [0, mlp_down_o0])

                for dob in pl.range(HIDDEN_BLOCKS):
                    down_proj_d0 = dob * K_CHUNK
                    down_acc = pl.create_tensor([TOK_TILE, K_CHUNK], dtype=pl.FP32)
                    down_partials = pl.create_tensor([MLP_OUT_BLOCKS * TOK_TILE, K_CHUNK], dtype=pl.FP32)
                    down_ready = pl.create_tensor([MLP_OUT_BLOCKS], dtype=pl.INT32)
                    for ob in pl.range(MLP_OUT_BLOCKS):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_matmul"):
                            down_proj_o0 = ob * MLP_OUT_CHUNK
                            mlp_chunk_i = pl.slice(mlp_down_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, down_proj_o0])
                            w_down_chunk_i = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + down_proj_o0, down_proj_d0])
                            down_partial = pl.matmul(mlp_chunk_i, w_down_chunk_i, out_dtype=pl.FP32)
                            down_partials = pl.assemble(down_partials, down_partial, [ob * TOK_TILE, 0])
                            pl.tensor.write(down_ready, [ob], pl.cast(ob * 0 + 1, target_type=pl.INT32))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_reduce"):
                        down_acc_local = pl.full([TOK_TILE, K_CHUNK], dtype=pl.FP32, value=0.0)
                        for ob in pl.unroll(MLP_OUT_BLOCKS):
                            down_partial_i = pl.slice(down_partials, [TOK_TILE, K_CHUNK], [ob * TOK_TILE, 0])
                            down_ready_i = pl.cast(pl.tensor.read(down_ready, [ob]), target_type=pl.FP32)
                            down_partial_i = pl.mul(down_partial_i, down_ready_i)
                            down_acc_local = pl.add(down_acc_local, down_partial_i)
                        down_acc = pl.assemble(down_acc, down_acc_local, [0, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_residual"):
                        out_chunk = pl.add(
                            down_acc,
                            pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, down_proj_d0]),
                        )
                        out_out_quant_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                        out_out_quant_chunk_valid = pl.tensor.set_validshape(
                            out_out_quant_chunk_bf16,
                            valid_tok,
                            K_CHUNK,
                        )
                        out = pl.assemble(out, out_out_quant_chunk_valid, [token_p0, down_proj_d0])

    return out


@pl.jit(auto_scope=False)
def prefill_hidden_a8w8(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.INT8],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.INT8],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.INT8],
    wq_scale: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wk_scale: pl.Tensor[[LAYER_DYN, KV_HIDDEN], pl.FP32],
    wv_scale: pl.Tensor[[LAYER_DYN, KV_HIDDEN], pl.FP32],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.INT8],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.INT8],
    k_cache_scale: pl.Tensor[[KV_CACHE_ROWS_DYN, 8], pl.FP32],
    v_cache_scale: pl.Tensor[[KV_CACHE_ROWS_DYN, 8], pl.FP32],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.INT8],
    wo_scale: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
    hidden_out: pl.Out[pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16]],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_offsets.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    hidden_out.bind_dynamic(0, PREFILL_TOKENS_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    k_cache_scale.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache_scale.bind_dynamic(0, KV_CACHE_ROWS_DYN)

    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    prefill_tokens = pl.tensor.dim(hidden_states, 0)
    cur = pl.create_tensor([prefill_tokens, HIDDEN], dtype=pl.BF16)
    for p0 in pl.parallel(0, prefill_tokens, TOK_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            valid_tok = pl.min(TOK_TILE, prefill_tokens - p0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [TOK_TILE, K_CHUNK],
                    [p0, k0],
                    valid_shape=[valid_tok, K_CHUNK],
                )
                cur = pl.assemble(cur, hidden_chunk, [p0, k0])

    for layer_idx in pl.range(num_layers_actual):
        next_hidden = pl.create_tensor([prefill_tokens, HIDDEN], dtype=pl.BF16)
        cur = prefill_layer(
            cur,
            seq_lens,
            chunk_lens,
            chunk_offsets,
            input_rms_weight,
            wq,
            wk,
            wv,
            wq_scale,
            wk_scale,
            wv_scale,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            block_table,
            slot_mapping,
            k_cache,
            v_cache,
            k_cache_scale,
            v_cache_scale,
            wo,
            wo_scale,
            post_rms_weight,
            w_gate,
            w_up,
            w_down,
            next_hidden,
            layer_idx,
        )

    for p0 in pl.parallel(0, prefill_tokens, TOK_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_hidden_out"):
            valid_tok = pl.min(TOK_TILE, prefill_tokens - p0)
            for kb in pl.range(HIDDEN_BLOCKS):
                store_hidden_k0 = kb * K_CHUNK
                hidden_out_chunk = pl.slice(
                    cur,
                    [TOK_TILE, K_CHUNK],
                    [p0, store_hidden_k0],
                    valid_shape=[valid_tok, K_CHUNK],
                )
                hidden_out = pl.assemble(hidden_out, hidden_out_chunk, [p0, store_hidden_k0])

    return hidden_out
