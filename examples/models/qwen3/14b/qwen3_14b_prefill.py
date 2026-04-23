# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B prefill Scope 1+2+3.

Fuses the full single-layer prefill path: input RMSNorm, Q/K/V projection,
RoPE, KV cache update, causal attention, output projection, post-attention
RMSNorm, SwiGLU MLP, and the final residual path.
"""
from __future__ import annotations

import pypto.language as pl


# Qwen3-14B model dimensions for initial validation.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM       # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 17408

# RMSNorm constants.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM

# Tiling constants shared across the three scopes.
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 64
Q_HEAD_BATCH = 5        # Q heads per attention group
Q_HEAD_BATCH_PAD = 8    # padded to 32-byte alignment (8 * 4 = 32)
Q_HEAD_PAD = 16         # padded Q rows for cube alignment
SEQ_TILE = 64           # sequence tile for attention
SB_BATCH = 64
MLP_OUT_CHUNK = 128


def build_qwen3_14b_prefill_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks = intermediate_size // MLP_OUT_CHUNK
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE
    hidden_inv = 1.0 / hidden
    head_dim_inv = 1.0 / head_dim

    @pl.program
    class Qwen314BPrefillProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_14b_prefill(
            self,
            hidden_states: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, intermediate_size], pl.BF16],
            w_up: pl.Tensor[[hidden, intermediate_size], pl.BF16],
            w_down: pl.Tensor[[intermediate_size, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, max_seq, hidden], pl.BF16]:
            for b in pl.parallel(0, batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # ── Scope 1: input RMSNorm + Q/K/V projection ──
                    normed_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                    # Stage 1.1: RMSNorm (vector ops).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                             valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            partial_sq = pl.add(
                                partial_sq,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        variance = pl.reshape(
                            pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                            [TOK_TILE, 1],
                        )
                        inv_rms = pl.recip(pl.sqrt(variance))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                             valid_shape=[1, valid_tok, K_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # Stage 1.2: Q projection (matmul + matmul_acc, FP32 output).
                    q_proj_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for ob in pl.parallel(q_out_blocks, chunk=4):
                            q0 = ob * Q_OUT_CHUNK
                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                            q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                            q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                    # Stage 1.3: K/V projection (matmul + matmul_acc in single incore).
                    k_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    v_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for ob in pl.parallel(kv_out_blocks, chunk=4):
                            kv0 = ob * KV_OUT_CHUNK

                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                            k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                            k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                            v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                                v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                            v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

                    # Stage 1.4: Q/K per-head RMSNorm (FP32 in-place on proj tiles).
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for qh in pl.parallel(0, num_heads, chunk=num_heads):
                            q_col = qh * head_dim
                            q_head = pl.slice(q_proj_tile, [TOK_TILE, head_dim], [0, q_col])
                            q_sq = pl.reshape(
                                pl.row_sum(pl.mul(q_head, q_head)),
                                [TOK_TILE, 1],
                            )
                            q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, head_dim_inv), EPS)))
                            q_normed = pl.col_expand_mul(
                                pl.row_expand_mul(q_head, q_inv_rms),
                                q_norm_weight,
                            )
                            q_proj_tile = pl.assemble(q_proj_tile, q_normed, [0, q_col])
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for kh in pl.parallel(0, num_kv_heads, chunk=num_kv_heads):
                            k_col = kh * head_dim
                            k_head = pl.slice(k_proj_tile, [TOK_TILE, head_dim], [0, k_col])
                            k_sq = pl.reshape(
                                pl.row_sum(pl.mul(k_head, k_head)),
                                [TOK_TILE, 1],
                            )
                            k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, head_dim_inv), EPS)))
                            k_normed = pl.col_expand_mul(
                                pl.row_expand_mul(k_head, k_inv_rms),
                                k_norm_weight,
                            )
                            k_proj_tile = pl.assemble(k_proj_tile, k_normed, [0, k_col])

                    # ── Scope 2: RoPE + KV cache update + causal attention ──
                    attn_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    for ti in pl.range(valid_tok):
                        pos = p0 + ti
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                        sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                        cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                        cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                        sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                        sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                        # Stage 2.1: K RoPE + cache update + V cache + Q RoPE + pad.
                        all_q_padded = pl.create_tensor([total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            for gi in pl.parallel(0, total_q_groups, chunk=total_q_groups):
                                all_q_padded = pl.assemble(
                                    all_q_padded,
                                    pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                                    [gi * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                                )
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            for ki in pl.parallel(0, num_kv_heads, chunk=8):
                                # K RoPE + cache update.
                                kv_col = ki * head_dim
                                k_lo = pl.reshape(pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col]), [1, half_dim])
                                k_hi = pl.reshape(pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col + half_dim]), [1, half_dim])
                                rot_lo = pl.sub(
                                    pl.col_expand_mul(k_lo, cos_lo),
                                    pl.col_expand_mul(k_hi, sin_lo),
                                )
                                rot_hi = pl.add(
                                    pl.col_expand_mul(k_hi, cos_hi),
                                    pl.col_expand_mul(k_lo, sin_hi),
                                )
                                cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.cast(rot_lo, target_type=pl.BF16),
                                    [cache_row, 0],
                                )
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.cast(rot_hi, target_type=pl.BF16),
                                    [cache_row, half_dim],
                                )
                                # V cache update.
                                v_cache = pl.assemble(
                                    v_cache,
                                    pl.cast(
                                        pl.reshape(pl.slice(v_proj_tile, [1, head_dim], [ti, ki * head_dim]), [1, head_dim]),
                                        target_type=pl.BF16,
                                    ),
                                    [cache_row, 0],
                                )
                                # Q RoPE + pad.
                                q_base = ki * q_per_kv
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    q_lo = pl.reshape(pl.slice(q_proj_tile, [1, half_dim], [ti, q_col]), [1, half_dim])
                                    q_hi = pl.reshape(pl.slice(q_proj_tile, [1, half_dim], [ti, q_col + half_dim]), [1, half_dim])
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
                                    all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [ki * Q_HEAD_PAD + qi, half_dim])

                        attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)

                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            qg = gi - kvh * q_groups
                            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH

                            q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [gi * Q_HEAD_PAD, 0])

                            all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                            all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                            all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                            all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)
                            all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)

                            # Stage 2.2: QK matmul for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    k_tile = pl.slice(k_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                                    all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                            # Stage 2.3: softmax for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    scores_valid = pl.slice(
                                        all_raw_scores, [Q_HEAD_BATCH_PAD, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                        valid_shape=[Q_HEAD_BATCH, valid_len],
                                    )
                                    scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                    scores = pl.mul(scores_padded, attn_scale)
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                    cur_li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH_PAD, 0])
                                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH_PAD, 0])

                            # Stage 2.4: SV matmul for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sb * Q_HEAD_PAD, 0])
                                    v_tile = pl.slice(v_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                    oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                            # Stage 2.5: online softmax accumulation.
                            with pl.at(level=pl.Level.CORE_GROUP):
                                oi = pl.full([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32, value=0.0)
                                li_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                li = pl.reshape(li_flat, [Q_HEAD_BATCH_PAD, 1])
                                mi_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                mi = pl.reshape(mi_flat, [Q_HEAD_BATCH_PAD, 1])

                            for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb0 + si
                                        if sb < ctx_blocks:
                                            oi_sb = pl.slice(all_oi_tmp, [Q_HEAD_BATCH_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                                            mi_sb = pl.slice(all_cur_mi, [Q_HEAD_BATCH_PAD, 1], [sb * Q_HEAD_BATCH_PAD, 0])
                                            li_sb = pl.slice(all_cur_li, [Q_HEAD_BATCH_PAD, 1], [sb * Q_HEAD_BATCH_PAD, 0])
                                            if sb == 0:
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

                            # Finalize ctx = oi / li and write back row by row.
                            ctx_tmp = pl.create_tensor([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32)
                            with pl.at(level=pl.Level.CORE_GROUP):
                                ctx = pl.row_expand_div(oi, li)
                                ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                            with pl.at(level=pl.Level.CORE_GROUP):
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    row = pl.slice(ctx_tmp, [1, head_dim], [qi, 0])
                                    row_bf16 = pl.cast(row, target_type=pl.BF16)
                                    attn_row = pl.assemble(attn_row, row_bf16, [0, q_col])

                        # Keep the attention row in the local tile.
                        attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])
                    # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
                    # Stage 3.1: Output projection + first residual.
                    resid1_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    for ob in pl.range(q_out_blocks):
                        o0 = ob * Q_OUT_CHUNK
                        # Cube matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        # Add the residual path.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                            resid1_tile = pl.assemble(resid1_tile, pl.add(mm_out, resid_chunk), [0, o0])

                    # Stage 3.2: Post-attention RMSNorm.
                    post_norm_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        post_inv_rms = pl.recip(pl.sqrt(pl.reshape(
                            pl.add(pl.mul(sq_sum, hidden_inv), EPS),
                            [TOK_TILE, 1],
                        )))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, post_inv_rms),
                                gamma,
                            )
                            post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # Stage 3.3a: MLP gate/up + SiLU.
                    mlp_silu_tile = pl.create_tensor([TOK_TILE, intermediate_size], dtype=pl.BF16)
                    for ob in pl.range(mlp_out_blocks):
                        o0 = ob * MLP_OUT_CHUNK

                        # Gate matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        # Up matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        # SiLU activation: silu(gate) * up -> BF16.
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                            mlp_silu_tile = pl.assemble(mlp_silu_tile, mlp_chunk_bf16, [0, o0])

                    # Stage 3.3b: Down projection (matmul_acc chain) + final residual -> output.
                    for dob in pl.range(hidden_blocks):
                        d0 = dob * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP):
                            mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, 0])
                            w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                            for ob in pl.range(1, mlp_out_blocks):
                                o0 = ob * MLP_OUT_CHUNK
                                mlp_chunk_i = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                                w_down_chunk_i = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            out_chunk = pl.add(
                                down_acc,
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                            )
                            out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                            out = pl.assemble(out, out_chunk_bf16, [b, p0, d0])

            return out

    return Qwen314BPrefillProgram


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    use_max_seq: bool = False,
):
    import torch
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    def init_hidden_states():
        return torch.rand(batch, max_seq, hidden_size) - 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_wk():
        return (torch.rand(hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_wv():
        return (torch.rand(hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.rand(1, head_dim) - 0.5

    def init_k_norm_weight():
        return torch.rand(1, head_dim) - 0.5

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_wo():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, max_seq, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [batch, max_seq, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_14b_prefill(tensors):
    """Reference implementation for Scope 1+2+3 prefill.

    Mirrors the kernel precision path through RMSNorm, Q/K/V projection, RoPE,
    KV cache update, online-softmax attention, output projection, post RMSNorm,
    SwiGLU MLP, and the final BF16 residual writeback.
    """
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    max_seq = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS

    input_rms_weight_f = input_rms_weight.float()
    wq_f = wq.float()
    wk_f = wk.float()
    wv_f = wv.float()
    wo_f = wo.float()
    post_rms_f = post_rms_weight.float()
    w_gate_f = w_gate.float()
    w_up_f = w_up.float()
    w_down_f = w_down.float()

    out_t = torch.zeros(batch, max_seq, hidden_size, dtype=torch.float32)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        if seq_len_b <= 0:
            continue

        S = seq_len_b

        # ── Scope 1: RMSNorm + Q/K/V projection ──
        x = hidden_states[b, :S, :].float()
        variance = x.square().mean(dim=-1, keepdim=True) + eps
        inv_rms = 1.0 / torch.sqrt(variance)
        normed_bf16 = (x * inv_rms * input_rms_weight_f).to(torch.bfloat16)

        normed_f32 = normed_bf16.float()
        q_proj_f = normed_f32 @ wq_f
        k_proj_f = normed_f32 @ wk_f
        v_proj_f = normed_f32 @ wv_f

        # Per-head RMSNorm on Q and K (FP32).
        q_norm_f = q_norm_weight.float().view(1, 1, head_dim)
        k_norm_f = k_norm_weight.float().view(1, 1, head_dim)
        q_proj_view = q_proj_f.view(S, num_heads, head_dim)
        q_var = q_proj_view.pow(2).mean(dim=-1, keepdim=True)
        q_proj_view = q_proj_view * torch.rsqrt(q_var + eps) * q_norm_f
        q_proj_f = q_proj_view.reshape(S, hidden_size)
        k_proj_view = k_proj_f.view(S, num_kv_heads, head_dim)
        k_var = k_proj_view.pow(2).mean(dim=-1, keepdim=True)
        k_proj_view = k_proj_view * torch.rsqrt(k_var + eps) * k_norm_f
        k_proj_f = k_proj_view.reshape(S, kv_hidden)

        # ── Scope 2: RoPE + KV cache + causal attention ──
        cos_row = rope_cos[:S, :]
        sin_row = rope_sin[:S, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE + cache write.
        k_row = k_proj_f.view(S, num_kv_heads, head_dim)
        k_lo, k_hi = k_row[:, :, :half], k_row[:, :, half:]
        k_rot = torch.cat([
            k_lo * cos_lo.unsqueeze(1) - k_hi * sin_lo.unsqueeze(1),
            k_hi * cos_hi.unsqueeze(1) + k_lo * sin_hi.unsqueeze(1),
        ], dim=-1).to(torch.bfloat16)
        cache_off = b * num_kv_heads * max_seq
        k_view = k_cache[cache_off:cache_off + num_kv_heads * max_seq, :].view(
            num_kv_heads, max_seq, head_dim)
        k_view[:, :S, :] = k_rot.permute(1, 0, 2)

        # V cache write.
        v_row_bf16 = v_proj_f.view(S, num_kv_heads, head_dim).to(torch.bfloat16)
        v_view = v_cache[cache_off:cache_off + num_kv_heads * max_seq, :].view(
            num_kv_heads, max_seq, head_dim)
        v_view[:, :S, :] = v_row_bf16.permute(1, 0, 2)

        # Q RoPE -> BF16.
        q_row = q_proj_f.view(S, num_heads, head_dim)
        q_lo, q_hi = q_row[:, :, :half], q_row[:, :, half:]
        q_rot_bf16 = torch.cat([
            q_lo * cos_lo.unsqueeze(1) - q_hi * sin_lo.unsqueeze(1),
            q_hi * cos_hi.unsqueeze(1) + q_lo * sin_hi.unsqueeze(1),
        ], dim=-1).to(torch.bfloat16)

        # Causal attention with tiled online softmax.
        max_blocks = (S + SEQ_TILE - 1) // SEQ_TILE
        padded_len = max_blocks * SEQ_TILE
        ctx_lens = torch.arange(1, S + 1)
        col_idx = torch.arange(SEQ_TILE)
        attn_result = torch.zeros(S, hidden_size, dtype=torch.float32)

        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp = q_rot_bf16[:S, q_base:q_base + Q_HEAD_BATCH, :]

                cache_base = b * num_kv_heads * max_seq + kvh * max_seq
                k_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                v_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                k_padded[:S] = k_cache[cache_base:cache_base + S, :]
                v_padded[:S] = v_cache[cache_base:cache_base + S, :]

                oi = li = mi = None

                for sb in range(max_blocks):
                    s0 = sb * SEQ_TILE
                    k_tile = k_padded[s0:s0 + SEQ_TILE]
                    v_tile = v_padded[s0:s0 + SEQ_TILE]

                    raw_scores = q_grp.float() @ k_tile.float().T

                    valid_lens = torch.clamp(ctx_lens - s0, min=0, max=SEQ_TILE)
                    mask = col_idx.unsqueeze(0) < valid_lens.unsqueeze(1)
                    raw_scores[~mask.unsqueeze(1).expand_as(raw_scores)] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi, li, mi = oi_tmp, cur_li, cur_mi
                    else:
                        active = valid_lens > 0
                        if active.any():
                            a = active
                            mi_new = torch.maximum(mi[a], cur_mi[a])
                            alpha = torch.exp(mi[a] - mi_new)
                            beta = torch.exp(cur_mi[a] - mi_new)
                            oi[a] = oi[a] * alpha + oi_tmp[a] * beta
                            li[a] = alpha * li[a] + beta * cur_li[a]
                            mi[a] = mi_new

                ctx = oi / li
                attn_result[:, q_base * head_dim:(q_base + Q_HEAD_BATCH) * head_dim] = \
                    ctx.reshape(S, Q_HEAD_BATCH * head_dim)

        attn_bf16 = attn_result.to(torch.bfloat16)

        # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
        attn_f = attn_bf16.float()
        hs = hidden_states[b, :S, :].float()

        # Output projection + first residual.
        resid1 = torch.matmul(attn_f, wo_f) + hs

        # Post-attention RMSNorm.
        variance = resid1.pow(2).mean(dim=-1, keepdim=True)
        post_inv_rms = torch.rsqrt(variance + eps)
        normed_bf16 = (resid1 * post_inv_rms * post_rms_f).bfloat16()

        # SwiGLU MLP.
        normed_post_f = normed_bf16.float()
        gate = torch.matmul(normed_post_f, w_gate_f)
        up = torch.matmul(normed_post_f, w_up_f)
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = torch.matmul(mlp_bf16.float(), w_down_f)

        # Final residual -> BF16.
        out_t[b, :S, :] = (down + resid1).bfloat16().float()

    tensors["out"][:] = out_t.to(torch.bfloat16)


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    use_max_seq: bool = False,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
    runtime_dir: str | None = None,
):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    program = build_qwen3_14b_prefill_program(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        use_max_seq=use_max_seq,
    )

    golden_data = str(Path(runtime_dir) / "data") if runtime_dir else None

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden_fn=golden_qwen3_14b_prefill,
        runtime_dir=runtime_dir,
        golden_data=golden_data,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=dump_passes),
            runtime=dict(
                platform=platform,
                device_id=device_id,
                runtime_profiling=runtime_profiling,
            ),
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument(
        "--runtime-dir", type=str, default=None, help="reuse a previous build_output dir and golden data"
    )
    parser.add_argument("--max-seq", action="store_true", default=False, help="set all seq_lens to MAX_SEQ")
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        use_max_seq=args.max_seq,
        runtime_profiling=args.runtime_profiling,
        runtime_dir=args.runtime_dir,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
