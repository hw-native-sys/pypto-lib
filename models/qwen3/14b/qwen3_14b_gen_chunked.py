# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B unified generation kernel: ONE L3 host_orch calling prefill (L2) and decode (L2).

Architecture
------------
Level 3 (HOST Orchestrator):
    host_orch — single entry point for all generation steps.
    Per layer in the chunk:
        1. qwen3_prefill_layer (L2, Orchestration) — if has_prefill=1
           Processes all prompt positions; writes KV for the chunk layer.
        2. qwen3_decode_layer (L2, Orchestration) — always
           Processes one decode token; reads KV just written.

This interleaved dispatch reuses the same weight chunk for both stages,
halving weight-load bandwidth on the first generation step.

has_prefill flag
----------------
has_prefill=1 (step 0, combined prefill + first decode):
    • prefill_hidden  : [batch, max_seq, hidden] — embedded prompt tokens
    • prefill_seq_lens: [batch]                  — actual prompt lengths
    • prefill_slot_mapping: [batch * max_seq]    — physical KV slots for all prompt positions
    • decode_hidden   : [batch, hidden]          — embed(last_prompt_token)
    • decode_seq_lens : [batch]                  — same as prefill_seq_lens (= N)
    • decode_slot_mapping : [batch]              — slot for position N-1 (last prompt slot)
    After: KV cache has N entries (0..N-1); decode_out holds the hidden state for
    predicting the first new token (same semantics as prefill_out[:, N-1, :]).

has_prefill=0 (steps 1+, pure decode):
    • prefill_hidden / prefill_slot_mapping are unused dummy tensors.
    • decode_hidden   : [batch, hidden]  — embed(current_decode_token)
    • decode_seq_lens : [batch]          — N+t (context length including current token)
    • decode_slot_mapping : [batch]      — slot for position N+t-1
    After: KV cache grows by 1 per step; decode_out holds next-token hidden state.

Stacking layout: per-chunk row-stacked 1D weights ([chunk_size, dim]) and
    flat-stacked 2D weights ([chunk_size * d_in, d_out]); KV cache stays
    full-stacked across all num_layers. See `stack_layer_weights_chunked`
    below for the exact field layout.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
KV_CACHE_ROWS_ALL_DYN = pl.dynamic("KV_CACHE_ROWS_ALL_DYN")
SLOT_MAPPING_DYN = pl.dynamic("SLOT_MAPPING_DYN")

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
INTERMEDIATE = 17408
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

EPS = 1e-6

# Shared tiling constants.
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
BATCH_TILE = 16

# Prefill-specific tiling.
TOK_TILE = 64
Q_HEAD_BATCH = 5
Q_HEAD_BATCH_PAD = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256
SB_BATCH = 64
BLOCK_SIZE = SEQ_TILE
MLP_OUT_CHUNK_PREFILL = 128

# Decode-specific tiling.
SCOPE1_K_CHUNK = 512
MLP_OUT_CHUNK_DECODE = 256


def build_qwen3_14b_gen_chunked_program(
    num_layers: int = 40,
    chunk_size: int = 4,
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    assert num_layers % chunk_size == 0, (
        f"num_layers={num_layers} must be divisible by chunk_size={chunk_size}"
    )
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    hidden_inv = 1.0 / hidden
    head_dim_inv = 1.0 / head_dim

    # Prefill tiling derived values.
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks_prefill = inter // MLP_OUT_CHUNK_PREFILL
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    half_dim = head_dim // 2
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE

    # Decode tiling derived values.
    scope1_hidden_blocks = hidden // SCOPE1_K_CHUNK
    mlp_out_blocks_decode = inter // MLP_OUT_CHUNK_DECODE

    @pl.program
    class Qwen3GenChunked:

        # ── L2: per-layer prefill ──────────────────────────────────────────────
        @pl.function(type=pl.FunctionType.Orchestration)
        def qwen3_prefill_layer(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            input_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            wq_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            wk_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            wv_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            q_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            k_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[SLOT_MAPPING_DYN], pl.INT32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            post_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            w_gate_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_up_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_down_chunk_flat: pl.Tensor[[chunk_size * inter, hidden], pl.BF16],
            local_layer_idx: pl.Scalar[pl.INT32],
            kv_layer_offset_base: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]:
            user_batch = pl.tensor.dim(hidden_states, 0)
            cache_rows_per_layer = pl.tensor.dim(k_cache_all, 0) // num_layers

            layer_off_h = local_layer_idx * hidden
            layer_off_inter = local_layer_idx * inter
            global_layer_idx = kv_layer_offset_base + local_layer_idx
            layer_off_cache = global_layer_idx * cache_rows_per_layer

            q_norm_w = pl.slice(q_norm_chunk, [1, head_dim], [local_layer_idx, 0])
            k_norm_w = pl.slice(k_norm_chunk, [1, head_dim], [local_layer_idx, 0])

            for b in pl.parallel(0, user_batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # ── Scope 1: input RMSNorm + Q/K/V projection ──
                    normed_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            partial_sq = pl.add(
                                partial_sq,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        variance = pl.reshape(
                            pl.add(pl.mul(partial_sq, hidden_inv), EPS),
                            [TOK_TILE, 1],
                        )
                        inv_rms = pl.recip(pl.sqrt(variance))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                        valid_shape=[1, valid_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_CHUNK],
                            )
                            gamma = pl.slice(input_rms_chunk, [1, K_CHUNK], [local_layer_idx, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            normed_tile = pl.assemble(
                                normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                            )

                    q_proj_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for ob in pl.parallel(q_out_blocks, chunk=4):
                            q0 = ob * Q_OUT_CHUNK
                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wq_chunk_flat, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, q0])
                            q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(
                                    wq_chunk_flat, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h + k0, q0]
                                )
                                q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                            q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                    k_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    v_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for ob in pl.parallel(kv_out_blocks, chunk=4):
                            kv0 = ob * KV_OUT_CHUNK
                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_wk = pl.slice(wk_chunk_flat, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0])
                            k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_wk_i = pl.slice(
                                    wk_chunk_flat, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h + k0, kv0]
                                )
                                k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                            k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                            tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_wv = pl.slice(wv_chunk_flat, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0])
                            v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_wv_i = pl.slice(
                                    wv_chunk_flat, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h + k0, kv0]
                                )
                                v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                            v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for qh in pl.parallel(0, num_heads, chunk=num_heads):
                            q_col = qh * head_dim
                            q_head = pl.slice(q_proj_tile, [TOK_TILE, head_dim], [0, q_col])
                            q_sq = pl.reshape(pl.row_sum(pl.mul(q_head, q_head)), [TOK_TILE, 1])
                            q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, head_dim_inv), EPS)))
                            q_normed = pl.col_expand_mul(pl.row_expand_mul(q_head, q_inv_rms), q_norm_w)
                            q_proj_tile = pl.assemble(q_proj_tile, q_normed, [0, q_col])
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for kh in pl.parallel(0, num_kv_heads, chunk=num_kv_heads):
                            k_col = kh * head_dim
                            k_head = pl.slice(k_proj_tile, [TOK_TILE, head_dim], [0, k_col])
                            k_sq = pl.reshape(pl.row_sum(pl.mul(k_head, k_head)), [TOK_TILE, 1])
                            k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, head_dim_inv), EPS)))
                            k_normed = pl.col_expand_mul(pl.row_expand_mul(k_head, k_inv_rms), k_norm_w)
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

                        all_q_padded = pl.create_tensor(
                            [total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16
                        )
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            for gi in pl.parallel(0, total_q_groups, chunk=total_q_groups):
                                all_q_padded = pl.assemble(
                                    all_q_padded,
                                    pl.cast(
                                        pl.full(
                                            [Q_HEAD_PAD - Q_HEAD_BATCH, head_dim],
                                            dtype=pl.FP32,
                                            value=0.0,
                                        ),
                                        target_type=pl.BF16,
                                    ),
                                    [gi * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                                )
                        cache_slot = pl.cast(
                            pl.tensor.read(slot_mapping, [b * max_seq + pos]), pl.INDEX
                        )
                        cache_slot_block = cache_slot // BLOCK_SIZE
                        cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            for ki in pl.parallel(0, num_kv_heads, chunk=8):
                                kv_col = ki * head_dim
                                k_lo = pl.reshape(
                                    pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col]), [1, half_dim]
                                )
                                k_hi = pl.reshape(
                                    pl.slice(k_proj_tile, [1, half_dim], [ti, kv_col + half_dim]),
                                    [1, half_dim],
                                )
                                rot_lo = pl.sub(
                                    pl.col_expand_mul(k_lo, cos_lo),
                                    pl.col_expand_mul(k_hi, sin_lo),
                                )
                                rot_hi = pl.add(
                                    pl.col_expand_mul(k_hi, cos_hi),
                                    pl.col_expand_mul(k_lo, sin_hi),
                                )
                                cache_row = (
                                    (cache_slot_block * num_kv_heads + ki) * BLOCK_SIZE
                                    + cache_slot_offset
                                )
                                k_cache_all = pl.assemble(
                                    k_cache_all,
                                    pl.cast(rot_lo, target_type=pl.BF16),
                                    [layer_off_cache + cache_row, 0],
                                )
                                k_cache_all = pl.assemble(
                                    k_cache_all,
                                    pl.cast(rot_hi, target_type=pl.BF16),
                                    [layer_off_cache + cache_row, half_dim],
                                )
                                v_cache_all = pl.assemble(
                                    v_cache_all,
                                    pl.cast(
                                        pl.reshape(
                                            pl.slice(v_proj_tile, [1, head_dim], [ti, ki * head_dim]),
                                            [1, head_dim],
                                        ),
                                        target_type=pl.BF16,
                                    ),
                                    [layer_off_cache + cache_row, 0],
                                )
                                q_base = ki * q_per_kv
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    q_lo = pl.reshape(
                                        pl.slice(q_proj_tile, [1, half_dim], [ti, q_col]),
                                        [1, half_dim],
                                    )
                                    q_hi = pl.reshape(
                                        pl.slice(q_proj_tile, [1, half_dim], [ti, q_col + half_dim]),
                                        [1, half_dim],
                                    )
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
                                    all_q_padded = pl.assemble(
                                        all_q_padded, rot_lo_bf16, [ki * Q_HEAD_PAD + qi, 0]
                                    )
                                    all_q_padded = pl.assemble(
                                        all_q_padded,
                                        rot_hi_bf16,
                                        [ki * Q_HEAD_PAD + qi, half_dim],
                                    )

                        attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            qg = gi - kvh * q_groups
                            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                            q_padded = pl.slice(
                                all_q_padded, [Q_HEAD_PAD, head_dim], [gi * Q_HEAD_PAD, 0]
                            )
                            all_raw_scores = pl.create_tensor(
                                [max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32
                            )
                            all_exp_padded = pl.create_tensor(
                                [max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16
                            )
                            all_oi_tmp = pl.create_tensor(
                                [max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32
                            )
                            all_cur_mi = pl.create_tensor(
                                [max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32
                            )
                            all_cur_li = pl.create_tensor(
                                [max_ctx_blocks * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32
                            )

                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    block_table_idx = b * max_blocks_per_seq + sb
                                    pbid = pl.cast(
                                        pl.tensor.read(block_table, [block_table_idx]), pl.INDEX
                                    )
                                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                                    k_tile = pl.slice(
                                        k_cache_all,
                                        [SEQ_TILE, head_dim],
                                        [layer_off_cache + cache_row0, 0],
                                    )
                                    raw_scores = pl.matmul(
                                        q_padded, k_tile, b_trans=True, out_dtype=pl.FP32
                                    )
                                    all_raw_scores = pl.assemble(
                                        all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0]
                                    )

                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    scores_valid = pl.slice(
                                        all_raw_scores,
                                        [Q_HEAD_BATCH_PAD, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                        valid_shape=[Q_HEAD_BATCH, valid_len],
                                    )
                                    scores_padded = pl.fillpad(
                                        scores_valid, pad_value=pl.PadValue.min
                                    )
                                    scores = pl.mul(scores_padded, attn_scale)
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                    cur_li = pl.row_sum(
                                        pl.cast(exp_scores_bf16, target_type=pl.FP32)
                                    )
                                    all_exp_padded = pl.assemble(
                                        all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0]
                                    )
                                    all_cur_mi = pl.assemble(
                                        all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH_PAD, 0]
                                    )
                                    all_cur_li = pl.assemble(
                                        all_cur_li, cur_li, [sb * Q_HEAD_BATCH_PAD, 0]
                                    )

                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    block_table_idx = b * max_blocks_per_seq + sb
                                    pbid = pl.cast(
                                        pl.tensor.read(block_table, [block_table_idx]), pl.INDEX
                                    )
                                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                                    exp_tile = pl.slice(
                                        all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sb * Q_HEAD_PAD, 0]
                                    )
                                    v_tile = pl.slice(
                                        v_cache_all,
                                        [SEQ_TILE, head_dim],
                                        [layer_off_cache + cache_row0, 0],
                                    )
                                    oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                    all_oi_tmp = pl.assemble(
                                        all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0]
                                    )

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
                                            oi_sb = pl.slice(
                                                all_oi_tmp,
                                                [Q_HEAD_BATCH_PAD, head_dim],
                                                [sb * Q_HEAD_PAD, 0],
                                            )
                                            mi_sb = pl.slice(
                                                all_cur_mi,
                                                [Q_HEAD_BATCH_PAD, 1],
                                                [sb * Q_HEAD_BATCH_PAD, 0],
                                            )
                                            li_sb = pl.slice(
                                                all_cur_li,
                                                [Q_HEAD_BATCH_PAD, 1],
                                                [sb * Q_HEAD_BATCH_PAD, 0],
                                            )
                                            if sb == 0:
                                                oi = oi_sb
                                                li = li_sb
                                                mi = mi_sb
                                            else:
                                                mi_new = pl.maximum(mi, mi_sb)
                                                alpha = pl.exp(pl.sub(mi, mi_new))
                                                beta = pl.exp(pl.sub(mi_sb, mi_new))
                                                li = pl.add(
                                                    pl.mul(alpha, li), pl.mul(beta, li_sb)
                                                )
                                                oi = pl.add(
                                                    pl.row_expand_mul(oi, alpha),
                                                    pl.row_expand_mul(oi_sb, beta),
                                                )
                                                mi = mi_new

                            ctx_tmp = pl.create_tensor([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32)
                            with pl.at(level=pl.Level.CORE_GROUP):
                                ctx = pl.row_expand_div(oi, li)
                                ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                            with pl.at(level=pl.Level.CORE_GROUP):
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    row = pl.slice(ctx_tmp, [1, head_dim], [qi, 0])
                                    attn_row = pl.assemble(
                                        attn_row, pl.cast(row, target_type=pl.BF16), [0, q_col]
                                    )

                        attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])

                    # ── Scope 3: Wo + residual + post-RMSNorm + MLP ──
                    resid1_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    for ob in pl.range(q_out_blocks):
                        o0 = ob * Q_OUT_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(
                                wo_chunk_flat, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, o0]
                            )
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(
                                    wo_chunk_flat, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h + k0, o0]
                                )
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        with pl.at(level=pl.Level.CORE_GROUP):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [1, TOK_TILE, Q_OUT_CHUNK],
                                        [b, p0, o0],
                                        valid_shape=[1, valid_tok, Q_OUT_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                            resid1_tile = pl.assemble(
                                resid1_tile, pl.add(mm_out, resid_chunk), [0, o0]
                            )

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
                        post_inv_rms = pl.recip(
                            pl.sqrt(
                                pl.reshape(pl.add(pl.mul(sq_sum, hidden_inv), EPS), [TOK_TILE, 1])
                            )
                        )
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_chunk, [1, K_CHUNK], [local_layer_idx, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, post_inv_rms), gamma
                            )
                            post_norm_tile = pl.assemble(
                                post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                            )

                    mlp_silu_tile = pl.create_tensor([TOK_TILE, inter], dtype=pl.BF16)
                    for ob in pl.range(mlp_out_blocks_prefill):
                        o0 = ob * MLP_OUT_CHUNK_PREFILL
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(
                                w_gate_chunk_flat, [K_CHUNK, MLP_OUT_CHUNK_PREFILL], [layer_off_h, o0]
                            )
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(
                                    w_gate_chunk_flat,
                                    [K_CHUNK, MLP_OUT_CHUNK_PREFILL],
                                    [layer_off_h + k0, o0],
                                )
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(
                                w_up_chunk_flat, [K_CHUNK, MLP_OUT_CHUNK_PREFILL], [layer_off_h, o0]
                            )
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(
                                    w_up_chunk_flat,
                                    [K_CHUNK, MLP_OUT_CHUNK_PREFILL],
                                    [layer_off_h + k0, o0],
                                )
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_silu_tile = pl.assemble(
                                mlp_silu_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0]
                            )

                    for dob in pl.range(hidden_blocks):
                        d0 = dob * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP):
                            mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK_PREFILL], [0, 0])
                            w_down_chunk_0 = pl.slice(
                                w_down_chunk_flat, [MLP_OUT_CHUNK_PREFILL, K_CHUNK], [layer_off_inter, d0]
                            )
                            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                            for ob in pl.range(1, mlp_out_blocks_prefill):
                                o0 = ob * MLP_OUT_CHUNK_PREFILL
                                mlp_chunk_i = pl.slice(
                                    mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK_PREFILL], [0, o0]
                                )
                                w_down_chunk_i = pl.slice(
                                    w_down_chunk_flat,
                                    [MLP_OUT_CHUNK_PREFILL, K_CHUNK],
                                    [layer_off_inter + o0, d0],
                                )
                                down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            out_chunk = pl.add(
                                down_acc,
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                            )
                            out = pl.assemble(
                                out,
                                pl.cast(out_chunk, target_type=pl.BF16),
                                [b, p0, d0],
                            )

            return out

        # ── L2: per-layer decode ───────────────────────────────────────────────
        @pl.function(type=pl.FunctionType.Orchestration)
        def qwen3_decode_layer(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16],
            input_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            wq_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            wk_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            wv_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            q_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            k_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            post_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            w_gate_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_up_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_down_chunk_flat: pl.Tensor[[chunk_size * inter, hidden], pl.BF16],
            local_layer_idx: pl.Scalar[pl.INT32],
            kv_layer_offset_base: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]:
            user_batch = pl.tensor.dim(hidden_states, 0)
            batch_padded = ((user_batch + BATCH_TILE - 1) // BATCH_TILE) * BATCH_TILE
            cache_rows_per_layer = pl.tensor.dim(k_cache_all, 0) // num_layers

            layer_off_h = local_layer_idx * hidden
            layer_off_inter = local_layer_idx * inter
            global_layer_idx = kv_layer_offset_base + local_layer_idx
            layer_off_cache = global_layer_idx * cache_rows_per_layer

            q_norm_w = pl.slice(q_norm_chunk, [1, head_dim], [local_layer_idx, 0])
            k_norm_w = pl.slice(k_norm_chunk, [1, head_dim], [local_layer_idx, 0])

            q_proj = pl.create_tensor([batch_padded, hidden], dtype=pl.FP32)
            k_proj = pl.create_tensor([batch_padded, kv_hidden], dtype=pl.FP32)
            v_proj = pl.create_tensor([batch_padded, kv_hidden], dtype=pl.FP32)
            q_proj_norm = pl.create_tensor([batch_padded, hidden], dtype=pl.FP32)
            k_proj_norm = pl.create_tensor([batch_padded, kv_hidden], dtype=pl.FP32)

            # Scope 1: input RMSNorm + Q/K/V projection.
            for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                cur_valid = pl.min(BATCH_TILE, user_batch - b0)
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)

                with pl.at(level=pl.Level.CORE_GROUP):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(scope1_hidden_blocks):
                        k0 = kb * SCOPE1_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(
                                hidden_states,
                                [BATCH_TILE, SCOPE1_K_CHUNK],
                                [b0, k0],
                                valid_shape=[cur_valid, SCOPE1_K_CHUNK],
                            ),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, hidden_inv), EPS),
                        [BATCH_TILE, 1],
                    )
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.range(scope1_hidden_blocks):
                        k0 = kb * SCOPE1_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(
                                hidden_states,
                                [BATCH_TILE, SCOPE1_K_CHUNK],
                                [b0, k0],
                                valid_shape=[cur_valid, SCOPE1_K_CHUNK],
                            ),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_chunk, [1, SCOPE1_K_CHUNK], [local_layer_idx, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile = pl.assemble(
                            normed_tile,
                            pl.cast(normed, target_type=pl.BF16),
                            [0, k0],
                        )

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(q_out_blocks, chunk=4):
                        q0 = ob * Q_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                        tile_b = pl.slice(
                            wq_chunk_flat, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [layer_off_h, q0]
                        )
                        q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                        for kb in pl.range(1, scope1_hidden_blocks):
                            k0 = kb * SCOPE1_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(
                                wq_chunk_flat,
                                [SCOPE1_K_CHUNK, Q_OUT_CHUNK],
                                [layer_off_h + k0, q0],
                            )
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(kv_out_blocks, chunk=4):
                        kv0 = ob * KV_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                        tile_wk = pl.slice(
                            wk_chunk_flat, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0]
                        )
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                        for kb in pl.range(1, scope1_hidden_blocks):
                            k0 = kb * SCOPE1_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(
                                wk_chunk_flat,
                                [SCOPE1_K_CHUNK, KV_OUT_CHUNK],
                                [layer_off_h + k0, kv0],
                            )
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

                        tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                        tile_wv = pl.slice(
                            wv_chunk_flat, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0]
                        )
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                        for kb in pl.range(1, scope1_hidden_blocks):
                            k0 = kb * SCOPE1_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                            tile_wv_i = pl.slice(
                                wv_chunk_flat,
                                [SCOPE1_K_CHUNK, KV_OUT_CHUNK],
                                [layer_off_h + k0, kv0],
                            )
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

            # HF-style per-head Q/K norm before RoPE.
            for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for h in pl.range(num_heads):
                        q0 = h * head_dim
                        q_chunk = pl.slice(q_proj, [BATCH_TILE, head_dim], [b0, q0])
                        q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                        q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, head_dim_inv), EPS))
                        q_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(q_chunk, q_inv_rms),
                            q_norm_w,
                        )
                        q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm, [b0, q0])

                    for h in pl.range(num_kv_heads):
                        k0 = h * head_dim
                        k_chunk = pl.slice(k_proj, [BATCH_TILE, head_dim], [b0, k0])
                        k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                        k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, head_dim_inv), EPS))
                        k_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(k_chunk, k_inv_rms),
                            k_norm_w,
                        )
                        k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])

            # Scope 2: RoPE + KV cache update + grouped decode attention.
            attn_out = pl.create_tensor([batch_padded, hidden], dtype=pl.BF16)
            all_q_padded = pl.create_tensor(
                [batch_padded * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16,
            )
            with pl.at(level=pl.Level.CORE_GROUP):
                for idx in pl.range(batch_padded * total_q_groups):
                    all_q_padded = pl.assemble(
                        all_q_padded,
                        pl.cast(
                            pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        ),
                        [idx * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                    )

            for b in pl.parallel(user_batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
                block_table_base = b * max_blocks_per_seq
                slot = pl.tensor.read(slot_mapping, [b])
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot - slot_block * BLOCK_SIZE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ki in pl.parallel(0, num_kv_heads, chunk=8):
                        kv_col = ki * head_dim
                        cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                        k_lo = pl.slice(k_proj_norm, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(k_proj_norm, [1, half_dim], [b, kv_col + half_dim])
                        rot_lo = pl.sub(
                            pl.col_expand_mul(k_lo, cos_lo),
                            pl.col_expand_mul(k_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(k_hi, cos_hi),
                            pl.col_expand_mul(k_lo, sin_hi),
                        )
                        k_cache_all = pl.assemble(
                            k_cache_all,
                            pl.cast(rot_lo, target_type=pl.BF16),
                            [layer_off_cache + cache_row, 0],
                        )
                        k_cache_all = pl.assemble(
                            k_cache_all,
                            pl.cast(rot_hi, target_type=pl.BF16),
                            [layer_off_cache + cache_row, half_dim],
                        )
                        v_cache_all = pl.assemble(
                            v_cache_all,
                            pl.cast(
                                pl.slice(v_proj, [1, head_dim], [b, kv_col]),
                                target_type=pl.BF16,
                            ),
                            [layer_off_cache + cache_row, 0],
                        )
                        q_base = ki * q_per_kv
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_lo = pl.slice(q_proj_norm, [1, half_dim], [b, q_col])
                            q_hi = pl.slice(q_proj_norm, [1, half_dim], [b, q_col + half_dim])
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
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_lo_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, 0],
                            )
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_hi_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, half_dim],
                            )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                attn_row_padded = pl.create_tensor(
                    [1, total_q_groups * Q_HEAD_PAD * head_dim],
                    dtype=pl.BF16,
                )
                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(
                        all_q_padded,
                        [Q_HEAD_PAD, head_dim],
                        [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
                    )
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            k_tile = pl.slice(
                                k_cache_all,
                                [BLOCK_SIZE, head_dim],
                                [layer_off_cache + cache_row0, 0],
                            )
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            s0 = sb * BLOCK_SIZE
                            valid_len = pl.min(BLOCK_SIZE, ctx_len - s0)
                            scores_valid = pl.slice(
                                all_raw_scores,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                                valid_shape=[Q_HEAD_PAD, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            scores = pl.mul(scores_padded, attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)
                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            exp_tile = pl.slice(
                                all_exp_padded,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                            )
                            v_tile = pl.slice(
                                v_cache_all,
                                [BLOCK_SIZE, head_dim],
                                [layer_off_cache + cache_row0, 0],
                            )
                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        oi = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [0, 0])
                        mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [0, 0])
                        li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [0, 0])
                        for sb in pl.range(1, ctx_blocks):
                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(
                                pl.row_expand_mul(oi, alpha),
                                pl.row_expand_mul(oi_tmp_valid, beta),
                            )
                            mi = mi_new
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat_padded = pl.reshape(ctx, [1, Q_HEAD_PAD * head_dim])
                        ctx_flat_padded_bf16 = pl.cast(ctx_flat_padded, target_type=pl.BF16)
                        attn_row_padded = pl.assemble(
                            attn_row_padded,
                            ctx_flat_padded_bf16,
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )

                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    with pl.at(level=pl.Level.CORE_GROUP):
                        ctx_flat_bf16 = pl.slice(
                            attn_row_padded,
                            [1, Q_HEAD_BATCH * head_dim],
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )
                        attn_row = pl.assemble(attn_row, ctx_flat_bf16, [0, q_base * head_dim])

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # Scope 3: Wo + residual + post-RMSNorm + MLP + residual.
            for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                cur_valid = pl.min(BATCH_TILE, user_batch - b0)
                resid1_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.FP32)

                for ob in pl.range(q_out_blocks):
                    o0 = ob * Q_OUT_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w_chunk_0 = pl.slice(
                            wo_chunk_flat, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, o0]
                        )
                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(
                                wo_chunk_flat,
                                [K_CHUNK, Q_OUT_CHUNK],
                                [layer_off_h + k0, o0],
                            )
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        resid = pl.cast(
                            pl.slice(
                                hidden_states,
                                [BATCH_TILE, Q_OUT_CHUNK],
                                [b0, o0],
                                valid_shape=[cur_valid, Q_OUT_CHUNK],
                            ),
                            target_type=pl.FP32,
                        )
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                post_norm_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH_TILE]),
                        )
                    inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS)))

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        post_gamma = pl.slice(post_rms_chunk, [1, K_CHUNK], [local_layer_idx, k0])
                        post_normed = pl.col_expand_mul(
                            pl.row_expand_mul(resid_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                            post_gamma,
                        )
                        normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                        post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                mlp_tile = pl.create_tensor([BATCH_TILE, inter], dtype=pl.BF16)
                for ob in pl.range(mlp_out_blocks_decode):
                    o0 = ob * MLP_OUT_CHUNK_DECODE
                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wg_0 = pl.slice(
                            w_gate_chunk_flat, [K_CHUNK, MLP_OUT_CHUNK_DECODE], [layer_off_h, o0]
                        )
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(
                                w_gate_chunk_flat,
                                [K_CHUNK, MLP_OUT_CHUNK_DECODE],
                                [layer_off_h + k0, o0],
                            )
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wu_0 = pl.slice(
                            w_up_chunk_flat, [K_CHUNK, MLP_OUT_CHUNK_DECODE], [layer_off_h, o0]
                        )
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(
                                w_up_chunk_flat,
                                [K_CHUNK, MLP_OUT_CHUNK_DECODE],
                                [layer_off_h + k0, o0],
                            )
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                for dob in pl.range(hidden_blocks):
                    d0 = dob * K_CHUNK
                    fp32_chunk_gm = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK_DECODE], [0, 0])
                        w_down_chunk_0 = pl.slice(
                            w_down_chunk_flat,
                            [MLP_OUT_CHUNK_DECODE, K_CHUNK],
                            [layer_off_inter, d0],
                        )
                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                        for ob in pl.range(1, mlp_out_blocks_decode):
                            o0 = ob * MLP_OUT_CHUNK_DECODE
                            down_mlp_chunk_bf16 = pl.slice(
                                mlp_tile,
                                [BATCH_TILE, MLP_OUT_CHUNK_DECODE],
                                [0, o0],
                            )
                            w_down_chunk = pl.slice(
                                w_down_chunk_flat,
                                [MLP_OUT_CHUNK_DECODE, K_CHUNK],
                                [layer_off_inter + o0, d0],
                            )
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
                        fp32_chunk_gm = pl.assemble(fp32_chunk_gm, down_acc, [0, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        down_chunk_fp32 = pl.slice(fp32_chunk_gm, [BATCH_TILE, K_CHUNK], [0, 0])
                        resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                        out_chunk = pl.add(down_chunk_fp32, resid_chunk_fp32)
                        out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                        out_chunk_trimmed = pl.slice(
                            out_chunk_cast,
                            [BATCH_TILE, K_CHUNK],
                            [0, 0],
                            valid_shape=[cur_valid, K_CHUNK],
                        )
                        out = pl.assemble(out, out_chunk_trimmed, [b0, d0])

            return out

        # ── L3: unified host orchestrator ──────────────────────────────────────
        #
        # WHY pl.unroll INSTEAD OF pl.range FOR THE LAYER LOOP:
        #   pl.range generates a runtime Python `for` loop in host_orch.py.
        #   Two problems arise when it is combined with in-place tensor writes:
        #
        #   (A) §2.3 violation — runtime scalar conditionals (pl.Scalar[pl.BOOL])
        #       inside a pl.range loop body may be miscompiled by
        #       distributed_codegen.cpp (no VisitExpr_ for the corresponding op),
        #       producing always-true or always-false branches and broken dispatch.
        #
        #   (B) WAW scheduling deadlock — with pl.range every iteration reuses
        #       the SAME dispatch call-site in the generated loop body.  When the
        #       in-place pattern ``out_d = f(out_d, …, out_d)`` is used across
        #       iterations the device scheduler cannot determine WAW order from
        #       the shared call-site alone → deadlock.
        #
        #   pl.unroll(N) compiles to N sequential, *distinct* flat dispatch
        #   calls (different code positions), matching the known-good handwritten
        #   pattern used in l3_two_l2.py and the spike tests.
        #
        # WHY if has_prefill IS OUTSIDE THE LOOP:
        #   Moving the conditional to the outermost level avoids placing ANY
        #   runtime branch inside a pl.unroll body, giving the code generator a
        #   simple top-level branch over two fully-unrolled sequential chains —
        #   the same structure as _build_with_flag / _build_no_flag in l3_two_l2.
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            # Prefill inputs (only used when has_prefill != 0).
            prefill_hidden: pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16],
            prefill_seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            prefill_slot_mapping: pl.Tensor[[SLOT_MAPPING_DYN], pl.INT32],
            # Decode inputs (always used).
            decode_hidden: pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16],
            decode_seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            decode_slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            # Shared stacked weights (chunk_size layers).
            input_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            wq_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            wk_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            wv_chunk_flat: pl.Tensor[[chunk_size * hidden, kv_hidden], pl.BF16],
            q_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            k_norm_chunk: pl.Tensor[[chunk_size, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo_chunk_flat: pl.Tensor[[chunk_size * hidden, hidden], pl.BF16],
            post_rms_chunk: pl.Tensor[[chunk_size, hidden], pl.FP32],
            w_gate_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_up_chunk_flat: pl.Tensor[[chunk_size * hidden, inter], pl.BF16],
            w_down_chunk_flat: pl.Tensor[[chunk_size * inter, hidden], pl.BF16],
            # Layer coordination.
            kv_layer_offset_base: pl.Scalar[pl.INT32],
            has_prefill: pl.Scalar[pl.BOOL],
            # Output buffers.
            prefill_out: pl.Out[pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]],
            decode_out: pl.Out[pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]:
            if has_prefill:
                # ── has_prefill=True: interleaved prefill + decode ──────────
                # Layer 0:
                out_p = self.qwen3_prefill_layer(
                    prefill_hidden, prefill_seq_lens,
                    input_rms_chunk,
                    wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                    q_norm_chunk, k_norm_chunk,
                    rope_cos, rope_sin,
                    block_table, prefill_slot_mapping,
                    k_cache_all, v_cache_all,
                    wo_chunk_flat, post_rms_chunk,
                    w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                    0, kv_layer_offset_base, prefill_out,
                )
                out_d = self.qwen3_decode_layer(
                    decode_hidden,
                    input_rms_chunk,
                    wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                    q_norm_chunk, k_norm_chunk,
                    decode_seq_lens, block_table, decode_slot_mapping,
                    rope_cos, rope_sin,
                    k_cache_all, v_cache_all,
                    wo_chunk_flat, post_rms_chunk,
                    w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                    0, kv_layer_offset_base, decode_out,
                )
                # Layers 1..chunk_size-1 (compile-time unroll — no runtime loop):
                for step in pl.unroll(chunk_size - 1):
                    local_layer_idx = step + 1
                    out_p = self.qwen3_prefill_layer(
                        out_p, prefill_seq_lens,
                        input_rms_chunk,
                        wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                        q_norm_chunk, k_norm_chunk,
                        rope_cos, rope_sin,
                        block_table, prefill_slot_mapping,
                        k_cache_all, v_cache_all,
                        wo_chunk_flat, post_rms_chunk,
                        w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                        local_layer_idx, kv_layer_offset_base, out_p,
                    )
                    out_d = self.qwen3_decode_layer(
                        out_d,
                        input_rms_chunk,
                        wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                        q_norm_chunk, k_norm_chunk,
                        decode_seq_lens, block_table, decode_slot_mapping,
                        rope_cos, rope_sin,
                        k_cache_all, v_cache_all,
                        wo_chunk_flat, post_rms_chunk,
                        w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                        local_layer_idx, kv_layer_offset_base, out_d,
                    )
            else:
                # ── has_prefill=False: decode-only ──────────────────────────
                # Layer 0:
                out_d = self.qwen3_decode_layer(
                    decode_hidden,
                    input_rms_chunk,
                    wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                    q_norm_chunk, k_norm_chunk,
                    decode_seq_lens, block_table, decode_slot_mapping,
                    rope_cos, rope_sin,
                    k_cache_all, v_cache_all,
                    wo_chunk_flat, post_rms_chunk,
                    w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                    0, kv_layer_offset_base, decode_out,
                )
                # Layers 1..chunk_size-1 (compile-time unroll — no runtime loop):
                for step in pl.unroll(chunk_size - 1):
                    local_layer_idx = step + 1
                    out_d = self.qwen3_decode_layer(
                        out_d,
                        input_rms_chunk,
                        wq_chunk_flat, wk_chunk_flat, wv_chunk_flat,
                        q_norm_chunk, k_norm_chunk,
                        decode_seq_lens, block_table, decode_slot_mapping,
                        rope_cos, rope_sin,
                        k_cache_all, v_cache_all,
                        wo_chunk_flat, post_rms_chunk,
                        w_gate_chunk_flat, w_up_chunk_flat, w_down_chunk_flat,
                        local_layer_idx, kv_layer_offset_base, out_d,
                    )
            return out_d

    return Qwen3GenChunked


# Re-export the shared weight-stacking helper (same layout as decode_chunked).
def stack_layer_weights_chunked(
    layers,
    *,
    chunk_size: int,
    hidden: int,
    kv_hidden: int,
    inter: int,
    head_dim: int,
):
    """Host-side helper: stack per-layer weights into chunk_size-sized chunks.

    Returns a list of per-chunk dicts (length = num_layers // chunk_size).
    Weight layout is identical to decode_chunked / prefill_chunked; the same
    chunks are used for both the prefill L2 and decode L2 in gen_chunked.
    """
    import torch

    num_layers = len(layers)
    assert num_layers % chunk_size == 0, (
        f"num_layers={num_layers} must be divisible by chunk_size={chunk_size}"
    )
    num_chunks = num_layers // chunk_size

    def _row_stack(chunk_layers, attr, dim):
        rows = [getattr(layer, attr).view(-1).contiguous() for layer in chunk_layers]
        for i, row in enumerate(rows):
            assert row.numel() == dim, (
                f"layer {i} {attr}: expected {dim} elems, got {row.numel()}"
            )
        return torch.stack(rows, dim=0).contiguous()

    def _flat_stack_kernel(chunk_layers, attr, in_dim, out_dim):
        kernels = []
        for i, layer in enumerate(chunk_layers):
            w = getattr(layer, attr)
            assert w.shape == (in_dim, out_dim), (
                f"layer {i} {attr}: expected shape ({in_dim}, {out_dim}), "
                f"got {tuple(w.shape)}"
            )
            kernels.append(w.contiguous())
        stacked = torch.stack(kernels, dim=0).contiguous()
        return stacked.view(len(chunk_layers) * in_dim, out_dim).contiguous()

    chunks: list[dict] = []
    for c in range(num_chunks):
        cl = layers[c * chunk_size : (c + 1) * chunk_size]
        chunks.append({
            "input_rms_chunk":   _row_stack(cl, "input_rms_weight", hidden),
            "wq_chunk_flat":     _flat_stack_kernel(cl, "wq", hidden, hidden),
            "wk_chunk_flat":     _flat_stack_kernel(cl, "wk", hidden, kv_hidden),
            "wv_chunk_flat":     _flat_stack_kernel(cl, "wv", hidden, kv_hidden),
            "q_norm_chunk":      _row_stack(cl, "q_norm_weight", head_dim),
            "k_norm_chunk":      _row_stack(cl, "k_norm_weight", head_dim),
            "wo_chunk_flat":     _flat_stack_kernel(cl, "wo", hidden, hidden),
            "post_rms_chunk":    _row_stack(cl, "post_rms_weight", hidden),
            "w_gate_chunk_flat": _flat_stack_kernel(cl, "w_gate", hidden, inter),
            "w_up_chunk_flat":   _flat_stack_kernel(cl, "w_up", hidden, inter),
            "w_down_chunk_flat": _flat_stack_kernel(cl, "w_down", inter, hidden),
        })
    return chunks
