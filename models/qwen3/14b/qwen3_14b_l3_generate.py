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
    host_orch -- single entry point for all generation steps.
    Dispatches two L2 functions (each iterates all layers internally):
        1. qwen3_prefill_all (L2, Orchestration) -- if has_prefill=1
           Processes all prompt positions across all layers; writes KV.
        2. qwen3_decode_all  (L2, Orchestration) -- always
           Processes one decode token across all layers; reads KV.

Each L2 contains ``for layer_idx in pl.range(num_layers)`` internally,
so L3 dispatches each L2 exactly once per generation step.

has_prefill flag
----------------
has_prefill=1 (step 0, combined prefill + first decode):
    prefill_hidden  : [batch, max_seq, hidden] -- embedded prompt tokens
    prefill_seq_lens: [batch]                  -- actual prompt lengths
    prefill_slot_mapping: [batch * max_seq]    -- physical KV slots for all prompt positions
    decode_hidden   : [batch, hidden]          -- embed(last_prompt_token)
    decode_seq_lens : [batch]                  -- same as prefill_seq_lens (= N)
    decode_slot_mapping : [batch]              -- slot for position N-1 (last prompt slot)
    After: KV cache has N entries (0..N-1); decode_out holds the hidden state for
    predicting the first new token (same semantics as prefill_out[:, N-1, :]).

has_prefill=0 (steps 1+, pure decode):
    prefill_hidden / prefill_slot_mapping are unused dummy tensors.
    decode_hidden   : [batch, hidden]  -- embed(current_decode_token)
    decode_seq_lens : [batch]          -- N+t (context length including current token)
    decode_slot_mapping : [batch]      -- slot for position N+t-1
    After: KV cache grows by 1 per step; decode_out holds next-token hidden state.

Stacking layout: row-stacked 1D weights ([num_layers, dim]) and
    flat-stacked 2D weights ([num_layers * d_in, d_out]); KV cache is
    full-stacked across all num_layers. See ``stack_layer_weights_full``
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
SEQ_TILE = 128
SB_BATCH = 64
BLOCK_SIZE = SEQ_TILE
MLP_OUT_CHUNK_PREFILL = 128

# Decode-specific tiling.
SCOPE1_K_CHUNK = 512
MLP_OUT_CHUNK_DECODE = 256


def build_qwen3_14b_l3_generate_program(
    num_layers: int = 40,
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    # Generation loop parameters.
    # max_new_tokens is compile-time so pl.unroll can expand the decode loop.
    max_new_tokens: int = 256,
    # padded_vocab must be a multiple of 64 (VOCAB_CHUNK alignment).
    padded_vocab: int = 152064,
    # page_size must equal SEQ_TILE (BLOCK_SIZE) used for KV cache indexing.
    page_size: int = SEQ_TILE,
):
    if page_size != SEQ_TILE:
        raise ValueError(
            f"page_size={page_size} must equal SEQ_TILE={SEQ_TILE} "
            f"(BLOCK_SIZE used for KV cache indexing in the L3 kernel)"
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

    # Decode cache layout (compile-time constant, matching baseline).
    layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE

    # Decode tiling derived values.
    scope1_hidden_blocks = hidden // SCOPE1_K_CHUNK
    mlp_out_blocks_decode = inter // MLP_OUT_CHUNK_DECODE

    # Final-RMS / LM-head tiling derived values.
    VOCAB_CHUNK = 64
    vocab_blocks = padded_vocab // VOCAB_CHUNK

    @pl.program
    class Qwen3GenChunked:

        # ── L2: all-layers prefill ─────────────────────────────────────────────
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_prefill_all(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            input_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            wq: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            wk: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            q_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[SLOT_MAPPING_DYN], pl.INT32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            w_gate: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_up: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_down: pl.Tensor[[num_layers * inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]:
            user_batch = pl.tensor.dim(hidden_states, 0)
            cache_rows_per_layer = pl.tensor.dim(k_cache_all, 0) // num_layers

            # Copy input into current_hidden so its type uses runtime user_batch
            # (matches next_hidden's type for reassignment inside pl.range).
            current_hidden = pl.create_tensor([user_batch, max_seq, hidden], dtype=pl.BF16)
            for b in pl.parallel(0, user_batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_copy_hidden"):
                            chunk = pl.slice(
                                hidden_states, [1, TOK_TILE, K_CHUNK], [b, p0, k0]
                            )
                            current_hidden = pl.assemble(current_hidden, chunk, [b, p0, k0])

            for layer_idx in pl.range(num_layers):
                layer_off_h = layer_idx * hidden
                layer_off_inter = layer_idx * inter
                layer_off_cache = layer_idx * cache_rows_per_layer

                q_norm_w = pl.slice(q_norm_weight, [1, head_dim], [layer_idx, 0])
                k_norm_w = pl.slice(k_norm_weight, [1, head_dim], [layer_idx, 0])

                next_hidden = pl.create_tensor([user_batch, max_seq, hidden], dtype=pl.BF16)

                for b in pl.parallel(0, user_batch, 1):
                    seq_len_b = pl.tensor.read(seq_lens, [b])
                    tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                    for p0_idx in pl.range(tok_blocks):
                        p0 = p0_idx * TOK_TILE
                        valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                        # ── Scope 1: input RMSNorm + Q/K/V projection ──
                        normed_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rmsnorm"):
                            partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                            for kb in pl.range(hidden_blocks):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.reshape(
                                    pl.cast(
                                        pl.slice(
                                            current_hidden, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
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
                                            current_hidden, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                            valid_shape=[1, valid_tok, K_CHUNK],
                                        ),
                                        target_type=pl.FP32,
                                    ),
                                    [TOK_TILE, K_CHUNK],
                                )
                                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                                normed_tile = pl.assemble(
                                    normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                                )

                        q_proj_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_q_proj"):
                            for ob in pl.parallel(q_out_blocks, chunk=4):
                                q0 = ob * Q_OUT_CHUNK
                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, q0])
                                q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_w_i = pl.slice(
                                        wq, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h + k0, q0]
                                    )
                                    q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                                q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                        k_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                        v_proj_tile = pl.create_tensor([TOK_TILE, kv_hidden], dtype=pl.FP32)
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_kv_proj"):
                            for ob in pl.parallel(kv_out_blocks, chunk=4):
                                kv0 = ob * KV_OUT_CHUNK
                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0])
                                k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_wk_i = pl.slice(
                                        wk, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h + k0, kv0]
                                    )
                                    k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                                k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                                tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0])
                                v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_wv_i = pl.slice(
                                        wv, [K_CHUNK, KV_OUT_CHUNK], [layer_off_h + k0, kv0]
                                    )
                                    v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                                v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_q_norm"):
                            for qh in pl.parallel(0, num_heads, chunk=num_heads):
                                q_col = qh * head_dim
                                q_head = pl.slice(q_proj_tile, [TOK_TILE, head_dim], [0, q_col])
                                q_sq = pl.reshape(pl.row_sum(pl.mul(q_head, q_head)), [TOK_TILE, 1])
                                q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, head_dim_inv), EPS)))
                                q_normed = pl.col_expand_mul(pl.row_expand_mul(q_head, q_inv_rms), q_norm_w)
                                q_proj_tile = pl.assemble(q_proj_tile, q_normed, [0, q_col])
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_k_norm"):
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
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_q_pad"):
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
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_rope_kv_cache"):
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

                                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_qk_matmul"):
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

                                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_softmax"):
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

                                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_sv_matmul"):
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

                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_online_softmax_init"):
                                    oi = pl.full([Q_HEAD_BATCH_PAD, head_dim], dtype=pl.FP32, value=0.0)
                                    li_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                    li = pl.reshape(li_flat, [Q_HEAD_BATCH_PAD, 1])
                                    mi_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                                    mi = pl.reshape(mi_flat, [Q_HEAD_BATCH_PAD, 1])

                                for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_online_softmax"):
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
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attention_context"):
                                    ctx = pl.row_expand_div(oi, li)
                                    ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attention_writeback"):
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
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_out_proj"):
                                tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_w = pl.slice(
                                    wo, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, o0]
                                )
                                o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_w_i = pl.slice(
                                        wo, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h + k0, o0]
                                    )
                                    o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
                                resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_out_proj_residual"):
                                resid_chunk = pl.reshape(
                                    pl.cast(
                                        pl.slice(
                                            current_hidden,
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
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_post_rmsnorm"):
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
                                gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                                normed = pl.col_expand_mul(
                                    pl.row_expand_mul(x_chunk, post_inv_rms), gamma
                                )
                                post_norm_tile = pl.assemble(
                                    post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                                )

                        mlp_silu_tile = pl.create_tensor([TOK_TILE, inter], dtype=pl.BF16)
                        for ob in pl.range(mlp_out_blocks_prefill):
                            o0 = ob * MLP_OUT_CHUNK_PREFILL
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gate_proj"):
                                pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                wg0 = pl.slice(
                                    w_gate, [K_CHUNK, MLP_OUT_CHUNK_PREFILL], [layer_off_h, o0]
                                )
                                gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    wgi = pl.slice(
                                        w_gate,
                                        [K_CHUNK, MLP_OUT_CHUNK_PREFILL],
                                        [layer_off_h + k0, o0],
                                    )
                                    gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_up_proj"):
                                pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                wu0 = pl.slice(
                                    w_up, [K_CHUNK, MLP_OUT_CHUNK_PREFILL], [layer_off_h, o0]
                                )
                                up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                                for kb in pl.range(1, hidden_blocks):
                                    k0 = kb * K_CHUNK
                                    pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    wui = pl.slice(
                                        w_up,
                                        [K_CHUNK, MLP_OUT_CHUNK_PREFILL],
                                        [layer_off_h + k0, o0],
                                    )
                                    up_acc = pl.matmul_acc(up_acc, pci, wui)

                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_silu"):
                                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                                mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                                mlp_silu_tile = pl.assemble(
                                    mlp_silu_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0]
                                )

                        for dob in pl.range(hidden_blocks):
                            d0 = dob * K_CHUNK
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_down_proj"):
                                mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK_PREFILL], [0, 0])
                                w_down_chunk_0 = pl.slice(
                                    w_down, [MLP_OUT_CHUNK_PREFILL, K_CHUNK], [layer_off_inter, d0]
                                )
                                down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                                for ob in pl.range(1, mlp_out_blocks_prefill):
                                    o0 = ob * MLP_OUT_CHUNK_PREFILL
                                    mlp_chunk_i = pl.slice(
                                        mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK_PREFILL], [0, o0]
                                    )
                                    w_down_chunk_i = pl.slice(
                                        w_down,
                                        [MLP_OUT_CHUNK_PREFILL, K_CHUNK],
                                        [layer_off_inter + o0, d0],
                                    )
                                    down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)

                            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_down_proj_residual"):
                                out_chunk = pl.add(
                                    down_acc,
                                    pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                                )
                                next_hidden = pl.assemble(
                                    next_hidden,
                                    pl.cast(out_chunk, target_type=pl.BF16),
                                    [b, p0, d0],
                                )


                current_hidden = next_hidden

            # Copy final layer output to out.
            for b in pl.parallel(0, user_batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_copy_out"):
                            chunk = pl.slice(
                                current_hidden, [1, TOK_TILE, K_CHUNK], [b, p0, k0]
                            )
                            out = pl.assemble(out, chunk, [b, p0, k0])

            return out

        # ── L2: all-layers decode ──────────────────────────────────────────────
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode_all(
            self,
            hidden_states: pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            wq: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            wk: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            q_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            w_gate: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_up: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_down: pl.Tensor[[num_layers * inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]:
            user_batch = pl.tensor.dim(hidden_states, 0)
            batch_padded = ((user_batch + BATCH_TILE - 1) // BATCH_TILE) * BATCH_TILE

            # Copy input into current_hidden (decode_fwd.py pattern).
            current_hidden = pl.create_tensor([batch, hidden], dtype=pl.BF16)
            for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                cur_valid = pl.min(BATCH_TILE, user_batch - b0)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_copy_hidden"):
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        hidden_chunk = pl.slice(
                            hidden_states,
                            [BATCH_TILE, K_CHUNK],
                            [b0, k0],
                            valid_shape=[cur_valid, K_CHUNK],
                        )
                        current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, k0])

            for layer_idx in pl.range(num_layers):
                layer_off_h = layer_idx * hidden
                layer_off_inter = layer_idx * inter
                layer_off_cache = layer_idx * layer_cache_rows

                q_norm_w = pl.slice(q_norm_weight, [1, head_dim], [layer_idx, 0])
                k_norm_w = pl.slice(k_norm_weight, [1, head_dim], [layer_idx, 0])

                next_hidden = pl.create_tensor([batch, hidden], dtype=pl.BF16)

                q_proj = pl.create_tensor([batch, hidden], dtype=pl.FP32)
                k_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
                v_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
                q_proj_norm = pl.create_tensor([batch, hidden], dtype=pl.FP32)
                k_proj_norm = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)

                # Scope 1: input RMSNorm + Q/K/V projection.
                for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                    cur_valid = pl.min(BATCH_TILE, user_batch - b0)
                    normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_rmsnorm"):
                        partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(scope1_hidden_blocks):
                            k0 = kb * SCOPE1_K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(
                                    current_hidden,
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
                                    current_hidden,
                                    [BATCH_TILE, SCOPE1_K_CHUNK],
                                    [b0, k0],
                                    valid_shape=[cur_valid, SCOPE1_K_CHUNK],
                                ),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, SCOPE1_K_CHUNK], [layer_idx, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            normed_tile = pl.assemble(
                                normed_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_q_proj"):
                        for ob in pl.parallel(q_out_blocks, chunk=4):
                            q0 = ob * Q_OUT_CHUNK
                            tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                            tile_b = pl.slice(
                                wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [layer_off_h, q0]
                            )
                            q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                            for kb in pl.range(1, scope1_hidden_blocks):
                                k0 = kb * SCOPE1_K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                                tile_b_i = pl.slice(
                                    wq,
                                    [SCOPE1_K_CHUNK, Q_OUT_CHUNK],
                                    [layer_off_h + k0, q0],
                                )
                                q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                            q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_kv_proj"):
                        for ob in pl.parallel(kv_out_blocks, chunk=4):
                            kv0 = ob * KV_OUT_CHUNK
                            tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                            tile_wk = pl.slice(
                                wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0]
                            )
                            k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                            for kb in pl.range(1, scope1_hidden_blocks):
                                k0 = kb * SCOPE1_K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                                tile_wk_i = pl.slice(
                                    wk,
                                    [SCOPE1_K_CHUNK, KV_OUT_CHUNK],
                                    [layer_off_h + k0, kv0],
                                )
                                k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                            k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

                            tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                            tile_wv = pl.slice(
                                wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_off_h, kv0]
                            )
                            v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                            for kb in pl.range(1, scope1_hidden_blocks):
                                k0 = kb * SCOPE1_K_CHUNK
                                tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                                tile_wv_i = pl.slice(
                                    wv,
                                    [SCOPE1_K_CHUNK, KV_OUT_CHUNK],
                                    [layer_off_h + k0, kv0],
                                )
                                v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                            v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

                # HF-style per-head Q/K norm before RoPE.
                for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_qk_norm"):
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
                attn_out = pl.create_tensor([batch, hidden], dtype=pl.BF16)
                all_q_padded = pl.create_tensor(
                    [batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16,
                )
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_q_pad"):
                    for idx in pl.range(batch * total_q_groups):
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

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_rope_kv_cache"):
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
                    all_raw_scores = pl.create_tensor(
                        [total_q_groups * max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32,
                    )
                    all_exp_padded = pl.create_tensor(
                        [total_q_groups * max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16,
                    )
                    all_oi_tmp = pl.create_tensor(
                        [total_q_groups * max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32,
                    )
                    all_cur_mi = pl.create_tensor(
                        [total_q_groups * max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32,
                    )
                    all_cur_li = pl.create_tensor(
                        [total_q_groups * max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32,
                    )

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_qk_matmul"):
                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            q_padded = pl.slice(
                                all_q_padded,
                                [Q_HEAD_PAD, head_dim],
                                [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
                            )
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
                                all_raw_scores = pl.assemble(
                                    all_raw_scores, raw_scores,
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_softmax"):
                        for gi in pl.range(total_q_groups):
                            for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                s0 = sb * BLOCK_SIZE
                                valid_len = pl.min(BLOCK_SIZE, ctx_len - s0)
                                scores_valid = pl.slice(
                                    all_raw_scores,
                                    [Q_HEAD_PAD, BLOCK_SIZE],
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                    valid_shape=[Q_HEAD_PAD, valid_len],
                                )
                                scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                scores = pl.mul(scores_padded, attn_scale)
                                cur_mi = pl.row_max(scores)
                                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                                cur_li = pl.row_sum(exp_scores_fp32)
                                all_exp_padded = pl.assemble(
                                    all_exp_padded, exp_scores_bf16,
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )
                                all_cur_mi = pl.assemble(
                                    all_cur_mi, cur_mi,
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )
                                all_cur_li = pl.assemble(
                                    all_cur_li, cur_li,
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_sv_matmul"):
                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                block_table_idx = block_table_base + sb
                                pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                                cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                                exp_tile = pl.slice(
                                    all_exp_padded,
                                    [Q_HEAD_PAD, BLOCK_SIZE],
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )
                                v_tile = pl.slice(
                                    v_cache_all,
                                    [BLOCK_SIZE, head_dim],
                                    [layer_off_cache + cache_row0, 0],
                                )
                                oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                all_oi_tmp = pl.assemble(
                                    all_oi_tmp, oi_tmp,
                                    [(gi * max_ctx_blocks + sb) * Q_HEAD_PAD, 0],
                                )

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_online_softmax"):
                        for gi in pl.range(total_q_groups):
                            base = gi * max_ctx_blocks * Q_HEAD_PAD
                            oi = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [base, 0])
                            mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [base, 0])
                            li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [base, 0])
                            for sb in pl.range(1, ctx_blocks):
                                off = base + sb * Q_HEAD_PAD
                                oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [off, 0])
                                cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [off, 0])
                                cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [off, 0])
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

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_attention_writeback"):
                        for gi in pl.range(total_q_groups):
                            kvh = gi // q_groups
                            qg = gi - kvh * q_groups
                            q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
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
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_out_proj"):
                            a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                            w_chunk_0 = pl.slice(
                                wo, [K_CHUNK, Q_OUT_CHUNK], [layer_off_h, o0]
                            )
                            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                                w_chunk = pl.slice(
                                    wo,
                                    [K_CHUNK, Q_OUT_CHUNK],
                                    [layer_off_h + k0, o0],
                                )
                                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_out_proj_residual"):
                            resid = pl.cast(
                                pl.slice(
                                    current_hidden,
                                    [BATCH_TILE, Q_OUT_CHUNK],
                                    [b0, o0],
                                    valid_shape=[cur_valid, Q_OUT_CHUNK],
                                ),
                                target_type=pl.FP32,
                            )
                            resid_sum = pl.add(o_acc, resid)
                            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                    post_norm_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_post_rmsnorm"):
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
                            post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                            post_normed = pl.col_expand_mul(
                                pl.row_expand_mul(resid_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                                post_gamma,
                            )
                            normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                            post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                    mlp_tile = pl.create_tensor([BATCH_TILE, inter], dtype=pl.BF16)
                    for ob in pl.range(mlp_out_blocks_decode):
                        o0 = ob * MLP_OUT_CHUNK_DECODE
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_gate_proj"):
                            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                            wg_0 = pl.slice(
                                w_gate, [K_CHUNK, MLP_OUT_CHUNK_DECODE], [layer_off_h, o0]
                            )
                            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                                wg = pl.slice(
                                    w_gate,
                                    [K_CHUNK, MLP_OUT_CHUNK_DECODE],
                                    [layer_off_h + k0, o0],
                                )
                                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_up_proj"):
                            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                            wu_0 = pl.slice(
                                w_up, [K_CHUNK, MLP_OUT_CHUNK_DECODE], [layer_off_h, o0]
                            )
                            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                                wu = pl.slice(
                                    w_up,
                                    [K_CHUNK, MLP_OUT_CHUNK_DECODE],
                                    [layer_off_h + k0, o0],
                                )
                                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="decode_silu"):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                            mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                    for dob in pl.range(hidden_blocks):
                        d0 = dob * K_CHUNK
                        fp32_chunk_gm = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_down_proj"):
                            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK_DECODE], [0, 0])
                            w_down_chunk_0 = pl.slice(
                                w_down,
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
                                    w_down,
                                    [MLP_OUT_CHUNK_DECODE, K_CHUNK],
                                    [layer_off_inter + o0, d0],
                                )
                                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
                            fp32_chunk_gm = pl.assemble(fp32_chunk_gm, down_acc, [0, 0])

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_down_proj_residual"):
                            down_chunk_fp32 = pl.slice(fp32_chunk_gm, [BATCH_TILE, K_CHUNK], [0, 0])
                            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                            out_chunk = pl.add(down_chunk_fp32, resid_chunk_fp32)
                            out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                            next_hidden = pl.assemble(next_hidden, out_chunk_cast, [b0, d0])


                current_hidden = next_hidden

            # Copy final layer output to out.
            for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
                cur_valid = pl.min(BATCH_TILE, user_batch - b0)
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_copy_out"):
                        chunk = pl.slice(
                            current_hidden, [BATCH_TILE, K_CHUNK], [b0, k0],
                            valid_shape=[cur_valid, K_CHUNK],
                        )
                        out = pl.assemble(out, chunk, [b0, k0])

            return out

        # ── L2: final RMSNorm ──────────────────────────────────────────────────
        # Applied to the padded decode output [BATCH_TILE, hidden] before lm_head.
        # BATCH_TILE == 16 == _LOGITS_BATCH_TILE; rows beyond actual_batch stay
        # zero (padding contract satisfied by the executor).
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_final_rms(
            self,
            x: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            gamma: pl.Tensor[[1, hidden], pl.FP32],
            out: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
        ) -> pl.Tensor[[BATCH_TILE, hidden], pl.BF16]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
                for b0 in pl.range(0, BATCH_TILE, BATCH_TILE):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                [1, BATCH_TILE],
                            ),
                        )
                    inv_rms = pl.reshape(
                        pl.rsqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS)),
                        [BATCH_TILE, 1],
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        g = pl.slice(gamma, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, inv_rms),
                            g,
                        )
                        out = pl.assemble(
                            out, pl.cast(normed, target_type=pl.BF16), [b0, k0]
                        )
            return out

        # ── L2: LM-head projection ──────────────────────────────────────────────
        # Projects the RMS-normed hidden state to vocabulary logits.
        # Weight layout: [padded_vocab, hidden] BF16 (HuggingFace nn.Linear).
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_lm_head(
            self,
            hidden_in: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            lm_head_weight: pl.Tensor[[padded_vocab, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="lm_head"):
                for b0 in pl.range(0, BATCH_TILE, BATCH_TILE):
                    for ob in pl.parallel(vocab_blocks, chunk=8):
                        o0 = ob * VOCAB_CHUNK
                        h0 = pl.slice(hidden_in, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w0 = pl.slice(lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, 0])
                        acc = pl.matmul(h0, w0, out_dtype=pl.FP32, b_trans=True)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            h_chunk = pl.slice(hidden_in, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(
                                lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, k0]
                            )
                            acc = pl.matmul_acc(acc, h_chunk, w_chunk, b_trans=True)
                        out = pl.assemble(out, acc, [b0, o0])
            return out

        # ── L2: fused final-RMSNorm + LM-head (single chip task) ───────────────
        # Combines qwen3_final_rms and qwen3_lm_head into one dispatch, saving
        # one full init_runtime + validate cycle (~20 ms at PTO2_RING_HEAP=512 MB).
        # rms_normed acts as an HBM scratch buffer between the two compute phases;
        # it is passed from host_orch but its post-call value is never read
        # externally (write-only scratch, InOutUseDiscipline satisfied).
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_rms_lmhead(
            self,
            x: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            gamma: pl.Tensor[[1, hidden], pl.FP32],
            lm_head_weight: pl.Tensor[[padded_vocab, hidden], pl.BF16],
            rms_normed: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
            out: pl.Out[pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32]],
        ) -> pl.Tuple[
            pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32],
        ]:
            # Phase 1 – RMSNorm: identical body to qwen3_final_rms.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_lmhead_rmsnorm"):
                for b0 in pl.range(0, BATCH_TILE, BATCH_TILE):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                [1, BATCH_TILE],
                            ),
                        )
                    inv_rms = pl.reshape(
                        pl.rsqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS)),
                        [BATCH_TILE, 1],
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(x, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        g = pl.slice(gamma, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, inv_rms),
                            g,
                        )
                        rms_normed = pl.assemble(
                            rms_normed, pl.cast(normed, target_type=pl.BF16), [b0, k0]
                        )
            # Phase 2 – LM-head GEMM: reads rms_normed written above (HBM).
            # Body identical to qwen3_lm_head with hidden_in → rms_normed.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="rms_lmhead_lm_head"):
                for b0 in pl.range(0, BATCH_TILE, BATCH_TILE):
                    for ob in pl.parallel(vocab_blocks, chunk=8):
                        o0 = ob * VOCAB_CHUNK
                        h0 = pl.slice(rms_normed, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w0 = pl.slice(lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, 0])
                        acc = pl.matmul(h0, w0, out_dtype=pl.FP32, b_trans=True)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            h_chunk = pl.slice(
                                rms_normed, [BATCH_TILE, K_CHUNK], [b0, k0]
                            )
                            w_chunk = pl.slice(
                                lm_head_weight, [VOCAB_CHUNK, K_CHUNK], [o0, k0]
                            )
                            acc = pl.matmul_acc(acc, h_chunk, w_chunk, b_trans=True)
                        out = pl.assemble(out, acc, [b0, o0])
            return rms_normed, out

        # ── HOST SubWorker: sample & prepare next decode inputs ─────────────────
        # Placeholder body — the actual implementation (closure over shared-memory
        # tensors) is injected by the executor at runtime before worker.run().
        # Parameter declarations here define the TaskArgs structure that
        # host_orch.py uses for dependency tracking:
        #   logits_padded  INPUT  — wait for lm_head chip task to finish
        #   decode_hidden  OUTPUT — next decode step's input depends on this write
        #   decode_seq_lens OUTPUT — next decode step waits for seq-len update
        #   decode_slot_mapping OUTPUT — next decode step waits for slot update
        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def sample_and_prepare(
            logits_padded: pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32],
            # decode_seq_lens / decode_slot_mapping are read by the executor
            # implementation and treated as updated-in-place at runtime.  Declaring
            # them as plain INPUT here avoids the InOutUseDiscipline requirement to
            # capture their return values; ordering still works because
            # decode_hidden (pl.Out below) forces the next decode_all to wait.
            decode_seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            decode_slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            # decode_hidden is the single Out tensor; the returned value creates
            # the RAW dependency chain between consecutive decode steps.
            decode_hidden: pl.Out[pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]],
        ) -> pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]:
            pass

        # ── L3: unified host orchestrator ──────────────────────────────────────
        #
        # Each L2 function iterates all num_layers internally via pl.range.
        # L3 dispatches prefill_all (step 0 only) + decode_all + final_rms +
        # lm_head + sample_and_prepare once, then repeats decode_all + final_rms
        # + lm_head + sample_and_prepare for max_new_tokens iterations via
        # pl.unroll (compile-time loop expansion).
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            # Prefill inputs (only used when has_prefill != 0).
            prefill_hidden: pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16],
            prefill_seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            prefill_slot_mapping: pl.Tensor[[SLOT_MAPPING_DYN], pl.INT32],
            # Decode inputs (updated in-place between steps by sample_and_prepare).
            decode_hidden: pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16],
            decode_seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            decode_slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            # Stacked weights (all num_layers).
            input_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            wq: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            wk: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[num_layers * hidden, kv_hidden], pl.BF16],
            q_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[num_layers, head_dim], pl.FP32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            k_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            v_cache_all: pl.Tensor[[KV_CACHE_ROWS_ALL_DYN, head_dim], pl.BF16],
            wo: pl.Tensor[[num_layers * hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[num_layers, hidden], pl.FP32],
            w_gate: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_up: pl.Tensor[[num_layers * hidden, inter], pl.BF16],
            w_down: pl.Tensor[[num_layers * inter, hidden], pl.BF16],
            # Runtime flags.
            has_prefill: pl.Scalar[pl.BOOL],
            # Output buffers.
            prefill_out: pl.Out[pl.Tensor[[USER_BATCH_DYN, max_seq, hidden], pl.BF16]],
            decode_out: pl.Out[pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]],
            # Final-RMS + LM-head tensors (shared with decode_out storage).
            # rms_x is the padded [BATCH_TILE, hidden] view of the same buffer as
            # decode_out; BATCH_TILE == 16 == _LOGITS_BATCH_TILE; no copy needed.
            rms_x: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            final_norm_weight: pl.Tensor[[1, hidden], pl.FP32],
            rms_normed: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
            lm_head_weight_t: pl.Tensor[[padded_vocab, hidden], pl.BF16],
            logits_padded: pl.Out[pl.Tensor[[BATCH_TILE, padded_vocab], pl.FP32]],
        ) -> pl.Tensor[[USER_BATCH_DYN, hidden], pl.BF16]:
            # ── Step 0: optional prefill + first decode + RMS + LM-head + sample ──
            # pypto InOutUseDiscipline: once a variable is passed as Out/InOut,
            # its "post-call" return value must be used for all subsequent reads.
            # Convention used here:
            #   out_d      — post-call decode_out    (updated by qwen3_decode_all)
            #   out_rms    — post-call rms_normed    (returned by qwen3_rms_lmhead[0])
            #   out_lm     — post-call logits_padded (returned by qwen3_rms_lmhead[1])
            #   upd_hidden — post-call decode_hidden (updated by sample_and_prepare)
            # qwen3_rms_lmhead returns (rms_normed, logits) as a pl.Tuple so that
            # both Out params satisfy InOutUseDiscipline at the call site.
            if has_prefill:
                self.qwen3_prefill_all(
                    prefill_hidden, prefill_seq_lens,
                    input_rms_weight, wq, wk, wv,
                    q_norm_weight, k_norm_weight,
                    rope_cos, rope_sin,
                    block_table, prefill_slot_mapping,
                    k_cache_all, v_cache_all,
                    wo, post_rms_weight,
                    w_gate, w_up, w_down,
                    prefill_out,
                )
            out_d = self.qwen3_decode_all(
                decode_hidden,
                input_rms_weight, wq, wk, wv,
                q_norm_weight, k_norm_weight,
                decode_seq_lens, block_table, decode_slot_mapping,
                rope_cos, rope_sin,
                k_cache_all, v_cache_all,
                wo, post_rms_weight,
                w_gate, w_up, w_down,
                decode_out,
            )
            out_rms, out_lm = self.qwen3_rms_lmhead(
                rms_x, final_norm_weight, lm_head_weight_t, rms_normed, logits_padded
            )
            # sample_and_prepare returns the updated decode_hidden (upd_hidden).
            # All subsequent decode_all calls use upd_hidden (not decode_hidden)
            # so that the RAW chain  sample_N.output → decode_{N+1}.input  is
            # visible to the runtime dependency tracker.
            upd_hidden = self.sample_and_prepare(
                out_lm, decode_seq_lens, decode_slot_mapping, decode_hidden,
            )

            # ── Steps 1..max_new_tokens-1: pure decode loop ────────────────────
            # The initial block above already generated token 1 (step 0).
            # This loop generates tokens 2..max_new_tokens (steps 1..max_new_tokens-1).
            # Total tokens = 1 + (max_new_tokens - 1) = max_new_tokens.  Using
            # pl.unroll(max_new_tokens) here would produce max_new_tokens+1 tokens
            # and leave the process blocked on the extra chip-level iteration.
            #
            # In-place re-use of out_d / out_rms / out_lm / upd_hidden satisfies
            # InOutUseDiscipline: each iteration reads the previous Out return
            # value and passes it as the next Out argument.
            # RAW chain per step:
            #   decode_N     (writes out_d)
            #   → rms_lmhead_N (reads rms_x, returns out_rms + out_lm)
            #   → sample_N     (reads out_lm, writes upd_hidden)
            #   → decode_{N+1} (reads upd_hidden) [explicit RAW dependency]
            for _ in pl.range(max_new_tokens - 1):
                out_d = self.qwen3_decode_all(
                    upd_hidden,
                    input_rms_weight, wq, wk, wv,
                    q_norm_weight, k_norm_weight,
                    decode_seq_lens, block_table, decode_slot_mapping,
                    rope_cos, rope_sin,
                    k_cache_all, v_cache_all,
                    wo, post_rms_weight,
                    w_gate, w_up, w_down,
                    out_d,
                )
                out_rms, out_lm = self.qwen3_rms_lmhead(
                    rms_x, final_norm_weight, lm_head_weight_t, out_rms, out_lm
                )
                upd_hidden = self.sample_and_prepare(
                    out_lm, decode_seq_lens, decode_slot_mapping, upd_hidden,
                )

            return out_d

    return Qwen3GenChunked


def stack_layer_weights_full(
    layers,
    *,
    hidden: int,
    kv_hidden: int,
    inter: int,
    head_dim: int,
):
    """Host-side helper: stack per-layer weights into full-model tensors.

    Returns a single dict with shapes [num_layers * hidden, ...] etc.,
    suitable for passing directly to host_orch.
    """
    import torch

    num_layers = len(layers)

    def _row_stack(attr, dim):
        rows = [getattr(layer, attr).view(-1).contiguous() for layer in layers]
        for i, row in enumerate(rows):
            assert row.numel() == dim, (
                f"layer {i} {attr}: expected {dim} elems, got {row.numel()}"
            )
        return torch.stack(rows, dim=0).contiguous()

    def _flat_stack_kernel(attr, in_dim, out_dim):
        kernels = []
        for i, layer in enumerate(layers):
            w = getattr(layer, attr)
            assert w.shape == (in_dim, out_dim), (
                f"layer {i} {attr}: expected shape ({in_dim}, {out_dim}), "
                f"got {tuple(w.shape)}"
            )
            kernels.append(w.contiguous())
        stacked = torch.stack(kernels, dim=0).contiguous()
        return stacked.view(num_layers * in_dim, out_dim).contiguous()

    return {
        "input_rms_weight":  _row_stack("input_rms_weight", hidden),
        "wq":                _flat_stack_kernel("wq", hidden, hidden),
        "wk":                _flat_stack_kernel("wk", hidden, kv_hidden),
        "wv":                _flat_stack_kernel("wv", hidden, kv_hidden),
        "q_norm_weight":     _row_stack("q_norm_weight", head_dim),
        "k_norm_weight":     _row_stack("k_norm_weight", head_dim),
        "wo":                _flat_stack_kernel("wo", hidden, hidden),
        "post_rms_weight":   _row_stack("post_rms_weight", hidden),
        "w_gate":            _flat_stack_kernel("w_gate", hidden, inter),
        "w_up":              _flat_stack_kernel("w_up", hidden, inter),
        "w_down":            _flat_stack_kernel("w_down", inter, hidden),
    }
