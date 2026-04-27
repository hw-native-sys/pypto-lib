# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B single-layer decode forward.

Scope 1:
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Scope 2:
  1. K RoPE + cache write, V cache write, Q RoPE + pad
  2. QK matmul
  3. Softmax
  4. SV matmul
  5. Online-softmax accumulation + final normalisation

Scope 3:
  1. Output projection: attn_out × wo
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition
"""

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
INTERMEDIATE = 25600
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Scope 1 tiling constants.
SCOPE1_K_CHUNK = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
BATCH_TILE = 16

# Scope 2 tiling constants.
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256
SB_BATCH = 64

# Scope 3 tiling constants.
K_CHUNK = 128
OUT_PROJ_K_CHUNK = 512  # out_proj K block: large value to fully utilize L0B (K*N*2 = 64KB)
MLP_OUT_CHUNK = 256


def build_qwen3_decode_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    scope1_hidden_blocks = hidden // SCOPE1_K_CHUNK
    out_proj_k_blocks = hidden // OUT_PROJ_K_CHUNK
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks = inter // MLP_OUT_CHUNK
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE

    @pl.program
    class Qwen3Decode:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            # Intermediate FP32 tensors between scope 1 and scope 2.
            q_proj = pl.create_tensor([batch, hidden], dtype=pl.FP32)
            k_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
            v_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)

            # ── Scope 1: input RMSNorm + Q/K/V projection ──
            for b0 in pl.parallel(0, batch, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(scope1_hidden_blocks, stage=4):
                        k0 = kb * SCOPE1_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    # Compute variance in [1, BATCH_TILE], then reshape to [BATCH_TILE, 1]
                    # for row_expand_mul broadcasting.
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                        [BATCH_TILE, 1],
                    )
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.pipeline(scope1_hidden_blocks, stage=4):
                        k0 = kb * SCOPE1_K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, SCOPE1_K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="q_proj"):
                    for ob in pl.parallel(q_out_blocks, chunk=4):
                        q0 = ob * Q_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                        tile_b = pl.slice(wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [0, q0])
                        q_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        tile_a_1 = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, SCOPE1_K_CHUNK])
                        tile_b_1 = pl.slice(wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [SCOPE1_K_CHUNK, q0])
                        q_acc = pl.matmul_acc(q_acc, tile_a_1, tile_b_1)

                        for kb in pl.pipeline(2, scope1_hidden_blocks, stage=2):
                            k0 = kb * SCOPE1_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="kv_proj"):
                    for ob in pl.parallel(kv_out_blocks, chunk=1):
                        kv0 = ob * KV_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, 0])
                        tile_wk = pl.slice(wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        tile_wv = pl.slice(wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)

                        tile_a_1 = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, SCOPE1_K_CHUNK])
                        tile_wk_1 = pl.slice(wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [SCOPE1_K_CHUNK, kv0])
                        tile_wv_1 = pl.slice(wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [SCOPE1_K_CHUNK, kv0])
                        k_acc = pl.matmul_acc(k_acc, tile_a_1, tile_wk_1)
                        v_acc = pl.matmul_acc(v_acc, tile_a_1, tile_wv_1)

                        for kb in pl.pipeline(2, scope1_hidden_blocks, stage=2):
                            k0 = kb * SCOPE1_K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            tile_wv_i = pl.slice(wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])
                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

            # ── Scope 2: RoPE + KV cache update + grouped-query attention ──
            all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)

            attn_out = pl.create_tensor([batch, hidden], dtype=pl.BF16)
            for b in pl.parallel(batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for ki in pl.range(num_kv_heads):
                        # K RoPE + cache update.
                        kv_col = ki * head_dim
                        k_lo = pl.slice(k_proj, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(k_proj, [1, half_dim], [b, kv_col + half_dim])
                        rot_lo = pl.sub(
                            pl.col_expand_mul(k_lo, cos_lo),
                            pl.col_expand_mul(k_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(k_hi, cos_hi),
                            pl.col_expand_mul(k_lo, sin_hi),
                        )
                        cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                        k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                        k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, half_dim])
                        # V cache update.
                        v_cache = pl.assemble(
                            v_cache,
                            pl.cast(pl.slice(v_proj, [1, head_dim], [b, ki * head_dim]), target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        # Q RoPE + pad (ki == kvh since q_groups == 1).
                        q_base = ki * q_per_kv
                        q_block = pl.reshape(
                            pl.slice(q_proj, [1, Q_HEAD_BATCH * head_dim], [b, q_base * head_dim]),
                            [Q_HEAD_BATCH, head_dim],
                        )
                        q_lo = pl.slice(q_block, [Q_HEAD_BATCH, half_dim], [0, 0])
                        q_hi = pl.slice(q_block, [Q_HEAD_BATCH, half_dim], [0, half_dim])
                        rot_lo_bf16 = pl.cast(
                            pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                            target_type=pl.BF16,
                        )
                        rot_hi_bf16 = pl.cast(
                            pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                            target_type=pl.BF16,
                        )
                        all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD, 0])
                        all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD, half_dim])
                        # Pad the trailing rows of this Q_HEAD_PAD block with zeros.
                        all_q_padded = pl.assemble(
                            all_q_padded,
                            pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                            [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                        )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                for gi in pl.parallel(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0])

                    # Stage 2: QK matmul for all active sb blocks.
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="qk_matmul"):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            s0 = sb * SEQ_TILE
                            cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                            k_tile = pl.slice(
                                k_cache,
                                [SEQ_TILE, head_dim],
                                [cache_row0, 0],
                            )
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    # Stage 3: softmax for all active sb blocks.
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="softmax"):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            s0 = sb * SEQ_TILE
                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                            scores_valid = pl.slice(
                                all_raw_scores,
                                [Q_HEAD_BATCH, SEQ_TILE],
                                [sb * Q_HEAD_PAD, 0],
                                valid_shape=[Q_HEAD_BATCH, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            scores = pl.mul(scores_padded, attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)
                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH, 0])
                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH, 0])

                    # Stage 4: SV matmul for all active sb blocks.
                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="sv_matmul"):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            s0 = sb * SEQ_TILE
                            cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                            exp_tile = pl.slice(
                                all_exp_padded,
                                [Q_HEAD_PAD, SEQ_TILE],
                                [sb * Q_HEAD_PAD, 0],
                            )
                            v_tile = pl.slice(
                                v_cache,
                                [SEQ_TILE, head_dim],
                                [cache_row0, 0],
                            )
                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    # Stage 5: online softmax accumulation and normalisation.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                        oi = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, head_dim], [0, 0])
                        mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [0, 0])
                        li = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [0, 0])
                        for sb in pl.range(1, ctx_blocks):
                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, head_dim], [sb * Q_HEAD_PAD, 0])
                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                            cur_li = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(pl.row_expand_mul(oi, alpha),
                                        pl.row_expand_mul(oi_tmp_valid, beta))
                            mi = mi_new
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim])
                        ctx_flat_bf16 = pl.cast(ctx_flat, target_type=pl.BF16)
                        attn_row = pl.assemble(
                            attn_row, ctx_flat_bf16, [0, q_base * head_dim],
                        )

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # ── Scope 3: output projection + residual + post RMSNorm + MLP + residual ──
            for b0 in pl.parallel(0, batch, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.FP32)

                # Stage 1 & 2: Output projection + residual addition with hidden_states
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="out_proj_residual"):
                    for ob in pl.parallel(0, q_out_blocks, chunk=6):
                        o0 = ob * Q_OUT_CHUNK
                        a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, 0])
                        w_chunk_0 = pl.slice(wo, [OUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        hidden_chunk = pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0])

                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)

                        a_chunk_1 = pl.slice(attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, OUT_PROJ_K_CHUNK])
                        w_chunk_1 = pl.slice(wo, [OUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [OUT_PROJ_K_CHUNK, o0])
                        o_acc = pl.matmul_acc(o_acc, a_chunk_1, w_chunk_1)

                        for kb in pl.pipeline(2, out_proj_k_blocks, stage=2):
                            k0 = kb * OUT_PROJ_K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, OUT_PROJ_K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [OUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                        resid = pl.cast(hidden_chunk, target_type=pl.FP32)
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                # Stage 3: Post-attention RMSNorm
                post_norm_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH_TILE]),
                        )
                    inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.pipeline(hidden_blocks, stage=2):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        post_normed = pl.col_expand_mul(
                            pl.row_expand_mul(resid_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                            post_gamma,
                        )
                        normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                        post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                # Stage 4 & 5 & 6: MLP: gate/up projections + SiLU.
                mlp_tile = pl.create_tensor([BATCH_TILE, inter], dtype=pl.BF16)
                for ob in pl.parallel(0, mlp_out_blocks, 1):
                    o0 = ob * MLP_OUT_CHUNK
                    post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, K_CHUNK])
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)

                        wg_1 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk_1, wg_1)

                        for kb in pl.pipeline(2, hidden_blocks, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)

                        wu_1 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk_1, wu_1)

                        for kb in pl.pipeline(2, hidden_blocks, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="silu"):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                # Stage 7 & 8: Down projection + final residual writeback.
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="down_proj_residual"):
                    for dob in pl.parallel(0, hidden_blocks, chunk=1):
                        d0 = dob * K_CHUNK
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        resid1_tile_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])

                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)

                        mlp_chunk_1 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, MLP_OUT_CHUNK])
                        w_down_chunk_1 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [MLP_OUT_CHUNK, d0])
                        down_acc = pl.matmul_acc(down_acc, mlp_chunk_1, w_down_chunk_1)

                        for ob in pl.pipeline(2, mlp_out_blocks, stage=2):
                            o0 = ob * MLP_OUT_CHUNK
                            down_mlp_chunk_bf16 = pl.slice(
                                mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0]
                            )
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

                        out_chunk = pl.add(down_acc, resid1_tile_chunk)
                        out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                        out = pl.assemble(out, out_chunk_cast, [b0, d0])

            return out

    return Qwen3Decode


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    cache_rows = batch * num_kv_heads * max_seq

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.rand(hidden_size, hidden_size) / hidden_size ** 0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        return torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

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
        return (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(inter, hidden_size) - 0.5) / inter ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
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
        TensorSpec("w_gate", [hidden_size, inter], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, inter], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [inter, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_decode(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    seq_lens = tensors["seq_lens"]
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
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    inter = w_gate.shape[1]
    eps = 1e-6

    # ── Scope 1 golden: RMSNorm + Q/K/V projection ──
    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, SCOPE1_K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + SCOPE1_K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = (normed.float() @ wq.float()).float()
        k_proj[b0:b_end, :] = (normed.float() @ wk.float()).float()
        v_proj[b0:b_end, :] = (normed.float() @ wv.float()).float()

    # ── Scope 2 golden: RoPE + cache update + attention ──
    attn_out = torch.zeros(batch, hidden_size, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat([q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi], dim=-1)

        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    # ── Scope 3 golden: output projection + residual + post RMSNorm + MLP + residual ──
    # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    # 2. Post-attention RMSNorm.
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    # 3. SwiGLU MLP: gate/up projections, silu activation, down projection.
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    # 4. Final residual + cast to BF16.
    tensors["out"][:] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_decode_program(),
        tensor_specs=build_tensor_specs(use_max_seq=args.max_seq),
        golden_fn=golden_qwen3_decode,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
