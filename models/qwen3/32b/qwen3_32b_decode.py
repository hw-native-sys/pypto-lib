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
CACHE_ROWS = BATCH * NUM_KV_HEADS * MAX_SEQ
HALF_DIM = HEAD_DIM // 2
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Scope 1 tiles
RMSNORM_K_TILE = 512
Q_OUT_TILE = 256
Q_PROJ_K_TILE = 128
KV_OUT_TILE = 256
KV_PROJ_K_TILE = 128
SCOPE2_STAGE_SPMD = 32

# Scope 2 tiles
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS

# Scope 3 tiles
K_TILE = 128
OUT_PROJ_K_TILE = 128
MLP_OUT_TILE = 256
DOWN_N_TILE = 256
DOWN_K_TILE = 128

BATCH_TILE = BATCH
MLP_SPMD_INNER = 2
MLP_GROUP_TILE = MLP_SPMD_INNER * MLP_OUT_TILE

assert (INTERMEDIATE // MLP_OUT_TILE) % MLP_SPMD_INNER == 0


def build_qwen3_decode_program():
    @pl.program
    class Qwen3Decode:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
            wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
            wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
            k_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
            w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
            k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
            v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

            # ── Scope 1: input RMSNorm + Q/K/V projection ──
            normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(HIDDEN // RMSNORM_K_TILE, stage=4):
                    k0 = kb * RMSNORM_K_TILE
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_TILE], target_type=pl.FP32)
                    partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]))
                variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
                inv_rms = pl.recip(pl.sqrt(variance))
                for kb in pl.pipeline(HIDDEN // RMSNORM_K_TILE, stage=4):
                    k0 = kb * RMSNORM_K_TILE
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K_TILE], target_type=pl.FP32)
                    gamma = input_rms_weight[:, k0 : k0 + RMSNORM_K_TILE]
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                    normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [0, k0])

            # Q projection.
            for qi in pl.spmd(HIDDEN // Q_OUT_TILE, name_hint="q_proj"):
                q0 = qi * Q_OUT_TILE
                q_acc = pl.create_tensor([BATCH, Q_OUT_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, HIDDEN // Q_PROJ_K_TILE, stage=2):
                    k0 = kb * Q_PROJ_K_TILE
                    tile_a_i = normed_states[:, k0 : k0 + Q_PROJ_K_TILE]
                    tile_b_i = wq[k0 : k0 + Q_PROJ_K_TILE, q0 : q0 + Q_OUT_TILE]
                    if k0 == 0:
                        q_acc = pl.matmul(tile_a_i, tile_b_i, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                q_proj = pl.assemble(q_proj, q_acc, [0, q0])

            # K/V projection.
            for kvi in pl.spmd(KV_HIDDEN // KV_OUT_TILE, name_hint="kv_proj"):
                kv0 = kvi * KV_OUT_TILE
                k_acc = pl.create_tensor([BATCH, KV_OUT_TILE], dtype=pl.FP32)
                v_acc = pl.create_tensor([BATCH, KV_OUT_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, HIDDEN // KV_PROJ_K_TILE, stage=2):
                    k0 = kb * KV_PROJ_K_TILE
                    tile_a_i = normed_states[:, k0 : k0 + KV_PROJ_K_TILE]
                    tile_wk_i = wk[k0 : k0 + KV_PROJ_K_TILE, kv0 : kv0 + KV_OUT_TILE]
                    tile_wv_i = wv[k0 : k0 + KV_PROJ_K_TILE, kv0 : kv0 + KV_OUT_TILE]
                    if k0 == 0:
                        k_acc = pl.matmul(tile_a_i, tile_wk_i, out_dtype=pl.FP32)
                        v_acc = pl.matmul(tile_a_i, tile_wv_i, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                k_proj = pl.assemble(k_proj, k_acc, [0, kv0])
                v_proj = pl.assemble(v_proj, v_acc, [0, kv0])

            # ── Scope 2: RoPE + KV cache update + grouped-query attention ──
            all_q_padded = pl.create_tensor([BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
            attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

            for b in pl.parallel(BATCH):
                ctx_len = pl.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_lo = rope_cos[pos : pos + 1, 0 : HALF_DIM]
                cos_hi = rope_cos[pos : pos + 1, HALF_DIM : HEAD_DIM]
                sin_lo = rope_sin[pos : pos + 1, 0 : HALF_DIM]
                sin_hi = rope_sin[pos : pos + 1, HALF_DIM : HEAD_DIM]

                # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for ki in pl.range(NUM_KV_HEADS):
                        kv_col = ki * HEAD_DIM
                        k_lo = k_proj[b : b + 1, kv_col : kv_col + HALF_DIM]
                        k_hi = k_proj[b : b + 1, kv_col + HALF_DIM : kv_col + HEAD_DIM]
                        rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                        rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                        cache_row = b * NUM_KV_HEADS * MAX_SEQ + ki * MAX_SEQ + pos
                        k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_row, 0])
                        k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_row, HALF_DIM])
                        v_row_bf16 = pl.cast(v_proj[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM], target_type=pl.BF16)
                        v_cache = pl.assemble(v_cache, v_row_bf16, [cache_row, 0])

                        q_base = ki * Q_PER_KV
                        q_block = pl.reshape(q_proj[b : b + 1, q_base * HEAD_DIM : (q_base + Q_HEAD_BATCH) * HEAD_DIM], [Q_HEAD_BATCH, HEAD_DIM])
                        q_lo = q_block[:, 0 : HALF_DIM]
                        q_hi = q_block[:, HALF_DIM : HEAD_DIM]
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_pad_row0 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                        all_q_padded = pl.assemble(all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_row0, 0])
                        all_q_padded = pl.assemble(all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_row0, HALF_DIM])
                        q_pad_zero = pl.cast(pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
                        all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])

                attn_row = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)

                # Stage 2: QK matmul.
                all_raw_scores = pl.create_tensor([TOTAL_Q_GROUPS * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                for gi in pl.spmd(TOTAL_Q_GROUPS // 2, name_hint="qk_matmul"):
                    gi0 = gi * 2
                    gi1 = gi * 2 + 1
                    kvh0 = gi0 // Q_GROUPS
                    kvh1 = gi1 // Q_GROUPS
                    q_pad_row0_0 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi0 * Q_HEAD_PAD
                    q_pad_row0_1 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi1 * Q_HEAD_PAD
                    q_padded0 = all_q_padded[q_pad_row0_0 : q_pad_row0_0 + Q_HEAD_PAD, :]
                    q_padded1 = all_q_padded[q_pad_row0_1 : q_pad_row0_1 + Q_HEAD_PAD, :]
                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0_0 = b * NUM_KV_HEADS * MAX_SEQ + kvh0 * MAX_SEQ + s0
                        k_tile_0 = k_cache[cache_row0_0 : cache_row0_0 + SEQ_TILE, :]
                        raw_scores_0 = pl.matmul(q_padded0, k_tile_0, b_trans=True, out_dtype=pl.FP32)
                        all_raw_scores = pl.assemble(all_raw_scores, raw_scores_0, [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])

                        cache_row0_1 = b * NUM_KV_HEADS * MAX_SEQ + kvh1 * MAX_SEQ + s0
                        k_tile_1 = k_cache[cache_row0_1 : cache_row0_1 + SEQ_TILE, :]
                        raw_scores_1 = pl.matmul(q_padded1, k_tile_1, b_trans=True, out_dtype=pl.FP32)
                        all_raw_scores = pl.assemble(all_raw_scores, raw_scores_1, [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])

                # Stage 3: softmax.
                all_exp_padded = pl.create_tensor([TOTAL_Q_GROUPS * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                all_cur_li = pl.create_tensor([TOTAL_Q_GROUPS * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                all_cur_mi = pl.create_tensor([TOTAL_Q_GROUPS * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                for gi in pl.spmd(TOTAL_Q_GROUPS // 2, name_hint="softmax"):
                    gi0 = gi * 2
                    gi1 = gi * 2 + 1
                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)

                        scores_valid_0 = pl.slice(all_raw_scores, [Q_HEAD_BATCH, SEQ_TILE], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0], valid_shape=[Q_HEAD_BATCH, valid_len])
                        scores_padded_0 = pl.fillpad(scores_valid_0, pad_value=pl.PadValue.min)
                        scores_0 = pl.mul(scores_padded_0, ATTN_SCALE)
                        cur_mi_0 = pl.row_max(scores_0)
                        exp_scores_0 = pl.exp(pl.row_expand_sub(scores_0, cur_mi_0))
                        exp_scores_bf16_0 = pl.cast(exp_scores_0, target_type=pl.BF16)
                        exp_scores_fp32_0 = pl.cast(exp_scores_bf16_0, target_type=pl.FP32)
                        cur_li_0 = pl.row_sum(exp_scores_fp32_0)
                        all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16_0, [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        all_cur_mi = pl.assemble(all_cur_mi, cur_mi_0, [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        all_cur_li = pl.assemble(all_cur_li, cur_li_0, [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])

                        scores_valid_1 = pl.slice(all_raw_scores, [Q_HEAD_BATCH, SEQ_TILE], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0], valid_shape=[Q_HEAD_BATCH, valid_len])
                        scores_padded_1 = pl.fillpad(scores_valid_1, pad_value=pl.PadValue.min)
                        scores_1 = pl.mul(scores_padded_1, ATTN_SCALE)
                        cur_mi_1 = pl.row_max(scores_1)
                        exp_scores_1 = pl.exp(pl.row_expand_sub(scores_1, cur_mi_1))
                        exp_scores_bf16_1 = pl.cast(exp_scores_1, target_type=pl.BF16)
                        exp_scores_fp32_1 = pl.cast(exp_scores_bf16_1, target_type=pl.FP32)
                        cur_li_1 = pl.row_sum(exp_scores_fp32_1)
                        all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16_1, [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        all_cur_mi = pl.assemble(all_cur_mi, cur_mi_1, [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        all_cur_li = pl.assemble(all_cur_li, cur_li_1, [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])

                # Stage 4: SV matmul.
                all_oi_tmp = pl.create_tensor([TOTAL_Q_GROUPS * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                for gi in pl.spmd(TOTAL_Q_GROUPS // 2, name_hint="sv_matmul"):
                    gi0 = gi * 2
                    gi1 = gi * 2 + 1
                    kvh0 = gi0 // Q_GROUPS
                    kvh1 = gi1 // Q_GROUPS
                    for sb in pl.range(ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0_0 = b * NUM_KV_HEADS * MAX_SEQ + kvh0 * MAX_SEQ + s0
                        exp_tile_0 = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        v_tile_0 = v_cache[cache_row0_0 : cache_row0_0 + SEQ_TILE, :]
                        oi_tmp_0 = pl.matmul(exp_tile_0, v_tile_0, out_dtype=pl.FP32)
                        all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp_0, [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])

                        cache_row0_1 = b * NUM_KV_HEADS * MAX_SEQ + kvh1 * MAX_SEQ + s0
                        exp_tile_1 = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        v_tile_1 = v_cache[cache_row0_1 : cache_row0_1 + SEQ_TILE, :]
                        oi_tmp_1 = pl.matmul(exp_tile_1, v_tile_1, out_dtype=pl.FP32)
                        all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp_1, [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])

                # Stage 5: online softmax accumulation and normalisation.
                for gi in pl.spmd(TOTAL_Q_GROUPS // 2, name_hint="online_softmax"):
                    gi0 = gi * 2
                    gi1 = gi * 2 + 1
                    kvh0 = gi0 // Q_GROUPS
                    qg0 = gi0 - kvh0 * Q_GROUPS
                    q_base0 = kvh0 * Q_PER_KV + qg0 * Q_HEAD_BATCH
                    kvh1 = gi1 // Q_GROUPS
                    qg1 = gi1 - kvh1 * Q_GROUPS
                    q_base1 = kvh1 * Q_PER_KV + qg1 * Q_HEAD_BATCH
                    oi_0 = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, HEAD_DIM], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD, 0])
                    mi_0 = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 0])
                    li_0 = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 0])
                    oi_1 = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, HEAD_DIM], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD, 0])
                    mi_1 = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 0])
                    li_1 = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH, 0])
                    for sb in pl.range(1, ctx_blocks):
                        oi_tmp_valid_0 = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, HEAD_DIM], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        cur_mi_0 = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        cur_li_0 = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [gi0 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        mi_new_0 = pl.maximum(mi_0, cur_mi_0)
                        alpha_0 = pl.exp(pl.sub(mi_0, mi_new_0))
                        beta_0 = pl.exp(pl.sub(cur_mi_0, mi_new_0))
                        li_0 = pl.add(pl.mul(alpha_0, li_0), pl.mul(beta_0, cur_li_0))
                        oi_0 = pl.add(pl.row_expand_mul(oi_0, alpha_0), pl.row_expand_mul(oi_tmp_valid_0, beta_0))
                        mi_0 = mi_new_0

                        oi_tmp_valid_1 = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, HEAD_DIM], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_PAD + sb * Q_HEAD_PAD, 0])
                        cur_mi_1 = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        cur_li_1 = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [gi1 * ((MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE) * Q_HEAD_BATCH + sb * Q_HEAD_BATCH, 0])
                        mi_new_1 = pl.maximum(mi_1, cur_mi_1)
                        alpha_1 = pl.exp(pl.sub(mi_1, mi_new_1))
                        beta_1 = pl.exp(pl.sub(cur_mi_1, mi_new_1))
                        li_1 = pl.add(pl.mul(alpha_1, li_1), pl.mul(beta_1, cur_li_1))
                        oi_1 = pl.add(pl.row_expand_mul(oi_1, alpha_1), pl.row_expand_mul(oi_tmp_valid_1, beta_1))
                        mi_1 = mi_new_1
                    ctx_0 = pl.row_expand_div(oi_0, li_0)
                    ctx_flat_bf16_0 = pl.cast(pl.reshape(ctx_0, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16)
                    attn_row = pl.assemble(attn_row, ctx_flat_bf16_0, [0, q_base0 * HEAD_DIM])

                    ctx_1 = pl.row_expand_div(oi_1, li_1)
                    ctx_flat_bf16_1 = pl.cast(pl.reshape(ctx_1, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16)
                    attn_row = pl.assemble(attn_row, ctx_flat_bf16_1, [0, q_base1 * HEAD_DIM])

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # ── Scope 3: output projection + residual + post RMSNorm + MLP + residual ──
            resid1_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)

            # Stage 1 & 2: Output projection + residual addition with hidden_states.
            for ob in pl.parallel(0, HIDDEN // Q_OUT_TILE, 2):
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)], name_hint="out_proj_residual"):
                    for oi in pl.range(ob, ob + 2):
                        o0 = oi * Q_OUT_TILE
                        hidden_chunk = hidden_states[:, o0 : o0 + Q_OUT_TILE]
                        o_acc = pl.create_tensor([BATCH, Q_OUT_TILE], dtype=pl.FP32)
                        for kb in pl.pipeline(0, HIDDEN // OUT_PROJ_K_TILE, stage=2):
                            k0 = kb * OUT_PROJ_K_TILE
                            a_chunk = attn_out[:, k0 : k0 + OUT_PROJ_K_TILE]
                            w_chunk = wo[k0 : k0 + OUT_PROJ_K_TILE, o0 : o0 + Q_OUT_TILE]
                            if k0 == 0:
                                o_acc = pl.matmul(a_chunk, w_chunk, out_dtype=pl.FP32)
                            else:
                                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
                        resid = pl.cast(hidden_chunk, target_type=pl.FP32)
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

            # Stage 3: Post-attention RMSNorm.
            post_norm_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
                    k0 = kb * K_TILE
                    resid_chunk = resid1_tile[:, k0 : k0 + K_TILE]
                    sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]))
                inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
                inv_rms_s3_col = pl.reshape(inv_rms_s3, [BATCH, 1])
                for kb in pl.pipeline(HIDDEN // K_TILE, stage=2):
                    k0 = kb * K_TILE
                    resid_chunk = resid1_tile[:, k0 : k0 + K_TILE]
                    post_gamma = post_rms_weight[:, k0 : k0 + K_TILE]
                    post_normed = pl.col_expand_mul(pl.row_expand_mul(resid_chunk, inv_rms_s3_col), post_gamma)
                    post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_normed, target_type=pl.BF16), [0, k0])

            # Stage 4~6: keep outer parallel, run smaller SPMD groups and cache per-group only.
            mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)
            for ob_base in pl.parallel(0, INTERMEDIATE // MLP_OUT_TILE, MLP_SPMD_INNER):
                gate_group = pl.create_tensor([BATCH_TILE, MLP_GROUP_TILE], dtype=pl.FP32)
                up_group = pl.create_tensor([BATCH_TILE, MLP_GROUP_TILE], dtype=pl.FP32)

                # Stage 4: gate projection.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="gate_proj_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_TILE
                    g0 = ob * MLP_OUT_TILE
                    post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, K_TILE])
                    wg_0 = pl.slice(w_gate, [K_TILE, MLP_OUT_TILE], [0, o0])
                    gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)

                    wg_1 = pl.slice(w_gate, [K_TILE, MLP_OUT_TILE], [K_TILE, o0])
                    gate_acc = pl.matmul_acc(gate_acc, post_chunk_1, wg_1)

                    for kb in pl.pipeline(2, HIDDEN // K_TILE, stage=2):
                        k0 = kb * K_TILE
                        post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, k0])
                        wg = pl.slice(w_gate, [K_TILE, MLP_OUT_TILE], [k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                    gate_group = pl.assemble(gate_group, gate_acc, [0, g0])

                # Stage 5: up projection.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="up_proj_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_TILE
                    g0 = ob * MLP_OUT_TILE
                    post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, K_TILE])
                    wu_0 = pl.slice(w_up, [K_TILE, MLP_OUT_TILE], [0, o0])
                    up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)

                    wu_1 = pl.slice(w_up, [K_TILE, MLP_OUT_TILE], [K_TILE, o0])
                    up_acc = pl.matmul_acc(up_acc, post_chunk_1, wu_1)

                    for kb in pl.pipeline(2, HIDDEN // K_TILE, stage=2):
                        k0 = kb * K_TILE
                        post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_TILE], [0, k0])
                        wu = pl.slice(w_up, [K_TILE, MLP_OUT_TILE], [k0, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
                    up_group = pl.assemble(up_group, up_acc, [0, g0])

                # Stage 6: SiLU + gate/up fuse.
                for ob in pl.spmd(MLP_SPMD_INNER, name_hint="silu_spmd"):
                    o0 = (ob_base + ob) * MLP_OUT_TILE
                    g0 = ob * MLP_OUT_TILE
                    gate_acc = pl.slice(gate_group, [BATCH_TILE, MLP_OUT_TILE], [0, g0])
                    up_acc = pl.slice(up_group, [BATCH_TILE, MLP_OUT_TILE], [0, g0])
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                    mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

            # Stage 7 & 8: Down projection + final residual writeback.
            for db in pl.parallel(0, HIDDEN // DOWN_N_TILE, 2):
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)], name_hint="down_proj_residual"):
                    for di in pl.range(db, db + 2):
                        d0 = di * DOWN_N_TILE
                        resid1_tile_chunk = resid1_tile[:, d0 : d0 + DOWN_N_TILE]
                        down_acc = pl.create_tensor([BATCH, DOWN_N_TILE], dtype=pl.FP32)
                        for ob in pl.pipeline(0, INTERMEDIATE // DOWN_K_TILE, stage=2):
                            o0 = ob * DOWN_K_TILE
                            down_mlp_chunk = mlp_tile[:, o0 : o0 + DOWN_K_TILE]
                            w_down_chunk = w_down[o0 : o0 + DOWN_K_TILE, d0 : d0 + DOWN_N_TILE]
                            if o0 == 0:
                                down_acc = pl.matmul(down_mlp_chunk, w_down_chunk, out_dtype=pl.FP32)
                            else:
                                down_acc = pl.matmul_acc(down_acc, down_mlp_chunk, w_down_chunk)
                        out_chunk = pl.add(down_acc, resid1_tile_chunk)
                        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [0, d0])

            return out

    return Qwen3Decode


def build_tensor_specs(use_max_seq: bool = False):
    import torch
    from golden import TensorSpec

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_rms_weight():
        return torch.rand(1, HIDDEN) - 0.5

    def init_wq():
        return torch.rand(HIDDEN, HIDDEN) / HIDDEN ** 0.5

    def init_wk():
        return torch.rand(HIDDEN, KV_HIDDEN) / HIDDEN ** 0.5

    def init_wv():
        return torch.rand(HIDDEN, KV_HIDDEN) / HIDDEN ** 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((BATCH,), MAX_SEQ, dtype=torch.int32)
        return torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_rope_cos():
        return torch.rand(MAX_SEQ, HEAD_DIM) - 0.5

    def init_rope_sin():
        return torch.rand(MAX_SEQ, HEAD_DIM) - 0.5

    def init_k_cache():
        return torch.rand(CACHE_ROWS, HEAD_DIM) - 0.5

    def init_v_cache():
        return torch.rand(CACHE_ROWS, HEAD_DIM) - 0.5

    def init_wo():
        return (torch.rand(HIDDEN, HIDDEN) - 0.5) / HIDDEN ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, HIDDEN)

    def init_w_gate():
        return (torch.rand(HIDDEN, INTERMEDIATE) - 0.5) / HIDDEN ** 0.5

    def init_w_up():
        return (torch.rand(HIDDEN, INTERMEDIATE) - 0.5) / HIDDEN ** 0.5

    def init_w_down():
        return (torch.rand(INTERMEDIATE, HIDDEN) - 0.5) / INTERMEDIATE ** 0.5

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, HIDDEN], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=init_wv),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [MAX_SEQ, HEAD_DIM], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, HEAD_DIM], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [CACHE_ROWS, HEAD_DIM], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [CACHE_ROWS, HEAD_DIM], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("wo", [HIDDEN, HIDDEN], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, HIDDEN], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [HIDDEN, INTERMEDIATE], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [HIDDEN, INTERMEDIATE], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [INTERMEDIATE, HIDDEN], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [BATCH, HIDDEN], torch.bfloat16, is_output=True),
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

    half = HEAD_DIM // 2
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # ── Scope 1 golden: RMSNorm + Q/K/V projection ──
    x_tile = hidden_states.float()
    sq_sum = (x_tile ** 2).sum(dim=-1, keepdim=True)
    variance = sq_sum / HIDDEN + EPS
    rms = torch.sqrt(variance)
    normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

    q_proj = (normed.float() @ wq.float()).float()
    k_proj = (normed.float() @ wk.float()).float()
    v_proj = (normed.float() @ wv.float()).float()

    # ── Scope 2 golden: RoPE + cache update + attention ──
    attn_out = torch.zeros(BATCH, HIDDEN, dtype=torch.float32)

    for b in range(BATCH):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(NUM_KV_HEADS, HEAD_DIM)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(NUM_KV_HEADS):
            cr = b * NUM_KV_HEADS * MAX_SEQ + ki * MAX_SEQ + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * HEAD_DIM : (ki + 1) * HEAD_DIM].to(torch.bfloat16)

        q_heads = q_proj[b].view(NUM_HEADS, HEAD_DIM)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat([q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi], dim=-1)

        for kvh in range(NUM_KV_HEADS):
            for qg in range(Q_GROUPS):
                q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, HEAD_DIM, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + s0

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
                    attn_out[b, qh * HEAD_DIM : (qh + 1) * HEAD_DIM] = ctx[qi]

    # ── Scope 3 golden: output projection + residual + post RMSNorm + MLP + residual ──
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_decode_program(),
        specs=build_tensor_specs(use_max_seq=args.max_seq),
        golden_fn=golden_qwen3_decode,
        compile_cfg=dict(dump_passes=False),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
