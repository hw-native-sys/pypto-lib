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
ACTIVATION_INIT_SCALE = 0.1
ROPE_INIT_SCALE = 0.1
CACHE_INIT_SCALE = 0.1
POST_RMS_INIT_SCALE = 0.2
OUT_PROJ_INIT_SCALE = 0.5

# Scope 1 tiles
Q_OUT_CHUNK = 256
Q_PROJ_K_CHUNK = 128
KV_OUT_CHUNK = 256
KV_PROJ_K_CHUNK = 128
HIDDEN_K_BLOCKS = HIDDEN // Q_PROJ_K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK

# Scope 2 tiles
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 256
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE

# Scope 3 tiles
K_CHUNK = 128
OUT_PROJ_K_CHUNK = 128
OUT_PROJ_N_CHUNK = 512
MLP_OUT_CHUNK = 256
DOWN_N_CHUNK = 512
DOWN_K_CHUNK = 64
K_BLOCKS = HIDDEN // K_CHUNK
OUT_PROJ_K_BLOCKS = HIDDEN // OUT_PROJ_K_CHUNK
OUT_PROJ_N_BLOCKS = HIDDEN // OUT_PROJ_N_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK
DOWN_N_BLOCKS = HIDDEN // DOWN_N_CHUNK
DOWN_K_BLOCKS = INTERMEDIATE // DOWN_K_CHUNK


def build_qwen3_decode_program():
    @pl.program
    class Qwen3Decode:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_decode(
            self,
            hidden_states: pl.Tensor[[HIDDEN_K_BLOCKS, 1, BATCH, Q_PROJ_K_CHUNK], pl.BF16],
            input_rms_weight: pl.Tensor[[HIDDEN_K_BLOCKS, 1, 1, Q_PROJ_K_CHUNK], pl.FP32],
            wq: pl.Tensor[[HIDDEN_K_BLOCKS, Q_OUT_BLOCKS, Q_PROJ_K_CHUNK, Q_OUT_CHUNK], pl.BF16],
            wk: pl.Tensor[[HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK], pl.BF16],
            wv: pl.Tensor[[HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK], pl.BF16],
            seq_lens: pl.Tensor[[BATCH, 1, 1, 1], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, 1, 1, HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, 1, 1, HEAD_DIM], pl.FP32],
            k_cache: pl.Tensor[[CACHE_ROWS // MAX_SEQ, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[CACHE_ROWS // MAX_SEQ, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[OUT_PROJ_K_BLOCKS, OUT_PROJ_N_BLOCKS, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK], pl.BF16],
            post_rms_weight: pl.Tensor[[K_BLOCKS, 1, 1, K_CHUNK], pl.FP32],
            w_gate: pl.Tensor[[K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK], pl.BF16],
            w_up: pl.Tensor[[K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK], pl.BF16],
            w_down: pl.Tensor[[DOWN_K_BLOCKS, DOWN_N_BLOCKS, DOWN_K_CHUNK, DOWN_N_CHUNK], pl.BF16],
            out: pl.Out[pl.Tensor[[DOWN_N_BLOCKS, 1, BATCH, DOWN_N_CHUNK], pl.BF16]],
        ) -> pl.Tensor[[DOWN_N_BLOCKS, 1, BATCH, DOWN_N_CHUNK], pl.BF16]:
            q_proj = pl.create_tensor([Q_OUT_BLOCKS, 1, BATCH, Q_OUT_CHUNK], dtype=pl.FP32)
            k_proj = pl.create_tensor([KV_OUT_BLOCKS, 1, BATCH, KV_OUT_CHUNK], dtype=pl.FP32)
            v_proj = pl.create_tensor([KV_OUT_BLOCKS, 1, BATCH, KV_OUT_CHUNK], dtype=pl.FP32)

            # ── Scope 1: input RMSNorm + Q/K/V projection ──
            normed_states = pl.create_tensor([HIDDEN_K_BLOCKS, 1, BATCH, Q_PROJ_K_CHUNK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(HIDDEN_K_BLOCKS, stage=4):
                    x_chunk = pl.cast(hidden_states[kb : kb + 1, :, :, :], target_type=pl.FP32)
                    sq_chunk = pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH])
                    partial_sq = pl.add(partial_sq, sq_chunk)
                inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS)))
                inv_rms_col = pl.reshape(inv_rms, [BATCH, 1])
                for kb in pl.pipeline(HIDDEN_K_BLOCKS, stage=4):
                    x_chunk = pl.cast(hidden_states[kb : kb + 1, :, :, :], target_type=pl.FP32)
                    gamma = input_rms_weight[kb : kb + 1, :, :, :]
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_col), gamma)
                    normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [kb, 0, 0, 0])

            # Q projection.
            for qb in pl.parallel(Q_OUT_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                    tile_a0 = normed_states[0:1, :, :, :]
                    tile_b0 = wq[0:1, qb : qb + 1, :, :]
                    q_acc = pl.matmul(tile_a0, tile_b0, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                        tile_a_i = normed_states[kb : kb + 1, :, :, :]
                        tile_b_i = wq[kb : kb + 1, qb : qb + 1, :, :]
                        q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_b_i)
                    q_proj = pl.assemble(q_proj, q_acc, [qb, 0, 0, 0])

            # K/V projection.
            for kvb in pl.parallel(KV_OUT_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
                    tile_a0 = normed_states[0:1, :, :, :]
                    tile_wk0 = wk[0:1, kvb : kvb + 1, :, :]
                    k_acc = pl.matmul(tile_a0, tile_wk0, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                        tile_a_i = normed_states[kb : kb + 1, :, :, :]
                        tile_wk_i = wk[kb : kb + 1, kvb : kvb + 1, :, :]
                        k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                    k_proj = pl.assemble(k_proj, k_acc, [kvb, 0, 0, 0])

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
                    tile_a0 = normed_states[0:1, :, :, :]
                    tile_wv0 = wv[0:1, kvb : kvb + 1, :, :]
                    v_acc = pl.matmul(tile_a0, tile_wv0, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                        tile_a_i = normed_states[kb : kb + 1, :, :, :]
                        tile_wv_i = wv[kb : kb + 1, kvb : kvb + 1, :, :]
                        v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                    v_proj = pl.assemble(v_proj, v_acc, [kvb, 0, 0, 0])

            # ── Scope 2: RoPE + KV cache update + grouped-query attention ──
            all_q_padded = pl.create_tensor([BATCH * TOTAL_Q_GROUPS, 1, Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
            attn_proj_tile = pl.create_tensor([OUT_PROJ_K_BLOCKS, 1, BATCH, OUT_PROJ_K_CHUNK], dtype=pl.BF16)

            for b in pl.parallel(BATCH):
                ctx_len = pl.read(seq_lens, [b, 0, 0, 0])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                pos_block = pos // SEQ_TILE
                pos_offset = pos - pos_block * SEQ_TILE
                cos_lo = rope_cos[pos : pos + 1, :, :, 0 : HALF_DIM]
                cos_hi = rope_cos[pos : pos + 1, :, :, HALF_DIM : HEAD_DIM]
                sin_lo = rope_sin[pos : pos + 1, :, :, 0 : HALF_DIM]
                sin_hi = rope_sin[pos : pos + 1, :, :, HALF_DIM : HEAD_DIM]

                # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for ki in pl.range(NUM_KV_HEADS):
                        kv_col = ki * HEAD_DIM
                        kv_block = kv_col // KV_OUT_CHUNK
                        kv_offset = kv_col - kv_block * KV_OUT_CHUNK
                        cache_idx = b * NUM_KV_HEADS + ki
                        k_lo = k_proj[kv_block : kv_block + 1, :, b : b + 1, kv_offset : kv_offset + HALF_DIM]
                        k_hi = k_proj[kv_block : kv_block + 1, :, b : b + 1, kv_offset + HALF_DIM : kv_offset + HEAD_DIM]
                        rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                        rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                        k_cache = pl.assemble(k_cache, pl.cast(rot_lo, target_type=pl.BF16), [cache_idx, pos_block, pos_offset, 0])
                        k_cache = pl.assemble(k_cache, pl.cast(rot_hi, target_type=pl.BF16), [cache_idx, pos_block, pos_offset, HALF_DIM])
                        v_row_bf16 = pl.cast(v_proj[kv_block : kv_block + 1, :, b : b + 1, kv_offset : kv_offset + HEAD_DIM], target_type=pl.BF16)
                        v_cache = pl.assemble(v_cache, v_row_bf16, [cache_idx, pos_block, pos_offset, 0])

                        q_base = ki * Q_PER_KV
                        q_pad_idx = b * TOTAL_Q_GROUPS + ki
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * HEAD_DIM
                            q_block = q_col // Q_OUT_CHUNK
                            q_offset = q_col - q_block * Q_OUT_CHUNK
                            q_lo = q_proj[q_block : q_block + 1, :, b : b + 1, q_offset : q_offset + HALF_DIM]
                            q_hi = q_proj[q_block : q_block + 1, :, b : b + 1, q_offset + HALF_DIM : q_offset + HEAD_DIM]
                            q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                            q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                            all_q_padded = pl.assemble(all_q_padded, pl.cast(q_rot_lo, target_type=pl.BF16), [q_pad_idx, 0, qi, 0])
                            all_q_padded = pl.assemble(all_q_padded, pl.cast(q_rot_hi, target_type=pl.BF16), [q_pad_idx, 0, qi, HALF_DIM])
                        q_pad_zero = pl.cast(pl.full([1, 1, Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
                        all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_idx, 0, Q_HEAD_BATCH, 0])

                for gi in pl.parallel(0, TOTAL_Q_GROUPS, 2):
                    gi0 = gi
                    gi1 = gi + 1

                    kvh0 = gi0 // Q_GROUPS
                    qg0 = gi0 - kvh0 * Q_GROUPS
                    q_base0 = kvh0 * Q_PER_KV + qg0 * Q_HEAD_BATCH
                    q_padded0 = all_q_padded[b * TOTAL_Q_GROUPS + gi0 : b * TOTAL_Q_GROUPS + gi0 + 1, :, :, :]

                    kvh1 = gi1 // Q_GROUPS
                    qg1 = gi1 - kvh1 * Q_GROUPS
                    q_base1 = kvh1 * Q_PER_KV + qg1 * Q_HEAD_BATCH
                    q_padded1 = all_q_padded[b * TOTAL_Q_GROUPS + gi1 : b * TOTAL_Q_GROUPS + gi1 + 1, :, :, :]

                    # Stage 2: QK matmul.
                    all_raw_scores0 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    all_raw_scores1 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                        for sb in pl.range(ctx_blocks):
                            k_tile_0 = k_cache[b * NUM_KV_HEADS + kvh0 : b * NUM_KV_HEADS + kvh0 + 1, sb : sb + 1, :, :]
                            raw_scores_0 = pl.matmul(q_padded0, k_tile_0, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores0 = pl.assemble(
                                all_raw_scores0,
                                raw_scores_0,
                                [sb, 0, 0, 0],
                            )

                            k_tile_1 = k_cache[b * NUM_KV_HEADS + kvh1 : b * NUM_KV_HEADS + kvh1 + 1, sb : sb + 1, :, :]
                            raw_scores_1 = pl.matmul(q_padded1, k_tile_1, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores1 = pl.assemble(
                                all_raw_scores1,
                                raw_scores_1,
                                [sb, 0, 0, 0],
                            )

                    # Stage 3: softmax.
                    all_exp_padded0 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_cur_li0 = pl.create_tensor([MAX_CTX_BLOCKS, Q_HEAD_BATCH], dtype=pl.FP32)
                    all_cur_mi0 = pl.create_tensor([MAX_CTX_BLOCKS, Q_HEAD_BATCH], dtype=pl.FP32)
                    all_exp_padded1 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_cur_li1 = pl.create_tensor([MAX_CTX_BLOCKS, Q_HEAD_BATCH], dtype=pl.FP32)
                    all_cur_mi1 = pl.create_tensor([MAX_CTX_BLOCKS, Q_HEAD_BATCH], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)

                            scores_valid_0 = pl.slice(
                                all_raw_scores0,
                                [1, 1, Q_HEAD_BATCH, SEQ_TILE],
                                [sb, 0, 0, 0],
                                valid_shape=[1, 1, Q_HEAD_BATCH, valid_len],
                            )
                            scores_padded_0 = pl.fillpad(scores_valid_0, pad_value=pl.PadValue.min)
                            scores_0 = pl.mul(scores_padded_0, ATTN_SCALE)
                            cur_mi_0 = pl.row_max(scores_0)
                            exp_scores_0 = pl.exp(pl.row_expand_sub(scores_0, cur_mi_0))
                            exp_scores_bf16_0 = pl.cast(exp_scores_0, target_type=pl.BF16)
                            exp_scores_fp32_0 = pl.cast(exp_scores_bf16_0, target_type=pl.FP32)
                            cur_li_0 = pl.row_sum(exp_scores_fp32_0)
                            all_exp_padded0 = pl.assemble(all_exp_padded0, exp_scores_bf16_0, [sb, 0, 0, 0])
                            all_cur_mi0 = pl.assemble(all_cur_mi0, pl.reshape(cur_mi_0, [1, Q_HEAD_BATCH]), [sb, 0])
                            all_cur_li0 = pl.assemble(all_cur_li0, pl.reshape(cur_li_0, [1, Q_HEAD_BATCH]), [sb, 0])

                            scores_valid_1 = pl.slice(
                                all_raw_scores1,
                                [1, 1, Q_HEAD_BATCH, SEQ_TILE],
                                [sb, 0, 0, 0],
                                valid_shape=[1, 1, Q_HEAD_BATCH, valid_len],
                            )
                            scores_padded_1 = pl.fillpad(scores_valid_1, pad_value=pl.PadValue.min)
                            scores_1 = pl.mul(scores_padded_1, ATTN_SCALE)
                            cur_mi_1 = pl.row_max(scores_1)
                            exp_scores_1 = pl.exp(pl.row_expand_sub(scores_1, cur_mi_1))
                            exp_scores_bf16_1 = pl.cast(exp_scores_1, target_type=pl.BF16)
                            exp_scores_fp32_1 = pl.cast(exp_scores_bf16_1, target_type=pl.FP32)
                            cur_li_1 = pl.row_sum(exp_scores_fp32_1)
                            all_exp_padded1 = pl.assemble(all_exp_padded1, exp_scores_bf16_1, [sb, 0, 0, 0])
                            all_cur_mi1 = pl.assemble(all_cur_mi1, pl.reshape(cur_mi_1, [1, Q_HEAD_BATCH]), [sb, 0])
                            all_cur_li1 = pl.assemble(all_cur_li1, pl.reshape(cur_li_1, [1, Q_HEAD_BATCH]), [sb, 0])

                    # Stage 4: SV matmul.
                    all_oi_tmp0 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                    all_oi_tmp1 = pl.create_tensor([MAX_CTX_BLOCKS, 1, Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                        for sb in pl.range(ctx_blocks):
                            exp_tile_0 = all_exp_padded0[sb : sb + 1, :, :, :]
                            v_tile_0 = v_cache[b * NUM_KV_HEADS + kvh0 : b * NUM_KV_HEADS + kvh0 + 1, sb : sb + 1, :, :]
                            oi_tmp_0 = pl.matmul(exp_tile_0, v_tile_0, out_dtype=pl.FP32)
                            all_oi_tmp0 = pl.assemble(
                                all_oi_tmp0,
                                oi_tmp_0,
                                [sb, 0, 0, 0],
                            )

                            exp_tile_1 = all_exp_padded1[sb : sb + 1, :, :, :]
                            v_tile_1 = v_cache[b * NUM_KV_HEADS + kvh1 : b * NUM_KV_HEADS + kvh1 + 1, sb : sb + 1, :, :]
                            oi_tmp_1 = pl.matmul(exp_tile_1, v_tile_1, out_dtype=pl.FP32)
                            all_oi_tmp1 = pl.assemble(
                                all_oi_tmp1,
                                oi_tmp_1,
                                [sb, 0, 0, 0],
                            )

                    # Stage 5: online softmax accumulation and normalisation.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                        oi_0 = pl.reshape(all_oi_tmp0[0 : 1, :, 0 : Q_HEAD_BATCH, :], [Q_HEAD_BATCH, HEAD_DIM])
                        mi_0 = pl.reshape(all_cur_mi0[0 : 1, :], [Q_HEAD_BATCH, 1])
                        li_0 = pl.reshape(all_cur_li0[0 : 1, :], [Q_HEAD_BATCH, 1])
                        oi_1 = pl.reshape(all_oi_tmp1[0 : 1, :, 0 : Q_HEAD_BATCH, :], [Q_HEAD_BATCH, HEAD_DIM])
                        mi_1 = pl.reshape(all_cur_mi1[0 : 1, :], [Q_HEAD_BATCH, 1])
                        li_1 = pl.reshape(all_cur_li1[0 : 1, :], [Q_HEAD_BATCH, 1])
                        for sb in pl.range(1, ctx_blocks):
                            oi_tmp_valid_0 = pl.reshape(
                                all_oi_tmp0[sb : sb + 1, :, 0 : Q_HEAD_BATCH, :],
                                [Q_HEAD_BATCH, HEAD_DIM],
                            )
                            cur_mi_os_0 = pl.reshape(all_cur_mi0[sb : sb + 1, :], [Q_HEAD_BATCH, 1])
                            cur_li_os_0 = pl.reshape(all_cur_li0[sb : sb + 1, :], [Q_HEAD_BATCH, 1])
                            mi_new_0 = pl.maximum(mi_0, cur_mi_os_0)
                            alpha_0 = pl.exp(pl.sub(mi_0, mi_new_0))
                            beta_0 = pl.exp(pl.sub(cur_mi_os_0, mi_new_0))
                            li_0 = pl.add(pl.mul(alpha_0, li_0), pl.mul(beta_0, cur_li_os_0))
                            oi_0 = pl.add(pl.row_expand_mul(oi_0, alpha_0), pl.row_expand_mul(oi_tmp_valid_0, beta_0))
                            mi_0 = mi_new_0

                            oi_tmp_valid_1 = pl.reshape(
                                all_oi_tmp1[sb : sb + 1, :, 0 : Q_HEAD_BATCH, :],
                                [Q_HEAD_BATCH, HEAD_DIM],
                            )
                            cur_mi_os_1 = pl.reshape(all_cur_mi1[sb : sb + 1, :], [Q_HEAD_BATCH, 1])
                            cur_li_os_1 = pl.reshape(all_cur_li1[sb : sb + 1, :], [Q_HEAD_BATCH, 1])
                            mi_new_1 = pl.maximum(mi_1, cur_mi_os_1)
                            alpha_1 = pl.exp(pl.sub(mi_1, mi_new_1))
                            beta_1 = pl.exp(pl.sub(cur_mi_os_1, mi_new_1))
                            li_1 = pl.add(pl.mul(alpha_1, li_1), pl.mul(beta_1, cur_li_os_1))
                            oi_1 = pl.add(pl.row_expand_mul(oi_1, alpha_1), pl.row_expand_mul(oi_tmp_valid_1, beta_1))
                            mi_1 = mi_new_1
                        ctx_0 = pl.row_expand_div(oi_0, li_0)
                        attn_proj_tile = pl.assemble(attn_proj_tile, pl.cast(ctx_0, target_type=pl.BF16), [q_base0, 0, b, 0])

                        ctx_1 = pl.row_expand_div(oi_1, li_1)
                        attn_proj_tile = pl.assemble(attn_proj_tile, pl.cast(ctx_1, target_type=pl.BF16), [q_base1, 0, b, 0])

            # ── Scope 3: output projection + residual + post RMSNorm + MLP + residual ──
            out_proj_tile = pl.create_tensor([OUT_PROJ_N_BLOCKS, 1, BATCH, OUT_PROJ_N_CHUNK], dtype=pl.FP32)

            # Stage 1: Output projection.
            for oi in pl.parallel(OUT_PROJ_N_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
                    a_chunk0 = attn_proj_tile[0:1, :, :, :]
                    w_chunk0_lo = wo[0:1, oi : oi + 1, :, 0:Q_OUT_CHUNK]
                    o_acc_lo = pl.matmul(a_chunk0, w_chunk0_lo, out_dtype=pl.FP32)
                    w_chunk0_hi = wo[0:1, oi : oi + 1, :, Q_OUT_CHUNK:OUT_PROJ_N_CHUNK]
                    o_acc_hi = pl.matmul(a_chunk0, w_chunk0_hi, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, OUT_PROJ_K_BLOCKS, stage=2):
                        a_chunk_i = attn_proj_tile[kb : kb + 1, :, :, :]
                        w_chunk_i_lo = wo[kb : kb + 1, oi : oi + 1, :, 0:Q_OUT_CHUNK]
                        o_acc_lo = pl.matmul_acc(o_acc_lo, a_chunk_i, w_chunk_i_lo)
                        w_chunk_i_hi = wo[kb : kb + 1, oi : oi + 1, :, Q_OUT_CHUNK:OUT_PROJ_N_CHUNK]
                        o_acc_hi = pl.matmul_acc(o_acc_hi, a_chunk_i, w_chunk_i_hi)
                    out_proj_tile = pl.assemble(out_proj_tile, o_acc_lo, [oi, 0, 0, 0])
                    out_proj_tile = pl.assemble(out_proj_tile, o_acc_hi, [oi, 0, 0, Q_OUT_CHUNK])

            # Stage 2: Residual addition with hidden_states.
            resid1_tile = pl.create_tensor([OUT_PROJ_N_BLOCKS, 1, BATCH, OUT_PROJ_N_CHUNK], dtype=pl.FP32)
            for oi in pl.parallel(OUT_PROJ_N_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_residual"):
                    hidden_block = oi * (OUT_PROJ_N_CHUNK // Q_PROJ_K_CHUNK)
                    out_proj_lo = out_proj_tile[oi : oi + 1, :, :, 0:Q_PROJ_K_CHUNK]
                    hidden_lo = pl.cast(hidden_states[hidden_block : hidden_block + 1, :, :, :], target_type=pl.FP32)
                    resid_lo = pl.add(out_proj_lo, hidden_lo)
                    resid1_tile = pl.assemble(resid1_tile, resid_lo, [oi, 0, 0, 0])

                    out_proj_hi = out_proj_tile[oi : oi + 1, :, :, Q_PROJ_K_CHUNK:Q_OUT_CHUNK]
                    hidden_hi = pl.cast(hidden_states[hidden_block + 1 : hidden_block + 2, :, :, :], target_type=pl.FP32)
                    resid_hi = pl.add(out_proj_hi, hidden_hi)
                    resid1_tile = pl.assemble(resid1_tile, resid_hi, [oi, 0, 0, Q_PROJ_K_CHUNK])

                    out_proj_mid = out_proj_tile[oi : oi + 1, :, :, Q_OUT_CHUNK : Q_OUT_CHUNK + Q_PROJ_K_CHUNK]
                    hidden_mid = pl.cast(hidden_states[hidden_block + 2 : hidden_block + 3, :, :, :], target_type=pl.FP32)
                    resid_mid = pl.add(out_proj_mid, hidden_mid)
                    resid1_tile = pl.assemble(resid1_tile, resid_mid, [oi, 0, 0, Q_OUT_CHUNK])

                    out_proj_tail = out_proj_tile[oi : oi + 1, :, :, Q_OUT_CHUNK + Q_PROJ_K_CHUNK : OUT_PROJ_N_CHUNK]
                    hidden_tail = pl.cast(hidden_states[hidden_block + 3 : hidden_block + 4, :, :, :], target_type=pl.FP32)
                    resid_tail = pl.add(out_proj_tail, hidden_tail)
                    resid1_tile = pl.assemble(resid1_tile, resid_tail, [oi, 0, 0, Q_OUT_CHUNK + Q_PROJ_K_CHUNK])

            # Stage 3: Post-attention RMSNorm.
            post_norm_tile = pl.create_tensor([K_BLOCKS, 1, BATCH, K_CHUNK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(K_BLOCKS, stage=2):
                    resid_block = kb // (OUT_PROJ_N_CHUNK // K_CHUNK)
                    resid_offset = (kb - resid_block * (OUT_PROJ_N_CHUNK // K_CHUNK)) * K_CHUNK
                    resid_chunk = resid1_tile[resid_block : resid_block + 1, :, :, resid_offset : resid_offset + K_CHUNK]
                    sq_chunk = pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH])
                    sq_sum = pl.add(sq_sum, sq_chunk)
                inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
                inv_rms_s3_col = pl.reshape(inv_rms_s3, [BATCH, 1])
                for kb in pl.pipeline(K_BLOCKS, stage=2):
                    resid_block = kb // (OUT_PROJ_N_CHUNK // K_CHUNK)
                    resid_offset = (kb - resid_block * (OUT_PROJ_N_CHUNK // K_CHUNK)) * K_CHUNK
                    resid_chunk = resid1_tile[resid_block : resid_block + 1, :, :, resid_offset : resid_offset + K_CHUNK]
                    post_gamma = post_rms_weight[kb : kb + 1, :, :, :]
                    post_normed = pl.col_expand_mul(pl.row_expand_mul(resid_chunk, inv_rms_s3_col), post_gamma)
                    post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_normed, target_type=pl.BF16), [kb, 0, 0, 0])

            # Stage 4 & 5 & 6: MLP gate/up projections + SiLU.
            mlp_tile = pl.create_tensor([MLP_OUT_BLOCKS, 1, BATCH, MLP_OUT_CHUNK], dtype=pl.BF16)
            for mb in pl.parallel(MLP_OUT_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                    post0 = post_norm_tile[0:1, :, :, :]
                    wg0 = w_gate[0:1, mb : mb + 1, :, :]
                    gate_acc = pl.matmul(post0, wg0, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, K_BLOCKS, stage=2):
                        post_chunk = post_norm_tile[kb : kb + 1, :, :, :]
                        wg = w_gate[kb : kb + 1, mb : mb + 1, :, :]
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                    post0 = post_norm_tile[0:1, :, :, :]
                    wu0 = w_up[0:1, mb : mb + 1, :, :]
                    up_acc = pl.matmul(post0, wu0, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, K_BLOCKS, stage=2):
                        post_chunk = post_norm_tile[kb : kb + 1, :, :, :]
                        wu = w_up[kb : kb + 1, mb : mb + 1, :, :]
                        up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_tile = pl.assemble(
                        mlp_tile,
                        pl.cast(mlp_chunk, target_type=pl.BF16),
                        [mb, 0, 0, 0],
                    )

            # Stage 7: Down projection.
            down_proj_tile = pl.create_tensor([DOWN_N_BLOCKS, 1, BATCH, DOWN_N_CHUNK], dtype=pl.FP32)
            for di in pl.parallel(DOWN_N_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj"):
                    down_a0 = mlp_tile[0:1, :, :, 0:DOWN_K_CHUNK]
                    down_b0 = w_down[0:1, di : di + 1, :, :]
                    down_acc = pl.matmul(down_a0, down_b0, out_dtype=pl.FP32)
                    for ob in pl.pipeline(1, DOWN_K_BLOCKS, stage=2):
                        mlp_block = ob // (MLP_OUT_CHUNK // DOWN_K_CHUNK)
                        mlp_offset = (ob - mlp_block * (MLP_OUT_CHUNK // DOWN_K_CHUNK)) * DOWN_K_CHUNK
                        down_mlp_chunk = mlp_tile[mlp_block : mlp_block + 1, :, :, mlp_offset : mlp_offset + DOWN_K_CHUNK]
                        w_down_chunk = w_down[ob : ob + 1, di : di + 1, :, :]
                        down_acc = pl.matmul_acc(down_acc, down_mlp_chunk, w_down_chunk)
                    down_proj_tile = pl.assemble(down_proj_tile, down_acc, [di, 0, 0, 0])

            # Stage 8: Final residual writeback.
            for di in pl.parallel(DOWN_N_BLOCKS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_residual"):
                    resid_block = di * (DOWN_N_CHUNK // OUT_PROJ_N_CHUNK)
                    final_down = down_proj_tile[di : di + 1, :, :, :]
                    final_resid = resid1_tile[resid_block : resid_block + 1, :, :, :]
                    final_out = pl.add(final_down, final_resid)
                    out = pl.assemble(out, pl.cast(final_out, target_type=pl.BF16), [di, 0, 0, 0])

            return out

    return Qwen3Decode


def build_tensor_specs(use_max_seq: bool = False):
    import torch
    from golden import TensorSpec

    def init_hidden_states():
        return (torch.rand(HIDDEN_K_BLOCKS, 1, BATCH, Q_PROJ_K_CHUNK) - 0.5) * ACTIVATION_INIT_SCALE

    def init_rms_weight():
        return (torch.rand(HIDDEN_K_BLOCKS, 1, 1, Q_PROJ_K_CHUNK) - 0.5) * ACTIVATION_INIT_SCALE

    def init_wq():
        return torch.rand(HIDDEN_K_BLOCKS, Q_OUT_BLOCKS, Q_PROJ_K_CHUNK, Q_OUT_CHUNK) / HIDDEN ** 0.5

    def init_wk():
        return torch.rand(HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK) / HIDDEN ** 0.5

    def init_wv():
        return torch.rand(HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK) / HIDDEN ** 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((BATCH, 1, 1, 1), MAX_SEQ, dtype=torch.int32)
        return torch.randint(1, MAX_SEQ + 1, (BATCH, 1, 1, 1), dtype=torch.int32)

    def init_rope_cos():
        return (torch.rand(MAX_SEQ, 1, 1, HEAD_DIM) - 0.5) * ROPE_INIT_SCALE

    def init_rope_sin():
        return (torch.rand(MAX_SEQ, 1, 1, HEAD_DIM) - 0.5) * ROPE_INIT_SCALE

    def init_k_cache():
        return (torch.rand(BATCH * NUM_KV_HEADS, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM) - 0.5) * CACHE_INIT_SCALE

    def init_v_cache():
        return (torch.rand(BATCH * NUM_KV_HEADS, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM) - 0.5) * CACHE_INIT_SCALE

    def init_wo():
        return (torch.rand(OUT_PROJ_K_BLOCKS, OUT_PROJ_N_BLOCKS, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK) - 0.5) * OUT_PROJ_INIT_SCALE / HIDDEN ** 0.5

    def init_post_rms_weight():
        return torch.ones(K_BLOCKS, 1, 1, K_CHUNK) * POST_RMS_INIT_SCALE

    def init_w_gate():
        return (torch.rand(K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK) - 0.5) / HIDDEN ** 0.5

    def init_w_up():
        return (torch.rand(K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK) - 0.5) / HIDDEN ** 0.5

    def init_w_down():
        return (torch.rand(DOWN_K_BLOCKS, DOWN_N_BLOCKS, DOWN_K_CHUNK, DOWN_N_CHUNK) - 0.5) / INTERMEDIATE ** 0.5

    return [
        TensorSpec("hidden_states", [HIDDEN_K_BLOCKS, 1, BATCH, Q_PROJ_K_CHUNK], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [HIDDEN_K_BLOCKS, 1, 1, Q_PROJ_K_CHUNK], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [HIDDEN_K_BLOCKS, Q_OUT_BLOCKS, Q_PROJ_K_CHUNK, Q_OUT_CHUNK], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [HIDDEN_K_BLOCKS, KV_OUT_BLOCKS, KV_PROJ_K_CHUNK, KV_OUT_CHUNK], torch.bfloat16, init_value=init_wv),
        TensorSpec("seq_lens", [BATCH, 1, 1, 1], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [MAX_SEQ, 1, 1, HEAD_DIM], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, 1, 1, HEAD_DIM], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [BATCH * NUM_KV_HEADS, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [BATCH * NUM_KV_HEADS, MAX_CTX_BLOCKS, SEQ_TILE, HEAD_DIM], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("wo", [OUT_PROJ_K_BLOCKS, OUT_PROJ_N_BLOCKS, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [K_BLOCKS, 1, 1, K_CHUNK], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [K_BLOCKS, MLP_OUT_BLOCKS, K_CHUNK, MLP_OUT_CHUNK], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [DOWN_K_BLOCKS, DOWN_N_BLOCKS, DOWN_K_CHUNK, DOWN_N_CHUNK], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [DOWN_N_BLOCKS, 1, BATCH, DOWN_N_CHUNK], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_decode(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import math

    import torch

    hidden_states_chunked = tensors["hidden_states"]
    input_rms_weight_chunked = tensors["input_rms_weight"]
    wq_chunked = tensors["wq"]
    wk_chunked = tensors["wk"]
    wv_chunked = tensors["wv"]
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo_chunked = tensors["wo"]
    post_rms_weight_chunked = tensors["post_rms_weight"]
    w_gate_chunked = tensors["w_gate"]
    w_up_chunked = tensors["w_up"]
    w_down_chunked = tensors["w_down"]

    hidden_states = hidden_states_chunked[:, 0, :, :].permute(1, 0, 2).reshape(BATCH, HIDDEN)
    input_rms_weight = input_rms_weight_chunked[:, 0, :, :].permute(1, 0, 2).reshape(1, HIDDEN)
    post_rms_weight = post_rms_weight_chunked[:, 0, :, :].permute(1, 0, 2).reshape(1, HIDDEN)

    half = HEAD_DIM // 2
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # ── Scope 1 golden: RMSNorm + Q/K/V projection ──
    x_tile = hidden_states.float()
    sq_sum = (x_tile ** 2).sum(dim=-1, keepdim=True)
    variance = sq_sum / HIDDEN + EPS
    rms = torch.sqrt(variance)
    normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

    wq = wq_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, HIDDEN)
    wk = wk_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, KV_HIDDEN)
    wv = wv_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, KV_HIDDEN)
    q_proj = (normed.float() @ wq.float()).float()
    k_proj = (normed.float() @ wk.float()).float()
    v_proj = (normed.float() @ wv.float()).float()

    # ── Scope 2 golden: RoPE + cache update + attention ──
    attn_proj_tile = torch.zeros(OUT_PROJ_K_BLOCKS, BATCH, OUT_PROJ_K_CHUNK, dtype=torch.bfloat16)

    for b in range(BATCH):
        ctx_len = seq_lens[b, 0, 0, 0].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos, 0, :, :]
        sin_row = rope_sin[pos, 0, :, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(NUM_KV_HEADS, HEAD_DIM)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(NUM_KV_HEADS):
            cache_idx = b * NUM_KV_HEADS + ki
            pos_block = pos // SEQ_TILE
            pos_offset = pos - pos_block * SEQ_TILE
            k_cache[cache_idx, pos_block, pos_offset, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_idx, pos_block, pos_offset, :] = v_proj[b, ki * HEAD_DIM : (ki + 1) * HEAD_DIM].to(torch.bfloat16)

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
                    cache_idx = b * NUM_KV_HEADS + kvh
                    k_tile = k_cache[cache_idx, sb, :, :]
                    v_tile = v_cache[cache_idx, sb, :, :]

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
                    attn_proj_tile[qh, b, :] = ctx[qi].to(torch.bfloat16)

    # ── Scope 3 golden: output projection + residual + post RMSNorm + MLP + residual ──
    out_proj_chunks = []
    for oi in range(OUT_PROJ_N_BLOCKS):
        o_acc_lo = torch.matmul(attn_proj_tile[0].float(), wo_chunked[0, oi, :, :Q_OUT_CHUNK].float())
        o_acc_hi = torch.matmul(attn_proj_tile[0].float(), wo_chunked[0, oi, :, Q_OUT_CHUNK:].float())
        for kb in range(1, OUT_PROJ_K_BLOCKS):
            o_acc_lo = o_acc_lo + torch.matmul(attn_proj_tile[kb].float(), wo_chunked[kb, oi, :, :Q_OUT_CHUNK].float())
            o_acc_hi = o_acc_hi + torch.matmul(attn_proj_tile[kb].float(), wo_chunked[kb, oi, :, Q_OUT_CHUNK:].float())
        out_proj_chunks.append(torch.cat([o_acc_lo, o_acc_hi], dim=-1))
    o_proj = torch.cat(out_proj_chunks, dim=-1)
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    w_gate = w_gate_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, INTERMEDIATE)
    w_up = w_up_chunked.permute(0, 2, 1, 3).reshape(HIDDEN, INTERMEDIATE)
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    mlp_blocks = mlp_bf16.reshape(BATCH, MLP_OUT_BLOCKS, MLP_OUT_CHUNK)
    down_chunks = []
    for di in range(DOWN_N_BLOCKS):
        down_acc = None
        for ob in range(DOWN_K_BLOCKS):
            mlp_block = ob // (MLP_OUT_CHUNK // DOWN_K_CHUNK)
            mlp_offset = (ob - mlp_block * (MLP_OUT_CHUNK // DOWN_K_CHUNK)) * DOWN_K_CHUNK
            mlp_chunk = mlp_blocks[:, mlp_block, mlp_offset : mlp_offset + DOWN_K_CHUNK].float()
            w_down_chunk = w_down_chunked[ob, di, :, :].float()
            down_part = torch.matmul(mlp_chunk, w_down_chunk)
            down_acc = down_part if down_acc is None else down_acc + down_part
        down_chunks.append(down_acc)
    down = torch.stack(down_chunks, dim=1).reshape(BATCH, HIDDEN)

    out_flat = (down + resid1).bfloat16()
    tensors["out"][:] = out_flat.reshape(BATCH, DOWN_N_BLOCKS, DOWN_N_CHUNK).permute(1, 0, 2).unsqueeze(1)


if __name__ == "__main__":
    import argparse
    import torch
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(0)

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
