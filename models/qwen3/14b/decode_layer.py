# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B single-layer decode forward.

Scope 1:
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Per-head q_norm / k_norm

Scope 2:
  1. K RoPE + paged cache write, V paged cache write, Q RoPE + pad
  2. fa_qks (one mixed root): QK matmul (cube) + tail-masked softmax
     (vec) for both paired streams (gi0 + gi1).
  3. fa_svo (one mixed root per gi stream): SV matmul (cube) + online-
     softmax recurrence (vec). Issued twice per outer iter (gi0 then
     gi1). Stashes the final FP32 ctx [Q_HEAD_PAD, HEAD_DIM] to GM.
  4. fa_attn_row (vec-only): trim Q_HEAD_PAD->Q_HEAD_BATCH=5, cast to
     BF16, and assemble into attn_row.

  Notes on why not a single fa_fused (the qwen3_14b_decode_mix style):
  - The Q_HEAD_BATCH=5 trim cannot live inside a mixed root: with
    SplitMode.UP_DOWN the split-dim 5 is odd and rejected; without
    UP_DOWN, pto.subview rejects the [5, HEAD_DIM] slice of the
    row_expand_div output (valid_row inference fails).
  - Co-packing both paired streams' SV + online recurrences in one
    fa_svo produces small numerical drift (compiles and runs but
    fails atol=3e-3 tolerance). Splitting fa_svo per stream restores
    correctness.
  - BLOCK_SIZE=128 (SEQ_TILE=128 in config.py) is required so each K/V
    tile is 32 KB and double-buffered L0B fits the 64 KB platform
    limit. With SEQ_TILE=256, any cube+vec fusion overflows L0B.

Scope 3:
  1. Output projection: attn_out × wo
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition

The final RMSNorm and LM head projection live in rms_lm_head.py.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    DOWN_MLP_CHUNK,
    DOWN_OUT_CHUNK,
    EPS,
    FINAL_RMS_K_CHUNK,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN,
    KV_OUT_CHUNK,
    KV_PROJ_K_CHUNK,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ,
    MLP_OUT_CHUNK,
    NUM_HEADS,
    NUM_KV_HEADS,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_GROUPS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_OUT_CHUNK,
    Q_PER_KV,
    ROPE_SEQ_DYN,
    TOTAL_Q_GROUPS,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)
from rms_lm_head import rms_lm_head


@pl.jit.inline
def decode_layer(
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK
    hidden_blocks = HIDDEN // K_CHUNK
    decode_q_out_blocks = HIDDEN // Q_OUT_CHUNK
    decode_mlp_out_blocks = INTERMEDIATE // MLP_OUT_CHUNK
    head_dim_inv = HEAD_DIM_INV
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    batch_padded = BATCH
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * decode_layer_cache_rows

    # Intermediate FP32 tensors between scope 1 and scope 2.
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # Scope 1: input RMSNorm + Q/K/V projection.
    # The JIT inline path follows the fixed-BATCH single-layer kernel
    # contract, so every matmul tile has a static M dim of BATCH_TILE.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(decode_scope1_hidden_blocks):
                sq_k0 = kb * INPUT_PROJ_K_CHUNK
                sq_chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                        [b0, sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(sq_chunk, sq_chunk)), [1, BATCH_TILE]),
                )
            variance = pl.reshape(
                pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                [BATCH_TILE, 1],
            )
            inv_rms = pl.recip(pl.sqrt(variance))

            for kb in pl.range(decode_scope1_hidden_blocks):
                norm_k0 = kb * INPUT_PROJ_K_CHUNK
                norm_chunk = pl.cast(
                    pl.slice(
                        current_hidden,
                        [BATCH_TILE, INPUT_PROJ_K_CHUNK],
                        [b0, norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(input_rms_weight, [1, INPUT_PROJ_K_CHUNK], [layer_idx, norm_k0])
                normed = pl.col_expand_mul(pl.row_expand_mul(norm_chunk, inv_rms), gamma)
                normed_tile = pl.assemble(
                    normed_tile,
                    pl.cast(normed, target_type=pl.BF16),
                    [0, norm_k0],
                )

        for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    q_k0 = kb * INPUT_PROJ_K_CHUNK
                    q_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, q_k0])
                    q_tile_b = pl.slice(wq, [INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q0])
                    if q_k0 == 0:
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
                q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

        for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
                k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    k_k0 = kb * INPUT_PROJ_K_CHUNK
                    k_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, k_k0])
                    k_tile_b = pl.slice(wk, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_k0, kv0])
                    if k_k0 == 0:
                        k_acc = pl.matmul(k_tile_a, k_tile_b, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
                k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
                v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(decode_scope1_hidden_blocks):
                    v_k0 = kb * INPUT_PROJ_K_CHUNK
                    v_tile_a = pl.slice(normed_tile, [BATCH_TILE, INPUT_PROJ_K_CHUNK], [0, v_k0])
                    v_tile_b = pl.slice(wv, [INPUT_PROJ_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_k0, kv0])
                    if v_k0 == 0:
                        v_acc = pl.matmul(v_tile_a, v_tile_b, out_dtype=pl.FP32)
                    else:
                        v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
                v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

    # HF-style per-head q_norm / k_norm before RoPE, matching the original
    # single-layer qwen3_decode grouping by KV head.
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm"):
            for h in pl.range(NUM_KV_HEADS):
                q0 = h * Q_PER_KV * HEAD_DIM
                q_chunk = pl.reshape(
                    pl.slice(q_proj, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM], [b0, q0]),
                    [BATCH_TILE * Q_HEAD_BATCH, HEAD_DIM],
                )
                q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, head_dim_inv), EPS))
                q_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(q_chunk, q_inv_rms),
                    pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                q_chunk_norm_flat = pl.reshape(q_chunk_norm, [BATCH_TILE, Q_HEAD_BATCH * HEAD_DIM])
                q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm_flat, [b0, q0])

                k0 = h * HEAD_DIM
                k_chunk = pl.slice(k_proj, [BATCH_TILE, HEAD_DIM], [b0, k0])
                k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, head_dim_inv), EPS))
                k_chunk_norm = pl.col_expand_mul(
                    pl.row_expand_mul(k_chunk, k_inv_rms),
                    pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                )
                k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])

    # Scope 2: RoPE + KV cache update + grouped decode attention.
    # This follows the original single-layer qwen3_decode paired-gi
    # attention structure, with layer_cache_base added for full-model caches.
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
    )

    # Scope 2 only touches runtime-visible rows; padded rows stay zero.
    for b in pl.parallel(user_batch):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table_base = b * MAX_BLOCKS_PER_SEQ
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HEAD_DIM
                cache_row = layer_cache_base + (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset
                k_lo = pl.slice(k_proj_norm, [1, HALF_DIM], [b, kv_col])
                k_hi = pl.slice(k_proj_norm, [1, HALF_DIM], [b, kv_col + HALF_DIM])
                rot_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo),
                    pl.col_expand_mul(k_hi, sin_lo),
                )
                rot_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi),
                    pl.col_expand_mul(k_lo, sin_hi),
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(rot_lo, target_type=pl.BF16),
                    [cache_row, 0],
                )
                k_cache = pl.assemble(
                    k_cache,
                    pl.cast(rot_hi, target_type=pl.BF16),
                    [cache_row, HALF_DIM],
                )
                v_cache = pl.assemble(
                    v_cache,
                    pl.cast(
                        pl.slice(v_proj, [1, HEAD_DIM], [b, kv_col]),
                        target_type=pl.BF16,
                    ),
                    [cache_row, 0],
                )
                q_base = ki * Q_PER_KV
                q_block = pl.reshape(
                    pl.slice(q_proj_norm, [1, Q_HEAD_BATCH * HEAD_DIM], [b, q_base * HEAD_DIM]),
                    [Q_HEAD_BATCH, HEAD_DIM],
                )
                q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
                q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, HALF_DIM])
                rot_lo_bf16 = pl.cast(
                    pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                    target_type=pl.BF16,
                )
                rot_hi_bf16 = pl.cast(
                    pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                    target_type=pl.BF16,
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_lo_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, 0],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    rot_hi_bf16,
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD, HALF_DIM],
                )
                all_q_padded = pl.assemble(
                    all_q_padded,
                    pl.cast(
                        pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        target_type=pl.BF16,
                    ),
                    [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                )

        attn_row = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
        for gi in pl.parallel(0, TOTAL_Q_GROUPS, 2):
            gi0 = gi
            gi1 = gi + 1

            kvh0 = gi0 // Q_GROUPS
            qg0 = gi0 - kvh0 * Q_GROUPS
            q_base0 = kvh0 * Q_PER_KV + qg0 * Q_HEAD_BATCH
            q_padded_row0 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi0 * Q_HEAD_PAD
            q_padded0 = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [q_padded_row0, 0])

            kvh1 = gi1 // Q_GROUPS
            qg1 = gi1 - kvh1 * Q_GROUPS
            q_base1 = kvh1 * Q_PER_KV + qg1 * Q_HEAD_BATCH
            q_padded_row1 = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi1 * Q_HEAD_PAD
            q_padded1 = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [q_padded_row1, 0])

            all_raw_scores0 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32)
            all_raw_scores1 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32)
            all_exp_padded0 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16)
            all_exp_padded1 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16)
            all_cur_mi0 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1], dtype=pl.FP32)
            all_cur_mi1 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1], dtype=pl.FP32)
            all_cur_li0 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1], dtype=pl.FP32)
            all_cur_li1 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, 1], dtype=pl.FP32)

            # fa_qks: cube (QK matmul) + vec (tail-masked softmax) for both
            # paired streams in one mixed root. Per-sb loop body: each stream
            # matmuls Q against its kvh K tile, stashes raw scores to GM,
            # then reads back via bracket slicing, attaches dynamic tail mask
            # via pl.set_validshape (the pl.slice(..., valid_shape=...)
            # pattern is disallowed inside a mixed root), and runs softmax.
            # Requires SEQ_TILE == 128 in config.py: each K tile is 32 KB and
            # the cube L0B holds at most two double-buffered tiles in 64 KB.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_qks"):
                for sb in pl.range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = pl.min(BLOCK_SIZE, ctx_len - s0)
                    qk_block_table_idx = block_table_base + sb
                    qk_pbid = pl.cast(pl.tensor.read(block_table, [qk_block_table_idx]), pl.INDEX)

                    qk_cache_row0 = layer_cache_base + (qk_pbid * NUM_KV_HEADS + kvh0) * BLOCK_SIZE
                    k_tile0 = k_cache[qk_cache_row0 : qk_cache_row0 + BLOCK_SIZE, :]
                    raw_scores0 = pl.matmul(q_padded0, k_tile0, b_trans=True, out_dtype=pl.FP32)
                    all_raw_scores0 = pl.assemble(all_raw_scores0, raw_scores0, [sb * Q_HEAD_PAD, 0])
                    scores_chunk0 = all_raw_scores0[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    scores_valid0 = pl.set_validshape(scores_chunk0, Q_HEAD_PAD, valid_len)
                    scores_padded0 = pl.fillpad(scores_valid0, pad_value=pl.PadValue.min)
                    scores0 = pl.mul(scores_padded0, decode_attn_scale)
                    softmax_cur_mi0 = pl.row_max(scores0)
                    exp_scores0 = pl.exp(pl.row_expand_sub(scores0, softmax_cur_mi0))
                    exp_scores_bf16_0 = pl.cast(exp_scores0, target_type=pl.BF16)
                    exp_scores_fp32_0 = pl.cast(exp_scores_bf16_0, target_type=pl.FP32)
                    softmax_cur_li0 = pl.row_sum(exp_scores_fp32_0)
                    all_exp_padded0 = pl.assemble(all_exp_padded0, exp_scores_bf16_0, [sb * Q_HEAD_PAD, 0])
                    all_cur_mi0 = pl.assemble(all_cur_mi0, softmax_cur_mi0, [sb * Q_HEAD_PAD, 0])
                    all_cur_li0 = pl.assemble(all_cur_li0, softmax_cur_li0, [sb * Q_HEAD_PAD, 0])

                    qk_cache_row1 = layer_cache_base + (qk_pbid * NUM_KV_HEADS + kvh1) * BLOCK_SIZE
                    k_tile1 = k_cache[qk_cache_row1 : qk_cache_row1 + BLOCK_SIZE, :]
                    raw_scores1 = pl.matmul(q_padded1, k_tile1, b_trans=True, out_dtype=pl.FP32)
                    all_raw_scores1 = pl.assemble(all_raw_scores1, raw_scores1, [sb * Q_HEAD_PAD, 0])
                    scores_chunk1 = all_raw_scores1[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    scores_valid1 = pl.set_validshape(scores_chunk1, Q_HEAD_PAD, valid_len)
                    scores_padded1 = pl.fillpad(scores_valid1, pad_value=pl.PadValue.min)
                    scores1 = pl.mul(scores_padded1, decode_attn_scale)
                    softmax_cur_mi1 = pl.row_max(scores1)
                    exp_scores1 = pl.exp(pl.row_expand_sub(scores1, softmax_cur_mi1))
                    exp_scores_bf16_1 = pl.cast(exp_scores1, target_type=pl.BF16)
                    exp_scores_fp32_1 = pl.cast(exp_scores_bf16_1, target_type=pl.FP32)
                    softmax_cur_li1 = pl.row_sum(exp_scores_fp32_1)
                    all_exp_padded1 = pl.assemble(all_exp_padded1, exp_scores_bf16_1, [sb * Q_HEAD_PAD, 0])
                    all_cur_mi1 = pl.assemble(all_cur_mi1, softmax_cur_mi1, [sb * Q_HEAD_PAD, 0])
                    all_cur_li1 = pl.assemble(all_cur_li1, softmax_cur_li1, [sb * Q_HEAD_PAD, 0])

            all_oi_tmp0 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
            all_oi_tmp1 = pl.create_tensor([MAX_BLOCKS_PER_SEQ * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
            # FP32 ctx stash: fa_svo writes [Q_HEAD_PAD, HEAD_DIM] FP32 here,
            # fa_attn_row trims/casts/assembles. The trim cannot live inside
            # fa_svo (mixed root): the row_expand_div output's valid_shape
            # stays uninitialised and pto.subview rejects the
            # [Q_HEAD_BATCH=5, HEAD_DIM] subview.
            ctx_gm0 = pl.create_tensor([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
            ctx_gm1 = pl.create_tensor([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)

            # fa_svo: per-stream cube (SV matmul) + vec (online-softmax
            # recurrence). Per-stream split: co-packing both streams in one
            # fa_svo gave a small numeric drift (~10% relative error,
            # confirmed wrong but close to golden), likely from the IR
            # serialising two parallel oi/mi/li recurrences inside a single
            # mixed root. Single-stream matches the unfused recurrence
            # semantics exactly.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_svo"):
                for sb in pl.range(ctx_blocks):
                    sv_block_table_idx = block_table_base + sb
                    sv_pbid = pl.cast(pl.tensor.read(block_table, [sv_block_table_idx]), pl.INDEX)
                    sv_cache_row0 = layer_cache_base + (sv_pbid * NUM_KV_HEADS + kvh0) * BLOCK_SIZE
                    exp_tile0 = all_exp_padded0[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    v_tile0 = v_cache[sv_cache_row0 : sv_cache_row0 + BLOCK_SIZE, :]
                    oi_tmp0 = pl.matmul(exp_tile0, v_tile0, out_dtype=pl.FP32)
                    all_oi_tmp0 = pl.assemble(all_oi_tmp0, oi_tmp0, [sb * Q_HEAD_PAD, 0])

                oi0 = all_oi_tmp0[0:Q_HEAD_PAD, :]
                mi0 = all_cur_mi0[0:Q_HEAD_PAD, :]
                li0 = all_cur_li0[0:Q_HEAD_PAD, :]
                for sb in pl.range(1, ctx_blocks):
                    oi_tmp_valid0 = all_oi_tmp0[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    online_cur_mi0 = all_cur_mi0[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    online_cur_li0 = all_cur_li0[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    mi_new0 = pl.maximum(mi0, online_cur_mi0)
                    alpha0 = pl.exp(pl.sub(mi0, mi_new0))
                    beta0 = pl.exp(pl.sub(online_cur_mi0, mi_new0))
                    li0 = pl.add(pl.mul(alpha0, li0), pl.mul(beta0, online_cur_li0))
                    oi0 = pl.add(pl.row_expand_mul(oi0, alpha0), pl.row_expand_mul(oi_tmp_valid0, beta0))
                    mi0 = mi_new0

                ctx0 = pl.row_expand_div(oi0, li0)
                ctx_gm0 = pl.assemble(ctx_gm0, ctx0, [0, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_svo"):
                for sb in pl.range(ctx_blocks):
                    sv_block_table_idx = block_table_base + sb
                    sv_pbid = pl.cast(pl.tensor.read(block_table, [sv_block_table_idx]), pl.INDEX)
                    sv_cache_row1 = layer_cache_base + (sv_pbid * NUM_KV_HEADS + kvh1) * BLOCK_SIZE
                    exp_tile1 = all_exp_padded1[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    v_tile1 = v_cache[sv_cache_row1 : sv_cache_row1 + BLOCK_SIZE, :]
                    oi_tmp1 = pl.matmul(exp_tile1, v_tile1, out_dtype=pl.FP32)
                    all_oi_tmp1 = pl.assemble(all_oi_tmp1, oi_tmp1, [sb * Q_HEAD_PAD, 0])

                oi1 = all_oi_tmp1[0:Q_HEAD_PAD, :]
                mi1 = all_cur_mi1[0:Q_HEAD_PAD, :]
                li1 = all_cur_li1[0:Q_HEAD_PAD, :]
                for sb in pl.range(1, ctx_blocks):
                    oi_tmp_valid1 = all_oi_tmp1[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    online_cur_mi1 = all_cur_mi1[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    online_cur_li1 = all_cur_li1[sb * Q_HEAD_PAD : (sb + 1) * Q_HEAD_PAD, :]
                    mi_new1 = pl.maximum(mi1, online_cur_mi1)
                    alpha1 = pl.exp(pl.sub(mi1, mi_new1))
                    beta1 = pl.exp(pl.sub(online_cur_mi1, mi_new1))
                    li1 = pl.add(pl.mul(alpha1, li1), pl.mul(beta1, online_cur_li1))
                    oi1 = pl.add(pl.row_expand_mul(oi1, alpha1), pl.row_expand_mul(oi_tmp_valid1, beta1))
                    mi1 = mi_new1

                ctx1 = pl.row_expand_div(oi1, li1)
                ctx_gm1 = pl.assemble(ctx_gm1, ctx1, [0, 0])

            # fa_attn_row: vec-only trim + cast + assemble.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_attn_row"):
                ctx_valid0 = pl.slice(ctx_gm0, [Q_HEAD_BATCH, HEAD_DIM], [0, 0])
                ctx_flat_bf16_0 = pl.cast(
                    pl.reshape(ctx_valid0, [1, Q_HEAD_BATCH * HEAD_DIM]),
                    target_type=pl.BF16,
                )
                attn_row = pl.assemble(attn_row, ctx_flat_bf16_0, [0, q_base0 * HEAD_DIM])

                ctx_valid1 = pl.slice(ctx_gm1, [Q_HEAD_BATCH, HEAD_DIM], [0, 0])
                ctx_flat_bf16_1 = pl.cast(
                    pl.reshape(ctx_valid1, [1, Q_HEAD_BATCH * HEAD_DIM]),
                    target_type=pl.BF16,
                )
                attn_row = pl.assemble(attn_row, ctx_flat_bf16_1, [0, q_base1 * HEAD_DIM])

        attn_out = pl.assemble(attn_out, attn_row, [b, 0])

    # Scope 3: output projection + residual + post RMSNorm + MLP + residual.
    # Loops over batch_padded so every iteration processes a full
    # [BATCH_TILE, *] tile (a2a3 matmul M-tile constraint).
    # Final down-proj + residual + cast uses the two-incore pattern
    # validated in dynamic_batch_pad_repro:
    #   cube incore : matmul_acc -> FP32 -> assemble to GM scratch
    #   vec incore  : tload FP32 chunk -> add FP32 resid -> cast BF16
    #                 (preserves ND layout) -> assemble to out
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        for ob in pl.range(decode_q_out_blocks):
            o0 = ob * Q_OUT_CHUNK

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
                a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, o0])
                o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                for kb in pl.range(1, hidden_blocks):
                    k0 = kb * K_CHUNK
                    a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                    w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, o0])
                    o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_residual"):
                resid = pl.cast(
                    pl.slice(current_hidden, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                    target_type=pl.FP32,
                )
                resid_sum = pl.add(o_acc, resid)
                resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)), [1, BATCH_TILE]),
                )
            inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

            for kb in pl.range(hidden_blocks):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0])
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0])
                post_normed = pl.col_expand_mul(
                    pl.row_expand_mul(post_norm_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])),
                    post_gamma,
                )
                normed_bf16 = pl.cast(post_normed, target_type=pl.BF16)
                post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, post_norm_k0])

        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        for ob in pl.range(decode_mlp_out_blocks):
            mlp_o0 = ob * MLP_OUT_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0])
                gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                for kb in pl.range(1, hidden_blocks):
                    gate_k0 = kb * K_CHUNK
                    gate_post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, gate_k0])
                    wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + gate_k0, mlp_o0])
                    gate_acc = pl.matmul_acc(gate_acc, gate_post_chunk, wg)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0])
                up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                for kb in pl.range(1, hidden_blocks):
                    up_k0 = kb * K_CHUNK
                    up_post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, up_k0])
                    wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + up_k0, mlp_o0])
                    up_acc = pl.matmul_acc(up_acc, up_post_chunk, wu)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

        for dob in pl.range(hidden_blocks):
            d0 = dob * K_CHUNK
            # FP32 GM scratch chunk used as the cube -> vec bridge.
            # Per-iter [BATCH_TILE, K_CHUNK] is small (16*256*4 =
            # 8 KiB) and avoids a large pre-allocated scratch.
            fp32_chunk_gm = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj"):
                mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0])
                down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                for ob in pl.range(1, decode_mlp_out_blocks):
                    down_o0 = ob * MLP_OUT_CHUNK
                    down_mlp_chunk_bf16 = pl.slice(
                        mlp_tile,
                        [BATCH_TILE, MLP_OUT_CHUNK],
                        [0, down_o0],
                    )
                    w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + down_o0, d0])
                    down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
                fp32_chunk_gm = pl.assemble(fp32_chunk_gm, down_acc, [0, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj_residual"):
                # Vec-only incore: tload FP32 cube output as ND vec
                # tile, add FP32 residual (also ND vec), cast to
                # BF16 (vec-to-vec cast preserves ND layout).
                down_chunk_fp32 = pl.slice(fp32_chunk_gm, [BATCH_TILE, K_CHUNK], [0, 0])
                resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
                out_chunk = pl.add(down_chunk_fp32, resid_chunk_fp32)
                out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                next_hidden = pl.assemble(next_hidden, out_chunk_cast, [b0, d0])

    return next_hidden


@pl.jit
def test_decode_layer(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    user_batch = pl.tensor.dim(hidden_states, 0)
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(HIDDEN // K_CHUNK):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    current_hidden = decode_layer(
        current_hidden,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        next_hidden,
        0,
    )
    out = rms_lm_head(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    vocab_size: int = VOCAB,
    use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    # Host allocates every batch-dependent tensor at the user-visible
    # batch (no host pad / no host trim). The kernel internally rounds
    # up to BATCH_TILE, zero-pads via valid_shape on input loads, and
    # trims via vec-to-vec textract on the BF16 output. A single
    # compiled program serves any batch <= host capacity (USER_BATCH_DYN
    # / KV_CACHE_ROWS_DYN / BLOCK_TABLE_FLAT_DYN are pl.dynamic dims).
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    vocab = vocab_size
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    synthetic_proj_scale = 0.5

    if use_max_seq:
        seq_lens_seed = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.rand(hidden_size, hidden_size) / hidden_size ** 0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.ones(1, head_dim)

    def init_k_norm_weight():
        return torch.ones(1, head_dim)

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return synthetic_proj_scale * (torch.rand(cache_rows, head_dim) - 0.5)

    def init_wo():
        return synthetic_proj_scale * (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return synthetic_proj_scale * (torch.rand(hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return synthetic_proj_scale * (torch.rand(inter, hidden_size) - 0.5) / inter ** 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return synthetic_proj_scale * (torch.rand(vocab, hidden_size) - 0.5) / hidden_size ** 0.5

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
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * MAX_BLOCKS_PER_SEQ], torch.int32,
                   init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32,
                   init_value=init_slot_mapping),
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
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32,
                   init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16,
                   init_value=init_lm_head_weight),
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


def golden_decode_layer(tensors):
    """PyTorch reference: scope1 (RMSNorm + projection), scope2 (attention), scope3 (output + MLP)."""
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ

    def tiled_matmul(lhs, rhs, k_chunk, n_chunk):
        out = torch.zeros(lhs.shape[0], rhs.shape[1], dtype=torch.float32)
        for n0 in range(0, rhs.shape[1], n_chunk):
            acc = torch.zeros(lhs.shape[0], n_chunk, dtype=torch.float32)
            for k0 in range(0, lhs.shape[1], k_chunk):
                acc = acc + lhs[:, k0 : k0 + k_chunk].float() @ rhs[
                    k0 : k0 + k_chunk,
                    n0 : n0 + n_chunk,
                ].float()
            out[:, n0 : n0 + n_chunk] = acc
        return out

    def chunked_row_sq_sum(x, k_chunk):
        acc = torch.zeros(x.shape[0], 1, dtype=torch.float32)
        for k0 in range(0, x.shape[1], k_chunk):
            x_chunk = x[:, k0 : k0 + k_chunk]
            acc = acc + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        return acc

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            out = out + lhs[:, k0 : k0 + k_chunk].float() @ rhs_t[:, k0 : k0 + k_chunk].float().T
        return out

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, INPUT_PROJ_K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + INPUT_PROJ_K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = tiled_matmul(normed, wq, INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK)
        k_proj[b0:b_end, :] = tiled_matmul(normed, wk, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)
        v_proj[b0:b_end, :] = tiled_matmul(normed, wv, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_variance = k_heads.pow(2).mean(dim=-1, keepdim=True)
        k_heads = k_heads * torch.rsqrt(k_variance + eps) * k_norm_weight.float()
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE

        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_variance = q_heads.pow(2).mean(dim=-1, keepdim=True)
        q_heads = q_heads * torch.rsqrt(q_variance + eps) * q_norm_weight.float()
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
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
                ctx_flat_bf16 = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row[
                    :,
                    q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                ] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    o_proj = tiled_matmul(attn_out, wo, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK)
    resid1 = o_proj + hidden_states.float()

    variance = chunked_row_sq_sum(resid1, K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = tiled_matmul(normed_bf16, w_gate, K_CHUNK, MLP_OUT_CHUNK)
    up = tiled_matmul(normed_bf16, w_up, K_CHUNK, MLP_OUT_CHUNK)
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = tiled_matmul(mlp_bf16, w_down, DOWN_MLP_CHUNK, DOWN_OUT_CHUNK)

    final_hidden = (down + resid1).bfloat16()

    variance = chunked_row_sq_sum(final_hidden.float(), FINAL_RMS_K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    final_normed = (final_hidden.float() * inv_rms * final_norm_weight.float()).bfloat16()

    tensors["out"][:] = tiled_lm_head(
        final_normed,
        lm_head_weight,
        LM_HEAD_K_CHUNK,
        VOCAB_CHUNK,
    )


if __name__ == "__main__":
    import argparse
    import sys
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH,
                        help=("User-visible batch size. Host allocates every "
                              "batch-dependent tensor at exactly this size; "
                              "the kernel internally rounds up to BATCH_TILE "
                              "(%d), zero-pads input loads via valid_shape, "
                              "and trims the BF16 output via vec-to-vec "
                              "textract. A single compiled program serves "
                              "any batch <= host KV-cache capacity. Default: "
                              "%%(default)s" % BATCH_TILE))
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument(
        "--export-kernel-insight",
        action="store_true",
        default=False,
        help=(
            "After a successful run, export msprof op-simulator Insight traces "
            "for all generated InCore kernels under the same build_output dir."
        ),
    )
    parser.add_argument(
        "--kernel-insight-func",
        action="append",
        default=[],
        help="Only export this generated kernel function; can be repeated.",
    )
    args = parser.parse_args()

    result = run_jit(
        fn=test_decode_layer,
        specs=build_tensor_specs(batch=args.batch, use_max_seq=args.max_seq),
        golden_fn=golden_decode_layer,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=3e-3,
        atol=3e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    if args.export_kernel_insight:
        if result.work_dir is None:
            print("kernel insight export failed: run result has no build_output directory", file=sys.stderr)
            raise SystemExit(1)
        from tools.export_all_kernel_insight import StepError, main as export_kernel_insight

        export_args = ["--build-dir", str(result.work_dir)]
        for func in args.kernel_insight_func:
            export_args.extend(["--func", func])
        try:
            export_rc = export_kernel_insight(export_args)
        except StepError as exc:
            print(f"kernel insight export failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if export_rc != 0:
            raise SystemExit(export_rc)
