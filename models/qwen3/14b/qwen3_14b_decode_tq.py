# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-14B TQ decode forward (single-file).

Contains the single-layer decode operator (decode_layer_tq) and the
full multi-layer runner (decode_fwd_tq), matching the prefill pattern.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

# pyright: reportUndefinedVariable=false


import math as _math

from turboquant_kv import (
    turboquant_kv_quantize,
    turboquant_kv_dequant_chunk,
)
from rms_lm_head import rms_lm_head

from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE,
    K_CHUNK,
    KV_HIDDEN,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    MAX_BLOCKS_PER_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    Q_GROUPS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_OUT_CHUNK,
    Q_PER_KV,
    ROPE_SEQ_DYN,
    TOTAL_Q_GROUPS,
    USER_BATCH_DYN,
    VOCAB,
)

# pyright: reportUndefinedVariable=false


from config import (
    MAX_SEQ,
    NUM_LAYERS,
)

# TQ-specific constants (not in config.py).

# Dynamic dims specific to TQ.
QUANT_CACHE_ROWS_DYN = pl.dynamic("QUANT_CACHE_ROWS_DYN")
KV_NORMS_OUT_DYN = pl.dynamic("KV_NORMS_OUT_DYN")

# Tiling constants (TQ-specific, differ from config.py values).
SCOPE1_K_CHUNK = 512
KV_OUT_CHUNK = 64
CMP_TILE = 64  # K dequant sub-tile (gather-based, matches turboquant_kv)
CMP_TILE_SV = 64  # V dequant sub-tile (BLOCK_SIZE=128 = 2 * CMP_TILE_SV)

# TurboQuant constants.
CMP_CHUNK = 32  # Sub-chunk for gather: 32 rows * 1B (UINT8) = 32-byte aligned
N_LEVELS = 16  # int4 -> 16 levels


# Derived constants.
MAX_CTX_BLOCKS = MAX_BLOCKS_PER_SEQ
SCOPE1_HIDDEN_BLOCKS = HIDDEN // SCOPE1_K_CHUNK
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // 256  # MLP_OUT_CHUNK = 256


@pl.jit.inline
def decode_layer_tq(
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
    quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    rot_matrices: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HEAD_DIM], pl.BF16],
    tq_codebook: pl.Tensor[[CMP_CHUNK, N_LEVELS], pl.FP32],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    # NOTE: Q_OUT_BLOCKS, HIDDEN_BLOCKS, MLP_OUT_BLOCKS are used directly
    # in pl.spmd(...) — pl.spmd outlines its body to a top-level function
    # and SSA-verifies the block count outside the JIT-inlined scope, so a
    # local alias defined here would trip "used outside its defining scope".
    head_dim_inv = HEAD_DIM_INV
    decode_attn_scale = ATTN_SCALE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    decode_layer_cmp_cache_rows = pl.tensor.dim(quant_k_cache, 0) // num_layers_actual
    user_batch = pl.tensor.dim(seq_lens, 0)
    batch_padded = BATCH
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cmp_base = layer_idx * decode_layer_cmp_cache_rows
    rot_base = layer_idx * HEAD_DIM
    rot_slice = pl.slice(rot_matrices, [HEAD_DIM, HEAD_DIM], [rot_base, 0])

    # Intermediate FP32 tensors between scope 1 and scope 2.
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    # ── Scope 1: input RMSNorm + Q/K/V projection ──
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                sq_k0 = kb * SCOPE1_K_CHUNK
                sq_chunk = pl.cast(
                    pl.slice(current_hidden, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, sq_k0]),
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

            for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                norm_k0 = kb * SCOPE1_K_CHUNK
                norm_chunk = pl.cast(
                    pl.slice(current_hidden, [BATCH_TILE, SCOPE1_K_CHUNK], [b0, norm_k0]),
                    target_type=pl.FP32,
                )
                gamma = pl.slice(input_rms_weight, [1, SCOPE1_K_CHUNK], [layer_idx, norm_k0])
                normed = pl.col_expand_mul(pl.row_expand_mul(norm_chunk, inv_rms), gamma)
                normed_tile = pl.assemble(
                    normed_tile, pl.cast(normed, target_type=pl.BF16), [0, norm_k0],
                )

        # Q projection.
        for q0 in pl.parallel(0, HIDDEN, Q_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    q_k0 = kb * SCOPE1_K_CHUNK
                    q_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, q_k0])
                    q_tile_b = pl.slice(wq, [SCOPE1_K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + q_k0, q0])
                    if q_k0 == 0:
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_tile_a, q_tile_b)
                q_proj = pl.assemble(q_proj, q_acc, [b0, q0])

        # K/V projection.
        for kv0 in pl.parallel(0, KV_HIDDEN, KV_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
                k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    k_k0 = kb * SCOPE1_K_CHUNK
                    k_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, k_k0])
                    k_tile_b = pl.slice(wk, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k_k0, kv0])
                    if k_k0 == 0:
                        k_acc = pl.matmul(k_tile_a, k_tile_b, out_dtype=pl.FP32)
                    else:
                        k_acc = pl.matmul_acc(k_acc, k_tile_a, k_tile_b)
                k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
                v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.range(SCOPE1_HIDDEN_BLOCKS):
                    v_k0 = kb * SCOPE1_K_CHUNK
                    v_tile_a = pl.slice(normed_tile, [BATCH_TILE, SCOPE1_K_CHUNK], [0, v_k0])
                    v_tile_b = pl.slice(wv, [SCOPE1_K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + v_k0, kv0])
                    if v_k0 == 0:
                        v_acc = pl.matmul(v_tile_a, v_tile_b, out_dtype=pl.FP32)
                    else:
                        v_acc = pl.matmul_acc(v_acc, v_tile_a, v_tile_b)
                v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

    # Q/K per-head norm (grouped by KV head).
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

    # ── Scope 2: RoPE + TQ inline KV quantization + fused dequant attention ──
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
    )

    # Per-batch: TQ inline KV quantization (via turboquant_kv_quantize) + Q RoPE.
    for b in pl.parallel(user_batch):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE
        cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
        sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
        cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
        cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
        sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
        sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

        # Allocate GM buffers for turboquant_kv_quantize.
        # Rows 1-15 are uninitialized; turboquant_kv_quantize processes each row
        # independently and we only consume row 0 of the output.
        k_padded = pl.create_tensor([16, KV_HIDDEN], dtype=pl.FP32)
        v_padded = pl.create_tensor([16, KV_HIDDEN], dtype=pl.FP32)
        cos_lo_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        cos_hi_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        sin_lo_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        sin_hi_all = pl.create_tensor([16, HALF_DIM], dtype=pl.FP32)
        quant_k_temp = pl.create_tensor([16, KV_HIDDEN // 2], dtype=pl.UINT8)
        quant_v_temp = pl.create_tensor([16, KV_HIDDEN // 2], dtype=pl.UINT8)
        k_scales_buf = pl.create_tensor([16, NUM_KV_HEADS], dtype=pl.FP32)
        v_scales_buf = pl.create_tensor([16, NUM_KV_HEADS], dtype=pl.FP32)

        # Pad K/V [1, KV_HIDDEN] -> [16, KV_HIDDEN] and expand cos/sin.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_pad"):
            k_all = pl.slice(k_proj_norm, [1, KV_HIDDEN], [b, 0])
            k_padded = pl.assemble(k_padded, k_all, [0, 0])

            v_all = pl.slice(v_proj, [1, KV_HIDDEN], [b, 0])
            v_padded = pl.assemble(v_padded, v_all, [0, 0])

            _ones = pl.full([16, HALF_DIM], dtype=pl.FP32, value=1.0)
            cos_lo_all = pl.assemble(
                cos_lo_all, pl.col_expand_mul(_ones, cos_lo), [0, 0],
            )
            cos_hi_all = pl.assemble(
                cos_hi_all, pl.col_expand_mul(_ones, cos_hi), [0, 0],
            )
            sin_lo_all = pl.assemble(
                sin_lo_all, pl.col_expand_mul(_ones, sin_lo), [0, 0],
            )
            sin_hi_all = pl.assemble(
                sin_hi_all, pl.col_expand_mul(_ones, sin_hi), [0, 0],
            )

        # Reuse prefill's TQ KV quantization: K RoPE + L2 norm + normalize +
        # rotate + Lloyd-Max quantize for both K and V.
        turboquant_kv_quantize(
            k_padded, v_padded, rot_slice,
            cos_lo_all, cos_hi_all, sin_lo_all, sin_hi_all,
            quant_k_temp, quant_v_temp, k_scales_buf, v_scales_buf,
        )

        # Scatter row 0 results to per-head cache positions.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
            for ki in pl.range(NUM_KV_HEADS):
                kv_col = ki * HALF_DIM
                quant_cache_row = layer_cmp_base + (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset

                # K: extract row 0, scatter to quantized cache + scale.
                # Index temps are nibble-packed, so each head is HALF_DIM wide.
                k_idx_u8 = pl.slice(quant_k_temp, [1, HALF_DIM], [0, kv_col])
                quant_k_cache = pl.assemble(quant_k_cache, k_idx_u8, [quant_cache_row, 0])
                k_scale = pl.read(k_scales_buf, [0, ki])
                quant_k_scales = pl.write(quant_k_scales, [quant_cache_row, 0], k_scale)

                # V: extract row 0, scatter to quantized cache + scale.
                v_idx_u8 = pl.slice(quant_v_temp, [1, HALF_DIM], [0, kv_col])
                quant_v_cache = pl.assemble(quant_v_cache, v_idx_u8, [quant_cache_row, 0])
                v_scale = pl.read(v_scales_buf, [0, ki])
                quant_v_scales = pl.write(quant_v_scales, [quant_cache_row, 0], v_scale)

        # Q RoPE + pad into all_q_padded — one head at a time (avoids pl.slice
        # batch bug on [Q_HEAD_BATCH, HEAD_DIM] tiles).
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_q_rope"):
            zero_row = pl.cast(pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
            for ki in pl.range(NUM_KV_HEADS):
                q_base = ki * Q_PER_KV
                row_base = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                for qi in pl.range(Q_HEAD_BATCH):
                    q_head = pl.reshape(
                        pl.slice(q_proj_norm, [1, HEAD_DIM], [b, (q_base + qi) * HEAD_DIM]),
                        [1, HEAD_DIM],
                    )
                    q_lo = pl.slice(q_head, [1, HALF_DIM], [0, 0])
                    q_hi = pl.slice(q_head, [1, HALF_DIM], [0, HALF_DIM])
                    rot_lo = pl.cast(
                        pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                        target_type=pl.BF16,
                    )
                    rot_hi = pl.cast(
                        pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                        target_type=pl.BF16,
                    )
                    all_q_padded = pl.assemble(all_q_padded, rot_lo, [row_base + qi, 0])
                    all_q_padded = pl.assemble(all_q_padded, rot_hi, [row_base + qi, HALF_DIM])
                for pi in pl.range(Q_HEAD_PAD - Q_HEAD_BATCH):
                    all_q_padded = pl.assemble(all_q_padded, zero_row,
                        [row_base + Q_HEAD_BATCH + pi, 0])
        # ── K/V scales + norms: TODO — requires scalar write or layout change ──
        # Constraint: Ascend requires FP32 tiles with ≥8 rows (32-byte column align).
        # Cannot create [1, 1] FP32 tiles, so individual scale writes are blocked.
        # Possible fixes:
        #   1. Change quant_k_scales from [ROWS, 1] to [ROWS, 8] (wider columns)
        #   2. Use pl.tensor.write scalar API (if available)
        #   3. Compute scales inline during dequant instead of storing them

        # ── Attention: per Q-group, matching prefill pattern ──
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for gi in pl.range(TOTAL_Q_GROUPS):
            kvh = gi // Q_GROUPS
            qg = gi - kvh * Q_GROUPS
            q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH

            q_padded = pl.slice(
                all_q_padded, [Q_HEAD_PAD, HEAD_DIM],
                [b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
            )
            # Per-group GM buffers (matching prefill layout).
            all_raw_scores = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32,
            )
            all_exp_padded = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16,
            )
            all_oi_tmp = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32,
            )
            all_cur_mi = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32,
            )
            all_cur_li = pl.create_tensor(
                [MAX_CTX_BLOCKS * Q_HEAD_PAD, 1], dtype=pl.FP32,
            )

            max_blocks = pl.tensor.dim(block_table, 0) // user_batch

            # Stage 2.2: QK inline dequant + matmul.
            # BF16 buffer filled directly by dequant function.
            k_bf16_buf = pl.create_tensor([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16)

            for sb in pl.range(ctx_blocks):
                bt_idx = b * max_blocks + sb
                pbid = pl.cast(pl.tensor.read(block_table, [bt_idx]), pl.INDEX)
                row_base = layer_cmp_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                # 2.2a: Dequant K: gather → renorm → scale → unrotate → BF16 (scopes inside function).
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_dequant"):
                    k_scales_full = pl.slice(quant_k_scales, [BLOCK_SIZE, 1], [row_base, 0])
                    for k_sub in pl.range(BLOCK_SIZE // CMP_CHUNK):
                        sub_row = k_sub * CMP_CHUNK
                        k_indices = pl.slice(quant_k_cache, [CMP_CHUNK, HALF_DIM], [row_base + sub_row, 0])
                        k_scales_sub = pl.slice(k_scales_full, [CMP_CHUNK, 1], [sub_row, 0])
                        chunk_out = pl.create_tensor([CMP_CHUNK, HEAD_DIM], dtype=pl.BF16)
                        turboquant_kv_dequant_chunk(k_indices, k_scales_sub, tq_codebook, rot_slice, chunk_out)
                        k_bf16_buf = pl.assemble(k_bf16_buf, chunk_out, [sub_row, 0])

                # 2.2e: QK matmul — both inputs as GM-slice BF16.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                    k_tile = pl.slice(k_bf16_buf, [BLOCK_SIZE, HEAD_DIM], [0, 0])
                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                    all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

            # Stage 2.3: softmax.
            for sb in pl.range(ctx_blocks):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                    valid_len = pl.min(BLOCK_SIZE, ctx_len - sb * BLOCK_SIZE)
                    raw = pl.slice(
                        all_raw_scores, [Q_HEAD_PAD, BLOCK_SIZE],
                        [sb * Q_HEAD_PAD, 0],
                    )
                    scores_scaled = pl.mul(raw, decode_attn_scale)
                    scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_PAD // 2, valid_len)
                    scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                    cur_mi = pl.row_max(scores)
                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                    exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                    cur_li = pl.row_sum(exp_scores_fp32)
                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])


            # Stage 2.4: SV inline dequant + matmul.
            for sb in pl.range(ctx_blocks):
                bt_idx = b * max_blocks + sb
                pbid = pl.cast(pl.tensor.read(block_table, [bt_idx]), pl.INDEX)
                row_base = layer_cmp_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                v_tile_full = pl.create_tensor([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_dequant"):
                    for v_sub in pl.range(BLOCK_SIZE // CMP_CHUNK):
                        v_sub_row = row_base + v_sub * CMP_CHUNK
                        v_indices = pl.slice(quant_v_cache, [CMP_CHUNK, HALF_DIM], [v_sub_row, 0])
                        v_scales = pl.slice(quant_v_scales, [CMP_CHUNK, 1], [v_sub_row, 0])
                        chunk_out = pl.create_tensor([CMP_CHUNK, HEAD_DIM], dtype=pl.BF16)
                        turboquant_kv_dequant_chunk(v_indices, v_scales, tq_codebook, rot_slice, chunk_out)
                        v_tile_full = pl.assemble(v_tile_full, chunk_out, [v_sub * CMP_CHUNK, 0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                    exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, BLOCK_SIZE], [sb * Q_HEAD_PAD, 0])
                    oi_tmp = pl.matmul(exp_tile, v_tile_full, out_dtype=pl.FP32)
                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

            # Stage 2.5: online softmax accumulation.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax_init"):
                oi = pl.slice(all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [0, 0])
                mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [0, 0])
                li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [0, 0])
            for sb in pl.range(1, ctx_blocks):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                    oi_sb = pl.slice(all_oi_tmp, [Q_HEAD_PAD, HEAD_DIM], [sb * Q_HEAD_PAD, 0])
                    mi_sb = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                    li_sb = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                    mi_new = pl.maximum(mi, mi_sb)
                    alpha = pl.exp(pl.sub(mi, mi_new))
                    beta = pl.exp(pl.sub(mi_sb, mi_new))
                    li = pl.add(pl.mul(alpha, li), pl.mul(beta, li_sb))
                    oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_sb, beta))
                    mi = mi_new

            # Finalize: ctx = oi / li, write to attn_out.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_writeback"):
                ctx = pl.row_expand_div(oi, li)
                ctx_flat_bf16 = pl.cast(
                    pl.reshape(ctx, [1, Q_HEAD_PAD * HEAD_DIM]),
                    target_type=pl.BF16,
                )
                attn_out = pl.assemble(attn_out, ctx_flat_bf16, [b, q_base * HEAD_DIM])

    # ── Scope 3: output projection + residual + MLP (SPMD + UP_DOWN) ──
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.FP32)

        # out_proj: fused cube matmul + vec residual.
        for ob in pl.spmd(Q_OUT_BLOCKS, name_hint="out_proj",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            o0 = ob * Q_OUT_CHUNK
            a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, o0])
            o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, o0])
                o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)
            resid = pl.cast(pl.slice(current_hidden, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]), target_type=pl.FP32)
            resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

        # Post-attention RMSNorm.
        post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
            sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                post_sq_k0 = kb * K_CHUNK
                post_sq_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_sq_k0])
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(post_sq_chunk, post_sq_chunk)), [1, BATCH_TILE]))
            inv_rms_s3 = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
            for kb in pl.range(HIDDEN_BLOCKS):
                post_norm_k0 = kb * K_CHUNK
                post_norm_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, post_norm_k0])
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, post_norm_k0])
                post_normed = pl.col_expand_mul(
                    pl.row_expand_mul(post_norm_chunk, pl.reshape(inv_rms_s3, [BATCH_TILE, 1])), post_gamma,
                )
                post_norm_tile = pl.assemble(post_norm_tile, pl.cast(post_normed, target_type=pl.BF16), [0, post_norm_k0])

        # gate_up_silu: fused gate + up + SiLU.
        mlp_tile = pl.create_tensor([BATCH_TILE, INTERMEDIATE], dtype=pl.BF16)
        for ob in pl.spmd(MLP_OUT_BLOCKS, name_hint="gate_up_silu",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            mlp_o0 = ob * 256  # MLP_OUT_CHUNK = 256
            post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
            wg_0 = pl.slice(w_gate, [K_CHUNK, 256], [layer_hidden_base, mlp_o0])
            wu_0 = pl.slice(w_up, [K_CHUNK, 256], [layer_hidden_base, mlp_o0])
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                wg = pl.slice(w_gate, [K_CHUNK, 256], [layer_hidden_base + k0, mlp_o0])
                wu = pl.slice(w_up, [K_CHUNK, 256], [layer_hidden_base + k0, mlp_o0])
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, mlp_o0])

        # down_proj: fused cube matmul + vec residual.
        for dob in pl.spmd(HIDDEN_BLOCKS, name_hint="down_proj",
                          optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            d0 = dob * K_CHUNK
            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, 256], [0, 0])
            w_down_chunk_0 = pl.slice(w_down, [256, K_CHUNK], [layer_inter_base, d0])
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
            for ob in pl.range(1, MLP_OUT_BLOCKS):
                down_o0 = ob * 256
                down_mlp = pl.slice(mlp_tile, [BATCH_TILE, 256], [0, down_o0])
                w_down_chunk = pl.slice(w_down, [256, K_CHUNK], [layer_inter_base + down_o0, d0])
                down_acc = pl.matmul_acc(down_acc, down_mlp, w_down_chunk)
            resid_chunk_fp32 = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])
            out_chunk = pl.add(down_acc, resid_chunk_fp32)
            next_hidden = pl.assemble(next_hidden, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

    return next_hidden



@pl.jit
def decode_fwd_tq(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
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
    quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
    quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
    rot_matrices: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HEAD_DIM], pl.BF16],
    tq_codebook: pl.Tensor[[1, N_LEVELS], pl.FP32],
    # QJL (Algorithm 2): projection matrices + K sign/norm caches.
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    hidden_states.bind_dynamic(0, USER_BATCH_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    slot_mapping.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    rope_cos.bind_dynamic(0, ROPE_SEQ_DYN)
    rope_sin.bind_dynamic(0, ROPE_SEQ_DYN)
    quant_k_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_k_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(hidden_states, 0)
    batch_padded = BATCH
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    hidden_blocks = HIDDEN // K_CHUNK

    # Expand codebook [1, N_LEVELS] -> [CMP_CHUNK, N_LEVELS] once for all layers.
    tq_codebook_expanded = pl.create_tensor([CMP_CHUNK, N_LEVELS], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="codebook_expand"):
        tq_codebook_expanded = pl.assemble(
            tq_codebook_expanded,
            pl.col_expand_mul(
                pl.full([CMP_CHUNK, N_LEVELS], dtype=pl.FP32, value=1.0),
                tq_codebook,
            ),
            [0, 0],
        )

    # Copy hidden states into padded [batch, hidden] tensor.
    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(hidden_blocks):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    # Multi-layer loop calling decode_layer_tq().
    for layer_idx in pl.range(num_layers_actual):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        current_hidden = decode_layer_tq(
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
            quant_k_cache,
            quant_v_cache,
            quant_k_scales,
            quant_v_scales,
            rot_matrices,
            tq_codebook_expanded,
            wo,
            post_rms_weight,
            w_gate,
            w_up,
            w_down,
            next_hidden,
            layer_idx,
        )

    # Final RMSNorm + LM head projection (same as decode_fwd).
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
    num_layers: int = NUM_LAYERS,
    n_levels: int = N_LEVELS,
    vocab_size: int = VOCAB,
):

    import torch

    from golden import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    vocab = vocab_size
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE
    cmp_cache_rows = num_layers * layer_cache_rows
    synthetic_proj_scale = 0.5

    seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_input_rms_weight():
        return torch.rand(num_layers, hidden_size) - 0.5

    def init_wq():
        return torch.rand(num_layers * hidden_size, hidden_size) / hidden_size ** 0.5

    def init_wk():
        return torch.rand(num_layers * hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(num_layers * hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.ones(num_layers, head_dim)

    def init_k_norm_weight():
        return torch.ones(num_layers, head_dim)

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        num_blocks = batch * max_blocks_per_seq
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * max_blocks_per_seq + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_quant_k_cache():
        return torch.zeros(cmp_cache_rows, head_dim // 2, dtype=torch.uint8)

    def init_quant_v_cache():
        return torch.zeros(cmp_cache_rows, head_dim // 2, dtype=torch.uint8)

    def init_quant_k_scales():
        return torch.zeros(cmp_cache_rows, 1, dtype=torch.float32)

    def init_quant_v_scales():
        return torch.zeros(cmp_cache_rows, 1, dtype=torch.float32)

    def init_rot_matrices():
        # Generate one random orthogonal matrix per layer, stacked.
        torch.manual_seed(42)  # Deterministic rotation matrices.
        rot = []
        for _ in range(num_layers):
            Q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
            rot.append(Q)
        return torch.cat(rot, dim=0).to(torch.bfloat16)

    # TQ codebook (single row of Lloyd-Max centroids).
    from turboquant_kv import solve_lloyd_max as _solve_lm
    _lm_centroids, _ = _solve_lm(head_dim, min(int(_math.log2(n_levels)), 4))
    _codebook_row = _lm_centroids.float().unsqueeze(0)  # [1, n_levels]

    def init_tq_codebook():
        return _codebook_row.clone().contiguous()

    def init_wo():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(num_layers, hidden_size)

    def init_w_gate():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return synthetic_proj_scale * (torch.rand(num_layers * inter, hidden_size) - 0.5) / inter ** 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return (torch.rand(vocab, hidden_size) - 0.5) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [num_layers, hidden_size], torch.float32, init_value=init_input_rms_weight),
        TensorSpec("wq", [num_layers * hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [num_layers * hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [num_layers * hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_norm_weight", [num_layers, head_dim], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [num_layers, head_dim], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * max_blocks_per_seq], torch.int32, init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=init_rope_sin),
        # TQ quantized KV cache.
        TensorSpec("quant_k_cache", [cmp_cache_rows, head_dim // 2], torch.uint8, init_value=init_quant_k_cache),
        TensorSpec("quant_v_cache", [cmp_cache_rows, head_dim // 2], torch.uint8, init_value=init_quant_v_cache),
        TensorSpec("quant_k_scales", [cmp_cache_rows, 1], torch.float32, init_value=init_quant_k_scales),
        TensorSpec("quant_v_scales", [cmp_cache_rows, 1], torch.float32, init_value=init_quant_v_scales),
        TensorSpec("rot_matrices", [num_layers * head_dim, head_dim], torch.bfloat16, init_value=init_rot_matrices),
        # TQ gather-based dequant tensors.
        TensorSpec("tq_codebook", [1, n_levels], torch.float32, init_value=init_tq_codebook),
        # Standard weights.
        TensorSpec("wo", [num_layers * hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [num_layers, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [num_layers * hidden_size, inter], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [num_layers * hidden_size, inter], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [num_layers * inter, hidden_size], torch.bfloat16, init_value=init_w_down),
        # Final norm + LM head.
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32, init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16, init_value=init_lm_head_weight),
        # Outputs.
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


def golden_decode_fwd_tq(tensors):
    """PyTorch reference for the full-layer Qwen3-14B TQ decode program."""
    import math

    import torch

    from turboquant_kv import solve_lloyd_max

    hidden_states = tensors["hidden_states"].clone()
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
    quant_k_cache = tensors["quant_k_cache"].clone()
    quant_v_cache = tensors["quant_v_cache"].clone()
    quant_k_scales = tensors["quant_k_scales"].clone()
    quant_v_scales = tensors["quant_v_scales"].clone()
    rot_matrices = tensors["rot_matrices"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_layers = input_rms_weight.shape[0]
    kv_hidden_size = wk.shape[1]
    num_kv_heads = kv_hidden_size // head_dim
    num_heads = hidden_size // head_dim
    intermediate_size = w_gate.shape[1]
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    layer_cmp_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE

    # Solve Lloyd-Max codebook.
    lm_bits = min(int(math.log2(N_LEVELS)), 4)
    centroids, boundaries = solve_lloyd_max(head_dim, lm_bits)
    centroids = centroids.tolist()
    boundaries = boundaries.tolist()

    def _quantize_row(vec, rot_matrix):
        """Quantize a 1-D vector: L2 norm -> normalize -> rotate -> Lloyd-Max.

        L2 norm matches the kernel: l2 = sqrt(sum(x²) + eps), and normalize
        via recip + row_expand_mul equivalent.  Boundary search uses cumulative
        GE (rotated >= bval), matching pl.cmp cmp_type=5 (GE).
        """
        x_f = vec.float()
        l2 = torch.sqrt(x_f.pow(2).sum() + eps)  # != norm().clamp(min=eps)
        inv_norm = 1.0 / l2
        normed_bf16 = (x_f * inv_norm).to(torch.bfloat16)
        rotated = normed_bf16.float() @ rot_matrix.float()
        # Boundary search: cumulative GE count (== pl.cmp cmp_type=5).
        indices = torch.zeros(rotated.shape, dtype=torch.long)
        for bi, bval in enumerate(boundaries):
            indices += (rotated >= bval).long()
        # Pack two 4-bit indices per UINT8 (half/half layout), matching the kernel:
        # byte c = (idx[c+half] << 4) | idx[c].  Cast mirrors kernel INT32->FP16->UINT8.
        packed = (indices[half:] << 4) | indices[:half]              # [half] long
        return packed.to(torch.float16).to(torch.uint8), l2, rotated

    def _dequant_block(cache_rows_slice, scales_rows_slice, rot_mat):
        """Dequantize compressed cache rows: centroid lookup + renormalize + scale (rotated) + unrotate.

        Renorm uses rsqrt (== mul * rsqrt(sq+eps)), matching the kernel's
        qk_renorm / sv_dequant.  Scale is applied in the rotated domain before
        unrotate.  Index cast goes through UINT8→FP16→INT32.
        """
        # Unpack nibbles (half/half layout, matching the kernel): byte c -> idx[c]
        # (lo), idx[c+half] (hi).  Cast mirrors kernel UINT8->FP16->INT32.
        packed = cache_rows_slice.to(torch.float16).to(torch.int32)  # [rows, half]
        lo = packed & 0xF                                            # [rows, half]
        hi = packed >> 4                                             # [rows, half]
        indices = torch.cat([lo, hi], dim=-1)                        # [rows, head_dim]
        dec = torch.zeros(indices.shape, dtype=torch.float32)
        for ci in range(len(centroids)):
            dec += (indices == ci).float() * centroids[ci]
        # Renormalize to unit sphere (rsqrt == kernel's qk_renorm).
        dec_sq = dec.pow(2).sum(dim=-1, keepdim=True)
        dec = dec * torch.rsqrt(dec_sq + eps)
        # Rescale by stored L2 norms in the ROTATED domain (kernel order).
        dec = dec * scales_rows_slice.float()  # [rows, 1] BF16→FP32
        # Unrotate back to original space.
        return dec @ rot_mat.float().T

    hidden = hidden_states
    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cmp_base = layer_idx * layer_cmp_cache_rows
        rot_base = layer_idx * head_dim
        rot_matrix = rot_matrices[rot_base : rot_base + head_dim, :].float()  # [head_dim, head_dim]

        layer_wq = wq[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wk = wk[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wv = wv[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wo = wo[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_gate = w_gate[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_up = w_up[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_down = w_down[layer_inter_base : layer_inter_base + intermediate_size, :]

        # ── Scope 1: RMSNorm + Q/K/V projection ──
        q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
        k_proj = torch.zeros(batch, kv_hidden_size, dtype=torch.float32)
        v_proj = torch.zeros(batch, kv_hidden_size, dtype=torch.float32)

        for b0 in range(0, batch, BATCH_TILE):
            b_end = min(b0 + BATCH_TILE, batch)
            x_tile = hidden[b0:b_end, :].float()
            sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
            for k0 in range(0, hidden_size, SCOPE1_K_CHUNK):
                x_chunk = x_tile[:, k0 : k0 + SCOPE1_K_CHUNK]
                sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
            normed = (
                x_tile
                * torch.rsqrt(sq_sum / hidden_size + eps)
                * input_rms_weight[layer_idx : layer_idx + 1, :].float()
            ).bfloat16()
            q_proj[b0:b_end, :] = (normed.float() @ layer_wq.float()).float()
            k_proj[b0:b_end, :] = (normed.float() @ layer_wk.float()).float()
            v_proj[b0:b_end, :] = (normed.float() @ layer_wv.float()).float()

        # ── Q/K per-head norm ──
        for b_idx in range(batch):
            k_heads = k_proj[b_idx].view(num_kv_heads, head_dim)
            k_heads = (
                k_heads
                * torch.rsqrt(k_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * k_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            k_proj[b_idx] = k_heads.reshape(-1)

            q_heads = q_proj[b_idx].view(num_heads, head_dim)
            q_heads = (
                q_heads
                * torch.rsqrt(q_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * q_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            q_proj[b_idx] = q_heads.reshape(-1)

        # ── Scope 2: RoPE + KV quantize + Q RoPE + attention ──
        attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
        for b_idx in range(batch):
            ctx_len = int(seq_lens[b_idx].item())
            pos = ctx_len - 1
            ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            cos_row = rope_cos[pos : pos + 1, :]
            sin_row = rope_sin[pos : pos + 1, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            slot = int(slot_mapping[b_idx].item())
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot % BLOCK_SIZE

            # ── K: RoPE + L2 norm + normalize + rotate + quantize ──
            k_heads = k_proj[b_idx].view(num_kv_heads, head_dim)
            k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
            k_rope = torch.cat(
                [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
                dim=-1,
            )  # [num_kv_heads, head_dim]

            # Quantize K/V and collect L2 norms.
            k_l2_norms = []
            v_l2_norms = []
            for ki in range(num_kv_heads):
                k_vec = k_rope[ki]  # [head_dim]
                quant_cache_row = layer_cmp_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                k_idx_u8, k_l2, k_rot = _quantize_row(k_vec, rot_matrix)
                quant_k_cache[quant_cache_row, :] = k_idx_u8
                k_l2_norms.append(k_l2.item())

                # ── V: L2 norm + normalize + rotate + quantize (no RoPE) ──
                v_vec = v_proj[b_idx, ki * head_dim : (ki + 1) * head_dim]
                v_idx_u8, v_l2, _ = _quantize_row(v_vec, rot_matrix)
                quant_v_cache[quant_cache_row, :] = v_idx_u8
                v_l2_norms.append(v_l2.item())

            # Write K/V scales: [1] per (token, kvh) cache row.
            for ki in range(num_kv_heads):
                quant_cache_row = layer_cmp_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                quant_k_scales[quant_cache_row, 0] = k_l2_norms[ki]
                quant_v_scales[quant_cache_row, 0] = v_l2_norms[ki]

            # ── Q: RoPE ──
            q_heads = q_proj[b_idx].view(num_heads, head_dim)
            q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
            q_rope = torch.cat(
                [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
                dim=-1,
            )  # [num_heads, head_dim]

            # ── Attention: dequant from quant cache + QK/SV matmul ──
            attn_row_padded = torch.zeros(1, total_q_groups * Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp = q_rope[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                    oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                    li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                    mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                    # All blocks are compressed; compute from ctx_len.
                    n_cmp_blks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

                    for sb in range(n_cmp_blks):
                        # Dequantize K for this block.
                        bt_idx = b_idx * max_blocks_per_seq + sb
                        pbid = int(block_table[bt_idx].item())
                        row_base = layer_cmp_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE

                        k_cache_block = quant_k_cache[row_base : row_base + BLOCK_SIZE, :]
                        k_scales_block = quant_k_scales[row_base : row_base + BLOCK_SIZE, 0:1]
                        k_dec = _dequant_block(k_cache_block, k_scales_block, rot_matrix)

                        # Dequantize V for this block.
                        v_cache_block = quant_v_cache[row_base : row_base + BLOCK_SIZE, :]
                        v_scales_block = quant_v_scales[row_base : row_base + BLOCK_SIZE, 0:1]
                        v_dec = _dequant_block(v_cache_block, v_scales_block, rot_matrix)

                        # QK matmul.
                        raw_scores = q_grp.float() @ k_dec.T  # [Q_HEAD_BATCH, BLOCK_SIZE]
                        s0 = sb * BLOCK_SIZE
                        valid_len = min(BLOCK_SIZE, ctx_len - s0)
                        if valid_len < BLOCK_SIZE:
                            raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale

                        # Softmax.
                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                        # SV matmul.
                        oi_tmp = exp_scores_bf16.float() @ v_dec

                        # Online softmax accumulation.
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

                    ctx = oi / li  # [Q_HEAD_BATCH, head_dim]

                    # No unrotate needed: K/V are already in original space after dequant.
                    ctx_bf16 = ctx.to(torch.bfloat16)
                    ctx_flat_padded = torch.zeros(1, Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
                    ctx_flat_padded[:, : Q_HEAD_BATCH * head_dim] = ctx_bf16.reshape(1, -1)
                    attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : (gi + 1) * Q_HEAD_PAD * head_dim,
                    ] = ctx_flat_padded

            # Write back attention output (strip Q_HEAD_PAD).
            attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    attn_row[
                        :,
                        q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                    ] = attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : gi * Q_HEAD_PAD * head_dim + Q_HEAD_BATCH * head_dim,
                    ]
            attn_out[b_idx : b_idx + 1, :] = attn_row

        # ── Scope 3: output projection + residual + post RMSNorm + MLP + final residual ──
        o_proj = attn_out.float() @ layer_wo.float()
        resid1 = o_proj + hidden.float()
        normed_bf16 = (
            resid1
            * torch.rsqrt(resid1.pow(2).mean(dim=-1, keepdim=True) + eps)
            * post_rms_weight[layer_idx : layer_idx + 1, :].float()
        ).bfloat16()
        gate = normed_bf16.float() @ layer_w_gate.float()
        up = normed_bf16.float() @ layer_w_up.float()
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = mlp_bf16.float() @ layer_w_down.float()
        hidden = (down + resid1).bfloat16()

    # Final norm + LM head (matching the NPU kernel's rms_lm_head).
    final_norm_w = tensors["final_norm_weight"].float()          # [1, HIDDEN]
    lm_head_w = tensors["lm_head_weight"].float()               # [VOCAB, HIDDEN]
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True) + eps
    normed = hidden_f * torch.rsqrt(variance) * final_norm_w
    logits = normed @ lm_head_w.T                                # [batch, VOCAB]
    tensors["out"][:] = logits


def make_pass_rate_compare(threshold: float):
    """Build a compare_fn that passes when >= `threshold` of elements are
    close (under the run's atol/rtol). Used for the BF16 long-tail on
    multi-layer decode: tolerates a small fraction of 1-2 ULP outliers
    while still catching systematic bias (which would tank the pass rate).
    """
    def cmp(actual, expected, *, rtol, atol, **_):
        import torch

        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        rate = close.float().mean().item()
        n_fail = int((~close).sum().item())
        ok = rate >= threshold
        msg = (
            f"    pass_rate={rate:.6f} (threshold {threshold:.6f}), "
            f"{n_fail}/{actual.numel()} mismatched  rtol={rtol} atol={atol}"
        )
        if not ok:
            flat_a = actual.flatten()
            flat_e = expected.flatten()
            idx = torch.where(~close.flatten())[0][:5]
            lines = [
                f"    [{i.item()}] actual={flat_a[i].item()}, expected={flat_e[i].item()}"
                for i in idx
            ]
            msg += "\n    first {} mismatches:\n".format(idx.numel()) + "\n".join(lines)
        return ok, msg

    cmp.__name__ = f"pass_rate>={threshold:.4f}"
    return cmp


def golden_decode_fwd_fp(tensors):
    """FP golden wrapper that accepts TQ-format tensors.

    Builds FP-compatible tensor dict from TQ inputs (skipping TQ-specific
    tensors like quant caches, rotation matrices, codebook), adds properly
    sized BF16 k_cache/v_cache for the FP golden, and writes the
    output back into tensors["out"].
    """
    import torch
    from decode_fwd import golden_decode_fwd as fp_golden

    num_layers = tensors["input_rms_weight"].shape[0]
    batch = tensors["hidden_states"].shape[0]
    head_dim = tensors["rope_cos"].shape[1]

    fp_skip = {
        "quant_k_cache", "quant_v_cache", "quant_k_scales",
        "quant_v_scales",
        "rot_matrices", "tq_codebook",
        "qjl_matrices", "quant_k_signs_cache", "qjl_norms_cache",
        # TQ test sizes block_table/slot_mapping for local max_seq (e.g. 128),
        # but FP golden expects MAX_BLOCKS_PER_SEQ (config.py, based on 4096).
        "block_table", "slot_mapping",
    }

    fp_tensors = {}
    for name, value in tensors.items():
        if name in fp_skip:
            continue
        if name == "out":
            fp_tensors[name] = torch.zeros_like(value)
        else:
            fp_tensors[name] = value.clone()

    # FP golden expects caches sized with MAX_BLOCKS_PER_SEQ (config constant).
    fp_cache_rows = num_layers * batch * MAX_BLOCKS_PER_SEQ * NUM_KV_HEADS * BLOCK_SIZE
    fp_tensors["k_cache"] = torch.zeros(fp_cache_rows, head_dim, dtype=torch.bfloat16)
    fp_tensors["v_cache"] = torch.zeros(fp_cache_rows, head_dim, dtype=torch.bfloat16)

    # Rebuild block_table and slot_mapping for FP golden's cache layout.
    fp_tensors["block_table"] = torch.arange(
        batch * MAX_BLOCKS_PER_SEQ, dtype=torch.int32,
    )
    seq_lens = tensors["seq_lens"]
    fp_slots = torch.empty(batch, dtype=torch.int32)
    for b in range(batch):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // BLOCK_SIZE
        page_offset = pos % BLOCK_SIZE
        phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
        fp_slots[b] = phys_block * BLOCK_SIZE + page_offset
    fp_tensors["slot_mapping"] = fp_slots

    fp_golden(fp_tensors)
    tensors["out"][:] = fp_tensors["out"]


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--pass-rate", type=float, default=0.98,
                        help="Fraction of `out` elements that must satisfy atol/rtol. "
                             "Default 0.98 tolerates the BF16 ULP long-tail.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for input tensor generation. Fixed by default "
                             "so pass_rate measurements are reproducible across runs.")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)

    if args.max_seq > MAX_SEQ:
        raise ValueError(
            f"decode_fwd_tq currently supports max_seq <= {MAX_SEQ}"
        )

    result = run_jit(
        fn=decode_fwd_tq,
        specs=build_tensor_specs(
            batch=args.batch,
            max_seq=args.max_seq,
            num_layers=args.num_layers,
        ),
        golden_fn=golden_decode_fwd_tq,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
        compare_fn={"out": make_pass_rate_compare(args.pass_rate)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = ["decode_fwd_tq", "build_tensor_specs", "golden_decode_fwd_tq",
           "golden_decode_fwd_fp", "make_pass_rate_compare"]



