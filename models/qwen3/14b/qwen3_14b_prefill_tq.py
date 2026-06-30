# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B full-layer prefill forward with TurboQuant inline KV compression.

Structurally identical to prefill_fwd.py except:
- KV cache is stored in compressed UINT8 format (quant_k_cache/quant_v_cache)
  and prefill attention reads DEQUANTIZED K/V from it (gather codebook ->
  renormalize -> unrotate -> rescale), exactly like decode. No FP k_cache/v_cache.
- K/V RoPE + inline quantization is batched across all tokens in the token
  block, so row_max operates on [TOK_TILE, half_dim] -> [TOK_TILE, 1] (aligned).
- Quantized scales are written to quant_k_scales / quant_v_scales per cache row.
- Attention QK/SV matmul dequantizes K/V on the fly from quant_k_cache /
  quant_v_cache (no FP path).
"""

import pypto.language as pl

from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE,
    KV_HIDDEN,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_GROUPS,
    Q_PER_KV,
    TOTAL_Q_GROUPS,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)
from rms_lm_head import rms_lm_head
from turboquant_kv import (
    turboquant_kv_quantize,
    turboquant_kv_dequant_chunk,
    _lm_centroids,
)

# ---------------------------------------------------------------------------
# Dynamic dims (prefill-specific)
# ---------------------------------------------------------------------------
PREFILL_TOKENS_DYN = pl.dynamic("PREFILL_TOKENS_DYN")
QUANT_CACHE_ROWS_DYN = pl.dynamic("QUANT_CACHE_ROWS_DYN")
LAYER_ROT_ROWS_DYN = pl.dynamic("LAYER_ROT_ROWS_DYN")

# ---------------------------------------------------------------------------
# TQ constants
# ---------------------------------------------------------------------------
N_LEVELS = 16  # int4 -> 16 levels


# ---------------------------------------------------------------------------
# Prefill-specific tiling constants (local, may differ from config.py)
# ---------------------------------------------------------------------------
MAX_SEQ = 128
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 16
Q_HEAD_BATCH = 5  # Q heads per attention group
Q_HEAD_BATCH_PAD = 16  # padded to 32-byte alignment (8 * 4 = 32)
Q_HEAD_PAD = 16  # padded Q rows for cube alignment
SEQ_TILE = 128  # sequence tile for attention
CMP_TILE = 64  # Smaller tile for fused dequant to fit A2A3 memory limits
CMP_TILE_SV = 64  # Even smaller tile for SV fused dequant (Vec buffer constraint)
CMP_CHUNK = 32  # Gather sub-tile for dequant (32 rows * 1B = 32-byte aligned), matches decode
SB_BATCH = 128
BLOCK_SIZE = SEQ_TILE
MLP_OUT_CHUNK = 128

HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE


# ---------------------------------------------------------------------------
# Single-layer prefill with TurboQuant
# ---------------------------------------------------------------------------


@pl.jit.inline
def prefill_layer_tq(
        hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
        seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
        wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
        rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
        # TQ compressed KV cache (written for decode).
        quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
        quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
        quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
        quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
        # Per-layer rotation matrix slice [HEAD_DIM, HEAD_DIM].
        rot_matrix: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
        # TQ codebook (expanded [CMP_CHUNK, N_LEVELS] by wrapper) for dequant attention.
        tq_codebook: pl.Tensor[[CMP_CHUNK, N_LEVELS], pl.FP32],
        # Standard layer weights.
        wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
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
    layer_cache_rows = pl.tensor.dim(quant_k_cache, 0) // num_layers_actual
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * layer_cache_rows
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch

    for b in pl.parallel(0, user_batch, 1):
        token_base = pl.cast(0, pl.INDEX)
        for prev_b in pl.range(b):
            token_base = token_base + pl.cast(
                pl.tensor.read(seq_lens, [prev_b]), pl.INDEX,
            )
        seq_len_b = pl.tensor.read(seq_lens, [b])
        tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
        for p0_idx in pl.range(tok_blocks):
            p0 = p0_idx * TOK_TILE
            token_p0 = token_base + p0
            valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

            # ── Scope 1: input RMSNorm + Q/K/V projection ──
            normed_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)

            # Stage 1.1: RMSNorm (vector ops).
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                partial_sq = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [token_p0, k0],
                                 valid_shape=[valid_tok, K_CHUNK]),
                        target_type=pl.FP32,
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

                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [token_p0, k0],
                                 valid_shape=[valid_tok, K_CHUNK]),
                        target_type=pl.FP32,
                    )
                    gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                    normed_tile = pl.assemble(
                        normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0],
                    )

            # Stage 1.2: Q projection (matmul + matmul_acc, FP32 output).
            q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
            for ob_chunk in pl.parallel(0, Q_OUT_BLOCKS, 4):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                    for ob in pl.range(ob_chunk, ob_chunk + 4):
                        q0 = ob * Q_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q0])
                        q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK],
                                                [layer_hidden_base + k0, q0])
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                        q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

            # Stage 1.3: K/V projection (matmul + matmul_acc in single incore).
            k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
            v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
            for ob_chunk in pl.parallel(0, KV_OUT_BLOCKS, 4):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
                    for ob in pl.range(ob_chunk, ob_chunk + 4):
                        kv0 = ob * KV_OUT_CHUNK

                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK],
                                           [layer_hidden_base, kv0])
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK],
                                                 [layer_hidden_base + k0, kv0])
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK],
                                           [layer_hidden_base, kv0])
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK],
                                                 [layer_hidden_base + k0, kv0])
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

            # Stage 1.4: Q/K per-head RMSNorm (FP32 in-place on proj tiles).
            for qh_chunk in pl.parallel(0, NUM_HEADS, NUM_HEADS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_norm"):
                    for qh in pl.range(qh_chunk, qh_chunk + NUM_HEADS):
                        q_col = qh * HEAD_DIM
                        q_head = pl.slice(q_proj_tile, [TOK_TILE, HEAD_DIM], [0, q_col])
                        q_sq = pl.reshape(
                            pl.row_sum(pl.mul(q_head, q_head)),
                            [TOK_TILE, 1],
                        )
                        q_inv_rms = pl.recip(
                            pl.sqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS)),
                        )
                        q_normed = pl.col_expand_mul(
                            pl.row_expand_mul(q_head, q_inv_rms),
                            pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
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
                        k_inv_rms = pl.recip(
                            pl.sqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS)),
                        )
                        k_normed = pl.col_expand_mul(
                            pl.row_expand_mul(k_head, k_inv_rms),
                            pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                        )
                        k_proj_tile = pl.assemble(k_proj_tile, k_normed, [0, k_col])

            # ── Scope 2a: Batched K/V RoPE + inline quantization ──
            quant_k_temp = pl.create_tensor([TOK_TILE, KV_HIDDEN // 2], dtype=pl.UINT8)
            quant_v_temp = pl.create_tensor([TOK_TILE, KV_HIDDEN // 2], dtype=pl.UINT8)
            k_scales_buf = pl.create_tensor(
                [TOK_TILE, NUM_KV_HEADS], dtype=pl.FP32,
            )
            v_scales_buf = pl.create_tensor(
                [TOK_TILE, NUM_KV_HEADS], dtype=pl.FP32,
            )

            # Load batched cos/sin for all positions in this token block.
            cos_all = pl.slice(rope_cos, [TOK_TILE, HEAD_DIM], [p0, 0],
                               valid_shape=[valid_tok, HEAD_DIM])
            sin_all = pl.slice(rope_sin, [TOK_TILE, HEAD_DIM], [p0, 0],
                               valid_shape=[valid_tok, HEAD_DIM])
            cos_lo_all = pl.slice(cos_all, [TOK_TILE, HALF_DIM], [0, 0])
            cos_hi_all = pl.slice(cos_all, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
            sin_lo_all = pl.slice(sin_all, [TOK_TILE, HALF_DIM], [0, 0])
            sin_hi_all = pl.slice(sin_all, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
            turboquant_kv_quantize(
                k_proj_tile, v_proj_tile, rot_matrix,
                cos_lo_all, cos_hi_all, sin_lo_all, sin_hi_all,
                quant_k_temp, quant_v_temp,
                k_scales_buf, v_scales_buf,
            )
            # ── Scope 2b: Per-token cache write + Q RoPE + causal attention ──
            attn_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
            for ti in pl.range(valid_tok):
                pos = p0 + ti
                ctx_len = pos + 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Scatter quantized K/V from temp tiles to cache.
                cache_slot = pl.cast(
                    pl.tensor.read(slot_mapping, [token_base + pos]), pl.INDEX,
                )
                cache_slot_block = cache_slot // BLOCK_SIZE
                cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE

                # RoPE cos/sin for this position (used by Q RoPE).
                cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
                cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
                sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
                sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_kv_cache"):
                    for ki in pl.range(NUM_KV_HEADS):
                        fp_cache_row = (
                                (cache_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE
                                + cache_slot_offset
                        )
                        kv_col = ki * HALF_DIM

                        # Quantized K/V: scatter to quant cache (absolute offset).
                        # Attention reads dequantized K/V from this cache (no FP path).
                        # Index temps are nibble-packed, so each head is HALF_DIM wide.
                        cache_row = layer_cache_base + fp_cache_row
                        quant_k_row = pl.slice(quant_k_temp, [1, HALF_DIM], [ti, kv_col])
                        quant_k_cache = pl.assemble(
                            quant_k_cache, quant_k_row, [cache_row, 0],
                        )
                        k_scale = pl.read(k_scales_buf, [ti, ki])
                        quant_k_scales = pl.write(quant_k_scales, [cache_row, 0], k_scale)
                        quant_v_row = pl.slice(quant_v_temp, [1, HALF_DIM], [ti, kv_col])
                        quant_v_cache = pl.assemble(
                            quant_v_cache, quant_v_row, [cache_row, 0],
                        )
                        v_scale = pl.read(v_scales_buf, [ti, ki])
                        quant_v_scales = pl.write(quant_v_scales, [cache_row, 0], v_scale)
                # Q RoPE + pad (per-token, position-dependent).
                all_q_padded = pl.create_tensor(
                    [TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
                )
                for gi_chunk in pl.parallel(0, TOTAL_Q_GROUPS, TOTAL_Q_GROUPS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_pad"):
                        for gi in pl.range(gi_chunk, gi_chunk + TOTAL_Q_GROUPS):
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                pl.cast(
                                    pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM],
                                            dtype=pl.FP32, value=0.0),
                                    target_type=pl.BF16,
                                ),
                                [gi * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                            )
                for ki_chunk in pl.parallel(0, NUM_KV_HEADS, 8):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope"):
                        for ki in pl.range(ki_chunk, ki_chunk + 8):
                            q_base = ki * Q_PER_KV
                            for qi in pl.range(Q_HEAD_BATCH):
                                q_col = (q_base + qi) * HEAD_DIM
                                q_lo = pl.reshape(
                                    pl.slice(q_proj_tile, [1, HALF_DIM],
                                             [ti, q_col]),
                                    [1, HALF_DIM],
                                )
                                q_hi = pl.reshape(
                                    pl.slice(q_proj_tile, [1, HALF_DIM],
                                             [ti, q_col + HALF_DIM]),
                                    [1, HALF_DIM],
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
                                    all_q_padded, rot_lo_bf16,
                                    [ki * Q_HEAD_PAD + qi, 0],
                                )
                                all_q_padded = pl.assemble(
                                    all_q_padded, rot_hi_bf16,
                                    [ki * Q_HEAD_PAD + qi, HALF_DIM],
                                )
                attn_row = pl.create_tensor([1, HIDDEN], dtype=pl.BF16)
                for gi in pl.range(TOTAL_Q_GROUPS):
                    kvh = gi // Q_GROUPS
                    qg = gi - kvh * Q_GROUPS
                    q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH

                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [gi * Q_HEAD_PAD, 0])

                    all_raw_scores = pl.create_tensor([MAX_CTX_BLOCKS * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([MAX_CTX_BLOCKS * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([MAX_CTX_BLOCKS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([MAX_CTX_BLOCKS * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([MAX_CTX_BLOCKS * Q_HEAD_BATCH_PAD, 1], dtype=pl.FP32)

                    # K dequant scratch (BF16, filled directly by dequant function).
                    k_bf16_buf = pl.create_tensor([SEQ_TILE, HEAD_DIM], dtype=pl.BF16)
                    # QK: inline dequant K from quant cache + matmul (mirrors decode).
                    for sb in pl.range(ctx_blocks):
                        block_table_idx = b * max_blocks_per_seq + sb
                        pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                        cmp_row_base = layer_cache_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE

                        # Dequant K: gather → renorm → scale → unrotate → BF16 (scopes inside function).
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_dequant"):
                            k_scales_full = pl.slice(quant_k_scales, [SEQ_TILE, 1], [cmp_row_base, 0])
                            for k_sub in pl.range(SEQ_TILE // CMP_CHUNK):
                                sub_row = k_sub * CMP_CHUNK
                                k_indices = pl.slice(quant_k_cache, [CMP_CHUNK, HALF_DIM], [cmp_row_base + sub_row, 0])
                                k_scales_sub = pl.slice(k_scales_full, [CMP_CHUNK, 1], [sub_row, 0])
                                chunk_out = pl.create_tensor([CMP_CHUNK, HEAD_DIM], dtype=pl.BF16)
                                turboquant_kv_dequant_chunk(k_indices, k_scales_sub, tq_codebook, rot_matrix, chunk_out)
                                k_bf16_buf = pl.assemble(k_bf16_buf, chunk_out, [sub_row, 0])

                        # QK matmul — both inputs as GM-slice BF16 (mirrors FP prefill).
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_matmul"):
                            k_tile = pl.slice(k_bf16_buf, [SEQ_TILE, HEAD_DIM], [0, 0])
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    # Stage 2.3: softmax for all active sb blocks.
                    for sb_chunk in pl.parallel(0, ctx_blocks, SB_BATCH):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
                            for si in pl.range(SB_BATCH):
                                sb = sb_chunk + si
                                if sb < ctx_blocks:
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    scores_valid = pl.slice(
                                        all_raw_scores, [Q_HEAD_BATCH_PAD, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                        valid_shape=[Q_HEAD_BATCH, valid_len],
                                    )
                                    scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                    scores = pl.mul(scores_padded, ATTN_SCALE)
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                    cur_li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH_PAD, 0])
                                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH_PAD, 0])

                    # SV: inline dequant V from quant cache + matmul (mirrors decode).
                    for sb in pl.range(ctx_blocks):
                        block_table_idx = b * max_blocks_per_seq + sb
                        pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                        cmp_row_base = layer_cache_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                        v_tile_full = pl.create_tensor([SEQ_TILE, HEAD_DIM], dtype=pl.BF16)

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_dequant"):
                            for v_sub in pl.range(SEQ_TILE // CMP_CHUNK):
                                v_sub_row = cmp_row_base + v_sub * CMP_CHUNK
                                v_indices = pl.slice(quant_v_cache, [CMP_CHUNK, HALF_DIM], [v_sub_row, 0])
                                v_scales_sub = pl.slice(quant_v_scales, [CMP_CHUNK, 1], [v_sub_row, 0])
                                chunk_out = pl.create_tensor([CMP_CHUNK, HEAD_DIM], dtype=pl.BF16)
                                turboquant_kv_dequant_chunk(v_indices, v_scales_sub, tq_codebook, rot_matrix, chunk_out)
                                v_tile_full = pl.assemble(v_tile_full, chunk_out, [v_sub * CMP_CHUNK, 0])

                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sv_matmul"):
                            exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sb * Q_HEAD_PAD, 0])
                            oi_tmp = pl.matmul(exp_tile, v_tile_full, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    # Stage 2.5: online softmax accumulation.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax_init"):
                        oi = pl.full([Q_HEAD_BATCH_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        li_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                        li = pl.reshape(li_flat, [Q_HEAD_BATCH_PAD, 1])
                        mi_flat = pl.full([1, Q_HEAD_BATCH_PAD], dtype=pl.FP32, value=0.0)
                        mi = pl.reshape(mi_flat, [Q_HEAD_BATCH_PAD, 1])

                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="online_softmax"):
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    oi_sb = pl.slice(all_oi_tmp, [Q_HEAD_BATCH_PAD, HEAD_DIM], [sb * Q_HEAD_PAD, 0])
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

                    # Finalize online softmax: ctx = oi / li.
                    # K/V were dequantized+unrotated back to original space, so ctx is too.
                    ctx_tmp = pl.create_tensor([Q_HEAD_BATCH_PAD, HEAD_DIM], dtype=pl.FP32)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_context"):
                        ctx = pl.row_expand_div(oi, li)
                        ctx_tmp = pl.assemble(ctx_tmp, ctx, [0, 0])
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attention_writeback"):
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * HEAD_DIM
                            row = pl.slice(ctx_tmp, [1, HEAD_DIM], [qi, 0])
                            row_bf16 = pl.cast(row, target_type=pl.BF16)
                            attn_row = pl.assemble(attn_row, row_bf16, [0, q_col])

                # Source must be a slice (raw created-tensor assemble has no codegen).
                # Keep the attention row in the local tile.
                attn_tile = pl.assemble(attn_tile, attn_row, [ti, 0])

            # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
            resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
            for ob in pl.range(Q_OUT_BLOCKS):
                o0 = ob * Q_OUT_CHUNK
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
                    tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                    tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK],
                                      [layer_hidden_base, o0])
                    o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK],
                                            [0, k0])
                        tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK],
                                            [layer_hidden_base + k0, o0])
                        o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
                    resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj_residual"):
                    resid_chunk = pl.cast(
                        pl.slice(hidden_states, [TOK_TILE, Q_OUT_CHUNK],
                                 [token_p0, o0],
                                 valid_shape=[valid_tok, Q_OUT_CHUNK]),
                        target_type=pl.FP32,
                    )
                    mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_CHUNK],
                                      [0, o0])
                    resid1_tile = pl.assemble(
                        resid1_tile, pl.add(mm_out, resid_chunk), [0, o0],
                    )

            # Post-attention RMSNorm.
            post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                   [1, TOK_TILE]),
                    )
                post_inv_rms = pl.recip(pl.sqrt(pl.reshape(
                    pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    [TOK_TILE, 1],
                )))

                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                    gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                    normed = pl.col_expand_mul(
                        pl.row_expand_mul(x_chunk, post_inv_rms),
                        gamma,
                    )
                    post_norm_tile = pl.assemble(
                        post_norm_tile, pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            # MLP gate/up + SiLU.
            mlp_silu_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.BF16)
            for ob in pl.range(MLP_OUT_BLOCKS):
                o0 = ob * MLP_OUT_CHUNK

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                    pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                    wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                                   [layer_hidden_base, o0])
                    gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK],
                                       [0, k0])
                        wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                                       [layer_hidden_base + k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                    pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                    wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK],
                                   [layer_hidden_base, o0])
                    up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK],
                                       [0, k0])
                        wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK],
                                       [layer_hidden_base + k0, o0])
                        up_acc = pl.matmul_acc(up_acc, pci, wui)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                    mlp_silu_tile = pl.assemble(
                        mlp_silu_tile, mlp_chunk_bf16, [0, o0],
                    )

            # Down projection + final residual.
            for dob in pl.range(HIDDEN_BLOCKS):
                d0 = dob * K_CHUNK
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj"):
                    mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK],
                                           [0, 0])
                    w_down_chunk_0 = pl.slice(
                        w_down, [MLP_OUT_CHUNK, K_CHUNK],
                        [layer_inter_base, d0],
                    )
                    down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0,
                                         out_dtype=pl.FP32)
                    for ob in pl.range(1, MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        mlp_chunk_i = pl.slice(
                            mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0],
                        )
                        w_down_chunk_i = pl.slice(
                            w_down, [MLP_OUT_CHUNK, K_CHUNK],
                            [layer_inter_base + o0, d0],
                        )
                        down_acc = pl.matmul_acc(down_acc, mlp_chunk_i,
                                                 w_down_chunk_i)

                with pl.at(level=pl.Level.CORE_GROUP,
                           name_hint="down_proj_residual"):
                    out_chunk = pl.add(
                        down_acc,
                        pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                    )
                    out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                    out = pl.assemble(out, out_chunk_bf16, [token_p0, d0])

    return out


# ---------------------------------------------------------------------------
# Top-level: multi-layer loop + gather last token + LM head
# ---------------------------------------------------------------------------


@pl.jit
def prefill_fwd_tq(
        hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
        seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
        input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
        wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
        wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
        q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
        rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
        rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
        block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
        slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
        quant_k_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
        quant_v_cache: pl.Tensor[[QUANT_CACHE_ROWS_DYN, HALF_DIM], pl.UINT8],
        quant_k_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
        quant_v_scales: pl.Tensor[[QUANT_CACHE_ROWS_DYN, 1], pl.FP32],
        rot_matrices: pl.Tensor[[LAYER_ROT_ROWS_DYN, HEAD_DIM], pl.BF16],
        tq_codebook: pl.Tensor[[1, N_LEVELS], pl.FP32],
        wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
        post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
        w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
        w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
        w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
        final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
        lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
        out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    quant_k_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_cache.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_k_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)
    quant_v_scales.bind_dynamic(0, QUANT_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    prefill_tokens = pl.tensor.dim(hidden_states, 0)
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
    for layer_idx in pl.range(num_layers_actual):
        rot_base = layer_idx * HEAD_DIM
        rot_slice = pl.slice(rot_matrices, [HEAD_DIM, HEAD_DIM], [rot_base, 0])
        next_hidden = pl.create_tensor([prefill_tokens, HIDDEN], dtype=pl.BF16)
        hidden_states = prefill_layer_tq(
            hidden_states,
            seq_lens,
            input_rms_weight,
            wq,
            wk,
            wv,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            block_table,
            slot_mapping,
            quant_k_cache,
            quant_v_cache,
            quant_k_scales,
            quant_v_scales,
            rot_slice,
            tq_codebook_expanded,
            wo,
            post_rms_weight,
            w_gate,
            w_up,
            w_down,
            next_hidden,
            layer_idx,
        )

    final_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_prefill_last_token"):
            for bi in pl.range(BATCH_TILE):
                b = b0 + bi
                if b < user_batch:
                    token_base = pl.cast(0, pl.INDEX)
                    for prev_b in pl.range(b):
                        token_base = token_base + pl.cast(
                            pl.tensor.read(seq_lens, [prev_b]), pl.INDEX,
                        )
                    seq_len_b = pl.tensor.read(seq_lens, [b])
                    if seq_len_b > 0:
                        last_token = token_base + pl.cast(seq_len_b, pl.INDEX) - 1
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            final_hidden_chunk = pl.slice(
                                hidden_states,
                                [1, K_CHUNK],
                                [last_token, k0],
                            )
                            final_hidden = pl.assemble(
                                final_hidden, final_hidden_chunk, [b, k0],
                            )

    out = rms_lm_head(
        final_hidden, final_norm_weight, lm_head_weight, seq_lens, out,
    )
    return out


# ---------------------------------------------------------------------------
# Tensor specs for the golden test harness
# ---------------------------------------------------------------------------


def build_tensor_specs(
        batch: int = BATCH,
        max_seq: int = MAX_SEQ,
        hidden_size: int = HIDDEN,
        num_heads: int = NUM_HEADS,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        intermediate_size: int = INTERMEDIATE,
        num_layers: int = NUM_LAYERS,
        vocab_size: int = VOCAB,
        use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    assert hidden_size == num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    vocab = vocab_size
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * max_blocks_per_seq
    layer_cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    cache_rows = num_layers * layer_cache_rows

    if use_max_seq:
        seq_lens_values = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        n_blocks = max(1, max_seq // TOK_TILE)
        seq_lens_values = (
                                  (torch.arange(batch, dtype=torch.int32) % n_blocks) + 1
                          ) * TOK_TILE
    total_tokens = int(seq_lens_values.sum().item())

    def init_tq_codebook():
        return _lm_centroids.float().unsqueeze(0).contiguous()  # [1, N_LEVELS]

    def init_hidden_states():
        return torch.rand(total_tokens, hidden_size) - 0.5

    def init_seq_lens():
        return seq_lens_values.clone()

    def init_rms_weight():
        return torch.rand(num_layers, hidden_size) - 0.5

    def init_wq():
        return (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_wk():
        return (torch.rand(num_layers * hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_wv():
        return (torch.rand(num_layers * hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.rand(num_layers, head_dim) - 0.5

    def init_k_norm_weight():
        return torch.rand(num_layers, head_dim) - 0.5

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(total_tokens, dtype=torch.int32)
        token_idx = 0
        for b in range(batch):
            seq_len = int(seq_lens_values[b].item())
            for pos in range(seq_len):
                logical_block = pos // BLOCK_SIZE
                page_offset = pos % BLOCK_SIZE
                phys_block = b * max_blocks_per_seq + logical_block
                slots[token_idx] = phys_block * BLOCK_SIZE + page_offset
                token_idx += 1
        return slots

    def init_quant_k_cache():
        return torch.full((cache_rows, head_dim // 2), 0, dtype=torch.uint8)

    def init_quant_v_cache():
        return torch.full((cache_rows, head_dim // 2), 0, dtype=torch.uint8)

    def init_quant_k_scales():
        return torch.full((cache_rows, 1), 0.0, dtype=torch.float32)

    def init_quant_v_scales():
        return torch.full((cache_rows, 1), 0.0, dtype=torch.float32)

    def init_rot_matrices():
        torch.manual_seed(0)
        rot = []
        for _ in range(num_layers):
            Q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
            rot.append(Q)
        return torch.cat(rot, dim=0).to(torch.bfloat16)


    def init_quant_k_signs_cache():
        return torch.zeros(cache_rows, head_dim, dtype=torch.uint8)


    def init_wo():
        return (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(num_layers, hidden_size)

    def init_w_gate():
        return (torch.rand(num_layers * hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(num_layers * hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(num_layers * intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return (torch.rand(vocab, hidden_size) - 0.5) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [total_tokens, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("input_rms_weight", [num_layers, hidden_size], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [num_layers * hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [num_layers * hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [num_layers * hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("q_norm_weight", [num_layers, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [num_layers, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("block_table", [batch * max_blocks_per_seq], torch.int32,
                   init_value=init_block_table),
        TensorSpec("slot_mapping", [total_tokens], torch.int32,
                   init_value=init_slot_mapping),
        TensorSpec("quant_k_cache", [cache_rows, head_dim // 2], torch.uint8,
                   init_value=init_quant_k_cache),
        TensorSpec("quant_v_cache", [cache_rows, head_dim // 2], torch.uint8,
                   init_value=init_quant_v_cache),
        TensorSpec("quant_k_scales", [cache_rows, 1], torch.float32,
                   init_value=init_quant_k_scales),
        TensorSpec("quant_v_scales", [cache_rows, 1], torch.float32,
                   init_value=init_quant_v_scales),
        TensorSpec("rot_matrices", [num_layers * head_dim, head_dim], torch.bfloat16,
                   init_value=init_rot_matrices),
        TensorSpec("tq_codebook", [1, N_LEVELS], torch.float32,
                   init_value=init_tq_codebook),
        TensorSpec("wo", [num_layers * hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [num_layers, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [num_layers * hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [num_layers * hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [num_layers * intermediate_size, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32,
                   init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16,
                   init_value=init_lm_head_weight),
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


# ---------------------------------------------------------------------------
# FP golden wrapper
# ---------------------------------------------------------------------------


def golden_qwen3_14b_prefill_tq(tensors):
    """TurboQuant golden: full prefill forward with dequant attention + KV compression.

    Aligned with the kernel's dequant-attention path:
      - K: per-head RMSNorm -> RoPE -> L2 norm -> normalize -> BF16 -> rotate -> Lloyd-Max quantize
      - V: L2 norm -> normalize -> BF16 -> rotate -> Lloyd-Max quantize
      - Q: per-head RMSNorm -> RoPE (no rotate, stays in original space)
      - Attention: Q vs dequant K/V (centroid gather -> renorm -> unrotate -> rescale)
      - Quantized K/V written to cache; attention reads dequantized K/V (matches decode)
      - Output projection + residual + post RMSNorm + SwiGLU MLP + residual
    """
    import math

    import torch

    from turboquant_kv import _lm_centroids, _lm_boundaries

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
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    quant_k_cache = tensors["quant_k_cache"]
    quant_v_cache = tensors["quant_v_cache"]
    quant_k_scales = tensors["quant_k_scales"]
    quant_v_scales = tensors["quant_v_scales"]
    rot_matrices = tensors["rot_matrices"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    batch = seq_lens.shape[0]
    total_tokens = hidden_states.shape[0]
    max_seq = rope_cos.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_layers = input_rms_weight.shape[0]
    intermediate_size = w_gate.shape[1]
    layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE
    centroids = _lm_centroids.float()  # [N_LEVELS]
    boundaries = _lm_boundaries.float()  # [N_LEVELS - 1]

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            lhs_chunk = lhs[:, k0:k0 + k_chunk].float()
            out += lhs_chunk @ rhs_t[:, k0:k0 + k_chunk].float().T
        return out

    def tq_quantize(x, rot_matrix_f):
        """Quantize: L2 norm -> normalize -> BF16 -> rotate -> Lloyd-Max boundary compare.

        Mirrors the kernel's quant path (turboquant_kv.turboquant_kv_quantize
        Scope B/D): the index is a cumulative GE boundary count
        idx = #{b in boundaries : x_rot >= b} (== pl.cmp cmp_type=5), NOT
        searchsorted. searchsorted/GT under-counts at exact boundary hits
        (e.g. the symmetric middle boundary b_mid == 0.0).
        """
        l2 = torch.sqrt(x.float().pow(2).sum(dim=-1, keepdim=True) + eps)
        x_norm = x.float() / l2
        x_rot = torch.matmul(x_norm.bfloat16(), rot_matrix_f.bfloat16()).float()
        # Cumulative GE boundary comparison (== pl.cmp cmp_type=5, GE).
        indices = torch.zeros(x_rot.shape, dtype=torch.int32)
        for bval in boundaries:
            indices += (x_rot >= bval).to(torch.int32)
        # Pack two 4-bit indices per UINT8 (half/half layout), matching the kernel:
        # byte c = (idx[c+half] << 4) | idx[c].  Cast mirrors kernel INT32->FP16->UINT8.
        packed = (indices[:, half:] << 4) | indices[:, :half]            # [N, half] int32
        packed_u8 = packed.to(torch.float16).to(torch.uint8)
        return packed_u8, l2.float(), x_rot

    def tq_dequant(indices, scales, rot_matrix_f):
        """Dequantize: centroid gather -> renormalize -> rescale (rotated) -> unrotate.

        Mirrors the kernel's dequant op order (qk_dequant -> qk_renorm ->
        qk_unrotate): the per-row L2 scale is applied in the ROTATED domain
        BEFORE the unrotate matmul, not after. Mathematically the scale
        commutes through the linear unrotate, but matching the kernel's op
        order keeps the rounding points aligned.
        """
        # Unpack nibbles (half/half layout, matching the kernel): byte c -> idx[c] (lo),
        # idx[c+half] (hi).  Cast mirrors kernel UINT8->FP16->INT32.
        idx_int32 = indices.to(torch.float16).to(torch.int32)        # [N, half]
        lo = (idx_int32 & 0xF).long()                                # [N, half]
        hi = (idx_int32 >> 4).long()                                 # [N, half]
        full = torch.cat([lo, hi], dim=-1)                           # [N, head_dim]
        y_hat = centroids[full]  # [N, HEAD_DIM] in rotated space

        # Renormalize to unit sphere (rsqrt == kernel's qk_renorm).
        y_norms_sq = y_hat.float().pow(2).sum(dim=-1, keepdim=True)
        y_hat = y_hat.float() * torch.rsqrt(y_norms_sq + eps)

        # Rescale by stored L2 norms in the ROTATED domain (kernel order).
        y_hat = y_hat * scales

        # Unrotate back to original space.
        return torch.matmul(y_hat, rot_matrix_f.T)

    hidden = hidden_states.clone()
    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cache_base = layer_idx * layer_cache_rows
        input_rms_weight_f = input_rms_weight[layer_idx:layer_idx + 1, :].float()
        wq_f = wq[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wk_f = wk[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wv_f = wv[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wo_f = wo[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        post_rms_f = post_rms_weight[layer_idx:layer_idx + 1, :].float()
        w_gate_f = w_gate[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_up_f = w_up[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_down_f = w_down[layer_inter_base:layer_inter_base + intermediate_size, :].float()

        # Per-layer rotation matrix.
        rot_base = layer_idx * head_dim
        rot_matrix_f = rot_matrices[rot_base:rot_base + head_dim, :].float()

        out_t = torch.zeros(total_tokens, hidden_size, dtype=torch.float32)
        token_base = 0
        for b in range(batch):
            seq_len_b = int(seq_lens[b].item())
            if seq_len_b <= 0:
                token_base += seq_len_b
                continue

            S = seq_len_b

            # ── Scope 1: RMSNorm + Q/K/V projection ──
            x = hidden[token_base:token_base + S, :].float()
            variance = x.square().mean(dim=-1, keepdim=True) + eps
            inv_rms = 1.0 / torch.sqrt(variance)
            normed_bf16 = (x * inv_rms * input_rms_weight_f).to(torch.bfloat16)

            normed_f32 = normed_bf16.float()
            q_proj_f = normed_f32 @ wq_f
            k_proj_f = normed_f32 @ wk_f
            v_proj_f = normed_f32 @ wv_f

            # Per-head RMSNorm on Q and K (FP32).
            q_norm_f = q_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            k_norm_f = k_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            q_proj_view = q_proj_f.view(S, num_heads, head_dim)
            q_var = q_proj_view.pow(2).mean(dim=-1, keepdim=True)
            q_proj_view = q_proj_view * torch.rsqrt(q_var + eps) * q_norm_f
            q_proj_f = q_proj_view.reshape(S, hidden_size)
            k_proj_view = k_proj_f.view(S, num_kv_heads, head_dim)
            k_var = k_proj_view.pow(2).mean(dim=-1, keepdim=True)
            k_proj_view = k_proj_view * torch.rsqrt(k_var + eps) * k_norm_f
            k_proj_f = k_proj_view.reshape(S, kv_hidden)

            # ── Scope 2a: RoPE + TQ compress K/V ──
            cos_row = rope_cos[:S, :]
            sin_row = rope_sin[:S, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            # K RoPE.
            k_view = k_proj_f.view(S, num_kv_heads, head_dim)
            k_lo, k_hi = k_view[:, :, :half], k_view[:, :, half:]
            k_rot_lo = k_lo * cos_lo.unsqueeze(1) - k_hi * sin_lo.unsqueeze(1)
            k_rot_hi = k_hi * cos_hi.unsqueeze(1) + k_lo * sin_hi.unsqueeze(1)
            k_rope = torch.cat([k_rot_lo, k_rot_hi], dim=-1)  # [S, num_kv_heads, head_dim]

            # TQ quantize K (per head) — pure PolarQuant, no QJL.
            quant_k_all = torch.zeros(S, num_kv_heads, half, dtype=torch.uint8)
            k_scales_all = torch.zeros(S, num_kv_heads, 1, dtype=torch.float32)
            for h in range(num_kv_heads):
                qk, ks, _ = tq_quantize(k_rope[:, h, :].reshape(-1, head_dim), rot_matrix_f)
                quant_k_all[:, h, :] = qk
                k_scales_all[:, h, :] = ks


            # TQ quantize V (per head, no RoPE).
            v_view = v_proj_f.view(S, num_kv_heads, head_dim)
            quant_v_all = torch.zeros(S, num_kv_heads, half, dtype=torch.uint8)
            v_scales_all = torch.zeros(S, num_kv_heads, 1, dtype=torch.float32)
            for h in range(num_kv_heads):
                qv, vs, _ = tq_quantize(v_view[:, h, :].reshape(-1, head_dim), rot_matrix_f)
                quant_v_all[:, h, :] = qv
                v_scales_all[:, h, :] = vs

            # Write compressed K/V to cache.
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                    quant_k_cache[cache_row, :] = quant_k_all[pos, ki, :]
                    quant_v_cache[cache_row, :] = quant_v_all[pos, ki, :]
                    quant_k_scales[cache_row, 0] = torch.tensor(
                        k_scales_all[pos, ki, 0].item(), dtype=torch.bfloat16)
                    quant_v_scales[cache_row, 0] = torch.tensor(
                        v_scales_all[pos, ki, 0].item(), dtype=torch.bfloat16)

            # Q RoPE (post per-head norm; matches kernel op at scope_rope_q).
            q_view = q_proj_f.view(S, num_heads, head_dim)
            q_lo, q_hi = q_view[:, :, :half], q_view[:, :, half:]
            q_rot_lo = q_lo * cos_lo.unsqueeze(1) - q_hi * sin_lo.unsqueeze(1)
            q_rot_hi = q_hi * cos_hi.unsqueeze(1) + q_lo * sin_hi.unsqueeze(1)
            q_rope = torch.cat([q_rot_lo, q_rot_hi], dim=-1)  # [S, num_heads, head_dim]
            q_rope_bf16 = q_rope.bfloat16()

            # ── Scope 2b: Causal attention with dequantized K/V (matches NPU prefill) ──
            max_blocks = (S + SEQ_TILE - 1) // SEQ_TILE
            padded_len = max_blocks * SEQ_TILE
            ctx_lens = torch.arange(1, S + 1)
            col_idx = torch.arange(SEQ_TILE)
            attn_result = torch.zeros(S, hidden_size, dtype=torch.float32)

            for kvh in range(num_kv_heads):
                # Dequant K/V from the compressed cache (centroids -> renorm -> unrotate -> rescale).
                k_deq = tq_dequant(
                    quant_k_all[:, kvh, :], k_scales_all[:, kvh, :], rot_matrix_f,
                )  # [S, head_dim] FP32
                v_deq = tq_dequant(
                    quant_v_all[:, kvh, :], v_scales_all[:, kvh, :], rot_matrix_f,
                )  # [S, head_dim] FP32

                # Padded buffers (sequential order); BF16 to match the kernel's matmul dtype.
                k_fp_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                v_fp_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                k_fp_padded[:S] = k_deq.bfloat16()
                v_fp_padded[:S] = v_deq.bfloat16()

                for qg in range(q_groups):
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp = q_rope_bf16[:S, q_base:q_base + Q_HEAD_BATCH, :]

                    oi = li = mi = None

                    for sb in range(max_blocks):
                        s0 = sb * SEQ_TILE
                        k_tile = k_fp_padded[s0:s0 + SEQ_TILE]
                        v_tile = v_fp_padded[s0:s0 + SEQ_TILE]

                        raw_scores = (q_grp @ k_tile.T).float()

                        valid_lens = torch.clamp(ctx_lens - s0, min=0, max=SEQ_TILE)
                        mask = col_idx.unsqueeze(0) < valid_lens.unsqueeze(1)
                        raw_scores[~mask.unsqueeze(1).expand_as(raw_scores)] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale

                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_bf16.float().sum(dim=-1, keepdim=True)

                        oi_tmp = (exp_bf16 @ v_tile).float()

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
            hs = hidden[token_base:token_base + S, :].float()

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
            out_t[token_base:token_base + S, :] = (down + resid1).bfloat16().float()
            token_base += S

        hidden = out_t.to(torch.bfloat16)

    final_hidden = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    token_base = 0
    for b in range(batch):
        seq_len_b = int(seq_lens[b].item())
        if seq_len_b > 0:
            final_hidden[b, :] = hidden[token_base + seq_len_b - 1, :]
        token_base += seq_len_b

    variance = final_hidden.float().pow(2).mean(dim=-1, keepdim=True)
    final_normed = (
        final_hidden.float()
        * torch.rsqrt(variance + eps)
        * final_norm_weight.float()
    ).bfloat16()
    tensors["out"][:] = tiled_lm_head(final_normed, lm_head_weight, LM_HEAD_K_CHUNK, VOCAB_CHUNK)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b", "--batch", type=int, default=BATCH,
        help=("User-visible batch size. Host allocates every "
              "batch-dependent tensor at exactly this size; "
              "every kernel signature batch-axis dim is a "
              "pl.dynamic() variable, so a single compiled "
              "program serves any batch <= host KV-cache "
              "capacity. Default: %(default)s"),
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False,
                        help="set all seq_lens to MAX_SEQ")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)

    result = run_jit(
        fn=prefill_fwd_tq,
        specs=build_tensor_specs(
            batch=args.batch, num_layers=args.num_layers,
            use_max_seq=args.max_seq,
        ),
        golden_fn=golden_qwen3_14b_prefill_tq,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)



