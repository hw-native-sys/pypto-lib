# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill indexer: compressed index KV cache + per-token compressed top-k."""

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    PREFILL_IDX_BLOCK_NUM,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    STATE_LEN as INNER_STATE_LEN,
    golden_prefill_indexer_compressor,
    prefill_indexer_compressor,
)

# Dynamic shape variables.
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
D = M.hidden_size
ROPE_HEAD_DIM = M.qk_rope_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
Q_LORA = M.q_lora_rank
WEIGHTS_SCALE = M.index_weights_scale
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window

COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers
IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
IDX_TOPK = M.index_topk
INNER_OVERLAP = COMPRESS_RATIO == 4
INNER_COFF = 1 + int(INNER_OVERLAP)
INNER_HEAD_DIM = IDX_HEAD_DIM
INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
B = 1
S = 128
T = B * S
START_POS = 0
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
# CP selector widths.
CP_INDEXER_SCORE_CAP = 1024
CP_INDEXER_SORT_LEN = 2048
CP_INDEXER_SELECTED_WIDTH = 256

# tiling
CACHE_TILE = 32
# Per-token sort-tile width. The sort32/mrgsort/gather path requires a wide tile: a narrow (256)
# sort faults on device (507018) even with a proper prefix. 2048 matches the indexer KV length and
# is the confirmed fault-free width. The real score occupies only the first INDEXER_SCORE_CAP
# columns; the rest stays -inf.
SORT_LEN = 2048

# History one tile reaches: SORT_LEN candidates, each covering COMPRESS_RATIO
# raw tokens. Shorter prompts skip the extra K blocks via valid_block_mask.
RAW_TOKEN_CAP = SORT_LEN * COMPRESS_RATIO
INDEXER_SCORE_CAP = SORT_LEN
INDEXER_SCORE_MAX_BLOCKS = INDEXER_SCORE_CAP // BLOCK_SIZE
INDEXER_TOPK_CAP = min(IDX_TOPK, INDEXER_SCORE_CAP)
assert RAW_TOKEN_CAP <= MAX_SEQ_LEN
assert INDEXER_SCORE_CAP == INDEXER_SCORE_MAX_BLOCKS * BLOCK_SIZE

# Named, not inlined: a python max() call is not traceable inside a loop bound.
INDEXER_SCORE_BLOCKS = max(1, (INDEXER_SCORE_CAP + CACHE_TILE - 1) // CACHE_TILE)
SCORE_TOKEN_TILE = 8
TOPK_TILE = 4
Q_TILE = 128
Q_OUT_TILE = 256
QR_PROJ_ROW_TILE = 16
HEAD_DIM_TILE = 32
D_TILE = 32
WEIGHTS_ROW_TILE = 32
QH_QUANT_TILE = 256
QH_QUANT_ROW_TILE = 64
ROPE_ROW_TILE = IDX_N_HEADS           # one token owns IDX_N_HEADS contiguous q rows + one cos/sin
ROPE_PREP_TOKEN_TILE = 16
QH_MM_TILE = 64
# topk_pairs (= 2*PREFILL_TOPK_CAP) must be a power of two aligned to the final mrgsort run: a
# misaligned prefix (e.g. 2*192) faults like a narrow sort. The real score cap is 256, so two
# merge stages are sufficient: the only finite values are sorted within the first 512-value run.
PREFILL_TOPK_CAP = INDEXER_TOPK_CAP
TOPK_PAIR_WIDTH = 2 * PREFILL_TOPK_CAP
SCORE_INIT_TILE = 16                   # rows per -inf init write; keeps [tile, SORT_LEN] under the Vec limit
SCORE_OUT_TOKEN_TILE = min(16, 16384 // INDEXER_SCORE_CAP)
assert T % SCORE_OUT_TOKEN_TILE == 0

# A misaligned topk prefix faults on device like a narrow sort.
assert TOPK_PAIR_WIDTH > 0 and (TOPK_PAIR_WIDTH & (TOPK_PAIR_WIDTH - 1)) == 0


@pl.jit.inline
def prefill_indexer(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    # === Q projection: int8 qr x int8 wq_b -> dequant (mirrors decode_indexer qr_proj) ===
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="prefill_idx_qr_proj"):
        o0 = idx * Q_OUT_TILE
        qr_acc = pl.create_tensor([T, Q_OUT_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
            q0 = kb * Q_TILE
            qr_tile = qr[:, q0 : q0 + Q_TILE]
            wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
            if q0 == 0:
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            else:
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
        wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        for r0 in pl.range(0, T, QR_PROJ_ROW_TILE):
            acc_fp32 = pl.cast(qr_acc[r0 : r0 + QR_PROJ_ROW_TILE, :], target_type=pl.FP32, mode="none")
            scale_dq = qr_scale[r0 : r0 + QR_PROJ_ROW_TILE, :]
            qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, scale_dq), wq_scale)
            qr_proj[r0 : r0 + QR_PROJ_ROW_TILE, o0 : o0 + Q_OUT_TILE] = qr_dequant

    # === Q RoPE + Hadamard rotation + per-row INT8 quant ===
    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)

    # Materialize fixed-shape interleaved RoPE rows.
    rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    for prep_idx in pl.spmd(T // ROPE_PREP_TOKEN_TILE, name_hint="prefill_idx_rope_prepare", allow_early_resolve=True):
        t0 = prep_idx * ROPE_PREP_TOKEN_TILE
        rope_ones = pl.full([ROPE_PREP_TOKEN_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        rope_col_fp32 = pl.cast(rope_col_i32, target_type=pl.FP32)
        rope_col = pl.col_expand_mul(rope_ones, rope_col_fp32)
        rope_dup_i32 = pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc")
        rope_dup_f = pl.cast(rope_dup_i32, target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_tile = cos[t0 : t0 + ROPE_PREP_TOKEN_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_tile = sin[t0 : t0 + ROPE_PREP_TOKEN_TILE, 0 : ROPE_HEAD_DIM // 2]
        cos_il = pl.gather(cos_tile, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_tile, dim=-1, index=rope_dup_idx)
        rope_cos_il[t0 : t0 + ROPE_PREP_TOKEN_TILE, :] = cos_il
        rope_sin_signed[t0 : t0 + ROPE_PREP_TOKEN_TILE, :] = pl.mul(sin_il, rope_sign)

    rope_swap_idx = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_rope_swap_idx", allow_early_resolve=True):
        swap_ones = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        swap_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        swap_col_fp32 = pl.cast(swap_col_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_col_fp32)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_index = pl.cast(pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0)), target_type=pl.INT32)
        rope_swap_idx[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = swap_index

    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for token_idx in pl.spmd(T, name_hint="prefill_idx_qr_rope", allow_early_resolve=True):
        r0 = token_idx * ROPE_ROW_TILE
        qr_nope_fp32 = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_nope = pl.cast(qr_nope_fp32, target_type=pl.BF16, mode="rint")
        qr_rope = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        rope_swap_tile = rope_swap_idx[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope, dim=-1, index=rope_swap_tile)
        rope_cos_tile = rope_cos_il[token_idx : token_idx + 1, :]
        rope_sin_tile = rope_sin_signed[token_idx : token_idx + 1, :]
        rope_main = pl.col_expand_mul(qr_rope, rope_cos_tile)
        rope_swapped = pl.col_expand_mul(qr_swapped, rope_sin_tile)
        rope_rot = pl.add(rope_main, rope_swapped)
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
        qr_bf16[r0 : r0 + ROPE_ROW_TILE, :] = pl.concat(qr_nope, rope_bf16)

    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for mm_idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="prefill_idx_qr_hadamard", allow_early_resolve=True):
        r0 = mm_idx * QH_MM_TILE
        qr_bf16_tile = qr_bf16[r0 : r0 + QH_MM_TILE, :]
        qh_acc = pl.matmul(qr_bf16_tile, hadamard, out_dtype=pl.FP32)
        qh_acc_gm[r0 : r0 + QH_MM_TILE, :] = qh_acc

    for quant_idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_ROW_TILE, name_hint="prefill_idx_qr_quant", allow_early_resolve=True):
        r0 = quant_idx * QH_QUANT_ROW_TILE
        qh_amax = pl.full(
            [1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS
        )
        for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h0 : h0 + HEAD_DIM_TILE,
            ]
            qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
            qh_row_max = pl.reshape(pl.row_max(qh_abs), [1, QH_QUANT_ROW_TILE])
            qh_amax = pl.maximum(qh_amax, qh_row_max)
        scale_max = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_quant_row = pl.div(scale_max, qh_amax)
        qr_scale_tile = pl.reshape(pl.recip(scale_quant_row), [QH_QUANT_ROW_TILE, 1])
        qr_hadamard_scale_dq[r0 : r0 + QH_QUANT_ROW_TILE, :] = qr_scale_tile
        scale_quant = pl.reshape(scale_quant_row, [QH_QUANT_ROW_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_quant_tile = qh_acc_gm[r0 : r0 + QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE]
            qh_scaled = pl.row_expand_mul(qh_quant_tile, scale_quant)
            qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
            qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[r0 : r0 + QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE] = qh_i8

    # === weights projection: (x @ weights_proj) * WEIGHTS_SCALE ===
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint="prefill_idx_weights_proj"):
        wrow0 = idx * WEIGHTS_ROW_TILE
        weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_tile = x[wrow0 : wrow0 + WEIGHTS_ROW_TILE, d0 : d0 + D_TILE]
            wp_tile = weights_proj[d0 : d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
        weights[wrow0 : wrow0 + WEIGHTS_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)

    # === inner compressor: build the paged compressed index KV cache ===
    idx_kv_cache_out, idx_kv_scale_out, inner_compress_state_out = prefill_indexer_compressor(
        x, inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape,
        inner_norm_w, freqs_cos, freqs_sin,
        hadamard, idx_kv_cache, idx_kv_scale,
        idx_block_table, position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )

    # === score: decode-style W8A8C16 scoring over the packed paged cache. The compressor already
    # stored each compressed row as INT8 + a per-position dequant scale (C8), so the score reads the
    # paged INT8 block and its scale directly, multiplies by the INT8 Hadamard Q tile with INT32
    # accumulation, then dequantizes and reduces in FP32. Runtime guards skip blocks beyond context.
    idx_block_num = pl.tensor.dim(idx_kv_cache_out, 0)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache_out, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale_out, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, SORT_LEN], dtype=pl.FP32)                                  # wide sort scratch

    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score_init"):
            score_wide[si : si + SCORE_INIT_TILE, :] = pl.full([SCORE_INIT_TILE, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)

    score_token_groups = T // SCORE_TOKEN_TILE
    for score_idx in pl.spmd(score_token_groups, name_hint="prefill_idx_score"):
        token0 = score_idx * SCORE_TOKEN_TILE
        last_pos = pl.read(position_ids, [num_tokens - 1])
        max_visible = pl.min((last_pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
        for cb in pl.range(INDEXER_SCORE_BLOCKS):
            cache0 = cb * CACHE_TILE
            if max_visible > cache0:
                kv_row0 = pl.cast(pl.read(idx_block_table, [cache0 // CACHE_TILE]), pl.INDEX) * IDX_STORAGE_BLOCK_SIZE
                # C8: the compressor stored this block as INT8 + a per-position dequant scale; read
                # both from the paged cache directly (no score-time re-quant).
                kv_q_i8_full = kv_cache_i8_flat[kv_row0 : kv_row0 + CACHE_TILE, 0 : IDX_HEAD_DIM]
                kv_cache_scale_dq = kv_scale_flat[kv_row0 : kv_row0 + CACHE_TILE, :]
                for token_offset in pl.range(SCORE_TOKEN_TILE):
                    t = token0 + token_offset
                    if t < num_tokens:
                        q_s0 = t * IDX_N_HEADS
                        qr_hadamard_tile = qr_hadamard_i8[q_s0 : q_s0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                        score_acc_s = pl.matmul(kv_q_i8_full, qr_hadamard_tile, out_dtype=pl.INT32, b_trans=True)
                        qh_scale_s = pl.reshape(qr_hadamard_scale_dq[q_s0 : q_s0 + IDX_N_HEADS, :], [1, IDX_N_HEADS])
                        score_tile_s = pl.cast(score_acc_s, target_type=pl.FP32, mode="none")
                        score_tile_s = pl.col_expand_mul(pl.row_expand_mul(score_tile_s, kv_cache_scale_dq), qh_scale_s)
                        relu_score_s = pl.maximum(score_tile_s, pl.mul(score_tile_s, 0.0))
                        weighted_score_s = pl.reshape(pl.row_sum(pl.col_expand_mul(relu_score_s, weights[t : t + 1, :])), [1, CACHE_TILE])
                        pos = pl.read(position_ids, [t])
                        visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                        if visible_t > cache0:
                            valid_len_t = pl.min(CACHE_TILE, visible_t - cache0)
                        else:
                            valid_len_t = 0
                        weighted_valid_t = pl.fillpad(pl.set_validshape(weighted_score_s, 1, valid_len_t), pad_value=pl.PadValue.min)
                        weighted_valid_t = pl.maximum(weighted_valid_t, pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF))
                        score_wide[t : t + 1, cache0 : cache0 + CACHE_TILE] = weighted_valid_t

    # Expose the real per-key scores (first INDEXER_SCORE_CAP cols of the wide sort scratch).
    score_out_flat = pl.reshape(score, [T, INDEXER_SCORE_CAP])
    if INDEXER_SCORE_CAP == 256:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_idx_score_out"):
            score_out_flat[0:T, :] = score_wide[0:T, 0:INDEXER_SCORE_CAP]
    else:
        for score_out_idx in pl.spmd(
            T // SCORE_OUT_TOKEN_TILE,
            name_hint="prefill_idx_score_out",
        ):
            score_out_t0 = score_out_idx * SCORE_OUT_TOKEN_TILE
            score_out_flat[
                score_out_t0 : score_out_t0 + SCORE_OUT_TOKEN_TILE,
                :,
            ] = score_wide[
                score_out_t0 : score_out_t0 + SCORE_OUT_TOKEN_TILE,
                0:INDEXER_SCORE_CAP,
            ]

    # === top-k per token over the visible (causally reachable) compressed positions ===
    for topk_idx in pl.spmd(T // TOPK_TILE, name_hint="prefill_idx_topk"):
        t0 = topk_idx * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            cmp_topk_indices[t : t + 1, 0:IDX_TOPK] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            if t < num_tokens:
                pos = pl.read(position_ids, [t])
                visible_t = pl.min((pos + 1) // COMPRESS_RATIO, INDEXER_SCORE_CAP)
                if visible_t > 0:
                    # Sort the wide score row and gather the top-k indices. Only the first 256
                    # values can be finite, so the 64/256 merge stages fully order the useful run.
                    score_row = score_wide[t : t + 1, :]
                    idx_init = pl.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    sorted_tile = pl.sort32(score_row, idx_init)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                    if INDEXER_SCORE_CAP > 256:
                        sorted_tile = pl.mrgsort(sorted_tile, block_len=1024)
                    topk_pairs = sorted_tile[:, 0:TOPK_PAIR_WIDTH]
                    topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid_topk = pl.min(PREFILL_TOPK_CAP, visible_t)
                    cmp_topk_indices[t : t + 1, 0:PREFILL_TOPK_CAP] = pl.set_validshape(
                        topk_idxs_tile, 1, valid_topk)

    return idx_kv_cache_out, idx_kv_scale_out, score, cmp_topk_indices


@pl.jit.inline
def _prefill_indexer_cp_score_topk(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    score: pl.Out[pl.Tensor[[T, CP_INDEXER_SCORE_CAP], pl.FP32]],
    cmp_topk_indices: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
):
    """Compute CP indexer query scores and top-k selections."""
    # Project quantized query rows.
    qr_proj = pl.create_tensor([T, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint="prefill_cp_idx_qr_proj"):
        o0 = idx * Q_OUT_TILE
        qr_acc = pl.create_tensor([T, Q_OUT_TILE], dtype=pl.INT32)
        for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
            q0 = kb * Q_TILE
            qr_tile = qr[:, q0 : q0 + Q_TILE]
            wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
            if q0 == 0:
                qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            else:
                qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
        wq_scale = pl.reshape(wq_b_scale[o0 : o0 + Q_OUT_TILE], [1, Q_OUT_TILE])
        for r0 in pl.range(0, T, QR_PROJ_ROW_TILE):
            qr_acc_tile = qr_acc[r0 : r0 + QR_PROJ_ROW_TILE, :]
            acc_fp32 = pl.cast(qr_acc_tile, target_type=pl.FP32, mode="none")
            scale_dq = qr_scale[r0 : r0 + QR_PROJ_ROW_TILE, :]
            qr_scaled = pl.row_expand_mul(acc_fp32, scale_dq)
            qr_dequant = pl.col_expand_mul(qr_scaled, wq_scale)
            qr_proj[r0 : r0 + QR_PROJ_ROW_TILE, o0 : o0 + Q_OUT_TILE] = qr_dequant

    # Materialize interleaved RoPE rows.
    rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    for prep_idx in pl.spmd(T // ROPE_PREP_TOKEN_TILE, name_hint="prefill_cp_idx_rope_prepare", allow_early_resolve=True):
        t0 = prep_idx * ROPE_PREP_TOKEN_TILE
        rope_ones = pl.full([ROPE_PREP_TOKEN_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        rope_col_fp32 = pl.cast(rope_col_i32, target_type=pl.FP32)
        rope_col = pl.col_expand_mul(rope_ones, rope_col_fp32)
        rope_dup_i32 = pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc")
        rope_dup_f = pl.cast(rope_dup_i32, target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_tile = cos[t0 : t0 + ROPE_PREP_TOKEN_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_tile = sin[t0 : t0 + ROPE_PREP_TOKEN_TILE, 0 : ROPE_HEAD_DIM // 2]
        cos_il = pl.gather(cos_tile, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_tile, dim=-1, index=rope_dup_idx)
        rope_cos_il[t0 : t0 + ROPE_PREP_TOKEN_TILE, :] = cos_il
        rope_sin_signed[t0 : t0 + ROPE_PREP_TOKEN_TILE, :] = pl.mul(sin_il, rope_sign)

    rope_swap_idx = pl.create_tensor([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cp_idx_rope_swap_idx", allow_early_resolve=True):
        swap_ones = pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        swap_col_i32 = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        swap_col_fp32 = pl.cast(swap_col_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_col_fp32)
        swap_dup_i32 = pl.cast(pl.mul(swap_col, 0.5), target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_index = pl.cast(pl.sub(pl.add(swap_col, 1.0), pl.mul(swap_lane, 2.0)), target_type=pl.INT32)
        rope_swap_idx[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = swap_index

    qr_proj_flat = pl.reshape(qr_proj, [T * IDX_N_HEADS, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)
    for token_idx in pl.spmd(T, name_hint="prefill_cp_idx_qr_rope", allow_early_resolve=True):
        r0 = token_idx * ROPE_ROW_TILE
        qr_nope_fp32 = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, 0 : IDX_NOPE_HEAD_DIM]
        qr_nope = pl.cast(qr_nope_fp32, target_type=pl.BF16, mode="rint")
        qr_rope = qr_proj_flat[r0 : r0 + ROPE_ROW_TILE, IDX_NOPE_HEAD_DIM : IDX_HEAD_DIM]
        rope_swap_tile = rope_swap_idx[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM]
        qr_swapped = pl.gather(qr_rope, dim=-1, index=rope_swap_tile)
        rope_cos_tile = rope_cos_il[token_idx : token_idx + 1, :]
        rope_sin_tile = rope_sin_signed[token_idx : token_idx + 1, :]
        rope_main = pl.col_expand_mul(qr_rope, rope_cos_tile)
        rope_swapped = pl.col_expand_mul(qr_swapped, rope_sin_tile)
        rope_rot = pl.add(rope_main, rope_swapped)
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
        qr_bf16[r0 : r0 + ROPE_ROW_TILE, :] = pl.concat(qr_nope, rope_bf16)

    qh_acc_gm = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
    for mm_idx in pl.spmd(T * IDX_N_HEADS // QH_MM_TILE, name_hint="prefill_cp_idx_qr_hadamard", allow_early_resolve=True):
        r0 = mm_idx * QH_MM_TILE
        qr_bf16_tile = qr_bf16[r0 : r0 + QH_MM_TILE, :]
        qh_acc = pl.matmul(qr_bf16_tile, hadamard, out_dtype=pl.FP32)
        qh_acc_gm[r0 : r0 + QH_MM_TILE, :] = qh_acc

    qr_hadamard_i8 = pl.create_tensor([T * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
    qr_hadamard_scale_dq = pl.create_tensor([T * IDX_N_HEADS, 1], dtype=pl.FP32)
    for quant_idx in pl.spmd(T * IDX_N_HEADS // QH_QUANT_ROW_TILE, name_hint="prefill_cp_idx_qr_quant", allow_early_resolve=True):
        r0 = quant_idx * QH_QUANT_ROW_TILE
        qh_amax = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_tile = qh_acc_gm[
                r0 : r0 + QH_QUANT_ROW_TILE,
                h0 : h0 + HEAD_DIM_TILE,
            ]
            qh_abs = pl.maximum(qh_tile, pl.neg(qh_tile))
            qh_row_max = pl.reshape(pl.row_max(qh_abs), [1, QH_QUANT_ROW_TILE])
            qh_amax = pl.maximum(qh_amax, qh_row_max)
        scale_max = pl.full([1, QH_QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_quant_row = pl.div(scale_max, qh_amax)
        qr_scale_tile = pl.reshape(pl.recip(scale_quant_row), [QH_QUANT_ROW_TILE, 1])
        qr_hadamard_scale_dq[r0 : r0 + QH_QUANT_ROW_TILE, :] = qr_scale_tile
        scale_quant = pl.reshape(scale_quant_row, [QH_QUANT_ROW_TILE, 1])
        for h1 in pl.range(0, IDX_HEAD_DIM, HEAD_DIM_TILE):
            qh_quant_tile = qh_acc_gm[r0 : r0 + QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE]
            qh_scaled = pl.row_expand_mul(qh_quant_tile, scale_quant)
            qh_i32 = pl.cast(qh_scaled, target_type=pl.INT32, mode="rint")
            qh_half = pl.cast(qh_i32, target_type=pl.FP16, mode="round")
            qh_i8 = pl.cast(qh_half, target_type=pl.INT8, mode="trunc")
            qr_hadamard_i8[r0 : r0 + QH_QUANT_ROW_TILE, h1 : h1 + HEAD_DIM_TILE] = qh_i8

    # Project per-head weights.
    weights = pl.create_tensor([T, IDX_N_HEADS], dtype=pl.FP32)
    for idx in pl.spmd(T // WEIGHTS_ROW_TILE, name_hint="prefill_cp_idx_weights_proj"):
        wrow0 = idx * WEIGHTS_ROW_TILE
        weights_acc = pl.create_tensor([WEIGHTS_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.pipeline(0, D // D_TILE, stage=2):
            d0 = db * D_TILE
            x_tile = x[wrow0 : wrow0 + WEIGHTS_ROW_TILE, d0 : d0 + D_TILE]
            wp_tile = weights_proj[d0 : d0 + D_TILE, :]
            if d0 == 0:
                weights_acc = pl.matmul(x_tile, wp_tile, out_dtype=pl.FP32)
            else:
                weights_acc = pl.matmul_acc(weights_acc, x_tile, wp_tile)
        weights[wrow0 : wrow0 + WEIGHTS_ROW_TILE, :] = pl.mul(weights_acc, WEIGHTS_SCALE)

    # Score paged INT8 cache rows.
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])
    score_wide = pl.create_tensor([T, CP_INDEXER_SORT_LEN], dtype=pl.FP32)
    for si in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cp_idx_score_init"):
            score_init_tile = pl.full([SCORE_INIT_TILE, CP_INDEXER_SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
            score_wide[si : si + SCORE_INIT_TILE, :] = score_init_tile

    for score_idx in pl.spmd(T // SCORE_TOKEN_TILE, name_hint="prefill_cp_idx_score"):
        token0 = score_idx * SCORE_TOKEN_TILE
        last_pos = pl.read(position_ids, [num_tokens - 1])
        visible_limit = pl.min((last_pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
        for cb in pl.range(CP_INDEXER_SCORE_CAP // CACHE_TILE):
            cache0 = cb * CACHE_TILE
            logical_block = cache0 // CACHE_TILE
            if visible_limit > cache0 and logical_block < IDX_CACHE_MAX_BLOCKS:
                physical_block_raw = pl.read(idx_block_table, [logical_block])
                if physical_block_raw >= 0 and physical_block_raw < idx_block_num:
                    kv_row0 = pl.cast(physical_block_raw, pl.INDEX) * IDX_STORAGE_BLOCK_SIZE
                    kv_q_i8_full = kv_cache_i8_flat[
                        kv_row0 : kv_row0 + CACHE_TILE, 0:IDX_HEAD_DIM
                    ]
                    kv_cache_scale_dq = kv_scale_flat[kv_row0 : kv_row0 + CACHE_TILE, :]
                    for token_offset in pl.range(SCORE_TOKEN_TILE):
                        t = token0 + token_offset
                        if t < num_tokens:
                            q_s0 = t * IDX_N_HEADS
                            qr_hadamard_i8_tile = qr_hadamard_i8[q_s0 : q_s0 + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                            score_acc_s = pl.matmul(kv_q_i8_full, qr_hadamard_i8_tile, out_dtype=pl.INT32, b_trans=True)
                            qh_scale_source = qr_hadamard_scale_dq[q_s0 : q_s0 + IDX_N_HEADS, :]
                            qh_scale_s = pl.reshape(qh_scale_source, [1, IDX_N_HEADS])
                            score_acc_fp32 = pl.cast(score_acc_s, target_type=pl.FP32, mode="none")
                            score_row_scaled = pl.row_expand_mul(score_acc_fp32, kv_cache_scale_dq)
                            score_scaled = pl.col_expand_mul(score_row_scaled, qh_scale_s)
                            score_zero = pl.mul(score_scaled, 0.0)
                            relu_score_s = pl.maximum(score_scaled, score_zero)
                            weighted_heads = pl.col_expand_mul(relu_score_s, weights[t : t + 1, :])
                            weighted_sum = pl.row_sum(weighted_heads)
                            weighted_score_s = pl.reshape(weighted_sum, [1, CACHE_TILE])
                            pos = pl.read(position_ids, [t])
                            visible_t = pl.min((pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
                            if visible_t > cache0:
                                valid_len_t = pl.min(CACHE_TILE, visible_t - cache0)
                            else:
                                valid_len_t = 0
                            weighted_visible = pl.set_validshape(weighted_score_s, 1, valid_len_t)
                            weighted_valid_t = pl.fillpad(weighted_visible, pad_value=pl.PadValue.min)
                            neg_inf_tile = pl.full([1, CACHE_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            weighted_valid_t = pl.maximum(weighted_valid_t, neg_inf_tile)
                            score_wide[t : t + 1, cache0 : cache0 + CACHE_TILE] = weighted_valid_t

    # Write 1024 scores and 256 top-k indices.
    for score_out0 in pl.parallel(0, T, SCORE_INIT_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cp_idx_score_out"):
            score_tile = score_wide[score_out0 : score_out0 + SCORE_INIT_TILE, 0:CP_INDEXER_SCORE_CAP]
            score[score_out0 : score_out0 + SCORE_INIT_TILE, :] = score_tile
    for topk_idx in pl.spmd(T // TOPK_TILE, name_hint="prefill_cp_idx_topk"):
        t0 = topk_idx * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            topk_init = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            cmp_topk_indices[t : t + 1, 0:IDX_TOPK] = topk_init
            if t < num_tokens:
                pos = pl.read(position_ids, [t])
                visible_t = pl.min((pos + 1) // COMPRESS_RATIO, CP_INDEXER_SCORE_CAP)
                if visible_t > 0:
                    score_row = score_wide[t : t + 1, :]
                    idx_init = pl.arange(0, [1, CP_INDEXER_SORT_LEN], dtype=pl.UINT32)
                    sorted_tile = pl.sort32(score_row, idx_init)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=64)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=256)
                    sorted_tile = pl.mrgsort(sorted_tile, block_len=1024)
                    topk_pairs = sorted_tile[:, 0 : 2 * CP_INDEXER_SELECTED_WIDTH]
                    topk_idxs_tile = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid_topk = pl.min(CP_INDEXER_SELECTED_WIDTH, visible_t)
                    topk_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                    cmp_topk_indices[t : t + 1, 0:CP_INDEXER_SELECTED_WIDTH] = topk_valid

    return score, cmp_topk_indices


def topk_prefix_contract_error(topk_indices, position_ids, num_tokens):
    """Return an error string if the top-k prefix contract is broken."""
    import torch

    if hasattr(num_tokens, "item"):
        num_tokens = num_tokens.item()
    num_tokens = int(num_tokens)
    for t in range(T):
        row = topk_indices[t]
        if t >= num_tokens:
            non_padding = int((row != -1).count_nonzero().item())
            if non_padding:
                return f"inactive top-k row {t} contains {non_padding} non--1 entries"
            continue
        visible = min(
            int((int(position_ids[t].item()) + 1) // COMPRESS_RATIO),
            INDEXER_TOPK_CAP,
        )
        prefix = row[:visible]
        if visible:
            out_of_range = int(
                ((prefix < 0) | (prefix >= visible)).count_nonzero().item()
            )
            if out_of_range:
                return (
                    f"top-k row {t} has {out_of_range} entries outside "
                    f"[0, {visible}) in its visible prefix"
                )
            unique_count = int(torch.unique(prefix).numel())
            if unique_count != visible:
                return (
                    f"top-k row {t} visible prefix has "
                    f"{unique_count}/{visible} unique entries"
                )
        tail_non_padding = int((row[visible:] != -1).count_nonzero().item())
        if tail_non_padding:
            return f"top-k row {t} tail contains {tail_non_padding} non--1 entries"
    return None


def golden_prefill_indexer_core(tensors):
    from utils import int8_quant_per_row
    import torch

    compressor_tensors = {
        "x": tensors["x"],
        "kv": torch.zeros(MAX_CMP_WRITES, IDX_HEAD_DIM, dtype=torch.bfloat16),
        "compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    }
    golden_prefill_indexer_compressor(compressor_tensors)
    tensors["idx_kv_cache"][:] = compressor_tensors["idx_kv_cache"]
    tensors["idx_kv_scale"][:] = compressor_tensors["idx_kv_scale"]

    # --- Real lightning-indexer score + per-token causal-masked top-k ---
    # Ports the official model.py Indexer.forward(start_pos==0) branch (and the deleted #505^
    # prefill indexer): score each token's query against the compressed index KV through the
    # W8A8C16 int8 path, causal-mask each token to the positions it can reach ((pos+1)//ratio),
    # then top-k. Replaces the old placeholder (sequential arange+offset, i.e. the dense
    # get_compress_topk_idxs pattern, which never exercised real selection).
    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"].long()
    rd = ROPE_HEAD_DIM
    cmp_topk_indices = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
    score_full = torch.full((T, INDEXER_SCORE_CAP), FP32_NEG_INF, dtype=torch.float32)
    visible = ((position_ids + 1) // COMPRESS_RATIO).clamp(max=INDEXER_SCORE_CAP)
    max_visible = int(visible[:num_tokens].max().item()) if num_tokens > 0 else 0
    if max_visible == 0:
        return cmp_topk_indices, score_full

    # Q: int8 qr x int8 wq_b -> dequant -> per-token interleaved RoPE -> Hadamard rotation.
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    hadamard = tensors["hadamard"].float()
    cos = tensors["cos"].float().view(T, 1, -1)
    sin = tensors["sin"].float().view(T, 1, -1)
    q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
    q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    q_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    q0, q1 = q_pair[..., 0], q_pair[..., 1]
    y0 = (q0 * cos - q1 * sin).to(torch.bfloat16)
    y1 = (q0 * sin + q1 * cos).to(torch.bfloat16)
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
    q = q.to(torch.bfloat16).float() @ hadamard

    weights = (tensors["x"].float() @ tensors["weights_proj"].float()) * WEIGHTS_SCALE  # [T, heads]

    # C8: the compressor already stored INT8 KV + a per-position dequant scale. Gather both in
    # compressed-position order through the paged block table (no score-time re-quant).
    cache_flat_i8 = tensors["idx_kv_cache"].reshape(
        PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM
    )
    scale_flat = tensors["idx_kv_scale"].float().reshape(
        PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE, 1
    )
    idx_block_table = tensors["idx_block_table"]
    rows = [
        int(idx_block_table[c // CACHE_TILE].item()) * IDX_STORAGE_BLOCK_SIZE + (c % CACHE_TILE)
        for c in range(max_visible)
    ]
    kv_i8 = torch.stack([cache_flat_i8[r] for r in rows], dim=0).to(torch.int32)  # [max_visible, dim]
    kv_sc = torch.stack([scale_flat[r] for r in rows], dim=0).view(1, 1, max_visible)

    # W8A8C16 int8 score, matching decode_indexer: per-row quantize q, INT32 matmul against the
    # pre-quantized KV, then dequantize by both scales before the FP32 head-weighted reduce.
    q_i8, q_sc = int8_quant_per_row(q.reshape(T * IDX_N_HEADS, IDX_HEAD_DIM))
    q_i8 = q_i8.view(T, IDX_N_HEADS, IDX_HEAD_DIM).to(torch.int32)
    q_sc = q_sc.view(T, IDX_N_HEADS, 1)
    score_i32 = torch.einsum("thd,cd->thc", q_i8, kv_i8)
    score = score_i32.float() * q_sc * kv_sc
    score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)  # [T, max_visible]

    # Per-token causal mask, then top-k over the visible compressed positions.
    col = torch.arange(max_visible).unsqueeze(0)
    score = score.masked_fill(col >= visible.unsqueeze(1), FP32_NEG_INF)
    score_full[:, :max_visible] = score
    for t in range(num_tokens):
        k = int(min(INDEXER_TOPK_CAP, int(visible[t].item())))
        if k > 0:
            sel = score[t].topk(k, dim=-1)[1]
            cmp_topk_indices[t, :k] = sel.to(torch.int32)
    return cmp_topk_indices, score_full


def golden_prefill_indexer(tensors):
    import torch

    cmp_topk_indices, score_full = golden_prefill_indexer_core(tensors)
    topk_idxs = cmp_topk_indices[:, 0:INDEXER_TOPK_CAP].to(torch.int32)
    tensors["score"][:] = score_full
    tensors["topk_idxs"][:] = topk_idxs


@pl.jit
def prefill_indexer_test(
    x: pl.Tensor[[T, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM // 2], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Out[pl.Tensor[[T, INDEXER_SCORE_CAP], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[T, INDEXER_TOPK_CAP], pl.INT32]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    prefill_indexer(
        x, qr, qr_scale, wq_b, wq_b_scale, weights_proj,
        cos, sin, freqs_cos, freqs_sin, hadamard,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        score, cmp_topk_indices,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )
    # Expose the kernel's topk as topk_idxs.
    for tb in pl.spmd(T // TOPK_TILE, name_hint="prefill_idx_topk_copy"):
        t0 = tb * TOPK_TILE
        for ti in pl.range(TOPK_TILE):
            t = t0 + ti
            topk_idxs[t : t + 1, 0:INDEXER_TOPK_CAP] = cmp_topk_indices[t : t + 1, 0:INDEXER_TOPK_CAP]
    return score, idx_kv_cache, idx_kv_scale, topk_idxs


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a per-output-channel-symmetric INT8 weight + FP32 scale on the real
    DeepSeek-V4-Flash MXFP8 grid (e4m3 + 128x128-block E8M0 scale), then re-quantize
    per-output-channel. Mirrors decode_indexer.gen_shared_weight; ``shape`` last dim is the
    reduction (in) dim, leading dims map to the per-output-channel scale ([out, in] -> [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs(start_pos: int = START_POS, num_tokens: int = T):
    from utils import int8_quant_per_row
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")
    max_visible = (start_pos + num_tokens) // COMPRESS_RATIO
    if max_visible > INDEXER_SCORE_CAP:
        raise ValueError(
            f"prefill_indexer needs max_visible={max_visible} compressed slots for start_pos={start_pos}, "
            f"but the standalone score cap is INDEXER_SCORE_CAP={INDEXER_SCORE_CAP}."
        )
    write_count = sum(1 for t in range(num_tokens) if (start_pos + t + 1) % COMPRESS_RATIO == 0)
    if write_count > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {write_count} compressed writes, cap is {MAX_CMP_WRITES}")

    def init_inner_compress_state_block_table():
        table = torch.full((INNER_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(INNER_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_INNER_STATE_PHYSICAL_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_inner_compress_state_block_table()
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        return int(table[block].item()) * INNER_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_hadamard():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return (h * (IDX_HEAD_DIM ** -0.5)).to(torch.bfloat16)
    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM)
        flat = state.view(-1, INNER_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - INNER_STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(INNER_COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash indexer inner compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_indexer.
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0293
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INNER_HEAD_DIM)
    # C8 historical index cache: completed compressed slots hold INT8 + a per-position dequant scale.
    # Build both from one bf16-rounded random draw so cache and scale stay consistent.
    _idx_hist = {}
    def _build_idx_hist():
        if "cache" in _idx_hist:
            return
        cache_i8 = torch.zeros(
            PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.int8
        )
        scale = torch.zeros(PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1)
        c_flat = cache_i8.view(PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM)
        s_flat = scale.view(PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE, 1)
        completed = start_pos // COMPRESS_RATIO
        for cmp_slot in range(completed):
            row = idx_row(cmp_slot)
            if row >= PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE:
                raise ValueError("fixture historical compressed slot exceeds standalone idx_kv_cache capacity")
            if row >= 0:
                hist_bf16 = ((torch.rand(IDX_HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
                hi8, hsc = int8_quant_per_row(hist_bf16.float().view(1, IDX_HEAD_DIM))
                c_flat[row] = hi8.view(IDX_HEAD_DIM)
                s_flat[row] = hsc.view(1)
        _idx_hist["cache"] = cache_i8
        _idx_hist["scale"] = scale
    def init_idx_kv_cache():
        _build_idx_hist()
        return _idx_hist["cache"].clone()
    def init_idx_kv_scale():
        _build_idx_hist()
        return _idx_hist["scale"].clone()
    def init_idx_block_table():
        table = torch.full((IDX_CACHE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(IDX_CACHE_MAX_BLOCKS):
            table[block] = block
        return table
    def idx_row(cmp_slot):
        table = init_idx_block_table()
        block = cmp_slot // IDX_STORAGE_BLOCK_SIZE
        intra = cmp_slot % IDX_STORAGE_BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * IDX_STORAGE_BLOCK_SIZE + intra
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_idx_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(num_tokens):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = idx_row((pos + 1) // COMPRESS_RATIO - 1)
                if dst_row >= PREFILL_IDX_BLOCK_NUM * IDX_STORAGE_BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone idx_kv_cache capacity")
                mapping[t] = dst_row
        return mapping
    def init_inner_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(num_tokens):
            mapping[t] = state_row(start_pos + t)
        return mapping
    def init_weights_proj():
        # weights_proj calibrated to the real DeepSeek-V4-Flash indexer weights projection.
        return torch.randn(D, IDX_N_HEADS) * 0.2313
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_position_ids().to(torch.int64))[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_position_ids().to(torch.int64))[1]

    # idx wq_b uses the real MXFP8 grid (not a benign randn int8); qr is per-row int8 like the
    # runtime W8A8C16 activation path.
    wq_b_i8_T, wq_b_scale = gen_shared_weight((IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = int8_quant_per_row(torch.rand(T, Q_LORA))

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [T, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [T, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [INNER_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("idx_kv_cache", [PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.int8, init_value=init_idx_kv_cache, is_output=True),
        TensorSpec("idx_kv_scale", [PREFILL_IDX_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale, is_output=True),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("score", [T, INDEXER_SCORE_CAP], torch.float32, is_output=True),
        TensorSpec("topk_idxs", [T, INDEXER_TOPK_CAP], torch.int32, is_output=True),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill indexer validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids and dense idx_slot_mapping.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token prefix; inactive top-k rows must remain all -1.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    def topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        score = actual_outputs["score"]
        a_top = actual[..., :IDX_TOPK]
        e_top = expected[..., :IDX_TOPK]
        contract_error = topk_prefix_contract_error(
            a_top,
            inputs["position_ids"],
            args.num_tokens,
        )
        if contract_error:
            return False, f"    {contract_error}"
        invalid_top = a_top < 0
        a_orig = a_top.long().clamp(min=0, max=score.shape[-1] - 1)
        paired = torch.gather(score, dim=-1, index=a_orig)
        paired = torch.where(invalid_top, torch.full_like(paired, -torch.inf), paired)
        synth_actual = {**actual_outputs, "_topk_paired_scores": paired}
        return topk_pair_compare("_topk_paired_scores")(
            a_top, e_top,
            actual_outputs=synth_actual,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol, atol=atol,
        )
    topk_idxs_compare.__name__ = "topk_pair_compare"

    result = run_jit(
        fn=prefill_indexer_test,
        specs=build_tensor_specs(args.start_pos, args.num_tokens),
        golden_fn=golden_prefill_indexer,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            # Inactive score rows carry -inf from the sort scratch, not zeros, hence no zero_tail.
            "score": ratio_allclose(atol=1e-4, rtol=1.0 / 128, valid_rows=args.num_tokens),
            "topk_idxs": topk_idxs_compare,
            # C8 cache: INT8 rows exact bar boundary +/-1 LSB; scale rides alongside.
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
