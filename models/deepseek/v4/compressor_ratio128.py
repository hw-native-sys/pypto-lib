# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 ratio-128 compressor decode and prefill paths."""

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    DECODE_CMP_BLOCK_NUM,
    FP32_NEG_INF,
    KV_CMP_MAX_BLOCKS,
    PREFILL_BATCH,
    PREFILL_SEQ,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
)
from compressor_schedule import build_decode_padded_write_schedule, build_prefill_write_schedule, gather_compressor_rope_rows


EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM
POOL_HEAD_TILE = 128
RATIO128_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE

# Decode shape and paging contract.
DECODE_B = DECODE_BATCH
DECODE_S = DECODE_SEQ
DECODE_T = DECODE_B * DECODE_S
DECODE_IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
DECODE_COMPRESS_STATE_BLOCK_SIZE = RATIO128_STATE_BLOCK_SIZE
DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS = 64
DECODE_COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + DECODE_COMPRESS_STATE_BLOCK_SIZE - 1) // DECODE_COMPRESS_STATE_BLOCK_SIZE
DECODE_COMPRESS_STATE_BLOCK_NUM = DECODE_B * DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS
DECODE_COMPRESSOR_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
DECODE_COMPRESSOR_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
if DECODE_IDX_KV_LEN > DECODE_COMPRESSOR_CMP_MAX_BLOCKS * BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# Decode tiling.
DECODE_ROPE_TILE = 32
DECODE_K_TILE = 512
DECODE_OUT_TILE = 64
DECODE_HEAD_TILE = 64
DECODE_B_TILE = 8
DECODE_MM_B_TILE = 16
DECODE_BS_PAD = ((DECODE_B * DECODE_S + DECODE_MM_B_TILE - 1) // DECODE_MM_B_TILE) * DECODE_MM_B_TILE
DECODE_RMS_TILE = 4
DECODE_RMS_PAD_TILE = 16
DECODE_RMS_PAD_TAIL = DECODE_RMS_PAD_TILE - DECODE_RMS_TILE
DECODE_RMS_PAD_ROWS = (DECODE_B // DECODE_RMS_TILE) * DECODE_RMS_PAD_TILE
DECODE_POOL_HEAD_TILE = 128

# Prefill shape and paging contract.
PREFILL_B = PREFILL_BATCH
PREFILL_S = PREFILL_SEQ
PREFILL_T = PREFILL_B * PREFILL_S
PREFILL_START_POS = 0

K_TILE = 512
OUT_TILE = 32  # prefill (large M): finer tiles fill the array
HEAD_TILE = 64
K_BLOCKS = D // K_TILE
OUT_BLOCKS = OUT_DIM // OUT_TILE
HEAD_BLOCKS = HEAD_DIM // HEAD_TILE

assert PREFILL_S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"

HCA_STATE_BLOCK_SIZE = RATIO128_STATE_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_MAX_BLOCKS
MAX_CMP_WRITES = max(1, PREFILL_T // COMPRESS_RATIO)
HCA_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
HCA_KV_STORE_TILE = 16
HCA_C128_RMS_TILE = 8
HCA_C128_RMS_PAD_ROWS = HCA_C128_RMS_TILE

PACKED_C128_PROJ_BLOCKS = OUT_BLOCKS
POOL_HEAD_BLOCKS = HEAD_DIM // POOL_HEAD_TILE
PACKED_C128_POOL_BLOCKS = MAX_CMP_WRITES * POOL_HEAD_BLOCKS


# Shared ratio128 sub-kernel tiling.
PROJ_ROWS = pl.dynamic("COMPRESSOR_PROJ_ROWS")
PROJ_ROWS_PAD = pl.dynamic("COMPRESSOR_PROJ_ROWS_PAD")
PROJ_MM_B_TILE = 16
PROJ_OUT_TILE = 64  # decode (small M): coarse tiles avoid dispatch overhead
PROJ_K_TILE = 512
RMS_ROW_TILE = 8
COMPRESSOR_RMS_ROWS = pl.dynamic("COMPRESSOR_RMS_ROWS")
POOL_STATE_ROWS = pl.dynamic("COMPRESSOR128_POOL_STATE_ROWS")
POOL_TABLE_ROWS = pl.dynamic("COMPRESSOR128_POOL_TABLE_ROWS")
POOL_TABLE_BLOCKS = pl.dynamic("COMPRESSOR128_POOL_TABLE_BLOCKS")
POOL_STATE_BLOCKS = STATE_LEN // RATIO128_STATE_BLOCK_SIZE

# Shared-core dynamic shapes (bind per caller: decode vs prefill). The state and
# cmp-cache tensors are passed as pre-reshaped flat views so their shapes stay
# statically inferable per call site (dim()-derived reshapes inside the core are
# not).
CORE_PROJ_ROWS = pl.dynamic("COMPRESSOR_CORE_PROJ_ROWS")
CORE_TOKENS = pl.dynamic("COMPRESSOR_CORE_TOKENS")
CORE_WRITE_ROWS = pl.dynamic("COMPRESSOR_CORE_WRITE_ROWS")
CORE_STATE_ROWS = pl.dynamic("COMPRESSOR_CORE_STATE_ROWS")
CORE_TABLE_ROWS = pl.dynamic("COMPRESSOR_CORE_TABLE_ROWS")
CORE_TABLE_BLOCKS = pl.dynamic("COMPRESSOR_CORE_TABLE_BLOCKS")
CORE_CMP_ROWS = pl.dynamic("COMPRESSOR_CORE_CMP_ROWS")
# Compact per-regime pool enumeration length: decode binds it to the batch-row
# count (real windows), prefill to its write-row count. Pooling only the real
# windows — instead of every RMS-padded write row — keeps decode off the padded
# grid that otherwise inflates its pool/init dispatch.
CORE_POOL_ROWS = pl.dynamic("COMPRESSOR_CORE_POOL_ROWS")
POOL_HEAD_BLOCKS_CORE = HEAD_DIM // POOL_HEAD_TILE


@pl.jit.inline
def compressor_ratio128_pool_math(
    score_state: pl.Tensor[[STATE_LEN, POOL_HEAD_TILE], pl.FP32],
    kv_state: pl.Tensor[[STATE_LEN, POOL_HEAD_TILE], pl.FP32],
):
    score_max = pl.col_max(score_state)
    score_exp = pl.col_expand_expdif(score_state, score_max)
    score_sum = pl.col_sum(score_exp)
    score_prob = pl.col_expand_mul(score_exp, pl.recip(score_sum))
    pooled_chunk = pl.col_sum(pl.mul(kv_state, score_prob))
    return pooled_chunk


@pl.jit.inline
def compressor_ratio128_pool_window(
    compress_state_rows: pl.Tensor[[POOL_STATE_ROWS, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[POOL_TABLE_ROWS, POOL_TABLE_BLOCKS], pl.INT32],
    table_row: pl.Scalar[pl.INDEX],
    write_pos: pl.Scalar[pl.INT32],
    h0: pl.Scalar[pl.INDEX],
):
    softmax_score_state = pl.create_tensor([STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
    softmax_kv_state = pl.create_tensor([STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
    state_pos0 = write_pos + 1 - COMPRESS_RATIO
    base_logical_blk = pl.cast(state_pos0 // RATIO128_STATE_BLOCK_SIZE, pl.INDEX)
    for blk_i in pl.pipeline(POOL_STATE_BLOCKS, stage=2):
        s0 = blk_i * RATIO128_STATE_BLOCK_SIZE
        slot_score = pl.full([RATIO128_STATE_BLOCK_SIZE, POOL_HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
        slot_kv = pl.full([RATIO128_STATE_BLOCK_SIZE, POOL_HEAD_TILE], dtype=pl.FP32, value=0.0)
        state_blk_raw = pl.read(compress_state_block_table, [table_row, base_logical_blk + blk_i])
        if state_blk_raw >= 0:
            state_blk_id = pl.cast(state_blk_raw, target_type=pl.INDEX)
            row0 = state_blk_id * RATIO128_STATE_BLOCK_SIZE
            slot_score = compress_state_rows[
                row0 : row0 + RATIO128_STATE_BLOCK_SIZE,
                OUT_DIM + h0 : OUT_DIM + h0 + POOL_HEAD_TILE,
            ]
            slot_kv = compress_state_rows[
                row0 : row0 + RATIO128_STATE_BLOCK_SIZE,
                h0 : h0 + POOL_HEAD_TILE,
            ]
        softmax_score_state[s0 : s0 + RATIO128_STATE_BLOCK_SIZE, :] = slot_score
        softmax_kv_state[s0 : s0 + RATIO128_STATE_BLOCK_SIZE, :] = slot_kv
    return compressor_ratio128_pool_math(softmax_score_state, softmax_kv_state)


@pl.jit.inline
def compressor_ratio128_proj(
    x: pl.Tensor[[PROJ_ROWS, D], pl.BF16],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    kv_proj_out: pl.Tensor[[PROJ_ROWS_PAD, OUT_DIM], pl.FP32],
    score_proj_out: pl.Tensor[[PROJ_ROWS_PAD, OUT_DIM], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    t_matmul = pl.tensor.dim(kv_proj_out, 0)
    for idx in pl.spmd(t_matmul * OUT_DIM // (PROJ_MM_B_TILE * PROJ_OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * PROJ_MM_B_TILE
        o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([PROJ_MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([PROJ_MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            x_rows = pl.min(PROJ_MM_B_TILE, t_dim - global_row0)
            x_tile = pl.slice(x, [PROJ_MM_B_TILE, PROJ_K_TILE], [global_row0, k0], valid_shape=[x_rows, PROJ_K_TILE])
            wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + PROJ_K_TILE]
            wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + PROJ_K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        kv_proj_out[global_row0 : global_row0 + PROJ_MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
        score_proj_out[global_row0 : global_row0 + PROJ_MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc


@pl.jit.inline
def prefill_compressor_ratio128_proj(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    kv_proj_out: pl.Tensor[[PREFILL_T, OUT_DIM], pl.FP32],
    score_proj_out: pl.Tensor[[PREFILL_T, OUT_DIM], pl.FP32],
):
    for proj_idx in pl.spmd(PACKED_C128_PROJ_BLOCKS, name_hint="prefill_hca_c128_kv_score_proj"):
        o0 = proj_idx * OUT_TILE
        kv_acc = pl.create_tensor([PREFILL_T, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([PREFILL_T, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, K_BLOCKS, stage=2):
            k0 = kb * K_TILE
            x_tile = x[0:PREFILL_T, k0 : k0 + K_TILE]
            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        kv_proj_out[0:PREFILL_T, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_out[0:PREFILL_T, o0 : o0 + OUT_TILE] = score_acc


@pl.jit.inline
def compressor_rmsnorm_rope(
    pooled_kv: pl.Tensor[[COMPRESSOR_RMS_ROWS, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos_b: pl.Tensor[[COMPRESSOR_RMS_ROWS, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[COMPRESSOR_RMS_ROWS, ROPE_HALF], pl.FP32],
    normed_kv: pl.Tensor[[COMPRESSOR_RMS_ROWS, HEAD_DIM], pl.FP32],
):
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    rows = pl.tensor.dim(pooled_kv, 0)
    for rt in pl.spmd(rows // RMS_ROW_TILE, name_hint="rmsnorm_rope"):
        r0 = rt * RMS_ROW_TILE
        partial_sq = pl.full([1, RMS_ROW_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv[r0 : r0 + RMS_ROW_TILE, rms_h0 : rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_ROW_TILE]))
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_ROW_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = rms_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv[r0 : r0 + RMS_ROW_TILE, norm_h0 : norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0 : norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[r0 : r0 + RMS_ROW_TILE, norm_h0 : norm_h0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[r0 : r0 + RMS_ROW_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_il = pl.gather(cos_b[r0 : r0 + RMS_ROW_TILE, 0:ROPE_HALF], dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b[r0 : r0 + RMS_ROW_TILE, 0:ROPE_HALF], dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv[r0 : r0 + RMS_ROW_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot

@pl.jit.inline
def compressor_core_ratio128(
    kv_proj: pl.Tensor[[CORE_PROJ_ROWS, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[CORE_PROJ_ROWS, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[CORE_TOKENS], pl.INT32],
    state_slot_mapping: pl.Tensor[[CORE_TOKENS], pl.INT64],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state_rows: pl.Tensor[[CORE_STATE_ROWS, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CORE_TABLE_ROWS, CORE_TABLE_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    write_pos_map: pl.Tensor[[1, CORE_WRITE_ROWS], pl.INT32],
    write_dst_map: pl.Tensor[[1, CORE_WRITE_ROWS], pl.INT32],
    state_table_row_map: pl.Tensor[[1, CORE_WRITE_ROWS], pl.INT32],
    pool_row_map: pl.Tensor[[1, CORE_POOL_ROWS], pl.INT32],
    cmp_kv_cache_flat: pl.Tensor[[CORE_CMP_ROWS, HEAD_DIM], pl.BF16],
    pooled_kv: pl.Tensor[[CORE_WRITE_ROWS, HEAD_DIM], pl.FP32],
    normed_kv: pl.Tensor[[CORE_WRITE_ROWS, HEAD_DIM], pl.FP32],
    cos_b: pl.Tensor[[CORE_WRITE_ROWS, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[CORE_WRITE_ROWS, ROPE_HALF], pl.FP32],
):
    """Shared decode/prefill ratio128 compression pipeline (mirrors
    _golden_compressor_ratio128_pipeline). Regime differences — write schedule,
    proj tiling, paged vs fresh state binding, per-token kv output — live in the
    caller; this is the batch-agnostic math core:

      scatter projected (kv, score+APE) into paged state (skip slot < 0)
        -> softmax-pool each complete window (block-table gather)
        -> rmsnorm + rope at the window position
        -> write the compressed row to cmp_kv_cache.

    On return `normed_kv` holds the per-write compressed rows (fp32), so a caller
    that also emits a per-token kv output (decode) scatters them from there."""
    token_rows = pl.tensor.dim(state_slot_mapping, 0)
    write_rows = pl.tensor.dim(write_dst_map, 1)

    # 1. Scatter projected (kv, score+APE) into the paged state buffer. Padding /
    # non-writing tokens carry state_slot_mapping < 0 and are skipped (uniform
    # decode/prefill validity contract; num_tokens is folded into the schedule).
    with pl.spmd(token_rows, name_hint="state_scatter_pre") as scatter_tid:
        scatter_t = pl.tile.get_block_idx()
        state_row_i64 = pl.read(state_slot_mapping, [scatter_t])
        if state_row_i64 >= 0:
            state_row = pl.cast(state_row_i64, target_type=pl.INDEX)
            token_pos = pl.read(position_ids, [scatter_t])
            token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
            ape_row = ape[token_ape_row : token_ape_row + 1, 0:OUT_DIM]
            kv_row = kv_proj[scatter_t : scatter_t + 1, 0:OUT_DIM]
            score_row = pl.add(score_proj[scatter_t : scatter_t + 1, 0:OUT_DIM], ape_row)
            compress_state_rows[state_row : state_row + 1, 0:OUT_DIM] = kv_row
            compress_state_rows[state_row : state_row + 1, OUT_DIM:COMPRESS_STATE_DIM] = score_row

    # 2. Zero the pooled scratch (invalid write rows must not feed rmsnorm garbage).
    # Coarse RMS_ROW_TILE-row full-width tiles: write_rows is a multiple of
    # RMS_ROW_TILE (rmsnorm tiles the same rows), so this covers every row exactly
    # while keeping the dispatch count tiny. Real rows are overwritten by the pool
    # below (ordered via init_tid dep), so zeroing them first is harmless.
    with pl.spmd(write_rows // RMS_ROW_TILE, name_hint="pooled_pad_init") as init_tid:
        init_r0 = pl.tile.get_block_idx() * RMS_ROW_TILE
        pooled_kv[init_r0 : init_r0 + RMS_ROW_TILE, 0:HEAD_DIM] = pl.full(
            [RMS_ROW_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0
        )

    # 3. Softmax-pool each complete window from the paged state. Iterate the compact
    # pool_row_map (one entry per real window) rather than every padded write row:
    # pool_row_map[p] -> padded write row, from which write_dst/write_pos/table_row
    # are read exactly as the padded grid would. Decode's scattered RMS padding thus
    # costs no extra pool tasks; prefill passes an identity map (no change).
    pool_rows = pl.tensor.dim(pool_row_map, 1)
    with pl.spmd(pool_rows * POOL_HEAD_BLOCKS_CORE, name_hint="softmax_pool", deps=[scatter_tid, init_tid]) as pool_tid:
        idx = pl.tile.get_block_idx()
        pool_p = idx // POOL_HEAD_BLOCKS_CORE
        h0 = (idx % POOL_HEAD_BLOCKS_CORE) * POOL_HEAD_TILE
        write_row = pl.cast(pl.read(pool_row_map, [0, pool_p]), target_type=pl.INDEX)
        write_slot_raw = pl.read(write_dst_map, [0, write_row])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_row])
            table_row = pl.cast(pl.read(state_table_row_map, [0, write_row]), target_type=pl.INDEX)
            pooled_chunk = compressor_ratio128_pool_window(
                compress_state_rows,
                compress_state_block_table,
                table_row,
                write_pos,
                h0,
            )
            pooled_kv[write_row : write_row + 1, h0 : h0 + POOL_HEAD_TILE] = pooled_chunk

    # 4. RoPE tables for each window position, then rmsnorm + rope.
    gather_compressor_rope_rows(
        freqs_cos,
        freqs_sin,
        write_pos_map,
        write_dst_map,
        pl.const(COMPRESS_RATIO, pl.INT32),
        cos_b,
        sin_b,
    )
    compressor_rmsnorm_rope(pooled_kv, norm_w, cos_b, sin_b, normed_kv)

    # 5. Write the compressed row to cmp_kv_cache. write_dst >= 0 and kv_out_row >= 0
    # are set together by the schedule, so gating on write_dst alone matches the
    # golden. A caller that also needs per-token kv reads normed_kv afterward.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_finalize", deps=[pool_tid]):
        for final_row in pl.range(write_rows):
            cmp_row_raw = pl.read(write_dst_map, [0, final_row])
            if cmp_row_raw >= 0:
                cmp_row = pl.cast(cmp_row_raw, target_type=pl.INDEX)
                kv_row = normed_kv[final_row : final_row + 1, 0:HEAD_DIM]
                cmp_kv_cache_flat[cmp_row : cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")


@pl.jit.inline
def decode_compressor_ratio128(
    x: pl.Tensor[[DECODE_T, D], pl.BF16],
    kv: pl.Tensor[[DECODE_T, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[DECODE_COMPRESS_STATE_BLOCK_NUM, DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[DECODE_B, DECODE_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv_cache: pl.Tensor[[DECODE_COMPRESSOR_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[DECODE_T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[DECODE_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[DECODE_T], pl.INT64],
):
    # Thin decode wrapper: build the padded per-batch write schedule, project
    # (dynamic-M tiling), then run the shared core with the paged decode state
    # and per-token kv output. See compressor_core_ratio128.
    write_pos_map = pl.create_tensor([1, DECODE_RMS_PAD_ROWS], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, DECODE_RMS_PAD_ROWS], dtype=pl.INT32)
    kv_out_row_map = pl.create_tensor([1, DECODE_RMS_PAD_ROWS], dtype=pl.INT32)
    state_table_row_map = pl.create_tensor([1, DECODE_RMS_PAD_ROWS], dtype=pl.INT32)
    build_decode_padded_write_schedule(
        position_ids,
        cmp_slot_mapping,
        pl.const(DECODE_S, pl.INT32),
        pl.const(COMPRESS_RATIO, pl.INT32),
        pl.const(DECODE_RMS_TILE, pl.INT32),
        pl.const(DECODE_RMS_PAD_TILE, pl.INT32),
        write_pos_map,
        write_dst_map,
        kv_out_row_map,
        state_table_row_map,
    )

    # Compact pool enumeration: one entry per batch row, mapping to its scattered
    # RMS-padded write row (pad_row = (b // RMS_TILE) * RMS_PAD_TILE + b % RMS_TILE,
    # the same placement build_decode_padded_write_schedule uses). The core pools
    # only these DECODE_B rows instead of all DECODE_RMS_PAD_ROWS padded rows.
    pool_row_map = pl.create_tensor([1, DECODE_B], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_pool_row_map"):
        for b in pl.range(DECODE_B):
            pad_row = (b // DECODE_RMS_TILE) * DECODE_RMS_PAD_TILE + (b % DECODE_RMS_TILE)
            pl.write(pool_row_map, [0, b], pl.cast(pad_row, pl.INT32))

    kv_proj_pad = pl.create_tensor([DECODE_BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([DECODE_BS_PAD, OUT_DIM], dtype=pl.FP32)
    compressor_ratio128_proj(x, wkv, wgate, kv_proj_pad, score_proj_pad)

    compress_state_rows = pl.reshape(
        compress_state,
        [DECODE_COMPRESS_STATE_BLOCK_NUM * DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
    )
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [DECODE_COMPRESSOR_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    pooled_kv = pl.create_tensor([DECODE_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([DECODE_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    cos_b = pl.create_tensor([DECODE_RMS_PAD_ROWS, ROPE_HALF], dtype=pl.FP32)
    sin_b = pl.create_tensor([DECODE_RMS_PAD_ROWS, ROPE_HALF], dtype=pl.FP32)
    compressor_core_ratio128(
        kv_proj_pad,
        score_proj_pad,
        position_ids,
        state_slot_mapping,
        ape,
        norm_w,
        compress_state_rows,
        compress_state_block_table,
        freqs_cos,
        freqs_sin,
        write_pos_map,
        write_dst_map,
        state_table_row_map,
        pool_row_map,
        cmp_kv_cache_flat,
        pooled_kv,
        normed_kv,
        cos_b,
        sin_b,
    )

    # Decode-only: scatter the compressed rows to the per-token kv output.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_kv_out"):
        for kv_write_row in pl.range(DECODE_RMS_PAD_ROWS):
            cmp_row_raw = pl.read(write_dst_map, [0, kv_write_row])
            if cmp_row_raw >= 0:
                kv_out_raw = pl.read(kv_out_row_map, [0, kv_write_row])
                if kv_out_raw >= 0:
                    kv_out_row = pl.cast(kv_out_raw, target_type=pl.INDEX)
                    kv[kv_out_row : kv_out_row + 1, :] = normed_kv[kv_write_row : kv_write_row + 1, 0:HEAD_DIM]
    return kv


@pl.jit
def decode_compressor_ratio128_test(
    x: pl.Tensor[[DECODE_T, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[DECODE_T, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[DECODE_COMPRESS_STATE_BLOCK_NUM, DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[DECODE_B, DECODE_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv_cache: pl.InOut[pl.Tensor[[DECODE_COMPRESSOR_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[DECODE_T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[DECODE_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[DECODE_T], pl.INT64],
):
    decode_compressor_ratio128(
        x, kv, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping,
    )
    return kv, compress_state, cmp_kv_cache


@pl.jit.inline
def prefill_compressor_ratio128(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    compress_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[PREFILL_B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[PREFILL_T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
):
    # Thin prefill wrapper: build the num_tokens-bounded write schedule, project
    # (static whole-M tiling), then run the shared core with the fresh prefill
    # state and no per-token kv output. num_tokens is consumed only by the
    # schedule builder; the core gates scatter on state_slot_mapping >= 0 (padding
    # is marked -1 by the host). See compressor_core_ratio128.
    write_pos_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    kv_out_row_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    state_table_row_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    build_prefill_write_schedule(
        position_ids,
        cmp_slot_mapping,
        num_tokens,
        write_pos_map,
        write_dst_map,
        kv_out_row_map,
        state_table_row_map,
    )

    # Prefill has no scattered RMS padding: its write rows are already compact, so
    # the pool enumeration is the identity over the write rows. The core then pools
    # exactly the same rows it did before this fix (behaviour unchanged for prefill).
    pool_row_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_pool_row_map"):
        for p in pl.range(HCA_C128_RMS_TILE):
            pl.write(pool_row_map, [0, p], pl.cast(p, pl.INT32))

    kv_proj_scratch = pl.create_tensor([PREFILL_T, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([PREFILL_T, OUT_DIM], dtype=pl.FP32)
    prefill_compressor_ratio128_proj(x, wkv, wgate, kv_proj_scratch, score_proj_scratch)

    compress_state_rows = pl.reshape(
        compress_state, [HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    pooled_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    cos_b = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, ROPE_HALF], dtype=pl.FP32)
    sin_b = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, ROPE_HALF], dtype=pl.FP32)
    compressor_core_ratio128(
        kv_proj_scratch,
        score_proj_scratch,
        position_ids,
        state_slot_mapping,
        ape,
        norm_w,
        compress_state_rows,
        compress_state_block_table,
        freqs_cos,
        freqs_sin,
        write_pos_map,
        write_dst_map,
        state_table_row_map,
        pool_row_map,
        cmp_kv_flat,
        pooled_kv_pad,
        normed_kv_pad,
        cos_b,
        sin_b,
    )
    return cmp_kv, compress_state


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[PREFILL_B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[PREFILL_T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
):
    return prefill_compressor_ratio128(
        x, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )


def _golden_compressor_ratio128_pipeline(
    tensors,
    *,
    write_pos_map,
    write_dst_map,
    kv_out_row_map,
    state_table_row_map,
    cmp_cache_name,
    kv_out_name=None,
):
    import torch

    x = tensors["x"].view(-1, D).float()
    position_ids = tensors["position_ids"].view(-1).to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].view(-1).to(torch.int64)
    compress_state = tensors["compress_state"]
    compress_state_rows = compress_state.view(-1, COMPRESS_STATE_DIM)
    compress_state_block_table = tensors["compress_state_block_table"]
    cmp_kv_cache = tensors[cmp_cache_name]
    cmp_kv_cache_flat = cmp_kv_cache.view(cmp_kv_cache.shape[0] * BLOCK_SIZE, HEAD_DIM)

    kv_proj = x @ tensors["wkv"].float().t()
    score_proj = x @ tensors["wgate"].float().t()
    ape = tensors["ape"]
    for token_id in range(x.shape[0]):
        dst_row = int(state_slot_mapping[token_id].item())
        if dst_row < 0:
            continue
        pos = int(position_ids[token_id].item())
        ape_slot = pos % COMPRESS_RATIO
        compress_state_rows[dst_row, 0:OUT_DIM] = kv_proj[token_id]
        compress_state_rows[dst_row, OUT_DIM:COMPRESS_STATE_DIM] = score_proj[token_id] + ape[ape_slot]

    def rmsnorm(x, w):
        var = x.square().mean(-1, keepdim=True)
        return x * torch.rsqrt(var + EPS) * w.float().view(1, HEAD_DIM)

    state_block_size = compress_state.shape[1]
    num_state_blocks = STATE_LEN // state_block_size
    kv_out = tensors[kv_out_name] if kv_out_name is not None else None
    for write_i, dst_row in enumerate(write_dst_map):
        if dst_row < 0:
            continue
        kv_out_row = kv_out_row_map[write_i]
        if kv_out_row < 0:
            continue
        state_table_row = state_table_row_map[write_i]
        if state_table_row < 0:
            continue
        write_pos = write_pos_map[write_i]
        state_pos0 = write_pos + 1 - COMPRESS_RATIO
        base_logical_blk = state_pos0 // state_block_size
        pool_kv_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32, device=x.device)
        pool_score_state = torch.full((STATE_LEN, OUT_DIM), float("-inf"), dtype=torch.float32, device=x.device)
        for blk_i in range(num_state_blocks):
            logical_blk = base_logical_blk + blk_i
            if logical_blk < 0 or logical_blk >= compress_state_block_table.shape[1]:
                continue
            state_blk = int(compress_state_block_table[state_table_row, logical_blk].item())
            if state_blk < 0:
                continue
            row0 = state_blk * state_block_size
            s0 = blk_i * state_block_size
            pool_kv_state[s0 : s0 + state_block_size] = compress_state_rows[row0 : row0 + state_block_size, 0:OUT_DIM]
            pool_score_state[s0 : s0 + state_block_size] = compress_state_rows[
                row0 : row0 + state_block_size, OUT_DIM:COMPRESS_STATE_DIM
            ]

        pooled = (pool_kv_state * pool_score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        normed = rmsnorm(pooled, tensors["norm_w"])
        rope_pair = normed[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even = rope_pair[..., 0].float()
        odd = rope_pair[..., 1].float()
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)

        if kv_out is not None:
            kv_out[kv_out_row : kv_out_row + 1, :] = normed.reshape(1, HEAD_DIM)
        cmp_kv_cache_flat[dst_row] = normed[0]

    tensors["compress_state"][:] = compress_state_rows.view_as(compress_state)
    tensors[cmp_cache_name][:] = cmp_kv_cache_flat.view_as(cmp_kv_cache)


def golden_decode_compressor_ratio128(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap)."""
    position_ids = tensors["position_ids"].view(-1).to("cpu")
    cmp_slot_mapping = tensors["cmp_slot_mapping"].view(-1).to("cpu")
    write_pos_map = [0] * DECODE_RMS_PAD_ROWS
    write_dst_map = [-1] * DECODE_RMS_PAD_ROWS
    kv_out_row_map = [-1] * DECODE_RMS_PAD_ROWS
    state_table_row_map = [-1] * DECODE_RMS_PAD_ROWS
    for b in range(DECODE_B):
        base_t = b * DECODE_S
        first_pos = int(position_ids[base_t].item())
        pos_in_window = first_pos % COMPRESS_RATIO
        if pos_in_window + DECODE_S >= COMPRESS_RATIO:
            boundary_s = COMPRESS_RATIO - 1 - pos_in_window
            token_t = base_t + boundary_s
            dst_row = int(cmp_slot_mapping[token_t].item())
            if dst_row >= 0:
                pad_row = (b // DECODE_RMS_TILE) * DECODE_RMS_PAD_TILE + (b % DECODE_RMS_TILE)
                if pad_row < DECODE_RMS_PAD_ROWS:
                    write_pos_map[pad_row] = first_pos + boundary_s
                    write_dst_map[pad_row] = dst_row
                    kv_out_row_map[pad_row] = base_t
                    state_table_row_map[pad_row] = b

    _golden_compressor_ratio128_pipeline(
        tensors,
        write_pos_map=write_pos_map,
        write_dst_map=write_dst_map,
        kv_out_row_map=kv_out_row_map,
        state_table_row_map=state_table_row_map,
        cmp_cache_name="cmp_kv_cache",
        kv_out_name="kv",
    )


def golden_prefill_compressor_ratio128(tensors):
    num_tokens = int(tensors["num_tokens"])
    position_ids = tensors["position_ids"].view(-1).to("cpu")
    cmp_slot_mapping = tensors["cmp_slot_mapping"].view(-1).to("cpu")
    write_pos_map = [0] * HCA_C128_RMS_TILE
    write_dst_map = [-1] * HCA_C128_RMS_TILE
    kv_out_row_map = [-1] * HCA_C128_RMS_TILE
    state_table_row_map = [-1] * HCA_C128_RMS_TILE
    write_i = 0
    for token_id in range(position_ids.numel()):
        if token_id >= num_tokens:
            break
        dst_row = int(cmp_slot_mapping[token_id].item())
        if dst_row < 0:
            continue
        if write_i >= HCA_C128_RMS_TILE:
            break
        write_pos_map[write_i] = int(position_ids[token_id].item())
        write_dst_map[write_i] = dst_row
        kv_out_row_map[write_i] = token_id
        state_table_row_map[write_i] = 0
        write_i += 1

    _golden_compressor_ratio128_pipeline(
        tensors,
        write_pos_map=write_pos_map,
        write_dst_map=write_dst_map,
        kv_out_row_map=kv_out_row_map,
        state_table_row_map=state_table_row_map,
        cmp_cache_name="cmp_kv",
    )


def build_decode_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from decode_metadata import (
        block_table,
        compressed_slot_mapping,
        hca_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(DECODE_B, DECODE_S, D).reshape(DECODE_T, D)
    def init_compress_state():
        return torch.zeros(DECODE_COMPRESS_STATE_BLOCK_NUM, DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    # Calibrated to the real DeepSeek-V4-Flash 150
    #  (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0246
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0316
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0340
    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv_cache():
        return torch.zeros(DECODE_COMPRESSOR_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_compress_state_block_table():
        return block_table(
            batch=DECODE_B,
            table_blocks=DECODE_COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS,
            permuted=True,
        )
    def init_cmp_block_table():
        return block_table(
            batch=DECODE_B,
            table_blocks=DECODE_COMPRESSOR_CMP_MAX_BLOCKS,
            physical_blocks=DECODE_COMPRESSOR_CMP_MAX_BLOCKS,
            permuted=True,
        )
    def init_default_start_pos():
        # Canonical HCA start-position set (ratio-128 compressor branches + 8k long-context).
        return hca_decode_start_set(
            batch=DECODE_B, compress_ratio=COMPRESS_RATIO, state_block_size=DECODE_COMPRESS_STATE_BLOCK_SIZE)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=DECODE_B,
            seq=DECODE_S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def _position_ids_bs():
        # [DECODE_B, DECODE_S] positions; token-major init_position_ids reshapes this to [DECODE_T].
        return position_ids_from_starts(init_start_pos(), seq=DECODE_S)
    def init_position_ids():
        # token-major [DECODE_T]; row t == (b * DECODE_S + s) carries position of sequence b, intra-token s
        return _position_ids_bs().reshape(DECODE_T)
    def init_state_slot_mapping():
        return state_slot_mapping(
            _position_ids_bs(),
            init_compress_state_block_table(),
            state_block_size=DECODE_COMPRESS_STATE_BLOCK_SIZE,
        ).reshape(DECODE_T)
    def init_cmp_slot_mapping():
        return compressed_slot_mapping(
            _position_ids_bs(),
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(DECODE_T)
    return [
        TensorSpec("x", [DECODE_T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [DECODE_T, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state", [DECODE_COMPRESS_STATE_BLOCK_NUM, DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [DECODE_B, DECODE_COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv_cache", [DECODE_COMPRESSOR_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache, is_output=True),
        TensorSpec("position_ids", [DECODE_T], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [DECODE_T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [DECODE_T], torch.int64, init_value=init_state_slot_mapping),
    ]


def build_prefill_tensor_specs(start_pos: int = PREFILL_START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    num_tokens = PREFILL_T
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + num_tokens > MAX_SEQ_LEN:
        raise ValueError("start_pos + num_tokens exceeds max_position_embeddings")

    def init_compress_state_block_table():
        table = torch.full((PREFILL_B, HCA_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for block in range(HCA_STATE_MAX_BLOCKS):
            table[0, block] = (block * 17 + 3) % HCA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[0, block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(PREFILL_T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row, 0:OUT_DIM] = (torch.rand(OUT_DIM) - 0.5) * 0.05
                flat[row, OUT_DIM:COMPRESS_STATE_DIM] = (torch.rand(OUT_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash HCA (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors the decode path.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0246
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0316
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0340
    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + PREFILL_T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((PREFILL_T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((PREFILL_T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            mapping[token_id] = state_row(start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [PREFILL_T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("compress_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [PREFILL_B, HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("position_ids", [PREFILL_T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("cmp_slot_mapping", [PREFILL_T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [PREFILL_T], torch.int64, init_value=init_state_slot_mapping),
    ]


def _run_decode_validation(args):
    from golden import ratio_allclose, run_jit

    return run_jit(
        fn=decode_compressor_ratio128_test,
        specs=build_decode_tensor_specs(args.start_pos),
        golden_fn=golden_decode_compressor_ratio128,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compile_only=args.compile_only,
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )


def _run_prefill_validation(args):
    from golden import ratio_allclose, run_jit

    start_pos = PREFILL_START_POS if args.start_pos is None else args.start_pos
    return run_jit(
        fn=prefill_compressor_ratio128_test,
        specs=build_prefill_tensor_specs(start_pos),
        golden_fn=golden_prefill_compressor_ratio128,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
        },
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 compressor ratio128 validation.")
    parser.add_argument("--mode", choices=["decode", "prefill", "both"], default="both")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--start-pos",
        type=int,
        default=None,
        help="Fixture-only start position override. Decode defaults to its canonical batch set; prefill defaults to 0.",
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes = ("decode", "prefill") if args.mode == "both" else (args.mode,)
    for mode in modes:
        result = _run_decode_validation(args) if mode == "decode" else _run_prefill_validation(args)
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
