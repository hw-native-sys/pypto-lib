# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 ratio-4 compressor decode and prefill paths."""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    FP32_NEG_INF,
    PREFILL_CMP_BLOCK_NUM,
)
from compressor_common import (
    build_prefill_write_schedule,
    compressor_rmsnorm_rope,
    finalize_compressor_writes,
    gather_compressor_rope_rows,
)


EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM
POOL_HEAD_TILE = HEAD_DIM
RATIO4_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

# Shared ratio4 sub-kernel tiling.
PROJ_ROWS = pl.dynamic("COMPRESSOR4_PROJ_ROWS")
PROJ_ROWS_PAD = pl.dynamic("COMPRESSOR4_PROJ_ROWS_PAD")
PROJ_MM_B_TILE = 16
PROJ_OUT_TILE = 64
PROJ_K_TILE = 512
POOL_STATE_ROWS = pl.dynamic("COMPRESSOR4_POOL_STATE_ROWS")
POOL_TABLE_ROWS = pl.dynamic("COMPRESSOR4_POOL_TABLE_ROWS")
POOL_TABLE_BLOCKS = pl.dynamic("COMPRESSOR4_POOL_TABLE_BLOCKS")


@pl.jit.inline
def compressor_ratio4_proj(
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
def compressor_ratio4_pool_math(
    score_state: pl.Tensor[[STATE_LEN, POOL_HEAD_TILE], pl.FP32],
    kv_state: pl.Tensor[[STATE_LEN, POOL_HEAD_TILE], pl.FP32],
):
    init_slot = STATE_LEN - 1
    mi_buf = pl.create_tensor([1, POOL_HEAD_TILE], dtype=pl.FP32)
    li_buf = pl.create_tensor([1, POOL_HEAD_TILE], dtype=pl.FP32)
    oi_buf = pl.create_tensor([1, POOL_HEAD_TILE], dtype=pl.FP32)
    mi_buf[0:1, 0:POOL_HEAD_TILE] = score_state[init_slot : init_slot + 1, 0 : POOL_HEAD_TILE]
    li_buf[0:1, 0:POOL_HEAD_TILE] = pl.exp(pl.sub(mi_buf[0:1, 0:POOL_HEAD_TILE], mi_buf[0:1, 0:POOL_HEAD_TILE]))
    oi_buf[0:1, 0:POOL_HEAD_TILE] = kv_state[init_slot : init_slot + 1, 0 : POOL_HEAD_TILE]
    for slot_i in pl.range(STATE_LEN - 1):
        mi = mi_buf[0:1, 0:POOL_HEAD_TILE]
        li = li_buf[0:1, 0:POOL_HEAD_TILE]
        oi = oi_buf[0:1, 0:POOL_HEAD_TILE]
        slot_score = score_state[slot_i : slot_i + 1, 0 : POOL_HEAD_TILE]
        slot_kv = kv_state[slot_i : slot_i + 1, 0 : POOL_HEAD_TILE]
        mi_next = pl.maximum(mi, slot_score)
        alpha = pl.exp(pl.sub(mi, mi_next))
        beta = pl.exp(pl.sub(slot_score, mi_next))
        li_buf[0:1, 0:POOL_HEAD_TILE] = pl.add(pl.mul(alpha, li), beta)
        oi_buf[0:1, 0:POOL_HEAD_TILE] = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
        mi_buf[0:1, 0:POOL_HEAD_TILE] = mi_next
    return pl.div(oi_buf[0:1, 0:POOL_HEAD_TILE], li_buf[0:1, 0:POOL_HEAD_TILE])


@pl.jit.inline
def compressor_ratio4_pool_window(
    compress_state_rows: pl.Tensor[[POOL_STATE_ROWS, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[POOL_TABLE_ROWS, POOL_TABLE_BLOCKS], pl.INT32],
    table_row: pl.Scalar[pl.INDEX],
    write_pos: pl.Scalar[pl.INT32],
    h0: pl.Scalar[pl.INDEX],
):
    pool_score_tile = pl.create_tensor([STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
    pool_kv_tile = pl.create_tensor([STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
    cur_start = write_pos + 1 - COMPRESS_RATIO
    prev_start = cur_start - COMPRESS_RATIO

    if write_pos >= 2 * COMPRESS_RATIO - 1:
        for pool_s in pl.range(COMPRESS_RATIO):
            prev_abs = prev_start + pool_s
            prev_state_block = pl.cast(prev_abs // RATIO4_STATE_BLOCK_SIZE, pl.INDEX)
            prev_state_intra = pl.cast(prev_abs - prev_state_block * RATIO4_STATE_BLOCK_SIZE, pl.INDEX)
            prev_phys_block_raw = pl.read(compress_state_block_table, [table_row, prev_state_block])
            if prev_phys_block_raw >= 0:
                prev_phys_block = pl.cast(prev_phys_block_raw, pl.INDEX)
                prev_state_row = prev_phys_block * RATIO4_STATE_BLOCK_SIZE + prev_state_intra
                pool_kv_tile[pool_s : pool_s + 1, 0:POOL_HEAD_TILE] = compress_state_rows[
                    prev_state_row : prev_state_row + 1,
                    h0 : h0 + POOL_HEAD_TILE,
                ]
                pool_score_tile[pool_s : pool_s + 1, 0:POOL_HEAD_TILE] = compress_state_rows[
                    prev_state_row : prev_state_row + 1,
                    OUT_DIM + h0 : OUT_DIM + h0 + POOL_HEAD_TILE,
                ]
            else:
                pool_kv_tile[pool_s : pool_s + 1, 0:POOL_HEAD_TILE] = pl.full(
                    [1, POOL_HEAD_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[pool_s : pool_s + 1, 0:POOL_HEAD_TILE] = pl.full(
                    [1, POOL_HEAD_TILE],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
    else:
        pool_kv_tile[0:COMPRESS_RATIO, 0:POOL_HEAD_TILE] = pl.full(
            [COMPRESS_RATIO, POOL_HEAD_TILE],
            dtype=pl.FP32,
            value=0.0,
        )
        pool_score_tile[0:COMPRESS_RATIO, 0:POOL_HEAD_TILE] = pl.full(
            [COMPRESS_RATIO, POOL_HEAD_TILE],
            dtype=pl.FP32,
            value=FP32_NEG_INF,
        )

    for pool_s in pl.range(COMPRESS_RATIO):
        cur_abs = cur_start + pool_s
        back_slot = COMPRESS_RATIO + pool_s
        cur_state_block = pl.cast(cur_abs // RATIO4_STATE_BLOCK_SIZE, pl.INDEX)
        cur_state_intra = pl.cast(cur_abs - cur_state_block * RATIO4_STATE_BLOCK_SIZE, pl.INDEX)
        cur_phys_block_raw = pl.read(compress_state_block_table, [table_row, cur_state_block])
        if cur_phys_block_raw >= 0:
            cur_phys_block = pl.cast(cur_phys_block_raw, pl.INDEX)
            cur_state_row = cur_phys_block * RATIO4_STATE_BLOCK_SIZE + cur_state_intra
            pool_kv_tile[back_slot : back_slot + 1, 0:POOL_HEAD_TILE] = compress_state_rows[
                cur_state_row : cur_state_row + 1,
                HEAD_DIM + h0 : HEAD_DIM + h0 + POOL_HEAD_TILE,
            ]
            pool_score_tile[back_slot : back_slot + 1, 0:POOL_HEAD_TILE] = compress_state_rows[
                cur_state_row : cur_state_row + 1,
                OUT_DIM + HEAD_DIM + h0 : OUT_DIM + HEAD_DIM + h0 + POOL_HEAD_TILE,
            ]
        else:
            pool_kv_tile[back_slot : back_slot + 1, 0:POOL_HEAD_TILE] = pl.full(
                [1, POOL_HEAD_TILE],
                dtype=pl.FP32,
                value=0.0,
            )
            pool_score_tile[back_slot : back_slot + 1, 0:POOL_HEAD_TILE] = pl.full(
                [1, POOL_HEAD_TILE],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )

    return compressor_ratio4_pool_math(pool_score_tile, pool_kv_tile)


# Shared-core dynamic shapes (bind per caller: decode vs prefill). State and
# cmp-cache tensors are passed as pre-reshaped flat views so their shapes stay
# statically inferable per call site.
CORE_PROJ_ROWS = pl.dynamic("COMPRESSOR4_CORE_PROJ_ROWS")
CORE_TOKENS = pl.dynamic("COMPRESSOR4_CORE_TOKENS")
CORE_WRITE_ROWS = pl.dynamic("COMPRESSOR4_CORE_WRITE_ROWS")
CORE_STATE_ROWS = pl.dynamic("COMPRESSOR4_CORE_STATE_ROWS")
CORE_TABLE_ROWS = pl.dynamic("COMPRESSOR4_CORE_TABLE_ROWS")
CORE_TABLE_BLOCKS = pl.dynamic("COMPRESSOR4_CORE_TABLE_BLOCKS")
# Compact per-regime pool enumeration: decode binds it to the batch-row count
# (real windows scattered across the RMS-padded schedule), prefill to its compact
# write-row count. Pooling only real windows keeps decode off the padded grid.
CORE_POOL_ROWS = pl.dynamic("COMPRESSOR4_CORE_POOL_ROWS")
POOL_HEAD_BLOCKS_CORE = HEAD_DIM // POOL_HEAD_TILE  # 1: ratio4 pools the whole head
CORE_INIT_ROW_TILE = 8  # divides both regimes' write_rows (== shared rmsnorm tile)
ROPE_HALF = ROPE_HEAD_DIM // 2


@pl.jit.inline
def compressor_core_ratio4(
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
    pooled_kv: pl.Tensor[[CORE_WRITE_ROWS, HEAD_DIM], pl.FP32],
    normed_kv: pl.Tensor[[CORE_WRITE_ROWS, HEAD_DIM], pl.FP32],
    cos_b: pl.Tensor[[CORE_WRITE_ROWS, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[CORE_WRITE_ROWS, ROPE_HALF], pl.FP32],
):
    """Prefill ratio4 compression math core (mirrors compressor_core_ratio128,
    adapted for the overlap window: COFF=2 state, STATE_LEN=8 online-softmax pool
    over the whole head, POOL_HEAD_BLOCKS=1).

      scatter projected (kv, score+APE) into paged state (skip slot < 0)
        -> softmax-pool each real window (compact pool_row_map, block-table gather)
        -> rmsnorm + rope at the window position.

    Stops at normed_kv; the prefill wrapper finalizes cmp_kv + keepalive. Decode no
    longer shares this core: it fuses scatter+pool and rmsnorm+rope+cache-write into
    two tasks over a single paged 16-row block (see decode_compressor_ratio4), which
    a packed multi-write prefill schedule cannot reuse."""
    token_rows = pl.tensor.dim(state_slot_mapping, 0)
    write_rows = pl.tensor.dim(write_dst_map, 1)
    pool_rows = pl.tensor.dim(pool_row_map, 1)

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

    # 2. Zero the pooled scratch coarsely (real rows overwritten by the pool via
    # the init_tid dep). write_rows is a multiple of CORE_INIT_ROW_TILE.
    with pl.spmd(write_rows // CORE_INIT_ROW_TILE, name_hint="pooled_pad_init") as init_tid:
        init_r0 = pl.tile.get_block_idx() * CORE_INIT_ROW_TILE
        pooled_kv[init_r0 : init_r0 + CORE_INIT_ROW_TILE, 0:HEAD_DIM] = pl.full(
            [CORE_INIT_ROW_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0
        )

    # 3. Softmax-pool each real window from the paged state. Iterate the compact
    # pool_row_map (one entry per real window) -> padded write row; decode's
    # scattered RMS padding thus costs no extra pool tasks, prefill passes identity.
    with pl.spmd(pool_rows * POOL_HEAD_BLOCKS_CORE, name_hint="softmax_pool", deps=[scatter_tid, init_tid]) as pool_tid:
        idx = pl.tile.get_block_idx()
        pool_p = idx // POOL_HEAD_BLOCKS_CORE
        h0 = (idx % POOL_HEAD_BLOCKS_CORE) * POOL_HEAD_TILE
        write_row = pl.cast(pl.read(pool_row_map, [0, pool_p]), target_type=pl.INDEX)
        write_slot_raw = pl.read(write_dst_map, [0, write_row])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_row])
            table_row = pl.cast(pl.read(state_table_row_map, [0, write_row]), target_type=pl.INDEX)
            pooled_chunk = compressor_ratio4_pool_window(
                compress_state_rows,
                compress_state_block_table,
                table_row,
                write_pos,
                h0,
            )
            pooled_kv[write_row : write_row + 1, h0 : h0 + POOL_HEAD_TILE] = pooled_chunk

    # 4. RoPE tables for each window position, then rmsnorm + rope -> normed_kv.
    gather_compressor_rope_rows(
        freqs_cos,
        freqs_sin,
        write_pos_map,
        write_dst_map,
        pl.const(COMPRESS_RATIO, pl.INT32),
        cos_b,
        sin_b,
    )
    normed_kv = compressor_rmsnorm_rope(pooled_kv, norm_w, cos_b, sin_b, normed_kv)
    return normed_kv


# Decode shape and paging contract.
DECODE_B = DECODE_BATCH
DECODE_S = DECODE_SEQ
DECODE_T = DECODE_B * DECODE_S
DECODE_IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
DECODE_COMPRESS_STATE_BLOCK_SIZE = RATIO4_STATE_BLOCK_SIZE
DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS = 65
DECODE_COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + DECODE_COMPRESS_STATE_BLOCK_SIZE - 1) // DECODE_COMPRESS_STATE_BLOCK_SIZE
DECODE_COMPRESS_STATE_BLOCK_NUM = DECODE_B * DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS
DECODE_COMPRESSOR_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
DECODE_COMPRESSOR_CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# Decode tiling.
DECODE_ROPE_TILE = 32
DECODE_K_TILE = 512
DECODE_OUT_TILE = 64
DECODE_B_TILE = 8
DECODE_MM_B_TILE = 16
DECODE_BS_PAD = ((DECODE_B * DECODE_S + DECODE_MM_B_TILE - 1) // DECODE_MM_B_TILE) * DECODE_MM_B_TILE
DECODE_HEAD_TILE = 64
DECODE_HEAD_DIM_TILE = 128
DECODE_RMS_PAD_TILE = 16  # pad DECODE_B rows into one 16-row block (min M for FP32 vec ops)
DECODE_RMS_PAD_ROWS = DECODE_RMS_PAD_TILE  # single block; requires DECODE_B <= RMS_PAD_TILE
assert DECODE_B <= DECODE_RMS_PAD_TILE

@pl.jit.inline
def decode_compressor_ratio4(
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
    # Decode fuses the paged CSA compressor into two tasks (upstream #734 single-block
    # form). scatter_softmax_pool computes each batch's overlap window from its own
    # just-scattered paged state (per-batch block table, so no cross-task barrier);
    # rmsnorm_rope_cache_write normalizes the single 16-row block and writes the
    # per-token kv output + paged cmp_kv_cache. Prefill keeps the multi-write shared
    # core (compressor_core_ratio4): a single paged block and packed multi-writes
    # diverge too much to share the fused path. Reads stay token-major ([T]).
    kv_flat = kv
    compress_state_flat = pl.reshape(compress_state, [DECODE_COMPRESS_STATE_BLOCK_NUM * DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [DECODE_COMPRESSOR_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    cmp4_kv_proj_pad = pl.create_tensor([DECODE_BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([DECODE_BS_PAD, OUT_DIM], dtype=pl.FP32)
    compressor_ratio4_proj(x, wkv, wgate, cmp4_kv_proj_pad, cmp4_score_proj_pad)

    # scatter_softmax_pool: per batch, scatter the padded proj rows into the paged
    # compress_state, then online-softmax pool that batch's window into pooled_kv.
    # One region -- each batch's pool reads only its own just-scattered state.
    pooled_kv = pl.create_tensor([DECODE_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool"):
        for c_idx in pl.range(DECODE_B):
            for s_sc in pl.pipeline(DECODE_S, stage=2):
                proj_row = c_idx * DECODE_S + s_sc
                token_pos = pl.read(position_ids, [proj_row])
                state_row_i64 = pl.read(state_slot_mapping, [proj_row])
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, pl.INDEX)
                    kv_tile = cmp4_kv_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    score_tile = cmp4_score_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    ape_tile = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    score_tile = pl.add(score_tile, ape_tile)
                    compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = kv_tile
                    compress_state_flat[state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM] = score_tile

            pad_idx = c_idx
            first_pos_b = pl.read(position_ids, [c_idx * DECODE_S])
            pos_b = first_pos_b % COMPRESS_RATIO
            pre_tokens_b = COMPRESS_RATIO - pos_b
            boundary_end_b = first_pos_b + pre_tokens_b - 1
            cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
            prev_window_start_b = cur_window_start_b - COMPRESS_RATIO

            if pos_b + DECODE_S >= COMPRESS_RATIO:
                # Head-chunk loop collapsed to one [1, HEAD_DIM] tile: the online
                # softmax is per-column elementwise, so widening is bit-identical.
                last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                last_blk_off = last_abs // DECODE_COMPRESS_STATE_BLOCK_SIZE
                last_intra = last_abs % DECODE_COMPRESS_STATE_BLOCK_SIZE
                last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
                last_row = last_blk_id * DECODE_COMPRESS_STATE_BLOCK_SIZE + last_intra
                mi = compress_state_flat[last_row : last_row + 1, OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM : OUT_DIM]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    front_score = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=FP32_NEG_INF)
                    front_kv = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    if first_pos_b >= COMPRESS_RATIO:
                        prev_blk_off = prev_abs // DECODE_COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % DECODE_COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX)
                        prev_row = prev_blk_id * DECODE_COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM : OUT_DIM + HEAD_DIM]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, 0 : HEAD_DIM]
                    mi_next_front = pl.maximum(mi, front_score)
                    alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                    beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                    li = pl.add(pl.mul(alpha_front, li), beta_front)
                    oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                    mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // DECODE_COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % DECODE_COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, cur_blk_off]), pl.INDEX)
                    cur_row = cur_blk_id * DECODE_COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_score = compress_state_flat[cur_row : cur_row + 1, OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM]
                    back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM : OUT_DIM]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv[pad_idx : pad_idx + 1, 0 : HEAD_DIM] = pooled_chunk

    # rmsnorm_rope_cache_write: normalize + rope the single 16-row block, then write
    # the per-token kv output + paged cmp_kv_cache. The cache write reads back only
    # this block's own normed_kv rows (intra-block RAW) -- no separate scope needed.
    normed_kv = pl.create_tensor([DECODE_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope_cache_write"):
        # single 16-row block: DECODE_B real rows at 0..DECODE_B-1, rest are pad.
        # In-kernel token-major gather of each batch's compressor rope row.
        cos_b = pl.full([DECODE_RMS_PAD_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([DECODE_RMS_PAD_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        for inner in pl.range(DECODE_B):
            first_pos_b = pl.read(position_ids, [inner * DECODE_S])
            cmp_pos_b = pl.cast(first_pos_b - (first_pos_b % COMPRESS_RATIO), pl.INDEX)
            cos_b[inner : inner + 1, 0 : ROPE_HALF] = pl.cast(
                freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HALF], target_type=pl.FP32)
            sin_b[inner : inner + 1, 0 : ROPE_HALF] = pl.cast(
                freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HALF], target_type=pl.FP32)

        partial_sq = pl.full([1, DECODE_RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, DECODE_HEAD_TILE):
            kv_rms_chunk = pooled_kv[0 : DECODE_RMS_PAD_TILE, k0 : k0 + DECODE_HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, DECODE_RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [DECODE_RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, DECODE_HEAD_TILE):
            kv_norm_chunk = pooled_kv[0 : DECODE_RMS_PAD_TILE, k0 : k0 + DECODE_HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + DECODE_HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0 : DECODE_RMS_PAD_TILE, k0 : k0 + DECODE_HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[0 : DECODE_RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather rope (swap/sign/dup indices built in-kernel from
        # pl.arange): out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]. normed_kv is
        # FP32 so rope_rot is written directly.
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([DECODE_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv[0 : DECODE_RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_rot

        for inner in pl.range(DECODE_B):
            c_idx = inner
            first_pos_b = pl.read(position_ids, [c_idx * DECODE_S])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + DECODE_S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row_fp32 = normed_kv[inner : inner + 1, 0 : HEAD_DIM]
                cache_row_i64 = pl.read(cmp_slot_mapping, [c_idx * DECODE_S + boundary_s])
                if cache_row_i64 >= 0:
                    cache_row = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat[c_idx * DECODE_S : c_idx * DECODE_S + 1, :] = kv_row_fp32
                    cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint")

    return kv_flat


@pl.jit
def decode_compressor_ratio4_test(
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
    decode_compressor_ratio4(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        freqs_cos,
        freqs_sin,
        cmp_kv_cache,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
    )
    return kv, compress_state, cmp_kv_cache


def golden_decode_compressor_ratio4(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    x = tensors["x"].float().reshape(DECODE_B, DECODE_S, D)
    compress_state = tensors["compress_state"]
    compress_state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
    position_ids = tensors["position_ids"].reshape(DECODE_B, DECODE_S).to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].reshape(DECODE_B, DECODE_S).to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].reshape(DECODE_B, DECODE_S).to(torch.int64)
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv.t()                    # [DECODE_B, DECODE_S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = x @ wgate.t()               # [DECODE_B, DECODE_S, OUT_DIM]

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    def read_front_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // DECODE_COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % DECODE_COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, :HEAD_DIM],
            compress_state[blk_id, intra, OUT_DIM:OUT_DIM + HEAD_DIM],
        )

    def read_back_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // DECODE_COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % DECODE_COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, HEAD_DIM:OUT_DIM],
            compress_state[blk_id, intra, OUT_DIM + HEAD_DIM:],
        )

    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        pre_tokens = min(DECODE_S, ratio - (first_pos % ratio))
        boundary_s = ratio - 1 - (first_pos % ratio)
        should_compress = 0 <= boundary_s < DECODE_S
        boundary_end = first_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        # Per-token ape add + state scatter through explicit token-major slots.
        for s in range(DECODE_S):
            pos = int(position_ids[b, s].item())
            token_ape_row = pos % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            state_row = int(state_slot_mapping[b, s].item())
            if state_row >= 0:
                blk_id = state_row // DECODE_COMPRESS_STATE_BLOCK_SIZE
                intra = state_row % DECODE_COMPRESS_STATE_BLOCK_SIZE
                compress_state[blk_id, intra, :OUT_DIM] = kv[b, s, :]
                compress_state[blk_id, intra, OUT_DIM:] = score[b, s, :]

        if should_compress:
            should_compress_rows[b] = True
            kv_rows = []
            score_rows = []
            for s in range(ratio):
                abs_pos = prev_window_start + s
                if abs_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device))
                    continue
                kv_row, score_row = read_front_state(b, abs_pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            for s in range(ratio):
                abs_pos = cur_window_start + s
                kv_row, score_row = read_back_state(b, abs_pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kvs = torch.stack(kv_rows, dim=0).unsqueeze(0)
            scs = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)

    tensors["compress_state"][:] = compress_state

    if not bool(should_compress_rows.any()):
        return

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        if not bool(should_compress_rows[b]):
            continue
        first_pos = int(position_ids[b, 0].item())
        boundary_s = ratio - 1 - (first_pos % ratio)
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        # cos/sin from the shared freqs table at the compression-window origin, matching
        # the in-kernel gather (window_start = first_pos - first_pos % ratio). freqs_* is
        # BF16; .float() replicates the kernel's BF16->FP32 cast.
        window_start_b = first_pos - (first_pos % ratio)
        cos_v = freqs_cos[window_start_b, : rd // 2].float().view(-1)
        sin_v = freqs_sin[window_start_b, : rd // 2].float().view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            # Kernel writes the committed pooled result to the sequence's first
            # token row (t = b * DECODE_S); non-boundary batches and other token rows
            # stay at the output tensor's zero-init.
            tensors["kv"][b * DECODE_S : b * DECODE_S + 1, :] = kv_b.reshape(1, HEAD_DIM)
            blk_id = cmp_row // BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_decode_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from decode_metadata import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
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
        state = torch.zeros(DECODE_COMPRESS_STATE_BLOCK_NUM, DECODE_COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        state[:, :, OUT_DIM:] = FP32_NEG_INF
        return state
    def init_compress_state_block_table():
        return block_table(
            batch=DECODE_B,
            table_blocks=DECODE_COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=DECODE_COMPRESS_STATE_PHYSICAL_BLOCKS,
        )
    # Calibrated to the real DeepSeek-V4-Flash CSA (ratio-4) main compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0245
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0388
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1243
    def init_norm_w():
        return 0.9666 + 0.1929 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv_cache():
        return torch.zeros(DECODE_COMPRESSOR_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_cmp_block_table():
        tbl = torch.full((DECODE_B, DECODE_COMPRESSOR_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(DECODE_B):
            for j in range(DECODE_COMPRESSOR_CMP_MAX_BLOCKS):
                tbl[b, j] = b * DECODE_COMPRESSOR_CMP_MAX_BLOCKS + j
        return tbl
    def init_default_start_pos():
        # Canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k).
        return csa_decode_start_set(
            batch=DECODE_B, seq=DECODE_S, compress_ratio=COMPRESS_RATIO,
            state_block_size=DECODE_COMPRESS_STATE_BLOCK_SIZE)
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


# Prefill shape and paging contract.
PREFILL_B = 1
PREFILL_S = 128
PREFILL_START_POS = 0
PREFILL_COMPRESSED_LEN = PREFILL_S // COMPRESS_RATIO
PREFILL_ROWS = PREFILL_B * PREFILL_COMPRESSED_LEN
assert HEAD_DIM % POOL_HEAD_TILE == 0
POOL_HEAD_BLOCKS = HEAD_DIM // POOL_HEAD_TILE
K_TILE = 512
OUT_TILE = 32
HEAD_TILE = 64
RMS_TILE = 16

PREFILL_T = PREFILL_B * PREFILL_S
CSA_STATE_BLOCK_SIZE = RATIO4_STATE_BLOCK_SIZE
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_MAX_BLOCKS
MAX_CMP_WRITES = max(1, PREFILL_T // COMPRESS_RATIO)
PACKED_PROJ_BLOCKS = OUT_DIM // OUT_TILE
PACKED_POOL_BLOCKS = MAX_CMP_WRITES * POOL_HEAD_BLOCKS
PACKED_STATE_UPDATE_TILE = 16
PACKED_RMS_TILE = 16


@pl.jit.inline
def prefill_compressor_ratio4_proj(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    kv_proj_out: pl.Tensor[[PREFILL_T, OUT_DIM], pl.FP32],
    score_proj_out: pl.Tensor[[PREFILL_T, OUT_DIM], pl.FP32],
):
    for proj_idx in pl.spmd(PACKED_PROJ_BLOCKS, name_hint="prefill_c4_kv_score_proj"):
        o0 = proj_idx * OUT_TILE
        kv_acc = pl.create_tensor([PREFILL_T, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([PREFILL_T, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
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
def prefill_compressor_ratio4(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    compress_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[PREFILL_B, CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[PREFILL_T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
):
    # Thin prefill wrapper: build the num_tokens-bounded write schedule, project
    # (static whole-M tiling), run the shared core with the fresh prefill state,
    # then write cmp_kv (+ keepalive). num_tokens is consumed only by the schedule
    # builder; the core gates scatter on state_slot_mapping >= 0. See
    # compressor_core_ratio4.
    compress_state_flat = pl.reshape(compress_state, [CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    write_pos_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    state_table_row_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    build_prefill_write_schedule(
        position_ids,
        cmp_slot_mapping,
        num_tokens,
        write_pos_map,
        write_dst_map,
        state_table_row_map,
    )

    # Prefill has no scattered RMS padding: its write rows are already compact, so
    # the pool enumeration is the identity over the write rows (core behaviour
    # unchanged for prefill).
    pool_row_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_pool_row_map"):
        for p in pl.range(MAX_CMP_WRITES):
            pl.write(pool_row_map, [0, p], pl.cast(p, pl.INT32))

    cmp4_kv_proj_scratch = pl.create_tensor([PREFILL_T, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_scratch = pl.create_tensor([PREFILL_T, OUT_DIM], dtype=pl.FP32)
    prefill_compressor_ratio4_proj(x, wkv, wgate, cmp4_kv_proj_scratch, cmp4_score_proj_scratch)

    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    cos_b = pl.create_tensor([MAX_CMP_WRITES, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    sin_b = pl.create_tensor([MAX_CMP_WRITES, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    normed_kv = compressor_core_ratio4(
        cmp4_kv_proj_scratch,
        cmp4_score_proj_scratch,
        position_ids,
        state_slot_mapping,
        ape,
        norm_w,
        compress_state_flat,
        compress_state_block_table,
        freqs_cos,
        freqs_sin,
        write_pos_map,
        write_dst_map,
        state_table_row_map,
        pool_row_map,
        pooled_kv,
        normed_kv,
        cos_b,
        sin_b,
    )

    cmp_kv_flat = finalize_compressor_writes(
        normed_kv,
        write_dst_map,
        cmp_kv_flat,
        pl.const(PACKED_RMS_TILE, pl.INT32),
        pl.const(1, pl.INT32),
    )

    cmp_kv = pl.reshape(cmp_kv_flat, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    compress_state = pl.reshape(compress_state_flat, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    return cmp_kv, compress_state


def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch

    x = tensors["x"].view(PREFILL_T, D).float()
    compress_state_flat = tensors["compress_state"].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv.t()    # wkv stored [OUT_DIM, D] for b_trans
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[0, block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra

    for token_id in range(int(tensors["num_tokens"])):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)

        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_row = state_row(prev_abs)
                if prev_row >= 0:
                    pool_kv[s] = compress_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = compress_state_flat[prev_row, OUT_DIM : OUT_DIM + HEAD_DIM]

            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = compress_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = compress_state_flat[cur_row, OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM]

        for t in range(int(tensors["num_tokens"])):
            pos = int(position_ids[t].item())
            if pos < prev_start or pos > write_pos:
                continue
            if pos < cur_start:
                pool_slot = pos - prev_start
                col0 = 0
            else:
                pool_slot = COMPRESS_RATIO + pos - cur_start
                col0 = HEAD_DIM
            ape_slot = pos % COMPRESS_RATIO
            pool_kv[pool_slot] = kv_proj[t, col0 : col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[t, col0 : col0 + HEAD_DIM] + ape[ape_slot, col0 : col0 + HEAD_DIM]

        init_slot = STATE_LEN - 1
        mi = pool_score[init_slot : init_slot + 1].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[init_slot : init_slot + 1].clone()
        for slot_i in range(STATE_LEN - 1):
            if slot_i < COMPRESS_RATIO and write_pos < 2 * COMPRESS_RATIO - 1:
                continue
            slot_score = pool_score[slot_i : slot_i + 1]
            slot_kv = pool_kv[slot_i : slot_i + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha = torch.exp(mi - mi_next)
            beta = torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        rot_even = rope_even * cos - rope_odd * sin
        rot_odd = rope_even * sin + rope_odd * cos
        normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

    for t in range(int(tensors["num_tokens"])):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        ape_slot = pos % COMPRESS_RATIO
        compress_state_flat[dst_row, 0:OUT_DIM] = kv_proj[t]
        compress_state_flat[dst_row, OUT_DIM:COMPRESS_STATE_DIM] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["cmp_kv"][:] = cmp_kv
    tensors["compress_state"][:] = compress_state_flat.view_as(tensors["compress_state"])


@pl.jit
def prefill_compressor_ratio4_test(
    x: pl.Tensor[[PREFILL_T, D], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[PREFILL_B, CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[PREFILL_T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
    state_slot_mapping: pl.Tensor[[PREFILL_T], pl.INT64],
):
    return prefill_compressor_ratio4(
        x, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )


def build_prefill_tensor_specs(start_pos: int = PREFILL_START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if start_pos < 0 or start_pos + PREFILL_T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - PREFILL_T}, got {start_pos}")

    def init_compress_state_block_table():
        table = torch.full((PREFILL_B, CSA_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for block in range(CSA_STATE_MAX_BLOCKS):
            table[0, block] = (block * 17 + 3) % CSA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        return int(table[0, block].item()) * CSA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(PREFILL_T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_compress_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row, 0:OUT_DIM] = (torch.rand(OUT_DIM) - 0.5) * 0.05
                flat[row, OUT_DIM:COMPRESS_STATE_DIM] = (torch.rand(OUT_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash CSA (ratio-4) compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors the decode path.
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0245
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0388
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1243
    def init_norm_w():
        return 0.9666 + 0.1929 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + PREFILL_T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((PREFILL_T,), -1, dtype=torch.int64)
        for t in range(PREFILL_T):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = (pos + 1) // COMPRESS_RATIO - 1
                if dst_row >= PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")
                mapping[t] = dst_row
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((PREFILL_T,), -1, dtype=torch.int64)
        for t in range(PREFILL_T):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [PREFILL_T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("compress_state", [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [PREFILL_B, CSA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("position_ids", [PREFILL_T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, PREFILL_T),
        TensorSpec("cmp_slot_mapping", [PREFILL_T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [PREFILL_T], torch.int64, init_value=init_state_slot_mapping),
    ]


def _run_decode_validation(args):
    from golden import ratio_allclose, run_jit

    return run_jit(
        fn=decode_compressor_ratio4_test,
        specs=build_decode_tensor_specs(args.start_pos),
        golden_fn=golden_decode_compressor_ratio4,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
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
        fn=prefill_compressor_ratio4_test,
        specs=build_prefill_tensor_specs(start_pos),
        golden_fn=golden_prefill_compressor_ratio4,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        compile_only=args.compile_only,
        compare_fn={
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 compressor ratio4 validation.")
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
