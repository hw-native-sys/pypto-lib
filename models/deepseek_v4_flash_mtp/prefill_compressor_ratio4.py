# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill attention compressor for ratio-4 overlapping KV cache (rotate=False)."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    CSA_STATE_PHYSICAL_BLOCKS,
    FLASH as M,
    FP32_NEG_INF,
    PREFILL_CMP_BLOCK_NUM,
)

# Dynamic shape variables.
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
WRITE_MAP_TOKENS_DYN = pl.dynamic("PREFILL_WRITE_MAP_TOKENS_DYN")
WRITE_MAP_ROWS_DYN = pl.dynamic("PREFILL_WRITE_MAP_ROWS_DYN")

# model config
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings
B = 1
S = 128
T = B * S
START_POS = 0
COMPRESS_RATIO = 4
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)

# paged compressed state / KV cache
CSA_STATE_BLOCK_SIZE = 4
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_PHYSICAL_BLOCKS

# tiling
K_TILE = 512                  # projection D (K) reduction tile
OUT_TILE = 64                 # projection OUT_DIM (N) tile
HEAD_D_TILE = 512             # head-dim tile for the softmax pool
HEAD_TILE = 64
STATE_UPDATE_TOKEN_TILE = 16
STATE_UPDATE_OUT_TILE = 512
PACKED_RMS_TILE = 16


@pl.jit.inline(auto_scope=False)
def _compressor_write_map(
    position_ids: pl.Tensor[[WRITE_MAP_TOKENS_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[WRITE_MAP_TOKENS_DYN], pl.INT64],
    token_start: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    map_row: pl.Scalar[pl.INDEX],
    write_pos_map: pl.Tensor[[WRITE_MAP_ROWS_DYN, MAX_CMP_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[WRITE_MAP_ROWS_DYN, MAX_CMP_WRITES], pl.INT32],
):
    """Compact one leaf's valid positions and physical cache destinations."""
    write_pos_tile = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=0)
    write_dst_tile = pl.full([1, MAX_CMP_WRITES], dtype=pl.INT32, value=-1)
    map_seen = pl.cast(0, pl.INDEX)
    for map_w in pl.range(T):
        if map_w < num_tokens:
            map_slot_raw = pl.read(slot_mapping, [token_start + map_w])
            if map_slot_raw >= 0:
                pl.write(write_pos_tile, [0, map_seen], pl.read(position_ids, [token_start + map_w]))
                pl.write(write_dst_tile, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                map_seen = map_seen + 1
    write_pos_map[map_row:map_row + 1, 0:MAX_CMP_WRITES] = write_pos_tile
    write_dst_map[map_row:map_row + 1, 0:MAX_CMP_WRITES] = write_dst_tile
    return write_pos_map, write_dst_map


@pl.jit.inline(auto_scope=False)
def _compressor_norm_rope_cache_write(
    pooled_kv: pl.Tensor[[MAX_CMP_WRITES, HEAD_DIM], pl.FP32],
    write_pos_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cache_rows: pl.Scalar[pl.INDEX],
    cache_block: pl.Scalar[pl.INDEX],
):
    """Normalize pooled rows, apply RoPE, and write the compressed cache."""
    cmp_kv_flat = pl.tensor.view(cmp_kv, shape=[cache_rows, HEAD_DIM])
    norm_w_view = pl.tensor.view(norm_w, shape=[1, HEAD_DIM])
    norm_w_2d = pl.load(norm_w_view, [0, 0], [1, HEAD_DIM])
    for final_block in pl.range(cache_block, MAX_CMP_WRITES // PACKED_RMS_TILE, 1):
        final_base = final_block * PACKED_RMS_TILE
        cos_b = pl.tile.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.tile.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            rms_write_slot_raw = pl.read(write_dst_map, [0, final_i])
            if rms_write_slot_raw >= 0:
                rms_write_pos = pl.read(write_pos_map, [0, final_i])
                cmp_pos = pl.cast(rms_write_pos + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    pl.load(freqs_cos, [cmp_pos, 0], [1, ROPE_HEAD_DIM // 2]),
                    target_type=pl.FP32,
                )
                sin_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    pl.load(freqs_sin, [cmp_pos, 0], [1, ROPE_HEAD_DIM // 2]),
                    target_type=pl.FP32,
                )

        partial_sq = pl.tile.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=0.0)
        rms_reduce_tmp = pl.tile.full([PACKED_RMS_TILE, HEAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pl.load(pooled_kv, [final_base, k0], [PACKED_RMS_TILE, HEAD_TILE])
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_row_sum = pl.row_sum(kv_rms_sq, tmp_tile=rms_reduce_tmp)
            partial_sq = pl.add(partial_sq, pl.reshape(kv_row_sum, [1, PACKED_RMS_TILE]))
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [PACKED_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        kv_nope = pl.load(pooled_kv, [final_base, 0], [PACKED_RMS_TILE, NOPE_HEAD_DIM])
        gamma_nope = pl.cast(norm_w_2d[:, 0:NOPE_HEAD_DIM], pl.FP32)
        nope_scaled = pl.row_expand_mul(kv_nope, inv_rms)
        normed_nope = pl.col_expand_mul(nope_scaled, gamma_nope)
        kv_rope_norm = pl.load(pooled_kv, [final_base, NOPE_HEAD_DIM], [PACKED_RMS_TILE, ROPE_HEAD_DIM])
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        # A3 interleaved swap-gather (matches decode): single data gather + sign trick instead of
        # the P0101/P1010 de-interleave gather + rotate + re-interleave scatter. swap_idx (j^1),
        # sign ([-1,+1,...]) and dup_idx (j>>1) are built in-kernel from pl.arange; cos_il/sin_il
        # dup-gather the half-width cos_b/sin_b. out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j].
        rope_ones = pl.tile.full([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)                                       # j>>1
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)                                                # [-1,+1,...]
        rope_row_seed = pl.cast(pl.tile.arange(0, [1, PACKED_RMS_TILE], dtype=pl.INT32), pl.FP32)
        rope_row_ones = pl.tile.full([ROPE_HEAD_DIM, PACKED_RMS_TILE], dtype=pl.FP32, value=1.0)
        rope_row_grid = pl.col_expand_mul(rope_row_ones, rope_row_seed)
        rope_rows = pl.transpose(rope_row_grid, axis1=0, axis2=1)
        rope_freq_offsets = pl.mul(rope_rows, ROPE_HEAD_DIM // 2)
        rope_value_offsets = pl.mul(rope_rows, ROPE_HEAD_DIM)
        rope_dup_flat = pl.add(rope_dup_idx, pl.cast(rope_freq_offsets, pl.INT32))
        rope_swap_flat = pl.add(rope_swap_idx, pl.cast(rope_value_offsets, pl.INT32))
        rope_gather_tmp = pl.tile.full([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32, value=0)
        cos_il = pl.tile.gather(cos_b, rope_dup_flat, rope_gather_tmp)
        sin_il = pl.tile.gather(sin_b, rope_dup_flat, rope_gather_tmp)
        swapped = pl.tile.gather(rope_normed, rope_swap_flat, rope_gather_tmp)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_rows = pl.tile.concat(normed_nope, rope_rot)
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            dst_row_raw = pl.read(write_dst_map, [0, final_i])
            if dst_row_raw >= 0:
                dst_row = pl.cast(dst_row_raw, pl.INDEX)
                final_row = pl.slice(normed_rows, [1, HEAD_DIM], [final_dt, 0])
                final_bf16 = pl.cast(final_row, target_type=pl.BF16, mode="rint")
                pl.store(final_bf16, [dst_row, 0], cmp_kv_flat)
            else:
                keepalive_row = cache_rows - MAX_CMP_WRITES + final_i
                keepalive = pl.load(cmp_kv_flat, [keepalive_row, 0], [1, HEAD_DIM])
                pl.store(keepalive, [keepalive_row, 0], cmp_kv_flat)

    return cmp_kv


@pl.jit.inline(auto_scope=False)
def _compressor_pool_reduce(
    pool_kv_tile: pl.Tensor[[STATE_LEN, HEAD_D_TILE], pl.FP32],
    pool_score_tile: pl.Tensor[[STATE_LEN, HEAD_D_TILE], pl.FP32],
    write_pos: pl.Scalar[pl.INT32],
    write_i: pl.Scalar[pl.INDEX],
    h0: pl.Scalar[pl.INDEX],
    pooled_kv: pl.Tensor[[MAX_CMP_WRITES, HEAD_DIM], pl.FP32],
):
    """Fold one main compressor window in its original slot order."""
    init_slot = STATE_LEN - 1
    mi_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
    li_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
    oi_buf = pl.create_tensor([1, HEAD_D_TILE], dtype=pl.FP32)
    mi_buf[0:1, 0:HEAD_D_TILE] = pool_score_tile[init_slot : init_slot + 1, 0:HEAD_D_TILE]
    li_buf[0:1, 0:HEAD_D_TILE] = pl.exp(pl.sub(mi_buf[0:1, 0:HEAD_D_TILE], mi_buf[0:1, 0:HEAD_D_TILE]))
    oi_buf[0:1, 0:HEAD_D_TILE] = pool_kv_tile[init_slot : init_slot + 1, 0:HEAD_D_TILE]
    for pool_slot_i in pl.range(STATE_LEN - 1):
        if pool_slot_i >= COMPRESS_RATIO or write_pos >= 2 * COMPRESS_RATIO - 1:
            mi = mi_buf[0:1, 0:HEAD_D_TILE]
            li = li_buf[0:1, 0:HEAD_D_TILE]
            oi = oi_buf[0:1, 0:HEAD_D_TILE]
            slot_score = pool_score_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_D_TILE]
            slot_kv = pool_kv_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_D_TILE]
            mi_next = pl.maximum(mi, slot_score)
            alpha = pl.exp(pl.sub(mi, mi_next))
            beta = pl.exp(pl.sub(slot_score, mi_next))
            li_next = pl.add(pl.mul(alpha, li), beta)
            oi_next = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
            mi_buf[0:1, 0:HEAD_D_TILE] = mi_next
            li_buf[0:1, 0:HEAD_D_TILE] = li_next
            oi_buf[0:1, 0:HEAD_D_TILE] = oi_next
    pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_D_TILE] = pl.div(
        oi_buf[0:1, 0:HEAD_D_TILE],
        li_buf[0:1, 0:HEAD_D_TILE],
    )


@pl.jit.inline(auto_scope=False)
def _compressor_store_state_rows(
    kv_proj: pl.Tensor[[T, OUT_DIM], pl.FP32],
    score_proj: pl.Tensor[[T, OUT_DIM], pl.FP32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    state: pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    update_base: pl.Scalar[pl.INDEX],
    update_token_tile: pl.Scalar[pl.INDEX],
    first_row: pl.Scalar[pl.INDEX],
):
    """Write projected KV and APE-adjusted score rows to one state bank."""
    state_rows = pl.tensor.dim(state, 0) * CSA_STATE_BLOCK_SIZE
    state_flat = pl.tensor.view(state, shape=[state_rows, COMPRESS_STATE_DIM])
    pool_dep_tile = pl.tile.full([1, STATE_UPDATE_OUT_TILE], dtype=pl.FP32, value=0.0)
    for update_dt in pl.range(update_token_tile):
        update_t = update_base + update_dt
        if update_t >= first_row and update_t < num_tokens:
            state_row_raw = pl.read(state_slot_mapping, [update_t])
            if state_row_raw >= 0:
                state_row = pl.cast(state_row_raw, pl.INDEX)
                update_pos = pl.read(position_ids, [update_t])
                ape_slot = pl.cast(update_pos % COMPRESS_RATIO, pl.INDEX)
                for update_ob in pl.range(OUT_DIM // STATE_UPDATE_OUT_TILE):
                    update_o0 = update_ob * STATE_UPDATE_OUT_TILE
                    ape_row = pl.load(ape, [ape_slot, update_o0], [1, STATE_UPDATE_OUT_TILE])
                    kv_row = pl.load(kv_proj, [update_t, update_o0], [1, STATE_UPDATE_OUT_TILE])
                    score_row = pl.load(score_proj, [update_t, update_o0], [1, STATE_UPDATE_OUT_TILE])
                    state_kv = pl.add(kv_row, pool_dep_tile)
                    state_score = pl.add(score_row, ape_row)
                    state_score = pl.add(state_score, pool_dep_tile)
                    score_o0 = OUT_DIM + update_o0
                    pl.store(state_kv, [state_row, update_o0], state_flat)
                    pl.store(state_score, [state_row, score_o0], state_flat)
    return state


@pl.jit.inline
def _compressor_prepare_state(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    write_pos_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    completion: pl.Array[1, pl.TASK_ID],
    next_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    forward_rows: pl.Scalar[pl.INDEX],
    history_pool_worker_cap: pl.Scalar[pl.INT32],
    emit_pool: pl.Scalar[pl.BOOL],
    emit_output_tasks: pl.Scalar[pl.INT32],
    skip_history: pl.Scalar[pl.INT32],
):
    state_block_num = pl.tensor.dim(compress_state, 0)
    cmp4_kv_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [state_block_num * CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(OUT_DIM // (2 * OUT_TILE), name_hint="prefill_c4_kv_score_proj") as projection_tid:
        proj_worker = pl.tile.get_block_idx()
        for proj_column in pl.range(2):
            o0 = (proj_worker * 2 + proj_column) * OUT_TILE
            kv_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
            score_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // K_TILE, stage=2):
                k0 = kb * K_TILE
                x_tile = x[0:T, k0 : k0 + K_TILE]
                # Transposed [OUT_DIM, D] weights with K-contiguous loads.
                wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(k0 == 0))
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(k0 == 0))
            cmp4_kv_proj_scratch[0:T, o0 : o0 + OUT_TILE] = kv_acc
            cmp4_score_proj_scratch[0:T, o0 : o0 + OUT_TILE] = score_acc

    pool_done = pl.array.create(1, pl.TASK_ID)
    pool_done[0] = projection_tid
    # Seed leaves consume projected state only; their pooled output has no reader.
    if emit_pool:
        # A contiguous active prefix can close at most ceil(num_tokens / ratio)
        # compressed rows, regardless of the absolute start-position alignment.
        active_pool_blocks = ((num_tokens + COMPRESS_RATIO - 1) // COMPRESS_RATIO) * (HEAD_DIM // HEAD_D_TILE)
        pool_workers = pl.min(active_pool_blocks, 12)
        current_pool_flags = pl.create_tensor([MAX_CMP_WRITES, 16], dtype=pl.INT32)
        if emit_output_tasks > 0:
            with pl.spmd(pool_workers, name_hint="prefill_c4_softmax_pool_current") as current_pool_tid:
                pool_worker = pl.tile.get_block_idx()
                for pool_idx in pl.range(pool_worker, active_pool_blocks, pl.max(pool_workers, 1)):
                    write_i = pool_idx // (HEAD_DIM // HEAD_D_TILE)
                    hb = pool_idx - write_i * (HEAD_DIM // HEAD_D_TILE)
                    h0 = hb * HEAD_D_TILE
                    write_slot_raw = pl.read(write_dst_map, [0, write_i])
                    if write_slot_raw >= 0:
                        write_pos = pl.read(write_pos_map, [0, write_i])
                        cur_start = write_pos + 1 - COMPRESS_RATIO
                        prev_start = cur_start - COMPRESS_RATIO
                        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
                        pool_kv_tile[0:STATE_LEN, 0:HEAD_D_TILE] = pl.full([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32, value=0.0)
                        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
                        pool_score_tile[0:STATE_LEN, 0:HEAD_D_TILE] = pl.full([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                        current_slots = pl.full([1, STATE_LEN], dtype=pl.INT32, value=0)
                        for pool_t in pl.range(T):
                            if pool_t < num_tokens:
                                pool_pos = pl.read(position_ids, [pool_t])
                                if pool_pos <= write_pos:
                                    if pool_pos >= prev_start:
                                        if pool_pos < cur_start:
                                            pool_slot = pl.cast(pool_pos - prev_start, pl.INDEX)
                                            pool_col0 = h0
                                        else:
                                            pool_slot = pl.cast(COMPRESS_RATIO + pool_pos - cur_start, pl.INDEX)
                                            pool_col0 = HEAD_DIM + h0
                                        pool_ape_slot = pl.cast(pool_pos % COMPRESS_RATIO, pl.INDEX)
                                        pool_ape = ape[pool_ape_slot : pool_ape_slot + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                        pool_score_row = cmp4_score_proj_scratch[pool_t : pool_t + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                        pool_score = pl.add(pool_score_row, pool_ape)
                                        pool_kv_tile[pool_slot : pool_slot + 1, 0:HEAD_D_TILE] = cmp4_kv_proj_scratch[
                                            pool_t : pool_t + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                        pool_score_tile[pool_slot : pool_slot + 1, 0:HEAD_D_TILE] = pool_score
                                        pl.write(current_slots, [0, pool_slot], pl.cast(1, pl.INT32))

                        current_complete = pl.cast(1, pl.INT32)
                        for coverage_slot in pl.range(STATE_LEN):
                            if coverage_slot >= COMPRESS_RATIO or write_pos >= 2 * COMPRESS_RATIO - 1:
                                if pl.read(current_slots, [0, coverage_slot]) == 0:
                                    current_complete = pl.cast(0, pl.INT32)
                        pl.write(current_pool_flags, [write_i, 0], current_complete)
                        if current_complete != 0:
                            _compressor_pool_reduce(
                                pool_kv_tile, pool_score_tile, write_pos, write_i, h0,
                                pooled_kv,
                            )
                        else:
                            pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=0.0)
                    else:
                        pl.write(current_pool_flags, [write_i, 0], pl.cast(1, pl.INT32))
                        pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=0.0)
            pool_done[0] = current_pool_tid
            if skip_history <= 0:
                history_pool_workers = pl.min(active_pool_blocks, history_pool_worker_cap)
                with pl.spmd(history_pool_workers, name_hint="prefill_c4_softmax_pool", deps=[current_pool_tid]) as history_pool_tid:
                    pool_worker = pl.tile.get_block_idx()
                    for pool_idx in pl.range(pool_worker, active_pool_blocks, pl.max(history_pool_workers, 1)):
                        write_i = pool_idx // (HEAD_DIM // HEAD_D_TILE)
                        if pl.read(current_pool_flags, [write_i, 0]) == 0:
                            hb = pool_idx - write_i * (HEAD_DIM // HEAD_D_TILE)
                            h0 = hb * HEAD_D_TILE
                            pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
                            pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_D_TILE], dtype=pl.FP32)
                            write_slot_raw = pl.read(write_dst_map, [0, write_i])
                            if write_slot_raw >= 0:
                                write_pos = pl.read(write_pos_map, [0, write_i])
                                cur_start = write_pos + 1 - COMPRESS_RATIO
                                prev_start = cur_start - COMPRESS_RATIO
                                for pool_s in pl.range(COMPRESS_RATIO):
                                    prev_abs = prev_start + pool_s
                                    front_slot = pool_s
                                    pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=0.0)
                                    pool_score_tile[front_slot : front_slot + 1, 0:HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                                    if write_pos >= 2 * COMPRESS_RATIO - 1:
                                        prev_state_block = pl.cast(prev_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                                        prev_state_intra = pl.cast(prev_abs - prev_state_block * CSA_STATE_BLOCK_SIZE, pl.INDEX)
                                        prev_phys_block_raw = pl.read(compress_state_block_table, [prev_state_block])
                                        if prev_phys_block_raw >= 0:
                                            prev_phys_block = pl.cast(prev_phys_block_raw, pl.INDEX)
                                            prev_state_row = prev_phys_block * CSA_STATE_BLOCK_SIZE + prev_state_intra
                                            prev_kv_col0 = h0
                                            prev_score_col0 = OUT_DIM + h0
                                            pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                                                prev_state_row : prev_state_row + 1, prev_kv_col0 : prev_kv_col0 + HEAD_D_TILE]
                                            pool_score_tile[front_slot : front_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                                                prev_state_row : prev_state_row + 1, prev_score_col0 : prev_score_col0 + HEAD_D_TILE]

                                    cur_abs = cur_start + pool_s
                                    back_slot = COMPRESS_RATIO + pool_s
                                    pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=0.0)
                                    pool_score_tile[back_slot : back_slot + 1, 0:HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                                    cur_state_block = pl.cast(cur_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                                    cur_state_intra = pl.cast(cur_abs - cur_state_block * CSA_STATE_BLOCK_SIZE, pl.INDEX)
                                    cur_phys_block_raw = pl.read(compress_state_block_table, [cur_state_block])
                                    if cur_phys_block_raw >= 0:
                                        cur_phys_block = pl.cast(cur_phys_block_raw, pl.INDEX)
                                        cur_state_row = cur_phys_block * CSA_STATE_BLOCK_SIZE + cur_state_intra
                                        cur_kv_col0 = HEAD_DIM + h0
                                        cur_score_col0 = OUT_DIM + HEAD_DIM + h0
                                        pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                                            cur_state_row : cur_state_row + 1, cur_kv_col0 : cur_kv_col0 + HEAD_D_TILE]
                                        pool_score_tile[back_slot : back_slot + 1, 0:HEAD_D_TILE] = compress_state_flat[
                                            cur_state_row : cur_state_row + 1, cur_score_col0 : cur_score_col0 + HEAD_D_TILE]

                                for pool_t in pl.range(T):
                                    if pool_t < num_tokens:
                                        pool_pos = pl.read(position_ids, [pool_t])
                                        if pool_pos <= write_pos:
                                            if pool_pos >= prev_start:
                                                if pool_pos < cur_start:
                                                    pool_slot = pl.cast(pool_pos - prev_start, pl.INDEX)
                                                    pool_col0 = h0
                                                else:
                                                    pool_slot = pl.cast(COMPRESS_RATIO + pool_pos - cur_start, pl.INDEX)
                                                    pool_col0 = HEAD_DIM + h0
                                                pool_ape_slot = pl.cast(pool_pos % COMPRESS_RATIO, pl.INDEX)
                                                pool_ape = ape[pool_ape_slot : pool_ape_slot + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                                pool_score_row = cmp4_score_proj_scratch[pool_t : pool_t + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                                pool_score = pl.add(pool_score_row, pool_ape)
                                                pool_kv_tile[pool_slot : pool_slot + 1, 0:HEAD_D_TILE] = cmp4_kv_proj_scratch[
                                                    pool_t : pool_t + 1, pool_col0 : pool_col0 + HEAD_D_TILE]
                                                pool_score_tile[pool_slot : pool_slot + 1, 0:HEAD_D_TILE] = pool_score

                                _compressor_pool_reduce(
                                    pool_kv_tile, pool_score_tile, write_pos, write_i, h0,
                                    pooled_kv,
                                )
                            else:
                                pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_D_TILE] = pl.full([1, HEAD_D_TILE], dtype=pl.FP32, value=0.0)

                pool_done[0] = history_pool_tid

    # State writeback or boundary forwarding to the next bank.
    state_update_token_tile = pl.cast(STATE_UPDATE_TOKEN_TILE, pl.INDEX)
    if forward_rows > 0:
        state_update_token_tile = pl.cast(4, pl.INDEX)
    state_update_rows = pl.max(pl.cast(num_tokens, pl.INDEX), 0)
    state_update_tiles = (state_update_rows + state_update_token_tile - 1) // state_update_token_tile
    first_state_update_tile = pl.cast(0, pl.INDEX)
    if forward_rows > 0:
        first_state_update_tile = pl.max(state_update_rows - forward_rows, 0) // state_update_token_tile
    state_update_workers = pl.max(1, pl.min(T // state_update_token_tile, state_update_tiles - first_state_update_tile))
    if forward_rows > 0:
        # The caller uses a distinct next bank and consumes only its live boundary.
        with pl.spmd(state_update_workers, name_hint="prefill_c4_state_forward") as forward_tid:
            forward_tb = first_state_update_tile + pl.tile.get_block_idx()
            forward_base = forward_tb * state_update_token_tile
            forward_first = pl.max(state_update_rows - forward_rows, 0)
            _compressor_store_state_rows(
                cmp4_kv_proj_scratch, cmp4_score_proj_scratch, ape, position_ids, num_tokens,
                state_slot_mapping, next_state, forward_base, state_update_token_tile, forward_first,
            )
        completion[0] = forward_tid
    else:
        # In-place state writes wait for all history-pool readers.
        with pl.spmd(state_update_workers, name_hint="prefill_c4_state_update", deps=[pool_done[0]]) as update_tid:
            update_tb = first_state_update_tile + pl.tile.get_block_idx()
            update_base = update_tb * state_update_token_tile
            _compressor_store_state_rows(
                cmp4_kv_proj_scratch, cmp4_score_proj_scratch, ape, position_ids, num_tokens,
                state_slot_mapping, compress_state, update_base, state_update_token_tile, 0,
            )
        completion[0] = update_tid
    return pooled_kv, compress_state


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    write_pos_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, MAX_CMP_WRITES], pl.INT32],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    completion: pl.Array[1, pl.TASK_ID],
):
    state_done = pl.array.create(1, pl.TASK_ID)
    pooled_kv, state_written = _compressor_prepare_state(
        x, compress_state, compress_state_block_table, wkv, wgate, ape,
        position_ids, num_tokens, write_pos_map, write_dst_map, state_slot_mapping, state_done,
        compress_state, 0,
        pl.cast(16, pl.INT32), True,
        1, 0,
    )
    cache_rows = pl.tensor.dim(cmp_kv, 0) * CMP_STORAGE_BLOCK_SIZE
    with pl.spmd(1, name_hint="prefill_c4_rmsnorm_rope_cache_write") as cache_write_tid:
        cache_block = pl.tile.get_block_idx()
        _compressor_norm_rope_cache_write(
            pooled_kv, write_pos_map, write_dst_map, norm_w, freqs_cos, freqs_sin,
            cmp_kv, cache_rows, cache_block,
        )
    completion[0] = pl.system.task_dummy(deps=[cache_write_tid, state_done[0]])
    return cmp_kv, state_written


def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch

    x = tensors["x"].view(T, D).float()
    compress_state_flat = tensors["compress_state"].view(-1, COMPRESS_STATE_DIM)
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv.t()    # wkv stored [OUT_DIM, D] for b_trans
    score_proj = x @ wgate.t()

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
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
                    pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = kv_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = score_state_flat[cur_row, HEAD_DIM:OUT_DIM]

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
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["cmp_kv"][:] = cmp_kv


@pl.jit
def prefill_compressor_ratio4_test(
    x: pl.Tensor[[T, D], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    write_pos_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, MAX_CMP_WRITES], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_write_map"):
        _compressor_write_map(
            position_ids, cmp_slot_mapping, pl.cast(0, pl.INDEX), num_tokens, pl.cast(0, pl.INDEX),
            write_pos_map, write_dst_map,
        )
    completion = pl.array.create(1, pl.TASK_ID)
    return compressor_ratio4(
        x, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, write_pos_map, write_dst_map, state_slot_mapping, completion,
    )


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")

    def init_compress_state_block_table():
        table = torch.full((CSA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(CSA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_STATE_PHYSICAL_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        return int(table[block].item()) * CSA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_compress_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        flat = state.view(-1, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash CSA (ratio-4) compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_compressor_ratio4.
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
        return torch.zeros(PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = (pos + 1) // COMPRESS_RATIO - 1
                if dst_row >= PREFILL_CMP_BLOCK_NUM * CMP_STORAGE_BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")
                mapping[t] = dst_row
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("compress_state", [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [CSA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, T),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio4 validation.")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False,
                        help="Compile/codegen only; also the implicit behavior on *sim platforms used by CI.")
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0, lowered into position_ids and cmp_slot_mapping.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=prefill_compressor_ratio4_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio4,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        compile_only=args.compile_only,
        compare_fn={
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
