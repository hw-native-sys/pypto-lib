# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer KV Compressor (decode incremental, ratio=4 overlap)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    IDX_CACHE_BLOCK_NUM,
    FP32_NEG_INF,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


# Dynamic shape variables.
# Under CP the indexer cache is built from the whole TP group's token stream while
# the indexer's query half stays on the local rows, so this half carries its own
# token and request axes.
B_DYN = pl.dynamic("DECODE_IDX_C4_B_DYN")
S_DYN = pl.dynamic("DECODE_IDX_C4_S_DYN")
T_DYN = pl.dynamic("DECODE_IDX_C4_T_DYN")  # T = B * S

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.index_head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.index_nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = (STATE_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

# tiling
K_TILE = 512
OUT_TILE = 64
PROJ_OUT_TILE = 32  # kv_score_proj N-tile
assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"
MM_B_TILE = 16
# kv_score_proj runs on persistent workers, each striding over the block list.
KV_SCORE_WORKERS = 24
# Scratch spans the CP group's whole token stream, not the rank-local B * S.
GROUP_BS = DECODE_BATCH * DECODE_SEQ
BS_PAD = ((GROUP_BS + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16  # 16-row block of B (hadamard matmul M multiple of 16)


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Token-local, interleave-duplicated cos and sign-folded sin. Only ratio-4
    # boundary rows are consumed.
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE  # ceil to whole 16-row cube tiles
    rms_blocks = (bs + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    x_flat = x
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    compress_state_flat = pl.reshape(compress_state, [compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = kv
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_block_num * BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * BLOCK_SIZE, 1])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(KV_SCORE_WORKERS, name_hint="kv_score_proj", deps=[late_dep]) as _kv_score_tid:
        kv_worker = pl.tile.get_block_idx()
        for idx in pl.range(kv_worker, t_matmul * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), KV_SCORE_WORKERS):
            global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
            o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
            kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
            score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // K_TILE, stage=2):
                k0 = kb * K_TILE
                x_rows = pl.min(MM_B_TILE, bs - global_row0)
                x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
                # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
                # GM->L1 load is a DN2ZN (each [PROJ_OUT_TILE, K_TILE] row is K-contiguous = long
                # bursts) instead of ND2NZ on [K_TILE, PROJ_OUT_TILE] (K strided = many short
                # bursts). Mirrors the main compressor (decode_compressor_ratio4); the strided
                # ND2NZ form here was ~2x slower on this matmul (43us -> ~20us per task).
                wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
                wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                    score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                    score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)

            kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
            score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc

    # Pool every ratio-4 boundary against the old persistent ring plus the
    # current-step projection overlay. State is committed only after all pools
    # have finished, so later tokens cannot overwrite rows needed by an earlier
    # boundary in the same S=8 step.
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    # One block per request: each c_idx owns its own state ring and writes only
    # its own pooled_kv rows, so the whole nest is parallel over requests.
    with pl.spmd(b_dim, name_hint="scatter_softmax_pool", deps=[_kv_score_tid]) as pool_tid:
        c_idx = pl.tile.get_block_idx()
        first_pos_b = pl.read(position_ids, [c_idx * s_dim])
        for s_idx in pl.range(s_dim):
            token = c_idx * s_dim + s_idx
            token_pos = pl.read(position_ids, [token])
            pooled_kv[token : token + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
            if (token_pos + 1) % COMPRESS_RATIO == 0:
                window_start = token_pos - STATE_LEN + 1
                for h0 in pl.range(0, HEAD_DIM, HEAD_TILE):
                    last_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                    mi = pl.add(
                        score_proj_pad[
                            token : token + 1,
                            HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE,
                        ],
                        ape[
                            last_ape_row : last_ape_row + 1,
                            HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE,
                        ],
                    )
                    li = pl.exp(pl.sub(mi, mi))
                    oi = kv_proj_pad[
                        token : token + 1,
                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE,
                    ]
                    for state_idx in pl.range(STATE_LEN - 1):
                        logical_pos = window_start + state_idx
                        value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                        score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                        state_half = 0
                        if state_idx >= COMPRESS_RATIO:
                            state_half = HEAD_DIM
                        if logical_pos >= 0 and logical_pos < first_pos_b:
                            ring_row = logical_pos % STATE_LEN
                            state_page_off = ring_row // COMPRESS_STATE_BLOCK_SIZE
                            state_blk_id_i32 = pl.read(
                                compress_state_block_table, [c_idx, state_page_off])
                            if state_blk_id_i32 >= 0:
                                state_blk_id = pl.cast(state_blk_id_i32, pl.INDEX)
                                state_row = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + ring_row % COMPRESS_STATE_BLOCK_SIZE
                                value = compress_state_flat[
                                    state_row : state_row + 1,
                                    state_half + h0 : state_half + h0 + HEAD_TILE,
                                ]
                                score = compress_state_flat[
                                    state_row : state_row + 1,
                                    OUT_DIM + state_half + h0 : OUT_DIM + state_half + h0 + HEAD_TILE,
                                ]
                        if logical_pos >= first_pos_b:
                            if logical_pos <= token_pos:
                                overlay_token = c_idx * s_dim + logical_pos - first_pos_b
                                ape_row = pl.cast(logical_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                                value = kv_proj_pad[
                                    overlay_token : overlay_token + 1,
                                    state_half + h0 : state_half + h0 + HEAD_TILE,
                                ]
                                score = pl.add(
                                    score_proj_pad[
                                        overlay_token : overlay_token + 1,
                                        state_half + h0 : state_half + h0 + HEAD_TILE,
                                    ],
                                    ape[
                                        ape_row : ape_row + 1,
                                        state_half + h0 : state_half + h0 + HEAD_TILE,
                                    ],
                                )
                        mi_next = pl.maximum(mi, score)
                        alpha = pl.exp(pl.sub(mi, mi_next))
                        beta = pl.exp(pl.sub(score, mi_next))
                        li = pl.add(pl.mul(alpha, li), beta)
                        oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                        mi = mi_next
                    pooled_kv[token : token + 1, h0 : h0 + HEAD_TILE] = pl.div(oi, li)

    # The recurrent state ring is a commit, not a source for the current step.
    # One block per request, like the pool above. Each token commits to its own
    # ring row: S <= STATE_LEN, so a request's tokens hold distinct positions mod
    # STATE_LEN, and requests hold distinct state pages.
    with pl.spmd(b_dim, name_hint="compress_state_commit", deps=[pool_tid]):
        c_idx = pl.tile.get_block_idx()
        for s_idx in pl.range(s_dim):
            token = c_idx * s_dim + s_idx
            state_row_i64 = pl.read(inner_state_slot_mapping, [token])
            if state_row_i64 >= 0:
                state_row = pl.cast(state_row_i64, pl.INDEX)
                token_pos = pl.read(position_ids, [token])
                ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = kv_proj_pad[
                    token : token + 1, 0 : OUT_DIM]
                compress_state_flat[state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM] = pl.add(
                    score_proj_pad[token : token + 1, 0 : OUT_DIM],
                    ape[ape_row : ape_row + 1, 0 : OUT_DIM],
                )

    normed_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.spmd(rms_blocks, name_hint="rmsnorm_rope", deps=[pool_tid]) as rms_tid:
        rms_blk = pl.tile.get_block_idx()
        # one 16-row token block; rows rms_blk_rows..15 are pad on the tail block.
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the
        # full ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        b0 = rms_blk * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, bs - b0)
        cos_b = pl.slice(cos, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        sin_b = pl.slice(sin, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HEAD_DIM, HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.pipeline(0, NOPE_HEAD_DIM, HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

        kv_rope_norm = pooled_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rms_norm_rope in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is BF16 -> cast on write.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = pl.cast(
            rope_rot,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_final = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(rms_blocks, name_hint="kv_hadamard", deps=[rms_tid]) as hadamard_tid:
        had_blk = pl.tile.get_block_idx()
        had_b0 = had_blk * RMS_PAD_TILE
        kv_proj_tile = normed_kv[had_b0 : had_b0 + RMS_PAD_TILE, 0 : HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[had_b0 : had_b0 + RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    with pl.spmd(rms_blocks, name_hint="kv_and_cache_write", deps=[hadamard_tid]) as _write_tid:
        wr_blk = pl.tile.get_block_idx()
        # C8 quant-on-write: per-row INT8 quant of the block (M=RMS_PAD_TILE keeps tiles 32B-aligned;
        # quantize the bf16-rounded value to match golden)
        wr_b0 = wr_blk * RMS_PAD_TILE
        wr_blk_rows = pl.min(RMS_PAD_TILE, bs - wr_b0)
        kv_blk_f32 = pl.cast(
            pl.cast(kv_final[wr_b0 : wr_b0 + RMS_PAD_TILE, 0 : HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32)
        # amax = max(|x|); abs-based (max(row_max, -row_min) is wrong on signed KV)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
        kv_scale_q_row = pl.div(pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for inner in pl.range(wr_blk_rows):
            token = wr_b0 + inner
            cache_row_i64 = pl.read(idx_slot_mapping, [token])
            if cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_flat[token : token + 1, :] = kv_final[token : token + 1, 0 : HEAD_DIM]
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = kv_i8_blk[inner : inner + 1, :]
                # scale is one value per position; a [1,1] tile store is sub-32B, so scalar-write it
                pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [inner, 0]))

    return _write_tid


@pl.jit
def compressor_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)

    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    indexer_compressor(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        hadamard,
        idx_kv_cache,
        idx_kv_scale,
        position_ids,
        idx_slot_mapping,
        inner_state_slot_mapping,
        late_dep,
    )
    return kv, compress_state, idx_kv_cache, idx_kv_scale


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

    # Rows this golden never writes stay NaN: ignored by the kv comparator.
    tensors["kv"].fill_(float("nan"))

    x = tensors["x"].float()
    compress_state = tensors["compress_state"]
    compress_state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()
    idx_kv_cache = tensors["idx_kv_cache"]
    idx_kv_scale = tensors["idx_kv_scale"]
    position_ids = tensors["position_ids"].to(torch.int64)
    idx_slot_mapping = tensors["idx_slot_mapping"].to(torch.int64)
    inner_state_slot_mapping = tensors["inner_state_slot_mapping"].to(torch.int64)
    tokens = x.shape[0]
    bsz = tokens // S
    position_ids = position_ids.view(bsz, S)
    idx_slot_mapping = idx_slot_mapping.view(bsz, S)
    inner_state_slot_mapping = inner_state_slot_mapping.view(bsz, S)
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv_proj = x @ wkv.t()
    score_proj = x @ wgate.t()
    old_state = compress_state.clone()
    pooled = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32, device=x.device)

    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        for s in range(S):
            token = b * S + s
            token_pos = int(position_ids[b, s].item())
            if (token_pos + 1) % ratio != 0:
                continue
            kv_rows = []
            score_rows = []
            for state_idx in range(STATE_LEN):
                logical_pos = token_pos - STATE_LEN + 1 + state_idx
                state_half = 0 if state_idx < ratio else HEAD_DIM
                if logical_pos < 0:
                    kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                    score_rows.append(torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device))
                elif logical_pos < first_pos:
                    ring_row = logical_pos % STATE_LEN
                    page_off, intra = divmod(ring_row, COMPRESS_STATE_BLOCK_SIZE)
                    block = int(compress_state_block_table[b, page_off].item())
                    if block >= 0:
                        kv_rows.append(old_state[block, intra, state_half : state_half + HEAD_DIM])
                        score_rows.append(
                            old_state[block, intra, OUT_DIM + state_half : OUT_DIM + state_half + HEAD_DIM])
                    else:
                        kv_rows.append(torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device))
                        score_rows.append(torch.full(
                            (HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device))
                else:
                    overlay_token = b * S + logical_pos - first_pos
                    kv_rows.append(kv_proj[overlay_token, state_half : state_half + HEAD_DIM])
                    score_rows.append(
                        score_proj[overlay_token, state_half : state_half + HEAD_DIM]
                        + ape[logical_pos % ratio, state_half : state_half + HEAD_DIM])
            kvs = torch.stack(kv_rows, dim=0)
            scores = torch.stack(score_rows, dim=0)
            pooled[token] = (kvs * scores.softmax(dim=0)).sum(dim=0)

    for b in range(bsz):
        for s in range(S):
            token = b * S + s
            state_row = int(inner_state_slot_mapping[b, s].item())
            if state_row < 0:
                continue
            block, intra = divmod(state_row, COMPRESS_STATE_BLOCK_SIZE)
            token_pos = int(position_ids[b, s].item())
            compress_state[block, intra, :OUT_DIM] = kv_proj[token]
            compress_state[block, intra, OUT_DIM:] = score_proj[token] + ape[token_pos % ratio]

    tensors["compress_state"][:] = compress_state

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b in range(bsz):
        for s in range(S):
            token = b * S + s
            token_pos = int(position_ids[b, s].item())
            if (token_pos + 1) % ratio != 0:
                continue
            kv_b = rmsnorm(pooled[token : token + 1], norm_w)
            rope_normed = kv_b[..., -rd:]
            rope_swapped = rope_normed.reshape(1, -1, 2).flip(-1).flatten(-2)
            rope_rot = rope_normed * cos[token] + rope_swapped * sin[token]
            kv_b = torch.cat([kv_b[..., :-rd], rope_rot], dim=-1)
            kv_b = kv_b.to(torch.bfloat16).float() @ hadamard

            cache_row = int(idx_slot_mapping[b, s].item())
            if cache_row < 0:
                continue
            tensors["kv"][token : token + 1, :] = kv_b
            blk_id = cache_row // BLOCK_SIZE
            intra = cache_row % BLOCK_SIZE
            # C8 quant-on-write: quantize the bf16-rounded compressed row to int8 + per-position scale
            row_bf16 = kv_b[0].to(torch.bfloat16).float()
            amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
            scale_q = INT8_SCALE_MAX / amax
            idx_kv_cache[blk_id, intra, 0] = torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
            idx_kv_scale[blk_id, intra, 0, 0] = 1.0 / scale_q

    tensors["idx_kv_cache"][:] = idx_kv_cache
    tensors["idx_kv_scale"][:] = idx_kv_scale


def build_tensor_specs(start_pos=None, batch=B):
    import torch  # type: ignore[import]
    from utils import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        token_local_rope,
    )
    from golden import TensorSpec

    def default_starts():
        # Keep the default standalone fixture on a complete recurrent window;
        # explicit --start-pos values still cover cold-start probes.
        values = csa_decode_start_set(
            batch=batch, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=COMPRESS_STATE_BLOCK_SIZE)
        return torch.where(values < STATE_LEN, values + STATE_LEN, values)

    starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=MAX_SEQ_LEN,
        default_fn=default_starts,
    )
    positions = position_ids_from_starts(starts, seq=S)
    state_block_num = batch * COMPRESS_STATE_MAX_BLOCKS
    state_block_table = torch.arange(
        state_block_num - 1, -1, -1, dtype=torch.int32
    ).reshape(batch, COMPRESS_STATE_MAX_BLOCKS)
    ring_rows = positions.to(torch.int64) % STATE_LEN
    state_pages = torch.gather(
        state_block_table.to(torch.int64), 1, ring_rows // COMPRESS_STATE_BLOCK_SIZE)
    state_slots = state_pages * COMPRESS_STATE_BLOCK_SIZE + ring_rows % COMPRESS_STATE_BLOCK_SIZE
    idx_block_table = block_table(
        batch=batch,
        table_blocks=IDX_MAX_BLOCKS,
        physical_blocks=IDX_CACHE_BLOCK_NUM,
    )
    idx_slots = compressed_slot_mapping(
        positions,
        idx_block_table,
        compress_ratio=COMPRESS_RATIO,
        block_size=BLOCK_SIZE,
    )
    rope_positions = torch.where(
        (positions.to(torch.int64) + 1) % COMPRESS_RATIO == 0,
        positions.to(torch.int64) - (COMPRESS_RATIO - 1),
        torch.zeros_like(positions, dtype=torch.int64),
    )
    rope_cos, rope_sin = token_local_rope(
        M,
        COMPRESS_RATIO,
        rope_positions,
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.float32,
    )
    rope_cos = rope_cos[:, : ROPE_HEAD_DIM // 2].repeat_interleave(2, dim=-1)
    rope_sin = rope_sin[:, : ROPE_HEAD_DIM // 2].repeat_interleave(2, dim=-1)
    rope_sign = torch.ones(ROPE_HEAD_DIM, dtype=torch.float32)
    rope_sign[0::2] = -1.0
    rope_sin = rope_sin * rope_sign

    def init_x():
        return torch.rand(batch * S, D)
    def init_compress_state():
        return torch.randn(
            state_block_num, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM
        ) * 0.05
    def init_compress_state_block_table():
        return state_block_table.clone()
    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 8/32 (the CSA inner / indexer compressor).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0270
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0513
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1524
    def init_norm_w():
        return 0.6903 + 0.2663 * torch.randn(HEAD_DIM)
    def init_cos():
        return rope_cos.clone()
    def init_sin():
        return rope_sin.clone()
    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM ** -0.5)
    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.int8)
    def init_idx_kv_scale():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)
    def init_position_ids():
        return positions.clone()
    def init_inner_state_slot_mapping():
        return state_slots.clone()
    def init_idx_slot_mapping():
        return idx_slots.clone()

    return [
        TensorSpec("x", [batch * S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [batch * S, HEAD_DIM], torch.float32),
        TensorSpec("compress_state", [state_block_num, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [batch, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [batch * S, ROPE_HEAD_DIM], torch.float32, init_value=init_cos),
        TensorSpec("sin", [batch * S, ROPE_HEAD_DIM], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.int8, init_value=init_idx_kv_cache),
        TensorSpec("idx_kv_scale", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale),
        TensorSpec("position_ids", [batch * S], torch.int32, init_value=lambda: init_position_ids().reshape(-1)),
        TensorSpec("idx_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_idx_slot_mapping().reshape(-1)),
        TensorSpec("inner_state_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_inner_state_slot_mapping().reshape(-1)),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count up to {B} (the compile-time upper bound). "
                             "The batch axes are pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--start-pos", type=str, default=None,
                        help="Fixture-only start position: one value for a uniform batch or "
                             "a comma-separated value per request.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")
    start_pos = None
    if args.start_pos is not None:
        try:
            start_values = [int(value) for value in args.start_pos.split(",")]
        except ValueError:
            parser.error(f"--start-pos must contain integers, got {args.start_pos!r}")
        start_pos = start_values[0] if len(start_values) == 1 else start_values

    result = run(
        fn=compressor_test,
        specs=build_tensor_specs(start_pos, batch=args.batch),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # kv leaves the 128x128 hadamard rotation, so an element is a 128-term sum of
            # rows reaching |x| ~ 10 and the row's dynamic range runs past 1e4:1. Its error
            # floor is set by that input scale, not by the cancelled output, so atol carries
            # it -- a 1e-4 floor asks small elements for a relative accuracy the bf16 inputs
            # cannot hold. rtol still bounds the large elements.
            "kv":          ratio_allclose(atol=1e-3, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
