# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # A2/A3 device validation; simulator CI is deferred for this model.
"""DeepSeek-V4 Indexer KV Compressor (decode incremental, ratio=4 overlap)."""


import pypto.language as pl

if __package__:
    from .config import FLASH as M, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
else:
    from config import FLASH as M, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
if __package__:
    from .decode_shapes import (
        DECODE_BATCH,
        DECODE_SEQ,
        BLOCK_SIZE,
        C4A_COMPRESSOR_BLOCK_SIZE,
        DECODE_IDX_BLOCK_NUM,
        IDX_CACHE_MAX_BLOCKS,
    )
else:
    from decode_shapes import DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_IDX_BLOCK_NUM, IDX_CACHE_MAX_BLOCKS
if __package__:
    from .rope_interleave import rope_interleave
else:
    from rope_interleave import rope_interleave


# model config
B = DECODE_BATCH
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
IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = 65
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_IDX_CACHE_BLOCK_NUM_DYN')
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_INNER_STATE_BLOCK_NUM_DYN')

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
PROJ_OUT_TILE = 32  # kv_score_proj N-tile
assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"
B_TILE = 8
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RMS_PAD_TILE = BS_PAD  # token-indexed rows, padded to a Cube M multiple of 16
RMS_PAD_ROWS = RMS_PAD_TILE


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    compress_state_flat = pl.reshape(
        compress_state, [compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(BS_PAD * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]):
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [PROJ_OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, PROJ_OUT_TILE] (K strided = many short
            # bursts). Mirrors the main compressor (decode_compressor_ratio4); the strided
            # ND2NZ form here was ~2x slower on this matmul (43us -> ~20us per task).
            wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(k0 == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(k0 == 0))

        kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc

    # Pool each boundary from immutable history plus this invocation's projections.
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool", allow_early_resolve=True
    ) as pool_tid:
        pooled_kv[:, :] = pl.full([BS_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c_idx in pl.range(B):
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            for s_idx in pl.range(S):
                token = c_idx * S + s_idx
                token_pos = pl.read(position_ids, [c_idx, s_idx])
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    window_start = token_pos - STATE_LEN + 1
                    for h0 in pl.range(0, HEAD_DIM, HEAD_TILE):
                        last_ape = pl.cast(token_pos % COMPRESS_RATIO, pl.INDEX)
                        last_score = score_proj_pad[
                            token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE
                        ]
                        last_bias = ape[last_ape : last_ape + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        mi = pl.add(last_score, last_bias)
                        initial_delta = pl.sub(mi, mi)
                        li = pl.exp(initial_delta)
                        oi = kv_proj_pad[token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            state_half = 0
                            if state_idx >= COMPRESS_RATIO:
                                state_half = HEAD_DIM
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                page = logical_pos // COMPRESS_STATE_BLOCK_SIZE
                                physical = pl.read(compress_state_block_table, [c_idx, page])
                                if physical >= 0:
                                    state_row = (
                                        physical * COMPRESS_STATE_BLOCK_SIZE
                                        + logical_pos % COMPRESS_STATE_BLOCK_SIZE
                                    )
                                    col = state_half + h0
                                    value = compress_state_flat[
                                        state_row : state_row + 1, col : col + HEAD_TILE
                                    ]
                                    score_col = OUT_DIM + col
                                    score = compress_state_flat[
                                        state_row : state_row + 1, score_col : score_col + HEAD_TILE
                                    ]
                            if logical_pos >= first_pos_b and logical_pos <= token_pos:
                                overlay = c_idx * S + logical_pos - first_pos_b
                                ape_row = pl.cast(logical_pos % COMPRESS_RATIO, pl.INDEX)
                                col = state_half + h0
                                value = kv_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
                                score_base = score_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
                                score_bias = ape[ape_row : ape_row + 1, col : col + HEAD_TILE]
                                score = pl.add(score_base, score_bias)
                            mi_next = pl.maximum(mi, score)
                            alpha_delta = pl.sub(mi, mi_next)
                            beta_delta = pl.sub(score, mi_next)
                            alpha = pl.exp(alpha_delta)
                            beta = pl.exp(beta_delta)
                            li_scaled = pl.mul(alpha, li)
                            li = pl.add(li_scaled, beta)
                            oi_scaled = pl.mul(oi, alpha)
                            value_scaled = pl.mul(value, beta)
                            oi = pl.add(oi_scaled, value_scaled)
                            mi = mi_next
                        pooled_kv[token : token + 1, h0 : h0 + HEAD_TILE] = pl.div(oi, li)

    # History must remain unchanged until every boundary has finished pooling.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="compress_state_commit", deps=[pool_tid]):
        for c_idx in pl.range(B):
            for s_idx in pl.range(S):
                commit_state_row_i64 = pl.read(inner_state_slot_mapping, [c_idx, s_idx])
                if commit_state_row_i64 >= 0:
                    commit_state_row = pl.cast(commit_state_row_i64, pl.INDEX)
                    token = c_idx * S + s_idx
                    token_pos = pl.read(position_ids, [c_idx, s_idx])
                    commit_ape_row = pl.cast(token_pos % COMPRESS_RATIO, pl.INDEX)
                    compress_state_flat[commit_state_row : commit_state_row + 1, 0:OUT_DIM] = kv_proj_pad[
                        token : token + 1, :
                    ]
                    commit_score_base = score_proj_pad[token : token + 1, :]
                    commit_score_bias = ape[commit_ape_row : commit_ape_row + 1, :]
                    compress_state_flat[
                        commit_state_row : commit_state_row + 1, OUT_DIM:COMPRESS_STATE_DIM
                    ] = pl.add(commit_score_base, commit_score_bias)

    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope", deps=[pool_tid]):
        # single 16-row block: B real rows at rows 0..B-1, rows B..15 are pad.
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the
        # full ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_b[0 : B * S, 0:ROPE_HEAD_DIM] = cos[0 : B * S, 0:ROPE_HEAD_DIM]
        sin_b[0 : B * S, 0:ROPE_HEAD_DIM] = sin[0 : B * S, 0:ROPE_HEAD_DIM]
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HEAD_DIM, HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[0:RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.pipeline(0, NOPE_HEAD_DIM, HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[0:RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0:RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

        kv_rope_norm = pooled_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rms_norm_rope in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is BF16 -> cast on write.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(
            rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        )
        rope_dup_f = pl.cast(
            pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
        )
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))  # j%2
        rope_swap_idx = pl.cast(
            pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32
        )  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[0:RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM] = pl.cast(
            rope_rot,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_final = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
        kv_proj_tile = normed_kv[0:RMS_PAD_TILE, 0:HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[0:RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write", allow_early_resolve=True):
        # C8 quant-on-write: per-row INT8 quant of the block (M=RMS_PAD_TILE keeps tiles 32B-aligned;
        # quantize the bf16-rounded value to match golden)
        kv_blk_f32 = pl.cast(
            pl.cast(kv_final[0:RMS_PAD_TILE, 0:HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32,
        )
        # amax = max(|x|); abs-based (max(row_max, -row_min) is wrong on signed KV)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, INT8_AMAX_EPS)
        qd_idxkv_numerator = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qd_idxkv_work = pl.mul(kv_amax, 5.421010862427522e-20)
        qd_idxkv_estimate = pl.div(qd_idxkv_numerator, qd_idxkv_work)
        qd_idxkv_left_split = pl.mul(qd_idxkv_estimate, 4097.0)
        qd_idxkv_left_delta = pl.sub(qd_idxkv_left_split, qd_idxkv_estimate)
        qd_idxkv_left_hi = pl.sub(qd_idxkv_left_split, qd_idxkv_left_delta)
        qd_idxkv_left_lo = pl.sub(qd_idxkv_estimate, qd_idxkv_left_hi)
        qd_idxkv_right_split = pl.mul(qd_idxkv_work, 4097.0)
        qd_idxkv_right_delta = pl.sub(qd_idxkv_right_split, qd_idxkv_work)
        qd_idxkv_right_hi = pl.sub(qd_idxkv_right_split, qd_idxkv_right_delta)
        qd_idxkv_right_lo = pl.sub(qd_idxkv_work, qd_idxkv_right_hi)
        qd_idxkv_product = pl.mul(qd_idxkv_estimate, qd_idxkv_work)
        qd_idxkv_hi_product = pl.mul(qd_idxkv_left_hi, qd_idxkv_right_hi)
        qd_idxkv_error = pl.sub(qd_idxkv_hi_product, qd_idxkv_product)
        qd_idxkv_hi_lo = pl.mul(qd_idxkv_left_hi, qd_idxkv_right_lo)
        qd_idxkv_error = pl.add(qd_idxkv_error, qd_idxkv_hi_lo)
        qd_idxkv_lo_hi = pl.mul(qd_idxkv_left_lo, qd_idxkv_right_hi)
        qd_idxkv_error = pl.add(qd_idxkv_error, qd_idxkv_lo_hi)
        qd_idxkv_lo_product = pl.mul(qd_idxkv_left_lo, qd_idxkv_right_lo)
        qd_idxkv_error = pl.add(qd_idxkv_error, qd_idxkv_lo_product)
        qd_idxkv_residual_hi = pl.sub(qd_idxkv_numerator, qd_idxkv_product)
        qd_idxkv_residual = pl.sub(qd_idxkv_residual_hi, qd_idxkv_error)
        qd_idxkv_correction = pl.div(qd_idxkv_residual, qd_idxkv_work)
        qd_idxkv_corrected_work = pl.add(qd_idxkv_estimate, qd_idxkv_correction)
        kv_scale_q_row = pl.mul(qd_idxkv_corrected_work, 5.421010862427522e-20)
        icrcp_result = kv_scale_q_row
        icrcp_ones_row = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=1.0)
        icrcp_ef_work = pl.mul(kv_scale_q_row, 65536.0)
        icrcp_ef_estimate = pl.div(icrcp_ones_row, icrcp_ef_work)
        icrcp_ef_left_split = pl.mul(icrcp_ef_estimate, 4097.0)
        icrcp_ef_left_delta = pl.sub(icrcp_ef_left_split, icrcp_ef_estimate)
        icrcp_ef_left_hi = pl.sub(icrcp_ef_left_split, icrcp_ef_left_delta)
        icrcp_ef_left_lo = pl.sub(icrcp_ef_estimate, icrcp_ef_left_hi)
        icrcp_ef_right_split = pl.mul(icrcp_ef_work, 4097.0)
        icrcp_ef_right_delta = pl.sub(icrcp_ef_right_split, icrcp_ef_work)
        icrcp_ef_right_hi = pl.sub(icrcp_ef_right_split, icrcp_ef_right_delta)
        icrcp_ef_right_lo = pl.sub(icrcp_ef_work, icrcp_ef_right_hi)
        icrcp_ef_product = pl.mul(icrcp_ef_estimate, icrcp_ef_work)
        icrcp_ef_hi_product = pl.mul(icrcp_ef_left_hi, icrcp_ef_right_hi)
        icrcp_ef_error = pl.sub(icrcp_ef_hi_product, icrcp_ef_product)
        icrcp_ef_hi_lo = pl.mul(icrcp_ef_left_hi, icrcp_ef_right_lo)
        icrcp_ef_error = pl.add(icrcp_ef_error, icrcp_ef_hi_lo)
        icrcp_ef_lo_hi = pl.mul(icrcp_ef_left_lo, icrcp_ef_right_hi)
        icrcp_ef_error = pl.add(icrcp_ef_error, icrcp_ef_lo_hi)
        icrcp_ef_lo_product = pl.mul(icrcp_ef_left_lo, icrcp_ef_right_lo)
        icrcp_ef_error = pl.add(icrcp_ef_error, icrcp_ef_lo_product)
        icrcp_ef_residual_hi = pl.sub(icrcp_ones_row, icrcp_ef_product)
        icrcp_ef_residual = pl.sub(icrcp_ef_residual_hi, icrcp_ef_error)
        icrcp_ef_correction = pl.div(icrcp_ef_residual, icrcp_ef_work)
        icrcp_ef_corrected_work = pl.add(icrcp_ef_estimate, icrcp_ef_correction)
        icrcp_result = pl.mul(icrcp_ef_corrected_work, 65536.0)
        kv_scale_dq_row = icrcp_result
        kv_scale_dq_col = pl.reshape(kv_scale_dq_row, [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for token in pl.range(B * S):
            c_idx = token // S
            s_idx = token - c_idx * S
            position = pl.read(position_ids, [c_idx, s_idx])
            cache_row_i64 = pl.read(idx_slot_mapping, [c_idx, s_idx])
            if (position + 1) % COMPRESS_RATIO == 0 and cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_flat[token : token + 1, :] = kv_final[token : token + 1, :]
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = kv_i8_blk[token : token + 1, :]
                pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [token, 0]))

    return kv


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    # The fused path builds these once in csa_rope_step; standalone does the same
    # prep here so the fixture / golden keep the half-width cos/sin ABI.
    cos_il = pl.create_tensor([B * S, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([B * S, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cos, sin, cos_il, sin_signed, B * S)
    indexer_compressor(
        x,
        kv,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos_il,
        sin_signed,
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

    if __package__:
        from .utils import bf16_matmul_fp32
    else:
        from utils import bf16_matmul_fp32

    # Unmapped boundaries and non-boundary rows are outside the output write set.
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
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = bf16_matmul_fp32(x, wkv.t())  # [B, S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = bf16_matmul_fp32(x, wgate.t())  # [B, S, OUT_DIM]

    # Each boundary stays at its input token index, as in the device output.
    pooled = torch.zeros(bsz, S, HEAD_DIM, dtype=torch.float32, device=x.device)
    visible_boundaries = []
    score = score + ape[position_ids % ratio]

    def read_pool_row(b, abs_pos, half):
        if abs_pos < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        first_pos = int(position_ids[b, 0].item())
        channel = half * HEAD_DIM
        if abs_pos >= first_pos:
            token = abs_pos - first_pos
            return kv[b, token, channel : channel + HEAD_DIM], score[b, token, channel : channel + HEAD_DIM]
        blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, channel : channel + HEAD_DIM],
            compress_state[blk_id, intra, OUT_DIM + channel : OUT_DIM + channel + HEAD_DIM],
        )

    # Read every prefix window before a commit can overwrite rolling physical state.
    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        first_boundary_s = ratio - 1 - first_pos % ratio
        for boundary_s in range(first_boundary_s, S, ratio):
            if int(idx_slot_mapping[b, boundary_s].item()) < 0:
                continue
            boundary_end = int(position_ids[b, boundary_s].item())
            current_start = boundary_end - ratio + 1
            rows = [read_pool_row(b, pos, 0) for pos in range(current_start - ratio, current_start)]
            rows += [read_pool_row(b, pos, 1) for pos in range(current_start, boundary_end + 1)]
            kvs = torch.stack([row[0] for row in rows])
            scs = torch.stack([row[1] for row in rows])
            pooled[b, boundary_s] = (kvs * scs.softmax(dim=0)).sum(dim=0)
            visible_boundaries.append((b, boundary_s))

    for b in range(bsz):
        for token in range(S):
            state_row = int(inner_state_slot_mapping[b, token].item())
            if state_row >= 0:
                blk_id = state_row // COMPRESS_STATE_BLOCK_SIZE
                intra = state_row % COMPRESS_STATE_BLOCK_SIZE
                compress_state[blk_id, intra, :OUT_DIM] = kv[b, token]
                compress_state[blk_id, intra, OUT_DIM:] = score[b, token]

    tensors["compress_state"][:] = compress_state

    def rmsnorm(x, w):
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + EPS)
        return w * x

    for b, boundary_s in visible_boundaries:
        kv_b = rmsnorm(pooled[b : b + 1, boundary_s : boundary_s + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos[b * S + boundary_s].view(-1), sin[b * S + boundary_s].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        kv_b = bf16_matmul_fp32(kv_b.to(torch.bfloat16).float(), hadamard)

        cache_row = int(idx_slot_mapping[b, boundary_s].item())
        if cache_row >= 0:
            tensors["kv"][b : b + 1, boundary_s : boundary_s + 1, :] = kv_b
            blk_id = cache_row // IDX_STORAGE_BLOCK_SIZE
            intra = cache_row % IDX_STORAGE_BLOCK_SIZE
            # C8 quant-on-write: quantize the bf16-rounded compressed row to int8 + per-position scale
            row_bf16 = kv_b[0, 0].to(torch.bfloat16).float()
            amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
            scale_q = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
            idx_kv_cache[blk_id, intra, 0] = (
                torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
            )
            idx_kv_scale[blk_id, intra, 0, 0] = 1.0 / scale_q

    tensors["idx_kv_cache"][:] = idx_kv_cache
    tensors["idx_kv_scale"][:] = idx_kv_scale


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]

    if __package__:
        from .utils import (
            block_table,
            compressed_slot_mapping,
            csa_decode_start_set,
            position_ids_from_starts,
            resolve_start_positions,
            state_slot_mapping,
        )
    else:
        from utils import (
            block_table,
            compressed_slot_mapping,
            csa_decode_start_set,
            position_ids_from_starts,
            resolve_start_positions,
            state_slot_mapping,
        )
    from golden import TensorSpec

    if __package__:
        from .utils import build_rope_tables, materialize_half_rope_tables
    else:
        from utils import build_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)

    def init_compress_state():
        state = torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        state[:, :, OUT_DIM:] = FP32_NEG_INF
        return state

    def init_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
        )

    # Calibrated to the real DeepSeek-V4-Flash CSA inner (indexer) compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0293

    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0512

    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1528

    def init_norm_w():
        return 0.6850 + 0.2610 * torch.randn(HEAD_DIM)

    def init_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundaries = (positions + 1) % COMPRESS_RATIO == 0
        return torch.where(boundaries, positions - COMPRESS_RATIO + 1, 0).reshape(-1)

    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]

    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]

    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM**-0.5)

    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.int8)

    def init_idx_kv_scale():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1)

    def init_idx_block_table():
        return block_table(
            batch=B,
            table_blocks=IDX_CACHE_MAX_BLOCKS,
            physical_blocks=IDX_CACHE_BLOCK_NUM,
        )

    def init_default_start_pos():
        # Canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k).
        return csa_decode_start_set(
            batch=B, seq=S, compress_ratio=COMPRESS_RATIO, state_block_size=COMPRESS_STATE_BLOCK_SIZE
        )

    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )

    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)

    def init_inner_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_compress_state_block_table(),
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        )

    def init_idx_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=IDX_STORAGE_BLOCK_SIZE,
        )

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32),
        TensorSpec(
            "compress_state",
            [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [B, COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            torch.int8,
            init_value=init_idx_kv_cache,
        ),
        TensorSpec(
            "idx_kv_scale",
            [IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, 1],
            torch.float32,
            init_value=init_idx_kv_scale,
        ),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("idx_slot_mapping", [B, S], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [B, S], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


def indexer_cache_compare_fn():
    """Apply numerical tolerances only to allocator-mapped cache rows."""
    from golden import mapped_pool_ratio_allclose

    return {
        "idx_kv_cache": mapped_pool_ratio_allclose(
            "idx_slot_mapping", mapping_shape=(B, S), block_size=IDX_STORAGE_BLOCK_SIZE,
            pool_name="idx_kv_cache", atol=1, rtol=0, max_error_ratio=0.01,
        ),
        "idx_kv_scale": mapped_pool_ratio_allclose(
            "idx_slot_mapping", mapping_shape=(B, S), block_size=IDX_STORAGE_BLOCK_SIZE,
            pool_name="idx_kv_scale", atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01,
        ),
    }


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    if __package__:
        from .config import CONTEXT_CAPACITY
    else:
        from config import CONTEXT_CAPACITY

    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_CAPACITY, help="import-time context capacity (default 1048576)")
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.platform.endswith("sim"),
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            **indexer_cache_compare_fn(),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
