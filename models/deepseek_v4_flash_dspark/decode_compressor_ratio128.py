# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=128 non-overlap).

Uses non-overlapping state layout with 128 slots.
Softmax+pool over all slots. No state shift needed."""

import pypto.language as pl

from rope_interleave import _rope_interleave_active_body
from config import (
    FLASH as M,
    BLOCK_SIZE,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    FP32_NEG_INF,
    HCA_KV_POOL_BLOCKS,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
    HCA_STATE_PHYSICAL_BLOCKS,
    HCA_STATE_ROWS_PER_REQUEST,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")  # T = B * S
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
EVENT_DYN = pl.dynamic("HCA_EVENT_DYN")

# model config
B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim

# kernel-local (ratio-128 non-overlap compressor)
COMPRESS_RATIO = 128
COFF = 1
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
# The semantic ratio-128 state ring is stored in sixteen 8-row allocator
# pages after main switched the paged-cache ABI to BLOCK_SIZE=32.  Page IDs
# therefore remain request-local descriptors; they are never inferred from B.
COMPRESS_STATE_BLOCK_SIZE = HCA_STATE_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = HCA_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_BLOCK_NUM = HCA_KV_POOL_BLOCKS
if HCA_STATE_PAGES_PER_REQUEST * COMPRESS_STATE_BLOCK_SIZE < STATE_LEN:
    raise ValueError("ratio128 state descriptor cannot cover the semantic state ring")
if HCA_STATE_ROWS_PER_REQUEST != STATE_LEN:
    raise ValueError("ratio128 state-row capacity must equal the compressor ratio")

# tiling
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
MM_B_TILE = 16
RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)
STATE_COMMIT_TOKEN_TILE = 8
# softmax_pool reduces over the state axis with column reductions (no transpose), so it can
# afford a wider head tile than HEAD_TILE: each wider tile loads each state block fewer times
# (HEAD_DIM/POOL_HEAD_TILE tiles/batch instead of HEAD_DIM/HEAD_TILE), cutting load redundancy.
POOL_HEAD_TILE = 128
# Gather rows per softmax_pool iteration. Held independent of the state page size: the
# two [STATE_LEN, POOL_HEAD_TILE] FP32 pools already take 128 KB of the 184 KB Vec space,
# leaving room for one double-buffered [POOL_STATE_TILE, POOL_HEAD_TILE] pair.
POOL_STATE_TILE = min(COMPRESS_STATE_BLOCK_SIZE, 8)
POOL_STATE_STEPS = STATE_LEN // POOL_STATE_TILE


@pl.jit.inline
def compressor_ratio128(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    state_page_ids: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    request_event_indices: pl.Tensor[[B_DYN], pl.INT32],
    cos: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(state_page_ids, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)
    event_count = pl.tensor.dim(cos, 0)

    x_flat = x
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
    rms_blocks = (b_dim + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    rms_rows = rms_blocks * RMS_PAD_TILE
    kv_proj_pad = pl.create_tensor([t_matmul, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([t_matmul, OUT_DIM], dtype=pl.FP32)

    with pl.spmd(
        t_matmul * OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]
    ) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, bs - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, OUT_TILE] (K strided = many short
            # bursts). Cuts the transaction-bound MTE2 cost. Matches ratio4/CSA layout.
            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)

        kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    compress_state_rows_num = compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE
    compress_state_rows = pl.reshape(compress_state, [compress_state_rows_num, COMPRESS_STATE_DIM])
    pooled_kv = pl.create_tensor([rms_rows, HEAD_DIM], dtype=pl.FP32)
    # Page-granular persistent gather plus the current-step overlay.
    head_tile_count = HEAD_DIM // POOL_HEAD_TILE
    pool_grid = b_dim * head_tile_count
    # The pool reads the projection outputs.
    with pl.spmd(
        pool_grid, name_hint="scatter_softmax_pool", deps=[_kv_score_tid]
    ) as pool_tid:
        block_idx = pl.tile.get_block_idx()
        global_c_idx = block_idx // head_tile_count
        head_tile = block_idx - global_c_idx * head_tile_count
        h0 = head_tile * POOL_HEAD_TILE
        # Zero the pooled slice for non-boundary steps.
        pooled_kv[
            global_c_idx : global_c_idx + 1, h0 : h0 + POOL_HEAD_TILE
        ] = pl.full([1, POOL_HEAD_TILE], dtype=pl.FP32, value=0.0)
        first_pos_gate = pl.read(position_ids, [global_c_idx * s_dim])
        pos_gate = first_pos_gate % COMPRESS_RATIO
        if pos_gate + s_dim >= COMPRESS_RATIO:
            compress_pos = first_pos_gate + (COMPRESS_RATIO - 1 - pos_gate)
            state_pos0 = compress_pos - (COMPRESS_RATIO - 1)
            state_valid_begin = pl.read(state_valid_ranges, [global_c_idx, 0])
            state_valid_end = pl.read(state_valid_ranges, [global_c_idx, 1])
            request_epoch = pl.read(request_epochs, [global_c_idx])
            softmax_score_state = pl.create_tensor(
                [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
            )
            softmax_kv_state = pl.create_tensor(
                [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
            )
            # The aligned state window maps gather steps to relative pages 0..15.
            for gather_i in pl.pipeline(POOL_STATE_STEPS, stage=2):
                s0 = gather_i * POOL_STATE_TILE
                slot_score = pl.full(
                    [POOL_STATE_TILE, POOL_HEAD_TILE],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                slot_kv = pl.full(
                    [POOL_STATE_TILE, POOL_HEAD_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                state_page_raw = pl.read(state_page_ids, [global_c_idx, gather_i])
                state_page_epoch = pl.read(
                    state_page_epochs, [global_c_idx, gather_i]
                )
                for gather_row in pl.range(POOL_STATE_TILE):
                    logical_pos = state_pos0 + s0 + gather_row
                    if state_page_raw >= 0:
                        if state_page_raw < compress_state_block_num:
                            if state_page_epoch == request_epoch:
                                if logical_pos >= state_valid_begin:
                                    if logical_pos < state_valid_end:
                                        state_page = pl.cast(
                                            state_page_raw, target_type=pl.INDEX
                                        )
                                        row0 = state_page * COMPRESS_STATE_BLOCK_SIZE
                                        slot_score[
                                            gather_row : gather_row + 1,
                                            0 : POOL_HEAD_TILE,
                                        ] = compress_state_rows[
                                            row0 + gather_row : row0 + gather_row + 1,
                                            OUT_DIM + h0 : OUT_DIM + h0 + POOL_HEAD_TILE,
                                        ]
                                        slot_kv[
                                            gather_row : gather_row + 1,
                                            0 : POOL_HEAD_TILE,
                                        ] = compress_state_rows[
                                            row0 + gather_row : row0 + gather_row + 1,
                                            h0 : h0 + POOL_HEAD_TILE,
                                        ]
                    if logical_pos >= first_pos_gate:
                        if logical_pos <= compress_pos:
                            overlay_s = logical_pos - first_pos_gate
                            overlay_row = global_c_idx * s_dim + overlay_s
                            token_ape_row = pl.cast(
                                logical_pos % COMPRESS_RATIO,
                                target_type=pl.INDEX,
                            )
                            slot_kv[
                                gather_row : gather_row + 1,
                                0 : POOL_HEAD_TILE,
                            ] = kv_proj_pad[
                                overlay_row : overlay_row + 1,
                                h0 : h0 + POOL_HEAD_TILE,
                            ]
                            slot_score[
                                gather_row : gather_row + 1,
                                0 : POOL_HEAD_TILE,
                            ] = pl.add(
                                score_proj_pad[
                                    overlay_row : overlay_row + 1,
                                    h0 : h0 + POOL_HEAD_TILE,
                                ],
                                ape[
                                    token_ape_row : token_ape_row + 1,
                                    h0 : h0 + POOL_HEAD_TILE,
                                ],
                            )
                softmax_score_state[s0 : s0 + POOL_STATE_TILE, :] = slot_score
                softmax_kv_state[s0 : s0 + POOL_STATE_TILE, :] = slot_kv
            score_max = pl.col_max(softmax_score_state)
            score_exp = pl.col_expand_expdif(softmax_score_state, score_max)
            score_sum = pl.col_sum(score_exp)
            score_prob = pl.col_expand_mul(score_exp, pl.recip(score_sum))
            pooled_chunk = pl.col_sum(pl.mul(softmax_kv_state, score_prob))
            pooled_kv[
                global_c_idx : global_c_idx + 1, h0 : h0 + POOL_HEAD_TILE
            ] = pooled_chunk

    state_commit_blocks = (bs + STATE_COMMIT_TOKEN_TILE - 1) // STATE_COMMIT_TOKEN_TILE
    with pl.spmd(
        state_commit_blocks,
        name_hint="state_ring_commit",
        deps=[pool_tid],
    ) as state_commit_tid:
        state_task = pl.tile.get_block_idx()
        state_t0 = state_task * STATE_COMMIT_TOKEN_TILE
        state_rows = pl.min(STATE_COMMIT_TOKEN_TILE, bs - state_t0)
        for state_dt in pl.range(state_rows):
            state_t = state_t0 + state_dt
            state_row_i64 = pl.read(state_slot_mapping, [state_t])
            if state_row_i64 >= 0:
                if state_row_i64 < compress_state_rows_num:
                    state_row = pl.cast(state_row_i64, target_type=pl.INDEX)
                    token_pos = pl.read(position_ids, [state_t])
                    token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                    compress_state_rows[state_row : state_row + 1, 0 : OUT_DIM] = kv_proj_pad[
                        state_t : state_t + 1, 0 : OUT_DIM
                    ]
                    compress_state_rows[
                        state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM
                    ] = pl.add(
                        score_proj_pad[state_t : state_t + 1, 0 : OUT_DIM],
                        ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM],
                    )

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([rms_rows, HEAD_DIM], dtype=pl.FP32)
    kv_flat = kv
    cmp_flat_rows = cmp_block_num * BLOCK_SIZE
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_flat_rows, HEAD_DIM])

    with pl.spmd(rms_blocks, name_hint="rmsnorm_rope_cache_write", deps=[pool_tid]) as _rms_tid:
        # one 16-row block of B; rows rms_blk_rows..15 are pad on the tail block
        b0 = pl.tile.get_block_idx() * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, b_dim - b0)
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the full
        # ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for local_c_idx in pl.range(rms_blk_rows):
            row_c_idx = b0 + local_c_idx
            event_index = pl.read(request_event_indices, [row_c_idx])
            if event_index >= 0:
                if event_index < event_count:
                    cos_b[local_c_idx : local_c_idx + 1, 0 : ROPE_HEAD_DIM] = cos[
                        event_index : event_index + 1, 0 : ROPE_HEAD_DIM
                    ]
                    sin_b[local_c_idx : local_c_idx + 1, 0 : ROPE_HEAD_DIM] = sin[
                        event_index : event_index + 1, 0 : ROPE_HEAD_DIM
                    ]
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, rms_h0 : rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = rms_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, norm_h0 : norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0 : norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0 : b0 + RMS_PAD_TILE, norm_h0 : norm_h0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rope_fused in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is FP32 -> write directly.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_rot

        for local_c_idx in pl.range(rms_blk_rows):
            global_c_idx = b0 + local_c_idx
            first_pos_b = pl.read(position_ids, [global_c_idx * s_dim])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + s_dim >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row = normed_kv[global_c_idx : global_c_idx + 1, 0 : HEAD_DIM]
                cmp_row_i64 = pl.read(cmp_slot_mapping, [global_c_idx * s_dim + boundary_s])
                if cmp_row_i64 >= 0:
                    if cmp_row_i64 < cmp_flat_rows:
                        cmp_row = pl.cast(cmp_row_i64, target_type=pl.INDEX)
                        kv_flat[global_c_idx * s_dim : global_c_idx * s_dim + 1, :] = kv_row
                        cmp_kv_cache_flat[cmp_row : cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")

    return _rms_tid, state_commit_tid


@pl.jit
def compressor_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    state_page_ids: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    state_page_epochs: pl.Tensor[[B_DYN, HCA_STATE_PAGES_PER_REQUEST], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    request_event_indices: pl.Tensor[[B_DYN], pl.INT32],
    cos: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[EVENT_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    state_page_ids.bind_dynamic(0, B_DYN)
    state_valid_ranges.bind_dynamic(0, B_DYN)
    state_page_epochs.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    request_event_indices.bind_dynamic(0, B_DYN)
    cos.bind_dynamic(0, EVENT_DYN)
    sin.bind_dynamic(0, EVENT_DYN)
    cmp_kv_cache.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)

    late_dep = pl.system.task_dummy(deps=[])
    # The fused path builds these once in hca_rope; standalone does the same prep here
    # so the fixture / golden keep the half-width cos/sin ABI.
    event_count = pl.tensor.dim(cos, 0)
    cos_il = pl.create_tensor([event_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([event_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_completion = pl.array.create(1, pl.TASK_ID)
    _rope_interleave_active_body(cos, sin, cos_il, sin_signed, rope_completion)
    rope_dep = pl.system.task_dummy(deps=[late_dep, rope_completion[0]])
    compressor_ratio128(
        x, kv, compress_state,
        state_page_ids, state_valid_ranges, state_page_epochs, request_epochs,
        wkv, wgate, ape, norm_w, request_event_indices,
        cos_il, sin_signed,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping, rope_dep,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    The recurrent state is a sixteen-page ring per request. Current-step state
    remains an overlay until a boundary pool has read the preceding block.
    """
    import torch

    x = tensors["x"].float()
    state_page_ids = tensors["state_page_ids"].to(torch.int64)
    state_valid_ranges = tensors["state_valid_ranges"].to(torch.int64)
    state_page_epochs = tensors["state_page_epochs"].to(torch.int64)
    request_epochs = tensors["request_epochs"].to(torch.int64)
    request_event_indices = tensors["request_event_indices"].to(torch.int64)
    position_ids = tensors["position_ids"].to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].to(torch.int64)
    compress_state = tensors["compress_state"]

    def read_state_row(b, pos):
        ring_row = pos % HCA_STATE_ROWS_PER_REQUEST
        relative_page = ring_row // COMPRESS_STATE_BLOCK_SIZE
        page = int(state_page_ids[b, relative_page].item())
        valid_begin = int(state_valid_ranges[b, 0].item())
        valid_end = int(state_valid_ranges[b, 1].item())
        page_epoch = int(state_page_epochs[b, relative_page].item())
        request_epoch = int(request_epochs[b].item())
        if (
            page < 0
            or page >= int(compress_state.shape[0])
            or page_epoch != request_epoch
            or pos < valid_begin
            or pos >= valid_end
        ):
            return (
                torch.zeros(OUT_DIM, dtype=torch.float32, device=compress_state.device),
                torch.full((OUT_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device),
            )
        return (
            compress_state[page, ring_row % COMPRESS_STATE_BLOCK_SIZE, :OUT_DIM],
            compress_state[page, ring_row % COMPRESS_STATE_BLOCK_SIZE, OUT_DIM:2 * OUT_DIM],
        )

    def write_state_row(slot, kv_row, score_row):
        if slot < 0 or slot >= int(compress_state.shape[0]) * COMPRESS_STATE_BLOCK_SIZE:
            return
        sblk = slot // COMPRESS_STATE_BLOCK_SIZE
        intra = slot % COMPRESS_STATE_BLOCK_SIZE
        compress_state[sblk, intra, :OUT_DIM] = kv_row
        compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM] = score_row

    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
    tokens = x.shape[0]
    bsz = tokens // S
    x = x.view(bsz, S, D)
    position_ids = position_ids.view(bsz, S)
    cmp_slot_mapping = cmp_slot_mapping.view(bsz, S)
    state_slot_mapping = state_slot_mapping.view(bsz, S)
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv.t()                    # [B, S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = x @ wgate.t()               # [B, S, OUT_DIM]
    for b in range(bsz):
        for s in range(S):
            pos = int(position_ids[b, s].item())
            score[b, s, :] = score[b, s, :] + ape[pos % ratio]
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        boundary_s = None
        for s in range(S):
            pos = int(position_ids[b, s].item())
            if (pos + 1) % ratio == 0:
                boundary_s = s

        if boundary_s is not None:
            should_compress_rows[b] = True
            compress_pos = int(position_ids[b, boundary_s].item())
            kv_rows = []
            score_rows = []
            for pos in range(compress_pos - ratio + 1, compress_pos + 1):
                overlay_s = pos - int(position_ids[b, 0].item())
                if 0 <= overlay_s <= boundary_s:
                    kv_row = kv[b, overlay_s, :]
                    score_row = score[b, overlay_s, :]
                else:
                    kv_row, score_row = read_state_row(b, pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
            score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

    for b in range(bsz):
        for s in range(S):
            write_state_row(
                int(state_slot_mapping[b, s].item()),
                kv[b, s, :],
                score[b, s, :],
            )
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
        kv_b = rmsnorm(pooled[b : b + 1], norm_w)

        x_pair = kv_b[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        event_index = int(request_event_indices[b].item())
        if event_index < 0 or event_index >= int(cos.shape[0]):
            continue
        cos_v = cos[event_index].view(-1)
        sin_v = sin[event_index].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        boundary_positions = torch.nonzero((position_ids[b, :S] + 1) % ratio == 0, as_tuple=False).flatten()
        if int(boundary_positions.numel()) == 0:
            continue
        boundary_s = int(boundary_positions[0].item())
        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if 0 <= cmp_row < int(cmp_kv_cache.shape[0]) * BLOCK_SIZE:
            # Kernel writes committed pooled result only to kv[:, 0, :]; leave
            # current-step boundary rows and kv[:, 1:, :] zero-initialized.
            tensors["kv"][b * S : b * S + 1, :] = kv_b[0]
            cblk = cmp_row // BLOCK_SIZE
            intra_offset = cmp_row % BLOCK_SIZE
            cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(start_pos=None, batch=B):
    import torch  # type: ignore[import]
    from utils import (
        hca_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        token_local_rope,
    )
    from golden import TensorSpec

    if batch < 1 or batch > B:
        raise ValueError(f"batch must be in [1, {B}], got {batch}")

    resolved_starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=M.max_position_embeddings,
        default_fn=lambda: hca_decode_start_set(
            batch=batch,
            compress_ratio=COMPRESS_RATIO,
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        ),
    )
    resolved_positions = position_ids_from_starts(resolved_starts, seq=S)
    required_cmp_pages = sum(
        (
            int(resolved_positions[request, -1].item() + 1) // COMPRESS_RATIO
            + BLOCK_SIZE
            - 1
        )
        // BLOCK_SIZE
        for request in range(batch)
    )
    if required_cmp_pages > HCA_KV_POOL_BLOCKS:
        raise ValueError(
            "compressor fixture exceeds the global HCA page pool; "
            "the 1M gate must use batch=1"
        )

    def init_x():
        return torch.rand(batch * S, D)
    def init_state_page_ids():
        generator = torch.Generator()
        generator.manual_seed(5503)
        required_pages = batch * HCA_STATE_PAGES_PER_REQUEST
        if required_pages > COMPRESS_STATE_BLOCK_NUM:
            raise ValueError("ratio128 fixture exceeds the physical state-page pool")
        return torch.randperm(
            COMPRESS_STATE_BLOCK_NUM, generator=generator, dtype=torch.int64
        )[:required_pages].reshape(batch, HCA_STATE_PAGES_PER_REQUEST).to(torch.int32)
    def init_state_valid_ranges():
        starts = init_start_pos().to(torch.int64)
        begins = (starts // COMPRESS_RATIO) * COMPRESS_RATIO
        return torch.stack([begins, starts], dim=-1).to(torch.int32)
    def init_state_page_epochs():
        return torch.full(
            (batch, HCA_STATE_PAGES_PER_REQUEST), 7, dtype=torch.int32
        )
    def init_request_epochs():
        return torch.full((batch,), 7, dtype=torch.int32)
    def init_compress_state():
        state = torch.zeros(
            COMPRESS_STATE_BLOCK_NUM,
            COMPRESS_STATE_BLOCK_SIZE,
            COMPRESS_STATE_DIM,
        )
        generator = torch.Generator()
        generator.manual_seed(5504)
        pages = init_state_page_ids().to(torch.int64)
        ranges = init_state_valid_ranges().to(torch.int64)
        for request in range(batch):
            begin = int(ranges[request, 0].item())
            end = int(ranges[request, 1].item())
            for position in range(begin, end):
                ring_row = position % HCA_STATE_ROWS_PER_REQUEST
                relative_page = ring_row // COMPRESS_STATE_BLOCK_SIZE
                page = int(pages[request, relative_page].item())
                state[page, ring_row % COMPRESS_STATE_BLOCK_SIZE] = (
                    torch.randn(COMPRESS_STATE_DIM, generator=generator) * 0.02
                )
        return state
    # Calibrated to the real DeepSeek-V4-Flash 150
    #  (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0309
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0332
    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_request_event_indices():
        positions = init_position_ids().to(torch.int64)
        indices = torch.full((batch,), -1, dtype=torch.int32)
        event = 0
        for request in range(batch):
            if bool((((positions[request] + 1) % COMPRESS_RATIO) == 0).any()):
                indices[request] = event
                event += 1
        return indices
    def init_rope_positions():
        first_pos = init_position_ids().to(torch.int64)[:, 0]
        event_requests = torch.nonzero(
            init_request_event_indices() >= 0, as_tuple=False
        ).reshape(-1)
        return (
            first_pos[event_requests] // COMPRESS_RATIO * COMPRESS_RATIO
        ).to(torch.int64)
    def init_cos():
        cos, _ = token_local_rope(
            M,
            COMPRESS_RATIO,
            init_rope_positions(),
            rope_dim=ROPE_HEAD_DIM,
            dtype=torch.float32,
        )
        return cos[:, : ROPE_HEAD_DIM // 2].contiguous()
    def init_sin():
        _, sin = token_local_rope(
            M,
            COMPRESS_RATIO,
            init_rope_positions(),
            rope_dim=ROPE_HEAD_DIM,
            dtype=torch.float32,
        )
        return sin[:, : ROPE_HEAD_DIM // 2].contiguous()
    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_start_pos():
        return resolved_starts.clone()
    def init_position_ids():
        return resolved_positions.clone()
    def init_state_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        page_ids = init_state_page_ids().to(torch.int64)
        ring_rows = positions % HCA_STATE_ROWS_PER_REQUEST
        relative_pages = torch.div(
            ring_rows, COMPRESS_STATE_BLOCK_SIZE, rounding_mode="floor"
        )
        pages = torch.gather(page_ids, 1, relative_pages)
        return pages * COMPRESS_STATE_BLOCK_SIZE + ring_rows % COMPRESS_STATE_BLOCK_SIZE
    def init_cmp_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        slots = torch.full((batch, S), -1, dtype=torch.int64)
        physical_page = 0
        for request in range(batch):
            candidate_rows = int(positions[request, -1].item() + 1) // COMPRESS_RATIO
            page_count = (candidate_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
            for local in range(S):
                position = int(positions[request, local].item())
                if (position + 1) % COMPRESS_RATIO == 0:
                    row = position // COMPRESS_RATIO
                    slots[request, local] = (
                        physical_page + row // BLOCK_SIZE
                    ) * BLOCK_SIZE + row % BLOCK_SIZE
            physical_page += page_count
        return slots
    return [
        TensorSpec("x", [batch * S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [batch * S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("state_page_ids", [batch, HCA_STATE_PAGES_PER_REQUEST], torch.int32, init_value=init_state_page_ids),
        TensorSpec("state_valid_ranges", [batch, 2], torch.int32, init_value=init_state_valid_ranges),
        TensorSpec("state_page_epochs", [batch, HCA_STATE_PAGES_PER_REQUEST], torch.int32, init_value=init_state_page_epochs),
        TensorSpec("request_epochs", [batch], torch.int32, init_value=init_request_epochs),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("request_event_indices", [batch], torch.int32, init_value=init_request_event_indices),
        TensorSpec("cos", [int((init_request_event_indices() >= 0).sum()), ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [int((init_request_event_indices() >= 0).sum()), ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("cmp_kv_cache", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache, is_output=True),
        TensorSpec("position_ids", [batch * S], torch.int32, init_value=lambda: init_position_ids().reshape(-1)),
        TensorSpec("cmp_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_cmp_slot_mapping().reshape(-1)),
        TensorSpec("state_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_state_slot_mapping().reshape(-1)),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count up to {B} (the compile-time "
                             "upper bound). The batch axes are pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch HCA set that includes the 8k point.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos, batch=args.batch),
        golden_fn=golden_compressor,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        # Precision reference: AscendC torch.ops.custom.compressor —
        # ops-transformer/experimental/attention/compressor/tests/pytest/compressor_golden.py
        compare_fn={
            "kv":            ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache":   ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
