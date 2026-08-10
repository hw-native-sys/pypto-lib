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

from rope_interleave import rope_interleave
from config import (
    FLASH as M,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    DECODE_CMP_BLOCK_NUM,
    FP32_NEG_INF,
    KV_CMP_MAX_BLOCKS,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("COMPRESSOR_R128_B_DYN")
T_DYN = pl.dynamic("COMPRESSOR_R128_T_DYN")  # T = B * S
S_DYN = pl.dynamic("S_DYN")
COMPRESS_STATE_MAX_BLOCKS_DYN = pl.dynamic("COMPRESS_STATE_MAX_BLOCKS_DYN")
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("COMPRESSOR_R128_CMP_BLOCK_NUM_DYN")

# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-128 non-overlap compressor)
COMPRESS_RATIO = 128
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COFF = 1
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
# Paged read contract:
# - compress_state_block_table is still used to read the historical ratio window
#   by absolute token position.
# - Persistent writes are explicit token-major contracts:
#     state_slot_mapping[b, s] -> flattened compressor-state row, -1 means no-write
#     cmp_slot_mapping[b, s]   -> flattened compressed-KV row, -1 means no-write
# - APE remains ratio-local:
#     ape_row = position_ids[b, s] % COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
# Logical state block tables cover MAX_SEQ_LEN while the physical state pool
# remains bounded to the per-request rolling state capacity.
COMPRESS_STATE_PHYSICAL_BLOCKS = 64
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
if IDX_KV_LEN > CMP_MAX_BLOCKS * BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
B_TILE = 8
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)
RMS_PAD_BLOCKS = (B + RMS_PAD_TILE - 1) // RMS_PAD_TILE
RMS_PAD_ROWS = RMS_PAD_BLOCKS * RMS_PAD_TILE
# softmax_pool reduces over the state axis with column reductions (no transpose), so it can
# afford a wider head tile than HEAD_TILE: each wider tile loads each state block fewer times
# (HEAD_DIM/POOL_HEAD_TILE tiles/batch instead of HEAD_DIM/HEAD_TILE), cutting load redundancy.
POOL_HEAD_TILE = 128
# Gather rows per softmax_pool iteration. Held independent of the state page size: the
# two [STATE_LEN, POOL_HEAD_TILE] FP32 pools already take 128 KB of the 184 KB Vec space,
# leaving room for one double-buffered [POOL_STATE_TILE, POOL_HEAD_TILE] pair.
POOL_STATE_TILE = min(COMPRESS_STATE_BLOCK_SIZE, 8)
POOL_PAGE_STEPS = COMPRESS_STATE_BLOCK_SIZE // POOL_STATE_TILE
POOL_STATE_STEPS = STATE_LEN // POOL_STATE_TILE


@pl.jit.inline
def compressor_ratio128(
    x: pl.Tensor,
    kv: pl.Tensor,
    compress_state: pl.Tensor,
    compress_state_block_table: pl.Tensor,
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor,
    sin: pl.Tensor,
    cmp_kv_cache: pl.Tensor,
    position_ids: pl.Tensor,
    cmp_slot_mapping: pl.Tensor,
    state_slot_mapping: pl.Tensor,
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)

    x_flat = x
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE  # ceil to whole 16-row cube tiles
    rms_blocks = (b_dim + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)

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
    pooled_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    # The target decode point is start_pos=8192, where the ratio-128 boundary branch
    # is inactive. Keep scatter and all pool gates in one task to minimize dispatches.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool") as pool_tid:
        for global_c_idx in pl.range(b_dim):
            for s_sc in pl.pipeline(s_dim, stage=2):
                proj_row = global_c_idx * s_dim + s_sc
                token_pos = pl.read(position_ids, [global_c_idx * s_dim + s_sc])
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                state_row_i64 = pl.read(state_slot_mapping, [global_c_idx * s_dim + s_sc])
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, target_type=pl.INDEX)
                    kv_row = kv_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    score_row = score_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    compress_state_rows[state_row : state_row + 1, 0 : OUT_DIM] = kv_row
                    compress_state_rows[
                        state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM
                    ] = pl.add(score_row, ape_row)

            first_pos_gate = pl.read(position_ids, [global_c_idx * s_dim])
            pos_gate = first_pos_gate % COMPRESS_RATIO
            if pos_gate + s_dim >= COMPRESS_RATIO:
                compress_pos = first_pos_gate + (COMPRESS_RATIO - 1 - pos_gate)
                state_pos0 = compress_pos - (COMPRESS_RATIO - 1)
                base_logical_blk = state_pos0 // COMPRESS_STATE_BLOCK_SIZE
                for h0 in pl.range(0, HEAD_DIM, POOL_HEAD_TILE):
                    softmax_score_state = pl.create_tensor(
                        [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
                    )
                    softmax_kv_state = pl.create_tensor(
                        [STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32
                    )
                    for gather_i in pl.pipeline(POOL_STATE_STEPS, stage=2):
                        blk_i = gather_i // POOL_PAGE_STEPS
                        intra0 = (gather_i % POOL_PAGE_STEPS) * POOL_STATE_TILE
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
                        state_blk_raw = pl.read(
                            compress_state_block_table,
                            [global_c_idx, base_logical_blk + blk_i],
                        )
                        if state_blk_raw >= 0:
                            state_blk_id = pl.cast(state_blk_raw, target_type=pl.INDEX)
                            row0 = state_blk_id * COMPRESS_STATE_BLOCK_SIZE + intra0
                            slot_score = compress_state_rows[
                                row0 : row0 + POOL_STATE_TILE,
                                OUT_DIM + h0 : OUT_DIM + h0 + POOL_HEAD_TILE,
                            ]
                            slot_kv = compress_state_rows[
                                row0 : row0 + POOL_STATE_TILE,
                                h0 : h0 + POOL_HEAD_TILE,
                            ]
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

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
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
            cos_b[local_c_idx : local_c_idx + 1, 0 : ROPE_HEAD_DIM] = cos[
                row_c_idx : row_c_idx + 1, 0 : ROPE_HEAD_DIM
            ]
            sin_b[local_c_idx : local_c_idx + 1, 0 : ROPE_HEAD_DIM] = sin[
                row_c_idx : row_c_idx + 1, 0 : ROPE_HEAD_DIM
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
                    cmp_row = pl.cast(cmp_row_i64, target_type=pl.INDEX)
                    kv_flat[global_c_idx * s_dim : global_c_idx * s_dim + 1, :] = kv_row
                    cmp_kv_cache_flat[cmp_row : cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")

    return kv


@pl.jit
def compressor_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(1, COMPRESS_STATE_MAX_BLOCKS_DYN)
    cos.bind_dynamic(0, B_DYN)
    sin.bind_dynamic(0, B_DYN)
    cmp_kv_cache.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)

    late_dep = pl.system.task_dummy(deps=[])
    # The fused path builds these once in hca_rope; standalone does the same prep here
    # so the fixture / golden keep the half-width cos/sin ABI.
    cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cos, sin, cos_il, sin_signed)
    compressor_ratio128(
        x, kv, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w,
        cos_il, sin_signed,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping, late_dep,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: compress_state (kv + score channels merged) and cmp_kv_cache,
    each addressed via the corresponding block_table.
    """
    import torch

    x = tensors["x"].float()
    compress_state_block_table = tensors["compress_state_block_table"]
    position_ids = tensors["position_ids"].to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].to(torch.int64)
    # Historical state reads still use absolute-position block-table addressing.
    # Persistent writes use token-major slot mappings. APE remains modulo ratio.
    compress_state = tensors["compress_state"]

    def read_state_row(b, pos):
        logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
        intra = pos % COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        if sblk < 0:
            return (
                torch.zeros(OUT_DIM, dtype=torch.float32, device=compress_state.device),
                torch.full((OUT_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device),
            )
        return (
            compress_state[sblk, intra, :OUT_DIM],
            compress_state[sblk, intra, OUT_DIM:2 * OUT_DIM],
        )

    def write_state_row(slot, kv_row, score_row):
        if slot < 0:
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
    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        boundary_s = None
        for s in range(S):
            pos = int(position_ids[b, s].item())
            token_ape_row = pos % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            write_state_row(int(state_slot_mapping[b, s].item()), kv[b, s, :], score[b, s, :])
            if (pos + 1) % ratio == 0:
                boundary_s = s

        if boundary_s is not None:
            should_compress_rows[b] = True
            compress_pos = int(position_ids[b, boundary_s].item())
            kv_rows = []
            score_rows = []
            for pos in range(compress_pos - ratio + 1, compress_pos + 1):
                kv_row, score_row = read_state_row(b, pos)
                kv_rows.append(kv_row)
                score_rows.append(score_row)
            kv_state = torch.stack(kv_rows, dim=0).unsqueeze(0)
            score_state = torch.stack(score_rows, dim=0).unsqueeze(0)
            pooled[b : b + 1] = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
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
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        boundary_positions = torch.nonzero((position_ids[b, :S] + 1) % ratio == 0, as_tuple=False).flatten()
        if int(boundary_positions.numel()) == 0:
            continue
        boundary_s = int(boundary_positions[0].item())
        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            # Kernel writes committed pooled result only to kv[:, 0, :]; leave
            # speculative-boundary rows and kv[:, 1:, :] zero-initialized.
            tensors["kv"][b * S : b * S + 1, :] = kv_b[0]
            cblk = cmp_row // BLOCK_SIZE
            intra_offset = cmp_row % BLOCK_SIZE
            cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(start_pos=None, batch=B):
    import torch  # type: ignore[import]
    from utils import (
        block_table,
        compressed_slot_mapping,
        hca_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
    from golden import TensorSpec
    from utils import build_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(batch * S, D)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
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
    def init_rope_positions():
        first_pos = init_position_ids().to(torch.int64)[:, 0]
        cmp_offset = COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)
        return (first_pos + cmp_offset - COMPRESS_RATIO).to(torch.int64)
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]
    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_compress_state_block_table():
        return block_table(
            batch=batch,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
            permuted=True,
        )
    def init_cmp_block_table():
        return block_table(
            batch=batch,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_MAX_BLOCKS,
            permuted=True,
        )
    def init_default_start_pos():
        # Canonical HCA start-position set (ratio-128 compressor branches + 8k long-context).
        return hca_decode_start_set(
            batch=batch, compress_ratio=COMPRESS_RATIO, state_block_size=COMPRESS_STATE_BLOCK_SIZE)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=batch,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)
    def init_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_compress_state_block_table(),
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        )
    def init_cmp_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        )
    return [
        TensorSpec("x", [batch * S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [batch * S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [batch, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [batch, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [batch, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
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
                        help=f"runtime request count; a multiple of 4 up to {B} (the compile-time "
                             "upper bound). The batch axes are pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch HCA set that includes the 8k point.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 4 or args.batch > B or args.batch % 4 != 0:
        parser.error(f"--batch must be a multiple of 4 in [4, {B}], got {args.batch}")

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
