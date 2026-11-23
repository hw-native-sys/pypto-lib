# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode KV compressors and their numerical references."""

import pypto.language as pl


if __package__:
    from .kv_proj_rope import rope_interleave
else:
    from kv_proj_rope import rope_interleave
if __package__:
    from .decode_shapes import DECODE_CONFIG as M
    from .config import FP32_NEG_INF
else:
    from decode_shapes import DECODE_CONFIG as M
    from config import FP32_NEG_INF
if __package__:
    from .decode_shapes import (
        DECODE_BATCH,
        DECODE_SEQ,
        BLOCK_SIZE,
        C4A_COMPRESSOR_BLOCK_SIZE,
        DECODE_CMP_BLOCK_NUM,
        KV_CMP_MAX_BLOCKS,
    )
else:
    from decode_shapes import DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, KV_CMP_MAX_BLOCKS


# model config
B = DECODE_BATCH
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
RATIO4_HEAD_DIM = M.head_dim
RATIO4_HEAD_DIM_INV = 1.0 / RATIO4_HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
RATIO4_NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
RATIO4_COMPRESS_RATIO = 4
RATIO4_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // RATIO4_COMPRESS_RATIO
RATIO4_OVERLAP = RATIO4_COMPRESS_RATIO == 4
RATIO4_COFF = 1 + int(RATIO4_OVERLAP)
RATIO4_OUT_DIM = RATIO4_COFF * RATIO4_HEAD_DIM
RATIO4_STATE_LEN = RATIO4_COFF * RATIO4_COMPRESS_RATIO
RATIO4_IDX_KV_LEN = MAX_SEQ_LEN // RATIO4_COMPRESS_RATIO
RATIO4_COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
RATIO4_COMPRESS_STATE_PHYSICAL_BLOCKS = 65
RATIO4_COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + RATIO4_COMPRESS_STATE_BLOCK_SIZE - 1) // RATIO4_COMPRESS_STATE_BLOCK_SIZE
RATIO4_COMPRESS_STATE_BLOCK_NUM = RATIO4_COMPRESS_STATE_PHYSICAL_BLOCKS
RATIO4_COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_CSA_STATE_BLOCK_NUM_DYN')
RATIO4_COMPRESS_STATE_DIM = 2 * RATIO4_OUT_DIM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
CMP_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_CMP_BLOCK_NUM_DYN')

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
B_TILE = 8
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RATIO4_RMS_PAD_TILE = BS_PAD  # token-indexed rows, padded to a Cube M multiple of 16
RATIO4_RMS_PAD_ROWS = RATIO4_RMS_PAD_TILE


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, RATIO4_HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [RATIO4_COMPRESS_STATE_BLOCK_NUM_DYN, RATIO4_COMPRESS_STATE_BLOCK_SIZE, RATIO4_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B, RATIO4_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[RATIO4_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[RATIO4_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[RATIO4_COMPRESS_RATIO, RATIO4_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[RATIO4_HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, RATIO4_CMP_STORAGE_BLOCK_SIZE, 1, RATIO4_HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    cmp4_kv_proj_pad = pl.create_tensor([BS_PAD, RATIO4_OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([BS_PAD, RATIO4_OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)
    compress_state_flat = pl.reshape(
        compress_state, [compress_state_block_num * RATIO4_COMPRESS_STATE_BLOCK_SIZE, RATIO4_COMPRESS_STATE_DIM]
    )
    kv_flat = pl.reshape(kv, [B * S, RATIO4_HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_block_num * RATIO4_CMP_STORAGE_BLOCK_SIZE, RATIO4_HEAD_DIM])

    with pl.spmd(
        BS_PAD * RATIO4_OUT_DIM // (MM_B_TILE * OUT_TILE),
        name_hint="kv_score_proj",
        allow_early_resolve=True,
        deps=[late_dep],
    ) as _kv_score_proj_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (RATIO4_OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (RATIO4_OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, OUT_TILE] (K strided = many short
            # bursts). Cuts the transaction-bound MTE2 cost ~14% busy / ~7% compressor wall.
            wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(kb == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(kb == 0))

        cmp4_kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        cmp4_score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    # Pool each boundary from immutable history plus this invocation's projections.
    pooled_kv = pl.create_tensor([BS_PAD, RATIO4_HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool", allow_early_resolve=True
    ) as pool_tid:
        pooled_kv[:, :] = pl.full([BS_PAD, RATIO4_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c_idx in pl.range(B):
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            for s_idx in pl.range(S):
                token = c_idx * S + s_idx
                token_pos = pl.read(position_ids, [c_idx, s_idx])
                if (token_pos + 1) % RATIO4_COMPRESS_RATIO == 0:
                    window_start = token_pos - RATIO4_STATE_LEN + 1
                    for h0 in pl.range(0, RATIO4_HEAD_DIM, HEAD_TILE):
                        last_ape = pl.cast(token_pos % RATIO4_COMPRESS_RATIO, pl.INDEX)
                        last_score = cmp4_score_proj_pad[
                            token : token + 1, RATIO4_HEAD_DIM + h0 : RATIO4_HEAD_DIM + h0 + HEAD_TILE
                        ]
                        last_bias = ape[last_ape : last_ape + 1, RATIO4_HEAD_DIM + h0 : RATIO4_HEAD_DIM + h0 + HEAD_TILE]
                        mi = pl.add(last_score, last_bias)
                        initial_delta = pl.sub(mi, mi)
                        li = pl.exp(initial_delta)
                        oi = cmp4_kv_proj_pad[token : token + 1, RATIO4_HEAD_DIM + h0 : RATIO4_HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(RATIO4_STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            state_half = 0
                            if state_idx >= RATIO4_COMPRESS_RATIO:
                                state_half = RATIO4_HEAD_DIM
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                page = logical_pos // RATIO4_COMPRESS_STATE_BLOCK_SIZE
                                physical = pl.read(compress_state_block_table, [c_idx, page])
                                if physical >= 0:
                                    state_row = (
                                        physical * RATIO4_COMPRESS_STATE_BLOCK_SIZE
                                        + logical_pos % RATIO4_COMPRESS_STATE_BLOCK_SIZE
                                    )
                                    col = state_half + h0
                                    value = compress_state_flat[
                                        state_row : state_row + 1, col : col + HEAD_TILE
                                    ]
                                    score_col = RATIO4_OUT_DIM + col
                                    score = compress_state_flat[
                                        state_row : state_row + 1, score_col : score_col + HEAD_TILE
                                    ]
                            if logical_pos >= first_pos_b and logical_pos <= token_pos:
                                overlay = c_idx * S + logical_pos - first_pos_b
                                ape_row = pl.cast(logical_pos % RATIO4_COMPRESS_RATIO, pl.INDEX)
                                col = state_half + h0
                                value = cmp4_kv_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
                                score_base = cmp4_score_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
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
                commit_state_row_i64 = pl.read(state_slot_mapping, [c_idx, s_idx])
                if commit_state_row_i64 >= 0:
                    commit_state_row = pl.cast(commit_state_row_i64, pl.INDEX)
                    token = c_idx * S + s_idx
                    token_pos = pl.read(position_ids, [c_idx, s_idx])
                    commit_ape_row = pl.cast(token_pos % RATIO4_COMPRESS_RATIO, pl.INDEX)
                    compress_state_flat[commit_state_row : commit_state_row + 1, 0:RATIO4_OUT_DIM] = (
                        cmp4_kv_proj_pad[token : token + 1, :]
                    )
                    commit_score_base = cmp4_score_proj_pad[token : token + 1, :]
                    commit_score_bias = ape[commit_ape_row : commit_ape_row + 1, :]
                    compress_state_flat[
                        commit_state_row : commit_state_row + 1, RATIO4_OUT_DIM:RATIO4_COMPRESS_STATE_DIM
                    ] = pl.add(commit_score_base, commit_score_bias)

    normed_kv = pl.create_tensor([RATIO4_RMS_PAD_ROWS, RATIO4_HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, RATIO4_HEAD_DIM])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="rmsnorm_rope_cache_write",
        deps=[pool_tid],
        allow_early_resolve=True,
    ):
        # The first B*S rows hold token results; remaining rows are padding.
        cos_b = pl.full([RATIO4_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RATIO4_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_b[0 : B * S, 0:ROPE_HEAD_DIM] = cos[0 : B * S, 0:ROPE_HEAD_DIM]
        sin_b[0 : B * S, 0:ROPE_HEAD_DIM] = sin[0 : B * S, 0:ROPE_HEAD_DIM]
        partial_sq = pl.full([1, RATIO4_RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, RATIO4_HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[0:RATIO4_RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RATIO4_RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, RATIO4_HEAD_DIM_INV), EPS), [RATIO4_RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, RATIO4_NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[0:RATIO4_RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0:RATIO4_RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[0:RATIO4_RMS_PAD_TILE, RATIO4_NOPE_HEAD_DIM:RATIO4_HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, RATIO4_NOPE_HEAD_DIM:RATIO4_HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rope_fused in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is FP32 -> write directly.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RATIO4_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
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
        normed_kv[0:RATIO4_RMS_PAD_TILE, RATIO4_NOPE_HEAD_DIM:RATIO4_HEAD_DIM] = rope_rot

        # cache write: reads back only this block's own normed_kv rows, so the normed_kv
        # RAW is intra-block -- no separate scope / cross-task barrier needed.
        for token in pl.range(B * S):
            c_idx = token // S
            s_idx = token - c_idx * S
            position = pl.read(position_ids, [c_idx, s_idx])
            cache_row_i64 = pl.read(cmp_slot_mapping, [c_idx, s_idx])
            if (position + 1) % RATIO4_COMPRESS_RATIO == 0 and cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_row_fp32 = normed_kv[token : token + 1, :]
                kv_flat[token : token + 1, :] = kv_row_fp32
                cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(kv_row_fp32, pl.BF16, mode="rint")

    return kv


@pl.jit
def compressor_test_ratio4(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, RATIO4_HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[RATIO4_COMPRESS_STATE_BLOCK_NUM_DYN, RATIO4_COMPRESS_STATE_BLOCK_SIZE, RATIO4_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, RATIO4_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[RATIO4_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[RATIO4_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[RATIO4_COMPRESS_RATIO, RATIO4_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[RATIO4_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, RATIO4_CMP_STORAGE_BLOCK_SIZE, 1, RATIO4_HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    # The fused path builds these once in csa_cmp_rope; standalone does the same prep
    # here so the fixture / golden keep the half-width cos/sin ABI.
    cos_il = pl.create_tensor([B * S, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([B * S, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cos, sin, cos_il, sin_signed, B * S)
    compressor_ratio4(
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
        cmp_kv_cache,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
        late_dep,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor_ratio4(tensors):
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
    cmp_kv_cache = tensors["cmp_kv_cache"]
    position_ids = tensors["position_ids"].to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].to(torch.int64)
    bsz, _, _ = x.shape
    ratio, rd = RATIO4_COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = bf16_matmul_fp32(x, wkv.t())  # [B, S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = bf16_matmul_fp32(x, wgate.t())  # [B, S, OUT_DIM]

    # Each boundary stays at its input token index, as in the device output.
    pooled = torch.zeros(bsz, S, RATIO4_HEAD_DIM, dtype=torch.float32, device=x.device)
    visible_boundaries = []
    score = score + ape[position_ids % ratio]

    def read_pool_row(b, abs_pos, half):
        if abs_pos < 0:
            return (
                torch.zeros(RATIO4_HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((RATIO4_HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        first_pos = int(position_ids[b, 0].item())
        channel = half * RATIO4_HEAD_DIM
        if abs_pos >= first_pos:
            token = abs_pos - first_pos
            return kv[b, token, channel : channel + RATIO4_HEAD_DIM], score[b, token, channel : channel + RATIO4_HEAD_DIM]
        blk_id = int(compress_state_block_table[b, abs_pos // RATIO4_COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(RATIO4_HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((RATIO4_HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % RATIO4_COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, channel : channel + RATIO4_HEAD_DIM],
            compress_state[blk_id, intra, RATIO4_OUT_DIM + channel : RATIO4_OUT_DIM + channel + RATIO4_HEAD_DIM],
        )

    # Read every prefix window before a commit can overwrite rolling physical state.
    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        first_boundary_s = ratio - 1 - first_pos % ratio
        for boundary_s in range(first_boundary_s, S, ratio):
            if int(cmp_slot_mapping[b, boundary_s].item()) < 0:
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
            state_row = int(state_slot_mapping[b, token].item())
            if state_row >= 0:
                blk_id = state_row // RATIO4_COMPRESS_STATE_BLOCK_SIZE
                intra = state_row % RATIO4_COMPRESS_STATE_BLOCK_SIZE
                compress_state[blk_id, intra, :RATIO4_OUT_DIM] = kv[b, token]
                compress_state[blk_id, intra, RATIO4_OUT_DIM:] = score[b, token]

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

        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            tensors["kv"][b : b + 1, boundary_s : boundary_s + 1, :] = kv_b
            blk_id = cmp_row // RATIO4_CMP_STORAGE_BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % RATIO4_CMP_STORAGE_BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs_ratio4(start_pos=None):
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

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, RATIO4_COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)

    def init_compress_state():
        state = torch.zeros(RATIO4_COMPRESS_STATE_BLOCK_NUM, RATIO4_COMPRESS_STATE_BLOCK_SIZE, RATIO4_COMPRESS_STATE_DIM)
        state[:, :, RATIO4_OUT_DIM:] = FP32_NEG_INF
        return state

    def init_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=RATIO4_COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=RATIO4_COMPRESS_STATE_PHYSICAL_BLOCKS,
        )

    # Calibrated to the real DeepSeek-V4-Flash CSA (ratio-4) main compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(RATIO4_OUT_DIM, D) * 0.0245

    def init_wgate():
        return torch.randn(RATIO4_OUT_DIM, D) * 0.0388

    def init_ape():
        return torch.randn(RATIO4_COMPRESS_RATIO, RATIO4_OUT_DIM) * 0.1243

    def init_norm_w():
        return 0.9666 + 0.1929 * torch.randn(RATIO4_HEAD_DIM)

    def init_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundaries = (positions + 1) % RATIO4_COMPRESS_RATIO == 0
        return torch.where(boundaries, positions - RATIO4_COMPRESS_RATIO + 1, 0).reshape(-1)

    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]

    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]

    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, RATIO4_CMP_STORAGE_BLOCK_SIZE, 1, RATIO4_HEAD_DIM)

    def init_cmp_block_table():
        return block_table(
            batch=B,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )

    def init_default_start_pos():
        # Canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k).
        return csa_decode_start_set(
            batch=B, seq=S, compress_ratio=RATIO4_COMPRESS_RATIO, state_block_size=RATIO4_COMPRESS_STATE_BLOCK_SIZE
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

    def init_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_compress_state_block_table(),
            state_block_size=RATIO4_COMPRESS_STATE_BLOCK_SIZE,
        )

    def init_cmp_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=RATIO4_COMPRESS_RATIO,
            block_size=RATIO4_CMP_STORAGE_BLOCK_SIZE,
        )

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, RATIO4_HEAD_DIM], torch.float32),
        TensorSpec(
            "compress_state",
            [RATIO4_COMPRESS_STATE_BLOCK_NUM, RATIO4_COMPRESS_STATE_BLOCK_SIZE, RATIO4_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [B, RATIO4_COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [RATIO4_OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [RATIO4_OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [RATIO4_COMPRESS_RATIO, RATIO4_OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [RATIO4_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec(
            "cmp_kv_cache",
            [CMP_BLOCK_NUM, RATIO4_CMP_STORAGE_BLOCK_SIZE, 1, RATIO4_HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv_cache,
        ),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [B, S], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [B, S], torch.int64, init_value=init_state_slot_mapping),
    ]


if __package__:
    from .kv_proj_rope import rope_interleave
else:
    from kv_proj_rope import rope_interleave
if __package__:
    from .decode_shapes import DECODE_CONFIG as M
    from .config import FP32_NEG_INF
else:
    from decode_shapes import DECODE_CONFIG as M
    from config import FP32_NEG_INF
if __package__:
    from .decode_shapes import (
        BLOCK_SIZE,
        C128_COMPRESSOR_BLOCK_SIZE,
        DECODE_BATCH,
        DECODE_SEQ,
        DECODE_CMP_BLOCK_NUM,
        KV_CMP_MAX_BLOCKS,
    )
else:
    from decode_shapes import BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ, DECODE_CMP_BLOCK_NUM, KV_CMP_MAX_BLOCKS
if __package__:
    from .utils import DECODE_START_POS
else:
    from utils import DECODE_START_POS

# Dynamic shape variables.
B_DYN = pl.dynamic('LOWLAT_TARGET_B_DYN')
S_DYN = pl.dynamic('LOWLAT_TARGET_S_DYN')
COMPRESS_STATE_MAX_BLOCKS_DYN = pl.dynamic('LOWLAT_TARGET_COMPRESS_STATE_MAX_BLOCKS_DYN')
RATIO128_COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_HCA_STATE_BLOCK_NUM_DYN')

# model config
RATIO128_HEAD_DIM = M.head_dim
RATIO128_HEAD_DIM_INV = 1.0 / RATIO128_HEAD_DIM
RATIO128_NOPE_HEAD_DIM = M.nope_head_dim

# kernel-local (ratio-128 non-overlap compressor)
RATIO128_COMPRESS_RATIO = 128
RATIO128_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // RATIO128_COMPRESS_RATIO
RATIO128_IDX_KV_LEN = MAX_SEQ_LEN // RATIO128_COMPRESS_RATIO
RATIO128_COFF = 1
RATIO128_OUT_DIM = RATIO128_COFF * RATIO128_HEAD_DIM
RATIO128_STATE_LEN = RATIO128_COFF * RATIO128_COMPRESS_RATIO
# Paged read contract:
# - compress_state_block_table is still used to read the historical ratio window
#   by absolute token position.
# - Persistent writes are explicit token-major contracts:
#     state_slot_mapping[b, s] -> flattened compressor-state row, -1 means no-write
#     cmp_slot_mapping[b, s]   -> flattened compressed-KV row, -1 means no-write
# - APE remains ratio-local:
#     ape_row = position_ids[b, s] % COMPRESS_RATIO
RATIO128_COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
# Logical state block tables cover MAX_SEQ_LEN while the physical state pool
# remains bounded to the per-request rolling state capacity.
RATIO128_COMPRESS_STATE_PHYSICAL_BLOCKS = 64
RATIO128_COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + RATIO128_COMPRESS_STATE_BLOCK_SIZE - 1) // RATIO128_COMPRESS_STATE_BLOCK_SIZE
RATIO128_COMPRESS_STATE_BLOCK_NUM = RATIO128_COMPRESS_STATE_PHYSICAL_BLOCKS
RATIO128_COMPRESS_STATE_DIM = 2 * RATIO128_OUT_DIM
if RATIO128_IDX_KV_LEN > CMP_MAX_BLOCKS * RATIO128_CMP_STORAGE_BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# tiling
RATIO128_RMS_PAD_TILE = 16
RATIO128_RMS_PAD_ROWS = RATIO128_RMS_PAD_TILE
assert B <= RATIO128_RMS_PAD_TILE
# softmax_pool reduces over the state axis with column reductions (no transpose), so it can
# afford a wider head tile than HEAD_TILE: each wider tile loads each state block fewer times
# (HEAD_DIM/POOL_HEAD_TILE tiles/batch instead of HEAD_DIM/HEAD_TILE), cutting load redundancy.
POOL_HEAD_TILE = 128


@pl.jit.inline
def compressor_ratio128(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Tensor[[B_DYN, S_DYN, RATIO128_HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [RATIO128_COMPRESS_STATE_BLOCK_NUM_DYN, RATIO128_COMPRESS_STATE_BLOCK_SIZE, RATIO128_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS_DYN], pl.INT32],
    wkv: pl.Tensor[[RATIO128_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[RATIO128_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[RATIO128_COMPRESS_RATIO, RATIO128_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[RATIO128_HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, RATIO128_CMP_STORAGE_BLOCK_SIZE, 1, RATIO128_HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[B_DYN, S_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    b_dim = pl.tensor.dim(x, 0)
    s_dim = pl.tensor.dim(x, 1)
    bs = b_dim * s_dim
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)

    x_flat = pl.reshape(x, [bs, D])
    t_matmul = pl.max(bs, MM_B_TILE)
    kv_proj_pad = pl.create_tensor([BS_PAD, RATIO128_OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, RATIO128_OUT_DIM], dtype=pl.FP32)

    with pl.spmd(
        t_matmul * RATIO128_OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint="kv_score_proj", allow_early_resolve=True
    ):
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (RATIO128_OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (RATIO128_OUT_DIM // OUT_TILE)) * OUT_TILE
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
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(kb == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(kb == 0))

        kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

    compress_state_rows_num = compress_state_block_num * RATIO128_COMPRESS_STATE_BLOCK_SIZE
    compress_state_rows = pl.reshape(compress_state, [compress_state_rows_num, RATIO128_COMPRESS_STATE_DIM])
    pooled_kv = pl.create_tensor([RATIO128_RMS_PAD_ROWS, RATIO128_HEAD_DIM], dtype=pl.FP32)
    staged_pool_kv = pl.create_tensor([b_dim * RATIO128_STATE_LEN, RATIO128_HEAD_DIM], dtype=pl.FP32)
    # Pool the old window with current projections before committing this chunk:
    # later speculative slots may alias rows still needed at the boundary.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool", allow_early_resolve=True
    ) as pool_tid:
        for global_c_idx in pl.range(b_dim):
            first_pos_gate = pl.read(position_ids, [global_c_idx, 0])
            pos_gate = first_pos_gate % RATIO128_COMPRESS_RATIO
            if pos_gate + s_dim >= RATIO128_COMPRESS_RATIO:
                compress_pos = first_pos_gate + (RATIO128_COMPRESS_RATIO - 1 - pos_gate)
                state_pos0 = compress_pos - (RATIO128_COMPRESS_RATIO - 1)
                base_logical_blk = state_pos0 // RATIO128_COMPRESS_STATE_BLOCK_SIZE
                for h0 in pl.range(0, RATIO128_HEAD_DIM, POOL_HEAD_TILE):
                    softmax_score_state = pl.create_tensor([RATIO128_STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
                    softmax_kv_state = pl.create_tensor([RATIO128_STATE_LEN, POOL_HEAD_TILE], dtype=pl.FP32)
                    for blk_i in pl.pipeline(RATIO128_STATE_LEN // RATIO128_COMPRESS_STATE_BLOCK_SIZE, stage=2):
                        s0 = blk_i * RATIO128_COMPRESS_STATE_BLOCK_SIZE
                        slot_score = pl.full(
                            [RATIO128_COMPRESS_STATE_BLOCK_SIZE, POOL_HEAD_TILE],
                            dtype=pl.FP32,
                            value=FP32_NEG_INF,
                        )
                        slot_kv = pl.full(
                            [RATIO128_COMPRESS_STATE_BLOCK_SIZE, POOL_HEAD_TILE],
                            dtype=pl.FP32,
                            value=0.0,
                        )
                        state_blk_raw = pl.read(
                            compress_state_block_table,
                            [global_c_idx, base_logical_blk + blk_i],
                        )
                        if state_blk_raw >= 0:
                            state_blk_id = pl.cast(state_blk_raw, target_type=pl.INDEX)
                            row0 = state_blk_id * RATIO128_COMPRESS_STATE_BLOCK_SIZE
                            slot_score = compress_state_rows[
                                row0 : row0 + RATIO128_COMPRESS_STATE_BLOCK_SIZE,
                                RATIO128_OUT_DIM + h0 : RATIO128_OUT_DIM + h0 + POOL_HEAD_TILE,
                            ]
                            slot_kv = compress_state_rows[
                                row0 : row0 + RATIO128_COMPRESS_STATE_BLOCK_SIZE,
                                h0 : h0 + POOL_HEAD_TILE,
                            ]
                        softmax_score_state[s0 : s0 + RATIO128_COMPRESS_STATE_BLOCK_SIZE, :] = slot_score
                        softmax_kv_state[s0 : s0 + RATIO128_COMPRESS_STATE_BLOCK_SIZE, :] = slot_kv

                    # Current projections contribute even when their persistent slot is invalid.
                    for source_s in pl.range(s_dim):
                        source_pos = pl.read(position_ids, [global_c_idx, source_s])
                        if source_pos >= state_pos0 and source_pos <= compress_pos:
                            pool_row = pl.cast(source_pos - state_pos0, pl.INDEX)
                            source_row = global_c_idx * s_dim + source_s
                            source_ape = pl.cast(source_pos % RATIO128_COMPRESS_RATIO, pl.INDEX)
                            current_kv = kv_proj_pad[source_row : source_row + 1, h0 : h0 + POOL_HEAD_TILE]
                            current_score = score_proj_pad[source_row : source_row + 1, h0 : h0 + POOL_HEAD_TILE]
                            current_bias = ape[source_ape : source_ape + 1, h0 : h0 + POOL_HEAD_TILE]
                            softmax_kv_state[pool_row : pool_row + 1, :] = current_kv
                            softmax_score_state[pool_row : pool_row + 1, :] = pl.add(current_score, current_bias)

                    stage_row = global_c_idx * RATIO128_STATE_LEN
                    staged_pool_kv[stage_row : stage_row + RATIO128_STATE_LEN, h0 : h0 + POOL_HEAD_TILE] = softmax_kv_state
                    score_max = pl.col_max(softmax_score_state)
                    score_exp = pl.col_expand_expdif(softmax_score_state, score_max)
                    score_sum = pl.col_sum(score_exp)
                    score_prob = pl.col_expand_mul(score_exp, pl.recip(score_sum))
                    pool_values = staged_pool_kv[stage_row : stage_row + RATIO128_STATE_LEN, h0 : h0 + POOL_HEAD_TILE]
                    pooled_chunk = pl.col_sum(pl.mul(pool_values, score_prob))
                    pooled_kv[global_c_idx : global_c_idx + 1, h0 : h0 + POOL_HEAD_TILE] = pooled_chunk

            for s_sc in pl.pipeline(s_dim, stage=2):
                proj_row = global_c_idx * s_dim + s_sc
                token_pos = pl.read(position_ids, [global_c_idx, s_sc])
                token_ape_row = pl.cast(token_pos % RATIO128_COMPRESS_RATIO, target_type=pl.INDEX)
                state_row_i64 = pl.read(state_slot_mapping, [global_c_idx, s_sc])
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, target_type=pl.INDEX)
                    kv_row = kv_proj_pad[proj_row : proj_row + 1, 0:RATIO128_OUT_DIM]
                    score_row = score_proj_pad[proj_row : proj_row + 1, 0:RATIO128_OUT_DIM]
                    ape_row = ape[token_ape_row : token_ape_row + 1, 0:RATIO128_OUT_DIM]
                    compress_state_rows[state_row : state_row + 1, 0:RATIO128_OUT_DIM] = kv_row
                    compress_state_rows[state_row : state_row + 1, RATIO128_OUT_DIM:RATIO128_COMPRESS_STATE_DIM] = pl.add(
                        score_row, ape_row
                    )

    norm_w_2d = pl.reshape(norm_w, [1, RATIO128_HEAD_DIM])
    normed_kv = pl.create_tensor([RATIO128_RMS_PAD_ROWS, RATIO128_HEAD_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [bs, RATIO128_HEAD_DIM])
    cmp_flat_rows = cmp_block_num * RATIO128_CMP_STORAGE_BLOCK_SIZE
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_flat_rows, RATIO128_HEAD_DIM])

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="rmsnorm_rope_cache_write",
        deps=[pool_tid],
        allow_early_resolve=True,
    ):
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the full
        # ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        cos_b = pl.full([RATIO128_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RATIO128_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for global_c_idx in pl.range(b_dim):
            cos_b[global_c_idx : global_c_idx + 1, 0:ROPE_HEAD_DIM] = cos[
                global_c_idx : global_c_idx + 1, 0:ROPE_HEAD_DIM
            ]
            sin_b[global_c_idx : global_c_idx + 1, 0:ROPE_HEAD_DIM] = sin[
                global_c_idx : global_c_idx + 1, 0:ROPE_HEAD_DIM
            ]
        partial_sq = pl.full([1, RATIO128_RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(RATIO128_HEAD_DIM // HEAD_TILE, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv[0:RATIO128_RMS_PAD_TILE, rms_h0 : rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RATIO128_RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, RATIO128_HEAD_DIM_INV), EPS), [RATIO128_RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(RATIO128_NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = rms_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv[0:RATIO128_RMS_PAD_TILE, norm_h0 : norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0 : norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0:RATIO128_RMS_PAD_TILE, norm_h0 : norm_h0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[0:RATIO128_RMS_PAD_TILE, RATIO128_NOPE_HEAD_DIM:RATIO128_HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, RATIO128_NOPE_HEAD_DIM:RATIO128_HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rope_fused in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is FP32 -> write directly.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RATIO128_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
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
        normed_kv[0:RATIO128_RMS_PAD_TILE, RATIO128_NOPE_HEAD_DIM:RATIO128_HEAD_DIM] = rope_rot

        for global_c_idx in pl.range(b_dim):
            first_pos_b = pl.read(position_ids, [global_c_idx, 0])
            pos_b = first_pos_b % RATIO128_COMPRESS_RATIO
            if pos_b + s_dim >= RATIO128_COMPRESS_RATIO:
                boundary_s = RATIO128_COMPRESS_RATIO - 1 - pos_b
                kv_row = normed_kv[global_c_idx : global_c_idx + 1, 0:RATIO128_HEAD_DIM]
                cmp_row_i64 = pl.read(cmp_slot_mapping, [global_c_idx, boundary_s])
                if cmp_row_i64 >= 0:
                    cmp_row = pl.cast(cmp_row_i64, target_type=pl.INDEX)
                    kv_flat[global_c_idx * s_dim : global_c_idx * s_dim + 1, :] = kv_row
                    cmp_kv_cache_flat[cmp_row : cmp_row + 1, :] = pl.cast(
                        kv_row, target_type=pl.BF16, mode="rint"
                    )

    return kv


@pl.jit
def compressor_test_ratio128(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B_DYN, S_DYN, RATIO128_HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[RATIO128_COMPRESS_STATE_BLOCK_NUM_DYN, RATIO128_COMPRESS_STATE_BLOCK_SIZE, RATIO128_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS_DYN], pl.INT32],
    wkv: pl.Tensor[[RATIO128_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[RATIO128_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[RATIO128_COMPRESS_RATIO, RATIO128_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[RATIO128_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, RATIO128_CMP_STORAGE_BLOCK_SIZE, 1, RATIO128_HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[B_DYN, S_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
):
    x.bind_dynamic(0, B_DYN)
    x.bind_dynamic(1, S_DYN)
    kv.bind_dynamic(0, B_DYN)
    kv.bind_dynamic(1, S_DYN)
    compress_state.bind_dynamic(0, RATIO128_COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(1, COMPRESS_STATE_MAX_BLOCKS_DYN)
    cos.bind_dynamic(0, B_DYN)
    sin.bind_dynamic(0, B_DYN)
    cmp_kv_cache.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    position_ids.bind_dynamic(0, B_DYN)
    position_ids.bind_dynamic(1, S_DYN)
    cmp_slot_mapping.bind_dynamic(0, B_DYN)
    cmp_slot_mapping.bind_dynamic(1, S_DYN)
    state_slot_mapping.bind_dynamic(0, B_DYN)
    state_slot_mapping.bind_dynamic(1, S_DYN)

    late_dep = pl.system.task_dummy(deps=[])
    # The fused path builds these once in hca_rope; standalone does the same prep here
    # so the fixture / golden keep the half-width cos/sin ABI.
    cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cos, sin, cos_il, sin_signed, B)
    compressor_ratio128(
        x, kv, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w,
        cos_il, sin_signed,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping, late_dep,
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor_ratio128(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=128 non-overlap).

    Operates on paged caches: compress_state (kv + score channels merged) and cmp_kv_cache,
    each addressed via the corresponding block_table.
    """
    import torch

    if __package__:
        from .utils import bf16_matmul_fp32
    else:
        from utils import bf16_matmul_fp32

    # Rows this golden never writes stay NaN: ignored by the kv comparator.
    tensors["kv"].fill_(float("nan"))

    x = tensors["x"].float()
    compress_state_block_table = tensors["compress_state_block_table"]
    position_ids = tensors["position_ids"].to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].to(torch.int64)
    # Historical state reads still use absolute-position block-table addressing.
    # Persistent writes use token-major slot mappings. APE remains modulo ratio.
    compress_state = tensors["compress_state"]

    def read_state_row(b, pos):
        first_pos = int(position_ids[b, 0].item())
        if first_pos <= pos < first_pos + S:
            token = pos - first_pos
            return kv[b, token], score[b, token]
        logical_blk = pos // RATIO128_COMPRESS_STATE_BLOCK_SIZE
        intra = pos % RATIO128_COMPRESS_STATE_BLOCK_SIZE
        sblk = int(compress_state_block_table[b, logical_blk].item())
        if sblk < 0:
            return (
                torch.zeros(RATIO128_OUT_DIM, dtype=torch.float32, device=compress_state.device),
                torch.full((RATIO128_OUT_DIM,), float("-inf"), dtype=torch.float32, device=compress_state.device),
            )
        return (
            compress_state[sblk, intra, :RATIO128_OUT_DIM],
            compress_state[sblk, intra, RATIO128_OUT_DIM : 2 * RATIO128_OUT_DIM],
        )

    def write_state_row(slot, kv_row, score_row):
        if slot < 0:
            return
        sblk = slot // RATIO128_COMPRESS_STATE_BLOCK_SIZE
        intra = slot % RATIO128_COMPRESS_STATE_BLOCK_SIZE
        compress_state[sblk, intra, :RATIO128_OUT_DIM] = kv_row
        compress_state[sblk, intra, RATIO128_OUT_DIM : 2 * RATIO128_OUT_DIM] = score_row

    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    cmp_kv_cache = tensors["cmp_kv_cache"]
    bsz, _, _ = x.shape
    ratio, rd = RATIO128_COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = bf16_matmul_fp32(x, wkv.t())  # [B, S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = bf16_matmul_fp32(x, wgate.t())  # [B, S, OUT_DIM]
    pooled = torch.zeros(bsz, 1, RATIO128_HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        boundary_s = None
        for s in range(S):
            pos = int(position_ids[b, s].item())
            token_ape_row = pos % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
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
        for token in range(S):
            write_state_row(int(state_slot_mapping[b, token].item()), kv[b, token], score[b, token])

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
            # speculative-boundary rows and kv[:, 1:, :] NaN (ignored).
            tensors["kv"][b : b + 1, 0:1, :] = kv_b
            cblk = cmp_row // RATIO128_CMP_STORAGE_BLOCK_SIZE
            intra_offset = cmp_row % RATIO128_CMP_STORAGE_BLOCK_SIZE
            cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache



def compare_compressor_kv(actual, expected, *, inputs, **kwargs):
    """Leave pure-Out rows unconstrained only for an input-proved empty write set."""
    import torch
    from golden import ratio_allclose

    if actual.shape != expected.shape:
        return False, "Compressor KV output shapes differ"
    positions = inputs.get("position_ids")
    mapping = inputs.get("cmp_slot_mapping")
    if positions is None or mapping is None or tuple(positions.shape) != (B, S):
        return False, "Missing or incompatible compressor position/mapping inputs"
    if positions.shape != mapping.shape:
        return False, "Compressor position and slot-mapping shapes differ"
    boundaries = (positions.cpu().to(torch.int64) + 1).remainder(RATIO128_COMPRESS_RATIO) == 0
    writes = boundaries & (mapping.cpu() >= 0)
    if not bool(writes.any()):
        if not bool(torch.isnan(expected).all()):
            return False, "Empty compressor write set must leave the pure-Out golden undefined"
        return True, ""
    return ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0, ignore_nan=True)(
        actual, expected, inputs=inputs, **kwargs,
    )

def build_tensor_specs_ratio128(start_pos=None):
    import torch  # type: ignore[import]

    if __package__:
        from .utils import (
            block_table,
            compressed_slot_mapping,
            hca_decode_start_set,
            position_ids_from_starts,
            resolve_start_positions,
            state_slot_mapping,
        )
    else:
        from utils import (
            block_table,
            compressed_slot_mapping,
            hca_decode_start_set,
            position_ids_from_starts,
            resolve_start_positions,
            state_slot_mapping,
        )
    from golden import TensorSpec

    if __package__:
        from .utils import build_rope_tables, materialize_half_rope_tables
    else:
        from utils import build_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, RATIO128_COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)

    def init_compress_state():
        return torch.zeros(RATIO128_COMPRESS_STATE_BLOCK_NUM, RATIO128_COMPRESS_STATE_BLOCK_SIZE, RATIO128_COMPRESS_STATE_DIM)

    # Calibrated to the real DeepSeek-V4-Flash 150
    #  (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(RATIO128_OUT_DIM, D) * 0.0246

    def init_wgate():
        return torch.randn(RATIO128_OUT_DIM, D) * 0.0316

    def init_ape():
        return torch.randn(RATIO128_COMPRESS_RATIO, RATIO128_OUT_DIM) * 0.0340

    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(RATIO128_HEAD_DIM)

    def init_rope_positions():
        first_pos = init_position_ids().to(torch.int64)[:, 0]
        cmp_offset = RATIO128_COMPRESS_RATIO - (first_pos % RATIO128_COMPRESS_RATIO)
        return (first_pos + cmp_offset - RATIO128_COMPRESS_RATIO).to(torch.int64)

    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]

    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]

    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, RATIO128_CMP_STORAGE_BLOCK_SIZE, 1, RATIO128_HEAD_DIM)

    def init_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=RATIO128_COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=RATIO128_COMPRESS_STATE_PHYSICAL_BLOCKS,
            permuted=True,
        )

    def init_cmp_block_table():
        return block_table(
            batch=B,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
            permuted=True,
        )

    def init_default_start_pos():
        # Canonical HCA start-position set (ratio-128 compressor branches + 8k long-context).
        # At B=1 only the 8k point survives, and [8192, 8192 + S) holds no compress
        # boundary, so the entry would write nothing it validates. Pull the 8k point
        # back half a step so its S tokens straddle the 8192 boundary.
        return hca_decode_start_set(
            batch=B,
            compress_ratio=RATIO128_COMPRESS_RATIO,
            state_block_size=RATIO128_COMPRESS_STATE_BLOCK_SIZE,
            long_pos=DECODE_START_POS - S // 2,
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

    def init_state_slot_mapping():
        return state_slot_mapping(
            init_position_ids(),
            init_compress_state_block_table(),
            state_block_size=RATIO128_COMPRESS_STATE_BLOCK_SIZE,
        )

    def init_cmp_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=RATIO128_COMPRESS_RATIO,
            block_size=RATIO128_CMP_STORAGE_BLOCK_SIZE,
        )

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, RATIO128_HEAD_DIM], torch.float32),
        TensorSpec(
            "compress_state",
            [RATIO128_COMPRESS_STATE_BLOCK_NUM, RATIO128_COMPRESS_STATE_BLOCK_SIZE, RATIO128_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [B, RATIO128_COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [RATIO128_OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [RATIO128_OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [RATIO128_COMPRESS_RATIO, RATIO128_OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [RATIO128_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec(
            "cmp_kv_cache",
            [CMP_BLOCK_NUM, RATIO128_CMP_STORAGE_BLOCK_SIZE, 1, RATIO128_HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv_cache,
        ),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [B, S], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [B, S], torch.int64, init_value=init_state_slot_mapping),
    ]


def build_boundary_missing_slot_specs(start_pos=None):
    """Exercise a boundary token that participates in pooling but is not persisted."""
    import torch

    if start_pos is None:
        start_pos = RATIO128_COMPRESS_RATIO - S // 2
    specs = build_tensor_specs_ratio128(start_pos)
    by_name = {spec.name: spec for spec in specs}
    positions = by_name["position_ids"].init_value()
    boundary = (positions + 1).remainder(RATIO128_COMPRESS_RATIO) == 0
    mapping = by_name["state_slot_mapping"].init_value()
    cmp_mapping = by_name["cmp_slot_mapping"].init_value()
    if not bool(boundary.any()) or not bool((mapping[boundary] >= 0).all()) or not bool(
        (cmp_mapping[boundary] >= 0).all()
    ):
        raise ValueError("Fixture requires a mapped compression boundary before masking its state slot")
    mapping[boundary] = -1
    by_name["state_slot_mapping"].init_value = mapping
    state = torch.zeros(by_name["compress_state"].shape, dtype=torch.float32)
    # Keep historical scores finite but negligible so the missing current row
    # has a visible effect while the normal state comparator remains active.
    state[..., RATIO128_OUT_DIM:] = -1e4
    by_name["compress_state"].init_value = state
    return specs


if __package__:
    from .decode_shapes import DECODE_CONFIG as M
    from .config import FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
else:
    from decode_shapes import DECODE_CONFIG as M
    from config import FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS
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
    from .kv_proj_rope import rope_interleave
else:
    from kv_proj_rope import rope_interleave


# model config
INDEXER_HEAD_DIM = M.index_head_dim
INDEXER_HEAD_DIM_INV = 1.0 / INDEXER_HEAD_DIM
INDEXER_NOPE_HEAD_DIM = M.index_nope_head_dim

# kernel-local (ratio-4 overlapping compressor)
INDEXER_COMPRESS_RATIO = 4
IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // INDEXER_COMPRESS_RATIO
INDEXER_OVERLAP = INDEXER_COMPRESS_RATIO == 4
INDEXER_COFF = 1 + int(INDEXER_OVERLAP)
INDEXER_OUT_DIM = INDEXER_COFF * INDEXER_HEAD_DIM
INDEXER_STATE_LEN = INDEXER_COFF * INDEXER_COMPRESS_RATIO
INDEXER_IDX_KV_LEN = MAX_SEQ_LEN // INDEXER_COMPRESS_RATIO
INDEXER_COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INDEXER_COMPRESS_STATE_PHYSICAL_BLOCKS = 65
INDEXER_COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INDEXER_COMPRESS_STATE_BLOCK_SIZE - 1) // INDEXER_COMPRESS_STATE_BLOCK_SIZE
INDEXER_COMPRESS_STATE_BLOCK_NUM = INDEXER_COMPRESS_STATE_PHYSICAL_BLOCKS
INDEXER_COMPRESS_STATE_DIM = 2 * INDEXER_OUT_DIM
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_IDX_CACHE_BLOCK_NUM_DYN')
INDEXER_COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('LOWLAT_TARGET_INNER_STATE_BLOCK_NUM_DYN')

# tiling
PROJ_OUT_TILE = 32  # kv_score_proj N-tile
assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"
INDEXER_RMS_PAD_TILE = BS_PAD  # token-indexed rows, padded to a Cube M multiple of 16
INDEXER_RMS_PAD_ROWS = INDEXER_RMS_PAD_TILE


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, INDEXER_HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[
        [INDEXER_COMPRESS_STATE_BLOCK_NUM_DYN, INDEXER_COMPRESS_STATE_BLOCK_SIZE, INDEXER_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B, INDEXER_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[INDEXER_COMPRESS_RATIO, INDEXER_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEXER_HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[INDEXER_HEAD_DIM, INDEXER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, INDEXER_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_pad = pl.create_tensor([BS_PAD, INDEXER_OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, INDEXER_OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    compress_state_flat = pl.reshape(
        compress_state, [compress_state_block_num * INDEXER_COMPRESS_STATE_BLOCK_SIZE, INDEXER_COMPRESS_STATE_DIM]
    )
    kv_flat = pl.reshape(kv, [B * S, INDEXER_HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, INDEXER_HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(BS_PAD * INDEXER_OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]):
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (INDEXER_OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
        o0 = (idx % (INDEXER_OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [PROJ_OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, PROJ_OUT_TILE] (K strided = many short
            # bursts). Mirrors the main compressor (decode_compressors); the strided
            # ND2NZ form here was ~2x slower on this matmul (43us -> ~20us per task).
            wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(k0 == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(k0 == 0))

        kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc

    # Pool each boundary from immutable history plus this invocation's projections.
    pooled_kv = pl.create_tensor([BS_PAD, INDEXER_HEAD_DIM], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool", allow_early_resolve=True
    ) as pool_tid:
        pooled_kv[:, :] = pl.full([BS_PAD, INDEXER_HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c_idx in pl.range(B):
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            for s_idx in pl.range(S):
                token = c_idx * S + s_idx
                token_pos = pl.read(position_ids, [c_idx, s_idx])
                if (token_pos + 1) % INDEXER_COMPRESS_RATIO == 0:
                    window_start = token_pos - INDEXER_STATE_LEN + 1
                    for h0 in pl.range(0, INDEXER_HEAD_DIM, HEAD_TILE):
                        last_ape = pl.cast(token_pos % INDEXER_COMPRESS_RATIO, pl.INDEX)
                        last_score = score_proj_pad[
                            token : token + 1, INDEXER_HEAD_DIM + h0 : INDEXER_HEAD_DIM + h0 + HEAD_TILE
                        ]
                        last_bias = ape[last_ape : last_ape + 1, INDEXER_HEAD_DIM + h0 : INDEXER_HEAD_DIM + h0 + HEAD_TILE]
                        mi = pl.add(last_score, last_bias)
                        initial_delta = pl.sub(mi, mi)
                        li = pl.exp(initial_delta)
                        oi = kv_proj_pad[token : token + 1, INDEXER_HEAD_DIM + h0 : INDEXER_HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(INDEXER_STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            state_half = 0
                            if state_idx >= INDEXER_COMPRESS_RATIO:
                                state_half = INDEXER_HEAD_DIM
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                page = logical_pos // INDEXER_COMPRESS_STATE_BLOCK_SIZE
                                physical = pl.read(compress_state_block_table, [c_idx, page])
                                if physical >= 0:
                                    state_row = (
                                        physical * INDEXER_COMPRESS_STATE_BLOCK_SIZE
                                        + logical_pos % INDEXER_COMPRESS_STATE_BLOCK_SIZE
                                    )
                                    col = state_half + h0
                                    value = compress_state_flat[
                                        state_row : state_row + 1, col : col + HEAD_TILE
                                    ]
                                    score_col = INDEXER_OUT_DIM + col
                                    score = compress_state_flat[
                                        state_row : state_row + 1, score_col : score_col + HEAD_TILE
                                    ]
                            if logical_pos >= first_pos_b and logical_pos <= token_pos:
                                overlay = c_idx * S + logical_pos - first_pos_b
                                ape_row = pl.cast(logical_pos % INDEXER_COMPRESS_RATIO, pl.INDEX)
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
                    commit_ape_row = pl.cast(token_pos % INDEXER_COMPRESS_RATIO, pl.INDEX)
                    compress_state_flat[commit_state_row : commit_state_row + 1, 0:INDEXER_OUT_DIM] = kv_proj_pad[
                        token : token + 1, :
                    ]
                    commit_score_base = score_proj_pad[token : token + 1, :]
                    commit_score_bias = ape[commit_ape_row : commit_ape_row + 1, :]
                    compress_state_flat[
                        commit_state_row : commit_state_row + 1, INDEXER_OUT_DIM:INDEXER_COMPRESS_STATE_DIM
                    ] = pl.add(commit_score_base, commit_score_bias)

    normed_kv = pl.create_tensor([INDEXER_RMS_PAD_ROWS, INDEXER_HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, INDEXER_HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope", deps=[pool_tid]):
        # single 16-row block: B real rows at rows 0..B-1, rows B..15 are pad.
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the
        # full ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        cos_b = pl.full([INDEXER_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([INDEXER_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_b[0 : B * S, 0:ROPE_HEAD_DIM] = cos[0 : B * S, 0:ROPE_HEAD_DIM]
        sin_b[0 : B * S, 0:ROPE_HEAD_DIM] = sin[0 : B * S, 0:ROPE_HEAD_DIM]
        partial_sq = pl.full([1, INDEXER_RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, INDEXER_HEAD_DIM, HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[0:INDEXER_RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, INDEXER_RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, INDEXER_HEAD_DIM_INV), EPS), [INDEXER_RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.pipeline(0, INDEXER_NOPE_HEAD_DIM, HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[0:INDEXER_RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0:INDEXER_RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

        kv_rope_norm = pooled_kv[0:INDEXER_RMS_PAD_TILE, INDEXER_NOPE_HEAD_DIM:INDEXER_HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, INDEXER_NOPE_HEAD_DIM:INDEXER_HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rms_norm_rope in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is BF16 -> cast on write.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([INDEXER_RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
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
        normed_kv[0:INDEXER_RMS_PAD_TILE, INDEXER_NOPE_HEAD_DIM:INDEXER_HEAD_DIM] = pl.cast(
            rope_rot,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_final = pl.create_tensor([INDEXER_RMS_PAD_ROWS, INDEXER_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
        kv_proj_tile = normed_kv[0:INDEXER_RMS_PAD_TILE, 0:INDEXER_HEAD_DIM]
        for o0 in pl.range(0, INDEXER_HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:INDEXER_HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[0:INDEXER_RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write", allow_early_resolve=True):
        # C8 quant-on-write: per-row INT8 quant of the block (M=RMS_PAD_TILE keeps tiles 32B-aligned;
        # quantize the bf16-rounded value to match golden)
        kv_blk_f32 = pl.cast(
            pl.cast(kv_final[0:INDEXER_RMS_PAD_TILE, 0:INDEXER_HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32,
        )
        # amax = max(|x|); abs-based (max(row_max, -row_min) is wrong on signed KV)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, INDEXER_RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, INT8_AMAX_EPS)
        qd_idxkv_numerator = pl.full([1, INDEXER_RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
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
        icrcp_ones_row = pl.full([1, INDEXER_RMS_PAD_TILE], dtype=pl.FP32, value=1.0)
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
        kv_scale_dq_col = pl.reshape(kv_scale_dq_row, [INDEXER_RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [INDEXER_RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for token in pl.range(B * S):
            c_idx = token // S
            s_idx = token - c_idx * S
            position = pl.read(position_ids, [c_idx, s_idx])
            cache_row_i64 = pl.read(idx_slot_mapping, [c_idx, s_idx])
            if (position + 1) % INDEXER_COMPRESS_RATIO == 0 and cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_flat[token : token + 1, :] = kv_final[token : token + 1, :]
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = kv_i8_blk[token : token + 1, :]
                pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [token, 0]))

    return kv


@pl.jit
def compressor_test_indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, INDEXER_HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[INDEXER_COMPRESS_STATE_BLOCK_NUM_DYN, INDEXER_COMPRESS_STATE_BLOCK_SIZE, INDEXER_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, INDEXER_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[INDEXER_COMPRESS_RATIO, INDEXER_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEXER_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[INDEXER_HEAD_DIM, INDEXER_HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, INDEXER_HEAD_DIM], pl.INT8]],
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


def golden_compressor_indexer(tensors):
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
    ratio, rd = INDEXER_COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = bf16_matmul_fp32(x, wkv.t())  # [B, S, OUT_DIM]  (wkv stored [OUT_DIM, D] for b_trans)
    score = bf16_matmul_fp32(x, wgate.t())  # [B, S, OUT_DIM]

    # Each boundary stays at its input token index, as in the device output.
    pooled = torch.zeros(bsz, S, INDEXER_HEAD_DIM, dtype=torch.float32, device=x.device)
    visible_boundaries = []
    score = score + ape[position_ids % ratio]

    def read_pool_row(b, abs_pos, half):
        if abs_pos < 0:
            return (
                torch.zeros(INDEXER_HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((INDEXER_HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        first_pos = int(position_ids[b, 0].item())
        channel = half * INDEXER_HEAD_DIM
        if abs_pos >= first_pos:
            token = abs_pos - first_pos
            return kv[b, token, channel : channel + INDEXER_HEAD_DIM], score[b, token, channel : channel + INDEXER_HEAD_DIM]
        blk_id = int(compress_state_block_table[b, abs_pos // INDEXER_COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(INDEXER_HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((INDEXER_HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % INDEXER_COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, channel : channel + INDEXER_HEAD_DIM],
            compress_state[blk_id, intra, INDEXER_OUT_DIM + channel : INDEXER_OUT_DIM + channel + INDEXER_HEAD_DIM],
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
                blk_id = state_row // INDEXER_COMPRESS_STATE_BLOCK_SIZE
                intra = state_row % INDEXER_COMPRESS_STATE_BLOCK_SIZE
                compress_state[blk_id, intra, :INDEXER_OUT_DIM] = kv[b, token]
                compress_state[blk_id, intra, INDEXER_OUT_DIM:] = score[b, token]

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


def build_tensor_specs_indexer(start_pos=None):
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

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, INDEXER_COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)

    def init_compress_state():
        state = torch.zeros(INDEXER_COMPRESS_STATE_BLOCK_NUM, INDEXER_COMPRESS_STATE_BLOCK_SIZE, INDEXER_COMPRESS_STATE_DIM)
        state[:, :, INDEXER_OUT_DIM:] = FP32_NEG_INF
        return state

    def init_compress_state_block_table():
        return block_table(
            batch=B,
            table_blocks=INDEXER_COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=INDEXER_COMPRESS_STATE_PHYSICAL_BLOCKS,
        )

    # Calibrated to the real DeepSeek-V4-Flash CSA inner (indexer) compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(INDEXER_OUT_DIM, D) * 0.0293

    def init_wgate():
        return torch.randn(INDEXER_OUT_DIM, D) * 0.0512

    def init_ape():
        return torch.randn(INDEXER_COMPRESS_RATIO, INDEXER_OUT_DIM) * 0.1528

    def init_norm_w():
        return 0.6850 + 0.2610 * torch.randn(INDEXER_HEAD_DIM)

    def init_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundaries = (positions + 1) % INDEXER_COMPRESS_RATIO == 0
        return torch.where(boundaries, positions - INDEXER_COMPRESS_RATIO + 1, 0).reshape(-1)

    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]

    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]

    def init_hadamard():
        return torch.rand(INDEXER_HEAD_DIM, INDEXER_HEAD_DIM) * (INDEXER_HEAD_DIM**-0.5)

    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, INDEXER_HEAD_DIM, dtype=torch.int8)

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
            batch=B, seq=S, compress_ratio=INDEXER_COMPRESS_RATIO, state_block_size=INDEXER_COMPRESS_STATE_BLOCK_SIZE
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
            state_block_size=INDEXER_COMPRESS_STATE_BLOCK_SIZE,
        )

    def init_idx_slot_mapping():
        positions = init_position_ids()
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=INDEXER_COMPRESS_RATIO,
            block_size=IDX_STORAGE_BLOCK_SIZE,
        )

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, INDEXER_HEAD_DIM], torch.float32),
        TensorSpec(
            "compress_state",
            [INDEXER_COMPRESS_STATE_BLOCK_NUM, INDEXER_COMPRESS_STATE_BLOCK_SIZE, INDEXER_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [B, INDEXER_COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [INDEXER_OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [INDEXER_OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [INDEXER_COMPRESS_RATIO, INDEXER_OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [INDEXER_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B * S, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("hadamard", [INDEXER_HEAD_DIM, INDEXER_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, IDX_STORAGE_BLOCK_SIZE, 1, INDEXER_HEAD_DIM],
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


def validate_ratio4():
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    if __package__:
        from .decode_shapes import CONTEXT_CAPACITY
    else:
        from decode_shapes import CONTEXT_CAPACITY

    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_CAPACITY, help="import-time context capacity (default 1048576)")
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=compressor_test_ratio4,
        specs=build_tensor_specs_ratio4(args.start_pos),
        golden_fn=golden_compressor_ratio4,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
            "compress_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)



def validate_ratio128():
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch HCA set that includes the 8k point.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--boundary-missing-slot-fixture", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=compressor_test_ratio128,
        specs=(build_boundary_missing_slot_specs(args.start_pos) if args.boundary_missing_slot_fixture
               else build_tensor_specs_ratio128(args.start_pos)),
        golden_fn=golden_compressor_ratio128,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        # Precision reference: AscendC torch.ops.custom.compressor —
        # ops-transformer/experimental/attention/compressor/tests/pytest/compressor_golden.py
        compare_fn={
            "kv":            compare_compressor_kv,
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache":   ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)



def validate_indexer():
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    if __package__:
        from .decode_shapes import CONTEXT_CAPACITY
    else:
        from decode_shapes import CONTEXT_CAPACITY

    parser.add_argument("--max-seq-len", type=int, default=CONTEXT_CAPACITY, help="import-time context capacity (default 1048576)")
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=compressor_test_indexer,
        specs=build_tensor_specs_indexer(args.start_pos),
        golden_fn=golden_compressor_indexer,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
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


if __name__ == "__main__":
    from models.deepseek_v4_flash_lowlat.build import run_operator_validations

    run_operator_validations(
        {
            'ratio4': validate_ratio4,
            'ratio128': validate_ratio128,
            'indexer': validate_indexer,
        },
    )
