# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental, ratio=4 overlap).

Uses overlapping state layout with 8 slots.
Front slots 0-3 at columns [0:HEAD_DIM], back slots 4-7 at columns [HEAD_DIM:OUT_DIM].
Online-softmax pooling followed by the recurrent-state commit."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    KV_CMP_BLOCK_NUM,
    FP32_NEG_INF,
)


# Dynamic shape variables. Under CP the compressor runs over the whole TP group's
# token stream while its caller stays on the local query rows, so its token and
# request axes are its own symbols rather than the caller's.
B_DYN = pl.dynamic("DECODE_CSA_C4_B_DYN")
S_DYN = pl.dynamic("DECODE_CSA_C4_S_DYN")
T_DYN = pl.dynamic("DECODE_CSA_C4_T_DYN")  # T = B * S

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
MAX_SEQ_LEN = M.max_position_embeddings

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = (STATE_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

# tiling
K_TILE = 512
OUT_TILE = 64
MM_B_TILE = 16
# kv_score_proj runs on persistent workers, each striding over the block list.
KV_SCORE_WORKERS = 24
# Scratch spans the CP group's whole token stream, which the compressor runs
# over, not the rank-local B * S.
GROUP_BS = DECODE_BATCH * DECODE_SEQ
BS_PAD = ((GROUP_BS + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16  # 16-row block of B (min M for FP32 vec ops)


@pl.jit.inline
def compressor_ratio4(
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
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    b_dim = pl.tensor.dim(compress_state_block_table, 0)
    bs = pl.tensor.dim(x, 0)
    s_dim = bs // b_dim
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE  # ceil to whole 16-row cube tiles
    rms_blocks = (bs + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    x_flat = x
    cmp4_kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)
    compress_state_flat = pl.reshape(compress_state, [compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = kv
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(KV_SCORE_WORKERS, name_hint="kv_score_proj", deps=[late_dep]) as _kv_score_tid:
        kv_worker = pl.tile.get_block_idx()
        for idx in pl.range(kv_worker, t_matmul * OUT_DIM // (MM_B_TILE * OUT_TILE), KV_SCORE_WORKERS):
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
                # bursts). Cuts the transaction-bound MTE2 cost ~14% busy / ~7% compressor wall.
                wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                    score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                    score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)

            cmp4_kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = kv_acc
            cmp4_score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + OUT_TILE] = score_acc

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
                        cmp4_score_proj_pad[
                            token : token + 1,
                            HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE,
                        ],
                        ape[
                            last_ape_row : last_ape_row + 1,
                            HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE,
                        ],
                    )
                    li = pl.exp(pl.sub(mi, mi))
                    oi = cmp4_kv_proj_pad[
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
                                value = cmp4_kv_proj_pad[
                                    overlay_token : overlay_token + 1,
                                    state_half + h0 : state_half + h0 + HEAD_TILE,
                                ]
                                score = pl.add(
                                    cmp4_score_proj_pad[
                                        overlay_token : overlay_token + 1,
                                        state_half + h0 : state_half + h0 + HEAD_TILE,
                                    ],
                                    ape[ape_row : ape_row + 1, state_half + h0 : state_half + h0 + HEAD_TILE],
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
            state_row_i64 = pl.read(state_slot_mapping, [token])
            if state_row_i64 >= 0:
                state_row = pl.cast(state_row_i64, pl.INDEX)
                token_pos = pl.read(position_ids, [token])
                ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = cmp4_kv_proj_pad[
                    token : token + 1, 0 : OUT_DIM]
                compress_state_flat[state_row : state_row + 1, OUT_DIM : COMPRESS_STATE_DIM] = pl.add(
                    cmp4_score_proj_pad[token : token + 1, 0 : OUT_DIM], ape[ape_row : ape_row + 1, 0 : OUT_DIM])

    normed_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.spmd(
        rms_blocks, name_hint="rmsnorm_rope_cache_write", deps=[pool_tid]
    ) as cache_write_tid:
        rms_blk = pl.tile.get_block_idx()
        b0 = rms_blk * RMS_PAD_TILE
        rms_blk_rows = pl.min(RMS_PAD_TILE, bs - b0)
        cos_b = pl.slice(cos, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        sin_b = pl.slice(sin, [RMS_PAD_TILE, ROPE_HEAD_DIM], [b0, 0], valid_shape=[rms_blk_rows, ROPE_HEAD_DIM])
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[b0 : b0 + RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

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

        for inner in pl.range(rms_blk_rows):
            token = b0 + inner
            cache_row_i64 = pl.read(cmp_slot_mapping, [token])
            if cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_row_fp32 = normed_kv[token : token + 1, 0 : HEAD_DIM]
                kv_flat[token : token + 1, :] = kv_row_fp32
                cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(
                    kv_row_fp32, target_type=pl.BF16, mode="rint")

    return kv, cache_write_tid


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
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)

    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    kv, _cache_write_tid = compressor_ratio4(
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
        cmp_kv_cache,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
        late_dep,
    )
    return kv, compress_state, cmp_kv_cache


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
    cmp_kv_cache = tensors["cmp_kv_cache"]
    position_ids = tensors["position_ids"].to(torch.int64)
    cmp_slot_mapping = tensors["cmp_slot_mapping"].to(torch.int64)
    state_slot_mapping = tensors["state_slot_mapping"].to(torch.int64)
    tokens = x.shape[0]
    bsz = tokens // S
    position_ids = position_ids.view(bsz, S)
    cmp_slot_mapping = cmp_slot_mapping.view(bsz, S)
    state_slot_mapping = state_slot_mapping.view(bsz, S)
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
                half = 0 if state_idx < ratio else HEAD_DIM
                value = torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device)
                score = torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device)
                if 0 <= logical_pos < first_pos:
                    ring_row = logical_pos % STATE_LEN
                    page_off, intra = divmod(ring_row, COMPRESS_STATE_BLOCK_SIZE)
                    block = int(compress_state_block_table[b, page_off].item())
                    if block >= 0:
                        value = old_state[block, intra, half : half + HEAD_DIM]
                        score = old_state[block, intra, OUT_DIM + half : OUT_DIM + half + HEAD_DIM]
                if first_pos <= logical_pos <= token_pos:
                    overlay = b * S + logical_pos - first_pos
                    value = kv_proj[overlay, half : half + HEAD_DIM]
                    score = score_proj[overlay, half : half + HEAD_DIM] + ape[
                        logical_pos % ratio, half : half + HEAD_DIM]
                kv_rows.append(value)
                score_rows.append(score)
            kvs = torch.stack(kv_rows)
            scores = torch.stack(score_rows)
            pooled[token] = (kvs * scores.softmax(dim=0)).sum(dim=0)

    for b in range(bsz):
        for s in range(S):
            token = b * S + s
            state_row = int(state_slot_mapping[b, s].item())
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
            cmp_row = int(cmp_slot_mapping[b, s].item())
            if cmp_row < 0:
                continue
            kv_b = rmsnorm(pooled[token : token + 1], norm_w)
            rope_normed = kv_b[..., -rd:]
            rope_swapped = rope_normed.reshape(1, -1, 2).flip(-1).flatten(-2)
            rope_rot = rope_normed * cos[token] + rope_swapped * sin[token]
            kv_b = torch.cat([kv_b[..., :-rd], rope_rot], dim=-1)
            tensors["kv"][token : token + 1] = kv_b
            blk_id = cmp_row // BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % BLOCK_SIZE, 0] = kv_b[0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


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
    cmp_block_table = block_table(
        batch=batch,
        table_blocks=CMP_MAX_BLOCKS,
        physical_blocks=CMP_BLOCK_NUM,
    )
    cmp_slots = compressed_slot_mapping(
        positions,
        cmp_block_table,
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
    # layers 8/32 (the ratio-4 CSA main compressor).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0381
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1226
    def init_norm_w():
        return 0.9569 + 0.1916 * torch.randn(HEAD_DIM)
    def init_cos():
        return rope_cos.clone()
    def init_sin():
        return rope_sin.clone()
    def init_cmp_kv_cache():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_position_ids():
        return positions.clone()
    def init_state_slot_mapping():
        return state_slots.clone()
    def init_cmp_slot_mapping():
        return cmp_slots.clone()

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
        TensorSpec("cmp_kv_cache", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache),
        TensorSpec("position_ids", [batch * S], torch.int32, init_value=lambda: init_position_ids().reshape(-1)),
        TensorSpec("cmp_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_cmp_slot_mapping().reshape(-1)),
        TensorSpec("state_slot_mapping", [batch * S], torch.int64, init_value=lambda: init_state_slot_mapping().reshape(-1)),
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
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0, ignore_nan=True),
            "compress_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
