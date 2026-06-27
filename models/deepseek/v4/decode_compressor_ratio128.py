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

from config import FLASH as M, BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, DECODE_BATCH, DECODE_SEQ

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")
S_DYN = pl.dynamic("S_DYN")
COMPRESS_STATE_MAX_BLOCKS_DYN = pl.dynamic("COMPRESS_STATE_MAX_BLOCKS_DYN")
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("COMPRESS_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

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
# - COMPRESS_STATE_MAX_BLOCKS=64 is only enough for the current tests
#   (64 * 8 positions per request), not full MAX_SEQ_LEN coverage.
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = 8
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
if IDX_KV_LEN > CMP_MAX_BLOCKS * BLOCK_SIZE:
    raise ValueError("ratio128 compressed KV cache capacity is smaller than max compressed sequence length")

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
HEAD_TILE = 64
B_TILE = 64
RMS_TILE = 8


@pl.jit.inline
def compressor_ratio128(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Tensor[[B_DYN, S_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS_DYN], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[B_DYN, S_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
):
    b_dim = pl.tensor.dim(x, 0)
    s_dim = pl.tensor.dim(x, 1)
    bs = b_dim * s_dim
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv_cache, 0)

    kv_proj_scratch = pl.create_tensor([bs, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([bs, OUT_DIM], dtype=pl.FP32)
    x_flat = pl.reshape(x, [bs, D])
    for idx in pl.spmd(bs * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_tile = x_flat[global_row0 : global_row0 + B_TILE, k0 : k0 + K_TILE]
            wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)

        kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    s_out = s_dim * OUT_DIM
    kv_proj_by_batch = pl.reshape(kv_proj_scratch, [b_dim, s_out])
    score_proj_by_batch = pl.reshape(score_proj_scratch, [b_dim, s_out])
    compress_state_flat = pl.reshape(compress_state, [compress_state_block_num, COMPRESS_STATE_BLOCK_SIZE * COMPRESS_STATE_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_pre"):
        for global_c_idx in pl.range(b_dim):
            for s in pl.pipeline(s_dim, stage=2):
                proj_col0 = s * OUT_DIM
                token_pos = pl.read(position_ids, [global_c_idx, s])
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                state_row = pl.cast(pl.read(state_slot_mapping, [global_c_idx, s]), target_type=pl.INDEX)
                if state_row >= 0:
                    state_blk_id = state_row // COMPRESS_STATE_BLOCK_SIZE
                    state_intra = state_row % COMPRESS_STATE_BLOCK_SIZE
                    slot_col0_s = state_intra * COMPRESS_STATE_DIM
                    ape_row = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    kv_row = kv_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = score_proj_by_batch[global_c_idx : global_c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_row = pl.add(score_row, ape_row)
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s : slot_col0_s + OUT_DIM] = kv_row
                    compress_state_flat[state_blk_id : state_blk_id + 1, slot_col0_s + OUT_DIM : slot_col0_s + 2 * OUT_DIM] = score_row

    pooled_kv = pl.create_tensor([b_dim, HEAD_DIM], dtype=pl.FP32)
    for idx in pl.spmd(b_dim * HEAD_DIM // HEAD_TILE, name_hint="softmax_pool"):
        global_c_idx = idx // (HEAD_DIM // HEAD_TILE)
        h0 = (idx % (HEAD_DIM // HEAD_TILE)) * HEAD_TILE
        first_pos_gate = pl.read(position_ids, [global_c_idx, 0])
        pos_gate = first_pos_gate % COMPRESS_RATIO
        if pos_gate + S >= COMPRESS_RATIO:
            softmax_score_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
            softmax_kv_state = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
            for s in pl.pipeline(STATE_LEN, stage=2):
                first_pos_b = pl.read(position_ids, [global_c_idx, 0])
                pos_b = first_pos_b % COMPRESS_RATIO
                compress_pos = first_pos_b + (COMPRESS_RATIO - 1 - pos_b)
                state_pos = compress_pos - (COMPRESS_RATIO - 1) + s
                state_logical_blk = state_pos // COMPRESS_STATE_BLOCK_SIZE
                state_intra = state_pos % COMPRESS_STATE_BLOCK_SIZE
                state_blk_id = pl.cast(pl.read(compress_state_block_table, [global_c_idx, state_logical_blk]), target_type=pl.INDEX)
                kv_col0 = state_intra * COMPRESS_STATE_DIM + h0
                score_col0 = state_intra * COMPRESS_STATE_DIM + OUT_DIM + h0
                slot_score = compress_state_flat[state_blk_id : state_blk_id + 1, score_col0 : score_col0 + HEAD_TILE]
                slot_kv = compress_state_flat[state_blk_id : state_blk_id + 1, kv_col0 : kv_col0 + HEAD_TILE]
                softmax_score_state[s : s + 1, :] = slot_score
                softmax_kv_state[s : s + 1, :] = slot_kv

            softmax_score_state_t = pl.transpose(softmax_score_state, axis1=0, axis2=1)
            softmax_kv_state_t = pl.transpose(softmax_kv_state, axis1=0, axis2=1)
            score_max = pl.row_max(softmax_score_state_t)
            score_exp = pl.exp(pl.row_expand_sub(softmax_score_state_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_chunk_t = pl.row_sum(pl.mul(softmax_kv_state_t, score_prob))
            pooled_chunk = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
            pooled_kv[global_c_idx : global_c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv = pl.create_tensor([b_dim, HEAD_DIM], dtype=pl.FP32)

    for batch_base_idx in pl.spmd(b_dim // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[batch_base : batch_base + RMS_TILE, rms_kb * HEAD_TILE : (rms_kb + 1) * HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rope_fused in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. swap_idx (j^1), sign
        # ([-1,+1,...]) and dup_idx (j>>1) are built IN-KERNEL from pl.arange; cos_il/sin_il
        # are dup-gathered from the per-batch cos/sin rows. normed_kv is FP32 -> write directly.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)                                       # j>>1
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)                                                # [-1,+1,...]
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_rot

    kv_flat = pl.reshape(kv, [bs, HEAD_DIM])
    cmp_flat_rows = cmp_block_num * BLOCK_SIZE
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [cmp_flat_rows, HEAD_DIM])

    for batch_base_idx in pl.spmd(b_dim // RMS_TILE, name_hint="kv_finalize"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            global_c_idx = batch_base + inner
            first_pos_b = pl.read(position_ids, [global_c_idx, 0])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + s_dim >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row = normed_kv[global_c_idx : global_c_idx + 1, 0 : HEAD_DIM]
                kv_flat[global_c_idx * s_dim : global_c_idx * s_dim + 1, :] = kv_row
                cmp_row = pl.cast(pl.read(cmp_slot_mapping, [global_c_idx, boundary_s]), target_type=pl.INDEX)
                if cmp_row >= 0:
                    cmp_kv_cache_flat[cmp_row : cmp_row + 1, :] = pl.cast(kv_row, target_type=pl.BF16, mode="rint")

    kv = pl.reshape(kv_flat, [b_dim, s_dim, HEAD_DIM])
    return kv


@pl.jit
def compressor_test(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B_DYN, S_DYN, HEAD_DIM], pl.FP32]],
    compress_state: pl.Out[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS_DYN], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.Out[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[B_DYN, S_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[B_DYN, S_DYN], pl.INT64],
):
    x.bind_dynamic(0, B_DYN)
    x.bind_dynamic(1, S_DYN)
    kv.bind_dynamic(0, B_DYN)
    kv.bind_dynamic(1, S_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
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

    compressor_ratio128(
        x, kv, compress_state, compress_state_block_table, wkv, wgate, ape, norm_w, cos, sin,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping,
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
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]
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

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        boundary_positions = torch.nonzero((position_ids[b, :S] + 1) % ratio == 0, as_tuple=False).flatten()
        if int(boundary_positions.numel()) == 0:
            continue
        boundary_s = int(boundary_positions[0].item())
        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            cblk = cmp_row // BLOCK_SIZE
            intra_offset = cmp_row % BLOCK_SIZE
            cmp_kv_cache[cblk, intra_offset, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    # Calibrated to the real DeepSeek-V4-Flash HCA (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform).
    def init_wkv():
        return torch.randn(D, OUT_DIM) * 0.0246
    def init_wgate():
        return torch.randn(D, OUT_DIM) * 0.0316
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
        # Always non-contiguous: swap logical blocks 0<->31 so the fixture
        # validates block-table indirection rather than a contiguous layout.
        tbl = torch.arange(B * COMPRESS_STATE_MAX_BLOCKS, dtype=torch.int32).view(B, COMPRESS_STATE_MAX_BLOCKS)
        for b in range(B):
            tbl[b, 0], tbl[b, 31] = tbl[b, 31].clone(), tbl[b, 0].clone()
        return tbl
    def init_cmp_block_table():
        tbl = torch.arange(B * CMP_MAX_BLOCKS, dtype=torch.int32).view(B, CMP_MAX_BLOCKS)
        for b in range(B):
            tbl[b, 0], tbl[b, 1] = tbl[b, 1].clone(), tbl[b, 0].clone()
        return tbl
    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Default per-batch pattern covers every ratio-128 decode branch:
        #   STATE_BLK-1 : no-compress, two MTP tokens straddle the state page (size 8)
        #   10          : no-compress, mid-window, single state page
        #   RATIO-S     : compress, boundary on 2nd token (pre-scatter writes both)
        #   RATIO-1     : compress, boundary on 1st token (2nd token spills to next window)
        #   2*RATIO-S   : compress aligned in the 2nd compressed block (cache_col=1)
        #   2*RATIO-1   : compress crossing in the 2nd block (state logical block 31->32)
        pattern = torch.tensor([
            COMPRESS_STATE_BLOCK_SIZE - 1,
            10,
            COMPRESS_RATIO - S,
            COMPRESS_RATIO - 1,
            COMPRESS_RATIO * 2 - S,
            COMPRESS_RATIO * 2 - 1,
        ], dtype=torch.int32)
        vals = torch.empty((B,), dtype=torch.int32)
        for b in range(B):
            vals[b] = pattern[b % int(pattern.numel())]
        return vals
    def init_position_ids():
        starts = init_start_pos().to(torch.int64)
        positions = torch.empty((B, S), dtype=torch.int32)
        for b in range(B):
            for s in range(S):
                positions[b, s] = starts[b] + s
        return positions
    def init_state_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_compress_state_block_table().to(torch.int64)
        mapping = torch.full((B, S), -1, dtype=torch.int64)
        for b in range(B):
            for s in range(S):
                pos = int(positions[b, s].item())
                logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
                intra = pos % COMPRESS_STATE_BLOCK_SIZE
                blk = int(block_table[b, logical_blk].item())
                mapping[b, s] = blk * COMPRESS_STATE_BLOCK_SIZE + intra
        return mapping
    def init_cmp_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_cmp_block_table().to(torch.int64)
        mapping = torch.full((B, S), -1, dtype=torch.int64)
        for b in range(B):
            for s in range(S):
                pos = int(positions[b, s].item())
                if (pos + 1) % COMPRESS_RATIO == 0:
                    cache_col = pos // COMPRESS_RATIO
                    logical_blk = cache_col // BLOCK_SIZE
                    intra = cache_col % BLOCK_SIZE
                    blk = int(block_table[b, logical_blk].item())
                    mapping[b, s] = blk * BLOCK_SIZE + intra
        return mapping
    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32, is_output=True),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [B, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("cmp_kv_cache", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache, is_output=True),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [B, S], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [B, S], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Fixture-only compatibility seed for position_ids and slot mappings; "
                             "otherwise use the default per-batch coverage pattern.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
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
