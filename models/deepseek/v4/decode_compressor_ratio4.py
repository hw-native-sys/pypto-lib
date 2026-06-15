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
Tree reduction for softmax+pool. State shift after compression."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, FP32_NEG_INF


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

# kernel-local (ratio-4 overlapping compressor)
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_MAX_BLOCKS = 65
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
ROPE_TILE = 32
K_TILE = 512
OUT_TILE = 64
B_TILE = 64
HEAD_TILE = 64
HEAD_DIM_TILE = 128
RMS_TILE = 16


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    x_flat = pl.reshape(x, [B * S, D])
    cmp4_kv_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_scratch = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    for idx in pl.spmd(B * S * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
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

        cmp4_kv_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = kv_acc
        cmp4_score_proj_scratch[global_row0 : global_row0 + B_TILE, o0 : o0 + OUT_TILE] = score_acc

    cmp4_kv_proj_by_batch = pl.reshape(cmp4_kv_proj_scratch, [B, S * OUT_DIM])
    cmp4_score_proj_by_batch = pl.reshape(cmp4_score_proj_scratch, [B, S * OUT_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):
        for c_idx in pl.range(B):
            for s in pl.pipeline(S, stage=2):
                token_pos = pl.read(position_ids, [c_idx, s])
                state_row = pl.cast(pl.read(state_slot_mapping, [c_idx, s]), pl.INDEX)
                proj_col0 = s * OUT_DIM
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row >= 0:
                    kv_tile = cmp4_kv_proj_by_batch[c_idx : c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    score_tile = cmp4_score_proj_by_batch[c_idx : c_idx + 1, proj_col0 : proj_col0 + OUT_DIM]
                    ape_tile = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    score_tile = pl.add(score_tile, ape_tile)
                    compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = kv_tile
                    compress_state_flat[state_row : state_row + 1, OUT_DIM : 2 * OUT_DIM] = score_tile

    pooled_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    for c_idx in pl.spmd(B, name_hint="softmax_pool"):
        first_pos_b = pl.read(position_ids, [c_idx, 0])
        pos_b = first_pos_b % COMPRESS_RATIO
        pre_tokens_b = COMPRESS_RATIO - pos_b
        boundary_end_b = first_pos_b + pre_tokens_b - 1
        cur_window_start_b = boundary_end_b - COMPRESS_RATIO + 1
        prev_window_start_b = cur_window_start_b - COMPRESS_RATIO

        if pos_b + S >= COMPRESS_RATIO:
            last_abs = cur_window_start_b + COMPRESS_RATIO - 1
            last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
            last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
            last_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, last_blk_off]), pl.INDEX)
            last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
            for hb in pl.range(HEAD_DIM // HEAD_TILE):
                h0 = hb * HEAD_TILE
                last_col0 = OUT_DIM + HEAD_DIM + h0
                mi = compress_state_flat[last_row : last_row + 1, last_col0 : last_col0 + HEAD_TILE]
                li = pl.exp(pl.sub(mi, mi))
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    if prev_abs >= 0:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, prev_blk_off]), pl.INDEX)
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM + h0 : OUT_DIM + h0 + HEAD_TILE]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, h0 : h0 + HEAD_TILE]
                        mi_next_front = pl.maximum(mi, front_score)
                        alpha_front = pl.exp(pl.sub(mi, mi_next_front))
                        beta_front = pl.exp(pl.sub(front_score, mi_next_front))
                        li = pl.add(pl.mul(alpha_front, li), beta_front)
                        oi = pl.add(pl.mul(oi, alpha_front), pl.mul(front_kv, beta_front))
                        mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_id = pl.cast(pl.read(compress_state_block_table, [c_idx, cur_blk_off]), pl.INDEX)
                    cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_col0 = OUT_DIM + HEAD_DIM + h0
                    back_score = compress_state_flat[cur_row : cur_row + 1, back_col0 : back_col0 + HEAD_TILE]
                    back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back = pl.exp(pl.sub(mi, mi_next_back))
                    beta_back = pl.exp(pl.sub(back_score, mi_next_back))
                    li = pl.add(pl.mul(alpha_back, li), beta_back)
                    oi = pl.add(pl.mul(oi, alpha_back), pl.mul(back_kv, beta_back))
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv[c_idx : c_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    normed_kv = pl.create_tensor([B, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="rmsnorm_rope"):
        batch_base = batch_base_idx * RMS_TILE
        cos_b = cos[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        sin_b = sin[batch_base : batch_base + RMS_TILE, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE]
            gamma = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[batch_base : batch_base + RMS_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
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

    for batch_base_idx in pl.spmd(B // RMS_TILE, name_hint="kv_and_cache_write"):
        batch_base = batch_base_idx * RMS_TILE
        for inner in pl.range(RMS_TILE):
            c_idx = batch_base + inner
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            pos_b = first_pos_b % COMPRESS_RATIO
            if pos_b + S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - pos_b
                kv_row_fp32 = normed_kv[c_idx : c_idx + 1, 0 : HEAD_DIM]
                kv_flat[c_idx * S : c_idx * S + 1, :] = kv_row_fp32
                cache_row = pl.cast(pl.read(cmp_slot_mapping, [c_idx, boundary_s]), pl.INDEX)
                if cache_row >= 0:
                    cmp_kv_cache_flat[cache_row : cache_row + 1, :] = pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint")

    compress_state = pl.reshape(compress_state_flat, [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    cmp_kv_cache = pl.reshape(cmp_kv_cache_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv, compress_state, cmp_kv_cache


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.Out[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.Out[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    kv, compress_state, cmp_kv_cache = compressor_ratio4(
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
    )
    return kv, compress_state, cmp_kv_cache


def golden_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch, ratio=4 overlap)."""
    import torch

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
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv                        # [B, S, OUT_DIM]
    score = x @ wgate                   # [B, S, OUT_DIM]

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        pre_tokens = min(S, ratio - (first_pos % ratio))
        boundary_s = ratio - 1 - (first_pos % ratio)
        should_compress = 0 <= boundary_s < S
        boundary_end = first_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

        # Per-token ape add + state scatter through explicit token-major slots.
        for s in range(S):
            pos = int(position_ids[b, s].item())
            token_ape_row = pos % ratio
            score[b, s, :] = score[b, s, :] + ape[token_ape_row]
            state_row = int(state_slot_mapping[b, s].item())
            if state_row >= 0:
                blk_id = state_row // COMPRESS_STATE_BLOCK_SIZE
                intra = state_row % COMPRESS_STATE_BLOCK_SIZE
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
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, :HEAD_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM:OUT_DIM + HEAD_DIM])
            for s in range(ratio):
                abs_pos = cur_window_start + s
                blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
                intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
                kv_rows.append(compress_state[blk_id, intra, HEAD_DIM:OUT_DIM])
                score_rows.append(compress_state[blk_id, intra, OUT_DIM + HEAD_DIM:])
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
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        # Kernel writes pooled result only to kv[:, 0, :]; leave kv[:, 1:, :] = 0
        # so the golden matches its [B, S, HEAD_DIM] zero-init.
        tensors["kv"][b : b + 1, 0:1, :] = kv_b

        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            blk_id = cmp_row // BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_half_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def init_x():
        return torch.rand(B, S, D)
    def init_compress_state():
        state = torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        state[:, :, OUT_DIM:] = FP32_NEG_INF
        return state
    def init_compress_state_block_table():
        tbl = torch.full((B, COMPRESS_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(COMPRESS_STATE_MAX_BLOCKS):
                tbl[b, j] = b * COMPRESS_STATE_MAX_BLOCKS + j
        return tbl
    def init_wkv():
        return torch.rand(D, OUT_DIM)
    def init_wgate():
        return torch.rand(D, OUT_DIM)
    def init_ape():
        return torch.rand(COMPRESS_RATIO, OUT_DIM)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
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
    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl
    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Default per-batch pattern covers every ratio-4 decode branch:
        #   0           : no-compress, window start
        #   1           : no-compress, mid-window
        #   RATIO-S     : compress, boundary on 2nd token with empty previous window
        #   RATIO-1     : compress, boundary on 1st token with 2nd token spilling to next window
        #   2*RATIO-S   : compress aligned in the 2nd window with previous-window overlap
        #   2*RATIO-1   : compress crossing in the 2nd window with previous-window overlap
        #   STATE_BLK*32-1: compress crossing state logical block 31->32
        pattern = torch.tensor([
            0,
            1,
            COMPRESS_RATIO - S,
            COMPRESS_RATIO - 1,
            COMPRESS_RATIO * 2 - S,
            COMPRESS_RATIO * 2 - 1,
            COMPRESS_STATE_BLOCK_SIZE * 32 - 1,
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
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
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
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv":          ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state":    ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / (CMP_BLOCK_NUM * BLOCK_SIZE)),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
