# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 overlapping ratio-4 decode KV compressor."""

import pypto.language as pl

from config import (
    ACTIVE as M,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_CMP_BLOCK_NUM,
    DECODE_SEQ,
    FP32_NEG_INF,
    KV_CMP_MAX_BLOCKS,
)

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

# compressor config
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
OUT_DIM = COFF * HEAD_DIM
COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = 65
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * OUT_DIM
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

# tiling
K_TILE = 512
OUT_TILE = 64
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16


@pl.jit.inline
def compressor_ratio4(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    cmp4_kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    cmp4_score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)

    with pl.spmd(
        BS_PAD * OUT_DIM // (MM_B_TILE * OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]
    ) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // OUT_TILE)) * OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
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

    compress_state_flat = pl.reshape(
        compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    pooled_kv = pl.create_tensor([RMS_PAD_TILE, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool"):
        for c_idx in pl.range(B):
            for s_sc in pl.pipeline(S, stage=2):
                token_pos = pl.read(position_ids, [c_idx, s_sc])
                state_row_i64 = pl.read(state_slot_mapping, [c_idx, s_sc])
                proj_row = c_idx * S + s_sc
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, pl.INDEX)
                    kv_tile = cmp4_kv_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    score_tile = cmp4_score_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    ape_tile = ape[token_ape_row : token_ape_row + 1, 0 : OUT_DIM]
                    score_ape_tile = pl.add(score_tile, ape_tile)
                    compress_state_flat[state_row : state_row + 1, 0 : OUT_DIM] = kv_tile
                    compress_state_flat[state_row : state_row + 1, OUT_DIM : 2 * OUT_DIM] = score_ape_tile

            pad_idx = c_idx
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
                last_blk_raw = pl.read(compress_state_block_table, [c_idx, last_blk_off])
                last_blk_id = pl.cast(last_blk_raw, pl.INDEX)
                last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                mi = compress_state_flat[last_row : last_row + 1, OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM]
                li_diff = pl.sub(mi, mi)
                li = pl.exp(li_diff)
                oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM : OUT_DIM]

                for s in pl.range(0, COMPRESS_RATIO):
                    prev_abs = prev_window_start_b + s
                    front_score = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=FP32_NEG_INF)
                    front_kv = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    if first_pos_b >= COMPRESS_RATIO:
                        prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                        prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                        prev_blk_raw = pl.read(compress_state_block_table, [c_idx, prev_blk_off])
                        prev_blk_id = pl.cast(prev_blk_raw, pl.INDEX)
                        prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                        front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM : OUT_DIM + HEAD_DIM]
                        front_kv = compress_state_flat[prev_row : prev_row + 1, 0 : HEAD_DIM]
                    mi_next_front = pl.maximum(mi, front_score)
                    alpha_front_diff = pl.sub(mi, mi_next_front)
                    alpha_front = pl.exp(alpha_front_diff)
                    beta_front_diff = pl.sub(front_score, mi_next_front)
                    beta_front = pl.exp(beta_front_diff)
                    li_front = pl.mul(alpha_front, li)
                    li = pl.add(li_front, beta_front)
                    oi_front = pl.mul(oi, alpha_front)
                    kv_front = pl.mul(front_kv, beta_front)
                    oi = pl.add(oi_front, kv_front)
                    mi = mi_next_front

                for s in pl.range(0, COMPRESS_RATIO - 1):
                    cur_abs = cur_window_start_b + s
                    cur_blk_off = cur_abs // COMPRESS_STATE_BLOCK_SIZE
                    cur_intra = cur_abs % COMPRESS_STATE_BLOCK_SIZE
                    cur_blk_raw = pl.read(compress_state_block_table, [c_idx, cur_blk_off])
                    cur_blk_id = pl.cast(cur_blk_raw, pl.INDEX)
                    cur_row = cur_blk_id * COMPRESS_STATE_BLOCK_SIZE + cur_intra
                    back_score = compress_state_flat[cur_row : cur_row + 1, OUT_DIM + HEAD_DIM : COMPRESS_STATE_DIM]
                    back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM : OUT_DIM]
                    mi_next_back = pl.maximum(mi, back_score)
                    alpha_back_diff = pl.sub(mi, mi_next_back)
                    alpha_back = pl.exp(alpha_back_diff)
                    beta_back_diff = pl.sub(back_score, mi_next_back)
                    beta_back = pl.exp(beta_back_diff)
                    li_back = pl.mul(alpha_back, li)
                    li = pl.add(li_back, beta_back)
                    oi_back = pl.mul(oi, alpha_back)
                    kv_back = pl.mul(back_kv, beta_back)
                    oi = pl.add(oi_back, kv_back)
                    mi = mi_next_back

                pooled_chunk = pl.div(oi, li)
                pooled_kv[pad_idx : pad_idx + 1, 0 : HEAD_DIM] = pooled_chunk

    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    cmp_kv_cache_flat = pl.reshape(cmp_kv_cache, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    normed_kv = pl.create_tensor([RMS_PAD_TILE, HEAD_DIM], dtype=pl.FP32)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope_cache_write"):
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        cos_b[0:B, 0 : ROPE_HEAD_DIM // 2] = cos[0:B, 0 : ROPE_HEAD_DIM // 2]
        sin_b[0:B, 0 : ROPE_HEAD_DIM // 2] = sin[0:B, 0 : ROPE_HEAD_DIM // 2]
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_sum = pl.row_sum(kv_rms_sq)
            kv_rms_rowsum = pl.reshape(kv_rms_sum, [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        partial_mean = pl.mul(partial_sq, HEAD_DIM_INV)
        variance_row = pl.add(partial_mean, EPS)
        variance = pl.reshape(variance_row, [RMS_PAD_TILE, 1])
        rms = pl.sqrt(variance)
        inv_rms = pl.recip(rms)
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma_bf16 = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(gamma_bf16, pl.FP32)
            kv_rms_scaled = pl.row_expand_mul(kv_norm_chunk, inv_rms)
            normed_chunk = pl.col_expand_mul(kv_rms_scaled, gamma)
            normed_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_chunk

        kv_rope_norm = pooled_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope_bf16 = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(gamma_rope_bf16, pl.FP32)
        # out[j] = n[j] * cos[j] + n[j ^ 1] * sign[j] * sin[j]
        rope_rms_scaled = pl.row_expand_mul(kv_rope_norm, inv_rms)
        rope_normed = pl.col_expand_mul(rope_rms_scaled, gamma_rope)
        rope_even = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even_base = pl.mul(rope_even, cos_b)
        rope_odd_neg = pl.neg(rope_odd)
        rope_even_delta = pl.mul(rope_odd_neg, sin_b)
        rope_even_rot = pl.add(rope_even_base, rope_even_delta)
        rope_odd_base = pl.mul(rope_odd, cos_b)
        rope_odd_delta = pl.mul(rope_even, sin_b)
        rope_odd_rot = pl.add(rope_odd_base, rope_odd_delta)
        rope_rot = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_rot = pl.tensor.scatter(rope_even_rot, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_rot)
        rope_rot = pl.tensor.scatter(rope_odd_rot, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_rot)
        normed_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_rot

        for inner in pl.range(B):
            write_c_idx = inner
            write_first_pos = pl.read(position_ids, [write_c_idx, 0])
            write_pos = write_first_pos % COMPRESS_RATIO
            if write_pos + S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - write_pos
                kv_row_fp32 = normed_kv[inner : inner + 1, 0 : HEAD_DIM]
                cache_row_i64 = pl.read(cmp_slot_mapping, [write_c_idx, boundary_s])
                if cache_row_i64 >= 0:
                    cache_row = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat[write_c_idx * S : write_c_idx * S + 1, :] = kv_row_fp32
                    kv_row_bf16 = pl.cast(kv_row_fp32, target_type=pl.BF16, mode="rint")
                    cmp_kv_cache_flat[cache_row : cache_row + 1, :] = kv_row_bf16

    kv_out = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv_out


@pl.jit
def compressor_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, S, HEAD_DIM], pl.FP32]],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_kv_cache: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    late_dep = pl.system.task_dummy(deps=[])
    compressor_ratio4(
        x, kv, compress_state, compress_state_block_table,
        wkv, wgate, ape, norm_w, cos, sin,
        cmp_kv_cache, position_ids, cmp_slot_mapping, state_slot_mapping,
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
    bsz, _, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM

    kv = x @ wkv.t()
    score = x @ wgate.t()

    pooled = torch.zeros(bsz, 1, HEAD_DIM, dtype=torch.float32, device=x.device)
    should_compress_rows = torch.zeros(bsz, dtype=torch.bool, device=x.device)

    def read_front_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, :HEAD_DIM],
            compress_state[blk_id, intra, OUT_DIM:OUT_DIM + HEAD_DIM],
        )

    def read_back_state(b, abs_pos):
        blk_id = int(compress_state_block_table[b, abs_pos // COMPRESS_STATE_BLOCK_SIZE].item())
        if blk_id < 0:
            return (
                torch.zeros(HEAD_DIM, dtype=torch.float32, device=x.device),
                torch.full((HEAD_DIM,), float("-inf"), dtype=torch.float32, device=x.device),
            )
        intra = abs_pos % COMPRESS_STATE_BLOCK_SIZE
        return (
            compress_state[blk_id, intra, HEAD_DIM:OUT_DIM],
            compress_state[blk_id, intra, OUT_DIM + HEAD_DIM:],
        )

    for b in range(bsz):
        first_pos = int(position_ids[b, 0].item())
        pre_tokens = min(S, ratio - (first_pos % ratio))
        boundary_s = ratio - 1 - (first_pos % ratio)
        should_compress = 0 <= boundary_s < S
        boundary_end = first_pos + pre_tokens - 1
        cur_window_start = boundary_end - ratio + 1
        prev_window_start = cur_window_start - ratio

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
        cos_v, sin_v = cos[b].view(-1), sin[b].view(-1)
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

        kv_b = torch.cat([kv_b[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

        cmp_row = int(cmp_slot_mapping[b, boundary_s].item())
        if cmp_row >= 0:
            tensors["kv"][b : b + 1, 0:1, :] = kv_b
            blk_id = cmp_row // BLOCK_SIZE
            cmp_kv_cache[blk_id, cmp_row % BLOCK_SIZE, 0] = kv_b[0, 0]

    tensors["cmp_kv_cache"][:] = cmp_kv_cache


def build_tensor_specs(start_pos=None):
    import torch
    from decode_metadata import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
    )
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
        return block_table(
            batch=B,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
        )
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0245
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0388
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1243
    def init_norm_w():
        return 0.9666 + 0.1929 * torch.randn(HEAD_DIM)
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
        return block_table(
            batch=B,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )
    def init_default_start_pos():
        return csa_decode_start_set(
            batch=B, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=COMPRESS_STATE_BLOCK_SIZE)
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
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, S, HEAD_DIM], torch.float32),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [B, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [B, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("cmp_kv_cache", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv_cache),
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
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch CSA set that includes the 8k point.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
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
