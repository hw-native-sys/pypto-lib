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
    ACTIVE as M,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_IDX_BLOCK_NUM,
    DECODE_SEQ,
    FP32_NEG_INF,
    IDX_CACHE_MAX_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from golden import mapped_pool_ratio_allclose

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
IDX_CACHE_BLOCK_NUM = DECODE_IDX_BLOCK_NUM

# tiling
K_TILE = 512
OUT_TILE = 64
PROJ_OUT_TILE = 32  # Cube N tile.
assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"
MM_B_TILE = 16
BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
HEAD_TILE = 64
RMS_PAD_TILE = 16


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.InOut[
        pl.Tensor[
            [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)

    with pl.spmd(
        BS_PAD * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]
    ) as _kv_score_tid:
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
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

    compress_state_flat = pl.reshape(
        compress_state, [COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM]
    )
    pooled_kv = pl.create_tensor([RMS_PAD_TILE, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool"):
        for c_idx in pl.range(B):
            for s_sc in pl.pipeline(S, stage=2):
                token_pos = pl.read(position_ids, [c_idx, s_sc])
                state_row_i64 = pl.read(inner_state_slot_mapping, [c_idx, s_sc])
                proj_row = c_idx * S + s_sc
                token_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                if state_row_i64 >= 0:
                    state_row = pl.cast(state_row_i64, pl.INDEX)
                    kv_tile = kv_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
                    score_tile = score_proj_pad[proj_row : proj_row + 1, 0 : OUT_DIM]
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
                for hb in pl.range(HEAD_DIM // HEAD_TILE):
                    h0 = hb * HEAD_TILE
                    last_abs = cur_window_start_b + COMPRESS_RATIO - 1
                    last_blk_off = last_abs // COMPRESS_STATE_BLOCK_SIZE
                    last_intra = last_abs % COMPRESS_STATE_BLOCK_SIZE
                    last_blk_raw = pl.read(compress_state_block_table, [c_idx, last_blk_off])
                    last_blk_id = pl.cast(last_blk_raw, pl.INDEX)
                    last_row = last_blk_id * COMPRESS_STATE_BLOCK_SIZE + last_intra
                    last_col0 = OUT_DIM + HEAD_DIM + h0
                    mi = compress_state_flat[last_row : last_row + 1, last_col0 : last_col0 + HEAD_TILE]
                    li_diff = pl.sub(mi, mi)
                    li = pl.exp(li_diff)
                    oi = compress_state_flat[last_row : last_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]

                    for s in pl.range(0, COMPRESS_RATIO):
                        prev_abs = prev_window_start_b + s
                        front_score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                        front_kv = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                        if first_pos_b >= COMPRESS_RATIO:
                            prev_blk_off = prev_abs // COMPRESS_STATE_BLOCK_SIZE
                            prev_intra = prev_abs % COMPRESS_STATE_BLOCK_SIZE
                            prev_blk_raw = pl.read(compress_state_block_table, [c_idx, prev_blk_off])
                            prev_blk_id = pl.cast(prev_blk_raw, pl.INDEX)
                            prev_row = prev_blk_id * COMPRESS_STATE_BLOCK_SIZE + prev_intra
                            front_score = compress_state_flat[prev_row : prev_row + 1, OUT_DIM + h0 : OUT_DIM + h0 + HEAD_TILE]
                            front_kv = compress_state_flat[prev_row : prev_row + 1, h0 : h0 + HEAD_TILE]
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
                        back_col0 = OUT_DIM + HEAD_DIM + h0
                        back_score = compress_state_flat[cur_row : cur_row + 1, back_col0 : back_col0 + HEAD_TILE]
                        back_kv = compress_state_flat[cur_row : cur_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
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
                    pooled_kv[pad_idx : pad_idx + 1, h0 : h0 + HEAD_TILE] = pooled_chunk

    cmp_cos_il = pl.create_tensor([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cmp_rope_tables"):
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        cos_b[0:B, 0 : ROPE_HEAD_DIM // 2] = cos[0:B, 0 : ROPE_HEAD_DIM // 2]
        sin_b[0:B, 0 : ROPE_HEAD_DIM // 2] = sin[0:B, 0 : ROPE_HEAD_DIM // 2]
        cos_il = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_il = pl.tensor.scatter(cos_b, mask_pattern=pl.tile.MaskPattern.P0101, dst=cos_il)
        cos_il = pl.tensor.scatter(cos_b, mask_pattern=pl.tile.MaskPattern.P1010, dst=cos_il)
        cmp_cos_il[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM] = cos_il
        sin_neg = pl.neg(sin_b)
        sin_signed = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_signed = pl.tensor.scatter(sin_neg, mask_pattern=pl.tile.MaskPattern.P0101, dst=sin_signed)
        sin_signed = pl.tensor.scatter(sin_b, mask_pattern=pl.tile.MaskPattern.P1010, dst=sin_signed)
        cmp_sin_signed[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM] = sin_signed

    normed_kv = pl.create_tensor([RMS_PAD_TILE, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope"):
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
            normed_bf16 = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")
            normed_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = normed_bf16

        kv_rope_norm = pooled_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope_bf16 = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(gamma_rope_bf16, pl.FP32)
        # out[j] = n[j] * cos[j] + n[j ^ 1] * sign[j] * sin[j]
        rope_rms_scaled = pl.row_expand_mul(kv_rope_norm, inv_rms)
        rope_normed = pl.col_expand_mul(rope_rms_scaled, gamma_rope)
        rope_even = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        swapped = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        swapped = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=swapped)
        swapped = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=swapped)
        cmp_cos = cmp_cos_il[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM]
        rope_cos = pl.mul(rope_normed, cmp_cos)
        cmp_sin = cmp_sin_signed[0:RMS_PAD_TILE, 0:ROPE_HEAD_DIM]
        rope_sin = pl.mul(swapped, cmp_sin)
        rope_rot = pl.add(rope_cos, rope_sin)
        rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
        normed_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_bf16

    kv_final = pl.create_tensor([RMS_PAD_TILE, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
        kv_proj_tile = normed_kv[0 : RMS_PAD_TILE, 0 : HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[0 : RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
        kv_final_tile = kv_final[0 : RMS_PAD_TILE, 0 : HEAD_DIM]
        kv_blk_bf16 = pl.cast(kv_final_tile, target_type=pl.BF16, mode="rint")
        kv_blk_f32 = pl.cast(kv_blk_bf16, target_type=pl.FP32)
        kv_abs = pl.abs(kv_blk_f32)
        kv_row_max = pl.row_max(kv_abs)
        kv_amax_row = pl.reshape(kv_row_max, [1, RMS_PAD_TILE])
        kv_amax_floor = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        kv_amax = pl.maximum(kv_amax_row, kv_amax_floor)
        kv_scale_max = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        kv_scale_q_row = pl.div(kv_scale_max, kv_amax)
        kv_scale_dq_row = pl.recip(kv_scale_q_row)
        kv_scale_dq_col = pl.reshape(kv_scale_dq_row, [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for inner in pl.range(B):
            write_c_idx = inner
            write_first_pos = pl.read(position_ids, [write_c_idx, 0])
            write_pos = write_first_pos % COMPRESS_RATIO
            if write_pos + S >= COMPRESS_RATIO:
                boundary_s = COMPRESS_RATIO - 1 - write_pos
                kv_row_fp32 = kv_final[inner : inner + 1, 0 : HEAD_DIM]
                cache_row_i64 = pl.read(idx_slot_mapping, [write_c_idx, boundary_s])
                if cache_row_i64 >= 0:
                    cache_row = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat[write_c_idx * S : write_c_idx * S + 1, :] = kv_row_fp32
                    kv_i8_row = kv_i8_blk[inner : inner + 1, :]
                    idx_kv_cache_flat[cache_row : cache_row + 1, :] = kv_i8_row
                    scale_dq = pl.read(kv_scale_dq_col, [inner, 0])
                    pl.write(idx_kv_scale_flat, [cache_row, 0], scale_dq)

    kv_out = pl.reshape(kv_flat, [B, S, HEAD_DIM])
    return kv_out, compress_state, idx_kv_cache, idx_kv_scale


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
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
):
    late_dep = pl.system.task_dummy(deps=[])
    indexer_compressor(
        x, kv, compress_state, compress_state_block_table,
        wkv, wgate, ape, norm_w, cos, sin, hadamard,
        idx_kv_cache, idx_kv_scale,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
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
            state_row = int(inner_state_slot_mapping[b, s].item())
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

        kv_b = kv_b.to(torch.bfloat16).float() @ hadamard

        cache_row = int(idx_slot_mapping[b, boundary_s].item())
        if cache_row >= 0:
            tensors["kv"][b : b + 1, 0:1, :] = kv_b
            blk_id = cache_row // BLOCK_SIZE
            intra = cache_row % BLOCK_SIZE
            # Quantize the BF16-rounded row to INT8 with a per-position scale.
            row_bf16 = kv_b[0, 0].to(torch.bfloat16).float()
            amax = row_bf16.abs().amax().clamp_min(INT8_AMAX_EPS)
            scale_q = INT8_SCALE_MAX / amax
            idx_kv_cache[blk_id, intra, 0] = torch.round(row_bf16 * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
            idx_kv_scale[blk_id, intra, 0, 0] = 1.0 / scale_q

    tensors["idx_kv_cache"][:] = idx_kv_cache
    tensors["idx_kv_scale"][:] = idx_kv_scale


def mapped_idx_cache_ratio_allclose(*, atol, rtol, max_error_ratio):
    """Compare index-cache writes selected by ``idx_slot_mapping``."""
    return mapped_pool_ratio_allclose(
        "idx_slot_mapping",
        mapping_shape=(B, S),
        block_size=BLOCK_SIZE,
        pool_name="index cache",
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )


def mapped_inner_state_ratio_allclose(*, atol, rtol, max_error_ratio):
    """Compare inner-state writes selected by ``inner_state_slot_mapping``."""
    return mapped_pool_ratio_allclose(
        "inner_state_slot_mapping",
        mapping_shape=(B, S),
        block_size=COMPRESS_STATE_BLOCK_SIZE,
        pool_name="inner compressor state",
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )


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
        return torch.randn(OUT_DIM, D) * 0.0293
    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0512
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1528
    def init_norm_w():
        return 0.6850 + 0.2610 * torch.randn(HEAD_DIM)
    def init_rope_positions():
        first_pos = init_position_ids().to(torch.int64)[:, 0]
        cmp_offset = COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)
        return (first_pos + cmp_offset - COMPRESS_RATIO).to(torch.int64)
    def init_cos():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[0]
    def init_sin():
        return materialize_half_rope_tables(shared_freqs_cos, shared_freqs_sin, init_rope_positions())[1]
    def init_hadamard():
        return torch.rand(HEAD_DIM, HEAD_DIM) * (HEAD_DIM ** -0.5)
    def init_idx_kv_cache():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.int8)
    def init_idx_kv_scale():
        return torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)
    def init_idx_block_table():
        return block_table(
            batch=B,
            table_blocks=IDX_CACHE_MAX_BLOCKS,
            physical_blocks=IDX_CACHE_MAX_BLOCKS,
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
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.int8, init_value=init_idx_kv_cache),
        TensorSpec("idx_kv_scale", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=init_idx_kv_scale),
        TensorSpec("position_ids", [B, S], torch.int32, init_value=init_position_ids),
        TensorSpec("idx_slot_mapping", [B, S], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [B, S], torch.int64, init_value=init_inner_state_slot_mapping),
    ]


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
    args = parser.parse_args()

    result = run(
        fn=compressor_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_compressor,
        runtime_dir=args.runtime_dir,
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
            "compress_state": mapped_inner_state_ratio_allclose(
                atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            # Compare rows selected by the current slot mappings.
            "idx_kv_cache": mapped_idx_cache_ratio_allclose(
                atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": mapped_idx_cache_ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
