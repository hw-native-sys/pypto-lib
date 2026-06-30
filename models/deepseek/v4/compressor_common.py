# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared DeepSeek-V4 compressor epilogues."""

import pypto.language as pl

from config import FLASH as M


EPS = M.rms_norm_eps
HEAD_TILE = 64
MAIN_HEAD_DIM = M.head_dim
MAIN_HEAD_DIM_INV = 1.0 / MAIN_HEAD_DIM
INDEX_HEAD_DIM = M.index_head_dim
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
MAIN_NOPE_HEAD_DIM = M.nope_head_dim
INDEX_NOPE_HEAD_DIM = M.index_nope_head_dim
TILE16 = 16
TILE8 = 8
PREFILL_TOKEN_TILE128 = 128
PREFILL_WRITE_TILE32 = 32
ROWS_DYN = pl.dynamic("COMPRESSOR_ROWS_DYN")


@pl.jit.inline
def rmsnorm_rope_main16_fp32(
    pooled_kv: pl.Tensor[[ROWS_DYN, MAIN_HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[MAIN_HEAD_DIM], pl.BF16],
    cos_b: pl.Tensor[[TILE16, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[TILE16, ROPE_HALF], pl.FP32],
    normed_kv: pl.Tensor[[ROWS_DYN, MAIN_HEAD_DIM], pl.FP32],
    row_base: pl.Scalar[pl.INDEX],
):
    partial_sq = pl.full([1, TILE16], dtype=pl.FP32, value=0.0)
    for k0 in pl.range(0, MAIN_HEAD_DIM, HEAD_TILE):
        kv_rms_chunk = pooled_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE]
        kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
        partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, TILE16]))

    variance = pl.reshape(pl.add(pl.mul(partial_sq, MAIN_HEAD_DIM_INV), EPS), [TILE16, 1])
    inv_rms = pl.recip(pl.sqrt(variance))
    for k0 in pl.range(0, MAIN_NOPE_HEAD_DIM, HEAD_TILE):
        kv_norm_chunk = pooled_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE]
        gamma = pl.cast(pl.reshape(norm_w[k0 : k0 + HEAD_TILE], [1, HEAD_TILE]), pl.FP32)
        normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
        normed_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE] = normed_chunk

    kv_rope_norm = pooled_kv[row_base : row_base + TILE16, MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM]
    gamma_rope = pl.cast(pl.reshape(norm_w[MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM], [1, ROPE_HEAD_DIM]), pl.FP32)
    rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
    rope_ones = pl.full([TILE16, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
    rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
    rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
    rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
    rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
    rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
    rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
    cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
    sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
    swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
    rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
    normed_kv[row_base : row_base + TILE16, MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM] = rope_rot
    return normed_kv


@pl.jit.inline
def rmsnorm_rope_main8_fp32(
    pooled_kv: pl.Tensor[[ROWS_DYN, MAIN_HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[MAIN_HEAD_DIM], pl.BF16],
    cos_b: pl.Tensor[[TILE8, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[TILE8, ROPE_HALF], pl.FP32],
    normed_kv: pl.Tensor[[ROWS_DYN, MAIN_HEAD_DIM], pl.FP32],
    row_base: pl.Scalar[pl.INDEX],
):
    partial_sq = pl.full([1, TILE8], dtype=pl.FP32, value=0.0)
    for k0 in pl.range(0, MAIN_HEAD_DIM, HEAD_TILE):
        kv_rms_chunk = pooled_kv[row_base : row_base + TILE8, k0 : k0 + HEAD_TILE]
        kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
        partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, TILE8]))

    variance = pl.reshape(pl.add(pl.mul(partial_sq, MAIN_HEAD_DIM_INV), EPS), [TILE8, 1])
    inv_rms = pl.recip(pl.sqrt(variance))
    for k0 in pl.range(0, MAIN_NOPE_HEAD_DIM, HEAD_TILE):
        kv_norm_chunk = pooled_kv[row_base : row_base + TILE8, k0 : k0 + HEAD_TILE]
        gamma = pl.cast(pl.reshape(norm_w[k0 : k0 + HEAD_TILE], [1, HEAD_TILE]), pl.FP32)
        normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
        normed_kv[row_base : row_base + TILE8, k0 : k0 + HEAD_TILE] = normed_chunk

    kv_rope_norm = pooled_kv[row_base : row_base + TILE8, MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM]
    gamma_rope = pl.cast(pl.reshape(norm_w[MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM], [1, ROPE_HEAD_DIM]), pl.FP32)
    rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
    rope_ones = pl.full([TILE8, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
    rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
    rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
    rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
    rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
    rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
    rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
    cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
    sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
    swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
    rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
    normed_kv[row_base : row_base + TILE8, MAIN_NOPE_HEAD_DIM:MAIN_HEAD_DIM] = rope_rot
    return normed_kv


@pl.jit.inline
def rmsnorm_rope_index16_bf16(
    pooled_kv: pl.Tensor[[ROWS_DYN, INDEX_HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    cos_b: pl.Tensor[[TILE16, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[TILE16, ROPE_HALF], pl.FP32],
    normed_kv: pl.Tensor[[ROWS_DYN, INDEX_HEAD_DIM], pl.BF16],
    row_base: pl.Scalar[pl.INDEX],
):
    partial_sq = pl.full([1, TILE16], dtype=pl.FP32, value=0.0)
    for k0 in pl.range(0, INDEX_HEAD_DIM, HEAD_TILE):
        kv_rms_chunk = pooled_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE]
        kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
        partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, TILE16]))

    variance = pl.reshape(pl.add(pl.mul(partial_sq, INDEX_HEAD_DIM_INV), EPS), [TILE16, 1])
    inv_rms = pl.recip(pl.sqrt(variance))
    for k0 in pl.range(0, INDEX_NOPE_HEAD_DIM, HEAD_TILE):
        kv_norm_chunk = pooled_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE]
        gamma = pl.cast(pl.reshape(norm_w[k0 : k0 + HEAD_TILE], [1, HEAD_TILE]), pl.FP32)
        normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
        normed_kv[row_base : row_base + TILE16, k0 : k0 + HEAD_TILE] = pl.cast(
            normed_chunk,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_rope_norm = pooled_kv[row_base : row_base + TILE16, INDEX_NOPE_HEAD_DIM:INDEX_HEAD_DIM]
    gamma_rope = pl.cast(pl.reshape(norm_w[INDEX_NOPE_HEAD_DIM:INDEX_HEAD_DIM], [1, ROPE_HEAD_DIM]), pl.FP32)
    rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
    rope_ones = pl.full([TILE16, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
    rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
    rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
    rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
    rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
    rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)
    rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
    cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
    sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
    swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
    rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
    normed_kv[row_base : row_base + TILE16, INDEX_NOPE_HEAD_DIM:INDEX_HEAD_DIM] = pl.cast(
        rope_rot,
        target_type=pl.BF16,
        mode="rint",
    )
    return normed_kv


@pl.jit.inline
def build_prefill_write_map_128x32(
    position_ids: pl.Tensor[[PREFILL_TOKEN_TILE128], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKEN_TILE128], pl.INT64],
    num_tokens: pl.Scalar[pl.INT32],
    write_pos_map: pl.Tensor[[1, PREFILL_WRITE_TILE32], pl.INT32],
    write_dst_map: pl.Tensor[[1, PREFILL_WRITE_TILE32], pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c4_write_map"):
        write_pos_map[0:1, 0:PREFILL_WRITE_TILE32] = pl.full([1, PREFILL_WRITE_TILE32], dtype=pl.INT32, value=0)
        write_dst_map[0:1, 0:PREFILL_WRITE_TILE32] = pl.full([1, PREFILL_WRITE_TILE32], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_w in pl.range(PREFILL_TOKEN_TILE128):
            if map_w < num_tokens:
                map_slot_raw = pl.read(slot_mapping, [map_w])
                if map_slot_raw >= 0:
                    pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_w]))
                    pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                    map_seen = map_seen + 1
    return write_pos_map


@pl.jit.inline
def index_hadamard16_bf16_to_fp32(
    normed_kv: pl.Tensor[[ROWS_DYN, INDEX_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_HEAD_DIM, INDEX_HEAD_DIM], pl.BF16],
    final_kv: pl.Tensor[[ROWS_DYN, INDEX_HEAD_DIM], pl.FP32],
    row_base: pl.Scalar[pl.INDEX],
):
    kv_proj_tile = normed_kv[row_base : row_base + TILE16, 0:INDEX_HEAD_DIM]
    for o0 in pl.range(0, INDEX_HEAD_DIM, HEAD_TILE):
        hadamard_tile = hadamard[0:INDEX_HEAD_DIM, o0 : o0 + HEAD_TILE]
        kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
        final_kv[row_base : row_base + TILE16, o0 : o0 + HEAD_TILE] = kv_hadamard_acc
    return final_kv
