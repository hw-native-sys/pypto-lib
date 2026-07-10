# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared DeepSeek-V4 compressor scheduling and compute primitives."""

import pypto.language as pl

from config import FLASH as M


EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = M.nope_head_dim

COMPRESSOR_RMS_ROW_TILE = 8
COMPRESSOR_HEAD_TILE = 64
COMPRESSOR_RMS_ROWS = pl.dynamic("COMPRESSOR_COMMON_RMS_ROWS")
COMPRESSOR_FINALIZE_ROWS = pl.dynamic("COMPRESSOR_FINALIZE_ROWS")
COMPRESSOR_FINALIZE_CACHE_ROWS = pl.dynamic("COMPRESSOR_FINALIZE_CACHE_ROWS")
SCHEDULE_TOKENS = pl.dynamic("COMPRESSOR_SCHEDULE_TOKENS")
SCHEDULE_WRITES = pl.dynamic("COMPRESSOR_SCHEDULE_WRITES")
SCHEDULE_ROPE_ROWS = pl.dynamic("COMPRESSOR_SCHEDULE_ROPE_ROWS")


@pl.jit.inline
def build_prefill_write_schedule(
    position_ids: pl.Tensor[[SCHEDULE_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[SCHEDULE_TOKENS], pl.INT64],
    num_tokens: pl.Scalar[pl.INT32],
    write_pos_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    state_table_row_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
):
    token_rows = pl.tensor.dim(position_ids, 0)
    write_rows = pl.tensor.dim(write_pos_map, 1)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cmp_write_schedule"):
        for init_i in pl.range(write_rows):
            pl.write(write_pos_map, [0, init_i], pl.cast(0, pl.INT32))
            pl.write(write_dst_map, [0, init_i], pl.cast(-1, pl.INT32))
            pl.write(state_table_row_map, [0, init_i], pl.cast(-1, pl.INT32))
        map_seen = pl.cast(0, pl.INDEX)
        for map_t in pl.range(token_rows):
            if map_t < num_tokens:
                map_slot_raw = pl.read(cmp_slot_mapping, [map_t])
                if map_slot_raw >= 0:
                    if map_seen < write_rows:
                        pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_t]))
                        pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                        pl.write(state_table_row_map, [0, map_seen], pl.cast(0, pl.INT32))
                        map_seen = map_seen + 1


@pl.jit.inline
def build_decode_padded_write_schedule(
    position_ids: pl.Tensor[[SCHEDULE_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[SCHEDULE_TOKENS], pl.INT64],
    seq_len: pl.Scalar[pl.INT32],
    compress_ratio: pl.Scalar[pl.INT32],
    rms_tile: pl.Scalar[pl.INT32],
    rms_pad_tile: pl.Scalar[pl.INT32],
    write_pos_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    kv_out_row_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    state_table_row_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
):
    token_rows = pl.tensor.dim(position_ids, 0)
    write_rows = pl.tensor.dim(write_pos_map, 1)
    seq_len_i = pl.cast(seq_len, pl.INDEX)
    compress_ratio_i = pl.cast(compress_ratio, pl.INDEX)
    rms_tile_i = pl.cast(rms_tile, pl.INDEX)
    rms_pad_tile_i = pl.cast(rms_pad_tile, pl.INDEX)
    batch_rows = token_rows // seq_len_i
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_cmp_write_schedule"):
        for init_i in pl.range(write_rows):
            pl.write(write_pos_map, [0, init_i], pl.cast(0, pl.INT32))
            pl.write(write_dst_map, [0, init_i], pl.cast(-1, pl.INT32))
            pl.write(kv_out_row_map, [0, init_i], pl.cast(-1, pl.INT32))
            pl.write(state_table_row_map, [0, init_i], pl.cast(-1, pl.INT32))
        for b in pl.range(batch_rows):
            base_t = b * seq_len_i
            first_pos = pl.read(position_ids, [base_t])
            pos_in_window = pl.cast(first_pos % compress_ratio, pl.INDEX)
            if pos_in_window + seq_len_i >= compress_ratio_i:
                boundary_s = compress_ratio_i - 1 - pos_in_window
                token_t = base_t + boundary_s
                dst_raw = pl.read(cmp_slot_mapping, [token_t])
                if dst_raw >= 0:
                    pad_row = (b // rms_tile_i) * rms_pad_tile_i + (b % rms_tile_i)
                    if pad_row < write_rows:
                        pl.write(
                            write_pos_map,
                            [0, pad_row],
                            pl.cast(first_pos + pl.cast(boundary_s, pl.INT32), pl.INT32),
                        )
                        pl.write(write_dst_map, [0, pad_row], pl.cast(dst_raw, pl.INT32))
                        pl.write(kv_out_row_map, [0, pad_row], pl.cast(base_t, pl.INT32))
                        pl.write(state_table_row_map, [0, pad_row], pl.cast(b, pl.INT32))


@pl.jit.inline
def gather_compressor_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    write_pos_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    write_dst_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    compress_ratio: pl.Scalar[pl.INT32],
    cos_b: pl.Tensor[[SCHEDULE_ROPE_ROWS, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[SCHEDULE_ROPE_ROWS, ROPE_HALF], pl.FP32],
):
    write_rows = pl.tensor.dim(write_pos_map, 1)
    rope_rows = pl.tensor.dim(cos_b, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cmp_rope_schedule"):
        for rope_i in pl.range(rope_rows):
            cos_b[rope_i : rope_i + 1, 0:ROPE_HALF] = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
            sin_b[rope_i : rope_i + 1, 0:ROPE_HALF] = pl.full([1, ROPE_HALF], dtype=pl.FP32, value=0.0)
            if rope_i < write_rows:
                write_slot_raw = pl.read(write_dst_map, [0, rope_i])
                if write_slot_raw >= 0:
                    cmp_pos = pl.cast(pl.read(write_pos_map, [0, rope_i]) + 1 - compress_ratio, pl.INDEX)
                    cos_b[rope_i : rope_i + 1, 0:ROPE_HALF] = pl.cast(
                        freqs_cos[cmp_pos : cmp_pos + 1, 0:ROPE_HALF],
                        target_type=pl.FP32,
                    )
                    sin_b[rope_i : rope_i + 1, 0:ROPE_HALF] = pl.cast(
                        freqs_sin[cmp_pos : cmp_pos + 1, 0:ROPE_HALF],
                        target_type=pl.FP32,
                    )


@pl.jit.inline
def compressor_rmsnorm_rope(
    pooled_kv: pl.Tensor[[COMPRESSOR_RMS_ROWS, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos_b: pl.Tensor[[COMPRESSOR_RMS_ROWS, ROPE_HALF], pl.FP32],
    sin_b: pl.Tensor[[COMPRESSOR_RMS_ROWS, ROPE_HALF], pl.FP32],
    normed_kv: pl.Tensor[[COMPRESSOR_RMS_ROWS, HEAD_DIM], pl.FP32],
):
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    rows = pl.tensor.dim(pooled_kv, 0)
    for rt in pl.spmd(rows // COMPRESSOR_RMS_ROW_TILE, name_hint="rmsnorm_rope"):
        r0 = rt * COMPRESSOR_RMS_ROW_TILE
        partial_sq = pl.full([1, COMPRESSOR_RMS_ROW_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // COMPRESSOR_HEAD_TILE, stage=2):
            rms_h0 = rms_kb * COMPRESSOR_HEAD_TILE
            kv_rms_chunk = pooled_kv[
                r0 : r0 + COMPRESSOR_RMS_ROW_TILE,
                rms_h0 : rms_h0 + COMPRESSOR_HEAD_TILE,
            ]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(kv_rms_sq), [1, COMPRESSOR_RMS_ROW_TILE]),
            )
        variance = pl.reshape(
            pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS),
            [COMPRESSOR_RMS_ROW_TILE, 1],
        )
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // COMPRESSOR_HEAD_TILE, stage=2):
            norm_h0 = rms_kb * COMPRESSOR_HEAD_TILE
            kv_norm_chunk = pooled_kv[
                r0 : r0 + COMPRESSOR_RMS_ROW_TILE,
                norm_h0 : norm_h0 + COMPRESSOR_HEAD_TILE,
            ]
            gamma = pl.cast(
                norm_w_2d[:, norm_h0 : norm_h0 + COMPRESSOR_HEAD_TILE],
                pl.FP32,
            )
            normed_chunk = pl.col_expand_mul(
                pl.row_expand_mul(kv_norm_chunk, inv_rms),
                gamma,
            )
            normed_kv[
                r0 : r0 + COMPRESSOR_RMS_ROW_TILE,
                norm_h0 : norm_h0 + COMPRESSOR_HEAD_TILE,
            ] = normed_chunk

        kv_rope_norm = pooled_kv[
            r0 : r0 + COMPRESSOR_RMS_ROW_TILE,
            NOPE_HEAD_DIM:HEAD_DIM,
        ]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full(
            [COMPRESSOR_RMS_ROW_TILE, ROPE_HEAD_DIM],
            dtype=pl.FP32,
            value=1.0,
        )
        rope_col = pl.col_expand_mul(
            rope_ones,
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        rope_dup_f = pl.cast(
            pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))
        rope_swap_idx = pl.cast(
            pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)),
            target_type=pl.INT32,
        )
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)
        cos_il = pl.gather(
            cos_b[r0 : r0 + COMPRESSOR_RMS_ROW_TILE, 0:ROPE_HALF],
            dim=-1,
            index=rope_dup_idx,
        )
        sin_il = pl.gather(
            sin_b[r0 : r0 + COMPRESSOR_RMS_ROW_TILE, 0:ROPE_HALF],
            dim=-1,
            index=rope_dup_idx,
        )
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(
            pl.mul(rope_normed, cos_il),
            pl.mul(pl.mul(swapped, rope_sign), sin_il),
        )
        normed_kv[
            r0 : r0 + COMPRESSOR_RMS_ROW_TILE,
            NOPE_HEAD_DIM:HEAD_DIM,
        ] = rope_rot
    return normed_kv


@pl.jit.inline
def finalize_compressor_writes(
    normed_kv: pl.Tensor[[COMPRESSOR_FINALIZE_ROWS, HEAD_DIM], pl.FP32],
    write_dst_map: pl.Tensor[[1, COMPRESSOR_FINALIZE_ROWS], pl.INT32],
    cmp_kv_cache_flat: pl.Tensor[[COMPRESSOR_FINALIZE_CACHE_ROWS, HEAD_DIM], pl.BF16],
    rows_per_task: pl.Scalar[pl.INT32],
    keepalive_invalid: pl.Scalar[pl.INT32],
):
    """Write scheduled compressed rows while preserving each caller's task tiling."""
    write_rows = pl.tensor.dim(write_dst_map, 1)
    cache_rows = pl.tensor.dim(cmp_kv_cache_flat, 0)
    row_tile = pl.cast(rows_per_task, pl.INDEX)
    for final_block in pl.spmd((write_rows + row_tile - 1) // row_tile, name_hint="compressor_cache_write"):
        final_base = final_block * row_tile
        for final_dt in pl.range(row_tile):
            final_row = final_base + final_dt
            if final_row < write_rows:
                dst_row_raw = pl.read(write_dst_map, [0, final_row])
                if dst_row_raw >= 0:
                    dst_row = pl.cast(dst_row_raw, pl.INDEX)
                    cmp_kv_cache_flat[dst_row : dst_row + 1, 0:HEAD_DIM] = pl.cast(
                        normed_kv[final_row : final_row + 1, 0:HEAD_DIM],
                        target_type=pl.BF16,
                        mode="rint",
                    )
                else:
                    if keepalive_invalid != 0:
                        keepalive_row = cache_rows - write_rows + final_row
                        cmp_kv_cache_flat[
                            keepalive_row : keepalive_row + 1,
                            0:HEAD_DIM,
                        ] = cmp_kv_cache_flat[
                            keepalive_row : keepalive_row + 1,
                            0:HEAD_DIM,
                        ]
    return cmp_kv_cache_flat
