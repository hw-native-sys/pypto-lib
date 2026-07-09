# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 compressor write-schedule helpers."""

import pypto.language as pl

from config import FLASH as M


MAX_SEQ_LEN = M.max_position_embeddings
ROPE_FREQ_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_FREQ_DIM // 2

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
    kv_out_row_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
    state_table_row_map: pl.Tensor[[1, SCHEDULE_WRITES], pl.INT32],
):
    token_rows = pl.tensor.dim(position_ids, 0)
    write_rows = pl.tensor.dim(write_pos_map, 1)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cmp_write_schedule"):
        for init_i in pl.range(write_rows):
            pl.write(write_pos_map, [0, init_i], pl.cast(0, pl.INT32))
            pl.write(write_dst_map, [0, init_i], pl.cast(-1, pl.INT32))
            pl.write(kv_out_row_map, [0, init_i], pl.cast(-1, pl.INT32))
            pl.write(state_table_row_map, [0, init_i], pl.cast(-1, pl.INT32))
        map_seen = pl.cast(0, pl.INDEX)
        for map_t in pl.range(token_rows):
            if map_t < num_tokens:
                map_slot_raw = pl.read(cmp_slot_mapping, [map_t])
                if map_slot_raw >= 0:
                    if map_seen < write_rows:
                        pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_t]))
                        pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                        pl.write(kv_out_row_map, [0, map_seen], pl.cast(map_t, pl.INT32))
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
                        pl.write(write_pos_map, [0, pad_row], pl.cast(first_pos + pl.cast(boundary_s, pl.INT32), pl.INT32))
                        pl.write(write_dst_map, [0, pad_row], pl.cast(dst_raw, pl.INT32))
                        pl.write(kv_out_row_map, [0, pad_row], pl.cast(base_t, pl.INT32))
                        pl.write(state_table_row_map, [0, pad_row], pl.cast(b, pl.INT32))


@pl.jit.inline
def gather_compressor_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_FREQ_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_FREQ_DIM], pl.BF16],
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
