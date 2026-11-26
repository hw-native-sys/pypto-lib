# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO sparse MLA with packed-cache dequantization and sink-aware Cube attention."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

from collections.abc import Callable
import math


import torch


from .softmax_precision import centered_exp_line, centered_exp_scores, normalized_quotient
from .gemm import bf16_dot_sequential, bf16_ordered_accumulate
from .compressor import golden_c1a_compressor
from .compressor import golden_c2a_compressor
from .qkv_proj_rope import golden_qkv_proj_rope as qkv_proj_rope
from .qkv_proj_rope import golden_rope_interleave as rope_interleave
from .o_proj import golden_o_proj, golden_grouped_output
from .qkv_proj_rope import golden_rope_tail
from .rmsnorm import golden_rms_norm
from golden import TensorSpec
from .utils import _publish_quantized_cache
from .quant import dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight, pack_mx_b_scale, quantize_mxfp4_cache, quantize_mxfp8_cache
from .gemm import golden_dynamic_int8
from .utils import packed_query_lengths, window_metadata


import pypto.language as pl
from .quant import decode_fp4, decode_fp8, decode_fp8_cache_values, encode_fp8, fp4_bf16_bits, make_fp8_cache_table

from .config import FLASH as M, MAX_BATCH_PER_DP, TP_SIZE
from .dsl_types import CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN

# Dynamic shape variables.
SELECTED_ROWS = pl.dynamic("SPARSE_SELECTED_ROWS")
SWA_KEY_ROWS = pl.dynamic("SWA_KEY_ROWS")
SWA_RUN_WIDTH = pl.dynamic("SWA_RUN_WIDTH")
BF16_HEADS = pl.dynamic("SPARSE_BF16_HEADS")
BF16_PAGES_W = pl.dynamic("SPARSE_BF16_PAGES_W")
BF16_PAGES_C = pl.dynamic("SPARSE_BF16_PAGES_C")
CACHE_WINDOW_ROWS = pl.dynamic("SPARSE_CACHE_WINDOW_ROWS")
CACHE_COMPRESSED_ROWS = pl.dynamic("SPARSE_CACHE_COMPRESSED_ROWS")

# Model config.
HEAD_DIM = M.head_dim
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
INDEX_TOPK = M.index_topk
LOCAL_H = M.num_attention_heads // TP_SIZE
WINDOW_KEYS = M.sliding_window
SELECTED_KEYS = WINDOW_KEYS + INDEX_TOPK
SCALE = HEAD_DIM**-0.5
SWA_WINDOW = M.sliding_window
SWA_GROUPS = HEAD_DIM // 32
SWA_SCALE_BYTES = SWA_GROUPS * 2
SWA_PACKED_ROW_BYTES = HEAD_DIM + SWA_SCALE_BYTES

# Tiling.
CUBE_HEADS = max(16, LOCAL_H)
HEAD_TILE = min(32, CUBE_HEADS)
QUERY_HEAD_TILE = min(16, LOCAL_H)
KEY_TILE = 128
GATHER_KEY_TILE = 16
FP4_PAIR_TILE = GATHER_KEY_TILE * (HEAD_DIM // 2)
TOKEN_TILE = 512
WORKERS = 32
PREFILL_ATTEND_WORKERS = 64
GATHER_WORKERS = 64
SELECTED_GATHER_TILE = 64
CACHE_WORKERS = 128
PREFILL_FP8_CACHE_WORKERS = 48
WINDOW_PUBLISH_WORKER_TILE = 12
PREFILL_WINDOW_ROW_TILE = 8
PREFILL_WINDOW_GROUP_TILE = PREFILL_WINDOW_ROW_TILE * SWA_GROUPS
STREAM_WORKER_TILE = 24
STREAM_KEY_TILE = 2 * SELECTED_GATHER_TILE
STREAM_HEAD_TILE = LOCAL_H // 2
STREAM_FEATURE_TILE = HEAD_DIM // 2
STREAM_PV_DELAY = 2
STREAM_SLOTS = STREAM_PV_DELAY + 1
STREAM_KEY_READY = 0
STREAM_SCORE_READY = 1
STREAM_PROB_READY = 2
STREAM_PV_READY = 3
CACHE_MAX_ROWS = 32768
PACKED_QUERY_TILE = 32
PACKED_COHORT_TILE = 24
PACKED_HEAD_TILE = min(32, CUBE_HEADS) if TP_SIZE == 1 else 16
PACKED_KEY_TILE = 128
PACKED_ATTEND_SLOT_TILE = 1 if TP_SIZE == 1 else 2
PACKED_KEY_PARTS = SELECTED_KEYS // PACKED_KEY_TILE


# SWA tiling.
SWA_HEAD_TILE = 16
# Leave Vector space for the value-preserving query startup before QK.
SWA_SHARED_HEAD_TILE = 16
SWA_GATHER_TILE = 16
SWA_SELECTED_GATHER_TILE = 64 if TP_SIZE == 1 else SWA_GATHER_TILE
SWA_WORKER_TILE = 32
SWA_CROSS_CORE_SLOT_TILE = 3
SWA_RUN_TILE = 32
SWA_RUN_PARTS = SWA_WINDOW // SWA_RUN_TILE
SWA_RUN_DESCRIPTOR_WIDTH = 8


@pl.jit.inline(auto_scope=False)
def prefill_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Quantize token tiles and publish contiguous or scattered cache rows."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, SWA_GROUPS])
    tokens = pl.tensor.dim(kv, 0)
    slot_row = pl.reshape(slots, [1, tokens])
    blocks = (num_tokens + PREFILL_WINDOW_ROW_TILE - 1) // PREFILL_WINDOW_ROW_TILE
    workers = pl.max(1, pl.min(pl.min(blocks, PREFILL_FP8_CACHE_WORKERS), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="pub_win", deps=[cache_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            start = block * PREFILL_WINDOW_ROW_TILE
            rows = pl.min(PREFILL_WINDOW_ROW_TILE, num_tokens - start)
            ids = pl.load(slot_row, [0, start], [1, PREFILL_WINDOW_ROW_TILE], valid_shape=[1, rows])
            first = pl.tile.read(ids, [0, 0])
            contiguous = first >= 0
            for lane in pl.unroll(1, PREFILL_WINDOW_ROW_TILE):
                if lane < rows:
                    contiguous = contiguous and pl.tile.read(ids, [0, lane]) == first + lane

            source = pl.load(kv, [start, 0], [PREFILL_WINDOW_ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            source = pl.fillpad(source, pad_value=pl.PadValue.zero)
            values = pl.cast(pl.set_validshape(source, PREFILL_WINDOW_ROW_TILE, HEAD_DIM), pl.FP32)
            values = pl.reshape(values, [PREFILL_WINDOW_GROUP_TILE, 32])
            reduce_tmp = pl.create_tile([PREFILL_WINDOW_GROUP_TILE, 32], dtype=pl.FP32)
            maximum = pl.maximum(pl.row_max(pl.abs(values), tmp_tile=reduce_tmp), 1e-4)
            scale = pl.cast(pl.mul(maximum, 1.0 / 448.0), pl.BF16, mode="rint")
            normalized = pl.row_expand_div(values, pl.cast(scale, pl.FP32))
            payload = encode_fp8(normalized, PREFILL_WINDOW_GROUP_TILE, 32, integer_bits=True)
            payload = pl.reshape(payload, [PREFILL_WINDOW_ROW_TILE, HEAD_DIM])
            scale = pl.reshape(scale, [PREFILL_WINDOW_ROW_TILE, SWA_GROUPS])
            if contiguous:
                slot = pl.cast(first, pl.INDEX)
                pl.store(pl.set_validshape(payload, rows, HEAD_DIM), [slot, 0], flat)
                pl.store(pl.set_validshape(scale, rows, SWA_GROUPS), [slot, 0], scale_flat)
            else:
                for lane in pl.unroll(PREFILL_WINDOW_ROW_TILE):
                    if lane < rows:
                        physical = pl.tile.read(ids, [0, lane])
                        if physical >= 0:
                            slot = pl.cast(physical, pl.INDEX)
                            payload_row = pl.tile.slice(payload, [1, HEAD_DIM], [lane, 0])
                            scale_row = pl.tile.slice(scale, [1, SWA_GROUPS], [lane, 0])
                            pl.store(payload_row, [slot, 0], flat)
                            pl.store(scale_row, [slot, 0], scale_flat)
    return publish_tid


@pl.jit.inline
def _publish_window_row(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    flat: pl.Tensor,
    scale_flat: pl.Tensor,
    t: pl.Scalar[pl.INT32],
):
    slot_i64 = pl.read(slots, [t])
    if slot_i64 >= 0:
        slot = pl.cast(slot_i64, pl.INDEX)
        source = pl.load(kv, [t, 0], [1, HEAD_DIM * 2], valid_shape=[1, HEAD_DIM])
        source_padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
        source = pl.set_validshape(source_padded, 1, HEAD_DIM * 2)
        source_fp32 = pl.cast(source, pl.FP32)
        value = pl.reshape(source_fp32, [HEAD_DIM // 16, 32])
        absolute = pl.abs(value)
        reduce_tmp = pl.create_tile([HEAD_DIM // 16, 32], dtype=pl.FP32)
        row_maximum = pl.row_max(absolute, tmp_tile=reduce_tmp)
        amax = pl.maximum(row_maximum, 1e-4)
        raw_scale = pl.mul(amax, 1.0 / 448.0)
        scale = pl.cast(raw_scale, pl.BF16, mode="rint")
        scale_fp32 = pl.cast(scale, pl.FP32)
        normalized = pl.row_expand_div(value, scale_fp32)
        payload = encode_fp8(normalized, HEAD_DIM // 16, 32, integer_bits=False)
        payload_row = pl.reshape(payload, [1, HEAD_DIM * 2])
        flat = pl.store(pl.set_validshape(payload_row, 1, HEAD_DIM), [slot, 0], flat)
        scale_row = pl.reshape(scale, [1, HEAD_DIM // 16])
        scale_flat = pl.store(pl.set_validshape(scale_row, 1, HEAD_DIM // 32), [slot, 0], scale_flat)
    return flat, scale_flat


@pl.jit.inline(auto_scope=False)
def publish_window_row(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    workers = pl.min(pl.min(num_tokens, WINDOW_PUBLISH_WORKER_TILE), pl.system.available_aiv_count())
    step = pl.max(1, workers)
    for worker in pl.spmd(workers, name_hint="pub_win"):
        for t in pl.range(worker, num_tokens, step):
            _publish_window_row(kv, slots, flat, scale_flat, t)
    return cache, scales


@pl.jit.inline(auto_scope=False)
def gather_window_batched(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    selected: pl.Tensor[[T_DYN, SWA_WINDOW, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [rows, HEAD_DIM])
    factors = pl.reshape(scales, [rows, SWA_GROUPS])
    tokens = pl.tensor.dim(selected, 0)
    dst_rows = tokens * SWA_WINDOW
    dst = pl.reshape(selected, [dst_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(pl.min(num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), SWA_WORKER_TILE), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_gather", allow_early_resolve=True) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), workers):
            token = block // (SWA_WINDOW // SWA_SELECTED_GATHER_TILE)
            column = block % (SWA_WINDOW // SWA_SELECTED_GATHER_TILE) * SWA_SELECTED_GATHER_TILE
            ids = pl.tile.load(indices, [token, column], [1, SWA_SELECTED_GATHER_TILE])
            safe_ids = pl.maximum(ids, 0)
            payload = pl.tile.create([SWA_SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.UINT8)
            factor_tile = pl.tile.create([SWA_SELECTED_GATHER_TILE, SWA_GROUPS], dtype=pl.BF16)
            # Check every index before coalescing the at-most-two physical page runs.
            # Arbitrary order, duplicate rows and negative masks keep the fallback.
            first = pl.tile.read(ids, [0, 0])
            first_rows_i32 = pl.cast(pl.min(SWA_SELECTED_GATHER_TILE, 128 - first % 128), pl.INT32)
            first_rows = pl.cast(first_rows_i32, pl.INDEX)
            second = first + first_rows_i32
            if first_rows < SWA_SELECTED_GATHER_TILE:
                second = pl.tile.read(ids, [0, first_rows])
            sequence_first = pl.tile.ci(first, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            sequence_second = pl.tile.ci(second - first_rows_i32, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            columns = pl.tile.ci(0, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            mask = pl.tile.cmps(pl.cast(columns, pl.FP32), pl.cast(first_rows_i32, pl.FP32), cmp_type=2)
            sequence = pl.tile.select(mask, sequence_first, sequence_second)
            delta = pl.abs(pl.cast(pl.sub(ids, sequence), pl.FP32))
            checks = pl.tile.full([8, SWA_SELECTED_GATHER_TILE], dtype=pl.FP32, value=0.0)
            checks = pl.tile.assemble(checks, delta, [0, 0])
            check_tmp = pl.tile.create([8, SWA_SELECTED_GATHER_TILE], dtype=pl.FP32)
            error = pl.tile.read(pl.row_max(checks, tmp_tile=check_tmp), [0, 0])
            first_wide = pl.cast(first, pl.INDEX)
            second_wide = pl.cast(second, pl.INDEX)
            page_runs = ((first >= 0) and (second >= 0) and (first_wide + first_rows <= rows)
                          and (second_wide + SWA_SELECTED_GATHER_TILE - first_rows <= rows) and (error == 0.0))
            if page_runs:
                payload = pl.tile.gather_row(
                    payload, flat, [0, 0], [first_wide, 0], [SWA_SELECTED_GATHER_TILE, HEAD_DIM],
                    valid_shape=[first_rows, HEAD_DIM],
                )
                factor_tile = pl.tile.gather_row(
                    factor_tile, factors, [0, 0], [first_wide, 0], [SWA_SELECTED_GATHER_TILE, SWA_GROUPS],
                    valid_shape=[first_rows, SWA_GROUPS],
                )
                if first_rows < SWA_SELECTED_GATHER_TILE:
                    payload = pl.tile.gather_row(
                        payload, flat, [first_rows, 0], [second_wide, 0], [SWA_SELECTED_GATHER_TILE, HEAD_DIM],
                        valid_shape=[SWA_SELECTED_GATHER_TILE - first_rows, HEAD_DIM],
                    )
                    factor_tile = pl.tile.gather_row(
                        factor_tile, factors, [first_rows, 0], [second_wide, 0], [SWA_SELECTED_GATHER_TILE, SWA_GROUPS],
                        valid_shape=[SWA_SELECTED_GATHER_TILE - first_rows, SWA_GROUPS],
                    )
            else:
                for lane in pl.range(SWA_SELECTED_GATHER_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    payload = pl.tile.gather_row(payload, flat, [lane, 0], [physical, 0], [1, HEAD_DIM])
                    factor_tile = pl.tile.gather_row(factor_tile, factors, [lane, 0], [physical, 0], [1, SWA_GROUPS])
            value = pl.reshape(pl.cast(decode_fp8(payload, SWA_SELECTED_GATHER_TILE, HEAD_DIM), pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 32])
            factor = pl.reshape(pl.cast(factor_tile, pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, factor)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_SELECTED_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [token * SWA_WINDOW + column, 0], dst)
            # Restore zeros for masked rows.
            if not page_runs:
                zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for lane in pl.range(SWA_SELECTED_GATHER_TILE):
                    if pl.tile.read(ids, [0, lane]) < 0:
                        pl.tile.store(zero, [token * SWA_WINDOW + column + lane, 0], dst)
    return gathered


@pl.jit.inline(auto_scope=False)
def gather_window_packed544(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_PACKED_ROW_BYTES], pl.UINT8],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    selected: pl.Tensor[[T_DYN, SWA_WINDOW, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(cache, 0) * 128
    flat_bytes = pl.reshape(cache, [rows, SWA_PACKED_ROW_BYTES])
    tokens = pl.tensor.dim(selected, 0)
    dst_rows = tokens * SWA_WINDOW
    dst = pl.reshape(selected, [dst_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(pl.min(num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), SWA_WORKER_TILE), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_gather_packed", allow_early_resolve=True) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), workers):
            token = block // (SWA_WINDOW // SWA_SELECTED_GATHER_TILE)
            column = block % (SWA_WINDOW // SWA_SELECTED_GATHER_TILE) * SWA_SELECTED_GATHER_TILE
            ids = pl.tile.load(indices, [token, column], [1, SWA_SELECTED_GATHER_TILE])
            safe_ids = pl.maximum(ids, 0)
            payload = pl.tile.create([SWA_SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.UINT8)
            scale_bytes = pl.tile.create([SWA_SELECTED_GATHER_TILE, SWA_SCALE_BYTES], dtype=pl.UINT8)
            # Check every index before coalescing the at-most-two physical page runs.
            # Arbitrary order, duplicate rows and negative masks keep the fallback.
            first = pl.tile.read(ids, [0, 0])
            first_rows_i32 = pl.cast(pl.min(SWA_SELECTED_GATHER_TILE, 128 - first % 128), pl.INT32)
            first_rows = pl.cast(first_rows_i32, pl.INDEX)
            second = first + first_rows_i32
            if first_rows < SWA_SELECTED_GATHER_TILE:
                second = pl.tile.read(ids, [0, first_rows])
            sequence_first = pl.tile.ci(first, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            sequence_second = pl.tile.ci(second - first_rows_i32, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            columns = pl.tile.ci(0, [1, SWA_SELECTED_GATHER_TILE], dtype=pl.INT32)
            mask = pl.tile.cmps(pl.cast(columns, pl.FP32), pl.cast(first_rows_i32, pl.FP32), cmp_type=2)
            sequence = pl.tile.select(mask, sequence_first, sequence_second)
            delta = pl.abs(pl.cast(pl.sub(ids, sequence), pl.FP32))
            checks = pl.tile.full([8, SWA_SELECTED_GATHER_TILE], dtype=pl.FP32, value=0.0)
            checks = pl.tile.assemble(checks, delta, [0, 0])
            check_tmp = pl.tile.create([8, SWA_SELECTED_GATHER_TILE], dtype=pl.FP32)
            error = pl.tile.read(pl.row_max(checks, tmp_tile=check_tmp), [0, 0])
            first_wide = pl.cast(first, pl.INDEX)
            second_wide = pl.cast(second, pl.INDEX)
            page_runs = ((first >= 0) and (second >= 0) and (first_wide + first_rows <= rows)
                          and (second_wide + SWA_SELECTED_GATHER_TILE - first_rows <= rows) and (error == 0.0))
            if page_runs:
                payload = pl.tile.gather_row(
                    payload, flat_bytes, [0, 0], [first_wide, 0], [SWA_SELECTED_GATHER_TILE, HEAD_DIM],
                    valid_shape=[first_rows, HEAD_DIM],
                )
                scale_bytes = pl.tile.gather_row(
                    scale_bytes, flat_bytes, [0, 0], [first_wide, HEAD_DIM],
                    [SWA_SELECTED_GATHER_TILE, SWA_SCALE_BYTES], valid_shape=[first_rows, SWA_SCALE_BYTES],
                )
                if first_rows < SWA_SELECTED_GATHER_TILE:
                    payload = pl.tile.gather_row(
                        payload, flat_bytes, [first_rows, 0], [second_wide, 0],
                        [SWA_SELECTED_GATHER_TILE, HEAD_DIM],
                        valid_shape=[SWA_SELECTED_GATHER_TILE - first_rows, HEAD_DIM],
                    )
                    scale_bytes = pl.tile.gather_row(
                        scale_bytes, flat_bytes, [first_rows, 0], [second_wide, HEAD_DIM],
                        [SWA_SELECTED_GATHER_TILE, SWA_SCALE_BYTES],
                        valid_shape=[SWA_SELECTED_GATHER_TILE - first_rows, SWA_SCALE_BYTES],
                    )
            else:
                for lane in pl.range(SWA_SELECTED_GATHER_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    payload = pl.tile.gather_row(payload, flat_bytes, [lane, 0], [physical, 0], [1, HEAD_DIM])
                    scale_bytes = pl.tile.gather_row(
                        scale_bytes, flat_bytes, [lane, 0], [physical, HEAD_DIM], [1, SWA_SCALE_BYTES],
                    )
            factor_bits = pl.tile.reinterpret_view(
                scale_bytes, pl.INT16, shape=[SWA_SELECTED_GATHER_TILE, SWA_GROUPS],
            )
            factors = pl.tile.reinterpret_view(factor_bits, pl.BF16)
            value = pl.reshape(pl.cast(decode_fp8(payload, SWA_SELECTED_GATHER_TILE, HEAD_DIM), pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 32])
            factor = pl.reshape(pl.cast(factors, pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, factor)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_SELECTED_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [token * SWA_WINDOW + column, 0], dst)
            # Restore zeros for masked rows.
            if not page_runs:
                zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for lane in pl.range(SWA_SELECTED_GATHER_TILE):
                    if pl.tile.read(ids, [0, lane]) < 0:
                        pl.tile.store(zero, [token * SWA_WINDOW + column + lane, 0], dst)
    return gathered


@pl.jit.inline(auto_scope=False)
def _swa_attention_tile(
    query: pl.Tensor,
    keys: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    query_offset: pl.Scalar[pl.INDEX],
    key_offset: pl.Scalar[pl.INDEX],
    index_row: pl.Scalar[pl.INDEX],
    head_offset: pl.Scalar[pl.INDEX],
    heads: pl.constexpr,
    key_rows: pl.constexpr,
    key_tile: pl.constexpr,
):
    """Accumulate SWA dot products in order and normalize the 128 causal key weights."""
    sinks = pl.load(sink, [0, head_offset], [1, heads])
    maximum = pl.mul(sinks, 1.0)
    denominator = pl.tile.full([1, heads], dtype=pl.FP32, value=1.0)
    numerator = pl.tile.full([heads, HEAD_DIM], dtype=pl.FP32, value=0.0)
    reduction = pl.create_tile([heads, key_tile], dtype=pl.FP32)
    for part in pl.range(key_rows // key_tile):
        column = part * key_tile
        first_key = key_offset + column
        product = bf16_dot_sequential(query, keys, query_offset, first_key, heads, key_tile, HEAD_DIM)
        scores = pl.mul(product, SCALE)
        ids = pl.load(indices, [index_row, column], [1, key_tile])
        values = pl.cast(ids, pl.FP32)
        valid = pl.add(values, 1.0)
        valid = pl.minimum(pl.maximum(valid, 0.0), 1.0)
        penalty = pl.mul(pl.sub(valid, 1.0), 1e30)
        scores = pl.col_expand_add(scores, penalty)
        score_maximum = pl.row_max(scores, tmp_tile=reduction)
        score_maximum = pl.reshape(score_maximum, [1, heads])
        next_maximum = pl.maximum(maximum, score_maximum)
        correction = centered_exp_line(pl.sub(maximum, next_maximum))
        maximum_column = pl.reshape(next_maximum, [heads, 1])
        centered = pl.row_expand_sub(scores, maximum_column)
        exp_values = centered_exp_scores(centered)
        probabilities = pl.col_expand_mul(exp_values, valid)
        part0 = pl.tile.full([heads, 8], dtype=pl.FP32, value=0.0)
        part1 = pl.tile.full([heads, 8], dtype=pl.FP32, value=0.0)
        part2 = pl.tile.full([heads, 8], dtype=pl.FP32, value=0.0)
        part3 = pl.tile.full([heads, 8], dtype=pl.FP32, value=0.0)
        for group in pl.unroll(key_tile // 32):
            values0 = pl.tile.slice(probabilities, [heads, 8], [0, group * 32 + 0])
            part0 = pl.add(part0, values0)
            values1 = pl.tile.slice(probabilities, [heads, 8], [0, group * 32 + 8])
            part1 = pl.add(part1, values1)
            values2 = pl.tile.slice(probabilities, [heads, 8], [0, group * 32 + 16])
            part2 = pl.add(part2, values2)
            values3 = pl.tile.slice(probabilities, [heads, 8], [0, group * 32 + 24])
            part3 = pl.add(part3, values3)
        partial = pl.add(pl.add(pl.add(part0, part1), part2), part3)
        by_lane = pl.tile.transpose(partial, 0, 1)
        probability_sum = pl.tile.full([1, heads], dtype=pl.FP32, value=0.0)
        for lane in pl.unroll(8):
            probability_sum = pl.add(probability_sum, pl.tile.slice(by_lane, [1, heads], [lane, 0]))
        denominator = pl.add(pl.mul(denominator, correction), probability_sum)
        rounded = pl.cast(probabilities, pl.BF16, mode="rint")
        correction_column = pl.reshape(correction, [heads, 1])
        for half in pl.unroll(2):
            feature = half * 256
            weighted = pl.tile.full([heads, 256], dtype=pl.FP32, value=0.0)
            for first in pl.range(0, key_tile, 16):
                left = pl.tile.slice(rounded, [heads, 16], [0, first])
                right = pl.load(keys, [first_key + first, feature], [16, 256])
                weighted = bf16_ordered_accumulate(weighted, left, right, heads, 256, 16)
            previous = pl.tile.slice(numerator, [heads, 256], [0, feature])
            previous = pl.row_expand_mul(previous, correction_column)
            accumulated = pl.add(previous, weighted)
            numerator = pl.tile.assemble(numerator, accumulated, [0, feature])
        maximum = next_maximum
    for group in pl.unroll(2):
        denominator_row = pl.tile.slice(denominator, [1, 8], [0, group * 8])
        denominator_column = pl.reshape(denominator_row, [8, 1])
        for stripe in pl.unroll(4):
            sub = pl.tile.slice(numerator, [8, 128], [group * 8, stripe * 128])
            normalized = normalized_quotient(sub, pl.row_expand(sub, denominator_column))
            result = pl.cast(normalized, pl.BF16, mode="rint")
            pl.store(result, [query_offset + group * 8, stripe * 128], output)


@pl.jit.inline(auto_scope=False)
def _attention_tile(
    query: pl.Tensor,
    keys: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    query_offset: pl.Scalar[pl.INDEX],
    key_offset: pl.Scalar[pl.INDEX],
    index_row: pl.Scalar[pl.INDEX],
    head_offset: pl.Scalar[pl.INDEX],
    heads: pl.constexpr,
    key_rows: pl.constexpr,
    key_tile: pl.constexpr,
    window_only: pl.constexpr,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    inverse_rope: pl.constexpr,
):
    """Evaluate BF16 QK and PV products with FP32 online softmax."""
    sinks = pl.load(sink, [0, head_offset], [1, heads])
    if window_only:
        maximum = pl.mul(sinks, 1.0)
        denominator = pl.tile.full([1, heads], dtype=pl.FP32, value=1.0)
    else:
        maximum = pl.tile.full([1, heads], dtype=pl.FP32, value=-1e30)
        denominator = pl.tile.full([1, heads], dtype=pl.FP32, value=0.0)
    numerator = pl.tile.full([heads, HEAD_DIM], dtype=pl.FP32, value=0.0)
    reduction = pl.create_tile([heads, key_tile], dtype=pl.FP32)
    q_low = pl.load(query, [query_offset, 0], [heads, 256], target_memory=pl.MemorySpace.Mat)
    q_high = pl.load(query, [query_offset, 256], [heads, 256], target_memory=pl.MemorySpace.Mat)
    for part in pl.range(key_rows // key_tile):
        column = part * key_tile
        first_key = key_offset + column
        k_low = pl.load(keys, [first_key, 0], [key_tile, 256], target_memory=pl.MemorySpace.Mat)
        k_high = pl.load(keys, [first_key, 256], [key_tile, 256], target_memory=pl.MemorySpace.Mat)
        qk = pl.create_tile([heads, key_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
        qk = pl.matmul_acc(qk, q_low, pl.tile.transpose_view(k_low), init_cond=True)
        qk = pl.matmul_acc(qk, q_high, pl.tile.transpose_view(k_high))
        product = pl.tile.move(qk, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
        scores = pl.mul(product, SCALE)
        ids = pl.load(indices, [index_row, column], [1, key_tile])
        values = pl.cast(ids, pl.FP32)
        valid = pl.add(values, 1.0)
        valid = pl.minimum(pl.maximum(valid, 0.0), 1.0)
        penalty = pl.mul(pl.sub(valid, 1.0), 1e30)
        scores = pl.col_expand_add(scores, penalty)
        score_maximum = pl.row_max(scores, tmp_tile=reduction)
        score_maximum = pl.reshape(score_maximum, [1, heads])
        next_maximum = pl.maximum(maximum, score_maximum)
        correction = pl.exp(pl.sub(maximum, next_maximum))
        maximum_column = pl.reshape(next_maximum, [heads, 1])
        centered = pl.row_expand_sub(scores, maximum_column)
        probabilities = pl.col_expand_mul(pl.exp(centered), valid)
        probability_sum = pl.row_sum(probabilities, tmp_tile=reduction)
        probability_sum = pl.reshape(probability_sum, [1, heads])
        denominator = pl.add(pl.mul(denominator, correction), probability_sum)
        rounded = pl.cast(probabilities, pl.BF16, mode="rint")
        weights = pl.tile.move(rounded, pl.MemorySpace.Mat)
        correction_column = pl.reshape(correction, [heads, 1])
        for half in pl.unroll(2):
            feature = half * 256
            if half == 0:
                weighted_acc = pl.tile.matmul(weights, k_low)
            else:
                weighted_acc = pl.tile.matmul(weights, k_high)
            weighted = pl.tile.move(weighted_acc, pl.MemorySpace.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
            previous = pl.tile.slice(numerator, [heads, 256], [0, feature])
            previous = pl.row_expand_mul(previous, correction_column)
            accumulated = pl.add(previous, weighted)
            numerator = pl.tile.assemble(numerator, accumulated, [0, feature])
        maximum = next_maximum
    if not window_only:
        final_maximum = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_maximum))
        sink_weight = pl.exp(pl.sub(sinks, final_maximum))
        denominator = pl.add(pl.mul(denominator, correction), sink_weight)
        correction_column = pl.reshape(correction, [heads, 1])
        numerator = pl.row_expand_mul(numerator, correction_column)
    denominator_column = pl.reshape(denominator, [heads, 1])
    normalized = pl.row_expand_div(numerator, denominator_column)
    result = pl.cast(normalized, pl.BF16, mode="rint")
    if inverse_rope:
        # Preserve the BF16 attention boundary before rotating the last 64 features.
        lanes = pl.cast(pl.tile.arange(0, [1, heads * 64], dtype=pl.INT32), pl.FP32)
        pairs = pl.cast(pl.mul(lanes, 0.5), pl.INT32, mode="trunc")
        parity = pl.sub(lanes, pl.mul(pl.cast(pairs, pl.FP32), 2.0))
        swap = pl.cast(pl.sub(pl.add(lanes, 1.0), pl.mul(parity, 2.0)), pl.INT32)
        pair_index = pl.tile.slice(pairs, [1, 64], [0, 0])
        head_parity = pl.tile.slice(parity, [1, 64], [0, 0])
        sign = pl.mul(pl.sub(pl.mul(head_parity, 2.0), 1.0), -1.0)
        tail = pl.cast(pl.tile.slice(result, [heads, 64], [0, 448]), pl.FP32)
        swap_tmp = pl.create_tile([1, heads * 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        swapped = pl.tile.gather(pl.reshape(tail, [1, heads * 64]), swap, swap_tmp)
        coeff_tmp = pl.create_tile([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
        cos = pl.tile.gather(pl.load(rope_cos, [0, 0], [1, 32]), pair_index, coeff_tmp)
        sin = pl.mul(pl.tile.gather(pl.load(rope_sin, [0, 0], [1, 32]), pair_index, coeff_tmp), sign)
        direct = pl.col_expand_mul(tail, cos)
        cross = pl.col_expand_mul(pl.reshape(swapped, [heads, 64]), sin)
        rotated = pl.cast(pl.add(direct, cross), pl.BF16, mode="rint")
        prefix = pl.tile.slice(result, [heads, 448], [0, 0])
        pl.store(prefix, [query_offset, 0], output)
        pl.store(rotated, [query_offset, 448], output)
    else:
        pl.store(result, [query_offset, 0], output)


@pl.jit.inline(auto_scope=False)
def attention_tile(
    query: pl.Tensor,
    keys: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    query_offset: pl.Scalar[pl.INDEX],
    key_offset: pl.Scalar[pl.INDEX],
    index_row: pl.Scalar[pl.INDEX],
    head_offset: pl.Scalar[pl.INDEX],
    heads: pl.constexpr,
    key_rows: pl.constexpr,
    key_tile: pl.constexpr,
    window_only: pl.constexpr,
):
    """Evaluate unrotated BF16 attention."""
    _attention_tile(
        query, keys, indices, sink, output, query_offset, key_offset, index_row,
        head_offset, heads, key_rows, key_tile, window_only, sink, sink, False,
    )


@pl.jit.inline(auto_scope=False)
def attend_window_tiled(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    keys: pl.Tensor[[SWA_KEY_ROWS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    key_offsets: pl.Tensor,
    use_offsets: pl.constexpr,
):
    """Attend SWA keys in logical order with FP32 score accumulation."""
    tokens = pl.tensor.dim(query, 0)
    qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
    oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    head_blocks = LOCAL_H // SWA_HEAD_TILE
    tasks = num_tokens * head_blocks
    workers = pl.max(1, pl.min(pl.min(tasks, SWA_WORKER_TILE), pl.system.available_cluster_count()))
    with pl.spmd(
        workers, name_hint="swa_attn", deps=[ready], optimizations=[pl.cross_core_slot(slot_num=SWA_CROSS_CORE_SLOT_TILE)],
    ) as attention_done:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, tasks, workers):
            token = block // head_blocks
            head = block % head_blocks * SWA_HEAD_TILE
            q0 = token * LOCAL_H + head
            if use_offsets:
                k0 = pl.cast(pl.read(key_offsets, [token, 0]), pl.INDEX)
            else:
                k0 = token * SWA_WINDOW
            attention_tile(
                qflat, keys, indices, sink_row, oflat,
                q0, k0, token, head, SWA_HEAD_TILE, SWA_WINDOW, SWA_WINDOW, True,
            )
    return attention_done


@pl.jit.inline(auto_scope=False)
def decode_shared_cache(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    decoded_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
):
    rows = pl.tensor.dim(cache, 0) * 128
    payload_flat = pl.reshape(cache, [rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [rows, SWA_GROUPS])
    decoded_flat = pl.reshape(decoded_cache, [rows, HEAD_DIM])
    # Physical pages are a multiple of 128 rows; GATHER_TILE=32 has no tail.
    # Decode each physical row once with contiguous transfers.
    workers = pl.max(1, pl.min(pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_decode_cache") as ready:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            value_full = pl.tile.full([32, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.tile.store(value_full, [0, 0], zero_row)
        for block in pl.range(worker, rows // SWA_GATHER_TILE, workers):
            row = block * SWA_GATHER_TILE
            payload = pl.tile.load(payload_flat, [row, 0], [SWA_GATHER_TILE, HEAD_DIM])
            factors = pl.tile.load(scale_flat, [row, 0], [SWA_GATHER_TILE, SWA_GROUPS])
            value = pl.reshape(pl.cast(decode_fp8(payload, SWA_GATHER_TILE, HEAD_DIM), pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
            multiplier = pl.reshape(pl.cast(factors, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, multiplier)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_flat)
    return ready


@pl.jit.inline(auto_scope=False)
def decode_shared_cache_packed(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_PACKED_ROW_BYTES], pl.UINT8],
    decoded_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
):
    rows = pl.tensor.dim(cache, 0) * 128
    flat_bytes = pl.reshape(cache, [rows, SWA_PACKED_ROW_BYTES])
    decoded_flat = pl.reshape(decoded_cache, [rows, HEAD_DIM])
    # Physical pages are a multiple of 128 rows; GATHER_TILE=32 has no tail.
    # Decode each physical row once with contiguous transfers.
    workers = pl.max(1, pl.min(pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_decode_packed") as ready:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            value_full = pl.tile.full([32, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.tile.store(value_full, [0, 0], zero_row)
        for block in pl.range(worker, rows // SWA_GATHER_TILE, workers):
            row = block * SWA_GATHER_TILE
            payload_bytes = pl.tile.load(flat_bytes, [row, 0], [SWA_GATHER_TILE, HEAD_DIM])
            factor_bytes = pl.tile.load(flat_bytes, [row, HEAD_DIM], [SWA_GATHER_TILE, SWA_SCALE_BYTES])
            payload_reinterpret_view = pl.tile.reinterpret_view(payload_bytes, pl.INT8)
            payload = pl.tile.reinterpret_view(
                payload_reinterpret_view, pl.UINT8,
                shape=[SWA_GATHER_TILE, HEAD_DIM],
            )
            factors_reinterpret_view = pl.tile.reinterpret_view(
                factor_bytes, pl.INT16,
                shape=[SWA_GATHER_TILE, SWA_GROUPS],
            )
            factors = pl.tile.reinterpret_view(factors_reinterpret_view, pl.BF16)
            value = pl.reshape(pl.cast(decode_fp8(payload, SWA_GATHER_TILE, HEAD_DIM), pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
            multiplier = pl.reshape(pl.cast(factors, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, multiplier)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_flat)
    return ready


@pl.jit.inline(auto_scope=False)
def _classify_swa_run(
    ids: pl.Tile[[1, SWA_RUN_WIDTH], pl.INT32],
    cache_rows: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    classify_empty: pl.constexpr,
):
    """Return the physical start, -1 for scattered keys, or -2 for an empty run."""
    ids_fixed = pl.reshape(ids, [1, width])
    first = pl.tile.read(ids_fixed, [0, 0])
    start = pl.cast(first, pl.INDEX)
    run = pl.cast(-1, pl.INT32)
    reduction = pl.tile.create([8, width], dtype=pl.FP32)
    if first >= 0 and start + width <= cache_rows:
        expected_ids = pl.tile.ci(first, [1, width], dtype=pl.INT32)
        # Subtract in INT32 before rounding to FP32.
        difference = pl.sub(ids_fixed, expected_ids)
        delta = pl.abs(pl.cast(difference, pl.FP32))
        delta_rows = pl.tile.full([8, width], dtype=pl.FP32, value=0.0)
        delta_rows = pl.tile.assemble(delta_rows, delta, [0, 0])
        error = pl.tile.read(pl.row_max(delta_rows, tmp_tile=reduction), [0, 0])
        if error == 0.0:
            run = first
    if classify_empty:
        if first < 0:
            index_rows = pl.tile.full([8, width], dtype=pl.FP32, value=0.0)
            index_values = pl.cast(ids_fixed, pl.FP32)
            index_rows = pl.tile.assemble(index_rows, index_values, [0, 0])
            largest = pl.tile.read(pl.row_max(index_rows, tmp_tile=reduction), [0, 0])
            if largest < 0.0:
                run = pl.cast(-2, pl.INT32)
    return run


@pl.jit.inline(auto_scope=False)
def analyze_index_runs(
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    descriptors: pl.Tensor[[T_DYN, SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_rows: pl.Scalar[pl.INDEX],
):
    """Describe exact physical runs on AIV."""
    tokens = pl.tensor.dim(indices, 0)
    descriptor_rows = tokens * SWA_RUN_PARTS
    descriptor_flat = pl.reshape(descriptors, [descriptor_rows, SWA_RUN_DESCRIPTOR_WIDTH])
    workers = pl.max(1, pl.min(pl.min(num_tokens * SWA_RUN_PARTS, SWA_WORKER_TILE * 2), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_plan_runs") as ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * SWA_RUN_PARTS, workers):
            token = block // SWA_RUN_PARTS
            part = block % SWA_RUN_PARTS
            ids = pl.tile.load(indices, [token, part * SWA_RUN_TILE], [1, SWA_RUN_TILE])
            run = _classify_swa_run(ids, cache_rows, SWA_RUN_TILE, True)
            # Only column zero stores the run descriptor.
            descriptor = pl.tile.ci(run, [1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
            pl.tile.store(descriptor, [block, 0], descriptor_flat)
    return ready


@pl.jit.inline(auto_scope=False)
def decode_shared_cache_after_publish(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    decoded_flat: pl.Tensor[[SWA_KEY_ROWS, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
    publish_ready: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(cache, 0) * 128
    payload_flat = pl.reshape(cache, [rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [rows, SWA_GROUPS])
    # Physical pages are a multiple of 128 rows; GATHER_TILE=32 has no tail.
    # Decode each physical row once with contiguous transfers.
    workers = pl.max(1, pl.min(pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_decode_cache", deps=[publish_ready]) as ready:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            value_full = pl.tile.full([32, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.tile.store(value_full, [0, 0], zero_row)
        for block in pl.range(worker, rows // SWA_GATHER_TILE, workers):
            row = block * SWA_GATHER_TILE
            payload = pl.tile.load(payload_flat, [row, 0], [SWA_GATHER_TILE, HEAD_DIM])
            factors = pl.tile.load(scale_flat, [row, 0], [SWA_GATHER_TILE, SWA_GROUPS])
            value = pl.reshape(pl.cast(decode_fp8(payload, SWA_GATHER_TILE, HEAD_DIM), pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
            multiplier = pl.reshape(pl.cast(factors, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, multiplier)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_flat)
    return ready


@pl.jit.inline(auto_scope=False)
def _load_swa_run(
    decoded: pl.Tensor,
    zero_row: pl.Tensor,
    indices: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    run: pl.Scalar[pl.INT32],
):
    """Load one zero, contiguous, or scattered run of SWA keys."""
    values = pl.create_tile([SWA_RUN_TILE, HEAD_DIM], dtype=pl.BF16)
    if run == -2:
        values = pl.load(zero_row, [0, 0], [SWA_RUN_TILE, HEAD_DIM])
    elif run >= 0:
        physical = pl.cast(run, pl.INDEX)
        values = pl.load(decoded, [physical, 0], [SWA_RUN_TILE, HEAD_DIM])
    else:
        for lane in pl.range(SWA_RUN_TILE):
            row = pl.read(indices, [token, column + lane])
            if row >= 0:
                physical = pl.cast(row, pl.INDEX)
                values = pl.tile.gather_row(values, decoded, [lane, 0], [physical, 0], [1, HEAD_DIM])
            else:
                values = pl.tile.gather_row(values, zero_row, [lane, 0], [0, 0], [1, HEAD_DIM])
    return values


@pl.jit.inline(auto_scope=False)
def attend_shared_cache(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    decoded_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
    descriptors: pl.Tensor[[T_DYN, SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH], pl.INT32],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    index_ready: pl.Scalar[pl.TASK_ID],
):
    """Stage physical SWA runs and attend their logical key order."""
    tokens = pl.tensor.dim(query, 0)
    cache_rows = pl.tensor.dim(decoded_cache, 0) * 128
    decoded = pl.reshape(decoded_cache, [cache_rows, HEAD_DIM])
    descriptor_flat = pl.reshape(descriptors, [tokens * SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH])
    selected = pl.create_tensor([tokens, SWA_WINDOW, HEAD_DIM], dtype=pl.BF16)
    selected_flat = pl.reshape(selected, [tokens * SWA_WINDOW, HEAD_DIM])
    tasks = num_tokens * SWA_RUN_PARTS
    workers = pl.max(1, pl.min(pl.min(tasks, SWA_WORKER_TILE * 2), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_stage_runs", deps=[ready, index_ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, tasks, workers):
            token = block // SWA_RUN_PARTS
            column = block % SWA_RUN_PARTS * SWA_RUN_TILE
            run = pl.read(descriptor_flat, [block, 0])
            values = _load_swa_run(decoded, zero_row, indices, token, column, run)
            destination = token * SWA_WINDOW + column
            pl.store(values, [destination, 0], selected_flat)
    return attend_window_tiled(query, selected_flat, indices, sink, output, num_tokens, gathered, indices, False)


@pl.jit.inline(auto_scope=False)
def attend_shared_cache_direct(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    keys: pl.Tensor[[SWA_KEY_ROWS, HEAD_DIM], pl.BF16],
    cache_rows: pl.Scalar[pl.INDEX],
    zero_row: pl.Tensor[[SWA_RUN_TILE, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Select contiguous windows and stage fragmented windows after the cache."""
    tokens = pl.tensor.dim(query, 0)
    key_offsets = pl.create_tensor([tokens, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
    workers = pl.max(1, pl.min(num_tokens, pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="swa_select_window_offsets", deps=[ready]) as selected:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            window_ids = pl.load(indices, [token, 0], [1, SWA_WINDOW])
            offset = _classify_swa_run(window_ids, cache_rows, SWA_WINDOW, False)
            if offset < 0:
                fallback_row = cache_rows + token * SWA_WINDOW
                for column in pl.range(0, SWA_WINDOW, SWA_RUN_TILE):
                    # A2/A3 tile extraction does not support INT32 indices.
                    run_ids = pl.load(indices, [token, column], [1, SWA_RUN_TILE])
                    run = _classify_swa_run(run_ids, cache_rows, SWA_RUN_TILE, True)
                    values = _load_swa_run(keys, zero_row, indices, token, column, run)
                    pl.store(values, [fallback_row + column, 0], keys)
                offset = pl.cast(fallback_row, pl.INT32)
            offsets = pl.tile.ci(offset, [1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
            pl.store(offsets, [token, 0], key_offsets)
    return attend_window_tiled(query, keys, indices, sink, output, num_tokens, selected, key_offsets, True)


@pl.jit.inline(auto_scope=False)
def attend_shared_cache_plain(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    decoded_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Analyze physical SWA runs before staging the selected rows."""
    tokens = pl.tensor.dim(query, 0)
    cache_rows = pl.tensor.dim(decoded_cache, 0) * 128
    descriptors = pl.create_tensor([tokens, SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
    analyzed = analyze_index_runs(indices, descriptors, num_tokens, cache_rows)
    return attend_shared_cache(query, decoded_cache, zero_row, descriptors, indices, sink, output, num_tokens, ready, analyzed)


@pl.jit.inline(auto_scope=False)
def swa_attention_partial(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Return conservative attention completion for subsequent cache-update ordering."""
    # Conservatively keep later cache updates after full mixed attention while
    # investigating PyPTO #2829. This boundary is not a confirmed causal fix.
    tokens = pl.tensor.dim(query, 0)
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    # Small calls and cache-heavy calls avoid decoding unused physical rows.
    # Bounded two-stage pools still submit one task for an empty active prefix.
    use_shared = pl.cast(0, pl.INT32)
    if num_tokens >= 32:
        if cache_rows <= num_tokens * SWA_WINDOW:
            use_shared = pl.cast(1, pl.INT32)
    if use_shared == 0:
        selected = pl.create_tensor([tokens, SWA_WINDOW, HEAD_DIM], dtype=pl.BF16)
        gathered = gather_window_batched(cache, scales, indices, selected, num_tokens)
        selected_flat = pl.reshape(selected, [tokens * SWA_WINDOW, HEAD_DIM])
        attention_done = attend_window_tiled(query, selected_flat, indices, sink, output, num_tokens, gathered, indices, False)
        cache_consumed = attention_done
    else:
        decoded_cache = pl.create_tensor([blocks, 128, 1, HEAD_DIM], dtype=pl.BF16)
        zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
        decoded = decode_shared_cache(cache, scales, decoded_cache, zero_row)
        if num_tokens < 128:
            attention_done = attend_shared_cache_plain(
                query, decoded_cache, zero_row, indices, sink, output, num_tokens, decoded,
            )
        else:
            descriptors = pl.create_tensor([tokens, SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
            analyzed = analyze_index_runs(indices, descriptors, num_tokens, cache_rows)
            attention_done = attend_shared_cache(
                query, decoded_cache, zero_row, descriptors, indices, sink, output, num_tokens, decoded, analyzed,
            )
        cache_consumed = attention_done
    return cache_consumed


@pl.jit.inline(auto_scope=False)
def swa_attention_partial_b1_rotated(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
):
    """Gather the live SWA row, then write the inverse-rotated attention output."""
    tokens = pl.tensor.dim(query, 0)
    selected = pl.create_tensor([tokens, SWA_WINDOW, HEAD_DIM], dtype=pl.BF16)
    gathered = gather_window_batched(cache, scales, indices, selected, 1)
    qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
    kflat = pl.reshape(selected, [tokens * SWA_WINDOW, HEAD_DIM])
    oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    head_tiles = LOCAL_H // SWA_HEAD_TILE
    head_workers = pl.max(1, pl.min(head_tiles, pl.system.available_cluster_count()))
    with pl.spmd(head_workers, name_hint="swa_attn", deps=[gathered],
                 optimizations=[pl.cross_core_slot(slot_num=1)]) as attention_done:
        head_worker = pl.tile.get_block_idx()
        for head_tile in pl.range(head_worker, head_tiles, head_workers):
            head = head_tile * SWA_HEAD_TILE
            _attention_tile(
                qflat, kflat, indices, sink_row, oflat, head, 0, 0, head,
                SWA_HEAD_TILE, SWA_WINDOW, SWA_WINDOW, True, rope_cos, rope_sin, True,
            )
    return attention_done


@pl.jit.inline(auto_scope=False)
def swa_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Attend the active query prefix; leave caller-owned inactive rows untouched."""
    swa_attention_partial(query, cache, scales, indices, sink, output, num_tokens)
    return output


@pl.jit.inline(auto_scope=False)
def _swa_attention_packed_partial(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_PACKED_ROW_BYTES], pl.UINT8],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Return conservative attention completion for subsequent cache-update ordering."""
    # Use the same conservative boundary as the split cache path while
    # investigating PyPTO #2829; no causal fix is established here.
    tokens = pl.tensor.dim(query, 0)
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    # Small calls and cache-heavy calls avoid decoding unused physical rows.
    # Bounded two-stage pools still submit one task for an empty active prefix.
    use_shared = pl.cast(0, pl.INT32)
    if num_tokens >= 32:
        if cache_rows <= num_tokens * SWA_WINDOW:
            use_shared = pl.cast(1, pl.INT32)
    if use_shared == 0:
        selected = pl.create_tensor([tokens, SWA_WINDOW, HEAD_DIM], dtype=pl.BF16)
        gathered = gather_window_packed544(cache, indices, selected, num_tokens)
        selected_flat = pl.reshape(selected, [tokens * SWA_WINDOW, HEAD_DIM])
        attention_done = attend_window_tiled(query, selected_flat, indices, sink, output, num_tokens, gathered, indices, False)
        cache_consumed = attention_done
    else:
        decoded_cache = pl.create_tensor([blocks, 128, 1, HEAD_DIM], dtype=pl.BF16)
        zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
        decoded = decode_shared_cache_packed(cache, decoded_cache, zero_row)
        if num_tokens < 128:
            attention_done = attend_shared_cache_plain(
                query, decoded_cache, zero_row, indices, sink, output, num_tokens, decoded,
            )
        else:
            descriptors = pl.create_tensor([tokens, SWA_RUN_PARTS, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
            analyzed = analyze_index_runs(indices, descriptors, num_tokens, cache_rows)
            attention_done = attend_shared_cache(
                query, decoded_cache, zero_row, descriptors, indices, sink, output, num_tokens, decoded, analyzed,
            )
        cache_consumed = attention_done
    return cache_consumed


@pl.jit.inline(auto_scope=False)
def swa_attention_packed(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_PACKED_ROW_BYTES], pl.UINT8],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Attend the active query prefix; leave caller-owned inactive rows untouched."""
    _swa_attention_packed_partial(query, cache, indices, sink, output, num_tokens)
    return output


@pl.jit(auto_scope=False)
def l2_sparse_attention_swa(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output = swa_attention(query, cache, scales, indices, sink, output, num_tokens)
    return output


@pl.jit(auto_scope=False)
def l2_sparse_attention_swa_direct(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    query.bind_dynamic(0, T_DYN)
    cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    tokens = pl.tensor.dim(query, 0)
    cache_rows = pl.tensor.dim(cache, 0) * SWA_WINDOW
    key_rows = cache_rows + tokens * SWA_WINDOW
    keys = pl.create_tensor([key_rows, HEAD_DIM], dtype=pl.BF16)
    zero_row = pl.create_tensor([SWA_RUN_TILE, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_fixture_ready") as cache_ready:
        _anchor = pl.read(indices, [0, 0])
    decoded = decode_shared_cache_after_publish(cache, scales, keys, zero_row, cache_ready)
    attend_shared_cache_direct(query, keys, cache_rows, zero_row, indices, sink, output, num_tokens, decoded)
    return output


@pl.jit(auto_scope=False)
def l2_sparse_attention_swa_packed(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_PACKED_ROW_BYTES], pl.UINT8],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output = swa_attention_packed(query, cache, indices, sink, output, num_tokens)
    return output


SWA_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a",  "wo_b",  "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)


def swa_official_reference(tensors, *, shared_cache=False, key_tile=128):
    """Evaluate FP8 projections, tiled BF16 attention weights, and FP32 output."""
    from .quant import unpack_fp8_nz
    t = tensors
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups = t["wo_a"].shape[0]
    group_in = t["wo_a"].shape[-2]

    qa_weight = unpack_fp8_nz(t["wq_a"])
    qb_weight = unpack_fp8_nz(t["wq_b"])
    kv_weight = unpack_fp8_nz(t["wkv"])
    projected = golden_dynamic_int8(t["x"], qa_weight, t["wq_a_scale"])
    qr = golden_rms_norm(projected, t["q_norm_weight"])
    q = golden_dynamic_int8(qr, qb_weight, t["wq_b_scale"])
    q = q.unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    projected_kv = golden_dynamic_int8(t["x"], kv_weight, t["wkv_scale"])
    kv = golden_rms_norm(projected_kv, t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])
    payload, scales = quantize_mxfp8_cache(kv)
    cache = t["window_cache"].clone()
    cache_scale = t["window_cache_scale"].clone()
    slots = t["window_slots"].long()
    valid_slots = slots >= 0
    cache.view(torch.uint8).reshape(-1, head_dim)[slots[valid_slots]] = payload.view(torch.uint8)[valid_slots]
    cache_scale.reshape(-1, head_dim // 32)[slots[valid_slots]] = scales[valid_slots]
    values = cache.float() * cache_scale.float().repeat_interleave(32, -1)
    idx = t["window_indices"].long()
    selected = values.reshape(-1, head_dim)[idx.clamp_min(0)].to(torch.bfloat16)
    selected = selected.masked_fill((idx < 0)[..., None], 0)
    maximum = torch.full(q.shape[:2], -torch.inf if shared_cache else -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q, dtype=torch.float32)
    for start in range(0, idx.shape[-1], key_tile):
        keys = selected[:, start:start + key_tile].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), keys) * head_dim ** -0.5
        valid = idx[:, start:start + key_tile] >= 0
        # Match the device's finite mask so an empty block keeps zero statistics.
        logits = logits.masked_fill(~valid[:, None], -1e30)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        if shared_cache:
            new_maximum = torch.maximum(new_maximum, t["attn_sink"][None])
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp().masked_fill(~valid[:, None], 0)
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), keys)
        maximum = new_maximum
    final_max = torch.maximum(maximum, t["attn_sink"][None])
    correction = (maximum - final_max).exp()
    denominator = denominator * correction + (t["attn_sink"][None] - final_max).exp()
    denominator = denominator.clamp_min(1.0)
    if shared_cache:
        multiplier = denominator.reciprocal()
    else:
        multiplier = correction / denominator
    attended = (numerator * multiplier[..., None]).to(torch.bfloat16)
    attended = golden_rope_tail(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(-1, groups, group_in)
    latent = golden_grouped_output(
                 grouped, t["wo_a"],
             )
    output = golden_dynamic_int8(latent.flatten(1), unpack_fp8_nz(t["wo_b"]), t["wo_b_scale"]).float()
    return output, cache, cache_scale


def make_swa_inputs(batch=MAX_BATCH_PER_DP, seed=17, position=127):
    """Distinct request pages, ragged visibility, nontrivial scales and rotary phases."""
    from .quant import pack_fp8_nz
    gen = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=gen)
    def weight(k, n):
        payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
        return payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)
    qa, qas = weight(D, Q_LORA)
    qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
    kv, kvs = weight(D, HEAD_DIM)
    ob, obs = weight(LOCAL_O_WIDTH, D)
    qa = pack_fp8_nz(qa)
    qb = pack_fp8_nz(qb)
    kv = pack_fp8_nz(kv)
    ob = pack_fp8_nz(ob)
    positions = torch.full((batch,), position, dtype=torch.int64)
    if batch > 1:
        positions[:min(batch, 6)] = torch.tensor([0, 1, 63, 127, 128, 1048575])[:min(batch, 6)]
    freq = 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
    angles = positions.float()[:, None] * freq
    physical = torch.randperm(batch + 1, generator=gen)[:batch]
    slots = physical.long() * 128 + positions.remainder(128)
    indices = torch.full((batch, 128), -1, dtype=torch.int32)
    for row, pos in enumerate(positions.tolist()):
        length = min(pos + 1, 128)
        logical = torch.arange(pos - length + 1, pos + 1)
        indices[row, :length] = (physical[row] * 128 + logical.remainder(128)).to(torch.int32)
    cache, cache_scale = quantize_mxfp8_cache(rand(batch + 1, 128, 1, HEAD_DIM).to(torch.bfloat16))
    values = (rand(batch, D).to(torch.bfloat16), qa, qas,
              (rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16), qb, qbs, kv, kvs,
              (rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16), rand(LOCAL_H) * 2,
              *quantize_grouped_mxfp8_weight(
                  (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
              ),
              ob, obs, angles.cos(), angles.sin(), slots, indices, cache,
              cache_scale)
    return dict(zip(SWA_INPUT_NAMES, values))


def make_packed_swa_inputs(tokens=257, requests=4, seed=17, case="mixed", start_pos=None, query_lens=None):
    if not 1 <= requests <= min(MAX_BATCH_PER_DP, tokens):
        raise ValueError(f"requests must be in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    values = make_swa_inputs(1, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1000)
    lengths = torch.tensor(packed_query_lengths(tokens, requests, query_lens), dtype=torch.int64)
    if query_lens is None and requests > 1 and lengths[-1] > 16:
        lengths[0] += 7
        lengths[-1] -= 7
    prefixes = torch.zeros(requests, dtype=torch.int64)
    if case == "prefix":
        prefixes = torch.tensor([127, 128, 511, 1048000] * 8)[:requests]
    if start_pos is not None:
        prefixes = torch.tensor(start_pos, dtype=torch.int64)
    ids = torch.repeat_interleave(torch.arange(requests), lengths)
    positions = torch.cat([torch.arange(int(p), int(p + n)) for p, n in zip(prefixes, lengths)])
    logical_capacity = int((prefixes + lengths + 129).max()) // 128 + 1
    table = torch.full((requests, logical_capacity), -1, dtype=torch.int32)
    mappings = []
    for request, (prefix, length) in enumerate(zip(prefixes.tolist(), lengths.tolist())):
        start = max(0, prefix - 127) // 128
        end = (prefix + length + 128) // 128
        mappings.extend((request, block) for block in range(start, end + 1))
    pages = torch.randperm(len(mappings) + 1, generator=gen)[:len(mappings)]
    for (request, block), page in zip(mappings, pages):
        table[request, block] = page
    slots, indices, lens = window_metadata(positions, ids, table)
    if slots.unique().numel() != tokens:
        raise ValueError("window fixture maps distinct tokens to the same physical slot")
    # Verify physical metadata independently, without relying on the helper's mask.
    for row, (request, position) in enumerate(zip(ids.tolist(), positions.tolist())):
        history = list(range(max(0, position - 127), position + 1))
        expected = [int(table[request, p // 128]) * 128 + p % 128 for p in history]
        if indices[row, :len(expected)].tolist() != expected:
            raise ValueError(f"window fixture has incorrect causal indices at row {row}")
        if not bool((indices[row, len(expected):] == -1).all()):
            raise ValueError(f"window fixture has incorrect padding at row {row}")
    angles = positions.float()[:, None] * (10000.0 ** (-torch.arange(0, 64, 2).float() / 64))
    cache, scales = quantize_mxfp8_cache(torch.randn(len(mappings) + 1, 128, 1, 512, generator=gen).bfloat16())
    hidden_random = torch.randn(tokens, 5120, generator=gen)
    hidden_bf16 = hidden_random.bfloat16()
    rope_cos = angles.cos()
    rope_sin = angles.sin()
    values.update(
        x=hidden_bf16,
        rope_cos=rope_cos, rope_sin=rope_sin,
        window_slots=slots, window_indices=indices,
        window_cache=cache, window_cache_scale=scales,
    )
    if case == "shuffle":
        order = torch.randperm(tokens, generator=gen)
        for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
            values[name] = values[name][order].contiguous()
    if case == "masked":
        values["window_slots"].fill_(-1)
        values["window_indices"].fill_(-1)
    if case == "zero":
        values["x"].zero_()
        values["window_cache"].view(torch.uint8).zero_()
    if case == "sink":
        values["attn_sink"].fill_(1000)
    print(f"[FIXTURE] tokens={tokens} requests={requests} lengths={lengths.tolist()} prefixes={prefixes.tolist()}")
    return values


GOLDEN_SELECTED_KEY_TILE = 128
GOLDEN_PREFILL_KEY_TILE = 128


def golden_selected_attention(
    q, window_value, window_indices, compressed_value, compressed_indices, sink,
    *, key_tile=GOLDEN_SELECTED_KEY_TILE,
):
    """Attend selected paged KV rows with BF16 probability tiles and one sink."""
    tokens, heads, head_dim = q.shape
    gathered = torch.cat(
        (
            window_value.flatten(0, 1).squeeze(-2)[window_indices.clamp_min(0).long()],
            compressed_value.flatten(0, 1).squeeze(-2)[compressed_indices.clamp_min(0).long()],
        ),
        dim=-2,
    ).to(torch.bfloat16).float()
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=-1)
    gathered = gathered.masked_fill(~visible.unsqueeze(-1), 0.0)
    maximum = torch.full((tokens, heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[-2], key_tile):
        tile = gathered[:, start : start + key_tile]
        tile_visible = visible[:, None, start : start + key_tile]
        logits = torch.einsum("thd,tkd->thk", q.float(), tile) * head_dim**-0.5
        logits = logits.masked_fill(~tile_visible, -1e30)
        next_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = torch.exp(maximum - next_maximum)
        probabilities = torch.exp(logits - next_maximum.unsqueeze(-1))
        probabilities = probabilities * tile_visible
        denominator = denominator * correction + probabilities.sum(-1)
        weights = probabilities.to(torch.bfloat16).float()
        numerator = numerator * correction.unsqueeze(-1) + torch.einsum("thk,tkd->thd", weights, tile)
        maximum = next_maximum
    sink_value = sink.float().view(1, -1)
    final_maximum = torch.maximum(maximum, sink_value)
    correction = torch.exp(maximum - final_maximum)
    denominator = denominator * correction + torch.exp(sink_value - final_maximum)
    return (numerator * (correction / denominator).unsqueeze(-1)).to(q.dtype)


def golden_prefill_c1a_attention(
    query: torch.Tensor,
    window_cache: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Independent Torch reference for the shared 128-key online BF16 PV arithmetic."""

    output = torch.empty_like(query)
    width = query.shape[-1]
    # Keep CPU gather temporaries bounded for 8K chunks.
    for token_begin in range(0, query.shape[0], 128):
        token_end = min(token_begin + 128, query.shape[0])
        q = query[token_begin:token_end].float()
        maximum = q.new_full(q.shape[:-1], -1e30)
        denominator = q.new_zeros(q.shape[:-1])
        numerator = torch.zeros_like(q)
        for cache, indices in ((window_cache, window_indices), (compressed_cache, compressed_indices)):
            flat_cache = cache.flatten(0, 1).squeeze(-2)
            for begin in range(0, indices.shape[-1], GOLDEN_PREFILL_KEY_TILE):
                rows = indices[token_begin:token_end, begin:begin + GOLDEN_PREFILL_KEY_TILE].long()
                valid = rows >= 0
                kv = flat_cache[rows.clamp_min(0)].to(query.dtype).float()
                kv = kv.masked_fill(~valid[..., None], 0)
                scores = torch.einsum("thd,tkd->thk", q, kv) * width**-0.5
                scores = scores + (valid.float() - 1).unsqueeze(1) * 1e30
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = torch.exp(maximum - next_maximum)
                probability = torch.exp(scores - next_maximum.unsqueeze(-1)) * valid.unsqueeze(1)
                denominator = denominator * correction + probability.sum(dim=-1)
                weighted = torch.einsum("thk,tkd->thd", probability.bfloat16().float(), kv)
                numerator = numerator * correction.unsqueeze(-1) + weighted
                maximum = next_maximum
        final_maximum = torch.maximum(maximum, sink.float())
        correction = torch.exp(maximum - final_maximum)
        denominator = denominator * correction + torch.exp(sink.float() - final_maximum)
        output[token_begin:token_end] = (numerator * (correction / denominator).unsqueeze(-1)).to(query.dtype)
    return output

def golden_paged_sparse_attention_stats(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-major online-softmax statistics from physical paged rows."""
    flat_cache = cache.flatten(0, 1).squeeze(-2)
    safe = indices.clamp_min(0).to(torch.long)
    selected = flat_cache[safe]
    logits = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    maximum = logits.amax(dim=-1)
    finite_maximum = maximum.masked_fill(~torch.isfinite(maximum), 0.0)
    exponentials = torch.exp(logits - finite_maximum.unsqueeze(-1))
    exponentials = exponentials.masked_fill(~torch.isfinite(logits), 0.0)
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("thk,tkd->thd", exponentials, selected.float())
    return maximum, denominator, numerator


def golden_paged_sparse_attention(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge token-major paged window and compressed attention with one sink."""
    parts = (
        golden_paged_sparse_attention_stats(q, window_cache, window_indices),
        golden_paged_sparse_attention_stats(q, compressed_cache, compressed_indices),
    )
    return golden_merge_attention_stats(parts, sink).to(q.dtype)


def golden_merge_attention_stats(
    parts: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge sparse sources with one attention sink in a common softmax denominator."""
    maxima = [part[0] for part in parts]
    sink_value = sink.float().view(*([1] * (maxima[0].ndim - 1)), -1)
    global_maximum = torch.maximum(torch.stack(maxima).amax(dim=0), sink_value)
    denominator = torch.exp(sink_value - global_maximum)
    numerator = torch.zeros_like(parts[0][2])
    for maximum, local_sum, local_numerator in parts:
        scale = torch.exp(maximum - global_maximum)
        scale = scale.masked_fill(~torch.isfinite(maximum), 0.0)
        denominator = denominator + local_sum * scale
        numerator = numerator + local_numerator * scale.unsqueeze(-1)
    return numerator / denominator.unsqueeze(-1)


@pl.jit.inline(auto_scope=False)
def _decode_gather_fp4(payload: pl.Tile[[GATHER_KEY_TILE, HEAD_DIM // 2], pl.UINT8]):
    """Expand both E2M1 nibbles with one lookup per packed byte."""
    bytes_i16 = pl.cast(pl.tile.arange(0, [1, 256], dtype=pl.INT32), pl.INT16, mode="rint")
    low = pl.cast(pl.cast(fp4_bf16_bits(bytes_i16), pl.FP32), pl.INT32, mode="rint")
    high = pl.cast(pl.cast(fp4_bf16_bits(pl.shrs(bytes_i16, 4)), pl.FP32), pl.INT32, mode="rint")
    low_unsigned = pl.sub(low, pl.mul(pl.shrs(low, 31), 65536))
    pair_table = pl.add(low_unsigned, pl.shls(high, 16))
    signed = pl.reinterpret_view(payload, pl.INT8)
    raw = pl.cast(pl.cast(signed, pl.FP16), pl.INT16, mode="rint")
    byte_ids = pl.ands(raw, 255)
    wide_ids = pl.cast(pl.cast(byte_ids, pl.FP32), pl.INT32, mode="rint")
    pair_indices = pl.reshape(wide_ids, [1, FP4_PAIR_TILE])
    pair_tmp = pl.create_tile([1, FP4_PAIR_TILE], dtype=pl.INT32)
    pair_bits = pl.tile.gather(pair_table, pair_indices, pair_tmp)
    values = pl.reinterpret_view(pair_bits, pl.BF16, shape=[1, GATHER_KEY_TILE * HEAD_DIM])
    return pl.reshape(values, [GATHER_KEY_TILE, HEAD_DIM])


@pl.jit.inline(auto_scope=False)
def _decode_window_fp8(payload: pl.Tile[[GATHER_KEY_TILE, HEAD_DIM], pl.UINT8]):
    """Widen E4M3FN cache values to FP16 without changing any finite value."""
    raw = pl.reinterpret_view(payload, pl.INT8)
    code = pl.cast(pl.cast(raw, pl.FP16), pl.INT16, mode="rint")
    magnitude = pl.ands(code, 127)
    normal_exponent_and_significand = pl.shls(magnitude, 7)
    normal_bits = pl.add(normal_exponent_and_significand, 8192)
    normal = pl.reinterpret_view(normal_bits, pl.FP16)
    magnitude_half = pl.cast(magnitude, pl.FP16)
    subnormal = pl.mul(magnitude_half, 0.001953125)
    is_subnormal = pl.tile.cmps(magnitude_half, 8.0, cmp_type=2)
    select_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
    absolute = pl.tile.sel(is_subnormal, subnormal, normal, select_tmp)
    sign_bit = pl.shls(pl.ands(code, 128), 8)
    bits = pl.or_(sign_bit, pl.reinterpret_view(absolute, pl.INT16))
    return pl.reinterpret_view(bits, pl.FP16)


@pl.jit.inline(auto_scope=False)
def _decode_window_run_start(ids: pl.Tile[[1, GATHER_KEY_TILE], pl.INT32]):
    first = pl.tile.read(ids, [0, 0])
    last = pl.tile.read(ids, [0, GATHER_KEY_TILE - 1])
    start = pl.cast(-1, pl.INT32)
    if first >= 0 and last - first == GATHER_KEY_TILE - 1:
        expected = pl.tile.ci(first, [1, GATHER_KEY_TILE], dtype=pl.INT32)
        mismatch = pl.tile.cmp(ids, expected, cmp_type=1)
        # Only the first two bytes of the packed mask are live.
        low = pl.tile.read(mismatch, [0, 0])
        high = pl.tile.read(mismatch, [0, 1])
        if low == 0 and high == 0:
            start = first
    return start


@pl.jit.inline(auto_scope=False)
def gather_packed(
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    max_workers: pl.constexpr,
    name_hint: pl.constexpr,
    gather_keys: pl.constexpr,
    decode: pl.constexpr,
):
    """Gather packed cache rows and dequantize them once for all query heads."""
    window_rows = pl.tensor.dim(window, 0) * 128
    compressed_rows = pl.tensor.dim(compressed, 0) * 128
    w = pl.reshape(window, [window_rows, HEAD_DIM])
    ws = pl.reshape(window_scale, [window_rows, HEAD_DIM // 32])
    c = pl.reshape(compressed, [compressed_rows, HEAD_DIM // 2])
    cs = pl.reshape(compressed_scale, [compressed_rows, HEAD_DIM // 16])
    selected_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    dst = pl.reshape(selected, [selected_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(pl.min(num_tokens * (gather_keys // GATHER_KEY_TILE), max_workers), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (gather_keys // GATHER_KEY_TILE), workers):
            token = block // (gather_keys // GATHER_KEY_TILE)
            column = block % (gather_keys // GATHER_KEY_TILE) * GATHER_KEY_TILE
            run_start = pl.cast(-1, pl.INT32)
            if column < 128:
                ids = pl.tile.load(wi, [token, column], [1, GATHER_KEY_TILE])
                safe_ids = pl.maximum(ids, 0)
                w_payload = pl.tile.create([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.UINT8)
                if decode:
                    run_start = _decode_window_run_start(ids)
                if run_start >= 0:
                    physical = pl.cast(run_start, pl.INDEX)
                    w_payload = pl.tile.load(w, [physical, 0], [GATHER_KEY_TILE, HEAD_DIM])
                    contiguous_scales = pl.tile.load(ws, [physical, 0], [GATHER_KEY_TILE, HEAD_DIM // 32])
                    contiguous_scales_fp32 = pl.cast(contiguous_scales, pl.FP32)
                    w_scale_column = pl.reshape(contiguous_scales_fp32, [GATHER_KEY_TILE * (HEAD_DIM // 32), 1])
                else:
                    w_factors = pl.tile.create([GATHER_KEY_TILE, 32], dtype=pl.BF16)
                    for lane in pl.range(GATHER_KEY_TILE):
                        physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                        w_payload = pl.tile.gather_row(w_payload, w, [lane, 0], [physical, 0], [1, HEAD_DIM])
                        w_factors = pl.tile.gather_row(w_factors, ws, [lane, 0], [physical, 0], [1, HEAD_DIM // 32])
                    w_scales = pl.cast(w_factors, pl.FP32)
                    # Compact sixteen scale values from each padded 32-wide row.
                    w_columns = pl.tile.arange(0, [1, GATHER_KEY_TILE * (HEAD_DIM // 32)], dtype=pl.INT32)
                    w_row_ids = pl.shrs(w_columns, 4)
                    w_row_bias = pl.shls(w_row_ids, 4)
                    w_order = pl.add(w_columns, w_row_bias)
                    w_scale_bits = pl.reinterpret_view(w_scales, pl.INT32)
                    w_gather_tmp = pl.tile.create([1, GATHER_KEY_TILE * (HEAD_DIM // 32)], dtype=pl.INT32)
                    w_valid_bits = pl.tile.gather(w_scale_bits, w_order, w_gather_tmp)
                    compact_scales = pl.reinterpret_view(w_valid_bits, pl.FP32)
                    w_scale_column = pl.reshape(compact_scales, [GATHER_KEY_TILE * (HEAD_DIM // 32), 1])
                if decode:
                    w_values = pl.cast(_decode_window_fp8(w_payload), pl.FP32)
                else:
                    w_values = pl.cast(decode_fp8(w_payload, GATHER_KEY_TILE, HEAD_DIM), pl.FP32)
                w_grouped = pl.reshape(w_values, [GATHER_KEY_TILE * (HEAD_DIM // 32), 32])
                w_scaled = pl.row_expand_mul(w_grouped, w_scale_column)
                w_decoded = pl.cast(w_scaled, pl.BF16, mode="rint")
                decoded = pl.reshape(w_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            else:
                ids = pl.tile.load(ci, [token, column - 128], [1, GATHER_KEY_TILE])
                safe_ids = pl.maximum(ids, 0)
                c_payload = pl.tile.create([GATHER_KEY_TILE, HEAD_DIM // 2], dtype=pl.UINT8)
                c_factors = pl.tile.create([GATHER_KEY_TILE, HEAD_DIM // 16], dtype=pl.BF16)
                for lane in pl.range(GATHER_KEY_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    c_payload = pl.tile.gather_row(c_payload, c, [lane, 0], [physical, 0], [1, HEAD_DIM // 2])
                    c_factors = pl.tile.gather_row(c_factors, cs, [lane, 0], [physical, 0], [1, HEAD_DIM // 16])
                if decode:
                    c_values = _decode_gather_fp4(c_payload)
                else:
                    c_values = decode_fp4(c_payload, GATHER_KEY_TILE, HEAD_DIM)
                c_grouped = pl.reshape(c_values, [GATHER_KEY_TILE * (HEAD_DIM // 16), 16])
                # BF16 scales can exceed the FP16 range; multiply in FP32.
                c_scales = pl.cast(c_factors, pl.FP32)
                c_scale_column = pl.reshape(c_scales, [GATHER_KEY_TILE * (HEAD_DIM // 16), 1])
                c_scaled = pl.row_expand_mul(pl.cast(c_grouped, pl.FP32), c_scale_column)
                c_decoded = pl.cast(c_scaled, pl.BF16, mode="rint")
                decoded = pl.reshape(c_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            pl.tile.store(decoded, [token * SELECTED_KEYS + column, 0], dst)
            if run_start < 0:
                zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for lane in pl.range(GATHER_KEY_TILE):
                    if pl.tile.read(ids, [0, lane]) < 0:
                        pl.tile.store(zero, [token * SELECTED_KEYS + column + lane, 0], dst)
            pl.tile.store(ids, [token, column], indices)
    return selected, indices, gathered


@pl.jit.inline(auto_scope=False)
def gather_packed_chunk(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[PACKED_QUERY_TILE, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    combined: pl.Tensor[[PACKED_QUERY_TILE, SELECTED_KEYS], pl.INT32],
    chunk_start: pl.Scalar[pl.INT32],
    active: pl.Scalar[pl.INT32],
    chunk_done: pl.Scalar[pl.TASK_ID],
):
    """Gather one query chunk with batched cache decoding."""
    window_chunk = pl.slice(window_indices, [active, WINDOW_KEYS], [chunk_start, 0])
    compressed_chunk = pl.slice(topk_indices, [active, INDEX_TOPK], [chunk_start, 0])
    selected, combined, gathered = gather_packed(
        window_cache, window_scales, compressed_cache, compressed_scales,
        window_chunk, compressed_chunk, selected, combined, active, chunk_done, max_workers=GATHER_WORKERS,
        name_hint="c2a_gather", gather_keys=640, decode=False,
    )
    return gathered


def make_attend_selected_chunk(*, vector_query=False, selected_rows=PACKED_QUERY_TILE, max_workers=32):
    """Build a bounded sparse attention launch over staged query cohorts."""
    sync_start = not vector_query

    @pl.jit.inline
    def attend_selected_chunk(
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        selected: pl.Tensor[[selected_rows, SELECTED_KEYS, HEAD_DIM], pl.BF16],
        combined: pl.Tensor[[selected_rows, SELECTED_KEYS], pl.INT32],
        sink: pl.Tensor[[LOCAL_H], pl.FP32],
        output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        chunk_start: pl.Scalar[pl.INT32],
        active: pl.Scalar[pl.INT32],
        gather_ready: pl.Scalar[pl.TASK_ID],
        name_hint: pl.constexpr,
    ):
        """Attend one sparse cohort with 64-key probability tiles."""
        tokens = pl.tensor.dim(query, 0)
        query_flat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
        output_flat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
        key_flat = pl.reshape(selected, [selected_rows * SELECTED_KEYS, HEAD_DIM])
        sink_row = pl.reshape(sink, [1, LOCAL_H])
        tasks = active * (LOCAL_H // PACKED_HEAD_TILE)
        workers = pl.max(1, pl.min(pl.min(tasks, max_workers), pl.system.available_cluster_count()))
        with pl.spmd(workers, name_hint=name_hint, deps=[gather_ready], sync_start=sync_start, optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, tasks, workers):
                row = block // (LOCAL_H // PACKED_HEAD_TILE)
                head = block % (LOCAL_H // PACKED_HEAD_TILE) * PACKED_HEAD_TILE
                q0 = (chunk_start + row) * LOCAL_H + head
                k0 = row * SELECTED_KEYS
                attention_tile(query_flat, key_flat, combined, sink_row, output_flat, q0, k0, row, head, PACKED_HEAD_TILE, SELECTED_KEYS, PACKED_KEY_TILE, False)
        return attended

    return attend_selected_chunk


@pl.jit.inline(auto_scope=False)
def attend_selected(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    key_tile: pl.constexpr,
    worker_limit: pl.constexpr,
):
    """Attend fixed sparse buffers with explicit probability tile boundaries."""
    tokens = pl.tensor.dim(query, 0)
    cache_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    head_blocks = LOCAL_H // HEAD_TILE
    tasks = num_tokens * head_blocks
    workers = pl.max(1, pl.min(pl.min(tasks, worker_limit), pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="sparse_online", deps=[selected_ready], optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, tasks, workers):
            token = block // head_blocks
            head = block % head_blocks * HEAD_TILE
            q0 = token * LOCAL_H + head
            k0 = token * SELECTED_KEYS
            attention_tile(qflat, kflat, indices, sink_row, oflat, q0, k0, token, head, HEAD_TILE, SELECTED_KEYS, key_tile, False)
    return attended


@pl.jit.inline(auto_scope=False)
def attend_decode_selected(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    key_tile: pl.constexpr,
    worker_limit: pl.constexpr,
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
):
    """Apply inverse RoPE during the final write of single-token sparse attention."""
    if num_tokens == 1:
        tokens = pl.tensor.dim(query, 0)
        cache_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
        qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
        kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
        oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
        sink_row = pl.reshape(sink, [1, LOCAL_H])
        head_tiles = LOCAL_H // HEAD_TILE
        head_workers = pl.max(1, pl.min(head_tiles, pl.system.available_cluster_count()))
        with pl.spmd(head_workers, name_hint="sparse_online", deps=[selected_ready],
                     optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
            head_worker = pl.tile.get_block_idx()
            for head_tile in pl.range(head_worker, head_tiles, head_workers):
                head = head_tile * HEAD_TILE
                _attention_tile(
                    qflat, kflat, indices, sink_row, oflat, head, 0, 0, head,
                    HEAD_TILE, SELECTED_KEYS, key_tile, False, rope_cos, rope_sin, True,
                )
    else:
        attended = attend_selected(
            query, selected, indices, sink, output, num_tokens, selected_ready, key_tile, worker_limit,
        )
    return attended


attend_c2a_chunk = make_attend_selected_chunk(vector_query=TP_SIZE != 1)


@pl.jit.inline(auto_scope=False)
def _attend_bf16_block(
    query: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    window: pl.Tensor[[BF16_PAGES_W, 128, HEAD_DIM], pl.FP32],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed: pl.Tensor[[BF16_PAGES_C, 128, HEAD_DIM], pl.FP32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[BF16_HEADS], pl.FP32],
    extents: pl.Tensor[[T_DYN, 2], pl.INT32],
    output: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    previous: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(query, 0)
    head_rows = rows * LOCAL_H
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    qbf16 = pl.create_tensor([rows, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    qbflat = pl.reshape(qbf16, [head_rows, HEAD_DIM])
    workers_q = pl.max(1, pl.min(pl.min(rows * (LOCAL_H // QUERY_HEAD_TILE), WORKERS), pl.system.available_aiv_count()))
    with pl.spmd(workers_q, name_hint="sparse_bf16_query", deps=[previous]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (LOCAL_H // QUERY_HEAD_TILE), workers_q):
            q0 = block * QUERY_HEAD_TILE
            q_values = qflat[q0:q0 + QUERY_HEAD_TILE, :]
            qbflat[q0:q0 + QUERY_HEAD_TILE, :] = pl.cast(q_values, pl.BF16, mode="rint")

    wrows = pl.tensor.dim(window, 0) * 128
    crows = pl.tensor.dim(compressed, 0) * 128
    w = pl.reshape(window, [wrows, HEAD_DIM])
    c = pl.reshape(compressed, [crows, HEAD_DIM])
    selected = pl.create_tensor([rows, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
    dst = pl.reshape(selected, [rows * SELECTED_KEYS, HEAD_DIM])
    indices = pl.create_tensor([rows, SELECTED_KEYS], dtype=pl.INT32)
    workers_kv = pl.max(1, pl.min(pl.min(rows * (SELECTED_KEYS // 16), WORKERS), pl.system.available_aiv_count()))
    with pl.spmd(workers_kv, name_hint="sparse_bf16_gather", deps=[previous]) as selected_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (SELECTED_KEYS // 16), workers_kv):
            token = block // (SELECTED_KEYS // 16)
            column = block % (SELECTED_KEYS // 16) * 16
            ww = pl.read(extents, [token, 0])
            cc = pl.read(extents, [token, 1])
            values = pl.tile.full([16, HEAD_DIM], dtype=pl.BF16, value=0.0)
            valid = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
            for lane in pl.range(16):
                position = column + lane
                row = pl.cast(-1, pl.INT32)
                value = pl.tile.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                if position < ww:
                    row = pl.read(wi, [token, position])
                    if row >= 0:
                        value = pl.tile.load(w, [pl.cast(row, pl.INDEX), 0], [1, HEAD_DIM])
                elif position < ww + cc:
                    row = pl.read(ci, [token, position - ww])
                    if row >= 0:
                        value = pl.tile.load(c, [pl.cast(row, pl.INDEX), 0], [1, HEAD_DIM])
                converted = pl.cast(value, pl.BF16, mode="rint")
                values = pl.tile.assemble(values, converted, [lane, 0])
                pl.tile.write(valid, [0, lane], row)
            pl.tile.store(values, [token * SELECTED_KEYS + column, 0], dst)
            pl.tile.store(valid, [token, column], indices)

    result = pl.create_tensor([rows, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attended = attend_selected(
        qbf16, selected, indices, sink, result, rows, selected_ready,
        KEY_TILE, PREFILL_ATTEND_WORKERS,
    )
    rflat = pl.reshape(result, [head_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    with pl.spmd(workers_q, name_hint="sparse_bf16_output", deps=[attended]) as converted:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows * (LOCAL_H // QUERY_HEAD_TILE), workers_q):
            q0 = block * QUERY_HEAD_TILE
            out_values = rflat[q0:q0 + QUERY_HEAD_TILE, :]
            oflat[q0:q0 + QUERY_HEAD_TILE, :] = pl.cast(out_values, pl.FP32)
    return converted


@pl.jit.inline(auto_scope=False)
def decode_packed_caches(
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    decoded_window: pl.Tensor[[CACHE_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    decoded_compressed: pl.Tensor[[CACHE_COMPRESSED_ROWS, HEAD_DIM], pl.BF16],
    ready: pl.Scalar[pl.TASK_ID],
):
    window_rows = pl.tensor.dim(window, 0) * 128
    w = pl.reshape(window, [window_rows, HEAD_DIM])
    ws = pl.reshape(window_scale, [window_rows, HEAD_DIM // 32])
    blocks = window_rows // GATHER_KEY_TILE
    workers = pl.max(1, pl.min(pl.min(blocks, PREFILL_FP8_CACHE_WORKERS), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="sparse_shared_window_decode", deps=[ready]) as window_ready:
        worker = pl.tile.get_block_idx()
        window_table = make_fp8_cache_table()
        begin = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.range(begin, end):
            row = block * GATHER_KEY_TILE
            w_payload = pl.tile.load(w, [row, 0], [GATHER_KEY_TILE, HEAD_DIM])
            w_factors = pl.tile.load(ws, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 32])
            w_scales = pl.cast(w_factors, pl.FP32)
            w_scale_column = pl.reshape(w_scales, [GATHER_KEY_TILE * (HEAD_DIM // 32), 1])
            w_values = decode_fp8_cache_values(w_payload, window_table, GATHER_KEY_TILE, HEAD_DIM)
            w_grouped = pl.reshape(w_values, [GATHER_KEY_TILE * (HEAD_DIM // 32), 32])
            w_scaled = pl.row_expand_mul(w_grouped, w_scale_column)
            w_decoded = pl.cast(w_scaled, pl.BF16, mode="rint")
            decoded = pl.reshape(w_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_window)

    compressed_rows = pl.tensor.dim(compressed, 0) * 128
    c = pl.reshape(compressed, [compressed_rows, HEAD_DIM // 2])
    cs = pl.reshape(compressed_scale, [compressed_rows, HEAD_DIM // 16])
    workers = pl.max(1, pl.min(pl.min(compressed_rows // GATHER_KEY_TILE, CACHE_WORKERS), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="sparse_shared_compressed_decode", deps=[ready]) as compressed_ready:
        worker = pl.tile.get_block_idx()
        byte = pl.cast(pl.tile.arange(0, [1, 256], dtype=pl.INT32), pl.INT16, mode="rint")
        low = pl.cast(pl.cast(fp4_bf16_bits(byte), pl.FP32), pl.INT32, mode="rint")
        high = pl.cast(pl.cast(fp4_bf16_bits(pl.shrs(byte, 4)), pl.FP32), pl.INT32, mode="rint")
        low_bits = pl.sub(low, pl.mul(pl.shrs(low, 31), 65536))
        pair_table = pl.add(low_bits, pl.shls(high, 16))
        pair_tmp = pl.create_tile([1, FP4_PAIR_TILE], dtype=pl.INT32)
        for block in pl.range(worker, compressed_rows // GATHER_KEY_TILE, workers):
            row = block * GATHER_KEY_TILE
            c_payload = pl.tile.load(c, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 2])
            c_factors = pl.tile.load(cs, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 16])
            signed = pl.reinterpret_view(c_payload, pl.INT8)
            half = pl.cast(signed, pl.FP16)
            byte_ids = pl.ands(pl.cast(half, pl.INT16, mode="rint"), 255)
            wide_ids = pl.cast(pl.cast(byte_ids, pl.FP32), pl.INT32, mode="rint")
            pair_indices = pl.reshape(wide_ids, [1, FP4_PAIR_TILE])
            pair_bits = pl.tile.gather(pair_table, pair_indices, pair_tmp)
            values = pl.reinterpret_view(pair_bits, pl.BF16, shape=[1, GATHER_KEY_TILE * HEAD_DIM])
            c_values = pl.reshape(values, [GATHER_KEY_TILE, HEAD_DIM])
            c_grouped = pl.reshape(c_values, [GATHER_KEY_TILE * (HEAD_DIM // 16), 16])
            # BF16 scales can exceed the FP16 range; multiply in FP32.
            c_scales = pl.cast(c_factors, pl.FP32)
            c_scale_column = pl.reshape(c_scales, [GATHER_KEY_TILE * (HEAD_DIM // 16), 1])
            c_values_fp32 = pl.cast(c_grouped, pl.FP32)
            c_scaled = pl.row_expand_mul(c_values_fp32, c_scale_column)
            c_decoded = pl.cast(c_scaled, pl.BF16, mode="rint")
            decoded = pl.reshape(c_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_compressed)
    return window_ready, compressed_ready


@pl.jit.inline(auto_scope=False)
def _matches_decoded_run(
    ids: pl.Tile[[1, SELECTED_GATHER_TILE], pl.INT32],
    expected: pl.Tile[[1, SELECTED_GATHER_TILE], pl.INT32],
):
    mismatch = pl.tile.cmp(ids, expected, cmp_type=1)
    bits = pl.tile.reinterpret_view(mismatch, pl.INT32)
    return pl.tile.read(bits, [0, 0]) == 0 and pl.tile.read(bits, [0, 1]) == 0


@pl.jit.inline(auto_scope=False)
def _copy_decoded_window_runs(
    cache: pl.Tensor,
    ids: pl.Tile[[1, SELECTED_GATHER_TILE], pl.INT32],
    output: pl.Tensor,
    output_row: pl.Scalar[pl.INDEX],
    first: pl.Scalar[pl.INT32],
    last: pl.Scalar[pl.INT32],
):
    rows = pl.cast(WINDOW_KEYS - first % WINDOW_KEYS, pl.INDEX)
    copied = pl.cast(0, pl.INT32)
    if first >= 0 and rows < SELECTED_GATHER_TILE:
        second = pl.tile.read(ids, [0, rows])
        if second >= 0 and last - second == SELECTED_GATHER_TILE - rows - 1:
            columns = pl.tile.arange(0, [1, SELECTED_GATHER_TILE], dtype=pl.INT32)
            first_sequence = pl.tile.ci(first, [1, SELECTED_GATHER_TILE], dtype=pl.INT32)
            second_base = second - pl.cast(rows, pl.INT32)
            second_sequence = pl.tile.ci(second_base, [1, SELECTED_GATHER_TILE], dtype=pl.INT32)
            rows_f32 = pl.cast(pl.cast(rows, pl.INT32), pl.FP32)
            first_mask = pl.tile.cmps(pl.cast(columns, pl.FP32), rows_f32, cmp_type=2)
            expected = pl.tile.select(first_mask, first_sequence, second_sequence)
            if _matches_decoded_run(ids, expected):
                first_row = pl.cast(first, pl.INDEX)
                second_row = pl.cast(second, pl.INDEX)
                values = pl.tile.create([SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.BF16)
                values = pl.tile.gather_row(
                    values, cache, [0, 0], [first_row, 0], [SELECTED_GATHER_TILE, HEAD_DIM],
                    valid_shape=[rows, HEAD_DIM],
                )
                values = pl.tile.gather_row(
                    values, cache, [rows, 0], [second_row, 0], [SELECTED_GATHER_TILE, HEAD_DIM],
                    valid_shape=[SELECTED_GATHER_TILE - rows, HEAD_DIM],
                )
                pl.tile.store(values, [output_row, 0], output)
                copied = pl.cast(1, pl.INT32)
    return copied


@pl.jit.inline(auto_scope=False)
def gather_decoded_rows(
    cache: pl.Tensor,
    ids: pl.Tile[[1, SELECTED_GATHER_TILE], pl.INT32],
    output: pl.Tensor,
    output_row: pl.Scalar[pl.INDEX],
    sliding_window: pl.constexpr,
):
    """Copy contiguous rows, gather irregular rows, and fill masked lanes."""
    first = pl.tile.read(ids, [0, 0])
    last = pl.tile.read(ids, [0, SELECTED_GATHER_TILE - 1])
    contiguous = pl.cast(0, pl.INT32)
    if first >= 0 and last - first == SELECTED_GATHER_TILE - 1:
        expected = pl.tile.ci(first, [1, SELECTED_GATHER_TILE], dtype=pl.INT32)
        # The 64-lane comparison mask occupies two INT32 words.
        contiguous = pl.cast(_matches_decoded_run(ids, expected), pl.INT32)
    if contiguous != 0:
        start = pl.cast(first, pl.INDEX)
        contiguous_rows = pl.tile.load(cache, [start, 0], [SELECTED_GATHER_TILE, HEAD_DIM])
        pl.tile.store(contiguous_rows, [output_row, 0], output)
    else:
        copied = pl.cast(0, pl.INT32)
        if sliding_window:
            copied = _copy_decoded_window_runs(cache, ids, output, output_row, first, last)
        if copied == 0:
            values = pl.tile.full([SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
            has_values = pl.cast(1, pl.INT32)
            if first < 0 and last < 0:
                sign = pl.shrs(ids, 31)
                valid = pl.tile.cmps(sign, 0, cmp_type=0)
                valid_bits = pl.tile.reinterpret_view(valid, pl.INT32)
                valid_low = pl.tile.read(valid_bits, [0, 0])
                valid_high = pl.tile.read(valid_bits, [0, 1])
                has_values = pl.cast(valid_low != 0 or valid_high != 0, pl.INT32)
            if has_values != 0:
                for lane in pl.range(SELECTED_GATHER_TILE):
                    physical = pl.tile.read(ids, [0, lane])
                    if physical >= 0:
                        row = pl.cast(physical, pl.INDEX)
                        values = pl.tile.gather_row(values, cache, [lane, 0], [row, 0], [1, HEAD_DIM])
            pl.tile.store(values, [output_row, 0], output)


@pl.jit.inline(auto_scope=False)
def _stream_decoded_kv(
    query: pl.Tensor,
    window: pl.Tensor,
    compressed: pl.Tensor,
    wi: pl.Tensor,
    ci: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    active: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    indices_ready: pl.Scalar[pl.TASK_ID],
):
    """Stream selected BF16 KV tiles through a per-worker ring."""
    capacity = pl.tensor.dim(query, 0)
    query_rows = capacity * LOCAL_H
    qflat = pl.reshape(query, [query_rows, HEAD_DIM])
    oflat = pl.reshape(output, [query_rows, HEAD_DIM])
    sinks = pl.reshape(sink, [1, LOCAL_H])
    parts = pl.create_tensor([capacity, 1], dtype=pl.INT32)
    token_groups = (active + 7) // 8
    workers = pl.max(1, pl.min(pl.min(token_groups, STREAM_WORKER_TILE * 2), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="prefill_sparse_kv_part_count", deps=[indices_ready]) as counted:
        worker = pl.tile.get_block_idx()
        columns = pl.tile.arange(0, [1, INDEX_TOPK], dtype=pl.INT32)
        part_codes = pl.cast(pl.add(pl.tile.shrs(columns, 7), 2), pl.FP32)
        scratch = pl.create_tile([8, INDEX_TOPK], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        for group in pl.range(worker, token_groups, workers):
            start = group * 8
            valid_rows = pl.min(active - start, 8)
            count_ids = pl.load(ci, [start, 0], [8, INDEX_TOPK], valid_shape=[valid_rows, INDEX_TOPK])
            ids_f32 = pl.cast(count_ids, pl.FP32)
            visible = pl.minimum(pl.maximum(pl.add(ids_f32, 1.0), 0.0), 1.0)
            marked = pl.col_expand_mul(visible, part_codes)
            last = pl.maximum(pl.row_max(marked, tmp_tile=scratch), 1.0)
            pl.store(pl.set_validshape(pl.cast(last, pl.INT32), valid_rows, 1), [start, 0], parts)

    blocks = pl.max(1, pl.min(pl.min(active, STREAM_WORKER_TILE), pl.system.available_cluster_count()))
    stages = blocks * STREAM_SLOTS
    key_ring = pl.create_tensor([stages * STREAM_KEY_TILE, HEAD_DIM], dtype=pl.BF16)
    scores = pl.create_tensor([stages * LOCAL_H, STREAM_KEY_TILE], dtype=pl.FP32)
    weights = pl.create_tensor([stages * LOCAL_H, STREAM_KEY_TILE], dtype=pl.BF16)
    projections = pl.create_tensor([stages * LOCAL_H, HEAD_DIM], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(blocks, name_hint="prefill_sparse_kv_pipeline", deps=[counted, ready]) as complete:
        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        for token in pl.range(worker, active, blocks):
            live = pl.read(parts, [token, 0])
            kv_l1 = pl.create_tile([STREAM_SLOTS * STREAM_KEY_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
            q = pl.load(qflat, [token * LOCAL_H, 0], [LOCAL_H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            for step in pl.range(live + STREAM_PV_DELAY):
                if step < live:
                    slot = step % STREAM_SLOTS
                    stage = worker * STREAM_SLOTS + slot
                    l1_row = slot * STREAM_KEY_TILE
                    pl.system.sync_wait(STREAM_KEY_READY, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    kv_l1 = pl.gather_row(kv_l1, key_ring, [l1_row, 0], [stage * STREAM_KEY_TILE, 0], [STREAM_KEY_TILE, HEAD_DIM])
                    k = pl.tile.slice(pl.tile.transpose_view(kv_l1), [HEAD_DIM, STREAM_KEY_TILE], [0, l1_row])
                    qk = pl.matmul(q, k, out_dtype=pl.FP32)
                    pl.store(qk, [stage * LOCAL_H, 0], scores)
                    pl.system.sync_set(STREAM_SCORE_READY, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if step >= STREAM_PV_DELAY:
                    part = step - STREAM_PV_DELAY
                    slot = part % STREAM_SLOTS
                    stage = worker * STREAM_SLOTS + slot
                    pl.system.sync_wait(STREAM_PROB_READY, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    prob = pl.load(weights, [stage * LOCAL_H, 0], [LOCAL_H, STREAM_KEY_TILE], target_memory=pl.MemorySpace.Mat)
                    v = pl.tile.slice(kv_l1, [STREAM_KEY_TILE, HEAD_DIM], [slot * STREAM_KEY_TILE, 0])
                    pv = pl.matmul(prob, v, out_dtype=pl.FP32)
                    pl.store(pv, [stage * LOCAL_H, 0], projections)
                    pl.system.sync_set(STREAM_PV_READY, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)

            for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts)
                head = lane * STREAM_HEAD_TILE
                initial_m = pl.tile.full([1, STREAM_HEAD_TILE], dtype=pl.FP32, value=-1e30)
                initial_l = pl.tile.full([1, STREAM_HEAD_TILE], dtype=pl.FP32, value=0.0)
                initial_left = pl.tile.full([STREAM_HEAD_TILE, STREAM_FEATURE_TILE], dtype=pl.FP32, value=0.0)
                initial_right = pl.tile.full([STREAM_HEAD_TILE, STREAM_FEATURE_TILE], dtype=pl.FP32, value=0.0)
                for step, (m, l, left, right, alpha_old, alpha_new) in pl.range(
                    live + STREAM_PV_DELAY + 1,
                    init_values=(initial_m, initial_l, initial_left, initial_right, initial_l, initial_l),
                ):
                    if step < live:
                        stage = worker * STREAM_SLOTS + step % STREAM_SLOTS
                        dst_row = stage * STREAM_KEY_TILE + lane * SELECTED_GATHER_TILE
                        if step == 0:
                            w_ids = pl.tile.load(wi, [token, lane * SELECTED_GATHER_TILE], [1, SELECTED_GATHER_TILE])
                            gather_decoded_rows(window, w_ids, key_ring, dst_row, True)
                        else:
                            col = (step - 1) * STREAM_KEY_TILE + lane * SELECTED_GATHER_TILE
                            c_ids = pl.tile.load(ci, [token, col], [1, SELECTED_GATHER_TILE])
                            gather_decoded_rows(compressed, c_ids, key_ring, dst_row, False)
                        pl.system.sync_set(STREAM_KEY_READY, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)

                    if step > 0 and step <= live:
                        part = step - 1
                        stage_row = (worker * STREAM_SLOTS + part % STREAM_SLOTS) * LOCAL_H + head
                        pl.system.sync_wait(STREAM_SCORE_READY, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        score = pl.load(scores, [stage_row, 0], [STREAM_HEAD_TILE, STREAM_KEY_TILE], target_memory=pl.MemorySpace.Vec)
                        score_ids = pl.tile.load(wi, [token, 0], [1, STREAM_KEY_TILE])
                        if part > 0:
                            score_ids = pl.tile.load(ci, [token, (part - 1) * STREAM_KEY_TILE], [1, STREAM_KEY_TILE])
                        wide = pl.cast(score_ids, pl.FP32)
                        valid = pl.minimum(pl.maximum(pl.add(wide, 1.0), 0.0), 1.0)
                        bias = pl.mul(pl.sub(valid, 1.0), 1e30)
                        scaled = pl.col_expand_add(pl.mul(score, SCALE), bias)
                        tmp = pl.create_tile([STREAM_HEAD_TILE, STREAM_KEY_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                        local_m = pl.reshape(pl.row_max(scaled, tmp), [1, STREAM_HEAD_TILE])
                        next_m = pl.maximum(m, local_m)
                        alpha = pl.exp(pl.sub(m, next_m))
                        max_col = pl.reshape(next_m, [STREAM_HEAD_TILE, 1])
                        p = pl.col_expand_mul(pl.exp(pl.row_expand_sub(scaled, max_col)), valid)
                        local_l = pl.reshape(pl.row_sum(p, tmp), [1, STREAM_HEAD_TILE])
                        next_l = pl.add(pl.mul(l, alpha), local_l)
                        pl.store(pl.cast(p, pl.BF16, mode="rint"), [stage_row, 0], weights)
                        pl.system.sync_set(STREAM_PROB_READY, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
                        softmax_m, softmax_l, softmax_alpha = pl.yield_(next_m, next_l, alpha)
                    else:
                        softmax_m, softmax_l, softmax_alpha = pl.yield_(m, l, initial_l)

                    if step >= STREAM_PV_DELAY + 1:
                        part = step - STREAM_PV_DELAY - 1
                        stage_row = (worker * STREAM_SLOTS + part % STREAM_SLOTS) * LOCAL_H + head
                        pl.system.sync_wait(STREAM_PV_READY, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        left_pv = pl.load(projections, [stage_row, 0], [STREAM_HEAD_TILE, STREAM_FEATURE_TILE])
                        right_pv = pl.load(projections, [stage_row, STREAM_FEATURE_TILE], [STREAM_HEAD_TILE, STREAM_FEATURE_TILE])
                        alpha_col = pl.reshape(alpha_old, [STREAM_HEAD_TILE, 1])
                        next_left = pl.add(pl.row_expand_mul(left, alpha_col), left_pv)
                        next_right = pl.add(pl.row_expand_mul(right, alpha_col), right_pv)
                        merge_left, merge_right = pl.yield_(next_left, next_right)
                    else:
                        merge_left, merge_right = pl.yield_(left, right)
                    final_m, final_l, final_left, final_right, final_old, final_new = pl.yield_(
                        softmax_m, softmax_l, merge_left, merge_right, alpha_new, softmax_alpha,
                    )

                local_sink = pl.load(sinks, [0, head], [1, STREAM_HEAD_TILE], target_memory=pl.MemorySpace.Vec)
                final_max = pl.maximum(local_sink, final_m)
                correction = pl.exp(pl.sub(final_m, final_max))
                final_den = pl.add(pl.mul(final_l, correction), pl.exp(pl.sub(local_sink, final_max)))
                scale = pl.reshape(pl.div(correction, final_den), [STREAM_HEAD_TILE, 1])
                out_left = pl.cast(pl.row_expand_mul(final_left, scale), pl.BF16, mode="rint")
                out_right = pl.cast(pl.row_expand_mul(final_right, scale), pl.BF16, mode="rint")
                row = token * LOCAL_H + head
                pl.store(out_left, [row, 0], oflat)
                pl.store(out_right, [row, STREAM_FEATURE_TILE], oflat)
    return complete


@pl.jit.inline(auto_scope=False)
def _attend_cache_shared(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    query_ready: pl.Scalar[pl.TASK_ID],
):
    window_rows = pl.tensor.dim(window, 0) * 128
    compressed_rows = pl.tensor.dim(compressed, 0) * 128
    decoded_window = pl.create_tensor([window_rows, HEAD_DIM], dtype=pl.BF16)
    decoded_compressed = pl.create_tensor([compressed_rows, HEAD_DIM], dtype=pl.BF16)
    window_ready, compressed_ready = decode_packed_caches(
        window, window_scale, compressed, compressed_scale,
        decoded_window, decoded_compressed, query_ready,
    )
    ready = pl.system.task_dummy(deps=[window_ready, compressed_ready])
    _stream_decoded_kv(
        query, decoded_window, decoded_compressed, window_indices, compressed_indices,
        sink, output, num_tokens, ready, query_ready,
    )
    return output


@pl.jit.inline(auto_scope=False)
def _attend_packed_block(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    query_ready: pl.Scalar[pl.TASK_ID],
    key_tile: pl.constexpr,
):
    selected, indices, gathered = gather_packed(
        window, window_scale, compressed, compressed_scale,
        window_indices, compressed_indices, selected, indices, num_tokens, query_ready, max_workers=GATHER_WORKERS,
        name_hint="c2a_gather", gather_keys=640, decode=False,
    )
    attended = attend_selected(
        query, selected, indices, sink, output, num_tokens, gathered,
        key_tile, PREFILL_ATTEND_WORKERS,
    )
    return attended


@pl.jit.inline(auto_scope=False)
def attend_packed(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
    key_tile: pl.constexpr,
    token_tile: pl.constexpr,
    shared_prefix: pl.Scalar[pl.INT32] = -1,
):
    marker = pl.create_tensor([1, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="sparse_query_ready", deps=[source_ready]) as ready:
        block = pl.tile.get_block_idx()
        marker[block:block + 1, :] = pl.cast(query[0:1, 0:16], pl.FP32)

    cache_rows = (pl.tensor.dim(window, 0) + pl.tensor.dim(compressed, 0)) * 128
    # The packed host groups longer chunks first. Only those rows use the
    # cache-streaming reduction selected for the equivalent single request.
    shared = pl.cast(pl.max(shared_prefix, 0), pl.INT32)
    if shared_prefix < 0:
        if cache_rows <= CACHE_MAX_ROWS and cache_rows < num_tokens * SELECTED_KEYS:
            shared = num_tokens
    if shared > 0:
        _attend_cache_shared(
            query, window_indices, window, window_scale, compressed_indices,
            compressed, compressed_scale, sink, output, shared, ready,
        )
    if shared < num_tokens:
        remaining = num_tokens - shared
        if TP_SIZE == 1 and remaining <= 12:
            cohort_q = pl.slice(query, [remaining, LOCAL_H * HEAD_DIM], [shared, 0])
            cohort_wi = pl.slice(window_indices, [remaining, 128], [shared, 0])
            cohort_ci = pl.slice(compressed_indices, [remaining, INDEX_TOPK], [shared, 0])
            cohort_out = pl.slice(output, [remaining, LOCAL_H * HEAD_DIM], [shared, 0])
            ready = attend_packed_cohorts(
                cohort_q, cohort_wi, window, window_scale, cohort_ci, compressed, compressed_scale,
                sink, cohort_out, remaining, ready, gather_name="c2a_gather", attn_name="c2a_sparse_attn",
            )
        else:
            capacity = pl.max(1, pl.min(token_tile, remaining))
            selected = pl.create_tensor([capacity, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
            indices = pl.create_tensor([capacity, SELECTED_KEYS], dtype=pl.INT32)
            for offset in pl.range(shared, num_tokens, token_tile):
                count = pl.min(token_tile, num_tokens - offset)
                q = pl.slice(query, [count, LOCAL_H * HEAD_DIM], [offset, 0])
                wi = pl.slice(window_indices, [count, 128], [offset, 0])
                ci = pl.slice(compressed_indices, [count, INDEX_TOPK], [offset, 0])
                out = pl.slice(output, [count, LOCAL_H * HEAD_DIM], [offset, 0])
                ready = _attend_packed_block(
                    q, wi, window, window_scale,
                    ci, compressed, compressed_scale,
                    sink, out, selected, indices, count, ready, key_tile,
                )
    return output


@pl.jit.inline(auto_scope=False)
def attend_bf16(
    query: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
    window: pl.Tensor[[BF16_PAGES_W, 128, HEAD_DIM], pl.FP32],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed: pl.Tensor[[BF16_PAGES_C, 128, HEAD_DIM], pl.FP32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[BF16_HEADS], pl.FP32],
    extents: pl.Tensor[[T_DYN, 2], pl.INT32],
    output: pl.Tensor[[T_DYN, BF16_HEADS, HEAD_DIM], pl.FP32],
):
    rows = pl.tensor.dim(query, 0)
    heads = pl.tensor.dim(query, 1)
    qflat = pl.reshape(query, [rows * LOCAL_H, HEAD_DIM])
    marker = pl.create_tensor([1, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="sparse_bf16_ready") as ready:
        block = pl.tile.get_block_idx()
        marker[block:block + 1, :] = qflat[0:1, 0:16]

    for offset in pl.range(0, rows, TOKEN_TILE):
        count = pl.min(TOKEN_TILE, rows - offset)
        q = pl.slice(query, [count, heads, HEAD_DIM], [offset, 0, 0])
        widx = pl.slice(wi, [count, 128], [offset, 0])
        cidx = pl.slice(ci, [count, INDEX_TOPK], [offset, 0])
        limits = pl.slice(extents, [count, 2], [offset, 0])
        out = pl.slice(output, [count, heads, HEAD_DIM], [offset, 0, 0])
        ready = _attend_bf16_block(q, window, widx, compressed, cidx, sink, limits, out, ready)
    return output


@pl.jit(auto_scope=False)
def l2_sparse_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
):
    rows = pl.tensor.dim(query, 0)
    source_ready = pl.system.task_dummy(deps=[])
    output = attend_packed(
        query, window_indices, window, window_scale, compressed_indices,
        compressed, compressed_scale, sink, output, rows, source_ready, KEY_TILE, TOKEN_TILE, -1,
    )
    return output


_attend_cohort = make_attend_selected_chunk(
    vector_query=False, selected_rows=PACKED_COHORT_TILE, max_workers=16,
)


@pl.jit.inline(auto_scope=False)
def attend_packed_cohorts(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    rows: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
    gather_name: pl.constexpr,
    attn_name: pl.constexpr,
):
    selected_a = pl.create_tensor([PACKED_COHORT_TILE, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
    selected_b = pl.create_tensor([PACKED_COHORT_TILE, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
    combined_a = pl.create_tensor([PACKED_COHORT_TILE, SELECTED_KEYS], dtype=pl.INT32)
    combined_b = pl.create_tensor([PACKED_COHORT_TILE, SELECTED_KEYS], dtype=pl.INT32)
    done_a = source_ready
    done_b = source_ready
    for start in pl.range(0, rows, 2 * PACKED_COHORT_TILE):
        width = pl.min(PACKED_COHORT_TILE, (rows - start + 1) // 2)
        active_a = width
        wi_a = pl.slice(window_indices, [active_a, WINDOW_KEYS], [start, 0])
        ci_a = pl.slice(compressed_indices, [active_a, INDEX_TOPK], [start, 0])
        query_a = pl.slice(query, [active_a, LOCAL_H * HEAD_DIM], [start, 0])
        output_a = pl.slice(output, [active_a, LOCAL_H * HEAD_DIM], [start, 0])
        selected_a, combined_a, gathered_a = gather_packed(
            window, window_scale, compressed, compressed_scale,
            wi_a, ci_a, selected_a, combined_a, active_a, done_a, max_workers=32,
            name_hint=gather_name, gather_keys=640, decode=False,
        )
        done_a = _attend_cohort(
            query_a, selected_a, combined_a, sink, output_a, 0, active_a, gathered_a, name_hint=attn_name,
        )
        if start + width < rows:
            active_b = pl.min(PACKED_COHORT_TILE, rows - start - width)
            wi_b = pl.slice(window_indices, [active_b, WINDOW_KEYS], [start + width, 0])
            ci_b = pl.slice(compressed_indices, [active_b, INDEX_TOPK], [start + width, 0])
            query_b = pl.slice(query, [active_b, LOCAL_H * HEAD_DIM], [start + width, 0])
            output_b = pl.slice(output, [active_b, LOCAL_H * HEAD_DIM], [start + width, 0])
            selected_b, combined_b, gathered_b = gather_packed(
                window, window_scale, compressed, compressed_scale,
                wi_b, ci_b, selected_b, combined_b, active_b, done_b, max_workers=32,
                name_hint=gather_name, gather_keys=640, decode=False,
            )
            done_b = _attend_cohort(
                query_b, selected_b, combined_b, sink, output_b, 0, active_b, gathered_b, name_hint=attn_name,
            )
    complete = pl.system.task_dummy(deps=[done_a, done_b])
    return complete


@pl.jit(auto_scope=False)
def l2_c2a_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
):
    query.bind_dynamic(0, T_DYN)
    window.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed.bind_dynamic(0, CMP_BLOCKS_DYN)
    rows = pl.tensor.dim(query, 0)
    source_ready = pl.system.task_dummy(deps=[])
    _done = attend_packed_cohorts(
        query, window_indices, window, window_scale, compressed_indices,
        compressed, compressed_scale, sink, output, rows, source_ready,
        gather_name="c2a_gather", attn_name="c2a_sparse_attn",
    )
    return output


@pl.jit(auto_scope=False)
def l2_sparse_attention_bf16(
    query: pl.Tensor[[T_DYN, LOCAL_H, HEAD_DIM], pl.FP32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, HEAD_DIM], pl.FP32],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, HEAD_DIM], pl.FP32],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    extents: pl.Tensor[[T_DYN, 2], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, HEAD_DIM], pl.FP32]],
):
    output = attend_bf16(query, window, window_indices, compressed, compressed_indices, sink, extents, output)
    return output


def swa_core_reference(inputs):
    """Compute sink-aware FP32 attention over the decoded BF16 SWA rows."""
    tokens = int(inputs["num_tokens"])
    cache = inputs["cache"]
    if "scales" in inputs:
        payload, scales = cache.view(torch.float8_e4m3fn), inputs["scales"]
    else:
        payload = cache[..., :HEAD_DIM].contiguous().view(torch.float8_e4m3fn)
        scale_bytes = cache[..., HEAD_DIM:].contiguous()
        scales = scale_bytes.view(torch.bfloat16)
    decoded = payload.float() * scales.float().repeat_interleave(32, dim=-1)
    decoded = decoded.bfloat16().reshape(-1, HEAD_DIM)
    query = inputs["query"][:tokens].reshape(tokens, LOCAL_H, HEAD_DIM).float()
    indices = inputs["indices"][:tokens].long()
    selected = decoded[indices.clamp_min(0)].float()
    selected = selected.masked_fill((indices < 0).unsqueeze(-1), 0.0)
    scores = torch.einsum("thd,tkd->thk", query, selected) * SCALE
    scores = scores.masked_fill((indices < 0).unsqueeze(1), -torch.inf)
    sink = inputs["sink"].view(1, LOCAL_H, 1).expand(tokens, -1, -1)
    probability = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
    attended = torch.einsum("thk,tkd->thd", probability, selected)
    reference = inputs["output"].float().clone()
    reference[:tokens] = attended.flatten(1)
    return reference


def make_swa_case(case, packed=False):
    """Create masked, prefix, wrap and irregular rows for one dispatch route."""
    cases = {"gather": 3, "plain": 33, "runs": 129}
    if case not in cases:
        raise ValueError(f"Unknown SWA case: {case}")
    active = cases[case]
    rows = active + 1
    generator = torch.Generator().manual_seed(4102026 + active)
    query = torch.randn(rows, LOCAL_H * HEAD_DIM, generator=generator).mul_(0.35).bfloat16()
    values = torch.randn(2, 128, 1, HEAD_DIM, generator=generator).mul_(0.6).bfloat16()
    cache, scales = quantize_mxfp8_cache(values)
    indices = torch.full((rows, SWA_WINDOW), -1, dtype=torch.int32)
    for row in range(active):
        kind = row % 6
        if kind == 1:
            indices[row, :13] = torch.arange(13, dtype=torch.int32) + 127
        elif kind == 2:
            indices[row] = (torch.arange(SWA_WINDOW, dtype=torch.int32) + 221) % 256
        elif kind == 3:
            indices[row] = torch.arange(SWA_WINDOW, dtype=torch.int32) + 96
        elif kind == 4:
            indices[row] = torch.randint(0, 256, (SWA_WINDOW,), generator=generator, dtype=torch.int32)
            indices[row, ::7] = -1
            indices[row, 5:9] = 17
        elif kind == 5:
            indices[row, :65] = torch.arange(65, dtype=torch.int32) + 64
    inputs = dict(query=query, cache=cache.view(torch.uint8), scales=scales, indices=indices)
    inputs["sink"] = torch.linspace(-6.0, -1.0, LOCAL_H)
    inputs["output"] = torch.full_like(query, 7.25)
    inputs["num_tokens"] = active
    if packed:
        payload_bytes = cache.view(torch.uint8)
        scale_bytes = scales.view(torch.uint8)
        inputs["cache"] = torch.cat((payload_bytes, scale_bytes), dim=-1).contiguous()
        del inputs["scales"]
    return inputs, swa_core_reference(inputs)


def build_swa_specs(inputs):
    """Describe the SWA core ABI, including its initialized InOut output."""
    from golden import ScalarSpec, TensorSpec

    specs = []
    for name, value in inputs.items():
        if name == "num_tokens":
            specs.append(ScalarSpec(name, torch.int32, int(value)))
        else:
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs


def golden_swa_core(tensors):
    """Write active attention rows and preserve the inactive output sentinel."""
    tensors["output"].copy_(swa_core_reference(tensors).bfloat16())


def compare_swa_core(actual, expected, *, inputs, reference, **kwargs):
    """Check the FP32 core gate, exact masked zeros and the inactive sentinel."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    difference = (actual[:active].float() - reference[:active]).abs()
    norm = float(reference[:active].norm())
    relative_l2 = float(difference.norm()) / norm if norm else float(difference.norm())
    element_check = ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.0)
    elements_passed, elements_detail = element_check(actual[:active], reference[:active], inputs=inputs, **kwargs)
    masked = (inputs["indices"][:active] < 0).all(dim=-1)
    masked_zero = bool((actual[:active][masked] == 0).all())
    inactive_exact = bool((actual[active:] == 7.25).all())
    passed = elements_passed and relative_l2 < 0.006 and masked_zero and inactive_exact
    detail = f"relative_l2={relative_l2:.8g}, masked_zero={masked_zero}, inactive_exact={inactive_exact}"
    if not elements_passed:
        detail += f", {elements_detail}"
    return passed, detail


def validate_swa_core(args):
    """Validate staged and direct SWA on the supported cache layouts."""
    from functools import partial

    from golden import run

    cases = ("gather", "plain", "runs") if args.swa_case == "all" else (args.swa_case,)
    entries = {
        "swa": l2_sparse_attention_swa,
        "swa-packed": l2_sparse_attention_swa_packed,
        "swa-direct": l2_sparse_attention_swa_direct,
    }
    modes = {"all": tuple(entries), "swa-both": ("swa", "swa-packed")}.get(args.mode, (args.mode,))
    records = []
    for mode in modes:
        entry = entries[mode]
        mode_cases = ("runs",) if args.mode == "all" and mode == "swa-direct" and args.swa_case == "all" else cases
        for case in mode_cases:
            inputs, reference = make_swa_case(case, packed=mode == "swa-packed")
            result = run(
                entry,
                build_swa_specs(inputs),
                golden_fn=golden_swa_core,
                compare_fn={"output": partial(compare_swa_core, reference=reference)},
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                compile_only=args.compile_only,
                save_data=True,
            )
            records.append(dict(mode=mode, case=case, passed=result.passed, work_dir=str(result.work_dir)))
            if not result.passed:
                raise RuntimeError(result.error)
            status = "COMPILE_ONLY" if args.compile_only else "DEVICE_VALIDATED"
            print(status, mode, case, result.work_dir)
    return records


def golden_swa_attention(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> dict:
    """Evaluate SWA and publish the current token KV rows."""
    query, window_kv, _ = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
    )
    window_payload, updated_window_scale = _publish_quantized_cache(
        window_cache,
        window_cache_scale,
        window_kv,
        window_slots,
        quantize_mxfp8_cache,
    )
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    window_stats = golden_paged_sparse_attention_stats(query, quantized_window, window_indices)
    attended = golden_merge_attention_stats((window_stats,), attn_sink).to(query.dtype)
    output = golden_o_proj(
                 attended, rope_cos, rope_sin, wo_a, wo_b,
             )
    return {
        'output': output,
        'window_cache': window_payload,
        'window_cache_scale': updated_window_scale,
        'compressed_cache': None,
        'compressed_cache_scale': None,
        'index_cache': None,
        'index_cache_scale': None,
        'state_cache': None,
        'topk_indices': None,
        'candidate_mask': None,
        'pool_block_ids': None,
        'pool_block_counts': None,
    }


def golden_compressed_attention(
    mode: str,
    ratio: int,
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    compressed_indices: torch.Tensor | None,
    compressor_wkv: torch.Tensor | None,
    compressor_wgate: torch.Tensor | None,
    compressor_norm_weight: torch.Tensor | None,
    state_block_table: torch.Tensor | None,
    state_cache: torch.Tensor | None,
    compressed_slots: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    compressed_rope_cos: torch.Tensor | None,
    compressed_rope_sin: torch.Tensor | None,
    index_wk: torch.Tensor | None,
    index_norm_weight: torch.Tensor | None,
    index_wq_b: torch.Tensor | None,
    index_wq_b_scale: torch.Tensor | None,
    index_weights_proj: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    index_cache_scale: torch.Tensor | None,
    index_block_table: torch.Tensor | None,
    request_ids: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
    attention_fn: Callable | None = None,
    output_dtype: torch.dtype | None = None,
    query_start_loc: torch.Tensor | None = None,
    sequential_query: bool = False,
    round_cache_to_even: bool = False,
    indexer_fn: Callable | None = None,
    index_key_fn: Callable | None = None,
    pool_selector_fn: Callable | None = None,
) -> dict:
    """Evaluate C2A/C1A full, reindex, or reuse with paged cache state."""
    query, window_kv, query_latent = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        sequential_query=sequential_query,
    )
    window_payload, updated_window_scale = _publish_quantized_cache(
        window_cache,
        window_cache_scale,
        window_kv,
        window_slots,
        quantize_mxfp8_cache,
    )
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    compressed_payload = compressed_cache
    updated_compressed_scale = compressed_cache_scale
    quantized_compressed = dequantize_mxfp4_cache(
        compressed_cache, compressed_cache_scale, group_size=16, scale_format="bf16"
    ).to(query.dtype)
    index_payload = index_cache
    updated_index_scale = index_cache_scale
    quantized_index = None
    if index_cache is not None and index_cache_scale is not None:
        quantized_index = dequantize_mxfp4_cache(
            index_cache, index_cache_scale, group_size=32, scale_format="e8m0"
        ).to(query.dtype)
    updated_state = None if state_cache is None else state_cache.clone()
    topk_indices = compressed_indices
    candidates = candidate_mask
    pool_ids = None
    pool_counts = None

    if mode == 'full':
        if compressor_wkv is None or compressor_norm_weight is None or compressed_slots is None:
            raise ValueError("full mode requires compressor weights and compressed slots")
        if compressed_rope_cos is None or compressed_rope_sin is None or quantized_index is None:
            raise ValueError("full mode requires compressed RoPE rows and an index cache")
        if ratio == 1:
            latent = golden_c1a_compressor(x, compressor_wkv, compressor_norm_weight)
            publish_mask = compressed_slots >= 0
        elif ratio == 2:
            if compressor_wgate is None or state_block_table is None or updated_state is None:
                raise ValueError("ratio-2 full mode requires gate weights and recurrent state")
            if position_ids is None or query_start_loc is None or request_ids is None:
                raise ValueError("ratio-2 full mode requires positions, query starts and token-to-request indices")
            latent, publish_mask = golden_c2a_compressor(
                x,
                query_start_loc,
                position_ids,
                request_ids,
                state_block_table,
                updated_state,
                compressor_wkv,
                compressor_wgate,
                compressor_norm_weight,
            )
        else:
            raise ValueError(f"unsupported compression ratio {ratio}")
        latent_tail = rope_interleave(
            latent[..., -compressed_rope_cos.shape[-1] * 2 :], compressed_rope_cos, compressed_rope_sin
        )
        rotated_latent = torch.cat((latent[..., : -compressed_rope_cos.shape[-1] * 2], latent_tail), dim=-1)
        publish_slots = compressed_slots.masked_fill(~publish_mask, -1)
        compressed_payload, updated_compressed_scale = _publish_quantized_cache(
            compressed_cache,
            compressed_cache_scale,
            rotated_latent,
            publish_slots,
            lambda value: quantize_mxfp4_cache(
                value, group_size=16, scale_format="bf16", round_to_even=round_cache_to_even
            ),
        )
        quantized_compressed = dequantize_mxfp4_cache(
            compressed_payload,
            updated_compressed_scale,
            group_size=16,
            scale_format="bf16",
        ).to(query.dtype)
        if index_wk is None or index_norm_weight is None:
            raise ValueError("full mode requires index-key weights")
        if index_key_fn is None:
            raise ValueError("full mode requires an index-key reference")
        keys = index_key_fn(
            latent,
            index_wk,
            index_norm_weight,
            compressed_rope_cos,
            compressed_rope_sin,
        )
        index_payload, updated_index_scale = _publish_quantized_cache(
            index_cache,
            index_cache_scale,
            keys,
            publish_slots,
            lambda value: quantize_mxfp4_cache(
                value, group_size=32, scale_format="e8m0", round_to_even=round_cache_to_even
            ),
        )
        quantized_index = dequantize_mxfp4_cache(
            index_payload,
            updated_index_scale,
            group_size=32,
            scale_format="e8m0",
        ).to(query.dtype)

    if mode in ('full', 'reindex'):
        if index_wq_b is None or index_weights_proj is None:
            raise ValueError("indexing modes require query and score weights")
        if (
            quantized_index is None
            or index_block_table is None
            or request_ids is None
            or compressed_lens is None
        ):
            raise ValueError("indexing modes require cache addressing and causal compressed lengths")
        if indexer_fn is None:
            raise ValueError("indexing modes require an indexer reference")
        index_args = (
            x, query_latent, request_ids, quantized_index, index_block_table, compressed_lens,
            index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
        )
        scores, topk_indices = indexer_fn(*index_args, topk=M.index_topk)
        if ratio == 1 and mode == 'full':
            if pool_selector_fn is None:
                raise ValueError("C1A Full requires a candidate-pool reference")
            pool_width = index_block_table.shape[-1] * 128
            pool_scores = torch.nn.functional.pad(scores, (0, pool_width - scores.shape[-1]), value=-torch.inf)
            pool_ids, pool_counts = pool_selector_fn(pool_scores, compressed_lens)
    if topk_indices is None:
        raise ValueError("reuse mode requires published compressed Top-K indices")

    attend = attention_fn or golden_paged_sparse_attention
    attended = attend(
        query, quantized_window, window_indices, quantized_compressed, topk_indices, attn_sink
    )
    output = golden_o_proj(attended, rope_cos, rope_sin, wo_a, wo_b, output_dtype=output_dtype)
    return {
        'output': output,
        'window_cache': window_payload,
        'window_cache_scale': updated_window_scale,
        'compressed_cache': compressed_payload,
        'compressed_cache_scale': updated_compressed_scale,
        'index_cache': index_payload,
        'index_cache_scale': updated_index_scale,
        'state_cache': updated_state,
        'topk_indices': topk_indices,
        'candidate_mask': candidates,
        'pool_block_ids': pool_ids,
        'pool_block_counts': pool_counts,
    }


prefill_attend_sparse = make_attend_selected_chunk(vector_query=True)


if __name__ == "__main__":
    import argparse

    import torch

    from golden import TensorSpec, ratio_allclose, run
    from .quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache,
        quantize_mxfp4_cache, quantize_mxfp8_cache,
    )

    parser = argparse.ArgumentParser(description="Validate sparse MLA and SWA on A3")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[4])
    rows_help = "single row count (default: 1, 7, 12, 13)"
    parser.add_argument("--rows", type=int, help=rows_help)
    mode_choices = ["packed", "bf16", "both", "c2a", "swa", "swa-packed", "swa-direct", "swa-both", "all"]
    parser.add_argument("--mode", default="all", choices=mode_choices)
    parser.add_argument("--swa-case", default="all", choices=["gather", "plain", "runs", "all"])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--runtime-dir", type=str)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        raise ValueError(f"--tp {args.tp} does not match configured TP_SIZE {TP_SIZE}")
    if (args.mode.startswith("swa") or args.mode == "all") and (args.fixture or args.runtime_dir):
        raise ValueError("--fixture and --runtime-dir require a sparse MLA mode: packed, bf16 or c2a")
    if args.runtime_dir and args.mode == "both":
        raise ValueError("--runtime-dir requires one --mode")
    if args.rows is not None and args.rows < 1:
        raise ValueError("--rows must be positive")
    if args.mode.startswith("swa"):
        _main_results = validate_swa_core(args)
    else:
        row_cases = [1, 7, 12, 13]
        if args.rows is not None:
            row_cases = [args.rows]
        records = []
        for rows in row_cases:
            fixture = torch.load(args.fixture, weights_only=True) if args.fixture else None
            if fixture is None:
                torch.manual_seed(4102026)
                query = (torch.randn(rows, LOCAL_H, HEAD_DIM) * 0.7).bfloat16()
                window = (torch.randn(256, HEAD_DIM) * 0.6).bfloat16()
                compressed = (torch.randn(512, HEAD_DIM) * 0.6).bfloat16()
                window_indices = torch.randint(0, 256, (rows, 128), dtype=torch.int32)
                compressed_indices = torch.randint(0, 512, (rows, INDEX_TOPK), dtype=torch.int32)
                if rows > 1:
                    window_indices[0] = -1
                    compressed_indices[0] = -1
                sink = torch.linspace(-2, 3, LOCAL_H)
                extents = torch.tensor([128, INDEX_TOPK], dtype=torch.int32).repeat(rows, 1)
                wp, ws = quantize_mxfp8_cache(window.reshape(-1, 128, 1, HEAD_DIM))
                cp, cs = quantize_mxfp4_cache(compressed.reshape(-1, 128, 1, HEAD_DIM), 16, "bf16")
                wd = dequantize_mxfp8_cache(wp, ws).bfloat16().reshape(-1, HEAD_DIM)
                cd = dequantize_mxfp4_cache(cp, cs, 16, "bf16").bfloat16().reshape(-1, HEAD_DIM)
            else:
                query = fixture["q"]
                if list(query.shape) != [rows, LOCAL_H, HEAD_DIM]:
                    raise ValueError("Fixture query shape does not match --rows and --tp")
                window, compressed = fixture["w"], fixture["c"]
                window_indices, compressed_indices = fixture["wi"], fixture["ci"]
                sink, extents = fixture["sink"], fixture["extents"]
                wp, ws, cp, cs = fixture["wp"], fixture["ws"], fixture["cp"], fixture["cs"]

            modes = ["packed", "bf16"] if args.mode in ("both", "all") else [args.mode]
            if args.mode == "all":
                modes.append("c2a")
            for mode in modes:
                reference_mode = "packed" if mode == "c2a" else mode
                if fixture is None:
                    wkeys, ckeys = (wd, cd) if reference_mode == "packed" else (window, compressed)
                    expected = torch.empty_like(query, dtype=torch.float32)
                    for row in range(rows):
                        ids = torch.cat([window_indices[row], compressed_indices[row]])
                        wvalues = wkeys[window_indices[row].clamp_min(0).long()]
                        cvalues = ckeys[compressed_indices[row].clamp_min(0).long()]
                        keys = torch.cat([wvalues, cvalues]).float()
                        keys[ids < 0] = 0
                        scores = query[row].float() @ keys.T * SCALE
                        scores[:, ids < 0] = -float("inf")
                        probability = torch.cat([scores, sink[:, None]], dim=1).softmax(dim=1)
                        expected[row] = probability[:, :-1] @ keys
                else:
                    expected = fixture[f"expected_{reference_mode}"]
                if reference_mode == "packed":
                    inputs = dict(query=query.flatten(1), window_indices=window_indices, window=wp.view(torch.uint8), window_scale=ws,
                                  compressed_indices=compressed_indices, compressed=cp, compressed_scale=cs, sink=sink)
                    reference = expected.flatten(1)
                    result_dtype = torch.bfloat16
                    entry = l2_c2a_attention if mode == "c2a" else l2_sparse_attention
                else:
                    inputs = dict(query=query.float(), window=window.float().reshape(-1, 128, HEAD_DIM),
                                  window_indices=window_indices, compressed=compressed.float().reshape(-1, 128, HEAD_DIM),
                                  compressed_indices=compressed_indices, sink=sink, extents=extents)
                    reference = expected
                    result_dtype = torch.float32
                    entry = l2_sparse_attention_bf16
                specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in inputs.items()]
                specs.append(TensorSpec("output", list(reference.shape), result_dtype))

                def golden(tensors):
                    tensors["output"].copy_(reference.bfloat16().to(result_dtype))

                result = run(
                    entry,
                    specs,
                    golden_fn=golden,
                    compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.0)},
                    config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                    compile_only=args.compile_only,
                    runtime_dir=args.runtime_dir,
                )
                records.append(dict(mode=mode, rows=rows, passed=result.passed, work_dir=str(result.work_dir)))
                if not result.passed:
                    raise RuntimeError(result.error)
                status = "COMPILE_ONLY" if args.compile_only else "DEVICE_VALIDATED"
                print(status, mode, f"rows={rows}", result.work_dir)
        if args.mode == "all":
            records.extend(validate_swa_core(args))
        _main_results = records
