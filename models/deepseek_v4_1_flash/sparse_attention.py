# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO sparse MLA with packed-cache dequantization and sink-aware Cube attention."""

from collections.abc import Callable
import math


import torch


from compressor import golden_c1a_compressor
from compressor import golden_c2a_compressor
from qkv_proj_rope import golden_qkv_proj_rope as qkv_proj_rope
from qkv_proj_rope import golden_rope_interleave as rope_interleave
from o_proj import golden_o_proj, golden_grouped_output
from qkv_proj_rope import golden_rope_tail
from rmsnorm import golden_rms_norm
from golden import TensorSpec
from utils import _publish_quantized_cache
from quant import dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight, pack_mx_b_scale, quantize_mxfp4_cache, quantize_mxfp8_cache
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential
from utils import packed_query_lengths, window_metadata

from pathlib import Path

import pypto.language as pl

from config import (
    FLASH as M, CMP_BLOCKS_DYN, MAX_BATCH_PER_DP, ORI_BLOCKS_DYN, T_DYN, TP_SIZE,
)

# Dynamic shape variables.
SELECTED_ROWS = pl.dynamic("SPARSE_SELECTED_ROWS")
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
KEY_TILE = 320 if TP_SIZE == 1 else 128
GATHER_KEY_TILE = 32
TOKEN_TILE = 512
WORKERS = 32
GATHER_WORKERS = 64
SELECTED_GATHER_TILE = 64
CACHE_WORKERS = 32
WINDOW_PUBLISH_WORKER_TILE = 12
SHARED_GATHER_WORKERS = 64
CACHE_MAX_ROWS = 32768
PACKED_QUERY_TILE = 32
PACKED_COHORT_TILE = 24
PACKED_HEAD_TILE = min(32, CUBE_HEADS) if TP_SIZE == 1 else 16
PACKED_KEY_TILE = 128 if TP_SIZE == 1 else 64
PACKED_ATTEND_SLOT_TILE = 1 if TP_SIZE == 1 else 2
PACKED_KEY_PARTS = SELECTED_KEYS // PACKED_KEY_TILE


# SWA tiling.
SWA_HEAD_TILE = min(64, max(16, LOCAL_H)) if TP_SIZE == 1 else 32
# Leave Vector space for the value-preserving query startup before QK.
SWA_SHARED_HEAD_TILE = 32
SWA_GATHER_TILE = 32
SWA_SELECTED_GATHER_TILE = 64 if TP_SIZE == 1 else SWA_GATHER_TILE
SWA_WORKER_TILE = 32
SWA_CROSS_CORE_SLOT_TILE = 1
SWA_RUN_TILE = 32
SWA_RUN_PARTS = SWA_WINDOW // SWA_RUN_TILE
SWA_RUN_DESCRIPTOR_WIDTH = 8


@pl.jit.inline
def prefill_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(num_tokens, name_hint="pub_win", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        _publish_window_row(kv, slots, cache, scales, t)
    return publish_tid


@pl.jit.inline
def prefill_publish_window_batched(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(64, name_hint="prefill_swa_cache_publish", deps=[cache_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, 64):
            _publish_window_row(kv, slots, cache, scales, t)
    return publish_tid


@pl.jit.inline
def _publish_window_row(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    t: pl.Scalar[pl.INT32],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.tensor.view(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.tensor.view(scales, [cache_rows, HEAD_DIM // 32])
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
        payload = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
        payload_row = pl.reshape(payload, [1, HEAD_DIM * 2])
        flat = pl.store(pl.set_validshape(payload_row, 1, HEAD_DIM), [slot, 0], flat)
        scale_row = pl.reshape(scale, [1, HEAD_DIM // 16])
        scale_flat = pl.store(pl.set_validshape(scale_row, 1, HEAD_DIM // 32), [slot, 0], scale_flat)
    return flat, scale_flat


@pl.jit.inline(auto_scope=False)
def publish_window_row(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    workers = pl.min(num_tokens, WINDOW_PUBLISH_WORKER_TILE)
    step = pl.max(1, workers)
    for worker in pl.spmd(workers, name_hint="pub_win"):
        for t in pl.range(worker, num_tokens, step):
            _publish_window_row(kv, slots, cache, scales, t)
    return cache, scales


@pl.jit.inline(auto_scope=False)
def gather_window_batched(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    workers = pl.max(1, pl.min(num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), SWA_WORKER_TILE))
    with pl.spmd(workers, name_hint="swa_gather", allow_early_resolve=True) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), workers):
            token = block // (SWA_WINDOW // SWA_SELECTED_GATHER_TILE)
            column = block % (SWA_WINDOW // SWA_SELECTED_GATHER_TILE) * SWA_SELECTED_GATHER_TILE
            ids = pl.tile.load(indices, [token, column], [1, SWA_SELECTED_GATHER_TILE])
            safe_ids = pl.maximum(ids, 0)
            payload = pl.tile.create([SWA_SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.FP8E4M3FN)
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
            mask = pl.tile.cmps(columns, first_rows_i32, cmp_type=2)
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
            value = pl.reshape(pl.cast(payload, pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 32])
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
    workers = pl.max(1, pl.min(num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), SWA_WORKER_TILE))
    with pl.spmd(workers, name_hint="swa_gather_packed", allow_early_resolve=True) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SWA_WINDOW // SWA_SELECTED_GATHER_TILE), workers):
            token = block // (SWA_WINDOW // SWA_SELECTED_GATHER_TILE)
            column = block % (SWA_WINDOW // SWA_SELECTED_GATHER_TILE) * SWA_SELECTED_GATHER_TILE
            ids = pl.tile.load(indices, [token, column], [1, SWA_SELECTED_GATHER_TILE])
            safe_ids = pl.maximum(ids, 0)
            raw = pl.tile.create([SWA_SELECTED_GATHER_TILE, SWA_PACKED_ROW_BYTES], dtype=pl.UINT8)
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
            mask = pl.tile.cmps(columns, first_rows_i32, cmp_type=2)
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
                raw = pl.tile.gather_row(
                    raw, flat_bytes, [0, 0], [first_wide, 0], [SWA_SELECTED_GATHER_TILE, SWA_PACKED_ROW_BYTES],
                    valid_shape=[first_rows, SWA_PACKED_ROW_BYTES],
                )
                if first_rows < SWA_SELECTED_GATHER_TILE:
                    raw = pl.tile.gather_row(
                        raw, flat_bytes, [first_rows, 0], [second_wide, 0],
                        [SWA_SELECTED_GATHER_TILE, SWA_PACKED_ROW_BYTES],
                        valid_shape=[SWA_SELECTED_GATHER_TILE - first_rows, SWA_PACKED_ROW_BYTES],
                    )
            else:
                for lane in pl.range(SWA_SELECTED_GATHER_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    raw = pl.tile.gather_row(raw, flat_bytes, [lane, 0], [physical, 0], [1, SWA_PACKED_ROW_BYTES])
            # Reinterpret the dense packed tile before slicing to preserve its physical pitch.
            signed_bytes = pl.tile.reinterpret_view(raw, pl.INT8)
            payload_all = pl.tile.reinterpret_view(
                signed_bytes, pl.FP8E4M3FN, shape=[SWA_SELECTED_GATHER_TILE, SWA_PACKED_ROW_BYTES],
            )
            payload = pl.tile.slice(payload_all, [SWA_SELECTED_GATHER_TILE, HEAD_DIM], [0, 0])
            factor_bits = pl.tile.reinterpret_view(
                raw, pl.INT16, shape=[SWA_SELECTED_GATHER_TILE, SWA_PACKED_ROW_BYTES // 2],
            )
            factors_all = pl.tile.reinterpret_view(factor_bits, pl.BF16)
            factors = pl.tile.slice(factors_all, [SWA_SELECTED_GATHER_TILE, SWA_GROUPS], [0, HEAD_DIM // 2])
            value = pl.reshape(pl.cast(payload, pl.FP32), [SWA_SELECTED_GATHER_TILE * SWA_GROUPS, 32])
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
def attend_window_tiled(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, SWA_WINDOW, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(query, 0)
    qflat_rows = tokens * LOCAL_H
    qflat = pl.reshape(query, [qflat_rows, HEAD_DIM])
    kflat_rows = tokens * SWA_WINDOW
    kflat = pl.reshape(selected, [kflat_rows, HEAD_DIM])
    oflat_rows = tokens * LOCAL_H
    oflat = pl.reshape(output, [oflat_rows, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    head_blocks = (LOCAL_H + SWA_HEAD_TILE - 1) // SWA_HEAD_TILE
    workers = pl.max(1, pl.min(num_tokens * head_blocks, SWA_WORKER_TILE))
    with pl.spmd(
        workers, name_hint="swa_attn", deps=[ready],
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ) as attention_done:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * head_blocks, workers):
            token = block // head_blocks
            head = block % head_blocks * SWA_HEAD_TILE
            # Full-window Mat KV for QK and PV.
            kv_mat = pl.tile.load(
                kflat, [token * SWA_WINDOW, 0], [SWA_WINDOW, HEAD_DIM], target_memory=pl.MemorySpace.Mat,
            )
            kv_transposed = pl.tile.transpose_view(kv_mat)
            ids = pl.tile.load(indices, [token, 0], [1, SWA_WINDOW])
            index_values = pl.cast(ids, pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(index_values, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            q0 = token * LOCAL_H + head
            valid_heads = pl.min(SWA_HEAD_TILE, LOCAL_H - head)
            q = pl.tile.load(
                qflat, [q0, 0], [SWA_HEAD_TILE, HEAD_DIM],
                valid_shape=[valid_heads, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
            )
            # Require Vector startup before local QK Cube transfers (hw-native-sys/pypto#2829).
            q = pl.mul(q, 1.0)
            q_mat = pl.tile.move(q, pl.MemorySpace.Mat)
            qk = pl.tile.matmul(q_mat, kv_transposed)
            scores_move = pl.tile.move(
                qk, pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
            )
            scores = pl.mul(scores_move, SCALE)
            scores = pl.col_expand_add(scores, bias)
            sinks = pl.tile.load(
                sink_row, [0, head], [1, SWA_HEAD_TILE],
                valid_shape=[1, valid_heads], target_memory=pl.MemorySpace.Vec,
            )
            reduction_tmp = pl.tile.create([SWA_HEAD_TILE, SWA_WINDOW], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            # Sink-aware full-window softmax.
            maximum_max = pl.row_max(scores, tmp_tile=reduction_tmp)
            maximum_shaped = pl.reshape(maximum_max, [1, SWA_HEAD_TILE])
            maximum = pl.maximum(maximum_shaped, sinks)
            probabilities_shaped = pl.reshape(maximum, [SWA_HEAD_TILE, 1])
            probabilities_centered = pl.row_expand_sub(scores, probabilities_shaped)
            probabilities_exp = pl.exp(probabilities_centered)
            probabilities = pl.col_expand_mul(probabilities_exp, valid)
            denominator_sum = pl.row_sum(probabilities, tmp_tile=reduction_tmp)
            denominator_shaped = pl.reshape(denominator_sum, [1, SWA_HEAD_TILE])
            denominator_sub = pl.sub(sinks, maximum)
            denominator_exp = pl.exp(denominator_sub)
            denominator = pl.add(denominator_shaped, denominator_exp)
            weights = pl.tile.move(pl.cast(probabilities, pl.BF16, mode="rint"), pl.MemorySpace.Mat)
            denominator = pl.maximum(denominator, 1.0)
            inverse_full = pl.tile.full([1, SWA_HEAD_TILE], dtype=pl.FP32, value=1.0)
            inverse_valid = pl.set_validshape(inverse_full, 1, valid_heads)
            inverse = pl.div(inverse_valid, denominator, high_precision=True)
            # Bound the PV accumulator and cross-core transfer to 256 channels.
            # The source slice retains the Mat tile's layout and row pitch.
            for d0 in pl.range(0, HEAD_DIM, 256):
                kv_part = pl.tile.slice(kv_mat, [SWA_WINDOW, 256], [0, d0])
                pv = pl.tile.matmul(weights, kv_part)
                numerator = pl.tile.move(
                    pv, pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                )
                result = pl.row_expand_mul(numerator, pl.reshape(inverse, [SWA_HEAD_TILE, 1]))
                value_bf16 = pl.cast(result, pl.BF16, mode='rint')
                value_valid = pl.set_validshape(value_bf16, valid_heads, 256)
                pl.tile.store(value_valid, [q0, d0], oflat)
    return attention_done


@pl.jit.inline(auto_scope=False)
def decode_shared_cache(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    workers = pl.max(1, pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE))
    with pl.spmd(workers, name_hint="swa_decode_cache") as ready:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            value_full = pl.tile.full([32, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.tile.store(value_full, [0, 0], zero_row)
        for block in pl.range(worker, rows // SWA_GATHER_TILE, workers):
            row = block * SWA_GATHER_TILE
            payload = pl.tile.load(payload_flat, [row, 0], [SWA_GATHER_TILE, HEAD_DIM])
            factors = pl.tile.load(scale_flat, [row, 0], [SWA_GATHER_TILE, SWA_GROUPS])
            value = pl.reshape(pl.cast(payload, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
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
    workers = pl.max(1, pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE))
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
                payload_reinterpret_view, pl.FP8E4M3FN,
                shape=[SWA_GATHER_TILE, HEAD_DIM],
            )
            factors_reinterpret_view = pl.tile.reinterpret_view(
                factor_bytes, pl.INT16,
                shape=[SWA_GATHER_TILE, SWA_GROUPS],
            )
            factors = pl.tile.reinterpret_view(factors_reinterpret_view, pl.BF16)
            value = pl.reshape(pl.cast(payload, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
            multiplier = pl.reshape(pl.cast(factors, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, multiplier)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_flat)
    return ready


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
    workers = pl.max(1, pl.min(num_tokens * SWA_RUN_PARTS, SWA_WORKER_TILE * 2))
    with pl.spmd(workers, name_hint="swa_plan_runs") as ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * SWA_RUN_PARTS, workers):
            token = block // SWA_RUN_PARTS
            part = block % SWA_RUN_PARTS
            ids = pl.tile.load(indices, [token, part * SWA_RUN_TILE], [1, SWA_RUN_TILE])
            first = pl.tile.read(ids, [0, 0])
            expected_ids = pl.tile.ci(first, [1, SWA_RUN_TILE], dtype=pl.INT32)
            # Subtract in INT32 before casting: FP32 rounding cannot turn a
            # nonzero integer difference into zero, including large row IDs.
            delta = pl.abs(pl.cast(pl.sub(ids, expected_ids), pl.FP32))
            check_data = pl.tile.full([8, SWA_RUN_TILE], dtype=pl.FP32, value=0.0)
            check_data = pl.tile.assemble(check_data, delta, [0, 0])
            check_tmp = pl.tile.create([8, SWA_RUN_TILE], dtype=pl.FP32)
            error = pl.tile.read(pl.row_max(check_data, tmp_tile=check_tmp), [0, 0])
            first_wide = pl.cast(first, pl.INDEX)
            contiguous = (first >= 0) and (first_wide + SWA_RUN_TILE <= cache_rows) and (error == 0.0)
            index_data = pl.tile.full([8, SWA_RUN_TILE], dtype=pl.FP32, value=0.0)
            index_data = pl.tile.assemble(index_data, pl.cast(ids, pl.FP32), [0, 0])
            largest_index = pl.tile.read(pl.row_max(index_data, tmp_tile=check_tmp), [0, 0])
            if largest_index < 0.0:
                # All invalid rows use a separate, initialized GM zero block.
                # No cache row zero is referenced, so poisoned cache cannot leak.
                value_full = pl.tile.full([1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32, value=-2)
                pl.tile.store(value_full, [block, 0], descriptor_flat)
            elif contiguous:
                # Runtime fill values are unsupported by this pinned tile.full.
                # The difference of two CI tiles broadcasts the INT32 start.
                descriptor_ci = pl.tile.ci(first, [1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
                descriptor_ci_2 = pl.tile.ci(0, [1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
                descriptor = pl.sub(descriptor_ci, descriptor_ci_2)
                pl.tile.store(descriptor, [block, 0], descriptor_flat)
            else:
                value_full_2 = pl.tile.full([1, SWA_RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32, value=-1)
                pl.tile.store(value_full_2, [block, 0], descriptor_flat)
    return ready


@pl.jit.inline(auto_scope=False)
def decode_shared_cache_after_publish(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    decoded_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.BF16],
    zero_row: pl.Tensor[[32, HEAD_DIM], pl.BF16],
    publish_ready: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(cache, 0) * 128
    payload_flat = pl.reshape(cache, [rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [rows, SWA_GROUPS])
    decoded_flat = pl.reshape(decoded_cache, [rows, HEAD_DIM])
    # Physical pages are a multiple of 128 rows; GATHER_TILE=32 has no tail.
    # Decode each physical row once with contiguous transfers.
    workers = pl.max(1, pl.min(rows // SWA_GATHER_TILE, SWA_WORKER_TILE))
    with pl.spmd(workers, name_hint="swa_decode_cache", deps=[publish_ready]) as ready:
        worker = pl.tile.get_block_idx()
        if worker == 0:
            value_full = pl.tile.full([32, HEAD_DIM], dtype=pl.BF16, value=0.0)
            pl.tile.store(value_full, [0, 0], zero_row)
        for block in pl.range(worker, rows // SWA_GATHER_TILE, workers):
            row = block * SWA_GATHER_TILE
            payload = pl.tile.load(payload_flat, [row, 0], [SWA_GATHER_TILE, HEAD_DIM])
            factors = pl.tile.load(scale_flat, [row, 0], [SWA_GATHER_TILE, SWA_GROUPS])
            value = pl.reshape(pl.cast(payload, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 32])
            multiplier = pl.reshape(pl.cast(factors, pl.FP32), [SWA_GATHER_TILE * SWA_GROUPS, 1])
            decoded_scaled = pl.row_expand_mul(value, multiplier)
            decoded_bf16 = pl.cast(decoded_scaled, pl.BF16, mode='rint')
            decoded = pl.reshape(decoded_bf16, [SWA_GATHER_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_flat)
    return ready


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
    tokens = pl.tensor.dim(query, 0)
    rows = pl.tensor.dim(decoded_cache, 0) * 128
    qflat_rows = tokens * LOCAL_H
    qflat = pl.reshape(query, [qflat_rows, HEAD_DIM])
    decoded_flat = pl.reshape(decoded_cache, [rows, HEAD_DIM])
    descriptor_rows = tokens * SWA_RUN_PARTS
    descriptor_flat = pl.reshape(descriptors, [descriptor_rows, SWA_RUN_DESCRIPTOR_WIDTH])
    oflat_rows = tokens * LOCAL_H
    oflat = pl.reshape(output, [oflat_rows, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    workers = pl.max(1, pl.min(num_tokens, SWA_WORKER_TILE))
    with pl.spmd(
        workers, name_hint="swa_attn_shared", deps=[ready, index_ready],
        optimizations=[pl.cross_core_slot(slot_num=SWA_CROSS_CORE_SLOT_TILE)],
    ) as attention_done:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            # The on-chip KV buffer lives in Mat/L1 throughout. GM scalar
            # reads feed the AIC gather offsets; no AIV tile.read value crosses
            # cores. Every lane is written, including explicit GM zero rows.
            kv_mat = pl.tile.create([SWA_WINDOW, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
            # Four GM descriptors bridge AIV index analysis to AIC gather.
            # Exact physical runs use one 32-row DMA; other chunks preserve
            # arbitrary order, duplicates, masks and the separate GM zero row.
            for part in pl.range(SWA_RUN_PARTS):
                column = part * SWA_RUN_TILE
                run_start = pl.read(descriptor_flat, [token * SWA_RUN_PARTS + part, 0])
                if run_start == -2:
                    kv_mat = pl.tile.gather_row(kv_mat, zero_row, [column, 0], [0, 0], [SWA_RUN_TILE, HEAD_DIM])
                elif run_start >= 0:
                    physical = pl.cast(run_start, pl.INDEX)
                    kv_mat = pl.tile.gather_row(
                        kv_mat, decoded_flat, [column, 0], [physical, 0], [SWA_RUN_TILE, HEAD_DIM],
                    )
                else:
                    for lane in pl.range(SWA_RUN_TILE):
                        row_id = pl.read(indices, [token, column + lane])
                        if row_id >= 0:
                            physical = pl.cast(row_id, pl.INDEX)
                            kv_mat = pl.tile.gather_row(
                                kv_mat, decoded_flat, [column + lane, 0], [physical, 0], [1, HEAD_DIM],
                            )
                        else:
                            kv_mat = pl.tile.gather_row(kv_mat, zero_row, [column + lane, 0], [0, 0], [1, HEAD_DIM])
            kv_transposed = pl.tile.transpose_view(kv_mat)
            ids = pl.tile.load(indices, [token, 0], [1, SWA_WINDOW], target_memory=pl.MemorySpace.Vec)
            index_values = pl.cast(ids, pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(index_values, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            # Gather is outside the head loop: all heads reuse this same tile.
            # Each query has one worker, so the tile stays private and on chip.
            for head in pl.range(0, LOCAL_H, SWA_SHARED_HEAD_TILE):
                q0 = token * LOCAL_H + head
                valid_heads = pl.min(SWA_SHARED_HEAD_TILE, LOCAL_H - head)
                q = pl.tile.load(
                    qflat, [q0, 0], [SWA_SHARED_HEAD_TILE, HEAD_DIM],
                    valid_shape=[valid_heads, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                )
                # Require Vector startup before local QK Cube transfers (hw-native-sys/pypto#2829).
                q = pl.cast(pl.cast(q, pl.FP32), pl.BF16, mode="rint")
                q_mat = pl.tile.move(q, pl.MemorySpace.Mat)
                qk = pl.tile.matmul(q_mat, kv_transposed)
                scores_move = pl.tile.move(
                    qk, pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                )
                scores = pl.mul(scores_move, SCALE)
                scores = pl.col_expand_add(scores, bias)
                sinks = pl.tile.load(
                    sink_row, [0, head], [1, SWA_SHARED_HEAD_TILE],
                    valid_shape=[1, valid_heads], target_memory=pl.MemorySpace.Vec,
                )
                reduction_tmp = pl.tile.create(
                    [SWA_SHARED_HEAD_TILE, SWA_WINDOW],
                    dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                )
                maximum_max = pl.row_max(scores, tmp_tile=reduction_tmp)
                maximum_shaped = pl.reshape(maximum_max, [1, SWA_SHARED_HEAD_TILE])
                maximum = pl.maximum(maximum_shaped, sinks)
                probabilities_shaped = pl.reshape(maximum, [SWA_SHARED_HEAD_TILE, 1])
                probabilities_centered = pl.row_expand_sub(scores, probabilities_shaped)
                probabilities_exp = pl.exp(probabilities_centered)
                probabilities = pl.col_expand_mul(probabilities_exp, valid)
                denominator_sum = pl.row_sum(probabilities, tmp_tile=reduction_tmp)
                denominator_shaped = pl.reshape(denominator_sum, [1, SWA_SHARED_HEAD_TILE])
                denominator_sub = pl.sub(sinks, maximum)
                denominator_exp = pl.exp(denominator_sub)
                denominator = pl.add(denominator_shaped, denominator_exp)
                weights = pl.tile.move(pl.cast(probabilities, pl.BF16, mode="rint"), pl.MemorySpace.Mat)
                denominator = pl.maximum(denominator, 1.0)
                inverse_full = pl.tile.full([1, SWA_SHARED_HEAD_TILE], dtype=pl.FP32, value=1.0)
                inverse_valid = pl.set_validshape(inverse_full, 1, valid_heads)
                inverse = pl.div(inverse_valid, denominator, high_precision=True)
                # Bound the PV accumulator and cross-core transfer to 256 channels.
                # The source slice retains the Mat tile's layout and row pitch.
                for d0 in pl.range(0, HEAD_DIM, 256):
                    kv_part = pl.tile.slice(kv_mat, [SWA_WINDOW, 256], [0, d0])
                    pv = pl.tile.matmul(weights, kv_part)
                    numerator = pl.tile.move(
                        pv, pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                    )
                    result = pl.row_expand_mul(numerator, pl.reshape(inverse, [SWA_SHARED_HEAD_TILE, 1]))
                    value_bf16 = pl.cast(result, pl.BF16, mode='rint')
                    value_valid = pl.set_validshape(value_bf16, valid_heads, 256)
                    pl.tile.store(value_valid, [q0, d0], oflat)
    return attention_done


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
    tokens = pl.tensor.dim(query, 0)
    rows = pl.tensor.dim(decoded_cache, 0) * 128
    qflat_rows = tokens * LOCAL_H
    qflat = pl.reshape(query, [qflat_rows, HEAD_DIM])
    decoded_flat = pl.reshape(decoded_cache, [rows, HEAD_DIM])
    oflat_rows = tokens * LOCAL_H
    oflat = pl.reshape(output, [oflat_rows, HEAD_DIM])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    workers = pl.max(1, pl.min(num_tokens, SWA_WORKER_TILE))
    with pl.spmd(
        workers, name_hint="swa_attn_shared", deps=[ready],
        optimizations=[pl.cross_core_slot(slot_num=SWA_CROSS_CORE_SLOT_TILE)],
    ) as attention_done:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            # The on-chip KV buffer lives in Mat/L1 throughout. GM scalar
            # reads feed the AIC gather offsets; no AIV tile.read value crosses
            # cores. Every lane is written, including explicit GM zero rows.
            kv_mat = pl.tile.create([SWA_WINDOW, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
            for lane in pl.range(SWA_WINDOW):
                row_id = pl.read(indices, [token, lane])
                if row_id >= 0:
                    physical = pl.cast(row_id, pl.INDEX)
                    kv_mat = pl.tile.gather_row(kv_mat, decoded_flat, [lane, 0], [physical, 0], [1, HEAD_DIM])
                else:
                    kv_mat = pl.tile.gather_row(kv_mat, zero_row, [lane, 0], [0, 0], [1, HEAD_DIM])
            kv_transposed = pl.tile.transpose_view(kv_mat)
            ids = pl.tile.load(indices, [token, 0], [1, SWA_WINDOW], target_memory=pl.MemorySpace.Vec)
            index_values = pl.cast(ids, pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(index_values, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            # Gather is outside the head loop: all heads reuse this same tile.
            # Each query has one worker, so the tile stays private and on chip.
            for head in pl.range(0, LOCAL_H, SWA_SHARED_HEAD_TILE):
                q0 = token * LOCAL_H + head
                valid_heads = pl.min(SWA_SHARED_HEAD_TILE, LOCAL_H - head)
                q = pl.tile.load(
                    qflat, [q0, 0], [SWA_SHARED_HEAD_TILE, HEAD_DIM],
                    valid_shape=[valid_heads, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                )
                # Require Vector startup before local QK Cube transfers (hw-native-sys/pypto#2829).
                q = pl.cast(pl.cast(q, pl.FP32), pl.BF16, mode="rint")
                q_mat = pl.tile.move(q, pl.MemorySpace.Mat)
                qk = pl.tile.matmul(q_mat, kv_transposed)
                scores_move = pl.tile.move(
                    qk, pl.MemorySpace.Vec,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                )
                scores = pl.mul(scores_move, SCALE)
                scores = pl.col_expand_add(scores, bias)
                sinks = pl.tile.load(
                    sink_row, [0, head], [1, SWA_SHARED_HEAD_TILE],
                    valid_shape=[1, valid_heads], target_memory=pl.MemorySpace.Vec,
                )
                reduction_tmp = pl.tile.create(
                    [SWA_SHARED_HEAD_TILE, SWA_WINDOW],
                    dtype=pl.FP32, target_memory=pl.MemorySpace.Vec,
                )
                maximum_max = pl.row_max(scores, tmp_tile=reduction_tmp)
                maximum_shaped = pl.reshape(maximum_max, [1, SWA_SHARED_HEAD_TILE])
                maximum = pl.maximum(maximum_shaped, sinks)
                probabilities_shaped = pl.reshape(maximum, [SWA_SHARED_HEAD_TILE, 1])
                probabilities_centered = pl.row_expand_sub(scores, probabilities_shaped)
                probabilities_exp = pl.exp(probabilities_centered)
                probabilities = pl.col_expand_mul(probabilities_exp, valid)
                denominator_sum = pl.row_sum(probabilities, tmp_tile=reduction_tmp)
                denominator_shaped = pl.reshape(denominator_sum, [1, SWA_SHARED_HEAD_TILE])
                denominator_sub = pl.sub(sinks, maximum)
                denominator_exp = pl.exp(denominator_sub)
                denominator = pl.add(denominator_shaped, denominator_exp)
                weights = pl.tile.move(pl.cast(probabilities, pl.BF16, mode="rint"), pl.MemorySpace.Mat)
                denominator = pl.maximum(denominator, 1.0)
                inverse_full = pl.tile.full([1, SWA_SHARED_HEAD_TILE], dtype=pl.FP32, value=1.0)
                inverse_valid = pl.set_validshape(inverse_full, 1, valid_heads)
                inverse = pl.div(inverse_valid, denominator, high_precision=True)
                # Bound the PV accumulator and cross-core transfer to 256 channels.
                # The source slice retains the Mat tile's layout and row pitch.
                for d0 in pl.range(0, HEAD_DIM, 256):
                    kv_part = pl.tile.slice(kv_mat, [SWA_WINDOW, 256], [0, d0])
                    pv = pl.tile.matmul(weights, kv_part)
                    numerator = pl.tile.move(
                        pv, pl.MemorySpace.Vec,
                        blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box,
                    )
                    result = pl.row_expand_mul(numerator, pl.reshape(inverse, [SWA_SHARED_HEAD_TILE, 1]))
                    value_bf16 = pl.cast(result, pl.BF16, mode='rint')
                    value_valid = pl.set_validshape(value_bf16, valid_heads, 256)
                    pl.tile.store(value_valid, [q0, d0], oflat)
    return attention_done


@pl.jit.inline(auto_scope=False)
def swa_attention_partial(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
        attention_done = attend_window_tiled(query, selected, indices, sink, output, num_tokens, gathered)
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
def swa_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
        attention_done = attend_window_tiled(query, selected, indices, sink, output, num_tokens, gathered)
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
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, SWA_GROUPS], pl.BF16],
    indices: pl.Tensor[[T_DYN, SWA_WINDOW], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output = swa_attention(query, cache, scales, indices, sink, output, num_tokens)
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
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)


def swa_official_reference(tensors, *, shared_cache=False, key_tile=128):
    """Evaluate FP8 projections, tiled BF16 attention weights, and FP32 output."""
    from quant import unpack_fp8_nz
    t = tensors
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups = t["wo_a"].shape[0]
    group_in = t["wo_a"].shape[-2]

    qa_weight = unpack_fp8_nz(t["wq_a"])
    qb_weight = unpack_fp8_nz(t["wq_b"])
    kv_weight = unpack_fp8_nz(t["wkv"])
    projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(t["x"], qa_weight, t["wq_a_scale"])
    qr = golden_rms_norm(projected, t["q_norm_weight"])
    q = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, qb_weight, t["wq_b_scale"])
    q = q.unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    projected_kv = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(t["x"], kv_weight, t["wkv_scale"])
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
    latent = golden_grouped_output(grouped, t["wo_a"], t["wo_a_scale"])
    output = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(latent.flatten(1), unpack_fp8_nz(t["wo_b"]), t["wo_b_scale"]).float()
    return output, cache, cache_scale


def make_swa_inputs(batch=MAX_BATCH_PER_DP, seed=17, position=127):
    """Distinct request pages, ragged visibility, nontrivial scales and rotary phases."""
    from quant import pack_fp8_nz
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
    assert slots.unique().numel() == tokens
    # Verify physical metadata independently, without relying on the helper's mask.
    for row, (request, position) in enumerate(zip(ids.tolist(), positions.tolist())):
        history = list(range(max(0, position - 127), position + 1))
        expected = [int(table[request, p // 128]) * 128 + p % 128 for p in history]
        assert indices[row, :len(expected)].tolist() == expected
        assert (indices[row, len(expected):] == -1).all()
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


GOLDEN_SELECTED_KEY_TILE = 64
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
def gather_packed(
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    workers = pl.max(1, pl.min(num_tokens * (SELECTED_KEYS // GATHER_KEY_TILE), max_workers))
    with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SELECTED_KEYS // GATHER_KEY_TILE), workers):
            token = block // (SELECTED_KEYS // GATHER_KEY_TILE)
            column = block % (SELECTED_KEYS // GATHER_KEY_TILE) * GATHER_KEY_TILE
            if column < 128:
                ids = pl.tile.load(wi, [token, column], [1, GATHER_KEY_TILE])
                safe_ids = pl.maximum(ids, 0)
                w_payload = pl.tile.create([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.FP8E4M3FN)
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
                w_scales = pl.reinterpret_view(w_valid_bits, pl.FP32)
                w_scale_column = pl.reshape(w_scales, [GATHER_KEY_TILE * (HEAD_DIM // 32), 1])
                w_values = pl.cast(w_payload, pl.FP32)
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
                c_fp4 = pl.reinterpret_view(c_payload, pl.FP4E2M1X2)
                c_values = pl.cast(c_fp4, pl.BF16)
                c_grouped = pl.reshape(c_values, [GATHER_KEY_TILE * (HEAD_DIM // 16), 16])
                # BF16 scales can exceed the FP16 range; multiply in FP32.
                c_scales = pl.cast(c_factors, pl.FP32)
                c_scale_column = pl.reshape(c_scales, [GATHER_KEY_TILE * (HEAD_DIM // 16), 1])
                c_scaled = pl.row_expand_mul(pl.cast(c_grouped, pl.FP32), c_scale_column)
                c_decoded = pl.cast(c_scaled, pl.BF16, mode="rint")
                decoded = pl.reshape(c_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            pl.tile.store(decoded, [token * SELECTED_KEYS + column, 0], dst)
            zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for lane in pl.range(GATHER_KEY_TILE):
                if pl.tile.read(ids, [0, lane]) < 0:
                    pl.tile.store(zero, [token * SELECTED_KEYS + column + lane, 0], dst)
            pl.tile.store(ids, [token, column], indices)
    return selected, indices, gathered


@pl.jit.inline(auto_scope=False)
def gather_packed_chunk(
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
        name_hint="c2a_gather",
    )
    return gathered


def make_attend_selected_chunk(*, vector_query=False, selected_rows=PACKED_QUERY_TILE, max_workers=32):
    """Build chunked sink attention with an optional Vector query startup."""
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
        """Attend one chunk over the window and compressed rows under a single sink."""
        tokens = pl.tensor.dim(query, 0)
        head_rows = tokens * LOCAL_H
        query_flat = pl.reshape(query, [head_rows, HEAD_DIM])
        output_flat = pl.reshape(output, [head_rows, HEAD_DIM])
        key_flat = pl.reshape(selected, [selected_rows * SELECTED_KEYS, HEAD_DIM])
        attend_tasks = active * (LOCAL_H // PACKED_HEAD_TILE)
        workers = attend_tasks
        if TP_SIZE == 1:
            workers = pl.min(attend_tasks, max_workers)
        if sync_start:
            workers = pl.min(workers, pl.system.available_cluster_count())
        # Direct-Q Cube transfers require ownership of the paired Vector UB.
        with pl.spmd(
            workers, name_hint=name_hint, deps=[gather_ready],
            sync_start=sync_start,
            optimizations=[pl.cross_core_slot(slot_num=PACKED_ATTEND_SLOT_TILE)],
        ) as attend_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, attend_tasks, workers):
                r = block // (LOCAL_H // PACKED_HEAD_TILE)
                h = block % (LOCAL_H // PACKED_HEAD_TILE) * PACKED_HEAD_TILE
                q0 = (chunk_start + r) * LOCAL_H + h
                maximum = pl.full([1, PACKED_HEAD_TILE], dtype=pl.FP32, value=-1e30)
                denominator = pl.full([1, PACKED_HEAD_TILE], dtype=pl.FP32, value=0.0)
                numerator = pl.full([PACKED_HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                # Reuse the query tile across all sparse KV parts.
                q = query_flat[q0 : q0 + PACKED_HEAD_TILE, :]
                if vector_query:
                    # Vector startup before the first score push.
                    q = pl.cast(pl.cast(q, pl.FP32), pl.BF16, mode="rint")
                for part in pl.range(PACKED_KEY_PARTS):
                    k0 = r * SELECTED_KEYS + part * PACKED_KEY_TILE
                    kv = key_flat[k0 : k0 + PACKED_KEY_TILE, :]
                    scores = pl.mul(pl.matmul(q, kv, b_trans=True), SCALE)
                    index_start = part * PACKED_KEY_TILE
                    idx = combined[r : r + 1, index_start : index_start + PACKED_KEY_TILE]
                    idx = pl.cast(idx, pl.FP32)
                    valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
                    scores = pl.col_expand_add(scores, pl.mul(pl.sub(valid, 1.0), 1e30))
                    next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, PACKED_HEAD_TILE]))
                    correction = pl.exp(pl.sub(maximum, next_max))
                    next_max_column = pl.reshape(next_max, [PACKED_HEAD_TILE, 1])
                    shifted = pl.row_expand_sub(scores, next_max_column)
                    probabilities = pl.col_expand_mul(pl.exp(shifted), valid)
                    previous_denominator = pl.mul(denominator, correction)
                    part_denominator = pl.reshape(pl.row_sum(probabilities), [1, PACKED_HEAD_TILE])
                    denominator = pl.add(previous_denominator, part_denominator)
                    weighted = pl.matmul(pl.cast(probabilities, pl.BF16, mode="rint"), kv)
                    correction_column = pl.reshape(correction, [PACKED_HEAD_TILE, 1])
                    previous_numerator = pl.row_expand_mul(numerator, correction_column)
                    numerator = pl.add(previous_numerator, weighted)
                    maximum = next_max
                sinks = pl.reshape(sink[h : h + PACKED_HEAD_TILE], [1, PACKED_HEAD_TILE])
                final_max = pl.maximum(maximum, sinks)
                correction = pl.exp(pl.sub(maximum, final_max))
                denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
                multiplier = pl.reshape(pl.div(correction, denominator), [PACKED_HEAD_TILE, 1])
                result = pl.row_expand_mul(numerator, multiplier)
                output_flat[q0 : q0 + PACKED_HEAD_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
        return attend_tid

    return attend_selected_chunk


@pl.jit.inline(auto_scope=False)
def _attend_cube(
    query: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[CUBE_HEADS], pl.FP32],
    output: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * CUBE_HEADS
    cache_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), WORKERS))
    with pl.spmd(workers, name_hint="sparse_online_cube", deps=[selected_ready], optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), workers):
            token = block // ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE)
            head = block % ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE) * HEAD_TILE
            q0 = token * CUBE_HEADS + head
            q = qflat[q0:q0 + HEAD_TILE, :]
            if TP_SIZE != 1:
                # Retain the Vector startup workaround outside the validated TP1 path.
                q = pl.cast(pl.cast(q, pl.FP32), pl.BF16, mode="rint")
            maximum = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=-1e30)
            denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
            numerator = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for part in pl.range(SELECTED_KEYS // KEY_TILE):
                k0 = token * SELECTED_KEYS + part * KEY_TILE
                kv = kflat[k0:k0 + KEY_TILE, :]
                qk = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
                scores = pl.mul(qk, SCALE)
                idx0 = part * KEY_TILE
                idx = pl.cast(indices[token:token + 1, idx0:idx0 + KEY_TILE], pl.FP32)
                valid = pl.add(idx, 1.0)
                valid = pl.maximum(valid, 0.0)
                valid = pl.minimum(valid, 1.0)
                penalty = pl.mul(pl.sub(valid, 1.0), 1e30)
                scores = pl.col_expand_add(scores, penalty)
                score_max = pl.reshape(pl.row_max(scores), [1, HEAD_TILE])
                next_max = pl.maximum(maximum, score_max)
                correction = pl.exp(pl.sub(maximum, next_max))
                max_col = pl.reshape(next_max, [HEAD_TILE, 1])
                shifted = pl.row_expand_sub(scores, max_col)
                probabilities = pl.exp(shifted)
                probabilities = pl.col_expand_mul(probabilities, valid)
                previous_denominator = pl.mul(denominator, correction)
                part_denominator = pl.reshape(pl.row_sum(probabilities), [1, HEAD_TILE])
                denominator = pl.add(previous_denominator, part_denominator)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv, out_dtype=pl.FP32)
                correction_col = pl.reshape(correction, [HEAD_TILE, 1])
                previous_numerator = pl.row_expand_mul(numerator, correction_col)
                numerator = pl.add(previous_numerator, weighted)
                maximum = next_max
            sinks = pl.reshape(sink[head:head + HEAD_TILE], [1, HEAD_TILE])
            final_max = pl.maximum(maximum, sinks)
            correction = pl.exp(pl.sub(maximum, final_max))
            corrected_denominator = pl.mul(denominator, correction)
            sink_weight = pl.exp(pl.sub(sinks, final_max))
            denominator = pl.add(corrected_denominator, sink_weight)
            multiplier = pl.reshape(pl.div(correction, denominator), [HEAD_TILE, 1])
            result = pl.row_expand_mul(numerator, multiplier)
            result = pl.cast(result, pl.BF16, mode="rint")
            oflat[q0:q0 + HEAD_TILE, :] = result
    return attended


@pl.jit.inline(auto_scope=False)
def _attend_padded(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
):
    # Cube uses at least 16 rows. TP8 retains only its eight real heads.
    rows = pl.tensor.dim(query, 0)
    qflat = pl.reshape(query, [rows * LOCAL_H, HEAD_DIM])
    padded_query = pl.create_tensor([rows, CUBE_HEADS * HEAD_DIM], dtype=pl.BF16)
    pqflat = pl.reshape(padded_query, [rows * CUBE_HEADS, HEAD_DIM])
    padded_sink = pl.create_tensor([CUBE_HEADS], dtype=pl.FP32)
    padded_sink_row = pl.reshape(padded_sink, [1, CUBE_HEADS])
    sink_row = pl.reshape(sink, [1, LOCAL_H])
    workers = pl.max(1, pl.min(num_tokens, WORKERS))
    with pl.spmd(workers, name_hint="sparse_pad_heads"):
        worker = pl.tile.get_block_idx()
        if worker == 0:
            sinks = pl.full([1, CUBE_HEADS], dtype=pl.FP32, value=0.0)
            sinks[:, 0:LOCAL_H] = sink_row[:, :]
            padded_sink_row[:, :] = sinks
        for token in pl.range(worker, num_tokens, workers):
            q0 = token * LOCAL_H
            pq0 = token * CUBE_HEADS
            q = pl.full([CUBE_HEADS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            q[0:LOCAL_H, :] = qflat[q0:q0 + LOCAL_H, :]
            pqflat[pq0:pq0 + CUBE_HEADS, :] = q

    padded_output = pl.create_tensor([rows, CUBE_HEADS * HEAD_DIM], dtype=pl.BF16)
    cube_ready = _attend_cube(
        padded_query, selected, indices, padded_sink, padded_output,
        num_tokens, selected_ready,
    )
    poflat = pl.reshape(padded_output, [rows * CUBE_HEADS, HEAD_DIM])
    oflat = pl.reshape(output, [rows * LOCAL_H, HEAD_DIM])
    with pl.spmd(workers, name_hint="sparse_unpad_heads", deps=[cube_ready]) as attended:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, workers):
            p0 = token * CUBE_HEADS
            o0 = token * LOCAL_H
            oflat[o0:o0 + LOCAL_H, :] = poflat[p0:p0 + LOCAL_H, :]
    return attended


attend_selected = _attend_cube if LOCAL_H >= 16 else _attend_padded
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
    workers_q = pl.max(1, pl.min(rows * (LOCAL_H // QUERY_HEAD_TILE), WORKERS))
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
    workers_kv = pl.max(1, pl.min(rows * (SELECTED_KEYS // 16), WORKERS))
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
    attended = attend_selected(qbf16, selected, indices, sink, result, rows, selected_ready)
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
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    workers = pl.max(1, pl.min(window_rows // GATHER_KEY_TILE, CACHE_WORKERS))
    with pl.spmd(workers, name_hint="sparse_shared_window_decode", deps=[ready]) as window_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, window_rows // GATHER_KEY_TILE, workers):
            row = block * GATHER_KEY_TILE
            w_payload = pl.tile.load(w, [row, 0], [GATHER_KEY_TILE, HEAD_DIM])
            w_factors = pl.tile.load(ws, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 32])
            w_scales = pl.cast(w_factors, pl.FP32)
            w_scale_column = pl.reshape(w_scales, [GATHER_KEY_TILE * (HEAD_DIM // 32), 1])
            w_values = pl.cast(w_payload, pl.FP32)
            w_grouped = pl.reshape(w_values, [GATHER_KEY_TILE * (HEAD_DIM // 32), 32])
            w_scaled = pl.row_expand_mul(w_grouped, w_scale_column)
            w_decoded = pl.cast(w_scaled, pl.BF16, mode="rint")
            decoded = pl.reshape(w_decoded, [GATHER_KEY_TILE, HEAD_DIM])
            pl.tile.store(decoded, [row, 0], decoded_window)

    compressed_rows = pl.tensor.dim(compressed, 0) * 128
    c = pl.reshape(compressed, [compressed_rows, HEAD_DIM // 2])
    cs = pl.reshape(compressed_scale, [compressed_rows, HEAD_DIM // 16])
    workers = pl.max(1, pl.min(compressed_rows // GATHER_KEY_TILE, CACHE_WORKERS))
    with pl.spmd(workers, name_hint="sparse_shared_compressed_decode", deps=[ready]) as compressed_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, compressed_rows // GATHER_KEY_TILE, workers):
            row = block * GATHER_KEY_TILE
            c_payload = pl.tile.load(c, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 2])
            c_factors = pl.tile.load(cs, [row, 0], [GATHER_KEY_TILE, HEAD_DIM // 16])
            c_fp4 = pl.reinterpret_view(c_payload, pl.FP4E2M1X2)
            c_values = pl.cast(c_fp4, pl.BF16)
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
def gather_decoded_caches(
    decoded_window: pl.Tensor[[CACHE_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    decoded_compressed: pl.Tensor[[CACHE_COMPRESSED_ROWS, HEAD_DIM], pl.BF16],
    wi: pl.Tensor[[T_DYN, 128], pl.INT32],
    ci: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    window_ready: pl.Scalar[pl.TASK_ID],
    compressed_ready: pl.Scalar[pl.TASK_ID],
    previous: pl.Scalar[pl.TASK_ID],
):
    selected_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    dst = pl.reshape(selected, [selected_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens * (SELECTED_KEYS // SELECTED_GATHER_TILE), SHARED_GATHER_WORKERS))
    with pl.spmd(
        workers, name_hint="sparse_shared_cache_gather",
        deps=[window_ready, compressed_ready, previous],
    ) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (SELECTED_KEYS // SELECTED_GATHER_TILE), workers):
            token = block // (SELECTED_KEYS // SELECTED_GATHER_TILE)
            column = block % (SELECTED_KEYS // SELECTED_GATHER_TILE) * SELECTED_GATHER_TILE
            values = pl.tile.create([SELECTED_GATHER_TILE, HEAD_DIM], dtype=pl.BF16)
            if column < 128:
                ids = pl.tile.load(wi, [token, column], [1, SELECTED_GATHER_TILE])
                safe_ids = pl.maximum(ids, 0)
                for lane in pl.range(SELECTED_GATHER_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    values = pl.tile.gather_row(
                        values, decoded_window, [lane, 0], [physical, 0], [1, HEAD_DIM]
                    )
            else:
                ids = pl.tile.load(ci, [token, column - 128], [1, SELECTED_GATHER_TILE])
                safe_ids = pl.maximum(ids, 0)
                for lane in pl.range(SELECTED_GATHER_TILE):
                    physical = pl.cast(pl.tile.read(safe_ids, [0, lane]), pl.INDEX)
                    values = pl.tile.gather_row(
                        values, decoded_compressed, [lane, 0], [physical, 0], [1, HEAD_DIM]
                    )
            pl.tile.store(values, [token * SELECTED_KEYS + column, 0], dst)
            zero = pl.tile.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for lane in pl.range(SELECTED_GATHER_TILE):
                if pl.tile.read(ids, [0, lane]) < 0:
                    pl.tile.store(zero, [token * SELECTED_KEYS + column + lane, 0], dst)
            pl.tile.store(ids, [token, column], indices)
    return gathered


@pl.jit.inline(auto_scope=False)
def _attend_cache_shared(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    capacity = pl.max(1, pl.min(TOKEN_TILE, num_tokens))
    selected = pl.create_tensor([capacity, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
    indices = pl.create_tensor([capacity, SELECTED_KEYS], dtype=pl.INT32)
    previous = query_ready
    for offset in pl.range(0, num_tokens, TOKEN_TILE):
        count = pl.min(TOKEN_TILE, num_tokens - offset)
        q = pl.slice(query, [count, LOCAL_H * HEAD_DIM], [offset, 0])
        wi = pl.slice(window_indices, [count, 128], [offset, 0])
        ci = pl.slice(compressed_indices, [count, INDEX_TOPK], [offset, 0])
        out = pl.slice(output, [count, LOCAL_H * HEAD_DIM], [offset, 0])
        gathered = gather_decoded_caches(
            decoded_window, decoded_compressed, wi, ci, selected, indices,
            count, window_ready, compressed_ready, previous,
        )
        previous = attend_selected(q, selected, indices, sink, out, count, gathered)
    return output


@pl.jit.inline(auto_scope=False)
def _attend_packed_block(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
):
    selected, indices, gathered = gather_packed(
        window, window_scale, compressed, compressed_scale,
        window_indices, compressed_indices, selected, indices, num_tokens, query_ready, max_workers=GATHER_WORKERS,
        name_hint="c2a_gather",
    )
    attended = attend_selected(query, selected, indices, sink, output, num_tokens, gathered)
    return attended


@pl.jit.inline(auto_scope=False)
def attend_packed(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 16], pl.BF16],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    marker = pl.create_tensor([1, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="sparse_query_ready", deps=[source_ready]) as ready:
        block = pl.tile.get_block_idx()
        marker[block:block + 1, :] = pl.cast(query[0:1, 0:16], pl.FP32)

    cache_rows = (pl.tensor.dim(window, 0) + pl.tensor.dim(compressed, 0)) * 128
    use_shared = pl.cast(0, pl.INT32)
    if cache_rows <= CACHE_MAX_ROWS:
        if cache_rows < num_tokens * SELECTED_KEYS:
            use_shared = pl.cast(1, pl.INT32)
    if use_shared == 1:
        output = _attend_cache_shared(
            query, window_indices, window, window_scale, compressed_indices,
            compressed, compressed_scale, sink, output, num_tokens, ready,
        )
    else:
        if TP_SIZE == 1 and num_tokens <= 12:
            ready = attend_packed_cohorts(
                query, window_indices, window, window_scale, compressed_indices,
                compressed, compressed_scale, sink, output, num_tokens, ready,
                gather_name="c2a_gather", attn_name="c2a_sparse_attn",
            )
        else:
            capacity = pl.max(1, pl.min(TOKEN_TILE, num_tokens))
            selected = pl.create_tensor([capacity, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
            indices = pl.create_tensor([capacity, SELECTED_KEYS], dtype=pl.INT32)
            for offset in pl.range(0, num_tokens, TOKEN_TILE):
                count = pl.min(TOKEN_TILE, num_tokens - offset)
                q = pl.slice(query, [count, LOCAL_H * HEAD_DIM], [offset, 0])
                wi = pl.slice(window_indices, [count, 128], [offset, 0])
                ci = pl.slice(compressed_indices, [count, INDEX_TOPK], [offset, 0])
                out = pl.slice(output, [count, LOCAL_H * HEAD_DIM], [offset, 0])
                ready = _attend_packed_block(
                    q, wi, window, window_scale,
                    ci, compressed, compressed_scale,
                    sink, out, selected, indices, count, ready,
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
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
        compressed, compressed_scale, sink, output, rows, source_ready,
    )
    return output


_attend_cohort = make_attend_selected_chunk(
    vector_query=False, selected_rows=PACKED_COHORT_TILE, max_workers=16,
)


@pl.jit.inline(auto_scope=False)
def attend_packed_cohorts(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
            wi_a, ci_a, selected_a, combined_a, active_a, done_a, max_workers=32, name_hint=gather_name,
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
                wi_b, ci_b, selected_b, combined_b, active_b, done_b, max_workers=32, name_hint=gather_name,
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
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
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
    done = attend_packed_cohorts(
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
        payload, scales = cache, inputs["scales"]
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
    inputs = dict(query=query, cache=cache, scales=scales, indices=indices)
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
    active = int(inputs["num_tokens"])
    difference = (actual[:active].float() - reference[:active]).abs()
    norm = float(reference[:active].norm())
    relative_l2 = float(difference.norm()) / norm if norm else float(difference.norm())
    bad = int((difference > 0.003 + 0.03 * reference[:active].abs()).sum())
    finite = bool(actual.isfinite().all())
    masked = (inputs["indices"][:active] < 0).all(dim=-1)
    masked_zero = bool((actual[:active][masked] == 0).all())
    inactive_exact = bool((actual[active:] == 7.25).all())
    passed = finite and relative_l2 < 0.006 and bad == 0 and masked_zero and inactive_exact
    detail = f"relative_l2={relative_l2:.8g}, bad={bad}, masked_zero={masked_zero}, inactive_exact={inactive_exact}"
    return passed, detail


def validate_swa_core(args):
    """Validate the split BF16-scale and packed544 production SWA paths."""
    from functools import partial

    from golden import run

    cases = ("gather", "plain", "runs") if args.swa_case == "all" else (args.swa_case,)
    layouts = (False,) if args.mode == "swa" else (True,)
    if args.mode in ("all", "swa-both"):
        layouts = (False, True)
    records = []
    for packed in layouts:
        entry = l2_sparse_attention_swa_packed if packed else l2_sparse_attention_swa
        mode = "swa-packed" if packed else "swa"
        for case in cases:
            inputs, reference = make_swa_case(case, packed)
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
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
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
        attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale, wo_a_scale=wo_a_scale,
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
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
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
    token_to_req_indices: torch.Tensor | None,
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
            if position_ids is None or query_start_loc is None or token_to_req_indices is None:
                raise ValueError("ratio-2 full mode requires positions, query starts and token-to-request indices")
            latent, publish_mask = golden_c2a_compressor(
                x,
                query_start_loc,
                position_ids,
                token_to_req_indices,
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
            or token_to_req_indices is None
            or compressed_lens is None
        ):
            raise ValueError("indexing modes require cache addressing and causal compressed lengths")
        if indexer_fn is None:
            raise ValueError("indexing modes require an indexer reference")
        index_args = (
            x, query_latent, token_to_req_indices, quantized_index, index_block_table, compressed_lens,
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
    output = golden_o_proj(
        attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale,
        wo_a_scale=wo_a_scale, output_dtype=output_dtype, sequential_projection=sequential_query,
    )
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
    from quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache,
        quantize_mxfp4_cache, quantize_mxfp8_cache,
    )

    parser = argparse.ArgumentParser(description="Validate sparse MLA and SWA on Ascend950")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4, 8])
    rows_help = "single row count (default: 1, 7, 12, 13; TP1 C2A also 32, 33, 80; fixture/runtime replay: 7)"
    parser.add_argument("--rows", type=int, help=rows_help)
    mode_choices = ["packed", "bf16", "both", "c2a", "swa", "swa-packed", "swa-both", "all"]
    parser.add_argument("--mode", default="all", choices=mode_choices)
    parser.add_argument("--swa-case", default="all", choices=["gather", "plain", "runs", "all"])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--runtime-dir", type=str)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        raise ValueError(f"--tp {args.tp} does not match configured TP_SIZE {TP_SIZE}")
    if args.mode == "c2a" and TP_SIZE != 1:
        parser.error("C2A standalone validation requires --tp 1")
    if args.mode in ("swa", "swa-packed", "swa-both", "all") and (args.fixture or args.runtime_dir):
        raise ValueError("--fixture and --runtime-dir require a sparse MLA mode: packed, bf16 or c2a")
    if args.runtime_dir and args.mode == "both":
        raise ValueError("--runtime-dir requires one --mode")
    if args.rows is not None and args.rows < 1:
        raise ValueError("--rows must be positive")
    if args.mode.startswith("swa"):
        _main_results = validate_swa_core(args)
    else:
        row_cases = [72]
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
            if args.mode == "all" and TP_SIZE == 1:
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
                    inputs = dict(query=query.flatten(1), window_indices=window_indices, window=wp, window_scale=ws,
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
