# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Gather sparse attention over packed FP8/FP4 caches with an attention sink."""

from pathlib import Path

import pypto.language as pl
import torch
from pypto.runtime import pto_isa_include_dir

from config import (
    FLASH as M, CACHE_ROWS_DYN, CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)
from quant import (
    dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp4_cache, quantize_mxfp8_cache,
    pack_quantized_cache, unpack_quantized_cache,
)

# model config
HEAD_DIM = M.head_dim
LOCAL_H = M.num_attention_heads
WINDOW_KEYS = M.sliding_window
INDEX_TOPK = M.index_topk
SCALE = HEAD_DIM ** -0.5

# tiling
SPARSE_VF_WORKER_TILE = 32
SPARSE_VF_KEY_TILE = 128
SPARSE_VF_GATHER_TILE = 32
SPARSE_VF_HEAD_TILE = min(32, LOCAL_H)
SPARSE_VF_CUBE_HEAD_TILE = max(16, SPARSE_VF_HEAD_TILE)
SPARSE_VF_PIPE_HEAD_TILE = min(64, LOCAL_H)
SPARSE_VF_PIPE_CUBE_HEAD_TILE = max(16, SPARSE_VF_PIPE_HEAD_TILE)
SPARSE_VF_PIPE_SPLIT = 1 if LOCAL_H >= 32 else 0
SPARSE_VF_PIPE_SLOT_BYTES = SPARSE_VF_PIPE_CUBE_HEAD_TILE * 256 * 4 // (2 if SPARSE_VF_PIPE_SPLIT else 1)
SPARSE_VF_PIPE_BUFFER_BYTES = SPARSE_VF_PIPE_SLOT_BYTES * 2
SPARSE_VF_REG_TILE = 128
SPARSE_VF_FUSED_HEAD_TILE = min(64, LOCAL_H)
SPARSE_VF_FUSED_CUBE_HEAD_TILE = max(16, SPARSE_VF_FUSED_HEAD_TILE)
SPARSE_VF_FUSED_AIV_HEAD_TILE = SPARSE_VF_FUSED_CUBE_HEAD_TILE // 2
SPARSE_VF_FUSED_GATHER_TILE = 16
SPARSE_VF_FUSED_SCALE_TILE = 8
SPARSE_VF_FUSED_FP4_SCALE_TILE = 16

# Packed cache layout.
SPARSE_VF_WINDOW_GROUP = 32
SPARSE_VF_COMPRESSED_GROUP = 16
SPARSE_VF_WINDOW_GROUPS = HEAD_DIM // SPARSE_VF_WINDOW_GROUP
SPARSE_VF_HALF_DIM = HEAD_DIM // 2
SPARSE_VF_WINDOW_HALFWORDS = WINDOW_CACHE_BYTES // 2
SPARSE_VF_COMPRESSED_HALFWORDS = COMPRESSED_CACHE_BYTES // 2
SPARSE_VF_WINDOW_SCALE_OFFSET = HEAD_DIM // 2
SPARSE_VF_COMPRESSED_SCALE_OFFSET = HEAD_DIM // 4
SPARSE_VF_WINDOW_SCALE_END = SPARSE_VF_WINDOW_SCALE_OFFSET + SPARSE_VF_WINDOW_GROUPS
SPARSE_VF_WINDOW_SHIFT = 5
SPARSE_VF_WINDOW_SCALE_TILE = SPARSE_VF_REG_TILE // SPARSE_VF_WINDOW_GROUP
SPARSE_VF_COMPRESSED_SCALE_TILE = SPARSE_VF_REG_TILE // SPARSE_VF_COMPRESSED_GROUP
SPARSE_VF_WINDOW_SCALE_ROW_TILE = SPARSE_VF_REG_TILE // SPARSE_VF_WINDOW_GROUPS


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "sparse_gather_vf.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "sparse_gather_vf.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _gather_sparse_vf_rows(
    output: pl.Out[pl.Tensor],
    query: pl.Tensor,
    sink: pl.Tensor,
    window: pl.Tensor,
    compressed: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_indices: pl.Tensor,
    partials: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
    window_keys: pl.Scalar[pl.INT32],
    compressed_keys: pl.Scalar[pl.INT32],
    balanced: pl.Scalar[pl.INT32],
    cache_mode: pl.constexpr,
):
    """Stream cache rows through L1 with sink-aware sparse attention.

    Scalar gather indices use refreshed GM cache lines; vector masks use
    the asynchronous index copy in UB.
    """
    ...


@pl.jit.inline(auto_scope=False)
def gather_sparse_attention(
    query: pl.Tensor,
    window_indices: pl.Tensor,
    window: pl.Tensor,
    compressed_indices: pl.Tensor,
    compressed: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    compressed_keys: pl.constexpr,
):
    """Attend H=32/64, D=512 packed caches with K=128 or 640 and both Vector lanes."""
    query = query[:num_tokens, :]
    active_output = output[:num_tokens, :]
    window_indices = window_indices[:num_tokens, :]
    compressed_indices = compressed_indices[:num_tokens, :]
    if compressed_keys == 0:
        query_waves = pl.max(1, (num_tokens + SPARSE_VF_WORKER_TILE - 1) // SPARSE_VF_WORKER_TILE)
        workers = pl.max(1, pl.min(SPARSE_VF_WORKER_TILE, (num_tokens + query_waves - 1) // query_waves))
    else:
        workers = pl.max(1, pl.min(num_tokens, SPARSE_VF_WORKER_TILE))
    workers = pl.min(workers, pl.system.available_cluster_count())
    parts = (WINDOW_KEYS + compressed_keys) // 128
    whole_tiles = ((num_tokens + workers - 1) // workers) * parts
    balanced_tiles = (num_tokens * parts + workers - 1) // workers
    balanced = pl.cast(0, pl.INT32)
    if num_tokens > workers:
        if whole_tiles - balanced_tiles >= 3:
            balanced = pl.cast(1, pl.INT32)
    slots = pl.max(1, 2 * workers * balanced)
    partials_size = slots * 64 * (HEAD_DIM + 2) + 2 * workers
    partials = pl.create_tensor([partials_size], dtype=pl.FP32)
    if balanced != 0:
        ready_offset = 2 * workers * 64 * (HEAD_DIM + 2)
        flat_partials = pl.reshape(partials, [1, partials_size])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sparse_gather_reset") as reset:
            flags = pl.tile.full([1, 2 * SPARSE_VF_WORKER_TILE], dtype=pl.FP32, value=0.0)
            flags = pl.tile.set_validshape(flags, 1, 2 * workers)
            pl.store(flags, [0, ready_offset], flat_partials)
        with pl.spmd(workers, name_hint="sparse_gather_vf", sync_start=True, deps=[reset]) as attended:
            _gather_sparse_vf_rows(
                active_output, query, sink, window, compressed,
                window_indices, compressed_indices, partials, num_tokens, LOCAL_H, WINDOW_KEYS, compressed_keys,
                balanced, cache_mode=compressed_keys,
            )
    else:
        with pl.spmd(workers, name_hint="sparse_gather_vf") as attended:
            _gather_sparse_vf_rows(
                active_output, query, sink, window, compressed,
                window_indices, compressed_indices, partials, num_tokens, LOCAL_H, WINDOW_KEYS, compressed_keys,
                balanced, cache_mode=compressed_keys,
            )
    return attended


@pl.jit.inline(auto_scope=False)
def sparse_vf_decode_pypto(
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    decoded: pl.Tensor,
    selected_ids: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
    compressed_keys: pl.constexpr,
):
    """Gather packed FP8/FP4 cache rows and apply BF16 group scales in registers."""
    keys = WINDOW_KEYS + compressed_keys
    window_rows = pl.tensor.dim(window, 0) * 128
    window_flat = pl.reshape(window, [window_rows, WINDOW_CACHE_BYTES])
    if compressed_keys > 0:
        compressed_rows = pl.tensor.dim(compressed, 0) * 128
        compressed_flat = pl.reshape(compressed, [compressed_rows, COMPRESSED_CACHE_BYTES])
    worker_limit = SPARSE_VF_WORKER_TILE
    if compressed_keys > 0:
        worker_limit = 2 * SPARSE_VF_WORKER_TILE
    workers = pl.min(num_tokens * (keys // SPARSE_VF_GATHER_TILE), worker_limit)
    workers = pl.max(1, workers)
    # A register scope cannot share a kernel with matmul operations.
    with pl.spmd(workers, name_hint="sparse_vf_decode", deps=[source_ready]) as gathered:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * (keys // SPARSE_VF_GATHER_TILE), workers):
            token = block // (keys // SPARSE_VF_GATHER_TILE)
            column = block % (keys // SPARSE_VF_GATHER_TILE) * SPARSE_VF_GATHER_TILE
            destination = token * keys + column
            if column < WINDOW_KEYS:
                # Scalar index reads require an explicit Tile load.
                ids = pl.tile.load(window_indices, [token, column], [1, SPARSE_VF_GATHER_TILE])
                raw = pl.tile.full([SPARSE_VF_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                first = pl.tile.read(ids, [0, 0])
                contiguous = pl.cast(0, pl.INT32)
                if first >= 0:
                    contiguous = pl.cast(1, pl.INT32)
                    for check_lane in pl.range(1, SPARSE_VF_GATHER_TILE):
                        check_row = pl.tile.read(ids, [0, check_lane])
                        if check_row != first + check_lane:
                            contiguous = pl.cast(0, pl.INT32)
                if contiguous != 0:
                    start = pl.cast(first, pl.INDEX)
                    raw = pl.tile.gather_row(
                        raw, window_flat, [0, 0], [start, 0], [SPARSE_VF_GATHER_TILE, WINDOW_CACHE_BYTES],
                    )
                else:
                    for lane in pl.range(SPARSE_VF_GATHER_TILE):
                        row = pl.tile.read(ids, [0, lane])
                        if row >= 0:
                            physical = pl.cast(row, pl.INDEX)
                            raw = pl.tile.gather_row(
                                raw, window_flat, [lane, 0], [physical, 0], [1, WINDOW_CACHE_BYTES],
                            )
                # Preserve the packed row pitch across reinterpretation and slicing.
                signed_bytes = pl.reinterpret_view(raw, pl.INT8)
                payload_all = pl.reinterpret_view(
                    signed_bytes, pl.FP8E4M3FN, shape=[SPARSE_VF_GATHER_TILE, WINDOW_CACHE_BYTES],
                )
                codes = payload_all[:, :HEAD_DIM]
                values_fp32 = pl.cast(codes, pl.FP32)
                values = pl.cast(values_fp32, pl.BF16, mode="rint")
                factor_bits = pl.reinterpret_view(
                    raw, pl.INT16, shape=[SPARSE_VF_GATHER_TILE, SPARSE_VF_WINDOW_HALFWORDS],
                )
                scale_all = pl.reinterpret_view(factor_bits, pl.BF16)
                scale_view = scale_all[:, SPARSE_VF_WINDOW_SCALE_OFFSET:SPARSE_VF_WINDOW_SCALE_END]
                factors = pl.tile.move(scale_view, pl.MemorySpace.Vec)
                pl.tile.store(ids, [token, column], selected_ids)
                value_flat = pl.reshape(values, [SPARSE_VF_GATHER_TILE * HEAD_DIM])
                factor_flat = pl.reshape(
                    factors, [SPARSE_VF_GATHER_TILE * SPARSE_VF_WINDOW_GROUPS],
                )
                # Expand the scale table in registers.
                with pl.reg_scope():
                    w_full = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                    w_group = pl.reg.index_gen(0, lanes=SPARSE_VF_REG_TILE, dtype=pl.UINT16)
                    w_group = pl.reg.shift_rights(w_group, SPARSE_VF_WINDOW_SHIFT, w_full)
                    for w_table in pl.hardware_loop(SPARSE_VF_GATHER_TILE // SPARSE_VF_WINDOW_SCALE_ROW_TILE):
                        w_table_offset = w_table * SPARSE_VF_REG_TILE
                        w_table_values = pl.reg.load(factor_flat, w_table_offset, lanes=SPARSE_VF_REG_TILE)
                        w_table_bits = pl.reg.interpret_cast(w_table_values, dtype=pl.UINT16)
                        for w_chunk in pl.hardware_loop(SPARSE_VF_REG_TILE // SPARSE_VF_WINDOW_SCALE_TILE):
                            w_base = pl.cast(w_chunk * SPARSE_VF_WINDOW_SCALE_TILE, pl.UINT16)
                            w_indices = pl.reg.adds(w_group, w_base, w_full)
                            w_lane_bits = pl.reg.gather_lanes(w_table_bits, w_indices)
                            w_factor = pl.reg.interpret_cast(w_lane_bits, dtype=pl.BF16)
                            w_table_base = w_table * (SPARSE_VF_REG_TILE // SPARSE_VF_WINDOW_SCALE_TILE)
                            w_offset = (w_table_base + w_chunk) * SPARSE_VF_REG_TILE
                            w_value = pl.reg.load(value_flat, w_offset, lanes=SPARSE_VF_REG_TILE)
                            w_scaled = pl.reg.mul(w_value, w_factor, w_full)
                            _w_write = pl.reg.store(value_flat, w_offset, w_scaled)
                scaled_values = pl.reshape(value_flat, [SPARSE_VF_GATHER_TILE, HEAD_DIM])
                pl.tile.store(scaled_values, [destination, 0], decoded)
            if compressed_keys > 0:
                if column >= WINDOW_KEYS:
                    c_column = column - WINDOW_KEYS
                    c_ids = pl.tile.load(compressed_indices, [token, c_column], [1, SPARSE_VF_GATHER_TILE])
                    c_raw = pl.tile.full([SPARSE_VF_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                    c_first = pl.tile.read(c_ids, [0, 0])
                    c_contiguous = pl.cast(0, pl.INT32)
                    if c_first >= 0:
                        c_contiguous = pl.cast(1, pl.INT32)
                        for c_check_lane in pl.range(1, SPARSE_VF_GATHER_TILE):
                            c_check_row = pl.tile.read(c_ids, [0, c_check_lane])
                            if c_check_row != c_first + c_check_lane:
                                c_contiguous = pl.cast(0, pl.INT32)
                    if c_contiguous != 0:
                        c_start = pl.cast(c_first, pl.INDEX)
                        c_raw = pl.tile.gather_row(
                            c_raw, compressed_flat, [0, 0], [c_start, 0],
                            [SPARSE_VF_GATHER_TILE, COMPRESSED_CACHE_BYTES],
                        )
                    else:
                        for c_lane in pl.range(SPARSE_VF_GATHER_TILE):
                            c_row = pl.tile.read(c_ids, [0, c_lane])
                            if c_row >= 0:
                                c_physical = pl.cast(c_row, pl.INDEX)
                                c_raw = pl.tile.gather_row(
                                    c_raw, compressed_flat, [c_lane, 0], [c_physical, 0],
                                    [1, COMPRESSED_CACHE_BYTES],
                                )
                    # Each FP4E2M1X2 carrier contains two FP4 values.
                    c_payload_all = pl.reinterpret_view(c_raw, pl.FP4E2M1X2)
                    c_codes = c_payload_all[:, :SPARSE_VF_HALF_DIM]
                    values = pl.cast(c_codes, pl.BF16)
                    c_factor_bits = pl.reinterpret_view(
                        c_raw, pl.INT16,
                        shape=[SPARSE_VF_GATHER_TILE, SPARSE_VF_COMPRESSED_HALFWORDS],
                    )
                    c_scale_all = pl.reinterpret_view(c_factor_bits, pl.BF16)
                    pl.tile.store(c_ids, [token, column], selected_ids)
                    c_value_flat = pl.reshape(values, [SPARSE_VF_GATHER_TILE * HEAD_DIM])
                    c_scale_words = pl.tile.reinterpret_view(c_scale_all, pl.UINT16)
                    c_factor_flat = pl.reshape(
                        c_scale_words, [SPARSE_VF_GATHER_TILE * SPARSE_VF_COMPRESSED_HALFWORDS],
                    )
                    with pl.reg_scope():
                        c_full = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                        for c_scale_row in pl.hardware_loop(SPARSE_VF_GATHER_TILE):
                            c_row_scale_offset = c_scale_row * SPARSE_VF_COMPRESSED_HALFWORDS
                            c_row_scale_offset = c_row_scale_offset + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                            for c_scale_chunk in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                c_scale_offset = c_row_scale_offset + c_scale_chunk * SPARSE_VF_COMPRESSED_SCALE_TILE
                                c_groups = pl.reg.load(c_factor_flat, c_scale_offset, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                c_group_bits = pl.reg.group_broadcast(c_groups, lanes=SPARSE_VF_REG_TILE)
                                c_factor = pl.reg.interpret_cast(c_group_bits, dtype=pl.BF16)
                                c_offset = c_scale_row * HEAD_DIM + c_scale_chunk * SPARSE_VF_REG_TILE
                                c_value = pl.reg.load(c_value_flat, c_offset, lanes=SPARSE_VF_REG_TILE)
                                c_scaled = pl.reg.mul(c_value, c_factor, c_full)
                                _c_write = pl.reg.store(c_value_flat, c_offset, c_scaled)
                    c_scaled_values = pl.reshape(c_value_flat, [SPARSE_VF_GATHER_TILE, HEAD_DIM])
                    pl.tile.store(c_scaled_values, [destination, 0], decoded)
    return gathered


@pl.jit.inline(auto_scope=False)
def sparse_vf_pipeline_worker_count(num_tokens: pl.Scalar[pl.INT32]):
    """Use the fewest clusters that preserve the existing maximum task count per cluster."""
    tasks = num_tokens * (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE)
    waves = pl.max(1, (tasks + SPARSE_VF_WORKER_TILE - 1) // SPARSE_VF_WORKER_TILE)
    workers = pl.max(1, (tasks + waves - 1) // waves)
    return workers


@pl.jit.incore(level=pl.Level.AIC)
def sparse_vf_pipeline_cube(
    query: pl.Tensor,
    decoded: pl.Tensor,
    selected_ids: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    compressed_keys: pl.constexpr,
):
    v2c_buffer = pl.reserve_buffer(name="sparse_vf_pipeline_weights", size=SPARSE_VF_PIPE_BUFFER_BYTES)
    c2v_peer = pl.import_peer_buffer(name="sparse_vf_pipeline_scores", peer_func="sparse_vf_pipeline_vector")
    pl.aic_initialize_pipe(c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buffer, dir_mask=3, slot_size=SPARSE_VF_PIPE_SLOT_BYTES, slot_num=2)
    keys = WINDOW_KEYS + compressed_keys
    workers = sparse_vf_pipeline_worker_count(num_tokens)
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, num_tokens * (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE), workers):
        token = block // (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE)
        head = block % (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE) * SPARSE_VF_PIPE_HEAD_TILE
        q0 = token * LOCAL_H + head
        q = pl.tile.load(query, [q0, 0], [SPARSE_VF_PIPE_CUBE_HEAD_TILE, HEAD_DIM], valid_shape=[SPARSE_VF_PIPE_HEAD_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        k0_0 = token * keys + 0 * SPARSE_VF_KEY_TILE
        kv_0 = pl.tile.load(decoded, [k0_0, 0], [SPARSE_VF_KEY_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        key_0 = pl.tile.transpose_view(kv_0)
        qk_0 = pl.tile.matmul(q, key_0)
        pl.tpush_to_aiv(qk_0, split=SPARSE_VF_PIPE_SPLIT)
        if compressed_keys > 0:
            k0_1 = token * keys + 1 * SPARSE_VF_KEY_TILE
            kv_1 = pl.tile.load(decoded, [k0_1, 0], [SPARSE_VF_KEY_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            key_1 = pl.tile.transpose_view(kv_1)
            qk_1 = pl.tile.matmul(q, key_1)
            pl.tpush_to_aiv(qk_1, split=SPARSE_VF_PIPE_SPLIT)
            weights_0 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_0 = pl.tile.move(weights_0, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_0)
            for value_column_0 in pl.range(0, HEAD_DIM, 256):
                value_mat_0 = pl.tile.slice(kv_0, [SPARSE_VF_KEY_TILE, 256], [0, value_column_0])
                value_right_0 = pl.tile.move(value_mat_0, pl.MemorySpace.Right)
                weighted_0 = pl.tile.matmul(weights_left_0, value_right_0)
                pl.tpush_to_aiv(weighted_0, split=SPARSE_VF_PIPE_SPLIT)
            k0_2 = token * keys + 2 * SPARSE_VF_KEY_TILE
            kv_2 = pl.tile.load(decoded, [k0_2, 0], [SPARSE_VF_KEY_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            key_2 = pl.tile.transpose_view(kv_2)
            qk_2 = pl.tile.matmul(q, key_2)
            pl.tpush_to_aiv(qk_2, split=SPARSE_VF_PIPE_SPLIT)
            weights_1 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_1 = pl.tile.move(weights_1, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_1)
            for value_column_1 in pl.range(0, HEAD_DIM, 256):
                value_mat_1 = pl.tile.slice(kv_1, [SPARSE_VF_KEY_TILE, 256], [0, value_column_1])
                value_right_1 = pl.tile.move(value_mat_1, pl.MemorySpace.Right)
                weighted_1 = pl.tile.matmul(weights_left_1, value_right_1)
                pl.tpush_to_aiv(weighted_1, split=SPARSE_VF_PIPE_SPLIT)
            k0_3 = token * keys + 3 * SPARSE_VF_KEY_TILE
            kv_3 = pl.tile.load(decoded, [k0_3, 0], [SPARSE_VF_KEY_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            key_3 = pl.tile.transpose_view(kv_3)
            qk_3 = pl.tile.matmul(q, key_3)
            pl.tpush_to_aiv(qk_3, split=SPARSE_VF_PIPE_SPLIT)
            weights_2 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_2 = pl.tile.move(weights_2, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_2)
            for value_column_2 in pl.range(0, HEAD_DIM, 256):
                value_mat_2 = pl.tile.slice(kv_2, [SPARSE_VF_KEY_TILE, 256], [0, value_column_2])
                value_right_2 = pl.tile.move(value_mat_2, pl.MemorySpace.Right)
                weighted_2 = pl.tile.matmul(weights_left_2, value_right_2)
                pl.tpush_to_aiv(weighted_2, split=SPARSE_VF_PIPE_SPLIT)
            k0_4 = token * keys + 4 * SPARSE_VF_KEY_TILE
            kv_4 = pl.tile.load(decoded, [k0_4, 0], [SPARSE_VF_KEY_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            key_4 = pl.tile.transpose_view(kv_4)
            qk_4 = pl.tile.matmul(q, key_4)
            pl.tpush_to_aiv(qk_4, split=SPARSE_VF_PIPE_SPLIT)
            weights_3 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_3 = pl.tile.move(weights_3, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_3)
            for value_column_3 in pl.range(0, HEAD_DIM, 256):
                value_mat_3 = pl.tile.slice(kv_3, [SPARSE_VF_KEY_TILE, 256], [0, value_column_3])
                value_right_3 = pl.tile.move(value_mat_3, pl.MemorySpace.Right)
                weighted_3 = pl.tile.matmul(weights_left_3, value_right_3)
                pl.tpush_to_aiv(weighted_3, split=SPARSE_VF_PIPE_SPLIT)
            weights_4 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_4 = pl.tile.move(weights_4, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_4)
            for value_column_4 in pl.range(0, HEAD_DIM, 256):
                value_mat_4 = pl.tile.slice(kv_4, [SPARSE_VF_KEY_TILE, 256], [0, value_column_4])
                value_right_4 = pl.tile.move(value_mat_4, pl.MemorySpace.Right)
                weighted_4 = pl.tile.matmul(weights_left_4, value_right_4)
                pl.tpush_to_aiv(weighted_4, split=SPARSE_VF_PIPE_SPLIT)
        else:
            weights_0 = pl.tpop_from_aiv(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=SPARSE_VF_PIPE_SPLIT)
            weights_left_0 = pl.tile.move(weights_0, pl.MemorySpace.Left)
            pl.tfree_to_aiv(weights_0)
            for value_column_0 in pl.range(0, HEAD_DIM, 256):
                value_mat_0 = pl.tile.slice(kv_0, [SPARSE_VF_KEY_TILE, 256], [0, value_column_0])
                value_right_0 = pl.tile.move(value_mat_0, pl.MemorySpace.Right)
                weighted_0 = pl.tile.matmul(weights_left_0, value_right_0)
                pl.tpush_to_aiv(weighted_0, split=SPARSE_VF_PIPE_SPLIT)
    return output


@pl.jit.incore(level=pl.Level.AIV)
def sparse_vf_pipeline_vector(
    query: pl.Tensor,
    decoded: pl.Tensor,
    selected_ids: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    compressed_keys: pl.constexpr,
):
    c2v_buffer = pl.reserve_buffer(name="sparse_vf_pipeline_scores", size=SPARSE_VF_PIPE_BUFFER_BYTES)
    v2c_peer = pl.import_peer_buffer(name="sparse_vf_pipeline_weights", peer_func="sparse_vf_pipeline_cube")
    pl.aiv_initialize_pipe(c2v_consumer_buf=c2v_buffer, v2c_consumer_buf=v2c_peer, dir_mask=3, slot_size=SPARSE_VF_PIPE_SLOT_BYTES, slot_num=2)
    _keys = WINDOW_KEYS + compressed_keys
    workers = sparse_vf_pipeline_worker_count(num_tokens)
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, num_tokens * (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE), workers):
        token = block // (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE)
        head = block % (LOCAL_H // SPARSE_VF_PIPE_HEAD_TILE) * SPARSE_VF_PIPE_HEAD_TILE
        q0 = token * LOCAL_H + head
        initial_tmp = pl.tile.create([SPARSE_VF_PIPE_CUBE_HEAD_TILE, 8], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        maximum_seed = pl.tile.full([SPARSE_VF_PIPE_CUBE_HEAD_TILE, 8], dtype=pl.FP32, value=-1e30)
        maximum_column = pl.tile.row_max(maximum_seed, initial_tmp)
        maximum = pl.tile.reshape(maximum_column, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
        denominator_seed = pl.tile.full([SPARSE_VF_PIPE_CUBE_HEAD_TILE, 8], dtype=pl.FP32, value=0.0)
        denominator_column = pl.tile.row_sum(denominator_seed, initial_tmp)
        denominator = pl.tile.reshape(denominator_column, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
        numerator_lo = pl.tile.full([SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, value=0.0)
        numerator_hi = pl.tile.full([SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, value=0.0)
        reduction_tmp = pl.tile.create([SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        qk_vec_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
        scores_0 = pl.mul(qk_vec_0, SCALE)
        ids_0 = pl.tile.load(selected_ids, [token, 0 * SPARSE_VF_KEY_TILE], [1, SPARSE_VF_KEY_TILE], target_memory=pl.MemorySpace.Vec)
        idx_0 = pl.cast(ids_0, pl.FP32)
        positive_0 = pl.add(idx_0, 1.0)
        nonnegative_0 = pl.maximum(positive_0, 0.0)
        valid_0 = pl.minimum(nonnegative_0, 1.0)
        masked_0 = pl.sub(valid_0, 1.0)
        penalty_0 = pl.mul(masked_0, 1e30)
        scores_0 = pl.col_expand_add(scores_0, penalty_0)
        row_max_0 = pl.tile.row_max(scores_0, reduction_tmp)
        tile_max_0 = pl.tile.reshape(row_max_0, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
        next_max_0 = pl.maximum(maximum, tile_max_0)
        correction_0 = pl.col_expand_expdif(maximum, next_max_0)
        next_max_col_0 = pl.tile.reshape(next_max_0, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
        exponentials_0 = pl.row_expand_expdif(scores_0, next_max_col_0)
        probabilities_0 = pl.col_expand_mul(exponentials_0, valid_0)
        row_sum_0 = pl.tile.row_sum(probabilities_0, reduction_tmp)
        tile_sum_0 = pl.tile.reshape(row_sum_0, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
        corrected_sum_0 = pl.mul(denominator, correction_0)
        denominator = pl.add(corrected_sum_0, tile_sum_0)
        weights_vec_0 = pl.cast(probabilities_0, pl.BF16, mode="rint")
        maximum = next_max_0
        pl.tpush_to_aic(weights_vec_0, split=SPARSE_VF_PIPE_SPLIT)
        pl.tfree_to_aic(qk_vec_0)
        if compressed_keys > 0:
            qk_vec_1 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scores_1 = pl.mul(qk_vec_1, SCALE)
            ids_1 = pl.tile.load(selected_ids, [token, 1 * SPARSE_VF_KEY_TILE], [1, SPARSE_VF_KEY_TILE], target_memory=pl.MemorySpace.Vec)
            idx_1 = pl.cast(ids_1, pl.FP32)
            positive_1 = pl.add(idx_1, 1.0)
            nonnegative_1 = pl.maximum(positive_1, 0.0)
            valid_1 = pl.minimum(nonnegative_1, 1.0)
            masked_1 = pl.sub(valid_1, 1.0)
            penalty_1 = pl.mul(masked_1, 1e30)
            scores_1 = pl.col_expand_add(scores_1, penalty_1)
            row_max_1 = pl.tile.row_max(scores_1, reduction_tmp)
            tile_max_1 = pl.tile.reshape(row_max_1, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            next_max_1 = pl.maximum(maximum, tile_max_1)
            correction_1 = pl.col_expand_expdif(maximum, next_max_1)
            next_max_col_1 = pl.tile.reshape(next_max_1, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            exponentials_1 = pl.row_expand_expdif(scores_1, next_max_col_1)
            probabilities_1 = pl.col_expand_mul(exponentials_1, valid_1)
            row_sum_1 = pl.tile.row_sum(probabilities_1, reduction_tmp)
            tile_sum_1 = pl.tile.reshape(row_sum_1, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            corrected_sum_1 = pl.mul(denominator, correction_1)
            denominator = pl.add(corrected_sum_1, tile_sum_1)
            weights_vec_1 = pl.cast(probabilities_1, pl.BF16, mode="rint")
            maximum = next_max_1
            pl.tpush_to_aic(weights_vec_1, split=SPARSE_VF_PIPE_SPLIT)
            pl.tfree_to_aic(qk_vec_1)
            correction_col_0 = pl.tile.reshape(correction_0, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_0_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_0_0 = pl.row_expand_mul(numerator_lo, correction_col_0)
            numerator_lo = pl.add(scaled_0_0, weighted_vec_0_0)
            pl.tfree_to_aic(weighted_vec_0_0)
            weighted_vec_0_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_0_256 = pl.row_expand_mul(numerator_hi, correction_col_0)
            numerator_hi = pl.add(scaled_0_256, weighted_vec_0_256)
            pl.tfree_to_aic(weighted_vec_0_256)
            qk_vec_2 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scores_2 = pl.mul(qk_vec_2, SCALE)
            ids_2 = pl.tile.load(selected_ids, [token, 2 * SPARSE_VF_KEY_TILE], [1, SPARSE_VF_KEY_TILE], target_memory=pl.MemorySpace.Vec)
            idx_2 = pl.cast(ids_2, pl.FP32)
            positive_2 = pl.add(idx_2, 1.0)
            nonnegative_2 = pl.maximum(positive_2, 0.0)
            valid_2 = pl.minimum(nonnegative_2, 1.0)
            masked_2 = pl.sub(valid_2, 1.0)
            penalty_2 = pl.mul(masked_2, 1e30)
            scores_2 = pl.col_expand_add(scores_2, penalty_2)
            row_max_2 = pl.tile.row_max(scores_2, reduction_tmp)
            tile_max_2 = pl.tile.reshape(row_max_2, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            next_max_2 = pl.maximum(maximum, tile_max_2)
            correction_2 = pl.col_expand_expdif(maximum, next_max_2)
            next_max_col_2 = pl.tile.reshape(next_max_2, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            exponentials_2 = pl.row_expand_expdif(scores_2, next_max_col_2)
            probabilities_2 = pl.col_expand_mul(exponentials_2, valid_2)
            row_sum_2 = pl.tile.row_sum(probabilities_2, reduction_tmp)
            tile_sum_2 = pl.tile.reshape(row_sum_2, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            corrected_sum_2 = pl.mul(denominator, correction_2)
            denominator = pl.add(corrected_sum_2, tile_sum_2)
            weights_vec_2 = pl.cast(probabilities_2, pl.BF16, mode="rint")
            maximum = next_max_2
            pl.tpush_to_aic(weights_vec_2, split=SPARSE_VF_PIPE_SPLIT)
            pl.tfree_to_aic(qk_vec_2)
            correction_col_1 = pl.tile.reshape(correction_1, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_1_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_1_0 = pl.row_expand_mul(numerator_lo, correction_col_1)
            numerator_lo = pl.add(scaled_1_0, weighted_vec_1_0)
            pl.tfree_to_aic(weighted_vec_1_0)
            weighted_vec_1_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_1_256 = pl.row_expand_mul(numerator_hi, correction_col_1)
            numerator_hi = pl.add(scaled_1_256, weighted_vec_1_256)
            pl.tfree_to_aic(weighted_vec_1_256)
            qk_vec_3 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scores_3 = pl.mul(qk_vec_3, SCALE)
            ids_3 = pl.tile.load(selected_ids, [token, 3 * SPARSE_VF_KEY_TILE], [1, SPARSE_VF_KEY_TILE], target_memory=pl.MemorySpace.Vec)
            idx_3 = pl.cast(ids_3, pl.FP32)
            positive_3 = pl.add(idx_3, 1.0)
            nonnegative_3 = pl.maximum(positive_3, 0.0)
            valid_3 = pl.minimum(nonnegative_3, 1.0)
            masked_3 = pl.sub(valid_3, 1.0)
            penalty_3 = pl.mul(masked_3, 1e30)
            scores_3 = pl.col_expand_add(scores_3, penalty_3)
            row_max_3 = pl.tile.row_max(scores_3, reduction_tmp)
            tile_max_3 = pl.tile.reshape(row_max_3, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            next_max_3 = pl.maximum(maximum, tile_max_3)
            correction_3 = pl.col_expand_expdif(maximum, next_max_3)
            next_max_col_3 = pl.tile.reshape(next_max_3, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            exponentials_3 = pl.row_expand_expdif(scores_3, next_max_col_3)
            probabilities_3 = pl.col_expand_mul(exponentials_3, valid_3)
            row_sum_3 = pl.tile.row_sum(probabilities_3, reduction_tmp)
            tile_sum_3 = pl.tile.reshape(row_sum_3, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            corrected_sum_3 = pl.mul(denominator, correction_3)
            denominator = pl.add(corrected_sum_3, tile_sum_3)
            weights_vec_3 = pl.cast(probabilities_3, pl.BF16, mode="rint")
            maximum = next_max_3
            pl.tpush_to_aic(weights_vec_3, split=SPARSE_VF_PIPE_SPLIT)
            pl.tfree_to_aic(qk_vec_3)
            correction_col_2 = pl.tile.reshape(correction_2, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_2_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_2_0 = pl.row_expand_mul(numerator_lo, correction_col_2)
            numerator_lo = pl.add(scaled_2_0, weighted_vec_2_0)
            pl.tfree_to_aic(weighted_vec_2_0)
            weighted_vec_2_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_2_256 = pl.row_expand_mul(numerator_hi, correction_col_2)
            numerator_hi = pl.add(scaled_2_256, weighted_vec_2_256)
            pl.tfree_to_aic(weighted_vec_2_256)
            qk_vec_4 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scores_4 = pl.mul(qk_vec_4, SCALE)
            ids_4 = pl.tile.load(selected_ids, [token, 4 * SPARSE_VF_KEY_TILE], [1, SPARSE_VF_KEY_TILE], target_memory=pl.MemorySpace.Vec)
            idx_4 = pl.cast(ids_4, pl.FP32)
            positive_4 = pl.add(idx_4, 1.0)
            nonnegative_4 = pl.maximum(positive_4, 0.0)
            valid_4 = pl.minimum(nonnegative_4, 1.0)
            masked_4 = pl.sub(valid_4, 1.0)
            penalty_4 = pl.mul(masked_4, 1e30)
            scores_4 = pl.col_expand_add(scores_4, penalty_4)
            row_max_4 = pl.tile.row_max(scores_4, reduction_tmp)
            tile_max_4 = pl.tile.reshape(row_max_4, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            next_max_4 = pl.maximum(maximum, tile_max_4)
            correction_4 = pl.col_expand_expdif(maximum, next_max_4)
            next_max_col_4 = pl.tile.reshape(next_max_4, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            exponentials_4 = pl.row_expand_expdif(scores_4, next_max_col_4)
            probabilities_4 = pl.col_expand_mul(exponentials_4, valid_4)
            row_sum_4 = pl.tile.row_sum(probabilities_4, reduction_tmp)
            tile_sum_4 = pl.tile.reshape(row_sum_4, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
            corrected_sum_4 = pl.mul(denominator, correction_4)
            denominator = pl.add(corrected_sum_4, tile_sum_4)
            weights_vec_4 = pl.cast(probabilities_4, pl.BF16, mode="rint")
            maximum = next_max_4
            pl.tpush_to_aic(weights_vec_4, split=SPARSE_VF_PIPE_SPLIT)
            pl.tfree_to_aic(qk_vec_4)
            correction_col_3 = pl.tile.reshape(correction_3, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_3_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_3_0 = pl.row_expand_mul(numerator_lo, correction_col_3)
            numerator_lo = pl.add(scaled_3_0, weighted_vec_3_0)
            pl.tfree_to_aic(weighted_vec_3_0)
            weighted_vec_3_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_3_256 = pl.row_expand_mul(numerator_hi, correction_col_3)
            numerator_hi = pl.add(scaled_3_256, weighted_vec_3_256)
            pl.tfree_to_aic(weighted_vec_3_256)
            correction_col_4 = pl.tile.reshape(correction_4, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_4_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_4_0 = pl.row_expand_mul(numerator_lo, correction_col_4)
            numerator_lo = pl.add(scaled_4_0, weighted_vec_4_0)
            pl.tfree_to_aic(weighted_vec_4_0)
            weighted_vec_4_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_4_256 = pl.row_expand_mul(numerator_hi, correction_col_4)
            numerator_hi = pl.add(scaled_4_256, weighted_vec_4_256)
            pl.tfree_to_aic(weighted_vec_4_256)
        else:
            correction_col_0 = pl.tile.reshape(correction_0, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
            weighted_vec_0_0 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_0_0 = pl.row_expand_mul(numerator_lo, correction_col_0)
            numerator_lo = pl.add(scaled_0_0, weighted_vec_0_0)
            pl.tfree_to_aic(weighted_vec_0_0)
            weighted_vec_0_256 = pl.tpop_from_aic(shape=[SPARSE_VF_PIPE_CUBE_HEAD_TILE, 256], dtype=pl.FP32, split=SPARSE_VF_PIPE_SPLIT)
            scaled_0_256 = pl.row_expand_mul(numerator_hi, correction_col_0)
            numerator_hi = pl.add(scaled_0_256, weighted_vec_0_256)
            pl.tfree_to_aic(weighted_vec_0_256)
        sink_full = pl.tile.load(sink, [0, head], [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE], valid_shape=[1, SPARSE_VF_PIPE_HEAD_TILE], target_memory=pl.MemorySpace.Vec)
        sink_column = pl.tile.reshape(sink_full, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
        sinks = pl.tile.reshape(sink_column, [1, SPARSE_VF_PIPE_CUBE_HEAD_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.col_expand_expdif(maximum, final_max)
        sink_weight = pl.col_expand_expdif(sinks, final_max)
        corrected_total = pl.mul(denominator, correction)
        total = pl.add(corrected_total, sink_weight)
        ratio = pl.div(correction, total)
        multiplier = pl.tile.reshape(ratio, [SPARSE_VF_PIPE_CUBE_HEAD_TILE, 1])
        normalized_lo = pl.row_expand_mul(numerator_lo, multiplier)
        result_lo = pl.cast(normalized_lo, pl.BF16, mode="rint")
        valid_result_lo = pl.set_validshape(result_lo, SPARSE_VF_PIPE_HEAD_TILE, 256)
        pl.tile.store(valid_result_lo, [q0, 0], output)
        normalized_hi = pl.row_expand_mul(numerator_hi, multiplier)
        result_hi = pl.cast(normalized_hi, pl.BF16, mode="rint")
        valid_result_hi = pl.set_validshape(result_hi, SPARSE_VF_PIPE_HEAD_TILE, 256)
        pl.tile.store(valid_result_hi, [q0, 256], output)
    return output


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def sparse_vf_pipeline_group(
    query: pl.Tensor,
    decoded: pl.Tensor,
    selected_ids: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    compressed_keys: pl.constexpr,
):
    cube_output = sparse_vf_pipeline_cube(query, decoded, selected_ids, sink, output, num_tokens, compressed_keys=compressed_keys)
    vector_output = sparse_vf_pipeline_vector(query, decoded, selected_ids, sink, cube_output, num_tokens, compressed_keys=compressed_keys)
    return vector_output


@pl.jit.inline(auto_scope=False)
def sparse_vf_attend_pypto(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    decoded: pl.Tensor,
    selected_ids: pl.Tensor,
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    decoded_ready: pl.Scalar[pl.TASK_ID],
    compressed_keys: pl.constexpr,
):
    """Use explicit on-chip pipeline for packed keys; preserve single-part SWA."""
    attended = pl.system.task_invalid()
    if compressed_keys > 0:
        tokens = pl.tensor.dim(query, 0)
        qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
        oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
        sink_column = pl.reshape(sink, [1, LOCAL_H])
        workers = sparse_vf_pipeline_worker_count(num_tokens)
        with pl.spmd(workers, name_hint="sparse_vf_attend", deps=[decoded_ready]) as packed_attended:
            _pipeline_output = sparse_vf_pipeline_group(qflat, decoded, selected_ids, sink_column, oflat, num_tokens, compressed_keys=compressed_keys)
        attended = packed_attended
    else:
        keys = WINDOW_KEYS + compressed_keys
        parts = keys // SPARSE_VF_KEY_TILE
        tokens = pl.tensor.dim(query, 0)
        qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
        oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
        sink_row = pl.reshape(sink, [1, LOCAL_H])
        workers = pl.min(num_tokens * (LOCAL_H // SPARSE_VF_HEAD_TILE), SPARSE_VF_WORKER_TILE)
        workers = pl.max(1, workers)
        with pl.spmd(
            workers, name_hint="sparse_vf_attend", deps=[decoded_ready],
            optimizations=[pl.cross_core_slot(slot_num=1)],
        ) as swa_attended:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, num_tokens * (LOCAL_H // SPARSE_VF_HEAD_TILE), workers):
                token = block // (LOCAL_H // SPARSE_VF_HEAD_TILE)
                head = block % (LOCAL_H // SPARSE_VF_HEAD_TILE) * SPARSE_VF_HEAD_TILE
                q0 = token * LOCAL_H + head
                # Pad Cube rows within the current token.
                q = pl.slice(qflat, [SPARSE_VF_CUBE_HEAD_TILE, HEAD_DIM], [q0, 0],
                             valid_shape=[SPARSE_VF_HEAD_TILE, HEAD_DIM])
                maximum = pl.full([1, SPARSE_VF_CUBE_HEAD_TILE], dtype=pl.FP32, value=-1e30)
                denominator = pl.full([1, SPARSE_VF_CUBE_HEAD_TILE], dtype=pl.FP32, value=0.0)
                numerator = pl.full([SPARSE_VF_CUBE_HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for part in pl.range(parts):
                    k0 = token * keys + part * SPARSE_VF_KEY_TILE
                    kv = decoded[k0:k0 + SPARSE_VF_KEY_TILE, :]
                    # Compute scaled QK scores and mask invalid keys.
                    qk = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
                    scores = pl.mul(qk, SCALE)
                    column = part * SPARSE_VF_KEY_TILE
                    ids = selected_ids[token:token + 1, column:column + SPARSE_VF_KEY_TILE]
                    idx = pl.cast(ids, pl.FP32)
                    positive = pl.add(idx, 1.0)
                    nonnegative = pl.maximum(positive, 0.0)
                    valid = pl.minimum(nonnegative, 1.0)
                    masked = pl.sub(valid, 1.0)
                    penalty = pl.mul(masked, 1e30)
                    scores = pl.col_expand_add(scores, penalty)
                    row_max = pl.row_max(scores)
                    tile_max = pl.reshape(row_max, [1, SPARSE_VF_CUBE_HEAD_TILE])
                    next_max = pl.maximum(maximum, tile_max)
                    max_delta = pl.sub(maximum, next_max)
                    correction = pl.exp(max_delta)
                    next_max_col = pl.reshape(next_max, [SPARSE_VF_CUBE_HEAD_TILE, 1])
                    shifted = pl.row_expand_sub(scores, next_max_col)
                    exponentials = pl.exp(shifted)
                    probabilities = pl.col_expand_mul(exponentials, valid)
                    row_sum = pl.row_sum(probabilities)
                    tile_sum = pl.reshape(row_sum, [1, SPARSE_VF_CUBE_HEAD_TILE])
                    corrected_sum = pl.mul(denominator, correction)
                    denominator = pl.add(corrected_sum, tile_sum)
                    # Accumulate BF16 PV with the online-softmax correction.
                    weights = pl.cast(probabilities, pl.BF16, mode="rint")
                    weighted = pl.matmul(weights, kv, out_dtype=pl.FP32)
                    correction_col = pl.reshape(correction, [SPARSE_VF_CUBE_HEAD_TILE, 1])
                    scaled = pl.row_expand_mul(numerator, correction_col)
                    numerator = pl.add(scaled, weighted)
                    maximum = next_max
                # Normalize the accumulated values with the sink contribution.
                sinks = pl.slice(sink_row, [1, SPARSE_VF_CUBE_HEAD_TILE], [0, head],
                                 valid_shape=[1, SPARSE_VF_HEAD_TILE])
                final_max = pl.maximum(maximum, sinks)
                final_delta = pl.sub(maximum, final_max)
                correction = pl.exp(final_delta)
                sink_delta = pl.sub(sinks, final_max)
                sink_weight = pl.exp(sink_delta)
                corrected_total = pl.mul(denominator, correction)
                total = pl.add(corrected_total, sink_weight)
                ratio = pl.div(correction, total)
                multiplier = pl.reshape(ratio, [SPARSE_VF_CUBE_HEAD_TILE, 1])
                normalized = pl.row_expand_mul(numerator, multiplier)
                result = pl.cast(normalized, pl.BF16, mode="rint")
                oflat[q0:q0 + SPARSE_VF_HEAD_TILE, :] = result[:SPARSE_VF_HEAD_TILE, :]
        attended = swa_attended
    return attended


@pl.jit.incore(level=pl.Level.AIC)
def sparse_vf_fused_cube(qflat: pl.Tensor, window_indices: pl.Tensor, compressed_indices: pl.Tensor, window_flat: pl.Tensor, compressed_flat: pl.Tensor, sink_row: pl.Tensor, oflat: pl.Out[pl.Tensor], num_tokens: pl.Scalar[pl.INT32], compressed_keys: pl.constexpr):
    pl.func_attr({'split_aiv': True})
    keys = WINDOW_KEYS + compressed_keys
    _parts = keys // SPARSE_VF_KEY_TILE
    workers = pl.min(num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE), SPARSE_VF_WORKER_TILE)
    workers = pl.max(1, workers)
    if compressed_keys > 0:
        kv_ring = pl.system.reserve_buffer(name='sparse_kv_ring', size=262144)
    else:
        kv_ring = pl.system.reserve_buffer(name='sparse_kv_ring', size=393216)
    p_ring = pl.system.reserve_buffer(name='sparse_p_ring', size=16384)
    pv_peer = pl.system.import_peer_buffer(name='sparse_qk_ring', peer_func='sparse_vf_fused_vector')
    if compressed_keys > 0:
        pl.system.aic_initialize_pipe(pv_peer, dir_mask=1, slot_size=8192, slot_num=2, id=0)
    else:
        pl.system.aic_initialize_pipe(pv_peer, dir_mask=1, slot_size=16384, slot_num=1, id=0)
    if compressed_keys > 0:
        pl.system.aic_initialize_pipe(v2c_consumer_buf=kv_ring, dir_mask=2, slot_size=65536, slot_num=4, id=1)
    else:
        pl.system.aic_initialize_pipe(v2c_consumer_buf=kv_ring, dir_mask=2, slot_size=131072, slot_num=3, id=1)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=p_ring, dir_mask=2, slot_size=16384, slot_num=1, id=2)
    pv_peer_next = pl.system.import_peer_buffer(name='sparse_pv_ring', peer_func='sparse_vf_fused_vector')
    if compressed_keys > 0:
        pl.system.aic_initialize_pipe(pv_peer_next, dir_mask=1, slot_size=16384, slot_num=3, id=3)
    else:
        pl.system.aic_initialize_pipe(pv_peer_next, dir_mask=1, slot_size=16384, slot_num=1, id=3)
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE), workers):
        token = block // (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE)
        head = block % (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE) * SPARSE_VF_FUSED_HEAD_TILE
        q0 = token * LOCAL_H + head
        q = pl.tile.load(qflat, [q0, 0], [SPARSE_VF_FUSED_CUBE_HEAD_TILE, HEAD_DIM], valid_shape=[SPARSE_VF_FUSED_HEAD_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
        if compressed_keys > 0:
            kv_0_a = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_0_a = pl.tile.transpose_view(kv_0_a)
            qk_0_a = pl.tile.matmul(q, kt_0_a)
            pl.system.tpush_to_aiv(qk_0_a, split=1, id=0)
            kv_0_b = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_0_b = pl.tile.transpose_view(kv_0_b)
            qk_0_b = pl.tile.matmul(q, kt_0_b)
            pl.system.tpush_to_aiv(qk_0_b, split=1, id=0)
            kv_1_a = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_1_a = pl.tile.transpose_view(kv_1_a)
            qk_1_a = pl.tile.matmul(q, kt_1_a)
            pl.system.tpush_to_aiv(qk_1_a, split=1, id=0)
            kv_1_b = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_1_b = pl.tile.transpose_view(kv_1_b)
            qk_1_b = pl.tile.matmul(q, kt_1_b)
            pl.system.tpush_to_aiv(qk_1_b, split=1, id=0)
            weights_0 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            weights_0_a = pl.tile.slice(weights_0, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, 0])
            weights_0_b = pl.tile.slice(weights_0, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, SPARSE_VF_KEY_TILE // 2])
            value_0_0_a = pl.tile.slice(kv_0_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            value_0_0_b = pl.tile.slice(kv_0_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            pv_0_0 = pl.matmul(weights_0_a, value_0_0_a, out_dtype=pl.FP32)
            pv_0_0 = pl.tile.matmul_acc(pv_0_0, weights_0_b, value_0_0_b)
            value_0_1_a = pl.tile.slice(kv_0_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            value_0_1_b = pl.tile.slice(kv_0_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            pv_0_1 = pl.matmul(weights_0_a, value_0_1_a, out_dtype=pl.FP32)
            pv_0_1 = pl.tile.matmul_acc(pv_0_1, weights_0_b, value_0_1_b)
            value_0_2_a = pl.tile.slice(kv_0_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            value_0_2_b = pl.tile.slice(kv_0_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            pv_0_2 = pl.matmul(weights_0_a, value_0_2_a, out_dtype=pl.FP32)
            pv_0_2 = pl.tile.matmul_acc(pv_0_2, weights_0_b, value_0_2_b)
            value_0_3_a = pl.tile.slice(kv_0_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            value_0_3_b = pl.tile.slice(kv_0_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            pv_0_3 = pl.matmul(weights_0_a, value_0_3_a, out_dtype=pl.FP32)
            pv_0_3 = pl.tile.matmul_acc(pv_0_3, weights_0_b, value_0_3_b)
            pl.system.tfree_to_aiv(weights_0, id=2)
            pl.system.tfree_to_aiv(kv_0_a, id=1)
            pl.system.tfree_to_aiv(kv_0_b, id=1)
            pl.system.tpush_to_aiv(pv_0_0, split=1, id=3)
            pl.system.tpush_to_aiv(pv_0_1, split=1, id=3)
            pl.system.tpush_to_aiv(pv_0_2, split=1, id=3)
            pl.system.tpush_to_aiv(pv_0_3, split=1, id=3)
            kv_2_a = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_2_a = pl.tile.transpose_view(kv_2_a)
            qk_2_a = pl.tile.matmul(q, kt_2_a)
            pl.system.tpush_to_aiv(qk_2_a, split=1, id=0)
            kv_2_b = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_2_b = pl.tile.transpose_view(kv_2_b)
            qk_2_b = pl.tile.matmul(q, kt_2_b)
            pl.system.tpush_to_aiv(qk_2_b, split=1, id=0)
            weights_1 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            weights_1_a = pl.tile.slice(weights_1, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, 0])
            weights_1_b = pl.tile.slice(weights_1, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, SPARSE_VF_KEY_TILE // 2])
            value_1_0_a = pl.tile.slice(kv_1_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            value_1_0_b = pl.tile.slice(kv_1_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            pv_1_0 = pl.matmul(weights_1_a, value_1_0_a, out_dtype=pl.FP32)
            pv_1_0 = pl.tile.matmul_acc(pv_1_0, weights_1_b, value_1_0_b)
            value_1_1_a = pl.tile.slice(kv_1_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            value_1_1_b = pl.tile.slice(kv_1_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            pv_1_1 = pl.matmul(weights_1_a, value_1_1_a, out_dtype=pl.FP32)
            pv_1_1 = pl.tile.matmul_acc(pv_1_1, weights_1_b, value_1_1_b)
            value_1_2_a = pl.tile.slice(kv_1_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            value_1_2_b = pl.tile.slice(kv_1_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            pv_1_2 = pl.matmul(weights_1_a, value_1_2_a, out_dtype=pl.FP32)
            pv_1_2 = pl.tile.matmul_acc(pv_1_2, weights_1_b, value_1_2_b)
            value_1_3_a = pl.tile.slice(kv_1_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            value_1_3_b = pl.tile.slice(kv_1_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            pv_1_3 = pl.matmul(weights_1_a, value_1_3_a, out_dtype=pl.FP32)
            pv_1_3 = pl.tile.matmul_acc(pv_1_3, weights_1_b, value_1_3_b)
            pl.system.tfree_to_aiv(weights_1, id=2)
            pl.system.tfree_to_aiv(kv_1_a, id=1)
            pl.system.tfree_to_aiv(kv_1_b, id=1)
            pl.system.tpush_to_aiv(pv_1_0, split=1, id=3)
            pl.system.tpush_to_aiv(pv_1_1, split=1, id=3)
            pl.system.tpush_to_aiv(pv_1_2, split=1, id=3)
            pl.system.tpush_to_aiv(pv_1_3, split=1, id=3)
            kv_3_a = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_3_a = pl.tile.transpose_view(kv_3_a)
            qk_3_a = pl.tile.matmul(q, kt_3_a)
            pl.system.tpush_to_aiv(qk_3_a, split=1, id=0)
            kv_3_b = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_3_b = pl.tile.transpose_view(kv_3_b)
            qk_3_b = pl.tile.matmul(q, kt_3_b)
            pl.system.tpush_to_aiv(qk_3_b, split=1, id=0)
            weights_2 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            weights_2_a = pl.tile.slice(weights_2, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, 0])
            weights_2_b = pl.tile.slice(weights_2, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, SPARSE_VF_KEY_TILE // 2])
            value_2_0_a = pl.tile.slice(kv_2_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            value_2_0_b = pl.tile.slice(kv_2_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            pv_2_0 = pl.matmul(weights_2_a, value_2_0_a, out_dtype=pl.FP32)
            pv_2_0 = pl.tile.matmul_acc(pv_2_0, weights_2_b, value_2_0_b)
            value_2_1_a = pl.tile.slice(kv_2_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            value_2_1_b = pl.tile.slice(kv_2_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            pv_2_1 = pl.matmul(weights_2_a, value_2_1_a, out_dtype=pl.FP32)
            pv_2_1 = pl.tile.matmul_acc(pv_2_1, weights_2_b, value_2_1_b)
            value_2_2_a = pl.tile.slice(kv_2_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            value_2_2_b = pl.tile.slice(kv_2_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            pv_2_2 = pl.matmul(weights_2_a, value_2_2_a, out_dtype=pl.FP32)
            pv_2_2 = pl.tile.matmul_acc(pv_2_2, weights_2_b, value_2_2_b)
            value_2_3_a = pl.tile.slice(kv_2_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            value_2_3_b = pl.tile.slice(kv_2_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            pv_2_3 = pl.matmul(weights_2_a, value_2_3_a, out_dtype=pl.FP32)
            pv_2_3 = pl.tile.matmul_acc(pv_2_3, weights_2_b, value_2_3_b)
            pl.system.tfree_to_aiv(weights_2, id=2)
            pl.system.tfree_to_aiv(kv_2_a, id=1)
            pl.system.tfree_to_aiv(kv_2_b, id=1)
            pl.system.tpush_to_aiv(pv_2_0, split=1, id=3)
            pl.system.tpush_to_aiv(pv_2_1, split=1, id=3)
            pl.system.tpush_to_aiv(pv_2_2, split=1, id=3)
            pl.system.tpush_to_aiv(pv_2_3, split=1, id=3)
            kv_4_a = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_4_a = pl.tile.transpose_view(kv_4_a)
            qk_4_a = pl.tile.matmul(q, kt_4_a)
            pl.system.tpush_to_aiv(qk_4_a, split=1, id=0)
            kv_4_b = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_4_b = pl.tile.transpose_view(kv_4_b)
            qk_4_b = pl.tile.matmul(q, kt_4_b)
            pl.system.tpush_to_aiv(qk_4_b, split=1, id=0)
            weights_3 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            weights_3_a = pl.tile.slice(weights_3, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, 0])
            weights_3_b = pl.tile.slice(weights_3, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, SPARSE_VF_KEY_TILE // 2])
            value_3_0_a = pl.tile.slice(kv_3_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            value_3_0_b = pl.tile.slice(kv_3_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            pv_3_0 = pl.matmul(weights_3_a, value_3_0_a, out_dtype=pl.FP32)
            pv_3_0 = pl.tile.matmul_acc(pv_3_0, weights_3_b, value_3_0_b)
            value_3_1_a = pl.tile.slice(kv_3_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            value_3_1_b = pl.tile.slice(kv_3_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            pv_3_1 = pl.matmul(weights_3_a, value_3_1_a, out_dtype=pl.FP32)
            pv_3_1 = pl.tile.matmul_acc(pv_3_1, weights_3_b, value_3_1_b)
            value_3_2_a = pl.tile.slice(kv_3_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            value_3_2_b = pl.tile.slice(kv_3_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            pv_3_2 = pl.matmul(weights_3_a, value_3_2_a, out_dtype=pl.FP32)
            pv_3_2 = pl.tile.matmul_acc(pv_3_2, weights_3_b, value_3_2_b)
            value_3_3_a = pl.tile.slice(kv_3_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            value_3_3_b = pl.tile.slice(kv_3_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            pv_3_3 = pl.matmul(weights_3_a, value_3_3_a, out_dtype=pl.FP32)
            pv_3_3 = pl.tile.matmul_acc(pv_3_3, weights_3_b, value_3_3_b)
            pl.system.tfree_to_aiv(weights_3, id=2)
            pl.system.tfree_to_aiv(kv_3_a, id=1)
            pl.system.tfree_to_aiv(kv_3_b, id=1)
            pl.system.tpush_to_aiv(pv_3_0, split=1, id=3)
            pl.system.tpush_to_aiv(pv_3_1, split=1, id=3)
            pl.system.tpush_to_aiv(pv_3_2, split=1, id=3)
            pl.system.tpush_to_aiv(pv_3_3, split=1, id=3)
            weights_4 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            weights_4_a = pl.tile.slice(weights_4, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, 0])
            weights_4_b = pl.tile.slice(weights_4, [SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], [0, SPARSE_VF_KEY_TILE // 2])
            value_4_0_a = pl.tile.slice(kv_4_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            value_4_0_b = pl.tile.slice(kv_4_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 0])
            pv_4_0 = pl.matmul(weights_4_a, value_4_0_a, out_dtype=pl.FP32)
            pv_4_0 = pl.tile.matmul_acc(pv_4_0, weights_4_b, value_4_0_b)
            value_4_1_a = pl.tile.slice(kv_4_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            value_4_1_b = pl.tile.slice(kv_4_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 128])
            pv_4_1 = pl.matmul(weights_4_a, value_4_1_a, out_dtype=pl.FP32)
            pv_4_1 = pl.tile.matmul_acc(pv_4_1, weights_4_b, value_4_1_b)
            value_4_2_a = pl.tile.slice(kv_4_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            value_4_2_b = pl.tile.slice(kv_4_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 256])
            pv_4_2 = pl.matmul(weights_4_a, value_4_2_a, out_dtype=pl.FP32)
            pv_4_2 = pl.tile.matmul_acc(pv_4_2, weights_4_b, value_4_2_b)
            value_4_3_a = pl.tile.slice(kv_4_a, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            value_4_3_b = pl.tile.slice(kv_4_b, [SPARSE_VF_KEY_TILE // 2, 128], [0, 384])
            pv_4_3 = pl.matmul(weights_4_a, value_4_3_a, out_dtype=pl.FP32)
            pv_4_3 = pl.tile.matmul_acc(pv_4_3, weights_4_b, value_4_3_b)
            pl.system.tfree_to_aiv(weights_4, id=2)
            pl.system.tfree_to_aiv(kv_4_a, id=1)
            pl.system.tfree_to_aiv(kv_4_b, id=1)
            pl.system.tpush_to_aiv(pv_4_0, split=1, id=3)
            pl.system.tpush_to_aiv(pv_4_1, split=1, id=3)
            pl.system.tpush_to_aiv(pv_4_2, split=1, id=3)
            pl.system.tpush_to_aiv(pv_4_3, split=1, id=3)
        else:
            kv_0 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_KEY_TILE, HEAD_DIM], dtype=pl.BF16, split=1, id=1)
            kt_0 = pl.tile.transpose_view(kv_0)
            qk_0 = pl.tile.matmul(q, kt_0)
            pl.system.tpush_to_aiv(qk_0, split=1, id=0)
            weights_0 = pl.system.tpop_from_aiv(shape=[SPARSE_VF_FUSED_CUBE_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, split=1, id=2)
            for value_column in pl.range(0, HEAD_DIM, 128):
                value_0 = pl.tile.slice(kv_0, [SPARSE_VF_KEY_TILE, 128], [0, value_column])
                pv_0 = pl.matmul(weights_0, value_0, out_dtype=pl.FP32)
                pl.system.tpush_to_aiv(pv_0, split=1, id=3)
            pl.system.tfree_to_aiv(weights_0, id=2)
            pl.system.tfree_to_aiv(kv_0, id=1)
    return oflat


@pl.jit.incore(level=pl.Level.AIV)
def sparse_vf_fused_vector(qflat: pl.Tensor, window_indices: pl.Tensor, compressed_indices: pl.Tensor, window_flat: pl.Tensor, compressed_flat: pl.Tensor, sink_row: pl.Tensor, oflat: pl.Out[pl.Tensor], num_tokens: pl.Scalar[pl.INT32], compressed_keys: pl.constexpr):
    pl.func_attr({'split_aiv': True, 'dual_aiv_dispatch': True})
    keys = WINDOW_KEYS + compressed_keys
    parts = keys // SPARSE_VF_KEY_TILE
    workers = pl.min(num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE), SPARSE_VF_WORKER_TILE)
    workers = pl.max(1, workers)
    aiv_id = pl.tile.get_subblock_idx()
    pv_ring = pl.system.reserve_buffer(name='sparse_qk_ring', size=16384)
    kv_peer = pl.system.import_peer_buffer(name='sparse_kv_ring', peer_func='sparse_vf_fused_cube')
    p_peer = pl.system.import_peer_buffer(name='sparse_p_ring', peer_func='sparse_vf_fused_cube')
    if compressed_keys > 0:
        pl.system.aiv_initialize_pipe(pv_ring, dir_mask=1, slot_size=8192, slot_num=2, id=0)
    else:
        pl.system.aiv_initialize_pipe(pv_ring, dir_mask=1, slot_size=16384, slot_num=1, id=0)
    if compressed_keys > 0:
        pl.system.aiv_initialize_pipe(v2c_consumer_buf=kv_peer, dir_mask=2, slot_size=65536, slot_num=4, id=1)
    else:
        pl.system.aiv_initialize_pipe(v2c_consumer_buf=kv_peer, dir_mask=2, slot_size=131072, slot_num=3, id=1)
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=p_peer, dir_mask=2, slot_size=16384, slot_num=1, id=2)
    if compressed_keys > 0:
        pv_ring_next = pl.system.reserve_buffer(name='sparse_pv_ring', size=49152)
        pl.system.aiv_initialize_pipe(pv_ring_next, dir_mask=1, slot_size=16384, slot_num=3, id=3)
    else:
        pv_ring_next = pl.system.reserve_buffer(name='sparse_pv_ring', size=16384)
        pl.system.aiv_initialize_pipe(pv_ring_next, dir_mask=1, slot_size=16384, slot_num=1, id=3)
    worker = pl.tile.get_block_idx()
    zero_raw_window = pl.tile.full([1, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
    zero_raw_compressed = pl.tile.slice(zero_raw_window, [1, COMPRESSED_CACHE_BYTES], [0, 0])
    for block in pl.range(worker, num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE), workers):
        token = block // (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE)
        head = block % (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE) * SPARSE_VF_FUSED_HEAD_TILE
        q0 = token * LOCAL_H + head
        maximum = pl.tile.full([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, value=-1e+30)
        denominator = pl.tile.full([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, value=0.0)
        numerator_0 = pl.tile.full([SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, value=0.0)
        numerator_1 = pl.tile.full([SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, value=0.0)
        numerator_2 = pl.tile.full([SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, value=0.0)
        numerator_3 = pl.tile.full([SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, value=0.0)
        if compressed_keys > 0:
            if block == worker or parts < 3:
                ids_seed0: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[0], pl.Mem.Vec] = pl.tile.load(window_indices, [token, 0 * 64 + aiv_id * 32], [1, 32])
                raw_seed0_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                raw_seed0 = pl.tile.reshape(raw_seed0_storage, [32, WINDOW_CACHE_BYTES])
                seed0_first = pl.tile.read(ids_seed0, [0, 0])
                seed0_contiguous = pl.cast(0, pl.INT32)
                seed0_all_valid = pl.cast(0, pl.INT32)
                if seed0_first >= 0:
                    seed0_all_valid = pl.cast(1, pl.INT32)
                    seed0_contiguous = pl.cast(1, pl.INT32)
                    for seed0_check in pl.range(1, 32):
                        seed0_index = pl.tile.read(ids_seed0, [0, seed0_check])
                        if seed0_index != seed0_first + seed0_check:
                            seed0_contiguous = pl.cast(0, pl.INT32)
                        if seed0_index < 0:
                            seed0_all_valid = pl.cast(0, pl.INT32)
                if seed0_contiguous != 0:
                    seed0_physical = pl.cast(seed0_first, pl.INDEX)
                    raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [0, 0], [seed0_physical, 0], [32, WINDOW_CACHE_BYTES])
                elif seed0_all_valid != 0:
                    for seed0_lane in pl.range(0, 32, 2):
                        seed0_a = pl.tile.read(ids_seed0, [0, seed0_lane])
                        seed0_b = pl.tile.read(ids_seed0, [0, seed0_lane + 1])
                        seed0_lower = pl.min(seed0_a, seed0_b)
                        seed0_upper = pl.max(seed0_a, seed0_b)
                        seed0_pair = pl.cast(0, pl.INT32)
                        if seed0_upper == seed0_lower + 1:
                            seed0_pair = pl.cast(1, pl.INT32)
                        if seed0_pair != 0:
                            seed0_physical_pair = pl.cast(seed0_lower, pl.INDEX)
                            raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane, 0], [seed0_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                        else:
                            seed0_physical_a = pl.cast(seed0_a, pl.INDEX)
                            raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane, 0], [seed0_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                            seed0_physical_b = pl.cast(seed0_b, pl.INDEX)
                            raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane + 1, 0], [seed0_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                else:
                    for seed0_lane in pl.range(0, 32, 2):
                        seed0_a = pl.tile.read(ids_seed0, [0, seed0_lane])
                        seed0_b = pl.tile.read(ids_seed0, [0, seed0_lane + 1])
                        seed0_lower = pl.min(seed0_a, seed0_b)
                        seed0_upper = pl.max(seed0_a, seed0_b)
                        seed0_pair = pl.cast(0, pl.INT32)
                        if seed0_lower >= 0:
                            if seed0_upper == seed0_lower + 1:
                                seed0_pair = pl.cast(1, pl.INT32)
                        if seed0_pair != 0:
                            seed0_physical_pair = pl.cast(seed0_lower, pl.INDEX)
                            raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane, 0], [seed0_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                        else:
                            if seed0_a >= 0:
                                seed0_physical_a = pl.cast(seed0_a, pl.INDEX)
                                raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane, 0], [seed0_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                            else:
                                raw_seed0 = pl.tile.assemble(raw_seed0, zero_raw_window, [seed0_lane, 0])
                            if seed0_b >= 0:
                                seed0_physical_b = pl.cast(seed0_b, pl.INDEX)
                                raw_seed0 = pl.tile.gather_row(raw_seed0, window_flat, [seed0_lane + 1, 0], [seed0_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                            else:
                                raw_seed0 = pl.tile.assemble(raw_seed0, zero_raw_window, [seed0_lane + 1, 0])
            ids_seed1: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[1], pl.Mem.Vec] = pl.tile.load(window_indices, [token, 1 * 64 + aiv_id * 32], [1, 32])
            raw_seed1_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
            raw_seed1 = pl.tile.reshape(raw_seed1_storage, [32, WINDOW_CACHE_BYTES])
            seed1_first = pl.tile.read(ids_seed1, [0, 0])
            seed1_contiguous = pl.cast(0, pl.INT32)
            seed1_all_valid = pl.cast(0, pl.INT32)
            if seed1_first >= 0:
                seed1_all_valid = pl.cast(1, pl.INT32)
                seed1_contiguous = pl.cast(1, pl.INT32)
                for seed1_check in pl.range(1, 32):
                    seed1_index = pl.tile.read(ids_seed1, [0, seed1_check])
                    if seed1_index != seed1_first + seed1_check:
                        seed1_contiguous = pl.cast(0, pl.INT32)
                    if seed1_index < 0:
                        seed1_all_valid = pl.cast(0, pl.INT32)
            if seed1_contiguous != 0:
                seed1_physical = pl.cast(seed1_first, pl.INDEX)
                raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [0, 0], [seed1_physical, 0], [32, WINDOW_CACHE_BYTES])
            elif seed1_all_valid != 0:
                for seed1_lane in pl.range(0, 32, 2):
                    seed1_a = pl.tile.read(ids_seed1, [0, seed1_lane])
                    seed1_b = pl.tile.read(ids_seed1, [0, seed1_lane + 1])
                    seed1_lower = pl.min(seed1_a, seed1_b)
                    seed1_upper = pl.max(seed1_a, seed1_b)
                    seed1_pair = pl.cast(0, pl.INT32)
                    if seed1_upper == seed1_lower + 1:
                        seed1_pair = pl.cast(1, pl.INT32)
                    if seed1_pair != 0:
                        seed1_physical_pair = pl.cast(seed1_lower, pl.INDEX)
                        raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane, 0], [seed1_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                    else:
                        seed1_physical_a = pl.cast(seed1_a, pl.INDEX)
                        raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane, 0], [seed1_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                        seed1_physical_b = pl.cast(seed1_b, pl.INDEX)
                        raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane + 1, 0], [seed1_physical_b, 0], [1, WINDOW_CACHE_BYTES])
            else:
                for seed1_lane in pl.range(0, 32, 2):
                    seed1_a = pl.tile.read(ids_seed1, [0, seed1_lane])
                    seed1_b = pl.tile.read(ids_seed1, [0, seed1_lane + 1])
                    seed1_lower = pl.min(seed1_a, seed1_b)
                    seed1_upper = pl.max(seed1_a, seed1_b)
                    seed1_pair = pl.cast(0, pl.INT32)
                    if seed1_lower >= 0:
                        if seed1_upper == seed1_lower + 1:
                            seed1_pair = pl.cast(1, pl.INT32)
                    if seed1_pair != 0:
                        seed1_physical_pair = pl.cast(seed1_lower, pl.INDEX)
                        raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane, 0], [seed1_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                    else:
                        if seed1_a >= 0:
                            seed1_physical_a = pl.cast(seed1_a, pl.INDEX)
                            raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane, 0], [seed1_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                        else:
                            raw_seed1 = pl.tile.assemble(raw_seed1, zero_raw_window, [seed1_lane, 0])
                        if seed1_b >= 0:
                            seed1_physical_b = pl.cast(seed1_b, pl.INDEX)
                            raw_seed1 = pl.tile.gather_row(raw_seed1, window_flat, [seed1_lane + 1, 0], [seed1_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        else:
                            raw_seed1 = pl.tile.assemble(raw_seed1, zero_raw_window, [seed1_lane + 1, 0])
            if 0 + 2 < parts * 2:
                ids_future_seed0: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[2], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (0 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                raw_future_seed0_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                raw_future_seed0_prefix = pl.tile.slice(raw_future_seed0_storage, [1, 10240], [0, 0])
                raw_future_seed0 = pl.tile.reshape(raw_future_seed0_prefix, [32, COMPRESSED_CACHE_BYTES])
                future_seed0_first = pl.tile.read(ids_future_seed0, [0, 0])
                future_seed0_contiguous = pl.cast(0, pl.INT32)
                future_seed0_all_valid = pl.cast(0, pl.INT32)
                if future_seed0_first >= 0:
                    # The OR of XOR differences is zero for a contiguous run and negative for an invalid index.
                    future_seed0_first_wide = pl.cast(future_seed0_first, pl.INT64)
                    future_seed0_mismatch = pl.cast(0, pl.INT64)
                    for future_seed0_check in pl.range(1, 32):
                        future_seed0_index = pl.tile.read(ids_future_seed0, [0, future_seed0_check])
                        future_seed0_expected = future_seed0_first_wide + pl.cast(future_seed0_check, pl.INT64)
                        future_seed0_delta = pl.cast(future_seed0_index, pl.INT64) ^ future_seed0_expected
                        future_seed0_mismatch = future_seed0_mismatch | future_seed0_delta
                    future_seed0_contiguous = pl.cast(future_seed0_mismatch == 0, pl.INT32)
                    future_seed0_all_valid = pl.cast(future_seed0_mismatch >= 0, pl.INT32)
                if future_seed0_contiguous != 0:
                    future_seed0_physical = pl.cast(future_seed0_first, pl.INDEX)
                    raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [0, 0], [future_seed0_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                elif future_seed0_all_valid != 0:
                    for future_seed0_lane in pl.range(0, 32, 2):
                        future_seed0_a = pl.tile.read(ids_future_seed0, [0, future_seed0_lane])
                        future_seed0_b = pl.tile.read(ids_future_seed0, [0, future_seed0_lane + 1])
                        future_seed0_lower = pl.min(future_seed0_a, future_seed0_b)
                        future_seed0_upper = pl.max(future_seed0_a, future_seed0_b)
                        future_seed0_pair = pl.cast(0, pl.INT32)
                        if future_seed0_upper == future_seed0_lower + 1:
                            future_seed0_pair = pl.cast(1, pl.INT32)
                        if future_seed0_pair != 0:
                            future_seed0_physical_pair = pl.cast(future_seed0_lower, pl.INDEX)
                            raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane, 0], [future_seed0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            future_seed0_physical_a = pl.cast(future_seed0_a, pl.INDEX)
                            raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane, 0], [future_seed0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            future_seed0_physical_b = pl.cast(future_seed0_b, pl.INDEX)
                            raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane + 1, 0], [future_seed0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                else:
                    for future_seed0_lane in pl.range(0, 32, 2):
                        future_seed0_a = pl.tile.read(ids_future_seed0, [0, future_seed0_lane])
                        future_seed0_b = pl.tile.read(ids_future_seed0, [0, future_seed0_lane + 1])
                        future_seed0_lower = pl.min(future_seed0_a, future_seed0_b)
                        future_seed0_upper = pl.max(future_seed0_a, future_seed0_b)
                        future_seed0_pair = pl.cast(0, pl.INT32)
                        if future_seed0_lower >= 0:
                            if future_seed0_upper == future_seed0_lower + 1:
                                future_seed0_pair = pl.cast(1, pl.INT32)
                        if future_seed0_pair != 0:
                            future_seed0_physical_pair = pl.cast(future_seed0_lower, pl.INDEX)
                            raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane, 0], [future_seed0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            if future_seed0_a >= 0:
                                future_seed0_physical_a = pl.cast(future_seed0_a, pl.INDEX)
                                raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane, 0], [future_seed0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed0 = pl.tile.assemble(raw_future_seed0, zero_raw_compressed, [future_seed0_lane, 0])
                            if future_seed0_b >= 0:
                                future_seed0_physical_b = pl.cast(future_seed0_b, pl.INDEX)
                                raw_future_seed0 = pl.tile.gather_row(raw_future_seed0, compressed_flat, [future_seed0_lane + 1, 0], [future_seed0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed0 = pl.tile.assemble(raw_future_seed0, zero_raw_compressed, [future_seed0_lane + 1, 0])
            kv_chunk_seed0: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            raw_current_window0_seed0: pl.Tile[[32, WINDOW_CACHE_BYTES], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([32, WINDOW_CACHE_BYTES], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
            fwa_raw_chunk_slot0_seed0 = pl.tile.slice(raw_current_window0_seed0, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [0, 0])
            fwa_signed_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.INT8)
            fwa_packed_slot0_seed0 = pl.reinterpret_view(fwa_signed_slot0_seed0, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot0_seed0 = fwa_packed_slot0_seed0[:, :HEAD_DIM]
            fwa_values_fp32_slot0_seed0 = pl.cast(fwa_codes_slot0_seed0, pl.FP32)
            fwa_values_slot0_seed0 = pl.cast(fwa_values_fp32_slot0_seed0, pl.BF16, mode='rint')
            fwa_scale_words_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot0_seed0 = pl.tile.reshape(fwa_scale_words_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot0_seed0 = pl.tile.reshape(fwa_values_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot0_seed0 = pl.tile.reshape(kv_chunk_seed0, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_0_fwa_all_r0_slot0_seed0 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_0_fwa_row_r0_slot0_seed0 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_0_fwa_chunk_r0_slot0_seed0 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_0_fwa_scale_offset_r0_slot0_seed0 = raw_stream_0_fwa_row_r0_slot0_seed0 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_0_fwa_scale_offset_r0_slot0_seed0 = raw_stream_0_fwa_scale_offset_r0_slot0_seed0 + raw_stream_0_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_0_fwa_groups_r0_slot0_seed0 = pl.reg.load(fwa_scale_flat_slot0_seed0, raw_stream_0_fwa_scale_offset_r0_slot0_seed0, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_0_fwa_factor_bits_r0_slot0_seed0 = pl.reg.group_broadcast(raw_stream_0_fwa_groups_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_0_fwa_factor_r0_slot0_seed0 = pl.reg.interpret_cast(raw_stream_0_fwa_factor_bits_r0_slot0_seed0, dtype=pl.BF16)
                        raw_stream_0_fwa_value_offset_r0_slot0_seed0 = raw_stream_0_fwa_row_r0_slot0_seed0 * HEAD_DIM + raw_stream_0_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_REG_TILE
                        raw_stream_0_fwa_value_r0_slot0_seed0 = pl.reg.load(fwa_value_flat_slot0_seed0, raw_stream_0_fwa_value_offset_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_0_fwa_scaled_r0_slot0_seed0 = pl.reg.mul(raw_stream_0_fwa_value_r0_slot0_seed0, raw_stream_0_fwa_factor_r0_slot0_seed0, raw_stream_0_fwa_all_r0_slot0_seed0)
                        raw_stream_0_fwa_target_row_r0_slot0_seed0 = 0 + 0 + raw_stream_0_fwa_row_r0_slot0_seed0
                        raw_stream_0_fwa_target_offset_r0_slot0_seed0 = raw_stream_0_fwa_target_row_r0_slot0_seed0 * 16 + raw_stream_0_fwa_chunk_r0_slot0_seed0 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_0_fwa_write_r0_slot0_seed0 = pl.reg.store(fwa_kv_flat_slot0_seed0, raw_stream_0_fwa_target_offset_r0_slot0_seed0, raw_stream_0_fwa_scaled_r0_slot0_seed0, raw_stream_0_fwa_all_r0_slot0_seed0, block_stride=32)
            fwa_raw_chunk_slot0_seed0 = pl.tile.slice(raw_current_window0_seed0, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [8, 0])
            fwa_signed_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.INT8)
            fwa_packed_slot0_seed0 = pl.reinterpret_view(fwa_signed_slot0_seed0, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot0_seed0 = fwa_packed_slot0_seed0[:, :HEAD_DIM]
            fwa_values_fp32_slot0_seed0 = pl.cast(fwa_codes_slot0_seed0, pl.FP32)
            fwa_values_slot0_seed0 = pl.cast(fwa_values_fp32_slot0_seed0, pl.BF16, mode='rint')
            fwa_scale_words_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot0_seed0 = pl.tile.reshape(fwa_scale_words_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot0_seed0 = pl.tile.reshape(fwa_values_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot0_seed0 = pl.tile.reshape(kv_chunk_seed0, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_1_fwa_all_r0_slot0_seed0 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_1_fwa_row_r0_slot0_seed0 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_1_fwa_chunk_r0_slot0_seed0 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_1_fwa_scale_offset_r0_slot0_seed0 = raw_stream_1_fwa_row_r0_slot0_seed0 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_1_fwa_scale_offset_r0_slot0_seed0 = raw_stream_1_fwa_scale_offset_r0_slot0_seed0 + raw_stream_1_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_1_fwa_groups_r0_slot0_seed0 = pl.reg.load(fwa_scale_flat_slot0_seed0, raw_stream_1_fwa_scale_offset_r0_slot0_seed0, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_1_fwa_factor_bits_r0_slot0_seed0 = pl.reg.group_broadcast(raw_stream_1_fwa_groups_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_1_fwa_factor_r0_slot0_seed0 = pl.reg.interpret_cast(raw_stream_1_fwa_factor_bits_r0_slot0_seed0, dtype=pl.BF16)
                        raw_stream_1_fwa_value_offset_r0_slot0_seed0 = raw_stream_1_fwa_row_r0_slot0_seed0 * HEAD_DIM + raw_stream_1_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_REG_TILE
                        raw_stream_1_fwa_value_r0_slot0_seed0 = pl.reg.load(fwa_value_flat_slot0_seed0, raw_stream_1_fwa_value_offset_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_1_fwa_scaled_r0_slot0_seed0 = pl.reg.mul(raw_stream_1_fwa_value_r0_slot0_seed0, raw_stream_1_fwa_factor_r0_slot0_seed0, raw_stream_1_fwa_all_r0_slot0_seed0)
                        raw_stream_1_fwa_target_row_r0_slot0_seed0 = 0 + 8 + raw_stream_1_fwa_row_r0_slot0_seed0
                        raw_stream_1_fwa_target_offset_r0_slot0_seed0 = raw_stream_1_fwa_target_row_r0_slot0_seed0 * 16 + raw_stream_1_fwa_chunk_r0_slot0_seed0 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_1_fwa_write_r0_slot0_seed0 = pl.reg.store(fwa_kv_flat_slot0_seed0, raw_stream_1_fwa_target_offset_r0_slot0_seed0, raw_stream_1_fwa_scaled_r0_slot0_seed0, raw_stream_1_fwa_all_r0_slot0_seed0, block_stride=32)
            fwa_raw_chunk_slot0_seed0 = pl.tile.slice(raw_current_window0_seed0, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [16, 0])
            fwa_signed_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.INT8)
            fwa_packed_slot0_seed0 = pl.reinterpret_view(fwa_signed_slot0_seed0, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot0_seed0 = fwa_packed_slot0_seed0[:, :HEAD_DIM]
            fwa_values_fp32_slot0_seed0 = pl.cast(fwa_codes_slot0_seed0, pl.FP32)
            fwa_values_slot0_seed0 = pl.cast(fwa_values_fp32_slot0_seed0, pl.BF16, mode='rint')
            fwa_scale_words_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot0_seed0 = pl.tile.reshape(fwa_scale_words_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot0_seed0 = pl.tile.reshape(fwa_values_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot0_seed0 = pl.tile.reshape(kv_chunk_seed0, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_2_fwa_all_r0_slot0_seed0 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_2_fwa_row_r0_slot0_seed0 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_2_fwa_chunk_r0_slot0_seed0 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_2_fwa_scale_offset_r0_slot0_seed0 = raw_stream_2_fwa_row_r0_slot0_seed0 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_2_fwa_scale_offset_r0_slot0_seed0 = raw_stream_2_fwa_scale_offset_r0_slot0_seed0 + raw_stream_2_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_2_fwa_groups_r0_slot0_seed0 = pl.reg.load(fwa_scale_flat_slot0_seed0, raw_stream_2_fwa_scale_offset_r0_slot0_seed0, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_2_fwa_factor_bits_r0_slot0_seed0 = pl.reg.group_broadcast(raw_stream_2_fwa_groups_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_2_fwa_factor_r0_slot0_seed0 = pl.reg.interpret_cast(raw_stream_2_fwa_factor_bits_r0_slot0_seed0, dtype=pl.BF16)
                        raw_stream_2_fwa_value_offset_r0_slot0_seed0 = raw_stream_2_fwa_row_r0_slot0_seed0 * HEAD_DIM + raw_stream_2_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_REG_TILE
                        raw_stream_2_fwa_value_r0_slot0_seed0 = pl.reg.load(fwa_value_flat_slot0_seed0, raw_stream_2_fwa_value_offset_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_2_fwa_scaled_r0_slot0_seed0 = pl.reg.mul(raw_stream_2_fwa_value_r0_slot0_seed0, raw_stream_2_fwa_factor_r0_slot0_seed0, raw_stream_2_fwa_all_r0_slot0_seed0)
                        raw_stream_2_fwa_target_row_r0_slot0_seed0 = 0 + 16 + raw_stream_2_fwa_row_r0_slot0_seed0
                        raw_stream_2_fwa_target_offset_r0_slot0_seed0 = raw_stream_2_fwa_target_row_r0_slot0_seed0 * 16 + raw_stream_2_fwa_chunk_r0_slot0_seed0 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_2_fwa_write_r0_slot0_seed0 = pl.reg.store(fwa_kv_flat_slot0_seed0, raw_stream_2_fwa_target_offset_r0_slot0_seed0, raw_stream_2_fwa_scaled_r0_slot0_seed0, raw_stream_2_fwa_all_r0_slot0_seed0, block_stride=32)
            fwa_raw_chunk_slot0_seed0 = pl.tile.slice(raw_current_window0_seed0, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [24, 0])
            fwa_signed_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.INT8)
            fwa_packed_slot0_seed0 = pl.reinterpret_view(fwa_signed_slot0_seed0, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot0_seed0 = fwa_packed_slot0_seed0[:, :HEAD_DIM]
            fwa_values_fp32_slot0_seed0 = pl.cast(fwa_codes_slot0_seed0, pl.FP32)
            fwa_values_slot0_seed0 = pl.cast(fwa_values_fp32_slot0_seed0, pl.BF16, mode='rint')
            fwa_scale_words_slot0_seed0 = pl.reinterpret_view(fwa_raw_chunk_slot0_seed0, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot0_seed0 = pl.tile.reshape(fwa_scale_words_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot0_seed0 = pl.tile.reshape(fwa_values_slot0_seed0, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot0_seed0 = pl.tile.reshape(kv_chunk_seed0, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_3_fwa_all_r0_slot0_seed0 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_3_fwa_row_r0_slot0_seed0 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_3_fwa_chunk_r0_slot0_seed0 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_3_fwa_scale_offset_r0_slot0_seed0 = raw_stream_3_fwa_row_r0_slot0_seed0 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_3_fwa_scale_offset_r0_slot0_seed0 = raw_stream_3_fwa_scale_offset_r0_slot0_seed0 + raw_stream_3_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_3_fwa_groups_r0_slot0_seed0 = pl.reg.load(fwa_scale_flat_slot0_seed0, raw_stream_3_fwa_scale_offset_r0_slot0_seed0, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_3_fwa_factor_bits_r0_slot0_seed0 = pl.reg.group_broadcast(raw_stream_3_fwa_groups_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_3_fwa_factor_r0_slot0_seed0 = pl.reg.interpret_cast(raw_stream_3_fwa_factor_bits_r0_slot0_seed0, dtype=pl.BF16)
                        raw_stream_3_fwa_value_offset_r0_slot0_seed0 = raw_stream_3_fwa_row_r0_slot0_seed0 * HEAD_DIM + raw_stream_3_fwa_chunk_r0_slot0_seed0 * SPARSE_VF_REG_TILE
                        raw_stream_3_fwa_value_r0_slot0_seed0 = pl.reg.load(fwa_value_flat_slot0_seed0, raw_stream_3_fwa_value_offset_r0_slot0_seed0, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_3_fwa_scaled_r0_slot0_seed0 = pl.reg.mul(raw_stream_3_fwa_value_r0_slot0_seed0, raw_stream_3_fwa_factor_r0_slot0_seed0, raw_stream_3_fwa_all_r0_slot0_seed0)
                        raw_stream_3_fwa_target_row_r0_slot0_seed0 = 0 + 24 + raw_stream_3_fwa_row_r0_slot0_seed0
                        raw_stream_3_fwa_target_offset_r0_slot0_seed0 = raw_stream_3_fwa_target_row_r0_slot0_seed0 * 16 + raw_stream_3_fwa_chunk_r0_slot0_seed0 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_3_fwa_write_r0_slot0_seed0 = pl.reg.store(fwa_kv_flat_slot0_seed0, raw_stream_3_fwa_target_offset_r0_slot0_seed0, raw_stream_3_fwa_scaled_r0_slot0_seed0, raw_stream_3_fwa_all_r0_slot0_seed0, block_stride=32)
            kv_nz_seed0: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(kv_nz_seed0, split=1, id=1)
            if 1 + 2 < parts * 2:
                ids_future_seed1: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[0], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (1 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                raw_future_seed1_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                raw_future_seed1_prefix = pl.tile.slice(raw_future_seed1_storage, [1, 10240], [0, 0])
                raw_future_seed1 = pl.tile.reshape(raw_future_seed1_prefix, [32, COMPRESSED_CACHE_BYTES])
                future_seed1_first = pl.tile.read(ids_future_seed1, [0, 0])
                future_seed1_contiguous = pl.cast(0, pl.INT32)
                future_seed1_all_valid = pl.cast(0, pl.INT32)
                if future_seed1_first >= 0:
                    future_seed1_first_wide = pl.cast(future_seed1_first, pl.INT64)
                    future_seed1_mismatch = pl.cast(0, pl.INT64)
                    for future_seed1_check in pl.range(1, 32):
                        future_seed1_index = pl.tile.read(ids_future_seed1, [0, future_seed1_check])
                        future_seed1_expected = future_seed1_first_wide + pl.cast(future_seed1_check, pl.INT64)
                        future_seed1_delta = pl.cast(future_seed1_index, pl.INT64) ^ future_seed1_expected
                        future_seed1_mismatch = future_seed1_mismatch | future_seed1_delta
                    future_seed1_contiguous = pl.cast(future_seed1_mismatch == 0, pl.INT32)
                    future_seed1_all_valid = pl.cast(future_seed1_mismatch >= 0, pl.INT32)
                if future_seed1_contiguous != 0:
                    future_seed1_physical = pl.cast(future_seed1_first, pl.INDEX)
                    raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [0, 0], [future_seed1_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                elif future_seed1_all_valid != 0:
                    for future_seed1_lane in pl.range(0, 32, 2):
                        future_seed1_a = pl.tile.read(ids_future_seed1, [0, future_seed1_lane])
                        future_seed1_b = pl.tile.read(ids_future_seed1, [0, future_seed1_lane + 1])
                        future_seed1_lower = pl.min(future_seed1_a, future_seed1_b)
                        future_seed1_upper = pl.max(future_seed1_a, future_seed1_b)
                        future_seed1_pair = pl.cast(0, pl.INT32)
                        if future_seed1_upper == future_seed1_lower + 1:
                            future_seed1_pair = pl.cast(1, pl.INT32)
                        if future_seed1_pair != 0:
                            future_seed1_physical_pair = pl.cast(future_seed1_lower, pl.INDEX)
                            raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane, 0], [future_seed1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            future_seed1_physical_a = pl.cast(future_seed1_a, pl.INDEX)
                            raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane, 0], [future_seed1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            future_seed1_physical_b = pl.cast(future_seed1_b, pl.INDEX)
                            raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane + 1, 0], [future_seed1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                else:
                    for future_seed1_lane in pl.range(0, 32, 2):
                        future_seed1_a = pl.tile.read(ids_future_seed1, [0, future_seed1_lane])
                        future_seed1_b = pl.tile.read(ids_future_seed1, [0, future_seed1_lane + 1])
                        future_seed1_lower = pl.min(future_seed1_a, future_seed1_b)
                        future_seed1_upper = pl.max(future_seed1_a, future_seed1_b)
                        future_seed1_pair = pl.cast(0, pl.INT32)
                        if future_seed1_lower >= 0:
                            if future_seed1_upper == future_seed1_lower + 1:
                                future_seed1_pair = pl.cast(1, pl.INT32)
                        if future_seed1_pair != 0:
                            future_seed1_physical_pair = pl.cast(future_seed1_lower, pl.INDEX)
                            raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane, 0], [future_seed1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            if future_seed1_a >= 0:
                                future_seed1_physical_a = pl.cast(future_seed1_a, pl.INDEX)
                                raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane, 0], [future_seed1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed1 = pl.tile.assemble(raw_future_seed1, zero_raw_compressed, [future_seed1_lane, 0])
                            if future_seed1_b >= 0:
                                future_seed1_physical_b = pl.cast(future_seed1_b, pl.INDEX)
                                raw_future_seed1 = pl.tile.gather_row(raw_future_seed1, compressed_flat, [future_seed1_lane + 1, 0], [future_seed1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed1 = pl.tile.assemble(raw_future_seed1, zero_raw_compressed, [future_seed1_lane + 1, 0])
            kv_chunk_seed1: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            raw_current_window1_seed1: pl.Tile[[32, WINDOW_CACHE_BYTES], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([32, WINDOW_CACHE_BYTES], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
            fwa_raw_chunk_slot1_seed1 = pl.tile.slice(raw_current_window1_seed1, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [0, 0])
            fwa_signed_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.INT8)
            fwa_packed_slot1_seed1 = pl.reinterpret_view(fwa_signed_slot1_seed1, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot1_seed1 = fwa_packed_slot1_seed1[:, :HEAD_DIM]
            fwa_values_fp32_slot1_seed1 = pl.cast(fwa_codes_slot1_seed1, pl.FP32)
            fwa_values_slot1_seed1 = pl.cast(fwa_values_fp32_slot1_seed1, pl.BF16, mode='rint')
            fwa_scale_words_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot1_seed1 = pl.tile.reshape(fwa_scale_words_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot1_seed1 = pl.tile.reshape(fwa_values_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot1_seed1 = pl.tile.reshape(kv_chunk_seed1, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_4_fwa_all_r0_slot1_seed1 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_4_fwa_row_r0_slot1_seed1 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_4_fwa_chunk_r0_slot1_seed1 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_4_fwa_scale_offset_r0_slot1_seed1 = raw_stream_4_fwa_row_r0_slot1_seed1 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_4_fwa_scale_offset_r0_slot1_seed1 = raw_stream_4_fwa_scale_offset_r0_slot1_seed1 + raw_stream_4_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_4_fwa_groups_r0_slot1_seed1 = pl.reg.load(fwa_scale_flat_slot1_seed1, raw_stream_4_fwa_scale_offset_r0_slot1_seed1, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_4_fwa_factor_bits_r0_slot1_seed1 = pl.reg.group_broadcast(raw_stream_4_fwa_groups_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_4_fwa_factor_r0_slot1_seed1 = pl.reg.interpret_cast(raw_stream_4_fwa_factor_bits_r0_slot1_seed1, dtype=pl.BF16)
                        raw_stream_4_fwa_value_offset_r0_slot1_seed1 = raw_stream_4_fwa_row_r0_slot1_seed1 * HEAD_DIM + raw_stream_4_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_REG_TILE
                        raw_stream_4_fwa_value_r0_slot1_seed1 = pl.reg.load(fwa_value_flat_slot1_seed1, raw_stream_4_fwa_value_offset_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_4_fwa_scaled_r0_slot1_seed1 = pl.reg.mul(raw_stream_4_fwa_value_r0_slot1_seed1, raw_stream_4_fwa_factor_r0_slot1_seed1, raw_stream_4_fwa_all_r0_slot1_seed1)
                        raw_stream_4_fwa_target_row_r0_slot1_seed1 = 0 + 0 + raw_stream_4_fwa_row_r0_slot1_seed1
                        raw_stream_4_fwa_target_offset_r0_slot1_seed1 = raw_stream_4_fwa_target_row_r0_slot1_seed1 * 16 + raw_stream_4_fwa_chunk_r0_slot1_seed1 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_4_fwa_write_r0_slot1_seed1 = pl.reg.store(fwa_kv_flat_slot1_seed1, raw_stream_4_fwa_target_offset_r0_slot1_seed1, raw_stream_4_fwa_scaled_r0_slot1_seed1, raw_stream_4_fwa_all_r0_slot1_seed1, block_stride=32)
            fwa_raw_chunk_slot1_seed1 = pl.tile.slice(raw_current_window1_seed1, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [8, 0])
            fwa_signed_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.INT8)
            fwa_packed_slot1_seed1 = pl.reinterpret_view(fwa_signed_slot1_seed1, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot1_seed1 = fwa_packed_slot1_seed1[:, :HEAD_DIM]
            fwa_values_fp32_slot1_seed1 = pl.cast(fwa_codes_slot1_seed1, pl.FP32)
            fwa_values_slot1_seed1 = pl.cast(fwa_values_fp32_slot1_seed1, pl.BF16, mode='rint')
            fwa_scale_words_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot1_seed1 = pl.tile.reshape(fwa_scale_words_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot1_seed1 = pl.tile.reshape(fwa_values_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot1_seed1 = pl.tile.reshape(kv_chunk_seed1, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_5_fwa_all_r0_slot1_seed1 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_5_fwa_row_r0_slot1_seed1 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_5_fwa_chunk_r0_slot1_seed1 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_5_fwa_scale_offset_r0_slot1_seed1 = raw_stream_5_fwa_row_r0_slot1_seed1 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_5_fwa_scale_offset_r0_slot1_seed1 = raw_stream_5_fwa_scale_offset_r0_slot1_seed1 + raw_stream_5_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_5_fwa_groups_r0_slot1_seed1 = pl.reg.load(fwa_scale_flat_slot1_seed1, raw_stream_5_fwa_scale_offset_r0_slot1_seed1, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_5_fwa_factor_bits_r0_slot1_seed1 = pl.reg.group_broadcast(raw_stream_5_fwa_groups_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_5_fwa_factor_r0_slot1_seed1 = pl.reg.interpret_cast(raw_stream_5_fwa_factor_bits_r0_slot1_seed1, dtype=pl.BF16)
                        raw_stream_5_fwa_value_offset_r0_slot1_seed1 = raw_stream_5_fwa_row_r0_slot1_seed1 * HEAD_DIM + raw_stream_5_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_REG_TILE
                        raw_stream_5_fwa_value_r0_slot1_seed1 = pl.reg.load(fwa_value_flat_slot1_seed1, raw_stream_5_fwa_value_offset_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_5_fwa_scaled_r0_slot1_seed1 = pl.reg.mul(raw_stream_5_fwa_value_r0_slot1_seed1, raw_stream_5_fwa_factor_r0_slot1_seed1, raw_stream_5_fwa_all_r0_slot1_seed1)
                        raw_stream_5_fwa_target_row_r0_slot1_seed1 = 0 + 8 + raw_stream_5_fwa_row_r0_slot1_seed1
                        raw_stream_5_fwa_target_offset_r0_slot1_seed1 = raw_stream_5_fwa_target_row_r0_slot1_seed1 * 16 + raw_stream_5_fwa_chunk_r0_slot1_seed1 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_5_fwa_write_r0_slot1_seed1 = pl.reg.store(fwa_kv_flat_slot1_seed1, raw_stream_5_fwa_target_offset_r0_slot1_seed1, raw_stream_5_fwa_scaled_r0_slot1_seed1, raw_stream_5_fwa_all_r0_slot1_seed1, block_stride=32)
            fwa_raw_chunk_slot1_seed1 = pl.tile.slice(raw_current_window1_seed1, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [16, 0])
            fwa_signed_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.INT8)
            fwa_packed_slot1_seed1 = pl.reinterpret_view(fwa_signed_slot1_seed1, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot1_seed1 = fwa_packed_slot1_seed1[:, :HEAD_DIM]
            fwa_values_fp32_slot1_seed1 = pl.cast(fwa_codes_slot1_seed1, pl.FP32)
            fwa_values_slot1_seed1 = pl.cast(fwa_values_fp32_slot1_seed1, pl.BF16, mode='rint')
            fwa_scale_words_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot1_seed1 = pl.tile.reshape(fwa_scale_words_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot1_seed1 = pl.tile.reshape(fwa_values_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot1_seed1 = pl.tile.reshape(kv_chunk_seed1, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_6_fwa_all_r0_slot1_seed1 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_6_fwa_row_r0_slot1_seed1 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_6_fwa_chunk_r0_slot1_seed1 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_6_fwa_scale_offset_r0_slot1_seed1 = raw_stream_6_fwa_row_r0_slot1_seed1 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_6_fwa_scale_offset_r0_slot1_seed1 = raw_stream_6_fwa_scale_offset_r0_slot1_seed1 + raw_stream_6_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_6_fwa_groups_r0_slot1_seed1 = pl.reg.load(fwa_scale_flat_slot1_seed1, raw_stream_6_fwa_scale_offset_r0_slot1_seed1, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_6_fwa_factor_bits_r0_slot1_seed1 = pl.reg.group_broadcast(raw_stream_6_fwa_groups_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_6_fwa_factor_r0_slot1_seed1 = pl.reg.interpret_cast(raw_stream_6_fwa_factor_bits_r0_slot1_seed1, dtype=pl.BF16)
                        raw_stream_6_fwa_value_offset_r0_slot1_seed1 = raw_stream_6_fwa_row_r0_slot1_seed1 * HEAD_DIM + raw_stream_6_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_REG_TILE
                        raw_stream_6_fwa_value_r0_slot1_seed1 = pl.reg.load(fwa_value_flat_slot1_seed1, raw_stream_6_fwa_value_offset_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_6_fwa_scaled_r0_slot1_seed1 = pl.reg.mul(raw_stream_6_fwa_value_r0_slot1_seed1, raw_stream_6_fwa_factor_r0_slot1_seed1, raw_stream_6_fwa_all_r0_slot1_seed1)
                        raw_stream_6_fwa_target_row_r0_slot1_seed1 = 0 + 16 + raw_stream_6_fwa_row_r0_slot1_seed1
                        raw_stream_6_fwa_target_offset_r0_slot1_seed1 = raw_stream_6_fwa_target_row_r0_slot1_seed1 * 16 + raw_stream_6_fwa_chunk_r0_slot1_seed1 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_6_fwa_write_r0_slot1_seed1 = pl.reg.store(fwa_kv_flat_slot1_seed1, raw_stream_6_fwa_target_offset_r0_slot1_seed1, raw_stream_6_fwa_scaled_r0_slot1_seed1, raw_stream_6_fwa_all_r0_slot1_seed1, block_stride=32)
            fwa_raw_chunk_slot1_seed1 = pl.tile.slice(raw_current_window1_seed1, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [24, 0])
            fwa_signed_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.INT8)
            fwa_packed_slot1_seed1 = pl.reinterpret_view(fwa_signed_slot1_seed1, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
            fwa_codes_slot1_seed1 = fwa_packed_slot1_seed1[:, :HEAD_DIM]
            fwa_values_fp32_slot1_seed1 = pl.cast(fwa_codes_slot1_seed1, pl.FP32)
            fwa_values_slot1_seed1 = pl.cast(fwa_values_fp32_slot1_seed1, pl.BF16, mode='rint')
            fwa_scale_words_slot1_seed1 = pl.reinterpret_view(fwa_raw_chunk_slot1_seed1, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
            fwa_scale_flat_slot1_seed1 = pl.tile.reshape(fwa_scale_words_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
            fwa_value_flat_slot1_seed1 = pl.tile.reshape(fwa_values_slot1_seed1, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
            fwa_kv_flat_slot1_seed1 = pl.tile.reshape(kv_chunk_seed1, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_7_fwa_all_r0_slot1_seed1 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_7_fwa_row_r0_slot1_seed1 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                    for raw_stream_7_fwa_chunk_r0_slot1_seed1 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_7_fwa_scale_offset_r0_slot1_seed1 = raw_stream_7_fwa_row_r0_slot1_seed1 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                        raw_stream_7_fwa_scale_offset_r0_slot1_seed1 = raw_stream_7_fwa_scale_offset_r0_slot1_seed1 + raw_stream_7_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_WINDOW_SCALE_TILE
                        raw_stream_7_fwa_groups_r0_slot1_seed1 = pl.reg.load(fwa_scale_flat_slot1_seed1, raw_stream_7_fwa_scale_offset_r0_slot1_seed1, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                        raw_stream_7_fwa_factor_bits_r0_slot1_seed1 = pl.reg.group_broadcast(raw_stream_7_fwa_groups_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_7_fwa_factor_r0_slot1_seed1 = pl.reg.interpret_cast(raw_stream_7_fwa_factor_bits_r0_slot1_seed1, dtype=pl.BF16)
                        raw_stream_7_fwa_value_offset_r0_slot1_seed1 = raw_stream_7_fwa_row_r0_slot1_seed1 * HEAD_DIM + raw_stream_7_fwa_chunk_r0_slot1_seed1 * SPARSE_VF_REG_TILE
                        raw_stream_7_fwa_value_r0_slot1_seed1 = pl.reg.load(fwa_value_flat_slot1_seed1, raw_stream_7_fwa_value_offset_r0_slot1_seed1, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_7_fwa_scaled_r0_slot1_seed1 = pl.reg.mul(raw_stream_7_fwa_value_r0_slot1_seed1, raw_stream_7_fwa_factor_r0_slot1_seed1, raw_stream_7_fwa_all_r0_slot1_seed1)
                        raw_stream_7_fwa_target_row_r0_slot1_seed1 = 0 + 24 + raw_stream_7_fwa_row_r0_slot1_seed1
                        raw_stream_7_fwa_target_offset_r0_slot1_seed1 = raw_stream_7_fwa_target_row_r0_slot1_seed1 * 16 + raw_stream_7_fwa_chunk_r0_slot1_seed1 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_7_fwa_write_r0_slot1_seed1 = pl.reg.store(fwa_kv_flat_slot1_seed1, raw_stream_7_fwa_target_offset_r0_slot1_seed1, raw_stream_7_fwa_scaled_r0_slot1_seed1, raw_stream_7_fwa_all_r0_slot1_seed1, block_stride=32)
            kv_nz_seed1: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(kv_nz_seed1, split=1, id=1)
            if 2 + 2 < parts * 2:
                ids_future_seed2: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[1], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                raw_future_seed2_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                raw_future_seed2_prefix = pl.tile.slice(raw_future_seed2_storage, [1, 10240], [0, 0])
                raw_future_seed2 = pl.tile.reshape(raw_future_seed2_prefix, [32, COMPRESSED_CACHE_BYTES])
                future_seed2_first = pl.tile.read(ids_future_seed2, [0, 0])
                future_seed2_contiguous = pl.cast(0, pl.INT32)
                future_seed2_all_valid = pl.cast(0, pl.INT32)
                if future_seed2_first >= 0:
                    future_seed2_first_wide = pl.cast(future_seed2_first, pl.INT64)
                    future_seed2_mismatch = pl.cast(0, pl.INT64)
                    for future_seed2_check in pl.range(1, 32):
                        future_seed2_index = pl.tile.read(ids_future_seed2, [0, future_seed2_check])
                        future_seed2_expected = future_seed2_first_wide + pl.cast(future_seed2_check, pl.INT64)
                        future_seed2_delta = pl.cast(future_seed2_index, pl.INT64) ^ future_seed2_expected
                        future_seed2_mismatch = future_seed2_mismatch | future_seed2_delta
                    future_seed2_contiguous = pl.cast(future_seed2_mismatch == 0, pl.INT32)
                    future_seed2_all_valid = pl.cast(future_seed2_mismatch >= 0, pl.INT32)
                if future_seed2_contiguous != 0:
                    future_seed2_physical = pl.cast(future_seed2_first, pl.INDEX)
                    raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [0, 0], [future_seed2_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                elif future_seed2_all_valid != 0:
                    for future_seed2_lane in pl.range(0, 32, 2):
                        future_seed2_a = pl.tile.read(ids_future_seed2, [0, future_seed2_lane])
                        future_seed2_b = pl.tile.read(ids_future_seed2, [0, future_seed2_lane + 1])
                        future_seed2_lower = pl.min(future_seed2_a, future_seed2_b)
                        future_seed2_upper = pl.max(future_seed2_a, future_seed2_b)
                        future_seed2_pair = pl.cast(0, pl.INT32)
                        if future_seed2_upper == future_seed2_lower + 1:
                            future_seed2_pair = pl.cast(1, pl.INT32)
                        if future_seed2_pair != 0:
                            future_seed2_physical_pair = pl.cast(future_seed2_lower, pl.INDEX)
                            raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane, 0], [future_seed2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            future_seed2_physical_a = pl.cast(future_seed2_a, pl.INDEX)
                            raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane, 0], [future_seed2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            future_seed2_physical_b = pl.cast(future_seed2_b, pl.INDEX)
                            raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane + 1, 0], [future_seed2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                else:
                    for future_seed2_lane in pl.range(0, 32, 2):
                        future_seed2_a = pl.tile.read(ids_future_seed2, [0, future_seed2_lane])
                        future_seed2_b = pl.tile.read(ids_future_seed2, [0, future_seed2_lane + 1])
                        future_seed2_lower = pl.min(future_seed2_a, future_seed2_b)
                        future_seed2_upper = pl.max(future_seed2_a, future_seed2_b)
                        future_seed2_pair = pl.cast(0, pl.INT32)
                        if future_seed2_lower >= 0:
                            if future_seed2_upper == future_seed2_lower + 1:
                                future_seed2_pair = pl.cast(1, pl.INT32)
                        if future_seed2_pair != 0:
                            future_seed2_physical_pair = pl.cast(future_seed2_lower, pl.INDEX)
                            raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane, 0], [future_seed2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            if future_seed2_a >= 0:
                                future_seed2_physical_a = pl.cast(future_seed2_a, pl.INDEX)
                                raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane, 0], [future_seed2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed2 = pl.tile.assemble(raw_future_seed2, zero_raw_compressed, [future_seed2_lane, 0])
                            if future_seed2_b >= 0:
                                future_seed2_physical_b = pl.cast(future_seed2_b, pl.INDEX)
                                raw_future_seed2 = pl.tile.gather_row(raw_future_seed2, compressed_flat, [future_seed2_lane + 1, 0], [future_seed2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed2 = pl.tile.assemble(raw_future_seed2, zero_raw_compressed, [future_seed2_lane + 1, 0])
            kv_chunk_seed2: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            raw_current_compressed2_flat_seed2: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
            raw_current_compressed2_prefix_seed2 = pl.tile.slice(raw_current_compressed2_flat_seed2, [1, 10240], [0, 0])
            raw_current_compressed2_seed2 = pl.tile.reshape(raw_current_compressed2_prefix_seed2, [32, COMPRESSED_CACHE_BYTES])
            fca_raw_chunk_slot2_seed2 = pl.tile.slice(raw_current_compressed2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
            fca_packed_slot2_seed2 = pl.reinterpret_view(fca_raw_chunk_slot2_seed2, pl.FP4E2M1X2)
            fca_codes_slot2_seed2 = fca_packed_slot2_seed2[:, :SPARSE_VF_HALF_DIM]
            fca_values_slot2_seed2 = pl.cast(fca_codes_slot2_seed2, pl.BF16)
            fca_scale_words_slot2_seed2 = pl.reinterpret_view(fca_raw_chunk_slot2_seed2, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_scale_flat_slot2_seed2 = pl.tile.reshape(fca_scale_words_slot2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_value_flat_slot2_seed2 = pl.tile.reshape(fca_values_slot2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
            fca_kv_flat_slot2_seed2 = pl.tile.reshape(kv_chunk_seed2, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_8_fca_all_r4_slot2_seed2 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_8_fca_row_r4_slot2_seed2 in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                    for raw_stream_8_fca_chunk_r4_slot2_seed2 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_8_fca_scale_offset_r4_slot2_seed2 = raw_stream_8_fca_row_r4_slot2_seed2 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                        raw_stream_8_fca_scale_offset_r4_slot2_seed2 = raw_stream_8_fca_scale_offset_r4_slot2_seed2 + raw_stream_8_fca_chunk_r4_slot2_seed2 * SPARSE_VF_COMPRESSED_SCALE_TILE
                        raw_stream_8_fca_groups_r4_slot2_seed2 = pl.reg.load(fca_scale_flat_slot2_seed2, raw_stream_8_fca_scale_offset_r4_slot2_seed2, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                        raw_stream_8_fca_factor_bits_r4_slot2_seed2 = pl.reg.group_broadcast(raw_stream_8_fca_groups_r4_slot2_seed2, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_8_fca_factor_r4_slot2_seed2 = pl.reg.interpret_cast(raw_stream_8_fca_factor_bits_r4_slot2_seed2, dtype=pl.BF16)
                        raw_stream_8_fca_value_offset_r4_slot2_seed2 = raw_stream_8_fca_row_r4_slot2_seed2 * HEAD_DIM + raw_stream_8_fca_chunk_r4_slot2_seed2 * SPARSE_VF_REG_TILE
                        raw_stream_8_fca_value_r4_slot2_seed2 = pl.reg.load(fca_value_flat_slot2_seed2, raw_stream_8_fca_value_offset_r4_slot2_seed2, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_8_fca_scaled_r4_slot2_seed2 = pl.reg.mul(raw_stream_8_fca_value_r4_slot2_seed2, raw_stream_8_fca_factor_r4_slot2_seed2, raw_stream_8_fca_all_r4_slot2_seed2)
                        raw_stream_8_fca_target_row_r4_slot2_seed2 = 0 + 0 + raw_stream_8_fca_row_r4_slot2_seed2
                        raw_stream_8_fca_target_offset_r4_slot2_seed2 = raw_stream_8_fca_target_row_r4_slot2_seed2 * 16 + raw_stream_8_fca_chunk_r4_slot2_seed2 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_8_fca_write_r4_slot2_seed2 = pl.reg.store(fca_kv_flat_slot2_seed2, raw_stream_8_fca_target_offset_r4_slot2_seed2, raw_stream_8_fca_scaled_r4_slot2_seed2, raw_stream_8_fca_all_r4_slot2_seed2, block_stride=32)
            fca_raw_chunk_slot2_seed2 = pl.tile.slice(raw_current_compressed2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
            fca_packed_slot2_seed2 = pl.reinterpret_view(fca_raw_chunk_slot2_seed2, pl.FP4E2M1X2)
            fca_codes_slot2_seed2 = fca_packed_slot2_seed2[:, :SPARSE_VF_HALF_DIM]
            fca_values_slot2_seed2 = pl.cast(fca_codes_slot2_seed2, pl.BF16)
            fca_scale_words_slot2_seed2 = pl.reinterpret_view(fca_raw_chunk_slot2_seed2, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_scale_flat_slot2_seed2 = pl.tile.reshape(fca_scale_words_slot2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_value_flat_slot2_seed2 = pl.tile.reshape(fca_values_slot2_seed2, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
            fca_kv_flat_slot2_seed2 = pl.tile.reshape(kv_chunk_seed2, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_10_fca_all_r4_slot2_seed2 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_10_fca_row_r4_slot2_seed2 in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                    for raw_stream_10_fca_chunk_r4_slot2_seed2 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_10_fca_scale_offset_r4_slot2_seed2 = raw_stream_10_fca_row_r4_slot2_seed2 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                        raw_stream_10_fca_scale_offset_r4_slot2_seed2 = raw_stream_10_fca_scale_offset_r4_slot2_seed2 + raw_stream_10_fca_chunk_r4_slot2_seed2 * SPARSE_VF_COMPRESSED_SCALE_TILE
                        raw_stream_10_fca_groups_r4_slot2_seed2 = pl.reg.load(fca_scale_flat_slot2_seed2, raw_stream_10_fca_scale_offset_r4_slot2_seed2, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                        raw_stream_10_fca_factor_bits_r4_slot2_seed2 = pl.reg.group_broadcast(raw_stream_10_fca_groups_r4_slot2_seed2, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_10_fca_factor_r4_slot2_seed2 = pl.reg.interpret_cast(raw_stream_10_fca_factor_bits_r4_slot2_seed2, dtype=pl.BF16)
                        raw_stream_10_fca_value_offset_r4_slot2_seed2 = raw_stream_10_fca_row_r4_slot2_seed2 * HEAD_DIM + raw_stream_10_fca_chunk_r4_slot2_seed2 * SPARSE_VF_REG_TILE
                        raw_stream_10_fca_value_r4_slot2_seed2 = pl.reg.load(fca_value_flat_slot2_seed2, raw_stream_10_fca_value_offset_r4_slot2_seed2, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_10_fca_scaled_r4_slot2_seed2 = pl.reg.mul(raw_stream_10_fca_value_r4_slot2_seed2, raw_stream_10_fca_factor_r4_slot2_seed2, raw_stream_10_fca_all_r4_slot2_seed2)
                        raw_stream_10_fca_target_row_r4_slot2_seed2 = 0 + 16 + raw_stream_10_fca_row_r4_slot2_seed2
                        raw_stream_10_fca_target_offset_r4_slot2_seed2 = raw_stream_10_fca_target_row_r4_slot2_seed2 * 16 + raw_stream_10_fca_chunk_r4_slot2_seed2 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_10_fca_write_r4_slot2_seed2 = pl.reg.store(fca_kv_flat_slot2_seed2, raw_stream_10_fca_target_offset_r4_slot2_seed2, raw_stream_10_fca_scaled_r4_slot2_seed2, raw_stream_10_fca_all_r4_slot2_seed2, block_stride=32)
            kv_nz_seed2: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(kv_nz_seed2, split=1, id=1)
            if 3 + 2 < parts * 2:
                ids_future_seed3: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[2], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (3 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                raw_future_seed3_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                raw_future_seed3_prefix = pl.tile.slice(raw_future_seed3_storage, [1, 10240], [0, 0])
                raw_future_seed3 = pl.tile.reshape(raw_future_seed3_prefix, [32, COMPRESSED_CACHE_BYTES])
                future_seed3_first = pl.tile.read(ids_future_seed3, [0, 0])
                future_seed3_contiguous = pl.cast(0, pl.INT32)
                future_seed3_all_valid = pl.cast(0, pl.INT32)
                if future_seed3_first >= 0:
                    future_seed3_first_wide = pl.cast(future_seed3_first, pl.INT64)
                    future_seed3_mismatch = pl.cast(0, pl.INT64)
                    for future_seed3_check in pl.range(1, 32):
                        future_seed3_index = pl.tile.read(ids_future_seed3, [0, future_seed3_check])
                        future_seed3_expected = future_seed3_first_wide + pl.cast(future_seed3_check, pl.INT64)
                        future_seed3_delta = pl.cast(future_seed3_index, pl.INT64) ^ future_seed3_expected
                        future_seed3_mismatch = future_seed3_mismatch | future_seed3_delta
                    future_seed3_contiguous = pl.cast(future_seed3_mismatch == 0, pl.INT32)
                    future_seed3_all_valid = pl.cast(future_seed3_mismatch >= 0, pl.INT32)
                if future_seed3_contiguous != 0:
                    future_seed3_physical = pl.cast(future_seed3_first, pl.INDEX)
                    raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [0, 0], [future_seed3_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                elif future_seed3_all_valid != 0:
                    for future_seed3_lane in pl.range(0, 32, 2):
                        future_seed3_a = pl.tile.read(ids_future_seed3, [0, future_seed3_lane])
                        future_seed3_b = pl.tile.read(ids_future_seed3, [0, future_seed3_lane + 1])
                        future_seed3_lower = pl.min(future_seed3_a, future_seed3_b)
                        future_seed3_upper = pl.max(future_seed3_a, future_seed3_b)
                        future_seed3_pair = pl.cast(0, pl.INT32)
                        if future_seed3_upper == future_seed3_lower + 1:
                            future_seed3_pair = pl.cast(1, pl.INT32)
                        if future_seed3_pair != 0:
                            future_seed3_physical_pair = pl.cast(future_seed3_lower, pl.INDEX)
                            raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane, 0], [future_seed3_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            future_seed3_physical_a = pl.cast(future_seed3_a, pl.INDEX)
                            raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane, 0], [future_seed3_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            future_seed3_physical_b = pl.cast(future_seed3_b, pl.INDEX)
                            raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane + 1, 0], [future_seed3_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                else:
                    for future_seed3_lane in pl.range(0, 32, 2):
                        future_seed3_a = pl.tile.read(ids_future_seed3, [0, future_seed3_lane])
                        future_seed3_b = pl.tile.read(ids_future_seed3, [0, future_seed3_lane + 1])
                        future_seed3_lower = pl.min(future_seed3_a, future_seed3_b)
                        future_seed3_upper = pl.max(future_seed3_a, future_seed3_b)
                        future_seed3_pair = pl.cast(0, pl.INT32)
                        if future_seed3_lower >= 0:
                            if future_seed3_upper == future_seed3_lower + 1:
                                future_seed3_pair = pl.cast(1, pl.INT32)
                        if future_seed3_pair != 0:
                            future_seed3_physical_pair = pl.cast(future_seed3_lower, pl.INDEX)
                            raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane, 0], [future_seed3_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                        else:
                            if future_seed3_a >= 0:
                                future_seed3_physical_a = pl.cast(future_seed3_a, pl.INDEX)
                                raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane, 0], [future_seed3_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed3 = pl.tile.assemble(raw_future_seed3, zero_raw_compressed, [future_seed3_lane, 0])
                            if future_seed3_b >= 0:
                                future_seed3_physical_b = pl.cast(future_seed3_b, pl.INDEX)
                                raw_future_seed3 = pl.tile.gather_row(raw_future_seed3, compressed_flat, [future_seed3_lane + 1, 0], [future_seed3_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                raw_future_seed3 = pl.tile.assemble(raw_future_seed3, zero_raw_compressed, [future_seed3_lane + 1, 0])
            kv_chunk_seed3: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            raw_current_compressed0_flat_seed3: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
            raw_current_compressed0_prefix_seed3 = pl.tile.slice(raw_current_compressed0_flat_seed3, [1, 10240], [0, 0])
            raw_current_compressed0_seed3 = pl.tile.reshape(raw_current_compressed0_prefix_seed3, [32, COMPRESSED_CACHE_BYTES])
            fca_raw_chunk_slot0_seed3 = pl.tile.slice(raw_current_compressed0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
            fca_packed_slot0_seed3 = pl.reinterpret_view(fca_raw_chunk_slot0_seed3, pl.FP4E2M1X2)
            fca_codes_slot0_seed3 = fca_packed_slot0_seed3[:, :SPARSE_VF_HALF_DIM]
            fca_values_slot0_seed3 = pl.cast(fca_codes_slot0_seed3, pl.BF16)
            fca_scale_words_slot0_seed3 = pl.reinterpret_view(fca_raw_chunk_slot0_seed3, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_scale_flat_slot0_seed3 = pl.tile.reshape(fca_scale_words_slot0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_value_flat_slot0_seed3 = pl.tile.reshape(fca_values_slot0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
            fca_kv_flat_slot0_seed3 = pl.tile.reshape(kv_chunk_seed3, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_12_fca_all_r4_slot0_seed3 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_12_fca_row_r4_slot0_seed3 in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                    for raw_stream_12_fca_chunk_r4_slot0_seed3 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_12_fca_scale_offset_r4_slot0_seed3 = raw_stream_12_fca_row_r4_slot0_seed3 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                        raw_stream_12_fca_scale_offset_r4_slot0_seed3 = raw_stream_12_fca_scale_offset_r4_slot0_seed3 + raw_stream_12_fca_chunk_r4_slot0_seed3 * SPARSE_VF_COMPRESSED_SCALE_TILE
                        raw_stream_12_fca_groups_r4_slot0_seed3 = pl.reg.load(fca_scale_flat_slot0_seed3, raw_stream_12_fca_scale_offset_r4_slot0_seed3, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                        raw_stream_12_fca_factor_bits_r4_slot0_seed3 = pl.reg.group_broadcast(raw_stream_12_fca_groups_r4_slot0_seed3, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_12_fca_factor_r4_slot0_seed3 = pl.reg.interpret_cast(raw_stream_12_fca_factor_bits_r4_slot0_seed3, dtype=pl.BF16)
                        raw_stream_12_fca_value_offset_r4_slot0_seed3 = raw_stream_12_fca_row_r4_slot0_seed3 * HEAD_DIM + raw_stream_12_fca_chunk_r4_slot0_seed3 * SPARSE_VF_REG_TILE
                        raw_stream_12_fca_value_r4_slot0_seed3 = pl.reg.load(fca_value_flat_slot0_seed3, raw_stream_12_fca_value_offset_r4_slot0_seed3, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_12_fca_scaled_r4_slot0_seed3 = pl.reg.mul(raw_stream_12_fca_value_r4_slot0_seed3, raw_stream_12_fca_factor_r4_slot0_seed3, raw_stream_12_fca_all_r4_slot0_seed3)
                        raw_stream_12_fca_target_row_r4_slot0_seed3 = 0 + 0 + raw_stream_12_fca_row_r4_slot0_seed3
                        raw_stream_12_fca_target_offset_r4_slot0_seed3 = raw_stream_12_fca_target_row_r4_slot0_seed3 * 16 + raw_stream_12_fca_chunk_r4_slot0_seed3 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_12_fca_write_r4_slot0_seed3 = pl.reg.store(fca_kv_flat_slot0_seed3, raw_stream_12_fca_target_offset_r4_slot0_seed3, raw_stream_12_fca_scaled_r4_slot0_seed3, raw_stream_12_fca_all_r4_slot0_seed3, block_stride=32)
            fca_raw_chunk_slot0_seed3 = pl.tile.slice(raw_current_compressed0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
            fca_packed_slot0_seed3 = pl.reinterpret_view(fca_raw_chunk_slot0_seed3, pl.FP4E2M1X2)
            fca_codes_slot0_seed3 = fca_packed_slot0_seed3[:, :SPARSE_VF_HALF_DIM]
            fca_values_slot0_seed3 = pl.cast(fca_codes_slot0_seed3, pl.BF16)
            fca_scale_words_slot0_seed3 = pl.reinterpret_view(fca_raw_chunk_slot0_seed3, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_scale_flat_slot0_seed3 = pl.tile.reshape(fca_scale_words_slot0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
            fca_value_flat_slot0_seed3 = pl.tile.reshape(fca_values_slot0_seed3, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
            fca_kv_flat_slot0_seed3 = pl.tile.reshape(kv_chunk_seed3, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
            with pl.reg_scope():
                raw_stream_14_fca_all_r4_slot0_seed3 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                for raw_stream_14_fca_row_r4_slot0_seed3 in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                    for raw_stream_14_fca_chunk_r4_slot0_seed3 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                        raw_stream_14_fca_scale_offset_r4_slot0_seed3 = raw_stream_14_fca_row_r4_slot0_seed3 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                        raw_stream_14_fca_scale_offset_r4_slot0_seed3 = raw_stream_14_fca_scale_offset_r4_slot0_seed3 + raw_stream_14_fca_chunk_r4_slot0_seed3 * SPARSE_VF_COMPRESSED_SCALE_TILE
                        raw_stream_14_fca_groups_r4_slot0_seed3 = pl.reg.load(fca_scale_flat_slot0_seed3, raw_stream_14_fca_scale_offset_r4_slot0_seed3, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                        raw_stream_14_fca_factor_bits_r4_slot0_seed3 = pl.reg.group_broadcast(raw_stream_14_fca_groups_r4_slot0_seed3, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_14_fca_factor_r4_slot0_seed3 = pl.reg.interpret_cast(raw_stream_14_fca_factor_bits_r4_slot0_seed3, dtype=pl.BF16)
                        raw_stream_14_fca_value_offset_r4_slot0_seed3 = raw_stream_14_fca_row_r4_slot0_seed3 * HEAD_DIM + raw_stream_14_fca_chunk_r4_slot0_seed3 * SPARSE_VF_REG_TILE
                        raw_stream_14_fca_value_r4_slot0_seed3 = pl.reg.load(fca_value_flat_slot0_seed3, raw_stream_14_fca_value_offset_r4_slot0_seed3, lanes=SPARSE_VF_REG_TILE)
                        raw_stream_14_fca_scaled_r4_slot0_seed3 = pl.reg.mul(raw_stream_14_fca_value_r4_slot0_seed3, raw_stream_14_fca_factor_r4_slot0_seed3, raw_stream_14_fca_all_r4_slot0_seed3)
                        raw_stream_14_fca_target_row_r4_slot0_seed3 = 0 + 16 + raw_stream_14_fca_row_r4_slot0_seed3
                        raw_stream_14_fca_target_offset_r4_slot0_seed3 = raw_stream_14_fca_target_row_r4_slot0_seed3 * 16 + raw_stream_14_fca_chunk_r4_slot0_seed3 * (32 * SPARSE_VF_REG_TILE)
                        _raw_stream_14_fca_write_r4_slot0_seed3 = pl.reg.store(fca_kv_flat_slot0_seed3, raw_stream_14_fca_target_offset_r4_slot0_seed3, raw_stream_14_fca_scaled_r4_slot0_seed3, raw_stream_14_fca_all_r4_slot0_seed3, block_stride=32)
            kv_nz_seed3: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pl.system.tpush_to_aic(kv_nz_seed3, split=1, id=1)
            for part in pl.range(parts):
                column = part * SPARSE_VF_KEY_TILE
                if compressed_keys > 0:
                    qk_half_a = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], dtype=pl.FP32, split=1, id=0)
                    qk_half_b = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], dtype=pl.FP32, split=1, id=0)
                    qk_half = pl.tile.create([SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    qk_half = pl.tile.assemble(qk_half, qk_half_a, [0, 0])
                    qk_half = pl.tile.assemble(qk_half, qk_half_b, [0, SPARSE_VF_KEY_TILE // 2])
                else:
                    qk_half = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=1, id=0)
                scores = pl.mul(qk_half, SCALE)
                # Copy scores before releasing the Cube FIFO.
                if compressed_keys > 0:
                    pl.system.tfree_to_aic(qk_half_a, id=0)
                    pl.system.tfree_to_aic(qk_half_b, id=0)
                else:
                    pl.system.tfree_to_aic(qk_half, id=0)
                column = part * SPARSE_VF_KEY_TILE
                if column < WINDOW_KEYS:
                    ids = pl.tile.load(window_indices, [token, column], [1, SPARSE_VF_KEY_TILE])
                else:
                    c_column = column - WINDOW_KEYS
                    ids = pl.tile.load(compressed_indices, [token, c_column], [1, SPARSE_VF_KEY_TILE])
                weights = pl.tile.create([SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                next_max = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                next_denominator = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                correction = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                sm_ids_owned = pl.tile.move(ids, pl.MemorySpace.Vec)
                sm_scores_flat = pl.tile.reshape(scores, [SPARSE_VF_FUSED_AIV_HEAD_TILE * SPARSE_VF_KEY_TILE])
                sm_ids_flat = pl.tile.reshape(sm_ids_owned, [SPARSE_VF_KEY_TILE])
                sm_weights_flat = pl.tile.reshape(weights, [SPARSE_VF_FUSED_AIV_HEAD_TILE * SPARSE_VF_KEY_TILE])
                sm_old_max_flat = pl.tile.reshape(maximum, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_old_den_flat = pl.tile.reshape(denominator, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_next_max_flat = pl.tile.reshape(next_max, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_next_den_flat = pl.tile.reshape(next_denominator, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_correction_flat = pl.tile.reshape(correction, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                with pl.reg_scope():
                    raw_stream_16_sm_all64 = pl.reg.full_mask(lanes=64)
                    raw_stream_16_sm_one = pl.reg.full_mask(lanes=1)
                    raw_stream_16_sm_ids0 = pl.reg.load(sm_ids_flat, 0, lanes=64)
                    raw_stream_16_sm_ids1 = pl.reg.load(sm_ids_flat, 64, lanes=64)
                    raw_stream_16_sm_valid0 = pl.reg.compares(raw_stream_16_sm_ids0, 0, raw_stream_16_sm_all64, predicate='ge')
                    raw_stream_16_sm_valid1 = pl.reg.compares(raw_stream_16_sm_ids1, 0, raw_stream_16_sm_all64, predicate='ge')
                    raw_stream_16_sm_invalid = pl.reg.broadcast(-1e+30, lanes=64, dtype=pl.FP32)
                    for raw_stream_16_sm_row in pl.hardware_loop(SPARSE_VF_FUSED_AIV_HEAD_TILE):
                        raw_stream_16_sm_offset = raw_stream_16_sm_row * SPARSE_VF_KEY_TILE
                        raw_stream_16_sm_x0 = pl.reg.load(sm_scores_flat, raw_stream_16_sm_offset, lanes=64)
                        raw_stream_16_sm_x1 = pl.reg.load(sm_scores_flat, raw_stream_16_sm_offset + 64, lanes=64)
                        raw_stream_16_sm_masked0 = pl.reg.select(raw_stream_16_sm_valid0, raw_stream_16_sm_x0, raw_stream_16_sm_invalid)
                        raw_stream_16_sm_masked1 = pl.reg.select(raw_stream_16_sm_valid1, raw_stream_16_sm_x1, raw_stream_16_sm_invalid)
                        raw_stream_16_sm_folded = pl.reg.maximum(raw_stream_16_sm_masked0, raw_stream_16_sm_masked1, raw_stream_16_sm_all64)
                        raw_stream_16_sm_tile_max = pl.reg.reduce_max(raw_stream_16_sm_folded, raw_stream_16_sm_all64)
                        raw_stream_16_sm_old_max = pl.reg.load(sm_old_max_flat, raw_stream_16_sm_row, lanes=1)
                        raw_stream_16_sm_new_max = pl.reg.maximum(raw_stream_16_sm_tile_max, raw_stream_16_sm_old_max, raw_stream_16_sm_one)
                        raw_stream_16_sm_delta = pl.reg.sub(raw_stream_16_sm_old_max, raw_stream_16_sm_new_max, raw_stream_16_sm_one)
                        raw_stream_16_sm_alpha = pl.reg.exp(raw_stream_16_sm_delta, raw_stream_16_sm_one)
                        raw_stream_16_sm_max64 = pl.reg.broadcast(raw_stream_16_sm_new_max, lanes=64, dtype=pl.FP32)
                        raw_stream_16_sm_probability0 = pl.reg.expdif(raw_stream_16_sm_masked0, raw_stream_16_sm_max64, raw_stream_16_sm_valid0)
                        raw_stream_16_sm_probability1 = pl.reg.expdif(raw_stream_16_sm_masked1, raw_stream_16_sm_max64, raw_stream_16_sm_valid1)
                        raw_stream_16_sm_sum0 = pl.reg.reduce_sum(raw_stream_16_sm_probability0, raw_stream_16_sm_all64, reassoc=True)
                        raw_stream_16_sm_sum1 = pl.reg.reduce_sum(raw_stream_16_sm_probability1, raw_stream_16_sm_all64, reassoc=True)
                        raw_stream_16_sm_sum0_ordered = pl.reg.adds(raw_stream_16_sm_sum0, 0.0, raw_stream_16_sm_one)
                        raw_stream_16_sm_tile_sum = pl.reg.add(raw_stream_16_sm_sum0_ordered, raw_stream_16_sm_sum1, raw_stream_16_sm_one)
                        raw_stream_16_sm_old_den = pl.reg.load(sm_old_den_flat, raw_stream_16_sm_row, lanes=1)
                        raw_stream_16_sm_corrected_den = pl.reg.mul(raw_stream_16_sm_old_den, raw_stream_16_sm_alpha, raw_stream_16_sm_one)
                        raw_stream_16_sm_new_den = pl.reg.add(raw_stream_16_sm_corrected_den, raw_stream_16_sm_tile_sum, raw_stream_16_sm_one)
                        raw_stream_16_sm_bf16_0 = pl.reg.convert(raw_stream_16_sm_probability0, dtype=pl.BF16, rounding='R', saturate=False)
                        raw_stream_16_sm_bf16_1 = pl.reg.convert(raw_stream_16_sm_probability1, dtype=pl.BF16, rounding='R', saturate=False)
                        _raw_stream_16_sm_store_p0 = pl.reg.store(sm_weights_flat, raw_stream_16_sm_offset, raw_stream_16_sm_bf16_0)
                        _raw_stream_16_sm_store_p1 = pl.reg.store(sm_weights_flat, raw_stream_16_sm_offset + 64, raw_stream_16_sm_bf16_1)
                        _raw_stream_16_sm_store_m = pl.reg.store(sm_next_max_flat, raw_stream_16_sm_row, raw_stream_16_sm_new_max)
                        _raw_stream_16_sm_store_l = pl.reg.store(sm_next_den_flat, raw_stream_16_sm_row, raw_stream_16_sm_new_den)
                        _raw_stream_16_sm_store_c = pl.reg.store(sm_correction_flat, raw_stream_16_sm_row, raw_stream_16_sm_alpha)
                maximum = next_max
                denominator = next_denominator
                pl.system.tpush_to_aic(weights, split=1, id=2)
                if part + 2 < parts:
                    if 2 * (part + 2) + 2 < parts * 2:
                        if (2 * (part + 2) + 2) % 3 == 0:
                            ids_future_part_a_s0: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[0], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_a_s0_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_a_s0_prefix = pl.tile.slice(raw_future_part_a_s0_storage, [1, 10240], [0, 0])
                            raw_future_part_a_s0 = pl.tile.reshape(raw_future_part_a_s0_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_a_s0_first = pl.tile.read(ids_future_part_a_s0, [0, 0])
                            future_part_a_s0_contiguous = pl.cast(0, pl.INT32)
                            future_part_a_s0_all_valid = pl.cast(0, pl.INT32)
                            if future_part_a_s0_first >= 0:
                                future_part_a_s0_first_wide = pl.cast(future_part_a_s0_first, pl.INT64)
                                future_part_a_s0_mismatch = pl.cast(0, pl.INT64)
                                for future_part_a_s0_check in pl.range(1, 32):
                                    future_part_a_s0_index = pl.tile.read(ids_future_part_a_s0, [0, future_part_a_s0_check])
                                    future_part_a_s0_expected = future_part_a_s0_first_wide + pl.cast(future_part_a_s0_check, pl.INT64)
                                    future_part_a_s0_delta = pl.cast(future_part_a_s0_index, pl.INT64) ^ future_part_a_s0_expected
                                    future_part_a_s0_mismatch = future_part_a_s0_mismatch | future_part_a_s0_delta
                                future_part_a_s0_contiguous = pl.cast(future_part_a_s0_mismatch == 0, pl.INT32)
                                future_part_a_s0_all_valid = pl.cast(future_part_a_s0_mismatch >= 0, pl.INT32)
                            if future_part_a_s0_contiguous != 0:
                                future_part_a_s0_physical = pl.cast(future_part_a_s0_first, pl.INDEX)
                                raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [0, 0], [future_part_a_s0_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_a_s0_all_valid != 0:
                                for future_part_a_s0_lane in pl.range(0, 32, 2):
                                    future_part_a_s0_a = pl.tile.read(ids_future_part_a_s0, [0, future_part_a_s0_lane])
                                    future_part_a_s0_b = pl.tile.read(ids_future_part_a_s0, [0, future_part_a_s0_lane + 1])
                                    future_part_a_s0_lower = pl.min(future_part_a_s0_a, future_part_a_s0_b)
                                    future_part_a_s0_upper = pl.max(future_part_a_s0_a, future_part_a_s0_b)
                                    future_part_a_s0_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s0_upper == future_part_a_s0_lower + 1:
                                        future_part_a_s0_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s0_pair != 0:
                                        future_part_a_s0_physical_pair = pl.cast(future_part_a_s0_lower, pl.INDEX)
                                        raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane, 0], [future_part_a_s0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_a_s0_physical_a = pl.cast(future_part_a_s0_a, pl.INDEX)
                                        raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane, 0], [future_part_a_s0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_a_s0_physical_b = pl.cast(future_part_a_s0_b, pl.INDEX)
                                        raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane + 1, 0], [future_part_a_s0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_a_s0_lane in pl.range(0, 32, 2):
                                    future_part_a_s0_a = pl.tile.read(ids_future_part_a_s0, [0, future_part_a_s0_lane])
                                    future_part_a_s0_b = pl.tile.read(ids_future_part_a_s0, [0, future_part_a_s0_lane + 1])
                                    future_part_a_s0_lower = pl.min(future_part_a_s0_a, future_part_a_s0_b)
                                    future_part_a_s0_upper = pl.max(future_part_a_s0_a, future_part_a_s0_b)
                                    future_part_a_s0_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s0_lower >= 0:
                                        if future_part_a_s0_upper == future_part_a_s0_lower + 1:
                                            future_part_a_s0_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s0_pair != 0:
                                        future_part_a_s0_physical_pair = pl.cast(future_part_a_s0_lower, pl.INDEX)
                                        raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane, 0], [future_part_a_s0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_a_s0_a >= 0:
                                            future_part_a_s0_physical_a = pl.cast(future_part_a_s0_a, pl.INDEX)
                                            raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane, 0], [future_part_a_s0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s0 = pl.tile.assemble(raw_future_part_a_s0, zero_raw_compressed, [future_part_a_s0_lane, 0])
                                        if future_part_a_s0_b >= 0:
                                            future_part_a_s0_physical_b = pl.cast(future_part_a_s0_b, pl.INDEX)
                                            raw_future_part_a_s0 = pl.tile.gather_row(raw_future_part_a_s0, compressed_flat, [future_part_a_s0_lane + 1, 0], [future_part_a_s0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s0 = pl.tile.assemble(raw_future_part_a_s0, zero_raw_compressed, [future_part_a_s0_lane + 1, 0])
                        elif (2 * (part + 2) + 2) % 3 == 1:
                            ids_future_part_a_s1: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[1], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_a_s1_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_a_s1_prefix = pl.tile.slice(raw_future_part_a_s1_storage, [1, 10240], [0, 0])
                            raw_future_part_a_s1 = pl.tile.reshape(raw_future_part_a_s1_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_a_s1_first = pl.tile.read(ids_future_part_a_s1, [0, 0])
                            future_part_a_s1_contiguous = pl.cast(0, pl.INT32)
                            future_part_a_s1_all_valid = pl.cast(0, pl.INT32)
                            if future_part_a_s1_first >= 0:
                                future_part_a_s1_first_wide = pl.cast(future_part_a_s1_first, pl.INT64)
                                future_part_a_s1_mismatch = pl.cast(0, pl.INT64)
                                for future_part_a_s1_check in pl.range(1, 32):
                                    future_part_a_s1_index = pl.tile.read(ids_future_part_a_s1, [0, future_part_a_s1_check])
                                    future_part_a_s1_expected = future_part_a_s1_first_wide + pl.cast(future_part_a_s1_check, pl.INT64)
                                    future_part_a_s1_delta = pl.cast(future_part_a_s1_index, pl.INT64) ^ future_part_a_s1_expected
                                    future_part_a_s1_mismatch = future_part_a_s1_mismatch | future_part_a_s1_delta
                                future_part_a_s1_contiguous = pl.cast(future_part_a_s1_mismatch == 0, pl.INT32)
                                future_part_a_s1_all_valid = pl.cast(future_part_a_s1_mismatch >= 0, pl.INT32)
                            if future_part_a_s1_contiguous != 0:
                                future_part_a_s1_physical = pl.cast(future_part_a_s1_first, pl.INDEX)
                                raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [0, 0], [future_part_a_s1_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_a_s1_all_valid != 0:
                                for future_part_a_s1_lane in pl.range(0, 32, 2):
                                    future_part_a_s1_a = pl.tile.read(ids_future_part_a_s1, [0, future_part_a_s1_lane])
                                    future_part_a_s1_b = pl.tile.read(ids_future_part_a_s1, [0, future_part_a_s1_lane + 1])
                                    future_part_a_s1_lower = pl.min(future_part_a_s1_a, future_part_a_s1_b)
                                    future_part_a_s1_upper = pl.max(future_part_a_s1_a, future_part_a_s1_b)
                                    future_part_a_s1_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s1_upper == future_part_a_s1_lower + 1:
                                        future_part_a_s1_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s1_pair != 0:
                                        future_part_a_s1_physical_pair = pl.cast(future_part_a_s1_lower, pl.INDEX)
                                        raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane, 0], [future_part_a_s1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_a_s1_physical_a = pl.cast(future_part_a_s1_a, pl.INDEX)
                                        raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane, 0], [future_part_a_s1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_a_s1_physical_b = pl.cast(future_part_a_s1_b, pl.INDEX)
                                        raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane + 1, 0], [future_part_a_s1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_a_s1_lane in pl.range(0, 32, 2):
                                    future_part_a_s1_a = pl.tile.read(ids_future_part_a_s1, [0, future_part_a_s1_lane])
                                    future_part_a_s1_b = pl.tile.read(ids_future_part_a_s1, [0, future_part_a_s1_lane + 1])
                                    future_part_a_s1_lower = pl.min(future_part_a_s1_a, future_part_a_s1_b)
                                    future_part_a_s1_upper = pl.max(future_part_a_s1_a, future_part_a_s1_b)
                                    future_part_a_s1_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s1_lower >= 0:
                                        if future_part_a_s1_upper == future_part_a_s1_lower + 1:
                                            future_part_a_s1_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s1_pair != 0:
                                        future_part_a_s1_physical_pair = pl.cast(future_part_a_s1_lower, pl.INDEX)
                                        raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane, 0], [future_part_a_s1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_a_s1_a >= 0:
                                            future_part_a_s1_physical_a = pl.cast(future_part_a_s1_a, pl.INDEX)
                                            raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane, 0], [future_part_a_s1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s1 = pl.tile.assemble(raw_future_part_a_s1, zero_raw_compressed, [future_part_a_s1_lane, 0])
                                        if future_part_a_s1_b >= 0:
                                            future_part_a_s1_physical_b = pl.cast(future_part_a_s1_b, pl.INDEX)
                                            raw_future_part_a_s1 = pl.tile.gather_row(raw_future_part_a_s1, compressed_flat, [future_part_a_s1_lane + 1, 0], [future_part_a_s1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s1 = pl.tile.assemble(raw_future_part_a_s1, zero_raw_compressed, [future_part_a_s1_lane + 1, 0])
                        else:
                            ids_future_part_a_s2: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[2], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_a_s2_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_a_s2_prefix = pl.tile.slice(raw_future_part_a_s2_storage, [1, 10240], [0, 0])
                            raw_future_part_a_s2 = pl.tile.reshape(raw_future_part_a_s2_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_a_s2_first = pl.tile.read(ids_future_part_a_s2, [0, 0])
                            future_part_a_s2_contiguous = pl.cast(0, pl.INT32)
                            future_part_a_s2_all_valid = pl.cast(0, pl.INT32)
                            if future_part_a_s2_first >= 0:
                                future_part_a_s2_first_wide = pl.cast(future_part_a_s2_first, pl.INT64)
                                future_part_a_s2_mismatch = pl.cast(0, pl.INT64)
                                for future_part_a_s2_check in pl.range(1, 32):
                                    future_part_a_s2_index = pl.tile.read(ids_future_part_a_s2, [0, future_part_a_s2_check])
                                    future_part_a_s2_expected = future_part_a_s2_first_wide + pl.cast(future_part_a_s2_check, pl.INT64)
                                    future_part_a_s2_delta = pl.cast(future_part_a_s2_index, pl.INT64) ^ future_part_a_s2_expected
                                    future_part_a_s2_mismatch = future_part_a_s2_mismatch | future_part_a_s2_delta
                                future_part_a_s2_contiguous = pl.cast(future_part_a_s2_mismatch == 0, pl.INT32)
                                future_part_a_s2_all_valid = pl.cast(future_part_a_s2_mismatch >= 0, pl.INT32)
                            if future_part_a_s2_contiguous != 0:
                                future_part_a_s2_physical = pl.cast(future_part_a_s2_first, pl.INDEX)
                                raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [0, 0], [future_part_a_s2_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_a_s2_all_valid != 0:
                                for future_part_a_s2_lane in pl.range(0, 32, 2):
                                    future_part_a_s2_a = pl.tile.read(ids_future_part_a_s2, [0, future_part_a_s2_lane])
                                    future_part_a_s2_b = pl.tile.read(ids_future_part_a_s2, [0, future_part_a_s2_lane + 1])
                                    future_part_a_s2_lower = pl.min(future_part_a_s2_a, future_part_a_s2_b)
                                    future_part_a_s2_upper = pl.max(future_part_a_s2_a, future_part_a_s2_b)
                                    future_part_a_s2_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s2_upper == future_part_a_s2_lower + 1:
                                        future_part_a_s2_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s2_pair != 0:
                                        future_part_a_s2_physical_pair = pl.cast(future_part_a_s2_lower, pl.INDEX)
                                        raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane, 0], [future_part_a_s2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_a_s2_physical_a = pl.cast(future_part_a_s2_a, pl.INDEX)
                                        raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane, 0], [future_part_a_s2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_a_s2_physical_b = pl.cast(future_part_a_s2_b, pl.INDEX)
                                        raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane + 1, 0], [future_part_a_s2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_a_s2_lane in pl.range(0, 32, 2):
                                    future_part_a_s2_a = pl.tile.read(ids_future_part_a_s2, [0, future_part_a_s2_lane])
                                    future_part_a_s2_b = pl.tile.read(ids_future_part_a_s2, [0, future_part_a_s2_lane + 1])
                                    future_part_a_s2_lower = pl.min(future_part_a_s2_a, future_part_a_s2_b)
                                    future_part_a_s2_upper = pl.max(future_part_a_s2_a, future_part_a_s2_b)
                                    future_part_a_s2_pair = pl.cast(0, pl.INT32)
                                    if future_part_a_s2_lower >= 0:
                                        if future_part_a_s2_upper == future_part_a_s2_lower + 1:
                                            future_part_a_s2_pair = pl.cast(1, pl.INT32)
                                    if future_part_a_s2_pair != 0:
                                        future_part_a_s2_physical_pair = pl.cast(future_part_a_s2_lower, pl.INDEX)
                                        raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane, 0], [future_part_a_s2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_a_s2_a >= 0:
                                            future_part_a_s2_physical_a = pl.cast(future_part_a_s2_a, pl.INDEX)
                                            raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane, 0], [future_part_a_s2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s2 = pl.tile.assemble(raw_future_part_a_s2, zero_raw_compressed, [future_part_a_s2_lane, 0])
                                        if future_part_a_s2_b >= 0:
                                            future_part_a_s2_physical_b = pl.cast(future_part_a_s2_b, pl.INDEX)
                                            raw_future_part_a_s2 = pl.tile.gather_row(raw_future_part_a_s2, compressed_flat, [future_part_a_s2_lane + 1, 0], [future_part_a_s2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_a_s2 = pl.tile.assemble(raw_future_part_a_s2, zero_raw_compressed, [future_part_a_s2_lane + 1, 0])
                    kv_chunk_part_a: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                    if 2 * (part + 2) % 3 == 0:
                        raw_current_compressed0_flat_part_a: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed0_prefix_part_a = pl.tile.slice(raw_current_compressed0_flat_part_a, [1, 10240], [0, 0])
                        raw_current_compressed0_part_a = pl.tile.reshape(raw_current_compressed0_prefix_part_a, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot0_part_a = pl.tile.slice(raw_current_compressed0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot0_part_a = pl.reinterpret_view(fca_raw_chunk_slot0_part_a, pl.FP4E2M1X2)
                        fca_codes_slot0_part_a = fca_packed_slot0_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot0_part_a = pl.cast(fca_codes_slot0_part_a, pl.BF16)
                        fca_scale_words_slot0_part_a = pl.reinterpret_view(fca_raw_chunk_slot0_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot0_part_a = pl.tile.reshape(fca_scale_words_slot0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot0_part_a = pl.tile.reshape(fca_values_slot0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot0_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_17_fca_all_r4_slot0_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_17_fca_row_r4_slot0_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_17_fca_chunk_r4_slot0_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_17_fca_scale_offset_r4_slot0_part_a = raw_stream_17_fca_row_r4_slot0_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_17_fca_scale_offset_r4_slot0_part_a = raw_stream_17_fca_scale_offset_r4_slot0_part_a + raw_stream_17_fca_chunk_r4_slot0_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_17_fca_groups_r4_slot0_part_a = pl.reg.load(fca_scale_flat_slot0_part_a, raw_stream_17_fca_scale_offset_r4_slot0_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_17_fca_factor_bits_r4_slot0_part_a = pl.reg.group_broadcast(raw_stream_17_fca_groups_r4_slot0_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_17_fca_factor_r4_slot0_part_a = pl.reg.interpret_cast(raw_stream_17_fca_factor_bits_r4_slot0_part_a, dtype=pl.BF16)
                                    raw_stream_17_fca_value_offset_r4_slot0_part_a = raw_stream_17_fca_row_r4_slot0_part_a * HEAD_DIM + raw_stream_17_fca_chunk_r4_slot0_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_17_fca_value_r4_slot0_part_a = pl.reg.load(fca_value_flat_slot0_part_a, raw_stream_17_fca_value_offset_r4_slot0_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_17_fca_scaled_r4_slot0_part_a = pl.reg.mul(raw_stream_17_fca_value_r4_slot0_part_a, raw_stream_17_fca_factor_r4_slot0_part_a, raw_stream_17_fca_all_r4_slot0_part_a)
                                    raw_stream_17_fca_target_row_r4_slot0_part_a = 0 + 0 + raw_stream_17_fca_row_r4_slot0_part_a
                                    raw_stream_17_fca_target_offset_r4_slot0_part_a = raw_stream_17_fca_target_row_r4_slot0_part_a * 16 + raw_stream_17_fca_chunk_r4_slot0_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_17_fca_write_r4_slot0_part_a = pl.reg.store(fca_kv_flat_slot0_part_a, raw_stream_17_fca_target_offset_r4_slot0_part_a, raw_stream_17_fca_scaled_r4_slot0_part_a, raw_stream_17_fca_all_r4_slot0_part_a, block_stride=32)
                        fca_raw_chunk_slot0_part_a = pl.tile.slice(raw_current_compressed0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot0_part_a = pl.reinterpret_view(fca_raw_chunk_slot0_part_a, pl.FP4E2M1X2)
                        fca_codes_slot0_part_a = fca_packed_slot0_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot0_part_a = pl.cast(fca_codes_slot0_part_a, pl.BF16)
                        fca_scale_words_slot0_part_a = pl.reinterpret_view(fca_raw_chunk_slot0_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot0_part_a = pl.tile.reshape(fca_scale_words_slot0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot0_part_a = pl.tile.reshape(fca_values_slot0_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot0_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_19_fca_all_r4_slot0_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_19_fca_row_r4_slot0_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_19_fca_chunk_r4_slot0_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_19_fca_scale_offset_r4_slot0_part_a = raw_stream_19_fca_row_r4_slot0_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_19_fca_scale_offset_r4_slot0_part_a = raw_stream_19_fca_scale_offset_r4_slot0_part_a + raw_stream_19_fca_chunk_r4_slot0_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_19_fca_groups_r4_slot0_part_a = pl.reg.load(fca_scale_flat_slot0_part_a, raw_stream_19_fca_scale_offset_r4_slot0_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_19_fca_factor_bits_r4_slot0_part_a = pl.reg.group_broadcast(raw_stream_19_fca_groups_r4_slot0_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_19_fca_factor_r4_slot0_part_a = pl.reg.interpret_cast(raw_stream_19_fca_factor_bits_r4_slot0_part_a, dtype=pl.BF16)
                                    raw_stream_19_fca_value_offset_r4_slot0_part_a = raw_stream_19_fca_row_r4_slot0_part_a * HEAD_DIM + raw_stream_19_fca_chunk_r4_slot0_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_19_fca_value_r4_slot0_part_a = pl.reg.load(fca_value_flat_slot0_part_a, raw_stream_19_fca_value_offset_r4_slot0_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_19_fca_scaled_r4_slot0_part_a = pl.reg.mul(raw_stream_19_fca_value_r4_slot0_part_a, raw_stream_19_fca_factor_r4_slot0_part_a, raw_stream_19_fca_all_r4_slot0_part_a)
                                    raw_stream_19_fca_target_row_r4_slot0_part_a = 0 + 16 + raw_stream_19_fca_row_r4_slot0_part_a
                                    raw_stream_19_fca_target_offset_r4_slot0_part_a = raw_stream_19_fca_target_row_r4_slot0_part_a * 16 + raw_stream_19_fca_chunk_r4_slot0_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_19_fca_write_r4_slot0_part_a = pl.reg.store(fca_kv_flat_slot0_part_a, raw_stream_19_fca_target_offset_r4_slot0_part_a, raw_stream_19_fca_scaled_r4_slot0_part_a, raw_stream_19_fca_all_r4_slot0_part_a, block_stride=32)
                    elif 2 * (part + 2) % 3 == 1:
                        raw_current_compressed1_flat_part_a: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed1_prefix_part_a = pl.tile.slice(raw_current_compressed1_flat_part_a, [1, 10240], [0, 0])
                        raw_current_compressed1_part_a = pl.tile.reshape(raw_current_compressed1_prefix_part_a, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot1_part_a = pl.tile.slice(raw_current_compressed1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot1_part_a = pl.reinterpret_view(fca_raw_chunk_slot1_part_a, pl.FP4E2M1X2)
                        fca_codes_slot1_part_a = fca_packed_slot1_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot1_part_a = pl.cast(fca_codes_slot1_part_a, pl.BF16)
                        fca_scale_words_slot1_part_a = pl.reinterpret_view(fca_raw_chunk_slot1_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot1_part_a = pl.tile.reshape(fca_scale_words_slot1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot1_part_a = pl.tile.reshape(fca_values_slot1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot1_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_25_fca_all_r4_slot1_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_25_fca_row_r4_slot1_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_25_fca_chunk_r4_slot1_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_25_fca_scale_offset_r4_slot1_part_a = raw_stream_25_fca_row_r4_slot1_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_25_fca_scale_offset_r4_slot1_part_a = raw_stream_25_fca_scale_offset_r4_slot1_part_a + raw_stream_25_fca_chunk_r4_slot1_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_25_fca_groups_r4_slot1_part_a = pl.reg.load(fca_scale_flat_slot1_part_a, raw_stream_25_fca_scale_offset_r4_slot1_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_25_fca_factor_bits_r4_slot1_part_a = pl.reg.group_broadcast(raw_stream_25_fca_groups_r4_slot1_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_25_fca_factor_r4_slot1_part_a = pl.reg.interpret_cast(raw_stream_25_fca_factor_bits_r4_slot1_part_a, dtype=pl.BF16)
                                    raw_stream_25_fca_value_offset_r4_slot1_part_a = raw_stream_25_fca_row_r4_slot1_part_a * HEAD_DIM + raw_stream_25_fca_chunk_r4_slot1_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_25_fca_value_r4_slot1_part_a = pl.reg.load(fca_value_flat_slot1_part_a, raw_stream_25_fca_value_offset_r4_slot1_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_25_fca_scaled_r4_slot1_part_a = pl.reg.mul(raw_stream_25_fca_value_r4_slot1_part_a, raw_stream_25_fca_factor_r4_slot1_part_a, raw_stream_25_fca_all_r4_slot1_part_a)
                                    raw_stream_25_fca_target_row_r4_slot1_part_a = 0 + 0 + raw_stream_25_fca_row_r4_slot1_part_a
                                    raw_stream_25_fca_target_offset_r4_slot1_part_a = raw_stream_25_fca_target_row_r4_slot1_part_a * 16 + raw_stream_25_fca_chunk_r4_slot1_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_25_fca_write_r4_slot1_part_a = pl.reg.store(fca_kv_flat_slot1_part_a, raw_stream_25_fca_target_offset_r4_slot1_part_a, raw_stream_25_fca_scaled_r4_slot1_part_a, raw_stream_25_fca_all_r4_slot1_part_a, block_stride=32)
                        fca_raw_chunk_slot1_part_a = pl.tile.slice(raw_current_compressed1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot1_part_a = pl.reinterpret_view(fca_raw_chunk_slot1_part_a, pl.FP4E2M1X2)
                        fca_codes_slot1_part_a = fca_packed_slot1_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot1_part_a = pl.cast(fca_codes_slot1_part_a, pl.BF16)
                        fca_scale_words_slot1_part_a = pl.reinterpret_view(fca_raw_chunk_slot1_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot1_part_a = pl.tile.reshape(fca_scale_words_slot1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot1_part_a = pl.tile.reshape(fca_values_slot1_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot1_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_27_fca_all_r4_slot1_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_27_fca_row_r4_slot1_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_27_fca_chunk_r4_slot1_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_27_fca_scale_offset_r4_slot1_part_a = raw_stream_27_fca_row_r4_slot1_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_27_fca_scale_offset_r4_slot1_part_a = raw_stream_27_fca_scale_offset_r4_slot1_part_a + raw_stream_27_fca_chunk_r4_slot1_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_27_fca_groups_r4_slot1_part_a = pl.reg.load(fca_scale_flat_slot1_part_a, raw_stream_27_fca_scale_offset_r4_slot1_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_27_fca_factor_bits_r4_slot1_part_a = pl.reg.group_broadcast(raw_stream_27_fca_groups_r4_slot1_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_27_fca_factor_r4_slot1_part_a = pl.reg.interpret_cast(raw_stream_27_fca_factor_bits_r4_slot1_part_a, dtype=pl.BF16)
                                    raw_stream_27_fca_value_offset_r4_slot1_part_a = raw_stream_27_fca_row_r4_slot1_part_a * HEAD_DIM + raw_stream_27_fca_chunk_r4_slot1_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_27_fca_value_r4_slot1_part_a = pl.reg.load(fca_value_flat_slot1_part_a, raw_stream_27_fca_value_offset_r4_slot1_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_27_fca_scaled_r4_slot1_part_a = pl.reg.mul(raw_stream_27_fca_value_r4_slot1_part_a, raw_stream_27_fca_factor_r4_slot1_part_a, raw_stream_27_fca_all_r4_slot1_part_a)
                                    raw_stream_27_fca_target_row_r4_slot1_part_a = 0 + 16 + raw_stream_27_fca_row_r4_slot1_part_a
                                    raw_stream_27_fca_target_offset_r4_slot1_part_a = raw_stream_27_fca_target_row_r4_slot1_part_a * 16 + raw_stream_27_fca_chunk_r4_slot1_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_27_fca_write_r4_slot1_part_a = pl.reg.store(fca_kv_flat_slot1_part_a, raw_stream_27_fca_target_offset_r4_slot1_part_a, raw_stream_27_fca_scaled_r4_slot1_part_a, raw_stream_27_fca_all_r4_slot1_part_a, block_stride=32)
                    else:
                        raw_current_compressed2_flat_part_a: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed2_prefix_part_a = pl.tile.slice(raw_current_compressed2_flat_part_a, [1, 10240], [0, 0])
                        raw_current_compressed2_part_a = pl.tile.reshape(raw_current_compressed2_prefix_part_a, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot2_part_a = pl.tile.slice(raw_current_compressed2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot2_part_a = pl.reinterpret_view(fca_raw_chunk_slot2_part_a, pl.FP4E2M1X2)
                        fca_codes_slot2_part_a = fca_packed_slot2_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot2_part_a = pl.cast(fca_codes_slot2_part_a, pl.BF16)
                        fca_scale_words_slot2_part_a = pl.reinterpret_view(fca_raw_chunk_slot2_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot2_part_a = pl.tile.reshape(fca_scale_words_slot2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot2_part_a = pl.tile.reshape(fca_values_slot2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot2_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_29_fca_all_r4_slot2_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_29_fca_row_r4_slot2_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_29_fca_chunk_r4_slot2_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_29_fca_scale_offset_r4_slot2_part_a = raw_stream_29_fca_row_r4_slot2_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_29_fca_scale_offset_r4_slot2_part_a = raw_stream_29_fca_scale_offset_r4_slot2_part_a + raw_stream_29_fca_chunk_r4_slot2_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_29_fca_groups_r4_slot2_part_a = pl.reg.load(fca_scale_flat_slot2_part_a, raw_stream_29_fca_scale_offset_r4_slot2_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_29_fca_factor_bits_r4_slot2_part_a = pl.reg.group_broadcast(raw_stream_29_fca_groups_r4_slot2_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_29_fca_factor_r4_slot2_part_a = pl.reg.interpret_cast(raw_stream_29_fca_factor_bits_r4_slot2_part_a, dtype=pl.BF16)
                                    raw_stream_29_fca_value_offset_r4_slot2_part_a = raw_stream_29_fca_row_r4_slot2_part_a * HEAD_DIM + raw_stream_29_fca_chunk_r4_slot2_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_29_fca_value_r4_slot2_part_a = pl.reg.load(fca_value_flat_slot2_part_a, raw_stream_29_fca_value_offset_r4_slot2_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_29_fca_scaled_r4_slot2_part_a = pl.reg.mul(raw_stream_29_fca_value_r4_slot2_part_a, raw_stream_29_fca_factor_r4_slot2_part_a, raw_stream_29_fca_all_r4_slot2_part_a)
                                    raw_stream_29_fca_target_row_r4_slot2_part_a = 0 + 0 + raw_stream_29_fca_row_r4_slot2_part_a
                                    raw_stream_29_fca_target_offset_r4_slot2_part_a = raw_stream_29_fca_target_row_r4_slot2_part_a * 16 + raw_stream_29_fca_chunk_r4_slot2_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_29_fca_write_r4_slot2_part_a = pl.reg.store(fca_kv_flat_slot2_part_a, raw_stream_29_fca_target_offset_r4_slot2_part_a, raw_stream_29_fca_scaled_r4_slot2_part_a, raw_stream_29_fca_all_r4_slot2_part_a, block_stride=32)
                        fca_raw_chunk_slot2_part_a = pl.tile.slice(raw_current_compressed2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot2_part_a = pl.reinterpret_view(fca_raw_chunk_slot2_part_a, pl.FP4E2M1X2)
                        fca_codes_slot2_part_a = fca_packed_slot2_part_a[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot2_part_a = pl.cast(fca_codes_slot2_part_a, pl.BF16)
                        fca_scale_words_slot2_part_a = pl.reinterpret_view(fca_raw_chunk_slot2_part_a, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot2_part_a = pl.tile.reshape(fca_scale_words_slot2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot2_part_a = pl.tile.reshape(fca_values_slot2_part_a, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot2_part_a = pl.tile.reshape(kv_chunk_part_a, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_31_fca_all_r4_slot2_part_a = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_31_fca_row_r4_slot2_part_a in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_31_fca_chunk_r4_slot2_part_a in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_31_fca_scale_offset_r4_slot2_part_a = raw_stream_31_fca_row_r4_slot2_part_a * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_31_fca_scale_offset_r4_slot2_part_a = raw_stream_31_fca_scale_offset_r4_slot2_part_a + raw_stream_31_fca_chunk_r4_slot2_part_a * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_31_fca_groups_r4_slot2_part_a = pl.reg.load(fca_scale_flat_slot2_part_a, raw_stream_31_fca_scale_offset_r4_slot2_part_a, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_31_fca_factor_bits_r4_slot2_part_a = pl.reg.group_broadcast(raw_stream_31_fca_groups_r4_slot2_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_31_fca_factor_r4_slot2_part_a = pl.reg.interpret_cast(raw_stream_31_fca_factor_bits_r4_slot2_part_a, dtype=pl.BF16)
                                    raw_stream_31_fca_value_offset_r4_slot2_part_a = raw_stream_31_fca_row_r4_slot2_part_a * HEAD_DIM + raw_stream_31_fca_chunk_r4_slot2_part_a * SPARSE_VF_REG_TILE
                                    raw_stream_31_fca_value_r4_slot2_part_a = pl.reg.load(fca_value_flat_slot2_part_a, raw_stream_31_fca_value_offset_r4_slot2_part_a, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_31_fca_scaled_r4_slot2_part_a = pl.reg.mul(raw_stream_31_fca_value_r4_slot2_part_a, raw_stream_31_fca_factor_r4_slot2_part_a, raw_stream_31_fca_all_r4_slot2_part_a)
                                    raw_stream_31_fca_target_row_r4_slot2_part_a = 0 + 16 + raw_stream_31_fca_row_r4_slot2_part_a
                                    raw_stream_31_fca_target_offset_r4_slot2_part_a = raw_stream_31_fca_target_row_r4_slot2_part_a * 16 + raw_stream_31_fca_chunk_r4_slot2_part_a * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_31_fca_write_r4_slot2_part_a = pl.reg.store(fca_kv_flat_slot2_part_a, raw_stream_31_fca_target_offset_r4_slot2_part_a, raw_stream_31_fca_scaled_r4_slot2_part_a, raw_stream_31_fca_all_r4_slot2_part_a, block_stride=32)
                    kv_nz_part_a: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                    pl.system.tpush_to_aic(kv_nz_part_a, split=1, id=1)
                    if 2 * (part + 2) + 1 + 2 < parts * 2:
                        if (2 * (part + 2) + 1 + 2) % 3 == 0:
                            ids_future_part_b_s0: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[0], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 1 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_b_s0_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_b_s0_prefix = pl.tile.slice(raw_future_part_b_s0_storage, [1, 10240], [0, 0])
                            raw_future_part_b_s0 = pl.tile.reshape(raw_future_part_b_s0_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_b_s0_first = pl.tile.read(ids_future_part_b_s0, [0, 0])
                            future_part_b_s0_contiguous = pl.cast(0, pl.INT32)
                            future_part_b_s0_all_valid = pl.cast(0, pl.INT32)
                            if future_part_b_s0_first >= 0:
                                future_part_b_s0_first_wide = pl.cast(future_part_b_s0_first, pl.INT64)
                                future_part_b_s0_mismatch = pl.cast(0, pl.INT64)
                                for future_part_b_s0_check in pl.range(1, 32):
                                    future_part_b_s0_index = pl.tile.read(ids_future_part_b_s0, [0, future_part_b_s0_check])
                                    future_part_b_s0_expected = future_part_b_s0_first_wide + pl.cast(future_part_b_s0_check, pl.INT64)
                                    future_part_b_s0_delta = pl.cast(future_part_b_s0_index, pl.INT64) ^ future_part_b_s0_expected
                                    future_part_b_s0_mismatch = future_part_b_s0_mismatch | future_part_b_s0_delta
                                future_part_b_s0_contiguous = pl.cast(future_part_b_s0_mismatch == 0, pl.INT32)
                                future_part_b_s0_all_valid = pl.cast(future_part_b_s0_mismatch >= 0, pl.INT32)
                            if future_part_b_s0_contiguous != 0:
                                future_part_b_s0_physical = pl.cast(future_part_b_s0_first, pl.INDEX)
                                raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [0, 0], [future_part_b_s0_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_b_s0_all_valid != 0:
                                for future_part_b_s0_lane in pl.range(0, 32, 2):
                                    future_part_b_s0_a = pl.tile.read(ids_future_part_b_s0, [0, future_part_b_s0_lane])
                                    future_part_b_s0_b = pl.tile.read(ids_future_part_b_s0, [0, future_part_b_s0_lane + 1])
                                    future_part_b_s0_lower = pl.min(future_part_b_s0_a, future_part_b_s0_b)
                                    future_part_b_s0_upper = pl.max(future_part_b_s0_a, future_part_b_s0_b)
                                    future_part_b_s0_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s0_upper == future_part_b_s0_lower + 1:
                                        future_part_b_s0_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s0_pair != 0:
                                        future_part_b_s0_physical_pair = pl.cast(future_part_b_s0_lower, pl.INDEX)
                                        raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane, 0], [future_part_b_s0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_b_s0_physical_a = pl.cast(future_part_b_s0_a, pl.INDEX)
                                        raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane, 0], [future_part_b_s0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_b_s0_physical_b = pl.cast(future_part_b_s0_b, pl.INDEX)
                                        raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane + 1, 0], [future_part_b_s0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_b_s0_lane in pl.range(0, 32, 2):
                                    future_part_b_s0_a = pl.tile.read(ids_future_part_b_s0, [0, future_part_b_s0_lane])
                                    future_part_b_s0_b = pl.tile.read(ids_future_part_b_s0, [0, future_part_b_s0_lane + 1])
                                    future_part_b_s0_lower = pl.min(future_part_b_s0_a, future_part_b_s0_b)
                                    future_part_b_s0_upper = pl.max(future_part_b_s0_a, future_part_b_s0_b)
                                    future_part_b_s0_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s0_lower >= 0:
                                        if future_part_b_s0_upper == future_part_b_s0_lower + 1:
                                            future_part_b_s0_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s0_pair != 0:
                                        future_part_b_s0_physical_pair = pl.cast(future_part_b_s0_lower, pl.INDEX)
                                        raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane, 0], [future_part_b_s0_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_b_s0_a >= 0:
                                            future_part_b_s0_physical_a = pl.cast(future_part_b_s0_a, pl.INDEX)
                                            raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane, 0], [future_part_b_s0_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s0 = pl.tile.assemble(raw_future_part_b_s0, zero_raw_compressed, [future_part_b_s0_lane, 0])
                                        if future_part_b_s0_b >= 0:
                                            future_part_b_s0_physical_b = pl.cast(future_part_b_s0_b, pl.INDEX)
                                            raw_future_part_b_s0 = pl.tile.gather_row(raw_future_part_b_s0, compressed_flat, [future_part_b_s0_lane + 1, 0], [future_part_b_s0_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s0 = pl.tile.assemble(raw_future_part_b_s0, zero_raw_compressed, [future_part_b_s0_lane + 1, 0])
                        elif (2 * (part + 2) + 1 + 2) % 3 == 1:
                            ids_future_part_b_s1: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[1], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 1 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_b_s1_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_b_s1_prefix = pl.tile.slice(raw_future_part_b_s1_storage, [1, 10240], [0, 0])
                            raw_future_part_b_s1 = pl.tile.reshape(raw_future_part_b_s1_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_b_s1_first = pl.tile.read(ids_future_part_b_s1, [0, 0])
                            future_part_b_s1_contiguous = pl.cast(0, pl.INT32)
                            future_part_b_s1_all_valid = pl.cast(0, pl.INT32)
                            if future_part_b_s1_first >= 0:
                                future_part_b_s1_first_wide = pl.cast(future_part_b_s1_first, pl.INT64)
                                future_part_b_s1_mismatch = pl.cast(0, pl.INT64)
                                for future_part_b_s1_check in pl.range(1, 32):
                                    future_part_b_s1_index = pl.tile.read(ids_future_part_b_s1, [0, future_part_b_s1_check])
                                    future_part_b_s1_expected = future_part_b_s1_first_wide + pl.cast(future_part_b_s1_check, pl.INT64)
                                    future_part_b_s1_delta = pl.cast(future_part_b_s1_index, pl.INT64) ^ future_part_b_s1_expected
                                    future_part_b_s1_mismatch = future_part_b_s1_mismatch | future_part_b_s1_delta
                                future_part_b_s1_contiguous = pl.cast(future_part_b_s1_mismatch == 0, pl.INT32)
                                future_part_b_s1_all_valid = pl.cast(future_part_b_s1_mismatch >= 0, pl.INT32)
                            if future_part_b_s1_contiguous != 0:
                                future_part_b_s1_physical = pl.cast(future_part_b_s1_first, pl.INDEX)
                                raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [0, 0], [future_part_b_s1_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_b_s1_all_valid != 0:
                                for future_part_b_s1_lane in pl.range(0, 32, 2):
                                    future_part_b_s1_a = pl.tile.read(ids_future_part_b_s1, [0, future_part_b_s1_lane])
                                    future_part_b_s1_b = pl.tile.read(ids_future_part_b_s1, [0, future_part_b_s1_lane + 1])
                                    future_part_b_s1_lower = pl.min(future_part_b_s1_a, future_part_b_s1_b)
                                    future_part_b_s1_upper = pl.max(future_part_b_s1_a, future_part_b_s1_b)
                                    future_part_b_s1_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s1_upper == future_part_b_s1_lower + 1:
                                        future_part_b_s1_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s1_pair != 0:
                                        future_part_b_s1_physical_pair = pl.cast(future_part_b_s1_lower, pl.INDEX)
                                        raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane, 0], [future_part_b_s1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_b_s1_physical_a = pl.cast(future_part_b_s1_a, pl.INDEX)
                                        raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane, 0], [future_part_b_s1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_b_s1_physical_b = pl.cast(future_part_b_s1_b, pl.INDEX)
                                        raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane + 1, 0], [future_part_b_s1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_b_s1_lane in pl.range(0, 32, 2):
                                    future_part_b_s1_a = pl.tile.read(ids_future_part_b_s1, [0, future_part_b_s1_lane])
                                    future_part_b_s1_b = pl.tile.read(ids_future_part_b_s1, [0, future_part_b_s1_lane + 1])
                                    future_part_b_s1_lower = pl.min(future_part_b_s1_a, future_part_b_s1_b)
                                    future_part_b_s1_upper = pl.max(future_part_b_s1_a, future_part_b_s1_b)
                                    future_part_b_s1_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s1_lower >= 0:
                                        if future_part_b_s1_upper == future_part_b_s1_lower + 1:
                                            future_part_b_s1_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s1_pair != 0:
                                        future_part_b_s1_physical_pair = pl.cast(future_part_b_s1_lower, pl.INDEX)
                                        raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane, 0], [future_part_b_s1_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_b_s1_a >= 0:
                                            future_part_b_s1_physical_a = pl.cast(future_part_b_s1_a, pl.INDEX)
                                            raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane, 0], [future_part_b_s1_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s1 = pl.tile.assemble(raw_future_part_b_s1, zero_raw_compressed, [future_part_b_s1_lane, 0])
                                        if future_part_b_s1_b >= 0:
                                            future_part_b_s1_physical_b = pl.cast(future_part_b_s1_b, pl.INDEX)
                                            raw_future_part_b_s1 = pl.tile.gather_row(raw_future_part_b_s1, compressed_flat, [future_part_b_s1_lane + 1, 0], [future_part_b_s1_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s1 = pl.tile.assemble(raw_future_part_b_s1, zero_raw_compressed, [future_part_b_s1_lane + 1, 0])
                        else:
                            ids_future_part_b_s2: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[2], pl.Mem.Vec] = pl.tile.load(compressed_indices, [token, (2 * (part + 2) + 1 + 2) * 64 + aiv_id * 32 - WINDOW_KEYS], [1, 32])
                            raw_future_part_b_s2_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                            raw_future_part_b_s2_prefix = pl.tile.slice(raw_future_part_b_s2_storage, [1, 10240], [0, 0])
                            raw_future_part_b_s2 = pl.tile.reshape(raw_future_part_b_s2_prefix, [32, COMPRESSED_CACHE_BYTES])
                            future_part_b_s2_first = pl.tile.read(ids_future_part_b_s2, [0, 0])
                            future_part_b_s2_contiguous = pl.cast(0, pl.INT32)
                            future_part_b_s2_all_valid = pl.cast(0, pl.INT32)
                            if future_part_b_s2_first >= 0:
                                future_part_b_s2_first_wide = pl.cast(future_part_b_s2_first, pl.INT64)
                                future_part_b_s2_mismatch = pl.cast(0, pl.INT64)
                                for future_part_b_s2_check in pl.range(1, 32):
                                    future_part_b_s2_index = pl.tile.read(ids_future_part_b_s2, [0, future_part_b_s2_check])
                                    future_part_b_s2_expected = future_part_b_s2_first_wide + pl.cast(future_part_b_s2_check, pl.INT64)
                                    future_part_b_s2_delta = pl.cast(future_part_b_s2_index, pl.INT64) ^ future_part_b_s2_expected
                                    future_part_b_s2_mismatch = future_part_b_s2_mismatch | future_part_b_s2_delta
                                future_part_b_s2_contiguous = pl.cast(future_part_b_s2_mismatch == 0, pl.INT32)
                                future_part_b_s2_all_valid = pl.cast(future_part_b_s2_mismatch >= 0, pl.INT32)
                            if future_part_b_s2_contiguous != 0:
                                future_part_b_s2_physical = pl.cast(future_part_b_s2_first, pl.INDEX)
                                raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [0, 0], [future_part_b_s2_physical, 0], [32, COMPRESSED_CACHE_BYTES])
                            elif future_part_b_s2_all_valid != 0:
                                for future_part_b_s2_lane in pl.range(0, 32, 2):
                                    future_part_b_s2_a = pl.tile.read(ids_future_part_b_s2, [0, future_part_b_s2_lane])
                                    future_part_b_s2_b = pl.tile.read(ids_future_part_b_s2, [0, future_part_b_s2_lane + 1])
                                    future_part_b_s2_lower = pl.min(future_part_b_s2_a, future_part_b_s2_b)
                                    future_part_b_s2_upper = pl.max(future_part_b_s2_a, future_part_b_s2_b)
                                    future_part_b_s2_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s2_upper == future_part_b_s2_lower + 1:
                                        future_part_b_s2_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s2_pair != 0:
                                        future_part_b_s2_physical_pair = pl.cast(future_part_b_s2_lower, pl.INDEX)
                                        raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane, 0], [future_part_b_s2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        future_part_b_s2_physical_a = pl.cast(future_part_b_s2_a, pl.INDEX)
                                        raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane, 0], [future_part_b_s2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        future_part_b_s2_physical_b = pl.cast(future_part_b_s2_b, pl.INDEX)
                                        raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane + 1, 0], [future_part_b_s2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            else:
                                for future_part_b_s2_lane in pl.range(0, 32, 2):
                                    future_part_b_s2_a = pl.tile.read(ids_future_part_b_s2, [0, future_part_b_s2_lane])
                                    future_part_b_s2_b = pl.tile.read(ids_future_part_b_s2, [0, future_part_b_s2_lane + 1])
                                    future_part_b_s2_lower = pl.min(future_part_b_s2_a, future_part_b_s2_b)
                                    future_part_b_s2_upper = pl.max(future_part_b_s2_a, future_part_b_s2_b)
                                    future_part_b_s2_pair = pl.cast(0, pl.INT32)
                                    if future_part_b_s2_lower >= 0:
                                        if future_part_b_s2_upper == future_part_b_s2_lower + 1:
                                            future_part_b_s2_pair = pl.cast(1, pl.INT32)
                                    if future_part_b_s2_pair != 0:
                                        future_part_b_s2_physical_pair = pl.cast(future_part_b_s2_lower, pl.INDEX)
                                        raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane, 0], [future_part_b_s2_physical_pair, 0], [2, COMPRESSED_CACHE_BYTES])
                                    else:
                                        if future_part_b_s2_a >= 0:
                                            future_part_b_s2_physical_a = pl.cast(future_part_b_s2_a, pl.INDEX)
                                            raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane, 0], [future_part_b_s2_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s2 = pl.tile.assemble(raw_future_part_b_s2, zero_raw_compressed, [future_part_b_s2_lane, 0])
                                        if future_part_b_s2_b >= 0:
                                            future_part_b_s2_physical_b = pl.cast(future_part_b_s2_b, pl.INDEX)
                                            raw_future_part_b_s2 = pl.tile.gather_row(raw_future_part_b_s2, compressed_flat, [future_part_b_s2_lane + 1, 0], [future_part_b_s2_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                                        else:
                                            raw_future_part_b_s2 = pl.tile.assemble(raw_future_part_b_s2, zero_raw_compressed, [future_part_b_s2_lane + 1, 0])
                    kv_chunk_part_b: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                    if (2 * (part + 2) + 1) % 3 == 0:
                        raw_current_compressed0_flat_part_b: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed0_prefix_part_b = pl.tile.slice(raw_current_compressed0_flat_part_b, [1, 10240], [0, 0])
                        raw_current_compressed0_part_b = pl.tile.reshape(raw_current_compressed0_prefix_part_b, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot0_part_b = pl.tile.slice(raw_current_compressed0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot0_part_b = pl.reinterpret_view(fca_raw_chunk_slot0_part_b, pl.FP4E2M1X2)
                        fca_codes_slot0_part_b = fca_packed_slot0_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot0_part_b = pl.cast(fca_codes_slot0_part_b, pl.BF16)
                        fca_scale_words_slot0_part_b = pl.reinterpret_view(fca_raw_chunk_slot0_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot0_part_b = pl.tile.reshape(fca_scale_words_slot0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot0_part_b = pl.tile.reshape(fca_values_slot0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot0_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_21_fca_all_r4_slot0_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_21_fca_row_r4_slot0_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_21_fca_chunk_r4_slot0_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_21_fca_scale_offset_r4_slot0_part_b = raw_stream_21_fca_row_r4_slot0_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_21_fca_scale_offset_r4_slot0_part_b = raw_stream_21_fca_scale_offset_r4_slot0_part_b + raw_stream_21_fca_chunk_r4_slot0_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_21_fca_groups_r4_slot0_part_b = pl.reg.load(fca_scale_flat_slot0_part_b, raw_stream_21_fca_scale_offset_r4_slot0_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_21_fca_factor_bits_r4_slot0_part_b = pl.reg.group_broadcast(raw_stream_21_fca_groups_r4_slot0_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_21_fca_factor_r4_slot0_part_b = pl.reg.interpret_cast(raw_stream_21_fca_factor_bits_r4_slot0_part_b, dtype=pl.BF16)
                                    raw_stream_21_fca_value_offset_r4_slot0_part_b = raw_stream_21_fca_row_r4_slot0_part_b * HEAD_DIM + raw_stream_21_fca_chunk_r4_slot0_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_21_fca_value_r4_slot0_part_b = pl.reg.load(fca_value_flat_slot0_part_b, raw_stream_21_fca_value_offset_r4_slot0_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_21_fca_scaled_r4_slot0_part_b = pl.reg.mul(raw_stream_21_fca_value_r4_slot0_part_b, raw_stream_21_fca_factor_r4_slot0_part_b, raw_stream_21_fca_all_r4_slot0_part_b)
                                    raw_stream_21_fca_target_row_r4_slot0_part_b = 0 + 0 + raw_stream_21_fca_row_r4_slot0_part_b
                                    raw_stream_21_fca_target_offset_r4_slot0_part_b = raw_stream_21_fca_target_row_r4_slot0_part_b * 16 + raw_stream_21_fca_chunk_r4_slot0_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_21_fca_write_r4_slot0_part_b = pl.reg.store(fca_kv_flat_slot0_part_b, raw_stream_21_fca_target_offset_r4_slot0_part_b, raw_stream_21_fca_scaled_r4_slot0_part_b, raw_stream_21_fca_all_r4_slot0_part_b, block_stride=32)
                        fca_raw_chunk_slot0_part_b = pl.tile.slice(raw_current_compressed0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot0_part_b = pl.reinterpret_view(fca_raw_chunk_slot0_part_b, pl.FP4E2M1X2)
                        fca_codes_slot0_part_b = fca_packed_slot0_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot0_part_b = pl.cast(fca_codes_slot0_part_b, pl.BF16)
                        fca_scale_words_slot0_part_b = pl.reinterpret_view(fca_raw_chunk_slot0_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot0_part_b = pl.tile.reshape(fca_scale_words_slot0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot0_part_b = pl.tile.reshape(fca_values_slot0_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot0_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_23_fca_all_r4_slot0_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_23_fca_row_r4_slot0_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_23_fca_chunk_r4_slot0_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_23_fca_scale_offset_r4_slot0_part_b = raw_stream_23_fca_row_r4_slot0_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_23_fca_scale_offset_r4_slot0_part_b = raw_stream_23_fca_scale_offset_r4_slot0_part_b + raw_stream_23_fca_chunk_r4_slot0_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_23_fca_groups_r4_slot0_part_b = pl.reg.load(fca_scale_flat_slot0_part_b, raw_stream_23_fca_scale_offset_r4_slot0_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_23_fca_factor_bits_r4_slot0_part_b = pl.reg.group_broadcast(raw_stream_23_fca_groups_r4_slot0_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_23_fca_factor_r4_slot0_part_b = pl.reg.interpret_cast(raw_stream_23_fca_factor_bits_r4_slot0_part_b, dtype=pl.BF16)
                                    raw_stream_23_fca_value_offset_r4_slot0_part_b = raw_stream_23_fca_row_r4_slot0_part_b * HEAD_DIM + raw_stream_23_fca_chunk_r4_slot0_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_23_fca_value_r4_slot0_part_b = pl.reg.load(fca_value_flat_slot0_part_b, raw_stream_23_fca_value_offset_r4_slot0_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_23_fca_scaled_r4_slot0_part_b = pl.reg.mul(raw_stream_23_fca_value_r4_slot0_part_b, raw_stream_23_fca_factor_r4_slot0_part_b, raw_stream_23_fca_all_r4_slot0_part_b)
                                    raw_stream_23_fca_target_row_r4_slot0_part_b = 0 + 16 + raw_stream_23_fca_row_r4_slot0_part_b
                                    raw_stream_23_fca_target_offset_r4_slot0_part_b = raw_stream_23_fca_target_row_r4_slot0_part_b * 16 + raw_stream_23_fca_chunk_r4_slot0_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_23_fca_write_r4_slot0_part_b = pl.reg.store(fca_kv_flat_slot0_part_b, raw_stream_23_fca_target_offset_r4_slot0_part_b, raw_stream_23_fca_scaled_r4_slot0_part_b, raw_stream_23_fca_all_r4_slot0_part_b, block_stride=32)
                    elif (2 * (part + 2) + 1) % 3 == 1:
                        raw_current_compressed1_flat_part_b: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[1], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed1_prefix_part_b = pl.tile.slice(raw_current_compressed1_flat_part_b, [1, 10240], [0, 0])
                        raw_current_compressed1_part_b = pl.tile.reshape(raw_current_compressed1_prefix_part_b, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot1_part_b = pl.tile.slice(raw_current_compressed1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot1_part_b = pl.reinterpret_view(fca_raw_chunk_slot1_part_b, pl.FP4E2M1X2)
                        fca_codes_slot1_part_b = fca_packed_slot1_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot1_part_b = pl.cast(fca_codes_slot1_part_b, pl.BF16)
                        fca_scale_words_slot1_part_b = pl.reinterpret_view(fca_raw_chunk_slot1_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot1_part_b = pl.tile.reshape(fca_scale_words_slot1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot1_part_b = pl.tile.reshape(fca_values_slot1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot1_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_33_fca_all_r4_slot1_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_33_fca_row_r4_slot1_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_33_fca_chunk_r4_slot1_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_33_fca_scale_offset_r4_slot1_part_b = raw_stream_33_fca_row_r4_slot1_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_33_fca_scale_offset_r4_slot1_part_b = raw_stream_33_fca_scale_offset_r4_slot1_part_b + raw_stream_33_fca_chunk_r4_slot1_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_33_fca_groups_r4_slot1_part_b = pl.reg.load(fca_scale_flat_slot1_part_b, raw_stream_33_fca_scale_offset_r4_slot1_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_33_fca_factor_bits_r4_slot1_part_b = pl.reg.group_broadcast(raw_stream_33_fca_groups_r4_slot1_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_33_fca_factor_r4_slot1_part_b = pl.reg.interpret_cast(raw_stream_33_fca_factor_bits_r4_slot1_part_b, dtype=pl.BF16)
                                    raw_stream_33_fca_value_offset_r4_slot1_part_b = raw_stream_33_fca_row_r4_slot1_part_b * HEAD_DIM + raw_stream_33_fca_chunk_r4_slot1_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_33_fca_value_r4_slot1_part_b = pl.reg.load(fca_value_flat_slot1_part_b, raw_stream_33_fca_value_offset_r4_slot1_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_33_fca_scaled_r4_slot1_part_b = pl.reg.mul(raw_stream_33_fca_value_r4_slot1_part_b, raw_stream_33_fca_factor_r4_slot1_part_b, raw_stream_33_fca_all_r4_slot1_part_b)
                                    raw_stream_33_fca_target_row_r4_slot1_part_b = 0 + 0 + raw_stream_33_fca_row_r4_slot1_part_b
                                    raw_stream_33_fca_target_offset_r4_slot1_part_b = raw_stream_33_fca_target_row_r4_slot1_part_b * 16 + raw_stream_33_fca_chunk_r4_slot1_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_33_fca_write_r4_slot1_part_b = pl.reg.store(fca_kv_flat_slot1_part_b, raw_stream_33_fca_target_offset_r4_slot1_part_b, raw_stream_33_fca_scaled_r4_slot1_part_b, raw_stream_33_fca_all_r4_slot1_part_b, block_stride=32)
                        fca_raw_chunk_slot1_part_b = pl.tile.slice(raw_current_compressed1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot1_part_b = pl.reinterpret_view(fca_raw_chunk_slot1_part_b, pl.FP4E2M1X2)
                        fca_codes_slot1_part_b = fca_packed_slot1_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot1_part_b = pl.cast(fca_codes_slot1_part_b, pl.BF16)
                        fca_scale_words_slot1_part_b = pl.reinterpret_view(fca_raw_chunk_slot1_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot1_part_b = pl.tile.reshape(fca_scale_words_slot1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot1_part_b = pl.tile.reshape(fca_values_slot1_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot1_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_35_fca_all_r4_slot1_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_35_fca_row_r4_slot1_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_35_fca_chunk_r4_slot1_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_35_fca_scale_offset_r4_slot1_part_b = raw_stream_35_fca_row_r4_slot1_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_35_fca_scale_offset_r4_slot1_part_b = raw_stream_35_fca_scale_offset_r4_slot1_part_b + raw_stream_35_fca_chunk_r4_slot1_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_35_fca_groups_r4_slot1_part_b = pl.reg.load(fca_scale_flat_slot1_part_b, raw_stream_35_fca_scale_offset_r4_slot1_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_35_fca_factor_bits_r4_slot1_part_b = pl.reg.group_broadcast(raw_stream_35_fca_groups_r4_slot1_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_35_fca_factor_r4_slot1_part_b = pl.reg.interpret_cast(raw_stream_35_fca_factor_bits_r4_slot1_part_b, dtype=pl.BF16)
                                    raw_stream_35_fca_value_offset_r4_slot1_part_b = raw_stream_35_fca_row_r4_slot1_part_b * HEAD_DIM + raw_stream_35_fca_chunk_r4_slot1_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_35_fca_value_r4_slot1_part_b = pl.reg.load(fca_value_flat_slot1_part_b, raw_stream_35_fca_value_offset_r4_slot1_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_35_fca_scaled_r4_slot1_part_b = pl.reg.mul(raw_stream_35_fca_value_r4_slot1_part_b, raw_stream_35_fca_factor_r4_slot1_part_b, raw_stream_35_fca_all_r4_slot1_part_b)
                                    raw_stream_35_fca_target_row_r4_slot1_part_b = 0 + 16 + raw_stream_35_fca_row_r4_slot1_part_b
                                    raw_stream_35_fca_target_offset_r4_slot1_part_b = raw_stream_35_fca_target_row_r4_slot1_part_b * 16 + raw_stream_35_fca_chunk_r4_slot1_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_35_fca_write_r4_slot1_part_b = pl.reg.store(fca_kv_flat_slot1_part_b, raw_stream_35_fca_target_offset_r4_slot1_part_b, raw_stream_35_fca_scaled_r4_slot1_part_b, raw_stream_35_fca_all_r4_slot1_part_b, block_stride=32)
                    else:
                        raw_current_compressed2_flat_part_b: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[2], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                        raw_current_compressed2_prefix_part_b = pl.tile.slice(raw_current_compressed2_flat_part_b, [1, 10240], [0, 0])
                        raw_current_compressed2_part_b = pl.tile.reshape(raw_current_compressed2_prefix_part_b, [32, COMPRESSED_CACHE_BYTES])
                        fca_raw_chunk_slot2_part_b = pl.tile.slice(raw_current_compressed2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [0, 0])
                        fca_packed_slot2_part_b = pl.reinterpret_view(fca_raw_chunk_slot2_part_b, pl.FP4E2M1X2)
                        fca_codes_slot2_part_b = fca_packed_slot2_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot2_part_b = pl.cast(fca_codes_slot2_part_b, pl.BF16)
                        fca_scale_words_slot2_part_b = pl.reinterpret_view(fca_raw_chunk_slot2_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot2_part_b = pl.tile.reshape(fca_scale_words_slot2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot2_part_b = pl.tile.reshape(fca_values_slot2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot2_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_37_fca_all_r4_slot2_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_37_fca_row_r4_slot2_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_37_fca_chunk_r4_slot2_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_37_fca_scale_offset_r4_slot2_part_b = raw_stream_37_fca_row_r4_slot2_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_37_fca_scale_offset_r4_slot2_part_b = raw_stream_37_fca_scale_offset_r4_slot2_part_b + raw_stream_37_fca_chunk_r4_slot2_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_37_fca_groups_r4_slot2_part_b = pl.reg.load(fca_scale_flat_slot2_part_b, raw_stream_37_fca_scale_offset_r4_slot2_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_37_fca_factor_bits_r4_slot2_part_b = pl.reg.group_broadcast(raw_stream_37_fca_groups_r4_slot2_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_37_fca_factor_r4_slot2_part_b = pl.reg.interpret_cast(raw_stream_37_fca_factor_bits_r4_slot2_part_b, dtype=pl.BF16)
                                    raw_stream_37_fca_value_offset_r4_slot2_part_b = raw_stream_37_fca_row_r4_slot2_part_b * HEAD_DIM + raw_stream_37_fca_chunk_r4_slot2_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_37_fca_value_r4_slot2_part_b = pl.reg.load(fca_value_flat_slot2_part_b, raw_stream_37_fca_value_offset_r4_slot2_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_37_fca_scaled_r4_slot2_part_b = pl.reg.mul(raw_stream_37_fca_value_r4_slot2_part_b, raw_stream_37_fca_factor_r4_slot2_part_b, raw_stream_37_fca_all_r4_slot2_part_b)
                                    raw_stream_37_fca_target_row_r4_slot2_part_b = 0 + 0 + raw_stream_37_fca_row_r4_slot2_part_b
                                    raw_stream_37_fca_target_offset_r4_slot2_part_b = raw_stream_37_fca_target_row_r4_slot2_part_b * 16 + raw_stream_37_fca_chunk_r4_slot2_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_37_fca_write_r4_slot2_part_b = pl.reg.store(fca_kv_flat_slot2_part_b, raw_stream_37_fca_target_offset_r4_slot2_part_b, raw_stream_37_fca_scaled_r4_slot2_part_b, raw_stream_37_fca_all_r4_slot2_part_b, block_stride=32)
                        fca_raw_chunk_slot2_part_b = pl.tile.slice(raw_current_compressed2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE, COMPRESSED_CACHE_BYTES], [16, 0])
                        fca_packed_slot2_part_b = pl.reinterpret_view(fca_raw_chunk_slot2_part_b, pl.FP4E2M1X2)
                        fca_codes_slot2_part_b = fca_packed_slot2_part_b[:, :SPARSE_VF_HALF_DIM]
                        fca_values_slot2_part_b = pl.cast(fca_codes_slot2_part_b, pl.BF16)
                        fca_scale_words_slot2_part_b = pl.reinterpret_view(fca_raw_chunk_slot2_part_b, pl.UINT16, shape=[SPARSE_VF_FUSED_FP4_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_scale_flat_slot2_part_b = pl.tile.reshape(fca_scale_words_slot2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                        fca_value_flat_slot2_part_b = pl.tile.reshape(fca_values_slot2_part_b, [SPARSE_VF_FUSED_FP4_SCALE_TILE * HEAD_DIM])
                        fca_kv_flat_slot2_part_b = pl.tile.reshape(kv_chunk_part_b, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                        with pl.reg_scope():
                            raw_stream_39_fca_all_r4_slot2_part_b = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for raw_stream_39_fca_row_r4_slot2_part_b in pl.hardware_loop(SPARSE_VF_FUSED_FP4_SCALE_TILE):
                                for raw_stream_39_fca_chunk_r4_slot2_part_b in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    raw_stream_39_fca_scale_offset_r4_slot2_part_b = raw_stream_39_fca_row_r4_slot2_part_b * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                    raw_stream_39_fca_scale_offset_r4_slot2_part_b = raw_stream_39_fca_scale_offset_r4_slot2_part_b + raw_stream_39_fca_chunk_r4_slot2_part_b * SPARSE_VF_COMPRESSED_SCALE_TILE
                                    raw_stream_39_fca_groups_r4_slot2_part_b = pl.reg.load(fca_scale_flat_slot2_part_b, raw_stream_39_fca_scale_offset_r4_slot2_part_b, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                    raw_stream_39_fca_factor_bits_r4_slot2_part_b = pl.reg.group_broadcast(raw_stream_39_fca_groups_r4_slot2_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_39_fca_factor_r4_slot2_part_b = pl.reg.interpret_cast(raw_stream_39_fca_factor_bits_r4_slot2_part_b, dtype=pl.BF16)
                                    raw_stream_39_fca_value_offset_r4_slot2_part_b = raw_stream_39_fca_row_r4_slot2_part_b * HEAD_DIM + raw_stream_39_fca_chunk_r4_slot2_part_b * SPARSE_VF_REG_TILE
                                    raw_stream_39_fca_value_r4_slot2_part_b = pl.reg.load(fca_value_flat_slot2_part_b, raw_stream_39_fca_value_offset_r4_slot2_part_b, lanes=SPARSE_VF_REG_TILE)
                                    raw_stream_39_fca_scaled_r4_slot2_part_b = pl.reg.mul(raw_stream_39_fca_value_r4_slot2_part_b, raw_stream_39_fca_factor_r4_slot2_part_b, raw_stream_39_fca_all_r4_slot2_part_b)
                                    raw_stream_39_fca_target_row_r4_slot2_part_b = 0 + 16 + raw_stream_39_fca_row_r4_slot2_part_b
                                    raw_stream_39_fca_target_offset_r4_slot2_part_b = raw_stream_39_fca_target_row_r4_slot2_part_b * 16 + raw_stream_39_fca_chunk_r4_slot2_part_b * (32 * SPARSE_VF_REG_TILE)
                                    _raw_stream_39_fca_write_r4_slot2_part_b = pl.reg.store(fca_kv_flat_slot2_part_b, raw_stream_39_fca_target_offset_r4_slot2_part_b, raw_stream_39_fca_scaled_r4_slot2_part_b, raw_stream_39_fca_all_r4_slot2_part_b, block_stride=32)
                    kv_nz_part_b: pl.Tile[[32, HEAD_DIM], pl.BF16, pl.MemRef('sparse_raw_kv_nz'), pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512)] = pl.tile.create([32, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                    pl.system.tpush_to_aic(kv_nz_part_b, split=1, id=1)
                if part + 3 == parts and block + workers < num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE):
                    cross_next_token = (block + workers) // (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE)
                    cross_next_ids_seed0: pl.Tile[[1, 32], pl.INT32, pl.MemRef('sparse_raw_ids3', slots=3)[0], pl.Mem.Vec] = pl.tile.load(window_indices, [cross_next_token, 0 * 64 + aiv_id * 32], [1, 32])
                    cross_next_raw_seed0_storage: pl.Tile[[1, 17408], pl.UINT8, pl.MemRef('sparse_raw3', slots=3)[0], pl.Mem.Vec] = pl.tile.create([1, 17408], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
                    cross_next_raw_seed0 = pl.tile.reshape(cross_next_raw_seed0_storage, [32, WINDOW_CACHE_BYTES])
                    cross_next_seed0_first = pl.tile.read(cross_next_ids_seed0, [0, 0])
                    cross_next_seed0_contiguous = pl.cast(0, pl.INT32)
                    cross_next_seed0_all_valid = pl.cast(0, pl.INT32)
                    if cross_next_seed0_first >= 0:
                        cross_next_seed0_all_valid = pl.cast(1, pl.INT32)
                        cross_next_seed0_contiguous = pl.cast(1, pl.INT32)
                        for cross_next_seed0_check in pl.range(1, 32):
                            cross_next_seed0_index = pl.tile.read(cross_next_ids_seed0, [0, cross_next_seed0_check])
                            if cross_next_seed0_index != cross_next_seed0_first + cross_next_seed0_check:
                                cross_next_seed0_contiguous = pl.cast(0, pl.INT32)
                            if cross_next_seed0_index < 0:
                                cross_next_seed0_all_valid = pl.cast(0, pl.INT32)
                    if cross_next_seed0_contiguous != 0:
                        cross_next_seed0_physical = pl.cast(cross_next_seed0_first, pl.INDEX)
                        cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [0, 0], [cross_next_seed0_physical, 0], [32, WINDOW_CACHE_BYTES])
                    elif cross_next_seed0_all_valid != 0:
                        for cross_next_seed0_lane in pl.range(0, 32, 2):
                            cross_next_seed0_a = pl.tile.read(cross_next_ids_seed0, [0, cross_next_seed0_lane])
                            cross_next_seed0_b = pl.tile.read(cross_next_ids_seed0, [0, cross_next_seed0_lane + 1])
                            cross_next_seed0_lower = pl.min(cross_next_seed0_a, cross_next_seed0_b)
                            cross_next_seed0_upper = pl.max(cross_next_seed0_a, cross_next_seed0_b)
                            cross_next_seed0_pair = pl.cast(0, pl.INT32)
                            if cross_next_seed0_upper == cross_next_seed0_lower + 1:
                                cross_next_seed0_pair = pl.cast(1, pl.INT32)
                            if cross_next_seed0_pair != 0:
                                cross_next_seed0_physical_pair = pl.cast(cross_next_seed0_lower, pl.INDEX)
                                cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane, 0], [cross_next_seed0_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                cross_next_seed0_physical_a = pl.cast(cross_next_seed0_a, pl.INDEX)
                                cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane, 0], [cross_next_seed0_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                cross_next_seed0_physical_b = pl.cast(cross_next_seed0_b, pl.INDEX)
                                cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane + 1, 0], [cross_next_seed0_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                    else:
                        for cross_next_seed0_lane in pl.range(0, 32, 2):
                            cross_next_seed0_a = pl.tile.read(cross_next_ids_seed0, [0, cross_next_seed0_lane])
                            cross_next_seed0_b = pl.tile.read(cross_next_ids_seed0, [0, cross_next_seed0_lane + 1])
                            cross_next_seed0_lower = pl.min(cross_next_seed0_a, cross_next_seed0_b)
                            cross_next_seed0_upper = pl.max(cross_next_seed0_a, cross_next_seed0_b)
                            cross_next_seed0_pair = pl.cast(0, pl.INT32)
                            if cross_next_seed0_lower >= 0:
                                if cross_next_seed0_upper == cross_next_seed0_lower + 1:
                                    cross_next_seed0_pair = pl.cast(1, pl.INT32)
                            if cross_next_seed0_pair != 0:
                                cross_next_seed0_physical_pair = pl.cast(cross_next_seed0_lower, pl.INDEX)
                                cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane, 0], [cross_next_seed0_physical_pair, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                if cross_next_seed0_a >= 0:
                                    cross_next_seed0_physical_a = pl.cast(cross_next_seed0_a, pl.INDEX)
                                    cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane, 0], [cross_next_seed0_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                else:
                                    cross_next_raw_seed0 = pl.tile.assemble(cross_next_raw_seed0, zero_raw_window, [cross_next_seed0_lane, 0])
                                if cross_next_seed0_b >= 0:
                                    cross_next_seed0_physical_b = pl.cast(cross_next_seed0_b, pl.INDEX)
                                    cross_next_raw_seed0 = pl.tile.gather_row(cross_next_raw_seed0, window_flat, [cross_next_seed0_lane + 1, 0], [cross_next_seed0_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                                else:
                                    cross_next_raw_seed0 = pl.tile.assemble(cross_next_raw_seed0, zero_raw_window, [cross_next_seed0_lane + 1, 0])
                correction_col = pl.reshape(correction, [SPARSE_VF_FUSED_AIV_HEAD_TILE, 1])
                weighted_0 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_0 = pl.row_expand_mul(numerator_0, correction_col)
                numerator_0 = pl.add(scaled_0, weighted_0)
                pl.system.tfree_to_aic(weighted_0, id=3)
                weighted_1 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_1 = pl.row_expand_mul(numerator_1, correction_col)
                numerator_1 = pl.add(scaled_1, weighted_1)
                pl.system.tfree_to_aic(weighted_1, id=3)
                weighted_2 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_2 = pl.row_expand_mul(numerator_2, correction_col)
                numerator_2 = pl.add(scaled_2, weighted_2)
                pl.system.tfree_to_aic(weighted_2, id=3)
                weighted_3 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_3 = pl.row_expand_mul(numerator_3, correction_col)
                numerator_3 = pl.add(scaled_3, weighted_3)
                pl.system.tfree_to_aic(weighted_3, id=3)
        else:
            for prefetch_part in pl.range(pl.min(parts, 2)):
                column = prefetch_part * SPARSE_VF_KEY_TILE
                if compressed_keys > 0:
                    kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                    first_key_base = column + aiv_id * (SPARSE_VF_KEY_TILE // 4)
                    if column < WINDOW_KEYS:
                        ids_fwa = pl.tile.load(window_indices, [token, first_key_base + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        ids_fwb = pl.tile.load(window_indices, [token, first_key_base + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwa_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwa_first = pl.tile.read(ids_fwa, [0, 0])
                        fwa_contiguous = pl.cast(0, pl.INT32)
                        if fwa_first >= 0:
                            fwa_contiguous = pl.cast(1, pl.INT32)
                            for fwa_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwa_check_row = pl.tile.read(ids_fwa, [0, fwa_check_lane])
                                if fwa_check_row != fwa_first + fwa_check_lane:
                                    fwa_contiguous = pl.cast(0, pl.INT32)
                        if fwa_contiguous != 0:
                            fwa_start = pl.cast(fwa_first, pl.INDEX)
                            fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [0, 0], [fwa_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwa_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwa_a = pl.tile.read(ids_fwa, [0, fwa_lane])
                                fwa_b = pl.tile.read(ids_fwa, [0, fwa_lane + 1])
                                fwa_lower = pl.min(fwa_a, fwa_b)
                                fwa_upper = pl.max(fwa_a, fwa_b)
                                fwa_pair = pl.cast(0, pl.INT32)
                                if fwa_lower >= 0:
                                    if fwa_upper == fwa_lower + 1:
                                        fwa_pair = pl.cast(1, pl.INT32)
                                if fwa_pair != 0:
                                    fwa_physical = pl.cast(fwa_lower, pl.INDEX)
                                    fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwa_a >= 0:
                                        fwa_physical_a = pl.cast(fwa_a, pl.INDEX)
                                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwa_b >= 0:
                                        fwa_physical_b = pl.cast(fwa_b, pl.INDEX)
                                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane + 1, 0], [fwa_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        fwb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwb_first = pl.tile.read(ids_fwb, [0, 0])
                        fwb_contiguous = pl.cast(0, pl.INT32)
                        if fwb_first >= 0:
                            fwb_contiguous = pl.cast(1, pl.INT32)
                            for fwb_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwb_check_row = pl.tile.read(ids_fwb, [0, fwb_check_lane])
                                if fwb_check_row != fwb_first + fwb_check_lane:
                                    fwb_contiguous = pl.cast(0, pl.INT32)
                        if fwb_contiguous != 0:
                            fwb_start = pl.cast(fwb_first, pl.INDEX)
                            fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [0, 0], [fwb_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwb_a = pl.tile.read(ids_fwb, [0, fwb_lane])
                                fwb_b = pl.tile.read(ids_fwb, [0, fwb_lane + 1])
                                fwb_lower = pl.min(fwb_a, fwb_b)
                                fwb_upper = pl.max(fwb_a, fwb_b)
                                fwb_pair = pl.cast(0, pl.INT32)
                                if fwb_lower >= 0:
                                    if fwb_upper == fwb_lower + 1:
                                        fwb_pair = pl.cast(1, pl.INT32)
                                if fwb_pair != 0:
                                    fwb_physical = pl.cast(fwb_lower, pl.INDEX)
                                    fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwb_a >= 0:
                                        fwb_physical_a = pl.cast(fwb_a, pl.INDEX)
                                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwb_b >= 0:
                                        fwb_physical_b = pl.cast(fwb_b, pl.INDEX)
                                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane + 1, 0], [fwb_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwa_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwa_cast_row0 = fwa_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwa_raw_chunk = pl.tile.slice(fwa_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwa_cast_row0, 0])
                            fwa_signed = pl.reinterpret_view(fwa_raw_chunk, pl.INT8)
                            fwa_packed = pl.reinterpret_view(fwa_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwa_codes = fwa_packed[:, :HEAD_DIM]
                            fwa_values_fp32 = pl.cast(fwa_codes, pl.FP32)
                            fwa_values = pl.cast(fwa_values_fp32, pl.BF16, mode='rint')
                            fwa_scale_words = pl.reinterpret_view(fwa_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwa_scale_flat = pl.tile.reshape(fwa_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwa_value_flat = pl.tile.reshape(fwa_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwa_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fwa_all_r0 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwa_row_r0 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwa_chunk_r0 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwa_scale_offset_r0 = fwa_row_r0 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwa_scale_offset_r0 = fwa_scale_offset_r0 + fwa_chunk_r0 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwa_groups_r0 = pl.reg.load(fwa_scale_flat, fwa_scale_offset_r0, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwa_factor_bits_r0 = pl.reg.group_broadcast(fwa_groups_r0, lanes=SPARSE_VF_REG_TILE)
                                        fwa_factor_r0 = pl.reg.interpret_cast(fwa_factor_bits_r0, dtype=pl.BF16)
                                        fwa_value_offset_r0 = fwa_row_r0 * HEAD_DIM + fwa_chunk_r0 * SPARSE_VF_REG_TILE
                                        fwa_value_r0 = pl.reg.load(fwa_value_flat, fwa_value_offset_r0, lanes=SPARSE_VF_REG_TILE)
                                        fwa_scaled_r0 = pl.reg.mul(fwa_value_r0, fwa_factor_r0, fwa_all_r0)
                                        fwa_target_row_r0 = 0 + fwa_cast_row0 + fwa_row_r0
                                        fwa_target_offset_r0 = fwa_target_row_r0 * HEAD_DIM + fwa_chunk_r0 * SPARSE_VF_REG_TILE
                                        _fwa_write_r0 = pl.reg.store(fwa_kv_flat, fwa_target_offset_r0, fwa_scaled_r0, fwa_all_r0)
                        ids_fwc = pl.tile.load(window_indices, [token, first_key_base + 64], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwc_first = pl.tile.read(ids_fwc, [0, 0])
                        fwc_contiguous = pl.cast(0, pl.INT32)
                        if fwc_first >= 0:
                            fwc_contiguous = pl.cast(1, pl.INT32)
                            for fwc_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwc_check_row = pl.tile.read(ids_fwc, [0, fwc_check_lane])
                                if fwc_check_row != fwc_first + fwc_check_lane:
                                    fwc_contiguous = pl.cast(0, pl.INT32)
                        if fwc_contiguous != 0:
                            fwc_start = pl.cast(fwc_first, pl.INDEX)
                            fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [0, 0], [fwc_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwc_a = pl.tile.read(ids_fwc, [0, fwc_lane])
                                fwc_b = pl.tile.read(ids_fwc, [0, fwc_lane + 1])
                                fwc_lower = pl.min(fwc_a, fwc_b)
                                fwc_upper = pl.max(fwc_a, fwc_b)
                                fwc_pair = pl.cast(0, pl.INT32)
                                if fwc_lower >= 0:
                                    if fwc_upper == fwc_lower + 1:
                                        fwc_pair = pl.cast(1, pl.INT32)
                                if fwc_pair != 0:
                                    fwc_physical = pl.cast(fwc_lower, pl.INDEX)
                                    fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwc_a >= 0:
                                        fwc_physical_a = pl.cast(fwc_a, pl.INDEX)
                                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwc_b >= 0:
                                        fwc_physical_b = pl.cast(fwc_b, pl.INDEX)
                                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane + 1, 0], [fwc_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwb_cast_row0 = fwb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwb_raw_chunk = pl.tile.slice(fwb_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwb_cast_row0, 0])
                            fwb_signed = pl.reinterpret_view(fwb_raw_chunk, pl.INT8)
                            fwb_packed = pl.reinterpret_view(fwb_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwb_codes = fwb_packed[:, :HEAD_DIM]
                            fwb_values_fp32 = pl.cast(fwb_codes, pl.FP32)
                            fwb_values = pl.cast(fwb_values_fp32, pl.BF16, mode='rint')
                            fwb_scale_words = pl.reinterpret_view(fwb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwb_scale_flat = pl.tile.reshape(fwb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwb_value_flat = pl.tile.reshape(fwb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwb_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fwb_all_r1 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwb_row_r1 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwb_chunk_r1 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwb_scale_offset_r1 = fwb_row_r1 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwb_scale_offset_r1 = fwb_scale_offset_r1 + fwb_chunk_r1 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwb_groups_r1 = pl.reg.load(fwb_scale_flat, fwb_scale_offset_r1, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwb_factor_bits_r1 = pl.reg.group_broadcast(fwb_groups_r1, lanes=SPARSE_VF_REG_TILE)
                                        fwb_factor_r1 = pl.reg.interpret_cast(fwb_factor_bits_r1, dtype=pl.BF16)
                                        fwb_value_offset_r1 = fwb_row_r1 * HEAD_DIM + fwb_chunk_r1 * SPARSE_VF_REG_TILE
                                        fwb_value_r1 = pl.reg.load(fwb_value_flat, fwb_value_offset_r1, lanes=SPARSE_VF_REG_TILE)
                                        fwb_scaled_r1 = pl.reg.mul(fwb_value_r1, fwb_factor_r1, fwb_all_r1)
                                        fwb_target_row_r1 = 16 + fwb_cast_row0 + fwb_row_r1
                                        fwb_target_offset_r1 = fwb_target_row_r1 * HEAD_DIM + fwb_chunk_r1 * SPARSE_VF_REG_TILE
                                        _fwb_write_r1 = pl.reg.store(fwb_kv_flat, fwb_target_offset_r1, fwb_scaled_r1, fwb_all_r1)
                        pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                        kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                        ids_fwd = pl.tile.load(window_indices, [token, first_key_base + 80], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwd_first = pl.tile.read(ids_fwd, [0, 0])
                        fwd_contiguous = pl.cast(0, pl.INT32)
                        if fwd_first >= 0:
                            fwd_contiguous = pl.cast(1, pl.INT32)
                            for fwd_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwd_check_row = pl.tile.read(ids_fwd, [0, fwd_check_lane])
                                if fwd_check_row != fwd_first + fwd_check_lane:
                                    fwd_contiguous = pl.cast(0, pl.INT32)
                        if fwd_contiguous != 0:
                            fwd_start = pl.cast(fwd_first, pl.INDEX)
                            fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [0, 0], [fwd_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwd_a = pl.tile.read(ids_fwd, [0, fwd_lane])
                                fwd_b = pl.tile.read(ids_fwd, [0, fwd_lane + 1])
                                fwd_lower = pl.min(fwd_a, fwd_b)
                                fwd_upper = pl.max(fwd_a, fwd_b)
                                fwd_pair = pl.cast(0, pl.INT32)
                                if fwd_lower >= 0:
                                    if fwd_upper == fwd_lower + 1:
                                        fwd_pair = pl.cast(1, pl.INT32)
                                if fwd_pair != 0:
                                    fwd_physical = pl.cast(fwd_lower, pl.INDEX)
                                    fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwd_a >= 0:
                                        fwd_physical_a = pl.cast(fwd_a, pl.INDEX)
                                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwd_b >= 0:
                                        fwd_physical_b = pl.cast(fwd_b, pl.INDEX)
                                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane + 1, 0], [fwd_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwc_cast_row0 = fwc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwc_raw_chunk = pl.tile.slice(fwc_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwc_cast_row0, 0])
                            fwc_signed = pl.reinterpret_view(fwc_raw_chunk, pl.INT8)
                            fwc_packed = pl.reinterpret_view(fwc_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwc_codes = fwc_packed[:, :HEAD_DIM]
                            fwc_values_fp32 = pl.cast(fwc_codes, pl.FP32)
                            fwc_values = pl.cast(fwc_values_fp32, pl.BF16, mode='rint')
                            fwc_scale_words = pl.reinterpret_view(fwc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwc_scale_flat = pl.tile.reshape(fwc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwc_value_flat = pl.tile.reshape(fwc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwc_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fwc_all_r2 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwc_row_r2 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwc_chunk_r2 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwc_scale_offset_r2 = fwc_row_r2 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwc_scale_offset_r2 = fwc_scale_offset_r2 + fwc_chunk_r2 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwc_groups_r2 = pl.reg.load(fwc_scale_flat, fwc_scale_offset_r2, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwc_factor_bits_r2 = pl.reg.group_broadcast(fwc_groups_r2, lanes=SPARSE_VF_REG_TILE)
                                        fwc_factor_r2 = pl.reg.interpret_cast(fwc_factor_bits_r2, dtype=pl.BF16)
                                        fwc_value_offset_r2 = fwc_row_r2 * HEAD_DIM + fwc_chunk_r2 * SPARSE_VF_REG_TILE
                                        fwc_value_r2 = pl.reg.load(fwc_value_flat, fwc_value_offset_r2, lanes=SPARSE_VF_REG_TILE)
                                        fwc_scaled_r2 = pl.reg.mul(fwc_value_r2, fwc_factor_r2, fwc_all_r2)
                                        fwc_target_row_r2 = 0 + fwc_cast_row0 + fwc_row_r2
                                        fwc_target_offset_r2 = fwc_target_row_r2 * HEAD_DIM + fwc_chunk_r2 * SPARSE_VF_REG_TILE
                                        _fwc_write_r2 = pl.reg.store(fwc_kv_flat, fwc_target_offset_r2, fwc_scaled_r2, fwc_all_r2)
                        for fwd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwd_cast_row0 = fwd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwd_raw_chunk = pl.tile.slice(fwd_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwd_cast_row0, 0])
                            fwd_signed = pl.reinterpret_view(fwd_raw_chunk, pl.INT8)
                            fwd_packed = pl.reinterpret_view(fwd_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwd_codes = fwd_packed[:, :HEAD_DIM]
                            fwd_values_fp32 = pl.cast(fwd_codes, pl.FP32)
                            fwd_values = pl.cast(fwd_values_fp32, pl.BF16, mode='rint')
                            fwd_scale_words = pl.reinterpret_view(fwd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwd_scale_flat = pl.tile.reshape(fwd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwd_value_flat = pl.tile.reshape(fwd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwd_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fwd_all_r3 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwd_row_r3 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwd_chunk_r3 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwd_scale_offset_r3 = fwd_row_r3 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwd_scale_offset_r3 = fwd_scale_offset_r3 + fwd_chunk_r3 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwd_groups_r3 = pl.reg.load(fwd_scale_flat, fwd_scale_offset_r3, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwd_factor_bits_r3 = pl.reg.group_broadcast(fwd_groups_r3, lanes=SPARSE_VF_REG_TILE)
                                        fwd_factor_r3 = pl.reg.interpret_cast(fwd_factor_bits_r3, dtype=pl.BF16)
                                        fwd_value_offset_r3 = fwd_row_r3 * HEAD_DIM + fwd_chunk_r3 * SPARSE_VF_REG_TILE
                                        fwd_value_r3 = pl.reg.load(fwd_value_flat, fwd_value_offset_r3, lanes=SPARSE_VF_REG_TILE)
                                        fwd_scaled_r3 = pl.reg.mul(fwd_value_r3, fwd_factor_r3, fwd_all_r3)
                                        fwd_target_row_r3 = 16 + fwd_cast_row0 + fwd_row_r3
                                        fwd_target_offset_r3 = fwd_target_row_r3 * HEAD_DIM + fwd_chunk_r3 * SPARSE_VF_REG_TILE
                                        _fwd_write_r3 = pl.reg.store(fwd_kv_flat, fwd_target_offset_r3, fwd_scaled_r3, fwd_all_r3)
                        pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                    else:
                        ids_fca = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        ids_fcb = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fca_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        for fca_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fca_a = pl.tile.read(ids_fca, [0, fca_lane])
                            fca_b = pl.tile.read(ids_fca, [0, fca_lane + 1])
                            fca_lower = pl.min(fca_a, fca_b)
                            fca_upper = pl.max(fca_a, fca_b)
                            fca_pair = pl.cast(0, pl.INT32)
                            if fca_lower >= 0:
                                if fca_upper == fca_lower + 1:
                                    fca_pair = pl.cast(1, pl.INT32)
                            if fca_pair != 0:
                                fca_physical = pl.cast(fca_lower, pl.INDEX)
                                fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane, 0], [fca_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                            else:
                                if fca_a >= 0:
                                    fca_physical_a = pl.cast(fca_a, pl.INDEX)
                                    fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane, 0], [fca_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                if fca_b >= 0:
                                    fca_physical_b = pl.cast(fca_b, pl.INDEX)
                                    fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane + 1, 0], [fca_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                        fcb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        for fcb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fcb_a = pl.tile.read(ids_fcb, [0, fcb_lane])
                            fcb_b = pl.tile.read(ids_fcb, [0, fcb_lane + 1])
                            fcb_lower = pl.min(fcb_a, fcb_b)
                            fcb_upper = pl.max(fcb_a, fcb_b)
                            fcb_pair = pl.cast(0, pl.INT32)
                            if fcb_lower >= 0:
                                if fcb_upper == fcb_lower + 1:
                                    fcb_pair = pl.cast(1, pl.INT32)
                            if fcb_pair != 0:
                                fcb_physical = pl.cast(fcb_lower, pl.INDEX)
                                fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane, 0], [fcb_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                            else:
                                if fcb_a >= 0:
                                    fcb_physical_a = pl.cast(fcb_a, pl.INDEX)
                                    fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane, 0], [fcb_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                if fcb_b >= 0:
                                    fcb_physical_b = pl.cast(fcb_b, pl.INDEX)
                                    fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane + 1, 0], [fcb_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                        for fca_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fca_cast_row0 = fca_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fca_raw_chunk = pl.tile.slice(fca_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fca_cast_row0, 0])
                            fca_packed = pl.reinterpret_view(fca_raw_chunk, pl.FP4E2M1X2)
                            fca_codes = fca_packed[:, :SPARSE_VF_HALF_DIM]
                            fca_values = pl.cast(fca_codes, pl.BF16)
                            fca_scale_words = pl.reinterpret_view(fca_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                            fca_scale_flat = pl.tile.reshape(fca_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                            fca_value_flat = pl.tile.reshape(fca_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fca_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fca_all_r4 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fca_row_r4 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fca_chunk_r4 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fca_scale_offset_r4 = fca_row_r4 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                        fca_scale_offset_r4 = fca_scale_offset_r4 + fca_chunk_r4 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                        fca_groups_r4 = pl.reg.load(fca_scale_flat, fca_scale_offset_r4, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                        fca_factor_bits_r4 = pl.reg.group_broadcast(fca_groups_r4, lanes=SPARSE_VF_REG_TILE)
                                        fca_factor_r4 = pl.reg.interpret_cast(fca_factor_bits_r4, dtype=pl.BF16)
                                        fca_value_offset_r4 = fca_row_r4 * HEAD_DIM + fca_chunk_r4 * SPARSE_VF_REG_TILE
                                        fca_value_r4 = pl.reg.load(fca_value_flat, fca_value_offset_r4, lanes=SPARSE_VF_REG_TILE)
                                        fca_scaled_r4 = pl.reg.mul(fca_value_r4, fca_factor_r4, fca_all_r4)
                                        fca_target_row_r4 = 0 + fca_cast_row0 + fca_row_r4
                                        fca_target_offset_r4 = fca_target_row_r4 * HEAD_DIM + fca_chunk_r4 * SPARSE_VF_REG_TILE
                                        _fca_write_r4 = pl.reg.store(fca_kv_flat, fca_target_offset_r4, fca_scaled_r4, fca_all_r4)
                        ids_fcc = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 64], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fcc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        for fcc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fcc_a = pl.tile.read(ids_fcc, [0, fcc_lane])
                            fcc_b = pl.tile.read(ids_fcc, [0, fcc_lane + 1])
                            fcc_lower = pl.min(fcc_a, fcc_b)
                            fcc_upper = pl.max(fcc_a, fcc_b)
                            fcc_pair = pl.cast(0, pl.INT32)
                            if fcc_lower >= 0:
                                if fcc_upper == fcc_lower + 1:
                                    fcc_pair = pl.cast(1, pl.INT32)
                            if fcc_pair != 0:
                                fcc_physical = pl.cast(fcc_lower, pl.INDEX)
                                fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane, 0], [fcc_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                            else:
                                if fcc_a >= 0:
                                    fcc_physical_a = pl.cast(fcc_a, pl.INDEX)
                                    fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane, 0], [fcc_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                if fcc_b >= 0:
                                    fcc_physical_b = pl.cast(fcc_b, pl.INDEX)
                                    fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane + 1, 0], [fcc_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                        for fcb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fcb_cast_row0 = fcb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fcb_raw_chunk = pl.tile.slice(fcb_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcb_cast_row0, 0])
                            fcb_packed = pl.reinterpret_view(fcb_raw_chunk, pl.FP4E2M1X2)
                            fcb_codes = fcb_packed[:, :SPARSE_VF_HALF_DIM]
                            fcb_values = pl.cast(fcb_codes, pl.BF16)
                            fcb_scale_words = pl.reinterpret_view(fcb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcb_scale_flat = pl.tile.reshape(fcb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcb_value_flat = pl.tile.reshape(fcb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fcb_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fcb_all_r5 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fcb_row_r5 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fcb_chunk_r5 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fcb_scale_offset_r5 = fcb_row_r5 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                        fcb_scale_offset_r5 = fcb_scale_offset_r5 + fcb_chunk_r5 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                        fcb_groups_r5 = pl.reg.load(fcb_scale_flat, fcb_scale_offset_r5, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                        fcb_factor_bits_r5 = pl.reg.group_broadcast(fcb_groups_r5, lanes=SPARSE_VF_REG_TILE)
                                        fcb_factor_r5 = pl.reg.interpret_cast(fcb_factor_bits_r5, dtype=pl.BF16)
                                        fcb_value_offset_r5 = fcb_row_r5 * HEAD_DIM + fcb_chunk_r5 * SPARSE_VF_REG_TILE
                                        fcb_value_r5 = pl.reg.load(fcb_value_flat, fcb_value_offset_r5, lanes=SPARSE_VF_REG_TILE)
                                        fcb_scaled_r5 = pl.reg.mul(fcb_value_r5, fcb_factor_r5, fcb_all_r5)
                                        fcb_target_row_r5 = 16 + fcb_cast_row0 + fcb_row_r5
                                        fcb_target_offset_r5 = fcb_target_row_r5 * HEAD_DIM + fcb_chunk_r5 * SPARSE_VF_REG_TILE
                                        _fcb_write_r5 = pl.reg.store(fcb_kv_flat, fcb_target_offset_r5, fcb_scaled_r5, fcb_all_r5)
                        pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                        kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                        ids_fcd = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 80], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fcd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        for fcd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fcd_a = pl.tile.read(ids_fcd, [0, fcd_lane])
                            fcd_b = pl.tile.read(ids_fcd, [0, fcd_lane + 1])
                            fcd_lower = pl.min(fcd_a, fcd_b)
                            fcd_upper = pl.max(fcd_a, fcd_b)
                            fcd_pair = pl.cast(0, pl.INT32)
                            if fcd_lower >= 0:
                                if fcd_upper == fcd_lower + 1:
                                    fcd_pair = pl.cast(1, pl.INT32)
                            if fcd_pair != 0:
                                fcd_physical = pl.cast(fcd_lower, pl.INDEX)
                                fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane, 0], [fcd_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                            else:
                                if fcd_a >= 0:
                                    fcd_physical_a = pl.cast(fcd_a, pl.INDEX)
                                    fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane, 0], [fcd_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                if fcd_b >= 0:
                                    fcd_physical_b = pl.cast(fcd_b, pl.INDEX)
                                    fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane + 1, 0], [fcd_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                        for fcc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fcc_cast_row0 = fcc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fcc_raw_chunk = pl.tile.slice(fcc_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcc_cast_row0, 0])
                            fcc_packed = pl.reinterpret_view(fcc_raw_chunk, pl.FP4E2M1X2)
                            fcc_codes = fcc_packed[:, :SPARSE_VF_HALF_DIM]
                            fcc_values = pl.cast(fcc_codes, pl.BF16)
                            fcc_scale_words = pl.reinterpret_view(fcc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcc_scale_flat = pl.tile.reshape(fcc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcc_value_flat = pl.tile.reshape(fcc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fcc_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fcc_all_r6 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fcc_row_r6 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fcc_chunk_r6 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fcc_scale_offset_r6 = fcc_row_r6 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                        fcc_scale_offset_r6 = fcc_scale_offset_r6 + fcc_chunk_r6 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                        fcc_groups_r6 = pl.reg.load(fcc_scale_flat, fcc_scale_offset_r6, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                        fcc_factor_bits_r6 = pl.reg.group_broadcast(fcc_groups_r6, lanes=SPARSE_VF_REG_TILE)
                                        fcc_factor_r6 = pl.reg.interpret_cast(fcc_factor_bits_r6, dtype=pl.BF16)
                                        fcc_value_offset_r6 = fcc_row_r6 * HEAD_DIM + fcc_chunk_r6 * SPARSE_VF_REG_TILE
                                        fcc_value_r6 = pl.reg.load(fcc_value_flat, fcc_value_offset_r6, lanes=SPARSE_VF_REG_TILE)
                                        fcc_scaled_r6 = pl.reg.mul(fcc_value_r6, fcc_factor_r6, fcc_all_r6)
                                        fcc_target_row_r6 = 0 + fcc_cast_row0 + fcc_row_r6
                                        fcc_target_offset_r6 = fcc_target_row_r6 * HEAD_DIM + fcc_chunk_r6 * SPARSE_VF_REG_TILE
                                        _fcc_write_r6 = pl.reg.store(fcc_kv_flat, fcc_target_offset_r6, fcc_scaled_r6, fcc_all_r6)
                        for fcd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fcd_cast_row0 = fcd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fcd_raw_chunk = pl.tile.slice(fcd_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcd_cast_row0, 0])
                            fcd_packed = pl.reinterpret_view(fcd_raw_chunk, pl.FP4E2M1X2)
                            fcd_codes = fcd_packed[:, :SPARSE_VF_HALF_DIM]
                            fcd_values = pl.cast(fcd_codes, pl.BF16)
                            fcd_scale_words = pl.reinterpret_view(fcd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcd_scale_flat = pl.tile.reshape(fcd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                            fcd_value_flat = pl.tile.reshape(fcd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fcd_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                            with pl.reg_scope():
                                fcd_all_r7 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fcd_row_r7 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fcd_chunk_r7 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fcd_scale_offset_r7 = fcd_row_r7 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                        fcd_scale_offset_r7 = fcd_scale_offset_r7 + fcd_chunk_r7 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                        fcd_groups_r7 = pl.reg.load(fcd_scale_flat, fcd_scale_offset_r7, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                        fcd_factor_bits_r7 = pl.reg.group_broadcast(fcd_groups_r7, lanes=SPARSE_VF_REG_TILE)
                                        fcd_factor_r7 = pl.reg.interpret_cast(fcd_factor_bits_r7, dtype=pl.BF16)
                                        fcd_value_offset_r7 = fcd_row_r7 * HEAD_DIM + fcd_chunk_r7 * SPARSE_VF_REG_TILE
                                        fcd_value_r7 = pl.reg.load(fcd_value_flat, fcd_value_offset_r7, lanes=SPARSE_VF_REG_TILE)
                                        fcd_scaled_r7 = pl.reg.mul(fcd_value_r7, fcd_factor_r7, fcd_all_r7)
                                        fcd_target_row_r7 = 16 + fcd_cast_row0 + fcd_row_r7
                                        fcd_target_offset_r7 = fcd_target_row_r7 * HEAD_DIM + fcd_chunk_r7 * SPARSE_VF_REG_TILE
                                        _fcd_write_r7 = pl.reg.store(fcd_kv_flat, fcd_target_offset_r7, fcd_scaled_r7, fcd_all_r7)
                        pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                else:
                    kv_half = pl.tile.create([SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                    first_key_base = column + aiv_id * (SPARSE_VF_KEY_TILE // 2)
                    ids_fwa = pl.tile.load(window_indices, [token, first_key_base + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                    ids_fwb = pl.tile.load(window_indices, [token, first_key_base + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                    fwa_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                    fwa_first = pl.tile.read(ids_fwa, [0, 0])
                    fwa_contiguous = pl.cast(0, pl.INT32)
                    if fwa_first >= 0:
                        fwa_contiguous = pl.cast(1, pl.INT32)
                        for fwa_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                            fwa_check_row = pl.tile.read(ids_fwa, [0, fwa_check_lane])
                            if fwa_check_row != fwa_first + fwa_check_lane:
                                fwa_contiguous = pl.cast(0, pl.INT32)
                    if fwa_contiguous != 0:
                        fwa_start = pl.cast(fwa_first, pl.INDEX)
                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [0, 0], [fwa_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                    else:
                        for fwa_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fwa_a = pl.tile.read(ids_fwa, [0, fwa_lane])
                            fwa_b = pl.tile.read(ids_fwa, [0, fwa_lane + 1])
                            fwa_lower = pl.min(fwa_a, fwa_b)
                            fwa_upper = pl.max(fwa_a, fwa_b)
                            fwa_pair = pl.cast(0, pl.INT32)
                            if fwa_lower >= 0:
                                if fwa_upper == fwa_lower + 1:
                                    fwa_pair = pl.cast(1, pl.INT32)
                            if fwa_pair != 0:
                                fwa_physical = pl.cast(fwa_lower, pl.INDEX)
                                fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                if fwa_a >= 0:
                                    fwa_physical_a = pl.cast(fwa_a, pl.INDEX)
                                    fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                if fwa_b >= 0:
                                    fwa_physical_b = pl.cast(fwa_b, pl.INDEX)
                                    fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane + 1, 0], [fwa_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                    fwb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                    fwb_first = pl.tile.read(ids_fwb, [0, 0])
                    fwb_contiguous = pl.cast(0, pl.INT32)
                    if fwb_first >= 0:
                        fwb_contiguous = pl.cast(1, pl.INT32)
                        for fwb_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                            fwb_check_row = pl.tile.read(ids_fwb, [0, fwb_check_lane])
                            if fwb_check_row != fwb_first + fwb_check_lane:
                                fwb_contiguous = pl.cast(0, pl.INT32)
                    if fwb_contiguous != 0:
                        fwb_start = pl.cast(fwb_first, pl.INDEX)
                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [0, 0], [fwb_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                    else:
                        for fwb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fwb_a = pl.tile.read(ids_fwb, [0, fwb_lane])
                            fwb_b = pl.tile.read(ids_fwb, [0, fwb_lane + 1])
                            fwb_lower = pl.min(fwb_a, fwb_b)
                            fwb_upper = pl.max(fwb_a, fwb_b)
                            fwb_pair = pl.cast(0, pl.INT32)
                            if fwb_lower >= 0:
                                if fwb_upper == fwb_lower + 1:
                                    fwb_pair = pl.cast(1, pl.INT32)
                            if fwb_pair != 0:
                                fwb_physical = pl.cast(fwb_lower, pl.INDEX)
                                fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                if fwb_a >= 0:
                                    fwb_physical_a = pl.cast(fwb_a, pl.INDEX)
                                    fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                if fwb_b >= 0:
                                    fwb_physical_b = pl.cast(fwb_b, pl.INDEX)
                                    fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane + 1, 0], [fwb_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                    for fwa_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                        fwa_cast_row0 = fwa_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                        fwa_raw_chunk = pl.tile.slice(fwa_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwa_cast_row0, 0])
                        fwa_signed = pl.reinterpret_view(fwa_raw_chunk, pl.INT8)
                        fwa_packed = pl.reinterpret_view(fwa_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                        fwa_codes = fwa_packed[:, :HEAD_DIM]
                        fwa_values_fp32 = pl.cast(fwa_codes, pl.FP32)
                        fwa_values = pl.cast(fwa_values_fp32, pl.BF16, mode='rint')
                        fwa_scale_words = pl.reinterpret_view(fwa_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                        fwa_scale_flat = pl.tile.reshape(fwa_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                        fwa_value_flat = pl.tile.reshape(fwa_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                        fwa_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                        with pl.reg_scope():
                            fwa_all_r8 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for fwa_row_r8 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                for fwa_chunk_r8 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    fwa_scale_offset_r8 = fwa_row_r8 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                    fwa_scale_offset_r8 = fwa_scale_offset_r8 + fwa_chunk_r8 * SPARSE_VF_WINDOW_SCALE_TILE
                                    fwa_groups_r8 = pl.reg.load(fwa_scale_flat, fwa_scale_offset_r8, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                    fwa_factor_bits_r8 = pl.reg.group_broadcast(fwa_groups_r8, lanes=SPARSE_VF_REG_TILE)
                                    fwa_factor_r8 = pl.reg.interpret_cast(fwa_factor_bits_r8, dtype=pl.BF16)
                                    fwa_value_offset_r8 = fwa_row_r8 * HEAD_DIM + fwa_chunk_r8 * SPARSE_VF_REG_TILE
                                    fwa_value_r8 = pl.reg.load(fwa_value_flat, fwa_value_offset_r8, lanes=SPARSE_VF_REG_TILE)
                                    fwa_scaled_r8 = pl.reg.mul(fwa_value_r8, fwa_factor_r8, fwa_all_r8)
                                    fwa_target_row_r8 = 0 + fwa_cast_row0 + fwa_row_r8
                                    fwa_target_offset_r8 = fwa_target_row_r8 * HEAD_DIM + fwa_chunk_r8 * SPARSE_VF_REG_TILE
                                    _fwa_write_r8 = pl.reg.store(fwa_kv_flat, fwa_target_offset_r8, fwa_scaled_r8, fwa_all_r8)
                    ids_fwc = pl.tile.load(window_indices, [token, first_key_base + 32], [1, SPARSE_VF_FUSED_GATHER_TILE])
                    fwc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                    fwc_first = pl.tile.read(ids_fwc, [0, 0])
                    fwc_contiguous = pl.cast(0, pl.INT32)
                    if fwc_first >= 0:
                        fwc_contiguous = pl.cast(1, pl.INT32)
                        for fwc_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                            fwc_check_row = pl.tile.read(ids_fwc, [0, fwc_check_lane])
                            if fwc_check_row != fwc_first + fwc_check_lane:
                                fwc_contiguous = pl.cast(0, pl.INT32)
                    if fwc_contiguous != 0:
                        fwc_start = pl.cast(fwc_first, pl.INDEX)
                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [0, 0], [fwc_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                    else:
                        for fwc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fwc_a = pl.tile.read(ids_fwc, [0, fwc_lane])
                            fwc_b = pl.tile.read(ids_fwc, [0, fwc_lane + 1])
                            fwc_lower = pl.min(fwc_a, fwc_b)
                            fwc_upper = pl.max(fwc_a, fwc_b)
                            fwc_pair = pl.cast(0, pl.INT32)
                            if fwc_lower >= 0:
                                if fwc_upper == fwc_lower + 1:
                                    fwc_pair = pl.cast(1, pl.INT32)
                            if fwc_pair != 0:
                                fwc_physical = pl.cast(fwc_lower, pl.INDEX)
                                fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                if fwc_a >= 0:
                                    fwc_physical_a = pl.cast(fwc_a, pl.INDEX)
                                    fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                if fwc_b >= 0:
                                    fwc_physical_b = pl.cast(fwc_b, pl.INDEX)
                                    fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane + 1, 0], [fwc_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                    for fwb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                        fwb_cast_row0 = fwb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                        fwb_raw_chunk = pl.tile.slice(fwb_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwb_cast_row0, 0])
                        fwb_signed = pl.reinterpret_view(fwb_raw_chunk, pl.INT8)
                        fwb_packed = pl.reinterpret_view(fwb_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                        fwb_codes = fwb_packed[:, :HEAD_DIM]
                        fwb_values_fp32 = pl.cast(fwb_codes, pl.FP32)
                        fwb_values = pl.cast(fwb_values_fp32, pl.BF16, mode='rint')
                        fwb_scale_words = pl.reinterpret_view(fwb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                        fwb_scale_flat = pl.tile.reshape(fwb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                        fwb_value_flat = pl.tile.reshape(fwb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                        fwb_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                        with pl.reg_scope():
                            fwb_all_r9 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for fwb_row_r9 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                for fwb_chunk_r9 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    fwb_scale_offset_r9 = fwb_row_r9 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                    fwb_scale_offset_r9 = fwb_scale_offset_r9 + fwb_chunk_r9 * SPARSE_VF_WINDOW_SCALE_TILE
                                    fwb_groups_r9 = pl.reg.load(fwb_scale_flat, fwb_scale_offset_r9, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                    fwb_factor_bits_r9 = pl.reg.group_broadcast(fwb_groups_r9, lanes=SPARSE_VF_REG_TILE)
                                    fwb_factor_r9 = pl.reg.interpret_cast(fwb_factor_bits_r9, dtype=pl.BF16)
                                    fwb_value_offset_r9 = fwb_row_r9 * HEAD_DIM + fwb_chunk_r9 * SPARSE_VF_REG_TILE
                                    fwb_value_r9 = pl.reg.load(fwb_value_flat, fwb_value_offset_r9, lanes=SPARSE_VF_REG_TILE)
                                    fwb_scaled_r9 = pl.reg.mul(fwb_value_r9, fwb_factor_r9, fwb_all_r9)
                                    fwb_target_row_r9 = 16 + fwb_cast_row0 + fwb_row_r9
                                    fwb_target_offset_r9 = fwb_target_row_r9 * HEAD_DIM + fwb_chunk_r9 * SPARSE_VF_REG_TILE
                                    _fwb_write_r9 = pl.reg.store(fwb_kv_flat, fwb_target_offset_r9, fwb_scaled_r9, fwb_all_r9)
                    ids_fwd = pl.tile.load(window_indices, [token, first_key_base + 48], [1, SPARSE_VF_FUSED_GATHER_TILE])
                    fwd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                    fwd_first = pl.tile.read(ids_fwd, [0, 0])
                    fwd_contiguous = pl.cast(0, pl.INT32)
                    if fwd_first >= 0:
                        fwd_contiguous = pl.cast(1, pl.INT32)
                        for fwd_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                            fwd_check_row = pl.tile.read(ids_fwd, [0, fwd_check_lane])
                            if fwd_check_row != fwd_first + fwd_check_lane:
                                fwd_contiguous = pl.cast(0, pl.INT32)
                    if fwd_contiguous != 0:
                        fwd_start = pl.cast(fwd_first, pl.INDEX)
                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [0, 0], [fwd_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                    else:
                        for fwd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                            fwd_a = pl.tile.read(ids_fwd, [0, fwd_lane])
                            fwd_b = pl.tile.read(ids_fwd, [0, fwd_lane + 1])
                            fwd_lower = pl.min(fwd_a, fwd_b)
                            fwd_upper = pl.max(fwd_a, fwd_b)
                            fwd_pair = pl.cast(0, pl.INT32)
                            if fwd_lower >= 0:
                                if fwd_upper == fwd_lower + 1:
                                    fwd_pair = pl.cast(1, pl.INT32)
                            if fwd_pair != 0:
                                fwd_physical = pl.cast(fwd_lower, pl.INDEX)
                                fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical, 0], [2, WINDOW_CACHE_BYTES])
                            else:
                                if fwd_a >= 0:
                                    fwd_physical_a = pl.cast(fwd_a, pl.INDEX)
                                    fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                if fwd_b >= 0:
                                    fwd_physical_b = pl.cast(fwd_b, pl.INDEX)
                                    fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane + 1, 0], [fwd_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                    for fwc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                        fwc_cast_row0 = fwc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                        fwc_raw_chunk = pl.tile.slice(fwc_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwc_cast_row0, 0])
                        fwc_signed = pl.reinterpret_view(fwc_raw_chunk, pl.INT8)
                        fwc_packed = pl.reinterpret_view(fwc_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                        fwc_codes = fwc_packed[:, :HEAD_DIM]
                        fwc_values_fp32 = pl.cast(fwc_codes, pl.FP32)
                        fwc_values = pl.cast(fwc_values_fp32, pl.BF16, mode='rint')
                        fwc_scale_words = pl.reinterpret_view(fwc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                        fwc_scale_flat = pl.tile.reshape(fwc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                        fwc_value_flat = pl.tile.reshape(fwc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                        fwc_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                        with pl.reg_scope():
                            fwc_all_r10 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for fwc_row_r10 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                for fwc_chunk_r10 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    fwc_scale_offset_r10 = fwc_row_r10 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                    fwc_scale_offset_r10 = fwc_scale_offset_r10 + fwc_chunk_r10 * SPARSE_VF_WINDOW_SCALE_TILE
                                    fwc_groups_r10 = pl.reg.load(fwc_scale_flat, fwc_scale_offset_r10, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                    fwc_factor_bits_r10 = pl.reg.group_broadcast(fwc_groups_r10, lanes=SPARSE_VF_REG_TILE)
                                    fwc_factor_r10 = pl.reg.interpret_cast(fwc_factor_bits_r10, dtype=pl.BF16)
                                    fwc_value_offset_r10 = fwc_row_r10 * HEAD_DIM + fwc_chunk_r10 * SPARSE_VF_REG_TILE
                                    fwc_value_r10 = pl.reg.load(fwc_value_flat, fwc_value_offset_r10, lanes=SPARSE_VF_REG_TILE)
                                    fwc_scaled_r10 = pl.reg.mul(fwc_value_r10, fwc_factor_r10, fwc_all_r10)
                                    fwc_target_row_r10 = 32 + fwc_cast_row0 + fwc_row_r10
                                    fwc_target_offset_r10 = fwc_target_row_r10 * HEAD_DIM + fwc_chunk_r10 * SPARSE_VF_REG_TILE
                                    _fwc_write_r10 = pl.reg.store(fwc_kv_flat, fwc_target_offset_r10, fwc_scaled_r10, fwc_all_r10)
                    for fwd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                        fwd_cast_row0 = fwd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                        fwd_raw_chunk = pl.tile.slice(fwd_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwd_cast_row0, 0])
                        fwd_signed = pl.reinterpret_view(fwd_raw_chunk, pl.INT8)
                        fwd_packed = pl.reinterpret_view(fwd_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                        fwd_codes = fwd_packed[:, :HEAD_DIM]
                        fwd_values_fp32 = pl.cast(fwd_codes, pl.FP32)
                        fwd_values = pl.cast(fwd_values_fp32, pl.BF16, mode='rint')
                        fwd_scale_words = pl.reinterpret_view(fwd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                        fwd_scale_flat = pl.tile.reshape(fwd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                        fwd_value_flat = pl.tile.reshape(fwd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                        fwd_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                        with pl.reg_scope():
                            fwd_all_r11 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                            for fwd_row_r11 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                for fwd_chunk_r11 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                    fwd_scale_offset_r11 = fwd_row_r11 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                    fwd_scale_offset_r11 = fwd_scale_offset_r11 + fwd_chunk_r11 * SPARSE_VF_WINDOW_SCALE_TILE
                                    fwd_groups_r11 = pl.reg.load(fwd_scale_flat, fwd_scale_offset_r11, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                    fwd_factor_bits_r11 = pl.reg.group_broadcast(fwd_groups_r11, lanes=SPARSE_VF_REG_TILE)
                                    fwd_factor_r11 = pl.reg.interpret_cast(fwd_factor_bits_r11, dtype=pl.BF16)
                                    fwd_value_offset_r11 = fwd_row_r11 * HEAD_DIM + fwd_chunk_r11 * SPARSE_VF_REG_TILE
                                    fwd_value_r11 = pl.reg.load(fwd_value_flat, fwd_value_offset_r11, lanes=SPARSE_VF_REG_TILE)
                                    fwd_scaled_r11 = pl.reg.mul(fwd_value_r11, fwd_factor_r11, fwd_all_r11)
                                    fwd_target_row_r11 = 48 + fwd_cast_row0 + fwd_row_r11
                                    fwd_target_offset_r11 = fwd_target_row_r11 * HEAD_DIM + fwd_chunk_r11 * SPARSE_VF_REG_TILE
                                    _fwd_write_r11 = pl.reg.store(fwd_kv_flat, fwd_target_offset_r11, fwd_scaled_r11, fwd_all_r11)
                    pl.system.tpush_to_aic(kv_half, split=1, id=1)
            for part in pl.range(parts):
                column = part * SPARSE_VF_KEY_TILE
                if compressed_keys > 0:
                    qk_half_a = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], dtype=pl.FP32, split=1, id=0)
                    qk_half_b = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE // 2], dtype=pl.FP32, split=1, id=0)
                    qk_half = pl.tile.create([SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    qk_half = pl.tile.assemble(qk_half, qk_half_a, [0, 0])
                    qk_half = pl.tile.assemble(qk_half, qk_half_b, [0, SPARSE_VF_KEY_TILE // 2])
                else:
                    qk_half = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.FP32, split=1, id=0)
                scores = pl.mul(qk_half, SCALE)
                column = part * SPARSE_VF_KEY_TILE
                if column < WINDOW_KEYS:
                    ids = pl.tile.load(window_indices, [token, column], [1, SPARSE_VF_KEY_TILE])
                else:
                    c_column = column - WINDOW_KEYS
                    ids = pl.tile.load(compressed_indices, [token, c_column], [1, SPARSE_VF_KEY_TILE])
                weights = pl.tile.create([SPARSE_VF_FUSED_AIV_HEAD_TILE, SPARSE_VF_KEY_TILE], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                next_max = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                next_denominator = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                correction = pl.tile.create([1, SPARSE_VF_FUSED_AIV_HEAD_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                sm_ids_owned = pl.tile.move(ids, pl.MemorySpace.Vec)
                sm_scores_flat = pl.tile.reshape(scores, [SPARSE_VF_FUSED_AIV_HEAD_TILE * SPARSE_VF_KEY_TILE])
                sm_ids_flat = pl.tile.reshape(sm_ids_owned, [SPARSE_VF_KEY_TILE])
                sm_weights_flat = pl.tile.reshape(weights, [SPARSE_VF_FUSED_AIV_HEAD_TILE * SPARSE_VF_KEY_TILE])
                sm_old_max_flat = pl.tile.reshape(maximum, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_old_den_flat = pl.tile.reshape(denominator, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_next_max_flat = pl.tile.reshape(next_max, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_next_den_flat = pl.tile.reshape(next_denominator, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                sm_correction_flat = pl.tile.reshape(correction, [SPARSE_VF_FUSED_AIV_HEAD_TILE])
                with pl.reg_scope():
                    sm_all64 = pl.reg.full_mask(lanes=64)
                    sm_one = pl.reg.full_mask(lanes=1)
                    sm_ids0 = pl.reg.load(sm_ids_flat, 0, lanes=64)
                    sm_ids1 = pl.reg.load(sm_ids_flat, 64, lanes=64)
                    sm_valid0 = pl.reg.compares(sm_ids0, 0, sm_all64, predicate='ge')
                    sm_valid1 = pl.reg.compares(sm_ids1, 0, sm_all64, predicate='ge')
                    sm_invalid = pl.reg.broadcast(-1e+30, lanes=64, dtype=pl.FP32)
                    for sm_row in pl.hardware_loop(SPARSE_VF_FUSED_AIV_HEAD_TILE):
                        sm_offset = sm_row * SPARSE_VF_KEY_TILE
                        sm_x0 = pl.reg.load(sm_scores_flat, sm_offset, lanes=64)
                        sm_x1 = pl.reg.load(sm_scores_flat, sm_offset + 64, lanes=64)
                        sm_masked0 = pl.reg.select(sm_valid0, sm_x0, sm_invalid)
                        sm_masked1 = pl.reg.select(sm_valid1, sm_x1, sm_invalid)
                        sm_folded = pl.reg.maximum(sm_masked0, sm_masked1, sm_all64)
                        sm_tile_max = pl.reg.reduce_max(sm_folded, sm_all64)
                        sm_old_max = pl.reg.load(sm_old_max_flat, sm_row, lanes=1)
                        sm_new_max = pl.reg.maximum(sm_tile_max, sm_old_max, sm_one)
                        sm_delta = pl.reg.sub(sm_old_max, sm_new_max, sm_one)
                        sm_alpha = pl.reg.exp(sm_delta, sm_one)
                        sm_max64 = pl.reg.broadcast(sm_new_max, lanes=64, dtype=pl.FP32)
                        sm_probability0 = pl.reg.expdif(sm_masked0, sm_max64, sm_valid0)
                        sm_probability1 = pl.reg.expdif(sm_masked1, sm_max64, sm_valid1)
                        sm_sum0 = pl.reg.reduce_sum(sm_probability0, sm_all64, reassoc=True)
                        sm_sum1 = pl.reg.reduce_sum(sm_probability1, sm_all64, reassoc=True)
                        sm_sum0_ordered = pl.reg.adds(sm_sum0, 0.0, sm_one)
                        sm_tile_sum = pl.reg.add(sm_sum0_ordered, sm_sum1, sm_one)
                        sm_old_den = pl.reg.load(sm_old_den_flat, sm_row, lanes=1)
                        sm_corrected_den = pl.reg.mul(sm_old_den, sm_alpha, sm_one)
                        sm_new_den = pl.reg.add(sm_corrected_den, sm_tile_sum, sm_one)
                        sm_bf16_0 = pl.reg.convert(sm_probability0, dtype=pl.BF16, rounding='R', saturate=False)
                        sm_bf16_1 = pl.reg.convert(sm_probability1, dtype=pl.BF16, rounding='R', saturate=False)
                        _sm_store_p0 = pl.reg.store(sm_weights_flat, sm_offset, sm_bf16_0)
                        _sm_store_p1 = pl.reg.store(sm_weights_flat, sm_offset + 64, sm_bf16_1)
                        _sm_store_m = pl.reg.store(sm_next_max_flat, sm_row, sm_new_max)
                        _sm_store_l = pl.reg.store(sm_next_den_flat, sm_row, sm_new_den)
                        _sm_store_c = pl.reg.store(sm_correction_flat, sm_row, sm_alpha)
                maximum = next_max
                denominator = next_denominator
                if compressed_keys > 0:
                    pl.system.tfree_to_aic(qk_half_a, id=0)
                    pl.system.tfree_to_aic(qk_half_b, id=0)
                else:
                    pl.system.tfree_to_aic(qk_half, id=0)
                pl.system.tpush_to_aic(weights, split=1, id=2)
                if part + 2 < parts:
                    column = (part + 2) * SPARSE_VF_KEY_TILE
                    if compressed_keys > 0:
                        kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                        first_key_base = column + aiv_id * (SPARSE_VF_KEY_TILE // 4)
                        if column < WINDOW_KEYS:
                            ids_fwa = pl.tile.load(window_indices, [token, first_key_base + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            ids_fwb = pl.tile.load(window_indices, [token, first_key_base + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fwa_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            fwa_first = pl.tile.read(ids_fwa, [0, 0])
                            fwa_contiguous = pl.cast(0, pl.INT32)
                            if fwa_first >= 0:
                                fwa_contiguous = pl.cast(1, pl.INT32)
                                for fwa_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                    fwa_check_row = pl.tile.read(ids_fwa, [0, fwa_check_lane])
                                    if fwa_check_row != fwa_first + fwa_check_lane:
                                        fwa_contiguous = pl.cast(0, pl.INT32)
                            if fwa_contiguous != 0:
                                fwa_start = pl.cast(fwa_first, pl.INDEX)
                                fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [0, 0], [fwa_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                            else:
                                for fwa_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                    fwa_a = pl.tile.read(ids_fwa, [0, fwa_lane])
                                    fwa_b = pl.tile.read(ids_fwa, [0, fwa_lane + 1])
                                    fwa_lower = pl.min(fwa_a, fwa_b)
                                    fwa_upper = pl.max(fwa_a, fwa_b)
                                    fwa_pair = pl.cast(0, pl.INT32)
                                    if fwa_lower >= 0:
                                        if fwa_upper == fwa_lower + 1:
                                            fwa_pair = pl.cast(1, pl.INT32)
                                    if fwa_pair != 0:
                                        fwa_physical = pl.cast(fwa_lower, pl.INDEX)
                                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical, 0], [2, WINDOW_CACHE_BYTES])
                                    else:
                                        if fwa_a >= 0:
                                            fwa_physical_a = pl.cast(fwa_a, pl.INDEX)
                                            fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                        if fwa_b >= 0:
                                            fwa_physical_b = pl.cast(fwa_b, pl.INDEX)
                                            fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane + 1, 0], [fwa_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                            fwb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            fwb_first = pl.tile.read(ids_fwb, [0, 0])
                            fwb_contiguous = pl.cast(0, pl.INT32)
                            if fwb_first >= 0:
                                fwb_contiguous = pl.cast(1, pl.INT32)
                                for fwb_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                    fwb_check_row = pl.tile.read(ids_fwb, [0, fwb_check_lane])
                                    if fwb_check_row != fwb_first + fwb_check_lane:
                                        fwb_contiguous = pl.cast(0, pl.INT32)
                            if fwb_contiguous != 0:
                                fwb_start = pl.cast(fwb_first, pl.INDEX)
                                fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [0, 0], [fwb_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                            else:
                                for fwb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                    fwb_a = pl.tile.read(ids_fwb, [0, fwb_lane])
                                    fwb_b = pl.tile.read(ids_fwb, [0, fwb_lane + 1])
                                    fwb_lower = pl.min(fwb_a, fwb_b)
                                    fwb_upper = pl.max(fwb_a, fwb_b)
                                    fwb_pair = pl.cast(0, pl.INT32)
                                    if fwb_lower >= 0:
                                        if fwb_upper == fwb_lower + 1:
                                            fwb_pair = pl.cast(1, pl.INT32)
                                    if fwb_pair != 0:
                                        fwb_physical = pl.cast(fwb_lower, pl.INDEX)
                                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical, 0], [2, WINDOW_CACHE_BYTES])
                                    else:
                                        if fwb_a >= 0:
                                            fwb_physical_a = pl.cast(fwb_a, pl.INDEX)
                                            fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                        if fwb_b >= 0:
                                            fwb_physical_b = pl.cast(fwb_b, pl.INDEX)
                                            fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane + 1, 0], [fwb_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                            for fwa_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fwa_cast_row0 = fwa_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fwa_raw_chunk = pl.tile.slice(fwa_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwa_cast_row0, 0])
                                fwa_signed = pl.reinterpret_view(fwa_raw_chunk, pl.INT8)
                                fwa_packed = pl.reinterpret_view(fwa_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                                fwa_codes = fwa_packed[:, :HEAD_DIM]
                                fwa_values_fp32 = pl.cast(fwa_codes, pl.FP32)
                                fwa_values = pl.cast(fwa_values_fp32, pl.BF16, mode='rint')
                                fwa_scale_words = pl.reinterpret_view(fwa_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                                fwa_scale_flat = pl.tile.reshape(fwa_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                                fwa_value_flat = pl.tile.reshape(fwa_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fwa_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fwa_all_r12 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fwa_row_r12 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fwa_chunk_r12 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fwa_scale_offset_r12 = fwa_row_r12 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                            fwa_scale_offset_r12 = fwa_scale_offset_r12 + fwa_chunk_r12 * SPARSE_VF_WINDOW_SCALE_TILE
                                            fwa_groups_r12 = pl.reg.load(fwa_scale_flat, fwa_scale_offset_r12, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                            fwa_factor_bits_r12 = pl.reg.group_broadcast(fwa_groups_r12, lanes=SPARSE_VF_REG_TILE)
                                            fwa_factor_r12 = pl.reg.interpret_cast(fwa_factor_bits_r12, dtype=pl.BF16)
                                            fwa_value_offset_r12 = fwa_row_r12 * HEAD_DIM + fwa_chunk_r12 * SPARSE_VF_REG_TILE
                                            fwa_value_r12 = pl.reg.load(fwa_value_flat, fwa_value_offset_r12, lanes=SPARSE_VF_REG_TILE)
                                            fwa_scaled_r12 = pl.reg.mul(fwa_value_r12, fwa_factor_r12, fwa_all_r12)
                                            fwa_target_row_r12 = 0 + fwa_cast_row0 + fwa_row_r12
                                            fwa_target_offset_r12 = fwa_target_row_r12 * HEAD_DIM + fwa_chunk_r12 * SPARSE_VF_REG_TILE
                                            _fwa_write_r12 = pl.reg.store(fwa_kv_flat, fwa_target_offset_r12, fwa_scaled_r12, fwa_all_r12)
                            ids_fwc = pl.tile.load(window_indices, [token, first_key_base + 64], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fwc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            fwc_first = pl.tile.read(ids_fwc, [0, 0])
                            fwc_contiguous = pl.cast(0, pl.INT32)
                            if fwc_first >= 0:
                                fwc_contiguous = pl.cast(1, pl.INT32)
                                for fwc_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                    fwc_check_row = pl.tile.read(ids_fwc, [0, fwc_check_lane])
                                    if fwc_check_row != fwc_first + fwc_check_lane:
                                        fwc_contiguous = pl.cast(0, pl.INT32)
                            if fwc_contiguous != 0:
                                fwc_start = pl.cast(fwc_first, pl.INDEX)
                                fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [0, 0], [fwc_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                            else:
                                for fwc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                    fwc_a = pl.tile.read(ids_fwc, [0, fwc_lane])
                                    fwc_b = pl.tile.read(ids_fwc, [0, fwc_lane + 1])
                                    fwc_lower = pl.min(fwc_a, fwc_b)
                                    fwc_upper = pl.max(fwc_a, fwc_b)
                                    fwc_pair = pl.cast(0, pl.INT32)
                                    if fwc_lower >= 0:
                                        if fwc_upper == fwc_lower + 1:
                                            fwc_pair = pl.cast(1, pl.INT32)
                                    if fwc_pair != 0:
                                        fwc_physical = pl.cast(fwc_lower, pl.INDEX)
                                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical, 0], [2, WINDOW_CACHE_BYTES])
                                    else:
                                        if fwc_a >= 0:
                                            fwc_physical_a = pl.cast(fwc_a, pl.INDEX)
                                            fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                        if fwc_b >= 0:
                                            fwc_physical_b = pl.cast(fwc_b, pl.INDEX)
                                            fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane + 1, 0], [fwc_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                            for fwb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fwb_cast_row0 = fwb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fwb_raw_chunk = pl.tile.slice(fwb_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwb_cast_row0, 0])
                                fwb_signed = pl.reinterpret_view(fwb_raw_chunk, pl.INT8)
                                fwb_packed = pl.reinterpret_view(fwb_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                                fwb_codes = fwb_packed[:, :HEAD_DIM]
                                fwb_values_fp32 = pl.cast(fwb_codes, pl.FP32)
                                fwb_values = pl.cast(fwb_values_fp32, pl.BF16, mode='rint')
                                fwb_scale_words = pl.reinterpret_view(fwb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                                fwb_scale_flat = pl.tile.reshape(fwb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                                fwb_value_flat = pl.tile.reshape(fwb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fwb_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fwb_all_r13 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fwb_row_r13 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fwb_chunk_r13 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fwb_scale_offset_r13 = fwb_row_r13 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                            fwb_scale_offset_r13 = fwb_scale_offset_r13 + fwb_chunk_r13 * SPARSE_VF_WINDOW_SCALE_TILE
                                            fwb_groups_r13 = pl.reg.load(fwb_scale_flat, fwb_scale_offset_r13, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                            fwb_factor_bits_r13 = pl.reg.group_broadcast(fwb_groups_r13, lanes=SPARSE_VF_REG_TILE)
                                            fwb_factor_r13 = pl.reg.interpret_cast(fwb_factor_bits_r13, dtype=pl.BF16)
                                            fwb_value_offset_r13 = fwb_row_r13 * HEAD_DIM + fwb_chunk_r13 * SPARSE_VF_REG_TILE
                                            fwb_value_r13 = pl.reg.load(fwb_value_flat, fwb_value_offset_r13, lanes=SPARSE_VF_REG_TILE)
                                            fwb_scaled_r13 = pl.reg.mul(fwb_value_r13, fwb_factor_r13, fwb_all_r13)
                                            fwb_target_row_r13 = 16 + fwb_cast_row0 + fwb_row_r13
                                            fwb_target_offset_r13 = fwb_target_row_r13 * HEAD_DIM + fwb_chunk_r13 * SPARSE_VF_REG_TILE
                                            _fwb_write_r13 = pl.reg.store(fwb_kv_flat, fwb_target_offset_r13, fwb_scaled_r13, fwb_all_r13)
                            pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                            kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                            ids_fwd = pl.tile.load(window_indices, [token, first_key_base + 80], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fwd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            fwd_first = pl.tile.read(ids_fwd, [0, 0])
                            fwd_contiguous = pl.cast(0, pl.INT32)
                            if fwd_first >= 0:
                                fwd_contiguous = pl.cast(1, pl.INT32)
                                for fwd_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                    fwd_check_row = pl.tile.read(ids_fwd, [0, fwd_check_lane])
                                    if fwd_check_row != fwd_first + fwd_check_lane:
                                        fwd_contiguous = pl.cast(0, pl.INT32)
                            if fwd_contiguous != 0:
                                fwd_start = pl.cast(fwd_first, pl.INDEX)
                                fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [0, 0], [fwd_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                            else:
                                for fwd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                    fwd_a = pl.tile.read(ids_fwd, [0, fwd_lane])
                                    fwd_b = pl.tile.read(ids_fwd, [0, fwd_lane + 1])
                                    fwd_lower = pl.min(fwd_a, fwd_b)
                                    fwd_upper = pl.max(fwd_a, fwd_b)
                                    fwd_pair = pl.cast(0, pl.INT32)
                                    if fwd_lower >= 0:
                                        if fwd_upper == fwd_lower + 1:
                                            fwd_pair = pl.cast(1, pl.INT32)
                                    if fwd_pair != 0:
                                        fwd_physical = pl.cast(fwd_lower, pl.INDEX)
                                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical, 0], [2, WINDOW_CACHE_BYTES])
                                    else:
                                        if fwd_a >= 0:
                                            fwd_physical_a = pl.cast(fwd_a, pl.INDEX)
                                            fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                        if fwd_b >= 0:
                                            fwd_physical_b = pl.cast(fwd_b, pl.INDEX)
                                            fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane + 1, 0], [fwd_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                            for fwc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fwc_cast_row0 = fwc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fwc_raw_chunk = pl.tile.slice(fwc_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwc_cast_row0, 0])
                                fwc_signed = pl.reinterpret_view(fwc_raw_chunk, pl.INT8)
                                fwc_packed = pl.reinterpret_view(fwc_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                                fwc_codes = fwc_packed[:, :HEAD_DIM]
                                fwc_values_fp32 = pl.cast(fwc_codes, pl.FP32)
                                fwc_values = pl.cast(fwc_values_fp32, pl.BF16, mode='rint')
                                fwc_scale_words = pl.reinterpret_view(fwc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                                fwc_scale_flat = pl.tile.reshape(fwc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                                fwc_value_flat = pl.tile.reshape(fwc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fwc_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fwc_all_r14 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fwc_row_r14 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fwc_chunk_r14 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fwc_scale_offset_r14 = fwc_row_r14 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                            fwc_scale_offset_r14 = fwc_scale_offset_r14 + fwc_chunk_r14 * SPARSE_VF_WINDOW_SCALE_TILE
                                            fwc_groups_r14 = pl.reg.load(fwc_scale_flat, fwc_scale_offset_r14, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                            fwc_factor_bits_r14 = pl.reg.group_broadcast(fwc_groups_r14, lanes=SPARSE_VF_REG_TILE)
                                            fwc_factor_r14 = pl.reg.interpret_cast(fwc_factor_bits_r14, dtype=pl.BF16)
                                            fwc_value_offset_r14 = fwc_row_r14 * HEAD_DIM + fwc_chunk_r14 * SPARSE_VF_REG_TILE
                                            fwc_value_r14 = pl.reg.load(fwc_value_flat, fwc_value_offset_r14, lanes=SPARSE_VF_REG_TILE)
                                            fwc_scaled_r14 = pl.reg.mul(fwc_value_r14, fwc_factor_r14, fwc_all_r14)
                                            fwc_target_row_r14 = 0 + fwc_cast_row0 + fwc_row_r14
                                            fwc_target_offset_r14 = fwc_target_row_r14 * HEAD_DIM + fwc_chunk_r14 * SPARSE_VF_REG_TILE
                                            _fwc_write_r14 = pl.reg.store(fwc_kv_flat, fwc_target_offset_r14, fwc_scaled_r14, fwc_all_r14)
                            for fwd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fwd_cast_row0 = fwd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fwd_raw_chunk = pl.tile.slice(fwd_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwd_cast_row0, 0])
                                fwd_signed = pl.reinterpret_view(fwd_raw_chunk, pl.INT8)
                                fwd_packed = pl.reinterpret_view(fwd_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                                fwd_codes = fwd_packed[:, :HEAD_DIM]
                                fwd_values_fp32 = pl.cast(fwd_codes, pl.FP32)
                                fwd_values = pl.cast(fwd_values_fp32, pl.BF16, mode='rint')
                                fwd_scale_words = pl.reinterpret_view(fwd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                                fwd_scale_flat = pl.tile.reshape(fwd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                                fwd_value_flat = pl.tile.reshape(fwd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fwd_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fwd_all_r15 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fwd_row_r15 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fwd_chunk_r15 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fwd_scale_offset_r15 = fwd_row_r15 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                            fwd_scale_offset_r15 = fwd_scale_offset_r15 + fwd_chunk_r15 * SPARSE_VF_WINDOW_SCALE_TILE
                                            fwd_groups_r15 = pl.reg.load(fwd_scale_flat, fwd_scale_offset_r15, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                            fwd_factor_bits_r15 = pl.reg.group_broadcast(fwd_groups_r15, lanes=SPARSE_VF_REG_TILE)
                                            fwd_factor_r15 = pl.reg.interpret_cast(fwd_factor_bits_r15, dtype=pl.BF16)
                                            fwd_value_offset_r15 = fwd_row_r15 * HEAD_DIM + fwd_chunk_r15 * SPARSE_VF_REG_TILE
                                            fwd_value_r15 = pl.reg.load(fwd_value_flat, fwd_value_offset_r15, lanes=SPARSE_VF_REG_TILE)
                                            fwd_scaled_r15 = pl.reg.mul(fwd_value_r15, fwd_factor_r15, fwd_all_r15)
                                            fwd_target_row_r15 = 16 + fwd_cast_row0 + fwd_row_r15
                                            fwd_target_offset_r15 = fwd_target_row_r15 * HEAD_DIM + fwd_chunk_r15 * SPARSE_VF_REG_TILE
                                            _fwd_write_r15 = pl.reg.store(fwd_kv_flat, fwd_target_offset_r15, fwd_scaled_r15, fwd_all_r15)
                            pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                        else:
                            ids_fca = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            ids_fcb = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fca_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            for fca_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fca_a = pl.tile.read(ids_fca, [0, fca_lane])
                                fca_b = pl.tile.read(ids_fca, [0, fca_lane + 1])
                                fca_lower = pl.min(fca_a, fca_b)
                                fca_upper = pl.max(fca_a, fca_b)
                                fca_pair = pl.cast(0, pl.INT32)
                                if fca_lower >= 0:
                                    if fca_upper == fca_lower + 1:
                                        fca_pair = pl.cast(1, pl.INT32)
                                if fca_pair != 0:
                                    fca_physical = pl.cast(fca_lower, pl.INDEX)
                                    fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane, 0], [fca_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                                else:
                                    if fca_a >= 0:
                                        fca_physical_a = pl.cast(fca_a, pl.INDEX)
                                        fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane, 0], [fca_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                    if fca_b >= 0:
                                        fca_physical_b = pl.cast(fca_b, pl.INDEX)
                                        fca_raw = pl.tile.gather_row(fca_raw, compressed_flat, [fca_lane + 1, 0], [fca_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            fcb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            for fcb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fcb_a = pl.tile.read(ids_fcb, [0, fcb_lane])
                                fcb_b = pl.tile.read(ids_fcb, [0, fcb_lane + 1])
                                fcb_lower = pl.min(fcb_a, fcb_b)
                                fcb_upper = pl.max(fcb_a, fcb_b)
                                fcb_pair = pl.cast(0, pl.INT32)
                                if fcb_lower >= 0:
                                    if fcb_upper == fcb_lower + 1:
                                        fcb_pair = pl.cast(1, pl.INT32)
                                if fcb_pair != 0:
                                    fcb_physical = pl.cast(fcb_lower, pl.INDEX)
                                    fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane, 0], [fcb_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                                else:
                                    if fcb_a >= 0:
                                        fcb_physical_a = pl.cast(fcb_a, pl.INDEX)
                                        fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane, 0], [fcb_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                    if fcb_b >= 0:
                                        fcb_physical_b = pl.cast(fcb_b, pl.INDEX)
                                        fcb_raw = pl.tile.gather_row(fcb_raw, compressed_flat, [fcb_lane + 1, 0], [fcb_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            for fca_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fca_cast_row0 = fca_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fca_raw_chunk = pl.tile.slice(fca_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fca_cast_row0, 0])
                                fca_packed = pl.reinterpret_view(fca_raw_chunk, pl.FP4E2M1X2)
                                fca_codes = fca_packed[:, :SPARSE_VF_HALF_DIM]
                                fca_values = pl.cast(fca_codes, pl.BF16)
                                fca_scale_words = pl.reinterpret_view(fca_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                                fca_scale_flat = pl.tile.reshape(fca_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                                fca_value_flat = pl.tile.reshape(fca_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fca_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fca_all_r16 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fca_row_r16 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fca_chunk_r16 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fca_scale_offset_r16 = fca_row_r16 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                            fca_scale_offset_r16 = fca_scale_offset_r16 + fca_chunk_r16 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                            fca_groups_r16 = pl.reg.load(fca_scale_flat, fca_scale_offset_r16, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                            fca_factor_bits_r16 = pl.reg.group_broadcast(fca_groups_r16, lanes=SPARSE_VF_REG_TILE)
                                            fca_factor_r16 = pl.reg.interpret_cast(fca_factor_bits_r16, dtype=pl.BF16)
                                            fca_value_offset_r16 = fca_row_r16 * HEAD_DIM + fca_chunk_r16 * SPARSE_VF_REG_TILE
                                            fca_value_r16 = pl.reg.load(fca_value_flat, fca_value_offset_r16, lanes=SPARSE_VF_REG_TILE)
                                            fca_scaled_r16 = pl.reg.mul(fca_value_r16, fca_factor_r16, fca_all_r16)
                                            fca_target_row_r16 = 0 + fca_cast_row0 + fca_row_r16
                                            fca_target_offset_r16 = fca_target_row_r16 * HEAD_DIM + fca_chunk_r16 * SPARSE_VF_REG_TILE
                                            _fca_write_r16 = pl.reg.store(fca_kv_flat, fca_target_offset_r16, fca_scaled_r16, fca_all_r16)
                            ids_fcc = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 64], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fcc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            for fcc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fcc_a = pl.tile.read(ids_fcc, [0, fcc_lane])
                                fcc_b = pl.tile.read(ids_fcc, [0, fcc_lane + 1])
                                fcc_lower = pl.min(fcc_a, fcc_b)
                                fcc_upper = pl.max(fcc_a, fcc_b)
                                fcc_pair = pl.cast(0, pl.INT32)
                                if fcc_lower >= 0:
                                    if fcc_upper == fcc_lower + 1:
                                        fcc_pair = pl.cast(1, pl.INT32)
                                if fcc_pair != 0:
                                    fcc_physical = pl.cast(fcc_lower, pl.INDEX)
                                    fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane, 0], [fcc_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                                else:
                                    if fcc_a >= 0:
                                        fcc_physical_a = pl.cast(fcc_a, pl.INDEX)
                                        fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane, 0], [fcc_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                    if fcc_b >= 0:
                                        fcc_physical_b = pl.cast(fcc_b, pl.INDEX)
                                        fcc_raw = pl.tile.gather_row(fcc_raw, compressed_flat, [fcc_lane + 1, 0], [fcc_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            for fcb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fcb_cast_row0 = fcb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fcb_raw_chunk = pl.tile.slice(fcb_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcb_cast_row0, 0])
                                fcb_packed = pl.reinterpret_view(fcb_raw_chunk, pl.FP4E2M1X2)
                                fcb_codes = fcb_packed[:, :SPARSE_VF_HALF_DIM]
                                fcb_values = pl.cast(fcb_codes, pl.BF16)
                                fcb_scale_words = pl.reinterpret_view(fcb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcb_scale_flat = pl.tile.reshape(fcb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcb_value_flat = pl.tile.reshape(fcb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fcb_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fcb_all_r17 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fcb_row_r17 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fcb_chunk_r17 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fcb_scale_offset_r17 = fcb_row_r17 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                            fcb_scale_offset_r17 = fcb_scale_offset_r17 + fcb_chunk_r17 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                            fcb_groups_r17 = pl.reg.load(fcb_scale_flat, fcb_scale_offset_r17, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                            fcb_factor_bits_r17 = pl.reg.group_broadcast(fcb_groups_r17, lanes=SPARSE_VF_REG_TILE)
                                            fcb_factor_r17 = pl.reg.interpret_cast(fcb_factor_bits_r17, dtype=pl.BF16)
                                            fcb_value_offset_r17 = fcb_row_r17 * HEAD_DIM + fcb_chunk_r17 * SPARSE_VF_REG_TILE
                                            fcb_value_r17 = pl.reg.load(fcb_value_flat, fcb_value_offset_r17, lanes=SPARSE_VF_REG_TILE)
                                            fcb_scaled_r17 = pl.reg.mul(fcb_value_r17, fcb_factor_r17, fcb_all_r17)
                                            fcb_target_row_r17 = 16 + fcb_cast_row0 + fcb_row_r17
                                            fcb_target_offset_r17 = fcb_target_row_r17 * HEAD_DIM + fcb_chunk_r17 * SPARSE_VF_REG_TILE
                                            _fcb_write_r17 = pl.reg.store(fcb_kv_flat, fcb_target_offset_r17, fcb_scaled_r17, fcb_all_r17)
                            pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                            kv_chunk = pl.tile.create([SPARSE_VF_KEY_TILE // 4, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                            ids_fcd = pl.tile.load(compressed_indices, [token, first_key_base - WINDOW_KEYS + 80], [1, SPARSE_VF_FUSED_GATHER_TILE])
                            fcd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8, value=0)
                            for fcd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fcd_a = pl.tile.read(ids_fcd, [0, fcd_lane])
                                fcd_b = pl.tile.read(ids_fcd, [0, fcd_lane + 1])
                                fcd_lower = pl.min(fcd_a, fcd_b)
                                fcd_upper = pl.max(fcd_a, fcd_b)
                                fcd_pair = pl.cast(0, pl.INT32)
                                if fcd_lower >= 0:
                                    if fcd_upper == fcd_lower + 1:
                                        fcd_pair = pl.cast(1, pl.INT32)
                                if fcd_pair != 0:
                                    fcd_physical = pl.cast(fcd_lower, pl.INDEX)
                                    fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane, 0], [fcd_physical, 0], [2, COMPRESSED_CACHE_BYTES])
                                else:
                                    if fcd_a >= 0:
                                        fcd_physical_a = pl.cast(fcd_a, pl.INDEX)
                                        fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane, 0], [fcd_physical_a, 0], [1, COMPRESSED_CACHE_BYTES])
                                    if fcd_b >= 0:
                                        fcd_physical_b = pl.cast(fcd_b, pl.INDEX)
                                        fcd_raw = pl.tile.gather_row(fcd_raw, compressed_flat, [fcd_lane + 1, 0], [fcd_physical_b, 0], [1, COMPRESSED_CACHE_BYTES])
                            for fcc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fcc_cast_row0 = fcc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fcc_raw_chunk = pl.tile.slice(fcc_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcc_cast_row0, 0])
                                fcc_packed = pl.reinterpret_view(fcc_raw_chunk, pl.FP4E2M1X2)
                                fcc_codes = fcc_packed[:, :SPARSE_VF_HALF_DIM]
                                fcc_values = pl.cast(fcc_codes, pl.BF16)
                                fcc_scale_words = pl.reinterpret_view(fcc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcc_scale_flat = pl.tile.reshape(fcc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcc_value_flat = pl.tile.reshape(fcc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fcc_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fcc_all_r18 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fcc_row_r18 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fcc_chunk_r18 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fcc_scale_offset_r18 = fcc_row_r18 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                            fcc_scale_offset_r18 = fcc_scale_offset_r18 + fcc_chunk_r18 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                            fcc_groups_r18 = pl.reg.load(fcc_scale_flat, fcc_scale_offset_r18, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                            fcc_factor_bits_r18 = pl.reg.group_broadcast(fcc_groups_r18, lanes=SPARSE_VF_REG_TILE)
                                            fcc_factor_r18 = pl.reg.interpret_cast(fcc_factor_bits_r18, dtype=pl.BF16)
                                            fcc_value_offset_r18 = fcc_row_r18 * HEAD_DIM + fcc_chunk_r18 * SPARSE_VF_REG_TILE
                                            fcc_value_r18 = pl.reg.load(fcc_value_flat, fcc_value_offset_r18, lanes=SPARSE_VF_REG_TILE)
                                            fcc_scaled_r18 = pl.reg.mul(fcc_value_r18, fcc_factor_r18, fcc_all_r18)
                                            fcc_target_row_r18 = 0 + fcc_cast_row0 + fcc_row_r18
                                            fcc_target_offset_r18 = fcc_target_row_r18 * HEAD_DIM + fcc_chunk_r18 * SPARSE_VF_REG_TILE
                                            _fcc_write_r18 = pl.reg.store(fcc_kv_flat, fcc_target_offset_r18, fcc_scaled_r18, fcc_all_r18)
                            for fcd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                                fcd_cast_row0 = fcd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                                fcd_raw_chunk = pl.tile.slice(fcd_raw, [SPARSE_VF_FUSED_SCALE_TILE, COMPRESSED_CACHE_BYTES], [fcd_cast_row0, 0])
                                fcd_packed = pl.reinterpret_view(fcd_raw_chunk, pl.FP4E2M1X2)
                                fcd_codes = fcd_packed[:, :SPARSE_VF_HALF_DIM]
                                fcd_values = pl.cast(fcd_codes, pl.BF16)
                                fcd_scale_words = pl.reinterpret_view(fcd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcd_scale_flat = pl.tile.reshape(fcd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_COMPRESSED_HALFWORDS])
                                fcd_value_flat = pl.tile.reshape(fcd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                                fcd_kv_flat = pl.tile.reshape(kv_chunk, [SPARSE_VF_KEY_TILE // 4 * HEAD_DIM])
                                with pl.reg_scope():
                                    fcd_all_r19 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                    for fcd_row_r19 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                        for fcd_chunk_r19 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                            fcd_scale_offset_r19 = fcd_row_r19 * SPARSE_VF_COMPRESSED_HALFWORDS + SPARSE_VF_COMPRESSED_SCALE_OFFSET
                                            fcd_scale_offset_r19 = fcd_scale_offset_r19 + fcd_chunk_r19 * SPARSE_VF_COMPRESSED_SCALE_TILE
                                            fcd_groups_r19 = pl.reg.load(fcd_scale_flat, fcd_scale_offset_r19, lanes=SPARSE_VF_COMPRESSED_SCALE_TILE)
                                            fcd_factor_bits_r19 = pl.reg.group_broadcast(fcd_groups_r19, lanes=SPARSE_VF_REG_TILE)
                                            fcd_factor_r19 = pl.reg.interpret_cast(fcd_factor_bits_r19, dtype=pl.BF16)
                                            fcd_value_offset_r19 = fcd_row_r19 * HEAD_DIM + fcd_chunk_r19 * SPARSE_VF_REG_TILE
                                            fcd_value_r19 = pl.reg.load(fcd_value_flat, fcd_value_offset_r19, lanes=SPARSE_VF_REG_TILE)
                                            fcd_scaled_r19 = pl.reg.mul(fcd_value_r19, fcd_factor_r19, fcd_all_r19)
                                            fcd_target_row_r19 = 16 + fcd_cast_row0 + fcd_row_r19
                                            fcd_target_offset_r19 = fcd_target_row_r19 * HEAD_DIM + fcd_chunk_r19 * SPARSE_VF_REG_TILE
                                            _fcd_write_r19 = pl.reg.store(fcd_kv_flat, fcd_target_offset_r19, fcd_scaled_r19, fcd_all_r19)
                            pl.system.tpush_to_aic(kv_chunk, split=1, id=1)
                    else:
                        kv_half = pl.tile.create([SPARSE_VF_KEY_TILE // 2, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec)
                        first_key_base = column + aiv_id * (SPARSE_VF_KEY_TILE // 2)
                        ids_fwa = pl.tile.load(window_indices, [token, first_key_base + 0], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        ids_fwb = pl.tile.load(window_indices, [token, first_key_base + 16], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwa_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwa_first = pl.tile.read(ids_fwa, [0, 0])
                        fwa_contiguous = pl.cast(0, pl.INT32)
                        if fwa_first >= 0:
                            fwa_contiguous = pl.cast(1, pl.INT32)
                            for fwa_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwa_check_row = pl.tile.read(ids_fwa, [0, fwa_check_lane])
                                if fwa_check_row != fwa_first + fwa_check_lane:
                                    fwa_contiguous = pl.cast(0, pl.INT32)
                        if fwa_contiguous != 0:
                            fwa_start = pl.cast(fwa_first, pl.INDEX)
                            fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [0, 0], [fwa_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwa_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwa_a = pl.tile.read(ids_fwa, [0, fwa_lane])
                                fwa_b = pl.tile.read(ids_fwa, [0, fwa_lane + 1])
                                fwa_lower = pl.min(fwa_a, fwa_b)
                                fwa_upper = pl.max(fwa_a, fwa_b)
                                fwa_pair = pl.cast(0, pl.INT32)
                                if fwa_lower >= 0:
                                    if fwa_upper == fwa_lower + 1:
                                        fwa_pair = pl.cast(1, pl.INT32)
                                if fwa_pair != 0:
                                    fwa_physical = pl.cast(fwa_lower, pl.INDEX)
                                    fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwa_a >= 0:
                                        fwa_physical_a = pl.cast(fwa_a, pl.INDEX)
                                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane, 0], [fwa_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwa_b >= 0:
                                        fwa_physical_b = pl.cast(fwa_b, pl.INDEX)
                                        fwa_raw = pl.tile.gather_row(fwa_raw, window_flat, [fwa_lane + 1, 0], [fwa_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        fwb_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwb_first = pl.tile.read(ids_fwb, [0, 0])
                        fwb_contiguous = pl.cast(0, pl.INT32)
                        if fwb_first >= 0:
                            fwb_contiguous = pl.cast(1, pl.INT32)
                            for fwb_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwb_check_row = pl.tile.read(ids_fwb, [0, fwb_check_lane])
                                if fwb_check_row != fwb_first + fwb_check_lane:
                                    fwb_contiguous = pl.cast(0, pl.INT32)
                        if fwb_contiguous != 0:
                            fwb_start = pl.cast(fwb_first, pl.INDEX)
                            fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [0, 0], [fwb_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwb_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwb_a = pl.tile.read(ids_fwb, [0, fwb_lane])
                                fwb_b = pl.tile.read(ids_fwb, [0, fwb_lane + 1])
                                fwb_lower = pl.min(fwb_a, fwb_b)
                                fwb_upper = pl.max(fwb_a, fwb_b)
                                fwb_pair = pl.cast(0, pl.INT32)
                                if fwb_lower >= 0:
                                    if fwb_upper == fwb_lower + 1:
                                        fwb_pair = pl.cast(1, pl.INT32)
                                if fwb_pair != 0:
                                    fwb_physical = pl.cast(fwb_lower, pl.INDEX)
                                    fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwb_a >= 0:
                                        fwb_physical_a = pl.cast(fwb_a, pl.INDEX)
                                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane, 0], [fwb_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwb_b >= 0:
                                        fwb_physical_b = pl.cast(fwb_b, pl.INDEX)
                                        fwb_raw = pl.tile.gather_row(fwb_raw, window_flat, [fwb_lane + 1, 0], [fwb_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwa_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwa_cast_row0 = fwa_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwa_raw_chunk = pl.tile.slice(fwa_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwa_cast_row0, 0])
                            fwa_signed = pl.reinterpret_view(fwa_raw_chunk, pl.INT8)
                            fwa_packed = pl.reinterpret_view(fwa_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwa_codes = fwa_packed[:, :HEAD_DIM]
                            fwa_values_fp32 = pl.cast(fwa_codes, pl.FP32)
                            fwa_values = pl.cast(fwa_values_fp32, pl.BF16, mode='rint')
                            fwa_scale_words = pl.reinterpret_view(fwa_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwa_scale_flat = pl.tile.reshape(fwa_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwa_value_flat = pl.tile.reshape(fwa_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwa_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                            with pl.reg_scope():
                                fwa_all_r20 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwa_row_r20 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwa_chunk_r20 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwa_scale_offset_r20 = fwa_row_r20 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwa_scale_offset_r20 = fwa_scale_offset_r20 + fwa_chunk_r20 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwa_groups_r20 = pl.reg.load(fwa_scale_flat, fwa_scale_offset_r20, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwa_factor_bits_r20 = pl.reg.group_broadcast(fwa_groups_r20, lanes=SPARSE_VF_REG_TILE)
                                        fwa_factor_r20 = pl.reg.interpret_cast(fwa_factor_bits_r20, dtype=pl.BF16)
                                        fwa_value_offset_r20 = fwa_row_r20 * HEAD_DIM + fwa_chunk_r20 * SPARSE_VF_REG_TILE
                                        fwa_value_r20 = pl.reg.load(fwa_value_flat, fwa_value_offset_r20, lanes=SPARSE_VF_REG_TILE)
                                        fwa_scaled_r20 = pl.reg.mul(fwa_value_r20, fwa_factor_r20, fwa_all_r20)
                                        fwa_target_row_r20 = 0 + fwa_cast_row0 + fwa_row_r20
                                        fwa_target_offset_r20 = fwa_target_row_r20 * HEAD_DIM + fwa_chunk_r20 * SPARSE_VF_REG_TILE
                                        _fwa_write_r20 = pl.reg.store(fwa_kv_flat, fwa_target_offset_r20, fwa_scaled_r20, fwa_all_r20)
                        ids_fwc = pl.tile.load(window_indices, [token, first_key_base + 32], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwc_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwc_first = pl.tile.read(ids_fwc, [0, 0])
                        fwc_contiguous = pl.cast(0, pl.INT32)
                        if fwc_first >= 0:
                            fwc_contiguous = pl.cast(1, pl.INT32)
                            for fwc_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwc_check_row = pl.tile.read(ids_fwc, [0, fwc_check_lane])
                                if fwc_check_row != fwc_first + fwc_check_lane:
                                    fwc_contiguous = pl.cast(0, pl.INT32)
                        if fwc_contiguous != 0:
                            fwc_start = pl.cast(fwc_first, pl.INDEX)
                            fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [0, 0], [fwc_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwc_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwc_a = pl.tile.read(ids_fwc, [0, fwc_lane])
                                fwc_b = pl.tile.read(ids_fwc, [0, fwc_lane + 1])
                                fwc_lower = pl.min(fwc_a, fwc_b)
                                fwc_upper = pl.max(fwc_a, fwc_b)
                                fwc_pair = pl.cast(0, pl.INT32)
                                if fwc_lower >= 0:
                                    if fwc_upper == fwc_lower + 1:
                                        fwc_pair = pl.cast(1, pl.INT32)
                                if fwc_pair != 0:
                                    fwc_physical = pl.cast(fwc_lower, pl.INDEX)
                                    fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwc_a >= 0:
                                        fwc_physical_a = pl.cast(fwc_a, pl.INDEX)
                                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane, 0], [fwc_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwc_b >= 0:
                                        fwc_physical_b = pl.cast(fwc_b, pl.INDEX)
                                        fwc_raw = pl.tile.gather_row(fwc_raw, window_flat, [fwc_lane + 1, 0], [fwc_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwb_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwb_cast_row0 = fwb_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwb_raw_chunk = pl.tile.slice(fwb_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwb_cast_row0, 0])
                            fwb_signed = pl.reinterpret_view(fwb_raw_chunk, pl.INT8)
                            fwb_packed = pl.reinterpret_view(fwb_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwb_codes = fwb_packed[:, :HEAD_DIM]
                            fwb_values_fp32 = pl.cast(fwb_codes, pl.FP32)
                            fwb_values = pl.cast(fwb_values_fp32, pl.BF16, mode='rint')
                            fwb_scale_words = pl.reinterpret_view(fwb_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwb_scale_flat = pl.tile.reshape(fwb_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwb_value_flat = pl.tile.reshape(fwb_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwb_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                            with pl.reg_scope():
                                fwb_all_r21 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwb_row_r21 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwb_chunk_r21 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwb_scale_offset_r21 = fwb_row_r21 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwb_scale_offset_r21 = fwb_scale_offset_r21 + fwb_chunk_r21 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwb_groups_r21 = pl.reg.load(fwb_scale_flat, fwb_scale_offset_r21, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwb_factor_bits_r21 = pl.reg.group_broadcast(fwb_groups_r21, lanes=SPARSE_VF_REG_TILE)
                                        fwb_factor_r21 = pl.reg.interpret_cast(fwb_factor_bits_r21, dtype=pl.BF16)
                                        fwb_value_offset_r21 = fwb_row_r21 * HEAD_DIM + fwb_chunk_r21 * SPARSE_VF_REG_TILE
                                        fwb_value_r21 = pl.reg.load(fwb_value_flat, fwb_value_offset_r21, lanes=SPARSE_VF_REG_TILE)
                                        fwb_scaled_r21 = pl.reg.mul(fwb_value_r21, fwb_factor_r21, fwb_all_r21)
                                        fwb_target_row_r21 = 16 + fwb_cast_row0 + fwb_row_r21
                                        fwb_target_offset_r21 = fwb_target_row_r21 * HEAD_DIM + fwb_chunk_r21 * SPARSE_VF_REG_TILE
                                        _fwb_write_r21 = pl.reg.store(fwb_kv_flat, fwb_target_offset_r21, fwb_scaled_r21, fwb_all_r21)
                        ids_fwd = pl.tile.load(window_indices, [token, first_key_base + 48], [1, SPARSE_VF_FUSED_GATHER_TILE])
                        fwd_raw = pl.tile.full([SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES], dtype=pl.UINT8, value=0)
                        fwd_first = pl.tile.read(ids_fwd, [0, 0])
                        fwd_contiguous = pl.cast(0, pl.INT32)
                        if fwd_first >= 0:
                            fwd_contiguous = pl.cast(1, pl.INT32)
                            for fwd_check_lane in pl.range(1, SPARSE_VF_FUSED_GATHER_TILE):
                                fwd_check_row = pl.tile.read(ids_fwd, [0, fwd_check_lane])
                                if fwd_check_row != fwd_first + fwd_check_lane:
                                    fwd_contiguous = pl.cast(0, pl.INT32)
                        if fwd_contiguous != 0:
                            fwd_start = pl.cast(fwd_first, pl.INDEX)
                            fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [0, 0], [fwd_start, 0], [SPARSE_VF_FUSED_GATHER_TILE, WINDOW_CACHE_BYTES])
                        else:
                            for fwd_lane in pl.range(0, SPARSE_VF_FUSED_GATHER_TILE, 2):
                                fwd_a = pl.tile.read(ids_fwd, [0, fwd_lane])
                                fwd_b = pl.tile.read(ids_fwd, [0, fwd_lane + 1])
                                fwd_lower = pl.min(fwd_a, fwd_b)
                                fwd_upper = pl.max(fwd_a, fwd_b)
                                fwd_pair = pl.cast(0, pl.INT32)
                                if fwd_lower >= 0:
                                    if fwd_upper == fwd_lower + 1:
                                        fwd_pair = pl.cast(1, pl.INT32)
                                if fwd_pair != 0:
                                    fwd_physical = pl.cast(fwd_lower, pl.INDEX)
                                    fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical, 0], [2, WINDOW_CACHE_BYTES])
                                else:
                                    if fwd_a >= 0:
                                        fwd_physical_a = pl.cast(fwd_a, pl.INDEX)
                                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane, 0], [fwd_physical_a, 0], [1, WINDOW_CACHE_BYTES])
                                    if fwd_b >= 0:
                                        fwd_physical_b = pl.cast(fwd_b, pl.INDEX)
                                        fwd_raw = pl.tile.gather_row(fwd_raw, window_flat, [fwd_lane + 1, 0], [fwd_physical_b, 0], [1, WINDOW_CACHE_BYTES])
                        for fwc_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwc_cast_row0 = fwc_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwc_raw_chunk = pl.tile.slice(fwc_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwc_cast_row0, 0])
                            fwc_signed = pl.reinterpret_view(fwc_raw_chunk, pl.INT8)
                            fwc_packed = pl.reinterpret_view(fwc_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwc_codes = fwc_packed[:, :HEAD_DIM]
                            fwc_values_fp32 = pl.cast(fwc_codes, pl.FP32)
                            fwc_values = pl.cast(fwc_values_fp32, pl.BF16, mode='rint')
                            fwc_scale_words = pl.reinterpret_view(fwc_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwc_scale_flat = pl.tile.reshape(fwc_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwc_value_flat = pl.tile.reshape(fwc_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwc_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                            with pl.reg_scope():
                                fwc_all_r22 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwc_row_r22 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwc_chunk_r22 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwc_scale_offset_r22 = fwc_row_r22 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwc_scale_offset_r22 = fwc_scale_offset_r22 + fwc_chunk_r22 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwc_groups_r22 = pl.reg.load(fwc_scale_flat, fwc_scale_offset_r22, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwc_factor_bits_r22 = pl.reg.group_broadcast(fwc_groups_r22, lanes=SPARSE_VF_REG_TILE)
                                        fwc_factor_r22 = pl.reg.interpret_cast(fwc_factor_bits_r22, dtype=pl.BF16)
                                        fwc_value_offset_r22 = fwc_row_r22 * HEAD_DIM + fwc_chunk_r22 * SPARSE_VF_REG_TILE
                                        fwc_value_r22 = pl.reg.load(fwc_value_flat, fwc_value_offset_r22, lanes=SPARSE_VF_REG_TILE)
                                        fwc_scaled_r22 = pl.reg.mul(fwc_value_r22, fwc_factor_r22, fwc_all_r22)
                                        fwc_target_row_r22 = 32 + fwc_cast_row0 + fwc_row_r22
                                        fwc_target_offset_r22 = fwc_target_row_r22 * HEAD_DIM + fwc_chunk_r22 * SPARSE_VF_REG_TILE
                                        _fwc_write_r22 = pl.reg.store(fwc_kv_flat, fwc_target_offset_r22, fwc_scaled_r22, fwc_all_r22)
                        for fwd_cast_chunk in pl.unroll(SPARSE_VF_FUSED_GATHER_TILE // SPARSE_VF_FUSED_SCALE_TILE):
                            fwd_cast_row0 = fwd_cast_chunk * SPARSE_VF_FUSED_SCALE_TILE
                            fwd_raw_chunk = pl.tile.slice(fwd_raw, [SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES], [fwd_cast_row0, 0])
                            fwd_signed = pl.reinterpret_view(fwd_raw_chunk, pl.INT8)
                            fwd_packed = pl.reinterpret_view(fwd_signed, pl.FP8E4M3FN, shape=[SPARSE_VF_FUSED_SCALE_TILE, WINDOW_CACHE_BYTES])
                            fwd_codes = fwd_packed[:, :HEAD_DIM]
                            fwd_values_fp32 = pl.cast(fwd_codes, pl.FP32)
                            fwd_values = pl.cast(fwd_values_fp32, pl.BF16, mode='rint')
                            fwd_scale_words = pl.reinterpret_view(fwd_raw_chunk, pl.UINT16, shape=[SPARSE_VF_FUSED_SCALE_TILE, SPARSE_VF_WINDOW_HALFWORDS])
                            fwd_scale_flat = pl.tile.reshape(fwd_scale_words, [SPARSE_VF_FUSED_SCALE_TILE * SPARSE_VF_WINDOW_HALFWORDS])
                            fwd_value_flat = pl.tile.reshape(fwd_values, [SPARSE_VF_FUSED_SCALE_TILE * HEAD_DIM])
                            fwd_kv_flat = pl.tile.reshape(kv_half, [SPARSE_VF_KEY_TILE // 2 * HEAD_DIM])
                            with pl.reg_scope():
                                fwd_all_r23 = pl.reg.full_mask(lanes=SPARSE_VF_REG_TILE)
                                for fwd_row_r23 in pl.hardware_loop(SPARSE_VF_FUSED_SCALE_TILE):
                                    for fwd_chunk_r23 in pl.hardware_loop(HEAD_DIM // SPARSE_VF_REG_TILE):
                                        fwd_scale_offset_r23 = fwd_row_r23 * SPARSE_VF_WINDOW_HALFWORDS + SPARSE_VF_WINDOW_SCALE_OFFSET
                                        fwd_scale_offset_r23 = fwd_scale_offset_r23 + fwd_chunk_r23 * SPARSE_VF_WINDOW_SCALE_TILE
                                        fwd_groups_r23 = pl.reg.load(fwd_scale_flat, fwd_scale_offset_r23, lanes=SPARSE_VF_WINDOW_SCALE_TILE)
                                        fwd_factor_bits_r23 = pl.reg.group_broadcast(fwd_groups_r23, lanes=SPARSE_VF_REG_TILE)
                                        fwd_factor_r23 = pl.reg.interpret_cast(fwd_factor_bits_r23, dtype=pl.BF16)
                                        fwd_value_offset_r23 = fwd_row_r23 * HEAD_DIM + fwd_chunk_r23 * SPARSE_VF_REG_TILE
                                        fwd_value_r23 = pl.reg.load(fwd_value_flat, fwd_value_offset_r23, lanes=SPARSE_VF_REG_TILE)
                                        fwd_scaled_r23 = pl.reg.mul(fwd_value_r23, fwd_factor_r23, fwd_all_r23)
                                        fwd_target_row_r23 = 48 + fwd_cast_row0 + fwd_row_r23
                                        fwd_target_offset_r23 = fwd_target_row_r23 * HEAD_DIM + fwd_chunk_r23 * SPARSE_VF_REG_TILE
                                        _fwd_write_r23 = pl.reg.store(fwd_kv_flat, fwd_target_offset_r23, fwd_scaled_r23, fwd_all_r23)
                        pl.system.tpush_to_aic(kv_half, split=1, id=1)
                correction_col = pl.reshape(correction, [SPARSE_VF_FUSED_AIV_HEAD_TILE, 1])
                weighted_0 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_0 = pl.row_expand_mul(numerator_0, correction_col)
                numerator_0 = pl.add(scaled_0, weighted_0)
                pl.system.tfree_to_aic(weighted_0, id=3)
                weighted_1 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_1 = pl.row_expand_mul(numerator_1, correction_col)
                numerator_1 = pl.add(scaled_1, weighted_1)
                pl.system.tfree_to_aic(weighted_1, id=3)
                weighted_2 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_2 = pl.row_expand_mul(numerator_2, correction_col)
                numerator_2 = pl.add(scaled_2, weighted_2)
                pl.system.tfree_to_aic(weighted_2, id=3)
                weighted_3 = pl.system.tpop_from_aic(shape=[SPARSE_VF_FUSED_AIV_HEAD_TILE, 128], dtype=pl.FP32, split=1, id=3)
                scaled_3 = pl.row_expand_mul(numerator_3, correction_col)
                numerator_3 = pl.add(scaled_3, weighted_3)
                pl.system.tfree_to_aic(weighted_3, id=3)
        lane_head = head + aiv_id * SPARSE_VF_FUSED_AIV_HEAD_TILE
        remaining = SPARSE_VF_FUSED_HEAD_TILE - aiv_id * SPARSE_VF_FUSED_AIV_HEAD_TILE
        valid_heads = pl.min(remaining, SPARSE_VF_FUSED_AIV_HEAD_TILE)
        if valid_heads > 0:
            sinks = pl.tile.load(sink_row, [0, lane_head], [1, SPARSE_VF_FUSED_AIV_HEAD_TILE], valid_shape=[1, valid_heads])
            final_max = pl.maximum(maximum, sinks)
            correction = pl.col_expand_expdif(maximum, final_max)
            sink_weight = pl.col_expand_expdif(sinks, final_max)
            corrected_total = pl.mul(denominator, correction)
            total = pl.add(corrected_total, sink_weight)
            ratio = pl.div(correction, total)
            multiplier = pl.reshape(ratio, [SPARSE_VF_FUSED_AIV_HEAD_TILE, 1])
            store_start = q0 + aiv_id * SPARSE_VF_FUSED_AIV_HEAD_TILE
            normalized_0 = pl.row_expand_mul(numerator_0, multiplier)
            result_0 = pl.cast(normalized_0, pl.BF16, mode='rint')
            valid_result_0 = pl.tile.set_validshape(result_0, valid_heads, 128)
            pl.tile.store(valid_result_0, [store_start, 0], oflat)
            normalized_1 = pl.row_expand_mul(numerator_1, multiplier)
            result_1 = pl.cast(normalized_1, pl.BF16, mode='rint')
            valid_result_1 = pl.tile.set_validshape(result_1, valid_heads, 128)
            pl.tile.store(valid_result_1, [store_start, 128], oflat)
            normalized_2 = pl.row_expand_mul(numerator_2, multiplier)
            result_2 = pl.cast(normalized_2, pl.BF16, mode='rint')
            valid_result_2 = pl.tile.set_validshape(result_2, valid_heads, 128)
            pl.tile.store(valid_result_2, [store_start, 256], oflat)
            normalized_3 = pl.row_expand_mul(numerator_3, multiplier)
            result_3 = pl.cast(normalized_3, pl.BF16, mode='rint')
            valid_result_3 = pl.tile.set_validshape(result_3, valid_heads, 128)
            pl.tile.store(valid_result_3, [store_start, 384], oflat)
    return oflat


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def sparse_vf_fused_group(
    qflat: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_indices: pl.Tensor,
    window_flat: pl.Tensor,
    compressed_flat: pl.Tensor,
    sink_row: pl.Tensor,
    oflat: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    compressed_keys: pl.constexpr,
):
    oflat = sparse_vf_fused_cube(
        qflat,
        window_indices, compressed_indices,
        window_flat, compressed_flat,
        sink_row,
        oflat,
        num_tokens, compressed_keys=compressed_keys,
    )
    oflat = sparse_vf_fused_vector(
        qflat,
        window_indices, compressed_indices,
        window_flat, compressed_flat,
        sink_row,
        oflat,
        num_tokens, compressed_keys=compressed_keys,
    )
    return oflat


@pl.jit.inline(auto_scope=False)
def gather_sparse_attention_pypto(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
    compressed_keys: pl.constexpr,
):
    """Fuse packed-cache decoding and attention; keep single-part SWA scheduling."""
    query = query[:num_tokens, :]
    output = output[:num_tokens, :]
    window_indices = window_indices[:num_tokens, :]
    compressed_indices = compressed_indices[:num_tokens, :]
    attended = pl.system.task_invalid()
    if compressed_keys > 0 and LOCAL_H >= 32:
        window_rows = pl.tensor.dim(window, 0) * 128
        window_flat = pl.reshape(window, [window_rows, WINDOW_CACHE_BYTES])
        compressed_rows = pl.tensor.dim(compressed, 0) * 128
        compressed_flat = pl.reshape(compressed, [compressed_rows, COMPRESSED_CACHE_BYTES])
        tokens = pl.tensor.dim(query, 0)
        qflat = pl.reshape(query, [tokens * LOCAL_H, HEAD_DIM])
        oflat = pl.reshape(output, [tokens * LOCAL_H, HEAD_DIM])
        sink_row = pl.reshape(sink, [1, LOCAL_H])
        workers = pl.min(num_tokens * (LOCAL_H // SPARSE_VF_FUSED_HEAD_TILE), SPARSE_VF_WORKER_TILE)
        workers = pl.max(1, workers)
        with pl.spmd(workers, name_hint='sparse_vf_fused', deps=[source_ready]) as fused:
            sparse_vf_fused_group(
                qflat,
                window_indices, compressed_indices,
                window_flat, compressed_flat,
                sink_row,
                oflat,
                num_tokens, compressed_keys=compressed_keys,
            )
        attended = fused
    else:
        keys = WINDOW_KEYS + compressed_keys
        active_rows = pl.max(1, num_tokens)
        rows = active_rows * keys
        decoded = pl.create_tensor([rows, HEAD_DIM], dtype=pl.BF16)
        selected_ids = pl.create_tensor([active_rows, keys], dtype=pl.INT32)
        gathered = sparse_vf_decode_pypto(
            window_indices, window,
            compressed_indices, compressed,
            decoded, selected_ids,
            num_tokens, source_ready, compressed_keys=compressed_keys,
        )
        attended = sparse_vf_attend_pypto(
            query,
            decoded, selected_ids,
            sink,
            output,
            num_tokens, gathered, compressed_keys=compressed_keys,
        )
    return attended


@pl.jit(auto_scope=False)
def l2_gather_sparse_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, CACHE_ROWS_DYN, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    start: pl.Scalar[pl.INT32],
    compressed_keys: pl.Scalar[pl.INT32],
):
    compressed.bind_dynamic(1, CACHE_ROWS_DYN)
    if num_tokens > 0:
        q = query[start:start + num_tokens, :]
        wi = window_indices[start:start + num_tokens, :]
        ci = compressed_indices[start:start + num_tokens, :]
        if compressed_keys == 0:
            gather_sparse_attention(q, wi, window, wi, window, sink, output, num_tokens, 0)
        else:
            gather_sparse_attention(q, wi, window, ci, compressed, sink, output, num_tokens, INDEX_TOPK)
    return output


def golden_gather_sparse_attention(
    query: torch.Tensor,
    window_indices: torch.Tensor,
    window: torch.Tensor,
    compressed_indices: torch.Tensor,
    compressed: torch.Tensor,
    sink: torch.Tensor,
    *,
    compressed_keys: int,
    num_tokens: int | None = None,
    key_tile: int = 128,
) -> torch.Tensor:
    """Attend packed FP8/FP4 cache rows with online BF16 PV and one final sink."""
    tokens = query.shape[0] if num_tokens is None else num_tokens
    heads = sink.numel()
    head_dim = query.shape[-1] if query.ndim == 3 else query.shape[-1] // heads
    q = query[:tokens].reshape(tokens, heads, head_dim)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    sources = [(window, window_indices, False)]
    if compressed_keys:
        sources.append((compressed, compressed_indices[:, :compressed_keys], True))
    for token_begin in range(0, tokens, 128):
        token_end = min(token_begin + 128, tokens)
        chunk = q[token_begin:token_end].float()
        maximum = chunk.new_full(chunk.shape[:-1], -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(chunk)
        for cache, indices, fp4 in sources:
            flat_cache = cache.reshape(-1, cache.shape[-1])
            for begin in range(0, indices.shape[-1], key_tile):
                rows = indices[token_begin:token_end, begin:begin + key_tile].long()
                valid = rows >= 0
                if not valid.any():
                    continue
                packed = flat_cache[rows.clamp_min(0)]
                payload, scales = unpack_quantized_cache(packed, fp4, head_dim=head_dim)
                if fp4:
                    values = dequantize_mxfp4_cache(payload, scales, 16, "bf16")
                else:
                    values = dequantize_mxfp8_cache(payload, scales)
                values = values.bfloat16().float().masked_fill(~valid.unsqueeze(-1), 0)
                scores = torch.einsum("thd,tkd->thk", chunk, values) * head_dim**-0.5
                scores = scores.masked_fill(~valid.unsqueeze(1), -1e30)
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = torch.exp(maximum - next_maximum)
                probabilities = torch.exp(scores - next_maximum.unsqueeze(-1))
                probabilities = probabilities.masked_fill(~valid.unsqueeze(1), 0)
                denominator = denominator * correction + probabilities.sum(dim=-1)
                weighted = torch.einsum("thk,tkd->thd", probabilities.bfloat16().float(), values)
                numerator = numerator * correction.unsqueeze(-1) + weighted
                maximum = next_maximum
        final_maximum = torch.maximum(maximum, sink.float())
        correction = torch.exp(maximum - final_maximum)
        denominator = denominator * correction + torch.exp(sink.float() - final_maximum)
        output[token_begin:token_end] = (numerator * (correction / denominator).unsqueeze(-1)).bfloat16()
    return output.reshape(query[:tokens].shape)


def build_sparse_attention_specs(rows, active, *, compressed_keys, start=0, case="normal"):
    """Build packed caches, irregular indices, poisoned query padding and output sentinels."""
    from golden import ScalarSpec, TensorSpec

    if not 0 <= start <= rows or not 0 <= active <= rows - start:
        raise ValueError("active query range must fit in rows")
    if compressed_keys not in (0, INDEX_TOPK) or case not in ("normal", "codes"):
        raise ValueError("unsupported compressed key count or fixture case")
    window_pages = 1 if case == "codes" else 2
    compressed_pages = 1 if not compressed_keys or case == "codes" else 4
    values = {}

    def initialize(name):
        if not values:
            generator = torch.Generator().manual_seed(4102026 + active)
            query = torch.randn(rows, LOCAL_H * HEAD_DIM, generator=generator).mul_(0.35).bfloat16()
            window = torch.randn(window_pages, 128, 1, HEAD_DIM, generator=generator).mul_(0.6).bfloat16()
            wp, ws = quantize_mxfp8_cache(window)
            wi = torch.full((rows, WINDOW_KEYS), -1, dtype=torch.int32)
            for row in range(rows):
                kind = row % 6
                if kind == 1:
                    wi[row, :13] = torch.arange(13, dtype=torch.int32) + 127
                elif kind == 2:
                    wi[row] = (torch.arange(WINDOW_KEYS, dtype=torch.int32) + 221) % 256
                elif kind == 3:
                    wi[row] = torch.arange(WINDOW_KEYS, dtype=torch.int32) + 96
                elif kind == 4:
                    wi[row] = torch.randint(0, 256, (WINDOW_KEYS,), generator=generator, dtype=torch.int32)
                    wi[row, ::7] = -1
                    wi[row, 5:9] = 17
                elif kind == 5:
                    wi[row, :65] = torch.arange(65, dtype=torch.int32) + 64
            compressed = torch.randn(compressed_pages, 128, 1, HEAD_DIM, generator=generator).mul_(0.6).bfloat16()
            cp, cs = quantize_mxfp4_cache(compressed, 16, "bf16")
            ci = torch.randint(0, compressed_pages * 128, (rows, INDEX_TOPK), generator=generator, dtype=torch.int32)
            ci[:, ::19] = torch.iinfo(torch.int32).min
            ci[:, 5:9] = 17
            if case == "codes":
                wp = torch.linspace(-448, 448, 128 * HEAD_DIM).reshape(1, 128, 1, HEAD_DIM)
                wp = wp.to(torch.float8_e4m3fn)
                physical_rows = torch.arange(128).reshape(128, 1)
                packed = torch.arange(128 * (HEAD_DIM // 2)).reshape(128, HEAD_DIM // 2)
                cp = ((packed * 37 + physical_rows * 13) % 256).to(torch.uint8)
                cp = cp.reshape(1, 128, 1, HEAD_DIM // 2)
                scales = torch.tensor([0.0, 0.125, 1.0, 131072.0], dtype=torch.bfloat16)
                row_scale = (2.0 ** (physical_rows.remainder(5) - 2)).bfloat16()
                ws = scales.repeat(128 * (HEAD_DIM // 32) // 4).reshape(128, HEAD_DIM // 32) * row_scale
                cs = scales.repeat(128 * (HEAD_DIM // 16) // 4).reshape(128, HEAD_DIM // 16) * row_scale
                ws = ws.reshape(1, 128, 1, HEAD_DIM // 32)
                cs = cs.reshape(1, 128, 1, HEAD_DIM // 16)
                wi.fill_(-2)
                ci.fill_(torch.iinfo(torch.int32).min)
                for row in range(rows):
                    if compressed_keys and row % 2:
                        ci[row, 0] = row % 128
                    else:
                        wi[row, 0] = row % 128
                query.zero_()
            else:
                ci[(wi < 0).all(dim=-1)] = -1
            if active:
                wi[start] = -1
                ci[start] = -1
            query[:start] = float("nan")
            query[start + active:] = float("nan")
            values.update(query=query, window_indices=wi, window=pack_quantized_cache(wp, ws))
            values.update(compressed_indices=ci, compressed=pack_quantized_cache(cp, cs))
            if not compressed_keys:
                values["compressed_indices"].fill_(-1)
                values["compressed"].zero_()
            values["sink"] = torch.full((LOCAL_H,), -100.0) if case == "codes" else torch.linspace(-6.0, -1.0, LOCAL_H)
            values["output"] = torch.full_like(query, 7.25)
        return values[name].clone()

    shapes = [
        ("query", [rows, LOCAL_H * HEAD_DIM], torch.bfloat16),
        ("window_indices", [rows, WINDOW_KEYS], torch.int32),
        ("window", [window_pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        ("compressed_indices", [rows, INDEX_TOPK], torch.int32),
        ("compressed", [compressed_pages, 128, 1, COMPRESSED_CACHE_BYTES], torch.uint8),
    ]
    shapes += [("sink", [LOCAL_H], torch.float32), ("output", [rows, LOCAL_H * HEAD_DIM], torch.bfloat16)]
    specs = []
    for name, shape, dtype in shapes:
        specs.append(TensorSpec(name, shape, dtype, init_value=lambda name=name: initialize(name)))
    specs += [ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True)]
    specs += [ScalarSpec("start", torch.int32, start, compile_runtime=True)]
    specs += [ScalarSpec("compressed_keys", torch.int32, compressed_keys)]
    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="Validate packed gather sparse attention on Ascend950")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, help="active row count; defaults to boundary cases")
    parser.add_argument("--compressed-keys", type=int, choices=[0, INDEX_TOPK], help="defaults to both key counts")
    parser.add_argument("--case", choices=["normal", "codes", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--runtime-dir", type=str)
    args = parser.parse_args()
    if args.rows is not None and args.rows < 0:
        parser.error("--rows must be nonnegative")
    if args.runtime_dir and (args.rows is None or args.compressed_keys is None or args.case == "all"):
        parser.error("--runtime-dir requires --rows, --compressed-keys and one --case")
    torch.set_num_threads(8)
    keys = (0, INDEX_TOPK) if args.compressed_keys is None else (args.compressed_keys,)
    cases = []
    if args.case in ("normal", "all"):
        cases = [(n + 1, n, 0, "normal") for n in (0, 1, 3, 33, 64, 96, 128, 129)]
        cases += [(13, 7, 3, "normal"), (73, 72, 1, "normal")]
    if args.case in ("codes", "all"):
        cases.append((13, 7, 3, "codes"))
    if args.rows is not None:
        cases = [(args.rows + 1, args.rows, 1, case) for case in ("normal", "codes") if args.case in (case, "all")]
    records = []
    for compressed_keys in keys:
        for rows, active, start, case in cases:
            specs = build_sparse_attention_specs(rows, active, compressed_keys=compressed_keys, start=start, case=case)
            query_rows = slice(start, start + active)
            exact = case == "codes" or active <= 1
            compare = ratio_allclose(atol=0.0 if exact else 0.003, rtol=0.0 if exact else 0.03, max_error_ratio=0.0)
            result = run(
                l2_gather_sparse_attention,
                specs,
                golden_fn=lambda tensors: tensors["output"][:active].copy_(golden_gather_sparse_attention(
                    tensors["query"][query_rows], tensors["window_indices"][query_rows], tensors["window"],
                    tensors["compressed_indices"][query_rows], tensors["compressed"], tensors["sink"],
                    compressed_keys=compressed_keys,
                )),
                compare_fn={"output": compare},
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                compile_only=args.compile_only,
                save_data=args.save_data,
                runtime_dir=args.runtime_dir,
            )
            if not result.passed:
                raise RuntimeError(result.error)
            records.append(dict(compressed_keys=compressed_keys, active=active, start=start, case=case, passed=True))
            status = "SIM_VALIDATED" if args.execution_mode else "DEVICE_VALIDATED"
            if args.compile_only:
                status = "COMPILE_ONLY"
            detail = f"compressed_keys={compressed_keys} active={active} start={start} case={case}"
            print(status, detail, result.work_dir)
    _main_results = records
