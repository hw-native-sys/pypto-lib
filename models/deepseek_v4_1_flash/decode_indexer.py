# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

import pypto.language as pl

from hierarchical_sparse_indexer import (
    make_index_query_quantizer,
    materialize_index_topk as _materialize_index_topk,
)

from qkv_proj_rope import (
    make_bf16_projection_with_deps,
    make_rope_with_deps,
)
from rmsnorm import make_norm_with_deps
from config import (
    B_DYN,
    CMP_POSITIONS_DYN,
    FLASH,
    INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP,
    INDEX_POOL_BLOCKS_DYN,
    INDEX_POOL_POSITIONS_DYN,
    TABLE_DYN,
    T_DYN,
)


# model config
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim

_PAGE_SIZE = 128
_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP
_SCORE_SCALE = (INDEX_DIM * INDEX_H) ** -0.5

# tiling
_SCORE_N_TILE = 256
_SCORE_MACRO_N_TILE = 512
_PAIR_M_TILE = INDEX_H * 2


def make_fp4_publish(width, group, scale_dtype, cache_dim):
    """Build an FP4 cache-publish kernel for one scale dtype.

    The BF16 (compressed) and E8M0 (index) encodings store different scale
    dtypes, so the two variants must be separate traced functions: PyPTO traces
    both arms of a Python ``if`` on a closure value and rejects the resulting
    dtype-divergent assignment.  This factory only dispatches between them.
    """
    if scale_dtype == pl.FP8E8M0:
        return _make_fp4_publish_e8m0(width, group, cache_dim)
    return _make_fp4_publish_bf16(width, group, cache_dim)


def _make_fp4_publish_bf16(width, group, cache_dim):
    padded_width = max(width, 32 * group)

    @pl.jit.inline(auto_scope=False)
    def publish(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        slots: pl.Tensor[[T_DYN], pl.INT64],
        cache: pl.Tensor[[cache_dim, 128, 1, width // 2], pl.UINT8],
        scales: pl.Tensor[[cache_dim, 128, 1, width // group], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        kv_ready: pl.Scalar[pl.TASK_ID],
    ):
        cache_rows = pl.tensor.dim(cache, 0) * 128
        flat = pl.reshape(cache, [cache_rows, width // 2])
        flat_scale = pl.reshape(scales, [cache_rows, width // group])
        # Scale rows may share a cache line, so one worker publishes them in order.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="c1a_publish_fp4", deps=[cache_ready, kv_ready]
        ) as published:
            for t in pl.range(num_tokens):
                slot = pl.read(slots, [t])
                if slot >= 0:
                    physical = pl.cast(slot, pl.INDEX)
                    raw = pl.load(source, [t, 0], [1, padded_width], valid_shape=[1, width])
                    raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 1, padded_width)
                    values = pl.reshape(pl.cast(raw, pl.FP32), [padded_width // group, group])
                    temporary = pl.tile.create([padded_width // group, group], dtype=pl.FP32)
                    amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                    factors_raw = pl.maximum(pl.mul(amax, 1.0 / 6.0), 0.001953125)
                    stored_scale = pl.cast(factors_raw, pl.BF16, mode="rint")
                    factors = pl.cast(stored_scale, pl.FP32)
                    normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                    magnitude = pl.abs(normalized)
                    rounded = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.0)
                    select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                    predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=2.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=3.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=4.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=6.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    # Convert the exact RNE levels to E2M1 codes, then pack adjacent lanes.
                    codes = pl.add(pl.minimum(pl.mul(rounded, 2.0), 4.0),
                                   pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0))
                    sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                    codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, padded_width])
                    low = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32
                    )
                    high = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    packed = pl.add(low, pl.mul(high, 16))
                    signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                    payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                    payload = pl.set_validshape(payload_bytes, 1, width // 2)
                    encoded = pl.set_validshape(pl.reshape(stored_scale, [1, padded_width // group]),
                                                1, width // group)
                    pl.store(payload, [physical, 0], flat)
                    pl.store(encoded, [physical, 0], flat_scale)
        return published

    return publish


def _make_fp4_publish_e8m0(width, group, cache_dim):
    padded_width = max(width, 32 * group)

    @pl.jit.inline(auto_scope=False)
    def publish(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        slots: pl.Tensor[[T_DYN], pl.INT64],
        cache: pl.Tensor[[cache_dim, 128, 1, width // 2], pl.UINT8],
        scales: pl.Tensor[[cache_dim, 128, 1, width // group], pl.FP8E8M0],
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        kv_ready: pl.Scalar[pl.TASK_ID],
    ):
        cache_rows = pl.tensor.dim(cache, 0) * 128
        flat = pl.reshape(cache, [cache_rows, width // 2])
        flat_scale = pl.reshape(scales, [cache_rows, width // group])
        # Scale rows may share a cache line, so one worker publishes them in order.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="c1a_publish_fp4", deps=[cache_ready, kv_ready]
        ) as published:
            for t in pl.range(num_tokens):
                slot = pl.read(slots, [t])
                if slot >= 0:
                    physical = pl.cast(slot, pl.INDEX)
                    raw = pl.load(source, [t, 0], [1, padded_width], valid_shape=[1, width])
                    raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 1, padded_width)
                    values = pl.reshape(pl.cast(raw, pl.FP32), [padded_width // group, group])
                    temporary = pl.tile.create([padded_width // group, group], dtype=pl.FP32)
                    amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                    bounded = pl.maximum(amax, 7.052966104933725e-38)
                    bits = pl.reinterpret_view(pl.mul(bounded, 1.0 / 6.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    factors = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    # Raw codes span 0..255; folding to a signed byte would saturate.
                    stored_codes = exponent
                    normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                    magnitude = pl.abs(normalized)
                    rounded = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.0)
                    select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                    predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=0.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=1.5)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=2.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=3.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=4.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                    level = pl.tile.full([padded_width // group, group], dtype=pl.FP32, value=6.0)
                    rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                    # Convert the exact RNE levels to E2M1 codes, then pack adjacent lanes.
                    codes = pl.add(pl.minimum(pl.mul(rounded, 2.0), 4.0),
                                   pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0))
                    sign = pl.ands(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 28), 8)
                    codes = pl.reshape(pl.or_(pl.cast(codes, pl.INT32), sign), [1, padded_width])
                    low = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32
                    )
                    high = pl.tile.gather_mask(
                        codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
                    )
                    packed = pl.add(low, pl.mul(high, 16))
                    signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                    payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                    payload = pl.set_validshape(payload_bytes, 1, width // 2)
                    encoded_codes = pl.reinterpret_view(pl.cast(stored_codes, pl.UINT8), pl.FP8E8M0)
                    encoded = pl.set_validshape(pl.reshape(encoded_codes, [1, padded_width // group]),
                                                1, width // group)
                    pl.store(payload, [physical, 0], flat)
                    pl.store(encoded, [physical, 0], flat_scale)
        return published

    return publish


@pl.jit.inline
def _index_key_decode_lut():
    """Map packed FP4 nibbles to exact FP8 carrier bits."""
    codes = pl.tile.arange(0, [1, 16], dtype=pl.INT32)
    magnitude = pl.ands(codes, 7)
    values = pl.add(pl.shls(magnitude, 2), 48)
    values = pl.tile.select(pl.tile.cmps(magnitude, 0, cmp_type=0), 0, values)
    values = pl.tile.select(pl.tile.cmps(magnitude, 1, cmp_type=0), 48, values)
    values = pl.or_(values, pl.shls(pl.ands(codes, 8), 4))
    return pl.cast(values, pl.INT16)


def _make_page_index_scores(workers, skip_short=False):
    """Share decoded K/scale locally; use M64 for two queries of one request."""
    tile_n = _PAGE_SIZE
    packed_width = INDEX_DIM // 2
    scale_count = INDEX_DIM // INDEX_CACHE_GROUP

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * 4], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        cache_rows = pl.tensor.dim(index_cache, 0) * 128
        cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scale_flat = pl.reshape(index_cache_scale, [1, cache_rows * scale_count])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
        tiles = (width + tile_n - 1) // tile_n
        work_items = ((num_tokens + 1) // 2) * tiles

        with pl.spmd(
            workers,
            name_hint="c1a_index_score_page_mxfp8",
            deps=[cache_ready, query_ready, weights_ready],
        ) as score_tid:
            lut = _index_key_decode_lut()
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, work_items, workers):
                first_token = (work // tiles) * 2
                panel = work % tiles
                logical_begin = panel * tile_n
                output_count = pl.min(tile_n, width - logical_begin)
                first_request = pl.read(request_ids, [first_token])
                second_token = pl.min(first_token + 1, num_tokens - 1)
                second_request = pl.read(request_ids, [second_token])
                for anchor_offset in pl.range(2):
                    anchor = first_token + anchor_offset
                    if anchor < num_tokens:
                        if anchor_offset == 0 or first_request != second_request:
                            anchor_request = pl.read(request_ids, [anchor])
                            maximum_visible = pl.min(pl.read(lengths, [anchor]), width)
                            if skip_short:
                                if maximum_visible <= INDEX_TOPK:
                                    maximum_visible = 0
                            if first_request == second_request:
                                second_visible = pl.min(pl.read(lengths, [second_token]), width)
                                if skip_short:
                                    if second_visible <= INDEX_TOPK:
                                        second_visible = 0
                                maximum_visible = pl.max(maximum_visible, second_visible)
                            if logical_begin < maximum_visible:
                                physical_page_i32 = pl.read(block_table,
                                    [pl.cast(anchor_request, pl.INDEX), logical_begin // 128])
                                if physical_page_i32 >= 0:
                                    physical_page = pl.cast(physical_page_i32, pl.INDEX)
                                    physical_row = physical_page * 128 + logical_begin % 128
                                    packed = pl.load(cache_flat, [physical_row, 0], [tile_n, packed_width])
                                    byte_values = pl.ands(pl.cast(pl.reinterpret_view(packed, pl.INT8), pl.INT16), 255)
                                    low = pl.ands(byte_values, 15)
                                    high = pl.ands(pl.shrs(byte_values, 4), 15)
                                    lut_tmp = pl.create_tile([1, 32], dtype=pl.UINT8)
                                    low_bytes = pl.tile.gather(lut, low, lut_tmp)
                                    high_bytes = pl.tile.gather(lut, high, lut_tmp)
                                    interleaved_words = pl.or_(low_bytes, pl.shls(high_bytes, 8))
                                    keys = pl.reinterpret_view(
                                        pl.reinterpret_view(interleaved_words, pl.INT8, shape=[tile_n, INDEX_DIM]),
                                        pl.FP8E4M3FN,
                                    )
                                    raw_scale = pl.reinterpret_view(
                                        pl.load(scale_flat, [0, physical_row * scale_count],
                                                [1, tile_n * scale_count]), pl.UINT8
                                    )
                                    scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                    key_scale_zz = pl.tmov_x2zz(
                                        raw_scale, scale_tmp, group_axis=1,
                                        dst_rows=tile_n, dst_cols=scale_count,
                                    )
                                    key_scale_nn = pl.tile.transpose_view(pl.reinterpret_view(key_scale_zz, pl.FP8E8M0))
                                    keys_kn = pl.reinterpret_view(
                                        pl.tile.transpose(pl.reinterpret_view(keys, pl.INT8), 0, 1), pl.FP8E4M3FN,
                                    )
                                    keys_mat = pl.tile.move(keys_kn, target_memory=pl.MemorySpace.Mat,
                                                            blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)
                                    keys_right = pl.tile.move(keys_mat, target_memory=pl.MemorySpace.Right)
                                    key_scale_mat = pl.tile.move(key_scale_nn, target_memory=pl.MemorySpace.Mat,
                                                                 blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major)
                                    key_scale_right = pl.tile.move(key_scale_mat, target_memory=pl.MemorySpace.RightScale)
                                    share_pair = pl.cast(0, pl.INT32)
                                    if second_token > first_token and first_request == second_request:
                                        share_pair = pl.cast(1, pl.INT32)
                                    if skip_short:
                                        if pl.read(lengths, [first_token]) <= INDEX_TOPK or pl.read(lengths, [second_token]) <= INDEX_TOPK:
                                            share_pair = pl.cast(0, pl.INT32)
                                    if share_pair != 0:
                                        query_pair = pl.load(query_flat, [first_token * INDEX_H, 0],
                                                             [2 * INDEX_H, INDEX_DIM], target_memory=pl.MemorySpace.Vec)
                                        pair_scale_raw = pl.reinterpret_view(
                                            pl.load(query_scale, [first_token, 0], [2, INDEX_H * scale_count]), pl.UINT8)
                                        pair_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                        pair_scale_zz = pl.reinterpret_view(pl.tmov_x2zz(
                                            pl.reshape(pair_scale_raw, [1, 2 * INDEX_H * scale_count]), pair_scale_tmp,
                                            group_axis=1, dst_rows=64, dst_cols=scale_count), pl.FP8E8M0)
                                        pair_dot = pl.matmul_mx(query_pair, pair_scale_zz, keys_right, key_scale_right)
                                        pair_dot = pl.cast(pair_dot, pl.BF16, mode="rint")
                                        for pair_query_offset in pl.unroll(2):
                                            pair_token = first_token + pair_query_offset
                                            pair_negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                            pair_masked = pl.reinterpret_view(pair_negative_bits, pl.FP32)
                                            pair_visible = pl.min(pl.read(lengths, [pair_token]), width)
                                            if logical_begin < pair_visible:
                                                pair_valid_count = pl.min(tile_n, pair_visible - logical_begin)
                                                pair_head_dot = pl.tile.slice(pair_dot, [INDEX_H, tile_n], [pair_query_offset * INDEX_H, 0])
                                                pair_relu = pl.maximum(pl.cast(pair_head_dot, pl.FP32), 0.0)
                                                pair_head_weight = pl.load(weights, [pair_token, 0], [1, INDEX_H])
                                                pair_head_weight = pl.mul(pl.cast(pair_head_weight, pl.FP32), ((INDEX_DIM * INDEX_H) ** -0.5))
                                                pair_rounded_weight = pl.cast(
                                                    pl.cast(pair_head_weight, pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_weighted = pl.row_expand_mul(
                                                    pair_relu, pl.reshape(pair_rounded_weight, [INDEX_H, 1])
                                                )
                                                pair_weighted = pl.cast(
                                                    pl.cast(pair_weighted, pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_total = pl.cast(
                                                    pl.cast(pl.col_sum(pair_weighted), pl.BF16, mode="rint"), pl.FP32
                                                )
                                                pair_valid_count_i32 = pl.cast(pair_valid_count, pl.INT32)
                                                pair_lanes_i32 = pl.tile.arange(0, [1, tile_n], dtype=pl.INT32)
                                                pair_predicate = pl.tile.cmps(
                                                    pair_lanes_i32, pair_valid_count_i32, cmp_type=2
                                                )
                                                pair_masked = pl.tile.select(pair_predicate, pair_total, pair_masked)
                                            pl.store(
                                                pl.set_validshape(pair_masked, 1, output_count),
                                                [pair_token, logical_begin],
                                                output,
                                            )
                                    else:
                                        for query_offset in pl.range(2):
                                            token = first_token + query_offset
                                            if token < num_tokens:
                                                query_request = pl.read(request_ids, [token])
                                                if query_request == anchor_request:
                                                    negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                                    masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                                    visible = pl.min(pl.read(lengths, [token]), width)
                                                    if skip_short:
                                                        if visible <= INDEX_TOPK:
                                                            visible = 0
                                                    if logical_begin < visible:
                                                        valid_count = pl.min(tile_n, visible - logical_begin)
                                                        query_panel = pl.load(
                                                            query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM],
                                                            target_memory=pl.MemorySpace.Vec,
                                                        )
                                                        raw_query_scale = pl.reinterpret_view(
                                                            pl.load(query_scale, [token, 0], [1, INDEX_H * scale_count]),
                                                            pl.UINT8,
                                                        )
                                                        query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                                                        query_scale_zz = pl.reinterpret_view(
                                                            pl.tmov_x2zz(raw_query_scale, query_scale_tmp, group_axis=1,
                                                                         dst_rows=INDEX_H, dst_cols=scale_count),
                                                            pl.FP8E8M0,
                                                        )
                                                        dot = pl.matmul_mx(query_panel, query_scale_zz,
                                                                               keys_right, key_scale_right)
                                                        dot = pl.cast(dot, pl.BF16, mode="rint")
                                                        relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                                                        head_weight = pl.load(weights, [token, 0], [1, INDEX_H])
                                                        head_weight = pl.mul(pl.cast(head_weight, pl.FP32), ((INDEX_DIM * INDEX_H) ** -0.5))
                                                        rounded_weight = pl.cast(
                                                            pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        weighted = pl.row_expand_mul(
                                                            relu, pl.reshape(rounded_weight, [INDEX_H, 1])
                                                        )
                                                        weighted = pl.cast(
                                                            pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        total = pl.cast(
                                                            pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"), pl.FP32
                                                        )
                                                        valid_count_i32 = pl.cast(valid_count, pl.INT32)
                                                        lanes_i32 = pl.tile.arange(0, [1, tile_n], dtype=pl.INT32)
                                                        predicate = pl.tile.cmps(lanes_i32, valid_count_i32, cmp_type=2)
                                                        masked = pl.tile.select(predicate, total, masked)
                                                    pl.store(
                                                        pl.set_validshape(masked, 1, output_count),
                                                        [token, logical_begin],
                                                        output,
                                                    )
                                else:
                                    for query_offset in pl.range(2):
                                        token = first_token + query_offset
                                        if token < num_tokens:
                                            query_request = pl.read(request_ids, [token])
                                            if query_request == anchor_request:
                                                negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                                masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                                pl.store(
                                                    pl.set_validshape(masked, 1, output_count),
                                                    [token, logical_begin],
                                                    output,
                                                )
                            else:
                                for query_offset in pl.range(2):
                                    token = first_token + query_offset
                                    if token < num_tokens:
                                        query_request = pl.read(request_ids, [token])
                                        if query_request == anchor_request:
                                            negative_bits = pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608)
                                            masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                            pl.store(
                                                pl.set_validshape(masked, 1, output_count),
                                                [token, logical_begin],
                                                output,
                                            )
        return score_tid

    return score


def _make_paired_block_index_scores(skip_short=False):
    """Share Q/Qscale across two independent N256 dots over four mapped pages."""
    packed_width = INDEX_DIM // 2

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, _PAGE_SIZE, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, _PAGE_SIZE, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        cache_rows = pl.tensor.dim(index_cache, 0) * _PAGE_SIZE
        cache_flat = pl.reshape(index_cache, [cache_rows, packed_width])
        scale_flat = pl.reshape(index_cache_scale, [1, cache_rows * _SCALE_COUNT])
        query_rows = pl.tensor.dim(query, 0) * INDEX_H
        query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
        macro_tiles = (width + _SCORE_MACRO_N_TILE - 1) // _SCORE_MACRO_N_TILE
        work_items = (num_tokens // 2) * macro_tiles

        with pl.spmd(
            32,
            name_hint="c1a_index_score_paired_block_mxfp8",
            deps=[cache_ready, query_ready, weights_ready],
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            lut = _index_key_decode_lut()

            for work in pl.range(worker, work_items, 32):
                first_token = (work // macro_tiles) * 2
                macro_index = work % macro_tiles
                logical_begin = macro_index * _SCORE_MACRO_N_TILE
                second_token = first_token + 1
                first_request = pl.read(request_ids, [first_token])

                first_visible = pl.min(pl.read(lengths, [first_token]), width)
                second_visible = pl.min(pl.read(lengths, [second_token]), width)
                if skip_short:
                    if first_visible <= INDEX_TOPK:
                        first_visible = 0
                    if second_visible <= INDEX_TOPK:
                        second_visible = 0
                maximum_visible = pl.max(first_visible, second_visible)
                query_pair = pl.load(
                    query_flat,
                    [first_token * INDEX_H, 0],
                    [_PAIR_M_TILE, INDEX_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                pair_scale_raw = pl.reinterpret_view(
                    pl.load(
                        query_scale,
                        [first_token, 0],
                        [2, INDEX_H * _SCALE_COUNT],
                        target_memory=pl.MemorySpace.Vec,
                    ),
                    pl.UINT8,
                )
                pair_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                pair_scale_zz = pl.reinterpret_view(
                    pl.tmov_x2zz(
                        pl.reshape(pair_scale_raw, [1, _PAIR_M_TILE * _SCALE_COUNT]),
                        pair_scale_tmp,
                        group_axis=1,
                        dst_rows=_PAIR_M_TILE,
                        dst_cols=_SCALE_COUNT,
                    ),
                    pl.FP8E8M0,
                )
                query_mat = pl.tile.move(
                    query_pair,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
                pair_scale_mat = pl.tile.move(
                    pair_scale_zz,
                    target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.row_major,
                )
                pair_scale_left = pl.tile.move(pair_scale_mat, target_memory=pl.MemorySpace.LeftScale)

                for half_index in pl.unroll(2):
                    half_begin = logical_begin + half_index * _SCORE_N_TILE

                    if half_begin < maximum_visible:
                        packed = pl.tile.full([_SCORE_N_TILE, packed_width], dtype=pl.INT8, value=0)
                        raw_scale = pl.tile.full([1, _SCORE_N_TILE * _SCALE_COUNT], dtype=pl.INT8, value=127)
                        for page_offset in pl.unroll(2):
                            page_begin = half_begin + page_offset * _PAGE_SIZE
                            if page_begin < maximum_visible:
                                page_id = pl.read(
                                    block_table,
                                    [pl.cast(first_request, pl.INDEX), page_begin // _PAGE_SIZE],
                                )
                                if page_id >= 0:
                                    page_row = pl.cast(page_id, pl.INDEX) * _PAGE_SIZE
                                    page_packed = pl.reinterpret_view(
                                        pl.load(
                                            cache_flat,
                                            [page_row, 0],
                                            [_PAGE_SIZE, packed_width],
                                            target_memory=pl.MemorySpace.Vec,
                                        ),
                                        pl.INT8,
                                    )
                                    packed = pl.tile.assemble(packed, page_packed, [page_offset * _PAGE_SIZE, 0])
                                    page_scale = pl.reinterpret_view(
                                        pl.reinterpret_view(
                                            pl.load(
                                                scale_flat,
                                                [0, page_row * _SCALE_COUNT],
                                                [1, _PAGE_SIZE * _SCALE_COUNT],
                                                target_memory=pl.MemorySpace.Vec,
                                            ),
                                            pl.UINT8,
                                        ),
                                        pl.INT8,
                                    )
                                    raw_scale = pl.tile.assemble(
                                        raw_scale,
                                        page_scale,
                                        [0, page_offset * _PAGE_SIZE * _SCALE_COUNT],
                                    )

                        byte_values = pl.ands(pl.cast(packed, pl.INT16), 255)
                        low = pl.ands(byte_values, 15)
                        high = pl.ands(pl.shrs(byte_values, 4), 15)
                        lut_tmp = pl.create_tile([1, 32], dtype=pl.UINT8)
                        low_bytes = pl.tile.gather(lut, low, lut_tmp)
                        high_bytes = pl.tile.gather(lut, high, lut_tmp)
                        interleaved_words = pl.or_(low_bytes, pl.shls(high_bytes, 8))
                        keys = pl.reinterpret_view(
                            pl.reinterpret_view(interleaved_words, pl.INT8, shape=[_SCORE_N_TILE, INDEX_DIM]),
                            pl.FP8E4M3FN,
                        )
                        key_scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
                        key_scale_zz = pl.reinterpret_view(
                            pl.tmov_x2zz(
                                pl.reinterpret_view(raw_scale, pl.UINT8),
                                key_scale_tmp,
                                group_axis=1,
                                dst_rows=_SCORE_N_TILE,
                                dst_cols=_SCALE_COUNT,
                            ),
                            pl.FP8E8M0,
                        )
                        key_scale_nn = pl.tile.transpose_view(key_scale_zz)
                        keys_kn = pl.reinterpret_view(
                            pl.tile.transpose(pl.reinterpret_view(keys, pl.INT8), 0, 1),
                            pl.FP8E4M3FN,
                        )
                        dot_bf16 = pl.cast(
                            pl.matmul_mx(query_left, pair_scale_left, keys_kn, key_scale_nn),
                            pl.BF16,
                            mode="rint",
                        )
                        for panel_index in pl.unroll(2):
                            output_begin = half_begin + panel_index * _PAGE_SIZE
                            if output_begin < width:
                                output_count = pl.min(_PAGE_SIZE, width - output_begin)
                                for pair_offset in pl.unroll(2):
                                    token = first_token + pair_offset
                                    visible = pl.min(pl.read(lengths, [token]), width)
                                    negative_bits = pl.tile.full([1, _PAGE_SIZE], dtype=pl.INT32, value=-8388608)
                                    masked = pl.reinterpret_view(negative_bits, pl.FP32)
                                    if output_begin < visible:
                                        valid_count = pl.min(_PAGE_SIZE, visible - output_begin)
                                        page_id = pl.read(
                                            block_table,
                                            [
                                                pl.cast(first_request, pl.INDEX),
                                                output_begin // _PAGE_SIZE,
                                            ],
                                        )
                                        if page_id >= 0:
                                            head_dot = pl.tile.slice(
                                                dot_bf16,
                                                [INDEX_H, _PAGE_SIZE],
                                                [pair_offset * INDEX_H, panel_index * _PAGE_SIZE],
                                            )
                                            relu = pl.maximum(pl.cast(head_dot, pl.FP32), 0.0)
                                            head_weight = pl.load(
                                                weights,
                                                [token, 0],
                                                [1, INDEX_H],
                                                target_memory=pl.MemorySpace.Vec,
                                            )
                                            head_weight = pl.mul(pl.cast(head_weight, pl.FP32), _SCORE_SCALE)
                                            rounded_weight = pl.cast(
                                                pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32
                                            )
                                            weighted = pl.row_expand_mul(relu, pl.reshape(rounded_weight, [INDEX_H, 1]))
                                            weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                                            total = pl.cast(
                                                pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"),
                                                pl.FP32,
                                            )
                                            lanes = pl.tile.arange(0, [1, _PAGE_SIZE], dtype=pl.INT32)
                                            in_range = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                                            masked = pl.tile.select(in_range, total, masked)
                                    pl.store(pl.set_validshape(masked, 1, output_count), [token, output_begin], output)
                    else:
                        for panel_index in pl.unroll(2):
                            output_begin = half_begin + panel_index * _PAGE_SIZE
                            if output_begin < width:
                                output_count = pl.min(_PAGE_SIZE, width - output_begin)
                                for pair_offset in pl.unroll(2):
                                    token = first_token + pair_offset
                                    negative_bits = pl.tile.full([1, _PAGE_SIZE], dtype=pl.INT32, value=-8388608)
                                    pl.store(
                                        pl.set_validshape(pl.reinterpret_view(negative_bits, pl.FP32), 1, output_count),
                                        [token, output_begin],
                                        output,
                                    )
        return score_tid

    return score


def make_index_scores(skip_short=False):
    """Route complete long query pairs to N512 macros and other queries to N128 pages."""
    general = _make_page_index_scores(workers=32, skip_short=skip_short)
    general_small = _make_page_index_scores(workers=16, skip_short=skip_short)
    paired_blocks = _make_paired_block_index_scores(skip_short=skip_short)

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * 4], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        cache_ready: pl.Scalar[pl.TASK_ID],
        query_ready: pl.Scalar[pl.TASK_ID],
        weights_ready: pl.Scalar[pl.TASK_ID],
    ):
        width = pl.tensor.dim(output, 1)
        panels = (width + _PAGE_SIZE - 1) // _PAGE_SIZE
        paired_panels = ((num_tokens + 1) // 2) * panels
        complete_pairs = 0
        if width >= 4096:
            if num_tokens % 2 == 0:
                complete_pairs = 1
                for token in pl.range(0, num_tokens, 2):
                    first_request = pl.read(request_ids, [token])
                    second_request = pl.read(request_ids, [token + 1])
                    if first_request != second_request:
                        complete_pairs = 0
                    if skip_short:
                        first_short = pl.cast(0, pl.INT32)
                        second_short = pl.cast(0, pl.INT32)
                        if pl.read(lengths, [token]) <= INDEX_TOPK:
                            first_short = pl.cast(1, pl.INT32)
                        if pl.read(lengths, [token + 1]) <= INDEX_TOPK:
                            second_short = pl.cast(1, pl.INT32)
                        if first_short != second_short:
                            complete_pairs = 0
        long_rows = pl.cast(num_tokens, pl.INT32)
        if skip_short:
            long_rows = pl.cast(0, pl.INT32)
            for token in pl.range(num_tokens):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    long_rows = long_rows + pl.cast(1, pl.INT32)
        if long_rows == 0:
            result = pl.system.task_dummy(deps=[cache_ready, query_ready, weights_ready])
        elif complete_pairs != 0:
            result = paired_blocks(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output,
                num_tokens, cache_ready, query_ready, weights_ready,
            )
        elif paired_panels <= 16:
            result = general_small(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output,
                num_tokens, cache_ready, query_ready, weights_ready,
            )
        else:
            result = general(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output,
                num_tokens, cache_ready, query_ready, weights_ready,
            )
        return result

    return score


score_full = make_index_scores()
score_c2a_full = make_index_scores(skip_short=True)


def _make_pool_index_scores(tile_n, panels_per_task, workers):
    """Score candidate blocks locally with the same FP8/MX arithmetic as Full."""
    blocks_per_panel = tile_n // 8
    blocks_per_task = blocks_per_panel * panels_per_task
    prefix_row = 0 if tile_n == 128 else panels_per_task

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        lengths: pl.Tensor[[T_DYN], pl.INT32],
        pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
        pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
        prefix: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        capacity = pl.tensor.dim(pool_block_ids, 1)
        cache_pages = pl.tensor.dim(index_cache, 0)
        cache_rows = cache_pages * 128
        cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scale_flat = pl.reshape(index_cache_scale, [cache_rows // 8, 32])
        query_flat = pl.reshape(query, [pl.tensor.dim(query, 0) * INDEX_H, INDEX_DIM])
        table_width = pl.tensor.dim(block_table, 1)
        request_count = pl.tensor.dim(block_table, 0)
        with pl.spmd(
            workers, name_hint="decode_index_pool_mxfp8", deps=[ready],
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ) as score_tid:
            worker = pl.tile.get_block_idx()
            lut = _index_key_decode_lut()
            work_items = pl.read(prefix, [prefix_row, num_tokens])
            for work in pl.range(worker, work_items, workers):
                lower = pl.cast(0, pl.INDEX)
                upper = pl.cast(num_tokens, pl.INDEX)
                # Bound the INT32 token-prefix search to 31 steps.
                for _ in pl.range(32):
                    if lower + 1 < upper:
                        middle = (lower + upper) // 2
                        if pl.read(prefix, [prefix_row, middle]) <= work:
                            lower = middle
                        else:
                            upper = middle
                token = lower
                macro = work - pl.read(prefix, [prefix_row, token])
                first_slot = macro * blocks_per_task
                count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
                request = pl.read(request_ids, [token])
                visible = pl.max(0, pl.read(lengths, [token]))
                query_panel = pl.load(query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM])
                query_scale_raw = pl.reinterpret_view(
                    pl.load(query_scale, [token, 0], [1, INDEX_H * _SCALE_COUNT]), pl.UINT8
                )
                query_scale_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                query_scale_zz = pl.reinterpret_view(
                    pl.tmov_x2zz(query_scale_raw, query_scale_tmp, group_axis=1,
                                 dst_rows=INDEX_H, dst_cols=_SCALE_COUNT), pl.FP8E8M0
                )
                query_mat = pl.tile.move(
                    query_panel, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                )
                query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
                query_scale_mat = pl.tile.move(
                    query_scale_zz, target_memory=pl.MemorySpace.Mat,
                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                )
                query_scale_left = pl.tile.move(query_scale_mat, target_memory=pl.MemorySpace.LeftScale)
                for panel in pl.unroll(panels_per_task):
                    slot_begin = first_slot + panel * blocks_per_panel
                    if slot_begin < count:
                        packed = pl.tile.full([tile_n, INDEX_DIM // 2], dtype=pl.INT8, value=0)
                        raw_scale = pl.tile.full([1, tile_n * _SCALE_COUNT], dtype=pl.INT8, value=127)
                        valid_lanes = pl.tile.full([1, tile_n], dtype=pl.INT32, value=0)
                        for block in pl.range(blocks_per_panel):
                            slot = slot_begin + block
                            if slot < count:
                                logical_block = pl.read(pool_block_ids, [token, slot])
                                if logical_block >= 0 and logical_block < (visible + 7) // 8:
                                    logical_page = logical_block // 16
                                    if request >= 0 and request < request_count and logical_page < table_width:
                                        page = pl.read(block_table, [pl.cast(request, pl.INDEX), logical_page])
                                        if page >= 0 and page < cache_pages:
                                            physical_row = pl.cast(page, pl.INDEX) * 128 + (logical_block % 16) * 8
                                            payload = pl.reinterpret_view(
                                                pl.load(cache_flat, [physical_row, 0], [8, INDEX_DIM // 2]), pl.INT8
                                            )
                                            packed = pl.tile.assemble(packed, payload, [block * 8, 0])
                                            scales = pl.reinterpret_view(
                                                pl.reinterpret_view(
                                                    pl.load(scale_flat, [physical_row // 8, 0], [1, 32]), pl.UINT8
                                                ), pl.INT8
                                            )
                                            raw_scale = pl.tile.assemble(raw_scale, scales, [0, block * 32])
                                            lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                            remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                            ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                            valid = pl.tile.select(pl.tile.cmps(lanes, remaining, cmp_type=2), ones, 0)
                                            valid_lanes = pl.tile.assemble(valid_lanes, valid, [0, block * 8])
                        # Missing-page lanes remain zero-filled and are masked after a real tile transfer.
                        byte_values = pl.ands(pl.cast(packed, pl.INT16), 255)
                        low = pl.ands(byte_values, 15)
                        high = pl.ands(pl.shrs(byte_values, 4), 15)
                        lut_tmp = pl.create_tile([1, 32], dtype=pl.UINT8)
                        low_bytes = pl.tile.gather(lut, low, lut_tmp)
                        high_bytes = pl.tile.gather(lut, high, lut_tmp)
                        words = pl.or_(low_bytes, pl.shls(high_bytes, 8))
                        keys = pl.reinterpret_view(
                            pl.reinterpret_view(words, pl.INT8, shape=[tile_n, INDEX_DIM]), pl.FP8E4M3FN
                        )
                        scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
                        scale_zz = pl.reinterpret_view(
                            pl.tmov_x2zz(pl.reinterpret_view(raw_scale, pl.UINT8), scale_tmp,
                                         group_axis=1, dst_rows=tile_n, dst_cols=_SCALE_COUNT), pl.FP8E8M0
                        )
                        scale_nn = pl.tile.transpose_view(scale_zz)
                        keys_kn = pl.reinterpret_view(
                            pl.tile.transpose(pl.reinterpret_view(keys, pl.INT8), 0, 1), pl.FP8E4M3FN
                        )
                        # Initialize an independent dot for each history interval.
                        dot = pl.cast(pl.matmul_mx(query_left, query_scale_left, keys_kn, scale_nn),
                                      pl.BF16, mode="rint")
                        relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
                        head_weight = pl.mul(pl.cast(pl.load(weights, [token, 0], [1, INDEX_H]), pl.FP32), _SCORE_SCALE)
                        rounded_weight = pl.cast(pl.cast(head_weight, pl.BF16, mode="rint"), pl.FP32)
                        weighted = pl.row_expand_mul(relu, pl.reshape(rounded_weight, [INDEX_H, 1]))
                        weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                        total = pl.cast(pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint"), pl.FP32)
                        invalid = pl.reinterpret_view(
                            pl.tile.full([1, tile_n], dtype=pl.INT32, value=-8388608), pl.FP32
                        )
                        masked = pl.tile.select(pl.tile.cmps(valid_lanes, 1, cmp_type=0), total, invalid)
                        valid_output = pl.min(tile_n, (capacity - slot_begin) * 8)
                        pl.store(pl.set_validshape(masked, 1, valid_output), [token, slot_begin * 8], output)
        return score_tid

    return score


_pool_page = _make_pool_index_scores(128, 1, 32)
_pool_page_small = _make_pool_index_scores(128, 1, 16)
_pool_block = _make_pool_index_scores(256, 1, 32)
_pool_macro = _make_pool_index_scores(256, 2, 32)


@pl.jit.inline(auto_scope=False)
def score_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Plan candidate work once, then run only the selected local FP8/MX leaf."""
    tokens = pl.tensor.dim(query, 0)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(output, 1)
    prefix = pl.create_tensor([3, tokens + 1], dtype=pl.INT32)
    route = pl.create_tensor([1, 8], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_index_pool_plan", deps=[pool_ready]) as plan_tid:
        small = pl.cast(0, pl.INT32)
        block = pl.cast(0, pl.INT32)
        macro = pl.cast(0, pl.INT32)
        for kind in pl.unroll(3):
            pl.write(prefix, [kind, 0], pl.cast(0, pl.INT32))
        for token in pl.range(num_tokens):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            small = pl.cast(small + (count + 15) // 16, pl.INT32)
            block = pl.cast(block + (count + 31) // 32, pl.INT32)
            macro = pl.cast(macro + (count + 63) // 64, pl.INT32)
            pl.write(prefix, [0, token + 1], pl.cast(small, pl.INT32))
            pl.write(prefix, [1, token + 1], pl.cast(block, pl.INT32))
            pl.write(prefix, [2, token + 1], pl.cast(macro, pl.INT32))
        values = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        pl.tile.write(values, [0, 0], pl.cast(small, pl.INT32))
        pl.tile.write(values, [0, 1], pl.cast(block, pl.INT32))
        pl.tile.write(values, [0, 2], pl.cast(macro, pl.INT32))
        pl.store(values, [0, 0], route)
    output_panels = (width + 127) // 128
    with pl.spmd(32, name_hint="decode_index_pool_clear", deps=[pool_ready]) as clear_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * output_panels, 32):
            token = work // output_panels
            begin = (work % output_panels) * 128
            invalid = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
            pl.store(pl.set_validshape(invalid, 1, pl.min(128, width - begin)), [token, begin], output)
    ready = pl.system.task_dummy(deps=[cache_ready, query_ready, weights_ready, plan_tid, clear_tid])
    small = pl.read(route, [0, 0])
    block = pl.read(route, [0, 1])
    macro = pl.read(route, [0, 2])
    wide = 0
    if 2 * block == small:
        if macro >= 32:
            wide = 2
        elif block >= 32:
            wide = 1
    if small == 0:
        result = ready
    elif wide == 2:
        result = _pool_macro(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                             request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    elif wide == 1:
        result = _pool_block(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                             request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    elif small <= 16:
        result = _pool_page_small(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                                  request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    else:
        result = _pool_page(query, query_scale, weights, index_cache, index_cache_scale, block_table,
                            request_ids, lengths, pool_block_ids, pool_block_counts, prefix, output, num_tokens, ready)
    return result


quantize_index_query = make_index_query_quantizer()


_TOPK_PAGE = 128
_TOPK_CACHED_PAGES = (FLASH.max_position_embeddings + 127) // 128
_TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
_TOPK_LEAF_TILE = 2048
_TOPK_SORT_FLOOR = -3.4028234663852886e38
_TOPK_INVALID_SCORE_BITS = -8388608


def _make_index_topk_pairs(query_rows=False):
    """Keep one sort/merge algorithm for flat or query-major pair storage."""
    token_stride = 1 if query_rows else 0
    slot_row_stride = 0 if query_rows else 1
    slot_column_stride = _TOPK_PAIR_WIDTH if query_rows else 0

    @pl.jit.inline
    def sort_leaf(
        scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
        pair_arena: pl.Tensor,
        token: pl.Scalar[pl.INDEX],
        logical_begin: pl.Scalar[pl.INDEX],
        valid_count: pl.Scalar[pl.INDEX],
        output_slot: pl.Scalar[pl.INDEX],
    ):
        raw = pl.load(scores, [token, logical_begin], [1, _TOPK_LEAF_TILE], valid_shape=[1, valid_count])
        padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
        values = pl.maximum(padded, pl.tile.full([1, _TOPK_LEAF_TILE], dtype=pl.FP32, value=_TOPK_SORT_FLOOR))
        indices = pl.add(pl.tile.arange(0, [1, _TOPK_LEAF_TILE], dtype=pl.INT32), pl.cast(logical_begin, pl.INT32))
        pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
        pairs = pl.tile.mrgsort(pairs, block_len=64)
        pairs = pl.tile.mrgsort(pairs, block_len=256)
        pairs = pl.tile.mrgsort(pairs, block_len=1024)
        pl.store(
            pl.tile.slice(pairs, [1, _TOPK_PAIR_WIDTH], [0, 0]),
            [token * token_stride + output_slot * slot_row_stride, output_slot * slot_column_stride], pair_arena,
        )

    @pl.jit.inline
    def merge_rows(
        pair_arena: pl.Tensor,
        left_slot: pl.Scalar[pl.INDEX],
        right_slot: pl.Scalar[pl.INDEX],
        output_slot: pl.Scalar[pl.INDEX],
        token: pl.Scalar[pl.INDEX],
    ):
        row = token * token_stride
        left = pl.load(
            pair_arena, [row + left_slot * slot_row_stride, left_slot * slot_column_stride], [1, _TOPK_PAIR_WIDTH],
        )
        right = pl.load(
            pair_arena, [row + right_slot * slot_row_stride, right_slot * slot_column_stride], [1, _TOPK_PAIR_WIDTH],
        )
        merge_tmp = pl.tile.create([1, 2 * _TOPK_PAIR_WIDTH], dtype=pl.FP32)
        merged = pl.tile.mrgsort(left, right, tmp=merge_tmp)
        pl.store(
            pl.tile.slice(merged, [1, _TOPK_PAIR_WIDTH], [0, 0]),
            [row + output_slot * slot_row_stride, output_slot * slot_column_stride], pair_arena,
        )

    return sort_leaf, merge_rows


_sort_index_topk_leaf, _merge_index_topk_rows = _make_index_topk_pairs()
_sort_c2a_topk_leaf, _merge_c2a_topk_rows = _make_index_topk_pairs(query_rows=True)




@pl.jit.inline(auto_scope=False)
def select_index_topk(
    index_scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Select exact Top-512 indices from dynamic C1A score rows."""
    width = pl.tensor.dim(index_scores, 1)
    tokens = pl.tensor.dim(index_scores, 0)
    leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
    arena_rows = tokens * leaf_count
    pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(32, name_hint="c1a_index_topk_short", deps=[scores_ready]) as short_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width <= INDEX_TOPK:
                invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                _materialize_index_topk(
                    index_scores, block_table, request_ids, output,
                    token, width, invalid_score, pl.cast(0, pl.INDEX),
                )
    total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
    with pl.spmd(32, name_hint="c1a_index_topk_leaf", deps=[scores_ready]) as leaf_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, total_leaves, 32):
            if width > INDEX_TOPK:
                token = work // leaf_count
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                if valid_count > 0:
                    output_slot = token * leaf_count + leaf
                    _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
    with pl.spmd(32, name_hint="c1a_index_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width > INDEX_TOPK:
                base = token * leaf_count
                count = leaf_count
                # INT32 logical positions require at most 21 binary merge levels.
                for _ in pl.range(21):
                    if count > 1:
                        output_count = (count + 1) // 2
                        for output_slot in pl.range(output_count):
                            left = base + output_slot * 2
                            right = left + 1
                            if right < base + count:
                                _merge_index_topk_rows(pair_arena, left, right, base + output_slot, token)
                            else:
                                forwarded = pl.load(pair_arena, [left, 0], [1, _TOPK_PAIR_WIDTH])
                                pl.store(forwarded, [base + output_slot, 0], pair_arena)
                        count = output_count
                root = pl.load(pair_arena, [base, 0], [1, _TOPK_PAIR_WIDTH])
                root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                above = pl.cast(0, pl.INDEX)
                for lane in pl.unroll(8):
                    partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                    above = above + pl.cast(partial, pl.INDEX)
                _materialize_index_topk(
                    index_scores, block_table, request_ids, output,
                    token, width, cutoff, INDEX_TOPK - above,
                )
    return merge_tid


@pl.jit.inline
def _materialize_pool_index_topk(
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    cutoff: pl.Scalar[pl.FP32],
    tie_quota: pl.Scalar[pl.INDEX],
):
    """Map compact winners through candidate block IDs before looking up pages."""
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
    if request >= 0 and request < pl.tensor.dim(block_table, 0) and pl.tensor.dim(block_table, 1) > 0:
        capacity = pl.tensor.dim(pool_block_ids, 1)
        raw_pool = pl.load(pool_block_ids, [token, 0], [1, 2048], valid_shape=[1, capacity])
        pool = pl.tile.fillpad(raw_pool, pad_value=pl.PadValue.zero)
        table_width = pl.tensor.dim(block_table, 1)
        valid_pages = pl.min(table_width, _TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, _TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.UINT8)
        count = pl.cast(0, pl.INDEX)
        remaining = tie_quota
        for start in pl.range(0, width, 256):
            if count < INDEX_TOPK:
                valid = pl.min(256, width - start)
                raw = pl.load(scores, [token, start], [1, 256], valid_shape=[1, valid])
                values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                strict = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
                ties = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
                if remaining > 0:
                    eq = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=0), ones, 0.0)
                    eq_valid = pl.tile.cmps(lanes, pl.cast(valid, pl.INT32), cmp_type=2)
                    eq = pl.tile.select(eq_valid, eq, 0.0)
                    eq_sums64 = pl.col_sum(pl.reshape(eq, [4, 64]))
                    eq_sums8 = pl.col_sum(pl.reshape(eq_sums64, [8, 8]))
                    available = pl.cast(0, pl.INDEX)
                    for eq_lane in pl.unroll(8):
                        eq_partial = pl.cast(pl.tile.read(eq_sums8, [0, eq_lane]), pl.INT32)
                        available = available + pl.cast(eq_partial, pl.INDEX)
                    take = pl.min(available, remaining)
                    if take > 0:
                        if take == available:
                            ties = eq
                        else:
                            # Only the chunk crossing the tie quota needs this boundary.
                            tie_keys = pl.mul(eq, priority_fp32)
                            tie_pairs = pl.tile.sort32(tie_keys, pl.reinterpret_view(lanes, pl.UINT32))
                            tie_pairs = pl.tile.mrgsort(tie_pairs, block_len=64)
                            tie_left = pl.tile.slice(tie_pairs, [1, 256], [0, 0])
                            tie_right = pl.tile.slice(tie_pairs, [1, 256], [0, 256])
                            tie_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                            tie_merged = pl.tile.mrgsort(tie_left, tie_right, tmp=tie_tmp)
                            equal_keys = pl.tile.gather_mask(tie_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                            last_key = pl.cast(pl.tile.read(equal_keys, [0, take - 1]), pl.INT32)
                            boundary = pl.cast(256, pl.INT32) - last_key
                            ties = pl.tile.select(pl.tile.cmps(lanes, boundary, cmp_type=3), eq, 0.0)
                        remaining = remaining - take
                keep = pl.maximum(strict, ties)
                keep_valid = pl.tile.set_validshape(keep, 1, valid)
                keep_padded = pl.tile.fillpad(keep_valid, pad_value=pl.PadValue.zero)
                keep_sums64 = pl.col_sum(pl.reshape(keep_padded, [4, 64]))
                keep_sums8 = pl.col_sum(pl.reshape(keep_sums64, [8, 8]))
                keep_count = pl.cast(0, pl.INDEX)
                for keep_lane in pl.unroll(8):
                    keep_partial = pl.cast(pl.tile.read(keep_sums8, [0, keep_lane]), pl.INT32)
                    keep_count = keep_count + pl.cast(keep_partial, pl.INDEX)
                selected_count = pl.min(keep_count, INDEX_TOPK - count)
                if selected_count > 0:
                    order_keys = pl.mul(keep_padded, priority_fp32)
                    order_pairs = pl.tile.sort32(order_keys, pl.reinterpret_view(lanes, pl.UINT32))
                    order_pairs = pl.tile.mrgsort(order_pairs, block_len=64)
                    order_left = pl.tile.slice(order_pairs, [1, 256], [0, 0])
                    order_right = pl.tile.slice(order_pairs, [1, 256], [0, 256])
                    order_tmp = pl.tile.create([1, 512], dtype=pl.FP32)
                    order_merged = pl.tile.mrgsort(order_left, order_right, tmp=order_tmp)
                    order_keys = pl.tile.gather_mask(order_merged, mask_pattern=pl.tile.MaskPattern.P0101)
                    offsets_i32 = pl.sub(lane_origin, pl.cast(order_keys, pl.INT32))
                    compact_valid = pl.tile.set_validshape(offsets_i32, 1, selected_count)
                    compact_safe = pl.tile.fillpad(compact_valid, pad_value=pl.PadValue.zero)
                    compact = pl.add(compact_safe, pl.cast(start, pl.INT32))
                    slots = pl.tile.shrs(compact, 3)
                    blocks = pl.tile.gather(pool, slots, tmp)
                    offsets_in_block = pl.sub(compact, pl.mul(slots, 8))
                    logical = pl.add(pl.mul(blocks, 8), offsets_in_block)
                    page_indices = pl.tile.shrs(logical, 7)
                    safe_indices = pl.minimum(page_indices, pl.cast(valid_pages - 1, pl.INT32))
                    page_ids = pl.tile.gather(pages, safe_indices, tmp)
                    # A sparse chunk can touch arbitrary logical pages, not only two adjacent ones.
                    for selected_lane in pl.range(selected_count):
                        logical_page = pl.tile.read(page_indices, [0, selected_lane])
                        if logical_page >= valid_pages:
                            if logical_page < table_width:
                                page = pl.read(block_table, [request, pl.cast(logical_page, pl.INDEX)])
                                pl.tile.write(page_ids, [0, selected_lane], page)
                    offsets = pl.sub(logical, pl.mul(page_indices, _TOPK_PAGE))
                    physical = pl.add(pl.mul(page_ids, _TOPK_PAGE), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
    return output


@pl.jit.inline(auto_scope=False)
def select_pool_index_topk(
    index_scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Select candidate scores and restore the original physical cache-row ABI."""
    width = pl.tensor.dim(index_scores, 1)
    tokens = pl.tensor.dim(index_scores, 0)
    leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
    arena_rows = tokens * leaf_count
    pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(32, name_hint="decode_pool_topk_short", deps=[scores_ready]) as short_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width <= INDEX_TOPK:
                invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                _materialize_pool_index_topk(
                    index_scores, pool_block_ids, block_table, request_ids, output,
                    token, width, invalid_score, pl.cast(0, pl.INDEX),
                )
    total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
    with pl.spmd(32, name_hint="decode_pool_topk_leaf", deps=[scores_ready]) as leaf_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, total_leaves, 32):
            if width > INDEX_TOPK:
                token = work // leaf_count
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                if valid_count > 0:
                    output_slot = token * leaf_count + leaf
                    _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
    with pl.spmd(32, name_hint="decode_pool_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if width > INDEX_TOPK:
                base = token * leaf_count
                count = leaf_count
                # INT32 logical positions require at most 21 binary merge levels.
                for _ in pl.range(21):
                    if count > 1:
                        output_count = (count + 1) // 2
                        for output_slot in pl.range(output_count):
                            left = base + output_slot * 2
                            right = left + 1
                            if right < base + count:
                                _merge_index_topk_rows(pair_arena, left, right, base + output_slot, token)
                            else:
                                forwarded = pl.load(pair_arena, [left, 0], [1, _TOPK_PAIR_WIDTH])
                                pl.store(forwarded, [base + output_slot, 0], pair_arena)
                        count = output_count
                root = pl.load(pair_arena, [base, 0], [1, _TOPK_PAIR_WIDTH])
                root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                above = pl.cast(0, pl.INDEX)
                for lane in pl.unroll(8):
                    partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                    above = above + pl.cast(partial, pl.INDEX)
                _materialize_pool_index_topk(
                    index_scores, pool_block_ids, block_table, request_ids, output,
                    token, width, cutoff, INDEX_TOPK - above,
                )
    return merge_tid


project_index_weights = make_bf16_projection_with_deps(
    D, INDEX_H, name_hint="c1a_index_weights_projection"
)
rotate_index_query = make_rope_with_deps(
    INDEX_H, head_dim=INDEX_DIM, name_hint="c1a_index_query_rope"
)


def make_indexer_prepare(project_query):
    """Bind the Full-owned MX query projection to indexer preparation."""

    @pl.jit.inline(auto_scope=False)
    def c1a_index(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        qr: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        score_width: pl.Scalar[pl.INDEX],
        entry_ready: pl.Scalar[pl.TASK_ID],
        qr_ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        projected_tid = project_query(
            qr, index_wq_b, index_wq_b_scale, projected, num_tokens, qr_ready,
        )
        rotated = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
        rotated_tid = rotate_index_query(projected, cos, sin, rotated, num_tokens, projected_tid)
        query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.FP8E4M3FN)
        query_scale = pl.create_tensor(
            [tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0
        )
        query_tid = quantize_index_query(rotated, query, query_scale, num_tokens, rotated_tid)
        weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
        weights_tid = project_index_weights(
            x, index_weights_proj, weights, num_tokens, entry_ready
        )
        scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
        return query, query_scale, weights, scores, query_tid, weights_tid

    return c1a_index


# C2A index-key projection, RMSNorm, and trailing RoPE.
c2a_project_index_key = make_bf16_projection_with_deps(HEAD_DIM, INDEX_DIM, name_hint="c2a_index_key_projection")
c2a_normalize_index_key = make_norm_with_deps(INDEX_DIM, name_hint="c2a_index_key_rmsnorm")
c2a_rotate_index_key = make_rope_with_deps(1, head_dim=INDEX_DIM, name_hint="c2a_index_key_rope")


@pl.jit.inline(auto_scope=False)
def prepare_c2a_index_key(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Return rotated C2A index keys and the task that publishes their value."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projected_tid = c2a_project_index_key(latent, index_wk, projected, num_tokens, latent_ready)
    normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    normalized_tid = c2a_normalize_index_key(projected, index_norm_weight, normalized, num_tokens, projected_tid)
    rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    rotated_tid = c2a_rotate_index_key(
        normalized,
        compressed_rope_cos, compressed_rope_sin,
        rotated,
        num_tokens, normalized_tid,
    )
    return rotated, rotated_tid


@pl.jit.inline
def _write_c2a_short_topk(
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
):
    """Map an unscored C2A history of at most TopK positions to physical rows."""
    length = pl.read(lengths, [token])
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    lanes = pl.tile.arange(0, [1, _TOPK_PAGE], dtype=pl.INT32)
    invalid = pl.tile.full([1, _TOPK_PAGE], dtype=pl.INT32, value=-1)
    for start in pl.range(0, INDEX_TOPK, _TOPK_PAGE):
        rows = invalid
        if start < length:
            valid_count = pl.min(_TOPK_PAGE, length - start)
            page = pl.read(block_table, [request, start // _TOPK_PAGE])
            if page >= 0:
                physical = pl.add(lanes, pl.cast(page * _TOPK_PAGE + start % _TOPK_PAGE, pl.INT32))
                valid = pl.tile.cmps(lanes, pl.cast(valid_count, pl.INT32), cmp_type=2)
                rows = pl.tile.select(valid, physical, invalid)
        pl.store(rows, [token, start], output)
    return output


@pl.jit.inline
def _sort_c2a_long_topk_physical(
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
):
    """Keep C2A long-history output in ascending physical-row order."""
    rows = pl.load(output, [token, 0], [1, INDEX_TOPK], target_memory=pl.MemorySpace.Vec)
    keys = pl.tile.select(pl.tile.cmps(rows, -1, cmp_type=4), pl.mul(pl.cast(rows, pl.FP32), -1.0), _TOPK_SORT_FLOOR)
    ordered = pl.tile.sort32(keys, pl.reinterpret_view(rows, pl.UINT32))
    ordered = pl.tile.mrgsort(ordered, block_len=64)
    ordered = pl.tile.mrgsort(ordered, block_len=256)
    final = pl.tile.gather_mask(
        pl.tile.slice(ordered, [1, _TOPK_PAIR_WIDTH], [0, 0]),
        mask_pattern=pl.tile.MaskPattern.P1010,
        output_dtype=pl.INT32,
    )
    pl.store(final, [token, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def select_c2a_index_topk(
    index_scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Map short histories directly and return long-history TopK in physical-row order."""
    width = pl.tensor.dim(index_scores, 1)
    tokens = pl.tensor.dim(index_scores, 0)
    leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
    # Preserve a directly visible query dimension in each generated wrapper.
    pair_storage = pl.create_tensor([tokens, leaf_count * _TOPK_PAIR_WIDTH], dtype=pl.FP32)

    with pl.spmd(32, name_hint="c2a_index_topk_short", deps=[scores_ready]) as short_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if pl.read(lengths, [token]) <= INDEX_TOPK:
                _write_c2a_short_topk(block_table, request_ids, lengths, output, token)

    total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
    with pl.spmd(32, name_hint="c2a_index_topk_leaf", deps=[scores_ready]) as leaf_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, total_leaves, 32):
            token = work // leaf_count
            if pl.read(lengths, [token]) > INDEX_TOPK:
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                if valid_count > 0:
                    _sort_c2a_topk_leaf(index_scores, pair_storage, token, logical_begin, valid_count, leaf)

    with pl.spmd(32, name_hint="c2a_index_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if pl.read(lengths, [token]) > INDEX_TOPK:
                count = leaf_count
                for _ in pl.range(21):
                    if count > 1:
                        output_count = (count + 1) // 2
                        for output_slot in pl.range(output_count):
                            left = output_slot * 2
                            right = left + 1
                            if right < count:
                                _merge_c2a_topk_rows(pair_storage, left, right, output_slot, token)
                            else:
                                forwarded = pl.load(
                                    pair_storage, [token, left * _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH]
                                )
                                pl.store(forwarded, [token, output_slot * _TOPK_PAIR_WIDTH], pair_storage)
                        count = output_count
                root = pl.load(pair_storage, [token, 0], [1, _TOPK_PAIR_WIDTH])
                root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                above = pl.cast(0, pl.INDEX)
                for lane in pl.unroll(8):
                    above = above + pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                _materialize_index_topk(
                    index_scores,
                    block_table,
                    request_ids,
                    output,
                    token,
                    width,
                    cutoff,
                    INDEX_TOPK - above,
                )

    with pl.spmd(32, name_hint="c2a_index_topk_physical_order", deps=[merge_tid]) as order_tid:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 32):
            if pl.read(lengths, [token]) > INDEX_TOPK:
                _sort_c2a_long_topk_physical(output, token)
    return order_tid


__all__ = [
    "make_index_scores",
    "score_full",
    "score_reindex",
    "score_c2a_full",
    "prepare_c2a_index_key",
    "select_c2a_index_topk",
    "select_pool_index_topk",
    "select_index_topk",
    "make_indexer_prepare",
]
