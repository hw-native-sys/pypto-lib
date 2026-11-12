# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 index-query projection, quantization and sparse-index controls."""

import pypto.language as pl

from .ops import N_TILE, make_dequantize, make_dot, make_pack_weight, make_quantize, make_rope
from .qkv_proj_rope import make_bf16_projection_staged


# Model configuration.
INDEX_HEADS = 32
INDEX_WIDTH = 128
INDEX_TOPK = 512
WINDOW_SIZE = 128

# Tiling.
QUANT_ROWS_TILE = 16
WEIGHT_ROWS_TILE = 8


def make_indexer_projection(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    rope_width = config["qk_rope_head_dim"]
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    packed_rows = latent * (index_width // N_TILE)
    weights_scale = INDEX_WIDTH ** -0.5 * INDEX_HEADS ** -0.5
    quantize = make_quantize(capacity, latent)
    pack = make_pack_weight(latent, index_width)
    dot = make_dot(capacity, latent, index_width)
    dequantize = make_dequantize(capacity, index_width)
    rotate = make_rope(capacity, INDEX_HEADS, INDEX_WIDTH, rope_width)
    project_weights = make_bf16_projection_staged(hidden, INDEX_HEADS)

    @pl.jit
    def indexer_projection(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        index_wq_b: pl.Tensor[[latent, index_width], pl.INT8],
        index_wq_b_scale: pl.Tensor[[index_width], pl.FP32],
        weights_proj: pl.Tensor[[hidden, INDEX_HEADS], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(index_wq_b, packed)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, packed, accumulator, active)
        projected = pl.create_tensor([capacity, index_width], dtype=pl.BF16)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)

        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint="a2_index_weights"):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS])
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            narrowed = pl.cast(scaled, pl.FP16, mode="rint")
            weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return query, weights

    return indexer_projection


def make_indexer_query_quantize(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(query, [capacity * INDEX_HEADS, INDEX_WIDTH])
        output = pl.reshape(payload, [capacity * INDEX_HEADS, INDEX_WIDTH])
        blocks = (active * INDEX_HEADS + QUANT_ROWS_TILE - 1) // QUANT_ROWS_TILE
        with pl.spmd(blocks, name_hint="a2_index_query_quant"):
            block = pl.tile.get_block_idx()
            row = block * QUANT_ROWS_TILE
            rows = pl.min(QUANT_ROWS_TILE, active * INDEX_HEADS - row)
            q = pl.slice(source, [QUANT_ROWS_TILE, INDEX_WIDTH], [row, 0], valid_shape=[rows, INDEX_WIDTH])
            q = pl.set_validshape(pl.fillpad(q, pad_value=pl.PadValue.zero), QUANT_ROWS_TILE, INDEX_WIDTH)
            value = pl.cast(q, pl.FP32)
            maximum = pl.row_max(pl.abs(value))
            raw_scale = pl.div(maximum, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0 ** -24)
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row:row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode="rint"), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token:token + 1, head:head + QUANT_ROWS_TILE] = scale_row
        return payload, scales

    return indexer_query_quantize


def make_empty_selection(capacity):
    """Publish the source empty-cache selection sentinel."""

    @pl.jit
    def empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_empty_sparse_selection"):
            token = pl.tile.get_block_idx()
            raw_selected[token:token + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        return raw_selected

    return empty_selection


def make_sparse_controls(capacity, requests, long_pages, window_pages, long_page_rows=64, *, compress_ratio=2):
    """Filter and sort QLI logical IDs, then map both caches to physical rows."""
    if long_page_rows not in (64, 128):
        raise ValueError("A2 compressed page rows must match the actual 64/128-row cache ABI")
    if compress_ratio not in (1, 2):
        raise ValueError("Sparse controls require the physical source compression ratio 1 or 2")

    @pl.jit
    def sparse_controls(
        raw_selected: pl.Tensor[[capacity, INDEX_TOPK], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        long_block_table: pl.Tensor[[requests, long_pages], pl.INT32],
        window_block_table: pl.Tensor[[requests, window_pages], pl.INT32],
        selected: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[capacity, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[capacity, INDEX_TOPK], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[capacity, WINDOW_SIZE], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="a2_prepare_sparse_indices") as controls_tid:
            token = pl.tile.get_block_idx()
            position = pl.cast(pl.read(positions, [token]), pl.INT32)
            visible = (position + 1) // compress_ratio
            request = pl.read(request_ids, [token])
            sort_scores = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=-1e30)
            sort_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(sort_scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(sort_ids, [0, lane], index)
            sorted_pairs = pl.tile.sort32(sort_scores, pl.reinterpret_view(sort_ids, pl.UINT32))
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.tile.mrgsort(sorted_pairs, block_len=256)
            logical_ids = pl.tile.gather_mask(sorted_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(logical_ids, [token, 0], selected)
            physical_ids = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
            for lane in pl.range(INDEX_TOPK):
                logical = pl.tile.read(logical_ids, [0, lane])
                if logical >= 0:
                    page = logical // long_page_rows
                    offset = logical % long_page_rows
                    block = pl.read(long_block_table, [request, page])
                    physical = pl.cast(block * long_page_rows + offset, pl.INT32)
                    pl.tile.write(physical_ids, [0, lane], physical)
            pl.store(physical_ids, [token, 0], compressed_indices)

            left = pl.max(0, position + 1 - WINDOW_SIZE)
            window_count = position + 1 - left
            window_ids = pl.tile.full([1, WINDOW_SIZE], dtype=pl.INT32, value=-1)
            for lane in pl.range(WINDOW_SIZE):
                if lane < window_count:
                    window_logical = left + lane
                    window_page = window_logical // WINDOW_SIZE
                    window_offset = window_logical % WINDOW_SIZE
                    window_block = pl.read(window_block_table, [request, window_page])
                    window_physical = pl.cast(window_block * WINDOW_SIZE + window_offset, pl.INT32)
                    pl.tile.write(window_ids, [0, lane], window_physical)
            pl.store(window_ids, [token, 0], window_indices)
        with pl.spmd(1, name_hint="a2_sparse_lengths_publish", deps=[controls_tid]):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                count = 0
                for lane in pl.range(INDEX_TOPK):
                    if pl.read(selected, [token, lane]) >= 0:
                        count = count + 1
                pl.write(selected_lengths, [token, 0], pl.cast(count, pl.INT32))
        return selected, selected_lengths, compressed_indices, window_indices

    return sparse_controls
