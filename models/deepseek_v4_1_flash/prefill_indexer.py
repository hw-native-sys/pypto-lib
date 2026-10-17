# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill C1A and C2A index-key preparation, paged scoring, and selection."""

import torch

import pypto.language as pl

from config import INDEX_CACHE_GROUP, FLASH as M, INDEX_BLOCKS_DYN, T_DYN
from hierarchical_sparse_indexer import (
    _index_key_decode_lut, golden_index_query_fp4, quantize_index_query, materialize_index_topk,
    publish_index_cache, select_candidate_block_pool,
)
from rmsnorm import EPS, rms_norm
from quant import bf16_gemm_staged, mxfp8_gemm, dequantize_mxfp4_cache, golden_mxfp8_linear
from qkv_proj_rope import golden_rope_tail


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


INDEX_SCORE_SCALE = INDEX_DIM ** -0.5 * INDEX_H ** -0.5
INDEX_MAX_LOGITS_BYTES = 64 * 1024 * 1024
TOPK_LEAF = 8192
TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
TOPK_GROUP_LEAVES = 2
IDX_SCALES = INDEX_DIM // 32
IDX_PACKED = INDEX_DIM // 2
FP4_MAX_INV = 1.0 / 6.0
FP4_SCALE_FLOOR = 2.0**-9
LEAF = 2048
SORT_FLOOR = -3.0e38

# tiling
M_TILE = 16
INDEX_PROJ_N_TILE = 128
INDEX_PROJ_K_TILE = 256


@pl.jit.inline
def _index_rope(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.constexpr,
    max_workers: pl.constexpr,
):
    """Rotate adjacent pairs in trailing lanes of INDEX_DIM-wide heads."""
    blocks = num_tokens * heads
    workers = pl.min(blocks, max_workers)
    for worker in pl.spmd(workers, name_hint="prefill_index_rope"):
        for block in pl.range(worker, blocks, workers):
            token = block // heads
            head = block % heads
            base = head * INDEX_DIM
            rope_start = base + INDEX_DIM - ROPE_DIM
            nope = pl.slice(x, [1, INDEX_DIM - ROPE_DIM], [token, base])
            output[token:token + 1, base:base + INDEX_DIM - ROPE_DIM] = nope
            tail = pl.cast(pl.slice(x, [1, ROPE_DIM], [token, rope_start]), pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[token:token + 1, :]
            sine = pl.mul(sin[token:token + 1, :], 1.0)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, sine))
            im = pl.add(pl.mul(even, sine), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token:token + 1, rope_start:rope_start + ROPE_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
    return output


@pl.jit.inline
def project_index_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project BF16 per-head weights with BF16 rounding before and after scaling."""
    tokens = pl.tensor.dim(x, 0)
    projected_weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = bf16_gemm_staged(
        x, weight, projected_weights, num_tokens, ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_H, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_weights",
    )
    with pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c1a_index_weights_scale") as weights_tid:
        block = pl.tile.get_block_idx()
        token = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - token)
        projected = pl.slice(projected_weights, [M_TILE, INDEX_H], [token, 0], valid_shape=[rows, INDEX_H])
        scaled = pl.mul(pl.cast(projected, pl.FP32), INDEX_SCORE_SCALE)
        rounded = pl.cast(scaled, pl.BF16, mode="rint")
        output[token:token + M_TILE, :] = pl.set_validshape(rounded, rows, INDEX_H)
    return weights_tid


@pl.jit.inline(auto_scope=False)
def prepare_prefill_index_query(
    x: pl.Tensor,
    latent: pl.Tensor,
    query_weight: pl.Tensor,
    query_weight_scale: pl.Tensor,
    weight_projection: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Project, rotate, and quantize query rows with group-32 scales."""
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    mxfp8_gemm(
        latent, query_weight, query_weight_scale, projected, num_tokens,
        width=Q_LORA, output_width=INDEX_H * INDEX_DIM, fp32_output=False,
        n_tile=128, name_hint="c1a_mx_projection",
    )
    rotated = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    _index_rope(projected, cos, sin, rotated, num_tokens, heads=INDEX_H, max_workers=32)
    query_tid = quantize_index_query(rotated, query, query_scale, num_tokens, ready, dequantize=False)
    weights_tid = project_index_weights(x, weight_projection, weights, num_tokens)
    return query_tid, weights_tid


@pl.jit.inline(auto_scope=False)
def _prefill_index_scores(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    tile_n: pl.constexpr,
    query_tile: pl.constexpr,
    pool: pl.constexpr,
    skip_short: pl.constexpr,
    head_rows: pl.constexpr,
    part_rows: pl.constexpr,
):
    """Score paged or compact-pool MX index keys."""
    groups = pl.cast((num_tokens + query_tile - 1) // query_tile, pl.INDEX)
    width = pl.tensor.dim(output, 1)
    capacity = pl.tensor.dim(pool_ids, 1)
    prefix_rows = groups + 1
    prefix = pl.create_tensor([prefix_rows], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_index_plan", deps=[ready]) as plan_tid:
        total = pl.cast(0, pl.INDEX)
        pl.write(prefix, [0], pl.cast(total, pl.INT32))
        for group in pl.range(groups):
            visible = pl.cast(0, pl.INDEX)
            for row in pl.unroll(query_tile):
                token = group * query_tile + row
                if token < num_tokens:
                    length = pl.cast(pl.max(0, pl.read(lengths, [token])), pl.INDEX)
                    if pool:
                        length = pl.min(capacity, pl.max(0, pl.read(pool_counts, [token]))) * 8
                    elif skip_short:
                        if length <= INDEX_TOPK:
                            length = pl.cast(0, pl.INDEX)
                    visible = pl.max(visible, pl.min(length, width))
            total = total + (visible + tile_n - 1) // tile_n
            pl.write(prefix, [group + 1], pl.cast(total, pl.INT32))

    output_panels = (width + 127) // 128
    with pl.spmd(32, name_hint="prefill_index_score_clear", deps=[ready]) as clear_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * output_panels, 32):
            token = work // output_panels
            begin = (work % output_panels) * 128
            clear_value = pl.reinterpret_view(pl.tile.full([1, 128], dtype=pl.INT32, value=-8388608), pl.FP32)
            pl.store(pl.set_validshape(clear_value, 1, pl.min(128, width - begin)), [token, begin], output)

    cache_pages = pl.tensor.dim(cache, 0)
    cache_rows = cache_pages * 128
    packed_flat = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])
    scale_rows = cache_rows // 8
    scale_flat = pl.reshape(cache_scale, [scale_rows, 32])
    query_rows = pl.tensor.dim(query, 0) * INDEX_H
    query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
    table_width = pl.tensor.dim(block_table, 1)
    requests = pl.tensor.dim(block_table, 0)
    with pl.spmd(32, name_hint="prefill_index_mx_scores", deps=[plan_tid, clear_tid],
                 optimizations=[pl.cross_core_slot(slot_num=2)]) as score_tid:
        worker = pl.tile.get_block_idx()
        lut = _index_key_decode_lut()
        work_items = pl.cast(pl.read(prefix, [groups]), pl.INDEX)
        for work in pl.range(worker, work_items, 32):
            lower = pl.cast(0, pl.INDEX)
            upper = groups
            for _ in pl.range(32):
                if lower + 1 < upper:
                    middle = (lower + upper) // 2
                    if pl.read(prefix, [middle]) <= work:
                        lower = middle
                    else:
                        upper = middle
            token = lower * query_tile
            macro = work - pl.read(prefix, [lower])
            begin = macro * tile_n
            request = pl.read(request_ids, [token])
            maximum_visible = pl.cast(0, pl.INDEX)
            for row in pl.unroll(query_tile):
                if token + row < num_tokens:
                    row_length = pl.cast(pl.max(0, pl.read(lengths, [token + row])), pl.INDEX)
                    if skip_short:
                        if row_length <= INDEX_TOPK:
                            row_length = pl.cast(0, pl.INDEX)
                    maximum_visible = pl.max(maximum_visible, row_length)
            valid_heads = pl.min(query_tile, num_tokens - token) * INDEX_H
            query_panel = pl.load(query_flat, [token * INDEX_H, 0], [head_rows, INDEX_DIM],
                                  valid_shape=[valid_heads, INDEX_DIM])
            query_padded = pl.tile.fillpad(pl.reinterpret_view(query_panel, pl.INT8),
                                           pad_value=pl.PadValue.zero)
            query_panel = pl.reinterpret_view(query_padded, pl.FP8E4M3FN)
            query_scales = pl.load(query_scale, [token, 0], [query_tile, INDEX_H * 4],
                                   valid_shape=[pl.min(query_tile, num_tokens - token), INDEX_H * 4])
            query_signed = pl.reinterpret_view(pl.reinterpret_view(query_scales, pl.UINT8), pl.INT8)
            query_signed = pl.tile.fillpad(query_signed, pad_value=pl.PadValue.zero)
            query_bytes = pl.reinterpret_view(query_signed, pl.UINT8)
            query_bytes = pl.reshape(query_bytes, [1, head_rows * 4])
            query_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
            query_zz = pl.tmov_x2zz(query_bytes, query_tmp, group_axis=1, dst_rows=head_rows, dst_cols=4)
            query_mat = pl.tile.move(query_panel, target_memory=pl.MemorySpace.Mat,
                                     blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
            query_left = pl.tile.move(query_mat, target_memory=pl.MemorySpace.Left)
            query_scale_mat = pl.tile.move(pl.reinterpret_view(query_zz, pl.FP8E8M0),
                                           target_memory=pl.MemorySpace.Mat,
                                           blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major)
            query_scale_left = pl.tile.move(query_scale_mat, target_memory=pl.MemorySpace.LeftScale)
            packed = pl.tile.full([tile_n, INDEX_DIM // 2], dtype=pl.INT8, value=0)
            raw_scale = pl.tile.full([1, tile_n * 4], dtype=pl.INT8, value=127)
            valid_lanes = pl.tile.full([1, tile_n], dtype=pl.INT32, value=0)
            for part in pl.range(tile_n // part_rows):
                if pool:
                    slot = begin // 8 + part
                    logical = pl.cast(-1, pl.INDEX)
                    if slot < pl.min(capacity, pl.max(0, pl.read(pool_counts, [token]))):
                        candidate = pl.read(pool_ids, [token, slot])
                        if candidate >= 0 and candidate < (maximum_visible + 7) // 8:
                            logical = pl.cast(candidate, pl.INDEX) * 8
                else:
                    logical = begin + part * 128
                if logical >= 0 and logical < maximum_visible:
                    logical_page = logical // 128
                    if request >= 0 and request < requests and logical_page < table_width:
                        page = pl.read(block_table, [pl.cast(request, pl.INDEX), logical_page])
                        if page >= 0 and page < cache_pages:
                            physical_row = pl.cast(page, pl.INDEX) * 128 + logical % 128
                            valid_rows = pl.min(part_rows, maximum_visible - logical)
                            payload_raw = pl.load(packed_flat, [physical_row, 0], [part_rows, INDEX_DIM // 2])
                            payload = pl.reinterpret_view(payload_raw, pl.INT8)
                            packed = pl.tile.assemble(packed, payload, [part * part_rows, 0])
                            scales = pl.load(scale_flat, [physical_row // 8, 0], [part_rows // 8, 32])
                            scale_bytes = pl.reinterpret_view(pl.reinterpret_view(scales, pl.UINT8), pl.INT8)
                            scale_bytes = pl.reshape(scale_bytes, [1, part_rows * 4])
                            raw_scale = pl.tile.assemble(raw_scale, scale_bytes, [0, part * part_rows * 4])
                            lanes = pl.tile.arange(0, [1, part_rows], dtype=pl.INT32)
                            ones = pl.tile.full([1, part_rows], dtype=pl.INT32, value=1)
                            part_mask = pl.tile.cmps(lanes, pl.cast(valid_rows, pl.INT32), cmp_type=2)
                            part_valid = pl.tile.select(part_mask, ones, 0)
                            valid_lanes = pl.tile.assemble(valid_lanes, part_valid, [0, part * part_rows])
            byte_values = pl.ands(pl.cast(packed, pl.INT16), 255)
            low = pl.ands(byte_values, 15)
            high = pl.ands(pl.shrs(byte_values, 4), 15)
            lut_tmp = pl.create_tile([1, 32], dtype=pl.UINT8)
            low_bytes = pl.tile.gather(lut, low, lut_tmp)
            high_bytes = pl.tile.gather(lut, high, lut_tmp)
            words = pl.or_(low_bytes, pl.shls(high_bytes, 8))
            carrier = pl.reinterpret_view(words, pl.INT8, shape=[tile_n, INDEX_DIM])
            keys = pl.reinterpret_view(carrier, pl.FP8E4M3FN)
            scale_tmp = pl.create_tile([1, 128], dtype=pl.UINT8)
            scale_zz = pl.tmov_x2zz(pl.reinterpret_view(raw_scale, pl.UINT8), scale_tmp,
                                   group_axis=1, dst_rows=tile_n, dst_cols=4)
            scale_nn = pl.tile.transpose_view(pl.reinterpret_view(scale_zz, pl.FP8E8M0))
            keys_kn = pl.reinterpret_view(pl.tile.transpose(pl.reinterpret_view(keys, pl.INT8), 0, 1),
                                          pl.FP8E4M3FN)
            dot = pl.cast(pl.matmul_mx(query_left, query_scale_left, keys_kn, scale_nn),
                          pl.BF16, mode="rint")
            relu = pl.maximum(pl.cast(dot, pl.FP32), 0.0)
            weight_rows = pl.load(weights, [token, 0], [query_tile, INDEX_H],
                                  valid_shape=[pl.min(query_tile, num_tokens - token), INDEX_H])
            weight_rows = pl.tile.fillpad(weight_rows, pad_value=pl.PadValue.zero)
            head_weights = pl.reshape(pl.cast(weight_rows, pl.FP32), [head_rows, 1])
            weighted = pl.row_expand_mul(relu, head_weights)
            weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
            for row in pl.unroll(query_tile):
                if token + row < num_tokens:
                    heads = pl.tile.slice(weighted, [INDEX_H, tile_n], [row * INDEX_H, 0])
                    reduced_scores = pl.cast(pl.cast(pl.col_sum(heads), pl.BF16, mode="rint"), pl.FP32)
                    length = pl.cast(pl.read(lengths, [token + row]), pl.INDEX)
                    if skip_short:
                        if length <= INDEX_TOPK:
                            length = pl.cast(0, pl.INDEX)
                    if pool:
                        valid = valid_lanes
                    else:
                        positions = pl.add(pl.tile.arange(0, [1, tile_n], dtype=pl.INT32),
                                           pl.cast(begin, pl.INT32))
                        visible_mask = pl.tile.cmps(positions, pl.cast(length, pl.INT32), cmp_type=2)
                        valid = pl.tile.select(visible_mask, valid_lanes, 0)
                    invalid = pl.reinterpret_view(pl.tile.full([1, tile_n], dtype=pl.INT32,
                                                              value=-8388608), pl.FP32)
                    masked = pl.tile.select(pl.tile.cmps(valid, 1, cmp_type=0), reduced_scores, invalid)
                    pl.store(pl.set_validshape(masked, 1, pl.min(tile_n, width - begin)),
                             [token + row, begin], output)
    return score_tid


@pl.jit.inline
def _merge_topk_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, TOPK_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_topk_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_topk_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_topk_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(
        scores,
        [token, logical_begin],
        [1, TOPK_LEAF],
        valid_shape=[1, valid_count],
    )
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, TOPK_LEAF], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, TOPK_LEAF], dtype=pl.INT32), logical_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _select_c1a_topk(
    scores: pl.Tensor,
    lengths: pl.Tensor,
    requests: pl.Tensor,
    table: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
):
    """Select stable C1A Top-K physical rows."""
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    groups_per_token = (width + TOPK_GROUP_LEAVES * TOPK_LEAF - 1) // (TOPK_GROUP_LEAVES * TOPK_LEAF)
    roots = tokens * groups_per_token
    arena_rows = roots + tokens * TOPK_GROUP_LEAVES
    arena = pl.create_tensor([arena_rows, TOPK_PAIR_WIDTH], dtype=pl.FP32)
    with pl.spmd(num_tokens, name_hint="prefill_index_topk_groups", deps=[ready]) as group_tid:
        token = pl.tile.get_block_idx()
        visible = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        if pool:
            visible = pl.min(pl.max(pl.read(pool_counts, [token]), 0) * 8, width)
        work_visible = pl.max(visible, 1)
        leaves = (work_visible + TOPK_LEAF - 1) // TOPK_LEAF
        group_count = (leaves + TOPK_GROUP_LEAVES - 1) // TOPK_GROUP_LEAVES
        scratch = roots + token * TOPK_GROUP_LEAVES
        for group in pl.range(group_count):
            leaf_begin = group * TOPK_GROUP_LEAVES
            count = pl.min(TOPK_GROUP_LEAVES, leaves - leaf_begin)
            root = token * groups_per_token + group
            if count == 1:
                begin = leaf_begin * TOPK_LEAF
                valid = pl.min(TOPK_LEAF, work_visible - begin)
                _sort_topk_leaf(scores, arena, token, begin, valid, root)
            else:
                for leaf in pl.unroll(TOPK_GROUP_LEAVES):
                    begin = (leaf_begin + leaf) * TOPK_LEAF
                    valid = pl.min(TOPK_LEAF, work_visible - begin)
                    _sort_topk_leaf(scores, arena, token, begin, valid, scratch + leaf)
                _merge_topk_pairs(arena, scratch, scratch + 1, root)
    with pl.spmd(num_tokens, name_hint="prefill_index_topk_merge", deps=[group_tid]) as selected:
        token = pl.tile.get_block_idx()
        visible = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        if pool:
            visible = pl.min(pl.max(pl.read(pool_counts, [token]), 0) * 8, width)
        count = (pl.max(visible, 1) + TOPK_GROUP_LEAVES * TOPK_LEAF - 1) // (TOPK_GROUP_LEAVES * TOPK_LEAF)
        base = token * groups_per_token
        for _ in pl.unroll(6):
            if count > 1:
                count = _merge_topk_level(arena, base, count)
        pairs = pl.load(arena, [base, 0], [1, TOPK_PAIR_WIDTH])
        values = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P0101)
        finite_values = pl.maximum(values, -1e29)
        cutoff = pl.tile.read(finite_values, [0, INDEX_TOPK - 1])
        ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
        flags = pl.tile.select(pl.tile.cmps(values, cutoff, cmp_type=4), ones, 0.0)
        sums64 = pl.col_sum(pl.reshape(flags, [8, 64]))
        sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
        above = pl.cast(0, pl.INDEX)
        for lane in pl.unroll(8):
            partial = pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
            above = above + pl.cast(partial, pl.INDEX)
        materialize_index_topk(
            scores, table, requests, pool_ids, output, token, visible, cutoff, INDEX_TOPK - above, pool=pool,
        )
    return selected


@pl.jit.inline(auto_scope=False)
def _select_c2a_topk(
    scores: pl.Tensor,
    lengths: pl.Tensor,
    requests: pl.Tensor,
    table: pl.Tensor,
    cache_pages: pl.Scalar[pl.INDEX],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Fold 2048-key leaves into the original running physical-row Top-K."""
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    leaf_rows = pl.create_tensor([tokens, LEAF], dtype=pl.INT32)
    with pl.spmd(num_tokens, name_hint="prefill_c2a_topk", deps=[ready]) as selected:
        token = pl.tile.get_block_idx()
        request = pl.read(requests, [token])
        length = pl.min(pl.max(pl.read(lengths, [token]), 0), width)
        empty = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        pl.store(empty, [token, 0], output)
        if length <= INDEX_TOPK:
            written = pl.cast(0, pl.INDEX)
            for part in pl.range(INDEX_TOPK // 128):
                begin = part * 128
                if begin < length and request >= 0 and request < pl.tensor.dim(table, 0):
                    page = pl.read(table, [request, part])
                    if page >= 0 and page < cache_pages:
                        count = pl.min(128, length - begin)
                        short_rows = pl.add(pl.tile.arange(0, [1, 128], dtype=pl.INT32), pl.cast(page * 128, pl.INT32))
                        pl.store(pl.set_validshape(short_rows, 1, count), [token, written], output)
                        written = written + count
        else:
            running = pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=SORT_FLOOR)
            for leaf, (previous,) in pl.range((length + LEAF - 1) // LEAF, init_values=(running,)):
                begin = leaf * LEAF
                for part in pl.range(LEAF // 128):
                    logical = begin + part * 128
                    physical_rows = pl.tile.full([1, 128], dtype=pl.INT32, value=-1)
                    if logical < length and request >= 0 and request < pl.tensor.dim(table, 0):
                        page = pl.read(table, [request, logical // 128])
                        if page >= 0 and page < cache_pages:
                            row_offsets = pl.tile.arange(0, [1, 128], dtype=pl.INT32)
                            physical_rows = pl.add(row_offsets, pl.cast(page * 128, pl.INT32))
                    pl.store(physical_rows, [token, part * 128], leaf_rows)
                raw = pl.load(scores, [token, begin], [1, LEAF], valid_shape=[1, pl.min(LEAF, length - begin)])
                padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                values = pl.maximum(padded, SORT_FLOOR)
                leaf_row_ids = pl.load(leaf_rows, [token, 0], [1, LEAF])
                pairs = pl.tile.sort32(values, pl.reinterpret_view(leaf_row_ids, pl.UINT32))
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
                pairs = pl.tile.mrgsort(pairs, block_len=1024)
                merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(previous, pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0]), tmp=merge_tmp)
                running = pl.yield_(pl.tile.slice(merged, [1, TOPK_PAIR_WIDTH], [0, 0]))
            chosen_values = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P0101)
            chosen = pl.tile.gather_mask(running, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            valid = pl.tile.cmps(chosen_values, -1e29, cmp_type=4)
            order_values = pl.tile.select(valid, pl.mul(pl.cast(chosen, pl.FP32), -1.0), SORT_FLOOR)
            order_rows = pl.tile.select(valid, chosen, -1)
            ordered = pl.tile.sort32(order_values, pl.reinterpret_view(order_rows, pl.UINT32))
            ordered = pl.tile.mrgsort(ordered, block_len=64)
            ordered = pl.tile.mrgsort(ordered, block_len=256)
            final = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            pl.store(final, [token, 0], output)
    return selected


@pl.jit.inline
def publish_index_key(
    keys: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish C2A group-32 MXFP4 keys after cache_ready.

    Scales have a 2**-9 floor and midpoint ties round toward zero;
    publish_index_cache uses a 2**-126 floor and round-to-nearest-even.
    Both payloads use the same group-32 MXFP4 scoring reader.
    """
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, IDX_PACKED])
    scale_flat = pl.reshape(scales, [cache_rows, IDX_SCALES])
    with pl.spmd(num_tokens, name_hint="c2a_index_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(keys, [1, INDEX_DIM * 8], [t, 0], valid_shape=[1, INDEX_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, INDEX_DIM * 8)
            groups = pl.reshape(pl.cast(source, pl.FP32), [INDEX_DIM * 8 // 32, 32])
            amax = pl.row_max(pl.abs(groups))
            raw = pl.maximum(pl.mul(amax, FP4_MAX_INV), FP4_SCALE_FLOOR)
            exponent = pl.shrs(pl.add(pl.reinterpret_view(raw, pl.INT32), 8388607), 23)
            factor = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.minimum(pl.maximum(pl.row_expand_div(groups, factor), -6.0), 6.0)
            magnitude = pl.abs(normalized)
            step0 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32), 1), 31)
            step1 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32), 1), 31)
            step2 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32), 1), 31)
            step3 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32), 1), 31)
            step4 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32), 1), 31)
            step5 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32), 1), 31)
            step6 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32), 1), 31)
            lower = pl.add(pl.add(step0, step1), pl.add(step2, step3))
            upper = pl.add(pl.add(step4, step5), pl.add(step6, 7))
            index = pl.add(lower, upper)
            sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
            codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, INDEX_DIM * 8])
            low = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            packed_bytes = pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8)
            flat[slot : slot + 1, :] = pl.set_validshape(packed_bytes, 1, IDX_PACKED)
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            scale_bytes = pl.reinterpret_view(pl.cast(signed_exponent, pl.INT8), pl.UINT8)
            encoded = pl.reinterpret_view(scale_bytes, pl.FP8E8M0)
            encoded_row = pl.reshape(encoded, [1, INDEX_DIM * 8 // 32])
            scale_flat[slot : slot + 1, :] = pl.set_validshape(encoded_row, 1, IDX_SCALES)
    return publish_tid


@pl.jit.inline
def prefill_index_key_project(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Project C2A keys, then normalize after the compressor pool_ready dependency."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = bf16_gemm_staged(
        latent, weight, projected, num_tokens, projection_ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_key",
    )
    norm_blocks = (num_tokens + M_TILE - 1) // M_TILE
    with pl.spmd(norm_blocks, name_hint="prefill_c2a_index_norm", deps=[pool_ready]) as norm_tid:
        t0 = pl.tile.get_block_idx() * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_DIM], [t0, 0], valid_shape=[rows, INDEX_DIM]), pl.FP32)
        square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / INDEX_DIM)
        inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
        gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, INDEX_DIM])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, INDEX_DIM)
    return norm_tid


@pl.jit.inline(auto_scope=False)
def publish_c1a_index_key(
    compressed_latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(compressed_latent, 0)
    index_projection = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    projected_result, projected_tid = bf16_gemm_staged(
        compressed_latent, index_wk, index_projection, num_tokens, projection_ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=2147483647, name_hint="prefill_index_key",
    )
    index_normalized = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(pl.max(1, norm_blocks), name_hint="attention_rmsnorm"):
        if norm_worker < norm_blocks:
            rms_norm(index_projection, index_norm_weight, index_normalized, norm_worker, num_tokens, width=INDEX_DIM)
    index_rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    _index_rope(
        index_normalized, compressed_rope_cos, compressed_rope_sin, index_rotated, num_tokens,
        heads=1, max_workers=2147483647,
    )
    index_ready = publish_index_cache(
        index_rotated, compressed_slots, index_cache, index_cache_scale, num_tokens,
    )
    return index_ready


@pl.jit.inline(auto_scope=False)
def publish_c2a_index_key(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    pool_tid: pl.Scalar[pl.TASK_ID],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(latent, 0)
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    prefill_index_key_project(latent, index_wk, index_norm_weight, index_key, num_tokens, pool_tid)
    rotated_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    _index_rope(
        index_key, compressed_rope_cos, compressed_rope_sin, rotated_key, num_tokens,
        heads=1, max_workers=2147483647,
    )
    index_publish_tid = publish_index_key(
        rotated_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready
    )
    return index_publish_tid


@pl.jit.inline(auto_scope=False)
def _clear_inactive_pool(
    ids: pl.Tensor,
    counts: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    rows = pl.tensor.dim(ids, 0)
    width = pl.tensor.dim(ids, 1)
    first = pl.cast(pl.max(num_tokens, 0), pl.INDEX)
    tiles = (width + 127) // 128
    with pl.spmd(32, name_hint="prefill_inactive_pool_ids", deps=[ready]) as ids_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, (rows - first) * tiles, 32):
            row = first + work // tiles
            column = work % tiles * 128
            invalid = pl.tile.full([1, 128], dtype=pl.INT32, value=-1)
            pl.store(pl.set_validshape(invalid, 1, pl.min(128, width - column)), [row, column], ids)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_inactive_pool_counts", deps=[ids_tid]) as counts_tid:
        for row in pl.range(first, rows):
            pl.write(counts, [row], pl.cast(0, pl.INT32))
    return counts_tid


@pl.jit.inline(auto_scope=False)
def prefill_indexer(
    x: pl.Tensor,
    latent: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    table: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query_weight: pl.Tensor,
    query_weight_scale: pl.Tensor,
    weight_projection: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
    c2a: pl.constexpr,
):
    """Prepare queries, score bounded chunks, and select sparse cache rows."""
    width = pl.tensor.dim(table, 1) * 128
    if pool:
        width = pl.tensor.dim(pool_ids, 1) * 8
    width = (width + 127) // 128 * 128
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = ready
    if not pool and not c2a:
        if num_tokens < pl.tensor.dim(x, 0):
            cleared = _clear_inactive_pool(pool_ids, pool_counts, num_tokens, ready)
            completion[0] = cleared
    if num_tokens > 0:
        active = pl.cast(num_tokens, pl.INDEX)
        chunk = pl.min(active, pl.max(1, INDEX_MAX_LOGITS_BYTES // (width * 4)))
        scores = pl.create_tensor([chunk, width], dtype=pl.FP32)
        for begin in pl.range(0, active, chunk):
            with pl.scope():
                rows = pl.min(chunk, active - begin)
                count = pl.cast(rows, pl.INT32)
                chunk_x = pl.slice(x, [rows, D], [begin, 0])
                chunk_latent = pl.slice(latent, [rows, Q_LORA], [begin, 0])
                chunk_requests = pl.slice(request_ids, [rows], [begin])
                chunk_lengths = pl.slice(lengths, [rows], [begin])
                chunk_cos = pl.slice(cos, [rows, ROPE_DIM // 2], [begin, 0])
                chunk_sin = pl.slice(sin, [rows, ROPE_DIM // 2], [begin, 0])
                capacity = pl.tensor.dim(pool_ids, 1)
                chunk_pool = pl.slice(pool_ids, [rows, capacity], [begin, 0])
                chunk_counts = pl.slice(pool_counts, [rows], [begin])
                chunk_output = pl.slice(output, [rows, INDEX_TOPK], [begin, 0])
                chunk_scores = pl.slice(scores, [rows, width], [0, 0])
                query = pl.create_tensor([rows, INDEX_H * INDEX_DIM], dtype=pl.FP8E4M3FN)
                query_scale = pl.create_tensor([rows, INDEX_H * 4], dtype=pl.FP8E8M0)
                weights = pl.create_tensor([rows, INDEX_H], dtype=pl.BF16)
                maximum_visible = pl.cast(0, pl.INDEX)
                paired = pl.cast(0, pl.INT32)
                if not pool:
                    if width >= 256:
                        paired = pl.cast(rows % 2 == 0, pl.INT32)
                    for token in pl.range(rows):
                        if c2a:
                            maximum_visible = pl.max(maximum_visible, pl.read(chunk_lengths, [token]))
                        if token % 2 == 1:
                            if pl.read(chunk_requests, [token]) != pl.read(chunk_requests, [token - 1]):
                                paired = pl.cast(0, pl.INT32)
                chunk_ready = completion[0]
                scoring = pl.array.create(1, pl.TASK_ID)
                scoring[0] = chunk_ready
                if not c2a or maximum_visible > INDEX_TOPK:
                    query_tid, weights_tid = prepare_prefill_index_query(
                        chunk_x, chunk_latent, query_weight, query_weight_scale, weight_projection,
                        chunk_cos, chunk_sin, query, query_scale, weights, count, chunk_ready,
                    )
                    prepared = pl.system.task_dummy(deps=[query_tid, weights_tid])
                    if paired != 0:
                        paired_tid = _prefill_index_scores(
                            query, query_scale, weights, cache, cache_scale, table,
                            chunk_requests, chunk_lengths, chunk_pool, chunk_counts, chunk_scores,
                            count, prepared, tile_n=256, query_tile=2, pool=False, skip_short=c2a,
                            head_rows=2 * INDEX_H, part_rows=128,
                        )
                        scoring[0] = paired_tid
                    else:
                        general_tid = _prefill_index_scores(
                            query, query_scale, weights, cache, cache_scale, table,
                            chunk_requests, chunk_lengths, chunk_pool, chunk_counts, chunk_scores,
                            count, prepared, tile_n=128, query_tile=1, pool=pool, skip_short=c2a,
                            head_rows=INDEX_H, part_rows=8 if pool else 128,
                        )
                        scoring[0] = general_tid
                score_ready = scoring[0]
                selection = pl.array.create(1, pl.TASK_ID)
                selection[0] = score_ready
                if c2a:
                    selected = _select_c2a_topk(chunk_scores, chunk_lengths, chunk_requests,
                                              table, pl.tensor.dim(cache, 0), chunk_output, count, score_ready)
                    selection[0] = selected
                else:
                    selected = _select_c1a_topk(
                        chunk_scores, chunk_lengths, chunk_requests, table,
                        chunk_pool, chunk_counts, chunk_output, count, score_ready, pool=pool,
                    )
                    selection[0] = selected
                topk_ready = selection[0]
                completion[0] = topk_ready
                if not pool and not c2a:
                    pool_ready = select_candidate_block_pool(chunk_scores, chunk_lengths, chunk_pool,
                                                            chunk_counts, count, score_ready)
                    completion[0] = pl.system.task_dummy(deps=[topk_ready, pool_ready])
    return completion[0]


def golden_prefill_indexer(
    x, qr, request_ids, index_cache, block_table, compressed_lens,
    wq_b, wq_b_scale, weights_proj, cos, sin, *,
    pool_block_ids=None, pool_block_counts=None, c2a=False, topk=INDEX_TOPK,
):
    """Reference logical FP4 queries, BF16 rounding, and sparse physical-row selection."""
    tokens = x.shape[0]
    width = int(compressed_lens.max()) if tokens else 0
    scores = torch.full((tokens, width), -torch.inf, dtype=torch.float32)
    selected = torch.full((tokens, topk), -1, dtype=torch.int32)
    query = golden_mxfp8_linear(qr, wq_b, wq_b_scale).reshape(tokens, INDEX_H, INDEX_DIM)
    query = golden_rope_tail(query, cos, sin)
    query = golden_index_query_fp4(query)
    weights = (x.float() @ weights_proj.float()).bfloat16()
    weights = (weights.float() * INDEX_SCORE_SCALE).bfloat16()
    keys = index_cache.flatten(0, 1).squeeze(-2)
    for token in range(tokens):
        length = max(0, int(compressed_lens[token]))
        positions = torch.arange(length)
        if pool_block_ids is not None:
            count = int(pool_block_counts[token])
            positions = (pool_block_ids[token, :count, None].long() * 8 + torch.arange(8)).flatten()
            positions = positions[(positions >= 0) & (positions < length)]
        if not positions.numel():
            continue
        pages = block_table[int(request_ids[token]), positions // 128]
        present = (pages >= 0) & (pages < index_cache.shape[0])
        positions, pages = positions[present], pages[present]
        physical = pages.long() * 128 + positions % 128
        if c2a and length <= topk:
            selected[token, :physical.numel()] = physical.int()
            continue
        dots = (query[token].float() @ keys[physical].T.float()).bfloat16().float().relu()
        weighted = (dots * weights[token].float()[:, None]).bfloat16().float()
        values = weighted.sum(0).bfloat16().float()
        scores[token, positions] = values
        count = min(topk, positions.numel())
        if count:
            chosen = torch.argsort(values, descending=True, stable=True)[:count]
            order = physical[chosen].argsort() if c2a else positions[chosen].argsort()
            selected[token, :count] = physical[chosen[order]].int()
    return scores, selected


def golden_prefill_index_scores(mode, rank, cache_outputs, inputs):
    """Use the same Prefill profile for the existing cutoff comparator."""
    from qkv_proj_rope import golden_qkv_proj_rope

    payload = cache_outputs["index_cache"] if mode == "full" else inputs["index_cache"]
    scales = cache_outputs["index_cache_scale"] if mode == "full" else inputs["index_cache_scale"]
    keys = dequantize_mxfp4_cache(payload[rank], scales[rank], 32, "e8m0").bfloat16()
    _, _, latent = golden_qkv_proj_rope(
        inputs["x"][rank], inputs["wq_a"][rank], inputs["wq_a_scale"][rank], inputs["q_norm_weight"][rank],
        inputs["wq_b"][rank], inputs["wq_b_scale"][rank], inputs["wkv"][rank], inputs["wkv_scale"][rank],
        inputs["kv_norm_weight"][rank], inputs["rope_cos"][rank], inputs["rope_sin"][rank],
    )
    pool = {} if mode == "full" else {name: inputs[name][rank] for name in ("pool_block_ids", "pool_block_counts")}
    scores, _ = golden_prefill_indexer(
        inputs["x"][rank], latent, inputs["request_ids"][rank], keys, inputs["index_block_table"][rank],
        inputs["compressed_lens"][rank], inputs["index_wq_b"][rank], inputs["index_wq_b_scale"][rank],
        inputs["index_weights_proj"][rank], inputs["rope_cos"][rank], inputs["rope_sin"][rank], **pool,
    )
    return scores
