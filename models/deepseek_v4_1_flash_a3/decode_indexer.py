# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"


from .gemm import gemm_bf16_to_bf16, gemm_dynamic_int8, gemm_int8
import pypto.language as pl
from .quant import CODE_ROWS_DYN, decode_fp4, decode_scale, fp4_bf16_bits
from .rmsnorm import make_norm_with_deps
from .config import FLASH, INDEX_CACHE_GROUP
from .dsl_types import (
    B_DYN, CMP_POSITIONS_DYN, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, INDEX_POOL_POSITIONS_DYN, TABLE_DYN,
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
INDEX_QUERY_WIDTH = FLASH.index_n_heads * INDEX_DIM
INDEX_QUERY_SCALE_WIDTH = INDEX_QUERY_WIDTH // 32
CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
TOPK_CACHED_PAGES = (FLASH.max_position_embeddings + 127) // 128
EPS = FLASH.rms_norm_eps

_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP

# tiling


INDEX_PROJ_N_TILE = 128
INDEX_QUERY_N_TILE = 256
INDEX_PROJ_K_TILE = 256


INDEX_PUBLISH_T_TILE = 8
INDEX_PUBLISH_WIDTH_TILE = INDEX_PUBLISH_T_TILE * INDEX_DIM
CANDIDATE_WORKER_TILE = 12
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")
VALID_ROWS_DYN = pl.dynamic("A3_VALID_ROWS_DYN")
VALID_COLS_DYN = pl.dynamic("A3_VALID_COLS_DYN")
SCORE_SCALE_COLS_DYN = pl.dynamic("A3_SCORE_SCALE_COLS_DYN")

# Request-group score capacities for the 8192-start-position workloads.
C1A_REQUEST_GROUP_SCORE_WIDTH = 8320
C2A_REQUEST_GROUP_SCORE_WIDTH = 4224


@pl.jit.inline(auto_scope=False)
def _decode_index_score_tile(
    codes: pl.Tile[[CODE_ROWS_DYN, INDEX_DIM // 2], pl.UINT8],
    scale_codes: pl.Tile[[1, SCORE_SCALE_COLS_DYN], pl.UINT8],
    rows: pl.constexpr,
):
    """Expand FP4 values and apply each shared scale to its group of 32."""
    magnitudes = decode_fp4(codes, rows, INDEX_DIM)
    scales = decode_scale(scale_codes, 1, rows * _SCALE_COUNT)
    groups = pl.reshape(pl.cast(magnitudes, pl.FP32), [rows * _SCALE_COUNT, INDEX_CACHE_GROUP])
    factors = pl.reshape(scales, [rows * _SCALE_COUNT, 1])
    values = pl.row_expand_mul(groups, factors)
    return pl.reshape(pl.cast(values, pl.BF16, mode="rint"), [rows, INDEX_DIM])


@pl.jit.inline(auto_scope=False)
def _index_scores(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    valid: pl.Tensor[[VALID_ROWS_DYN, VALID_COLS_DYN], pl.INT32],
    width: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    skip_threshold: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Score packed FP4 index histories through BF16 Cube products."""
    q_flat = query
    qs_flat = query_scale
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    key_flat = index_cache
    scale_flat = index_cache_scale
    panels = (width + 63) // 64
    pair_table = _index_fp4_pairs()
    for token in pl.range(num_tokens):
        # Match the previous flattened (token, panel) assignment to each worker.
        start = (worker + workers - token * panels % workers) % workers
        length = pl.max(0, pl.min(width, pl.read(lengths, [token])))
        q_codes = pl.load(q_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM // 2])
        q_codes_scale = pl.load(qs_flat, [0, token * INDEX_H * 4], [1, INDEX_H * 4])
        q_value = _index_key_score_tile(q_codes, q_codes_scale, pair_table, INDEX_H)
        raw_weight = pl.cast(pl.load(weights, [token, 0], [1, INDEX_H]), pl.FP32)
        scaled_weight = pl.mul(raw_weight, 0.015625)
        head_weight = pl.cast(pl.cast(scaled_weight, pl.BF16, mode="rint"), pl.FP32)
        head_weight = pl.reshape(head_weight, [INDEX_H, 1])
        for panel in pl.range(start, panels, workers):
            begin = panel * 64
            page = pl.cast(-1, pl.INT32)
            if gathered_rows > 0:
                length = width
                page = pl.cast(0, pl.INT32)
            elif length > skip_threshold and begin < length:
                request = pl.read(request_ids, [token])
                if request >= 0 and begin // 128 < table_width:
                    page = pl.read(block_table, [pl.cast(request, pl.INDEX), begin // 128])
            negative = pl.reinterpret_view(pl.tile.full([1, 64], dtype=pl.INT32, value=-8388608), pl.FP32)
            valid_cols = pl.min(64, width - begin)
            if page >= 0:
                if gathered_rows > 0:
                    key_row = token * gathered_rows + begin
                else:
                    key_row = pl.cast(page, pl.INDEX) * 128 + begin % 128
                k_codes = pl.load(key_flat, [key_row, 0], [64, INDEX_DIM // 2])
                k_codes_scale = pl.load(scale_flat, [0, key_row * 4], [1, 64 * 4])
                k_value = _index_key_score_tile(k_codes, k_codes_scale, pair_table, 64)
                dot = pl.matmul(q_value, pl.tile.transpose(k_value, 0, 1), out_dtype=pl.FP32)
                rounded = pl.cast(dot, pl.BF16, mode="rint")
                relu = pl.maximum(pl.cast(rounded, pl.FP32), 0.0)
                weighted = pl.row_expand_mul(relu, head_weight)
                weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                score = pl.tile.slice(weighted, [1, 64], [0, 0])
                for pair in pl.unroll(15):
                    left = pl.tile.slice(weighted, [1, 64], [1 + 2 * pair, 0])
                    right = pl.tile.slice(weighted, [1, 64], [2 + 2 * pair, 0])
                    score = pl.add(score, pl.add(left, right))
                score = pl.add(score, pl.tile.slice(weighted, [1, 64], [31, 0]))
                score = pl.cast(pl.cast(score, pl.BF16, mode="rint"), pl.FP32)
                lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.cast(begin, pl.INT32))
                if gathered_rows > 0:
                    mask_values = pl.load(valid, [0, token * gathered_rows + begin], [1, 64])
                    mask = pl.tile.cmps(mask_values, 1, cmp_type=0)
                else:
                    mask = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(length, pl.INT32), pl.FP32), cmp_type=2)
                masked = pl.tile.select(mask, score, negative)
                pl.store(pl.set_validshape(masked, 1, valid_cols), [token, begin], output)
            else:
                pl.store(pl.set_validshape(negative, 1, valid_cols), [token, begin], output)
    return output


@pl.jit.inline(auto_scope=False)
def _predecode_index_queries(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    query_values: pl.Tensor,
    head_weights: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
):
    """Decode each FP4 query once before the parallel score workers start."""
    capacity = pl.tensor.dim(query, 0)
    query_flat = pl.reshape(query, [capacity * INDEX_H, INDEX_DIM // 2])
    query_scale_flat = pl.reshape(query_scale, [1, capacity * INDEX_H * _SCALE_COUNT])
    workers = pl.max(1, pl.min(pl.min(num_tokens, 8), pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="idx_decode_query", deps=[query_ready, weights_ready]) as decoded:
        worker = pl.tile.get_block_idx()
        pair_table = _index_fp4_pairs()
        for token in pl.range(worker, num_tokens, workers):
            q_codes = pl.load(query_flat, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM // 2])
            q_scale_codes = pl.load(
                query_scale_flat,
                [0, token * INDEX_H * _SCALE_COUNT],
                [1, INDEX_H * _SCALE_COUNT],
            )
            q_value = _index_key_score_tile(q_codes, q_scale_codes, pair_table, INDEX_H)
            pl.store(q_value, [token * INDEX_H, 0], query_values)

            raw_weight = pl.cast(pl.load(weights, [token, 0], [1, INDEX_H]), pl.FP32)
            scaled_weight = pl.mul(raw_weight, 0.015625)
            rounded_weight = pl.cast(scaled_weight, pl.BF16, mode="rint")
            pl.store(rounded_weight, [0, token * INDEX_H], head_weights)
    return decoded


@pl.jit.inline(auto_scope=False)
def _index_scores_predecoded(
    query_values: pl.Tensor,
    head_weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    valid: pl.Tensor[[VALID_ROWS_DYN, VALID_COLS_DYN], pl.INT32],
    width: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    skip_threshold: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Score packed histories while reusing one decoded query per token."""
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    key_flat = index_cache
    scale_flat = index_cache_scale
    panels = (width + 63) // 64
    pair_table = _index_fp4_pairs()
    for token in pl.range(num_tokens):
        start = (worker + workers - token * panels % workers) % workers
        length = pl.max(0, pl.min(width, pl.read(lengths, [token])))
        q_value = pl.load(query_values, [token * INDEX_H, 0], [INDEX_H, INDEX_DIM])
        head_weight = pl.reshape(
            pl.cast(pl.load(head_weights, [0, token * INDEX_H], [1, INDEX_H]), pl.FP32),
            [INDEX_H, 1],
        )
        request = pl.read(request_ids, [token])
        for panel in pl.range(start, panels, workers):
            begin = panel * 64
            page = pl.cast(-1, pl.INT32)
            if gathered_rows > 0:
                length = width
                page = pl.cast(0, pl.INT32)
            elif length > skip_threshold and begin < length:
                if request >= 0 and begin // 128 < table_width:
                    page = pl.read(block_table, [pl.cast(request, pl.INDEX), begin // 128])
            negative = pl.reinterpret_view(
                pl.tile.full([1, 64], dtype=pl.INT32, value=-8388608), pl.FP32,
            )
            valid_cols = pl.min(64, width - begin)
            if page >= 0:
                if gathered_rows > 0:
                    key_row = token * gathered_rows + begin
                else:
                    key_row = pl.cast(page, pl.INDEX) * 128 + begin % 128
                k_codes = pl.load(key_flat, [key_row, 0], [64, INDEX_DIM // 2])
                k_codes_scale = pl.load(
                    scale_flat, [0, key_row * _SCALE_COUNT], [1, 64 * _SCALE_COUNT],
                )
                k_value = _index_key_score_tile(k_codes, k_codes_scale, pair_table, 64)
                dot = pl.matmul(q_value, pl.tile.transpose(k_value, 0, 1), out_dtype=pl.FP32)
                rounded = pl.cast(dot, pl.BF16, mode="rint")
                relu = pl.maximum(pl.cast(rounded, pl.FP32), 0.0)
                weighted = pl.row_expand_mul(relu, head_weight)
                weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                score = pl.tile.slice(weighted, [1, 64], [0, 0])
                for pair in pl.unroll(15):
                    left = pl.tile.slice(weighted, [1, 64], [1 + 2 * pair, 0])
                    right = pl.tile.slice(weighted, [1, 64], [2 + 2 * pair, 0])
                    score = pl.add(score, pl.add(left, right))
                score = pl.add(score, pl.tile.slice(weighted, [1, 64], [31, 0]))
                score = pl.cast(pl.cast(score, pl.BF16, mode="rint"), pl.FP32)
                lanes = pl.add(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.cast(begin, pl.INT32))
                if gathered_rows > 0:
                    mask_values = pl.load(valid, [0, token * gathered_rows + begin], [1, 64])
                    mask = pl.tile.cmps(mask_values, 1, cmp_type=0)
                else:
                    mask = pl.tile.cmps(
                        pl.cast(lanes, pl.FP32), pl.cast(pl.cast(length, pl.INT32), pl.FP32), cmp_type=2,
                    )
                masked = pl.tile.select(mask, score, negative)
                pl.store(pl.set_validshape(masked, 1, valid_cols), [token, begin], output)
            else:
                pl.store(pl.set_validshape(negative, 1, valid_cols), [token, begin], output)
    return output



@pl.jit.inline(auto_scope=False)
def quantize_index_query(
    source: pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH], pl.BF16],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor,
    output_scale: pl.Tensor[[T_DYN, INDEX_QUERY_SCALE_WIDTH], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Rotate query heads and pack group-32 E2M1 values with E8M0 scales."""
    # Materialize the producer dependency before entering the scoped task.
    entry_deps = pl.array.create(1, pl.TASK_ID)
    entry_deps[0] = ready
    query_workers = pl.max(1, pl.min(pl.min(num_tokens, 32), pl.system.available_aiv_count()))
    with pl.spmd(query_workers, name_hint="idx_q_fp4", deps=[entry_deps[0]]) as quant_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, query_workers):
            raw = pl.load(source, [t, 0], [1, INDEX_QUERY_WIDTH])
            heads = pl.reshape(raw, [INDEX_H, INDEX_DIM])
            tail = pl.cast(heads[:, INDEX_DIM - ROPE_DIM:INDEX_DIM], pl.FP32)
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(cos, [t, 0], [1, ROPE_DIM // 2])
            sine = pl.mul(pl.load(sin, [t, 0], [1, ROPE_DIM // 2]), 1.0)
            even_cos = pl.col_expand_mul(even, cosine)
            odd_sin = pl.col_expand_mul(odd, sine)
            real = pl.sub(even_cos, odd_sin)
            even_sin = pl.col_expand_mul(even, sine)
            odd_cos = pl.col_expand_mul(odd, cosine)
            imag = pl.add(even_sin, odd_cos)
            real_lanes = pl.tile.full([INDEX_H, ROPE_DIM], dtype=pl.FP32, value=0.0)
            real_lanes = pl.tile.scatter_mask(real_lanes, real, mask_pattern=pl.tile.MaskPattern.P0101)
            imag_lanes = pl.tile.full([INDEX_H, ROPE_DIM], dtype=pl.FP32, value=0.0)
            imag_lanes = pl.tile.scatter_mask(imag_lanes, imag, mask_pattern=pl.tile.MaskPattern.P1010)
            real_bits = pl.reinterpret_view(real_lanes, pl.INT16)
            imag_bits = pl.reinterpret_view(imag_lanes, pl.INT16)
            rotated_bits = pl.or_(real_bits, imag_bits)
            rotated = pl.reinterpret_view(rotated_bits, pl.FP32)
            rounded_tail = pl.cast(rotated, pl.BF16, mode="rint")
            rotated_heads = pl.tile.assemble(heads, rounded_tail, [0, INDEX_DIM - ROPE_DIM])
            values = pl.reshape(pl.cast(rotated_heads, pl.FP32), [INDEX_QUERY_SCALE_WIDTH, 32])
            reduce_tmp = pl.tile.create([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32)
            raw_maximum = pl.row_max(pl.abs(values), tmp_tile=reduce_tmp)
            maximum = pl.maximum(raw_maximum, 7.052966104933725e-38)
            bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 6.0), pl.INT32)
            exponent = pl.shrs(pl.add(bits, 8388607), 23)
            scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
            normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, scale), -6.0), 6.0)
            magnitude = pl.abs(normalized)
            rounded = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=0.0)
            select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
            predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=0.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=1.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=1.5)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=2.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=3.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=4.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
            level = pl.tile.full([INDEX_QUERY_SCALE_WIDTH, 32], dtype=pl.FP32, value=6.0)
            rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
            codes_base = pl.minimum(pl.mul(rounded, 2.0), 4.0)
            codes_extra = pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0)
            codes = pl.cast(pl.add(codes_base, codes_extra), pl.INT32)
            sign = pl.mul(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31), -8)
            codes = pl.reshape(pl.add(codes, sign), [1, INDEX_QUERY_WIDTH])
            low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            payload = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
            pl.store(payload, [t, 0], output)
            codes = pl.cast(pl.reshape(exponent, [1, INDEX_QUERY_SCALE_WIDTH]), pl.UINT8)
            pl.store(codes, [t, 0], output_scale)
    return quant_tid


def _make_materialize_index_topk():
    """Keep inferred row capacities independent across selector instances."""
    @pl.jit.inline
    def materialize(
        scores: pl.Tensor,
        block_table: pl.Tensor,
        request_ids: pl.Tensor,
        output: pl.Tensor,
        token: pl.Scalar[pl.INDEX],
        width: pl.Scalar[pl.INDEX],
        cutoff: pl.Scalar[pl.FP32],
        tie_quota: pl.Scalar[pl.INDEX],
    ):
        """Compact winners in logical order and gather physical pages from UB."""
        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
        table_width = pl.tensor.dim(block_table, 1)
        valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [token, 0], output)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        index_tmp = pl.create_tile([1, 192], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32, tmp=index_tmp)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.INT32)
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
                    eq_valid = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
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
                            ties = pl.tile.select(pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(boundary, pl.INT32), pl.FP32), cmp_type=3), eq, 0.0)
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
                    logical = pl.add(compact_safe, pl.cast(start, pl.INT32))
                    page_indices = pl.tile.shrs(logical, 7)
                    page_ids = pl.tile.gather(pages, page_indices, tmp)
                    offsets = pl.sub(logical, pl.mul(page_indices, 128))
                    physical = pl.add(pl.mul(page_ids, 128), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
        return output

    return materialize


_materialize_index_topk = _make_materialize_index_topk()

@pl.jit.inline
def _merge_candidate_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, CANDIDATE_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_candidate_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_candidate_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_candidate_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    block_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, block_begin], [1, CANDIDATE_LEAF_TILE], valid_shape=[1, valid_count])
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF_TILE], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(pl.tile.arange(0, [1, CANDIDATE_LEAF_TILE], dtype=pl.INT32), block_begin_i32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline
def _sort_candidate_short_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(scores, [token, 0], [1, CANDIDATE_SHORT_LEAF_TILE], valid_shape=[1, valid_count])
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF_TILE], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _publish_candidate_counts(
    compressed_lens: pl.Tensor,
    pool_block_counts: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    positions: pl.Scalar[pl.INDEX],
    capacity: pl.Scalar[pl.INDEX],
):
    """Publish the complete count vector from one worker."""
    tokens = pl.tensor.dim(pool_block_counts, 0)
    for publish_token in pl.range(tokens):
        published_count = pl.cast(0, pl.INT32)
        if publish_token < active_tokens:
            publish_visible = pl.min(pl.max(pl.read(compressed_lens, [publish_token]), 0), positions)
            publish_blocks = (publish_visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            published_count = pl.cast(pl.min(pl.min(publish_blocks, CANDIDATE_TOPK_BLOCKS), capacity), pl.INT32)
        pl.write(pool_block_counts, [publish_token], published_count)
    return pool_block_counts


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish sorted logical candidate block IDs and per-query counts."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    if block_count <= pl.min(capacity, CANDIDATE_TOPK_BLOCKS):
        all_workers = pl.max(1, pl.min(pl.min(tokens, CANDIDATE_WORKER_TILE), pl.system.available_aiv_count()))
        with pl.spmd(all_workers, name_hint="pool_all") as all_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, tokens, all_workers):
                lanes = pl.tile.arange(0, [1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32)
                ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                if token < active_tokens:
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                    valid = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(visible_blocks, pl.INT32), pl.FP32), cmp_type=2)
                    ids = pl.tile.select(valid, lanes, ids)
                pl.store(pl.tile.set_validshape(ids, 1, capacity), [token, 0], pool_block_ids)
            if worker == 0:
                _publish_candidate_counts(compressed_lens, pool_block_counts, active_tokens, positions, capacity)
        pool_tid = pl.yield_(all_tid)
    else:
        padded_block_count = (block_count + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE * CANDIDATE_LEAF_TILE
        score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
        block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
        # Dispatch one guarded work item for an empty batch.
        score_task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
        score_work_items = score_task_tokens * score_tiles
        score_workers = pl.min(pl.min(score_work_items, 32), pl.system.available_aiv_count())

        with pl.spmd(score_workers, name_hint="pool_score", deps=[scores_ready]) as score_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, score_work_items, score_workers):
                token = work // score_tiles
                score_tile = work % score_tiles
                if token < active_tokens:
                    block_begin = score_tile * CANDIDATE_SCORE_TILE
                    position_begin = block_begin * CANDIDATE_BLOCK_SIZE
                    valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
                    valid_positions = pl.min(CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE, positions - position_begin)
                    raw = pl.load(
                        index_scores, [token, position_begin],
                        [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE], valid_shape=[1, valid_positions],
                    )
                    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
                    grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
                    reduce_tmp = pl.create_tile([CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE], dtype=pl.FP32)
                    maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    if visible > 0:
                        last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
                        if last_block >= block_begin and last_block < block_begin + valid_blocks:
                            pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
                    maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
                    pl.store(maxima, [token, block_begin], block_scores)

        arena = pl.create_tensor([tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
        select_tiles = tokens
        select_workers = pl.max(1, pl.min(select_tiles, pl.system.available_aiv_count()))
        with pl.spmd(select_workers, name_hint="pool_select", deps=[score_tid]) as selected_tid:
            select_worker = pl.tile.get_block_idx()
            for token in pl.range(select_worker, select_tiles, select_workers):
                sorted_ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                if token < active_tokens:
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                    arena_base = token * CANDIDATE_MAX_LEAVES
                    count = 0
                    if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF_TILE:
                        _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
                        count = 1
                    elif visible_blocks > 0:
                        leaf_count = (visible_blocks + CANDIDATE_LEAF_TILE - 1) // CANDIDATE_LEAF_TILE
                        for leaf in pl.range(leaf_count):
                            block_begin = leaf * CANDIDATE_LEAF_TILE
                            valid_count = pl.min(CANDIDATE_LEAF_TILE, visible_blocks - block_begin)
                            _sort_candidate_leaf(block_scores, arena, token, block_begin, valid_count, arena_base + leaf)
                        count = leaf_count
                    if count > 1:
                        count = _merge_candidate_level(arena, arena_base, count)
                    if count > 1:
                        count = _merge_candidate_level(arena, arena_base, count)
                    if count > 1:
                        count = _merge_candidate_level(arena, arena_base, count)
                    if count > 1:
                        count = _merge_candidate_level(arena, arena_base, count)

                    selected_count = pl.min(pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS), capacity)
                    ordered_scores = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.FP32, value=-1e30)
                    ordered_indices = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                    if selected_count > 0:
                        root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
                        selected = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                        for lane in pl.range(selected_count):
                            block_id = pl.tile.read(selected, [0, lane])
                            pl.tile.write(ordered_scores, [0, lane], -pl.cast(pl.cast(block_id, pl.INT32), pl.FP32))
                            pl.tile.write(ordered_indices, [0, lane], block_id)
                    ordered = pl.tile.sort32(ordered_scores, pl.reinterpret_view(ordered_indices, pl.UINT32))
                    ordered = pl.tile.mrgsort(ordered, block_len=64)
                    ordered = pl.tile.mrgsort(ordered, block_len=256)
                    ordered = pl.tile.mrgsort(ordered, block_len=1024)
                    sorted_ids = pl.tile.gather_mask(ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                # Publish the sorted prefix and -1 padding as one aligned pool row.
                pl.store(pl.tile.set_validshape(sorted_ids, 1, capacity), [token, 0], pool_block_ids)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="pool_count", deps=[selected_tid]) as counts_tid:
            _publish_candidate_counts(compressed_lens, pool_block_counts, active_tokens, positions, capacity)
        pool_tid = pl.yield_(counts_tid)
    return pool_tid


@pl.jit.inline
def _index_key_rope_block(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    block: pl.Scalar[pl.INT32],
):
    t = block
    base = 0
    output[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)] = x[t : t + 1, base : base + (INDEX_DIM - ROPE_DIM)]
    tail = pl.cast(x[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM], pl.FP32)
    even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
    odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
    c = cos[t : t + 1, :]
    s = pl.mul(sin[t : t + 1, :], 1.0)
    re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
    im = pl.add(pl.mul(even, s), pl.mul(odd, c))
    rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
    rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
    rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
    output[t : t + 1, base + (INDEX_DIM - ROPE_DIM) : base + INDEX_DIM] = pl.cast(
        rotated, pl.BF16, mode="rint"
    )
    return output


@pl.jit.inline(auto_scope=False)
def c1a_norm_rotate_index_key(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse Decode index-key RMSNorm and RoPE while keeping BF16 rounding."""
    blocks = (num_tokens + 7) // 8
    workers = pl.min(pl.min(blocks, 32), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="norm_rope_idx_k", deps=[ready]) as completed:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            row = block * 8
            rows = pl.max(0, pl.min(8, num_tokens - row))
            raw = pl.load(source, [row, 0], [8, INDEX_DIM], valid_shape=[rows, INDEX_DIM])
            padded = pl.set_validshape(
                pl.fillpad(raw, pad_value=pl.PadValue.zero),
                8, INDEX_DIM,
            )
            value = pl.cast(padded, pl.FP32)
            scratch = pl.create_tile([8, INDEX_DIM], dtype=pl.FP32)
            row_sum = pl.row_sum(pl.mul(value, value), tmp_tile=scratch)
            mean = pl.add(pl.mul(row_sum, 1.0 / INDEX_DIM), EPS)
            inverse = pl.tile.rsqrt(mean, pl.create_tile([8, 1], dtype=pl.FP32), high_precision=True)
            gamma = pl.reshape(
                pl.cast(pl.load(norm_weight, [0], [INDEX_DIM]), pl.FP32),
                [1, INDEX_DIM],
            )
            normalized = pl.cast(
                pl.col_expand_mul(pl.row_expand_mul(value, inverse), gamma),
                pl.BF16,
                mode="rint",
            )
            prefix = pl.tile.slice(
                normalized, [8, INDEX_DIM - ROPE_DIM], [0, 0],
                valid_shape=[rows, INDEX_DIM - ROPE_DIM],
            )
            pl.store(prefix, [row, 0], output)
            tail = pl.cast(
                pl.tile.slice(
                    normalized, [8, ROPE_DIM], [0, INDEX_DIM - ROPE_DIM],
                    valid_shape=[rows, ROPE_DIM],
                ),
                pl.FP32,
            )
            even = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.tile.gather_mask(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.load(cos, [row, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            sine = pl.load(sin, [row, 0], [8, ROPE_DIM // 2], valid_shape=[rows, ROPE_DIM // 2])
            cosine = pl.set_validshape(
                pl.fillpad(cosine, pad_value=pl.PadValue.zero),
                8, ROPE_DIM // 2,
            )
            sine = pl.set_validshape(
                pl.fillpad(sine, pad_value=pl.PadValue.zero),
                8, ROPE_DIM // 2,
            )
            real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
            imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
            real_lanes = pl.tile.scatter_mask(
                pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0),
                real, mask_pattern=pl.tile.MaskPattern.P0101,
            )
            imag_lanes = pl.tile.scatter_mask(
                pl.tile.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0),
                imag, mask_pattern=pl.tile.MaskPattern.P1010,
            )
            bits = pl.or_(
                pl.reinterpret_view(real_lanes, pl.INT16),
                pl.reinterpret_view(imag_lanes, pl.INT16),
            )
            rotated = pl.tile.slice(
                pl.reinterpret_view(bits, pl.FP32), [8, ROPE_DIM], [0, 0],
                valid_shape=[rows, ROPE_DIM],
            )
            pl.store(
                pl.cast(rotated, pl.BF16, mode="rint"),
                [row, INDEX_DIM - ROPE_DIM],
                output,
            )
    return completed


@pl.jit.inline(auto_scope=False)
def publish_index_rows(
    source: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    kv_ready: pl.Scalar[pl.TASK_ID],
):
    """Quantize and publish index rows with E8M0 scales."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])
    flat_scale = pl.reshape(scales, [cache_rows, INDEX_DIM // INDEX_CACHE_GROUP])
    # Each physical page has one writer for its packed scale rows.
    workers = pl.min(pl.min(num_tokens, 4), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="pub_idx", deps=[cache_ready, kv_ready]) as published:
        worker = pl.tile.get_block_idx()
        for t in pl.range(0, num_tokens, INDEX_PUBLISH_T_TILE):
            rows = pl.min(INDEX_PUBLISH_T_TILE, num_tokens - t)
            owned = pl.cast(0, pl.INT32)
            for row in pl.range(rows):
                slot = pl.read(slots, [t + row])
                if slot >= 0 and (slot // 128) % workers == worker:
                    owned = pl.cast(1, pl.INT32)
            if owned != 0:
                raw = pl.load(source, [t, 0], [INDEX_PUBLISH_T_TILE, INDEX_DIM], valid_shape=[rows, INDEX_DIM])
                raw = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), INDEX_PUBLISH_T_TILE, INDEX_DIM)
                values = pl.reshape(pl.cast(raw, pl.FP32), [INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP])
                temporary = pl.tile.create([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32)
                amax = pl.row_max(pl.abs(values), tmp_tile=temporary)
                bounded = pl.maximum(amax, 7.052966104933725e-38)
                bits = pl.reinterpret_view(pl.mul(bounded, 1.0 / 6.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                factors = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                # Raw codes span 0..255; folding to a signed byte would saturate.
                stored_codes = exponent
                normalized = pl.minimum(pl.maximum(pl.row_expand_div(values, factors), -6.0), 6.0)
                magnitude = pl.abs(normalized)
                rounded = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=0.0)
                select_tmp = pl.tile.create([1, 16], dtype=pl.UINT32)
                predicate = pl.tile.cmps(magnitude, 0.25, cmp_type=4)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=0.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 0.75, cmp_type=5)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=1.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.25, cmp_type=4)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=1.5)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 1.75, cmp_type=5)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=2.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 2.5, cmp_type=4)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=3.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 3.5, cmp_type=5)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=4.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                predicate = pl.tile.cmps(magnitude, 5.0, cmp_type=4)
                level = pl.tile.full([INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP, INDEX_CACHE_GROUP], dtype=pl.FP32, value=6.0)
                rounded = pl.tile.sel(predicate, level, rounded, select_tmp)
                # Pack adjacent E2M1 codes.
                codes_base = pl.minimum(pl.mul(rounded, 2.0), 4.0)
                codes_extra = pl.minimum(pl.maximum(pl.sub(rounded, 2.0), 0.0), 3.0)
                codes = pl.add(codes_base, codes_extra)
                sign = pl.mul(pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31), -8)
                codes = pl.reshape(pl.add(pl.cast(codes, pl.INT32), sign), [1, INDEX_PUBLISH_WIDTH_TILE])
                low = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
                high = pl.tile.gather_mask(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                packed = pl.add(low, pl.mul(high, 16))
                signed_bytes = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
                payload_bytes = pl.reinterpret_view(pl.cast(signed_bytes, pl.INT8), pl.UINT8)
                payload_rows = pl.reshape(payload_bytes, [INDEX_PUBLISH_T_TILE, INDEX_DIM // 2])
                scale_codes = pl.reshape(stored_codes, [1, INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP])
                scale_lanes = pl.tile.arange(0, [1, 32], dtype=pl.INT32)
                scale_tmp = pl.tile.create([1, 128], dtype=pl.INT32)
                for row in pl.range(rows):
                    slot = pl.read(slots, [t + row])
                    if slot >= 0 and (slot // 128) % workers == worker:
                        physical = pl.cast(slot, pl.INDEX)
                        payload = pl.tile.slice(payload_rows, [1, INDEX_DIM // 2], [row, 0])
                        scale_offset = pl.cast(row * (INDEX_DIM // INDEX_CACHE_GROUP), pl.INT32)
                        indices = pl.minimum(pl.add(scale_lanes, scale_offset), INDEX_PUBLISH_WIDTH_TILE // INDEX_CACHE_GROUP - 1)
                        selected_codes = pl.tile.gather(scale_codes, indices, scale_tmp)
                        encoded_bytes = pl.cast(selected_codes, pl.UINT8)
                        encoded = encoded_bytes
                        encoded = pl.set_validshape(encoded, 1, INDEX_DIM // INDEX_CACHE_GROUP)
                        pl.store(payload, [physical, 0], flat)
                        pl.store(encoded, [physical, 0], flat_scale)
    return published


@pl.jit.inline(auto_scope=False)
def _index_fp4_pairs():
    byte = pl.cast(pl.tile.arange(0, [1, 256], dtype=pl.INT32), pl.INT16, mode="rint")
    low = pl.cast(pl.cast(fp4_bf16_bits(byte), pl.FP32), pl.INT32, mode="rint")
    high = pl.cast(pl.cast(fp4_bf16_bits(pl.shrs(byte, 4)), pl.FP32), pl.INT32, mode="rint")
    low_bits = pl.sub(low, pl.mul(pl.shrs(low, 31), 65536))
    return pl.add(low_bits, pl.shls(high, 16))


@pl.jit.inline(auto_scope=False)
def _index_key_score_tile(
    payload: pl.Tile[[CODE_ROWS_DYN, INDEX_DIM // 2], pl.UINT8],
    scale_codes: pl.Tile[[1, SCORE_SCALE_COLS_DYN], pl.UINT8],
    pair_table: pl.Tile[[1, 256], pl.INT32],
    rows: pl.constexpr,
):
    packed = pl.reinterpret_view(payload, pl.INT8)
    fp16 = pl.cast(packed, pl.FP16)
    byte_ids = pl.ands(pl.cast(fp16, pl.INT16, mode="rint"), 255)
    indices = pl.cast(pl.cast(byte_ids, pl.FP32), pl.INT32, mode="rint")
    flat_indices = pl.reshape(indices, [1, rows * INDEX_DIM // 2])
    scratch = pl.tile.create([1, rows * INDEX_DIM // 2], dtype=pl.INT32)
    pairs = pl.tile.gather(pair_table, flat_indices, scratch)
    values = pl.reinterpret_view(pairs, pl.BF16, shape=[1, rows * INDEX_DIM])
    factors = pl.reshape(decode_scale(scale_codes, 1, rows * _SCALE_COUNT), [rows * _SCALE_COUNT, 1])
    groups = pl.reshape(pl.cast(values, pl.FP32), [rows * _SCALE_COUNT, INDEX_CACHE_GROUP])
    scaled = pl.cast(pl.row_expand_mul(groups, factors), pl.BF16, mode="rint")
    return pl.reshape(scaled, [rows, INDEX_DIM])


@pl.jit.inline(auto_scope=False)
def _score_single_paged_query(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    skip_threshold: pl.constexpr,
):
    """Expand one query and its visible key pages before mixed scoring."""
    width = pl.tensor.dim(output, 1)
    pages = (width + 127) // 128
    table_width = pl.tensor.dim(block_table, 1)
    request_count = pl.tensor.dim(block_table, 0)
    cache_pages = pl.tensor.dim(cache, 0)
    query_values = pl.create_tensor([INDEX_H, INDEX_DIM], dtype=pl.BF16)
    key_values = pl.create_tensor([pages * 128, INDEX_DIM], dtype=pl.BF16)
    key_payloads = pl.reshape(cache, [cache_pages * 128, INDEX_DIM // 2])
    key_scales = pl.reshape(cache_scale, [1, cache_pages * 128 * _SCALE_COUNT])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_expand_query", deps=[query_ready]) as expanded_q:
        raw_query = pl.tile.load(query, [0, 0], [1, INDEX_H * INDEX_DIM // 2])
        q_codes = pl.reshape(raw_query, [INDEX_H, INDEX_DIM // 2])
        q_bytes = pl.tile.load(query_scale, [0, 0], [1, INDEX_H * _SCALE_COUNT])
        q_decoded = _index_key_score_tile(q_codes, q_bytes, _index_fp4_pairs(), INDEX_H)
        pl.tile.store(q_decoded, [0, 0], query_values)
    unpack_workers = pl.min(pl.min(48, pages * 4), pl.system.available_aiv_count())
    with pl.spmd(unpack_workers, name_hint="idx_expand_keys", deps=[cache_ready]) as expanded_k:
        unpack_worker = pl.tile.get_block_idx()
        pair_table = _index_fp4_pairs()
        unpack_request = pl.read(request_ids, [0])
        unpack_visible = pl.min(width, pl.read(lengths, [0]))
        if unpack_visible > skip_threshold and unpack_request >= 0 and unpack_request < request_count:
            blocks = pl.min(table_width, (unpack_visible + 127) // 128) * 4
            for block in pl.range(unpack_worker, blocks, unpack_workers):
                physical_page = pl.read(block_table, [unpack_request, block // 4])
                if physical_page >= 0 and physical_page < cache_pages:
                    physical = physical_page * 128 + block % 4 * 32
                    k_codes = pl.load(key_payloads, [physical, 0], [32, INDEX_DIM // 2], target_memory=pl.MemorySpace.Vec)
                    k_scales = pl.load(key_scales, [0, physical * _SCALE_COUNT], [1, 32 * _SCALE_COUNT],
                                       target_memory=pl.MemorySpace.Vec)
                    k_decoded = _index_key_score_tile(k_codes, k_scales, pair_table, 32)
                    pl.store(k_decoded, [block * 32, 0], key_values)
    workers = pl.min(pl.min(pages, 16), pl.system.available_cluster_count())
    with pl.spmd(
        workers, name_hint="idx_score_single", sync_start=True,
        deps=[expanded_q, expanded_k, weights_ready],
    ) as scored:
        worker = pl.tile.get_block_idx()
        request = pl.read(request_ids, [0])
        visible = pl.min(width, pl.read(lengths, [0]))
        q_tile = pl.load(query_values, [0, 0], [INDEX_H, INDEX_DIM], target_memory=pl.MemorySpace.Mat)
        for _ in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
            weights_fp32 = pl.cast(pl.load(weights, [0, 0], [1, INDEX_H]), pl.FP32)
            weights_rounded = pl.cast(pl.mul(weights_fp32, 0.015625), pl.BF16, mode="rint")
            head_weights = pl.reshape(pl.cast(weights_rounded, pl.FP32), [INDEX_H, 1])
            lanes = pl.cast(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.FP32)
            invalid = pl.reinterpret_view(pl.tile.full([1, 64], dtype=pl.INT32, value=-8388608), pl.FP32)
        for panel in pl.range(worker, pages, workers):
            page = pl.cast(-1, pl.INT32)
            begin = panel * 128
            if begin < visible and visible > skip_threshold:
                if request >= 0 and request < request_count and panel < table_width:
                    page = pl.read(block_table, [request, panel])
            if page >= 0 and page < cache_pages:
                k_tile = pl.load(key_values, [begin, 0], [128, INDEX_DIM], target_memory=pl.MemorySpace.Mat)
                dot = pl.matmul(q_tile, pl.tile.transpose_view(k_tile), out_dtype=pl.FP32)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    product = pl.aiv_shard(dot)
                    rounded = pl.cast(product, pl.BF16, mode="rint")
                    relu = pl.maximum(pl.cast(rounded, pl.FP32), 0.0)
                    weighted = pl.row_expand_mul(relu, head_weights)
                    weighted = pl.cast(pl.cast(weighted, pl.BF16, mode="rint"), pl.FP32)
                    sums = pl.cast(pl.col_sum(weighted), pl.BF16, mode="rint")
                    position = begin + lane * 64
                    if position < width:
                        edge = pl.cast(pl.cast(visible - position, pl.INT32), pl.FP32)
                        mask = pl.tile.cmps(lanes, edge, cmp_type=2)
                        scores = pl.tile.select(mask, pl.cast(sums, pl.FP32), invalid)
                        pl.store(pl.set_validshape(scores, 1, pl.min(64, width - position)), [0, position], output)
            else:
                for lane in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
                    position = begin + lane * 64
                    if position < width:
                        pl.store(pl.set_validshape(invalid, 1, pl.min(64, width - position)), [0, position], output)
    return scored


def make_index_scores(skip_short=False, request_workers=32):
    """Score all paged histories with native FP4 request groups."""
    skip_threshold = INDEX_TOPK if skip_short else 0

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.UINT8],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.UINT8],
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
        capacity = pl.tensor.dim(query, 0)
        # Paged scores derive validity from lengths. Only the pooled call consumes this argument.
        unused_valid = pl.create_tensor([4, capacity + 1], dtype=pl.INT32)
        launch_workers = pl.max(1, pl.min(request_workers, pl.system.available_cluster_count()))
        table_width = pl.tensor.dim(block_table, 1)
        cache_rows = pl.tensor.dim(index_cache, 0) * 128
        payload_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
        scales_flat = pl.reshape(index_cache_scale, [1, cache_rows * 4])
        query_flat = pl.reshape(query, [capacity * INDEX_H, INDEX_DIM // 2])
        query_scales_flat = pl.reshape(query_scale, [1, capacity * INDEX_H * 4])
        if num_tokens == 1 and width <= C1A_REQUEST_GROUP_SCORE_WIDTH:
            score_tid = _score_single_paged_query(
                query, query_scale, weights, index_cache, index_cache_scale,
                block_table, request_ids, lengths, output, cache_ready, query_ready, weights_ready,
                skip_threshold,
            )
        else:
            if request_workers <= 16:
                query_values = pl.create_tensor([capacity * INDEX_H, INDEX_DIM], dtype=pl.BF16)
                # A flat scratch view leaves the score arena's dynamic row dimension independent.
                head_weights = pl.create_tensor([1, capacity * INDEX_H], dtype=pl.BF16)
                query_decode_tid = _predecode_index_queries(
                    query, query_scale, weights, query_values, head_weights,
                    num_tokens, query_ready, weights_ready,
                )
                with pl.spmd(
                    launch_workers, name_hint="idx_score_req", sync_start=True,
                    deps=[cache_ready, query_decode_tid],
                ) as fallback_tid:
                    _worker = pl.tile.get_block_idx()
                    _index_scores_predecoded(
                        query_values, head_weights, payload_flat, scales_flat,
                        block_table, request_ids, lengths, output, unused_valid,
                        width, table_width, pl.cast(skip_threshold, pl.INT32), 0, num_tokens,
                    )
                score_tid = fallback_tid
            else:
                with pl.spmd(
                    launch_workers, name_hint="idx_score_req", sync_start=True,
                    deps=[cache_ready, query_ready, weights_ready],
                ) as fallback_tid:
                    _worker = pl.tile.get_block_idx()
                    _index_scores(
                        query_flat, query_scales_flat, weights, payload_flat, scales_flat,
                        block_table, request_ids, lengths, output, unused_valid,
                        width, table_width, pl.cast(skip_threshold, pl.INT32), 0, num_tokens,
                    )
                score_tid = fallback_tid
        return score_tid

    return score


score_full = make_index_scores()
score_c2a_full = make_index_scores(skip_short=True)
_score_c1a_request_group = make_index_scores(request_workers=16)
_score_c2a_request_group = make_index_scores(skip_short=True, request_workers=16)


@pl.jit.inline
def _use_request_groups(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    expected_width: pl.constexpr,
    skip_short: pl.constexpr,
):
    grouped = pl.cast(0, pl.INT32)
    middle = pl.cast(num_tokens // 2, pl.INDEX)
    if width == expected_width and middle > 0:
        if pl.read(request_ids, [middle - 1]) != pl.read(request_ids, [middle]):
            grouped = pl.cast(1, pl.INT32)
        if skip_short:
            for token in pl.range(num_tokens):
                if pl.read(lengths, [token]) <= INDEX_TOPK:
                    grouped = pl.cast(0, pl.INT32)
    return grouped


def _make_pool_index_scores(tile_n, panels_per_task, workers):
    """Gather packed candidate blocks and score them with native FP4."""
    blocks_per_panel = tile_n // 8
    blocks_per_task = blocks_per_panel * panels_per_task
    prefix_row = 0 if tile_n == 128 else panels_per_task

    @pl.jit.inline
    def score(
        query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
        query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.UINT8],
        weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.UINT8],
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
        tokens = pl.tensor.dim(query, 0)
        gathered_rows = ((capacity + blocks_per_panel - 1) // blocks_per_panel) * tile_n
        total_rows = tokens * gathered_rows
        total_scales = total_rows * _SCALE_COUNT
        payloads = pl.create_tensor([total_rows, INDEX_DIM // 2], dtype=pl.UINT8)
        scales = pl.create_tensor([total_rows, 4], dtype=pl.UINT8)
        scale_rows_flat = pl.reshape(scales, [1, total_scales])
        valid = pl.create_tensor([1, total_rows], dtype=pl.INT32)
        table_width = pl.tensor.dim(block_table, 1)
        request_count = pl.tensor.dim(block_table, 0)
        gather_workers = pl.max(1, pl.min(workers, pl.system.available_aiv_count()))
        with pl.spmd(gather_workers, name_hint="idx_gather_pool", deps=[ready]) as gather_tid:
            worker = pl.tile.get_block_idx()
            work_items = pl.read(prefix, [prefix_row, num_tokens])
            for work in pl.range(worker, work_items, gather_workers):
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
                for panel in pl.unroll(panels_per_task):
                    slot_begin = first_slot + panel * blocks_per_panel
                    if slot_begin < count:
                        packed_zero = pl.tile.full([tile_n, INDEX_DIM // 2], dtype=pl.FP32, value=0.0)
                        packed = pl.cast(packed_zero, pl.INT8, mode="rint")
                        scale_ones = pl.tile.full([1, tile_n * _SCALE_COUNT], dtype=pl.FP32, value=127.0)
                        raw_scale = pl.cast(scale_ones, pl.INT8, mode="rint")
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
                                            scale_bytes = pl.load(scale_flat, [physical_row // 8, 0], [1, 32])
                                            block_scales = pl.reinterpret_view(scale_bytes, pl.INT8)
                                            raw_scale = pl.tile.assemble(raw_scale, block_scales, [0, block * 32])
                                            lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                            remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                            ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                            block_valid = pl.tile.select(pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(remaining, pl.INT32), pl.FP32), cmp_type=2), ones, 0)
                                            valid_lanes = pl.tile.assemble(valid_lanes, block_valid, [0, block * 8])
                        row_offset = token * gathered_rows + slot_begin * 8
                        packed_u8 = pl.reinterpret_view(packed, pl.UINT8)
                        scale_u8 = pl.reinterpret_view(raw_scale, pl.UINT8)
                        pl.store(packed_u8, [row_offset, 0], payloads)
                        pl.store(scale_u8, [0, row_offset * _SCALE_COUNT], scale_rows_flat)
                        pl.store(valid_lanes, [0, row_offset], valid)
        launch_workers = pl.max(1, pl.min(workers, pl.system.available_cluster_count()))
        width = pl.tensor.dim(output, 1)
        query_flat = pl.reshape(query, [tokens * INDEX_H, INDEX_DIM // 2])
        query_scales_flat = pl.reshape(query_scale, [1, tokens * INDEX_H * 4])
        with pl.spmd(
            launch_workers, name_hint="idx_score_pool", sync_start=True, deps=[gather_tid],
        ) as score_tid:
            _worker = pl.tile.get_block_idx()
            _index_scores(
                query_flat, query_scales_flat, weights, payloads, scale_rows_flat,
                block_table, request_ids, lengths, output, valid,
                width, table_width, pl.cast(0, pl.INT32), gathered_rows, num_tokens,
            )
        return score_tid

    return score


_pool_page = _make_pool_index_scores(128, 1, 32)
_pool_page_small = _make_pool_index_scores(128, 1, 16)
_pool_block = _make_pool_index_scores(256, 1, 32)
_pool_macro = _make_pool_index_scores(256, 2, 32)


@pl.jit.inline(auto_scope=False)
def score_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.UINT8],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.UINT8],
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
    """Route identity pools to paged FP4 scoring and sparse pools to gathered FP4 scoring."""
    tokens = pl.tensor.dim(query, 0)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(output, 1)
    prefix = pl.create_tensor([3, tokens + 1], dtype=pl.INT32)
    route = pl.create_tensor([1, 8], dtype=pl.INT32)
    workers = pl.max(1, pl.min(pl.min(num_tokens, 4), pl.system.available_aiv_count()))
    chunk_rows = (num_tokens + workers - 1) // workers
    # Worker scratch rows occupy separate 128-byte-aligned regions.
    prefix_width = (tokens + 31) // 32 * 32
    partial_prefix = pl.create_tensor([workers * 3, prefix_width], dtype=pl.INT32)
    partial_route = pl.create_tensor([workers, 32], dtype=pl.INT32)
    with pl.spmd(workers, name_hint="idx_plan_pool", deps=[pool_ready]) as partial_tid:
        worker = pl.tile.get_block_idx()
        begin = worker * chunk_rows
        end = pl.min(begin + chunk_rows, num_tokens)
        small = pl.cast(0, pl.INT32)
        block = pl.cast(0, pl.INT32)
        macro = pl.cast(0, pl.INT32)
        dense = pl.cast(1, pl.INT32)
        mismatch_sums = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
        for token in pl.range(begin, end):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            visible = pl.max(0, pl.read(lengths, [token]))
            request = pl.read(request_ids, [token])
            if request < 0 or request >= pl.tensor.dim(block_table, 0):
                dense = pl.cast(0, pl.INT32)
            if count != (visible + 7) // 8 or visible > pl.tensor.dim(block_table, 1) * 128:
                dense = pl.cast(0, pl.INT32)
            if dense == 1 and count > 0:
                pool_raw = pl.load(pool_block_ids, [token, 0], [1, 2048], valid_shape=[1, capacity])
                pool = pl.tile.fillpad(pool_raw, pad_value=pl.PadValue.zero)
                lanes = pl.tile.arange(0, [1, 2048], dtype=pl.INT32)
                difference = pl.cast(pl.sub(pool, lanes), pl.FP32)
                active = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(count, pl.INT32), pl.FP32), cmp_type=2)
                mismatch = pl.tile.select(active, pl.abs(difference), 0.0)
                sums64 = pl.col_sum(pl.reshape(mismatch, [32, 64]))
                mismatch_sums = pl.add(mismatch_sums, sums64)
                page_count = (visible + 127) // 128
                table_columns = pl.min(128, pl.tensor.dim(block_table, 1))
                pages_raw = pl.load(block_table, [request, 0], [1, 128], valid_shape=[1, table_columns])
                pages = pl.tile.fillpad(pages_raw, pad_value=pl.PadValue.zero)
                page_lanes = pl.tile.arange(0, [1, 128], dtype=pl.INT32)
                bad_page = pl.tile.cmps(pl.cast(pages, pl.FP32), pl.cast(pl.cast(pl.tensor.dim(index_cache, 0), pl.INT32), pl.FP32), cmp_type=5)
                page_errors = pl.tile.select(bad_page, pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                page_active = pl.tile.cmps(pl.cast(page_lanes, pl.FP32), pl.cast(pl.cast(page_count, pl.INT32), pl.FP32), cmp_type=2)
                page_errors = pl.tile.select(page_active, page_errors, 0.0)
                page_sums = pl.col_sum(pl.reshape(page_errors, [2, 64]))
                mismatch_sums = pl.add(mismatch_sums, page_sums)
            small = pl.cast(small + (count + 15) // 16, pl.INT32)
            block = pl.cast(block + (count + 31) // 32, pl.INT32)
            macro = pl.cast(macro + (count + 63) // 64, pl.INT32)
            pl.write(partial_prefix, [worker * 3 + 0, token - begin], pl.cast(small, pl.INT32))
            pl.write(partial_prefix, [worker * 3 + 1, token - begin], pl.cast(block, pl.INT32))
            pl.write(partial_prefix, [worker * 3 + 2, token - begin], pl.cast(macro, pl.INT32))
        sums8 = pl.col_sum(pl.reshape(mismatch_sums, [8, 8]))
        for lane in pl.unroll(8):
            if pl.tile.read(sums8, [0, lane]) > 0.0:
                dense = pl.cast(0, pl.INT32)
        values = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        pl.tile.write(values, [0, 0], pl.cast(small, pl.INT32))
        pl.tile.write(values, [0, 1], pl.cast(block, pl.INT32))
        pl.tile.write(values, [0, 2], pl.cast(macro, pl.INT32))
        pl.tile.write(values, [0, 3], dense)
        pl.store(values, [worker, 0], partial_route)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_merge_pool", deps=[partial_tid]) as plan_tid:
        totals = pl.tile.full([1, 8], dtype=pl.INT32, value=0)
        dense = pl.cast(1, pl.INT32)
        zero_prefix = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
        for kind in pl.unroll(3):
            pl.store(pl.set_validshape(zero_prefix, 1, 1), [kind, 0], prefix)
        for worker in pl.range(workers):
            begin = worker * chunk_rows
            rows = pl.max(0, pl.min(chunk_rows, num_tokens - begin))
            summary = pl.load(partial_route, [worker, 0], [1, 8])
            if pl.tile.read(summary, [0, 3]) == 0:
                dense = pl.cast(0, pl.INT32)
            for kind in pl.unroll(3):
                offset = pl.tile.read(totals, [0, kind])
                for start in pl.range(0, rows, 128):
                    prefix_rows = pl.min(128, rows - start)
                    local = pl.load(partial_prefix, [worker * 3 + kind, start], [1, 128], valid_shape=[1, prefix_rows])
                    adjusted = pl.add(local, offset)
                    pl.store(adjusted, [kind, begin + start + 1], prefix)
            totals = pl.add(totals, summary)
        pl.tile.write(totals, [0, 3], dense)
        pl.store(totals, [0, 0], route)
    output_panels = (width + 127) // 128
    clear_workers = pl.max(1, pl.min(pl.min(num_tokens * output_panels, 32), pl.system.available_aiv_count()))
    with pl.spmd(clear_workers, name_hint="idx_clear_pool", deps=[pool_ready]) as clear_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * output_panels, clear_workers):
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
    dense = pl.read(route, [0, 3])
    if small == 0:
        result = ready
    elif dense == 1:
        result = score_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, output, num_tokens, ready, ready, ready,
        )
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


_TOPK_PAGE = 128
_TOPK_PAIR_WIDTH = 2 * INDEX_TOPK
_TOPK_LEAF_TILE = 2048
_TOPK_SORT_FLOOR = -3.4028234663852886e38
_TOPK_INVALID_SCORE_BITS = -8388608


def _make_index_topk_pairs(query_rows=False, stable_bf16=False):
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
        index_tmp = pl.create_tile([1, 192], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        lane_indices = pl.tile.arange(0, [1, _TOPK_LEAF_TILE], dtype=pl.INT32, tmp=index_tmp)
        indices = pl.add(lane_indices, pl.cast(logical_begin, pl.INT32))
        # BF16 scores widened to FP32 leave room to encode a deterministic tie break.
        # The bit pattern stays in the positive normal FP32 range for every finite BF16 score.
        if stable_bf16 and pl.tensor.dim(scores, 1) <= 16384:
            upper = pl.tile.shrs(pl.reinterpret_view(padded, pl.INT32), 16)
            negative = pl.tile.cmps(padded, 0.0, cmp_type=2)
            zero = pl.tile.cmps(padded, 0.0, cmp_type=0)
            ascending = pl.tile.select(negative, pl.sub(pl.mul(upper, -1), 1), pl.add(upper, 32768))
            ascending = pl.tile.select(zero, 32768, ascending)
            reverse_index = pl.add(pl.mul(indices, -1), 16383)
            key_bits = pl.add(pl.add(pl.shls(ascending, 14), reverse_index), 8388608)
            key = pl.reinterpret_view(key_bits, pl.FP32)
            finite_low = pl.tile.cmps(padded, _TOPK_SORT_FLOOR, cmp_type=4)
            finite_high = pl.tile.cmps(padded, -_TOPK_SORT_FLOOR, cmp_type=3)
            keys = pl.tile.select(finite_low, pl.tile.select(finite_high, key, 0.0), 0.0)
            keys = pl.tile.fillpad(keys, pad_value=pl.PadValue.min)
            pairs = pl.tile.sort32(keys, pl.reinterpret_view(indices, pl.UINT32))
        else:
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
_sort_pool_topk_leaf_bf16, _ = _make_index_topk_pairs(stable_bf16=True)
_sort_c2a_topk_leaf, _merge_c2a_topk_rows = _make_index_topk_pairs(query_rows=True)


def _make_select_index_topk(leaf_workers=32, row_workers=32):
    materialize = _make_materialize_index_topk()

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
        if width <= INDEX_TOPK:
            short_workers = pl.max(1, pl.min(pl.min(num_tokens, row_workers), pl.system.available_aiv_count()))
            with pl.spmd(short_workers, name_hint="idx_topk_short", deps=[scores_ready]) as short_tid:
                worker = pl.tile.get_block_idx()
                for token in pl.range(worker, num_tokens, short_workers):
                    invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                    invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                    materialize(
                        index_scores, block_table, request_ids, output,
                        token, width, invalid_score, pl.cast(0, pl.INDEX),
                    )
            topk_tid = pl.yield_(short_tid)
        else:
            leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
            arena_rows = tokens * leaf_count
            pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
            total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
            sort_workers = pl.max(1, pl.min(pl.min(total_leaves, leaf_workers), pl.system.available_aiv_count()))
            with pl.spmd(sort_workers, name_hint="idx_topk_leaf", deps=[scores_ready]) as leaf_tid:
                worker = pl.tile.get_block_idx()
                for work in pl.range(worker, total_leaves, sort_workers):
                    token = work // leaf_count
                    leaf = work - token * leaf_count
                    logical_begin = leaf * _TOPK_LEAF_TILE
                    valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                    output_slot = token * leaf_count + leaf
                    _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
            merge_workers = pl.max(1, pl.min(pl.min(num_tokens, row_workers), pl.system.available_aiv_count()))
            with pl.spmd(merge_workers, name_hint="idx_topk_merge", deps=[leaf_tid]) as merge_tid:
                worker = pl.tile.get_block_idx()
                for token in pl.range(worker, num_tokens, merge_workers):
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
                    materialize(
                        index_scores, block_table, request_ids, output,
                        token, width, cutoff, INDEX_TOPK - above,
                    )
            topk_tid = pl.yield_(merge_tid)
        return topk_tid


    return select_index_topk


select_index_topk = _make_select_index_topk()
_select_c1a_request_group_topk = _make_select_index_topk(leaf_workers=16, row_workers=16)


@pl.jit.inline(auto_scope=False)
def score_select_c1a_groups(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    scores: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_capacity: pl.Scalar[pl.INDEX],
):
    """Score and select two request-aligned query groups or the full score arena."""
    width = pl.tensor.dim(scores, 1)
    grouped = _use_request_groups(request_ids, lengths, num_tokens, width, C1A_REQUEST_GROUP_SCORE_WIDTH, False)
    block_count = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    # Grouped scoring leaves the caller's scores arena unwritten. Keep grouping
    # disabled when the pool selector needs those scores to rank excess blocks.
    if block_count > pl.min(pool_capacity, CANDIDATE_TOPK_BLOCKS):
        grouped = pl.cast(0, pl.INT32)
    completion = pl.array.create(2, pl.TASK_ID)
    if grouped != 0:
        group_done = pl.array.create(2, pl.TASK_ID)
        split_rows = pl.cast(num_tokens, pl.INDEX) // 2
        for group in pl.unroll(2):
            begin = group * split_rows
            group_rows = split_rows + group * (pl.cast(num_tokens, pl.INDEX) % 2)
            group_query = pl.slice(query, [group_rows, INDEX_H * INDEX_DIM // 2], [begin, 0])
            group_scale = pl.slice(query_scale, [group_rows, INDEX_H * _SCALE_COUNT], [begin, 0])
            group_weights = pl.slice(weights, [group_rows, INDEX_H], [begin, 0])
            group_requests = pl.slice(request_ids, [group_rows], [begin])
            group_lengths = pl.slice(lengths, [group_rows], [begin])
            group_scores = pl.create_tensor([group_rows, width], dtype=pl.FP32)
            score_tid = _score_c1a_request_group(
                group_query, group_scale, group_weights, index_cache, index_cache_scale,
                block_table, group_requests, group_lengths, group_scores,
                pl.cast(group_rows, pl.INT32), cache_ready, query_ready, weights_ready,
            )
            group_output = pl.slice(output, [group_rows, INDEX_TOPK], [begin, 0])
            selected = _select_c1a_request_group_topk(
                group_scores, block_table, group_requests, group_output, pl.cast(group_rows, pl.INT32), score_tid,
            )
            group_done[group] = selected
        joined = pl.system.task_dummy(deps=[group_done[0], group_done[1]])
        completion[0] = joined
        completion[1] = joined
    else:
        scores_tid = score_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, scores,
            num_tokens, cache_ready, query_ready, weights_ready,
        )
        selected = select_index_topk(scores, block_table, request_ids, output, num_tokens, scores_tid)
        completion[0] = selected
        completion[1] = scores_tid
    return completion[0], completion[1]


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
        valid_pages = pl.min(table_width, TOPK_CACHED_PAGES)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        ones = pl.tile.full([1, 256], dtype=pl.FP32, value=1.0)
        lanes = pl.tile.arange(0, [1, 256], dtype=pl.INT32)
        lane_origin = pl.tile.full([1, 256], dtype=pl.INT32, value=256)
        # Unique integer lane keys: selected=256-lane, discarded=0.
        priority_fp32 = pl.cast(pl.sub(lane_origin, lanes), pl.FP32)
        tmp = pl.tile.create([1, 256], dtype=pl.INT32)
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
                    eq_valid = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(valid, pl.INT32), pl.FP32), cmp_type=2)
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
                            ties = pl.tile.select(pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(boundary, pl.INT32), pl.FP32), cmp_type=3), eq, 0.0)
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
                    safe_indices = pl.maximum(page_indices, 0)
                    safe_indices = pl.minimum(safe_indices, pl.cast(valid_pages - 1, pl.INT32))
                    page_ids = pl.tile.gather(pages, safe_indices, tmp)
                    offsets = pl.sub(logical, pl.mul(page_indices, _TOPK_PAGE))
                    physical = pl.add(pl.mul(page_ids, _TOPK_PAGE), offsets)
                    pl.store(pl.tile.set_validshape(physical, 1, selected_count), [token, count], output)
                    count = count + selected_count
    return output


@pl.jit.inline
def _map_pool_topk_rows(
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
    compact: pl.Tile[[1, INDEX_TOPK], pl.INT32],
):
    """Restore the cache-row ABI in ascending compact index order."""
    request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
    valid_pages = pl.min(pl.tensor.dim(block_table, 1), TOPK_CACHED_PAGES)
    if request >= 0 and request < pl.tensor.dim(block_table, 0) and valid_pages > 0:
        capacity = pl.tensor.dim(pool_block_ids, 1)
        raw_pool = pl.load(pool_block_ids, [token, 0], [1, 2048], valid_shape=[1, capacity])
        pool = pl.tile.fillpad(raw_pool, pad_value=pl.PadValue.zero)
        raw_pages = pl.load(block_table, [request, 0], [1, TOPK_CACHED_PAGES], valid_shape=[1, valid_pages])
        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
        scratch = pl.tile.create([1, INDEX_TOPK], dtype=pl.INT32)
        compact_safe = pl.maximum(compact, 0)
        slots = pl.tile.shrs(compact_safe, 3)
        blocks = pl.tile.gather(pool, slots, scratch)
        logical = pl.add(pl.mul(blocks, 8), pl.sub(compact_safe, pl.mul(slots, 8)))
        page_indices = pl.tile.shrs(logical, 7)
        safe_pages = pl.minimum(pl.maximum(page_indices, 0), pl.cast(valid_pages - 1, pl.INT32))
        page_ids = pl.tile.gather(pages, safe_pages, scratch)
        rows = pl.add(pl.mul(page_ids, _TOPK_PAGE), pl.sub(logical, pl.mul(page_indices, _TOPK_PAGE)))
        valid = pl.tile.cmps(compact, -1, cmp_type=1)
        physical = pl.tile.select(valid, rows, -1)
        pl.store(physical, [token, 0], output)
    else:
        invalid = pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
        pl.store(invalid, [token, 0], output)


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
    scores_bf16: pl.constexpr = False,
):
    """Select candidate scores and restore the original physical cache-row ABI."""
    width = pl.tensor.dim(index_scores, 1)
    tokens = pl.tensor.dim(index_scores, 0)
    if width <= INDEX_TOPK:
        short_workers = pl.max(1, pl.min(pl.min(num_tokens, 32), pl.system.available_aiv_count()))
        with pl.spmd(short_workers, name_hint="pool_topk_short", deps=[scores_ready]) as short_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, short_workers):
                invalid_bits = pl.tile.full([1, 8], dtype=pl.INT32, value=_TOPK_INVALID_SCORE_BITS)
                invalid_score = pl.tile.read(pl.reinterpret_view(invalid_bits, pl.FP32), [0, 0])
                _materialize_pool_index_topk(
                    index_scores, pool_block_ids, block_table, request_ids, output,
                    token, width, invalid_score, pl.cast(0, pl.INDEX),
                )
        result = pl.yield_(short_tid)
    else:
        leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
        arena_rows = tokens * leaf_count
        pair_arena = pl.create_tensor([arena_rows, _TOPK_PAIR_WIDTH], dtype=pl.FP32)
        total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
        leaf_workers = pl.max(1, pl.min(pl.min(total_leaves, 32), pl.system.available_aiv_count()))
        with pl.spmd(leaf_workers, name_hint="pool_topk_leaf", deps=[scores_ready]) as leaf_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, total_leaves, leaf_workers):
                token = work // leaf_count
                leaf = work - token * leaf_count
                logical_begin = leaf * _TOPK_LEAF_TILE
                valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                output_slot = token * leaf_count + leaf
                if scores_bf16:
                    _sort_pool_topk_leaf_bf16(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
                else:
                    _sort_index_topk_leaf(index_scores, pair_arena, token, logical_begin, valid_count, output_slot)
        merge_workers = pl.max(1, pl.min(pl.min(num_tokens, 32), pl.system.available_aiv_count()))
        with pl.spmd(merge_workers, name_hint="pool_topk_merge", deps=[leaf_tid]) as merge_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, merge_workers):
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
                if scores_bf16 and width <= 16384:
                    root_ids = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    valid = pl.tile.cmps(root_scores, 0.0, cmp_type=4)
                    compact = _order_topk_rows(pl.tile.select(valid, root_ids, -1))
                    _map_pool_topk_rows(pool_block_ids, block_table, request_ids, output, token, compact)
                else:
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
        result = pl.yield_(merge_tid)
    return result


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor,
    weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16(
        source, weight, output, num_tokens, ready,
        width=D, output_width=INDEX_H, n_tile=INDEX_H, k_tile=INDEX_PROJ_K_TILE,
        workers=8, name_hint="proj_idx_w", source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[
        [1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    projection_payload: pl.Tensor,
    projection_scale_backing: pl.Tensor,
):
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    gemm_int8(
        projection_payload, projection_scale_backing, index_wq_b, index_wq_b_scale,
        projected, num_tokens, Q_LORA, INDEX_QUERY_WIDTH, "proj_idx_q", 64, 256, False,
    )
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    query_scale = pl.create_tensor(
        [tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.UINT8
    )
    query_tid = quantize_index_query(projected, cos, sin, query, query_scale, num_tokens, entry_ready)
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(
        x, index_weights_proj, weights, num_tokens, entry_ready
    )
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, query_scale, weights, scores, query_tid, weights_tid


# C2A index-key projection, RMSNorm, and trailing RoPE.


@pl.jit.inline(auto_scope=False)
def c2a_project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=32, name_hint='proj_idx_k', source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    return completed

c2a_normalize_index_key = make_norm_with_deps(INDEX_DIM, name_hint="norm_idx_k")


@pl.jit.inline(auto_scope=False)
def c2a_rotate_index_key(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(pl.min(blocks, 12), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint='rope_idx_k', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _index_key_rope_block(x, cos, sin, output, block)
    return rope_tid


@pl.jit.inline(auto_scope=False)
def prepare_c2a_index_key(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
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
    rotated = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    rotated_tid = c1a_norm_rotate_index_key(
        projected,
        index_norm_weight,
        compressed_rope_cos,
        compressed_rope_sin,
        rotated,
        num_tokens,
        projected_tid,
    )
    return rotated, rotated_tid


@pl.jit.inline
def _write_c2a_short_topk(
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
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
                valid = pl.tile.cmps(pl.cast(lanes, pl.FP32), pl.cast(pl.cast(valid_count, pl.INT32), pl.FP32), cmp_type=2)
                rows = pl.tile.select(valid, physical, invalid)
        pl.store(rows, [token, start], output)
    return output


@pl.jit.inline
def _order_topk_rows(rows: pl.Tile[[1, INDEX_TOPK], pl.INT32]):
    """Sort valid row indices and keep -1 padding at the end."""
    keys = pl.tile.select(pl.tile.cmps(pl.cast(rows, pl.FP32), pl.cast(pl.cast(-1, pl.INT32), pl.FP32), cmp_type=4), pl.mul(pl.cast(rows, pl.FP32), -1.0), _TOPK_SORT_FLOOR)
    ordered = pl.tile.sort32(keys, pl.reinterpret_view(rows, pl.UINT32))
    ordered = pl.tile.mrgsort(ordered, block_len=64)
    ordered = pl.tile.mrgsort(ordered, block_len=256)
    final = pl.tile.gather_mask(
        pl.tile.slice(ordered, [1, _TOPK_PAIR_WIDTH], [0, 0]),
        mask_pattern=pl.tile.MaskPattern.P1010,
        output_dtype=pl.INT32,
    )
    return final


@pl.jit.inline
def _sort_c2a_long_topk_physical(
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    token: pl.Scalar[pl.INDEX],
):
    """Keep C2A long-history output in ascending physical-row order."""
    rows = pl.load(output, [token, 0], [1, INDEX_TOPK], target_memory=pl.MemorySpace.Vec)
    final = _order_topk_rows(rows)
    pl.store(final, [token, 0], output)
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_topk_all_long_ready(
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    scores_ready: pl.Scalar[pl.TASK_ID],
):
    return scores_ready


def _make_c2a_short_topk(row_workers):
    @pl.jit.inline(auto_scope=False)
    def short_stage(
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor,
        lengths: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        scores_ready: pl.Scalar[pl.TASK_ID],
    ):
        short_workers = pl.max(1, pl.min(pl.min(num_tokens, row_workers), pl.system.available_aiv_count()))
        with pl.spmd(short_workers, name_hint="idx_topk_short", deps=[scores_ready]) as short_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, short_workers):
                if pl.read(lengths, [token]) <= INDEX_TOPK:
                    _write_c2a_short_topk(block_table, request_ids, lengths, output, token)
        return short_tid

    return short_stage


@pl.jit.inline
def _merge_c2a_pair_tiles(
    left: pl.Tile[[1, _TOPK_PAIR_WIDTH], pl.FP32],
    right: pl.Tile[[1, _TOPK_PAIR_WIDTH], pl.FP32],
):
    scratch = pl.tile.create([1, 2 * _TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged = pl.tile.mrgsort(left, right, tmp=scratch)
    return pl.tile.slice(merged, [1, _TOPK_PAIR_WIDTH], [0, 0])


@pl.jit.inline
def _merge_c2a_small_history(
    pairs: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    leaf_count: pl.Scalar[pl.INDEX],
):
    # Each leaf stores its best INDEX_TOPK pairs. Keep the intermediate
    # pairs in Vector memory when a history fits in at most four leaves.
    first = pl.load(pairs, [token, 0], [1, _TOPK_PAIR_WIDTH])
    root = pl.tile.slice(first, [1, _TOPK_PAIR_WIDTH], [0, 0])
    if leaf_count >= 2:
        second = pl.load(pairs, [token, _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH])
        root = _merge_c2a_pair_tiles(root, second)
    if leaf_count >= 3:
        third = pl.load(pairs, [token, 2 * _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH])
        tail = pl.tile.slice(third, [1, _TOPK_PAIR_WIDTH], [0, 0])
        if leaf_count == 4:
            fourth = pl.load(pairs, [token, 3 * _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH])
            tail = _merge_c2a_pair_tiles(tail, fourth)
        root = _merge_c2a_pair_tiles(root, tail)
    return root


def _make_select_c2a_index_topk(leaf_workers=32, row_workers=32, all_long=False, stable_bf16=False):
    sort_leaf, merge_rows = _make_index_topk_pairs(query_rows=True, stable_bf16=stable_bf16)
    materialize = _make_materialize_index_topk()
    short_stage = _c2a_topk_all_long_ready if all_long or stable_bf16 else _make_c2a_short_topk(row_workers)

    @pl.jit.inline(auto_scope=False)
    def select_c2a_index_topk(
        index_scores: pl.Tensor,
        block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        request_ids: pl.Tensor,
        lengths: pl.Tensor,
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        scores_ready: pl.Scalar[pl.TASK_ID],
    ):
        """Map short histories directly and return long-history TopK in physical-row order."""
        width = pl.tensor.dim(index_scores, 1)
        tokens = pl.tensor.dim(index_scores, 0)
        leaf_count = (width + _TOPK_LEAF_TILE - 1) // _TOPK_LEAF_TILE
        # Preserve a directly visible query dimension in each generated wrapper.
        pair_storage = pl.create_tensor([tokens, leaf_count * _TOPK_PAIR_WIDTH], dtype=pl.FP32)

        short_tid = short_stage(block_table, request_ids, lengths, output, num_tokens, scores_ready)

        total_leaves = pl.cast(num_tokens, pl.INDEX) * leaf_count
        sort_workers = pl.max(1, pl.min(pl.min(total_leaves, leaf_workers), pl.system.available_aiv_count()))
        with pl.spmd(sort_workers, name_hint="idx_topk_leaf", deps=[scores_ready]) as leaf_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(worker, total_leaves, sort_workers):
                token = work // leaf_count
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    leaf = work - token * leaf_count
                    logical_begin = leaf * _TOPK_LEAF_TILE
                    valid_count = pl.min(_TOPK_LEAF_TILE, width - logical_begin)
                    sort_leaf(index_scores, pair_storage, token, logical_begin, valid_count, leaf)

        merge_workers = pl.max(1, pl.min(pl.min(num_tokens, row_workers), pl.system.available_aiv_count()))
        with pl.spmd(merge_workers, name_hint="idx_topk_merge", deps=[short_tid, leaf_tid]) as merge_tid:
            worker = pl.tile.get_block_idx()
            for token in pl.range(worker, num_tokens, merge_workers):
                if pl.read(lengths, [token]) > INDEX_TOPK:
                    if stable_bf16 and leaf_count <= 4:
                        root = _merge_c2a_small_history(pair_storage, token, leaf_count)
                    else:
                        count = leaf_count
                        for _ in pl.range(21):
                            if count > 1:
                                output_count = (count + 1) // 2
                                for output_slot in pl.range(output_count):
                                    left = output_slot * 2
                                    right = left + 1
                                    if right < count:
                                        merge_rows(pair_storage, left, right, output_slot, token)
                                    else:
                                        forwarded = pl.load(
                                            pair_storage, [token, left * _TOPK_PAIR_WIDTH], [1, _TOPK_PAIR_WIDTH]
                                        )
                                        pl.store(forwarded, [token, output_slot * _TOPK_PAIR_WIDTH], pair_storage)
                                count = output_count
                        loaded_root = pl.load(pair_storage, [token, 0], [1, _TOPK_PAIR_WIDTH])
                        root = pl.tile.slice(loaded_root, [1, _TOPK_PAIR_WIDTH], [0, 0])
                    root_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101)
                    if stable_bf16 and width <= 16384:
                        valid = pl.tile.cmps(root_scores, 0.0, cmp_type=4)
                        root_ids = pl.tile.gather_mask(
                            root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32,
                        )
                        logical = pl.tile.select(valid, root_ids, 0)
                        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
                        page_count = pl.min(pl.tensor.dim(block_table, 1), 128)
                        raw_pages = pl.load(block_table, [request, 0], [1, 128], valid_shape=[1, page_count])
                        pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
                        page_offsets = pl.tile.shrs(logical, 7)
                        scratch = pl.tile.create([1, INDEX_TOPK], dtype=pl.INT32)
                        page_ids = pl.tile.gather(pages, page_offsets, scratch)
                        physical = pl.add(pl.mul(page_ids, 128), pl.sub(logical, pl.mul(page_offsets, 128)))
                        ordered = _order_topk_rows(pl.tile.select(valid, physical, -1))
                        pl.store(ordered, [token, 0], output)
                    else:
                        cutoff = pl.tile.read(root_scores, [0, INDEX_TOPK - 1])
                        root_ones = pl.tile.full([1, INDEX_TOPK], dtype=pl.FP32, value=1.0)
                        above_flags = pl.tile.select(pl.tile.cmps(root_scores, cutoff, cmp_type=4), root_ones, 0.0)
                        sums64 = pl.col_sum(pl.reshape(above_flags, [8, 64]))
                        sums8 = pl.col_sum(pl.reshape(sums64, [8, 8]))
                        above = pl.cast(0, pl.INDEX)
                        for lane in pl.unroll(8):
                            above = above + pl.cast(pl.tile.read(sums8, [0, lane]), pl.INT32)
                        materialize(
                            index_scores, block_table, request_ids, output,
                            token, width, cutoff, INDEX_TOPK - above,
                        )
                elif stable_bf16 and not all_long:
                    _write_c2a_short_topk(block_table, request_ids, lengths, output, token)

        if stable_bf16 and width <= 16384:
            result = pl.yield_(merge_tid)
        else:
            order_workers = pl.max(1, pl.min(pl.min(num_tokens, row_workers), pl.system.available_aiv_count()))
            with pl.spmd(order_workers, name_hint="idx_topk_order", deps=[merge_tid]) as order_tid:
                order_worker = pl.tile.get_block_idx()
                for token in pl.range(order_worker, num_tokens, order_workers):
                    if pl.read(lengths, [token]) > INDEX_TOPK:
                        _sort_c2a_long_topk_physical(output, token)
            result = pl.yield_(order_tid)
        return result

    return select_c2a_index_topk


select_c2a_index_topk = _make_select_c2a_index_topk()
_select_c2a_request_group_topk = _make_select_c2a_index_topk(leaf_workers=16, row_workers=16, all_long=True, stable_bf16=True)

select_c2a_stable_bf16_topk = _make_select_c2a_index_topk(stable_bf16=True)


@pl.jit.inline(auto_scope=False)
def _score_select_c2a_groups(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.UINT8],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.UINT8],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
):
    """Select C2A rows with independent score and TopK groups."""
    width = pl.tensor.dim(scores, 1)
    request_grouped = _use_request_groups(request_ids, lengths, num_tokens, width, C2A_REQUEST_GROUP_SCORE_WIDTH, True)
    if request_grouped != 0:
        group_done = pl.array.create(2, pl.TASK_ID)
        split_rows = pl.cast(num_tokens, pl.INDEX) // 2
        for group in pl.unroll(2):
            begin = group * split_rows
            group_rows = split_rows + group * (pl.cast(num_tokens, pl.INDEX) % 2)
            group_query = pl.slice(query, [group_rows, INDEX_H * INDEX_DIM // 2], [begin, 0])
            group_scale = pl.slice(query_scale, [group_rows, INDEX_H * _SCALE_COUNT], [begin, 0])
            group_weights = pl.slice(weights, [group_rows, INDEX_H], [begin, 0])
            group_requests = pl.slice(request_ids, [group_rows], [begin])
            group_lengths = pl.slice(lengths, [group_rows], [begin])
            group_scores = pl.create_tensor([group_rows, width], dtype=pl.FP32)
            score_tid = _score_c2a_request_group(
                group_query, group_scale, group_weights, index_cache, index_cache_scale,
                block_table, group_requests, group_lengths, group_scores,
                pl.cast(group_rows, pl.INT32), cache_ready, query_ready, weights_ready,
            )
            group_output = pl.slice(output, [group_rows, INDEX_TOPK], [begin, 0])
            selected = _select_c2a_request_group_topk(
                group_scores, block_table, group_requests, group_lengths, group_output, pl.cast(group_rows, pl.INT32), score_tid,
            )
            group_done[group] = selected
        result = pl.system.task_dummy(deps=[group_done[0], group_done[1]])
    else:
        scores_tid = score_c2a_full(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, scores,
            num_tokens, cache_ready, query_ready, weights_ready,
        )
        result = select_c2a_stable_bf16_topk(
            scores, block_table, request_ids, lengths, output, num_tokens, scores_tid,
        )
    return result


@pl.jit.inline(auto_scope=False)
def decode_c2a_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    projection_payload: pl.Tensor,
    projection_scales: pl.Tensor,
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // 2)], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // INDEX_CACHE_GROUP)], pl.UINT8],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish C2A keys and select long histories through the shared MX scorer."""
    index_key, index_key_tid = prepare_c2a_index_key(
        latent, index_wk, index_norm_weight,
        compressed_rope_cos, compressed_rope_sin,
        num_tokens, latent_ready,
    )
    index_publish_tid = publish_index_rows(
        index_key, compressed_slots,
        index_cache, index_cache_scale,
        num_tokens, cache_ready, index_key_tid,
    )
    # The dense arena follows logical page-table capacity, never allocated cache pages.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    query, query_scale, raw_weights, scores, query_tid, weights_tid = c1a_index(
        x,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        rope_cos, rope_sin,
        num_tokens, score_width,
        cache_ready,
        projection_payload, projection_scales,
    )
    return _score_select_c2a_groups(
        query, query_scale, raw_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, scores, topk_indices,
        num_tokens, index_publish_tid, query_tid, weights_tid,
    )


import torch
import torch.nn.functional as F
from .qkv_proj_rope import golden_rope_tail, golden_rope_interleave
from .quant import dequantize_mxfp4_cache, quantize_mxfp4_cache
from .gemm import golden_dynamic_int8
from .rmsnorm import golden_rms_norm


def golden_index_query_fp4(value: torch.Tensor) -> torch.Tensor:
    """Round group-32 index queries to E2M1 with upward E8M0 scales."""
    payload, scales = quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True)
    return dequantize_mxfp4_cache(payload, scales, 32, "e8m0").to(value.dtype)


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
    key = golden_rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_select_candidate_block_pool(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = FLASH.candidate_topk_blocks,
    block_size: int = FLASH.candidate_block_size,
    *,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted logical block IDs and counts with a causal tail and fixed selected prefix."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if block_size != CANDIDATE_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANDIDATE_BLOCK_SIZE}, got {block_size}")
    if topk_blocks != CANDIDATE_TOPK_BLOCKS:
        raise ValueError(f"topk_blocks must be {CANDIDATE_TOPK_BLOCKS}, got {topk_blocks}")

    tokens, width = logits.shape
    # Inactive tensor rows publish canonical empty pool state.
    active_tokens = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    capacity = index_pool_block_capacity(width)
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=logits.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=logits.device)
    if width == 0:
        return ids, counts

    lengths = torch.as_tensor(compressed_lens, device=logits.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.ndim != 1 or lengths.numel() != tokens:
        raise ValueError("compressed_lens must provide one causal length per score row")

    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    for token in range(active_tokens):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        selected_count = min(visible_blocks, topk_blocks, capacity)
        # Rank invalid block maxima at the device sort floor.
        scores = block_scores[token, :visible_blocks].clone().clamp_min(-1e30)
        # Include the newest causal block independently of its score.
        scores[visible_blocks - 1] = 1e30
        # Resolve equal block maxima by stable logical order.
        selected = torch.argsort(scores, descending=True, stable=True)[:selected_count].to(torch.int32)
        ids[token, :selected_count] = selected.sort().values
        counts[token] = selected_count
    return ids, counts


def golden_c1a_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from .qkv_proj_rope import golden_rope_interleave as rope_interleave
    from .quant import dequantize_mxfp4_cache

    x = a["x"]
    lengths = a["compressed_lens"]
    # Logical page-table score columns.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_dynamic_int8(qr, a["index_wq_b"], a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    iq = golden_index_query_fp4(iq)
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = ((x).float() @ a["index_weights_proj"].float()).to(torch.bfloat16).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = pages >= 0
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # Device order: BF16-cast the dot, ReLU in FP32, multiply by the rounded weight,
        # BF16-cast the product, sum in FP32, BF16-cast the total.
        dots = ((iq[t]).float() @ (keys[rows.long()].T).float()).to(torch.bfloat16)
        product = dots.float().relu() * weight[t, :, None]
        values = product.to(torch.bfloat16).float().sum(0).to(torch.bfloat16).float()
        scores[t, valid_pos] = values
    return scores, physical


def golden_index_scores(scores, physical, pool_block_ids, pool_block_counts):
    """Pack dense logical scores into the Reindex block-pool score layout."""
    tokens, dense_width = scores.shape
    capacity = pool_block_ids.shape[-1]
    compact_width = capacity * FLASH.candidate_block_size
    compact_scores = torch.full((tokens, compact_width), -torch.inf, dtype=scores.dtype)
    compact_physical = torch.full((tokens, compact_width), -1, dtype=physical.dtype)
    for token in range(tokens):
        count = max(0, min(int(pool_block_counts[token]), capacity))
        for slot in range(count):
            block = int(pool_block_ids[token, slot])
            logical_begin = block * FLASH.candidate_block_size
            if block < 0 or logical_begin >= dense_width:
                continue
            logical_end = min(logical_begin + FLASH.candidate_block_size, dense_width)
            compact_begin = slot * FLASH.candidate_block_size
            compact_end = compact_begin + (logical_end - logical_begin)
            compact_scores[token, compact_begin:compact_end] = scores[token, logical_begin:logical_end]
            compact_physical[token, compact_begin:compact_end] = physical[token, logical_begin:logical_end]
    return compact_scores, compact_physical


def golden_c1a_index_topk(scores, physical, lengths, pool_block_counts=None):
    """Select finite TopK rows with stable ties and ascending physical order."""
    topk = torch.full((scores.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(scores.shape[0]):
        if pool_block_counts is None:
            count = min(int(lengths[token]), scores.shape[-1])
        else:
            count = min(int(pool_block_counts[token]) * FLASH.candidate_block_size, scores.shape[-1])
        chosen = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        chosen = chosen[torch.isfinite(scores[token, chosen])].sort().values
        topk[token, :chosen.numel()] = physical[token, chosen].int()
    return topk


def _select_shared_c2a_topk(scores, physical_rows, lengths):
    """Apply C2A's direct-short and common logical-earliest selection contract."""
    tokens, width = scores.shape
    output = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        count = max(0, min(int(lengths[token]), width))
        if count <= INDEX_TOPK:
            output[token, :count] = physical_rows[token, :count].to(torch.int32)
            continue
        # Stable descending order makes the common logical-earliest cutoff explicit.
        logical = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        logical = logical[torch.isfinite(scores[token, logical])]
        rows = physical_rows[token, logical]
        rows = rows[rows >= 0].sort().values
        output[token, :rows.numel()] = rows.to(torch.int32)
    return output


def golden_c2a_index(t, qr, index_cache, index_cache_scale):
    """Score and select C2A paged history with BF16 head reduction."""
    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    index_values = dequantize_mxfp4_cache(index_cache, index_cache_scale, INDEX_CACHE_GROUP, "e8m0").flatten(0, -2)
    # Accumulate the index-query projection in sequential group-32 partials.
    index_q = golden_dynamic_int8(qr, t["index_wq_b"], t["index_wq_b_scale"]).unflatten(-1, (index_heads, index_dim))
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    index_q = golden_index_query_fp4(index_q)
    head_weights = ((x).float() @ t["index_weights_proj"].float()).to(torch.bfloat16).float()
    head_weights = (head_weights * (index_dim * index_heads) ** -0.5).to(torch.bfloat16).float()
    lens = t["compressed_lens"].long()
    score_width = t["index_block_table"].shape[1] * 128
    scores = torch.full((tokens, score_width), -torch.inf)
    physical_rows = torch.full((tokens, score_width), -1, dtype=torch.int64)
    for token in range(tokens):
        count = max(0, min(int(lens[token]), score_width))
        request = int(t["token_to_req_indices"][token])
        if not count or not 0 <= request < t["index_block_table"].shape[0]:
            continue
        logical = torch.arange(count)
        pages = t["index_block_table"][request, logical // 128].long()
        rows = pages * 128 + logical.remainder(128)
        valid = (pages >= 0) & (rows >= 0) & (rows < index_values.shape[0])
        valid_logical = logical[valid]
        valid_rows = rows[valid]
        physical_rows[token, valid_logical] = valid_rows
        if count <= INDEX_TOPK or not valid_rows.numel():
            continue
        # Round dot, weighted values, and the head sum to BF16.
        dots = ((index_q[token]).float() @ (index_values[valid_rows].to(torch.bfloat16).T).float()).to(torch.bfloat16)
        weighted = dots.float().relu() * head_weights[token, :, None]
        weighted = weighted.to(torch.bfloat16).float()
        total = weighted.sum(dim=0).to(torch.bfloat16).float()
        scores[token, valid_logical] = total
    return _select_shared_c2a_topk(scores, physical_rows, lens)


__all__ = [
    "c1a_index",
    "decode_c2a_indexer",
    "golden_c2a_index",
    "golden_c1a_index_scores",
    "golden_c1a_index_topk",
    "make_index_scores",
    "score_full",
    "score_reindex",
    "score_c2a_full",
    "prepare_c2a_index_key",
    "select_c2a_index_topk",
    "select_pool_index_topk",
    "select_index_topk",
]


@pl.jit.inline(auto_scope=False)
def c1a_project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    result, completed = gemm_bf16_to_bf16(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=8, name_hint='proj_idx_k', source_col=0, output_col=0, ordered=False, row_tile=16, stages=1,
    )
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_rotate_index_key(
    x: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    blocks = num_tokens
    workers = pl.min(pl.min(blocks, 32), pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint='rope_idx_k', deps=[ready]) as rope_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            _index_key_rope_block(x, cos, sin, output, block)
    return rope_tid


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = FLASH.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(FLASH.candidate_topk_blocks, aligned)


@pl.jit(auto_scope=False)
def l2_decode_indexer_validation(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query_weight: pl.Tensor[[Q_LORA, INDEX_QUERY_WIDTH], pl.INT8],
    query_weight_scale: pl.Tensor[[1, INDEX_QUERY_WIDTH], pl.FP32],
    weight_projection: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, 4], pl.UINT8],
    table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    lengths: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    query: pl.InOut[pl.Tensor[[T_DYN, INDEX_QUERY_WIDTH // 2], pl.UINT8]],
    query_scale: pl.InOut[pl.Tensor[[T_DYN, INDEX_QUERY_SCALE_WIDTH], pl.UINT8]],
    weights: pl.InOut[pl.Tensor[[T_DYN, INDEX_H], pl.BF16]],
    scores: pl.InOut[pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32]],
    c1a: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    c2a: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
):
    entry = pl.system.task_dummy(deps=[])
    projected = pl.create_tensor([pl.tensor.dim(x, 0), INDEX_QUERY_WIDTH], dtype=pl.BF16)
    gemm_dynamic_int8(
        latent, query_weight, query_weight_scale, projected, num_tokens,
        Q_LORA, INDEX_QUERY_WIDTH, "proj_idx_q", False, False,
    )
    query_tid = quantize_index_query(projected, cos, sin, query, query_scale, num_tokens, entry)
    weights_tid = project_index_weights(x, weight_projection, weights, num_tokens, entry)
    scored = score_full(
        query, query_scale, weights, cache, cache_scale, table,
        request_ids, lengths, scores, num_tokens, entry, query_tid, weights_tid,
    )
    select_index_topk(scores, table, request_ids, c1a, num_tokens, scored)
    select_c2a_index_topk(scores, table, request_ids, lengths, c2a, num_tokens, scored)
    select_candidate_block_pool(scores, lengths, pool_ids, pool_counts, num_tokens, scored)
    return query, query_scale, weights, scores, c1a, c2a, pool_ids, pool_counts


def golden_decode_indexer_validation(tensors):
    """Check packed query bytes, page-mapped scores, stable TopK and block pools."""
    from .indexer_reference import indexer_reference

    reference = indexer_reference(tensors)
    active = int(tensors["num_tokens"])
    for target, source in (("query", "codes"), ("query_scale", "scales"), ("weights", "raw_weights")):
        tensors[target][:active] = reference[source][:active]
    tensors["scores"][:active] = reference["scores"][:active]
    tensors["c1a"][:active] = reference["c1a"][:active]
    tensors["c2a"][:active] = reference["c2a"][:active]
    ids, counts = golden_select_candidate_block_pool(reference["scores"], tensors["lengths"], num_tokens=active)
    tensors["pool_ids"][:active] = ids[:active]
    tensors["pool_counts"][:active] = counts[:active]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from .indexer_reference import compare_index_scores, make_index_case
    from .weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--history", type=int, default=640)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.history < 16:
        parser.error("--history must be at least 16")
    torch.set_num_threads(8)
    checkpoint = Checkpoint(args.checkpoint) if args.checkpoint is not None else None
    records = []
    for count, ties in ((1, False), (12, False), (12, True)):
        values = make_index_case(checkpoint, args.layer, args.history, ties=ties, num_tokens=count)
        width = values["table"].shape[1] * 128
        capacity = index_pool_block_capacity(width)
        values.update(query=torch.full((16, INDEX_QUERY_WIDTH // 2), 99, dtype=torch.uint8),
                      query_scale=torch.full((16, INDEX_QUERY_SCALE_WIDTH), 11, dtype=torch.uint8),
                      weights=torch.full((16, INDEX_H), 7.25, dtype=torch.bfloat16),
                      scores=torch.full((16, width), 23.5),
                      c1a=torch.full((16, INDEX_TOPK), -1, dtype=torch.int32),
                      c2a=torch.full((16, INDEX_TOPK), -1, dtype=torch.int32),
                      pool_ids=torch.full((16, capacity), -1, dtype=torch.int32),
                      pool_counts=torch.zeros(16, dtype=torch.int32))
        specs = [ScalarSpec(name, torch.int32, value) if name == "num_tokens"
                 else TensorSpec(name, list(value.shape), value.dtype, init_value=value)
                 for name, value in values.items()]
        exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
        comparisons = {name: exact for name in ("query", "query_scale", "c1a", "c2a", "pool_ids", "pool_counts")}
        comparisons.update(weights=ratio_allclose(atol=0.003, rtol=1.0 / 128.0, max_error_ratio=0.0),
                           scores=compare_index_scores)
        result = run(
            l2_decode_indexer_validation,
            specs,
            golden_fn=golden_decode_indexer_validation,
            config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device),
            compare_fn=comparisons,
            compile_only=args.compile_only,
            save_data=checkpoint is not None,
        )
        records.append(dict(num_tokens=count, ties=ties, history=args.history,
                            passed=result.passed, work_dir=str(result.work_dir)))
        if not result.passed:
            raise RuntimeError(result.error)
    if args.report:
        args.report.write_text(json.dumps(dict(checkpoint=checkpoint.identity if checkpoint else None,
                                              cases=records), indent=2) + "\n")
