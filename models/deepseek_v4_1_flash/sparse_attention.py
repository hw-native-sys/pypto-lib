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

import torch

from config import AttentionMode
from compressor import golden_compressor_ratio1 as compressor_ratio1
from compressor import golden_compressor_ratio2_paged as compressor_ratio2_paged
from hierarchical_sparse_indexer import golden_index_key as index_key
from hierarchical_sparse_indexer import golden_paged_indexer as paged_indexer
from hierarchical_sparse_indexer import select_candidate_blocks
from qkv_proj_rope import golden_qkv_proj_rope as qkv_proj_rope
from qkv_proj_rope import golden_rope_interleave as rope_interleave
from o_proj import _project_output
from quant import (
    _publish_quantized_cache,
    dequantize_mxfp4_cache,
    dequantize_mxfp8_cache,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
)
from utils import AttentionGoldenResult

# ci: no-sim

from pathlib import Path

import pypto.language as pl

from config import (
    FLASH as M, CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, TP_SIZE,
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
INDEX_TOPK = M.index_topk
LOCAL_H = M.num_attention_heads // TP_SIZE
SELECTED_KEYS = 128 + INDEX_TOPK
SCALE = HEAD_DIM**-0.5
FP4_FP16_BASE = 0x3800
FP4_FP16_SHIFT = 9
FP4_SIGN_SHIFT = 12

# Tiling.
CUBE_HEADS = max(16, LOCAL_H)
HEAD_TILE = min(32, CUBE_HEADS)
QUERY_HEAD_TILE = min(16, LOCAL_H)
KEY_TILE = 128
GATHER_KEY_TILE = 32
TOKEN_TILE = 512
WORKERS = 32
GATHER_WORKERS = 64
SELECTED_GATHER_TILE = 64
CACHE_WORKERS = 32
SHARED_GATHER_WORKERS = 64
CACHE_MAX_ROWS = 32768


M_TILE = 16

ATTENTION_TILE = 32

def golden_prefill_c1a_attention(query: torch.Tensor, window_cache: torch.Tensor, window_indices: torch.Tensor, compressed_cache: torch.Tensor, compressed_indices: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """Mirror C1A tiled online softmax, BF16 PV, and its FP32 first-vector patch."""
    import torch
    output = torch.empty_like(query)
    width = query.shape[-1]
    for token in range(query.shape[0]):
        q = query[token].float()
        maximum = q.new_full((q.shape[0],), -1e+30)
        denominator = q.new_zeros(q.shape[0])
        numerator = torch.zeros_like(q)
        for cache, indices in ((window_cache, window_indices), (compressed_cache, compressed_indices)):
            flat_cache = cache.flatten(0, 1).squeeze(-2)
            for begin in range(0, indices.shape[-1], ATTENTION_TILE):
                rows = indices[token, begin:begin + ATTENTION_TILE].long()
                valid = rows >= 0
                if not bool(valid.any()):
                    continue
                kv = flat_cache[rows.clamp_min(0)].clone()
                kv[~valid] = 0
                scores = torch.matmul(q, kv.float().T) * width ** (-0.5)
                scores = scores + (valid.float() - 1).unsqueeze(0) * 1e+30
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = torch.exp(maximum - next_maximum)
                probability = torch.exp(scores - next_maximum.unsqueeze(-1)) * valid.unsqueeze(0)
                denominator = denominator * correction + probability.sum(dim=-1)
                weighted = torch.matmul(probability.to(torch.bfloat16).float(), kv.float())
                weighted[::M_TILE, :16] = (probability[::M_TILE].unsqueeze(-1) * kv[:, :16].float().unsqueeze(0)).sum(dim=1)
                numerator = numerator * correction.unsqueeze(-1) + weighted
                maximum = next_maximum
        final_maximum = torch.maximum(maximum, sink.float())
        correction = torch.exp(maximum - final_maximum)
        denominator = denominator * correction + torch.exp(sink.float() - final_maximum)
        output[token] = (numerator * (correction / denominator).unsqueeze(-1)).to(query.dtype)
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
    workers = pl.max(1, pl.min(num_tokens * (SELECTED_KEYS // GATHER_KEY_TILE), GATHER_WORKERS))
    with pl.spmd(workers, name_hint="sparse_packed_gather", deps=[ready]) as gathered:
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
                c_signed = pl.reinterpret_view(c_payload, pl.INT8)
                c_wide = pl.cast(c_signed, pl.INT16)
                byte_values = pl.ands(c_wide, 255)
                low = pl.ands(byte_values, 15)
                high = pl.shrs(byte_values, 4)
                # E2M1 magnitudes map exactly to FP16 exponent and mantissa bits.
                low_code = pl.ands(low, 7)
                low_exponent = pl.shls(low_code, FP4_FP16_SHIFT)
                low_base = pl.add(low_exponent, FP4_FP16_BASE)
                low_subnormal = pl.minimum(pl.sub(low_code, 2), 0)
                low_adjust = pl.shls(low_subnormal, FP4_FP16_SHIFT)
                low_magnitude = pl.add(low_base, low_adjust)
                low_nonzero = pl.minimum(low_code, 1)
                low_bits = pl.mul(low_magnitude, low_nonzero)
                low_sign_code = pl.ands(low, 8)
                low_sign = pl.shls(low_sign_code, FP4_SIGN_SHIFT)
                low_signed = pl.or_(low_bits, low_sign)
                low_values = pl.reinterpret_view(low_signed, pl.FP16)
                high_code = pl.ands(high, 7)
                high_exponent = pl.shls(high_code, FP4_FP16_SHIFT)
                high_base = pl.add(high_exponent, FP4_FP16_BASE)
                high_subnormal = pl.minimum(pl.sub(high_code, 2), 0)
                high_adjust = pl.shls(high_subnormal, FP4_FP16_SHIFT)
                high_magnitude = pl.add(high_base, high_adjust)
                high_nonzero = pl.minimum(high_code, 1)
                high_bits = pl.mul(high_magnitude, high_nonzero)
                high_sign_code = pl.ands(high, 8)
                high_sign = pl.shls(high_sign_code, FP4_SIGN_SHIFT)
                high_signed = pl.or_(high_bits, high_sign)
                high_values = pl.reinterpret_view(high_signed, pl.FP16)
                # Each mask scatter zero-fills its unused lanes.
                low_lanes = pl.tile.full([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.FP16, value=0.0)
                high_lanes = pl.tile.full([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.FP16, value=0.0)
                low_lanes = pl.tile.scatter_mask(low_lanes, low_values, pl.tile.MaskPattern.P0101)
                high_lanes = pl.tile.scatter_mask(high_lanes, high_values, pl.tile.MaskPattern.P1010)
                low_lane_bits = pl.reinterpret_view(low_lanes, pl.INT16)
                high_lane_bits = pl.reinterpret_view(high_lanes, pl.INT16)
                c_value_bits = pl.or_(low_lane_bits, high_lane_bits)
                c_values = pl.reinterpret_view(c_value_bits, pl.FP16)
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
def _attend_cube(
    query: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[SELECTED_ROWS, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[CUBE_HEADS], pl.FP32],
    output: pl.Tensor[[T_DYN, CUBE_HEADS * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    selected_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * CUBE_HEADS
    cache_rows = pl.tensor.dim(selected, 0) * SELECTED_KEYS
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    workers = pl.max(1, pl.min(num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), WORKERS))
    with pl.spmd(workers, name_hint="sparse_online_cube", deps=[selected_ready, query_ready], optimizations=[pl.cross_core_slot(slot_num=1)]) as attended:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE), workers):
            token = block // ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE)
            head = block % ((CUBE_HEADS + HEAD_TILE - 1) // HEAD_TILE) * HEAD_TILE
            q0 = token * CUBE_HEADS + head
            q = qflat[q0:q0 + HEAD_TILE, :]
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
    query_ready: pl.Scalar[pl.TASK_ID],
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
    with pl.spmd(workers, name_hint="sparse_pad_heads", deps=[query_ready]) as padded:
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
        num_tokens, selected_ready, padded,
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
    with pl.spmd(workers_q, name_hint="sparse_bf16_query", deps=[previous]) as query_ready:
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
    attended = attend_selected(qbf16, selected, indices, sink, result, rows, selected_ready, query_ready)
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
            c_signed = pl.reinterpret_view(c_payload, pl.INT8)
            c_wide = pl.cast(c_signed, pl.INT16)
            byte_values = pl.ands(c_wide, 255)
            low = pl.ands(byte_values, 15)
            high = pl.shrs(byte_values, 4)
            # E2M1 magnitudes map exactly to FP16 exponent and mantissa bits.
            low_code = pl.ands(low, 7)
            low_exponent = pl.shls(low_code, FP4_FP16_SHIFT)
            low_base = pl.add(low_exponent, FP4_FP16_BASE)
            low_subnormal = pl.minimum(pl.sub(low_code, 2), 0)
            low_adjust = pl.shls(low_subnormal, FP4_FP16_SHIFT)
            low_magnitude = pl.add(low_base, low_adjust)
            low_nonzero = pl.minimum(low_code, 1)
            low_bits = pl.mul(low_magnitude, low_nonzero)
            low_sign_code = pl.ands(low, 8)
            low_sign = pl.shls(low_sign_code, FP4_SIGN_SHIFT)
            low_signed = pl.or_(low_bits, low_sign)
            low_values = pl.reinterpret_view(low_signed, pl.FP16)
            high_code = pl.ands(high, 7)
            high_exponent = pl.shls(high_code, FP4_FP16_SHIFT)
            high_base = pl.add(high_exponent, FP4_FP16_BASE)
            high_subnormal = pl.minimum(pl.sub(high_code, 2), 0)
            high_adjust = pl.shls(high_subnormal, FP4_FP16_SHIFT)
            high_magnitude = pl.add(high_base, high_adjust)
            high_nonzero = pl.minimum(high_code, 1)
            high_bits = pl.mul(high_magnitude, high_nonzero)
            high_sign_code = pl.ands(high, 8)
            high_sign = pl.shls(high_sign_code, FP4_SIGN_SHIFT)
            high_signed = pl.or_(high_bits, high_sign)
            high_values = pl.reinterpret_view(high_signed, pl.FP16)
            # Each mask scatter zero-fills its unused lanes.
            low_lanes = pl.tile.full([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.FP16, value=0.0)
            high_lanes = pl.tile.full([GATHER_KEY_TILE, HEAD_DIM], dtype=pl.FP16, value=0.0)
            low_lanes = pl.tile.scatter_mask(low_lanes, low_values, pl.tile.MaskPattern.P0101)
            high_lanes = pl.tile.scatter_mask(high_lanes, high_values, pl.tile.MaskPattern.P1010)
            low_lane_bits = pl.reinterpret_view(low_lanes, pl.INT16)
            high_lane_bits = pl.reinterpret_view(high_lanes, pl.INT16)
            c_value_bits = pl.or_(low_lane_bits, high_lane_bits)
            c_values = pl.reinterpret_view(c_value_bits, pl.FP16)
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
        previous = attend_selected(q, selected, indices, sink, out, count, gathered, previous)
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
        window_indices, compressed_indices, selected, indices, num_tokens, query_ready,
    )
    attended = attend_selected(query, selected, indices, sink, output, num_tokens, gathered, query_ready)
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
):
    marker = pl.create_tensor([1, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="sparse_query_ready") as ready:
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
def sparse_mla(
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
    output = attend_packed(
        query, window_indices, window, window_scale, compressed_indices,
        compressed, compressed_scale, sink, output, rows,
    )
    return output


@pl.jit(auto_scope=False)
def sparse_mla_bf16(
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


def main():
    import argparse

    import torch

    from golden import TensorSpec, run
    from quant import (
        dequantize_mxfp4_cache, dequantize_mxfp8_cache,
        quantize_mxfp4_cache, quantize_mxfp8_cache,
    )

    parser = argparse.ArgumentParser(description="Validate sparse MLA on Ascend950")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--mode", default="both", choices=["packed", "bf16", "both"])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--runtime-dir", type=str)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        raise ValueError(f"--tp {args.tp} does not match configured TP_SIZE {TP_SIZE}")
    if args.runtime_dir and args.mode == "both":
        raise ValueError("--runtime-dir requires one --mode")
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    fixture = torch.load(args.fixture, weights_only=True) if args.fixture else None
    if fixture is None:
        torch.manual_seed(4102026)
        query = (torch.randn(args.rows, LOCAL_H, HEAD_DIM) * 0.7).bfloat16()
        window = (torch.randn(256, HEAD_DIM) * 0.6).bfloat16()
        compressed = (torch.randn(512, HEAD_DIM) * 0.6).bfloat16()
        window_indices = torch.randint(0, 256, (args.rows, 128), dtype=torch.int32)
        compressed_indices = torch.randint(0, 512, (args.rows, INDEX_TOPK), dtype=torch.int32)
        window_indices[0] = -1
        compressed_indices[0] = -1
        sink = torch.linspace(-2, 3, LOCAL_H)
        extents = torch.tensor([128, INDEX_TOPK], dtype=torch.int32).repeat(args.rows, 1)
        wp, ws = quantize_mxfp8_cache(window.reshape(-1, 128, 1, HEAD_DIM))
        cp, cs = quantize_mxfp4_cache(compressed.reshape(-1, 128, 1, HEAD_DIM), 16, "bf16")
        wd = dequantize_mxfp8_cache(wp, ws).bfloat16().reshape(-1, HEAD_DIM)
        cd = dequantize_mxfp4_cache(cp, cs, 16, "bf16").bfloat16().reshape(-1, HEAD_DIM)
    else:
        query = fixture["q"]
        if list(query.shape) != [args.rows, LOCAL_H, HEAD_DIM]:
            raise ValueError("Fixture query shape does not match --rows and --tp")
        window, compressed = fixture["w"], fixture["c"]
        window_indices, compressed_indices = fixture["wi"], fixture["ci"]
        sink, extents = fixture["sink"], fixture["extents"]
        wp, ws, cp, cs = fixture["wp"], fixture["ws"], fixture["cp"], fixture["cs"]

    modes = ["packed", "bf16"] if args.mode == "both" else [args.mode]
    records = []
    for mode in modes:
        if fixture is None:
            wkeys, ckeys = (wd, cd) if mode == "packed" else (window, compressed)
            expected = torch.empty_like(query, dtype=torch.float32)
            for row in range(args.rows):
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
            expected = fixture[f"expected_{mode}"]
        if mode == "packed":
            inputs = dict(query=query.flatten(1), window_indices=window_indices, window=wp, window_scale=ws,
                          compressed_indices=compressed_indices, compressed=cp, compressed_scale=cs, sink=sink)
            reference = expected.flatten(1)
            result_dtype = torch.bfloat16
            entry = sparse_mla
        else:
            inputs = dict(query=query.float(), window=window.float().reshape(-1, 128, HEAD_DIM),
                          window_indices=window_indices, compressed=compressed.float().reshape(-1, 128, HEAD_DIM),
                          compressed_indices=compressed_indices, sink=sink, extents=extents)
            reference = expected
            result_dtype = torch.float32
            entry = sparse_mla_bf16
        specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in inputs.items()]
        specs.append(TensorSpec("output", list(reference.shape), result_dtype))

        def golden(tensors):
            tensors["output"].copy_(reference.bfloat16().to(result_dtype))

        def compare(actual, _expected, **_kwargs):
            difference = (actual.float() - reference).abs()
            norm = float(reference.norm())
            relative_l2 = float(difference.norm()) / norm if norm else float(difference.norm())
            bad = int((difference > 0.003 + 0.03 * reference.abs()).sum())
            finite = bool(actual.isfinite().all())
            return finite and relative_l2 < 0.006 and bad == 0, f"relative_l2={relative_l2:.6f}, bad={bad}"

        result = run(
            entry,
            specs,
            golden_fn=golden,
            compare_fn={"output": compare},
            config=dict(platform=args.platform, device_id=args.device),
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            save_data=True,
        )
        records.append(dict(mode=mode, passed=result.passed, work_dir=str(result.work_dir)))
        if not result.passed:
            raise RuntimeError(result.error)
        status = "COMPILE_ONLY" if args.compile_only else "DEVICE_VALIDATED"
        print(status, mode, result.work_dir)
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
    wo_b_scale: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
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
    output = _project_output(attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale)
    return AttentionGoldenResult(
        output, window_payload, updated_window_scale, None, None, None, None, None, None, None
    )



def golden_compressed_attention(
    mode: AttentionMode,
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
    request_ids: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
    attention_fn: Callable | None = None,
    output_dtype: torch.dtype | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> AttentionGoldenResult:
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

    if mode is AttentionMode.FULL:
        if compressor_wkv is None or compressor_norm_weight is None or compressed_slots is None:
            raise ValueError("full mode requires compressor weights and compressed slots")
        if compressed_rope_cos is None or compressed_rope_sin is None or quantized_index is None:
            raise ValueError("full mode requires compressed RoPE rows and an index cache")
        if ratio == 1:
            latent = compressor_ratio1(x, compressor_wkv, compressor_norm_weight)
            publish_mask = compressed_slots >= 0
        elif ratio == 2:
            if compressor_wgate is None or state_block_table is None or updated_state is None:
                raise ValueError("ratio-2 full mode requires gate weights and recurrent state")
            if position_ids is None or query_start_loc is None or request_ids is None:
                raise ValueError("ratio-2 full mode requires positions, query starts and token-to-request indices")
            latent, publish_mask = compressor_ratio2_paged(
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
            lambda value: quantize_mxfp4_cache(value, group_size=16, scale_format="bf16"),
        )
        quantized_compressed = dequantize_mxfp4_cache(
            compressed_payload,
            updated_compressed_scale,
            group_size=16,
            scale_format="bf16",
        ).to(query.dtype)
        if index_wk is None or index_norm_weight is None:
            raise ValueError("full mode requires index-key weights")
        keys = index_key(
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
            lambda value: quantize_mxfp4_cache(value, group_size=32, scale_format="e8m0"),
        )
        quantized_index = dequantize_mxfp4_cache(
            index_payload,
            updated_index_scale,
            group_size=32,
            scale_format="e8m0",
        ).to(query.dtype)

    if mode in (AttentionMode.FULL, AttentionMode.REINDEX):
        if index_wq_b is None or index_weights_proj is None:
            raise ValueError("indexing modes require query and score weights")
        if (
            quantized_index is None
            or index_block_table is None
            or request_ids is None
            or compressed_lens is None
        ):
            raise ValueError("indexing modes require cache addressing and causal compressed lengths")
        scores, topk_indices = paged_indexer(
            x,
            query_latent,
            request_ids,
            quantized_index,
            index_block_table,
            compressed_lens,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            rope_cos,
            rope_sin,
            candidates=candidate_mask,
            topk=M.index_topk,
        )
        if ratio == 1 and mode is AttentionMode.FULL:
            candidates = select_candidate_blocks(
                scores,
                compressed_lens,
                M.candidate_topk_blocks,
                M.candidate_block_size,
            )
    if topk_indices is None:
        raise ValueError("reuse mode requires published compressed Top-K indices")

    attend = attention_fn or golden_paged_sparse_attention
    attended = attend(
        query, quantized_window, window_indices, quantized_compressed, topk_indices, attn_sink
    )
    output = _project_output(
        attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale, output_dtype=output_dtype
    )
    return AttentionGoldenResult(
        output,
        window_payload,
        updated_window_scale,
        compressed_payload,
        updated_compressed_scale,
        index_payload,
        updated_index_scale,
        updated_state,
        topk_indices,
        candidates,
    )



if __name__ == "__main__":
    _main_results = main()
