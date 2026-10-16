# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C2A full attention."""

import argparse
import math
import os

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, run

from config import (
    FLASH as M,
    MAX_BATCH_PER_DP,
    AttentionMode,
    B_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    Q_START_DYN,
    T_DYN,
    TABLE_DYN,
    STATE_BLOCKS_DYN,
    TP_SIZE,
    BLOCK_SIZE,
    COMPRESSED_CACHE_GROUP,
    DECODE_MAX_TOKENS,
    INDEX_CACHE_GROUP,
    STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)

from compressor import STATE_WIDTH
from config import EP_SIZE


from utils import AttentionGoldenResult
from quant import quantize_grouped_mxfp8_weight
from sparse_attention import (
    PACKED_KEY_TILE as ATTEND_TILE, PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH,
    gather_packed_chunk, golden_compressed_attention, make_attend_selected_chunk,
)
from config import LOCAL_T_DYN
from o_proj import (
    decode_tp_output_all_reduce,
    decode_tp_output_reduce_scatter,
    golden_grouped_output,
)
from compressor import compressor_ratio2
from rmsnorm import EPS, golden_rms_norm
from qkv_proj_rope import (
    K_TILE,
    M_TILE,
    N_TILE,
    make_mx_projection,
    make_rope,
)
from decode_attn_swa import publish_window
from o_proj import _make_o_proj, _project_ob, grouped_output, o_proj
from qkv_proj_rope import golden_rope_tail, kv_proj_rope, q_proj_qr, qkv_proj_rope
from utils import paged_slots, window_metadata
from quant import (
    dequantize_mxfp4_cache,
    golden_bf16_gemm,
    golden_mxfp8_linear,
    golden_mxfp8_quantize,
    pack_mx_b_scale,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
)
from rope_tables import (
    ROPE_ROWS_DYN, materialize_rope_rows, precompute_rope_tables,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
MX_GROUP = M.weight_block_size[0]


# Cache codec. Payloads are packed E2M1, two logical values per byte.
CMP_SCALES = HEAD_DIM // 16
IDX_SCALES = INDEX_DIM // 32
CMP_PACKED = HEAD_DIM // 2
IDX_PACKED = INDEX_DIM // 2
FP4_MAX_INV = 1.0 / 6.0
FP4_SCALE_FLOOR = 2.0**-9
E8M0_FLOOR_BITS = 4194304

# Sparse attention geometry: one sliding window block plus four compressed blocks.
WINDOW = M.sliding_window

# Indexer selection. Candidates are scored one leaf at a time and folded into a
# running Top-K, which keeps the sort width independent of the context length.
SCORE_TILE = 64
LEAF = 2048
LEAF_TILES = LEAF // SCORE_TILE
PAIR_WIDTH = 2 * INDEX_TOPK
SORT_FLOOR = -3.0e38
MASK_BIAS = 1.0e30
COMPRESSOR_K_TILE = 64
COMPRESSOR_GEMM_K_TILE = 256
INDEX_DIM_SCALE = INDEX_DIM**-0.5
INDEX_HEAD_SCALE = INDEX_H**-0.5


def make_wide_rope(heads, width, inverse=False, *, max_workers=None, permute=False):
    """Specialize interleaved RoPE over the trailing ROPE_DIM lanes of each head."""
    sign = -1.0 if inverse else 1.0
    nope = width - ROPE_DIM
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * width], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint="c2a_rope"):
            for block in pl.range(worker, blocks, workers):
                t = block // heads
                h = block % heads
                base = h * width
                if not permute:
                    output[t : t + 1, base : base + nope] = x[t : t + 1, base : base + nope]
                tail = pl.cast(x[t : t + 1, base + nope : base + width], pl.FP32)
                even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                c = cos[t : t + 1, :]
                s = pl.mul(sin[t : t + 1, :], sign)
                re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
                im = pl.add(pl.mul(even, s), pl.mul(odd, c))
                rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
                rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
                rotated_bf16 = pl.cast(rotated, pl.BF16, mode="rint")
                if permute:
                    # Preserve the RoPE BF16 rounding before the lossless lane permutation.
                    source = pl.concat(x[t : t + 1, base : base + nope], rotated_bf16)
                    source_fp32 = pl.cast(source, pl.FP32)
                    even_lanes = pl.gather(source_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
                    odd_lanes = pl.gather(source_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
                    output[t : t + 1, base : base + width // 2] = pl.cast(even_lanes, pl.BF16, mode="rint")
                    output[t : t + 1, base + width // 2 : base + width] = pl.cast(odd_lanes, pl.BF16, mode="rint")
                else:
                    output[t : t + 1, base + nope : base + width] = rotated_bf16
        return output

    return rotate


@pl.jit.inline
def compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Stage three BF16 weight terms with separate high and residual accumulators."""
    high_weight = pl.create_tensor([2 * D, HEAD_DIM], dtype=pl.BF16)
    low_weight = pl.create_tensor([2 * D, HEAD_DIM], dtype=pl.BF16)
    tail_weight = pl.create_tensor([2 * D, HEAD_DIM], dtype=pl.BF16)
    split_blocks = D // COMPRESSOR_K_TILE * (HEAD_DIM // N_TILE)
    with pl.spmd(16, name_hint="c2a_compressor_split") as split_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, split_blocks, 16):
            k0 = block // (HEAD_DIM // N_TILE) * COMPRESSOR_K_TILE
            n0 = block % (HEAD_DIM // N_TILE) * N_TILE
            for stream in pl.unroll(2):
                if stream == 0:
                    weight = wkv[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                else:
                    weight = wgate[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                high = pl.cast(weight, pl.BF16, mode="rint")
                residual = pl.sub(weight, pl.cast(high, pl.FP32))
                low = pl.cast(residual, pl.BF16, mode="rint")
                tail = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
                row = stream * D + k0
                high_weight[row : row + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = high
                low_weight[row : row + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = low
                tail_weight[row : row + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = tail
    output_blocks = (num_tokens + M_TILE - 1) // M_TILE * (HEAD_DIM // N_TILE)
    with pl.spmd(output_blocks * 2, name_hint="c2a_compressor_cube", deps=[split_tid]):
        block = pl.tile.get_block_idx()
        tile = block // 2
        stream = block % 2
        t0 = tile // (HEAD_DIM // N_TILE) * M_TILE
        n0 = tile % (HEAD_DIM // N_TILE) * N_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        high_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        residual_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(D // COMPRESSOR_GEMM_K_TILE):
            k0 = kb * COMPRESSOR_GEMM_K_TILE
            row = stream * D + k0
            a = pl.slice(x, [M_TILE, COMPRESSOR_GEMM_K_TILE], [t0, k0], valid_shape=[rows, COMPRESSOR_GEMM_K_TILE])
            a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
            gemm_high = high_weight[row : row + COMPRESSOR_GEMM_K_TILE, n0 : n0 + N_TILE]
            gemm_low = low_weight[row : row + COMPRESSOR_GEMM_K_TILE, n0 : n0 + N_TILE]
            gemm_tail = tail_weight[row : row + COMPRESSOR_GEMM_K_TILE, n0 : n0 + N_TILE]
            high_acc = pl.matmul_acc(high_acc, a, gemm_high, init_cond=(kb == 0))
            residual_acc = pl.matmul_acc(residual_acc, a, gemm_low, init_cond=(kb == 0))
            residual_acc = pl.matmul_acc(residual_acc, a, gemm_tail)
        result = pl.set_validshape(pl.add(high_acc, residual_acc), rows, N_TILE)
        if stream == 0:
            kv_out[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = result
        else:
            score_out[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = result
    return kv_out, score_out


@pl.jit.inline
def index_key_project(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Derive normalized index keys from the unrotated compressor latent."""
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c2a_index_key", deps=[pool_ready]):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, INDEX_DIM], dtype=pl.FP32)
        for kb in pl.range(HEAD_DIM // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(latent, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            b = weight[k0 : k0 + K_TILE, :]
            acc = pl.matmul_acc(acc, a, b, init_cond=(kb == 0))
        value = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
        square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / INDEX_DIM)
        inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
        gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, INDEX_DIM])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(
            pl.cast(normalized, pl.BF16, mode="rint"), rows, INDEX_DIM
        )
    return output


@pl.jit.inline
def indexer_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Per-head indexer score weights, rounded to BF16 after each reference scaling."""
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="c2a_index_weights"):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, INDEX_H], dtype=pl.FP32)
        for kb in pl.range(D // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(x, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            b = weight[k0 : k0 + K_TILE, :]
            acc = pl.matmul_acc(acc, a, b, init_cond=(kb == 0))
        value = pl.cast(pl.cast(acc, pl.BF16, mode="rint"), pl.FP32)
        value = pl.cast(pl.cast(pl.mul(value, INDEX_DIM_SCALE), pl.BF16, mode="rint"), pl.FP32)
        scaled = pl.cast(pl.mul(value, INDEX_HEAD_SCALE), pl.BF16, mode="rint")
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(scaled, rows, INDEX_H)
    return output


@pl.jit.inline
def publish_compressed(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    scales: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish rotated compressed KV rows as packed E2M1 with group-16 BF16 scales."""
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, CMP_PACKED])
    scale_flat = pl.reshape(scales, [cache_rows, CMP_SCALES])
    with pl.spmd(num_tokens, name_hint="c2a_compressed_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(latent, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            groups = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM * 2 // 16, 16])
            amax = pl.row_max(pl.abs(groups))
            stored = pl.cast(pl.maximum(pl.mul(amax, FP4_MAX_INV), FP4_SCALE_FLOOR), pl.BF16, mode="rint")
            normalized = pl.minimum(
                pl.maximum(pl.row_expand_div(groups, pl.cast(stored, pl.FP32)), -6.0), 6.0
            )
            magnitude = pl.abs(normalized)
            # Nearest E2M1 magnitude index: seven minus the number of table midpoints
            # at or above |v|, which resolves an exact midpoint toward zero.
            step0 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.25), pl.INT32), 1), 31)
            step1 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 0.75), pl.INT32), 1), 31)
            step2 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.25), pl.INT32), 1), 31)
            step3 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 1.75), pl.INT32), 1), 31)
            step4 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 2.5), pl.INT32), 1), 31)
            step5 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 3.5), pl.INT32), 1), 31)
            step6 = pl.shrs(pl.sub(pl.reinterpret_view(pl.sub(magnitude, 5.0), pl.INT32), 1), 31)
            index = pl.add(
                pl.add(pl.add(step0, step1), pl.add(step2, step3)),
                pl.add(pl.add(step4, step5), pl.add(step6, 7)),
            )
            sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
            codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, HEAD_DIM * 2])
            low = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8), 1, CMP_PACKED
            )
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(stored, [1, HEAD_DIM * 2 // 16]), 1, CMP_SCALES
            )
    return publish_tid


@pl.jit.inline
def publish_index_key(
    keys: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish index keys as packed E2M1 with group-32 E8M0 scales."""
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
            index = pl.add(
                pl.add(pl.add(step0, step1), pl.add(step2, step3)),
                pl.add(pl.add(step4, step5), pl.add(step6, 7)),
            )
            sign = pl.shrs(pl.reinterpret_view(normalized, pl.INT32), 31)
            codes = pl.reshape(pl.sub(index, pl.mul(sign, 8)), [1, INDEX_DIM * 8])
            low = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.INT32)
            high = pl.gather(codes, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            packed = pl.add(low, pl.mul(high, 16))
            signed = pl.sub(packed, pl.mul(pl.shrs(packed, 7), 256))
            flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reinterpret_view(pl.cast(signed, pl.INT8), pl.UINT8), 1, IDX_PACKED
            )
            signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
            encoded = pl.reinterpret_view(
                pl.reinterpret_view(pl.cast(signed_exponent, pl.INT8), pl.UINT8), pl.FP8E8M0
            )
            scale_flat[slot : slot + 1, :] = pl.set_validshape(
                pl.reshape(encoded, [1, INDEX_DIM * 8 // 32]), 1, IDX_SCALES
            )
    return publish_tid
@pl.jit.inline
def index_select(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM], pl.BF16],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    leaf_scores: pl.Tensor[[T_DYN, LEAF], pl.FP32],
    leaf_rows: pl.Tensor[[T_DYN, LEAF], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    publish_ready: pl.Scalar[pl.TASK_ID],
):
    """Select up to INDEX_TOPK compressed cache rows per token.

    A token that can reach at most INDEX_TOPK positions takes all of them and
    never scores. Longer histories are scored one LEAF-sized block at a time and
    folded into a running Top-K, so the sort width does not grow with context.
    The sort carries physical cache rows, which removes a per-entry block-table
    lookup from the result.
    """
    query_rows = pl.tensor.dim(query, 0) * INDEX_H
    query_flat = pl.reshape(query, [query_rows, INDEX_DIM])
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, IDX_PACKED])
    # Four E8M0 codes per row is below the 32-byte tile row, so the scale pool is
    # read eight cache rows at a time and reshaped back to one row per key.
    scale_wide = pl.reshape(scales, [cache_rows // 8, 32])
    with pl.spmd(num_tokens, name_hint="c2a_index_select", deps=[publish_ready]) as select_tid:
        t = pl.tile.get_block_idx()
        length = pl.read(compressed_lens, [t])
        request = pl.cast(pl.read(token_to_req_indices, [t]), pl.INDEX)
        if length <= INDEX_TOPK:
            for c in pl.range(INDEX_TOPK // SCORE_TILE):
                p0 = c * SCORE_TILE
                # Inactive tiles need no page-table entry.
                if p0 < length:
                    block_id = pl.read(block_table, [request, p0 // 128])
                    ramp = pl.arange(0, [1, SCORE_TILE], dtype=pl.INT32)
                    # Integer scalars broadcast correctly into a tensor op; an FP32 runtime
                    # scalar does not, and a one-element column is below the tile alignment.
                    slack = pl.cast(pl.add(pl.mul(ramp, -1), pl.cast(length - p0, pl.INT32)), pl.FP32)
                    visible = pl.minimum(pl.maximum(slack, 0.0), 1.0)
                    block_base = pl.cast(block_id * 128 + p0 % 128, pl.INT32)
                    physical = pl.cast(pl.add(ramp, block_base), pl.FP32)
                    masked = pl.sub(pl.mul(visible, pl.add(physical, 1.0)), 1.0)
                    topk_indices[t : t + 1, p0 : p0 + SCORE_TILE] = pl.cast(masked, pl.INT32, mode="rint")
                else:
                    topk_indices[t : t + 1, p0 : p0 + SCORE_TILE] = pl.full(
                        [1, SCORE_TILE], dtype=pl.INT32, value=-1
                    )
        else:
            running = pl.tile.full([1, PAIR_WIDTH], dtype=pl.FP32, value=SORT_FLOOR)
            weight_row = pl.reshape(pl.cast(weights[t : t + 1, :], pl.FP32), [INDEX_H, 1])
            for leaf, (running_iter,) in pl.range((length + LEAF - 1) // LEAF, init_values=(running,)):
                leaf_base = leaf * LEAF
                for c in pl.range(LEAF_TILES):
                    p0 = leaf_base + c * SCORE_TILE
                    # The last leaf may extend beyond the visible page table.
                    if p0 < length:
                        block_id = pl.read(block_table, [request, p0 // 128])
                        base = pl.cast(pl.max(block_id, 0), pl.INDEX) * 128 + p0 % 128
                        packed = flat[base : base + SCORE_TILE, :]
                        wide = pl.reinterpret_view(pl.cast(packed, target_type=pl.UINT16), pl.INT16)
                        low = pl.ands(wide, 15)
                        high = pl.ands(pl.shrs(wide, 4), 15)
                        low_sign = pl.cast(pl.shrs(low, 3), pl.FP32)
                        low_exponent = pl.cast(pl.shrs(pl.ands(low, 7), 1), pl.FP32)
                        low_fraction = pl.cast(pl.ands(low, 1), pl.FP32)
                        # Assemble the exact FP32 exponent for E2M1 exponent codes 0..3.
                        low_exponent_i32 = pl.cast(pl.shrs(pl.ands(low, 7), 1), pl.INT32)
                        low_exponent_bits = pl.shls(pl.add(low_exponent_i32, 127), 23)
                        low_pow2 = pl.reinterpret_view(low_exponent_bits, pl.FP32)
                        low_normal = pl.mul(pl.mul(low_pow2, 0.5), pl.add(pl.mul(low_fraction, 0.5), 1.0))
                        low_is_normal = pl.minimum(low_exponent, 1.0)
                        low_magnitude = pl.add(
                            pl.mul(low_is_normal, low_normal),
                            pl.mul(pl.mul(pl.sub(low_is_normal, 1.0), -1.0), pl.mul(low_fraction, 0.5)),
                        )
                        low_value = pl.mul(low_magnitude, pl.mul(pl.sub(pl.mul(low_sign, 2.0), 1.0), -1.0))
                        high_sign = pl.cast(pl.shrs(high, 3), pl.FP32)
                        high_exponent = pl.cast(pl.shrs(pl.ands(high, 7), 1), pl.FP32)
                        high_fraction = pl.cast(pl.ands(high, 1), pl.FP32)
                        # Assemble the exact FP32 exponent for E2M1 exponent codes 0..3.
                        high_exponent_i32 = pl.cast(pl.shrs(pl.ands(high, 7), 1), pl.INT32)
                        high_exponent_bits = pl.shls(pl.add(high_exponent_i32, 127), 23)
                        high_pow2 = pl.reinterpret_view(high_exponent_bits, pl.FP32)
                        high_normal = pl.mul(pl.mul(high_pow2, 0.5), pl.add(pl.mul(high_fraction, 0.5), 1.0))
                        high_is_normal = pl.minimum(high_exponent, 1.0)
                        high_magnitude = pl.add(
                            pl.mul(high_is_normal, high_normal),
                            pl.mul(pl.mul(pl.sub(high_is_normal, 1.0), -1.0), pl.mul(high_fraction, 0.5)),
                        )
                        high_value = pl.mul(high_magnitude, pl.mul(pl.sub(pl.mul(high_sign, 2.0), 1.0), -1.0))
                        scale_tile = pl.slice(scale_wide, [SCORE_TILE // 8, 32], [base // 8, 0])
                        raw_codes = pl.reinterpret_view(scale_tile, pl.UINT8)
                        signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                        codes = pl.ands(signed_codes, 255)
                        factors = pl.reinterpret_view(
                            pl.maximum(pl.shls(codes, 23), E8M0_FLOOR_BITS), pl.FP32
                        )
                        factor_column = pl.reshape(factors, [SCORE_TILE * IDX_SCALES, 1])
                        low_scaled = pl.reshape(
                            pl.row_expand_mul(
                                pl.reshape(low_value, [SCORE_TILE * IDX_SCALES, 16]), factor_column
                            ),
                            [SCORE_TILE, IDX_PACKED],
                        )
                        high_scaled = pl.reshape(
                            pl.row_expand_mul(
                                pl.reshape(high_value, [SCORE_TILE * IDX_SCALES, 16]), factor_column
                            ),
                            [SCORE_TILE, IDX_PACKED],
                        )
                        keys = pl.concat(
                            pl.cast(low_scaled, pl.BF16, mode="rint"),
                            pl.cast(high_scaled, pl.BF16, mode="rint"),
                        )
                        query_tile = query_flat[t * INDEX_H : t * INDEX_H + INDEX_H, :]
                        scored = pl.maximum(pl.matmul(query_tile, keys, b_trans=True), 0.0)
                        row = pl.col_sum(pl.row_expand_mul(scored, weight_row))
                        ramp = pl.arange(0, [1, SCORE_TILE], dtype=pl.INT32)
                        slack = pl.cast(pl.add(pl.mul(ramp, -1), pl.cast(length - p0, pl.INT32)), pl.FP32)
                        visible = pl.minimum(pl.maximum(slack, 0.0), 1.0)
                        row = pl.maximum(pl.add(row, pl.mul(pl.sub(visible, 1.0), MASK_BIAS)), SORT_FLOOR)
                        leaf_scores[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = row
                        leaf_rows[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.add(
                            ramp, pl.cast(base, pl.INT32)
                        )
                    else:
                        leaf_scores[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.full(
                            [1, SCORE_TILE], dtype=pl.FP32, value=SORT_FLOOR
                        )
                        leaf_rows[t : t + 1, c * SCORE_TILE : c * SCORE_TILE + SCORE_TILE] = pl.full(
                            [1, SCORE_TILE], dtype=pl.INT32, value=-1
                        )
                score_row = pl.load(leaf_scores, [t, 0], [1, LEAF], target_memory=pl.Mem.Vec)
                row_ids = pl.load(leaf_rows, [t, 0], [1, LEAF], target_memory=pl.Mem.Vec)
                pairs = pl.sort32(score_row, pl.reinterpret_view(row_ids, pl.UINT32))
                pairs = pl.mrgsort(pairs, block_len=64)
                pairs = pl.mrgsort(pairs, block_len=256)
                pairs = pl.mrgsort(pairs, block_len=1024)
                merge_tmp = pl.tile.create([1, 2 * PAIR_WIDTH], dtype=pl.FP32)
                merged = pl.tile.mrgsort(
                    running_iter, pl.tile.slice(pairs, [1, PAIR_WIDTH], [0, 0]), tmp=merge_tmp
                )
                running = pl.yield_(pl.tile.slice(merged, [1, PAIR_WIDTH], [0, 0]))
            chosen = pl.tile.gather_mask(
                running, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            ordered = pl.sort32(
                pl.mul(pl.cast(chosen, pl.FP32), -1.0), pl.reinterpret_view(chosen, pl.UINT32)
            )
            ordered = pl.mrgsort(ordered, block_len=64)
            ordered = pl.mrgsort(ordered, block_len=256)
            final = pl.tile.gather_mask(
                pl.tile.slice(ordered, [1, PAIR_WIDTH], [0, 0]),
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            topk_indices = pl.store(final, [t, 0], topk_indices)
    return select_tid


attend_sparse = make_attend_selected_chunk()
project_index_q = make_mx_projection(Q_LORA, INDEX_H * INDEX_DIM)
rotate_compressed = make_rope(1)
rotate_index_key = make_wide_rope(1, INDEX_DIM)


# Wider output tiles amortize MXFP8 activation quantization across query heads.
C2A_QB_N_TILE = 256
_c2a_rotate_q = make_rope(LOCAL_H, max_workers=64, name_hint="c2a_q_rope")
_c2a_rotate_output = make_rope(LOCAL_H, inverse=True, max_workers=64, name_hint="c2a_o_rope")
c2a_o_proj = _make_o_proj(_c2a_rotate_output, grouped_output, _project_ob)
c2a_index_query = make_wide_rope(INDEX_H, INDEX_DIM, max_workers=64, permute=True)


_c2a_project_qb = make_mx_projection(
    Q_LORA, LOCAL_H * HEAD_DIM, n_tile=C2A_QB_N_TILE, name_hint="c2a_q_b"
)

@pl.jit.inline(auto_scope=False)
def c2a_qkv(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
    tokens = pl.tensor.dim(x, 0)
    expanded_q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    _c2a_project_qb(query_latent, wq_b, wq_b_scale, expanded_q, num_tokens)
    _c2a_rotate_q(expanded_q, rope_cos, rope_sin, query, num_tokens)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    return query_latent, query, window_kv


@pl.jit.inline(auto_scope=False)
def c2a_full_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write the FP32 local output and publish this layer's compressed and index caches."""
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    c2a_qkv(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, query, window_kv, num_tokens,
    )
    publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_project(x, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = compressor_ratio2(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    # Index keys come from the unrotated latent; the compressed cache stores the rotated one.
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    index_key_project(latent, index_wk, index_norm_weight, index_key, num_tokens, pool_tid)
    rotated_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    rotate_index_key(index_key, compressed_rope_cos, compressed_rope_sin, rotated_key, num_tokens)
    index_publish_tid = publish_index_key(
        rotated_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready
    )
    rotated_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    rotate_compressed(latent, compressed_rope_cos, compressed_rope_sin, rotated_latent, num_tokens)
    publish_compressed(
        rotated_latent, compressed_slots, compressed_cache, compressed_cache_scale,
        num_tokens, cache_ready,
    )

    index_query_raw = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    project_index_q(qr, index_wq_b, index_wq_b_scale, index_query_raw, num_tokens)
    index_query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    c2a_index_query(index_query_raw, rope_cos, rope_sin, index_query, num_tokens)
    index_weight = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    indexer_weights(x, index_weights_proj, index_weight, num_tokens)
    leaf_scores = pl.create_tensor([tokens, LEAF], dtype=pl.FP32)
    leaf_rows = pl.create_tensor([tokens, LEAF], dtype=pl.INT32)
    select_tid = index_select(
        index_query, index_weight, index_cache, index_cache_scale, index_block_table,
        token_to_req_indices, compressed_lens, leaf_scores, leaf_rows, topk_indices,
        num_tokens, index_publish_tid,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = pl.system.task_dummy(deps=[select_tid, state_tid])
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, topk_indices, selected, combined, start, active, chunk_done,
        )
        chunk_done = attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    c2a_o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return chunk_done


def golden_decode_attn_c2a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    position_ids: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_wgate: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C2A decode with request-owned ring state and paged caches."""
    return golden_compressed_attention(
        mode=AttentionMode.FULL,
        ratio=2,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a, wo_a_scale=wo_a_scale,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=compressor_wkv,
        compressor_wgate=compressor_wgate,
        compressor_norm_weight=compressor_norm_weight,
        query_start_loc=query_start_loc,
        state_block_table=state_block_table,
        state_cache=state_cache,
        compressed_slots=compressed_slots,
        position_ids=position_ids,
        compressed_lens=compressed_lens,
        compressed_rope_cos=compressed_rope_cos,
        compressed_rope_sin=compressed_rope_sin,
        index_wk=index_wk,
        index_norm_weight=index_norm_weight,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=token_to_req_indices,
        candidate_mask=None,
    )


def make_decode_attn_c2a_full(output_reduce=None):
    """Build the decode_attn_c2a_full leaf around one output collective."""
    reducer = output_reduce or decode_tp_output_all_reduce

    @pl.jit.inline(auto_scope=False)
    def decode_attn_c2a_full(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ],
        token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[T_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
        compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
        query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
        compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
        # A later epoch must not overwrite cache or transport storage still being read.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="c2a_previous_epoch", allow_early_resolve=False
        ) as cache_ready:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(
                    output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                    cmp=pld.WaitCmp.Ge,
                )
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        c2a_full_partial(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices,
            compressed_lens, index_cache, index_cache_scale, index_block_table, position_ids,
            compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
            query_start_loc, state_block_table, state_cache, compressor_norm_weight, compressed_slots,
            index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, partial, num_tokens, cache_ready,
        )
        reducer(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
            attention_epoch,
        )
        return output

    return decode_attn_c2a_full


decode_attn_c2a_full = make_decode_attn_c2a_full()
decode_attn_c2a_full_sharded = make_decode_attn_c2a_full(decode_tp_output_reduce_scatter)


# Independent CPU reference, transcribed from the official implementation.
# Transcription of DeepSeek-V4.1-Flash inference/model.py and inference/kernel.py at revision
# dba1be0a40aa45a94ad051997016db3960a90277 (Compressor, Indexer, Attention, sparse_attn); not an
# execution of the CUDA kernels.  Takes the tensors of golden_decode_attn_c2a_full.
#
# Like sparse_attention.golden_compressed_attention, only rows named by a non-negative slot are
# rewritten, so untouched cache bytes stay bit-identical and can be checked.  This matters because
# MXFP4 re-quantization is not byte-idempotent: an E8M0 group whose largest nibble is 3 re-derives a
# scale one binade lower, moving the bytes while the decoded values stay put.
#
# Cache ABI: compressed-KV and index payloads are PACKED E2M1, two logical values per byte with
# logical element 2i in the LOW nibble, so the stored last dimension is half the logical width --
# what quantize_mxfp4_cache emits and dequantize_mxfp4_cache consumes, and the deliberate
# correction of the scaffold's pl.FP4 annotation, which spells the logical width.
#
# Arithmetic ABI: the BF16 GEMMs accumulate in FP32 and round once (quant.golden_bf16_gemm,
# o_proj.golden_grouped_output), and the online softmax feeds BF16 probabilities to the PV GEMM,
# as the A5 kernels do.  Both differ from the shared reference -- together they are the entire
# measured gap against it.

E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

RATIO = 2
TOPK = M.index_topk
CMP_GROUP = COMPRESSED_CACHE_GROUP
IDX_GROUP = INDEX_CACHE_GROUP

def _write_rows(cache, rows, payload, width):
    """Rewrite only ``rows`` of a paged cache, leaving every other byte alone."""
    cache.view(torch.uint8).reshape(-1, width)[rows] = payload.contiguous().view(torch.uint8).reshape(-1, width)

def _flat_cache_rows(payload, scale, group_size, scale_format):
    """Packed E2M1 paged cache -> flattened FP32 physical rows."""
    return dequantize_mxfp4_cache(payload, scale, group_size, scale_format).flatten(0, -2)

def golden_reference_c2a(inputs: dict, *, fp32_output=False) -> dict:
    """Ratio-2 compressed sparse attention; publishes only the addressed cache rows.

    Returns the per-rank partial output (before TP reduction) plus every mutated state: the
    three paged caches with scales, the compressor state, and the physical Top-K rows.
    """
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    groups, group_in, _ = t["wo_a"].shape

    # a. Query/KV projection and rotary embedding.
    qr = golden_rms_norm(golden_mxfp8_linear(x, t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = golden_mxfp8_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(x, t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])

    # b. Sliding-window cache: MXFP8, group 32, BF16 scales; addressed rows only.
    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_scales = quantize_mxfp8_cache(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    # Boolean indexing of Float8 on CPU requires an exact FP32 roundtrip.
    visible_payload = window_payload.float()[window_valid].to(window_payload.dtype)
    _write_rows(window_cache, window_rows, visible_payload, head_dim)
    window_cache_scale.reshape(-1, head_dim // MX_GROUP)[window_rows] = window_scales[window_valid]

    # c. Ratio-2 compressor: per-channel two-way softmax over the pair, pooled row rounded to
    #    BF16 before RMSNorm, published exactly when position_ids % 2 == 1. Ring rows
    #    contain the KV projection followed by the gate score.
    state_cache = t["state_cache"].clone()
    pending_kv = x.float() @ t["compressor_wkv"].float()
    pending_score = x.float() @ t["compressor_wgate"].float()
    position_ids = t["position_ids"].long()
    latent = torch.zeros(tokens, head_dim, dtype=torch.bfloat16)
    publish = torch.zeros(tokens, dtype=torch.bool)
    capacity = state_cache.shape[1]
    valid_tokens = min(tokens, int(t["query_start_loc"][-1]))
    for token in range(valid_tokens):
        request = int(t["token_to_req_indices"][token])
        if not 0 <= request < t["state_block_table"].shape[0]:
            continue
        block = int(t["state_block_table"][request, 0])
        if not 0 <= block < state_cache.shape[0]:
            continue
        position = int(position_ids[token])
        if position < 0:
            continue
        if position % RATIO:
            previous = state_cache[block, (position - 1) % capacity]
            pair_kv = torch.stack((previous[:head_dim], pending_kv[token]))
            pair_score = torch.stack((previous[head_dim:], pending_score[token]))
            pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
            latent[token] = golden_rms_norm(pooled.to(torch.bfloat16), t["compressor_norm_weight"])
            publish[token] = True
        state_cache[block, position % capacity, :head_dim] = pending_kv[token]
        state_cache[block, position % capacity, head_dim:] = pending_score[token]

    # d. Compressed KV: rotate the latent tail, then packed FP4 with group-16 BF16 scales.
    compressed_cache = t["compressed_cache"].clone()
    compressed_cache_scale = t["compressed_cache_scale"].clone()
    rotated = golden_rope_tail(latent, t["compressed_rope_cos"], t["compressed_rope_sin"])
    compressed_slots = t["compressed_slots"].long()
    compressed_valid = (compressed_slots >= 0) & publish
    compressed_rows = compressed_slots[compressed_valid]
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(rotated[compressed_valid], CMP_GROUP, "bf16")
        _write_rows(compressed_cache, compressed_rows, payload, head_dim // 2)
        # BF16 scales are 2 bytes/element; _write_rows' width is in bytes, so the
        # group-16 scale row (head_dim // CMP_GROUP elements) is twice that in bytes.
        _write_rows(compressed_cache_scale, compressed_rows, scale, (head_dim // CMP_GROUP) * 2)

    # e. Index keys come from the UNROTATED latent; packed FP4, group 32, E8M0, same slots.
    index_cache = t["index_cache"].clone()
    index_cache_scale = t["index_cache_scale"].clone()
    index_key = golden_rms_norm(golden_bf16_gemm(latent, t["index_wk"]), t["index_norm_weight"])
    index_key = golden_rope_tail(index_key, t["compressed_rope_cos"], t["compressed_rope_sin"])
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(index_key[compressed_valid], IDX_GROUP, "e8m0")
        _write_rows(index_cache, compressed_rows, payload, index_dim // 2)
        _write_rows(index_cache_scale, compressed_rows, scale, index_dim // IDX_GROUP)

    index_values = _flat_cache_rows(index_cache, index_cache_scale, IDX_GROUP, "e8m0")

    # f. Indexer: score every causal compressed position against the freshly written keys.
    index_q = golden_mxfp8_linear(qr, t["index_wq_b"], t["index_wq_b_scale"]).unflatten(
        -1, (index_heads, index_dim)
    )
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    head_weights = golden_bf16_gemm(x, t["index_weights_proj"])
    head_weights = head_weights * index_dim**-0.5 * index_heads**-0.5
    lens = t["compressed_lens"].long()
    max_len = int(lens.max()) if lens.numel() else 0
    topk_indices = torch.full((tokens, TOPK), -1, dtype=torch.int32)
    if max_len:
        positions = torch.arange(max_len)
        blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        offsets = positions.remainder(BLOCK_SIZE)
        pages = t["index_block_table"][t["token_to_req_indices"].long().unsqueeze(-1), blocks]
        physical_rows = pages.to(torch.int64) * BLOCK_SIZE + offsets
        keys = index_values[physical_rows.reshape(-1)].reshape(tokens, max_len, index_dim)
        scores = torch.einsum("thd,tkd->thk", index_q.float(), keys.float()).relu()
        scores = (scores * head_weights.float().unsqueeze(-1)).sum(dim=-2)
        scores = scores.masked_fill(positions.unsqueeze(0) >= lens.unsqueeze(-1), -torch.inf)
        count = min(TOPK, max_len)
        logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
        selected_scores = scores.gather(-1, logical)
        selected_rows = physical_rows.gather(-1, logical)
        rows = torch.where(torch.isfinite(selected_scores), selected_rows, -1)
        for token in range(tokens):
            if int(lens[token]) > TOPK:
                # The scored device path orders winners by physical cache row.
                rows[token] = rows[token].sort().values
        topk_indices[:, :count] = rows.to(torch.int32)

    # g. Sparse attention: one online softmax over window rows plus selected compressed rows.
    window_values = (
        window_cache.float() * window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = _flat_cache_rows(compressed_cache, compressed_cache_scale, CMP_GROUP, "bf16")
    window_indices = t["window_indices"].long()
    gathered = torch.cat(
        (
            window_values[window_indices.clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
            compressed_values[topk_indices.long().clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
        ),
        dim=1,
    ).to(torch.bfloat16)
    visible = torch.cat((window_indices >= 0, topk_indices >= 0), dim=1)
    gathered = gathered.masked_fill(~visible[..., None], 0)

    maximum = torch.full((tokens, local_heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, local_heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[1], ATTEND_TILE):
        values = gathered[:, start : start + ATTEND_TILE].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), values) * head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None, start : start + ATTEND_TILE], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), values
        )
        maximum = new_maximum
    final_maximum = torch.maximum(maximum, t["attn_sink"].float()[None])
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (t["attn_sink"].float()[None] - final_maximum).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)

    # h. Inverse RoPE, grouped low-rank output projection, per-rank partial.
    attended = golden_rope_tail(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(tokens, groups, group_in)
    projected = golden_grouped_output(grouped, t["wo_a"], t["wo_a_scale"])
    output = golden_mxfp8_linear(projected.flatten(1), t["wo_b"], t["wo_b_scale"], fp32=fp32_output)

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_scale,
        "state_cache": state_cache,
        "topk_indices": topk_indices,
    }

INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
    "compressed_cache", "compressed_cache_scale", "token_to_req_indices", "compressed_lens",
    "index_cache", "index_cache_scale", "index_block_table", "position_ids",
    "compressed_rope_cos", "compressed_rope_sin", "compressor_wkv", "compressor_wgate",
    "query_start_loc", "state_block_table", "state_cache", "compressor_norm_weight", "compressed_slots",
    "index_wk", "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
)

FULL_ROPE_NAMES = {
    "rope_cos": "freqs_cos", "rope_sin": "freqs_sin",
    "compressed_rope_cos": "compressed_freqs_cos",
    "compressed_rope_sin": "compressed_freqs_sin",
}
PROGRAM_INPUT_NAMES = tuple(
    entry
    for name in INPUT_NAMES
    for entry in (("compressed_rope_positions", name) if name == "compressor_wkv"
                  else (FULL_ROPE_NAMES.get(name, name),))
)

# "long" drives compressed_lens above INDEX_TOPK for part of the batch so the indexer must
# score and select; the short requests in the same batch keep the -1 padding path live.
_PREFIXES = {
    ("mixed", "decode"): (1, 63, 128, 255, 0, 511, 129, 256),
    ("mixed", "prefill"): (0, 1, 64, 128, 0, 255, 2, 127),
    ("long", "decode"): (1024, 1027, 1290, 63, 1535, 128, 2048, 255),
    ("long", "prefill"): (1024, 1027, 1291, 62, 1534, 129, 2049, 254),
}

def _weight(gen, k, n):
    """MXFP8 input-major weight with Cube-ordered MX_B_NN scales."""
    payload, codes = golden_mxfp8_quantize(torch.randn(n, k, generator=gen) / math.sqrt(k))
    return payload.T.contiguous(), pack_mx_b_scale(codes.T.contiguous()).view(E8M0)

def _plan(tokens, requests, case, mode, start_pos=None):
    """Ragged query lengths and per-request KV prefixes."""
    lengths = [tokens // requests + (index < tokens % requests) for index in range(requests)]
    if mode == "prefill" and requests > 1 and lengths[-1] > 4:
        lengths[0] += 3
        lengths[-1] -= 3
    pattern = _PREFIXES[(case if case == "long" else "mixed", mode)]
    prefixes = [pattern[index % len(pattern)] for index in range(requests)]
    if start_pos is not None:
        prefixes = list(start_pos)
    return lengths, prefixes

def make_c2a_inputs(tokens=24, requests=6, seed=17, case="mixed", mode="decode", full_rope_tables=False, start_pos=None):
    """Deterministic inputs for ``golden_decode_attn_c2a_full``.

    case: "mixed" keeps contexts short so every causal position is selected, "long" pushes
    part of the batch past index_topk so the scored Top-K path runs, "masked" clears every
    slot and index, "zero" zeroes activations and all three cache payloads.  mode: "decode"
    (few rows per request, long prefixes) or "prefill" (ragged multi-token requests).
    """
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")
    if case not in ("mixed", "long", "masked", "zero"):
        raise ValueError(f"unsupported case {case!r}")
    if not 1 <= requests <= min(MAX_BATCH_PER_DP, tokens):
        raise ValueError(f"requests must be in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    gen = torch.Generator().manual_seed(seed)

    lengths, prefixes = _plan(tokens, requests, case, mode, start_pos)
    token_to_req_indices = torch.repeat_interleave(torch.arange(requests), torch.tensor(lengths))
    positions = torch.cat(
        [torch.arange(prefix, prefix + length) for prefix, length in zip(prefixes, lengths)]
    )
    maximum = int(positions.max())
    window_columns = maximum // BLOCK_SIZE + 1
    compressed_columns = max(4, (maximum // RATIO) // BLOCK_SIZE + 1)

    # One spare page per cache is never mapped: a control row for untouched-byte checks.
    window_pages = requests * window_columns + 1
    compressed_pages = requests * compressed_columns + 1
    def table(pages, columns):
        rows = torch.randperm(pages, generator=gen)[: requests * columns]
        return rows.reshape(requests, columns).to(torch.int32)

    window_table = table(window_pages, window_columns)
    index_block_table = table(compressed_pages, compressed_columns)

    window_slots, window_indices, _ = window_metadata(positions, token_to_req_indices, window_table)
    compressed_slots = paged_slots(positions, token_to_req_indices, index_block_table, BLOCK_SIZE, RATIO, True)
    compressed_lens = torch.div(positions + 1, RATIO, rounding_mode="floor").to(torch.int32)

    # Re-derive every piece of addressing independently of the metadata helpers.
    assert window_slots.unique().numel() == tokens
    for row, (request, position) in enumerate(zip(token_to_req_indices.tolist(), positions.tolist())):
        expected_slot = int(window_table[request, position // BLOCK_SIZE]) * BLOCK_SIZE
        assert int(window_slots[row]) == expected_slot + position % BLOCK_SIZE
        history = list(range(max(0, position - WINDOW + 1), position + 1))
        expected = [
            int(window_table[request, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE
            for p in history
        ]
        assert window_indices[row, : len(expected)].tolist() == expected
        assert bool((window_indices[row, len(expected) :] == -1).all())
        logical = position // RATIO
        published = (position + 1) % RATIO == 0
        expected_compressed = (
            int(index_block_table[request, logical // BLOCK_SIZE]) * BLOCK_SIZE
            + logical % BLOCK_SIZE
            if published
            else -1
        )
        assert int(compressed_slots[row]) == expected_compressed
        assert int(compressed_lens[row]) == (position + 1) // RATIO
    published_slots = compressed_slots[compressed_slots >= 0]
    assert published_slots.unique().numel() == published_slots.numel()

    # Fixture initialization only: production callers retain these full tables.
    table_rows = int(positions.max()) + 1
    freqs_cos, freqs_sin = precompute_rope_tables(table_rows, compressed_attention=True)
    compressed_freqs_cos, compressed_freqs_sin = freqs_cos, freqs_sin
    rope_cos, rope_sin = freqs_cos[positions], freqs_sin[positions]
    complete = (positions + 1).remainder(RATIO) == 0
    pair_start = torch.where(complete, positions + 1 - RATIO, torch.full_like(positions, -1))
    compressed_rope_cos = compressed_freqs_cos[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 1.0)
    compressed_rope_sin = compressed_freqs_sin[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 0.0)

    wq_a, wq_a_scale = _weight(gen, D, Q_LORA)
    wq_b, wq_b_scale = _weight(gen, Q_LORA, LOCAL_H * HEAD_DIM)
    wkv, wkv_scale = _weight(gen, D, HEAD_DIM)
    wo_b, wo_b_scale = _weight(gen, LOCAL_O_WIDTH, D)
    index_wq_b, index_wq_b_scale = _weight(gen, Q_LORA, INDEX_H * INDEX_DIM)

    compressor_wkv = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    compressor_wgate = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    wo_a_values = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=gen)
    wo_a, wo_a_scale = quantize_grouped_mxfp8_weight(wo_a_values / math.sqrt(O_GROUP_IN))
    # Seed each request's predecessor projections in its physical ring block.
    pending = torch.randn(MAX_BATCH_PER_DP, D, generator=gen).bfloat16().float()
    state_cache = torch.zeros(MAX_BATCH_PER_DP + 1, STATE_CAPACITY, STATE_WIDTH)
    state_block_table = torch.arange(requests - 1, -1, -1, dtype=torch.int32).unsqueeze(1) + 1
    for request, prefix in enumerate(prefixes):
        if prefix:
            block = int(state_block_table[request, 0])
            slot = (prefix - 1) % STATE_CAPACITY
            state_cache[block, slot, :HEAD_DIM] = pending[request] @ compressor_wkv
            state_cache[block, slot, HEAD_DIM:] = pending[request] @ compressor_wgate
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    window_cache, window_cache_scales = quantize_mxfp8_cache(
        torch.randn(window_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16()
    )
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16(),
        CMP_GROUP,
        "bf16",
    )
    index_cache, index_cache_codes = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, INDEX_DIM, generator=gen).bfloat16(),
        IDX_GROUP,
        "e8m0",
    )

    values = {
        "x": torch.randn(tokens, D, generator=gen).bfloat16(),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": (torch.randn(Q_LORA, generator=gen) * 0.1 + 1).bfloat16(),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "attn_sink": torch.randn(LOCAL_H, generator=gen) * 2,
        "wo_a": wo_a,
        "wo_a_scale": wo_a_scale,
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "window_slots": window_slots,
        "window_indices": window_indices,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scales,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "token_to_req_indices": token_to_req_indices.to(torch.int32),
        "compressed_lens": compressed_lens,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_codes.view(E8M0),
        "index_block_table": index_block_table,
        "position_ids": positions.to(torch.int32),
        "compressed_rope_cos": compressed_rope_cos,
        "compressed_rope_sin": compressed_rope_sin,
        "compressor_wkv": compressor_wkv,
        "compressor_wgate": compressor_wgate,
        "query_start_loc": query_start_loc,
        "state_block_table": state_block_table,
        "state_cache": state_cache,
        "compressor_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "compressed_slots": compressed_slots,
        "index_wk": (torch.randn(HEAD_DIM, INDEX_DIM, generator=gen) / math.sqrt(HEAD_DIM)).bfloat16(),
        "index_norm_weight": (torch.randn(INDEX_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": (torch.randn(D, INDEX_H, generator=gen) / math.sqrt(D)).bfloat16(),
    }

    if case == "masked":
        values["window_slots"] = torch.full_like(window_slots, -1)
        values["window_indices"] = torch.full_like(window_indices, -1)
        values["compressed_slots"] = torch.full_like(compressed_slots, -1)
        values["compressed_lens"] = torch.zeros_like(compressed_lens)
        values["state_block_table"] = torch.full_like(state_block_table, -1)
    if case == "zero":
        values["x"] = torch.zeros_like(values["x"])
        values["window_cache"] = torch.zeros_like(window_cache.view(torch.uint8)).view(window_cache.dtype)
        values["compressed_cache"] = torch.zeros_like(compressed_cache)
        values["index_cache"] = torch.zeros_like(index_cache)
        values["state_cache"] = torch.zeros_like(state_cache)

    print(
        f"[FIXTURE] c2a tokens={tokens} requests={requests} mode={mode} case={case} "
        f"lengths={lengths} prefixes={prefixes} "
        f"max_compressed_len={int(values['compressed_lens'].max())} "
        f"window_pages={window_pages} compressed_pages={compressed_pages} "
        f"index_block_table={tuple(index_block_table.shape)}"
    )
    if full_rope_tables:
        values.update(
            freqs_cos=freqs_cos, freqs_sin=freqs_sin,
            compressed_freqs_cos=compressed_freqs_cos, compressed_freqs_sin=compressed_freqs_sin,
            compressed_rope_positions=pair_start.to(torch.int32),
        )
        return {name: values[name] for name in PROGRAM_INPUT_NAMES}
    return {name: values[name] for name in INPUT_NAMES}


MUTABLE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "state_cache",
)
SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale")
CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


def cache_compare(world_size, tokens, names=None):
    """Mapped rows must match byte for byte, except a derived scale allows a one-code mismatch.

    An E8M0 scale is a bare exponent, coarse enough that device and CPU agree exactly, and so
    does every payload here. A BF16 scale (``compressed_cache_scale``) is a derived quantity
    (amax over a group), so the device and the CPU golden can differ by a ULP and flip one
    code at a quantization boundary the same way a payload can; every other mapped row stays
    unchanged.
    """
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
    )
    inexact_scales = {
        scale_name for scale_name, _, scale_format in CACHE_PAYLOADS.values()
        if scale_format not in (None, "e8m0")
    }
    return {
        name: mapped_pool_ratio_allclose(
            CACHE_SLOTS[name],
            pool_name=name,
            max_error_ratio=0.01 if name in inexact_scales else 0.0,
            **options,
        )
        for name in (CACHE_SLOTS if names is None else names)
    }
CACHE_PAYLOADS = {
    "window_cache": ("window_cache_scale", None, None),
    "compressed_cache": ("compressed_cache_scale", 16, "bf16"),
    "index_cache": ("index_cache_scale", 32, "e8m0"),
}


def golden_c2a(tensors, full_rope_tables=True, sharded=False):
    """Reference each TP shard once, then perform the FP32 TP reduction."""
    world_size = tensors["x"].shape[0]
    active = int(tensors["num_tokens"])
    for base in range(0, world_size, TP_SIZE):
        partials = []
        for rank in range(base, base + TP_SIZE):
            if full_rope_tables:
                inputs = {name: tensors[name][rank] for name in INPUT_NAMES if name not in FULL_ROPE_NAMES}
                for row_name, table_name in FULL_ROPE_NAMES.items():
                    positions_name = "compressed_rope_positions" if row_name.startswith("compressed_") else "position_ids"
                    positions = tensors[positions_name][rank].to(torch.long)
                    identity = 1.0 if row_name.endswith("cos") else 0.0
                    inputs[row_name] = tensors[table_name][rank][positions.clamp_min(0)].masked_fill(
                        positions[:, None] < 0, identity
                    )
            else:
                inputs = {name: tensors[name][rank] for name in INPUT_NAMES}
            result = golden_reference_c2a(inputs)
            partials.append(result["output"].float())
            for name in MUTABLE_NAMES:
                tensors[name][rank].copy_(result[name])
            tensors["topk_indices"][rank].copy_(result["topk_indices"])
        reduced = sum(partials).bfloat16()
        if sharded:
            # Ownership follows the fixed physical slab, so the active count only
            # masks its suffix and T < TP keeps the AllGather row mapping.
            width = tensors["output"].shape[1]
            for offset in range(TP_SIZE):
                first = min(offset * width, active)
                count = max(0, min(width, active - first))
                tensors["output"][base + offset].zero_()
                if count:
                    tensors["output"][base + offset][:count].copy_(reduced[first : first + count])
        else:
            tensors["output"][base : base + TP_SIZE].copy_(
                reduced.unsqueeze(0).expand(TP_SIZE, -1, -1)
            )


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_decode_attn_c2a(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one rank of C2A attention, skipping idle work."""
    freqs_cos.bind_dynamic(0, ROPE_ROWS_DYN)
    x.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    if num_tokens > 0:
        tokens = pl.tensor.dim(x, 0)
        rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
        rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
        compressed_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
        compressed_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
        materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos, rope_sin)
        materialize_rope_rows(
            compressed_freqs_cos, compressed_freqs_sin, compressed_rope_positions, num_tokens,
            compressed_rope_cos, compressed_rope_sin,
        )
        decode_attn_c2a_full(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
            index_cache, index_cache_scale, index_block_table, position_ids,
            compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
            query_start_loc, state_block_table, state_cache, compressor_norm_weight,
            compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
            index_weights_proj, topk_indices, output_window, output_arrived, output,
            rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens, attention_epoch,
        )
    return (
        output, topk_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, index_cache, index_cache_scale, state_cache,
    )


@pl.jit.host
def l3_decode_attn_c2a(
    x: pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]],
    token_to_req_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[RANKS_DYN, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS_DYN, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[RANKS_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch C2A attention across TP groups with shared output windows."""
    x.bind_dynamic(0, RANKS_DYN)
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    x.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = index_wq_b_scale[rank]
        l2_decode_attn_c2a(
            x[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
            wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_r, freqs_cos[rank],
            freqs_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
            window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            token_to_req_indices[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], position_ids[rank],
            compressed_freqs_cos[rank], compressed_freqs_sin[rank], compressed_rope_positions[rank], compressor_wkv[rank],
            compressor_wgate[rank], query_start_loc[rank], state_block_table[rank], state_cache[rank],
            compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank],
            index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
            index_weights_proj[rank], topk_indices[rank], output[rank], data, signal, rank,
            num_tokens, attention_epoch, device=rank,
        )


def build_specs(args, mode, full_rope_tables=True):
    """Build shape-only specs; replay and compile-only never generate random weights."""
    input_names = PROGRAM_INPUT_NAMES if full_rope_tables else INPUT_NAMES
    world_size = TP_SIZE * args.dp
    ranks = {}

    def initialize(name):
        if not ranks:
            for rank in range(world_size):
                ranks[rank] = make_c2a_inputs(
                    tokens=args.tokens,
                    requests=args.requests,
                    seed=args.seed + rank,
                    case=args.case,
                    mode=mode, full_rope_tables=full_rope_tables, start_pos=getattr(args, "start_pos", None),
                )
            # Every rank of a TP group sees the same tokens, metadata and caches.
            for rank in range(world_size):
                leader = ranks[rank // TP_SIZE * TP_SIZE]
                for key in input_names:
                    if key not in SHARDED_NAMES:
                        ranks[rank][key] = leader[key]
        column = [ranks[rank][name] for rank in range(world_size)]
        if column[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            stacked = torch.stack([value.view(torch.uint8) for value in column])
            return stacked.view(column[0].dtype)
        return torch.stack(column)

    shapes = make_c2a_inputs(
        tokens=args.tokens, requests=args.requests, seed=args.seed, case=args.case, mode=mode, full_rope_tables=full_rope_tables, start_pos=getattr(args, "start_pos", None)
    )
    specs = [
        TensorSpec(
            name,
            [world_size, *shapes[name].shape],
            shapes[name].dtype,
            init_value=(lambda name=name: initialize(name)),
            resident="stacked",
        )
        for name in input_names
    ]
    specs.append(
        TensorSpec("topk_indices", [world_size, args.tokens, INDEX_TOPK], torch.int32, resident="stacked")
    )
    specs.append(
        TensorSpec("output", [world_size, args.tokens, D], torch.bfloat16, resident="stacked")
    )
    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens, compile_runtime=True))
    specs.append(
        ScalarSpec(
            "attention_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            benchmark_step=1 if args.bench else None,
        )
    )
    return specs


@pl.jit.inline(auto_scope=False)
def c2a_reuse_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write the FP32 local output for a layer that reuses a published selection.

    Reuse owns only the sliding-window cache. The compressed pool and the Top-K row
    list both belong to this ratio's source layer and are read without being written,
    so there is no compressor, no index-key publication and no indexer here.
    """
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, query, window_kv, num_tokens,
    )
    publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = cache_ready
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, compressed_indices, selected, combined, start, active,
            chunk_done,
        )
        chunk_done = attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return chunk_done


REUSE_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale", "compressed_cache",
    "compressed_cache_scale", "compressed_indices",
)
REUSE_MUTABLE_NAMES = ("window_cache", "window_cache_scale")
REUSE_SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale")


def golden_reference_c2a_reuse(inputs: dict, *, fp32_output=False) -> dict:
    """Ratio-2 reuse attention; publishes only the addressed sliding-window rows.

    Same transcription as the full-mode reference, with the compressor, the index-key
    publication and the indexer removed: the selection arrives as an input and the
    compressed pool is read-only.
    """
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    groups, group_in, _ = t["wo_a"].shape

    qr = golden_rms_norm(golden_mxfp8_linear(x, t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = golden_mxfp8_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(x, t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])

    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_scales = quantize_mxfp8_cache(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    # Boolean indexing of Float8 on CPU requires an exact FP32 roundtrip.
    visible_payload = window_payload.float()[window_valid].to(window_payload.dtype)
    _write_rows(window_cache, window_rows, visible_payload, head_dim)
    window_cache_scale.reshape(-1, head_dim // MX_GROUP)[window_rows] = window_scales[window_valid]

    window_values = (
        window_cache.float() * window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = _flat_cache_rows(
        t["compressed_cache"], t["compressed_cache_scale"], CMP_GROUP, "bf16"
    )
    window_indices = t["window_indices"].long()
    compressed_indices = t["compressed_indices"].long()
    gathered = torch.cat(
        (
            window_values[window_indices.clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
            compressed_values[compressed_indices.clamp_min(0).reshape(-1)].reshape(
                tokens, -1, head_dim
            ),
        ),
        dim=1,
    ).to(torch.bfloat16)
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=1)
    gathered = gathered.masked_fill(~visible[..., None], 0)

    maximum = torch.full((tokens, local_heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, local_heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[1], ATTEND_TILE):
        values = gathered[:, start : start + ATTEND_TILE].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), values) * head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None, start : start + ATTEND_TILE], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), values
        )
        maximum = new_maximum
    final_maximum = torch.maximum(maximum, t["attn_sink"].float()[None])
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (t["attn_sink"].float()[None] - final_maximum).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)

    attended = golden_rope_tail(attended, t["rope_cos"], t["rope_sin"], inverse=True)
    grouped = attended.reshape(tokens, groups, group_in)
    projected = golden_grouped_output(grouped, t["wo_a"], t["wo_a_scale"])
    output = golden_mxfp8_linear(projected.flatten(1), t["wo_b"], t["wo_b_scale"], fp32=fp32_output)

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
    }


def make_c2a_reuse_inputs(tokens=32, requests=6, seed=17, case="mixed", mode="decode", start_pos=None):
    """Inputs for a reuse layer, taken from what a full layer actually published.

    The compressed pool and the Top-K row list come from running the full-mode
    reference on the same batch, so the fixture exercises the real Full-to-Reuse
    hand-off rather than a synthetic selection.
    """
    source = make_c2a_inputs(tokens=tokens, requests=requests, seed=seed, case=case, mode=mode, start_pos=start_pos)
    published = golden_reference_c2a(source)
    values = {name: source[name] for name in REUSE_INPUT_NAMES if name != "compressed_indices"}
    values["compressed_cache"] = published["compressed_cache"]
    values["compressed_cache_scale"] = published["compressed_cache_scale"]
    values["compressed_indices"] = published["topk_indices"]
    reachable = int((published["topk_indices"] >= 0).sum())
    print(
        f"[FIXTURE] c2a reuse tokens={tokens} requests={requests} mode={mode} case={case} "
        f"selected_rows={reachable} source_published_rows="
        f"{int((source['compressed_slots'] >= 0).sum())}"
    )
    return values


# Preserve the original public reference; device validation uses the main-aligned reference above.
def golden_decode_attn_c2a_reuse(
    x: "torch.Tensor",
    wq_a: "torch.Tensor",
    wq_a_scale: "torch.Tensor",
    q_norm_weight: "torch.Tensor",
    wq_b: "torch.Tensor",
    wq_b_scale: "torch.Tensor",
    wkv: "torch.Tensor",
    wkv_scale: "torch.Tensor",
    kv_norm_weight: "torch.Tensor",
    attn_sink: "torch.Tensor",
    wo_a: "torch.Tensor",
    wo_a_scale: "torch.Tensor",
    wo_b: "torch.Tensor",
    wo_b_scale: "torch.Tensor",
    rope_cos: "torch.Tensor",
    rope_sin: "torch.Tensor",
    window_slots: "torch.Tensor",
    window_indices: "torch.Tensor",
    window_cache: "torch.Tensor",
    window_cache_scale: "torch.Tensor",
    compressed_cache: "torch.Tensor",
    compressed_cache_scale: "torch.Tensor",
    compressed_indices: "torch.Tensor",
) -> "AttentionGoldenResult":
    """Evaluate C2A decode reuse against published compressed cache rows."""
    from sparse_attention import golden_compressed_attention
    from config import AttentionMode

    return golden_compressed_attention(
        mode=AttentionMode.REUSE,
        ratio=2,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a, wo_a_scale=wo_a_scale,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=compressed_indices,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=None,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=None,
        index_wq_b_scale=None,
        index_weights_proj=None,
        index_cache=None,
        index_cache_scale=None,
        index_block_table=None,
        request_ids=None,
        candidate_mask=None,
    )


def make_decode_attn_c2a_reuse(output_reduce=None):
    """Build the decode_attn_c2a_reuse leaf around one output collective."""
    reducer = output_reduce or decode_tp_output_all_reduce

    @pl.jit.inline(auto_scope=False)
    def decode_attn_c2a_reuse(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
        wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ],
        compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
        # A later epoch must not overwrite cache or transport storage still being read.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="c2a_reuse_previous_epoch", allow_early_resolve=False
        ) as cache_ready:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(
                    output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                    cmp=pld.WaitCmp.Ge,
                )
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        c2a_reuse_partial(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            compressed_indices, partial, num_tokens, cache_ready,
        )
        reducer(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
            attention_epoch,
        )
        return output

    return decode_attn_c2a_reuse


decode_attn_c2a_reuse = make_decode_attn_c2a_reuse()
decode_attn_c2a_reuse_sharded = make_decode_attn_c2a_reuse(decode_tp_output_reduce_scatter)


def golden_c2a_reuse(tensors, sharded=False):
    """Use main's FP32 WoA/BF16 PV reference on the active prefix of each TP group."""
    active = int(tensors["num_tokens"])
    world_size = tensors["x"].shape[0]
    for base in range(0, world_size, TP_SIZE):
        partials = []
        for rank in range(base, base + TP_SIZE):
            inputs = {name: tensors[name][rank] for name in REUSE_INPUT_NAMES}
            inputs = {
                name: value[:active] if name in TOKEN_INPUT_NAMES else value
                for name, value in inputs.items()
            }
            result = golden_reference_c2a_reuse(inputs)
            partials.append(result["output"].float())
            for name in REUSE_MUTABLE_NAMES:
                tensors[name][rank].copy_(result[name])
        reduced = sum(partials).bfloat16()
        if sharded:
            # Ownership follows the fixed physical slab, so the active count only
            # masks its suffix and T < TP keeps the AllGather row mapping.
            width = tensors["output"].shape[1]
            for offset in range(TP_SIZE):
                first = min(offset * width, active)
                count = max(0, min(width, active - first))
                tensors["output"][base + offset].zero_()
                if count:
                    tensors["output"][base + offset][:count].copy_(reduced[first : first + count])
        else:
            tensors["output"][base:base + TP_SIZE, :active].copy_(
                reduced.unsqueeze(0).expand(TP_SIZE, -1, -1)
            )


TOKEN_INPUT_NAMES = (
    "x", "rope_cos", "rope_sin", "window_slots", "window_indices", "compressed_indices",
)


def compare_owned_cache(name, initial_cache):
    """Check numerical publication and preserve unmapped bytes from the input cache."""
    def compare(actual, expected, *, inputs, **kwargs):
        """Validate every rank against its original cache and the CPU reference."""
        active = int(inputs["num_tokens"])
        slots = inputs["window_slots"][:, :active]
        source = initial_cache[name].contiguous().view(torch.uint8)
        written = actual.contiguous().view(torch.uint8)
        width = actual.shape[-1]
        for rank in range(actual.shape[0]):
            rows = slots[rank].long()
            rows = rows[rows >= 0]
            untouched = torch.ones(actual[rank].numel() // width, dtype=torch.bool)
            untouched[rows] = False
            if not torch.equal(
                written[rank].reshape(-1, width * actual.element_size())[untouched],
                source[rank].reshape(-1, width * actual.element_size())[untouched],
            ):
                return False, f"{name}: unmapped bytes differ from the input on rank {rank}"
        # Composed upstream rows can flip one FP8 code at a quantization boundary: allow one
        # subnormal step (2**-9) plus one relative step (2**-3) on the window payload.
        payload = name == "window_cache"
        shared_compare = mapped_pool_ratio_allclose(
            "window_slots",
            mapping_shape=tuple(slots.shape),
            block_size=BLOCK_SIZE,
            leading_rank_axis=True,
            pool_name=name,
            atol=2 ** -9 if payload else 0.0,
            rtol=2 ** -3 if payload else 0.0,
            max_error_ratio=0.0,
        )
        return shared_compare(actual, expected, inputs={**inputs, "window_slots": slots}, **kwargs)

    return compare


def input_metadata(tokens, requests, active, case, mode, start_pos=None):
    """Describe C2A validation tensor shapes without constructing random tensors."""
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")
    lengths, prefixes = _plan(active, requests, case, mode, start_pos)
    maximum = max(prefix + length - 1 for prefix, length in zip(prefixes, lengths))
    window_pages = requests * (maximum // BLOCK_SIZE + 1) + 1
    compressed_pages = requests * max(4, (maximum // RATIO) // BLOCK_SIZE + 1) + 1
    return (
        ("x", [tokens, D], bf),
        ("wq_a", [D, Q_LORA], fp),
        ("wq_a_scale", [D // 32, Q_LORA], mx),
        ("q_norm_weight", [Q_LORA], bf),
        ("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], fp),
        ("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], mx),
        ("wkv", [D, HEAD_DIM], fp),
        ("wkv_scale", [D // 32, HEAD_DIM], mx),
        ("kv_norm_weight", [HEAD_DIM], bf),
        ("attn_sink", [LOCAL_H], torch.float32),
        ("wo_a", [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn),
        ("wo_a_scale", [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], torch.float8_e8m0fnu),
        ("wo_b", [LOCAL_O_WIDTH, D], fp),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], mx),
        ("rope_cos", [tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [tokens, ROPE_DIM // 2], torch.float32),
        ("window_slots", [tokens], torch.int64),
        ("window_indices", [tokens, M.sliding_window], torch.int32),
        ("window_cache", [window_pages, BLOCK_SIZE, 1, HEAD_DIM], fp),
        ("window_cache_scale", [window_pages, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], bf),
        ("compressed_cache", [compressed_pages, BLOCK_SIZE, 1, CMP_PACKED], torch.uint8),
        ("compressed_cache_scale", [compressed_pages, BLOCK_SIZE, 1, CMP_SCALES], bf),
        ("compressed_indices", [tokens, INDEX_TOPK], torch.int32),
    )

def build_reuse_specs(args, mode, initial_cache):
    """Describe shapes lazily and populate full-to-reuse fixtures only for device runs."""
    metadata = input_metadata(args.tokens, args.requests, args.active_tokens, args.case, mode, getattr(args, "start_pos", None))
    world_size = TP_SIZE * args.dp
    ranks = {}

    def initialize(name):
        """Create deterministic rank inputs once and retain immutable cache snapshots."""
        if not ranks:
            for rank in range(world_size):
                values = make_c2a_reuse_inputs(
                    tokens=args.active_tokens, requests=args.requests,
                    seed=args.seed + rank, case=args.case, mode=mode, start_pos=getattr(args, "start_pos", None),
                )
                if args.active_tokens < args.tokens:
                    for key in TOKEN_INPUT_NAMES:
                        value = values[key]
                        shape = (args.tokens, *value.shape[1:])
                        # Invalid suffix indices must never be dereferenced by the kernel.
                        fill = 2147483647 if key.endswith(("indices", "slots")) else 13
                        padded = torch.full(shape, fill, dtype=value.dtype)
                        padded[:args.active_tokens] = value
                        values[key] = padded
                # Fail before device execution if the shared fixture and specs diverge.
                for key, shape, dtype in metadata:
                    value = values[key]
                    if list(value.shape) != shape or value.dtype != dtype:
                        raise ValueError(
                            f"{key}: fixture {list(value.shape)}/{value.dtype} does not match "
                            f"spec {shape}/{dtype}"
                        )
                ranks[rank] = values
            for rank in range(world_size):
                leader = ranks[rank // TP_SIZE * TP_SIZE]
                for key in REUSE_INPUT_NAMES:
                    if key not in REUSE_SHARDED_NAMES:
                        ranks[rank][key] = leader[key]
        column = [ranks[rank][name] for rank in range(world_size)]
        if column[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            stacked = torch.stack([value.view(torch.uint8) for value in column]).view(column[0].dtype)
        else:
            stacked = torch.stack(column)
        if name in REUSE_MUTABLE_NAMES:
            initial_cache[name] = stacked.clone()
        return stacked

    specs = [
        TensorSpec(name, [world_size, *shape], dtype,
                   init_value=lambda name=name: initialize(name), resident="stacked")
        for name, shape, dtype in metadata
    ]
    specs.append(TensorSpec(
        "output", [world_size, args.tokens, D], torch.bfloat16,
        init_value=lambda: torch.full((world_size, args.tokens, D), 13.0, dtype=torch.bfloat16),
        resident="stacked",
    ))
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(ScalarSpec(
        "attention_epoch", torch.int32, 1, compile_runtime=True,
        benchmark_step=1 if args.bench else None,
    ))
    return specs


if __name__ == "__main__":
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 decode C2A full: A5 precision and timing")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=24)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", help="replay a compatible compiled build without recompiling")
    parser.add_argument("--save-data", action="store_true", help="save validated inputs and golden outputs for replay")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp {args.tp} does not match import-time TP_SIZE {TP_SIZE}")

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(TP_SIZE * args.dp))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {TP_SIZE * args.dp} distinct non-negative device IDs")
    capacity = DECODE_MAX_TOKENS
    if not 1 <= args.tokens <= capacity:
        parser.error(f"--tokens must be in [1, {capacity}]")
    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"--requests must be in [1, {min(MAX_BATCH_PER_DP, args.tokens)}]")
    if args.bench and args.enable_chip_swimlane:
        parser.error("benchmark epoch stepping and multi-pass chip swimlane must be run separately")
    torch.set_num_threads(8)

    print(
        f"[C2A] mode=decode tokens={args.tokens} requests={args.requests} TP={TP_SIZE} "
        f"DP={args.dp} case={args.case} seed={args.seed} devices={devices}"
    )
    if args.bench:
        print(
            "[C2A] Resident device timing excludes compilation, input generation and CPU golden; "
            "each dispatch advances the communication epoch. "
        )

    # Zero and masked outputs are exactly zero, so any residue is a failure.
    output_compare = (
        ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
        if args.case in ("masked", "zero") else ratio_allclose(atol=1e-2, rtol=1.0 / 128)
    )
    compare = {
        "output": output_compare,
        "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
    }
    compare.update(cache_compare(len(devices), args.tokens))

    result = run(
        fn=l3_decode_attn_c2a,
        specs=build_specs(args, "decode"),
        golden_fn=golden_c2a,
        runtime_dir=args.runtime_dir,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compare_fn=compare,
    )
    print(f"[C2A] work_dir={result.work_dir}")
    if args.compile_only:
        print("[C2A] Compilation passed; device accuracy was NOT validated.")
    if args.save_data and result.work_dir:
        print(f"[C2A] Validated snapshot: {result.work_dir}/data")
    if not result.passed:
        raise SystemExit(result.error or 1)
