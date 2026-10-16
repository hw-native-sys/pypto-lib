# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A attention leaves: full and reuse."""


import pypto.language as pl
import pypto.language.distributed as pld
import torch


from config import (
    FLASH as M,
    AttentionMode,
    TP_SIZE,
    T_DYN,
    ORI_BLOCKS_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    B_DYN,
    TABLE_DYN,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    PREFILL_MAX_TOKENS,
    STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)

from compressor import STATE_WIDTH


from utils import AttentionGoldenResult
from sparse_attention import (
    PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH,
    gather_packed_chunk, golden_compressed_attention, make_attend_selected_chunk,
)
from o_proj import prefill_tp_output_all_reduce
from rmsnorm import EPS
from qkv_proj_rope import (
    M_TILE,
    N_TILE,
    make_bf16_projection_staged,
    make_rope,
)
from o_proj import prefill_o_proj
from qkv_proj_rope import (
    prefill_q_proj_qr, prefill_q_proj_rope, kv_proj_rope, _make_q_proj_rope, _prefill_rotate_q,
)
from compressor import compressor_ratio2
from qkv_proj_rope import make_mx_projection


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


PREFILL_COMPRESSOR_SEGMENTS = 16
PREFILL_COMPRESSOR_K_CHUNK = D // PREFILL_COMPRESSOR_SEGMENTS
PREFILL_C2A_FULL_RING_HEAP = (2 * 1024 * 1024 * 1024,) * 4


def make_prefill_c2a_qkv(query_projection_operator):
    """Bind the query expansion while retaining the prefill Q-A schedule."""
    @pl.jit.inline(auto_scope=False)
    def prefill_c2a_qkv(
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
        """Use group-32 Q-A accumulation and bounded prefill head-RoPE workers."""
        prefill_q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
        query_projection_operator(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
        kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
        return query_latent, query, window_kv

    return prefill_c2a_qkv


prefill_c2a_qkv = make_prefill_c2a_qkv(prefill_q_proj_rope)
# Keep the wider query expansion scoped to C2A full attention.
_prefill_c2a_project_qb = make_mx_projection(
    Q_LORA, LOCAL_H * HEAD_DIM, n_tile=256, name_hint="c2a_q_b"
)
prefill_c2a_full_qkv = make_prefill_c2a_qkv(
    _make_q_proj_rope(_prefill_c2a_project_qb, _prefill_rotate_q)
)


CMP_SCALES = HEAD_DIM // 16

IDX_SCALES = INDEX_DIM // 32

CMP_PACKED = HEAD_DIM // 2

IDX_PACKED = INDEX_DIM // 2

FP4_MAX_INV = 1.0 / 6.0

FP4_SCALE_FLOOR = 2.0**-9

E8M0_FLOOR_BITS = 4194304


SCORE_TILE = 64

LEAF = 2048

LEAF_TILES = LEAF // SCORE_TILE

PAIR_WIDTH = 2 * INDEX_TOPK

SORT_FLOOR = -3.0e38

MASK_BIAS = 1.0e30

COMPRESSOR_K_TILE = 64

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


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="swa_cache_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
            value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
            amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
            scale = pl.cast(pl.mul(amax, 1.0 / 448.0), pl.BF16, mode="rint")
            payload = pl.cast(pl.row_expand_div(value, pl.cast(scale, pl.FP32)), pl.FP8E4M3FN, mode="rint")
            flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
            scale_flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(scale, [1, HEAD_DIM // 16]),
                                                         1, HEAD_DIM // 32)
    return cache, scales


project_index_q = make_mx_projection(Q_LORA, INDEX_H * INDEX_DIM)

rotate_compressed = make_rope(1)

rotate_index_key = make_wide_rope(1, INDEX_DIM)


@pl.jit.inline
def prefill_compressor_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Retain FP32 weight residuals through high and residual accumulators.

    Three BF16 weight terms preserve FP32 weights. Bound each Cube reduction
    to one K chunk, then compensate the Vector sum of the staged GM products
    to reduce accumulation error near pooled BF16 rounding boundaries.
    Mixed producers use Vector startup and FP32 Vector stores.
    """
    tokens = pl.tensor.dim(x, 0)
    high_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    low_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    tail_weight = pl.create_tensor([D, HEAD_DIM], dtype=pl.BF16)
    high_product = pl.create_tensor([PREFILL_COMPRESSOR_SEGMENTS * tokens, HEAD_DIM], dtype=pl.FP32)
    residual_product = pl.create_tensor([PREFILL_COMPRESSOR_SEGMENTS * tokens, HEAD_DIM], dtype=pl.FP32)
    split_blocks = D // COMPRESSOR_K_TILE * (HEAD_DIM // N_TILE)
    split_workers = pl.min(split_blocks, 64)
    with pl.spmd(split_workers, name_hint="prefill_compressor_split") as split_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, split_blocks, split_workers):
            k0 = block // (HEAD_DIM // N_TILE) * COMPRESSOR_K_TILE
            n0 = block % (HEAD_DIM // N_TILE) * N_TILE
            value = weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
            high = pl.cast(value, pl.BF16, mode="rint")
            residual = pl.sub(value, pl.cast(high, pl.FP32))
            low = pl.cast(residual, pl.BF16, mode="rint")
            tail = pl.cast(pl.sub(residual, pl.cast(low, pl.FP32)), pl.BF16, mode="rint")
            high_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = high
            low_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = low
            tail_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE] = tail
    output_blocks = (num_tokens + M_TILE - 1) // M_TILE * (HEAD_DIM // N_TILE)
    blocks = output_blocks * PREFILL_COMPRESSOR_SEGMENTS
    # Synchronous workers must fit both the physical and simulator core count.
    workers = pl.min(blocks, pl.min(16, pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="prefill_compressor_cube", deps=[split_tid], sync_start=True) as cube_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            segment = block % PREFILL_COMPRESSOR_SEGMENTS
            tile_block = block // PREFILL_COMPRESSOR_SEGMENTS
            t0 = tile_block // (HEAD_DIM // N_TILE) * M_TILE
            n0 = tile_block % (HEAD_DIM // N_TILE) * N_TILE
            rows = pl.min(M_TILE, num_tokens - t0)
            high_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            residual_acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            for k0 in pl.range(segment * PREFILL_COMPRESSOR_K_CHUNK, (segment + 1) * PREFILL_COMPRESSOR_K_CHUNK, COMPRESSOR_K_TILE):
                a = pl.slice(x, [M_TILE, COMPRESSOR_K_TILE], [t0, k0], valid_shape=[rows, COMPRESSOR_K_TILE])
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                high = high_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                low = low_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                tail = tail_weight[k0 : k0 + COMPRESSOR_K_TILE, n0 : n0 + N_TILE]
                high_acc = pl.matmul_acc(high_acc, a, high, init_cond=(k0 == segment * PREFILL_COMPRESSOR_K_CHUNK))
                residual_acc = pl.matmul_acc(residual_acc, a, low, init_cond=(k0 == segment * PREFILL_COMPRESSOR_K_CHUNK))
                residual_acc = pl.matmul_acc(residual_acc, a, tail)
            high_product[segment * tokens + t0 : segment * tokens + t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(high_acc, 1.0), rows, N_TILE)
            residual_product[segment * tokens + t0 : segment * tokens + t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(residual_acc, 1.0), rows, N_TILE)
    with pl.spmd(pl.min(output_blocks, 16), name_hint="prefill_compressor_sum", deps=[cube_tid]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, output_blocks, pl.min(output_blocks, 16)):
            t0 = block // (HEAD_DIM // N_TILE) * M_TILE
            n0 = block % (HEAD_DIM // N_TILE) * N_TILE
            rows = pl.min(M_TILE, num_tokens - t0)
            high_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            low_sum = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            high_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            low_error = pl.full([M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for segment in pl.range(PREFILL_COMPRESSOR_SEGMENTS):
                high_part = pl.slice(high_product, [M_TILE, N_TILE], [segment * tokens + t0, n0], valid_shape=[rows, N_TILE])
                low_part = pl.slice(residual_product, [M_TILE, N_TILE], [segment * tokens + t0, n0], valid_shape=[rows, N_TILE])
                high_delta = pl.sub(high_part, high_error)
                high_next = pl.add(high_sum, high_delta)
                high_error = pl.sub(pl.sub(high_next, high_sum), high_delta)
                high_sum = high_next
                low_delta = pl.sub(low_part, low_error)
                low_next = pl.add(low_sum, low_delta)
                low_error = pl.sub(pl.sub(low_next, low_sum), low_delta)
                low_sum = low_next
            output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.add(high_sum, low_sum), rows, N_TILE)
    return output


@pl.jit.inline
def prefill_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_compressor_projection(x, wkv, kv_out, num_tokens)
    prefill_compressor_projection(x, wgate, score_out, num_tokens)
    return kv_out, score_out


_prefill_index_key_projection = make_bf16_projection_staged(HEAD_DIM, INDEX_DIM)
_prefill_index_weight_projection = make_bf16_projection_staged(D, INDEX_H)


@pl.jit.inline
def prefill_index_key_project(
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Stage the index-key projection before its normalization."""
    tokens = pl.tensor.dim(latent, 0)
    projected = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    _prefill_index_key_projection(latent, weight, projected, num_tokens)
    with pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="prefill_c2a_index_norm", deps=[pool_ready]) as norm_tid:
        t0 = pl.tile.get_block_idx() * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_DIM], [t0, 0], valid_shape=[rows, INDEX_DIM]), pl.FP32)
        square = pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / INDEX_DIM)
        inv = pl.rsqrt(pl.add(square, EPS), high_precision=True)
        gamma = pl.reshape(pl.cast(norm_weight[:], pl.FP32), [1, INDEX_DIM])
        normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, INDEX_DIM)
    return norm_tid


@pl.jit.inline
def prefill_indexer_weights(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, INDEX_H], pl.BF16],
    output: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Preserve the index-weight rounding boundaries with a staged projection."""
    tokens = pl.tensor.dim(x, 0)
    projected = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    _prefill_index_weight_projection(x, weight, projected, num_tokens)
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE, name_hint="prefill_c2a_index_weight_scale"):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.cast(pl.slice(projected, [M_TILE, INDEX_H], [t0, 0], valid_shape=[rows, INDEX_H]), pl.FP32)
        scaled = pl.cast(pl.mul(value, INDEX_DIM**-0.5), pl.BF16, mode="rint")
        result = pl.cast(pl.mul(pl.cast(scaled, pl.FP32), INDEX_H**-0.5), pl.BF16, mode="rint")
        output[t0 : t0 + M_TILE, :] = pl.set_validshape(result, rows, INDEX_H)
    return output


prefill_attend_sparse = make_attend_selected_chunk(vector_query=True)
c2a_index_query = make_wide_rope(INDEX_H, INDEX_DIM, max_workers=64, permute=True)

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
    prefill_c2a_full_qkv(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, qr, query, window_kv, num_tokens,
    )
    publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    prefill_compressor_project(x, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = compressor_ratio2(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    # Index keys come from the unrotated latent; the compressed cache stores the rotated one.
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    prefill_index_key_project(latent, index_wk, index_norm_weight, index_key, num_tokens, pool_tid)
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
    prefill_indexer_weights(x, index_weights_proj, index_weight, num_tokens)
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
        chunk_done = prefill_attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    prefill_o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return chunk_done


def golden_prefill_attn_c2a_full(
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
    """Reference the C2A Full leaf: attention plus every cache and state it publishes."""
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
        wo_a=wo_a,
        wo_a_scale=wo_a_scale,
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


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_full_partial(
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
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write FP32 head-TP partial output using zero-initialized windows and consecutive 1-based epochs."""
    # A later epoch must not overwrite cache or transport storage still being read.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
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
    return partial


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_full(
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
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Return replicated output for standalone C2A attention callers."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    prefill_attn_c2a_full_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b,
        wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
        index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
        compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight,
        compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_arrived, partial, num_tokens,
        attention_epoch,
    )
    prefill_tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
        attention_epoch,
    )
    return output


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
    prefill_c2a_qkv(
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
        chunk_done = prefill_attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    prefill_o_proj(attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens)
    return chunk_done


def golden_prefill_attn_c2a_reuse(
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
    compressed_indices: torch.Tensor,
) -> AttentionGoldenResult:
    """Reference the C2A Reuse leaf: attention plus every cache and state it publishes."""
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
        wo_a=wo_a,
        wo_a_scale=wo_a_scale,
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


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_reuse_partial(
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
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write FP32 head-TP partial output using zero-initialized windows and consecutive 1-based epochs."""
    # A later epoch must not overwrite cache or transport storage still being read.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_reuse_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    c2a_reuse_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        compressed_indices, partial, num_tokens, cache_ready,
    )
    return partial


@pl.jit.inline(auto_scope=False)
def prefill_attn_c2a_reuse(
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
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Return replicated output for standalone C2A attention callers."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    prefill_attn_c2a_reuse_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b,
        wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
        compressed_cache_scale, compressed_indices, output_arrived, partial, num_tokens,
        attention_epoch,
    )
    prefill_tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens,
        attention_epoch,
    )
    return output
