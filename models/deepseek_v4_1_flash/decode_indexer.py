# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

from pathlib import Path

from pypto.runtime import pto_isa_include_dir

from gemm import gemm_bf16_to_bf16, gemm_mxfp8_to_bf16
import pypto.language as pl
from config import (
    B_DYN, FLASH, INDEX_BLOCKS_DYN, INDEX_CACHE_GROUP, INDEX_POOL_BLOCKS_DYN,
    INDEX_POOL_POSITIONS_DYN, TABLE_DYN, T_DYN,
)


# model config
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
Q_LORA = FLASH.q_lora_rank
QUERY_WIDTH = INDEX_H * INDEX_DIM
ROPE_DIM = FLASH.qk_rope_head_dim
CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS

INDEX_KEY_EPS = FLASH.rms_norm_eps
_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP

# tiling
INDEX_PROJ_N_TILE = 128
INDEX_WEIGHT_K_SPLIT_TILE = 4
INDEX_WEIGHT_K_TILE = 256
INDEX_PROJ_K_TILE = 256
INDEX_KEY_BYTE_TILE = 256
SCORE_TILE = 256
SCORE_RUN_TILE = 6
SCORE_ROWS_TILE = SCORE_RUN_TILE * INDEX_H
SCORE_BATCH_TILE = 16
SCORE_SHARD_PANELS_TILE = 16
SCORE_SHARD_CACHE = SCORE_SHARD_PANELS_TILE * SCORE_TILE
SCORE_MERGE_CACHE = 16384
PROJ_M_TILE = 80
PROJ_K_TILE = 128
PROJ_N_TILE = 128


CANDIDATE_WORKER_TILE = 12
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF_TILE = 8192
CANDIDATE_SHORT_LEAF_TILE = 2048
CANDIDATE_MAX_LEAVES = (FLASH.max_position_embeddings // CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_LEAF_TILE + 1
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")

@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "idx_q.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "idx_q.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_q(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Out[pl.Tensor],
    output_scale: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Project and quantize query heads with two C-to-V transfers per cube worker."""
    ...


@pl.jit.inline(auto_scope=False)
def project_index_query_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(output, 0)
    padded = pl.tensor.dim(payload, 0)
    scale_chunks = padded // PROJ_M_TILE
    scale_chunk_width = PROJ_M_TILE * (Q_LORA // 32)
    a_backing = pl.reshape(activation_scale, [scale_chunks, scale_chunk_width])
    projected = pl.create_tensor([tokens, QUERY_WIDTH], dtype=pl.BF16)
    last_projection = ready
    for row in pl.range(0, num_tokens, PROJ_M_TILE):
        projection_scale_row = row // PROJ_M_TILE
        projected_rows, projection_tid = gemm_mxfp8_to_bf16(
            payload, a_backing, weight, weight_scale, projected, num_tokens,
            row, 0, 0,
            Q_LORA, QUERY_WIDTH, "idx_q_projection", PROJ_N_TILE, PROJ_K_TILE,
            1, projection_scale_row, row, PROJ_M_TILE, last_projection, 16, 2,
        )
        projected = projected_rows
        last_projection = projection_tid

    with pl.spmd(16, name_hint="idx_q_rope_quant_reg", deps=[last_projection]) as completed:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, 16):
            cos_row = pl.tile.load(cosine, [token, 0], [1, 32], target_memory=pl.MemorySpace.Vec)
            sin_row = pl.tile.load(sine, [token, 0], [1, 32], target_memory=pl.MemorySpace.Vec)
            cos_pad = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
            sin_pad = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
            cos_pad[0:1, 0:32] = cos_row
            sin_pad[0:1, 0:32] = sin_row
            cos_vec = pl.reshape(pl.tile.move(cos_pad, target_memory=pl.MemorySpace.Vec), [64])
            sin_vec = pl.reshape(pl.tile.move(sin_pad, target_memory=pl.MemorySpace.Vec), [64])
            for head in pl.range(INDEX_H):
                source = pl.tile.load(projected, [token, head * INDEX_DIM], [1, INDEX_DIM], target_memory=pl.MemorySpace.Vec)
                source_vec = pl.reshape(source, [INDEX_DIM])
                packed = pl.tile.full([128], dtype=pl.UINT8, value=0)
                packed = pl.tile.move(packed, target_memory=pl.MemorySpace.Vec)
                exponent_tile = pl.tile.full([128], dtype=pl.INT32, value=0)
                exponent_tile = pl.tile.move(exponent_tile, target_memory=pl.MemorySpace.Vec)
                scratch = pl.tile.full([64], dtype=pl.FP32, value=0.0)
                scratch = pl.tile.move(scratch, target_memory=pl.MemorySpace.Vec)
                with pl.reg_scope():
                    r0_mask = pl.reg.full_mask(lanes=64)
                    r0_first = pl.reg.active_mask(32, lanes=64)
                    r0_second = pl.reg.mask_not(r0_first)
                    r0_raw = pl.reg.load(source_vec, 0, lanes=64)
                    r0_half = pl.reg.interpret_cast(r0_raw, dtype=pl.UINT16)
                    r0_bits = pl.reg.convert(r0_half, dtype=pl.UINT32)
                    r0_bits = pl.reg.shift_lefts(r0_bits, 16, r0_mask)
                    r0_values = pl.reg.interpret_cast(r0_bits, dtype=pl.FP32)
                    r0_magnitude = pl.reg.abs(r0_values)
                    r0_lo_max = pl.reg.reduce_max(r0_magnitude, r0_first)
                    r0_hi_max = pl.reg.reduce_max(r0_magnitude, r0_second)
                    scratch = pl.reg.store(scratch, value=r0_lo_max)
                    r0_lo_values = pl.reg.load(scratch, lanes=64)
                    r0_zeros = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
                    r0_lo_values = pl.reg.gather_lanes(r0_lo_values, r0_zeros)
                    scratch = pl.reg.store(scratch, value=r0_hi_max)
                    r0_hi_values = pl.reg.load(scratch, lanes=64)
                    r0_hi_values = pl.reg.gather_lanes(r0_hi_values, r0_zeros)
                    r0_maximum = pl.reg.select(r0_first, r0_lo_values, r0_hi_values)
                    r0_minimum = pl.reg.broadcast(6.0 * 2.0**-126, lanes=64, dtype=pl.FP32)
                    r0_maximum = pl.reg.maximum(r0_maximum, r0_minimum)
                    r0_raw_scale = pl.reg.muls(r0_maximum, 1.0 / 6.0, r0_mask)
                    r0_scale_bits = pl.reg.interpret_cast(r0_raw_scale, dtype=pl.INT32)
                    r0_scale_bits = pl.reg.adds(r0_scale_bits, 8388607, r0_mask)
                    r0_exponent = pl.reg.shift_rights(r0_scale_bits, 23, r0_mask)
                    r0_factor_bits = pl.reg.shift_lefts(r0_exponent, 23, r0_mask)
                    r0_factor = pl.reg.interpret_cast(r0_factor_bits, dtype=pl.FP32)
                    r0_normalized = pl.reg.div(r0_values, r0_factor)
                    r0_magnitude = pl.reg.abs(r0_normalized)
                    r0_codes = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                    r0_selected0 = pl.reg.compares(r0_magnitude, 0.25, r0_mask, predicate="gt")
                    r0_level0 = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected0, r0_level0, r0_codes)
                    r0_selected1 = pl.reg.compares(r0_magnitude, 0.75, r0_mask, predicate="ge")
                    r0_level1 = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected1, r0_level1, r0_codes)
                    r0_selected2 = pl.reg.compares(r0_magnitude, 1.25, r0_mask, predicate="gt")
                    r0_level2 = pl.reg.broadcast(3, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected2, r0_level2, r0_codes)
                    r0_selected3 = pl.reg.compares(r0_magnitude, 1.75, r0_mask, predicate="ge")
                    r0_level3 = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected3, r0_level3, r0_codes)
                    r0_selected4 = pl.reg.compares(r0_magnitude, 2.5, r0_mask, predicate="gt")
                    r0_level4 = pl.reg.broadcast(5, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected4, r0_level4, r0_codes)
                    r0_selected5 = pl.reg.compares(r0_magnitude, 3.5, r0_mask, predicate="ge")
                    r0_level5 = pl.reg.broadcast(6, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected5, r0_level5, r0_codes)
                    r0_selected6 = pl.reg.compares(r0_magnitude, 5.0, r0_mask, predicate="gt")
                    r0_level6 = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
                    r0_codes = pl.reg.select(r0_selected6, r0_level6, r0_codes)
                    r0_value_bits = pl.reg.interpret_cast(r0_normalized, dtype=pl.INT32)
                    r0_sign = pl.reg.shift_rights(r0_value_bits, 28, r0_mask)
                    r0_sign_mask = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                    r0_sign = pl.reg.bitwise_and(r0_sign, r0_sign_mask)
                    r0_codes = pl.reg.bitwise_or(r0_codes, r0_sign)
                    r0_even_codes, r0_odd_codes = pl.reg.dintlv(r0_codes, r0_codes, r0_mask)
                    r0_odd_codes = pl.reg.shift_lefts(r0_odd_codes, 4, r0_mask)
                    r0_packed_codes = pl.reg.bitwise_or(r0_even_codes, r0_odd_codes)
                    r0_signed = pl.reg.convert(r0_packed_codes, dtype=pl.INT8, saturate=False)
                    r0_bytes = pl.reg.interpret_cast(r0_signed, dtype=pl.UINT8)
                    packed = pl.reg.store(packed, 0, r0_bytes, r0_first)
                    exponent_tile = pl.reg.store(exponent_tile, 0, r0_exponent)
                with pl.reg_scope():
                    r1_mask = pl.reg.full_mask(lanes=64)
                    r1_first = pl.reg.active_mask(32, lanes=64)
                    r1_second = pl.reg.mask_not(r1_first)
                    r1_raw = pl.reg.load(source_vec, 64, lanes=64)
                    r1_half = pl.reg.interpret_cast(r1_raw, dtype=pl.UINT16)
                    r1_bits = pl.reg.convert(r1_half, dtype=pl.UINT32)
                    r1_bits = pl.reg.shift_lefts(r1_bits, 16, r1_mask)
                    r1_values = pl.reg.interpret_cast(r1_bits, dtype=pl.FP32)
                    r1_even, r1_odd = pl.reg.dintlv(r1_values, r1_values, r1_mask)
                    r1_cos = pl.reg.load(cos_vec, lanes=64)
                    r1_sin = pl.reg.load(sin_vec, lanes=64)
                    r1_ec = pl.reg.mul(r1_even, r1_cos)
                    r1_os = pl.reg.mul(r1_odd, r1_sin)
                    r1_es = pl.reg.mul(r1_even, r1_sin)
                    r1_oc = pl.reg.mul(r1_odd, r1_cos)
                    r1_real = pl.reg.sub(r1_ec, r1_os)
                    r1_imag = pl.reg.add(r1_es, r1_oc)
                    r1_rotated, r1_unused = pl.reg.intlv(r1_real, r1_imag, r1_mask)
                    r1_rotation_bits = pl.reg.interpret_cast(r1_rotated, dtype=pl.UINT32)
                    r1_high = pl.reg.shift_rights(r1_rotation_bits, 16, r1_mask)
                    r1_one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                    r1_lsb = pl.reg.bitwise_and(r1_high, r1_one)
                    r1_bias = pl.reg.broadcast(32767, lanes=64, dtype=pl.UINT32)
                    r1_rounded = pl.reg.add(pl.reg.add(r1_rotation_bits, r1_bias), r1_lsb)
                    r1_keep = pl.reg.broadcast(4294901760, lanes=64, dtype=pl.UINT32)
                    r1_rounded = pl.reg.bitwise_and(r1_rounded, r1_keep)
                    r1_values = pl.reg.interpret_cast(r1_rounded, dtype=pl.FP32)
                    r1_magnitude = pl.reg.abs(r1_values)
                    r1_lo_max = pl.reg.reduce_max(r1_magnitude, r1_first)
                    r1_hi_max = pl.reg.reduce_max(r1_magnitude, r1_second)
                    scratch = pl.reg.store(scratch, value=r1_lo_max)
                    r1_lo_values = pl.reg.load(scratch, lanes=64)
                    r1_zeros = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
                    r1_lo_values = pl.reg.gather_lanes(r1_lo_values, r1_zeros)
                    scratch = pl.reg.store(scratch, value=r1_hi_max)
                    r1_hi_values = pl.reg.load(scratch, lanes=64)
                    r1_hi_values = pl.reg.gather_lanes(r1_hi_values, r1_zeros)
                    r1_maximum = pl.reg.select(r1_first, r1_lo_values, r1_hi_values)
                    r1_minimum = pl.reg.broadcast(6.0 * 2.0**-126, lanes=64, dtype=pl.FP32)
                    r1_maximum = pl.reg.maximum(r1_maximum, r1_minimum)
                    r1_raw_scale = pl.reg.muls(r1_maximum, 1.0 / 6.0, r1_mask)
                    r1_scale_bits = pl.reg.interpret_cast(r1_raw_scale, dtype=pl.INT32)
                    r1_scale_bits = pl.reg.adds(r1_scale_bits, 8388607, r1_mask)
                    r1_exponent = pl.reg.shift_rights(r1_scale_bits, 23, r1_mask)
                    r1_factor_bits = pl.reg.shift_lefts(r1_exponent, 23, r1_mask)
                    r1_factor = pl.reg.interpret_cast(r1_factor_bits, dtype=pl.FP32)
                    r1_normalized = pl.reg.div(r1_values, r1_factor)
                    r1_magnitude = pl.reg.abs(r1_normalized)
                    r1_codes = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                    r1_selected0 = pl.reg.compares(r1_magnitude, 0.25, r1_mask, predicate="gt")
                    r1_level0 = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected0, r1_level0, r1_codes)
                    r1_selected1 = pl.reg.compares(r1_magnitude, 0.75, r1_mask, predicate="ge")
                    r1_level1 = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected1, r1_level1, r1_codes)
                    r1_selected2 = pl.reg.compares(r1_magnitude, 1.25, r1_mask, predicate="gt")
                    r1_level2 = pl.reg.broadcast(3, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected2, r1_level2, r1_codes)
                    r1_selected3 = pl.reg.compares(r1_magnitude, 1.75, r1_mask, predicate="ge")
                    r1_level3 = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected3, r1_level3, r1_codes)
                    r1_selected4 = pl.reg.compares(r1_magnitude, 2.5, r1_mask, predicate="gt")
                    r1_level4 = pl.reg.broadcast(5, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected4, r1_level4, r1_codes)
                    r1_selected5 = pl.reg.compares(r1_magnitude, 3.5, r1_mask, predicate="ge")
                    r1_level5 = pl.reg.broadcast(6, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected5, r1_level5, r1_codes)
                    r1_selected6 = pl.reg.compares(r1_magnitude, 5.0, r1_mask, predicate="gt")
                    r1_level6 = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
                    r1_codes = pl.reg.select(r1_selected6, r1_level6, r1_codes)
                    r1_value_bits = pl.reg.interpret_cast(r1_normalized, dtype=pl.INT32)
                    r1_sign = pl.reg.shift_rights(r1_value_bits, 28, r1_mask)
                    r1_sign_mask = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                    r1_sign = pl.reg.bitwise_and(r1_sign, r1_sign_mask)
                    r1_codes = pl.reg.bitwise_or(r1_codes, r1_sign)
                    r1_even_codes, r1_odd_codes = pl.reg.dintlv(r1_codes, r1_codes, r1_mask)
                    r1_odd_codes = pl.reg.shift_lefts(r1_odd_codes, 4, r1_mask)
                    r1_packed_codes = pl.reg.bitwise_or(r1_even_codes, r1_odd_codes)
                    r1_signed = pl.reg.convert(r1_packed_codes, dtype=pl.INT8, saturate=False)
                    r1_bytes = pl.reg.interpret_cast(r1_signed, dtype=pl.UINT8)
                    packed = pl.reg.store(packed, 32, r1_bytes, r1_first)
                    exponent_tile = pl.reg.store(exponent_tile, 64, r1_exponent)
                scale = pl.tile.full([64], dtype=pl.UINT8, value=0)
                scale = pl.tile.move(scale, target_memory=pl.MemorySpace.Vec)
                with pl.reg_scope():
                    r2_mask = pl.reg.full_mask(lanes=64)
                    r2_four = pl.reg.active_mask(4, lanes=64)
                    r2_offsets = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                    r2_offsets = pl.reg.muls(r2_offsets, 32, r2_mask)
                    r2_values = pl.reg.gather(exponent_tile, r2_offsets, r2_four)
                    r2_bytes_i8 = pl.reg.convert(r2_values, dtype=pl.INT8, saturate=False)
                    r2_bytes_u8 = pl.reg.interpret_cast(r2_bytes_i8, dtype=pl.UINT8)
                    scale = pl.reg.store(scale, value=r2_bytes_u8, mask=r2_four)
                payload_row = pl.reshape(packed, [1, 128])
                output = pl.store(pl.set_validshape(payload_row, 1, 64), [token, head * 64], output)
                scale_row = pl.reshape(scale, [1, 64])
                encoded = pl.reinterpret_view(scale_row, pl.FP8E8M0)
                output_scale = pl.store(pl.set_validshape(encoded, 1, 4), [token, head * 4], output_scale)
                pl.system.barrier(pipe=pl.PipeType.ALL)
    return completed


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_score_topk(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Out[pl.Tensor],
    runs: pl.Tensor,
    candidate_keys: pl.Out[pl.Tensor],
    candidate_indices: pl.Out[pl.Tensor],
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    width: pl.Scalar[pl.INDEX],
    run_stride: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Score history shards and merge local TopK candidates within one MIX."""
    ...


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "index_score_topk.cpp", dual_aiv_dispatch=True,
               include_dirs=(pto_isa_include_dir(),))
def _idx_pool_score_topk(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Out[pl.Tensor],
    runs: pl.Tensor,
    candidate_keys: pl.Out[pl.Tensor],
    candidate_indices: pl.Out[pl.Tensor],
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    width: pl.Scalar[pl.INDEX],
    run_stride: pl.Scalar[pl.INDEX],
    table_width: pl.Scalar[pl.INDEX],
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Score gathered pool shards and merge local TopK candidates within one MIX."""
    ...


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
    tokens = pl.tensor.dim(pool_block_ids, 0)
    positions = pl.tensor.dim(index_scores, 1)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > tokens:
        active_tokens = pl.cast(tokens, pl.INDEX)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    if block_count <= pl.min(capacity, CANDIDATE_TOPK_BLOCKS):
        all_workers = pl.max(1, pl.min(tokens, CANDIDATE_WORKER_TILE))
        with pl.spmd(all_workers, name_hint="pool_all") as all_tid:
            worker = pl.tile.get_block_idx()
            lane_tmp = pl.create_tile([1, 256], dtype=pl.FP32)
            lanes = pl.tile.arange(0, [1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, tmp=lane_tmp)
            for token in pl.range(worker, tokens, all_workers):
                ids = pl.tile.full([1, CANDIDATE_TOPK_BLOCKS], dtype=pl.INT32, value=-1)
                if token < active_tokens:
                    visible = pl.min(pl.max(pl.read(compressed_lens, [token]), 0), positions)
                    visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                    valid = pl.tile.cmps(lanes, pl.cast(visible_blocks, pl.INT32), cmp_type=2)
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
        score_workers = pl.min(score_work_items, 32)

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
        with pl.spmd(tokens, name_hint="pool_select", deps=[score_tid]) as selected_tid:
            token = pl.tile.get_block_idx()
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
                        pl.tile.write(ordered_scores, [0, lane], -pl.cast(block_id, pl.FP32))
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


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "idx_k_norm_rope_publish.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _idx_k_norm_rope_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, rotate, and publish group-32 MXFP4 index keys with E8M0 scales."""
    ...


@pl.jit.inline(auto_scope=False)
def publish_index_rows(
    source: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    kv_ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse index RMSNorm, BF16 RoPE, and cache publication with one writer per page."""
    workers = pl.max(1, pl.min(num_tokens, 4))
    with pl.spmd(workers, name_hint="idx_k_norm_rope_publish", deps=[cache_ready, kv_ready]) as published:
        cache = _idx_k_norm_rope_publish(source, weight, cosine, sine, slots, cache, scales, num_tokens)
    return published


@pl.jit.inline(auto_scope=False)
def publish_index_rows_pypto(
    source: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    scales: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, _SCALE_COUNT], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    kv_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish index keys with the PyPTO register functional implementation.

    C1A/C2A decode keep using publish_index_rows until this implementation
    passes performance acceptance.
    """
    workers = pl.max(1, pl.min(num_tokens, 4))
    cache_rows = pl.tensor.dim(cache, 0) * 128
    cache_flat = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(scales, [cache_rows, _SCALE_COUNT])
    with pl.spmd(workers, name_hint="idx_k_norm_rope_publish", deps=[cache_ready, kv_ready]) as published:
        worker = pl.tile.get_block_idx()
        gamma_tile = pl.load(weight, [0], [INDEX_DIM], target_memory=pl.MemorySpace.Vec)
        for token in pl.range(num_tokens):
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                if (slot // 128) % workers == worker:
                    source_tile = pl.load(source, [token, 0], [1, INDEX_DIM], target_memory=pl.MemorySpace.Vec)
                    source_tile = pl.reshape(source_tile, [INDEX_DIM])
                    cos_tile = pl.load(cosine, [token, 0], [1, ROPE_DIM // 2], target_memory=pl.MemorySpace.Vec)
                    sin_tile = pl.load(sine, [token, 0], [1, ROPE_DIM // 2], target_memory=pl.MemorySpace.Vec)
                    cos_tile = pl.reshape(cos_tile, [ROPE_DIM // 2])
                    sin_tile = pl.reshape(sin_tile, [ROPE_DIM // 2])
                    payload = pl.tile.full([INDEX_KEY_BYTE_TILE], dtype=pl.UINT8, value=0)
                    scale_bytes = pl.tile.full([INDEX_KEY_BYTE_TILE], dtype=pl.UINT8, value=0)
                    with pl.reg_scope():
                        full = pl.reg.full_mask(lanes=64)
                        lower = pl.reg.active_mask(32, lanes=64)
                        index = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
                        bit_one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                        bf16_mask = pl.reg.broadcast(4294901760, lanes=64, dtype=pl.UINT32)
                        cos_index = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                        cos_mask = pl.reg.broadcast(31, lanes=64, dtype=pl.UINT32)
                        cos_index = pl.reg.bitwise_and(cos_index, cos_mask)
                        cosv = pl.reg.gather(cos_tile, cos_index, full)
                        sinv = pl.reg.gather(sin_tile, cos_index, full)

                        # Expand BF16 bits with existing integer register operations.
                        source0_bf = pl.reg.load(source_tile, offset=0, lanes=64)
                        source0_u16 = pl.reg.interpret_cast(source0_bf, dtype=pl.UINT16)
                        source0_bits = pl.reg.convert(source0_u16, dtype=pl.UINT32)
                        source0_bits = pl.reg.shift_lefts(source0_bits, 16, full)
                        source0 = pl.reg.interpret_cast(source0_bits, dtype=pl.FP32)
                        gamma0_bf = pl.reg.load(gamma_tile, offset=0, lanes=64)
                        gamma0_u16 = pl.reg.interpret_cast(gamma0_bf, dtype=pl.UINT16)
                        gamma0_bits = pl.reg.convert(gamma0_u16, dtype=pl.UINT32)
                        gamma0_bits = pl.reg.shift_lefts(gamma0_bits, 16, full)
                        gamma0 = pl.reg.interpret_cast(gamma0_bits, dtype=pl.FP32)
                        square0 = pl.reg.mul(source0, source0)
                        source1_bf = pl.reg.load(source_tile, offset=64, lanes=64)
                        source1_u16 = pl.reg.interpret_cast(source1_bf, dtype=pl.UINT16)
                        source1_bits = pl.reg.convert(source1_u16, dtype=pl.UINT32)
                        source1_bits = pl.reg.shift_lefts(source1_bits, 16, full)
                        source1 = pl.reg.interpret_cast(source1_bits, dtype=pl.FP32)
                        gamma1_bf = pl.reg.load(gamma_tile, offset=64, lanes=64)
                        gamma1_u16 = pl.reg.interpret_cast(gamma1_bf, dtype=pl.UINT16)
                        gamma1_bits = pl.reg.convert(gamma1_u16, dtype=pl.UINT32)
                        gamma1_bits = pl.reg.shift_lefts(gamma1_bits, 16, full)
                        gamma1 = pl.reg.interpret_cast(gamma1_bits, dtype=pl.FP32)
                        square1 = pl.reg.mul(source1, source1)

                        # Match the two adjacent-pair 64-lane sum trees in the C++ kernel.
                        stride0 = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                        indices0 = pl.reg.bitwise_xor(index, stride0)
                        peer0_0 = pl.reg.gather_lanes(square0, indices0)
                        square0 = pl.reg.add(square0, peer0_0)
                        peer1_0 = pl.reg.gather_lanes(square1, indices0)
                        square1 = pl.reg.add(square1, peer1_0)
                        stride1 = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                        indices1 = pl.reg.bitwise_xor(index, stride1)
                        peer0_1 = pl.reg.gather_lanes(square0, indices1)
                        square0 = pl.reg.add(square0, peer0_1)
                        peer1_1 = pl.reg.gather_lanes(square1, indices1)
                        square1 = pl.reg.add(square1, peer1_1)
                        stride2 = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
                        indices2 = pl.reg.bitwise_xor(index, stride2)
                        peer0_2 = pl.reg.gather_lanes(square0, indices2)
                        square0 = pl.reg.add(square0, peer0_2)
                        peer1_2 = pl.reg.gather_lanes(square1, indices2)
                        square1 = pl.reg.add(square1, peer1_2)
                        stride3 = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                        indices3 = pl.reg.bitwise_xor(index, stride3)
                        peer0_3 = pl.reg.gather_lanes(square0, indices3)
                        square0 = pl.reg.add(square0, peer0_3)
                        peer1_3 = pl.reg.gather_lanes(square1, indices3)
                        square1 = pl.reg.add(square1, peer1_3)
                        stride4 = pl.reg.broadcast(16, lanes=64, dtype=pl.INT32)
                        indices4 = pl.reg.bitwise_xor(index, stride4)
                        peer0_4 = pl.reg.gather_lanes(square0, indices4)
                        square0 = pl.reg.add(square0, peer0_4)
                        peer1_4 = pl.reg.gather_lanes(square1, indices4)
                        square1 = pl.reg.add(square1, peer1_4)
                        stride5 = pl.reg.broadcast(32, lanes=64, dtype=pl.INT32)
                        indices5 = pl.reg.bitwise_xor(index, stride5)
                        peer0_5 = pl.reg.gather_lanes(square0, indices5)
                        square0 = pl.reg.add(square0, peer0_5)
                        peer1_5 = pl.reg.gather_lanes(square1, indices5)
                        square1 = pl.reg.add(square1, peer1_5)
                        square_sum = pl.reg.add(square0, square1)
                        mean = pl.reg.muls(square_sum, 0.0078125, full)
                        shifted = pl.reg.adds(mean, INDEX_KEY_EPS, full)
                        denominator = pl.reg.sqrt(shifted)
                        inverse = pl.reg.div(one, denominator)

                        # Normalize segment 0; preserve the BF16 RNE boundary in FP32 bits.
                        values0 = pl.reg.mul(source0, inverse)
                        values0 = pl.reg.mul(values0, gamma0)
                        values0_bits = pl.reg.interpret_cast(values0, dtype=pl.UINT32)
                        values0_high = pl.reg.shift_rights(values0_bits, 16, full)
                        values0_lsb = pl.reg.bitwise_and(values0_high, bit_one)
                        values0_bias = pl.reg.adds(values0_lsb, 32767, full)
                        values0_bits = pl.reg.add(values0_bits, values0_bias)
                        values0_bits = pl.reg.bitwise_and(values0_bits, bf16_mask)
                        values0 = pl.reg.interpret_cast(values0_bits, dtype=pl.FP32)
                        maximum0 = pl.reg.abs(values0)
                        max_peer0_0 = pl.reg.gather_lanes(maximum0, indices0)
                        maximum0 = pl.reg.maximum(maximum0, max_peer0_0)
                        max_peer0_1 = pl.reg.gather_lanes(maximum0, indices1)
                        maximum0 = pl.reg.maximum(maximum0, max_peer0_1)
                        max_peer0_2 = pl.reg.gather_lanes(maximum0, indices2)
                        maximum0 = pl.reg.maximum(maximum0, max_peer0_2)
                        max_peer0_3 = pl.reg.gather_lanes(maximum0, indices3)
                        maximum0 = pl.reg.maximum(maximum0, max_peer0_3)
                        max_peer0_4 = pl.reg.gather_lanes(maximum0, indices4)
                        maximum0 = pl.reg.maximum(maximum0, max_peer0_4)
                        maximum0 = pl.reg.maximums(maximum0, 7.052966104933725e-38, full)
                        raw_scale0 = pl.reg.muls(maximum0, 0.16666666666666666, full)
                        scale_bits0 = pl.reg.interpret_cast(raw_scale0, dtype=pl.INT32)
                        scale_bits0 = pl.reg.adds(scale_bits0, 8388607, full)
                        exponent0 = pl.reg.shift_rights(scale_bits0, 23, full)
                        factor_bits0 = pl.reg.shift_lefts(exponent0, 23, full)
                        factor0 = pl.reg.interpret_cast(factor_bits0, dtype=pl.FP32)
                        normalized0 = pl.reg.div(values0, factor0)
                        magnitude0 = pl.reg.abs(normalized0)
                        codes0 = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                        selected0_1 = pl.reg.compares(magnitude0, 0.25, full, predicate="gt")
                        level0_1 = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_1, level0_1, codes0)
                        selected0_2 = pl.reg.compares(magnitude0, 0.75, full, predicate="ge")
                        level0_2 = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_2, level0_2, codes0)
                        selected0_3 = pl.reg.compares(magnitude0, 1.25, full, predicate="gt")
                        level0_3 = pl.reg.broadcast(3, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_3, level0_3, codes0)
                        selected0_4 = pl.reg.compares(magnitude0, 1.75, full, predicate="ge")
                        level0_4 = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_4, level0_4, codes0)
                        selected0_5 = pl.reg.compares(magnitude0, 2.5, full, predicate="gt")
                        level0_5 = pl.reg.broadcast(5, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_5, level0_5, codes0)
                        selected0_6 = pl.reg.compares(magnitude0, 3.5, full, predicate="ge")
                        level0_6 = pl.reg.broadcast(6, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_6, level0_6, codes0)
                        selected0_7 = pl.reg.compares(magnitude0, 5.0, full, predicate="gt")
                        level0_7 = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
                        codes0 = pl.reg.select(selected0_7, level0_7, codes0)
                        sign0 = pl.reg.interpret_cast(normalized0, dtype=pl.INT32)
                        sign0 = pl.reg.shift_rights(sign0, 28, full)
                        sign_mask0 = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                        sign0 = pl.reg.bitwise_and(sign0, sign_mask0)
                        codes0 = pl.reg.bitwise_or(codes0, sign0)
                        low0, high0 = pl.reg.dintlv(codes0, codes0, full)
                        high0 = pl.reg.shift_lefts(high0, 4, lower)
                        packed0 = pl.reg.bitwise_or(low0, high0)
                        packed0 = pl.reg.convert(packed0, dtype=pl.INT8, saturate=False)
                        packed0 = pl.reg.interpret_cast(packed0, dtype=pl.UINT8)
                        payload = pl.reg.store(payload, offset=0, value=packed0, mask=lower)

                        # Normalize segment 1; preserve the BF16 RNE boundary in FP32 bits.
                        values1 = pl.reg.mul(source1, inverse)
                        values1 = pl.reg.mul(values1, gamma1)
                        values1_bits = pl.reg.interpret_cast(values1, dtype=pl.UINT32)
                        values1_high = pl.reg.shift_rights(values1_bits, 16, full)
                        values1_lsb = pl.reg.bitwise_and(values1_high, bit_one)
                        values1_bias = pl.reg.adds(values1_lsb, 32767, full)
                        values1_bits = pl.reg.add(values1_bits, values1_bias)
                        values1_bits = pl.reg.bitwise_and(values1_bits, bf16_mask)
                        values1 = pl.reg.interpret_cast(values1_bits, dtype=pl.FP32)
                        # Rotate adjacent pairs in the trailing 64 lanes without FMA contraction.
                        even, odd = pl.reg.dintlv(values1, values1, full)
                        real_a = pl.reg.mul(even, cosv)
                        real_b = pl.reg.mul(odd, sinv)
                        imag_a = pl.reg.mul(even, sinv)
                        imag_b = pl.reg.mul(odd, cosv)
                        real = pl.reg.sub(real_a, real_b)
                        imag = pl.reg.add(imag_a, imag_b)
                        rotated, unused = pl.reg.intlv(real, imag, full)
                        rotated_bits = pl.reg.interpret_cast(rotated, dtype=pl.UINT32)
                        rotated_high = pl.reg.shift_rights(rotated_bits, 16, full)
                        rotated_lsb = pl.reg.bitwise_and(rotated_high, bit_one)
                        rotated_bias = pl.reg.adds(rotated_lsb, 32767, full)
                        rotated_bits = pl.reg.add(rotated_bits, rotated_bias)
                        rotated_bits = pl.reg.bitwise_and(rotated_bits, bf16_mask)
                        rotated = pl.reg.interpret_cast(rotated_bits, dtype=pl.FP32)
                        maximum1 = pl.reg.abs(rotated)
                        max_peer1_0 = pl.reg.gather_lanes(maximum1, indices0)
                        maximum1 = pl.reg.maximum(maximum1, max_peer1_0)
                        max_peer1_1 = pl.reg.gather_lanes(maximum1, indices1)
                        maximum1 = pl.reg.maximum(maximum1, max_peer1_1)
                        max_peer1_2 = pl.reg.gather_lanes(maximum1, indices2)
                        maximum1 = pl.reg.maximum(maximum1, max_peer1_2)
                        max_peer1_3 = pl.reg.gather_lanes(maximum1, indices3)
                        maximum1 = pl.reg.maximum(maximum1, max_peer1_3)
                        max_peer1_4 = pl.reg.gather_lanes(maximum1, indices4)
                        maximum1 = pl.reg.maximum(maximum1, max_peer1_4)
                        maximum1 = pl.reg.maximums(maximum1, 7.052966104933725e-38, full)
                        raw_scale1 = pl.reg.muls(maximum1, 0.16666666666666666, full)
                        scale_bits1 = pl.reg.interpret_cast(raw_scale1, dtype=pl.INT32)
                        scale_bits1 = pl.reg.adds(scale_bits1, 8388607, full)
                        exponent1 = pl.reg.shift_rights(scale_bits1, 23, full)
                        factor_bits1 = pl.reg.shift_lefts(exponent1, 23, full)
                        factor1 = pl.reg.interpret_cast(factor_bits1, dtype=pl.FP32)
                        normalized1 = pl.reg.div(rotated, factor1)
                        magnitude1 = pl.reg.abs(normalized1)
                        codes1 = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                        selected1_1 = pl.reg.compares(magnitude1, 0.25, full, predicate="gt")
                        level1_1 = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_1, level1_1, codes1)
                        selected1_2 = pl.reg.compares(magnitude1, 0.75, full, predicate="ge")
                        level1_2 = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_2, level1_2, codes1)
                        selected1_3 = pl.reg.compares(magnitude1, 1.25, full, predicate="gt")
                        level1_3 = pl.reg.broadcast(3, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_3, level1_3, codes1)
                        selected1_4 = pl.reg.compares(magnitude1, 1.75, full, predicate="ge")
                        level1_4 = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_4, level1_4, codes1)
                        selected1_5 = pl.reg.compares(magnitude1, 2.5, full, predicate="gt")
                        level1_5 = pl.reg.broadcast(5, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_5, level1_5, codes1)
                        selected1_6 = pl.reg.compares(magnitude1, 3.5, full, predicate="ge")
                        level1_6 = pl.reg.broadcast(6, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_6, level1_6, codes1)
                        selected1_7 = pl.reg.compares(magnitude1, 5.0, full, predicate="gt")
                        level1_7 = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
                        codes1 = pl.reg.select(selected1_7, level1_7, codes1)
                        sign1 = pl.reg.interpret_cast(normalized1, dtype=pl.INT32)
                        sign1 = pl.reg.shift_rights(sign1, 28, full)
                        sign_mask1 = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                        sign1 = pl.reg.bitwise_and(sign1, sign_mask1)
                        codes1 = pl.reg.bitwise_or(codes1, sign1)
                        low1, high1 = pl.reg.dintlv(codes1, codes1, full)
                        high1 = pl.reg.shift_lefts(high1, 4, lower)
                        packed1 = pl.reg.bitwise_or(low1, high1)
                        packed1 = pl.reg.convert(packed1, dtype=pl.INT8, saturate=False)
                        packed1 = pl.reg.interpret_cast(packed1, dtype=pl.UINT8)
                        payload = pl.reg.store(payload, offset=32, value=packed1, mask=lower)

                        # Pack four scale codes into one aligned 32-bit word.
                        first_index = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                        second_index = pl.reg.broadcast(32, lanes=64, dtype=pl.INT32)
                        exp0 = pl.reg.gather_lanes(exponent0, first_index)
                        exp1 = pl.reg.gather_lanes(exponent0, second_index)
                        exp2 = pl.reg.gather_lanes(exponent1, first_index)
                        exp3 = pl.reg.gather_lanes(exponent1, second_index)
                        exp1 = pl.reg.shift_lefts(exp1, 8, full)
                        exp2 = pl.reg.shift_lefts(exp2, 16, full)
                        exp3 = pl.reg.shift_lefts(exp3, 24, full)
                        lower_word = pl.reg.bitwise_or(exp0, exp1)
                        upper_word = pl.reg.bitwise_or(exp2, exp3)
                        scale_word = pl.reg.bitwise_or(lower_word, upper_word)
                        scale_value = pl.reg.interpret_cast(scale_word, dtype=pl.UINT8)
                        scale_mask = pl.reg.active_mask(4, lanes=256)
                        scale_bytes = pl.reg.store(scale_bytes, offset=0, value=scale_value, mask=scale_mask)
                    payload_row = pl.reshape(payload, [1, INDEX_KEY_BYTE_TILE])
                    scale_row = pl.reshape(scale_bytes, [1, INDEX_KEY_BYTE_TILE])
                    payload_row = pl.set_validshape(payload_row, 1, INDEX_DIM // 2)
                    scale_row = pl.set_validshape(scale_row, 1, _SCALE_COUNT)
                    encoded = pl.reinterpret_view(scale_row, pl.FP8E8M0)
                    pl.store(payload_row, [slot, 0], cache_flat)
                    pl.store(encoded, [slot, 0], scale_flat)
                    # Serialize repeated slots and adjacent four-byte scale writes.
                    pl.system.barrier(pipe=pl.PipeType.ALL)
    return published

@pl.jit.inline(auto_scope=False)
def _plan_index_runs(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    runs: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    skip_short: pl.constexpr,
):
    """Publish request groups synchronously before using their count to size shards."""
    run_count = pl.cast(0, pl.INT32)
    run_start = pl.cast(0, pl.INT32)
    previous_request = pl.cast(-1, pl.INT32)
    maximum_visible = pl.cast(0, pl.INT32)
    for token in pl.range(num_tokens):
        request = pl.read(request_ids, [token])
        if token > 0:
            if request != previous_request or token - run_start == 6:
                pl.write(runs, [0, run_count], run_start)
                pl.write(runs, [1, run_count], pl.cast(token, pl.INT32))
                pl.write(runs, [2, run_count], maximum_visible)
                run_count = run_count + pl.cast(1, pl.INT32)
                run_start = pl.cast(token, pl.INT32)
                maximum_visible = pl.cast(0, pl.INT32)
        visible = pl.cast(pl.max(0, pl.min(pl.read(lengths, [token]), width)), pl.INT32)
        if skip_short:
            if visible <= INDEX_TOPK:
                visible = pl.cast(0, pl.INT32)
        maximum_visible = pl.max(maximum_visible, visible)
        pl.write(runs, [3, token + 1], run_count)
        previous_request = request
    if num_tokens > 0:
        pl.write(runs, [0, run_count], run_start)
        pl.write(runs, [1, run_count], num_tokens)
        pl.write(runs, [2, run_count], maximum_visible)
        run_count = run_count + pl.cast(1, pl.INT32)
    pl.write(runs, [3, 0], run_count)
    return run_count


@pl.jit.inline(auto_scope=False)
def score_select_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Route identity candidate pools to paged MIX and sparse pools to gathered MIX."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(scores, 1)
    table_width = pl.tensor.dim(block_table, 1)
    request_count = pl.tensor.dim(block_table, 0)
    cache_pages = pl.tensor.dim(index_cache, 0)
    flags = pl.create_tensor([32, 32], dtype=pl.INT32)
    with pl.spmd(32, name_hint="idx_plan_pool", deps=[cache_ready, pool_ready]) as plan_tid:
        worker = pl.tile.get_block_idx()
        dense = pl.cast(1, pl.INT32)
        mismatch = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
        for token in pl.range(worker, num_tokens, 32):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            visible = pl.max(0, pl.read(lengths, [token]))
            request = pl.read(request_ids, [token])
            if count != (visible + 7) // 8 or visible > width or visible > table_width * 128:
                dense = pl.cast(0, pl.INT32)
            if request < 0 or request >= request_count:
                dense = pl.cast(0, pl.INT32)
            if dense == 1:
                for start in pl.range(0, count, 1024):
                    active = pl.min(1024, count - start)
                    raw = pl.load(pool_block_ids, [token, start], [1, 1024], valid_shape=[1, active])
                    ids = pl.tile.fillpad(raw, pad_value=pl.PadValue.zero)
                    lanes = pl.tile.arange(0, [1, 1024], dtype=pl.INT32)
                    expected = pl.add(lanes, pl.cast(start, pl.INT32))
                    error = pl.abs(pl.cast(pl.sub(ids, expected), pl.FP32))
                    valid = pl.tile.cmps(lanes, pl.cast(active, pl.INT32), cmp_type=2)
                    error = pl.tile.select(valid, error, 0.0)
                    mismatch = pl.add(mismatch, pl.col_sum(pl.reshape(error, [16, 64])))
                page_count = (visible + 127) // 128
                for start in pl.range(0, page_count, 128):
                    active = pl.min(128, page_count - start)
                    raw_pages = pl.load(block_table, [request, start], [1, 128], valid_shape=[1, active])
                    pages = pl.tile.fillpad(raw_pages, pad_value=pl.PadValue.zero)
                    bad = pl.tile.cmps(pages, pl.cast(cache_pages, pl.INT32), cmp_type=5)
                    page_errors = pl.tile.select(bad, pl.tile.full([1, 128], dtype=pl.FP32, value=1.0), 0.0)
                    mismatch = pl.add(mismatch, pl.col_sum(pl.reshape(page_errors, [2, 64])))
        sums = pl.col_sum(pl.reshape(mismatch, [8, 8]))
        for lane in pl.unroll(8):
            if pl.tile.read(sums, [0, lane]) != 0.0:
                dense = pl.cast(0, pl.INT32)
        pl.write(flags, [worker, 0], dense)
    dense = pl.cast(1, pl.INT32)
    for worker in pl.range(32):
        dense = dense * pl.read(flags, [worker, 0])
    if dense != 0:
        tokens = pl.tensor.dim(output, 0)
        dense_width = pl.min(width, table_width * 128)
        paged_scores = pl.create_tensor([tokens, dense_width], dtype=pl.FP32)
        selected, scores_tid = score_select_paged(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, paged_scores, output,
            num_tokens, plan_tid, query_ready, weights_ready, capacity, False,
        )
        result = selected
    else:
        result = _score_select_pool(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, pool_block_ids, pool_block_counts, scores, output,
            num_tokens, cache_ready, query_ready, weights_ready, plan_tid,
        )
    return result


@pl.jit.inline(auto_scope=False)
def _score_select_pool(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Gather candidate blocks and select compact scores with physical-row mappings."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    tokens = pl.tensor.dim(output, 0)
    width = capacity * CANDIDATE_BLOCK_SIZE
    panels = (width + 255) // 256
    gathered_rows = panels * 256
    total_rows = tokens * gathered_rows
    total_scales = total_rows * _SCALE_COUNT
    cache_pages = pl.tensor.dim(index_cache, 0)
    cache_rows = cache_pages * 128
    cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(index_cache_scale, [cache_rows // 8, 32])
    payloads = pl.create_tensor([total_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    scales = pl.create_tensor([1, total_scales], dtype=pl.UINT8)
    valid = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    physical = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    runs = pl.create_tensor([4, tokens + 1], dtype=pl.INT32)
    table_width = pl.tensor.dim(block_table, 1)
    request_count = pl.tensor.dim(block_table, 0)
    with pl.spmd(32, name_hint="idx_gather_pool", deps=[cache_ready, pool_ready]) as gather_tid:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, num_tokens * panels, 32):
            token = work // panels
            panel = work % panels
            slot_begin = panel * 32
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            request = pl.read(request_ids, [token])
            visible = pl.max(0, pl.read(lengths, [token]))
            packed = pl.tile.full([256, INDEX_DIM // 2], dtype=pl.INT8, value=0)
            raw_scale = pl.tile.full([1, 256 * _SCALE_COUNT], dtype=pl.INT8, value=127)
            valid_lanes = pl.tile.full([1, 256], dtype=pl.INT32, value=0)
            physical_lanes = pl.tile.full([1, 256], dtype=pl.INT32, value=-1)
            for block in pl.range(32):
                slot = slot_begin + block
                if slot < count:
                    logical_block = pl.read(pool_block_ids, [token, slot])
                    if logical_block >= 0 and logical_block < (visible + 7) // 8:
                        logical_page = logical_block // 16
                        if request >= 0 and request < request_count and logical_page < table_width:
                            page = pl.read(block_table, [pl.cast(request, pl.INDEX), logical_page])
                            if page >= 0 and page < cache_pages:
                                physical_row = pl.cast(page, pl.INDEX) * 128 + (logical_block % 16) * 8
                                payload_raw = pl.load(cache_flat, [physical_row, 0], [8, INDEX_DIM // 2], target_memory=pl.MemorySpace.Vec)
                                payload = pl.reinterpret_view(payload_raw, pl.INT8)
                                packed[block * 8:block * 8 + 8, :] = payload
                                scales_raw = pl.load(scale_flat, [physical_row // 8, 0], [1, 32], target_memory=pl.MemorySpace.Vec)
                                block_scale_bytes = pl.reinterpret_view(scales_raw, pl.UINT8)
                                block_scales = pl.reinterpret_view(block_scale_bytes, pl.INT8)
                                raw_scale[:, block * 32:block * 32 + 32] = block_scales
                                lanes = pl.tile.arange(0, [1, 8], dtype=pl.INT32)
                                remaining = pl.cast(pl.min(8, visible - logical_block * 8), pl.INT32)
                                lane_valid = pl.tile.cmps(lanes, remaining, cmp_type=2)
                                ones = pl.tile.full([1, 8], dtype=pl.INT32, value=1)
                                block_valid = pl.tile.select(lane_valid, ones, 0)
                                block_physical = pl.add(lanes, pl.cast(physical_row, pl.INT32))
                                block_physical = pl.tile.select(lane_valid, block_physical, -1)
                                valid_lanes[:, block * 8:block * 8 + 8] = block_valid
                                physical_lanes[:, block * 8:block * 8 + 8] = block_physical
            row_offset = token * gathered_rows + panel * 256
            packed_bytes = pl.reinterpret_view(packed, pl.UINT8)
            pl.store(packed_bytes, [row_offset, 0], payloads)
            scale_bytes = pl.reinterpret_view(raw_scale, pl.UINT8)
            pl.store(scale_bytes, [0, row_offset * _SCALE_COUNT], scales)
            pl.store(valid_lanes, [0, row_offset], valid)
            pl.store(physical_lanes, [0, row_offset], physical)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_plan_pool_runs", deps=[pool_ready]) as runs_tid:
        for token in pl.range(num_tokens):
            count = pl.max(0, pl.min(pl.read(pool_block_counts, [token]), capacity))
            pl.write(runs, [0, token], pl.cast(token, pl.INT32))
            pl.write(runs, [1, token], pl.cast(token + 1, pl.INT32))
            pl.write(runs, [2, token], pl.cast(count * 8, pl.INT32))
            pl.write(runs, [3, token + 1], pl.cast(token, pl.INT32))
        pl.write(runs, [3, 0], num_tokens)
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    candidate_keys = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.INT32)
    run_stride = pl.tensor.dim(runs, 1)
    query_rows = pl.slice(query, [tokens, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [tokens, INDEX_H * _SCALE_COUNT], [0, 0])
    weight_rows = pl.slice(weights, [tokens, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [tokens])
    length_rows = pl.reshape(lengths, [tokens])
    output_rows = pl.reshape(output, [tokens, INDEX_TOPK])
    score_rows = pl.slice(scores, [tokens, width], [0, 0])
    with pl.spmd(workers, name_hint="idx_score_topk", sync_start=True, deps=[gather_tid, runs_tid, query_ready, weights_ready]) as mix_tid:
        _idx_pool_score_topk(
            query_rows, scale_rows, weight_rows, payloads, scales,
            block_table, request_rows, length_rows, output_rows, runs, candidate_keys, candidate_indices,
            valid, physical, score_rows, width, run_stride, table_width, num_tokens, gathered_rows,
            pl.cast(0, pl.INT32), workers, pl.cast(0, pl.INT32),
        )
    return mix_tid


@pl.jit.inline(auto_scope=False)
def score_select_paged(
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
    c2a: pl.constexpr,
):
    """Score history shards and merge their TopK within one MIX."""
    width = pl.tensor.dim(scores, 1)
    capacity = pl.tensor.dim(output, 0)
    runs = pl.create_tensor([4, capacity + 1], dtype=pl.INT32)
    run_count = _plan_index_runs(request_ids, lengths, runs, num_tokens, width, c2a)
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    panels = (width + 255) // 256
    shards = workers
    if panels > 100:
        shards = pl.max(workers, (run_count * panels + 99) // 100)
    block_count = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    emit_scores = pl.cast(0, pl.INT32)
    if not c2a:
        if block_count > pl.min(pool_capacity, CANDIDATE_TOPK_BLOCKS):
            emit_scores = pl.cast(1, pl.INT32)
    run_stride = pl.tensor.dim(runs, 1)
    table_width = pl.tensor.dim(block_table, 1)
    active_tokens = pl.cast(num_tokens, pl.INT32)
    direct_short = pl.cast(0, pl.INT32)
    if c2a:
        direct_short = pl.cast(1, pl.INT32)
    query_rows = pl.slice(query, [capacity, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [capacity, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], [0, 0])
    weight_rows = pl.slice(weights, [capacity, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [capacity])
    length_rows = pl.reshape(lengths, [capacity])
    mixed_output = pl.reshape(output, [capacity, INDEX_TOPK])
    score_rows = pl.slice(scores, [capacity, width], [0, 0])
    candidate_keys = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.INT32)
    with pl.spmd(workers, name_hint="idx_score_topk", sync_start=True, deps=[cache_ready, query_ready, weights_ready]) as mix_tid:
        mixed_output = _idx_score_topk(
            query_rows, scale_rows, weight_rows, index_cache, index_cache_scale,
            block_table, request_rows, length_rows, mixed_output, runs, candidate_keys, candidate_indices,
            block_table, block_table, score_rows, width, run_stride, table_width, active_tokens, 0,
            emit_scores, shards, direct_short,
        )
    return mix_tid, mix_tid


@pl.jit.inline(auto_scope=False)
def _plan_score_runs(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_counts: pl.Tensor,
    runs: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    capacity: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="idx_score_plan_runs", deps=[ready]) as planned:
        if pool:
            for token in pl.range(num_tokens):
                count = pl.max(0, pl.min(pl.read(pool_counts, [token]), capacity))
                pl.write(runs, [0, token], pl.cast(token, pl.INT32))
                pl.write(runs, [1, token], pl.cast(token + 1, pl.INT32))
                pl.write(runs, [2, token], pl.cast(pl.min(width, count * 8), pl.INT32))
                pl.write(runs, [3, token + 1], pl.cast(token, pl.INT32))
            pl.write(runs, [3, 0], num_tokens)
        else:
            run_count = pl.cast(0, pl.INT32)
            run_start = pl.cast(0, pl.INT32)
            previous_request = pl.cast(-1, pl.INT32)
            maximum_visible = pl.cast(0, pl.INT32)
            for token in pl.range(num_tokens):
                request = pl.read(request_ids, [token])
                if token > 0:
                    if request != previous_request or token - run_start == SCORE_RUN_TILE:
                        pl.write(runs, [0, run_count], run_start)
                        pl.write(runs, [1, run_count], pl.cast(token, pl.INT32))
                        pl.write(runs, [2, run_count], maximum_visible)
                        run_count = run_count + pl.cast(1, pl.INT32)
                        run_start = pl.cast(token, pl.INT32)
                        maximum_visible = pl.cast(0, pl.INT32)
                visible = pl.cast(
                    pl.max(0, pl.min(pl.read(lengths, [token]), width)),
                    pl.INT32,
                )
                maximum_visible = pl.max(maximum_visible, visible)
                pl.write(runs, [3, token + 1], run_count)
                previous_request = request
            if num_tokens > 0:
                pl.write(runs, [0, run_count], run_start)
                pl.write(runs, [1, run_count], num_tokens)
                pl.write(runs, [2, run_count], maximum_visible)
                run_count = run_count + pl.cast(1, pl.INT32)
            pl.write(runs, [3, 0], run_count)
    return planned

@pl.jit.inline(auto_scope=False)
def _score_index_rows(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_ids: pl.Tensor,
    pool_counts: pl.Tensor,
    scores: pl.Tensor,
    physical: pl.Tensor,
    runs: pl.Tensor,
    runs_ready: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
    pool: pl.constexpr,
):
    tokens = pl.tensor.dim(query, 0)
    query_rows = tokens * INDEX_H
    query_packed = pl.reshape(query, [query_rows, INDEX_DIM // 2])
    pages = pl.tensor.dim(cache, 0)
    cache_rows = pages * 128
    packed_cache = pl.reshape(cache, [cache_rows, INDEX_DIM // 2])

    width = pl.tensor.dim(scores, 1)
    panels = (width + SCORE_TILE - 1) // SCORE_TILE
    capacity = pl.tensor.dim(pool_ids, 1)
    table_width = pl.tensor.dim(table, 1)
    requests = pl.tensor.dim(table, 0)
    scale_rows = cache_rows // 8
    query_scale_width = tokens * INDEX_H * 4
    flat_scales = pl.reshape(cache_scale, [scale_rows, 32])
    flat_query_scales = pl.reshape(query_scale, [1, query_scale_width])
    run_count = pl.cast(
        pl.read(runs, [3, 0]),
        pl.INDEX,
    )
    query_storage_rows = pl.max(1, run_count) * SCORE_ROWS_TILE
    query_storage = pl.create_tensor([query_storage_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    query_fp4 = pl.reinterpret_view(query_storage, pl.FP4E2M1X2)
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    work_count = run_count * panels
    batch_capacity = pl.max(1, pl.min(work_count, workers * SCORE_BATCH_TILE))
    dot_rows = batch_capacity * SCORE_ROWS_TILE
    key_rows = batch_capacity * SCORE_TILE
    panel_slots = batch_capacity * 16
    dot_backing = pl.create_tensor([dot_rows, SCORE_TILE], dtype=pl.BF16)
    staged_keys = pl.create_tensor([key_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    packed_keys = pl.reinterpret_view(staged_keys, pl.FP4E2M1X2)
    staged_scales = pl.create_tensor([batch_capacity, SCORE_TILE * 4], dtype=pl.UINT8)
    panel_valid = pl.create_tensor([panel_slots], dtype=pl.INT32)
    shards = pl.max(workers, (work_count + SCORE_SHARD_PANELS_TILE - 1) // SCORE_SHARD_PANELS_TILE)
    owners = pl.min(shards, work_count)
    previous_batch = runs_ready
    batch_limit = pl.max(1, work_count)
    for batch_begin in pl.range(0, batch_limit, batch_capacity):
        batch_end = pl.min(work_count, batch_begin + batch_capacity)
        with pl.spmd(workers, name_hint="idx_score_gather_reg", deps=[cache_ready, pool_ready, query_ready, previous_batch]) as gather_tid:
            worker = pl.tile.get_block_idx()
            for work in pl.range(batch_begin + worker, batch_end, workers):
                staging_slot = work - batch_begin
                run_id = work // panels
                begin = (work % panels) * SCORE_TILE
                first_token = pl.cast(
                    pl.read(runs, [0, run_id]),
                    pl.INDEX,
                )
                end_token = pl.cast(
                    pl.read(runs, [1, run_id]),
                    pl.INDEX,
                )
                maximum = pl.cast(
                    pl.read(runs, [2, run_id]),
                    pl.INDEX,
                )
                if begin == 0:
                    query_pad_rows = (end_token - first_token) * INDEX_H
                    query_pad_bytes = pl.tile.load(
                        query_packed, [first_token * INDEX_H, 0],
                        [SCORE_ROWS_TILE, INDEX_DIM // 2], valid_shape=[query_pad_rows, INDEX_DIM // 2],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    query_padded_bytes = pl.fillpad(query_pad_bytes, pad_value=pl.PadValue.zero)
                    query_full_bytes = pl.set_validshape(query_padded_bytes, SCORE_ROWS_TILE, INDEX_DIM // 2)
                    query_storage = pl.store(query_full_bytes, [run_id * SCORE_ROWS_TILE, 0], query_storage)
                request = pl.read(request_ids, [first_token])
                keys = pl.tile.full([SCORE_TILE, INDEX_DIM // 2], dtype=pl.INT8, value=0)
                scale_bytes = pl.tile.full([1, SCORE_TILE * 4], dtype=pl.INT8, value=127)
                mapping = pl.tile.full([SCORE_TILE], dtype=pl.INT32, value=-1)
                mapping = pl.tile.move(mapping, target_memory=pl.MemorySpace.Vec)
                have_dot = pl.cast(0, pl.INT32)
                if pool:
                    visible = pl.max(0, pl.read(lengths, [first_token]))
                    count = pl.max(0, pl.min(pl.read(pool_counts, [first_token]), capacity))
                    for part in pl.range(SCORE_TILE // 8):
                        slot = begin // 8 + part
                        if slot < count:
                            logical_block = pl.read(pool_ids, [first_token, slot])
                            logical = pl.cast(logical_block, pl.INDEX) * 8
                            if logical_block >= 0 and logical < visible and request >= 0 and request < requests:
                                logical_page = logical // 128
                                if logical_page < table_width:
                                    page_id = pl.read(table, [pl.cast(request, pl.INDEX), logical_page])
                                    if page_id >= 0 and page_id < pages:
                                        row = pl.cast(page_id, pl.INDEX) * 128 + logical % 128
                                        raw_key = pl.tile.load(packed_cache, [row, 0], [8, INDEX_DIM // 2], target_memory=pl.MemorySpace.Vec)
                                        keys = pl.tile.assemble(keys, pl.reinterpret_view(raw_key, pl.INT8), [part * 8, 0])
                                        raw_scale = pl.tile.load(flat_scales, [row // 8, 0], [1, 32], target_memory=pl.MemorySpace.Vec)
                                        raw_bytes = pl.reinterpret_view(pl.reinterpret_view(raw_scale, pl.UINT8), pl.INT8)
                                        scale_bytes = pl.tile.assemble(scale_bytes, raw_bytes, [0, part * 32])
                                        valid_rows = pl.min(8, visible - logical)
                                        row_i32 = pl.cast(row, pl.INT32)
                                        mapping_base = pl.cast(part * 8, pl.UINT32)
                                        with pl.reg_scope():
                                            pool_valid = pl.reg.create_group_mask(valid_rows, num_groups=1, group_size=64)
                                            pool_rows = pl.reg.index_gen(row_i32, lanes=64, dtype=pl.INT32)
                                            pool_addresses = pl.reg.index_gen(mapping_base, lanes=64, dtype=pl.UINT32)
                                            mapping = pl.reg.scatter(mapping, pool_addresses, pool_rows, pool_valid)
                                        have_dot = pl.cast(1, pl.INT32)
                else:
                    for piece in pl.range(2):
                        logical = begin + piece * 128
                        if logical < maximum and request >= 0 and request < requests:
                            logical_page = logical // 128
                            if logical_page < table_width:
                                page_id = pl.read(table, [pl.cast(request, pl.INDEX), logical_page])
                                if page_id >= 0 and page_id < pages:
                                    row = pl.cast(page_id, pl.INDEX) * 128
                                    raw_page = pl.tile.load(packed_cache, [row, 0], [128, INDEX_DIM // 2], target_memory=pl.MemorySpace.Vec)
                                    keys = pl.tile.assemble(keys, pl.reinterpret_view(raw_page, pl.INT8), [piece * 128, 0])
                                    raw_page_scale = pl.tile.load(flat_scales, [row // 8, 0], [16, 32], target_memory=pl.MemorySpace.Vec)
                                    page_scale_bytes = pl.reinterpret_view(pl.reinterpret_view(raw_page_scale, pl.UINT8), pl.INT8)
                                    page_scale_row = pl.reshape(page_scale_bytes, [1, 512])
                                    scale_bytes = pl.tile.assemble(scale_bytes, page_scale_row, [0, piece * 512])
                                    page_row_i32 = pl.cast(row, pl.INT32)
                                    page_mapping_base = pl.cast(piece * 128, pl.UINT32)
                                    with pl.reg_scope():
                                        page_mask = pl.reg.full_mask(lanes=64)
                                        for page_half in pl.hardware_loop(2):
                                            page_rows = pl.reg.index_gen(page_row_i32, lanes=64, dtype=pl.INT32)
                                            page_rows = pl.reg.adds(page_rows, pl.cast(page_half * 64, pl.INT32), page_mask)
                                            page_addresses = pl.reg.index_gen(page_mapping_base, lanes=64, dtype=pl.UINT32)
                                            page_addresses = pl.reg.adds(page_addresses, pl.cast(page_half * 64, pl.UINT32), page_mask)
                                            _page_mapping_write = pl.reg.scatter(mapping, page_addresses, page_rows, page_mask)
                                    have_dot = pl.cast(1, pl.INT32)
                staged_keys = pl.store(pl.reinterpret_view(keys, pl.UINT8), [staging_slot * SCORE_TILE, 0], staged_keys)
                staged_scales = pl.store(pl.reinterpret_view(scale_bytes, pl.UINT8), [staging_slot, 0], staged_scales)
                pl.write(panel_valid, [staging_slot * 16], have_dot)
                for token in pl.range(first_token, end_token):
                    token_mapping = mapping
                    if not pool:
                        token_mapping = pl.tile.full([SCORE_TILE], dtype=pl.INT32, value=-1)
                        token_mapping = pl.tile.move(token_mapping, target_memory=pl.MemorySpace.Vec)
                        visible_i32 = pl.cast(
                            pl.max(0, pl.read(lengths, [token])),
                            pl.INT32,
                        )
                        logical_i32 = pl.cast(begin, pl.INT32)
                        with pl.reg_scope():
                            token_mask = pl.reg.full_mask(lanes=64)
                            token_invalid = pl.reg.broadcast(-1, lanes=64, dtype=pl.INT32)
                            for token_part in pl.hardware_loop(SCORE_TILE // 64):
                                token_ids = pl.reg.load(mapping, token_part * 64, lanes=64)
                                token_logical = pl.reg.index_gen(logical_i32, lanes=64, dtype=pl.INT32)
                                token_logical = pl.reg.adds(token_logical, pl.cast(token_part * 64, pl.INT32), token_mask)
                                token_visible = pl.reg.compares(token_logical, visible_i32, token_mask, predicate="lt")
                                token_ids = pl.reg.select(token_visible, token_ids, token_invalid)
                                _token_mapping_write = pl.reg.store(token_mapping, token_part * 64, token_ids)
                    mapping_row = pl.reshape(token_mapping, [1, SCORE_TILE])
                    active = pl.min(SCORE_TILE, width - begin)
                    physical = pl.store(pl.set_validshape(mapping_row, 1, active), [token, begin], physical)

        with pl.spmd(workers, name_hint="idx_score_dot", deps=[gather_tid, query_ready], optimizations=[pl.cross_core_slot(slot_num=1)]) as dot_tid:
            worker = pl.tile.get_block_idx()
            for shard in pl.range(worker, owners, workers):
                first_work = pl.max(batch_begin, work_count * shard // owners)
                end_work = pl.min(batch_end, work_count * (shard + 1) // owners)
                if first_work < end_work:
                    first_run = first_work // panels
                    end_run = (end_work + panels - 1) // panels
                    for run_id in pl.range(first_run, end_run):
                        first_panel = pl.max(0, first_work - run_id * panels)
                        end_panel = pl.min(panels, end_work - run_id * panels)
                        first_token = pl.cast(
                            pl.read(runs, [0, run_id]),
                            pl.INDEX,
                        )
                        end_token = pl.cast(
                            pl.read(runs, [1, run_id]),
                            pl.INDEX,
                        )
                        valid_query_scales = (end_token - first_token) * INDEX_H * 4
                        raw_query_scale = pl.tile.load(
                            flat_query_scales, [0, first_token * INDEX_H * 4],
                            [1, SCORE_ROWS_TILE * 4], valid_shape=[1, valid_query_scales],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        padded_query_scale = pl.fillpad(raw_query_scale, pad_value=pl.PadValue.zero)
                        full_query_scale = pl.set_validshape(padded_query_scale, 1, SCORE_ROWS_TILE * 4)
                        query_bytes = pl.reinterpret_view(full_query_scale, pl.UINT8)
                        query_temporary = pl.create_tile([1, 128], dtype=pl.UINT8)
                        query_zz = pl.tmov_x2zz(query_bytes, query_temporary, group_axis=1, dst_rows=SCORE_ROWS_TILE, dst_cols=4)
                        query_mat0 = pl.tile.load(
                            query_fp4, [run_id * SCORE_ROWS_TILE, 0], [96, INDEX_DIM // 2],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        query_left0 = pl.tile.move(query_mat0, target_memory=pl.MemorySpace.Left)
                        query_mat1 = pl.tile.load(
                            query_fp4, [run_id * SCORE_ROWS_TILE + 96, 0], [96, INDEX_DIM // 2],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        query_left1 = pl.tile.move(query_mat1, target_memory=pl.MemorySpace.Left)
                        for panel in pl.range(first_panel, end_panel):
                            work = run_id * panels + panel
                            staging_slot = work - batch_begin
                            if pl.read(panel_valid, [staging_slot * 16]) != 0:
                                dot_keys = pl.tile.load(
                                    packed_keys, [staging_slot * SCORE_TILE, 0],
                                    [SCORE_TILE, INDEX_DIM // 2],
                                    target_memory=pl.MemorySpace.Mat,
                                )
                                dot_scale_bytes = pl.tile.load(staged_scales, [staging_slot, 0], [1, SCORE_TILE * 4], target_memory=pl.MemorySpace.Vec)
                                scale_temporary = pl.create_tile([1, 128], dtype=pl.UINT8)
                                key_zz = pl.tmov_x2zz(dot_scale_bytes, scale_temporary, group_axis=1, dst_rows=SCORE_TILE, dst_cols=4)
                                key_scale_nn = pl.tile.transpose_view(pl.reinterpret_view(key_zz, pl.FP8E8M0))
                                keys_mat = pl.tile.transpose_view(dot_keys)
                                keys_right = pl.tile.move(keys_mat, target_memory=pl.MemorySpace.Right)
                                key_scale_mat = pl.tile.move(
                                    key_scale_nn, target_memory=pl.MemorySpace.Mat,
                                    blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.col_major,
                                )
                                key_right_scale = pl.tile.move(key_scale_mat, target_memory=pl.MemorySpace.RightScale)
                                query_scale0 = pl.tile.slice(pl.reinterpret_view(query_zz, pl.FP8E8M0), [96, 4], [0, 0])
                                query_scale_mat0 = pl.tile.move(
                                    query_scale0, target_memory=pl.MemorySpace.Mat,
                                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                                )
                                query_left_scale0 = pl.tile.move(query_scale_mat0, target_memory=pl.MemorySpace.LeftScale)
                                dot0 = pl.matmul_mx(query_left0, query_left_scale0, keys_right, key_right_scale)
                                dot_backing = pl.store(dot0, [staging_slot * SCORE_ROWS_TILE, 0], dot_backing)
                                query_scale1 = pl.tile.slice(pl.reinterpret_view(query_zz, pl.FP8E8M0), [96, 4], [96, 0])
                                query_scale_mat1 = pl.tile.move(
                                    query_scale1, target_memory=pl.MemorySpace.Mat,
                                    blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major,
                                )
                                query_left_scale1 = pl.tile.move(query_scale_mat1, target_memory=pl.MemorySpace.LeftScale)
                                dot1 = pl.matmul_mx(query_left1, query_left_scale1, keys_right, key_right_scale)
                                dot_backing = pl.store(dot1, [staging_slot * SCORE_ROWS_TILE + 96, 0], dot_backing)

        with pl.spmd(workers, name_hint="idx_score_reduce", deps=[dot_tid, weights_ready]) as completed:
            worker = pl.tile.get_block_idx()
            for work in pl.range(batch_begin + worker, batch_end, workers):
                staging_slot = work - batch_begin
                run_id = work // panels
                begin = (work % panels) * SCORE_TILE
                first_token = pl.cast(
                    pl.read(runs, [0, run_id]),
                    pl.INDEX,
                )
                end_token = pl.cast(
                    pl.read(runs, [1, run_id]),
                    pl.INDEX,
                )
                for run_row in pl.range(end_token - first_token):
                    token = first_token + run_row
                    active = pl.min(SCORE_TILE, width - begin)
                    reduce_dot = pl.tile.full([INDEX_H, SCORE_TILE], dtype=pl.BF16, value=0.0)
                    if pl.read(panel_valid, [staging_slot * 16]) != 0:
                        reduce_dot = pl.tile.load(
                            dot_backing, [staging_slot * SCORE_ROWS_TILE + run_row * INDEX_H, 0],
                            [INDEX_H, SCORE_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                    weight_row = pl.tile.load(weights, [token, 0], [1, INDEX_H], target_memory=pl.MemorySpace.Vec)
                    weight_pad = pl.tile.full([1, 64], dtype=pl.BF16, value=0.0)
                    weight_pad = pl.tile.assemble(weight_pad, weight_row, [0, 0])
                    weight_vec = pl.reshape(pl.tile.move(weight_pad, target_memory=pl.MemorySpace.Vec), [64])
                    total = pl.tile.full([SCORE_TILE], dtype=pl.FP32, value=0.0)
                    total = pl.tile.move(total, target_memory=pl.MemorySpace.Vec)
                    reduce_dot = pl.reshape(reduce_dot, [INDEX_H * SCORE_TILE])
                    reduce_mapping = pl.tile.load(
                        physical, [token, begin],
                        [1, SCORE_TILE], valid_shape=[1, active],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    reduce_mapping = pl.fillpad(reduce_mapping, pad_value=pl.PadValue.min)
                    mapping_vec = pl.reshape(pl.set_validshape(reduce_mapping, 1, SCORE_TILE), [SCORE_TILE])
                    with pl.reg_scope():
                        r0_mask64 = pl.reg.full_mask(lanes=64)
                        r0_mask = pl.reg.full_mask(lanes=128)
                        r0_raw_weights = pl.reg.load(weight_vec, lanes=64)
                        r0_weight_bits16 = pl.reg.interpret_cast(r0_raw_weights, dtype=pl.UINT16)
                        r0_weight_bits = pl.reg.convert(r0_weight_bits16, dtype=pl.UINT32)
                        r0_weight_bits = pl.reg.shift_lefts(r0_weight_bits, 16, r0_mask64)
                        r0_weight_values = pl.reg.interpret_cast(r0_weight_bits, dtype=pl.FP32)
                        r0_weight_values = pl.reg.muls(r0_weight_values, 0.015625, r0_mask64)
                        r0_scaled_bits = pl.reg.interpret_cast(r0_weight_values, dtype=pl.UINT32)
                        r0_high = pl.reg.shift_rights(r0_scaled_bits, 16, r0_mask64)
                        r0_one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                        r0_lsb = pl.reg.bitwise_and(r0_high, r0_one)
                        r0_bias = pl.reg.broadcast(32767, lanes=64, dtype=pl.UINT32)
                        r0_rounded = pl.reg.add(pl.reg.add(r0_scaled_bits, r0_bias), r0_lsb)
                        r0_halfwords = pl.reg.interpret_cast(r0_rounded, dtype=pl.UINT16)
                        r0_even, r0_weights_bits = pl.reg.dintlv(r0_halfwords, r0_halfwords, r0_mask)
                        r0_zero = pl.reg.broadcast(0.0, lanes=128, dtype=pl.BF16)
                        for reduce_part in pl.hardware_loop(SCORE_TILE // 128):
                            r0_acc = r0_zero
                            for head in pl.unroll(INDEX_H):
                                r0_dots = pl.reg.load(reduce_dot, head * SCORE_TILE + reduce_part * 128, lanes=128)
                                r0_positive = pl.reg.maximum(r0_dots, r0_zero)
                                r0_indices = pl.reg.broadcast(pl.cast(head, pl.UINT16), lanes=128, dtype=pl.UINT16)
                                r0_repeated_bits = pl.reg.gather_lanes(r0_weights_bits, r0_indices)
                                r0_repeated = pl.reg.interpret_cast(r0_repeated_bits, dtype=pl.BF16)
                                r0_acc = pl.reg.mula(r0_acc, r0_positive, r0_repeated)
                            r0_acc = pl.reg.add(r0_acc, r0_zero)
                            r0_total_bits16 = pl.reg.interpret_cast(r0_acc, dtype=pl.UINT16)
                            r0_total_bits = pl.reg.convert(r0_total_bits16, dtype=pl.UINT32)
                            r0_total_bits = pl.reg.shift_lefts(r0_total_bits, 16, r0_mask)
                            r0_values = pl.reg.interpret_cast(r0_total_bits, dtype=pl.FP32)
                            r0_physical_ids = pl.reg.load(mapping_vec, reduce_part * 128, lanes=128)
                            r0_valid = pl.reg.compares(r0_physical_ids, 0, r0_mask, predicate="ge")
                            r0_invalid_bits = pl.reg.broadcast(-8388608, lanes=128, dtype=pl.INT32)
                            r0_invalid = pl.reg.interpret_cast(r0_invalid_bits, dtype=pl.FP32)
                            r0_values = pl.reg.select(r0_valid, r0_values, r0_invalid)
                            _score_write = pl.reg.store(total, reduce_part * 128, r0_values)
                    total_row = pl.reshape(total, [1, SCORE_TILE])
                    scores = pl.store(pl.set_validshape(total_row, 1, active), [token, begin], scores)
        previous_batch = completed
    return previous_batch

@pl.jit.inline(auto_scope=False)
def _select_index_topk_sharded(
    scores: pl.Tensor,
    physical: pl.Tensor,
    lengths: pl.Tensor,
    runs: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    direct_short: pl.constexpr,
):
    width = pl.tensor.dim(scores, 1)
    capacity = pl.tensor.dim(scores, 0)
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    run_count = pl.cast(
        pl.read(runs, [3, 0]),
        pl.INT32,
    )
    panels = pl.cast((width + 255) // 256, pl.INT32)
    total = pl.cast(run_count * panels, pl.INT32)
    owners = pl.cast(
        pl.min(shards, total),
        pl.INT32,
    )
    candidate_width = shards * INDEX_TOPK
    candidate_keys = pl.create_tensor([capacity, candidate_width], dtype=pl.UINT16)
    candidate_ids = pl.create_tensor([capacity, candidate_width], dtype=pl.INT32)
    with pl.spmd(workers, name_hint="idx_topk_shards_reg", deps=[ready]) as local_done:
        worker = pl.tile.get_block_idx()
        for shard in pl.range(worker, shards, workers):
            if shard < owners:
                first = pl.cast(total * shard // owners, pl.INT32)
                finish = pl.cast(total * (shard + 1) // owners, pl.INT32)
                first_run = first // panels
                last_run = (finish - 1) // panels
                for run_id in pl.range(first_run, last_run + 1):
                    first_panel = pl.max(first, run_id * panels) - run_id * panels
                    last_panel = pl.min(finish, (run_id + 1) * panels) - run_id * panels
                    logical_begin = pl.cast(first_panel * 256, pl.INT32)
                    local_width = pl.cast(
                        pl.min((last_panel - first_panel) * 256, width - logical_begin),
                        pl.INT32,
                    )
                    run_begin = pl.cast(
                        pl.read(runs, [0, run_id]),
                        pl.INT32,
                    )
                    run_end = pl.cast(
                        pl.read(runs, [1, run_id]),
                        pl.INT32,
                    )
                    for token in pl.range(run_begin, run_end):
                        compute_local = pl.cast(1, pl.INT32)
                        if direct_short:
                            if pl.read(lengths, [token]) <= INDEX_TOPK:
                                compute_local = pl.cast(0, pl.INT32)
                        if compute_local == 1:
                            cache_raw = pl.tile.load(
                                scores, [token, logical_begin],
                                [1, SCORE_SHARD_CACHE], valid_shape=[1, pl.cast(local_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            cache_pad = pl.fillpad(cache_raw, pad_value=pl.PadValue.min)
                            cache_full = pl.set_validshape(cache_pad, 1, SCORE_SHARD_CACHE)
                            cache_values = pl.reshape(cache_full, [SCORE_SHARD_CACHE])
                            cached_keys = pl.tile.full([SCORE_SHARD_CACHE], dtype=pl.UINT16, value=127)
                            cached_keys = pl.tile.move(cached_keys, target_memory=pl.MemorySpace.Vec)
                            cached_high = pl.tile.full([256], dtype=pl.UINT16, value=0)
                            cached_high = pl.tile.move(cached_high, target_memory=pl.MemorySpace.Vec)
                            cache_segments = pl.cast((local_width + 63) // 64, pl.INT32)
                            cache_pack = pl.tile.full([128], dtype=pl.UINT16, value=0)
                            cache_pack = pl.tile.move(cache_pack, target_memory=pl.MemorySpace.Vec)
                            with pl.reg_scope():
                                cache_mask = pl.reg.full_mask(lanes=64)
                                cache_pack_mask = pl.reg.full_mask(lanes=128)
                                cache_sign = pl.reg.broadcast(32768, lanes=64, dtype=pl.UINT32)
                                cache_word = pl.reg.broadcast(65535, lanes=64, dtype=pl.UINT32)
                                cache_invalid = pl.reg.broadcast(127, lanes=64, dtype=pl.UINT32)
                                cache_hist_initial = pl.reg.load(cached_high, lanes=256)
                                for cache_segment, (cache_hist_acc,) in pl.hardware_loop(cache_segments, init_values=(cache_hist_initial,)):
                                    cache_input = pl.reg.load(cache_values, cache_segment * 64, lanes=64)
                                    cache_bits = pl.reg.interpret_cast(cache_input, dtype=pl.UINT32)
                                    cache_bf16 = pl.reg.shift_rights(cache_bits, 16, cache_mask)
                                    cache_negative = pl.reg.compares(cache_bf16, 32768, cache_mask, predicate="ge")
                                    cache_negative_key = pl.reg.bitwise_and(pl.reg.bitwise_not(cache_bf16), cache_word)
                                    cache_positive_key = pl.reg.bitwise_xor(cache_bf16, cache_sign)
                                    cache_key32 = pl.reg.select(cache_negative, cache_negative_key, cache_positive_key)
                                    cache_positions = pl.reg.index_gen(pl.cast(cache_segment * 64, pl.INT32), lanes=64, dtype=pl.INT32)
                                    cache_active = pl.reg.compares(cache_positions, local_width, cache_mask, predicate="lt")
                                    cache_key32 = pl.reg.select(cache_active, cache_key32, cache_invalid)
                                    cache_key_halves = pl.reg.interpret_cast(cache_key32, dtype=pl.UINT16)
                                    cache_key_even, cache_key_odd = pl.reg.dintlv(cache_key_halves, cache_key_halves, cache_pack_mask)
                                    _cache_pack_write = pl.reg.store(cache_pack, value=cache_key_even)
                                    cache_key16 = pl.reg.load(cache_pack, lanes=64)
                                    _cache_key_write = pl.reg.store(cached_keys, cache_segment * 64, cache_key16)
                                    cache_high32 = pl.reg.shift_rights(cache_key32, 8, cache_mask)
                                    cache_high_i32 = pl.reg.interpret_cast(cache_high32, dtype=pl.INT32)
                                    cache_high_i8 = pl.reg.convert(cache_high_i32, dtype=pl.INT8, saturate=False)
                                    cache_high_bytes = pl.reg.interpret_cast(cache_high_i8, dtype=pl.UINT8)
                                    cache_hist_next = pl.reg.chist(cache_hist_acc, cache_high_bytes, cache_active)
                                    cache_hist_final = pl.yield_(cache_hist_next)
                                _cache_hist_write = pl.reg.store(cached_high, value=cache_hist_final)
                            local_cutoff = pl.cast(127, pl.INT32)
                            local_ties = pl.cast(INDEX_TOPK, pl.INT32)
                            if local_width > INDEX_TOPK:
                                local_rank = pl.cast(local_width - INDEX_TOPK + 1, pl.INT32)
                                local_cutoff = pl.cast(0, pl.INT32)
                                local_high_search_lower = pl.cast(0, pl.INT32)
                                local_high_search_upper = pl.cast(255, pl.INT32)
                                for local_high_search_step in pl.range(8):
                                    local_high_search_middle = pl.cast((local_high_search_lower + local_high_search_upper) // 2, pl.INT32)
                                    local_high_search_cumulative = pl.cast(
                                        pl.tile.read(cached_high, [pl.cast(local_high_search_middle, pl.INDEX)]),
                                        pl.INT32,
                                    )
                                    if local_high_search_cumulative >= local_rank:
                                        local_high_search_upper = local_high_search_middle
                                    else:
                                        local_high_search_lower = pl.cast(local_high_search_middle + 1, pl.INT32)
                                local_high_search_previous = pl.cast(0, pl.INT32)
                                if local_high_search_lower > 0:
                                    local_high_search_previous = pl.cast(
                                        pl.tile.read(cached_high, [pl.cast(local_high_search_lower - 1, pl.INDEX)]),
                                        pl.INT32,
                                    )
                                _local_high_search_bin_count = pl.cast(pl.tile.read(cached_high, [pl.cast(local_high_search_lower, pl.INDEX)]), pl.INT32) - local_high_search_previous
                                local_rank = pl.cast(local_rank - local_high_search_previous, pl.INT32)
                                local_cutoff = pl.cast(local_cutoff + local_high_search_lower * 256, pl.INT32)
                                local_low_hist = pl.tile.full([256], dtype=pl.UINT16, value=0)
                                local_low_hist = pl.tile.move(local_low_hist, target_memory=pl.MemorySpace.Vec)
                                local_low_segments = pl.cast((local_width + 63) // 64, pl.INT32)
                                local_low_high = pl.cast(local_cutoff // 256, pl.UINT32)
                                with pl.reg_scope():
                                    local_low_mask = pl.reg.full_mask(lanes=64)
                                    local_low_byte_mask = pl.reg.broadcast(255, lanes=64, dtype=pl.UINT32)
                                    local_low_hist_initial = pl.reg.load(local_low_hist, lanes=256)
                                    for local_low_segment, (local_low_hist_acc,) in pl.hardware_loop(local_low_segments, init_values=(local_low_hist_initial,)):
                                        local_low_keys16 = pl.reg.load(cached_keys, local_low_segment * 64, lanes=64)
                                        local_low_keys32 = pl.reg.convert(local_low_keys16, dtype=pl.UINT32)
                                        local_low_upper = pl.reg.shift_rights(local_low_keys32, 8, local_low_mask)
                                        local_low_lower = pl.reg.bitwise_and(local_low_keys32, local_low_byte_mask)
                                        local_low_signed = pl.reg.interpret_cast(local_low_lower, dtype=pl.INT32)
                                        local_low_bytes_i8 = pl.reg.convert(local_low_signed, dtype=pl.INT8, saturate=False)
                                        local_low_bytes = pl.reg.interpret_cast(local_low_bytes_i8, dtype=pl.UINT8)
                                        local_low_positions = pl.reg.index_gen(pl.cast(local_low_segment * 64, pl.INT32), lanes=64, dtype=pl.INT32)
                                        local_low_active = pl.reg.compares(local_low_positions, local_width, local_low_mask, predicate="lt")
                                        local_low_matching = pl.reg.compares(local_low_upper, local_low_high, local_low_mask, predicate="eq")
                                        local_low_selected = pl.reg.mask_and(local_low_active, local_low_matching)
                                        local_low_next = pl.reg.chist(local_low_hist_acc, local_low_bytes, local_low_selected)
                                        local_low_hist_final = pl.yield_(local_low_next)
                                    _local_low_hist_write = pl.reg.store(local_low_hist, value=local_low_hist_final)
                                local_low_search_lower = pl.cast(0, pl.INT32)
                                local_low_search_upper = pl.cast(255, pl.INT32)
                                for local_low_search_step in pl.range(8):
                                    local_low_search_middle = pl.cast((local_low_search_lower + local_low_search_upper) // 2, pl.INT32)
                                    local_low_search_cumulative = pl.cast(
                                        pl.tile.read(local_low_hist, [pl.cast(local_low_search_middle, pl.INDEX)]),
                                        pl.INT32,
                                    )
                                    if local_low_search_cumulative >= local_rank:
                                        local_low_search_upper = local_low_search_middle
                                    else:
                                        local_low_search_lower = pl.cast(local_low_search_middle + 1, pl.INT32)
                                local_low_search_previous = pl.cast(0, pl.INT32)
                                if local_low_search_lower > 0:
                                    local_low_search_previous = pl.cast(
                                        pl.tile.read(local_low_hist, [pl.cast(local_low_search_lower - 1, pl.INDEX)]),
                                        pl.INT32,
                                    )
                                local_low_search_bin_count = pl.cast(pl.tile.read(local_low_hist, [pl.cast(local_low_search_lower, pl.INDEX)]), pl.INT32) - local_low_search_previous
                                local_rank = pl.cast(local_rank - local_low_search_previous, pl.INT32)
                                local_cutoff = pl.cast(local_cutoff + local_low_search_lower * 1, pl.INT32)
                                local_ties = pl.cast(local_low_search_bin_count - local_rank + 1, pl.INT32)
                            mapping_raw = pl.tile.load(
                                physical, [token, logical_begin],
                                [1, SCORE_SHARD_CACHE], valid_shape=[1, pl.cast(local_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            mapping_pad = pl.fillpad(mapping_raw, pad_value=pl.PadValue.min)
                            mapping_full = pl.set_validshape(mapping_pad, 1, SCORE_SHARD_CACHE)
                            local_mapping = pl.reshape(mapping_full, [SCORE_SHARD_CACHE])
                            local_keys = pl.tile.full([INDEX_TOPK], dtype=pl.UINT16, value=0)
                            local_keys = pl.tile.move(local_keys, target_memory=pl.MemorySpace.Vec)
                            local_ids = pl.tile.full([INDEX_TOPK], dtype=pl.INT32, value=-1)
                            local_ids = pl.tile.move(local_ids, target_memory=pl.MemorySpace.Vec)
                            local_select_state = pl.tile.full([128], dtype=pl.INT32, value=0)
                            local_select_state = pl.tile.move(local_select_state, target_memory=pl.MemorySpace.Vec)
                            with pl.reg_scope():
                                local_select_initial_ties = pl.reg.broadcast(local_ties, lanes=64, dtype=pl.INT32)
                                local_select_state = pl.reg.store(local_select_state, 64, local_select_initial_ties)
                            local_select_cutoff = pl.cast(local_cutoff, pl.UINT32)
                            local_select_segments = pl.cast((local_width + 63) // 64, pl.INT32)
                            local_select_offset_pack = pl.tile.full([128], dtype=pl.UINT16, value=0)
                            local_select_offset_pack = pl.tile.move(local_select_offset_pack, target_memory=pl.MemorySpace.Vec)
                            with pl.reg_scope():
                                local_select_mask = pl.reg.full_mask(lanes=64)
                                local_select_pack_mask = pl.reg.full_mask(lanes=128)
                                local_select_positions = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                                local_select_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                                local_select_one = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                                local_select_last = pl.reg.broadcast(63, lanes=64, dtype=pl.UINT32)
                                for local_select_segment in pl.hardware_loop(local_select_segments):
                                    local_select_keys16 = pl.reg.load(cached_keys, local_select_segment * 64, lanes=64)
                                    local_select_keys = pl.reg.convert(local_select_keys16, dtype=pl.UINT32)
                                    local_select_ids = pl.reg.load(local_mapping, local_select_segment * 64, lanes=64)
                                    local_select_valid = pl.reg.compares(local_select_ids, 0, local_select_mask, predicate="ge")
                                    local_select_finite = pl.reg.compares(local_select_keys, 127, local_select_valid, predicate="gt")
                                    local_select_strict = pl.reg.compares(local_select_keys, local_select_cutoff, local_select_finite, predicate="gt")
                                    local_select_equal = pl.reg.compares(local_select_keys, local_select_cutoff, local_select_finite, predicate="eq")
                                    local_select_equal_prefix = pl.reg.select(local_select_equal, local_select_one, local_select_zero)
                                    local_select_equal_prior1 = pl.reg.adds(local_select_positions, -1, local_select_mask)
                                    local_select_equal_prior1 = pl.reg.maximum(local_select_equal_prior1, local_select_zero)
                                    local_select_equal_offset1 = pl.reg.interpret_cast(local_select_equal_prior1, dtype=pl.UINT32)
                                    local_select_equal_shifted1 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset1)
                                    local_select_equal_tail1 = pl.reg.compares(local_select_positions, 1, local_select_mask, predicate="ge")
                                    local_select_equal_shifted1 = pl.reg.select(local_select_equal_tail1, local_select_equal_shifted1, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted1)
                                    local_select_equal_prior2 = pl.reg.adds(local_select_positions, -2, local_select_mask)
                                    local_select_equal_prior2 = pl.reg.maximum(local_select_equal_prior2, local_select_zero)
                                    local_select_equal_offset2 = pl.reg.interpret_cast(local_select_equal_prior2, dtype=pl.UINT32)
                                    local_select_equal_shifted2 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset2)
                                    local_select_equal_tail2 = pl.reg.compares(local_select_positions, 2, local_select_mask, predicate="ge")
                                    local_select_equal_shifted2 = pl.reg.select(local_select_equal_tail2, local_select_equal_shifted2, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted2)
                                    local_select_equal_prior4 = pl.reg.adds(local_select_positions, -4, local_select_mask)
                                    local_select_equal_prior4 = pl.reg.maximum(local_select_equal_prior4, local_select_zero)
                                    local_select_equal_offset4 = pl.reg.interpret_cast(local_select_equal_prior4, dtype=pl.UINT32)
                                    local_select_equal_shifted4 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset4)
                                    local_select_equal_tail4 = pl.reg.compares(local_select_positions, 4, local_select_mask, predicate="ge")
                                    local_select_equal_shifted4 = pl.reg.select(local_select_equal_tail4, local_select_equal_shifted4, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted4)
                                    local_select_equal_prior8 = pl.reg.adds(local_select_positions, -8, local_select_mask)
                                    local_select_equal_prior8 = pl.reg.maximum(local_select_equal_prior8, local_select_zero)
                                    local_select_equal_offset8 = pl.reg.interpret_cast(local_select_equal_prior8, dtype=pl.UINT32)
                                    local_select_equal_shifted8 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset8)
                                    local_select_equal_tail8 = pl.reg.compares(local_select_positions, 8, local_select_mask, predicate="ge")
                                    local_select_equal_shifted8 = pl.reg.select(local_select_equal_tail8, local_select_equal_shifted8, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted8)
                                    local_select_equal_prior16 = pl.reg.adds(local_select_positions, -16, local_select_mask)
                                    local_select_equal_prior16 = pl.reg.maximum(local_select_equal_prior16, local_select_zero)
                                    local_select_equal_offset16 = pl.reg.interpret_cast(local_select_equal_prior16, dtype=pl.UINT32)
                                    local_select_equal_shifted16 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset16)
                                    local_select_equal_tail16 = pl.reg.compares(local_select_positions, 16, local_select_mask, predicate="ge")
                                    local_select_equal_shifted16 = pl.reg.select(local_select_equal_tail16, local_select_equal_shifted16, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted16)
                                    local_select_equal_prior32 = pl.reg.adds(local_select_positions, -32, local_select_mask)
                                    local_select_equal_prior32 = pl.reg.maximum(local_select_equal_prior32, local_select_zero)
                                    local_select_equal_offset32 = pl.reg.interpret_cast(local_select_equal_prior32, dtype=pl.UINT32)
                                    local_select_equal_shifted32 = pl.reg.gather_lanes(local_select_equal_prefix, local_select_equal_offset32)
                                    local_select_equal_tail32 = pl.reg.compares(local_select_positions, 32, local_select_mask, predicate="ge")
                                    local_select_equal_shifted32 = pl.reg.select(local_select_equal_tail32, local_select_equal_shifted32, local_select_zero)
                                    local_select_equal_prefix = pl.reg.add(local_select_equal_prefix, local_select_equal_shifted32)
                                    local_select_remaining = pl.reg.load(local_select_state, 64, lanes=64)
                                    local_select_quota = pl.reg.compare(local_select_equal_prefix, local_select_remaining, local_select_mask, predicate="le")
                                    local_select_tied = pl.reg.mask_and(local_select_equal, local_select_quota)
                                    local_select_keep = pl.reg.mask_or(local_select_strict, local_select_tied)
                                    local_select_keep_prefix = pl.reg.select(local_select_keep, local_select_one, local_select_zero)
                                    local_select_keep_prior1 = pl.reg.adds(local_select_positions, -1, local_select_mask)
                                    local_select_keep_prior1 = pl.reg.maximum(local_select_keep_prior1, local_select_zero)
                                    local_select_keep_offset1 = pl.reg.interpret_cast(local_select_keep_prior1, dtype=pl.UINT32)
                                    local_select_keep_shifted1 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset1)
                                    local_select_keep_tail1 = pl.reg.compares(local_select_positions, 1, local_select_mask, predicate="ge")
                                    local_select_keep_shifted1 = pl.reg.select(local_select_keep_tail1, local_select_keep_shifted1, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted1)
                                    local_select_keep_prior2 = pl.reg.adds(local_select_positions, -2, local_select_mask)
                                    local_select_keep_prior2 = pl.reg.maximum(local_select_keep_prior2, local_select_zero)
                                    local_select_keep_offset2 = pl.reg.interpret_cast(local_select_keep_prior2, dtype=pl.UINT32)
                                    local_select_keep_shifted2 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset2)
                                    local_select_keep_tail2 = pl.reg.compares(local_select_positions, 2, local_select_mask, predicate="ge")
                                    local_select_keep_shifted2 = pl.reg.select(local_select_keep_tail2, local_select_keep_shifted2, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted2)
                                    local_select_keep_prior4 = pl.reg.adds(local_select_positions, -4, local_select_mask)
                                    local_select_keep_prior4 = pl.reg.maximum(local_select_keep_prior4, local_select_zero)
                                    local_select_keep_offset4 = pl.reg.interpret_cast(local_select_keep_prior4, dtype=pl.UINT32)
                                    local_select_keep_shifted4 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset4)
                                    local_select_keep_tail4 = pl.reg.compares(local_select_positions, 4, local_select_mask, predicate="ge")
                                    local_select_keep_shifted4 = pl.reg.select(local_select_keep_tail4, local_select_keep_shifted4, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted4)
                                    local_select_keep_prior8 = pl.reg.adds(local_select_positions, -8, local_select_mask)
                                    local_select_keep_prior8 = pl.reg.maximum(local_select_keep_prior8, local_select_zero)
                                    local_select_keep_offset8 = pl.reg.interpret_cast(local_select_keep_prior8, dtype=pl.UINT32)
                                    local_select_keep_shifted8 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset8)
                                    local_select_keep_tail8 = pl.reg.compares(local_select_positions, 8, local_select_mask, predicate="ge")
                                    local_select_keep_shifted8 = pl.reg.select(local_select_keep_tail8, local_select_keep_shifted8, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted8)
                                    local_select_keep_prior16 = pl.reg.adds(local_select_positions, -16, local_select_mask)
                                    local_select_keep_prior16 = pl.reg.maximum(local_select_keep_prior16, local_select_zero)
                                    local_select_keep_offset16 = pl.reg.interpret_cast(local_select_keep_prior16, dtype=pl.UINT32)
                                    local_select_keep_shifted16 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset16)
                                    local_select_keep_tail16 = pl.reg.compares(local_select_positions, 16, local_select_mask, predicate="ge")
                                    local_select_keep_shifted16 = pl.reg.select(local_select_keep_tail16, local_select_keep_shifted16, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted16)
                                    local_select_keep_prior32 = pl.reg.adds(local_select_positions, -32, local_select_mask)
                                    local_select_keep_prior32 = pl.reg.maximum(local_select_keep_prior32, local_select_zero)
                                    local_select_keep_offset32 = pl.reg.interpret_cast(local_select_keep_prior32, dtype=pl.UINT32)
                                    local_select_keep_shifted32 = pl.reg.gather_lanes(local_select_keep_prefix, local_select_keep_offset32)
                                    local_select_keep_tail32 = pl.reg.compares(local_select_positions, 32, local_select_mask, predicate="ge")
                                    local_select_keep_shifted32 = pl.reg.select(local_select_keep_tail32, local_select_keep_shifted32, local_select_zero)
                                    local_select_keep_prefix = pl.reg.add(local_select_keep_prefix, local_select_keep_shifted32)
                                    local_select_count = pl.reg.load(local_select_state, lanes=64)
                                    local_select_destinations = pl.reg.add(local_select_keep_prefix, local_select_count)
                                    local_select_destinations = pl.reg.adds(local_select_destinations, -1, local_select_mask)
                                    local_select_addresses32 = pl.reg.interpret_cast(local_select_destinations, dtype=pl.UINT32)
                                    local_select_address_halves = pl.reg.interpret_cast(local_select_addresses32, dtype=pl.UINT16)
                                    local_select_address_even, local_select_address_odd = pl.reg.dintlv(
                                        local_select_address_halves, local_select_address_halves,
                                        local_select_pack_mask,
                                    )
                                    _local_select_pack_write = pl.reg.store(local_select_offset_pack, value=local_select_address_even)
                                    local_select_addresses16 = pl.reg.load(local_select_offset_pack, lanes=64)
                                    _local_select_ids_write = pl.reg.scatter(
                                        local_ids, local_select_addresses32,
                                        local_select_ids, local_select_keep,
                                    )
                                    _local_select_keys_write = pl.reg.scatter(
                                        local_keys, local_select_addresses16,
                                        local_select_keys16, local_select_keep,
                                    )
                                    local_select_kept_total = pl.reg.gather_lanes(local_select_keep_prefix, local_select_last)
                                    local_select_equal_total = pl.reg.gather_lanes(local_select_equal_prefix, local_select_last)
                                    local_select_next_count = pl.reg.add(local_select_count, local_select_kept_total)
                                    local_select_next_ties = pl.reg.sub(local_select_remaining, local_select_equal_total)
                                    local_select_next_ties = pl.reg.maximum(local_select_next_ties, local_select_zero)
                                    _local_select_count_write = pl.reg.store(local_select_state, value=local_select_next_count)
                                    _local_select_ties_write = pl.reg.store(local_select_state, 64, local_select_next_ties)
                            local_keys_row = pl.reshape(local_keys, [1, INDEX_TOPK])
                            local_ids_row = pl.reshape(local_ids, [1, INDEX_TOPK])
                            candidate_keys = pl.store(local_keys_row, [token, shard * INDEX_TOPK], candidate_keys)
                            candidate_ids = pl.store(local_ids_row, [token, shard * INDEX_TOPK], candidate_ids)
    with pl.spmd(workers, name_hint="idx_topk_merge_reg", deps=[local_done]) as completed:
        merge_worker = pl.tile.get_block_idx()
        for merge_token in pl.range(merge_worker, num_tokens, workers):
            final_ids = pl.tile.full([INDEX_TOPK], dtype=pl.INT32, value=-1)
            final_ids = pl.tile.move(final_ids, target_memory=pl.MemorySpace.Vec)
            short = pl.cast(0, pl.INT32)
            if direct_short:
                if pl.read(lengths, [merge_token]) <= INDEX_TOPK:
                    short = pl.cast(1, pl.INT32)
            if short == 1:
                direct_active = pl.min(INDEX_TOPK, width)
                direct_raw = pl.tile.load(
                    physical, [merge_token, 0],
                    [1, INDEX_TOPK], valid_shape=[1, direct_active],
                    target_memory=pl.MemorySpace.Vec,
                )
                direct_pad = pl.fillpad(direct_raw, pad_value=pl.PadValue.min)
                direct_full = pl.set_validshape(direct_pad, 1, INDEX_TOPK)
                direct_ids = pl.reshape(direct_full, [INDEX_TOPK])
                with pl.reg_scope():
                    direct_mask = pl.reg.full_mask(lanes=64)
                    direct_holes = pl.reg.broadcast(-1, lanes=64, dtype=pl.INT32)
                    for direct_segment in pl.hardware_loop(INDEX_TOPK // 64):
                        direct_values = pl.reg.load(direct_ids, direct_segment * 64, lanes=64)
                        direct_valid = pl.reg.compares(direct_values, 0, direct_mask, predicate="ge")
                        direct_values = pl.reg.select(direct_valid, direct_values, direct_holes)
                        _direct_write = pl.reg.store(final_ids, direct_segment * 64, direct_values)
            else:
                if total > 0:
                    group = pl.cast(
                        pl.read(runs, [3, merge_token + 1]),
                        pl.INT32,
                    )
                    first_shard = pl.cast(((group * panels + 1) * owners - 1) // total, pl.INT32)
                    last_shard = pl.cast((((group + 1) * panels) * owners - 1) // total, pl.INT32)
                    shard_count = last_shard - first_shard + 1
                    merge_batches = 1 + (pl.max(shard_count - 32, 0) + 30) // 31
                    retained_keys = pl.tile.full([INDEX_TOPK], dtype=pl.UINT16, value=0)
                    retained_keys = pl.tile.move(retained_keys, target_memory=pl.MemorySpace.Vec)
                    retained_ids = pl.tile.full([INDEX_TOPK], dtype=pl.INT32, value=-1)
                    retained_ids = pl.tile.move(retained_ids, target_memory=pl.MemorySpace.Vec)
                    for merge_batch in pl.range(merge_batches):
                        merge_keys = pl.tile.full([SCORE_MERGE_CACHE], dtype=pl.UINT16, value=0)
                        merge_keys = pl.tile.move(merge_keys, target_memory=pl.MemorySpace.Vec)
                        merge_ids = pl.tile.full([SCORE_MERGE_CACHE], dtype=pl.INT32, value=-1)
                        merge_ids = pl.tile.move(merge_ids, target_memory=pl.MemorySpace.Vec)
                        if merge_batch == 0:
                            merge_cursor = pl.cast(first_shard, pl.INT32)
                            merge_take = pl.min(32, shard_count)
                            merge_width = pl.cast(merge_take * INDEX_TOPK, pl.INT32)
                            merge_keys_raw = pl.tile.load(
                                candidate_keys, [merge_token, merge_cursor * INDEX_TOPK],
                                [1, SCORE_MERGE_CACHE], valid_shape=[1, pl.cast(merge_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            merge_ids_raw = pl.tile.load(
                                candidate_ids, [merge_token, merge_cursor * INDEX_TOPK],
                                [1, SCORE_MERGE_CACHE], valid_shape=[1, pl.cast(merge_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            merge_keys_pad = pl.fillpad(merge_keys_raw, pad_value=pl.PadValue.min)
                            merge_ids_pad = pl.fillpad(merge_ids_raw, pad_value=pl.PadValue.min)
                            merge_keys_full = pl.set_validshape(merge_keys_pad, 1, SCORE_MERGE_CACHE)
                            merge_ids_full = pl.set_validshape(merge_ids_pad, 1, SCORE_MERGE_CACHE)
                            merge_first_keys = pl.reshape(merge_keys_full, [SCORE_MERGE_CACHE])
                            merge_first_ids = pl.reshape(merge_ids_full, [SCORE_MERGE_CACHE])
                            merge_first_segments = pl.cast(merge_width // 64, pl.INT32)
                            with pl.reg_scope():
                                for first_copy in pl.hardware_loop(merge_first_segments):
                                    first_keys = pl.reg.load(merge_first_keys, first_copy * 64, lanes=64)
                                    first_ids = pl.reg.load(merge_first_ids, first_copy * 64, lanes=64)
                                    _first_key_write = pl.reg.store(merge_keys, first_copy * 64, first_keys)
                                    _first_id_write = pl.reg.store(merge_ids, first_copy * 64, first_ids)
                        else:
                            merge_cursor = pl.cast(first_shard + 32 + (merge_batch - 1) * 31, pl.INT32)
                            merge_take = pl.min(31, last_shard - merge_cursor + 1)
                            merge_load_width = pl.cast(merge_take * INDEX_TOPK, pl.INT32)
                            merge_width = pl.cast(INDEX_TOPK + merge_load_width, pl.INT32)
                            merge_next_keys_raw = pl.tile.load(
                                candidate_keys, [merge_token, merge_cursor * INDEX_TOPK],
                                [1, SCORE_MERGE_CACHE - INDEX_TOPK], valid_shape=[1, pl.cast(merge_load_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            merge_next_ids_raw = pl.tile.load(
                                candidate_ids, [merge_token, merge_cursor * INDEX_TOPK],
                                [1, SCORE_MERGE_CACHE - INDEX_TOPK], valid_shape=[1, pl.cast(merge_load_width, pl.INDEX)],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            merge_next_keys_pad = pl.fillpad(merge_next_keys_raw, pad_value=pl.PadValue.min)
                            merge_next_ids_pad = pl.fillpad(merge_next_ids_raw, pad_value=pl.PadValue.min)
                            merge_next_keys_full = pl.set_validshape(merge_next_keys_pad, 1, SCORE_MERGE_CACHE - INDEX_TOPK)
                            merge_next_ids_full = pl.set_validshape(merge_next_ids_pad, 1, SCORE_MERGE_CACHE - INDEX_TOPK)
                            merge_next_keys = pl.reshape(merge_next_keys_full, [SCORE_MERGE_CACHE - INDEX_TOPK])
                            merge_next_ids = pl.reshape(merge_next_ids_full, [SCORE_MERGE_CACHE - INDEX_TOPK])
                            merge_next_segments = pl.cast(merge_load_width // 64, pl.INT32)
                            with pl.reg_scope():
                                for carry_copy in pl.hardware_loop(INDEX_TOPK // 64):
                                    carry_keys = pl.reg.load(retained_keys, carry_copy * 64, lanes=64)
                                    carry_ids = pl.reg.load(retained_ids, carry_copy * 64, lanes=64)
                                    _carry_key_write = pl.reg.store(merge_keys, carry_copy * 64, carry_keys)
                                    _carry_id_write = pl.reg.store(merge_ids, carry_copy * 64, carry_ids)
                                for next_copy in pl.hardware_loop(merge_next_segments):
                                    next_keys = pl.reg.load(merge_next_keys, next_copy * 64, lanes=64)
                                    next_ids = pl.reg.load(merge_next_ids, next_copy * 64, lanes=64)
                                    _next_key_write = pl.reg.store(merge_keys, INDEX_TOPK + next_copy * 64, next_keys)
                                    _next_id_write = pl.reg.store(merge_ids, INDEX_TOPK + next_copy * 64, next_ids)
                        merge_high_hist = pl.tile.full([256], dtype=pl.UINT16, value=0)
                        merge_high_hist = pl.tile.move(merge_high_hist, target_memory=pl.MemorySpace.Vec)
                        merge_segments = pl.cast((merge_width + 63) // 64, pl.INT32)
                        with pl.reg_scope():
                            merge_hist_mask = pl.reg.full_mask(lanes=64)
                            merge_hist_initial = pl.reg.load(merge_high_hist, lanes=256)
                            for merge_hist_segment, (merge_hist_acc,) in pl.hardware_loop(merge_segments, init_values=(merge_hist_initial,)):
                                merge_hist_keys16 = pl.reg.load(merge_keys, merge_hist_segment * 64, lanes=64)
                                merge_hist_keys = pl.reg.convert(merge_hist_keys16, dtype=pl.UINT32)
                                merge_hist_high = pl.reg.shift_rights(merge_hist_keys, 8, merge_hist_mask)
                                merge_hist_signed = pl.reg.interpret_cast(merge_hist_high, dtype=pl.INT32)
                                merge_hist_i8 = pl.reg.convert(merge_hist_signed, dtype=pl.INT8, saturate=False)
                                merge_hist_bytes = pl.reg.interpret_cast(merge_hist_i8, dtype=pl.UINT8)
                                merge_hist_positions = pl.reg.index_gen(pl.cast(merge_hist_segment * 64, pl.INT32), lanes=64, dtype=pl.INT32)
                                merge_hist_active = pl.reg.compares(merge_hist_positions, merge_width, merge_hist_mask, predicate="lt")
                                merge_hist_next = pl.reg.chist(merge_hist_acc, merge_hist_bytes, merge_hist_active)
                                merge_hist_final = pl.yield_(merge_hist_next)
                            _merge_hist_write = pl.reg.store(merge_high_hist, value=merge_hist_final)
                        merge_cutoff = pl.cast(127, pl.INT32)
                        merge_ties = pl.cast(INDEX_TOPK, pl.INT32)
                        if merge_width > INDEX_TOPK:
                            merge_rank = pl.cast(merge_width - INDEX_TOPK + 1, pl.INT32)
                            merge_cutoff = pl.cast(0, pl.INT32)
                            merge_high_search_lower = pl.cast(0, pl.INT32)
                            merge_high_search_upper = pl.cast(255, pl.INT32)
                            for merge_high_search_step in pl.range(8):
                                merge_high_search_middle = pl.cast((merge_high_search_lower + merge_high_search_upper) // 2, pl.INT32)
                                merge_high_search_cumulative = pl.cast(
                                    pl.tile.read(merge_high_hist, [pl.cast(merge_high_search_middle, pl.INDEX)]),
                                    pl.INT32,
                                )
                                if merge_high_search_cumulative >= merge_rank:
                                    merge_high_search_upper = merge_high_search_middle
                                else:
                                    merge_high_search_lower = pl.cast(merge_high_search_middle + 1, pl.INT32)
                            merge_high_search_previous = pl.cast(0, pl.INT32)
                            if merge_high_search_lower > 0:
                                merge_high_search_previous = pl.cast(
                                    pl.tile.read(merge_high_hist, [pl.cast(merge_high_search_lower - 1, pl.INDEX)]),
                                    pl.INT32,
                                )
                            _merge_high_search_bin_count = pl.cast(pl.tile.read(merge_high_hist, [pl.cast(merge_high_search_lower, pl.INDEX)]), pl.INT32) - merge_high_search_previous
                            merge_rank = pl.cast(merge_rank - merge_high_search_previous, pl.INT32)
                            merge_cutoff = pl.cast(merge_cutoff + merge_high_search_lower * 256, pl.INT32)
                            merge_low_hist = pl.tile.full([256], dtype=pl.UINT16, value=0)
                            merge_low_hist = pl.tile.move(merge_low_hist, target_memory=pl.MemorySpace.Vec)
                            merge_low_segments = pl.cast((merge_width + 63) // 64, pl.INT32)
                            merge_low_high = pl.cast(merge_cutoff // 256, pl.UINT32)
                            with pl.reg_scope():
                                merge_low_mask = pl.reg.full_mask(lanes=64)
                                merge_low_byte_mask = pl.reg.broadcast(255, lanes=64, dtype=pl.UINT32)
                                merge_low_hist_initial = pl.reg.load(merge_low_hist, lanes=256)
                                for merge_low_segment, (merge_low_hist_acc,) in pl.hardware_loop(merge_low_segments, init_values=(merge_low_hist_initial,)):
                                    merge_low_keys16 = pl.reg.load(merge_keys, merge_low_segment * 64, lanes=64)
                                    merge_low_keys32 = pl.reg.convert(merge_low_keys16, dtype=pl.UINT32)
                                    merge_low_upper = pl.reg.shift_rights(merge_low_keys32, 8, merge_low_mask)
                                    merge_low_lower = pl.reg.bitwise_and(merge_low_keys32, merge_low_byte_mask)
                                    merge_low_signed = pl.reg.interpret_cast(merge_low_lower, dtype=pl.INT32)
                                    merge_low_bytes_i8 = pl.reg.convert(merge_low_signed, dtype=pl.INT8, saturate=False)
                                    merge_low_bytes = pl.reg.interpret_cast(merge_low_bytes_i8, dtype=pl.UINT8)
                                    merge_low_positions = pl.reg.index_gen(pl.cast(merge_low_segment * 64, pl.INT32), lanes=64, dtype=pl.INT32)
                                    merge_low_active = pl.reg.compares(merge_low_positions, merge_width, merge_low_mask, predicate="lt")
                                    merge_low_matching = pl.reg.compares(merge_low_upper, merge_low_high, merge_low_mask, predicate="eq")
                                    merge_low_selected = pl.reg.mask_and(merge_low_active, merge_low_matching)
                                    merge_low_next = pl.reg.chist(merge_low_hist_acc, merge_low_bytes, merge_low_selected)
                                    merge_low_hist_final = pl.yield_(merge_low_next)
                                _merge_low_hist_write = pl.reg.store(merge_low_hist, value=merge_low_hist_final)
                            merge_low_search_lower = pl.cast(0, pl.INT32)
                            merge_low_search_upper = pl.cast(255, pl.INT32)
                            for merge_low_search_step in pl.range(8):
                                merge_low_search_middle = pl.cast((merge_low_search_lower + merge_low_search_upper) // 2, pl.INT32)
                                merge_low_search_cumulative = pl.cast(
                                    pl.tile.read(merge_low_hist, [pl.cast(merge_low_search_middle, pl.INDEX)]),
                                    pl.INT32,
                                )
                                if merge_low_search_cumulative >= merge_rank:
                                    merge_low_search_upper = merge_low_search_middle
                                else:
                                    merge_low_search_lower = pl.cast(merge_low_search_middle + 1, pl.INT32)
                            merge_low_search_previous = pl.cast(0, pl.INT32)
                            if merge_low_search_lower > 0:
                                merge_low_search_previous = pl.cast(
                                    pl.tile.read(merge_low_hist, [pl.cast(merge_low_search_lower - 1, pl.INDEX)]),
                                    pl.INT32,
                                )
                            merge_low_search_bin_count = pl.cast(pl.tile.read(merge_low_hist, [pl.cast(merge_low_search_lower, pl.INDEX)]), pl.INT32) - merge_low_search_previous
                            merge_rank = pl.cast(merge_rank - merge_low_search_previous, pl.INT32)
                            merge_cutoff = pl.cast(merge_cutoff + merge_low_search_lower * 1, pl.INT32)
                            merge_ties = pl.cast(merge_low_search_bin_count - merge_rank + 1, pl.INT32)
                        selected_merge_keys = pl.tile.full([INDEX_TOPK], dtype=pl.UINT16, value=0)
                        selected_merge_keys = pl.tile.move(selected_merge_keys, target_memory=pl.MemorySpace.Vec)
                        selected_merge_ids = pl.tile.full([INDEX_TOPK], dtype=pl.INT32, value=-1)
                        selected_merge_ids = pl.tile.move(selected_merge_ids, target_memory=pl.MemorySpace.Vec)
                        merge_select_state = pl.tile.full([128], dtype=pl.INT32, value=0)
                        merge_select_state = pl.tile.move(merge_select_state, target_memory=pl.MemorySpace.Vec)
                        with pl.reg_scope():
                            merge_select_initial_ties = pl.reg.broadcast(merge_ties, lanes=64, dtype=pl.INT32)
                            merge_select_state = pl.reg.store(merge_select_state, 64, merge_select_initial_ties)
                        merge_select_cutoff = pl.cast(merge_cutoff, pl.UINT32)
                        merge_select_segments = pl.cast((merge_width + 63) // 64, pl.INT32)
                        merge_select_offset_pack = pl.tile.full([128], dtype=pl.UINT16, value=0)
                        merge_select_offset_pack = pl.tile.move(merge_select_offset_pack, target_memory=pl.MemorySpace.Vec)
                        with pl.reg_scope():
                            merge_select_mask = pl.reg.full_mask(lanes=64)
                            merge_select_pack_mask = pl.reg.full_mask(lanes=128)
                            merge_select_positions = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                            merge_select_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                            merge_select_one = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                            merge_select_last = pl.reg.broadcast(63, lanes=64, dtype=pl.UINT32)
                            for merge_select_segment in pl.hardware_loop(merge_select_segments):
                                merge_select_keys16 = pl.reg.load(merge_keys, merge_select_segment * 64, lanes=64)
                                merge_select_keys = pl.reg.convert(merge_select_keys16, dtype=pl.UINT32)
                                merge_select_ids = pl.reg.load(merge_ids, merge_select_segment * 64, lanes=64)
                                merge_select_valid = pl.reg.compares(merge_select_ids, 0, merge_select_mask, predicate="ge")
                                merge_select_finite = pl.reg.compares(merge_select_keys, 127, merge_select_valid, predicate="gt")
                                merge_select_strict = pl.reg.compares(merge_select_keys, merge_select_cutoff, merge_select_finite, predicate="gt")
                                merge_select_equal = pl.reg.compares(merge_select_keys, merge_select_cutoff, merge_select_finite, predicate="eq")
                                merge_select_equal_prefix = pl.reg.select(merge_select_equal, merge_select_one, merge_select_zero)
                                merge_select_equal_prior1 = pl.reg.adds(merge_select_positions, -1, merge_select_mask)
                                merge_select_equal_prior1 = pl.reg.maximum(merge_select_equal_prior1, merge_select_zero)
                                merge_select_equal_offset1 = pl.reg.interpret_cast(merge_select_equal_prior1, dtype=pl.UINT32)
                                merge_select_equal_shifted1 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset1)
                                merge_select_equal_tail1 = pl.reg.compares(merge_select_positions, 1, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted1 = pl.reg.select(merge_select_equal_tail1, merge_select_equal_shifted1, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted1)
                                merge_select_equal_prior2 = pl.reg.adds(merge_select_positions, -2, merge_select_mask)
                                merge_select_equal_prior2 = pl.reg.maximum(merge_select_equal_prior2, merge_select_zero)
                                merge_select_equal_offset2 = pl.reg.interpret_cast(merge_select_equal_prior2, dtype=pl.UINT32)
                                merge_select_equal_shifted2 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset2)
                                merge_select_equal_tail2 = pl.reg.compares(merge_select_positions, 2, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted2 = pl.reg.select(merge_select_equal_tail2, merge_select_equal_shifted2, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted2)
                                merge_select_equal_prior4 = pl.reg.adds(merge_select_positions, -4, merge_select_mask)
                                merge_select_equal_prior4 = pl.reg.maximum(merge_select_equal_prior4, merge_select_zero)
                                merge_select_equal_offset4 = pl.reg.interpret_cast(merge_select_equal_prior4, dtype=pl.UINT32)
                                merge_select_equal_shifted4 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset4)
                                merge_select_equal_tail4 = pl.reg.compares(merge_select_positions, 4, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted4 = pl.reg.select(merge_select_equal_tail4, merge_select_equal_shifted4, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted4)
                                merge_select_equal_prior8 = pl.reg.adds(merge_select_positions, -8, merge_select_mask)
                                merge_select_equal_prior8 = pl.reg.maximum(merge_select_equal_prior8, merge_select_zero)
                                merge_select_equal_offset8 = pl.reg.interpret_cast(merge_select_equal_prior8, dtype=pl.UINT32)
                                merge_select_equal_shifted8 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset8)
                                merge_select_equal_tail8 = pl.reg.compares(merge_select_positions, 8, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted8 = pl.reg.select(merge_select_equal_tail8, merge_select_equal_shifted8, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted8)
                                merge_select_equal_prior16 = pl.reg.adds(merge_select_positions, -16, merge_select_mask)
                                merge_select_equal_prior16 = pl.reg.maximum(merge_select_equal_prior16, merge_select_zero)
                                merge_select_equal_offset16 = pl.reg.interpret_cast(merge_select_equal_prior16, dtype=pl.UINT32)
                                merge_select_equal_shifted16 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset16)
                                merge_select_equal_tail16 = pl.reg.compares(merge_select_positions, 16, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted16 = pl.reg.select(merge_select_equal_tail16, merge_select_equal_shifted16, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted16)
                                merge_select_equal_prior32 = pl.reg.adds(merge_select_positions, -32, merge_select_mask)
                                merge_select_equal_prior32 = pl.reg.maximum(merge_select_equal_prior32, merge_select_zero)
                                merge_select_equal_offset32 = pl.reg.interpret_cast(merge_select_equal_prior32, dtype=pl.UINT32)
                                merge_select_equal_shifted32 = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_equal_offset32)
                                merge_select_equal_tail32 = pl.reg.compares(merge_select_positions, 32, merge_select_mask, predicate="ge")
                                merge_select_equal_shifted32 = pl.reg.select(merge_select_equal_tail32, merge_select_equal_shifted32, merge_select_zero)
                                merge_select_equal_prefix = pl.reg.add(merge_select_equal_prefix, merge_select_equal_shifted32)
                                merge_select_remaining = pl.reg.load(merge_select_state, 64, lanes=64)
                                merge_select_quota = pl.reg.compare(merge_select_equal_prefix, merge_select_remaining, merge_select_mask, predicate="le")
                                merge_select_tied = pl.reg.mask_and(merge_select_equal, merge_select_quota)
                                merge_select_keep = pl.reg.mask_or(merge_select_strict, merge_select_tied)
                                merge_select_keep_prefix = pl.reg.select(merge_select_keep, merge_select_one, merge_select_zero)
                                merge_select_keep_prior1 = pl.reg.adds(merge_select_positions, -1, merge_select_mask)
                                merge_select_keep_prior1 = pl.reg.maximum(merge_select_keep_prior1, merge_select_zero)
                                merge_select_keep_offset1 = pl.reg.interpret_cast(merge_select_keep_prior1, dtype=pl.UINT32)
                                merge_select_keep_shifted1 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset1)
                                merge_select_keep_tail1 = pl.reg.compares(merge_select_positions, 1, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted1 = pl.reg.select(merge_select_keep_tail1, merge_select_keep_shifted1, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted1)
                                merge_select_keep_prior2 = pl.reg.adds(merge_select_positions, -2, merge_select_mask)
                                merge_select_keep_prior2 = pl.reg.maximum(merge_select_keep_prior2, merge_select_zero)
                                merge_select_keep_offset2 = pl.reg.interpret_cast(merge_select_keep_prior2, dtype=pl.UINT32)
                                merge_select_keep_shifted2 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset2)
                                merge_select_keep_tail2 = pl.reg.compares(merge_select_positions, 2, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted2 = pl.reg.select(merge_select_keep_tail2, merge_select_keep_shifted2, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted2)
                                merge_select_keep_prior4 = pl.reg.adds(merge_select_positions, -4, merge_select_mask)
                                merge_select_keep_prior4 = pl.reg.maximum(merge_select_keep_prior4, merge_select_zero)
                                merge_select_keep_offset4 = pl.reg.interpret_cast(merge_select_keep_prior4, dtype=pl.UINT32)
                                merge_select_keep_shifted4 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset4)
                                merge_select_keep_tail4 = pl.reg.compares(merge_select_positions, 4, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted4 = pl.reg.select(merge_select_keep_tail4, merge_select_keep_shifted4, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted4)
                                merge_select_keep_prior8 = pl.reg.adds(merge_select_positions, -8, merge_select_mask)
                                merge_select_keep_prior8 = pl.reg.maximum(merge_select_keep_prior8, merge_select_zero)
                                merge_select_keep_offset8 = pl.reg.interpret_cast(merge_select_keep_prior8, dtype=pl.UINT32)
                                merge_select_keep_shifted8 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset8)
                                merge_select_keep_tail8 = pl.reg.compares(merge_select_positions, 8, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted8 = pl.reg.select(merge_select_keep_tail8, merge_select_keep_shifted8, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted8)
                                merge_select_keep_prior16 = pl.reg.adds(merge_select_positions, -16, merge_select_mask)
                                merge_select_keep_prior16 = pl.reg.maximum(merge_select_keep_prior16, merge_select_zero)
                                merge_select_keep_offset16 = pl.reg.interpret_cast(merge_select_keep_prior16, dtype=pl.UINT32)
                                merge_select_keep_shifted16 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset16)
                                merge_select_keep_tail16 = pl.reg.compares(merge_select_positions, 16, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted16 = pl.reg.select(merge_select_keep_tail16, merge_select_keep_shifted16, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted16)
                                merge_select_keep_prior32 = pl.reg.adds(merge_select_positions, -32, merge_select_mask)
                                merge_select_keep_prior32 = pl.reg.maximum(merge_select_keep_prior32, merge_select_zero)
                                merge_select_keep_offset32 = pl.reg.interpret_cast(merge_select_keep_prior32, dtype=pl.UINT32)
                                merge_select_keep_shifted32 = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_keep_offset32)
                                merge_select_keep_tail32 = pl.reg.compares(merge_select_positions, 32, merge_select_mask, predicate="ge")
                                merge_select_keep_shifted32 = pl.reg.select(merge_select_keep_tail32, merge_select_keep_shifted32, merge_select_zero)
                                merge_select_keep_prefix = pl.reg.add(merge_select_keep_prefix, merge_select_keep_shifted32)
                                merge_select_count = pl.reg.load(merge_select_state, lanes=64)
                                merge_select_destinations = pl.reg.add(merge_select_keep_prefix, merge_select_count)
                                merge_select_destinations = pl.reg.adds(merge_select_destinations, -1, merge_select_mask)
                                merge_select_addresses32 = pl.reg.interpret_cast(merge_select_destinations, dtype=pl.UINT32)
                                merge_select_address_halves = pl.reg.interpret_cast(merge_select_addresses32, dtype=pl.UINT16)
                                merge_select_address_even, merge_select_address_odd = pl.reg.dintlv(
                                    merge_select_address_halves, merge_select_address_halves,
                                    merge_select_pack_mask,
                                )
                                _merge_select_pack_write = pl.reg.store(merge_select_offset_pack, value=merge_select_address_even)
                                merge_select_addresses16 = pl.reg.load(merge_select_offset_pack, lanes=64)
                                _merge_select_ids_write = pl.reg.scatter(
                                    selected_merge_ids, merge_select_addresses32,
                                    merge_select_ids, merge_select_keep,
                                )
                                _merge_select_keys_write = pl.reg.scatter(
                                    selected_merge_keys, merge_select_addresses16,
                                    merge_select_keys16, merge_select_keep,
                                )
                                merge_select_kept_total = pl.reg.gather_lanes(merge_select_keep_prefix, merge_select_last)
                                merge_select_equal_total = pl.reg.gather_lanes(merge_select_equal_prefix, merge_select_last)
                                merge_select_next_count = pl.reg.add(merge_select_count, merge_select_kept_total)
                                merge_select_next_ties = pl.reg.sub(merge_select_remaining, merge_select_equal_total)
                                merge_select_next_ties = pl.reg.maximum(merge_select_next_ties, merge_select_zero)
                                _merge_select_count_write = pl.reg.store(merge_select_state, value=merge_select_next_count)
                                _merge_select_ties_write = pl.reg.store(merge_select_state, 64, merge_select_next_ties)
                        retained_keys = selected_merge_keys
                        retained_ids = selected_merge_ids
                    final_ids = retained_ids
            final_row = pl.reshape(final_ids, [1, INDEX_TOPK])
            output = pl.store(final_row, [merge_token, 0], output)
    return completed

@pl.jit.inline(auto_scope=False)
def score_select_paged_pypto(
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
    c2a: pl.constexpr,
):
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    physical = pl.create_tensor([tokens, width], dtype=pl.INT32)
    run_stride = tokens + 1
    runs = pl.create_tensor([4, run_stride], dtype=pl.INT32)
    runs_ready = _plan_score_runs(request_ids, lengths, request_ids, runs, num_tokens, width, pl.cast(0, pl.INDEX), cache_ready, False)
    completed = _score_index_rows(
        query, query_scale, weights,
        index_cache, index_cache_scale,
        block_table, request_ids, lengths,
        block_table, request_ids,
        scores, physical, runs, runs_ready,
        num_tokens, cache_ready, query_ready, weights_ready,
        cache_ready, False,
    )
    panels = (width + SCORE_TILE - 1) // SCORE_TILE
    run_count = pl.cast(
        pl.read(runs, [3, 0]),
        pl.INDEX,
    )
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    work_count = run_count * panels
    shards = pl.cast(
        pl.max(workers, (work_count + SCORE_SHARD_PANELS_TILE - 1) // SCORE_SHARD_PANELS_TILE),
        pl.INT32,
    )
    selected = _select_index_topk_sharded(scores, physical, lengths, runs, output, num_tokens, shards, completed, c2a)
    return selected, completed

@pl.jit.inline(auto_scope=False)
def score_select_reindex_pypto(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor,
    pool_block_counts: pl.Tensor,
    scores: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(scores, 0)
    width = pl.tensor.dim(scores, 1)
    physical = pl.create_tensor([tokens, width], dtype=pl.INT32)
    run_stride = tokens + 1
    runs = pl.create_tensor([4, run_stride], dtype=pl.INT32)
    capacity = pl.tensor.dim(pool_block_ids, 1)
    runs_ready = _plan_score_runs(request_ids, lengths, pool_block_counts, runs, num_tokens, width, capacity, pool_ready, True)
    completed = _score_index_rows(
        query, query_scale, weights,
        index_cache, index_cache_scale,
        block_table, request_ids, lengths,
        pool_block_ids, pool_block_counts,
        scores, physical, runs, runs_ready,
        num_tokens, cache_ready, query_ready, weights_ready,
        pool_ready, True,
    )
    panels = (width + SCORE_TILE - 1) // SCORE_TILE
    run_count = pl.cast(
        pl.read(runs, [3, 0]),
        pl.INDEX,
    )
    workers = pl.max(1, pl.min(32, pl.system.available_cluster_count()))
    work_count = run_count * panels
    shards = pl.cast(
        pl.max(workers, (work_count + SCORE_SHARD_PANELS_TILE - 1) // SCORE_SHARD_PANELS_TILE),
        pl.INT32,
    )
    return _select_index_topk_sharded(scores, physical, lengths, runs, output, num_tokens, shards, completed, False)


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor,
    weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(source, 0)
    partials = pl.create_tensor([INDEX_WEIGHT_K_SPLIT_TILE * tokens, INDEX_H], dtype=pl.FP32)
    blocks = (num_tokens + 79) // 80
    with pl.spmd(INDEX_WEIGHT_K_SPLIT_TILE, name_hint="proj_idx_w", deps=[ready]) as cube_tid:
        split = pl.tile.get_block_idx()
        begin = split * (D // INDEX_WEIGHT_K_SPLIT_TILE)
        for block in pl.range(blocks):
            row = block * 80
            rows = pl.min(80, num_tokens - row)
            a = pl.load(source, [row, begin], [80, INDEX_WEIGHT_K_TILE], valid_shape=[rows, INDEX_WEIGHT_K_TILE], target_memory=pl.MemorySpace.Mat)
            w = pl.load(weight, [begin, 0], [INDEX_WEIGHT_K_TILE, INDEX_H], target_memory=pl.MemorySpace.Mat)
            full_a = pl.set_validshape(a, 80, INDEX_WEIGHT_K_TILE)
            acc = pl.matmul(full_a, w)
            k_begin = begin + INDEX_WEIGHT_K_TILE
            k_end = begin + D // INDEX_WEIGHT_K_SPLIT_TILE
            for k in pl.pipeline(k_begin, k_end, INDEX_WEIGHT_K_TILE, stage=2):
                a = pl.load(source, [row, k], [80, INDEX_WEIGHT_K_TILE], valid_shape=[rows, INDEX_WEIGHT_K_TILE], target_memory=pl.MemorySpace.Mat)
                w = pl.load(weight, [k, 0], [INDEX_WEIGHT_K_TILE, INDEX_H], target_memory=pl.MemorySpace.Mat)
                full_a = pl.set_validshape(a, 80, INDEX_WEIGHT_K_TILE)
                acc = pl.matmul_acc(acc, full_a, w)
            value = pl.set_validshape(acc, rows, INDEX_H)
            partials = pl.store(value, [split * tokens + row, 0], partials)
    with pl.spmd(pl.max(pl.min((num_tokens + 15) // 16, 8), 1), name_hint="reduce_idx_w", deps=[cube_tid]) as completed:
        worker = pl.tile.get_block_idx()
        workers = pl.tile.get_block_num()
        for row in pl.range(worker * 16, num_tokens, workers * 16):
            rows = pl.min(16, num_tokens - row)
            reduced = pl.load(partials, [row, 0], [16, INDEX_H], valid_shape=[rows, INDEX_H])
            for split in pl.range(1, INDEX_WEIGHT_K_SPLIT_TILE):
                partial_value = pl.load(partials, [split * tokens + row, 0], [16, INDEX_H], valid_shape=[rows, INDEX_H])
                reduced = pl.add(reduced, partial_value)
            rounded = pl.cast(reduced, pl.BF16, mode="rint")
            output = pl.store(rounded, [row, 0], output)
    return completed


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    projection_payload: pl.Tensor,
    projection_scale_backing: pl.Tensor,
):
    tokens = pl.tensor.dim(x, 0)
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    query_scale = pl.create_tensor([tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0)
    with pl.spmd(16, name_hint="idx_q", deps=[entry_ready]) as query_tid:
        query = _idx_q(
            projection_payload, index_wq_b, projection_scale_backing, index_wq_b_scale,
            cos, sin, query, query_scale, num_tokens,
        )
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(
        x, index_weights_proj, weights, num_tokens, entry_ready
    )
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, query_scale, weights, scores, query_tid, weights_tid


@pl.jit.inline(auto_scope=False)
def project_index_key(
    source: pl.Tensor,
    weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    max_workers: pl.constexpr,
):
    result, completed = gemm_bf16_to_bf16(
        source, weight, output, num_tokens, ready,
        width=HEAD_DIM, output_width=INDEX_DIM, n_tile=INDEX_PROJ_N_TILE, k_tile=INDEX_PROJ_K_TILE,
        workers=max_workers, name_hint='proj_idx_k',
    )
    return completed


@pl.jit.inline(auto_scope=False)
def decode_c2a_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    projection_payload: pl.Tensor,
    projection_scales: pl.Tensor,
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // 2)], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // INDEX_CACHE_GROUP)], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish C2A keys and select long histories through the shared MX scorer."""
    tokens = pl.tensor.dim(latent, 0)
    projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projected_tid = project_index_key(latent, index_wk, projected_key, num_tokens, latent_ready, max_workers=32)
    index_publish_tid = publish_index_rows(
        projected_key, index_norm_weight, compressed_rope_cos, compressed_rope_sin, compressed_slots,
        index_cache, index_cache_scale, num_tokens, cache_ready, projected_tid,
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
    selected_tid, scores_tid = score_select_paged(
        query, query_scale, raw_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, scores, topk_indices,
        num_tokens, index_publish_tid, query_tid, weights_tid, score_width // CANDIDATE_BLOCK_SIZE, True,
    )
    return selected_tid


import torch
import torch.nn.functional as F
from qkv_proj_rope import golden_rope_tail, golden_rope_interleave
from quant import dequantize_mxfp4_cache, quantize_mxfp4_cache
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential
from rmsnorm import golden_rms_norm


def mapped_fp4_cache_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    pool_name: str,
    leading_rank_axis: bool = False,
    max_error_ratio: float = 0.005,
):
    """Bound sparse E2M1 changes after floating-point model computation.

    The existing mapped comparator keeps every unwritten byte exact and bounds
    differing written bytes by the declared ratio. A second, unconditional gate
    rejects sign changes (including signed zero) or a jump over more than one
    magnitude level in either nibble. This is an end-to-end numerical policy,
    not an exact quantizer test or a proof of a floating-point error bound.
    """
    from golden import mapped_pool_ratio_allclose

    mapped = mapped_pool_ratio_allclose(
        mapping_name, mapping_shape=mapping_shape, block_size=block_size,
        leading_rank_axis=leading_rank_axis, pool_name=pool_name,
        atol=0.0, rtol=0.0, max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.dtype != torch.uint8 or expected.dtype != torch.uint8:
            return False, f"{pool_name}: packed E2M1 payload must have UINT8 storage"
        ok, detail = mapped(actual, expected, **kwargs)
        if not ok:
            return False, detail
        a = actual.detach().cpu().to(torch.int16)
        e = expected.detach().cpu().to(torch.int16)
        sign_changed = torch.bitwise_and(torch.bitwise_xor(a, e), 0x88) != 0
        if bool(sign_changed.any()):
            return False, f"{pool_name}: E2M1 sign changed in {int(sign_changed.sum())} byte(s)"
        low_distance = ((a & 7) - (e & 7)).abs()
        high_distance = (((a >> 4) & 7) - ((e >> 4) & 7)).abs()
        invalid = (low_distance > 1) | (high_distance > 1)
        if bool(invalid.any()):
            return False, f"{pool_name}: E2M1 magnitude skips a level in {int(invalid.sum())} byte(s)"
        return True, ""

    return compare


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
    from quant import unpack_bf16_nz

    projected = torch.matmul(latent.float(), unpack_bf16_nz(wk).float()).to(latent.dtype)
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


def golden_index_weighted_sum(dots, weights):
    """Accumulate heads in order with one BF16 rounding per fused multiply-add."""
    values = torch.zeros(dots.shape[-1], dtype=torch.bfloat16, device=dots.device)
    positive = dots.relu().double()
    head_weights = weights.double()
    for head in range(dots.shape[0]):
        values = (positive[head] * head_weights[head] + values.double()).bfloat16()
    return values.float()


def golden_c1a_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from qkv_proj_rope import golden_rope_interleave as rope_interleave
    from quant import dequantize_mxfp4_cache, unpack_bf16_nz, unpack_fp8_nz

    x = a["x"]
    lengths = a["compressed_lens"]
    # Logical page-table score columns.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(a["index_wq_b"]), a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    iq = golden_index_query_fp4(iq)
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = ((x).float() @ unpack_bf16_nz(a["index_weights_proj"]).float()).to(torch.bfloat16).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = (pages >= 0) & (pages < index_payload.shape[0])
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # BF16 dot products, ReLU, and sequential BF16 head FMAs.
        dots = ((iq[t]).float() @ (keys[rows.long()].T).float()).to(torch.bfloat16)
        values = golden_index_weighted_sum(dots, weight[t])
        scores[t, valid_pos] = values
    return scores, physical


def golden_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts):
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
    """Select finite TopK rows with stable ties in logical-position order."""
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
    """Select C2A rows in logical order, including every valid short-history row."""
    tokens, width = scores.shape
    output = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        count = max(0, min(int(lengths[token]), width))
        if count <= INDEX_TOPK:
            output[token, :count] = physical_rows[token, :count].to(torch.int32)
            continue
        # Stable descending order makes the common logical-earliest cutoff explicit.
        logical = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        logical = logical[torch.isfinite(scores[token, logical])].sort().values
        rows = physical_rows[token, logical]
        rows = rows[rows >= 0]
        output[token, :rows.numel()] = rows.to(torch.int32)
    return output


def golden_c2a_index(t, qr, index_cache, index_cache_scale):
    """Score and select C2A paged history with BF16 head reduction."""
    from quant import unpack_bf16_nz, unpack_fp8_nz

    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    index_values = dequantize_mxfp4_cache(index_cache, index_cache_scale, INDEX_CACHE_GROUP, "e8m0").flatten(0, -2)
    # Accumulate the index-query projection in sequential group-32 partials.
    index_q = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(t["index_wq_b"]), t["index_wq_b_scale"]).unflatten(-1, (index_heads, index_dim))
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    index_q = golden_index_query_fp4(index_q)
    head_weights = ((x).float() @ unpack_bf16_nz(t["index_weights_proj"]).float()).to(torch.bfloat16).float()
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
        # Round each fused head multiply-add to BF16.
        dots = ((index_q[token]).float() @ (index_values[valid_rows].to(torch.bfloat16).T).float()).to(torch.bfloat16)
        total = golden_index_weighted_sum(dots, head_weights[token])
        scores[token, valid_logical] = total
    return _select_shared_c2a_topk(scores, physical_rows, lens)


__all__ = [
    "c1a_index",
    "decode_c2a_indexer",
    "golden_c2a_index",
    "golden_c1a_index_scores",
    "golden_pool_index_scores",
    "golden_c1a_index_topk",
    "score_select_reindex",
]


def index_pool_block_capacity(score_width: int) -> int:
    """Return the aligned compact-pool capacity for one score row."""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = FLASH.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(FLASH.candidate_topk_blocks, aligned)
