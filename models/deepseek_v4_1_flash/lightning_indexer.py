# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared Index-Q projection and native paged lightning indexer bindings."""
import os
from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir

from config import FLASH


# model config
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
Q_LORA = FLASH.q_lora_rank
QUERY_WIDTH = INDEX_H * INDEX_DIM

# tiling
INDEX_Q_M_TILE = 80
INDEX_Q_WIDE_M_TILE = 128
INDEX_Q_K_TILE = 128


KERNEL_DIR = Path(__file__).parent / "kernels"
LI_SOURCE = KERNEL_DIR / "lightning_indexer.cpp"
CANN = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
WORKERS = 32


LI_INCLUDES = (pto_isa_include_dir(), CANN / "asc", CANN / "asc/include",
               CANN / "asc/include/basic_api", CANN / "include")
# Stage tags in the lightning_indexer.cpp task ABI.
METADATA, ATTENTION, SCORE, POOL, SCORE_CANDIDATES = range(5)


@pl.jit.extern(core_type="aiv", aiv_source=LI_SOURCE, include_dirs=LI_INCLUDES)
def _score_prepare(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Out[pl.Tensor],
    mode: pl.Scalar[pl.INT64],
    active: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=KERNEL_DIR / "prefill_lightning_indexer.cpp",
               dual_aiv_dispatch=True, include_dirs=(*LI_INCLUDES, KERNEL_DIR))
def _prefill_score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    active: pl.Scalar[pl.INT64],
    mode: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.extern(core_type="mixed", aic_source=LI_SOURCE, aiv_source=LI_SOURCE,
               dual_aiv_dispatch=True, include_dirs=LI_INCLUDES)
def _score_candidates(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Out[pl.Tensor],
    block_scores: pl.Out[pl.Tensor],
    mode: pl.Scalar[pl.INT64],
    active: pl.Scalar[pl.INT64],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def prepare_score_metadata(
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    active_rows = pl.cast(num_tokens, pl.INT64)
    metadata = pl.create_tensor([33, 2], dtype=pl.INT32)
    with pl.spmd(1, name_hint="qli_score_metadata"):
        _score_prepare(request_ids, lengths, metadata, METADATA, active_rows)
    return metadata


@pl.jit.inline(auto_scope=False)
def score_paged(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    key: pl.Tensor,
    key_scale: pl.Tensor,
    table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    metadata: pl.Tensor,
    scores: pl.Tensor,
    active: pl.Scalar[pl.INT32],
    block_scores: pl.Tensor,
    candidates: pl.constexpr = False,
):
    active_rows = pl.cast(active, pl.INT64)
    with pl.spmd(WORKERS, name_hint="qli_score"):
        if candidates:
            _score_candidates(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, block_scores, SCORE_CANDIDATES, active_rows)
        else:
            _score(query, query_scale, weights, key, key_scale, table, request_ids, lengths, metadata, scores, active_rows, SCORE)


@pl.jit.inline(auto_scope=False)
def _project_index_query_tiles(
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
    row_tile: pl.constexpr,
    worker_limit: pl.constexpr,
    route: pl.Tensor,
    guarded: pl.constexpr,
):
    """Keep each two-head weight panel in Mat across the row tiles in a task."""
    scale_width = Q_LORA // 32
    capacity = pl.tensor.dim(payload, 0)
    flat_scales = pl.reshape(activation_scale, [capacity * scale_width])
    nz_weight: pl.Tensor[[Q_LORA, QUERY_WIDTH], pl.FP8E4M3FN, pl.NZ] = weight[:, :]
    b_scale_backing = pl.tensor.view(weight_scale, [1, Q_LORA // 32 * QUERY_WIDTH], layout=pl.ND)
    b_scale = pl.tensor.view(b_scale_backing, [Q_LORA // 32, QUERY_WIDTH], layout=pl.MX_B_NN)
    workers = pl.min(worker_limit, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="idx_q_fused_reg", deps=[ready]) as completed:
        worker = pl.tile.get_block_idx()
        block_num = pl.tile.get_block_num()
        waves = (INDEX_H // 2 + block_num - 1) // block_num
        if guarded:
            if pl.read(route, [0]) == 0:
                waves = pl.cast(0, pl.INDEX)
        for wave in pl.range(waves):
            block = wave * block_num + worker
            if block < INDEX_H // 2:
                full_b = pl.tile.load(nz_weight, [0, block * 256], [Q_LORA, 256], target_memory=pl.MemorySpace.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP)
                full_sb = pl.tile.load(b_scale, [0, block * 256], [scale_width, 256], target_memory=pl.MemorySpace.Mat)
                for begin in pl.range(0, num_tokens, row_tile):
                    full_a = pl.tile.load(payload, [begin, 0], [row_tile, Q_LORA], target_memory=pl.MemorySpace.Mat)
                    # Each M-aligned row tile is contiguous in the packed activation scale stream.
                    scale_bytes = pl.tensor.view(flat_scales, [row_tile * scale_width], offsets=[begin * scale_width])
                    a_scale = pl.tensor.view(scale_bytes, [row_tile, scale_width], layout=pl.MX_A_ZZ)
                    full_sa = pl.tile.load(a_scale, [0, 0], [row_tile, scale_width], target_memory=pl.MemorySpace.Mat)
                    for prep_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        row = begin + prep_lane * (row_tile // 2)
                        rows = pl.max(0, pl.min(row_tile // 2, num_tokens - row))
                        # Reserve a zero tail row for the 64-lane loads at 32-column stride.
                        cos_rows = pl.tile.load(cosine, [row, 0], [row_tile // 2 + 1, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
                        sin_rows = pl.tile.load(sine, [row, 0], [row_tile // 2 + 1, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
                        cos_rows = pl.tile.fillpad_inplace(cos_rows)
                        sin_rows = pl.tile.fillpad_inplace(sin_rows)
                    for head in pl.range(2):
                        # Every other scale slot matches the address of its paired L0 data slot.
                        col = head * INDEX_DIM
                        a0: pl.Tile[[row_tile, INDEX_Q_K_TILE], pl.FP8E4M3FN, pl.MemRef("idx_q_a_l0", slots=2)[0], pl.Mem.Left] = pl.tile.extract(
                            full_a, 0, 0, [row_tile, INDEX_Q_K_TILE], target_memory=pl.Mem.Left,
                        )
                        sa0: pl.Tile[[row_tile, INDEX_Q_K_TILE // 32], pl.FP8E8M0, pl.MemRef("idx_q_sa_l0", slots=4)[0], pl.Mem.LeftScale] = pl.tile.extract(
                            full_sa, 0, 0, [row_tile, INDEX_Q_K_TILE // 32], target_memory=pl.Mem.LeftScale,
                        )
                        b0: pl.Tile[[INDEX_Q_K_TILE, INDEX_DIM], pl.FP8E4M3FN, pl.MemRef("idx_q_b_l0", slots=2)[0], pl.Mem.Right] = pl.tile.extract(
                            full_b, 0, col, [INDEX_Q_K_TILE, INDEX_DIM], target_memory=pl.Mem.Right,
                        )
                        sb0: pl.Tile[[INDEX_Q_K_TILE // 32, INDEX_DIM], pl.FP8E8M0, pl.MemRef("idx_q_sb_l0", slots=4)[0], pl.Mem.RightScale] = pl.tile.extract(
                            full_sb, 0, col, [INDEX_Q_K_TILE // 32, INDEX_DIM], target_memory=pl.Mem.RightScale,
                        )
                        acc = pl.matmul_mx(a0, sa0, b0, sb0)
                        for k0 in pl.unroll(INDEX_Q_K_TILE, Q_LORA, INDEX_Q_K_TILE):
                            k_slot = k0 // INDEX_Q_K_TILE % 2
                            scale_slot = 2 * k_slot
                            scale_k = k0 // 32
                            a: pl.Tile[[row_tile, INDEX_Q_K_TILE], pl.FP8E4M3FN, pl.MemRef("idx_q_a_l0", slots=2)[k_slot], pl.Mem.Left] = pl.tile.extract(
                                full_a, 0, k0, [row_tile, INDEX_Q_K_TILE], target_memory=pl.Mem.Left,
                            )
                            sa: pl.Tile[[row_tile, INDEX_Q_K_TILE // 32], pl.FP8E8M0, pl.MemRef("idx_q_sa_l0", slots=4)[scale_slot], pl.Mem.LeftScale] = pl.tile.extract(
                                full_sa, 0, scale_k, [row_tile, INDEX_Q_K_TILE // 32], target_memory=pl.Mem.LeftScale,
                            )
                            b: pl.Tile[[INDEX_Q_K_TILE, INDEX_DIM], pl.FP8E4M3FN, pl.MemRef("idx_q_b_l0", slots=2)[k_slot], pl.Mem.Right] = pl.tile.extract(
                                full_b, k0, col, [INDEX_Q_K_TILE, INDEX_DIM], target_memory=pl.Mem.Right,
                            )
                            sb: pl.Tile[[INDEX_Q_K_TILE // 32, INDEX_DIM], pl.FP8E8M0, pl.MemRef("idx_q_sb_l0", slots=4)[scale_slot], pl.Mem.RightScale] = pl.tile.extract(
                                full_sb, scale_k, col, [INDEX_Q_K_TILE // 32, INDEX_DIM], target_memory=pl.Mem.RightScale,
                            )
                            acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            raw = pl.aiv_shard(acc)
                            row = begin + lane * (row_tile // 2)
                            rows = pl.max(0, pl.min(row_tile // 2, num_tokens - row))
                            if rows > 0:
                                source_vec = pl.reshape(raw, [row_tile // 2 * INDEX_DIM])
                                packed = pl.create_tile([row_tile // 2 * 64], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec)
                                # Reserve one full register after the last active scale word.
                                scale_words = pl.create_tile([(row_tile // 2 + 8) * 8], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                                # The UB inputs, payload, and scale output occupy disjoint ranges.
                                with pl.reg_scope(auto_sync=False):
                                    q_mask = pl.reg.full_mask(lanes=64)
                                    q_single = pl.reg.active_mask(1, lanes=64)
                                    q_lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                                    q_one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                                    q_low_bit = pl.reg.bitwise_and(q_lane, q_one)
                                    q_pair_index = pl.reg.shift_lefts(q_low_bit, 5, q_mask)
                                    q_fp4_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                                    q_fp4_one = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
                                    q_fp4_normal_min = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
                                    q_fp4_max = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
                                    for token_row in pl.hardware_loop(rows):
                                        r1_cos = pl.reg.load(cos_rows, token_row * 32, lanes=64)
                                        r1_sin = pl.reg.load(sin_rows, token_row * 32, lanes=64)
                                        r0_mask = pl.reg.full_mask(lanes=64)
                                        r0_first = pl.reg.active_mask(32, lanes=64)
                                        r0_raw = pl.reg.load(source_vec, token_row * INDEX_DIM + 0, lanes=64)
                                        r0_values = pl.reg.convert(r0_raw, dtype=pl.FP32)
                                        r0_magnitude = pl.reg.abs(r0_values)
                                        r0_group_max = pl.reg.reduce_max(r0_magnitude, r0_mask, group_size=32)
                                        r0_maximum = pl.reg.group_broadcast(r0_group_max, lanes=64)
                                        r0_minimum = pl.reg.broadcast(6.0 * 2.0**-126, lanes=64, dtype=pl.FP32)
                                        r0_maximum = pl.reg.maximum(r0_maximum, r0_minimum)
                                        r0_raw_scale = pl.reg.muls(r0_maximum, 1.0 / 6.0, r0_mask)
                                        r0_scale_bits = pl.reg.interpret_cast(r0_raw_scale, dtype=pl.INT32)
                                        r0_scale_bits = pl.reg.adds(r0_scale_bits, 8388607, r0_mask)
                                        r0_exponent = pl.reg.shift_rights(r0_scale_bits, 23, r0_mask)
                                        r0_inverse_bias = pl.reg.broadcast(254, lanes=64, dtype=pl.INT32)
                                        r0_inverse_exp = pl.reg.sub(r0_inverse_bias, r0_exponent)
                                        r0_inverse_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                                        r0_inverse_exp = pl.reg.maximum(r0_inverse_exp, r0_inverse_zero)
                                        r0_factor_bits = pl.reg.shift_lefts(r0_inverse_exp, 23, r0_mask)
                                        r0_factor = pl.reg.interpret_cast(r0_factor_bits, dtype=pl.FP32)
                                        r0_normalized = pl.reg.mul(r0_values, r0_factor)
                                        r0_magnitude = pl.reg.abs(r0_normalized)
                                        # Round the normal FP4 mantissa at bit 22 and handle subnormals separately.
                                        r0_magnitude_bits = pl.reg.interpret_cast(r0_magnitude, dtype=pl.INT32)
                                        r0_rounded_bits = pl.reg.adds(r0_magnitude_bits, 2097152, r0_mask)
                                        r0_normal = pl.reg.shift_rights(r0_rounded_bits, 22, r0_mask)
                                        r0_normal = pl.reg.adds(r0_normal, -252, r0_mask)
                                        r0_normal = pl.reg.maximum(r0_normal, q_fp4_normal_min)
                                        r0_normal = pl.reg.minimum(r0_normal, q_fp4_max)
                                        r0_nonzero = pl.reg.compares(r0_magnitude, 0.25, r0_mask, predicate="ge")
                                        r0_is_normal = pl.reg.compares(r0_magnitude, 0.75, r0_mask, predicate="ge")
                                        r0_small = pl.reg.select(r0_nonzero, q_fp4_one, q_fp4_zero)
                                        r0_codes = pl.reg.select(r0_is_normal, r0_normal, r0_small)
                                        r0_value_bits = pl.reg.interpret_cast(r0_normalized, dtype=pl.INT32)
                                        r0_sign = pl.reg.shift_rights(r0_value_bits, 28, r0_mask)
                                        r0_sign_mask = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                                        r0_sign = pl.reg.bitwise_and(r0_sign, r0_sign_mask)
                                        r0_codes = pl.reg.bitwise_or(r0_codes, r0_sign)
                                        r0_even_codes, r0_odd_codes = pl.reg.dintlv(r0_codes, r0_codes, r0_mask)
                                        r0_odd_codes = pl.reg.shift_lefts(r0_odd_codes, 4, r0_mask)
                                        r0_packed_codes = pl.reg.bitwise_or(r0_even_codes, r0_odd_codes)
                                        r1_mask = pl.reg.full_mask(lanes=64)
                                        r1_first = pl.reg.active_mask(32, lanes=64)
                                        r1_raw = pl.reg.load(source_vec, token_row * INDEX_DIM + 64, lanes=64)
                                        r1_values = pl.reg.convert(r1_raw, dtype=pl.FP32)
                                        r1_even, r1_odd = pl.reg.dintlv(r1_values, r1_values, r1_mask)
                                        r1_ec = pl.reg.mul(r1_even, r1_cos, r1_first)
                                        r1_os = pl.reg.mul(r1_odd, r1_sin, r1_first)
                                        r1_es = pl.reg.mul(r1_even, r1_sin, r1_first)
                                        r1_oc = pl.reg.mul(r1_odd, r1_cos, r1_first)
                                        r1_real = pl.reg.sub(r1_ec, r1_os)
                                        r1_imag = pl.reg.add(r1_es, r1_oc)
                                        r1_rotated, r1_unused = pl.reg.intlv(r1_real, r1_imag, r1_mask)
                                        r1_values = r1_rotated
                                        r1_magnitude = pl.reg.abs(r1_values)
                                        r1_group_max = pl.reg.reduce_max(r1_magnitude, r1_mask, group_size=32)
                                        r1_maximum = pl.reg.group_broadcast(r1_group_max, lanes=64)
                                        r1_minimum = pl.reg.broadcast(6.0 * 2.0**-126, lanes=64, dtype=pl.FP32)
                                        r1_maximum = pl.reg.maximum(r1_maximum, r1_minimum)
                                        r1_raw_scale = pl.reg.muls(r1_maximum, 1.0 / 6.0, r1_mask)
                                        r1_scale_bits = pl.reg.interpret_cast(r1_raw_scale, dtype=pl.INT32)
                                        r1_scale_bits = pl.reg.adds(r1_scale_bits, 8388607, r1_mask)
                                        r1_exponent = pl.reg.shift_rights(r1_scale_bits, 23, r1_mask)
                                        r1_inverse_bias = pl.reg.broadcast(254, lanes=64, dtype=pl.INT32)
                                        r1_inverse_exp = pl.reg.sub(r1_inverse_bias, r1_exponent)
                                        r1_inverse_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                                        r1_inverse_exp = pl.reg.maximum(r1_inverse_exp, r1_inverse_zero)
                                        r1_factor_bits = pl.reg.shift_lefts(r1_inverse_exp, 23, r1_mask)
                                        r1_factor = pl.reg.interpret_cast(r1_factor_bits, dtype=pl.FP32)
                                        r1_normalized = pl.reg.mul(r1_values, r1_factor)
                                        r1_magnitude = pl.reg.abs(r1_normalized)
                                        # Round the normal FP4 mantissa at bit 22 and handle subnormals separately.
                                        r1_magnitude_bits = pl.reg.interpret_cast(r1_magnitude, dtype=pl.INT32)
                                        r1_rounded_bits = pl.reg.adds(r1_magnitude_bits, 2097152, r1_mask)
                                        r1_normal = pl.reg.shift_rights(r1_rounded_bits, 22, r1_mask)
                                        r1_normal = pl.reg.adds(r1_normal, -252, r1_mask)
                                        r1_normal = pl.reg.maximum(r1_normal, q_fp4_normal_min)
                                        r1_normal = pl.reg.minimum(r1_normal, q_fp4_max)
                                        r1_nonzero = pl.reg.compares(r1_magnitude, 0.25, r1_mask, predicate="ge")
                                        r1_is_normal = pl.reg.compares(r1_magnitude, 0.75, r1_mask, predicate="ge")
                                        r1_small = pl.reg.select(r1_nonzero, q_fp4_one, q_fp4_zero)
                                        r1_codes = pl.reg.select(r1_is_normal, r1_normal, r1_small)
                                        r1_value_bits = pl.reg.interpret_cast(r1_normalized, dtype=pl.INT32)
                                        r1_sign = pl.reg.shift_rights(r1_value_bits, 28, r1_mask)
                                        r1_sign_mask = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
                                        r1_sign = pl.reg.bitwise_and(r1_sign, r1_sign_mask)
                                        r1_codes = pl.reg.bitwise_or(r1_codes, r1_sign)
                                        r1_even_codes, r1_odd_codes = pl.reg.dintlv(r1_codes, r1_codes, r1_mask)
                                        r1_odd_codes = pl.reg.shift_lefts(r1_odd_codes, 4, r1_mask)
                                        r1_packed_codes = pl.reg.bitwise_or(r1_even_codes, r1_odd_codes)
                                        # Join the duplicated packed halves into one 64-byte payload.
                                        payload_codes = pl.reg.select(r0_first, r0_packed_codes, r1_packed_codes)
                                        payload_signed = pl.reg.convert(payload_codes, dtype=pl.INT8, saturate=False)
                                        payload_bytes = pl.reg.interpret_cast(payload_signed, dtype=pl.UINT8)
                                        _payload_written = pl.reg.store(packed, token_row * 64, payload_bytes, q_mask)
                                        # Pack four E8M0 exponents into one 32-bit word.
                                        r0_pair_scale = pl.reg.gather_lanes(r0_exponent, q_pair_index)
                                        r0_lo_scale, r0_hi_scale = pl.reg.dintlv(r0_pair_scale, r0_pair_scale, q_mask)
                                        r0_shifted_scale = pl.reg.shift_lefts(r0_hi_scale, 8, q_mask)
                                        r0_packed_scale = pl.reg.bitwise_or(r0_lo_scale, r0_shifted_scale)
                                        r1_pair_scale = pl.reg.gather_lanes(r1_exponent, q_pair_index)
                                        r1_lo_scale, r1_hi_scale = pl.reg.dintlv(r1_pair_scale, r1_pair_scale, q_mask)
                                        r1_shifted_scale = pl.reg.shift_lefts(r1_hi_scale, 8, q_mask)
                                        r1_packed_scale = pl.reg.bitwise_or(r1_lo_scale, r1_shifted_scale)
                                        r1_scale_word = pl.reg.shift_lefts(r1_packed_scale, 16, q_mask)
                                        scale_word = pl.reg.bitwise_or(r0_packed_scale, r1_scale_word)
                                        _scale_written = pl.reg.store(scale_words, token_row * 8, scale_word, q_single)
                                payload_tile = pl.reshape(packed, [row_tile // 2, 64])
                                payload_valid = pl.set_validshape(payload_tile, rows, 64)
                                output = pl.tile.store(payload_valid, [row, block * 128 + head * 64], output)
                                scale_matrix = pl.reshape(scale_words, [row_tile // 2 + 8, 8])
                                scale_tile = pl.reinterpret_view(scale_matrix, pl.UINT8)
                                encoded = pl.reinterpret_view(scale_tile, pl.FP8E8M0)
                                scale_valid = pl.set_validshape(encoded, rows, 4)
                                output_scale = pl.tile.store(scale_valid, [row, block * 8 + head * 4], output_scale)
    return completed


@pl.jit.inline(auto_scope=False)
def idx_q_pypto(
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
    worker_limit: pl.constexpr,
    route: pl.Tensor,
    guarded: pl.constexpr,
):
    """Select a row tile for the fused MXFP8 projection and MXFP4 RoPE epilogue."""
    completed = ready
    if num_tokens > 0:
        capacity = pl.tensor.dim(payload, 0)
        narrow_rows = ((num_tokens + INDEX_Q_M_TILE - 1) // INDEX_Q_M_TILE) * INDEX_Q_M_TILE
        wide_rows = ((num_tokens + INDEX_Q_WIDE_M_TILE - 1) // INDEX_Q_WIDE_M_TILE) * INDEX_Q_WIDE_M_TILE
        if wide_rows <= narrow_rows and wide_rows <= capacity:
            completed = _project_index_query_tiles(
                payload, weight, activation_scale, weight_scale,
                cosine, sine, output, output_scale, num_tokens, completed, INDEX_Q_WIDE_M_TILE,
                worker_limit, route, guarded,
            )
        else:
            completed = _project_index_query_tiles(
                payload, weight, activation_scale, weight_scale,
                cosine, sine, output, output_scale, num_tokens, completed, INDEX_Q_M_TILE,
                worker_limit, route, guarded,
            )
    return completed
