# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded layer-2 C1/C2 composition for the A2 W8A8 verifier.

This entry keeps the official A2 cache ABI: ratio-2 compressor state is FP32,
compressed KV is BF16, and index keys are BF16 before INT8/FP16 publication.
It is deliberately a module-level boundary; sparse selection and composed
attention remain separate dependencies.
"""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .ops import VEC_M_TILE, make_norm, make_rope
from .compressor_core import compressor_ratio2, RING_ROWS_DYN
from .config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN, T_DYN
from .qkv_w8a8 import make_qkv_w8a8_inline
from .qkv_proj_rope import make_bf16_projection_staged


HEAD_DIM = 512
INDEX_WIDTH = 128


def make_fp32_projection(width, output_width):
    """Project widened BF16 storage with FP32 operands and accumulation."""
    m_tile, k_tile, n_tile = 16, 256, min(64, output_width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        blocks = (active + m_tile - 1) // m_tile * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="a2_fp32_projection") as projection_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * m_tile
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(m_tile, active - row)
            acc = pl.create_tensor([m_tile, n_tile], dtype=pl.FP32)
            for k0 in pl.range(0, width, k_tile):
                a = pl.slice(source, [m_tile, k_tile], [row, k0], valid_shape=[rows, k_tile])
                a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), m_tile, k_tile)
                a = pl.cast(a, pl.FP32)
                b = pl.slice(weight, [k_tile, n_tile], [k0, col])
                b = pl.cast(b, pl.FP32)
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            output[row : row + m_tile, col : col + n_tile] = pl.set_validshape(acc, rows, n_tile)
        return projection_tid

    return project


def make_indexer_cache_publish_128(capacity, blocks):
    """Publish width-128 rows without the generic 256-wide quantizer assumption."""
    rows = blocks * 128
    QUANT_M_TILE = 16

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        with pl.spmd((active + QUANT_M_TILE - 1) // QUANT_M_TILE, name_hint="a2_c1_index_quant"):
            block = pl.tile.get_block_idx()
            t0 = block * QUANT_M_TILE
            rows_valid = pl.min(QUANT_M_TILE, active - t0)
            source = pl.slice(value, [QUANT_M_TILE, INDEX_WIDTH], [t0, 0], valid_shape=[rows_valid, INDEX_WIDTH])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), QUANT_M_TILE, INDEX_WIDTH)
            value_fp32 = pl.cast(source, pl.FP32)
            maximum_row = pl.reshape(pl.row_max(pl.abs(value_fp32)), [1, QUANT_M_TILE])
            maximum_row = pl.maximum(maximum_row, 1e-30)
            scale = pl.reshape(pl.mul(maximum_row, 1.0 / 127.0), [QUANT_M_TILE, 1])
            # Match scalar Torch 127 / maximum: rounded unit reciprocal then 127.
            # Split the approximate reciprocal so BF16-maximum products retain
            # their residual on A2 without an FMA or A5-only precision flag.
            reciprocal = pl.div(pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=1.0), maximum_row)
            reciprocal_hi = pl.cast(pl.cast(reciprocal, pl.BF16, mode="rint"), pl.FP32)
            reciprocal_lo = pl.sub(reciprocal, reciprocal_hi)
            residual_hi = pl.sub(pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=1.0), pl.mul(reciprocal_hi, maximum_row))
            residual = pl.sub(residual_hi, pl.mul(reciprocal_lo, maximum_row))
            refined_reciprocal = pl.add(reciprocal, pl.mul(residual, reciprocal))
            inverse = pl.reshape(pl.mul(refined_reciprocal, 127.0), [QUANT_M_TILE, 1])
            normalized = pl.row_expand_mul(value_fp32, inverse)
            rounded = pl.cast(pl.minimum(pl.maximum(normalized, -127.0), 127.0), pl.INT32, mode="rint")
            payload[t0 : t0 + QUANT_M_TILE, :] = pl.set_validshape(pl.cast(rounded, pl.INT8, mode="trunc"), rows_valid, INDEX_WIDTH)
            row_scales[t0 : t0 + QUANT_M_TILE, :] = pl.row_expand_mul(
                pl.full([QUANT_M_TILE, 8], dtype=pl.FP32, value=1.0), scale
            )
        cache_flat = pl.reshape(cache, [rows, INDEX_WIDTH])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_index_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is narrower than the device cache line.  Publish
        # scales from one worker so writes to adjacent slots cannot race and
        # silently restore a neighboring row's sentinel values.
        with pl.spmd(1, name_hint="a2_c1_index_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    pl.write(scale_flat, [slot, 0], pl.cast(pl.read(row_scales, [token, 0]), pl.FP16))
        return cache, scales

    return publish


def make_c2_index_publish_projected_inline(capacity, config, cache_blocks=2):
    """Pool projected FP32 rows, then normalize, rotate and publish caches."""
    pairs = config["qk_rope_head_dim"] // 2
    eps = config["rms_norm_eps"]
    project_index = make_bf16_projection_staged(HEAD_DIM, INDEX_WIDTH)
    norm_index = make_norm(capacity, INDEX_WIDTH, eps)
    rope_index = make_rope(capacity, 1, INDEX_WIDTH, config["qk_rope_head_dim"])
    rope_compressed = make_rope(capacity, 1, HEAD_DIM, config["qk_rope_head_dim"])
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, HEAD_DIM)
    publish_index = make_indexer_cache_publish_128(capacity, cache_blocks)

    @pl.jit.inline
    def publish(
        kv_proj: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        score_proj: pl.Tensor[[capacity, HEAD_DIM], pl.FP32],
        compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        norm_ones = pl.create_tensor([1, HEAD_DIM], dtype=pl.BF16)
        for norm_row in pl.spmd(1, name_hint="a2_c2_unweighted_norm"):
            norm_ones[norm_row:norm_row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=1.0)
        norm_ones_flat = pl.reshape(norm_ones, [HEAD_DIM])
        unweighted_compressed = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.BF16)
        compressor_ratio2(
            kv_proj, score_proj, query_start, positions, requests,
            state_block_table, state, norm_ones_flat,
            unweighted_compressed, active, projections_ready,
        )
        # Native RMSNorm narrows the normalized value before its BF16 gamma
        # multiplication. Keep that boundary while sharing the existing pool.
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="a2_c2_norm_gamma"):
            t0 = block * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - t0)
            value = unweighted_compressed[t0:t0 + VEC_M_TILE, :]
            gamma = pl.reshape(pl.cast(compressor_norm[:], pl.FP32), [1, HEAD_DIM])
            weighted = pl.col_expand_mul(pl.cast(value, pl.FP32), gamma)
            compressed[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(
                pl.cast(weighted, pl.BF16, mode="rint"), rows, HEAD_DIM,
            )
        compressed_rotated = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.BF16)
        rope_compressed(compressed, source_cos, source_sin, compressed_rotated, active)
        publish_compressed(compressed_rotated, compressed_slots, compressed_cache, active)

        projected_index = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, index_cache, index_scale, active)
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_c2_index_publish_inline(capacity, config, cache_blocks=2):
    """Project BF16 inputs and publish C2 and index caches."""
    hidden = config["hidden_size"]
    pairs = config["qk_rope_head_dim"] // 2
    project_kv = make_fp32_projection(hidden, HEAD_DIM)
    publish_projected = make_c2_index_publish_projected_inline(capacity, config, cache_blocks)

    @pl.jit.inline
    def publish(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        compressor_wkv: pl.Tensor[[hidden, HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT32],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * HEAD_DIM], pl.FP32],
        compressed_cache: pl.Tensor[[cache_blocks, 128, 1, HEAD_DIM], pl.BF16],
        index_cache: pl.Tensor[[cache_blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        index_scale: pl.Tensor[[cache_blocks, 128, 1, 1], pl.FP16],
        compressed: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        index_key: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        kv_proj = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
        score_proj = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
        project_kv(x, compressor_wkv, kv_proj, active)
        score_ready = project_kv(x, compressor_wgate, score_proj, active)
        publish_projected(
            kv_proj, score_proj, compressor_norm, index_wk, index_norm,
            source_cos, source_sin, positions, compressed_slots,
            query_start, requests, state_block_table,
            state, compressed_cache, index_cache, index_scale,
            compressed, index_key,
            active, score_ready,
        )
        return state, compressed, compressed_cache, index_key, index_cache, index_scale

    return publish


def make_layer2_c1c2(capacity, config):
    """Build QKV, ratio-2 C2 pooling and C1 index-key publication."""
    hidden = config["hidden_size"]
    latent_width = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    query_width = heads * head_dim
    pair_width = config["qk_rope_head_dim"] // 2
    qkv = make_qkv_w8a8_inline(capacity, config)
    publish_c2 = make_c2_index_publish_inline(capacity, config)

    @pl.jit
    def layer2_c1c2(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent_width], pl.INT8],
        wq_a_scale: pl.Tensor[[latent_width], pl.FP32],
        q_norm_weight: pl.Tensor[[latent_width], pl.BF16],
        wq_b: pl.Tensor[[latent_width, query_width], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pair_width], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pair_width], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        positions32: pl.Tensor[[capacity], pl.INT32],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent_width], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        qkv_cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        compressor_wkv: pl.Tensor[[hidden, HEAD_DIM], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, HEAD_DIM], pl.BF16],
        compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        index_wk: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        index_norm: pl.Tensor[[INDEX_WIDTH], pl.BF16],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 2 * HEAD_DIM], pl.FP32]],
        compressed_cache: pl.InOut[pl.Tensor[[2, 128, 1, HEAD_DIM], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[2, 128, 1, INDEX_WIDTH], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[2, 128, 1, 1], pl.FP16]],
        compressed: pl.InOut[pl.Tensor[[capacity, HEAD_DIM], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        qkv(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, positions,
            query_latent, query, window_kv, qkv_cache, cache_lens, active,
        )

        publish_c2(
            x, compressor_wkv, compressor_wgate, compressor_norm, index_wk, index_norm,
            rope_cos, rope_sin, positions32, compressed_slots, query_start, requests,
            state_block_table, state, compressed_cache, index_cache, index_scale,
            compressed, index_key, active,
        )
        return (
            positions, query_latent, query, window_kv, qkv_cache, cache_lens,
            state, compressed, compressed_cache, index_key, index_cache, index_scale,
        )

    return layer2_c1c2
