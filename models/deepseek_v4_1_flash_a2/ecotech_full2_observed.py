# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full2 Prefill projection, pooling, indexing and device observation ports."""
import pypto.language as pl
from .layer2_c1c2 import HEAD_DIM, INDEX_WIDTH, make_fp32_projection
from .c1c2 import make_compressed_cache_publish
from .ops import VEC_M_TILE, N_TILE, make_norm, make_rope, make_dequantize, make_dot, make_pack_weight, make_quantize
from .compressor_core import RING_ROWS_DYN
from .c2_prefill_pool import compressor_ratio2_prefill
from .operator_config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN
from .qkv_proj_rope import make_bf16_projection_staged
from .decode_indexer import INDEX_HEADS, WEIGHT_ROWS_TILE, QUANT_ROWS_TILE


def make_wk_input_observed_inline(capacity):
    project = make_bf16_projection_staged(HEAD_DIM, INDEX_WIDTH)

    @pl.jit.inline
    def observe(
        source: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        weight: pl.Tensor[[HEAD_DIM, INDEX_WIDTH], pl.BF16],
        output: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        consumer_input: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint="L2_observe_actual_WK_input"):
            row = pl.tile.get_block_idx() * VEC_M_TILE
            rows = pl.min(VEC_M_TILE, active - row)
            value = pl.slice(source, [VEC_M_TILE, HEAD_DIM], [row, 0], valid_shape=[rows, HEAD_DIM])
            consumer_input[row:row + VEC_M_TILE, :] = pl.set_validshape(value, rows, HEAD_DIM)
        project(source, weight, output, active)
        return output, consumer_input

    return observe

def make_indexer_cache_publish_128_observed(capacity, blocks):
    """Publish width-128 rows without the generic 256-wide quantizer assumption."""
    rows = blocks * 128

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, INDEX_WIDTH], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, INDEX_WIDTH], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        payload: pl.Tensor[[capacity, INDEX_WIDTH], pl.INT8],
        row_scales: pl.Tensor[[capacity, 8], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE, name_hint='a2_c1_index_quant'):
            block = pl.tile.get_block_idx()
            t0 = block * VEC_M_TILE
            rows_valid = pl.min(VEC_M_TILE, active - t0)
            source = pl.slice(value, [VEC_M_TILE, INDEX_WIDTH], [t0, 0], valid_shape=[rows_valid, INDEX_WIDTH])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), VEC_M_TILE, INDEX_WIDTH)
            value_fp32 = pl.cast(source, pl.FP32)
            maximum_row = pl.reshape(pl.row_max(pl.abs(value_fp32)), [1, VEC_M_TILE])
            maximum_row = pl.maximum(maximum_row, 1e-30)
            scale = pl.reshape(pl.mul(maximum_row, 1.0 / 127.0), [VEC_M_TILE, 1])
            reciprocal_maximum = pl.full([1, VEC_M_TILE], dtype=pl.FP32, value=0.0)
            for reciprocal_lane in pl.range(VEC_M_TILE):
                maximum_scalar: pl.Scalar[pl.FP32] = pl.read(maximum_row, [0, reciprocal_lane])
                reciprocal_scalar: pl.Scalar[pl.FP32] = 1.0 / maximum_scalar
                pl.write(reciprocal_maximum, [0, reciprocal_lane], reciprocal_scalar)
            inverse_row = pl.mul(reciprocal_maximum, 127.0)
            inverse = pl.reshape(inverse_row, [VEC_M_TILE, 1])
            normalized = pl.row_expand_mul(value_fp32, inverse)
            rounded = pl.cast(pl.minimum(pl.maximum(normalized, -127.0), 127.0), pl.INT32, mode='rint')
            payload[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(pl.cast(rounded, pl.INT8, mode='trunc'), rows_valid, INDEX_WIDTH)
            row_scales[t0:t0 + VEC_M_TILE, :] = pl.set_validshape(pl.row_expand_mul(pl.full([VEC_M_TILE, 8], dtype=pl.FP32, value=1.0), scale), rows_valid, 8)
        cache_flat = pl.reshape(cache, [rows, INDEX_WIDTH])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint='a2_c1_index_publish'):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot:slot + 1, :] = payload[token:token + 1, :]
        with pl.spmd(1, name_hint='a2_c1_index_scale_publish'):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    pl.write(scale_flat, [slot, 0], pl.cast(pl.read(row_scales, [token, 0]), pl.FP16))
        return (cache, scales, payload, row_scales)
    return publish

def make_c2_index_publish_projected_observed_inline(capacity, config, cache_blocks=2):
    """Pool projected FP32 rows, then normalize, rotate and publish caches."""
    pairs = config['qk_rope_head_dim'] // 2
    eps = config['rms_norm_eps']
    project_index = make_wk_input_observed_inline(capacity)
    norm_index = make_norm(capacity, INDEX_WIDTH, eps)
    rope_index = make_rope(capacity, 1, INDEX_WIDTH, config['qk_rope_head_dim'])
    rope_compressed = make_rope(capacity, 1, HEAD_DIM, config['qk_rope_head_dim'])
    publish_compressed = make_compressed_cache_publish(capacity, cache_blocks, HEAD_DIM)
    publish_index = make_indexer_cache_publish_128_observed(capacity, cache_blocks)

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
        compressed_rotated: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        key_payload: pl.Tensor[[capacity, INDEX_WIDTH], pl.INT8],
        key_scale_f32_8: pl.Tensor[[capacity, 8], pl.FP32],
        wk_input_readback: pl.Tensor[[capacity, HEAD_DIM], pl.BF16],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        compressor_ratio2_prefill(
            kv_proj, score_proj,
            query_start, positions, requests, state_block_table,
            state,
            compressor_norm, compressed,
            active, projections_ready,
        )
        rope_compressed(compressed, source_cos, source_sin, compressed_rotated, active)
        publish_compressed(compressed_rotated, compressed_slots, compressed_cache, active)
        projected_index = pl.create_tensor([capacity, INDEX_WIDTH], dtype=pl.BF16)
        project_index(compressed, index_wk, projected_index, wk_input_readback, active)
        norm_index(projected_index, index_norm, index_key, active)
        rope_index(index_key, source_cos, source_sin, index_key, active)
        publish_index(index_key, compressed_slots, index_cache, index_scale, key_payload, key_scale_f32_8, active)
        return (
            state,
            compressed,
            compressed_cache,
            index_key,
            index_cache, index_scale,
            compressed_rotated, key_payload, key_scale_f32_8, wk_input_readback,
        )
    return publish

def make_indexer_projection_observed(capacity, config):
    """Project and rotate index queries and emit scaled FP16 head weights."""
    hidden = config['hidden_size']
    latent = config['q_lora_rank']
    rope_width = config['qk_rope_head_dim']
    pairs = rope_width // 2
    index_width = INDEX_HEADS * INDEX_WIDTH
    packed_rows = latent * (index_width // N_TILE)
    weights_scale = INDEX_WIDTH ** (-0.5) * INDEX_HEADS ** (-0.5)
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
        projected: pl.Tensor[[capacity, index_width], pl.BF16],
        scaled_weights: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        latent_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        latent_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(query_latent, latent_i8, latent_scale, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(index_wq_b, packed)
        accumulator = pl.create_tensor([capacity, index_width], dtype=pl.INT32)
        dot(latent_i8, packed, accumulator, active)
        dequantize(accumulator, index_wq_b_scale, latent_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, index_width])
        rotate(projected, cosine, sine, query_flat, active)
        projected_weights = pl.create_tensor([capacity, INDEX_HEADS], dtype=pl.BF16)
        project_weights(x, weights_proj, projected_weights, active)
        with pl.spmd((active + WEIGHT_ROWS_TILE - 1) // WEIGHT_ROWS_TILE, name_hint='a2_index_weights'):
            block = pl.tile.get_block_idx()
            row = block * WEIGHT_ROWS_TILE
            rows = pl.min(WEIGHT_ROWS_TILE, active - row)
            value = pl.slice(projected_weights, [WEIGHT_ROWS_TILE, INDEX_HEADS], [row, 0], valid_shape=[rows, INDEX_HEADS])
            scaled = pl.mul(pl.cast(value, pl.FP32), weights_scale)
            scaled_weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(scaled, rows, INDEX_HEADS)
            narrowed = pl.cast(scaled, pl.FP16, mode='rint')
            weights[row:row + WEIGHT_ROWS_TILE, :] = pl.set_validshape(narrowed, rows, INDEX_HEADS)
        return (query, weights, projected, scaled_weights)
    return indexer_projection

def make_indexer_query_quantize_observed(capacity):
    """Quantize each index head using its rounded FP16 scale."""

    @pl.jit
    def indexer_query_quantize(
        query: pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS, INDEX_WIDTH], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, INDEX_HEADS], pl.FP16]],
        raw_scale_f32: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
        effective_scale_f32: pl.Tensor[[capacity, INDEX_HEADS], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(query, [capacity * INDEX_HEADS, INDEX_WIDTH])
        output = pl.reshape(payload, [capacity * INDEX_HEADS, INDEX_WIDTH])
        blocks = (active * INDEX_HEADS + QUANT_ROWS_TILE - 1) // QUANT_ROWS_TILE
        with pl.spmd(blocks, name_hint='a2_index_query_quant'):
            block = pl.tile.get_block_idx()
            row = block * QUANT_ROWS_TILE
            rows = pl.min(QUANT_ROWS_TILE, active * INDEX_HEADS - row)
            q = pl.slice(source, [QUANT_ROWS_TILE, INDEX_WIDTH], [row, 0], valid_shape=[rows, INDEX_WIDTH])
            q = pl.set_validshape(pl.fillpad(q, pad_value=pl.PadValue.zero), QUANT_ROWS_TILE, INDEX_WIDTH)
            value = pl.cast(q, pl.FP32)
            maximum = pl.row_max(pl.abs(value))
            raw_scale = pl.div(maximum, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode='rint'), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0 ** (-24))
            normalized = pl.row_expand_div(value, scale)
            rounded = pl.cast(normalized, pl.INT32, mode='rint')
            bounded = pl.minimum(pl.maximum(rounded, -127), 127)
            quantized = pl.cast(bounded, pl.INT8, mode='trunc')
            output[row:row + QUANT_ROWS_TILE, :] = pl.set_validshape(quantized, rows, INDEX_WIDTH)
            scale_row = pl.reshape(pl.cast(scale, pl.FP16, mode='rint'), [1, QUANT_ROWS_TILE])
            token = row // INDEX_HEADS
            head = row % INDEX_HEADS
            scales[token:token + 1, head:head + QUANT_ROWS_TILE] = scale_row
            raw_scale_row = pl.reshape(raw_scale, [1, QUANT_ROWS_TILE])
            raw_scale_f32[token:token + 1, head:head + QUANT_ROWS_TILE] = raw_scale_row
            effective_scale_row = pl.reshape(scale, [1, QUANT_ROWS_TILE])
            effective_scale_f32[token:token + 1, head:head + QUANT_ROWS_TILE] = effective_scale_row
        return (payload, scales, raw_scale_f32, effective_scale_f32)
    return indexer_query_quantize

def make_c2_program(config, *, project_inputs=True):
    """Pool and publish C2 rows, optionally projecting the source in DSL."""
    if type(project_inputs) is not bool:
        raise ValueError("The compressor projection boundary must be selected by the host")
    project = make_fp32_projection(5120, 512)
    publish_c2 = make_c2_index_publish_projected_observed_inline(16, config, cache_blocks=1)

    @pl.jit
    def c2_observed_program(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        compressor_wkv: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_wgate: pl.InOut[pl.Tensor[[5120, 512], pl.BF16]],
        compressor_norm: pl.InOut[pl.Tensor[[512], pl.BF16]],
        index_wk: pl.InOut[pl.Tensor[[512, 128], pl.BF16]],
        index_norm: pl.InOut[pl.Tensor[[128], pl.BF16]],
        source_cos: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        source_sin: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        positions32: pl.InOut[pl.Tensor[[16], pl.INT32]],
        compressed_slots: pl.InOut[pl.Tensor[[16], pl.INT64]],
        query_start: pl.InOut[pl.Tensor[[2], pl.INT32]],
        requests: pl.InOut[pl.Tensor[[16], pl.INT32]],
        state_block_table: pl.InOut[pl.Tensor[[1, 1], pl.INT32]],
        state: pl.InOut[pl.Tensor[[2, 32, 1024], pl.FP32]],
        compressed_cache: pl.InOut[pl.Tensor[[1, 128, 1, 512], pl.BF16]],
        index_cache: pl.InOut[pl.Tensor[[1, 128, 1, 128], pl.INT8]],
        index_scale: pl.InOut[pl.Tensor[[1, 128, 1, 1], pl.FP16]],
        kv_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        scores_projected: pl.InOut[pl.Tensor[[16, 512], pl.FP32]],
        compressed: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[16, 128], pl.BF16]],
        compressed_rotated: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        key_payload: pl.InOut[pl.Tensor[[16, 128], pl.INT8]],
        key_scale_f32_8: pl.InOut[pl.Tensor[[16, 8], pl.FP32]],
        wk_input_readback: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        if project_inputs:
            kv_ready = project(x, compressor_wkv, kv_projected, active)
            scores_ready = project(x, compressor_wgate, scores_projected, active)
            ready = pl.system.task_dummy(deps=[kv_ready, scores_ready])
            publish_c2(
                kv_projected, scores_projected,
                compressor_norm, index_wk, index_norm,
                source_cos, source_sin,
                positions32, compressed_slots, query_start, requests, state_block_table,
                state, compressed_cache, index_cache, index_scale,
                compressed, index_key, compressed_rotated, key_payload, key_scale_f32_8, wk_input_readback,
                active, ready,
            )
        else:
            ready = pl.system.task_dummy(deps=[])
            publish_c2(
                kv_projected, scores_projected,
                compressor_norm, index_wk, index_norm,
                source_cos, source_sin,
                positions32, compressed_slots, query_start, requests, state_block_table,
                state, compressed_cache, index_cache, index_scale,
                compressed, index_key, compressed_rotated, key_payload, key_scale_f32_8, wk_input_readback,
                active, ready,
            )
        return (
            x,
            compressor_wkv, compressor_wgate, compressor_norm, index_wk, index_norm,
            source_cos, source_sin,
            positions32, compressed_slots, query_start, requests, state_block_table,
            state, compressed_cache, index_cache, index_scale,
            kv_projected, scores_projected,
            compressed, index_key, compressed_rotated, key_payload, key_scale_f32_8, wk_input_readback,
        )

    return c2_observed_program
