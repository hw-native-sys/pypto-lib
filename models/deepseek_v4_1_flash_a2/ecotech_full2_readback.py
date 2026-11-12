# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observation-only InOut wrappers; each unchanged entry is called once."""
import pypto.language as pl

def make_readback_entries(entries, active_rows):
    compute_HC_post = pl.jit.inline(entries['HC_post']._func)
    assert compute_HC_post._func is entries['HC_post']._func
    @pl.jit
    def readback_HC_post(
        sublayer: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        residual: pl.InOut[pl.Tensor[[16, 4, 5120], pl.BF16]],
        post_mix: pl.InOut[pl.Tensor[[16, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[16, 4, 4], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 4, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_HC_post(sublayer, residual, post_mix, comb, output, active)
        return (sublayer, residual, post_mix, comb, output)

    compute_attention_norm = pl.jit.inline(entries['attention_norm']._func)
    assert compute_attention_norm._func is entries['attention_norm']._func
    @pl.jit
    def readback_attention_norm(
        collapsed: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_attention_norm(collapsed, weight, normed, active)
        return (collapsed, weight, normed)

    compute_c2_project_pool_publish = pl.jit.inline(entries['c2_project_pool_publish']._func)
    assert compute_c2_project_pool_publish._func is entries['c2_project_pool_publish']._func
    @pl.jit
    def readback_c2_project_pool_publish(
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
        compute_c2_project_pool_publish(
            x,
            compressor_wkv, compressor_wgate, compressor_norm, index_wk, index_norm,
            source_cos, source_sin,
            positions32, compressed_slots, query_start, requests, state_block_table,
            state, compressed_cache, index_cache, index_scale,
            kv_projected, scores_projected,
            compressed, index_key, compressed_rotated, key_payload, key_scale_f32_8, wk_input_readback,
            active,
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

    compute_empty_selection = pl.jit.inline(entries['empty_selection']._func)
    assert compute_empty_selection._func is entries['empty_selection']._func
    @pl.jit
    def readback_empty_selection(
        raw_selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_empty_selection(raw_selected, active)
        return (raw_selected,)

    compute_grouped_wo_a = pl.jit.inline(entries['grouped_wo_a']._func)
    assert compute_grouped_wo_a._func is entries['grouped_wo_a']._func
    @pl.jit
    def readback_grouped_wo_a(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[8, 1024, 4096], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 8192], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_grouped_wo_a(x, weight, output, active)
        return (x, weight, output)

    compute_index_projection = pl.jit.inline(entries['index_projection']._func)
    assert compute_index_projection._func is entries['index_projection']._func
    @pl.jit
    def readback_index_projection(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        query_latent: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        index_wq_b: pl.InOut[pl.Tensor[[1280, 4096], pl.INT8]],
        index_wq_b_scale: pl.InOut[pl.Tensor[[4096], pl.FP32]],
        weights_proj: pl.InOut[pl.Tensor[[5120, 32], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        query: pl.InOut[pl.Tensor[[16, 32, 128], pl.BF16]],
        weights: pl.InOut[pl.Tensor[[16, 32], pl.FP16]],
        projected: pl.InOut[pl.Tensor[[16, 4096], pl.BF16]],
        scaled_weights: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_index_projection(
            x, query_latent,
            index_wq_b, index_wq_b_scale, weights_proj,
            cosine, sine,
            query, weights, projected, scaled_weights,
            active,
        )
        return (
            x, query_latent,
            index_wq_b, index_wq_b_scale, weights_proj,
            cosine, sine,
            query, weights, projected, scaled_weights,
        )

    compute_index_quantize = pl.jit.inline(entries['index_quantize']._func)
    assert compute_index_quantize._func is entries['index_quantize']._func
    @pl.jit
    def readback_index_quantize(
        query: pl.InOut[pl.Tensor[[16, 32, 128], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[16, 32, 128], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[16, 32], pl.FP16]],
        raw_scale_f32: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        effective_scale_f32: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_index_quantize(query, payload, scales, raw_scale_f32, effective_scale_f32, active)
        return (query, payload, scales, raw_scale_f32, effective_scale_f32)

    compute_inverse_rope = pl.jit.inline(entries['inverse_rope']._func)
    assert compute_inverse_rope._func is entries['inverse_rope']._func
    @pl.jit
    def readback_inverse_rope(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_inverse_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_kv_norm = pl.jit.inline(entries['kv_norm']._func)
    assert compute_kv_norm._func is entries['kv_norm']._func
    @pl.jit
    def readback_kv_norm(
        x: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[512], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_kv_norm(x, weight, output, active)
        return (x, weight, output)

    compute_kv_rope = pl.jit.inline(entries['kv_rope']._func)
    assert compute_kv_rope._func is entries['kv_rope']._func
    @pl.jit
    def readback_kv_rope(
        x: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_kv_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_narrow_sparse_BF16 = pl.jit.inline(entries['narrow_sparse_BF16']._func)
    assert compute_narrow_sparse_BF16._func is entries['narrow_sparse_BF16']._func
    @pl.jit
    def readback_narrow_sparse_BF16(
        source: pl.InOut[pl.Tensor[[16, 32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_narrow_sparse_BF16(source, output, active)
        return (source, output)

    compute_q_norm = pl.jit.inline(entries['q_norm']._func)
    assert compute_q_norm._func is entries['q_norm']._func
    @pl.jit
    def readback_q_norm(
        x: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[1280], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_q_norm(x, weight, output, active)
        return (x, weight, output)

    compute_q_rope = pl.jit.inline(entries['q_rope']._func)
    assert compute_q_rope._func is entries['q_rope']._func
    @pl.jit
    def readback_q_rope(
        x: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        cosine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        sine: pl.InOut[pl.Tensor[[16, 32], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_q_rope(x, cosine, sine, output, active)
        return (x, cosine, sine, output)

    compute_sparse_controls = pl.jit.inline(entries['sparse_controls']._func)
    assert compute_sparse_controls._func is entries['sparse_controls']._func
    @pl.jit
    def readback_sparse_controls(
        raw_selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        positions: pl.InOut[pl.Tensor[[16], pl.INT64]],
        request_ids: pl.InOut[pl.Tensor[[16], pl.INT32]],
        long_block_table: pl.InOut[pl.Tensor[[1, 2], pl.INT32]],
        window_block_table: pl.InOut[pl.Tensor[[1, 2], pl.INT32]],
        selected: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        selected_lengths: pl.InOut[pl.Tensor[[16, 1], pl.INT32]],
        compressed_indices: pl.InOut[pl.Tensor[[16, 512], pl.INT32]],
        window_indices: pl.InOut[pl.Tensor[[16, 128], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_sparse_controls(
            raw_selected,
            positions, request_ids,
            long_block_table, window_block_table,
            selected, selected_lengths, compressed_indices, window_indices,
            active,
        )
        return (
            raw_selected,
            positions, request_ids,
            long_block_table, window_block_table,
            selected, selected_lengths, compressed_indices, window_indices,
        )

    compute_sparse_core = pl.jit.inline(entries['sparse_core']._func)
    assert compute_sparse_core._func is entries['sparse_core']._func
    @pl.jit
    def readback_sparse_core(
        query: pl.InOut[pl.Tensor[[active_rows, 64, 512], pl.FP32]],
        window: pl.InOut[pl.Tensor[[2, 128, 512], pl.FP32]],
        window_indices: pl.InOut[pl.Tensor[[active_rows, 128], pl.INT32]],
        compressed: pl.InOut[pl.Tensor[[1, 128, 512], pl.FP32]],
        compressed_indices: pl.InOut[pl.Tensor[[active_rows, 512], pl.INT32]],
        sink: pl.InOut[pl.Tensor[[64], pl.FP32]],
        extents: pl.InOut[pl.Tensor[[active_rows, 2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[active_rows, 64, 512], pl.FP32]],
    ):
        compute_sparse_core(query, window, window_indices, compressed, compressed_indices, sink, extents, output)
        return (query, window, window_indices, compressed, compressed_indices, sink, extents, output)

    compute_swa_publish = pl.jit.inline(entries['swa_publish']._func)
    assert compute_swa_publish._func is entries['swa_publish']._func
    @pl.jit
    def readback_swa_publish(
        value: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        slots: pl.InOut[pl.Tensor[[16], pl.INT64]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_swa_publish(value, slots, cache, active)
        return (value, slots, cache)

    compute_wkv = pl.jit.inline(entries['wkv']._func)
    assert compute_wkv._func is entries['wkv']._func
    @pl.jit
    def readback_wkv(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120, 512], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[512], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wkv(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    compute_wo_b = pl.jit.inline(entries['wo_b']._func)
    assert compute_wo_b._func is entries['wo_b']._func
    @pl.jit
    def readback_wo_b(
        x: pl.InOut[pl.Tensor[[16, 8192], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[8192, 5120], pl.BF16]],
        output: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wo_b(x, weight, output, active)
        return (x, weight, output)

    compute_wq_a = pl.jit.inline(entries['wq_a']._func)
    assert compute_wq_a._func is entries['wq_a']._func
    @pl.jit
    def readback_wq_a(
        x: pl.InOut[pl.Tensor[[16, 5120], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[5120, 1280], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[1280], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wq_a(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    compute_wq_b = pl.jit.inline(entries['wq_b']._func)
    assert compute_wq_b._func is entries['wq_b']._func
    @pl.jit
    def readback_wq_b(
        x: pl.InOut[pl.Tensor[[16, 1280], pl.BF16]],
        weight: pl.InOut[pl.Tensor[[1280, 32768], pl.INT8]],
        channel_scale: pl.InOut[pl.Tensor[[32768], pl.FP32]],
        output: pl.InOut[pl.Tensor[[16, 32768], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute_wq_b(x, weight, channel_scale, output, active)
        return (x, weight, channel_scale, output)

    return {
        'HC_post': readback_HC_post,
        'attention_norm': readback_attention_norm,
        'c2_project_pool_publish': readback_c2_project_pool_publish,
        'empty_selection': readback_empty_selection,
        'grouped_wo_a': readback_grouped_wo_a,
        'index_projection': readback_index_projection,
        'index_quantize': readback_index_quantize,
        'inverse_rope': readback_inverse_rope,
        'kv_norm': readback_kv_norm,
        'kv_rope': readback_kv_rope,
        'narrow_sparse_BF16': readback_narrow_sparse_BF16,
        'q_norm': readback_q_norm,
        'q_rope': readback_q_rope,
        'sparse_controls': readback_sparse_controls,
        'sparse_core': readback_sparse_core,
        'swa_publish': readback_swa_publish,
        'wkv': readback_wkv,
        'wo_b': readback_wo_b,
        'wq_a': readback_wq_a,
        'wq_b': readback_wq_b,
    }
