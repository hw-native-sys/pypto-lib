# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 layer2 prefill QKV/C2 producer with separate native source-position RoPE.

The compressed caches use a lossless physical-row view: native [2,64,1,D]
becomes [1,128,1,D]. Flattened slots retain page*64+offset addresses. Sparse
selection and attention consume these actual outputs in subsequent stages.
"""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .layer2_c1c2 import HEAD_DIM, make_c2_index_publish_projected_inline, make_fp32_projection
from .compressor_core import RING_ROWS_DYN
from .operator_config import B_DYN, Q_START_DYN, STATE_BLOCKS_DYN
from .qkv_w8a8 import make_qkv_w8a8_inline


def make_layer2_prefill_producer(capacity, config, preprojected=False):
    """Produce QKV and caches from inline or external C2 projections."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    width = config["num_attention_heads"] * config["head_dim"]
    pairs = config["qk_rope_head_dim"] // 2
    qkv = make_qkv_w8a8_inline(capacity, config)
    c2 = make_c2_index_publish_projected_inline(capacity, config, cache_blocks=1)
    publish_window = make_compressed_cache_publish(capacity, 2, 512)

    project_kv = make_fp32_projection(hidden, HEAD_DIM)

    @pl.jit.inline
    def producer_body(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, 512], pl.INT8],
        wkv_scale: pl.Tensor[[512], pl.FP32],
        kv_norm: pl.Tensor[[512], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        positions32: pl.Tensor[[capacity], pl.INT32],
        window_slots: pl.Tensor[[capacity], pl.INT64],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        compressor_norm: pl.Tensor[[512], pl.BF16],
        index_wk: pl.Tensor[[512, 128], pl.BF16],
        index_norm: pl.Tensor[[128], pl.BF16],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, width], pl.BF16],
        window_kv: pl.Tensor[[capacity, 512], pl.BF16],
        swa_cache: pl.Tensor[[2, 128, 1, 512], pl.BF16],
        state: pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 1024], pl.FP32],
        compressed: pl.Tensor[[capacity, 512], pl.BF16],
        long_cache: pl.Tensor[[1, 128, 1, 512], pl.BF16],
        index_key: pl.Tensor[[capacity, 128], pl.BF16],
        key_cache: pl.Tensor[[1, 128, 1, 128], pl.INT8],
        key_scale: pl.Tensor[[1, 128, 1, 1], pl.FP16],
        source_positions: pl.Tensor[[capacity], pl.INT64],
        cache_lengths: pl.Tensor[[B_DYN, 2], pl.INT32],
        kv_proj: pl.Tensor[[capacity, 512], pl.FP32],
        score_proj: pl.Tensor[[capacity, 512], pl.FP32],
        active: pl.Scalar[pl.INT32],
        projections_ready: pl.Scalar[pl.TASK_ID],
    ):
        flat_window = pl.reshape(swa_cache, [2, 128, 512])
        legacy_lengths = pl.create_tensor([2], dtype=pl.INT32)
        qkv(
            x, wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm,
            rope_cos, rope_sin, positions, query_latent, query, window_kv,
            flat_window, legacy_lengths, active,
        )
        publish_window(window_kv, window_slots, swa_cache, active)
        c2(
            kv_proj, score_proj, compressor_norm, index_wk, index_norm,
            source_cos, source_sin, positions32, compressed_slots, query_start, requests,
            state_block_table, state, long_cache, key_cache, key_scale, compressed, index_key,
            active, projections_ready,
        )
        with pl.spmd(1, name_hint="a2_prefill_lengths_and_sources"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                position = pl.read(positions, [token])
                source = pl.cast(0, pl.INT64)
                if position % 2 == 1:
                    source = position - 1
                pl.write(source_positions, [token], source)
                request = pl.read(requests, [token])
                length = pl.cast(position + 1, pl.INT32)
                pl.write(cache_lengths, [request, 0], length)
                pl.write(cache_lengths, [request, 1], pl.cast(length // 2, pl.INT32))
        return (
            positions, query_latent, query, window_kv, swa_cache, state, compressed,
            long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths,
        )

    if preprojected:
        @pl.jit
        def producer_projected(
            x: pl.Tensor[[capacity, hidden], pl.BF16],
            wq_a: pl.Tensor[[hidden, latent], pl.INT8],
            wq_a_scale: pl.Tensor[[latent], pl.FP32],
            q_norm: pl.Tensor[[latent], pl.BF16],
            wq_b: pl.Tensor[[latent, width], pl.INT8],
            wq_b_scale: pl.Tensor[[width], pl.FP32],
            wkv: pl.Tensor[[hidden, 512], pl.INT8],
            wkv_scale: pl.Tensor[[512], pl.FP32],
            kv_norm: pl.Tensor[[512], pl.BF16],
            rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
            rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
            source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
            source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
            positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
            positions32: pl.Tensor[[capacity], pl.INT32],
            window_slots: pl.Tensor[[capacity], pl.INT64],
            compressed_slots: pl.Tensor[[capacity], pl.INT64],
            query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
            requests: pl.Tensor[[capacity], pl.INT32],
            state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
            compressor_wkv: pl.Tensor[[hidden, 512], pl.BF16],
            compressor_wgate: pl.Tensor[[hidden, 512], pl.BF16],
            compressor_norm: pl.Tensor[[512], pl.BF16],
            index_wk: pl.Tensor[[512, 128], pl.BF16],
            index_norm: pl.Tensor[[128], pl.BF16],
            query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
            query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
            window_kv: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
            swa_cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
            state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 1024], pl.FP32]],
            compressed: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
            long_cache: pl.InOut[pl.Tensor[[1, 128, 1, 512], pl.BF16]],
            index_key: pl.InOut[pl.Tensor[[capacity, 128], pl.BF16]],
            key_cache: pl.InOut[pl.Tensor[[1, 128, 1, 128], pl.INT8]],
            key_scale: pl.InOut[pl.Tensor[[1, 128, 1, 1], pl.FP16]],
            source_positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
            cache_lengths: pl.InOut[pl.Tensor[[B_DYN, 2], pl.INT32]],
            kv_proj: pl.Tensor[[capacity, 512], pl.FP32],
            score_proj: pl.Tensor[[capacity, 512], pl.FP32],
            active: pl.Scalar[pl.INT32],
        ):
            copied_kv = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
            copied_scores = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
            with pl.spmd(pl.max(active, 1), name_hint="a2_c2_projected_inputs") as projections_ready:
                row = pl.tile.get_block_idx()
                if row < active:
                    copied_kv[row:row + 1, :] = kv_proj[row:row + 1, :]
                    copied_scores[row:row + 1, :] = score_proj[row:row + 1, :]
            producer_body(
                x,
                wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm,
                rope_cos, rope_sin, source_cos, source_sin,
                positions, positions32, window_slots, compressed_slots,
                query_start, requests, state_block_table,
                compressor_norm, index_wk, index_norm,
                query_latent, query, window_kv,
                swa_cache, state, compressed, long_cache,
                index_key, key_cache, key_scale,
                source_positions, cache_lengths,
                copied_kv, copied_scores,
                active, projections_ready,
            )
            return (
                positions, query_latent, query, window_kv, swa_cache, state, compressed,
                long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths,
            )

        return producer_projected

    @pl.jit
    def producer(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, 512], pl.INT8],
        wkv_scale: pl.Tensor[[512], pl.FP32],
        kv_norm: pl.Tensor[[512], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        source_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        source_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        positions32: pl.Tensor[[capacity], pl.INT32],
        window_slots: pl.Tensor[[capacity], pl.INT64],
        compressed_slots: pl.Tensor[[capacity], pl.INT64],
        query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
        requests: pl.Tensor[[capacity], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        compressor_wkv: pl.Tensor[[hidden, 512], pl.BF16],
        compressor_wgate: pl.Tensor[[hidden, 512], pl.BF16],
        compressor_norm: pl.Tensor[[512], pl.BF16],
        index_wk: pl.Tensor[[512, 128], pl.BF16],
        index_norm: pl.Tensor[[128], pl.BF16],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
        swa_cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
        state: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, RING_ROWS_DYN, 1024], pl.FP32]],
        compressed: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
        long_cache: pl.InOut[pl.Tensor[[1, 128, 1, 512], pl.BF16]],
        index_key: pl.InOut[pl.Tensor[[capacity, 128], pl.BF16]],
        key_cache: pl.InOut[pl.Tensor[[1, 128, 1, 128], pl.INT8]],
        key_scale: pl.InOut[pl.Tensor[[1, 128, 1, 1], pl.FP16]],
        source_positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        cache_lengths: pl.InOut[pl.Tensor[[B_DYN, 2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        kv_proj = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
        score_proj = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.FP32)
        project_kv(x, compressor_wkv, kv_proj, active)
        score_ready = project_kv(x, compressor_wgate, score_proj, active)
        producer_body(
            x,
            wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm,
            rope_cos, rope_sin, source_cos, source_sin,
            positions, positions32, window_slots, compressed_slots,
            query_start, requests, state_block_table,
            compressor_norm, index_wk, index_norm,
            query_latent, query, window_kv,
            swa_cache, state, compressed, long_cache,
            index_key, key_cache, key_scale,
            source_positions, cache_lengths,
            kv_proj, score_proj,
            active, score_ready,
        )
        return (
            positions, query_latent, query, window_kv, swa_cache, state, compressed,
            long_cache, index_key, key_cache, key_scale, source_positions, cache_lengths,
        )

    return producer
