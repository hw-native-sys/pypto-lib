# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Own QKV/SWA, shared sparse Attention, and grouped output projection."""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .o_proj import make_o_proj_inline
from .ops import N_TILE
from .qkv_w8a8 import make_qkv_w8a8_inline
from .sparse_attention import attend_bf16


def make_consumer_qkv_publish(capacity, config):
    """Compose own QKV/RoPE and own SWA publication without a compressor."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    width = config["num_attention_heads"] * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    qkv = make_qkv_w8a8_inline(capacity, config)
    publish = make_compressed_cache_publish(capacity, 2, head_dim)

    @pl.jit
    def consumer_qkv_publish(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        window_slots: pl.Tensor[[capacity], pl.INT64],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        swa_cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        flat_window = pl.reshape(swa_cache, [2, 128, head_dim])
        unused_legacy_lengths = pl.create_tensor([2], dtype=pl.INT32)
        qkv(
            x, wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm,
            rope_cos, rope_sin, positions, query_latent, query, window_kv,
            flat_window, unused_legacy_lengths, active,
        )
        publish(window_kv, window_slots, swa_cache, active)
        return positions, query_latent, query, window_kv, swa_cache

    return consumer_qkv_publish


def make_consumer_sparse_core(active, config, *, compressed_blocks=1):
    """Expose exact readonly device staging around the existing sparse core."""
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    if compressed_blocks < 1:
        raise ValueError("Compressed cache staging requires its actual positive block count")

    @pl.jit(auto_scope=False)
    def consumer_sparse_core(
        query: pl.Tensor[[active, heads, head_dim], pl.FP32],
        window: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.FP32]],
        window_indices: pl.Tensor[[active, 128], pl.INT32],
        compressed: pl.InOut[pl.Tensor[[compressed_blocks, 128, head_dim], pl.FP32]],
        compressed_indices: pl.Tensor[[active, 512], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        extents: pl.Tensor[[active, 2], pl.INT32],
        output: pl.Out[pl.Tensor[[active, heads, head_dim], pl.FP32]],
    ):
        attend_bf16(query, window, window_indices, compressed, compressed_indices, sink, extents, output)
        return window, compressed, output

    return consumer_sparse_core


def make_consumer_projection(capacity, config):
    """Narrow sparse output, then reuse inverse RoPE and grouped BF16/W8A8."""
    hidden = config["hidden_size"]
    width = config["num_attention_heads"] * config["head_dim"]
    pairs = config["qk_rope_head_dim"] // 2
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    project = make_o_proj_inline(capacity, config)

    @pl.jit
    def consumer_projection(
        attention: pl.Tensor[[capacity, width], pl.FP32],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        narrowed_attention = pl.create_tensor([capacity, width], dtype=pl.BF16)
        for narrow_block in pl.spmd(active * (width // N_TILE), name_hint="consumer_attention_bf16"):
            narrow_token = narrow_block // (width // N_TILE)
            narrow_column = narrow_block % (width // N_TILE) * N_TILE
            narrow_source = attention[narrow_token:narrow_token + 1, narrow_column:narrow_column + N_TILE]
            narrow_result = pl.cast(narrow_source, pl.BF16, mode="rint")
            narrowed_attention[narrow_token:narrow_token + 1, narrow_column:narrow_column + N_TILE] = narrow_result
        project(narrowed_attention, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output, active)
        return output

    return consumer_projection
