# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-width layer-zero A2 W8A8 attention with BF16 paged SWA."""

import pypto.language as pl

from .qkv_w8a8 import make_qkv_w8a8_inline
from .swa import make_swa_inline
from .o_proj import make_o_proj_inline


def make_attention_inline(capacity, config):
    """Build QKV, paged SWA, inverse RoPE and both output projections."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    qkv = make_qkv_w8a8_inline(capacity, config)
    swa = make_swa_inline(capacity, 2, config)
    project = make_o_proj_inline(capacity, config)

    @pl.jit.inline
    def attention_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        cache: pl.Tensor[[2, 128, 1, head_dim], pl.BF16],
        cache_lens: pl.Tensor[[2], pl.INT32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_cache = pl.reshape(cache, [2, 128, head_dim])
        qkv(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, rope_cos, rope_sin,
            positions, query_latent, query, window_kv, flat_cache, cache_lens, active,
        )
        attended = pl.create_tensor([capacity, width], dtype=pl.BF16)
        swa(query, window_kv, slots, indices, sink, cache, attended, active)
        project(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output, active)
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention_inline


def make_attention(capacity, config):
    """Build the standalone JIT wrapper for shared inline attention."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    heads = config["num_attention_heads"]
    width = heads * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    window = config["sliding_window"]
    groups = config["o_groups"]
    group_input = width // groups
    rank = config["o_lora_rank"]
    compute = make_attention_inline(capacity, config)

    @pl.jit
    def attention(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        compute(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, wo_a, wo_b, wo_b_scale,
            rope_cos, rope_sin, slots, indices, sink,
            positions, query_latent, query, window_kv, cache, cache_lens, output, active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens, output

    return attention
