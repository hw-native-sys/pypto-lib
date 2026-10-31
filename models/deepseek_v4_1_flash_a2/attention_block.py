# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Delayed HC, RMSNorm, A2 W8A8 attention and BF16 residual expansion."""

import pypto.language as pl

from .attention import make_attention_inline
from .hc import make_hc_pre_inline, make_hc_post_inline
from .rmsnorm import D, EPS, NORM_T_TILE, rms_norm_tiled


def make_attention_block(capacity, config):
    """Build the layer-zero attention block with delayed HC coefficients."""
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
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    if hidden != D or config["rms_norm_eps"] != EPS:
        raise ValueError("The shared model RMSNorm configuration must match this block")
    pre = make_hc_pre_inline(capacity, config)
    attention = make_attention_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def attention_block(
        x_hc: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        hc_function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
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
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        hidden_output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):

        pre(x_hc, hc_function, hc_scale, hc_base, incoming, next_pre, post, comb, collapsed, active)
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_block_norm"):
            rms_norm_tiled(collapsed, norm_weight, normed, block, active)
        attention(
            normed, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, wo_a, wo_b, wo_b_scale,
            rope_cos, rope_sin, slots, indices, sink,
            positions, query_latent, query, window_kv, cache, cache_lens, output, active,
        )
        post_fn(output, x_hc, post, comb, hidden_output, active)
        return (
            positions, query_latent, query, window_kv, cache, cache_lens, output,
            next_pre, post, comb, collapsed, normed, hidden_output,
        )

    return attention_block
