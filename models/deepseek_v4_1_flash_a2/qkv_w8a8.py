# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 W8A8 QKV entry point for the independently frozen V4.1 module contract."""

import pypto.language as pl

from .ops import N_TILE, make_quantize, make_pack_weight, make_dot, make_dequantize, make_norm, make_rope


def make_qkv_w8a8_inline(capacity, config):
    """Build shared inline QKV compute with canonical [K,N] INT8 weights."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    query_width = heads * head_dim
    pair_width = rope_dim // 2
    eps = config["rms_norm_eps"]
    qa_packed_rows = hidden * (latent // N_TILE)
    qb_packed_rows = latent * (query_width // N_TILE)
    kv_packed_rows = hidden * (head_dim // N_TILE)

    quant_x = make_quantize(capacity, hidden)
    quant_q = make_quantize(capacity, latent)
    pack_qa = make_pack_weight(hidden, latent)
    pack_qb = make_pack_weight(latent, query_width)
    pack_kv = make_pack_weight(hidden, head_dim)
    dot_qa = make_dot(capacity, hidden, latent)
    dot_qb = make_dot(capacity, latent, query_width)
    dot_kv = make_dot(capacity, hidden, head_dim)
    dequant_qa = make_dequantize(capacity, latent)
    dequant_qb = make_dequantize(capacity, query_width, combine_scales=True)
    dequant_kv = make_dequantize(capacity, head_dim)
    norm_q = make_norm(capacity, latent, eps)
    norm_kv = make_norm(capacity, head_dim, eps)
    rope_q = make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, query_width], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pair_width], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pair_width], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        query_latent: pl.Tensor[[capacity, latent], pl.BF16],
        query: pl.Tensor[[capacity, query_width], pl.BF16],
        window_kv: pl.Tensor[[capacity, head_dim], pl.BF16],
        cache: pl.Tensor[[2, 128, head_dim], pl.BF16],
        cache_lens: pl.Tensor[[2], pl.INT32],
        active: pl.Scalar[pl.INT32],
    ):
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_x(x, x_i8, x_scale, active)

        qa_packed = pl.create_tensor([qa_packed_rows, N_TILE], dtype=pl.INT8)
        pack_qa(wq_a, qa_packed)
        qa_int = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, qa_packed, qa_int, active)
        qa_bf16 = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        dequant_qa(qa_int, wq_a_scale, x_scale, qa_bf16, active)
        norm_q(qa_bf16, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_packed = pl.create_tensor([qb_packed_rows, N_TILE], dtype=pl.INT8)
        pack_qb(wq_b, qb_packed)
        qb_int = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, qb_packed, qb_int, active)
        qb_bf16 = pl.create_tensor([capacity, query_width], dtype=pl.BF16)
        dequant_qb(qb_int, wq_b_scale, q_scale, qb_bf16, active)
        rope_q(qb_bf16, rope_cos, rope_sin, query, active)

        kv_packed = pl.create_tensor([kv_packed_rows, N_TILE], dtype=pl.INT8)
        pack_kv(wkv, kv_packed)
        kv_int = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, kv_packed, kv_int, active)
        kv_bf16 = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        dequant_kv(kv_int, wkv_scale, x_scale, kv_bf16, active)
        kv_normalized = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        norm_kv(kv_bf16, kv_norm_weight, kv_normalized, active)
        rope_kv(kv_normalized, rope_cos, rope_sin, window_kv, active)
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8_inline


def make_qkv_w8a8(capacity, config):
    """Build the standalone JIT wrapper for shared inline QKV compute."""
    hidden = config["hidden_size"]
    latent = config["q_lora_rank"]
    head_dim = config["head_dim"]
    width = config["num_attention_heads"] * head_dim
    pairs = config["qk_rope_head_dim"] // 2
    compute = make_qkv_w8a8_inline(capacity, config)

    @pl.jit
    def qkv_w8a8(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        rope_cos: pl.Tensor[[capacity, pairs], pl.FP32],
        rope_sin: pl.Tensor[[capacity, pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
            kv_norm_weight, rope_cos, rope_sin,
            positions, query_latent, query, window_kv, cache, cache_lens, active,
        )
        return positions, query_latent, query, window_kv, cache, cache_lens

    return qkv_w8a8
