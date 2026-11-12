# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech Prefill QKV publication and BF16 sparse output projection."""

import pypto.language as pl

from .c1c2 import make_compressed_cache_publish
from .ecotech_swa_prefill import GEOMETRY, make_builders
from .hc import make_hc_post_inline
from .decode_indexer import make_sparse_controls
from .rmsnorm import make_prefill_norm
from .sparse_consumer_prefill import make_consumer_sparse_core


# tiling
D_TILE = 256


def _inline(entry):
    result = pl.jit.inline(entry._func)
    if result._func is not entry._func or result._func_type != "inline":
        raise RuntimeError("Require the original compute function")
    return result


def make_qkv_publish(capacity, config):
    """Compose Eco-Tech projections, query/KV normalization and own SWA publication."""
    base = make_builders(capacity, config)
    project_qa = _inline(base["wq_a"])
    normalize_q = _inline(base["q_norm"])
    project_qb = _inline(base["wq_b"])
    rotate_q = _inline(base["q_rope"])
    project_kv = _inline(base["wkv"])
    normalize_kv = _inline(base["kv_norm"])
    rotate_kv = _inline(base["kv_rope"])
    publish = make_compressed_cache_publish(capacity, 2, 512)

    @pl.jit
    def qkv_publish(
        x: pl.Tensor[[capacity, 5120], pl.BF16],
        wq_a: pl.Tensor[[5120, 1280], pl.INT8],
        wq_a_scale: pl.Tensor[[1280], pl.FP32],
        q_norm: pl.Tensor[[1280], pl.BF16],
        wq_b: pl.Tensor[[1280, 32768], pl.INT8],
        wq_b_scale: pl.Tensor[[32768], pl.FP32],
        wkv: pl.Tensor[[5120, 512], pl.INT8],
        wkv_scale: pl.Tensor[[512], pl.FP32],
        kv_norm: pl.Tensor[[512], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        window_slots: pl.Tensor[[capacity], pl.INT64],
        query_projected: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        query_normalized: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, 32768], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
        swa_cache: pl.InOut[pl.Tensor[[2, 128, 1, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        project_qa(x, wq_a, wq_a_scale, query_projected, active)
        normalize_q(query_projected, q_norm, query_normalized, active)
        query_unrotated = pl.create_tensor([capacity, 32768], dtype=pl.BF16)
        project_qb(query_normalized, wq_b, wq_b_scale, query_unrotated, active)
        rotate_q(query_unrotated, cosine, sine, query, active)

        kv_projected = pl.create_tensor([capacity, 512], dtype=pl.BF16)
        project_kv(x, wkv, wkv_scale, kv_projected, active)
        kv_normalized = pl.create_tensor([capacity, 512], dtype=pl.BF16)
        normalize_kv(kv_projected, kv_norm, kv_normalized, active)
        rotate_kv(kv_normalized, cosine, sine, window_kv, active)
        publish(window_kv, window_slots, swa_cache, active)
        return query_projected, query_normalized, query, window_kv, swa_cache

    return qkv_publish


def make_sparse_projection(capacity, config):
    """Narrow sparse output and compose inverse RoPE, grouped WoA and BF16 WoB."""
    base = make_builders(capacity, config)
    inverse = _inline(base["inverse_rope"])
    project_a = _inline(base["grouped_wo_a"])
    project_b = _inline(base["wo_b"])

    @pl.jit
    def sparse_projection(
        attention: pl.Tensor[[capacity, 32768], pl.FP32],
        wo_a: pl.Tensor[[8, 1024, 4096], pl.BF16],
        wo_b: pl.Tensor[[8192, 5120], pl.BF16],
        cosine: pl.Tensor[[capacity, 32], pl.FP32],
        sine: pl.Tensor[[capacity, 32], pl.FP32],
        narrowed: pl.InOut[pl.Tensor[[capacity, 32768], pl.BF16]],
        unrotated: pl.InOut[pl.Tensor[[capacity, 32768], pl.BF16]],
        grouped: pl.InOut[pl.Tensor[[capacity, 8192], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (32768 // D_TILE), name_hint="ecotech_consumer_sparse_bf16"):
            token = block // (32768 // D_TILE)
            column = block % (32768 // D_TILE) * D_TILE
            values = attention[token : token + 1, column : column + D_TILE]
            narrowed[token : token + 1, column : column + D_TILE] = pl.cast(values, pl.BF16, mode="rint")
        inverse(narrowed, cosine, sine, unrotated, active)
        project_a(unrotated, wo_a, grouped, active)
        project_b(grouped, wo_b, output, active)
        return narrowed, unrotated, grouped, output

    return sparse_projection


def make_hc_post(capacity, config):
    """Expose the existing HC-post with explicit output readback."""
    compute = make_hc_post_inline(capacity, config)

    @pl.jit
    def hc_post(
        sublayer: pl.Tensor[[capacity, 5120], pl.BF16],
        residual: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        post_mix: pl.Tensor[[capacity, 4], pl.FP32],
        comb: pl.Tensor[[capacity, 4, 4], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, 4, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(sublayer, residual, post_mix, comb, output, active)
        return output

    return hc_post


def make_consumer_entries(active, *, compress_ratio, compressed_blocks=1):
    """Build bounded TP1 consumer entries with explicit shared-source controls."""
    if type(active) is not int or active not in (1, 7, 16):
        raise ValueError("The initial slice retains capacity16 and active1/7/16")
    if type(compress_ratio) is not int or compress_ratio not in (1, 2):
        raise ValueError("Retain the original source compression ratio")
    if type(compressed_blocks) is not int or compressed_blocks < 1:
        raise ValueError("Require the actual product cache block extent")
    config = dict(GEOMETRY)
    page_rows = 64 if compress_ratio == 2 else 128
    native_pages = compressed_blocks * 128 // page_rows
    return dict(
        attention_norm=make_prefill_norm(16, 5120),
        qkv_publish=make_qkv_publish(16, config),
        sparse_controls=make_sparse_controls(16, 1, native_pages, 2, page_rows, compress_ratio=compress_ratio),
        sparse_core=make_consumer_sparse_core(active, config, compressed_blocks=compressed_blocks),
        projection=make_sparse_projection(16, config),
        HC_post=make_hc_post(16, config),
    )
