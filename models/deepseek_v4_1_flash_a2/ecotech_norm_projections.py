# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exported-basis input RMSNorm, WQA/query RMSNorm/index query and WKV."""

import pypto.language as pl

from .ecotech_linear import make_ecotech_dynamic_linear
from .ops import make_norm
from .rmsnorm import D, EPS, NORM_T_TILE, rms_norm_tiled


def _inline_linear(capacity, input_width, output_width):
    standalone = make_ecotech_dynamic_linear(capacity, input_width, output_width)
    inline = pl.jit.inline(standalone._func)
    if inline._func is not standalone._func or inline._func_type != "inline":
        raise RuntimeError("The projection must retain the existing formal function")
    return inline


def make_ecotech_norm_projections(capacity, *, eps=1e-20):
    """Compose the normalized WQA/indexer query and pre-kv_norm WKV outputs."""
    if capacity <= 0 or capacity % 16 or D != 5120 or EPS != eps or eps != 1e-20:
        raise ValueError("Require capacity/16, original hidden5120 and source epsilon1e-20")
    wqa = _inline_linear(capacity, 5120, 1280)
    indexer = _inline_linear(capacity, 1280, 4096)
    wkv = _inline_linear(capacity, 5120, 512)
    query_norm = make_norm(capacity, 1280, eps)

    @pl.jit
    def projections(
        x: pl.Tensor[[capacity, 5120], pl.BF16],
        attn_norm_weight: pl.Tensor[[5120], pl.BF16],
        query_norm_weight: pl.Tensor[[1280], pl.BF16],
        wqa_weight: pl.Tensor[[5120, 1280], pl.INT8],
        wqa_scale: pl.Tensor[[1280], pl.FP32],
        index_weight: pl.Tensor[[1280, 4096], pl.INT8],
        index_scale: pl.Tensor[[4096], pl.FP32],
        wkv_weight: pl.Tensor[[5120, 512], pl.INT8],
        wkv_scale: pl.Tensor[[512], pl.FP32],
        input_normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        query_latent: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        query_normed: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        index_query: pl.InOut[pl.Tensor[[capacity, 4096], pl.BF16]],
        window_latent: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="ecotech_input_norm"):
            rms_norm_tiled(x, attn_norm_weight, input_normed, block, active)
        wqa(input_normed, wqa_weight, wqa_scale, query_latent, active)
        query_norm(query_latent, query_norm_weight, query_normed, active)
        indexer(query_normed, index_weight, index_scale, index_query, active)
        wkv(input_normed, wkv_weight, wkv_scale, window_latent, active)
        return input_normed, query_latent, query_normed, index_query, window_latent

    return projections
