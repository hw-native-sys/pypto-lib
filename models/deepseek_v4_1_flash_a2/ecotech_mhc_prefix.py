# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four-stream mHC pre, delayed collapse, RMSNorm and W8A8 projections."""

import pypto.language as pl

from .ecotech_norm_projections import make_ecotech_norm_projections
from .hc import make_hc_pre_inline


def make_ecotech_mhc_prefix(capacity, config):
    """Compose explicit incoming-premix mHC pre with the normalized projections."""
    expected = dict(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    if capacity <= 0 or capacity % 16 or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Require capacity/16 and original four-stream L2 source attributes")
    hc_pre = make_hc_pre_inline(capacity, config)
    standalone = make_ecotech_norm_projections(capacity, eps=config["rms_norm_eps"])
    project = pl.jit.inline(standalone._func)
    if project._func is not standalone._func or project._func_type != "inline":
        raise RuntimeError("The composition must retain the existing formal projection function")

    @pl.jit
    def prefix(
        x: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        incoming: pl.Tensor[[capacity, 4], pl.FP32],
        hc_function: pl.Tensor[[24, 20480], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        attn_norm_weight: pl.Tensor[[5120], pl.BF16],
        query_norm_weight: pl.Tensor[[1280], pl.BF16],
        wqa_weight: pl.Tensor[[5120, 1280], pl.INT8],
        wqa_scale: pl.Tensor[[1280], pl.FP32],
        index_weight: pl.Tensor[[1280, 4096], pl.INT8],
        index_scale: pl.Tensor[[4096], pl.FP32],
        wkv_weight: pl.Tensor[[5120, 512], pl.INT8],
        wkv_scale: pl.Tensor[[512], pl.FP32],
        next_pre: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        input_normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        query_latent: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        query_normed: pl.InOut[pl.Tensor[[capacity, 1280], pl.BF16]],
        index_query: pl.InOut[pl.Tensor[[capacity, 4096], pl.BF16]],
        window_latent: pl.InOut[pl.Tensor[[capacity, 512], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        hc_pre(x, hc_function, hc_scale, hc_base, incoming, next_pre, post, comb, collapsed, active)
        project(
            collapsed, attn_norm_weight, query_norm_weight,
            wqa_weight, wqa_scale, index_weight, index_scale, wkv_weight, wkv_scale,
            input_normed, query_latent, query_normed, index_query, window_latent,
            active,
        )
        return next_pre, post, comb, collapsed, input_normed, query_latent, query_normed, index_query, window_latent

    return prefix
