# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card A2 FFN mHC-pre/RMSNorm and mHC-post boundaries."""

import pypto.language as pl

from .hc import make_hc_post_inline, make_hc_pre_inline
from .rmsnorm import NORM_T_TILE, rms_norm_tiled
from .ops import make_quantize


def make_moe_input_quantize(capacity, hidden):
    """Build dynamic INT8 quantization before host expert dispatch."""
    quantize = make_quantize(capacity, hidden)

    @pl.jit
    def moe_input_quantize(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, hidden], pl.INT8]],
        scale: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(x, payload, scale, active)
        return payload, scale

    return moe_input_quantize


def make_ffn_pre(capacity, config):
    """Build FFN mHC-pre + RMSNorm for a fixed single-card capacity."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_hc_pre_inline(capacity, config)

    @pl.jit
    def ffn_pre(
        x_hc: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        norm_weight: pl.Tensor[[hidden], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x_hc, function, scale, base, incoming, next_pre, post, comb,
            collapsed, collapsed_fp32, active,
        )
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="a2_ffn_norm"):
            rms_norm_tiled(collapsed, norm_weight, normed, block, active)
        return next_pre, post, comb, collapsed, normed

    return ffn_pre


def make_ffn_post(capacity, config):
    """Build FFN mHC-post residual expansion for a fixed single-card capacity."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    post = make_hc_post_inline(capacity, config)

    @pl.jit
    def ffn_post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        post(sublayer, residual, post_mix, comb, output, active)
        return output

    return ffn_post
