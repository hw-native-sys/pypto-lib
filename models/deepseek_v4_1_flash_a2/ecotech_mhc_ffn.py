# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Attention mHC merge, delayed FFN collapse, RMSNorm and shared W8A8 MLP."""

import pypto.language as pl

from .ecotech_shared import make_ecotech_shared_inline
from .hc import make_hc_post_inline, make_hc_pre_inline
from .prefill_norm import make_prefill_norm


def make_ecotech_mhc_ffn(capacity, config, intermediate=2304):
    """Compose a supplied attention sublayer with the local shared FFN branch."""
    expected = dict(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
    if capacity <= 0 or capacity % 16 or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Require capacity/16 and the original four-stream source attributes")
    hc_pre = make_hc_pre_inline(capacity, config)
    hc_post = make_hc_post_inline(capacity, config)
    standalone_norm = make_prefill_norm(capacity, 5120)
    norm = pl.jit.inline(standalone_norm._func)
    if norm._func is not standalone_norm._func or norm._func_type != "inline":
        raise RuntimeError("The composition must retain the existing formal RMSNorm function")
    shared = make_ecotech_shared_inline(capacity, 5120, intermediate)
    width = intermediate * 2
    padded_intermediate = ((intermediate + 255) // 256) * 256

    @pl.jit
    def mhc_ffn(
        x: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        incoming: pl.Tensor[[capacity, 4], pl.FP32],
        attention_sublayer: pl.Tensor[[capacity, 5120], pl.BF16],
        attn_function: pl.Tensor[[24, 20480], pl.FP32],
        attn_scale: pl.Tensor[[3], pl.FP32],
        attn_base: pl.Tensor[[24], pl.FP32],
        ffn_function: pl.Tensor[[24, 20480], pl.FP32],
        ffn_scale: pl.Tensor[[3], pl.FP32],
        ffn_base: pl.Tensor[[24], pl.FP32],
        norm_weight: pl.Tensor[[5120], pl.BF16],
        gate_weight: pl.Tensor[[5120, width], pl.INT8],
        gate_scale: pl.Tensor[[width], pl.FP32],
        down_weight: pl.Tensor[[padded_intermediate, 5120], pl.INT8],
        down_scale: pl.Tensor[[5120], pl.FP32],
        attn_pre: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        attn_post: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        attn_comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        attn_collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        changed_hidden: pl.InOut[pl.Tensor[[capacity, 4, 5120], pl.BF16]],
        ffn_pre: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        ffn_post: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        ffn_comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        ffn_collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        shared_output: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        hc_pre(x, attn_function, attn_scale, attn_base, incoming, attn_pre, attn_post, attn_comb, attn_collapsed, active)
        hc_post(attention_sublayer, x, attn_post, attn_comb, changed_hidden, active)
        hc_pre(
            changed_hidden, ffn_function, ffn_scale, ffn_base,
            attn_pre, ffn_pre, ffn_post, ffn_comb, ffn_collapsed,
            active,
        )
        norm(ffn_collapsed, norm_weight, normed, active)
        shared_result, input_payload, input_scale, accumulator, activation, down_payload, down_token_scale = shared(
            normed, gate_weight, gate_scale, down_weight, down_scale,
            shared_output,
            active,
        )
        return (
            attn_pre, attn_post, attn_comb, attn_collapsed, changed_hidden,
            ffn_pre, ffn_post, ffn_comb, ffn_collapsed, normed, shared_result,
            input_payload, input_scale, accumulator, activation, down_payload, down_token_scale,
        )

    return mhc_ffn
