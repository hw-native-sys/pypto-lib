# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Gated DeltaNet block as one float64 chain, hidden states to hidden states.

The operator references and the delta rule compose the block output.

:func:`block` is two references in one. On bf16 weights with no activation
quantisation it is the float64 truth: how far a result is from the model. On
int8 weights (`weights.quantize_weights`) with `quant_x` / `quant_y` it is the
W8A8 chain the projection kernels implement, with their exact quantisation
arithmetic -- the correctness gate for those kernels, since quantisation alone
puts the block ~3e-2 from the truth, ten times the model's own bf16 rounding.
"""
from __future__ import annotations

import torch

from config import Qwen38Config
from gated_rmsnorm import gated_rmsnorm_ref
from gdn_layer import delta_rule_ref
from qk_norm_gate import qk_norm_gate_ref
from reference import dequantize, linear, quantize_rows, weight
from short_conv import short_conv_ref


def block(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen38Config,
          chunk: int, quant_x: bool = False, quant_y: bool = False
          ) -> dict[str, torch.Tensor]:
    """The whole block in float64, hidden states `[T, hidden]` to `out` of the same shape.

    Returns every intermediate: the projections `qkv`, `z`, `a_proj`, `b_proj`
    (`a` is the delta rule's key-key matrix, as everywhere else here); the conv
    output `qkv_conv`; the delta rule's inputs `q`, `k`, `v`, `beta`, `g` and
    its output `o` (plus every intermediate `gdn_layer.delta_rule_ref` keeps); the normed `y`;
    and `out`. Nothing is narrowed anywhere.

    What you pass is what you get. bf16 weights compute the truth; int8 ones
    (`weights.quantize_weights`) are used at their dequantised values. *quant_x*
    quantises the hidden states per token before every projection reads them,
    *quant_y* the normed output per token before `out_proj` -- the two
    activation quantisations a W8A8 block performs, at the places the kernels
    do them. The int8 tensors and fp32 scales come back as `x_q`, `x_scale`,
    `y_q`, `y_scale`, so a kernel can be checked at the quantised hand-off too.
    `y` is quantised from its fp32 rounding, as the norm kernel that produces
    it will hold it.
    """
    h, hg, d = (cfg.linear_num_value_heads, cfg.linear_num_key_heads,
                cfg.linear_value_head_dim)
    t = x.shape[0]
    key_width = hg * d

    st = {}
    xf = x
    if quant_x:
        st["x_q"], st["x_scale"] = quantize_rows(x)
        xf = dequantize(st["x_q"], st["x_scale"])
    st["qkv"] = linear(xf, weight(w, "in_proj_qkv"))
    st["z"] = linear(xf, weight(w, "in_proj_z")).reshape(t, h, d)
    st["a_proj"] = linear(xf, weight(w, "in_proj_a"))
    st["b_proj"] = linear(xf, weight(w, "in_proj_b"))
    st["qkv_conv"] = short_conv_ref(st["qkv"], w["conv1d.weight"])
    q, k, v = torch.split(st["qkv_conv"], [key_width, key_width, h * d], dim=-1)
    q, k, beta, g = qk_norm_gate_ref(q.reshape(t, hg, d), k.reshape(t, hg, d),
                                 st["a_proj"], st["b_proj"], w["A_log"], w["dt_bias"])
    st.update(delta_rule_ref(q, k, v.reshape(t, h, d), beta, g, chunk, narrow=False))
    st["y"] = gated_rmsnorm_ref(st["o"], st["z"], w["norm.weight"], cfg.rms_norm_eps)
    y = st["y"].reshape(t, h * d)
    if quant_y:
        st["y_q"], st["y_scale"] = quantize_rows(y)
        y = dequantize(st["y_q"], st["y_scale"])
    st["out"] = linear(y, weight(w, "out_proj"))
    return st
