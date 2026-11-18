# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Gated DeltaNet block as one float64 chain, hidden states to hidden states.

`reference.py` holds the block's pieces and `gdn_layer.delta_rule_ref` the six
stages between them; this strings them together, and :func:`block_inputs` hands
one tensor of the chain to a kernel's specs. It sits above `gdn_layer` so the
imports stay one-way: a block kernel draws from here, and nothing here knows
what the block kernels are.

:func:`block` is two references in one. On bf16 weights with no activation
quantisation it is the float64 truth: how far a result is from the model. On
int8 weights (`reference.quantize_weights`) with `quant_x` / `quant_y` it is the
W8A8 chain the projection kernels implement, with their exact quantisation
arithmetic -- the correctness gate for those kernels, since quantisation alone
puts the block ~3e-2 from the truth, ten times the model's own bf16 rounding.
"""
from __future__ import annotations

import torch

from config import Qwen38Config
from gdn_layer import delta_rule_ref
from reference import (dequantize, gated_rmsnorm, linear, make_block_inputs,
                       make_block_weights, qk_norm_gate, quantize_rows,
                       short_conv, weight)


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
    (`reference.quantize_weights`) are used at their dequantised values. *quant_x*
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
    st["qkv_conv"] = short_conv(st["qkv"], w["conv1d.weight"])
    q, k, v = torch.split(st["qkv_conv"], [key_width, key_width, h * d], dim=-1)
    q, k, beta, g = qk_norm_gate(q.reshape(t, hg, d), k.reshape(t, hg, d),
                                 st["a_proj"], st["b_proj"], w["A_log"], w["dt_bias"])
    st.update(delta_rule_ref(q, k, v.reshape(t, h, d), beta, g, chunk, narrow=False))
    st["y"] = gated_rmsnorm(st["o"], st["z"], w["norm.weight"], cfg.rms_norm_eps)
    y = st["y"].reshape(t, h * d)
    if quant_y:
        st["y_q"], st["y_scale"] = quantize_rows(y)
        y = dequantize(st["y_q"], st["y_scale"])
    st["out"] = linear(y, weight(w, "out_proj"))
    return st


_BLOCK_CACHE: dict[tuple, dict] = {}


def block_inputs(key: str, t: int, cfg: Qwen38Config, chunk: int,
                 transform=None, seed: int = 42, weights: str | None = None,
                 quant_x: bool = False, quant_y: bool = False):
    """A no-argument callable returning one tensor of the block chain, computed once.

    The chain runs on first use and is cached, so the several specs of a kernel
    that draw from it pay for it once. *weights* is a path to a layer
    `weights.py` wrote, or None for the random set.
    """
    def load():
        cache_key = (t, cfg.name, chunk, seed, weights, quant_x, quant_y)
        st = _BLOCK_CACHE.get(cache_key)
        if st is None:
            if weights is None:
                w = make_block_weights(cfg, seed)
            else:
                w = torch.load(weights, weights_only=True)
            st = _BLOCK_CACHE[cache_key] = block(
                make_block_inputs(t, cfg, seed), w, cfg, chunk,
                quant_x=quant_x, quant_y=quant_y)
        value = st[key]
        return transform(value) if transform is not None else value

    return load
