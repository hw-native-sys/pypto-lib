# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP HC head scaffold."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


B = DECODE_BATCH
S = DECODE_SEQ
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps

@pl.jit
def mtp_hc_head(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    # TODO: kernel implementation
    return y


def golden_mtp_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + EPS)
    mixes = torch.nn.functional.linear(x_flat, tensors["hc_head_fn"].float()) * rsqrt
    pre = torch.sigmoid(mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()) + HC_EPS
    tensors["y"][:] = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2).to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=lambda: torch.randn(B, S, HC_MULT, D)),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=lambda: torch.randn(HC_MULT, HC_DIM) / HC_DIM ** 0.5),
        TensorSpec("hc_head_scale", [1], torch.float32, init_value=lambda: torch.ones(1) * 0.5),
        TensorSpec("hc_head_base", [HC_MULT], torch.float32, init_value=lambda: torch.zeros(HC_MULT)),
        TensorSpec("y", [B, S, D], torch.bfloat16, is_output=True),
    ]
