# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP input projection scaffold.

Mirrors the MTP-only prolog in the official implementation:
``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


B = DECODE_BATCH
S = DECODE_SEQ
D = M.hidden_size
EPS = M.rms_norm_eps

@pl.jit
def mtp_projection(
    hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[B, S, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.BF16],
    h_proj_w: pl.Tensor[[D, D], pl.BF16],
    hidden_states_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    # TODO: kernel implementation
    return hidden_states_out


def _rms_norm(x, weight):
    import torch

    return x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS) * weight.float()


def golden_mtp_projection(tensors):
    import torch

    hidden_states = _rms_norm(tensors["hidden_states"], tensors["enorm_w"]).to(torch.bfloat16)
    prev_hidden_states = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"]).to(torch.bfloat16)
    hidden_e = hidden_states.float().matmul(tensors["e_proj_w"].float().t())
    hidden_h = prev_hidden_states.float().matmul(tensors["h_proj_w"].float().t())
    tensors["hidden_states_out"][:] = (hidden_e + hidden_h).to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("hidden_states", [B, S, D], torch.bfloat16, init_value=lambda: torch.randn(B, S, D)),
        TensorSpec("prev_hidden_states", [B, S, D], torch.bfloat16, init_value=lambda: torch.randn(B, S, D)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec(
            "e_proj_w", [D, D], torch.bfloat16,
            init_value=lambda: (torch.randn(D, D) / D ** 0.5).to(torch.bfloat16),
        ),
        TensorSpec(
            "h_proj_w", [D, D], torch.bfloat16,
            init_value=lambda: (torch.randn(D, D) / D ** 0.5).to(torch.bfloat16),
        ),
        TensorSpec("hidden_states_out", [B, S, D], torch.bfloat16, is_output=True),
    ]
