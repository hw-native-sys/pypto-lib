# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA front end: the q/k/v projections and every gate the delta rule consumes.

For each of the 34 linear-attention layers (64 heads x 128, so 4 heads per rank at
TP16):

* ``q_proj`` / ``k_proj`` / ``v_proj``: ``[D] -> [KDA_H * KDA_DIM]``, head-sharded.
* forget gate: ``g = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``
  with ``gate_lower_bound = -5.0``. ``f_a_proj`` is a rank-128 bottleneck and stays
  replicated; ``f_b_proj``, ``dt_bias`` and ``A_log`` are head-sharded.
* input gate: ``beta = sigmoid(b_proj(x))``, one scalar per head.
* output gate: ``g_b(g_a(x))``, the same 128-wide bottleneck shape as the forget gate.

The checkpoint keeps all of these BF16 — none of the KDA projections carry a
``weight_scale_inv``, and the vLLM Ascend port nulls the quant config for the whole
KDA block for the same reason.

There is no ``sigmoid`` tile op in the pypto DSL, so every gate here has to be
composed from ``pl.exp`` and ``pl.recip``.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KDA_DIM, KDA_GATE_LOWER_BOUND, LOCAL_KDA_H
from models.glm5_3_flash.config import LOCAL_KDA_QKV_DIM, T_DYN


def golden_kda_projection(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_f_a: torch.Tensor,
    w_f_b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_g_a: torch.Tensor,
    w_g_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("KDA projection golden is assigned with the kernel")


@pl.jit.inline
def kda_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, LOCAL_KDA_H], pl.FP32],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    raise NotImplementedError("KDA projection kernel body is assigned independently")


__all__ = [
    "KDA_GATE_LOWER_BOUND",
    "golden_kda_projection",
    "kda_projection",
]
