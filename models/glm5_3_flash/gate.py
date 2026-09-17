# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sigmoid ``noaux_tc`` router over 288 experts, fused with the deferred norm and quant.

``scores = sigmoid(x @ gate_weight.T)`` in FP32; selection ranks
``scores + e_score_correction_bias`` and takes the top 8; the returned weights are
the *unbiased* scores, renormalised (``norm_topk_prob``) and scaled by
``routed_scaling_factor = 2.5``. ``n_group`` and ``topk_group`` are both 1, so the
grouped masking of ``DeepseekV3TopkRouter`` degenerates to a plain top-k.

``mlp.gate.weight`` is stored BF16 ``[288, 4096]`` and ``e_score_correction_bias``
is FP32 ``[288]``; ``moe_router_dtype`` is ``float32``, so the matmul accumulates
and everything downstream of it stays in FP32.

Like the a2a3 sibling's ``models/deepseek_v4_flash_mtp/gate.py``, this kernel also
owns the deferred RMSNorm and the per-token INT8 quantization, so the INT8 view is
produced once and reused by both the shared expert and the EP dispatch payload.

**Known blocker for the assignee.** That sibling pads the expert row to
``SCORE_PAD = 256`` and sorts with ``pl.sort32`` followed by two ``pl.mrgsort``
stages (``[1,256] -> [1,512]`` in 8 runs of 64, merge at 64, then merge the two
256 halves). GLM-5.3-Flash has **288** experts, which does not fit that pad: the
pad has to grow to 512 and the merge tree gains a level.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, N_EXPERTS, T_DYN, TOPK
from models.glm5_3_flash.golden import gate, rms_norm


SCORE_PAD = 512  # 288 experts padded to the next sort32-friendly width


def golden_gate(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gate(rms_norm(x, norm_weight), gate_weight, correction_bias)


@pl.jit.inline
def moe_gate(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    route_weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
    route_indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("MoE gate kernel body is assigned independently")


__all__ = [
    "SCORE_PAD",
    "golden_gate",
    "moe_gate",
]
