# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single shared expert, evaluated for every token on every sparse layer.

Same clamped SwiGLU as a routed expert at ``moe_intermediate_size = 2048``
(``n_shared_experts = 1``), but with no routing weight and no dispatch: it runs on
the rank's own token rows while the routed half is still in flight, which is what
hides part of the all-to-all latency.

The shared expert's weights are TP-sharded on the intermediate axis, so each rank
computes ``MOE_INTER / TP_SIZE`` channels and the result joins the routed output in
the same reduction.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, MOE_INTER, T_DYN, TP_SIZE
from models.glm5_3_flash.golden import expert


LOCAL_MOE_INTER = MOE_INTER // TP_SIZE


def golden_expert_shared(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return expert(x, w_gate, w_up, w_down)


@pl.jit.inline
def expert_shared(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    raise NotImplementedError("shared expert kernel body is assigned independently")


__all__ = [
    "LOCAL_MOE_INTER",
    "expert_shared",
    "golden_expert_shared",
]
