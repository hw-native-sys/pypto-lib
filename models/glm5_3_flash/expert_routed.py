# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 grouped expert FFN over the ``N_LOCAL_EXPERTS`` experts this EP rank owns.

Per expert: ``down(silu(clamp(gate(x), max=10)) * clamp(up(x), -10, 10))`` with
``moe_intermediate_size = 2048``. Both matmuls accumulate INT32 on the cube; the
gate/up epilogue dequantises by ``per-token scale x per-channel weight scale``,
applies the clamped SwiGLU and requantises per row for ``down_proj``. The routing
weight is folded into the output row rather than applied after the combine.

At EP16 each rank owns 288 / 16 = 18 experts, and the routed weights are the bulk
of the model: 43 sparse layers x 288 experts x 3 matrices is 311.7 G parameters,
so 19.5 GB per rank in INT8 out of a ~21.6 GB per-rank weight budget.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, MOE_INTER, N_LOCAL_EXPERTS, RECV_DYN
from models.glm5_3_flash.golden import expert


def golden_expert_routed(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    route_weight: torch.Tensor,
) -> torch.Tensor:
    """``route_weight`` is ``[tokens, 1]``, matching the kernel ABI."""
    return expert(x, w_gate, w_up, w_down, route_weight)


@pl.jit.inline
def expert_routed(
    x_int8: pl.Tensor[[RECV_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    route_weights: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    output: pl.Tensor[[RECV_DYN, D], pl.FP32],
):
    raise NotImplementedError("routed expert kernel body is assigned independently")


__all__ = [
    "expert_routed",
    "golden_expert_routed",
]
