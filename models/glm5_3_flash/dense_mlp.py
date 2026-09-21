# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The dense clamped-SwiGLU MLP of layers 0, 1 and 2.

``first_k_dense_replace = 3``, so the first three layers have no router and no
experts: a single MLP at ``intermediate_size = 12288``, six times wider than one
routed expert. All three of these layers are KDA attention layers.

W8A8 with the same per-channel weight scales and per-token activation scales as the
experts; the intermediate is TP-sharded, so each rank owns 12288 / 16 = 768
channels and the ``down_proj`` result needs the layer's TP all-reduce.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, DENSE_INTER, T_DYN, TP_SIZE
from models.glm5_3_flash.golden import expert


LOCAL_DENSE_INTER = DENSE_INTER // TP_SIZE


def golden_dense_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return expert(x, w_gate, w_up, w_down)


@pl.jit.inline
def dense_mlp(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    raise NotImplementedError("dense MLP kernel body is assigned independently")


__all__ = [
    "LOCAL_DENSE_INTER",
    "dense_mlp",
    "golden_dense_mlp",
]
