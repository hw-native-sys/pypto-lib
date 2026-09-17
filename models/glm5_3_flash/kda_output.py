# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The KDA epilogue: the gated output norm and the output projection.

``Glm5NextTextRMSNormGated`` normalises each 128-wide head in **strict FP32** (the
gamma is not downcast) and then multiplies by ``sigmoid(out_gate)`` before the heads
are flattened back to ``[T, KDA_H * KDA_DIM]`` and projected to ``[T, D]``.

``o_proj`` is head-sharded, so the projection is row-parallel and its result needs
the layer's TP16 all-reduce before mHC folds it back into the residual stream. Like
the rest of the KDA block, the weight stays BF16.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, T_DYN
from models.glm5_3_flash.golden import rms_norm_gated


def golden_kda_output(
    core_attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    out_gate: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm_gated(core_attn_out, norm_weight, out_gate)
    flattened = normalized.reshape(*normalized.shape[:-2], -1)
    # The kernel emits an FP32 partial for attention_tp to reduce, so the golden
    # accumulates in FP32 rather than returning the BF16 input dtype.
    return torch.nn.functional.linear(flattened.float(), w_o.float())


@pl.jit.inline
def kda_output(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    raise NotImplementedError("KDA output kernel body is assigned independently")


__all__ = [
    "golden_kda_output",
    "kda_output",
]
