# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm and the fused residual-add RMSNorm used across the backbone.

Every layer applies ``input_layernorm`` before attention and
``post_attention_layernorm`` before the MLP; both read a stream that mHC has
already collapsed, so the kernel sees a plain ``[T, D]`` row. The gamma vectors
stay BF16 and are never quantized (``modules_to_not_convert`` lists every
``*_layernorm`` and ``*_norm``).

The fused form exists because the MTP layer and the model tail add a residual
immediately before the norm; folding the add saves one GM round trip per token.

``rmsnorm_quant`` covers the three **dense** layers (0, 1 and 2). On a sparse layer
the router kernel in :mod:`models.glm5_3_flash.gate` owns the deferred norm and
produces the per-token INT8 view; a dense layer has no router, so without this
variant the K = 4096 activation feeding ``dense_mlp`` would have no owner at all.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, T_DYN
from models.glm5_3_flash.golden import rms_norm
from models.glm5_3_flash.quantization import quantize_per_token_int8


def golden_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x, weight)


def golden_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise and emit the per-token INT8 view the dense MLP consumes."""
    normalized = rms_norm(x, weight)
    quantized, scale = quantize_per_token_int8(normalized)
    return normalized, quantized, scale


def golden_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalised activation and the updated residual."""
    updated = x.float() + residual.float()
    return rms_norm(updated.to(x.dtype), weight), updated.to(x.dtype)


@pl.jit.inline
def rmsnorm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("rmsnorm kernel body is assigned independently")


@pl.jit.inline
def rmsnorm_quant(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    output_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    output_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    raise NotImplementedError("rmsnorm_quant kernel body is assigned independently")


@pl.jit.inline
def add_rmsnorm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    updated_residual: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("add_rmsnorm kernel body is assigned independently")


__all__ = [
    "add_rmsnorm",
    "golden_add_rmsnorm",
    "golden_rmsnorm",
    "golden_rmsnorm_quant",
    "rmsnorm",
    "rmsnorm_quant",
]
