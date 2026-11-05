# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash RMSNorm kernels and shared Torch references."""

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import FLASH as M


# tiling
NORM_T_TILE = 2


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "rms_norm.cpp",
               include_dirs=(pto_isa_include_dir(),))
def rms_norm(
    x: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
):
    """Normalize two BF16 rows of width 128 or 5120 and zero inactive output rows."""
    ...


def golden_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = M.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(dtype)
