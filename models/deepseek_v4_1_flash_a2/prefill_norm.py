# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 Prefill RMSNorm over active rows."""

import pypto.language as pl

from .rmsnorm import NORM_T_TILE, rms_norm_tiled


def make_prefill_norm(capacity, hidden):
    @pl.jit
    def prefill_norm(
        collapsed: pl.Tensor[[capacity, hidden], pl.BF16],
        weight: pl.Tensor[[hidden], pl.BF16],
        normed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="prefill_norm"):
            rms_norm_tiled(collapsed, weight, normed, block, active)
        return normed

    return prefill_norm
