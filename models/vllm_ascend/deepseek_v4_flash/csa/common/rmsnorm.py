# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""上游 pypto-lib `rmsnorm.py` 里 hc_pre 需要的那两个 inline 片段。

只取 `rms_norm_inverse` 与 `rms_norm_apply`：hc_pre 把 attention 的
input_layernorm 融进了自己的收尾（见 `hc_pre.hc_pre_norm`），复用上游这两段
可以保证归一化的数值路径与 pypto-lib 参考实现逐步一致。整支
`rms_norm` 入口本包用不到，不搬。
"""

import pypto.language as pl

from .config import FLASH as M

T_TILE = 8
CHUNK_D_DYN = pl.dynamic("RMS_NORM_CHUNK_D_DYN")

D = M.hidden_size
EPS = M.rms_norm_eps


@pl.jit.inline
def rms_norm_inverse(sq_sum: pl.Tensor[[1, T_TILE], pl.FP32]):
    """Compute inverse RMS from the per-token full-width square sums."""
    mean = pl.mul(sq_sum, 1.0 / D)
    variance = pl.add(mean, EPS)
    inverse = pl.rsqrt(variance, high_precision=True)
    return inverse


@pl.jit.inline
def rms_norm_apply(
    x: pl.Tensor[[T_TILE, CHUNK_D_DYN], pl.BF16],
    norm_w: pl.Tensor[[1, CHUNK_D_DYN], pl.BF16],
    inverse: pl.Tensor[[1, T_TILE], pl.FP32],
):
    """Apply inverse RMS and norm weights to one BF16 activation chunk."""
    x_fp32 = pl.cast(x, pl.FP32)
    inverse_col = pl.reshape(inverse, [T_TILE, 1])
    norm_w_fp32 = pl.cast(norm_w, pl.FP32)
    scaled = pl.row_expand_mul(x_fp32, inverse_col)
    normed = pl.col_expand_mul(scaled, norm_w_fp32)
    result = pl.cast(normed, pl.BF16, mode="rint")
    return result
