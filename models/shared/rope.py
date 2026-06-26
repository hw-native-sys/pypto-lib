# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared RoPE rotate-half primitive: ``@pl.inline``.

See ``~/workspace/subprojects/pypto/docs/guide-ct-args.md`` for the full
context on ``@pl.inline`` + CT kwargs (in the compiler subproject, not here).
"""

import pypto.language as pl


@pl.inline
def rope_rotate_half(x_lo, x_hi, cos_lo, cos_hi, sin_lo, sin_hi):
    """Rotate-half: apply (cos, sin) to (lo, hi) halves.

    Applies:
        rot_lo = x_lo * cos_lo - x_hi * sin_lo
        rot_hi = x_hi * cos_hi + x_lo * sin_hi

    Each half uses its own cos/sin slice (lo half ← cos_lo/sin_lo,
    hi half ← cos_hi/sin_hi).

    Args:
        x_lo: Low half of the tensor.
        x_hi: High half of the tensor.
        cos_lo: Cosine for the low half.
        cos_hi: Cosine for the high half.
        sin_lo: Sine for the low half.
        sin_hi: Sine for the high half.

    Returns:
        Tuple ``(rot_lo, rot_hi)``.
    """
    rot_lo = pl.sub(pl.col_expand_mul(x_lo, cos_lo), pl.col_expand_mul(x_hi, sin_lo))
    rot_hi = pl.add(pl.col_expand_mul(x_hi, cos_hi), pl.col_expand_mul(x_lo, sin_hi))
    return rot_lo, rot_hi
