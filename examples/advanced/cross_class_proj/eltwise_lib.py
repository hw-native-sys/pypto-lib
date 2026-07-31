# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Elementwise kernels packaged as static methods on an ``Elementwise`` class.

Same dispatch pattern as ``proj_lib.Projections``: the ``@pl.jit.inline`` body
lives inside a class for namespacing, and a module-level alias re-exports it
so ``@pl.jit`` dep auto-discovery (which only matches bare-name calls) finds
the helper from the entry function.
"""

import pypto.language as pl

from config import ADD_OUT_CHUNK, BATCH, HIDDEN


class Elementwise:
    """Elementwise helpers used after the projection step."""
    @pl.jit.inline
    def residual_add(
        a: pl.Tensor[[BATCH, HIDDEN], pl.FP32],
        b: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    ):
        """``out = a + cast(b, FP32)`` — N parallel, no K reduction."""
        for n0 in pl.parallel(0, HIDDEN, ADD_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_add"):
                a_tile = a[:, n0 : n0 + ADD_OUT_CHUNK]
                b_tile = b[:, n0 : n0 + ADD_OUT_CHUNK]
                b_f32 = pl.cast(b_tile, target_type=pl.FP32)
                sum_tile = pl.add(a_tile, b_f32)
                out = pl.assemble(out, sum_tile, [0, n0])
        return out

