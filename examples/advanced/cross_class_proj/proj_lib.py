# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Projection kernels packaged as static methods on a ``Projections`` class.

Demonstrates organising ``@pl.jit.inline`` helpers behind a class namespace
while keeping them discoverable by ``@pl.jit`` dep auto-discovery.

Why the module-level alias?
---------------------------
``_discover_deps`` in ``pypto/python/pypto/jit/decorator.py`` only scans the
entry function's AST for **bare-name** calls (``ast.Call`` whose ``func`` is
an ``ast.Name``). Method-style calls like ``Projections.linear(...)`` are
``ast.Attribute`` and are *not* picked up. Re-exporting the static method as
a module-level binding (``linear = Projections.linear``) lets the entry
function call it as ``linear(...)`` so dep discovery succeeds.
"""

import pypto.language as pl

from config import BATCH, HIDDEN, K_PROJ_CHUNK, N_OUT_CHUNK


class Projections:
    """Linear-projection helpers grouped under a class namespace.

    Each method is decorated with ``@pl.jit.inline`` so its body is spliced
    into the caller by the ``InlineFunctions`` IR pass. ``@staticmethod`` is
    layered on top so Python returns the underlying ``JITFunction`` when the
    attribute is read off the class (``Projections.linear``).
    """
    @pl.jit.inline
    def linear(
        x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        w: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
        y: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    ):
        """``y = x @ w`` — N parallel, K reduction pipelined inside each scope."""
        for n0 in pl.parallel(0, HIDDEN, N_OUT_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear"):
                acc = pl.create_tensor([BATCH, N_OUT_CHUNK], dtype=pl.FP32)
                for kb in pl.pipeline(0, HIDDEN // K_PROJ_CHUNK, stage=2):
                    k0 = kb * K_PROJ_CHUNK
                    tile_x = x[:, k0 : k0 + K_PROJ_CHUNK]
                    tile_w = w[k0 : k0 + K_PROJ_CHUNK, n0 : n0 + N_OUT_CHUNK]
                    if k0 == 0:
                        acc = pl.matmul(tile_x, tile_w, out_dtype=pl.FP32)
                    else:
                        acc = pl.matmul_acc(acc, tile_x, tile_w)
                y = pl.assemble(y, acc, [0, n0])
        return y

