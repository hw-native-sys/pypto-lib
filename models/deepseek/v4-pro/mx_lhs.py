# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Canonical LeftScale wiring for dyn MX quant → matmul_mx.

``pl.mx_quant`` / tquant emits flat scale ``[1, groups]``. Cube LeftScale needs
``[M, K/32]``. Direct ``Mat ND → LeftScale`` is numerically wrong even with
target_shape; stage through GM then TLOAD with ``mx_layout="mx_a_zz"``.

**Production data plane (all ND A-scale stores):** store row-major ND bytes to
GM; ptoas EmitC only emits ``Layout::MX_A_ZZ``, so
``_ptoas_preprocess._rewrite_mx_a_zz_e8m0_to_nd`` rewrites to ``MX_A_ND``
(AND2ZZ). Disabling that rewrite raises ``RuntimeError`` until a full ZZ-on-GM
rollout exists.

Do not reuse one scale GM slot across K-chunks (AIV can overwrite while AIC
TLOADs). Prefer ``pl.range`` / ``pl.unroll`` over ``pl.pipeline`` on MX V2C
scale edges (pipeline can reorder FIFO).

Reference implementation: ``qkv_proj_rope.py`` (qr_proj / qproj / kv paths).

Call sites currently **inline** this pattern (a shared ``@pl.jit.inline`` helper
is not portable across modules: Tile+DynVar annotations get stripped by the
AST rewrite). Copy-paste template::

    # workspace once per kernel (concurrent SPMD × K-chunk slots):
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * M_TILE, K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )

    # per chunk after mx_quant (unique srow = slot_idx * M_TILE):
    la = pl.move(
        pl.move(pl.tile.reinterpret_view(q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    la = pl.set_validshape(la, rows, K_TILE)
    pl.store(pl.tile.reinterpret_view(s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
    las = pl.move(
        pl.load(
            mx_scale_ws, [srow, 0], [M_TILE, K_TILE // MX_BLOCK_K],
            target_memory=pl.Mem.Mat, mx_layout="mx_a_zz",
        ),
        target_memory=pl.Mem.LeftScale,
    )
    las = pl.tget_scale_addr(las, la)
    las = pl.set_validshape(las, rows, K_TILE // MX_BLOCK_K)

For MXFP4 activations use ``pl.FP4`` in the reinterpret_view of ``q``.
"""
