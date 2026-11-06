# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

ROWS = pl.dynamic("NATIVE_D_HISTORY_INPUT_ROWS")

MAIN_IN = 12288

RING_ROWS = 128


@pl.jit.inline(auto_scope=False)
def history_ring_to_window(
    ring: pl.Tensor[[ROWS, MAIN_IN], pl.BF16],
    anchors: pl.Tensor,
    ordered: pl.Out[pl.Tensor[[ROWS, MAIN_IN], pl.BF16]],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    ring.bind_dynamic(0, ROWS)
    ordered.bind_dynamic(0, ROWS)
    rows = pl.tensor.dim(ring, 0)
    with pl.spmd(
        rows, name_hint="history_ring_unpack", allow_early_resolve=True, deps=[prior_cycle_gate]
    ) as ready_tid:
        row = pl.tile.get_block_idx()
        position = pl.cast(pl.read(anchors, [0]), pl.INDEX)
        count = pl.min(position + 1, RING_ROWS)
        first = position + 1 - count
        if row < count:
            source_row = (first + row) % RING_ROWS
            ordered[row : row + 1, 0:MAIN_IN] = ring[source_row : source_row + 1, 0:MAIN_IN]
        else:
            ordered[row : row + 1, 0:MAIN_IN] = pl.full([1, MAIN_IN], dtype=pl.BF16, value=0.0)
    return (ordered, ready_tid)
