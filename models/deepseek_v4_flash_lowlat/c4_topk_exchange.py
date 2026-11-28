# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal C4 exact-ID transport for six query owners."""

import pypto.language as pl
import pypto.language.distributed as pld


RANKS = 8
OWNERS = 6
K = 512
LANES = 2
SIGNAL_COLUMNS = 32


@pl.jit.inline(auto_scope=False)
def publish_topk(
    local_ids: pl.Tensor[[1, K], pl.INT32],
    ids_window: pld.DistributedTensor[[LANES * OWNERS, K], pl.INT32],
    ready: pld.DistributedTensor[[LANES * OWNERS, SIGNAL_COLUMNS], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    selected: pl.Scalar[pl.TASK_ID],
):
    """Call only on the six owners after exact selection has completed."""
    row = pl.cast((epoch - 1) % LANES, pl.INDEX) * OWNERS + pl.cast(rank, pl.INDEX)
    with pl.spmd(RANKS - 1, name_hint="c4_topk_publish_peer", allow_early_resolve=True,
                 deps=[selected]) as sent:
        link = pl.cast(pl.tile.get_block_idx(), pl.INT32)
        peer = link
        if link >= rank:
            peer = pl.cast(link + 1, pl.INT32)
        pld.tensor.put(dst=ids_window, peer=peer, src=local_ids,
                       dst_offsets=[row, 0], src_offsets=[0, 0], shape=[1, K])
        # Single owner and independent 128-byte slot. An absolute epoch
        # tolerates replicate-path skips in the same request.
        pld.system.notify(target=ready, peer=peer, offsets=[row, 0],
                          value=epoch, op=pld.NotifyOp.Set)
    return sent


@pl.jit.inline(auto_scope=False)
def wait_for_topk(
    ready: pld.DistributedTensor[[LANES * OWNERS, SIGNAL_COLUMNS], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    layer_local_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    row = pl.cast((epoch - 1) % LANES, pl.INDEX) * OWNERS
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="c4_topk_ready",
               deps=[layer_local_ready]) as done:
        for owner in pl.unroll(OWNERS):
            if owner != rank:
                pld.system.defer_wait(signal=ready, offsets=[row + owner, 0],
                                      expected=epoch, cmp=pld.WaitCmp.Ge)
    return done
