# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real eight-rank, monotonic-epoch gate for reusing manual communication slots."""

import pypto.language as pl
import pypto.language.distributed as pld


@pl.jit.inline(auto_scope=False)
def eight_rank_reuse_gate(
    current_epoch: pl.Tensor[[1], pl.INT32],
    arrival_signal: pld.DistributedTensor[[8, 32], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    local_reuse_ready: pl.Scalar[pl.TASK_ID],
):
    # One sender owns each row. Signals retain their values; epochs must never
    # rewind when a benchmark restores the logical prefix/cache snapshot.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="cycle_publish_local_exit", deps=[local_reuse_ready]
    ) as publish_tid:
        pl.system.cacheinvalid()
        epoch = pl.read(current_epoch, [0])
        for peer in pl.range(8):
            if peer != my_rank:
                pld.system.notify(
                    target=arrival_signal, peer=peer, offsets=[my_rank, 0], value=epoch, op=pld.NotifyOp.Set
                )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="cycle_wait_all_local_exits", deps=[publish_tid]
    ) as wait_tid:
        epoch = pl.read(current_epoch, [0])
        for peer in pl.range(8):
            if peer != my_rank:
                pld.system.defer_wait(
                    signal=arrival_signal, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge
                )
    # This depends on deferred completion, not just the wait kernel's END.
    global_reuse_ready = pl.system.task_dummy(deps=[publish_tid, wait_tid])
    return global_reuse_ready
