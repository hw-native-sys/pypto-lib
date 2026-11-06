# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

import pypto.language.distributed as pld


@pl.jit.inline(auto_scope=False)
def device_input_bridge(
    anchor_input: pl.Tensor[[6], pl.INT64],
    markov_ids: pl.Tensor[[5], pl.INT32],
    expected_epoch: pl.Tensor[[1], pl.INT32],
    prepared_ids: pl.Tensor[[6], pl.INT64],
    observed_epoch: pl.Tensor[[1], pl.INT32],
    ids_window: pld.DistributedTensor[[8], pl.INT64],
    ready_signal: pld.DistributedTensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    peers: pl.Scalar[pl.INT32],
):
    packed = pl.create_tensor([8], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_compose_publish") as publish_tid:
        if my_rank == 0:
            epoch = pl.read(expected_epoch, [0])
            pl.write(packed, [0], pl.read(anchor_input, [0]))
            for k in pl.range(5):
                token = pl.read(markov_ids, [k])
                pl.write(packed, [k + 1], pl.cast(token, pl.INT64))
            pl.write(packed, [6], pl.cast(epoch, pl.INT64))
            pl.write(packed, [7], pl.cast(0, pl.INT64))
            # Scalar GM stores use D-cache; publish before MTE2 TPUT reads.
            pl.system.cacheinvalid()
            pl.system.fence()
            for peer in pl.range(peers):
                pld.tensor.put(
                    dst=ids_window, peer=peer, src=packed, dst_offsets=[0], src_offsets=[0], shape=[8]
                )
                pld.system.notify(
                    target=ready_signal, peer=peer, offsets=[0], value=epoch, op=pld.NotifyOp.Set
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_wait") as wait_tid:
        epoch = pl.read(expected_epoch, [0])
        pld.system.defer_wait(signal=ready_signal, offsets=[0], expected=epoch, cmp=pld.WaitCmp.Ge)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="bridge_readback", deps=[publish_tid, wait_tid]
    ) as ready_tid:
        # defer_wait resolves in the scheduler, outside this consumer kernel.
        pl.system.cacheinvalid()
        for token in pl.range(6):
            pl.write(prepared_ids, [token], pl.read(ids_window, [token]))
        pl.write(observed_epoch, [0], pl.cast(pl.read(ids_window, [6]), pl.INT32))
    return prepared_ids, ready_tid
