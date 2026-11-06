# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize S6 metadata and publish the committed request state."""

import pypto.language as pl




@pl.jit.inline(auto_scope=False)
def prepare_device_inputs_after(
    request_state: pl.Tensor[[4], pl.INT64],
    input_ids: pl.Tensor[[6], pl.INT64],
    position_ids: pl.Tensor[[6], pl.INT32],
    kv_seq_lens: pl.Tensor[[1], pl.INT32],
    sampling_positions: pl.Tensor[[6], pl.INT32],
    bridge_epoch: pl.Tensor[[1], pl.INT32],
    request_snapshot: pl.Tensor[[4], pl.INT64],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="s6_metadata_from_device_state", deps=[prior_cycle_gate]
    ) as ready_tid:
        pl.system.cacheinvalid()
        epoch = pl.read(request_state, [0])
        anchor = pl.read(request_state, [1])
        last_sampled = pl.read(request_state, [3])
        for field in pl.range(4):
            pl.write(request_snapshot, [field], pl.read(request_state, [field]))
        for token in pl.range(6):
            position = pl.cast(anchor + 1 + token, pl.INT32)
            pl.write(position_ids, [token], position)
            pl.write(sampling_positions, [token], position)
            pl.write(input_ids, [token], pl.const(0, pl.INT64))
        pl.write(input_ids, [0], last_sampled)
        pl.write(kv_seq_lens, [0], pl.cast(anchor + 7, pl.INT32))
        pl.write(bridge_epoch, [0], pl.cast(epoch, pl.INT32))
        pl.system.cacheinvalid()
        pl.system.fence()
    return input_ids, position_ids, kv_seq_lens, sampling_positions, bridge_epoch, request_snapshot, ready_tid
