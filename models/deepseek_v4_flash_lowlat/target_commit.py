# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl


@pl.jit.inline(auto_scope=False)
def device_commit_inline(
    draft_ids: pl.Tensor[[5], pl.INT32],
    sampled_ids: pl.Tensor[[6, 8], pl.INT32],
    target_features: pl.Tensor[[6, 12288], pl.BF16],
    request_state: pl.Tensor[[4], pl.INT64],
    history_ring: pl.InOut[pl.Tensor[[136, 12288], pl.BF16]],
    control: pl.Out[pl.Tensor[[16], pl.INT64]],
    emitted_ids: pl.Out[pl.Tensor[[6], pl.INT64]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_accept_and_metadata") as decision:
        epoch = pl.read(request_state, [0])
        anchor = pl.read(request_state, [1])
        remaining = pl.read(request_state, [2])
        accepted = pl.const(0, pl.INDEX)
        matching = pl.const(1, pl.INT32)
        for step in pl.range(5):
            if matching != 0:
                if pl.read(draft_ids, [step]) == pl.read(sampled_ids, [step, 0]):
                    accepted = accepted + 1
                else:
                    matching = pl.const(0, pl.INT32)
        advance = pl.cast(accepted + 1, pl.INT64)
        actual = advance
        if remaining < advance:
            actual = remaining
        bonus = pl.cast(pl.read(sampled_ids, [accepted, 0]), pl.INT64)
        last_sampled = pl.read(request_state, [3])
        for step in pl.range(6):
            value = pl.const(-1, pl.INT64)
            if step < actual:
                if step < accepted:
                    value = pl.cast(pl.read(draft_ids, [step]), pl.INT64)
                else:
                    value = bonus
                last_sampled = value
            pl.write(emitted_ids, [step], value)
        for field in pl.range(16):
            pl.write(control, [field], pl.const(0, pl.INT64))
        pl.write(control, [0], epoch)
        pl.write(control, [2], anchor)
        pl.write(control, [3], pl.cast(accepted, pl.INT64))
        pl.write(control, [4], advance)
        pl.write(control, [5], actual)
        pl.write(control, [6], anchor + advance)
        pl.write(control, [7], bonus)
        pl.write(control, [8], last_sampled)
        pl.write(control, [9], remaining - actual)
        pl.write(control, [11], anchor + pl.const(1, pl.INT64))
        pl.write(control, [12], advance)
        if remaining == actual:
            pl.write(control, [13], pl.const(1, pl.INT64))
    with pl.spmd(6 * 24, name_hint="commit_target_features_to_history", deps=[decision]) as copies:
        # The decision task publishes scalar metadata at its task boundary.
        pl.system.cacheinvalid()
        block = pl.tile.get_block_idx()
        row = block // 24
        column = (block % 24) * 512
        advance = pl.read(control, [4])
        start = pl.read(control, [11])
        if row < advance:
            destination = (start + row) % 128
            feature = pl.load(target_features, [row, column], [1, 512])
            pl.store(feature, [destination, column], history_ring)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="publish_history_commit_ack", deps=[copies]
    ) as commit_ack_tid:
        pl.write(control, [1], pl.const(1, pl.INT64))
        pl.write(control, [10], pl.const(1, pl.INT64))
    return history_ring, control, emitted_ids, commit_ack_tid
