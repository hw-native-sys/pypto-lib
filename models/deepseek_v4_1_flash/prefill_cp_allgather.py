# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill sequence-parallel gather of contiguous token shards across the attention TP group."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import PREFILL_MAX_TOKENS, FLASH as M, SP_T_DYN, T_DYN, TP_SIZE


# Model config.
D = M.hidden_size

# Tiling.
D_TILE = 512


def make_prefill_sp_input_allgather(capacity=PREFILL_MAX_TOKENS):
    """Build the prefill collective for the specified communication capacity."""
    @pl.jit.inline(auto_scope=False)
    def prefill_sp_input_allgather(
        local_hidden: pl.Tensor[[SP_T_DYN, D], pl.BF16],
        input_window: pld.DistributedTensor[[capacity, D], pl.BF16],
        input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        hidden: pl.Tensor[[T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Gather contiguous token shards and zero the inactive global rows."""
        local_tokens = pl.tensor.dim(local_hidden, 0)
        tokens = pl.tensor.dim(hidden, 0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_reuse", allow_early_resolve=False) as reuse_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(
                    input_arrived,
                    offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
                )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_publish", deps=[reuse_tid]) as publish_tid:
            active = pl.max(0, pl.min(local_tokens, num_tokens - tp_rank * local_tokens))
            if active > 0:
                pld.tensor.put(
                    dst=input_window, peer=group_base + tp_rank, src=local_hidden,
                    dst_offsets=[0, 0], src_offsets=[0, 0], shape=[active, D],
                    chunk_rows=1, chunk_cols=D_TILE, pipeline=True,
                )
            for peer in pl.range(TP_SIZE):
                pld.system.notify(
                    input_arrived,
                    peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_wait", deps=[publish_tid],
                   allow_early_resolve=False) as wait_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
        with pl.spmd(64, name_hint="sp_gather_copy", deps=[wait_tid]) as copy_tid:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, tokens * (D // D_TILE), 64):
                row = tile // (D // D_TILE)
                col = tile % (D // D_TILE) * D_TILE
                if row < num_tokens:
                    source = row // local_tokens
                    source_row = row % local_tokens
                    value = pld.tile.remote_load(
                        input_window,
                        peer=group_base + source, offsets=[source_row, col], shape=[1, D_TILE],
                    )
                    hidden = pl.store(value, [row, col], hidden)
                else:
                    zero = pl.tile.full([1, D_TILE], dtype=pl.BF16, value=0.0)
                    hidden = pl.store(zero, [row, col], hidden)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_release", deps=[copy_tid]) as release_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.notify(
                    input_arrived,
                    peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_consumed", deps=[release_tid],
                   allow_early_resolve=False):
            for peer in pl.range(TP_SIZE):
                pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        return hidden

    return prefill_sp_input_allgather


prefill_sp_input_allgather = make_prefill_sp_input_allgather()
