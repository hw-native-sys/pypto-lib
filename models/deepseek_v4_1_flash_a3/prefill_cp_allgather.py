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

from .config import PREFILL_MAX_TOKENS, FLASH as M, TP_SIZE
from .dsl_types import SP_T_DYN, T_DYN


# Model config.
D = M.hidden_size

# Tiling.
SP_ROW_TILE = 8
SP_COLUMN_TILE = 2560
SP_WORKER_TILE = 64


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
        active = pl.max(0, pl.min(local_tokens, num_tokens - tp_rank * local_tokens))
        column_tiles = D // SP_COLUMN_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_reuse", allow_early_resolve=False) as reuse_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(
                    input_arrived,
                    offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
                )
        publish_tiles = (active + SP_ROW_TILE - 1) // SP_ROW_TILE * column_tiles
        publish_workers = pl.max(1, pl.min(pl.min(SP_WORKER_TILE, publish_tiles), pl.system.available_aiv_count()))
        with pl.spmd(publish_workers, name_hint="sp_gather_publish", deps=[reuse_tid]) as publish_tid:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, publish_tiles, publish_workers):
                row = tile // column_tiles * SP_ROW_TILE
                col = tile % column_tiles * SP_COLUMN_TILE
                rows = pl.min(SP_ROW_TILE, active - row)
                pld.tensor.put(
                    dst=input_window, peer=group_base + tp_rank, src=local_hidden,
                    dst_offsets=[row, col], src_offsets=[row, col], shape=[rows, SP_COLUMN_TILE],
                    chunk_rows=SP_ROW_TILE, chunk_cols=SP_COLUMN_TILE,
                )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_ready", deps=[publish_tid]) as ready_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.notify(
                    input_arrived,
                    peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_wait", deps=[ready_tid],
                   allow_early_resolve=False) as wait_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
        tiles_per_shard = (local_tokens + SP_ROW_TILE - 1) // SP_ROW_TILE
        copy_tiles = TP_SIZE * tiles_per_shard * column_tiles
        copy_workers = pl.max(1, pl.min(pl.min(SP_WORKER_TILE, copy_tiles), pl.system.available_aiv_count()))
        with pl.spmd(copy_workers, name_hint="sp_gather_copy", deps=[wait_tid]) as copy_tid:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, copy_tiles, copy_workers):
                row_tile = tile // column_tiles
                source = row_tile // tiles_per_shard
                source_row = row_tile % tiles_per_shard * SP_ROW_TILE
                row = source * local_tokens + source_row
                col = tile % column_tiles * SP_COLUMN_TILE
                extent = pl.min(SP_ROW_TILE, pl.min(local_tokens - source_row, tokens - row))
                live = pl.min(extent, num_tokens - row)
                if live == SP_ROW_TILE:
                    value = pld.tile.remote_load(
                        input_window,
                        peer=group_base + source, offsets=[source_row, col], shape=[SP_ROW_TILE, SP_COLUMN_TILE],
                    )
                    hidden = pl.store(value, [row, col], hidden)
                elif live > 0:
                    tail = pld.tile.remote_load(
                        input_window, peer=group_base + source, offsets=[source_row, col],
                        shape=[SP_ROW_TILE, SP_COLUMN_TILE], valid_shape=[live, SP_COLUMN_TILE],
                    )
                    padded = pl.fillpad(tail, pad_value=pl.PadValue.zero)
                    hidden = pl.store(pl.set_validshape(padded, extent, SP_COLUMN_TILE), [row, col], hidden)
                elif extent > 0:
                    zero = pl.tile.full([SP_ROW_TILE, SP_COLUMN_TILE], dtype=pl.BF16, value=0.0)
                    hidden = pl.store(pl.set_validshape(zero, extent, SP_COLUMN_TILE), [row, col], hidden)
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
            anchor = pl.read(hidden, [0, 0])
            pl.write(hidden, [0, 0], anchor)
        return hidden

    return prefill_sp_input_allgather


prefill_sp_input_allgather = make_prefill_sp_input_allgather()
