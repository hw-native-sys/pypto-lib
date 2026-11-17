# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill token gathering and restoration across attention parallel groups."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import PREFILL_MAX_TOKENS, FLASH as M, LOCAL_T_DYN, T_DYN


# Model config.
D = M.hidden_size
HC_DIM = M.hc_dim
HC_MULT = M.hc_mult

# Tiling.
GATHER_ROW_TILE = 16


@pl.jit.inline(auto_scope=False)
def prefill_cp_gather_rows(
    local_hidden: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor,
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    group_size: pl.constexpr,
):
    """Push token shards into global-row receive slots and gather from the local window."""
    local_tokens = pl.tensor.dim(local_hidden, 0)
    tokens = pl.tensor.dim(hidden, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_gather_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(group_size):
            pld.system.wait(
                input_arrived,
                offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
    with pl.spmd(64, name_hint="cp_gather_publish", deps=[reuse_tid]) as copied_tid:
        worker = pl.tile.get_block_idx()
        active = pl.max(0, pl.min(local_tokens, num_tokens - cp_rank * local_tokens))
        for block in pl.range(worker, (active + GATHER_ROW_TILE - 1) // GATHER_ROW_TILE, 64):
            row = block * GATHER_ROW_TILE
            target_row = cp_rank * local_tokens + row
            if row + GATHER_ROW_TILE <= active:
                publish_value = pl.load(local_hidden, [row, 0], [GATHER_ROW_TILE, D])
                for peer in pl.range(group_size):
                    pld.tile.remote_store(publish_value, input_window, group_base + peer, [target_row, 0])
            else:
                for tail in pl.range(row, active):
                    tail_publish = pl.load(local_hidden, [tail, 0], [1, D])
                    for peer in pl.range(group_size):
                        target_tail = cp_rank * local_tokens + tail
                        pld.tile.remote_store(tail_publish, input_window, group_base + peer, [target_tail, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_gather_notify", deps=[copied_tid]) as publish_tid:
        for peer in pl.range(group_size):
            pld.system.notify(
                input_arrived,
                peer=group_base + peer, offsets=[cp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_gather_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(group_size):
            pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="cp_gather_copy", deps=[wait_tid]) as copy_tid:
        worker = pl.tile.get_block_idx()
        local_peer = group_base + cp_rank
        for block in pl.range(worker, (tokens + GATHER_ROW_TILE - 1) // GATHER_ROW_TILE, 64):
            row = block * GATHER_ROW_TILE
            if row + GATHER_ROW_TILE <= num_tokens:
                value = pld.tile.remote_load(input_window, local_peer, [row, 0], [GATHER_ROW_TILE, D])
                hidden = pl.store(value, [row, 0], hidden)
            else:
                for tail in pl.range(row, pl.min(row + GATHER_ROW_TILE, tokens)):
                    if tail < num_tokens:
                        tail_value = pld.tile.remote_load(input_window, local_peer, [tail, 0], [1, D])
                    else:
                        tail_value = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
                    hidden = pl.store(tail_value, [tail, 0], hidden)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_gather_release", deps=[copy_tid]) as release_tid:
        for peer in pl.range(group_size):
            pld.system.notify(
                input_arrived,
                peer=group_base + peer, offsets=[cp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_gather_consumed", deps=[release_tid],
               allow_early_resolve=False) as consumed_tid:
        for peer in pl.range(group_size):
            pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(hidden, [0, 0])
        pl.write(hidden, [0, 0], anchor)
    return hidden, consumed_tid


@pl.jit.inline(auto_scope=False)
def prefill_cp_input_allgather(
    local_hidden: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor,
    restore_indices: pl.Tensor[[T_DYN], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    group_size: pl.constexpr,
):
    """Gather padded rank-major rows and restore the global scheduled token order."""
    local_rows = pl.tensor.dim(local_hidden, 0)
    gathered_rows = local_rows * group_size
    rank_hidden = pl.create_tensor([gathered_rows, D], dtype=pl.BF16)
    rank_hidden, gathered = prefill_cp_gather_rows(
        local_hidden, input_window, input_arrived, rank_hidden,
        group_base, cp_rank, gathered_rows, attention_epoch, group_size=group_size,
    )
    rows = pl.tensor.dim(hidden, 0)
    with pl.spmd(32, name_hint="cp_restore_hidden", deps=[gathered]) as restored:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, rows, 32):
            source = pl.cast(pl.read(restore_indices, [row]), pl.INDEX)
            if source >= 0:
                hidden[row:row + 1, :] = rank_hidden[source:source + 1, :]
            else:
                hidden[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return hidden, restored


@pl.jit.inline(auto_scope=False)
def prefill_cp_gather_hc(
    local_hidden: pl.Tensor,
    local_pre_mix: pl.Tensor,
    restore_indices: pl.Tensor,
    hidden: pl.Tensor,
    pre_mix: pl.Tensor,
    window: pld.DistributedTensor,
    arrived: pld.DistributedTensor,
    group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    group_size: pl.constexpr,
):
    """Gather local HC shards into scheduler order using the CP restoration map."""
    local_rows = pl.tensor.dim(local_hidden, 0)
    rows = pl.tensor.dim(hidden, 0)
    local_flat = pl.reshape(local_hidden, [local_rows, HC_DIM])
    global_flat = pl.reshape(hidden, [rows, HC_DIM])
    padded_pre = pl.create_tensor([local_rows, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="cp_hc_pack", deps=[ready]) as packed:
        writer = pl.tile.get_block_idx()
        for row in pl.range(writer, local_rows):
            for lane in pl.range(16):
                value = pl.cast(0.0, pl.FP32)
                if lane < HC_MULT:
                    value = pl.read(local_pre_mix, [row, lane])
                pl.write(padded_pre, [row, lane], value)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_reuse", deps=[packed]) as reusable:
        for peer in pl.range(group_size):
            pld.system.wait(arrived, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint="cp_hc_publish", deps=[reusable]) as published:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, local_rows, 32):
            for column in pl.range(0, HC_DIM, D):
                published_value = pl.tile.load(local_flat, [row, column], [1, D])
                pld.tile.remote_store(published_value, window, peer=group_base + cp_rank, offsets=[row, column])
            coefficients = pl.tile.load(padded_pre, [row, 0], [1, 16])
            pld.tile.remote_store(coefficients, window, peer=group_base + cp_rank, offsets=[row, HC_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_announce", deps=[published]) as announced:
        for peer in pl.range(group_size):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[cp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_wait", deps=[announced]) as received:
        for peer in pl.range(group_size):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint="cp_hc_restore", deps=[received]) as copied:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, rows, 32):
            source = pl.read(restore_indices, [row])
            for column in pl.range(0, HC_DIM, D):
                if source >= 0:
                    owner = source // local_rows
                    offset = source % local_rows
                    restored_value = pld.tile.remote_load(window, peer=group_base + owner, offsets=[offset, column], shape=[1, D])
                else:
                    restored_value = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
                global_flat = pl.store(restored_value, [row, column], global_flat)
    with pl.spmd(1, name_hint="cp_hc_restore_mix", deps=[received]) as mix_copied:
        writer = pl.tile.get_block_idx()
        for row in pl.range(writer, rows):
            source = pl.read(restore_indices, [row])
            if source >= 0:
                owner = source // local_rows
                offset = source % local_rows
                coefficients = pld.tile.remote_load(window, peer=group_base + owner, offsets=[offset, HC_DIM], shape=[1, 16])
                for lane in pl.range(HC_MULT):
                    value = pl.tile.read(coefficients, [0, lane])
                    pl.write(pre_mix, [row, lane], value)
            else:
                for lane in pl.range(HC_MULT):
                    pl.write(pre_mix, [row, lane], pl.cast(0.0, pl.FP32))
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_release", deps=[copied, mix_copied]) as released:
        for peer in pl.range(group_size):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[cp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_consumed", deps=[released]) as consumed:
        for peer in pl.range(group_size):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(hidden, [0, 0, 0])
        pl.write(hidden, [0, 0, 0], anchor)
    return consumed
