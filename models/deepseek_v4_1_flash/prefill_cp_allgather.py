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
HC_DIM = M.hc_dim
HC_MULT = M.hc_mult

# Tiling.
D_TILE = D


@pl.jit.inline(auto_scope=False)
def prefill_sp_input_allgather(
    local_hidden: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
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
    with pl.spmd(64, name_hint="sp_gather_publish", deps=[reuse_tid]) as copied_tid:
        worker = pl.tile.get_block_idx()
        active = pl.max(0, pl.min(local_tokens, num_tokens - tp_rank * local_tokens))
        for row in pl.range(worker, active, 64):
            publish_value = pl.load(local_hidden, [row, 0], [1, D])
            pld.tile.remote_store(publish_value, input_window, peer=group_base + tp_rank, offsets=[row, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_gather_notify", deps=[copied_tid]) as publish_tid:
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
        anchor = pl.read(hidden, [0, 0])
        pl.write(hidden, [0, 0], anchor)
    return hidden


@pl.jit.inline(auto_scope=False)
def prefill_cp_restore_hc(
    hidden: pl.Tensor,
    pre_mix: pl.Tensor,
    window: pld.DistributedTensor,
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Gather contiguous owner HC rows and delayed mixing weights in FP32."""
    rows = pl.tensor.dim(hidden, 0)
    width = rows // TP_SIZE
    begin = tp_rank * width
    flat = pl.reshape(hidden, [rows, HC_DIM])
    padded_pre = pl.create_tensor([rows, 16], dtype=pl.FP32)
    with pl.spmd(1, name_hint="cp_pre_pack", deps=[ready]) as packed:
        writer = pl.tile.get_block_idx()
        for row in pl.range(begin + writer, begin + width):
            for lane in pl.range(16):
                pre_value = pl.cast(0.0, pl.FP32)
                if lane < HC_MULT:
                    pre_value = pl.read(pre_mix, [row, lane])
                pl.write(padded_pre, [row, lane], pre_value)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_reuse", deps=[packed]) as reusable:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint="cp_hc_publish", deps=[reusable]) as published:
        worker = pl.tile.get_block_idx()
        for row in pl.range(begin + worker, begin + width, 32):
            for col in pl.range(0, HC_DIM, D):
                value = pl.tile.load(flat, [row, col], [1, D])
                pld.tile.remote_store(value, window, peer=group_base + tp_rank, offsets=[row, col])
            coefficients = pl.tile.load(padded_pre, [row, 0], [1, 16])
            pld.tile.remote_store(coefficients, window, peer=group_base + tp_rank, offsets=[row, HC_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_announce", deps=[published]) as announced:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_wait", deps=[announced]) as received:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(32, name_hint="cp_hc_gather", deps=[received]) as copied:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, (rows + 3) // 4, 32):
            for row in pl.range(block * 4, pl.min(block * 4 + 4, rows)):
                owner = row // width
                if owner != tp_rank:
                    for col in pl.range(0, HC_DIM, D):
                        value = pld.tile.remote_load(window, peer=group_base + owner, offsets=[row, col], shape=[1, D])
                        flat = pl.store(value, [row, col], flat)
                    pre_tile = pld.tile.remote_load(window, peer=group_base + owner, offsets=[row, HC_DIM], shape=[1, 16])
                    for lane in pl.range(HC_MULT):
                        coefficient = pl.tile.read(pre_tile, [0, lane])
                        pl.write(pre_mix, [row, lane], coefficient)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_release", deps=[copied]) as released:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hc_consumed", deps=[released]) as consumed:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(hidden, [0, 0, 0])
        pl.write(hidden, [0, 0, 0], anchor)
    return consumed
