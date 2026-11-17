# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tensor-parallel group reductions shared by the prefill and decode layers.

The attention output projection and the lightning indexer are sharded over the
configured TP ranks: each rank produces a partial output (head/group slices) or a
partial score sum (the local index heads).  Both reductions publish one partial per
rank into its own window, sum every group peer's window, and write the result
back locally.

Window reuse across the layer loop uses one folded signal counter per rank:
publish adds one, release adds one, so epoch ``e`` starts at ``2*(e-1)`` and the
next epoch cannot overwrite a region before every peer released it.
"""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_MAX_TOKENS, EP_SIZE, FLASH as M, TP_SIZE


D = M.hidden_size
D_TILE = 512
SCORE_TILE = 256


@pl.jit.inline(auto_scope=False)
def tp_output_reduce(
    partial: pl.Tensor,
    window: pld.DistributedTensor,
    signal: pld.DistributedTensor,
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Sum the BF16 output-projection partials of one TP group in FP32."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                signal, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
        anchor = pl.read(partial, [0, 0])
        pl.write(partial, [0, 0], anchor)
    with pl.spmd(rows, name_hint="tp_out_publish", deps=[reuse_tid]) as publish_tid:
        row = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=window, peer=group_base + tp_rank, src=partial,
            dst_offsets=[row, 0], src_offsets=[row, 0], shape=[1, D],
            chunk_rows=1, chunk_cols=D_TILE, pipeline=True,
        )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(signal, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(rows, name_hint="tp_out_sum", deps=[wait_tid]) as sum_tid:
        row = pl.tile.get_block_idx()
        acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
        for peer in pl.range(TP_SIZE):
            value = pld.tile.remote_load(
                window, peer=group_base + peer, offsets=[row, 0], shape=[1, D],
            )
            acc = pl.add(acc, pl.cast(value, target_type=pl.FP32))
        pl.store(pl.cast(acc, target_type=pl.BF16, mode="rint"), [row, 0], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_release", deps=[sum_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_out_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(signal, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def tp_score_reduce(
    scores: pl.Tensor,
    window: pld.DistributedTensor,
    signal: pld.DistributedTensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    rows: pl.Scalar[pl.INT32],
    cols: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Sum the FP32 index-score partials of one TP group in place."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_score_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                signal, offsets=[peer, 0], expected=(epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
        anchor = pl.read(scores, [0, 0])
        pl.write(scores, [0, 0], anchor)
    with pl.spmd(rows, name_hint="tp_score_publish", deps=[reuse_tid]) as publish_tid:
        row = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=window, peer=group_base + tp_rank, src=scores,
            dst_offsets=[row, 0], src_offsets=[row, 0], shape=[1, cols],
            chunk_rows=1, chunk_cols=SCORE_TILE, pipeline=True,
        )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_score_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_score_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(signal, offsets=[peer, 0], expected=epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(rows, name_hint="tp_score_sum", deps=[wait_tid]) as sum_tid:
        row = pl.tile.get_block_idx()
        for c0 in pl.range(0, cols, SCORE_TILE):
            width = pl.min(SCORE_TILE, cols - c0)
            acc = pl.tile.full([1, SCORE_TILE], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(
                    window, peer=group_base + peer, offsets=[row, c0], shape=[1, SCORE_TILE],
                    valid_shape=[1, width],
                )
                acc = pl.add(acc, value)
            pl.store(pl.set_validshape(acc, 1, width), [row, c0], scores)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_score_release", deps=[sum_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_score_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(signal, offsets=[peer, 0], expected=epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(scores, [0, 0])
        pl.write(scores, [0, 0], anchor)
    return scores


@pl.jit
def tp_reduce_test_rank(
    partial: pl.InOut[pl.Tensor[[2, D], pl.BF16]],
    summed: pl.Out[pl.Tensor[[2, D], pl.BF16]],
    window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(partial, 0)
    tp_output_reduce(partial, window, signal, summed, group_base, tp_rank, rows,
                     pl.const(1, pl.INT32))


@pl.jit.host
def tp_reduce_test(
    partial: pl.InOut[pl.Tensor[[EP_SIZE, 2, D], pl.BF16]],
    summed: pl.Out[pl.Tensor[[EP_SIZE, 2, D], pl.BF16]],
):
    """TP-group reduce with per-rank distinguishable partials.

    Each rank feeds a distinct partial; the reference is the group sum, which
    only correct peer resolution (and a correct store/load path) can match.
    """
    buf = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.BF16)
    sig_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        win = pld.window(buf, [DECODE_MAX_TOKENS, D], dtype=pl.BF16)
        sig = pld.window(sig_buf, [TP_SIZE, 1], dtype=pl.INT32)
        tp_reduce_test_rank(partial[rank], summed[rank], win, sig, rank,
                            rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, device=rank)
