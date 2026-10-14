# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode-phase gather of per-rank hidden token slabs across the attention TP group."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_MAX_TOKENS, FLASH as M, T_DYN, TP_SIZE


# Model config.
D = M.hidden_size


@pl.jit.inline(auto_scope=False)
def decode_tp_input_all_gather(
    local_input: pl.Tensor,
    input_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    gathered: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Gather every rank's locally owned rows into the full Attention input.

    Called after the local mHC collapse and RMSNorm, so the gathered tensor is
    the ``[T_local, D] -> [T, D]`` AllGather the sequence-parallel decode block
    needs; the residual stream itself never leaves the rank.

    Every rank publishes its whole physical slab, and the scratch is fully
    materialized on every rank: rows ``[0, num_tokens)`` are the gathered rows,
    and rows past the active range are zeros.  A rank whose slab holds no active
    row still publishes zeros and still takes part in the handshake, so ``T < TP``
    and empty owners need no special case in the caller.
    """
    width = pl.tensor.dim(local_input, 0)
    # Every rank publishes one physical slab, including a fully padded slab on
    # ranks beyond the active range.  Zero-length remote puts are not a valid
    # synchronization primitive on A5 and can leave the peer waits stuck; the
    # gathered result still reads only rows ``[0, num_tokens)``.
    local_count = pl.cast(width, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_input_reuse",
               allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(input_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                            cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_input_publish",
               deps=[reuse_tid]) as publish_tid:
        pld.tensor.put(dst=input_window, peer=group_base + tp_rank, src=local_input,
                       dst_offsets=[0, 0], src_offsets=[0, 0], shape=[local_count, D],
                       chunk_rows=1, chunk_cols=512, pipeline=True)
        for peer in pl.range(TP_SIZE):
            pld.system.notify(input_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_input_wait", deps=[publish_tid],
               allow_early_resolve=False) as arrived_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1,
                            cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="decode_tp_input_gather", deps=[arrived_tid]) as gather_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            peer = row // width
            value = pld.tile.remote_load(input_window, peer=group_base + peer,
                                         offsets=[row - peer * width, col], shape=[1, 512])
            gathered = pl.store(value, [row, col], gathered)
    with pl.spmd(64, name_hint="decode_tp_input_zero_padding", deps=[gather_tid]) as zeroed_tid:
        # Rows past the active range are never gathered; they are still written
        # as zeros so the whole scratch is deterministic on every rank.  The
        # harness seeds the buffer with a sentinel to prove this write happens.
        worker = pl.tile.get_block_idx()
        rows = pl.tensor.dim(gathered, 0)
        for tile in pl.range(worker, rows * (D // 512), 64):
            row = tile // (D // 512)
            if row >= num_tokens:
                col = tile % (D // 512) * 512
                gathered = pl.store(pl.tile.full([1, 512], dtype=pl.BF16, value=0.0), [row, col], gathered)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_input_release",
               deps=[zeroed_tid]) as released_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(input_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_input_consumed",
               deps=[released_tid], allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(input_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                            cmp=pld.WaitCmp.Ge)
    return gathered


@pl.jit.inline
def slab_owner(
    tp_rank: pl.Scalar[pl.INT32],
    slab: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Return ``(first, count)``: the active rows the physical slab *tp_rank* owns.

    ``slab`` is the rank's fixed physical extent (``pl.tensor.dim(x, 0)`` of the
    per-rank tensor), never the active count, so a padded batch, ``T < TP`` and a
    rank whose slab starts past the active range keep the same row mapping on both
    sides of every collective.  This mirrors
    ``utils.sequence_parallel_bounds`` on the CPU side; keep the
    two in step.
    """
    first = pl.cast(pl.min(tp_rank * slab, num_tokens), pl.INT32)
    count = pl.cast(pl.max(0, pl.min(slab, num_tokens - first)), pl.INT32)
    return first, count
