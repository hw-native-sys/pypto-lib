# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure tensor-parallel output reduction shared by every attention mode."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.config import D, DECODE_MAX_TOKENS, PREFILL_MAX_TOKENS, T_DYN, TP_SIZE

# Token extent of the sequence-parallel shard a rank keeps after the attention
# output reduce-scatter; the plain all-reduce path keeps the full extent.
OUTPUT_T_DYN = pl.dynamic("V41_TP_OUTPUT_T_DYN")


def golden_tp_output_all_reduce(output_partials: torch.Tensor) -> torch.Tensor:
    """Sum the row-parallel output projection from every TP rank."""
    return output_partials.float().sum(dim=0).to(output_partials.dtype)


def golden_tp_output_reduce_scatter(
    output_partials: torch.Tensor, tp_rank: int, capacity: int | None = None
) -> torch.Tensor:
    """Sum every rank's partial, then keep this rank's contiguous token rows.

    ``capacity`` is one rank's fixed physical slab (``ceil(tokens / tp)``); it
    defaults to the active row count, which is only correct when every capacity
    row is active.  Ownership follows the slab, so a rank whose slab starts past
    the active range owns no rows and returns an empty tensor instead of rows
    belonging to a later slab.
    """
    reduced = output_partials.float().sum(dim=0)
    num_tokens = reduced.shape[0]
    width = capacity if capacity is not None else (num_tokens + output_partials.shape[0] - 1) // (
        output_partials.shape[0]
    )
    first = min(tp_rank * width, num_tokens)
    count = max(0, min(width, num_tokens - first))
    return reduced[first : first + count].to(output_partials.dtype)


def golden_tp_input_all_gather(local_input: torch.Tensor) -> torch.Tensor:
    """Concatenate every rank's locally owned rows in rank order."""
    return torch.cat(list(local_input), dim=0)


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            value = pl.load(output_partial, [row, col], [1, 512])
            pld.tile.remote_store(value, output_window, peer=group_base + tp_rank, offsets=[row, col])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, num_tokens * (D // 512), 64):
            row = tile // (D // 512)
            col = tile % (D // 512) * 512
            acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(output_window, peer=group_base + peer, offsets=[row, col], shape=[1, 512])
                acc = pl.add(acc, value)
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
    return output


def make_decode_tp_output_reduce(scatter=False):
    """Build the decode attention-output collective.

    ``scatter=False`` keeps the historical all-reduce result (every rank holds
    the full token range). ``scatter=True`` is the sequence-parallel boundary:
    every rank still contributes its own partial, but only the contiguous token
    rows this rank owns are written, which is the ReduceScatter(SUM) result and
    never a scaled copy of an all-reduce.

    Ownership follows the rank's *physical slab* (``ceil(tokens / tp)``) rather
    than the active row count, so ``T < TP`` and a partially filled batch keep
    the same row mapping the Attention-input AllGather publishes.  Rows outside
    the active range are written as zeros: the slab is always fully materialized.
    """

    @pl.jit.inline(auto_scope=False)
    def decode_tp_output_reduce(
        output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        first = pl.cast(0, pl.INT32)
        count = pl.cast(num_tokens, pl.INT32)
        if scatter:
            # The output tensor is the fixed physical slab.  Ownership must use
            # that capacity, not active ``num_tokens``; otherwise T<TP and
            # inactive suffixes shift rows between ranks.
            width = pl.tensor.dim(output, 0)
            first = pl.cast(pl.min(tp_rank * width, num_tokens), pl.INT32)
            count = pl.cast(pl.max(0, pl.min(width, num_tokens - first)), pl.INT32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_reuse", allow_early_resolve=False) as reuse_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                                cmp=pld.WaitCmp.Ge)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_publish", deps=[reuse_tid]) as publish_tid:
            pld.tensor.put(dst=output_window, peer=group_base + tp_rank, src=output_partial,
                           dst_offsets=[0, 0], src_offsets=[0, 0], shape=[num_tokens, D],
                           chunk_rows=1, chunk_cols=512, pipeline=True)
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_reduce", deps=[publish_tid],
                   allow_early_resolve=False) as reduce_tid:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1,
                                cmp=pld.WaitCmp.Ge)
            for t in pl.range(first, first + count):
                for col in pl.range(0, D, 512):
                    acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
                    for peer in pl.range(TP_SIZE):
                        value = pld.tile.remote_load(output_window, peer=group_base + peer,
                                                     offsets=[t, col], shape=[1, 512])
                        acc = pl.add(acc, value)
                    output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [t - first, col], output)
            if scatter:
                # The physical slab is fully materialized.  Rows outside this
                # rank's active range (an empty owner, or the padded tail of the
                # last owner) are written as zeros, so no stale window or sentinel
                # value is ever handed to the next stage.
                for t in pl.range(count, pl.tensor.dim(output, 0)):
                    for col in pl.range(0, D, 512):
                        output = pl.store(
                            pl.tile.full([1, 512], dtype=pl.BF16, value=0.0), [t, col], output
                        )
            for peer in pl.range(TP_SIZE):
                pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                                  value=1, op=pld.NotifyOp.AtomicAdd)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_consumed", deps=[reduce_tid],
                   allow_early_resolve=False):
            for peer in pl.range(TP_SIZE):
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                                cmp=pld.WaitCmp.Ge)
        return output

    return decode_tp_output_reduce


decode_tp_output_all_reduce = make_decode_tp_output_reduce()
decode_tp_output_reduce_scatter = make_decode_tp_output_reduce(scatter=True)


@pl.jit.inline(auto_scope=False)
def decode_tp_input_all_gather(
    local_input: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
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


__all__ = [
    "OUTPUT_T_DYN",
    "decode_tp_input_all_gather",
    "decode_tp_output_all_reduce",
    "decode_tp_output_reduce_scatter",
    "golden_tp_output_all_reduce",
    "golden_tp_output_reduce_scatter",
    "prefill_tp_output_all_reduce",
]


if __name__ == "__main__":
    torch.manual_seed(17)
    partials = torch.randn(TP_SIZE, 7, 5)
    reduced = golden_tp_output_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.float().sum(dim=0).to(partials.dtype))
    # The scatter form must equal the all-reduce rows this rank owns.
    gathered = []
    capacity = (7 + TP_SIZE - 1) // TP_SIZE
    for tp_rank in range(TP_SIZE):
        shard = golden_tp_output_reduce_scatter(partials, tp_rank, capacity)
        first = min(tp_rank * capacity, 7)
        count = max(0, min(capacity, 7 - first))
        torch.testing.assert_close(shard, reduced[first : first + count])
        gathered.append(shard)
    # T < TP: one active row keeps the slab mapping, the other owners are empty.
    short = torch.randn(TP_SIZE, 1, 5)
    owners = [golden_tp_output_reduce_scatter(short, tp_rank, 1) for tp_rank in range(TP_SIZE)]
    assert [tuple(owner.shape) for owner in owners] == [(1, 5)] + [(0, 5)] * (TP_SIZE - 1)
    torch.testing.assert_close(owners[0], short.float().sum(dim=0).to(short.dtype))
    # Local rows gathered in rank order rebuild the full token order.
    local = torch.cat([torch.full((capacity, 5), float(r)) for r in range(TP_SIZE)])
    stitched = golden_tp_input_all_gather(local.view(TP_SIZE, -1, 5))
    assert stitched.shape[0] == TP_SIZE * capacity
    torch.testing.assert_close(stitched, local)
    print(f"[GOLDEN] PASS attention TP collectives all-reduce={tuple(reduced.shape)} "
          f"rs_rows={[tuple(s.shape) for s in gathered]} t_lt_tp_rows={[tuple(o.shape) for o in owners]} "
          f"gather={tuple(stitched.shape)}")
