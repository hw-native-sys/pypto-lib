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

OUTPUT_T_DYN = pl.dynamic("V41_TP_OUTPUT_T_DYN")


def golden_tp_output_all_reduce(output_partials: torch.Tensor) -> torch.Tensor:
    """Sum the row-parallel output projection from every TP rank."""
    return output_partials.float().sum(dim=0).to(output_partials.dtype)


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    reduce_scatter: pl.constexpr = False,
):
    first = pl.cast(0, pl.INT32)
    count = pl.cast(num_tokens, pl.INT32)
    if reduce_scatter:
        width = (num_tokens + TP_SIZE - 1) // TP_SIZE
        first = pl.cast(pl.min(tp_rank * width, num_tokens), pl.INT32)
        count = pl.cast(pl.min(width, num_tokens - first), pl.INT32)
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
        for tile in pl.range(worker, count * (D // 512), 64):
            row = first + tile // (D // 512)
            col = tile % (D // 512) * 512
            acc = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(output_window, peer=group_base + peer, offsets=[row, col], shape=[1, 512])
                acc = pl.add(acc, value)
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row - first, col], output)
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


@pl.jit.inline(auto_scope=False)
def decode_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.FP32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[OUTPUT_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    reduce_scatter: pl.constexpr = False,
):
    first = pl.cast(0, pl.INT32)
    count = pl.cast(num_tokens, pl.INT32)
    if reduce_scatter:
        width = (num_tokens + TP_SIZE - 1) // TP_SIZE
        first = pl.cast(pl.min(tp_rank * width, num_tokens), pl.INT32)
        count = pl.cast(pl.min(width, num_tokens - first), pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                            cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_publish", deps=[reuse_tid]) as publish_tid:
        if num_tokens > 0:
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
        for peer in pl.range(TP_SIZE):
            pld.system.notify(output_arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                              value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_tp_consumed", deps=[reduce_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2,
                            cmp=pld.WaitCmp.Ge)
    return output


__all__ = [
    "decode_tp_output_all_reduce", "prefill_tp_output_all_reduce",
    "golden_tp_output_all_reduce",
]


if __name__ == "__main__":
    torch.manual_seed(17)
    partials = torch.randn(4, 7, 5)
    reduced = golden_tp_output_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.float().sum(dim=0).to(partials.dtype))
    print(f"[GOLDEN] PASS attention pure-TP all-reduce output={tuple(reduced.shape)}")
