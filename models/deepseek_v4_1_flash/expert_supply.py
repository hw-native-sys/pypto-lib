# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Bind V4.1's six packed fields to the shared AIV supply nodes."""

import pypto.language as pl
import pypto.language.distributed as pld
from config import EP_SIZE, FLASH, MOE_TOKENS
from models.common.expert_supply import mapped_copy_workers, supply_packed_wave

WEIGHT_WORKERS = mapped_copy_workers(FLASH.hidden_size * FLASH.moe_intermediate_size // 2)
SCALE_WORKERS = mapped_copy_workers(FLASH.hidden_size * FLASH.moe_intermediate_size // 32)


@pl.jit.inline(auto_scope=False)
def supply_experts(counts: pl.Tensor, state: pl.Tensor, descriptors: pl.Tensor,
                   w1: pl.Tensor, s1: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor,
                   w3: pl.Tensor, s3: pl.Tensor, layer: pl.Scalar[pl.INDEX],
                   inputs_ready: pl.Scalar[pl.TASK_ID], experts: pl.constexpr):
    slots = pl.create_tensor([experts], dtype=pl.INT32)
    misses = pl.create_tensor([experts], dtype=pl.INT32)
    slot_map, bank1, scale1, bank2, scale2, bank3, scale3, ready = supply_packed_wave(
        counts, state, descriptors, slots, misses, w1, s1, w2, s2, w3, s3, layer,
        0, experts, inputs_ready, WEIGHT_WORKERS, SCALE_WORKERS)
    return slot_map, bank1, scale1, bank2, scale2, bank3, scale3, ready


@pl.jit.inline(auto_scope=False)
def publish_routed_counts(
    indices: pl.Tensor, counts: pl.Tensor, recv_meta_local: pl.Tensor,
    recv_meta: pld.DistributedTensor, data_arrived: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr, topk: pl.constexpr,
):
    flat_counts = pl.reshape(counts, [1, local_experts])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_count_publish", allow_early_resolve=False) as published:
        values = pl.tile.full([1, EP_SIZE * local_experts], dtype=pl.INT32, value=0)
        ids = pl.load(indices, [0, 0], [MOE_TOKENS, 8], valid_shape=[MOE_TOKENS, topk])
        for token in pl.range(num_tokens):
            for k in pl.range(topk):
                expert = pl.tile.read(ids, [token, k])
                previous = pl.tile.read(values, [0, expert])
                pl.tile.write(values, [0, expert], previous + pl.const(1, pl.INT32))
        if EP_SIZE == 1:
            pl.store(values, [0, 0], flat_counts)
            pl.store(values, [0, 0], recv_meta_local)
        else:
            for peer in pl.range(EP_SIZE):
                part = pl.tile.slice(values, [1, local_experts], [0, peer * local_experts])
                pld.tile.remote_store(part, target=recv_meta, peer=peer, offsets=[rank, 0])
                if peer != rank:
                    # Metadata and payload signals occupy separate 64-byte cache lines.
                    pld.system.notify(target=data_arrived, peer=peer, offsets=[rank, 0, 16], value=epoch, op=pld.NotifyOp.Set)
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = published
    if EP_SIZE > 1:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_count_wait", deps=[published]) as received:
            for src in pl.range(EP_SIZE):
                if src != rank:
                    pld.system.defer_wait(signal=data_arrived, offsets=[src, 0, 16], expected=epoch, cmp=pld.WaitCmp.Ge)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_count_sum", deps=[received]) as summed:
            for expert in pl.range(local_experts):
                total = pl.const(0, pl.INT32)
                for src in pl.range(EP_SIZE):
                    count = pl.read(recv_meta, [src, expert])
                    pl.write(recv_meta_local, [src, expert], count)
                    total = total + count
                pl.write(counts, [expert, 0], total)
        completion[0] = summed
    return completion[0]
