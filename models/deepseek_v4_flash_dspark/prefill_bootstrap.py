# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish terminal-prefill tokens into the persistent decode-state ABI."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP, TP

REQUESTS_DYN = pl.dynamic("DSPARK_BOOTSTRAP_REQUESTS_DYN")
LEASES_DYN = pl.dynamic("DSPARK_BOOTSTRAP_LEASES_DYN")
MAX_REQUESTS = 128
DESCRIPTOR_WIDTH = 11
PAYLOAD_WIDTH = 16
NOISE_TOKEN_ID = 128799


@pl.jit
def prepare_bootstrap_token_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[MAX_REQUESTS, 8], pl.INT32],
    next_tokens: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT64]],
    tp_rank: pl.Scalar[pl.INT32],
):
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    next_tokens.bind_dynamic(0, REQUESTS_DYN)
    requests = pl.tensor.dim(descriptors, 0)
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_token"):
        for request in pl.range(requests):
            pl.write(next_tokens, [request], pl.cast(0, pl.INT64))
            length = pl.read(descriptors, [request, 4])
            start = pl.read(descriptors, [request, 3])
            prompt = pl.read(descriptors, [request, 5])
            row = pl.read(descriptors, [request, 9])
            flags = pl.read(descriptors, [request, 10])
            if tp_rank == 0 and length > 0 and row >= 0 and row < MAX_REQUESTS:
                if start + length >= prompt and flags % 2 == 1:
                    pl.write(next_tokens, [request], pl.cast(pl.read(sampled_ids, [row, 0]), pl.INT64))
    return next_tokens


@pl.jit.host
def l3_prepare_bootstrap_tokens(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[EP, MAX_REQUESTS, 8], pl.INT32],
    next_tokens: pl.Out[pl.Tensor[[EP, REQUESTS_DYN], pl.INT64]],
):
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    next_tokens.bind_dynamic(1, REQUESTS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_bootstrap_token_rank(descriptors[rank], sampled_ids[rank], next_tokens[rank], rank % TP, device=rank)


@pl.jit
def publish_bootstrap_rank(
    descriptors: pl.Tensor[[REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[MAX_REQUESTS, 8], pl.INT32],
    drafts: pl.Tensor[[REQUESTS_DYN, 7], pl.INT32],
    tokens: pl.InOut[pl.Tensor[[LEASES_DYN, 8], pl.INT64]],
    meta: pl.InOut[pl.Tensor[[LEASES_DYN, 6], pl.INT32]],
    window: pl.InOut[pld.DistributedTensor[[TP * MAX_REQUESTS, PAYLOAD_WIDTH], pl.INT64]],
    signal: pl.InOut[pld.DistributedTensor[[TP, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    descriptors.bind_dynamic(0, REQUESTS_DYN)
    drafts.bind_dynamic(0, REQUESTS_DYN)
    tokens.bind_dynamic(0, LEASES_DYN)
    meta.bind_dynamic(0, LEASES_DYN)
    requests = pl.tensor.dim(descriptors, 0)
    leases = pl.tensor.dim(meta, 0)
    payload = pl.create_tensor([requests, PAYLOAD_WIDTH], dtype=pl.INT64)
    received = pl.create_tensor([requests, PAYLOAD_WIDTH], dtype=pl.INT64)
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_payload"):
        for request in pl.range(requests):
            for column in pl.range(PAYLOAD_WIDTH):
                pl.write(payload, [request, column], pl.cast(0, pl.INT64))
            slot = pl.read(descriptors, [request, 0])
            generation = pl.read(descriptors, [request, 1])
            start = pl.read(descriptors, [request, 3])
            length = pl.read(descriptors, [request, 4])
            prompt = pl.read(descriptors, [request, 5])
            sample_row = pl.read(descriptors, [request, 9])
            flags = pl.read(descriptors, [request, 10])
            if tp_rank == 0 and slot >= 0 and slot < leases and length > 0:
                if start + length >= prompt and flags % 2 == 1 and sample_row >= 0 and sample_row < MAX_REQUESTS:
                    resident_generation = pl.read(meta, [slot, 1])
                    valid = pl.read(meta, [slot, 0])
                    limit = pl.read(meta, [slot, 5])
                    sampled = pl.read(sampled_ids, [sample_row, 0])
                    if resident_generation == generation and valid == 0 and sampled >= 0:
                        count = pl.cast(0, pl.INT32)
                        if flags // 2 % 2 == 1 and prompt + 6 < limit:
                            count = pl.cast(7, pl.INT32)
                        pl.write(payload, [request, 0], pl.cast(slot, pl.INT64))
                        pl.write(payload, [request, 1], pl.cast(generation, pl.INT64))
                        pl.write(payload, [request, 2], pl.cast(prompt, pl.INT64))
                        pl.write(payload, [request, 3], pl.cast(count, pl.INT64))
                        pl.write(payload, [request, 4], pl.cast(sampled, pl.INT64))
                        for offset in pl.range(7):
                            token = pl.cast(NOISE_TOKEN_ID, pl.INT64)
                            if count > 0:
                                token = pl.cast(pl.read(drafts, [request, offset]), pl.INT64)
                            pl.write(payload, [request, 5 + offset], token)
                        pl.write(payload, [request, 12], pl.cast(1, pl.INT64))
    with pl.spmd(TP, name_hint="dspark_bootstrap_push", allow_early_resolve=True) as push_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=window, peer=group_base + peer, src=payload,
            dst_offsets=[tp_rank * MAX_REQUESTS, 0], src_offsets=[0, 0],
            shape=[requests, PAYLOAD_WIDTH], chunk_rows=1, chunk_cols=PAYLOAD_WIDTH,
        )
        if peer != tp_rank:
            pld.system.notify(
                target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_wait") as wait_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.defer_wait(
                    signal=signal, offsets=[peer, 0], expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_read", deps=[push_tid, wait_tid],
    ) as read_tid:
        for row in pl.range(requests):
            received[row:row + 1, :] = pl.slice(window, [1, PAYLOAD_WIDTH], [row, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_ack", deps=[read_tid]) as ack_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.notify(
                    target=signal, peer=group_base + peer, offsets=[tp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_ack_wait") as ack_wait_tid:
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.defer_wait(
                    signal=signal, offsets=[peer, 0], expected=pl.cast(2, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_retire", deps=[ack_tid, ack_wait_tid],
    ):
        anchor = pl.read(received, [0, 0])
        for peer in pl.range(TP):
            if peer != tp_rank:
                pld.system.notify(
                    target=signal, peer=group_base + tp_rank, offsets=[peer, 0],
                    value=pl.cast(-2, pl.INT32), op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(received, [0, 0], anchor)
    for _core in pl.spmd(1, name_hint="dspark_bootstrap_commit"):
        for request in pl.range(requests):
            ready = pl.read(received, [request, 12])
            slot = pl.cast(pl.read(received, [request, 0]), pl.INT32)
            if ready == 1 and slot >= 0 and slot < leases:
                generation = pl.cast(pl.read(received, [request, 1]), pl.INT32)
                resident_generation = pl.read(meta, [slot, 1])
                valid = pl.read(meta, [slot, 0])
                if generation == resident_generation and valid == 0:
                    pl.write(tokens, [slot, 0], pl.read(received, [request, 4]))
                    for offset in pl.range(7):
                        pl.write(tokens, [slot, 1 + offset], pl.read(received, [request, 5 + offset]))
                    prompt = pl.cast(pl.read(received, [request, 2]), pl.INT32)
                    pl.write(meta, [slot, 2], prompt)
                    pl.write(meta, [slot, 3], prompt)
                    pl.write(meta, [slot, 4], pl.cast(pl.read(received, [request, 3]), pl.INT32))
                    pl.write(meta, [slot, 0], pl.cast(1, pl.INT32))
    return tokens, meta, signal


@pl.jit.host
def l3_publish_bootstrap(
    descriptors: pl.Tensor[[EP, REQUESTS_DYN, DESCRIPTOR_WIDTH], pl.INT32],
    sampled_ids: pl.Tensor[[EP, MAX_REQUESTS, 8], pl.INT32],
    drafts: pl.Tensor[[EP, REQUESTS_DYN, 7], pl.INT32],
    tokens: pl.InOut[pl.Tensor[[EP, LEASES_DYN, 8], pl.INT64]],
    meta: pl.InOut[pl.Tensor[[EP, LEASES_DYN, 6], pl.INT32]],
):
    descriptors.bind_dynamic(1, REQUESTS_DYN)
    drafts.bind_dynamic(1, REQUESTS_DYN)
    tokens.bind_dynamic(1, LEASES_DYN)
    meta.bind_dynamic(1, LEASES_DYN)
    window_buffer = pld.alloc_window_buffer([TP * MAX_REQUESTS, PAYLOAD_WIDTH], dtype=pl.INT64)
    signal_buffer = pld.alloc_window_buffer([TP, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        window = pld.window(window_buffer, [TP * MAX_REQUESTS, PAYLOAD_WIDTH], dtype=pl.INT64)
        signal = pld.window(signal_buffer, [TP, 1], dtype=pl.INT32)
        publish_bootstrap_rank(
            descriptors[rank], sampled_ids[rank], drafts[rank], tokens[rank], meta[rank],
            window, signal, rank // TP * TP, rank % TP, device=rank,
        )
