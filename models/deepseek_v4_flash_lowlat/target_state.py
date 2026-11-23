# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Target transaction and request-state publication."""

import pypto.language as pl
import pypto.language.distributed as pld


@pl.jit.inline(auto_scope=False)
def prepare_device_inputs_after(
    request_state: pl.Tensor[[4], pl.INT64],
    input_ids: pl.Tensor[[6], pl.INT64],
    position_ids: pl.Tensor[[6], pl.INT32],
    kv_seq_lens: pl.Tensor[[1], pl.INT32],
    sampling_positions: pl.Tensor[[6], pl.INT32],
    bridge_epoch: pl.Tensor[[1], pl.INT32],
    request_snapshot: pl.Tensor[[4], pl.INT64],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="s6_metadata_from_device_state", deps=[prior_cycle_gate]
    ) as ready_tid:
        pl.system.cacheinvalid()
        epoch = pl.read(request_state, [0])
        anchor = pl.read(request_state, [1])
        last_sampled = pl.read(request_state, [3])
        for field in pl.range(4):
            pl.write(request_snapshot, [field], pl.read(request_state, [field]))
        for token in pl.range(6):
            position = pl.cast(anchor + 1 + token, pl.INT32)
            pl.write(position_ids, [token], position)
            pl.write(sampling_positions, [token], position)
            pl.write(input_ids, [token], pl.const(0, pl.INT64))
        pl.write(input_ids, [0], last_sampled)
        pl.write(kv_seq_lens, [0], pl.cast(anchor + 7, pl.INT32))
        pl.write(bridge_epoch, [0], pl.cast(epoch, pl.INT32))
        pl.system.cacheinvalid()
        pl.system.fence()
    return input_ids, position_ids, kv_seq_lens, sampling_positions, bridge_epoch, request_snapshot, ready_tid


@pl.jit.inline(auto_scope=False)
def device_input_bridge(
    anchor_input: pl.Tensor[[6], pl.INT64],
    markov_ids: pl.Tensor[[5], pl.INT32],
    expected_epoch: pl.Tensor[[1], pl.INT32],
    prepared_ids: pl.Tensor[[6], pl.INT64],
    observed_epoch: pl.Tensor[[1], pl.INT32],
    ids_window: pld.DistributedTensor[[8], pl.INT64],
    ready_signal: pld.DistributedTensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    peers: pl.Scalar[pl.INT32],
):
    packed = pl.create_tensor([8], dtype=pl.INT64)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_compose_publish") as publish_tid:
        if my_rank == 0:
            epoch = pl.read(expected_epoch, [0])
            pl.write(packed, [0], pl.read(anchor_input, [0]))
            for k in pl.range(5):
                token = pl.read(markov_ids, [k])
                pl.write(packed, [k + 1], pl.cast(token, pl.INT64))
            pl.write(packed, [6], pl.cast(epoch, pl.INT64))
            pl.write(packed, [7], pl.cast(0, pl.INT64))
            # Scalar GM stores use D-cache; publish before MTE2 TPUT reads.
            pl.system.cacheinvalid()
            pl.system.fence()
            for peer in pl.range(peers):
                pld.tensor.put(
                    dst=ids_window, peer=peer, src=packed, dst_offsets=[0], src_offsets=[0], shape=[8]
                )
                pld.system.notify(
                    target=ready_signal, peer=peer, offsets=[0], value=epoch, op=pld.NotifyOp.Set
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="bridge_wait") as wait_tid:
        epoch = pl.read(expected_epoch, [0])
        pld.system.defer_wait(signal=ready_signal, offsets=[0], expected=epoch, cmp=pld.WaitCmp.Ge)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="bridge_readback", deps=[publish_tid, wait_tid]
    ) as ready_tid:
        # defer_wait resolves in the scheduler, outside this consumer kernel.
        pl.system.cacheinvalid()
        for token in pl.range(6):
            pl.write(prepared_ids, [token], pl.read(ids_window, [token]))
        pl.write(observed_epoch, [0], pl.cast(pl.read(ids_window, [6]), pl.INT32))
    return prepared_ids, ready_tid


@pl.jit.inline(auto_scope=False)
def device_commit_inline(
    draft_ids: pl.Tensor[[5], pl.INT32],
    sampled_ids: pl.Tensor[[6, 8], pl.INT32],
    target_features: pl.Tensor[[6, 12288], pl.BF16],
    request_state: pl.Tensor[[4], pl.INT64],
    history_ring: pl.InOut[pl.Tensor[[136, 12288], pl.BF16]],
    control: pl.Out[pl.Tensor[[16], pl.INT64]],
    emitted_ids: pl.Out[pl.Tensor[[6], pl.INT64]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="greedy_accept_and_metadata") as decision:
        epoch = pl.read(request_state, [0])
        anchor = pl.read(request_state, [1])
        remaining = pl.read(request_state, [2])
        accepted = pl.const(0, pl.INDEX)
        matching = pl.const(1, pl.INT32)
        for step in pl.range(5):
            if matching != 0:
                if pl.read(draft_ids, [step]) == pl.read(sampled_ids, [step, 0]):
                    accepted = accepted + 1
                else:
                    matching = pl.const(0, pl.INT32)
        advance = pl.cast(accepted + 1, pl.INT64)
        actual = advance
        if remaining < advance:
            actual = remaining
        bonus = pl.cast(pl.read(sampled_ids, [accepted, 0]), pl.INT64)
        last_sampled = pl.read(request_state, [3])
        for step in pl.range(6):
            value = pl.const(-1, pl.INT64)
            if step < actual:
                if step < accepted:
                    value = pl.cast(pl.read(draft_ids, [step]), pl.INT64)
                else:
                    value = bonus
                last_sampled = value
            pl.write(emitted_ids, [step], value)
        for field in pl.range(16):
            pl.write(control, [field], pl.const(0, pl.INT64))
        pl.write(control, [0], epoch)
        pl.write(control, [2], anchor)
        pl.write(control, [3], pl.cast(accepted, pl.INT64))
        pl.write(control, [4], advance)
        pl.write(control, [5], actual)
        pl.write(control, [6], anchor + advance)
        pl.write(control, [7], bonus)
        pl.write(control, [8], last_sampled)
        pl.write(control, [9], remaining - actual)
        pl.write(control, [11], anchor + pl.const(1, pl.INT64))
        pl.write(control, [12], advance)
        if remaining == actual:
            pl.write(control, [13], pl.const(1, pl.INT64))
    with pl.spmd(6 * 24, name_hint="commit_target_features_to_history", deps=[decision]) as copies:
        # The decision task publishes scalar metadata at its task boundary.
        pl.system.cacheinvalid()
        block = pl.tile.get_block_idx()
        row = block // 24
        column = (block % 24) * 512
        advance = pl.read(control, [4])
        start = pl.read(control, [11])
        if row < advance:
            destination = (start + row) % 128
            feature = pl.load(target_features, [row, column], [1, 512])
            pl.store(feature, [destination, column], history_ring)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="publish_history_commit_ack", deps=[copies]
    ) as commit_ack_tid:
        pl.write(control, [1], pl.const(1, pl.INT64))
        pl.write(control, [10], pl.const(1, pl.INT64))
    return history_ring, control, emitted_ids, commit_ack_tid


@pl.jit.inline(auto_scope=False)
def publish_next_request_state(
    control: pl.Tensor[[16], pl.INT64],
    request_state: pl.Tensor[[4], pl.INT64],
    current_epoch: pl.Tensor[[1], pl.INT32],
    state_window: pld.DistributedTensor[[4], pl.INT64],
    state_signal: pld.DistributedTensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    peers: pl.Scalar[pl.INT32],
    commit_ack: pl.Scalar[pl.TASK_ID],
):
    packed = pl.create_tensor([4], dtype=pl.INT64)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="s6_publish_next_state", deps=[commit_ack]
    ) as publish_tid:
        if my_rank == 0:
            pl.system.cacheinvalid()
            epoch = pl.read(current_epoch, [0])
            pl.write(packed, [0], pl.read(control, [0]) + 1)
            pl.write(packed, [1], pl.read(control, [6]))
            pl.write(packed, [2], pl.read(control, [9]))
            pl.write(packed, [3], pl.read(control, [8]))
            pl.system.cacheinvalid()
            pl.system.fence()
            for peer in pl.range(peers):
                pld.tensor.put(
                    dst=state_window, peer=peer, src=packed, dst_offsets=[0], src_offsets=[0], shape=[4]
                )
                pld.system.notify(
                    target=state_signal, peer=peer, offsets=[0], value=epoch, op=pld.NotifyOp.Set
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s6_wait_next_state") as wait_tid:
        epoch = pl.read(current_epoch, [0])
        pld.system.defer_wait(signal=state_signal, offsets=[0], expected=epoch, cmp=pld.WaitCmp.Ge)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="s6_finalize_next_state", deps=[publish_tid, wait_tid]
    ) as ready_tid:
        pl.system.cacheinvalid()
        for field in pl.range(4):
            pl.write(request_state, [field], pl.read(state_window, [field]))
        pl.system.cacheinvalid()
        pl.system.fence()
    return request_state, ready_tid


@pl.jit.inline(auto_scope=False)
def eight_rank_reuse_gate(
    current_epoch: pl.Tensor[[1], pl.INT32],
    arrival_signal: pld.DistributedTensor[[8, 32], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    local_reuse_ready: pl.Scalar[pl.TASK_ID],
):
    # One sender owns each row. Signals retain their values; epochs must never
    # rewind when a benchmark restores the logical prefix/cache snapshot.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="cycle_publish_local_exit", deps=[local_reuse_ready]
    ) as publish_tid:
        pl.system.cacheinvalid()
        epoch = pl.read(current_epoch, [0])
        for peer in pl.range(8):
            if peer != my_rank:
                pld.system.notify(
                    target=arrival_signal, peer=peer, offsets=[my_rank, 0], value=epoch, op=pld.NotifyOp.Set
                )
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="cycle_wait_all_local_exits", deps=[publish_tid]
    ) as wait_tid:
        epoch = pl.read(current_epoch, [0])
        for peer in pl.range(8):
            if peer != my_rank:
                pld.system.defer_wait(
                    signal=arrival_signal, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge
                )
    # This depends on deferred completion, not just the wait kernel's END.
    global_reuse_ready = pl.system.task_dummy(deps=[publish_tid, wait_tid])
    return global_reuse_ready
