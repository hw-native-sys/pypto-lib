# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""EPLB ABI: replica routing, per-slot load publication, and resident migration.

Frozen conventions (serving checks them through ``EPLB_ABI_VERSION``):

- placement: ``logical_to_physical[bank_layers * E, R]`` and
  ``replica_count[bank_layers * E]``; layer ``L`` reads bank ``L % bank_layers``.
- routing: ``make_moe_eplb`` maps the gate's logical top-k ids to physical
  slots. Every route of an invalid token row stays ``-1``; dispatch, expert
  scatter, and combine all skip negative routes.
- load: ``publish_counts`` writes ``[bank_layer, 0:C]`` of a
  ``[bank_layers, C_PAD]`` history when ``control[1]`` is nonzero, optionally
  accumulating into the slot counts of the same wave.
- migration: ``make_eplb_transfer`` moves the six resident expert tensors
  ``[R, C, ...]`` through receive windows and two completion barriers. Serving
  dispatches every migrated layer on one persistent L3 worker with retained
  CommDomains and ``reset_persistent_windows=False``, so each peer's signal
  count keeps growing across calls: the caller passes ``epoch`` >= 1, the
  1-based call number.
"""

from math import gcd

import pypto.language as pl
import pypto.language.distributed as pld

EPLB_ABI_VERSION = 1

# An a3 allocation exposes 24 AIC / 48 AIV; a larger SPMD launch would queue
# blocks on the same cores instead of adding parallelism. Each worker strides
# over its share of the token blocks so the tail is covered either way.
MAX_ROUTE_WORKERS = 48


# === Routing =================================================================


def make_eplb_routing(*, logical_experts: int, ranks: int, tokens: int, top_k: int,
                      bank_layers: int = 1, boolean_mask: bool = True, single_replica: bool = False):
    """Specialize replica selection independently from the gate's score/hash policy."""
    if min(logical_experts, ranks, tokens, top_k, bank_layers) <= 0 or top_k > 8:
        raise ValueError("EPLB routing requires positive dimensions and cache-line aligned route rows")
    E = logical_experts
    R = ranks
    T = tokens
    K = top_k
    BANK_ROWS = bank_layers * E
    # tiling
    TOKEN_TILE = min(T, 16 // gcd(K, 16))
    K_TILE = 8
    if T % TOKEN_TILE:
        raise ValueError("EPLB token blocks must own complete 64-byte route spans")
    TOKEN_BLOCKS = T // TOKEN_TILE
    ROUTE_WORKERS = min(TOKEN_BLOCKS, MAX_ROUTE_WORKERS)

    @pl.jit.inline
    def route_experts(
        logical_ids: pl.Tensor[[T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        replica_count: pl.Tensor[[BANK_ROWS], pl.INT32],
        valid_rows: pl.Tensor[[T], pl.INT32],
        physical_ids: pl.Tensor[[T, K], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        route_epoch: pl.Scalar[pl.INT32],
        layer_id: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(ROUTE_WORKERS, name_hint="eplb_select_replica"):
            worker = pl.tile.get_block_idx()
            for token_block in pl.range(worker, TOKEN_BLOCKS, ROUTE_WORKERS):
                token0 = token_block * TOKEN_TILE
                logical_tile = pl.tile.load(
                    logical_ids, [token0, 0], [TOKEN_TILE, K_TILE], valid_shape=[TOKEN_TILE, K])
                physical_tile = pl.tile.full([TOKEN_TILE, K_TILE], dtype=pl.INT32, value=-1)
                for row in pl.range(TOKEN_TILE):
                    token = token0 + row
                    row_marker = pl.read(valid_rows, [token])
                    if boolean_mask:
                        valid = row_marker != 0
                    else:
                        valid = row_marker >= 0
                    if token < num_tokens and valid:
                        for route in pl.range(K):
                            expert = pl.tile.read(logical_tile, [row, route])
                            bank_expert = (layer_id % bank_layers) * E + expert
                            # With no redundant slots, coverage of every logical
                            # expert guarantees exactly one replica even after a
                            # rebalance. Avoid a dynamic count read and remainder
                            # for every route; the placement lookup still applies.
                            if single_replica:
                                replica = 0
                            else:
                                count = pl.read(replica_count, [bank_expert])
                                seed = token + my_rank * T + route + route_epoch
                                replica = seed % count
                            physical = pl.read(logical_to_physical, [bank_expert, replica])
                            pl.tile.write(physical_tile, [row, route], physical)
                physical_valid = pl.set_validshape(physical_tile, TOKEN_TILE, K)
                pl.tile.store(physical_valid, [token0, 0], physical_ids)
        return physical_ids

    return route_experts


def make_moe_eplb(*, enabled: bool, logical_experts: int, ranks: int, tokens: int,
                  top_k: int, local_slots: int, bank_layers: int):
    """Bind route control and padded load publication for one MoE capacity."""
    E, R, T, K = logical_experts, ranks, tokens, top_k
    C = local_slots
    C_PAD = (C + 15) // 16 * 16
    BANK_ROWS = bank_layers * E
    route_experts = make_eplb_routing(
        logical_experts=E, ranks=R, tokens=T, top_k=K,
        bank_layers=bank_layers, boolean_mask=False, single_replica=C * R == E,
    ) if enabled else None

    @pl.jit.inline
    def mapped_routes(
        indices: pl.Tensor[[T, K], pl.INT32],
        mapping: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        counts: pl.Tensor[[BANK_ROWS], pl.INT32],
        source_rows: pl.Tensor[[T], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        physical = pl.create_tensor([T, K], dtype=pl.INT32)
        epoch = pl.read(control, [0])
        route_experts(indices, mapping, counts, source_rows, physical, rank, epoch, layer, num_tokens)
        return physical

    @pl.jit.inline
    def identity_routes(
        indices: pl.Tensor[[T, K], pl.INT32],
        mapping: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        counts: pl.Tensor[[BANK_ROWS], pl.INT32],
        source_rows: pl.Tensor[[T], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        return indices

    @pl.jit.inline
    def publish_counts(
        counts: pl.Tensor[[C, 1], pl.INT32],
        output: pl.Tensor[[bank_layers, C_PAD], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        accumulate: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_export_load", deps=[ready]):
            bank_layer = layer % bank_layers
            row = pl.tile.full([1, C_PAD], dtype=pl.INT32, value=0)
            if pl.read(control, [1]) != 0:
                for expert in pl.range(C):
                    value = pl.read(counts, [expert, 0])
                    if accumulate != 0:
                        value = value + pl.read(output, [bank_layer, expert])
                    pl.tile.write(row, [0, expert], value)
            pl.tile.store(row, [bank_layer, 0], output)
        return counts

    @pl.jit.inline
    def discard_counts(
        counts: pl.Tensor[[C, 1], pl.INT32],
        output: pl.Tensor[[bank_layers, C_PAD], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        accumulate: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        return counts

    return (mapped_routes, publish_counts) if enabled else (identity_routes, discard_counts)


# === Migration ===============================================================


def _eplb_transfer_dims(ranks, local_slots, hidden_size, intermediate_size):
    if min(ranks, local_slots, hidden_size, intermediate_size) <= 0:
        raise ValueError("EPLB transfer dimensions must be positive")
    return ranks, local_slots, hidden_size * intermediate_size, intermediate_size, hidden_size


def _make_eplb_transfer_layer(*, ranks: int, local_slots: int, hidden_size: int, intermediate_size: int):
    """Specialize the six-tensor migration kernel to one resident weight bank."""
    R, C, W, I, D = _eplb_transfer_dims(ranks, local_slots, hidden_size, intermediate_size)
    # tiling
    COPY_COL_TILE = 8192

    @pl.jit
    def eplb_transfer_layer(
        w1: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w1_scale: pl.InOut[pl.Tensor[[C, I], pl.FP32]],
        w3: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w3_scale: pl.InOut[pl.Tensor[[C, I], pl.FP32]],
        w2: pl.InOut[pl.Tensor[[C, W], pl.INT8]],
        w2_scale: pl.InOut[pl.Tensor[[C, D], pl.FP32]],
        sources: pl.Tensor[[R, C, 2], pl.INT32],
        changed: pl.Tensor[[R, C], pl.INT32],
        receive_w1: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w1_scale: pld.DistributedTensor[[C, I], pl.FP32],
        receive_w3: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w3_scale: pld.DistributedTensor[[C, I], pl.FP32],
        receive_w2: pld.DistributedTensor[[C, W], pl.INT8],
        receive_w2_scale: pld.DistributedTensor[[C, D], pl.FP32],
        sources_done: pld.DistributedTensor[[R, 1], pl.INT32],
        installed: pld.DistributedTensor[[R, 1], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        # The two signal windows keep their notification counts across calls:
        # serving runs one persistent L3 worker with retained CommDomains and
        # `reset_persistent_windows=False` (npu_runner._shared_l3_worker), and
        # the benchmark replay does the same. Each peer adds exactly one
        # notification per call, so the caller passes the 1-based call number.
        # All resident reads finish before any resident write, including local moves.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_read_sources") as push_tid:
            for peer in pl.range(R):
                for slot in pl.range(C):
                    move = pl.read(changed, [peer, slot])
                    source_rank = pl.read(sources, [peer, slot, 0])
                    source_slot = pl.read(sources, [peer, slot, 1])
                    if move != 0 and source_rank == my_rank:
                        pld.tensor.put(
                            receive_w1, peer, w1,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w1_scale, peer, w1_scale,
                            [slot, 0], [source_slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w3, peer, w3,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w3_scale, peer, w3_scale,
                            [slot, 0], [source_slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w2, peer, w2,
                            [slot, 0], [source_slot, 0], [1, W], chunk_cols=COPY_COL_TILE,
                        )
                        pld.tensor.put(
                            receive_w2_scale, peer, w2_scale,
                            [slot, 0], [source_slot, 0], [1, D], chunk_cols=COPY_COL_TILE,
                        )
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.notify(sources_done, peer, [my_rank, 0], 1, op=pld.NotifyOp.AtomicAdd)

        # Anchor the deferred wait after every local push/notify.  Without this
        # dependency the scheduler may submit the wait before the matching
        # epoch signal is published, which can leave the transfer lane in an
        # all-idle state on a first real migration.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_wait_sources", deps=[push_tid]) as read_wait_tid:
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.defer_wait(sources_done, [peer, 0], epoch, cmp=pld.WaitCmp.Ge)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_install", deps=[push_tid, read_wait_tid]) as install_tid:
            for slot in pl.range(C):
                if pl.read(changed, [my_rank, slot]) != 0:
                    pld.tensor.get(w1, my_rank, receive_w1, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w1_scale, my_rank, receive_w1_scale,
                        [slot, 0], [slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                    )
                    pld.tensor.get(w3, my_rank, receive_w3, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w3_scale, my_rank, receive_w3_scale,
                        [slot, 0], [slot, 0], [1, I], chunk_cols=COPY_COL_TILE,
                    )
                    pld.tensor.get(w2, my_rank, receive_w2, [slot, 0], [slot, 0], [1, W], chunk_cols=COPY_COL_TILE)
                    pld.tensor.get(
                        w2_scale, my_rank, receive_w2_scale,
                        [slot, 0], [slot, 0], [1, D], chunk_cols=COPY_COL_TILE,
                    )
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.notify(installed, peer, [my_rank, 0], 1, op=pld.NotifyOp.AtomicAdd)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_wait_install", deps=[install_tid]) as wait_install_tid:
            for peer in pl.range(R):
                if peer != my_rank:
                    pld.system.defer_wait(installed, [peer, 0], epoch, cmp=pld.WaitCmp.Ge)

        return w1, w1_scale, w3, w3_scale, w2, w2_scale

    return eplb_transfer_layer


def make_eplb_transfer(*, ranks: int, local_slots: int, hidden_size: int, intermediate_size: int):
    """EPLB HOST ABI v1: one migration call whose `epoch` is the 1-based call number."""
    R, C, W, I, D = _eplb_transfer_dims(ranks, local_slots, hidden_size, intermediate_size)
    eplb_transfer_layer = _make_eplb_transfer_layer(
        ranks=ranks, local_slots=local_slots,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
    )

    @pl.jit.host
    def l3_eplb_transfer_layer(
        w1: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w1_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w3: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w3_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w2: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w2_scale: pl.InOut[pl.Tensor[[R, C, D], pl.FP32]],
        sources: pl.Tensor[[R, R, C, 2], pl.INT32],
        changed: pl.Tensor[[R, R, C], pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        """EPLB HOST ABI v1: six-tensor migration and completion epochs."""
        receive_w1_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w1_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w3_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w3_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w2_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w2_scale_buf = pld.alloc_window_buffer([C, D], dtype=pl.FP32)
        sources_done_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        installed_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        for rank in pl.range(R):
            receive_w1 = pld.window(receive_w1_buf, [C, W], dtype=pl.INT8)
            receive_w1_scale = pld.window(receive_w1_scale_buf, [C, I], dtype=pl.FP32)
            receive_w3 = pld.window(receive_w3_buf, [C, W], dtype=pl.INT8)
            receive_w3_scale = pld.window(receive_w3_scale_buf, [C, I], dtype=pl.FP32)
            receive_w2 = pld.window(receive_w2_buf, [C, W], dtype=pl.INT8)
            receive_w2_scale = pld.window(receive_w2_scale_buf, [C, D], dtype=pl.FP32)
            sources_done = pld.window(sources_done_buf, [R, 1], dtype=pl.INT32)
            installed = pld.window(installed_buf, [R, 1], dtype=pl.INT32)
            eplb_transfer_layer(
                w1[rank], w1_scale[rank], w3[rank], w3_scale[rank], w2[rank], w2_scale[rank],
                sources[rank], changed[rank],
                receive_w1, receive_w1_scale, receive_w3, receive_w3_scale, receive_w2, receive_w2_scale,
                sources_done, installed,
                rank, epoch, device=rank,
            )
        return w1, w1_scale, w3, w3_scale, w2, w2_scale

    return l3_eplb_transfer_layer


def make_eplb_transfer_rounds(*, rounds: int, ranks: int, local_slots: int,
                              hidden_size: int, intermediate_size: int):
    """Validation entry: `rounds` consecutive migrations on one retained domain.

    Serving calls this HOST entry once per migrated layer on a persistent
    worker whose signal windows are not reset, so the second call must wait on
    the second notification. This entry reproduces that sequence in one host
    call and takes the per-round plans and the first epoch.
    """
    if rounds < 1:
        raise ValueError("EPLB transfer validation needs at least one round")
    R, C, W, I, D = _eplb_transfer_dims(ranks, local_slots, hidden_size, intermediate_size)
    eplb_transfer_layer = _make_eplb_transfer_layer(
        ranks=ranks, local_slots=local_slots,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
    )

    @pl.jit.host
    def l3_eplb_transfer_rounds(
        w1: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w1_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w3: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w3_scale: pl.InOut[pl.Tensor[[R, C, I], pl.FP32]],
        w2: pl.InOut[pl.Tensor[[R, C, W], pl.INT8]],
        w2_scale: pl.InOut[pl.Tensor[[R, C, D], pl.FP32]],
        sources: pl.Tensor[[R, rounds, R, C, 2], pl.INT32],
        changed: pl.Tensor[[R, rounds, R, C], pl.INT32],
        epoch0: pl.Scalar[pl.INT32],
    ):
        receive_w1_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w1_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w3_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w3_scale_buf = pld.alloc_window_buffer([C, I], dtype=pl.FP32)
        receive_w2_buf = pld.alloc_window_buffer([C, W], dtype=pl.INT8)
        receive_w2_scale_buf = pld.alloc_window_buffer([C, D], dtype=pl.FP32)
        sources_done_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        installed_buf = pld.alloc_window_buffer([R, 1], dtype=pl.INT32)
        for round_idx in pl.range(rounds):
            epoch = epoch0 + pl.cast(round_idx, pl.INT32)
            for rank in pl.range(R):
                receive_w1 = pld.window(receive_w1_buf, [C, W], dtype=pl.INT8)
                receive_w1_scale = pld.window(receive_w1_scale_buf, [C, I], dtype=pl.FP32)
                receive_w3 = pld.window(receive_w3_buf, [C, W], dtype=pl.INT8)
                receive_w3_scale = pld.window(receive_w3_scale_buf, [C, I], dtype=pl.FP32)
                receive_w2 = pld.window(receive_w2_buf, [C, W], dtype=pl.INT8)
                receive_w2_scale = pld.window(receive_w2_scale_buf, [C, D], dtype=pl.FP32)
                sources_done = pld.window(sources_done_buf, [R, 1], dtype=pl.INT32)
                installed = pld.window(installed_buf, [R, 1], dtype=pl.INT32)
                eplb_transfer_layer(
                    w1[rank], w1_scale[rank], w3[rank], w3_scale[rank], w2[rank], w2_scale[rank],
                    sources[rank, round_idx], changed[rank, round_idx],
                    receive_w1, receive_w1_scale, receive_w3, receive_w3_scale, receive_w2, receive_w2_scale,
                    sources_done, installed,
                    rank, epoch, device=rank,
                )
        return w1, w1_scale, w3, w3_scale, w2, w2_scale

    return l3_eplb_transfer_rounds


# === Routing validation ======================================================


def make_routing_test(*, logical_experts: int, ranks: int, tokens: int = 8, top_k: int = 6,
                      single_replica: bool = False, bank_layers: int = 1,
                      boolean_mask: bool = True, layer: int = 0):
    """Build a distributed golden entry for replica IDs and invalid rows."""
    E, R, T, K = logical_experts, ranks, tokens, top_k
    BANK_ROWS = bank_layers * E
    route_experts = make_eplb_routing(logical_experts=E, ranks=R, tokens=T, top_k=K,
                                     bank_layers=bank_layers, boolean_mask=boolean_mask,
                                     single_replica=single_replica)

    @pl.jit
    def routing_test(
        logical_ids: pl.Tensor[[T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        replica_count: pl.Tensor[[BANK_ROWS], pl.INT32],
        valid_rows: pl.Tensor[[T], pl.INT32],
        physical_ids: pl.Out[pl.Tensor[[T, K], pl.INT32]],
        rank: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        route_experts(logical_ids, logical_to_physical, replica_count, valid_rows, physical_ids,
                      rank, epoch, layer, T)
        return physical_ids

    @pl.jit.host
    def l3_routing_test(
        logical_ids: pl.Tensor[[R, T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[R, BANK_ROWS, R], pl.INT32],
        replica_count: pl.Tensor[[R, BANK_ROWS], pl.INT32],
        valid_rows: pl.Tensor[[R, T], pl.INT32],
        physical_ids: pl.Out[pl.Tensor[[R, T, K], pl.INT32]],
        epoch: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(R):
            routing_test(
                logical_ids[rank], logical_to_physical[rank], replica_count[rank], valid_rows[rank],
                physical_ids[rank],
                rank, epoch, device=rank,
            )
        return physical_ids

    return l3_routing_test


def make_golden_routing(*, bank_layers: int = 1, boolean_mask: bool = True, layer: int = 0):
    """Build the torch reference for one routing subcase."""

    def golden_routing(tensors):
        import torch

        logical = tensors["logical_ids"].long()
        inverse = tensors["logical_to_physical"]
        counts = tensors["replica_count"]
        valid = tensors["valid_rows"]
        epoch = int(tensors["epoch"])
        bank_experts = inverse.shape[1] // bank_layers
        result = torch.full_like(logical, -1, dtype=torch.int32)
        ranks, tokens, top_k = logical.shape
        for rank in range(ranks):
            for token in range(tokens):
                row_marker = int(valid[rank, token])
                row_valid = row_marker != 0 if boolean_mask else row_marker >= 0
                if row_valid:
                    for route in range(top_k):
                        expert = int(logical[rank, token, route])
                        bank_expert = (layer % bank_layers) * bank_experts + expert
                        replica = (token + rank * tokens + route + epoch) % int(counts[rank, bank_expert])
                        result[rank, token, route] = inverse[rank, bank_expert, replica]
        tensors["physical_ids"] = result

    return golden_routing


def _routing_placement(physical, logical_experts, ranks, bank_layers):
    """Distinct per-bank placements so a wrong bank lookup cannot pass."""
    import torch

    mapping = torch.empty((bank_layers, physical), dtype=torch.int32)
    for bank in range(bank_layers):
        # Global bank row: bank * E + logical expert, with a per-bank rotation
        # so the two banks place the same expert on different ranks.
        mapping[bank] = (bank * logical_experts
                         + torch.roll(torch.arange(physical) % logical_experts, shifts=bank + 1))
    return mapping.reshape(-1)


def build_routing_specs(*, logical_experts: int, ranks: int, redundant_experts: int, tokens: int = 8,
                        top_k: int = 6, bank_layers: int = 1, boolean_mask: bool = True,
                        layer: int = 0):
    import torch

    from golden import ScalarSpec, TensorSpec

    physical = logical_experts + redundant_experts
    bank_rows = bank_layers * logical_experts
    if logical_experts < 8 or physical % ranks or physical // ranks > logical_experts:
        raise ValueError("Invalid routing fixture capacity")
    # Exercise a changed placement as well as replica selection. In particular,
    # the single-replica specialization must still honor nonidentity mappings.
    mapping = _routing_placement(physical, logical_experts, ranks, bank_layers)
    inverse = torch.full((bank_rows, ranks), -1, dtype=torch.int32)
    counts = torch.bincount(mapping, minlength=bank_rows).to(torch.int32)
    if int(counts.max()) > ranks:
        raise ValueError("A bank row cannot host more replicas than the placement table's rank axis")
    for bank_expert in range(bank_rows):
        locations = torch.nonzero(mapping == bank_expert).flatten().to(torch.int32)
        inverse[bank_expert, :len(locations)] = locations
    ids = torch.arange(ranks * tokens * top_k).reshape(ranks, tokens, top_k).remainder(logical_experts).int()
    valid = torch.ones((ranks, tokens), dtype=torch.int32)
    if boolean_mask:
        valid[:, 1::3] = 0
    else:
        # Production masking: negative is invalid, zero and positive are valid.
        valid[:, 1::3] = -1
        valid[0, 2] = 0
        valid[-1, 0] = 7
    inverse_stacked = inverse.unsqueeze(0).repeat(ranks, 1, 1)
    counts_stacked = counts.unsqueeze(0).repeat(ranks, 1)
    return [
        TensorSpec("logical_ids", [ranks, tokens, top_k], torch.int32, init_value=lambda: ids),
        TensorSpec("logical_to_physical", list(inverse_stacked.shape), torch.int32, init_value=lambda: inverse_stacked),
        TensorSpec("replica_count", list(counts_stacked.shape), torch.int32, init_value=lambda: counts_stacked),
        TensorSpec("valid_rows", [ranks, tokens], torch.int32, init_value=lambda: valid),
        TensorSpec("physical_ids", [ranks, tokens, top_k], torch.int32),
        ScalarSpec("epoch", torch.int32, 5),
    ]


# === Migration validation ====================================================


TRANSFER_ROUNDS = 2


def build_transfer_specs(*, ranks: int, local_slots: int, hidden_size: int,
                         intermediate_size: int, rounds: int = TRANSFER_ROUNDS):
    """Small two-card fixture: two consecutive plans, local and cross-rank moves."""
    import torch

    from golden import ScalarSpec, TensorSpec

    R, C, D, I = ranks, local_slots, hidden_size, intermediate_size
    W = D * I
    if C < 4:
        raise ValueError("The transfer fixture needs at least four local slots")
    if rounds < 1:
        raise ValueError("The transfer fixture needs at least one round")

    def resident(limit):
        return torch.arange(R * C * limit, dtype=torch.float32).reshape(R, C, limit)

    w1 = (resident(W).to(torch.int32) % 97).to(torch.int8)
    w3 = (resident(W).to(torch.int32) % 89).to(torch.int8)
    w2 = (resident(W).to(torch.int32) % 83).to(torch.int8)
    w1_scale = resident(I) * 0.25
    w3_scale = resident(I) * 0.5 + 0.125
    w2_scale = resident(D) * 0.75 + 0.25

    # Round 0: local swaps and cross-rank moves. Round 1: moves that read slots
    # round 0 already rewrote, so a stale window or an early barrier shows up.
    plans = [
        [(0, 1, (1, 2)), (0, 2, (0, 3)), (1, 0, (1, 3)), (1, 2, (0, 0))],
        [(0, 3, (0, 1)), (1, 3, (0, 2)), (1, 1, (1, 0))],
    ][:rounds]
    plan_changed = torch.zeros((rounds, R, C), dtype=torch.int32)
    plan_sources = torch.zeros((rounds, R, C, 2), dtype=torch.int32)
    for round_idx, moves in enumerate(plans):
        for dest, slot, (source_rank, source_slot) in moves:
            plan_changed[round_idx, dest, slot] = 1
            plan_sources[round_idx, dest, slot] = torch.tensor(
                [source_rank, source_slot], dtype=torch.int32)
    # Every rank submits the same plan for all destinations.
    changed = plan_changed.unsqueeze(1).repeat(1, R, 1, 1).permute(1, 0, 2, 3).contiguous()
    sources = plan_sources.unsqueeze(1).repeat(1, R, 1, 1, 1).permute(1, 0, 2, 3, 4).contiguous()

    return [
        TensorSpec("w1", [R, C, W], torch.int8, init_value=lambda: w1),
        TensorSpec("w1_scale", [R, C, I], torch.float32, init_value=lambda: w1_scale),
        TensorSpec("w3", [R, C, W], torch.int8, init_value=lambda: w3),
        TensorSpec("w3_scale", [R, C, I], torch.float32, init_value=lambda: w3_scale),
        TensorSpec("w2", [R, C, W], torch.int8, init_value=lambda: w2),
        TensorSpec("w2_scale", [R, C, D], torch.float32, init_value=lambda: w2_scale),
        TensorSpec("sources", [R, rounds, R, C, 2], torch.int32, init_value=lambda: sources),
        TensorSpec("changed", [R, rounds, R, C], torch.int32, init_value=lambda: changed),
        ScalarSpec("epoch0", torch.int32, 1),
    ]


def golden_transfer(tensors):
    """Reference: apply each round's plan to the state it starts from."""
    import torch

    names = ("w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale")
    changed = tensors["changed"]
    sources = tensors["sources"]
    ranks, rounds, local_slots = changed.shape[0], changed.shape[1], changed.shape[3]
    for round_idx in range(rounds):
        # Ranks must agree on the plan; check before trusting one copy of it.
        for rank in range(ranks):
            if not torch.equal(changed[rank, round_idx], changed[0, round_idx]):
                raise AssertionError("transfer plans differ across ranks")
            if not torch.equal(sources[rank, round_idx], sources[0, round_idx]):
                raise AssertionError("transfer plans differ across ranks")
        before = {name: tensors[name].clone() for name in names}
        for dest in range(ranks):
            for slot in range(local_slots):
                if int(changed[0, round_idx, dest, slot]) != 0:
                    source_rank, source_slot = (
                        int(value) for value in sources[0, round_idx, dest, slot])
                    for name in names:
                        tensors[name][dest, slot] = before[name][source_rank, source_slot]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--case", choices=("all", "routing", "transfer"), default="all")
    parser.add_argument("--ep", type=int, choices=(2, 4, 8, 16), default=2)
    parser.add_argument("--logical-experts", type=int, default=256)
    parser.add_argument("--redundant-experts", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=8,
                        help="active token rows per rank; must be a multiple of the route tile")
    parser.add_argument("--top-k", type=int, default=6,
                        help="routes per token (at most 8)")
    parser.add_argument("--local-slots", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=16)
    parser.add_argument("-d", "--devices", default=None)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.devices is None and not args.sim:
        parser.error("--devices is required for device execution")
    try:
        devices = list(range(args.ep)) if args.devices is None else [int(d) for d in args.devices.split(",")]
    except ValueError:
        parser.error("--devices must be a comma-separated integer list")
    if len(devices) != args.ep:
        parser.error("Device count must match EP")
    if any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        parser.error("Device IDs must be distinct and non-negative")
    config = dict(backend_type=args.platform, execution_mode=args.sim,
                  distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    results = []
    if args.case in ("all", "routing"):
        single_replica = args.redundant_experts == 0
        # Boolean-mask routing (default) and the production `>= 0` masking with a
        # nonzero bank offset: the latter must read the selected bank and treat
        # zero markers as valid.
        routing_cases = [
            dict(bank_layers=1, boolean_mask=True, layer=0),
            dict(bank_layers=2, boolean_mask=False, layer=3),
        ]
        for subcase in routing_cases:
            fn = make_routing_test(logical_experts=args.logical_experts, ranks=args.ep,
                                   tokens=args.num_tokens, top_k=args.top_k,
                                   single_replica=single_replica, **subcase)
            results.append(run(
                fn=fn,
                specs=build_routing_specs(logical_experts=args.logical_experts, ranks=args.ep,
                                          redundant_experts=args.redundant_experts,
                                          tokens=args.num_tokens, top_k=args.top_k, **subcase),
                golden_fn=make_golden_routing(**subcase),
                compile_only=args.compile_only,
                config=config,
                rtol=0,
                atol=0,
                compare_fn={"physical_ids": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
            ))
    if args.case in ("all", "transfer"):
        # Two consecutive migrations share one retained domain, exactly like
        # serving's per-layer calls on its persistent worker.
        fn = make_eplb_transfer_rounds(rounds=TRANSFER_ROUNDS, ranks=args.ep,
                                       local_slots=args.local_slots,
                                       hidden_size=args.hidden_size,
                                       intermediate_size=args.intermediate_size)
        results.append(run(
            fn=fn,
            specs=build_transfer_specs(ranks=args.ep, local_slots=args.local_slots,
                                       hidden_size=args.hidden_size,
                                       intermediate_size=args.intermediate_size,
                                       rounds=TRANSFER_ROUNDS),
            golden_fn=golden_transfer,
            compile_only=args.compile_only,
            config=config,
            rtol=0,
            atol=0,
        ))
    if not all(result.passed for result in results):
        raise SystemExit(next((result.error for result in results if not result.passed),
                              "EPLB validation failed"))
