# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Descriptor-derived metadata for batched bootstrap requests in each TP group."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP, TP

LEASES_DYN = pl.dynamic("DSPARK_BOOT_META_LEASES_DYN")
POSITIONS_DYN = pl.dynamic("DSPARK_BOOT_META_POSITIONS_DYN")
BATCH = 4
TAIL_CONTEXT = 128
CONTEXT = BATCH * TAIL_CONTEXT
QUERY = 16 * 7
GROUP_QUERY = TP * QUERY
ROPE_DIM = 64
TABLE_DEPTH = 32768
TABLE_TILE = 1024
METADATA_NAMES = (
    "tail_selectors",
    "num_sampled",
    "last_sampled",
    "anchor_positions",
    "block_tables",
    "context_group_position_ids",
    "context_group_slot_mapping",
    "context_group_freqs_cos",
    "context_group_freqs_sin",
    "query_group_position_ids",
    "query_group_slot_mapping",
    "query_freqs_cos",
    "query_freqs_sin",
    "query_group_freqs_cos",
    "query_group_freqs_sin",
    "logit_row_indices",
)


@pl.jit
def prepare_bootstrap_metadata_rank(
    descriptors: pl.Tensor[[BATCH, 11], pl.INT32],
    meta: pl.Tensor[[LEASES_DYN, 6], pl.INT32],
    full_cos: pl.Tensor[[POSITIONS_DYN, ROPE_DIM], pl.BF16],
    full_sin: pl.Tensor[[POSITIONS_DYN, ROPE_DIM], pl.BF16],
    tail_selectors: pl.Out[pl.Tensor[[BATCH, 4], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[BATCH], pl.INT64]],
    anchor_positions: pl.Out[pl.Tensor[[BATCH], pl.INT32]],
    block_tables: pl.Out[pl.Tensor[[3, BATCH, TABLE_DEPTH], pl.INT32]],
    context_group_position_ids: pl.Out[pl.Tensor[[CONTEXT], pl.INT32]],
    context_group_slot_mapping: pl.Out[pl.Tensor[[3, CONTEXT], pl.INT64]],
    context_group_freqs_cos: pl.Out[pl.Tensor[[CONTEXT, ROPE_DIM], pl.BF16]],
    context_group_freqs_sin: pl.Out[pl.Tensor[[CONTEXT, ROPE_DIM], pl.BF16]],
    query_group_position_ids: pl.Out[pl.Tensor[[GROUP_QUERY], pl.INT32]],
    query_group_slot_mapping: pl.Out[pl.Tensor[[3, GROUP_QUERY], pl.INT64]],
    query_freqs_cos: pl.Out[pl.Tensor[[QUERY, ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.Out[pl.Tensor[[QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.Out[pl.Tensor[[GROUP_QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_sin: pl.Out[pl.Tensor[[GROUP_QUERY, ROPE_DIM], pl.BF16]],
    logit_row_indices: pl.Out[pl.Tensor[[128], pl.INT32]],
    tp_rank: pl.Scalar[pl.INT32],
):
    meta.bind_dynamic(0, LEASES_DYN)
    full_cos.bind_dynamic(0, POSITIONS_DYN)
    full_sin.bind_dynamic(0, POSITIONS_DYN)
    leases = pl.tensor.dim(meta, 0)
    rope_positions = pl.tensor.dim(full_cos, 0)
    info = pl.create_tensor([BATCH, 2], dtype=pl.INT32)
    pattern = pl.create_tensor([6, TABLE_TILE], dtype=pl.INT32)
    flat_tables = pl.reshape(block_tables, [3 * BATCH, TABLE_DEPTH])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_bootstrap_metadata"):
        for request in pl.range(BATCH):
            slot = pl.read(descriptors, [request, 0])
            generation = pl.read(descriptors, [request, 1])
            start = pl.read(descriptors, [request, 3])
            length = pl.read(descriptors, [request, 4])
            prompt = pl.read(descriptors, [request, 5])
            flags = pl.read(descriptors, [request, 10])
            eligible = pl.cast(0, pl.INT32)
            if slot >= 0 and slot < leases and length > 0 and prompt > 0:
                resident_generation = pl.read(meta, [slot, 1])
                valid = pl.read(meta, [slot, 0])
                limit = pl.read(meta, [slot, 5])
                if generation == resident_generation and valid == 0 and start + length >= prompt:
                    if (
                        flags % 2 == 1 and flags // 2 % 2 == 1
                        and prompt + 6 < limit and prompt + 6 < rope_positions
                    ):
                        eligible = pl.cast(1, pl.INT32)
            tail_length = pl.cast(pl.min(prompt, TAIL_CONTEXT), pl.INT32)
            tail_start = prompt - tail_length
            pl.write(info, [request, 0], eligible)
            pl.write(info, [request, 1], slot)
            for field in pl.range(4):
                pl.write(tail_selectors, [request, field], pl.cast(-1, pl.INT32))
            if eligible == 1:
                pl.write(tail_selectors, [request, 0], slot)
                pl.write(tail_selectors, [request, 1], generation)
                pl.write(tail_selectors, [request, 2], pl.cast(tail_start, pl.INT32))
                pl.write(tail_selectors, [request, 3], tail_length)
            pl.write(num_sampled, [request], pl.cast(0, pl.INT32))
            pl.write(last_sampled, [request], pl.cast(0, pl.INT64))
            pl.write(anchor_positions, [request], pl.cast(0, pl.INT32))
            if eligible == 1 and tp_rank == 0:
                pl.write(anchor_positions, [request], pl.cast(prompt - 1, pl.INT32))
        for row in pl.range(128):
            logit = pl.cast(-1, pl.INT32)
            request = row // 7
            if request < BATCH and tp_rank == 0 and pl.read(info, [request, 0]) == 1:
                logit = pl.cast(row, pl.INT32)
            pl.write(logit_row_indices, [row], logit)
        for row in pl.range(CONTEXT):
            owner = row // (CONTEXT // TP)
            request = row % (CONTEXT // TP) // (TAIL_CONTEXT // TP)
            offset = owner * (TAIL_CONTEXT // TP) + row % (TAIL_CONTEXT // TP)
            eligible = pl.read(info, [request, 0])
            prompt = pl.read(descriptors, [request, 5])
            tail_length = pl.cast(pl.min(prompt, TAIL_CONTEXT), pl.INT32)
            position = pl.cast(0, pl.INT32)
            live = pl.cast(0, pl.INT32)
            if eligible == 1 and offset < tail_length:
                position = pl.cast(prompt - tail_length + offset, pl.INT32)
                live = pl.cast(1, pl.INT32)
            pl.write(context_group_position_ids, [row], position)
            for layer in pl.range(3):
                cache_slot = pl.cast(-1, pl.INT64)
                if live == 1:
                    slot = pl.read(info, [request, 1])
                    page = slot * 6 + (position // 32 + 7 * layer) % 6
                    cache_slot = pl.cast(page * 32 + position % 32, pl.INT64)
                pl.write(context_group_slot_mapping, [layer, row], cache_slot)
        for row in pl.range(GROUP_QUERY):
            request = row // 7
            position = pl.cast(0, pl.INT32)
            live = pl.cast(0, pl.INT32)
            if request < BATCH and pl.read(info, [request, 0]) == 1:
                prompt = pl.read(descriptors, [request, 5])
                position = pl.cast(prompt + row % 7, pl.INT32)
                live = pl.cast(1, pl.INT32)
            pl.write(query_group_position_ids, [row], position)
            for layer in pl.range(3):
                cache_slot = pl.cast(-1, pl.INT64)
                if live == 1:
                    slot = pl.read(info, [request, 1])
                    page = slot * 6 + (position // 32 + 7 * layer) % 6
                    cache_slot = pl.cast(page * 32 + position % 32, pl.INT64)
                pl.write(query_group_slot_mapping, [layer, row], cache_slot)
        # Exact integer phase templates avoid vector integer-division assumptions.
        for phase in pl.range(6):
            for column in pl.range(TABLE_TILE):
                pl.write(pattern, [phase, column], pl.cast((phase + column) % 6, pl.INT32))
    for core in pl.spmd(48, name_hint="dspark_bootstrap_tables"):
        for work in pl.range(core, 3 * BATCH * (TABLE_DEPTH // TABLE_TILE), 48):
            row = work // (TABLE_DEPTH // TABLE_TILE)
            layer = row // BATCH
            request = row % BATCH
            begin = work % (TABLE_DEPTH // TABLE_TILE) * TABLE_TILE
            base = pl.cast(384, pl.INT32)
            if pl.read(info, [request, 0]) == 1 and tp_rank == 0:
                lease = pl.read(info, [request, 1])
                base = pl.cast(lease * 6, pl.INT32)
            phase = (begin + 7 * layer) % 6
            values = pl.add(pl.slice(pattern, [1, TABLE_TILE], [phase, 0]), base)
            flat_tables[row : row + 1, begin : begin + TABLE_TILE] = values
    for core in pl.spmd(48, name_hint="dspark_bootstrap_rope"):
        for work in pl.range(core, CONTEXT + GROUP_QUERY + QUERY, 48):
            if work < CONTEXT:
                position = pl.read(context_group_position_ids, [work])
                context_group_freqs_cos[work : work + 1, :] = full_cos[position : position + 1, :]
                context_group_freqs_sin[work : work + 1, :] = full_sin[position : position + 1, :]
            elif work < CONTEXT + GROUP_QUERY:
                row = work - CONTEXT
                position = pl.read(query_group_position_ids, [row])
                query_group_freqs_cos[row : row + 1, :] = full_cos[position : position + 1, :]
                query_group_freqs_sin[row : row + 1, :] = full_sin[position : position + 1, :]
            else:
                row = work - CONTEXT - GROUP_QUERY
                position = pl.read(query_group_position_ids, [tp_rank * QUERY + row])
                query_freqs_cos[row : row + 1, :] = full_cos[position : position + 1, :]
                query_freqs_sin[row : row + 1, :] = full_sin[position : position + 1, :]
    return (
        tail_selectors,
        num_sampled,
        last_sampled,
        anchor_positions,
        block_tables,
        context_group_position_ids,
        context_group_slot_mapping,
        context_group_freqs_cos,
        context_group_freqs_sin,
        query_group_position_ids,
        query_group_slot_mapping,
        query_freqs_cos,
        query_freqs_sin,
        query_group_freqs_cos,
        query_group_freqs_sin,
        logit_row_indices,
    )


@pl.jit.host
def l3_prepare_bootstrap_metadata(
    descriptors: pl.Tensor[[EP, BATCH, 11], pl.INT32],
    meta: pl.Tensor[[EP, LEASES_DYN, 6], pl.INT32],
    full_cos: pl.Tensor[[EP, POSITIONS_DYN, ROPE_DIM], pl.BF16],
    full_sin: pl.Tensor[[EP, POSITIONS_DYN, ROPE_DIM], pl.BF16],
    tail_selectors: pl.Out[pl.Tensor[[EP, BATCH, 4], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP, BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP, BATCH], pl.INT64]],
    anchor_positions: pl.Out[pl.Tensor[[EP, BATCH], pl.INT32]],
    block_tables: pl.Out[pl.Tensor[[EP, 3, BATCH, TABLE_DEPTH], pl.INT32]],
    context_group_position_ids: pl.Out[pl.Tensor[[EP, CONTEXT], pl.INT32]],
    context_group_slot_mapping: pl.Out[pl.Tensor[[EP, 3, CONTEXT], pl.INT64]],
    context_group_freqs_cos: pl.Out[pl.Tensor[[EP, CONTEXT, ROPE_DIM], pl.BF16]],
    context_group_freqs_sin: pl.Out[pl.Tensor[[EP, CONTEXT, ROPE_DIM], pl.BF16]],
    query_group_position_ids: pl.Out[pl.Tensor[[EP, GROUP_QUERY], pl.INT32]],
    query_group_slot_mapping: pl.Out[pl.Tensor[[EP, 3, GROUP_QUERY], pl.INT64]],
    query_freqs_cos: pl.Out[pl.Tensor[[EP, QUERY, ROPE_DIM], pl.BF16]],
    query_freqs_sin: pl.Out[pl.Tensor[[EP, QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_cos: pl.Out[pl.Tensor[[EP, GROUP_QUERY, ROPE_DIM], pl.BF16]],
    query_group_freqs_sin: pl.Out[pl.Tensor[[EP, GROUP_QUERY, ROPE_DIM], pl.BF16]],
    logit_row_indices: pl.Out[pl.Tensor[[EP, 128], pl.INT32]],
):
    meta.bind_dynamic(1, LEASES_DYN)
    full_cos.bind_dynamic(1, POSITIONS_DYN)
    full_sin.bind_dynamic(1, POSITIONS_DYN)
    for rank in pl.range(pld.world_size()):
        prepare_bootstrap_metadata_rank(
            descriptors[rank],
            meta[rank],
            full_cos[rank],
            full_sin[rank],
            tail_selectors[rank],
            num_sampled[rank],
            last_sampled[rank],
            anchor_positions[rank],
            block_tables[rank],
            context_group_position_ids[rank],
            context_group_slot_mapping[rank],
            context_group_freqs_cos[rank],
            context_group_freqs_sin[rank],
            query_group_position_ids[rank],
            query_group_slot_mapping[rank],
            query_freqs_cos[rank],
            query_freqs_sin[rank],
            query_group_freqs_cos[rank],
            query_group_freqs_sin[rank],
            logit_row_indices[rank],
            rank % TP,
            device=rank,
        )
