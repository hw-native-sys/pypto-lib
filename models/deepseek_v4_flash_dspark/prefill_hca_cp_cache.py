# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish owner-computed C128 cache rows for aligned CP prefill requests."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import BLOCK_SIZE
from prefill_compressor_ratio128 import (
    COMPRESS_RATIO, COMPRESS_STATE_DIM, HEAD_DIM, HCA_STATE_BLOCK_SIZE,
    PREFILL_STATE_TILE, STATE_STORAGE_LEN, prefill_compressor_ratio128_projected,
)
from prefill_cp_token_allgather import COMM_ROW_TILE, TP_SIZE, _prefill_cp_projection_retire
from prefill_metadata import CHECKPOINTS_PER_TILE

COMPRESSED_COL = HEAD_DIM
STATE_COL = 2 * HEAD_DIM
STATE_WORDS = COMPRESS_STATE_DIM * 2


@pl.jit.inline(auto_scope=False)
def hca_owner_eligible(
    starts: pl.Tensor, checkpoints: pl.Tensor, positions: pl.Tensor,
    raw_slots: pl.Tensor, cmp_slots: pl.Tensor, state_slots: pl.Tensor, decision: pl.Tensor,
):
    """Validate replicated cache destinations and the recurrent ring's final writers."""
    total = pl.tensor.dim(positions, 0)
    pages = total // BLOCK_SIZE
    flags = pl.create_tensor([pages, 16], dtype=pl.INT32)
    with pl.spmd(pages, name_hint="hca_cp_owner_pages") as pages_tid:
        page = pl.tile.get_block_idx()
        begin = page * BLOCK_SIZE
        ok = pl.cast(1, pl.INT32)
        raw_page = pl.read(raw_slots, [begin])
        if raw_page < 0 or raw_page % BLOCK_SIZE != 0:
            ok = pl.cast(0, pl.INT32)
        for prev in pl.range(page):
            if pl.read(raw_slots, [prev * BLOCK_SIZE]) == raw_page:
                ok = pl.cast(0, pl.INT32)
        for offset in pl.range(BLOCK_SIZE):
            row = begin + offset
            if pl.read(positions, [row]) != row or pl.read(raw_slots, [row]) != raw_page + offset:
                ok = pl.cast(0, pl.INT32)
            last = total - 1 - (total - 1 - row) % STATE_STORAGE_LEN
            state = pl.read(state_slots, [row])
            if state < 0 or state != pl.read(state_slots, [last]):
                ok = pl.cast(0, pl.INT32)
            if row >= total - STATE_STORAGE_LEN:
                first = row - row % HCA_STATE_BLOCK_SIZE
                start = pl.read(state_slots, [first])
                if start % HCA_STATE_BLOCK_SIZE != 0 or state != start + row % HCA_STATE_BLOCK_SIZE:
                    ok = pl.cast(0, pl.INT32)
                if row == first:
                    for earlier in pl.range(total - STATE_STORAGE_LEN, row, HCA_STATE_BLOCK_SIZE):
                        if pl.read(state_slots, [earlier]) == start:
                            ok = pl.cast(0, pl.INT32)
            cmp = pl.read(cmp_slots, [row])
            if row % COMPRESS_RATIO == COMPRESS_RATIO - 1:
                if cmp < 0:
                    ok = pl.cast(0, pl.INT32)
                for prior in pl.range(COMPRESS_RATIO - 1, row, COMPRESS_RATIO):
                    if pl.read(cmp_slots, [prior]) == cmp:
                        ok = pl.cast(0, pl.INT32)
            else:
                if cmp >= 0:
                    ok = pl.cast(0, pl.INT32)
        pl.write(flags, [page, 0], ok)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_cp_owner_plan", deps=[pages_tid]):
        ok = pl.cast(1, pl.INT32)
        if pl.read(starts, [0]) != 0 or pl.read(starts, [1]) != total:
            ok = pl.cast(0, pl.INT32)
        for tile in pl.range(total // PREFILL_STATE_TILE):
            for checkpoint in pl.range(CHECKPOINTS_PER_TILE):
                if pl.read(checkpoints, [0, tile, checkpoint]) != 0:
                    ok = pl.cast(0, pl.INT32)
        for page in pl.range(pages):
            if pl.read(flags, [page, 0]) == 0:
                ok = pl.cast(0, pl.INT32)
        pl.write(decision, [0], ok)


@pl.jit.inline(auto_scope=False)
def hca_publish_owner_caches(
    raw: pl.Tensor, values: pl.Tensor, scores: pl.Tensor, checkpoints: pl.Tensor,
    ape: pl.Tensor, norm: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor, positions: pl.Tensor,
    raw_slots: pl.Tensor, cmp_slots: pl.Tensor, state_slots: pl.Tensor, state_table: pl.Tensor,
    raw_cache: pl.Tensor, cmp_cache: pl.Tensor, state: pl.Tensor,
    window: pl.InOut[pld.DistributedTensor], signal: pl.InOut[pld.DistributedTensor],
    group: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Pool this shard and publish raw, compressed and final recurrent cache rows."""
    local = pl.tensor.dim(raw, 0)
    total = pl.tensor.dim(positions, 0)
    base = local * rank
    starts = pl.create_tensor([2], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_cp_owner_range"):
        pl.write(starts, [0], pl.cast(0, pl.INT32))
        pl.write(starts, [1], pl.cast(local, pl.INT32))
    rope_width = pl.tensor.dim(cos, 1)
    local_cos = pl.slice(cos, [local, rope_width], [base, 0])
    local_sin = pl.slice(sin, [local, rope_width], [base, 0])
    local_positions = pl.slice(positions, [local], [base])
    local_cmp = pl.slice(cmp_slots, [local], [base])
    local_state = pl.slice(state_slots, [local], [base])
    prefill_compressor_ratio128_projected(
        values, scores, starts, checkpoints, state, state_table, ape, norm, local_cos, local_sin,
        cmp_cache, local_positions, local_cmp, local_state,
    )

    cmp_flat = pl.reshape(cmp_cache, [pl.tensor.dim(cmp_cache, 0), HEAD_DIM])
    state_rows = pl.tensor.dim(state, 0) * HCA_STATE_BLOCK_SIZE
    state_flat = pl.reshape(state, [state_rows, COMPRESS_STATE_DIM])
    cmp_packet = pl.create_tensor([local // COMPRESS_RATIO, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd(local // COMPRESS_RATIO, name_hint="hca_cp_owner_compressed") as cmp_tid:
        row = pl.tile.get_block_idx()
        src = pl.cast(pl.read(cmp_slots, [base + (row + 1) * COMPRESS_RATIO - 1]), pl.INDEX)
        value = pl.load(cmp_flat, [src, 0], [1, HEAD_DIM])
        pl.store(value, [row, 0], cmp_packet)

    state_packet = pl.create_tensor([STATE_STORAGE_LEN, STATE_WORDS], dtype=pl.BF16)
    with pl.spmd(STATE_STORAGE_LEN, name_hint="hca_cp_owner_final_state") as state_tid:
        if rank == TP_SIZE - 1:
            row = pl.tile.get_block_idx()
            slot = pl.cast(pl.read(state_slots, [total - STATE_STORAGE_LEN + row]), pl.INDEX)
            tail_value = pl.load(state_flat, [slot, 0], [1, COMPRESS_STATE_DIM])
            pl.store(pl.reinterpret_view(tail_value, pl.BF16), [row, 0], state_packet)

    packet_ready = pl.system.task_dummy(deps=[cmp_tid, state_tid])
    with pl.spmd(TP_SIZE, name_hint="hca_cp_owner_push", deps=[packet_ready], allow_early_resolve=True) as push_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(dst=window, peer=group + peer, src=raw, dst_offsets=[base, 0], src_offsets=[0, 0],
                       shape=[local, HEAD_DIM], chunk_rows=COMM_ROW_TILE, chunk_cols=HEAD_DIM, pipeline=True)
        pld.tensor.put(dst=window, peer=group + peer, src=cmp_packet, dst_offsets=[base // COMPRESS_RATIO, COMPRESSED_COL], src_offsets=[0, 0],
                       shape=[local // COMPRESS_RATIO, HEAD_DIM], chunk_rows=1, chunk_cols=HEAD_DIM, pipeline=True)
        if rank == TP_SIZE - 1:
            pld.tensor.put(dst=window, peer=group + peer, src=state_packet, dst_offsets=[0, STATE_COL], src_offsets=[0, 0],
                           shape=[STATE_STORAGE_LEN, STATE_WORDS], chunk_rows=COMM_ROW_TILE, chunk_cols=STATE_WORDS, pipeline=True)
        if peer != rank:
            pld.system.notify(target=signal, peer=group + peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_cp_owner_wait", deps=[packet_ready]) as arrival_tid:
        for peer in pl.range(TP_SIZE):
            if peer != rank:
                pld.system.defer_wait(signal=signal, offsets=[peer, 0], expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge)

    raw_flat = pl.reshape(raw_cache, [pl.tensor.dim(raw_cache, 0) * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(total // BLOCK_SIZE, name_hint="hca_cp_owner_raw_read", deps=[push_tid, arrival_tid]) as raw_done:
        first = pl.tile.get_block_idx() * BLOCK_SIZE
        slot = pl.cast(pl.read(raw_slots, [first]), pl.INDEX)
        rows = pl.load(window, [first, 0], [BLOCK_SIZE, HEAD_DIM])
        pl.store(rows, [slot, 0], raw_flat)
    with pl.spmd(total // COMPRESS_RATIO, name_hint="hca_cp_owner_cmp_read", deps=[push_tid, arrival_tid]) as cmp_done:
        row = pl.tile.get_block_idx()
        slot = pl.cast(pl.read(cmp_slots, [(row + 1) * COMPRESS_RATIO - 1]), pl.INDEX)
        value = pl.load(window, [row, COMPRESSED_COL], [1, HEAD_DIM])
        pl.store(value, [slot, 0], cmp_flat)
    with pl.spmd(STATE_STORAGE_LEN, name_hint="hca_cp_owner_state_read", deps=[push_tid, arrival_tid]) as state_done:
        row = pl.tile.get_block_idx()
        slot = pl.cast(pl.read(state_slots, [total - STATE_STORAGE_LEN + row]), pl.INDEX)
        words = pl.load(window, [row, STATE_COL], [1, STATE_WORDS])
        pl.store(pl.reinterpret_view(words, pl.FP32), [slot, 0], state_flat)
    read_tid = pl.system.task_dummy(deps=[raw_done, cmp_done, state_done])
    return _prefill_cp_projection_retire(raw, signal, group, rank, read_tid, packet_ready, 0,
                                         "hca_cp_owner_ack", "hca_cp_owner_ack_wait", "hca_cp_owner_retire")
