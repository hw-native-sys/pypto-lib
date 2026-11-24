# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replicate local CSA cache updates for tile-aligned prefill."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import BLOCK_SIZE, FLASH
from prefill_cp_token_allgather import (
    COMM_ROW_TILE, TP_SIZE, _prefill_cp_projection_publish, _prefill_cp_projection_retire,
)
from prefill_compressor_ratio4 import (
    CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, COMPRESS_STATE_DIM as MAIN_STATE_WIDTH,
    PREFILL_STATE_TILE, compressor_ratio4,
)
from prefill_indexer_compressor import (
    COMPRESS_STATE_DIM as INDEX_STATE_WIDTH, INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS,
    prefill_indexer_compressor,
)
from prefill_metadata import CHECKPOINTS_PER_TILE

D = FLASH.hidden_size
HEAD = FLASH.head_dim
INDEX_HEAD = FLASH.index_head_dim
ROPE = FLASH.qk_rope_head_dim
HALO = 8
C4 = 4
RAW = 0
COMPRESSED = RAW + HEAD
INDEX_KV = COMPRESSED + HEAD // C4
SCALE = INDEX_KV + INDEX_HEAD // (2 * C4)
MAIN_STATE = SCALE + 16
CARRIERS = PREFILL_STATE_TILE // HALO
INDEX_STATE = MAIN_STATE + 2 * MAIN_STATE_WIDTH // CARRIERS
PAYLOAD_WORDS = INDEX_STATE + 2 * INDEX_STATE_WIDTH // CARRIERS


@pl.jit.inline(auto_scope=False)
def csa_compact_eligible(
    starts: pl.Tensor, checkpoints: pl.Tensor, pos: pl.Tensor,
    raw_map: pl.Tensor, cmp_map: pl.Tensor, index_map: pl.Tensor,
    state_map: pl.Tensor, index_state_map: pl.Tensor, decision: pl.Tensor,
):
    """Check the full group's replicated scheduling metadata on each rank."""
    total = pl.tensor.dim(pos, 0)
    pages = total // BLOCK_SIZE
    flags = pl.create_tensor([pages, 16], dtype=pl.INT32)
    with pl.spmd(pages, name_hint="csa_cp_validate_pages") as checks_tid:
        page = pl.tile.get_block_idx()
        begin = page * BLOCK_SIZE
        first = pl.read(raw_map, [begin])
        ok = pl.cast(1, pl.INT32)
        if first < 0 or first % BLOCK_SIZE != 0:
            ok = pl.cast(0, pl.INT32)
        for prev in pl.range(page):
            if pl.read(raw_map, [prev * BLOCK_SIZE]) == first:
                ok = pl.cast(0, pl.INT32)
        cmp_origin = (begin // (C4 * BLOCK_SIZE)) * C4 * BLOCK_SIZE
        first_cmp = pl.read(cmp_map, [cmp_origin + C4 - 1])
        first_idx = pl.read(index_map, [cmp_origin + C4 - 1])
        if first_cmp < 0 or first_idx < 0 or first_cmp % BLOCK_SIZE != 0 or first_idx % BLOCK_SIZE != 0:
            ok = pl.cast(0, pl.INT32)
        if begin == cmp_origin:
            for earlier in pl.range(0, cmp_origin, C4 * BLOCK_SIZE):
                if pl.read(cmp_map, [earlier + C4 - 1]) == first_cmp or pl.read(index_map, [earlier + C4 - 1]) == first_idx:
                    ok = pl.cast(0, pl.INT32)
        for j in pl.range(BLOCK_SIZE):
            t = begin + j
            if pl.read(pos, [t]) != t or pl.read(raw_map, [t]) != first + j:
                ok = pl.cast(0, pl.INT32)
            if pl.read(state_map, [t]) < 0 or pl.read(index_state_map, [t]) < 0:
                ok = pl.cast(0, pl.INT32)
            tail = t % PREFILL_STATE_TILE - (PREFILL_STATE_TILE - HALO)
            if tail > 0:
                for earlier in pl.range(t - tail, t):
                    if pl.read(state_map, [earlier]) == pl.read(state_map, [t]) or pl.read(index_state_map, [earlier]) == pl.read(index_state_map, [t]):
                        ok = pl.cast(0, pl.INT32)
            c = pl.read(cmp_map, [t])
            i = pl.read(index_map, [t])
            if t % C4 == C4 - 1:
                offset = (t - cmp_origin) // C4
                if c != first_cmp + offset or i != first_idx + offset:
                    ok = pl.cast(0, pl.INT32)
            else:
                if c >= 0 or i >= 0:
                    ok = pl.cast(0, pl.INT32)
        pl.write(flags, [page, 0], ok)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cp_check_plan", deps=[checks_tid]):
        accepted = pl.cast(1, pl.INT32)
        if pl.read(starts, [0]) != 0 or pl.read(starts, [1]) != total:
            accepted = pl.cast(0, pl.INT32)
        for t in pl.range(total // PREFILL_STATE_TILE):
            for c in pl.range(CHECKPOINTS_PER_TILE):
                if pl.read(checkpoints, [0, t, c]) != 0:
                    accepted = pl.cast(0, pl.INT32)
        for page in pl.range(pages):
            if pl.read(flags, [page, 0]) == 0:
                accepted = pl.cast(0, pl.INT32)
        pl.write(decision, [0], accepted)


@pl.jit.inline(auto_scope=False)
def csa_previous_rows(
    x: pl.Tensor, halo: pl.Tensor, window: pl.InOut[pld.DistributedTensor], signal: pl.InOut[pld.DistributedTensor],
    group: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    local = pl.tensor.dim(x, 0)
    tail = pl.slice(x, [HALO, D], [local - HALO, 0])
    pushed, arrived = _prefill_cp_projection_publish(tail, window, signal, HALO, group, rank, ready, D, "csa_cp_halo_push", "csa_cp_halo_wait")
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cp_halo_read", deps=[pushed, arrived]) as copied:
        prior = (rank + TP_SIZE - 1) % TP_SIZE
        tile = pl.load(window, [prior * HALO, 0], [HALO, D])
        pl.store(tile, [0, 0], halo)
    return _prefill_cp_projection_retire(halo, signal, group, rank, copied, ready, 0, "csa_cp_halo_ack", "csa_cp_halo_ack_wait", "csa_cp_halo_retire")


@pl.jit.inline(auto_scope=False)
def csa_compact_caches(
    x: pl.Tensor, kv: pl.Tensor, halo: pl.Tensor, comp_cos: pl.Tensor, comp_sin: pl.Tensor,
    wkv: pl.Tensor, wgate: pl.Tensor, ape: pl.Tensor, norm: pl.Tensor,
    iwkv: pl.Tensor, iwgate: pl.Tensor, iape: pl.Tensor, inorm: pl.Tensor, hadamard: pl.Tensor,
    raw_map: pl.Tensor, cmp_map: pl.Tensor, index_map: pl.Tensor, state_map: pl.Tensor, index_state_map: pl.Tensor,
    raw_cache: pl.Tensor, cmp_cache: pl.Tensor, index_cache: pl.Tensor, scales: pl.Tensor,
    state: pl.Tensor, index_state: pl.Tensor, window: pl.InOut[pld.DistributedTensor], signal: pl.InOut[pld.DistributedTensor],
    group: pl.Scalar[pl.INT32], rank: pl.Scalar[pl.INT32], halo_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    local = pl.tensor.dim(x, 0)
    total = pl.tensor.dim(raw_map, 0)
    base = rank * local
    extent = HALO + local
    local_x = pl.create_tensor([extent, D], dtype=pl.BF16)
    local_cos = pl.create_tensor([extent, ROPE], dtype=pl.BF16)
    local_sin = pl.create_tensor([extent, ROPE], dtype=pl.BF16)
    positions = pl.create_tensor([extent], dtype=pl.INT32)
    private_state_slots = pl.create_tensor([extent], dtype=pl.INT64)
    private_cache_slots = pl.create_tensor([extent], dtype=pl.INT64)
    starts = pl.create_tensor([3], dtype=pl.INT32)
    tiles = local // PREFILL_STATE_TILE
    checkpoints = pl.create_tensor([2, tiles, CHECKPOINTS_PER_TILE], dtype=pl.INT32)
    main_table = pl.create_tensor([2, CSA_STATE_MAX_BLOCKS], dtype=pl.INT32)
    index_table = pl.create_tensor([2, INNER_STATE_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cp_local_plan", deps=[halo_ready]):
        pl.write(starts, [0], pl.cast(0, pl.INT32))
        pl.write(starts, [1], pl.cast(HALO, pl.INT32))
        pl.write(starts, [2], pl.cast(extent, pl.INT32))
        for tile in pl.range(tiles):
            for ck in pl.range(CHECKPOINTS_PER_TILE):
                pl.write(checkpoints, [0, tile, ck], pl.cast(0, pl.INT32))
                pl.write(checkpoints, [1, tile, ck], pl.cast(0, pl.INT32))
        if rank > 0:
            for page in pl.range(HALO // CSA_STATE_BLOCK_SIZE):
                pl.write(main_table, [1, (base - HALO) // CSA_STATE_BLOCK_SIZE + page], pl.cast(page, pl.INT32))
            for page in pl.range(HALO // INNER_STATE_BLOCK_SIZE):
                pl.write(index_table, [1, (base - HALO) // INNER_STATE_BLOCK_SIZE + page], pl.cast(page, pl.INT32))
        previous_x = pl.load(halo, [0, 0], [HALO, D])
        pl.store(previous_x, [0, 0], local_x)
        z = pl.tile.full([HALO, ROPE], dtype=pl.BF16, value=0.0)
        pl.store(z, [0, 0], local_cos)
        pl.store(z, [0, 0], local_sin)
        for j in pl.range(HALO):
            pl.write(positions, [j], pl.cast(pl.max(0, base - HALO + j), pl.INT32))
            pl.write(private_state_slots, [j], pl.cast(j, pl.INT64))
            pl.write(private_cache_slots, [j], pl.cast(-1, pl.INT64))
    position_row = pl.reshape(positions, [1, extent])
    with pl.spmd(local // COMM_ROW_TILE, name_hint="csa_cp_local_inputs"):
        t = pl.tile.get_block_idx() * COMM_ROW_TILE
        activations = pl.load(x, [t, 0], [COMM_ROW_TILE, D])
        pl.store(activations, [HALO + t, 0], local_x)
        cosine = pl.load(comp_cos, [base + t, 0], [COMM_ROW_TILE, ROPE])
        sine = pl.load(comp_sin, [base + t, 0], [COMM_ROW_TILE, ROPE])
        pl.store(cosine, [HALO + t, 0], local_cos)
        pl.store(sine, [HALO + t, 0], local_sin)
        lane = pl.tile.arange(0, [1, COMM_ROW_TILE], dtype=pl.INT32)
        local_positions = pl.add(lane, pl.cast(base + t, pl.INT32))
        pl.store(local_positions, [0, HALO + t], position_row)
        for j in pl.range(COMM_ROW_TILE):
            row = HALO + t + j
            pl.write(private_state_slots, [row], pl.cast(row, pl.INT64))
            if j % C4 == C4 - 1:
                pl.write(private_cache_slots, [row], pl.cast((t + j) // C4, pl.INT64))
            else:
                pl.write(private_cache_slots, [row], pl.cast(-1, pl.INT64))

    main_scratch = pl.create_tensor([extent // CSA_STATE_BLOCK_SIZE, CSA_STATE_BLOCK_SIZE, MAIN_STATE_WIDTH], dtype=pl.FP32)
    index_scratch = pl.create_tensor([extent // INNER_STATE_BLOCK_SIZE, INNER_STATE_BLOCK_SIZE, INDEX_STATE_WIDTH], dtype=pl.FP32)
    cache_pages = local // (C4 * BLOCK_SIZE)
    main_rows = pl.create_tensor([cache_pages, BLOCK_SIZE, 1, HEAD], dtype=pl.BF16)
    index_rows = pl.create_tensor([cache_pages, BLOCK_SIZE, 1, INDEX_HEAD], dtype=pl.INT8)
    index_scales = pl.create_tensor([cache_pages, BLOCK_SIZE, 1, 1], dtype=pl.FP32)
    compressor_ratio4(local_x, starts, checkpoints, main_scratch, main_table, wkv, wgate, ape, norm, local_cos, local_sin, main_rows, positions, private_cache_slots, private_state_slots)
    index_done = pl.array.create(1, pl.TASK_ID)
    prefill_indexer_compressor(local_x, starts, checkpoints, index_scratch, index_table, iwkv, iwgate, iape, inorm, local_cos, local_sin, hadamard, index_rows, index_scales, positions, private_cache_slots, private_state_slots, index_done)
    main_flat = pl.reshape(main_rows, [local // C4, HEAD])
    index_flat = pl.reshape(index_rows, [local // C4, INDEX_HEAD])
    scale_flat = pl.reshape(index_scales, [1, local // C4])
    main_states = pl.reshape(main_scratch, [extent, MAIN_STATE_WIDTH])
    index_states = pl.reshape(index_scratch, [extent, INDEX_STATE_WIDTH])
    packet = pl.create_tensor([local, PAYLOAD_WORDS], dtype=pl.BF16)
    with pl.spmd(local // COMM_ROW_TILE, name_hint="csa_cp_packet_kv", deps=[index_done[0]]) as kv_packed:
        t = pl.tile.get_block_idx() * COMM_ROW_TILE
        raw = pl.load(kv, [t, 0], [COMM_ROW_TILE, HEAD])
        pl.store(raw, [t, RAW], packet)
        m = pl.load(main_flat, [t // C4, 0], [COMM_ROW_TILE // C4, HEAD])
        m_words = pl.reshape(m, [COMM_ROW_TILE, HEAD // C4])
        pl.store(m_words, [t, COMPRESSED], packet)
        i = pl.load(index_flat, [t // C4, 0], [COMM_ROW_TILE // C4, INDEX_HEAD])
        i_rows = pl.reshape(i, [COMM_ROW_TILE, INDEX_HEAD // C4])
        i_words = pl.reinterpret_view(i_rows, pl.BF16)
        pl.store(i_words, [t, INDEX_KV], packet)
    # Move aligned FP32 scale rows as their original bits through the BF16 window.
    with pl.spmd(local // (8 * C4), name_hint="csa_cp_packet_scale", deps=[index_done[0]]) as scales_packed:
        offset = pl.tile.get_block_idx() * 8
        scale_tile = pl.load(scale_flat, [0, offset], [1, 8])
        scale_words = pl.reinterpret_view(scale_tile, pl.BF16)
        pl.store(scale_words, [offset * C4, SCALE], packet)
    with pl.spmd(tiles * HALO, name_hint="csa_cp_packet_state") as states_packed:
        ordinal = pl.tile.get_block_idx()
        tile = ordinal // HALO
        tail = ordinal % HALO
        state_row = (tile + 1) * PREFILL_STATE_TILE + tail
        dst_row = ordinal * CARRIERS
        m_state = pl.load(main_states, [state_row, 0], [1, MAIN_STATE_WIDTH])
        m_carrier = pl.reshape(m_state, [CARRIERS, MAIN_STATE_WIDTH // CARRIERS])
        pl.store(pl.reinterpret_view(m_carrier, pl.BF16), [dst_row, MAIN_STATE], packet)
        i_state = pl.load(index_states, [state_row, 0], [1, INDEX_STATE_WIDTH])
        i_carrier = pl.reshape(i_state, [CARRIERS, INDEX_STATE_WIDTH // CARRIERS])
        pl.store(pl.reinterpret_view(i_carrier, pl.BF16), [dst_row, INDEX_STATE], packet)
    packet_done = pl.system.task_dummy(deps=[kv_packed, scales_packed, states_packed, halo_ready])
    pushed, arrived = _prefill_cp_projection_publish(packet, window, signal, local, group, rank, packet_done, PAYLOAD_WORDS, "csa_cp_packet_push", "csa_cp_packet_wait")

    with pl.spmd(total // COMM_ROW_TILE, name_hint="csa_cp_cache_read", deps=[pushed, arrived]) as caches_done:
        t = pl.tile.get_block_idx() * COMM_ROW_TILE
        raw = pl.load(window, [t, RAW], [COMM_ROW_TILE, HEAD])
        raw_dst = pl.cast(pl.read(raw_map, [t]), pl.INDEX)
        pl.store(pl.reshape(raw, [1, COMM_ROW_TILE, 1, HEAD]), [raw_dst // BLOCK_SIZE, raw_dst % BLOCK_SIZE, 0, 0], raw_cache)
        m_words = pl.load(window, [t, COMPRESSED], [COMM_ROW_TILE, HEAD // C4])
        m = pl.reshape(m_words, [COMM_ROW_TILE // C4, HEAD])
        m_dst = pl.cast(pl.read(cmp_map, [t + C4 - 1]), pl.INDEX)
        pl.store(pl.reshape(m, [1, COMM_ROW_TILE // C4, 1, HEAD]), [m_dst // BLOCK_SIZE, m_dst % BLOCK_SIZE, 0, 0], cmp_cache)
        i_words = pl.load(window, [t, INDEX_KV], [COMM_ROW_TILE, INDEX_HEAD // (C4 * 2)])
        i_rows = pl.reinterpret_view(i_words, pl.INT8)
        i = pl.reshape(i_rows, [COMM_ROW_TILE // C4, INDEX_HEAD])
        i_dst = pl.cast(pl.read(index_map, [t + C4 - 1]), pl.INDEX)
        pl.store(pl.reshape(i, [1, COMM_ROW_TILE // C4, 1, INDEX_HEAD]), [i_dst // BLOCK_SIZE, i_dst % BLOCK_SIZE, 0, 0], index_cache)
    scale_pages = pl.tensor.dim(scales, 0)
    public_scale_row = pl.reshape(scales, [1, scale_pages * BLOCK_SIZE])
    with pl.spmd(total // (8 * C4), name_hint="csa_cp_scale_read", deps=[pushed, arrived]) as scales_done:
        first = pl.tile.get_block_idx() * 8 * C4
        words = pl.load(window, [first, SCALE], [1, 16])
        values = pl.reinterpret_view(words, pl.FP32)
        scale_dst = pl.cast(pl.read(index_map, [first + C4 - 1]), pl.INDEX)
        pl.store(values, [0, scale_dst], public_scale_row)

    with pl.spmd(HALO, name_hint="csa_cp_state_read", deps=[pushed, arrived]) as states_done:
        worker = pl.tile.get_block_idx()
        for ordinal in pl.range(total // CARRIERS):
            tile = ordinal // HALO
            tail = ordinal % HALO
            position = (tile + 1) * PREFILL_STATE_TILE - HALO + tail
            src = ordinal * CARRIERS
            m_dst = pl.cast(pl.read(state_map, [position]), pl.INDEX)
            if m_dst % HALO == worker:
                mw = pl.load(window, [src, MAIN_STATE], [CARRIERS, 2 * MAIN_STATE_WIDTH // CARRIERS])
                main_row = pl.reshape(pl.reinterpret_view(mw, pl.FP32), [1, 1, MAIN_STATE_WIDTH])
                pl.store(main_row, [m_dst // CSA_STATE_BLOCK_SIZE, m_dst % CSA_STATE_BLOCK_SIZE, 0], state)
            i_dst = pl.cast(pl.read(index_state_map, [position]), pl.INDEX)
            if i_dst % HALO == worker:
                iw = pl.load(window, [src, INDEX_STATE], [CARRIERS, 2 * INDEX_STATE_WIDTH // CARRIERS])
                idx_row = pl.reshape(pl.reinterpret_view(iw, pl.FP32), [1, 1, INDEX_STATE_WIDTH])
                pl.store(idx_row, [i_dst // INNER_STATE_BLOCK_SIZE, i_dst % INNER_STATE_BLOCK_SIZE, 0], index_state)
    read_done = pl.system.task_dummy(deps=[caches_done, scales_done, states_done])
    return _prefill_cp_projection_retire(packet, signal, group, rank, read_done, packet_done, 0, "csa_cp_packet_ack", "csa_cp_packet_ack_wait", "csa_cp_packet_retire")
