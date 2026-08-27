# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek V4 context-parallel CSA prefill."""

import argparse
import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    FLASH as M,
    IDX_CACHE_MAX_BLOCKS,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_IDX_BLOCK_NUM,
    PREFILL_IDX_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
)
from hc_post import golden_hc_post_prefill, hc_post_prefill
from hc_pre import golden_hc_pre, hc_pre
from prefill_compressor_ratio4 import (
    CMP_STORAGE_BLOCK_SIZE,
    HEAD_DIM,
    OUT_DIM as MAIN_OUT_DIM,
    build_tensor_specs as build_compressor_tensor_specs,
    golden_prefill_compressor_ratio4,
    compressor_ratio4,
)
from prefill_cp_exchange import (
    INNER_STATE_DIM,
    INNER_STATE_ROWS,
    MAIN_CACHE_ROWS,
    MAIN_STATE_DIM,
    MAIN_STATE_ROWS,
    META_DIM,
    RECORDS_PER_WINDOW,
    ROWS_PER_RANK,
    SCALE_TILE_COLS,
    STATE_META_DIM,
    STATE_RECORDS_PER_WINDOW,
    STATE_ROWS_PER_RANK,
    _prefill_cp_dual_tail_exchange_wave,
    _prefill_cp_csa_compact_finish_wave,
    _prefill_cp_csa_compact_transport_wave,
    _prefill_cp_sparse_stage,
)
from prefill_cp_zigzag import (
    CP_CHOICES,
    CP_SIZE,
    CP_TAIL_WINDOW_ROWS,
    EPOCHS,
    NUM_SEGMENTS,
    TAIL_ROWS,
    cp_final_window_sources,
    cp_owner_tables,
    cp_owner_part,
    cp_owner_rank,
    cp_reverse_index,
)
from prefill_indexer import (
    CP_INDEXER_SCORE_CAP,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_TOPK,
    _prefill_indexer_cp_score_topk,
    build_tensor_specs as build_indexer_tensor_specs,
)
from prefill_indexer_compressor import (
    OUT_DIM as INNER_OUT_DIM,
    golden_prefill_indexer_compressor,
    _prefill_indexer_compressor_with_completion,
)
from prefill_sparse_attn import (
    PREFILL_SPARSE_PAD,
    ROPE_DIM,
    VALID_BLOCK_MASK_COLS,
    sparse_attn_math,
    build_tensor_specs as build_sparse_attn_tensor_specs,
    golden_prefill_sparse_attn,
)
from qkv_proj_rope import (
    build_tensor_specs as build_qkv_tensor_specs,
    golden_qkv_proj_rope,
    materialize_rope_rows,
    qkv_proj_rope,
)
from rmsnorm import golden_rms_norm, rms_norm
from utils import build_rope_tables

# model config
D = M.hidden_size
H = M.num_attention_heads
HC_DIM = M.hc_dim
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
O_GROUPS = M.o_groups
O_GROUP_IN = H * M.head_dim // O_GROUPS
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim

# CP layout
T = 128
COMPRESS_RATIO = 4
STATE_LEN = 8
WIN = M.sliding_window
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_SEED_ROWS = COMPRESS_RATIO + 3
MAX_COMPRESSED_ROWS_PER_TILE = T // COMPRESS_RATIO
MAX_COMPRESSED_ROWS_PER_SEGMENT = 2 * MAX_COMPRESSED_ROWS_PER_TILE

MAIN_STATE_BLOCK_SIZE = 4
INNER_STATE_BLOCK_SIZE = 4
MAIN_STATE_MAX_BLOCKS = (M.max_position_embeddings + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = (M.max_position_embeddings + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE

CP_CANDIDATE_CAPACITY = 1024
SPARSE_SELECTED_WIDTH = 256

NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
# §8.17.8e.2 leaf-capture completion token: number of x_out tile producers
# (== NUM_LOCAL_TILES here) and also the row count of the rank-local
# completion_token published by the terminal cp_csa_rank_complete task. Each
# tile producer writes a distinct token row so no write is dead-store-
# eliminated. Mirrors prefill_cp_fwd.py:NUM_MOE_WAVES / prefill_cp_layer.py.
NUM_MOE_WAVES = NUM_LOCAL_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * T
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
MAIN_LEAF_CACHE_BLOCKS = PREFILL_CMP_BLOCK_NUM
MAIN_LEAF_CACHE_ROWS = MAIN_LEAF_CACHE_BLOCKS * CMP_STORAGE_BLOCK_SIZE
IDX_LEAF_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
IDX_LEAF_CACHE_ROWS = IDX_LEAF_CACHE_BLOCKS * CMP_STORAGE_BLOCK_SIZE
LOCAL_LEAVES = LOCAL_PARTS * MAX_COMPRESS_LEAVES
IDX_CACHE_ROWS = PREFILL_IDX_BLOCK_NUM * CMP_STORAGE_BLOCK_SIZE
QK_ROPE_HALF = ROPE_HEAD_DIM // 2

def owner_segments(cp_size: int) -> list[list[int]]:
    """Build rank/part ownership tables."""
    table = [[-1, -1] for _ in range(cp_size)]
    for segment in range(2 * cp_size):
        rank = cp_owner_rank(segment, cp_size)
        part = cp_owner_part(segment, cp_size)
        if table[rank][part] != -1:
            raise AssertionError(
                f"ownership collision at rank {rank}, part {part}"
            )
        table[rank][part] = segment
    if any(-1 in row for row in table):
        raise AssertionError(f"incomplete CP ownership table: {table}")
    if sorted(segment for row in table for segment in row) != list(range(2 * cp_size)):
        raise AssertionError("CP ownership is not a segment bijection")
    return table


def _segment_starts(prefix: int, span: int, nseg: int) -> list[int]:
    return [prefix + segment * span for segment in range(nseg)]


def _active_tile(length: int, tile: int) -> int:
    begin = tile * T
    return max(0, min(T, length - begin))


def _predecessor_segment(segment: int) -> int:
    return segment - 1 if segment > 0 else -1


def tail_start(segment_start: int, segment_len: int) -> int:
    return segment_start + max(0, segment_len - T)


def ring_phys_row(position: int) -> int:
    return position % ORI_CACHE_ROWS


def lower_key_build(
    key_abs: int,
    segment: int,
    tile: int,
    starts: list[int],
    lengths: list[int],
    prefix: int,
) -> int:
    tile_start = starts[segment] + tile * T
    tile_len = _active_tile(lengths[segment], tile)
    if key_abs < prefix:
        return ring_phys_row(key_abs)
    if tile_start <= key_abs < tile_start + tile_len:
        return ORI_CACHE_ROWS + T + key_abs - tile_start
    if key_abs >= tile_start:
        return -1
    if tile == 0:
        predecessor = _predecessor_segment(segment)
        if predecessor < 0:
            return -1
        predecessor_start = tail_start(starts[predecessor], lengths[predecessor])
        predecessor_len = min(T, lengths[predecessor])
    else:
        predecessor_start = tile_start - T
        predecessor_len = _active_tile(lengths[segment], tile - 1)
    if predecessor_start <= key_abs < predecessor_start + predecessor_len:
        return ORI_CACHE_ROWS + key_abs - predecessor_start
    return -1


def _seed_length(segment_start: int, segment: int, segment_length: int) -> int:
    if segment == 0 or segment_length <= 0:
        return 0
    return segment_start % COMPRESS_RATIO + COMPRESS_RATIO


def _lower_row(table: torch.Tensor, logical_slot: int, block_size: int) -> int:
    if logical_slot < 0:
        return -1
    logical_block = logical_slot // block_size
    if logical_block >= table.numel():
        return -1
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return physical_block * block_size + logical_slot % block_size


def validate_cp_indexer_capacity(candidate_history: int) -> None:
    """Validate the fixed CP indexer candidate capacity."""
    if candidate_history < 0:
        raise ValueError(f"candidate history must be non-negative, got {candidate_history}")
    if candidate_history > CP_CANDIDATE_CAPACITY:
        raise ValueError(
            f"CP-CSA candidate history {candidate_history} exceeds the supported "
            f"capacity {CP_CANDIDATE_CAPACITY}"
        )



def _build_block_tables(cp_size: int) -> dict[str, torch.Tensor]:
    def table(logical_blocks: int, physical_blocks: int) -> torch.Tensor:
        logical = torch.arange(logical_blocks, dtype=torch.int32)
        return (logical % physical_blocks).unsqueeze(0).repeat(cp_size, 1)

    return {
        "cmp_block_table": table(
            PREFILL_CMP_MAX_BLOCKS, PREFILL_CMP_BLOCK_NUM
        ),
        "idx_block_table": table(
            PREFILL_IDX_MAX_BLOCKS, PREFILL_IDX_BLOCK_NUM
        ),
        "compress_state_block_table": table(
            MAIN_STATE_MAX_BLOCKS, CSA_STATE_PHYSICAL_BLOCKS
        ),
        "inner_compress_state_block_table": table(
            INNER_STATE_MAX_BLOCKS, CSA_INNER_STATE_PHYSICAL_BLOCKS
        ),
    }


def _build_metadata_tensors(
    cp_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if cp_size not in CP_CHOICES:
        raise ValueError(f"CP size must be one of {CP_CHOICES}, got {cp_size}")

    prefix = 0
    span = T
    lengths = [T] * (2 * cp_size)
    nseg = 2 * cp_size
    starts = _segment_starts(prefix, span, nseg)
    owners = owner_segments(cp_size)
    request_end = max(
        (starts[segment] + lengths[segment] for segment in range(nseg) if lengths[segment]),
        default=prefix,
    )
    # Exclusive request end and compressed boundary count.
    candidate_history = request_end // COMPRESS_RATIO
    validate_cp_indexer_capacity(candidate_history)

    tables = _build_block_tables(cp_size)
    boundary_positions = torch.full(
        (nseg, MAX_COMPRESSED_ROWS_PER_SEGMENT), -1, dtype=torch.int32
    )
    main_logical_slots = torch.full_like(boundary_positions, -1)
    idx_logical_slots = torch.full_like(boundary_positions, -1)
    boundary_owner_rank = torch.full_like(boundary_positions, -1)
    boundary_owner_part = torch.full_like(boundary_positions, -1)
    boundary_count = torch.zeros(nseg, dtype=torch.int32)

    for segment in range(nseg):
        count = 0
        for position in range(starts[segment], starts[segment] + lengths[segment]):
            if (position + 1) % COMPRESS_RATIO != 0:
                continue
            if count >= MAX_COMPRESSED_ROWS_PER_SEGMENT:
                raise AssertionError(f"too many compressed rows in segment {segment}")
            logical_slot = (position + 1) // COMPRESS_RATIO - 1
            boundary_positions[segment, count] = position
            main_logical_slots[segment, count] = logical_slot
            idx_logical_slots[segment, count] = logical_slot
            boundary_owner_rank[segment, count] = cp_owner_rank(segment, cp_size)
            boundary_owner_part[segment, count] = cp_owner_part(segment, cp_size)
            count += 1
        boundary_count[segment] = count

    seed_positions = torch.full(
        (nseg, MAX_SEED_ROWS), -1, dtype=torch.int32
    )
    seed_lengths = torch.zeros(nseg, dtype=torch.int32)
    seed_cache_mapping = torch.full(
        (cp_size, nseg, MAX_SEED_ROWS), -1, dtype=torch.int32
    )
    seed_main_state_mapping = torch.full_like(seed_cache_mapping, -1)
    seed_inner_state_mapping = torch.full_like(seed_cache_mapping, -1)

    for segment in range(nseg):
        seed_len = _seed_length(starts[segment], segment, lengths[segment])
        seed_lengths[segment] = seed_len
        if seed_len == 0:
            continue
        seed_start = starts[segment] - seed_len
        seed_positions[segment, :seed_len] = torch.arange(
            seed_start, starts[segment], dtype=torch.int32
        )
        for rank in range(cp_size):
            main_table = tables["compress_state_block_table"][rank]
            inner_table = tables["inner_compress_state_block_table"][rank]
            for row in range(seed_len):
                position = seed_start + row
                seed_main_state_mapping[rank, segment, row] = _lower_row(
                    main_table, position, MAIN_STATE_BLOCK_SIZE
                )
                seed_inner_state_mapping[rank, segment, row] = _lower_row(
                    inner_table, position, INNER_STATE_BLOCK_SIZE
                )

    main_slot_mapping = torch.full(
        (cp_size, nseg, MAX_COMPRESSED_ROWS_PER_SEGMENT), -1, dtype=torch.int32
    )
    idx_slot_mapping = torch.full_like(main_slot_mapping, -1)
    for rank in range(cp_size):
        cmp_table = tables["cmp_block_table"][rank]
        idx_table = tables["idx_block_table"][rank]
        for segment in range(nseg):
            for row in range(MAX_COMPRESSED_ROWS_PER_SEGMENT):
                logical_main = int(main_logical_slots[segment, row].item())
                logical_idx = int(idx_logical_slots[segment, row].item())
                main_slot_mapping[rank, segment, row] = _lower_row(
                    cmp_table, logical_main, CMP_STORAGE_BLOCK_SIZE
                )
                idx_slot_mapping[rank, segment, row] = _lower_row(
                    idx_table, logical_idx, CMP_STORAGE_BLOCK_SIZE
                )

    segment_active_lengths = torch.zeros(
        (cp_size, LOCAL_PARTS), dtype=torch.int32
    )
    tile_active_lengths = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES), 0, dtype=torch.int32
    )
    predecessor_segments = torch.full(
        (cp_size, LOCAL_PARTS), -1, dtype=torch.int32
    )
    query_position_ids = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T), -1, dtype=torch.int32
    )
    query_segment_ids = torch.full_like(query_position_ids, -1)
    visible_candidate_lengths = torch.zeros_like(query_position_ids)

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            segment_active_lengths[rank, part] = lengths[segment]
            if lengths[segment] == 0:
                continue
            predecessor_segments[rank, part] = _predecessor_segment(segment)
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(lengths[segment], tile)
                tile_active_lengths[rank, part, tile] = active
                tile_start = starts[segment] + tile * T
                if active:
                    query_position_ids[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    query_segment_ids[rank, part, tile, :active] = segment
                    visible_candidate_lengths[rank, part, tile, :active] = torch.tensor(
                        [min((position + 1) // COMPRESS_RATIO, CP_CANDIDATE_CAPACITY)
                         for position in range(tile_start, tile_start + active)],
                        dtype=torch.int32,
                    )

    final_segment = max(
        (segment for segment, length in enumerate(lengths) if length),
        default=-1,
    )
    final_owner_rank = cp_owner_rank(final_segment, cp_size) if final_segment >= 0 else -1
    final_owner_part = cp_owner_part(final_segment, cp_size) if final_segment >= 0 else -1
    final_snapshot_positions = torch.full((STATE_LEN,), -1, dtype=torch.int32)
    snapshot_start = request_end - STATE_LEN
    for row in range(STATE_LEN):
        position = snapshot_start + row
        if position >= 0:
            final_snapshot_positions[row] = position

    final_main_state_mapping = torch.full(
        (cp_size, STATE_LEN), -1, dtype=torch.int32
    )
    final_inner_state_mapping = torch.full_like(final_main_state_mapping, -1)
    for rank in range(cp_size):
        main_table = tables["compress_state_block_table"][rank]
        inner_table = tables["inner_compress_state_block_table"][rank]
        for row, position in enumerate(final_snapshot_positions.tolist()):
            final_main_state_mapping[rank, row] = _lower_row(
                main_table, position, MAIN_STATE_BLOCK_SIZE
            )
            final_inner_state_mapping[rank, row] = _lower_row(
                inner_table, position, INNER_STATE_BLOCK_SIZE
            )

    dense_idx_prefix = torch.full(
        (cp_size, CP_CANDIDATE_CAPACITY), -1, dtype=torch.int32
    )
    dense_cmp_prefix = torch.full_like(dense_idx_prefix, -1)
    for rank in range(cp_size):
        for logical_slot in range(candidate_history):
            dense_idx_prefix[rank, logical_slot] = _lower_row(
                tables["idx_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            dense_cmp_prefix[rank, logical_slot] = _lower_row(
                tables["cmp_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )

    tensors = {
        "segment_starts": torch.tensor(starts, dtype=torch.int32),
        "segment_lengths": torch.tensor(lengths, dtype=torch.int32),
        "owner_segments": torch.tensor(owners, dtype=torch.int32),
        "reverse_index": cp_reverse_index(cp_size).to(torch.int32),
        "segment_active_lengths": segment_active_lengths,
        "tile_active_lengths": tile_active_lengths,
        "predecessor_segments": predecessor_segments,
        "query_position_ids": query_position_ids,
        "query_segment_ids": query_segment_ids,
        "visible_candidate_lengths": visible_candidate_lengths,
        "boundary_positions": boundary_positions,
        "boundary_count": boundary_count,
        "boundary_owner_rank": boundary_owner_rank,
        "boundary_owner_part": boundary_owner_part,
        "main_logical_slots": main_logical_slots,
        "idx_logical_slots": idx_logical_slots,
        "seed_positions": seed_positions,
        "seed_lengths": seed_lengths,
        "seed_cache_mapping": seed_cache_mapping,
        "seed_main_state_mapping": seed_main_state_mapping,
        "seed_inner_state_mapping": seed_inner_state_mapping,
        "cmp_block_table": tables["cmp_block_table"],
        "idx_block_table": tables["idx_block_table"],
        "compress_state_block_table": tables["compress_state_block_table"],
        "inner_compress_state_block_table": tables[
            "inner_compress_state_block_table"
        ],
        "main_slot_mapping": main_slot_mapping,
        "idx_slot_mapping": idx_slot_mapping,
        "final_snapshot_positions": final_snapshot_positions,
        "final_main_state_mapping": final_main_state_mapping,
        "final_inner_state_mapping": final_inner_state_mapping,
        "dense_cmp_prefix": dense_cmp_prefix,
        "dense_idx_prefix": dense_idx_prefix,
        "candidate_history": torch.tensor(candidate_history, dtype=torch.int32),
        "final_segment": torch.tensor(final_segment, dtype=torch.int32),
        "final_owner_rank": torch.tensor(final_owner_rank, dtype=torch.int32),
        "final_owner_part": torch.tensor(final_owner_part, dtype=torch.int32),
    }
    ctx = {
        "cp_size": cp_size,
        "prefix": prefix,
        "segment_span": span,
        "lengths": lengths,
        "starts": starts,
        "owners": owners,
        "request_end": request_end,
        "candidate_history": candidate_history,
        "final_segment": final_segment,
        "final_owner_rank": final_owner_rank,
        "final_owner_part": final_owner_part,
    }
    return tensors, ctx



def _build_raw_attention_metadata(cp_size: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Build canonical zero-history raw and overlay metadata."""
    prefix = 0
    span = T
    lengths = [T] * (2 * cp_size)
    starts = _segment_starts(prefix, span, 2 * cp_size)
    owners = owner_segments(cp_size)
    query_positions = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, dtype=torch.int32
    )
    query_requests = torch.full_like(query_positions, -1)
    overlay_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS),
        -1,
        dtype=torch.int32,
    )
    overlay_requests = torch.full_like(overlay_positions, -1)
    overlay_lengths = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        OVERLAY_SOURCES,
        dtype=torch.int32,
    )
    swa_indices = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN),
        -1,
        dtype=torch.int32,
    )
    segment_active = torch.zeros(cp_size, LOCAL_PARTS, dtype=torch.int32)
    predecessors = torch.full_like(segment_active, -1)

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            segment_len = lengths[segment]
            segment_active[rank, part] = segment_len
            predecessors[rank, part] = _predecessor_segment(segment)
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(segment_len, tile)
                tile_start = starts[segment] + tile * T
                if active:
                    query_positions[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    query_requests[rank, part, tile, :active] = 0
                if tile == 0:
                    predecessor = _predecessor_segment(segment)
                    if predecessor >= 0:
                        predecessor_len = min(T, lengths[predecessor])
                        predecessor_start = tail_start(
                            starts[predecessor], lengths[predecessor]
                        )
                    else:
                        predecessor_len = 0
                        predecessor_start = 0
                else:
                    predecessor_len = _active_tile(segment_len, tile - 1)
                    predecessor_start = starts[segment] + (tile - 1) * T
                if predecessor_len:
                    overlay_positions[
                        rank, part, tile, :predecessor_len
                    ] = torch.arange(
                        predecessor_start,
                        predecessor_start + predecessor_len,
                        dtype=torch.int32,
                    )
                    overlay_requests[rank, part, tile, :predecessor_len] = 0
                if active:
                    overlay_positions[
                        rank, part, tile, T : T + active
                    ] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    overlay_requests[rank, part, tile, T : T + active] = 0
                overlay_lengths[rank, part, tile, 0] = predecessor_len
                overlay_lengths[rank, part, tile, 1] = active
                for query_row in range(active):
                    query_abs = tile_start + query_row
                    for sparse_col in range(WIN):
                        key_abs = query_abs - WIN + 1 + sparse_col
                        if 0 <= key_abs <= query_abs:
                            swa_indices[
                                rank, part, tile, query_row, sparse_col
                            ] = lower_key_build(
                                key_abs,
                                segment,
                                tile,
                                starts,
                                lengths,
                                prefix,
                            )

    final_seg_src, final_row_src = cp_final_window_sources(lengths)
    final_slot_mapping = torch.full((T,), -1, dtype=torch.int32)
    total = sum(lengths)
    for row in range(T):
        position = prefix + total - T + row
        if position >= prefix:
            final_slot_mapping[row] = ring_phys_row(position)
    active_segments = [
        segment for segment, length in enumerate(lengths) if length > 0
    ]
    final_segment = active_segments[-1]
    owner_rank_table, _ = cp_owner_tables(cp_size)
    tensors = {
        "segment_starts_t": torch.tensor(starts, dtype=torch.int32),
        "segment_lengths_t": torch.tensor(lengths, dtype=torch.int32),
        "segment_active_lengths": segment_active,
        "owner_segments_t": torch.tensor(owners, dtype=torch.int32),
        "predecessor_segments": predecessors,
        "query_positions": query_positions,
        "query_requests": query_requests,
        "overlay_positions": overlay_positions,
        "overlay_requests": overlay_requests,
        "overlay_active_lengths": overlay_lengths,
        "swa_indices": swa_indices,
        "final_segment_t": torch.tensor([final_segment], dtype=torch.int32),
        "reverse_index": cp_reverse_index(cp_size).to(torch.int32),
        "owner_rank_table": owner_rank_table.to(torch.int32),
        "final_win_seg_src": final_seg_src.to(torch.int32),
        "final_win_row_src": final_row_src.to(torch.int32),
        "final_slot_mapping": final_slot_mapping,
    }
    return tensors, {
        "prefix": prefix,
        "span": span,
        "lengths": lengths,
        "starts": starts,
        "owners": owners,
        "final_segment": final_segment,
    }


def build_tensor_specs(cp_size: int = CP_SIZE):
    """Build the canonical CP-CSA fixture."""
    from golden import TensorSpec

    if cp_size != CP_SIZE:
        raise ValueError(
            f"runtime cp_size={cp_size} does not match static CP_SIZE={CP_SIZE}"
    )
    metadata, ctx = _build_metadata_tensors(cp_size)
    raw, raw_ctx = _build_raw_attention_metadata(cp_size)
    torch.manual_seed(4100 + cp_size * 31)
    qkv_specs = {spec.name: spec for spec in build_qkv_tensor_specs(1, T)}
    sparse_specs = {
        spec.name: spec
        for spec in build_sparse_attn_tensor_specs(COMPRESS_RATIO, T)
    }
    compressor_specs = {
        spec.name: spec for spec in build_compressor_tensor_specs(0)
    }
    indexer_specs = {
        spec.name: spec for spec in build_indexer_tensor_specs(0, T)
    }
    qkv_names = (
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
    )
    tail_names = ("attn_sink", "wo_a", "wo_b", "wo_b_scale")
    csa_values = {name: qkv_specs[name].create_tensor() for name in qkv_names}
    csa_values.update(
        {name: sparse_specs[name].create_tensor() for name in tail_names}
    )
    for source_name, target_name in (
        ("wkv", "cmp_wkv"),
        ("wgate", "cmp_wgate"),
        ("ape", "cmp_ape"),
        ("norm_w", "cmp_norm_w"),
    ):
        csa_values[target_name] = compressor_specs[source_name].create_tensor()
    for source_name, target_name in (
        ("hadamard", "hadamard_idx"),
        ("wq_b", "idx_wq_b"),
        ("wq_b_scale", "idx_wq_b_scale"),
        ("weights_proj", "idx_weights_proj"),
        ("inner_wkv", "inner_wkv"),
        ("inner_wgate", "inner_wgate"),
        ("inner_ape", "inner_ape"),
        ("inner_norm_w", "inner_norm_w"),
    ):
        csa_values[target_name] = indexer_specs[source_name].create_tensor()
    csa_values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    csa_values["hc_attn_scale"] = torch.randn(3)
    csa_values["hc_attn_base"] = torch.randn(MIX_HC)
    csa_values["attn_norm_w"] = torch.ones(D, dtype=torch.bfloat16)
    csa_values["freqs_cos"], csa_values["freqs_sin"] = (
        build_rope_tables(
            M, COMPRESS_RATIO, dtype=torch.bfloat16
        )
    )
    x_generator = torch.Generator().manual_seed(4100 + cp_size * 31)
    x_hc = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        HC_MULT,
        D,
        dtype=torch.float32,
    )
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            active = int(raw["segment_active_lengths"][rank, part])
            for tile in range(MAX_SEGMENT_TILES):
                tile_active = _active_tile(active, tile)
                if tile_active:
                    x_hc[rank, part, tile, :tile_active].uniform_(
                        -1.0, 1.0, generator=x_generator
                    )
    kv_cache = torch.zeros(
        cp_size,
        ORI_MAX_BLOCKS,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )

    shared_names = (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "freqs_cos",
        "freqs_sin",
        "cmp_wkv",
        "cmp_wgate",
        "cmp_ape",
        "cmp_norm_w",
        "hadamard_idx",
        "idx_wq_b",
        "idx_wq_b_scale",
        "idx_weights_proj",
        "inner_wkv",
        "inner_wgate",
        "inner_ape",
        "inner_norm_w",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
    )
    specs = [
        TensorSpec("x_hc", list(x_hc.shape), x_hc.dtype, init_value=x_hc)
    ]
    for name in shared_names:
        value = csa_values[name]
        specs.append(
            TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        )

    generator = torch.Generator().manual_seed(
        20260802 + cp_size * 101
    )
    main_state = torch.zeros(
        cp_size,
        CSA_STATE_PHYSICAL_BLOCKS,
        MAIN_STATE_BLOCK_SIZE,
        MAIN_STATE_DIM,
        dtype=torch.float32,
    )
    inner_state = torch.zeros(
        cp_size,
        CSA_INNER_STATE_PHYSICAL_BLOCKS,
        INNER_STATE_BLOCK_SIZE,
        INNER_STATE_DIM,
        dtype=torch.float32,
    )
    prefix = int(raw_ctx["prefix"])
    logical_main_state = {
        position: (torch.rand(MAIN_STATE_DIM, generator=generator) - 0.5) * 0.05
        for position in range(max(0, prefix - STATE_LEN), prefix)
    }
    logical_inner_state = {
        position: (torch.rand(INNER_STATE_DIM, generator=generator) - 0.5) * 0.05
        for position in range(max(0, prefix - STATE_LEN), prefix)
    }
    for rank in range(cp_size):
        main_flat = main_state[rank].view(-1, MAIN_STATE_DIM)
        inner_flat = inner_state[rank].view(-1, INNER_STATE_DIM)
        for position, value in logical_main_state.items():
            row = _lower_row(
                metadata["compress_state_block_table"][rank],
                position,
                MAIN_STATE_BLOCK_SIZE,
            )
            if row >= 0:
                main_flat[row] = value
        for position, value in logical_inner_state.items():
            row = _lower_row(
                metadata["inner_compress_state_block_table"][rank],
                position,
                INNER_STATE_BLOCK_SIZE,
            )
            if row >= 0:
                inner_flat[row] = value

    cmp_cache = torch.zeros(
        cp_size,
        PREFILL_CMP_BLOCK_NUM,
        CMP_STORAGE_BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    idx_cache = torch.zeros(
        cp_size,
        PREFILL_IDX_BLOCK_NUM,
        CMP_STORAGE_BLOCK_SIZE,
        1,
        IDX_HEAD_DIM,
        dtype=torch.int8,
    )
    idx_scale = torch.zeros(
        cp_size,
        PREFILL_IDX_BLOCK_NUM,
        CMP_STORAGE_BLOCK_SIZE,
        1,
        1,
        dtype=torch.float32,
    )
    completed = prefix // COMPRESS_RATIO
    logical_cmp = (
        (torch.rand(completed, HEAD_DIM, generator=generator) - 0.5)
        .to(torch.bfloat16)
        .contiguous()
    )
    logical_idx = torch.randint(
        -31,
        32,
        (completed, IDX_HEAD_DIM),
        generator=generator,
        dtype=torch.int8,
    )
    logical_scale = torch.rand(completed, 1, generator=generator) * 0.01 + 1e-4
    for rank in range(cp_size):
        cmp_flat = cmp_cache[rank].view(-1, HEAD_DIM)
        idx_flat = idx_cache[rank].view(-1, IDX_HEAD_DIM)
        scale_flat = idx_scale[rank].view(-1, 1)
        for logical_slot in range(completed):
            cmp_row = _lower_row(
                metadata["cmp_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            idx_row = _lower_row(
                metadata["idx_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            if cmp_row >= 0:
                cmp_flat[cmp_row] = logical_cmp[logical_slot]
            if idx_row >= 0:
                idx_flat[idx_row] = logical_idx[logical_slot]
                scale_flat[idx_row] = logical_scale[logical_slot]

    root_values = {
        "compress_state": main_state,
        "compress_state_block_table": metadata[
            "compress_state_block_table"
        ],
        "inner_compress_state": inner_state,
        "inner_compress_state_block_table": metadata[
            "inner_compress_state_block_table"
        ],
        "kv_cache": kv_cache,
        "cmp_kv": cmp_cache,
        "cmp_block_table": metadata["cmp_block_table"],
        "idx_kv_cache": idx_cache,
        "idx_kv_scale": idx_scale,
        "idx_block_table": metadata["idx_block_table"],
    }
    for name, value in root_values.items():
        specs.append(
            TensorSpec(
                name,
                list(value.shape),
                value.dtype,
                init_value=value,
                is_output=name
                in {
                    "compress_state",
                    "inner_compress_state",
                    "kv_cache",
                    "cmp_kv",
                    "idx_kv_cache",
                    "idx_kv_scale",
                },
            )
        )
    for name, value in raw.items():
        specs.append(
            TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        )

    leaf_shape = (cp_size, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T)
    leaf_positions_input = torch.zeros(leaf_shape, dtype=torch.int32)
    leaf_main_slots_input = torch.full(leaf_shape, -1, dtype=torch.int64)
    leaf_idx_slots_input = torch.full(leaf_shape, -1, dtype=torch.int64)
    leaf_main_state_slots_input = torch.full(leaf_shape, -1, dtype=torch.int64)
    leaf_inner_state_slots_input = torch.full(leaf_shape, -1, dtype=torch.int64)
    leaf_num_tokens_input = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_COMPRESS_LEAVES, dtype=torch.int32
    )
    starts = [int(value) for value in raw_ctx["starts"]]
    lengths = [int(value) for value in raw_ctx["lengths"]]
    owners = raw_ctx["owners"]
    for rank in range(cp_size):
        main_state_table = metadata["compress_state_block_table"][rank]
        inner_state_table = metadata["inner_compress_state_block_table"][rank]
        cmp_table = metadata["cmp_block_table"][rank]
        idx_table = metadata["idx_block_table"][rank]
        for part in range(LOCAL_PARTS):
            segment = int(owners[rank][part])
            predecessor = _predecessor_segment(segment)
            predecessor_length = (
                min(lengths[predecessor], T) if predecessor >= 0 else 0
            )
            seed_length = (
                min(starts[segment] % COMPRESS_RATIO + COMPRESS_RATIO,
                    predecessor_length)
                if predecessor >= 0 and lengths[segment] > 0
                else 0
            )
            leaf_num_tokens_input[rank, part, 0] = seed_length
            for row in range(seed_length):
                position = starts[segment] - seed_length + row
                leaf_positions_input[rank, part, 0, row] = position
                leaf_main_state_slots_input[rank, part, 0, row] = _lower_row(
                    main_state_table, position, MAIN_STATE_BLOCK_SIZE
                )
                leaf_inner_state_slots_input[rank, part, 0, row] = _lower_row(
                    inner_state_table, position, INNER_STATE_BLOCK_SIZE
                )
            for tile in range(MAX_SEGMENT_TILES):
                leaf = 1 + tile
                active = _active_tile(lengths[segment], tile)
                leaf_num_tokens_input[rank, part, leaf] = active
                for row in range(active):
                    position = int(raw["query_positions"][rank, part, tile, row])
                    leaf_positions_input[rank, part, leaf, row] = position
                    leaf_main_state_slots_input[rank, part, leaf, row] = _lower_row(
                        main_state_table, position, MAIN_STATE_BLOCK_SIZE
                    )
                    leaf_inner_state_slots_input[rank, part, leaf, row] = _lower_row(
                        inner_state_table, position, INNER_STATE_BLOCK_SIZE
                    )
                    if (position + 1) % COMPRESS_RATIO == 0:
                        logical_slot = (position + 1) // COMPRESS_RATIO - 1
                        leaf_main_slots_input[rank, part, leaf, row] = _lower_row(
                            cmp_table, logical_slot, CMP_STORAGE_BLOCK_SIZE
                        )
                        leaf_idx_slots_input[rank, part, leaf, row] = _lower_row(
                            idx_table, logical_slot, CMP_STORAGE_BLOCK_SIZE
                        )
    for name, value in {
        "leaf_positions_input": leaf_positions_input,
        "leaf_main_slots_input": leaf_main_slots_input,
        "leaf_idx_slots_input": leaf_idx_slots_input,
        "leaf_main_state_slots_input": leaf_main_state_slots_input,
        "leaf_inner_state_slots_input": leaf_inner_state_slots_input,
        "leaf_num_tokens_input": leaf_num_tokens_input,
    }.items():
        specs.append(
            TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        )
    specs.append(
        TensorSpec(
            "effective_x_workspace",
            [cp_size, LOCAL_LEAVES * T, D],
            torch.bfloat16,
            init_value=0.0,
        )
    )
    for part in range(LOCAL_PARTS):
        specs.extend(
            [
                TensorSpec(
                    f"main_state_workspace{part}",
                    [
                        cp_size,
                        CSA_STATE_PHYSICAL_BLOCKS,
                        MAIN_STATE_BLOCK_SIZE,
                        MAIN_STATE_DIM,
                    ],
                    torch.float32,
                    init_value=0.0,
                ),
                TensorSpec(
                    f"inner_state_workspace{part}",
                    [
                        cp_size,
                        CSA_INNER_STATE_PHYSICAL_BLOCKS,
                        INNER_STATE_BLOCK_SIZE,
                        INNER_STATE_DIM,
                    ],
                    torch.float32,
                    init_value=0.0,
                ),
            ]
        )
    specs.append(
        TensorSpec(
            "x_out", list(x_hc.shape), torch.float32, is_output=True
        )
    )
    golden_prefill_cp_csa._ctx = {
        "cp_size": cp_size,
        **raw_ctx,
    }
    return specs


@pl.jit.inline
def _cp_csa_compress_pack_part(
    effective_x: pl.Tensor[[MAX_COMPRESS_LEAVES * T, D], pl.BF16],
    leaf_positions: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT32],
    leaf_main_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64],
    leaf_idx_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64],
    leaf_main_state_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64],
    leaf_inner_state_slots: pl.Tensor[[MAX_COMPRESS_LEAVES * T], pl.INT64],
    leaf_num_tokens: pl.Tensor[[MAX_COMPRESS_LEAVES], pl.INT32],
    main_state_workspace: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32],
    inner_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    segment: pl.Scalar[pl.INT32],
    segment_start: pl.Scalar[pl.INT32],
    segment_active_length: pl.Scalar[pl.INT32],
    final_segment: pl.Scalar[pl.INT32],
):
    """Build one owner part's compact payloads."""
    # Receiver-local compressed cache and state roots.
    payload_rows = EPOCHS * MAX_COMPRESSED_ROWS_PER_SEGMENT
    state_payload_rows = EPOCHS * STATE_ROWS_PER_RANK
    main_cache = pl.create_tensor(
        [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
        dtype=pl.BF16,
        init_value=0.0,
    )
    idx_cache = pl.create_tensor(
        [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
        dtype=pl.INT8,
        init_value=0,
    )
    idx_scale = pl.create_tensor(
        [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, 1],
        dtype=pl.FP32,
        init_value=0.0,
    )
    main_payload = pl.create_tensor([payload_rows, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    idx_payload = pl.create_tensor([payload_rows, IDX_HEAD_DIM], dtype=pl.INT8, init_value=0)
    idx_scale_payload = pl.create_tensor([payload_rows, SCALE_TILE_COLS], dtype=pl.FP32, init_value=0.0)
    record_meta = pl.create_tensor([payload_rows, META_DIM], dtype=pl.INT32, init_value=-1)
    main_state_payload = pl.create_tensor([state_payload_rows, MAIN_STATE_DIM], dtype=pl.FP32, init_value=0.0)
    inner_state_payload = pl.create_tensor([state_payload_rows, INNER_STATE_DIM], dtype=pl.FP32, init_value=0.0)
    main_state_meta = pl.create_tensor([state_payload_rows, STATE_META_DIM], dtype=pl.INT32, init_value=-1)
    inner_state_meta = pl.create_tensor([state_payload_rows, STATE_META_DIM], dtype=pl.INT32, init_value=-1)
    main_state = main_state_workspace
    inner_state = inner_state_workspace

    # Each compressor leaf is a coarse stage.
    for leaf in pl.unroll(MAX_COMPRESS_LEAVES):
        row0 = leaf * T
        x_leaf = pl.slice(effective_x, [T, D], [row0, 0])
        pos_leaf = pl.slice(leaf_positions, [T], [row0])
        main_map_leaf = pl.slice(leaf_main_slots, [T], [row0])
        idx_map_leaf = pl.slice(leaf_idx_slots, [T], [row0])
        main_state_map_leaf = pl.slice(leaf_main_state_slots, [T], [row0])
        inner_state_map_leaf = pl.slice(leaf_inner_state_slots, [T], [row0])
        active = pl.read(leaf_num_tokens, [leaf])
        active_eff = pl.max(active, 1)

        main_completion = pl.array.create(1, pl.TASK_ID)
        main_cache_written, main_state_written = compressor_ratio4(
            x_leaf, main_state, main_state_block_table,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            freqs_cos, freqs_sin,
            main_cache, pos_leaf, active_eff,
            main_map_leaf, main_state_map_leaf,
            main_completion,
        )
        inner_completion = pl.array.create(1, pl.TASK_ID)
        idx_cache_written, idx_scale_written, inner_state_written = _prefill_indexer_compressor_with_completion(
            x_leaf, inner_state, inner_state_block_table,
            inner_wkv, inner_wgate, inner_ape, inner_norm_w,
            freqs_cos, freqs_sin, hadamard_idx,
            idx_cache, idx_scale, idx_block_table,
            pos_leaf, active_eff, idx_map_leaf, inner_state_map_leaf,
            inner_completion,
        )

        main_state_next = pl.create_tensor(
            [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            dtype=pl.FP32,
        )
        inner_state_next = pl.create_tensor(
            [
                CSA_INNER_STATE_PHYSICAL_BLOCKS,
                INNER_STATE_BLOCK_SIZE,
                INNER_STATE_DIM,
            ],
            dtype=pl.FP32,
        )
        main_cache_flat = pl.reshape(main_cache_written, [MAIN_LEAF_CACHE_ROWS, HEAD_DIM])
        idx_cache_flat = pl.reshape(idx_cache_written, [IDX_LEAF_CACHE_ROWS, IDX_HEAD_DIM])
        idx_scale_flat = pl.reshape(idx_scale_written, [IDX_LEAF_CACHE_ROWS, 1])
        main_state_written_flat = pl.reshape(main_state_written, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
        inner_state_written_flat = pl.reshape(inner_state_written, [INNER_STATE_ROWS, INNER_STATE_DIM])
        main_state_next_flat = pl.reshape(main_state_next, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
        inner_state_next_flat = pl.reshape(inner_state_next, [INNER_STATE_ROWS, INNER_STATE_DIM])
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="cp_csa_materialize_leaf",
            deps=[main_completion[0], inner_completion[0]],
        ):
            for state_row in pl.range(MAIN_STATE_ROWS):
                main_state_next_flat[state_row : state_row + 1, :] = (
                    main_state_written_flat[state_row : state_row + 1, :]
                )
            for state_row in pl.range(INNER_STATE_ROWS):
                inner_state_next_flat[state_row : state_row + 1, :] = (
                    inner_state_written_flat[state_row : state_row + 1, :]
                )
            if leaf > 0:
                segment_slot0 = segment_start // COMPRESS_RATIO
                for row in pl.range(T):
                    if row < active:
                        position = pl.read(pos_leaf, [row])
                        main_source_i64 = pl.read(main_map_leaf, [row])
                        idx_source_i64 = pl.read(idx_map_leaf, [row])
                        if main_source_i64 >= 0 and idx_source_i64 >= 0:
                            logical_slot = pl.cast(
                                (position + 1) // COMPRESS_RATIO - 1,
                                pl.INT32,
                            )
                            record_row = logical_slot - segment_slot0
                            main_source = pl.cast(main_source_i64, pl.INDEX)
                            idx_source = pl.cast(idx_source_i64, pl.INDEX)
                            for epoch in pl.range(EPOCHS):
                                destination = (
                                    epoch * MAX_COMPRESSED_ROWS_PER_SEGMENT
                                    + record_row
                                )
                                main_payload[
                                    destination : destination + 1, :
                                ] = main_cache_flat[
                                    main_source : main_source + 1, :
                                ]
                                idx_payload[
                                    destination : destination + 1, :
                                ] = idx_cache_flat[
                                    idx_source : idx_source + 1, :
                                ]
                                scale_value = pl.read(
                                    idx_scale_flat, [idx_source, 0]
                                )
                                for scale_col in pl.range(SCALE_TILE_COLS):
                                    pl.write(
                                        idx_scale_payload,
                                        [destination, scale_col],
                                        scale_value,
                                    )
                                pl.write(
                                    record_meta,
                                    [destination, 0],
                                    pl.cast(1, pl.INT32),
                                )
                                pl.write(record_meta, [destination, 1], segment)
                                pl.write(record_meta, [destination, 2], position)
                                pl.write(
                                    record_meta,
                                    [destination, 3],
                                    pl.cast(record_row, pl.INT32),
                                )
                                pl.write(
                                    record_meta,
                                    [destination, 4],
                                    pl.cast(1, pl.INT32),
                                )
                                pl.write(
                                    record_meta,
                                    [destination, 5],
                                    logical_slot,
                                )
                                pl.write(
                                    record_meta,
                                    [destination, 6],
                                    pl.cast(1, pl.INT32),
                                )
                                pl.write(
                                    record_meta,
                                    [destination, 7],
                                    logical_slot,
                                )
        main_state = main_state_next
        inner_state = inner_state_next

    main_state_flat = pl.reshape(main_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    inner_state_flat = pl.reshape(
        inner_state, [INNER_STATE_ROWS, INNER_STATE_DIM]
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_pack_final_state"):
        if segment == final_segment:
            segment_end = segment_start + segment_active_length
            state_start = pl.max(segment_end - STATE_LEN, 0)
            for state_row in pl.range(STATE_ROWS_PER_RANK):
                position = pl.cast(state_start + state_row, pl.INT32)
                main_block = pl.read(
                    main_state_block_table,
                    [position // MAIN_STATE_BLOCK_SIZE],
                )
                inner_block = pl.read(
                    inner_state_block_table,
                    [position // INNER_STATE_BLOCK_SIZE],
                )
                if position < segment_end and main_block >= 0 and inner_block >= 0:
                    main_source = (
                        pl.cast(main_block, pl.INDEX) * MAIN_STATE_BLOCK_SIZE
                        + position % MAIN_STATE_BLOCK_SIZE
                    )
                    inner_source = (
                        pl.cast(inner_block, pl.INDEX) * INNER_STATE_BLOCK_SIZE
                        + position % INNER_STATE_BLOCK_SIZE
                    )
                    for epoch in pl.range(EPOCHS):
                        destination = epoch * STATE_ROWS_PER_RANK + state_row
                        main_state_payload[destination : destination + 1, :] = (
                            main_state_flat[main_source : main_source + 1, :]
                        )
                        inner_state_payload[destination : destination + 1, :] = (
                            inner_state_flat[inner_source : inner_source + 1, :]
                        )
                        pl.write(
                            main_state_meta,
                            [destination, 0],
                            pl.cast(1, pl.INT32),
                        )
                        pl.write(main_state_meta, [destination, 1], position)
                        pl.write(main_state_meta, [destination, 2], position)
                        pl.write(
                            inner_state_meta,
                            [destination, 0],
                            pl.cast(1, pl.INT32),
                        )
                        pl.write(inner_state_meta, [destination, 1], position)
                        pl.write(inner_state_meta, [destination, 2], position)

    return (
        main_payload,
        idx_payload,
        idx_scale_payload,
        record_meta,
        main_state_payload,
        inner_state_payload,
        main_state_meta,
        inner_state_meta,
    )


@pl.jit.inline
def prefill_cp_csa_core(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    main_state_workspace0: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
        ]
    ],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32
    ],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[T], pl.INT32],
    final_win_row_src: pl.Tensor[[T], pl.INT32],
    final_slot_mapping: pl.Tensor[[T], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32
    ],
    effective_x_workspace: pl.Tensor[[LOCAL_LEAVES * T, D], pl.BF16],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, HEAD_DIM], pl.BF16
    ],
    idx_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8
    ],
    scale_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP32
    ],
    record_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, META_DIM], pl.INT32
    ],
    main_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    # §8.17.8e.2 leaf-capture completion token. Published atomically by the
    # terminal cp_csa_rank_complete task (deps=[resource_done_tid]), which fans
    # the four leaf-internal commit/transport TIDs via a pl.system.task_dummy.
    # [NUM_MOE_WAVES, 1, 8] = 4 rows x 32B; one row per x_out tile producer so
    # no row write is dead-store-eliminated. Consumed downstream by
    # _attention_stage_barrier (copies row 0 into stage_token). The token's
    # payload value is irrelevant -- only its producer/consumer edges matter.
    # Placed before the scalars (Scalars-last rule: no tensor arg after a
    # scalar arg in runtime TaskArgs).
    completion_token: pl.Out[
        pl.Tensor[[NUM_MOE_WAVES, 1, 8], pl.FP32]
    ],
    my_rank: pl.Scalar[pl.INT32],
    tail_comm_epoch: pl.Scalar[pl.INT32],
    compact_comm_epoch_base: pl.Scalar[pl.INT32],
):
    """Run inline CP-CSA attention for one rank.

    ``tail_comm_epoch`` drives the shared cross-layer tail ready/consumed
    counters; ``compact_comm_epoch_base`` is the base for the CSA compact
    counters (``compact_comm_epoch_base + local_epoch`` per local wave).
    Local payload rows stay at ``local_epoch`` (``EPOCHS == 1`` in this
    phase). Standalone/single-layer callers pass 0 for both.
    """
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    local_kv = pl.create_tensor([LOCAL_ROWS, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    normed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16, init_value=0.0)
    mixed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    qr = pl.create_tensor([LOCAL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([LOCAL_ROWS, 1], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    query_positions_flat = pl.reshape(query_positions, [LOCAL_ROWS])
    query_requests_flat = pl.reshape(query_requests, [LOCAL_ROWS])
    overlay_positions_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_requests_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    overlay_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_indices_flat = pl.reshape(swa_indices, [LOCAL_ROWS, WIN])
    predecessor_segments_local = predecessor_segments
    segment_starts_local = segment_starts_t
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * T
        x_tile = pl.slice(x_flat, [T, HC_MULT, D], [row0, 0, 0])
        mixed_tile = pl.slice(mixed, [T, D], [row0, 0])
        post_tile = pl.slice(post, [T, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [T, HC_MULT * HC_MULT], [row0, 0])
        position_tile = pl.slice(query_positions_flat, [T], [row0])
        cos_tile = pl.slice(rope_cos_flat, [T, ROPE_HEAD_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [T, ROPE_HEAD_DIM], [row0, 0])
        normed_tile = pl.slice(normed, [T, D], [row0, 0])
        q_tile = pl.slice(q, [T, H, HEAD_DIM], [row0, 0, 0])
        kv_tile = pl.slice(local_kv, [T, HEAD_DIM], [row0, 0])
        qr_tile = pl.slice(qr, [T, Q_LORA], [row0, 0])
        qr_scale_tile = pl.slice(qr_scale, [T, 1], [row0, 0])
        hc_pre(
            x_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
            mixed_tile, post_tile, comb_tile,
        )
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_flat, [tile, 1])
        materialize_rope_rows(
            freqs_cos, freqs_sin, position_tile, active,
            cos_tile, sin_tile,
        )
        qkv_proj_rope(
            normed_tile,
            wq_a, wq_b, wq_b_scale, wkv,
            cos_tile, sin_tile,
            gamma_cq, gamma_ckv,
            q_tile, kv_tile, qr_tile, qr_scale_tile, late_dep,
        )

    local_hidden_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * T, D], dtype=pl.BF16, init_value=0.0)
    local_kv_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * T, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_tail_assemble"):
        for part in pl.range(LOCAL_PARTS):
            active = pl.read(segment_active_lengths, [part])
            valid = pl.min(active, T)
            tail0 = pl.max(active - T, 0)
            for row in pl.range(T):
                if row < valid:
                    source = part * MAX_SEGMENT_TILES * T + tail0 + row
                    destination = part * T + row
                    local_hidden_tail[destination : destination + 1, :] = normed[source : source + 1, :]
                    local_kv_tail[destination : destination + 1, :] = local_kv[source : source + 1, :]

    logical_hidden = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    logical_kv = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_tail_exchange") as tail_exchange_tid:
        _prefill_cp_dual_tail_exchange_wave(
            local_hidden_tail, local_kv_tail,
            reverse_index, owner_rank_table,
            hidden_tail_window, kv_tail_window, tail_ready, tail_consumed,
            logical_hidden, logical_kv,
            my_rank, pl.cast(0, pl.INT32), tail_comm_epoch,
        )

    effective_x = effective_x_workspace
    leaf_positions = pl.reshape(leaf_positions_input, [LOCAL_LEAVES * T])
    leaf_main_slots = pl.reshape(leaf_main_slots_input, [LOCAL_LEAVES * T])
    leaf_idx_slots = pl.reshape(leaf_idx_slots_input, [LOCAL_LEAVES * T])
    leaf_main_state_slots = pl.reshape(leaf_main_state_slots_input, [LOCAL_LEAVES * T])
    leaf_inner_state_slots = pl.reshape(leaf_inner_state_slots_input, [LOCAL_LEAVES * T])
    leaf_num_tokens = leaf_num_tokens_input
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_leaf_data_stage"):
        for part in pl.range(LOCAL_PARTS):
            predecessor = pl.read(predecessor_segments_local, [part])
            predecessor_length = pl.cast(0, pl.INT32)
            if predecessor >= 0:
                predecessor_length = pl.cast(
                    pl.min(pl.read(segment_lengths_t, [predecessor]), T),
                    pl.INT32,
                )
            seed_length = pl.read(leaf_num_tokens, [part, 0])
            seed_source0 = predecessor_length - seed_length
            for row in pl.range(T):
                leaf_row = (part * MAX_COMPRESS_LEAVES) * T + row
                if row < seed_length:
                    source = predecessor * T + seed_source0 + row
                    effective_x[leaf_row : leaf_row + 1, :] = logical_hidden[
                        source : source + 1, :
                    ]
            for tile in pl.range(MAX_SEGMENT_TILES):
                leaf = 1 + tile
                active = pl.read(leaf_num_tokens, [part, leaf])
                for row in pl.range(T):
                    source = (part * MAX_SEGMENT_TILES + tile) * T + row
                    leaf_row = (part * MAX_COMPRESS_LEAVES + leaf) * T + row
                    if row < active:
                        effective_x[leaf_row : leaf_row + 1, :] = normed[
                            source : source + 1, :
                        ]

    persistent_main_flat = pl.reshape(compress_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM])
    persistent_inner_flat = pl.reshape(
        inner_compress_state, [INNER_STATE_ROWS, INNER_STATE_DIM]
    )
    scratch_main0_flat = pl.reshape(
        main_state_workspace0, [MAIN_STATE_ROWS, MAIN_STATE_DIM]
    )
    scratch_main1_flat = pl.reshape(
        main_state_workspace1, [MAIN_STATE_ROWS, MAIN_STATE_DIM]
    )
    scratch_inner0_flat = pl.reshape(
        inner_state_workspace0, [INNER_STATE_ROWS, INNER_STATE_DIM]
    )
    scratch_inner1_flat = pl.reshape(
        inner_state_workspace1, [INNER_STATE_ROWS, INNER_STATE_DIM]
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_seed_persistent_state"):
        if pl.read(owner_segments_t, [0]) == 0:
            for row in pl.range(MAIN_STATE_ROWS):
                scratch_main0_flat[row : row + 1, :] = persistent_main_flat[
                    row : row + 1, :
                ]
            for row in pl.range(INNER_STATE_ROWS):
                scratch_inner0_flat[row : row + 1, :] = persistent_inner_flat[
                    row : row + 1, :
                ]
        if pl.read(owner_segments_t, [1]) == 0:
            for row in pl.range(MAIN_STATE_ROWS):
                scratch_main1_flat[row : row + 1, :] = persistent_main_flat[
                    row : row + 1, :
                ]
            for row in pl.range(INNER_STATE_ROWS):
                scratch_inner1_flat[row : row + 1, :] = persistent_inner_flat[
                    row : row + 1, :
                ]

    part_leaf_rows = MAX_COMPRESS_LEAVES * T
    leaf_num_tokens_flat = pl.reshape(leaf_num_tokens, [LOCAL_LEAVES])
    final_segment = pl.read(final_segment_t, [0])

    part0_x = pl.slice(effective_x, [part_leaf_rows, D], [0, 0])
    part0_positions = pl.slice(leaf_positions, [part_leaf_rows], [0])
    part0_main_slots = pl.slice(leaf_main_slots, [part_leaf_rows], [0])
    part0_idx_slots = pl.slice(leaf_idx_slots, [part_leaf_rows], [0])
    part0_main_state_slots = pl.slice(
        leaf_main_state_slots, [part_leaf_rows], [0]
    )
    part0_inner_state_slots = pl.slice(
        leaf_inner_state_slots, [part_leaf_rows], [0]
    )
    part0_num_tokens = pl.slice(
        leaf_num_tokens_flat, [MAX_COMPRESS_LEAVES], [0]
    )
    part0_segment = pl.read(owner_segments_t, [0])
    part0_segment_start = pl.read(
        segment_starts_local, [pl.cast(part0_segment, pl.INDEX)]
    )
    (
        part0_main_payload,
        part0_idx_payload,
        part0_idx_scale_payload,
        part0_record_meta,
        part0_main_state_payload,
        part0_inner_state_payload,
        part0_main_state_meta,
        part0_inner_state_meta,
    ) = _cp_csa_compress_pack_part(
        part0_x,
        part0_positions,
        part0_main_slots,
        part0_idx_slots,
        part0_main_state_slots,
        part0_inner_state_slots,
        part0_num_tokens,
        main_state_workspace0,
        inner_state_workspace0,
        compress_state_block_table,
        inner_compress_state_block_table,
        idx_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        freqs_cos,
        freqs_sin,
        hadamard_idx,
        part0_segment,
        part0_segment_start,
        pl.read(segment_active_lengths, [0]),
        final_segment,
    )

    part1_row0 = part_leaf_rows
    part1_x = pl.slice(effective_x, [part_leaf_rows, D], [part1_row0, 0])
    part1_positions = pl.slice(leaf_positions, [part_leaf_rows], [part1_row0])
    part1_main_slots = pl.slice(
        leaf_main_slots, [part_leaf_rows], [part1_row0]
    )
    part1_idx_slots = pl.slice(leaf_idx_slots, [part_leaf_rows], [part1_row0])
    part1_main_state_slots = pl.slice(
        leaf_main_state_slots, [part_leaf_rows], [part1_row0]
    )
    part1_inner_state_slots = pl.slice(
        leaf_inner_state_slots, [part_leaf_rows], [part1_row0]
    )
    part1_num_tokens = pl.slice(
        leaf_num_tokens_flat,
        [MAX_COMPRESS_LEAVES],
        [MAX_COMPRESS_LEAVES],
    )
    part1_segment = pl.read(owner_segments_t, [1])
    part1_segment_start = pl.read(
        segment_starts_local, [pl.cast(part1_segment, pl.INDEX)]
    )
    (
        part1_main_payload,
        part1_idx_payload,
        part1_idx_scale_payload,
        part1_record_meta,
        part1_main_state_payload,
        part1_inner_state_payload,
        part1_main_state_meta,
        part1_inner_state_meta,
    ) = _cp_csa_compress_pack_part(
        part1_x,
        part1_positions,
        part1_main_slots,
        part1_idx_slots,
        part1_main_state_slots,
        part1_inner_state_slots,
        part1_num_tokens,
        main_state_workspace1,
        inner_state_workspace1,
        compress_state_block_table,
        inner_compress_state_block_table,
        idx_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        freqs_cos,
        freqs_sin,
        hadamard_idx,
        part1_segment,
        part1_segment_start,
        pl.read(segment_active_lengths, [1]),
        final_segment,
    )

    packed_main_payload = pl.create_tensor(
        [EPOCHS * ROWS_PER_RANK, HEAD_DIM], dtype=pl.BF16, init_value=0.0
    )
    packed_idx_payload = pl.create_tensor(
        [EPOCHS * ROWS_PER_RANK, IDX_HEAD_DIM], dtype=pl.INT8, init_value=0
    )
    packed_idx_scale_payload = pl.create_tensor(
        [EPOCHS * ROWS_PER_RANK, SCALE_TILE_COLS],
        dtype=pl.FP32,
        init_value=0.0,
    )
    packed_record_meta = pl.create_tensor(
        [EPOCHS * ROWS_PER_RANK, META_DIM], dtype=pl.INT32, init_value=-1
    )
    packed_main_state_payload = pl.create_tensor(
        [EPOCHS * STATE_ROWS_PER_RANK, MAIN_STATE_DIM],
        dtype=pl.FP32,
        init_value=0.0,
    )
    packed_inner_state_payload = pl.create_tensor(
        [EPOCHS * STATE_ROWS_PER_RANK, INNER_STATE_DIM],
        dtype=pl.FP32,
        init_value=0.0,
    )
    packed_main_state_meta = pl.create_tensor(
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM],
        dtype=pl.INT32,
        init_value=-1,
    )
    packed_inner_state_meta = pl.create_tensor(
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM],
        dtype=pl.INT32,
        init_value=-1,
    )

    # Compressor-to-communication boundary.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_merge_part_payloads"):
        for epoch in pl.range(EPOCHS):
            source_row0 = epoch * MAX_COMPRESSED_ROWS_PER_SEGMENT
            destination_epoch0 = epoch * ROWS_PER_RANK
            destination_part1 = (
                destination_epoch0 + MAX_COMPRESSED_ROWS_PER_SEGMENT
            )
            for row in pl.range(MAX_COMPRESSED_ROWS_PER_SEGMENT):
                source = source_row0 + row
                destination0 = destination_epoch0 + row
                destination1 = destination_part1 + row
                packed_main_payload[destination0 : destination0 + 1, :] = (
                    part0_main_payload[source : source + 1, :]
                )
                packed_main_payload[destination1 : destination1 + 1, :] = (
                    part1_main_payload[source : source + 1, :]
                )
                packed_idx_payload[destination0 : destination0 + 1, :] = (
                    part0_idx_payload[source : source + 1, :]
                )
                packed_idx_payload[destination1 : destination1 + 1, :] = (
                    part1_idx_payload[source : source + 1, :]
                )
                packed_idx_scale_payload[
                    destination0 : destination0 + 1, :
                ] = part0_idx_scale_payload[source : source + 1, :]
                packed_idx_scale_payload[
                    destination1 : destination1 + 1, :
                ] = part1_idx_scale_payload[source : source + 1, :]
                for col in pl.range(META_DIM):
                    pl.write(
                        packed_record_meta,
                        [destination0, col],
                        pl.read(part0_record_meta, [source, col]),
                    )
                    pl.write(
                        packed_record_meta,
                        [destination1, col],
                        pl.read(part1_record_meta, [source, col]),
                    )

            state_source0 = epoch * STATE_ROWS_PER_RANK
            for row in pl.range(STATE_ROWS_PER_RANK):
                state_source = state_source0 + row
                state_destination = state_source
                if pl.read(owner_segments_t, [0]) == final_segment:
                    packed_main_state_payload[
                        state_destination : state_destination + 1, :
                    ] = part0_main_state_payload[
                        state_source : state_source + 1, :
                    ]
                    packed_inner_state_payload[
                        state_destination : state_destination + 1, :
                    ] = part0_inner_state_payload[
                        state_source : state_source + 1, :
                    ]
                    for col in pl.range(STATE_META_DIM):
                        pl.write(
                            packed_main_state_meta,
                            [state_destination, col],
                            pl.read(part0_main_state_meta, [state_source, col]),
                        )
                        pl.write(
                            packed_inner_state_meta,
                            [state_destination, col],
                            pl.read(part0_inner_state_meta, [state_source, col]),
                        )
                if pl.read(owner_segments_t, [1]) == final_segment:
                    packed_main_state_payload[
                        state_destination : state_destination + 1, :
                    ] = part1_main_state_payload[
                        state_source : state_source + 1, :
                    ]
                    packed_inner_state_payload[
                        state_destination : state_destination + 1, :
                    ] = part1_inner_state_payload[
                        state_source : state_source + 1, :
                    ]
                    for col in pl.range(STATE_META_DIM):
                        pl.write(
                            packed_main_state_meta,
                            [state_destination, col],
                            pl.read(part1_main_state_meta, [state_source, col]),
                        )
                        pl.write(
                            packed_inner_state_meta,
                            [state_destination, col],
                            pl.read(part1_inner_state_meta, [state_source, col]),
                        )

    cmp_flat = pl.reshape(cmp_kv, [MAIN_CACHE_ROWS, HEAD_DIM])
    idx_flat = pl.reshape(idx_kv_cache, [IDX_CACHE_ROWS, IDX_HEAD_DIM])
    idx_scale_flat = pl.reshape(idx_kv_scale, [IDX_CACHE_ROWS, 1])
    main_state_flat = pl.reshape(
        compress_state, [MAIN_STATE_ROWS, MAIN_STATE_DIM]
    )
    inner_state_flat = pl.reshape(
        inner_compress_state, [INNER_STATE_ROWS, INNER_STATE_DIM]
    )
    # §8.17.8e.2 leaf-capture completion token: collect the TaskId of every
    # leaf-internal commit/transport task so the terminal cp_csa_rank_complete
    # task can fan them in via pl.system.task_dummy(deps=[...]). With EPOCHS==1
    # each array holds one entry; the array form keeps the epoch loop's existing
    # per-epoch structure (idiom: prefill_cp_csa.py:1084 pl.array.create(1, ...) +
    # :1125 deps=[main_completion[0], inner_completion[0]]).
    compact_transport_tids = pl.array.create(EPOCHS, pl.TASK_ID)
    receiver_commit_tids = pl.array.create(EPOCHS, pl.TASK_ID)
    for epoch in pl.range(EPOCHS):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="cp_csa_compact_transport",
        ) as compact_transport_tid:
            _prefill_cp_csa_compact_transport_wave(
                packed_main_payload,
                packed_idx_payload,
                packed_idx_scale_payload,
                packed_record_meta,
                packed_main_state_payload,
                packed_inner_state_payload,
                packed_main_state_meta,
                packed_inner_state_meta,
                main_window,
                idx_window,
                scale_window,
                record_window,
                main_state_window,
                main_state_meta_window,
                inner_state_window,
                inner_state_meta_window,
                compact_ready,
                compact_consumed,
                my_rank,
                pl.cast(epoch, pl.INT32),
                pl.cast(compact_comm_epoch_base + epoch, pl.INT32),
            )
        # Store the captured TaskId for this epoch (idiom:
        # prefill_sparse_attn.py:300 proj_a_tids[...] = pa_tid).
        compact_transport_tids[epoch] = compact_transport_tid
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="cp_csa_receiver_commit",
        ) as receiver_commit_tid:
            for source_rank in pl.range(CP_SIZE):
                source_row = source_rank * ROWS_PER_RANK
                source_state_row = source_rank * STATE_ROWS_PER_RANK
                for row in pl.range(ROWS_PER_RANK):
                    meta_row = source_row + row
                    record_valid = pl.read(record_window, [meta_row, 0])
                    logical_segment = pl.read(record_window, [meta_row, 1])
                    boundary = pl.read(record_window, [meta_row, 2])
                    if record_valid > 0 and logical_segment >= 0 and boundary >= 0:
                        main_valid = pl.read(record_window, [meta_row, 4])
                        main_slot = pl.read(record_window, [meta_row, 5])
                        if main_valid > 0 and main_slot >= 0:
                            logical_block = main_slot // CMP_STORAGE_BLOCK_SIZE
                            if logical_block < PREFILL_CMP_MAX_BLOCKS:
                                physical_block = pl.read(
                                    cmp_block_table, [logical_block]
                                )
                                if physical_block >= 0:
                                    destination = (
                                        pl.cast(physical_block, pl.INDEX)
                                        * CMP_STORAGE_BLOCK_SIZE
                                        + main_slot % CMP_STORAGE_BLOCK_SIZE
                                    )
                                    received_main_tile = main_window[meta_row : meta_row + 1, 0:HEAD_DIM]
                                    cmp_flat[destination : destination + 1, 0:HEAD_DIM] = received_main_tile
                        idx_valid = pl.read(record_window, [meta_row, 6])
                        idx_slot = pl.read(record_window, [meta_row, 7])
                        if idx_valid > 0 and idx_slot >= 0:
                            logical_block = idx_slot // CMP_STORAGE_BLOCK_SIZE
                            if logical_block < IDX_CACHE_MAX_BLOCKS:
                                physical_block = pl.read(
                                    idx_block_table, [logical_block]
                                )
                                if physical_block >= 0:
                                    destination = (
                                        pl.cast(physical_block, pl.INDEX)
                                        * CMP_STORAGE_BLOCK_SIZE
                                        + idx_slot % CMP_STORAGE_BLOCK_SIZE
                                    )
                                    received_idx_tile = idx_window[meta_row : meta_row + 1, 0:IDX_HEAD_DIM]
                                    idx_flat[destination : destination + 1, 0:IDX_HEAD_DIM] = received_idx_tile
                                    pl.write(
                                        idx_scale_flat,
                                        [destination, 0],
                                        pl.read(scale_window, [meta_row, 0]),
                                    )

                for state_row in pl.range(STATE_ROWS_PER_RANK):
                    meta_row = source_state_row + state_row
                    main_valid = pl.read(
                        main_state_meta_window, [meta_row, 0]
                    )
                    main_position = pl.read(
                        main_state_meta_window, [meta_row, 2]
                    )
                    if main_valid > 0 and main_position >= 0:
                        logical_block = main_position // MAIN_STATE_BLOCK_SIZE
                        if logical_block < MAIN_STATE_MAX_BLOCKS:
                            physical_block = pl.read(
                                compress_state_block_table, [logical_block]
                            )
                            if physical_block >= 0:
                                destination = (
                                    pl.cast(physical_block, pl.INDEX)
                                    * MAIN_STATE_BLOCK_SIZE
                                    + main_position % MAIN_STATE_BLOCK_SIZE
                                )
                                received_main_state = main_state_window[meta_row : meta_row + 1, 0:MAIN_STATE_DIM]
                                main_state_flat[destination : destination + 1, 0:MAIN_STATE_DIM] = received_main_state
                    inner_valid = pl.read(
                        inner_state_meta_window, [meta_row, 0]
                    )
                    inner_position = pl.read(
                        inner_state_meta_window, [meta_row, 2]
                    )
                    if inner_valid > 0 and inner_position >= 0:
                        logical_block = inner_position // INNER_STATE_BLOCK_SIZE
                        if logical_block < INNER_STATE_MAX_BLOCKS:
                            physical_block = pl.read(
                                inner_compress_state_block_table, [logical_block]
                            )
                            if physical_block >= 0:
                                destination = (
                                    pl.cast(physical_block, pl.INDEX)
                                    * INNER_STATE_BLOCK_SIZE
                                    + inner_position % INNER_STATE_BLOCK_SIZE
                                )
                                received_inner_state = inner_state_window[meta_row : meta_row + 1, 0:INNER_STATE_DIM]
                                inner_destination_end = destination + 1
                                inner_state_flat[destination:inner_destination_end, :] = received_inner_state
            _prefill_cp_csa_compact_finish_wave(compact_consumed, my_rank)
        # Store the captured receiver-commit TaskId for this epoch (idiom:
        # prefill_sparse_attn.py:300 proj_a_tids[...] = pa_tid).
        receiver_commit_tids[epoch] = receiver_commit_tid

    cmp_indices = pl.create_tensor([LOCAL_ROWS, IDX_TOPK], dtype=pl.INT32, init_value=-1)
    for tile in pl.range(NUM_LOCAL_TILES):
        active = pl.read(overlay_active_flat, [tile, 1])
        if active > 0:
            row0 = tile * T
            index_x_tile = pl.slice(normed, [T, D], [row0, 0])
            qr_tile = pl.slice(qr, [T, Q_LORA], [row0, 0])
            qr_scale_tile = pl.slice(qr_scale, [T, 1], [row0, 0])
            pos_tile = pl.slice(query_positions_flat, [T], [row0])
            score = pl.create_tensor([T, CP_INDEXER_SCORE_CAP], dtype=pl.FP32)
            topk_tile = pl.slice(cmp_indices, [T, IDX_TOPK], [row0, 0])
            idx_cos = pl.create_tensor([T, QK_ROPE_HALF], dtype=pl.FP32)
            idx_sin = pl.create_tensor([T, QK_ROPE_HALF], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_indexer_rope_rows"):
                idx_cos[0:T, :] = pl.cast(
                    rope_cos_flat[row0 : row0 + T, 0:QK_ROPE_HALF],
                    target_type=pl.FP32,
                )
                idx_sin[0:T, :] = pl.cast(
                    rope_sin_flat[row0 : row0 + T, 0:QK_ROPE_HALF],
                    target_type=pl.FP32,
                )
            score, topk_tile = _prefill_indexer_cp_score_topk(
                index_x_tile, qr_tile, qr_scale_tile,
                idx_wq_b, idx_wq_b_scale, idx_weights_proj,
                idx_cos, idx_sin, hadamard_idx,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                pos_tile, active, score, topk_tile,
            )
            cmp_indices[row0 : row0 + T, 0:IDX_TOPK] = topk_tile

    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32, init_value=-3.0e38)
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, init_value=0)
    _prefill_cp_sparse_stage(
        cache_flat, local_kv, logical_kv,
        cmp_kv,
        cmp_block_table,
        pl.cast(CMP_STORAGE_BLOCK_SIZE, pl.INT32),
        query_positions_flat, query_requests_flat,
        overlay_positions_flat, overlay_requests_flat,
        predecessor_segments_local, segment_starts_local,
        swa_indices_flat, cmp_indices,
        sparse_kv, sparse_bias, valid_mask, overlay_active_flat,
    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_csa_raw_commit") as raw_commit_tid:
        for row in pl.range(T):
            raw_segment = pl.read(final_win_seg_src, [row])
            raw_source_row = pl.read(final_win_row_src, [row])
            raw_destination = pl.read(final_slot_mapping, [row])
            if raw_segment >= 0 and raw_source_row >= 0 and raw_destination >= 0:
                raw_source = raw_segment * T + raw_source_row
                cache_flat[raw_destination : raw_destination + 1, :] = logical_kv[raw_source : raw_source + 1, :]

    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * T
        sparse_row0 = row0 * PREFILL_SPARSE_PAD
        q_tile = pl.slice(q, [T, H, HEAD_DIM], [row0, 0, 0])
        sparse_tile = pl.slice(
            sparse_kv,
            [T * PREFILL_SPARSE_PAD, HEAD_DIM],
            [sparse_row0, 0],
        )
        bias_tile = pl.slice(
            sparse_bias, [T, PREFILL_SPARSE_PAD], [row0, 0]
        )
        mask_tile = pl.slice(
            valid_mask, [T, VALID_BLOCK_MASK_COLS], [row0, 0]
        )
        cos_tile = pl.slice(rope_cos_flat, [T, ROPE_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [T, ROPE_DIM], [row0, 0])
        post_tile = pl.slice(post, [T, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [T, HC_MULT * HC_MULT], [row0, 0])
        residual_tile = pl.slice(x_flat, [T, HC_MULT, D], [row0, 0, 0])
        active = pl.read(overlay_active_flat, [tile, 1])
        attn_out_tile = pl.create_tensor([T, D], dtype=pl.BF16)
        y_tile = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32, init_value=0.0)
        sparse_attn_math(
            q_tile, sparse_tile, bias_tile, mask_tile,
            attn_sink, cos_tile, sin_tile,
            wo_a, wo_b, wo_b_scale,
            attn_out_tile, active,
        )
        hc_post_prefill(
            attn_out_tile, residual_tile,
            post_tile, comb_tile,
            y_tile, active,
        )
        x_out_flat[row0 : row0 + T, 0:HC_MULT, 0:D] = y_tile

    # §8.17.8e.2 leaf-capture completion token. Fan the four leaf-internal
    # commit/transport TaskIds into a single resource_done_tid via
    # pl.system.task_dummy (a no-op task that only waits -- idiom:
    # prefill_compressor_ratio4.py:338 completion[0] = pl.system.task_dummy(...)).
    # Only these four paths are waited; they are NOT serialized against each
    # other, no new dep is added into any existing internal task, and no dep is
    # added into the MoE. compact_transport_tids / receiver_commit_tids are
    # pl.array.create(EPOCHS, pl.TASK_ID); with EPOCHS==1 each contributes one
    # entry via [0]. tail_exchange_tid and raw_commit_tid are scalar TaskIds
    # captured by the `as <tid>:` form (idiom: prefill_sparse_attn.py:289
    # `with pl.at(..., deps=[merge_tid]) as pa_tid`).
    resource_done_tid = pl.system.task_dummy(
        deps=[
            tail_exchange_tid,
            compact_transport_tids[0],
            receiver_commit_tids[0],
            raw_commit_tid,
        ]
    )
    # Terminal rank-complete task: deps=[resource_done_tid] (the four-path
    # fan-in) AND an AUTO dep on the four x_out tile producers established by
    # the slice-copy reads of x_out_flat below. Writes one tile slice per
    # distinct token row --禁止反复覆盖同一个 [0:1, 0:1, 0:8]，否则前面三次写可能
    # 被 dead-store elimination 消除 (user design requirement). Each row is
    # 8 × FP32 = 32B, satisfying the pto.alloc_tile 32-byte column alignment.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_csa_rank_complete",
        deps=[resource_done_tid],
    ):
        for tile in pl.range(NUM_MOE_WAVES):
            completion_token[tile : tile + 1, 0:1, 0:8] = pl.slice(
                x_out_flat, [1, 1, 8], [tile * T, 0, 0]
            )

    return pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D])

@pl.jit
def prefill_cp_csa_rank(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    main_state_workspace0: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
        ]
    ],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32
    ],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[T], pl.INT32],
    final_win_row_src: pl.Tensor[[T], pl.INT32],
    final_slot_mapping: pl.Tensor[[T], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32
    ],
    effective_x_workspace: pl.Tensor[[LOCAL_LEAVES * T, D], pl.BF16],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, HEAD_DIM], pl.BF16
    ],
    idx_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8
    ],
    scale_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP32
    ],
    record_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, META_DIM], pl.INT32
    ],
    main_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run standalone CP-CSA attention for one rank."""
    # §8.17.8e.2 leaf-capture completion token: child-local, allocated here and
    # passed into prefill_cp_csa_core (which publishes it via its terminal
    # cp_csa_rank_complete task). The standalone test does not consume the
    # token -- it only certifies, within this rank's graph, that the
    # leaf-internal commit/transport tasks retired before x_out is published.
    # Keeping it child-local avoids changing the standalone test's host
    # interface / build_tensor_specs (the token is not a host-visible output).
    completion_token = pl.create_tensor(
        [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0
    )
    return prefill_cp_csa_core(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape,
        cmp_norm_w, hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj, inner_wkv, inner_wgate, inner_ape,
        inner_norm_w, main_state_workspace0, inner_state_workspace0, main_state_workspace1, inner_state_workspace1, compress_state, compress_state_block_table, inner_compress_state,
        inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, segment_starts_t,
        segment_lengths_t, segment_active_lengths, owner_segments_t, predecessor_segments, query_positions, query_requests, overlay_positions, overlay_requests,
        overlay_active_lengths, swa_indices, final_segment_t, reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping,
        leaf_positions_input, leaf_main_slots_input, leaf_idx_slots_input, leaf_main_state_slots_input, leaf_inner_state_slots_input, leaf_num_tokens_input, effective_x_workspace, hidden_tail_window,
        kv_tail_window, tail_ready, tail_consumed, main_window, idx_window, scale_window, record_window, main_state_window,
        main_state_meta_window, inner_state_window, inner_state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b,
        wo_b_scale, x_out, completion_token, my_rank,
        pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
    )

@pl.jit.host
def prefill_cp_csa_test(
    x_hc: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[M.max_position_embeddings, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [CP_SIZE, MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [CP_SIZE, INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                PREFILL_CMP_BLOCK_NUM,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[
        [CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                PREFILL_IDX_BLOCK_NUM,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                IDX_HEAD_DIM,
            ],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CP_SIZE, PREFILL_IDX_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, 1],
            pl.FP32,
        ]
    ],
    idx_block_table: pl.Tensor[[CP_SIZE, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    query_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32
    ],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[T], pl.INT32],
    final_win_row_src: pl.Tensor[[T], pl.INT32],
    final_slot_mapping: pl.Tensor[[T], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32
    ],
    effective_x_workspace: pl.Tensor[
        [CP_SIZE, LOCAL_LEAVES * T, D], pl.BF16
    ],
    main_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    x_out: pl.Out[
        pl.Tensor[
            [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
):
    hidden_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
    )
    kv_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    main_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16
    )
    idx_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
    )
    scale_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32
    )
    record_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
    )
    main_state_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], dtype=pl.FP32
    )
    main_state_meta_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    inner_state_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], dtype=pl.FP32
    )
    inner_state_meta_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(
            hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        kv_tail_window = pld.window(
            kv_tail_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        tail_ready = pld.window(tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        main_window = pld.window(
            main_buf, [RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16
        )
        idx_window = pld.window(
            idx_buf, [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
        )
        scale_window = pld.window(
            scale_buf, [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32
        )
        record_window = pld.window(
            record_buf, [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
        )
        main_state_window = pld.window(
            main_state_buf,
            [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM],
            dtype=pl.FP32,
        )
        main_state_meta_window = pld.window(
            main_state_meta_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM],
            dtype=pl.INT32,
        )
        inner_state_window = pld.window(
            inner_state_buf,
            [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM],
            dtype=pl.FP32,
        )
        inner_state_meta_window = pld.window(
            inner_state_meta_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM],
            dtype=pl.INT32,
        )
        compact_ready = pld.window(
            compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        compact_consumed = pld.window(
            compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        prefill_cp_csa_rank(
            x_hc[rank],
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            freqs_cos,
            freqs_sin,
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            hadamard_idx,
            idx_wq_b,
            idx_wq_b_scale,
            idx_weights_proj,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            main_state_workspace0[rank],
            inner_state_workspace0[rank],
            main_state_workspace1[rank],
            inner_state_workspace1[rank],
            compress_state[rank],
            compress_state_block_table[rank],
            inner_compress_state[rank],
            inner_compress_state_block_table[rank],
            kv_cache[rank],
            cmp_kv[rank],
            cmp_block_table[rank],
            idx_kv_cache[rank],
            idx_kv_scale[rank],
            idx_block_table[rank],
            segment_starts_t,
            segment_lengths_t,
            segment_active_lengths[rank],
            owner_segments_t[rank],
            predecessor_segments[rank],
            query_positions[rank],
            query_requests[rank],
            overlay_positions[rank],
            overlay_requests[rank],
            overlay_active_lengths[rank],
            swa_indices[rank],
            final_segment_t,
            reverse_index,
            owner_rank_table,
            final_win_seg_src,
            final_win_row_src,
            final_slot_mapping,
            leaf_positions_input[rank],
            leaf_main_slots_input[rank],
            leaf_idx_slots_input[rank],
            leaf_main_state_slots_input[rank],
            leaf_inner_state_slots_input[rank],
            leaf_num_tokens_input[rank],
            effective_x_workspace[rank],
            hidden_tail_window,
            kv_tail_window,
            tail_ready,
            tail_consumed,
            main_window,
            idx_window,
            scale_window,
            record_window,
            main_state_window,
            main_state_meta_window,
            inner_state_window,
            inner_state_meta_window,
            compact_ready,
            compact_consumed,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_out[rank],
            rank,
            device=rank,
        )


def golden_prefill_cp_csa(tensors):
    """Compose CP-CSA golden outputs in logical-segment commit order."""
    from utils import int8_quant_per_row

    ctx = getattr(golden_prefill_cp_csa, "_ctx", None)
    if ctx is None:
        raise RuntimeError("CP-CSA golden context was not installed")
    cp_size = int(ctx["cp_size"])
    starts = [int(value) for value in ctx["starts"]]
    lengths = [int(value) for value in ctx["lengths"]]
    owners = ctx["owners"]
    reverse = tensors["reverse_index"].tolist()

    local_norm = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        D,
        dtype=torch.bfloat16,
    )
    local_q = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        H,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_kv = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_qr = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        Q_LORA,
        dtype=torch.int8,
    )
    local_qr_scale = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        1,
        dtype=torch.float32,
    )
    local_post = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT
    )
    local_comb = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        T,
        HC_MULT * HC_MULT,
    )
    logical_norm: dict[int, torch.Tensor] = {}
    logical_kv: dict[int, torch.Tensor] = {}

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = int(owners[rank][part])
            active_segment = lengths[segment]
            segment_norm_rows = []
            segment_kv_rows = []
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(active_segment, tile)
                x_tile = tensors["x_hc"][rank, part, tile]
                mixed = torch.zeros(T, D, dtype=torch.bfloat16)
                post = torch.zeros(T, HC_MULT)
                comb = torch.zeros(T, HC_MULT * HC_MULT)
                golden_hc_pre(
                    {
                        "x": x_tile,
                        "hc_fn": tensors["hc_attn_fn"],
                        "hc_scale": tensors["hc_attn_scale"],
                        "hc_base": tensors["hc_attn_base"],
                        "x_mixed": mixed,
                        "post": post,
                        "comb": comb,
                    }
                )
                normed = golden_rms_norm(mixed, tensors["attn_norm_w"])
                positions = tensors["query_positions"][rank, part, tile]
                rope_rows = positions.clamp_min(0).long()
                q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
                qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
                qr_scale = torch.zeros(T, 1)
                golden_qkv_proj_rope(
                    {
                        "x": normed,
                        "wq_a": tensors["wq_a"],
                        "wq_b": tensors["wq_b"],
                        "wq_b_scale": tensors["wq_b_scale"],
                        "wkv": tensors["wkv"],
                        "rope_cos": tensors["freqs_cos"].index_select(
                            0, rope_rows
                        ),
                        "rope_sin": tensors["freqs_sin"].index_select(
                            0, rope_rows
                        ),
                        "gamma_cq": tensors["gamma_cq"],
                        "gamma_ckv": tensors["gamma_ckv"],
                        "q": q,
                        "kv": kv,
                        "qr": qr,
                        "qr_scale": qr_scale,
                    }
                )
                local_norm[rank, part, tile] = normed
                local_q[rank, part, tile] = q
                local_kv[rank, part, tile] = kv
                local_qr[rank, part, tile] = qr
                local_qr_scale[rank, part, tile] = qr_scale
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb
                if active:
                    segment_norm_rows.append(normed[:active])
                    segment_kv_rows.append(kv[:active])
            logical_norm[segment] = (
                torch.cat(segment_norm_rows)
                if segment_norm_rows
                else torch.zeros(0, D, dtype=torch.bfloat16)
            )
            logical_kv[segment] = (
                torch.cat(segment_kv_rows)
                if segment_kv_rows
                else torch.zeros(0, HEAD_DIM, dtype=torch.bfloat16)
            )

    cmp_published: dict[int, torch.Tensor] = {}
    idx_published: dict[int, torch.Tensor] = {}
    scale_published: dict[int, torch.Tensor] = {}
    final_main_state = None
    final_inner_state = None
    final_owner_rank = -1

    for segment in range(NUM_SEGMENTS):
        source_slot = int(reverse[segment])
        rank = source_slot // LOCAL_PARTS
        part = source_slot % LOCAL_PARTS
        main_state = torch.zeros_like(tensors["compress_state"][rank])
        inner_state = torch.zeros_like(tensors["inner_compress_state"][rank])
        if segment == 0:
            main_state.copy_(tensors["compress_state"][rank])
            inner_state.copy_(tensors["inner_compress_state"][rank])

        leaves: list[tuple[torch.Tensor, torch.Tensor]] = []
        if segment > 0 and lengths[segment] > 0:
            predecessor = segment - 1
            predecessor_rows = logical_norm[predecessor]
            seed_len = min(
                starts[segment] % COMPRESS_RATIO + COMPRESS_RATIO,
                min(T, lengths[predecessor]),
            )
            seed = torch.zeros(T, D, dtype=torch.bfloat16)
            positions = torch.zeros(T, dtype=torch.int32)
            if seed_len:
                seed[:seed_len] = predecessor_rows[-seed_len:]
                positions[:seed_len] = torch.arange(
                    starts[segment] - seed_len,
                    starts[segment],
                    dtype=torch.int32,
                )
            leaves.append((seed, positions))
        else:
            leaves.append(
                (
                    torch.zeros(T, D, dtype=torch.bfloat16),
                    torch.zeros(T, dtype=torch.int32),
                )
            )
        for tile in range(MAX_SEGMENT_TILES):
            leaves.append(
                (
                    local_norm[rank, part, tile],
                    tensors["query_positions"][rank, part, tile],
                )
            )

        for leaf, (x_leaf, positions) in enumerate(leaves):
            if leaf == 0:
                active = (
                    min(
                        starts[segment] % COMPRESS_RATIO + COMPRESS_RATIO,
                        min(T, lengths[segment - 1]),
                    )
                    if segment > 0 and lengths[segment] > 0
                    else 0
                )
            else:
                active = _active_tile(lengths[segment], leaf - 1)
            if active <= 0:
                continue
            main_cache = torch.zeros(
                MAIN_LEAF_CACHE_BLOCKS,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                HEAD_DIM,
                dtype=torch.bfloat16,
            )
            idx_cache = torch.zeros(
                IDX_LEAF_CACHE_BLOCKS,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                IDX_HEAD_DIM,
                dtype=torch.int8,
            )
            idx_scale = torch.zeros(
                IDX_LEAF_CACHE_BLOCKS, CMP_STORAGE_BLOCK_SIZE, 1, 1
            )
            main_map = torch.full((T,), -1, dtype=torch.int64)
            idx_map = torch.full((T,), -1, dtype=torch.int64)
            main_state_map = torch.full((T,), -1, dtype=torch.int64)
            inner_state_map = torch.full((T,), -1, dtype=torch.int64)
            write_count = 0
            for row in range(active):
                position = int(positions[row])
                main_state_map[row] = _lower_row(
                    tensors["compress_state_block_table"][rank],
                    position,
                    MAIN_STATE_BLOCK_SIZE,
                )
                inner_state_map[row] = _lower_row(
                    tensors["inner_compress_state_block_table"][rank],
                    position,
                    INNER_STATE_BLOCK_SIZE,
                )
                if leaf > 0 and (position + 1) % COMPRESS_RATIO == 0:
                    main_map[row] = write_count
                    idx_map[row] = write_count
                    write_count += 1
            golden_prefill_compressor_ratio4(
                {
                    "x": x_leaf,
                    "compress_state": main_state,
                    "compress_state_block_table": tensors[
                        "compress_state_block_table"
                    ][rank],
                    "wkv": tensors["cmp_wkv"],
                    "wgate": tensors["cmp_wgate"],
                    "ape": tensors["cmp_ape"],
                    "norm_w": tensors["cmp_norm_w"],
                    "freqs_cos": tensors["freqs_cos"],
                    "freqs_sin": tensors["freqs_sin"],
                    "cmp_kv": main_cache,
                    "position_ids": positions,
                    "num_tokens": active,
                    "cmp_slot_mapping": main_map,
                    "state_slot_mapping": main_state_map,
                }
            )
            golden_prefill_indexer_compressor(
                {
                    "x": x_leaf,
                    "compress_state": inner_state,
                    "inner_compress_state_block_table": tensors[
                        "inner_compress_state_block_table"
                    ][rank],
                    "wkv": tensors["inner_wkv"],
                    "wgate": tensors["inner_wgate"],
                    "ape": tensors["inner_ape"],
                    "norm_w": tensors["inner_norm_w"],
                    "freqs_cos": tensors["freqs_cos"],
                    "freqs_sin": tensors["freqs_sin"],
                    "hadamard": tensors["hadamard_idx"],
                    "kv": torch.zeros(
                        T // COMPRESS_RATIO,
                        IDX_HEAD_DIM,
                        dtype=torch.int8,
                    ),
                    "idx_kv_cache": idx_cache,
                    "idx_kv_scale": idx_scale,
                    "idx_block_table": tensors["idx_block_table"][rank],
                    "position_ids": positions,
                    "num_tokens": active,
                    "idx_slot_mapping": idx_map,
                    "inner_state_slot_mapping": inner_state_map,
                }
            )
            for row in range(active):
                local_row = int(main_map[row])
                if local_row >= 0:
                    logical_slot = (int(positions[row]) + 1) // COMPRESS_RATIO - 1
                    cmp_published[logical_slot] = main_cache.view(-1, HEAD_DIM)[
                        local_row
                    ].clone()
                    idx_published[logical_slot] = idx_cache.view(
                        -1, IDX_HEAD_DIM
                    )[local_row].clone()
                    scale_published[logical_slot] = idx_scale.view(-1, 1)[
                        local_row
                    ].clone()

        if segment == int(ctx["final_segment"]):
            final_main_state = main_state
            final_inner_state = inner_state
            final_owner_rank = rank

    cmp_out = tensors["cmp_kv"].clone()
    idx_out = tensors["idx_kv_cache"].clone()
    scale_out = tensors["idx_kv_scale"].clone()
    for rank in range(cp_size):
        cmp_flat = cmp_out[rank].view(-1, HEAD_DIM)
        idx_flat = idx_out[rank].view(-1, IDX_HEAD_DIM)
        scale_flat = scale_out[rank].view(-1, 1)
        for logical_slot, value in cmp_published.items():
            row = _lower_row(
                tensors["cmp_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            if row >= 0:
                cmp_flat[row] = value
        for logical_slot, value in idx_published.items():
            row = _lower_row(
                tensors["idx_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            if row >= 0:
                idx_flat[row] = value
                scale_flat[row] = scale_published[logical_slot]

    main_state_out = tensors["compress_state"].clone()
    inner_state_out = tensors["inner_compress_state"].clone()
    final_end = starts[int(ctx["final_segment"])] + lengths[
        int(ctx["final_segment"])
    ]
    final_positions = range(max(0, final_end - STATE_LEN), final_end)
    for receiver in range(cp_size):
        for position in final_positions:
            source = _lower_row(
                tensors["compress_state_block_table"][final_owner_rank],
                position,
                MAIN_STATE_BLOCK_SIZE,
            )
            destination = _lower_row(
                tensors["compress_state_block_table"][receiver],
                position,
                MAIN_STATE_BLOCK_SIZE,
            )
            if source >= 0 and destination >= 0:
                main_state_out[receiver].view(-1, MAIN_STATE_DIM)[destination] = (
                    final_main_state.view(-1, MAIN_STATE_DIM)[source]
                )
            source = _lower_row(
                tensors["inner_compress_state_block_table"][final_owner_rank],
                position,
                INNER_STATE_BLOCK_SIZE,
            )
            destination = _lower_row(
                tensors["inner_compress_state_block_table"][receiver],
                position,
                INNER_STATE_BLOCK_SIZE,
            )
            if source >= 0 and destination >= 0:
                inner_state_out[receiver].view(-1, INNER_STATE_DIM)[
                    destination
                ] = final_inner_state.view(-1, INNER_STATE_DIM)[source]

    topk_by_rank = []
    for rank in range(cp_size):
        norm_flat = local_norm[rank].reshape(LOCAL_ROWS, D)
        qr_flat = local_qr[rank].reshape(LOCAL_ROWS, Q_LORA)
        qr_scale_flat = local_qr_scale[rank].reshape(LOCAL_ROWS, 1)
        positions = tensors["query_positions"][rank].reshape(LOCAL_ROWS)
        q_i32 = qr_flat.to(torch.int32) @ tensors["idx_wq_b"].to(torch.int32)
        query = (
            q_i32.float()
            * qr_scale_flat
            * tensors["idx_wq_b_scale"].float().view(1, -1)
        ).view(LOCAL_ROWS, IDX_N_HEADS, IDX_HEAD_DIM)
        rope_pair = query[..., -ROPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        rope_rows = positions.clamp_min(0).long()
        cos = tensors["freqs_cos"].index_select(0, rope_rows)[
            :, :QK_ROPE_HALF
        ].float().unsqueeze(1)
        sin = tensors["freqs_sin"].index_select(0, rope_rows)[
            :, :QK_ROPE_HALF
        ].float().unsqueeze(1)
        query = torch.cat(
            [
                query[..., :-ROPE_HEAD_DIM],
                torch.stack(
                    [
                        rope_pair[..., 0] * cos - rope_pair[..., 1] * sin,
                        rope_pair[..., 0] * sin + rope_pair[..., 1] * cos,
                    ],
                    dim=-1,
                )
                .flatten(-2)
                .to(torch.bfloat16)
                .float(),
            ],
            dim=-1,
        )
        query = query @ tensors["hadamard_idx"].float()
        weights = (
            norm_flat.float() @ tensors["idx_weights_proj"].float()
        ) * M.index_weights_scale
        query_i8, query_scale = int8_quant_per_row(
            query.reshape(LOCAL_ROWS * IDX_N_HEADS, IDX_HEAD_DIM)
        )
        max_visible = min(
            CP_CANDIDATE_CAPACITY,
            max(0, (int(positions.max()) + 1) // COMPRESS_RATIO),
        )
        cache_rows = []
        cache_scales = []
        idx_flat = idx_out[rank].view(-1, IDX_HEAD_DIM)
        scale_flat = scale_out[rank].view(-1)
        for logical_slot in range(max_visible):
            row = _lower_row(
                tensors["idx_block_table"][rank],
                logical_slot,
                CMP_STORAGE_BLOCK_SIZE,
            )
            cache_rows.append(
                idx_flat[row]
                if row >= 0
                else torch.zeros(IDX_HEAD_DIM, dtype=torch.int8)
            )
            cache_scales.append(float(scale_flat[row]) if row >= 0 else 0.0)
        topk = torch.full((LOCAL_ROWS, IDX_TOPK), -1, dtype=torch.int32)
        if max_visible:
            cache_i8 = torch.stack(cache_rows).to(torch.int32)
            cache_scale = torch.tensor(cache_scales).view(1, 1, -1)
            query_i32 = query_i8.view(
                LOCAL_ROWS, IDX_N_HEADS, IDX_HEAD_DIM
            ).to(torch.int32)
            query_scale = query_scale.view(LOCAL_ROWS, IDX_N_HEADS, 1)
            score = (
                torch.einsum("thd,cd->thc", query_i32, cache_i8).float()
                * query_scale
                * cache_scale
            )
            score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=1)
            for row in range(LOCAL_ROWS):
                visible = min(
                    CP_CANDIDATE_CAPACITY,
                    max(0, (int(positions[row]) + 1) // COMPRESS_RATIO),
                )
                selected = min(SPARSE_SELECTED_WIDTH, visible)
                if selected:
                    topk[row, :selected] = score[row, :visible].topk(
                        selected
                    ).indices.to(torch.int32)
        topk_by_rank.append(topk)

    kv_before_commit = tensors["kv_cache"].clone()
    x_out = torch.zeros_like(tensors["x_out"])
    for rank in range(cp_size):
        topk = topk_by_rank[rank].view(
            LOCAL_PARTS, MAX_SEGMENT_TILES, T, IDX_TOPK
        )
        for part in range(LOCAL_PARTS):
            segment = int(owners[rank][part])
            for tile in range(MAX_SEGMENT_TILES):
                active = _active_tile(lengths[segment], tile)
                overlay = torch.zeros(
                    OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16
                )
                predecessor_len = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 0]
                )
                current_len = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 1]
                )
                if predecessor_len:
                    if tile == 0:
                        predecessor = segment - 1
                        overlay[:predecessor_len] = logical_kv[predecessor][
                            -predecessor_len:
                        ]
                    else:
                        overlay[:predecessor_len] = local_kv[
                            rank, part, tile - 1, :predecessor_len
                        ]
                overlay[T : T + current_len] = local_kv[
                    rank, part, tile, :current_len
                ]
                sparse_source = torch.zeros(
                    ORI_CACHE_ROWS + OVERLAY_ROWS,
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                sparse_source[:ORI_CACHE_ROWS] = kv_before_commit[rank].view(
                    -1, HEAD_DIM
                )
                sparse_source[ORI_CACHE_ROWS:] = overlay
                sparse_source_cache = sparse_source.view(
                    (ORI_CACHE_ROWS + OVERLAY_ROWS) // BLOCK_SIZE,
                    BLOCK_SIZE,
                    1,
                    HEAD_DIM,
                )
                attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
                positions = tensors["query_positions"][rank, part, tile]
                golden_prefill_sparse_attn(
                    {
                        "q": local_q[rank, part, tile],
                        "ori_kv": sparse_source_cache,
                        "swa_indices": tensors["swa_indices"][rank, part, tile],
                        "cmp_kv": cmp_out[rank],
                        "cmp_block_table": tensors["cmp_block_table"][rank],
                        "cmp_storage_block_size": CMP_STORAGE_BLOCK_SIZE,
                        "cmp_indices": topk[part, tile],
                        "attn_sink": tensors["attn_sink"],
                        "num_tokens": active,
                        "freqs_cos": tensors["freqs_cos"].index_select(
                            0, positions.clamp_min(0).long()
                        ),
                        "freqs_sin": tensors["freqs_sin"].index_select(
                            0, positions.clamp_min(0).long()
                        ),
                        "wo_a": tensors["wo_a"],
                        "wo_b": tensors["wo_b"],
                        "wo_b_scale": tensors["wo_b_scale"],
                        "overlay_kv": overlay,
                        "overlay_position_ids": tensors["overlay_positions"][
                            rank, part, tile
                        ],
                        "overlay_token_to_request": tensors[
                            "overlay_requests"
                        ][rank, part, tile],
                        "overlay_active_lengths": tensors[
                            "overlay_active_lengths"
                        ][rank, part, tile],
                        "query_position_ids": positions,
                        "query_token_to_request": tensors["query_requests"][
                            rank, part, tile
                        ],
                        "attn_out": attn_out,
                    }
                )
                y = torch.zeros(T, HC_MULT, D)
                golden_hc_post_prefill(
                    {
                        "x": attn_out,
                        "residual": tensors["x_hc"][rank, part, tile],
                        "post": local_post[rank, part, tile],
                        "comb": local_comb[rank, part, tile],
                        "y": y,
                        "num_tokens": active,
                    }
                )
                x_out[rank, part, tile] = y

    kv_out = kv_before_commit.clone()
    for row in range(T):
        segment = int(tensors["final_win_seg_src"][row])
        source_row = int(tensors["final_win_row_src"][row])
        destination = int(tensors["final_slot_mapping"][row])
        if segment >= 0 and source_row >= 0 and destination >= 0:
            value = logical_kv[segment][-min(T, lengths[segment]) :][source_row]
            for rank in range(cp_size):
                kv_out[rank].view(-1, HEAD_DIM)[destination] = value

    tensors["compress_state"][:] = main_state_out
    tensors["inner_compress_state"][:] = inner_state_out
    tensors["cmp_kv"][:] = cmp_out
    tensors["idx_kv_cache"][:] = idx_out
    tensors["idx_kv_scale"][:] = scale_out
    tensors["kv_cache"][:] = kv_out
    tensors["x_out"][:] = x_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 context-parallel CSA test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=CP_CHOICES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", action="store_true")
    args = parser.parse_args()
    from golden import ratio_allclose, ratio_reldiff, run_jit

    if args.cp != CP_SIZE:
        raise SystemExit(f"--cp={args.cp} does not match import-time CP_SIZE={CP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")
    result = run_jit(
        fn=prefill_cp_csa_test,
        specs=build_tensor_specs(args.cp),
        golden_fn=golden_prefill_cp_csa,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[: args.cp], num_sub_workers=0
            ),
            dump_passes=args.dump_passes,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "idx_kv_cache": ratio_allclose(
                atol=1, rtol=0, max_error_ratio=0.01
            ),
            "idx_kv_scale": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
