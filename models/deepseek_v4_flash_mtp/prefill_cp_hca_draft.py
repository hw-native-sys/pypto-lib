# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek V4 context-parallel HCA prefill."""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    FLASH as M,
    FP32_NEG_INF,
    HCA_STATE_PHYSICAL_BLOCKS,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
)
from prefill_compressor_ratio128 import (
    CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_RATIO,
    COMPRESS_STATE_DIM,
    HCA_CMP_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
    build_tensor_specs as build_compressor_tensor_specs,
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from prefill_cp_exchange import (
    CMP_META_DIM,
    CMP_ROWS_PER_RANK,
    CMP_ROWS_PER_SEGMENT,
    CMP_WINDOW_ROWS,
    STATE_META_DIM,
    STATE_WINDOW_ROWS,
    _prefill_cp_dual_tail_exchange_wave,
    _prefill_cp_hca_compact_exchange_commit_wave,
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
    cp_owner_part,
    cp_owner_rank,
    cp_owner_tables,
    cp_reverse_index,
)
from hc_post import golden_hc_post_prefill, hc_post_prefill
from hc_pre import golden_hc_pre, hc_pre
from prefill_sparse_attn import (
    HEAD_DIM,
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
IDX_TOPK = M.index_topk
WIN = M.sliding_window

# CP layout
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_COMPRESSED_ROWS_PER_SEGMENT = MAX_SEGMENT_TILES
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
LEAF_CMP_BLOCKS = HCA_CMP_BLOCK_NUM
LEAF_CMP_ROWS = (
    LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS * CMP_STORAGE_BLOCK_SIZE
)
STATE_ROWS = HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE

def active_tile(segment_len: int, tile: int) -> int:
    return max(0, min(TAIL_ROWS, segment_len - tile * TAIL_ROWS))


def segment_starts(prefix: int, span: int, nseg: int):
    return [prefix + segment * span for segment in range(nseg)]


def owner_segments(cp_size: int):
    owners = [[-1, -1] for _ in range(cp_size)]
    for segment in range(2 * cp_size):
        owners[cp_owner_rank(segment, cp_size)][cp_owner_part(segment, cp_size)] = segment
    return owners


def _tail_start(segment_start: int, segment_len: int) -> int:
    return segment_start + max(0, segment_len - TAIL_ROWS)


def _ring_phys_row(position: int) -> int:
    return position % ORI_CACHE_ROWS


def _lower_raw_key(
    key_abs: int,
    segment: int,
    tile: int,
    starts: list[int],
    lengths: list[int],
    prefix: int,
) -> int:
    segment_start = starts[segment]
    tile_start = segment_start + tile * TAIL_ROWS
    tile_len = active_tile(lengths[segment], tile)
    if key_abs < prefix:
        return _ring_phys_row(key_abs)
    if tile_start <= key_abs < tile_start + tile_len:
        return OVERLAY_BASE + TAIL_ROWS + key_abs - tile_start
    if key_abs >= tile_start:
        return -1
    if tile == 0:
        predecessor = segment - 1
        if predecessor < 0:
            return -1
        predecessor_start = _tail_start(starts[predecessor], lengths[predecessor])
        predecessor_len = min(TAIL_ROWS, lengths[predecessor])
    else:
        predecessor_start = tile_start - TAIL_ROWS
        predecessor_len = active_tile(lengths[segment], tile - 1)
    if predecessor_start <= key_abs < predecessor_start + predecessor_len:
        return OVERLAY_BASE + key_abs - predecessor_start
    return -1


def _build_raw_attention_metadata(cp_size: int):
    import torch

    prefix = 0
    span = TAIL_ROWS
    lengths = [TAIL_ROWS] * (2 * cp_size)
    starts = segment_starts(prefix, span, 2 * cp_size)
    owners = owner_segments(cp_size)
    query_positions = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, dtype=torch.int32
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
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN),
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
            predecessors[rank, part] = segment - 1
            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(segment_len, tile)
                tile_start = starts[segment] + tile * TAIL_ROWS
                if active:
                    query_positions[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    query_requests[rank, part, tile, :active] = 0
                if tile == 0:
                    predecessor = segment - 1
                    predecessor_len = (
                        min(TAIL_ROWS, lengths[predecessor])
                        if predecessor >= 0
                        else 0
                    )
                    predecessor_start = (
                        _tail_start(starts[predecessor], lengths[predecessor])
                        if predecessor >= 0
                        else 0
                    )
                else:
                    predecessor_len = active_tile(segment_len, tile - 1)
                    predecessor_start = tile_start - TAIL_ROWS
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
                        rank, part, tile, TAIL_ROWS:TAIL_ROWS + active
                    ] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    overlay_requests[
                        rank, part, tile, TAIL_ROWS:TAIL_ROWS + active
                    ] = 0
                overlay_lengths[rank, part, tile, 0] = predecessor_len
                overlay_lengths[rank, part, tile, 1] = active
                for query_row in range(active):
                    query_abs = tile_start + query_row
                    for sparse_col in range(WIN):
                        key_abs = query_abs - WIN + 1 + sparse_col
                        if 0 <= key_abs <= query_abs:
                            swa_indices[
                                rank, part, tile, query_row, sparse_col
                            ] = _lower_raw_key(
                                key_abs,
                                segment,
                                tile,
                                starts,
                                lengths,
                                prefix,
                            )
    final_seg_src, final_row_src = cp_final_window_sources(lengths)
    final_slot_mapping = torch.tensor(
        [
            _ring_phys_row(prefix + sum(lengths) - TAIL_ROWS + row)
            for row in range(TAIL_ROWS)
        ],
        dtype=torch.int32,
    )
    return {
        "segment_active_lengths": segment_active,
        "predecessor_segments": predecessors,
        "query_position_ids": query_positions,
        "query_token_to_request": query_requests,
        "overlay_position_ids": overlay_positions,
        "overlay_token_to_request": overlay_requests,
        "overlay_active_lengths": overlay_lengths,
        "swa_indices": swa_indices,
        "reverse_index": cp_reverse_index(cp_size).to(torch.int32),
        "final_win_seg_src": final_seg_src.to(torch.int32),
        "final_win_row_src": final_row_src.to(torch.int32),
        "final_slot_mapping": final_slot_mapping,
    }


def _cmp_slot(boundary_position: int) -> int:
    if (boundary_position + 1) % COMPRESS_RATIO:
        return -1
    return (boundary_position + 1) // COMPRESS_RATIO - 1


def _cmp_block_tables(cp_size: int):
    import torch

    tables = torch.full(
        (cp_size, PREFILL_CMP_MAX_BLOCKS), -1, dtype=torch.int32
    )
    for rank in range(cp_size):
        for logical_block in range(PREFILL_CMP_MAX_BLOCKS):
            physical = logical_block % PREFILL_CMP_BLOCK_NUM
            tables[rank, logical_block] = physical
    return tables


def _state_block_tables(cp_size: int):
    import torch

    tables = torch.empty(
        cp_size, HCA_STATE_MAX_BLOCKS, dtype=torch.int32
    )
    for rank in range(cp_size):
        for logical_block in range(HCA_STATE_MAX_BLOCKS):
            tables[rank, logical_block] = (
                logical_block * 17 + 3
            ) % HCA_STATE_PHYSICAL_BLOCKS
    return tables


def build_hca_metadata(cp_size: int = CP_SIZE):
    """Build canonical zero-history CP-HCA metadata."""
    import torch

    if cp_size not in CP_CHOICES:
        raise ValueError(f"cp_size must be one of {CP_CHOICES}, got {cp_size}")

    prefix = 0
    span = TAIL_ROWS
    lengths = [TAIL_ROWS] * (2 * cp_size)
    starts = segment_starts(prefix, span, 2 * cp_size)
    owners = owner_segments(cp_size)

    query_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS),
        -1,
        dtype=torch.int32,
    )
    query_active = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, dtype=torch.int32
    )
    cmp_indices = torch.full(
        (
            cp_size,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            IDX_TOPK,
        ),
        -1,
        dtype=torch.int32,
    )
    segment_cmp_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_COMPRESSED_ROWS_PER_SEGMENT),
        -1,
        dtype=torch.int32,
    )
    segment_cmp_slots = torch.full_like(segment_cmp_positions, -1)
    snapshot_positions = torch.full(
        (cp_size, LOCAL_PARTS, TAIL_ROWS), -1, dtype=torch.int32
    )
    snapshot_valid = torch.zeros(
        cp_size, LOCAL_PARTS, dtype=torch.int32
    )
    segment_tail_positions = torch.full(
        (2 * cp_size, TAIL_ROWS), -1, dtype=torch.int32
    )

    for segment in range(2 * cp_size):
        segment_end = starts[segment] + lengths[segment]
        valid = min(TAIL_ROWS, lengths[segment])
        if valid:
            tail_begin = segment_end - valid
            segment_tail_positions[segment, :valid] = torch.arange(
                tail_begin, segment_end, dtype=torch.int32
            )

    for rank, rank_segments in enumerate(owners):
        for part, segment in enumerate(rank_segments):
            segment_start = starts[segment]
            segment_len = lengths[segment]
            segment_end = segment_start + segment_len

            boundaries = [
                position
                for position in range(segment_start, segment_end)
                if _cmp_slot(position) >= 0
            ]
            if len(boundaries) > MAX_COMPRESSED_ROWS_PER_SEGMENT:
                raise ValueError(
                    f"segment {segment} has {len(boundaries)} compressed rows; "
                    f"capacity is {MAX_COMPRESSED_ROWS_PER_SEGMENT}"
                )
            for index, boundary in enumerate(boundaries):
                segment_cmp_positions[rank, part, index] = boundary
                segment_cmp_slots[rank, part, index] = _cmp_slot(boundary)

            live_valid = min(TAIL_ROWS, segment_end)
            live_start = segment_end - live_valid
            snapshot_valid[rank, part] = live_valid
            if live_valid:
                snapshot_positions[rank, part, :live_valid] = torch.arange(
                    live_start, segment_end, dtype=torch.int32
                )

            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(segment_len, tile)
                query_active[rank, part, tile] = active
                tile_start = segment_start + tile * TAIL_ROWS
                if active:
                    query_positions[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                for row in range(active):
                    absolute_position = tile_start + row
                    visible = min(
                        IDX_TOPK,
                        (absolute_position + 1) // COMPRESS_RATIO,
                    )
                    if visible:
                        cmp_indices[rank, part, tile, row, :visible] = (
                            torch.arange(visible, dtype=torch.int32)
                        )

    active_segments = [
        segment for segment, length in enumerate(lengths) if length > 0
    ]
    if not active_segments:
        raise ValueError("CP-HCA requires at least one active logical segment")
    final_segment = active_segments[-1]
    final_owner_rank = next(
        rank
        for rank, rank_segments in enumerate(owners)
        if final_segment in rank_segments
    )
    final_owner_part = owners[final_owner_rank].index(final_segment)
    owner_rank_table, owner_part_table = cp_owner_tables(cp_size)

    metadata = {
        "cp_size": cp_size,
        "prefix": prefix,
        "segment_span": span,
        "segment_lengths": torch.tensor(lengths, dtype=torch.int32),
        "segment_starts": torch.tensor(starts, dtype=torch.int32),
        "owner_segments": torch.tensor(owners, dtype=torch.int32),
        "query_positions": query_positions,
        "query_active": query_active,
        "cmp_indices": cmp_indices,
        "segment_cmp_positions": segment_cmp_positions,
        "segment_cmp_slots": segment_cmp_slots,
        "snapshot_positions": snapshot_positions,
        "snapshot_valid": snapshot_valid,
        "segment_tail_positions": segment_tail_positions,
        "final_segment": final_segment,
        "final_segment_t": torch.tensor([final_segment], dtype=torch.int32),
        "final_owner_rank": final_owner_rank,
        "final_owner_part": final_owner_part,
        "owner_rank_table": owner_rank_table,
        "owner_part_table": owner_part_table,
        "cmp_block_table": _cmp_block_tables(cp_size),
        "compress_state_block_table": _state_block_tables(cp_size),
    }
    return metadata


@pl.jit.inline
def prefill_cp_hca_core(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
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
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
    tail_comm_epoch: pl.Scalar[pl.INT32],
    compact_comm_epoch_base: pl.Scalar[pl.INT32],
):
    """CP-HCA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit).

    ``tail_comm_epoch`` and ``compact_comm_epoch_base`` drive the shared
    cross-layer ready/consumed counters of the dual-tail and HCA compact
    domains respectively; local payload rows stay at 0 (``EPOCHS == 1``).
    Standalone/single-layer callers pass 0 for both, preserving behavior.
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
    cmp_indices_flat = pl.reshape(cmp_indices, [LOCAL_ROWS, IDX_TOPK])

    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [row0, 0, 0])
        mixed_tile = pl.slice(mixed, [TAIL_ROWS, D], [row0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [row0, 0])
        position_tile = pl.slice(query_positions_flat, [TAIL_ROWS], [row0])
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        normed_tile = pl.slice(normed, [TAIL_ROWS, D], [row0, 0])
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [row0, 0, 0])
        kv_tile = pl.slice(local_kv, [TAIL_ROWS, HEAD_DIM], [row0, 0])
        qr_tile = pl.slice(qr, [TAIL_ROWS, Q_LORA], [row0, 0])
        qr_scale_tile = pl.slice(qr_scale, [TAIL_ROWS, 1], [row0, 0])
        hc_pre(
            x_tile,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            mixed_tile,
            post_tile,
            comb_tile,
        )
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
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

    local_hidden_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], dtype=pl.BF16, init_value=0.0)
    local_kv_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_tail_assemble"):
        for part in pl.range(LOCAL_PARTS):
            first_len = pl.read(overlay_active_lengths, [part, 0, 1])
            second_len = pl.read(overlay_active_lengths, [part, 1, 1])
            total = first_len + second_len
            tail_offset0 = pl.max(total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_offset0 + row
                if tail_offset < total:
                    if tail_offset < TAIL_ROWS:
                        source = (
                            part * MAX_SEGMENT_TILES * TAIL_ROWS
                            + tail_offset
                        )
                    else:
                        source = (
                            part * MAX_SEGMENT_TILES * TAIL_ROWS
                            + tail_offset
                        )
                    destination = part * TAIL_ROWS + row
                    local_hidden_tail[destination : destination + 1, :] = normed[source : source + 1, :]
                    local_kv_tail[destination : destination + 1, :] = local_kv[source : source + 1, :]

    logical_hidden = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    logical_kv = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_tail_exchange"):
        _prefill_cp_dual_tail_exchange_wave(
            local_hidden_tail, local_kv_tail,
            reverse_index, owner_rank_table,
            hidden_tail_window, kv_tail_window, tail_ready, tail_consumed,
            logical_hidden, logical_kv,
            my_rank, pl.cast(0, pl.INT32), tail_comm_epoch,
        )

    effective_x = pl.create_tensor(
        [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS, D],
        dtype=pl.BF16,
        init_value=0.0,
    )
    leaf_positions = pl.create_tensor(
        [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS],
        dtype=pl.INT32,
        init_value=0,
    )
    leaf_num_tokens = pl.create_tensor(
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES],
        dtype=pl.INT32,
        init_value=0,
    )
    leaf_cmp_slots = pl.create_tensor(
        [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS],
        dtype=pl.INT64,
        init_value=-1,
    )
    leaf_state_slots = pl.create_tensor(
        [LOCAL_PARTS * MAX_COMPRESS_LEAVES * TAIL_ROWS],
        dtype=pl.INT64,
        init_value=-1,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_leaf_lowering"):
        for part in pl.range(LOCAL_PARTS):
            predecessor = pl.read(predecessor_segments, [part])
            predecessor_valid = pl.read(
                overlay_active_lengths, [part, 0, 0]
            )
            pl.write(leaf_num_tokens, [part, 0], predecessor_valid)
            for leaf_row in pl.range(TAIL_ROWS):
                leaf_index = part * MAX_COMPRESS_LEAVES * TAIL_ROWS + leaf_row
                if predecessor >= 0 and leaf_row < predecessor_valid:
                    source = predecessor * TAIL_ROWS + leaf_row
                    effective_x[leaf_index:leaf_index + 1, :] = logical_hidden[source:source + 1, :]
                    position = pl.read(
                        segment_tail_positions, [predecessor, leaf_row]
                    )
                    pl.write(leaf_positions, [leaf_index], position)
                    if position >= 0:
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(
                            compress_state_block_table, [logical_block]
                        )
                        if physical_block >= 0:
                            predecessor_state_row = (
                                pl.cast(physical_block, pl.INT64)
                                * HCA_STATE_BLOCK_SIZE
                                + position % HCA_STATE_BLOCK_SIZE
                            )
                            pl.write(
                                leaf_state_slots,
                                [leaf_index],
                                predecessor_state_row,
                            )
            for tile in pl.range(MAX_SEGMENT_TILES):
                leaf = 1 + tile
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                pl.write(leaf_num_tokens, [part, leaf], active)
                local_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
                leaf_row0 = (
                    part * MAX_COMPRESS_LEAVES + leaf
                ) * TAIL_ROWS
                for leaf_row in pl.range(TAIL_ROWS):
                    destination = leaf_row0 + leaf_row
                    if leaf_row < active:
                        source = local_row0 + leaf_row
                        effective_x[destination:destination + 1, :] = normed[source:source + 1, :]
                        position = pl.read(query_positions_flat, [source])
                        pl.write(leaf_positions, [destination], position)
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(
                            compress_state_block_table, [logical_block]
                        )
                        if physical_block >= 0:
                            local_state_row = (
                                pl.cast(physical_block, pl.INT64)
                                * HCA_STATE_BLOCK_SIZE
                                + position % HCA_STATE_BLOCK_SIZE
                            )
                            pl.write(
                                leaf_state_slots,
                                [destination],
                                local_state_row,
                            )
                        if (position + 1) % COMPRESS_RATIO == 0:
                            pl.write(
                                leaf_cmp_slots,
                                [destination],
                                pl.cast(0, pl.INT64),
                            )

    scratch_state = pl.create_tensor(
        [
            LOCAL_PARTS * HCA_STATE_PHYSICAL_BLOCKS,
            HCA_STATE_BLOCK_SIZE,
            COMPRESS_STATE_DIM,
        ],
        dtype=pl.FP32,
        init_value=0.0,
    )
    persistent_state_flat = pl.reshape(compress_state, [STATE_ROWS, COMPRESS_STATE_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * STATE_ROWS, COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_seed_state"):
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            if segment == 0:
                for state_row in pl.range(STATE_ROWS):
                    destination = part * STATE_ROWS + state_row
                    persistent_state_row = persistent_state_flat[state_row : state_row + 1, :]
                    scratch_state_flat[destination : destination + 1, :] = persistent_state_row

    leaf_cmp = pl.create_tensor(
        [
            LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS,
            CMP_STORAGE_BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        dtype=pl.BF16,
        init_value=0.0,
    )
    for part in pl.range(LOCAL_PARTS):
        state_base = part * HCA_STATE_PHYSICAL_BLOCKS
        state_part = pl.slice(
            scratch_state,
            [
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            [state_base, 0, 0],
        )
        for leaf in pl.range(MAX_COMPRESS_LEAVES):
            leaf_index = part * MAX_COMPRESS_LEAVES + leaf
            token0 = leaf_index * TAIL_ROWS
            cmp_block0 = leaf_index * LEAF_CMP_BLOCKS
            x_leaf = pl.slice(effective_x, [TAIL_ROWS, D], [token0, 0])
            position_leaf = pl.slice(leaf_positions, [TAIL_ROWS], [token0])
            cmp_slots_leaf = pl.slice(leaf_cmp_slots, [TAIL_ROWS], [token0])
            state_slots_leaf = pl.slice(leaf_state_slots, [TAIL_ROWS], [token0])
            cmp_leaf = pl.slice(
                leaf_cmp,
                [LEAF_CMP_BLOCKS, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [cmp_block0, 0, 0, 0],
            )
            active = pl.read(leaf_num_tokens, [part, leaf])
            cmp_leaf, state_part = prefill_compressor_ratio128(
                x_leaf, state_part, compress_state_block_table,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                freqs_cos, freqs_sin,
                cmp_leaf, position_leaf, active,
                cmp_slots_leaf, state_slots_leaf,
            )
            leaf_cmp = pl.assemble(leaf_cmp, cmp_leaf, [cmp_block0, 0, 0, 0])
        scratch_state = pl.assemble(scratch_state, state_part, [state_base, 0, 0])

    local_cmp_payload = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    local_cmp_meta = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], dtype=pl.INT32, init_value=-1)
    local_state_payload = pl.create_tensor([EPOCHS * TAIL_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32, init_value=0.0)
    local_state_meta = pl.create_tensor([EPOCHS, STATE_META_DIM], dtype=pl.INT32, init_value=-1)
    leaf_cmp_flat = pl.reshape(leaf_cmp, [LEAF_CMP_ROWS, HEAD_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * STATE_ROWS, COMPRESS_STATE_DIM])
    final_segment = pl.read(final_segment_t, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_pack_compact"):
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            for tile in pl.range(MAX_SEGMENT_TILES):
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                query_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
                for row in pl.range(TAIL_ROWS):
                    if row < active:
                        position = pl.read(
                            query_positions_flat, [query_row0 + row]
                        )
                        if (position + 1) % COMPRESS_RATIO == 0:
                            destination = part * CMP_ROWS_PER_SEGMENT + tile
                            leaf_index = (
                                part * MAX_COMPRESS_LEAVES + 1 + tile
                            )
                            source = (
                                leaf_index
                                * LEAF_CMP_BLOCKS
                                * CMP_STORAGE_BLOCK_SIZE
                            )
                            local_cmp_payload[
                                destination:destination + 1, :
                            ] = leaf_cmp_flat[source:source + 1, :]
                            pl.write(
                                local_cmp_meta,
                                [destination, 0],
                                pl.cast(1, pl.INT32),
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 1],
                                segment,
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 2],
                                position,
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 3],
                                pl.cast(
                                    (position + 1) // COMPRESS_RATIO - 1,
                                    pl.INT32,
                                ),
                            )
            if segment == final_segment:
                valid = pl.read(snapshot_valid, [part])
                end_position = (
                    pl.read(segment_starts_t, [segment])
                    + pl.read(segment_active_lengths, [part])
                )
                pl.write(
                    local_state_meta, [0, 0], pl.cast(1, pl.INT32)
                )
                pl.write(local_state_meta, [0, 1], segment)
                pl.write(local_state_meta, [0, 2], valid)
                pl.write(local_state_meta, [0, 3], end_position)
                for row in pl.range(TAIL_ROWS):
                    if row < valid:
                        position = pl.read(snapshot_positions, [part, row])
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(
                            compress_state_block_table, [logical_block]
                        )
                        if physical_block >= 0:
                            source = (
                                part * STATE_ROWS
                                + pl.cast(physical_block, pl.INDEX)
                                * HCA_STATE_BLOCK_SIZE
                                + position % HCA_STATE_BLOCK_SIZE
                            )
                            local_state_payload[
                                row:row + 1, :
                            ] = scratch_state_flat[source:source + 1, :]

    cmp_kv_flat = pl.reshape(
        cmp_kv,
        [PREFILL_CMP_BLOCK_NUM * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM],
    )
    compress_state_flat = pl.reshape(
        compress_state, [STATE_ROWS, COMPRESS_STATE_DIM]
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_hca_compact_commit",
    ):
        cmp_kv_flat = _prefill_cp_hca_compact_exchange_commit_wave(
            local_cmp_payload,
            local_cmp_meta,
            local_state_payload,
            local_state_meta,
            owner_rank_table,
            owner_part_table,
            cmp_block_table,
            compress_state_block_table,
            cmp_window,
            cmp_meta_window,
            state_window,
            state_meta_window,
            compact_ready,
            compact_consumed,
            cmp_kv_flat,
            compress_state_flat,
            my_rank,
            pl.cast(0, pl.INT32),
            compact_comm_epoch_base,
        )

    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32, init_value=FP32_NEG_INF)
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, init_value=0)
    _prefill_cp_sparse_stage(
        cache_flat, local_kv, logical_kv,
        cmp_kv,
        cmp_block_table,
        pl.cast(CMP_STORAGE_BLOCK_SIZE, pl.INT32),
        query_positions_flat, query_requests_flat,
        overlay_positions_flat, overlay_requests_flat,
        predecessor_segments, segment_starts_t,
        swa_indices_flat, cmp_indices_flat,
        sparse_kv, sparse_bias, valid_mask, overlay_active_flat,
    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_raw_commit"):
        for row in pl.range(TAIL_ROWS):
            raw_segment = pl.read(final_win_seg_src, [row])
            raw_source_row = pl.read(final_win_row_src, [row])
            raw_destination = pl.read(final_slot_mapping, [row])
            if raw_segment >= 0 and raw_source_row >= 0 and raw_destination >= 0:
                raw_source = raw_segment * TAIL_ROWS + raw_source_row
                cache_flat[raw_destination : raw_destination + 1, :] = logical_kv[raw_source : raw_source + 1, :]

    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        sparse_row0 = row0 * PREFILL_SPARSE_PAD
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [row0, 0, 0])
        sparse_tile = pl.slice(
            sparse_kv,
            [TAIL_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM],
            [sparse_row0, 0],
        )
        bias_tile = pl.slice(
            sparse_bias, [TAIL_ROWS, PREFILL_SPARSE_PAD], [row0, 0]
        )
        mask_tile = pl.slice(
            valid_mask, [TAIL_ROWS, VALID_BLOCK_MASK_COLS], [row0, 0]
        )
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_DIM], [row0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [row0, 0])
        residual_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [row0, 0, 0])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        attn_out_tile = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
        y_tile = pl.create_tensor(
            [TAIL_ROWS, HC_MULT, D], dtype=pl.FP32, init_value=0.0
        )
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
        x_out_flat[row0 : row0 + TAIL_ROWS, 0:HC_MULT, 0:D] = y_tile

    return pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])


@pl.jit
def prefill_cp_hca_rank(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
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
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
):
    """Standalone CP-HCA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    return prefill_cp_hca_core(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        segment_starts_t, segment_active_lengths,
        owner_segments_t, predecessor_segments,
        query_positions, query_requests,
        overlay_positions, overlay_requests,
        overlay_active_lengths, swa_indices, cmp_indices,
        segment_tail_positions,
        snapshot_positions, snapshot_valid, final_segment_t,
        reverse_index, owner_rank_table, owner_part_table,
        final_win_seg_src, final_win_row_src, final_slot_mapping,
        hidden_tail_window, kv_tail_window,
        tail_ready, tail_consumed,
        cmp_window, cmp_meta_window,
        state_window, state_meta_window,
        compact_ready, compact_consumed,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, my_rank,
        pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
    )


@pl.jit.host
def prefill_cp_hca_test(
    x_hc: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            HC_MULT,
            D,
        ],
        pl.FP32,
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [CP_SIZE, HCA_STATE_MAX_BLOCKS], pl.INT32
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
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    query_positions: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
        ],
        pl.INT32,
    ],
    query_requests: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
        ],
        pl.INT32,
    ],
    overlay_positions: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_ROWS,
        ],
        pl.INT32,
    ],
    overlay_requests: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_ROWS,
        ],
        pl.INT32,
    ],
    overlay_active_lengths: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_SOURCES,
        ],
        pl.INT32,
    ],
    swa_indices: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            WIN,
        ],
        pl.INT32,
    ],
    cmp_indices: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            IDX_TOPK,
        ],
        pl.INT32,
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_valid: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [
                CP_SIZE,
                LOCAL_PARTS,
                MAX_SEGMENT_TILES,
                TAIL_ROWS,
                HC_MULT,
                D,
            ],
            pl.FP32,
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
    tail_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    cmp_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    cmp_meta_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
    )
    state_window_buf = pld.alloc_window_buffer(
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32
    )
    state_meta_window_buf = pld.alloc_window_buffer(
        [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
    )
    compact_ready_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    compact_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(
            hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        kv_tail_window = pld.window(
            kv_tail_buf,
            [CP_TAIL_WINDOW_ROWS, HEAD_DIM],
            dtype=pl.BF16,
        )
        tail_ready = pld.window(
            tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        cmp_window = pld.window(
            cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        cmp_meta_window = pld.window(
            cmp_meta_window_buf,
            [CMP_WINDOW_ROWS, CMP_META_DIM],
            dtype=pl.INT32,
        )
        state_window = pld.window(
            state_window_buf,
            [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM],
            dtype=pl.FP32,
        )
        state_meta_window = pld.window(
            state_meta_window_buf,
            [CP_SIZE, STATE_META_DIM],
            dtype=pl.INT32,
        )
        compact_ready = pld.window(
            compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        compact_consumed = pld.window(
            compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        prefill_cp_hca_rank(
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
            compress_state[rank],
            compress_state_block_table[rank],
            kv_cache[rank],
            cmp_kv[rank],
            cmp_block_table[rank],
            segment_starts_t,
            segment_active_lengths[rank],
            owner_segments_t[rank],
            predecessor_segments[rank],
            query_positions[rank],
            query_requests[rank],
            overlay_positions[rank],
            overlay_requests[rank],
            overlay_active_lengths[rank],
            swa_indices[rank],
            cmp_indices[rank],
            segment_tail_positions,
            snapshot_positions[rank],
            snapshot_valid[rank],
            final_segment_t,
            reverse_index,
            owner_rank_table,
            owner_part_table,
            final_win_seg_src,
            final_win_row_src,
            final_slot_mapping,
            hidden_tail_window,
            kv_tail_window,
            tail_ready,
            tail_consumed,
            cmp_window,
            cmp_meta_window,
            state_window,
            state_meta_window,
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


def _state_physical_row(table, absolute_position: int) -> int:
    if absolute_position < 0 or absolute_position >= MAX_SEQ_LEN:
        return -1
    logical_block = absolute_position // HCA_STATE_BLOCK_SIZE
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return (
        physical_block * HCA_STATE_BLOCK_SIZE
        + absolute_position % HCA_STATE_BLOCK_SIZE
    )


def _cmp_physical_row(table, logical_slot: int) -> int:
    if logical_slot < 0:
        return -1
    logical_block = logical_slot // CMP_STORAGE_BLOCK_SIZE
    if logical_block >= table.numel():
        return -1
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return (
        physical_block * CMP_STORAGE_BLOCK_SIZE
        + logical_slot % CMP_STORAGE_BLOCK_SIZE
    )


def build_tensor_specs(cp_size: int = CP_SIZE):
    """Build the canonical CP-HCA fixture."""
    import torch
    from golden import TensorSpec

    if cp_size != CP_SIZE:
        raise ValueError(
            f"runtime cp_size={cp_size} does not match static CP_SIZE={CP_SIZE}"
    )
    metadata = build_hca_metadata(cp_size)
    raw_metadata = _build_raw_attention_metadata(cp_size)
    torch.manual_seed(4100 + cp_size * 31)
    qkv_specs = {spec.name: spec for spec in build_qkv_tensor_specs(1, TAIL_ROWS)}
    sparse_specs = {
        spec.name: spec
        for spec in build_sparse_attn_tensor_specs(COMPRESS_RATIO, TAIL_ROWS)
    }
    compressor_specs = {
        spec.name: spec for spec in build_compressor_tensor_specs(0)
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
    hca_values = {name: qkv_specs[name].create_tensor() for name in qkv_names}
    hca_values.update(
        {name: sparse_specs[name].create_tensor() for name in tail_names}
    )
    for source_name, target_name in (
        ("wkv", "cmp_wkv"),
        ("wgate", "cmp_wgate"),
        ("ape", "cmp_ape"),
        ("norm_w", "cmp_norm_w"),
    ):
        hca_values[target_name] = compressor_specs[source_name].create_tensor()
    hca_values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    hca_values["hc_attn_scale"] = torch.randn(3)
    hca_values["hc_attn_base"] = torch.randn(MIX_HC)
    hca_values["attn_norm_w"] = torch.ones(D, dtype=torch.bfloat16)
    hca_values["freqs_cos"], hca_values["freqs_sin"] = (
        build_rope_tables(
            M, COMPRESS_RATIO, dtype=torch.bfloat16
        )
    )
    x_generator = torch.Generator().manual_seed(4100 + cp_size * 31)
    x_hc = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT,
        D,
        dtype=torch.float32,
    )
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    raw_metadata["overlay_active_lengths"][rank, part, tile, 1]
                )
                if active:
                    x_hc[rank, part, tile, :active].uniform_(
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

    common_names = (
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
    )
    specs = [
        TensorSpec(
            "x_hc",
            list(x_hc.shape),
            torch.float32,
            init_value=x_hc,
        )
    ]
    for name in common_names:
        value = hca_values[name]
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )

    state_tables = metadata["compress_state_block_table"]
    generator = torch.Generator().manual_seed(
        20260731 + cp_size * 101
    )
    state = torch.zeros(
        cp_size,
        HCA_STATE_PHYSICAL_BLOCKS,
        HCA_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
        dtype=torch.float32,
    )
    prefix = int(metadata["prefix"])
    if prefix:
        logical_values = {
            position: (
                torch.rand(COMPRESS_STATE_DIM, generator=generator) - 0.5
            )
            * 0.05
            for position in range(max(0, prefix - COMPRESS_RATIO), prefix)
        }
        for rank in range(cp_size):
            flat = state[rank].view(-1, COMPRESS_STATE_DIM)
            for position, value in logical_values.items():
                row = _state_physical_row(state_tables[rank], position)
                if row >= 0:
                    flat[row] = value

    cmp_cache = torch.zeros(
        cp_size,
        PREFILL_CMP_BLOCK_NUM,
        CMP_STORAGE_BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    completed_prefix = prefix // COMPRESS_RATIO
    if completed_prefix:
        logical_cmp = (
            torch.rand(completed_prefix, HEAD_DIM, generator=generator) - 0.5
        ).to(torch.bfloat16) * 0.1
        for rank in range(cp_size):
            flat = cmp_cache[rank].view(-1, HEAD_DIM)
            for slot in range(completed_prefix):
                row = _cmp_physical_row(
                    metadata["cmp_block_table"][rank], slot
                )
                if row >= 0:
                    flat[row] = logical_cmp[slot]

    specs.extend(
        [
            TensorSpec(
                "compress_state",
                list(state.shape),
                state.dtype,
                init_value=state,
                is_output=True,
            ),
            TensorSpec(
                "compress_state_block_table",
                list(state_tables.shape),
                state_tables.dtype,
                init_value=state_tables,
            ),
            TensorSpec(
                "kv_cache",
                list(kv_cache.shape),
                kv_cache.dtype,
                init_value=kv_cache,
                is_output=True,
            ),
            TensorSpec(
                "cmp_kv",
                list(cmp_cache.shape),
                cmp_cache.dtype,
                init_value=cmp_cache,
                is_output=True,
            ),
            TensorSpec(
                "cmp_block_table",
                list(metadata["cmp_block_table"].shape),
                metadata["cmp_block_table"].dtype,
                init_value=metadata["cmp_block_table"],
            ),
        ]
    )

    device_values = {
        "segment_starts_t": metadata["segment_starts"],
        "segment_active_lengths": raw_metadata["segment_active_lengths"],
        "owner_segments_t": metadata["owner_segments"],
        "predecessor_segments": raw_metadata["predecessor_segments"],
        "query_positions": raw_metadata["query_position_ids"],
        "query_requests": raw_metadata["query_token_to_request"],
        "overlay_positions": raw_metadata["overlay_position_ids"],
        "overlay_requests": raw_metadata["overlay_token_to_request"],
        "overlay_active_lengths": raw_metadata["overlay_active_lengths"],
        "swa_indices": raw_metadata["swa_indices"],
        "cmp_indices": metadata["cmp_indices"],
        "segment_tail_positions": metadata["segment_tail_positions"],
        "snapshot_positions": metadata["snapshot_positions"],
        "snapshot_valid": metadata["snapshot_valid"],
        "final_segment_t": metadata["final_segment_t"],
        "reverse_index": raw_metadata["reverse_index"],
        "owner_rank_table": metadata["owner_rank_table"],
        "owner_part_table": metadata["owner_part_table"],
        "final_win_seg_src": raw_metadata["final_win_seg_src"],
        "final_win_row_src": raw_metadata["final_win_row_src"],
        "final_slot_mapping": raw_metadata["final_slot_mapping"],
    }
    for name, value in device_values.items():
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )
    for name in tail_names:
        value = hca_values[name]
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )
    specs.append(
        TensorSpec(
            "x_out",
            list(x_hc.shape),
            torch.float32,
            is_output=True,
        )
    )

    golden_prefill_cp_hca._ctx = {
        "cp_size": cp_size,
        "prefix": prefix,
        "lengths": [int(value) for value in metadata["segment_lengths"]],
        "starts": [int(value) for value in metadata["segment_starts"]],
        "owners": metadata["owner_segments"].tolist(),
        "final_segment": int(metadata["final_segment"]),
    }
    return specs


def golden_prefill_cp_hca(tensors):
    """Compose CP-HCA golden outputs in logical-segment order."""
    import torch

    ctx = getattr(golden_prefill_cp_hca, "_ctx", None)
    if ctx is None:
        raise RuntimeError("CP-HCA golden context was not installed")
    cp_size = ctx["cp_size"]
    starts = ctx["starts"]
    lengths = ctx["lengths"]
    owners = ctx["owners"]

    local_q = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        H,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_kv = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_norm = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        D,
        dtype=torch.bfloat16,
    )
    local_post = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT,
    )
    local_comb = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT * HC_MULT,
    )
    logical_hidden = torch.zeros(
        NUM_SEGMENTS, TAIL_ROWS, D, dtype=torch.bfloat16
    )
    logical_kv = torch.zeros(
        NUM_SEGMENTS, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16
    )

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 1]
                )
                x_tile = tensors["x_hc"][rank, part, tile]
                mixed = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                post = torch.zeros(TAIL_ROWS, HC_MULT)
                comb = torch.zeros(TAIL_ROWS, HC_MULT * HC_MULT)
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
                rope_positions = positions.clamp_min(0).to(torch.long)
                q = torch.zeros(
                    TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16
                )
                kv = torch.zeros(
                    TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16
                )
                golden_qkv_proj_rope(
                    {
                        "x": normed,
                        "wq_a": tensors["wq_a"],
                        "wq_b": tensors["wq_b"],
                        "wq_b_scale": tensors["wq_b_scale"],
                        "wkv": tensors["wkv"],
                        "rope_cos": tensors["freqs_cos"].index_select(
                            0, rope_positions
                        ),
                        "rope_sin": tensors["freqs_sin"].index_select(
                            0, rope_positions
                        ),
                        "gamma_cq": tensors["gamma_cq"],
                        "gamma_ckv": tensors["gamma_ckv"],
                        "q": q,
                        "kv": kv,
                        "qr": torch.zeros(
                            TAIL_ROWS, Q_LORA, dtype=torch.int8
                        ),
                        "qr_scale": torch.zeros(TAIL_ROWS, 1),
                    }
                )
                local_q[rank, part, tile] = q
                local_kv[rank, part, tile] = kv
                local_norm[rank, part, tile] = normed
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb

            first = int(
                tensors["overlay_active_lengths"][rank, part, 0, 1]
            )
            second = int(
                tensors["overlay_active_lengths"][rank, part, 1, 1]
            )
            total = first + second
            if total:
                hidden_rows = torch.cat(
                    [
                        local_norm[rank, part, 0, :first],
                        local_norm[rank, part, 1, :second],
                    ],
                    dim=0,
                )
                kv_rows = torch.cat(
                    [
                        local_kv[rank, part, 0, :first],
                        local_kv[rank, part, 1, :second],
                    ],
                    dim=0,
                )
                valid = min(TAIL_ROWS, total)
                logical_hidden[segment, :valid] = hidden_rows[-valid:]
                logical_kv[segment, :valid] = kv_rows[-valid:]

    compressed_rows = []
    segment_snapshots = {}
    initial_state = tensors["compress_state"].clone()
    for segment in range(NUM_SEGMENTS):
        owner = cp_owner_rank(segment, cp_size)
        part = cp_owner_part(segment, cp_size)
        state_table = tensors["compress_state_block_table"][owner]
        scratch = torch.zeros_like(initial_state[owner])
        if segment == 0:
            scratch.copy_(initial_state[owner])

        leaves = []
        predecessor = segment - 1
        if predecessor >= 0:
            pred_valid = min(TAIL_ROWS, lengths[predecessor])
            pred_x = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
            pred_positions = torch.zeros(TAIL_ROWS, dtype=torch.int32)
            if pred_valid:
                pred_x[:pred_valid] = logical_hidden[predecessor, :pred_valid]
                pred_positions[:pred_valid] = tensors[
                    "segment_tail_positions"
                ][predecessor, :pred_valid]
            leaves.append((pred_x, pred_positions, pred_valid, False))
        else:
            leaves.append(
                (
                    torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16),
                    torch.zeros(TAIL_ROWS, dtype=torch.int32),
                    0,
                    False,
                )
            )
        for tile in range(MAX_SEGMENT_TILES):
            active = int(
                tensors["overlay_active_lengths"][owner, part, tile, 1]
            )
            leaves.append(
                (
                    local_norm[owner, part, tile],
                    tensors["query_positions"][owner, part, tile],
                    active,
                    True,
                )
            )

        for leaf, (leaf_x, positions, active, publish) in enumerate(leaves):
            cmp_slots = torch.full(
                (TAIL_ROWS,), -1, dtype=torch.int64
            )
            state_slots = torch.full_like(cmp_slots, -1)
            logical_slot = -1
            for row in range(active):
                position = int(positions[row])
                state_slots[row] = _state_physical_row(
                    state_table, position
                )
                if publish and (position + 1) % COMPRESS_RATIO == 0:
                    cmp_slots[row] = 0
                    logical_slot = (position + 1) // COMPRESS_RATIO - 1
            leaf_cmp = torch.zeros(
                LEAF_CMP_BLOCKS,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                HEAD_DIM,
                dtype=torch.bfloat16,
            )
            golden_prefill_compressor_ratio128(
                {
                    "x": leaf_x,
                    "compress_state": scratch,
                    "compress_state_block_table": state_table,
                    "wkv": tensors["cmp_wkv"],
                    "wgate": tensors["cmp_wgate"],
                    "ape": tensors["cmp_ape"],
                    "norm_w": tensors["cmp_norm_w"],
                    "freqs_cos": tensors["freqs_cos"],
                    "freqs_sin": tensors["freqs_sin"],
                    "cmp_kv": leaf_cmp,
                    "position_ids": positions,
                    "num_tokens": active,
                    "cmp_slot_mapping": cmp_slots,
                    "state_slot_mapping": state_slots,
                }
            )
            if logical_slot >= 0:
                compressed_rows.append(
                    (logical_slot, leaf_cmp.view(-1, HEAD_DIM)[0].clone())
                )

        valid = int(tensors["snapshot_valid"][owner, part])
        snapshot = torch.zeros(TAIL_ROWS, COMPRESS_STATE_DIM)
        scratch_flat = scratch.view(-1, COMPRESS_STATE_DIM)
        for row in range(valid):
            position = int(tensors["snapshot_positions"][owner, part, row])
            source = _state_physical_row(state_table, position)
            if source >= 0:
                snapshot[row] = scratch_flat[source]
        segment_snapshots[segment] = snapshot

    cmp_result = tensors["cmp_kv"].clone()
    for logical_slot, value in compressed_rows:
        for receiver in range(cp_size):
            destination = _cmp_physical_row(
                tensors["cmp_block_table"][receiver], logical_slot
            )
            if destination >= 0:
                cmp_result[receiver].view(-1, HEAD_DIM)[destination] = value

    state_result = initial_state.clone()
    final_segment = ctx["final_segment"]
    final_snapshot = segment_snapshots[final_segment]
    final_owner = cp_owner_rank(final_segment, cp_size)
    final_part = cp_owner_part(final_segment, cp_size)
    final_valid = int(tensors["snapshot_valid"][final_owner, final_part])
    for receiver in range(cp_size):
        state_flat = state_result[receiver].view(-1, COMPRESS_STATE_DIM)
        table = tensors["compress_state_block_table"][receiver]
        for row in range(final_valid):
            position = int(
                tensors["snapshot_positions"][final_owner, final_part, row]
            )
            destination = _state_physical_row(table, position)
            if destination >= 0:
                state_flat[destination] = final_snapshot[row]

    raw_initial = tensors["kv_cache"].clone()
    output = torch.zeros_like(tensors["x_out"])
    for rank in range(cp_size):
        persistent = raw_initial[rank].view(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 1]
                )
                fake = torch.zeros(
                    ORI_CACHE_ROWS + OVERLAY_ROWS,
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                fake[:ORI_CACHE_ROWS] = persistent
                predecessor = int(
                    tensors["predecessor_segments"][rank, part]
                )
                pred_valid = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 0]
                )
                if pred_valid:
                    if tile == 0 and predecessor >= 0:
                        fake[
                            OVERLAY_BASE:OVERLAY_BASE + pred_valid
                        ] = logical_kv[predecessor, :pred_valid]
                    elif tile > 0:
                        fake[
                            OVERLAY_BASE:OVERLAY_BASE + pred_valid
                        ] = local_kv[rank, part, tile - 1, :pred_valid]
                fake[
                    OVERLAY_BASE + PRED_OVERLAY_ROWS:
                    OVERLAY_BASE + PRED_OVERLAY_ROWS + active
                ] = local_kv[rank, part, tile, :active]
                fake_cache = fake.view(
                    -1, BLOCK_SIZE, 1, HEAD_DIM
                )
                positions = tensors["query_positions"][rank, part, tile]
                rope_positions = positions.clamp_min(0).to(torch.long)
                attn = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                golden_prefill_sparse_attn(
                    {
                        "q": local_q[rank, part, tile],
                        "ori_kv": fake_cache,
                        "swa_indices": tensors["swa_indices"][
                            rank, part, tile
                        ],
                        "cmp_kv": cmp_result[rank],
                        "cmp_block_table": tensors["cmp_block_table"][rank],
                        "cmp_storage_block_size": CMP_STORAGE_BLOCK_SIZE,
                        "cmp_indices": tensors["cmp_indices"][
                            rank, part, tile
                        ],
                        "attn_sink": tensors["attn_sink"],
                        "num_tokens": active,
                        "freqs_cos": tensors["freqs_cos"].index_select(
                            0, rope_positions
                        ),
                        "freqs_sin": tensors["freqs_sin"].index_select(
                            0, rope_positions
                        ),
                        "wo_a": tensors["wo_a"],
                        "wo_b": tensors["wo_b"],
                        "wo_b_scale": tensors["wo_b_scale"],
                        "attn_out": attn,
                    }
                )
                y = torch.zeros(TAIL_ROWS, HC_MULT, D)
                golden_hc_post_prefill(
                    {
                        "x": attn,
                        "residual": tensors["x_hc"][rank, part, tile],
                        "post": local_post[rank, part, tile],
                        "comb": local_comb[rank, part, tile],
                        "y": y,
                        "num_tokens": active,
                    }
                )
                output[rank, part, tile] = y

    raw_result = raw_initial.clone().view(cp_size, -1, HEAD_DIM)
    for row in range(TAIL_ROWS):
        segment = int(tensors["final_win_seg_src"][row])
        source_row = int(tensors["final_win_row_src"][row])
        destination = int(tensors["final_slot_mapping"][row])
        if segment >= 0 and source_row >= 0 and destination >= 0:
            raw_result[:, destination] = logical_kv[segment, source_row]

    tensors["compress_state"][:] = state_result
    tensors["cmp_kv"][:] = cmp_result
    tensors["kv_cache"][:] = raw_result.view_as(tensors["kv_cache"])
    tensors["x_out"][:] = output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 context-parallel HCA test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=list(CP_CHOICES))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", action="store_true")
    args = parser.parse_args()

    from golden import ratio_allclose, ratio_reldiff, run_jit

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")
    result = run_jit(
        fn=prefill_cp_hca_test,
        specs=build_tensor_specs(args.cp),
        golden_fn=golden_prefill_cp_hca,
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
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
