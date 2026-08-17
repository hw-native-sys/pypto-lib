# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Context-parallel prefill tail exchange, compact-cache exchange, and sparse-source staging."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    FLASH as M,
    FP32_NEG_INF,
    HCA_STATE_PHYSICAL_BLOCKS,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
)
from prefill_compressor_ratio128 import (
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_STATE_DIM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
)
from prefill_compressor_ratio4 import (
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_STATE_DIM as MAIN_STATE_DIM,
    CSA_STATE_BLOCK_SIZE as MAIN_STATE_BLOCK_SIZE,
    HEAD_DIM as MAIN_HEAD_DIM,
)
from prefill_cp_zigzag import (
    CP_SIZE,
    CP_TAIL_WINDOW_ROWS,
    EPOCHS,
    HEAD_DIM,
    NUM_SEGMENTS,
    ROW_TILE,
    TAIL_ROWS,
)
from prefill_indexer_compressor import (
    COMPRESS_STATE_DIM as INNER_STATE_DIM,
    HEAD_DIM as INNER_HEAD_DIM,
    INNER_STATE_BLOCK_SIZE,
)
from prefill_sparse_attn import (
    BIAS_TOKEN_TILE,
    PREFILL_ATTN_BLOCKS,
    PREFILL_ATTN_TILE,
    PREFILL_SPARSE_PAD,
    SPARSE_BIAS_COLS,
    SPARSE_CMP_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
)

CP_CMP_BLOCK_NUM_DYN = pl.dynamic("CP_CMP_BLOCK_NUM_DYN")
CP_CMP_STORAGE_BLOCK_SIZE_DYN = pl.dynamic("CP_CMP_STORAGE_BLOCK_SIZE_DYN")


# model config
D = M.hidden_size
WIN = M.sliding_window
IDX_TOPK = M.index_topk

# CP exchange layout
LOCAL_PARTS = 2
MAX_SEGMENT_TILES = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
ORI_CACHE_ROWS = PREFILL_ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2

CMP_ROWS_PER_SEGMENT = 2
CMP_ROWS_PER_RANK = LOCAL_PARTS * CMP_ROWS_PER_SEGMENT
CMP_META_DIM = 8
STATE_META_DIM = 8
CMP_WINDOW_ROWS = CP_SIZE * CMP_ROWS_PER_RANK
STATE_WINDOW_ROWS = CP_SIZE * TAIL_ROWS

ROWS_PER_RANK = 128
STATE_ROWS_PER_RANK = 8
META_DIM = 8
RECORDS_PER_WINDOW = CP_SIZE * ROWS_PER_RANK
STATE_RECORDS_PER_WINDOW = CP_SIZE * STATE_ROWS_PER_RANK
SCALE_TILE_COLS = 8
MAIN_CACHE_ROWS = PREFILL_CMP_BLOCK_NUM * CSA_CMP_STORAGE_BLOCK_SIZE
MAIN_STATE_ROWS = CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE
INNER_STATE_ROWS = CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE


@pl.jit.inline
def _prefill_cp_dual_tail_exchange_wave(
    local_hidden_tail: pl.Tensor[
        [EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], pl.BF16
    ],
    local_kv_tail: pl.Tensor[
        [EPOCHS * LOCAL_PARTS * TAIL_ROWS, HEAD_DIM], pl.BF16
    ],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    hidden_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_hidden_out: pl.Out[
        pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]
    ],
    logical_kv_out: pl.Out[
        pl.Tensor[
            [EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]:
    """Exchange hidden and KV tails with one barrier.

    ``payload_epoch`` selects rows in the invocation-local payload/output
    tensors; ``comm_epoch`` drives the shared cross-layer ready/consumed
    counters (``consumed >= comm_epoch``, ``ready >= comm_epoch + 1``).
    Splitting the two lets a multi-layer FWD pass a monotonic communication
    epoch per global layer while keeping the local payload index at 0.
    """
    epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_epoch, cmp=pld.WaitCmp.Ge,
            )

    for peer in pl.range(CP_SIZE):
        for part in pl.range(LOCAL_PARTS):
            publish_pos = my_rank * LOCAL_PARTS + part
            publish_dst_row = publish_pos * TAIL_ROWS
            src_row_base = payload_epoch * LOCAL_PARTS * TAIL_ROWS + part * TAIL_ROWS
            pld.tensor.put(
                dst=hidden_window, peer=peer, src=local_hidden_tail,
                dst_offsets=[publish_dst_row, 0], src_offsets=[src_row_base, 0], shape=[TAIL_ROWS, D],
                chunk_rows=ROW_TILE, chunk_cols=D, pipeline=True,
            )
            pld.tensor.put(
                dst=kv_window, peer=peer, src=local_kv_tail,
                dst_offsets=[publish_dst_row, 0], src_offsets=[src_row_base, 0], shape=[TAIL_ROWS, HEAD_DIM],
                chunk_rows=ROW_TILE, chunk_cols=HEAD_DIM, pipeline=True,
            )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    for seg in pl.range(NUM_SEGMENTS):
        gather_pos = reverse_index[seg]
        owner = owner_rank_table[seg]
        if owner != my_rank:
            pld.system.wait(
                signal=ready, offsets=[owner, 0],
                expected=epoch_value, cmp=pld.WaitCmp.Ge,
            )
        gather_src_row = gather_pos * TAIL_ROWS
        gather_dst_row = payload_epoch * CP_TAIL_WINDOW_ROWS + seg * TAIL_ROWS
        for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
            hidden_tile = hidden_window[gather_src_row + t0 : gather_src_row + t0 + ROW_TILE, 0:D]
            kv_tile = kv_window[gather_src_row + t0 : gather_src_row + t0 + ROW_TILE, 0:HEAD_DIM]
            logical_hidden_out[gather_dst_row + t0 : gather_dst_row + t0 + ROW_TILE, 0:D] = hidden_tile
            logical_kv_out[gather_dst_row + t0 : gather_dst_row + t0 + ROW_TILE, 0:HEAD_DIM] = kv_tile

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    # Return the logical hidden root.
    return logical_hidden_out


@pl.jit.inline
def _prefill_cp_hca_compact_exchange_commit_wave(
    local_cmp_payload: pl.Tensor[
        [EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], pl.BF16
    ],
    local_cmp_meta: pl.Tensor[
        [EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], pl.INT32
    ],
    local_state_payload: pl.Tensor[
        [EPOCHS * TAIL_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    local_state_meta: pl.Tensor[[EPOCHS, STATE_META_DIM], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
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
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM * HCA_CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], pl.BF16
        ]
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_PHYSICAL_BLOCKS * HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[
    [PREFILL_CMP_BLOCK_NUM * HCA_CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], pl.BF16
]:
    """Publish HCA compact rows and commit receiver-local cache/state.

    ``payload_epoch`` selects rows in the invocation-local payload tensors
    (``local_cmp_payload``/``local_cmp_meta``/``local_state_payload``/
    ``local_state_meta``); ``comm_epoch`` drives the HCA compact
    ready/consumed counters (``consumed >= comm_epoch``,
    ``ready >= comm_epoch + 1``).
    """
    epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_epoch, cmp=pld.WaitCmp.Ge,
            )

    cmp_src_row = payload_epoch * CMP_ROWS_PER_RANK
    state_src_row = payload_epoch * TAIL_ROWS
    for peer in pl.range(CP_SIZE):
        cmp_dst_row = my_rank * CMP_ROWS_PER_RANK
        state_dst_row = my_rank * TAIL_ROWS
        pld.tensor.put(
            dst=cmp_window, peer=peer, src=local_cmp_payload,
            dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, HEAD_DIM],
            chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=cmp_meta_window, peer=peer, src=local_cmp_meta,
            dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, CMP_META_DIM],
            chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=CMP_META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=state_window, peer=peer, src=local_state_payload,
            dst_offsets=[state_dst_row, 0], src_offsets=[state_src_row, 0], shape=[TAIL_ROWS, COMPRESS_STATE_DIM],
            chunk_rows=ROW_TILE, chunk_cols=COMPRESS_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=state_meta_window, peer=peer, src=local_state_meta,
            dst_offsets=[my_rank, 0], src_offsets=[payload_epoch, 0], shape=[1, STATE_META_DIM],
            chunk_rows=1, chunk_cols=STATE_META_DIM, pipeline=True,
        )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=ready, offsets=[peer, 0],
                expected=epoch_value, cmp=pld.WaitCmp.Ge,
            )

    for segment in pl.range(NUM_SEGMENTS):
        cmp_owner = owner_rank_table[segment]
        owner_part = owner_part_table[segment]
        for row in pl.range(CMP_ROWS_PER_SEGMENT):
            cmp_source_row = cmp_owner * CMP_ROWS_PER_RANK + owner_part * CMP_ROWS_PER_SEGMENT + row
            valid = pl.read(cmp_meta_window, [cmp_source_row, 0])
            meta_segment = pl.read(cmp_meta_window, [cmp_source_row, 1])
            logical_slot = pl.read(cmp_meta_window, [cmp_source_row, 3])
            if valid > 0:
                if meta_segment == segment:
                    if logical_slot >= 0:
                        logical_block = pl.cast(logical_slot // HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                        if logical_block < PREFILL_CMP_MAX_BLOCKS:
                            physical_block = pl.read(cmp_block_table, [logical_block])
                            if physical_block >= 0:
                                intra = pl.cast(logical_slot % HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                                cmp_row_tile = cmp_window[cmp_source_row : cmp_source_row + 1, 0:HEAD_DIM]
                                cache_row = (
                                    pl.cast(physical_block, pl.INDEX)
                                    * HCA_CMP_STORAGE_BLOCK_SIZE
                                    + intra
                                )
                                cmp_kv[cache_row : cache_row + 1, 0:HEAD_DIM] = cmp_row_tile

    for state_owner in pl.range(CP_SIZE):
        state_valid = pl.read(state_meta_window, [state_owner, 0])
        valid_rows = pl.read(state_meta_window, [state_owner, 2])
        end_position = pl.read(state_meta_window, [state_owner, 3])
        if state_valid > 0:
            for row in pl.range(TAIL_ROWS):
                if row < valid_rows:
                    absolute_position = end_position - valid_rows + row
                    if absolute_position >= 0:
                        if absolute_position < MAX_SEQ_LEN:
                            logical_block = pl.cast(absolute_position // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                            if logical_block < HCA_STATE_MAX_BLOCKS:
                                physical_block = pl.read(compress_state_block_table, [logical_block])
                                if physical_block >= 0:
                                    intra = pl.cast(absolute_position % HCA_STATE_BLOCK_SIZE, pl.INDEX)
                                    state_source_row = state_owner * TAIL_ROWS + row
                                    state_row_tile = pl.slice(
                                        state_window,
                                        [1, COMPRESS_STATE_DIM],
                                        [state_source_row, 0],
                                    )
                                    state_row = pl.cast(physical_block, pl.INDEX) * HCA_STATE_BLOCK_SIZE + intra
                                    compress_state[state_row : state_row + 1, 0:COMPRESS_STATE_DIM] = state_row_tile

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    return cmp_kv


@pl.jit.inline
def _prefill_cp_csa_compact_transport_wave(
    main_payload: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, MAIN_HEAD_DIM], pl.BF16
    ],
    idx_payload: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, INNER_HEAD_DIM], pl.INT8
    ],
    idx_scale: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, SCALE_TILE_COLS], pl.FP32
    ],
    record_meta: pl.Tensor[[EPOCHS * ROWS_PER_RANK, META_DIM], pl.INT32],
    main_state_payload: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, MAIN_STATE_DIM], pl.FP32
    ],
    inner_state_payload: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, INNER_STATE_DIM], pl.FP32
    ],
    main_state_meta: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32
    ],
    inner_state_meta: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32
    ],
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, MAIN_HEAD_DIM], pl.BF16
    ],
    idx_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, INNER_HEAD_DIM], pl.INT8
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
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
):
    comm_i32 = pl.cast(comm_epoch, pl.INT32)
    ready_expected = pl.cast(comm_i32 + 1, pl.INT32)
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_i32, cmp=pld.WaitCmp.Ge,
            )

    payload_row = payload_epoch * ROWS_PER_RANK
    state_row = payload_epoch * STATE_ROWS_PER_RANK
    destination_row = my_rank * ROWS_PER_RANK
    destination_state_row = my_rank * STATE_ROWS_PER_RANK
    for peer in pl.range(CP_SIZE):
        pld.tensor.put(
            dst=main_window, peer=peer, src=main_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, MAIN_HEAD_DIM],
            chunk_rows=8, chunk_cols=MAIN_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=idx_window, peer=peer, src=idx_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, INNER_HEAD_DIM],
            chunk_rows=8, chunk_cols=INNER_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=scale_window, peer=peer, src=idx_scale,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, SCALE_TILE_COLS],
            chunk_rows=8, chunk_cols=SCALE_TILE_COLS, pipeline=True,
        )
        pld.tensor.put(
            dst=record_window, peer=peer, src=record_meta,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, META_DIM],
            chunk_rows=8, chunk_cols=META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_window, peer=peer, src=main_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, MAIN_STATE_DIM],
            chunk_rows=4, chunk_cols=MAIN_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_meta_window, peer=peer, src=main_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_window, peer=peer, src=inner_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, INNER_STATE_DIM],
            chunk_rows=4, chunk_cols=INNER_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_meta_window, peer=peer, src=inner_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=ready, offsets=[peer, 0],
                expected=ready_expected, cmp=pld.WaitCmp.Ge,
            )


@pl.jit.inline
def _prefill_cp_csa_compact_finish_wave(
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )


@pl.jit.inline
def _prefill_cp_sparse_stage(
    cache_flat: pl.Tensor[[ORI_CACHE_ROWS, HEAD_DIM], pl.BF16],
    local_kv: pl.Tensor[[LOCAL_ROWS, HEAD_DIM], pl.BF16],
    logical_tails: pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[
        [
            CP_CMP_BLOCK_NUM_DYN,
            CP_CMP_STORAGE_BLOCK_SIZE_DYN,
            1,
            HEAD_DIM,
        ],
        pl.BF16,
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    cmp_storage_block_size: pl.Scalar[pl.INT32],
    query_positions: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    query_requests: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    overlay_positions: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_requests: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_ROWS, WIN], pl.INT32],
    cmp_indices: pl.Tensor[[LOCAL_ROWS, IDX_TOPK], pl.INT32],
    sparse_kv: pl.Tensor[[LOCAL_SPARSE_ROWS, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[LOCAL_ROWS, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[LOCAL_ROWS, VALID_BLOCK_MASK_COLS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_SOURCES], pl.INT32],
):
    """Stage persistent, overlay, and compressed sparse sources."""
    cmp_cache_rows = pl.tensor.dim(cmp_kv, 0) * pl.tensor.dim(cmp_kv, 1)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    prefix = pl.read(segment_starts_t, [0])
    with pl.spmd((LOCAL_ROWS // 2) * PREFILL_ATTN_BLOCKS, name_hint="prefill_cp_gather_kv"):
        block = pl.tile.get_block_idx()
        schedule = block // PREFILL_ATTN_BLOCKS
        sparse_block = block - schedule * PREFILL_ATTN_BLOCKS
        token_block = (LOCAL_ROWS // 2) - 1 - schedule
        token0 = token_block * 2
        key0 = sparse_block * PREFILL_ATTN_TILE
        for token_delta in pl.range(2):
            row = token0 + token_delta
            if row < LOCAL_ROWS:
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                for key_delta in pl.range(PREFILL_ATTN_TILE):
                    sparse_col = key0 + key_delta
                    if sparse_col < WIN:
                        raw = pl.read(swa_indices, [row, sparse_col])
                        if raw >= 0:
                            query_abs = pl.read(query_positions, [row])
                            query_req = pl.read(query_requests, [row])
                            key_abs = query_abs - WIN + 1 + sparse_col
                            if raw < ORI_CACHE_ROWS:
                                if key_abs < prefix:
                                    if key_abs <= query_abs and query_req >= 0:
                                        source = pl.cast(raw, pl.INDEX)
                                        stage[key_delta:key_delta + 1, :] = cache_flat[
                                            source:source + 1, :
                                        ]
                            elif raw < OVERLAY_BASE + OVERLAY_ROWS:
                                tile = row // TAIL_ROWS
                                overlay_row = raw - OVERLAY_BASE
                                if overlay_row >= PRED_OVERLAY_ROWS:
                                    source_kind = 1
                                    source_row = overlay_row - PRED_OVERLAY_ROWS
                                else:
                                    source_kind = 0
                                    source_row = overlay_row
                                overlay_index = source_row
                                if source_kind == 1:
                                    overlay_index = PRED_OVERLAY_ROWS + source_row
                                active = pl.read(overlay_active_lengths, [tile, source_kind])
                                overlay_abs = pl.read(overlay_positions, [tile, overlay_index])
                                overlay_req = pl.read(overlay_requests, [tile, overlay_index])
                                if source_row >= 0 and source_row < active:
                                    if overlay_abs == key_abs and overlay_abs <= query_abs:
                                        if overlay_req == query_req and overlay_req >= 0:
                                            if source_kind == 1:
                                                source = tile * TAIL_ROWS + source_row
                                                stage[key_delta:key_delta + 1, :] = local_kv[
                                                    source:source + 1, :
                                                ]
                                            elif tile % MAX_SEGMENT_TILES == 0:
                                                part = tile // MAX_SEGMENT_TILES
                                                predecessor = pl.read(predecessor_segments, [part])
                                                if predecessor >= 0:
                                                    source = predecessor * TAIL_ROWS + source_row
                                                    stage[key_delta:key_delta + 1, :] = logical_tails[
                                                        source:source + 1, :
                                                    ]
                                            else:
                                                source = (tile - 1) * TAIL_ROWS + source_row
                                                stage[key_delta:key_delta + 1, :] = local_kv[
                                                    source:source + 1, :
                                                ]
                    else:
                        cmp_col = sparse_col - WIN
                        if cmp_col < IDX_TOPK:
                            logical_slot = pl.read(cmp_indices, [row, cmp_col])
                            if logical_slot >= 0:
                                logical_block = logical_slot // cmp_storage_block_size
                                if logical_block < PREFILL_CMP_MAX_BLOCKS:
                                    physical_block = pl.read(cmp_block_table, [logical_block])
                                    if physical_block >= 0:
                                        source_block = (
                                            pl.cast(physical_block, pl.INDEX)
                                            * cmp_storage_block_size
                                        )
                                        source_intra = pl.cast(
                                            logical_slot % cmp_storage_block_size,
                                            pl.INDEX,
                                        )
                                        source = source_block + source_intra
                                        stage[key_delta:key_delta + 1, :] = cmp_kv_flat[
                                            source:source + 1, :
                                        ]
                output_row = row * PREFILL_SPARSE_PAD + key0
                sparse_kv[output_row:output_row + PREFILL_ATTN_TILE, :] = stage

    with pl.spmd(LOCAL_ROWS // BIAS_TOKEN_TILE, name_hint="prefill_cp_build_bias"):
        bias_block = pl.tile.get_block_idx()
        row0 = bias_block * BIAS_TOKEN_TILE
        raw_idx = pl.cast(swa_indices[row0:row0 + BIAS_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        raw_valid = pl.minimum(pl.maximum(pl.add(raw_idx, 1.0), 0.0), 1.0)
        raw_bias = pl.sub(raw_valid, 1.0)
        sparse_bias[row0:row0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(raw_bias, -FP32_NEG_INF)
        if SPARSE_CMP_BIAS_COLS > 0:
            cmp_idx = pl.cast(cmp_indices[row0:row0 + BIAS_TOKEN_TILE, 0:SPARSE_CMP_BIAS_COLS], target_type=pl.FP32)
            cmp_valid = pl.minimum(pl.maximum(pl.add(cmp_idx, 1.0), 0.0), 1.0)
            cmp_bias = pl.sub(cmp_valid, 1.0)
            sparse_bias[row0:row0 + BIAS_TOKEN_TILE, WIN:SPARSE_BIAS_COLS] = pl.mul(cmp_bias, -FP32_NEG_INF)
        if SPARSE_BIAS_COLS < PREFILL_SPARSE_PAD:
            sparse_bias[
                row0:row0 + BIAS_TOKEN_TILE,
                SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD,
            ] = pl.full(
                [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )

    with pl.spmd(LOCAL_ROWS, name_hint="prefill_cp_build_valid_mask"):
        row = pl.tile.get_block_idx()
        mask = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
        for sparse_block in pl.range(PREFILL_ATTN_BLOCKS):
            block_valid = pl.cast(0, pl.INT32)
            key0 = sparse_block * PREFILL_ATTN_TILE
            for key_delta in pl.range(PREFILL_ATTN_TILE):
                sparse_col = key0 + key_delta
                raw = pl.cast(-1, pl.INT32)
                if sparse_col < WIN:
                    raw = pl.read(swa_indices, [row, sparse_col])
                else:
                    cmp_col = sparse_col - WIN
                    if cmp_col < IDX_TOPK:
                        raw = pl.read(cmp_indices, [row, cmp_col])
                if raw >= 0:
                    block_valid = pl.cast(1, pl.INT32)
            pl.write(mask, [0, sparse_block], block_valid)
        valid_block_mask[row:row + 1, 0:VALID_BLOCK_MASK_COLS] = mask

    return sparse_kv
