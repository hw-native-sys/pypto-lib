# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim

"""Context-parallel prefill tail exchange, compact-cache exchange, and sparse-source staging."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    FLASH as M,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
)
from prefill_compressor_ratio128 import (
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
)
from prefill_compressor_ratio4 import (
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM as MAIN_STATE_DIM,
    CSA_STATE_BLOCK_SIZE as MAIN_STATE_BLOCK_SIZE,
    HEAD_DIM as MAIN_HEAD_DIM,
)
from prefill_cp_zigzag import (
    CP_SIZE,
    CP_PREFILL_CMP_BLOCK_NUM as PREFILL_CMP_BLOCK_NUM,
    CP_TAIL_WINDOW_ROWS,
    EPOCHS,
    HEAD_DIM,
    MAX_SEGMENT_TILES,
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
    HCA_MAX_COMPRESSED_ROWS,
    PREFILL_SPARSE_PAD,
)

HCA_STATE_BLOCKS_DYN = pl.dynamic("CP_HCA_STATE_BLOCKS_DYN")
CP_CMP_BLOCK_NUM_DYN = pl.dynamic("CP_CMP_BLOCK_NUM_DYN")
CP_CMP_ROWS_DYN = pl.dynamic("CP_CMP_ROWS_DYN")
CP_CMP_STORAGE_BLOCK_SIZE_DYN = pl.dynamic("CP_CMP_STORAGE_BLOCK_SIZE_DYN")
HCA_RAW_BLOCKS_DYN = pl.dynamic("CP_HCA_HISTORY_RAW_BLOCKS_DYN")

# model config
D = M.hidden_size
WIN = M.sliding_window
IDX_TOPK = M.index_topk

# CP exchange layout
LOCAL_PARTS = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
ORI_CACHE_ROWS = PREFILL_ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2

CMP_ROWS_PER_SEGMENT = (MAX_SEGMENT_TILES * TAIL_ROWS // HCA_COMPRESS_RATIO)
CMP_ROWS_PER_RANK = LOCAL_PARTS * CMP_ROWS_PER_SEGMENT
CMP_META_DIM = 8
STATE_META_DIM = 8
CMP_WINDOW_ROWS = CP_SIZE * CMP_ROWS_PER_RANK
HCA_STATE_ROWS_PER_PART = MAX_SEGMENT_TILES * TAIL_ROWS
HCA_STATE_ROWS_PER_RANK = LOCAL_PARTS * HCA_STATE_ROWS_PER_PART
HCA_STATE_RECORDS = CP_SIZE * LOCAL_PARTS
STATE_WINDOW_ROWS = CP_SIZE * HCA_STATE_ROWS_PER_RANK

ROWS_PER_RANK = (LOCAL_PARTS * MAX_SEGMENT_TILES * TAIL_ROWS // CSA_COMPRESS_RATIO)
STATE_ROWS_PER_RANK = LOCAL_PARTS * (MAX_SEGMENT_TILES + 1) * 8
META_DIM = 8
RECORDS_PER_WINDOW = CP_SIZE * ROWS_PER_RANK
STATE_RECORDS_PER_WINDOW = CP_SIZE * STATE_ROWS_PER_RANK
# FP16 scale rows must remain 32-byte aligned on PTOAS 0.60.
SCALE_TILE_COLS = 16
MAIN_CACHE_ROWS = PREFILL_CMP_BLOCK_NUM * CSA_CMP_STORAGE_BLOCK_SIZE
MAIN_STATE_ROWS = CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE
INNER_STATE_ROWS = CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE
CP_LAST_HIDDEN_EPOCH = 1


# Serving request transfer uses bounded tiles, independent of prompt length.
CP_REQUEST_TOKENS_DYN = pl.dynamic("CP_REQUEST_TOKENS_DYN")
CP_REQUEST_HC_DIM = M.hc_mult * D
CP_REQUEST_COPY_COLS = 512
CP_REQUEST_CAPACITY = CP_SIZE * LOCAL_ROWS
CP_REQUEST_MAIN_TABLE_COLS = (MAX_SEQ_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
CP_REQUEST_INNER_TABLE_COLS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
CP_REQUEST_TABLE_COLS = max(PREFILL_ORI_MAX_BLOCKS, PREFILL_CMP_MAX_BLOCKS,
    HCA_STATE_MAX_BLOCKS, CP_REQUEST_MAIN_TABLE_COLS, CP_REQUEST_INNER_TABLE_COLS)


@pl.jit.inline
def _prefill_cp_request_header(
    num_tokens_per_owner: pl.Tensor[[CP_SIZE], pl.INT32],
    position_ids: pl.Tensor[[CP_REQUEST_TOKENS_DYN], pl.INT32],
    header: pl.Tensor[[1, 16], pl.INT32],
    request_owner: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    for header_block in pl.spmd(1):
        owner = pl.cast(request_owner, pl.INDEX)
        length = pl.cast(pl.read(num_tokens_per_owner, [owner]), pl.INDEX)
        # A continued chunk arrives with an absolute base; the caches it reads
        # were written by the earlier chunks of the same request.
        base = pl.cast(0, pl.INDEX)
        mode = pl.cast(0, pl.INDEX)
        if owner == my_rank:
            base = pl.cast(pl.read(position_ids, [0]), pl.INDEX)
            if base >= 0:
                if length <= CP_REQUEST_CAPACITY:
                    mode = pl.cast(1, pl.INDEX)
        span = pl.max(TAIL_ROWS, (length + NUM_SEGMENTS - 1) // NUM_SEGMENTS)
        for col in pl.range(16):
            value = pl.cast(0, pl.INT32)
            if col == 0:
                value = pl.cast(mode, pl.INT32)
            elif col == 1:
                value = pl.cast(owner, pl.INT32)
            elif col == 2:
                value = pl.cast(length, pl.INT32)
            elif col == 3:
                value = pl.cast(span, pl.INT32)
            elif col == 4:
                value = pl.cast(base, pl.INT32)
            pl.write(header, [0, col], value)
    return header


@pl.jit.incore
def _prefill_cp_scatter_request(
    header: pl.Tensor[[1, 16], pl.INT32],
    x_hc: pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    input_ids: pl.Tensor[[1, CP_REQUEST_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[1, PREFILL_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[1, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32],
    header_window: pld.DistributedTensor[[1, 16], pl.INT32],
    input_window: pld.DistributedTensor[[LOCAL_ROWS, CP_REQUEST_HC_DIM], pl.FP32],
    ids_window: pld.DistributedTensor[[1, LOCAL_ROWS * 2], pl.INT32],
    tables_window: pld.DistributedTensor[[7, CP_REQUEST_TABLE_COLS], pl.INT32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    control: pl.Out[pl.Tensor[[1, 16], pl.INT32]],
    local_x_hc: pl.Out[pl.Tensor[[LOCAL_ROWS, CP_REQUEST_HC_DIM], pl.FP32]],
    local_input_ids: pl.Out[pl.Tensor[[1, LOCAL_ROWS], pl.INT64]],
    local_ori_block_table: pl.Out[pl.Tensor[[1, PREFILL_ORI_MAX_BLOCKS], pl.INT32]],
    local_hca_cmp_block_table: pl.Out[pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32]],
    local_csa_cmp_block_table: pl.Out[pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32]],
    local_idx_block_table: pl.Out[pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32]],
    local_hca_compress_state_block_table: pl.Out[pl.Tensor[[1, HCA_STATE_MAX_BLOCKS], pl.INT32]],
    local_csa_compress_state_block_table: pl.Out[pl.Tensor[[1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32]],
    local_csa_inner_compress_state_block_table: pl.Out[pl.Tensor[[1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32]],
    my_rank: pl.Scalar[pl.INT32],
):
    owner = pl.read(header, [0, 1])
    if owner >= 0:
        if owner == my_rank:
            for peer in pl.range(CP_SIZE):
                pld.tensor.put(
                    dst=header_window, peer=peer, src=header,
                    dst_offsets=[0, 0], src_offsets=[0, 0], shape=[1, 16],
                    chunk_rows=1, chunk_cols=16,
                )
                if pl.read(header, [0, 0]) == 1:
                    length = pl.read(header, [0, 2])
                    span = pl.read(header, [0, 3])
                    for part in pl.range(LOCAL_PARTS):
                        if part == 0:
                            segment = peer
                        else:
                            segment = NUM_SEGMENTS - 1 - peer
                        start = segment * span
                        for tile in pl.range(MAX_SEGMENT_TILES * TAIL_ROWS // ROW_TILE):
                            active = pl.max(0, pl.min(ROW_TILE, pl.min(span, length - start) - tile * ROW_TILE))
                            destination = part * MAX_SEGMENT_TILES * TAIL_ROWS + tile * ROW_TILE
                            for column in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                                if active > 0:
                                    valid_x = pl.load(
                                        x_hc, [start + tile * ROW_TILE, column * CP_REQUEST_COPY_COLS],
                                        [ROW_TILE, CP_REQUEST_COPY_COLS], valid_shape=[active, CP_REQUEST_COPY_COLS],
                                    )
                                    padded_x = pl.tile.fillpad(valid_x, pad_value=pl.PadValue.zero)
                                    full_x = pl.tile.set_validshape(padded_x, ROW_TILE, CP_REQUEST_COPY_COLS)
                                    pld.tile.remote_store(full_x, input_window, peer, [destination, column * CP_REQUEST_COPY_COLS])
                                else:
                                    zero_x = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.FP32)
                                    pld.tile.remote_store(zero_x, input_window, peer, [destination, column * CP_REQUEST_COPY_COLS])
                            if active > 0:
                                valid_ids = pl.load(input_ids, [0, start + tile * ROW_TILE], [1, ROW_TILE], valid_shape=[1, active])
                                valid_words = pl.reinterpret_view(valid_ids, pl.INT32)
                                padded_words = pl.tile.fillpad(valid_words, pad_value=pl.PadValue.zero)
                                full_words = pl.tile.set_validshape(padded_words, 1, ROW_TILE * 2)
                                pld.tile.remote_store(full_words, ids_window, peer, [0, destination * 2])
                            else:
                                zero_words = pl.tile.full([1, ROW_TILE * 2], value=0, dtype=pl.INT32)
                                pld.tile.remote_store(zero_words, ids_window, peer, [0, destination * 2])
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=ori_block_table,
                        dst_offsets=[0, 0], src_offsets=[0, 0], shape=[1, PREFILL_ORI_MAX_BLOCKS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=hca_cmp_block_table,
                        dst_offsets=[1, 0], src_offsets=[0, 0], shape=[1, PREFILL_CMP_MAX_BLOCKS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=csa_cmp_block_table,
                        dst_offsets=[2, 0], src_offsets=[0, 0], shape=[1, PREFILL_CMP_MAX_BLOCKS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=idx_block_table,
                        dst_offsets=[3, 0], src_offsets=[0, 0], shape=[1, PREFILL_CMP_MAX_BLOCKS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=hca_compress_state_block_table,
                        dst_offsets=[4, 0], src_offsets=[0, 0], shape=[1, HCA_STATE_MAX_BLOCKS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=csa_compress_state_block_table,
                        dst_offsets=[5, 0], src_offsets=[0, 0], shape=[1, CP_REQUEST_MAIN_TABLE_COLS],
                        chunk_rows=1, chunk_cols=16,
                    )
                    pld.tensor.put(
                        dst=tables_window, peer=peer, src=csa_inner_compress_state_block_table,
                        dst_offsets=[6, 0], src_offsets=[0, 0], shape=[1, CP_REQUEST_INNER_TABLE_COLS],
                        chunk_rows=1, chunk_cols=16,
                    )
            for peer in pl.range(CP_SIZE):
                if peer != my_rank:
                    pld.system.notify(target=ready, peer=peer, offsets=[owner, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        else:
            pld.system.wait(signal=ready, offsets=[owner, 0], expected=1, cmp=pld.WaitCmp.Ge)
        received_header = pl.load(header_window, [0, 0], [1, 16])
    else:
        received_header = pl.load(header, [0, 0], [1, 16])
    pl.store(received_header, [0, 0], control)
    if pl.read(control, [0, 0]) == 1:
        for tile in pl.range(LOCAL_ROWS // ROW_TILE):
            for column in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                received_x = pl.load(input_window, [tile * ROW_TILE, column * CP_REQUEST_COPY_COLS], [ROW_TILE, CP_REQUEST_COPY_COLS])
                pl.store(received_x, [tile * ROW_TILE, column * CP_REQUEST_COPY_COLS], local_x_hc)
            received_words = pl.load(ids_window, [0, tile * ROW_TILE * 2], [1, ROW_TILE * 2])
            received_ids = pl.reinterpret_view(received_words, pl.INT64)
            pl.store(received_ids, [0, tile * ROW_TILE], local_input_ids)
        for table_chunk in pl.range(PREFILL_ORI_MAX_BLOCKS // 16):
            received_page = pl.load(tables_window, [0, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_ori_block_table)
        for table_chunk in pl.range(PREFILL_CMP_MAX_BLOCKS // 16):
            received_page = pl.load(tables_window, [1, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_hca_cmp_block_table)
        for table_chunk in pl.range(PREFILL_CMP_MAX_BLOCKS // 16):
            received_page = pl.load(tables_window, [2, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_csa_cmp_block_table)
        for table_chunk in pl.range(PREFILL_CMP_MAX_BLOCKS // 16):
            received_page = pl.load(tables_window, [3, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_idx_block_table)
        for table_chunk in pl.range(HCA_STATE_MAX_BLOCKS // 16):
            received_page = pl.load(tables_window, [4, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_hca_compress_state_block_table)
        for table_chunk in pl.range(CP_REQUEST_MAIN_TABLE_COLS // 16):
            received_page = pl.load(tables_window, [5, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_csa_compress_state_block_table)
        for table_chunk in pl.range(CP_REQUEST_INNER_TABLE_COLS // 16):
            received_page = pl.load(tables_window, [6, table_chunk * 16], [1, 16])
            pl.store(received_page, [0, table_chunk * 16], local_csa_inner_compress_state_block_table)
    # The only sender has finished publishing before receivers clear this bank.
    for peer in pl.range(CP_SIZE):
        pl.write(ready, [peer, 0], pl.cast(0, pl.INT32))
    return (control, local_x_hc, local_input_ids,
        local_ori_block_table,
        local_hca_cmp_block_table,
        local_csa_cmp_block_table,
        local_idx_block_table,
        local_hca_compress_state_block_table,
        local_csa_compress_state_block_table,
        local_csa_inner_compress_state_block_table,
)






@pl.jit.inline
def _clear_prefill_cp_exchange_signals(
    completion_anchor: pl.Tensor[[1, 1, 8], pl.FP32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Retire one request's exchange credits before retained-window reuse."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_exchange_signal_clear") as clear_tid:
        _completion = pl.read(completion_anchor, [0, 0, 0])
        cp_rank = my_rank % CP_SIZE
        # Every final ready notification was observed before completion. Wait
        # for the final consumed notifications too before overwriting a bank:
        # scalar stores must not race remote atomics on the same cache line.
        for peer in pl.range(CP_SIZE):
            if peer != cp_rank:
                pld.system.wait(signal=consumed, offsets=[peer, 0], expected=completed_epochs, cmp=pld.WaitCmp.Ge)
        for peer in pl.range(CP_SIZE):
            pl.write(ready, [peer, 0], pl.cast(0, pl.INT32))
            pl.write(consumed, [peer, 0], pl.cast(0, pl.INT32))
    return clear_tid


@pl.jit.inline
def _prefill_cp_hidden_tail_exchange_wave(
    local_hidden_tail: pl.Tensor[[EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    hidden_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_hidden_out: pl.Out[pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]:
    """Exchange hidden tails using per-layer ready/consumed epochs."""
    cp_rank = my_rank % CP_SIZE
    cp_group_base = my_rank - cp_rank
    epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.wait(signal=consumed, offsets=[peer, 0], expected=comm_epoch, cmp=pld.WaitCmp.Ge)

    for peer in pl.range(CP_SIZE):
        for part in pl.range(LOCAL_PARTS):
            publish_pos = cp_rank * LOCAL_PARTS + part
            publish_dst_row = publish_pos * TAIL_ROWS
            src_row_base = (payload_epoch * LOCAL_PARTS * TAIL_ROWS + part * TAIL_ROWS)
            pld.tensor.put(
                dst=hidden_window,
                peer=cp_group_base + peer,
                src=local_hidden_tail,
                dst_offsets=[publish_dst_row, 0],
                src_offsets=[src_row_base, 0],
                shape=[TAIL_ROWS, D],
                chunk_rows=ROW_TILE,
                chunk_cols=D,
                pipeline=True,
            )

    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.notify(
                target=ready, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    for seg in pl.range(NUM_SEGMENTS):
        gather_pos = reverse_index[seg]
        owner = owner_rank_table[seg]
        if owner != cp_rank:
            pld.system.wait(signal=ready, offsets=[owner, 0], expected=epoch_value, cmp=pld.WaitCmp.Ge)
        gather_src_row = gather_pos * TAIL_ROWS
        gather_dst_row = (payload_epoch * CP_TAIL_WINDOW_ROWS + seg * TAIL_ROWS)
        for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
            hidden_tile = hidden_window[gather_src_row + t0:gather_src_row + t0 + ROW_TILE, 0:D]
            logical_hidden_out[gather_dst_row + t0:gather_dst_row + t0 + ROW_TILE, 0:D] = hidden_tile

    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.notify(
                target=consumed, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    return logical_hidden_out


@pl.jit.inline
def _prefill_cp_hca_history_exchange(
    raw_cache: pl.Tensor[[HCA_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_cache: pl.Tensor[[CP_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    state: pl.Tensor[[HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    raw_slots: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    cmp_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    state_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    history_raw: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    history_state: pl.Out[pl.Tensor[[TAIL_ROWS, COMPRESS_STATE_DIM], pl.FP32]],
    history_cmp: pl.InOut[pl.Tensor[[HCA_MAX_COMPRESSED_ROWS, HEAD_DIM], pl.BF16]],
    base: pl.Scalar[pl.INT32],
    history_count: pl.Scalar[pl.INT32],
    cache_owner_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Stage owner history without writing other requests' persistent pools.

    One phase transfers the raw/state tail; subsequent phases stream compressed
    rows through the existing compact window. Every phase is consumed before
    the next overwrite, including the later current-chunk compact publication.
    State-window lanes0/1 hold FP32 state/raw respectively (CP is at least2);
    conversion of BF16 raw KV to FP32 and back is exact.
    """
    state_rows = pl.tensor.dim(state, 0) * HCA_STATE_BLOCK_SIZE
    state_flat = pl.reshape(state, [state_rows, COMPRESS_STATE_DIM])
    raw_rows = pl.tensor.dim(raw_cache, 0) * BLOCK_SIZE
    raw_flat = pl.reshape(raw_cache, [raw_rows, HEAD_DIM])
    cmp_rows = pl.tensor.dim(cmp_cache, 0) * HCA_CMP_STORAGE_BLOCK_SIZE
    cmp_flat = pl.reshape(cmp_cache, [cmp_rows, HEAD_DIM])
    state_payload = pl.create_tensor([TAIL_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32)
    raw_payload = pl.create_tensor([TAIL_ROWS, HEAD_DIM], dtype=pl.FP32)
    cmp_payload = pl.create_tensor([CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_owner_history") as history_tid:
        cp_rank = my_rank % CP_SIZE
        group_base = my_rank - cp_rank
        compressed = history_count
        waves = (compressed + CMP_WINDOW_ROWS - 1) // CMP_WINDOW_ROWS
        for phase in pl.range(waves + 1):
            epoch = epoch_base + phase
            for peer in pl.range(CP_SIZE):
                if peer != cp_rank:
                    pld.system.wait(signal=consumed, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
            if my_rank == cache_owner_rank:
                if phase == 0:
                    for row in pl.range(TAIL_ROWS):
                        state_payload[row:row + 1, :] = pl.full([1, COMPRESS_STATE_DIM], dtype=pl.FP32, value=0.0)
                        raw_payload[row:row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        position = base - TAIL_ROWS + row
                        if position >= 0:
                            page = pl.read(state_table, [position // HCA_STATE_BLOCK_SIZE])
                            source = page * HCA_STATE_BLOCK_SIZE + position % HCA_STATE_BLOCK_SIZE
                            if page >= 0 and source < state_rows:
                                state_payload[row:row + 1, :] = state_flat[source:source + 1, :]
                        raw_source = pl.read(raw_slots, [0, row])
                        if raw_source >= 0 and raw_source < raw_rows:
                            raw_payload[row:row + 1, :] = pl.cast(raw_flat[raw_source:raw_source + 1, :], pl.FP32)
                    for peer in pl.range(CP_SIZE):
                        pld.tensor.put(
                            dst=state_window, peer=group_base + peer, src=state_payload,
                            dst_offsets=[0, 0], src_offsets=[0, 0], shape=[TAIL_ROWS, COMPRESS_STATE_DIM],
                            chunk_rows=ROW_TILE, chunk_cols=COMPRESS_STATE_DIM, pipeline=True,
                        )
                        pld.tensor.put(
                            dst=state_window, peer=group_base + peer, src=raw_payload,
                            dst_offsets=[TAIL_ROWS, 0], src_offsets=[0, 0], shape=[TAIL_ROWS, HEAD_DIM],
                            chunk_rows=ROW_TILE, chunk_cols=HEAD_DIM, pipeline=True,
                        )
                else:
                    for row in pl.range(CMP_WINDOW_ROWS):
                        logical = (phase - 1) * CMP_WINDOW_ROWS + row
                        cmp_payload[row:row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        if logical < compressed:
                            page = pl.read(cmp_table, [logical // HCA_CMP_STORAGE_BLOCK_SIZE])
                            source = page * HCA_CMP_STORAGE_BLOCK_SIZE + logical % HCA_CMP_STORAGE_BLOCK_SIZE
                            if page >= 0 and source < cmp_rows:
                                cmp_payload[row:row + 1, :] = cmp_flat[source:source + 1, :]
                    for peer in pl.range(CP_SIZE):
                        pld.tensor.put(
                            dst=cmp_window, peer=group_base + peer, src=cmp_payload,
                            dst_offsets=[0, 0], src_offsets=[0, 0], shape=[CMP_WINDOW_ROWS, HEAD_DIM],
                            chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=HEAD_DIM, pipeline=True,
                        )
            for peer in pl.range(CP_SIZE):
                if peer != cp_rank:
                    pld.system.notify(
                        target=ready, peer=group_base + peer, offsets=[cp_rank, 0],
                        value=1, op=pld.NotifyOp.AtomicAdd,
                    )
            for peer in pl.range(CP_SIZE):
                if peer != cp_rank:
                    pld.system.wait(signal=ready, offsets=[peer, 0], expected=epoch + 1, cmp=pld.WaitCmp.Ge)
            if phase == 0:
                for row0 in pl.range(0, TAIL_ROWS, ROW_TILE):
                    history_state[row0:row0 + ROW_TILE, :] = state_window[row0:row0 + ROW_TILE, :]
                    history_raw[row0:row0 + ROW_TILE, :] = pl.cast(
                        state_window[TAIL_ROWS + row0:TAIL_ROWS + row0 + ROW_TILE, 0:HEAD_DIM], pl.BF16)
            else:
                for row in pl.range(CMP_WINDOW_ROWS):
                    logical = (phase - 1) * CMP_WINDOW_ROWS + row
                    if logical < compressed:
                        history_cmp[logical:logical + 1, :] = cmp_window[row:row + 1, :]
            for peer in pl.range(CP_SIZE):
                if peer != cp_rank:
                    pld.system.notify(
                        target=consumed, peer=group_base + peer, offsets=[cp_rank, 0],
                        value=1, op=pld.NotifyOp.AtomicAdd,
                    )
    return history_tid


@pl.jit.inline
def _prefill_cp_hca_compact_exchange_commit_wave(
    local_cmp_payload: pl.Tensor[[EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], pl.BF16],
    local_cmp_meta: pl.Tensor[[EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], pl.INT32],
    local_state_payload: pl.Tensor[[EPOCHS * HCA_STATE_ROWS_PER_RANK, COMPRESS_STATE_DIM], pl.FP32],
    local_state_meta: pl.Tensor[[EPOCHS * LOCAL_PARTS, STATE_META_DIM], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32],
    state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32],
    state_meta_window: pld.DistributedTensor[[HCA_STATE_RECORDS, STATE_META_DIM], pl.INT32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CP_CMP_ROWS_DYN, HEAD_DIM], pl.BF16]],
    compress_state: pl.InOut[pl.Tensor[[HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    attn_cmp_kv: pl.InOut[pl.Tensor[[HCA_MAX_COMPRESSED_ROWS, HEAD_DIM], pl.BF16]],
    cache_owner_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
    payload_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather HCA attention rows and commit persistent cache/state on the request owner.

    ``payload_epoch`` selects rows in the invocation-local payload tensors
    (``local_cmp_payload``/``local_cmp_meta``/``local_state_payload``/
    ``local_state_meta``); ``comm_epoch`` drives the HCA compact
    ready/consumed counters (``consumed >= comm_epoch``,
    ``ready >= comm_epoch + 1``).
    """
    state_blocks = pl.tensor.dim(compress_state, 0)
    state_rows = state_blocks * HCA_STATE_BLOCK_SIZE
    state_flat = pl.reshape(compress_state, [state_rows, COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_compact_commit", deps=[payload_ready]) as commit_tid:
        cp_rank = my_rank % CP_SIZE
        cp_group_base = my_rank - cp_rank
        epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

        for peer in pl.range(CP_SIZE):
            if peer != cp_rank:
                pld.system.wait(signal=consumed, offsets=[peer, 0], expected=comm_epoch, cmp=pld.WaitCmp.Ge)

        cmp_src_row = payload_epoch * CMP_ROWS_PER_RANK
        state_src_row = payload_epoch * HCA_STATE_ROWS_PER_RANK
        for peer in pl.range(CP_SIZE):
            cmp_dst_row = cp_rank * CMP_ROWS_PER_RANK
            state_dst_row = cp_rank * HCA_STATE_ROWS_PER_RANK
            pld.tensor.put(
                dst=cmp_window, peer=cp_group_base + peer, src=local_cmp_payload,
                dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, HEAD_DIM],
                chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=HEAD_DIM, pipeline=True,
            )
            pld.tensor.put(
                dst=cmp_meta_window, peer=cp_group_base + peer, src=local_cmp_meta,
                dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, CMP_META_DIM],
                chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=CMP_META_DIM, pipeline=True,
            )
            pld.tensor.put(
                dst=state_window, peer=cp_group_base + peer, src=local_state_payload,
                dst_offsets=[state_dst_row, 0], src_offsets=[state_src_row, 0], shape=[HCA_STATE_ROWS_PER_RANK, COMPRESS_STATE_DIM],
                chunk_rows=ROW_TILE, chunk_cols=COMPRESS_STATE_DIM, pipeline=True,
            )
            pld.tensor.put(
                dst=state_meta_window, peer=cp_group_base + peer, src=local_state_meta,
                dst_offsets=[cp_rank * LOCAL_PARTS, 0], src_offsets=[payload_epoch * LOCAL_PARTS, 0], shape=[LOCAL_PARTS, STATE_META_DIM],
                chunk_rows=1, chunk_cols=STATE_META_DIM, pipeline=True,
            )

        for peer in pl.range(CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=ready, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        for peer in pl.range(CP_SIZE):
            if peer != cp_rank:
                pld.system.wait(signal=ready, offsets=[peer, 0], expected=epoch_value, cmp=pld.WaitCmp.Ge)

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
                        if logical_slot >= 0 and logical_slot < HCA_MAX_COMPRESSED_ROWS:
                            cmp_row_tile = cmp_window[cmp_source_row : cmp_source_row + 1, 0:HEAD_DIM]
                            attn_cmp_kv[logical_slot:logical_slot + 1, :] = cmp_row_tile
                            if my_rank == cache_owner_rank:
                                logical_block = pl.cast(logical_slot // HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                                if logical_block < PREFILL_CMP_MAX_BLOCKS:
                                    physical_block = pl.read(cmp_block_table, [logical_block])
                                    if physical_block >= 0 and physical_block < pl.tensor.dim(cmp_kv, 0) // HCA_CMP_STORAGE_BLOCK_SIZE:
                                        intra = pl.cast(logical_slot % HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                                        cache_row = (
                                            pl.cast(physical_block, pl.INDEX)
                                            * HCA_CMP_STORAGE_BLOCK_SIZE
                                            + intra
                                        )
                                        cmp_kv[cache_row : cache_row + 1, 0:HEAD_DIM] = cmp_row_tile

        for state_segment in pl.range(NUM_SEGMENTS):
            state_owner = pl.read(owner_rank_table, [state_segment]) * LOCAL_PARTS + pl.read(owner_part_table, [state_segment])
            state_valid = pl.read(state_meta_window, [state_owner, 0])
            valid_rows = pl.read(state_meta_window, [state_owner, 2])
            end_position = pl.read(state_meta_window, [state_owner, 3])
            if state_valid > 0 and my_rank == cache_owner_rank:
                for row in pl.range(HCA_STATE_ROWS_PER_PART):
                    if row < valid_rows:
                        absolute_position = end_position - valid_rows + row
                        if absolute_position >= 0:
                            if absolute_position < MAX_SEQ_LEN:
                                logical_block = pl.cast(absolute_position // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                                if logical_block < HCA_STATE_MAX_BLOCKS:
                                    physical_block = pl.read(compress_state_block_table, [logical_block])
                                    if physical_block >= 0 and physical_block < state_blocks:
                                        intra = pl.cast(absolute_position % HCA_STATE_BLOCK_SIZE, pl.INDEX)
                                        state_source_row = state_owner * HCA_STATE_ROWS_PER_PART + row
                                        state_row_tile = pl.slice(
                                            state_window,
                                            [1, COMPRESS_STATE_DIM],
                                            [state_source_row, 0],
                                        )
                                        state_row = pl.cast(physical_block, pl.INDEX) * HCA_STATE_BLOCK_SIZE + intra
                                        state_flat[state_row : state_row + 1, 0:COMPRESS_STATE_DIM] = state_row_tile

        for peer in pl.range(CP_SIZE):
            if peer != cp_rank:
                pld.system.notify(
                    target=consumed, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )
        # ``cmp_kv`` is caller-owned InOut storage.  Do not return a scoped SSA
        # alias: PyPTO 0.60 would classify the compact task argument as Out and
        # discard untouched cache rows instead of preserving their input values.
    return commit_tid


@pl.jit.inline
def _prefill_cp_csa_compact_transport_wave(
    main_payload: pl.Tensor[[EPOCHS * ROWS_PER_RANK, MAIN_HEAD_DIM], pl.BF16],
    idx_payload: pl.Tensor[[EPOCHS * ROWS_PER_RANK, INNER_HEAD_DIM], pl.INT8],
    idx_scale: pl.Tensor[[EPOCHS * ROWS_PER_RANK, SCALE_TILE_COLS], pl.FP16],
    record_meta: pl.Tensor[[EPOCHS * ROWS_PER_RANK, META_DIM], pl.INT32],
    main_state_payload: pl.Tensor[[EPOCHS * STATE_ROWS_PER_RANK, MAIN_STATE_DIM], pl.FP32],
    inner_state_payload: pl.Tensor[[EPOCHS * STATE_ROWS_PER_RANK, INNER_STATE_DIM], pl.FP32],
    main_state_meta: pl.Tensor[[EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32],
    inner_state_meta: pl.Tensor[[EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32],
    main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, MAIN_HEAD_DIM], pl.BF16],
    idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, INNER_HEAD_DIM], pl.INT8],
    scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP16],
    record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32],
    main_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32],
    main_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    inner_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32],
    inner_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
):
    cp_rank = my_rank % CP_SIZE
    cp_group_base = my_rank - cp_rank
    comm_i32 = pl.cast(comm_epoch, pl.INT32)
    ready_expected = pl.cast(comm_i32 + 1, pl.INT32)
    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.wait(signal=consumed, offsets=[peer, 0], expected=comm_i32, cmp=pld.WaitCmp.Ge)

    payload_row = payload_epoch * ROWS_PER_RANK
    state_row = payload_epoch * STATE_ROWS_PER_RANK
    destination_row = cp_rank * ROWS_PER_RANK
    destination_state_row = cp_rank * STATE_ROWS_PER_RANK
    for peer in pl.range(CP_SIZE):
        pld.tensor.put(
            dst=main_window, peer=cp_group_base + peer, src=main_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, MAIN_HEAD_DIM],
            chunk_rows=8, chunk_cols=MAIN_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=idx_window, peer=cp_group_base + peer, src=idx_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, INNER_HEAD_DIM],
            chunk_rows=8, chunk_cols=INNER_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=scale_window, peer=cp_group_base + peer, src=idx_scale,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, SCALE_TILE_COLS],
            chunk_rows=8, chunk_cols=SCALE_TILE_COLS, pipeline=True,
        )
        pld.tensor.put(
            dst=record_window, peer=cp_group_base + peer, src=record_meta,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, META_DIM],
            chunk_rows=8, chunk_cols=META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_window, peer=cp_group_base + peer, src=main_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, MAIN_STATE_DIM],
            chunk_rows=4, chunk_cols=MAIN_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_meta_window, peer=cp_group_base + peer, src=main_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_window, peer=cp_group_base + peer, src=inner_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, INNER_STATE_DIM],
            chunk_rows=4, chunk_cols=INNER_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_meta_window, peer=cp_group_base + peer, src=inner_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )

    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.notify(
                target=ready, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.wait(signal=ready, offsets=[peer, 0], expected=ready_expected, cmp=pld.WaitCmp.Ge)


@pl.jit.inline
def _prefill_cp_csa_compact_finish_wave(
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    cp_rank = my_rank % CP_SIZE
    cp_group_base = my_rank - cp_rank
    for peer in pl.range(CP_SIZE):
        if peer != cp_rank:
            pld.system.notify(
                target=consumed, peer=cp_group_base + peer, offsets=[cp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )


@pl.jit.inline
def _prefill_cp_gather_hidden(
    control: pl.Tensor[[1, 16], pl.INT32],
    local_hidden: pl.Tensor[[LOCAL_ROWS, D], pl.BF16],
    local_pre_hc_hidden: pl.Tensor[[LOCAL_ROWS, CP_REQUEST_HC_DIM], pl.FP32],
    hidden_window: pld.DistributedTensor[[CP_REQUEST_CAPACITY, D], pl.BF16],
    pre_hc_window: pld.DistributedTensor[[CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], pl.FP32],
    complete: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hidden_out: pl.Tensor[[CP_REQUEST_TOKENS_DYN, D], pl.BF16],
    pre_hc_hidden_out: pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    my_rank: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gather_hidden"):
        # Other owners' outputs survive subsequent requests in the same call.
        if pl.read(control, [0, 1]) == my_rank:
            output_rows = pl.tensor.dim(hidden_out, 0)
            for row_start in pl.range(0, output_rows, ROW_TILE):
                valid_rows = pl.min(ROW_TILE, output_rows - row_start)
                for column in pl.range(D // CP_REQUEST_COPY_COLS):
                    zero_hidden = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.BF16)
                    zero_valid = pl.tile.set_validshape(zero_hidden, valid_rows, CP_REQUEST_COPY_COLS)
                    pl.store(zero_valid, [row_start, column * CP_REQUEST_COPY_COLS], hidden_out)
            for row_start in pl.range(0, output_rows, ROW_TILE):
                valid_rows = pl.min(ROW_TILE, output_rows - row_start)
                for column in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                    zero_pre_hc = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.FP32)
                    zero_pre_hc_valid = pl.tile.set_validshape(zero_pre_hc, valid_rows, CP_REQUEST_COPY_COLS)
                    pl.store(zero_pre_hc_valid, [row_start, column * CP_REQUEST_COPY_COLS], pre_hc_hidden_out)
        if pl.read(control, [0, 0]) == 1:
            owner = pl.read(control, [0, 1])
            length = pl.read(control, [0, 2])
            span = pl.read(control, [0, 3])
            # Rank-major slabs keep padded segments disjoint for arbitrary spans.
            pld.tensor.put(
                dst=hidden_window, peer=owner, src=local_hidden,
                dst_offsets=[my_rank * LOCAL_ROWS, 0], src_offsets=[0, 0],
                shape=[LOCAL_ROWS, D], chunk_rows=ROW_TILE, chunk_cols=CP_REQUEST_COPY_COLS,
            )
            pld.tensor.put(
                dst=pre_hc_window, peer=owner, src=local_pre_hc_hidden,
                dst_offsets=[my_rank * LOCAL_ROWS, 0], src_offsets=[0, 0],
                shape=[LOCAL_ROWS, CP_REQUEST_HC_DIM], chunk_rows=ROW_TILE, chunk_cols=CP_REQUEST_COPY_COLS,
            )
            if my_rank != owner:
                pld.system.notify(target=complete, peer=owner, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
            else:
                for peer in pl.range(CP_SIZE):
                    if peer != my_rank:
                        pld.system.wait(signal=complete, offsets=[peer, 0], expected=1, cmp=pld.WaitCmp.Ge)
                for segment in pl.range(NUM_SEGMENTS):
                    segment_start = segment * span
                    segment_length = pl.max(0, pl.min(span, length - segment_start))
                    if segment < CP_SIZE:
                        source_rank = segment
                        source_part = pl.cast(0, pl.INDEX)
                    else:
                        source_rank = NUM_SEGMENTS - 1 - segment
                        source_part = pl.cast(1, pl.INDEX)
                    for local_row in pl.range(0, segment_length, ROW_TILE):
                        active = pl.min(ROW_TILE, segment_length - local_row)
                        source_base = source_rank * LOCAL_ROWS + source_part * MAX_SEGMENT_TILES * TAIL_ROWS + local_row
                        for column in pl.range(D // CP_REQUEST_COPY_COLS):
                            restored = pl.load(
                                hidden_window, [source_base, column * CP_REQUEST_COPY_COLS],
                                [ROW_TILE, CP_REQUEST_COPY_COLS], valid_shape=[active, CP_REQUEST_COPY_COLS],
                            )
                            pl.store(restored, [segment_start + local_row, column * CP_REQUEST_COPY_COLS], hidden_out)
                        for hc_column in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                            restored_pre_hc = pl.load(
                                pre_hc_window, [source_base, hc_column * CP_REQUEST_COPY_COLS],
                                [ROW_TILE, CP_REQUEST_COPY_COLS], valid_shape=[active, CP_REQUEST_COPY_COLS],
                            )
                            pl.store(restored_pre_hc, [segment_start + local_row, hc_column * CP_REQUEST_COPY_COLS], pre_hc_hidden_out)
        for peer in pl.range(CP_SIZE):
            pl.write(complete, [peer, 0], pl.cast(0, pl.INT32))
    return hidden_out, pre_hc_hidden_out


@pl.jit.inline
def _prefill_cp_raw_cache_commit(
    local_kv: pl.Tensor[[LOCAL_ROWS, HEAD_DIM], pl.BF16],
    slots: pl.Tensor,
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_cache: pl.Tensor,
    cache_owner: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
    attention_done: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Persist every current KV row after all old-cache readers finish.

    Reuse the retired hidden-tail window and its credit protocol. One segment's
    KV fits inside its hidden-tail slot because HEAD_DIM is smaller than D.
    """
    packed = pl.create_tensor([LOCAL_PARTS * TAIL_ROWS, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_raw_cache_pack", deps=[attention_done]):
        for part in pl.range(LOCAL_PARTS):
            for row in pl.range(0, TAIL_ROWS, ROW_TILE):
                packed[part * TAIL_ROWS + row:part * TAIL_ROWS + row + ROW_TILE, :] = pl.full(
                    [ROW_TILE, D], dtype=pl.BF16, value=0.0)
            for row in pl.range(MAX_SEGMENT_TILES * TAIL_ROWS * HEAD_DIM // D):
                source = pl.slice(local_kv, [D // HEAD_DIM, HEAD_DIM],
                                  [part * MAX_SEGMENT_TILES * TAIL_ROWS + row * D // HEAD_DIM, 0])
                packed_source = pl.reshape(source, [1, D])
                packed[part * TAIL_ROWS + row:part * TAIL_ROWS + row + 1, :] = packed_source


    gathered = pl.create_tensor([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_raw_cache_exchange"):
        _prefill_cp_hidden_tail_exchange_wave(
            packed, reverse_index, owner_rank_table, window, ready, consumed,
            gathered, my_rank, pl.cast(0, pl.INT32), comm_epoch,
        )
    gathered_kv = pl.reshape(gathered, [CP_TAIL_WINDOW_ROWS * D // HEAD_DIM, HEAD_DIM])
    raw_rows = pl.tensor.dim(kv_cache, 0) * BLOCK_SIZE
    raw = pl.reshape(kv_cache, [raw_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_raw_cache_commit") as commit_tid:
        if my_rank == cache_owner:
            for segment in pl.range(NUM_SEGMENTS):
                for row in pl.range(MAX_SEGMENT_TILES * TAIL_ROWS):
                    destination = pl.read(slots, [TAIL_ROWS + segment * MAX_SEGMENT_TILES * TAIL_ROWS + row])
                    if destination >= 0 and destination < raw_rows:
                        source_row = segment * TAIL_ROWS * D // HEAD_DIM + row
                        raw[destination:destination + 1, :] = gathered_kv[source_row:source_row + 1, :]
    return commit_tid


@pl.jit.inline
def _prefill_cp_request_barrier(
    hidden_out: pl.Tensor[[CP_REQUEST_TOKENS_DYN, D], pl.BF16],
    pre_hc_hidden_out: pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    request_barrier_epochs: pld.DistributedTensor[[CP_SIZE, 16], pl.INT32],
    cp_tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_hca_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_hca_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_csa_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_csa_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_count_signal: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_prefill_moe_x_signal: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_prefill_moe_reverse_signal: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    entry_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    # Wait for all CP ranks to finish this request chunk before window reuse.
    # Each rank owns one cache-line-separated epoch counter. Credits remain
    # monotonic across owners and repeated calls; no reset races the next call.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_request_barrier", allow_early_resolve=False):
        _hidden_anchor = pl.read(hidden_out, [0, 0])
        _tail_anchor = pl.read(pre_hc_hidden_out, [0, 0])
        _cp_tail_ready_anchor = pl.read(cp_tail_ready, [0, 0])
        _cp_tail_consumed_anchor = pl.read(cp_tail_consumed, [0, 0])
        _cp_hca_compact_ready_anchor = pl.read(cp_hca_compact_ready, [0, 0])
        _cp_hca_compact_consumed_anchor = pl.read(cp_hca_compact_consumed, [0, 0])
        _cp_csa_compact_ready_anchor = pl.read(cp_csa_compact_ready, [0, 0])
        _cp_csa_compact_consumed_anchor = pl.read(cp_csa_compact_consumed, [0, 0])
        _cp_count_signal_anchor = pl.read(cp_count_signal, [0, 0])
        _cp_prefill_moe_x_signal_anchor = pl.read(cp_prefill_moe_x_signal, [0, 0])
        _cp_prefill_moe_reverse_signal_anchor = pl.read(cp_prefill_moe_reverse_signal, [0, 0])
        _entry_ready_anchor = pl.read(entry_ready, [0, 0])
        epoch = pl.cast(pl.read(request_barrier_epochs, [my_rank, 0]) + 1, pl.INT32)
        pl.write(request_barrier_epochs, [my_rank, 0], epoch)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(target=request_barrier_epochs, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(signal=request_barrier_epochs, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    return request_barrier_epochs


# ---------------------------------------------------------------------------
# Fixtures and golden validation.
# ---------------------------------------------------------------------------
REQUEST_TEST_TABLES = (
    "ori_block_table",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
)
REQUEST_TEST_TABLE_COLS = (
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    HCA_STATE_MAX_BLOCKS,
    CP_REQUEST_MAIN_TABLE_COLS,
    CP_REQUEST_INNER_TABLE_COLS,
)


@pl.jit(auto_scope=False)
def _prefill_cp_request_test(
    num_tokens_per_owner: pl.Tensor[[CP_SIZE], pl.INT32],
    position_ids: pl.Tensor[[CP_REQUEST_TOKENS_DYN], pl.INT32],
    x_hc: pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    input_ids: pl.Tensor[[1, CP_REQUEST_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[1, PREFILL_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[1, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32],
    header_window: pld.DistributedTensor[[1, 16], pl.INT32],
    input_window: pld.DistributedTensor[[LOCAL_ROWS, CP_REQUEST_HC_DIM], pl.FP32],
    ids_window: pld.DistributedTensor[[1, LOCAL_ROWS * 2], pl.INT32],
    tables_window: pld.DistributedTensor[[7, CP_REQUEST_TABLE_COLS], pl.INT32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hidden_window: pld.DistributedTensor[[CP_REQUEST_CAPACITY, D], pl.BF16],
    tail_window: pld.DistributedTensor[[CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], pl.FP32],
    complete: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    request_barrier_epochs: pld.DistributedTensor[[CP_SIZE, 16], pl.INT32],
    hidden_out: pl.InOut[pl.Tensor[[CP_REQUEST_TOKENS_DYN, D], pl.BF16]],
    tail_out: pl.InOut[pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32]],
    audit: pl.InOut[pl.Tensor[[CP_SIZE, 9, CP_REQUEST_TABLE_COLS], pl.INT32]],
    request_owner: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    header = pl.create_tensor([1, 16], dtype=pl.INT32)
    if request_owner == 0:
        for row in pl.spmd(CP_SIZE * 9):
            for col in pl.range(CP_REQUEST_TABLE_COLS // 128):
                zeros = pl.tile.full([1, 1, 128], value=0, dtype=pl.INT32)
                pl.store(zeros, [row // 9, row % 9, col * 128], audit)
        for row in pl.spmd(CP_REQUEST_CAPACITY // ROW_TILE):
            for col in pl.range(D // CP_REQUEST_COPY_COLS):
                zero = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.BF16)
                pl.store(zero, [row * ROW_TILE, col * CP_REQUEST_COPY_COLS], hidden_out)
        for row in pl.spmd(CP_REQUEST_CAPACITY // ROW_TILE):
            for col in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                zero_tail = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.FP32)
                pl.store(zero_tail, [row * ROW_TILE, col * CP_REQUEST_COPY_COLS], tail_out)
    if pl.read(num_tokens_per_owner, [request_owner]) > 0:
        control = pl.create_tensor([1, 16], dtype=pl.INT32)
        local_x_hc = pl.create_tensor([LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
        local_input_ids = pl.create_tensor([1, LOCAL_ROWS], dtype=pl.INT64)
        local_ori_block_table = pl.create_tensor([1, PREFILL_ORI_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_cmp_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_cmp_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_idx_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_compress_state_block_table = pl.create_tensor([1, HCA_STATE_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_compress_state_block_table = pl.create_tensor([1, CP_REQUEST_MAIN_TABLE_COLS], dtype=pl.INT32)
        local_csa_inner_compress_state_block_table = pl.create_tensor([1, CP_REQUEST_INNER_TABLE_COLS], dtype=pl.INT32)
        _prefill_cp_request_header(num_tokens_per_owner, position_ids, header, request_owner, my_rank)
        (
            control,
            local_x_hc,
            local_input_ids,
            local_ori_block_table,
            local_hca_cmp_block_table,
            local_csa_cmp_block_table,
            local_idx_block_table,
            local_hca_compress_state_block_table,
            local_csa_compress_state_block_table,
            local_csa_inner_compress_state_block_table,
        ) = _prefill_cp_scatter_request(
            header, x_hc, input_ids,
            ori_block_table, hca_cmp_block_table, csa_cmp_block_table, idx_block_table,
            hca_compress_state_block_table, csa_compress_state_block_table, csa_inner_compress_state_block_table,
            header_window, input_window, ids_window, tables_window, ready,
            control, local_x_hc, local_input_ids,
            local_ori_block_table, local_hca_cmp_block_table, local_csa_cmp_block_table, local_idx_block_table,
            local_hca_compress_state_block_table, local_csa_compress_state_block_table, local_csa_inner_compress_state_block_table,
            my_rank,
        )
        ids_flat = pl.reshape(local_input_ids, [1, 1, LOCAL_ROWS])
        metadata = pl.reshape(control, [1, 1, 16])
        table_2 = pl.reshape(local_ori_block_table, [1, 1, PREFILL_ORI_MAX_BLOCKS])
        table_3 = pl.reshape(local_hca_cmp_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_4 = pl.reshape(local_csa_cmp_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_5 = pl.reshape(local_idx_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_6 = pl.reshape(local_hca_compress_state_block_table, [1, 1, HCA_STATE_MAX_BLOCKS])
        table_7 = pl.reshape(local_csa_compress_state_block_table, [1, 1, CP_REQUEST_MAIN_TABLE_COLS])
        table_8 = pl.reshape(local_csa_inner_compress_state_block_table, [1, 1, CP_REQUEST_INNER_TABLE_COLS])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="capture_request_metadata"):
            for col in pl.range(LOCAL_ROWS * 2 // 128):
                ids = pl.load(ids_flat, [0, 0, col * 64], [1, 1, 64])
                ids_words = pl.reinterpret_view(ids, pl.INT32)
                pl.store(ids_words, [request_owner, 0, col * 128], audit)
            control_values = pl.load(metadata, [0, 0, 0], [1, 1, 16])
            pl.store(control_values, [request_owner, 1, 0], audit)
            for col in pl.range(PREFILL_ORI_MAX_BLOCKS // 128):
                values_2 = pl.load(table_2, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_2, [request_owner, 2, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_3 = pl.load(table_3, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_3, [request_owner, 3, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_4 = pl.load(table_4, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_4, [request_owner, 4, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_5 = pl.load(table_5, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_5, [request_owner, 5, col * 128], audit)
            for col in pl.range(HCA_STATE_MAX_BLOCKS // 128):
                values_6 = pl.load(table_6, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_6, [request_owner, 6, col * 128], audit)
            for col in pl.range(CP_REQUEST_MAIN_TABLE_COLS // 128):
                values_7 = pl.load(table_7, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_7, [request_owner, 7, col * 128], audit)
            for col in pl.range(CP_REQUEST_INNER_TABLE_COLS // 128):
                values_8 = pl.load(table_8, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_8, [request_owner, 8, col * 128], audit)
        local_hidden = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
        for block in pl.spmd(LOCAL_ROWS // ROW_TILE):
            for col in pl.range(D // CP_REQUEST_COPY_COLS):
                values = pl.load(local_x_hc, [block * ROW_TILE, col * CP_REQUEST_COPY_COLS], [ROW_TILE, CP_REQUEST_COPY_COLS])
                rounded = pl.cast(values, pl.BF16, mode="rint")
                pl.store(rounded, [block * ROW_TILE, col * CP_REQUEST_COPY_COLS], local_hidden)
        _prefill_cp_gather_hidden(
            control, local_hidden, local_x_hc,
            hidden_window, tail_window, complete,
            hidden_out, tail_out, my_rank,
        )
    _prefill_cp_request_barrier(
        hidden_out, tail_out, request_barrier_epochs,
        ready, ready, ready, ready, ready, ready, ready, ready, ready, ready,
        my_rank,
    )
    return hidden_out, tail_out


@pl.jit.host
def prefill_cp_exchange_test(
    num_tokens_per_owner: pl.Tensor[[CP_SIZE], pl.INT32],
    position_ids: pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN], pl.INT32],
    x_hc: pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    input_ids: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN, D], pl.BF16]],
    tail_out: pl.Out[pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32]],
    audit: pl.Out[pl.Tensor[[CP_SIZE, CP_SIZE, 9, CP_REQUEST_TABLE_COLS], pl.INT32]],
):
    hidden_out.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    tail_out.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    x_hc.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    input_ids.bind_dynamic(2, CP_REQUEST_TOKENS_DYN)
    position_ids.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    header_window_buf = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    input_window_buf = pld.alloc_window_buffer([LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
    ids_window_buf = pld.alloc_window_buffer([1, LOCAL_ROWS * 2], dtype=pl.INT32)
    tables_window_buf = pld.alloc_window_buffer([7, CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hidden_window_buf = pld.alloc_window_buffer([CP_REQUEST_CAPACITY, D], dtype=pl.BF16)
    tail_window_buf = pld.alloc_window_buffer([CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], dtype=pl.FP32)
    complete_buf = pld.alloc_window_buffer([CP_SIZE, 16], dtype=pl.INT32)
    request_barrier_epochs_buf = pld.alloc_window_buffer([CP_SIZE, 16], dtype=pl.INT32)
    for repetition in pl.range(3):
        for request_owner in pl.range(CP_SIZE):
            for rank in pl.range(CP_SIZE):
                hidden_window = pld.window(hidden_window_buf, [CP_REQUEST_CAPACITY, D], dtype=pl.BF16)
                tail_window = pld.window(tail_window_buf, [CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], dtype=pl.FP32)
                complete = pld.window(complete_buf, [CP_SIZE, 1], dtype=pl.INT32)
                request_barrier_epochs = pld.window(request_barrier_epochs_buf, [CP_SIZE, 16], dtype=pl.INT32)
                header_window = pld.window(header_window_buf, [1, 16], dtype=pl.INT32)
                input_window = pld.window(input_window_buf, [LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
                ids_window = pld.window(ids_window_buf, [1, LOCAL_ROWS * 2], dtype=pl.INT32)
                tables_window = pld.window(tables_window_buf, [7, CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
                ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
                _prefill_cp_request_test(
                    num_tokens_per_owner, position_ids[rank], x_hc[rank], input_ids[rank],
                    ori_block_table[rank], hca_cmp_block_table[rank], csa_cmp_block_table[rank], idx_block_table[rank],
                    hca_compress_state_block_table[rank], csa_compress_state_block_table[rank], csa_inner_compress_state_block_table[rank],
                    header_window, input_window, ids_window, tables_window, ready,
                    hidden_window, tail_window, complete, request_barrier_epochs,
                    hidden_out[rank], tail_out[rank], audit[rank],
                    request_owner, rank,
                    device=rank,
                )


def golden_prefill_cp_exchange(tensors):
    import torch

    counts = tensors["num_tokens_per_owner"].tolist()
    owners = [rank for rank, count in enumerate(counts) if count > 0]
    tensors["hidden_out"].zero_()
    tensors["tail_out"].zero_()
    for owner in owners:
        length = counts[owner]
        tensors["hidden_out"][owner, :length].copy_(tensors["x_hc"][owner, :length, :D].to(torch.bfloat16))
        tensors["tail_out"][owner, :length].copy_(tensors["x_hc"][owner, :length])
    tensors["audit"].zero_()
    for owner in owners:
        length = counts[owner]
        span = max(TAIL_ROWS, (length + NUM_SEGMENTS - 1) // NUM_SEGMENTS)
        starts = [segment * span for segment in range(NUM_SEGMENTS)]
        lengths = [max(0, min(span, length - start)) for start in starts]
        for rank in range(CP_SIZE):
            out = tensors["audit"][rank, owner]
            out[1, :5] = torch.tensor([1, owner, length, span, tensors["position_ids"][owner, 0].item()])
            for part, segment in enumerate((rank, NUM_SEGMENTS - 1 - rank)):
                active = lengths[segment]
                start = starts[segment]
                dest = part * MAX_SEGMENT_TILES * TAIL_ROWS
                out[0, dest * 2 : (dest + active) * 2].copy_(
                    tensors["input_ids"][owner, 0, start : start + active].view(torch.int32)
                )
            for index, name in enumerate(REQUEST_TEST_TABLES, 2):
                value = tensors[name][owner, 0]
                out[index, : value.numel()].copy_(value)


def build_tensor_specs(counts, start_pos):
    import torch
    from golden import TensorSpec

    positions = torch.arange(CP_REQUEST_CAPACITY, dtype=torch.int32).unsqueeze(0)
    positions = positions + torch.arange(CP_SIZE, dtype=torch.int32).unsqueeze(1) * 128 + start_pos
    ids = (2**40 + torch.arange(CP_SIZE * CP_REQUEST_CAPACITY, dtype=torch.int64)).reshape(CP_SIZE, 1, CP_REQUEST_CAPACITY)
    rows = torch.arange(CP_SIZE * CP_REQUEST_CAPACITY, dtype=torch.float32).reshape(CP_SIZE, CP_REQUEST_CAPACITY, 1) * 0.01
    columns = torch.arange(CP_REQUEST_HC_DIM, dtype=torch.float32).reshape(1, 1, -1) * 0.0001
    specs = [
        TensorSpec("num_tokens_per_owner", [CP_SIZE], torch.int32, init_value=counts),
        TensorSpec("position_ids", [CP_SIZE, CP_REQUEST_CAPACITY], torch.int32, init_value=positions),
        TensorSpec("x_hc", [CP_SIZE, CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], torch.float32, init_value=rows + columns),
        TensorSpec("input_ids", [CP_SIZE, 1, CP_REQUEST_CAPACITY], torch.int64, init_value=ids),
    ]
    for name, size in zip(REQUEST_TEST_TABLES, REQUEST_TEST_TABLE_COLS):
        table = torch.stack([torch.arange(size, dtype=torch.int32).roll(rank + 3) for rank in range(CP_SIZE)])
        table[:, 1] = -1
        specs.append(TensorSpec(name, [CP_SIZE, 1, size], torch.int32, init_value=table.unsqueeze(1)))
    specs.extend(
        [
            TensorSpec("hidden_out", [CP_SIZE, CP_REQUEST_CAPACITY, D], torch.bfloat16),
            TensorSpec("tail_out", [CP_SIZE, CP_REQUEST_CAPACITY, CP_REQUEST_HC_DIM], torch.float32),
            TensorSpec("audit", [CP_SIZE, CP_SIZE, 9, CP_REQUEST_TABLE_COLS], torch.int32),
        ]
    )
    return specs


def compare_exact(actual, expected, **_kwargs):
    import torch

    return torch.equal(actual, expected), "Exact request and cache-owner isolation"


def main():
    import argparse

    import torch
    from pypto.ir import DistributedConfig
    from golden import run

    parser = argparse.ArgumentParser(description="Exact CP request transfer, owner output isolation and communication reuse tests.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE)
    parser.add_argument("--ep", type=int, default=CP_SIZE)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != CP_SIZE or args.cp != CP_SIZE or args.ep != CP_SIZE:
        parser.error("The device count and CP/EP group sizes must agree.")
    torch.set_num_threads(4)
    cases = [
        (0, "none", 0),
        (129, "last", 0),
        (129, "all", 0),
        (50, "all", 1024),
        (257, "ends", 1152),
        (CP_REQUEST_CAPACITY, "first", 0),
        (178, "all", 1024),
    ]
    runtime_dir = None
    for length, owners, start_pos in cases:
        counts = torch.zeros(CP_SIZE, dtype=torch.int32)
        if owners == "none":
            pass
        elif owners == "all":
            counts[:] = torch.tensor([length + rank * 17 for rank in range(CP_SIZE)], dtype=torch.int32)
        elif owners == "last":
            counts[-1] = length
        elif owners == "first":
            counts[0] = length
        else:
            counts[0], counts[-1] = length, 50
        print(f"CP request case: counts={counts.tolist()}, base={start_pos}", flush=True)
        result = run(
            fn=prefill_cp_exchange_test,
            specs=build_tensor_specs(counts, start_pos),
            golden_fn=golden_prefill_cp_exchange,
            compare_fn={name: compare_exact for name in ("hidden_out", "tail_out", "audit")},
            compile_only=args.compile_only,
            runtime_dir=runtime_dir,
            save_data=False,
            config=dict(platform=args.platform, distributed_config=DistributedConfig(device_ids=devices)),
        )
        if not result.passed:
            raise RuntimeError(result.error)
        if args.compile_only:
            return
        runtime_dir = str(result.work_dir)
    print("CP request isolation and repeated-window validation PASS", flush=True)


if __name__ == "__main__":
    main()
