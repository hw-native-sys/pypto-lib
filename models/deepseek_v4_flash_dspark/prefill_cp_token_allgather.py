# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Context-parallel prefill token-row all-gather into rank-major order on every rank."""

from math import gcd

import sys

# runtime configuration
_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2


def _parse_int_argv(flag: str, default: int) -> int:
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


TP_SIZE = _parse_int_argv("--tp", _TP_DEFAULT)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M, MOE_TOKENS, DECODE_SEQ


# Dynamic shape variables.
CP_Q_T_DYN = pl.dynamic("PREFILL_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_CP_KV_T_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HEAD_DIM = M.head_dim

# communication bounds
PREFILL_GROUP_CAP = 8192
PREFILL_LOCAL_CAP = PREFILL_GROUP_CAP // TP_SIZE

# tiling
COMM_ROW_TILE = 8
READBACK_ROW_TILE = gcd(MOE_TOKENS // DECODE_SEQ * (DECODE_SEQ - 1), 16)
READBACK_WORKERS = 16
# Raw BF16 words per coefficient row. Round post to a 32-byte DMA block.
COEFF_COMB_WORDS = HC_MULT * HC_MULT * 2
COEFF_POST_WORDS = ((HC_MULT * 2 + 15) // 16) * 16
COEFF_PAYLOAD_WORDS = COEFF_COMB_WORDS + COEFF_POST_WORDS

# Packed HCA projection columns in the BF16 communication window.
HCA_PROJECTION_BITS_DIM = 2 * HEAD_DIM
HCA_PROJECTION_VALUES_OFFSET = HEAD_DIM
HCA_PROJECTION_SCORES_OFFSET = HCA_PROJECTION_VALUES_OFFSET + HCA_PROJECTION_BITS_DIM
HCA_PROJECTION_PADDING_OFFSET = HCA_PROJECTION_SCORES_OFFSET + HCA_PROJECTION_BITS_DIM
HCA_PROJECTION_PADDING_COLS = 16
HCA_PROJECTION_PAYLOAD_DIM = HCA_PROJECTION_PADDING_OFFSET + HCA_PROJECTION_PADDING_COLS

# fixture
FIXTURE_ROUNDS = 2
FIXTURE_LOCAL_T = PREFILL_LOCAL_CAP // (HC_MULT * 2) + 1


@pl.jit.inline
def prefill_cp_token_allgather_into(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    hidden_full: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather rank-major rows and retire the complete two-phase signal epoch."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    hidden_full.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t

    # Each peer has an independent destination and arrival signal.
    with pl.spmd(
        TP_SIZE, name_hint="prefill_cp_token_allgather_push",
        deps=[input_ready_tid], allow_early_resolve=True,
    ) as _push_tid:
        peer_tp = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer_tp,
            src=hidden_local,
            dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D,
            pipeline=True,
        )
        if peer_tp != tp_rank:
            pld.system.notify(
                target=gather_signal, peer=group_base + peer_tp,
                offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )

    # Register the peer payload conditions as deferred completion.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_cp_token_allgather_payload_wait", deps=[input_ready_tid],
    ) as _payload_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Fixed readers own strided row tiles; release the window after every reader.
    group_rows = TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    with pl.spmd(
        READBACK_WORKERS,
        name_hint="prefill_cp_token_allgather_readback",
        deps=[_push_tid, _payload_wait_tid],
    ) as _readback_tid:
        worker = pl.tile.get_block_idx()
        for tile_row in pl.range(
            worker * READBACK_ROW_TILE,
            full_rows,
            READBACK_WORKERS * READBACK_ROW_TILE,
        ):
            window_tile = gather_window[tile_row : tile_row + READBACK_ROW_TILE, 0:D]
            hidden_full[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = window_tile
        for tail_row in pl.range(full_rows + worker, group_rows, READBACK_WORKERS):
            window_row = gather_window[tail_row : tail_row + 1, 0:D]
            hidden_full[tail_row : tail_row + 1, 0:D] = window_row
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_notify",
        deps=[_readback_tid],
    ) as _readback_notify_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_cp_token_allgather_readback_wait", deps=[input_ready_tid],
    ) as _readback_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(2, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Retire peer credits and anchor raw output words to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_retire",
        deps=[_readback_notify_tid, _readback_wait_tid],
    ) as retire_tid:
        anchor_bits = pl.load(hidden_full, [0, 0], [1, 16])
        anchor_words = pl.reinterpret_view(anchor_bits, pl.INT32)
        completion_anchor = pl.tile.read(anchor_words, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + tp_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
        pl.tile.write(anchor_words, [0, 0], completion_anchor)
        anchor_output = pl.reinterpret_view(anchor_words, pl.BF16)
        pl.store(anchor_output, [0, 0], hidden_full)

    return retire_tid


def _prefill_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather BF16 rows through the shared retained-window protocol."""
    input_ready_tid = pl.system.task_dummy(deps=[])
    retire_tid = prefill_cp_token_allgather_into(
        hidden_local, group_out, gather_window, gather_signal, group_base, tp_rank, input_ready_tid,
    )

    return group_out, gather_signal


prefill_cp_token_allgather_step = pl.jit.inline(_prefill_cp_token_allgather_step)


@pl.jit.inline(auto_scope=False)
def _prefill_cp_projection_publish(
    payload_local: pl.Tensor,
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    local_rows: pl.Scalar[pl.INDEX],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
    PAYLOAD_DIM: pl.constexpr,
    PUSH_HINT: pl.constexpr,
    ARRIVAL_HINT: pl.constexpr,
) -> tuple[pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    """Publish packed rows and wait for remote payload arrivals."""
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t
    with pl.spmd(TP_SIZE, name_hint=PUSH_HINT, deps=[input_ready_tid], allow_early_resolve=True) as push_tid:
        peer_tp = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer_tp, src=payload_local,
            dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[local_t, PAYLOAD_DIM],
            chunk_rows=COMM_ROW_TILE, chunk_cols=PAYLOAD_DIM, pipeline=True,
        )
        if peer_tp != tp_rank:
            pld.system.notify(target=gather_signal, peer=group_base + peer_tp, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint=ARRIVAL_HINT, deps=[input_ready_tid],
    ) as payload_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(signal=gather_signal, offsets=[source_tp, 0], expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge)

    return push_tid, payload_wait_tid


@pl.jit.inline(auto_scope=False)
def _prefill_cp_projection_retire(
    completion_tensor: pl.Tensor,
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    readback_tid: pl.Scalar[pl.TASK_ID],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
    ANCHOR_COL: pl.constexpr,
    NOTIFY_HINT: pl.constexpr,
    WAIT_HINT: pl.constexpr,
    RETIRE_HINT: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    """Acknowledge readback and retire the packed-row signal epoch."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint=NOTIFY_HINT, deps=[readback_tid]) as readback_notify_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint=WAIT_HINT, deps=[input_ready_tid],
    ) as readback_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(signal=gather_signal, offsets=[source_tp, 0], expected=pl.cast(2, pl.INT32), cmp=pld.WaitCmp.Ge)

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint=RETIRE_HINT,
        deps=[readback_notify_tid, readback_wait_tid],
    ) as retire_tid:
        anchor_bits = pl.load(completion_tensor, [0, ANCHOR_COL], [1, 16])
        anchor_words = pl.reinterpret_view(anchor_bits, pl.INT32)
        completion_anchor = pl.tile.read(anchor_words, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + tp_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
        pl.tile.write(anchor_words, [0, 0], completion_anchor)
        anchor_output = pl.reinterpret_view(anchor_words, completion_tensor.dtype)
        pl.store(anchor_output, [0, ANCHOR_COL], completion_tensor)

    return retire_tid


@pl.jit.inline
def prefill_cp_hca_projection_allgather_step(
    kv_local: pl.Tensor[[CP_Q_T_DYN, HEAD_DIM], pl.BF16],
    cmp_values_local: pl.Tensor,
    cmp_scores_local: pl.Tensor,
    kv_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HEAD_DIM], pl.BF16]],
    cmp_values_full: pl.Out[pl.Tensor],
    cmp_scores_full: pl.Out[pl.Tensor],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather KV and lossless FP32 compressor projections in one epoch."""
    kv_local.bind_dynamic(0, CP_Q_T_DYN)
    kv_full.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(kv_local, 0)
    payload_local = pl.create_tensor([local_rows, HCA_PROJECTION_PAYLOAD_DIM], dtype=pl.BF16)
    with pl.spmd((local_rows + COMM_ROW_TILE - 1) // COMM_ROW_TILE, name_hint="prefill_cp_hca_projection_pack") as pack_tid:
        row = pl.tile.get_block_idx() * COMM_ROW_TILE
        rows = pl.min(COMM_ROW_TILE, local_rows - row)
        kv_tile = pl.load(kv_local, [row, 0], [COMM_ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(kv_tile, [row, 0], payload_local)
        values_tile = pl.load(cmp_values_local, [row, 0], [COMM_ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        values_bits = pl.reinterpret_view(values_tile, pl.BF16)
        pl.store(values_bits, [row, HCA_PROJECTION_VALUES_OFFSET], payload_local)
        scores_tile = pl.load(cmp_scores_local, [row, 0], [COMM_ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        scores_bits = pl.reinterpret_view(scores_tile, pl.BF16)
        pl.store(scores_bits, [row, HCA_PROJECTION_SCORES_OFFSET], payload_local)
        padding = pl.tile.full([COMM_ROW_TILE, HCA_PROJECTION_PADDING_COLS], dtype=pl.BF16, value=0.0)
        padding_valid = pl.set_validshape(padding, rows, HCA_PROJECTION_PADDING_COLS)
        pl.store(padding_valid, [row, HCA_PROJECTION_PADDING_OFFSET], payload_local)

    push_tid, payload_wait_tid = _prefill_cp_projection_publish(
        payload_local, gather_window, gather_signal,
        local_rows, group_base, tp_rank, pack_tid, HCA_PROJECTION_PAYLOAD_DIM,
        "prefill_cp_hca_projection_allgather_push", "prefill_cp_hca_projection_allgather_payload_wait",
    )

    group_rows = TP_SIZE * local_rows
    padded_group_rows = pl.tensor.dim(cmp_values_full, 0)
    with pl.spmd(READBACK_WORKERS, name_hint="prefill_cp_hca_projection_allgather_readback", deps=[push_tid, payload_wait_tid]) as readback_tid:
        unpack_worker = pl.tile.get_block_idx()
        for unpack_row in pl.range(unpack_worker * COMM_ROW_TILE, group_rows, READBACK_WORKERS * COMM_ROW_TILE):
            unpack_rows = pl.min(COMM_ROW_TILE, group_rows - unpack_row)
            unpack_kv_tile = pl.load(gather_window, [unpack_row, 0], [COMM_ROW_TILE, HEAD_DIM], valid_shape=[unpack_rows, HEAD_DIM])
            pl.store(unpack_kv_tile, [unpack_row, 0], kv_full)
            unpack_values_bits = pl.load(
                gather_window, [unpack_row, HCA_PROJECTION_VALUES_OFFSET], [COMM_ROW_TILE, HCA_PROJECTION_BITS_DIM],
                valid_shape=[unpack_rows, HCA_PROJECTION_BITS_DIM],
            )
            unpack_values_tile = pl.reinterpret_view(unpack_values_bits, pl.FP32)
            pl.store(unpack_values_tile, [unpack_row, 0], cmp_values_full)
            unpack_scores_bits = pl.load(
                gather_window, [unpack_row, HCA_PROJECTION_SCORES_OFFSET], [COMM_ROW_TILE, HCA_PROJECTION_BITS_DIM],
                valid_shape=[unpack_rows, HCA_PROJECTION_BITS_DIM],
            )
            unpack_scores_tile = pl.reinterpret_view(unpack_scores_bits, pl.FP32)
            pl.store(unpack_scores_tile, [unpack_row, 0], cmp_scores_full)
        for unpack_row in pl.range(group_rows + unpack_worker * COMM_ROW_TILE, padded_group_rows, READBACK_WORKERS * COMM_ROW_TILE):
            unpack_rows = pl.min(COMM_ROW_TILE, padded_group_rows - unpack_row)
            unpack_projection_zero = pl.tile.full([COMM_ROW_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            unpack_projection_zero_valid = pl.set_validshape(unpack_projection_zero, unpack_rows, HEAD_DIM)
            pl.store(unpack_projection_zero_valid, [unpack_row, 0], cmp_values_full)
            pl.store(unpack_projection_zero_valid, [unpack_row, 0], cmp_scores_full)

    retire_tid = _prefill_cp_projection_retire(
        payload_local, gather_signal, group_base, tp_rank, readback_tid,
        pack_tid, HCA_PROJECTION_PADDING_OFFSET,
        "prefill_cp_hca_projection_allgather_readback_notify", "prefill_cp_hca_projection_allgather_readback_wait",
        "prefill_cp_hca_projection_allgather_retire",
    )
    return retire_tid


@pl.jit.inline
def prefill_cp_swa_projection_allgather_step(
    kv_local: pl.Tensor[[CP_Q_T_DYN, HEAD_DIM], pl.BF16],
    kv_full: pl.Tensor[[CP_KV_T_DYN, HEAD_DIM], pl.BF16],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather only projected KV rows."""
    kv_local.bind_dynamic(0, CP_Q_T_DYN)
    kv_full.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(kv_local, 0)
    push_tid, payload_wait_tid = _prefill_cp_projection_publish(
        kv_local, gather_window, gather_signal,
        local_rows, group_base, tp_rank, input_ready_tid, HEAD_DIM,
        "prefill_cp_swa_projection_allgather_push", "prefill_cp_swa_projection_allgather_payload_wait",
    )

    group_rows = TP_SIZE * local_rows
    with pl.spmd(
        READBACK_WORKERS, name_hint="prefill_cp_swa_projection_allgather_readback",
        deps=[push_tid, payload_wait_tid],
    ) as readback_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker * COMM_ROW_TILE, group_rows, READBACK_WORKERS * COMM_ROW_TILE):
            rows = pl.min(COMM_ROW_TILE, group_rows - row)
            kv_tile = pl.load(gather_window, [row, 0], [COMM_ROW_TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            pl.store(kv_tile, [row, 0], kv_full)

    retire_tid = _prefill_cp_projection_retire(
        kv_full, gather_signal, group_base, tp_rank, readback_tid,
        input_ready_tid, 0,
        "prefill_cp_swa_projection_allgather_readback_notify", "prefill_cp_swa_projection_allgather_readback_wait",
        "prefill_cp_swa_projection_allgather_retire",
    )
    return retire_tid


@pl.jit.inline(auto_scope=False)
def prefill_cp_materialize_hc_full(
    hc_local: pl.Tensor[[CP_Q_T_DYN, HC_MULT, D], pl.FP32],
    hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather HC rows in raw-word tiles and unpack directly into the public output."""
    local_rows = pl.tensor.dim(hc_local, 0)
    group_rows = pl.tensor.dim(hc_full, 0)
    raw_local_rows = local_rows * HC_MULT * 2
    raw_group_rows = group_rows * HC_MULT * 2
    hc_flat_local = pl.reshape(hc_local, [raw_local_rows, D // 2])
    hc_flat_full = pl.reshape(hc_full, [raw_group_rows, D // 2])
    completion_tids = pl.array.create(1, pl.TASK_ID)
    completion_tids[0] = input_ready_tid
    for raw_base in pl.range(0, raw_local_rows, PREFILL_LOCAL_CAP):
        with pl.scope():
            previous_tid = completion_tids[0]
            chunk_rows = pl.min(PREFILL_LOCAL_CAP, raw_local_rows - raw_base)
            packet_local = pl.create_tensor([chunk_rows, D], dtype=pl.BF16)
            with pl.spmd(READBACK_WORKERS, name_hint="prefill_cp_hc_pack", deps=[previous_tid]) as pack_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker * COMM_ROW_TILE, chunk_rows, READBACK_WORKERS * COMM_ROW_TILE):
                    rows = pl.min(COMM_ROW_TILE, chunk_rows - row)
                    values = pl.load(hc_flat_local, [raw_base + row, 0], [COMM_ROW_TILE, D // 2], valid_shape=[rows, D // 2])
                    words = pl.reinterpret_view(values, pl.BF16)
                    pl.store(words, [row, 0], packet_local)

            push_tid, wait_tid = _prefill_cp_projection_publish(
                packet_local, gather_window, gather_signal, chunk_rows, group_base, tp_rank, pack_tid, D,
                "prefill_cp_hc_push", "prefill_cp_hc_payload_wait",
            )
            with pl.spmd(
                TP_SIZE * READBACK_WORKERS, name_hint="prefill_cp_hc_readback", deps=[push_tid, wait_tid],
            ) as readback_tid:
                block = pl.tile.get_block_idx()
                source_rank = block // READBACK_WORKERS
                worker = block % READBACK_WORKERS
                for row in pl.range(worker * COMM_ROW_TILE, chunk_rows, READBACK_WORKERS * COMM_ROW_TILE):
                    rows = pl.min(COMM_ROW_TILE, chunk_rows - row)
                    window_row = source_rank * chunk_rows + row
                    output_row = source_rank * raw_local_rows + raw_base + row
                    words = pl.load(gather_window, [window_row, 0], [COMM_ROW_TILE, D], valid_shape=[rows, D])
                    output_values = pl.reinterpret_view(words, pl.FP32)
                    pl.store(output_values, [output_row, 0], hc_flat_full)
            retire_tid = _prefill_cp_projection_retire(
                hc_flat_full, gather_signal, group_base, tp_rank, readback_tid, previous_tid, 0,
                "prefill_cp_hc_readback_notify", "prefill_cp_hc_readback_wait", "prefill_cp_hc_retire",
            )
            completion_tids[0] = retire_tid

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_cp_hc_publish", deps=[completion_tids[0]],
    ) as publish_tid:
        anchor_tile = pl.load(hc_flat_full, [0, 0], [1, 8])
        anchor_words = pl.reinterpret_view(anchor_tile, pl.INT32)
        anchor = pl.tile.read(anchor_words, [0, 0])
        pl.tile.write(anchor_words, [0, 0], anchor)
        anchor_output = pl.reinterpret_view(anchor_words, pl.FP32)
        pl.store(anchor_output, [0, 0], hc_flat_full)
    return publish_tid


@pl.jit.inline
def prefill_cp_materialize_ffn_coeffs(
    post_local: pl.Tensor[[CP_Q_T_DYN, HC_MULT], pl.FP32],
    comb_local: pl.Tensor[[CP_Q_T_DYN, HC_MULT * HC_MULT], pl.FP32],
    post_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT], pl.FP32]],
    comb_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT * HC_MULT], pl.FP32]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    input_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Publish the two rank-major FFN coefficient matrices in one raw-word epoch."""
    local_rows = pl.tensor.dim(post_local, 0)
    payload = pl.create_tensor([local_rows, COEFF_PAYLOAD_WORDS], dtype=pl.BF16)
    with pl.spmd((local_rows + COMM_ROW_TILE - 1) // COMM_ROW_TILE, name_hint="prefill_cp_ffn_coeff_pack", deps=[input_ready_tid]) as pack_tid:
        row = pl.tile.get_block_idx() * COMM_ROW_TILE
        rows = pl.min(COMM_ROW_TILE, local_rows - row)
        comb = pl.load(comb_local, [row, 0], [COMM_ROW_TILE, HC_MULT * HC_MULT], valid_shape=[rows, HC_MULT * HC_MULT])
        comb_bits = pl.reinterpret_view(comb, pl.BF16)
        pl.store(comb_bits, [row, 0], payload)
        post = pl.load(post_local, [row, 0], [COMM_ROW_TILE, COEFF_POST_WORDS // 2], valid_shape=[rows, HC_MULT])
        post_bits = pl.reinterpret_view(post, pl.BF16)
        post_padded = pl.fillpad(post_bits, pad_value=pl.PadValue.zero)
        post_valid = pl.set_validshape(post_padded, rows, COEFF_POST_WORDS)
        pl.store(post_valid, [row, COEFF_COMB_WORDS], payload)

    push_tid, arrival_tid = _prefill_cp_projection_publish(
        payload, gather_window, gather_signal, local_rows, group_base, tp_rank, pack_tid, COEFF_PAYLOAD_WORDS,
        "prefill_cp_ffn_coeff_push", "prefill_cp_ffn_coeff_wait",
    )
    group_rows = TP_SIZE * local_rows
    with pl.spmd(READBACK_WORKERS, name_hint="prefill_cp_ffn_coeff_unpack", deps=[push_tid, arrival_tid]) as unpack_tid:
        worker = pl.tile.get_block_idx()
        for row0 in pl.range(worker * COMM_ROW_TILE, group_rows, READBACK_WORKERS * COMM_ROW_TILE):
            nrows = pl.min(COMM_ROW_TILE, group_rows - row0)
            comb_raw = pl.load(gather_window, [row0, 0], [COMM_ROW_TILE, COEFF_COMB_WORDS], valid_shape=[nrows, COEFF_COMB_WORDS])
            comb_output = pl.reinterpret_view(comb_raw, pl.FP32)
            pl.store(comb_output, [row0, 0], comb_full)
            post_raw = pl.load(gather_window, [row0, COEFF_COMB_WORDS], [COMM_ROW_TILE, COEFF_POST_WORDS], valid_shape=[nrows, COEFF_POST_WORDS])
            post_output = pl.reinterpret_view(post_raw, pl.FP32)
            post_output_valid = pl.set_validshape(post_output, nrows, HC_MULT)
            pl.store(post_output_valid, [row0, 0], post_full)
    retire_tid = _prefill_cp_projection_retire(
        payload, gather_signal, group_base, tp_rank, unpack_tid, pack_tid, COEFF_COMB_WORDS,
        "prefill_cp_ffn_coeff_readback_notify", "prefill_cp_ffn_coeff_readback_wait", "prefill_cp_ffn_coeff_retire",
    )
    return retire_tid


@pl.jit
def prefill_cp_token_allgather_fixture(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    hc_local: pl.Tensor[[CP_Q_T_DYN, HC_MULT, D], pl.FP32],
    hc_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather BF16 rows, then reuse the window for FP32 HC rows."""
    input_ready_tid = pl.system.task_dummy(deps=[])
    group_base = pl.const(0, pl.INT32)
    bf16_ready_tid = prefill_cp_token_allgather_into(
        hidden_local, group_out, gather_window, gather_signal, group_base, tp_rank, input_ready_tid,
    )
    prefill_cp_materialize_hc_full(
        hc_local, hc_full, gather_window, gather_signal, group_base, tp_rank, bf16_ready_tid,
    )
    return group_out, hc_full, gather_signal


@pl.jit.host
def l3_prefill_cp_token_allgather_fixture(
    hidden_local: pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_KV_T_DYN, D], pl.BF16]],
    hc_local: pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_Q_T_DYN, HC_MULT, D], pl.FP32],
    hc_full: pl.Out[pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
):
    """Gather BF16 token rows and materialize FP32 HC rows in two rounds."""
    hidden_local.bind_dynamic(2, CP_Q_T_DYN)
    group_out.bind_dynamic(2, CP_KV_T_DYN)
    hc_local.bind_dynamic(2, CP_Q_T_DYN)
    hc_full.bind_dynamic(2, CP_KV_T_DYN)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for round_id in pl.range(FIXTURE_ROUNDS):
        for rank in pl.range(pld.world_size()):
            gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
            gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            prefill_cp_token_allgather_fixture(
                hidden_local[round_id, rank], group_out[round_id, rank],
                hc_local[round_id, rank], hc_full[round_id, rank],
                gather_window, gather_signal, rank, device=rank,
            )


def materialize_spec(spec):
    """Materialise one shared init_value for all ranks."""
    import torch

    value = spec.init_value
    if value is None:
        return torch.zeros(spec.shape, dtype=spec.dtype)
    if isinstance(value, (int, float)):
        return torch.full(spec.shape, float(value), dtype=spec.dtype)
    if callable(value):
        value = value()
    return value.to(spec.dtype).reshape(spec.shape)


def cp_stack(value, tp_size):
    """Replicate one materialised tensor across the CP group."""
    return value.unsqueeze(0).expand(tp_size, *value.shape).contiguous()


def build_tensor_specs(local_t=FIXTURE_LOCAL_T):
    """Build two distinct rounds of per-rank inputs."""
    import torch

    from golden import TensorSpec

    if local_t < 1 or local_t > PREFILL_LOCAL_CAP:
        raise ValueError(f"local_t must be in [1, {PREFILL_LOCAL_CAP}], got {local_t}")
    group_t = TP_SIZE * local_t

    def init_hidden_local():
        shape = (FIXTURE_ROUNDS, TP_SIZE, local_t, D)
        values = torch.arange(FIXTURE_ROUNDS * TP_SIZE * local_t * D, dtype=torch.int32)
        values = values.remainder(251).reshape(shape).to(torch.bfloat16)
        for round_id in range(FIXTURE_ROUNDS):
            for rank in range(TP_SIZE):
                values[round_id, rank, :, 0] = float(round_id * TP_SIZE + rank)
        return values

    def init_hc_local():
        rows = torch.arange(FIXTURE_ROUNDS * TP_SIZE * local_t, dtype=torch.float32)
        lanes = torch.arange(HC_MULT * D, dtype=torch.float32).reshape(1, 1, 1, HC_MULT, D)
        return rows.reshape(FIXTURE_ROUNDS, TP_SIZE, local_t, 1, 1) + lanes / 32768

    return [
        TensorSpec("hidden_local", [FIXTURE_ROUNDS, TP_SIZE, local_t, D], torch.bfloat16, init_value=init_hidden_local),
        TensorSpec("group_out", [FIXTURE_ROUNDS, TP_SIZE, group_t, D], torch.bfloat16),
        TensorSpec("hc_local", [FIXTURE_ROUNDS, TP_SIZE, local_t, HC_MULT, D], torch.float32, init_value=init_hc_local),
        TensorSpec("hc_full", [FIXTURE_ROUNDS, TP_SIZE, group_t, HC_MULT, D], torch.float32),
    ]


def golden_prefill_cp_token_allgather(tensors):
    """Every rank receives its round's rank-major concatenation."""
    hidden_local = tensors["hidden_local"]
    rounds, tp_size, local_t, _ = hidden_local.shape
    gathered = hidden_local.reshape(rounds, tp_size * local_t, D)
    tensors["group_out"][:] = gathered.unsqueeze(1)
    hc_gathered = tensors["hc_local"].reshape(rounds, tp_size * local_t, HC_MULT, D)
    tensors["hc_full"][:] = hc_gathered.unsqueeze(1)


if __name__ == "__main__":
    import argparse

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="Standalone context-parallel prefill token-row all-gather test.")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument(
        "--local-t", type=int, default=FIXTURE_LOCAL_T,
        help=f"per-rank token count, 1..{PREFILL_LOCAL_CAP}",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        raise SystemExit(f"--tp={args.tp} does not match import-time TP_SIZE={TP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if not 1 <= args.local_t <= PREFILL_LOCAL_CAP:
        parser.error(f"--local-t must be in [1, {PREFILL_LOCAL_CAP}], got {args.local_t}")

    result = run(
        fn=l3_prefill_cp_token_allgather_fixture,
        specs=build_tensor_specs(args.local_t),
        golden_fn=golden_prefill_cp_token_allgather,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
