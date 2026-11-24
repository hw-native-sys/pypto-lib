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
"""DeepSeek-V4.1 Flash vocabulary projection and distributed logits assembly."""

import sys
from math import gcd

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import EP_SIZE, FLASH as M, LM_HEAD_TP_SIZE as MODEL_LM_HEAD_TP_SIZE, SUPPORTED_LM_HEAD_TP_SIZES
from quant import pack_bf16_nz, unpack_bf16_nz


# Dynamic shape variables.
T_DYN = pl.dynamic("LM_HEAD_T_DYN")
PANEL_ROWS_DYN = pl.dynamic("LM_HEAD_PANEL_ROWS")
PANEL_COLUMNS_DYN = pl.dynamic("LM_HEAD_PANEL_COLUMNS")

# model config
D = M.hidden_size
VOCAB = M.vocab_size


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# DP groups built by the l3_lm_head fixture; the kernel carries no DP extent.
LM_HEAD_TP_SIZE = _parse_int_argv("--lm-head-tp", 2) if __name__ == "__main__" else MODEL_LM_HEAD_TP_SIZE
DP_SIZE = _parse_int_argv("--dp", 1) if __name__ == "__main__" else EP_SIZE // LM_HEAD_TP_SIZE
WORLD_SIZE = LM_HEAD_TP_SIZE * DP_SIZE if __name__ == "__main__" else EP_SIZE
VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE

# Standalone selected-row fixture.
MAX_LOGIT_ROWS = 16
if __name__ == "__main__":
    from golden.runner import _bench_enabled

    # Replay and profiling use the small fixture unless a row count is specified.
    single_case_flags = {"--num-tokens", "--golden-data", "--runtime-dir", "--save-data", "--enable-chip-swimlane"}
    cli_flags = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    if LM_HEAD_TP_SIZE == 2 and DP_SIZE == 1 and not (_bench_enabled() or cli_flags & single_case_flags):
        # The 256-row group exercises a full 192-row tile and a 64-row tail.
        MAX_LOGIT_ROWS = 128
    MAX_LOGIT_ROWS = _parse_int_argv("--logit-rows", MAX_LOGIT_ROWS)
GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * MAX_LOGIT_ROWS

# tiling
LM_HEAD_M_TILE = 80
LM_HEAD_N_TILE = 256
LM_HEAD_K_TILE = 128
LM_HEAD_PUSH_M_TILE = 192
LM_HEAD_PUSH_WIDE_M_TILE = 224
LM_HEAD_PUSH_SLOT_BYTES = LM_HEAD_PUSH_M_TILE // 2 * LM_HEAD_N_TILE * 4
LM_HEAD_PUSH_FIFO_BYTES = LM_HEAD_PUSH_SLOT_BYTES * 2
LM_HEAD_PUSH_WIDE_SLOT_BYTES = LM_HEAD_PUSH_WIDE_M_TILE // 2 * LM_HEAD_N_TILE * 4
LM_HEAD_PUSH_WIDE_FIFO_BYTES = LM_HEAD_PUSH_WIDE_SLOT_BYTES * 2
LM_HEAD_PUSH_L1_K_TILE = 256
LM_HEAD_PUSH_MULTI_L1_K_TILE = 128
LM_HEAD_PUSH_L0_K_TILE = 64
LM_HEAD_GATHER_M_TILE = 80
LM_HEAD_TAIL_TILE = VOCAB_PER_TP % LM_HEAD_N_TILE or LM_HEAD_N_TILE
HIDDEN_OWNER_ROW_TILE = gcd(MAX_LOGIT_ROWS, 16)
HIDDEN_PUSH_CHUNK_TILE = min(4, HIDDEN_OWNER_ROW_TILE)

# Padded sampled-ids row width shared with the decode sampling operators.
SAMPLED_IDS_PAD = 8


if __name__ == "__main__" and DP_SIZE < 1:
    raise ValueError(f"--dp must be positive, got {DP_SIZE}")


def _lm_head_push_row_tile(num_rows: int) -> int:
    return min(LM_HEAD_PUSH_M_TILE, LM_HEAD_TP_SIZE * num_rows)


@pl.jit.inline(auto_scope=False)
def _lm_head_publish_panel(
    panel: pl.Tile[[PANEL_ROWS_DYN, PANEL_COLUMNS_DYN], pl.FP32],
    received: pld.DistributedTensor,
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    selected_rows: pl.Scalar[pl.INDEX],
    num_rows: pl.constexpr,
    lane_rows: pl.Scalar[pl.INDEX],
    column_tile: pl.constexpr,
):
    # Each lane starts at its own half of the valid rows, including a short M tile.
    # Panel storage is ordered by source shard, then by vocabulary panel.
    panel_view = pl.tensor.view(received, [num_rows * (VOCAB // column_tile), column_tile])
    panel_row = (tp_rank * VOCAB_PER_TP + column) * selected_rows // column_tile
    for offset in pl.range(0, lane_rows, HIDDEN_OWNER_ROW_TILE):
        stripe = row + lane * lane_rows + offset
        owner = stripe // HIDDEN_OWNER_ROW_TILE % LM_HEAD_TP_SIZE
        owner_row = stripe // (HIDDEN_OWNER_ROW_TILE * LM_HEAD_TP_SIZE) * HIDDEN_OWNER_ROW_TILE
        block = pl.tile.slice(panel, [HIDDEN_OWNER_ROW_TILE, column_tile], [offset, 0])
        value = pl.cast(block, pl.BF16, mode="rint")
        pld.tile.remote_store(value, panel_view, group_base + owner, [panel_row + owner_row, 0])


@pl.jit.inline(auto_scope=False)
def _lm_head_project_vocab_tail(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    column_tile: pl.constexpr,
) -> pl.Tile:
    seed = pl.tile.create([row_tile, column_tile], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
    acc = pl.set_validshape(seed, rows, column_tile)
    for k in pl.pipeline(0, D, LM_HEAD_PUSH_MULTI_L1_K_TILE, stage=4):
        a = pl.load(hidden, [row, k], [row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE],
                    valid_shape=[rows, LM_HEAD_PUSH_MULTI_L1_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.load(
            weight, [k, column], [LM_HEAD_PUSH_MULTI_L1_K_TILE, column_tile],
            target_memory=pl.Mem.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP,
        )
        for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
            left = pl.tile.extract(
                a, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
            )
            right = pl.tile.extract(
                b, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, column_tile], target_memory=pl.Mem.Right,
            )
            acc = pl.matmul_acc(acc, left, right, init_cond=(k + inner == 0))
    return acc


@pl.jit.inline(auto_scope=False)
def _lm_head_prefetch_panel(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    step: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    k_tile: pl.constexpr,
) -> pl.Tuple[pl.Tile, pl.Tile]:
    k = step % (D // k_tile) * k_tile
    column = (worker + step // (D // k_tile) * 32) * LM_HEAD_N_TILE
    a = pl.load(hidden, [row, k], [row_tile, k_tile],
                valid_shape=[rows, k_tile], target_memory=pl.Mem.Mat)
    b = pl.load(
        weight, [k, column], [k_tile, LM_HEAD_N_TILE],
        target_memory=pl.Mem.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP,
    )
    return a, b


@pl.jit.inline(auto_scope=False)
def _lm_head_receive_panel(
    received: pld.DistributedTensor,
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    selected_rows: pl.Scalar[pl.INDEX],
    num_rows: pl.constexpr,
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    column_tile: pl.constexpr,
):
    panel = pl.tpop_from_aic(shape=[row_tile // 2, column_tile], dtype=pl.FP32, split=1)
    _lm_head_publish_panel(
        panel, received, tp_rank, group_base, row, column, lane, selected_rows, num_rows, rows // 2, column_tile,
    )
    pl.tfree_to_aic(panel)


@pl.jit.inline(auto_scope=False)
def _lm_head_project_push(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    received: pld.DistributedTensor,
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    dispatch_ready: pl.Scalar[pl.TASK_ID],
    dispatch_notified: pl.Scalar[pl.TASK_ID],
    selected_rows: pl.Scalar[pl.INDEX],
    num_rows: pl.constexpr,
    row_tile: pl.constexpr,
    multi_row_tiles: pl.constexpr,
    slot_bytes: pl.constexpr,
    fifo_bytes: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(
        32, name_hint="lm_head_matmul_push", deps=[dispatch_ready, dispatch_notified],
    ) as logits_pushed:
        # Each AIV receives half the FP32 rows and publishes its BF16 stripes.
        peer = pl.import_peer_buffer(name="projection_panels")
        pl.aic_initialize_pipe(
            c2v_consumer_buf=peer,
            dir_mask=1,
            slot_size=slot_bytes,
            slot_num=2,
            ptoas_plan_memory=multi_row_tiles,
            ptoas_buffer_id_sync=True,
        )
        buffer = pl.reserve_buffer(name="projection_panels", size=fifo_bytes)
        pl.aiv_initialize_pipe(
            c2v_consumer_buf=buffer,
            dir_mask=1,
            slot_size=slot_bytes,
            slot_num=2,
            ptoas_plan_memory=multi_row_tiles,
            ptoas_buffer_id_sync=True,
        )
        full_panels = VOCAB_PER_TP // LM_HEAD_N_TILE
        worker = pl.tile.get_block_idx()
        group_rows = LM_HEAD_TP_SIZE * selected_rows
        if multi_row_tiles:
            panel_count = (full_panels + 31 - worker) // 32
            fragments = D // LM_HEAD_PUSH_MULTI_L1_K_TILE
            steps = panel_count * fragments
            for row in pl.range(0, group_rows, row_tile):
                rows = pl.min(row_tile, group_rows - row)
                seed = pl.tile.create([row_tile, LM_HEAD_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                acc_pipe = pl.set_validshape(seed, rows, LM_HEAD_N_TILE)
                # Carry four L1 slots across vocabulary panels.
                a0, b0 = _lm_head_prefetch_panel(hidden, weight, row, worker, 0, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE)
                a1, b1 = _lm_head_prefetch_panel(hidden, weight, row, worker, 1, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE)
                a2, b2 = _lm_head_prefetch_panel(hidden, weight, row, worker, 2, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE)
                a3, b3 = _lm_head_prefetch_panel(hidden, weight, row, worker, 3, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE)
                for step in pl.range(0, steps - 4, 4):
                    ordinal = pl.cast(step, pl.UINT32)
                    fragment = ordinal % fragments
                    for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                        left = pl.tile.extract(
                            a0, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                        )
                        right = pl.tile.extract(
                            b0, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                        )
                        acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=(fragment == 0 and inner == 0))
                    a0, b0 = _lm_head_prefetch_panel(
                        hidden, weight, row, worker, step + 4, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE,
                    )
                    for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                        left = pl.tile.extract(
                            a1, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                        )
                        right = pl.tile.extract(
                            b1, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                        )
                        acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                    a1, b1 = _lm_head_prefetch_panel(
                        hidden, weight, row, worker, step + 5, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE,
                    )
                    for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                        left = pl.tile.extract(
                            a2, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                        )
                        right = pl.tile.extract(
                            b2, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                        )
                        acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                    a2, b2 = _lm_head_prefetch_panel(
                        hidden, weight, row, worker, step + 6, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE,
                    )
                    for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                        left = pl.tile.extract(
                            a3, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                        )
                        right = pl.tile.extract(
                            b3, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                        )
                        acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                    a3, b3 = _lm_head_prefetch_panel(
                        hidden, weight, row, worker, step + 7, rows, row_tile, LM_HEAD_PUSH_MULTI_L1_K_TILE,
                    )
                    if fragment == fragments - 4:
                        column = (worker + pl.cast(ordinal // fragments, pl.INDEX) * 32) * LM_HEAD_N_TILE
                        pl.tpush_to_aiv(acc_pipe, split=1)
                        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            _lm_head_receive_panel(
                                received, tp_rank, group_base, row, column, lane,
                                selected_rows, num_rows, rows, row_tile, LM_HEAD_N_TILE,
                            )
                for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                    left = pl.tile.extract(
                        a0, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                    )
                    right = pl.tile.extract(
                        b0, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                    )
                    acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                    left = pl.tile.extract(
                        a1, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                    )
                    right = pl.tile.extract(
                        b1, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                    )
                    acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                    left = pl.tile.extract(
                        a2, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                    )
                    right = pl.tile.extract(
                        b2, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                    )
                    acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                for inner in pl.pipeline(0, LM_HEAD_PUSH_MULTI_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
                    left = pl.tile.extract(
                        a3, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
                    )
                    right = pl.tile.extract(
                        b3, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], target_memory=pl.Mem.Right,
                    )
                    acc_pipe = pl.matmul_acc(acc_pipe, left, right, init_cond=False)
                n_last = (worker + (panel_count - 1) * 32) * LM_HEAD_N_TILE
                pl.tpush_to_aiv(acc_pipe, split=1)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    _lm_head_receive_panel(
                        received, tp_rank, group_base, row, n_last, lane,
                        selected_rows, num_rows, rows, row_tile, LM_HEAD_N_TILE,
                    )
                if VOCAB_PER_TP % LM_HEAD_N_TILE != 0 and worker == full_panels % 32:
                    column = full_panels * LM_HEAD_N_TILE
                    acc_tail = _lm_head_project_vocab_tail(hidden, weight, row, column, rows, row_tile, LM_HEAD_TAIL_TILE)
                    pl.tpush_to_aiv(acc_tail, split=1)
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        _lm_head_receive_panel(
                            received, tp_rank, group_base, row, column, lane,
                            selected_rows, num_rows, rows, row_tile, LM_HEAD_TAIL_TILE,
                        )
        else:
            # Share two L0A/L0B slots across the full and tail panels.
            for row in pl.range(0, group_rows, row_tile):
                panel_count = (full_panels + 31 - worker) // 32
                steps = panel_count * (D // LM_HEAD_PUSH_L1_K_TILE)
                seed = pl.tile.create([row_tile, LM_HEAD_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                acc_pipe = pl.set_validshape(seed, group_rows, LM_HEAD_N_TILE)
                # Carry both L1 slots across full vocabulary panels.
                a_even, b_even = _lm_head_prefetch_panel(hidden, weight, row, worker, 0, group_rows, row_tile, LM_HEAD_PUSH_L1_K_TILE)
                a_odd, b_odd = _lm_head_prefetch_panel(hidden, weight, row, worker, 1, group_rows, row_tile, LM_HEAD_PUSH_L1_K_TILE)
                for step in pl.range(0, steps - 2, 2):
                    ordinal = pl.cast(step, pl.UINT32)
                    k_even = pl.cast(ordinal % (D // LM_HEAD_PUSH_L1_K_TILE), pl.INDEX) * LM_HEAD_PUSH_L1_K_TILE
                    for inner in pl.range(0, LM_HEAD_PUSH_L1_K_TILE, 2 * LM_HEAD_PUSH_L0_K_TILE):
                        left_0: pl.Tile[
                            [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[0], pl.Mem.Left
                        ] = pl.tile.extract(a_even, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                            target_memory=pl.Mem.Left)
                        right_0: pl.Tile[
                            [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[0], pl.Mem.Right
                        ] = pl.tile.extract(b_even, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                            target_memory=pl.Mem.Right)
                        left_1: pl.Tile[
                            [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[1], pl.Mem.Left
                        ] = pl.tile.extract(a_even, 0, inner + LM_HEAD_PUSH_L0_K_TILE, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                            target_memory=pl.Mem.Left)
                        right_1: pl.Tile[
                            [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[1], pl.Mem.Right
                        ] = pl.tile.extract(b_even, inner + LM_HEAD_PUSH_L0_K_TILE, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                            target_memory=pl.Mem.Right)
                        acc_pipe = pl.matmul_acc(acc_pipe, left_0, right_0, init_cond=(k_even == 0 and inner == 0))
                        acc_pipe = pl.matmul_acc(acc_pipe, left_1, right_1)
                    a_even, b_even = _lm_head_prefetch_panel(hidden, weight, row, worker, step + 2, group_rows, row_tile, LM_HEAD_PUSH_L1_K_TILE)
                    for inner in pl.range(0, LM_HEAD_PUSH_L1_K_TILE, 2 * LM_HEAD_PUSH_L0_K_TILE):
                        left_0: pl.Tile[
                            [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[0], pl.Mem.Left
                        ] = pl.tile.extract(a_odd, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                            target_memory=pl.Mem.Left)
                        right_0: pl.Tile[
                            [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[0], pl.Mem.Right
                        ] = pl.tile.extract(b_odd, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                            target_memory=pl.Mem.Right)
                        left_1: pl.Tile[
                            [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[1], pl.Mem.Left
                        ] = pl.tile.extract(a_odd, 0, inner + LM_HEAD_PUSH_L0_K_TILE, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                            target_memory=pl.Mem.Left)
                        right_1: pl.Tile[
                            [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[1], pl.Mem.Right
                        ] = pl.tile.extract(b_odd, inner + LM_HEAD_PUSH_L0_K_TILE, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                            target_memory=pl.Mem.Right)
                        acc_pipe = pl.matmul_acc(acc_pipe, left_0, right_0, init_cond=False)
                        acc_pipe = pl.matmul_acc(acc_pipe, left_1, right_1)
                    a_odd, b_odd = _lm_head_prefetch_panel(hidden, weight, row, worker, step + 3, group_rows, row_tile, LM_HEAD_PUSH_L1_K_TILE)
                    if k_even == D - 2 * LM_HEAD_PUSH_L1_K_TILE:
                        n = (worker + pl.cast(ordinal // (D // LM_HEAD_PUSH_L1_K_TILE), pl.INDEX) * 32) * LM_HEAD_N_TILE
                        pl.tpush_to_aiv(acc_pipe, split=1)
                        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                            _lm_head_receive_panel(
                                received, tp_rank, group_base, row, n, lane,
                                selected_rows, num_rows, group_rows, row_tile, LM_HEAD_N_TILE,
                            )
                for inner in pl.range(0, LM_HEAD_PUSH_L1_K_TILE, 2 * LM_HEAD_PUSH_L0_K_TILE):
                    left_0: pl.Tile[
                        [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[0], pl.Mem.Left
                    ] = pl.tile.extract(a_even, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                        target_memory=pl.Mem.Left)
                    right_0: pl.Tile[
                        [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[0], pl.Mem.Right
                    ] = pl.tile.extract(b_even, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                        target_memory=pl.Mem.Right)
                    left_1: pl.Tile[
                        [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[1], pl.Mem.Left
                    ] = pl.tile.extract(a_even, 0, inner + LM_HEAD_PUSH_L0_K_TILE, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                        target_memory=pl.Mem.Left)
                    right_1: pl.Tile[
                        [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[1], pl.Mem.Right
                    ] = pl.tile.extract(b_even, inner + LM_HEAD_PUSH_L0_K_TILE, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                        target_memory=pl.Mem.Right)
                    acc_pipe = pl.matmul_acc(acc_pipe, left_0, right_0, init_cond=False)
                    acc_pipe = pl.matmul_acc(acc_pipe, left_1, right_1)
                for inner in pl.range(0, LM_HEAD_PUSH_L1_K_TILE, 2 * LM_HEAD_PUSH_L0_K_TILE):
                    left_0: pl.Tile[
                        [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[0], pl.Mem.Left
                    ] = pl.tile.extract(a_odd, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                        target_memory=pl.Mem.Left)
                    right_0: pl.Tile[
                        [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[0], pl.Mem.Right
                    ] = pl.tile.extract(b_odd, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                        target_memory=pl.Mem.Right)
                    left_1: pl.Tile[
                        [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[1], pl.Mem.Left
                    ] = pl.tile.extract(a_odd, 0, inner + LM_HEAD_PUSH_L0_K_TILE, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                        target_memory=pl.Mem.Left)
                    right_1: pl.Tile[
                        [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[1], pl.Mem.Right
                    ] = pl.tile.extract(b_odd, inner + LM_HEAD_PUSH_L0_K_TILE, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_N_TILE],
                                        target_memory=pl.Mem.Right)
                    acc_pipe = pl.matmul_acc(acc_pipe, left_0, right_0, init_cond=False)
                    acc_pipe = pl.matmul_acc(acc_pipe, left_1, right_1)
                n_last = (worker + (panel_count - 1) * 32) * LM_HEAD_N_TILE
                pl.tpush_to_aiv(acc_pipe, split=1)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    _lm_head_receive_panel(
                        received, tp_rank, group_base, row, n_last, lane,
                        selected_rows, num_rows, group_rows, row_tile, LM_HEAD_N_TILE,
                    )
                if VOCAB_PER_TP % LM_HEAD_N_TILE != 0 and worker == full_panels % 32:
                    column = full_panels * LM_HEAD_N_TILE
                    seed_tail = pl.tile.create(
                        [row_tile, LM_HEAD_TAIL_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True,
                    )
                    acc_tail = pl.set_validshape(seed_tail, group_rows, LM_HEAD_TAIL_TILE)
                    for k in pl.pipeline(0, D, LM_HEAD_PUSH_L1_K_TILE, stage=2):
                        a_tail = pl.load(
                            hidden, [row, k], [row_tile, LM_HEAD_PUSH_L1_K_TILE],
                            valid_shape=[group_rows, LM_HEAD_PUSH_L1_K_TILE], target_memory=pl.Mem.Mat,
                        )
                        b_tail = pl.load(
                            weight, [k, column], [LM_HEAD_PUSH_L1_K_TILE, LM_HEAD_TAIL_TILE],
                            target_memory=pl.Mem.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP,
                        )
                        for inner in pl.range(0, LM_HEAD_PUSH_L1_K_TILE, 2 * LM_HEAD_PUSH_L0_K_TILE):
                            left_0: pl.Tile[
                                [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[0], pl.Mem.Left
                            ] = pl.tile.extract(a_tail, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                                target_memory=pl.Mem.Left)
                            right_tail_0: pl.Tile[
                                [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_TAIL_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[0], pl.Mem.Right
                            ] = pl.tile.extract(b_tail, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_TAIL_TILE],
                                                target_memory=pl.Mem.Right)
                            left_1: pl.Tile[
                                [row_tile, LM_HEAD_PUSH_L0_K_TILE], pl.BF16, pl.MemRef("project_left", slots=2)[1], pl.Mem.Left
                            ] = pl.tile.extract(a_tail, 0, inner + LM_HEAD_PUSH_L0_K_TILE, [row_tile, LM_HEAD_PUSH_L0_K_TILE],
                                                target_memory=pl.Mem.Left)
                            right_tail_1: pl.Tile[
                                [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_TAIL_TILE], pl.BF16, pl.MemRef("project_right", slots=2)[1], pl.Mem.Right
                            ] = pl.tile.extract(b_tail, inner + LM_HEAD_PUSH_L0_K_TILE, 0, [LM_HEAD_PUSH_L0_K_TILE, LM_HEAD_TAIL_TILE],
                                                target_memory=pl.Mem.Right)
                            acc_tail = pl.matmul_acc(acc_tail, left_0, right_tail_0, init_cond=(k == 0 and inner == 0))
                            acc_tail = pl.matmul_acc(acc_tail, left_1, right_tail_1)
                    pl.tpush_to_aiv(acc_tail, split=1)
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                        _lm_head_receive_panel(
                            received, tp_rank, group_base, row, column, lane,
                            selected_rows, num_rows, group_rows, row_tile, LM_HEAD_TAIL_TILE,
                        )
    return logits_pushed


@pl.jit.inline(auto_scope=False)
def lm_head_project(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB], pl.BF16, pl.NZ],
    logits: pl.Tensor,
):
    """Project vocabulary logits with BF16 rounding and FP32 output."""
    rows = pl.tensor.dim(logits, 0)
    hidden_rows = pl.tensor.dim(hidden_states, 0)
    blocks = (rows + LM_HEAD_M_TILE - 1) // LM_HEAD_M_TILE * (VOCAB // LM_HEAD_N_TILE)
    with pl.spmd(32, name_hint="lm_head_matmul", optimizations=[pl.cross_core_slot(slot_num=1)]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, 32):
            row = block // (VOCAB // LM_HEAD_N_TILE) * LM_HEAD_M_TILE
            column = block % (VOCAB // LM_HEAD_N_TILE) * LM_HEAD_N_TILE
            output_rows = pl.min(LM_HEAD_M_TILE, rows - row)
            source_rows = pl.max(pl.min(LM_HEAD_M_TILE, hidden_rows - row), 0)
            acc = pl.tile.create([LM_HEAD_M_TILE, LM_HEAD_N_TILE], dtype=pl.FP32)
            for k in pl.pipeline(0, D, LM_HEAD_K_TILE, stage=2):
                a = pl.load(hidden_states, [row, k], [LM_HEAD_M_TILE, LM_HEAD_K_TILE], valid_shape=[source_rows, LM_HEAD_K_TILE])
                a = pl.fillpad(a, pad_value=pl.PadValue.zero)
                a = pl.set_validshape(a, LM_HEAD_M_TILE, LM_HEAD_K_TILE)
                w = pl.load(lm_head_weight, [k, column], [LM_HEAD_K_TILE, LM_HEAD_N_TILE], target_memory=pl.MemorySpace.Mat)
                acc = pl.matmul_acc(acc, a, w, init_cond=(k == 0))
            rounded = pl.cast(acc, pl.BF16, mode="rint")
            result = pl.cast(rounded, pl.FP32)
            value = pl.set_validshape(result, output_rows, LM_HEAD_N_TILE)
            pl.store(value, [row, column], logits)


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor,
    logits: pl.Tensor,
    hidden_window: pld.DistributedTensor,
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor,
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_rows: pl.constexpr,
):
    """Project indexed hidden rows with local or TP-sharded weights."""
    hidden_rows = pl.tensor.dim(hidden_states, 0)
    group_base = rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE
    tp_rank = rank % LM_HEAD_TP_SIZE
    selected_rows = pl.tensor.dim(logits, 0)
    selected_hidden = pl.create_tensor([selected_rows, D], dtype=pl.BF16)
    with pl.spmd(64, name_hint="lm_head_select_rows") as selected_ready:
        block = pl.tile.get_block_idx()
        for row in pl.range(block, selected_rows, 64):
            source_raw = pl.read(logit_row_indices, [row])
            source = pl.cast(pl.max(pl.min(source_raw, hidden_rows - 1), 0), pl.INDEX)
            selected = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if source_raw >= 0:
                selected = hidden_states[source:source + 1, :]
            selected_hidden[row:row + 1, :] = selected
    if LM_HEAD_TP_SIZE == 1:
        lm_head_project(selected_hidden, lm_head_weight, logits)
    else:
        pl.static_assert(num_rows % HIDDEN_OWNER_ROW_TILE == 0, "LM-head owner rows must fill complete interleaved stripes")
        row_tiles = selected_rows // HIDDEN_OWNER_ROW_TILE
        transfer_count = LM_HEAD_TP_SIZE * row_tiles
        push_workers = pl.max(1, pl.min(64, transfer_count))
        # Consecutive workers send the same row stripe to different peers.
        with pl.spmd(push_workers, name_hint="lm_head_dispatch_push", deps=[selected_ready]) as dispatch_pushed:
            worker = pl.tile.get_block_idx()
            for tile in pl.range(worker, transfer_count, 64):
                peer = (tp_rank + tile % LM_HEAD_TP_SIZE) % LM_HEAD_TP_SIZE
                stripe = tile // LM_HEAD_TP_SIZE
                row = stripe * HIDDEN_OWNER_ROW_TILE
                destination = (stripe * LM_HEAD_TP_SIZE + tp_rank) * HIDDEN_OWNER_ROW_TILE
                pld.tensor.put(
                    hidden_window, peer=group_base + peer, src=selected_hidden, dst_offsets=[destination, 0],
                    src_offsets=[row, 0], shape=[HIDDEN_OWNER_ROW_TILE, D], chunk_rows=HIDDEN_PUSH_CHUNK_TILE,
                    chunk_cols=D, pipeline=True,
                )
        with pl.spmd(LM_HEAD_TP_SIZE, name_hint="lm_head_dispatch_notify", deps=[dispatch_pushed]) as dispatch_notified:
            peer = pl.tile.get_block_idx()
            if peer != tp_rank:
                pld.system.notify(
                    hidden_done, peer=group_base + peer, offsets=[tp_rank, 0], value=64, op=pld.NotifyOp.AtomicAdd,
                )

        # Wait after all local hidden-row publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[dispatch_pushed], no_dep_args=[hidden_done], allow_early_resolve=False) as dispatch_ready:
            for owner_tp in pl.range(LM_HEAD_TP_SIZE):
                if owner_tp != tp_rank:
                    pld.system.wait(
                        signal=hidden_done, offsets=[owner_tp, 0], expected=pl.cast(64, pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        project_tp_rank = pl.cast(tp_rank, pl.INT32)
        project_group_base = pl.cast(group_base, pl.INT32)
        group_rows = LM_HEAD_TP_SIZE * selected_rows
        bands = (group_rows + LM_HEAD_PUSH_M_TILE - 1) // LM_HEAD_PUSH_M_TILE
        wide_bands = (group_rows + LM_HEAD_PUSH_WIDE_M_TILE - 1) // LM_HEAD_PUSH_WIDE_M_TILE
        if wide_bands < bands:
            logits_pushed = _lm_head_project_push(
                hidden_window, lm_head_weight, logits_window, project_tp_rank, project_group_base,
                dispatch_ready, dispatch_notified, selected_rows, num_rows, LM_HEAD_PUSH_WIDE_M_TILE, True,
                LM_HEAD_PUSH_WIDE_SLOT_BYTES, LM_HEAD_PUSH_WIDE_FIFO_BYTES,
            )
        else:
            logits_pushed = _lm_head_project_push(
                hidden_window, lm_head_weight, logits_window, project_tp_rank, project_group_base,
                dispatch_ready, dispatch_notified, selected_rows, num_rows, _lm_head_push_row_tile(num_rows),
                LM_HEAD_TP_SIZE * num_rows > LM_HEAD_PUSH_M_TILE,
                LM_HEAD_PUSH_SLOT_BYTES, LM_HEAD_PUSH_FIFO_BYTES,
            )

        # Each publishing worker drains its remote stores before this task runs.
        with pl.spmd(LM_HEAD_TP_SIZE, name_hint="lm_head_combine_notify", deps=[logits_pushed]) as combine_notified:
            owner_tp = pl.tile.get_block_idx()
            if owner_tp != tp_rank:
                pld.system.notify(
                    logits_done, peer=group_base + owner_tp, offsets=[tp_rank, 0], value=64, op=pld.NotifyOp.AtomicAdd,
                )

        # Wait after all local vocab-shard publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[logits_pushed], no_dep_args=[logits_done], allow_early_resolve=False) as combine_ready:
            for src_tp in pl.range(LM_HEAD_TP_SIZE):
                if src_tp != tp_rank:
                    pld.system.wait(
                        signal=logits_done, offsets=[src_tp, 0], expected=pl.cast(64, pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        with pl.spmd(64, name_hint="lm_head_combine_gather", deps=[combine_ready, combine_notified]):
            gather_worker = pl.tile.get_block_idx()
            row_tiles = (selected_rows + LM_HEAD_GATHER_M_TILE - 1) // LM_HEAD_GATHER_M_TILE
            full_panels = VOCAB_PER_TP // LM_HEAD_N_TILE
            panel_tasks = LM_HEAD_TP_SIZE * row_tiles * full_panels
            panels_u32 = pl.cast(full_panels, pl.UINT32)
            rows_u32 = pl.cast(row_tiles, pl.UINT32)
            panels = pl.tensor.view(logits_window, [num_rows * (VOCAB // LM_HEAD_N_TILE), LM_HEAD_N_TILE])
            # Each slot owns one input buffer, one output buffer and its own pipeline credits.
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
            pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, 1)
            pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
            pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 1)
            for work in pl.range(gather_worker, panel_tasks, 64):
                ordinal = pl.cast(work, pl.UINT32)
                packed_row = ordinal // panels_u32
                shard = pl.cast(packed_row // rows_u32, pl.INDEX)
                row = pl.cast(packed_row % rows_u32, pl.INDEX) * LM_HEAD_GATHER_M_TILE
                panel = pl.cast(ordinal - packed_row * panels_u32, pl.INDEX)
                input_row = shard * (VOCAB_PER_TP * selected_rows // LM_HEAD_N_TILE) + panel * selected_rows + row
                output_column = shard * VOCAB_PER_TP + panel * LM_HEAD_N_TILE
                rows = pl.min(LM_HEAD_GATHER_M_TILE, selected_rows - row)
                slot = pl.cast((ordinal // pl.cast(64, pl.UINT32)) % pl.cast(2, pl.UINT32), pl.INDEX)
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
                panel_input: pl.Tile[
                    [LM_HEAD_GATHER_M_TILE, LM_HEAD_N_TILE], pl.BF16, pl.MemRef("input", slots=2)[slot], pl.Mem.Vec
                ] = pl.load(
                    panels, [input_row, 0], [LM_HEAD_GATHER_M_TILE, LM_HEAD_N_TILE], valid_shape=[rows, LM_HEAD_N_TILE],
                )
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
                pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
                pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
                value: pl.Tile[
                    [LM_HEAD_GATHER_M_TILE, LM_HEAD_N_TILE], pl.FP32, pl.MemRef("output", slots=2)[slot], pl.Mem.Vec
                ] = pl.cast(
                    panel_input, pl.FP32, mode="none",
                )
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
                pl.store(value, [row, output_column], logits)
                pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
            # The narrow tail reuses slot 0 after its full-panel transfers finish.
            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 1)
            pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 1)
            tail_panels = pl.tensor.view(logits_window, [num_rows * (VOCAB // LM_HEAD_TAIL_TILE), LM_HEAD_TAIL_TILE])
            for tail in pl.range(gather_worker, LM_HEAD_TP_SIZE * row_tiles, 64):
                shard = tail // row_tiles
                row = tail % row_tiles * LM_HEAD_GATHER_M_TILE
                column = VOCAB_PER_TP - LM_HEAD_TAIL_TILE
                output_column = shard * VOCAB_PER_TP + column
                input_row = output_column * selected_rows // LM_HEAD_TAIL_TILE + row
                rows = pl.min(LM_HEAD_GATHER_M_TILE, selected_rows - row)
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
                tail_tile: pl.Tile[
                    [LM_HEAD_GATHER_M_TILE, LM_HEAD_TAIL_TILE], pl.BF16, pl.MemRef("input", slots=2)[0], pl.Mem.Vec
                ] = pl.load(
                    tail_panels, [input_row, 0], [LM_HEAD_GATHER_M_TILE, LM_HEAD_TAIL_TILE], valid_shape=[rows, LM_HEAD_TAIL_TILE],
                )
                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
                pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
                pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
                tail_value: pl.Tile[
                    [LM_HEAD_GATHER_M_TILE, LM_HEAD_TAIL_TILE], pl.FP32, pl.MemRef("output", slots=2)[0], pl.Mem.Vec
                ] = pl.cast(
                    tail_tile, pl.FP32, mode="none",
                )
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, 0)
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, 0)
                pl.store(tail_value, [row, output_column], logits)
                pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
            pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)


@pl.jit.inline(auto_scope=False)
def clear_lm_head_signals(logits: pl.Tensor, hidden_done: pld.DistributedTensor, logits_done: pld.DistributedTensor):
    if LM_HEAD_TP_SIZE > 1:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear", allow_early_resolve=False):
            anchor = pl.read(logits, [0, 0])
            zero = pl.cast(0, pl.INT32)
            for peer in pl.range(LM_HEAD_TP_SIZE):
                pl.write(hidden_done, [peer, 0], zero)
                pl.write(logits_done, [peer, 0], zero)
            pl.write(logits, [0, 0], anchor)


@pl.jit
def l2_lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Validate selected-row vocabulary projection."""
    hidden_states.bind_dynamic(0, T_DYN)
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        rank, MAX_LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, hidden_done, logits_done)
    return logits


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, D, VOCAB_PER_TP], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    # Windows are group-local: hidden_window holds one row slot per group member,
    # and every card receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 2)
    hidden_done_buf = pld.alloc_window_buffer(LM_HEAD_TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(LM_HEAD_TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.BF16)
        logits_done = pld.window(logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done, r, device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    # Vocabulary shards in global token order.
    weight = unpack_bf16_nz(tensors["lm_head_weight"]).float()
    full_weight = torch.cat([weight[tp] for tp in range(LM_HEAD_TP_SIZE)], dim=-1)
    full_logits = []
    for owner_rank in range(WORLD_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight).to(torch.bfloat16).float())
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs(num_tokens=MAX_LOGIT_ROWS):
    import torch
    from golden import TensorSpec

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, MAX_LOGIT_ROWS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        checkpoint = (torch.randn(LM_HEAD_TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        shards = pack_bf16_nz(checkpoint.transpose(-1, -2).contiguous())
        return torch.stack([shards[r % LM_HEAD_TP_SIZE] for r in range(WORLD_SIZE)], dim=0)

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    specs = [
        TensorSpec("hidden_states", [WORLD_SIZE, MAX_LOGIT_ROWS, D], torch.bfloat16, init_value=init_hidden_states),
        # Resident vocabulary shard per world rank.
        TensorSpec("lm_head_weight", [WORLD_SIZE, D, VOCAB_PER_TP], torch.bfloat16, init_value=init_lm_head_weight, resident="stacked"),
        TensorSpec("logits", [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], torch.float32),
        TensorSpec("logit_row_indices", [WORLD_SIZE, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
    ]
    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--lm-head-tp", type=int, default=LM_HEAD_TP_SIZE, choices=SUPPORTED_LM_HEAD_TP_SIZES, help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, help="DP groups (world size = tp * dp)")
    parser.add_argument(
        "--logit-rows", type=int, choices=[16, 64, 72, 80, 128, 320, 384], default=MAX_LOGIT_ROWS,
        help="Rows per rank. Default TP2 validation uses 128 rows.",
    )
    parser.add_argument("--num-tokens", type=int, default=MAX_LOGIT_ROWS, help="Active hidden rows each owner projects (0 = empty shard)")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(WORLD_SIZE)), help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.execution_mode and LM_HEAD_TP_SIZE > 1:
        parser.error("fused TP LM head requires A5 hardware; use --lm-head-tp 1 for simulation")

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < WORLD_SIZE:
        parser.error(f"need at least {WORLD_SIZE} devices, got {device_ids}")
    if args.lm_head_tp != LM_HEAD_TP_SIZE or args.dp != DP_SIZE:
        parser.error("--lm-head-tp and --dp must match the import-time configuration")
    if not 0 <= args.num_tokens <= MAX_LOGIT_ROWS:
        parser.error(f"--num-tokens must be between 0 and {MAX_LOGIT_ROWS}")

    result = run(
        fn=l3_lm_head,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_lm_head,
        compare_fn={"logits": ratio_allclose(atol=1e-3, rtol=1.0 / 128, max_error_ratio=0.0)},
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:WORLD_SIZE], num_sub_workers=0),
            backend_type=args.backend_type, execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
