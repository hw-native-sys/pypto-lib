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
"""DeepSeek-V4.1 Flash vocabulary projection, distributed logits assembly and greedy sampling."""

import sys
from math import gcd, lcm

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
MAX_LOGIT_ROWS = _parse_int_argv("--logit-rows", 16) if __name__ == "__main__" else 16
GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * MAX_LOGIT_ROWS

# tiling
LM_HEAD_M_TILE = 80
LM_HEAD_N_TILE = 256
LM_HEAD_K_TILE = 128
LM_HEAD_PUSH_M_TILE = 192
LM_HEAD_PUSH_L1_K_TILE = 256
LM_HEAD_PUSH_L0_K_TILE = 64
LM_HEAD_GATHER_M_TILE = 72
LM_HEAD_TAIL_TILE = VOCAB_PER_TP % LM_HEAD_N_TILE or LM_HEAD_N_TILE
HIDDEN_OWNER_ROW_TILE = gcd(MAX_LOGIT_ROWS, 48)
HIDDEN_PUSH_WORKER_TILE = (MAX_LOGIT_ROWS + 64 // LM_HEAD_TP_SIZE - 1) // (64 // LM_HEAD_TP_SIZE)
HIDDEN_PUSH_ROW_TILE = gcd(HIDDEN_OWNER_ROW_TILE, HIDDEN_PUSH_WORKER_TILE)
HIDDEN_PUSH_CHUNK_TILE = min(12, HIDDEN_PUSH_ROW_TILE)

# Greedy reduction tiles and first-index selection.
GREEDY_VOCAB_TILE = 8192
GREEDY_ROW_TILE = 8
GREEDY_SEGMENT_TILE = GREEDY_VOCAB_TILE // GREEDY_ROW_TILE
GREEDY_INDEX_TILE = GREEDY_SEGMENT_TILE // GREEDY_ROW_TILE
GREEDY_NEGATIVE_INF_BITS = -8388608
GREEDY_INDEX_SENTINEL = 1073741824
SAMPLED_IDS_PAD = 8


if __name__ == "__main__" and DP_SIZE < 1:
    raise ValueError(f"--dp must be positive, got {DP_SIZE}")


def _lm_head_push_row_tile(num_rows: int) -> int:
    group_rows = LM_HEAD_TP_SIZE * num_rows
    # Cube row alignment and complete owner stripes on each AIV lane.
    row_alignment = lcm(16, 2 * HIDDEN_OWNER_ROW_TILE)
    if LM_HEAD_TP_SIZE > 1 and group_rows % row_alignment != 0:
        raise ValueError(f"LM head projection row count {group_rows} must be a multiple of {row_alignment}")
    return min(LM_HEAD_PUSH_M_TILE, group_rows)


@pl.jit.inline(auto_scope=False)
def _lm_head_project_tile(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr,
    column_tile: pl.constexpr,
) -> pl.Tile:
    acc = pl.tile.create([row_tile, column_tile], dtype=pl.FP32)
    for k in pl.pipeline(0, D, LM_HEAD_PUSH_L1_K_TILE, stage=2):
        a = pl.load(hidden, [row, k], [row_tile, LM_HEAD_PUSH_L1_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.load(
            weight, [k, column], [LM_HEAD_PUSH_L1_K_TILE, column_tile],
            target_memory=pl.Mem.Mat, cache=pl.CachePolicy.NOT_ALLOC_KEEP,
        )
        for inner in pl.pipeline(0, LM_HEAD_PUSH_L1_K_TILE, LM_HEAD_PUSH_L0_K_TILE, stage=2):
            left = pl.tile.extract(
                a, 0, inner, [row_tile, LM_HEAD_PUSH_L0_K_TILE], target_memory=pl.Mem.Left,
            )
            right = pl.tile.extract(
                b, inner, 0, [LM_HEAD_PUSH_L0_K_TILE, column_tile], target_memory=pl.Mem.Right,
            )
            acc = pl.matmul_acc(acc, left, right, init_cond=(k + inner == 0))
    return acc


@pl.jit.inline(auto_scope=False)
def _lm_head_publish_panel(
    shard: pl.Tile[[PANEL_ROWS_DYN, PANEL_COLUMNS_DYN], pl.FP32],
    received: pld.DistributedTensor,
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    num_rows: pl.constexpr,
    row_tile: pl.constexpr,
    column_tile: pl.constexpr,
):
    # Panel storage is ordered by source shard, then by vocabulary panel.
    panel_view = pl.tensor.view(received, [num_rows * (VOCAB // column_tile), column_tile])
    panel_row = (tp_rank * VOCAB_PER_TP + column) * num_rows // column_tile
    for offset in pl.range(0, row_tile // 2, HIDDEN_OWNER_ROW_TILE):
        stripe = row + lane * (row_tile // 2) + offset
        owner = stripe // HIDDEN_OWNER_ROW_TILE % LM_HEAD_TP_SIZE
        owner_row = stripe // (HIDDEN_OWNER_ROW_TILE * LM_HEAD_TP_SIZE) * HIDDEN_OWNER_ROW_TILE
        block = pl.tile.slice(shard, [HIDDEN_OWNER_ROW_TILE, column_tile], [offset, 0])
        panel_bf16 = pl.cast(block, pl.BF16, mode="rint")
        pld.tile.remote_store(panel_bf16, panel_view, group_base + owner, [panel_row + owner_row, 0])


@pl.jit.inline(auto_scope=False)
def _lm_head_project_push(
    hidden: pl.Tensor,
    weight: pl.Tensor,
    received: pld.DistributedTensor,
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    dispatch_ready: pl.Scalar[pl.TASK_ID],
    dispatch_notified: pl.Scalar[pl.TASK_ID],
    num_rows: pl.constexpr,
    row_tile: pl.constexpr,
    last_row_tile: pl.constexpr,
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(
        32, name_hint="lm_head_matmul_push", deps=[dispatch_ready, dispatch_notified],
        optimizations=[pl.cross_core_slot(slot_num=1)],
    ) as logits_pushed:
        full_panels = VOCAB_PER_TP // LM_HEAD_N_TILE
        worker = pl.tile.get_block_idx()
        group_rows = LM_HEAD_TP_SIZE * num_rows
        last_row = group_rows // row_tile * row_tile
        for row in pl.range(0, last_row, row_tile):
            for panel in pl.range(worker, full_panels, 32):
                column = panel * LM_HEAD_N_TILE
                acc = _lm_head_project_tile(hidden, weight, row, column, row_tile, LM_HEAD_N_TILE)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    shard = pl.aiv_shard(acc)
                    _lm_head_publish_panel(
                        shard, received, tp_rank, group_base, row, column, lane, num_rows, row_tile, LM_HEAD_N_TILE,
                    )
            if VOCAB_PER_TP % LM_HEAD_N_TILE != 0 and worker == full_panels % 32:
                column = full_panels * LM_HEAD_N_TILE
                acc_tail = _lm_head_project_tile(hidden, weight, row, column, row_tile, LM_HEAD_TAIL_TILE)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    shard_tail = pl.aiv_shard(acc_tail)
                    _lm_head_publish_panel(
                        shard_tail, received, tp_rank, group_base, row, column, lane,
                        num_rows, row_tile, LM_HEAD_TAIL_TILE,
                    )
        if group_rows != last_row:
            for panel in pl.range(worker, full_panels, 32):
                column = panel * LM_HEAD_N_TILE
                acc_last = _lm_head_project_tile(
                    hidden, weight, last_row, column, last_row_tile, LM_HEAD_N_TILE,
                )
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    shard_last = pl.aiv_shard(acc_last)
                    _lm_head_publish_panel(
                        shard_last, received, tp_rank, group_base, last_row, column, lane,
                        num_rows, last_row_tile, LM_HEAD_N_TILE,
                    )
            if VOCAB_PER_TP % LM_HEAD_N_TILE != 0 and worker == full_panels % 32:
                column = full_panels * LM_HEAD_N_TILE
                acc_last_tail = _lm_head_project_tile(
                    hidden, weight, last_row, column, last_row_tile, LM_HEAD_TAIL_TILE,
                )
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    shard_last_tail = pl.aiv_shard(acc_last_tail)
                    _lm_head_publish_panel(
                        shard_last_tail, received, tp_rank, group_base, last_row, column, lane,
                        num_rows, last_row_tile, LM_HEAD_TAIL_TILE,
                    )
    return logits_pushed


@pl.jit.inline(auto_scope=False)
def _lm_head_panel_gather_tile(
    received: pld.DistributedTensor,
    logits: pl.Tensor,
    shard: pl.Scalar[pl.INDEX],
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    num_rows: pl.constexpr,
    column_tile: pl.constexpr,
):
    output_column = shard * VOCAB_PER_TP + column
    input_row = output_column * num_rows // column_tile + row
    rows = pl.min(LM_HEAD_GATHER_M_TILE, num_rows - row)
    panel_view = pl.tensor.view(received, [num_rows * (VOCAB // column_tile), column_tile])
    tile = pl.load(panel_view, [input_row, 0], [LM_HEAD_GATHER_M_TILE, column_tile], valid_shape=[rows, column_tile])
    value = pl.cast(tile, pl.FP32, mode="none")
    pl.store(value, [row, output_column], logits)


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
def greedy_sample(
    logits: pl.Tensor,
    sampled_ids: pl.Tensor,
):
    """Select the first maximum token ID for each requested logits row."""
    rows = pl.tensor.dim(sampled_ids, 0)
    workers = pl.min(64, rows)
    with pl.spmd(workers, name_hint="lm_head_greedy_sample"):
        worker = pl.tile.get_block_idx()
        ramp = pl.arange(0, [1, GREEDY_ROW_TILE * GREEDY_ROW_TILE], dtype=pl.INT32)
        lane_grid = pl.reshape(ramp, [GREEDY_ROW_TILE, GREEDY_ROW_TILE])
        lane_indices = pl.shr(lane_grid, 3)
        lane_bases = pl.mul(lane_indices, GREEDY_SEGMENT_TILE)
        for row in pl.range(worker, rows, workers):
            inf_bits = pl.full([GREEDY_ROW_TILE, GREEDY_ROW_TILE], dtype=pl.INT32, value=GREEDY_NEGATIVE_INF_BITS)
            best_values = pl.reinterpret_view(inf_bits, pl.FP32)
            best_bases = lane_bases
            for offset in pl.pipeline(0, VOCAB, GREEDY_VOCAB_TILE, stage=2):
                valid = pl.min(GREEDY_VOCAB_TILE, VOCAB - offset)
                scores = pl.slice(logits, [1, GREEDY_VOCAB_TILE], [row, offset], valid_shape=[1, valid])
                scores = pl.fillpad(scores, pad_value=pl.PadValue.min)
                scores = pl.set_validshape(scores, 1, GREEDY_VOCAB_TILE)
                scores = pl.reshape(scores, [GREEDY_ROW_TILE, GREEDY_SEGMENT_TILE])
                maxima = pl.row_max(scores)
                current_zeros = pl.full([GREEDY_ROW_TILE, GREEDY_ROW_TILE], dtype=pl.FP32, value=0.0)
                current = pl.row_expand(current_zeros, maxima)
                update_mask = pl.cmp(current, best_values, cmp_type=4)
                update = pl.cast(update_mask, pl.INT32)
                offset_i32 = pl.cast(offset, pl.INT32)
                current_bases = pl.add(lane_bases, offset_i32)
                base_delta = pl.sub(current_bases, best_bases)
                updated_delta = pl.mul(update, base_delta)
                best_bases = pl.add(best_bases, updated_delta)
                best_values = pl.maximum(best_values, current)
            maximum = pl.col_max(best_values)
            best_value = pl.read(maximum, [0, 0])
            winning_mask = pl.cmp(best_values, best_value, cmp_type=0)
            winning_lanes = pl.cast(winning_mask, pl.INT32)
            base_offsets = pl.add(best_bases, -GREEDY_INDEX_SENTINEL)
            matched_bases = pl.mul(winning_lanes, base_offsets)
            base_candidates = pl.add(matched_bases, GREEDY_INDEX_SENTINEL)
            first_base = pl.col_min(base_candidates)
            best_base = pl.read(first_base, [0, 0])
            base = pl.cast(best_base, pl.INDEX)
            tail_cols = pl.min(GREEDY_SEGMENT_TILE, VOCAB - base)
            segment = pl.slice(logits, [1, GREEDY_SEGMENT_TILE], [row, base], valid_shape=[1, tail_cols])
            segment = pl.fillpad(segment, pad_value=pl.PadValue.min)
            segment = pl.set_validshape(segment, 1, GREEDY_SEGMENT_TILE)
            segment = pl.reshape(segment, [GREEDY_ROW_TILE, GREEDY_INDEX_TILE])
            indices = pl.arange(0, [1, GREEDY_SEGMENT_TILE], dtype=pl.INT32)
            indices = pl.reshape(indices, [GREEDY_ROW_TILE, GREEDY_INDEX_TILE])
            match_mask = pl.cmp(segment, best_value, cmp_type=0)
            matches = pl.cast(match_mask, pl.INT32)
            index_offsets = pl.add(indices, -GREEDY_INDEX_SENTINEL)
            matched_indices = pl.mul(matches, index_offsets)
            candidates = pl.add(matched_indices, GREEDY_INDEX_SENTINEL)
            minima = pl.row_min(candidates)
            min_zeros = pl.full([GREEDY_ROW_TILE, GREEDY_ROW_TILE], dtype=pl.INT32, value=0)
            min_grid = pl.row_expand(min_zeros, minima)
            first_index = pl.col_min(min_grid)
            index = pl.read(first_index, [0, 0])
            result = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            pl.write(result, [0, 0], best_base + index)
            sampled_ids[row : row + 1, :] = result


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
    selected_hidden = pl.create_tensor([num_rows, D], dtype=pl.BF16)
    with pl.spmd(64, name_hint="lm_head_select_rows") as selected_ready:
        block = pl.tile.get_block_idx()
        for row in pl.range(block, num_rows, 64):
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
        with pl.spmd(64, name_hint="lm_head_dispatch_push", deps=[selected_ready]) as dispatch_pushed:
            worker = pl.tile.get_block_idx()
            row_tiles = num_rows // HIDDEN_PUSH_ROW_TILE
            for tile in pl.range(worker, LM_HEAD_TP_SIZE * row_tiles, 64):
                peer = (tp_rank + tile // row_tiles) % LM_HEAD_TP_SIZE
                row = tile % row_tiles * HIDDEN_PUSH_ROW_TILE
                owner_stripe = row // HIDDEN_OWNER_ROW_TILE * LM_HEAD_TP_SIZE + tp_rank
                destination = owner_stripe * HIDDEN_OWNER_ROW_TILE + row % HIDDEN_OWNER_ROW_TILE
                pld.tensor.put(
                    hidden_window, peer=group_base + peer, src=selected_hidden, dst_offsets=[destination, 0],
                    src_offsets=[row, 0], shape=[HIDDEN_PUSH_ROW_TILE, D], chunk_rows=HIDDEN_PUSH_CHUNK_TILE,
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
        logits_pushed = _lm_head_project_push(
            hidden_window, lm_head_weight, logits_window, project_tp_rank, project_group_base,
            dispatch_ready, dispatch_notified, num_rows, _lm_head_push_row_tile(num_rows),
            (LM_HEAD_TP_SIZE * num_rows - 1) % _lm_head_push_row_tile(num_rows) + 1,
        )

        # Publish one completion update after all vector workers finish their puts.
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

        with pl.spmd(64, name_hint="lm_head_combine_gather", deps=[combine_ready, combine_notified]) as gathered:
            gather_worker = pl.tile.get_block_idx()
            row_tiles = (num_rows + LM_HEAD_GATHER_M_TILE - 1) // LM_HEAD_GATHER_M_TILE
            full_panels = (VOCAB_PER_TP - LM_HEAD_TAIL_TILE) // LM_HEAD_N_TILE
            for work in pl.pipeline(gather_worker, LM_HEAD_TP_SIZE * row_tiles * full_panels, 64, stage=2):
                shard = work // (row_tiles * full_panels)
                row = work // full_panels % row_tiles * LM_HEAD_GATHER_M_TILE
                column = work % full_panels * LM_HEAD_N_TILE
                _lm_head_panel_gather_tile(logits_window, logits, shard, row, column, num_rows, LM_HEAD_N_TILE)
            for tail in pl.range(gather_worker, LM_HEAD_TP_SIZE * row_tiles, 64):
                shard = tail // row_tiles
                row = tail % row_tiles * LM_HEAD_GATHER_M_TILE
                column = VOCAB_PER_TP - LM_HEAD_TAIL_TILE
                _lm_head_panel_gather_tile(logits_window, logits, shard, row, column, num_rows, LM_HEAD_TAIL_TILE)

        # Clear this rank's counters after its final logits-window read.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear", deps=[gathered], allow_early_resolve=False):
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
    parser.add_argument("--logit-rows", type=int, choices=[16, 72, 384], default=MAX_LOGIT_ROWS)
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
