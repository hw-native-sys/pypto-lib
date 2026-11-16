# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4.1 Flash vocabulary projection, distributed logits assembly and greedy sampling."""

import sys
from math import gcd
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.runtime import pto_isa_include_dir

from config import EP_SIZE, FLASH as M, LM_HEAD_TP_SIZE, SUPPORTED_LM_HEAD_TP_SIZES
from hc import hc_head, golden_hc_collapse
from rmsnorm import rms_norm, golden_rms_norm_tiled
from quant import pack_bf16_nz, unpack_bf16_nz


# Dynamic shape variables.
T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size
HC_MULT = M.hc_mult

def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# DP groups built by the l3_lm_head fixture; the kernel carries no DP extent.
DP_SIZE = _parse_int_argv("--dp", 1) if __name__ == "__main__" else EP_SIZE // LM_HEAD_TP_SIZE
WORLD_SIZE = LM_HEAD_TP_SIZE * DP_SIZE if __name__ == "__main__" else EP_SIZE
VOCAB_PER_TP = VOCAB // LM_HEAD_TP_SIZE

# Standalone selected-row fixture.
MAX_LOGIT_ROWS = _parse_int_argv("--logit-rows", 16) if __name__ == "__main__" else 16
GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * MAX_LOGIT_ROWS

# tiling
LM_HEAD_M_TILE = 80
GROUP_M_TILE = min(192, GROUP_LOGIT_ROWS)
LOGITS_CAST_ROW_TILE = 8
HIDDEN_OWNER_ROW_TILE = gcd(MAX_LOGIT_ROWS, 48)
HIDDEN_PUSH_WORKER_TILE = (MAX_LOGIT_ROWS + 64 // LM_HEAD_TP_SIZE - 1) // (64 // LM_HEAD_TP_SIZE)
HIDDEN_PUSH_ROW_TILE = gcd(HIDDEN_OWNER_ROW_TILE, HIDDEN_PUSH_WORKER_TILE)
HIDDEN_PUSH_CHUNK_TILE = min(12, HIDDEN_PUSH_ROW_TILE)
LOGITS_COMM_TILE = 512

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


@pl.jit.extern(
    core_type="aic", source=Path(__file__).parent / "kernels" / "lm_head_projection.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _lm_head_projection(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor,
    logits: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
    vocabulary: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project BF16 hidden states against packed NZ vocabulary weights."""
    ...


@pl.jit.extern(
    core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "lm_head_project_push.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "lm_head_project_push.cpp", dual_aiv_dispatch=True,
    include_dirs=(pto_isa_include_dir(),),
)
def _lm_head_project_push(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor,
    received: pl.Out[pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_size: pl.Scalar[pl.INDEX],
    owner_rows: pl.Scalar[pl.INDEX],
    vocabulary: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
    owner_row_tile: pl.Scalar[pl.INDEX],
) -> pld.DistributedTensor:
    return received


@pl.jit.extern(
    core_type="aiv", source=Path(__file__).parent / "kernels" / "lm_head_panel_gather.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _lm_head_panel_gather(
    received: pl.Tensor,
    logits: pl.Out[pl.Tensor],
    tp_size: pl.Scalar[pl.INDEX],
    owner_rows: pl.Scalar[pl.INDEX],
    vocabulary: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def lm_head_project(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logits: pl.Tensor,
    m_tile: pl.constexpr,
):
    """Write vocabulary logits in place and return the completion task."""
    rows = pl.tensor.dim(logits, 0)
    with pl.spmd(32, name_hint="lm_head_matmul") as projected:
        _lm_head_projection(hidden_states, lm_head_weight, logits, rows, VOCAB_PER_TP, m_tile)
    return projected


@pl.jit.inline(auto_scope=False)
def lm_head_cast_logits(source: pl.Tensor, logits: pl.Tensor, ready: pl.Scalar[pl.TASK_ID]):
    """Convert BF16 projection output to public FP32 logits."""
    rows = pl.tensor.dim(source, 0)
    columns = pl.tensor.dim(source, 1)
    column_tiles = (columns + LOGITS_COMM_TILE - 1) // LOGITS_COMM_TILE
    tiles = (rows + LOGITS_CAST_ROW_TILE - 1) // LOGITS_CAST_ROW_TILE * column_tiles
    with pl.spmd(32, name_hint="lm_head_cast_logits", deps=[ready]) as converted:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, tiles, 32):
            row = tile // column_tiles * LOGITS_CAST_ROW_TILE
            column = tile % column_tiles * LOGITS_COMM_TILE
            valid_rows = pl.min(LOGITS_CAST_ROW_TILE, rows - row)
            valid_columns = pl.min(LOGITS_COMM_TILE, columns - column)
            chunk = pl.slice(source, [LOGITS_CAST_ROW_TILE, LOGITS_COMM_TILE], [row, column], valid_shape=[valid_rows, valid_columns])
            result = pl.cast(chunk, pl.FP32)
            logits[row:row + valid_rows, column:column + valid_columns] = result
    return logits, converted


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
    first_epoch: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], pl.Scalar[pl.TASK_ID]]:
    if LM_HEAD_TP_SIZE == 1:
        selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
        for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_select_rows"):
            source_row_raw = pl.read(logit_row_indices, [row])
            hidden_rows = pl.tensor.dim(hidden_states, 0)
            safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
            selected_row = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if source_row_raw >= 0:
                source_row = pl.cast(safe_raw, pl.INDEX)
                selected_row = hidden_states[source_row : source_row + 1, :]
            selected_hidden[row : row + 1, :] = selected_row
        local_logits = pl.create_tensor([MAX_LOGIT_ROWS, VOCAB], dtype=pl.BF16)
        projected = lm_head_project(selected_hidden, lm_head_weight, local_logits, MAX_LOGIT_ROWS)
        logits, completed = lm_head_cast_logits(local_logits, logits, projected)
    else:
        # Selected hidden rows in GM.
        selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)

        with pl.spmd(64, name_hint="lm_head_select_rows") as _select_tid:
            block = pl.tile.get_block_idx()
            hidden_rows = pl.tensor.dim(hidden_states, 0)
            for row in pl.range(block, MAX_LOGIT_ROWS, 64):
                source_raw = pl.read(logit_row_indices, [row])
                source = pl.cast(pl.max(pl.min(source_raw, hidden_rows - 1), 0), pl.INDEX)
                selected = pl.full([1, D], dtype=pl.BF16, value=0.0)
                if source_raw >= 0:
                    selected = hidden_states[source:source + 1, :]
                selected_hidden[row:row + 1, :] = selected
        with pl.spmd(64, name_hint="lm_head_dispatch_push", deps=[_select_tid]) as _dispatch_push_tid:
            worker = pl.tile.get_block_idx()
            row_tiles = MAX_LOGIT_ROWS // HIDDEN_PUSH_ROW_TILE
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
        with pl.spmd(LM_HEAD_TP_SIZE, name_hint="lm_head_dispatch_notify", deps=[_dispatch_push_tid]) as _dispatch_notify_tid:
            peer = pl.tile.get_block_idx()
            if peer != tp_rank:
                pld.system.notify(
                    hidden_done, peer=group_base + peer, offsets=[tp_rank, 0], value=64, op=pld.NotifyOp.AtomicAdd,
                )

        # Wait after all local hidden-row publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid], no_dep_args=[hidden_done], allow_early_resolve=False) as _dwait_tid:
            for owner_tp in pl.range(LM_HEAD_TP_SIZE):
                if owner_tp != tp_rank:
                    pld.system.wait(
                        signal=hidden_done, offsets=[owner_tp, 0], expected=pl.cast((done_epoch - first_epoch + 1) * 64, pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        with pl.spmd(32, name_hint="lm_head_matmul_push", deps=[_dwait_tid, _dispatch_notify_tid]) as _push_tid:
            logits_window = _lm_head_project_push(
                hidden_window, lm_head_weight, logits_window, tp_rank, group_base, LM_HEAD_TP_SIZE,
                MAX_LOGIT_ROWS, VOCAB_PER_TP, GROUP_M_TILE, HIDDEN_OWNER_ROW_TILE,
            )

        # Publish one completion update after all vector workers finish their puts.
        with pl.spmd(LM_HEAD_TP_SIZE, name_hint="lm_head_combine_notify", deps=[_push_tid]) as _notify_tid:
            owner_tp = pl.tile.get_block_idx()
            if owner_tp != tp_rank:
                pld.system.notify(
                    logits_done, peer=group_base + owner_tp, offsets=[tp_rank, 0], value=64, op=pld.NotifyOp.AtomicAdd,
                )

        # Wait after all local vocab-shard publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid], no_dep_args=[logits_done], allow_early_resolve=False) as _cwait_tid:
            for src_tp in pl.range(LM_HEAD_TP_SIZE):
                if src_tp != tp_rank:
                    pld.system.wait(
                        signal=logits_done, offsets=[src_tp, 0], expected=pl.cast((done_epoch - first_epoch + 1) * 64, pl.INT32),
                        cmp=pld.WaitCmp.Ge,
                    )

        with pl.spmd(64, name_hint="lm_head_combine_gather", deps=[_cwait_tid, _notify_tid]) as _gather_tid:
            logits = _lm_head_panel_gather(logits_window, logits, LM_HEAD_TP_SIZE, MAX_LOGIT_ROWS, VOCAB_PER_TP)

        # Clear this rank's counters after its final logits-window read.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear", deps=[_gather_tid], allow_early_resolve=False) as _clear_tid:
            anchor = pl.read(logits, [0, 0])
            if done_epoch == completed_epochs:
                zero = pl.cast(0, pl.INT32)
                for peer in pl.range(LM_HEAD_TP_SIZE):
                    pl.write(hidden_done, [peer, 0], zero)
                    pl.write(logits_done, [peer, 0], zero)
            pl.write(logits, [0, 0], anchor)
        completed = _clear_tid

    return logits, completed


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor,
    sampled_ids: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID] = None,
):
    """Select the first maximum token ID for each requested logits row."""
    rows = pl.tensor.dim(sampled_ids, 0)
    workers = pl.min(64, rows)
    for worker in pl.spmd(workers, name_hint="lm_head_greedy_sample", deps=[ready]):
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
    return sampled_ids


@pl.jit.extern(
    core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "lm_head_sample.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "lm_head_sample.cpp", dual_aiv_dispatch=True,
    include_dirs=(pto_isa_include_dir(),),
)
def _lm_head_sample_partial(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor,
    partial_values: pl.Out[pl.Tensor],
    partial_indices: pl.Out[pl.Tensor],
    active_widths: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    return partial_values, partial_indices


@pl.jit.extern(
    core_type="aiv", source=Path(__file__).parent / "kernels" / "lm_head_sample_merge.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _lm_head_sample_merge(
    partial_values: pl.Tensor,
    partial_indices: pl.Tensor,
    sampled_ids: pl.Out[pl.Tensor],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def lm_head_sample(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB], pl.BF16, pl.NZ],
    active_widths: pl.Tensor,
    sampled_ids: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID] = None,
):
    """Project and greedily sample the bounded TP1 verification rows."""
    rows = pl.tensor.dim(sampled_ids, 0)
    partial_values = pl.create_tensor([32, LM_HEAD_M_TILE], dtype=pl.FP32)
    partial_indices = pl.create_tensor([32, LM_HEAD_M_TILE], dtype=pl.INT32)
    with pl.spmd(32, name_hint="lm_head_sample_partial", deps=[ready], allow_early_resolve=True) as projected:
        partial_values, partial_indices = _lm_head_sample_partial(
            hidden_states, lm_head_weight, partial_values, partial_indices, active_widths, rows,
        )
    with pl.spmd((rows + SAMPLED_IDS_PAD - 1) // SAMPLED_IDS_PAD, name_hint="lm_head_sample_merge", deps=[projected]):
        sampled_ids = _lm_head_sample_merge(partial_values, partial_indices, sampled_ids)
    return sampled_ids


@pl.jit.inline(auto_scope=False)
def _lm_head_has_rows(logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]):
    active = pl.cast(0, pl.INT32)
    for row in pl.range(MAX_LOGIT_ROWS):
        if active == 0:
            source_row = pl.read(logit_row_indices, [row])
            if source_row >= 0:
                active = pl.cast(1, pl.INT32)
    return active


@pl.jit.inline(auto_scope=False)
def _lm_head_zero_outputs(logits: pl.Tensor, sampled_ids: pl.Tensor):
    with pl.spmd((VOCAB + LOGITS_COMM_TILE - 1) // LOGITS_COMM_TILE, name_hint="lm_head_empty"):
        block = pl.tile.get_block_idx()
        col = block * LOGITS_COMM_TILE
        cols = pl.min(LOGITS_COMM_TILE, VOCAB - col)
        zeros = pl.tile.full([MAX_LOGIT_ROWS, LOGITS_COMM_TILE], dtype=pl.FP32, value=0.0)
        valid_zeros = pl.set_validshape(zeros, MAX_LOGIT_ROWS, cols)
        logits = pl.store(valid_zeros, [0, col], logits)
        if block == 0:
            empty_ids = pl.tile.full([MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            sampled_ids = pl.store(empty_ids, [0, 0], sampled_ids)
    return logits, sampled_ids


@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Project logits and sample top-1 tokens in one opaque L2 entry."""
    active = pl.cast(1, pl.INT32)
    if LM_HEAD_TP_SIZE == 1:
        active = _lm_head_has_rows(logit_row_indices)
    if active > 0:
        logits, projected = lm_head(
            hidden_states, lm_head_weight, logit_row_indices, logits, hidden_window, hidden_done,
            logits_window, logits_done, group_base, tp_rank, done_epoch, completed_epochs, 1,
        )
        greedy_sample(logits, sampled_ids, projected)
    else:
        logits, sampled_ids = _lm_head_zero_outputs(logits, sampled_ids)
    return logits, sampled_ids


@pl.jit
def l2_lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Standalone opaque entry for projection plus greedy sampling tests."""
    return lm_head_with_sampling(
        hidden_states, lm_head_weight, logit_row_indices, logits, sampled_ids, hidden_window,
        hidden_done, logits_window, logits_done, group_base, tp_rank, done_epoch, completed_epochs,
    )


@pl.jit.host
def l3_final_output(
    x_hc: pl.Tensor[[WORLD_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[WORLD_SIZE, T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[WORLD_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, D, VOCAB_PER_TP], pl.BF16],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    """Collapse final HC state, normalize, project and sample. Complete all ranks before reusing windows."""
    x_hc.bind_dynamic(1, T_DYN)
    hidden_buffer = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    logits_buffer = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.BF16)
    logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_buffer, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_buffer, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.BF16)
        logits_done = pld.window(logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_final_output(
            x_hc[rank], pre_mix[rank], final_norm_weight[rank], lm_head_weight[rank],
            logit_row_indices[rank], logits[rank], sampled_ids[rank], hidden_window, hidden_done,
            logits_window, logits_done, rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE, rank % LM_HEAD_TP_SIZE,
            1, 1, device=rank,
        )


def _final_output(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.BF16],
    logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Compose final HC, Norm and LM head from the current layer state."""
    x_hc.bind_dynamic(0, T_DYN)
    active = pl.cast(1, pl.INT32)
    if LM_HEAD_TP_SIZE == 1:
        active = _lm_head_has_rows(logit_row_indices)
    if active > 0:
        rows = pl.tensor.dim(x_hc, 0)
        collapsed = pl.create_tensor([rows, D], dtype=pl.BF16)
        hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
        normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
        rms_norm(collapsed, final_norm_weight, normalized, rows, width=D, zero_tail=False)
        lm_head_with_sampling(
            normalized, lm_head_weight, logit_row_indices, logits, sampled_ids, hidden_window, hidden_done,
            logits_window, logits_done, group_base, tp_rank, done_epoch, completed_epochs,
        )
    else:
        logits, sampled_ids = _lm_head_zero_outputs(logits, sampled_ids)
    return logits, sampled_ids


final_output = pl.jit.inline(_final_output)
l2_final_output = pl.jit(_final_output)


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, D, VOCAB_PER_TP], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
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
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r], sampled_ids[r],
            hidden_window, hidden_done, logits_window, logits_done, r // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
            r % LM_HEAD_TP_SIZE, 1, 1, device=r,
        )


@pl.jit(auto_scope=False)
def lm_head_sample_validation(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB], pl.BF16, pl.NZ],
    active_widths: pl.Tensor[[12], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[72, SAMPLED_IDS_PAD], pl.INT32]],
):
    ready = pl.system.task_dummy(deps=[])
    return lm_head_sample(hidden_states, lm_head_weight, active_widths, sampled_ids, ready)


def golden_lm_head_sample(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    weight = unpack_bf16_nz(tensors["lm_head_weight"]).float()
    rounded_logits = torch.matmul(hidden, weight).to(torch.bfloat16).float()
    ids = torch.argmax(rounded_logits, dim=-1).to(torch.int32)
    output = tensors["sampled_ids"]
    output.zero_()
    rows = min(hidden.shape[0], output.shape[0])
    row_indices = torch.arange(rows)
    active = row_indices % 6 < tensors["active_widths"][row_indices // 6]
    output[:rows, 0] = torch.where(active, ids[:rows], 0)


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
    if "sampled_ids" in tensors:
        tensors["sampled_ids"].zero_()
        tensors["sampled_ids"][:, :, 0] = torch.argmax(tensors["logits"], dim=-1).to(torch.int32)


def golden_final_output(tensors):
    """Evaluate HC collapse, BF16 Norm and head projection independently on CPU."""
    import torch

    hidden = []
    for rank in range(WORLD_SIZE):
        collapsed = golden_hc_collapse(tensors["x_hc"][rank], tensors["pre_mix"][rank]).to(torch.bfloat16)
        normalized = golden_rms_norm_tiled(collapsed, tensors["final_norm_weight"][rank])
        hidden.append(normalized.to(torch.bfloat16))
    golden_lm_head({**tensors, "hidden_states": torch.stack(hidden)})


def build_sample_tensor_specs():
    import torch
    from golden import TensorSpec

    tokens = 47

    def init_hidden():
        hidden = torch.eye(tokens, D, dtype=torch.bfloat16)
        hidden[:4, tokens] = 1.0 / 1024
        return hidden

    def init_weight():
        weight = torch.zeros(D, VOCAB, dtype=torch.bfloat16)
        for row in range(tokens):
            if row % 3 == 0:
                weight[row, 4096 + row] = 2
                weight[row, 16384 + row] = 2
            elif row % 3 == 1:
                weight[row, VOCAB - 1 - row // 3] = 3
            else:
                weight[row, 512 * row + 31] = 4
        # FP32-separated logits that tie after BF16 rounding.
        for row, token_pair in enumerate(((1, 2), (1, 513), (4096, 16384), (1, VOCAB - 1))):
            weight[row].zero_()
            weight[row, token_pair[0]] = 1.0
            weight[row, token_pair[1]] = 1.0
            weight[tokens, token_pair[1]] = 1.0
        return pack_bf16_nz(weight)

    def init_widths():
        return torch.tensor([6, 3, 0, 1, 5, 2, 6, 6, 6, 0, 4, 6], dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [tokens, D], torch.bfloat16, init_value=init_hidden),
        TensorSpec("lm_head_weight", [D, VOCAB], torch.bfloat16, init_value=init_weight),
        TensorSpec("active_widths", [12], torch.int32, init_value=init_widths),
        TensorSpec("sampled_ids", [72, SAMPLED_IDS_PAD], torch.int32),
    ]


def build_tensor_specs(num_tokens=MAX_LOGIT_ROWS, *, final=False):
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
        if final and DP_SIZE > 1:
            indices[-LM_HEAD_TP_SIZE:] = -1
        return indices

    specs = [
        TensorSpec("hidden_states", [WORLD_SIZE, MAX_LOGIT_ROWS, D], torch.bfloat16, init_value=init_hidden_states),
        # Resident vocabulary shard per world rank.
        TensorSpec("lm_head_weight", [WORLD_SIZE, D, VOCAB_PER_TP], torch.bfloat16, init_value=init_lm_head_weight, resident="stacked"),
        TensorSpec("logits", [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], torch.float32),
        TensorSpec("sampled_ids", [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32),
        TensorSpec("logit_row_indices", [WORLD_SIZE, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
    ]
    if final:
        specs = [
            TensorSpec("x_hc", [WORLD_SIZE, MAX_LOGIT_ROWS, HC_MULT, D], torch.float32, init_value=torch.randn),
            TensorSpec("pre_mix", [WORLD_SIZE, MAX_LOGIT_ROWS, HC_MULT], torch.float32, init_value=torch.rand),
            TensorSpec("final_norm_weight", [WORLD_SIZE, D], torch.bfloat16, init_value=torch.ones),
            specs[1], specs[4], specs[2], specs[3],
        ]
    return specs


def compare_sampled_ids(actual, expected, *, actual_outputs, **_):
    import torch

    logits_ids = torch.zeros_like(actual)
    logits_ids[:, :, 0] = torch.argmax(actual_outputs["logits"].cpu(), dim=-1).to(torch.int32)
    if torch.equal(actual, expected) and torch.equal(actual, logits_ids):
        return True, ""
    mismatch = actual != expected
    return False, (
        f"sampled_ids mismatch: bad={int(mismatch.sum())}/{actual.numel()} "
        f"actual={actual.tolist()} expected={expected.tolist()}"
    )


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
    parser.add_argument("--final-output", action="store_true", help="include final HC collapse and Norm")
    parser.add_argument("--fused-sample", action="store_true", help="validate only the TP1 fused sampler")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.execution_mode and args.lm_head_tp > 1:
        parser.error("LM-head TP > 1 requires device Cube/Vector intra-block signaling")

    device_ids = [int(d) for d in args.device.split(",")]
    if args.fused_sample and (args.execution_mode or args.final_output):
        parser.error("--fused-sample requires device execution without --final-output")
    required_devices = 1 if args.fused_sample else WORLD_SIZE
    if len(device_ids) < required_devices:
        parser.error(f"need at least {required_devices} devices, got {device_ids}")
    if args.lm_head_tp != LM_HEAD_TP_SIZE or args.dp != DP_SIZE:
        parser.error("--lm-head-tp and --dp must match the import-time configuration")
    if not 0 <= args.num_tokens <= MAX_LOGIT_ROWS:
        parser.error(f"--num-tokens must be between 0 and {MAX_LOGIT_ROWS}")

    if not args.fused_sample:
        result = run(
            fn=l3_final_output if args.final_output else l3_lm_head,
            specs=build_tensor_specs(args.num_tokens, final=args.final_output),
            golden_fn=golden_final_output if args.final_output else golden_lm_head,
            compare_fn={
                "logits": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
                "sampled_ids": compare_sampled_ids,
            },
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            save_data=args.save_data,
            config=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig( device_ids=device_ids[:required_devices], num_sub_workers=0, ),
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

    # Default device validation also covers the TP1 sampler on the first card.
    default_sample = not (args.execution_mode or args.final_output or args.runtime_dir)
    if args.fused_sample or default_sample:
        result = run(
            fn=lm_head_sample_validation,
            specs=build_sample_tensor_specs(),
            golden_fn=golden_lm_head_sample,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir if args.fused_sample else None,
            config=dict(
                dump_passes=args.dump_passes, backend_type=args.backend_type, execution_mode=False,
                device_id=device_ids[0], enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=0,
            atol=0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
