# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
# ci: no-sim    # CI marker: multi-card fixture; the fused C->V shard overflows the *sim UB limit -- device-only
"""DeepSeek-V4 Flash DSpark LM head: fused matmul+push projection with DP-owned hidden and TP vocab shards."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M, FP32_NEG_INF


T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size
MAX_LOGIT_ROWS = DECODE_TOKENS

# parallelism
_TP_CHOICES = (1, 2, 4, 8, 16)
_DP_CHOICES = (1, 2, 4, 8, 16)
_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT
# --dp sizes the standalone fixture: DP groups of TP ranks each; the
# kernel carries no DP extent.
DP_SIZE: int = _parse_int_argv("--dp") or 1
WORLD_SIZE = TP_SIZE * DP_SIZE
VOCAB_PER_TP = VOCAB // TP_SIZE
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
TEST_TOKENS = 2 * MAX_LOGIT_ROWS  # standalone fixture: hidden rows per card

# tiling
# MM_ROW_TILE x FUSED_VOCAB_TILE fp32 keeps the accumulator inside the 128KiB
# Acc space and divides MAX_LOGIT_ROWS: every row block is one owner's rows.
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 256
MM_ROW_TILE = 64
HIDDEN_GATHER_TILE = 512
HIDDEN_GATHER_ROW_TILE = min(GROUP_LOGIT_ROWS, 16)
PUSH_ROW_TILE = 16  # rows pushed per dispatch_push block (fewer, larger ring puts)
LOGITS_GATHER_ROW_TILE = min(MAX_LOGIT_ROWS, 8)
LOGITS_COMM_TILE = 2048
# Greedy sampling scans each row as a [GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH] grid.
# 8 rows keeps a reduction result at 32 B; alloc_tile rejects the 4 B a [1, 1] gives.
GREEDY_ROW_WIDTH = 808
GREEDY_BLOCK_ROWS = 8
SAMPLED_IDS_PAD = 8
FUSED_LM_HEAD_CORES = 24
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_OWNER_ROW_BLOCKS = MAX_LOGIT_ROWS // LOGITS_GATHER_ROW_TILE
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS
LOGITS_FLAT_PUT_TILE = 16384
LOGITS_FLAT_PUT_TILES = VOCAB_PER_TP // LOGITS_FLAT_PUT_TILE
LOGITS_FLAT_PUT_TAIL = VOCAB_PER_TP % LOGITS_FLAT_PUT_TILE
GREEDY_GRID_ROWS = VOCAB // GREEDY_ROW_WIDTH
GREEDY_BLOCK_SPAN = GREEDY_BLOCK_ROWS * GREEDY_ROW_WIDTH
# 2^30: above every vocab id, clear of int32 overflow once a block base is added.
GREEDY_INDEX_SENTINEL = 1073741824
DONE_VALUE = 1
LM_HEAD_RING_HEAP = 1 << 30

assert MAX_LOGIT_ROWS % MM_ROW_TILE == 0, "each row block must be one owner's rows"
assert MAX_LOGIT_ROWS % PUSH_ROW_TILE == 0, "row block must cover whole rows"
assert MAX_LOGIT_ROWS % LOGITS_GATHER_ROW_TILE == 0, "logits row blocks must cover whole rows"
assert GREEDY_BLOCK_ROWS * 4 % 32 == 0, "reduction result must clear the 32 B column floor"
assert GREEDY_ROW_WIDTH * 4 % 32 == 0, "block row must clear the 32 B row floor"
assert VOCAB < GREEDY_INDEX_SENTINEL, "sentinel must lose every row_min against a real id"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS * VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    hidden_ready_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[
    pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    pl.Scalar[pl.TASK_ID],
]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    # One block per PUSH_ROW_TILE rows: PUSH_ROW_TILE-row ring puts instead
    # of one [1, D] put per row. This cuts the per-dispatch put and notify
    # counts (512 -> 32 per card) to the MTP fixture's scale; the earlier
    # per-row form stalled lm_head_dispatch_gather on persistent re-dispatch
    # (SCHEDULER_TIMEOUT S1 with the gather task's core never exiting).
    with pl.spmd(
        MAX_LOGIT_ROWS // PUSH_ROW_TILE,
        name_hint="lm_head_dispatch_push",
        deps=[hidden_ready_tid],
    ) as _dispatch_push_tid:
        blk = pl.tile.get_block_idx()
        r0 = blk * PUSH_ROW_TILE
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        for i in pl.range(PUSH_ROW_TILE):
            row = r0 + i
            source_row_raw = pl.read(logit_row_indices, [row])
            # Clamp so the load address is always inside hidden_states even if a
            # caller hands over a stale index; the -1 guard below decides whether
            # the row is actually used.
            safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
            selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if source_row_raw >= 0:
                source_row = pl.cast(safe_raw, target_type=pl.INDEX)
                selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        # Self-target rides the same put; put drains before the notify issues.
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=hidden_window,
                peer=group_base + peer_tp,
                src=selected_hidden,
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + r0, 0],
                src_offsets=[r0, 0],
                shape=[PUSH_ROW_TILE, D],
            )

        # Notify folded into the push: one notify per block per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Start the wait after this rank has published. Signal credits persist until
    # the final clear, so peer notifies may safely arrive before this task starts;
    # anchoring avoids an idle wait occupying a core group while the push is pending.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid]
    ) as _dwait_tid:
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * (MAX_LOGIT_ROWS // PUSH_ROW_TILE), pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    with pl.spmd(
        (GROUP_LOGIT_ROWS // HIDDEN_GATHER_ROW_TILE) * (D // HIDDEN_GATHER_TILE),
        name_hint="lm_head_dispatch_gather",
        deps=[_dwait_tid, _dispatch_push_tid],
    ) as _dgather_tid:
        gblock = pl.tile.get_block_idx()
        gk0 = (gblock % (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_TILE
        gr0 = (gblock // (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_ROW_TILE
        owner_hiddens[gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[
            gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE
        ]

    # Keep the rank-local vocabulary shard in GM before publishing it. The
    # current PyPTO pin exposes tensor.put for a Tensor source, but does not
    # expose a high-level Tensor form of tile.remote_store for aiv_shard output.
    logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, VOCAB_PER_TP], dtype=pl.FP32)
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul",
        deps=[_dgather_tid],
    ) as _matmul_tid:
        lm_core = pl.tile.get_block_idx()
        for mm_ob in pl.range(lm_core, VOCAB_FULL_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            for mm_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                mm_r0 = mm_rb * MM_ROW_TILE
                mm_hidden0 = owner_hiddens[mm_r0 : mm_r0 + MM_ROW_TILE, 0:FUSED_K_TILE]
                mm_weight0 = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, 0:FUSED_K_TILE]
                mm_acc = pl.matmul(mm_hidden0, mm_weight0, b_trans=True, out_dtype=pl.FP32)
                for mm_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                    mm_k0 = mm_kb * FUSED_K_TILE
                    mm_hidden_tile = owner_hiddens[mm_r0 : mm_r0 + MM_ROW_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_weight_tile = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True)
                logits_shards[
                    mm_r0 : mm_r0 + MM_ROW_TILE,
                    mm_o0 : mm_o0 + FUSED_VOCAB_TILE,
                ] = mm_acc

        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_FULL_TILES % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_FULL_TILES * FUSED_VOCAB_TILE
                for tail_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                    tail_r0 = tail_rb * MM_ROW_TILE
                    tail_hidden0 = owner_hiddens[tail_r0 : tail_r0 + MM_ROW_TILE, 0:FUSED_K_TILE]
                    tail_weight0 = lm_head_weight[mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL, 0:FUSED_K_TILE]
                    tail_acc = pl.matmul(tail_hidden0, tail_weight0, b_trans=True, out_dtype=pl.FP32)
                    for tail_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                        tail_k0 = tail_kb * FUSED_K_TILE
                        tail_hidden_tile = owner_hiddens[
                            tail_r0 : tail_r0 + MM_ROW_TILE, tail_k0 : tail_k0 + FUSED_K_TILE
                        ]
                        tail_weight_tile = lm_head_weight[
                            mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                            tail_k0 : tail_k0 + FUSED_K_TILE,
                        ]
                        tail_acc = pl.matmul_acc(tail_acc, tail_hidden_tile, tail_weight_tile, b_trans=True)
                    logits_shards[
                        tail_r0 : tail_r0 + MM_ROW_TILE,
                        mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                    ] = tail_acc

    # Flatten TP logits communication views.
    logits_shards_flat = pl.reshape(logits_shards, [GROUP_LOGIT_ROWS * VOCAB_PER_TP])
    with pl.spmd(
        LOGITS_OWNER_ROW_BLOCKS,
        name_hint="lm_head_combine_push",
        deps=[_matmul_tid],
    ) as _push_tid:
        row_block = pl.tile.get_block_idx()
        row_offset = row_block * LOGITS_GATHER_ROW_TILE
        vocab_base = tp_rank * VOCAB_PER_TP
        for owner_tp in pl.range(TP_SIZE):
            source_row_base = owner_tp * MAX_LOGIT_ROWS
            for row_lane in pl.unroll(8):
                row = row_offset + row_lane
                dst_row_base = row * VOCAB + vocab_base
                src_row_base = (source_row_base + row) * VOCAB_PER_TP
                for output_block in pl.range(LOGITS_FLAT_PUT_TILES):
                    output_offset = output_block * LOGITS_FLAT_PUT_TILE
                    pld.tensor.put(
                        dst=logits_window,
                        peer=group_base + owner_tp,
                        src=logits_shards_flat,
                        dst_offsets=[dst_row_base + output_offset],
                        src_offsets=[src_row_base + output_offset],
                        shape=[LOGITS_FLAT_PUT_TILE],
                    )

                if LOGITS_FLAT_PUT_TAIL != 0:
                    tail_offset = LOGITS_FLAT_PUT_TILES * LOGITS_FLAT_PUT_TILE
                    pld.tensor.put(
                        dst=logits_window,
                        peer=group_base + owner_tp,
                        src=logits_shards_flat,
                        dst_offsets=[dst_row_base + tail_offset],
                        src_offsets=[src_row_base + tail_offset],
                        shape=[LOGITS_FLAT_PUT_TAIL],
                    )

        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.notify(
                    target=logits_done,
                    peer=group_base + owner_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Wait only (the notify rides inside the push). deps on the push scope so the
    # wait runs alongside our own push; an unanchored wait dispatches immediately
    # and spins holding a core group.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]
    ) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * LOGITS_OWNER_ROW_BLOCKS, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(
        LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]
    ) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                for gr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                    for row_lane in pl.unroll(8):
                        row = gr + row_lane
                        flat_offset = row * VOCAB + lo
                        flat_logits = pl.load(logits_window, [flat_offset], [LOGITS_COMM_TILE])
                        logits_row = pl.reshape(flat_logits, [1, LOGITS_COMM_TILE])
                        pl.store(logits_row, [row, lo], logits)

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    for tr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                        for tail_lane in pl.unroll(8):
                            tail_row = tr + tail_lane
                            tail_flat_offset = tail_row * VOCAB + tl
                            flat_tail = pl.load(logits_window, [tail_flat_offset], [LOGITS_COMM_TAIL])
                            tail_logits_row = pl.reshape(flat_tail, [1, LOGITS_COMM_TAIL])
                            pl.store(tail_logits_row, [tail_row, tl], logits)

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_signal_clear",
        deps=[_gather_tid],
    ) as _clear_tid:
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, _clear_tid


@pl.jit
def l2_lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS * VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_input_ready",
    ) as hidden_ready_tid:
        _input_anchor = pl.read(hidden_states, [0, 0])
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, hidden_ready_tid,
    )
    return logits


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Select the first maximum token id from each full-vocabulary logits row."""
    # One pass per row into a [BLOCK_ROWS, ROW_WIDTH] accumulator carrying the
    # running maximum and the block that set it; a lane's column is its position.
    logits_grid = pl.reshape(logits, [MAX_LOGIT_ROWS * GREEDY_GRID_ROWS, GREEDY_ROW_WIDTH])
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_greedy_sample"):
        row_base = row * GREEDY_GRID_ROWS
        running_max = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
        running_base = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0)
        for block in pl.range(GREEDY_GRID_ROWS // GREEDY_BLOCK_ROWS):
            block_row = row_base + block * GREEDY_BLOCK_ROWS
            scores = logits_grid[block_row : block_row + GREEDY_BLOCK_ROWS, 0:GREEDY_ROW_WIDTH]
            # Strict greater-than, so a lane keeps the earliest block it peaked at.
            is_newer = pl.cmp(scores, running_max, cmp_type=4)
            newer = pl.cast(is_newer, target_type=pl.INT32)
            running_max = pl.maximum(running_max, scores)
            block_base = pl.cast(block * GREEDY_BLOCK_SPAN, pl.INT32)
            to_new = pl.neg(pl.sub(running_base, block_base))
            running_base = pl.add(running_base, pl.mul(newer, to_new))

        # Broadcast the lane maxima back and column-reduce: every entry is then the
        # row maximum. A scalar pl.max over the lanes miscompiles on fp32
        # (ptoas_bitcast has no float overload).
        lane_maxima = pl.row_max(running_max)
        lane_zeros = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=0.0)
        lane_broadcast = pl.row_expand_add(lane_zeros, lane_maxima)
        best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])

        # Flat index of every lane still at the row maximum, sentinel for the rest.
        # The lane * width term folds into the scalar combine below, keeping the
        # ramp a broadcast row rather than an (illegal) 2D arange.
        ramp_zeros = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0)
        column_ramp = pl.col_expand(ramp_zeros, pl.arange(0, [1, GREEDY_ROW_WIDTH], dtype=pl.INT32))
        flat_index = pl.add(running_base, column_ramp)
        is_max = pl.cmp(running_max, best_value, cmp_type=0)
        hit = pl.cast(is_max, target_type=pl.INT32)
        offset_index = pl.sub(flat_index, GREEDY_INDEX_SENTINEL)
        candidates = pl.add(pl.mul(hit, offset_index), GREEDY_INDEX_SENTINEL)
        lane_indices = pl.row_min(candidates)
        best_index = pl.read(lane_indices, [0, 0])
        for lane in pl.range(1, GREEDY_BLOCK_ROWS):
            lane_term = pl.cast(lane * GREEDY_ROW_WIDTH, pl.INT32)
            lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
            best_index = pl.min(best_index, lane_best)

        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
        pl.write(sampled_row, [0, 0], best_index)
        sampled_ids[row : row + 1, :] = sampled_row

    return sampled_ids


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    # Windows are group-local: hidden_window holds one row slot per group member,
    # and every card receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS * VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE; concatenating shards in index order
    # reproduces the global vocabulary order.
    weight = tensors["lm_head_weight"].float()
    full_weight = torch.cat([weight[tp] for tp in range(TP_SIZE)], dim=0)
    full_logits = []
    for owner_rank in range(WORLD_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs(num_tokens=TEST_TOKENS):
    import torch
    from golden import TensorSpec

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        shards = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return torch.stack([shards[r % TP_SIZE] for r in range(WORLD_SIZE)], dim=0)

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    return [
        TensorSpec("hidden_states", [WORLD_SIZE, TEST_TOKENS, D], torch.bfloat16, init_value=init_hidden_states),
        # One vocab shard per DP rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight", [WORLD_SIZE, VOCAB_PER_TP, D], torch.bfloat16,
            init_value=init_lm_head_weight, resident="stacked",
        ),
        TensorSpec("logits", [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], torch.float32, is_output=True),
        TensorSpec("logit_row_indices", [WORLD_SIZE, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES), help="DP groups (world size = tp * dp)")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS, help="Active hidden rows each owner projects")
    device_default = ",".join(str(i) for i in range(WORLD_SIZE))
    parser.add_argument("-d", "--device", type=str, default=device_default, help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=(0, 1, 2, 4))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, f"need at least {required_devices} devices, got {device_ids}"
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs(args.num_tokens)
    golden_fn = golden_lm_head

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:required_devices],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            ring_heap=LM_HEAD_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
