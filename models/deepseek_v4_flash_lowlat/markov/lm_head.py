# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M, MOE_TOKENS

D = M.hidden_size

VOCAB = M.vocab_size

MAX_LOGIT_ROWS = MOE_TOKENS

_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT

VOCAB_PER_TP = VOCAB // TP_SIZE

GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

FUSED_K_TILE = 256

FUSED_VOCAB_TILE = 128

MM_ROW_TILE = 32

HIDDEN_GATHER_TILE = 1024

HIDDEN_GATHER_ROW_TILE = min(GROUP_LOGIT_ROWS, 64)

PUSH_ROW_TILE = 16  # rows pushed per dispatch_push block (fewer, larger ring puts)

LOGITS_GATHER_ROW_TILE = min(MAX_LOGIT_ROWS, 8)

LOGITS_COMM_TILE = 2048

GREEDY_ROW_WIDTH = 808

GREEDY_BLOCK_ROWS = 8

FUSED_LM_HEAD_CORES = 24

AIV_LANES = 2

LANE_ROWS = MM_ROW_TILE // AIV_LANES

VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE

VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE

LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE

N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE

LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))

LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

GREEDY_INDEX_SENTINEL = 1073741824

DONE_VALUE = 1

assert MAX_LOGIT_ROWS % MM_ROW_TILE == 0, "each row block must be one owner's rows"

assert MAX_LOGIT_ROWS % PUSH_ROW_TILE == 0, "row block must cover whole rows"

assert MAX_LOGIT_ROWS % LOGITS_GATHER_ROW_TILE == 0, "logits row blocks must cover whole rows"

assert TP_SIZE == 1 or TP_SIZE % AIV_LANES == 0, "peer owners must pair across the AIV lanes"

assert GREEDY_BLOCK_ROWS * 4 % 32 == 0, "reduction result must clear the 32 B column floor"

assert GREEDY_ROW_WIDTH * 4 % 32 == 0, "block row must clear the 32 B row floor"

assert VOCAB < GREEDY_INDEX_SENTINEL, "sentinel must lose every row_min against a real id"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
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

    # Project the owner0 row block on every TP vocabulary shard.
    # Fused cube+comm kernel: matmul one [MM_ROW_TILE, FUSED_VOCAB_TILE] tile,
    # carry the accumulator across the C->V edge with pl.aiv_shard, and
    # remote_store each lane's row half to its owner's window while the cube
    # projects the next tile. No GM staging: every row block is one owner's rows,
    # so each lane's shard is already the exact rows it pushes.
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        deps=[_dgather_tid],
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(lm_head_weight, pl.CachePolicy.BYPASS)
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_FULL_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            for mm_rb in pl.range(MAX_LOGIT_ROWS // MM_ROW_TILE):
                mm_r0 = mm_rb * MM_ROW_TILE
                mm_acc = pl.create_tensor([MM_ROW_TILE, FUSED_VOCAB_TILE], dtype=pl.FP32)
                for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                    mm_k0 = mm_kb * FUSED_K_TILE
                    mm_hidden_tile = owner_hiddens[mm_r0 : mm_r0 + MM_ROW_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_weight_tile = lm_head_weight[
                        mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE
                    ]
                    mm_acc = pl.matmul_acc(
                        mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0)
                    )

                # The block is one owner's rows; the two lanes push its two
                # contiguous halves straight to that owner's window.
                for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                    mm_shard = pl.aiv_shard(mm_acc)
                    owner_tp = mm_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                    owner_r0 = (mm_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                    pld.tensor.remote_store(
                        mm_shard,
                        logits_window,
                        group_base + owner_tp,
                        [owner_r0, vocab_base + mm_o0],
                    )

        # Ragged tail: its own matmul on one block; the accumulator is already
        # VOCAB_TAIL wide. Hoisted out of the strided loop: a dynamic branch would
        # give the accumulator two static shapes across the C->V edge.
        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_FULL_TILES % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_FULL_TILES * FUSED_VOCAB_TILE
                for tail_rb in pl.range(MAX_LOGIT_ROWS // MM_ROW_TILE):
                    tail_r0 = tail_rb * MM_ROW_TILE
                    tail_acc = pl.create_tensor([MM_ROW_TILE, VOCAB_TAIL], dtype=pl.FP32)
                    for tail_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        tail_k0 = tail_kb * FUSED_K_TILE
                        tail_hidden_tile = owner_hiddens[
                            tail_r0 : tail_r0 + MM_ROW_TILE, tail_k0 : tail_k0 + FUSED_K_TILE
                        ]
                        tail_weight_tile = lm_head_weight[
                            mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                            tail_k0 : tail_k0 + FUSED_K_TILE,
                        ]
                        tail_acc = pl.matmul_acc(
                            tail_acc,
                            tail_hidden_tile,
                            tail_weight_tile,
                            b_trans=True,
                            init_cond=(tail_kb == 0),
                        )

                    for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                        tail_shard = pl.aiv_shard(tail_acc)
                        owner_tp = tail_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        owner_r0 = (
                            tail_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        ) * MM_ROW_TILE + aiv_id * LANE_ROWS
                        pld.tensor.remote_store(
                            tail_shard,
                            logits_window,
                            group_base + owner_tp,
                            [owner_r0, vocab_base + mm_tail_o0],
                        )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(TP_SIZE // AIV_LANES):
                owner_tp = notify_pair * AIV_LANES + aiv_id
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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                for gr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                    if tp_rank == 0:
                        logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = logits_window[
                            gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE
                        ]
                    else:
                        zero_logits = pl.full(
                            [LOGITS_GATHER_ROW_TILE, LOGITS_COMM_TILE], dtype=pl.FP32, value=0.0
                        )
                        logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = zero_logits

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    for tr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                        if tp_rank == 0:
                            logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = (
                                logits_window[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL]
                            )
                        else:
                            zero_tail = pl.full(
                                [LOGITS_GATHER_ROW_TILE, LOGITS_COMM_TAIL], dtype=pl.FP32, value=0.0
                            )
                            logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = zero_tail

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
