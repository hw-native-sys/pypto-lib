# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 LM head projection with DP-owned hidden and TP vocab shards.

Hidden states must already have passed the final RMSNorm.

The DP world is ``--dp`` groups of ``--tp`` TP ranks each. Every card is both an owner and
a TP rank: it holds vocab shard ``rank % TP_SIZE`` and serves only its own group,
so every ``peer`` is ``group_base + tp_rank``.

Dispatch all-gathers the hidden rows, the matmul projects every group row against
this card's vocab shard, and combine all-to-alls the logits so each owner ends up
with its own rows over the full vocabulary. Both collectives are a push (peer
``pld.tensor.put``) plus a folded notify, a wait-only scope, and a parallel
gather; the barrier's ``expected`` therefore scales with the pushing scope's
block count.

Per-card cost tracks ``VOCAB_PER_TP``, not the DP world size: the matmul M extent
is always ``TP_SIZE * MAX_LOGIT_ROWS``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M


T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# Model
D = M.hidden_size
VOCAB = M.vocab_size

# Parallelism. Static in the frontend, so both worlds are parsed off argv here.
_TP_CHOICES = (2, 4, 8, 16)
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
# --dp sizes the standalone l3_lm_head fixture: how many DP groups it builds.
# The kernel itself carries no DP extent, so composed callers never pass it.
DP_SIZE: int = _parse_int_argv("--dp") or 1
VOCAB_PER_TP = VOCAB // TP_SIZE
WORLD_SIZE = TP_SIZE * DP_SIZE

# Rows. Decode specializations override DECODE_TOKENS before importing this
# module; standalone and prefill callers retain the checked-in eight-row tile.
# logit_row_indices picks the sources and unused rows stay zero.
MAX_LOGIT_ROWS = DECODE_TOKENS
TEST_TOKENS = 16  # standalone fixture: hidden rows per card, > MAX_LOGIT_ROWS
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

# Tiling. Both matmul tiles clear the 512 B contiguous-transfer floor: the
# K-contiguous weight slice moves FUSED_K_TILE * 2 B and the FP32 logits store
# moves FUSED_VOCAB_TILE * 4 B.
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 256
HIDDEN_GATHER_TILE = 512
LOGITS_COMM_TILE = 2048
# The vocab shard rarely divides the matmul tile, so the last tile is ragged. It
# rides the same loop as every other tile rather than getting a tail case: the
# weight load carries a valid_shape, which is what keeps the load inside
# lm_head_weight, and the tile count is a plain ceil. The narrow extent is then
# dropped at the crossing and re-applied at the push -- see the two comments in
# the fused scope. logits_shards is padded to whole tiles so the last tile's
# full-width store stays in bounds.
VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE
VOCAB_LAST_TILE = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_TILES = VOCAB_FULL_TILES + (1 if VOCAB_LAST_TILE != 0 else 0)
SHARDS_VOCAB = VOCAB_TILES * FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
FUSED_LM_HEAD_CORES = 24
# AIV lanes per AICore. A mode=NONE split region runs its body on both, so the
# fused kernel shards owners across them to keep every push and notify single.
AIV_LANES = 2
OWNER_PAIRS = TP_SIZE // AIV_LANES
# The shard cannot be sliced, so it lands whole in a GM scratch and the per-owner
# pushes slice that scratch instead -- a plain tensor slice, which is legal.
SHARD_ROWS = GROUP_LOGIT_ROWS // AIV_LANES
OWNERS_PER_LANE = TP_SIZE // AIV_LANES
DONE_VALUE = 1

# Greedy sampling uses exact 256-token chunks so the real vocabulary has no
# padded tail. The 505 chunk maxima are padded to 512 for the final merge sort.
GREEDY_VOCAB_CHUNK = 256
GREEDY_NUM_VOCAB_CHUNKS = VOCAB // GREEDY_VOCAB_CHUNK
GREEDY_CHUNK_PAD = 512
GREEDY_TOPK = 16
SAMPLED_IDS_PAD = 8

# Combine blocks: one per vocab comm tile, capped at the core count; the tail tile
# rides the block the strided loop hands it next. Raising the cap does not help --
# the push is cross-card bandwidth bound, not core bound.
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(
    FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0)
)
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

assert TP_SIZE % AIV_LANES == 0, "owners must divide evenly across the AIV lanes"
# What keeps the GM staging race-free: an UP_DOWN lane writes exactly the row
# band it later reads, and the two lanes' bands are disjoint, so no GM address is
# touched by both lanes and the store->push RAW stays inside one lane. That holds
# only while a lane's row half equals the rows of the owners it serves -- assert
# it rather than rely on the coincidence.
assert SHARD_ROWS == OWNERS_PER_LANE * MAX_LOGIT_ROWS, (
    "each AIV lane's shard rows must be exactly the rows of the owners it pushes"
)
assert D % FUSED_K_TILE == 0
assert D % HIDDEN_GATHER_TILE == 0
assert VOCAB % TP_SIZE == 0
assert VOCAB % GREEDY_VOCAB_CHUNK == 0
assert GREEDY_NUM_VOCAB_CHUNKS <= GREEDY_CHUNK_PAD
assert GROUP_LOGIT_ROWS % 16 == 0, "matmul M extent must be a multiple of 16"
assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"
assert DP_SIZE in _DP_CHOICES, f"--dp must be one of {_DP_CHOICES} (got {DP_SIZE})"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_dispatch_push"):
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        source_row_raw = pl.read(logit_row_indices, [row])
        # Clamp so the load address is always inside hidden_states even if a
        # caller hands over a stale index; the -1 guard below decides whether
        # the row is actually used.
        safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
        # Full-width [1, D] tile: the block owns the whole row.
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
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + row, 0],
                src_offsets=[row, 0],
                shape=[1, D],
            )

        # Notify folded into the push: MAX_LOGIT_ROWS notifies per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Barrier on the group's publishes. The hidden_states read is an anchor, not
    # data: it deps this task on the hidden-state producer so the wait runs
    # alongside our own push instead of trailing it.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait") as _dwait_tid:
        _hidden_anchor = pl.read(hidden_states, [0, 0])
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * MAX_LOGIT_ROWS, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    with pl.spmd(
        D // HIDDEN_GATHER_TILE, name_hint="lm_head_dispatch_gather", deps=[_dwait_tid]
    ) as _dgather_tid:
        gkb = pl.tile.get_block_idx()
        gk0 = gkb * HIDDEN_GATHER_TILE
        owner_hiddens[:, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[:, gk0 : gk0 + HIDDEN_GATHER_TILE]

    # Mixed cube + comm kernel: the cube projects a vocab tile, pl.aiv_shard
    # carries that accumulator across the C->V edge, and pld.tensor.remote_store
    # pushes each owner's rows to that peer's window from the vector lane. Each
    # tile therefore goes on the wire while the cube projects the next one,
    # instead of the whole shard waiting for the last matmul -- that overlap is
    # the point of fusing the projection and the push into one scope.
    # The ragged last tile is just a padded tile in the same loop: its weight load
    # carries valid_shape, its padded columns ride to GM as don't-care, and the
    # push is narrowed back to the real width.
    logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, SHARDS_VOCAB], dtype=pl.FP32)
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            # Real width of this tile: the full tile everywhere except the ragged
            # last one, where the weight rows run out early. The box stays static
            # so the accumulator keeps one shape for every iteration.
            mm_valid_n = pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE)
            # M is the full group extent -- one matmul per vocab tile.
            mm_hidden0 = owner_hiddens[:, 0:FUSED_K_TILE]
            mm_weight0 = pl.slice(
                lm_head_weight,
                [FUSED_VOCAB_TILE, FUSED_K_TILE],
                [mm_o0, 0],
                valid_shape=[mm_valid_n, FUSED_K_TILE],
            )
            mm_acc = pl.matmul(mm_hidden0, mm_weight0, b_trans=True, out_dtype=pl.FP32)
            for mm_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                mm_k0 = mm_kb * FUSED_K_TILE
                mm_hidden_tile = owner_hiddens[:, mm_k0 : mm_k0 + FUSED_K_TILE]
                mm_weight_tile = pl.slice(
                    lm_head_weight,
                    [FUSED_VOCAB_TILE, FUSED_K_TILE],
                    [mm_o0, mm_k0],
                    valid_shape=[mm_valid_n, FUSED_K_TILE],
                )
                mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True)

            # Declare the columns fully valid before the C->V crossing. The
            # boundary transports the full box and can only rebuild a narrowed
            # column extent with the compile-time-static pto.treshape, so a
            # runtime-valued one is rejected outright. The padded columns of the
            # ragged tile are don't-care here: they ride to GM and the push below
            # is what cuts them off, so they never reach a peer's window.
            mm_acc = pl.set_validshape(mm_acc, GROUP_LOGIT_ROWS, FUSED_VOCAB_TILE)

            # WORKAROUND -- revert once a subview of a pl.aiv_shard result is
            # legal (filed against ptoas). The shard cannot be sliced, so it is
            # stored whole to GM and the per-owner pushes read slices back out.
            # That costs an extra GM write and an extra GM read of every tile,
            # traffic a direct shard-to-peer push would not pay at all. It is
            # still the cheaper option: pinning the matmul M extent to one owner
            # per lane instead -- the only slice-free alternative -- triples cube
            # time at tp=8 and turns the fusion into a regression there.
            #
            # Race-free by construction: UP_DOWN gives each lane a disjoint row
            # band that it both writes and reads, so no GM address is touched by
            # both lanes. The SHARD_ROWS assert above is what keeps that true.
            for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                mm_shard = pl.aiv_shard(mm_acc)
                lane_r0 = aiv_id * SHARD_ROWS
                logits_shards[
                    lane_r0 : lane_r0 + SHARD_ROWS, mm_o0 : mm_o0 + FUSED_VOCAB_TILE
                ] = mm_shard
                for owner_slot in pl.range(OWNERS_PER_LANE):
                    owner_tp = aiv_id * OWNERS_PER_LANE + owner_slot
                    src_r0 = owner_tp * MAX_LOGIT_ROWS
                    # valid_shape rather than a narrower box: the box has to stay
                    # static for the tile type, and a full-width push of the ragged
                    # tile would run past this rank's slice of the shared window
                    # into the next rank's. pl.min is inlined instead of reusing
                    # mm_valid_n because a named cube-side scalar cannot be
                    # materialized inside the AIV function.
                    pld.tensor.remote_store(
                        pl.slice(
                            logits_shards,
                            [MAX_LOGIT_ROWS, FUSED_VOCAB_TILE],
                            [src_r0, mm_o0],
                            valid_shape=[
                                MAX_LOGIT_ROWS,
                                pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE),
                            ],
                        ),
                        logits_window,
                        group_base + owner_tp,
                        [0, vocab_base + mm_o0],
                    )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(OWNER_PAIRS):
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
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]
    ) as _cwait_tid:
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
    with pl.spmd(
        LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]
    ) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                logits[:, lo : lo + LOGITS_COMM_TILE] = logits_window[:, lo : lo + LOGITS_COMM_TILE]

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    logits[:, tl : tl + LOGITS_COMM_TAIL] = logits_window[:, tl : tl + LOGITS_COMM_TAIL]

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear"):
        _completion_anchor = pl.read(logits, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits


@pl.jit
def lm_head_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch,
    )
    return logits


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Select the first maximum token id from each full-vocabulary logits row."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_greedy_sample"):
        chunk_idx_init = pl.arange(0, [1, GREEDY_VOCAB_CHUNK], dtype=pl.UINT32)
        chunk_maxima = pl.create_tensor([1, GREEDY_CHUNK_PAD], dtype=pl.FP32)
        chunk_maxima[:, :] = pl.full(
            [1, GREEDY_CHUNK_PAD],
            dtype=pl.FP32,
            value=-3.402823e38,
        )
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            chunk_start = chunk * GREEDY_VOCAB_CHUNK
            scores = logits[
                row : row + 1,
                chunk_start : chunk_start + GREEDY_VOCAB_CHUNK,
            ]
            sorted_pairs = pl.sort32(scores, chunk_idx_init)
            sorted_pairs = pl.mrgsort(sorted_pairs, block_len=64)
            sorted_pairs = pl.mrgsort(
                sorted_pairs[:, 0:GREEDY_VOCAB_CHUNK],
                sorted_pairs[:, GREEDY_VOCAB_CHUNK : 2 * GREEDY_VOCAB_CHUNK],
            )
            top_pair = sorted_pairs[:, 0 : 2 * GREEDY_TOPK]
            top_values = pl.gather(top_pair, mask_pattern=pl.tile.MaskPattern.P0101)
            pl.write(chunk_maxima, [0, chunk], pl.read(top_values, [0, 0]))

        maxima_idx_init = pl.arange(0, [1, GREEDY_CHUNK_PAD], dtype=pl.UINT32)
        sorted_maxima = pl.sort32(chunk_maxima, maxima_idx_init)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=64)
        sorted_maxima = pl.mrgsort(sorted_maxima, block_len=256)
        top_maximum_pair = sorted_maxima[:, 0 : 2 * GREEDY_TOPK]
        top_maximum_values = pl.gather(
            top_maximum_pair,
            mask_pattern=pl.tile.MaskPattern.P0101,
        )
        best_value = pl.read(top_maximum_values, [0, 0])

        # Reverse scans leave the lowest matching index selected, matching
        # torch.argmax's first-occurrence tie behavior.
        winning_chunk = pl.cast(0, pl.INT32)
        for chunk in pl.range(GREEDY_NUM_VOCAB_CHUNKS):
            scan_chunk = GREEDY_NUM_VOCAB_CHUNKS - 1 - chunk
            if pl.read(chunk_maxima, [0, scan_chunk]) == best_value:
                winning_chunk = pl.cast(scan_chunk, pl.INT32)

        chunk_base = winning_chunk * pl.cast(GREEDY_VOCAB_CHUNK, pl.INT32)
        winning_scores = pl.slice(
            logits,
            [1, GREEDY_VOCAB_CHUNK],
            [pl.cast(row, pl.INDEX), pl.cast(chunk_base, pl.INDEX)],
        )
        winning_offset = pl.cast(0, pl.INT32)
        for offset in pl.range(GREEDY_VOCAB_CHUNK):
            scan_offset = GREEDY_VOCAB_CHUNK - 1 - offset
            if pl.read(winning_scores, [0, scan_offset]) == best_value:
                winning_offset = pl.cast(scan_offset, pl.INT32)
        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_row[:, :] = pl.full(
            [1, SAMPLED_IDS_PAD],
            dtype=pl.INT32,
            value=0,
        )
        pl.write(sampled_row, [0, 0], chunk_base + winning_offset)
        sampled_ids[row : row + 1, :] = sampled_row

    return sampled_ids


@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Project logits and sample top-1 tokens in one opaque L2 entry."""
    lm_head(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        logits,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        done_epoch,
    )
    greedy_sample(logits, sampled_ids)
    return logits, sampled_ids


@pl.jit
def lm_head_with_sampling_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Standalone opaque entry for projection plus greedy sampling tests."""
    return lm_head_with_sampling(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        logits,
        sampled_ids,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        done_epoch,
    )


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[
        pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]
    ],
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
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_with_sampling_test(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            sampled_ids[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE, so concatenating shards in index order
    # reproduces the global vocabulary order every owner assembles.
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
    if "sampled_ids" in tensors:
        tensors["sampled_ids"].zero_()
        tensors["sampled_ids"][:, :, 0] = torch.argmax(
            tensors["logits"],
            dim=-1,
        ).to(torch.int32)


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
        TensorSpec(
            "hidden_states",
            [WORLD_SIZE, TEST_TOKENS, D],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        # One vocab shard per DP rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight",
            [WORLD_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            resident="stacked",
        ),
        TensorSpec(
            "logits",
            [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "sampled_ids",
            [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
            torch.int32,
            is_output=True,
        ),
        TensorSpec(
            "logit_row_indices",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_logit_row_indices,
        ),
    ]


def compare_logits(actual, expected, **_):
    import torch

    close = torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)
    if bool(close.all()):
        return True, ""
    lines = []
    for owner in range(actual.shape[0]):
        for shard in range(TP_SIZE):
            start = shard * VOCAB_PER_TP
            end = start + VOCAB_PER_TP
            shard_actual = actual[owner, :, start:end]
            shard_close = close[owner, :, start:end]
            lines.append(
                f"    owner={owner} shard={shard}: "
                f"bad={int((~shard_close).sum())}/{MAX_LOGIT_ROWS * VOCAB_PER_TP} "
                f"zeros={int((shard_actual == 0).sum())}"
            )
    return False, "\n".join(lines)


def compare_sampled_ids(actual, _expected, *, actual_outputs, **_):
    import torch

    expected = torch.zeros_like(actual)
    expected[:, :, 0] = torch.argmax(
        actual_outputs["logits"].cpu(),
        dim=-1,
    ).to(torch.int32)
    if torch.equal(actual, expected):
        return True, ""
    mismatch = actual != expected
    return False, (
        f"sampled_ids mismatch: bad={int(mismatch.sum())}/{actual.numel()} "
        f"actual={actual.tolist()} expected={expected.tolist()}"
    )


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES),
                        help="DP groups (world size = tp * dp)")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS,
                        help="Active hidden rows each owner projects")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(WORLD_SIZE)),
                        help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=(0, 1, 2, 4))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, (
        f"need at least {required_devices} devices, got {device_ids}"
    )
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs(args.num_tokens)
    golden_fn = golden_lm_head
    compare_fn = {
        "logits": compare_logits,
        "sampled_ids": compare_sampled_ids,
    }

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn=compare_fn,
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
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
