# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4.1 Flash FP32 vocabulary projection and greedy token selection.

Hidden states have already passed final RMSNorm. Head weights use packed BF16
NZ bytes with logical shape [D, VOCAB_PER_TP]; checkpoints retain [VOCAB, D].

The public selected-row interface has 16 row slots. TP1 selects and projects
locally. TP groups gather owner rows and exchange vocabulary shards within
that group, then retire communication signals after all readers complete.
Complete a dispatch on every rank before replacing shared inputs or reusing
its windows. The DSpark target consumes a separate 80-row masked batch.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

# Standalone LM-head validation defaults to TP2/DP1.

from config import DP_SIZE as CONFIG_DP_SIZE, FLASH as M, TP_SIZE as CONFIG_TP_SIZE
from hc import hc_head, golden_hc_collapse
from rmsnorm import rms_norm_tiled_compensated, golden_rms_norm_tiled
from quant import pack_bf16_nz, unpack_bf16_nz


# model config
D = M.hidden_size
VOCAB = M.vocab_size
LM_HEAD_WEIGHT_LAYOUT = "bf16_nz_kn"
LM_HEAD_WEIGHT_CACHE = getattr(pl.CachePolicy, "NOT_ALLOC_KEEP", pl.CachePolicy.DEFAULT)


T_DYN = pl.dynamic("LM_HEAD_T_DYN")
HC_MULT = M.hc_mult

# Parallelism. Static in the frontend, so both worlds are parsed off argv here.
# Imported kernels inherit TP from the composing program.
_TP_CHOICES = (1, 2, 4, 8)
_DP_CHOICES = (1, 2, 4, 8)
_TP_DEFAULT = 2 if __name__ == "__main__" else CONFIG_TP_SIZE


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT
# DP groups built by the l3_lm_head fixture; the kernel carries no DP extent.
DP_SIZE: int = _parse_int_argv("--dp") or (1 if __name__ == "__main__" else CONFIG_DP_SIZE)
WORLD_SIZE = TP_SIZE * DP_SIZE
VOCAB_PER_TP = VOCAB // TP_SIZE

# Small decode fixture size (not DECODE_MAX_TOKENS). TP1 * 16 == 16 satisfies
# the matmul M % 16 == 0 constraint on Pro's simpler path.
MAX_LOGIT_ROWS = 16
TEST_TOKENS = 16  # standalone fixture: hidden rows per card
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

# Tiling
FUSED_K_TILE = 128
FUSED_VOCAB_TILE = 512 if TP_SIZE < 8 else 256
LM_HEAD_M_TILE = 80
LM_HEAD_STAGE = 3
GROUP_M_TILE = min(LM_HEAD_M_TILE, GROUP_LOGIT_ROWS)
HIDDEN_GATHER_TILE = 512
LOGITS_COMM_TILE = 2048
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
FUSED_LM_HEAD_CORES = 32
# Repeated projections advance epochs and clear signals after the final projection.
_REPEAT_CHOICES = (1, 2, 4)
REPEAT: int = _parse_int_argv("--repeat") or 1

# Greedy reduction tiles and first-index selection.
GREEDY_VOCAB_TILE = 8192
GREEDY_ROW_TILE = 8
GREEDY_SEGMENT_TILE = GREEDY_VOCAB_TILE // GREEDY_ROW_TILE
GREEDY_INDEX_TILE = GREEDY_SEGMENT_TILE // GREEDY_ROW_TILE
GREEDY_WORKERS = 64
GREEDY_NEGATIVE_INF_BITS = -8388608
GREEDY_INDEX_SENTINEL = 1073741824
SAMPLED_IDS_PAD = 8

# Combine blocks: one per vocab comm tile, capped at the core count; the tail tile
# rides the block the strided loop hands it next. Raising the cap does not help --
# the push is cross-card bandwidth bound, not core bound.
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(
    FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0)
)
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

assert D % FUSED_K_TILE == 0
assert D % HIDDEN_GATHER_TILE == 0
assert VOCAB % TP_SIZE == 0
assert GROUP_LOGIT_ROWS % 16 == 0, "matmul M extent must be a multiple of 16"
assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"
assert DP_SIZE in _DP_CHOICES, f"--dp must be one of {_DP_CHOICES} (got {DP_SIZE})"
assert REPEAT in _REPEAT_CHOICES, f"--repeat must be one of {_REPEAT_CHOICES} (got {REPEAT})"


@pl.jit.inline(auto_scope=False)
def _lm_head_project_tile(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    m_tile: pl.constexpr,
    n_tile: pl.constexpr,
):
    first_a = pl.load(hidden_states, [row, 0], [m_tile, FUSED_K_TILE],
                      valid_shape=[valid_rows, FUSED_K_TILE], target_memory=pl.Mem.Mat)
    first_a_full = pl.set_validshape(first_a, m_tile, FUSED_K_TILE)
    first_b = pl.load(lm_head_weight, [0, col], [FUSED_K_TILE, n_tile],
                      target_memory=pl.Mem.Mat, cache=LM_HEAD_WEIGHT_CACHE)
    accum = pl.matmul(first_a_full, first_b, out_dtype=pl.FP32)
    for k in pl.pipeline(FUSED_K_TILE, D, FUSED_K_TILE, stage=LM_HEAD_STAGE):
        a = pl.load(hidden_states, [row, k], [m_tile, FUSED_K_TILE],
                    valid_shape=[valid_rows, FUSED_K_TILE], target_memory=pl.Mem.Mat)
        a_full = pl.set_validshape(a, m_tile, FUSED_K_TILE)
        b = pl.load(lm_head_weight, [k, col], [FUSED_K_TILE, n_tile],
                    target_memory=pl.Mem.Mat, cache=LM_HEAD_WEIGHT_CACHE)
        accum = pl.matmul_acc(accum, a_full, b)
    return accum


@pl.jit.inline(auto_scope=False)
def lm_head_project(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logits: pl.Tensor,
    m_tile: pl.constexpr,
):
    """Write vocabulary logits in place and return the completion task."""
    rows = pl.tensor.dim(logits, 0)
    with pl.spmd(FUSED_LM_HEAD_CORES, name_hint="lm_head_matmul") as projected:
        worker = pl.tile.get_block_idx()
        for row in pl.range(0, rows, m_tile):
            valid_rows = pl.min(m_tile, rows - row)
            tile_count = (VOCAB_PER_TP // FUSED_VOCAB_TILE + FUSED_LM_HEAD_CORES - 1 - worker) // FUSED_LM_HEAD_CORES
            accum = pl.tile.create([m_tile, FUSED_VOCAB_TILE], dtype=pl.FP32)
            for step in pl.pipeline(0, tile_count * (D // FUSED_K_TILE), 1, stage=LM_HEAD_STAGE):
                k = step % (D // FUSED_K_TILE) * FUSED_K_TILE
                col = (worker + step // (D // FUSED_K_TILE) * FUSED_LM_HEAD_CORES) * FUSED_VOCAB_TILE
                a = pl.load(hidden_states, [row, k], [m_tile, FUSED_K_TILE],
                            valid_shape=[valid_rows, FUSED_K_TILE], target_memory=pl.Mem.Mat)
                a_full = pl.set_validshape(a, m_tile, FUSED_K_TILE)
                b = pl.load(lm_head_weight, [k, col], [FUSED_K_TILE, FUSED_VOCAB_TILE],
                            target_memory=pl.Mem.Mat, cache=LM_HEAD_WEIGHT_CACHE)
                accum = pl.matmul_acc(accum, a_full, b, init_cond=(k == 0))
                if k + FUSED_K_TILE == D:
                    valid_accum = pl.set_validshape(accum, valid_rows, FUSED_VOCAB_TILE)
                    logits = pl.store(valid_accum, [row, col], logits)
            if VOCAB_TAIL != 0:
                if worker == (VOCAB_PER_TP // FUSED_VOCAB_TILE) % FUSED_LM_HEAD_CORES:
                    col = VOCAB_PER_TP // FUSED_VOCAB_TILE * FUSED_VOCAB_TILE
                    tail_accum = _lm_head_project_tile(
                        hidden_states, lm_head_weight, row, col, valid_rows, m_tile, VOCAB_TAIL,
                    )
                    valid_tail = pl.set_validshape(tail_accum, valid_rows, VOCAB_TAIL)
                    logits = pl.store(valid_tail, [row, col], logits)
    return projected


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
) -> tuple[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32], pl.Scalar[pl.TASK_ID]]:
    if TP_SIZE == 1:
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
        completed = lm_head_project(selected_hidden, lm_head_weight, logits, MAX_LOGIT_ROWS)
    else:
        # Scratch is allocated just outside the scope that first writes it: a
        # create_tensor inside a pl.at yields a tile, not a GM tensor view.
        selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
        owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)

        # Publish this card's logit rows into every group member's window slot: the
        # window holds one slot per group member and each card writes only its own,
        # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
        with pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_dispatch_push") as _dispatch_push_tid:
            row = pl.tile.get_block_idx()
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

        # Wait after all local hidden-row publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid],
                   allow_early_resolve=False) as _dwait_tid:
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

        logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, VOCAB_PER_TP], dtype=pl.FP32)
        _project_tid = lm_head_project(owner_hiddens, lm_head_weight, logits_shards, GROUP_M_TILE)

        # Send each owner its slice of this card's vocab shard, split over vocab comm
        # tiles: block blk pushes its tiles to every owner.
        with pl.spmd(LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_push", deps=[_project_tid]) as _push_tid:
            blk = pl.tile.get_block_idx()
            vocab_base = tp_rank * VOCAB_PER_TP
            for owner_tp in pl.range(TP_SIZE):
                source_row_base = owner_tp * MAX_LOGIT_ROWS

                # put, not tile.remote_store: remote_store does not drain before the
                # notify issues (PTOAS#872), so the peer's gather reads tiles still in
                # flight. Self-target rides the same put -- a local pl.store makes a new
                # SSA version of logits_window that the gather cannot read across scopes
                # without a comm ctx.
                for ob in pl.range(blk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                    o0 = ob * LOGITS_COMM_TILE
                    pld.tensor.put(
                        dst=logits_window,
                        peer=group_base + owner_tp,
                        src=logits_shards,
                        dst_offsets=[0, vocab_base + o0],
                        src_offsets=[source_row_base, o0],
                        shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TILE],
                    )

                if LOGITS_COMM_TAIL != 0:
                    if blk == LOGITS_TAIL_BLOCK:
                        tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                        pld.tensor.put(
                            dst=logits_window,
                            peer=group_base + owner_tp,
                            src=logits_shards,
                            dst_offsets=[0, vocab_base + tail_o0],
                            src_offsets=[source_row_base, tail_o0],
                            shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TAIL],
                        )

            # Notify folded into the push: each block signals every peer after its own
            # stores, so a peer sees LOGITS_COMM_BLOCKS notifies per source per epoch.
            for owner_tp in pl.range(TP_SIZE):
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

        # Wait after all local vocab-shard publishes complete.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid],
                   allow_early_resolve=False) as _cwait_tid:
            for src_tp in pl.range(TP_SIZE):
                if src_tp != tp_rank:
                    pld.system.wait(
                        signal=logits_done,
                        offsets=[src_tp, 0],
                        expected=pl.cast((done_epoch - 1) * (LOGITS_COMM_BLOCKS + 1) + LOGITS_COMM_BLOCKS, pl.INT32),
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

        # Publish completion of every local logits-window reader.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_readback_notify", deps=[_gather_tid]) as _readback_notify_tid:
            for peer in pl.range(TP_SIZE):
                if peer != tp_rank:
                    pld.system.notify(
                        logits_done, peer=group_base + peer, offsets=[tp_rank, 0],
                        value=1, op=pld.NotifyOp.AtomicAdd,
                    )
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_readback_wait") as _readback_wait_tid:
            for peer in pl.range(TP_SIZE):
                if peer != tp_rank:
                    pld.system.defer_wait(
                        logits_done, offsets=[peer, 0], expected=done_epoch * (LOGITS_COMM_BLOCKS + 1), cmp=pld.WaitCmp.Ge,
                    )
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_retire",
            deps=[_readback_notify_tid, _readback_wait_tid], allow_early_resolve=False,
        ) as _retire_tid:
            anchor = pl.read(logits, [0, 0])
            if done_epoch == completed_epochs:
                for peer in pl.range(TP_SIZE):
                    if peer != tp_rank:
                        pld.system.notify(
                            hidden_done, peer=group_base + tp_rank, offsets=[peer, 0],
                            value=0, op=pld.NotifyOp.Set,
                        )
                        pld.system.notify(
                            logits_done, peer=group_base + tp_rank, offsets=[peer, 0],
                            value=0, op=pld.NotifyOp.Set,
                        )
            pl.write(logits, [0, 0], anchor)
        completed = _retire_tid
    return logits, completed


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor,
    sampled_ids: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID] = None,
):
    """Select the first maximum token ID for each requested logits row."""
    rows = pl.tensor.dim(sampled_ids, 0)
    workers = pl.min(GREEDY_WORKERS, rows)
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


@pl.jit.inline(auto_scope=False)
def lm_head_sample(
    hidden_states: pl.Tensor[[LM_HEAD_M_TILE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB_PER_TP], pl.BF16, pl.NZ],
    sampled_ids: pl.Tensor,
):
    """Project a masked TP1 batch and select its requested token IDs."""
    logits = pl.create_tensor([LM_HEAD_M_TILE, VOCAB], dtype=pl.FP32)
    projected = lm_head_project(hidden_states, lm_head_weight, logits, LM_HEAD_M_TILE)
    greedy_sample(logits, sampled_ids, projected)
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
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Project logits and sample top-1 tokens in one opaque L2 entry."""
    active = pl.cast(1, pl.INT32)
    if TP_SIZE == 1:
        active = _lm_head_has_rows(logit_row_indices)
    if active > 0:
        logits, projected = lm_head(
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
            completed_epochs,
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
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
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
        completed_epochs,
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
    """Collapse final-layer residual/pre-mix, normalize, project and sample.

    Consume the native FP32 state returned by prefill or decode without
    reinitializing it from embeddings. Each row index is rank-local; -1
    produces zero logits and is ignored by the caller, including idle ranks.
    This entry only reads layer state and never mutates caches. Wait for the
    backbone to complete before dispatch and for this call before reading
    logits or reusing state. Use persistent=True with reset_persistent_windows=False for repeated calls.
    Wait for each dispatch to complete on all ranks before the next call. TP1/DP8 uses eight full head replicas.
    """
    x_hc.bind_dynamic(1, T_DYN)
    hidden_buffer = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    logits_buffer = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for epoch_idx in pl.range(REPEAT):
        for rank in pl.range(pld.world_size()):
            hidden_window = pld.window(hidden_buffer, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
            hidden_done = pld.window(hidden_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            logits_window = pld.window(logits_buffer, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
            logits_done = pld.window(logits_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            l2_final_output(
                x_hc[rank], pre_mix[rank], final_norm_weight[rank], lm_head_weight[rank],
                logit_row_indices[rank], logits[rank], sampled_ids[rank],
                hidden_window, hidden_done, logits_window, logits_done,
                rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, epoch_idx + 1, REPEAT, device=rank,
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
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    completed_epochs: pl.Scalar[pl.INT32],
):
    """Compose final HC, Norm and LM head from the current layer state."""
    x_hc.bind_dynamic(0, T_DYN)
    active = pl.cast(1, pl.INT32)
    if TP_SIZE == 1:
        active = _lm_head_has_rows(logit_row_indices)
    if active > 0:
        rows = pl.tensor.dim(x_hc, 0)
        collapsed = pl.create_tensor([rows, D], dtype=pl.BF16)
        hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
        normalized = pl.create_tensor([rows, D], dtype=pl.BF16)
        for norm_block in pl.spmd((rows + 7) // 8, name_hint="final_rms_norm"):
            rms_norm_tiled_compensated(collapsed, final_norm_weight, normalized, norm_block, rows)
        lm_head_with_sampling(
            normalized, lm_head_weight, logit_row_indices, logits, sampled_ids,
            hidden_window, hidden_done, logits_window, logits_done, group_base, tp_rank, done_epoch, completed_epochs,
        )
    else:
        logits, sampled_ids = _lm_head_zero_outputs(logits, sampled_ids)
    return logits, sampled_ids


final_output = pl.jit.inline(_final_output)
l2_final_output = pl.jit(_final_output)


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, D, VOCAB_PER_TP], pl.BF16],
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

    # Epochs advance inside a dispatch; the final projection clears its signals.
    for epoch_idx in pl.range(REPEAT):
        for r in pl.range(pld.world_size()):
            hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
            hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
            logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
            logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
            l2_lm_head(
                hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
                sampled_ids[r],
                hidden_window, hidden_done, logits_window, logits_done,
                r // TP_SIZE * TP_SIZE, r % TP_SIZE, epoch_idx + 1, REPEAT, device=r,
            )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE, so concatenating shards in index order
    # reproduces the global vocabulary order every owner assembles.
    weight = unpack_bf16_nz(tensors["lm_head_weight"]).float()
    full_weight = torch.cat([weight[tp] for tp in range(TP_SIZE)], dim=-1)
    full_logits = []
    for owner_rank in range(WORLD_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)
    if "sampled_ids" in tensors:
        tensors["sampled_ids"].zero_()
        tensors["sampled_ids"][:, :, 0] = torch.argmax(
            tensors["logits"],
            dim=-1,
        ).to(torch.int32)


def golden_final_output(tensors):
    """Evaluate HC collapse, BF16 Norm and head projection independently on CPU."""
    import torch

    hidden = []
    for rank in range(WORLD_SIZE):
        collapsed = golden_hc_collapse(tensors["x_hc"][rank], tensors["pre_mix"][rank]).to(torch.bfloat16)
        normalized = golden_rms_norm_tiled(collapsed, tensors["final_norm_weight"][rank])
        hidden.append(normalized.to(torch.bfloat16))
    golden_lm_head({**tensors, "hidden_states": torch.stack(hidden)})


def build_tensor_specs(num_tokens=TEST_TOKENS, *, final=False):
    import torch
    from golden import TensorSpec

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        checkpoint = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        shards = pack_bf16_nz(checkpoint.transpose(-1, -2).contiguous())
        return torch.stack([shards[r % TP_SIZE] for r in range(WORLD_SIZE)], dim=0)

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        if final and DP_SIZE > 1:
            indices[-TP_SIZE:] = -1
        return indices

    specs = [
        TensorSpec(
            "hidden_states",
            [WORLD_SIZE, TEST_TOKENS, D],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        # One vocab shard per world rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight",
            [WORLD_SIZE, D, VOCAB_PER_TP],
            torch.bfloat16,
            init_value=init_lm_head_weight,
            resident="stacked",
        ),
        TensorSpec(
            "logits",
            [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB],
            torch.float32,
        ),
        TensorSpec(
            "sampled_ids",
            [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
            torch.int32,
        ),
        TensorSpec(
            "logit_row_indices",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_logit_row_indices,
        ),
    ]
    if final:
        specs = [
            TensorSpec("x_hc", [WORLD_SIZE, TEST_TOKENS, HC_MULT, D], torch.float32, init_value=torch.randn),
            TensorSpec("pre_mix", [WORLD_SIZE, TEST_TOKENS, HC_MULT], torch.float32, init_value=torch.rand),
            TensorSpec("final_norm_weight", [WORLD_SIZE, D], torch.bfloat16, init_value=torch.ones),
            specs[1], specs[4], specs[2], specs[3],
        ]
    return specs


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

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES),
                        help="DP groups (world size = tp * dp)")
    parser.add_argument("--repeat", type=int, default=REPEAT, choices=list(_REPEAT_CHOICES),
                        help="projections sharing windows, cleared after the final projection")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS,
                        help="Active hidden rows each owner projects (0 = empty shard)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(WORLD_SIZE)),
                        help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--final-output", action="store_true", help="include final HC collapse and Norm")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, (
        f"need at least {required_devices} devices, got {device_ids}"
    )
    assert args.tp == TP_SIZE and args.dp == DP_SIZE and args.repeat == REPEAT
    assert 0 <= args.num_tokens <= TEST_TOKENS

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
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:required_devices],
                num_sub_workers=0,
            ),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
