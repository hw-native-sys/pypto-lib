# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 LM head: dispatch hidden rows, project against a TP vocab shard, combine full-vocab logits.

Hidden states must already have passed the final RMSNorm. Every card is both an
owner and a TP rank: it holds vocab shard ``rank % TP_SIZE`` and serves only its
own group, so every peer is ``group_base + tp_rank``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import DECODE_TOKENS, FLASH as M
from sample import golden_sample, sample


T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size

# Parallelism, parsed off argv: the frontend needs both worlds static.
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
# DP groups built by the l3_lm_head fixture; the kernel carries no DP extent.
DP_SIZE: int = _parse_int_argv("--dp") or 1
WORLD_SIZE = TP_SIZE * DP_SIZE
VOCAB_PER_TP = VOCAB // TP_SIZE
# AIV lanes per AICore; owners shard across them to keep every push and notify single.
AIV_LANES = 2
OWNER_PAIRS = TP_SIZE // AIV_LANES
OWNERS_PER_LANE = TP_SIZE // AIV_LANES
FUSED_LM_HEAD_CORES = 24
DONE_VALUE = 1

# Rows. Decode specializations override DECODE_TOKENS before importing this
# module. logit_row_indices picks the sources; unused rows stay zero.
MAX_LOGIT_ROWS = DECODE_TOKENS
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
SAMPLED_IDS_PAD = 8
TEST_TOKENS = 16  # standalone fixture: hidden rows per card, > MAX_LOGIT_ROWS

# tiling -- both matmul tiles clear the 512 B contiguous-transfer floor
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 256
HIDDEN_GATHER_TILE = 512
LOGITS_COMM_TILE = 2048

# Derived layout. The vocab shard rarely divides the matmul tile, so the last tile
# is ragged: a plain ceil tile count, the weight load carries a valid_shape, and
# logits_shards pads to whole tiles so its store stays in bounds.
VOCAB_LAST_TILE = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE + (1 if VOCAB_LAST_TILE != 0 else 0)
SHARDS_VOCAB = VOCAB_TILES * FUSED_VOCAB_TILE
# The shard lands whole in a GM scratch; per-owner pushes slice that scratch.
SHARD_ROWS = GROUP_LOGIT_ROWS // AIV_LANES
# Combine blocks: one per vocab comm tile, capped at the core count; the tail tile
# rides the block the strided loop hands it next.
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

# Keeps the GM staging race-free: each UP_DOWN lane writes exactly the row band it
# later reads, and the two lanes' bands are disjoint.
assert SHARD_ROWS == OWNERS_PER_LANE * MAX_LOGIT_ROWS, (
    "each AIV lane's shard rows must be exactly the rows of the owners it pushes"
)
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

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait") as _dwait_tid:
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.defer_wait(
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

    # Mixed cube + comm kernel: the cube projects a vocab tile, pl.aiv_shard carries
    # that accumulator across the C->V edge, and pld.tensor.remote_store pushes each
    # owner's rows to that peer's window, so a tile goes on the wire while the cube
    # projects the next. The ragged last tile rides the same loop as a padded tile.
    logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, SHARDS_VOCAB], dtype=pl.FP32)
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ):
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

            # Columns must be fully valid across the C->V crossing: the boundary
            # transports the full box and rejects a runtime-valued column extent.
            # The ragged tile's padding is cut off by the push below.
            mm_acc = pl.set_validshape(mm_acc, GROUP_LOGIT_ROWS, FUSED_VOCAB_TILE)

            # WORKAROUND -- revert once a subview of a pl.aiv_shard result is legal
            # (filed against ptoas). The shard cannot be sliced, so it is stored whole
            # to GM and the per-owner pushes read slices back out.
            # Race-free by construction: UP_DOWN gives each lane a disjoint row band
            # that it both writes and reads, guarded by the SHARD_ROWS assert above.
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

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait") as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.defer_wait(
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


@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
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
    """Project logits and sample tokens in one opaque L2 entry."""
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
    sampled_ids = sample(
        logits,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
        sampled_ids,
    )
    return logits, sampled_ids


@pl.jit
def lm_head_with_sampling_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
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
    """Standalone opaque entry for projection plus sampling tests."""
    return lm_head_with_sampling(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
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
    sampling_temperatures: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
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
            hidden_states[r], lm_head_weight[r], logit_row_indices[r],
            sampling_temperatures[r], sampling_top_ks[r],
            sampling_seeds[r], sampling_positions[r],
            logits[r], sampled_ids[r],
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
    world_size = hidden.shape[0]
    for owner_rank in range(world_size):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)
    if "sampled_ids" in tensors:
        for rank in range(world_size):
            golden_sample(
                {
                    "logits": tensors["logits"][rank],
                    "temperatures": tensors["sampling_temperatures"][rank],
                    "top_ks": tensors["sampling_top_ks"][rank],
                    "seeds": tensors["sampling_seeds"][rank],
                    "positions": tensors["sampling_positions"][rank],
                    "sampled_ids": tensors["sampled_ids"][rank],
                }
            )


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
            "sampling_temperatures",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.float32,
            init_value=lambda: torch.zeros(WORLD_SIZE, MAX_LOGIT_ROWS),
        ),
        TensorSpec(
            "sampling_top_ks",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=lambda: torch.full(
                (WORLD_SIZE, MAX_LOGIT_ROWS), VOCAB, dtype=torch.int32
            ),
        ),
        TensorSpec(
            "sampling_seeds",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=lambda: torch.zeros(WORLD_SIZE, MAX_LOGIT_ROWS, dtype=torch.int32),
        ),
        TensorSpec(
            "sampling_positions",
            [WORLD_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=lambda: torch.zeros(WORLD_SIZE, MAX_LOGIT_ROWS, dtype=torch.int32),
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
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
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
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
