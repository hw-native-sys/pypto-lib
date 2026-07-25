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

The DP world is cut into ``--dp // --tp`` groups. Every card is both an owner and
a TP rank: it holds vocab shard ``rank % TP_SIZE`` and serves only its own group,
so every ``peer`` is ``group_base + tp_rank``. Four stages: publish selected rows,
project the group's rows in one matmul, route each owner's shard back, gather
full-vocabulary logits.

Per-card cost tracks ``VOCAB_PER_TP``, not the DP world size: the matmul M extent
is always ``TP_SIZE * MAX_LOGIT_ROWS``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import FLASH as M


T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# Model
D = M.hidden_size
VOCAB = M.vocab_size

# Parallelism. Static in the frontend, so both worlds are parsed off argv here.
_TP_CHOICES = (2, 4, 8, 16)
_DP_CHOICES = (2, 4, 8, 16)
_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT
# --dp is optional and defaults to a single TP group. The composed decode_fwd /
# prefill_fwd drivers and pypto-serving spell this same world --ep, where the
# attention-DP and MoE expert-parallel rank counts coincide.
DP_SIZE: int = _parse_int_argv("--dp") or _parse_int_argv("--ep") or TP_SIZE
VOCAB_PER_TP = VOCAB // TP_SIZE

# Rows. logit_row_indices picks the sources; unused rows stay zero.
MAX_LOGIT_ROWS = 8
TEST_TOKENS = 16  # standalone fixture: hidden rows per card, > MAX_LOGIT_ROWS
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

# Tiling
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 128
HIDDEN_COMM_TILE = 512
LOGITS_COMM_TILE = 2048
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
FUSED_LM_HEAD_CORES = 24
DONE_VALUE = 1

assert D % FUSED_K_TILE == 0
assert D % HIDDEN_COMM_TILE == 0
assert VOCAB % TP_SIZE == 0
assert GROUP_LOGIT_ROWS % 16 == 0, "matmul M extent must be a multiple of 16"
assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"
assert DP_SIZE in _DP_CHOICES, f"--dp must be one of {_DP_CHOICES} (got {DP_SIZE})"
assert DP_SIZE % TP_SIZE == 0, f"--dp must be a multiple of --tp, got dp={DP_SIZE}, tp={TP_SIZE}"


@pl.jit
def lm_head(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    num_tokens_per_owner: pl.Tensor[[DP_SIZE], pl.INT32],
    logit_row_indices: pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    num_logit_rows: pl.Tensor[[DP_SIZE], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)

    # Select this card's logit rows, publish them, barrier, then pull in every
    # owner's rows. Notify and wait must share one scope: a window written in one
    # scope and read in another is threaded through orchestration as a new SSA
    # version, which then needs a comm ctx materialized at that level.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch"):
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        owner_tokens = pl.read(num_tokens_per_owner, [my_rank])
        active_tokens = pl.max(pl.min(owner_tokens, hidden_rows), 0)
        active_rows = pl.max(pl.min(pl.read(num_logit_rows, [my_rank]), MAX_LOGIT_ROWS), 0)
        for row in pl.range(MAX_LOGIT_ROWS):
            source_row_raw = pl.read(logit_row_indices, [my_rank, row])
            for kb in pl.range(D // HIDDEN_COMM_TILE):
                k0 = kb * HIDDEN_COMM_TILE
                zero_tile = pl.full([1, HIDDEN_COMM_TILE], dtype=pl.BF16, value=0.0)
                selected_hidden[row : row + 1, k0 : k0 + HIDDEN_COMM_TILE] = zero_tile
                if row < active_rows:
                    if source_row_raw >= 0:
                        if source_row_raw < active_tokens:
                            source_row = pl.cast(source_row_raw, target_type=pl.INDEX)
                            src = hidden_states[source_row : source_row + 1, k0 : k0 + HIDDEN_COMM_TILE]
                            selected_hidden[row : row + 1, k0 : k0 + HIDDEN_COMM_TILE] = src

        for kb in pl.range(D // HIDDEN_COMM_TILE):
            k0 = kb * HIDDEN_COMM_TILE
            hidden_window[:, k0 : k0 + HIDDEN_COMM_TILE] = selected_hidden[:, k0 : k0 + HIDDEN_COMM_TILE]
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        # Barrier on the group's publishes, then pull every owner's rows in.
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=done_epoch,
                    cmp=pld.WaitCmp.Ge,
                )
        for owner_tp in pl.range(TP_SIZE):
            for kb in pl.range(D // HIDDEN_COMM_TILE):
                k0 = kb * HIDDEN_COMM_TILE
                if owner_tp == tp_rank:
                    row0 = owner_tp * MAX_LOGIT_ROWS
                    owner_hiddens[row0 : row0 + MAX_LOGIT_ROWS, k0 : k0 + HIDDEN_COMM_TILE] = hidden_window[:, k0 : k0 + HIDDEN_COMM_TILE]
                else:
                    remote_tile = pld.tile.remote_load(
                        hidden_window,
                        peer=group_base + owner_tp,
                        offsets=[0, k0],
                        shape=[MAX_LOGIT_ROWS, HIDDEN_COMM_TILE],
                    )
                    pl.store(remote_tile, [owner_tp * MAX_LOGIT_ROWS, k0], owner_hiddens)

    logits_shards = pl.create_tensor([GROUP_LOGIT_ROWS, VOCAB_PER_TP], dtype=pl.FP32)
    # Project all group-owner rows in one matmul M tile.
    for lm_core in pl.spmd(FUSED_LM_HEAD_CORES, name_hint="lm_head_matmul"):
        for mm_ob in pl.range(lm_core, VOCAB_PER_TP // FUSED_VOCAB_TILE, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            mm_hidden0 = owner_hiddens[:, 0:FUSED_K_TILE]
            mm_weight0 = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, 0:FUSED_K_TILE]
            mm_acc = pl.matmul(mm_hidden0, mm_weight0, b_trans=True, out_dtype=pl.FP32)
            for mm_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                mm_k0 = mm_kb * FUSED_K_TILE
                mm_hidden_tile = owner_hiddens[:, mm_k0 : mm_k0 + FUSED_K_TILE]
                mm_weight_tile = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True)
            logits_shards[:, mm_o0 : mm_o0 + FUSED_VOCAB_TILE] = mm_acc

        if VOCAB_TAIL != 0:
            if lm_core == (VOCAB_PER_TP // FUSED_VOCAB_TILE) % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_PER_TP // FUSED_VOCAB_TILE * FUSED_VOCAB_TILE
                mm_hidden_t0 = owner_hiddens[:, 0:FUSED_K_TILE]
                mm_weight_t0 = lm_head_weight[mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL, 0:FUSED_K_TILE]
                mm_acc_tail = pl.matmul(mm_hidden_t0, mm_weight_t0, b_trans=True, out_dtype=pl.FP32)
                for mm_tail_kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                    mm_tail_k0 = mm_tail_kb * FUSED_K_TILE
                    mm_hidden_tk = owner_hiddens[:, mm_tail_k0 : mm_tail_k0 + FUSED_K_TILE]
                    mm_weight_tk = lm_head_weight[
                        mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL, mm_tail_k0 : mm_tail_k0 + FUSED_K_TILE
                    ]
                    mm_acc_tail = pl.matmul_acc(mm_acc_tail, mm_hidden_tk, mm_weight_tk, b_trans=True)
                logits_shards[:, mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL] = mm_acc_tail

    # Send each owner its slice of this card's vocab shard, barrier, then
    # assemble full-vocabulary logits. Same one-scope rule as above.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_combine"):
        vocab_base = tp_rank * VOCAB_PER_TP
        for owner_tp in pl.range(TP_SIZE):
            source_row_base = owner_tp * MAX_LOGIT_ROWS

            for ob in pl.range(VOCAB_PER_TP // LOGITS_COMM_TILE):
                o0 = ob * LOGITS_COMM_TILE
                tile = pl.load(logits_shards, [source_row_base, o0], [MAX_LOGIT_ROWS, LOGITS_COMM_TILE])
                if owner_tp == tp_rank:
                    pl.store(tile, [0, vocab_base + o0], logits_window)
                else:
                    pld.tile.remote_store(
                        tile, logits_window, group_base + owner_tp, [0, vocab_base + o0],
                    )

            if LOGITS_COMM_TAIL != 0:
                tail_o0 = VOCAB_PER_TP // LOGITS_COMM_TILE * LOGITS_COMM_TILE
                tile = pl.load(
                    logits_shards,
                    [source_row_base, tail_o0],
                    [MAX_LOGIT_ROWS, LOGITS_COMM_TAIL],
                )
                if owner_tp == tp_rank:
                    pl.store(tile, [0, vocab_base + tail_o0], logits_window)
                else:
                    pld.tile.remote_store(
                        tile, logits_window, group_base + owner_tp, [0, vocab_base + tail_o0],
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

        # Barrier on every TP rank, then assemble full-vocabulary logits.
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=done_epoch,
                    cmp=pld.WaitCmp.Ge,
                )
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(VOCAB_PER_TP // LOGITS_COMM_TILE):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                logits[:, lo : lo + LOGITS_COMM_TILE] = logits_window[:, lo : lo + LOGITS_COMM_TILE]

            if LOGITS_COMM_TAIL != 0:
                tail_o0 = VOCAB_PER_TP // LOGITS_COMM_TILE * LOGITS_COMM_TILE
                tl = src_vocab_base + tail_o0
                logits[:, tl : tl + LOGITS_COMM_TAIL] = logits_window[:, tl : tl + LOGITS_COMM_TAIL]
    return logits


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[DP_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    num_tokens_per_owner: pl.Tensor[[DP_SIZE], pl.INT32],
    logit_row_indices: pl.Tensor[[DP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    num_logit_rows: pl.Tensor[[DP_SIZE], pl.INT32],
):
    # Windows are group-local: every card publishes only its own selected rows
    # and receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [MAX_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head(
            hidden_states[r], lm_head_weight[r % TP_SIZE], num_tokens_per_owner,
            logit_row_indices, num_logit_rows, logits[r], hidden_window, hidden_done,
            logits_window, logits_done, r, r // TP_SIZE * TP_SIZE, r % TP_SIZE,
            DONE_VALUE, device=r,
        )


def golden_lm_head(tensors):
    import torch

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE, so concatenating shards in index order
    # reproduces the global vocabulary order every owner assembles.
    full_weight = tensors["lm_head_weight"].float().reshape(TP_SIZE * VOCAB_PER_TP, D)
    full_logits = []
    for owner_rank in range(DP_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        active_tokens = max(
            min(int(tensors["num_tokens_per_owner"][owner_rank]), hidden.shape[1]), 0,
        )
        active_rows = max(min(int(tensors["num_logit_rows"][owner_rank]), MAX_LOGIT_ROWS), 0)
        for row in range(active_rows):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if 0 <= source_row < active_tokens:
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs(num_tokens=TEST_TOKENS):
    import torch
    from golden import TensorSpec

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(DP_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        return (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

    def init_logit_row_indices():
        indices = torch.full((DP_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    return [
        TensorSpec(
            "hidden_states",
            [DP_SIZE, TEST_TOKENS, D],
            torch.bfloat16,
            init_value=init_hidden_states,
        ),
        # Leading dim is TP_SIZE while the world is DP_SIZE (card r consumes
        # shard r % TP_SIZE), so the resident="stacked" contract does not apply.
        TensorSpec(
            "lm_head_weight",
            [TP_SIZE, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
        ),
        TensorSpec(
            "logits",
            [DP_SIZE, MAX_LOGIT_ROWS, VOCAB],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "num_tokens_per_owner",
            [DP_SIZE],
            torch.int32,
            init_value=lambda: torch.full((DP_SIZE,), num_tokens, dtype=torch.int32),
        ),
        TensorSpec(
            "logit_row_indices",
            [DP_SIZE, MAX_LOGIT_ROWS],
            torch.int32,
            init_value=init_logit_row_indices,
        ),
        TensorSpec(
            "num_logit_rows",
            [DP_SIZE],
            torch.int32,
            init_value=lambda: torch.full((DP_SIZE,), active, dtype=torch.int32),
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


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES),
                        help="Attention-DP world size (hidden-row owners)")
    parser.add_argument("--num-tokens", type=int, default=TEST_TOKENS,
                        help="Active hidden rows each owner projects")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(DP_SIZE)),
                        help=f"comma-separated device ids; need at least {DP_SIZE}")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = DP_SIZE
    assert len(device_ids) >= required_devices, (
        f"need at least {required_devices} devices, got {device_ids}"
    )
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs(args.num_tokens)
    golden_fn = golden_lm_head
    compare_fn = {"logits": compare_logits}

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
