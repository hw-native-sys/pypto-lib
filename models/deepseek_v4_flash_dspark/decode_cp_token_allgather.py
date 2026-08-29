# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""Context-parallel decode token-row all-gather into rank-major order on every rank."""

import sys

import config


# TP-derived shapes freeze at import time, so select the TP world before the
# config read below.
_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
config.TP = TP_SIZE

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    DECODE_TOKENS,
    FLASH as M,
)


# Dynamic shape variables. The local query axis and the gathered KV axis carry
# different row counts and must stay separate.
CP_Q_T_DYN = pl.dynamic("DECODE_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("DECODE_CP_KV_T_DYN")
# Request axis of the gathered stream: the whole group's requests, not the rank's.
CP_KV_B_DYN = pl.dynamic("DECODE_CP_KV_B_DYN")

# model config
D = M.hidden_size

# communication bounds
DECODE_GROUP_CAP = DECODE_TOKENS
DECODE_LOCAL_CAP = DECODE_GROUP_CAP // TP_SIZE

# tiling
COMM_ROW_TILE = 8
READBACK_ROW_TILE = 16

# fixture
FIXTURE_ROUNDS = 2
FIXTURE_LOCAL_T = DECODE_LOCAL_CAP

if DECODE_GROUP_CAP % TP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_GROUP_CAP} must be divisible by TP size {TP_SIZE}")


@pl.jit.inline
def decode_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather rank-major rows and retire the complete two-phase signal epoch."""
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t

    # Publish the payload and first-phase arrival from one producer task.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="decode_cp_token_allgather_push", allow_early_resolve=True,
    ) as _push_tid:
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=gather_window, peer=group_base + peer_tp,
                src=hidden_local,
                dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[local_t, D],
                chunk_rows=COMM_ROW_TILE, chunk_cols=D,
            )
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    # Register the peer payload conditions as deferred completion.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_cp_token_allgather_payload_wait") as _payload_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Copy peer payloads and publish local readback completion.
    group_rows = TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="decode_cp_token_allgather_readback",
        deps=[_push_tid, _payload_wait_tid],
    ) as _readback_tid:
        for tile_row in pl.range(0, full_rows, READBACK_ROW_TILE):
            window_tile = gather_window[tile_row : tile_row + READBACK_ROW_TILE, 0:D]
            group_out[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = window_tile
        for tail_row in pl.range(full_rows, group_rows):
            window_row = gather_window[tail_row : tail_row + 1, 0:D]
            group_out[tail_row : tail_row + 1, 0:D] = window_row
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_cp_token_allgather_readback_wait") as _readback_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(2, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Retire peer credits and anchor output consumption to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="decode_cp_token_allgather_retire",
        deps=[_readback_tid, _readback_wait_tid],
    ):
        completion_anchor = pl.read(group_out, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=self_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_out, [0, 0], completion_anchor)

    return group_out, gather_signal


@pl.jit
def decode_cp_token_allgather_fixture(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Run one rank of the decode token-row all-gather."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    group_out.bind_dynamic(0, CP_KV_T_DYN)
    group_out, gather_signal = decode_cp_token_allgather_step(
        hidden_local, group_out,
        gather_window, gather_signal,
        group_base, tp_rank,
    )
    return group_out, gather_signal


@pl.jit.host
def l3_decode_cp_token_allgather_fixture(
    hidden_local: pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_KV_T_DYN, D], pl.BF16]],
):
    """Launch two all-gather rounds on one retained TP window."""
    hidden_local.bind_dynamic(2, CP_Q_T_DYN)
    group_out.bind_dynamic(2, CP_KV_T_DYN)
    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for round_id in pl.range(FIXTURE_ROUNDS):
        for rank in pl.range(pld.world_size()):
            gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
            gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            decode_cp_token_allgather_fixture(
                hidden_local[round_id, rank], group_out[round_id, rank],
                gather_window, gather_signal,
                0, rank,
                device=rank,
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


def cp_split(value, tp_size):
    """Split one materialised tensor's leading token axis rank-major."""
    return value.reshape(tp_size, value.shape[0] // tp_size, *value.shape[1:]).contiguous()


def build_tensor_specs(local_t=FIXTURE_LOCAL_T):
    """Build two distinct rounds of per-rank inputs."""
    import torch

    from golden import TensorSpec

    if local_t < 1 or local_t > DECODE_LOCAL_CAP:
        raise ValueError(f"local_t must be in [1, {DECODE_LOCAL_CAP}], got {local_t}")
    group_t = TP_SIZE * local_t

    def init_hidden_local():
        shape = (FIXTURE_ROUNDS, TP_SIZE, local_t, D)
        values = torch.arange(FIXTURE_ROUNDS * TP_SIZE * local_t * D, dtype=torch.int32)
        values = values.remainder(251).reshape(shape).to(torch.bfloat16)
        for round_id in range(FIXTURE_ROUNDS):
            for rank in range(TP_SIZE):
                values[round_id, rank, :, 0] = float(round_id * TP_SIZE + rank)
        return values

    return [
        TensorSpec("hidden_local", [FIXTURE_ROUNDS, TP_SIZE, local_t, D], torch.bfloat16, init_value=init_hidden_local),
        TensorSpec("group_out", [FIXTURE_ROUNDS, TP_SIZE, group_t, D], torch.bfloat16, is_output=True),
    ]


def golden_decode_cp_token_allgather(tensors):
    """Every rank receives its round's rank-major concatenation."""
    hidden_local = tensors["hidden_local"]
    rounds, tp_size, local_t, _ = hidden_local.shape
    gathered = hidden_local.reshape(rounds, tp_size * local_t, D)
    tensors["group_out"][:] = gathered.unsqueeze(1)


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser(description="Standalone context-parallel decode token-row all-gather test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument(
        "--local-t", type=int, default=FIXTURE_LOCAL_T,
        help=f"per-rank token count, 1..{DECODE_LOCAL_CAP}",
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
    if not 1 <= args.local_t <= DECODE_LOCAL_CAP:
        parser.error(f"--local-t must be in [1, {DECODE_LOCAL_CAP}], got {args.local_t}")

    result = run_jit(
        fn=l3_decode_cp_token_allgather_fixture,
        specs=build_tensor_specs(args.local_t),
        golden_fn=golden_decode_cp_token_allgather,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
