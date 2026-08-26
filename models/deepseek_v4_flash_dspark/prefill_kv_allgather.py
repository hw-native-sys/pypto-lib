# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""Context-parallel prefill KV all-gather: each rank's token rows gathered into rank-major order on every rank."""

import sys

_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2


def _parse_int_argv(flag: str, default: int) -> int:
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


TP_SIZE = _parse_int_argv("--tp", _TP_DEFAULT)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")

import config

config.TP = TP_SIZE

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M
from decode_o_proj import reset_tp_group_signal, tp_group_barrier


# Dynamic shape variables. The local slice and the gathered stream carry
# different extents in one program.
CP_Q_T_DYN = pl.dynamic("PREFILL_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_CP_KV_T_DYN")

# model config
D = M.hidden_size

# Window capacity: one fixed allocation shared by the TP group, bounding the
# gathered stream. Bump together with prefill_csa's own token bound.
PREFILL_GROUP_CAP = 8192
PREFILL_LOCAL_CAP = PREFILL_GROUP_CAP // TP_SIZE

# tiling
COMM_ROW_TILE = 8
READBACK_ROW_TILE = _parse_int_argv("--readback-tile", 16)
if READBACK_ROW_TILE < 1:
    raise ValueError(f"--readback-tile must be >= 1 (got {READBACK_ROW_TILE})")

# fixture
FIXTURE_LOCAL_T = min(257, PREFILL_LOCAL_CAP)


@pl.jit.incore
def prefill_kv_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.InOut[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather every rank's token rows into rank-major order on every rank."""
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t
    for peer_tp in pl.range(TP_SIZE):
        pld.tensor.put(
            dst=gather_window,
            peer=group_base + peer_tp,
            src=hidden_local,
            dst_offsets=[target_row, 0],
            src_offsets=[0, 0],
            shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE,
            chunk_cols=D,
        )

    expected_one = pl.cast(1, pl.INT32)
    gather_signal = tp_group_barrier(gather_signal, group_base, tp_rank, expected_one)

    # Whole tiles take the wide path; the ragged remainder falls back to rows.
    group_rows = TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    for tile_row in pl.range(0, full_rows, READBACK_ROW_TILE):
        group_out[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = gather_window[
            tile_row : tile_row + READBACK_ROW_TILE, 0:D
        ]
    for tail_row in pl.range(full_rows, group_rows):
        group_out[tail_row : tail_row + 1, 0:D] = gather_window[tail_row : tail_row + 1, 0:D]

    expected_two = pl.cast(2, pl.INT32)
    gather_signal = tp_group_barrier(gather_signal, group_base, tp_rank, expected_two)
    gather_signal = reset_tp_group_signal(gather_signal, group_base, tp_rank)
    return group_out, gather_signal


@pl.jit
def prefill_kv_allgather_fixture(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.InOut[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Run the prefill KV all-gather on one TP rank, with no compute attached."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    group_out.bind_dynamic(0, CP_KV_T_DYN)
    group_out, gather_signal = prefill_kv_token_allgather_step(
        hidden_local, group_out, gather_window, gather_signal, group_base, tp_rank,
    )
    return group_out, gather_signal


@pl.jit.host
def l3_prefill_kv_allgather_fixture(
    hidden_local: pl.Tensor[[TP_SIZE, CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.InOut[pl.Tensor[[TP_SIZE, CP_KV_T_DYN, D], pl.BF16]],
):
    """Launch the prefill all-gather fixture on one TP group."""
    hidden_local.bind_dynamic(1, CP_Q_T_DYN)
    group_out.bind_dynamic(1, CP_KV_T_DYN)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_kv_allgather_fixture(
            hidden_local[rank],
            group_out[rank],
            gather_window,
            gather_signal,
            0,
            rank,
            device=rank,
        )


def materialize_spec(spec):
    """Realise a spec's init_value once, so every rank gets the same draw."""
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


def build_tensor_specs(local_t=FIXTURE_LOCAL_T):
    """Build per-rank inputs; each rank's rows carry its rank in the leading column."""
    import torch

    from golden import TensorSpec

    if local_t < 1 or local_t > PREFILL_LOCAL_CAP:
        raise ValueError(f"local_t must be in [1, {PREFILL_LOCAL_CAP}], got {local_t}")
    group_t = TP_SIZE * local_t

    def init_hidden_local():
        shape = (TP_SIZE, local_t, D)
        values = torch.arange(TP_SIZE * local_t * D, dtype=torch.int32)
        values = values.remainder(251).reshape(shape).to(torch.bfloat16)
        for rank in range(TP_SIZE):
            values[rank, :, 0] = float(rank)
        return values

    return [
        TensorSpec("hidden_local", [TP_SIZE, local_t, D], torch.bfloat16, init_value=init_hidden_local),
        # init_value keeps the spec on the InOut branch the kernel declares.
        TensorSpec("group_out", [TP_SIZE, group_t, D], torch.bfloat16, init_value=0.0, is_output=True),
    ]


def golden_prefill_kv_allgather(tensors):
    """Every rank ends up holding the same rank-major concatenation."""
    hidden_local = tensors["hidden_local"]
    tp_size, local_t, _ = hidden_local.shape
    gathered = hidden_local.reshape(tp_size * local_t, D)
    tensors["group_out"][:] = gathered.unsqueeze(0)


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser(description="Standalone context-parallel prefill KV all-gather test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument(
        "--local-t", type=int, default=FIXTURE_LOCAL_T,
        help=f"per-rank token count, 1..{PREFILL_LOCAL_CAP}",
    )
    parser.add_argument("--readback-tile", type=int, default=READBACK_ROW_TILE)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        raise SystemExit(f"--tp={args.tp} does not match import-time TP_SIZE={TP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if not 1 <= args.local_t <= PREFILL_LOCAL_CAP:
        parser.error(f"--local-t must be in [1, {PREFILL_LOCAL_CAP}], got {args.local_t}")

    result = run_jit(
        fn=l3_prefill_kv_allgather_fixture,
        specs=build_tensor_specs(args.local_t),
        golden_fn=golden_prefill_kv_allgather,
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
