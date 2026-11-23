# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill context ownership metadata and token exchanges."""

import pypto.language as pl
from math import gcd
import sys
import pypto.language.distributed as pld
from models.deepseek_v4_flash_lowlat.config import FLASH as M
from prefill_shapes import MOE_TOKENS, DECODE_SEQ


# Dynamic dimensions shared by packed prefill metadata consumers.
REQUESTS_DYN = pl.dynamic("PREFILL_METADATA_REQUESTS_DYN")
QUERY_START_LOC_DYN = pl.dynamic("PREFILL_METADATA_QUERY_START_LOC_DYN")
LOCAL_TOKENS_DYN = pl.dynamic("PREFILL_METADATA_LOCAL_TOKENS_DYN")
CHECKPOINT_TILES_DYN = pl.dynamic("PREFILL_METADATA_CHECKPOINT_TILES_DYN")

# tiling
METADATA_TOKEN_TILE = 512
CHECKPOINTS_PER_TILE = 6


@pl.jit.inline
def lower_local_request_ids(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
):
    """Lower packed query starts into request ids for one TP rank."""
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    base = pl.cast(local_base, pl.INT32)
    local_token_count = pl.tensor.dim(local_request_ids, 0)

    local_request_ids_row = pl.reshape(local_request_ids, [1, local_token_count])
    metadata_token_span = local_token_count + METADATA_TOKEN_TILE - 1
    for token_block in pl.spmd(
        metadata_token_span // METADATA_TOKEN_TILE,
        name_hint="prefill_lower_local_request_ids",
    ):
        token_start = token_block * METADATA_TOKEN_TILE
        valid_tokens = pl.min(METADATA_TOKEN_TILE, local_token_count - token_start)
        packed_base = base + pl.cast(token_start, pl.INT32)
        request_ids_tile = pl.full([1, METADATA_TOKEN_TILE], dtype=pl.INT32, value=-1)
        if request_count == 1:
            request_start = pl.read(query_start_loc, [0])
            request_end = pl.read(query_start_loc, [1])
            packed_end = packed_base + pl.cast(valid_tokens, pl.INT32)
            request_margin = pl.min(packed_base - request_start, request_end - packed_end)
            if request_margin >= 0:
                request_ids_tile = pl.full([1, METADATA_TOKEN_TILE], dtype=pl.INT32, value=0)
            else:
                for tile_token in pl.range(valid_tokens):
                    packed_token = packed_base + pl.cast(tile_token, pl.INT32)
                    if packed_token >= request_start:
                        if packed_token < request_end:
                            pl.write(request_ids_tile, [0, tile_token], pl.cast(0, pl.INT32))
        else:
            for tile_token in pl.range(valid_tokens):
                packed_token = packed_base + pl.cast(tile_token, pl.INT32)
                for request in pl.range(request_count):
                    request_start = pl.read(query_start_loc, [request])
                    request_end = pl.read(query_start_loc, [request + 1])
                    if packed_token >= request_start:
                        if packed_token < request_end:
                            pl.write(request_ids_tile, [0, tile_token], pl.cast(request, pl.INT32))
        request_ids_valid = pl.set_validshape(request_ids_tile, 1, valid_tokens)
        local_request_ids_row[0:1, token_start : token_start + METADATA_TOKEN_TILE] = request_ids_valid
    return local_request_ids


@pl.jit
def prefill_metadata_test(
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_base: pl.Scalar[pl.INT32],
    request_ids: pl.Out[pl.Tensor[[LOCAL_TOKENS_DYN], pl.INT32]],
):
    """Test rank-local packed prefill metadata lowering."""
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    request_ids.bind_dynamic(0, LOCAL_TOKENS_DYN)
    return lower_local_request_ids(query_start_loc, request_ids, local_base)


def build_tensor_specs_metadata():
    import torch
    from golden import ScalarSpec, TensorSpec

    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    local_token_count = 4
    request_ids = torch.full((local_token_count,), -1, dtype=torch.int32)
    local_base = local_token_count
    return [
        TensorSpec("query_start_loc", list(query_start_loc.shape), torch.int32, init_value=query_start_loc),
        ScalarSpec("local_base", torch.int32, local_base),
        TensorSpec("request_ids", list(request_ids.shape), torch.int32),
    ]


def golden_prefill_metadata(tensors):
    import torch

    query_start_loc = tensors["query_start_loc"]
    local_token_count = tensors["request_ids"].shape[0]
    local_base = int(tensors["local_base"])
    total_tokens = int(query_start_loc[-1])
    request_ids = tensors["request_ids"]
    request_ids.fill_(-1)
    for local_token in range(local_token_count):
        packed_token = local_base + local_token
        if packed_token >= total_tokens:
            continue
        for request in range(query_start_loc.numel() - 1):
            if query_start_loc[request] <= packed_token < query_start_loc[request + 1]:
                request_ids[local_token] = request
                break
    expected = torch.tensor([1, 1, 1, -1], dtype=torch.int32)
    if not torch.equal(request_ids, expected):
        raise AssertionError(f"unexpected request ids: {request_ids.tolist()}")


# ci: devices=2



# runtime configuration
_TP_CHOICES = (1, 2, 4, 8)
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




# Dynamic shape variables.
CP_Q_T_DYN = pl.dynamic("PREFILL_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_CP_KV_T_DYN")

# model config
D = M.hidden_size

# communication bounds
PREFILL_GROUP_CAP = 8192
PREFILL_LOCAL_CAP = PREFILL_GROUP_CAP // TP_SIZE

# tiling
COMM_ROW_TILE = 8
READBACK_ROW_TILE = gcd(MOE_TOKENS // DECODE_SEQ * (DECODE_SEQ - 1), 16)
READBACK_WORKERS = 16

# fixture
FIXTURE_ROUNDS = 2
FIXTURE_LOCAL_T = min(257, PREFILL_LOCAL_CAP)


def _prefill_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather rank-major rows and retire the complete two-phase signal epoch."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    group_out.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t

    # Each peer has an independent destination and arrival signal.
    with pl.spmd(TP_SIZE, name_hint="prefill_cp_token_allgather_push", allow_early_resolve=True) as _push_tid:
        peer_tp = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer_tp,
            src=hidden_local,
            dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D,
            pipeline=True,
        )
        if peer_tp != tp_rank:
            pld.system.notify(
                target=gather_signal, peer=group_base + peer_tp,
                offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )

    # Register the peer payload conditions as deferred completion.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cp_token_allgather_payload_wait") as _payload_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Fixed readers own strided row tiles; release the window after every reader.
    group_rows = TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    with pl.spmd(
        READBACK_WORKERS,
        name_hint="prefill_cp_token_allgather_readback",
        deps=[_push_tid, _payload_wait_tid],
    ) as _readback_tid:
        worker = pl.tile.get_block_idx()
        for tile_row in pl.range(
            worker * READBACK_ROW_TILE,
            full_rows,
            READBACK_WORKERS * READBACK_ROW_TILE,
        ):
            window_tile = gather_window[tile_row : tile_row + READBACK_ROW_TILE, 0:D]
            group_out[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = window_tile
        for tail_row in pl.range(full_rows + worker, group_rows, READBACK_WORKERS):
            window_row = gather_window[tail_row : tail_row + 1, 0:D]
            group_out[tail_row : tail_row + 1, 0:D] = window_row
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_notify",
        deps=[_readback_tid],
    ) as _readback_notify_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_cp_token_allgather_readback_wait") as _readback_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=pl.cast(2, pl.INT32), cmp=pld.WaitCmp.Ge,
                )

    # Retire peer credits and anchor output consumption to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_retire",
        deps=[_readback_notify_tid, _readback_wait_tid],
    ):
        completion_anchor = pl.read(group_out, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + tp_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_out, [0, 0], completion_anchor)

    return group_out, gather_signal


prefill_cp_token_allgather_step = pl.jit.inline(_prefill_cp_token_allgather_step)
prefill_cp_token_allgather_fixture = pl.jit(_prefill_cp_token_allgather_step)


@pl.jit.host
def l3_prefill_cp_token_allgather_fixture(
    hidden_local: pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[FIXTURE_ROUNDS, TP_SIZE, CP_KV_T_DYN, D], pl.BF16]],
):
    """Launch two all-gather rounds on one retained TP window."""
    hidden_local.bind_dynamic(2, CP_Q_T_DYN)
    group_out.bind_dynamic(2, CP_KV_T_DYN)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for round_id in pl.range(FIXTURE_ROUNDS):
        for rank in pl.range(pld.world_size()):
            gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
            gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
            prefill_cp_token_allgather_fixture(
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


def build_tensor_specs_allgather(local_t=FIXTURE_LOCAL_T):
    """Build two distinct rounds of per-rank inputs."""
    import torch

    from golden import TensorSpec

    if local_t < 1 or local_t > PREFILL_LOCAL_CAP:
        raise ValueError(f"local_t must be in [1, {PREFILL_LOCAL_CAP}], got {local_t}")
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
        TensorSpec("group_out", [FIXTURE_ROUNDS, TP_SIZE, group_t, D], torch.bfloat16),
    ]


def golden_prefill_cp_token_allgather(tensors):
    """Every rank receives its round's rank-major concatenation."""
    hidden_local = tensors["hidden_local"]
    rounds, tp_size, local_t, _ = hidden_local.shape
    gathered = hidden_local.reshape(rounds, tp_size * local_t, D)
    tensors["group_out"][:] = gathered.unsqueeze(1)


def validate_metadata():
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=prefill_metadata_test,
        specs=build_tensor_specs_metadata(),
        golden_fn=golden_prefill_metadata,
        compile_only=args.compile_only,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)



def validate_allgather():
    import argparse

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="Standalone context-parallel prefill token-row all-gather test.")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument(
        "--local-t", type=int, default=FIXTURE_LOCAL_T,
        help=f"per-rank token count, 1..{PREFILL_LOCAL_CAP}",
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
    if not 1 <= args.local_t <= PREFILL_LOCAL_CAP:
        parser.error(f"--local-t must be in [1, {PREFILL_LOCAL_CAP}], got {args.local_t}")

    result = run(
        fn=l3_prefill_cp_token_allgather_fixture,
        specs=build_tensor_specs_allgather(args.local_t),
        golden_fn=golden_prefill_cp_token_allgather,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    from models.deepseek_v4_flash_lowlat.build import run_operator_validations

    run_operator_validations(
        {
            'metadata': validate_metadata,
            'allgather': validate_allgather,
        },
        single_device=('metadata',),
        automatic=('allgather', 'metadata'),
    )
