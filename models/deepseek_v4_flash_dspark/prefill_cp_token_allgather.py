# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Context-parallel prefill token-row all-gather into rank-major order on every rank."""

from math import gcd

import sys

# runtime configuration
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

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M, MOE_TOKENS, DECODE_SEQ


# Dynamic shape variables.
CP_Q_T_DYN = pl.dynamic("PREFILL_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_CP_KV_T_DYN")
CP_COEFF_DYN = pl.dynamic("PREFILL_CP_COEFF_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult

# communication bounds
PREFILL_GROUP_CAP = 8192
PREFILL_LOCAL_CAP = PREFILL_GROUP_CAP // TP_SIZE

# tiling
COMM_ROW_TILE = 8
READBACK_ROW_TILE = gcd(MOE_TOKENS // DECODE_SEQ * (DECODE_SEQ - 1), 16)
READBACK_WORKERS = 16
COEFF_ROW_TILE = 4

# fixture
FIXTURE_ROUNDS = 2
FIXTURE_LOCAL_T = min(257, PREFILL_LOCAL_CAP)


@pl.jit.inline
def prefill_cp_token_allgather_into(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
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

    # Retire peer credits and anchor raw output words to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_retire",
        deps=[_readback_notify_tid, _readback_wait_tid],
    ) as retire_tid:
        anchor_bits = pl.load(group_out, [0, 0], [1, 16])
        anchor_words = pl.reinterpret_view(anchor_bits, pl.INT32)
        completion_anchor = pl.tile.read(anchor_words, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + tp_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
        pl.tile.write(anchor_words, [0, 0], completion_anchor)
        anchor_output = pl.reinterpret_view(anchor_words, pl.BF16)
        pl.store(anchor_output, [0, 0], group_out)

    return retire_tid


def _prefill_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather BF16 rows through the shared retained-window protocol."""
    retire_tid = prefill_cp_token_allgather_into(
        hidden_local, group_out, gather_window, gather_signal, group_base, tp_rank,
    )

    return group_out, gather_signal


prefill_cp_token_allgather_step = pl.jit.inline(_prefill_cp_token_allgather_step)
prefill_cp_token_allgather_fixture = pl.jit(_prefill_cp_token_allgather_step)


@pl.jit.inline(auto_scope=False)
def prefill_cp_materialize_owner_hc(
    local_hc: pl.Tensor[[CP_Q_T_DYN, HC_MULT, D], pl.FP32],
    full_hc: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Materialize rank-major FP32 HC rows through serialized raw-word epochs."""
    local_rows = pl.tensor.dim(local_hc, 0)
    group_rows = pl.tensor.dim(full_hc, 0)
    raw_local_rows = local_rows * HC_MULT * 2
    local_flat_rows = local_rows * HC_MULT
    group_flat_rows = group_rows * HC_MULT
    local_flat = pl.reshape(local_hc, [local_flat_rows, D])
    full_flat = pl.reshape(full_hc, [group_flat_rows, D])
    completion_tids = pl.array.create(1, pl.TASK_ID)
    completion_tids[0] = pl.system.task_dummy(deps=[])
    for raw_base in pl.range(0, raw_local_rows, PREFILL_LOCAL_CAP):
        with pl.scope():
            chunk_rows = pl.min(PREFILL_LOCAL_CAP, raw_local_rows - raw_base)
            local_packet = pl.create_tensor([chunk_rows, D], dtype=pl.BF16)
            with pl.spmd(
                READBACK_WORKERS, name_hint="prefill_cp_hc_pack", deps=[completion_tids[0]],
            ) as pack_tid:
                pack_worker = pl.tile.get_block_idx()
                for pack_row in pl.range(pack_worker, chunk_rows, READBACK_WORKERS):
                    source_row = raw_base + pack_row
                    source_hc_row = source_row // 2
                    source_col = (source_row % 2) * (D // 2)
                    source_tile = pl.load(local_flat, [source_hc_row, source_col], [1, D // 2])
                    raw_row = pl.reinterpret_view(source_tile, pl.BF16)
                    pl.store(raw_row, [pack_row, 0], local_packet)

            packet_group_rows = TP_SIZE * chunk_rows
            group_packet = pl.create_tensor([packet_group_rows, D], dtype=pl.BF16)
            gather_tid = prefill_cp_token_allgather_into(
                local_packet, group_packet, gather_window, gather_signal, group_base, tp_rank,
            )
            with pl.spmd(
                READBACK_WORKERS, name_hint="prefill_cp_hc_unpack", deps=[gather_tid],
            ) as unpack_tid:
                unpack_worker = pl.tile.get_block_idx()
                for packet_row in pl.range(unpack_worker, TP_SIZE * chunk_rows, READBACK_WORKERS):
                    source_rank = packet_row // chunk_rows
                    source_row = packet_row % chunk_rows
                    output_row = source_rank * raw_local_rows + raw_base + source_row
                    raw_row = pl.load(group_packet, [packet_row, 0], [1, D])
                    output_tile = pl.reinterpret_view(raw_row, pl.FP32)
                    output_hc_row = output_row // 2
                    output_col = (output_row % 2) * (D // 2)
                    pl.store(output_tile, [output_hc_row, output_col], full_flat)
            completion_tids[0] = unpack_tid

    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_cp_hc_publish", deps=[completion_tids[0]],
    ) as publish_tid:
        anchor_tile = pl.load(full_flat, [0, 0], [1, 8])
        anchor_words = pl.reinterpret_view(anchor_tile, pl.INT32)
        anchor = pl.tile.read(anchor_words, [0, 0])
        pl.tile.write(anchor_words, [0, 0], anchor)
        anchor_output = pl.reinterpret_view(anchor_words, pl.FP32)
        pl.store(anchor_output, [0, 0], full_flat)
    return publish_tid


@pl.jit.inline(auto_scope=False)
def prefill_cp_materialize_owner_coeff(
    local_coeff: pl.Tensor[[CP_Q_T_DYN, CP_COEFF_DYN], pl.FP32],
    full_coeff: pl.Tensor[[CP_KV_T_DYN, CP_COEFF_DYN], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Materialize four- or sixteen-column FP32 coefficients as raw words."""
    local_rows = pl.tensor.dim(local_coeff, 0)
    group_rows = pl.tensor.dim(full_coeff, 0)
    coeff_cols = pl.tensor.dim(local_coeff, 1)
    with pl.scope():
        local_packet = pl.create_tensor([local_rows, D], dtype=pl.BF16)
        with pl.spmd(READBACK_WORKERS, name_hint="prefill_cp_coeff_pack") as pack_tid:
            coeff_pack_worker = pl.tile.get_block_idx()
            for token in pl.range(coeff_pack_worker, local_rows, READBACK_WORKERS):
                pack_source_tile = pl.load(local_coeff, [token, 0], [1, 16], valid_shape=[1, coeff_cols])
                pack_source_words = pl.reinterpret_view(pack_source_tile, pl.INT32)
                word_row = pl.tile.full([1, D // 2], dtype=pl.INT32, value=0)
                for column in pl.range(coeff_cols):
                    pack_word = pl.tile.read(pack_source_words, [0, column])
                    pl.tile.write(word_row, [0, column], pack_word)
                raw_row = pl.reinterpret_view(word_row, pl.BF16)
                pl.store(raw_row, [token, 0], local_packet)

        group_packet = pl.create_tensor([group_rows, D], dtype=pl.BF16)
        gather_tid = prefill_cp_token_allgather_into(
            local_packet, group_packet, gather_window, gather_signal, group_base, tp_rank,
        )
        unpack_tids = pl.array.create(1, pl.TASK_ID)
        if coeff_cols == 4:
            token_blocks = (group_rows + COEFF_ROW_TILE - 1) // COEFF_ROW_TILE
            with pl.spmd(token_blocks, name_hint="prefill_cp_coeff4_unpack", deps=[gather_tid]) as unpack4_tid:
                token0 = pl.tile.get_block_idx() * COEFF_ROW_TILE
                valid_rows = pl.min(COEFF_ROW_TILE, group_rows - token0)
                output4_words = pl.tile.full([COEFF_ROW_TILE, 8], dtype=pl.INT32, value=0)
                for row in pl.range(valid_rows):
                    source4_bits = pl.load(group_packet, [token0 + row, 0], [1, 32])
                    source4_words = pl.reinterpret_view(source4_bits, pl.INT32)
                    for column in pl.unroll(4):
                        word4 = pl.tile.read(source4_words, [0, column])
                        pl.tile.write(output4_words, [row, column], word4)
                output4_tile = pl.reinterpret_view(output4_words, pl.FP32)
                output4_valid = pl.set_validshape(output4_tile, valid_rows, 4)
                pl.store(output4_valid, [token0, 0], full_coeff)
            unpack_tids[0] = unpack4_tid
        else:
            with pl.spmd(group_rows, name_hint="prefill_cp_coeff16_unpack", deps=[gather_tid]) as unpack16_tid:
                output_token = pl.tile.get_block_idx()
                source16_bits = pl.load(group_packet, [output_token, 0], [1, 32])
                source16_words = pl.reinterpret_view(source16_bits, pl.INT32)
                output16_words = pl.tile.full([1, 16], dtype=pl.INT32, value=0)
                for column in pl.unroll(16):
                    word16 = pl.tile.read(source16_words, [0, column])
                    pl.tile.write(output16_words, [0, column], word16)
                output16_tile = pl.reinterpret_view(output16_words, pl.FP32)
                pl.store(output16_tile, [output_token, 0], full_coeff)
            unpack_tids[0] = unpack16_tid

        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="prefill_cp_coeff_publish", deps=[unpack_tids[0]],
        ) as publish_tid:
            anchor_tile = pl.load(full_coeff, [0, 0], [1, 16], valid_shape=[1, coeff_cols])
            anchor_words = pl.reinterpret_view(anchor_tile, pl.INT32)
            anchor = pl.tile.read(anchor_words, [0, 0])
            pl.tile.write(anchor_words, [0, 0], anchor)
            anchor_output = pl.reinterpret_view(anchor_words, pl.FP32)
            anchor_valid = pl.set_validshape(anchor_output, 1, coeff_cols)
            pl.store(anchor_valid, [0, 0], full_coeff)
    return publish_tid


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


def build_tensor_specs(local_t=FIXTURE_LOCAL_T):
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


if __name__ == "__main__":
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
        specs=build_tensor_specs(args.local_t),
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
