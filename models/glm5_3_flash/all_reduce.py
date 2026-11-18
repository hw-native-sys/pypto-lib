# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""GLM-5.3-Flash tensor-parallel all-reduce."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__" and not any(
    tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv
):
    sys.argv += ["--tp", "2", "--ep", "2"]

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.glm5_3_flash.config import EP_SIZE, TP_SIZE, D

T_DYN = pl.dynamic('GLM53_T_DYN')

TP_D_TILE = 512  # reduction tile width; D is a whole number of these
TP_WORKERS = 8  # AIV blocks walking the row/chunk tiles
TP_CAP_TEST = 16  # window capacity exercised by the test entry
if EP_SIZE != TP_SIZE:
    raise ValueError(
        f"tp_all_reduce reduces over the EP world, so EP{EP_SIZE} with TP{TP_SIZE} would sum "
        "each TP shard once per DP replica. Reduce over the TP group (TP_SIZE peers at "
        "group_base) before enabling DP > 1."
    )


@pl.jit.inline(auto_scope=False)
def tp_all_reduce(
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    exchange: pld.DistributedTensor[[T_DYN, D], pl.FP32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Reduce every group rank's row-parallel partial into ``output``.

    ``arrived`` counts, per group-local rank, one publish notify and one release
    notify per call; ``epoch`` is the 1-based call id, so the third phase's
    expected value is ``2 * epoch`` and the first phase's is ``2 * (epoch - 1)``.
    """
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=pl.cast(epoch - 1, pl.INT32) * 2, cmp=pld.WaitCmp.Ge)

    with pl.spmd(TP_WORKERS, name_hint="tp_publish", deps=[reuse_tid], allow_early_resolve=False) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, active * (D // TP_D_TILE), TP_WORKERS):
            row = tile // (D // TP_D_TILE)
            col = tile % (D // TP_D_TILE) * TP_D_TILE
            value = pl.load(partial, [row, col], [1, TP_D_TILE])
            pl.store(value, [row, col], exchange)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_wait", deps=[ready_tid], allow_early_resolve=False) as wait_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=pl.cast(epoch * 2 - 1, pl.INT32), cmp=pld.WaitCmp.Ge)

    with pl.spmd(TP_WORKERS, name_hint="tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, active * (D // TP_D_TILE), TP_WORKERS):
            row = tile // (D // TP_D_TILE)
            col = tile % (D // TP_D_TILE) * TP_D_TILE
            acc = pl.tile.full([1, TP_D_TILE], dtype=pl.FP32, value=0.0)
            correction = pl.tile.full([1, TP_D_TILE], dtype=pl.FP32, value=0.0)
            for peer in pl.range(EP_SIZE):
                peer_value = pld.tile.remote_load(
                    exchange, peer=group_base + peer, offsets=[row, col], shape=[1, TP_D_TILE],
                )
                updated = pl.add(acc, peer_value)
                recovered = pl.sub(updated, acc)
                error = pl.add(pl.sub(acc, pl.sub(updated, recovered)), pl.sub(peer_value, recovered))
                correction = pl.add(correction, error)
                acc = updated
            output = pl.store(pl.cast(pl.add(acc, correction), pl.BF16, mode="rint"), [row, col], output)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.notify(arrived, peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_consumed", deps=[release_tid], allow_early_resolve=False):
        for peer in pl.range(EP_SIZE):
            pld.system.wait(arrived, offsets=[peer, 0], expected=pl.cast(epoch * 2, pl.INT32), cmp=pld.WaitCmp.Ge)

    return output


@pl.jit(auto_scope=False)
def l2_tp_all_reduce(
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    exchange: pld.DistributedTensor[[TP_CAP_TEST, D], pl.FP32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Per-rank wrapper: fix the group base at 0 (single DP group)."""
    tp_all_reduce(partial, exchange, arrived, output, num_tokens, tp_rank, 0, epoch)
    return output


@pl.jit.host
def l3_tp_all_reduce(
    partials: pl.Tensor[[EP_SIZE, TP_CAP_TEST, D], pl.FP32],
    output: pl.Out[pl.Tensor[[EP_SIZE, TP_CAP_TEST, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Launch one rank's orchestration per chip, sharing the window buffers."""
    exchange_buf = pld.alloc_window_buffer([TP_CAP_TEST, D], dtype=pl.FP32)
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)

    # Two calls over the same windows: the second one only retires if the
    # reuse/consumed fences really order call N's readers before call N+1's
    # publish, so the single call would not cover the multi-layer protocol.
    for call in pl.range(2):
        for r in pl.range(pld.world_size()):
            exchange = pld.window(exchange_buf, [TP_CAP_TEST, D], dtype=pl.FP32)
            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            l2_tp_all_reduce(
                partials[r], exchange, arrived, output[r], num_tokens, r, pl.cast(call + 1, pl.INT32),
                device=r,
            )


def golden_tp_all_reduce(partials: torch.Tensor) -> torch.Tensor:
    """Sum one ``[TP_SIZE, T, D]`` stack of per-rank partials into ``[T, D]``.

    The golden takes the stack because a single-process reference cannot observe a
    collective; the kernel takes one rank's slice and the window.
    """
    if partials.shape[0] != TP_SIZE:
        raise ValueError(f"expected {TP_SIZE} partials, got {partials.shape[0]}")
    return partials.double().sum(dim=0).float().to(torch.bfloat16)


def golden_l3_tp_all_reduce(tensors):
    """Per-rank expected output: the group sum of every rank's partial, in BF16."""
    num_tokens = max(0, min(TP_CAP_TEST, int(tensors.get("num_tokens", TP_CAP_TEST))))
    out = torch.zeros_like(tensors["output"])
    for r in range(EP_SIZE):
        out[r, :num_tokens] = golden_tp_all_reduce(tensors["partials"][:, :num_tokens])
    tensors["output"][:] = out


def build_tensor_specs(num_tokens=TP_CAP_TEST):
    from golden import ScalarSpec, TensorSpec

    num_tokens = max(0, min(TP_CAP_TEST, int(num_tokens)))
    gen = torch.Generator().manual_seed(11)

    def init_partials():
        return torch.randn(EP_SIZE, TP_CAP_TEST, D, generator=gen)

    return [
        TensorSpec("partials", [EP_SIZE, TP_CAP_TEST, D], torch.float32, init_value=init_partials),
        TensorSpec("output", [EP_SIZE, TP_CAP_TEST, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size (config.py reads argv)")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument(
        "-d", "--device", type=str, default=",".join((str(i) for i in range(EP_SIZE))),
        help=f"comma-separated device ids (need {EP_SIZE})",
    )
    parser.add_argument("--num-tokens", type=int, default=TP_CAP_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == EP_SIZE, f"need exactly {EP_SIZE} devices, got {device_ids}"

    torch.manual_seed(17)
    partials = torch.randn(TP_SIZE, 7, 5)
    reduced = golden_tp_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.double().sum(dim=0).float().to(torch.bfloat16))

    result = run(
        fn=l3_tp_all_reduce,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_l3_tp_all_reduce,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type, execution_mode=args.execution_mode,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=0.001,
        atol=0.001,
        compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01, valid_rows=args.num_tokens, valid_axis=1)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
