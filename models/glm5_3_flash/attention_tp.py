# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""The shared TP all-reduce over row-parallel partial sums.

Four kernels finish a layer by producing an FP32 partial of the full hidden width
rather than a finished activation: ``kda_output``, ``mla_epilog_prefill``,
``mla_epilog_decode`` and ``dense_mlp``. All four are row-parallel — they contract
over a head or intermediate axis that TP split — so each rank holds a summand, and
the layer is not complete until the group's summands are added. The shared expert
(``expert_shared.py``) joins the same reduction.

This file owns that addition, once, for all of them.

The protocol is the hand-rolled mesh the sibling port uses
(``models/deepseek_v4_1_flash/attention_tp.py``): each rank publishes its partial
into **its own** window slot at ``[row, col]``, notifies every group peer, waits
for theirs, then sums them with ``remote_load``. The window holds one partial, not
one slot per rank — the rank axis is the peer argument, so every transfer is a 2-D
remote op. Two notifies per rank per call (publish, then release) let the same
window be reused across layers: the ``reuse`` wait at the top fences the previous
call's readers, and ``consumed`` at the bottom fences this call's readers before
the next publish.

The group is the whole EP world (``group_base`` 0, ``DP = 1``), which is the
single-node A3 deployment; ``EP_SIZE == TP_SIZE`` is enforced at import so a DP
split fails loudly instead of summing each TP shard once per replica.

The L3 entry runs the reduction twice over one set of windows — a single call
would not exercise the reuse/consumed fences — and compares both passes against
``golden_tp_all_reduce``.
"""

import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch

# ``--tp`` / ``--ep`` are read from argv by config.py at import time and every
# sub-kernel inherits the frozen shapes. config.py's own default is the 16-die
# deployment shape, while the CI sweeps run each entry at its default world size
# (ep2 / 2-card, see ``# ci: devices=2``), so pin that bring-up default before
# the first ``models.glm5_3_flash`` import when the command line pins neither axis.
if not any(tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv):
    sys.argv += ["--tp", "2", "--ep", "2"]

from models.glm5_3_flash.config import D, EP_SIZE, T_DYN, TP_SIZE

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TP_CHUNK = 512           # reduction tile width; D is a whole number of these
TP_WORKERS = 8           # AIV blocks walking the row/chunk tiles
TP_CAP_TEST = 16         # window capacity exercised by the test entry

if EP_SIZE != TP_SIZE:
    raise ValueError(
        f"tp_all_reduce reduces over the EP world, so EP{EP_SIZE} with TP{TP_SIZE} would sum "
        "each TP shard once per DP replica. Reduce over the TP group (TP_SIZE peers at "
        "group_base) before enabling DP > 1."
    )


def golden_tp_all_reduce(partials: torch.Tensor) -> torch.Tensor:
    """Sum one ``[TP_SIZE, T, D]`` stack of per-rank partials into ``[T, D]``.

    The golden takes the stack because a single-process reference cannot observe a
    collective; the kernel takes one rank's slice and the window.
    """
    if partials.shape[0] != TP_SIZE:
        raise ValueError(f"expected {TP_SIZE} partials, got {partials.shape[0]}")
    return partials.float().sum(dim=0).to(torch.bfloat16)


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
            pld.system.wait(
                arrived, offsets=[peer, 0],
                expected=pl.cast(epoch - 1, pl.INT32) * 2, cmp=pld.WaitCmp.Ge,
            )

    with pl.spmd(TP_WORKERS, name_hint="tp_publish", deps=[reuse_tid], allow_early_resolve=False) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, active * (D // TP_CHUNK), TP_WORKERS):
            row = tile // (D // TP_CHUNK)
            col = tile % (D // TP_CHUNK) * TP_CHUNK
            value = pl.load(partial, [row, col], [1, TP_CHUNK])
            pl.store(value, [row, col], exchange)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.wait(
                arrived, offsets=[peer, 0],
                expected=pl.cast(epoch * 2 - 1, pl.INT32), cmp=pld.WaitCmp.Ge,
            )

    with pl.spmd(TP_WORKERS, name_hint="tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, active * (D // TP_CHUNK), TP_WORKERS):
            row = tile // (D // TP_CHUNK)
            col = tile % (D // TP_CHUNK) * TP_CHUNK
            acc = pl.tile.full([1, TP_CHUNK], dtype=pl.FP32, value=0.0)
            for peer in pl.range(EP_SIZE):
                peer_value = pld.tile.remote_load(
                    exchange, peer=group_base + peer, offsets=[row, col], shape=[1, TP_CHUNK],
                )
                acc = pl.add(acc, peer_value)
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(EP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + peer, offsets=[tp_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(EP_SIZE):
            pld.system.wait(
                arrived, offsets=[peer, 0],
                expected=pl.cast(epoch * 2, pl.INT32), cmp=pld.WaitCmp.Ge,
            )

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
                partials[r], exchange, arrived, output[r],
                num_tokens, r, pl.cast(call + 1, pl.INT32),
                device=r,
            )


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

    from golden import ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size (config.py reads argv)")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)),
                        help=f"comma-separated device ids (need {EP_SIZE})")
    parser.add_argument("--num-tokens", type=int, default=TP_CAP_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == EP_SIZE, f"need exactly {EP_SIZE} devices, got {device_ids}"

    torch.manual_seed(17)
    partials = torch.randn(TP_SIZE, 7, 5)
    reduced = golden_tp_all_reduce(partials)
    torch.testing.assert_close(reduced, partials.float().sum(dim=0).to(torch.bfloat16))

    result = run(
        fn=l3_tp_all_reduce,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_l3_tp_all_reduce,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # The kernel only writes rows < num_tokens; the capacity tail of
            # ``output`` is never touched and may decode as NaN/Inf, so clip the
            # comparison to the active extent (valid_axis=1: the rank axis
            # precedes the token axis).
            "output": ratio_reldiff(diff_thd=3e-3, pct_thd=0.01,
                                    valid_rows=args.num_tokens, valid_axis=1),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "golden_tp_all_reduce",
    "tp_all_reduce",
]
