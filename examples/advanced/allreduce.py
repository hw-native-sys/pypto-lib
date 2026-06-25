# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""L3 multi-card mesh all-reduce (``@pl.jit`` / ``@pl.jit.host``).

Every rank contributes a row and ends up holding the element-wise sum of all
rows, computed over the L3 distributed stack: HCCL window buffers, notify/wait
barriers, and remote tile loads. Fixed at P=2 (two ranks).

Run:  python examples/advanced/allreduce.py -p a2a3 -d 0,1
"""

import pypto.language as pl
import pypto.language.distributed as pld

SIZE = 256  # element-wise reduction length per rank
N_RANKS = 2  # this example runs P=2 only; the window shapes need it statically


@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Mesh all-reduce on window-bound ``data`` / ``signal``."""
    # Stage the local input into this rank's window slice.
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)

    # Barrier: notify every peer, then wait on every peer slot. The window
    # buffer is zero-initialised, so AtomicAdd + Ge(1) is safe.
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal=signal,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # Load my own slice, then add every peer's slice via remote_load.
    acc = pl.load(data, [0, 0], [1, SIZE])
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
            acc = pl.add(acc, recv)

    # Store the reduced accumulator into the local output.
    return pl.store(acc, [0, 0], out)


@pl.jit
def allreduce(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Per-device orchestration wrapper around ``reduce_step``."""
    return reduce_step(inp, out, data, signal, my_rank)


@pl.jit.host
def l3_allreduce(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    """Launch one chip orchestration per rank, sharing the window buffers."""
    data_buf = pld.alloc_window_buffer(SIZE * 4)  # 1xSIZE x FP32 (4 bytes)
    signal_buf = pld.alloc_window_buffer(N_RANKS * 4)  # NR x 1 x INT32

    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        allreduce(inputs[r], outputs[r], data, signal, r, device=r)


def build_tensor_specs():
    """Distinct per-rank input rows so the reduced sum is non-trivial."""
    import torch

    from golden import TensorSpec

    def init_inputs():
        rows = [
            torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
            for r in range(N_RANKS)
        ]
        return torch.stack(rows)

    return [
        TensorSpec("inputs",  [N_RANKS, 1, SIZE], torch.float32, init_value=init_inputs),
        TensorSpec("outputs", [N_RANKS, 1, SIZE], torch.float32, is_output=True),
    ]


def golden_allreduce(tensors):
    """Every rank ends up holding the element-wise sum of all rank inputs."""
    reduced = tensors["inputs"].sum(dim=0, keepdim=True)  # [1, 1, SIZE]
    tensors["outputs"][:] = reduced.expand_as(tensors["outputs"])


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help=f"comma-separated device ids (need exactly {N_RANKS})")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_allreduce,
        specs=build_tensor_specs(),
        golden_fn=golden_allreduce,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
