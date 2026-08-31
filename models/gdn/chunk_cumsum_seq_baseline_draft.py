# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Baseline: the reference's sequential prefix-sum scan, ported to PyPTO.

Straight port of megagdn-pto kernels/pto/chunk_cumsum.cpp: a per-row recurrence
acc += g[i]; g_sum[i] = acc, one add per token, no cube.

The reference keeps the chunk UB-resident and mutates the accumulator in place,
storing the [C, H] block once. PyPTO is functional at tensor level, so each row
is stored as it is produced; the arithmetic is identical.

Exists only to separate framework from algorithm: measured against
models/gdn/chunk_cumsum.py (matmul scan) at the same shape on the same device.
"""
import pypto.language as pl

T = 32768
H = 16
CHUNK = 128


@pl.jit
def gdn_chunk_cumsum_seq(
    g: pl.Tensor[[T, H], pl.FP32],
    g_sum: pl.Out[pl.Tensor[[T, H], pl.FP32]],
):
    for t0 in pl.parallel(0, T, CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunk_cumsum_seq"):
            zero = pl.full([1, H], dtype=pl.FP32, value=0.0)
            for i, (acc, out) in pl.range(CHUNK, init_values=(zero, g_sum)):
                acc_next = pl.add(acc, pl.slice(g, [1, H], [t0 + i, 0]))
                out_next = pl.assemble(out, acc_next, [t0 + i, 0])
                acc_end, out_end = pl.yield_(acc_next, out_next)
            g_sum = out_end
    return g_sum


def build_tensor_specs(t: int = T, h: int = H):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("g", [t, h], torch.float32, init_value=torch.randn),
        TensorSpec("g_sum", [t, h], torch.float32, is_output=True),
    ]


def golden_seq(tensors):
    g = tensors["g"]
    out = tensors["g_sum"]
    for t0 in range(0, g.shape[0], CHUNK):
        out[t0 : t0 + CHUNK] = g[t0 : t0 + CHUNK].cumsum(dim=0)


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = run_jit(
        fn=gdn_chunk_cumsum_seq,
        specs=build_tensor_specs(),
        golden_fn=golden_seq,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
