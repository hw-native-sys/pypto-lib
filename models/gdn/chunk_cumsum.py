# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_cumsum — chunk-local prefix sum of the gate logits.

    g_sum[t, h] = sum_{i in chunk, i <= t} g[i, h]

Stage 1 of the GDN pipeline. The prefix sum is what lets every later stage form
a decay coefficient as exp(g_sum[i] - g_sum[j]).

The scan is a matmul against a lower-triangular ones matrix: row t of `tril`
selects exactly the chunk-local rows up to t. Chunks are independent, so they
run one per core group. FP32 in and out — a 128-row accumulation in FP16 loses
the small increments that decide exp(g_sum) downstream.
"""
import pypto.language as pl

# Model
T = 32768               # tokens (single sequence, B = 1)
H = 16                  # gate heads
CHUNK = 128             # chunk size in tokens

# Tiling
CHUNK_GROUP = 16        # chunks sharing one tril load; T must divide CHUNK * CHUNK_GROUP


@pl.jit
def gdn_chunk_cumsum(
    g: pl.Tensor[[T, H], pl.FP32],
    tril: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    g_sum: pl.Out[pl.Tensor[[H, T], pl.FP32]],
):
    for t0 in pl.parallel(0, T, CHUNK * CHUNK_GROUP):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunk_cumsum"):
            # tril is 64 KB against 8 KB of gate per chunk, so it is loaded once
            # per scope and reused, not re-read per chunk.
            tl = tril[:, :]
            for c in pl.unroll(CHUNK_GROUP):
                s0 = t0 + c * CHUNK
                # [H, CHUNK] = (tril @ g_chunk)^T; the cube absorbs both transposes.
                g_sum[:, s0 : s0 + CHUNK] = pl.matmul(
                    g[s0 : s0 + CHUNK, :], tl, a_trans=True, b_trans=True)
    return g_sum


def build_tensor_specs(t: int = T, h: int = H, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    return [
        TensorSpec("g", [t, h], torch.float32, init_value=torch.randn),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("g_sum", [h, t], torch.float32, is_output=True),
    ]


def golden_gdn_chunk_cumsum(tensors):
    g = tensors["g"]
    out = tensors["g_sum"]
    for t0 in range(0, g.shape[0], CHUNK):
        out[:, t0 : t0 + CHUNK] = g[t0 : t0 + CHUNK].cumsum(dim=0).t()


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=gdn_chunk_cumsum,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_chunk_cumsum,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-4,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
