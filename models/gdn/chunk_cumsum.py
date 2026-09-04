# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_cumsum: chunk-local prefix sum of the gate logits,
g_sum[t, h] = sum over i <= t within the chunk of g[i, h]."""
import pypto.language as pl

# model config
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # gate heads
D = 128                 # head dimension; unused here, drawn inputs match the pipeline
CHUNK = 128             # chunk size in tokens

# tiling
GROUP_TILE = 16         # chunks per dispatch, sharing one tril load


def group_tile(t: int = T, chunk: int = CHUNK, want: int = GROUP_TILE) -> int:
    """Largest group size up to *want* that divides the chunk count evenly."""
    group = min(want, t // chunk)
    while (t // chunk) % group:
        group -= 1
    return group


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    """The stage kernel at one shape; `d` is unused and accepted for a uniform signature."""
    group = group_tile(t, chunk)

    @pl.jit
    def gdn_chunk_cumsum(
        g: pl.Tensor[[t, h], pl.FP32],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        g_sum: pl.Out[pl.Tensor[[h, t], pl.FP32]],
    ):
        for c0 in pl.spmd(t // (chunk * group), name_hint="chunk_cumsum"):
            t0 = c0 * chunk * group
            tl = tril[:, :]                        # constant, held for the whole scope
            for c in pl.unroll(group):
                s0 = t0 + c * chunk
                # [H, CHUNK] = (tril @ g_chunk)^T; the cube absorbs both transposes
                g_sum[:, s0 : s0 + chunk] = pl.matmul(g[s0 : s0 + chunk, :], tl, a_trans=True, b_trans=True)
        return g_sum

    return gdn_chunk_cumsum


gdn_chunk_cumsum = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    from models.gdn import reference

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    return [
        TensorSpec("g", [t, h], torch.float32,
                   init_value=reference.lazy("chunk_cumsum", "g", t, h, d, chunk)),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("g_sum", [h, t], torch.float32, is_output=True),
    ]


def golden_gdn_chunk_cumsum(tensors):
    from models.gdn import reference

    g = tensors["g"]
    out = tensors["g_sum"]
    chunk = tensors["tril"].shape[0]
    out.copy_(reference.to_hT(reference.cumsum(g, chunk)))


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
