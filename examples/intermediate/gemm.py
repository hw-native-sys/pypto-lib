# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GEMM — tiled matrix multiplication with M/N/K blocking.

    C[m, n] = A[m, k] @ B[k, n]

Authored in the ``@pl.jit`` form (see docs/pypto-coding-style.md §1):
``gemm`` is a reusable ``@pl.jit.inline`` sub-kernel called by the thin
``@pl.jit`` entry. ``pl.parallel`` tiles the M and N output dimensions
across core-groups; inside each ``pl.at`` scope ``pl.pipeline(stage=2)``
software-pipelines the K reduction — the canonical matmul-reduction shape
(see examples/advanced/multi_proj.py): ``pl.matmul`` seeds the FP32
accumulator on the first K-tile, ``pl.matmul_acc`` adds the rest.

Input and output matrices are FP32.
"""
import pypto.language as pl

# ---------------------------------------------------------------------------
# GEMM parameters — edit these to change problem size and tiling
# ---------------------------------------------------------------------------
M = 256         # total rows of A / C
N = 256         # total cols of B / C
K = 256         # total cols of A / rows of B
M_TILE = 64     # tile size along M dimension
N_TILE = 64     # tile size along N dimension
K_TILE = 64     # tile size along K dimension (reduction)

K_BLOCKS = K // K_TILE


@pl.jit.inline
def gemm(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gemm"):
                acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    tile_a = a[mb : mb + M_TILE, k0 : k0 + K_TILE]
                    tile_b = b[k0 : k0 + K_TILE, nb : nb + N_TILE]
                    if k0 == 0:
                        # First K-tile seeds the accumulator
                        acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
                    else:
                        # Remaining K-tiles accumulate in place
                        acc = pl.matmul_acc(acc, tile_a, tile_b)
                c[mb : mb + M_TILE, nb : nb + N_TILE] = acc

    return c


@pl.jit
def gemm_kernel(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    c = gemm(a, b, c)
    return c


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("a", [M, K], torch.float32, init_value=torch.randn),
        TensorSpec("b", [K, N], torch.float32, init_value=torch.randn),
        TensorSpec("c", [M, N], torch.float32, is_output=True),
    ]


def golden_gemm(tensors):
    tensors["c"][:] = tensors["a"] @ tensors["b"]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=gemm_kernel,
        specs=build_tensor_specs(),
        golden_fn=golden_gemm,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
