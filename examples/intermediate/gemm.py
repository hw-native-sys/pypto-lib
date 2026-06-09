# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GEMM — tiled matrix multiply with M/N/K blocking.

    C[m, n] = A[m, k] @ B[k, n]

M and N are tiled across core-groups via nested pl.parallel; K is reduced with
pl.matmul on the first K-tile then pl.matmul_acc over the rest. FP32 in/out.
"""
import pypto.language as pl

M = 256         # total rows of A / C
N = 256         # total cols of B / C
K = 256         # total cols of A / rows of B
M_TILE = 64     # tile size along M dimension
N_TILE = 64     # tile size along N dimension
K_TILE = 64     # tile size along K dimension (reduction)


@pl.jit
def gemm(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gemm_tile"):
                acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
                for kb in pl.range(K // K_TILE):
                    k0 = kb * K_TILE
                    tile_a = a[mb : mb + M_TILE, k0 : k0 + K_TILE]
                    tile_b = b[k0 : k0 + K_TILE, nb : nb + N_TILE]
                    if kb == 0:
                        acc = pl.matmul(tile_a, tile_b)
                    else:
                        acc = pl.matmul_acc(acc, tile_a, tile_b)
                c[mb : mb + M_TILE, nb : nb + N_TILE] = acc
    return c


def build_tensor_specs(
    m: int = M,
    n: int = N,
    k: int = K,
):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("a", [m, k], torch.float32, init_value=torch.randn),
        TensorSpec("b", [k, n], torch.float32, init_value=torch.randn),
        TensorSpec("c", [m, n], torch.float32, is_output=True),
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
        fn=gemm,
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
