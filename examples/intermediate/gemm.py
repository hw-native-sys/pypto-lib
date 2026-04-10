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

M and N are parallelised via pl.parallel; K is tiled and reduced using
pl.matmul (first K-tile) + pl.matmul_acc (remaining K-tiles).

Input and output matrices are FP32.
"""
from __future__ import annotations

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
M_CHUNK = 2     # M-tiles grouped per incore chunk
N_CHUNK = 2     # N-tiles grouped per incore chunk


def build_gemm_program(
    m: int = M,
    n: int = N,
    k: int = K,
    m_tile: int = M_TILE,
    n_tile: int = N_TILE,
    k_tile: int = K_TILE,
    m_chunk: int = M_CHUNK,
    n_chunk: int = N_CHUNK,
):
    k_blocks = k // k_tile

    @pl.program
    class GemmProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def gemm(
            self,
            a: pl.Tensor[[m, k], pl.FP32],
            b: pl.Tensor[[k, n], pl.FP32],
            c: pl.Out[pl.Tensor[[m, n], pl.FP32]],
        ) -> pl.Tensor[[m, n], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for mb in pl.parallel(0, m, m_tile, chunk=m_chunk):
                    for nb in pl.parallel(0, n, n_tile, chunk=n_chunk):
                        # First K-tile: initialize accumulator via matmul
                        tile_a = pl.slice(a, [m_tile, k_tile], [mb, 0])
                        tile_b = pl.slice(b, [k_tile, n_tile], [0, nb])
                        acc = pl.matmul(tile_a, tile_b)

                        # Remaining K-tiles: accumulate via matmul_acc
                        for kb in pl.range(1, k_blocks):
                            k0 = kb * k_tile
                            tile_a_i = pl.slice(a, [m_tile, k_tile], [mb, k0])
                            tile_b_i = pl.slice(b, [k_tile, n_tile], [k0, nb])
                            acc = pl.matmul_acc(acc, tile_a_i, tile_b_i)

                        c = pl.assemble(c, acc, [mb, nb])

            return c

    return GemmProgram


def build_tensor_specs(
    m: int = M,
    n: int = N,
    k: int = K,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("a", [m, k], torch.float32, init_value=torch.randn),
        TensorSpec("b", [k, n], torch.float32, init_value=torch.randn),
        TensorSpec("c", [m, n], torch.float32, is_output=True),
    ]


def golden_gemm(tensors, params):
    tensors["c"][:] = tensors["a"] @ tensors["b"]


def compile_and_run(
    m: int = M,
    n: int = N,
    k: int = K,
    m_tile: int = M_TILE,
    n_tile: int = N_TILE,
    k_tile: int = K_TILE,
    m_chunk: int = M_CHUNK,
    n_chunk: int = N_CHUNK,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_gemm_program(
        m=m, n=n, k=k,
        m_tile=m_tile, n_tile=n_tile, k_tile=k_tile,
        m_chunk=m_chunk, n_chunk=n_chunk,
    )
    tensor_specs = build_tensor_specs(m=m, n=n, k=k)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_gemm,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        runtime_profiling=args.runtime_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
