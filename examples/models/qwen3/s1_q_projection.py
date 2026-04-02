# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — s1: Q projection only.

GEMM-style: normed [batch, hidden] BF16 @ wq [hidden, hidden] BF16 -> q_proj [batch, hidden] FP32.
Uses matmul + matmul_acc for K-dimension reduction in a single incore block,
following the same pattern as examples/intermediate/gemm.py.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 5120

K_CHUNK = 128
Q_OUT_CHUNK = 64
BATCH_TILE = 16


def build_q_projection_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
):
    hidden = hidden_size
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK

    @pl.program
    class QProjectionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def q_projection(
            self,
            normed: pl.Tensor[[batch, hidden], pl.BF16],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                for ob in pl.range(q_out_blocks):
                    q0 = ob * Q_OUT_CHUNK

                    with pl.incore():
                        # First K-tile: initialize accumulator via matmul.
                        tile_a = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, 0])
                        tile_b = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                        acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        # Remaining K-tiles: accumulate via matmul_acc.
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, k0])
                            tile_b_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            acc = pl.matmul_acc(acc, tile_a_i, tile_b_i)

                        q_proj = pl.assemble(q_proj, acc, [b0, q0])

            return q_proj

    return QProjectionProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("normed", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("q_proj", [batch, hidden_size], torch.float32, is_output=True),
    ]


def golden_q_projection(tensors, params):
    """PyTorch reference: BF16 matmul with FP32 output."""
    tensors["q_proj"][:] = (tensors["normed"].float() @ tensors["wq"].float()).float()


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_q_projection_program(batch=batch, hidden_size=hidden_size)
    tensor_specs = build_tensor_specs(batch=batch, hidden_size=hidden_size)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_q_projection,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            enable_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        enable_profiling=args.enable_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
