# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm — Root Mean Square layer normalization with row + column tiling.

    output[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * gamma[c]

Rows are parallelised via pl.parallel (batch dimension).
The hidden dimension is chunked with pl.range to accumulate the
squared-sum reduction, then a second pass normalises and applies gamma.

This two-pass column-chunking pattern is the standard approach used
in production LLM kernels (see qwen3/deepseek examples) where the
hidden dimension exceeds on-chip buffer capacity.

Input and output are FP32; gamma is a [1, hidden] weight vector.
"""
from __future__ import annotations

import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 512            # hidden dimension (normalised axis)
ROW_CHUNK = 64          # rows per parallel tile
HIDDEN_CHUNK = 64       # columns per sequential chunk
EPS = 1e-6


def build_rms_norm_program(
    rows: int = ROWS,
    hidden: int = HIDDEN,
    row_chunk: int = ROW_CHUNK,
    hidden_chunk: int = HIDDEN_CHUNK,
    eps: float = EPS,
):
    hidden_blocks = hidden // hidden_chunk
    hidden_inv = 1.0 / hidden

    @pl.program
    class RMSNormProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def rms_norm(
            self,
            x: pl.Tensor[[rows, hidden], pl.FP32],
            gamma: pl.Tensor[[1, hidden], pl.FP32],
            y: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        ) -> pl.Tensor[[rows, hidden], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for r in pl.parallel(0, rows, row_chunk, chunk=1):
                    # Pass 1: accumulate sum(x^2) across hidden chunks
                    # row_sum produces [row_chunk, 1] col_major; scalar ops
                    # need row_major, so accumulate in [1, row_chunk] shape.
                    sq_sum = pl.create_tensor([1, row_chunk], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for hb in pl.range(hidden_blocks):
                        h0 = hb * hidden_chunk
                        x_chunk = pl.slice(x, [row_chunk, hidden_chunk], [r, h0])
                        rs = pl.row_sum(pl.mul(x_chunk, x_chunk))
                        sq_sum = pl.add(sq_sum, pl.reshape(rs, [1, row_chunk]))

                    # inv_rms = 1 / sqrt(mean(x^2) + eps)
                    inv_rms_T = pl.rsqrt(pl.add(pl.mul(sq_sum, hidden_inv), eps))
                    inv_rms = pl.reshape(inv_rms_T, [row_chunk, 1])

                    # Pass 2: normalise and apply gamma weight
                    for hb in pl.range(hidden_blocks):
                        h0 = hb * hidden_chunk
                        x_chunk = pl.slice(x, [row_chunk, hidden_chunk], [r, h0])
                        gamma_chunk = pl.slice(gamma, [1, hidden_chunk], [0, h0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, inv_rms), gamma_chunk
                        )
                        y = pl.assemble(y, normed, [r, h0])

            return y

    return RMSNormProgram


def build_tensor_specs(
    rows: int = ROWS,
    hidden: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("x", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch.float32, is_output=True),
    ]


def golden_rms_norm(tensors, params):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    tensors["y"][:] = x / rms * gamma


def compile_and_run(
    rows: int = ROWS,
    hidden: int = HIDDEN,
    row_chunk: int = ROW_CHUNK,
    hidden_chunk: int = HIDDEN_CHUNK,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_rms_norm_program(
        rows=rows,
        hidden=hidden,
        row_chunk=row_chunk,
        hidden_chunk=hidden_chunk,
    )
    tensor_specs = build_tensor_specs(
        rows=rows,
        hidden=hidden,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_rms_norm,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-2,
            atol=1e-2,
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
