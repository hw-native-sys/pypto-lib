# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Softmax — row-wise numerically stable softmax with row-chunk tiling.

    output[r, c] = exp(x[r, c] - max_row(x)) / sum_row(exp(x[r, c] - max_row(x)))

The matrix is split into row chunks via pl.parallel so softmax is computed
independently on each chunk.  Each chunk's rows are self-contained because
softmax normalises across the column (hidden) dimension only.

Input and output are FP32.
"""
from __future__ import annotations

import pypto.language as pl

ROWS = 512
COLS = 256
ROW_CHUNK = 64          # rows per incore tile


def build_softmax_program(
    rows: int = ROWS,
    cols: int = COLS,
    row_chunk: int = ROW_CHUNK,
):
    @pl.program
    class SoftmaxProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def softmax(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            y: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for r in pl.parallel(0, rows, row_chunk, chunk=1):
                    tile_x = pl.slice(x, [row_chunk, cols], [r, 0])

                    # Step 1: row-wise max for numerical stability
                    row_max = pl.row_max(tile_x)

                    # Step 2: subtract row max: x - max(x)
                    shifted = pl.row_expand_sub(tile_x, row_max)

                    # Step 3: exp(x - max(x))
                    exp_shifted = pl.exp(shifted)

                    # Step 4: row-wise sum of exp values
                    row_sum = pl.row_sum(exp_shifted)

                    # Step 5: divide each row by its sum
                    result = pl.row_expand_div(exp_shifted, row_sum)

                    y = pl.assemble(y, result, [r, 0])

            return y

    return SoftmaxProgram


def build_tensor_specs(
    rows: int = ROWS,
    cols: int = COLS,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, cols], torch.float32, is_output=True),
    ]


def golden_softmax(tensors, params):
    import torch

    tensors["y"][:] = torch.softmax(tensors["x"], dim=-1)


def compile_and_run(
    rows: int = ROWS,
    cols: int = COLS,
    row_chunk: int = ROW_CHUNK,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_softmax_program(
        rows=rows,
        cols=cols,
        row_chunk=row_chunk,
    )
    tensor_specs = build_tensor_specs(
        rows=rows,
        cols=cols,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_softmax,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-5,
            atol=1e-5,
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
