# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LayerNorm — full layer normalization with row-only tiling.

    output[r, c] = (x[r, c] - mean(x[r, :])) / sqrt(var(x[r, :]) + eps) * gamma[c] + beta[c]

Rows are parallelised via pl.parallel (batch dimension).
The hidden dimension is loaded in full per tile (no column chunking),
keeping the kernel simple and single-pass friendly.

Input and output are FP32; gamma and beta are [1, hidden] weight vectors.
"""
from __future__ import annotations

import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 256            # hidden dimension (normalised axis, fits in one tile)
ROW_CHUNK = 32          # rows per parallel tile
EPS = 1e-5


def build_layer_norm_program(
    rows: int = ROWS,
    hidden: int = HIDDEN,
    row_chunk: int = ROW_CHUNK,
    eps: float = EPS,
):
    hidden_inv = 1.0 / hidden

    @pl.program
    class LayerNormProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def layer_norm(
            self,
            x: pl.Tensor[[rows, hidden], pl.FP32],
            gamma: pl.Tensor[[1, hidden], pl.FP32],
            beta: pl.Tensor[[1, hidden], pl.FP32],
            y: pl.Out[pl.Tensor[[rows, hidden], pl.FP32]],
        ) -> pl.Tensor[[rows, hidden], pl.FP32]:
            with pl.auto_incore():
                for r in pl.parallel(0, rows, row_chunk, chunk=1):
                    tile_x = pl.slice(x, [row_chunk, hidden], [r, 0])
                    gamma_tile = pl.slice(gamma, [1, hidden], [0, 0])
                    beta_tile = pl.slice(beta, [1, hidden], [0, 0])

                    # Step 1: row mean — pre-scale before row_sum, no reshape
                    mean = pl.row_sum(pl.mul(tile_x, hidden_inv))

                    # Step 2: row variance + eps — pre-scale and pre-add
                    centred = pl.row_expand_sub(tile_x, mean)
                    var_eps = pl.row_sum(
                        pl.mul(pl.add(pl.mul(centred, centred), eps), hidden_inv)
                    )

                    # Step 3: normalise — single reshape pair for sqrt
                    std = pl.reshape(
                        pl.sqrt(pl.reshape(var_eps, [1, row_chunk])),
                        [row_chunk, 1],
                    )
                    normed = pl.row_expand_div(centred, std)

                    # Step 4: apply gamma scale and beta offset
                    scaled = pl.col_expand_mul(normed, gamma_tile)
                    ones = pl.add(pl.sub(tile_x, tile_x), 1.0)
                    result = pl.add(scaled, pl.col_expand_mul(ones, beta_tile))
                    y = pl.assemble(y, result, [r, 0])

            return y

    return LayerNormProgram


def build_tensor_specs(
    rows: int = ROWS,
    hidden: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("x", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("beta", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch.float32, is_output=True),
    ]


def golden_layer_norm(tensors, params):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    beta = tensors["beta"]
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    tensors["y"][:] = (x - mean) / torch.sqrt(var + 1e-5) * gamma + beta


def compile_and_run(
    rows: int = ROWS,
    hidden: int = HIDDEN,
    row_chunk: int = ROW_CHUNK,
    platform: str = "a2a3",
    device_id: int = 11,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_layer_norm_program(
        rows=rows,
        hidden=hidden,
        row_chunk=row_chunk,
    )
    tensor_specs = build_tensor_specs(
        rows=rows,
        hidden=hidden,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_layer_norm,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-2,
            atol=1e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.Ascend910B_PTO,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).\n")
        print(result.error)
    elif not result.passed and result.error:
        print(f"Result: {result.error}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(
        platform="a2a3sim" if args.sim else "a2a3",
        device_id=args.device,
    )
    if not result.passed:
        raise SystemExit(1)
