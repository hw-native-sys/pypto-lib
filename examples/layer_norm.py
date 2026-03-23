# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LayerNorm — full layer normalization with row + column tiling.

    output[r, c] = (x[r, c] - mean(x[r, :])) / sqrt(var(x[r, :]) + eps) * gamma[c] + beta[c]

Rows are parallelised via pl.parallel (batch dimension).
The hidden dimension is chunked with pl.range to accumulate the
sum and squared-sum reductions, then a second pass centres, normalises,
and applies gamma/beta.

This two-pass column-chunking pattern follows the same approach used
by rms_norm.py and the production LLM kernels (qwen3/deepseek).
Variance is computed via E[x^2] - E[x]^2 to avoid materialising the
centred tensor during the accumulation pass.

Input and output are FP32; gamma and beta are [1, hidden] weight vectors.
"""
from __future__ import annotations

import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 512            # hidden dimension (normalised axis)
ROW_CHUNK = 32          # rows per parallel tile
HIDDEN_CHUNK = 64       # columns per sequential chunk
EPS = 1e-5


def build_layer_norm_program(
    rows: int = ROWS,
    hidden: int = HIDDEN,
    row_chunk: int = ROW_CHUNK,
    hidden_chunk: int = HIDDEN_CHUNK,
    eps: float = EPS,
):
    hidden_blocks = hidden // hidden_chunk
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
                    # Pass 1: accumulate sum(x) and sum(x^2) across hidden chunks
                    # row_sum produces [row_chunk, 1] col_major; accumulate
                    # in [1, row_chunk] for scalar ops (same as rms_norm).
                    x_sum = pl.create_tensor([1, row_chunk], dtype=pl.FP32)
                    x_sum = pl.mul(x_sum, 0.0)
                    sq_sum = pl.create_tensor([1, row_chunk], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for hb in pl.range(hidden_blocks):
                        h0 = hb * hidden_chunk
                        x_chunk = pl.slice(x, [row_chunk, hidden_chunk], [r, h0])
                        x_sum = pl.add(
                            x_sum, pl.reshape(pl.row_sum(x_chunk), [1, row_chunk])
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, row_chunk]
                            ),
                        )

                    # mean and inv_std via E[x^2] - E[x]^2
                    mean_T = pl.mul(x_sum, hidden_inv)
                    var_T = pl.sub(
                        pl.mul(sq_sum, hidden_inv), pl.mul(mean_T, mean_T)
                    )
                    inv_std_T = pl.rsqrt(pl.add(var_T, eps))
                    mean = pl.reshape(mean_T, [row_chunk, 1])
                    inv_std = pl.reshape(inv_std_T, [row_chunk, 1])

                    # Pass 2: centre, normalise, apply gamma/beta
                    for hb in pl.range(hidden_blocks):
                        h0 = hb * hidden_chunk
                        x_chunk = pl.slice(x, [row_chunk, hidden_chunk], [r, h0])
                        gamma_chunk = pl.slice(gamma, [1, hidden_chunk], [0, h0])
                        beta_chunk = pl.slice(beta, [1, hidden_chunk], [0, h0])
                        centred = pl.row_expand_sub(x_chunk, mean)
                        normed = pl.row_expand_mul(centred, inv_std)
                        scaled = pl.col_expand_mul(normed, gamma_chunk)
                        ones = pl.add(pl.sub(x_chunk, x_chunk), 1.0)
                        result = pl.add(
                            scaled, pl.col_expand_mul(ones, beta_chunk)
                        )
                        y = pl.assemble(y, result, [r, h0])

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
    hidden_chunk: int = HIDDEN_CHUNK,
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
        hidden_chunk=hidden_chunk,
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
