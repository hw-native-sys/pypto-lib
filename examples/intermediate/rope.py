# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RoPE — Rotary Position Embedding with half-vector rotation.

    y_lo = x_lo * cos_lo - x_hi * sin_lo
    y_hi = x_hi * cos_hi + x_lo * sin_hi
    y    = concat(y_lo, y_hi)

The input vector is split into two halves along the head dimension.
Each half is multiplied by the corresponding cos/sin values and combined
via subtraction/addition to produce the rotated output.

x is laid out as [BATCH * NUM_HEADS, HEAD_DIM]; each group of NUM_HEADS
rows belongs to one batch item.  cos and sin are [1, HEAD_DIM] (a single
position's embedding broadcast across all heads via col_expand_mul) —
matching the decode pattern in Qwen3.

The outer loop parallelises over the batch dimension (BATCH=16).
Each iteration processes NUM_HEADS=8 rows of HEAD_DIM=128, giving
half-vector operands of [8, 64] FP32 = 2 KB = TILELET MAX.

Structure matches Qwen3 decode tilelet (Scope 2):
  auto_incore → parallel(BATCH) → cos/sin split → rotate → store.

Input and output are FP32; cos and sin are FP32.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16          # batch size (= Qwen3 BATCH)
NUM_HEADS = 8       # heads per batch item (= Qwen3 NUM_KV_HEADS)
HEAD_DIM = 128      # dimension per head (= Qwen3 HEAD_DIM)
BATCH_CHUNK = 4     # parallel chunk size for batch loop


def build_rope_program(
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    batch_chunk: int = BATCH_CHUNK,
):
    total_rows = batch * num_heads
    half_dim = head_dim // 2

    @pl.program
    class RoPEProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def rope(
            self,
            x: pl.Tensor[[total_rows, head_dim], pl.FP32],
            cos: pl.Tensor[[1, head_dim], pl.FP32],
            sin: pl.Tensor[[1, head_dim], pl.FP32],
            y: pl.Out[pl.Tensor[[total_rows, head_dim], pl.FP32]],
        ) -> pl.Tensor[[total_rows, head_dim], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, batch, 1, chunk=batch_chunk):
                    # Slice cos/sin lo/hi halves directly from tensor
                    # so each becomes a separate tile.load (no textract).
                    cos_lo = pl.slice(cos, [1, half_dim], [0, 0])
                    cos_hi = pl.slice(cos, [1, half_dim], [0, half_dim])
                    sin_lo = pl.slice(sin, [1, half_dim], [0, 0])
                    sin_hi = pl.slice(sin, [1, half_dim], [0, half_dim])

                    base = b * num_heads
                    x_lo = pl.slice(x, [num_heads, half_dim], [base, 0])
                    x_hi = pl.slice(x, [num_heads, half_dim], [base, half_dim])

                    rot_lo = pl.sub(
                        pl.col_expand_mul(x_lo, cos_lo),
                        pl.col_expand_mul(x_hi, sin_lo),
                    )
                    rot_hi = pl.add(
                        pl.col_expand_mul(x_hi, cos_hi),
                        pl.col_expand_mul(x_lo, sin_hi),
                    )
                    y = pl.assemble(y, rot_lo, [base, 0])
                    y = pl.assemble(y, rot_hi, [base, half_dim])

            return y

    return RoPEProgram


def build_tensor_specs(
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    total_rows = batch * num_heads

    return [
        TensorSpec("x", [total_rows, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("cos", [1, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("sin", [1, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("y", [total_rows, head_dim], torch.float32, is_output=True),
    ]


def golden_rope(tensors, params):
    import torch

    x = tensors["x"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    half = x.shape[-1] // 2

    tensors["y"][:] = torch.cat(
        [
            x[:, :half] * cos[:, :half] - x[:, half:] * sin[:, :half],
            x[:, half:] * cos[:, half:] + x[:, :half] * sin[:, half:],
        ],
        dim=-1,
    )


def compile_and_run(
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    batch_chunk: int = BATCH_CHUNK,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_rope_program(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_chunk=batch_chunk,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_rope,
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
