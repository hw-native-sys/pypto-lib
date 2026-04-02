# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — fa0: RMSNorm only.

Single-incore test: compute RMSNorm of input hidden states and apply weights.
Output is the FP32 normed tensor (before projection).
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 5120

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
BATCH_TILE = 16


def build_rmsnorm_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
):
    hidden = hidden_size
    hidden_blocks = hidden // K_CHUNK

    @pl.program
    class RMSNormProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def rmsnorm(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            normed_out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            for b0 in pl.range(0, batch, BATCH_TILE):

                with pl.incore():
                    partial_sq_flat = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    partial_sq = pl.reshape(partial_sq_flat, [BATCH_TILE, 1])
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(partial_sq, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    variance = pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS)

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, variance), gamma)
                        normed_out = pl.assemble(normed_out, pl.cast(normed, target_type=pl.BF16), [b0, k0])

            return normed_out

    return RMSNormProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("normed_out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_rmsnorm(tensors, params):
    """PyTorch reference matching kernel precision path."""
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    normed_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        normed = (x_tile * variance * input_rms_weight.float()).bfloat16()
        normed_out[b0:b_end, :] = normed

    tensors["normed_out"][:] = normed_out


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

    program = build_rmsnorm_program(batch=batch, hidden_size=hidden_size)
    tensor_specs = build_tensor_specs(batch=batch, hidden_size=hidden_size)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_rmsnorm,
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
