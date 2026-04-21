# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GEMM + Elementwise — matrix multiplication fused with elementwise addition.

    output = matmul(attn_out, wo) + hidden_states

Stage 0 (matmul: attn_out x wo) and Stage 1 (residual add) can be:
  - Fused: single pl.at block with chunked_loop_optimizer (mix mode)
  - Split: separate pl.at blocks for each stage (split mode)

Input and hidden_states are BF16; wo is BF16; output is FP32.
"""
from __future__ import annotations

import pypto.language as pl

# ---------------------------------------------------------------------------
# GEMM + Elementwise parameters — edit these to change problem size and tiling
# ---------------------------------------------------------------------------
BATCH = 16
HIDDEN = 8192
K_CHUNK = 128        # K dimension tile size for matmul
N_CHUNK = 64         # N dimension tile size for matmul output
BATCH_TILE = 16      # Batch dimension tile size


def build_gemm_eltwise_mix_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    k_chunk: int = K_CHUNK,
    n_chunk: int = N_CHUNK,
    batch_tile: int = BATCH_TILE,
    chunk: int = 4,
):
    """Build fused matmul + elementwise program with chunked_loop_optimizer."""
    k_blocks = hidden // k_chunk
    n_blocks = hidden // n_chunk

    @pl.program
    class GemmEltwiseMixProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def gemm_eltwise(
            self,
            attn_out: pl.Tensor[[batch, hidden], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            resid: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
                for nb in pl.parallel(0, n_blocks, chunk=chunk):
                    n0 = nb * n_chunk
                    # First K-tile: initialize accumulator via matmul
                    a_chunk_0 = pl.slice(attn_out, [batch_tile, k_chunk], [0, 0])
                    w_chunk_0 = pl.slice(wo, [k_chunk, n_chunk], [0, n0])
                    acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)

                    # Remaining K-tiles: accumulate via matmul_acc
                    for kb in pl.range(1, k_blocks):
                        k0 = kb * k_chunk
                        a_chunk = pl.slice(attn_out, [batch_tile, k_chunk], [0, k0])
                        w_chunk = pl.slice(wo, [k_chunk, n_chunk], [k0, n0])
                        acc = pl.matmul_acc(acc, a_chunk, w_chunk)

                    # Elementwise residual addition
                    hidden_chunk = pl.slice(hidden_states, [batch_tile, n_chunk], [0, n0])
                    hidden_chunk_f32 = pl.cast(hidden_chunk, target_type=pl.FP32)
                    resid_sum = pl.add(acc, hidden_chunk_f32)
                    resid = pl.assemble(resid, resid_sum, [0, n0])

            return resid

    return GemmEltwiseMixProgram


def build_gemm_eltwise_split_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    k_chunk: int = K_CHUNK,
    n_chunk: int = N_CHUNK,
    batch_tile: int = BATCH_TILE,
):
    """Build unfused matmul + elementwise program with separate pl.at blocks."""
    k_blocks = hidden // k_chunk
    n_blocks = hidden // n_chunk

    @pl.program
    class GemmEltwiseSplitProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def gemm_eltwise(
            self,
            attn_out: pl.Tensor[[batch, hidden], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            resid: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            for nb in pl.range(n_blocks):
                n0 = nb * n_chunk

                # Stage 0: matmul
                with pl.at(level=pl.Level.CORE_GROUP):
                    a_chunk_0 = pl.slice(attn_out, [batch_tile, k_chunk], [0, 0])
                    w_chunk_0 = pl.slice(wo, [k_chunk, n_chunk], [0, n0])
                    acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, k_blocks):
                        k0 = kb * k_chunk
                        a_chunk = pl.slice(attn_out, [batch_tile, k_chunk], [0, k0])
                        w_chunk = pl.slice(wo, [k_chunk, n_chunk], [k0, n0])
                        acc = pl.matmul_acc(acc, a_chunk, w_chunk)

                # Stage 1: elementwise residual addition
                with pl.at(level=pl.Level.CORE_GROUP):
                    hidden_chunk = pl.slice(hidden_states, [batch_tile, n_chunk], [0, n0])
                    hidden_chunk_f32 = pl.cast(hidden_chunk, target_type=pl.FP32)
                    resid_sum = pl.add(acc, hidden_chunk_f32)
                resid = pl.assemble(resid, resid_sum, [0, n0])

            return resid

    return GemmEltwiseSplitProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden: int = HIDDEN,
):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("attn_out", [batch, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("hidden_states", [batch, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [hidden, hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("resid", [batch, hidden], torch.float32, is_output=True),
    ]


def golden_gemm_eltwise(tensors):
    import torch

    o_proj = torch.matmul(tensors["attn_out"].float(), tensors["wo"].float())
    tensors["resid"][:] = o_proj + tensors["hidden_states"].float()


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4,
                        help="Chunk size for parallel loop (smaller = more parallel tasks)")
    parser.add_argument("--mix", action="store_true",
                        help="Use fused mix version (default: split version)")
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    if args.mix:
        program = build_gemm_eltwise_mix_program(chunk=args.chunk)
    else:
        program = build_gemm_eltwise_split_program()

    result = run(
        program=program,
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_gemm_eltwise,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
