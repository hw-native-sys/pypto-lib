# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross Core — Output projection + Residual addition fusion.

    output = matmul(attn_out, wo) + hidden_states

Stage 0 (matmul: attn_out × wo) and Stage 1 (residual add) are fused
using chunked_loop_optimizer for cross-core parallel execution.

This file isolates these two stages for debugging the chunk parameter's
effect on numerical accuracy. Use --no-fusion to test the unfused baseline.

Input and hidden_states are BF16; wo is BF16; output is FP32.
"""
from __future__ import annotations

import argparse

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, TensorSpec, run

BATCH = 16
HIDDEN = 8192
K_CHUNK = 128
Q_OUT_CHUNK = 64
BATCH_TILE = 16


def build_cross_core_fusion_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    k_chunk: int = K_CHUNK,
    q_out_chunk: int = Q_OUT_CHUNK,
    batch_tile: int = BATCH_TILE,
    chunk: int = 4,
):
    """Build fused Stage 0 & 1 program with chunked_loop_optimizer."""
    hidden_blocks = hidden // k_chunk
    q_out_blocks = hidden // q_out_chunk

    @pl.program
    class CrossCoreFusionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def cross_core(
            self,
            attn_out: pl.Tensor[[batch, hidden], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            resid: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN),
            ):
                for ob in pl.parallel(0, q_out_blocks, chunk=chunk):
                    o0 = ob * q_out_chunk
                    a_chunk_0 = pl.slice(attn_out, [batch_tile, k_chunk], [0, 0])
                    w_chunk_0 = pl.slice(wo, [k_chunk, q_out_chunk], [0, o0])
                    o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        a_chunk = pl.slice(attn_out, [batch_tile, k_chunk], [0, k0])
                        w_chunk = pl.slice(wo, [k_chunk, q_out_chunk], [k0, o0])
                        o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    hidden_chunk = pl.cast(
                        pl.slice(hidden_states, [batch_tile, q_out_chunk], [0, o0]),
                        target_type=pl.FP32,
                    )
                    resid_sum = pl.add(o_acc, hidden_chunk)
                    resid = pl.assemble(resid, resid_sum, [0, o0])

            return resid

    return CrossCoreFusionProgram


def build_cross_core_split_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    k_chunk: int = K_CHUNK,
    q_out_chunk: int = Q_OUT_CHUNK,
    batch_tile: int = BATCH_TILE,
):
    """Build unfused Stage 0 & 1 program with separate pl.at blocks."""
    hidden_blocks = hidden // k_chunk
    q_out_blocks = hidden // q_out_chunk

    @pl.program
    class CrossCoreSplitProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def cross_core(
            self,
            attn_out: pl.Tensor[[batch, hidden], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            resid: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            for ob in pl.range(q_out_blocks):
                o0 = ob * q_out_chunk

                with pl.at(level=pl.Level.CORE_GROUP):
                    a_chunk_0 = pl.slice(attn_out, [batch_tile, k_chunk], [0, 0])
                    w_chunk_0 = pl.slice(wo, [k_chunk, q_out_chunk], [0, o0])
                    o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        a_chunk = pl.slice(attn_out, [batch_tile, k_chunk], [0, k0])
                        w_chunk = pl.slice(wo, [k_chunk, q_out_chunk], [k0, o0])
                        o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                with pl.at(level=pl.Level.CORE_GROUP):
                    hidden_chunk = pl.cast(
                        pl.slice(hidden_states, [batch_tile, q_out_chunk], [0, o0]),
                        target_type=pl.FP32,
                    )
                    resid_sum = pl.add(o_acc, hidden_chunk)
                    resid = pl.assemble(resid, resid_sum, [0, o0])

            return resid

    return CrossCoreSplitProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden: int = HIDDEN,
):
    import torch

    def init_attn_out():
        return torch.rand(batch, hidden) - 0.5

    def init_hidden_states():
        return torch.rand(batch, hidden) - 0.5

    def init_wo():
        return (torch.rand(hidden, hidden) - 0.5) / hidden ** 0.5

    return [
        TensorSpec("attn_out", [batch, hidden], torch.bfloat16, init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("wo", [hidden, hidden], torch.bfloat16, init_value=init_wo),
        TensorSpec("resid", [batch, hidden], torch.float32, is_output=True),
    ]


def golden_cross_core(tensors, params):
    import torch

    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid = o_proj + hidden_states.float()
    tensors["resid"][:] = resid


def compile_and_run(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    k_chunk: int = K_CHUNK,
    q_out_chunk: int = Q_OUT_CHUNK,
    batch_tile: int = BATCH_TILE,
    chunk: int = 4,
    use_fusion: bool = True,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    if use_fusion:
        program = build_cross_core_fusion_program(
            batch=batch,
            hidden=hidden,
            k_chunk=k_chunk,
            q_out_chunk=q_out_chunk,
            batch_tile=batch_tile,
            chunk=chunk,
        )
    else:
        program = build_cross_core_split_program(
            batch=batch,
            hidden=hidden,
            k_chunk=k_chunk,
            q_out_chunk=q_out_chunk,
            batch_tile=batch_tile,
        )

    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden=hidden,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_cross_core,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=3e-3,
            atol=3e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Core — Stage 0 & 1 fusion test")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4, choices=[1, 2, 4, 8, 16],
                        help="Chunk size for parallel loop (smaller = more parallel tasks)")
    parser.add_argument("--no-fusion", action="store_true",
                        help="Use unfused version (separate pl.at blocks)")
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Cross Core — Stage 0 & 1 Fusion Test")
    print(f"{'='*60}")
    print(f"  Platform: {args.platform}")
    print(f"  Device: {args.device}")
    print(f"  Chunk: {args.chunk}")
    print(f"  Fusion: {'OFF (separate pl.at blocks)' if args.no_fusion else 'ON (chunked_loop_optimizer)'}")
    expected_tasks = HIDDEN // Q_OUT_CHUNK // args.chunk if not args.no_fusion else HIDDEN // Q_OUT_CHUNK
    print(f"  Expected tasks: {expected_tasks}")
    print(f"{'='*60}\n")

    result = compile_and_run(
        batch=BATCH,
        hidden=HIDDEN,
        k_chunk=K_CHUNK,
        q_out_chunk=Q_OUT_CHUNK,
        batch_tile=BATCH_TILE,
        chunk=args.chunk,
        use_fusion=not args.no_fusion,
        platform=args.platform,
        device_id=args.device,
        runtime_profiling=args.runtime_profiling,
    )

    if not result.passed:
        print(f"\n{'='*60}")
        print(f"FAILED: {result.error}")
        print(f"{'='*60}")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"PASSED")
    print(f"{'='*60}")
