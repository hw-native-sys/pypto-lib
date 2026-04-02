# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — s2: K/V projection.

GEMM-style: normed [batch, hidden] BF16 @ wk/wv [hidden, kv_hidden] BF16
  -> k_proj [batch, kv_hidden] FP32, v_proj [batch, kv_hidden] FP32.
Uses matmul + matmul_acc for K-dimension reduction in a single incore block.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

K_CHUNK = 128
KV_OUT_CHUNK = 64
BATCH_TILE = 16


def build_kv_projection_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    hidden_blocks = hidden // K_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK

    @pl.program
    class KVProjectionProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def kv_projection(
            self,
            normed: pl.Tensor[[batch, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            k_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
            v_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[batch, kv_hidden], pl.FP32],
            pl.Tensor[[batch, kv_hidden], pl.FP32],
        ]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                for ob in pl.range(kv_out_blocks):
                    kv0 = ob * KV_OUT_CHUNK

                    with pl.incore():
                        # K projection: matmul + matmul_acc.
                        tile_a = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, 0])
                        tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)

                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, k0])
                            tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)

                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

                    with pl.incore():
                        # V projection: matmul + matmul_acc.
                        tile_a = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, 0])
                        tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)

                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed, [BATCH_TILE, K_CHUNK], [b0, k0])
                            tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)

                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])

            return k_proj, v_proj

    return KVProjectionProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim

    return [
        TensorSpec("normed", [batch, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=torch.randn),
        TensorSpec("k_proj", [batch, kv_hidden], torch.float32, is_output=True),
        TensorSpec("v_proj", [batch, kv_hidden], torch.float32, is_output=True),
    ]


def golden_kv_projection(tensors, params):
    """PyTorch reference: BF16 matmul with FP32 output."""
    normed = tensors["normed"].float()
    tensors["k_proj"][:] = (normed @ tensors["wk"].float()).float()
    tensors["v_proj"][:] = (normed @ tensors["wv"].float()).float()


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_kv_projection_program(
        batch=batch, hidden_size=hidden_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch, hidden_size=hidden_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_kv_projection,
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
