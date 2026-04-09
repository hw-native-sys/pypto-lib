# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 1 — tile DSL version of input RMSNorm + Q/K/V projection.

This keeps the same I/O contract and chunking strategy as
`qwen3_32b_decode_scope1.py`, but rewrites the kernel body into explicit tile
DSL:

  - vector reduction kernels for the hidden-state squared-sum pass
  - vector kernel for applying the per-row normalization factor and RMS weight
  - cube kernels for Q/K/V projection blocks using explicit load/move/matmul

The arithmetic intentionally matches the existing scope1 example, including its
current RMSNorm path based on `inv_rms = rsqrt(mean(x^2) + eps)`, so the two
programs are easy to compare structurally.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16


def build_decode_projection_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK

    @pl.program
    class DecodeProjectionTileProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def build_normed_tile(
            self,
            hidden_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            output: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
        ) -> pl.Tensor[[BATCH_TILE, hidden], pl.BF16]:
            partial_sq = pl.create_tile([1, BATCH_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            partial_sq = pl.mul(partial_sq, 0.0)

            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                tile_x = pl.load(
                    hidden_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                tile_x_f32 = pl.cast(tile_x, target_type=pl.FP32)
                squared = pl.mul(tile_x_f32, tile_x_f32)
                tmp = pl.create_tile([BATCH_TILE, K_CHUNK], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                row_sum: pl.Tile[[BATCH_TILE, 1], pl.FP32] = pl.row_sum(squared, tmp)
                partial_sq = pl.add(partial_sq, pl.reshape(row_sum, [1, BATCH_TILE]))

            variance_t: pl.Tile[[1, BATCH_TILE], pl.FP32] = pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS)
            variance: pl.Tile[[BATCH_TILE, 1], pl.FP32] = pl.reshape(variance_t, [BATCH_TILE, 1])
            rms = pl.sqrt(variance)
            inv_rms = pl.recip(rms)

            for kb, (out_iter,) in pl.range(hidden_blocks, init_values=(output,)):
                k0 = kb * K_CHUNK
                tile_x = pl.load(
                    hidden_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                tile_x_f32 = pl.cast(tile_x, target_type=pl.FP32)
                tile_gamma = pl.load(
                    input_rms_weight,
                    [0, k0],
                    [1, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                scaled = pl.row_expand_mul(tile_x_f32, inv_rms)
                weighted = pl.col_expand_mul(scaled, tile_gamma)
                weighted_bf16 = pl.cast(weighted, target_type=pl.BF16)
                out_next = pl.store(weighted_bf16, [0, k0], out_iter)
                (out_carry,) = pl.yield_(out_next)

            return out_carry

        @pl.function(type=pl.FunctionType.InCore)
        def q_proj_reduce(
            self,
            normed_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            weight: pl.Tensor[[hidden, hidden], pl.BF16],
            out_row: pl.Scalar[pl.INDEX],
            out_col: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            tile_a_l1 = pl.load(normed_tile, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(weight, [0, out_col], [K_CHUNK, Q_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(tile_a_l0a, tile_b_l0b)

            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                tile_a_i_l1 = pl.load(normed_tile, [0, k0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                tile_b_i_l1 = pl.load(weight, [k0, out_col], [K_CHUNK, Q_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                tile_a_i_l0a = pl.move(tile_a_i_l1, target_memory=pl.MemorySpace.Left)
                tile_b_i_l0b = pl.move(tile_b_i_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul_acc(acc, tile_a_i_l0a, tile_b_i_l0b)

            out = pl.store(acc, [out_row, out_col], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def kv_proj_reduce(
            self,
            normed_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            weight: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            out_row: pl.Scalar[pl.INDEX],
            out_col: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, kv_hidden], pl.FP32]:
            tile_a_l1 = pl.load(normed_tile, [0, 0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(weight, [0, out_col], [K_CHUNK, KV_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(tile_a_l0a, tile_b_l0b)

            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                tile_a_i_l1 = pl.load(normed_tile, [0, k0], [BATCH_TILE, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                tile_b_i_l1 = pl.load(weight, [k0, out_col], [K_CHUNK, KV_OUT_CHUNK], target_memory=pl.MemorySpace.Mat)
                tile_a_i_l0a = pl.move(tile_a_i_l1, target_memory=pl.MemorySpace.Left)
                tile_b_i_l0b = pl.move(tile_b_i_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul_acc(acc, tile_a_i_l0a, tile_b_i_l0b)

            out = pl.store(acc, [out_row, out_col], output)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def decode_projection(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            q_proj: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
            k_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
            v_proj: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
            ) -> tuple[
            pl.Tensor[[batch, hidden], pl.FP32],
            pl.Tensor[[batch, kv_hidden], pl.FP32],
            pl.Tensor[[batch, kv_hidden], pl.FP32],
        ]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                hidden_tile = pl.slice(hidden_states, [BATCH_TILE, hidden], [b0, 0])
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                normed_tile = self.build_normed_tile(hidden_tile, input_rms_weight, normed_tile)

                for ob in pl.range(q_out_blocks):
                    q0 = ob * Q_OUT_CHUNK
                    q_proj = self.q_proj_reduce(normed_tile, wq, b0, q0, q_proj)

                for ob in pl.range(kv_out_blocks):
                    kv0 = ob * KV_OUT_CHUNK
                    k_proj = self.kv_proj_reduce(normed_tile, wk, b0, kv0, k_proj)
                    v_proj = self.kv_proj_reduce(normed_tile, wv, b0, kv0, v_proj)

            return q_proj, k_proj, v_proj

    return DecodeProjectionTileProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    kv_hidden = num_kv_heads * head_dim

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.randn(hidden_size, hidden_size) / hidden_size ** 0.5

    def init_wk():
        return torch.randn(hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return torch.randn(hidden_size, kv_hidden) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_proj", [batch, hidden_size], torch.float32, is_output=True),
        TensorSpec("k_proj", [batch, kv_hidden], torch.float32, is_output=True),
        TensorSpec("v_proj", [batch, kv_hidden], torch.float32, is_output=True),
    ]


def golden_decode_projection(tensors, params):
    """PyTorch reference matching the existing scope1 precision path."""
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = (normed.float() @ wq.float()).float()
        k_proj[b0:b_end, :] = (normed.float() @ wk.float()).float()
        v_proj[b0:b_end, :] = (normed.float() @ wv.float()).float()

    tensors["q_proj"][:] = q_proj
    tensors["k_proj"][:] = k_proj
    tensors["v_proj"][:] = v_proj


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

    program = build_decode_projection_program(
        batch=batch,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_decode_projection,
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
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
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
