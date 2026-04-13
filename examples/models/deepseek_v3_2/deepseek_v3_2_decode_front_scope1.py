# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V3.2-EXP decode Scope 1 — input RMSNorm + Q/K/V projection.

Based on qwen3_32b_decode_scope1.py structure (simplified without Q norm step).
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
LORA_CHUNK = 64
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 64
BATCH_TILE = 16


def build_deepseek_v3_2_decode_scope1_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_head_dim: int = QK_HEAD_DIM,
):
    hidden = hidden_size
    kv_a_out_dim = kv_lora_rank + QK_ROPE_HEAD_DIM
    q_out_dim = num_heads * qk_head_dim
    
    hidden_blocks = hidden // K_CHUNK
    lora_blocks = q_lora_rank // LORA_CHUNK
    q_out_blocks = q_out_dim // Q_OUT_CHUNK
    kv_out_blocks = kv_a_out_dim // KV_OUT_CHUNK

    @pl.program
    class DeepSeekV32DecodeScope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_scope1(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq_a: pl.Tensor[[hidden, q_lora_rank], pl.BF16],
            wq_b: pl.Tensor[[q_lora_rank, q_out_dim], pl.BF16],
            wkv_a: pl.Tensor[[hidden, kv_a_out_dim], pl.BF16],
            qr_proj: pl.Out[pl.Tensor[[batch, q_lora_rank], pl.FP32]],
            q_proj: pl.Out[pl.Tensor[[batch, q_out_dim], pl.FP32]],
            kv_proj: pl.Out[pl.Tensor[[batch, kv_a_out_dim], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[batch, q_lora_rank], pl.FP32],
            pl.Tensor[[batch, q_out_dim], pl.FP32],
            pl.Tensor[[batch, kv_a_out_dim], pl.FP32],
        ]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)

                # Stage 1: RMSNorm + apply weights (vector ops only).
                with pl.incore():
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )
                    variance = pl.reshape(
                        pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS),
                        [BATCH_TILE, 1],
                    )

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, variance), gamma)
                        normed_tile = pl.assemble(normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                # Stage 2: Q compression (matmul + matmul_acc in single incore).
                for ob in pl.range(lora_blocks):
                    q0 = ob * LORA_CHUNK

                    with pl.incore():
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        tile_b = pl.slice(wq_a, [K_CHUNK, LORA_CHUNK], [0, q0])
                        qr_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(wq_a, [K_CHUNK, LORA_CHUNK], [k0, q0])
                            qr_acc = pl.matmul_acc(qr_acc, tile_a_i, tile_b_i)

                        qr_proj = pl.assemble(qr_proj, qr_acc, [b0, q0])

                # Stage 3: KV compression (matmul + matmul_acc in single incore).
                for ob in pl.range(kv_out_blocks):
                    kv0 = ob * KV_OUT_CHUNK

                    with pl.incore():
                        tile_a = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        tile_b = pl.slice(wkv_a, [K_CHUNK, KV_OUT_CHUNK], [0, kv0])
                        kv_acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)

                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            tile_b_i = pl.slice(wkv_a, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            kv_acc = pl.matmul_acc(kv_acc, tile_a_i, tile_b_i)

                        kv_proj = pl.assemble(kv_proj, kv_acc, [b0, kv0])

            return qr_proj, q_proj, kv_proj

    return DeepSeekV32DecodeScope1


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_head_dim: int = QK_HEAD_DIM,
):
    import torch
    from pypto.runtime import TensorSpec

    q_out_dim = num_heads * qk_head_dim
    kv_a_out_dim = kv_lora_rank + QK_ROPE_HEAD_DIM

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq_a():
        return torch.rand(hidden_size, q_lora_rank) - 0.5

    def init_wq_b():
        return torch.rand(q_lora_rank, q_out_dim) - 0.5

    def init_wkv_a():
        return torch.rand(hidden_size, kv_a_out_dim) - 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq_a", [hidden_size, q_lora_rank], torch.bfloat16,
                   init_value=init_wq_a),
        TensorSpec("wq_b", [q_lora_rank, q_out_dim], torch.bfloat16,
                   init_value=init_wq_b),
        TensorSpec("wkv_a", [hidden_size, kv_a_out_dim], torch.bfloat16,
                   init_value=init_wkv_a),
        TensorSpec("qr_proj", [batch, q_lora_rank], torch.float32, is_output=True),
        TensorSpec("q_proj", [batch, q_out_dim], torch.float32, is_output=True),
        TensorSpec("kv_proj", [batch, kv_a_out_dim], torch.float32, is_output=True),
    ]


def golden_mla_projection(tensors, params):
    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq_a = tensors["wq_a"]
    wq_b = tensors["wq_b"]
    wkv_a = tensors["wkv_a"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    qr_proj = torch.zeros(batch, Q_LORA_RANK, dtype=torch.float32)
    q_proj = torch.zeros(batch, NUM_HEADS * QK_HEAD_DIM, dtype=torch.float32)
    kv_proj = torch.zeros(batch, KV_A_OUT, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, K_CHUNK):
            x_chunk = x_tile[:, k0:k0 + K_CHUNK]
            sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        normed = (x_tile * variance * input_rms_weight.float()).bfloat16()

        # Q compression
        qr_proj[b0:b_end, :] = (normed.float() @ wq_a.float()).float()

        # Q expansion (computed separately for verification)
        q_proj[b0:b_end, :] = (qr_proj[b0:b_end].float() @ wq_b.float()).float()

        # KV compression
        kv_proj[b0:b_end, :] = (normed.float() @ wkv_a.float()).float()

    tensors["qr_proj"][:] = qr_proj
    tensors["q_proj"][:] = q_proj
    tensors["kv_proj"][:] = kv_proj


def compile_and_run(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_head_dim: int = QK_HEAD_DIM,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_deepseek_v3_2_decode_scope1_program(
        batch=batch,
        hidden_size=hidden_size,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        hidden_size=hidden_size,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_mla_projection,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=enable_profiling,
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