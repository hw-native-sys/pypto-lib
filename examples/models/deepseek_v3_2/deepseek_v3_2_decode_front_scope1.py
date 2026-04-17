# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

"""
DeepSeek V3.2-EXP single-layer decode FRONT part — Scope 1 only (batch=16).

Scope 1: input RMSNorm + Q/KV projection.
- Compute RMSNorm of hidden_states
- Project to Q latent (qr) via wq_a
- Project from Q latent to Q heads (q_proj) via wq_b after q_norm
- Project to KV latent (kv_a) via wkv_a

Aligned to official v3.2-exp MLA shapes:
- qk_nope_head_dim = 128
- qk_rope_head_dim = 64
- kv_lora_rank = 512
"""


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

# Tile sizes tuned for standalone scope-1 incore boundaries:
# - PROJ_K = K-dimension chunk for projection matmuls (kept at 512).
# - LORA_CHUNK, KV_OUT_CHUNK = 64 so AIC Right buffer ≤ 65536
#   (512 * 64 * 2 = 65536).
# - Q_OUT_CHUNK = 64 for the same reason on the wq_b side
#   (LORA_CHUNK * Q_OUT_CHUNK * 2 = 64 * 64 * 2 = 8192).
# - BATCH_TILE = 16 (full batch).  The compiler dispatches projection matmuls
#   to AIC Cube, which requires M ≥ InnerRows (16); BATCH_TILE=4 causes a
#   static_assert in pto_tile.hpp.  The original 3-scope pipeline used
#   BATCH_TILE=4 because the combined scopes allowed a different split.
# - LOCAL_PAD_WIDTH removed; the pad tensor was a tuning hint for the
#   combined scope1+2+3 pipeline and is not needed for scope1 alone.
RMSNORM_K = 512
PROJ_K = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
LORA_CHUNK = 64
BATCH_TILE = 16


def build_deepseek_v3_2_decode_front_scope1_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    Q_LORA_RANK_CFG = q_lora_rank
    KV_LORA_RANK_CFG = kv_lora_rank
    QK_NOPE_HEAD_DIM_CFG = qk_nope_head_dim
    QK_ROPE_HEAD_DIM_CFG = qk_rope_head_dim
    QK_HEAD_DIM_CFG = qk_nope_head_dim + qk_rope_head_dim
    V_HEAD_DIM_CFG = v_head_dim
    KV_A_OUT_CFG = kv_lora_rank + qk_rope_head_dim

    RMSNORM_BLOCKS = (HIDDEN_CFG + RMSNORM_K - 1) // RMSNORM_K
    PROJ_BLOCKS = (HIDDEN_CFG + PROJ_K - 1) // PROJ_K
    QR_BLOCKS = (Q_LORA_RANK_CFG + LORA_CHUNK - 1) // LORA_CHUNK
    Q_OUT_BLOCKS = (NUM_HEADS_CFG * QK_HEAD_DIM_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_A_BLOCKS = (KV_A_OUT_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK

    @pl.program
    class DeepSeekV32DecodeFrontScope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope1(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN_CFG, Q_LORA_RANK_CFG], pl.BF16],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK_CFG], pl.FP32],
            wq_b: pl.Tensor[[Q_LORA_RANK_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN_CFG, KV_A_OUT_CFG], pl.BF16],
            # Output buffers
            qr_out: pl.Tensor[[BATCH_CFG, Q_LORA_RANK_CFG], pl.BF16],
            q_proj_out: pl.Tensor[[BATCH_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16],
            kv_a_out: pl.Tensor[[BATCH_CFG, KV_A_OUT_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16]:
            # Scope 1: input RMSNorm + Q/KV projection.
            qr = pl.create_tensor([BATCH_CFG, Q_LORA_RANK_CFG], dtype=pl.BF16)
            q_proj = pl.create_tensor([BATCH_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], dtype=pl.BF16)
            kv_a = pl.create_tensor([BATCH_CFG, KV_A_OUT_CFG], dtype=pl.BF16)

            for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                qr_fp32_tile = pl.create_tensor([BATCH_TILE, Q_LORA_RANK_CFG], dtype=pl.FP32)
                qr_tile = pl.create_tensor([BATCH_TILE, Q_LORA_RANK_CFG], dtype=pl.BF16)
                kv_a_fp32_tile = pl.create_tensor([BATCH_TILE, KV_A_OUT_CFG], dtype=pl.FP32)

                # Stage 1: RMSNorm + apply weights, matching the Qwen3
                # batch-tile outer-loop structure.
                with pl.at(level=pl.Level.CORE_GROUP):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(RMSNORM_BLOCKS):
                        k0 = kb * RMSNORM_K
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, RMSNORM_K], [b0, k0]),
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
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.range(PROJ_BLOCKS):
                        k0 = kb * PROJ_K
                        x_chunk_bf16 = pl.slice(hidden_states, [BATCH_TILE, PROJ_K], [b0, k0])
                        x_tile = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                        gamma = pl.slice(input_rms_weight, [1, PROJ_K], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                        normed_tile = pl.assemble(
                            normed_tile,
                            pl.cast(normed, target_type=pl.BF16),
                            [0, k0],
                        )

                # Stage 2: Q latent projection, accumulated in Cube Acc.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, QR_BLOCKS, 1, chunk=4):
                        q0 = ob * LORA_CHUNK
                        q_tile_a = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, 0])
                        q_tile_b = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [0, q0])
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                        for kb in pl.range(1, PROJ_BLOCKS):
                            k0 = kb * PROJ_K
                            q_tile_a_i = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, k0])
                            q_tile_b_i = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [k0, q0])
                            q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_b_i)
                        qr_fp32_tile = pl.assemble(qr_fp32_tile, q_acc, [0, q0])

                # Stage 3: cast Q latent output to BF16.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk_bf16 = pl.cast(
                            pl.slice(qr_fp32_tile, [BATCH_TILE, LORA_CHUNK], [0, k0]),
                            target_type=pl.BF16,
                        )
                        qr_tile = pl.assemble(qr_tile, qr_chunk_bf16, [0, k0])
                        qr = pl.assemble(qr, qr_chunk_bf16, [b0, k0])

                # Stage 4: Q head projection. Keep the original per-K-block
                # accumulation form; materializing the full FP32 q_proj
                # temporary exposes a block-write issue for this 24576-wide output.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        q0 = ob * Q_OUT_CHUNK
                        q_out_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(QR_BLOCKS):
                            k0 = kb * LORA_CHUNK
                            q_chunk = pl.cast(
                                pl.slice(qr_tile, [BATCH_TILE, LORA_CHUNK], [0, k0]),
                                target_type=pl.FP32,
                            )
                            q_gamma = pl.slice(q_norm_weight, [1, LORA_CHUNK], [0, k0])
                            qn = pl.col_expand_mul(q_chunk, q_gamma)
                            wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_out_acc = pl.add(
                                q_out_acc,
                                pl.matmul(pl.cast(qn, target_type=pl.BF16), wq_b_chunk),
                            )
                        q_proj = pl.assemble(q_proj, pl.cast(q_out_acc, target_type=pl.BF16), [b0, q0])

                # Stage 5: KV latent projection, accumulated in Cube Acc.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, KV_A_BLOCKS, 1, chunk=8):
                        kv0 = ob * KV_OUT_CHUNK
                        kv_tile_a = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, 0])
                        kv_tile_b = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [0, kv0])
                        kv_acc = pl.matmul(kv_tile_a, kv_tile_b, out_dtype=pl.FP32)
                        for kb in pl.range(1, PROJ_BLOCKS):
                            k0 = kb * PROJ_K
                            kv_tile_a_i = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, k0])
                            kv_tile_b_i = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [k0, kv0])
                            kv_acc = pl.matmul_acc(kv_acc, kv_tile_a_i, kv_tile_b_i)
                        kv_a_fp32_tile = pl.assemble(kv_a_fp32_tile, kv_acc, [0, kv0])

                # Stage 6: final KV output cast from FP32 projection temporary.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, KV_A_BLOCKS, 1, chunk=8):
                        kv0 = ob * KV_OUT_CHUNK
                        kv_chunk = pl.cast(
                            pl.slice(kv_a_fp32_tile, [BATCH_TILE, KV_OUT_CHUNK], [0, kv0]),
                            target_type=pl.BF16,
                        )
                        kv_a = pl.assemble(kv_a, kv_chunk, [b0, kv0])

            qr_out = pl.assemble(qr_out, qr, [0, 0])
            q_proj_out = pl.assemble(q_proj_out, q_proj, [0, 0])
            kv_a_out = pl.assemble(kv_a_out, kv_a, [0, 0])
            return q_proj_out

    return DeepSeekV32DecodeFrontScope1


def golden_decode_front_scope1(tensors):
    import torch  # type: ignore[import]

    hidden_states = tensors["hidden_states"].float()
    input_rms_weight = tensors["input_rms_weight"].float()
    wq_a = tensors["wq_a"].float()
    q_norm_weight = tensors["q_norm_weight"].float()
    wq_b = tensors["wq_b"].float()
    wkv_a = tensors["wkv_a"].float()

    # RMSNorm
    sq_sum = torch.sum(hidden_states * hidden_states, dim=1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * HIDDEN_INV + EPS)
    normed = (hidden_states * inv_rms * input_rms_weight).to(torch.bfloat16).float()

    # Q latent projection
    qr = (normed @ wq_a).to(torch.bfloat16)

    # Q head projection
    qn = (qr.float() * q_norm_weight).to(torch.bfloat16).float()
    q_proj = (qn @ wq_b).to(torch.bfloat16)

    # KV latent projection
    kv_a = (normed @ wkv_a).to(torch.bfloat16)

    # Write into output tensor slots
    tensors["qr_out"].copy_(qr)
    tensors["q_proj_out"].copy_(q_proj)
    tensors["kv_a_out"].copy_(kv_a)


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out = kv_lora_rank + qk_rope_head_dim

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_q_norm_weight():
        return torch.rand(1, q_lora_rank) - 0.5

    def init_wq_a():
        return (torch.rand(hidden_size, q_lora_rank) - 0.5) / hidden_size ** 0.5

    def init_wq_b():
        return (torch.rand(q_lora_rank, num_heads * qk_head_dim) - 0.5) / q_lora_rank ** 0.5

    def init_wkv_a():
        return (torch.rand(hidden_size, kv_a_out) - 0.5) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq_a", [hidden_size, q_lora_rank], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("q_norm_weight", [1, q_lora_rank], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("wq_b", [q_lora_rank, num_heads * qk_head_dim], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv_a", [hidden_size, kv_a_out], torch.bfloat16, init_value=init_wkv_a),
        TensorSpec("qr_out", [batch, q_lora_rank], torch.bfloat16, is_output=True),
        TensorSpec("q_proj_out", [batch, num_heads * qk_head_dim], torch.bfloat16, is_output=True),
        TensorSpec("kv_a_out", [batch, kv_a_out], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_decode_front_scope1_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_decode_front_scope1,
        config=RunConfig(
            rtol=2e-2,
            atol=2e-2,
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
