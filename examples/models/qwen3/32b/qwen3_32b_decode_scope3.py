# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Qwen3 Scope 3: output projection + residual + post RMSNorm + MLP + residual.

Extracted from the full single-layer decode program to test this scope in
isolation. Covers:
  1. Output projection: attn_out × wo, accumulated in Q_OUT_CHUNK tiles
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition
"""

import pypto.language as pl

BATCH = 16
HIDDEN = 8192
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 256
BATCH_TILE = 16


def build_qwen3_scope3_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size

    HIDDEN_BLOCKS = HIDDEN_CFG // K_CHUNK
    Q_OUT_BLOCKS = HIDDEN_CFG // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = INTER_CFG // MLP_OUT_CHUNK

    @pl.program
    class Qwen3Scope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3(
            self,
            attn_out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)

                # Stage 0 & 1: Output projection: attn_out × wo, tiled by Q_OUT_CHUNK & Residual addition with hidden_states
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="out_proj_residual"):
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, chunk=6):
                        o0 = ob * Q_OUT_CHUNK
                        a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        hidden_chunk = pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0])

                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)

                        a_chunk_1 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, K_CHUNK])
                        w_chunk_1 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [K_CHUNK, o0])
                        o_acc = pl.matmul_acc(o_acc, a_chunk_1, w_chunk_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                        resid = pl.cast(
                            hidden_chunk,
                            target_type=pl.FP32,
                        )
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                # Stage 2: Post-attention RMSNorm
                post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rmsnorm"):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]))
                    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])), gamma)
                        normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                        post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                # Stage 3 & 4 & 5: MLP: gate/up projections + SiLU
                mlp_tile = pl.create_tensor([BATCH_TILE, INTER_CFG], dtype=pl.BF16)
                for ob in pl.parallel(0, MLP_OUT_BLOCKS, 1):
                    o0 = ob * MLP_OUT_CHUNK

                    post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                    post_chunk_1 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, K_CHUNK])
                    wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                    wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj"):
                        # Gate projection
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)

                        wg_1 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk_1, wg_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj"):
                        # Up projection
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)

                        wu_1 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [K_CHUNK, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk_1, wu_1)

                        for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="silu"):
                        # SiLU activation
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                    mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                # Stage 6 & 7: Down projection + final residual writeback (fused with chunked_loop_optimizer)
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)], name_hint="down_proj_residual"):
                    for dob in pl.parallel(0, HIDDEN_BLOCKS, chunk=1):
                        d0 = dob * K_CHUNK
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        resid1_tile_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0])

                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)

                        mlp_chunk_1 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, MLP_OUT_CHUNK])
                        w_down_chunk_1 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [MLP_OUT_CHUNK, d0])
                        down_acc = pl.matmul_acc(down_acc, mlp_chunk_1, w_down_chunk_1)

                        for ob in pl.pipeline(2, MLP_OUT_BLOCKS, stage=2):
                            o0 = ob * MLP_OUT_CHUNK
                            down_mlp_chunk_bf16 = pl.slice(
                                mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0]
                            )
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

                        out_chunk = pl.add(
                            down_acc,
                            resid1_tile_chunk
                        )
                        out_chunk_cast = pl.cast(out_chunk, target_type=pl.BF16)
                        out = pl.assemble(out, out_chunk_cast, [b0, d0])

            return out

    return Qwen3Scope3


def golden(tensors):
    """Reference computation for Scope 3.

    Steps:
      1. Output projection: attn_out (cast BF16) × wo, FP32 accumulation + residual
      2. Post-attention RMSNorm
      3. SwiGLU MLP: gate/up projections → silu(gate) * up → down projection
      4. Final residual addition → BF16 output
    """
    import torch

    attn_out = tensors["attn_out"]  # [B, H], FP32
    hidden_states = tensors["hidden_states"]  # [B, H], BF16
    wo = tensors["wo"]  # [H, H], BF16
    post_rms_weight = tensors["post_rms_weight"]  # [1, H], FP32
    w_gate = tensors["w_gate"]  # [H, I], BF16
    w_up = tensors["w_up"]  # [H, I], BF16
    w_down = tensors["w_down"]  # [I, H], BF16

    eps = 1e-6

    # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    # 2. Post-attention RMSNorm.
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    # 3. SwiGLU MLP: gate/up projections, silu activation, down projection.
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    # 4. Final residual + cast to BF16.
    tensors["out"][:] = (down + resid1).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch
    from golden import TensorSpec

    def init_attn_out():
        return torch.rand(batch, hidden_size) - 0.5

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_wo():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("attn_out", [batch, hidden_size], torch.bfloat16,
                   init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_scope3_program(),
        specs=build_tensor_specs(),
        golden_fn=golden,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
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
