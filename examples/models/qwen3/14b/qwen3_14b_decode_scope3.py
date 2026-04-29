# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode Scope 3 — output projection + residual + post RMSNorm + MLP + residual."""

import pypto.language as pl

BATCH = 16
HIDDEN = 5120
INTERMEDIATE = 17408

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
    hidden_blocks = hidden_size // K_CHUNK
    q_out_blocks = hidden_size // Q_OUT_CHUNK
    mlp_out_blocks = intermediate_size // MLP_OUT_CHUNK

    @pl.program
    class Qwen3Scope3:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope3(
            self,
            attn_out: pl.Tensor[[batch, hidden_size], pl.BF16],
            hidden_states: pl.Tensor[[batch, hidden_size], pl.BF16],
            wo: pl.Tensor[[hidden_size, hidden_size], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden_size], pl.FP32],
            w_gate: pl.Tensor[[hidden_size, intermediate_size], pl.BF16],
            w_up: pl.Tensor[[hidden_size, intermediate_size], pl.BF16],
            w_down: pl.Tensor[[intermediate_size, hidden_size], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, hidden_size], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden_size], pl.BF16]:
            for b0 in pl.range(0, batch, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, hidden_size], dtype=pl.FP32)

                for ob in pl.range(q_out_blocks):
                    o0 = ob * Q_OUT_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        a_chunk_0 = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, 0])
                        w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid_sum = pl.add(o_acc, resid)
                        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                post_norm_tile = pl.create_tensor([BATCH_TILE, hidden_size], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH_TILE]),
                        )
                    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        resid_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(resid_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])),
                            gamma,
                        )
                        post_norm_tile = pl.assemble(
                            post_norm_tile,
                            pl.cast(normed, target_type=pl.BF16),
                            [0, k0],
                        )

                mlp_tile = pl.create_tensor([BATCH_TILE, intermediate_size], dtype=pl.BF16)
                for ob in pl.range(mlp_out_blocks):
                    o0 = ob * MLP_OUT_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, hidden_blocks):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0])

                for dob in pl.range(hidden_blocks):
                    d0 = dob * K_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                        for ob in pl.range(1, mlp_out_blocks):
                            o0 = ob * MLP_OUT_CHUNK
                            down_chunk = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_chunk, w_down_chunk)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        out_chunk = pl.add(down_acc, pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0]))
                        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

            return out

    return Qwen3Scope3


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
        TensorSpec("attn_out", [batch, hidden_size], torch.bfloat16, init_value=init_attn_out),
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_scope3(tensors):
    import torch

    attn_out = tensors["attn_out"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())
    tensors["out"][:] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_scope3_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_qwen3_scope3,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(platform=args.platform, device_id=args.device, runtime_profiling=args.runtime_profiling),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
