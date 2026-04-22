# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 3 — output projection + residual + post RMSNorm + MLP.

For each batch element with variable-length tokens (processed in TOK_TILE chunks):
  1. Output projection: attn_out x wo + first residual
  2. Post-attention RMSNorm
  3. MLP gate/up projections, SiLU activation, down projection
  4. Final residual addition -> BF16 output
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
HIDDEN = 8192
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Tiling constants.
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 128
TOK_TILE = 64


def build_prefill_scope3_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    hidden = hidden_size
    inter = intermediate_size
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    mlp_out_blocks = inter // MLP_OUT_CHUNK
    hidden_inv = 1.0 / hidden

    @pl.program
    class PrefillScope3Program:
        @pl.function(type=pl.FunctionType.Opaque)
        def prefill_scope3(
            self,
            attn_out: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            hidden_states: pl.Tensor[[batch, max_seq, hidden], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, max_seq, hidden], pl.BF16]:
            for b in pl.parallel(0, batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # GM intermediate tensors.
                    resid1_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.FP32)
                    attn_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)

                    # Stage 1: Copy attn_out 3D -> attn_tile 2D.
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            a_chunk_bf16 = pl.reshape(
                                pl.slice(attn_out, [1, TOK_TILE, K_CHUNK], [b, p0, k0],
                                         valid_shape=[1, valid_tok, K_CHUNK]),
                                [TOK_TILE, K_CHUNK],
                            )
                            attn_tile = pl.assemble(attn_tile, a_chunk_bf16, [0, k0])

                    # Stage 2: Output projection + first residual.
                    for ob in pl.range(q_out_blocks):
                        o0 = ob * Q_OUT_CHUNK

                        with pl.at(level=pl.Level.CORE_GROUP):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_CHUNK]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_CHUNK],
                            )
                            resid_sum = pl.add(o_acc, resid_chunk)
                            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                    # Stage 3: Post-attention RMSNorm.
                    post_norm_tile = pl.create_tensor([TOK_TILE, hidden], dtype=pl.BF16)
                    with pl.at(level=pl.Level.CORE_GROUP):
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, hidden_inv), EPS)))

                        for kb in pl.range(hidden_blocks):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [TOK_TILE, 1])),
                                gamma,
                            )
                            normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                            post_norm_tile = pl.assemble(
                                post_norm_tile, normed_bf16, [0, k0])

                    # Stage 4: MLP gate/up + SiLU -> mlp_tile.
                    mlp_tile = pl.create_tensor([TOK_TILE, inter], dtype=pl.BF16)
                    for ob in pl.range(mlp_out_blocks):
                        o0 = ob * MLP_OUT_CHUNK

                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, hidden_blocks):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                            mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                    # Stage 5: Down projection (matmul_acc chain) + final residual -> output.
                    for dob in pl.range(hidden_blocks):
                        d0 = dob * K_CHUNK
                        with pl.at(level=pl.Level.CORE_GROUP):
                            mlp_chunk_0 = pl.slice(mlp_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, 0])
                            w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                            for ob in pl.range(1, mlp_out_blocks):
                                o0 = ob * MLP_OUT_CHUNK
                                mlp_chunk_i = pl.slice(mlp_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                                w_down_chunk_i = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                                down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            out_chunk = pl.add(
                                down_acc,
                                pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                            )
                            out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                            out = pl.assemble(out, out_chunk_bf16, [b, p0, d0])

            return out

    return PrefillScope3Program


def golden_prefill_scope3(tensors):
    """Reference computation for Scope 3 (prefill).

    Steps:
      1. Output projection: attn_out x wo + residual
      2. Post-attention RMSNorm
      3. SwiGLU MLP: gate/up projections, silu(gate) * up, down projection
      4. Final residual addition -> BF16 output
    """
    import torch

    attn_out_t = tensors["attn_out"]
    seq_lens = tensors["seq_lens"]
    hidden_states = tensors["hidden_states"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    eps = EPS
    out_t = tensors["out"]

    # Pre-convert weights to FP32.
    wo_f = wo.float()
    post_rms_f = post_rms_weight.float()
    w_gate_f = w_gate.float()
    w_up_f = w_up.float()
    w_down_f = w_down.float()

    for b in range(attn_out_t.shape[0]):
        seq_len_b = seq_lens[b].item()
        sl = slice(0, seq_len_b)

        attn = attn_out_t[b, sl, :].float()
        hs = hidden_states[b, sl, :].float()

        # 1. Output projection + first residual.
        resid1 = torch.matmul(attn, wo_f) + hs

        # 2. Post-attention RMSNorm.
        variance = resid1.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        normed_bf16 = (resid1 * inv_rms * post_rms_f).bfloat16()

        # 3. SwiGLU MLP.
        normed_f = normed_bf16.float()
        gate = torch.matmul(normed_f, w_gate_f)
        up = torch.matmul(normed_f, w_up_f)
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = torch.matmul(mlp_bf16.float(), w_down_f)

        # 4. Final residual -> BF16.
        out_t[b, sl, :] = (down + resid1).bfloat16()


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch
    from golden import TensorSpec

    def init_seq_lens():
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_attn_out():
        return torch.rand(batch, max_seq, hidden_size) - 0.5

    def init_hidden_states():
        return torch.rand(batch, max_seq, hidden_size) - 0.5

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
        TensorSpec("attn_out", [batch, max_seq, hidden_size], torch.bfloat16,
                   init_value=init_attn_out),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("hidden_states", [batch, max_seq, hidden_size], torch.bfloat16,
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
        TensorSpec("out", [batch, max_seq, hidden_size], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a5"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_prefill_scope3_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_prefill_scope3,
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
