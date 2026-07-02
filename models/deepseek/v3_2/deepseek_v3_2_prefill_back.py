# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer prefill BACK part (batch=16, max_seq=4096).
[NOTE] Current test is reduced to batch=4, max_seq=128 for faster iteration and dodge HBM OOM issue.

BACK boundary:
- read combine tensor
- run complete residual + MLP + output path
"""

import pypto.language as pl


BATCH = 1
MAX_SEQ = 128
HIDDEN = 7168
INTERMEDIATE = 18432
NUM_HEADS = 128
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EP_NODES = 128

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_TILE = 128
Q_OUT_TILE = 64
MLP_OUT_TILE = 128
TOK_TILE = 64
# RMSNorm block-chunking: each core scope normalizes NORM_TILE contiguous
# HIDDEN // K_TILE blocks (= 56 blocks; 56 = 7 * 8).
NORM_TILE = 8


def build_deepseek_v3_2_prefill_back_program():
    @pl.program
    class DeepSeekV32PrefillBack:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_prefill_back_layer(
            self,
            hidden_states: pl.Tensor[[BATCH, MAX_SEQ, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            combine_buf: pl.Tensor[[EP_NODES, BATCH, MAX_SEQ, ATTN_OUT], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT, HIDDEN], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
            w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, MAX_SEQ, HIDDEN], pl.BF16]],
        ) -> pl.Tensor[[BATCH, MAX_SEQ, HIDDEN], pl.BF16]:
            for b in pl.parallel(0, BATCH, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    # GM intermediate tensors.
                    resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                    attn_tile = pl.create_tensor([TOK_TILE, ATTN_OUT], dtype=pl.BF16)

                    # Stage 1: Copy combine_buf 4D -> attn_tile 2D.
                    with pl.at(level=pl.Level.CORE_GROUP):
                        for kb in pl.range(ATTN_OUT // K_TILE):
                            k0 = kb * K_TILE
                            a_chunk_fp32 = pl.reshape(
                                pl.cast(
                                    pl.slice(combine_buf, [1, 1, TOK_TILE, K_TILE], [0, b, p0, k0],
                                             valid_shape=[1, 1, valid_tok, K_TILE]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, K_TILE],
                            )
                            a_chunk_bf16 = pl.cast(a_chunk_fp32, target_type=pl.BF16)
                            attn_tile = pl.assemble(attn_tile, a_chunk_bf16, [0, k0])

                    # Stage 2: Output projection + first residual.
                    for ob in pl.range(HIDDEN // Q_OUT_TILE):
                        o0 = ob * Q_OUT_TILE

                        # Cube: chained matmul.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            tile_a = pl.slice(attn_tile, [TOK_TILE, K_TILE], [0, 0])
                            tile_w = pl.slice(wo, [K_TILE, Q_OUT_TILE], [0, o0])
                            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                            for kb in pl.range(1, ATTN_OUT // K_TILE):
                                k0 = kb * K_TILE
                                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_TILE], [0, k0])
                                tile_w_i = pl.slice(wo, [K_TILE, Q_OUT_TILE], [k0, o0])
                                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)

                            resid1_tile = pl.assemble(resid1_tile, o_acc, [0, o0])

                        # Vector: add residual.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            resid_chunk = pl.reshape(
                                pl.cast(
                                    pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_TILE], [b, p0, o0],
                                             valid_shape=[1, valid_tok, Q_OUT_TILE]),
                                    target_type=pl.FP32,
                                ),
                                [TOK_TILE, Q_OUT_TILE],
                            )
                            mm_out = pl.slice(resid1_tile, [TOK_TILE, Q_OUT_TILE], [0, o0])
                            resid_sum = pl.add(mm_out, resid_chunk)
                            resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])

                    # Stage 3: Post-attention RMSNorm.
                    post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)

                    # 3a: Compute inv_rms (reduction — sequential).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        sq_sum = pl.full([1, TOK_TILE], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN // K_TILE):
                            k0 = kb * K_TILE
                            x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_TILE], [0, k0])
                            sq_sum = pl.add(
                                sq_sum,
                                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, TOK_TILE]),
                            )
                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    # 3b: Normalize + gamma. Chunk HIDDEN // K_TILE across cores; each
                    # core scope handles NORM_TILE contiguous blocks.
                    for kb0 in pl.parallel(0, HIDDEN // K_TILE, NORM_TILE):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for kb in pl.range(kb0, kb0 + NORM_TILE):
                                k0 = kb * K_TILE
                                x_chunk = pl.slice(resid1_tile, [TOK_TILE, K_TILE], [0, k0])
                                gamma = pl.slice(post_rms_weight, [1, K_TILE], [0, k0])
                                normed = pl.col_expand_mul(
                                    pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [TOK_TILE, 1])),
                                    gamma,
                                )
                                normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                                post_norm_tile = pl.assemble(post_norm_tile, normed_bf16, [0, k0])

                    # Stage 4: MLP gate/up + SiLU -> mlp_tile.
                    mlp_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.BF16)
                    for ob in pl.range(INTERMEDIATE // MLP_OUT_TILE):
                        o0 = ob * MLP_OUT_TILE

                        # Gate matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_TILE], [0, 0])
                            wg0 = pl.slice(w_gate, [K_TILE, MLP_OUT_TILE], [0, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.range(1, HIDDEN // K_TILE):
                                k0 = kb * K_TILE
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_TILE], [0, k0])
                                wgi = pl.slice(w_gate, [K_TILE, MLP_OUT_TILE], [k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)

                        # Up matmul chain.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_TILE], [0, 0])
                            wu0 = pl.slice(w_up, [K_TILE, MLP_OUT_TILE], [0, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.range(1, HIDDEN // K_TILE):
                                k0 = kb * K_TILE
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_TILE], [0, k0])
                                wui = pl.slice(w_up, [K_TILE, MLP_OUT_TILE], [k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)

                        # SiLU activation -> mlp_tile.
                        with pl.at(level=pl.Level.CORE_GROUP):
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_tile = pl.assemble(mlp_tile, pl.cast(mlp_chunk, target_type=pl.BF16), [0, o0])

                    # Stage 5: Down projection + final residual -> BF16 output.
                    for dob in pl.range(HIDDEN // K_TILE):
                        d0 = dob * K_TILE

                        with pl.at(level=pl.Level.CORE_GROUP):
                            mlp0 = pl.slice(mlp_tile, [TOK_TILE, MLP_OUT_TILE], [0, 0])
                            wd0 = pl.slice(w_down, [MLP_OUT_TILE, K_TILE], [0, d0])
                            down_acc = pl.matmul(mlp0, wd0, out_dtype=pl.FP32)
                            for ob in pl.range(1, INTERMEDIATE // MLP_OUT_TILE):
                                o0 = ob * MLP_OUT_TILE
                                mlpi = pl.slice(mlp_tile, [TOK_TILE, MLP_OUT_TILE], [0, o0])
                                wdi = pl.slice(w_down, [MLP_OUT_TILE, K_TILE], [o0, d0])
                                down_acc = pl.matmul_acc(down_acc, mlpi, wdi)

                        with pl.at(level=pl.Level.CORE_GROUP):
                            out_chunk = pl.add(
                                down_acc,
                                pl.slice(resid1_tile, [TOK_TILE, K_TILE], [0, d0]),
                            )
                            out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [b, p0, d0])

            return out

    return DeepSeekV32PrefillBack


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_seq_lens():
        n_blocks = max_seq_len // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_hidden_states():
        return torch.rand(batch, max_seq_len, hidden_size) - 0.5

    def init_combine_buf():
        return torch.rand(ep_nodes, batch, max_seq_len, attn_out_size) - 0.5

    def init_wo():
        return (torch.rand(attn_out_size, hidden_size) - 0.5) / attn_out_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, max_seq_len, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("combine_buf", [ep_nodes, batch, max_seq_len, attn_out_size], torch.bfloat16, init_value=init_combine_buf),
        TensorSpec("wo", [attn_out_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_prefill_back(tensors):
    """Reference computation for DeepSeek V3.2 prefill back.

    Steps:
      1. Read combine_buf[node_id] as attn input
      2. Output projection: attn x wo + residual (chunked BF16 matmul)
      3. Post-attention RMSNorm
      4. SwiGLU MLP: gate/up projections, silu(gate) * up, down projection
      5. Final residual addition -> BF16 output

    All matmuls use chunked BF16 inputs with FP32 accumulation to match
    the hardware kernel's precision path exactly.
    """
    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    combine_buf = tensors["combine_buf"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    out_t = tensors["out"]

    eps = EPS
    post_rms_f = post_rms_weight.float()

    def chunked_bf16_matmul(a_bf16, b_bf16, k_chunk=K_TILE):
        """BF16 x BF16 -> FP32 with chunked K-dim accumulation."""
        K = a_bf16.shape[-1]
        acc = torch.zeros(*a_bf16.shape[:-1], b_bf16.shape[-1], dtype=torch.float32)
        for k0 in range(0, K, k_chunk):
            k1 = min(k0 + k_chunk, K)
            acc += torch.matmul(a_bf16[..., k0:k1].float(), b_bf16[k0:k1, :].float())
        return acc

    batch = hidden_states.shape[0]
    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        sl = slice(0, seq_len_b)

        # 1. Attn input from combine_buf.
        attn_bf16 = combine_buf[0, b, sl, :]
        hs = hidden_states[b, sl, :].float()

        # 2. Output projection + first residual (chunked BF16 matmul, FP32 accum).
        resid1 = chunked_bf16_matmul(attn_bf16, wo) + hs

        # 3. Post-attention RMSNorm.
        variance = resid1.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        normed_bf16 = (resid1 * inv_rms * post_rms_f).bfloat16()

        # 4. SwiGLU MLP (chunked BF16 matmul paths match kernel).
        gate = chunked_bf16_matmul(normed_bf16, w_gate)
        up = chunked_bf16_matmul(normed_bf16, w_up)
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = chunked_bf16_matmul(mlp_bf16, w_down, k_chunk=MLP_OUT_TILE)

        # 5. Final residual -> BF16.
        out_t[b, sl, :] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_prefill_back_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_back,
        compile_cfg=dict(dump_passes=False),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

