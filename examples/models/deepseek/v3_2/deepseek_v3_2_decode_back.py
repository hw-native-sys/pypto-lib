# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode BACK part (batch=16, max_seq=4096).

BACK boundary:
- start from combine buffer read
- run full residual + MLP + output path
"""


import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096

HIDDEN = 7168
INTERMEDIATE = 18432
NUM_HEADS = 128
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EP_NODES = 128

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# tiling constants.
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 256
BATCH_TILE = 16


def build_deepseek_v3_2_decode_back_program(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    BATCH_SIZE = batch
    HIDDEN_SIZE = hidden_size
    INTER_SIZE = intermediate_size
    ATTN_OUT_SIZE = attn_out_size
    EP_NODES_SIZE = ep_nodes

    ATTN_BLOCKS = (ATTN_OUT_SIZE + K_CHUNK - 1) // K_CHUNK
    HIDDEN_BLOCKS = (HIDDEN_SIZE + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_SIZE + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_SIZE + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK

    @pl.program
    class DeepSeekV32DecodeBack:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_back_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
            node_id_t: pl.Tensor[[1], pl.INT32],
            # combine buffer from cross-node communication
            combine_buf: pl.Tensor[[EP_NODES_SIZE, BATCH_SIZE, ATTN_OUT_SIZE], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_SIZE, HIDDEN_SIZE], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_SIZE], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_SIZE, INTER_SIZE], pl.BF16],
            w_down: pl.Tensor[[INTER_SIZE, HIDDEN_SIZE], pl.BF16],
            out: pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16],
        ) -> pl.Tensor[[BATCH_SIZE, HIDDEN_SIZE], pl.BF16]:
            # Scope: output projection + residual + post-rms + MLP + residual.
            node_id = pl.cast(pl.tensor.read(node_id_t, [0]), pl.INDEX)
            for b0 in pl.range(0, BATCH_SIZE, BATCH_TILE):
                resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_SIZE], dtype=pl.FP32)
                # Read combine results from this node view.
                combined_3d = pl.slice(
                    combine_buf, [1, BATCH_TILE, ATTN_OUT_SIZE], [node_id, b0, 0]
                )
                combined = pl.reshape(combined_3d, [BATCH_TILE, ATTN_OUT_SIZE])

                # O projection and residual.
                for ob in pl.range(Q_OUT_BLOCKS):
                    o0 = ob * Q_OUT_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        a_chunk_0 = pl.slice(combined, [BATCH_TILE, K_CHUNK], [0, 0])
                        w_chunk_0 = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [0, o0])
                        o_acc = pl.matmul(a_chunk_0, w_chunk_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, ATTN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.slice(combined, [BATCH_TILE, K_CHUNK], [0, k0])
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.matmul_acc(o_acc, a_chunk, w_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]), target_type=pl.FP32
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                # Post RMSNorm.
                post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_SIZE], dtype=pl.BF16)
                with pl.at(level=pl.Level.CORE_GROUP):
                    sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]))
                    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(
                            pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1])), gamma
                        )
                        post_norm_tile = pl.assemble(
                            post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                        )

                # MLP.
                mlp_tile = pl.create_tensor([BATCH_TILE, INTER_SIZE], dtype=pl.BF16)
                for ob in pl.range(MLP_OUT_BLOCKS):
                    o0 = ob * MLP_OUT_CHUNK

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
                        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
                        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])

                # Down projection + final residual writeback.
                for dob in pl.range(HIDDEN_BLOCKS):
                    d0 = dob * K_CHUNK
                    with pl.at(level=pl.Level.CORE_GROUP):
                        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, 0])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [0, d0])
                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                        for ob in pl.range(1, MLP_OUT_BLOCKS):
                            o0 = ob * MLP_OUT_CHUNK
                            down_mlp_chunk_bf16 = pl.slice(mlp_tile, [BATCH_TILE, MLP_OUT_CHUNK], [0, o0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [o0, d0])
                            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        out_chunk = pl.add(
                            down_acc,
                            pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, d0]),
                        )
                        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [b0, d0])

            return out

    return DeepSeekV32DecodeBack


def build_tensor_specs(
    batch: int = BATCH,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    node_id_data = torch.tensor([0], dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_combine_buf():
        return torch.rand(ep_nodes, batch, attn_out_size) - 0.5

    def init_wo():
        return (torch.rand(attn_out_size, hidden_size) - 0.5) / (attn_out_size ** 0.5)

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5)

    def init_w_up():
        return (torch.rand(hidden_size, intermediate_size) - 0.5) / (hidden_size ** 0.5)

    def init_w_down():
        return (torch.rand(intermediate_size, hidden_size) - 0.5) / (intermediate_size ** 0.5)

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("node_id_t", [1], torch.int32, init_value=node_id_data),
        TensorSpec("combine_buf", [ep_nodes, batch, attn_out_size], torch.bfloat16, init_value=init_combine_buf),
        TensorSpec("wo", [attn_out_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, intermediate_size], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [intermediate_size, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_deepseek_v3_2_decode_back(tensors):
    """PyTorch reference for decode-back: combine selection + scope-3 math."""
    import torch

    hidden_states = tensors["hidden_states"]
    node_id = int(tensors["node_id_t"][0].item())
    combine_buf = tensors["combine_buf"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    inter_size = w_gate.shape[1]
    attn_out_size = combine_buf.shape[2]

    # Match the chunked accumulation order used by the A2/A3 path to reduce
    # BF16/FP32 drift at validation time.
    k_chunk = K_CHUNK
    q_out_chunk = Q_OUT_CHUNK
    mlp_out_chunk = MLP_OUT_CHUNK

    combined = combine_buf[node_id]
    resid1 = torch.zeros(batch, hidden_size, dtype=torch.float32)

    # 1. Output projection (BF16 inputs, FP32 accumulation) + residual.
    for o0 in range(0, hidden_size, q_out_chunk):
        o_acc = torch.zeros(batch, q_out_chunk, dtype=torch.float32)
        for k0 in range(0, attn_out_size, k_chunk):
            o_acc += combined[:, k0:k0 + k_chunk].float() @ wo[k0:k0 + k_chunk, o0:o0 + q_out_chunk].float()
        resid1[:, o0:o0 + q_out_chunk] = o_acc + hidden_states[:, o0:o0 + q_out_chunk].float()

    # 2. Post-attention RMSNorm.
    sq_sum = torch.zeros(1, batch, dtype=torch.float32)
    for k0 in range(0, hidden_size, k_chunk):
        x_chunk = resid1[:, k0:k0 + k_chunk]
        sq_sum += (x_chunk * x_chunk).sum(dim=-1, keepdim=True).T
    inv_rms = torch.reciprocal(torch.sqrt(sq_sum * HIDDEN_INV + EPS))

    post_norm_bf16 = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for k0 in range(0, hidden_size, k_chunk):
        x_chunk = resid1[:, k0:k0 + k_chunk]
        gamma = post_rms_weight[:, k0:k0 + k_chunk].float()
        normed = x_chunk * inv_rms.T * gamma
        post_norm_bf16[:, k0:k0 + k_chunk] = normed.bfloat16()

    # 3. SwiGLU MLP with chunked projections.
    mlp_bf16 = torch.zeros(batch, inter_size, dtype=torch.bfloat16)
    for o0 in range(0, inter_size, mlp_out_chunk):
        gate_acc = torch.zeros(batch, mlp_out_chunk, dtype=torch.float32)
        up_acc = torch.zeros(batch, mlp_out_chunk, dtype=torch.float32)
        for k0 in range(0, hidden_size, k_chunk):
            post_chunk = post_norm_bf16[:, k0:k0 + k_chunk].float()
            gate_acc += post_chunk @ w_gate[k0:k0 + k_chunk, o0:o0 + mlp_out_chunk].float()
            up_acc += post_chunk @ w_up[k0:k0 + k_chunk, o0:o0 + mlp_out_chunk].float()
        sigmoid = torch.reciprocal(torch.exp(-gate_acc) + 1.0)
        mlp_bf16[:, o0:o0 + mlp_out_chunk] = (gate_acc * sigmoid * up_acc).bfloat16()

    # 4. Down projection + final residual.
    out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for d0 in range(0, hidden_size, k_chunk):
        down_acc = torch.zeros(batch, k_chunk, dtype=torch.float32)
        for o0 in range(0, inter_size, mlp_out_chunk):
            down_acc += mlp_bf16[:, o0:o0 + mlp_out_chunk].float() @ w_down[o0:o0 + mlp_out_chunk, d0:d0 + k_chunk].float()
        out[:, d0:d0 + k_chunk] = (down_acc + resid1[:, d0:d0 + k_chunk]).bfloat16()

    tensors["out"][:] = out


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
        program=build_deepseek_v3_2_decode_back_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v3_2_decode_back,
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
