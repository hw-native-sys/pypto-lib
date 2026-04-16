# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP prefill BACK Scope 1+2 — wo projection + residual + post RMSNorm.

Two-loop approach: scope1 writes to resid1 (Out tensor), scope2 reads from resid1.
"""
from __future__ import annotations

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
NUM_HEADS = 128
V_HEAD_DIM = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
EP_NODES = 128

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

K_CHUNK = 512
Q_OUT_CHUNK = 128
TOK_TILE = 16
COMBINE_CHUNK = 128

_NODE_ID = 0  # Module-level variable for golden function


def build_deepseek_v3_2_prefill_back_scope12_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
    node_id: int = 0,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    ATTN_OUT_CFG = attn_out_size
    EP_NODES_CFG = ep_nodes

    HIDDEN_INV_CFG = 1.0 / HIDDEN_CFG

    COMBINE_BLOCKS = (ATTN_OUT_CFG + COMBINE_CHUNK - 1) // COMBINE_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK

    @pl.program
    class DeepSeekV32PrefillBackScope12:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope12(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            combine_buf: pl.Tensor[[EP_NODES_CFG, BATCH_CFG, MAX_SEQ_CFG, ATTN_OUT_CFG], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_CFG, HIDDEN_CFG], pl.BF16],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            resid1: pl.Out[pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.FP32]],
            post_norm: pl.Out[pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16]:
            for b in pl.parallel(0, BATCH_CFG, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    combine_local = pl.create_tensor([TOK_TILE, ATTN_OUT_CFG], dtype=pl.BF16)

                    for ob in pl.range(Q_OUT_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK
                        with pl.incore():
                            for cb in pl.range(COMBINE_BLOCKS):
                                c0 = cb * COMBINE_CHUNK
                                chunk = pl.reshape(
                                    pl.slice(combine_buf, [1, 1, TOK_TILE, COMBINE_CHUNK], [node_id, b, p0, c0]),
                                    [TOK_TILE, COMBINE_CHUNK],
                                )
                                combine_local = pl.assemble(combine_local, chunk, [0, c0])
                        with pl.incore():
                            c_tile_0 = pl.slice(combine_local, [TOK_TILE, COMBINE_CHUNK], [0, 0])
                            w_tile_0 = pl.slice(wo, [COMBINE_CHUNK, Q_OUT_CHUNK], [0, o0])
                            o_acc = pl.matmul(c_tile_0, w_tile_0, out_dtype=pl.FP32)
                            for cb in pl.range(1, COMBINE_BLOCKS):
                                c0 = cb * COMBINE_CHUNK
                                c_tile_i = pl.slice(combine_local, [TOK_TILE, COMBINE_CHUNK], [0, c0])
                                w_tile_i = pl.slice(wo, [COMBINE_CHUNK, Q_OUT_CHUNK], [c0, o0])
                                o_acc = pl.matmul_acc(o_acc, c_tile_i, w_tile_i)
                            resid1 = pl.assemble(resid1, o_acc, [b, p0, o0])
                        with pl.incore():
                            proj = pl.slice(resid1, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0])
                            resid = pl.cast(
                                pl.slice(hidden_states, [1, TOK_TILE, Q_OUT_CHUNK], [b, p0, o0]),
                                target_type=pl.FP32,
                            )
                            resid1 = pl.assemble(resid1, pl.add(proj, resid), [b, p0, o0])

                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    with pl.incore():
                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.reshape(
                                pl.slice(resid1, [1, TOK_TILE, K_CHUNK], [b, p0, k0]),
                                [TOK_TILE, K_CHUNK]
                            )
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))

                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV_CFG), EPS))

                    post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        with pl.incore():
                            x_chunk = pl.reshape(
                                pl.slice(resid1, [1, TOK_TILE, K_CHUNK], [b, p0, k0]),
                                [TOK_TILE, K_CHUNK]
                            )
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                            post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    post_norm = pl.assemble(post_norm, post_norm_tile, [b, p0, 0])

            return post_norm

    return DeepSeekV32PrefillBackScope12


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
):
    import torch
    from pypto.runtime import TensorSpec

    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, max_seq_len, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("combine_buf", [ep_nodes, batch, max_seq_len, attn_out_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [attn_out_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn),
        TensorSpec("resid1", [batch, max_seq_len, hidden_size], torch.float32, is_output=True),
        TensorSpec("post_norm", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
    ]


def golden_scope12(tensors, params):
    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    node_id = _NODE_ID  # Use module-level variable
    combine_buf = tensors["combine_buf"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]

    batch = hidden_states.shape[0]
    max_seq = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    eps = 1e-6

    resid1 = torch.zeros(batch, max_seq, hidden_size, dtype=torch.float32)
    post_norm = torch.zeros(batch, max_seq, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
        for p0_idx in range(tok_blocks):
            p0 = p0_idx * TOK_TILE
            combined = combine_buf[node_id, b, p0:p0 + TOK_TILE, :].float()
            o_proj = torch.matmul(combined, wo.float())
            resid = hidden_states[b, p0:p0 + TOK_TILE, :].float()
            resid1[b, p0:p0 + TOK_TILE, :] = o_proj + resid

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        for p0 in range(0, seq_len_b, TOK_TILE):
            resid1_tile_full = resid1[b, p0:p0 + TOK_TILE, :]
            sq_sum = (resid1_tile_full ** 2).sum(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(sq_sum / hidden_size + eps)
            normed = resid1_tile_full * inv_rms * post_rms_weight.float()
            post_norm[b, p0:p0 + TOK_TILE, :] = normed.bfloat16()

    tensors["resid1"][:] = resid1
    tensors["post_norm"][:] = post_norm


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    attn_out_size: int = ATTN_OUT,
    ep_nodes: int = EP_NODES,
    node_id: int = 0,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    global _NODE_ID
    _NODE_ID = node_id  # Set module-level variable for golden function

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_deepseek_v3_2_prefill_back_scope12_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        attn_out_size=attn_out_size,
        ep_nodes=ep_nodes,
        node_id=node_id,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        attn_out_size=attn_out_size,
        ep_nodes=ep_nodes,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_scope12,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            runtime_profiling=runtime_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--attn-out", type=int, default=512)
    parser.add_argument("--ep-nodes", type=int, default=2)
    parser.add_argument("--node-id", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(
        batch=args.batch,
        max_seq_len=args.max_seq,
        hidden_size=args.hidden,
        attn_out_size=args.attn_out,
        ep_nodes=args.ep_nodes,
        node_id=args.node_id,
        platform=args.platform,
        device_id=args.device,
        runtime_profiling=args.runtime_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
