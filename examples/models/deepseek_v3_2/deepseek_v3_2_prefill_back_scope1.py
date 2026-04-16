# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP prefill BACK Scope 1 — wo projection + residual.
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

K_CHUNK = 512
Q_OUT_CHUNK = 128
TOK_TILE = 16
COMBINE_CHUNK = 128

_NODE_ID = 0  # Module-level variable for golden function


def build_deepseek_v3_2_prefill_back_scope1_program(
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

    COMBINE_BLOCKS = (ATTN_OUT_CFG + COMBINE_CHUNK - 1) // COMBINE_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK

    @pl.program
    class DeepSeekV32PrefillBackScope1:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope1(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            combine_buf: pl.Tensor[[EP_NODES_CFG, BATCH_CFG, MAX_SEQ_CFG, ATTN_OUT_CFG], pl.BF16],
            wo: pl.Tensor[[ATTN_OUT_CFG, HIDDEN_CFG], pl.BF16],
            resid1: pl.Out[pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.FP32]],
        ) -> pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.FP32]:
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

            return resid1

    return DeepSeekV32PrefillBackScope1


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
        TensorSpec("resid1", [batch, max_seq_len, hidden_size], torch.float32, is_output=True),
    ]


def golden_scope1(tensors, params):
    import torch

    hidden_states = tensors["hidden_states"]
    seq_lens = tensors["seq_lens"]
    node_id = _NODE_ID  # Use module-level variable
    combine_buf = tensors["combine_buf"]
    wo = tensors["wo"]

    batch = hidden_states.shape[0]
    max_seq = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]

    resid1 = torch.zeros(batch, max_seq, hidden_size, dtype=torch.float32)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
        for p0_idx in range(tok_blocks):
            p0 = p0_idx * TOK_TILE
            combined = combine_buf[node_id, b, p0:p0 + TOK_TILE, :].float()
            o_proj = torch.matmul(combined, wo.float())
            resid = hidden_states[b, p0:p0 + TOK_TILE, :].float()
            resid1[b, p0:p0 + TOK_TILE, :] = o_proj + resid

    tensors["resid1"][:] = resid1


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

    program = build_deepseek_v3_2_prefill_back_scope1_program(
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
        golden=golden_scope1,
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
