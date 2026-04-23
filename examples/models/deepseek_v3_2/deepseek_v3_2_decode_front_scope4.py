# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode FRONT part — Scope 4 only.

Standalone post-topk boundary:
- inputs already include the Scope 1 rotated q_proj
- sparse attention consumes topk_idx, kv_cache, and pe_cache
- outputs are written into the cross-node dispatch buffer

This file keeps the same scope-4 kernel entry signature as the current
standalone validation path while rebuilding the wrapper and implementation from
clean references:
- decode_back.py for wrapper/runtime style
- ds32exp.py Scope 4 for device functionality
- decode_front_scope2c.py.bak for the kernel input contract

Current standalone differences versus ds32exp.py Scope 4:
- the device path keeps duplicated 16-row intermediates through q-nope
    projection, online softmax state, and latent-to-V projection because that is
    the backend-safe lowering shape on a2a3, not because the model needs 16 rows
    - 1-row lowering is not allowed on a2a3; yet it's required on a5 (line 597 
        and 601 in ds32exp.py).
- the kernel now consumes `topk_idx` with a global 1-D internal view
    (`topk_idx_flat = pl.reshape(topk_idx, [batch * index_topk])`) and linear
    indexing (`topk_base + kk`) while keeping the input signature 2-D

Note: The current `--profile full` preset keeps `max_seq_len=128` while restoring the large
    inner dimensions from `max_seq_len=4096`, this can accelerate the test while still 
    validating the large-dimension logic (around < 10s), and it can be easily 
    switched to `max_seq_len=4096` if desired (time < 6min)

Kernel stage order in the rewritten reduced-profile path:
- Stage 1: load per-head q_pe and project q_nope into the latent space
- Stage 2: run sparse online softmax accumulation in latent space
- Stage 3: project latent context back to V chunks and assemble the row
- Stage 4: cast and stash the finished attention row
- Stage 5: route the finished row into the cross-node dispatch buffer

Defaults are intentionally reduced for faster standalone validation.
"""

import pypto.language as pl

import os
os.environ.setdefault("PTO2_RING_TASK_WINDOW", "524288")
os.environ.setdefault("PTO2_RING_DEP_POOL", "1048576")
os.environ.setdefault("PTO2_RING_HEAP", "4294967296")


REDUCED_PROFILE = {
    "batch": 16,
    "max_seq_len": 128,
    "num_heads": 16,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "index_topk": 16,
    "ep_nodes": 8,
}

FULL_PROFILE = {
    "batch": 16,
    "max_seq_len": 128, #"max_seq_len": 4096
    "num_heads": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_topk": 2048,
    "ep_nodes": 128,
}


BATCH = REDUCED_PROFILE["batch"]
MAX_SEQ = REDUCED_PROFILE["max_seq_len"]
NUM_HEADS = REDUCED_PROFILE["num_heads"]
KV_LORA_RANK = REDUCED_PROFILE["kv_lora_rank"]
QK_NOPE_HEAD_DIM = REDUCED_PROFILE["qk_nope_head_dim"]
QK_ROPE_HEAD_DIM = REDUCED_PROFILE["qk_rope_head_dim"]
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = REDUCED_PROFILE["v_head_dim"]
ATTN_OUT = NUM_HEADS * V_HEAD_DIM
INDEX_TOPK = REDUCED_PROFILE["index_topk"]
EP_NODES = REDUCED_PROFILE["ep_nodes"]
CACHE_ROWS = BATCH * MAX_SEQ

ATTN_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
Q_LATENT_CHUNK = 128
V_OUT_CHUNK = 16
HEAD_CHUNK = 8
BATCH_CHUNK = 4

def build_deepseek_v3_2_decode_front_scope4_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_topk: int = INDEX_TOPK,
    ep_nodes: int = EP_NODES,
):
    batch_cfg = batch
    max_seq_cfg = max_seq_len
    num_heads_cfg = num_heads
    kv_lora_rank_cfg = kv_lora_rank
    qk_nope_head_dim_cfg = qk_nope_head_dim
    qk_rope_head_dim_cfg = qk_rope_head_dim
    qk_head_dim_cfg = qk_nope_head_dim + qk_rope_head_dim
    v_head_dim_cfg = v_head_dim
    attn_out_cfg = num_heads * v_head_dim
    index_topk_cfg = index_topk
    topk_flat_elems_cfg = batch_cfg * index_topk_cfg
    ep_nodes_cfg = ep_nodes
    cache_rows_cfg = batch * max_seq_len
    v_out_blocks = (v_head_dim_cfg + V_OUT_CHUNK - 1) // V_OUT_CHUNK
    q_latent_blocks = (kv_lora_rank_cfg + Q_LATENT_CHUNK - 1) // Q_LATENT_CHUNK
    softmax_dup_cfg = HEAD_CHUNK
    matmul_row_pad_cfg = 16

    @pl.program
    class DeepSeekV32DecodeFrontScope4:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope4(
            self,
            q_proj: pl.Tensor[[batch_cfg, num_heads_cfg * qk_head_dim_cfg], pl.BF16],
            kv_cache: pl.Tensor[[cache_rows_cfg, kv_lora_rank_cfg], pl.BF16],
            pe_cache: pl.Tensor[[cache_rows_cfg, qk_rope_head_dim_cfg], pl.BF16],
            topk_idx: pl.Tensor[[batch_cfg, index_topk_cfg], pl.INT32],
            seq_lens: pl.Tensor[[batch_cfg], pl.INT32],
            layer_id_t: pl.Tensor[[1], pl.INT32],
            w_q_nope_to_latent: pl.Tensor[[num_heads_cfg, qk_nope_head_dim_cfg, kv_lora_rank_cfg], pl.BF16],
            w_latent_to_v: pl.Tensor[[num_heads_cfg, kv_lora_rank_cfg, v_head_dim_cfg], pl.BF16],
            dispatch_buf: pl.Tensor[[ep_nodes_cfg, batch_cfg, attn_out_cfg], pl.BF16],
        ) -> pl.Tensor[[ep_nodes_cfg, batch_cfg, attn_out_cfg], pl.BF16]:
            attn_front = pl.create_tensor([batch_cfg, attn_out_cfg], dtype=pl.BF16)
            topk_idx_flat = pl.reshape(topk_idx, [topk_flat_elems_cfg])

            for b in pl.parallel(0, batch_cfg, 1):
                attn_row = pl.create_tensor([1, attn_out_cfg], dtype=pl.FP32)
                sparse_k = pl.min(index_topk_cfg, pl.tensor.read(seq_lens, [b]))
                topk_base = b * index_topk_cfg

                for h in pl.parallel(0, num_heads_cfg, 1):
                    q_col = h * qk_head_dim_cfg
                    v_col = h * v_head_dim_cfg
                    with pl.at(level=pl.Level.CORE_GROUP):
                        q_pe = pl.cast(
                            pl.slice(
                                q_proj,
                                [1, qk_rope_head_dim_cfg],
                                [b, q_col + qk_nope_head_dim_cfg],
                            ),
                            target_type=pl.FP32,
                        )
                        q_pe_batch = pl.col_expand(
                            pl.full([matmul_row_pad_cfg, qk_rope_head_dim_cfg], dtype=pl.FP32, value=0.0),
                            q_pe,
                        )
                        q_nope_padded = pl.cast(
                            pl.full([matmul_row_pad_cfg, qk_nope_head_dim_cfg], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        )
                        q_nope_padded = pl.col_expand(
                            q_nope_padded,
                            pl.slice(q_proj, [1, qk_nope_head_dim_cfg], [b, q_col]),
                        )
                        q_nope_latent_batch = pl.full(
                            [matmul_row_pad_cfg, kv_lora_rank_cfg],
                            dtype=pl.FP32,
                            value=0.0,
                        )
                        for qb in pl.range(q_latent_blocks):
                            q0 = qb * Q_LATENT_CHUNK
                            w_qn_h = pl.reshape(
                                pl.slice(
                                    w_q_nope_to_latent,
                                    [1, qk_nope_head_dim_cfg, Q_LATENT_CHUNK],
                                    [h, 0, q0],
                                ),
                                [qk_nope_head_dim_cfg, Q_LATENT_CHUNK],
                            )
                            q_nope_latent_part = pl.matmul(
                                q_nope_padded,
                                w_qn_h,
                                out_dtype=pl.FP32,
                            )
                            q_nope_latent_batch = pl.assemble(q_nope_latent_batch, q_nope_latent_part, [0, q0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        topk_pos0 = pl.tensor.read(topk_idx_flat, [topk_base])
                        cache_s0 = b * max_seq_cfg + topk_pos0
                        kv_s0 = pl.cast(
                            pl.slice(kv_cache, [1, kv_lora_rank_cfg], [cache_s0, 0]),
                            target_type=pl.FP32,
                        )
                        pe_s0 = pl.cast(
                            pl.slice(pe_cache, [1, qk_rope_head_dim_cfg], [cache_s0, 0]),
                            target_type=pl.FP32,
                        )
                        oi = pl.col_expand(
                            pl.full([matmul_row_pad_cfg, kv_lora_rank_cfg], dtype=pl.FP32, value=0.0),
                            kv_s0,
                        )
                        pe_batch0 = pl.col_expand(
                            pl.full([matmul_row_pad_cfg, qk_rope_head_dim_cfg], dtype=pl.FP32, value=0.0),
                            pe_s0,
                        )
                        score_nope0 = pl.row_sum(pl.mul(q_nope_latent_batch, oi))
                        score_pe0 = pl.row_sum(pl.mul(q_pe_batch, pe_batch0))
                        mi = pl.mul(pl.add(score_nope0, score_pe0), ATTN_SCALE)
                        li = pl.exp(pl.sub(mi, mi))

                        for kk in pl.range(1, sparse_k):
                            topk_pos = pl.tensor.read(topk_idx_flat, [topk_base + kk])
                            cache_s = b * max_seq_cfg + topk_pos
                            kv_s = pl.cast(
                                pl.slice(kv_cache, [1, kv_lora_rank_cfg], [cache_s, 0]),
                                target_type=pl.FP32,
                            )
                            pe_s = pl.cast(
                                pl.slice(pe_cache, [1, qk_rope_head_dim_cfg], [cache_s, 0]),
                                target_type=pl.FP32,
                            )
                            kv_batch = pl.col_expand(
                                pl.full([matmul_row_pad_cfg, kv_lora_rank_cfg], dtype=pl.FP32, value=0.0),
                                kv_s,
                            )
                            pe_batch = pl.col_expand(
                                pl.full([matmul_row_pad_cfg, qk_rope_head_dim_cfg], dtype=pl.FP32, value=0.0),
                                pe_s,
                            )
                            score_nope = pl.row_sum(pl.mul(q_nope_latent_batch, kv_batch))
                            score_pe = pl.row_sum(pl.mul(q_pe_batch, pe_batch))
                            cur_mi = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(kv_batch, beta))
                            mi = mi_new
                        ctx_latent_batch = pl.row_expand_div(oi, li)

                    with pl.at(level=pl.Level.CORE_GROUP):
                        ctx_v_batch = pl.full([matmul_row_pad_cfg, v_head_dim_cfg], dtype=pl.FP32, value=0.0)
                        for vb in pl.range(v_out_blocks):
                            v0 = vb * V_OUT_CHUNK
                            wv_tile = pl.reshape(
                                pl.slice(
                                    w_latent_to_v,
                                    [1, kv_lora_rank_cfg, V_OUT_CHUNK],
                                    [h, 0, v0],
                                ),
                                [kv_lora_rank_cfg, V_OUT_CHUNK],
                            )
                            v_part_batch = pl.matmul(
                                pl.cast(ctx_latent_batch, target_type=pl.BF16),
                                wv_tile,
                                out_dtype=pl.FP32,
                            )
                            ctx_v_batch = pl.assemble(ctx_v_batch, v_part_batch, [0, v0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        ctx_v = pl.slice(ctx_v_batch, [1, v_head_dim_cfg], [0, 0])
                        attn_row = pl.assemble(attn_row, ctx_v, [0, v_col])

                with pl.at(level=pl.Level.CORE_GROUP):
                    attn_front = pl.assemble(attn_front, pl.cast(attn_row, target_type=pl.BF16), [b, 0])

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                layer_id = pl.tensor.read(layer_id_t, [0])
                for b in pl.parallel(0, batch_cfg, 1, chunk=4):
                    target_node = (b + layer_id) % ep_nodes_cfg
                    token_row = pl.slice(attn_front, [1, attn_out_cfg], [b, 0])
                    dispatch_buf = pl.assemble(dispatch_buf, token_row, [target_node, b, 0])

            return dispatch_buf

    return DeepSeekV32DecodeFrontScope4


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_topk: int = INDEX_TOPK,
    ep_nodes: int = EP_NODES,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    torch.manual_seed(4242)

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    cache_rows = batch * max_seq_len
    attn_out = num_heads * v_head_dim
    sparse_k = min(max_seq_len, index_topk)

    def init_q_proj():
        return (torch.rand(batch, num_heads * qk_head_dim, dtype=torch.float32) - 0.5)

    def init_kv_cache():
        return (torch.rand(cache_rows, kv_lora_rank, dtype=torch.float32) - 0.5)

    def init_pe_cache():
        return (torch.rand(cache_rows, qk_rope_head_dim, dtype=torch.float32) - 0.5)

    def init_topk_idx():
        topk_idx = torch.full((batch, index_topk), -1, dtype=torch.int32)
        for b in range(batch):
            topk_row = torch.randperm(max_seq_len, dtype=torch.int64)[:sparse_k].to(torch.int32)
            topk_idx[b, :sparse_k] = torch.sort(topk_row).values
        return topk_idx

    def init_seq_lens():
        return torch.full((batch,), sparse_k, dtype=torch.int32)

    layer_id_data = torch.tensor([0], dtype=torch.int32)

    def init_w_q_nope_to_latent():
        return (
            (torch.rand(num_heads, qk_nope_head_dim, kv_lora_rank, dtype=torch.float32) - 0.5)
            / (qk_nope_head_dim ** 0.5)
        )

    def init_w_latent_to_v():
        return (
            (torch.rand(num_heads, kv_lora_rank, v_head_dim, dtype=torch.float32) - 0.5)
            / (kv_lora_rank ** 0.5)
        )

    return [
        TensorSpec("q_proj", [batch, num_heads * qk_head_dim], torch.bfloat16, init_value=init_q_proj),
        TensorSpec("kv_cache", [cache_rows, kv_lora_rank], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("pe_cache", [cache_rows, qk_rope_head_dim], torch.bfloat16, init_value=init_pe_cache),
        TensorSpec("topk_idx", [batch, index_topk], torch.int32, init_value=init_topk_idx),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("layer_id_t", [1], torch.int32, init_value=layer_id_data),
        TensorSpec("w_q_nope_to_latent", [num_heads, qk_nope_head_dim, kv_lora_rank], torch.bfloat16, init_value=init_w_q_nope_to_latent),
        TensorSpec("w_latent_to_v", [num_heads, kv_lora_rank, v_head_dim], torch.bfloat16, init_value=init_w_latent_to_v),
        TensorSpec("dispatch_buf", [ep_nodes, batch, attn_out], torch.bfloat16, is_output=True),
    ]


def golden_decode_front_scope4(tensors):
    import torch

    q_proj = tensors["q_proj"].float()
    kv_cache = tensors["kv_cache"].float()
    pe_cache = tensors["pe_cache"].float()
    topk_idx = tensors["topk_idx"]
    seq_lens = tensors["seq_lens"]
    layer_id = int(tensors["layer_id_t"][0].item())
    w_q_nope_to_latent = tensors["w_q_nope_to_latent"].float()
    w_latent_to_v = tensors["w_latent_to_v"].float()
    dispatch_buf = tensors["dispatch_buf"]

    batch = q_proj.shape[0]
    num_heads = w_q_nope_to_latent.shape[0]
    kv_lora_rank = w_q_nope_to_latent.shape[2]
    qk_rope_head_dim = pe_cache.shape[1]
    qk_head_dim = q_proj.shape[1] // num_heads
    qk_nope_head_dim = qk_head_dim - qk_rope_head_dim
    v_head_dim = w_latent_to_v.shape[2]
    attn_out = num_heads * v_head_dim
    index_topk = topk_idx.shape[1]
    max_seq = kv_cache.shape[0] // batch
    ep_nodes = dispatch_buf.shape[0]
    attn_scale = 1.0 / (qk_head_dim ** 0.5)

    attn_front = torch.zeros(batch, attn_out, dtype=torch.float32)
    dispatch_buf.zero_()

    for b in range(batch):
        sparse_k = min(index_topk, int(seq_lens[b].item()))
        for h in range(num_heads):
            q_col = h * qk_head_dim
            q_nope = q_proj[b : b + 1, q_col : q_col + qk_nope_head_dim]
            q_pe = q_proj[b : b + 1, q_col + qk_nope_head_dim : q_col + qk_head_dim]
            q_nope_latent = q_nope @ w_q_nope_to_latent[h]

            oi = torch.zeros(1, kv_lora_rank, dtype=torch.float32)
            li = torch.zeros(1, 1, dtype=torch.float32)
            mi = torch.zeros(1, 1, dtype=torch.float32)

            for kk in range(sparse_k):
                topk_pos = int(topk_idx[b, kk].item())
                if topk_pos < 0:
                    continue
                cache_s = b * max_seq + topk_pos
                kv_s = kv_cache[cache_s : cache_s + 1]
                pe_s = pe_cache[cache_s : cache_s + 1]
                score_nope = (q_nope_latent * kv_s).sum(dim=-1, keepdim=True)
                score_pe = (q_pe * pe_s).sum(dim=-1, keepdim=True)
                cur_mi = (score_nope + score_pe) * attn_scale
                cur_li = torch.ones(1, 1, dtype=torch.float32)
                oi_tmp = kv_s * cur_li
                if kk == 0:
                    oi = oi_tmp
                    li = cur_li
                    mi = cur_mi
                else:
                    mi_new = torch.maximum(mi, cur_mi)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(cur_mi - mi_new)
                    li = alpha * li + beta * cur_li
                    oi = oi * alpha + oi_tmp * beta
                    mi = mi_new

            ctx_latent = oi / li.clamp_min(1e-30)
            ctx_v = ctx_latent @ w_latent_to_v[h]
            v_col = h * v_head_dim
            attn_front[b, v_col : v_col + v_head_dim] = ctx_v.squeeze(0)

    for b in range(batch):
        target_node = (b + layer_id) % ep_nodes
        dispatch_buf[target_node, b].copy_(attn_front[b].to(torch.bfloat16))


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--profile", type=str, default="full", choices=["reduced", "full"])
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    profile = REDUCED_PROFILE if args.profile == "reduced" else FULL_PROFILE

    program = build_deepseek_v3_2_decode_front_scope4_program(**profile)
    tensor_specs = build_tensor_specs(**profile)

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden_fn=golden_decode_front_scope4,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
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