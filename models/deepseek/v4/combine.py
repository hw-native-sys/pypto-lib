# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed combine -- decode, single-card EP.

    recv_y / recv_token / recv_weights / recv_expert_count + sh -> ffn_out
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[B, S, D], pl.BF16],
):
    ffn_out_flat = pl.reshape(ffn_out, [T, D])

    # [T, N_LOCAL_EXPERTS, D] scratch indexed by (token, expert). Padding
    # (t, e) slots stay zero and contribute nothing to the dense reduce.
    routed_y_buf = pl.create_tensor([T, N_LOCAL_EXPERTS, D], dtype=pl.BF16)
    routed_w_buf = pl.create_tensor([T, N_LOCAL_EXPERTS], dtype=pl.FP32)
    # Flat 2D views for slice ops: ptoas tensor.slice rejects 3D bases.
    routed_y_buf_flat = pl.reshape(routed_y_buf, [T * N_LOCAL_EXPERTS, D])
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_scatter"):
        # Init via pl.write to keep routed_w_buf single-access-style.
        for r0 in pl.range(T):
            routed_y_buf[r0, :, :] = pl.full([N_LOCAL_EXPERTS, D], dtype=pl.BF16, value=0.0)
            for e0 in pl.range(N_LOCAL_EXPERTS):
                pl.write(routed_w_buf, [r0, e0], 0.0)

        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_expert_count, [e, 0]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [e, s]), pl.INDEX)
                dst = t * N_LOCAL_EXPERTS + e
                routed_y_buf_flat[dst:dst+1, :] = recv_y_flat[i:i+1, :]
                pl.write(routed_w_buf, [t, e], pl.read(recv_weights, [e, s]))

    for tb in pl.spmd(T // 4, name_hint="combine_reduce"):
        for tt in pl.range(4):
            t = tb * 4 + tt
            base = t * N_LOCAL_EXPERTS
            acc = pl.cast(sh[t:t+1, :], target_type=pl.FP32)
            for e in pl.pipeline(N_LOCAL_EXPERTS, stage=2):
                src = base + e
                row_fp32 = pl.cast(routed_y_buf_flat[src:src+1, :], target_type=pl.FP32)
                w = pl.read(routed_w_buf, [t, e])
                acc = pl.add(acc, pl.mul(row_fp32, w))
            ffn_out_flat[t:t+1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")


@pl.jit
def combine_test(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    combine(recv_y, recv_token, recv_weights, recv_expert_count, sh, ffn_out)
    return ffn_out


def golden_combine(tensors):
    import torch

    recv_y = tensors["recv_y"]
    recv_token = tensors["recv_token"]
    recv_weights = tensors["recv_weights"]
    recv_expert_count = tensors["recv_expert_count"]
    sh = tensors["sh"]

    routed_y_buf = torch.zeros(T, N_LOCAL_EXPERTS, D, dtype=torch.bfloat16)
    routed_w_buf = torch.zeros(T, N_LOCAL_EXPERTS, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        for s in range(int(recv_expert_count[e, 0].item())):
            t = int(recv_token[e, s].item())
            routed_y_buf[t, e, :] = recv_y[e, s, :]
            routed_w_buf[t, e] = float(recv_weights[e, s].item())

    ffn_out = sh.float()
    for e in range(N_LOCAL_EXPERTS):
        ffn_out = ffn_out + routed_y_buf[:, e, :].float() * routed_w_buf[:, e : e + 1]
    tensors["ffn_out"][:] = ffn_out.to(torch.bfloat16).reshape(B, S, D)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    # Simulate dispatch routing: each of T tokens routes to topk distinct
    # local experts. counts[e] = #tokens routed to e (sum = T*topk before
    # RECV_MAX clamp); recv_token[e, :counts[e]] is the matching token list
    # with no per-expert duplicates. Mirrors expert_routed.build_tensor_specs.
    topk = min(M.num_experts_per_tok, N_LOCAL_EXPERTS)
    gen = torch.Generator().manual_seed(0)
    routing = torch.stack(
        [torch.randperm(N_LOCAL_EXPERTS, generator=gen)[:topk] for _ in range(T)]
    )

    per_expert_tokens = [[] for _ in range(N_LOCAL_EXPERTS)]
    for t in range(T):
        for k in range(topk):
            per_expert_tokens[int(routing[t, k].item())].append(t)
    counts = torch.tensor(
        [min(len(toks), RECV_MAX) for toks in per_expert_tokens],
        dtype=torch.int32,
    )

    def init_recv_y():
        return torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D)

    def init_recv_token():
        recv_token = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
        for e in range(N_LOCAL_EXPERTS):
            n = int(counts[e].item())
            if n > 0:
                recv_token[e, :n] = torch.tensor(
                    per_expert_tokens[e][:n], dtype=torch.int32
                )
        return recv_token

    def init_recv_weights():
        # Tail slots poisoned with large garbage to verify the kernel reads
        # only s < recv_expert_count[e].
        mean_w = M.routed_scaling_factor / M.num_experts_per_tok
        w = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
        for e in range(N_LOCAL_EXPERTS):
            count = int(counts[e].item())
            if count > 0:
                w[e, :count] = (torch.rand(count) * mean_w + mean_w * 0.5).float()
            if count < RECV_MAX:
                w[e, count:] = (torch.randn(RECV_MAX - count) * 1e3).float()
        return w

    def init_recv_expert_count():
        return counts.reshape(N_LOCAL_EXPERTS, 1)

    def init_sh():
        return torch.randn(T, D)

    return [
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, init_value=init_recv_y),
        TensorSpec("recv_token", [N_LOCAL_EXPERTS, RECV_MAX], torch.int32, init_value=init_recv_token),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("sh", [T, D], torch.bfloat16, init_value=init_sh),
        TensorSpec("ffn_out", [B, S, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=combine_test,
        specs=build_tensor_specs(),
        golden_fn=golden_combine,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
