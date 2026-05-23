# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed combine -- decode, single-card EP.

Two-phase structure mirrors runtime/examples/workers/l3/ep_dispatch_combine
combine.cpp so multi-rank port only needs to replace Phase-1 local writes
with cross-rank TPUT:
  * Phase 1 (sequential pl.at): scatter recv_y rows into [T*TOPK, D] scratch.
  * Phase 2 (pl.parallel(T)): sum TOPK weighted rows on top of sh, cast to BF16.
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

COL_CHUNK = 512


@pl.jit.inline
def moe_combine(
    recv_y:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
    sh:                pl.Tensor[[T, D],                         pl.BF16],
    ffn_out:           pl.Tensor[[B, S, D],                      pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_token_flat = pl.reshape(recv_token, [N_LOCAL_EXPERTS * RECV_MAX])
    recv_weights_flat = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX])
    ffn_out_flat = pl.reshape(ffn_out, [T, D])

    # Matches combine.cpp routed_y_buf [T, TOPK, D]. Multi-rank derives the
    # k-within-t slot via dispatch's recv_idx_out; single-card reconstructs
    # it by counting on the fly with slot_count.
    routed_y_buf = pl.create_tensor([T * TOPK, D], dtype=pl.BF16)
    routed_w_buf = pl.create_tensor([T * TOPK], dtype=pl.FP32)
    slot_count = pl.create_tensor([T], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_scatter"):
        # routed_w_buf / slot_count use pl.write loops for init because their
        # scatter loop also uses pl.write — mixing pl.full slice-assign with
        # pl.read/pl.write on one tensor breaks SSA typing in ptoas.
        for r0 in pl.range(0, T * TOPK, TOPK):
            routed_y_buf[r0:r0+TOPK, :] = pl.full([TOPK, D], dtype=pl.BF16, value=0.0)
        for r in pl.range(T * TOPK):
            pl.write(routed_w_buf, [r], 0.0)
        for t0 in pl.range(T):
            pl.write(slot_count, [t0], pl.cast(0, pl.INT32))

        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_expert_count, [e, 0]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token_flat, [i]), pl.INDEX)
                k_i32 = pl.read(slot_count, [t])
                pl.write(slot_count, [t], pl.cast(k_i32 + 1, pl.INT32))
                dst = t * TOPK + pl.cast(k_i32, pl.INDEX)
                routed_y_buf = pl.assemble(
                    routed_y_buf,
                    pl.slice(recv_y_flat, [1, D], [i, 0]),
                    [dst, 0],
                )
                pl.write(routed_w_buf, [dst], pl.read(recv_weights_flat, [i]))

    for t in pl.parallel(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_reduce"):
            base = t * TOPK
            for d0 in pl.range(0, D, COL_CHUNK):
                acc = pl.cast(sh[t:t+1, d0:d0+COL_CHUNK], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    src = base + k
                    row_fp32 = pl.cast(
                        routed_y_buf[src:src+1, d0:d0+COL_CHUNK], target_type=pl.FP32
                    )
                    w = pl.read(routed_w_buf, [src])
                    acc = pl.add(acc, pl.mul(row_fp32, w))
                ffn_out_flat[t:t+1, d0:d0+COL_CHUNK] = pl.cast(
                    acc, target_type=pl.BF16, mode="rint"
                )


@pl.jit
def moe_combine_test(
    recv_y:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
    sh:                pl.Tensor[[T, D],                         pl.BF16],
    ffn_out:           pl.Out[pl.Tensor[[B, S, D],               pl.BF16]],
):
    moe_combine(recv_y, recv_token, recv_weights, recv_expert_count, sh, ffn_out)
    return ffn_out


def golden_moe_combine(tensors):
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
    # with no per-expert duplicates. Mirrors moe_expert.build_tensor_specs.
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
        TensorSpec("recv_y",            [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, init_value=init_recv_y),
        TensorSpec("recv_token",        [N_LOCAL_EXPERTS, RECV_MAX],    torch.int32,    init_value=init_recv_token),
        TensorSpec("recv_weights",      [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1],           torch.int32,    init_value=init_recv_expert_count),
        TensorSpec("sh",                [T, D],                         torch.bfloat16, init_value=init_sh),
        TensorSpec("ffn_out",           [B, S, D],                      torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_combine_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_combine,
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
