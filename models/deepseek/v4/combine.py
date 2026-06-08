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
import pypto.language.distributed as pld

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

# EP (cross-rank) layout — used only by combine_ep below, which moe_ep.py
# imports after overriding config to the 2-rank DEMO preset (harmless/uncompiled
# in the single-card moe.py process).
N_RANKS = EP_WORLD_SIZE
N_LOCAL = N_LOCAL_EXPERTS  # all-rank naming used by the EP kernel
N_ROUTES = T * TOPK


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
):
    # recv_y already has the per-row routing weight applied by expert_routed,
    # so combine is a pure scatter + dense reduce (sum, no second mul).

    # [T, N_LOCAL_EXPERTS, D] scratch indexed by (token, expert). Padding
    # (t, e) slots stay zero and contribute nothing to the dense reduce.
    routed_y_buf = pl.create_tensor([T, N_LOCAL_EXPERTS, D], dtype=pl.BF16)
    # Flat 2D view for slice ops: ptoas tensor.slice rejects 3D bases.
    routed_y_buf_flat = pl.reshape(routed_y_buf, [T * N_LOCAL_EXPERTS, D])
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_scatter"):
        for rb in pl.range(T * N_LOCAL_EXPERTS // 16):
            r0 = rb * 16
            routed_y_buf_flat[r0:r0 + 16, :] = pl.full([16, D], dtype=pl.BF16, value=0.0)

        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_expert_count, [e, 0]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [e, s]), pl.INDEX)
                dst = t * N_LOCAL_EXPERTS + e
                routed_y_buf_flat[dst:dst+1, :] = recv_y_flat[i:i+1, :]

    for tb in pl.spmd(T // 4, name_hint="combine_reduce"):
        for tt in pl.range(4):
            t = tb * 4 + tt
            base = t * N_LOCAL_EXPERTS
            acc = pl.cast(sh[t:t+1, :], target_type=pl.FP32)
            for e in pl.pipeline(N_LOCAL_EXPERTS, stage=2):
                src = base + e
                row_fp32 = pl.cast(routed_y_buf_flat[src:src+1, :], target_type=pl.FP32)
                acc = pl.add(acc, row_fp32)
            ffn_out[t:t+1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")


@pl.jit.incore
def combine_ep(
    recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]:
    # ---------- push: TPUT recv_y rows to peer's routed_y_buf ----------
    # For each dst rank and each local expert e on MY rank, the n rows I sent to
    # dst's expert e originally landed at slots [src_off, src_off + n) on dst
    # (src_off = Σ_{s<dst} counts). Per-row pl.load uses 3D indexing on recv_y
    # directly — DON'T reshape recv_y to 2D first, that would tile.load the full
    # [N_LOCAL, RECV_MAX, D] tensor into Vec (≫ 192 KB).
    for dst in pl.range(N_RANKS):
        for e in pl.range(N_LOCAL):
            n = pl.cast(pl.read(pub_counts, [dst * N_RANKS + my_rank, e]), pl.INDEX)
            src_off = pl.const(0, pl.INT32)
            for s in pl.range(N_RANKS):
                if s < dst:
                    src_off = src_off + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
            src_off_idx = pl.cast(src_off, pl.INDEX)
            for row in pl.range(n):
                slot = src_off_idx + row
                r_route = pl.read(recv_r_route_out, [e, slot])
                y_tile_3d = pl.load(recv_y, [e, slot, 0], [1, 1, D])
                y_tile = pl.reshape(y_tile_3d, [1, D])
                pld.tile.remote_store(
                    y_tile, target=routed_y_buf, peer=dst, offsets=[r_route, 0]
                )

    # ---------- combine_done barrier ----------
    # Single-writer per-src cell — Set, not Add.
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                target=combine_done,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.Set,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal=combine_done,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # ---------- reduce: ffn_out[t] = sh[t] + Σ_k routed_y_buf[t*TOPK+k] ----------
    # Kept in this InCore kernel (after the push + barrier), NOT split into a
    # separate parallel pl.spmd reducer: the orchestration wouldn't order a
    # separate routed_y_buf reader after the cross-rank push, since a
    # remote_store target isn't tracked as a window write — pypto#1670.
    # Per-row tile load + accumulate (token-major ffn_out [T, D]); the window is
    # read with pl.load (tensor.slice rejects a DistributedTensor — pypto#1672).
    for t in pl.range(T):
        sh_tile = pl.load(sh, [t, 0], [1, D])
        acc = pl.cast(sh_tile, target_type=pl.FP32)
        for k in pl.range(TOPK):
            y_k = pl.load(routed_y_buf, [t * TOPK + k, 0], [1, D])
            y_k_fp = pl.cast(y_k, target_type=pl.FP32)
            acc = pl.add(acc, y_k_fp)
        acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(acc_bf16, [t, 0], ffn_out)
    return ffn_out


@pl.jit
def combine_test(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    combine(recv_y, recv_token, recv_expert_count, sh, ffn_out)
    return ffn_out


def golden_combine(tensors):
    import torch

    recv_y = tensors["recv_y"]
    recv_token = tensors["recv_token"]
    recv_expert_count = tensors["recv_expert_count"]
    sh = tensors["sh"]

    # recv_y already carries the per-row routing weight applied by
    # expert_routed; combine just scatters by token and sums.
    routed_y_buf = torch.zeros(T, N_LOCAL_EXPERTS, D, dtype=torch.bfloat16)
    for e in range(N_LOCAL_EXPERTS):
        for s in range(int(recv_expert_count[e, 0].item())):
            t = int(recv_token[e, s].item())
            routed_y_buf[t, e, :] = recv_y[e, s, :]

    ffn_out = sh.float()
    for e in range(N_LOCAL_EXPERTS):
        ffn_out = ffn_out + routed_y_buf[:, e, :].float()
    tensors["ffn_out"][:] = ffn_out.to(torch.bfloat16)


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

    def init_recv_expert_count():
        return counts.reshape(N_LOCAL_EXPERTS, 1)

    def init_sh():
        return torch.randn(T, D)

    return [
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, init_value=init_recv_y),
        TensorSpec("recv_token", [N_LOCAL_EXPERTS, RECV_MAX], torch.int32, init_value=init_recv_token),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("sh", [T, D], torch.bfloat16, init_value=init_sh),
        TensorSpec("ffn_out", [T, D], torch.bfloat16, is_output=True),
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
