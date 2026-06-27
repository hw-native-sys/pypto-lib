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


@pl.jit.inline
def combine_ep1(
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


@pl.jit.inline
def combine(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_r_route_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    pub_counts: pld.DistributedTensor[[EP_WORLD_SIZE * EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[T * TOPK, D], pl.BF16],
    combine_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based ordinal of this MoE call. combine_done is a monotonic per-source
    # counter, so waiting for `moe_epoch` isolates serial MoE calls that reuse
    # the same distributed window.
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Push recv_y rows to each peer's routed_y_buf, then a cross-rank barrier, in
    # one pl.at(CORE_GROUP) so the TPUT + notify/wait stay one atomic task. The
    # flat 2-D view is only a put src (GM-to-peer-GM); needs pypto#1732 (put dst
    # window declared add_output) for the RAW edge into combine_reduce.
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine_push"):
        # Each (dst, e) block of n rows landed at slots [src_off, src_off+n) on
        # dst.
        for dst in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                n = pl.cast(pl.read(pub_counts, [dst * EP_WORLD_SIZE + my_rank, e]), pl.INDEX)
                src_off = pl.const(0, pl.INT32)
                for s in pl.range(EP_WORLD_SIZE):
                    if s < dst:
                        src_off = src_off + pl.read(pub_counts, [s * EP_WORLD_SIZE + my_rank, e])
                src_off_idx = pl.cast(src_off, pl.INDEX)
                for row in pl.range(n):
                    slot = src_off_idx + row
                    r_route = pl.cast(pl.read(recv_r_route_out, [e, slot]), pl.INDEX)
                    pld.tensor.put(
                        dst=routed_y_buf,
                        peer=dst,
                        src=recv_y_flat,
                        dst_offsets=[r_route, 0],
                        src_offsets=[e * RECV_MAX + slot, 0],
                        shape=[1, D],
                    )

        # combine_done barrier — per-src signal cells. Each MoE increments once;
        # receivers wait for this call's epoch instead of a fixed value, so a
        # completed earlier combine cannot release the current reduce.
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=combine_done,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=combine_done,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

    # reduce: ffn_out[t] = sh[t] + Σ_k routed_y_buf[t*TOPK+k]. Separate pl.spmd
    # scope (ordered after the push via the window write->read dep).
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    for tb in pl.spmd(T // 4, name_hint="combine_reduce"):
        for tt in pl.range(4):
            t = tb * 4 + tt
            if t < active_tokens:
                acc = pl.cast(sh[t:t + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    r = t * TOPK + k
                    acc = pl.add(acc, pl.cast(routed_y_buf[r:r + 1, :], target_type=pl.FP32))
                ffn_out[t:t + 1, :] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            else:
                ffn_out[t:t + 1, :] = sh[t:t + 1, :]


@pl.jit
def combine_test(
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    sh: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    combine_ep1(recv_y, recv_token, recv_expert_count, sh, ffn_out)
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
    routing = torch.stack(
        [torch.randperm(N_LOCAL_EXPERTS)[:topk] for _ in range(T)]
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
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=combine_test,
        specs=build_tensor_specs(),
        golden_fn=golden_combine,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
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
