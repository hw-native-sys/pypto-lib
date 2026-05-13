# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE dispatch + (stand-in) expert + combine -- decode, single-card EP.

EP_WORLD_SIZE == 1 walk of the MoE block's dispatch/expert/combine triple (see
deepseek_v4_decode_single_layer.md): no AllToAllv, so dispatch/combine are pure
local regroups; the per-expert FFN is replaced by a routing-weight scale.

    x_norm  [T, D]      bf16   FFN-normed hidden states   --+
    indices [T, TOPK]   int32  per-token expert ids         +-- == moe_router outputs
    weights [T, TOPK]   fp32   per-token routing weights  --+
        -> routed_y [T, D] bf16   per-token combined expert output

recv_expert_count rides in/out as a zero-initialized output (it doubles as the
dispatch write cursor); the rest of the per-expert layout (recv_x / recv_weights
/ recv_token / recv_y) and the [token, expert] scatter target routed_y_buf are
kernel-internal scratch.
"""


import pypto.language as pl

from config import DEMO as M, DECODE_BATCH, DECODE_SEQ


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
N_EXPERTS = M.n_routed_experts

# EP layout / recv buffers
EP_WORLD_SIZE = 1   # demo 1; flash/pro depend on deployment (e.g. pro 16)
EP_RANK = 0
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS
RECV_MAX = 32       # per-(local-expert) row upper bound (must match moe_expert)

# tiling
COL_CHUNK = 512


@pl.jit.inline
def moe_dispatch_combine(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_y: pl.Tensor[[T, D], pl.BF16],
):
    # Scratch: recv_* hold moe_expert's per-(local-expert) layout collapsed to
    # one row per (expert, slot) pair at offset `e * RECV_MAX + slot`; the
    # per-pair scalar tables stay 1-D so pypto's scalar read/write codegen
    # accepts a bare index. routed_y_buf is the [token, expert] scatter target.
    recv_x       = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.BF16)
    recv_y       = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.BF16)
    recv_weights = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.FP32)
    recv_token   = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.INT32)
    routed_y_buf = pl.create_tensor([T * N_LOCAL_EXPERTS, D], dtype=pl.BF16)

    recv_weights_col = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    count_flat       = pl.reshape(recv_expert_count, [N_LOCAL_EXPERTS])
    indices_flat     = pl.reshape(indices, [T * TOPK])
    weights_flat     = pl.reshape(weights, [T * TOPK])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        # recv_expert_count arrives zero-initialized and doubles as the running
        # per-expert write cursor: bumped by one per appended row, ends holding
        # the valid count. Strictly sequential over the T*TOPK routed pairs in
        # (token, k) order so a torch reference iterating that order ties slots
        # identically.
        for p in pl.range(T * TOPK):
            t = p // TOPK
            e = pl.cast(pl.read(indices_flat, [p]), pl.INDEX)
            slot_i32 = pl.read(count_flat, [e])
            dst = e * RECV_MAX + pl.cast(slot_i32, pl.INDEX)

            recv_x = pl.assemble(recv_x, pl.slice(x_norm, [1, D], [t, 0]), [dst, 0])
            pl.write(recv_weights, [dst], pl.read(weights_flat, [p]))
            pl.write(recv_token, [dst], pl.cast(t, pl.INT32))
            pl.write(count_flat, [e], pl.cast(slot_i32 + 1, pl.INT32))

    # FFN stand-in: recv_y[e, s, :] = recv_x[e, s, :] * recv_weights[e, s]. Tail
    # rows (slot >= count[e]) are never read by combine, so left as is. Loops
    # inside the pl.at (-> pl.range, not pl.parallel) so this is one task reading
    # recv_x rather than many tasks each carrying that dependency.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_expert"):
        for r0 in pl.range(0, N_LOCAL_EXPERTS * RECV_MAX, RECV_MAX):
            for d0 in pl.range(0, D, COL_CHUNK):
                x_col = pl.cast(recv_x[r0 : r0 + RECV_MAX, d0 : d0 + COL_CHUNK], target_type=pl.FP32)
                y_col = pl.row_expand_mul(x_col, recv_weights_col[r0 : r0 + RECV_MAX, :])
                # mode="rint" = round half to even, matching torch's `.to(bfloat16)`.
                recv_y[r0 : r0 + RECV_MAX, d0 : d0 + COL_CHUNK] = pl.cast(y_col, target_type=pl.BF16, mode="rint")

    # routed_y_buf must start zero so experts a token did not pick add nothing.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="routed_y_buf_init"):
        for r0 in pl.range(0, T * N_LOCAL_EXPERTS, N_LOCAL_EXPERTS):
            for d0 in pl.range(0, D, COL_CHUNK):
                routed_y_buf[r0 : r0 + N_LOCAL_EXPERTS, d0 : d0 + COL_CHUNK] = pl.full(
                    [N_LOCAL_EXPERTS, COL_CHUNK], dtype=pl.BF16, value=0.0
                )

    # Combine: scatter every valid recv_y row to routed_y_buf[t, e, :] (no
    # collisions: a token lands at most once per expert), then reduce over the
    # expert axis -> routed_y[t, :], casting each bf16 row to fp32 and adding in
    # expert order so the result matches torch bit-for-bit.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="combine"):
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(count_flat, [e]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token, [i]), pl.INDEX)
                routed_y_buf = pl.assemble(
                    routed_y_buf, pl.slice(recv_y, [1, D], [i, 0]), [t * N_LOCAL_EXPERTS + e, 0]
                )
        for t in pl.range(T):
            base = t * N_LOCAL_EXPERTS
            for d0 in pl.range(0, D, COL_CHUNK):
                acc = pl.cast(routed_y_buf[base : base + 1, d0 : d0 + COL_CHUNK], target_type=pl.FP32)
                for e in pl.range(1, N_LOCAL_EXPERTS):
                    row = pl.cast(routed_y_buf[base + e : base + e + 1, d0 : d0 + COL_CHUNK], target_type=pl.FP32)
                    acc = pl.add(acc, row)
                routed_y[t : t + 1, d0 : d0 + COL_CHUNK] = pl.cast(acc, target_type=pl.BF16, mode="rint")


@pl.jit
def moe_dispatch_combine_test(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
    routed_y: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    moe_dispatch_combine(x_norm, indices, weights, recv_expert_count, routed_y)
    return routed_y


def golden_moe_dispatch_combine(tensors):
    """Torch reference: regroup the per-token top-k assignments by expert id
    (iterating (token, k) in kernel order so slot assignments match), scale each
    grouped row by its routing weight, then scatter back per token and reduce
    over the expert axis. The reduction casts each bf16 row to fp32 and adds in
    expert order (not torch's `sum`), matching the kernel bit-for-bit."""
    import torch

    x_norm  = tensors["x_norm"]
    indices = tensors["indices"]   # [T, TOPK] int32
    weights = tensors["weights"]   # [T, TOPK] fp32

    recv_x       = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token   = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    cursor = [0] * N_LOCAL_EXPERTS
    for t in range(T):
        for k in range(TOPK):
            e = int(indices[t, k].item())
            s = cursor[e]
            assert s < RECV_MAX, f"expert {e} received > RECV_MAX={RECV_MAX} rows"
            recv_x[e, s, :]    = x_norm[t, :]
            recv_weights[e, s] = float(weights[t, k].item())
            recv_token[e, s]   = t
            cursor[e] = s + 1

    recv_y = (recv_x.float() * recv_weights.unsqueeze(-1)).to(torch.bfloat16)

    routed_y_buf = torch.zeros(T, N_LOCAL_EXPERTS, D, dtype=torch.bfloat16)
    for e in range(N_LOCAL_EXPERTS):
        for s in range(cursor[e]):
            t = int(recv_token[e, s].item())
            routed_y_buf[t, e, :] = recv_y[e, s, :]
    routed_y_acc = routed_y_buf[:, 0, :].float()
    for e in range(1, N_LOCAL_EXPERTS):
        routed_y_acc = routed_y_acc + routed_y_buf[:, e, :].float()

    recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    for e in range(N_LOCAL_EXPERTS):
        recv_count[e, 0] = cursor[e]
    tensors["recv_expert_count"][:] = recv_count
    tensors["routed_y"][:]          = routed_y_acc.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_norm():
        return torch.randn(T, D) * 0.1

    def init_indices():
        # Mirror the router: each token picks TOPK *distinct* experts.
        rows = [torch.randperm(N_EXPERTS)[:TOPK] for _ in range(T)]
        return torch.stack(rows).to(torch.int32)

    def init_weights():
        # Positive, row-normalized (ROUTE_SCALE == 1.0 on the flash demo).
        w = torch.rand(T, TOPK) + 0.1
        return (w / w.sum(dim=-1, keepdim=True)).float()

    return [
        TensorSpec("x_norm",  [T, D],    torch.bfloat16, init_value=init_x_norm),
        TensorSpec("indices", [T, TOPK], torch.int32,    init_value=init_indices),
        TensorSpec("weights", [T, TOPK], torch.float32,  init_value=init_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32,    is_output=True),
        TensorSpec("routed_y",          [T, D],            torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_dispatch_combine_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_dispatch_combine,
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
