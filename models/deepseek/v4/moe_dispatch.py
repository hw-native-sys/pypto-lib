# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed dispatch -- decode, single-card EP.

EP_WORLD_SIZE == 1 dispatch is a local regroup from token-major router outputs
to the per-local-expert layout consumed by ``moe_expert``.

    x_norm  [T, D]      bf16   FFN-normed hidden states   --+
    indices [T, TOPK]   int32  per-token expert ids         +-- == moe_router outputs
    weights [T, TOPK]   fp32   per-token routing weights  --+
        -> recv_x / recv_weights / recv_token / recv_expert_count
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
def moe_dispatch(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
):
    # recv_x is stored in moe_expert's 3-D layout. Metadata stays 1-D during
    # packed writes so scalar load/store lowering sees a bare flat index.
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_weights_flat = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.FP32)
    recv_token_flat = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX], dtype=pl.INT32)
    count_flat       = pl.reshape(recv_expert_count, [N_LOCAL_EXPERTS])
    indices_flat     = pl.reshape(indices, [T * TOPK])
    weights_flat     = pl.reshape(weights, [T * TOPK])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_dispatch"):
        for r0 in pl.range(0, N_LOCAL_EXPERTS * RECV_MAX, RECV_MAX):
            for d0 in pl.range(0, D, COL_CHUNK):
                recv_x_flat = pl.assemble(
                    recv_x_flat,
                    pl.full([RECV_MAX, COL_CHUNK], dtype=pl.BF16, value=0.0),
                    [r0, d0],
                )
        for r in pl.range(N_LOCAL_EXPERTS * RECV_MAX):
            pl.write(recv_weights_flat, [r], 0.0)
            pl.write(recv_token_flat, [r], pl.cast(0, pl.INT32))
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(count_flat, [e], pl.cast(0, pl.INT32))
        for t in pl.range(T):
            for k in pl.unroll(TOPK):
                p = t * TOPK + k
                e_global = pl.read(indices_flat, [p])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(count_flat, [e])
                dst = e * RECV_MAX + pl.cast(slot_i32, pl.INDEX)

                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_weights_flat, [dst], pl.read(weights_flat, [p]))
                pl.write(recv_token_flat, [dst], pl.cast(t, pl.INT32))
                pl.write(count_flat, [e], pl.cast(slot_i32 + 1, pl.INT32))

    for e in pl.unroll(N_LOCAL_EXPERTS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_materialize_metadata"):
            w_row_1d = pl.slice(recv_weights_flat, [RECV_MAX], [e * RECV_MAX])
            tok_row_1d = pl.slice(recv_token_flat, [RECV_MAX], [e * RECV_MAX])
            recv_weights = pl.assemble(recv_weights, pl.reshape(w_row_1d, [1, RECV_MAX]), [e, 0])
            recv_token = pl.assemble(recv_token, pl.reshape(tok_row_1d, [1, RECV_MAX]), [e, 0])


@pl.jit
def moe_dispatch_test(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
    recv_weights:      pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_token:        pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    moe_dispatch(
        x_norm, indices, weights,
        recv_x, recv_weights, recv_token, recv_expert_count,
    )
    return recv_x, recv_weights, recv_token, recv_expert_count


def golden_moe_dispatch(tensors):
    """Torch reference for the packed dispatch contract."""
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
            e = int(indices[t, k].item()) - EXPERTS_START_IDX
            s = cursor[e]
            assert 0 <= e < N_LOCAL_EXPERTS
            assert s < RECV_MAX, f"expert {e} received > RECV_MAX={RECV_MAX} rows"
            recv_x[e, s, :]    = x_norm[t, :]
            recv_weights[e, s] = float(weights[t, k].item())
            recv_token[e, s]   = t
            cursor[e] = s + 1

    recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    for e in range(N_LOCAL_EXPERTS):
        recv_count[e, 0] = cursor[e]

    tensors["recv_x"][:]            = recv_x
    tensors["recv_weights"][:]      = recv_weights
    tensors["recv_token"][:]        = recv_token
    tensors["recv_expert_count"][:] = recv_count


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
        TensorSpec("recv_x",            [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, is_output=True),
        TensorSpec("recv_weights",      [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_token",        [N_LOCAL_EXPERTS, RECV_MAX],    torch.int32,    is_output=True),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1],           torch.int32,    is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_dispatch_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_dispatch,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
