# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed combine -- decode, single-card EP.

Combines local expert rows from ``moe_expert`` back to token-major FFN output;
the routed contribution is FP32-weighted by ``recv_weights``.

    recv_y / recv_token / recv_weights / recv_expert_count + sh -> ffn_out
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
N_EXPERTS = M.n_routed_experts

# EP layout / recv buffers
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE

# tiling
COL_CHUNK = 512


@pl.jit.inline
def moe_combine(
    recv_y:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
    sh:                pl.Tensor[[T, D],                         pl.BF16],
    # ``ffn_out`` is [B, S, D] so the immediate consumer (``hc_post``'s ``x``
    # input) can use the buffer as-is. The body reshapes to a [T, D] view
    # before any kernel scope so the inner loop indexes flat.
    ffn_out:           pl.Tensor[[B, S, D],                      pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_token_flat = pl.reshape(recv_token, [N_LOCAL_EXPERTS * RECV_MAX])
    recv_weights_flat = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX])
    count_flat = pl.reshape(recv_expert_count, [N_LOCAL_EXPERTS])
    routed_y_buf = pl.create_tensor([T * N_LOCAL_EXPERTS, D], dtype=pl.BF16)
    # Zero entries contribute nothing so accumulation can run dense.
    routed_w_buf = pl.create_tensor([T * N_LOCAL_EXPERTS], dtype=pl.FP32)
    ffn_out_flat = pl.reshape(ffn_out, [T, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_combine_init"):
        for r0 in pl.range(0, T * N_LOCAL_EXPERTS, N_LOCAL_EXPERTS):
            for d0 in pl.range(0, D, COL_CHUNK):
                routed_y_buf[r0 : r0 + N_LOCAL_EXPERTS, d0 : d0 + COL_CHUNK] = pl.full(
                    [N_LOCAL_EXPERTS, COL_CHUNK], dtype=pl.BF16, value=0.0
                )
        for r in pl.range(T * N_LOCAL_EXPERTS):
            pl.write(routed_w_buf, [r], 0.0)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_combine"):
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(count_flat, [e]), pl.INDEX)
            for s in pl.range(n_rows):
                i = e * RECV_MAX + s
                t = pl.cast(pl.read(recv_token_flat, [i]), pl.INDEX)
                dst = t * N_LOCAL_EXPERTS + e
                routed_y_buf = pl.assemble(
                    routed_y_buf,
                    pl.slice(recv_y_flat, [1, D], [i, 0]),
                    [dst, 0],
                )
                pl.write(routed_w_buf, [dst], pl.read(recv_weights_flat, [i]))
        for t in pl.range(T):
            base = t * N_LOCAL_EXPERTS
            for d0 in pl.range(0, D, COL_CHUNK):
                acc = pl.cast(sh[t : t + 1, d0 : d0 + COL_CHUNK], target_type=pl.FP32)
                for e in pl.range(N_LOCAL_EXPERTS):
                    row = pl.cast(routed_y_buf[base + e : base + e + 1, d0 : d0 + COL_CHUNK], target_type=pl.FP32)
                    w = pl.read(routed_w_buf, [base + e])
                    acc = pl.add(acc, pl.mul(row, w))
                ffn_out_flat[t : t + 1, d0 : d0 + COL_CHUNK] = pl.cast(acc, target_type=pl.BF16, mode="rint")


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

    counts = torch.tensor(
        [(e * 3 + 1) % (T + 1) for e in range(N_LOCAL_EXPERTS)],
        dtype=torch.int32,
    )
    counts[0] = 0
    counts[-1] = min(T, RECV_MAX)

    def init_recv_y():
        return torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D)

    def init_recv_token():
        recv_token = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
        base = torch.arange(RECV_MAX, dtype=torch.int32)
        for e in range(N_LOCAL_EXPERTS):
            count = int(counts[e].item())
            recv_token[e, :count] = (base[:count] * 5 + e) % T
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
    args = parser.parse_args()

    result = run_jit(
        fn=moe_combine_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_combine,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
