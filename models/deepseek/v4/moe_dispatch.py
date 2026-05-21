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
to the per-local-expert layout consumed by ``moe_expert``. Per-token INT8
quantization of ``x_norm`` happens once here; the dequant scale travels
through to ``moe_expert`` so the routed path does not re-quantize per tile.

    x_norm  [T, D]      bf16   FFN-normed hidden states   --+
    indices [T, TOPK]   int32  per-token expert ids         +-- == moe_router outputs
    weights [T, TOPK]   fp32   per-token routing weights  --+
        -> recv_x (INT8) / recv_scale_dq (per-token dequant scale) /
           recv_weights / recv_token / recv_expert_count
"""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP_WORLD_SIZE, EP_RANK, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

# tiling
COL_CHUNK = 64 if T >= 128 else (128 if T >= 64 else 512)
QUANT_CHUNK = 32 if T >= 128 else (128 if T >= 64 else 256)  # column chunk for two-pass per-token INT8 quant


@pl.jit.inline
def moe_dispatch(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq:     pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
):
    # Stage 0: per-token symmetric INT8 quant of x_norm. One amax per token,
    # reused across all TOPK destinations and consumed by moe_expert without
    # re-quant.
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale_dq_buf = pl.create_tensor([T, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_norm_q"):
        xn_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, D, QUANT_CHUNK):
            xn_a_f32 = pl.cast(x_norm[:, k0 : k0 + QUANT_CHUNK], target_type=pl.FP32)
            xn_a_abs = pl.maximum(xn_a_f32, pl.neg(xn_a_f32))
            xn_a_max = pl.reshape(pl.row_max(xn_a_abs), [1, T])
            xn_amax = pl.maximum(xn_amax, xn_a_max)
        xn_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), xn_amax)
        x_norm_scale_dq_buf[:, 0:1] = pl.reshape(pl.recip(xn_sq_row), [T, 1])
        xn_sq_col = pl.reshape(xn_sq_row, [T, 1])
        for k1 in pl.range(0, D, QUANT_CHUNK):
            xn_q_f32 = pl.cast(x_norm[:, k1 : k1 + QUANT_CHUNK], target_type=pl.FP32)
            xn_q_scaled = pl.row_expand_mul(xn_q_f32, xn_sq_col)
            xn_q_i32 = pl.cast(xn_q_scaled, target_type=pl.INT32, mode="rint")
            xn_q_half = pl.cast(xn_q_i32, target_type=pl.FP16, mode="round")
            x_norm_i8[:, k1 : k1 + QUANT_CHUNK] = pl.cast(xn_q_half, target_type=pl.INT8, mode="trunc")

    # recv_x still uses a flat row view because its destination row is
    # data-dependent. Small route metadata uses the natural 2-D layout.
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="packed_dispatch"):
        # recv_x tail rows (slot >= recv_expert_count[e]) intentionally left
        # uninitialized — the matching recv_scale_dq slot is 0, so garbage INT8
        # rows neutralize to 0.0 after dequant. (Skipping the i8 zero-fill also
        # works around `pto.texpands` not supporting i8 broadcast.)
        for e in pl.range(N_LOCAL_EXPERTS):
            for s in pl.range(RECV_MAX):
                pl.write(recv_scale_dq, [e, s], 0.0)
                pl.write(recv_weights, [e, s], 0.0)
                pl.write(recv_token, [e, s], pl.cast(0, pl.INT32))
            # This read-reset is an ordering edge, not dead arithmetic. It
            # makes packed_dispatch consume the route_count materialization
            # before reusing recv_expert_count as the dynamic cursor.
            dep_count_i32 = pl.read(recv_expert_count, [e, 0])
            dep_zero_i32 = pl.cast(dep_count_i32 - dep_count_i32, pl.INT32)
            pl.write(recv_expert_count, [e, 0], dep_zero_i32)

        for t in pl.range(T):
            for k in pl.unroll(TOPK):
                e_global = pl.read(indices, [t, k])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(recv_expert_count, [e, 0])
                slot = pl.cast(slot_i32, pl.INDEX)
                dst = e * RECV_MAX + slot

                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm_i8, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_scale_dq, [e, slot], pl.read(x_norm_scale_dq_buf, [t, 0]))
                pl.write(recv_weights, [e, slot], pl.read(weights, [t, k]))
                pl.write(recv_token, [e, slot], pl.cast(t, pl.INT32))
                pl.write(recv_expert_count, [e, 0], pl.cast(slot_i32 + 1, pl.INT32))


@pl.jit
def moe_dispatch_test(
    x_norm:  pl.Tensor[[T, D],    pl.BF16],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]],
    recv_scale_dq:     pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_weights:      pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_token:        pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    moe_dispatch(
        x_norm, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count,
    )
    return recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count


# Module-level constants pulled in from config so golden mirrors the kernel quant.
from config import INT8_AMAX_EPS as _INT8_AMAX_EPS, INT8_SCALE_MAX as _INT8_SCALE_MAX


def _per_token_int8_quant(x_bf16):
    """Per-token symmetric INT8 quant matching the kernel's cast chain.

    Kernel: FP32(scaled) -> INT32(rint) -> FP16(round) -> INT8(trunc).
    Torch reproduction uses ``torch.round`` (half-to-even) explicitly so the
    INT32 stage is the rounded value, not truncated toward zero.
    """
    import torch
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(_INT8_AMAX_EPS)
    scale_q = _INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)  # [T]
    return x_i8, scale_dq


def golden_moe_dispatch(tensors):
    """Torch reference for the packed dispatch contract."""
    import torch

    x_norm  = tensors["x_norm"]
    indices = tensors["indices"]   # [T, TOPK] int32
    weights = tensors["weights"]   # [T, TOPK] fp32

    x_norm_i8, x_norm_scale_dq = _per_token_int8_quant(x_norm)

    recv_x        = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.int8)
    recv_scale_dq = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights  = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token    = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    cursor = [0] * N_LOCAL_EXPERTS
    for t in range(T):
        for k in range(TOPK):
            e = int(indices[t, k].item()) - EXPERTS_START_IDX
            s = cursor[e]
            assert 0 <= e < N_LOCAL_EXPERTS
            assert s < RECV_MAX, f"expert {e} received > RECV_MAX={RECV_MAX} rows"
            recv_x[e, s, :]      = x_norm_i8[t, :]
            recv_scale_dq[e, s]  = float(x_norm_scale_dq[t].item())
            recv_weights[e, s]   = float(weights[t, k].item())
            recv_token[e, s]     = t
            cursor[e] = s + 1

    recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    for e in range(N_LOCAL_EXPERTS):
        recv_count[e, 0] = cursor[e]

    tensors["recv_x"][:]            = recv_x
    tensors["recv_scale_dq"][:]     = recv_scale_dq
    tensors["recv_weights"][:]      = recv_weights
    tensors["recv_token"][:]        = recv_token
    tensors["recv_expert_count"][:] = recv_count


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_norm():
        return torch.randn(T, D)

    def init_indices():
        # Each token picks TOPK distinct experts.
        rows = [torch.randperm(N_LOCAL_EXPERTS)[:TOPK] for _ in range(T)]
        return torch.stack(rows).to(torch.int32)

    def init_weights():
        # Per-row weights normalized to sum=routed_scaling_factor.
        w = torch.rand(T, TOPK) + 0.1
        w = w / w.sum(dim=-1, keepdim=True) * M.routed_scaling_factor
        return w.float()

    return [
        TensorSpec("x_norm",  [T, D],    torch.bfloat16, init_value=init_x_norm),
        TensorSpec("indices", [T, TOPK], torch.int32,    init_value=init_indices),
        TensorSpec("weights", [T, TOPK], torch.float32,  init_value=init_weights),
        TensorSpec("recv_x",            [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8,     is_output=True),
        TensorSpec("recv_scale_dq",     [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_weights",      [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_token",        [N_LOCAL_EXPERTS, RECV_MAX],    torch.int32,    is_output=True),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1],           torch.int32,    is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_dispatch_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_dispatch,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # ULP-level FP32 mul drift flips rint at k.5 boundaries → INT8
            # off-by-one with same dequant magnitude. Allow ≤0.1% bad.
            "recv_x": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
