# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed local expert compute (decode, EP single-card).

Only the routed-expert path lives here. The shared expert was split out
into ``expert_shared.py``; both kernels are composed in ``moe.py``.
"""


import pypto.language as pl

from config import (ACTIVE as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP_WORLD_SIZE, RECV_MAX)
from routed_mxfp4_expert import (
    MX_K_TILE,
    MX_N_TILE,
    W13_SCALE_ROWS,
    W13_SCALE_ROWS_PER_EXPERT,
    W2_SCALE_ROWS,
    W2_SCALE_ROWS_PER_EXPERT,
    routed_mxfp4_expert,
)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE



@pl.jit.inline
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w1_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w3_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS * MOE_INTER, D], pl.FP4],
    routed_w2_scale: pl.Tensor[[W2_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    routed_mxfp4_expert(
        recv_x,
        recv_scale_dq,
        recv_weights,
        recv_expert_count,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        recv_y,
    )
    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w1_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS * D, MOE_INTER], pl.FP4],
    routed_w3_scale: pl.Tensor[[W13_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS * MOE_INTER, D], pl.FP4],
    routed_w2_scale: pl.Tensor[[W2_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale, recv_y,
    )
    return recv_y


def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    import torch
    import torch.nn.functional as F

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [E, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [E, 1] int32
    w1_fp4 = tensors["routed_w1"]
    w3_fp4 = tensors["routed_w3"]
    w2_fp4 = tensors["routed_w2"]

    from expert_shared import (
        _dequant_mxfp8_a,
        _dequant_mxfp8_b,
        _dynamic_mx_quant_e4m3,
        _unpack_b_scale_tiled,
    )

    def mx_matmul(a, b, b_scale):
        result = None
        for k0 in range(0, a.shape[-1], MX_K_TILE):
            a_q, a_scale = _dynamic_mx_quant_e4m3(
                a[:, k0 : k0 + MX_K_TILE]
            )
            partial = _dequant_mxfp8_a(a_q, a_scale) @ _dequant_mxfp8_b(
                b[k0 : k0 + MX_K_TILE],
                b_scale[
                    k0 // 32 : (k0 + MX_K_TILE) // 32,
                ],
            )
            result = partial if result is None else result + partial
        return result

    w1_scale_packed = tensors["routed_w1_scale"]
    w3_scale_packed = tensors["routed_w3_scale"]
    w2_scale_packed = tensors["routed_w2_scale"]

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = (
            recv_x_i8[e, :n_rows].float()
            * recv_scale_dq[e, :n_rows].reshape(-1, 1)
        )
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        w1 = _decode_fp4_weight(w1_fp4[e * D : (e + 1) * D]).to(torch.float8_e4m3fn)
        w3 = _decode_fp4_weight(w3_fp4[e * D : (e + 1) * D]).to(torch.float8_e4m3fn)
        w13_scale0 = e * W13_SCALE_ROWS_PER_EXPERT
        w1_scale = _unpack_b_scale_tiled(
            w1_scale_packed[
                w13_scale0 : w13_scale0 + W13_SCALE_ROWS_PER_EXPERT
            ],
            D,
            MOE_INTER,
            k_tile=MX_K_TILE,
            n_tile=MX_N_TILE,
        )
        w3_scale = _unpack_b_scale_tiled(
            w3_scale_packed[
                w13_scale0 : w13_scale0 + W13_SCALE_ROWS_PER_EXPERT
            ],
            D,
            MOE_INTER,
            k_tile=MX_K_TILE,
            n_tile=MX_N_TILE,
        )
        gate = mx_matmul(
            x_sub,
            w1,
            w1_scale,
        )
        up = mx_matmul(
            x_sub,
            w3,
            w3_scale,
        )
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        w2 = _decode_fp4_weight(
            w2_fp4[e * MOE_INTER : (e + 1) * MOE_INTER]
        ).to(torch.float8_e4m3fn)
        w2_scale0 = e * W2_SCALE_ROWS_PER_EXPERT
        w2_scale = _unpack_b_scale_tiled(
            w2_scale_packed[
                w2_scale0 : w2_scale0 + W2_SCALE_ROWS_PER_EXPERT
            ],
            MOE_INTER,
            D,
            k_tile=MX_K_TILE,
            n_tile=MX_N_TILE,
        )
        y = mx_matmul(
            h,
            w2,
            w2_scale,
        )
        recv_y[e, :n_rows] = y * w_per_row

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def _decode_fp4_weight(weight):
    """Decode a packed x2 E2M1 tensor to its logical FP32 matrix."""
    import torch

    packed = weight.contiguous().view(torch.uint8)
    codes = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.long
    )
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(
        [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ],
        dtype=torch.float32,
    )
    return values[codes]


def gen_routed_mxfp4_weight(experts, k, n, scale_code):
    """Generate packed E2M1 data plus one native MX_B_NN E8M0 scale tensor."""
    import torch
    from expert_shared import _pack_b_scale_tiled

    generator = torch.Generator().manual_seed(
        7301 + experts * 13 + k * 3 + n + scale_code
    )
    packed_experts = []
    packed_scales = []
    for _ in range(experts):
        codes = torch.randint(
            0, 16, (k, n), generator=generator, dtype=torch.uint8
        )
        packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
        packed_experts.append(packed)
        scale_codes = torch.full(
            (k // 32, n), scale_code, dtype=torch.uint8
        )
        packed_scales.append(
            _pack_b_scale_tiled(
                scale_codes.view(torch.float8_e8m0fnu),
                k_tile=MX_K_TILE,
                n_tile=MX_N_TILE,
            )
        )
    packed_weight = torch.cat(packed_experts, dim=0).contiguous()
    packed_scale = torch.cat(packed_scales, dim=0)
    return (
        packed_weight.view(torch.float4_e2m1fn_x2),
        packed_scale,
    )


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    torch.manual_seed(20260814)

    # Keep the standalone correctness fixture small enough for the scheduler,
    # while covering multiple experts so expert-local weight/scale offsets are
    # still exercised. Production counts remain fully dynamic kernel inputs.
    total = B * S * M.num_experts_per_tok
    fixture_experts = min(4, N_LOCAL_EXPERTS)
    counts = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
    counts[:fixture_experts] = total // fixture_experts
    counts[0] += total % fixture_experts
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    # Build a consistent INT8 recv_x + per-row dequant scale (dispatch is
    # responsible for per-token quantization). Invalid tail rows go to INT8 0
    # with scale 0 so dequant produces 0.
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    recv_x_i8_pre, recv_scale_dq_pre = _int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_pre = torch.where(
        valid_mask_2d,
        recv_scale_dq_pre.squeeze(-1),
        torch.zeros_like(recv_scale_dq_pre.squeeze(-1)),
    )

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) stay 0 so
    # they don't perturb the BF16 round-trip in expert_routed.
    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(
        valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre)
    )

    def init_recv_weights():
        return recv_weights_pre

    # Native routed weights: packed E2M1 data with per-32 E8M0 scales. Code 120
    # gives a dequantized standard deviation close to the model's ~2.5e-2 range.
    w1_fp4, w1_s = gen_routed_mxfp4_weight(N_LOCAL_EXPERTS, D, MOE_INTER, 120)
    w3_fp4, w3_s = gen_routed_mxfp4_weight(N_LOCAL_EXPERTS, D, MOE_INTER, 120)
    w2_fp4, w2_s = gen_routed_mxfp4_weight(N_LOCAL_EXPERTS, MOE_INTER, D, 120)

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS * D, MOE_INTER // 2], torch.float4_e2m1fn_x2, init_value=lambda: w1_fp4),
        TensorSpec("routed_w1_scale", [W13_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS * D, MOE_INTER // 2], torch.float4_e2m1fn_x2, init_value=lambda: w3_fp4),
        TensorSpec("routed_w3_scale", [W13_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS * MOE_INTER, D // 2], torch.float4_e2m1fn_x2, init_value=lambda: w2_fp4),
        TensorSpec("routed_w2_scale", [W2_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
