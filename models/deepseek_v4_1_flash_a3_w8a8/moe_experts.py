# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 routed and shared experts with the V4.1 SwiGLU limit."""

import pypto.language as pl
import torch

from config import FLASH as M
from gemm_w8a8 import w8a8_gemm
from quant import quantize_int8_per_token
from utils import int8_quant_per_token, unpack_nz


# Dynamic shape variable.

# model config
D = M.hidden_size
INTER = M.moe_intermediate_size
LIMIT = M.swiglu_limit
ACT_TILE = 384


@pl.jit.inline(auto_scope=False)
def expert_routed(
    recv_x: pl.Tensor[[16, D], pl.BF16],
    routed_weight: pl.Tensor[[16, 1], pl.FP32],
    w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[INTER], pl.FP32],
    w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[INTER], pl.FP32],
    w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[16, D], pl.BF16],
    rows: pl.constexpr,
):
    """Apply one routed expert: W8A8 gate/up projections, SwiGLU, and the down projection."""
    x_i8 = pl.create_tensor([rows, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([rows, 1], dtype=pl.FP32)
    quantize_int8_per_token(recv_x, x_i8, x_scale, width=D)
    gate = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    up = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    w8a8_gemm(x_i8, x_scale, w1, w1_scale, gate, width=D, output_width=INTER, output_fp32=False)
    w8a8_gemm(x_i8, x_scale, w3, w3_scale, up, width=D, output_width=INTER, output_fp32=False)
    activated = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swiglu"):
        # Scalar-load the per-route weight into a DN tile: taking the tensor slice
        # directly makes the onboard backend emit TLOAD ND->DN (col-major), which
        # the a2a3 path rejects (only ND2ND/DN2DN/NZ2NZ). The block then works on
        # tiles so the weight and value operands stay at the same DSL level.
        weight = pl.tile.create([rows, 1], dtype=pl.FP32)
        for weight_row in pl.range(rows):
            pl.tile.write(weight, [weight_row, 0], pl.read(routed_weight, [weight_row, 0]))
        weight = pl.set_validshape(weight, rows, 1)
        for c0 in pl.pipeline(0, INTER, ACT_TILE, stage=2):
            gate_tile = pl.cast(pl.load(gate, [0, c0], [rows, ACT_TILE]), target_type=pl.FP32)
            up_tile = pl.cast(pl.load(up, [0, c0], [rows, ACT_TILE]), target_type=pl.FP32)
            gate_tile = pl.minimum(gate_tile, LIMIT)
            up_tile = pl.minimum(pl.maximum(up_tile, -LIMIT), LIMIT)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
            value = pl.mul(pl.mul(gate_tile, sigmoid), up_tile)
            value = pl.row_expand_mul(value, weight)
            pl.store(pl.cast(value, target_type=pl.BF16, mode="rint"), [0, c0], activated)
    act_i8 = pl.create_tensor([rows, INTER], dtype=pl.INT8)
    act_scale = pl.create_tensor([rows, 1], dtype=pl.FP32)
    quantize_int8_per_token(activated, act_i8, act_scale, width=INTER)
    w8a8_gemm(act_i8, act_scale, w2, w2_scale, output, width=INTER, output_width=D, output_fp32=False)
    return output


@pl.jit.inline(auto_scope=False)
def expert_shared(
    x: pl.Tensor[[16, D], pl.BF16],
    w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[INTER], pl.FP32],
    w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[INTER], pl.FP32],
    w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[16, D], pl.BF16],
    rows: pl.constexpr,
):
    """Apply the shared expert with the same W8A8 SwiGLU path without a routing weight."""
    x_i8 = pl.create_tensor([rows, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([rows, 1], dtype=pl.FP32)
    quantize_int8_per_token(x, x_i8, x_scale, width=D)
    gate = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    up = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    w8a8_gemm(x_i8, x_scale, w1, w1_scale, gate, width=D, output_width=INTER, output_fp32=False)
    w8a8_gemm(x_i8, x_scale, w3, w3_scale, up, width=D, output_width=INTER, output_fp32=False)
    activated = pl.create_tensor([rows, INTER], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swiglu_shared"):
        for c0 in pl.pipeline(0, INTER, ACT_TILE, stage=2):
            gate_tile = pl.cast(gate[:, c0 : c0 + ACT_TILE], target_type=pl.FP32)
            up_tile = pl.cast(up[:, c0 : c0 + ACT_TILE], target_type=pl.FP32)
            gate_tile = pl.minimum(gate_tile, LIMIT)
            up_tile = pl.minimum(pl.maximum(up_tile, -LIMIT), LIMIT)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
            value = pl.mul(pl.mul(gate_tile, sigmoid), up_tile)
            activated[:, c0 : c0 + ACT_TILE] = pl.cast(value, target_type=pl.BF16, mode="rint")
    act_i8 = pl.create_tensor([rows, INTER], dtype=pl.INT8)
    act_scale = pl.create_tensor([rows, 1], dtype=pl.FP32)
    quantize_int8_per_token(activated, act_i8, act_scale, width=INTER)
    w8a8_gemm(act_i8, act_scale, w2, w2_scale, output, width=INTER, output_width=D, output_fp32=False)
    return output


def _gemm_golden(x_i8, x_scale, weight, weight_scale):
    acc = torch.matmul(x_i8.to(torch.int32), weight.to(torch.int32))
    return (acc.float() * x_scale * weight_scale.reshape(1, -1)).to(torch.bfloat16)


def _expert_golden(tensors, routed: bool):
    recv_x = tensors["recv_x"]
    x_i8, x_scale = int8_quant_per_token(recv_x)
    w1 = unpack_nz(tensors["w1"])
    w3 = unpack_nz(tensors["w3"])
    w2 = unpack_nz(tensors["w2"])
    gate = _gemm_golden(x_i8, x_scale, w1, tensors["w1_scale"]).float()
    up = _gemm_golden(x_i8, x_scale, w3, tensors["w3_scale"]).float()
    gate = torch.clamp(gate, max=LIMIT)
    up = torch.clamp(up, min=-LIMIT, max=LIMIT)
    value = gate * torch.sigmoid(gate) * up
    if routed:
        value = value * tensors["routed_weight"].float().reshape(-1, 1)
    value = value.to(torch.bfloat16)
    act_i8, act_scale = int8_quant_per_token(value)
    tensors["output"][:] = _gemm_golden(act_i8, act_scale, w2, tensors["w2_scale"])


# Device-calibrated expert-output gate. The second per-token INT8 activation
# quantization amplifies one BF16 ULP of activation rounding into a several
# percent elementwise spread: host calibration that perturbs the activation by
# one BF16 ULP gives rel-L2 0.88-1.27 % and 14-27 % of points outside the plain
# (2e-2, 2e-2) band. Real device runs measure 0.32-0.64 % rel-L2 and 0.66-2.89 %
# of points outside (1e-1, 5e-2) across fixtures (the 2.89 % case is the
# c2a_reuse layer, still far below the one-ULP envelope's ~5.0 %). The gate keeps
# a 1 % rel-L2 bound (below the one-ULP floor) plus a 4 % elementwise outlier
# allowance, so a wiring fault, a wrong route/weight, or a misplaced expert
# still fails by orders of magnitude rather than being absorbed by the
# quantization noise.
EXPERT_REL_L2 = 1e-2
EXPERT_ATOL = 1e-1
EXPERT_RTOL = 5e-2
EXPERT_MAX_ERROR_RATIO = 4e-2


def expert_output_compare(
    actual, expected, *,
    actual_outputs=None, expected_outputs=None, inputs=None,
    rtol=None, atol=None, **_kwargs,
):
    """Compare one expert output on the calibrated device arithmetic floor."""
    from golden import ratio_allclose

    ok, detail = ratio_allclose(
        atol=EXPERT_ATOL, rtol=EXPERT_RTOL, max_error_ratio=EXPERT_MAX_ERROR_RATIO
    )(
        actual, expected,
        actual_outputs=actual_outputs, expected_outputs=expected_outputs,
        inputs=inputs, rtol=rtol, atol=atol,
    )
    a = actual.float()
    e = expected.float()
    rel_l2 = ((a - e).norm() / e.norm()).item()
    ok = ok and rel_l2 <= EXPERT_REL_L2
    return ok, f"{detail}; rel_l2={rel_l2:.4%}"


def build_tensor_specs(rows: int = 16):
    from golden import TensorSpec

    holder = {}

    def weight_arg(name, n, k):
        from utils import pack_nz, quant_w_per_channel

        def build():
            logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
            holder[name] = scale
            return pack_nz(logical.transpose(0, 1).contiguous())

        return build

    def scale_arg(name):
        return lambda: holder[name]

    return [
        TensorSpec("recv_x", [rows, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("routed_weight", [rows, 1], torch.float32, init_value=lambda: torch.rand(rows, 1) + 0.1),
        TensorSpec("w1", [D, INTER], torch.int8, init_value=weight_arg("w1", INTER, D)),
        TensorSpec("w1_scale", [INTER], torch.float32, init_value=scale_arg("w1")),
        TensorSpec("w3", [D, INTER], torch.int8, init_value=weight_arg("w3", INTER, D)),
        TensorSpec("w3_scale", [INTER], torch.float32, init_value=scale_arg("w3")),
        TensorSpec("w2", [INTER, D], torch.int8, init_value=weight_arg("w2", D, INTER)),
        TensorSpec("w2_scale", [D], torch.float32, init_value=scale_arg("w2")),
        TensorSpec("output", [rows, D], torch.bfloat16),
    ]


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[16, 5120], pl.BF16],
    routed_weight: pl.Tensor[[16, 1], pl.FP32],
    w1: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[2304], pl.FP32],
    w3: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[2304], pl.FP32],
    w2: pl.Tensor[[2304, 5120], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[5120], pl.FP32],
    output: pl.Out[pl.Tensor[[16, 5120], pl.BF16]],
):
    expert_routed(recv_x, routed_weight, w1, w1_scale, w3, w3_scale, w2, w2_scale, output, rows=16)
    return output


@pl.jit
def expert_shared_test(
    recv_x: pl.Tensor[[16, 5120], pl.BF16],
    w1: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[2304], pl.FP32],
    w3: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[2304], pl.FP32],
    w2: pl.Tensor[[2304, 5120], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[5120], pl.FP32],
    output: pl.Out[pl.Tensor[[16, 5120], pl.BF16]],
):
    expert_shared(recv_x, w1, w1_scale, w3, w3_scale, w2, w2_scale, output, rows=16)
    return output


def _shared_specs(rows: int):
    specs = build_tensor_specs(rows)
    return [spec for spec in specs if spec.name != "routed_weight"]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    routed = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(args.rows),
        golden_fn=lambda tensors: _expert_golden(tensors, routed=True),
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"output": expert_output_compare},
    )
    if not routed.passed:
        if routed.error:
            print(routed.error)
        raise SystemExit(1)
    shared = run(
        fn=expert_shared_test,
        specs=_shared_specs(args.rows),
        golden_fn=lambda tensors: _expert_golden(tensors, routed=False),
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"output": expert_output_compare},
    )
    if not shared.passed:
        if shared.error:
            print(shared.error)
        raise SystemExit(1)
