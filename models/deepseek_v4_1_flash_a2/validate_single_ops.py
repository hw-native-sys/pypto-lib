"""Golden-harness entry for the first A2/A3 W8A8 leaf validation."""

from __future__ import annotations

import argparse

import pypto.language as pl
import torch

from .moe import make_shared_expert_projection
from .quant import (
    quantize_per_channel_int8,
    quantize_per_token_int8,
    w8a8_linear,
)


CAPACITY = 16
WIDTH = 256
OUTPUT_WIDTH = 64
ACTIVE = 13

_expert_projection = make_shared_expert_projection(CAPACITY, WIDTH, OUTPUT_WIDTH)


@pl.jit
def w8a8_single_op(
    value: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    packed_weight: pl.Tensor[[WIDTH, OUTPUT_WIDTH], pl.INT8],
    channel_scale: pl.Tensor[[OUTPUT_WIDTH], pl.FP32],
    payload: pl.InOut[pl.Tensor[[CAPACITY, WIDTH], pl.INT8]],
    token_scale: pl.InOut[pl.Tensor[[CAPACITY, 1], pl.FP32]],
    accumulator: pl.InOut[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.INT32]],
    output: pl.InOut[pl.Tensor[[CAPACITY, OUTPUT_WIDTH], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    _expert_projection(
        value,
        packed_weight,
        channel_scale,
        payload,
        token_scale,
        accumulator,
        output,
        active,
    )
    return payload, token_scale, accumulator, output


def build_specs():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(41)
    value = torch.randn(CAPACITY, WIDTH, dtype=torch.bfloat16)
    weight_fp = torch.randn(OUTPUT_WIDTH, WIDTH, dtype=torch.float32)
    weight_i8, channel_scale = quantize_per_channel_int8(weight_fp)
    packed_weight = weight_i8.transpose(0, 1).contiguous()
    return [
        TensorSpec("value", [CAPACITY, WIDTH], torch.bfloat16, init_value=value),
        TensorSpec("packed_weight", [WIDTH, OUTPUT_WIDTH], torch.int8, init_value=packed_weight),
        TensorSpec("channel_scale", [OUTPUT_WIDTH], torch.float32, init_value=channel_scale.reshape(-1)),
        TensorSpec("payload", [CAPACITY, WIDTH], torch.int8, init_value=torch.zeros),
        TensorSpec("token_scale", [CAPACITY, 1], torch.float32, init_value=torch.zeros),
        TensorSpec("accumulator", [CAPACITY, OUTPUT_WIDTH], torch.int32, init_value=torch.zeros),
        TensorSpec("output", [CAPACITY, OUTPUT_WIDTH], torch.bfloat16, init_value=torch.zeros),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def golden_single_op(values):
    value = values["value"]
    weight = values["packed_weight"].transpose(0, 1).contiguous()
    scale = values["channel_scale"]
    output = w8a8_linear(value, weight, scale, out_dtype=torch.bfloat16)
    payload, token_scale = quantize_per_token_int8(value)
    accumulator = torch.matmul(payload.to(torch.int32), weight.to(torch.int32).transpose(0, 1))
    accumulator[ACTIVE:] = 0
    output[ACTIVE:] = 0
    values["payload"][:] = payload
    values["token_scale"][:] = token_scale
    values["accumulator"][:] = accumulator
    values["output"][:] = output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    from golden import run

    result = run(
        fn=w8a8_single_op,
        specs=build_specs(),
        golden_fn=golden_single_op,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compile_only=args.compile_only,
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    print(result)
    if not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
