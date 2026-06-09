# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RoPE — rotary position embedding via half-vector rotation.

    y_lo = x_lo * cos_lo - x_hi * sin_lo
    y_hi = x_hi * cos_hi + x_lo * sin_hi

x is laid out as [BATCH * NUM_HEADS, HEAD_DIM]; cos and sin are [1, HEAD_DIM]
broadcast across heads via col_expand_mul. pl.parallel tiles the batch; each
item rotates its NUM_HEADS rows. FP32 throughout.
"""
import pypto.language as pl

BATCH = 16          # batch size
NUM_HEADS = 8       # heads per batch item
HEAD_DIM = 128      # dimension per head

TOTAL_ROWS = BATCH * NUM_HEADS
HALF_DIM = HEAD_DIM // 2


@pl.jit
def rope(
    x: pl.Tensor[[TOTAL_ROWS, HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    y: pl.Out[pl.Tensor[[TOTAL_ROWS, HEAD_DIM], pl.FP32]],
):
    for b in pl.parallel(0, BATCH, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_rotate"):
            cos_lo = cos[:, 0:HALF_DIM]
            cos_hi = cos[:, HALF_DIM:HEAD_DIM]
            sin_lo = sin[:, 0:HALF_DIM]
            sin_hi = sin[:, HALF_DIM:HEAD_DIM]

            base = b * NUM_HEADS
            x_lo = x[base : base + NUM_HEADS, 0:HALF_DIM]
            x_hi = x[base : base + NUM_HEADS, HALF_DIM:HEAD_DIM]

            # lower half: x_lo * cos_lo - x_hi * sin_lo
            lo_cos = pl.col_expand_mul(x_lo, cos_lo)
            lo_sin = pl.col_expand_mul(x_hi, sin_lo)
            y[base : base + NUM_HEADS, 0:HALF_DIM] = pl.sub(lo_cos, lo_sin)

            # upper half: x_hi * cos_hi + x_lo * sin_hi
            hi_cos = pl.col_expand_mul(x_hi, cos_hi)
            hi_sin = pl.col_expand_mul(x_lo, sin_hi)
            y[base : base + NUM_HEADS, HALF_DIM:HEAD_DIM] = pl.add(hi_cos, hi_sin)
    return y


def build_tensor_specs(
    batch: int = BATCH,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
):
    import torch
    from golden import TensorSpec

    total_rows = batch * num_heads

    return [
        TensorSpec("x", [total_rows, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("cos", [1, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("sin", [1, head_dim], torch.float32, init_value=torch.randn),
        TensorSpec("y", [total_rows, head_dim], torch.float32, is_output=True),
    ]


def golden_rope(tensors):
    import torch

    x = tensors["x"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    half = x.shape[-1] // 2

    tensors["y"][:] = torch.cat(
        [
            x[:, :half] * cos[:, :half] - x[:, half:] * sin[:, :half],
            x[:, half:] * cos[:, half:] + x[:, :half] * sin[:, half:],
        ],
        dim=-1,
    )


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=rope,
        specs=build_tensor_specs(),
        golden_fn=golden_rope,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
