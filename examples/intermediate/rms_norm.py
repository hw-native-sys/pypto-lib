# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm — Root Mean Square layer normalization with row + column tiling.

    output[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * gamma[c]

Authored in the ``@pl.jit`` form (see docs/pypto-coding-style.md §1):
``rms_norm`` is a reusable ``@pl.jit.inline`` sub-kernel that the thin
``@pl.jit`` entry calls. Inside, ``pl.spmd`` dispatches one row-tile per
core and ``pl.pipeline(stage=2)`` software-pipelines the two passes over
the hidden dimension — the standard column-chunking pattern used in
production LLM kernels (see models/deepseek/v4/decode_rmsnorm.py) where
the hidden dimension exceeds on-chip buffer capacity.

Input and output are FP32; gamma is a [hidden] weight vector (the 1-D
per-channel form used by production norm kernels).
"""
import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 512            # hidden dimension (normalised axis)
ROW_CHUNK = 64          # rows per SPMD tile
HIDDEN_CHUNK = 128      # columns per pipelined chunk (512B fp32 = L2 line)
EPS = 1e-6

ROW_TILES = ROWS // ROW_CHUNK
HIDDEN_BLOCKS = HIDDEN // HIDDEN_CHUNK


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    for rb in pl.spmd(ROW_TILES, name_hint="rms_norm"):
        r = rb * ROW_CHUNK

        # Pass 1: accumulate sum(x^2) across hidden chunks. row_sum yields
        # [ROW_CHUNK, 1] col_major; scalar ops need row_major, so carry the
        # reduction in [1, ROW_CHUNK] shape.
        sq_sum = pl.full([1, ROW_CHUNK], dtype=pl.FP32, value=0.0)
        for hb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
            h0 = hb * HIDDEN_CHUNK
            x_chunk = x[r : r + ROW_CHUNK, h0 : h0 + HIDDEN_CHUNK]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, ROW_CHUNK]))

        # inv_rms = 1 / sqrt(mean(x^2) + eps)
        inv_rms = pl.reshape(pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / HIDDEN), EPS))), [ROW_CHUNK, 1])

        # Pass 2: normalise and apply gamma weight
        for hb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
            h0 = hb * HIDDEN_CHUNK
            x_chunk = x[r : r + ROW_CHUNK, h0 : h0 + HIDDEN_CHUNK]
            gamma_chunk = pl.reshape(gamma[h0 : h0 + HIDDEN_CHUNK], [1, HIDDEN_CHUNK])
            y[r : r + ROW_CHUNK, h0 : h0 + HIDDEN_CHUNK] = pl.col_expand_mul(
                pl.row_expand_mul(x_chunk, inv_rms), gamma_chunk
            )

    return y


@pl.jit
def rms_norm_kernel(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    y = rms_norm(x, gamma, y)
    return y


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [HIDDEN], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, HIDDEN], torch.float32, is_output=True),
    ]


def golden_rms_norm(tensors):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    tensors["y"][:] = x / rms * gamma


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
        fn=rms_norm_kernel,
        specs=build_tensor_specs(),
        golden_fn=golden_rms_norm,
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
