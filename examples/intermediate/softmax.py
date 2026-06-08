# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Softmax — row-wise numerically stable softmax with row-chunk tiling.

    output[r, c] = exp(x[r, c] - max_row(x)) / sum_row(exp(x[r, c] - max_row(x)))

Authored in the ``@pl.jit`` form (see docs/pypto-coding-style.md §1):
``softmax`` is a reusable ``@pl.jit.inline`` sub-kernel called by the thin
``@pl.jit`` entry. ``pl.spmd`` dispatches one row-tile per core; each tile
is self-contained because softmax normalises across the column (hidden)
dimension only, and the full row width fits in one on-chip tile (no inner
column loop).

Input and output are FP32.
"""
import pypto.language as pl

ROWS = 512
COLS = 256
ROW_CHUNK = 64          # rows per SPMD tile

ROW_TILES = ROWS // ROW_CHUNK


@pl.jit.inline
def softmax(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for rb in pl.spmd(ROW_TILES, name_hint="softmax"):
        r = rb * ROW_CHUNK
        tile_x = x[r : r + ROW_CHUNK, 0:COLS]

        # Step 1: row-wise max for numerical stability
        row_max = pl.row_max(tile_x)

        # Step 2: subtract row max: x - max(x)
        shifted = pl.row_expand_sub(tile_x, row_max)

        # Step 3: exp(x - max(x))
        exp_shifted = pl.exp(shifted)

        # Step 4: row-wise sum of exp values
        row_sum = pl.row_sum(exp_shifted)

        # Step 5: divide each row by its sum
        y[r : r + ROW_CHUNK, 0:COLS] = pl.row_expand_div(exp_shifted, row_sum)

    return y


@pl.jit
def softmax_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    y = softmax(x, y)
    return y


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [ROWS, COLS], torch.float32, init_value=torch.randn),
        TensorSpec("y", [ROWS, COLS], torch.float32, is_output=True),
    ]


def golden_softmax(tensors):
    import torch

    tensors["y"][:] = torch.softmax(tensors["x"], dim=-1)


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
        fn=softmax_kernel,
        specs=build_tensor_specs(),
        golden_fn=golden_softmax,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
