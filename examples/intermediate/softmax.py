# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Softmax — numerically stable row-wise softmax with row tiling.

    output[r, c] = exp(x[r, c] - max_row(x)) / sum_row(exp(x[r, c] - max_row(x)))

Rows are tiled across core-groups via pl.parallel; each tile is normalised
independently along the column axis. FP32 in/out.
"""
import pypto.language as pl

ROWS = 512
COLS = 256
ROW_TILE = 64           # rows per core-group


@pl.jit
def softmax(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            tile_x = x[r : r + ROW_TILE, :]
            # subtract the row max for numerical stability, then exponentiate
            row_max = pl.row_max(tile_x)
            shifted = pl.row_expand_sub(tile_x, row_max)
            exp_shifted = pl.exp(shifted)
            # divide each row by the sum of its exponentials
            denom = pl.row_sum(exp_shifted)
            y[r : r + ROW_TILE, :] = pl.row_expand_div(exp_shifted, denom)
    return y


def build_tensor_specs(
    rows: int = ROWS,
    cols: int = COLS,
):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, cols], torch.float32, is_output=True),
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
        fn=softmax,
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
