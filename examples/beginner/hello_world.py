# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hello World — add a scalar to every element of a matrix.

    output[r, c] = input[r, c] + a

pl.parallel distributes the row tiles across core-groups; an inner pl.range
walks the column tiles so each pl.at InCore scope works on one
[ROW_TILE, COL_TILE] block: load it, add the scalar, store it.
"""
import pypto.language as pl

ROWS = 1024
COLS = 512
ROW_TILE = 128
COL_TILE = 256


@pl.jit
def hello_world(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    a: pl.Scalar[pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        for c in pl.range(0, COLS, COL_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_scalar"):
                tile_x = x[r : r + ROW_TILE, c : c + COL_TILE]
                y[r : r + ROW_TILE, c : c + COL_TILE] = pl.add(tile_x, a)
    return y


def build_specs(
    rows: int = ROWS,
    cols: int = COLS,
    a: float = 1.0,
):
    import torch
    from golden import ScalarSpec, TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        ScalarSpec("a", torch.float32, a),
        TensorSpec("y", [rows, cols], torch.float32, is_output=True),
    ]


def golden_hello_world(values):
    values["y"][:] = values["x"] + values["a"]


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
        fn=hello_world,
        specs=build_specs(),
        golden_fn=golden_hello_world,
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
