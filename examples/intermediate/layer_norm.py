# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""LayerNorm — full layer normalization with row-only tiling.

    output[r, c] = (x[r, c] - mean(x[r, :])) / sqrt(var(x[r, :]) + eps) * gamma[c] + beta[c]

Rows are tiled across core-groups via pl.parallel; the hidden dimension fits in
one tile (no column chunking). FP32 in/out; gamma and beta are [1, hidden].
"""
import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 256            # hidden dimension (normalised axis, fits in one tile)
ROW_TILE = 32           # rows per core-group
EPS = 1e-5


@pl.jit
def layer_norm(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    beta: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="layer_norm_rows"):
            tile_x = x[r : r + ROW_TILE, :]

            # row mean (pre-scale before row_sum to keep it row-major)
            scaled_x = pl.mul(tile_x, 1.0 / HIDDEN)
            mean = pl.row_sum(scaled_x)
            centred = pl.row_expand_sub(tile_x, mean)

            # row variance + eps, then standard deviation
            sq = pl.mul(centred, centred)
            sq = pl.add(sq, EPS)
            sq = pl.mul(sq, 1.0 / HIDDEN)
            var_eps = pl.row_sum(sq)
            var_eps = pl.reshape(var_eps, [1, ROW_TILE])
            std = pl.sqrt(var_eps)
            std = pl.reshape(std, [ROW_TILE, 1])
            normed = pl.row_expand_div(centred, std)

            # apply gamma scale and beta offset
            scaled = pl.col_expand_mul(normed, gamma[:, :])
            ones = pl.sub(tile_x, tile_x)
            ones = pl.add(ones, 1.0)
            offset = pl.col_expand_mul(ones, beta[:, :])
            y[r : r + ROW_TILE, :] = pl.add(scaled, offset)
    return y


def build_tensor_specs(
    rows: int = ROWS,
    hidden: int = HIDDEN,
):
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("x", [rows, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("gamma", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("beta", [1, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("y", [rows, hidden], torch.float32, is_output=True),
    ]


def golden_layer_norm(tensors):
    import torch

    x = tensors["x"]
    gamma = tensors["gamma"]
    beta = tensors["beta"]
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    tensors["y"][:] = (x - mean) / torch.sqrt(var + 1e-5) * gamma + beta


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
        fn=layer_norm,
        specs=build_tensor_specs(),
        golden_fn=golden_layer_norm,
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
