# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm — root-mean-square normalization with row + column tiling.

    output[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * gamma[c]

Rows are tiled across core-groups via pl.parallel. The hidden dimension is
chunked with pl.range: one pass accumulates sum(x^2), a second normalises and
applies gamma. FP32 in/out; gamma is a [1, hidden] weight vector.
"""
import pypto.language as pl

ROWS = 512              # batch / sequence length
HIDDEN = 512            # hidden dimension (normalised axis)
ROW_TILE = 64           # rows per core-group
HIDDEN_TILE = 64        # columns per sequential chunk
EPS = 1e-6


@pl.jit
def rms_norm(
    x: pl.Tensor[[ROWS, HIDDEN], pl.FP32],
    gamma: pl.Tensor[[1, HIDDEN], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, HIDDEN], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_norm_rows"):
            # accumulate sum(x^2) across hidden chunks; keep it [1, ROW_TILE]
            # so the scalar rsqrt below stays row-major
            sq_sum = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
            for hb in pl.range(HIDDEN // HIDDEN_TILE):
                h0 = hb * HIDDEN_TILE
                x_tile = x[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE]
                sq = pl.mul(x_tile, x_tile)
                sq_row = pl.row_sum(sq)
                sq_row = pl.reshape(sq_row, [1, ROW_TILE])
                sq_sum = pl.add(sq_sum, sq_row)

            mean_sq = pl.mul(sq_sum, 1.0 / HIDDEN)
            mean_sq = pl.add(mean_sq, EPS)
            inv_rms = pl.rsqrt(mean_sq)
            inv_rms = pl.reshape(inv_rms, [ROW_TILE, 1])

            # normalise each chunk and apply the gamma weight
            for hb in pl.range(HIDDEN // HIDDEN_TILE):
                h0 = hb * HIDDEN_TILE
                x_tile = x[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE]
                gamma_tile = gamma[:, h0 : h0 + HIDDEN_TILE]
                normed = pl.row_expand_mul(x_tile, inv_rms)
                normed = pl.col_expand_mul(normed, gamma_tile)
                y[r : r + ROW_TILE, h0 : h0 + HIDDEN_TILE] = normed
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
        TensorSpec("y", [rows, hidden], torch.float32, is_output=True),
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
        fn=rms_norm,
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
