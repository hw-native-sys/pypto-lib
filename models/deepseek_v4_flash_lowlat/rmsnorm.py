# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm (dynamic shape): normalizes token-major
activations for the decode attention path."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
REDUCE_ROW_TILE = 16

@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    norm_w_view = pl.reshape(norm_w, [1, D])
    with pl.spmd(t_dim, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        tok = pl.tile.get_block_idx()
        x_row_bf16 = pl.tile.load(x, [tok, 0], [1, D])
        w_row_bf16 = pl.tile.load(norm_w_view, [0, 0], [1, D])
        x_row = pl.cast(x_row_bf16, target_type=pl.FP32)
        w_row = pl.cast(w_row_bf16, target_type=pl.FP32)
        x_sq = pl.mul(x_row, x_row)
        sq_rows = pl.reshape(x_sq, [REDUCE_ROW_TILE, D // REDUCE_ROW_TILE])
        partial_tmp = pl.create_tile([REDUCE_ROW_TILE, D // REDUCE_ROW_TILE], dtype=pl.FP32)
        partial = pl.row_sum(sq_rows, partial_tmp)
        reduce_tile = pl.create_tile([REDUCE_ROW_TILE, REDUCE_ROW_TILE], dtype=pl.FP32)
        reduce_tile[0:1, :] = pl.reshape(partial, [1, REDUCE_ROW_TILE])
        reduce_tile = pl.set_validshape(reduce_tile, 1, REDUCE_ROW_TILE)
        sum_tmp = pl.create_tile([REDUCE_ROW_TILE, REDUCE_ROW_TILE], dtype=pl.FP32)
        sum_tile = pl.row_sum(reduce_tile, sum_tmp)
        sum_row = pl.reshape(sum_tile, [1, REDUCE_ROW_TILE])
        x_sq_sum = pl.set_validshape(sum_row, 1, 1)
        x_mean_sq = pl.mul(x_sq_sum, 1.0 / D)
        x_mean_eps = pl.add(x_mean_sq, EPS)
        rsqrt_tmp = pl.create_tile([1, REDUCE_ROW_TILE], dtype=pl.FP32)
        x_inv_rms = pl.tile.rsqrt(x_mean_eps, rsqrt_tmp)
        inv_rms_scalar = pl.tile.read(x_inv_rms, [0, 0])
        x_scaled = pl.mul(x_row, inv_rms_scalar)
        x_weighted = pl.mul(x_scaled, w_row)
        normed_bf16 = pl.cast(x_weighted, target_type=pl.BF16, mode="rint")
        pl.tile.store(normed_bf16, [tok, 0], x_normed, shapes=[1, D])

    return rms_tid


@pl.jit
def rms_norm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)

    rms_norm(x, norm_w, x_normed)
    return x_normed


def golden_rms_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)


def golden_rms_norm_test(tensors):
    tensors["x_normed"][:] = golden_rms_norm(tensors["x"], tensors["norm_w"])


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_normed", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {"decode": (DECODE_BATCH, DECODE_SEQ)}

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode"], default="decode")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- rms_norm_test {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=rms_norm_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_rms_norm_test,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
