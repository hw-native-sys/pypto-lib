# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm for dynamic decode and prefill shapes."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
REDUCE_D_TILE = D // 4
APPLY_D_TILE = D // 4
T_TILE = 8


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    with pl.spmd((t_dim // T_TILE) * (D // APPLY_D_TILE), name_hint="rms_norm") as rms_tid:
        rms_blk = pl.tile.get_block_idx()
        tg_idx = rms_blk // (D // APPLY_D_TILE)
        rms_dsplit = rms_blk - tg_idx * (D // APPLY_D_TILE)
        tg = tg_idx * T_TILE
        x_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for rms_d0 in pl.pipeline(0, D, REDUCE_D_TILE, stage=2):
            rms_x_chunk = pl.cast(x[tg : tg + T_TILE, rms_d0 : rms_d0 + REDUCE_D_TILE], target_type=pl.FP32)
            rms_x_sq = pl.mul(rms_x_chunk, rms_x_chunk)
            rms_x_sq_sum = pl.row_sum(rms_x_sq)
            rms_x_sq_row = pl.reshape(rms_x_sq_sum, [1, T_TILE])
            x_sq_sum = pl.add(x_sq_sum, rms_x_sq_row)
        x_sq_mean = pl.mul(x_sq_sum, 1.0 / D)
        x_sq_eps = pl.add(x_sq_mean, EPS)
        x_inv_rms = pl.rsqrt(x_sq_eps, high_precision=True)
        x_inv_rms_t = pl.reshape(x_inv_rms, [T_TILE, 1])
        apply_d_base = rms_dsplit * APPLY_D_TILE
        for apply_d0 in pl.pipeline(apply_d_base, apply_d_base + APPLY_D_TILE, APPLY_D_TILE, stage=2):
            apply_x_chunk = pl.cast(x[tg : tg + T_TILE, apply_d0 : apply_d0 + APPLY_D_TILE], target_type=pl.FP32)
            norm_w_row = pl.reshape(norm_w[apply_d0 : apply_d0 + APPLY_D_TILE], [1, APPLY_D_TILE])
            norm_w_chunk = pl.cast(norm_w_row, pl.FP32)
            x_scaled = pl.row_expand_mul(apply_x_chunk, x_inv_rms_t)
            x_normed_chunk = pl.col_expand_mul(x_scaled, norm_w_chunk)
            x_normed_bf16 = pl.cast(x_normed_chunk, target_type=pl.BF16, mode="rint")
            x_normed[tg : tg + T_TILE, apply_d0 : apply_d0 + APPLY_D_TILE] = x_normed_bf16

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
    # Accumulate squared chunks in kernel order.
    sq_sum = torch.zeros(*x.shape[:-1], 1, dtype=torch.float32)
    for d0 in range(0, D, REDUCE_D_TILE):
        x_chunk = x[..., d0:d0 + REDUCE_D_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
    inv = torch.rsqrt(sq_sum * (1.0 / D) + EPS)
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

    MODES = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    mode_help = "Use decode or prefill batch sizes, or 'all' to test both."
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all", help=mode_help)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

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
