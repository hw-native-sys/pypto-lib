# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm (dynamic shape): normalizes token-major
activations for both decode and prefill attention paths."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
D_TILE = 128
T_TILE = 8
# Blocks per token-tile. A decode step has t_dim == T == 8 == T_TILE, so the grid was
# ONE block and rms_norm sat on the observed critical path as 22.4us of single-core
# work with 83 cores idle (level-4 capture: it is task #4 of the 24-task path, and the
# 0-88us window averages 5% engine occupancy).
#
# The token axis cannot go finer: T_TILE < 8 puts the [1, T_TILE] FP32 row accumulator
# below the 32-byte alloc-tile row floor and codegen rejects it. So split D instead.
# Each block recomputes the SAME full-row sum of squares -- deliberately redundant, so
# the task stays one dispatch with no cross-block reduction -- and then applies the
# normalization to its own D // RMS_D_SPLIT columns only. Bit-identical: every block
# derives x_inv_rms from the same inputs in the same chunk order, and the applied
# column ranges are disjoint.
#
# The trade is core-time for wall-time: 4-way costs ~35us of extra vector busy but the
# task's wall drops 26.1 -> 16.0us in the swimlane. Device-measured on the a5 CSA decode
# case, paired interleaved A/B, 5 reps x 100 rounds: 715.6 -> 709.7us (-5.9). 2-way gave
# -4.2us and 7-way -8.1us with more variance, so 4 is the knee.
RMS_D_SPLIT = 4
assert D % D_TILE == 0, "D must be divisible by D_TILE"
assert (D // D_TILE) % RMS_D_SPLIT == 0, "apply-pass D-chunk loop must cover D"
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    # Capture form (not `for ... in pl.spmd`): callers need the producer TaskId to
    # hang a `pl.system.task_dummy` barrier off it and defer non-critical consumers.
    with pl.spmd((t_dim // T_TILE) * RMS_D_SPLIT, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        rms_blk = pl.tile.get_block_idx()
        tg_idx = rms_blk // RMS_D_SPLIT
        rms_dsplit = rms_blk - tg_idx * RMS_D_SPLIT
        tg = tg_idx * T_TILE
        x_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(D // D_TILE, stage=2):
            rms_d0 = rms_db * D_TILE
            rms_x_chunk = pl.cast(x[tg : tg + T_TILE, rms_d0 : rms_d0 + D_TILE], target_type=pl.FP32)
            x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, T_TILE]))
        x_inv_rms = pl.rsqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS), high_precision=True)
        x_inv_rms_t = pl.reshape(x_inv_rms, [T_TILE, 1])
        apply_dbs = (D // D_TILE) // RMS_D_SPLIT
        for apply_db in pl.pipeline(apply_dbs, stage=2):
            apply_d0 = (rms_dsplit * apply_dbs + apply_db) * D_TILE
            apply_x_chunk = pl.cast(x[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE], target_type=pl.FP32)
            norm_w_chunk = pl.cast(pl.reshape(norm_w[apply_d0 : apply_d0 + D_TILE], [1, D_TILE]), pl.FP32)
            x_normed_chunk = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
            x_normed[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE] = pl.cast(
                x_normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

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
    # Mirror rms_norm's loop-carried FP32 reduction exactly.  A monolithic
    # torch.mean uses a different reduction tree and can cross a BF16 rounding
    # boundary even though both expressions are algebraically equivalent.
    sq_sum = torch.zeros(*x.shape[:-1], 1, dtype=torch.float32)
    for d0 in range(0, D, D_TILE):
        x_chunk = x[..., d0:d0 + D_TILE]
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
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
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
