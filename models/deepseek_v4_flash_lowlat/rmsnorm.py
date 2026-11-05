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

import inspect

import pypto.language as pl

from config import FLASH as M
from decode_shapes import DECODE_BATCH, DECODE_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
REDUCE_ROW_TILE = 16

# Select the adapter in Python before JIT parsing: the old frontend rejects
# the new keyword, while the explicit-precision frontend requires it with tmp.
if "high_precision" in inspect.signature(pl.tile.rsqrt).parameters:

    @pl.jit.inline
    def _rsqrt_high_precision(
        value: pl.Tile[[1, REDUCE_ROW_TILE], pl.FP32],
        workspace: pl.Tile[[1, REDUCE_ROW_TILE], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=workspace, high_precision=True)

else:

    @pl.jit.inline
    def _rsqrt_high_precision(
        value: pl.Tile[[1, REDUCE_ROW_TILE], pl.FP32],
        workspace: pl.Tile[[1, REDUCE_ROW_TILE], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=workspace)


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
        square_sum_scalar = pl.tile.read(x_sq_sum, [0, 0])
        # Compensate bounded positive sums without changing the overflow path.
        if square_sum_scalar <= pl.cast(1099511627776.0, pl.FP32):
            rms_2048_a = pl.tile.slice(x_sq, [1, 2048], [0, 0])
            rms_2048_b = pl.tile.slice(x_sq, [1, 2048], [0, 2048])
            rms_2048_hi = pl.add(rms_2048_a, rms_2048_b)
            rms_2048_z = pl.sub(rms_2048_hi, rms_2048_a)
            rms_2048_back = pl.sub(rms_2048_hi, rms_2048_z)
            rms_2048_ae = pl.sub(rms_2048_a, rms_2048_back)
            rms_2048_be = pl.sub(rms_2048_b, rms_2048_z)
            rms_2048_error = pl.add(rms_2048_ae, rms_2048_be)
            rms_2048_lo = rms_2048_error
            rms_1024_a = pl.tile.slice(rms_2048_hi, [1, 1024], [0, 0])
            rms_1024_b = pl.tile.slice(rms_2048_hi, [1, 1024], [0, 1024])
            rms_1024_hi = pl.add(rms_1024_a, rms_1024_b)
            rms_1024_z = pl.sub(rms_1024_hi, rms_1024_a)
            rms_1024_back = pl.sub(rms_1024_hi, rms_1024_z)
            rms_1024_ae = pl.sub(rms_1024_a, rms_1024_back)
            rms_1024_be = pl.sub(rms_1024_b, rms_1024_z)
            rms_1024_error = pl.add(rms_1024_ae, rms_1024_be)
            rms_1024_la = pl.tile.slice(rms_2048_lo, [1, 1024], [0, 0])
            rms_1024_lb = pl.tile.slice(rms_2048_lo, [1, 1024], [0, 1024])
            rms_1024_low_sum = pl.add(rms_1024_la, rms_1024_lb)
            rms_1024_lo = pl.add(rms_1024_low_sum, rms_1024_error)
            rms_512_a = pl.tile.slice(rms_1024_hi, [1, 512], [0, 0])
            rms_512_b = pl.tile.slice(rms_1024_hi, [1, 512], [0, 512])
            rms_512_hi = pl.add(rms_512_a, rms_512_b)
            rms_512_z = pl.sub(rms_512_hi, rms_512_a)
            rms_512_back = pl.sub(rms_512_hi, rms_512_z)
            rms_512_ae = pl.sub(rms_512_a, rms_512_back)
            rms_512_be = pl.sub(rms_512_b, rms_512_z)
            rms_512_error = pl.add(rms_512_ae, rms_512_be)
            rms_512_la = pl.tile.slice(rms_1024_lo, [1, 512], [0, 0])
            rms_512_lb = pl.tile.slice(rms_1024_lo, [1, 512], [0, 512])
            rms_512_low_sum = pl.add(rms_512_la, rms_512_lb)
            rms_512_lo = pl.add(rms_512_low_sum, rms_512_error)
            rms_256_a = pl.tile.slice(rms_512_hi, [1, 256], [0, 0])
            rms_256_b = pl.tile.slice(rms_512_hi, [1, 256], [0, 256])
            rms_256_hi = pl.add(rms_256_a, rms_256_b)
            rms_256_z = pl.sub(rms_256_hi, rms_256_a)
            rms_256_back = pl.sub(rms_256_hi, rms_256_z)
            rms_256_ae = pl.sub(rms_256_a, rms_256_back)
            rms_256_be = pl.sub(rms_256_b, rms_256_z)
            rms_256_error = pl.add(rms_256_ae, rms_256_be)
            rms_256_la = pl.tile.slice(rms_512_lo, [1, 256], [0, 0])
            rms_256_lb = pl.tile.slice(rms_512_lo, [1, 256], [0, 256])
            rms_256_low_sum = pl.add(rms_256_la, rms_256_lb)
            rms_256_lo = pl.add(rms_256_low_sum, rms_256_error)
            rms_128_a = pl.tile.slice(rms_256_hi, [1, 128], [0, 0])
            rms_128_b = pl.tile.slice(rms_256_hi, [1, 128], [0, 128])
            rms_128_hi = pl.add(rms_128_a, rms_128_b)
            rms_128_z = pl.sub(rms_128_hi, rms_128_a)
            rms_128_back = pl.sub(rms_128_hi, rms_128_z)
            rms_128_ae = pl.sub(rms_128_a, rms_128_back)
            rms_128_be = pl.sub(rms_128_b, rms_128_z)
            rms_128_error = pl.add(rms_128_ae, rms_128_be)
            rms_128_la = pl.tile.slice(rms_256_lo, [1, 128], [0, 0])
            rms_128_lb = pl.tile.slice(rms_256_lo, [1, 128], [0, 128])
            rms_128_low_sum = pl.add(rms_128_la, rms_128_lb)
            rms_128_lo = pl.add(rms_128_low_sum, rms_128_error)
            rms_64_a = pl.tile.slice(rms_128_hi, [1, 64], [0, 0])
            rms_64_b = pl.tile.slice(rms_128_hi, [1, 64], [0, 64])
            rms_64_hi = pl.add(rms_64_a, rms_64_b)
            rms_64_z = pl.sub(rms_64_hi, rms_64_a)
            rms_64_back = pl.sub(rms_64_hi, rms_64_z)
            rms_64_ae = pl.sub(rms_64_a, rms_64_back)
            rms_64_be = pl.sub(rms_64_b, rms_64_z)
            rms_64_error = pl.add(rms_64_ae, rms_64_be)
            rms_64_la = pl.tile.slice(rms_128_lo, [1, 64], [0, 0])
            rms_64_lb = pl.tile.slice(rms_128_lo, [1, 64], [0, 64])
            rms_64_low_sum = pl.add(rms_64_la, rms_64_lb)
            rms_64_lo = pl.add(rms_64_low_sum, rms_64_error)
            rms_32_a = pl.tile.slice(rms_64_hi, [1, 32], [0, 0])
            rms_32_b = pl.tile.slice(rms_64_hi, [1, 32], [0, 32])
            rms_32_hi = pl.add(rms_32_a, rms_32_b)
            rms_32_z = pl.sub(rms_32_hi, rms_32_a)
            rms_32_back = pl.sub(rms_32_hi, rms_32_z)
            rms_32_ae = pl.sub(rms_32_a, rms_32_back)
            rms_32_be = pl.sub(rms_32_b, rms_32_z)
            rms_32_error = pl.add(rms_32_ae, rms_32_be)
            rms_32_la = pl.tile.slice(rms_64_lo, [1, 32], [0, 0])
            rms_32_lb = pl.tile.slice(rms_64_lo, [1, 32], [0, 32])
            rms_32_low_sum = pl.add(rms_32_la, rms_32_lb)
            rms_32_lo = pl.add(rms_32_low_sum, rms_32_error)
            rms_16_a = pl.tile.slice(rms_32_hi, [1, 16], [0, 0])
            rms_16_b = pl.tile.slice(rms_32_hi, [1, 16], [0, 16])
            rms_16_hi = pl.add(rms_16_a, rms_16_b)
            rms_16_z = pl.sub(rms_16_hi, rms_16_a)
            rms_16_back = pl.sub(rms_16_hi, rms_16_z)
            rms_16_ae = pl.sub(rms_16_a, rms_16_back)
            rms_16_be = pl.sub(rms_16_b, rms_16_z)
            rms_16_error = pl.add(rms_16_ae, rms_16_be)
            rms_16_la = pl.tile.slice(rms_32_lo, [1, 16], [0, 0])
            rms_16_lb = pl.tile.slice(rms_32_lo, [1, 16], [0, 16])
            rms_16_low_sum = pl.add(rms_16_la, rms_16_lb)
            rms_16_lo = pl.add(rms_16_low_sum, rms_16_error)
            rms_8_a = pl.tile.slice(rms_16_hi, [1, 8], [0, 0])
            rms_8_b = pl.tile.slice(rms_16_hi, [1, 8], [0, 8])
            rms_8_hi = pl.add(rms_8_a, rms_8_b)
            rms_8_z = pl.sub(rms_8_hi, rms_8_a)
            rms_8_back = pl.sub(rms_8_hi, rms_8_z)
            rms_8_ae = pl.sub(rms_8_a, rms_8_back)
            rms_8_be = pl.sub(rms_8_b, rms_8_z)
            rms_8_error = pl.add(rms_8_ae, rms_8_be)
            rms_8_la = pl.tile.slice(rms_16_lo, [1, 8], [0, 0])
            rms_8_lb = pl.tile.slice(rms_16_lo, [1, 8], [0, 8])
            rms_8_low_sum = pl.add(rms_8_la, rms_8_lb)
            rms_8_lo = pl.add(rms_8_low_sum, rms_8_error)
            tail_hi = pl.cast(0.0, pl.FP32)
            tail_lo = pl.cast(0.0, pl.FP32)
            for tail_lane in pl.range(8):
                tail_value = pl.tile.read(rms_8_hi, [0, tail_lane])
                tail_error = pl.tile.read(rms_8_lo, [0, tail_lane])
                tail_sum = tail_hi + tail_value
                tail_z = tail_sum - tail_hi
                tail_back = tail_sum - tail_z
                tail_ae = tail_hi - tail_back
                tail_be = tail_value - tail_z
                tail_rounding = tail_ae + tail_be
                tail_lo = (tail_lo + tail_error) + tail_rounding
                tail_hi = tail_sum
            compensated_scalar = tail_hi + tail_lo
            pl.tile.write(x_sq_sum, [0, 0], compensated_scalar)
        x_mean_sq = pl.mul(x_sq_sum, 1.0 / D)
        x_mean_eps = pl.add(x_mean_sq, EPS)
        rsqrt_tmp = pl.create_tile([1, REDUCE_ROW_TILE], dtype=pl.FP32)
        raw_inverse = _rsqrt_high_precision(x_mean_eps, rsqrt_tmp)
        inverse_work = pl.tile.minimums(x_mean_eps, 1099511627776.0)
        inverse_square_as = pl.mul(raw_inverse, 4097.0)
        inverse_square_ad = pl.sub(inverse_square_as, raw_inverse)
        inverse_square_ah = pl.sub(inverse_square_as, inverse_square_ad)
        inverse_square_al = pl.sub(raw_inverse, inverse_square_ah)
        inverse_square_bs = pl.mul(raw_inverse, 4097.0)
        inverse_square_bd = pl.sub(inverse_square_bs, raw_inverse)
        inverse_square_bh = pl.sub(inverse_square_bs, inverse_square_bd)
        inverse_square_bl = pl.sub(raw_inverse, inverse_square_bh)
        inverse_square_hi = pl.mul(raw_inverse, raw_inverse)
        inverse_square_hh = pl.mul(inverse_square_ah, inverse_square_bh)
        inverse_square_e0 = pl.sub(inverse_square_hh, inverse_square_hi)
        inverse_square_hl = pl.mul(inverse_square_ah, inverse_square_bl)
        inverse_square_e1 = pl.add(inverse_square_e0, inverse_square_hl)
        inverse_square_lh = pl.mul(inverse_square_al, inverse_square_bh)
        inverse_square_e2 = pl.add(inverse_square_e1, inverse_square_lh)
        inverse_square_ll = pl.mul(inverse_square_al, inverse_square_bl)
        inverse_square_lo = pl.add(inverse_square_e2, inverse_square_ll)
        inverse_product_as = pl.mul(inverse_work, 4097.0)
        inverse_product_ad = pl.sub(inverse_product_as, inverse_work)
        inverse_product_ah = pl.sub(inverse_product_as, inverse_product_ad)
        inverse_product_al = pl.sub(inverse_work, inverse_product_ah)
        inverse_product_bs = pl.mul(inverse_square_hi, 4097.0)
        inverse_product_bd = pl.sub(inverse_product_bs, inverse_square_hi)
        inverse_product_bh = pl.sub(inverse_product_bs, inverse_product_bd)
        inverse_product_bl = pl.sub(inverse_square_hi, inverse_product_bh)
        inverse_product_hi = pl.mul(inverse_work, inverse_square_hi)
        inverse_product_hh = pl.mul(inverse_product_ah, inverse_product_bh)
        inverse_product_e0 = pl.sub(inverse_product_hh, inverse_product_hi)
        inverse_product_hl = pl.mul(inverse_product_ah, inverse_product_bl)
        inverse_product_e1 = pl.add(inverse_product_e0, inverse_product_hl)
        inverse_product_lh = pl.mul(inverse_product_al, inverse_product_bh)
        inverse_product_e2 = pl.add(inverse_product_e1, inverse_product_lh)
        inverse_product_ll = pl.mul(inverse_product_al, inverse_product_bl)
        inverse_product_lo = pl.add(inverse_product_e2, inverse_product_ll)
        inverse_low_product = pl.mul(inverse_work, inverse_square_lo)
        inverse_product_error = pl.add(inverse_product_lo, inverse_low_product)
        inverse_one = pl.tile.full([1, REDUCE_ROW_TILE], dtype=pl.FP32, value=1.0)
        inverse_residual_hi = pl.sub(inverse_one, inverse_product_hi)
        inverse_residual = pl.sub(inverse_residual_hi, inverse_product_error)
        inverse_half = pl.mul(raw_inverse, 0.5)
        inverse_step = pl.mul(inverse_half, inverse_residual)
        inverse_in_range = pl.cast(0.0, pl.FP32)
        variance_scalar = pl.tile.read(x_mean_eps, [0, 0])
        if variance_scalar <= pl.cast(1099511627776.0, pl.FP32):
            inverse_in_range = pl.cast(1.0, pl.FP32)
        inverse_correction = pl.mul(inverse_step, inverse_in_range)
        x_inv_rms = pl.add(raw_inverse, inverse_correction)
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
    # Use a high-precision statistical reduction with explicit FP32 boundaries.
    square_sum = x.square().double().sum(-1, keepdim=True).float()
    mean_square = square_sum * (1.0 / x.shape[-1])
    variance = mean_square + EPS
    inv = torch.rsqrt(variance.double()).float()
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
        TensorSpec("x_normed", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {"decode": (DECODE_BATCH, DECODE_SEQ)}

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 attention RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true")
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
        result = run(
            fn=rms_norm_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_rms_norm_test,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            config=dict(
                dump_passes=args.dump_passes,
                backend_type=args.platform.removesuffix("sim"),
                execution_mode=args.sim or args.platform.endswith("sim"),
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
