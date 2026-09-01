# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections head projection."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_DIM_INV = 1.0 / HC_DIM

# tiling
T_TILE = 8
MIX_K_TILE = 512
D_TILE = 512
MIX_SPLIT_TILE = 4 * MIX_K_TILE
D_SPLIT_TILE = 2 * D_TILE
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0

# packed statistics layout
# Packed stats row: [sum(x^2) | mix_0 | mix_1 | mix_2 | mix_3], T_TILE columns each.
STAT_COLS = (HC_MULT + 1) * T_TILE


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    stats = pl.create_tensor([(T_MAX // T_TILE) * (HC_DIM // MIX_SPLIT_TILE), STAT_COLS], dtype=pl.FP32)

    for task in pl.spmd(
        (t_dim // T_TILE) * (HC_DIM // MIX_SPLIT_TILE),
        name_hint="hc_head_mix",
    ):
        tt = task // (HC_DIM // MIX_SPLIT_TILE)
        split = task - tt * (HC_DIM // MIX_SPLIT_TILE)
        t0 = tt * T_TILE
        k_base = split * MIX_SPLIT_TILE
        acc = pl.full([1, STAT_COLS], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(k_base, k_base + MIX_SPLIT_TILE, MIX_K_TILE, stage=4):
            x_chunk = x_flat[t0 : t0 + T_TILE, k0 : k0 + MIX_K_TILE]
            w0 = hc_head_fn[0:1, k0 : k0 + MIX_K_TILE]
            w1 = hc_head_fn[1:2, k0 : k0 + MIX_K_TILE]
            w2 = hc_head_fn[2:3, k0 : k0 + MIX_K_TILE]
            w3 = hc_head_fn[3:4, k0 : k0 + MIX_K_TILE]
            x_sq = pl.mul(x_chunk, x_chunk)
            x_sq_sum = pl.row_sum(x_sq)
            x_sq_row = pl.reshape(x_sq_sum, [1, T_TILE])
            mix0_weighted = pl.col_expand_mul(x_chunk, w0)
            mix0_sum = pl.row_sum(mix0_weighted)
            mix0_row = pl.reshape(mix0_sum, [1, T_TILE])
            stats01 = pl.concat(x_sq_row, mix0_row)
            mix1_weighted = pl.col_expand_mul(x_chunk, w1)
            mix1_sum = pl.row_sum(mix1_weighted)
            mix1_row = pl.reshape(mix1_sum, [1, T_TILE])
            mix2_weighted = pl.col_expand_mul(x_chunk, w2)
            mix2_sum = pl.row_sum(mix2_weighted)
            mix2_row = pl.reshape(mix2_sum, [1, T_TILE])
            stats12 = pl.concat(mix1_row, mix2_row)
            mix3_weighted = pl.col_expand_mul(x_chunk, w3)
            mix3_sum = pl.row_sum(mix3_weighted)
            mix3_row = pl.reshape(mix3_sum, [1, T_TILE])
            stats123 = pl.concat(stats12, mix3_row)
            chunk_stats = pl.concat(stats01, stats123)
            acc = pl.add(acc, chunk_stats)
        stats_row = tt * (HC_DIM // MIX_SPLIT_TILE) + split
        stats[stats_row : stats_row + 1, 0:STAT_COLS] = acc

    y_flat = pl.reshape(y, [t_dim, D])
    for task in pl.spmd((t_dim // T_TILE) * (D // D_SPLIT_TILE), name_hint="hc_head_reduce"):
        tt = task // (D // D_SPLIT_TILE)
        d_split = task - tt * (D // D_SPLIT_TILE)
        t0 = tt * T_TILE
        s0 = tt * (HC_DIM // MIX_SPLIT_TILE)
        total = pl.col_sum(stats[s0 : s0 + (HC_DIM // MIX_SPLIT_TILE), 0:STAT_COLS])
        total_sq = total[0:1, 0:T_TILE]
        total_sq_mean = pl.mul(total_sq, HC_DIM_INV)
        total_sq_eps = pl.add(total_sq_mean, EPS)
        inv = pl.rsqrt(total_sq_eps, high_precision=True)
        scale = pl.read(hc_head_scale, [0])
        mix0 = total[0:1, 1 * T_TILE : 2 * T_TILE]
        mix1 = total[0:1, 2 * T_TILE : 3 * T_TILE]
        mix2 = total[0:1, 3 * T_TILE : 4 * T_TILE]
        mix3 = total[0:1, 4 * T_TILE : 5 * T_TILE]
        mix0_norm = pl.mul(mix0, inv)
        mix0_scaled = pl.mul(mix0_norm, scale)
        base0 = pl.read(hc_head_base, [0])
        logit0 = pl.add(mix0_scaled, base0)
        mix1_norm = pl.mul(mix1, inv)
        mix1_scaled = pl.mul(mix1_norm, scale)
        base1 = pl.read(hc_head_base, [1])
        logit1 = pl.add(mix1_scaled, base1)
        mix2_norm = pl.mul(mix2, inv)
        mix2_scaled = pl.mul(mix2_norm, scale)
        base2 = pl.read(hc_head_base, [2])
        logit2 = pl.add(mix2_scaled, base2)
        mix3_norm = pl.mul(mix3, inv)
        mix3_scaled = pl.mul(mix3_norm, scale)
        base3 = pl.read(hc_head_base, [3])
        logit3 = pl.add(mix3_scaled, base3)
        neg0 = pl.neg(logit0)
        exp0 = pl.exp(neg0)
        denom0 = pl.add(exp0, 1.0)
        sigmoid0 = pl.recip(denom0)
        pre0_eps = pl.add(sigmoid0, HC_EPS)
        pre0 = pl.reshape(pre0_eps, [T_TILE, 1])
        neg1 = pl.neg(logit1)
        exp1 = pl.exp(neg1)
        denom1 = pl.add(exp1, 1.0)
        sigmoid1 = pl.recip(denom1)
        pre1_eps = pl.add(sigmoid1, HC_EPS)
        pre1 = pl.reshape(pre1_eps, [T_TILE, 1])
        neg2 = pl.neg(logit2)
        exp2 = pl.exp(neg2)
        denom2 = pl.add(exp2, 1.0)
        sigmoid2 = pl.recip(denom2)
        pre2_eps = pl.add(sigmoid2, HC_EPS)
        pre2 = pl.reshape(pre2_eps, [T_TILE, 1])
        neg3 = pl.neg(logit3)
        exp3 = pl.exp(neg3)
        denom3 = pl.add(exp3, 1.0)
        sigmoid3 = pl.recip(denom3)
        pre3_eps = pl.add(sigmoid3, HC_EPS)
        pre3 = pl.reshape(pre3_eps, [T_TILE, 1])
        d_base = d_split * D_SPLIT_TILE
        for d0 in pl.pipeline(d_base, d_base + D_SPLIT_TILE, D_TILE, stage=2):
            x_h0 = x_flat[t0 : t0 + T_TILE, 0 * D + d0 : 0 * D + d0 + D_TILE]
            x_h1 = x_flat[t0 : t0 + T_TILE, 1 * D + d0 : 1 * D + d0 + D_TILE]
            x_h2 = x_flat[t0 : t0 + T_TILE, 2 * D + d0 : 2 * D + d0 + D_TILE]
            x_h3 = x_flat[t0 : t0 + T_TILE, 3 * D + d0 : 3 * D + d0 + D_TILE]
            y0 = pl.row_expand_mul(x_h0, pre0)
            y1 = pl.row_expand_mul(x_h1, pre1)
            y01 = pl.add(y0, y1)
            y2 = pl.row_expand_mul(x_h2, pre2)
            y3 = pl.row_expand_mul(x_h3, pre3)
            y23 = pl.add(y2, y3)
            y_tile = pl.add(y01, y23)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16

    y = pl.reshape(y_flat, [t_dim, D])
    return y


@pl.jit
def hc_head_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    y = hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, y)
    return y


def hc_head_stats_golden(x_flat_2d, hc_head_fn):
    """Compute split-K sum-of-squares and raw mixes in kernel order."""
    import torch

    rows = x_flat_2d.shape[0]
    sq_sum = torch.zeros(rows, 1, dtype=torch.float32)
    mixes = torch.zeros(rows, HC_MULT, dtype=torch.float32)
    for split in range(HC_DIM // MIX_SPLIT_TILE):
        k_base = split * MIX_SPLIT_TILE
        sq_split = torch.zeros(rows, 1, dtype=torch.float32)
        mix_split = torch.zeros(rows, HC_MULT, dtype=torch.float32)
        for k0 in range(k_base, k_base + MIX_SPLIT_TILE, MIX_K_TILE):
            x_chunk = x_flat_2d[:, k0:k0 + MIX_K_TILE]
            sq_split += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
            for h in range(HC_MULT):
                w_chunk = hc_head_fn[h:h + 1, k0:k0 + MIX_K_TILE]
                mix_split[:, h:h + 1] += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        sq_sum += sq_split
        mixes += mix_split
    return sq_sum, mixes


def golden_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    x_flat_2d = x.reshape(T, HC_DIM).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum, mixes_raw = hc_head_stats_golden(x_flat_2d, hc_head_fn)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)
    mixes = mixes_raw * rsqrt

    pre = torch.sigmoid(mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()) + HC_EPS
    x_view = x.float().view(shape)
    if HC_MULT == 4:
        y = (
            x_view[:, 0, :] * pre[:, 0:1]
            + x_view[:, 1, :] * pre[:, 1:2]
        ) + (
            x_view[:, 2, :] * pre[:, 2:3]
            + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(T, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h:h + 1]

    tensors["y"][:] = y.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_hc():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_head_fn():
        return torch.randn(HC_MULT, HC_DIM) * 0.0519

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [1], torch.float32, init_value=lambda: torch.tensor([0.076099])),
        TensorSpec(
            "hc_head_base", [HC_MULT], torch.float32,
            init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124]),
        ),
        TensorSpec("y", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run(
        fn=hc_head_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "y": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


def golden_hc_head_rows(tensors):
    """Compute the Hyper-Connections head reference for the input row count."""
    import torch

    x = tensors["x_hc"]
    rows = x.shape[0]
    hc_dim = M.hc_dim
    x_flat_2d = x.reshape(rows, hc_dim).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum, mixes_raw = hc_head_stats_golden(x_flat_2d, hc_head_fn)
    rsqrt = torch.rsqrt(sq_sum * (1.0 / hc_dim) + M.rms_norm_eps)
    mixes = mixes_raw * rsqrt

    logits = mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()
    pre = torch.sigmoid(logits) + M.hc_eps
    x_view = x.float()
    if HC_MULT == 4:
        y = (
            x_view[:, 0, :] * pre[:, 0:1]
            + x_view[:, 1, :] * pre[:, 1:2]
        ) + (
            x_view[:, 2, :] * pre[:, 2:3]
            + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(rows, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h:h + 1]

    tensors["y"][:] = y.to(torch.bfloat16)
