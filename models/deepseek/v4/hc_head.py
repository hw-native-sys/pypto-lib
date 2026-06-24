# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 Hyper-Connections head projection."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_DIM_INV = 1.0 / HC_DIM

HC_PAD = 16
T_TILE = 8
LINEAR_T_TILE = 64
TRANSPOSE_T_TILE = 64  # device-swept: 64/stage=2 minimizes hc_head_pre_fused (Exec 3.7us)
RMS_K_CHUNK = 512
LINEAR_K_CHUNK = 256  # cube K-fragment per matmul call; keeps Mat at 352KB (< 512KB L1)
D_CHUNK = 512
# Head projection hc_head_fn @ x^T is a skinny GEMM (M=T, N=HC_MULT<=HC_PAD=16,
# K=HC_DIM=16384). Two structural choices, both ~10-15% over the original fused
# (UP_DOWN) kernel that was ~151us:
#  - Pure-AIC matmul: a dedicated cast scope streams the BF16 activations to an
#    FP32 x_fp32, so both the matmul and the inv_rms reduce read FP32 directly.
#    The matmul is then a clean cube-only kernel (~86% Exec, no vector subblock)
#    instead of a mixed cube+cast kernel.
#  - Split-K: with one task per token-tile only T/T_TILE cube cores run (8 of
#    ~24). LINEAR_OK splits the K reduction into OK slices that atomic-add their
#    FP32 partials into a zero-seeded mixes_raw, filling the idle cores and
#    shortening each core's matmul_acc chain. OK=2 is the device-sweep peak
#    (16 tasks = one wave, minimal atomic contention); OK=4/8 spill past ~24
#    cores into extra waves where atomic contention overtakes the shorter chain.
# Tile-size tuning can't help (FP32 operands make L1/Mat the wall at K=512);
# split-K is the lever hint_l1_tile ranked #1. Golden-validated; device best-of-5
# median ~128-134us.
LINEAR_OK = 8
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
NUM_T_TILES = T // T_TILE
NUM_LINEAR_T_TILES = T // LINEAR_T_TILE
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK
assert HC_DIM % LINEAR_OK == 0 and LINEAR_K_PER_SPLIT % LINEAR_K_CHUNK == 0
# The 4-way reduce (pre0..pre3 / x_h0..x_h3) is hardcoded for HC_MULT == 4, and the
# fixed-size token tiles must evenly divide T or tail rows are silently dropped.
assert HC_MULT == 4, f"hc_head reduce is hardcoded for HC_MULT == 4, got {HC_MULT}"
assert T % T_TILE == 0 and T % LINEAR_T_TILE == 0 and T % TRANSPOSE_T_TILE == 0, (
    f"T ({T}) must be a multiple of T_TILE ({T_TILE}), LINEAR_T_TILE ({LINEAR_T_TILE}), "
    f"and TRANSPOSE_T_TILE ({TRANSPOSE_T_TILE})"
)

@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Tensor[[T, D], pl.BF16],
):
    x_flat = pl.reshape(x_hc, [T, HC_DIM])
    y_flat = pl.reshape(y, [T, D])
    inv_rms = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_fp32 = pl.create_tensor([T, HC_DIM], dtype=pl.FP32)  # FP32 activations, produced by the RMS scope
    mixes_raw = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    pre_t = pl.create_tensor([HC_PAD, T], dtype=pl.FP32)
    # Cast scope: stream x (BF16) -> x_fp32 (FP32) once. Both the head-projection
    # matmul (pure-AIC) and the inv_rms reduce below consume these FP32 activations.
    for t in pl.spmd(NUM_T_TILES, name_hint="hc_head_cast", allow_early_resolve=True):
        t0 = t * T_TILE
        for kb in pl.pipeline(RMS_K_BLOCKS, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = pl.cast(x_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK], target_type=pl.FP32)
            x_fp32[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK] = x_chunk

    # inv_rms scope: read the FP32 activations back and reduce sum-of-squares -> rsqrt.
    for t in pl.spmd(NUM_T_TILES, name_hint="hc_head_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(RMS_K_BLOCKS, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = x_fp32[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK]
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
            )
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), EPS)), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])

    # Split-K head projection: zero-seed mixes_raw, then dispatch
    # NUM_T_TILES * LINEAR_OK tasks -- each owns one token-tile and one 1/OK
    # K-slice, reduces it, and atomic-adds its [T_TILE, HC_PAD] FP32 partial.
    # Token tiles touch disjoint rows, so only the LINEAR_OK tasks per row block
    # contend; the seed write -> atomic RMW WAW dependency orders seed first.
    for tc in pl.spmd(NUM_T_TILES, name_hint="hc_head_seed", allow_early_resolve=True):
        ts0 = tc * T_TILE
        mixes_raw[ts0 : ts0 + T_TILE, 0:HC_PAD] = pl.full(
            [T_TILE, HC_PAD], dtype=pl.FP32, value=0.0
        )

    for task in pl.spmd(NUM_LINEAR_T_TILES * LINEAR_OK, name_hint="hc_head_linear", allow_early_resolve=True):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        k_base = (task % LINEAR_OK) * LINEAR_K_PER_SPLIT
        acc = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = x_fp32[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_CHUNK]  # pre-cast FP32 -> pure-AIC matmul
            w_chunk = pl.slice(
                hc_head_fn,
                [HC_PAD, LINEAR_K_CHUNK],
                [0, k0],
                valid_shape=[HC_MULT, LINEAR_K_CHUNK],
            )
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)

    # Fused scale + sigmoid + transpose -> pre_t in one scope (one dispatch).
    # Per TRANSPOSE_T_TILE block: row-scale the raw projection by inv_rms, apply
    # sigmoid(mix * scale + base) + HC_EPS, then transpose the [TILE, HC_PAD] block
    # straight into pre_t[:, tile]. The mixes and pre intermediates never touch GM.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_pre_fused", allow_early_resolve=True):
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_PAD], [0], valid_shape=[HC_MULT]), [1, HC_PAD])
        for t0 in pl.pipeline(0, T, TRANSPOSE_T_TILE, stage=2):
            scaled = pl.row_expand_mul(
                mixes_raw[t0 : t0 + TRANSPOSE_T_TILE, 0:HC_PAD],
                inv_rms[t0 : t0 + TRANSPOSE_T_TILE, 0:1],
            )
            logits = pl.add(
                pl.mul(scaled, scale),
                pl.col_expand(scaled, base),
            )
            pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
            pre_t[:, t0 : t0 + TRANSPOSE_T_TILE] = pl.transpose(pre_val, axis1=0, axis2=1)

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_reduce", allow_early_resolve=True):
            # Per head h, pre_t[h] is a contiguous row of per-token scales; slice it
            # and reshape [1, T_TILE] -> [T_TILE, 1] for the row-broadcast multiply.
            pre0 = pl.reshape(pre_t[0:1, t0 : t0 + T_TILE], [T_TILE, 1])
            pre1 = pl.reshape(pre_t[1:2, t0 : t0 + T_TILE], [T_TILE, 1])
            pre2 = pl.reshape(pre_t[2:3, t0 : t0 + T_TILE], [T_TILE, 1])
            pre3 = pl.reshape(pre_t[3:4, t0 : t0 + T_TILE], [T_TILE, 1])
            for db in pl.pipeline(D_BLOCKS, stage=2):
                d0 = db * D_CHUNK
                x_h0 = pl.cast(x_flat[t0 : t0 + T_TILE, 0 * D + d0 : 0 * D + d0 + D_CHUNK], target_type=pl.FP32)
                x_h1 = pl.cast(x_flat[t0 : t0 + T_TILE, 1 * D + d0 : 1 * D + d0 + D_CHUNK], target_type=pl.FP32)
                x_h2 = pl.cast(x_flat[t0 : t0 + T_TILE, 2 * D + d0 : 2 * D + d0 + D_CHUNK], target_type=pl.FP32)
                x_h3 = pl.cast(x_flat[t0 : t0 + T_TILE, 3 * D + d0 : 3 * D + d0 + D_CHUNK], target_type=pl.FP32)
                y_tile = pl.add(
                    pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1)),
                    pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3)),
                )
                y_flat[t0 : t0 + T_TILE, d0 : d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")

    y = pl.reshape(y_flat, [T, D])
    return y


@pl.jit
def hc_head_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    y = hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, y)
    return y


def golden_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    x_flat_2d = x.reshape(T, HC_DIM).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_head_fn[h:h + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(T, HC_MULT)

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

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["y"][:] = _to_device_bf16(y)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_hc():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_head_fn():
        return torch.randn(HC_MULT, HC_DIM) * 0.0519

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [1], torch.float32,
                   init_value=lambda: torch.tensor([0.076099])),
        TensorSpec("hc_head_base", [HC_MULT], torch.float32,
                   init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124])),
        TensorSpec("y", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # Int mode (0=off; 1=timing only, most accurate; 2=timing + dep graph, two runs).
    # `nargs="?"` so a bare `--enable-l2-swimlane` -> mode 1 (int, not bool True).
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=hc_head_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head,
        compile_cfg=dict(
            dump_passes=True,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
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
