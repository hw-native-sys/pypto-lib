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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_DIM_INV = 1.0 / HC_DIM

# tiling
HC_PAD = 16
HC_VPAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
RMS_K_TILE = 512
LINEAR_K_TILE = 256
D_TILE = 512
D_SPMD = 512
LINEAR_OK = 16
RMS_OK = 16


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = pl.max(t_dim, LINEAR_T_TILE)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    # rms: split-K sum-of-squares, fanned over (token-tile x K-slice)
    sq_part = pl.create_tensor([RMS_OK, t_dim], dtype=pl.FP32)
    for task in pl.spmd((t_dim // T_TILE) * RMS_OK, name_hint="hc_head_rms"):
        t0 = (task // RMS_OK) * T_TILE
        ok = task % RMS_OK
        k_base = ok * (HC_DIM // RMS_OK)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_OK // RMS_K_TILE, stage=4):
            k0 = k_base + kb * RMS_K_TILE
            x_rms = x_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_TILE]
            sq_col = pl.row_sum(pl.mul(x_rms, x_rms))
            sq_sum = pl.add(sq_sum, pl.reshape(sq_col, [1, T_TILE]))
        sq_part = pl.assemble(sq_part, sq_sum, [ok, t0])

    # linear: split-K head projection, fanned over (row-block x K-slice); each task
    # atomic-adds its [LINEAR_T_TILE, HC_PAD] FP32 partial into the AICPU-zeroed mixes_raw
    mixes_raw = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32, init_value=0)
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_head_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        k_base = (task % LINEAR_OK) * (HC_DIM // LINEAR_OK)
        acc = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_OK // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            x_lin = x_flat[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
            w = pl.slice(hc_head_fn, [HC_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[HC_MULT, LINEAR_K_TILE])
            if kb == 0:
                acc = pl.matmul(x_lin, w, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_lin, w, b_trans=True)
        mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)

    # reduce: gate + hc mix, fanned over (token-tile x D-slice). The rsqrt/sigmoid gate is
    # recomputed per task instead of being published by its own scope.
    for blk in pl.spmd((t_dim // T_TILE) * (D // D_SPMD), name_hint="hc_head_reduce"):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_VPAD], [0], valid_shape=[HC_MULT]), [1, HC_VPAD])
        # sq_part is [RMS_OK, T], so the per-token total is a column sum; transposing the
        # slab makes it one row_sum landing in the [T_TILE, 1] shape row_expand_mul wants
        ssq_slab = pl.set_validshape(sq_part[0:RMS_OK, t0 : t0 + T_TILE], RMS_OK, T_TILE)
        ssq = pl.row_sum(pl.transpose(ssq_slab, axis1=0, axis2=1))
        inv_row = pl.rsqrt(pl.add(pl.mul(ssq, HC_DIM_INV), EPS), high_precision=True)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])
        mix = mixes_raw[t0 : t0 + T_TILE, 0:HC_VPAD]
        scaled = pl.mul(pl.row_expand_mul(mix, inv_col), scale)
        logits = pl.add(scaled, pl.col_expand(scaled, base))
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        # pin the valid shape: the gate carries valid=?x? from the dynamic-offset slice
        pre_val = pl.set_validshape(pre_val, T_TILE, HC_VPAD)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x_h0 = x_flat[t0 : t0 + T_TILE, 0 * D + d0 : 0 * D + d0 + D_TILE]
            x_h1 = x_flat[t0 : t0 + T_TILE, 1 * D + d0 : 1 * D + d0 + D_TILE]
            x_h2 = x_flat[t0 : t0 + T_TILE, 2 * D + d0 : 2 * D + d0 + D_TILE]
            x_h3 = x_flat[t0 : t0 + T_TILE, 3 * D + d0 : 3 * D + d0 + D_TILE]
            y01 = pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1))
            y23 = pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3))
            y_tile = pl.add(y01, y23)
            y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")

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


def golden_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    x_flat_2d = x.reshape(T, HC_DIM).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_TILE):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_TILE]
            w_chunk = hc_head_fn[h:h + 1, k0:k0 + LINEAR_K_TILE]
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
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
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
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=hc_head_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
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
