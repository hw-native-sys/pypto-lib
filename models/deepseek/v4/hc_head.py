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
T_TILE = 16
RMS_K_CHUNK = 512
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK

@pl.jit
def hc_head(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
):
    x_flat = pl.reshape(x_hc, [T, HC_DIM])
    y_flat = pl.reshape(y, [T, D])
    inv_rms = pl.create_tensor([T, 1], dtype=pl.FP32)
    mixes = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    pre = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    pre_t = pl.create_tensor([HC_PAD, T], dtype=pl.FP32)

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_rms"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(RMS_K_BLOCKS, stage=2):
                k0 = kb * RMS_K_CHUNK
                x_chunk = pl.cast(x_flat[t0 : t0 + T_TILE, k0 : k0 + RMS_K_CHUNK], target_type=pl.FP32)
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
                )
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), EPS)), [T_TILE, 1])
            inv_rms = pl.assemble(inv_rms, inv, [t0, 0])

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
            name_hint="hc_head_linear",
        ):
            x0 = pl.cast(x_flat[t0 : t0 + T_TILE, 0:LINEAR_K_CHUNK], target_type=pl.FP32)
            w0 = pl.slice(
                hc_head_fn,
                [HC_PAD, LINEAR_K_CHUNK],
                [0, 0],
                valid_shape=[HC_MULT, LINEAR_K_CHUNK],
            )
            acc = pl.matmul(x0, w0, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_BLOCKS, stage=2):
                k0 = kb * LINEAR_K_CHUNK
                x_chunk = pl.cast(x_flat[t0 : t0 + T_TILE, k0 : k0 + LINEAR_K_CHUNK], target_type=pl.FP32)
                w_chunk = pl.slice(
                    hc_head_fn,
                    [HC_PAD, LINEAR_K_CHUNK],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_CHUNK],
                )
                acc = pl.matmul_acc(acc, x_chunk, w_chunk, b_trans=True)
            scaled = pl.row_expand_mul(acc, inv_rms[t0 : t0 + T_TILE, 0:1])
            mixes = pl.assemble(mixes, scaled, [t0, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_pre"):
        scale = pl.tensor.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_PAD], [0], valid_shape=[HC_MULT]), [1, HC_PAD])
        logits = pl.add(
            pl.mul(pl.slice(mixes, [T, HC_PAD], [0, 0]), scale),
            pl.col_expand_mul(pl.full([T, HC_PAD], dtype=pl.FP32, value=1.0), base),
        )
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        pre = pl.assemble(pre, pre_val, [0, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_transpose_pre"):
        for t0 in pl.range(0, T, T_TILE):
            pre_tile = pl.load(
                pre,
                [t0, 0],
                [T_TILE, HC_PAD],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_t = pl.store(pl.transpose(pre_tile, axis1=0, axis2=1), [0, t0], pre_t)

    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_head_reduce"):
            pre0 = pl.reshape(
                pl.load(pre_t, [0, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre1 = pl.reshape(
                pl.load(pre_t, [1, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre2 = pl.reshape(
                pl.load(pre_t, [2, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre3 = pl.reshape(
                pl.load(pre_t, [3, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x_h0 = pl.cast(
                    pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x_h1 = pl.cast(
                    pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x_h2 = pl.cast(
                    pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x_h3 = pl.cast(
                    pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                y_tile = pl.add(
                    pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1)),
                    pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3)),
                )
                y_flat = pl.store(pl.cast(y_tile, target_type=pl.BF16, mode="rint"), [t0, d0], y_flat)

    y = pl.reshape(y_flat, [B, S, D])
    return y


def golden_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    x_flat = x.flatten(2)
    x_flat_2d = x_flat.reshape(T, HC_DIM).float()
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
    mixes = torch.cat(mix_cols, dim=1).reshape(B, S, HC_MULT)

    pre = torch.sigmoid(mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()) + HC_EPS
    x_view = x.float().view(shape)
    if HC_MULT == 4:
        y = (
            x_view[:, :, 0, :] * pre[:, :, 0:1]
            + x_view[:, :, 1, :] * pre[:, :, 1:2]
        ) + (
            x_view[:, :, 2, :] * pre[:, :, 2:3]
            + x_view[:, :, 3, :] * pre[:, :, 3:4]
        )
    else:
        y = torch.zeros(B, S, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, :, h, :] * pre[:, :, h:h + 1]

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["y"][:] = _to_device_bf16(y)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_hc():
        return torch.rand(B, S, HC_MULT, D) - 0.5

    def init_hc_head_fn():
        return (torch.randn(HC_MULT, HC_DIM) - 0.5) / HC_DIM ** 0.5

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [1], torch.float32, init_value=lambda: torch.ones(1) * 0.5),
        TensorSpec("hc_head_base", [HC_MULT], torch.float32, init_value=lambda: torch.zeros(HC_MULT)),
        TensorSpec("y", [B, S, D], torch.bfloat16, is_output=True),
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
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=hc_head,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_head,
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
