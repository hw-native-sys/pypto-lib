# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections pre-mix for prefill attention."""

import pypto.language as pl

from config import FLASH as M, PREFILL_BATCH, PREFILL_SEQ


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps

# kernel-local
MIX_PAD = 32
HC_PAD = 8

# tiling
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4


@pl.jit.inline
def prefill_hc_pre(
    x:        pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
    hc_scale: pl.Tensor[[3],                pl.FP32],
    hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
    x_mixed:  pl.Tensor[[T, D],             pl.BF16],
    post:     pl.Tensor[[T, HC_MULT],       pl.FP32],
    comb:     pl.Tensor[[T, HC_MULT, HC_MULT], pl.FP32],
):
    x_flat = pl.reshape(x, [T, HC_DIM])
    post_2d = post
    comb_flat = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    mixes = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)
    mix_raw = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)
    pre_val_store = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    post_pad_store = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    pre_val_t = pl.create_tensor([HC_PAD, T], dtype=pl.FP32)

    # Official HC pre starts with RMS over the concatenated hidden-choice lanes.
    for t0 in pl.parallel(0, T, RMS_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_rms"):
            sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(RMS_K_BLOCKS, stage=RMS_PIPE_STAGE):
                k0 = kb * RMS_K_CHUNK
                x_chunk = pl.cast(
                    pl.slice(x_flat, [RMS_T_TILE, RMS_K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, RMS_T_TILE]),
                )
            inv_rms_val = pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), EPS), high_precision=True)
            inv_rms = pl.assemble(inv_rms, inv_rms_val, [0, t0])

    # Project normalized HC features to pre/post/comb logits.
    for t0 in pl.parallel(0, T, LINEAR_T_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
            name_hint="prefill_hc_linear",
        ):
            x_lin_0 = pl.cast(
                pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, 0]),
                target_type=pl.FP32,
            )
            w_lin_0 = pl.slice(
                hc_fn,
                [MIX_PAD, LINEAR_K_CHUNK],
                [0, 0],
                valid_shape=[MIX_HC, LINEAR_K_CHUNK],
            )
            mix_acc = pl.matmul(x_lin_0, w_lin_0, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_BLOCKS, stage=2):
                kl0 = kb * LINEAR_K_CHUNK
                x_lin = pl.cast(
                    pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, kl0]),
                    target_type=pl.FP32,
                )
                w_lin = pl.slice(
                    hc_fn,
                    [MIX_PAD, LINEAR_K_CHUNK],
                    [0, kl0],
                    valid_shape=[MIX_HC, LINEAR_K_CHUNK],
                )
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)
            mix_raw = pl.assemble(mix_raw, mix_acc, [t0, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_linear_scale"):
        mixes = pl.assemble(
            mixes,
            pl.row_expand_mul(pl.slice(mix_raw, [T, MIX_PAD], [0, 0]), pl.reshape(inv_rms, [T, 1])),
            [0, 0],
        )

    # Split logits into pre gates, post gates, and the comb Sinkhorn matrix.
    comb_logits = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    for split_t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_split"):
            scale0 = pl.read(hc_scale, [0])
            scale1 = pl.read(hc_scale, [1])
            scale2 = pl.read(hc_scale, [2])

            ones_hc = pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0)
            pre_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [0]), [1, HC_PAD])
            pre_logits = pl.add(
                pl.mul(pl.slice(mixes, [T_TILE, HC_PAD], [split_t0, 0]), scale0),
                pl.col_expand_mul(ones_hc, pre_base),
            )
            pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
            pre_val_store = pl.assemble(pre_val_store, pre_val, [split_t0, 0])

            post_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [HC_MULT]), [1, HC_PAD])
            post_logits = pl.add(
                pl.mul(pl.slice(mixes, [T_TILE, HC_PAD], [split_t0, HC_MULT]), scale1),
                pl.col_expand_mul(ones_hc, post_base),
            )
            post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            post_pad_store = pl.assemble(post_pad_store, post_pad, [split_t0, 0])

            ones_comb = pl.full([T_TILE, HC_MULT * HC_MULT], dtype=pl.FP32, value=1.0)
            comb_base = pl.reshape(
                pl.slice(hc_base, [HC_MULT * HC_MULT], [HC_MULT * 2]),
                [1, HC_MULT * HC_MULT],
            )
            comb_mix = pl.slice(mixes, [T_TILE, HC_MULT * HC_MULT], [split_t0, HC_MULT * 2])
            comb_logits_val = pl.add(
                pl.mul(comb_mix, scale2),
                pl.col_expand_mul(ones_comb, comb_base),
            )
            comb_logits = pl.assemble(comb_logits, comb_logits_val, [split_t0, 0])

    for t0 in pl.parallel(0, T, COMB_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_write_post"):
            post_tile = pl.load(
                post_pad_store,
                [t0, 0],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            post_2d = pl.store(post_tile, [t0, 0], post_2d)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_transpose_pre"):
        for t0 in pl.range(0, T, T_TILE):
            pre_tile = pl.load(
                pre_val_store,
                [t0, 0],
                [T_TILE, HC_PAD],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
            pre_val_t = pl.store(pre_tile_t, [0, t0], pre_val_t)

    for t0 in pl.parallel(0, T, COMB_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_comb_sinkhorn"):
            row0 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 0 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row1 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 1 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row2 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 2 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row3 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 3 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)

            row_max_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_exp = pl.exp(pl.row_expand_sub(row0, pl.row_max(row0, row_max_tmp)))
            row1_exp = pl.exp(pl.row_expand_sub(row1, pl.row_max(row1, row_max_tmp)))
            row2_exp = pl.exp(pl.row_expand_sub(row2, pl.row_max(row2, row_max_tmp)))
            row3_exp = pl.exp(pl.row_expand_sub(row3, pl.row_max(row3, row_max_tmp)))
            row0_soft = pl.add(pl.row_expand_div(row0_exp, pl.row_sum(row0_exp, row_sum_tmp)), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, pl.row_sum(row1_exp, row_sum_tmp)), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, pl.row_sum(row2_exp, row_sum_tmp)), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, pl.row_sum(row3_exp, row_sum_tmp)), HC_EPS)

            row0_eff = pl.fillpad(pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row1_eff = pl.fillpad(pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row2_eff = pl.fillpad(pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row3_eff = pl.fillpad(pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)

            row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_eff, col_sum)
            row1_cur = pl.div(row1_eff, col_sum)
            row2_cur = pl.div(row2_eff, col_sum)
            row3_cur = pl.div(row3_eff, col_sum)

            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row0_norm = pl.row_expand_div(row0_cur, pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS))
                row1_norm = pl.row_expand_div(row1_cur, pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS))
                row2_norm = pl.row_expand_div(row2_cur, pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS))
                row3_norm = pl.row_expand_div(row3_cur, pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS))
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)

            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            comb_flat = pl.store(row0_out, [t0, 0 * HC_MULT], comb_flat)
            comb_flat = pl.store(row1_out, [t0, 1 * HC_MULT], comb_flat)
            comb_flat = pl.store(row2_out, [t0, 2 * HC_MULT], comb_flat)
            comb_flat = pl.store(row3_out, [t0, 3 * HC_MULT], comb_flat)

    # Mix the HC lanes into the model dimension before qkv projection.
    x_mixed_view = x_mixed
    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_mix_x"):
            pre0 = pl.reshape(
                pl.load(pre_val_t, [0, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre1 = pl.reshape(
                pl.load(pre_val_t, [1, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre2 = pl.reshape(
                pl.load(pre_val_t, [2, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre3 = pl.reshape(
                pl.load(pre_val_t, [3, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x0 = pl.cast(
                    pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x1 = pl.cast(
                    pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x2 = pl.cast(
                    pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x3 = pl.cast(
                    pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                y_tile = pl.add(
                    pl.add(pl.row_expand_mul(x0, pre0), pl.row_expand_mul(x1, pre1)),
                    pl.add(pl.row_expand_mul(x2, pre2), pl.row_expand_mul(x3, pre3)),
                )
                x_mixed_view = pl.store(
                    pl.cast(y_tile, target_type=pl.BF16, mode="rint"),
                    [t0, d0],
                    x_mixed_view,
                )
    x_mixed = x_mixed_view
    post = post_2d
    comb = pl.reshape(comb_flat, [T, HC_MULT, HC_MULT])
    return x_mixed, post, comb


def golden_prefill_hc_pre(tensors):
    import torch

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()

    x_token = x.reshape(T, HC_MULT, D)
    x_flat_2d = x_token.reshape(T, HC_DIM)
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes_2d = torch.cat(mix_cols, dim=1)
    mixes = mixes_2d.reshape(B, S, MIX_HC)

    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    pre = torch.sigmoid(pre_logits) + HC_EPS

    post_logits = (
        mixes_2d[:, HC_MULT:HC_MULT + HC_PAD] * hc_scale[1]
        + hc_base[HC_MULT:HC_MULT + HC_PAD]
    )
    post = (2 * torch.sigmoid(post_logits[:, :HC_MULT])).reshape(B, S, HC_MULT)

    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]).view(
        B, S, HC_MULT, HC_MULT
    )
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    pre_2d = pre.reshape(T, HC_MULT)
    y01 = x_token[:, 0, :] * pre_2d[:, 0:1] + x_token[:, 1, :] * pre_2d[:, 1:2]
    y23 = x_token[:, 2, :] * pre_2d[:, 2:3] + x_token[:, 3, :] * pre_2d[:, 3:4]
    y = y01 + y23

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y).reshape_as(tensors["x_mixed"])
    tensors["post"][:] = post.reshape_as(tensors["post"])
    tensors["comb"][:] = comb_t.reshape_as(tensors["comb"])


@pl.jit
def prefill_hc_pre_test(
    x:        pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_fn:    pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base:  pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed:  pl.Out[pl.Tensor[[T, D], pl.BF16]],
    post:     pl.Out[pl.Tensor[[T, HC_MULT], pl.FP32]],
    comb:     pl.Out[pl.Tensor[[T, HC_MULT, HC_MULT], pl.FP32]],
):
    return prefill_hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(T, HC_MULT, D) * 0.05
    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_scale():
        return torch.ones(3) * 0.5
    def init_hc_base():
        return torch.zeros(MIX_HC)

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16, is_output=True),
        TensorSpec("post", [T, HC_MULT], torch.float32, is_output=True),
        TensorSpec("comb", [T, HC_MULT, HC_MULT], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill HC pre validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_hc_pre_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_hc_pre,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "post": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "comb": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
