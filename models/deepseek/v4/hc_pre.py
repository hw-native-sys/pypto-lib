# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections pre-mix (decode): mixes the hc-stack into a single sublayer input
and produces the post/comb weights used by hc_post."""


import pypto.language as pl

from config import DEMO as M, DECODE_BATCH, DECODE_SEQ


# model config
B                = DECODE_BATCH
S                = DECODE_SEQ
T                = B * S
D                = M.hidden_size
HC_MULT          = M.hc_mult
MIX_HC           = M.mix_hc
HC_DIM           = M.hc_dim
HC_DIM_INV       = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS           = M.hc_eps
NORM_EPS         = M.rms_norm_eps

# kernel-local
MIX_PAD          = 32       # MIX_HC padded for vector ops
HC_PAD           = 8        # HC_MULT padded
NEG_INF          = -1e20

# tiling
T_TILE           = 16
K_CHUNK          = 512
D_CHUNK          = 512
HC_DIM_BLOCKS    = HC_DIM // K_CHUNK
D_BLOCKS         = D // D_CHUNK


@pl.jit.inline
def hc_pre(
    x:        pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
    hc_scale: pl.Tensor[[3],                pl.FP32],
    hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
    x_mixed:  pl.Tensor[[B, S, D],            pl.BF16],
    post:     pl.Tensor[[B, S, HC_MULT],      pl.FP32],
    comb:     pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
):
    x_flat = pl.reshape(x, [T, HC_DIM])
    post_flat = pl.reshape(post, [T * HC_MULT])
    comb_flat = pl.reshape(comb, [T * HC_MULT * HC_MULT])
    x_flat_fp32 = pl.create_tensor([T, HC_DIM], dtype=pl.FP32)
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    mixes = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)

    for kb in pl.parallel(HC_DIM_BLOCKS):
        k0 = kb * K_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_x"):
            x_chunk_fp32 = pl.cast(
                pl.slice(x_flat, [T, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            x_flat_fp32 = pl.assemble(x_flat_fp32, x_chunk_fp32, [0, k0])

    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="rms"):
        sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM_BLOCKS, stage=4):
            k0 = kb * K_CHUNK
            x_chunk = pl.slice(x_flat_fp32, [T, K_CHUNK], [0, k0])
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]),
            )
        inv_rms_val = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)))
        inv_rms = pl.assemble(inv_rms, inv_rms_val, [0, 0])

    mixes_flat = pl.reshape(mixes, [T * MIX_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="linear"):
        for m in pl.range(MIX_HC):
            mix_col = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HC_DIM_BLOCKS):
                k0 = kb * K_CHUNK
                x_lin_chunk = pl.slice(x_flat_fp32, [T, K_CHUNK], [0, k0])
                w_row = pl.slice(hc_fn, [1, K_CHUNK], [m, k0])
                prod = pl.col_expand_mul(x_lin_chunk, w_row)
                mix_col = pl.add(
                    mix_col,
                    pl.reshape(pl.row_sum(prod), [1, T]),
                )
            mix_col = pl.mul(mix_col, inv_rms)
            mix_col_flat = pl.reshape(mix_col, [T])
            for t in pl.unroll(T):
                pl.write(
                    mixes_flat,
                    [t * MIX_PAD + m],
                    pl.read(mix_col_flat, [t]),
                )
    mixes = pl.reshape(mixes_flat, [T, MIX_PAD])

    comb_logits = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="split_pre_post"):
        scale0 = pl.tensor.read(hc_scale, [0])
        scale1 = pl.tensor.read(hc_scale, [1])
        scale2 = pl.tensor.read(hc_scale, [2])

        ones_hc = pl.full([T, HC_PAD], dtype=pl.FP32, value=1.0)
        pre_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [0]), [1, HC_PAD])
        pre_logits = pl.add(
            pl.mul(pl.slice(mixes, [T, HC_PAD], [0, 0], valid_shape=[T, HC_MULT]), scale0),
            pl.col_expand_mul(ones_hc, pre_base),
        )
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)

        post_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [HC_MULT]), [1, HC_PAD])
        post_logits = pl.add(
            pl.mul(pl.slice(mixes, [T, HC_PAD], [0, HC_MULT]), scale1),
            pl.col_expand_mul(ones_hc, post_base),
        )
        post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)

        ones_comb = pl.full([T, HC_MULT * HC_MULT], dtype=pl.FP32, value=1.0)
        comb_base = pl.reshape(
            pl.slice(hc_base, [HC_MULT * HC_MULT], [HC_MULT * 2]),
            [1, HC_MULT * HC_MULT],
        )
        comb_mix = pl.slice(mixes, [T, HC_MULT * HC_MULT], [0, HC_MULT * 2])
        comb_logits_val = pl.add(
            pl.mul(comb_mix, scale2),
            pl.col_expand_mul(ones_comb, comb_base),
        )
        comb_logits = pl.assemble(comb_logits, comb_logits_val, [0, 0])

    post_pad_flat = pl.reshape(post_pad, [T * HC_PAD])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="comb_sinkhorn"):
        row0 = pl.fillpad(pl.load(
            comb_logits,
            [0, 0 * HC_MULT],
            [T, HC_PAD],
            valid_shapes=[T, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        ), pad_value=pl.PadValue.min)
        row1 = pl.fillpad(pl.load(
            comb_logits,
            [0, 1 * HC_MULT],
            [T, HC_PAD],
            valid_shapes=[T, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        ), pad_value=pl.PadValue.min)
        row2 = pl.fillpad(pl.load(
            comb_logits,
            [0, 2 * HC_MULT],
            [T, HC_PAD],
            valid_shapes=[T, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        ), pad_value=pl.PadValue.min)
        row3 = pl.fillpad(pl.load(
            comb_logits,
            [0, 3 * HC_MULT],
            [T, HC_PAD],
            valid_shapes=[T, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        ), pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_exp = pl.exp(pl.row_expand_sub(row0, pl.row_max(row0, row_max_tmp)))
        row1_exp = pl.exp(pl.row_expand_sub(row1, pl.row_max(row1, row_max_tmp)))
        row2_exp = pl.exp(pl.row_expand_sub(row2, pl.row_max(row2, row_max_tmp)))
        row3_exp = pl.exp(pl.row_expand_sub(row3, pl.row_max(row3, row_max_tmp)))
        row0_soft = pl.add(pl.row_expand_div(row0_exp, pl.row_sum(row0_exp, row_sum_tmp)), HC_EPS)
        row1_soft = pl.add(pl.row_expand_div(row1_exp, pl.row_sum(row1_exp, row_sum_tmp)), HC_EPS)
        row2_soft = pl.add(pl.row_expand_div(row2_exp, pl.row_sum(row2_exp, row_sum_tmp)), HC_EPS)
        row3_soft = pl.add(pl.row_expand_div(row3_exp, pl.row_sum(row3_exp, row_sum_tmp)), HC_EPS)

        row0_eff = pl.tile.fillpad(pl.tile.set_validshape(row0_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
        row1_eff = pl.tile.fillpad(pl.tile.set_validshape(row1_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
        row2_eff = pl.tile.fillpad(pl.tile.set_validshape(row2_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
        row3_eff = pl.tile.fillpad(pl.tile.set_validshape(row3_soft, T, HC_MULT), pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

        for t in pl.unroll(T):
            for c in pl.unroll(HC_MULT):
                pl.write(
                    comb_flat,
                    [t * HC_MULT * HC_MULT + 0 * HC_MULT + c],
                    pl.read(row0_cur, [t, c]),
                )
                pl.write(
                    comb_flat,
                    [t * HC_MULT * HC_MULT + 1 * HC_MULT + c],
                    pl.read(row1_cur, [t, c]),
                )
                pl.write(
                    comb_flat,
                    [t * HC_MULT * HC_MULT + 2 * HC_MULT + c],
                    pl.read(row2_cur, [t, c]),
                )
                pl.write(
                    comb_flat,
                    [t * HC_MULT * HC_MULT + 3 * HC_MULT + c],
                    pl.read(row3_cur, [t, c]),
                )

    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="write_post"):
        for t in pl.parallel(0, T, 1, chunk=16):
            for h in pl.unroll(HC_MULT):
                pl.write(
                    post_flat,
                    [t * HC_MULT + h],
                    pl.read(post_pad_flat, [t * HC_PAD + h]),
                )

    pre_val_flat = pl.reshape(pre_val, [T * HC_PAD])
    x_mixed_view = pl.reshape(x_mixed, [T, D])
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="mix_x"):
        for t in pl.parallel(0, T, 1, chunk=16):
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                y_row = pl.full([1, D_CHUNK], dtype=pl.FP32, value=0.0)
                for h in pl.range(HC_MULT):
                    pre_th = pl.read(pre_val_flat, [t * HC_PAD + h])
                    x_row = pl.slice(x_flat_fp32, [1, D_CHUNK], [t, h * D + d0])
                    y_row = pl.add(y_row, pl.mul(x_row, pre_th))
                x_mixed_view = pl.assemble(
                    x_mixed_view,
                    pl.cast(y_row, target_type=pl.BF16),
                    [t, d0],
                )
    x_mixed = pl.reshape(x_mixed_view, [B, S, D])
    return x_mixed

@pl.jit
def hc_pre_test(
    x:        pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
    hc_scale: pl.Tensor[[3],                pl.FP32],
    hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
    x_mixed:  pl.Out[pl.Tensor[[B, S, D],            pl.BF16]],
    post:     pl.Out[pl.Tensor[[B, S, HC_MULT],      pl.FP32]],
    comb:     pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
):
    x_mixed = hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed

def golden_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre 674-682 + hc_split_sinkhorn."""
    import torch

    x        = tensors["x"].float()                        # [B, S, hc, D]
    hc_fn    = tensors["hc_fn"].float()                    # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()                 # [3]
    hc_base  = tensors["hc_base"].float()                  # [mix_hc]

    shape = x.size()
    x_flat = x.flatten(2)                                  # [B, S, hc*D]
    x_flat_2d = x_flat.reshape(T, HC_DIM)

    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(B, S, MIX_HC)  # [B, S, mix_hc]

    # hc_split_sinkhorn (port of kernel.py 372-427)
    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(*mixes.shape[:-1], HC_MULT, HC_MULT)

    # First step: row-softmax then col-normalize, with eps after softmax
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    # Sinkhorn iterations
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = torch.zeros(B, S, D, dtype=torch.float32)
    for h in range(HC_MULT):
        y += x[:, :, h, :] * pre[:, :, h:h + 1]

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y)
    tensors["post"][:]    = post_t
    tensors["comb"][:]    = comb_t


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.rand(B, S, HC_MULT, D) - 0.5
    def init_hc_fn():
        return (torch.randn(MIX_HC, HC_DIM) - 0.5) / (HC_DIM ** 0.5)
    def init_hc_scale():
        return torch.ones(3) * 0.5
    def init_hc_base():
        return torch.zeros(MIX_HC)

    return [
        TensorSpec("x",        [B, S, HC_MULT, D],       torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn",    [MIX_HC, HC_DIM],         torch.float32,  init_value=init_hc_fn),
        TensorSpec("hc_scale", [3],                      torch.float32,  init_value=init_hc_scale),
        TensorSpec("hc_base",  [MIX_HC],                 torch.float32,  init_value=init_hc_base),
        TensorSpec("x_mixed",  [B, S, D],                torch.bfloat16, is_output=True),
        TensorSpec("post",     [B, S, HC_MULT],          torch.float32,  is_output=True),
        TensorSpec("comb",     [B, S, HC_MULT, HC_MULT], torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=hc_pre_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_pre,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
