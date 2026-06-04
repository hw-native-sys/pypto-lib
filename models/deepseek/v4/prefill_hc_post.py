# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix for prefill attention."""

import pypto.language as pl

from config import FLASH as M, PREFILL_BATCH, PREFILL_SEQ


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

# tiling
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK
HC_POST_TOKEN_TILE = 16
HC_POST_TOKEN_BLOCKS = T // HC_POST_TOKEN_TILE
HC_POST_SPMD_BLOCKS = HC_MULT * HC_POST_TOKEN_BLOCKS * D_BLOCKS
assert T % HC_POST_TOKEN_TILE == 0, "T must be divisible by HC_POST_TOKEN_TILE"


@pl.jit.inline
def prefill_hc_post_packed(
    x:        pl.Tensor[[T, D], pl.BF16],
    residual: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    post:     pl.Tensor[[T, HC_MULT], pl.FP32],
    comb:     pl.Tensor[[T, HC_MULT, HC_MULT], pl.FP32],
    y:        pl.Tensor[[T, HC_MULT, D], pl.BF16],
):
    x_flat = x
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    post_t = post
    comb_t = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    y_flat = pl.reshape(y, [T, HC_DIM])

    for block in pl.spmd(HC_POST_SPMD_BLOCKS, name_hint="prefill_hc_post"):
        db = block % D_BLOCKS
        token_block_and_h = block // D_BLOCKS
        token_block = token_block_and_h % HC_POST_TOKEN_BLOCKS
        out_h = token_block_and_h // HC_POST_TOKEN_BLOCKS
        t0 = token_block * HC_POST_TOKEN_TILE
        d0 = db * D_CHUNK

        for dt in pl.range(HC_POST_TOKEN_TILE):
            t = t0 + dt
            post_w = pl.read(post_t, [t, out_h])
            comb0_w = pl.read(comb_t, [t, 0 * HC_MULT + out_h])
            comb1_w = pl.read(comb_t, [t, 1 * HC_MULT + out_h])
            comb2_w = pl.read(comb_t, [t, 2 * HC_MULT + out_h])
            comb3_w = pl.read(comb_t, [t, 3 * HC_MULT + out_h])
            x_row = pl.cast(
                pl.slice(x_flat, [1, D_CHUNK], [t, d0]),
                target_type=pl.FP32,
            )
            y_row = pl.mul(x_row, post_w)
            residual0 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 0 * D + d0]),
                target_type=pl.FP32,
            )
            residual1 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 1 * D + d0]),
                target_type=pl.FP32,
            )
            residual2 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 2 * D + d0]),
                target_type=pl.FP32,
            )
            residual3 = pl.cast(
                pl.slice(residual_flat, [1, D_CHUNK], [t, 3 * D + d0]),
                target_type=pl.FP32,
            )
            y_row = pl.add(y_row, pl.mul(residual0, comb0_w))
            y_row = pl.add(y_row, pl.mul(residual1, comb1_w))
            y_row = pl.add(y_row, pl.mul(residual2, comb2_w))
            y_row = pl.add(y_row, pl.mul(residual3, comb3_w))
            y_flat[
                t:t + 1,
                out_h * D + d0:out_h * D + d0 + D_CHUNK,
            ] = pl.cast(y_row, target_type=pl.BF16, mode="rint")
    y = pl.reshape(y_flat, [T, HC_MULT, D])
    return y


@pl.jit.inline
def prefill_hc_post(
    x:               pl.Tensor[[B, S, D],             pl.BF16],
    residual:        pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
    post:            pl.Tensor[[B, S, HC_MULT],       pl.FP32],
    comb:            pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y:               pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
):
    x_packed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_packed = pl.reshape(x, [T, D])
    residual_packed = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    residual_packed = pl.reshape(residual, [T, HC_MULT, D])
    post_packed = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    post_packed = pl.reshape(post, [T, HC_MULT])
    comb_packed = pl.create_tensor([T, HC_MULT, HC_MULT], dtype=pl.FP32)
    comb_packed = pl.reshape(comb, [T, HC_MULT, HC_MULT])
    y_packed = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    y_packed = pl.reshape(y, [T, HC_MULT, D])
    y_packed = prefill_hc_post_packed(
        x_packed,
        residual_packed,
        post_packed,
        comb_packed,
        y_packed,
    )
    return pl.reshape(y_packed, [B, S, HC_MULT, D])


@pl.jit
def prefill_hc_post_test(
    x:               pl.Tensor[[B, S, D],             pl.BF16],
    residual:        pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
    post:            pl.Tensor[[B, S, HC_MULT],       pl.FP32],
    comb:            pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y:               pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
):
    y = prefill_hc_post(
        x,
        residual,
        post,
        comb,
        y,
    )
    return y


def golden_prefill_hc_post(tensors):
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float()

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    tensors["y"][:] = y_fp32.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.05
    def init_residual():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_post():
        return torch.rand(B, S, HC_MULT) + 0.1
    def init_comb():
        return torch.rand(B, S, HC_MULT, HC_MULT) * 0.25

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_residual),
        TensorSpec("post", [B, S, HC_MULT], torch.float32, init_value=init_post),
        TensorSpec("comb", [B, S, HC_MULT, HC_MULT], torch.float32, init_value=init_comb),
        TensorSpec("y", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_hc_post_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_hc_post,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
