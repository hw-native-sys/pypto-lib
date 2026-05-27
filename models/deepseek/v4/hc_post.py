# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix (decode): combines a sublayer output with its hc-residual
into the next hc-stack via post and comb weights."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


@pl.jit.inline
def hc_post(
    x: pl.Tensor[[B, S, D], pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[B, S, HC_MULT], pl.FP32],
    comb: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    post_t = pl.reshape(post, [T, HC_MULT])
    comb_t = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    y_flat = pl.reshape(y, [T, HC_DIM])

    for tb in pl.spmd(T // 2, name_hint="hc_post"):
        for tt in pl.range(2):
            t = tb * 2 + tt
            x_row = pl.cast(x_flat[t : t + 1, 0:D], target_type=pl.FP32)
            for out_h in pl.range(HC_MULT):
                post_w = pl.read(post_t, [t, out_h])
                y_row = pl.mul(x_row, post_w)
                for in_h in pl.range(HC_MULT):
                    comb_w = pl.read(comb_t, [t, in_h * HC_MULT + out_h])
                    res_d = in_h * D
                    residual_row = pl.cast(residual_flat[t : t + 1, res_d : res_d + D], target_type=pl.FP32)
                    y_row = pl.add(y_row, pl.mul(residual_row, comb_w))
                y_d = out_h * D
                y_bf16 = pl.cast(y_row, target_type=pl.BF16, mode="rint")
                y_flat[t : t + 1, y_d : y_d + D] = y_bf16
    y = pl.reshape(y_flat, [B, S, HC_MULT, D])
    return y


@pl.jit
def hc_post_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    residual: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[B, S, HC_MULT], pl.FP32],
    comb: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
):
    y = hc_post(x, residual, post, comb, y)
    return y


def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch

    x        = tensors["x"].float()           # [B, S, D]
    residual = tensors["residual"].float()    # [B, S, HC, D]
    post     = tensors["post"].float()        # [B, S, HC]
    comb     = tensors["comb"].float()        # [B, S, HC, HC]

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    y = y_fp32.to(torch.bfloat16)

    tensors["y"][:] = y


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_residual():
        return torch.randn(B, S, HC_MULT, D) * 0.1
    def init_post():
        p = torch.rand(B, S, HC_MULT) + 0.1
        return p / p.sum(dim=-1, keepdim=True)
    def init_comb():
        c = torch.rand(B, S, HC_MULT, HC_MULT) + 0.1
        return c / c.sum(dim=-1, keepdim=True)

    return [
        TensorSpec("x",        [B, S, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [B, S, HC_MULT, D],        torch.bfloat16, init_value=init_residual),
        TensorSpec("post",     [B, S, HC_MULT],           torch.float32,  init_value=init_post),
        TensorSpec("comb",     [B, S, HC_MULT, HC_MULT],  torch.float32,  init_value=init_comb),
        TensorSpec("y",        [B, S, HC_MULT, D],        torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=hc_post_test,
        specs=build_tensor_specs(),
        golden_fn=golden_hc_post,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
