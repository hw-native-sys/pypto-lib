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


B       = 16                 # demo 4
S       = 1
D       = 4096               # v4-pro 7168
HC_MULT = 4
T       = B * S
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK
HC_DIM  = HC_MULT * D


def build_deepseek_v4_decode_hc_post_program():
    @pl.program
    class DeepSeekV4DecodeHcPost:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_hc_post(
            self,
            x:        pl.Tensor[[B, S, D],                    pl.BF16],
            residual: pl.Tensor[[B, S, HC_MULT, D],           pl.BF16],
            post:     pl.Tensor[[B, S, HC_MULT],              pl.FP32],
            comb:     pl.Tensor[[B, S, HC_MULT, HC_MULT],     pl.FP32],
            y:        pl.Out[pl.Tensor[[B, S, HC_MULT, D],    pl.BF16]],
        ):
            x_flat = pl.reshape(x, [T, D])
            residual_flat = pl.reshape(residual, [T, HC_DIM])
            post_flat = pl.reshape(post, [T * HC_MULT])
            comb_flat = pl.reshape(comb, [T * HC_MULT * HC_MULT])
            y_flat = pl.reshape(y, [T, HC_DIM])

            for out_h in pl.parallel(HC_MULT):
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="hc_post"):
                    for t in pl.parallel(0, T, 1, chunk=16):
                        post_w = pl.read(post_flat, [t * HC_MULT + out_h])
                        for db in pl.range(D_BLOCKS):
                            d0 = db * D_CHUNK
                            x_row = pl.cast(
                                pl.slice(x_flat, [1, D_CHUNK], [t, d0]),
                                target_type=pl.FP32,
                            )
                            y_row = pl.mul(x_row, post_w)
                            for in_h in pl.range(HC_MULT):
                                comb_w = pl.read(
                                    comb_flat,
                                    [t * HC_MULT * HC_MULT + in_h * HC_MULT + out_h],
                                )
                                residual_row = pl.cast(
                                    pl.slice(residual_flat, [1, D_CHUNK], [t, in_h * D + d0]),
                                    target_type=pl.FP32,
                                )
                                y_row = pl.add(y_row, pl.mul(residual_row, comb_w))
                            y_flat = pl.assemble(
                                y_flat,
                                pl.cast(y_row, target_type=pl.BF16),
                                [t, out_h * D + d0],
                            )
            y = pl.reshape(y_flat, [B, S, HC_MULT, D])
            return y

    return DeepSeekV4DecodeHcPost


def golden_deepseek_v4_decode_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch

    x        = tensors["x"].float()           # [B, S, D]
    residual = tensors["residual"].float()    # [B, S, HC, D]
    post     = tensors["post"].float()        # [B, S, HC]
    comb     = tensors["comb"].float()        # [B, S, HC, HC]

    # post.unsqueeze(-1) * x.unsqueeze(-2): [B,S,HC,1] * [B,S,1,D] -> [B,S,HC,D]
    # comb.unsqueeze(-1) * residual.unsqueeze(-2): [B,S,HC,HC,1] * [B,S,1,HC,D]
    #   -> [B,S,HC,HC,D], sum over dim=2 -> [B,S,HC,D]
    term1 = post.unsqueeze(-1) * x.unsqueeze(-2)
    term2 = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
    y = (term1 + term2).to(torch.bfloat16)

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
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v4_decode_hc_post_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_hc_post,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
