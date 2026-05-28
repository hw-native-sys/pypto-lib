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

# kernel-local
HC_PAD = 8

# tiling
D_CHUNK = 512
D_BLOCKS = D // D_CHUNK


@pl.jit.inline
def prefill_hc_post_from_padded(
    x:               pl.Tensor[[B, S, D],             pl.BF16],
    residual:        pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
    post_pad:        pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb0_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb1_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb2_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb3_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    y:               pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    y_flat = pl.reshape(y, [T, HC_DIM])

    # Same math as prefill_hc_post, but consume the padded SSA tensors emitted
    # by prefill_hc_pre.  This gives the scheduler explicit producer->consumer
    # dependencies and avoids using side-effect-only post/comb scratch writes.
    for out_h in pl.parallel(HC_MULT):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer,
                   name_hint="prefill_hc_post_from_padded"):
            for t in pl.parallel(0, T, 1, chunk=16):
                post_w = pl.read(post_pad, [t, out_h])
                comb0_w = pl.read(comb0_pad, [t, out_h])
                comb1_w = pl.read(comb1_pad, [t, out_h])
                comb2_w = pl.read(comb2_pad, [t, out_h])
                comb3_w = pl.read(comb3_pad, [t, out_h])
                for db in pl.range(D_BLOCKS):
                    d0 = db * D_CHUNK
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
                    y_flat = pl.assemble(
                        y_flat,
                        pl.cast(y_row, target_type=pl.BF16, mode="rint"),
                        [t, out_h * D + d0],
                    )
    y = pl.reshape(y_flat, [B, S, HC_MULT, D])
    return y


@pl.jit
def prefill_hc_post_test(
    x:               pl.Tensor[[B, S, D],             pl.BF16],
    residual:        pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
    post_pad:        pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb0_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb1_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb2_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb3_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    y:               pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
):
    y = prefill_hc_post_from_padded(
        x,
        residual,
        post_pad,
        comb0_pad,
        comb1_pad,
        comb2_pad,
        comb3_pad,
        y,
    )
    return y


def golden_prefill_hc_post(tensors):
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post_pad = tensors["post_pad"].float().view(B, S, HC_PAD)
    comb_pads = [
        tensors["comb0_pad"].float().view(B, S, HC_PAD),
        tensors["comb1_pad"].float().view(B, S, HC_PAD),
        tensors["comb2_pad"].float().view(B, S, HC_PAD),
        tensors["comb3_pad"].float().view(B, S, HC_PAD),
    ]

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post_pad[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb_pads[in_h][:, :, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    tensors["y"][:] = y_fp32.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.05
    def init_residual():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_post_pad():
        pad = torch.zeros(T, HC_PAD)
        pad[:, :HC_MULT] = torch.rand(T, HC_MULT) + 0.1
        return pad
    def init_comb_pad():
        pad = torch.zeros(T, HC_PAD)
        pad[:, :HC_MULT] = torch.rand(T, HC_MULT) * 0.25
        return pad

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_residual),
        TensorSpec("post_pad", [T, HC_PAD], torch.float32, init_value=init_post_pad),
        TensorSpec("comb0_pad", [T, HC_PAD], torch.float32, init_value=init_comb_pad),
        TensorSpec("comb1_pad", [T, HC_PAD], torch.float32, init_value=init_comb_pad),
        TensorSpec("comb2_pad", [T, HC_PAD], torch.float32, init_value=init_comb_pad),
        TensorSpec("comb3_pad", [T, HC_PAD], torch.float32, init_value=init_comb_pad),
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
