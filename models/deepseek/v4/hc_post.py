# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix (dynamic shape): combines a sublayer
output with its hc-residual into the next hc-stack via post and comb weights.
Supports both decode and prefill batch/sequence sizes via dynamic-shape tensors."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim

# tiling
T_TILE = 16
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    for block in pl.spmd((t_dim // T_TILE) * HC_MULT, name_hint="hc_post"):
        token_block = block // HC_MULT
        out_h = block % HC_MULT
        t0 = token_block * T_TILE
        for t in pl.pipeline(t0, t0 + T_TILE, stage=4):
            post_w = pl.read(post, [t, out_h])
            x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
            y_row = pl.mul(x_row, post_w)
            for in_h in pl.range(HC_MULT):
                comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                res_d = in_h * D
                res_row = pl.cast(residual_flat[t : t + 1, res_d : res_d + D], target_type=pl.FP32)
                weighted = pl.mul(res_row, comb_w)
                y_row = pl.add(y_row, weighted)
            y_out = pl.cast(y_row, target_type=pl.BF16, mode="rint")
            y_flat[t : t + 1, out_h * D : out_h * D + D] = y_out
    return y


@pl.jit
def hc_post_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)

    hc_post(x, residual, post, comb, y)
    return y


def golden_hc_post(tensors):
    """Torch reference, direct port of model.py Block.hc_post 684-687."""
    import torch

    x        = tensors["x"].float()
    residual = tensors["residual"].float()
    post     = tensors["post"].float()
    comb     = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)  # [B, S, HC, HC]

    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    y = y_fp32.to(torch.bfloat16)

    tensors["y"][:] = y


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, D) * 0.05
    def init_residual():
        return torch.randn(T, HC_MULT, D) * 0.05
    def init_post():
        return 2.0 * torch.sigmoid(torch.randn(T, HC_MULT))
    def init_comb():
        c = torch.rand(B, S, HC_MULT, HC_MULT) + 0.1
        return (c / c.sum(dim=-1, keepdim=True)).reshape(T, HC_MULT * HC_MULT)
    return [
        TensorSpec("x",        [T, D],                    torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [T, HC_MULT, D],           torch.bfloat16, init_value=init_residual),
        TensorSpec("post",     [T, HC_MULT],              torch.float32,  init_value=init_post),
        TensorSpec("comb",     [T, HC_MULT * HC_MULT],    torch.float32,  init_value=init_comb),
        TensorSpec("y",        [T, HC_MULT, D],           torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_post {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=hc_post_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_post,
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
