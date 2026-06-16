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

# Tiling config
D_CHUNK = 256
D_STEPS = D // D_CHUNK          # number of D-chunks per hidden dim
assert D % D_CHUNK == 0, f"D ({D}) must be divisible by D_CHUNK ({D_CHUNK})"

# Keep T dynamic while packing 16 tokens per task; current decode/prefill T values are 16-aligned.
HC_POST_TOKEN_TILE = 16
HC_POST_LOCAL_BLOCKS = D_STEPS
assert (DECODE_BATCH * DECODE_SEQ) % HC_POST_TOKEN_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % HC_POST_TOKEN_TILE == 0
# The per-token post/comb mix is hand-unrolled for HC_MULT == 4: the four out_h
# blocks, res0..res3, the four post_cols writes, the comb_t column strides
# (in_h * HC_MULT -> 0/4/8/12) and the 0..15 columns, and the in_h * D residual
# offsets all assume it. Fail loudly if the model config changes hc_mult instead
# of silently mixing the wrong comb columns / leaving out_h rows unwritten.
assert HC_MULT == 4, (
    f"hc_post is hand-specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the out_h/in_h unrolling for the new hc_mult before using it."
)


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

    token_blocks = t_dim // HC_POST_TOKEN_TILE

    for block in pl.spmd(token_blocks * HC_POST_LOCAL_BLOCKS, name_hint="hc_post"):
        token_block = block // HC_POST_LOCAL_BLOCKS
        d0 = (block % HC_POST_LOCAL_BLOCKS) * D_CHUNK
        t0 = token_block * HC_POST_TOKEN_TILE
        # Process the 16-token tile as 2-D blocks: per-token post/comb weights
        # are [TILE, 1] columns applied via row_expand_mul instead of 16
        # separate [1, D_CHUNK] row operations. Load full contiguous weight
        # rows once (a [TILE, 1] strided GM load is not a legal TLOAD layout
        # on a2a3); out_h/in_h are hand-unrolled because tile subscript lower
        # bounds must be compile-time constants.
        post_cols = pl.create_tensor([HC_MULT, HC_POST_TOKEN_TILE], dtype=pl.FP32)
        for dt in pl.range(HC_POST_TOKEN_TILE):
            pl.write(post_cols, [0, dt], pl.read(post, [t0 + dt, 0]))
            pl.write(post_cols, [1, dt], pl.read(post, [t0 + dt, 1]))
            pl.write(post_cols, [2, dt], pl.read(post, [t0 + dt, 2]))
            pl.write(post_cols, [3, dt], pl.read(post, [t0 + dt, 3]))
        comb_t = pl.transpose(comb[t0 : t0 + HC_POST_TOKEN_TILE, 0 : HC_MULT * HC_MULT], axis1=0, axis2=1)
        x_tile = pl.cast(x[t0 : t0 + HC_POST_TOKEN_TILE, d0 : d0 + D_CHUNK], target_type=pl.FP32)
        res0 = pl.cast(residual_flat[t0 : t0 + HC_POST_TOKEN_TILE, 0 * D + d0 : 0 * D + d0 + D_CHUNK], target_type=pl.FP32)
        res1 = pl.cast(residual_flat[t0 : t0 + HC_POST_TOKEN_TILE, 1 * D + d0 : 1 * D + d0 + D_CHUNK], target_type=pl.FP32)
        res2 = pl.cast(residual_flat[t0 : t0 + HC_POST_TOKEN_TILE, 2 * D + d0 : 2 * D + d0 + D_CHUNK], target_type=pl.FP32)
        res3 = pl.cast(residual_flat[t0 : t0 + HC_POST_TOKEN_TILE, 3 * D + d0 : 3 * D + d0 + D_CHUNK], target_type=pl.FP32)

        # out_h = 0: comb columns 0, 4, 8, 12
        y_tile = pl.row_expand_mul(x_tile, pl.reshape(post_cols[0 : 1, :], [HC_POST_TOKEN_TILE, 1]))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res0, pl.reshape(comb_t[0:1, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res1, pl.reshape(comb_t[4:5, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res2, pl.reshape(comb_t[8:9, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res3, pl.reshape(comb_t[12:13, :], [HC_POST_TOKEN_TILE, 1])))
        y_flat[t0 : t0 + HC_POST_TOKEN_TILE, 0 * D + d0 : 0 * D + d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")

        # out_h = 1: comb columns 1, 5, 9, 13
        y_tile = pl.row_expand_mul(x_tile, pl.reshape(post_cols[1 : 2, :], [HC_POST_TOKEN_TILE, 1]))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res0, pl.reshape(comb_t[1:2, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res1, pl.reshape(comb_t[5:6, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res2, pl.reshape(comb_t[9:10, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res3, pl.reshape(comb_t[13:14, :], [HC_POST_TOKEN_TILE, 1])))
        y_flat[t0 : t0 + HC_POST_TOKEN_TILE, 1 * D + d0 : 1 * D + d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")

        # out_h = 2: comb columns 2, 6, 10, 14
        y_tile = pl.row_expand_mul(x_tile, pl.reshape(post_cols[2 : 3, :], [HC_POST_TOKEN_TILE, 1]))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res0, pl.reshape(comb_t[2:3, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res1, pl.reshape(comb_t[6:7, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res2, pl.reshape(comb_t[10:11, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res3, pl.reshape(comb_t[14:15, :], [HC_POST_TOKEN_TILE, 1])))
        y_flat[t0 : t0 + HC_POST_TOKEN_TILE, 2 * D + d0 : 2 * D + d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")

        # out_h = 3: comb columns 3, 7, 11, 15
        y_tile = pl.row_expand_mul(x_tile, pl.reshape(post_cols[3 : 4, :], [HC_POST_TOKEN_TILE, 1]))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res0, pl.reshape(comb_t[3:4, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res1, pl.reshape(comb_t[7:8, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res2, pl.reshape(comb_t[11:12, :], [HC_POST_TOKEN_TILE, 1])))
        y_tile = pl.add(y_tile, pl.row_expand_mul(res3, pl.reshape(comb_t[15:16, :], [HC_POST_TOKEN_TILE, 1])))
        y_flat[t0 : t0 + HC_POST_TOKEN_TILE, 3 * D + d0 : 3 * D + d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
    # Data is already written through the y_flat view; skip the reshape-back
    # to avoid a static-vs-dynamic type mismatch when inlined into callers
    # with concrete shapes (e.g. moe_ep.py).
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

    y = hc_post(x, residual, post, comb, y)
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
        return torch.randn(T, D) * 0.1
    def init_residual():
        return torch.randn(T, HC_MULT, D) * 0.1
    def init_post():
        p = torch.rand(B, S, HC_MULT) + 0.1
        return (p / p.sum(dim=-1, keepdim=True)).reshape(T, HC_MULT)
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
