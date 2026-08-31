# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections post-mix for dynamic decode and prefill shapes."""

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
DECODE_ROWS_MAX = DECODE_BATCH * DECODE_SEQ

# tiling
T_TILE = 4
INACTIVE_FILL_T_TILE = 16
INACTIVE_FILL_D_TILE = 256
D_TILE = D // 4
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])
    residual_rows = pl.reshape(residual, [t_dim * HC_MULT, D])
    y_rows = pl.reshape(y, [t_dim * HC_MULT, D])

    if t_dim <= DECODE_ROWS_MAX:
        for t in pl.spmd(t_dim, name_hint="hc_post"):
            h0 = t * HC_MULT
            for d0 in pl.pipeline(0, D, D_TILE, stage=2):
                x_chunk = pl.cast(x[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                res_tile = residual_rows[h0 : h0 + HC_MULT, d0 : d0 + D_TILE]
                for out_h in pl.unroll(HC_MULT):
                    post_w_chunk = pl.read(post, [t, out_h])
                    y_chunk = pl.mul(x_chunk, post_w_chunk)
                    for in_h in pl.unroll(HC_MULT):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_row = res_tile[in_h : in_h + 1, 0:D_TILE]
                        res_weighted_chunk = pl.mul(res_row, comb_w)
                        y_chunk = pl.add(y_chunk, res_weighted_chunk)
                    y_rows[h0 + out_h : h0 + out_h + 1, d0 : d0 + D_TILE] = y_chunk
    else:
        for block in pl.spmd((t_dim // T_TILE) * HC_MULT, name_hint="hc_post"):
            token_block = block // HC_MULT
            out_h = block % HC_MULT
            t0 = token_block * T_TILE
            for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
                post_w = pl.read(post, [t, out_h])
                x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                y_row = pl.mul(x_row, post_w)
                for in_h in pl.pipeline(HC_MULT, stage=4):
                    comb_wide = pl.read(comb, [t, in_h * HC_MULT + out_h])
                    res_d = in_h * D
                    res_wide = residual_flat[t : t + 1, res_d : res_d + D]
                    weighted = pl.mul(res_wide, comb_wide)
                    y_row = pl.add(y_row, weighted)
                y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row
    return y


@pl.jit.inline
def hc_post_prefill(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compute prefill active rows and deterministically pad the static tail."""
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    t_dim = pl.tensor.dim(x, 0)
    if active_tokens > t_dim:
        active_tokens = t_dim

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    if active_tokens > 0:
        for block in pl.spmd(((active_tokens + T_TILE - 1) // T_TILE) * HC_MULT, name_hint="hc_post_prefill"):
            token_block = block // HC_MULT
            out_h = block % HC_MULT
            t0 = token_block * T_TILE
            for t in pl.pipeline(t0, t0 + T_TILE, stage=2):
                if t < active_tokens:
                    post_w = pl.read(post, [t, out_h])
                    x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.pipeline(HC_MULT, stage=4):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D
                        res_row = residual_flat[t : t + 1, res_d : res_d + D]
                        weighted = pl.mul(res_row, comb_w)
                        y_row = pl.add(y_row, weighted)
                    y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row

    inactive_tokens = t_dim - active_tokens
    if inactive_tokens > 0:
        for fill_block in pl.spmd(
            ((inactive_tokens + INACTIVE_FILL_T_TILE - 1) // INACTIVE_FILL_T_TILE) * HC_MULT,
            name_hint="hc_post_inactive_pad",
        ):
            fill_tile = fill_block // HC_MULT
            out_h = fill_block % HC_MULT
            fill_t0 = active_tokens + fill_tile * INACTIVE_FILL_T_TILE
            zero = pl.full([1, INACTIVE_FILL_D_TILE], dtype=pl.FP32, value=0.0)
            for fill_dt in pl.range(INACTIVE_FILL_T_TILE):
                fill_t = fill_t0 + fill_dt
                if fill_t < t_dim:
                    for fill_d0 in pl.range(0, D, INACTIVE_FILL_D_TILE):
                        out_d0 = out_h * D + fill_d0
                        y_flat[fill_t : fill_t + 1, out_d0 : out_d0 + INACTIVE_FILL_D_TILE] = zero
    return y


@pl.jit
def hc_post_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)

    hc_post(x, residual, post, comb, y)
    return y


def golden_hc_post(tensors):
    """Compute the Hyper-Connections post-mix reference."""
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float().reshape(-1, HC_MULT, HC_MULT)

    T = x.shape[0]
    y_fp32 = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, in_h, :] * comb[:, in_h, out_h:out_h + 1]
        y_fp32[:, out_h, :] = y_row
    tensors["y"][:] = y_fp32


def golden_hc_post_prefill(tensors):
    """Prefill reference: compute active rows and zero the static tail."""
    golden_hc_post(tensors)
    num_tokens = int(tensors["num_tokens"])
    tensors["y"][num_tokens:] = 0


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
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [T, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post", [T, HC_MULT], torch.float32, init_value=init_post),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32, init_value=init_comb),
        TensorSpec("y", [T, HC_MULT, D], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    MODES = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    mode_help = "Use decode or prefill batch sizes, or 'all' to test both."
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="decode", help=mode_help)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_post {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=hc_post_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_post,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
