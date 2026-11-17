# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-pass hyper-connections: coefficient projection, Sinkhorn mixing, collapse, and expansion.

``hc_pre`` projects the concatenated HC streams into 24 coefficients per
token, splits them into pre/post/residual groups, runs the doubly-stochastic
residual Sinkhorn normalization, and collapses the incoming streams.
``hc_post`` expands a sublayer result back into four residual streams.
``hc_head`` performs the final collapse.
"""

import math

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M, T_DYN


# Dynamic shape variable.

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
NORM_EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
SINKHORN_ITER = M.hc_sinkhorn_iters

# tiling
M_TILE = 16
T_TILE = 8
K_TILE = 512
MIX_PAD = 32
HC_PAD = 8
HC_MULT_F = float(HC_MULT)
HC_PAD_F = float(HC_PAD)
HC_PAD_INV = 1.0 / HC_PAD
D_TILE = 512


@pl.jit.inline(auto_scope=False)
def hc_mixes(
    x_hc: pl.Tensor,
    function: pl.Tensor,
    mixes: pl.Tensor,
):
    """Project the flattened HC streams and scale by each token's inverse RMS."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_inv_rms"):
        t0 = block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HC_DIM, K_TILE, stage=2):
            source = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(source, source)), [1, T_TILE]))
        inverse = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / HC_DIM), NORM_EPS), high_precision=True)
        inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(inverse, [T_TILE, 1])

    for block in pl.spmd((t_dim + M_TILE - 1) // M_TILE, name_hint="hc_project"):
        t0 = block * M_TILE
        rows = pl.min(M_TILE, t_dim - t0)
        acc = pl.create_tensor([M_TILE, MIX_PAD], dtype=pl.FP32)
        for k0 in pl.pipeline(0, HC_DIM, K_TILE, stage=2):
            linear_source = pl.slice(x_flat, [M_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            linear_source = pl.set_validshape(pl.fillpad(linear_source, pad_value=pl.PadValue.zero), M_TILE, K_TILE)
            weight = function[0:MIX_PAD, k0 : k0 + K_TILE]
            acc = pl.matmul_acc(acc, linear_source, weight, b_trans=True, init_cond=(k0 == 0))
        row_inverse = pl.reshape(inv_rms[t0 : t0 + M_TILE, 0:1], [M_TILE, 1])
        scaled = pl.row_expand_mul(acc, row_inverse)
        mixes[t0 : t0 + M_TILE, 0:MIX_PAD] = pl.set_validshape(scaled, rows, MIX_PAD)
    return mixes


@pl.jit.inline(auto_scope=False)
def hc_coefficients(
    mixes: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Split coefficients and normalize each residual matrix with Sinkhorn iterations."""
    tokens = pl.tensor.dim(mixes, 0)
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    for token in pl.spmd(tokens, name_hint="hc_coefficients"):
        lane_f = pl.cast(pl.tile.arange(0, [1, HC_PAD * HC_PAD], dtype=pl.INT32), target_type=pl.FP32)
        row_index = pl.cast(pl.mul(lane_f, HC_PAD_INV), target_type=pl.INT32, mode="trunc")
        row_f = pl.cast(row_index, target_type=pl.FP32)
        col_f = pl.sub(lane_f, pl.mul(row_f, HC_PAD_F))
        row_lane = pl.reshape(row_f, [HC_PAD, HC_PAD])
        col_lane = pl.reshape(col_f, [HC_PAD, HC_PAD])
        ones = pl.tile.full([HC_PAD, HC_PAD], dtype=pl.FP32, value=1.0)
        zeros = pl.tile.full([HC_PAD, HC_PAD], dtype=pl.FP32, value=0.0)
        select_tmp = pl.tile.create([HC_PAD, HC_PAD], dtype=pl.FP32)
        mask = pl.tile.sel(pl.tile.cmps(row_lane, HC_MULT_F, cmp_type=2), ones, zeros, select_tmp)
        mask = pl.tile.sel(pl.tile.cmps(col_lane, HC_MULT_F, cmp_type=2), mask, zeros, select_tmp)
        row = pl.tile.load(mixes, [token, 0], [1, MIX_PAD])
        # Copy the pre/post coefficient ranges into fresh tiles. A
        # `tile.slice` view at column HC_MULT (=4 FP32 = 16 B) is not
        # 32-byte aligned, and a VEC instruction on that view faults on
        # device ("UB address accessed by the VEC instruction is not aligned").
        pre_raw = pl.tile.create([1, HC_PAD], dtype=pl.FP32)
        post_raw = pl.tile.create([1, HC_PAD], dtype=pl.FP32)
        for lane in pl.unroll(HC_PAD):
            pl.tile.write(pre_raw, [0, lane], pl.tile.read(row, [0, lane]))
            pl.tile.write(post_raw, [0, lane], pl.tile.read(row, [0, HC_MULT + lane]))
        pre_raw = pl.set_validshape(pre_raw, 1, HC_PAD)
        post_raw = pl.set_validshape(post_raw, 1, HC_PAD)
        pre_bias = pl.tile.full([1, HC_PAD], dtype=pl.FP32, value=0.0)
        for lane in pl.unroll(HC_MULT):
            pl.tile.write(pre_bias, [0, lane], pl.read(base, [lane]))
        pre_logits = pl.col_expand_add(pl.mul(pre_raw, pl.read(scale, [0])), pre_bias)
        pre = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre = pl.add(pre, HC_EPS)
        pl.store(pl.set_validshape(pre, 1, HC_MULT), [token, 0], pre_mix)
        post_bias = pl.tile.full([1, HC_PAD], dtype=pl.FP32, value=0.0)
        for lane in pl.unroll(HC_MULT):
            pl.tile.write(post_bias, [0, lane], pl.read(base, [HC_MULT + lane]))
        post_logits = pl.col_expand_add(pl.mul(post_raw, pl.read(scale, [1])), post_bias)
        post = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
        pl.store(pl.set_validshape(post, 1, HC_MULT), [token, 0], post_mix)
        logits = pl.tile.full([HC_PAD, HC_PAD], dtype=pl.FP32, value=0.0)
        for out_h in pl.unroll(HC_MULT):
            for in_h in pl.unroll(HC_MULT):
                raw = pl.tile.read(row, [0, HC_MULT * 2 + in_h * HC_MULT + out_h])
                bias = pl.read(base, [HC_MULT * 2 + in_h * HC_MULT + out_h])
                pl.tile.write(logits, [in_h, out_h], raw * pl.read(scale, [2]) + bias)
        reduce_tmp = pl.tile.create([HC_PAD, HC_PAD], dtype=pl.FP32)
        highest = pl.tile.row_max(logits, tmp_tile=reduce_tmp)
        exponent = pl.mul(pl.exp(pl.row_expand_sub(logits, highest)), mask)
        sum_tmp = pl.tile.create([HC_PAD, HC_PAD], dtype=pl.FP32)
        probability = pl.row_expand_div(exponent, pl.add(pl.row_sum(exponent, tmp_tile=sum_tmp), HC_EPS))
        probability = pl.add(probability, HC_EPS)
        probability = pl.mul(probability, mask)
        probability = pl.col_expand_div(
            probability, pl.add(pl.col_sum(probability, tmp_tile=sum_tmp), HC_EPS)
        )
        for _ in pl.unroll(SINKHORN_ITER - 1):
            probability = pl.row_expand_div(
                probability, pl.add(pl.row_sum(probability, tmp_tile=sum_tmp), HC_EPS)
            )
            probability = pl.col_expand_div(
                probability, pl.add(pl.col_sum(probability, tmp_tile=sum_tmp), HC_EPS)
            )
        packed = pl.tile.full([1, HC_MULT * HC_MULT], dtype=pl.FP32, value=0.0)
        for out_h in pl.unroll(HC_MULT):
            for in_h in pl.unroll(HC_MULT):
                pl.tile.write(packed, [0, in_h * HC_MULT + out_h], pl.tile.read(probability, [in_h, out_h]))
        pl.store(packed, [token, 0], residual_flat)
    return pre_mix, post_mix, residual_mix


@pl.jit.inline(auto_scope=False)
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Collapse the four HC streams with the incoming pre-mix."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for block in pl.spmd(t_dim, name_hint="hc_head"):
        token = pl.tile.get_block_idx()
        pre0 = pl.read(pre_mix, [token, 0])
        pre1 = pl.read(pre_mix, [token, 1])
        pre2 = pl.read(pre_mix, [token, 2])
        pre3 = pl.read(pre_mix, [token, 3])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x0 = pl.mul(pl.tile.load(x_flat, [token, d0], [1, D_TILE]), pre0)
            x1 = pl.mul(pl.tile.load(x_flat, [token, D + d0], [1, D_TILE]), pre1)
            x2 = pl.mul(pl.tile.load(x_flat, [token, 2 * D + d0], [1, D_TILE]), pre2)
            x3 = pl.mul(pl.tile.load(x_flat, [token, 3 * D + d0], [1, D_TILE]), pre3)
            collapsed = pl.add(pl.add(x0, x1), pl.add(x2, x3))
            pl.store(pl.cast(collapsed, target_type=pl.BF16, mode="rint"), [token, d0], output)
    return output


@pl.jit.inline(auto_scope=False)
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Mix residual streams, round to BF16 and store the rounded values in FP32."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    for block in pl.spmd(t_dim * HC_MULT, name_hint="hc_post"):
        token = block // HC_MULT
        out_h = block % HC_MULT
        post = pl.read(post_mix, [token, out_h])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            value = pl.mul(pl.cast(pl.tile.load(sublayer, [token, d0], [1, D_TILE]), target_type=pl.FP32), post)
            for in_h in pl.unroll(HC_MULT):
                coefficient = pl.read(mix_flat, [token, in_h * HC_MULT + out_h])
                lane = pl.tile.load(residual_flat, [token, in_h * D + d0], [1, D_TILE])
                value = pl.add(value, pl.mul(lane, coefficient))
            rounded_bf16 = pl.cast(value, target_type=pl.BF16, mode="rint")
            rounded_fp32 = pl.cast(rounded_bf16, target_type=pl.FP32)
            pl.store(rounded_fp32, [token, out_h * D + d0], output_flat)
    return output


@pl.jit.inline(auto_scope=False)
def hc_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    function: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    output: pl.Tensor,
    next_pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Generate delayed coefficients and collapse the incoming HC streams."""
    t_dim = pl.tensor.dim(x_hc, 0)
    mixes = pl.create_tensor([t_dim, MIX_PAD], dtype=pl.FP32)
    mixes = hc_mixes(x_hc, function, mixes)
    pre_result, post_result, residual_result = hc_coefficients(
        mixes, scale, base, next_pre_mix, post_mix, residual_mix
    )
    output_result = hc_head(x_hc, incoming_pre_mix, output)
    return output


@pl.jit
def hc_pre_test(
    x_hc: pl.Tensor[[T_DYN, 4, 5120], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, 4], pl.FP32],
    function: pl.Tensor[[32, 20480], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[24], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 5120], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, 4], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, 4], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, 4, 4], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    result = hc_pre(
        x_hc, incoming_pre_mix, function, scale, base, output, pre_mix, post_mix, residual_mix
    )
    return output, pre_mix, post_mix, residual_mix


@pl.jit
def hc_post_test(
    sublayer: pl.Tensor[[T_DYN, 5120], pl.BF16],
    residual: pl.Tensor[[T_DYN, 4, 5120], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, 4], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, 4, 4], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 4, 5120], pl.FP32]],
):
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    result = hc_post(sublayer, residual, post_mix, residual_mix, output)
    return output


@pl.jit
def hc_head_test(
    x_hc: pl.Tensor[[T_DYN, 4, 5120], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, 4], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 5120], pl.BF16]],
):
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    result = hc_head(x_hc, pre_mix, output)
    return output


def golden_mhc_mixes(x_hc, function, scale, base, iterations=SINKHORN_ITER):
    """Reference HC coefficient split with Sinkhorn-normalized residual mixing."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(flat, function.float())[:, :MIX_HC] * inverse_rms
    pre_raw, post_raw, residual_raw = mixes.split((HC_MULT, HC_MULT, HC_MULT * HC_MULT), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[HC_MULT : 2 * HC_MULT])
    residual_base = base[2 * HC_MULT :].unflatten(-1, (HC_MULT, HC_MULT))
    residual = (residual_raw.unflatten(-1, (HC_MULT, HC_MULT)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + HC_EPS
    residual = residual / (residual.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + HC_EPS)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + HC_EPS)
    return pre, post, residual


def golden_hc_collapse(x, pre_mix):
    products = x.float() * pre_mix.float().unsqueeze(-1)
    return ((products[..., 0, :] + products[..., 1, :]) + (products[..., 2, :] + products[..., 3, :])).to(torch.bfloat16)


def golden_hc_pre(tensors):
    pre, post, residual = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"]
    )
    tensors["pre_mix"][:] = pre
    tensors["post_mix"][:] = post
    tensors["residual_mix"][:] = residual
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])


def golden_hc_post(tensors):
    update = tensors["post_mix"].float().unsqueeze(-1) * tensors["sublayer"].float().unsqueeze(-2)
    skip = (tensors["residual_mix"].float().unsqueeze(-1) * tensors["residual"].float().unsqueeze(-2)).sum(dim=-3)
    rounded = (update + skip).to(torch.bfloat16)
    tensors["output"][:] = rounded.float()


def golden_hc_head(tensors):
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"])


def _hc_mix_inputs(tokens, seed=0):
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    return {
        "x_hc": x_hc,
        "incoming_pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + HC_EPS,
        "function": torch.cat(
            (
                torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
                torch.zeros(MIX_PAD - MIX_HC, HC_DIM),
            ),
            dim=0,
        ),
        "scale": torch.randn(3, generator=gen),
        "base": torch.randn(MIX_HC, generator=gen),
    }


def build_pre_specs(tokens: int = 16):
    from golden import TensorSpec

    values = _hc_mix_inputs(tokens)
    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=values["x_hc"]),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=values["incoming_pre_mix"]),
        TensorSpec("function", [MIX_PAD, HC_DIM], torch.float32, init_value=values["function"]),
        TensorSpec("scale", [3], torch.float32, init_value=values["scale"]),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=values["base"]),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


def build_post_specs(tokens: int = 16, seed: int = 7):
    from golden import TensorSpec

    gen = torch.Generator().manual_seed(seed)
    mix = torch.rand(tokens, HC_MULT, HC_MULT, generator=gen) + HC_EPS
    for _ in range(SINKHORN_ITER):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=lambda: torch.randn(tokens, D).bfloat16()),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=lambda: torch.randn(tokens, HC_MULT, D)),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=lambda: 2 * torch.sigmoid(torch.randn(tokens, HC_MULT))),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


def build_head_specs(tokens: int = 16, seed: int = 3):
    from golden import TensorSpec

    gen = torch.Generator().manual_seed(seed)
    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=lambda: torch.randn(tokens, HC_MULT, D)),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=lambda: torch.rand(tokens, HC_MULT)),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    failures = []
    cases = [
        (
            "pre",
            hc_pre_test,
            build_pre_specs(args.tokens),
            golden_hc_pre,
            {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=5e-3),
                "output": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=5e-3),
            },
        ),
        (
            "post",
            hc_post_test,
            build_post_specs(args.tokens),
            golden_hc_post,
            {"output": ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=5e-3)},
        ),
        (
            "head",
            hc_head_test,
            build_head_specs(args.tokens),
            golden_hc_head,
            {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=5e-3)},
        ),
    ]
    for name, fn, specs, golden_fn, compare in cases:
        print(f"[HC] {name}", flush=True)
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
