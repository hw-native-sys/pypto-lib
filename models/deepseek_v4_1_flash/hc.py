# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import math

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M, SP_T_DYN, T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 1
RMS_K_TILE = 1024
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
D_TILE = 2560
OUT_TILE = 2
D_TILES = D // D_TILE
POST_SPMD_BLOCKS = 4096


@pl.jit.inline
def hc_mixes(
    x_hc: pl.Tensor,
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="mhc_rms"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            rms_x_tile = pl.slice(
                x_flat,
                [T_TILE, RMS_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, RMS_K_TILE],
            )
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
        rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
            pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
        )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="mhc_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        split = task % LINEAR_OK
        k_base = split * LINEAR_K_PER_SPLIT
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            linear_x_tile = pl.slice(
                x_flat,
                [LINEAR_T_TILE, LINEAR_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, LINEAR_K_TILE],
            )
            w_tile = pl.slice(
                function,
                [MIX_PAD, LINEAR_K_TILE],
                [0, k0],
                valid_shape=[MIX_HC, LINEAR_K_TILE],
            )
            acc = pl.matmul_acc(acc, linear_x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
        partial = split * t_linear + t0
        mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc

    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    with pl.spmd((t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="mhc_split_sinkhorn") as mixes_ready:
        block = pl.tile.get_block_idx()
        t0 = block * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_scalar = pl.read(inv_rms, [t0, 0])
        total = pl.load(
            mixes_partials,
            [t0, 0],
            [COMB_T_TILE, MIX_PAD],
            valid_shape=[valid_rows, MIX_HC],
            target_memory=pl.MemorySpace.Vec,
        )
        partial1 = pl.load(
            mixes_partials,
            [t_linear + t0, 0],
            [COMB_T_TILE, MIX_PAD],
            valid_shape=[valid_rows, MIX_HC],
            target_memory=pl.MemorySpace.Vec,
        )
        partial2 = pl.load(
            mixes_partials,
            [2 * t_linear + t0, 0],
            [COMB_T_TILE, MIX_PAD],
            valid_shape=[valid_rows, MIX_HC],
            target_memory=pl.MemorySpace.Vec,
        )
        partial3 = pl.load(
            mixes_partials,
            [3 * t_linear + t0, 0],
            [COMB_T_TILE, MIX_PAD],
            valid_shape=[valid_rows, MIX_HC],
            target_memory=pl.MemorySpace.Vec,
        )
        total = pl.add(pl.add(total, partial1), pl.add(partial2, partial3))
        pre_logits = pl.tile.full([1, HC_PAD], dtype=pl.FP32, value=0.0)
        post_logits = pl.tile.full([1, HC_PAD], dtype=pl.FP32, value=0.0)
        for j in pl.unroll(HC_PAD):
            pre_value_j = pl.tile.read(total, [0, j]) * inv_scalar * scale0 + pl.read(base, [j])
            post_value_j = (
                pl.tile.read(total, [0, HC_MULT + j % HC_MULT]) * inv_scalar * scale1
                + pl.read(base, [HC_MULT + j % HC_MULT])
            )
            pl.tile.write(pre_logits, [0, j], pre_value_j)
            pl.tile.write(post_logits, [0, j], post_value_j)
        pre_value = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
        post_value = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
        pl.store(pl.set_validshape(pre_value, valid_rows, HC_MULT), [t0, 0], pre_mix)
        pl.store(pl.set_validshape(post_value, valid_rows, HC_MULT), [t0, 0], post_mix)
        comb_offset = HC_MULT * 2
        logits = pl.tile.full([HC_PAD, HC_PAD], dtype=pl.FP32, value=0.0)
        for i in pl.unroll(HC_PAD):
            for j in pl.unroll(HC_PAD):
                lane = comb_offset + (i % HC_MULT) * HC_MULT + j % HC_MULT
                value = pl.tile.read(total, [0, lane]) * inv_scalar * scale2 + pl.read(base, [lane])
                pl.tile.write(logits, [i, j], value)
        row_tmp = pl.create_tile([HC_PAD, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        rows = pl.exp(pl.row_expand_sub(logits, pl.row_max(logits, row_tmp)))
        rows = pl.add(pl.row_expand_div(rows, pl.mul(pl.row_sum(rows, row_tmp), 0.5)), HC_EPS)
        rows = pl.col_expand_div(rows, pl.add(pl.mul(pl.col_sum(rows), 0.5), HC_EPS))
        for _ in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            rows = pl.row_expand_div(rows, pl.add(pl.mul(pl.row_sum(rows, row_tmp), 0.5), HC_EPS))
            rows = pl.col_expand_div(rows, pl.add(pl.mul(pl.col_sum(rows), 0.5), HC_EPS))
        for row in pl.unroll(HC_MULT):
            logical_row = pl.slice(
                rows,
                [1, HC_PAD],
                [row, 0],
                valid_shape=[1, HC_MULT],
            )
            pl.store(logical_row, [t0, row * HC_MULT], residual_mix_flat)
    return mixes_ready


@pl.jit.inline
def hc_pre(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="hc_pre", allow_early_resolve=True):
        pre0 = pl.read(pre_mix, [t, 0])
        pre1 = pl.read(pre_mix, [t, 1])
        pre2 = pl.read(pre_mix, [t, 2])
        pre3 = pl.read(pre_mix, [t, 3])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x0 = pl.load(x_flat, [t, d0], [1, D_TILE])
            x1 = pl.load(x_flat, [t, D + d0], [1, D_TILE])
            x2 = pl.load(x_flat, [t, 2 * D + d0], [1, D_TILE])
            x3 = pl.load(x_flat, [t, 3 * D + d0], [1, D_TILE])
            y0 = pl.mul(x0, pre0)
            y1 = pl.mul(x1, pre1)
            y2 = pl.mul(x2, pre2)
            y3 = pl.mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        for row in pl.spmd(tokens, name_hint="decode_sp_zero_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.inline
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    total_work = t_dim * (HC_MULT // OUT_TILE) * D_TILES
    workers = pl.min(total_work, POST_SPMD_BLOCKS)
    for worker in pl.spmd(workers, name_hint="hc_post"):
        for block in pl.range(worker, total_work, POST_SPMD_BLOCKS):
            pair_block = block // D_TILES
            t = pair_block // (HC_MULT // OUT_TILE)
            out_base = (pair_block % (HC_MULT // OUT_TILE)) * OUT_TILE
            d0 = (block % D_TILES) * D_TILE
            x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
            residual0 = residual_flat[t : t + 1, d0 : d0 + D_TILE]
            residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + D_TILE]
            residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + D_TILE]
            residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + D_TILE]
            for out_offset in pl.unroll(OUT_TILE):
                out_h = out_base + out_offset
                value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
                value = pl.add(value, pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h])))
                value = pl.add(value, pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h])))
                value = pl.add(value, pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h])))
                value = pl.add(value, pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h])))
                output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = pl.cast(
                    pl.cast(value, target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
    return output


@pl.jit.inline
def hc_post_after(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    sublayer_ready: pl.Scalar[pl.TASK_ID],
    mixes_ready: pl.Scalar[pl.TASK_ID],
):
    """Run mHC residual expansion after the sublayer output is complete."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    total_work = t_dim * HC_MULT
    workers = pl.min(total_work, POST_SPMD_BLOCKS)
    with pl.spmd(workers, name_hint="hc_post", deps=[sublayer_ready, mixes_ready]):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, total_work, POST_SPMD_BLOCKS):
            t = block // HC_MULT
            out_h = block % HC_MULT
            for d0 in pl.pipeline(0, D, 256, stage=2):
                x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + 256], target_type=pl.FP32)
                value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
                for in_h in pl.unroll(HC_MULT):
                    residual_tile = residual_flat[t : t + 1, in_h * D + d0 : in_h * D + d0 + 256]
                    value = pl.add(
                        value,
                        pl.mul(residual_tile, pl.read(residual_mix_flat, [t, in_h * HC_MULT + out_h])),
                    )
                output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + 256] = pl.cast(
                    pl.cast(value, target_type=pl.BF16, mode="rint"),
                    target_type=pl.FP32,
                )
    return output


@pl.jit.inline
def prefill_sp_post(
    sublayer: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[SP_T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run local mHC-post, writing active values and zero padding in one producer.

    A later pure-output clear is not an ordering edge for the post task. Each
    residual element is instead written exactly once, with hc_post's FP32
    accumulation order and single BF16 rounding boundary.
    """
    tokens = pl.tensor.dim(residual, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    for block in pl.spmd(tokens * HC_MULT, name_hint="sp_mhc_post"):
        row = block // HC_MULT
        out_h = block % HC_MULT
        for col in pl.pipeline(0, D, 256, stage=2):
            value = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            if tp_rank * tokens + row < num_tokens:
                hidden = pl.cast(sublayer[row:row + 1, col:col + 256], pl.FP32)
                value = pl.mul(hidden, pl.read(post_mix, [row, out_h]))
                for in_h in pl.unroll(HC_MULT):
                    lane = residual_flat[row:row + 1, in_h * D + col:in_h * D + col + 256]
                    value = pl.add(value, pl.mul(lane, pl.read(mix_flat, [row, in_h * HC_MULT + out_h])))
            output_flat[row:row + 1, out_h * D + col:out_h * D + col + 256] = pl.cast(
                pl.cast(value, pl.BF16, mode="rint"), pl.FP32,
            )
    for row in pl.spmd(tokens, name_hint="sp_mix_zero_padding"):
        if tp_rank * tokens + row >= num_tokens:
            zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return output, next_pre_mix


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the final HC streams with the delayed pre-mix via hc_pre."""
    return hc_pre(x_hc, pre_mix, output)


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = M.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    mixes = F.linear(flat, function.float()) * inverse_rms
    hc = M.hc_mult
    pre_raw, post_raw, residual_raw = mixes.split((hc, hc, hc * hc), dim=-1)
    pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + M.hc_eps
    post = 2 * torch.sigmoid(post_raw * scale[1] + base[hc : 2 * hc])
    residual_base = base[2 * hc :].unflatten(-1, (hc, hc))
    residual = (residual_raw.unflatten(-1, (hc, hc)) * scale[2] + residual_base).softmax(dim=-1)
    residual = residual + M.hc_eps
    residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(dim=-1, keepdim=True) + M.hc_eps)
        residual = residual / (residual.sum(dim=-2, keepdim=True) + M.hc_eps)
    return pre, post, residual


def golden_mhc_mixes_case(tensors):
    """Fill the expected coefficient outputs."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"]
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix


def golden_hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


def golden_mhc_pre_case(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_pre(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return (update.float() + skip.float()).to(sublayer.dtype)


@pl.jit
def l2_hc_mixes(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Run mHC coefficient generation for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    hc_mixes(x_hc, function, scale, base, pre_mix, post_mix, residual_mix)
    return residual_mix


def build_mhc_mixes_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and outputs for coefficient validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run mHC stream collapse for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_pre(x_hc, pre_mix, output)
    return output


def build_mhc_pre_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 HC mixes and pre validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--op", choices=["mixes", "pre", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = [
        ("mixes", l2_hc_mixes, build_mhc_mixes_tensor_specs, golden_mhc_mixes_case),
        ("pre", l2_hc_pre, build_mhc_pre_tensor_specs, golden_mhc_pre_case),
    ]
    failures = []
    for name, fn, build_specs, golden_fn in cases:
        if args.op != "all" and args.op != name:
            continue
        print(f"[HC] {name}", flush=True)
        if name == "mixes":
            compare_fn = {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            }
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        result = run(
            fn=fn,
            specs=build_specs(args.batch, args.sequence),
            golden_fn=golden_fn,
            config={"platform": args.platform, "device_id": args.device},
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare_fn,
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))
