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
SINKHORN_LOAD_T_TILE = 8
SINKHORN_T_TILE = 2
RMS_K_TILE = 1024
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
D_TILE = 2560
PRE_T_TILE = 2
POST_WORKER_TILE = 64


@pl.jit.inline
def hc_sinkhorn(
    inv_rms: pl.Tensor,
    partials: pl.Tensor,
    scale: pl.Tensor,
    base: pl.Tensor,
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    """Normalize two independent HC matrices in the diagonal blocks of one 8x8 tile."""
    tokens = pl.tensor.dim(pre_mix, 0)
    linear_tokens = pl.tensor.dim(inv_rms, 0)
    base_row = pl.reshape(base, [1, MIX_HC])
    residual_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    with pl.spmd(blocks, name_hint="mhc_sinkhorn") as ready:
        block = pl.tile.get_block_idx()
        t0 = block * SINKHORN_T_TILE
        valid_rows = pl.min(SINKHORN_T_TILE, tokens - t0)
        load_t0 = t0 // SINKHORN_LOAD_T_TILE * SINKHORN_LOAD_T_TILE
        local_t0 = t0 - load_t0
        total = pl.tile.full([SINKHORN_LOAD_T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)
        for split in pl.unroll(LINEAR_OK):
            offset = split * linear_tokens + load_t0
            part = pl.load(partials, [offset, 0], [SINKHORN_LOAD_T_TILE, MIX_PAD])
            total = pl.add(total, part)
        inverse = pl.load(inv_rms, [load_t0, 0], [SINKHORN_LOAD_T_TILE, 1])
        normalized = pl.row_expand_mul(total, inverse)
        bias = pl.load(base_row, [0, 0], [1, MIX_PAD], valid_shape=[1, MIX_HC])
        bias = pl.fillpad(bias, pad_value=pl.PadValue.zero)
        pre_raw = pl.slice(normalized, [SINKHORN_LOAD_T_TILE, HC_PAD], [0, 0])
        pre_bias = pl.slice(bias, [1, HC_PAD], [0, 0])
        pre_scaled = pl.mul(pre_raw, scale0)
        pre_logits = pl.col_expand_add(pre_scaled, pre_bias)
        pre_negated = pl.neg(pre_logits)
        pre_exp = pl.exp(pre_negated)
        pre_denominator = pl.add(pre_exp, 1.0)
        pre = pl.recip(pre_denominator)
        pre = pl.add(pre, HC_EPS)
        pre = pl.slice(pre, [SINKHORN_T_TILE, HC_PAD], [local_t0, 0])
        pre = pl.mul(pre, 1.0)
        pre = pl.set_validshape(pre, valid_rows, HC_MULT)
        pl.store(pre, [t0, 0], pre_mix)
        post_raw = pl.slice(normalized, [SINKHORN_LOAD_T_TILE, HC_PAD], [0, HC_MULT])
        post_bias = pl.slice(bias, [1, HC_PAD], [0, HC_MULT])
        post_scaled = pl.mul(post_raw, scale1)
        post_logits = pl.col_expand_add(post_scaled, post_bias)
        post_negated = pl.neg(post_logits)
        post_exp = pl.exp(post_negated)
        post_denominator = pl.add(post_exp, 1.0)
        post = pl.recip(post_denominator)
        post = pl.mul(post, 2.0)
        post = pl.slice(post, [SINKHORN_T_TILE, HC_PAD], [local_t0, 0])
        post = pl.mul(post, 1.0)
        post = pl.set_validshape(post, valid_rows, HC_MULT)
        pl.store(post, [t0, 0], post_mix)
        comb_scaled = pl.mul(normalized, scale2)
        comb_logits = pl.col_expand_add(comb_scaled, bias)
        comb_logits = pl.slice(comb_logits, [SINKHORN_T_TILE, MIX_PAD], [local_t0, 0])
        comb_logits = pl.mul(comb_logits, 1.0)
        comb_logits = pl.set_validshape(comb_logits, valid_rows, MIX_HC)
        comb_logits = pl.fillpad(comb_logits, pad_value=pl.PadValue.zero)
        comb_logits = pl.set_validshape(comb_logits, SINKHORN_T_TILE, MIX_PAD)
        flat_logits = pl.reshape(comb_logits, [1, SINKHORN_T_TILE * MIX_PAD])
        lanes = pl.tile.arange(0, [1, HC_PAD * HC_PAD], dtype=pl.INT32)
        token = pl.shrs(lanes, 5)
        row = pl.shrs(lanes, 3)
        row = pl.ands(row, 3)
        column = pl.ands(lanes, 3)
        token_offset = pl.mul(token, MIX_PAD)
        row_offset = pl.mul(row, HC_MULT)
        indices = pl.add(token_offset, row_offset)
        indices = pl.add(indices, column)
        indices = pl.add(indices, HC_MULT * 2)
        gather_tmp = pl.create_tile([1, HC_PAD * HC_PAD], dtype=pl.FP32)
        gathered = pl.tile.gather(flat_logits, indices, gather_tmp)
        rows = pl.reshape(gathered, [HC_PAD, HC_PAD])
        column_token = pl.shrs(lanes, 2)
        column_token = pl.ands(column_token, 1)
        mask_tmp = pl.create_tile([1, HC_PAD * HC_PAD], dtype=pl.INT32)
        different_token = pl.xor(token, column_token, mask_tmp)
        diagonal = pl.mul(different_token, -1)
        diagonal = pl.add(diagonal, 1)
        diagonal = pl.cast(diagonal, pl.FP32)
        diagonal = pl.reshape(diagonal, [HC_PAD, HC_PAD])
        row_tmp = pl.create_tile([HC_PAD, HC_PAD], dtype=pl.FP32)
        maximum = pl.row_max(rows, row_tmp)
        rows = pl.row_expand_sub(rows, maximum)
        rows = pl.exp(rows)
        rows = pl.mul(rows, diagonal)
        row_sum = pl.row_sum(rows, row_tmp)
        rows = pl.row_expand_div(rows, row_sum)
        rows = pl.add(rows, HC_EPS)
        rows = pl.mul(rows, diagonal)
        column_sum = pl.col_sum(rows)
        column_sum = pl.add(column_sum, HC_EPS)
        rows = pl.col_expand_div(rows, column_sum)
        for _ in pl.unroll(HC_SINKHORN_ITER - 1):
            row_sum = pl.row_sum(rows, row_tmp)
            row_sum = pl.add(row_sum, HC_EPS)
            rows = pl.row_expand_div(rows, row_sum)
            column_sum = pl.col_sum(rows)
            column_sum = pl.add(column_sum, HC_EPS)
            rows = pl.col_expand_div(rows, column_sum)
        output_lanes = pl.tile.arange(0, [1, 2 * HC_MULT * HC_MULT], dtype=pl.INT32)
        output_token = pl.shrs(output_lanes, 4)
        output_row = pl.shrs(output_lanes, 2)
        output_row = pl.ands(output_row, 3)
        output_column = pl.ands(output_lanes, 3)
        output_token_offset = pl.mul(output_token, HC_MULT * (HC_PAD + 1))
        output_row_offset = pl.mul(output_row, HC_PAD)
        output_indices = pl.add(output_token_offset, output_row_offset)
        output_indices = pl.add(output_indices, output_column)
        flat_rows = pl.reshape(rows, [1, HC_PAD * HC_PAD])
        output_tmp = pl.create_tile([1, 2 * HC_MULT * HC_MULT], dtype=pl.FP32)
        packed = pl.tile.gather(flat_rows, output_indices, output_tmp)
        output = pl.reshape(packed, [SINKHORN_T_TILE, HC_MULT * HC_MULT])
        output = pl.set_validshape(output, valid_rows, HC_MULT * HC_MULT)
        pl.store(output, [t0, 0], residual_flat)
    return ready


@pl.jit.inline
def hc_project(x_hc: pl.Tensor, function: pl.Tensor):
    """Compute the inverse RMS and split-K coefficient projection."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

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

    return inv_rms, mixes_partials


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for block in pl.spmd((t_dim + PRE_T_TILE - 1) // PRE_T_TILE, name_hint="hc_collapse", allow_early_resolve=True):
        for t in pl.range(block * PRE_T_TILE, pl.min((block + 1) * PRE_T_TILE, t_dim)):
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
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                y_tile = pl.add(y01, y23)
                y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.inline
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
    """Generate delayed HC coefficients and collapse the incoming streams."""
    inv_rms, partials = hc_project(x_hc, function)
    mixes_ready = hc_sinkhorn(inv_rms, partials, scale, base, next_pre_mix, post_mix, residual_mix)
    hc_head(x_hc, incoming_pre_mix, output)
    return mixes_ready


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        for row in pl.spmd(tokens, name_hint="decode_sp_zero_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.inline(auto_scope=False)
def hc_post_block(
    sublayer: pl.Tensor,
    residual_flat: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix_flat: pl.Tensor,
    output_flat: pl.Tensor,
    block: pl.Scalar[pl.INDEX],
    stream_tile: pl.constexpr,
):
    """Mix one hidden-width tile for one or four output streams."""
    pair_block = block // (D // D_TILE)
    t = pair_block // (HC_MULT // stream_tile)
    out_base = (pair_block % (HC_MULT // stream_tile)) * stream_tile
    d0 = (block % (D // D_TILE)) * D_TILE
    x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + D_TILE], pl.FP32)
    residual0 = residual_flat[t : t + 1, d0 : d0 + D_TILE]
    residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + D_TILE]
    residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + D_TILE]
    residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + D_TILE]
    for out_offset in pl.unroll(stream_tile):
        out_h = out_base + out_offset
        value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
        term0 = pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h]))
        value = pl.add(value, term0)
        term1 = pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h]))
        value = pl.add(value, term1)
        term2 = pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h]))
        value = pl.add(value, term2)
        term3 = pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h]))
        value = pl.add(value, term3)
        rounded = pl.cast(value, pl.BF16, mode="rint")
        output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = pl.cast(rounded, pl.FP32)
    return output_flat


@pl.jit.inline
def hc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    deps: pl.Array[2, pl.TASK_ID],
):
    """Mix all four residual streams after the supplied producer dependencies."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    total_work = t_dim * (D // D_TILE)
    full_work = total_work // POST_WORKER_TILE * POST_WORKER_TILE
    tail_work = (total_work - full_work) * HC_MULT
    workers = pl.min(total_work * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        worker = pl.tile.get_block_idx()
        for block in pl.pipeline(worker, full_work, POST_WORKER_TILE, stage=1):
            hc_post_block(sublayer, residual_flat, post_mix, residual_mix_flat, output_flat, block, HC_MULT)
        for tail in pl.range(worker, tail_work, POST_WORKER_TILE):
            tail_chunk = full_work + tail // HC_MULT
            tail_stream = (tail_chunk // (D // D_TILE)) * HC_MULT + tail % HC_MULT
            tail_block = tail_stream * (D // D_TILE) + tail_chunk % (D // D_TILE)
            hc_post_block(sublayer, residual_flat, post_mix, residual_mix_flat, output_flat, tail_block, 1)
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


def golden_hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


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


def golden_l2_hc_pre(tensors):
    """Compute the four outputs with the incoming mix independent of the next mix."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])
    tensors["output"][:] = collapsed.to(torch.bfloat16)


def golden_l2_hc_post(tensors):
    """Fill the expanded residual stream with BF16 rounding and FP32 storage."""
    tensors["output"][:] = golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ).float()


def golden_l2_hc_head(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    # The residual stream holds BF16 values; FP32 storage only keeps hc_post's single rounding.
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + M.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Validate the shared HcPre computation."""
    x_hc.bind_dynamic(0, T_DYN)
    incoming_pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    hc_pre(x_hc, incoming_pre_mix, function, scale, base, output, pre_mix, post_mix, residual_mix)
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)
    incoming_generator = torch.Generator().manual_seed(5)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_incoming_pre_mix():
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=incoming_generator)) + HC_EPS

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_incoming_pre_mix),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("output", [tokens, D], torch.bfloat16),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


@pl.jit
def l2_hc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Validate HC post mapping and residual mixing."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


def build_l2_hc_post_specs(batch: int = 33, sequence: int = 1):
    """Build BF16-valued residual streams and nonsymmetric mixing coefficients."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(7)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16().float()

    def init_post_mix():
        return 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))

    def init_residual_mix():
        mix = torch.rand(tokens, HC_MULT, HC_MULT, generator=generator) + HC_EPS
        for _ in range(HC_SINKHORN_ITER):
            mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
            mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
        return mix

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


@pl.jit
def l2_hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Validate the final HC stream collapse."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    hc_head(x_hc, pre_mix, output)
    return output


def build_l2_hc_head_specs(batch: int = 33, sequence: int = 1):
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

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 HC pre, post, and head validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=33)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--op", choices=["pre", "post", "head", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, choices=[1, 2, 3, 4], default=0)
    args = parser.parse_args()
    if args.op == "all" and (args.golden_data or args.runtime_dir):
        parser.error("replay requires selecting one --op")
    cases = [
        ("pre", l2_hc_pre, build_l2_hc_pre_specs, golden_l2_hc_pre),
        ("post", l2_hc_post, build_l2_hc_post_specs, golden_l2_hc_post),
        ("head", l2_hc_head, build_l2_hc_head_specs, golden_l2_hc_head),
    ]
    failures = []
    for name, fn, build_specs, golden_fn in cases:
        if args.op != "all" and args.op != name:
            continue
        print(f"[HC] {name}", flush=True)
        if name == "pre":
            compare_fn = {
                "pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "post_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "residual_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            }
            compare_fn["output"] = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
        else:
            compare_fn = {"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)}
        config = {"backend_type": args.backend_type, "execution_mode": args.execution_mode}
        config.update(device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane)
        result = run(
            fn=fn,
            specs=build_specs(args.batch, args.sequence),
            golden_fn=golden_fn,
            config=config,
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare_fn,
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        raise SystemExit("; ".join(failures))
