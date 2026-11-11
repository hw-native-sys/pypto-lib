# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""V4.1 HC coefficient generation, stream collapse, residual expansion, and final collapse."""

import math
from pathlib import Path

import pypto.language as pl
import torch
import torch.nn.functional as F
from pypto.runtime import pto_isa_include_dir

from config import FLASH as M, T_DYN



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
HF32_LINEAR_T_TILE = 80  # Matches M_TILE in kernels/hc_linear.cpp.
SINKHORN_T_TILE = 6
SINKHORN_WIDE_T_TILE = 16
SINKHORN_WORKER_TILE = 12
RMS_K_TILE = 1024
LINEAR_OK = 4
D_TILE = 2560
PRE_T_TILE = 8
PRE_NORM_T_TILE = 8
PRE_NORM_STORE_T_TILE = 2
POST_WORKER_TILE = 64


_KERNELS = Path(__file__).parent / "kernels"


@pl.jit.extern(core_type="aic", source=_KERNELS / "hc_linear.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_linear(
    x: pl.Tensor,
    weight: pl.Tensor,
    partials: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
    padded_rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project FP32 HC operands with nearest-away HF32 and FP32 accumulation."""
    ...


@pl.jit.inline
def hc_sinkhorn(
    inv_rms: pl.Tensor, partials: pl.Tensor, scale: pl.Tensor, base: pl.Tensor,
    pre_mix: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
):
    """Normalize six HC matrices per worker, keeping all four rows in registers."""
    tokens = pl.tensor.dim(pre_mix, 0)
    padded_rows = pl.tensor.dim(inv_rms, 0)
    inverse_flat = pl.reshape(inv_rms, [1, padded_rows])
    scale_flat = pl.reshape(scale, [1, 3])
    bias_flat = pl.reshape(base, [1, MIX_HC])
    comb_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    blocks = (tokens + SINKHORN_T_TILE - 1) // SINKHORN_T_TILE
    workers = pl.min(blocks, SINKHORN_WORKER_TILE)
    with pl.spmd(workers, name_hint="mhc_sinkhorn") as ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, blocks, workers):
            start = block * SINKHORN_T_TILE
            rows = pl.min(SINKHORN_T_TILE, tokens - start)
            p0 = pl.load(partials, [start, 0], [6, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
            p1 = pl.load(partials, [padded_rows + start, 0], [6, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
            p2 = pl.load(partials, [2 * padded_rows + start, 0], [6, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
            p3 = pl.load(partials, [3 * padded_rows + start, 0], [6, 32], valid_shape=[rows, 32], target_memory=pl.MemorySpace.Vec)
            inverse = pl.load(inverse_flat, [0, start], [1, 32], valid_shape=[1, rows], target_memory=pl.MemorySpace.Vec)
            bias = pl.load(bias_flat, [0, 0], [1, 32], valid_shape=[1, MIX_HC], target_memory=pl.MemorySpace.Vec)
            scales = pl.load(scale_flat, [0, 0], [1, 8], valid_shape=[1, 3], target_memory=pl.MemorySpace.Vec)
            s0_scalar = pl.tile.read(scales, [0, 0])
            s1_scalar = pl.tile.read(scales, [0, 1])
            s2_scalar = pl.tile.read(scales, [0, 2])
            # Scatter overwrites every valid output element before it is stored.
            # Allocate UB without initializing padding, matching the C++ kernel.
            pre = pl.create_tile([48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            post = pl.create_tile([48], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            comb = pl.create_tile([96], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            with pl.reg_scope():
                full = pl.reg.full_mask(lanes=64)
                lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                mask = pl.reg.active_mask(rows * HC_MULT, lanes=64)
                token = pl.reg.shift_rights(lane, 2, full)
                low_bits = pl.reg.broadcast(3, lanes=64, dtype=pl.UINT32)
                column = pl.reg.bitwise_and(lane, low_bits, full)
                index = pl.reg.shift_lefts(token, 5, full)
                index = pl.reg.add(index, column, full)
                index4 = pl.reg.adds(index, 4, full)
                index8 = pl.reg.adds(index, 8, full)
                index12 = pl.reg.adds(index, 12, full)
                index16 = pl.reg.adds(index, 16, full)
                index20 = pl.reg.adds(index, 20, full)
                bias_index4 = pl.reg.adds(column, 4, full)
                bias_index8 = pl.reg.adds(column, 8, full)
                bias_index12 = pl.reg.adds(column, 12, full)
                bias_index16 = pl.reg.adds(column, 16, full)
                bias_index20 = pl.reg.adds(column, 20, full)
                pre_index = pl.reg.shift_lefts(token, 3, full)
                pre_index = pl.reg.add(pre_index, column, full)
                comb_index = pl.reg.shift_lefts(token, 4, full)
                comb_index = pl.reg.add(comb_index, column, full)
                comb_index4 = pl.reg.adds(comb_index, 4, full)
                comb_index8 = pl.reg.adds(comb_index, 8, full)
                comb_index12 = pl.reg.adds(comb_index, 12, full)
                xor1 = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
                xor2 = pl.reg.broadcast(2, lanes=64, dtype=pl.UINT32)
                perm1 = pl.reg.bitwise_xor(lane, xor1, full)
                perm2 = pl.reg.bitwise_xor(lane, xor2, full)
                rms = pl.reg.gather(inverse, token, mask)
                s0 = pl.reg.broadcast(s0_scalar, lanes=64, dtype=pl.FP32)
                s1 = pl.reg.broadcast(s1_scalar, lanes=64, dtype=pl.FP32)
                s2 = pl.reg.broadcast(s2_scalar, lanes=64, dtype=pl.FP32)
                one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
                pre_value = pl.reg.gather(p0, index, mask)
                part = pl.reg.gather(p1, index, mask)
                pre_value = pl.reg.add(pre_value, part, mask)
                part = pl.reg.gather(p2, index, mask)
                pre_value = pl.reg.add(pre_value, part, mask)
                part = pl.reg.gather(p3, index, mask)
                pre_value = pl.reg.add(pre_value, part, mask)
                pre_value = pl.reg.mul(pre_value, rms, mask)
                pre_value = pl.reg.mul(pre_value, s0, mask)
                bias_value = pl.reg.gather(bias, column, mask)
                pre_value = pl.reg.add(pre_value, bias_value, mask)
                post_value = pl.reg.gather(p0, index4, mask)
                part = pl.reg.gather(p1, index4, mask)
                post_value = pl.reg.add(post_value, part, mask)
                part = pl.reg.gather(p2, index4, mask)
                post_value = pl.reg.add(post_value, part, mask)
                part = pl.reg.gather(p3, index4, mask)
                post_value = pl.reg.add(post_value, part, mask)
                post_value = pl.reg.mul(post_value, rms, mask)
                post_value = pl.reg.mul(post_value, s1, mask)
                bias_value = pl.reg.gather(bias, bias_index4, mask)
                post_value = pl.reg.add(post_value, bias_value, mask)
                m0 = pl.reg.gather(p0, index8, mask)
                part = pl.reg.gather(p1, index8, mask)
                m0 = pl.reg.add(m0, part, mask)
                part = pl.reg.gather(p2, index8, mask)
                m0 = pl.reg.add(m0, part, mask)
                part = pl.reg.gather(p3, index8, mask)
                m0 = pl.reg.add(m0, part, mask)
                m0 = pl.reg.mul(m0, rms, mask)
                m0 = pl.reg.mul(m0, s2, mask)
                bias_value = pl.reg.gather(bias, bias_index8, mask)
                m0 = pl.reg.add(m0, bias_value, mask)
                m1 = pl.reg.gather(p0, index12, mask)
                part = pl.reg.gather(p1, index12, mask)
                m1 = pl.reg.add(m1, part, mask)
                part = pl.reg.gather(p2, index12, mask)
                m1 = pl.reg.add(m1, part, mask)
                part = pl.reg.gather(p3, index12, mask)
                m1 = pl.reg.add(m1, part, mask)
                m1 = pl.reg.mul(m1, rms, mask)
                m1 = pl.reg.mul(m1, s2, mask)
                bias_value = pl.reg.gather(bias, bias_index12, mask)
                m1 = pl.reg.add(m1, bias_value, mask)
                m2 = pl.reg.gather(p0, index16, mask)
                part = pl.reg.gather(p1, index16, mask)
                m2 = pl.reg.add(m2, part, mask)
                part = pl.reg.gather(p2, index16, mask)
                m2 = pl.reg.add(m2, part, mask)
                part = pl.reg.gather(p3, index16, mask)
                m2 = pl.reg.add(m2, part, mask)
                m2 = pl.reg.mul(m2, rms, mask)
                m2 = pl.reg.mul(m2, s2, mask)
                bias_value = pl.reg.gather(bias, bias_index16, mask)
                m2 = pl.reg.add(m2, bias_value, mask)
                m3 = pl.reg.gather(p0, index20, mask)
                part = pl.reg.gather(p1, index20, mask)
                m3 = pl.reg.add(m3, part, mask)
                part = pl.reg.gather(p2, index20, mask)
                m3 = pl.reg.add(m3, part, mask)
                part = pl.reg.gather(p3, index20, mask)
                m3 = pl.reg.add(m3, part, mask)
                m3 = pl.reg.mul(m3, rms, mask)
                m3 = pl.reg.mul(m3, s2, mask)
                bias_value = pl.reg.gather(bias, bias_index20, mask)
                m3 = pl.reg.add(m3, bias_value, mask)
                pre_value = pl.reg.muls(pre_value, -1.0, mask)
                pre_value = pl.reg.exp(pre_value, mask)
                pre_value = pl.reg.adds(pre_value, 1.0, mask)
                pre_value = pl.reg.div(one, pre_value, mask)
                pre_value = pl.reg.adds(pre_value, HC_EPS, mask)
                pre = pl.reg.scatter(pre, pre_index, pre_value, mask)
                post_value = pl.reg.muls(post_value, -1.0, mask)
                post_value = pl.reg.exp(post_value, mask)
                post_value = pl.reg.adds(post_value, 1.0, mask)
                post_value = pl.reg.div(one, post_value, mask)
                post_value = pl.reg.muls(post_value, 2.0, mask)
                post = pl.reg.scatter(post, pre_index, post_value, mask)
                # XOR1 pairs adjacent columns, XOR2 combines the two pairs.
                # This keeps the original (c0+c1)+(c2+c3) reduction order.
                adjacent = pl.reg.gather_lanes(m0, perm1)
                red = pl.reg.maximum(m0, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.maximum(red, adjacent, full)
                m0 = pl.reg.sub(m0, red, mask)
                m0 = pl.reg.exp(m0, mask)
                adjacent = pl.reg.gather_lanes(m0, perm1)
                red = pl.reg.add(m0, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.add(red, adjacent, full)
                m0 = pl.reg.div(m0, red, mask)
                m0 = pl.reg.adds(m0, HC_EPS, mask)
                adjacent = pl.reg.gather_lanes(m1, perm1)
                red = pl.reg.maximum(m1, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.maximum(red, adjacent, full)
                m1 = pl.reg.sub(m1, red, mask)
                m1 = pl.reg.exp(m1, mask)
                adjacent = pl.reg.gather_lanes(m1, perm1)
                red = pl.reg.add(m1, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.add(red, adjacent, full)
                m1 = pl.reg.div(m1, red, mask)
                m1 = pl.reg.adds(m1, HC_EPS, mask)
                adjacent = pl.reg.gather_lanes(m2, perm1)
                red = pl.reg.maximum(m2, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.maximum(red, adjacent, full)
                m2 = pl.reg.sub(m2, red, mask)
                m2 = pl.reg.exp(m2, mask)
                adjacent = pl.reg.gather_lanes(m2, perm1)
                red = pl.reg.add(m2, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.add(red, adjacent, full)
                m2 = pl.reg.div(m2, red, mask)
                m2 = pl.reg.adds(m2, HC_EPS, mask)
                adjacent = pl.reg.gather_lanes(m3, perm1)
                red = pl.reg.maximum(m3, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.maximum(red, adjacent, full)
                m3 = pl.reg.sub(m3, red, mask)
                m3 = pl.reg.exp(m3, mask)
                adjacent = pl.reg.gather_lanes(m3, perm1)
                red = pl.reg.add(m3, adjacent, full)
                adjacent = pl.reg.gather_lanes(red, perm2)
                red = pl.reg.add(red, adjacent, full)
                m3 = pl.reg.div(m3, red, mask)
                m3 = pl.reg.adds(m3, HC_EPS, mask)
                total = pl.reg.add(m0, m1, mask)
                total = pl.reg.add(total, m2, mask)
                total = pl.reg.add(total, m3, mask)
                total = pl.reg.adds(total, HC_EPS, mask)
                m0 = pl.reg.div(m0, total, mask)
                m1 = pl.reg.div(m1, total, mask)
                m2 = pl.reg.div(m2, total, mask)
                m3 = pl.reg.div(m3, total, mask)
                for iteration, (m0, m1, m2, m3) in pl.hardware_loop(HC_SINKHORN_ITER - 1, init_values=(m0, m1, m2, m3)):
                    adjacent = pl.reg.gather_lanes(m0, perm1)
                    red = pl.reg.add(m0, adjacent, full)
                    adjacent = pl.reg.gather_lanes(red, perm2)
                    red = pl.reg.add(red, adjacent, full)
                    red = pl.reg.adds(red, HC_EPS, mask)
                    m0 = pl.reg.div(m0, red, mask)
                    adjacent = pl.reg.gather_lanes(m1, perm1)
                    red = pl.reg.add(m1, adjacent, full)
                    adjacent = pl.reg.gather_lanes(red, perm2)
                    red = pl.reg.add(red, adjacent, full)
                    red = pl.reg.adds(red, HC_EPS, mask)
                    m1 = pl.reg.div(m1, red, mask)
                    adjacent = pl.reg.gather_lanes(m2, perm1)
                    red = pl.reg.add(m2, adjacent, full)
                    adjacent = pl.reg.gather_lanes(red, perm2)
                    red = pl.reg.add(red, adjacent, full)
                    red = pl.reg.adds(red, HC_EPS, mask)
                    m2 = pl.reg.div(m2, red, mask)
                    adjacent = pl.reg.gather_lanes(m3, perm1)
                    red = pl.reg.add(m3, adjacent, full)
                    adjacent = pl.reg.gather_lanes(red, perm2)
                    red = pl.reg.add(red, adjacent, full)
                    red = pl.reg.adds(red, HC_EPS, mask)
                    m3 = pl.reg.div(m3, red, mask)
                    total = pl.reg.add(m0, m1, mask)
                    total = pl.reg.add(total, m2, mask)
                    total = pl.reg.add(total, m3, mask)
                    total = pl.reg.adds(total, HC_EPS, mask)
                    m0 = pl.reg.div(m0, total, mask)
                    m1 = pl.reg.div(m1, total, mask)
                    m2 = pl.reg.div(m2, total, mask)
                    m3 = pl.reg.div(m3, total, mask)
                    m0, m1, m2, m3 = pl.yield_(m0, m1, m2, m3)
                comb = pl.reg.scatter(comb, comb_index, m0, mask)
                comb = pl.reg.scatter(comb, comb_index4, m1, mask)
                comb = pl.reg.scatter(comb, comb_index8, m2, mask)
                comb = pl.reg.scatter(comb, comb_index12, m3, mask)
            pre = pl.reshape(pre, [6, 8])
            post = pl.reshape(post, [6, 8])
            comb = pl.reshape(comb, [6, 16])
            pre = pl.set_validshape(pre, rows, HC_MULT)
            post = pl.set_validshape(post, rows, HC_MULT)
            comb = pl.set_validshape(comb, rows, HC_MULT * HC_MULT)
            pl.store(pre, [start, 0], pre_mix)
            pl.store(post, [start, 0], post_mix)
            pl.store(comb, [start, 0], comb_flat)
    return ready



@pl.jit.inline
def hc_project(x_hc: pl.Tensor, function: pl.Tensor, rms_worker_limit: pl.constexpr):
    """Compute FP32 inverse RMS and split-K HC projection coefficients."""
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    rms_blocks = (t_dim + T_TILE - 1) // T_TILE
    rms_workers = pl.max(pl.min(rms_blocks, rms_worker_limit), 1)
    for worker in pl.spmd(rms_workers, name_hint="hc_rms"):
        for block in pl.range(worker, rms_blocks, rms_workers):
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
    tasks = ((t_dim + HF32_LINEAR_T_TILE - 1) // HF32_LINEAR_T_TILE) * LINEAR_OK
    workers = pl.min(tasks, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="hc_linear"):
        _hc_linear(x_flat, function, mixes_partials, t_dim, t_linear)

    return inv_rms, mixes_partials


@pl.jit.extern(core_type="aiv", source=_KERNELS / "hc_collapse.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_collapse(
    x: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor,
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    collapsed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INDEX],
    mode: pl.Scalar[pl.INDEX],
):
    """Collapse streams; mode 1 normalizes, and mode 2 also stores raw BF16 rows."""
    ...


@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
    worker_limit: pl.constexpr,
):
    """Collapse HC streams with the delayed pre-mix and BF16 rounding."""
    t_dim = pl.tensor.dim(x_hc, 0)
    input_view = pl.reshape(x_hc, [t_dim, HC_MULT, D])
    mix_view = pl.reshape(pre_mix, [t_dim, HC_MULT])
    output_view = pl.reshape(output, [t_dim, D])
    collapsed_view = pl.reshape(output, [t_dim, D])
    workers = pl.max(pl.min((t_dim + PRE_T_TILE - 1) // PRE_T_TILE, worker_limit), 1)
    with pl.spmd(workers, name_hint="hc_collapse", allow_early_resolve=True):
        _hc_collapse(input_view, mix_view, mix_view, output_view, collapsed_view, t_dim, t_dim, 0)
    return output


@pl.jit.inline(auto_scope=False)
def hc_head_norm(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor,
    worker_limit: pl.constexpr,
):
    """Collapse eight HC rows and normalize their rounded BF16 values."""
    tokens = pl.tensor.dim(x_hc, 0)
    workers = pl.min((tokens + PRE_NORM_T_TILE - 1) // PRE_NORM_T_TILE, worker_limit)
    input_view = pl.reshape(x_hc, [tokens, HC_MULT, D])
    mix_view = pl.reshape(pre_mix, [tokens, HC_MULT])
    output_view = pl.reshape(output, [tokens, D])
    collapsed_view = pl.reshape(output, [tokens, D])
    with pl.spmd(workers, name_hint="hc_collapse_norm") as ready:
        _hc_collapse(input_view, mix_view, norm_weight, output_view, collapsed_view, tokens, tokens, 1)
    return ready


@pl.jit.inline(auto_scope=False)
def hc_head_norm_store(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    norm_weight: pl.Tensor,
    output: pl.Tensor,
    collapsed: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr,
):
    """Collapse all rows, normalize active rows, and zero normalized padding."""
    tokens = pl.tensor.dim(x_hc, 0)
    active_rows = pl.max(pl.min(pl.cast(num_tokens, pl.INDEX), tokens), 0)
    workers = pl.min((tokens + PRE_NORM_STORE_T_TILE - 1) // PRE_NORM_STORE_T_TILE, worker_limit)
    input_view = pl.reshape(x_hc, [tokens, HC_MULT, D])
    mix_view = pl.reshape(pre_mix, [tokens, HC_MULT])
    output_view = pl.reshape(output, [tokens, D])
    collapsed_view = pl.reshape(collapsed, [tokens, D])
    with pl.spmd(workers, name_hint="hc_collapse_norm_store") as ready:
        _hc_collapse(input_view, mix_view, norm_weight, output_view, collapsed_view, tokens, active_rows, 2)
    return ready


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Zero padding rows beyond the local active token count."""
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0 and valid_rows < tokens:
        for row in pl.spmd(tokens, name_hint="zero_bf16_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.extern(core_type="aiv", source=_KERNELS / "hc_post.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _hc_post(
    sublayer: pl.Tensor, residual: pl.Tensor, post_mix: pl.Tensor, residual_mix: pl.Tensor,
    output: pl.Out[pl.Tensor], tokens: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Mix four residual streams with ordered FP32 register operations."""
    ...


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
    tokens = pl.tensor.dim(sublayer, 0)
    workers = pl.min(tokens * (D // D_TILE) * HC_MULT, POST_WORKER_TILE)
    with pl.spmd(workers, name_hint="hc_post", deps=[deps]):
        _hc_post(sublayer, residual, post_mix, residual_mix, output, tokens)
    return output




def _round_hf32(value: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to E8M10, with halfway values away from zero."""
    bits = value.contiguous().view(torch.int32)
    rounded = ((bits + 4096) & -8192).view(torch.float32)
    return torch.where(torch.isfinite(value), rounded, value)


def golden_hc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    iterations: int = M.hc_sinkhorn_iters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one HC stream into pre, post, and doubly-stochastic residual mixes."""
    flat = x_hc.flatten(-2).float()
    inverse_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    projection_input = _round_hf32(flat)
    projection_weight = _round_hf32(function.float())
    mixes = F.linear(projection_input, projection_weight) * inverse_rms
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


def golden_hc_norm(collapsed: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    """Normalize BF16 HC rows with sequential 64-lane FP32 accumulators."""
    value = collapsed.float()
    width = value.shape[-1]
    square = value.square().reshape(*value.shape[:-1], width // 64, 64)
    partial = torch.zeros_like(square[..., 0, :])
    for chunk in range(width // 64):
        partial = partial + square[..., chunk, :]
    while partial.shape[-1] > 1:
        partial = partial[..., 0::2] + partial[..., 1::2]
    argument = partial * (1.0 / width) + NORM_EPS
    inverse = torch.ones_like(argument) / torch.sqrt(argument)
    return (value * inverse * norm_weight.float()).to(torch.bfloat16)


def golden_hc_head_norm(x: torch.Tensor, pre_mix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams to BF16 and normalize the resulting rows."""
    collapsed = golden_hc_collapse(x, pre_mix).bfloat16()
    return golden_hc_norm(collapsed, weight)


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams in FP32."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return update.float() + skip.float()


def golden_l2_hc_pre(tensors):
    """Compute the four outputs with the incoming mix independent of the next mix."""
    pre_mix, post_mix, residual_mix = golden_hc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"],
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"])
    tensors["output"][:] = collapsed.to(torch.bfloat16)


def golden_l2_hc_post(tensors):
    """Fill the expanded residual stream with FP32 accumulation and storage."""
    tensors["output"][:] = golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    )


def golden_l2_hc_head(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_hc_collapse(tensors["x_hc"], tensors["pre_mix"]).to(torch.bfloat16)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")


def make_hc_inputs(tokens, seed, case):
    """Seeded HC fixtures with V4.1 layer-1 attention weight statistics."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()

    # Per-row standard deviations and gate parameters from the V4.1 Flash 1017 checkpoint.
    function_std = torch.tensor([
        0.029376, 0.031490, 0.055490, 0.049310, 0.023323, 0.181760, 0.112393, 0.096347,
        0.033303, 0.034775, 0.035849, 0.039710, 0.033325, 0.030450, 0.034318, 0.033540,
        0.033511, 0.050113, 0.030742, 0.028955, 0.027775, 0.029948, 0.031218, 0.038125,
    ])
    scale = torch.tensor([0.070230, 0.019080, 0.172269])
    base = torch.tensor([
        -2.338990, 2.601740, -2.912670, -3.033931, -0.503342, -0.577662, -3.500769, -2.494710,
        2.750622, -2.458451, -3.305034, -3.223911, -2.937424, 2.832767, -0.933163, -2.659624,
        -2.655424, -1.275563, 2.152474, -3.270429, -2.550060, -2.554717, -2.587215, 2.604832,
    ])

    # Incoming gates use the preceding layer's FFN pre-gate weight statistics.
    incoming_std = torch.tensor([0.041462, 0.027009, 0.054705, 0.025310]) * math.sqrt(HC_DIM) * 0.031293
    incoming_base = torch.tensor([2.531060, -2.808948, -3.796850, -2.947976])
    incoming_logits = torch.randn(tokens, HC_MULT, generator=gen) * incoming_std + incoming_base
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(incoming_logits) + HC_EPS,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) * function_std[:, None],
        "hc_attn_scale": scale,
        "hc_attn_base": base,
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.001745 + 0.021617).bfloat16(),
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
    inv_rms, partials = hc_project(x_hc, function, rms_worker_limit=64)
    hc_sinkhorn(inv_rms, partials, scale, base, pre_mix, post_mix, residual_mix)
    hc_head(x_hc, incoming_pre_mix, output, worker_limit=64)
    return output


def build_l2_hc_pre_specs(batch: int = 33, sequence: int = 1):
    """Build coefficient and independent incoming-mix fixtures."""
    from golden import TensorSpec

    tokens = batch * sequence
    inputs = make_hc_inputs(tokens, seed=3, case="normal")
    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=inputs["x_hc"]),
        TensorSpec("incoming_pre_mix", [tokens, HC_MULT], torch.float32, init_value=inputs["pre_mix"]),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=inputs["hc_attn_fn"]),
        TensorSpec("scale", [3], torch.float32, init_value=inputs["hc_attn_scale"]),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=inputs["hc_attn_base"]),
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
        return torch.randn(tokens, HC_MULT, D, generator=generator)

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
    hc_head(x_hc, pre_mix, output, worker_limit=64)
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
        elif name == "post":
            compare_fn = {"output": ratio_allclose(atol=1e-6, rtol=1e-5)}
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
