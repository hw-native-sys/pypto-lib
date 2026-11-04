# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dense clamped-SwiGLU MLP with W8A8 or BF16 weights and TP-local outputs."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import FLASH, TP_SIZE, D, Glm53FlashConfig
from models.glm5_3_flash.elementwise import accurate_divide, accurate_exp
from models.glm5_3_flash.linear import bf16_linear, golden_bf16_linear
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8
from models.glm5_3_flash.utils import _assert_close, _randn

T_DYN = pl.dynamic('GLM53_T_DYN')

LOCAL_DENSE_INTER = FLASH.intermediate_size // TP_SIZE
SWIGLU_LIMIT = FLASH.swiglu_limit
DN_M_TILE = 16  # cube M-tile: matmul rows must be a multiple of 16 (fractal)
DN_ROW_PAD = 8
DN_ROW_TILE = 2
K_TILE = 256  # gate/up K tile over D = 4096 (right-buffer bound)
MM_INTER_TILE = 128  # gate/up N tile; 2 * LOCAL_DENSE_INTER must be a multiple
ACT_INTER_TILE = min(128, LOCAL_DENSE_INTER)  # 768 = 2*384; smaller TP keeps it aligned
QUANT_TILE = min(384, LOCAL_DENSE_INTER)
INTER_K_TILE = min(128, LOCAL_DENSE_INTER)  # w2 K tile (= local intermediate)
D_OUT_TILE = 128  # w2 N tile over D
D_ACT_TILE = 512
W2_GROUP_TILE = 8
BF16_M_TILE = 16
BF16_N_TILE = 128
BF16_ACT_M_TILE = 8
T_DYN_TEST = 64


@pl.jit.inline
def dense_mlp(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    token_rows = pl.tensor.dim(x_int8, 0)
    for mt in pl.parallel((token_rows + DN_M_TILE - 1) // DN_M_TILE):
        ts0 = mt * DN_M_TILE
        valid_rows = pl.min(pl.cast(DN_M_TILE, pl.INDEX), token_rows - ts0)
        gate_up_i32 = pl.create_tensor([DN_M_TILE, 2 * LOCAL_DENSE_INTER], dtype=pl.INT32)
        for nb in pl.spmd(2 * LOCAL_DENSE_INTER // MM_INTER_TILE, name_hint="dn_gate_up_mm", allow_early_resolve=True):
            n0 = nb * MM_INTER_TILE
            acc = pl.create_tensor([DN_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_int8, [DN_M_TILE, K_TILE], [ts0, k0], valid_shape=[valid_rows, K_TILE])
                w_k = w_gate_up[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, xs_k, w_k, b_trans=True, init_cond=k0 == 0)
            gate_up_i32[:, n0 : n0 + MM_INTER_TILE] = acc
        h_fp32 = pl.create_tensor([DN_M_TILE, LOCAL_DENSE_INTER], dtype=pl.FP32)
        h_i8 = pl.create_tensor([DN_M_TILE, LOCAL_DENSE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([DN_M_TILE, DN_ROW_PAD], dtype=pl.FP32, manual_dep=True)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dn_h_i8_init"):
            for init_k0 in pl.pipeline(0, LOCAL_DENSE_INTER, QUANT_TILE, stage=1):
                zero_fp16 = pl.full([DN_M_TILE, QUANT_TILE], dtype=pl.FP16, value=0.0)
                zero_i8 = pl.cast(zero_fp16, target_type=pl.INT8, mode="trunc")
                h_i8[:, init_k0 : init_k0 + QUANT_TILE] = zero_i8
        n_act_blocks = (valid_rows + DN_ROW_TILE - 1) // DN_ROW_TILE
        for row_block in pl.spmd(n_act_blocks, name_hint="dn_gate_up_act_q"):
            row0 = row_block * DN_ROW_TILE
            blk_rows = pl.min(pl.cast(DN_ROW_TILE, pl.INDEX), valid_rows - row0)
            x_scale_blk = pl.slice(x_scale, [DN_ROW_PAD, 1], [ts0 + row0, 0], valid_shape=[blk_rows, 1])
            row_amax = pl.full([1, DN_ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for part in pl.pipeline(LOCAL_DENSE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_chunk_i32 = pl.slice(gate_up_i32, [DN_ROW_PAD, ACT_INTER_TILE], [row0, n0], valid_shape=[DN_ROW_TILE, ACT_INTER_TILE])
                up_chunk_i32 = pl.slice(gate_up_i32, [DN_ROW_PAD, ACT_INTER_TILE], [row0, LOCAL_DENSE_INTER + n0], valid_shape=[DN_ROW_TILE, ACT_INTER_TILE])
                gate_w_scale = pl.reshape(w_gate_up_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                up_w_scale = pl.reshape(w_gate_up_scale[LOCAL_DENSE_INTER + n0:LOCAL_DENSE_INTER + n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_fp32 = pl.cast(gate_chunk_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_chunk_i32, target_type=pl.FP32, mode="none")
                gate_fp32 = pl.col_expand_mul(pl.row_expand_mul(gate_fp32, x_scale_blk), gate_w_scale)
                up_fp32 = pl.col_expand_mul(pl.row_expand_mul(up_fp32, x_scale_blk), up_w_scale)
                gate_fp32 = pl.cast(pl.cast(gate_fp32, pl.BF16, mode="rint"), pl.FP32)
                up_fp32 = pl.cast(pl.cast(up_fp32, pl.BF16, mode="rint"), pl.FP32)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                gate_buffer = pl.create_tensor([DN_ROW_PAD, ACT_INTER_TILE], dtype=pl.FP32)
                gate_buffer[:, :] = pl.set_validshape(gate_fp32, DN_ROW_PAD, ACT_INTER_TILE)
                exp_argument = pl.create_tensor([DN_ROW_PAD, ACT_INTER_TILE], dtype=pl.FP32)
                exp_argument[:, :] = pl.neg(gate_buffer)
                exponential = accurate_exp(exp_argument)
                denominator = pl.create_tensor([DN_ROW_PAD, ACT_INTER_TILE], dtype=pl.FP32)
                denominator[:, :] = pl.add(exponential, 1.0)
                silu_value = accurate_divide(gate_buffer, denominator)
                silu = pl.cast(silu_value, pl.BF16, mode="rint")
                product = pl.mul(pl.cast(silu, pl.FP32), up_fp32)
                gated = pl.cast(pl.cast(product, pl.BF16, mode="rint"), pl.FP32)
                gated = pl.set_validshape(gated, DN_ROW_TILE, ACT_INTER_TILE)
                chunk_amax = pl.reshape(pl.row_max(pl.abs(gated)), [1, DN_ROW_PAD])
                row_amax = pl.maximum(row_amax, chunk_amax)
                h_fp32[row0:row0 + DN_ROW_TILE, n0:n0 + ACT_INTER_TILE] = gated[0:DN_ROW_TILE, :]
            amax_buffer = pl.create_tensor([1, DN_ROW_PAD], dtype=pl.FP32)
            numerator = pl.create_tensor([1, DN_ROW_PAD], dtype=pl.FP32)
            amax_buffer[:, :] = row_amax
            numerator[:, :] = pl.full([1, DN_ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            row_scale_q_buffer = pl.create_tensor([1, DN_ROW_PAD], dtype=pl.FP32)
            quantization_scale = accurate_divide(numerator, amax_buffer)
            row_scale_q_buffer[:, :] = quantization_scale
            row_scale_q = row_scale_q_buffer
            row_scale_q_col = pl.reshape(row_scale_q, [DN_ROW_PAD, 1])
            one_buffer = pl.create_tensor([1, DN_ROW_PAD], dtype=pl.FP32)
            one_buffer[:, :] = pl.full([1, DN_ROW_PAD], dtype=pl.FP32, value=1.0)
            dequant = accurate_divide(one_buffer, row_scale_q_buffer)
            row_scale_dq_col = pl.reshape(dequant, [DN_ROW_PAD, 1])
            row_scale_dq_full = pl.full([DN_ROW_PAD, DN_ROW_PAD], dtype=pl.FP32, value=0.0)
            row_scale_dq = pl.row_expand(row_scale_dq_full, row_scale_dq_col)
            h_scale_dq[row0 : row0 + DN_ROW_TILE, :] = row_scale_dq[0:DN_ROW_TILE, :]
            for q_idx in pl.pipeline(0, LOCAL_DENSE_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32_slice = pl.slice(h_fp32, [DN_ROW_PAD, QUANT_TILE], [row0, k0], valid_shape=[DN_ROW_TILE, QUANT_TILE])
                h_scaled = pl.row_expand_mul(h_fp32_slice, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_fp16_cast = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")
                h_i8[row0 : row0 + DN_ROW_TILE, k0 : k0 + QUANT_TILE] = h_fp16_cast[0:DN_ROW_TILE, :]
        y_i32 = pl.create_tensor([DN_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="dn_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([DN_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, LOCAL_DENSE_INTER, INTER_K_TILE, stage=2):
                # Keep a reshape view for dimension folding in composed loops.
                h_i8_view = pl.reshape(h_i8, [DN_M_TILE, LOCAL_DENSE_INTER])
                hs_k = h_i8_view[:, k0 : k0 + INTER_K_TILE]
                sw2_k = w_down[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K_TILE]
                y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> FP32.
        for db_idx in pl.spmd(D // (W2_GROUP_TILE * D_ACT_TILE), name_hint="dn_w2_act", allow_early_resolve=True):
            d_base = db_idx * (W2_GROUP_TILE * D_ACT_TILE)
            h_scale = pl.row_max(h_scale_dq[:, :])
            for dg in pl.pipeline(W2_GROUP_TILE, stage=2):
                d0 = d_base + dg * D_ACT_TILE
                y_2d_i32 = y_i32[:, d0 : d0 + D_ACT_TILE]
                w2_scale_chunk = pl.reshape(w_down_scale[d0 : d0 + D_ACT_TILE], [1, D_ACT_TILE])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale), w2_scale_chunk)
                y_2d = pl.cast(pl.cast(y_2d, pl.BF16, mode="rint"), pl.FP32)
                y_valid = pl.set_validshape(y_2d, valid_rows, D_ACT_TILE)
                output[ts0:ts0 + DN_M_TILE, d0:d0 + D_ACT_TILE] = y_valid
    return output


@pl.jit.inline
def dense_mlp_bf16(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Run BF16 gate/up, clamped SwiGLU and a BF16 local down projection."""
    tokens = pl.tensor.dim(x, 0)
    row_tiles = (tokens + BF16_M_TILE - 1) // BF16_M_TILE
    gate_up = pl.create_tensor([tokens, 2 * LOCAL_DENSE_INTER], dtype=pl.BF16)
    for task in pl.spmd(row_tiles * (2 * LOCAL_DENSE_INTER // BF16_N_TILE), name_hint="dense_bf16_gate_up"):
        t0 = task // (2 * LOCAL_DENSE_INTER // BF16_N_TILE) * BF16_M_TILE
        n0 = task % (2 * LOCAL_DENSE_INTER // BF16_N_TILE) * BF16_N_TILE
        rows = pl.min(BF16_M_TILE, tokens - t0)
        columns = pl.cast(BF16_N_TILE, pl.INDEX)
        projected = bf16_linear(x, w_gate_up, t0, n0, rows, columns, BF16_M_TILE, BF16_N_TILE, D, True)
        projected_bf16 = pl.cast(projected, pl.BF16, mode="rint")
        gate_up[t0:t0 + BF16_M_TILE, n0:n0 + BF16_N_TILE] = pl.set_validshape(projected_bf16, rows, BF16_N_TILE)

    activation_tiles = (tokens + BF16_ACT_M_TILE - 1) // BF16_ACT_M_TILE
    activated = pl.create_tensor([tokens, LOCAL_DENSE_INTER], dtype=pl.BF16)
    for task in pl.spmd(activation_tiles * (LOCAL_DENSE_INTER // BF16_N_TILE), name_hint="dense_bf16_swiglu"):
        t0 = task // (LOCAL_DENSE_INTER // BF16_N_TILE) * BF16_ACT_M_TILE
        n0 = task % (LOCAL_DENSE_INTER // BF16_N_TILE) * BF16_N_TILE
        u0 = LOCAL_DENSE_INTER + n0
        rows = pl.min(BF16_ACT_M_TILE, tokens - t0)
        gate_tile = pl.slice(gate_up, [BF16_ACT_M_TILE, BF16_N_TILE], [t0, n0], valid_shape=[rows, BF16_N_TILE])
        up_tile = pl.slice(gate_up, [BF16_ACT_M_TILE, BF16_N_TILE], [t0, u0], valid_shape=[rows, BF16_N_TILE])
        gate = pl.cast(gate_tile, pl.FP32)
        up = pl.cast(up_tile, pl.FP32)
        if SWIGLU_LIMIT > 0.0:
            gate = pl.minimum(gate, SWIGLU_LIMIT)
            up = pl.maximum(pl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        gate_buffer = pl.create_tensor([BF16_ACT_M_TILE, BF16_N_TILE], dtype=pl.FP32)
        gate_buffer[:, :] = gate
        exp_argument = pl.create_tensor([BF16_ACT_M_TILE, BF16_N_TILE], dtype=pl.FP32)
        exp_argument[:, :] = pl.neg(gate)
        exponential = accurate_exp(exp_argument)
        denominator = pl.create_tensor([BF16_ACT_M_TILE, BF16_N_TILE], dtype=pl.FP32)
        denominator[:, :] = pl.add(exponential, 1.0)
        silu_value = accurate_divide(gate_buffer, denominator)
        silu = pl.cast(silu_value, pl.BF16, mode="rint")
        product = pl.mul(pl.cast(silu, pl.FP32), up)
        product_bf16 = pl.cast(product, pl.BF16, mode="rint")
        activated[t0:t0 + BF16_ACT_M_TILE, n0:n0 + BF16_N_TILE] = pl.set_validshape(product_bf16, rows, BF16_N_TILE)

    for task in pl.spmd(row_tiles * (D // BF16_N_TILE), name_hint="dense_bf16_down"):
        t0 = task // (D // BF16_N_TILE) * BF16_M_TILE
        d0 = task % (D // BF16_N_TILE) * BF16_N_TILE
        rows = pl.min(BF16_M_TILE, tokens - t0)
        columns = pl.cast(BF16_N_TILE, pl.INDEX)
        projected = bf16_linear(
            activated, w_down, t0, d0, rows, columns, BF16_M_TILE, BF16_N_TILE, LOCAL_DENSE_INTER, True,
        )
        projected_bf16 = pl.cast(projected, pl.BF16, mode="rint")
        output[t0:t0 + BF16_M_TILE, d0:d0 + BF16_N_TILE] = pl.set_validshape(projected_bf16, rows, BF16_N_TILE)
    return output


@pl.jit
def dense_mlp_bf16_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    dense_mlp_bf16(x, w_gate_up, w_down, output)
    return output


@pl.jit
def dense_mlp_test(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    x_int8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)

    dense_mlp(x_int8, x_scale, w_gate_up, w_gate_up_scale, w_down, w_down_scale, output)
    return output


def golden_dense_mlp_fn(tensors):
    """Torch reference for the dense MLP over the fused gate/up layout."""

    x_int8 = tensors["x_int8"].to(torch.int32)
    x_scale = tensors["x_scale"].float()
    gate_up_i8 = tensors["w_gate_up"].to(torch.int32)
    gate_up_scale = tensors["w_gate_up_scale"].float()
    w_down_i8 = tensors["w_down"].to(torch.int32)
    w_down_scale = tensors["w_down_scale"].float()

    local = LOCAL_DENSE_INTER
    gate_i = x_int8 @ gate_up_i8[:local].T
    up_i = x_int8 @ gate_up_i8[local:].T
    gate_v = gate_i.float() * x_scale * gate_up_scale[:local].view(1, -1)
    up_v = up_i.float() * x_scale * gate_up_scale[local:].view(1, -1)
    gate_v = gate_v.bfloat16().float()
    up_v = up_v.bfloat16().float()
    if SWIGLU_LIMIT > 0:
        gate_v = gate_v.clamp(max=SWIGLU_LIMIT)
        up_v = up_v.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    silu = (gate_v.double() / (1.0 + torch.exp(-gate_v.double()).float()).double()).float().bfloat16().float()
    h = (silu * up_v).bfloat16()
    h_i8, h_sd = quantize_per_token_int8(h)
    out_i = h_i8.to(torch.int32) @ w_down_i8.T
    out = out_i.float() * h_sd * w_down_scale.view(1, -1)

    tensors["output"][:] = out.bfloat16().float()


def gen_dense_weight(shape, dequant_std, chan_cv, *, generator=None):
    """Synthesize an INT8 weight + per-channel FP32 scale with a realistic
    per-channel magnitude spread (see the DeepSeek-V4 sibling for the grid)."""
    channel_gain = torch.exp(chan_cv * torch.randn(*shape[:-1], 1, generator=generator))
    W = torch.randn(*shape, generator=generator) * channel_gain
    amax = W.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(W / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def golden_dense_mlp_bf16(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Apply the BF16 linear and activation boundaries of the dense FFN."""
    projected = golden_bf16_linear(x, w_gate_up).bfloat16()
    gate, up = projected.float().chunk(2, dim=-1)
    if SWIGLU_LIMIT > 0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    silu = (gate.double() / (1.0 + torch.exp(-gate.double()).float()).double()).float().bfloat16().float()
    activated = (silu * up).bfloat16()
    return golden_bf16_linear(activated, w_down).bfloat16()


def golden_dense_mlp_bf16_fn(tensors):
    tensors["output"].copy_(golden_dense_mlp_bf16(tensors["x"], tensors["w_gate_up"], tensors["w_down"]))


def build_bf16_tensor_specs(num_tokens=17, *, weights=None, tp_rank=0):
    """Build deterministic BF16 inputs and optionally use resolved TP weight slices."""
    from golden import TensorSpec

    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    generator = torch.Generator().manual_seed(533)

    def init_x():
        value = (2 * torch.randn(num_tokens, D, generator=generator)).bfloat16()
        if num_tokens > 1:
            value[0] = 0
            value[1] *= 4
        return value

    def init_gate_up():
        return (torch.randn(2 * LOCAL_DENSE_INTER, D, generator=generator) / D**0.5).bfloat16()

    def init_down():
        return (torch.randn(D, LOCAL_DENSE_INTER, generator=generator) / LOCAL_DENSE_INTER**0.5).bfloat16()

    gate_up_init, down_init = init_gate_up, init_down
    if weights is not None:
        if not 0 <= tp_rank < TP_SIZE:
            raise ValueError(f"tp_rank must be in [0, {TP_SIZE})")
        resolved = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        gate_up_init, down_init = resolved["w_gate_up"][tp_rank], resolved["w_down"][tp_rank]
        if gate_up_init.dtype != torch.bfloat16 or down_init.dtype != torch.bfloat16:
            raise ValueError("resolved FFN weights must be BF16")
        if tuple(gate_up_init.shape) != (2 * LOCAL_DENSE_INTER, D):
            raise ValueError("resolved gate/up shape does not match the selected TP size")
        if tuple(down_init.shape) != (D, LOCAL_DENSE_INTER):
            raise ValueError("resolved down shape does not match the selected TP size")

    return [
        TensorSpec("x", [num_tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("w_gate_up", [2 * LOCAL_DENSE_INTER, D], torch.bfloat16, init_value=gate_up_init),
        TensorSpec("w_down", [D, LOCAL_DENSE_INTER], torch.bfloat16, init_value=down_init),
        TensorSpec("output", [num_tokens, D], torch.bfloat16),
    ]


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(533)
    x_bf16 = torch.randn(num_tokens, D, generator=generator).bfloat16()
    if num_tokens > 1:
        x_bf16[0] = 0
        x_bf16[-1] *= 8

    def init_x_int8():

        x_i8, x_sd = quantize_per_token_int8(x_bf16)
        return x_i8

    def init_x_scale():

        _, x_sd = quantize_per_token_int8(x_bf16)
        return x_sd

    DENSE_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    (dw_gu_i8, dw_gu_s) = gen_dense_weight(
        (2 * LOCAL_DENSE_INTER, D), DENSE_DEQUANT_STD["w1"], chan_cv=0.5, generator=generator,
    )
    (dw2_i8, dw2_s) = gen_dense_weight(
        (D, LOCAL_DENSE_INTER), DENSE_DEQUANT_STD["w2"], chan_cv=0.33, generator=generator,
    )

    return [
        TensorSpec("x_int8", [num_tokens, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [num_tokens, 1], torch.float32, init_value=init_x_scale),
        TensorSpec("w_gate_up", [2 * LOCAL_DENSE_INTER, D], torch.int8, init_value=lambda: dw_gu_i8),
        TensorSpec("w_gate_up_scale", [2 * LOCAL_DENSE_INTER], torch.float32, init_value=lambda: dw_gu_s),
        TensorSpec("w_down", [D, LOCAL_DENSE_INTER], torch.int8, init_value=lambda: dw2_i8),
        TensorSpec("w_down_scale", [D], torch.float32, init_value=lambda: dw2_s),
        TensorSpec("output", [num_tokens, D], torch.float32),
    ]


@dataclass
class DenseTensors:
    """One FFN as separate FP32 gate/up/down (dense MLP or shared expert)."""

    gate: torch.Tensor  # [inter, D]
    up: torch.Tensor  # [inter, D]
    down: torch.Tensor  # [D, inter]


def swiglu_limited(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU: clamp(gate, max=limit), clamp(up, ±limit), silu(gate)*up."""
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def dense_mlp_forward(x: torch.Tensor, weights: DenseTensors, model: Glm53FlashConfig) -> torch.Tensor:
    """Dense MLP (port of ``Glm5NextTextMLP.forward``) with the SwiGLU clamp."""
    gate = x @ weights.gate.t()
    up = x @ weights.up.t()
    return swiglu_limited(gate, up, model.swiglu_limit) @ weights.down.t()


def synthetic_dense(model: Glm53FlashConfig, generator: torch.Generator, intermediate: int) -> DenseTensors:
    return DenseTensors(
        gate=_randn((intermediate, model.hidden_size), generator),
        up=_randn((intermediate, model.hidden_size), generator),
        down=_randn((model.hidden_size, intermediate), generator),
    )


def _test_dense_mlp(model: Glm53FlashConfig) -> None:
    torch.manual_seed(29)
    weights = synthetic_dense(model, torch.Generator().manual_seed(2), model.intermediate_size)
    x = torch.randn(2, 9, model.hidden_size) * 0.5
    out = dense_mlp_forward(x, weights, model)
    gate = (x @ weights.gate.t()).clamp(max=model.swiglu_limit)
    up = (x @ weights.up.t()).clamp(-model.swiglu_limit, model.swiglu_limit)
    _assert_close(out, (F.silu(gate) * up) @ weights.down.t(), "dense MLP with clamps")
    over = torch.full((1, 1, model.intermediate_size), 100.0)
    clamped = swiglu_limited(over, over, model.swiglu_limit)
    assert float(clamped.abs().max()) < 100.0
    print("  dense MLP: manual equivalence, SwiGLU clamp")


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int)
    parser.add_argument("--dtype", choices=["int8", "bf16"], default="int8")
    parser.add_argument("--case", choices=["all", "single", "ragged", "aligned"], default="all")
    parser.add_argument("--weights", type=Path, help="resolved BF16 TP weights with w_gate_up and w_down entries")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.weights is not None and args.dtype != "bf16":
        parser.error("--weights requires --dtype bf16")
    if args.num_tokens is not None and args.num_tokens < 1:
        parser.error("--num-tokens must be positive")
    if args.dtype == "bf16":
        cases = {"single": 1, "ragged": 17, "aligned": 16}
        selected = (args.case,) if args.case != "all" else ("single", "ragged")
        counts = [args.num_tokens] if args.num_tokens is not None else [cases[name] for name in selected]
        for count in counts:
            result = run(
                fn=dense_mlp_bf16_test,
                specs=build_bf16_tensor_specs(count, weights=args.weights, tp_rank=args.tp_rank),
                golden_fn=golden_dense_mlp_bf16_fn,
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device, dump_passes=args.dump_passes),
                compile_only=args.compile_only,
                compare_fn={"output": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0)},
            )
            print(result)
            if not result.passed:
                raise SystemExit(result.error or 1)
        raise SystemExit(0)

    cases = {"single": 1, "ragged": 17, "aligned": 16}
    selected = (args.case,) if args.case != "all" else tuple(cases)
    counts = [args.num_tokens] if args.num_tokens is not None else [cases[name] for name in selected]
    for count in counts:
        result = run(
            fn=dense_mlp_test,
            specs=build_tensor_specs(count),
            golden_fn=golden_dense_mlp_fn,
            config=dict(dump_passes=args.dump_passes, backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane),
            rtol=0.001,
            atol=0.001,
            compile_only=args.compile_only,
            compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01)},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
