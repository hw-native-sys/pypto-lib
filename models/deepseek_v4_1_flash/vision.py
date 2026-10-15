# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash BF16 ViT and visual aligner for one image on A5."""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Dynamic shape variables.
ROWS = pl.dynamic("V41_VISION_ROWS")
INPUT = pl.dynamic("V41_VISION_INPUT")
OUTPUT = pl.dynamic("V41_VISION_OUTPUT")

# tiling
M_TILE = 16
N_TILE = 64
K_TILE = 64
ATTENTION_TILE = 64


@dataclass(frozen=True)
class VisionConfig:
    """Released vision dimensions; reduced configurations support validation."""

    dim: int = 1024
    layers: int = 32
    heads: int = 16
    intermediate: int = 2816
    patch_size: int = 14
    language_dim: int = 5120
    downsample: int = 3
    rope_theta: float = 10000.0

    def __post_init__(self):
        values = (self.dim, self.layers, self.heads, self.intermediate,
                  self.patch_size, self.language_dim, self.downsample)
        if any(value < 1 for value in values):
            raise ValueError("Vision dimensions must be positive")
        if self.dim % 128:
            raise ValueError("Vision channel width must be divisible by 128")
        if self.dim % self.heads or self.dim // self.heads % 32:
            raise ValueError("Vision head width must be divisible by 32")
        if self.dim // self.heads > 128 or self.downsample > 4:
            raise ValueError("Supported head width <= 128 and downsample <= 4")
        if self.rope_theta <= 0:
            raise ValueError("RoPE theta must be positive")


def vision_rope(n_h, n_w, cfg=VisionConfig()):
    """Return FP32 height frequencies followed by width frequencies."""
    if n_h < 1 or n_w < 1:
        raise ValueError("Image grid dimensions must be positive")
    half = cfg.dim // cfg.heads // 2
    frequencies = torch.arange(0, half, 2, dtype=torch.float32, device="cpu")
    inverse = 1.0 / (cfg.rope_theta ** (frequencies / half))
    h = torch.arange(n_h, device="cpu").repeat_interleave(n_w)
    w = torch.arange(n_w, device="cpu").repeat(n_h)
    angles = torch.stack((h, w), dim=1).float().unsqueeze(-1) * inverse
    angles = angles.flatten(1)
    return angles.cos().contiguous(), angles.sin().contiguous()


@pl.jit.inline(auto_scope=False)
def linear_inline(
    x: pl.Tensor[[ROWS, INPUT], pl.BF16],
    weight: pl.Tensor[[INPUT, OUTPUT], pl.BF16],
    bias: pl.Tensor,
    output: pl.Tensor[[ROWS, OUTPUT], pl.BF16],
    has_bias: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    columns = pl.tensor.dim(weight, 1)
    column_tiles = (columns + N_TILE - 1) // N_TILE
    for worker in pl.spmd(32, name_hint="vision_linear"):
        for block in pl.range(worker, ((rows + M_TILE - 1) // M_TILE) * column_tiles, 32):
            r0 = block // column_tiles * M_TILE
            c0 = block % column_tiles * N_TILE
            nr = pl.min(M_TILE, rows - r0)
            nc = pl.min(N_TILE, columns - c0)
            acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
            for k0 in pl.range(0, width, K_TILE):
                nk = pl.min(K_TILE, width - k0)
                a = pl.slice(x, [M_TILE, K_TILE], [r0, k0], valid_shape=[nr, nk])
                a_fillpad = pl.fillpad(a, pad_value=pl.PadValue.zero)
                a = pl.set_validshape(a_fillpad, M_TILE, K_TILE)
                b = pl.slice(weight, [K_TILE, N_TILE], [k0, c0], valid_shape=[nk, nc])
                b_fillpad = pl.fillpad(b, pad_value=pl.PadValue.zero)
                b = pl.set_validshape(b_fillpad, K_TILE, N_TILE)
                acc = pl.matmul_acc(acc, a, b, init_cond=(k0 == 0))
            if has_bias == 1:
                bias_slice = pl.slice(bias, [N_TILE], [c0], valid_shape=[nc])
                bias_row = pl.reshape(bias_slice, [1, N_TILE])
                acc_cast = pl.cast(bias_row, pl.FP32)
                acc = pl.col_expand_add(acc, acc_cast)
            result = pl.cast(acc, pl.BF16, mode="rint")
            output[r0:r0 + M_TILE, c0:c0 + N_TILE] = pl.set_validshape(result, nr, nc)
    return output


def _make_norm(width):
    @pl.jit.inline(auto_scope=False)
    def norm(
        x: pl.Tensor[[ROWS, width], pl.BF16],
        weight: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[ROWS, width], pl.BF16],
    ):
        rows = pl.tensor.dim(x, 0)
        for worker in pl.spmd(32, name_hint="vision_rmsnorm"):
            for block in pl.range(worker, (rows + 7) // 8, 32):
                r0 = block * 8
                nr = pl.min(8, rows - r0)
                total = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                for c0 in pl.range(0, width, 128):
                    nc = pl.min(128, width - c0)
                    value_slice_1 = pl.slice(x, [8, 128], [r0, c0], valid_shape=[nr, nc])
                    value = pl.cast(value_slice_1, pl.FP32)
                    value_fillpad = pl.fillpad(value, pad_value=pl.PadValue.zero)
                    value = pl.set_validshape(value_fillpad, 8, 128)
                    square = pl.mul(value, value)
                    square_sum_row_sum = pl.row_sum(square)
                    square_sum = pl.reshape(square_sum_row_sum, [1, 8])
                    total = pl.add(total, square_sum)
                mean_square_mul = pl.mul(total, 1.0 / width)
                mean_square = pl.add(mean_square_mul, 1e-06)
                inverse_row = pl.rsqrt(mean_square, high_precision=True)
                inverse = pl.reshape(inverse_row, [8, 1])
                for c0 in pl.range(0, width, 128):
                    nc = pl.min(128, width - c0)
                    value_slice_2 = pl.slice(x, [8, 128], [r0, c0], valid_shape=[nr, nc])
                    value = pl.cast(value_slice_2, pl.FP32)
                    gamma_slice = pl.slice(weight, [128], [c0], valid_shape=[nc])
                    gamma = pl.reshape(gamma_slice, [1, 128])
                    normalized = pl.row_expand_mul(value, inverse)
                    result_col_expand_mul = pl.col_expand_mul(normalized, gamma)
                    result = pl.cast(result_col_expand_mul, pl.BF16, mode='rint')
                    output[r0:r0 + 8, c0:c0 + 128] = pl.set_validshape(result, nr, nc)
        return output
    return norm


@pl.jit.inline(auto_scope=False)
def add_inline(
    x: pl.Tensor[[ROWS, INPUT], pl.BF16],
    residual: pl.Tensor[[ROWS, INPUT], pl.BF16],
    output: pl.Tensor[[ROWS, INPUT], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for worker in pl.spmd(32, name_hint="vision_residual"):
        for block in pl.range(worker, (rows + 7) // 8, 32):
            r0 = block * 8
            nr = pl.min(8, rows - r0)
            for c0 in pl.range(0, width, 128):
                nc = pl.min(128, width - c0)
                a_slice = pl.slice(x, [8, 128], [r0, c0], valid_shape=[nr, nc])
                a = pl.cast(a_slice, pl.FP32)
                b_slice = pl.slice(residual, [8, 128], [r0, c0], valid_shape=[nr, nc])
                b = pl.cast(b_slice, pl.FP32)
                result_add = pl.add(a, b)
                result = pl.cast(result_add, pl.BF16, mode='rint')
                output[r0:r0 + 8, c0:c0 + 128] = pl.set_validshape(result, nr, nc)
    return output


def _make_attention(cfg):
    width, heads = cfg.dim, cfg.heads
    channels = width // heads
    half = channels // 2

    @pl.jit.inline(auto_scope=False)
    def attention(
        qkv: pl.Tensor[[ROWS, 3 * width], pl.BF16],
        cos: pl.Tensor[[ROWS, half], pl.FP32],
        sin: pl.Tensor[[ROWS, half], pl.FP32],
        output: pl.Tensor[[ROWS, width], pl.BF16],
    ):
        rows = pl.tensor.dim(qkv, 0)
        q = pl.create_tensor([heads, rows, channels], dtype=pl.BF16)
        k = pl.create_tensor([heads, rows, channels], dtype=pl.BF16)
        v = pl.create_tensor([heads, rows, channels], dtype=pl.BF16)
        for worker in pl.spmd(32, name_hint="vision_split_half_rope"):
            for block in pl.range(worker, heads * rows, 32):
                r = block // heads
                h = block % heads
                c = cos[r:r + 1, :]
                s = sin[r:r + 1, :]
                qa = pl.cast(qkv[r:r + 1, h * channels:h * channels + half], pl.FP32)
                qb = pl.cast(qkv[r:r + 1, h * channels + half:(h + 1) * channels], pl.FP32)
                ka = pl.cast(qkv[r:r + 1, width + h * channels:width + h * channels + half], pl.FP32)
                kb = pl.cast(qkv[r:r + 1, width + h * channels + half:width + (h + 1) * channels], pl.FP32)
                q_left_mul = pl.mul(qa, c)
                q_left_mul_1 = pl.mul(qb, s)
                q_left_sub = pl.sub(q_left_mul, q_left_mul_1)
                q_left = pl.cast(q_left_sub, pl.BF16, mode='rint')
                q_right_mul = pl.mul(qb, c)
                q_right_mul_1 = pl.mul(qa, s)
                q_right_add = pl.add(q_right_mul, q_right_mul_1)
                q_right = pl.cast(q_right_add, pl.BF16, mode='rint')
                k_left_mul = pl.mul(ka, c)
                k_left_mul_1 = pl.mul(kb, s)
                k_left_sub = pl.sub(k_left_mul, k_left_mul_1)
                k_left = pl.cast(k_left_sub, pl.BF16, mode='rint')
                k_right_mul = pl.mul(kb, c)
                k_right_mul_1 = pl.mul(ka, s)
                k_right_add = pl.add(k_right_mul, k_right_mul_1)
                k_right = pl.cast(k_right_add, pl.BF16, mode='rint')
                q[h:h + 1, r:r + 1, :half] = pl.reshape(q_left, [1, 1, half])
                q[h:h + 1, r:r + 1, half:] = pl.reshape(q_right, [1, 1, half])
                k[h:h + 1, r:r + 1, :half] = pl.reshape(k_left, [1, 1, half])
                k[h:h + 1, r:r + 1, half:] = pl.reshape(k_right, [1, 1, half])
                values_row = qkv[r:r + 1, 2 * width + h * channels:2 * width + (h + 1) * channels]
                v[h:h + 1, r:r + 1, :] = pl.reshape(values_row, [1, 1, channels])
        scratch_width = (rows + ATTENTION_TILE - 1) // ATTENTION_TILE * ATTENTION_TILE
        scratch = pl.create_tensor([32 * M_TILE, scratch_width], dtype=pl.FP32)
        for worker in pl.spmd(32, name_hint="vision_attention"):
            for block in pl.range(worker, heads * ((rows + M_TILE - 1) // M_TILE), 32):
                h = block // ((rows + M_TILE - 1) // M_TILE)
                r0 = block % ((rows + M_TILE - 1) // M_TILE) * M_TILE
                nr = pl.min(M_TILE, rows - r0)
                scratch_row = worker * M_TILE
                query_slice = pl.slice(q, [1, M_TILE, channels], [h, r0, 0], valid_shape=[1, nr, channels])
                query = pl.reshape(query_slice, [M_TILE, channels])
                query_fillpad = pl.fillpad(query, pad_value=pl.PadValue.zero)
                query = pl.set_validshape(query_fillpad, M_TILE, channels)
                query_cast = pl.cast(query, pl.FP32)
                query = pl.mul(query_cast, channels ** (-0.25))
                maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
                for t0 in pl.range(0, rows, ATTENTION_TILE):
                    nt = pl.min(ATTENTION_TILE, rows - t0)
                    key_slice = pl.slice(k, [1, ATTENTION_TILE, channels], [h, t0, 0], valid_shape=[1, nt, channels])
                    keys = pl.reshape(key_slice, [ATTENTION_TILE, channels])
                    keys_fillpad = pl.fillpad(keys, pad_value=pl.PadValue.zero)
                    keys = pl.set_validshape(keys_fillpad, ATTENTION_TILE, channels)
                    keys_cast = pl.cast(keys, pl.FP32)
                    keys = pl.mul(keys_cast, channels ** (-0.25))
                    scores = pl.matmul(query, keys, b_trans=True)
                    scores = pl.set_validshape(scores, M_TILE, nt)
                    scores_fillpad = pl.fillpad(scores, pad_value=pl.PadValue.min)
                    scores = pl.set_validshape(scores_fillpad, M_TILE, ATTENTION_TILE)
                    scratch[scratch_row:scratch_row + M_TILE, t0:t0 + ATTENTION_TILE] = scores
                    score_max_row_max = pl.row_max(scores)
                    score_max = pl.reshape(score_max_row_max, [1, M_TILE])
                    maximum = pl.maximum(maximum, score_max)
                denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
                maximum_column = pl.reshape(maximum, [M_TILE, 1])
                for t0 in pl.range(0, rows, ATTENTION_TILE):
                    nt = pl.min(ATTENTION_TILE, rows - t0)
                    scores = scratch[scratch_row:scratch_row + M_TILE, t0:t0 + ATTENTION_TILE]
                    probability_row_expand_sub = pl.row_expand_sub(scores, maximum_column)
                    probability = pl.exp(probability_row_expand_sub)
                    mass_row_sum = pl.row_sum(probability)
                    mass = pl.reshape(mass_row_sum, [1, M_TILE])
                    denominator = pl.add(denominator, mass)
                    scratch[scratch_row:scratch_row + M_TILE, t0:t0 + ATTENTION_TILE] = probability
                denominator = pl.maximum(denominator, 1e-20)
                inverse = pl.recip(denominator, high_precision=True)
                inverse_column = pl.reshape(inverse, [M_TILE, 1])
                numerator = pl.create_tensor([M_TILE, channels], dtype=pl.FP32)
                for t0 in pl.range(0, rows, ATTENTION_TILE):
                    nt = pl.min(ATTENTION_TILE, rows - t0)
                    probability = scratch[scratch_row:scratch_row + M_TILE, t0:t0 + ATTENTION_TILE]
                    probability = pl.row_expand_mul(probability, inverse_column)
                    value_slice = pl.slice(v, [1, ATTENTION_TILE, channels], [h, t0, 0], valid_shape=[1, nt, channels])
                    values = pl.reshape(value_slice, [ATTENTION_TILE, channels])
                    values_fillpad = pl.fillpad(values, pad_value=pl.PadValue.zero)
                    values = pl.set_validshape(values_fillpad, ATTENTION_TILE, channels)
                    values = pl.cast(values, pl.FP32)
                    numerator = pl.matmul_acc(numerator, probability, values, init_cond=(t0 == 0))
                result_bf16 = pl.cast(numerator, pl.BF16, mode="rint")
                output[r0:r0 + M_TILE, h * channels:(h + 1) * channels] = pl.set_validshape(result_bf16, nr, channels)
        return output
    return attention


def _make_swiglu(width):
    @pl.jit.inline(auto_scope=False)
    def swiglu(
        x: pl.Tensor[[ROWS, 2 * width], pl.BF16],
        output: pl.Tensor[[ROWS, width], pl.BF16],
    ):
        rows = pl.tensor.dim(x, 0)
        for worker in pl.spmd(32, name_hint="vision_swiglu"):
            for block in pl.range(worker, (rows + 7) // 8, 32):
                r0 = block * 8
                nr = pl.min(8, rows - r0)
                for c0 in pl.range(0, width, 128):
                    nc = pl.min(128, width - c0)
                    gate_slice = pl.slice(x, [8, 128], [r0, c0], valid_shape=[nr, nc])
                    gate = pl.cast(gate_slice, pl.FP32)
                    up_slice = pl.slice(x, [8, 128], [r0, width + c0], valid_shape=[nr, nc])
                    up = pl.cast(up_slice, pl.FP32)
                    silu_neg = pl.neg(gate)
                    silu_exp = pl.exp(silu_neg)
                    silu_add = pl.add(silu_exp, 1.0)
                    silu_recip = pl.recip(silu_add, high_precision=True)
                    silu = pl.mul(gate, silu_recip)
                    silu_cast = pl.cast(silu, pl.BF16, mode='rint')
                    silu = pl.cast(silu_cast, pl.FP32)
                    result_mul = pl.mul(silu, up)
                    result = pl.cast(result_mul, pl.BF16, mode='rint')
                    output[r0:r0 + 8, c0:c0 + 128] = pl.set_validshape(result, nr, nc)
        return output
    return swiglu


def _make_vit(cfg):
    d, l, i = cfg.dim, cfg.layers, cfg.intermediate
    patch = 3 * cfg.patch_size ** 2
    norm = _make_norm(d)
    attention = _make_attention(cfg)
    swiglu = _make_swiglu(i)

    @pl.jit.inline(auto_scope=False)
    def vit(
        patches: pl.Tensor[[ROWS, patch], pl.BF16],
        patch_w: pl.Tensor[[patch, d], pl.BF16],
        patch_b: pl.Tensor[[d], pl.BF16],
        norm1_w: pl.Tensor[[l, d], pl.FP32],
        qkv_w: pl.Tensor[[l, d, 3 * d], pl.BF16],
        qkv_b: pl.Tensor[[l, 3 * d], pl.BF16],
        out_w: pl.Tensor[[l, d, d], pl.BF16],
        out_b: pl.Tensor[[l, d], pl.BF16],
        norm2_w: pl.Tensor[[l, d], pl.FP32],
        mlp1_w: pl.Tensor[[l, d, 2 * i], pl.BF16],
        mlp2_w: pl.Tensor[[l, i, d], pl.BF16],
        norm_w: pl.Tensor[[d], pl.FP32],
        cos: pl.Tensor[[ROWS, d // cfg.heads // 2], pl.FP32],
        sin: pl.Tensor[[ROWS, d // cfg.heads // 2], pl.FP32],
        features: pl.Tensor[[ROWS, d], pl.BF16],
    ):
        rows = pl.tensor.dim(patches, 0)
        x = pl.create_tensor([rows, d], dtype=pl.BF16)
        linear_inline(patches, patch_w, patch_b, x, 1)
        next_x = pl.create_tensor([rows, d], dtype=pl.BF16)
        for layer in pl.range(l):
            normalized = pl.create_tensor([rows, d], dtype=pl.BF16)
            norm1_slice = norm1_w[layer:layer + 1, :]
            norm1 = pl.reshape(norm1_slice, [d])
            norm(x, norm1, normalized)
            qkv = pl.create_tensor([rows, 3 * d], dtype=pl.BF16)
            wqkv_slice = qkv_w[layer:layer + 1, :, :]
            wqkv = pl.reshape(wqkv_slice, [d, 3 * d])
            bqkv_slice = qkv_b[layer:layer + 1, :]
            bqkv = pl.reshape(bqkv_slice, [3 * d])
            linear_inline(normalized, wqkv, bqkv, qkv, 1)
            attended = pl.create_tensor([rows, d], dtype=pl.BF16)
            attention(qkv, cos, sin, attended)
            projected = pl.create_tensor([rows, d], dtype=pl.BF16)
            wo_slice = out_w[layer:layer + 1, :, :]
            wo = pl.reshape(wo_slice, [d, d])
            bo_slice = out_b[layer:layer + 1, :]
            bo = pl.reshape(bo_slice, [d])
            linear_inline(attended, wo, bo, projected, 1)
            residual = pl.create_tensor([rows, d], dtype=pl.BF16)
            add_inline(projected, x, residual)
            mlp_input = pl.create_tensor([rows, d], dtype=pl.BF16)
            norm2_slice = norm2_w[layer:layer + 1, :]
            norm2 = pl.reshape(norm2_slice, [d])
            norm(residual, norm2, mlp_input)
            gate_up = pl.create_tensor([rows, 2 * i], dtype=pl.BF16)
            w1_slice = mlp1_w[layer:layer + 1, :, :]
            w1 = pl.reshape(w1_slice, [d, 2 * i])
            linear_inline(mlp_input, w1, patch_b, gate_up, 0)
            activated = pl.create_tensor([rows, i], dtype=pl.BF16)
            swiglu(gate_up, activated)
            mlp_output = pl.create_tensor([rows, d], dtype=pl.BF16)
            w2_slice = mlp2_w[layer:layer + 1, :, :]
            w2 = pl.reshape(w2_slice, [i, d])
            linear_inline(activated, w2, patch_b, mlp_output, 0)
            add_inline(mlp_output, residual, next_x)
            x = next_x
        norm(x, norm_w, features)
        return features
    return vit


def _make_aligner(cfg):
    d, r, ld = cfg.dim, cfg.downsample, cfg.language_dim
    aligned_rows = pl.dynamic("V41_VISION_ALIGNED_ROWS")

    @pl.jit.inline(auto_scope=False)
    def aligner(
        features: pl.Tensor[[ROWS, d], pl.BF16],
        w1: pl.Tensor[[d * r * r, ld], pl.BF16],
        b1: pl.Tensor[[ld], pl.BF16],
        w2: pl.Tensor[[ld, ld], pl.BF16],
        b2: pl.Tensor[[ld], pl.BF16],
        embeddings: pl.Tensor[[aligned_rows, ld], pl.BF16],
        n_h: pl.Scalar[pl.INT32],
        n_w: pl.Scalar[pl.INT32],
    ):
        out_w = (n_w + r - 1) // r
        total = pl.tensor.dim(embeddings, 0)
        unshuffled = pl.create_tensor([total * d, r * r], dtype=pl.BF16)
        for worker in pl.spmd(32, name_hint="vision_unshuffle"):
            for block in pl.range(worker, total * ((d + 127) // 128), 32):
                row = block // ((d + 127) // 128)
                c0 = block % ((d + 127) // 128) * 128
                nc = 128
                y0 = row // out_w * r
                x0 = row % out_w * r
                patches = pl.tile.full([16, 128], dtype=pl.BF16, value=0.0)
                for dy in pl.unroll(r):
                    for dx in pl.unroll(r):
                        if y0 + dy < n_h and x0 + dx < n_w:
                            value = pl.load(features, [(y0 + dy) * n_w + x0 + dx, c0],
                                            [1, 128], valid_shape=[1, nc])
                            value = pl.set_validshape(value, 1, 128)
                            patches[dy * r + dx:dy * r + dx + 1, :] = value
                packed = pl.transpose(patches, axis1=0, axis2=1)
                unshuffled_set_validshape = pl.set_validshape(packed, nc, r * r)
                unshuffled = pl.store(unshuffled_set_validshape, [row * d + c0, 0], unshuffled)
        hidden = pl.create_tensor([total, ld], dtype=pl.BF16)
        unshuffled_rows = pl.reshape(unshuffled, [total, d * r * r])
        linear_inline(unshuffled_rows, w1, b1, hidden, 1)
        activated = pl.create_tensor([total, ld], dtype=pl.BF16)
        for worker in pl.spmd(32, name_hint="vision_gelu"):
            for block in pl.range(worker, (total + 7) // 8, 32):
                t0 = block * 8
                nt = pl.min(8, total - t0)
                for c0 in pl.range(0, ld, 128):
                    nc = pl.min(128, ld - c0)
                    x_load = pl.load(hidden, [t0, c0], [8, 128], valid_shape=[nt, nc])
                    x = pl.cast(x_load, pl.FP32)
                    z = pl.mul(x, 0.7071067811865476)
                    t_abs = pl.abs(z)
                    t_mul = pl.mul(t_abs, 0.3275911)
                    t_add = pl.add(t_mul, 1.0)
                    t = pl.recip(t_add, high_precision=True)
                    p_mul = pl.mul(t, 1.061405429)
                    p = pl.add(p_mul, -1.453152027)
                    p_mul_1 = pl.mul(t, p)
                    p = pl.add(p_mul_1, 1.421413741)
                    p_mul_2 = pl.mul(t, p)
                    p = pl.add(p_mul_2, -0.284496736)
                    p_mul_3 = pl.mul(t, p)
                    p = pl.add(p_mul_3, 0.254829592)
                    erf_abs_mul = pl.mul(t, p)
                    erf_abs_mul_1 = pl.mul(z, z)
                    erf_abs_neg = pl.neg(erf_abs_mul_1)
                    erf_abs_exp = pl.exp(erf_abs_neg)
                    erf_abs_mul_2 = pl.mul(erf_abs_mul, erf_abs_exp)
                    erf_abs_neg_1 = pl.neg(erf_abs_mul_2)
                    erf_abs = pl.add(erf_abs_neg_1, 1.0)
                    tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
                    erf_cmps = pl.cmps(z, 0.0, cmp_type=5)
                    erf_neg = pl.neg(erf_abs)
                    erf = pl.sel(erf_cmps, erf_abs, erf_neg, tmp)
                    gelu_mul = pl.mul(x, 0.5)
                    gelu_add = pl.add(erf, 1.0)
                    gelu = pl.mul(gelu_mul, gelu_add)
                    activated_cast = pl.cast(gelu, pl.BF16, mode='rint')
                    activated_set_validshape = pl.set_validshape(activated_cast, nt, nc)
                    activated = pl.store(activated_set_validshape, [t0, c0], activated)
        destination = pl.reshape(embeddings, [total, ld])
        linear_inline(activated, w2, b2, destination, 1)
        return embeddings
    return aligner


def make_vision_program(cfg=VisionConfig()):
    """Build a callable full encoder/aligner; weights are caller-owned inputs.

    ``features`` is [n_h*n_w, dim], ``embeddings`` is
    [ceil(n_h/r)*ceil(n_w/r), language_dim]. Process images independently.
    Provide the tables from ``vision_rope`` for that same grid.
    """
    vit = _make_vit(cfg)
    aligner = _make_aligner(cfg)
    d, l, i, ld = cfg.dim, cfg.layers, cfg.intermediate, cfg.language_dim
    patch = 3 * cfg.patch_size ** 2
    aligned = pl.dynamic("V41_VISION_ALIGNED_ROWS")
    half = d // cfg.heads // 2

    @pl.jit
    def vision(
        patches: pl.Tensor[[ROWS, patch], pl.BF16],
        patch_w: pl.Tensor[[patch, d], pl.BF16],
        patch_b: pl.Tensor[[d], pl.BF16],
        norm1_w: pl.Tensor[[l, d], pl.FP32],
        qkv_w: pl.Tensor[[l, d, 3 * d], pl.BF16],
        qkv_b: pl.Tensor[[l, 3 * d], pl.BF16],
        out_w: pl.Tensor[[l, d, d], pl.BF16],
        out_b: pl.Tensor[[l, d], pl.BF16],
        norm2_w: pl.Tensor[[l, d], pl.FP32],
        mlp1_w: pl.Tensor[[l, d, 2 * i], pl.BF16],
        mlp2_w: pl.Tensor[[l, i, d], pl.BF16],
        norm_w: pl.Tensor[[d], pl.FP32],
        cos: pl.Tensor[[ROWS, half], pl.FP32],
        sin: pl.Tensor[[ROWS, half], pl.FP32],
        align1_w: pl.Tensor[[d * cfg.downsample ** 2, ld], pl.BF16],
        align1_b: pl.Tensor[[ld], pl.BF16],
        align2_w: pl.Tensor[[ld, ld], pl.BF16],
        align2_b: pl.Tensor[[ld], pl.BF16],
        features: pl.Out[pl.Tensor[[ROWS, d], pl.BF16]],
        embeddings: pl.Out[pl.Tensor[[aligned, ld], pl.BF16]],
        n_h: pl.Scalar[pl.INT32],
        n_w: pl.Scalar[pl.INT32],
    ):
        patches.bind_dynamic(0, ROWS)
        cos.bind_dynamic(0, ROWS)
        sin.bind_dynamic(0, ROWS)
        features.bind_dynamic(0, ROWS)
        embeddings.bind_dynamic(0, aligned)
        vit(
            patches, patch_w, patch_b,
            norm1_w, qkv_w, qkv_b, out_w, out_b,
            norm2_w, mlp1_w, mlp2_w, norm_w, cos, sin,
            features,
        )
        aligner(features, align1_w, align1_b, align2_w, align2_b, embeddings, n_h, n_w)
        return features, embeddings
    return vision


WEIGHT_NAMES = ("patch_w", "patch_b", "norm1_w", "qkv_w", "qkv_b", "out_w", "out_b",
                "norm2_w", "mlp1_w", "mlp2_w", "norm_w")
ALIGNER_NAMES = ("align1_w", "align1_b", "align2_w", "align2_b")


class VisionEncoder:
    """Reusable image encoder with caller-provided patches and weights.

    Patches use channel, patch height, patch width order. Projection weights
    use contiguous input-major layout. ``run_config`` is a PyPTO RunConfig
    or its keyword dictionary. The caller
    must allocate the selected device before execution. Reuse this object to
    retain its JIT cache across independently processed images.
    """

    def __init__(self, cfg=VisionConfig(), *, run_config=None):
        from pypto.runtime import RunConfig

        self.cfg = cfg
        self.program = make_vision_program(cfg)
        self.run_config = (RunConfig(**run_config) if isinstance(run_config, dict)
                           else run_config or RunConfig(platform="a5"))

    def __call__(self, patches, weights, n_h, n_w):
        if n_h < 1 or n_w < 1:
            raise ValueError("Image grid dimensions must be positive")
        cfg = self.cfg
        patch_width = 3 * cfg.patch_size ** 2
        patches = patches.flatten(1)
        if tuple(patches.shape) != (n_h * n_w, patch_width):
            raise ValueError("Patch shape does not match the image grid/configuration")
        if patches.dtype != torch.bfloat16 or patches.device.type != "cpu":
            raise ValueError("Patches must be CPU BF16 tensors for the PyPTO runtime")
        d, l, i, ld = cfg.dim, cfg.layers, cfg.intermediate, cfg.language_dim
        shapes = ((patch_width, d), (d,), (l, d), (l, d, 3 * d), (l, 3 * d),
                  (l, d, d), (l, d), (l, d), (l, d, 2 * i), (l, i, d), (d,),
                  (d * cfg.downsample ** 2, ld), (ld,), (ld, ld), (ld,))
        for name, shape in zip((*WEIGHT_NAMES, *ALIGNER_NAMES), shapes):
            weight = weights[name]
            dtype = torch.float32 if name in ("norm1_w", "norm2_w", "norm_w") else torch.bfloat16
            if tuple(weight.shape) != shape or weight.dtype != dtype or weight.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU {dtype} tensor of shape {shape}")
            if not weight.is_contiguous():
                raise ValueError(f"{name} must use contiguous input-major layout")
        cos, sin = vision_rope(n_h, n_w, cfg)
        features = torch.empty(n_h * n_w, d, dtype=torch.bfloat16, device="cpu")
        aligned_rows = ((n_h + cfg.downsample - 1) // cfg.downsample) * (
            (n_w + cfg.downsample - 1) // cfg.downsample)
        embeddings = torch.empty(aligned_rows, ld, dtype=torch.bfloat16, device="cpu")
        self.program(patches.contiguous(), *(weights[name] for name in WEIGHT_NAMES), cos, sin,
                     *(weights[name] for name in ALIGNER_NAMES), features, embeddings, n_h, n_w,
                     config=self.run_config)
        return features, embeddings


def load_vision_weights(checkpoint, cfg=VisionConfig()):
    """Load only vision/aligner tensors from released sharded safetensors."""
    import json
    from safetensors import safe_open

    root = Path(checkpoint)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    cache = {}

    def read(name, transpose=False, fp32=False):
        shard = index[name]
        if shard not in cache:
            cache[shard] = safe_open(root / shard, framework="pt")
        value = cache[shard].get_tensor(name)
        if transpose:
            value = value.t()
        return value.to(torch.float32 if fp32 else torch.bfloat16).contiguous()

    def stack(stem, transpose=False, fp32=False):
        return torch.stack([read(f"vision.blocks.{layer}.{stem}", transpose, fp32)
                            for layer in range(cfg.layers)])

    return dict(
        patch_w=read("vision.patch_embed.proj.weight", True),
        patch_b=read("vision.patch_embed.proj.bias"),
        norm1_w=stack("norm1.weight", fp32=True),
        qkv_w=stack("attn.wqkv.weight", True), qkv_b=stack("attn.wqkv.bias"),
        out_w=stack("attn.wo.weight", True), out_b=stack("attn.wo.bias"),
        norm2_w=stack("norm2.weight", fp32=True),
        mlp1_w=stack("mlp.w1.weight", True), mlp2_w=stack("mlp.w2.weight", True),
        norm_w=read("vision.norm.weight", fp32=True),
        align1_w=read("aligner.w1.weight", True), align1_b=read("aligner.w1.bias"),
        align2_w=read("aligner.w2.weight", True), align2_b=read("aligner.w2.bias"),
    )


def _golden_norm(x, weight):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * weight).bfloat16()


def golden_vision(patches, weights, n_h, n_w, cfg=VisionConfig()):
    """Independent Torch transcription with SDPA and unfold, including BF16 boundaries."""
    def linear(x, name, bias=None):
        return F.linear(x, weights[name].t(), None if bias is None else weights[bias])

    d, heads = cfg.dim, cfg.heads
    channels = d // heads
    half = channels // 2
    frequencies = torch.arange(0, half, 2, dtype=torch.float32, device=patches.device)
    inverse = 1.0 / (cfg.rope_theta ** (frequencies / half))
    hpos = torch.arange(n_h, device=patches.device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=patches.device).unsqueeze(0).expand(n_h, n_w)
    angle = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inverse).flatten(1)
    cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)

    def rotate(x):
        a, b = x.float().chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).bfloat16()

    x = linear(patches.flatten(1), "patch_w", "patch_b")
    for layer in range(cfg.layers):
        normalized = _golden_norm(x, weights["norm1_w"][layer])
        qkv = F.linear(normalized, weights["qkv_w"][layer].t(), weights["qkv_b"][layer])
        q, k, v = (value.reshape(-1, heads, channels) for value in qkv.chunk(3, dim=-1))
        query = rotate(q).transpose(0, 1)
        keys = rotate(k).transpose(0, 1)
        values = v.transpose(0, 1)
        with sdpa_kernel(SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(query, keys, values)
        attended = attended.transpose(0, 1).reshape(-1, d)
        x = x + F.linear(attended, weights["out_w"][layer].t(), weights["out_b"][layer])
        normalized = _golden_norm(x, weights["norm2_w"][layer])
        gate, up = F.linear(normalized, weights["mlp1_w"][layer].t()).chunk(2, dim=-1)
        x = x + F.linear(F.silu(gate) * up, weights["mlp2_w"][layer].t())
    features = _golden_norm(x, weights["norm_w"])
    r = cfg.downsample
    image = features.reshape(n_h, n_w, d).permute(2, 0, 1)
    image = F.pad(image, (0, -n_w % r, 0, -n_h % r))
    unfolded = F.unfold(image.unsqueeze(0), r, stride=r).squeeze(0).t().contiguous()
    embeddings = linear(F.gelu(linear(unfolded, "align1_w", "align1_b")), "align2_w", "align2_b")
    return features, embeddings


def build_inputs(n_h, n_w, cfg=VisionConfig(), *, checkpoint=None, seed=41, patches=None):
    """Prepare caller inputs; checkpoint loading and fixtures never run on import."""
    if n_h < 1 or n_w < 1:
        raise ValueError("Image grid dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    def random(shape, scale=1.0, dtype=torch.bfloat16):
        value = torch.randn(*shape, generator=generator, dtype=torch.float32, device="cpu")
        return (value * scale).to(dtype)

    d, l, i, ld = cfg.dim, cfg.layers, cfg.intermediate, cfg.language_dim
    p = 3 * cfg.patch_size ** 2
    if patches is None:
        patches = random((n_h * n_w, p))
    patches = patches.flatten(1).bfloat16().contiguous()
    if tuple(patches.shape) != (n_h * n_w, p):
        raise ValueError("Patch shape does not match the image grid/configuration")
    weights = load_vision_weights(checkpoint, cfg) if checkpoint else dict(
        patch_w=random((p, d), p ** -0.5), patch_b=random((d,), 0.05),
        norm1_w=torch.ones(l, d, dtype=torch.float32, device="cpu"), qkv_w=random((l, d, 3 * d), d ** -0.5),
        qkv_b=random((l, 3 * d), 0.05), out_w=random((l, d, d), d ** -0.5), out_b=random((l, d), 0.05),
        norm2_w=torch.ones(l, d, dtype=torch.float32, device="cpu"), mlp1_w=random((l, d, 2 * i), d ** -0.5),
        mlp2_w=random((l, i, d), i ** -0.5), norm_w=torch.ones(d, dtype=torch.float32, device="cpu"),
        align1_w=random((d * cfg.downsample ** 2, ld), (d * cfg.downsample ** 2) ** -0.5),
        align1_b=random((ld,), 0.05), align2_w=random((ld, ld), ld ** -0.5), align2_b=random((ld,), 0.05),
    )
    cos, sin = vision_rope(n_h, n_w, cfg)
    return {"patches": patches, **weights, "cos": cos, "sin": sin}


def accuracy(actual, expected, *, rel_l2_limit=0.02, row_l2_limit=0.05, cosine_limit=None, **_kwargs):
    """Fixed engineering gates; these are not official DeepSeek accuracy specifications."""
    a, b = actual.double(), expected.double()
    delta = a - b
    rel_l2 = float(delta.norm() / b.norm().clamp_min(1e-12))
    worst = float((delta.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).max())
    cosine = 1.0 if bool((a == b).all()) else float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    detail = f"relative_l2={rel_l2:.8g}, cosine={cosine:.8g}, worst_row_l2={worst:.8g}"
    detail += f" (limits rel_l2={rel_l2_limit}, cosine={cosine_limit}, row_l2={row_l2_limit})"
    print(detail, flush=True)
    passed = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and rel_l2 <= rel_l2_limit)
    if row_l2_limit is not None:
        passed = passed and worst <= row_l2_limit
    if cosine_limit is not None:
        passed = passed and cosine >= cosine_limit
    return passed, detail


def validation_comparators(cfg):
    """Comparison policies for reduced fixtures and the released BF16 encoder."""
    if cfg.layers == 32 and cfg.dim == 1024:
        return {
            "features": partial(accuracy, rel_l2_limit=0.05, row_l2_limit=None, cosine_limit=0.999),
            "embeddings": partial(accuracy, rel_l2_limit=0.03, row_l2_limit=None, cosine_limit=0.999),
        }
    return {"features": accuracy, "embeddings": accuracy}


def main(argv=None):
    """Compile, execute and compare both device outputs with independent Torch."""
    from golden import ScalarSpec, TensorSpec, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--grid", default="4,5", help="Patch grid height,width")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full", action="store_true", help="Released dimensions (default with checkpoint)")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--patches", type=Path, help="Torch tensor containing preprocessed image patches")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=Path, help="Directory with frozen input and expected output tensors")
    parser.add_argument("--runtime-dir", type=Path, help="Reuse a compiled artifact for device execution")
    parser.add_argument("--ring-heap", type=int, help="Temporary heap bytes per runtime ring (power of two)")
    args = parser.parse_args(argv)
    n_h, n_w = map(int, args.grid.split(","))
    if n_h < 1 or n_w < 1:
        parser.error("Image grid dimensions must be positive")
    cfg = VisionConfig() if args.full or args.checkpoint else VisionConfig(
        dim=128, layers=2, heads=2, intermediate=96, patch_size=2, language_dim=128)
    patches = torch.load(args.patches, weights_only=True) if args.patches else None
    input_names = ("patches", *WEIGHT_NAMES, "cos", "sin", *ALIGNER_NAMES)
    if args.golden_data:
        inputs = {name: torch.load(args.golden_data / "in" / f"{name}.pt", weights_only=True, mmap=True)
                  for name in input_names}
        for name, value in (("n_h", n_h), ("n_w", n_w)):
            saved = torch.load(args.golden_data / "in" / f"{name}.pt", weights_only=True)
            if int(saved) != value:
                parser.error("--grid must match the frozen input grid")
    else:
        inputs = build_inputs(n_h, n_w, cfg, checkpoint=args.checkpoint, seed=args.seed, patches=patches)
    out_rows = ((n_h + cfg.downsample - 1) // cfg.downsample) * ((n_w + cfg.downsample - 1) // cfg.downsample)
    specs = [TensorSpec(name, list(inputs[name].shape), inputs[name].dtype, init_value=inputs[name])
             for name in input_names]
    specs += [TensorSpec("features", [n_h * n_w, cfg.dim], torch.bfloat16),
              TensorSpec("embeddings", [out_rows, cfg.language_dim], torch.bfloat16),
              ScalarSpec("n_h", torch.int32, n_h), ScalarSpec("n_w", torch.int32, n_w)]

    def golden(tensors):
        features, embeddings = golden_vision(tensors["patches"], tensors, n_h, n_w, cfg)
        tensors["features"].copy_(features)
        tensors["embeddings"].copy_(embeddings)

    result = run(make_vision_program(cfg), specs, golden_fn=golden,
                 compare_fn=validation_comparators(cfg),
                 config={"platform": args.platform, "device_id": args.device, "ring_heap": args.ring_heap},
                 compile_only=args.compile_only, save_data=args.save_data,
                 golden_data=args.golden_data, runtime_dir=args.runtime_dir)
    if not result.passed:
        raise SystemExit(result.error or 1)
    if args.compile_only:
        print("COMPILE ONLY: device accuracy was not checked")
    print(f"WORK_DIR={result.work_dir}")


if __name__ == "__main__":
    main()
