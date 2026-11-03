# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash RMSNorm, residual normalization and activation quantization."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, D
from models.glm5_3_flash.elementwise import (
    _sum_fp32_reference, accurate_divide_parts, accurate_row_sum_parts, accurate_rsqrt,
)
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8
from models.glm5_3_flash.utils import _check, _report

T_DYN = pl.dynamic('GLM53_T_DYN')

NORM_WIDTH = pl.dynamic("GLM53_NORM_WIDTH")
EPS = FLASH.rms_norm_eps
NORM_T_TILE = 8
NORM_D_TILE = 128
QUANT_D_TILE = D // NORM_T_TILE


@pl.jit.inline
def rmsnorm(
    x: pl.Tensor[[T_DYN, NORM_WIDTH], pl.BF16],
    weight: pl.Tensor[[NORM_WIDTH], pl.BF16],
    output: pl.Tensor[[T_DYN, NORM_WIDTH], pl.BF16],
):
    """Normalize 64-aligned widths, up to eight tokens per task."""
    t_dim = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="glm53_rmsnorm"):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        square_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        square_error = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for norm_k in pl.pipeline(0, width, NORM_D_TILE, stage=1):
            norm_columns = pl.min(NORM_D_TILE, width - norm_k)
            norm_part = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, norm_k], valid_shape=[valid_rows, norm_columns])
            norm_value = pl.cast(norm_part, pl.FP32)
            norm_square = pl.create_tensor([NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32)
            norm_square[:, :] = pl.full([NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32, value=0.0)
            norm_square[:, :] = pl.mul(norm_value, norm_value)
            part_high, part_low = accurate_row_sum_parts(norm_square)
            part_sum = pl.reshape(part_high, [1, NORM_T_TILE])
            part_error = pl.reshape(part_low, [1, NORM_T_TILE])
            updated_sum = pl.add(square_sum, part_sum)
            recovered = pl.sub(updated_sum, square_sum)
            error = pl.add(pl.sub(square_sum, pl.sub(updated_sum, recovered)), pl.sub(part_sum, recovered))
            square_error = pl.add(square_error, pl.add(error, part_error))
            square_sum = updated_sum
        sum_high = pl.create_tensor([1, NORM_T_TILE], dtype=pl.FP32)
        sum_low = pl.create_tensor([1, NORM_T_TILE], dtype=pl.FP32)
        denominator = pl.create_tensor([1, NORM_T_TILE], dtype=pl.FP32)
        sum_high[:, :] = square_sum
        sum_low[:, :] = square_error
        denominator[:, :] = pl.add(pl.mul(square_sum, 0.0), pl.cast(pl.cast(width, pl.INT32), pl.FP32))
        mean = accurate_divide_parts(sum_high, sum_low, denominator, 1, NORM_T_TILE)
        norm_argument = pl.create_tensor([1, NORM_T_TILE], dtype=pl.FP32)
        norm_argument[:, :] = pl.add(mean, EPS)
        inverse = accurate_rsqrt(norm_argument)
        inv_rms = pl.reshape(inverse, [NORM_T_TILE, 1])
        for k0 in pl.pipeline(0, width, NORM_D_TILE, stage=2):
            norm_columns = pl.min(NORM_D_TILE, width - k0)
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, norm_columns])
            source_fillpad = pl.fillpad(source, pad_value=pl.PadValue.zero)
            source = pl.set_validshape(source_fillpad, NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, pl.FP32)
            gamma_values = pl.slice(weight, [NORM_D_TILE], [k0], valid_shape=[norm_columns])
            gamma_row = pl.reshape(gamma_values, [1, NORM_D_TILE])
            gamma_buffer = pl.create_tensor([1, NORM_D_TILE], dtype=pl.BF16)
            gamma_buffer[:, :] = pl.full([1, NORM_D_TILE], dtype=pl.BF16, value=0.0)
            gamma_buffer[:, :] = gamma_row
            gamma = pl.cast(gamma_buffer[:, :], pl.FP32)
            unit = pl.row_expand_mul(value, inv_rms)
            unit_bf16 = pl.cast(unit, pl.BF16, mode="rint")
            normalized = pl.col_expand_mul(pl.cast(unit_bf16, pl.FP32), gamma)
            normalized_bf16 = pl.cast(normalized, pl.BF16, mode="rint")
            output[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = pl.set_validshape(normalized_bf16, valid_rows, norm_columns)
    return output


@pl.jit.inline
def rmsnorm_quant(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    output_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    output_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Fuse BF16 RMSNorm with the dense MLP's per-token INT8 view."""
    t_dim = pl.tensor.dim(x, 0)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > t_dim:
        active_tokens = t_dim
    quant_source_fp32 = pl.create_tensor([t_dim, D], dtype=pl.FP32)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="glm53_rmsnorm_quant"):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        square_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        square_error = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for norm_k in pl.pipeline(0, D, NORM_D_TILE, stage=1):
            norm_part = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, norm_k], valid_shape=[valid_rows, NORM_D_TILE])
            norm_value = pl.cast(norm_part, pl.FP32)
            norm_square = pl.create_tensor([NORM_T_TILE, NORM_D_TILE], dtype=pl.FP32)
            norm_square[:, :] = pl.mul(norm_value, norm_value)
            part_high, part_low = accurate_row_sum_parts(norm_square)
            part_sum = pl.reshape(part_high, [1, NORM_T_TILE])
            part_error = pl.reshape(part_low, [1, NORM_T_TILE])
            updated_sum = pl.add(square_sum, part_sum)
            recovered = pl.sub(updated_sum, square_sum)
            error = pl.add(pl.sub(square_sum, pl.sub(updated_sum, recovered)), pl.sub(part_sum, recovered))
            square_error = pl.add(square_error, pl.add(error, part_error))
            square_sum = updated_sum
        norm_argument = pl.create_tensor([1, NORM_T_TILE], dtype=pl.FP32)
        norm_argument[:, :] = pl.add(pl.mul(pl.add(square_sum, square_error), 1.0 / D), EPS)
        inverse = accurate_rsqrt(norm_argument)
        inv_rms = pl.reshape(inverse, [NORM_T_TILE, 1])
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source_fillpad = pl.fillpad(source, pad_value=pl.PadValue.zero)
            source = pl.set_validshape(source_fillpad, NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, pl.FP32)
            gamma = pl.reshape(pl.cast(weight[k0 : k0 + NORM_D_TILE], pl.FP32), [1, NORM_D_TILE])
            unit = pl.row_expand_mul(value, inv_rms)
            unit_bf16 = pl.cast(unit, pl.BF16, mode="rint")
            normalized = pl.col_expand_mul(pl.cast(unit_bf16, pl.FP32), gamma)
            normalized_bf16 = pl.cast(normalized, pl.BF16, mode="rint")
            quant_source = pl.cast(normalized_bf16, pl.FP32)
            quant_source_fp32[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = pl.set_validshape(quant_source, valid_rows, NORM_D_TILE)
            output[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = pl.set_validshape(normalized_bf16, valid_rows, NORM_D_TILE)
        for row in pl.range(NORM_T_TILE):
            t = t0 + row
            if t < t_dim:
                if t < active_tokens:
                    quant_row_fp32 = pl.tile.load(quant_source_fp32, [t, 0], [1, D])
                    abs_rows = pl.reshape(pl.abs(quant_row_fp32), [NORM_T_TILE, QUANT_D_TILE])
                    partial_tmp = pl.create_tile([NORM_T_TILE, QUANT_D_TILE], dtype=pl.FP32)
                    partial = pl.row_max(abs_rows, partial_tmp)
                    reduce_tile = pl.create_tile([NORM_T_TILE, NORM_T_TILE], dtype=pl.FP32)
                    reduce_tile[0:1, :] = pl.reshape(partial, [1, NORM_T_TILE])
                    reduce_tile = pl.set_validshape(reduce_tile, 1, NORM_T_TILE)
                    amax_tmp = pl.create_tile([NORM_T_TILE, NORM_T_TILE], dtype=pl.FP32)
                    amax = pl.row_max(reduce_tile, amax_tmp)
                    amax = pl.set_validshape(pl.reshape(amax, [1, NORM_T_TILE]), 1, 1)
                    floor = pl.set_validshape(pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS), 1, 1)
                    amax = pl.maximum(amax, floor)
                    # Scalar division matches the Qwen quantizer and the FP32 reference.
                    # Tile/scalar division rounds via reciprocal multiplication on A3.
                    amax_scalar = pl.tile.read(amax, [0, 0])
                    scale_quant = INT8_SCALE_MAX / amax_scalar
                    scale_dequant = amax_scalar / INT8_SCALE_MAX
                    scale = scale_dequant
                    scaled = pl.mul(quant_row_fp32, scale_quant)
                    rounded = pl.cast(scaled, pl.INT32, mode="rint")
                    rounded_maximum = pl.maximum(rounded, pl.tile.full([1, D], dtype=pl.INT32, value=-127))
                    rounded = pl.minimum(rounded_maximum, pl.tile.full([1, D], dtype=pl.INT32, value=127))
                    quantized = pl.cast(pl.cast(rounded, pl.FP16, mode="round"), pl.INT8, mode="trunc")
                    pl.tile.store(quantized, [t, 0], output_int8, shapes=[1, D])
                    scale_tile = pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
                    pl.tile.write(scale_tile, [0, 0], scale)
                    scale_tile = pl.set_validshape(scale_tile, 1, 1)
                    pl.tile.store(scale_tile, [t, 0], output_scale, shapes=[1, 1])
                else:
                    zero_int8 = pl.cast(pl.tile.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
                    pl.tile.store(zero_int8, [t, 0], output_int8, shapes=[1, D])
                    zero_scale = pl.set_validshape(pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0), 1, 1)
                    pl.tile.store(zero_scale, [t, 0], output_scale, shapes=[1, 1])
    return (output, output_int8, output_scale)


@pl.jit.inline
def add_rmsnorm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    updated_residual: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Fuse the BF16 residual update and the subsequent RMSNorm."""
    t_dim = pl.tensor.dim(x, 0)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="glm53_add_residual"):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            x_tile = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            x_tile_fillpad = pl.fillpad(x_tile, pad_value=pl.PadValue.zero)
            x_tile = pl.set_validshape(x_tile_fillpad, NORM_T_TILE, NORM_D_TILE)
            residual_tile = pl.slice(residual, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            residual_tile_fillpad = pl.fillpad(residual_tile, pad_value=pl.PadValue.zero)
            residual_tile = pl.set_validshape(residual_tile_fillpad, NORM_T_TILE, NORM_D_TILE)
            x_tile_fp32 = pl.cast(x_tile, pl.FP32)
            residual_tile_fp32 = pl.cast(residual_tile, pl.FP32)
            updated_add = pl.add(x_tile_fp32, residual_tile_fp32)
            updated = pl.cast(updated_add, pl.BF16, mode="rint")
            updated_residual[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = pl.set_validshape(updated, valid_rows, NORM_D_TILE)
    rmsnorm(updated_residual, weight, output)
    return (output, updated_residual)


@pl.jit
def rmsnorm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    rmsnorm(x, weight, output)
    return output


@pl.jit
def rmsnorm_quant_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output_int8: pl.Out[pl.Tensor[[T_DYN, D], pl.INT8]],
    output_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    output_int8.bind_dynamic(0, T_DYN)
    output_scale.bind_dynamic(0, T_DYN)
    rmsnorm_quant(x, weight, output, output_int8, output_scale, num_tokens)
    return output


@pl.jit
def add_rmsnorm_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    updated_residual: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    updated_residual.bind_dynamic(0, T_DYN)
    add_rmsnorm(x, residual, weight, output, updated_residual)
    return output


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = FLASH.rms_norm_eps) -> torch.Tensor:
    """Apply RMSNorm in FP32 and restore the activation dtype."""
    dtype = x.dtype
    value = x.float()
    square_mean = value.double().square().mean(dim=-1, keepdim=True).float()
    value = value * torch.rsqrt((square_mean + eps).double()).float()
    return (weight.to(dtype).float() * value.to(dtype).float()).to(dtype)


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = FLASH.rms_norm_eps,
) -> torch.Tensor:
    """KDA output norm: strict FP32 RMSNorm over ``linear_head_dim``, then a sigmoid gate.

    Mirrors ``Glm5NextTextRMSNormGated``: the weights are *not* downcast, and the
    gate is applied after the norm, in FP32.
    """
    dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt((_sum_fp32_reference(value.square()) / value.shape[-1] + eps).double()).float()
    value = weight.float() * value
    sigmoid = torch.reciprocal(1.0 + torch.exp(-gate.double()).float())
    return (value * sigmoid).to(dtype)


def golden_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit BF16 RMSNorm and its dynamically quantized INT8 view."""
    value = x.float()
    square_mean = value.double().square().mean(dim=-1, keepdim=True).float()
    inverse = torch.rsqrt((square_mean + EPS).double()).float()
    unit = (value * inverse).to(x.dtype).float()
    normalized = (unit * weight.to(x.dtype).float()).to(x.dtype)
    quant_source = normalized.float()
    quantized, _scale = quantize_per_token_int8(quant_source)
    amax = quant_source.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    return normalized.to(x.dtype), quantized, scale


def golden_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the normalised activation and the updated residual."""
    updated = x.float() + residual.float()
    return rms_norm(updated.to(x.dtype), weight), updated.to(x.dtype)


def build_rmsnorm_tensor_specs(case: str, tokens: int = 5, *, fixture: str = "random"):
    """Exercise decode, unaligned token counts, and inactive quantization rows."""
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(53)

    def init_x():
        if fixture == "rounding":
            pattern = torch.tensor([-127, -126.5, -1.5078125, -1.5, -0.5, 0.5, 1.5, 127])
            value = pattern.repeat(tokens, D // pattern.numel()).bfloat16()
        else:
            value = torch.randn(tokens, D, generator=generator).bfloat16()
        if tokens > 1:
            value[0] = 0
        return value

    def init_weight():
        if fixture == "rounding":
            return torch.ones(D, dtype=torch.bfloat16)
        return (1 + 0.1 * torch.randn(D, generator=generator)).bfloat16()

    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("weight", [D], torch.bfloat16, init_value=init_weight),
    ]
    if case == "add":
        specs.insert(1, TensorSpec("residual", [tokens, D], torch.bfloat16, init_value=init_x))
    specs.append(TensorSpec("output", [tokens, D], torch.bfloat16))
    if case == "quant":
        specs.extend(
            [TensorSpec("output_int8", [tokens, D], torch.int8), TensorSpec("output_scale", [tokens, 1], torch.float32), ScalarSpec("num_tokens", torch.int32, tokens - 1 if tokens > 1 else 1)],
        )
    if case == "add":
        specs.append(TensorSpec("updated_residual", [tokens, D], torch.bfloat16))
    return specs


def golden_rmsnorm_case(tensors):
    tensors["output"][:] = rms_norm(tensors["x"], tensors["weight"])


def golden_rmsnorm_quant_case(tensors):
    normalized, quantized, scale = golden_rmsnorm_quant(tensors["x"], tensors["weight"])
    active = tensors["num_tokens"]
    tensors["output"][:] = normalized
    tensors["output_int8"][:active] = quantized[:active]
    tensors["output_int8"][active:] = 0
    tensors["output_scale"][:active] = scale[:active]
    tensors["output_scale"][active:] = 0


def golden_add_rmsnorm_case(tensors):
    normalized, updated = golden_add_rmsnorm(tensors["x"], tensors["residual"], tensors["weight"])
    tensors["output"][:] = normalized
    tensors["updated_residual"][:] = updated


def compare_quantized(actual, expected, *, actual_outputs, inputs, **_kwargs):
    """Check normalized-input quantization and exact inactive-row zeroing."""
    active = int(inputs["num_tokens"])
    if torch.count_nonzero(actual[active:]):
        return False, "inactive INT8 rows are not zero"
    if torch.count_nonzero(actual_outputs["output_scale"][active:]):
        return False, "inactive dequantization scales are not zero"
    error = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    mismatches = int(error.count_nonzero())
    if int(error.max()) > 1 or mismatches > max(1, actual.numel() // 1000):
        return False, f"INT8 differs from the torch RMSNorm reference at {mismatches}/{actual.numel()} values"
    return True, ""


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalisation used on the KDA query and key.

    Note the ``+ eps`` inside the square root rather than ``max(norm, eps)``; the
    reference is explicit that this differs from ``F.normalize``.
    """
    value = x.float()
    norm = torch.sqrt((_sum_fp32_reference(value.square()) + eps).double()).float()
    return (value / norm).to(x.dtype)


def run_norm_goldens(golden_rms_norm, golden_rms_norm_gated, golden_l2norm) -> None:
    """Exercise the three normalisations the backbone shares."""
    torch.manual_seed(19)
    tokens, hidden = 5, 64
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(hidden, dtype=torch.bfloat16)

    normalized = golden_rms_norm(x, weight)
    _check(normalized.shape == x.shape, f"rms_norm shape {tuple(normalized.shape)}")
    _check(normalized.dtype is x.dtype, f"rms_norm dtype {normalized.dtype}")

    heads, head_dim = 4, 16
    value = torch.randn(tokens, heads, head_dim)
    gate_value = torch.randn(tokens, heads, head_dim)
    gated = golden_rms_norm_gated(value, torch.ones(head_dim), gate_value)
    _check(gated.shape == value.shape, f"rms_norm_gated shape {tuple(gated.shape)}")
    ungated = golden_rms_norm_gated(value, torch.ones(head_dim), torch.full_like(gate_value, 1e9))
    _check(
        torch.allclose(gated, ungated * torch.sigmoid(gate_value), atol=0.0001),
        "rms_norm_gated must apply a sigmoid gate after the norm",
    )

    unit = golden_l2norm(torch.randn(tokens, head_dim))
    _check((unit.square().sum(dim=-1) - 1.0).abs().max() < 0.0001, "l2norm must return unit rows")

    _report("norms")


def reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Glm5NextTextRMSNorm: FP32 variance, weight applied after the inverse cast."""
    variance = x.pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return weight * normalized


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Glm5NextTextUnweightedRMSNorm (mHC stream-norm), no learned weight."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def reference_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: ``eps`` added inside the sqrt (intentionally not max(...))."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def main():
    run_norm_goldens(rms_norm, rms_norm_gated, l2norm)
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="GLM-5.3-Flash RMSNorm kernel validation")
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, help="token count; default covers ragged 5 and aligned 16")
    parser.add_argument("--case", choices=["all", "plain", "quant", "add"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fixture", choices=["random", "rounding"], default="random")
    args = parser.parse_args()

    if args.tokens is not None and args.tokens < 1:
        parser.error("--tokens must be positive")
    cases = (
        ("plain", rmsnorm_test, golden_rmsnorm_case),
        ("quant", rmsnorm_quant_test, golden_rmsnorm_quant_case),
        ("add", add_rmsnorm_test, golden_add_rmsnorm_case),
    )
    for tokens in [args.tokens] if args.tokens is not None else [5, 16]:
        for name, fn, golden_fn in cases:
            if args.case not in ("all", name):
                continue
            result = run(
                fn=fn,
                specs=build_rmsnorm_tensor_specs(name, tokens, fixture=args.fixture),
                golden_fn=golden_fn,
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                compare_fn={"output": ratio_allclose(atol=0.0001, rtol=1.0 / 128), "output_int8": compare_quantized, "output_scale": ratio_allclose(atol=0, rtol=1e-06, max_error_ratio=0)},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            print(f"[GOLDEN] PASS rmsnorm {name}")


if __name__ == "__main__":
    main()
