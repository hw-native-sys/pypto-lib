# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMSNorm and the fused residual-add RMSNorm used across the backbone.

Every layer applies ``input_layernorm`` before attention and
``post_attention_layernorm`` before the MLP; both read a stream that mHC has
already collapsed, so the kernel sees a plain ``[T, D]`` row. The gamma vectors
stay BF16 and are never quantized (``modules_to_not_convert`` lists every
``*_layernorm`` and ``*_norm``).

The fused add-and-norm form serves callers that explicitly update a residual
before normalization. The mHC backbone and the model tail use plain RMSNorm.

``rmsnorm_quant`` covers the three **dense** layers (0, 1 and 2). On a sparse layer
the router kernel in :mod:`models.glm5_3_flash.gate` owns the norm and
produces the per-token INT8 view; a dense layer has no router, so without this
variant the K = 4096 activation feeding ``dense_mlp`` would have no owner at all.
The INT8 source is the BF16 normalized output, preserving the activation
boundary before dynamic quantization.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, T_DYN, D
from models.glm5_3_flash.golden import rms_norm
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8

NORM_WIDTH = pl.dynamic("GLM53_NORM_WIDTH")
NORM_T_TILE = 8
NORM_D_TILE = 512
QUANT_D_TILE = D // NORM_T_TILE
EPS = FLASH.rms_norm_eps
assert D % NORM_D_TILE == 0


def golden_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x, weight)


def golden_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Emit BF16 RMSNorm and its dynamically quantized INT8 view."""
    value = x.float()
    square_mean = value.square().mean(dim=-1, keepdim=True)
    inverse = torch.rsqrt(square_mean + EPS)
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


@pl.jit.inline
def rmsnorm(
    x: pl.Tensor[[T_DYN, NORM_WIDTH], pl.BF16],
    weight: pl.Tensor[[NORM_WIDTH], pl.BF16],
    output: pl.Tensor[[T_DYN, NORM_WIDTH], pl.BF16],
):
    """Normalize 512-aligned widths, up to eight tokens per task."""
    t_dim = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint='glm53_rmsnorm'):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        square_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for norm_k in pl.pipeline(0, width, NORM_D_TILE, stage=1):
            norm_part = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, norm_k], valid_shape=[valid_rows, NORM_D_TILE])
            norm_value = pl.cast(norm_part, pl.FP32)
            norm_square = pl.mul(norm_value, norm_value)
            part_sum = pl.reshape(pl.row_sum(norm_square), [1, NORM_T_TILE])
            square_sum = pl.add(square_sum, part_sum)
        norm_argument = pl.add(pl.div(square_sum, pl.cast(pl.cast(width, pl.INT32), pl.FP32)), EPS)
        inverse = pl.rsqrt(norm_argument, high_precision=True)
        inv_rms = pl.reshape(inverse, [NORM_T_TILE, 1])
        for kb in pl.pipeline(width // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source_v1 = pl.fillpad(source, pad_value=pl.PadValue.zero)
            source = pl.set_validshape(source_v1, NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, pl.FP32)
            gamma = pl.reshape(pl.cast(weight[k0:k0 + NORM_D_TILE], pl.FP32), [1, NORM_D_TILE])
            unit = pl.row_expand_mul(value, inv_rms)
            unit_bf16 = pl.cast(unit, pl.BF16, mode="rint")
            normalized = pl.col_expand_mul(pl.cast(unit_bf16, pl.FP32), gamma)
            part_v2 = pl.cast(normalized, pl.BF16, mode='rint')
            output[t0:t0 + NORM_T_TILE, k0:k0 + NORM_D_TILE] = pl.set_validshape(part_v2, valid_rows, NORM_D_TILE)
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
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint='glm53_rmsnorm_quant'):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        square_sum = pl.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
        for norm_k in pl.pipeline(0, D, NORM_D_TILE, stage=1):
            norm_part = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, norm_k], valid_shape=[valid_rows, NORM_D_TILE])
            norm_value = pl.cast(norm_part, pl.FP32)
            norm_square = pl.mul(norm_value, norm_value)
            part_sum = pl.reshape(pl.row_sum(norm_square), [1, NORM_T_TILE])
            square_sum = pl.add(square_sum, part_sum)
        norm_argument = pl.add(pl.mul(square_sum, 1.0 / D), EPS)
        inverse = pl.rsqrt(norm_argument, high_precision=True)
        inv_rms = pl.reshape(inverse, [NORM_T_TILE, 1])
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            source = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            source_v1 = pl.fillpad(source, pad_value=pl.PadValue.zero)
            source = pl.set_validshape(source_v1, NORM_T_TILE, NORM_D_TILE)
            value = pl.cast(source, pl.FP32)
            gamma = pl.reshape(pl.cast(weight[k0:k0 + NORM_D_TILE], pl.FP32), [1, NORM_D_TILE])
            unit = pl.row_expand_mul(value, inv_rms)
            unit_bf16 = pl.cast(unit, pl.BF16, mode="rint")
            normalized = pl.col_expand_mul(pl.cast(unit_bf16, pl.FP32), gamma)
            normalized_bf16 = pl.cast(normalized, pl.BF16, mode='rint')
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
                    rounded = pl.cast(scaled, pl.INT32, mode='rint')
                    rounded_v2 = pl.maximum(rounded, pl.tile.full([1, D], dtype=pl.INT32, value=-127))
                    rounded = pl.minimum(rounded_v2, pl.tile.full([1, D], dtype=pl.INT32, value=127))
                    quantized = pl.cast(pl.cast(rounded, pl.FP16, mode='round'), pl.INT8, mode='trunc')
                    pl.tile.store(quantized, [t, 0], output_int8, shapes=[1, D])
                    scale_tile = pl.tile.full([1, NORM_T_TILE], dtype=pl.FP32, value=0.0)
                    pl.tile.write(scale_tile, [0, 0], scale)
                    scale_tile = pl.set_validshape(scale_tile, 1, 1)
                    pl.tile.store(scale_tile, [t, 0], output_scale, shapes=[1, 1])
                else:
                    zero_int8 = pl.cast(pl.tile.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode='trunc')
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
    for block in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint='glm53_add_residual'):
        t0 = block * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        for kb in pl.pipeline(D // NORM_D_TILE, stage=2):
            k0 = kb * NORM_D_TILE
            x_tile = pl.slice(x, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            x_tile_v1 = pl.fillpad(x_tile, pad_value=pl.PadValue.zero)
            x_tile = pl.set_validshape(x_tile_v1, NORM_T_TILE, NORM_D_TILE)
            residual_tile = pl.slice(residual, [NORM_T_TILE, NORM_D_TILE], [t0, k0], valid_shape=[valid_rows, NORM_D_TILE])
            residual_tile_v2 = pl.fillpad(residual_tile, pad_value=pl.PadValue.zero)
            residual_tile = pl.set_validshape(residual_tile_v2, NORM_T_TILE, NORM_D_TILE)
            updated_v3 = pl.cast(x_tile, pl.FP32)
            updated_v4 = pl.cast(residual_tile, pl.FP32)
            updated_v5 = pl.add(updated_v3, updated_v4)
            updated = pl.cast(updated_v5, pl.BF16, mode='rint')
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
            [
                TensorSpec("output_int8", [tokens, D], torch.int8),
                TensorSpec("output_scale", [tokens, 1], torch.float32),
                ScalarSpec("num_tokens", torch.int32, tokens - 1 if tokens > 1 else 1),
            ]
        )
    if case == "add":
        specs.append(TensorSpec("updated_residual", [tokens, D], torch.bfloat16))
    return specs


def golden_rmsnorm_case(tensors):
    tensors["output"][:] = golden_rmsnorm(tensors["x"], tensors["weight"])


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


def main():
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
    for tokens in ([args.tokens] if args.tokens is not None else [5, 16]):
        for name, fn, golden_fn in cases:
            if args.case not in ("all", name):
                continue
            result = run(
                fn=fn,
                specs=build_rmsnorm_tensor_specs(name, tokens, fixture=args.fixture),
                golden_fn=golden_fn,
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                compare_fn={
                    "output": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                    "output_int8": compare_quantized,
                    # FP32 tiled reduction and rsqrt need not round like torch.mean/rsqrt.
                    # Keep every scale within 1 ppm; zero scales still compare exactly.
                    "output_scale": ratio_allclose(atol=0, rtol=1e-6, max_error_ratio=0),
                },
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            print(f"[GOLDEN] PASS rmsnorm {name}")


__all__ = [
    "add_rmsnorm",
    "golden_add_rmsnorm",
    "golden_rmsnorm",
    "golden_rmsnorm_quant",
    "rmsnorm",
    "rmsnorm_quant",
]


if __name__ == "__main__":
    main()
