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
the router kernel in :mod:`models.glm5_3_flash.gate` owns the deferred norm and
produces the per-token INT8 view; a dense layer has no router, so without this
variant the K = 4096 activation feeding ``dense_mlp`` would have no owner at all.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FLASH, T_DYN
from models.glm5_3_flash.golden import rms_norm
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, quantize_per_token_int8


ROW_PAD = 8
REDUCE_TILE = D // ROW_PAD
EPS = FLASH.rms_norm_eps
assert D % ROW_PAD == 0


def golden_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return rms_norm(x, weight)


def golden_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise and emit the per-token INT8 view the dense MLP consumes."""
    normalized = rms_norm(x, weight)
    quantized, scale = quantize_per_token_int8(normalized)
    return normalized, quantized, scale


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
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Normalize one 4096-wide token per task, accumulating the RMS in FP32."""
    t_dim = pl.tensor.dim(x, 0)
    weight_2d = pl.reshape(weight, [1, D])
    for t in pl.spmd(t_dim, name_hint="glm53_rmsnorm"):
        x_fp32 = pl.cast(pl.tile.load(x, [t, 0], [1, D]), pl.FP32)
        w_fp32 = pl.cast(pl.tile.load(weight_2d, [0, 0], [1, D]), pl.FP32)

        squares = pl.reshape(pl.mul(x_fp32, x_fp32), [ROW_PAD, REDUCE_TILE])
        partial_tmp = pl.create_tile([ROW_PAD, REDUCE_TILE], dtype=pl.FP32)
        partial = pl.row_sum(squares, partial_tmp)
        reduce_tile = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        reduce_tile[0:1, :] = pl.reshape(partial, [1, ROW_PAD])
        reduce_tile = pl.set_validshape(reduce_tile, 1, ROW_PAD)
        sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_sum = pl.row_sum(reduce_tile, sq_sum_tmp)
        sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
        rsqrt_tmp = pl.create_tile([1, ROW_PAD], dtype=pl.FP32)
        inv_rms = pl.tile.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), rsqrt_tmp)
        normalized = pl.mul(pl.mul(x_fp32, pl.tile.read(inv_rms, [0, 0])), w_fp32)
        pl.tile.store(pl.cast(normalized, pl.BF16, mode="rint"), [t, 0], output, shapes=[1, D])
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
    weight_2d = pl.reshape(weight, [1, D])
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > t_dim:
        active_tokens = t_dim
    for t in pl.spmd(t_dim, name_hint="glm53_rmsnorm_quant"):
        x_fp32 = pl.cast(pl.tile.load(x, [t, 0], [1, D]), pl.FP32)
        w_fp32 = pl.cast(pl.tile.load(weight_2d, [0, 0], [1, D]), pl.FP32)
        squares = pl.reshape(pl.mul(x_fp32, x_fp32), [ROW_PAD, REDUCE_TILE])
        sq_partial_tmp = pl.create_tile([ROW_PAD, REDUCE_TILE], dtype=pl.FP32)
        sq_partial = pl.row_sum(squares, sq_partial_tmp)
        sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
        sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
        sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
        sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
        rsqrt_tmp = pl.create_tile([1, ROW_PAD], dtype=pl.FP32)
        inv_rms = pl.tile.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), rsqrt_tmp)
        normalized_fp32 = pl.mul(pl.mul(x_fp32, pl.tile.read(inv_rms, [0, 0])), w_fp32)
        normalized_bf16 = pl.cast(normalized_fp32, pl.BF16, mode="rint")
        pl.tile.store(normalized_bf16, [t, 0], output, shapes=[1, D])

        if t < active_tokens:
            normalized = pl.cast(normalized_bf16, pl.FP32)
            abs_rows = pl.reshape(pl.abs(normalized), [ROW_PAD, REDUCE_TILE])
            partial_tmp = pl.create_tile([ROW_PAD, REDUCE_TILE], dtype=pl.FP32)
            partial = pl.row_max(abs_rows, partial_tmp)
            reduce_tile = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            reduce_tile[0:1, :] = pl.reshape(partial, [1, ROW_PAD])
            reduce_tile = pl.set_validshape(reduce_tile, 1, ROW_PAD)
            amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            amax = pl.row_max(reduce_tile, amax_tmp)
            amax = pl.set_validshape(pl.reshape(amax, [1, ROW_PAD]), 1, 1)
            floor = pl.set_validshape(pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS), 1, 1)
            amax = pl.maximum(amax, floor)
            scale = pl.div(amax, INT8_SCALE_MAX)
            scaled = pl.mul(normalized, pl.div(INT8_SCALE_MAX, pl.tile.read(amax, [0, 0])))
            rounded = pl.cast(scaled, pl.INT32, mode="rint")
            rounded = pl.minimum(
                pl.maximum(rounded, pl.tile.full([1, D], dtype=pl.INT32, value=-127)),
                pl.tile.full([1, D], dtype=pl.INT32, value=127),
            )
            quantized = pl.cast(pl.cast(rounded, pl.FP16, mode="round"), pl.INT8, mode="trunc")
            pl.tile.store(quantized, [t, 0], output_int8, shapes=[1, D])
            pl.tile.store(scale, [t, 0], output_scale, shapes=[1, 1])
        else:
            zero_int8 = pl.cast(pl.tile.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
            pl.tile.store(zero_int8, [t, 0], output_int8, shapes=[1, D])
            zero_scale = pl.set_validshape(pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=0.0), 1, 1)
            pl.tile.store(zero_scale, [t, 0], output_scale, shapes=[1, 1])
    return output, output_int8, output_scale


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
    weight_2d = pl.reshape(weight, [1, D])
    for t in pl.spmd(t_dim, name_hint="glm53_add_residual"):
        x_fp32 = pl.cast(pl.tile.load(x, [t, 0], [1, D]), pl.FP32)
        residual_fp32 = pl.cast(pl.tile.load(residual, [t, 0], [1, D]), pl.FP32)
        updated = pl.cast(pl.add(x_fp32, residual_fp32), pl.BF16, mode="rint")
        pl.tile.store(updated, [t, 0], updated_residual, shapes=[1, D])
        updated_fp32 = pl.cast(updated, pl.FP32)
        w_fp32 = pl.cast(pl.tile.load(weight_2d, [0, 0], [1, D]), pl.FP32)
        squares = pl.reshape(pl.mul(updated_fp32, updated_fp32), [ROW_PAD, REDUCE_TILE])
        partial_tmp = pl.create_tile([ROW_PAD, REDUCE_TILE], dtype=pl.FP32)
        partial = pl.row_sum(squares, partial_tmp)
        reduce_tile = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        reduce_tile[0:1, :] = pl.reshape(partial, [1, ROW_PAD])
        reduce_tile = pl.set_validshape(reduce_tile, 1, ROW_PAD)
        sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_sum = pl.row_sum(reduce_tile, sq_sum_tmp)
        sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
        rsqrt_tmp = pl.create_tile([1, ROW_PAD], dtype=pl.FP32)
        inv_rms = pl.tile.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), EPS), rsqrt_tmp)
        normalized = pl.mul(pl.mul(updated_fp32, pl.tile.read(inv_rms, [0, 0])), w_fp32)
        pl.tile.store(pl.cast(normalized, pl.BF16, mode="rint"), [t, 0], output, shapes=[1, D])
    return output, updated_residual


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


def build_rmsnorm_tensor_specs(case: str, tokens: int = 5):
    """Exercise decode, unaligned token counts, and inactive quantization rows."""
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(53)

    def init_x():
        value = torch.randn(tokens, D, generator=generator).bfloat16()
        if tokens > 1:
            value[0] = 0
        return value

    def init_weight():
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


def compare_quantized_with_normalized(actual, expected, *, actual_outputs, inputs, **_kwargs):
    """Allow BF16 RMS rounding to move rare INT8 results by one level."""
    quantized, _ = quantize_per_token_int8(actual_outputs["output"])
    active = int(inputs["num_tokens"])
    quantized[active:] = 0
    if not torch.equal(actual, quantized):
        return False, "INT8 output does not match quantization of the emitted BF16 output"
    error = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    mismatches = int(error.count_nonzero())
    if int(error.max()) > 1 or mismatches > max(1, actual.numel() // 1000):
        return False, f"INT8 differs from the torch RMSNorm reference at {mismatches}/{actual.numel()} values"
    return True, ""


def main():
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="GLM-5.3-Flash RMSNorm kernel validation")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--case", choices=["all", "plain", "quant", "add"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be positive")
    cases = (
        ("plain", rmsnorm_test, golden_rmsnorm_case),
        ("quant", rmsnorm_quant_test, golden_rmsnorm_quant_case),
        ("add", add_rmsnorm_test, golden_add_rmsnorm_case),
    )
    for name, fn, golden_fn in cases:
        if args.case not in ("all", name):
            continue
        result = run(
            fn=fn,
            specs=build_rmsnorm_tensor_specs(name, args.tokens),
            golden_fn=golden_fn,
            config=dict(platform=args.platform, device_id=args.device),
            compare_fn={
                "output": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "output_int8": compare_quantized_with_normalized,
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
