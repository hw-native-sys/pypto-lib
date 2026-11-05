# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped BF16 Wo-A and dynamic W8A8 Wo-B output projection."""

from __future__ import annotations

import torch
import pypto.language as pl

from .gemm import make_fp32_dequantize, make_int8_gemm, make_pack_weight
from .norm_rope import rope_reference
from .quant import int8_matmul, make_dynamic_int8_quantize, quantize_per_token_int8
from .staged import make_bf16_linear


def output_projection_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for inverse RoPE, grouped Wo-A, and Wo-B W8A8."""

    groups, group_input, rank = wo_a.shape
    rotated = rope_reference(attention, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float()).reshape(
        rotated.shape[0], groups * rank
    ).to(torch.bfloat16)
    payload, token_scale = quantize_per_token_int8(grouped)
    # Prepared A2/A3 assets use canonical ``[K, N]`` layout; the reference
    # helper consumes output-major ``[N, K]`` weights.
    accumulator = int8_matmul(payload, wo_b.transpose(0, 1).contiguous())
    output = (accumulator.float() * token_scale.float() * wo_b_scale.float().reshape(1, -1)).to(torch.bfloat16)
    return {"rotated": rotated, "grouped": grouped, "output": output}


def output_projection_bf16_reference(
    attention: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """CPU golden for the BF16 Wo-A/Wo-B checkpoint path."""

    groups, group_input, rank = wo_a.shape
    flat = attention.reshape(attention.shape[0], heads * head_dim)
    rotated = rope_reference(flat, cosine, -sine, heads, head_dim, rope_dim)
    chunks = rotated.reshape(rotated.shape[0], groups, group_input)
    grouped = torch.einsum("tgi,gir->tgr", chunks.float(), wo_a.float()).to(torch.bfloat16)
    output = torch.matmul(grouped.float().reshape(grouped.shape[0], groups * rank), wo_b.float())
    return {
        "rotated": rotated,
        "grouped": grouped.reshape(grouped.shape[0], groups * rank),
        "output": output.to(torch.bfloat16),
    }


def make_output_projection_bf16(
    capacity: int,
    heads: int,
    head_dim: int,
    rope_dim: int,
    groups: int,
    rank: int,
    hidden: int = 5120,
):
    """Build the grouped BF16 Wo-A/Wo-B projection used by target layers."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64 or hidden % 256:
        raise ValueError("BF16 grouped output projection dimensions are not tile aligned")
    rotate = _make_inverse_rope(capacity, heads, head_dim, rope_dim)
    project_b_entry = make_bf16_linear(capacity, groups * rank, hidden)
    project_b = pl.jit.inline(project_b_entry._func)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        sine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        rotated: pl.Out[pl.Tensor[[capacity, width], pl.BF16]],
        grouped: pl.Out[pl.Tensor[[capacity, groups * rank], pl.BF16]],
        output: pl.Out[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(capacity, name_hint="dsv41_a2a3_bf16_output_init"):
            token = pl.tile.get_block_idx()
            if token >= active:
                rotated_zero = pl.tile.full([1, width], dtype=pl.BF16, value=0.0)
                grouped_zero = pl.tile.full([1, groups * rank], dtype=pl.BF16, value=0.0)
                output_zero = pl.tile.full([1, hidden], dtype=pl.BF16, value=0.0)
                pl.store(rotated_zero, [token, 0], rotated)
                pl.store(grouped_zero, [token, 0], grouped)
                pl.store(output_zero, [token, 0], output)
        inverse_sine = pl.create_tensor([capacity, rope_dim // 2], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_bf16_inverse_sine"):
            inverse_sine[block:block + 1, :] = pl.mul(sine[block:block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        n_tile = 128
        k_tile = 256
        blocks = (groups * rank) // n_tile
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_bf16_wo_a"):
            token0 = (block // blocks) * 16
            column = (block % blocks) * n_tile
            group = column // rank
            group_column = column % rank
            rows = pl.min(16, active - token0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, group_input, k_tile, stage=2):
                source_tile = pl.slice(
                    rotated, [16, k_tile], [token0, group * group_input + k0],
                    valid_shape=[rows, k_tile],
                )
                source_tile = pl.set_validshape(pl.fillpad(source_tile, pad_value=pl.PadValue.zero), 16, k_tile)
                weight_tile = pl.reshape(
                    wo_a[group:group + 1, k0:k0 + k_tile, group_column:group_column + n_tile],
                    [k_tile, n_tile],
                )
                acc = pl.matmul_acc(acc, source_tile, weight_tile, init_cond=(k0 == 0))
            grouped[token0:token0 + 16, column:column + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        project_b(grouped, wo_b, output, active)
        return rotated, grouped, output

    return project


def make_output_projection(capacity: int, heads: int, head_dim: int, rope_dim: int, groups: int, rank: int):
    """Build the standalone A2/A3 output projection."""

    width = heads * head_dim
    group_input = width // groups
    if width % groups or group_input % 256 or rank % 64:
        raise ValueError("grouped output projection requires aligned group and rank widths")
    if rope_dim % 2:
        raise ValueError("rope dimension must be even")
    pairs = rope_dim // 2
    rotate = _make_inverse_rope(capacity, heads, head_dim, rope_dim)
    quantize = make_dynamic_int8_quantize(capacity, groups * rank)
    hidden = 5120
    pack = make_pack_weight(groups * rank, hidden)
    dot = make_int8_gemm(capacity, groups * rank, hidden)
    dequant = make_fp32_dequantize(capacity, hidden)

    @pl.jit.inline
    def project(
        attention: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        rotated = pl.create_tensor([capacity, width], dtype=pl.BF16)
        inverse_sine = pl.create_tensor([capacity, pairs], dtype=pl.FP32)
        for block in pl.spmd(active, name_hint="dsv41_a2a3_inverse_sine"):
            inverse_sine[block:block + 1, :] = pl.mul(sine[block:block + 1, :], -1.0)
        rotate(attention, cosine, inverse_sine, rotated, active)
        grouped = pl.create_tensor([capacity, groups * rank], dtype=pl.BF16)
        for block in pl.spmd(active * groups, name_hint="dsv41_a2a3_wo_a"):
            token = block // groups
            group = block % groups
            source = rotated[token:token + 1, group * group_input:(group + 1) * group_input]
            weights = wo_a[group:group + 1, :, :]
            projected = pl.matmul(source, pl.reshape(weights, [group_input, rank]), out_dtype=pl.FP32)
            grouped[token:token + 1, group * rank:(group + 1) * rank] = pl.cast(projected, pl.BF16, mode="rint")
        payload = pl.create_tensor([capacity, groups * rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize(grouped, payload, token_scale, active)
        packed = pl.create_tensor([groups * rank * (hidden // 64), 64], dtype=pl.INT8)
        pack(wo_b, packed)
        accumulator = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequant(accumulator, token_scale, wo_b_scale, output, active)
        return rotated, grouped, output

    return project


def _make_inverse_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    """Use the common RoPE kernel with a caller-supplied negated sine table."""

    from .norm_rope import make_rope

    return make_rope(capacity, heads, head_dim, rope_dim)


__all__ = [
    "make_output_projection",
    "make_output_projection_bf16",
    "output_projection_bf16_reference",
    "output_projection_reference",
]
