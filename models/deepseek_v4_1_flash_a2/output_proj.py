"""Grouped BF16 Wo-A and dynamic W8A8 Wo-B output projection."""

from __future__ import annotations

import torch
import pypto.language as pl

from .gemm import make_fp32_dequantize, make_int8_gemm, make_pack_weight
from .norm_rope import rope_reference
from .quant import int8_matmul, make_dynamic_int8_quantize, quantize_per_token_int8


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


__all__ = ["make_output_projection", "output_projection_reference"]
