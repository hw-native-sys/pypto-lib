"""A2/A3 indexer query projection and per-head INT8 quantization.

The V4.1 indexer is separate from the attention QKV path.  vLLM-Ascend
projects ``q_lora`` into ``index_n_heads * index_head_dim`` and quantizes each
128-wide head with an FP16 scale before calling QLI.  A2/A3 has no FP8
indexer path, so this module keeps the complete boundary in BF16/INT8/FP16.
"""

from __future__ import annotations

import torch
import pypto.language as pl

from .quant import make_dynamic_int8_quantize
from .norm_rope import make_rope, rope_reference
from .quant import w8a8_linear


INDEX_HEAD_DIM = 128
INDEX_HEADS = 32
INDEX_ROPE_DIM = 64
INDEX_SCALE_MIN = 2.0 ** -24


def quantize_indexer_query_reference(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for vLLM's A2/A3 ``quantize_indexer_query``.

    The scale is rounded to FP16 before it is used to form the INT8 payload;
    this differs from the index-cache writer, whose payload uses the raw FP32
    scale and publishes an FP16 copy.
    """

    if query.ndim != 3 or query.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("indexer query must have shape [tokens, heads, 128]")
    source = query.detach().float()
    scale = (source.abs().amax(dim=-1) / 127.0).to(torch.float16)
    scale = torch.maximum(scale, torch.full_like(scale, INDEX_SCALE_MIN, dtype=torch.float16))
    payload = torch.round(source / scale.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload, scale


def _make_indexer_query_quantize_inline(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build the PyPTO per-head INT8 query quantizer used by QLI."""

    if capacity % 16 or heads <= 0 or head_dim != INDEX_HEAD_DIM:
        raise ValueError("indexer quantization requires capacity divisible by 16 and head_dim=128")
    total_rows = capacity * heads

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [total_rows, head_dim])
        output = pl.reshape(payload, [total_rows, head_dim])
        scale_output = pl.reshape(scales, [total_rows, 1])
        blocks = (active * heads + 15) // 16
        for block in pl.spmd(blocks, name_hint="dsv41_a2a3_indexer_query_quant"):
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.div(amax, 127.0)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            rounded_scale = pl.maximum(rounded_scale, INDEX_SCALE_MIN)
            normalized = pl.row_expand_div(value_fp32, rounded_scale)
            rounded = pl.cast(
                pl.minimum(pl.maximum(normalized, -127.0), 127.0),
                pl.INT32,
                mode="rint",
            )
            output[row:row + 16, :] = pl.set_validshape(
                pl.cast(rounded, pl.INT8, mode="trunc"), valid_rows, head_dim
            )
            scale_output[row:row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(rounded_scale, pl.FP16, mode="rint"), [16, 1]),
                valid_rows,
                1,
            )
        return payload, scales

    return quantize


def make_indexer_query_quantize(
    capacity: int,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
):
    """Build a standalone PyPTO indexer quantization entry point."""

    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, scales, active)
        return payload, scales

    return kernel


def make_l3_indexer_query(
    capacity: int,
    q_lora_rank: int = 1280,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
):
    """Project ``q_lora`` and publish the QLI INT8 query boundary.

    ``weight`` is canonical A2/A3 ``[K, N]`` INT8 and ``weight_scale`` is
    FP32 per output channel.  ``query`` remains BF16 for diagnostics while
    ``payload``/``scales`` are the exact QLI ABI.
    """

    if heads * head_dim != 4096 or rope_dim != INDEX_ROPE_DIM:
        raise ValueError("V4.1 A2/A3 indexer uses 32 heads x 128 with a 64-wide RoPE tail")
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    quantize = _make_indexer_query_quantize_inline(capacity, heads, head_dim)
    quantize_input = make_dynamic_int8_quantize(capacity, q_lora_rank)
    pairs = rope_dim // 2
    width = heads * head_dim
    blocks = width // 64

    @pl.jit.inline
    def project(
        value: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, q_lora_rank], dtype=pl.INT8)
        token_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quantize_input(value, payload, token_scale, active)
        for block in pl.spmd(((active + 15) // 16) * blocks,
                             name_hint="dsv41_a2a3_l3_indexer_gemm"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, 64], dtype=pl.INT32)
            for k0 in pl.pipeline(0, q_lora_rank, 256, stage=2):
                lhs = pl.slice(payload, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0:k0 + 256, n0:n0 + 64]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            value_fp32 = pl.cast(acc, pl.FP32)
            value_fp32 = pl.col_expand_mul(value_fp32, pl.reshape(weight_scale[n0:n0 + 64], [1, 64]))
            value_fp32 = pl.row_expand_mul(value_fp32, token_scale[t0:t0 + 16, :])
            output[t0:t0 + 16, n0:n0 + 64] = pl.set_validshape(
                pl.cast(value_fp32, pl.BF16, mode="rint"), rows, 64
            )
        return output

    @pl.jit
    def kernel(
        q_lora: pl.Tensor[[capacity, q_lora_rank], pl.BF16],
        weight: pl.Tensor[[q_lora_rank, width], pl.INT8],
        weight_scale: pl.Tensor[[width], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        query: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.BF16]],
        payload: pl.InOut[pl.Tensor[[capacity, heads, head_dim], pl.INT8]],
        scales: pl.InOut[pl.Tensor[[capacity, heads], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        projected = pl.create_tensor([capacity, width], dtype=pl.BF16)
        project(q_lora, weight, weight_scale, projected, active)
        query_flat = pl.reshape(query, [capacity, width])
        rotate(projected, cosine, sine, query_flat, active)
        quantize(query, payload, scales, active)
        return query, payload, scales

    return kernel


def indexer_query_reference(
    q_lora: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    rope_dim: int = INDEX_ROPE_DIM,
) -> dict[str, torch.Tensor]:
    """CPU reference for the complete L3 indexer query boundary."""

    # The PyPTO ABI stores canonical [K, N], while the Torch helper consumes
    # output-major [N, K] weights.
    projected = w8a8_linear(q_lora, weight.transpose(0, 1).contiguous(), weight_scale)
    query = rope_reference(projected, cosine, sine, heads, head_dim, rope_dim)
    query = query.reshape(query.shape[0], heads, head_dim)
    payload, scales = quantize_indexer_query_reference(query)
    return {"query": query, "payload": payload, "scales": scales}


__all__ = [
    "INDEX_HEAD_DIM",
    "INDEX_HEADS",
    "INDEX_ROPE_DIM",
    "indexer_query_reference",
    "make_indexer_query_quantize",
    "make_l3_indexer_query",
    "quantize_indexer_query_reference",
]
