"""Layer-0 QKV W8A8 projection for the A2/A3 decoder contract.

The vLLM-Ascend A2/A3 path keeps the three projection weights in canonical
``[K, N]`` INT8 layout.  This module deliberately exposes every intermediate
boundary so a layer-0 dump can be compared without running the full decoder:
the input projection, Q RMSNorm, Q-B projection, KV projection, KV RMSNorm,
and the two RoPE outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import pypto.language as pl

from .gemm import make_fp32_dequantize, make_int8_gemm, make_pack_weight
from .norm_rope import make_rope, make_rmsnorm, rms_norm_reference, rope_reference
from .quant import int8_matmul, make_dynamic_int8_quantize, quantize_per_token_int8


@dataclass(frozen=True)
class QKVShape:
    """Static dimensions required by the layer-0 QKV leaf."""

    hidden_size: int
    q_lora_rank: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    rms_norm_eps: float = 1.0e-20

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def rope_pairs(self) -> int:
        return self.qk_rope_head_dim // 2


def qkv_shape(config: Mapping[str, Any] | QKVShape) -> QKVShape:
    """Normalize a model config or an already materialized shape object."""

    if isinstance(config, QKVShape):
        return config
    return QKVShape(
        hidden_size=int(config["hidden_size"]),
        q_lora_rank=int(config["q_lora_rank"]),
        num_attention_heads=int(config["num_attention_heads"]),
        head_dim=int(config["head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        rms_norm_eps=float(config.get("rms_norm_eps", 1.0e-20)),
    )


def qkv_reference(
    value: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cosine: torch.Tensor,
    sine: torch.Tensor,
    config: Mapping[str, Any] | QKVShape,
) -> dict[str, torch.Tensor]:
    """CPU/PyTorch golden for the same boundaries as :func:`make_qkv_w8a8`.

    The returned tensors include the three projected BF16 values and the two
    RoPE outputs.  All matrix products use an explicit INT32 accumulator and
    FP32 token/channel dequantization, matching ``W8A8_DYNAMIC``.
    """

    shape = qkv_shape(config)
    x_i8, x_scale = quantize_per_token_int8(value)
    qa_i32 = int8_matmul(x_i8, weights["wq_a"].transpose(-2, -1).contiguous())
    qa = (qa_i32.float() * x_scale.float() * weights["wq_a_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    latent = rms_norm_reference(qa, weights["q_norm_weight"], shape.rms_norm_eps)

    q_i8, q_scale = quantize_per_token_int8(latent)
    qb_i32 = int8_matmul(q_i8, weights["wq_b"].transpose(-2, -1).contiguous())
    qb = (qb_i32.float() * q_scale.float() * weights["wq_b_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    query = rope_reference(
        qb,
        cosine,
        sine,
        shape.num_attention_heads,
        shape.head_dim,
        shape.qk_rope_head_dim,
    )

    kv_i32 = int8_matmul(x_i8, weights["wkv"].transpose(-2, -1).contiguous())
    kv_linear = (kv_i32.float() * x_scale.float() * weights["wkv_scale"].float().reshape(1, -1)).to(torch.bfloat16)
    kv_normed = rms_norm_reference(kv_linear, weights["kv_norm_weight"], shape.rms_norm_eps)
    window_kv = rope_reference(kv_normed, cosine, sine, 1, shape.head_dim, shape.qk_rope_head_dim)
    return {
        "query_latent": latent,
        "query": query,
        "window_kv": window_kv,
        "wq_a_output": qa,
        "wq_b_output": qb,
        "wkv_output": kv_linear,
        "wkv_normed_output": kv_normed,
    }


def make_qkv_w8a8(capacity: int, config: Mapping[str, Any] | QKVShape):
    """Build the standalone layer-0 QKV W8A8 PyPTO entry point.

    ``cache`` and ``cache_lens`` are opaque state guards here.  The QKV leaf
    does not write SWA state; cache publication is intentionally performed by
    :mod:`swa_cache` so its slot mapping can be validated independently.
    """

    shape = qkv_shape(config)
    if capacity <= 0 or capacity % 16:
        raise ValueError("QKV capacity must be a positive multiple of 16 for the INT8 GEMM tile")
    if shape.qk_rope_head_dim % 2 or shape.qk_rope_head_dim > shape.head_dim:
        raise ValueError("qk_rope_head_dim must be even and no larger than head_dim")

    hidden = shape.hidden_size
    latent = shape.q_lora_rank
    heads = shape.num_attention_heads
    head_dim = shape.head_dim
    query_width = shape.query_width
    rope_pairs = shape.rope_pairs
    rope_dim = shape.qk_rope_head_dim

    quant_x = make_dynamic_int8_quantize(capacity, hidden)
    quant_q = make_dynamic_int8_quantize(capacity, latent)
    pack_qa = make_pack_weight(hidden, latent)
    pack_qb = make_pack_weight(latent, query_width)
    pack_kv = make_pack_weight(hidden, head_dim)
    dot_qa = make_int8_gemm(capacity, hidden, latent)
    dot_qb = make_int8_gemm(capacity, latent, query_width)
    dot_kv = make_int8_gemm(capacity, hidden, head_dim)
    dequant_qa = make_fp32_dequantize(capacity, latent)
    dequant_qb = make_fp32_dequantize(capacity, query_width)
    dequant_kv = make_fp32_dequantize(capacity, head_dim)
    norm_q = make_rmsnorm(capacity, latent, shape.rms_norm_eps)
    norm_kv = make_rmsnorm(capacity, head_dim, shape.rms_norm_eps)
    rope_q = make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = make_rope(capacity, 1, head_dim, rope_dim)

    @pl.jit.inline
    def qkv_w8a8_inline(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, query_width], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_i8, x_scale, active)

        qa_packed = pl.create_tensor([hidden * (latent // 64), 64], dtype=pl.INT8)
        pack_qa(wq_a, qa_packed)
        qa_i32 = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, qa_packed, qa_i32, active)
        dequant_qa(qa_i32, x_scale, wq_a_scale, wq_a_output, active)
        norm_q(wq_a_output, q_norm_weight, query_latent, active)

        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_packed = pl.create_tensor([latent * (query_width // 64), 64], dtype=pl.INT8)
        pack_qb(wq_b, qb_packed)
        qb_i32 = pl.create_tensor([capacity, query_width], dtype=pl.INT32)
        dot_qb(q_i8, qb_packed, qb_i32, active)
        dequant_qb(qb_i32, q_scale, wq_b_scale, wq_b_output, active)
        rope_q(wq_b_output, cosine, sine, query, active)

        kv_packed = pl.create_tensor([hidden * (head_dim // 64), 64], dtype=pl.INT8)
        pack_kv(wkv, kv_packed)
        kv_i32 = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, kv_packed, kv_i32, active)
        dequant_kv(kv_i32, x_scale, wkv_scale, wkv_output, active)
        norm_kv(wkv_output, kv_norm_weight, wkv_normed_output, active)
        rope_kv(wkv_normed_output, cosine, sine, window_kv, active)
        return (
            positions, query_latent, query, window_kv,
            wq_a_output, wq_b_output, wkv_output, wkv_normed_output,
            cache, cache_lens,
        )

    @pl.jit
    def qkv_w8a8(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, latent], pl.INT8],
        wq_a_scale: pl.Tensor[[latent], pl.FP32],
        q_norm_weight: pl.Tensor[[latent], pl.BF16],
        wq_b: pl.Tensor[[latent, query_width], pl.INT8],
        wq_b_scale: pl.Tensor[[query_width], pl.FP32],
        wkv: pl.Tensor[[hidden, head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[head_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        sine: pl.Tensor[[capacity, rope_pairs], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wq_a_output: pl.InOut[pl.Tensor[[capacity, latent], pl.BF16]],
        wq_b_output: pl.InOut[pl.Tensor[[capacity, query_width], pl.BF16]],
        wkv_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        wkv_normed_output: pl.InOut[pl.Tensor[[capacity, head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        return qkv_w8a8_inline(
            value, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, cosine, sine, positions,
            query_latent, query, window_kv, wq_a_output, wq_b_output,
            wkv_output, wkv_normed_output, cache, cache_lens, active,
        )

    qkv_w8a8.inline = qkv_w8a8_inline

    return qkv_w8a8


__all__ = ["QKVShape", "make_qkv_w8a8", "qkv_reference", "qkv_shape"]
