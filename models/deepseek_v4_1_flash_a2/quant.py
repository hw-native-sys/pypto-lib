"""Torch goldens and PyPTO entry points for A2/A3 W8A8 quantization."""

from __future__ import annotations

import torch
import pypto.language as pl

from .config import CONTRACT

INT8_AMAX_EPS = CONTRACT.int8_amax_epsilon
K_TILE = 256


def _round_symmetric_int8(value: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(value).clamp(-CONTRACT.int8_scale_max, CONTRACT.int8_scale_max)
    return rounded.to(torch.int32).to(torch.float16).to(torch.int8)


def quantize_per_token_int8(
    value: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row to INT8 and return an FP32 dequant scale.

    This mirrors ``torch_npu.npu_dynamic_quant`` used by
    ``AscendW8A8DynamicLinearMethod``.  The returned scale has a trailing
    singleton dimension and is suitable for broadcasting over the row.
    """

    source = value.float()
    if smooth_scale is not None:
        source = source * smooth_scale.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an output-major ``[N, K]`` weight with FP32 channel scales."""

    source = weight.float()
    amax = source.abs().amax(dim=-1, keepdim=True).clamp_min(CONTRACT.int8_amax_epsilon)
    dequant_scale = amax / CONTRACT.int8_scale_max
    payload = _round_symmetric_int8(source / dequant_scale)
    return payload, dequant_scale


def int8_matmul(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``[..., K] @ [N, K].T`` with an INT32 accumulator."""

    if value.shape[-1] != weight.shape[-1]:
        raise ValueError(f"K mismatch: activation={value.shape[-1]}, weight={weight.shape[-1]}")
    if value.shape[-1] % CONTRACT.int8_k_alignment:
        raise ValueError(
            f"A2/A3 INT8 GEMM requires K aligned to {CONTRACT.int8_k_alignment}, "
            f"got {value.shape[-1]}"
        )
    return torch.matmul(value.to(torch.int32), weight.to(torch.int32).transpose(-2, -1))


def dequantize_int32(
    accumulator: torch.Tensor,
    token_scale: torch.Tensor,
    channel_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Apply FP32 token/channel scales after INT32 accumulation."""

    result = accumulator.float() * token_scale.float()
    result = result * channel_scale.float().reshape(*([1] * (result.ndim - 1)), -1)
    return result.to(out_dtype)


def w8a8_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference for dynamic W8A8 linear on A2/A3."""

    payload, token_scale = quantize_per_token_int8(value, smooth_scale)
    accumulator = int8_matmul(payload, weight)
    result = dequantize_int32(accumulator, token_scale, weight_scale, out_dtype)
    if bias is not None:
        result = (result.float() + bias.float()).to(out_dtype)
    return result


def make_dynamic_int8_quantize(capacity: int, width: int):
    """Build a PyPTO kernel for BF16 to INT8 per-token quantization.

    The scale uses the vLLM/A2/A3 contract directly: one FP32 dequantization
    scale per token, stored as ``[T, 1]``.
    """

    if capacity % 8 or width % K_TILE:
        raise ValueError("capacity must be divisible by 8 and width by 256")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, width], pl.BF16],
        payload: pl.Tensor[[capacity, width], pl.INT8],
        scales: pl.Tensor[[capacity, 1], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(capacity // 8, name_hint="dsv41_a2a3_dynamic_quant"):
            t0 = block * 8
            maximum = pl.full([1, 8], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.range(0, width, K_TILE):
                source = pl.slice(value, [8, K_TILE], [t0, k0])
                maximum = pl.maximum(maximum, pl.reshape(pl.row_max(pl.abs(pl.cast(source, pl.FP32))), [1, 8]))
            inverse = pl.reshape(
                pl.div(pl.full([1, 8], dtype=pl.FP32, value=127.0), maximum), [8, 1]
            )
            scale = pl.reshape(pl.div(maximum, 127.0), [8, 1])
            scales[t0:t0 + 8, 0:1] = scale
            for k0 in pl.range(0, width, K_TILE):
                source_fp32 = pl.cast(pl.slice(value, [8, K_TILE], [t0, k0]), pl.FP32)
                normalized = pl.row_expand_mul(source_fp32, inverse)
                bounded = pl.minimum(pl.maximum(normalized, -127.0), 127.0)
                rounded = pl.cast(bounded, pl.INT32, mode="rint")
                payload[t0:t0 + 8, k0:k0 + K_TILE] = pl.cast(rounded, pl.INT8, mode="trunc")
        return payload, scales

    return quantize


__all__ = [
    "dequantize_int32",
    "int8_matmul",
    "make_dynamic_int8_quantize",
    "quantize_per_channel_int8",
    "quantize_per_token_int8",
    "w8a8_linear",
]
