"""INT8 index cache with FP16 per-token/head scales."""

from __future__ import annotations

import torch
import pypto.language as pl

from .config import CONTRACT


# ``npu_dynamic_quant`` is nearest rounding for ordinary values, but an exact
# half-way value is rounded toward zero on the A2/A3 kernel.  Keep the small
# bias in FP32 so the published FP16 scale remains unchanged.
_ROUND_TIE_EPS = 2.0 ** -20


def quantize_index_cache(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., 128]`` index vectors using FP16 dequant scales."""

    if value.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError(f"index cache width must be {CONTRACT.index_head_dim}")
    source = value.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    # vLLM-Ascend computes the INT8 payload with the unrounded FP32 scale and
    # only converts the published row scale to FP16.  Reusing the rounded
    # FP16 scale for the payload changes exact half-way cases (for example
    # -63.5) and causes a one-code mismatch with ``npu_dynamic_quant``.
    raw_scale = torch.maximum(
        amax / CONTRACT.int8_scale_max,
        torch.full_like(amax, CONTRACT.int8_amax_epsilon),
    )
    scale = torch.maximum(raw_scale.to(torch.float16), torch.full_like(raw_scale, 2.0 ** -24, dtype=torch.float16))
    normalized = source / raw_scale
    epsilon = torch.maximum(normalized.abs(), torch.ones_like(normalized)) * _ROUND_TIE_EPS
    # The A2/A3 kernel's observed tie behavior is asymmetric: positive
    # half-way values round down while negative values retain the normal rint
    # rule.  This only affects exact half-way cases.
    rounded = torch.where(
        normalized >= 0,
        torch.round(normalized - epsilon),
        torch.round(normalized),
    )
    payload = rounded.clamp(-127, 127).to(torch.int8)
    return payload, scale.squeeze(-1)


def dequantize_index_cache(payload: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return payload.float() * scale.float().unsqueeze(-1)


def make_index_cache_quantize(capacity: int, heads: int, head_dim: int = 128):
    if head_dim != CONTRACT.index_head_dim:
        raise ValueError("A2/A3 index cache uses 128-wide heads")

    @pl.jit.inline
    def quantize(
        value: pl.Tensor[[capacity, heads, head_dim], pl.BF16],
        payload: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        scales: pl.Tensor[[capacity, heads], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        source = pl.reshape(value, [capacity * heads, head_dim])
        output = pl.reshape(payload, [capacity * heads, head_dim])
        rows = (active * heads + 15) // 16
        with pl.spmd(rows, name_hint="dsv41_a2a3_index_cache_quant"):
            block = pl.tile.get_block_idx()
            row = block * 16
            valid_rows = pl.min(16, active * heads - row)
            tile = pl.slice(source, [16, head_dim], [row, 0], valid_shape=[valid_rows, head_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, head_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            amax = pl.row_max(pl.abs(value_fp32))
            raw_scale = pl.maximum(pl.div(amax, 127.0), 1.0e-30)
            rounded_scale = pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32)
            scale = pl.maximum(rounded_scale, 2.0 ** -24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            # Match the A2/A3 dynamic-quant tie rule: nearest for ordinary
            # values, half-way values toward zero.  Splitting signs keeps the
            # conversion in truncation mode, which is legal for INT32 output.
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), _ROUND_TIE_EPS)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            rounded = pl.cast(bounded, pl.INT8, mode="trunc")
            output[row:row + 16, :] = pl.set_validshape(rounded, valid_rows, head_dim)
            head = row % heads
            token = row // heads
            scale_row = pl.reshape(pl.cast(scale, pl.FP16), [1, 16])
            scales[token:token + 1, head:head + 16] = scale_row
        return payload, scales

    return quantize


__all__ = ["dequantize_index_cache", "make_index_cache_quantize", "quantize_index_cache"]
