"""BF16 RMSNorm and adjacent-pair RoPE for A2/A3."""

from __future__ import annotations

import torch
import pypto.language as pl

from .operator_config import CONTRACT


def rms_norm_reference(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    source = value.float()
    inverse = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
    return (source * inverse * weight.float()).to(torch.bfloat16)


def rope_reference(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    heads: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim % 2 or value.shape[-1] != heads * head_dim:
        raise ValueError("RoPE dimensions must be even and match heads * head_dim")
    output = value.clone()
    nope = head_dim - rope_dim
    shaped = value.view(value.shape[0], heads, head_dim)
    tail = shaped[..., nope:]
    even, odd = tail[..., 0::2].float(), tail[..., 1::2].float()
    c, s = cosine.float().unsqueeze(1), sine.float().unsqueeze(1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)
    output.view(value.shape[0], heads, head_dim)[..., nope:] = rotated.to(torch.bfloat16)
    return output


def make_rmsnorm(capacity: int, width: int, eps: float = 1e-6):
    if capacity % CONTRACT.rmsnorm_m_tile:
        raise ValueError("capacity must be divisible by 8")

    @pl.jit.inline
    def norm(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd((active + 7) // 8, name_hint="dsv41_a2a3_rmsnorm"):
            t0 = block * 8
            rows = pl.min(8, active - t0)
            source = pl.slice(value, [8, width], [t0, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
            source_fp32 = pl.cast(source, pl.FP32)
            mean = pl.mul(pl.row_sum(pl.mul(source_fp32, source_fp32)), 1.0 / width)
            inverse = pl.reshape(pl.rsqrt(pl.add(mean, eps), high_precision=True), [8, 1])
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            normalized = pl.col_expand_mul(pl.row_expand_mul(source_fp32, inverse), gamma)
            output[t0:t0 + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, width)
        return output

    return norm


def make_rope(capacity: int, heads: int, head_dim: int, rope_dim: int):
    if rope_dim % 2 or capacity <= 0:
        raise ValueError("RoPE capacity and dimension must be positive and even")
    width = heads * head_dim
    nope = head_dim - rope_dim
    pairs = rope_dim // 2

    @pl.jit.inline
    def rope(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * heads, name_hint="dsv41_a2a3_rope"):
            token = block // heads
            base = (block % heads) * head_dim
            output[token:token + 1, base:base + nope] = value[token:token + 1, base:base + nope]
            tail = pl.cast(value[token:token + 1, base + nope:base + head_dim], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cosine[token:token + 1, :]
            s = sine[token:token + 1, :]
            real = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            imag = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[token:token + 1, base + nope:base + head_dim] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rope


__all__ = ["make_rmsnorm", "make_rope", "rms_norm_reference", "rope_reference"]
