"""Memory-bounded staged kernels for a real-width A2/A3 L0 assembly.

Each entry point owns one stable decoder boundary.  Keeping the boundaries
separate lets the runtime reuse device buffers between projections and avoids
materializing packed V4.1 matrices in the per-core Mat buffer.
"""

from __future__ import annotations

from collections.abc import Mapping

import pypto.language as pl

from .gemm import make_int8_linear_direct
from .index_cache import make_index_cache_quantize
from .norm_rope import make_rope, make_rmsnorm
from .qkv import qkv_shape
from .swa_attention import make_swa_attention

ACT_TILE = 256


def make_linear(capacity: int, width: int, output_width: int):
    """Expose the direct canonical-weight W8A8 kernel."""

    return make_int8_linear_direct(capacity, width, output_width)


def make_bf16_linear(capacity: int, width: int, output_width: int):
    """BF16-input/FP32-accumulate linear for compressor and Engram paths."""

    if capacity % 16 or width % 256 or output_width % 32:
        raise ValueError("BF16 linear dimensions are not A2/A3 aligned")
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks,
                             name_hint="dsv41_a2a3_bf16_linear"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0:k0 + 256, n0:n0 + n_tile]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0:t0 + 16, n0:n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear


def make_fp32_linear(capacity: int, width: int, output_width: int):
    """BF16-input/FP32-weight linear used by the ratio-2 compressor."""

    if capacity % 16 or width % 256 or output_width % 32:
        raise ValueError("FP32 linear dimensions are not A2/A3 aligned")
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, output_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks,
                             name_hint="dsv41_a2a3_fp32_linear"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                rhs = weight[k0:k0 + 256, n0:n0 + n_tile]
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0:t0 + 16, n0:n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear


def make_norm(capacity: int, width: int, eps: float):
    norm = make_rmsnorm(capacity, width, eps)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        norm(value, weight, output, active)
        return output

    return kernel


def make_rope_kernel(capacity: int, heads: int, head_dim: int, rope_dim: int):
    rotate = make_rope(capacity, heads, head_dim, rope_dim)
    pairs = rope_dim // 2
    width = heads * head_dim

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, width], pl.BF16],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        rotate(value, cosine, sine, output, active)
        return output

    return kernel


def make_grouped_wo_a(capacity: int, heads: int, head_dim: int,
                      groups: int, rank: int):
    """Apply grouped BF16 Wo-A while retaining only one GEMM tile on core."""

    width = heads * head_dim
    if width % groups or (width // groups) % 256 or rank % 64:
        raise ValueError("Wo-A dimensions are not A2/A3 aligned")
    group_input = width // groups
    group_blocks = rank // 64
    input_blocks = group_input // 256

    @pl.jit
    def project(
        rotated: pl.Tensor[[capacity, width], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        grouped: pl.InOut[pl.Tensor[[capacity, groups * rank], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * groups * group_blocks,
                             name_hint="dsv41_a2a3_grouped_wo_a"):
            token = (block // (groups * group_blocks)) * 16
            group = (block % (groups * group_blocks)) // group_blocks
            n0 = (block % group_blocks) * 64
            rows = pl.min(16, active - token)
            acc = pl.create_tensor([16, 64], dtype=pl.FP32)
            for k0 in pl.pipeline(0, group_input, 256, stage=2):
                source = pl.slice(
                    rotated, [16, 256], [token, group * group_input + k0],
                    valid_shape=[rows, 256],
                )
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 16, 256)
                weight = wo_a[group:group + 1, k0:k0 + 256, n0:n0 + 64]
                weight = pl.reshape(weight, [256, 64])
                acc = pl.matmul_acc(acc, source, weight, init_cond=(k0 == 0))
            grouped[token:token + 16, group * rank + n0:group * rank + n0 + 64] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, 64
            )
        return grouped

    return project


def make_swiglu(capacity: int, intermediate: int, clamp_limit: float = 10.0):
    """Evaluate the BF16 SwiGLU boundary in small vector tiles."""

    if intermediate % 256:
        raise ValueError("SwiGLU intermediate width must be divisible by 256")

    @pl.jit
    def activation(
        gate_up: pl.Tensor[[capacity, intermediate * 2], pl.BF16],
        output: pl.InOut[pl.Tensor[[capacity, intermediate], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        tiles = intermediate // 256
        for block in pl.spmd(((active + 7) // 8) * tiles,
                             name_hint="dsv41_a2a3_swiglu_v2"):
            token = (block // tiles) * 8
            column = (block % tiles) * 256
            rows = pl.min(8, active - token)
            # Keep the full vector tile in the traced arithmetic.  On A2/A3
            # a partial ``valid_shape`` changes the elementwise lowering and
            # can perturb the last valid row's BF16 rounding.
            gate = pl.cast(pl.slice(gate_up, [8, 256], [token, column]), pl.FP32)
            up = pl.cast(pl.slice(gate_up, [8, 256], [token, intermediate + column]), pl.FP32)
            gate = pl.minimum(gate, clamp_limit)
            up = pl.maximum(pl.minimum(up, clamp_limit), -clamp_limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
            silu = pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32))
            silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
            activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
            output[token:token + 8, column:column + 256] = pl.set_validshape(
                pl.cast(activated, pl.BF16, mode="rint"), rows, 256
            )
        return output

    return activation


def make_index_quantize(capacity: int, heads: int, head_dim: int = 128):
    quantize = make_index_cache_quantize(capacity, heads, head_dim)

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


def make_swa(capacity: int, config: Mapping[str, object]):
    shape = qkv_shape(config)
    window = int(config["sliding_window"])
    attend = make_swa_attention(capacity, shape.num_attention_heads, shape.head_dim, window)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, shape.query_width], pl.BF16],
        window_kv: pl.Tensor[[capacity, shape.head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[shape.num_attention_heads], pl.FP32],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, shape.head_dim], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, shape.query_width], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        attend(query, window_kv, slots, indices, sink, cache, output, active)
        return cache, output

    return kernel


__all__ = [
    "make_grouped_wo_a",
    "make_bf16_linear",
    "make_fp32_linear",
    "make_index_quantize",
    "make_linear",
    "make_norm",
    "make_rope_kernel",
    "make_swa",
    "make_swiglu",
]
