"""Composite tensor functions (softmax, rmsnorm, gelu, etc.).

Each composite is implemented as a **single tiling loop** whose body
contains multiple tile-level operations — this is "manual fusion":
all tile ops for one tile run before moving to the next tile.
"""

import pypto.language as pl
from pypto.language import block as pl_block

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N, compute_actual_size


# ===================================================================
# softmax
# ===================================================================

@pl.function
def tensor_softmax(x: pl.Tensor,
                   output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Softmax over the last axis (axis=-1).

    Manual-fused tile body: row_max → sub → exp → row_sum → row_expand_div.
    Tiling only over the M (row) dimension; the N (column / reduction) axis
    is covered entirely per tile.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        x_tile = cast_tensor_to_tile(x_sub)

        max_tile = pl.row_max(x_tile)
        shifted = pl.row_expand_sub(x_tile, max_tile)
        exp_tile = pl.exp(shifted)
        sum_tile = pl.row_sum(exp_tile)
        out_tile = pl.row_expand_div(exp_tile, sum_tile)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# rmsnorm
# ===================================================================

@pl.function
def tensor_rmsnorm(x: pl.Tensor, weight: pl.Tensor,
                   output: pl.Out[pl.Tensor],
                   eps: float = 1e-6) -> pl.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight.

    Manual-fused tile body: mul(sq) → row_sum → muls(1/N) → adds(eps)
    → rsqrt → row_expand_mul(x) → mul(weight).
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M
    inv_n = 1.0 / N

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        w_sub = pl.slice(weight, [1, N], [0, 0])

        x_tile = cast_tensor_to_tile(x_sub)
        w_tile = cast_tensor_to_tile(w_sub)

        sq_tile = pl.mul(x_tile, x_tile)
        sum_tile = pl.row_sum(sq_tile)
        mean_tile = pl_block.muls(sum_tile, inv_n)
        eps_tile = pl_block.adds(mean_tile, eps)
        rsqrt_tile = pl.rsqrt(eps_tile)
        normed = pl.row_expand_mul(x_tile, rsqrt_tile)
        out_tile = pl.mul(normed, w_tile)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# layernorm
# ===================================================================

@pl.function
def tensor_layernorm(x: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor,
                     output: pl.Out[pl.Tensor],
                     eps: float = 1e-5) -> pl.Tensor:
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias."""
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M
    inv_n = 1.0 / N

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        w_sub = pl.slice(weight, [1, N], [0, 0])
        b_sub = pl.slice(bias, [1, N], [0, 0])

        x_tile = cast_tensor_to_tile(x_sub)
        w_tile = cast_tensor_to_tile(w_sub)
        b_tile = cast_tensor_to_tile(b_sub)

        # mean
        sum_tile = pl.row_sum(x_tile)
        mean_tile = pl_block.muls(sum_tile, inv_n)

        # x - mean
        centered = pl.row_expand_sub(x_tile, mean_tile)

        # variance
        sq_tile = pl.mul(centered, centered)
        var_sum = pl.row_sum(sq_tile)
        var_tile = pl_block.muls(var_sum, inv_n)
        var_eps = pl_block.adds(var_tile, eps)
        rstd = pl.rsqrt(var_eps)

        # normalise
        normed = pl.row_expand_mul(centered, rstd)
        scaled = pl.mul(normed, w_tile)
        out_tile = pl.add(scaled, b_tile)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# rope (Rotary Position Embedding)
# ===================================================================

@pl.function
def tensor_rope(q: pl.Tensor, k: pl.Tensor,
                cos: pl.Tensor, sin: pl.Tensor,
                q_out: pl.Out[pl.Tensor],
                k_out: pl.Out[pl.Tensor]) -> tuple:
    """Apply Rotary Position Embedding to query and key.

    For each row:  q_out = q * cos + rotate_half(q) * sin
    rotate_half splits the last dim in half, negates the first half,
    and swaps halves.

    Simplified 2-D implementation: [seq_len, head_dim].
    """
    M, N = q.shape[0], q.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M
    half_n = N // 2

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        q_sub = pl.slice(q, [actual_m, N], [r, 0])
        k_sub = pl.slice(k, [actual_m, N], [r, 0])
        cos_sub = pl.slice(cos, [actual_m, N], [r, 0])
        sin_sub = pl.slice(sin, [actual_m, N], [r, 0])

        q_tile = cast_tensor_to_tile(q_sub)
        k_tile = cast_tensor_to_tile(k_sub)
        cos_tile = cast_tensor_to_tile(cos_sub)
        sin_tile = cast_tensor_to_tile(sin_sub)

        # rotate_half for q
        q_first = pl.slice(q_tile, [actual_m, half_n], [0, 0])
        q_second = pl.slice(q_tile, [actual_m, half_n], [0, half_n])
        q_rot_first = pl.neg(q_second)
        # TODO: concat q_rot_first and q_first to form rotated q
        # Placeholder: q_rotated = concat(neg(q[..., N//2:]), q[..., :N//2])

        # q_out = q * cos + q_rotated * sin
        qcos = pl.mul(q_tile, cos_tile)
        # qrot_sin = pl.mul(q_rotated, sin_tile)
        # q_result = pl.add(qcos, qrot_sin)

        q_result = qcos  # simplified placeholder

        q_out_sub = cast_tile_to_tensor(q_result)
        pl.assemble(q_out, q_out_sub, [r, 0])

        # Same for k
        kcos = pl.mul(k_tile, cos_tile)
        k_result = kcos  # simplified placeholder

        k_out_sub = cast_tile_to_tensor(k_result)
        pl.assemble(k_out, k_out_sub, [r, 0])

    return q_out, k_out


# ===================================================================
# swiglu
# ===================================================================

@pl.function
def tensor_swiglu(x: pl.Tensor,
                  output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """SwiGLU activation: sigmoid(x1) * x2.

    Input x has shape [M, 2*D]; split into x1=[M,D] and x2=[M,D].
    """
    M = x.shape[0]
    D = x.shape[1] // 2
    TILE_M = DEFAULT_VEC_TILE_M

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x1_sub = pl.slice(x, [actual_m, D], [r, 0])
        x2_sub = pl.slice(x, [actual_m, D], [r, D])

        x1_tile = cast_tensor_to_tile(x1_sub)
        x2_tile = cast_tensor_to_tile(x2_sub)

        # sigmoid(x1) = 1 / (1 + exp(-x1))
        neg_x1 = pl.neg(x1_tile)
        exp_neg = pl.exp(neg_x1)
        one_plus = pl_block.adds(exp_neg, 1.0)
        sig_tile = pl.recip(one_plus)

        out_tile = pl.mul(sig_tile, x2_tile)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# gelu
# ===================================================================

@pl.function
def tensor_gelu(x: pl.Tensor,
                output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """GELU activation (sigmoid approximation): x * sigmoid(1.702 * x)."""
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M
    GELU_COEFF = 1.702

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        x_tile = cast_tensor_to_tile(x_sub)

        scaled = pl_block.muls(x_tile, GELU_COEFF)
        neg_scaled = pl.neg(scaled)
        exp_neg = pl.exp(neg_scaled)
        one_plus = pl_block.adds(exp_neg, 1.0)
        sig = pl.recip(one_plus)
        out_tile = pl.mul(x_tile, sig)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# symmetric quantization
# ===================================================================

@pl.function
def tensor_symmetric_quant(x: pl.Tensor,
                           output: pl.Out[pl.Tensor],
                           scale_out: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Per-token symmetric quantization.

    scale = amax(abs(x), axis=-1) / 127
    output = cast_to_int8(x / scale)
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M
    inv_127 = 1.0 / 127.0

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        x_tile = cast_tensor_to_tile(x_sub)

        abs_tile = pl.abs(x_tile)
        amax_tile = pl.row_max(abs_tile)
        scale_tile = pl_block.muls(amax_tile, inv_127)

        # store scale
        scale_sub = cast_tile_to_tensor(scale_tile)
        pl.assemble(scale_out, scale_sub, [r, 0])

        # quantise
        quant_tile = pl.row_expand_div(x_tile, scale_tile)
        quant_int = pl.cast(quant_tile, pl.INT8)

        quant_sub = cast_tile_to_tensor(quant_int)
        pl.assemble(output, quant_sub, [r, 0])

    return output


# ===================================================================
# dequantization
# ===================================================================

@pl.function
def tensor_dequant(x: pl.Tensor, scale: pl.Tensor,
                   output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Dequantize: output = cast_to_fp(x) * scale."""
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        s_sub = pl.slice(scale, [actual_m, 1], [r, 0])

        x_tile = cast_tensor_to_tile(x_sub)
        s_tile = cast_tensor_to_tile(s_sub)

        fp_tile = pl.cast(x_tile, pl.FP32)
        out_tile = pl.row_expand_mul(fp_tile, s_tile)

        out_sub = cast_tile_to_tensor(out_tile)
        pl.assemble(output, out_sub, [r, 0])

    return output


# ===================================================================
# topk (placeholder)
# ===================================================================

@pl.function
def tensor_topk(x: pl.Tensor,
                values_out: pl.Out[pl.Tensor],
                indices_out: pl.Out[pl.Tensor],
                k: int = 1, axis: int = -1) -> tuple:
    """Top-K selection along *axis*.

    TODO: full tile-level implementation using topk_sort / topk_merge /
    topk_extract once those ops are available in the frontend.
    """
    raise NotImplementedError(
        "tensor_topk requires tile-level topk ops not yet in pypto frontend"
    )
