"""Elementwise unary tensor functions.

Each function implements: tiling loop → view → cast_tensor_to_tile →
tile-level unary op → cast_tile_to_tensor → assemble.
"""

import pypto.language as pl
from pypto.language import block as pl_block

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N, compute_actual_size


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _tiled_unary_op(x, output, tile_op):
    """Apply *tile_op* element-wise over *x* with 2-D tiling."""
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            result_tile = tile_op(x_tile)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


# ---------------------------------------------------------------------------
# Public tensor functions
# ---------------------------------------------------------------------------

@pl.function
def tensor_exp(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise exponential."""
    return _tiled_unary_op(x, output, pl.exp)


@pl.function
def tensor_sqrt(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise square root."""
    return _tiled_unary_op(x, output, pl.sqrt)


@pl.function
def tensor_rsqrt(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise reciprocal square root."""
    return _tiled_unary_op(x, output, pl.rsqrt)


@pl.function
def tensor_abs(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise absolute value."""
    return _tiled_unary_op(x, output, pl.abs)


@pl.function
def tensor_neg(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise negation."""
    return _tiled_unary_op(x, output, pl.neg)


@pl.function
def tensor_log(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise natural logarithm."""
    return _tiled_unary_op(x, output, pl.log)


@pl.function
def tensor_relu(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise ReLU."""
    return _tiled_unary_op(x, output, pl.relu)


@pl.function
def tensor_recip(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise reciprocal (1/x)."""
    return _tiled_unary_op(x, output, pl.recip)


@pl.function
def tensor_sigmoid(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise sigmoid: 1 / (1 + exp(-x)).

    Composed from tile-level primitives: neg → exp → adds(1) → recip.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            neg_tile = pl.neg(x_tile)
            exp_tile = pl.exp(neg_tile)
            one_plus = pl_block.adds(exp_tile, 1.0)
            result_tile = pl.recip(one_plus)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


@pl.function
def tensor_logical_not(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise logical NOT.

    TODO: pl.not is not yet in the frontend; uses cmps(tile, 0) as placeholder.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            # logical_not: compare equal to zero
            result_tile = pl.cmps(x_tile, 0, "eq")

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output
