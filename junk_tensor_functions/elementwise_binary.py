"""Elementwise binary tensor functions.

Each function implements: tiling loop → view → cast_tensor_to_tile →
tile-level binary op → cast_tile_to_tensor → assemble.
"""

import pypto.language as pl

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N, compute_actual_size


# ---------------------------------------------------------------------------
# Internal helper — generic 2-D tiled binary operation
# ---------------------------------------------------------------------------

def _tiled_binary_op(x, y, output, tile_op):
    """Apply *tile_op* element-wise over x and y with 2-D tiling.

    Parameters
    ----------
    x, y : pl.Tensor — input tensors (same shape or broadcastable).
    output : pl.Tensor — pre-allocated output tensor.
    tile_op : callable(tile_a, tile_b) -> tile — tile-level binary op.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            y_sub = pl.slice(y, [actual_m, actual_n], [r, c])

            x_tile = cast_tensor_to_tile(x_sub)
            y_tile = cast_tensor_to_tile(y_sub)

            result_tile = tile_op(x_tile, y_tile)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


# ---------------------------------------------------------------------------
# Public tensor functions
# ---------------------------------------------------------------------------

@pl.function
def tensor_add(x: pl.Tensor, y: pl.Tensor,
               output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise addition with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.add)


@pl.function
def tensor_sub(x: pl.Tensor, y: pl.Tensor,
               output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise subtraction with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.sub)


@pl.function
def tensor_mul(x: pl.Tensor, y: pl.Tensor,
               output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise multiplication with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.mul)


@pl.function
def tensor_div(x: pl.Tensor, y: pl.Tensor,
               output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise division with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.div)


@pl.function
def tensor_maximum(x: pl.Tensor, y: pl.Tensor,
                   output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise maximum with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.maximum)


@pl.function
def tensor_minimum(x: pl.Tensor, y: pl.Tensor,
                   output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise minimum with 2-D tiling."""
    return _tiled_binary_op(x, y, output, pl.minimum)
