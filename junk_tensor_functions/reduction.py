"""Reduction tensor functions (sum, amax, amin along an axis).

Tiling strategy: the **reduction axis** is covered entirely (no tiling
along it), while the **non-reduction axis** is tiled normally.
For a 2-D tensor [M, N] with axis=-1 (reduce over N):
    tile loop over M-dimension only → each tile is [tile_m, N].
"""

import pypto.language as pl

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, compute_actual_size


# ---------------------------------------------------------------------------
# Row reductions (axis = -1, reduce over last dimension)
# ---------------------------------------------------------------------------

def _tiled_row_reduction(x, output, tile_reduce_op):
    """Reduce over the last axis with tiling along the first axis only."""
    M, N = x.shape[0], x.shape[1]
    TILE_M = DEFAULT_VEC_TILE_M

    for r in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, r, TILE_M)

        x_sub = pl.slice(x, [actual_m, N], [r, 0])
        x_tile = cast_tensor_to_tile(x_sub)

        result_tile = tile_reduce_op(x_tile)

        result_sub = cast_tile_to_tensor(result_tile)
        pl.assemble(output, result_sub, [r, 0])

    return output


@pl.function
def tensor_sum(x: pl.Tensor, output: pl.Out[pl.Tensor],
               axis: int = -1, keepdim: bool = True) -> pl.Tensor:
    """Sum reduction along *axis* (default: last axis).

    When axis=-1, uses ``pl.row_sum``; other axes may require transposition
    or col_sum (not yet generalised).
    """
    if axis == -1 or axis == len(x.shape) - 1:
        return _tiled_row_reduction(x, output, pl.row_sum)
    raise NotImplementedError("tensor_sum only supports axis=-1 for now")


@pl.function
def tensor_amax(x: pl.Tensor, output: pl.Out[pl.Tensor],
                axis: int = -1, keepdim: bool = True) -> pl.Tensor:
    """Max reduction along *axis*."""
    if axis == -1 or axis == len(x.shape) - 1:
        return _tiled_row_reduction(x, output, pl.row_max)
    raise NotImplementedError("tensor_amax only supports axis=-1 for now")


@pl.function
def tensor_amin(x: pl.Tensor, output: pl.Out[pl.Tensor],
                axis: int = -1, keepdim: bool = True) -> pl.Tensor:
    """Min reduction along *axis*."""
    if axis == -1 or axis == len(x.shape) - 1:
        return _tiled_row_reduction(x, output, pl.row_min)
    raise NotImplementedError("tensor_amin only supports axis=-1 for now")
