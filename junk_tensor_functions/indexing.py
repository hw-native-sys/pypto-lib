"""Indexing tensor functions (gather, scatter, where).

These operations involve indirect addressing and are implemented with
tiling over the index/condition tensor dimensions.

Note: some tile-level operations (gather, scatter, sel) are not yet in
the pypto frontend's public API; they are referenced as pl.block.*
placeholders pending frontend support.
"""

import pypto.language as pl
from pypto.language import block as pl_block

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N, compute_actual_size


@pl.function
def tensor_gather(x: pl.Tensor, indices: pl.Tensor,
                  output: pl.Out[pl.Tensor],
                  axis: int = 0) -> pl.Tensor:
    """Gather elements from *x* along *axis* using *indices*.

    TODO: tile-level gather op not yet in pypto frontend;
    this is a structural placeholder showing the intended tiling pattern.
    """
    M, N = output.shape[0], output.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            idx_sub = pl.slice(indices, [actual_m, actual_n], [r, c])
            idx_tile = cast_tensor_to_tile(idx_sub)

            x_sub = pl.slice(x, [x.shape[0], actual_n], [0, c])
            x_tile = cast_tensor_to_tile(x_sub)

            # TODO: replace with actual tile-level gather when available
            result_tile = x_tile  # placeholder

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


@pl.function
def tensor_scatter_update(x: pl.Tensor, indices: pl.Tensor,
                          src: pl.Tensor,
                          output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Scatter *src* into *output* (initialised from *x*) at *indices*.

    TODO: tile-level scatter op not yet in pypto frontend;
    placeholder copies x and then overlays src at indices.
    """
    M_src, N_src = src.shape[0], src.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    M_x, N_x = x.shape[0], x.shape[1]
    for r in pl.range(0, M_x, TILE_M):
        for c in pl.range(0, N_x, TILE_N):
            actual_m = compute_actual_size(M_x, r, TILE_M)
            actual_n = compute_actual_size(N_x, c, TILE_N)
            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)
            x_out = cast_tile_to_tensor(x_tile)
            pl.assemble(output, x_out, [r, c])

    # TODO: tile-level scatter at index positions when frontend supports it
    for r in pl.range(0, M_src, TILE_M):
        for c in pl.range(0, N_src, TILE_N):
            actual_m = compute_actual_size(M_src, r, TILE_M)
            actual_n = compute_actual_size(N_src, c, TILE_N)

            src_sub = pl.slice(src, [actual_m, actual_n], [r, c])
            src_tile = cast_tensor_to_tile(src_sub)
            src_out = cast_tile_to_tensor(src_tile)
            # placeholder: writes to [r, c]; actual scatter needs index remapping
            pl.assemble(output, src_out, [r, c])

    return output


@pl.function
def tensor_where(condition: pl.Tensor, x: pl.Tensor, y: pl.Tensor,
                 output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Element-wise conditional selection: output = cond ? x : y.

    Uses tile-level comparison and select pattern:
    mask = cmp(cond, 0, "ne")  →  result = mask * x + (1-mask) * y.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            cond_sub = pl.slice(condition, [actual_m, actual_n], [r, c])
            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            y_sub = pl.slice(y, [actual_m, actual_n], [r, c])

            cond_tile = cast_tensor_to_tile(cond_sub)
            x_tile = cast_tensor_to_tile(x_sub)
            y_tile = cast_tensor_to_tile(y_sub)

            # cond != 0 → select x, else y
            mask = pl.cmps(cond_tile, 0, "ne")
            selected_x = pl.mul(mask, x_tile)
            inv_mask = pl.cmps(cond_tile, 0, "eq")
            selected_y = pl.mul(inv_mask, y_tile)
            result_tile = pl.add(selected_x, selected_y)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output
