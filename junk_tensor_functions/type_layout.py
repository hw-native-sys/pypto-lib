"""Type-conversion and layout tensor functions.

Some operations are pure metadata (reshape, view) and do not need
tile-level computation.  Others (cast, transpose, full, clone) require
a tiling loop with cast_tensor_to_tile / cast_tile_to_tensor.
"""

import pypto.language as pl

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N, compute_actual_size


# ---------------------------------------------------------------------------
# Metadata-only operations (no tiling needed)
# ---------------------------------------------------------------------------

@pl.function
def tensor_reshape(x: pl.Tensor, new_shape: list) -> pl.Tensor:
    """Reshape tensor (metadata-only, no data movement)."""
    return pl.reshape(x, new_shape)


@pl.function
def tensor_view(x: pl.Tensor, shape: list, offset: list) -> pl.Tensor:
    """Return a view (sub-tensor) of *x* (metadata-only)."""
    return pl.slice(x, shape, offset)


# ---------------------------------------------------------------------------
# Operations that require tiling
# ---------------------------------------------------------------------------

@pl.function
def tensor_cast(x: pl.Tensor, output: pl.Out[pl.Tensor],
                dtype=None) -> pl.Tensor:
    """Element-wise type cast with tiling."""
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            result_tile = pl.cast(x_tile, dtype)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


@pl.function
def tensor_transpose(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """2-D transpose with tiling.

    For a [M, N] input the output is [N, M].  Each tile [tile_m, tile_n]
    is transposed to [tile_n, tile_m] and assembled at the transposed offset.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            result_tile = pl.transpose(x_tile)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [c, r])

    return output


@pl.function
def tensor_full(output: pl.Out[pl.Tensor], value: float) -> pl.Tensor:
    """Fill *output* with a constant *value* using tiling."""
    M, N = output.shape[0], output.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            out_sub = pl.slice(output, [actual_m, actual_n], [r, c])
            out_tile = cast_tensor_to_tile(out_sub)

            result_tile = pl.expands(out_tile, value)

            result_sub = cast_tile_to_tensor(result_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


@pl.function
def tensor_clone(x: pl.Tensor, output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """Deep copy of *x* into *output* with tiling."""
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for r in pl.range(0, M, TILE_M):
        for c in pl.range(0, N, TILE_N):
            actual_m = compute_actual_size(M, r, TILE_M)
            actual_n = compute_actual_size(N, c, TILE_N)

            x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
            x_tile = cast_tensor_to_tile(x_sub)

            result_sub = cast_tile_to_tensor(x_tile)
            pl.assemble(output, result_sub, [r, c])

    return output


@pl.function
def tensor_concat(a: pl.Tensor, b: pl.Tensor,
                  output: pl.Out[pl.Tensor], axis: int = 0) -> pl.Tensor:
    """Concatenate *a* and *b* along *axis* into *output*.

    Simple implementation: copy a then b at the appropriate offset.
    """
    if axis == 0:
        M_a = a.shape[0]
        N = a.shape[1]
        TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

        for r in pl.range(0, M_a, TILE_M):
            for c in pl.range(0, N, TILE_N):
                actual_m = compute_actual_size(M_a, r, TILE_M)
                actual_n = compute_actual_size(N, c, TILE_N)
                a_sub = pl.slice(a, [actual_m, actual_n], [r, c])
                a_tile = cast_tensor_to_tile(a_sub)
                a_out = cast_tile_to_tensor(a_tile)
                pl.assemble(output, a_out, [r, c])

        M_b = b.shape[0]
        for r in pl.range(0, M_b, TILE_M):
            for c in pl.range(0, N, TILE_N):
                actual_m = compute_actual_size(M_b, r, TILE_M)
                actual_n = compute_actual_size(N, c, TILE_N)
                b_sub = pl.slice(b, [actual_m, actual_n], [r, c])
                b_tile = cast_tensor_to_tile(b_sub)
                b_out = cast_tile_to_tensor(b_tile)
                pl.assemble(output, b_out, [M_a + r, c])
    else:
        raise NotImplementedError("tensor_concat only supports axis=0 for now")

    return output


@pl.function
def tensor_expand_clone(x: pl.Tensor, output: pl.Out[pl.Tensor],
                        repeat_axis: int = 0,
                        repeats: int = 1) -> pl.Tensor:
    """Broadcast-expand *x* along *repeat_axis* and clone into *output*.

    TODO: generalise to arbitrary broadcast shapes.
    """
    M, N = x.shape[0], x.shape[1]
    TILE_M, TILE_N = DEFAULT_VEC_TILE_M, DEFAULT_VEC_TILE_N

    for rep in pl.range(0, repeats, 1):
        for r in pl.range(0, M, TILE_M):
            for c in pl.range(0, N, TILE_N):
                actual_m = compute_actual_size(M, r, TILE_M)
                actual_n = compute_actual_size(N, c, TILE_N)

                x_sub = pl.slice(x, [actual_m, actual_n], [r, c])
                x_tile = cast_tensor_to_tile(x_sub)
                result_sub = cast_tile_to_tensor(x_tile)

                if repeat_axis == 0:
                    pl.assemble(output, result_sub, [rep * M + r, c])
                else:
                    pl.assemble(output, result_sub, [r, rep * N + c])

    return output
