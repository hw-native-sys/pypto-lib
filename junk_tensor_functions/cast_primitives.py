"""Cast primitives for Tensor ↔ Tile type conversion.

These are placeholder APIs for pure type casts between Tensor and Tile.
The pypto frontend does not yet implement these operations; they are
defined here so that tensor_functions can be written now and will work
once the frontend adds native support.

Semantics
---------
- cast_tensor_to_tile(tensor) -> Tile
    Pure type cast: the returned Tile has the **same shape** as the input
    Tensor.  No data movement, no size change.  The caller is responsible
    for extracting the desired sub-region via ``pl.slice()`` *before*
    calling this function.

    After the compiler establishes incore boundaries, a later pass will
    replace this cast with an actual ``tload`` at the correct location.

- cast_tile_to_tensor(tile) -> Tensor
    Pure type cast: the returned Tensor has the **same shape** as the input
    Tile.  No data movement, no size change.  The caller writes the result
    to the output tensor via ``pl.assemble()`` *after* this call.

    A later compiler pass will replace this cast with an actual ``tstore``.
"""

import pypto.language as pl


def cast_tensor_to_tile(tensor: pl.Tensor) -> pl.Tile:
    """Tensor → Tile (same shape, pure type cast, no data movement).

    Usage::

        x_sub = pl.slice(x, [tile_m, tile_n], [row, col])
        x_tile = cast_tensor_to_tile(x_sub)
        # ... tile-level operations on x_tile ...
    """
    # TODO: replace with pl.cast_tensor_to_tile when pypto frontend adds support
    raise NotImplementedError(
        "cast_tensor_to_tile is a placeholder; "
        "awaiting pypto frontend implementation"
    )


def cast_tile_to_tensor(tile: pl.Tile) -> pl.Tensor:
    """Tile → Tensor (same shape, pure type cast, no data movement).

    Usage::

        result_sub = cast_tile_to_tensor(result_tile)
        pl.assemble(output, result_sub, [row, col])
    """
    # TODO: replace with pl.cast_tile_to_tensor when pypto frontend adds support
    raise NotImplementedError(
        "cast_tile_to_tensor is a placeholder; "
        "awaiting pypto frontend implementation"
    )
