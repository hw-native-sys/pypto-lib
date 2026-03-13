"""Linear-algebra tensor functions (matmul).

Matmul uses 3-D tiling over (M, N, K):
  - M, N: output row/col dimensions, tiled normally.
  - K: reduction (contraction) dimension, tiled with accumulation.

Pattern per (m, n) output tile:
    acc_tile = 0
    for k in range(0, K, TILE_K):
        a_tile = cast(view(a, [tile_m, tile_k], [m, k]))
        b_tile = cast(view(b, [tile_k, tile_n], [k, n]))
        acc_tile = pl.matmul_acc(a_tile, b_tile, acc_tile)
    assemble(output, cast_tile_to_tensor(acc_tile), [m, n])
"""

import pypto.language as pl

from .cast_primitives import cast_tensor_to_tile, cast_tile_to_tensor
from .tiling import (
    DEFAULT_MAT_TILE_K,
    DEFAULT_MAT_TILE_M,
    DEFAULT_MAT_TILE_N,
    compute_actual_size,
)


@pl.function
def tensor_matmul(
    a: pl.Tensor,
    b: pl.Tensor,
    output: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Matrix multiplication  output = a @ b  with M/N/K tiling.

    a: [M, K],  b: [K, N],  output: [M, N].
    """
    M, K_a = a.shape[0], a.shape[1]
    K_b, N = b.shape[0], b.shape[1]

    TILE_M = DEFAULT_MAT_TILE_M
    TILE_N = DEFAULT_MAT_TILE_N
    TILE_K = DEFAULT_MAT_TILE_K

    for m in pl.range(0, M, TILE_M):
        actual_m = compute_actual_size(M, m, TILE_M)
        for n in pl.range(0, N, TILE_N):
            actual_n = compute_actual_size(N, n, TILE_N)

            first_k = True
            for k in pl.range(0, K_a, TILE_K):
                actual_k = compute_actual_size(K_a, k, TILE_K)

                a_sub = pl.slice(a, [actual_m, actual_k], [m, k])
                b_sub = pl.slice(b, [actual_k, actual_n], [k, n])

                a_tile = cast_tensor_to_tile(a_sub)
                b_tile = cast_tensor_to_tile(b_sub)

                if first_k:
                    acc_tile = pl.matmul(a_tile, b_tile)
                    first_k = False
                else:
                    acc_tile = pl.matmul_acc(a_tile, b_tile, acc_tile)

            result_sub = cast_tile_to_tensor(acc_tile)
            pl.assemble(output, result_sub, [m, n])

    return output


@pl.function
def tensor_batch_matmul(
    a: pl.Tensor,
    b: pl.Tensor,
    output: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Batched matrix multiplication  output[i] = a[i] @ b[i].

    a: [B, M, K],  b: [B, K, N],  output: [B, M, N].
    """
    B = a.shape[0]
    M, K_a = a.shape[1], a.shape[2]
    N = b.shape[2]

    TILE_M = DEFAULT_MAT_TILE_M
    TILE_N = DEFAULT_MAT_TILE_N
    TILE_K = DEFAULT_MAT_TILE_K

    for bi in pl.range(0, B, 1):
        a_batch = pl.slice(a, [1, M, K_a], [bi, 0, 0])
        b_batch = pl.slice(b, [1, K_a, N], [bi, 0, 0])

        for m in pl.range(0, M, TILE_M):
            actual_m = compute_actual_size(M, m, TILE_M)
            for n in pl.range(0, N, TILE_N):
                actual_n = compute_actual_size(N, n, TILE_N)

                first_k = True
                for k in pl.range(0, K_a, TILE_K):
                    actual_k = compute_actual_size(K_a, k, TILE_K)

                    a_sub = pl.slice(a_batch, [actual_m, actual_k], [0, m, k])
                    b_sub = pl.slice(b_batch, [actual_k, actual_n], [0, k, n])

                    a_tile = cast_tensor_to_tile(a_sub)
                    b_tile = cast_tensor_to_tile(b_sub)

                    if first_k:
                        acc_tile = pl.matmul(a_tile, b_tile)
                        first_k = False
                    else:
                        acc_tile = pl.matmul_acc(a_tile, b_tile, acc_tile)

                result_sub = cast_tile_to_tensor(acc_tile)
                pl.assemble(output, result_sub, [bi, m, n])

    return output
