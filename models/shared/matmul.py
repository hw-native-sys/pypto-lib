# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared matmul tiling primitives: ``@pl.inline`` with CT kwargs.

Two variants:

- ``matmul_tiled`` — 2D tiled matmul: caller passes the **full** activation
  and weight tensors plus a **dynamic** N-dimension offset. The inline owns
  all K-loop slicing from the full tensors. The caller owns the N-tile outer
  loop (SPMD/parallel dispatch) and ``pl.assemble``.

  Use case: Q, K, V, output, gate, up, down projections.

- ``matmul_tiled_4d`` — 4D block-major variant for ``32b_decode_4d.py``.
  Same principle: caller passes full 4D tensors plus a dynamic N1 offset.

Both use ``@pl.inline`` with keyword-only CT kwargs (``k_chunk``, ``n_chunk``,
``k_blocks``) for compile‑time tile sizes, and a **positional** offset parameter
for the dynamic N‑dimension (which is computed from SPMD/parallel loop vars).

The caller MUST capture the return value — the body is scope-isolated (variables
do not leak to the caller).

See ``~/workspace/subprojects/pypto/docs/guide-ct-args.md`` for the full
context on ``@pl.inline`` + CT kwargs (in the compiler subproject, not here).
"""

import pypto.language as pl


@pl.inline
def matmul_tiled(
    a, b, n_offset,
    *,
    m: int,
    k_chunk: int,
    n_chunk: int,
    k_blocks: int,
    stages: int = 2,
):
    """Tiled matmul with pipelined K-accumulation.

    Caller passes full tensors ``a`` (activation) and ``b`` (weight) plus a
    dynamic ``n_offset`` (the N‑tile start, e.g. ``qi * N_CHUNK`` from an
    SPMD/parallel loop).  The inline owns the K‑loop slicing internally,
    generating per‑iteration Mat‑space tiles that fit within 512 KB.

    Args:
        a: Activation tensor ``[M, K]`` — **full** K‑dimension.  No pre-slicing.
        b: Weight tensor ``[K, N]`` — **full** K‑dimension.  No pre-slicing.
        n_offset: Dynamic N‑dimension offset into ``b`` (a Scalar, e.g. from
            a ``pl.spmd`` iter var).  Not a CT kwarg — evaluated at runtime.
        m: Row (batch) dimension of ``a`` (CT kwarg, must match ``a.shape[0]``).
        k_chunk: K‑dimension chunk size for the reduction pipeline (CT kwarg).
        n_chunk: N‑dimension chunk size for the output tile (CT kwarg).
        k_blocks: Number of K-blocks (``K // k_chunk``).  CT kwarg.
        stages: Software pipeline depth for the K-loop (default 2).

    Returns:
        Accumulated output ``[M, n_chunk]`` (FP32).  Caller MUST capture.
    """
    acc = pl.create_tensor([m, n_chunk], dtype=pl.FP32)
    for kb in pl.pipeline(0, k_blocks, stage=stages):
        k0 = kb * k_chunk
        tile_a = a[:, k0:k0 + k_chunk]
        tile_b = b[k0:k0 + k_chunk, n_offset:n_offset + n_chunk]
        if kb == 0:
            acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
        else:
            acc = pl.matmul_acc(acc, tile_a, tile_b)
    return acc


@pl.inline
def matmul_tiled_4d(
    a, b, n1_offset,
    *,
    batch: int,
    k_chunk: int,
    n1_chunk: int,
    k_blocks: int,
    stages: int = 2,
):
    """4D block-major tiled matmul for ``32b_qwen3_32b_decode_4d.py``.

    Peeled-first K-accumulation: first iteration (kb=0) runs ``pl.matmul``,
    then iterations 1..k_blocks-1 run ``pl.matmul_acc``.  This matches the
    proven pattern from the original hand-written 4D code.

    Args:
        a: Activation tensor ``[k_blocks, 1, batch, k_chunk]`` — full.
        b: Weight tensor ``[k_blocks, n1_blocks, k_chunk, n1_chunk]`` — full.
        n1_offset: Dynamic offset into the second axis of ``b``.
        batch: Batch dimension of the tensors (CT kwarg, must match ``a.shape[2]``).
        k_chunk: K‑dimension chunk size (4th axis of the 4D tensors, CT kwarg).
        n1_chunk: Second-axis chunk size (CT kwarg).
        k_blocks: Number of K-blocks (CT kwarg, fixed at parse time).
        stages: Software pipeline depth for the K-loop.

    Returns:
        Accumulated output ``[1, 1, batch, n1_chunk]`` (FP32).
    """
    tile_a = a[0:1, :, :, :]
    tile_b = b[0:1, n1_offset:n1_offset + 1, :, :]
    acc = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32)
    for kb in pl.pipeline(1, k_blocks, stage=stages):
        tile_a = a[kb:kb + 1, :, :, :]
        tile_b = b[kb:kb + 1, n1_offset:n1_offset + 1, :, :]
        acc = pl.matmul_acc(acc, tile_a, tile_b)
    return acc
