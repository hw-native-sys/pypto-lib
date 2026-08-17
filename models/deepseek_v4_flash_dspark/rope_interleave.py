# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared RoPE cos/sin interleave-duplication for the DeepSeek-V4 Flash indexer.

The A3 interleaved rotation needs, per rope column ``j``:

    cos_il[j]     = cos_half[j >> 1]
    sin_signed[j] = sin_half[j >> 1] * sign[j],  sign = [-1, +1, -1, +1, ...]

so the forward rotation is ``out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j]`` and the
inverse (conjugate) rotation is the same expression with ``pl.sub``.

Building this per consumer block means re-running the ``j >> 1`` dup-gather on every
block. ``pl.gather`` lowers to a per-row ``TGATHER`` loop, so the cost scales with
(blocks x rows): the indexer's ``qr_rope`` alone spent 16 blocks x 32 rows x 2 tables
= 1024 row-gathers per layer rebuilding one small position-invariant table, plus 32
more in its compressor. This runs it once per layer over ``B_MAX`` rows instead.

Folding the sign into sin here rather than at each consumer is exact: multiplying by
+/-1 only flips the sign bit, so ``(x*sign)*sin`` and ``x*(sin*sign)`` are bit-identical.

Phase A (Run 055) keeps the existing D-Spark ``B_DYN``/``B_MAX`` consumer ABI
unchanged while adding a standalone active-row entry and device/golden harness.
The harness covers position 0, the 128 boundary, 12K, 16K and the 1M tail
without binding work to the maximum context capacity.
"""

import pypto.language as pl

from config import FLASH as M, DECODE_LOCAL_REQUESTS


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # runtime request count
ROWS_DYN = pl.dynamic("ROWS_DYN")  # standalone active token-local RoPE rows

# model config
B_MAX = DECODE_LOCAL_REQUESTS
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2

# tiling
B_TILE = 4  # rows per gather block; runtime B is a multiple of 4


@pl.jit.inline
def rope_interleave(
    cos_half: pl.Tensor[[B_DYN, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[B_DYN, HALF_ROPE], pl.FP32],
    cos_il: pl.Tensor[[B_MAX, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[B_MAX, ROPE_HEAD_DIM], pl.FP32],
):
    """Expand half-width cos/sin rows to the interleaved, sign-folded rope layout."""
    b_dim = pl.tensor.dim(cos_half, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_interleave", allow_early_resolve=True):
        il_ones = pl.full([B_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)                                    # j>>1
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))                                         # j%2
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)                                             # [-1,+1,...]
        # Rows [b_dim, B_MAX) of the scratch stay unwritten; no consumer reads them.
        for il_blk in pl.range(b_dim // B_TILE):
            il_b0 = il_blk * B_TILE
            cos_il[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = pl.gather(
                cos_half[il_b0 : il_b0 + B_TILE, 0:HALF_ROPE], dim=-1, index=il_dup_idx)
            sin_signed[il_b0 : il_b0 + B_TILE, 0:ROPE_HEAD_DIM] = pl.mul(
                pl.gather(sin_half[il_b0 : il_b0 + B_TILE, 0:HALF_ROPE], dim=-1, index=il_dup_idx), il_sign)

@pl.jit.inline
def _rope_interleave_active_body(
    cos_half: pl.Tensor[[ROWS_DYN, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[ROWS_DYN, HALF_ROPE], pl.FP32],
    cos_il: pl.Out[pl.Tensor[[ROWS_DYN, ROPE_HEAD_DIM], pl.FP32]],
    sin_signed: pl.Out[pl.Tensor[[ROWS_DYN, ROPE_HEAD_DIM], pl.FP32]],
    completion: pl.Array[1, pl.TASK_ID],
):
    """Inline body for :func:`rope_interleave_active`.

    The active row count is a ``pl.dynamic`` axis, so it cannot be passed to
    ``pl.full`` (which needs a compile-time ``ConstInt`` shape). Following the
    established dspark pattern, the column index/dup/sign tables are built once
    at a static tile granularity (``ROWS_TILE``) and applied per tile in a
    ``pl.range`` loop driven by the runtime row count. This keeps one compiled
    program valid for any active-row count. The loop is in the ``.inline`` body
    so it inlines into the ``@pl.jit`` wrapper, matching how
    ``build_decode_metadata`` inlines into ``decode_metadata``.
    """
    n_rows = pl.tensor.dim(cos_half, 0)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="rope_interleave_active",
        allow_early_resolve=True,
    ) as rope_tid:
        # Keep the row extent as a literal ConstInt.  A named local constant
        # becomes an SSA Var after this helper is nested-inline expanded into
        # the monolithic 43-layer program, and tile.full rejects Var extents.
        il_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for r in pl.range(n_rows):
            cos_il[r:r + 1, 0:ROPE_HEAD_DIM] = pl.gather(
                cos_half[r:r + 1, 0:HALF_ROPE], dim=-1, index=il_dup_idx)
            sin_signed[r:r + 1, 0:ROPE_HEAD_DIM] = pl.mul(
                pl.gather(sin_half[r:r + 1, 0:HALF_ROPE], dim=-1, index=il_dup_idx), il_sign)
    completion[0] = rope_tid
    return cos_il, sin_signed


@pl.jit
def rope_interleave_active(
    cos_half: pl.Tensor[[ROWS_DYN, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[ROWS_DYN, HALF_ROPE], pl.FP32],
    cos_il: pl.Out[pl.Tensor[[ROWS_DYN, ROPE_HEAD_DIM], pl.FP32]],
    sin_signed: pl.Out[pl.Tensor[[ROWS_DYN, ROPE_HEAD_DIM], pl.FP32]],
):
    """Active-row counterpart of :func:`rope_interleave`.

    The first dimension is the runtime active row count (number of token-local
    RoPE rows actually consumed this step), not a fixed ``DECODE_BATCH``.
    Used by the Phase A standalone harness to prove the work shape tracks the
    active set, not the 1M ceiling nor a fixed B.

    Thin ``@pl.jit`` compile entry that delegates to
    :func:`_rope_interleave_active_body` (``@pl.jit.inline``), mirroring how
    ``decode_metadata`` delegates to ``build_decode_metadata``.
    """
    completion = pl.array.create(1, pl.TASK_ID)
    _rope_interleave_active_body(
        cos_half, sin_half, cos_il, sin_signed, completion)
    return cos_il, sin_signed


# ---------------------------------------------------------------------------
# Phase A standalone device/golden harness (run_055 plan §5 A5). Covers
# position 0, 128 boundary, 12K, 16K, and the 1M tail (1M-2, 1M-1). The
# active row count is exactly the number of positions exercised, never
# padded to 1M nor fixed to DECODE_BATCH.
# ---------------------------------------------------------------------------

def _rope_positions_case(case: str) -> list[int]:
    if case == "position_boundaries":
        return [0, 127, 128, 129, 12287, 16383, 16384, M.max_position_embeddings - 2, M.max_position_embeddings - 1]
    raise ValueError(f"unknown rope_interleave case: {case!r}")


def _golden_rope_interleave(scratch):
    """Reference: interleave half-width cos/sin to full width with sign fold.

    Conforms to the golden-runner contract: read inputs from / write outputs
    back into the ``scratch`` dict keyed by the TensorSpec names
    (``cos_half``, ``sin_half`` -> ``cos_il``, ``sin_signed``).
    """
    import torch
    cos_half = scratch["cos_half"].to(torch.float32)
    sin_half = scratch["sin_half"].to(torch.float32)
    n, half = cos_half.shape
    full = 2 * half
    j = torch.arange(full, dtype=torch.float32)
    dup = (j // 2).to(torch.int64)
    sign = torch.where(j % 2 == 0, -1.0, 1.0).to(torch.float32)
    scratch["cos_il"] = torch.gather(cos_half, 1, dup.unsqueeze(0).expand(n, full)).contiguous()
    scratch["sin_signed"] = (torch.gather(sin_half, 1, dup.unsqueeze(0).expand(n, full)) * sign).contiguous()


def build_rope_interleave_specs(case: str):
    import torch
    from golden import TensorSpec
    from config import FLASH as CFG
    from utils import token_local_rope

    positions = _rope_positions_case(case)
    n = len(positions)
    dim = CFG.qk_rope_head_dim
    pos_t = torch.tensor(positions, dtype=torch.int32)
    cos_full, sin_full = token_local_rope(
        CFG, compress_ratio=0, position_ids=pos_t, rope_dim=dim,
    )
    half = dim // 2
    cos_half = cos_full[:, :half].contiguous()
    sin_half = sin_full[:, :half].contiguous()
    specs = [
        TensorSpec("cos_half", [n, half], torch.float32, init_value=cos_half),
        TensorSpec("sin_half", [n, half], torch.float32, init_value=sin_half),
        TensorSpec("cos_il", [n, dim], torch.float32, is_output=True),
        TensorSpec("sin_signed", [n, dim], torch.float32, is_output=True),
    ]
    return specs


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=["position_boundaries"], default="position_boundaries")
    args = parser.parse_args()

    result = run_jit(
        fn=rope_interleave_active,
        specs=build_rope_interleave_specs(args.case),
        golden_fn=_golden_rope_interleave,
        compile_only=args.compile_only,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
