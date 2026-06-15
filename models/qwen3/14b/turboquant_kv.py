# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TurboQuant KV compression operators for Qwen3-14B.

Contains the Lloyd-Max codebook computation, the prefill KV
quantization function, and the QJL (Algorithm 2) K residual
quantization for inner product preservation.

Typical call chain (nested jit.inline):
  prefill_fwd_tq (@pl.jit)
    └── prefill_layer_tq (@pl.jit.inline)
          ├── turboquant_kv_prefill (@pl.jit.inline)    ← PolarQuant
          └── turboquant_qjl_k     (@pl.jit.inline)    ← QJL (K only)
"""
import math as _math

import pypto.language as pl

from config import (
    EPS,
    HALF_DIM,
    HEAD_DIM,
    KV_HIDDEN,
    NUM_KV_HEADS,
)

# ---------------------------------------------------------------------------
# TQ constants
# ---------------------------------------------------------------------------
N_LEVELS = 16  # int4 -> 16 levels

# ---------------------------------------------------------------------------
# Prefill-specific tiling
# ---------------------------------------------------------------------------
TOK_TILE = 16  # Token block size for prefill
SEQ_TILE = 128  # Sequence tile for attention (used by QJL codebook gather)
CMP_TILE = 64  # Tile for K fused dequant (reduced for scope-separated dequant+matmul)
CMP_TILE_SV = 64  # Smaller tile for V fused dequant (Vec buffer constraint)
CMP_CHUNK = 32  # Sub-chunk for gather: 32 rows * 1B (UINT8) = 32-byte aligned

# ---------------------------------------------------------------------------
# Dynamic dims
# ---------------------------------------------------------------------------
QUANT_CACHE_ROWS_DYN = pl.dynamic("QUANT_CACHE_ROWS_DYN")
SCALES_ROWS_DYN = pl.dynamic("SCALES_ROWS_DYN")

# ---------------------------------------------------------------------------
# Lloyd-Max codebook (computed once at module load)
# ---------------------------------------------------------------------------
from lloyd_max import solve_lloyd_max

_lm_centroids, _lm_boundaries = solve_lloyd_max(
    HEAD_DIM, min(int(_math.log2(N_LEVELS)), 4),
)
_b0, _b1, _b2, _b3, _b4, _b5, _b6, _b7 = [
    float(x) for x in _lm_boundaries[:8]
]
_b8, _b9, _b10, _b11, _b12, _b13, _b14 = [
    float(x) for x in _lm_boundaries[8:]
]


# ---------------------------------------------------------------------------
# Prefill: batched K/V RoPE + L2 norm + normalize + rotate + Lloyd-Max quant
# ---------------------------------------------------------------------------


@pl.jit.inline
def turboquant_kv_prefill(
    # Inputs (produced by Scope 1 in caller).
    k_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    v_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    rot_matrix: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    cos_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    cos_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    # Outputs (caller creates, callee fills via pl.assemble).
    quant_k_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.UINT8],
    quant_v_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.UINT8],
    k_scales_buf: pl.Tensor[[TOK_TILE, NUM_KV_HEADS], pl.FP32],
    v_scales_buf: pl.Tensor[[TOK_TILE, NUM_KV_HEADS], pl.FP32],
):
    """Batched K/V RoPE + inline quantization for prefill.

    For each KV head:
      Scope A: K RoPE + L2 norm + normalize  -> k_norm_buf, k_scales_buf
      Scope B: K rotate + Lloyd-Max quantize  -> quant_k_temp, k_rot_buf
      Scope C: V L2 norm + normalize          -> v_norm_buf, v_scales_buf
      Scope D: V rotate + Lloyd-Max quantize  -> quant_v_temp
    """
    # Internal intermediates (not exposed to caller).
    k_norm_buf = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.BF16)
    v_norm_buf = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.BF16)

    for ki_chunk in pl.parallel(0, NUM_KV_HEADS, 8):
        # Scope A: K RoPE + L2 norm + normalize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_k_norm"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM

                k_lo = pl.slice(k_proj_tile, [TOK_TILE, HALF_DIM], [0, kv_col])
                k_hi = pl.slice(k_proj_tile, [TOK_TILE, HALF_DIM],
                                [0, kv_col + HALF_DIM])
                rot_lo = pl.sub(
                    pl.mul(k_lo, cos_lo_all),
                    pl.mul(k_hi, sin_lo_all),
                )
                rot_hi = pl.add(
                    pl.mul(k_hi, cos_hi_all),
                    pl.mul(k_lo, sin_hi_all),
                )

                sq_lo = pl.reshape(
                    pl.row_sum(pl.mul(rot_lo, rot_lo)), [TOK_TILE, 1],
                )
                sq_hi = pl.reshape(
                    pl.row_sum(pl.mul(rot_hi, rot_hi)), [TOK_TILE, 1],
                )
                k_sq_sum = pl.add(sq_lo, sq_hi)
                k_l2_norm = pl.sqrt(pl.add(k_sq_sum, EPS))

                inv_norm = pl.recip(k_l2_norm)
                k_norm_lo = pl.row_expand_mul(rot_lo, inv_norm)
                k_norm_hi = pl.row_expand_mul(rot_hi, inv_norm)
                k_norm_lo_bf16 = pl.cast(k_norm_lo, target_type=pl.BF16)
                k_norm_hi_bf16 = pl.cast(k_norm_hi, target_type=pl.BF16)
                k_norm_buf = pl.assemble(
                    k_norm_buf, k_norm_lo_bf16, [0, kv_col],
                )
                k_norm_buf = pl.assemble(
                    k_norm_buf, k_norm_hi_bf16, [0, kv_col + HALF_DIM],
                )

                # Per-row write: k_l2_norm is [TOK_TILE, 1] (ColMajor after
                # reshape), pl.assemble to a narrow column in a RowMajor
                # buffer only fills row 0.  Flatten and write per-row instead.
                k_l2_norm_flat = pl.reshape(k_l2_norm, [TOK_TILE])
                for _ti in pl.range(TOK_TILE):
                    k_scales_buf = pl.write(k_scales_buf, [_ti, ki],
                                            pl.read(k_l2_norm_flat, [_ti]))

        # Scope B: K rotate + quantize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_rotate_quant"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM
                k_norm_bf16 = pl.slice(
                    k_norm_buf, [TOK_TILE, HEAD_DIM], [0, kv_col],
                )
                k_rot = pl.matmul(k_norm_bf16, rot_matrix, out_dtype=pl.FP32)

                k_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b0), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b1), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b2), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b3), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b4), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b5), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b6), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b7), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b8), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b9), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b10), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b11), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b12), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b13), cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b14), cmp_type=5))
                # FP32 -> INT32 -> FP16 -> UINT8 (pypto cast 4->1 byte workaround)
                k_idx_i32 = pl.cast(k_idx, pl.INT32, mode="trunc")
                k_idx_f16 = pl.cast(k_idx_i32, pl.FP16, mode="round")
                qk_lo = pl.cast(k_idx_f16, pl.UINT8, mode="trunc")
                quant_k_temp = pl.assemble(quant_k_temp, qk_lo, [0, kv_col])

        # Scope C: V L2 norm + normalize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_norm"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM

                v_all = pl.slice(v_proj_tile, [TOK_TILE, HEAD_DIM],
                                 [0, kv_col])
                v_sq_sum = pl.reshape(
                    pl.row_sum(pl.mul(v_all, v_all)), [TOK_TILE, 1],
                )
                v_l2_norm = pl.sqrt(pl.add(v_sq_sum, EPS))
                v_norm = pl.row_expand_mul(v_all, pl.recip(v_l2_norm))
                v_norm_bf16 = pl.cast(v_norm, target_type=pl.BF16)
                v_norm_buf = pl.assemble(v_norm_buf, v_norm_bf16, [0, kv_col])

                v_l2_norm_flat = pl.reshape(v_l2_norm, [TOK_TILE])
                for _ti in pl.range(TOK_TILE):
                    v_scales_buf = pl.write(v_scales_buf, [_ti, ki],
                                             pl.read(v_l2_norm_flat, [_ti]))

        # Scope D: V rotate + quantize.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_rotate_quant"):
            for ki in pl.range(ki_chunk, ki_chunk + 8):
                kv_col = ki * HEAD_DIM
                v_norm_bf16 = pl.slice(
                    v_norm_buf, [TOK_TILE, HEAD_DIM], [0, kv_col],
                )
                v_rot = pl.matmul(v_norm_bf16, rot_matrix, out_dtype=pl.FP32)

                v_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b0), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b1), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b2), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b3), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b4), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b5), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b6), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b7), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b8), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b9), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b10), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b11), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b12), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b13), cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=_b14), cmp_type=5))
                # FP32 -> INT32 -> FP16 -> UINT8 (pypto cast 4->1 byte workaround)
                v_idx_i32 = pl.cast(v_idx, pl.INT32, mode="trunc")
                v_idx_f16 = pl.cast(v_idx_i32, pl.FP16, mode="round")
                qv = pl.cast(v_idx_f16, pl.UINT8, mode="trunc")
                quant_v_temp = pl.assemble(quant_v_temp, qv, [0, kv_col])


