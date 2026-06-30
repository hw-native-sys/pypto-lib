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
          ├── turboquant_kv_quantize (@pl.jit.inline)   ← PolarQuant
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

def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d).

    Uses fully vectorized operations:
    - Analytical Gaussian CDF via torch.erf for centroid updates
    - No per-element Python loops in the hot path

    Returns:
        centroids: sorted tensor of 2^bits optimal centroids
        boundaries: sorted tensor of 2^bits - 1 boundaries
    """
    import torch

    n_levels = 2 ** bits
    sigma = 1.0 / _math.sqrt(d)

    # Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = torch.linspace(lo, hi, n_levels + 2)[1:-1]  # n_levels points, excluding endpoints

    for _ in range(max_iter):
        # Step 1: boundaries = midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Step 2: update centroids via analytical Gaussian conditional mean.
        # For N(0, sigma^2) in interval [a, b]:
        #   E[X | a < X < b] = sigma * (phi(a/sigma) - phi(b/sigma))
        #                       / (Phi(b/sigma) - Phi(a/sigma))
        # where phi = standard normal PDF, Phi = CDF.
        # phi(x) = exp(-x^2/2) / sqrt(2pi), Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        edges = torch.zeros(n_levels + 1)
        edges[0] = lo * 3
        edges[1:-1] = boundaries
        edges[-1] = hi * 3

        # Standardize edges
        z = edges / sigma

        # phi(z) = exp(-z^2/2) / sqrt(2*pi)
        neg_half_z2 = -0.5 * z * z
        phi_z = neg_half_z2.exp() * (1.0 / _math.sqrt(2.0 * _math.pi))

        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf_z = 0.5 * (1.0 + torch.erf(z * (1.0 / _math.sqrt(2.0))))

        # Conditional means: sigma * (phi(z_a) - phi(z_b)) / (Phi(z_b) - Phi(z_a))
        phi_diff = phi_z[:-1] - phi_z[1:]   # phi(z_a) - phi(z_b), shape (n_levels,)
        cdf_diff = cdf_z[1:] - cdf_z[:-1]   # Phi(z_b) - Phi(z_a), shape (n_levels,)

        # Guard against tiny intervals
        valid = cdf_diff > 1e-15
        new_centroids = torch.where(
            valid,
            sigma * phi_diff / cdf_diff.clamp(min=1e-15),
            centroids,
        )

        # Check convergence
        max_shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.clone(), boundaries.clone()


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
# Shared: INT4 KV dequant (gather → renormalize → scale → unrotate)
# ---------------------------------------------------------------------------


@pl.jit.inline
def turboquant_kv_dequant_chunk(
    quant_indices: pl.Tensor[[CMP_CHUNK, HALF_DIM], pl.UINT8],
    quant_scales: pl.Tensor[[CMP_CHUNK, 1], pl.FP32],
    tq_codebook: pl.Tensor[[CMP_CHUNK, N_LEVELS], pl.FP32],
    rot_slice: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    out_bf16: pl.Tensor[[CMP_CHUNK, HEAD_DIM], pl.BF16],
):
    """Dequantize one CMP_CHUNK of packed-INT4 KV cache: unpack → gather → renormalize → scale → unrotate.

    Indices are nibble-packed (two 4-bit indices per UINT8) in the half/half
    layout: byte c holds (idx[c + HALF_DIM] << 4) | idx[c].  Caller must wrap
    this call in a pl.at scope.  Caller slices a CMP_CHUNK of packed UINT8
    indices + FP32 scales from the quant cache, creates a BF16 output buffer,
    and assembles the result into a larger buffer.

    Steps (all inlined into the caller's scope):
      1. Unpack: UINT8 → FP16 → FP32, split into lo (fmods 16) / hi (div 16, trunc)
      2. Gather: pl.gather(codebook) per half (kept separate; no concat/assemble)
      3. Renormalize + scale: rsqrt(Σlo²+Σhi² + EPS), then ×L2 scales (each half)
      4. Unrotate: lo@rot[:,0:HALF]^T + hi@rot[:,HALF:]^T (split matmul) → cast BF16
    """
    # Unpack nibbles (half/half layout): byte c -> idx[c]=lo (mod 16),
    # idx[c+HALF_DIM]=hi (floor(/16)).  Arithmetic via unified div/fmods
    # (ands/shrs are tile-only and reject GM-slice tensors).  idx/16 is exact
    # in FP (16 = 2^4), so trunc cast gives the exact floor.  The two halves are
    # kept SEPARATE through renorm/scale/unrotate — no concat/assemble of gather
    # results (codegen's tmov rejects assembling gather sub-tiles, and concat
    # deadlocks the runtime).  Linearity makes the split exact:
    #   ||dec||² = Σlo² + Σhi² ;  dec @ rot^T = lo @ rot[:,0:HALF_DIM]^T
    #                                 + hi @ rot[:,HALF_DIM:]^T.
    idx_fp32 = pl.cast(pl.cast(quant_indices, target_type=pl.FP16), target_type=pl.FP32)
    lo_idx = pl.cast(pl.fmods(idx_fp32, 16.0), target_type=pl.INT32, mode="trunc")  # 0..15
    hi_idx = pl.cast(pl.div(idx_fp32, 16.0), target_type=pl.INT32, mode="trunc")    # 0..15
    centroids_lo = pl.gather(tq_codebook, dim=-1, index=lo_idx)  # [CMP_CHUNK, HALF_DIM] FP32
    centroids_hi = pl.gather(tq_codebook, dim=-1, index=hi_idx)  # [CMP_CHUNK, HALF_DIM] FP32

    # Renormalize to unit sphere (||dec||² = Σlo² + Σhi²) + rescale by L2 norms.
    dec_sq = pl.reshape(
        pl.add(
            pl.row_sum(pl.mul(centroids_lo, centroids_lo)),
            pl.row_sum(pl.mul(centroids_hi, centroids_hi)),
        ),
        [CMP_CHUNK, 1],
    )
    inv_norm = pl.rsqrt(pl.add(dec_sq, EPS))
    lo_scaled = pl.row_expand_mul(
        pl.row_expand_mul(centroids_lo, inv_norm), quant_scales,
    )
    hi_scaled = pl.row_expand_mul(
        pl.row_expand_mul(centroids_hi, inv_norm), quant_scales,
    )

    # Unrotate: dec @ rot^T = lo @ rot_lo^T + hi @ rot_hi^T, output BF16.
    rot_lo = pl.slice(rot_slice, [HEAD_DIM, HALF_DIM], [0, 0])
    rot_hi = pl.slice(rot_slice, [HEAD_DIM, HALF_DIM], [0, HALF_DIM])
    dec_unrot = pl.add(
        pl.matmul(pl.cast(lo_scaled, target_type=pl.BF16), rot_lo, b_trans=True, out_dtype=pl.FP32),
        pl.matmul(pl.cast(hi_scaled, target_type=pl.BF16), rot_hi, b_trans=True, out_dtype=pl.FP32),
    )
    out_bf16 = pl.assemble(out_bf16, pl.cast(dec_unrot, target_type=pl.BF16), [0, 0])


# ---------------------------------------------------------------------------
# Quantize: batched K/V RoPE + L2 norm + normalize + rotate + Lloyd-Max quant
# ---------------------------------------------------------------------------


@pl.jit.inline
def turboquant_kv_quantize(
    # Inputs (produced by Scope 1 in caller).
    k_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    v_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    rot_matrix: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    cos_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    cos_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_lo_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    sin_hi_all: pl.Tensor[[TOK_TILE, HALF_DIM], pl.FP32],
    # Outputs (caller creates, callee fills via pl.assemble). Index temps are
    # nibble-packed: [TOK_TILE, KV_HIDDEN // 2] (two 4-bit indices per UINT8).
    quant_k_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN // 2], pl.UINT8],
    quant_v_temp: pl.Tensor[[TOK_TILE, KV_HIDDEN // 2], pl.UINT8],
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

                # Lloyd-Max boundary search: idx = #{b : k_rot >= b} in [0, 15].
                # Compare against the SCALAR boundary _bN, not pl.full(...). A
                # tile-vs-scalar cmp lowers to tile.cmps -> TCMPS, whose CPU-sim
                # impl is correct; a tile-vs-tile cmp (pl.full rhs) lowers to
                # tile.cmp -> TCMP, whose CPU-sim impl can't model the FP32-src
                # -> UINT8-mask dst and fails to compile on a2a3sim. Numerically
                # identical; keep the scalar form (device is fine either way).
                k_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b0, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b1, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b2, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b3, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b4, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b5, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b6, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b7, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b8, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b9, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b10, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b11, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b12, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b13, cmp_type=5))
                k_idx = pl.add(k_idx, pl.cmp(k_rot, _b14, cmp_type=5))
                # Pack two 4-bit indices per UINT8 (half/half layout):
                # byte c = idx[c+HALF_DIM]*16 + idx[c], c in 0..HALF_DIM-1.
                # Arithmetic (not bitwise): shls/and_/or_ are tile-only ops and
                # reject GM-slice tensors; unified mul/add accept them.
                k_lo = pl.slice(k_idx, [TOK_TILE, HALF_DIM], [0, 0])
                k_hi = pl.slice(k_idx, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
                k_packed = pl.add(pl.mul(k_hi, 16.0), k_lo)            # FP32, 0..255 (exact)
                # FP32 -> INT32 -> FP16 -> UINT8 (pypto cast 4->1 byte workaround)
                k_packed_u8 = pl.cast(
                    pl.cast(pl.cast(k_packed, pl.INT32, mode="trunc"), pl.FP16, mode="round"),
                    pl.UINT8, mode="trunc",
                )
                quant_k_temp = pl.assemble(quant_k_temp, k_packed_u8, [0, ki * HALF_DIM])

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

                # Boundary search (see Scope B): compare against scalar _bN.
                v_idx = pl.full([TOK_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b0, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b1, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b2, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b3, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b4, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b5, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b6, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b7, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b8, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b9, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b10, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b11, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b12, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b13, cmp_type=5))
                v_idx = pl.add(v_idx, pl.cmp(v_rot, _b14, cmp_type=5))
                # Pack two 4-bit indices per UINT8 (half/half layout, see Scope B).
                v_lo = pl.slice(v_idx, [TOK_TILE, HALF_DIM], [0, 0])
                v_hi = pl.slice(v_idx, [TOK_TILE, HALF_DIM], [0, HALF_DIM])
                v_packed = pl.add(pl.mul(v_hi, 16.0), v_lo)
                v_packed_u8 = pl.cast(
                    pl.cast(pl.cast(v_packed, pl.INT32, mode="trunc"), pl.FP16, mode="round"),
                    pl.UINT8, mode="trunc",
                )
                quant_v_temp = pl.assemble(quant_v_temp, v_packed_u8, [0, ki * HALF_DIM])


