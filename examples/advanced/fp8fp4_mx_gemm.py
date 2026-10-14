# Copyright (c) PyPTO Contributors.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# Refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MXFP8 x MXFP4 dense GEMM on Ascend950 (a5) — two selectable weight layouts.

    C[m, n] = (A_fp8[m, k] * 2^SA[m, k/32]) @ (B_fp4 * 2^SB)

A is FP8E4M3FN with per-32-element E8M0 scales (physical MX_A_ZZ layout); B is
packed FP4E2M1 (torch ``float4_e2m1fn_x2``) with per-32-element E8M0 scales.
Each FP4 B tile is cast to FP8E4M3FN in-core — E2M1 values are exactly
representable in E4M3, so the per-group scales pass through unchanged — then
feeds ``pl.matmul_mx_acc``: the pypto rendering of DeepGEMM-Ascend's
AIV-dequant -> FP8-MAD pipeline. Documented head-to-head target (README dense
row): FP8xFP4 4096x7168x16384 = 1117.8 us / 861 TFLOPS / 99.5% of the 865
TFLOPS hardware limit (Ascend 950DT, CANN 9.20, bench_msprof cold-L2).

Weight layouts (``--layout {kn,nk}``, default ``kn``):

  kn  B is [K, N] packed along N (physical [K, N/2] bytes), scales MX_B_NN
      [K/32, N]. The rhs tile is cast in place and consumed directly by
      ``matmul_mx``.
  nk  B is [N, K] packed along K (physical [N, K/2] bytes), scales [N, K/32]
      (MX_A_ZZ-blocked bytes are byte-identical to MX_B_NN [K/32, N] of the
      transposed matrix) — DeepGEMM's ``fp8_fp4_gemm_nt`` transposed-operand
      contract, the layout output-major routed-expert weights naturally land
      in after K-major quantization. The rhs tile goes
      cast -> ``pl.move``(Mat, NZ repack) -> ``pl.tile.transpose_view`` (ZN
      dual) -> ``matmul_mx`` rhs — the cube consumes the ZN[N,K] == NZ[K,N]
      dual bytes directly, with no physical element-level transpose on AIV.

Both layouts peel K-chunk 0 through ``pl.matmul_mx`` to seed the L0C
accumulator (``matmul_mx_acc`` has no ``init_cond``); chunks >= 1 accumulate
in FP32. Variant flags: ``--mstack`` (cast-reuse over R=2 L0C
sub-accumulators, the measured optimum), ``--bf16`` (f32 accumulator drains
through the hardware fixpipe, not a vector-side cast), ``--fp4-lhs`` /
``--pure-fp8`` / ``--order`` / ``--persistent-spmd`` controls and
``--hoist-cast`` two-phase orchestration (kn only), ``--dump-cast``
byte-level cast probe (kn only), ``--bench``.

Perf figures quoted in the kernel docstrings below are archived campaign
measurements (Ascend 950DT, device 0) recorded where these kernels were
developed; they are not re-measured by this example. The scale-pack helpers
are copied from pypto's A5 MX system test (tests/st/runtime/ops/test_matmul_mx.py).
"""
# Contract (t36 device validation): offsets into a packed-FP4 tensor must be
# EVEN (block-aligned) — the carrier byte grid addresses nibble pairs. A
# statically provable odd offset is rejected at compile time; dynamic offsets
# are halved symbolically and are exact for even runtime values (the
# documented divsi contract).

import pypto.language as pl

M = 4096        # total rows of A / C
N = 7168        # total cols of C
K = 16384       # total reduction depth
STAGE = 2       # K-loop software pipeline depth
MX_GROUP = 32   # MX scale group size (fixed by the format)

# Layout-specific defaults, applied by __main__ per --layout (the live module
# globals below carry the kn defaults so an `import fp8fp4_mx_gemm` sees a
# consistent kn schedule, exactly like the standalone kn example did).
KN_TILE = (128, 256, 256)    # (M_TILE, N_TILE, K_TILE)
NK_TILE = (1024, 64, 64)
M_TILE, N_TILE, K_TILE = KN_TILE
KN_MSTACK = (2, 256)         # (MSTACK_R, MSTACK_MSUB): both layouts ship the
NK_MSTACK = (2, 256)         # measured-optimum R=2 x MSUB=256 unroll only
MSTACK_R, MSTACK_MSUB = KN_MSTACK

ORDER = "mn"    # block traversal: "mn" M-loop outer, "nm" N-loop outer
N_OUTER = False  # bare-bool kernel form of ORDER
PERSIST = False  # persistent spmd dispatch (one block per core, grid-stride)
PERSIST_BLOCKS = 32  # min(32, nblk) recomputed by _refresh_schedule()
NBLK_M = 32  # m-block count of the current tiling
NBLK_N = 28  # n-block count of the current tiling

# Analytic FP8 peak: 16384 flops/cycle * 1650 MHz * 32 AIC (2x the bf16 model).
FP8_PEAK_TFLOPS = 864.0
# DeepGEMM-Ascend README documented reference (dense table, cold L2).
DG_REF_US = 1117.8
DG_REF_TFLOPS = 861.0


def _refresh_schedule() -> None:
    """Recompute the flat-grid schedule values from the live tiles."""
    global NBLK_M, NBLK_N, PERSIST_BLOCKS
    NBLK_M = M // M_TILE
    NBLK_N = N // N_TILE
    PERSIST_BLOCKS = min(32, NBLK_M * NBLK_N)


def _check_shape_constants() -> None:
    """Validate shapes/tiles against MX alignment and on-chip budgets."""
    assert M % M_TILE == 0 and N % N_TILE == 0, "full tiles only: M % M_TILE and N % N_TILE must be 0"
    assert K % K_TILE == 0, "K must be a multiple of K_TILE (peel-free K reduction)"
    assert M_TILE % 16 == 0 and N_TILE % 16 == 0, "scale block alignment: 16 | M_TILE and 16 | N_TILE"
    assert K_TILE % (2 * MX_GROUP) == 0, "MX scale c0=2 packing: 64 | K_TILE"
    assert M_TILE * N_TILE * 4 <= 256 * 1024, (
        f"FP32 accumulator tile {M_TILE * N_TILE * 4 // 1024} KiB exceeds the 256 KiB A5 L0C"
    )
    assert (M_TILE + N_TILE) * K_TILE * STAGE <= 512 * 1024, (
        "double-buffered FP8 operand tiles exceed the 512 KiB L1 budget"
    )


_check_shape_constants()


# ===========================================================================
# Layout kn: B is [K, N], scales MX_B_NN — rhs cast feeds matmul_mx directly
# ===========================================================================


@pl.jit
def fp8fp4_mx_gemm(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP4],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp8fp4_mx_gemm_tile",
                # No UP_DOWN split / cross-core slot here (unlike the pure-FP8
                # kernels): a data-parallel pl.split region that pushes the
                # cast rhs tile to the cube deadlocks or corrupts random tiles
                # on the current PTOAS/pto-isa split-pipe protocol at scale.
                # ExpandMixedKernel warns on the combination; keep this
                # kernel split-free until that is fixed.
            ):
                # Peeled chunk 0: matmul_mx seeds the accumulator. All operands
                # go through pl.load: matmul_mx requires a uniform Tile level
                # and tensor.slice rejects MX-layout scale tensors.
                lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.cast(pl.load(b, [0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.cast(pl.load(b, [k0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp8fp4_mx_gemm_bf16(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP4],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.BF16]],
):
    """bf16 output variant: the f32 L0C accumulator drains through the hardware
    fixpipe (pto-isa TSTORE's intrinsic f32->bf16 narrowing, PIPE_FIX in the
    generated C++), not a vector-side pl.cast."""
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp8fp4_mx_gemm_bf16_tile",
            ):
                lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.cast(pl.load(b, [0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.cast(pl.load(b, [k0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp8fp4_mx_gemm_mstack(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP4],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """kn cast-reuse variant (kernel-level, DeepGEMM fp8_dequant_gemm shape).

    One core task owns BLOCK_M = MSTACK_R * MSTACK_MSUB rows of C for a fixed
    nb column block, with the K loop as the OUTER loop of the task body:
    each B[kb, nb] tile is loaded and cast FP4->FP8 ONCE per (task, kb) and
    the single cast result feeds MSTACK_R sub-tile ``matmul_mx_acc`` calls,
    each accumulating into its own [MSTACK_MSUB, N_TILE] L0C sub-accumulator
    (the m_mad-loop structure of DeepGEMM's ``fp8_dequant_gemm_impl``).

    cast_passes = M / (R*MSUB) per B column -- the in-task reuse cancels the
    per-sub-block repetition; the remaining (M/BLOCK_M)x repetition is the
    L0C-bounded floor (R*MSUB*N_TILE*4 <= 256 KiB), identical in count law to
    the single-acc kernel at equal BLOCK_M, BUT the decoupling buys K_TILE=256
    at BLOCK_M=512 (L0A is budgeted per [MSUB, K_TILE] sub-operand, not per
    BLOCK_M): half the K iterations, half the per-iteration scalar/flag
    overhead, 2x bigger cast tiles per fixed cost. The R-way unroll is written
    with explicit accumulators (acc0..acc{R-1}) because the DSL parser has no
    tuple-typed carried expressions; this example ships the measured-optimum
    R=2 x MSUB=256 unroll only (the development branch also carried R=4/8;
    __main__ rejects --mstack-r != 2 before tracing).
    """
    for mg in pl.parallel(0, M, MSTACK_R * MSTACK_MSUB):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp8fp4_mx_gemm_mstack_tile",
                # Split-free like every FP4 kernel here.
            ):
                # Peeled chunk 0: ONE cast seeds all R sub-accumulators.
                rhs0 = pl.cast(pl.load(b, [0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                m0 = mg + 0 * MSTACK_MSUB
                l0 = pl.load(a, [m0, 0], [MSTACK_MSUB, K_TILE])
                l0s = pl.load(a_scale, [m0, 0], [MSTACK_MSUB, K_TILE // MX_GROUP])
                acc0 = pl.matmul_mx(l0, l0s, rhs0, rhs0_scale)
                m1 = mg + 1 * MSTACK_MSUB
                l1 = pl.load(a, [m1, 0], [MSTACK_MSUB, K_TILE])
                l1s = pl.load(a_scale, [m1, 0], [MSTACK_MSUB, K_TILE // MX_GROUP])
                acc1 = pl.matmul_mx(l1, l1s, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    rhs = pl.cast(pl.load(b, [k0, nb], [K_TILE, N_TILE]), pl.FP8E4M3FN)
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    l0 = pl.load(a, [m0, k0], [MSTACK_MSUB, K_TILE])
                    l0s = pl.load(a_scale, [m0, kg], [MSTACK_MSUB, K_TILE // MX_GROUP])
                    acc0 = pl.matmul_mx_acc(acc0, l0, l0s, rhs, rhs_scale)
                    l1 = pl.load(a, [m1, k0], [MSTACK_MSUB, K_TILE])
                    l1s = pl.load(a_scale, [m1, kg], [MSTACK_MSUB, K_TILE // MX_GROUP])
                    acc1 = pl.matmul_mx_acc(acc1, l1, l1s, rhs, rhs_scale)
                c = pl.store(acc0, [m0, nb], c)
                c = pl.store(acc1, [m1, nb], c)
    return c


@pl.jit
def fp8fp8_mx_gemm(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Pure MXFP8 x MXFP8 control (B host-dequantized from FP4)."""
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp8fp8_mx_gemm_tile",
            ):
                lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.load(b, [0, nb], [K_TILE, N_TILE])
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.load(b, [k0, nb], [K_TILE, N_TILE])
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


CAST_K_TILE = 384
CAST_N_TILE = 192


@pl.jit
def fp4_to_fp8_pass(
    b: pl.Tensor[[K, N], pl.FP4],
    b_f8: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    """Phase-1 cast kernel (two-phase orchestration): one pass over B, K*N
    nibbles total instead of (M/M_TILE)*K*N inside the GEMM loop. Large
    streaming tiles for bandwidth (the GEMM tiles are too small for a
    memory-bound standalone pass)."""
    for kb in pl.parallel(0, K, CAST_K_TILE):
        for nb in pl.parallel(0, N, CAST_N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp4_cast_pass_tile",
            ):
                b_f8 = pl.store(
                    pl.cast(pl.load(b, [kb, nb], [CAST_K_TILE, CAST_N_TILE]), pl.FP8E4M3FN),
                    [kb, nb], b_f8,
                )
    return b_f8


@pl.jit
def fp8fp8_mx_gemm_nm(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Pure MXFP8 variant with the N block loop outer (order=nm)."""
    for nb in pl.parallel(0, N, N_TILE):
        for mb in pl.parallel(0, M, M_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp8fp8_mx_gemm_tile",
            ):
                lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.load(b, [0, nb], [K_TILE, N_TILE])
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.load(b, [k0, nb], [K_TILE, N_TILE])
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp8fp8_mx_gemm_persist(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Persistent spmd pure-MXFP8 variant: min(32, nblk) blocks, in-core
    grid-stride over the flat tile grid with the order=ORDER mapping."""
    for blk in pl.spmd(
        PERSIST_BLOCKS,
        name_hint="fp8fp8_mx_gemm_tile",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=1)],
    ):
        for gi in pl.range(blk, NBLK_M * NBLK_N, PERSIST_BLOCKS):
            if N_OUTER:
                nb_i = gi // NBLK_M
                mb_i = gi - nb_i * NBLK_M
            else:
                mb_i = gi // NBLK_N
                nb_i = gi - mb_i * NBLK_N
            mb = mb_i * M_TILE
            nb = nb_i * N_TILE
            lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
            lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
            rhs0 = pl.load(b, [0, nb], [K_TILE, N_TILE])
            rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
            acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
            for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                k0 = kb * K_TILE
                kg = kb * (K_TILE // MX_GROUP)
                lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                rhs = pl.load(b, [k0, nb], [K_TILE, N_TILE])
                rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
            c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp4fp8_mx_gemm_lhs(
    a: pl.Tensor[[M, K], pl.FP4],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Control variant: FP4 on the LHS (the ST-tested orientation)."""
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp4fp8_mx_gemm_lhs_tile",
            ):
                lhs0 = pl.cast(pl.load(a, [mb, 0], [M_TILE, K_TILE]), pl.FP8E4M3FN)
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.load(b, [0, nb], [K_TILE, N_TILE])
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.cast(pl.load(a, [mb, k0], [M_TILE, K_TILE]), pl.FP8E4M3FN)
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.load(b, [k0, nb], [K_TILE, N_TILE])
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp4fp8_mx_gemm_lhs_bf16(
    a: pl.Tensor[[M, K], pl.FP4],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP, N], pl.FP8E8M0, pl.MX_B_NN],
    c: pl.Out[pl.Tensor[[M, N], pl.BF16]],
):
    """bf16 output variant of the lhs-FP4 control: fixpipe intrinsic drain."""
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="fp4fp8_lhs_bf16_tile",
            ):
                lhs0 = pl.cast(pl.load(a, [mb, 0], [M_TILE, K_TILE]), pl.FP8E4M3FN)
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                rhs0 = pl.load(b, [0, nb], [K_TILE, N_TILE])
                rhs0_scale = pl.load(b_scale, [0, nb], [K_TILE // MX_GROUP, N_TILE])
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.cast(pl.load(a, [mb, k0], [M_TILE, K_TILE]), pl.FP8E4M3FN)
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.load(b, [k0, nb], [K_TILE, N_TILE])
                    rhs_scale = pl.load(b_scale, [kg, nb], [K_TILE // MX_GROUP, N_TILE])
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def dump_fp4_cast(
    b: pl.Tensor[[K, N], pl.FP4],
    c: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    """Debug probe: FP4 load + cast in ORCHESTRATION form (pl.at CORE_GROUP)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dump_fp4_cast_tile"):
        c = pl.store(pl.cast(pl.load(b, [0, 0], [K, N]), pl.FP8E4M3FN), [0, 0], c)
    return c


@pl.jit.incore
def dump_fp4_cast_incore_kernel(
    b: pl.Tensor[[K, N], pl.FP4],
    c: pl.Out[pl.Tensor[[K, N], pl.FP8E4M3FN]],
):
    return pl.store(pl.cast(pl.load(b, [0, 0], [K, N]), pl.FP8E4M3FN), [0, 0], c)


@pl.jit
def dump_fp4_cast_incore(
    b: pl.Tensor[[K, N], pl.FP4],
    c: pl.Out[pl.Tensor[[K, N], pl.BF16]],
):
    """Debug probe control: identical FP4 load + cast in @pl.jit.incore form."""
    return dump_fp4_cast_incore_kernel(b, c)


# ===========================================================================
# Layout nk: B is [N, K] (DeepGEMM fp8_fp4_gemm_nt contract) — rhs goes
# cast -> move(Mat, NZ) -> transpose_view(ZN dual) -> matmul_mx
# ===========================================================================


@pl.jit
def fp8fp4_mx_gemm_nk(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[N, K], pl.FP4],
    b_scale: pl.Tensor[[N, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    for mb in pl.parallel(0, M, M_TILE):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp8fp4_mx_gemm_nk_tile"):
                lhs0 = pl.load(a, [mb, 0], [M_TILE, K_TILE])
                lhs0_scale = pl.load(a_scale, [mb, 0], [M_TILE, K_TILE // MX_GROUP])
                # B[n, k] tile -> FP8 [N_TILE, K_TILE] identity cast -> move to
                # Mat (NZ repack) -> transpose_view (ZN dual) -> matmul_mx rhs
                # [K_TILE, N_TILE]: the cube consumes the ZN[N,K] == NZ[K,N]
                # dual bytes directly, with no physical element-level transpose
                # on the AIV side.
                rhs0 = pl.tile.transpose_view(
                    pl.move(
                        pl.cast(pl.load(b, [nb, 0], [N_TILE, K_TILE]), pl.FP8E4M3FN),
                        pl.MemorySpace.Mat,
                    )
                )
                # sfB[n, k/32] ZZ tile -> NN view = logical [K_TILE/32, N_TILE].
                rhs0_scale = pl.tile.transpose_view(
                    pl.load(b_scale, [nb, 0], [N_TILE, K_TILE // MX_GROUP])
                )
                acc = pl.matmul_mx(lhs0, lhs0_scale, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    lhs = pl.load(a, [mb, k0], [M_TILE, K_TILE])
                    lhs_scale = pl.load(a_scale, [mb, kg], [M_TILE, K_TILE // MX_GROUP])
                    rhs = pl.tile.transpose_view(
                        pl.move(
                            pl.cast(pl.load(b, [nb, k0], [N_TILE, K_TILE]), pl.FP8E4M3FN),
                            pl.MemorySpace.Mat,
                        )
                    )
                    rhs_scale = pl.tile.transpose_view(
                        pl.load(b_scale, [nb, kg], [N_TILE, K_TILE // MX_GROUP])
                    )
                    acc = pl.matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale)
                c = pl.store(acc, [mb, nb], c)
    return c


@pl.jit
def fp8fp4_mx_gemm_nk_mstack(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[N, K], pl.FP4],
    b_scale: pl.Tensor[[N, K // MX_GROUP], pl.FP8E8M0, pl.MX_A_ZZ],
    c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """nk cast-reuse variant: Route E rhs + the kn mstack structure.

    One core task owns BLOCK_M = MSTACK_R * MSTACK_MSUB rows of C for a fixed
    nb column block. Each B[nb, kb] tile is loaded and cast FP4->FP8 ONCE per
    (task, kb) via Route E (cast -> move(->Mat) -> transpose_view(ZN dual)),
    feeding MSTACK_R sub-tile matmul_mx_acc calls. Written for MSTACK_R=2
    (MSTACK_MSUB=256 was the tuned optimum on the development device)."""
    for mg in pl.parallel(0, M, MSTACK_R * MSTACK_MSUB):
        for nb in pl.parallel(0, N, N_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fp8fp4_mx_gemm_nk_mstack_tile"):
                # Peeled chunk 0: ONE Route E cast seeds all R sub-accumulators.
                rhs0 = pl.tile.transpose_view(
                    pl.move(
                        pl.cast(pl.load(b, [nb, 0], [N_TILE, K_TILE]), pl.FP8E4M3FN),
                        pl.MemorySpace.Mat,
                    )
                )
                rhs0_scale = pl.tile.transpose_view(
                    pl.load(b_scale, [nb, 0], [N_TILE, K_TILE // MX_GROUP])
                )
                m0 = mg + 0 * MSTACK_MSUB
                l0 = pl.load(a, [m0, 0], [MSTACK_MSUB, K_TILE])
                l0s = pl.load(a_scale, [m0, 0], [MSTACK_MSUB, K_TILE // MX_GROUP])
                acc0 = pl.matmul_mx(l0, l0s, rhs0, rhs0_scale)
                m1 = mg + 1 * MSTACK_MSUB
                l1 = pl.load(a, [m1, 0], [MSTACK_MSUB, K_TILE])
                l1s = pl.load(a_scale, [m1, 0], [MSTACK_MSUB, K_TILE // MX_GROUP])
                acc1 = pl.matmul_mx(l1, l1s, rhs0, rhs0_scale)
                for kb in pl.pipeline(1, K // K_TILE, stage=STAGE):
                    k0 = kb * K_TILE
                    kg = kb * (K_TILE // MX_GROUP)
                    rhs = pl.tile.transpose_view(
                        pl.move(
                            pl.cast(pl.load(b, [nb, k0], [N_TILE, K_TILE]), pl.FP8E4M3FN),
                            pl.MemorySpace.Mat,
                        )
                    )
                    rhs_scale = pl.tile.transpose_view(
                        pl.load(b_scale, [nb, kg], [N_TILE, K_TILE // MX_GROUP])
                    )
                    l0 = pl.load(a, [m0, k0], [MSTACK_MSUB, K_TILE])
                    l0s = pl.load(a_scale, [m0, kg], [MSTACK_MSUB, K_TILE // MX_GROUP])
                    acc0 = pl.matmul_mx_acc(acc0, l0, l0s, rhs, rhs_scale)
                    l1 = pl.load(a, [m1, k0], [MSTACK_MSUB, K_TILE])
                    l1s = pl.load(a_scale, [m1, kg], [MSTACK_MSUB, K_TILE // MX_GROUP])
                    acc1 = pl.matmul_mx_acc(acc1, l1, l1s, rhs, rhs_scale)
                c = pl.store(acc0, [m0, nb], c)
                c = pl.store(acc1, [m1, nb], c)
    return c


# ===========================================================================
# Host-side helpers: FP4/scale packing, inputs, golden references, bench
# ===========================================================================

_SCALE_BLOCK = 16
_SCALE_C0 = 2
_FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _pack_a_scale(codes):
    """Pack logical [M, K/32] A-scale codes into the MX_A_ZZ physical layout."""
    m, k_groups = codes.shape
    return (
        codes.reshape(m // _SCALE_BLOCK, _SCALE_BLOCK, k_groups // _SCALE_C0, _SCALE_C0)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(codes):
    """Pack logical [K/32, N] B-scale codes into the MX_B_NN physical layout."""
    k_groups, n = codes.shape
    return (
        codes.reshape(k_groups // _SCALE_C0, _SCALE_C0, n // _SCALE_BLOCK, _SCALE_BLOCK)
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _unpack_a_scale(packed):
    m, k_groups = packed.shape
    return (
        packed.reshape(m // _SCALE_BLOCK, k_groups // _SCALE_C0, _SCALE_BLOCK, _SCALE_C0)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_b_scale(packed):
    k_groups, n = packed.shape
    return (
        packed.reshape(n // _SCALE_BLOCK, k_groups // _SCALE_C0, _SCALE_BLOCK, _SCALE_C0)
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def _decode_fp4(data, rows, cols):
    """Decode a physical x2-packed FP4 tensor into a float32 matrix."""
    import torch

    packed = data.contiguous().view(torch.uint8).reshape(rows, cols // 2)
    codes = torch.empty((rows, cols), dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(_FP4_VALUES, dtype=torch.float32)
    return values[codes]


def make_inputs(m: int, n: int, k: int, seed: int = 23, fp4_lhs: bool = False,
                pure_fp8: bool = False, layout: str = "kn"):
    """Deterministic host-prequantized MX inputs.

    layout="kn": b is [K, N] packed along N, b_scale is MX_B_NN [K/32, N].
    layout="nk": b is [N, K] packed along K, b_scale lives at [N, K/32] as the
    MX_A_ZZ-blocked byte stream of the transposed code matrix (byte-identical
    to MX_B_NN [K/32, N]; pack the transposed codes with _pack_b_scale and
    re-view at [N, K/32]). fp4_lhs/pure_fp8 are kn-only controls.
    """
    import os

    import torch

    generator = torch.Generator().manual_seed(seed)
    fp8_ab = torch.randint(-2, 3, (m, k), generator=generator).to(torch.float8_e4m3fn)
    if os.environ.get("MX_DEBUG_A_IDENT"):  # debug: A = I, so C must equal decode(B)[0:M]
        fp8_ab = torch.eye(m, k).to(torch.float8_e4m3fn)

    def pack_x2(codes):
        return (((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)).contiguous().view(
            torch.float4_e2m1fn_x2
        )

    a_scale_codes = torch.randint(126, 130, (m, k // MX_GROUP), generator=generator).to(torch.uint8)
    if layout == "nk":
        assert not fp4_lhs and not pure_fp8, "--fp4-lhs/--pure-fp8 are kn-only controls"
        a = fp8_ab
        codes_b = torch.randint(0, 16, (n, k), generator=generator).to(torch.uint8)  # [N, K] logical
        b = pack_x2(codes_b)  # packed along K -> physical [N, K/2]
        b_sc = torch.randint(126, 130, (n, k // MX_GROUP), generator=generator).to(torch.uint8)  # [N, G]
        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_sc.T.contiguous()).reshape(n, k // MX_GROUP).view(torch.float8_e8m0fnu)
        return a, a_scale, b, b_scale

    fp8_b = torch.randint(-2, 3, (k, n), generator=generator).to(torch.float8_e4m3fn)
    codes_a = torch.randint(0, 16, (m, k), generator=generator).to(torch.uint8)
    if os.environ.get("MX_DEBUG_B_RAMP"):  # debug: code[j] = j % 16 -> value reveals addressing
        codes_b = (torch.arange(k * n) % 16).reshape(k, n).to(torch.uint8)
    else:
        codes_b = torch.randint(0, 16, (k, n), generator=generator).to(torch.uint8)

    a = pack_x2(codes_a) if fp4_lhs else fp8_ab
    b = fp8_b if (fp4_lhs or pure_fp8) else pack_x2(codes_b)
    if os.environ.get("MX_FLAT_SCALES"):  # debug: force every scale to 2^0
        a_scale_codes = torch.full((m, k // MX_GROUP), 127, dtype=torch.uint8)
        b_scale_codes = torch.full((k // MX_GROUP, n), 127, dtype=torch.uint8)
    else:
        b_scale_codes = torch.randint(126, 130, (k // MX_GROUP, n), generator=generator).to(torch.uint8)
    a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
    b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)
    return a, a_scale, b, b_scale


def build_tensor_specs(m: int = M, n: int = N, k: int = K, fp4_lhs: bool = False,
                       pure_fp8: bool = False, layout: str = "kn"):
    from golden import TensorSpec
    import torch

    a, a_scale, b, b_scale = make_inputs(m, n, k, fp4_lhs=fp4_lhs, pure_fp8=pure_fp8,
                                         layout=layout)
    fp8_dt, fp4_dt = torch.float8_e4m3fn, torch.float4_e2m1fn_x2
    # torch x2 dtypes count *carrier* elements (1 byte = 2 nibbles) per dim, so
    # an FP4 spec must declare the packed carrier shape [rows, logical_cols/2].
    # A logical [rows, cols] spec makes the JIT compile dummy twice as wide as
    # the real tensor and the GM view row pitch doubles.
    if layout == "nk":
        return [
            TensorSpec("a", [m, k], fp8_dt, init_value=a),
            TensorSpec("a_scale", [m, k // MX_GROUP], torch.float8_e8m0fnu, init_value=a_scale),
            TensorSpec("b", [n, k // 2], fp4_dt, init_value=b),
            TensorSpec("b_scale", [n, k // MX_GROUP], torch.float8_e8m0fnu, init_value=b_scale),
            TensorSpec("c", [m, n], torch.float32),
        ]
    return [
        TensorSpec("a", [m, k // 2] if fp4_lhs else [m, k], fp4_dt if fp4_lhs else fp8_dt, init_value=a),
        TensorSpec("a_scale", [m, k // MX_GROUP], torch.float8_e8m0fnu, init_value=a_scale),
        TensorSpec("b", [k, n] if (fp4_lhs or pure_fp8) else [k, n // 2],
                   fp8_dt if (fp4_lhs or pure_fp8) else fp4_dt, init_value=b),
        TensorSpec("b_scale", [k // MX_GROUP, n], torch.float8_e8m0fnu, init_value=b_scale),
        TensorSpec("c", [m, n], torch.float32),
    ]


def golden_fp8fp4_mx_gemm(tensors):
    """FP32 reference for layout kn: decode values, apply per-group E8M0
    scales, matmul."""
    import torch

    a_t, b_t = tensors["a"], tensors["b"]
    a = _decode_fp4(a_t, M, K) if a_t.dtype == torch.float4_e2m1fn_x2 else a_t.to(torch.float32)
    b = _decode_fp4(b_t, K, N) if b_t.dtype == torch.float4_e2m1fn_x2 else b_t.to(torch.float32)
    a_codes = _unpack_a_scale(tensors["a_scale"].view(torch.uint8))
    b_codes = _unpack_b_scale(tensors["b_scale"].view(torch.uint8))
    k_group = torch.arange(K) // MX_GROUP
    a_scaled = a * torch.pow(2.0, a_codes.to(torch.float32) - 127)[:, k_group]
    b_scaled = b * torch.pow(2.0, b_codes.to(torch.float32) - 127)[k_group, :]
    tensors["c"][:] = torch.matmul(a_scaled, b_scaled)


def golden_fp8fp4_mx_gemm_nk(tensors):
    """FP32 reference for layout nk: decode [N, K] values + per-group E8M0
    scales, A @ B^T."""
    import torch as _t

    a = tensors["a"].to(_t.float32)
    b = _decode_fp4(tensors["b"], N, K)  # [N, K]
    a_codes = _unpack_a_scale(tensors["a_scale"].view(_t.uint8))  # [M, G]
    # b_scale bytes are MX_B_NN-blocked (packed from codes.T via _pack_b_scale):
    # re-view the flat stream at [G, N] and unpack with the matching inverse.
    b_flat = tensors["b_scale"].view(_t.uint8).reshape(-1).reshape(K // MX_GROUP, N)
    b_codes = _unpack_b_scale(b_flat).T.contiguous()  # [N, G]
    a_f = a * _t.pow(2.0, a_codes.to(_t.float32) - 127).repeat_interleave(32, dim=1)
    b_f = b * _t.pow(2.0, b_codes.to(_t.float32) - 127).repeat_interleave(32, dim=1)
    tensors["c"][:] = a_f @ b_f.T


def golden_dump_fp4_cast(tensors):
    """Reference for the cast dump probe: decode B and cast to FP8."""
    import torch

    tensors["c"][:] = _decode_fp4(tensors["b"], K, N).to(torch.float8_e4m3fn)


def _bench_effective_us(stats) -> float:
    """Mean effective time in us from PYPTO_BENCH stats (0 if unavailable)."""
    import statistics

    if stats is None:
        return 0.0
    eff = [float(e) for e in stats.per_round("effective") if e > 0.0]
    return statistics.fmean(eff) if eff else 0.0


def _report_bench(stats, m: int, n: int, k: int, layout: str = "kn") -> None:
    """Print kernel time / TFLOPS / bandwidth and the DeepGEMM documented delta."""
    import statistics

    if stats is None:
        print("[BENCH] no benchmark stats captured")
        raise SystemExit(0)
    eff = [float(e) for e in stats.per_round("effective") if e > 0.0]
    if not eff:
        print("[BENCH] effective_us unavailable (simulator platform or non-profiling build)")
        raise SystemExit(0)
    mean_us = statistics.fmean(eff)
    tflops = 2.0 * m * n * k / mean_us / 1e6
    gbps = (m * k + k * n * 0.5 + m * n * 4 + (m * k + k * n) / MX_GROUP) / mean_us / 1e3
    tag = "persist" if PERSIST else ORDER
    print(f"[BENCH] fp8fp4_mx_gemm[{layout}] M={m} N={n} K={k} "
          f"tile={M_TILE}x{N_TILE}x{K_TILE} s{STAGE} {tag}")
    print(
        f"[BENCH] effective_us n={len(eff)} mean={mean_us:.1f} "
        f"median={statistics.median(eff):.1f} min={min(eff):.1f} max={max(eff):.1f}"
    )
    print(f"[BENCH] TFLOPS={tflops:.1f} BW={gbps:.0f}GB/s "
          f"peak_share={100.0 * tflops / FP8_PEAK_TFLOPS:.1f}% (of {FP8_PEAK_TFLOPS} TFLOPS fp8)")
    if (m, n, k) == (4096, 7168, 16384):
        print(f"[BENCH] DeepGEMM documented (dense FP8xFP4, cold L2): {DG_REF_US}us "
              f"{DG_REF_TFLOPS} TFLOPS 99.5%; gap={mean_us / DG_REF_US:.3f}x")


if __name__ == "__main__":
    import argparse
    import os

    import torch
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5sim", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layout", choices=["kn", "nk"], default="kn",
                        help="weight layout: kn = B [K, N] MX_B_NN; nk = B [N, K] "
                             "(DeepGEMM fp8_fp4_gemm_nt contract)")
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--m-tile", type=int, default=None,
                        help=f"M_TILE (layout default: kn {KN_TILE[0]}, nk {NK_TILE[0]})")
    parser.add_argument("--n-tile", type=int, default=None,
                        help=f"N_TILE (layout default: kn {KN_TILE[1]}, nk {NK_TILE[1]})")
    parser.add_argument("--k-tile", type=int, default=None,
                        help=f"K_TILE (layout default: kn {KN_TILE[2]}, nk {NK_TILE[2]})")
    parser.add_argument("--stage", type=int, default=STAGE)
    parser.add_argument("--bench", action="store_true",
                        help="time the kernel after validation via PYPTO_BENCH and report TFLOPS")
    parser.add_argument("--fp4-lhs", action="store_true",
                        help="kn-only control variant: FP4 on the A side (the ST-tested orientation)")
    parser.add_argument("--pure-fp8", action="store_true",
                        help="kn-only control variant: both sides FP8 (B host-dequantized)")
    parser.add_argument("--bf16", action="store_true",
                        help="bf16 output: f32 accumulator drains through the hardware fixpipe "
                             "(intrinsic narrowing, not a vector-side pl.cast)")
    parser.add_argument("--hoist-cast", action="store_true",
                        help="kn-only two-phase orchestration: a pass-1 kernel casts B FP4->FP8 "
                             "once (K*N nibbles), then the pure-FP8 GEMM consumes the pre-cast "
                             "data. Eliminates the (M/M_TILE)x in-loop cast redundancy.")
    parser.add_argument("--dump-cast", choices=["orch", "incore"], default=None,
                        help="kn-only debug probe: dump the FP4->FP8 cast of B (single tile)")
    parser.add_argument("--order", choices=["mn", "nm"], default="mn",
                        help="kn-only block traversal order (pure-fp8 path): mn / nm")
    parser.add_argument("--persistent-spmd", action="store_true",
                        help="kn-only persistent spmd dispatch (pure-fp8 path): min(32,nblk) blocks")
    parser.add_argument("--mstack", action="store_true",
                        help="cast-reuse variant: one core task owns R*MSTACK_MSUB rows of C "
                             "for a fixed nb; the K loop is outer so each B tile is cast once "
                             "and feeds R sub-tile matmul_mx_acc accumulators (DeepGEMM "
                             "m_mad shape). Ships the R=2 unroll only.")
    parser.add_argument("--mstack-r", type=int, default=None,
                        help=f"mstack R (layout default: kn {KN_MSTACK[0]}, nk {NK_MSTACK[0]})")
    parser.add_argument("--mstack-msub", type=int, default=None,
                        help=f"mstack rows per sub-accumulator (layout default: kn "
                             f"{KN_MSTACK[1]}, nk {NK_MSTACK[1]})")
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--bench-rounds", type=int, default=50)
    args = parser.parse_args()

    nk = args.layout == "nk"
    if nk and (args.fp4_lhs or args.pure_fp8 or args.hoist_cast or args.dump_cast
               or args.order != "mn" or args.persistent_spmd):
        raise SystemExit("--fp4-lhs / --pure-fp8 / --hoist-cast / --dump-cast / --order / "
                         "--persistent-spmd are kn-only controls; drop them for --layout nk")

    # Rebind the module-level constants the JIT reads when golden.run() compiles;
    # the trace happens inside run(), so the overrides are picked up.
    M, N, K = args.m, args.n, args.k
    M_TILE = args.m_tile if args.m_tile is not None else (NK_TILE[0] if nk else KN_TILE[0])
    N_TILE = args.n_tile if args.n_tile is not None else (NK_TILE[1] if nk else KN_TILE[1])
    K_TILE = args.k_tile if args.k_tile is not None else (NK_TILE[2] if nk else KN_TILE[2])
    STAGE = args.stage
    MSTACK_R = args.mstack_r if args.mstack_r is not None else (NK_MSTACK[0] if nk else KN_MSTACK[0])
    MSTACK_MSUB = (args.mstack_msub if args.mstack_msub is not None
                   else (NK_MSTACK[1] if nk else KN_MSTACK[1]))

    if args.dump_cast:
        # Cast-dump probe: whole-tensor single shot, B ramp pattern recommended
        # (MX_DEBUG_B_RAMP=1) so the value at (k, n) identifies the source column.
        from golden import TensorSpec

        _, _, b_probe, _ = make_inputs(16, args.n, args.k)  # only b is used
        dump_fn = dump_fp4_cast if args.dump_cast == "orch" else dump_fp4_cast_incore
        result = run(
            fn=dump_fn,
            specs=[
                # Carrier shape: torch x2 dims count carriers (2 nibbles).
                TensorSpec("b", [K, N // 2], torch.float4_e2m1fn_x2, init_value=b_probe),
                TensorSpec("c", [K, N], torch.float8_e4m3fn),
            ],
            golden_fn=golden_dump_fp4_cast,
            config=dict(platform=args.platform, device_id=args.device),
            rtol=0.0,
            atol=0.0,
        )
        print("[DUMP-CAST] PASS" if result.passed else f"[DUMP-CAST] FAIL: {result.error}")
        raise SystemExit(0 if result.passed else 1)

    if not args.pure_fp8:
        # Without a split (see the kn kernel note) the whole in-core FP4->FP8
        # cast chain lives on one AIV lane, so cap the cast-side tile extent
        # to the 245760 B A5 Vec limit (~40960 logical nibbles). Power-of-two
        # candidates keep the block grid exact for the canonical N values
        # (7168 / 512 / ...); 16 is the scale-block alignment floor.
        envelope = 40960
        cap = envelope // K_TILE
        fitted = next((c for c in (256, 128, 64, 32, 16) if c <= cap), 16)
        if args.fp4_lhs and M_TILE * K_TILE > envelope:
            print(f"[TILE] FP4-lhs: shrinking M_TILE {M_TILE} -> {fitted} (single-lane Vec envelope)")
            M_TILE = fitted
        if not args.fp4_lhs and N_TILE * K_TILE > envelope:
            print(f"[TILE] FP4-rhs[{args.layout}]: shrinking N_TILE {N_TILE} -> {fitted} "
                  f"(single-lane Vec envelope)")
            N_TILE = fitted
    ORDER = args.order if args.order else "mn"
    N_OUTER = ORDER == "nm"
    PERSIST = args.persistent_spmd
    if args.mstack:
        if MSTACK_R != 2:
            raise SystemExit("--mstack ships the R=2 unroll only (the measured optimum); "
                             "the R=4/8 development unrolls are not part of this example")
        # mstack tile legality (checked here, not in _check_shape_constants,
        # because BLOCK_M = R*MSUB replaces M_TILE in every budget):
        #   L0C  : R*MSUB*N_TILE*4 <= 256 KiB  (R simultaneous sub-accumulators)
        #   L0A/L0B : per sub-operand MSUB*K_TILE / N_TILE*K_TILE <= 64 KiB
        #   scale alignment: 16 | MSUB, 64 | K_TILE (MX c0=2 packing)
        assert M % (MSTACK_R * MSTACK_MSUB) == 0, \
            f"mstack: M must be a multiple of BLOCK_M = {MSTACK_R * MSTACK_MSUB}"
        assert MSTACK_MSUB % 16 == 0, "mstack: 16 | MSTACK_MSUB (MX_A_ZZ scale block)"
        assert K_TILE % 64 == 0, "mstack: 64 | K_TILE (MX scale c0=2 packing)"
        assert MSTACK_R * MSTACK_MSUB * N_TILE * 4 <= 256 * 1024, (
            f"mstack: {MSTACK_R}x{MSTACK_MSUB}x{N_TILE} FP32 sub-accumulators = "
            f"{MSTACK_R * MSTACK_MSUB * N_TILE * 4 // 1024} KiB exceeds the 256 KiB A5 L0C"
        )
        assert MSTACK_MSUB * K_TILE <= 64 * 1024, (
            f"mstack: L0A sub-operand MSUB*K_TILE = {MSTACK_MSUB * K_TILE // 1024} KiB > 64 KiB"
        )
        assert N_TILE * K_TILE <= 64 * 1024, (
            f"mstack: L0B operand N_TILE*K_TILE = {N_TILE * K_TILE // 1024} KiB > 64 KiB"
        )
        if not nk:
            cast_red = M // (MSTACK_R * MSTACK_MSUB)
            print(f"[MSTACK] BLOCK_M={MSTACK_R * MSTACK_MSUB} (R={MSTACK_R} x MSUB={MSTACK_MSUB}) "
                  f"tile=<{MSTACK_R * MSTACK_MSUB}x{N_TILE}x{K_TILE}> s{STAGE}: "
                  f"cast passes/B-column = {cast_red}x, iters/task = {K // K_TILE}")
    _refresh_schedule()
    _check_shape_constants()

    if args.bench:
        os.environ.setdefault("PYPTO_BENCH", "1")
        os.environ.setdefault("PYPTO_BENCH_WARMUP", str(args.bench_warmup))
        os.environ.setdefault("PYPTO_BENCH_ROUNDS", str(args.bench_rounds))

    if args.hoist_cast and not args.fp4_lhs and not args.pure_fp8 and not args.bf16:
        # ---- two-phase orchestration: cast once, then pure-FP8 GEMM ----
        from golden import TensorSpec
        a, a_scale, b_fp4, b_scale = make_inputs(M, N, K, fp4_lhs=False, pure_fp8=False)
        b_f8_host = _decode_fp4(b_fp4, K, N).to(torch.float8_e4m3fn)  # same values the device cast produces

        cast_specs = [
            TensorSpec("b", [K, N // 2], torch.float4_e2m1fn_x2, init_value=b_fp4),
            TensorSpec("b_f8", [K, N], torch.float8_e4m3fn),
        ]

        def golden_cast(tensors):
            tensors["b_f8"][:] = b_f8_host

        cast_result = run(
            fn=fp4_to_fp8_pass,
            specs=cast_specs,
            golden_fn=golden_cast,
            config=dict(platform=args.platform, device_id=args.device),
            rtol=0, atol=0,  # exact identity
        )
        if not cast_result.passed:
            print(f"[CAST] FAIL: {cast_result.error}")
            raise SystemExit(1)
        print("[CAST] PASS")

        gemm_specs = [
            TensorSpec("a", [M, K], torch.float8_e4m3fn, init_value=a),
            TensorSpec("a_scale", [M, K // MX_GROUP], torch.float8_e8m0fnu,
                       init_value=a_scale),
            TensorSpec("b", [K, N], torch.float8_e4m3fn, init_value=b_f8_host),
            TensorSpec("b_scale", [K // MX_GROUP, N], torch.float8_e8m0fnu,
                       init_value=b_scale),
            TensorSpec("c", [M, N], torch.float32),
        ]
        gemm_result = run(
            fn=fp8fp8_mx_gemm,
            specs=gemm_specs,
            golden_fn=golden_fp8fp4_mx_gemm,
            config=dict(platform=args.platform, device_id=args.device),
            rtol=1e-3, atol=1e-3,
        )
        if not gemm_result.passed:
            print(f"[GEMM] FAIL: {gemm_result.error}")
            raise SystemExit(1)
        print("[GOLDEN] PASS (hoist-cast two-phase)")
        if args.bench:
            cast_us = _bench_effective_us(cast_result.bench)
            gemm_us = _bench_effective_us(gemm_result.bench)
            total_us = cast_us + gemm_us
            tflops = 2.0 * M * N * K / total_us / 1e6
            print(f"[BENCH] hoist-cast two-phase: cast={cast_us:.1f}us + gemm={gemm_us:.1f}us "
                  f"= {total_us:.1f}us")
            print(f"[BENCH] TFLOPS={tflops:.1f} (two-phase; vs DeepGEMM 1117.8us)")
        raise SystemExit(0)

    if nk:
        if args.mstack:
            fn = fp8fp4_mx_gemm_nk_mstack
        else:
            fn = fp8fp4_mx_gemm_nk
        golden_fn = golden_fp8fp4_mx_gemm_nk
        specs = build_tensor_specs(m=M, n=N, k=K, layout="nk")
    else:
        if args.pure_fp8 and PERSIST:
            fn = fp8fp8_mx_gemm_persist
        elif args.pure_fp8 and N_OUTER:
            fn = fp8fp8_mx_gemm_nm
        elif args.pure_fp8:
            fn = fp8fp8_mx_gemm
        elif args.fp4_lhs and args.bf16:
            fn = fp4fp8_mx_gemm_lhs_bf16
        elif args.fp4_lhs:
            fn = fp4fp8_mx_gemm_lhs
        elif args.mstack and args.bf16:
            raise SystemExit("--mstack currently supports fp32 output only (drop --bf16)")
        elif args.mstack:
            fn = fp8fp4_mx_gemm_mstack
        elif args.bf16:
            fn = fp8fp4_mx_gemm_bf16
        else:
            fn = fp8fp4_mx_gemm
        golden_fn = golden_fp8fp4_mx_gemm  # fp32 ref auto-narrows into the bf16 c
        specs = build_tensor_specs(m=M, n=N, k=K, fp4_lhs=args.fp4_lhs, pure_fp8=args.pure_fp8)

    out_dtype = torch.bfloat16 if args.bf16 and not args.pure_fp8 else torch.float32
    out_rtol = 1e-2 if args.bf16 else 1e-3  # bf16 has ~8 bits mantissa
    out_atol = 1e-2 if args.bf16 else 1e-3
    # Swap the c spec to bf16 for --bf16 (build_tensor_specs creates fp32).
    if out_dtype == torch.bfloat16:
        specs = [s if s.name != "c" else type(s)(s.name, s.shape, out_dtype)
                 for s in specs]

    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=out_rtol,
        atol=out_atol,
    )
    if not result.passed:
        print(f"[GOLDEN] FAIL: {result.error}")
        raise SystemExit(1)
    print("[GOLDEN] PASS")
    if args.bench:
        _report_bench(result.bench, M, N, K, layout=args.layout)
