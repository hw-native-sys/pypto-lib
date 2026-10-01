# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # needs PTOAS timg2col; public assembler pin does not parse it
"""Two standalone int8 conv2d passes with a GM ``mid`` bounce.

Pass 1 is a full-image conv1: each C1 plane is ``pl.load``'d (contiguous GM)
and img2col'd with C=32, then FIXPIPE VREQ8 ``fixpipe_store`` writes NHWC
``mid`` at ``chunk * M2``. Do not ``gather_row`` the two C1 strips into one
NC1HWC0 L1 band: on CPU-sim that path is wrong when the band sits at L1
address 0 (fused avoids it because weights occupy address 0 first).
Pass 2 reloads NHWC ``mid`` windows and runs conv2 with the same banding /
padding as fused. Default ``__main__`` runs fused then both passes on the
same inputs, reports ``mid`` vs torch by band, and compares ``y``.

    python models/fused_conv2d/unfused_conv2d.py -p a5sim
"""
from pathlib import Path
import sys

import pypto.language as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_conv2d import (
    A_LAST,
    ACC_M,
    C0,
    C1,
    CI,
    COUT1,
    COUT2,
    FusedConv2d,
    HB,
    HI,
    HO1,
    HO2,
    K1,
    K2,
    KH,
    KW,
    M1,
    M2,
    MID_ROWS,
    M_TILE,
    OUT_ROWS,
    PAD,
    PLANE_IN,
    STRIDE1,
    STRIDE2,
    TILE_K1,
    TILE_K2,
    WI,
    WO1,
    WO2,
    XROWS,
    _conv_i32,
    _requantize,
    build_tensor_specs,
    golden_fused_conv2d,
)

MID_PLANE = HO1 * WO1
# Standalone conv1 covers HB output rows per band (no conv2 halo).
# stride*(HB-1)+KH = 37; first band pad_top adds one extra valid out row we do not store.
XROWS1 = STRIDE1 * (HB - 1) + KH
N_BANDS = HO1 // HB
# One C1 plane is C=32; KH*KW*C0 must be tiled without spanning the other C1.
K_C1 = KH * KW * C0
TILE_KC = 32
FM1 = XROWS1 * WI
FM_FIRST = XROWS * WI


@pl.program
class Conv2dPass1:
    """Standalone conv1. FIXPIPE VREQ8 writes the full NHWC ``mid`` image to GM.

    Each C1 plane is a contiguous GM rectangle, loaded with ``pl.load``.
    ``gather_row`` into an NC1HWC0 L1 band at address 0 is wrong on CPU-sim.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[MID_PLANE, COUT1], pl.INT8]],
    ) -> pl.Tensor[[MID_PLANE, COUT1], pl.INT8]:
        wt1 = pl.load(filter1, [0, 0], [K1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_mat = pl.load(bias1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_t = pl.move(bias1_mat, target_memory=pl.MemorySpace.Bias)
        scale1_l1 = pl.load(scale1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        scale1_fp = pl.move(scale1_l1, target_memory=pl.MemorySpace.LeftScale)

        fm0 = pl.load(x, [0, 0], [FM1, C0], target_memory=pl.MemorySpace.Mat)
        fm1 = pl.load(x, [PLANE_IN, 0], [FM1, C0], target_memory=pl.MemorySpace.Mat)
        for mt in pl.range(HB // M_TILE):
            m0 = mt * ACC_M
            acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for g in pl.range(K_C1 // TILE_KC):
                k0 = g * TILE_KC
                a1 = pl.tile.img2col(
                    fm0, m0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS1, WI, C0],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
                )
                b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
                if g == 0:
                    acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
                else:
                    acc1 = pl.tile.matmul_acc(acc1, a1, b1)
            for g in pl.range(K_C1 // TILE_KC):
                k0 = g * TILE_KC
                a1 = pl.tile.img2col(
                    fm1, m0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS1, WI, C0],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
                )
                b1 = pl.tile.extract(wt1, K_C1 + k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
                acc1 = pl.tile.matmul_acc(acc1, a1, b1)
            mid = pl.tile.fixpipe_store(acc1, scale1_fp, mid, [m0, 0])

        for chunk in pl.range(1, N_BANDS):
            b = STRIDE1 * (HB * chunk) - PAD
            src0 = WI * b
            src1 = WI * b + PLANE_IN
            fm0 = pl.load(x, [src0, 0], [FM1, C0], target_memory=pl.MemorySpace.Mat)
            fm1 = pl.load(x, [src1, 0], [FM1, C0], target_memory=pl.MemorySpace.Mat)
            for mt in pl.range(HB // M_TILE):
                m0 = mt * ACC_M
                acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
                for g in pl.range(K_C1 // TILE_KC):
                    k0 = g * TILE_KC
                    a1 = pl.tile.img2col(
                        fm0, m0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS1, WI, C0],
                        kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE1, STRIDE1],
                    )
                    b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
                    if g == 0:
                        acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
                    else:
                        acc1 = pl.tile.matmul_acc(acc1, a1, b1)
                for g in pl.range(K_C1 // TILE_KC):
                    k0 = g * TILE_KC
                    a1 = pl.tile.img2col(
                        fm1, m0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS1, WI, C0],
                        kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE1, STRIDE1],
                    )
                    b1 = pl.tile.extract(wt1, K_C1 + k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
                    acc1 = pl.tile.matmul_acc(acc1, a1, b1)
                mid = pl.tile.fixpipe_store(acc1, scale1_fp, mid, [chunk * M2 + m0, 0])
        return mid

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[MID_PLANE, COUT1], pl.INT8]],
    ) -> pl.Tensor[[MID_PLANE, COUT1], pl.INT8]:
        return self.kernel(x, filter1, bias1, scale1, mid)


@pl.program
class Conv2dPass1FirstTile:
    """First conv1 Acc tile via per-C1 ``pl.load`` (same path as standalone pass1)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        wt1 = pl.load(filter1, [0, 0], [K1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_mat = pl.load(bias1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_t = pl.move(bias1_mat, target_memory=pl.MemorySpace.Bias)
        scale1_l1 = pl.load(scale1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        scale1_fp = pl.move(scale1_l1, target_memory=pl.MemorySpace.LeftScale)
        fm0 = pl.load(x, [0, 0], [FM_FIRST, C0], target_memory=pl.MemorySpace.Mat)
        fm1 = pl.load(x, [PLANE_IN, 0], [FM_FIRST, C0], target_memory=pl.MemorySpace.Mat)
        acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
        for g in pl.range(K_C1 // TILE_KC):
            k0 = g * TILE_KC
            a1 = pl.tile.img2col(
                fm0, 0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS, WI, C0],
                kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
            )
            b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
            if g == 0:
                acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
            else:
                acc1 = pl.tile.matmul_acc(acc1, a1, b1)
        for g in pl.range(K_C1 // TILE_KC):
            k0 = g * TILE_KC
            a1 = pl.tile.img2col(
                fm1, 0, k0, shape=[ACC_M, TILE_KC], image_shape=[XROWS, WI, C0],
                kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
            )
            b1 = pl.tile.extract(wt1, K_C1 + k0, 0, shape=[TILE_KC, COUT1], target_memory=pl.MemorySpace.Right)
            acc1 = pl.tile.matmul_acc(acc1, a1, b1)
        return pl.tile.fixpipe_store(acc1, scale1_fp, mid, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        return self.kernel(x, filter1, bias1, scale1, mid)


PACKED = C1 * XROWS * WI


@pl.program
class Conv2dPass1Packed:
    """First conv1 Acc tile, ``pl.load`` of host-packed NC1HWC0 (no gather_row)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x_band: pl.Tensor[[PACKED, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        fm = pl.load(x_band, [0, 0], [PACKED, C0], target_memory=pl.MemorySpace.Mat)
        wt1 = pl.load(filter1, [0, 0], [K1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_mat = pl.load(bias1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias1_t = pl.move(bias1_mat, target_memory=pl.MemorySpace.Bias)
        scale1_l1 = pl.load(scale1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        scale1_fp = pl.move(scale1_l1, target_memory=pl.MemorySpace.LeftScale)
        acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
        for g in pl.range(K1 // TILE_K1):
            k0 = g * TILE_K1
            a1 = pl.tile.img2col(
                fm, 0, k0, shape=[ACC_M, TILE_K1], image_shape=[XROWS, WI, CI],
                kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
            )
            b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_K1, COUT1], target_memory=pl.MemorySpace.Right)
            if g == 0:
                acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
            else:
                acc1 = pl.tile.matmul_acc(acc1, a1, b1)
        return pl.tile.fixpipe_store(acc1, scale1_fp, mid, [0, 0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x_band: pl.Tensor[[PACKED, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        return self.kernel(x_band, filter1, bias1, scale1, mid)


@pl.program
class Conv2dPass1FusedAlloc:
    """First conv1 Acc tile with the same L1 residents as fused (fm, mid, wt1, wt2)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        filter2: pl.Tensor[[K2, COUT2], pl.INT8],
        bias2: pl.Tensor[[1, COUT2], pl.INT32],
        scale2: pl.Tensor[[1, COUT2], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        fm = pl.tile.create([C1 * XROWS * WI, C0], pl.INT8, target_memory=pl.MemorySpace.Mat)
        mid_l1 = pl.tile.create([M1, COUT1], pl.INT8, target_memory=pl.MemorySpace.Mat)
        wt1 = pl.load(filter1, [0, 0], [K1, COUT1], target_memory=pl.MemorySpace.Mat)
        wt2 = pl.load(filter2, [0, 0], [K2, COUT2], target_memory=pl.MemorySpace.Mat)
        bias1_mat = pl.load(bias1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias2_mat = pl.load(bias2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        bias1_t = pl.move(bias1_mat, target_memory=pl.MemorySpace.Bias)
        bias2_t = pl.move(bias2_mat, target_memory=pl.MemorySpace.Bias)
        scale1_l1 = pl.load(scale1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        scale2_l1 = pl.load(scale2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        scale1_fp = pl.move(scale1_l1, target_memory=pl.MemorySpace.LeftScale)
        scale2_fp = pl.move(scale2_l1, target_memory=pl.MemorySpace.LeftScale)
        for strip in pl.range(C1):
            src_row = strip * PLANE_IN
            dst_row = strip * XROWS * WI
            fm = pl.tile.gather_row(
                fm, x, dst_offset=[dst_row, 0], src_offset=[src_row, 0], shapes=[XROWS * WI, C0]
            )
        acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
        for g in pl.range(K1 // TILE_K1):
            k0 = g * TILE_K1
            a1 = pl.tile.img2col(
                fm, 0, k0, shape=[ACC_M, TILE_K1], image_shape=[XROWS, WI, CI],
                kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
            )
            b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_K1, COUT1], target_memory=pl.MemorySpace.Right)
            if g == 0:
                acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
            else:
                acc1 = pl.tile.matmul_acc(acc1, a1, b1)
        # Keep fused L1 residents live so fm sits at the fused address.
        # TSTORE first so requant cannot consume Acc before the GM dump.
        mid = pl.tile.fixpipe_store(acc1, scale1_fp, mid, [0, 0])
        mid_l1 = pl.tile.fixpipe_requant(mid_l1, acc1, scale1_fp, [0, 0])
        a2 = pl.tile.img2col(
            mid_l1, 0, 0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
            kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE2, STRIDE2],
        )
        b2 = pl.tile.extract(wt2, 0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
        acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
        acc2 = pl.tile.matmul_acc(acc2, a2, b2)
        y_keep = pl.tile.create([ACC_M, COUT2], pl.INT8, target_memory=pl.MemorySpace.Mat)
        y_keep = pl.tile.fixpipe_requant(y_keep, acc2, scale2_fp, [0, 0])
        return mid

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        filter2: pl.Tensor[[K2, COUT2], pl.INT8],
        bias2: pl.Tensor[[1, COUT2], pl.INT32],
        scale2: pl.Tensor[[1, COUT2], pl.UINT64],
        mid: pl.Out[pl.Tensor[[ACC_M, COUT1], pl.INT8]],
    ) -> pl.Tensor[[ACC_M, COUT1], pl.INT8]:
        return self.kernel(x, filter1, bias1, scale1, filter2, bias2, scale2, mid)


@pl.program
class Conv2dPass2:
    """Standalone conv2. Reloads each L1 ``mid`` window from GM."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        mid: pl.Tensor[[MID_PLANE, COUT1], pl.INT8],
        filter2: pl.Tensor[[K2, COUT2], pl.INT8],
        bias2: pl.Tensor[[1, COUT2], pl.INT32],
        scale2: pl.Tensor[[1, COUT2], pl.UINT64],
        y: pl.Out[pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]],
    ) -> pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]:
        wt2 = pl.load(filter2, [0, 0], [K2, COUT2], target_memory=pl.MemorySpace.Mat)
        bias2_mat = pl.load(bias2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        bias2_t = pl.move(bias2_mat, target_memory=pl.MemorySpace.Bias)
        scale2_l1 = pl.load(scale2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        scale2_fp = pl.move(scale2_l1, target_memory=pl.MemorySpace.LeftScale)

        fm = pl.tile.create([M1, COUT1], pl.INT8, target_memory=pl.MemorySpace.Mat)
        fm = pl.tile.gather_row(
            fm, mid, dst_offset=[0, 0], src_offset=[0, 0], shapes=[M1, COUT1]
        )
        for mt in pl.range(HB // M_TILE):
            m0 = mt * ACC_M
            acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for t in pl.range(K2 // TILE_K2):
                k0 = t * TILE_K2
                a2 = pl.tile.img2col(
                    fm, m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE2, STRIDE2],
                )
                b2 = pl.tile.extract(wt2, k0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
                if t == 0:
                    acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
                else:
                    acc2 = pl.tile.matmul_acc(acc2, a2, b2)
            y = pl.tile.fixpipe_store(acc2, scale2_fp, y, [m0, 0])

        for chunk in pl.range(1, HO2 // HB - 1):
            src_row = (HB * chunk - 1) * WO1
            fm = pl.tile.gather_row(
                fm, mid, dst_offset=[0, 0], src_offset=[src_row, 0], shapes=[M1, COUT1]
            )
            for mt in pl.range(HB // M_TILE):
                m0 = mt * ACC_M
                acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
                for t in pl.range(K2 // TILE_K2):
                    k0 = t * TILE_K2
                    a2 = pl.tile.img2col(
                        fm, m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
                        kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE2, STRIDE2],
                    )
                    b2 = pl.tile.extract(wt2, k0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
                    if t == 0:
                        acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
                    else:
                        acc2 = pl.tile.matmul_acc(acc2, a2, b2)
                y = pl.tile.fixpipe_store(acc2, scale2_fp, y, [chunk * M2 + m0, 0])

        src_row = A_LAST * WO1
        fm = pl.tile.gather_row(
            fm, mid, dst_offset=[0, 0], src_offset=[src_row, 0], shapes=[M1, COUT1]
        )
        for mt in pl.range(HB // M_TILE):
            m0 = mt * ACC_M
            acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for t in pl.range(K2 // TILE_K2):
                k0 = t * TILE_K2
                a2 = pl.tile.img2col(
                    fm, 2 * WO2 + m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, PAD], stride=[STRIDE2, STRIDE2],
                )
                b2 = pl.tile.extract(wt2, k0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
                if t == 0:
                    acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
                else:
                    acc2 = pl.tile.matmul_acc(acc2, a2, b2)
            y = pl.tile.fixpipe_store(acc2, scale2_fp, y, [(HO2 // HB - 1) * M2 + m0, 0])
        return y

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        mid: pl.Tensor[[MID_PLANE, COUT1], pl.INT8],
        filter2: pl.Tensor[[K2, COUT2], pl.INT8],
        bias2: pl.Tensor[[1, COUT2], pl.INT32],
        scale2: pl.Tensor[[1, COUT2], pl.UINT64],
        y: pl.Out[pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]],
    ) -> pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]:
        return self.kernel(mid, filter2, bias2, scale2, y)


def _load_pt(data_in: Path, name: str):
    import torch

    return torch.load(data_in / f"{name}.pt", weights_only=True)


def _report_diff(name: str, actual, expected) -> int:
    import torch

    n_bad = int((actual != expected).sum().item())
    max_abs = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max().item())
    print(f"    [{name}] mismatches={n_bad}/{actual.numel()} max_abs={max_abs}", flush=True)
    if n_bad:
        idx = (actual.flatten() != expected.flatten()).nonzero().flatten()[:8]
        for i in idx:
            print(
                f"        [{int(i)}] actual={int(actual.flatten()[i])} expected={int(expected.flatten()[i])}",
                flush=True,
            )
    return n_bad


def _report_mid_bands(actual, expected) -> None:
    a = actual.reshape(HO1, WO1, COUT1)
    e = expected.reshape(HO1, WO1, COUT1)
    for band in range(N_BANDS):
        h0 = band * HB
        h1 = h0 + HB
        sl = a[h0:h1]
        n_bad = int((sl != e[h0:h1]).sum().item())
        print(
            f"        mid H[{h0}:{h1}] mismatches={n_bad}/{sl.numel()}",
            flush=True,
        )


def _nd_from_nz(nz, rows: int, cols: int, inner_r: int, inner_c: int):
    import torch

    rr = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    cc = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    off = (
        (cc // inner_c) * rows * inner_c
        + (rr // inner_r) * (inner_r * inner_c)
        + (rr % inner_r) * inner_c
        + (cc % inner_c)
    )
    return nz.reshape(-1)[off]


def _analyze_mid_layout(actual, expected) -> None:
    n = actual.numel()
    print("        per-channel mismatches (first 8):", flush=True)
    for c in range(min(8, actual.shape[1])):
        n_bad = int((actual[:, c] != expected[:, c]).sum().item())
        print(f"            c={c}: {n_bad}/{actual.shape[0]}", flush=True)
    rows, cols = int(actual.shape[0]), int(actual.shape[1])
    for ir, ic in ((16, 32), (16, 16), (16, 8)):
        if rows % ir or cols % ic:
            continue
        decoded = _nd_from_nz(actual, rows, cols, ir, ic)
        n_bad = int((decoded != expected).sum().item())
        print(f"        NZ({ir}x{ic})->ND mismatches={n_bad}/{n}", flush=True)
    if rows == MID_PLANE:
        nhwc_img = expected.reshape(HO1, WO1, COUT1)
        nc1 = nhwc_img.reshape(HO1, WO1, COUT1 // C0, C0).permute(2, 0, 1, 3).reshape(-1, C0)
        as_c0 = actual.reshape(-1, C0)
        if as_c0.shape == nc1.shape:
            n_bad = int((as_c0 != nc1).sum().item())
            print(f"        actual as NC1HWC0 vs torch NC1HWC0 mismatches={n_bad}/{nc1.numel()}", flush=True)


def _make_capture(store: dict, key: str, require_equal: bool = True):
    def capture_y(actual, expected, **_kwargs):
        store[key] = actual.detach().clone()
        if key == "mid_gm":
            store["mid_torch"] = expected.detach().clone()
            n_bad = _report_diff("GM mid vs torch", actual, expected)
            _report_mid_bands(actual, expected)
            _analyze_mid_layout(actual, expected)
        elif key == "mid_tile":
            n_bad = _report_diff("first Acc tile vs torch mid[:112]", actual, expected)
            _analyze_mid_layout(actual, expected)
        elif key == "mid_packed":
            n_bad = _report_diff("packed-load first tile vs torch mid[:112]", actual, expected)
            _analyze_mid_layout(actual, expected)
        elif key == "mid_fusedalloc":
            n_bad = _report_diff("fused-L1 first tile vs torch mid[:112]", actual, expected)
            _analyze_mid_layout(actual, expected)
        else:
            n_bad = int((actual != expected).sum().item())
        if not require_equal or n_bad == 0:
            return True, ""
        return False, f"    mismatches vs expected: {n_bad}/{actual.numel()}"

    capture_y.__name__ = f"capture_{key}"
    return capture_y


def golden_conv1(tensors):
    x_nc1hwc0 = tensors["x"].reshape(C1, HI, WI, C0)
    x_nchw = x_nc1hwc0.permute(0, 3, 1, 2).reshape(1, CI, HI, WI)
    acc1 = _conv_i32(x_nchw, tensors["filter1"], tensors["bias1"], STRIDE1)
    mid = _requantize(acc1, tensors["scale1"])
    tensors["mid"][:] = mid.reshape(MID_PLANE, COUT1)


def golden_first_tile(tensors):
    buf = tensors["mid"].new_empty(MID_PLANE, COUT1)
    golden_conv1(
        {
            "x": tensors["x"],
            "filter1": tensors["filter1"],
            "bias1": tensors["bias1"],
            "scale1": tensors["scale1"],
            "mid": buf,
        }
    )
    tensors["mid"][:] = buf[:ACC_M]


def _pack_first_band(x):
    import torch

    parts = [x[c1 * PLANE_IN : c1 * PLANE_IN + XROWS * WI] for c1 in range(C1)]
    return torch.cat(parts, 0)


def _run_compare(platform: str, device_id: int, stage: str = "all") -> None:
    import torch
    from golden import TensorSpec, run

    runtime_cfg = dict(platform=platform, device_id=device_id)
    captured: dict[str, torch.Tensor] = {}

    print("[COMPARE] fused_conv2d (L1 mid, no GM bounce)", flush=True)
    fused = run(
        program=FusedConv2d,
        specs=build_tensor_specs(),
        golden_fn=golden_fused_conv2d,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        save_data=True,
        compare_fn={"y": _make_capture(captured, "y_fused")},
    )
    if not fused.passed:
        raise SystemExit(fused.error or "fused_conv2d failed")
    data_in = fused.work_dir / "data" / "in"

    print("[COMPARE] conv1 first Acc tile TSTORE vs torch mid[:112]", flush=True)
    tile = run(
        program=Conv2dPass1FirstTile,
        specs=[
            TensorSpec("x", [C1 * PLANE_IN, C0], torch.int8, init_value=_load_pt(data_in, "x")),
            TensorSpec("filter1", [K1, COUT1], torch.int8, init_value=_load_pt(data_in, "filter1")),
            TensorSpec("bias1", [1, COUT1], torch.int32, init_value=_load_pt(data_in, "bias1")),
            TensorSpec("scale1", [1, COUT1], torch.int64, init_value=_load_pt(data_in, "scale1")),
            TensorSpec("mid", [ACC_M, COUT1], torch.int8, is_output=True),
        ],
        golden_fn=golden_first_tile,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"mid": _make_capture(captured, "mid_tile", require_equal=False)},
    )
    if not tile.passed:
        raise SystemExit(tile.error or "conv1 first tile failed")

    x_full = _load_pt(data_in, "x")

    def golden_packed_tile(tensors):
        buf = tensors["mid"].new_empty(MID_PLANE, COUT1)
        golden_conv1(
            {
                "x": x_full,
                "filter1": tensors["filter1"],
                "bias1": tensors["bias1"],
                "scale1": tensors["scale1"],
                "mid": buf,
            }
        )
        tensors["mid"][:] = buf[:ACC_M]

    print("[COMPARE] conv1 first Acc tile pl.load packed band (no gather_row)", flush=True)
    packed = run(
        program=Conv2dPass1Packed,
        specs=[
            TensorSpec("x_band", [PACKED, C0], torch.int8, init_value=_pack_first_band(x_full)),
            TensorSpec("filter1", [K1, COUT1], torch.int8, init_value=_load_pt(data_in, "filter1")),
            TensorSpec("bias1", [1, COUT1], torch.int32, init_value=_load_pt(data_in, "bias1")),
            TensorSpec("scale1", [1, COUT1], torch.int64, init_value=_load_pt(data_in, "scale1")),
            TensorSpec("mid", [ACC_M, COUT1], torch.int8, is_output=True),
        ],
        golden_fn=golden_packed_tile,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"mid": _make_capture(captured, "mid_packed", require_equal=False)},
    )
    if not packed.passed:
        raise SystemExit(packed.error or "conv1 packed first tile failed")

    print("[COMPARE] conv1 first Acc tile with fused L1 residents", flush=True)
    fused_alloc = run(
        program=Conv2dPass1FusedAlloc,
        specs=[
            TensorSpec("x", [C1 * PLANE_IN, C0], torch.int8, init_value=x_full),
            TensorSpec("filter1", [K1, COUT1], torch.int8, init_value=_load_pt(data_in, "filter1")),
            TensorSpec("bias1", [1, COUT1], torch.int32, init_value=_load_pt(data_in, "bias1")),
            TensorSpec("scale1", [1, COUT1], torch.int64, init_value=_load_pt(data_in, "scale1")),
            TensorSpec("filter2", [K2, COUT2], torch.int8, init_value=_load_pt(data_in, "filter2")),
            TensorSpec("bias2", [1, COUT2], torch.int32, init_value=_load_pt(data_in, "bias2")),
            TensorSpec("scale2", [1, COUT2], torch.int64, init_value=_load_pt(data_in, "scale2")),
            TensorSpec("mid", [ACC_M, COUT1], torch.int8, is_output=True),
        ],
        golden_fn=golden_first_tile,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"mid": _make_capture(captured, "mid_fusedalloc", require_equal=False)},
    )
    if not fused_alloc.passed:
        raise SystemExit(fused_alloc.error or "conv1 fused-L1 first tile failed")

    if stage == "tile":
        mid_ref = x_full.new_empty(MID_PLANE, COUT1)
        golden_conv1(
            {
                "x": x_full,
                "filter1": _load_pt(data_in, "filter1"),
                "bias1": _load_pt(data_in, "bias1"),
                "scale1": _load_pt(data_in, "scale1"),
                "mid": mid_ref,
            }
        )
        ref = mid_ref[:ACC_M]
        n_g = _report_diff("tile gather vs torch", captured["mid_tile"], ref)
        n_p = _report_diff("tile packed vs torch", captured["mid_packed"], ref)
        n_f = _report_diff("tile fused-L1 vs torch", captured["mid_fusedalloc"], ref)
        if n_g or n_p or n_f:
            raise SystemExit(1)
        print("[COMPARE] PASS: first Acc tile matches torch (gather / packed / fused-L1)", flush=True)
        return

    print("[COMPARE] conv2d pass1 -> GM mid", flush=True)
    pass1 = run(
        program=Conv2dPass1,
        specs=[
            TensorSpec("x", [C1 * PLANE_IN, C0], torch.int8, init_value=_load_pt(data_in, "x")),
            TensorSpec("filter1", [K1, COUT1], torch.int8, init_value=_load_pt(data_in, "filter1")),
            TensorSpec("bias1", [1, COUT1], torch.int32, init_value=_load_pt(data_in, "bias1")),
            TensorSpec("scale1", [1, COUT1], torch.int64, init_value=_load_pt(data_in, "scale1")),
            TensorSpec("mid", [MID_PLANE, COUT1], torch.int8, is_output=True),
        ],
        golden_fn=golden_conv1,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"mid": _make_capture(captured, "mid_gm", require_equal=False)},
    )
    if not pass1.passed:
        raise SystemExit(pass1.error or "conv2d pass1 failed")

    def golden_from_fused(tensors):
        tensors["y"][:] = captured["y_fused"]

    pass2_specs_tail = [
        TensorSpec("filter2", [K2, COUT2], torch.int8, init_value=_load_pt(data_in, "filter2")),
        TensorSpec("bias2", [1, COUT2], torch.int32, init_value=_load_pt(data_in, "bias2")),
        TensorSpec("scale2", [1, COUT2], torch.int64, init_value=_load_pt(data_in, "scale2")),
        TensorSpec("y", [OUT_ROWS, COUT2], torch.int8, is_output=True),
    ]

    print("[COMPARE] conv2d pass2 <- torch mid (isolates gather/img2col)", flush=True)
    pass2_torch = run(
        program=Conv2dPass2,
        specs=[
            TensorSpec("mid", [MID_PLANE, COUT1], torch.int8, init_value=captured["mid_torch"]),
            *pass2_specs_tail,
        ],
        golden_fn=golden_from_fused,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"y": _make_capture(captured, "y_torch_mid", require_equal=False)},
    )
    if not pass2_torch.passed:
        raise SystemExit(pass2_torch.error or "conv2d pass2 (torch mid) failed")

    print("[COMPARE] conv2d pass2 <- GM mid", flush=True)
    pass2 = run(
        program=Conv2dPass2,
        specs=[
            TensorSpec("mid", [MID_PLANE, COUT1], torch.int8, init_value=captured["mid_gm"]),
            *pass2_specs_tail,
        ],
        golden_fn=golden_from_fused,
        runtime_cfg=runtime_cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"y": _make_capture(captured, "y_unfused", require_equal=False)},
    )
    if not pass2.passed:
        raise SystemExit(pass2.error or "conv2d pass2 failed")

    y_fused = captured["y_fused"]
    n_mid = _report_diff("summary GM mid vs torch", captured["mid_gm"], captured["mid_torch"])
    n_torch_mid = _report_diff("pass2(torch mid) vs fused y", captured["y_torch_mid"], y_fused)
    n_gm = _report_diff("pass2(GM mid) vs fused y", captured["y_unfused"], y_fused)
    if n_mid or n_torch_mid or n_gm:
        raise SystemExit(1)
    print("[COMPARE] PASS: fused matches two standalone conv2d passes (GM mid)", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a5sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--stage", type=str, default="all", choices=["all", "tile"],
        help="tile: fused + first Acc tile variants only (gather / packed / fused-L1)",
    )
    args = parser.parse_args()
    _run_compare(args.platform, args.device, args.stage)
