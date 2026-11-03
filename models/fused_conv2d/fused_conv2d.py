# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # needs PTOAS timg2col; public assembler pin does not parse it
"""Fused int8 3x3 conv pair with in-core L0C -> L1 VREQ8 (no GM mid).

The fusion point matches the AscendC MC62 / 5102 path: conv1 accumulates in
INT32 L0C, FIXPIPE VREQ8 writes INT8 ``mid`` in L1, and conv2's hardware
img2col reads that same L1 buffer. A GM or UB bounce of ``mid`` is a failed
fusion, not a fallback.

Geometry is the INT8 MC62 set (C0=32): x[1,64,288,112] -> mid[1,64,144,56]
-> y[1,64,144,56]. Packed UINT64 VREQ8 scale tables are an operator input.

PyPTO has no dav-5102 / 5102sim backend yet. Compile / CPU-sim checks use
a5sim (Ascend950) or a2a3sim (910B). That is a documented platform gap,
not a silent mapping of 5102 onto 950/910B.

Device/sim execute needs a PTOAS that parses ``pto.timg2col``; the public
assembler pin does not include that dialect yet.
"""

import pypto.language as pl


# model config -- INT8 / C0=32, matching fused_conv2d_geometry.h
C0 = 32
CI = 64
HI = 288
WI = 112
C1 = CI // C0
COUT1 = 64
STRIDE1 = 2
HO1 = 144
WO1 = 56
COUT2 = 64
STRIDE2 = 1
HO2 = 144
WO2 = 56
KH = 3
KW = 3
PAD = 1
MID_C1 = COUT1 // C0
K1 = C1 * KH * KW * C0
K2 = MID_C1 * KH * KW * C0

HB = 18
MID_ROWS = HB + 2
XROWS = STRIDE1 * (MID_ROWS - 1) + KH
M1 = MID_ROWS * WO1
M2 = HB * WO2
M_TILE = 2
ACC_M = M_TILE * WO1
TILE_K1 = 64
TILE_K2 = 32
PLANE_IN = HI * WI
A_LAST = HO1 - MID_ROWS
OUT_ROWS = HO2 * WO2


@pl.program
class FusedConv2d:
    """In-core fused conv1+conv2. One InCore function, L1 ``fm`` + ``mid``."""

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
        y: pl.Out[pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]],
    ) -> pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]:
        # L1 feature-map band (NC1HWC0-as-2D) and the fused INT8 intermediate.
        # mid is destination-passing: FIXPIPE writes it, img2col reads it.
        fm = pl.tile.create([C1 * XROWS * WI, C0], pl.INT8, target_memory=pl.MemorySpace.Mat)
        mid = pl.tile.create([M1, COUT1], pl.INT8, target_memory=pl.MemorySpace.Mat)
        wt1 = pl.load(filter1, [0, 0], [K1, COUT1], target_memory=pl.MemorySpace.Mat)
        wt2 = pl.load(filter2, [0, 0], [K2, COUT2], target_memory=pl.MemorySpace.Mat)
        bias1_mat = pl.load(bias1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        bias2_mat = pl.load(bias2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        bias1_t = pl.move(bias1_mat, target_memory=pl.MemorySpace.Bias)
        bias2_t = pl.move(bias2_mat, target_memory=pl.MemorySpace.Bias)
        scale1_l1 = pl.load(scale1, [0, 0], [1, COUT1], target_memory=pl.MemorySpace.Mat)
        scale2_l1 = pl.load(scale2, [0, 0], [1, COUT2], target_memory=pl.MemorySpace.Mat)
        # PTOAS tinsert/tstore fp requires loc=scaling (Fixpipe FBUF). UINT64
        # LeftScale lowers to that loc without the MX fractal-32 layout.
        scale1_fp = pl.move(scale1_l1, target_memory=pl.MemorySpace.LeftScale)
        scale2_fp = pl.move(scale2_l1, target_memory=pl.MemorySpace.LeftScale)

        # First / interior / last bands are unrolled so img2col padding stays ConstInt.
        # First band: image-top clip becomes conv1/conv2 pad_top.
        for strip in pl.range(C1):
            src_row = strip * PLANE_IN
            dst_row = strip * XROWS * WI
            fm = pl.tile.gather_row(
                fm, x, dst_offset=[dst_row, 0], src_offset=[src_row, 0], shapes=[XROWS * WI, C0]
            )
        for mt in pl.range(MID_ROWS // M_TILE):
            m0 = mt * ACC_M
            acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for g in pl.range(K1 // TILE_K1):
                k0 = g * TILE_K1
                a1 = pl.tile.img2col(
                    fm, m0, k0, shape=[ACC_M, TILE_K1], image_shape=[XROWS, WI, CI],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE1, STRIDE1],
                )
                b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_K1, COUT1], target_memory=pl.MemorySpace.Right)
                if g == 0:
                    acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
                else:
                    acc1 = pl.tile.matmul_acc(acc1, a1, b1)
            mid = pl.tile.fixpipe_requant(mid, acc1, scale1_fp, [m0, 0])
        for mt in pl.range(HB // M_TILE):
            m0 = mt * ACC_M
            acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for t in pl.range(K2 // TILE_K2):
                k0 = t * TILE_K2
                a2 = pl.tile.img2col(
                    mid, m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
                    kernel=[KH, KW], padding=[PAD, PAD, PAD, 0], stride=[STRIDE2, STRIDE2],
                )
                b2 = pl.tile.extract(wt2, k0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
                if t == 0:
                    acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
                else:
                    acc2 = pl.tile.matmul_acc(acc2, a2, b2)
            y = pl.tile.fixpipe_store(acc2, scale2_fp, y, [m0, 0])

        # Interior bands: a = HB*chunk-1, b = STRIDE1*a-1, no extra pad clip.
        for chunk in pl.range(1, HO2 // HB - 1):
            b = STRIDE1 * (HB * chunk - 1) - 1
            for strip in pl.range(C1):
                src_row = WI * b + strip * PLANE_IN
                dst_row = strip * XROWS * WI
                fm = pl.tile.gather_row(
                    fm, x, dst_offset=[dst_row, 0], src_offset=[src_row, 0], shapes=[XROWS * WI, C0]
                )
            for mt in pl.range(MID_ROWS // M_TILE):
                m0 = mt * ACC_M
                acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
                for g in pl.range(K1 // TILE_K1):
                    k0 = g * TILE_K1
                    a1 = pl.tile.img2col(
                        fm, m0, k0, shape=[ACC_M, TILE_K1], image_shape=[XROWS, WI, CI],
                        kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE1, STRIDE1],
                    )
                    b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_K1, COUT1], target_memory=pl.MemorySpace.Right)
                    if g == 0:
                        acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
                    else:
                        acc1 = pl.tile.matmul_acc(acc1, a1, b1)
                mid = pl.tile.fixpipe_requant(mid, acc1, scale1_fp, [m0, 0])
            for mt in pl.range(HB // M_TILE):
                m0 = mt * ACC_M
                acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
                for t in pl.range(K2 // TILE_K2):
                    k0 = t * TILE_K2
                    a2 = pl.tile.img2col(
                        mid, m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
                        kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE2, STRIDE2],
                    )
                    b2 = pl.tile.extract(wt2, k0, 0, shape=[TILE_K2, COUT2], target_memory=pl.MemorySpace.Right)
                    if t == 0:
                        acc2 = pl.tile.matmul_bias(a2, b2, bias2_t)
                    else:
                        acc2 = pl.tile.matmul_acc(acc2, a2, b2)
                y = pl.tile.fixpipe_store(acc2, scale2_fp, y, [chunk * M2 + m0, 0])

        # Last band: image-bottom clip becomes conv2 pad_top and pad_bottom.
        b = STRIDE1 * A_LAST - 1
        for strip in pl.range(C1):
            src_row = WI * b + strip * PLANE_IN
            dst_row = strip * XROWS * WI
            fm = pl.tile.gather_row(
                fm, x, dst_offset=[dst_row, 0], src_offset=[src_row, 0], shapes=[XROWS * WI, C0]
            )
        for mt in pl.range(MID_ROWS // M_TILE):
            m0 = mt * ACC_M
            acc1 = pl.tile.create([ACC_M, COUT1], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for g in pl.range(K1 // TILE_K1):
                k0 = g * TILE_K1
                a1 = pl.tile.img2col(
                    fm, m0, k0, shape=[ACC_M, TILE_K1], image_shape=[XROWS, WI, CI],
                    kernel=[KH, KW], padding=[PAD, PAD, 0, 0], stride=[STRIDE1, STRIDE1],
                )
                b1 = pl.tile.extract(wt1, k0, 0, shape=[TILE_K1, COUT1], target_memory=pl.MemorySpace.Right)
                if g == 0:
                    acc1 = pl.tile.matmul_bias(a1, b1, bias1_t)
                else:
                    acc1 = pl.tile.matmul_acc(acc1, a1, b1)
            mid = pl.tile.fixpipe_requant(mid, acc1, scale1_fp, [m0, 0])
        for mt in pl.range(HB // M_TILE):
            m0 = mt * ACC_M
            acc2 = pl.tile.create([ACC_M, COUT2], pl.INT32, target_memory=pl.MemorySpace.Acc)
            for t in pl.range(K2 // TILE_K2):
                k0 = t * TILE_K2
                a2 = pl.tile.img2col(
                    mid, 2 * WO2 + m0, k0, shape=[ACC_M, TILE_K2], image_shape=[MID_ROWS, WO1, COUT1],
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
        x: pl.Tensor[[C1 * PLANE_IN, C0], pl.INT8],
        filter1: pl.Tensor[[K1, COUT1], pl.INT8],
        bias1: pl.Tensor[[1, COUT1], pl.INT32],
        scale1: pl.Tensor[[1, COUT1], pl.UINT64],
        filter2: pl.Tensor[[K2, COUT2], pl.INT8],
        bias2: pl.Tensor[[1, COUT2], pl.INT32],
        scale2: pl.Tensor[[1, COUT2], pl.UINT64],
        y: pl.Out[pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]],
    ) -> pl.Tensor[[OUT_ROWS, COUT2], pl.INT8]:
        return self.kernel(x, filter1, bias1, scale1, filter2, bias2, scale2, y)


def _randint(shape, low, high, seed):
    import torch

    def init():
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(low, high, shape, dtype=torch.int8, generator=generator)

    return init


def _float19(values):
    import torch

    bits = values.contiguous().view(torch.int32)
    return bits.bitwise_and(-8192).view(values.dtype)


def pack_vreq8(scale, offset, signed=True):
    """Pack per-channel VREQ8 table entries (AscendC L0C2L1_VREQ8 encoding).

    [31:13] float19 scale, [45:37] signed 9-bit offset, [46] signed saturate.
    """
    import torch

    scale19 = _float19(scale.to(torch.float32))
    bits = scale19.contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xFFFFFFFF)
    packed = bits | ((offset.to(torch.int64) & 0x1FF) << 37)
    if signed:
        packed = packed | (1 << 46)
    return packed.to(torch.int64)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    bias1_data = (torch.arange(COUT1, dtype=torch.int32).remainder(7) - 3).reshape(1, COUT1)
    bias2_data = (torch.arange(COUT2, dtype=torch.int32).remainder(5) - 2).reshape(1, COUT2)
    scale1_f = _float19(torch.full([COUT1], 1.0 / 16.0, dtype=torch.float32))
    scale2_f = _float19(torch.full([COUT2], 1.0 / 32.0, dtype=torch.float32))
    offset1 = torch.arange(COUT1, dtype=torch.int32).remainder(3) - 1
    offset2 = 1 - torch.arange(COUT2, dtype=torch.int32).remainder(3)
    return [
        TensorSpec("x", [C1 * PLANE_IN, C0], torch.int8, init_value=_randint([C1 * PLANE_IN, C0], -3, 4, 1)),
        TensorSpec("filter1", [K1, COUT1], torch.int8, init_value=_randint([K1, COUT1], -2, 3, 2)),
        TensorSpec("bias1", [1, COUT1], torch.int32, init_value=bias1_data),
        TensorSpec("scale1", [1, COUT1], torch.int64, init_value=pack_vreq8(scale1_f, offset1).reshape(1, COUT1)),
        TensorSpec("filter2", [K2, COUT2], torch.int8, init_value=_randint([K2, COUT2], -2, 3, 3)),
        TensorSpec("bias2", [1, COUT2], torch.int32, init_value=bias2_data),
        TensorSpec("scale2", [1, COUT2], torch.int64, init_value=pack_vreq8(scale2_f, offset2).reshape(1, COUT2)),
        TensorSpec("y", [OUT_ROWS, COUT2], torch.int8, is_output=True),
    ]


def _unpack_vreq8(packed):
    import torch

    p = packed.reshape(-1).to(torch.int64)
    scale_bits = (p & 0xFFFFFFFF).to(torch.int32)
    scale = scale_bits.view(torch.float32)
    offset = ((p >> 37) & 0x1FF).to(torch.int32)
    offset = torch.where(offset >= 256, offset - 512, offset)
    return scale, offset


def _im2col_c1khkwc0(x_nchw, stride):
    """Im2col with K walked as (c1, kh, kw, c0), matching hardware load3d / TIMG2COL."""
    import torch
    import torch.nn.functional as F

    cin = x_nchw.shape[1]
    c1n = cin // C0
    patches = F.unfold(x_nchw.float(), kernel_size=KH, padding=PAD, stride=stride)
    length = patches.shape[-1]
    patches = patches.reshape(1, c1n, C0, KH, KW, length)
    patches = patches.permute(0, 5, 1, 3, 4, 2).reshape(1, length, c1n * KH * KW * C0)
    return patches.to(dtype=torch.int32)


def _conv_i32(x_nchw, weight_kn, bias, stride):
    import torch

    patches = _im2col_c1khkwc0(x_nchw, stride)
    weight = weight_kn.to(dtype=torch.int32)
    return patches @ weight + bias.reshape(1, 1, -1)


def _requantize(acc, packed_scale):
    """Match pto-isa CPU VREQ8: nearbyint(fp32(acc) * scale + offset), half-to-even."""
    import torch

    scale, offset = _unpack_vreq8(packed_scale)
    view = [1] * (acc.dim() - 1) + [-1]
    scaled = acc.to(torch.float32) * scale.reshape(view) + offset.to(torch.float32).reshape(view)
    rounded = torch.round(scaled).to(torch.int32)
    return torch.clamp(rounded, -128, 127).to(torch.int8)


def golden_fused_conv2d(tensors):
    # x is NC1HWC0 flattened as [C1*HI*WI, C0]; restore NCHW for the torch ref.
    x_nc1hwc0 = tensors["x"].reshape(C1, HI, WI, C0)
    x_nchw = x_nc1hwc0.permute(0, 3, 1, 2).reshape(1, CI, HI, WI)
    acc1 = _conv_i32(x_nchw, tensors["filter1"], tensors["bias1"], STRIDE1)
    mid = _requantize(acc1, tensors["scale1"])
    mid_nchw = mid.reshape(1, HO1, WO1, COUT1).permute(0, 3, 1, 2)
    acc2 = _conv_i32(mid_nchw, tensors["filter2"], tensors["bias2"], STRIDE2)
    out = _requantize(acc2, tensors["scale2"])
    tensors["y"][:] = out.reshape(OUT_ROWS, COUT2)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a5sim",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=FusedConv2d,
        specs=build_tensor_specs(),
        golden_fn=golden_fused_conv2d,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
