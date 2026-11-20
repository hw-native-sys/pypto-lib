# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DeepSeek-V4.1 shared MXFP8 W13, SwiGLU requantization, and W2."""

import sys
from math import gcd
from pathlib import Path

import pypto.language as pl

from pypto.runtime import pto_isa_include_dir

import config


def _parse_int_option(flag, default):
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(flag + "="):
            return int(argument.split("=", 1)[1])
    return default


if __name__ == "__main__":
    config.set_moe_tokens(_parse_int_option("--tokens", 80))


from config import FLASH as M

from config import MOE_TOKENS


# model config
T = MOE_TOKENS
T_PAD = ((T + 15) // 16) * 16
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = M.weight_block_size[0]

# tiling
GATE_UP_N = 2 * MOE_INTER
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP
W13_M_TILE = 80
W13_N_TILE = 256
W13_K_TILE = 128
W13_GROUPS = (T_PAD + W13_M_TILE - 1) // W13_M_TILE
W13_ROWS = W13_GROUPS * W13_M_TILE
W13_PAD_ROWS = 16  # staging granularity when the token block is not a whole tile
SWIGLU_ROW_TILE = 16
SWIGLU_COL_TILE = 2 * MX_GROUP
SWIGLU_SCALE_TILE = 2 * SWIGLU_ROW_TILE
# Each scale slot reserves one full register of 32-bit words.
SWIGLU_SCALE_REGISTER_TILE = 64
SWIGLU_CHAIN_TILE = 4
SWIGLU_WORKER_TILE = 8 if T > 80 else 20
W13_WORKER_TILE = 4 if T > 80 else GATE_UP_N // W13_N_TILE
W2_WORKER_TILE = 4 if T > 80 else D // 256
W2_M_TILE = gcd(T_PAD, 256) if T > 80 else gcd(T_PAD, W13_M_TILE)
W2_N_TILE = 256
W2_K_TILE = 128 if T > 80 else 256
W2_K_STAGE = 2

assert T_PAD % W13_PAD_ROWS == 0
assert T_PAD % 16 == 0
assert GATE_UP_N % W13_N_TILE == 0
assert MOE_INTER % W2_K_TILE == 0
assert D % W2_N_TILE == 0
assert W2_M_TILE % 16 == 0


@pl.jit.extern(
    core_type="aic",
    source=Path(__file__).parent / "kernels" / "expert_shared_w13.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _shared_w13_mm(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
    shared_w1: pl.Tensor,
    shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor,
    shared_w3_scale: pl.Tensor,
    gate_up: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND MXFP8 activations with ND weights and resident MX scales."""
    ...


@pl.jit.inline(auto_scope=False)
def _shared_swiglu_prefetch(
    gate_up: pl.Tensor,
    gate: pl.Tile[[2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], pl.BF16, pl.Mem.Vec],
    up: pl.Tile[[2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], pl.BF16, pl.Mem.Vec],
    work: pl.Scalar[pl.INDEX],
    slot: pl.Scalar[pl.INDEX],
):
    row = work // (MOE_INTER // SWIGLU_COL_TILE) * SWIGLU_ROW_TILE
    col = work % (MOE_INTER // SWIGLU_COL_TILE) * SWIGLU_COL_TILE
    up_col = col + MOE_INTER
    slot_row = slot * SWIGLU_ROW_TILE
    gate = pl.tile.gather_row(gate, gate_up, [slot_row, 0], [row, col], [SWIGLU_ROW_TILE, SWIGLU_COL_TILE])
    up = pl.tile.gather_row(up, gate_up, [slot_row, 0], [row, up_col], [SWIGLU_ROW_TILE, SWIGLU_COL_TILE])
    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
    return gate, up


@pl.jit.inline(auto_scope=False)
def _shared_swiglu_compute(
    gate: pl.Tile[[2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], pl.BF16, pl.Mem.Vec],
    up: pl.Tile[[2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], pl.BF16, pl.Mem.Vec],
    payload: pl.Tile[[2 * SWIGLU_ROW_TILE * SWIGLU_COL_TILE], pl.INT8, pl.Mem.Vec],
    scales: pl.Tile[[2 * SWIGLU_SCALE_REGISTER_TILE], pl.UINT32, pl.Mem.Vec],
    slot: pl.Scalar[pl.INDEX],
):
    gate_flat = pl.reshape(gate, [2 * SWIGLU_ROW_TILE * SWIGLU_COL_TILE])
    up_flat = pl.reshape(up, [2 * SWIGLU_ROW_TILE * SWIGLU_COL_TILE])
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=SWIGLU_COL_TILE)
        lo = pl.reg.broadcast(0, lanes=SWIGLU_COL_TILE, dtype=pl.UINT32)
        hi = pl.reg.broadcast(MX_GROUP, lanes=SWIGLU_COL_TILE, dtype=pl.UINT32)
        bias = pl.reg.broadcast(254, lanes=SWIGLU_COL_TILE, dtype=pl.INT32)
        scale_lane = pl.reg.index_gen(0, lanes=SWIGLU_SCALE_REGISTER_TILE, dtype=pl.UINT32)
        zero = pl.reg.broadcast(0, lanes=SWIGLU_SCALE_REGISTER_TILE, dtype=pl.UINT32)
        for group, (packed_scales,) in pl.hardware_loop(SWIGLU_ROW_TILE // SWIGLU_CHAIN_TILE, init_values=(zero,)):
            base = slot * SWIGLU_ROW_TILE + group * SWIGLU_CHAIN_TILE
            off0 = base * SWIGLU_COL_TILE
            off1 = off0 + SWIGLU_COL_TILE
            off2 = off1 + SWIGLU_COL_TILE
            off3 = off2 + SWIGLU_COL_TILE

            # Clamp gate and up and evaluate SiLU(gate) * up in FP32.
            g0_bf = pl.reg.load(gate_flat, off0, lanes=SWIGLU_COL_TILE)
            g1_bf = pl.reg.load(gate_flat, off1, lanes=SWIGLU_COL_TILE)
            g2_bf = pl.reg.load(gate_flat, off2, lanes=SWIGLU_COL_TILE)
            g3_bf = pl.reg.load(gate_flat, off3, lanes=SWIGLU_COL_TILE)
            g0 = pl.reg.convert(g0_bf, dtype=pl.FP32)
            g1 = pl.reg.convert(g1_bf, dtype=pl.FP32)
            g2 = pl.reg.convert(g2_bf, dtype=pl.FP32)
            g3 = pl.reg.convert(g3_bf, dtype=pl.FP32)
            u0_bf = pl.reg.load(up_flat, off0, lanes=SWIGLU_COL_TILE)
            u1_bf = pl.reg.load(up_flat, off1, lanes=SWIGLU_COL_TILE)
            u2_bf = pl.reg.load(up_flat, off2, lanes=SWIGLU_COL_TILE)
            u3_bf = pl.reg.load(up_flat, off3, lanes=SWIGLU_COL_TILE)
            u0 = pl.reg.convert(u0_bf, dtype=pl.FP32)
            u1 = pl.reg.convert(u1_bf, dtype=pl.FP32)
            u2 = pl.reg.convert(u2_bf, dtype=pl.FP32)
            u3 = pl.reg.convert(u3_bf, dtype=pl.FP32)
            g0 = pl.reg.minimums(g0, SWIGLU_LIMIT, full)
            g1 = pl.reg.minimums(g1, SWIGLU_LIMIT, full)
            g2 = pl.reg.minimums(g2, SWIGLU_LIMIT, full)
            g3 = pl.reg.minimums(g3, SWIGLU_LIMIT, full)
            u0 = pl.reg.minimums(u0, SWIGLU_LIMIT, full)
            u1 = pl.reg.minimums(u1, SWIGLU_LIMIT, full)
            u2 = pl.reg.minimums(u2, SWIGLU_LIMIT, full)
            u3 = pl.reg.minimums(u3, SWIGLU_LIMIT, full)
            u0 = pl.reg.maximums(u0, -SWIGLU_LIMIT, full)
            u1 = pl.reg.maximums(u1, -SWIGLU_LIMIT, full)
            u2 = pl.reg.maximums(u2, -SWIGLU_LIMIT, full)
            u3 = pl.reg.maximums(u3, -SWIGLU_LIMIT, full)
            s0 = pl.reg.muls(g0, -1.0, full)
            s1 = pl.reg.muls(g1, -1.0, full)
            s2 = pl.reg.muls(g2, -1.0, full)
            s3 = pl.reg.muls(g3, -1.0, full)
            s0 = pl.reg.exp(s0, full)
            s1 = pl.reg.exp(s1, full)
            s2 = pl.reg.exp(s2, full)
            s3 = pl.reg.exp(s3, full)
            s0 = pl.reg.adds(s0, 1.0, full)
            s1 = pl.reg.adds(s1, 1.0, full)
            s2 = pl.reg.adds(s2, 1.0, full)
            s3 = pl.reg.adds(s3, 1.0, full)
            h0 = pl.reg.div(g0, s0, full)
            h1 = pl.reg.div(g1, s1, full)
            h2 = pl.reg.div(g2, s2, full)
            h3 = pl.reg.div(g3, s3, full)
            h0 = pl.reg.mul(h0, u0, full)
            h1 = pl.reg.mul(h1, u1, full)
            h2 = pl.reg.mul(h2, u2, full)
            h3 = pl.reg.mul(h3, u3, full)

            # Round each group maximum upward to an E8M0 exponent.
            m0 = pl.reg.abs(h0, full)
            m1 = pl.reg.abs(h1, full)
            m2 = pl.reg.abs(h2, full)
            m3 = pl.reg.abs(h3, full)
            peak0 = pl.reg.reduce_max(m0, full, group_size=MX_GROUP)
            peak1 = pl.reg.reduce_max(m1, full, group_size=MX_GROUP)
            peak2 = pl.reg.reduce_max(m2, full, group_size=MX_GROUP)
            peak3 = pl.reg.reduce_max(m3, full, group_size=MX_GROUP)
            m0 = pl.reg.group_broadcast(peak0, lanes=SWIGLU_COL_TILE)
            m1 = pl.reg.group_broadcast(peak1, lanes=SWIGLU_COL_TILE)
            m2 = pl.reg.group_broadcast(peak2, lanes=SWIGLU_COL_TILE)
            m3 = pl.reg.group_broadcast(peak3, lanes=SWIGLU_COL_TILE)
            m0 = pl.reg.maximums(m0, 1e-4, full)
            m1 = pl.reg.maximums(m1, 1e-4, full)
            m2 = pl.reg.maximums(m2, 1e-4, full)
            m3 = pl.reg.maximums(m3, 1e-4, full)
            m0 = pl.reg.muls(m0, 1.0 / 448.0, full)
            m1 = pl.reg.muls(m1, 1.0 / 448.0, full)
            m2 = pl.reg.muls(m2, 1.0 / 448.0, full)
            m3 = pl.reg.muls(m3, 1.0 / 448.0, full)
            e0 = pl.reg.interpret_cast(m0, dtype=pl.INT32)
            e1 = pl.reg.interpret_cast(m1, dtype=pl.INT32)
            e2 = pl.reg.interpret_cast(m2, dtype=pl.INT32)
            e3 = pl.reg.interpret_cast(m3, dtype=pl.INT32)
            e0 = pl.reg.adds(e0, 8388607, full)
            e1 = pl.reg.adds(e1, 8388607, full)
            e2 = pl.reg.adds(e2, 8388607, full)
            e3 = pl.reg.adds(e3, 8388607, full)
            e0 = pl.reg.shift_rights(e0, 23, full)
            e1 = pl.reg.shift_rights(e1, 23, full)
            e2 = pl.reg.shift_rights(e2, 23, full)
            e3 = pl.reg.shift_rights(e3, 23, full)

            # Pack the two exponent bytes per row in MX_A_ZZ order.
            lo_code0 = pl.reg.gather_lanes(e0, lo)
            lo_code1 = pl.reg.gather_lanes(e1, lo)
            lo_code2 = pl.reg.gather_lanes(e2, lo)
            lo_code3 = pl.reg.gather_lanes(e3, lo)
            hi_code0 = pl.reg.gather_lanes(e0, hi)
            hi_code1 = pl.reg.gather_lanes(e1, hi)
            hi_code2 = pl.reg.gather_lanes(e2, hi)
            hi_code3 = pl.reg.gather_lanes(e3, hi)
            hi_code0 = pl.reg.shift_lefts(hi_code0, 8, full)
            hi_code1 = pl.reg.shift_lefts(hi_code1, 8, full)
            hi_code2 = pl.reg.shift_lefts(hi_code2, 8, full)
            hi_code3 = pl.reg.shift_lefts(hi_code3, 8, full)
            codes0 = pl.reg.bitwise_or(lo_code0, hi_code0, full)
            codes1 = pl.reg.bitwise_or(lo_code1, hi_code1, full)
            codes2 = pl.reg.bitwise_or(lo_code2, hi_code2, full)
            codes3 = pl.reg.bitwise_or(lo_code3, hi_code3, full)
            # Each aligned 32-bit word carries the exponents of two adjacent rows.
            codes1 = pl.reg.shift_lefts(codes1, 16, full)
            codes3 = pl.reg.shift_lefts(codes3, 16, full)
            pair0 = pl.reg.bitwise_or(codes0, codes1, full)
            pair1 = pl.reg.bitwise_or(codes2, codes3, full)
            pair0 = pl.reg.interpret_cast(pair0, dtype=pl.UINT32)
            pair1 = pl.reg.interpret_cast(pair1, dtype=pl.UINT32)
            word0 = pl.cast(group * 2, pl.UINT32)
            word1 = pl.cast(group * 2 + 1, pl.UINT32)
            select0 = pl.reg.compares(scale_lane, word0, full, predicate="eq")
            select1 = pl.reg.compares(scale_lane, word1, full, predicate="eq")
            packed_scales = pl.reg.select(select0, pair0, packed_scales)
            packed_scales = pl.reg.select(select1, pair1, packed_scales)

            # Apply the reciprocal scale in FP32.
            r0 = pl.reg.sub(bias, e0, full)
            r1 = pl.reg.sub(bias, e1, full)
            r2 = pl.reg.sub(bias, e2, full)
            r3 = pl.reg.sub(bias, e3, full)
            r0 = pl.reg.shift_lefts(r0, 23, full)
            r1 = pl.reg.shift_lefts(r1, 23, full)
            r2 = pl.reg.shift_lefts(r2, 23, full)
            r3 = pl.reg.shift_lefts(r3, 23, full)
            recip0 = pl.reg.interpret_cast(r0, dtype=pl.FP32)
            recip1 = pl.reg.interpret_cast(r1, dtype=pl.FP32)
            recip2 = pl.reg.interpret_cast(r2, dtype=pl.FP32)
            recip3 = pl.reg.interpret_cast(r3, dtype=pl.FP32)
            h0 = pl.reg.mul(h0, recip0, full)
            h1 = pl.reg.mul(h1, recip1, full)
            h2 = pl.reg.mul(h2, recip2, full)
            h3 = pl.reg.mul(h3, recip3, full)

            # Clamp to the E4M3 range and store the packed payload.
            h0 = pl.reg.maximums(h0, -448.0, full)
            h1 = pl.reg.maximums(h1, -448.0, full)
            h2 = pl.reg.maximums(h2, -448.0, full)
            h3 = pl.reg.maximums(h3, -448.0, full)
            h0 = pl.reg.minimums(h0, 448.0, full)
            h1 = pl.reg.minimums(h1, 448.0, full)
            h2 = pl.reg.minimums(h2, 448.0, full)
            h3 = pl.reg.minimums(h3, 448.0, full)
            q0 = pl.reg.convert(h0, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
            q1 = pl.reg.convert(h1, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
            q2 = pl.reg.convert(h2, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
            q3 = pl.reg.convert(h3, dtype=pl.FP8E4M3FN, rounding="R", saturate=True)
            bytes0 = pl.reg.interpret_cast(q0, dtype=pl.INT8)
            bytes1 = pl.reg.interpret_cast(q1, dtype=pl.INT8)
            bytes2 = pl.reg.interpret_cast(q2, dtype=pl.INT8)
            bytes3 = pl.reg.interpret_cast(q3, dtype=pl.INT8)
            _payload0 = pl.reg.store(payload, off0, bytes0)
            _payload1 = pl.reg.store(payload, off1, bytes1)
            _payload2 = pl.reg.store(payload, off2, bytes2)
            _payload3 = pl.reg.store(payload, off3, bytes3)
            packed_scales = pl.yield_(packed_scales)
        scale_offset = slot * SWIGLU_SCALE_REGISTER_TILE
        _scale_write = pl.reg.store(scales, scale_offset, packed_scales)
    return payload, scales


@pl.jit.inline
def _w13_padded_inputs(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
):
    """Pad payload and packed scales to whole W13 row tiles."""
    x_padded = pl.create_tensor([W13_ROWS, D], dtype=pl.FP8E4M3FN)
    scale_padded = pl.create_tensor([1, W13_ROWS * K_SCALE], dtype=pl.FP8E8M0)
    scale_col = W13_PAD_ROWS * K_SCALE
    pad_workers = pl.min(T_PAD // W13_PAD_ROWS, 16)
    with pl.spmd(pad_workers, name_hint="sh_w13_pad"):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, T_PAD // W13_PAD_ROWS, pad_workers):
            row0 = block * W13_PAD_ROWS
            x_rows = pl.load(x_local, [row0, 0], [W13_PAD_ROWS, D])
            x_padded = pl.store(x_rows, [row0, 0], x_padded)
            scale_rows = pl.load(x_local_scale, [0, row0 * K_SCALE], [1, scale_col])
            scale_padded = pl.store(scale_rows, [0, row0 * K_SCALE], scale_padded)
    return x_padded, scale_padded


@pl.jit.inline
def _w13_direct_inputs(
    x_local: pl.Tensor,
    x_local_scale: pl.Tensor,
):
    """The physical token block already spans whole W13 row tiles."""
    return x_local, x_local_scale


# Selected at import time so the fused program traces exactly one input path.
_W13_INPUTS = _w13_direct_inputs if W13_ROWS == T_PAD else _w13_padded_inputs


@pl.jit.inline
def _project_w13_pypto(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    gate_up: pl.Out[pl.Tensor],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    projection: pl.constexpr,
) -> pl.Tensor:
    """Accumulate one W1 or W3 panel in FP32 and store its active rows in BF16."""
    x_scale_zz = pl.tensor.view(x_scale, [W13_ROWS, K_SCALE], layout=pl.MX_A_ZZ)
    acc = pl.create_tile([W13_M_TILE, W13_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
    for k in pl.range(0, D, W13_K_TILE):
        scale_k = k // MX_GROUP
        a = pl.tile.load(x, [row, k], [W13_M_TILE, W13_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.tile.load(weight, [k, col], [W13_K_TILE, W13_N_TILE], target_memory=pl.Mem.Mat)
        # Stream each K128 E8M0 fragment into Mat for the MX matmul.
        sa = pl.tile.load(x_scale_zz, [row, scale_k], [W13_M_TILE, W13_K_TILE // MX_GROUP])
        sb = pl.tile.load(weight_scale, [scale_k, col], [W13_K_TILE // MX_GROUP, W13_N_TILE])
        if k == 0:
            acc = pl.tile.matmul_mx(a, sa, b, sb)
        else:
            acc = pl.tile.matmul_mx_acc(acc, a, sa, b, sb)
    active = pl.min(T_PAD - row, W13_M_TILE)
    out = pl.tile.set_validshape(acc, active, W13_N_TILE)
    out_col = projection * MOE_INTER + col
    return pl.tile.store(out, [row, out_col], gate_up)


@pl.jit.inline
def shared_w13_pypto(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * K_SCALE], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    gate_up: pl.Out[pl.Tensor[[T_PAD, GATE_UP_N], pl.BF16]],
) -> pl.Tensor:
    """Project MXFP8 activations into concatenated BF16 W1 and W3 panels."""
    x, x_scale = _W13_INPUTS(x_local, x_local_scale)
    n_tiles = MOE_INTER // W13_N_TILE
    workers = pl.min(n_tiles, W13_WORKER_TILE)
    with pl.spmd(workers, name_hint="sh_w13_pypto"):
        block = pl.tile.get_block_idx()
        blocks = pl.tile.get_block_num()
        for band in pl.range((n_tiles + blocks - 1) // blocks):
            panel = block + band * blocks
            if panel < n_tiles:
                col = panel * W13_N_TILE
                for row in pl.range(0, T_PAD, W13_M_TILE):
                    gate_up = _project_w13_pypto(x, x_scale, shared_w1, shared_w1_scale, gate_up, row, col, 0)
                    gate_up = _project_w13_pypto(x, x_scale, shared_w3, shared_w3_scale, gate_up, row, col, 1)
    return gate_up


import torch.nn.functional as F


@pl.jit.inline
def expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    """Shared expert with device MXFP8 activation quantization before both projections."""
    # Merged W1/W3 projection into BF16 gate/up columns.
    gate_up = pl.create_tensor([T_PAD, GATE_UP_N], dtype=pl.BF16)
    x_w13, x_w13_scale = _W13_INPUTS(x_local, x_local_scale)
    w13_workers = pl.min(W13_GROUPS * (GATE_UP_N // W13_N_TILE), W13_WORKER_TILE)
    with pl.spmd(w13_workers, name_hint="sh_w13_mx_mm"):
        gate_up = _shared_w13_mm(
            x_w13, x_w13_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            gate_up, T_PAD,
        )

    # FP32 SwiGLU, FP8 payload, and packed MX_A_ZZ scales.
    h_tile_mx = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.FP8E4M3FN)
    h_scale_mx = pl.create_tensor([1, T_PAD * H_SCALE], dtype=pl.FP8E8M0)
    for block in pl.spmd(SWIGLU_WORKER_TILE, name_hint="sh_swiglu_vf"):
        blocks = pl.tile.get_block_num()
        swiglu_rows = pl.tensor.dim(gate_up, 0)
        pairs = MOE_INTER // SWIGLU_COL_TILE
        tasks = swiglu_rows // SWIGLU_ROW_TILE * pairs

        # Contiguous work ranges with Q16 rounding.
        base16 = tasks * 65536 // blocks
        first = pl.min(tasks, (block * base16 + 32768) // 65536)
        last = tasks
        if block + 1 < blocks:
            last = pl.min(tasks, ((block + 1) * base16 + 32768) // 65536)

        if first < last:
            gate = pl.create_tile([2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            up = pl.create_tile([2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            ub_payload = pl.create_tile([2 * SWIGLU_ROW_TILE * SWIGLU_COL_TILE], dtype=pl.INT8, target_memory=pl.Mem.Vec)
            ub_scales = pl.create_tile([2 * SWIGLU_SCALE_REGISTER_TILE], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
            gate, up = _shared_swiglu_prefetch(gate_up, gate, up, first, 0)
            if first + 1 < last:
                gate, up = _shared_swiglu_prefetch(gate_up, gate, up, first + 1, 1)

            for work in pl.range(first, last):
                iteration = work - first
                slot = iteration % 2
                pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, slot)
                if iteration >= 2:
                    pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)
                # Use static UB offsets for each register scope.
                if slot == 0:
                    _shared_swiglu_compute(gate, up, ub_payload, ub_scales, 0)
                else:
                    _shared_swiglu_compute(gate, up, ub_payload, ub_scales, 1)
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
                pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)
                if work + 2 < last:
                    pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, slot)
                    gate, up = _shared_swiglu_prefetch(gate_up, gate, up, work + 2, slot)
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, slot)

                row = work // pairs * SWIGLU_ROW_TILE
                col = work % pairs * SWIGLU_COL_TILE
                payload_rows = pl.reshape(ub_payload, [2 * SWIGLU_ROW_TILE, SWIGLU_COL_TILE])
                slot_row = slot * SWIGLU_ROW_TILE
                payload_slot = pl.tile.slice(payload_rows, [SWIGLU_ROW_TILE, SWIGLU_COL_TILE], [slot_row, 0])
                payload_fp8 = pl.reinterpret_view(payload_slot, pl.FP8E4M3FN)
                scale_bytes = pl.reinterpret_view(ub_scales, pl.UINT8, shape=[2, 4 * SWIGLU_SCALE_REGISTER_TILE])
                scale_slot = pl.tile.slice(scale_bytes, [1, SWIGLU_SCALE_TILE], [slot, 0])
                scale_fp8 = pl.reinterpret_view(scale_slot, pl.FP8E8M0)
                pl.store(payload_fp8, [row, col], h_tile_mx)
                pl.store(scale_fp8, [0, work * SWIGLU_SCALE_TILE], h_scale_mx)
                pl.system.set_flag(pl.PipeType.MTE3, pl.PipeType.V, slot)

            pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 0)
            pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 0)
            if first + 1 < last:
                pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE2, 1)
                pl.system.wait_flag(pl.PipeType.MTE3, pl.PipeType.V, 1)
        pl.system.barrier(pipe=pl.PipeType.ALL)

    # Complete row tiles keep MX_A_ZZ scale loads inside the packed backing.
    h_tile_scale_zz = pl.tensor.view(
        h_scale_mx, [T_PAD, H_SCALE], layout=pl.MX_A_ZZ,
    )
    rows = pl.tensor.dim(sh, 0)
    with pl.spmd(W2_WORKER_TILE, name_hint="sh_w2_mm"):
        worker = pl.tile.get_block_idx()
        for lane in pl.range((D // W2_N_TILE + W2_WORKER_TILE - 1) // W2_WORKER_TILE):
            column = worker + lane * W2_WORKER_TILE
            if column < D // W2_N_TILE:
                d0 = column * W2_N_TILE
                for wm in pl.range((T_PAD + W2_M_TILE - 1) // W2_M_TILE):
                    row0 = wm * W2_M_TILE
                    valid_m = pl.min(W2_M_TILE, T_PAD - row0)
                    valid_out = pl.min(W2_M_TILE, rows - row0)
                    hs0 = pl.load(h_tile_mx, [row0, 0], [W2_M_TILE, W2_K_TILE], valid_shape=[valid_m, W2_K_TILE])
                    hs_scale0 = pl.load(h_tile_scale_zz, [row0, 0], [W2_M_TILE, W2_K_TILE // MX_GROUP], valid_shape=[valid_m, W2_K_TILE // MX_GROUP])
                    sw2_0 = pl.load(shared_w2, [0, d0], [W2_K_TILE, W2_N_TILE])
                    sw2_scale0 = pl.load(
                        shared_w2_scale, [0, d0], [W2_K_TILE // MX_GROUP, W2_N_TILE],
                    )
                    y_acc = pl.matmul_mx(hs0, hs_scale0, sw2_0, sw2_scale0)
                    for k0 in pl.pipeline(W2_K_TILE, MOE_INTER, W2_K_TILE, stage=W2_K_STAGE):
                        hs_part = pl.load(h_tile_mx, [row0, k0], [W2_M_TILE, W2_K_TILE], valid_shape=[valid_m, W2_K_TILE])
                        hs_scale_part = pl.load(
                            h_tile_scale_zz,
                            [row0, k0 // MX_GROUP],
                            [W2_M_TILE, W2_K_TILE // MX_GROUP], valid_shape=[valid_m, W2_K_TILE // MX_GROUP],
                        )
                        sw2_part = pl.load(shared_w2, [k0, d0], [W2_K_TILE, W2_N_TILE])
                        sw2_scale_part = pl.load(
                            shared_w2_scale, [k0 // MX_GROUP, d0], [W2_K_TILE // MX_GROUP, W2_N_TILE],
                        )
                        y_acc = pl.matmul_mx_acc(
                            y_acc, hs_part, hs_scale_part, sw2_part, sw2_scale_part,
                        )
                    # Fixpipe rounds the accumulator directly to BF16.
                    sh = pl.store(pl.set_validshape(y_acc, valid_out, W2_N_TILE), [row0, d0], sh)

    return sh


@pl.jit
def l2_expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local, x_local_scale, shared_w1, shared_w1_scale,
        shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, sh,
    )
    return sh


@pl.jit
def l2_shared_w13_pypto(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * K_SCALE], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    gate_up: pl.Out[pl.Tensor[[T_PAD, GATE_UP_N], pl.BF16]],
):
    shared_w13_pypto(x_local, x_local_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, gate_up)
    return gate_up


def _golden_w13(tensors, rows):
    from quant import unpack_mx_scale_codes as decode_e8m0_codes
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    x_fp8 = tensors["x_local"][:rows]
    x_scale = decode_e8m0_codes(tensors["x_local_scale"].reshape(T_PAD, K_SCALE), side="a")[:rows]
    w1_fp8 = tensors["shared_w1"]
    w1_scale = decode_e8m0_codes(tensors["shared_w1_scale"], side="b")
    w3_fp8 = tensors["shared_w3"]
    w3_scale = decode_e8m0_codes(tensors["shared_w3_scale"], side="b")
    gate = matmul_mx_golden(x_fp8, x_scale, w1_fp8, w1_scale).bfloat16()
    up = matmul_mx_golden(x_fp8, x_scale, w3_fp8, w3_scale).bfloat16()
    return gate, up


def golden_shared_w13(tensors):
    """Reference for the concatenated BF16 output of W1 and W3."""
    gate, up = _golden_w13(tensors, T_PAD)
    tensors["gate_up"][:, :MOE_INTER] = gate
    tensors["gate_up"][:, MOE_INTER:] = up


def golden_expert_shared(tensors):
    """Reference for MXFP8 W1/W3, SwiGLU requantization, and MXFP8 W2."""
    import torch

    from quant import unpack_mx_scale_codes as decode_e8m0_codes, quantize_mxfp8_upward_e8m0 as host_quant_mxfp8
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    gate, up = _golden_w13(tensors, T)
    sh_gate = gate.float()
    sh_up = up.float()
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_fp8, sh_h_scale = host_quant_mxfp8(sh_h)
    w2_fp8 = tensors["shared_w2"]
    w2_scale = decode_e8m0_codes(tensors["shared_w2_scale"], side="b")
    sh = matmul_mx_golden(sh_h_fp8, sh_h_scale, w2_fp8, w2_scale)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs(*, w13_only=False):
    import torch
    from golden.spec import TensorSpec
    from quant import generate_mxfp8_weight_kn as gen_mxfp8_weight_kn_device, quantize_mxfp8_upward_e8m0 as host_mxfp8_activation

    x_local_bf16 = torch.randn(T_PAD, D, dtype=torch.bfloat16)
    if T < T_PAD:
        x_local_bf16[T:, :] = 0
    x_local_fp8, x_local_scale_codes = host_mxfp8_activation(x_local_bf16)
    # matmul_mx expects the physical MX_A_ZZ scale order.
    from quant import pack_mx_a_scale
    x_local_scale_codes = pack_mx_a_scale(x_local_scale_codes)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_local_scale = x_local_scale_codes.contiguous().view(fp8_e8m0)

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_fp8, sw1_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w1"], chan_cv=0.50, seed=1)
    sw3_fp8, sw3_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w3"], chan_cv=0.50, seed=2)
    sw1_scale = sw1_scale.contiguous().view(fp8_e8m0)
    sw3_scale = sw3_scale.contiguous().view(fp8_e8m0)

    fp8 = torch.float8_e4m3fn

    specs = [
        TensorSpec("x_local", [T_PAD, D], fp8, init_value=lambda: x_local_fp8),
        TensorSpec("x_local_scale", [1, T_PAD * (D // MX_GROUP)], fp8_e8m0, init_value=lambda: x_local_scale.reshape(1, -1)),
        TensorSpec("shared_w1", [D, MOE_INTER], fp8, init_value=lambda: sw1_fp8),
        TensorSpec("shared_w1_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw1_scale),
        TensorSpec("shared_w3", [D, MOE_INTER], fp8, init_value=lambda: sw3_fp8),
        TensorSpec("shared_w3_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw3_scale),
    ]
    if w13_only:
        return [*specs, TensorSpec("gate_up", [T_PAD, GATE_UP_N], torch.bfloat16)]

    sw2_fp8, sw2_scale = gen_mxfp8_weight_kn_device(D, MOE_INTER, SHARED_DEQUANT_STD["w2"], chan_cv=0.33, seed=3)
    sw2_scale = sw2_scale.contiguous().view(fp8_e8m0)
    return [
        *specs,
        TensorSpec("shared_w2", [MOE_INTER, D], fp8, init_value=lambda: sw2_fp8),
        TensorSpec("shared_w2_scale", [MOE_INTER // MX_GROUP, D], fp8_e8m0, init_value=lambda: sw2_scale),
        TensorSpec("sh", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden.validation import ratio_reldiff
    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--pypto-w13", action="store_true", help="validate the standalone PyPTO W1/W3 projection")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("this entry requires hardware; --sim is unsupported")
    if args.tokens is None:
        if args.golden_data or args.runtime_dir:
            parser.error("--golden-data and --runtime-dir require an explicit --tokens value")
        import subprocess

        modes = ((),) if args.pypto_w13 else ((), ("--pypto-w13",))
        for tokens in (16, 80, 96):
            for mode in modes:
                case = "PyPTO W13" if mode or args.pypto_w13 else "shared expert"
                print(f"Validating {case} with {tokens} tokens", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--tokens", str(tokens), *mode, *sys.argv[1:]]
                completed = subprocess.run(command, check=False)
                if completed.returncode:
                    raise SystemExit(completed.returncode)
        raise SystemExit(0)

    result = run(
        fn=l2_shared_w13_pypto if args.pypto_w13 else l2_expert_shared,
        specs=build_tensor_specs(w13_only=args.pypto_w13),
        golden_fn=golden_shared_w13 if args.pypto_w13 else golden_expert_shared,
        golden_data=args.golden_data,
        save_data=args.save_data,
        runtime_dir=args.runtime_dir,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            ("gate_up" if args.pypto_w13 else "sh"): ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
