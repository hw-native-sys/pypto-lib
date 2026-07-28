# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed local expert — temporary MXFP8 weights + acts.

AscendC Hybrid is W4A8 (MXFP4 weight × MXFP8 act). On A5, FP4 Mat→Right TMov
is blocked by PTOAS EmitC (cannot fix in pypto-lib alone), so this kernel
temporarily uses MXFP8 weights+acts (same path as ``expert_shared``). Switch
back to Hybrid MXFP4 when EmitC FP4 Right is fixed.

  BF16 recv_x → dynamic MXFP8 (e4m3 + e8m0) → MX GEMM (FP8 w1/w3) → SwiGLU
  → dynamic MXFP8 → MX GEMM (FP8 w2) → × routing weight → BF16 recv_y

Weights are Right matrices for ``matmul_mx`` (``pl.FP8E4M3FN``):
  w1/w3: ``[E, D, MOE_INTER]`` + tiled scale ``[E, _W13_SCALE_ROWS, MM_INTER_TILE]``
  w2:    ``[E, MOE_INTER, D]`` + tiled scale ``[E, _W2_SCALE_ROWS, D_OUT_TILE]``
"""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    EP_WORLD_SIZE,
    RECV_MAX,
    MX_BLOCK_K,
)
from mx_quant_common import (
    ATOL_RTOL,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
)


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

RECV_TILE = 16
K_TILE = 128
MM_INTER_TILE = 128
ACT_INTER_TILE = 256
D_OUT_TILE = 128

assert RECV_MAX % RECV_TILE == 0
assert D % MX_BLOCK_K == 0 and MOE_INTER % MX_BLOCK_K == 0
assert SWIGLU_LIMIT > 0.0

_GATE_SPMD = MOE_INTER // MM_INTER_TILE
_GATE_K_CHUNKS = D // K_TILE
_DOWN_SPMD = D // D_OUT_TILE
_DOWN_K_CHUNKS = MOE_INTER // K_TILE
_KS = K_TILE // MX_BLOCK_K
# Tiled MX_B_NN: each (K-chunk, N-tile) independently convert_x2'd; col offset 0.
_W13_SCALE_ROWS = _GATE_SPMD * _GATE_K_CHUNKS * _KS
_W2_SCALE_ROWS = _DOWN_SPMD * _DOWN_K_CHUNKS * _KS
# Concurrent: all local experts × recv tiles × SPMD × K-chunks
_MX_PHASE_SLOTS = max(_GATE_SPMD * _GATE_K_CHUNKS, _DOWN_SPMD * _DOWN_K_CHUNKS)
_MX_WS_SLOTS = N_LOCAL_EXPERTS * (RECV_MAX // RECV_TILE) * _MX_PHASE_SLOTS
assert _GATE_K_CHUNKS == 32 and _DOWN_K_CHUNKS == 16  # pl.unroll literals below


@pl.jit.inline
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    # Left-scale: store flat tquant exp to per-slot GM → mx_a_zz (AND2ZZ)
    # → LeftScale. Direct Mat ND→LeftScale is numerically wrong.
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * RECV_TILE, K_TILE // MX_BLOCK_K], dtype=pl.FP8E8M0
    )
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        e_rows = pl.read(recv_expert_count, [local_e, 0])
        e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE
        # 2D GM views for MX weight/scale loads (mx_layout needs rank-2).
        w1_e = pl.reshape(
            pl.slice(routed_w1, [1, D, MOE_INTER], [local_e, 0, 0]),
            [D, MOE_INTER],
        )
        w1s_e = pl.reshape(
            pl.slice(routed_w1_scale, [1, _W13_SCALE_ROWS, MM_INTER_TILE], [local_e, 0, 0]),
            [_W13_SCALE_ROWS, MM_INTER_TILE],
        )
        w3_e = pl.reshape(
            pl.slice(routed_w3, [1, D, MOE_INTER], [local_e, 0, 0]),
            [D, MOE_INTER],
        )
        w3s_e = pl.reshape(
            pl.slice(routed_w3_scale, [1, _W13_SCALE_ROWS, MM_INTER_TILE], [local_e, 0, 0]),
            [_W13_SCALE_ROWS, MM_INTER_TILE],
        )
        w2_e = pl.reshape(
            pl.slice(routed_w2, [1, MOE_INTER, D], [local_e, 0, 0]),
            [MOE_INTER, D],
        )
        w2s_e = pl.reshape(
            pl.slice(routed_w2_scale, [1, _W2_SCALE_ROWS, D_OUT_TILE], [local_e, 0, 0]),
            [_W2_SCALE_ROWS, D_OUT_TILE],
        )

        for tt in pl.parallel(e_tiles):
            tt0 = tt * RECV_TILE
            # Static slot base (worst-case RECV_MAX tiles) for concurrent experts × tiles.
            tile_base = (local_e * (RECV_MAX // RECV_TILE) + tt) * _MX_PHASE_SLOTS

            gate_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            up_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

            for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="exp_gate_mm"):
                n0 = nb_idx * MM_INTER_TILE
                # Peel K=0 with matmul_mx; pl.unroll remaining so Acc SSA chains.
                k0 = 0
                x_tile = pl.load(
                    recv_x,
                    [local_e, tt0, k0],
                    [1, RECV_TILE, K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                x_q, x_s = pl.mx_quant(
                    pl.cast(x_2d, target_type=pl.FP32, mode="none"), mode="mxfp8_e4m3"
                )
                w_tile = pl.load(w1_e, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat)
                ws_tile = pl.load(
                    w1s_e,
                    [nb_idx * _GATE_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                    [_KS, MM_INTER_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = (tile_base + nb_idx * _GATE_K_CHUNKS) * RECV_TILE
                la = pl.move(
                    pl.move(pl.tile.reinterpret_view(x_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                la = pl.set_validshape(la, RECV_TILE, K_TILE)
                pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [RECV_TILE, _KS],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                las = pl.tget_scale_addr(las, la)
                las = pl.set_validshape(las, RECV_TILE, _KS)
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb)
                gate_acc = pl.matmul_mx(la, las, rb, rbs)
                for db in pl.unroll(31):  # _GATE_K_CHUNKS - 1
                    k0 = (db + 1) * K_TILE
                    x_tile = pl.load(
                        recv_x,
                        [local_e, tt0, k0],
                        [1, RECV_TILE, K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                    x_q, x_s = pl.mx_quant(
                        pl.cast(x_2d, target_type=pl.FP32, mode="none"), mode="mxfp8_e4m3"
                    )
                    w_tile = pl.load(w1_e, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat)
                    ws_tile = pl.load(
                        w1s_e,
                        [nb_idx * _GATE_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                        [_KS, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + nb_idx * _GATE_K_CHUNKS + (db + 1)
                    ) * RECV_TILE
                    la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(x_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la2 = pl.set_validshape(la2, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, _KS],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las2 = pl.tget_scale_addr(las2, la2)
                    las2 = pl.set_validshape(las2, RECV_TILE, _KS)
                    rb2 = pl.move(w_tile, target_memory=pl.Mem.Right)
                    rbs2 = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb2)
                    gate_acc = pl.matmul_mx_acc(gate_acc, la2, las2, rb2, rbs2)
                pl.store(gate_acc, [0, n0], gate_fp32)

            for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="exp_up_mm"):
                n0 = nb_idx * MM_INTER_TILE
                k0 = 0
                x_tile = pl.load(
                    recv_x,
                    [local_e, tt0, k0],
                    [1, RECV_TILE, K_TILE],
                    target_memory=pl.Mem.Vec,
                )
                x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                x_q, x_s = pl.mx_quant(
                    pl.cast(x_2d, target_type=pl.FP32, mode="none"), mode="mxfp8_e4m3"
                )
                w_tile = pl.load(w3_e, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat)
                ws_tile = pl.load(
                    w3s_e,
                    [nb_idx * _GATE_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                    [_KS, MM_INTER_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = (tile_base + nb_idx * _GATE_K_CHUNKS) * RECV_TILE
                la = pl.move(
                    pl.move(pl.tile.reinterpret_view(x_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                la = pl.set_validshape(la, RECV_TILE, K_TILE)
                pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [RECV_TILE, _KS],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                las = pl.tget_scale_addr(las, la)
                las = pl.set_validshape(las, RECV_TILE, _KS)
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb)
                up_acc = pl.matmul_mx(la, las, rb, rbs)
                for db in pl.unroll(31):
                    k0 = (db + 1) * K_TILE
                    x_tile = pl.load(
                        recv_x,
                        [local_e, tt0, k0],
                        [1, RECV_TILE, K_TILE],
                        target_memory=pl.Mem.Vec,
                    )
                    x_2d = pl.reshape(x_tile, [RECV_TILE, K_TILE])
                    x_q, x_s = pl.mx_quant(
                        pl.cast(x_2d, target_type=pl.FP32, mode="none"), mode="mxfp8_e4m3"
                    )
                    w_tile = pl.load(w3_e, [k0, n0], [K_TILE, MM_INTER_TILE], target_memory=pl.Mem.Mat)
                    ws_tile = pl.load(
                        w3s_e,
                        [nb_idx * _GATE_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                        [_KS, MM_INTER_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + nb_idx * _GATE_K_CHUNKS + (db + 1)
                    ) * RECV_TILE
                    la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(x_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la2 = pl.set_validshape(la2, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(x_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, _KS],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las2 = pl.tget_scale_addr(las2, la2)
                    las2 = pl.set_validshape(las2, RECV_TILE, _KS)
                    rb2 = pl.move(w_tile, target_memory=pl.Mem.Right)
                    rbs2 = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb2)
                    up_acc = pl.matmul_mx_acc(up_acc, la2, las2, rb2, rbs2)
                pl.store(up_acc, [0, n0], up_fp32)

            h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            for part in pl.spmd(MOE_INTER // ACT_INTER_TILE, name_hint="exp_swiglu"):
                n0 = part * ACT_INTER_TILE
                gate_rows = gate_fp32[:, n0 : n0 + ACT_INTER_TILE]
                up_rows = up_fp32[:, n0 : n0 + ACT_INTER_TILE]
                gate_clamped = pl.minimum(gate_rows, SWIGLU_LIMIT)
                up_clamped = pl.maximum(pl.minimum(up_rows, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
                gated = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
                h_tile_fp32[:, n0 : n0 + ACT_INTER_TILE] = gated

            for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="exp_w2_mm"):
                d0 = db_idx * D_OUT_TILE
                k0 = 0
                h_tile = pl.load(
                    h_tile_fp32, [0, k0], [RECV_TILE, K_TILE], target_memory=pl.Mem.Vec
                )
                h_q, h_s = pl.mx_quant(h_tile, mode="mxfp8_e4m3")
                w_tile = pl.load(w2_e, [k0, d0], [K_TILE, D_OUT_TILE], target_memory=pl.Mem.Mat)
                ws_tile = pl.load(
                    w2s_e,
                    [db_idx * _DOWN_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                    [_KS, D_OUT_TILE],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = (tile_base + db_idx * _DOWN_K_CHUNKS) * RECV_TILE
                la = pl.move(
                    pl.move(pl.tile.reinterpret_view(h_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                la = pl.set_validshape(la, RECV_TILE, K_TILE)
                pl.store(pl.tile.reinterpret_view(h_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [RECV_TILE, _KS],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                las = pl.tget_scale_addr(las, la)
                las = pl.set_validshape(las, RECV_TILE, _KS)
                rb = pl.move(w_tile, target_memory=pl.Mem.Right)
                rbs = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb)
                y_acc = pl.matmul_mx(la, las, rb, rbs)
                for kb in pl.unroll(15):  # _DOWN_K_CHUNKS - 1
                    k0 = (kb + 1) * K_TILE
                    h_tile = pl.load(
                        h_tile_fp32, [0, k0], [RECV_TILE, K_TILE], target_memory=pl.Mem.Vec
                    )
                    h_q, h_s = pl.mx_quant(h_tile, mode="mxfp8_e4m3")
                    w_tile = pl.load(w2_e, [k0, d0], [K_TILE, D_OUT_TILE], target_memory=pl.Mem.Mat)
                    ws_tile = pl.load(
                        w2s_e,
                        [db_idx * _DOWN_K_CHUNKS * _KS + (k0 // K_TILE) * _KS, 0],
                        [_KS, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = (
                        tile_base + db_idx * _DOWN_K_CHUNKS + (kb + 1)
                    ) * RECV_TILE
                    la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(h_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    la2 = pl.set_validshape(la2, RECV_TILE, K_TILE)
                    pl.store(pl.tile.reinterpret_view(h_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [RECV_TILE, _KS],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    las2 = pl.tget_scale_addr(las2, la2)
                    las2 = pl.set_validshape(las2, RECV_TILE, _KS)
                    rb2 = pl.move(w_tile, target_memory=pl.Mem.Right)
                    rbs2 = pl.tget_scale_addr(pl.move(ws_tile, target_memory=pl.Mem.RightScale), rb2)
                    y_acc = pl.matmul_mx_acc(y_acc, la2, las2, rb2, rbs2)
                # Apply routing weights then store BF16.
                w_col = pl.load(
                    recv_weights,
                    [local_e, tt0],
                    [1, RECV_TILE],
                    target_memory=pl.Mem.Vec,
                )
                w_col = pl.reshape(w_col, [RECV_TILE, 1])
                # trowexpandmul requires row-major dst/src (Acc is col_major).
                y_fp32 = pl.move(
                    y_acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major
                )
                y_scaled = pl.row_expand_mul(y_fp32, w_col)
                y_bf16 = pl.cast(y_scaled, target_type=pl.BF16, mode="rint")
                # tstore needs ND (row_major + none_box); Acc→Vec path leaves boxed slayout.
                y_bf16 = pl.move(
                    y_bf16,
                    target_memory=pl.Mem.Vec,
                    blayout=pl.TileLayout.row_major,
                    slayout=pl.TileLayout.none_box,
                )
                pl.store(y_bf16, [local_e, tt0, d0], recv_y)

    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, _W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


def gen_routed_weight(shape, dequant_std, chan_cv, n_tile, k_tile=K_TILE):
    """Synthesize routed-expert MXFP8 weight + tiled MX_B_NN scale.

    ``shape`` is historical ``[E, out, in]``. Returns Right-matrix storage
    ``(w_kn [E, in, out] float8_e4m3fn, scale [E, scale_rows, n_tile] float8_e8m0fnu)``.
    """
    import torch

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim
    ws = []
    ss = []
    for _ in range(n_lead):
        w, s = gen_mxfp8_weight_kn(
            (inn, out),
            dequant_std=dequant_std,
            chan_cv=chan_cv,
            pack_nn=True,
            n_tile=n_tile,
            k_tile=k_tile,
        )
        ws.append(w)
        ss.append(s)
    w_out = torch.stack(ws, dim=0).reshape(*lead, inn, out)
    s_out = torch.stack(ss, dim=0).reshape(*lead, *ss[0].shape)
    return w_out, s_out


def golden_expert_routed(tensors):
    """Torch reference: per-K-tile dyn MXFP8 + tiled MX_B_NN unpack (matches device)."""
    import torch
    import torch.nn.functional as F

    def _b_scale(s, n_tile, logical_k, logical_n):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_KS,
            n_tile=n_tile,
            logical_k=logical_k // MX_BLOCK_K,
            logical_n=logical_n,
        )

    def mx_matmul_act_tiled(x_f, w, w_s, k_tile):
        acc = None
        for k0 in range(0, x_f.shape[-1], k_tile):
            xq, xs = dynamic_mx_quant_e4m3(x_f[..., k0 : k0 + k_tile])
            part = mx_matmul_fp8(
                xq, xs, w[k0 : k0 + k_tile], w_s[k0 // MX_BLOCK_K : (k0 + k_tile) // MX_BLOCK_K]
            )
            acc = part if acc is None else acc + part
        return acc

    recv_x = tensors["recv_x"]
    recv_weights = tensors["recv_weights"]
    recv_expert_count = tensors["recv_expert_count"]
    e, recv_max, d = recv_x.shape
    out = torch.zeros(e, recv_max, d, dtype=torch.float32)

    for ei in range(e):
        n_rows = int(recv_expert_count[ei, 0].item())
        if n_rows <= 0:
            continue
        x = recv_x[ei, :n_rows, :].float()
        w1_s = _b_scale(tensors["routed_w1_scale"][ei], MM_INTER_TILE, D, MOE_INTER)
        w3_s = _b_scale(tensors["routed_w3_scale"][ei], MM_INTER_TILE, D, MOE_INTER)
        w2_s = _b_scale(tensors["routed_w2_scale"][ei], D_OUT_TILE, MOE_INTER, D)
        gate = mx_matmul_act_tiled(x, tensors["routed_w1"][ei], w1_s, K_TILE)
        up = mx_matmul_act_tiled(x, tensors["routed_w3"][ei], w3_s, K_TILE)
        if SWIGLU_LIMIT and SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        y = mx_matmul_act_tiled(h, tensors["routed_w2"][ei], w2_s, K_TILE)
        y = y * recv_weights[ei, :n_rows].reshape(-1, 1).float()
        out[ei, :n_rows, :] = y

    tensors["recv_y"][:] = out.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(
        torch.randint(0, N_LOCAL_EXPERTS, (total,)),
        minlength=N_LOCAL_EXPERTS,
    ).clamp(max=RECV_MAX)
    recv_expert_count = counts.to(torch.int32).reshape(N_LOCAL_EXPERTS, 1)

    recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts[e].item())
        if n == 0:
            continue
        recv_x[e, :n] = torch.randn(n, D, dtype=torch.bfloat16)
        recv_weights[e, :n] = torch.rand(n) * 0.5 + 0.5

    # Real MXFP8 grid (block=32); chan_cv reproduces per-output-channel magnitude spread.
    w1, w1_s = gen_routed_weight(
        (N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"], chan_cv=0.50, n_tile=MM_INTER_TILE
    )
    w3, w3_s = gen_routed_weight(
        (N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"], chan_cv=0.50, n_tile=MM_INTER_TILE
    )
    w2, w2_s = gen_routed_weight(
        (N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"], chan_cv=0.33, n_tile=D_OUT_TILE
    )

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, init_value=lambda: recv_x),
        TensorSpec(
            "recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: recv_weights
        ),
        TensorSpec(
            "recv_expert_count",
            [N_LOCAL_EXPERTS, 1],
            torch.int32,
            init_value=lambda: recv_expert_count,
        ),
        TensorSpec(
            "routed_w1",
            [N_LOCAL_EXPERTS, D, MOE_INTER],
            torch.float8_e4m3fn,
            init_value=lambda: w1,
        ),
        TensorSpec(
            "routed_w1_scale",
            [N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: w1_s,
        ),
        TensorSpec(
            "routed_w3",
            [N_LOCAL_EXPERTS, D, MOE_INTER],
            torch.float8_e4m3fn,
            init_value=lambda: w3,
        ),
        TensorSpec(
            "routed_w3_scale",
            [N_LOCAL_EXPERTS, _W13_SCALE_ROWS, MM_INTER_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: w3_s,
        ),
        TensorSpec(
            "routed_w2",
            [N_LOCAL_EXPERTS, MOE_INTER, D],
            torch.float8_e4m3fn,
            init_value=lambda: w2,
        ),
        TensorSpec(
            "routed_w2_scale",
            [N_LOCAL_EXPERTS, _W2_SCALE_ROWS, D_OUT_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: w2_s,
        ),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    moe_tol = ATOL_RTOL["moe_mx"]
    result = run_jit(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=moe_tol["rtol"],
        atol=moe_tol["atol"],
        compare_fn={
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
