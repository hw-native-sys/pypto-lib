# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP input projection — Hybrid MXFP8.

Mirrors the MTP-only prolog in the official implementation:
``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.

``prev_hidden_states`` and ``hidden_states_out`` use token-major pre-``hc_head``
hidden states with HC lanes: ``[T, HC_MULT, D]``.

Weights are stored as Right matrices for ``matmul_mx`` (AscendC ``MxFp8LinearMethod``):

  e_proj_w: ``[D, D]`` FP8E4M3FN + e_proj_w_scale: tiled MX_B_NN
    ``[OUT_BLOCKS * D_BLOCKS * (D_CHUNK/32), OUT_CHUNK]`` FP8E8M0
  h_proj_w: same layout as e_proj

RMSNorm weights (``enorm_w``, ``hnorm_w``) remain FP32/BF16. Dynamic MXFP8 activation
quant after RMSNorm replaces the legacy smooth-quant INT8 path (``e_proj_smooth`` /
``h_proj_smooth`` removed).
"""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    MX_BLOCK_K,
    PREFILL_BATCH,
    PREFILL_SEQ,
)
from mx_quant_common import (
    ATOL_RTOL,
    dynamic_mx_quant_e4m3,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
)


T_DYN = pl.dynamic("T_DYN")
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = HC_MULT * D
EPS = M.rms_norm_eps
D_INV = 1.0 / D
D_SCALE = D // MX_BLOCK_K

T_TILE = 8
LINEAR_T_TILE = 16
D_CHUNK = 128
OUT_CHUNK = 128
D_BLOCKS = D // D_CHUNK
OUT_BLOCKS = D // OUT_CHUNK
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert D % MX_BLOCK_K == 0
assert D_CHUNK % MX_BLOCK_K == 0
# Tiled MX_B_NN: each (N-tile=OUT_CHUNK, K-chunk=D_CHUNK) independently convert_x2'd.
_KS = D_CHUNK // MX_BLOCK_K
_WQ_SCALE_ROWS = OUT_BLOCKS * D_BLOCKS * _KS
assert _KS == 4
# pl.unroll literal in mtp_projection_linear (peel K=0 + D_BLOCKS-1 acc).
assert D_BLOCKS == 32
assert HC_MULT == 4  # pl.unroll(4) literals in mtp_projection_linear

# Worst-case concurrent linear tiles (prefill) × N-blocks × K-chunks.
_T_LINEAR_MAX = (
    ((PREFILL_BATCH * PREFILL_SEQ + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
)
_MX_WS_SLOTS = (_T_LINEAR_MAX // LINEAR_T_TILE) * OUT_BLOCKS * D_BLOCKS


@pl.jit.inline
def mtp_projection(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    e_proj_w_scale: pl.Tensor[[_WQ_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    h_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    h_proj_w_scale: pl.Tensor[[_WQ_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    hidden_states_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    t_dim = pl.tensor.dim(hidden_states, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    hidden_flat = pl.reshape(hidden_states, [t_dim, D])
    prev_flat = pl.reshape(prev_hidden_states, [t_dim, HC_DIM])
    out_flat = pl.reshape(hidden_states_out, [t_dim, HC_DIM])
    hidden_norm = pl.create_tensor([t_linear, D], dtype=pl.FP32)
    prev_norm = pl.create_tensor([t_linear, HC_DIM], dtype=pl.FP32)
    hidden_inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    prev_inv_rms = pl.create_tensor([HC_MULT, t_linear], dtype=pl.FP32)
    out_pad = pl.create_tensor([t_linear, HC_DIM], dtype=pl.FP32)
    # e_proj Acc is materialized once; h Acc lands in out_pad; a separate vec
    # scope adds them (mixed AIC/AIV was dropping the FP32 add after matmul_mx).
    e_proj_pad = pl.create_tensor([t_linear, D], dtype=pl.FP32)
    mx_scale_ws = pl.create_tensor(
        [_MX_WS_SLOTS * LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K], dtype=pl.FP8E8M0
    )

    for t0 in pl.parallel(0, t_dim, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_rms"):
            hidden_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(D_BLOCKS, stage=2):
                k0 = kb * D_CHUNK
                hidden_chunk = pl.cast(hidden_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                hidden_sq_sum = pl.add(
                    hidden_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(hidden_chunk, hidden_chunk)), [1, T_TILE]),
                )
            hidden_var = pl.add(pl.mul(hidden_sq_sum, D_INV), EPS)
            hidden_inv = pl.reshape(pl.rsqrt(hidden_var, high_precision=True), [T_TILE, 1])
            hidden_inv_rms = pl.assemble(hidden_inv_rms, hidden_inv, [t0, 0])
            for hc in pl.range(HC_MULT):
                prev_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(D_BLOCKS, stage=2):
                    k0 = kb * D_CHUNK
                    prev_k0 = hc * D + k0
                    prev_chunk = pl.cast(prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + D_CHUNK], target_type=pl.FP32)
                    prev_sq_sum = pl.add(
                        prev_sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(prev_chunk, prev_chunk)), [1, T_TILE]),
                    )
                prev_var = pl.add(pl.mul(prev_sq_sum, D_INV), EPS)
                prev_inv = pl.reshape(pl.rsqrt(prev_var, high_precision=True), [T_TILE, 1])
                prev_inv_rms = pl.assemble(prev_inv_rms, pl.reshape(prev_inv, [1, T_TILE]), [hc, t0])

    for t0 in pl.parallel(0, t_dim, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_norm"):
            hidden_inv = hidden_inv_rms[t0 : t0 + T_TILE, 0:1]
            for kb in pl.range(D_BLOCKS):
                k0 = kb * D_CHUNK
                hidden_chunk = pl.cast(hidden_flat[t0 : t0 + T_TILE, k0 : k0 + D_CHUNK], target_type=pl.FP32)
                enorm = pl.reshape(enorm_w[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                hidden_norm_tile = pl.col_expand_mul(
                    pl.row_expand_mul(hidden_chunk, hidden_inv),
                    enorm,
                )
                hidden_norm = pl.assemble(hidden_norm, hidden_norm_tile, [t0, k0])
                hnorm = pl.reshape(hnorm_w[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                for hc in pl.range(HC_MULT):
                    prev_k0 = hc * D + k0
                    prev_inv = pl.reshape(prev_inv_rms[hc : hc + 1, t0 : t0 + T_TILE], [T_TILE, 1])
                    prev_chunk = pl.cast(prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + D_CHUNK], target_type=pl.FP32)
                    prev_norm_tile = pl.col_expand_mul(
                        pl.row_expand_mul(prev_chunk, prev_inv),
                        hnorm,
                    )
                    prev_norm = pl.assemble(prev_norm, prev_norm_tile, [t0, prev_k0])

    for t0 in pl.parallel(0, t_linear, LINEAR_T_TILE):
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        tc = t0 // LINEAR_T_TILE
        for nb in pl.parallel(0, OUT_BLOCKS, 1):
            n0 = nb * OUT_CHUNK
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_linear_e"):
                # Match v4-flash linear peel (matmul then matmul_acc from kb=1) and
                # the pro MX convention (matmul_mx init Acc; matmul_mx_acc rest).
                # Bare create_tile(Acc)+matmul_mx_acc stalls on A5 (507018 / S1).
                # Separate CORE_GROUP from h so e_proj_pad is visible before the add.
                k0 = 0
                hidden_tile = pl.load(
                    hidden_norm,
                    [t0, k0],
                    [LINEAR_T_TILE, D_CHUNK],
                    valid_shapes=[t_rows, D_CHUNK],
                    target_memory=pl.Mem.Vec,
                )
                hidden_q, hidden_s = pl.mx_quant(hidden_tile, mode="mxfp8_e4m3")
                e_w_tile = pl.load(
                    e_proj_w,
                    [k0, n0],
                    [D_CHUNK, OUT_CHUNK],
                    target_memory=pl.Mem.Mat,
                )
                e_ws_tile = pl.load(
                    e_proj_w_scale,
                    [(nb * D_BLOCKS + k0 // D_CHUNK) * _KS, 0],
                    [_KS, OUT_CHUNK],
                    target_memory=pl.Mem.Mat,
                    mx_layout="mx_b_nn",
                )
                srow = ((tc * OUT_BLOCKS + nb) * D_BLOCKS + k0 // D_CHUNK) * LINEAR_T_TILE
                e_la = pl.move(
                    pl.move(pl.tile.reinterpret_view(hidden_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                e_la = pl.set_validshape(e_la, LINEAR_T_TILE, D_CHUNK)
                pl.store(pl.tile.reinterpret_view(hidden_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                e_las = pl.move(
                    pl.load(
                        mx_scale_ws,
                        [srow, 0],
                        [LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_a_zz",
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                e_las = pl.tget_scale_addr(e_las, e_la)
                e_las = pl.set_validshape(e_las, LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K)
                e_rb = pl.move(e_w_tile, target_memory=pl.Mem.Right)
                e_rbs = pl.move(e_ws_tile, target_memory=pl.Mem.RightScale)
                e_rbs = pl.tget_scale_addr(e_rbs, e_rb)
                e_acc = pl.matmul_mx(e_la, e_las, e_rb, e_rbs)
                for db in pl.unroll(31):  # D_BLOCKS - 1
                    k0 = (db + 1) * D_CHUNK
                    hidden_tile2 = pl.load(
                        hidden_norm,
                        [t0, k0],
                        [LINEAR_T_TILE, D_CHUNK],
                        valid_shapes=[t_rows, D_CHUNK],
                        target_memory=pl.Mem.Vec,
                    )
                    hidden_q2, hidden_s2 = pl.mx_quant(hidden_tile2, mode="mxfp8_e4m3")
                    e_w_tile2 = pl.load(
                        e_proj_w,
                        [k0, n0],
                        [D_CHUNK, OUT_CHUNK],
                        target_memory=pl.Mem.Mat,
                    )
                    e_ws_tile2 = pl.load(
                        e_proj_w_scale,
                        [(nb * D_BLOCKS + k0 // D_CHUNK) * _KS, 0],
                        [_KS, OUT_CHUNK],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow2 = ((tc * OUT_BLOCKS + nb) * D_BLOCKS + k0 // D_CHUNK) * LINEAR_T_TILE
                    e_la2 = pl.move(
                        pl.move(pl.tile.reinterpret_view(hidden_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    e_la2 = pl.set_validshape(e_la2, LINEAR_T_TILE, D_CHUNK)
                    pl.store(pl.tile.reinterpret_view(hidden_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                    e_las2 = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow2, 0],
                            [LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    e_las2 = pl.tget_scale_addr(e_las2, e_la2)
                    e_las2 = pl.set_validshape(e_las2, LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K)
                    e_rb2 = pl.move(e_w_tile2, target_memory=pl.Mem.Right)
                    e_rbs2 = pl.move(e_ws_tile2, target_memory=pl.Mem.RightScale)
                    e_rbs2 = pl.tget_scale_addr(e_rbs2, e_rb2)
                    e_acc = pl.matmul_mx_acc(e_acc, e_la2, e_las2, e_rb2, e_rbs2)
                pl.store(e_acc, [t0, n0], e_proj_pad)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_linear_h"):
                for hc in pl.unroll(4):  # HC_MULT
                    prev_base = hc * D
                    prev_out = prev_base + n0
                    k0 = 0
                    prev_tile = pl.load(
                        prev_norm,
                        [t0, prev_base + k0],
                        [LINEAR_T_TILE, D_CHUNK],
                        valid_shapes=[t_rows, D_CHUNK],
                        target_memory=pl.Mem.Vec,
                    )
                    prev_q, prev_s = pl.mx_quant(prev_tile, mode="mxfp8_e4m3")
                    h_w_tile = pl.load(
                        h_proj_w,
                        [k0, n0],
                        [D_CHUNK, OUT_CHUNK],
                        target_memory=pl.Mem.Mat,
                    )
                    h_ws_tile = pl.load(
                        h_proj_w_scale,
                        [(nb * D_BLOCKS + k0 // D_CHUNK) * _KS, 0],
                        [_KS, OUT_CHUNK],
                        target_memory=pl.Mem.Mat,
                        mx_layout="mx_b_nn",
                    )
                    srow = ((tc * OUT_BLOCKS + nb) * D_BLOCKS + k0 // D_CHUNK) * LINEAR_T_TILE
                    h_la = pl.move(
                        pl.move(pl.tile.reinterpret_view(prev_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                        target_memory=pl.Mem.Left,
                    )
                    h_la = pl.set_validshape(h_la, LINEAR_T_TILE, D_CHUNK)
                    pl.store(pl.tile.reinterpret_view(prev_s, pl.FP8E8M0), [srow, 0], mx_scale_ws)
                    h_las = pl.move(
                        pl.load(
                            mx_scale_ws,
                            [srow, 0],
                            [LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_a_zz",
                        ),
                        target_memory=pl.Mem.LeftScale,
                    )
                    h_las = pl.tget_scale_addr(h_las, h_la)
                    h_las = pl.set_validshape(h_las, LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K)
                    h_rb = pl.move(h_w_tile, target_memory=pl.Mem.Right)
                    h_rbs = pl.move(h_ws_tile, target_memory=pl.Mem.RightScale)
                    h_rbs = pl.tget_scale_addr(h_rbs, h_rb)
                    h_acc = pl.matmul_mx(h_la, h_las, h_rb, h_rbs)
                    for db in pl.unroll(31):  # D_BLOCKS - 1
                        k0 = (db + 1) * D_CHUNK
                        prev_tile2 = pl.load(
                            prev_norm,
                            [t0, prev_base + k0],
                            [LINEAR_T_TILE, D_CHUNK],
                            valid_shapes=[t_rows, D_CHUNK],
                            target_memory=pl.Mem.Vec,
                        )
                        prev_q2, prev_s2 = pl.mx_quant(prev_tile2, mode="mxfp8_e4m3")
                        h_w_tile2 = pl.load(
                            h_proj_w,
                            [k0, n0],
                            [D_CHUNK, OUT_CHUNK],
                            target_memory=pl.Mem.Mat,
                        )
                        h_ws_tile2 = pl.load(
                            h_proj_w_scale,
                            [(nb * D_BLOCKS + k0 // D_CHUNK) * _KS, 0],
                            [_KS, OUT_CHUNK],
                            target_memory=pl.Mem.Mat,
                            mx_layout="mx_b_nn",
                        )
                        srow2 = ((tc * OUT_BLOCKS + nb) * D_BLOCKS + k0 // D_CHUNK) * LINEAR_T_TILE
                        h_la2 = pl.move(
                            pl.move(pl.tile.reinterpret_view(prev_q2, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
                            target_memory=pl.Mem.Left,
                        )
                        h_la2 = pl.set_validshape(h_la2, LINEAR_T_TILE, D_CHUNK)
                        pl.store(pl.tile.reinterpret_view(prev_s2, pl.FP8E8M0), [srow2, 0], mx_scale_ws)
                        h_las2 = pl.move(
                            pl.load(
                                mx_scale_ws,
                                [srow2, 0],
                                [LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K],
                                target_memory=pl.Mem.Mat,
                                mx_layout="mx_a_zz",
                            ),
                            target_memory=pl.Mem.LeftScale,
                        )
                        h_las2 = pl.tget_scale_addr(h_las2, h_la2)
                        h_las2 = pl.set_validshape(h_las2, LINEAR_T_TILE, D_CHUNK // MX_BLOCK_K)
                        h_rb2 = pl.move(h_w_tile2, target_memory=pl.Mem.Right)
                        h_rbs2 = pl.move(h_ws_tile2, target_memory=pl.Mem.RightScale)
                        h_rbs2 = pl.tget_scale_addr(h_rbs2, h_rb2)
                        h_acc = pl.matmul_mx_acc(h_acc, h_la2, h_las2, h_rb2, h_rbs2)
                    # Store h only; FP32 e+h runs in mtp_projection_linear_add.
                    pl.store(h_acc, [t0, prev_out], out_pad)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_linear_add"):
                # Pure vec scope so the add is not dropped beside cube matmul_mx.
                e_fp = pl.load(
                    e_proj_pad,
                    [t0, n0],
                    [LINEAR_T_TILE, OUT_CHUNK],
                    valid_shapes=[t_rows, OUT_CHUNK],
                    target_memory=pl.Mem.Vec,
                )
                for hc in pl.unroll(4):  # HC_MULT
                    prev_out = hc * D + n0
                    h_fp = pl.load(
                        out_pad,
                        [t0, prev_out],
                        [LINEAR_T_TILE, OUT_CHUNK],
                        valid_shapes=[t_rows, OUT_CHUNK],
                        target_memory=pl.Mem.Vec,
                    )
                    pl.store(pl.add(e_fp, h_fp), [t0, prev_out], out_pad)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_output"):
        for t0 in pl.pipeline(0, t_dim, T_TILE, stage=2):
            for hc in pl.range(HC_MULT):
                out_base = hc * D
                for n0 in pl.range(0, D, OUT_CHUNK):
                    out_flat[t0 : t0 + T_TILE, out_base + n0 : out_base + n0 + OUT_CHUNK] = pl.cast(
                        out_pad[t0 : t0 + T_TILE, out_base + n0 : out_base + n0 + OUT_CHUNK],
                        target_type=pl.BF16,
                        mode="rint",
                    )

    hidden_states_out = pl.reshape(out_flat, [t_dim, HC_MULT, D])
    return hidden_states_out


@pl.jit
def mtp_projection_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    e_proj_w_scale: pl.Tensor[[_WQ_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    h_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    h_proj_w_scale: pl.Tensor[[_WQ_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    hidden_states_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    hidden_states.bind_dynamic(0, T_DYN)
    prev_hidden_states.bind_dynamic(0, T_DYN)
    hidden_states_out.bind_dynamic(0, T_DYN)
    return mtp_projection(
        hidden_states,
        prev_hidden_states,
        enorm_w,
        hnorm_w,
        e_proj_w,
        e_proj_w_scale,
        h_proj_w,
        h_proj_w_scale,
        hidden_states_out,
    )


def _rms_norm(x, weight):
    import torch

    shape = x.shape
    x_2d = x.reshape(-1, D).float()
    sq_sum = torch.zeros(x_2d.shape[0], 1, dtype=torch.float32)
    for k0 in range(0, D, D_CHUNK):
        x_chunk = x_2d[:, k0:k0 + D_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    inv = torch.rsqrt(sq_sum * D_INV + EPS)
    return (x_2d * inv * weight.float().view(1, D)).reshape(shape)


def golden_mtp_projection(tensors):
    import torch

    t = tensors["hidden_states"].shape[0]
    hidden_norm = _rms_norm(tensors["hidden_states"], tensors["enorm_w"])
    prev_norm = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"])

    def _b_scale(s):
        return unpack_scale_b_nn_tiled(
            s,
            k_tile_rows=_KS,
            n_tile=OUT_CHUNK,
            logical_k=D_SCALE,
            logical_n=D,
        )

    hidden_q, hidden_s = dynamic_mx_quant_e4m3(hidden_norm.float())
    hidden_e = mx_matmul_fp8(
        hidden_q,
        hidden_s,
        tensors["e_proj_w"],
        _b_scale(tensors["e_proj_w_scale"]),
    )

    prev_q, prev_s = dynamic_mx_quant_e4m3(prev_norm.float().reshape(-1, D))
    prev_h = mx_matmul_fp8(
        prev_q,
        prev_s,
        tensors["h_proj_w"],
        _b_scale(tensors["h_proj_w_scale"]),
    ).reshape(t, HC_MULT, D)

    tensors["hidden_states_out"][:] = (hidden_e.unsqueeze(1) + prev_h).to(torch.bfloat16)


def build_tensor_specs(batch=DECODE_BATCH, seq=DECODE_SEQ):
    import torch
    from golden import TensorSpec

    t = batch * seq

    e_proj_w, e_proj_w_scale = gen_mxfp8_weight_kn(
        (D, D), dequant_std=0.1, chan_cv=0.50, n_tile=OUT_CHUNK, k_tile=D_CHUNK
    )
    h_proj_w, h_proj_w_scale = gen_mxfp8_weight_kn(
        (D, D), dequant_std=0.1, chan_cv=0.50, n_tile=OUT_CHUNK, k_tile=D_CHUNK
    )

    return [
        TensorSpec("hidden_states", [t, D], torch.bfloat16, init_value=lambda: torch.randn(t, D)),
        TensorSpec("prev_hidden_states", [t, HC_MULT, D], torch.bfloat16, init_value=lambda: torch.randn(t, HC_MULT, D)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("e_proj_w", [D, D], torch.float8_e4m3fn, init_value=lambda: e_proj_w),
        TensorSpec(
            "e_proj_w_scale", [_WQ_SCALE_ROWS, OUT_CHUNK], torch.float8_e8m0fnu, init_value=lambda: e_proj_w_scale
        ),
        TensorSpec("h_proj_w", [D, D], torch.float8_e4m3fn, init_value=lambda: h_proj_w),
        TensorSpec(
            "h_proj_w_scale", [_WQ_SCALE_ROWS, OUT_CHUNK], torch.float8_e8m0fnu, init_value=lambda: h_proj_w_scale
        ),
        TensorSpec("hidden_states_out", [t, HC_MULT, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    modes = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }
    mtp_tol = ATOL_RTOL["mtp_mxfp8"]

    for mode in (modes if args.mode == "all" else [args.mode]):
        batch, seq = modes[mode]
        print(f"--- mtp_projection {mode}: B={batch}, S={seq} ---")
        result = run_jit(
            fn=mtp_projection_test,
            specs=build_tensor_specs(batch, seq),
            golden_fn=golden_mtp_projection,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=mtp_tol["rtol"],
            atol=mtp_tol["atol"],
            compare_fn={
                # Two MXFP8 terms sum before the next block's normalization; near-zero
                # cancellation leaves more relative outliers than single-projection paths.
                "hidden_states_out": ratio_allclose(
                    atol=mtp_tol["atol"],
                    rtol=mtp_tol["rtol"],
                    max_error_ratio=mtp_tol["pct"],
                ),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
