# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 MTP input projection scaffold.

Mirrors the MTP-only prolog in the official implementation:
``e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))``.

``prev_hidden_states`` and ``hidden_states_out`` use token-major pre-``hc_head``
hidden states with HC lanes: ``[T, HC_MULT, D]``.
"""

import pypto.language as pl

from config import (
    ACTIVE as M,
    DECODE_BATCH,
    DECODE_SEQ,
    PREFILL_BATCH,
    PREFILL_SEQ,
)


T_DYN = pl.dynamic("T_DYN")
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = HC_MULT * D
EPS = M.rms_norm_eps
D_INV = 1.0 / D

T_TILE = 8
LINEAR_T_TILE = 16
D_CHUNK = 128
OUT_CHUNK = 256
D_BLOCKS = D // D_CHUNK
OUT_BLOCKS = D // OUT_CHUNK
MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_K_TILES = D // MX_K_TILE
MX_SCALE_ROWS = OUT_BLOCKS * MX_K_TILES * MX_K_SCALE_TILE
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def mtp_projection(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    e_proj_w_scale: pl.Tensor[[MX_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    h_proj_w_scale: pl.Tensor[[MX_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
                    prev_chunk = prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + D_CHUNK]
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
                e_smooth = pl.reshape(e_proj_smooth[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                hidden_norm_tile = pl.col_expand_mul(
                    pl.col_expand_mul(pl.row_expand_mul(hidden_chunk, hidden_inv), enorm),
                    e_smooth,
                )
                hidden_norm = pl.assemble(hidden_norm, hidden_norm_tile, [t0, k0])
                hnorm = pl.reshape(hnorm_w[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                h_smooth = pl.reshape(h_proj_smooth[k0 : k0 + D_CHUNK], [1, D_CHUNK])
                for hc in pl.range(HC_MULT):
                    prev_k0 = hc * D + k0
                    prev_inv = pl.reshape(prev_inv_rms[hc : hc + 1, t0 : t0 + T_TILE], [T_TILE, 1])
                    prev_chunk = prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + D_CHUNK]
                    prev_norm_tile = pl.col_expand_mul(
                        pl.col_expand_mul(pl.row_expand_mul(prev_chunk, prev_inv), hnorm),
                        h_smooth,
                    )
                    prev_norm = pl.assemble(prev_norm, prev_norm_tile, [t0, prev_k0])

    hidden_mx = pl.create_tensor([t_linear, D], dtype=pl.FP8E4M3FN)
    hidden_scale_store = pl.create_tensor(
        [(t_linear // LINEAR_T_TILE) * MX_K_TILES, LINEAR_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    for quant_idx in pl.parallel((t_linear // LINEAR_T_TILE) * MX_K_TILES):
        mt = quant_idx // MX_K_TILES
        kb = quant_idx % MX_K_TILES
        t0 = mt * LINEAR_T_TILE
        k0 = kb * MX_K_TILE
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_e_quant"):
            hidden_src = pl.load(
                hidden_norm,
                [t0, k0],
                [LINEAR_T_TILE, MX_K_TILE],
                valid_shape=[valid_rows, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            hidden_src = pl.fillpad(hidden_src, pad_value=pl.PadValue.zero)
            hidden_q, hidden_scale = pl.quant_mx(hidden_src, layout=pl.MX_A_ZZ)
            pl.store(hidden_q, [t0, k0], hidden_mx)
            hidden_scale_flat = pl.reshape(hidden_scale, [1, LINEAR_T_TILE * MX_K_SCALE_TILE])
            pl.store(hidden_scale_flat, [quant_idx, 0], hidden_scale_store)

    prev_mx = pl.create_tensor([t_linear, HC_DIM], dtype=pl.FP8E4M3FN)
    prev_scale_store = pl.create_tensor(
        [HC_MULT * (t_linear // LINEAR_T_TILE) * MX_K_TILES, LINEAR_T_TILE * MX_K_SCALE_TILE],
        dtype=pl.FP8E8M0,
    )
    for quant_idx in pl.parallel(HC_MULT * (t_linear // LINEAR_T_TILE) * MX_K_TILES):
        hc = quant_idx // ((t_linear // LINEAR_T_TILE) * MX_K_TILES)
        local_idx = quant_idx % ((t_linear // LINEAR_T_TILE) * MX_K_TILES)
        mt = local_idx // MX_K_TILES
        kb = local_idx % MX_K_TILES
        t0 = mt * LINEAR_T_TILE
        k0 = kb * MX_K_TILE
        prev_k0 = hc * D + k0
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_h_quant"):
            prev_src = pl.load(
                prev_norm,
                [t0, prev_k0],
                [LINEAR_T_TILE, MX_K_TILE],
                valid_shape=[valid_rows, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            prev_src = pl.fillpad(prev_src, pad_value=pl.PadValue.zero)
            prev_q, prev_scale = pl.quant_mx(prev_src, layout=pl.MX_A_ZZ)
            pl.store(prev_q, [t0, prev_k0], prev_mx)
            prev_scale_flat = pl.reshape(prev_scale, [1, LINEAR_T_TILE * MX_K_SCALE_TILE])
            pl.store(prev_scale_flat, [quant_idx, 0], prev_scale_store)

    e_partial = pl.create_tensor(
        [(t_linear // LINEAR_T_TILE) * MX_K_TILES * LINEAR_T_TILE, D], dtype=pl.FP32
    )
    for task_idx in pl.parallel((t_linear // LINEAR_T_TILE) * MX_K_TILES * OUT_BLOCKS):
        mt = task_idx // (MX_K_TILES * OUT_BLOCKS)
        local_idx = task_idx % (MX_K_TILES * OUT_BLOCKS)
        kb = local_idx // OUT_BLOCKS
        nb = local_idx % OUT_BLOCKS
        t0 = mt * LINEAR_T_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * OUT_CHUNK
        scale_idx = mt * MX_K_TILES + kb
        hidden_scale_slice = hidden_scale_store[scale_idx : scale_idx + 1, :]
        hidden_scale_mx = pl.tensor.view(
            hidden_scale_slice, [LINEAR_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ
        )
        e_scale_offset = (nb * MX_K_TILES + kb) * MX_K_SCALE_TILE
        e_scale_slice = e_proj_w_scale[e_scale_offset : e_scale_offset + MX_K_SCALE_TILE, :]
        e_scale_mx = pl.tensor.view(e_scale_slice, [MX_K_SCALE_TILE, OUT_CHUNK], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_e_linear"):
            hidden_k = pl.move(
                pl.load(hidden_mx, [t0, k0], [LINEAR_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            hidden_scale_k = pl.move(
                pl.load(hidden_scale_mx, [0, 0], [LINEAR_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            e_w_k = pl.move(
                pl.load(e_proj_w, [k0, n0], [MX_K_TILE, OUT_CHUNK], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            e_scale_k = pl.move(
                pl.load(e_scale_mx, [0, 0], [MX_K_SCALE_TILE, OUT_CHUNK], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            e_acc = pl.matmul_mx(hidden_k, hidden_scale_k, e_w_k, e_scale_k)
            e_row = (mt * MX_K_TILES + kb) * LINEAR_T_TILE
            pl.store(e_acc, [e_row, n0], e_partial)

    h_partial = pl.create_tensor(
        [HC_MULT * (t_linear // LINEAR_T_TILE) * MX_K_TILES * LINEAR_T_TILE, D], dtype=pl.FP32
    )
    for task_idx in pl.parallel(HC_MULT * (t_linear // LINEAR_T_TILE) * MX_K_TILES * OUT_BLOCKS):
        hc = task_idx // ((t_linear // LINEAR_T_TILE) * MX_K_TILES * OUT_BLOCKS)
        local_idx = task_idx % ((t_linear // LINEAR_T_TILE) * MX_K_TILES * OUT_BLOCKS)
        mt = local_idx // (MX_K_TILES * OUT_BLOCKS)
        tile_idx = local_idx % (MX_K_TILES * OUT_BLOCKS)
        kb = tile_idx // OUT_BLOCKS
        nb = tile_idx % OUT_BLOCKS
        t0 = mt * LINEAR_T_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * OUT_CHUNK
        prev_k0 = hc * D + k0
        scale_idx = (hc * (t_linear // LINEAR_T_TILE) + mt) * MX_K_TILES + kb
        prev_scale_slice = prev_scale_store[scale_idx : scale_idx + 1, :]
        prev_scale_mx = pl.tensor.view(prev_scale_slice, [LINEAR_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ)
        h_scale_offset = (nb * MX_K_TILES + kb) * MX_K_SCALE_TILE
        h_scale_slice = h_proj_w_scale[h_scale_offset : h_scale_offset + MX_K_SCALE_TILE, :]
        h_scale_mx = pl.tensor.view(h_scale_slice, [MX_K_SCALE_TILE, OUT_CHUNK], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_h_linear"):
            prev_k = pl.move(
                pl.load(prev_mx, [t0, prev_k0], [LINEAR_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            prev_scale_k = pl.move(
                pl.load(prev_scale_mx, [0, 0], [LINEAR_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            h_w_k = pl.move(
                pl.load(h_proj_w, [k0, n0], [MX_K_TILE, OUT_CHUNK], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            h_scale_k = pl.move(
                pl.load(h_scale_mx, [0, 0], [MX_K_SCALE_TILE, OUT_CHUNK], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            h_acc = pl.matmul_mx(prev_k, prev_scale_k, h_w_k, h_scale_k)
            h_row = ((hc * (t_linear // LINEAR_T_TILE) + mt) * MX_K_TILES + kb) * LINEAR_T_TILE
            pl.store(h_acc, [h_row, n0], h_partial)

    for task_idx in pl.parallel(HC_MULT * (t_linear // LINEAR_T_TILE) * OUT_BLOCKS):
        hc = task_idx // ((t_linear // LINEAR_T_TILE) * OUT_BLOCKS)
        local_idx = task_idx % ((t_linear // LINEAR_T_TILE) * OUT_BLOCKS)
        mt = local_idx // OUT_BLOCKS
        nb = local_idx % OUT_BLOCKS
        t0 = mt * LINEAR_T_TILE
        n0 = nb * OUT_CHUNK
        out_n0 = hc * D + n0
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_reduce"):
            e_sum = pl.tile.full([LINEAR_T_TILE, OUT_CHUNK], dtype=pl.FP32, value=0.0)
            h_sum = pl.tile.full([LINEAR_T_TILE, OUT_CHUNK], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(MX_K_TILES, stage=2):
                e_row = (mt * MX_K_TILES + kb) * LINEAR_T_TILE
                h_row = ((hc * (t_linear // LINEAR_T_TILE) + mt) * MX_K_TILES + kb) * LINEAR_T_TILE
                e_part = pl.load(
                    e_partial, [e_row, n0], [LINEAR_T_TILE, OUT_CHUNK], target_memory=pl.Mem.Vec
                )
                h_part = pl.load(
                    h_partial, [h_row, n0], [LINEAR_T_TILE, OUT_CHUNK], target_memory=pl.Mem.Vec
                )
                e_sum = pl.add(e_sum, e_part)
                h_sum = pl.add(h_sum, h_part)
            out_tile = pl.add(e_sum, h_sum)
            pl.store(out_tile, [t0, out_n0], out_pad)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_output"):
        for t0 in pl.pipeline(0, t_dim, T_TILE, stage=2):
            for hc in pl.range(HC_MULT):
                out_base = hc * D
                for n0 in pl.range(0, D, OUT_CHUNK):
                    out_flat[t0 : t0 + T_TILE, out_base + n0 : out_base + n0 + OUT_CHUNK] = out_pad[
                        t0 : t0 + T_TILE,
                        out_base + n0 : out_base + n0 + OUT_CHUNK,
                    ]

    hidden_states_out = pl.reshape(out_flat, [t_dim, HC_MULT, D])
    return hidden_states_out


@pl.jit
def mtp_projection_test(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    e_proj_w_scale: pl.Tensor[[MX_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.FP8E4M3FN],
    h_proj_w_scale: pl.Tensor[[MX_SCALE_ROWS, OUT_CHUNK], pl.FP8E8M0],
    h_proj_smooth: pl.Tensor[[D], pl.FP32],
    hidden_states_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
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
        e_proj_smooth,
        h_proj_w,
        h_proj_w_scale,
        h_proj_smooth,
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
    from expert_shared import _dynamic_mxfp8_matmul, _unpack_b_scale_tiled

    hidden_states = _rms_norm(tensors["hidden_states"], tensors["enorm_w"]) * tensors["e_proj_smooth"].float()
    prev_hidden_states = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"]) * tensors["h_proj_smooth"].float()
    e_scale = _unpack_b_scale_tiled(tensors["e_proj_w_scale"], D, D)
    h_scale = _unpack_b_scale_tiled(tensors["h_proj_w_scale"], D, D)
    hidden_e = _dynamic_mxfp8_matmul(hidden_states.float(), tensors["e_proj_w"], e_scale)
    prev_flat = prev_hidden_states.reshape(-1, D)
    hidden_h = _dynamic_mxfp8_matmul(prev_flat, tensors["h_proj_w"], h_scale).reshape_as(prev_hidden_states)
    tensors["hidden_states_out"][:] = (hidden_e.unsqueeze(1) + hidden_h).to(torch.float32)


def build_tensor_specs(batch=DECODE_BATCH, seq=DECODE_SEQ):
    import torch
    from expert_shared import _gen_mxfp8_weight_kn
    from golden import TensorSpec
    t = batch * seq

    def init_proj_pair():
        return _gen_mxfp8_weight_kn((D, D), dequant_std=0.25 / D ** 0.5, chan_cv=0.25)

    e_proj_cache = None
    h_proj_cache = None

    def init_e_proj_w():
        nonlocal e_proj_cache
        e_proj_cache = init_proj_pair()
        return e_proj_cache[0]

    def init_e_proj_w_scale():
        nonlocal e_proj_cache
        if e_proj_cache is None:
            e_proj_cache = init_proj_pair()
        return e_proj_cache[1]

    def init_h_proj_w():
        nonlocal h_proj_cache
        h_proj_cache = init_proj_pair()
        return h_proj_cache[0]

    def init_h_proj_w_scale():
        nonlocal h_proj_cache
        if h_proj_cache is None:
            h_proj_cache = init_proj_pair()
        return h_proj_cache[1]

    return [
        TensorSpec("hidden_states", [t, D], torch.bfloat16, init_value=lambda: torch.randn(t, D)),
        TensorSpec("prev_hidden_states", [t, HC_MULT, D], torch.float32, init_value=lambda: torch.randn(t, HC_MULT, D)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("e_proj_w", [D, D], torch.float8_e4m3fn, init_value=init_e_proj_w),
        TensorSpec("e_proj_w_scale", [MX_SCALE_ROWS, OUT_CHUNK], torch.float8_e8m0fnu, init_value=init_e_proj_w_scale),
        TensorSpec("e_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("h_proj_w", [D, D], torch.float8_e4m3fn, init_value=init_h_proj_w),
        TensorSpec("h_proj_w_scale", [MX_SCALE_ROWS, OUT_CHUNK], torch.float8_e8m0fnu, init_value=init_h_proj_w_scale),
        TensorSpec("h_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hidden_states_out", [t, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    modes = {
        "decode": (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }
    for mode in (modes if args.mode == "all" else [args.mode]):
        batch, seq = modes[mode]
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
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                # Raw MTP projection adds two W8A8 terms before the next block's
                # normalization, so near-zero cancellation leaves more relative
                # outliers than qkv_proj_rope's post-RMSNorm q/kv outputs.
                "hidden_states_out": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
