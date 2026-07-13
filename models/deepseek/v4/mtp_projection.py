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
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
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
OUT_CHUNK = 128
D_BLOCKS = D // D_CHUNK
OUT_BLOCKS = D // OUT_CHUNK
QUANT_CHUNK = 128
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def mtp_projection(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    prev_hidden_states: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    enorm_w: pl.Tensor[[D], pl.FP32],
    hnorm_w: pl.Tensor[[D], pl.FP32],
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
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
    hidden_i8 = pl.create_tensor([t_linear, D], dtype=pl.INT8)
    prev_i8 = pl.create_tensor([t_linear, HC_DIM], dtype=pl.INT8)
    hidden_inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    prev_inv_rms = pl.create_tensor([HC_MULT, t_linear], dtype=pl.FP32)
    hidden_amax_parts = pl.create_tensor([D_BLOCKS, t_linear], dtype=pl.FP32)
    prev_amax_parts = pl.create_tensor([HC_MULT * D_BLOCKS, t_linear], dtype=pl.FP32)
    hidden_scale_dq = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    prev_scale_dq = pl.create_tensor([HC_MULT, t_linear], dtype=pl.FP32)
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
                hidden_abs = pl.maximum(hidden_norm_tile, pl.neg(hidden_norm_tile))
                hidden_amax_parts = pl.assemble(hidden_amax_parts, pl.reshape(pl.row_max(hidden_abs), [1, T_TILE]), [kb, t0])
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
                    prev_abs = pl.maximum(prev_norm_tile, pl.neg(prev_norm_tile))
                    prev_amax_parts = pl.assemble(
                        prev_amax_parts,
                        pl.reshape(pl.row_max(prev_abs), [1, T_TILE]),
                        [hc * D_BLOCKS + kb, t0],
                    )

    for t0 in pl.parallel(0, t_dim, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_quant"):
            hidden_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for ab in pl.range(D_BLOCKS):
                hidden_amax = pl.maximum(hidden_amax, hidden_amax_parts[ab : ab + 1, t0 : t0 + T_TILE])
            hidden_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), hidden_amax)
            hidden_scale_dq = pl.assemble(hidden_scale_dq, pl.reshape(pl.recip(hidden_sq_row), [T_TILE, 1]), [t0, 0])
            hidden_sq_col = pl.reshape(hidden_sq_row, [T_TILE, 1])
            for k0 in pl.range(0, D, QUANT_CHUNK):
                hidden_q_f32 = hidden_norm[t0 : t0 + T_TILE, k0 : k0 + QUANT_CHUNK]
                hidden_q_i32 = pl.cast(pl.row_expand_mul(hidden_q_f32, hidden_sq_col), target_type=pl.INT32, mode="rint")
                hidden_q_half = pl.cast(hidden_q_i32, target_type=pl.FP16, mode="round")
                hidden_i8 = pl.assemble(hidden_i8, pl.cast(hidden_q_half, target_type=pl.INT8, mode="trunc"), [t0, k0])
            for hc in pl.range(HC_MULT):
                prev_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for ab in pl.range(D_BLOCKS):
                    prev_amax = pl.maximum(prev_amax, prev_amax_parts[hc * D_BLOCKS + ab : hc * D_BLOCKS + ab + 1, t0 : t0 + T_TILE])
                prev_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), prev_amax)
                prev_scale_dq = pl.assemble(prev_scale_dq, pl.reshape(pl.recip(prev_sq_row), [1, T_TILE]), [hc, t0])
                prev_sq_col = pl.reshape(prev_sq_row, [T_TILE, 1])
                for k0 in pl.range(0, D, QUANT_CHUNK):
                    prev_k0 = hc * D + k0
                    prev_q_f32 = prev_norm[t0 : t0 + T_TILE, prev_k0 : prev_k0 + QUANT_CHUNK]
                    prev_q_i32 = pl.cast(pl.row_expand_mul(prev_q_f32, prev_sq_col), target_type=pl.INT32, mode="rint")
                    prev_q_half = pl.cast(prev_q_i32, target_type=pl.FP16, mode="round")
                    prev_i8 = pl.assemble(prev_i8, pl.cast(prev_q_half, target_type=pl.INT8, mode="trunc"), [t0, prev_k0])
    for t0 in pl.parallel(0, t_linear, LINEAR_T_TILE):
        for nb in pl.parallel(0, OUT_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_projection_linear"):
                n0 = nb * OUT_CHUNK
                hidden_a0 = hidden_i8[t0 : t0 + LINEAR_T_TILE, 0:D_CHUNK]
                e_w0 = e_proj_w[n0 : n0 + OUT_CHUNK, 0:D_CHUNK]
                hidden_acc = pl.matmul(hidden_a0, e_w0, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, D_BLOCKS, stage=2):
                    k0 = kb * D_CHUNK
                    hidden_a = hidden_i8[t0 : t0 + LINEAR_T_TILE, k0 : k0 + D_CHUNK]
                    e_w = e_proj_w[n0 : n0 + OUT_CHUNK, k0 : k0 + D_CHUNK]
                    hidden_acc = pl.matmul_acc(hidden_acc, hidden_a, e_w, b_trans=True)
                e_scale = pl.reshape(e_proj_w_scale[n0 : n0 + OUT_CHUNK], [1, OUT_CHUNK])
                hidden_deq = pl.col_expand_mul(
                    pl.row_expand_mul(pl.cast(hidden_acc, target_type=pl.FP32, mode="none"), hidden_scale_dq[t0 : t0 + LINEAR_T_TILE, 0:1]),
                    e_scale,
                )
                h_w0 = h_proj_w[n0 : n0 + OUT_CHUNK, 0:D_CHUNK]
                h_scale = pl.reshape(h_proj_w_scale[n0 : n0 + OUT_CHUNK], [1, OUT_CHUNK])
                for hc in pl.range(HC_MULT):
                    prev_base = hc * D
                    prev_out = prev_base + n0
                    prev_a0 = prev_i8[t0 : t0 + LINEAR_T_TILE, prev_base : prev_base + D_CHUNK]
                    prev_acc = pl.matmul(prev_a0, h_w0, b_trans=True, out_dtype=pl.INT32)
                    for kb in pl.pipeline(1, D_BLOCKS, stage=2):
                        k0 = kb * D_CHUNK
                        prev_k0 = prev_base + k0
                        prev_a = prev_i8[t0 : t0 + LINEAR_T_TILE, prev_k0 : prev_k0 + D_CHUNK]
                        h_w = h_proj_w[n0 : n0 + OUT_CHUNK, k0 : k0 + D_CHUNK]
                        prev_acc = pl.matmul_acc(prev_acc, prev_a, h_w, b_trans=True)
                    prev_deq = pl.col_expand_mul(
                        pl.row_expand_mul(
                            pl.cast(prev_acc, target_type=pl.FP32, mode="none"),
                            pl.reshape(prev_scale_dq[hc : hc + 1, t0 : t0 + LINEAR_T_TILE], [LINEAR_T_TILE, 1]),
                        ),
                        h_scale,
                    )
                    acc = pl.add(hidden_deq, prev_deq)
                    out_pad = pl.assemble(out_pad, acc, [t0, prev_out])

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
    e_proj_w: pl.Tensor[[D, D], pl.INT8],
    e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    e_proj_smooth: pl.Tensor[[D], pl.FP32],
    h_proj_w: pl.Tensor[[D, D], pl.INT8],
    h_proj_w_scale: pl.Tensor[[D], pl.FP32],
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

    hidden_states = _rms_norm(tensors["hidden_states"], tensors["enorm_w"]) * tensors["e_proj_smooth"].float()
    prev_hidden_states = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"]) * tensors["h_proj_smooth"].float()
    hidden_i8, hidden_scale = _quantize_rows(hidden_states.float())
    prev_i8, prev_scale = _quantize_rows(prev_hidden_states.float())
    hidden_e = hidden_i8.to(torch.int32).matmul(tensors["e_proj_w"].to(torch.int32).t()).float()
    hidden_e = hidden_e * hidden_scale * tensors["e_proj_w_scale"].float().view(1, D)
    hidden_h = prev_i8.to(torch.int32).matmul(tensors["h_proj_w"].to(torch.int32).t()).float()
    hidden_h = hidden_h * prev_scale * tensors["h_proj_w_scale"].float().view(1, 1, D)
    tensors["hidden_states_out"][:] = (hidden_e.unsqueeze(1) + hidden_h).to(torch.float32)


def _quantize_rows(x):
    import torch

    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    x_i32 = torch.round(x * scale_quant).to(torch.int32)
    x_i32 = torch.clamp(x_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return x_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def _quantize_weight_per_out(w):
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i32 = torch.round(w.float() * scale_quant.view(-1, 1)).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), 1.0 / scale_quant


def build_tensor_specs(batch=DECODE_BATCH, seq=DECODE_SEQ):
    import torch
    from golden import TensorSpec
    t = batch * seq

    def init_proj_pair():
        w = (0.25 * torch.rand(D, D) / D ** 0.5).to(torch.bfloat16)
        return _quantize_weight_per_out(w)

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
        return e_proj_cache[1].float()

    def init_h_proj_w():
        nonlocal h_proj_cache
        h_proj_cache = init_proj_pair()
        return h_proj_cache[0]

    def init_h_proj_w_scale():
        nonlocal h_proj_cache
        if h_proj_cache is None:
            h_proj_cache = init_proj_pair()
        return h_proj_cache[1].float()

    return [
        TensorSpec("hidden_states", [t, D], torch.bfloat16, init_value=lambda: torch.randn(t, D)),
        TensorSpec("prev_hidden_states", [t, HC_MULT, D], torch.float32, init_value=lambda: torch.randn(t, HC_MULT, D)),
        TensorSpec("enorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hnorm_w", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("e_proj_w", [D, D], torch.int8, init_value=init_e_proj_w),
        TensorSpec("e_proj_w_scale", [D], torch.float32, init_value=init_e_proj_w_scale),
        TensorSpec("e_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("h_proj_w", [D, D], torch.int8, init_value=init_h_proj_w),
        TensorSpec("h_proj_w_scale", [D], torch.float32, init_value=init_h_proj_w_scale),
        TensorSpec("h_proj_smooth", [D], torch.float32, init_value=lambda: torch.ones(D)),
        TensorSpec("hidden_states_out", [t, HC_MULT, D], torch.float32, is_output=True),
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
