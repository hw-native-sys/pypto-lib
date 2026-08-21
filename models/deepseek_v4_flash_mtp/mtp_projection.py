# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MTP input projection: e_proj(enorm(hidden_states)) + h_proj(hnorm(prev_hidden_states))."""

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


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = HC_MULT * D
EPS = M.rms_norm_eps
D_INV = 1.0 / D

# tiling
T_TILE = 8
LINEAR_T_TILE = 16
LINEAR_HC_TILE = LINEAR_T_TILE * HC_MULT
D_TILE = 1024
OUT_TILE = 1024
LINEAR_OUT_TILE = 256
LINEAR_K_TILE = 512
QUANT_TILE = 1024


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
    hidden_i8 = pl.create_tensor([t_linear, D], dtype=pl.INT8)
    hidden_scale_dq = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for hidden_idx in pl.spmd(t_dim // T_TILE, name_hint="mtp_hidden_norm_quant", allow_early_resolve=True):
        t0 = hidden_idx * T_TILE
        hidden_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        hidden_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.pipeline(0, D, D_TILE, stage=1):
            hidden_bf16 = hidden_flat[t0 : t0 + T_TILE, k0 : k0 + D_TILE]
            hidden_chunk = pl.cast(hidden_bf16, target_type=pl.FP32)
            enorm = pl.reshape(enorm_w[k0 : k0 + D_TILE], [1, D_TILE])
            e_smooth = pl.reshape(e_proj_smooth[k0 : k0 + D_TILE], [1, D_TILE])
            hidden_weight = pl.mul(enorm, e_smooth)
            hidden_xg_tile = pl.col_expand_mul(hidden_chunk, hidden_weight)
            hidden_sq = pl.mul(hidden_chunk, hidden_chunk)
            hidden_row_sum = pl.reshape(pl.row_sum(hidden_sq), [1, T_TILE])
            hidden_sq_sum = pl.add(hidden_sq_sum, hidden_row_sum)
            hidden_abs = pl.abs(hidden_xg_tile)
            hidden_row_max = pl.reshape(pl.row_max(hidden_abs), [1, T_TILE])
            hidden_amax = pl.maximum(hidden_amax, hidden_row_max)
        hidden_mean_sq = pl.mul(hidden_sq_sum, D_INV)
        hidden_var = pl.add(hidden_mean_sq, EPS)
        hidden_inv = pl.rsqrt(hidden_var, high_precision=True)
        hidden_scale_max = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        hidden_sq_row = pl.div(hidden_scale_max, hidden_amax)
        hidden_scale_recip = pl.recip(hidden_sq_row)
        hidden_scale = pl.mul(hidden_inv, hidden_scale_recip)
        hidden_scale_tile = pl.reshape(hidden_scale, [T_TILE, 1])
        hidden_scale_dq[t0 : t0 + T_TILE, 0:1] = hidden_scale_tile
        hidden_sq_col = pl.reshape(hidden_sq_row, [T_TILE, 1])
        for k0 in pl.pipeline(0, D, QUANT_TILE, stage=2):
            hidden_quant_bf16 = hidden_flat[t0 : t0 + T_TILE, k0 : k0 + QUANT_TILE]
            hidden_quant_chunk = pl.cast(hidden_quant_bf16, target_type=pl.FP32)
            enorm_quant = pl.reshape(enorm_w[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            e_smooth_quant = pl.reshape(e_proj_smooth[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            hidden_quant_weight = pl.mul(enorm_quant, e_smooth_quant)
            hidden_quant_xg = pl.col_expand_mul(hidden_quant_chunk, hidden_quant_weight)
            hidden_quant_scaled = pl.row_expand_mul(hidden_quant_xg, hidden_sq_col)
            hidden_q_i32 = pl.cast(hidden_quant_scaled, target_type=pl.INT32, mode="rint")
            hidden_q_half = pl.cast(hidden_q_i32, target_type=pl.FP16, mode="round")
            hidden_q_i8 = pl.cast(hidden_q_half, target_type=pl.INT8, mode="trunc")
            hidden_i8[t0 : t0 + T_TILE, k0 : k0 + QUANT_TILE] = hidden_q_i8

    t_linear_hc = t_linear * HC_MULT
    prev_flat = pl.reshape(prev_hidden_states, [t_dim, HC_DIM])
    prev_i8_rows = pl.create_tensor([t_linear_hc, D], dtype=pl.INT8)
    prev_scale_dq = pl.create_tensor([HC_MULT, t_linear], dtype=pl.FP32)
    prev_norm_tasks = (t_dim // T_TILE) * HC_MULT
    for norm_idx in pl.spmd(prev_norm_tasks, name_hint="mtp_prev_norm_quant", allow_early_resolve=True):
        norm_t_idx = norm_idx // HC_MULT
        hc = norm_idx - norm_t_idx * HC_MULT
        t0 = norm_t_idx * T_TILE
        prev_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        prev_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.pipeline(0, D, D_TILE, stage=1):
            prev_k0 = hc * D + k0
            prev_chunk = prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + D_TILE]
            hnorm = pl.reshape(hnorm_w[k0 : k0 + D_TILE], [1, D_TILE])
            h_smooth = pl.reshape(h_proj_smooth[k0 : k0 + D_TILE], [1, D_TILE])
            prev_weight = pl.mul(hnorm, h_smooth)
            prev_xg_tile = pl.col_expand_mul(prev_chunk, prev_weight)
            prev_sq = pl.mul(prev_chunk, prev_chunk)
            prev_row_sum = pl.reshape(pl.row_sum(prev_sq), [1, T_TILE])
            prev_sq_sum = pl.add(prev_sq_sum, prev_row_sum)
            prev_abs = pl.abs(prev_xg_tile)
            prev_row_max = pl.reshape(pl.row_max(prev_abs), [1, T_TILE])
            prev_amax = pl.maximum(prev_amax, prev_row_max)
        prev_mean_sq = pl.mul(prev_sq_sum, D_INV)
        prev_var = pl.add(prev_mean_sq, EPS)
        prev_inv = pl.rsqrt(prev_var, high_precision=True)
        prev_scale_max = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        prev_sq_row = pl.div(prev_scale_max, prev_amax)
        prev_scale_recip = pl.recip(prev_sq_row)
        prev_scale = pl.mul(prev_inv, prev_scale_recip)
        prev_scale_dq[hc : hc + 1, t0 : t0 + T_TILE] = prev_scale
        prev_sq_col = pl.reshape(prev_sq_row, [T_TILE, 1])
        prev_q_group = (t0 // LINEAR_T_TILE) * LINEAR_HC_TILE
        prev_q_hc = hc * LINEAR_T_TILE
        prev_q_row0 = prev_q_group + prev_q_hc + t0 % LINEAR_T_TILE
        for k0 in pl.pipeline(0, D, QUANT_TILE, stage=2):
            prev_k0 = hc * D + k0
            prev_quant_chunk = prev_flat[t0 : t0 + T_TILE, prev_k0 : prev_k0 + QUANT_TILE]
            hnorm_quant = pl.reshape(hnorm_w[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            h_smooth_quant = pl.reshape(h_proj_smooth[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            prev_quant_weight = pl.mul(hnorm_quant, h_smooth_quant)
            prev_quant_xg = pl.col_expand_mul(prev_quant_chunk, prev_quant_weight)
            prev_quant_scaled = pl.row_expand_mul(prev_quant_xg, prev_sq_col)
            prev_q_i32 = pl.cast(prev_quant_scaled, target_type=pl.INT32, mode="rint")
            prev_q_half = pl.cast(prev_q_i32, target_type=pl.FP16, mode="round")
            prev_q_i8 = pl.cast(prev_q_half, target_type=pl.INT8, mode="trunc")
            prev_i8_rows[prev_q_row0 : prev_q_row0 + T_TILE, k0 : k0 + QUANT_TILE] = prev_q_i8

    hidden_acc_pad = pl.create_tensor([t_linear, D], dtype=pl.INT32)
    prev_acc_pad = pl.create_tensor([t_linear_hc, D], dtype=pl.INT32)
    linear_tasks = (t_linear // LINEAR_T_TILE) * (D // LINEAR_OUT_TILE)
    for linear_idx in pl.spmd(linear_tasks, name_hint="mtp_linear"):
        t0 = (linear_idx // (D // LINEAR_OUT_TILE)) * LINEAR_T_TILE
        n0 = (linear_idx % (D // LINEAR_OUT_TILE)) * LINEAR_OUT_TILE
        hidden_a0 = hidden_i8[t0 : t0 + LINEAR_T_TILE, 0:LINEAR_K_TILE]
        e_w0 = e_proj_w[n0 : n0 + LINEAR_OUT_TILE, 0:LINEAR_K_TILE]
        hidden_cube_acc = pl.matmul(hidden_a0, e_w0, b_trans=True, out_dtype=pl.INT32)
        for k0 in pl.pipeline(LINEAR_K_TILE, D, LINEAR_K_TILE, stage=2):
            hidden_a = hidden_i8[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
            e_w = e_proj_w[n0 : n0 + LINEAR_OUT_TILE, k0 : k0 + LINEAR_K_TILE]
            hidden_cube_acc = pl.matmul_acc(hidden_cube_acc, hidden_a, e_w, b_trans=True)
        hidden_acc_pad[t0 : t0 + LINEAR_T_TILE, n0 : n0 + LINEAR_OUT_TILE] = hidden_cube_acc

        prev_row0 = t0 * HC_MULT
        h_w0 = h_proj_w[n0 : n0 + LINEAR_OUT_TILE, 0:LINEAR_K_TILE]
        prev_a0 = prev_i8_rows[prev_row0 : prev_row0 + LINEAR_HC_TILE, 0:LINEAR_K_TILE]
        prev_cube_acc = pl.matmul(prev_a0, h_w0, b_trans=True, out_dtype=pl.INT32)
        for k0 in pl.pipeline(LINEAR_K_TILE, D, LINEAR_K_TILE, stage=2):
            prev_a = prev_i8_rows[prev_row0 : prev_row0 + LINEAR_HC_TILE, k0 : k0 + LINEAR_K_TILE]
            h_w = h_proj_w[n0 : n0 + LINEAR_OUT_TILE, k0 : k0 + LINEAR_K_TILE]
            prev_cube_acc = pl.matmul_acc(prev_cube_acc, prev_a, h_w, b_trans=True)
        prev_acc_pad[prev_row0 : prev_row0 + LINEAR_HC_TILE, n0 : n0 + LINEAR_OUT_TILE] = prev_cube_acc

    out_flat = pl.reshape(hidden_states_out, [t_dim, HC_DIM])
    dequant_tasks = (t_linear // LINEAR_T_TILE) * (D // OUT_TILE)
    for linear_idx in pl.spmd(dequant_tasks, name_hint="mtp_dequant"):
        t0 = (linear_idx // (D // OUT_TILE)) * LINEAR_T_TILE
        n0 = (linear_idx % (D // OUT_TILE)) * OUT_TILE
        e_scale = pl.reshape(e_proj_w_scale[n0 : n0 + OUT_TILE], [1, OUT_TILE])
        hidden_acc = hidden_acc_pad[t0 : t0 + LINEAR_T_TILE, n0 : n0 + OUT_TILE]
        hidden_acc_f32 = pl.cast(hidden_acc, target_type=pl.FP32, mode="none")
        hidden_deq_scaled = pl.row_expand_mul(hidden_acc_f32, hidden_scale_dq[t0 : t0 + LINEAR_T_TILE, 0:1])
        hidden_deq = pl.col_expand_mul(hidden_deq_scaled, e_scale)
        h_scale = pl.reshape(h_proj_w_scale[n0 : n0 + OUT_TILE], [1, OUT_TILE])
        prev_row0 = t0 * HC_MULT
        linear_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        for hc in pl.range(HC_MULT):
            prev_out = hc * D + n0
            prev_hc_row0 = prev_row0 + hc * LINEAR_T_TILE
            prev_acc_i32 = prev_acc_pad[prev_hc_row0 : prev_hc_row0 + LINEAR_T_TILE, n0 : n0 + OUT_TILE]
            prev_hc_acc = pl.cast(prev_acc_i32, target_type=pl.FP32, mode="none")
            prev_hc_scale_row = prev_scale_dq[hc : hc + 1, t0 : t0 + LINEAR_T_TILE]
            prev_hc_scale = pl.reshape(prev_hc_scale_row, [LINEAR_T_TILE, 1])
            prev_hc_scaled = pl.row_expand_mul(prev_hc_acc, prev_hc_scale)
            prev_hc = pl.col_expand_mul(prev_hc_scaled, h_scale)
            acc = pl.add(hidden_deq, prev_hc)
            acc_valid = pl.set_validshape(acc, linear_rows, OUT_TILE)
            out_flat[t0 : t0 + LINEAR_T_TILE, prev_out : prev_out + OUT_TILE] = acc_valid

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
        hidden_states, prev_hidden_states,
        enorm_w, hnorm_w,
        e_proj_w, e_proj_w_scale, e_proj_smooth,
        h_proj_w, h_proj_w_scale, h_proj_smooth,
        hidden_states_out,
    )


def _rms_norm(x, weight):
    import torch

    shape = x.shape
    x_2d = x.reshape(-1, D).float()
    sq_sum = torch.zeros(x_2d.shape[0], 1, dtype=torch.float32)
    for k0 in range(0, D, D_TILE):
        x_chunk = x_2d[:, k0 : k0 + D_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    inv = torch.rsqrt(sq_sum * D_INV + EPS)
    return (x_2d * inv * weight.float().view(1, D)).reshape(shape)


def golden_mtp_projection(tensors):
    import torch

    hidden_norm = _rms_norm(tensors["hidden_states"], tensors["enorm_w"])
    hidden_states = hidden_norm * tensors["e_proj_smooth"].float()
    prev_hidden_norm = _rms_norm(tensors["prev_hidden_states"], tensors["hnorm_w"])
    prev_hidden_states = prev_hidden_norm * tensors["h_proj_smooth"].float()
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
    prev_shape = [t, HC_MULT, D]

    def init_proj_pair():
        w = (0.25 * torch.rand(D, D) / D ** 0.5).to(torch.bfloat16)
        return _quantize_weight_per_out(w)

    def init_prev_hidden_states():
        return torch.randn(*prev_shape)

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
        TensorSpec("prev_hidden_states", prev_shape, torch.float32, init_value=init_prev_hidden_states),
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
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument(
        "--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
        choices=(0, 1, 2, 4),
    )
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

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
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "hidden_states_out": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
