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
    ACTIVE as M,
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
LINEAR_K_TILE = 512
OUT_TILE = 256
NORM_K_TILE = 1024
QUANT_TILE = 1024
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
    hidden_flat = pl.reshape(hidden_states, [t_dim, D])
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    hidden_i8 = pl.create_tensor([t_linear, D], dtype=pl.INT8)
    hidden_scale_dq = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for hidden_block in pl.spmd(t_dim // T_TILE, name_hint="mtp_projection_hidden"):
        t0 = hidden_block * T_TILE
        hidden_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        hidden_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.pipeline(0, D, NORM_K_TILE, stage=4):
            hidden_chunk_bf16 = hidden_flat[t0 : t0 + T_TILE, k0 : k0 + NORM_K_TILE]
            hidden_chunk = pl.cast(hidden_chunk_bf16, target_type=pl.FP32)
            enorm = pl.reshape(enorm_w[k0 : k0 + NORM_K_TILE], [1, NORM_K_TILE])
            e_smooth = pl.reshape(e_proj_smooth[k0 : k0 + NORM_K_TILE], [1, NORM_K_TILE])
            enorm_smooth = pl.mul(enorm, e_smooth)
            hidden_xg = pl.col_expand_mul(hidden_chunk, enorm_smooth)
            hidden_sq = pl.mul(hidden_chunk, hidden_chunk)
            hidden_sq_row = pl.row_sum(hidden_sq)
            hidden_sq_partial = pl.reshape(hidden_sq_row, [1, T_TILE])
            hidden_sq_sum = pl.add(hidden_sq_sum, hidden_sq_partial)
            hidden_abs = pl.abs(hidden_xg)
            hidden_amax_row = pl.row_max(hidden_abs)
            hidden_amax_col = pl.reshape(hidden_amax_row, [1, T_TILE])
            hidden_amax = pl.maximum(hidden_amax, hidden_amax_col)
        hidden_sq_mean = pl.mul(hidden_sq_sum, D_INV)
        hidden_rms_arg = pl.add(hidden_sq_mean, EPS)
        hidden_inv = pl.rsqrt(hidden_rms_arg, high_precision=True)
        hidden_scale_max = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        hidden_quant_scale_row = pl.div(hidden_scale_max, hidden_amax)
        hidden_sq_recip = pl.recip(hidden_quant_scale_row)
        hidden_scale_row = pl.mul(hidden_inv, hidden_sq_recip)
        hidden_scale_col = pl.reshape(hidden_scale_row, [T_TILE, 1])
        hidden_scale_dq[t0 : t0 + T_TILE, 0:1] = hidden_scale_col
        hidden_quant_scale_col = pl.reshape(hidden_quant_scale_row, [T_TILE, 1])
        for k0 in pl.pipeline(0, D, QUANT_TILE, stage=4):
            hidden_q_bf16 = hidden_flat[t0 : t0 + T_TILE, k0 : k0 + QUANT_TILE]
            hidden_q_chunk = pl.cast(hidden_q_bf16, target_type=pl.FP32)
            enorm_q = pl.reshape(enorm_w[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            e_smooth_q = pl.reshape(e_proj_smooth[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            enorm_smooth_q = pl.mul(enorm_q, e_smooth_q)
            hidden_q_xg = pl.col_expand_mul(hidden_q_chunk, enorm_smooth_q)
            hidden_q_scaled = pl.row_expand_mul(hidden_q_xg, hidden_quant_scale_col)
            hidden_q_i32 = pl.cast(hidden_q_scaled, target_type=pl.INT32, mode="rint")
            hidden_q_half = pl.cast(hidden_q_i32, target_type=pl.FP16, mode="round")
            hidden_q_i8 = pl.cast(hidden_q_half, target_type=pl.INT8, mode="trunc")
            hidden_i8[t0 : t0 + T_TILE, k0 : k0 + QUANT_TILE] = hidden_q_i8

    prev_flat = pl.reshape(prev_hidden_states, [t_dim, HC_DIM])
    prev_linear_rows = t_linear * HC_MULT
    prev_i8 = pl.create_tensor([prev_linear_rows, D], dtype=pl.INT8)
    prev_scale_dq = pl.create_tensor([HC_MULT, t_linear], dtype=pl.FP32)
    for prev_block in pl.spmd((t_dim // T_TILE) * HC_MULT, name_hint="mtp_projection_prev"):
        prev_t = prev_block // HC_MULT
        hc = prev_block - prev_t * HC_MULT
        t0 = prev_t * T_TILE
        prev_base = hc * D
        prev_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        prev_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.pipeline(0, D, NORM_K_TILE, stage=4):
            prev_chunk = prev_flat[t0 : t0 + T_TILE, prev_base + k0 : prev_base + k0 + NORM_K_TILE]
            hnorm = pl.reshape(hnorm_w[k0 : k0 + NORM_K_TILE], [1, NORM_K_TILE])
            h_smooth = pl.reshape(h_proj_smooth[k0 : k0 + NORM_K_TILE], [1, NORM_K_TILE])
            hnorm_smooth = pl.mul(hnorm, h_smooth)
            prev_xg = pl.col_expand_mul(prev_chunk, hnorm_smooth)
            prev_sq = pl.mul(prev_chunk, prev_chunk)
            prev_sq_row = pl.row_sum(prev_sq)
            prev_sq_partial = pl.reshape(prev_sq_row, [1, T_TILE])
            prev_sq_sum = pl.add(prev_sq_sum, prev_sq_partial)
            prev_abs = pl.abs(prev_xg)
            prev_amax_row = pl.row_max(prev_abs)
            prev_amax_col = pl.reshape(prev_amax_row, [1, T_TILE])
            prev_amax = pl.maximum(prev_amax, prev_amax_col)
        prev_sq_mean = pl.mul(prev_sq_sum, D_INV)
        prev_rms_arg = pl.add(prev_sq_mean, EPS)
        prev_inv = pl.rsqrt(prev_rms_arg, high_precision=True)
        prev_scale_max = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        prev_quant_scale_row = pl.div(prev_scale_max, prev_amax)
        prev_sq_recip = pl.recip(prev_quant_scale_row)
        prev_scale_row = pl.mul(prev_inv, prev_sq_recip)
        prev_scale_dq[hc : hc + 1, t0 : t0 + T_TILE] = prev_scale_row
        prev_quant_scale_col = pl.reshape(prev_quant_scale_row, [T_TILE, 1])
        prev_q_block = t0 // LINEAR_T_TILE
        prev_q_row0 = prev_q_block * LINEAR_HC_TILE + hc * LINEAR_T_TILE + (t0 - prev_q_block * LINEAR_T_TILE)
        for k0 in pl.pipeline(0, D, QUANT_TILE, stage=4):
            prev_q_chunk = prev_flat[t0 : t0 + T_TILE, prev_base + k0 : prev_base + k0 + QUANT_TILE]
            hnorm_q = pl.reshape(hnorm_w[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            h_smooth_q = pl.reshape(h_proj_smooth[k0 : k0 + QUANT_TILE], [1, QUANT_TILE])
            hnorm_smooth_q = pl.mul(hnorm_q, h_smooth_q)
            prev_q_xg = pl.col_expand_mul(prev_q_chunk, hnorm_smooth_q)
            prev_q_scaled = pl.row_expand_mul(prev_q_xg, prev_quant_scale_col)
            prev_q_i32 = pl.cast(prev_q_scaled, target_type=pl.INT32, mode="rint")
            prev_q_half = pl.cast(prev_q_i32, target_type=pl.FP16, mode="round")
            prev_q_i8 = pl.cast(prev_q_half, target_type=pl.INT8, mode="trunc")
            prev_i8[prev_q_row0 : prev_q_row0 + T_TILE, k0 : k0 + QUANT_TILE] = prev_q_i8

    out_flat = pl.reshape(hidden_states_out, [t_dim, HC_DIM])
    for linear_block in pl.spmd((t_linear // LINEAR_T_TILE) * (D // OUT_TILE), name_hint="mtp_projection_linear"):
        linear_t = linear_block // (D // OUT_TILE)
        t0 = linear_t * LINEAR_T_TILE
        n0 = (linear_block - linear_t * (D // OUT_TILE)) * OUT_TILE
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        hidden_a0 = hidden_i8[t0 : t0 + LINEAR_T_TILE, 0:LINEAR_K_TILE]
        e_w0 = e_proj_w[n0 : n0 + OUT_TILE, 0:LINEAR_K_TILE]
        hidden_acc = pl.matmul(hidden_a0, e_w0, b_trans=True, out_dtype=pl.INT32)
        for k0 in pl.pipeline(LINEAR_K_TILE, D, LINEAR_K_TILE, stage=2):
            hidden_a = hidden_i8[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
            e_w = e_proj_w[n0 : n0 + OUT_TILE, k0 : k0 + LINEAR_K_TILE]
            hidden_acc = pl.matmul_acc(hidden_acc, hidden_a, e_w, b_trans=True)
        e_scale = pl.reshape(e_proj_w_scale[n0 : n0 + OUT_TILE], [1, OUT_TILE])
        hidden_acc_fp32 = pl.cast(hidden_acc, target_type=pl.FP32, mode="none")
        hidden_row_deq = pl.row_expand_mul(hidden_acc_fp32, hidden_scale_dq[t0 : t0 + LINEAR_T_TILE, 0:1])
        hidden_deq = pl.col_expand_mul(hidden_row_deq, e_scale)
        prev_row0 = linear_t * LINEAR_HC_TILE
        prev_a0 = prev_i8[prev_row0 : prev_row0 + LINEAR_HC_TILE, 0:LINEAR_K_TILE]
        h_w0 = h_proj_w[n0 : n0 + OUT_TILE, 0:LINEAR_K_TILE]
        prev_acc = pl.matmul(prev_a0, h_w0, b_trans=True, out_dtype=pl.INT32)
        for k0 in pl.pipeline(LINEAR_K_TILE, D, LINEAR_K_TILE, stage=2):
            prev_a = prev_i8[prev_row0 : prev_row0 + LINEAR_HC_TILE, k0 : k0 + LINEAR_K_TILE]
            h_w = h_proj_w[n0 : n0 + OUT_TILE, k0 : k0 + LINEAR_K_TILE]
            prev_acc = pl.matmul_acc(prev_acc, prev_a, h_w, b_trans=True)
        prev_deq_all = pl.cast(prev_acc, target_type=pl.FP32, mode="none")
        h_scale = pl.reshape(h_proj_w_scale[n0 : n0 + OUT_TILE], [1, OUT_TILE])
        for hc in pl.unroll(HC_MULT):
            hc_row0 = hc * LINEAR_T_TILE
            prev_deq_tile = prev_deq_all[hc_row0 : hc_row0 + LINEAR_T_TILE, 0:OUT_TILE]
            prev_scale = prev_scale_dq[hc : hc + 1, t0 : t0 + LINEAR_T_TILE]
            prev_scale_col = pl.reshape(prev_scale, [LINEAR_T_TILE, 1])
            prev_row_deq = pl.row_expand_mul(prev_deq_tile, prev_scale_col)
            prev_deq = pl.col_expand_mul(prev_row_deq, h_scale)
            acc = pl.add(hidden_deq, prev_deq)
            prev_out = hc * D + n0
            acc_valid = pl.set_validshape(acc, t_rows, OUT_TILE)
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
    for k0 in range(0, D, NORM_K_TILE):
        x_chunk = x_2d[:, k0:k0 + NORM_K_TILE]
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
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
