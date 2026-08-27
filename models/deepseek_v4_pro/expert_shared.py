# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared-expert FFN for decode."""


import pypto.language as pl

from config import (ACTIVE as M, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS)


# model config
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 16
SH_ROW_PAD = 8
SH_ROW_TILE = 2
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE
assert T < SH_M_TILE or T % SH_M_TILE == 0
assert SH_VALID_M % SH_ROW_TILE == 0
K_TILE = 512
INTER_K_TILE = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = 1024
D_OUT_TILE = 256
QUANT_TILE = 2048 if M.name == "flash" else 1024


@pl.jit.inline
def expert_shared(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    for mt in pl.parallel(T_PAD // SH_M_TILE):
        ts0 = mt * SH_M_TILE

        gate_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)

        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_gate_mm"):
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw1_k = shared_w1[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True)
            gate_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        up_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)

        for nb_idx in pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_up_mm"):
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw3_k = shared_w3[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True)
            up_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        h_tile_i8 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT8)
        h_tile_scale_dq = pl.create_tensor([SH_M_TILE, SH_ROW_PAD], dtype=pl.FP32, manual_dep=True)
        for row_block in pl.spmd(SH_VALID_M // SH_ROW_TILE, name_hint="sh_gate_up_act_q"):
            row0 = row_block * SH_ROW_TILE
            x_scale = pl.slice(x_local_scale_dq, [SH_ROW_PAD, 1], [ts0 + row0, 0], valid_shape=[SH_ROW_TILE, 1])
            row_amax = pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for part in pl.pipeline(0, MOE_INTER // ACT_INTER_TILE, stage=2):
                n0 = part * ACT_INTER_TILE
                gate_rows_i32 = pl.slice(
                    gate_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROW_TILE, ACT_INTER_TILE],
                )
                up_rows_i32 = pl.slice(
                    up_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROW_TILE, ACT_INTER_TILE],
                )
                w1_scale = pl.reshape(shared_w1_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                w3_scale = pl.reshape(shared_w3_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_fp32 = pl.cast(gate_rows_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_rows_i32, target_type=pl.FP32, mode="none")
                gate_scaled = pl.row_expand_mul(gate_fp32, x_scale)
                gate_fp32 = pl.col_expand_mul(gate_scaled, w1_scale)
                up_scaled = pl.row_expand_mul(up_fp32, x_scale)
                up_fp32 = pl.col_expand_mul(up_scaled, w3_scale)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_max = pl.minimum(up_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(up_max, -SWIGLU_LIMIT)
                gate_neg = pl.neg(gate_fp32)
                gate_exp = pl.exp(gate_neg)
                gate_exp_one = pl.add(gate_exp, 1.0)
                sigmoid = pl.recip(gate_exp_one)
                silu = pl.mul(gate_fp32, sigmoid)
                gated = pl.mul(silu, up_fp32)
                gated_abs = pl.abs(gated)
                gated_amax = pl.row_max(gated_abs)
                part_amax = pl.reshape(gated_amax, [1, SH_ROW_PAD])
                row_amax = pl.maximum(row_amax, part_amax)
                h_tile_fp32[row0 : row0 + SH_ROW_TILE, n0 : n0 + ACT_INTER_TILE] = gated[0:SH_ROW_TILE, :]

            scale_numerator = pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            row_scale_q = pl.div(scale_numerator, row_amax)
            row_scale_q_col = pl.reshape(row_scale_q, [SH_ROW_PAD, 1])
            row_scale_dq = pl.recip(row_scale_q)
            row_scale_dq_col = pl.reshape(row_scale_dq, [SH_ROW_PAD, 1])
            scale_base = pl.full([SH_ROW_PAD, SH_ROW_PAD], dtype=pl.FP32, value=0.0)
            row_scale_dq_full = pl.row_expand(scale_base, row_scale_dq_col)
            h_tile_scale_dq[row0 : row0 + SH_ROW_TILE, :] = row_scale_dq_full[0:SH_ROW_TILE, :]
            for q_idx in pl.pipeline(0, MOE_INTER // QUANT_TILE, stage=2):
                k0 = q_idx * QUANT_TILE
                h_fp32 = pl.slice(
                    h_tile_fp32,
                    [SH_ROW_PAD, QUANT_TILE],
                    [row0, k0],
                    valid_shape=[SH_ROW_TILE, QUANT_TILE],
                )
                h_scaled = pl.row_expand_mul(h_fp32, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_i8 = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")
                h_tile_i8[row0 : row0 + SH_ROW_TILE, k0 : k0 + QUANT_TILE] = h_i8[0:SH_ROW_TILE, :]

        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K_TILE, stage=2):
                hs_k = h_tile_i8[:, k0 : k0 + INTER_K_TILE]
                sw2_k = shared_w2[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K_TILE]
                if k0 == 0:
                    y_acc = pl.matmul(hs_k, sw2_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True)

            h_scale = pl.row_max(h_tile_scale_dq[:, :])
            w2_scale_tile = pl.reshape(shared_w2_scale[d0 : d0 + D_OUT_TILE], [1, D_OUT_TILE])
            y_2d = pl.cast(y_acc, target_type=pl.FP32, mode="none")
            y_scaled = pl.row_expand_mul(y_2d, h_scale)
            y_2d = pl.col_expand_mul(y_scaled, w2_scale_tile)
            y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode="rint")
            sh[ts0 : ts0 + SH_VALID_M, d0 : d0 + D_OUT_TILE] = y_bf16[0:SH_VALID_M, :]

    return sh


@pl.jit
def expert_shared_test(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local_i8, x_local_scale_dq,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )
    return sh


def _int8_quant_per_row(x):
    """Quantize each row symmetrically to INT8."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_expert_shared(tensors):
    """Compute the shared-expert reference with INT32 accumulation."""
    import torch
    import torch.nn.functional as F

    x_local_i8 = tensors["x_local_i8"]
    x_local_scale_dq = tensors["x_local_scale_dq"].float()
    w1_i8 = tensors["shared_w1"]
    w1_scale = tensors["shared_w1_scale"].float()
    w3_i8 = tensors["shared_w3"]
    w3_scale = tensors["shared_w3_scale"].float()
    w2_i8 = tensors["shared_w2"]
    w2_scale = tensors["shared_w2_scale"].float()

    gate_int = x_local_i8.to(torch.int32) @ w1_i8.to(torch.int32).T
    sh_gate = gate_int.to(torch.float32) * x_local_scale_dq * w1_scale.unsqueeze(0)
    up_int = x_local_i8.to(torch.int32) @ w3_i8.to(torch.int32).T
    sh_up = up_int.to(torch.float32) * x_local_scale_dq * w3_scale.unsqueeze(0)
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_i8, sh_h_sd = _int8_quant_per_row(sh_h)
    sh_int = sh_h_i8.to(torch.int32) @ w2_i8.to(torch.int32).T
    sh = sh_int.to(torch.float32) * sh_h_sd * w2_scale.unsqueeze(0)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize per-channel INT8 weights and FP32 scales from an MXFP8 grid."""
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        block_amax = Wb.abs().amax(dim=(1, 3), keepdim=True)
        block_scale = (block_amax / FP8_MAX).clamp_min(TINY)
        scale = torch.exp2(torch.ceil(torch.log2(block_scale)))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    weight_base = torch.randn(*shape)
    channel_noise = torch.randn(*shape[:-1], 1)
    channel_gain = torch.exp(chan_cv * channel_noise)
    W = weight_base * channel_gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_local_i8_pre, x_local_sd_pre = _int8_quant_per_row(x_local_bf16)

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)

    return [
        TensorSpec("x_local_i8", [T, D], torch.int8, init_value=lambda: x_local_i8_pre),
        TensorSpec("x_local_scale_dq", [T, 1], torch.float32, init_value=lambda: x_local_sd_pre.float()),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.int8, init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale", [MOE_INTER], torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.int8, init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale", [MOE_INTER], torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2", [D, MOE_INTER], torch.int8, init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("sh", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
