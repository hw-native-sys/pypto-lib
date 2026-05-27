# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared expert compute (decode, EP single-card).

Split out of ``expert_routed.py``: only the shared-expert FFN path lives here.
The routed local experts are computed by ``expert_routed.py``; both kernels
are composed inside ``moe.py``.

The shared expert reuses the per-token INT8 quant already produced by
``gate`` (``x_norm_i8`` + ``x_norm_scale``) — the same INT8 view
that ``dispatch`` packs for the routed path. This avoids a second
amax+rescale of the same tokens.
"""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size

# tiling
T_TILE = 32
K_TILE = 512
INTER_K = 512
SH_INTER_TILE = 64
SH_D_OUT_TILE = 64
QUANT_TILE = 256


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
    sh_tile_fp32 = pl.create_tensor([T, MOE_INTER], dtype=pl.FP32)
    sh_tile_i8 = pl.create_tensor([T, MOE_INTER], dtype=pl.INT8)
    sh_tile_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    for gu_block in pl.spmd((T // T_TILE) * (MOE_INTER // (8 * SH_INTER_TILE)), name_hint="sh_gate_up"):
        gu_tb = gu_block // (MOE_INTER // (8 * SH_INTER_TILE))
        gu_nb = gu_block - gu_tb * (MOE_INTER // (8 * SH_INTER_TILE))
        ts0 = gu_tb * T_TILE
        n_base = gu_nb * (8 * SH_INTER_TILE)
        x_local_scale_dq_tile = x_local_scale_dq[ts0 : ts0 + T_TILE, 0:1]
        for ng in pl.range(8):
            n0 = n_base + ng * SH_INTER_TILE
            sh_gate_acc = pl.create_tensor([T_TILE, SH_INTER_TILE], dtype=pl.INT32)
            sh_up_acc = pl.create_tensor([T_TILE, SH_INTER_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, D // K_TILE, stage=2):
                k0 = kb * K_TILE
                xs_k = x_local_i8[ts0 : ts0 + T_TILE, k0 : k0 + K_TILE]
                sw1_k = shared_w1[n0 : n0 + SH_INTER_TILE, k0 : k0 + K_TILE]
                sw3_k = shared_w3[n0 : n0 + SH_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    sh_gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                    sh_up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    sh_gate_acc = pl.matmul_acc(sh_gate_acc, xs_k, sw1_k, b_trans=True)
                    sh_up_acc = pl.matmul_acc(sh_up_acc, xs_k, sw3_k, b_trans=True)

            sw1_scale_chunk = pl.reshape(shared_w1_scale[n0 : n0 + SH_INTER_TILE], [1, SH_INTER_TILE])
            sw3_scale_chunk = pl.reshape(shared_w3_scale[n0 : n0 + SH_INTER_TILE], [1, SH_INTER_TILE])
            sh_gate = pl.cast(sh_gate_acc, target_type=pl.FP32, mode="none")
            sh_up = pl.cast(sh_up_acc, target_type=pl.FP32, mode="none")
            sh_gate = pl.col_expand_mul(pl.row_expand_mul(sh_gate, x_local_scale_dq_tile), sw1_scale_chunk)
            sh_up = pl.col_expand_mul(pl.row_expand_mul(sh_up, x_local_scale_dq_tile), sw3_scale_chunk)
            sh_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sh_gate)), 1.0))
            sh_silu = pl.mul(sh_gate, sh_sigmoid)
            sh_gated = pl.mul(sh_silu, sh_up)
            sh_tile_fp32[ts0 : ts0 + T_TILE, n0 : n0 + SH_INTER_TILE] = sh_gated

    for q_tb in pl.spmd(T // T_TILE, name_hint="sh_h_q"):
        ts0 = q_tb * T_TILE
        shq_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, MOE_INTER, QUANT_TILE):
            shq_a_f32 = sh_tile_fp32[ts0 : ts0 + T_TILE, k0 : k0 + QUANT_TILE]
            shq_a_abs = pl.maximum(shq_a_f32, pl.neg(shq_a_f32))
            shq_a_max = pl.reshape(pl.row_max(shq_a_abs), [1, T_TILE])
            shq_amax = pl.maximum(shq_amax, shq_a_max)
        shq_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), shq_amax)
        sh_tile_scale_dq[ts0 : ts0 + T_TILE, 0:1] = pl.reshape(pl.recip(shq_sq_row), [T_TILE, 1])
        shq_sq_col = pl.reshape(shq_sq_row, [T_TILE, 1])
        for k1 in pl.range(0, MOE_INTER, QUANT_TILE):
            shq_q_f32 = sh_tile_fp32[ts0 : ts0 + T_TILE, k1 : k1 + QUANT_TILE]
            shq_q_scaled = pl.row_expand_mul(shq_q_f32, shq_sq_col)
            shq_q_i32 = pl.cast(shq_q_scaled, target_type=pl.INT32, mode="rint")
            shq_q_half = pl.cast(shq_q_i32, target_type=pl.FP16, mode="round")
            sh_tile_i8[ts0 : ts0 + T_TILE, k1 : k1 + QUANT_TILE] = pl.cast(shq_q_half, target_type=pl.INT8, mode="trunc")

    for w2_block in pl.spmd((T // T_TILE) * (D // (16 * SH_D_OUT_TILE)), name_hint="sh_w2"):
        w2_tb = w2_block // (D // (16 * SH_D_OUT_TILE))
        w2_db = w2_block - w2_tb * (D // (16 * SH_D_OUT_TILE))
        ts0 = w2_tb * T_TILE
        d_base = w2_db * (16 * SH_D_OUT_TILE)
        sh_tile_scale_dq_tile = sh_tile_scale_dq[ts0 : ts0 + T_TILE, 0:1]
        for dg in pl.range(16):
            d0 = d_base + dg * SH_D_OUT_TILE
            hs_init = sh_tile_i8[ts0 : ts0 + T_TILE, 0 : INTER_K]
            sw2_init = shared_w2[d0 : d0 + SH_D_OUT_TILE, 0 : INTER_K]
            sh_y_acc = pl.matmul(hs_init, sw2_init, b_trans=True, out_dtype=pl.INT32)
            for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                hs_k = sh_tile_i8[ts0 : ts0 + T_TILE, k0 : k0 + INTER_K]
                sw2_k = shared_w2[d0 : d0 + SH_D_OUT_TILE, k0 : k0 + INTER_K]
                sh_y_acc = pl.matmul_acc(sh_y_acc, hs_k, sw2_k, b_trans=True)

            sw2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + SH_D_OUT_TILE], [1, SH_D_OUT_TILE])
            sh_y = pl.cast(sh_y_acc, target_type=pl.FP32, mode="none")
            sh_y = pl.col_expand_mul(pl.row_expand_mul(sh_y, sh_tile_scale_dq_tile), sw2_scale_chunk)
            sh[ts0 : ts0 + T_TILE, d0 : d0 + SH_D_OUT_TILE] = pl.cast(sh_y, target_type=pl.BF16, mode="rint")


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
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis. Returns (i8_tensor, dequant_scale).

    For w shaped [..., N, K] (b_trans=True layout), the per-channel scale has shape [..., N].
    """
    import torch
    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    x_local_i8 = tensors["x_local_i8"]                       # [T, D] int8
    x_local_scale_dq = tensors["x_local_scale_dq"].float()   # [T, 1]
    x_local = x_local_i8.float() * x_local_scale_dq
    sw1 = dequant_w(tensors["shared_w1"], tensors["shared_w1_scale"].float())
    sw3 = dequant_w(tensors["shared_w3"], tensors["shared_w3_scale"].float())
    sw2 = dequant_w(tensors["shared_w2"], tensors["shared_w2_scale"].float())

    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_i8, sh_h_sd = _int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    # Pre-quantize x_local once so the i8 / scale specs see consistent values
    # (mirrors what gate produces in the full pipeline).
    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_local_i8_pre, x_local_sd_pre = _int8_quant_per_row(x_local_bf16)

    sw1_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw3_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw2_bf16 = (torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)

    sw1_i8, sw1_s = _quant_w_per_channel(sw1_bf16)
    sw3_i8, sw3_s = _quant_w_per_channel(sw3_bf16)
    sw2_i8, sw2_s = _quant_w_per_channel(sw2_bf16)

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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
