# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single shared expert, evaluated for every token on every sparse layer.

Same clamped SwiGLU as a routed expert at ``moe_intermediate_size = 2048``
(``n_shared_experts = 1``), but with no routing weight and no dispatch: it runs on
the rank's own token rows while the routed half is still in flight, which is what
hides part of the all-to-all latency.

The shared expert's weights are TP-sharded on the intermediate axis, so each rank
computes ``MOE_INTER / TP_SIZE`` channels and the result joins the routed output in
the same reduction.

Unlike the DeepSeek-V4 sibling (separate ``w1``/``w3`` tensors), the GLM-5.3-Flash
W8A8 checkpoint stores one fused ``w_gate_up [2 * LOCAL_MOE_INTER, D]``: the gate
rows occupy ``[0, LOCAL_MOE_INTER)`` and the up rows ``[LOCAL_MOE_INTER, 2 * LOCAL)``,
with a single per-channel scale vector in the same order. One cube matmul produces
both halves, and the SwiGLU epilogue slices them apart. The output is FP32: the
shared expert joins the routed reduction in FP32 before the residual.
"""

import sys
from pathlib import Path

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FLASH, T_DYN, TP_SIZE
from models.glm5_3_flash.golden import expert
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


LOCAL_MOE_INTER = FLASH.moe_intermediate_size // TP_SIZE

# tiling
SH_M_TILE = 16           # cube M-tile: matmul rows must be a multiple of 16 (fractal)
SH_ROW_PAD = 8
SH_ROWS_PER_BLOCK = 2
K_TILE = 512             # gate/up K tile over D = 4096
MM_INTER_TILE = 128      # gate/up N tile; 2 * LOCAL_MOE_INTER must be a multiple
ACT_INTER_TILE = min(256, LOCAL_MOE_INTER)
QUANT_TILE = min(256, LOCAL_MOE_INTER)
INTER_K = min(128, LOCAL_MOE_INTER)  # w2 K tile (= local intermediate)
D_OUT_TILE = 256         # w2 N tile over D
D_OUT_TILE_ACT = 512
W2_ACT_INNER = 8
SWIGLU_LIMIT = FLASH.swiglu_limit

assert (2 * LOCAL_MOE_INTER) % MM_INTER_TILE == 0
assert LOCAL_MOE_INTER % ACT_INTER_TILE == 0
assert LOCAL_MOE_INTER % QUANT_TILE == 0
assert LOCAL_MOE_INTER % INTER_K == 0
assert D % D_OUT_TILE == 0


def golden_expert_shared(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return expert(x, w_gate, w_up, w_down)


@pl.jit.inline
def expert_shared(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    token_rows = pl.tensor.dim(x_int8, 0)
    for mt in pl.parallel((token_rows + SH_M_TILE - 1) // SH_M_TILE):
        ts0 = mt * SH_M_TILE
        valid_rows = pl.min(pl.cast(SH_M_TILE, pl.INDEX), token_rows - ts0)

        # Fused gate/up cube matmul: w_gate_up rows [0, 2*LOCAL) contract the same
        # INT8 x rows, so one K-loop fills both halves of the INT32 accumulator.
        gate_up_i32 = pl.create_tensor([SH_M_TILE, 2 * LOCAL_MOE_INTER], dtype=pl.INT32)
        for nb in pl.spmd((2 * LOCAL_MOE_INTER) // MM_INTER_TILE, name_hint="sh_gate_up_mm", allow_early_resolve=True):
            n0 = nb * MM_INTER_TILE
            acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_int8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[valid_rows, K_TILE])
                w_k = w_gate_up[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, xs_k, w_k, b_trans=True, init_cond=(k0 == 0))
            gate_up_i32[:, n0 : n0 + MM_INTER_TILE] = acc

        # Activation epilogue: dequant (per-token x scale x per-channel weight
        # scale), clamped SwiGLU, per-row amax, and INT8 requant for w2. Each
        # AIV block owns two rows across the full local intermediate axis, so
        # this stage can start alongside the EP dispatch.
        # The activation/quant walk covers SH_ROWS_PER_BLOCK rows per block. A
        # trailing odd row runs in one extra block whose second row reads past
        # ``valid_rows`` and is never written back: the w2 epilogue masks every
        # row at or beyond ``valid_rows``.
        h_fp32 = pl.create_tensor([SH_M_TILE, LOCAL_MOE_INTER], dtype=pl.FP32)
        h_i8 = pl.create_tensor([SH_M_TILE, LOCAL_MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([SH_M_TILE, SH_ROW_PAD], dtype=pl.FP32, manual_dep=True)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_h_i8_init"):
            h_i8[:, :] = pl.cast(
                pl.full([SH_M_TILE, LOCAL_MOE_INTER], dtype=pl.FP16, value=0.0),
                target_type=pl.INT8,
                mode="trunc",
            )
        n_act_blocks = (valid_rows + SH_ROWS_PER_BLOCK - 1) // SH_ROWS_PER_BLOCK
        for row_block in pl.spmd(n_act_blocks, name_hint="sh_gate_up_act_q"):
            row0 = row_block * SH_ROWS_PER_BLOCK
            blk_rows = pl.min(pl.cast(SH_ROWS_PER_BLOCK, pl.INDEX), valid_rows - row0)
            x_scale_blk = pl.slice(x_scale, [SH_ROW_PAD, 1], [ts0 + row0, 0], valid_shape=[blk_rows, 1])
            row_amax = pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for part in pl.pipeline(LOCAL_MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_chunk_i32 = pl.slice(
                    gate_up_i32, [SH_ROW_PAD, ACT_INTER_TILE], [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                up_chunk_i32 = pl.slice(
                    gate_up_i32, [SH_ROW_PAD, ACT_INTER_TILE], [row0, LOCAL_MOE_INTER + n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                gate_w_scale = pl.reshape(
                    w_gate_up_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE]
                )
                up_w_scale = pl.reshape(
                    w_gate_up_scale[LOCAL_MOE_INTER + n0 : LOCAL_MOE_INTER + n0 + ACT_INTER_TILE],
                    [1, ACT_INTER_TILE],
                )
                gate_fp32 = pl.cast(gate_chunk_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_chunk_i32, target_type=pl.FP32, mode="none")
                gate_fp32 = pl.col_expand_mul(pl.row_expand_mul(gate_fp32, x_scale_blk), gate_w_scale)
                up_fp32 = pl.col_expand_mul(pl.row_expand_mul(up_fp32, x_scale_blk), up_w_scale)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                gated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                # A 1-row trailing block reads one scale row; re-widen its dummy
                # second row so the store keeps its static slice. That row is
                # masked at the w2 output.
                gated = pl.set_validshape(gated, SH_ROWS_PER_BLOCK, ACT_INTER_TILE)
                chunk_amax = pl.reshape(pl.row_max(pl.abs(gated)), [1, SH_ROW_PAD])
                row_amax = pl.maximum(row_amax, chunk_amax)
                h_fp32[row0 : row0 + SH_ROWS_PER_BLOCK, n0 : n0 + ACT_INTER_TILE] = gated[0:SH_ROWS_PER_BLOCK, :]

            row_scale_q = pl.div(pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
            row_scale_q_col = pl.reshape(row_scale_q, [SH_ROW_PAD, 1])
            row_scale_dq_col = pl.reshape(pl.recip(row_scale_q), [SH_ROW_PAD, 1])
            row_scale_dq = pl.row_expand(pl.full([SH_ROW_PAD, SH_ROW_PAD], dtype=pl.FP32, value=0.0), row_scale_dq_col)
            h_scale_dq[row0 : row0 + SH_ROWS_PER_BLOCK, :] = row_scale_dq[0:SH_ROWS_PER_BLOCK, :]
            for q_idx in pl.pipeline(0, LOCAL_MOE_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32_slice = pl.slice(
                    h_fp32, [SH_ROW_PAD, QUANT_TILE], [row0, k0],
                    valid_shape=[SH_ROWS_PER_BLOCK, QUANT_TILE],
                )
                h_scaled = pl.row_expand_mul(h_fp32_slice, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_i8[row0 : row0 + SH_ROWS_PER_BLOCK, k0 : k0 + QUANT_TILE] = pl.cast(
                    h_fp16, target_type=pl.INT8, mode="trunc"
                )[0:SH_ROWS_PER_BLOCK, :]

        # w2 (down) cube matmul: [rows, LOCAL] INT8 x [D, LOCAL] INT8 -> INT32.
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, LOCAL_MOE_INTER, INTER_K, stage=2):
                hs_k = h_i8[:, k0 : k0 + INTER_K]
                sw2_k = w_down[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> FP32.
        for db_idx in pl.spmd(D // (W2_ACT_INNER * D_OUT_TILE_ACT), name_hint="sh_w2_act", allow_early_resolve=True):
            d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
            h_scale = pl.row_max(h_scale_dq[:, :])
            for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + dg * D_OUT_TILE_ACT
                y_2d_i32 = y_i32[:, d0 : d0 + D_OUT_TILE_ACT]
                w2_scale_chunk = pl.reshape(w_down_scale[d0 : d0 + D_OUT_TILE_ACT], [1, D_OUT_TILE_ACT])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale), w2_scale_chunk)
                y_valid = pl.set_validshape(y_2d, valid_rows, D_OUT_TILE_ACT)
                output = pl.assemble(output, y_valid, [ts0, d0])

    return output


@pl.jit
def expert_shared_test(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * LOCAL_MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    x_int8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)

    expert_shared(
        x_int8, x_scale,
        w_gate_up, w_gate_up_scale,
        w_down, w_down_scale,
        output,
    )
    return output


def golden_expert_shared_fn(tensors):
    """Torch reference for the shared expert over the fused gate/up layout."""
    from models.glm5_3_flash.quantization import quantize_per_token_int8

    x_int8 = tensors["x_int8"].to(torch.int32)
    x_scale = tensors["x_scale"].float()
    gate_up_i8 = tensors["w_gate_up"].to(torch.int32)
    gate_up_scale = tensors["w_gate_up_scale"].float()
    w_down_i8 = tensors["w_down"].to(torch.int32)
    w_down_scale = tensors["w_down_scale"].float()

    local = LOCAL_MOE_INTER
    gate_i = x_int8 @ gate_up_i8[:local].T
    up_i = x_int8 @ gate_up_i8[local:].T
    gate_v = gate_i.float() * x_scale * gate_up_scale[:local].view(1, -1)
    up_v = up_i.float() * x_scale * gate_up_scale[local:].view(1, -1)
    if SWIGLU_LIMIT > 0:
        gate_v = gate_v.clamp(max=SWIGLU_LIMIT)
        up_v = up_v.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sigmoid = torch.reciprocal(torch.exp(-gate_v) + 1.0)
    h = (gate_v * sigmoid) * up_v
    h_i8, h_sd = quantize_per_token_int8(h)
    out_i = h_i8.to(torch.int32) @ w_down_i8.T
    out = out_i.float() * h_sd * w_down_scale.view(1, -1)

    tensors["output"][:] = out.to(torch.float32)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize an INT8 weight + per-channel FP32 scale with a realistic
    per-channel magnitude spread (see the DeepSeek-V4 sibling for the grid)."""
    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
    amax = W.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(W / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


T_DYN_TEST = 64


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import TensorSpec

    x_bf16 = torch.randn(num_tokens, D, dtype=torch.bfloat16)

    def init_x_int8():
        from models.glm5_3_flash.quantization import quantize_per_token_int8
        x_i8, x_sd = quantize_per_token_int8(x_bf16)
        return x_i8

    def init_x_scale():
        from models.glm5_3_flash.quantization import quantize_per_token_int8
        _, x_sd = quantize_per_token_int8(x_bf16)
        return x_sd

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw_gu_i8, sw_gu_s = gen_shared_weight(
        (2 * LOCAL_MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50
    )
    sw2_i8, sw2_s = gen_shared_weight(
        (D, LOCAL_MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33
    )

    return [
        TensorSpec("x_int8", [num_tokens, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [num_tokens, 1], torch.float32, init_value=init_x_scale),
        TensorSpec("w_gate_up", [2 * LOCAL_MOE_INTER, D], torch.int8, init_value=lambda: sw_gu_i8),
        TensorSpec("w_gate_up_scale", [2 * LOCAL_MOE_INTER], torch.float32, init_value=lambda: sw_gu_s),
        TensorSpec("w_down", [D, LOCAL_MOE_INTER], torch.int8, init_value=lambda: sw2_i8),
        TensorSpec("w_down_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("output", [num_tokens, D], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T_DYN_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=expert_shared_test,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_expert_shared_fn,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Deterministic INT8 golden: 3e-3 per point, 1% of points.
            "output": ratio_reldiff(diff_thd=3e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "LOCAL_MOE_INTER",
    "expert_shared",
    "golden_expert_shared",
]
