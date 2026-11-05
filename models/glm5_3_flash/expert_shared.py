# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash shared W8A8 SwiGLU expert and FFN branch addition."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.glm5_3_flash.all_reduce import tp_all_reduce
from models.glm5_3_flash.config import EP_SIZE, FLASH, TP_SIZE, D
from models.glm5_3_flash.elementwise import accurate_divide, accurate_sigmoid
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, expert, quantize_per_token_int8
from models.glm5_3_flash.utils import pack_int8_nz, unpack_int8_nz

T_DYN = pl.dynamic('GLM53_T_DYN')

_shared_parser = argparse.ArgumentParser(add_help=False)
_shared_parser.add_argument("--shared-tp", type=int, choices=(1, 2, 4, 8, 16), default=TP_SIZE)
SHARED_TP = _shared_parser.parse_known_args()[0].shared_tp
MOE_INTER = FLASH.moe_intermediate_size // SHARED_TP
SWIGLU_LIMIT = FLASH.swiglu_limit
SH_M_TILE = 16  # cube M-tile: matmul rows must be a multiple of 16 (fractal)
SH_ROW_PAD = 8
SH_ROW_TILE = 2
K_TILE = 512  # gate/up K tile over D = 4096
MM_INTER_TILE = 128  # gate/up N tile; 2 * MOE_INTER must be a multiple
ACT_INTER_TILE = min(256, MOE_INTER)
QUANT_TILE = min(256, MOE_INTER)
INTER_K_TILE = 128  # w2 K tile over the complete intermediate axis
D_OUT_TILE = 256  # w2 N tile over D
D_ACT_TILE = 512
W2_GROUP_TILE = 8
T_DYN_TEST = 64


@pl.jit.inline
def expert_shared(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * MOE_INTER, D], pl.INT8, pl.NZ],
    w_gate_up_scale: pl.Tensor[[2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Compute one TP shard; the caller combines the partial expert outputs."""
    token_rows = pl.tensor.dim(x_int8, 0)
    for mt in pl.parallel((token_rows + SH_M_TILE - 1) // SH_M_TILE):
        ts0 = mt * SH_M_TILE
        valid_rows = pl.min(pl.cast(SH_M_TILE, pl.INDEX), token_rows - ts0)

        # Fused gate/up cube matmul: w_gate_up rows [0, 2*MOE_INTER) contract the same
        # INT8 x rows, so one K-loop fills both halves of the INT32 accumulator.
        gate_up_i32 = pl.create_tensor([SH_M_TILE, 2 * MOE_INTER], dtype=pl.INT32)
        for nb in pl.spmd(2 * MOE_INTER // MM_INTER_TILE, name_hint="sh_gate_up_mm", allow_early_resolve=True):
            n0 = nb * MM_INTER_TILE
            acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_int8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[valid_rows, K_TILE])
                w_k = w_gate_up[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                acc = pl.matmul_acc(acc, xs_k, w_k, b_trans=True, init_cond=(k0 == 0))
            gate_up_i32[:, n0 : n0 + MM_INTER_TILE] = acc

        # Activation epilogue: dequant (per-token x scale x per-channel weight
        # scale), clamped SwiGLU, per-row amax, and INT8 requant for w2. Each
        # AIV block owns two rows across the full intermediate axis, so
        # this stage can start alongside the EP dispatch.
        # The activation/quant walk covers SH_ROW_TILE rows per block. A
        # trailing odd row runs in one extra block whose second row reads past
        # ``valid_rows`` and is never written back: the w2 epilogue masks every
        # row at or beyond ``valid_rows``.
        h_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        h_i8 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([SH_M_TILE, SH_ROW_PAD], dtype=pl.FP32, manual_dep=True)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_h_i8_init"):
            h_i8_value_full = pl.full([SH_M_TILE, MOE_INTER], dtype=pl.FP16, value=0.0)
            h_i8[:, :] = pl.cast(h_i8_value_full, target_type=pl.INT8, mode="trunc")
        n_act_blocks = (valid_rows + SH_ROW_TILE - 1) // SH_ROW_TILE
        for row_block in pl.spmd(n_act_blocks, name_hint="sh_gate_up_act_q"):
            row0 = row_block * SH_ROW_TILE
            blk_rows = pl.min(pl.cast(SH_ROW_TILE, pl.INDEX), valid_rows - row0)
            x_scale_blk = pl.slice(x_scale, [SH_ROW_PAD, 1], [ts0 + row0, 0], valid_shape=[blk_rows, 1])
            row_amax = pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=0.0)
            for part in pl.pipeline(MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_chunk_i32 = pl.slice(gate_up_i32, [SH_ROW_PAD, ACT_INTER_TILE], [row0, n0], valid_shape=[SH_ROW_TILE, ACT_INTER_TILE])
                up_chunk_i32 = pl.slice(gate_up_i32, [SH_ROW_PAD, ACT_INTER_TILE], [row0, MOE_INTER + n0], valid_shape=[SH_ROW_TILE, ACT_INTER_TILE])
                gate_w_scale = pl.reshape(w_gate_up_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                up_w_scale = pl.reshape(w_gate_up_scale[MOE_INTER + n0:MOE_INTER + n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_w_bf16 = pl.cast(gate_w_scale, pl.BF16, mode="rint")
                up_w_bf16 = pl.cast(up_w_scale, pl.BF16, mode="rint")
                gate_w_scale = pl.cast(gate_w_bf16, pl.FP32)
                up_w_scale = pl.cast(up_w_bf16, pl.FP32)
                gate_fp32 = pl.cast(gate_chunk_i32, target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(up_chunk_i32, target_type=pl.FP32, mode="none")
                gate_fp32 = pl.col_expand_mul(pl.row_expand_mul(gate_fp32, x_scale_blk), gate_w_scale)
                up_fp32 = pl.col_expand_mul(pl.row_expand_mul(up_fp32, x_scale_blk), up_w_scale)
                gate_bf16 = pl.cast(gate_fp32, pl.BF16, mode="rint")
                up_bf16 = pl.cast(up_fp32, pl.BF16, mode="rint")
                gate_fp32 = pl.cast(gate_bf16, pl.FP32)
                up_fp32 = pl.cast(up_bf16, pl.FP32)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                gate_buffer = pl.create_tensor([SH_ROW_PAD, ACT_INTER_TILE], dtype=pl.FP32)
                gate_buffer[:, :] = gate_fp32
                sigmoid = accurate_sigmoid(gate_buffer, SH_ROW_PAD, ACT_INTER_TILE)
                gated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                gated_bf16 = pl.cast(gated, pl.BF16, mode="rint")
                gated = pl.cast(gated_bf16, pl.FP32)
                # A 1-row trailing block reads one scale row; re-widen its dummy
                # second row so the store keeps its static slice. That row is
                # masked at the w2 output.
                gated = pl.set_validshape(gated, SH_ROW_TILE, ACT_INTER_TILE)
                chunk_amax = pl.reshape(pl.row_max(pl.abs(gated)), [1, SH_ROW_PAD])
                row_amax = pl.maximum(row_amax, chunk_amax)
                h_fp32[row0:row0 + SH_ROW_TILE, n0:n0 + ACT_INTER_TILE] = gated[0:SH_ROW_TILE, :]

            amax_buffer = pl.create_tensor([1, SH_ROW_PAD], dtype=pl.FP32)
            zero_rows = pl.cmp(row_amax, 0.0, cmp_type=0)
            amax_buffer[:, :] = pl.add(row_amax, zero_rows)
            row_scale_q = pl.create_tensor([1, SH_ROW_PAD], dtype=pl.FP32)
            one = pl.create_tensor([1, SH_ROW_PAD], dtype=pl.FP32)
            one[:, :] = pl.full([1, SH_ROW_PAD], dtype=pl.FP32, value=1.0)
            inverse_maximum = accurate_divide(one, amax_buffer)
            row_scale_q[:, :] = pl.mul(inverse_maximum, INT8_SCALE_MAX)
            scale_dq_buffer = pl.create_tensor([1, SH_ROW_PAD], dtype=pl.FP32)
            scale_dq_buffer[:, :] = pl.mul(row_amax, 1.0 / INT8_SCALE_MAX)
            row_scale_q_col = pl.reshape(row_scale_q, [SH_ROW_PAD, 1])
            row_scale_dq_col = pl.reshape(scale_dq_buffer, [SH_ROW_PAD, 1])
            row_scale_dq = pl.row_expand(pl.full([SH_ROW_PAD, SH_ROW_PAD], dtype=pl.FP32, value=0.0), row_scale_dq_col)
            h_scale_dq[row0 : row0 + SH_ROW_TILE, :] = row_scale_dq[0:SH_ROW_TILE, :]
            for q_idx in pl.pipeline(0, MOE_INTER // QUANT_TILE, stage=1):
                k0 = q_idx * QUANT_TILE
                h_fp32_slice = pl.slice(h_fp32, [SH_ROW_PAD, QUANT_TILE], [row0, k0], valid_shape=[SH_ROW_TILE, QUANT_TILE])
                h_scaled = pl.row_expand_mul(h_fp32_slice, row_scale_q_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_i8[row0 : row0 + SH_ROW_TILE, k0 : k0 + QUANT_TILE] = pl.cast(
                    h_fp16, target_type=pl.INT8, mode="trunc"
                )[0:SH_ROW_TILE, :]

        # w2 (down) cube matmul: [rows, MOE_INTER] x [D, MOE_INTER] -> INT32.
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K_TILE, stage=2):
                hs_k = h_i8[:, k0 : k0 + INTER_K_TILE]
                sw2_k = w_down[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K_TILE]
                y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> FP32.
        for db_idx in pl.spmd(D // (W2_GROUP_TILE * D_ACT_TILE), name_hint="sh_w2_act", allow_early_resolve=True):
            d_base = db_idx * (W2_GROUP_TILE * D_ACT_TILE)
            h_scale = pl.row_max(h_scale_dq[:, :])
            for dg in pl.pipeline(W2_GROUP_TILE, stage=2):
                d0 = d_base + dg * D_ACT_TILE
                y_2d_i32 = y_i32[:, d0 : d0 + D_ACT_TILE]
                w2_scale_chunk = pl.reshape(w_down_scale[d0 : d0 + D_ACT_TILE], [1, D_ACT_TILE])
                w2_scale_bf16 = pl.cast(w2_scale_chunk, pl.BF16, mode="rint")
                w2_scale_chunk = pl.cast(w2_scale_bf16, pl.FP32)
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale), w2_scale_chunk)
                y_bf16 = pl.cast(y_2d, pl.BF16, mode="rint")
                y_2d = pl.cast(y_bf16, pl.FP32)
                y_valid = pl.set_validshape(y_2d, valid_rows, D_ACT_TILE)
                output[ts0:ts0 + SH_M_TILE, d0:d0 + D_ACT_TILE] = y_valid

    return output


@pl.jit
def expert_shared_test(
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[2 * MOE_INTER, D], pl.INT8, pl.NZ],
    w_gate_up_scale: pl.Tensor[[2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    x_int8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)

    expert_shared(x_int8, x_scale, w_gate_up, w_gate_up_scale, w_down, w_down_scale, output)
    return output


@pl.jit.inline(auto_scope=False)
def moe_tp_combine(
    routed: pl.Tensor[[T_DYN, D], pl.FP32],
    shared: pl.Tensor[[T_DYN, D], pl.FP32],
    exchange: pld.DistributedTensor[[T_DYN, D], pl.FP32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Combine owner-masked routed rows with shared TP partials and reduce."""
    rows = pl.tensor.dim(output, 0)
    partial = pl.create_tensor([rows, D], dtype=pl.FP32)
    with pl.spmd(rows, name_hint="moe_shared_tp_partial"):
        row = pl.tile.get_block_idx()
        value = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
        if row < num_tokens:
            shared_row = pl.tile.load(shared, [row, 0], [1, D])
            routed_row = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            if row % TP_SIZE == rank:
                routed_row = pl.tile.load(routed, [row, 0], [1, D])
            routed_bf16 = pl.cast(routed_row, pl.BF16, mode="rint")
            combined = pl.add(pl.cast(routed_bf16, pl.FP32), shared_row)
            combined_bf16 = pl.cast(combined, pl.BF16, mode="rint")
            value = pl.cast(combined_bf16, pl.FP32)
        pl.tile.store(value, [row, 0], partial)
    reduced = pl.create_tensor([rows, D], dtype=pl.BF16)
    tp_all_reduce(partial, exchange, arrived, reduced, num_tokens, rank, 0, epoch)
    with pl.spmd(rows, name_hint="moe_shared_tp_output"):
        row = pl.tile.get_block_idx()
        value = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
        if row < num_tokens:
            value = pl.cast(pl.tile.load(reduced, [row, 0], [1, D]), pl.FP32)
        pl.tile.store(value, [row, 0], output)
    return output


def golden_moe_tp_combine(tensors, routed, x_int8, x_scale, num_tokens):
    """Reference the BF16 partial combine and shared TP reduction."""
    ranks, rows, width = routed.shape
    if ranks != TP_SIZE or ranks != EP_SIZE:
        raise ValueError("standalone MoE requires one TP/EP group with DP1")
    hidden = tensors["x_mixed"]
    if not torch.equal(hidden, hidden[:1].expand_as(hidden)):
        raise ValueError("standalone TP MoE inputs must be replicated across ranks")
    partials = torch.zeros_like(routed)
    for rank in range(ranks):
        work = {
            "x_int8": x_int8[rank], "x_scale": x_scale[rank],
            "w_gate_up": tensors["shared_w_gate_up"][rank],
            "w_gate_up_scale": tensors["shared_w_gate_up_scale"][rank],
            "w_down": tensors["shared_w_down"][rank],
            "w_down_scale": tensors["shared_w_down_scale"][rank],
            "output": torch.zeros(rows, width, dtype=torch.float32),
        }
        golden_expert_shared_fn(work)
        partial = work["output"]
        owned = torch.arange(rows) % TP_SIZE == rank
        partial[owned] += routed[rank, owned].bfloat16().float()
        partials[rank, :num_tokens] = partial[:num_tokens].bfloat16().float()
    reduced = partials.double().sum(dim=0).float().bfloat16().float()
    return reduced.unsqueeze(0).expand_as(routed).contiguous()


def golden_expert_shared(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    return expert(x, w_gate, w_up, w_down)


def golden_expert_shared_fn(tensors):
    """Evaluate a shared-expert shard with BF16 linear and activation boundaries."""

    x_int8 = tensors["x_int8"].to(torch.int32)
    x_scale = tensors["x_scale"].float()
    gate_up_i8 = unpack_int8_nz(tensors["w_gate_up"]).to(torch.int32)
    gate_up_scale = tensors["w_gate_up_scale"].bfloat16().float()
    w_down_i8 = unpack_int8_nz(tensors["w_down"]).to(torch.int32)
    w_down_scale = tensors["w_down_scale"].bfloat16().float()

    gate_i = x_int8 @ gate_up_i8[:MOE_INTER].T
    up_i = x_int8 @ gate_up_i8[MOE_INTER:].T
    gate_v = gate_i.float() * x_scale * gate_up_scale[:MOE_INTER].view(1, -1)
    up_v = up_i.float() * x_scale * gate_up_scale[MOE_INTER:].view(1, -1)
    gate_v = gate_v.bfloat16().float()
    up_v = up_v.bfloat16().float()
    if SWIGLU_LIMIT > 0:
        gate_v = gate_v.clamp(max=SWIGLU_LIMIT)
        up_v = up_v.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sigmoid = (1.0 / (torch.exp(-gate_v.double()).float() + 1.0).double()).float()
    h = ((gate_v * sigmoid) * up_v).bfloat16()
    h_i8, h_sd = quantize_per_token_int8(h)
    out_i = h_i8.to(torch.int32) @ w_down_i8.T
    out = out_i.float() * h_sd * w_down_scale.view(1, -1)

    tensors["output"][:] = out.bfloat16().float()


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize an INT8 weight + per-channel FP32 scale with a realistic
    per-channel magnitude spread (see the DeepSeek-V4 sibling for the grid)."""
    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
    amax = W.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(W / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import TensorSpec

    x_bf16 = torch.randn(num_tokens, D, dtype=torch.bfloat16)
    if num_tokens > 1:
        x_bf16[0] = 0

    def init_x_int8():

        x_i8, x_sd = quantize_per_token_int8(x_bf16)
        return x_i8

    def init_x_scale():

        _, x_sd = quantize_per_token_int8(x_bf16)
        return x_sd

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw_gu_i8, sw_gu_s = gen_shared_weight((2 * MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)

    return [
        TensorSpec("x_int8", [num_tokens, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [num_tokens, 1], torch.float32, init_value=init_x_scale),
        TensorSpec("w_gate_up", [2 * MOE_INTER, D], torch.int8, init_value=lambda: pack_int8_nz(sw_gu_i8)),
        TensorSpec("w_gate_up_scale", [2 * MOE_INTER], torch.float32, init_value=lambda: sw_gu_s),
        TensorSpec("w_down", [D, MOE_INTER], torch.int8, init_value=lambda: pack_int8_nz(sw2_i8)),
        TensorSpec("w_down_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("output", [num_tokens, D], torch.float32),
    ]


def main():

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T_DYN_TEST)
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shared-tp", type=int, choices=(1, 2, 4, 8, 16), default=SHARED_TP)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.num_tokens < 1:
        parser.error("--num-tokens must be positive")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    result = run(
        fn=expert_shared_test,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_expert_shared_fn,
        config=dict(dump_passes=args.dump_passes, backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane),
        compile_only=args.compile_only,
        rtol=0.001,
        atol=0.001,
        compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
