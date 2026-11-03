# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash routed W8A8 SwiGLU experts."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import (
    EP_SIZE, FLASH, MOE_INTER, N_EXPERTS, N_LOCAL_EXPERTS, D, Glm53FlashConfig,
)
from models.glm5_3_flash.dense_mlp import DenseTensors, dense_mlp_forward, swiglu_limited
from models.glm5_3_flash.elementwise import accurate_divide, accurate_sigmoid
from models.glm5_3_flash.gate import RouterTensors, router_forward
from models.glm5_3_flash.mhc import hc_mixes, hc_pre, hc_seed
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX, expert, quantize_per_token_int8
from models.glm5_3_flash.utils import (
    _FF_DOWN, _FF_FUSED, _FF_GATE, _FF_UP, TEXT_LAYER_ROOTS, Checkpoint, _resolve_name, pack_int8_nz,
    read_reference_float,
)

RECV_DYN = pl.dynamic('GLM53_RECV_DYN')

SWIGLU_LIMIT = FLASH.swiglu_limit
RECV_TILE = 16  # expert row tile: matmul rows must be a multiple of 16
K_TILE = 512  # gate/up K tile over D = 4096
MM_INTER_TILE = 256  # gate/up N tile; 2 * MOE_INTER must be a multiple
ACT_INTER_TILE = 128  # Bounds the compensated sigmoid working set
QUANT_TILE = 512  # per-row requant tile
INTER_K_TILE = 512  # w2 K tile (= moe_intermediate_size)
D_OUT_TILE = 256  # w2 N tile over D
D_ACT_TILE = 512
W2_GROUP_TILE = 8
T_DYN_TEST = 64


@pl.jit.inline
def expert_routed(
    x_int8: pl.Tensor[[RECV_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    route_weights: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    output: pl.Tensor[[RECV_DYN, D], pl.FP32],
):
    # One expert per outer block; inside it, tile the expert's rows (bounded by
    # expert_offsets[e+1] - expert_offsets[e], dynamic per expert) in RECV_TILE
    # chunks. The flat RECV_DYN row space needs no per-expert reshape.
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        e_start = pl.cast(pl.read(expert_offsets, [local_e]), pl.INDEX)
        e_end = pl.cast(pl.read(expert_offsets, [local_e + 1]), pl.INDEX)
        n_rows = e_end - e_start
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(n_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(pl.cast(RECV_TILE, pl.INDEX), n_rows - tile_row)
            flat_row = e_start + tile_row

            # Fused gate/up cube matmul: w_gate_up rows [0, 2*MOE_INTER) contract
            # the same INT8 x rows, so one K-loop fills both halves of the INT32
            # accumulator. w_k is the per-expert [1, N, K] slice.
            gate_up_i32 = pl.create_tensor([RECV_TILE, 2 * MOE_INTER], dtype=pl.INT32)
            for nb in pl.spmd(2 * MOE_INTER // MM_INTER_TILE, name_hint="exp_gate_up_mm", allow_early_resolve=True):
                n0 = nb * MM_INTER_TILE
                acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    xs_k = pl.slice(x_int8, [RECV_TILE, K_TILE], [flat_row, k0], valid_shape=[valid_rows, K_TILE])
                    w_k = w_gate_up[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    acc = pl.matmul_acc(acc, xs_k, w_k, b_trans=True, init_cond=(k0 == 0))
                gate_up_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(acc, [RECV_TILE, MM_INTER_TILE])

            # Activation epilogue: dequant (per-token x scale x per-channel weight
            # scale), clamped SwiGLU, per-row amax, INT8 requant for w2. The
            # intermediate is NOT TP-sharded here (EP shards experts instead), so
            # the full 2048 columns walk ACT_INTER_TILE at a time.
            h_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
            h_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
            h_scale_dq = pl.create_tensor([RECV_TILE, 1], dtype=pl.FP32, manual_dep=True)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_i8_init"):
                h_i8_value_full = pl.full([RECV_TILE, MOE_INTER], dtype=pl.FP16, value=0.0)
                h_i8[:, :] = pl.cast(h_i8_value_full, target_type=pl.INT8, mode="trunc")
            for act_blk in pl.spmd(MOE_INTER // ACT_INTER_TILE, name_hint="exp_gate_up_act"):
                a0 = act_blk * ACT_INTER_TILE
                x_scale_col = pl.slice(x_scale, [RECV_TILE, 1], [flat_row, 0], valid_shape=[valid_rows, 1])
                gate_w_scale = pl.reshape(w_gate_up_scale[local_e, a0:a0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                up_w_scale = pl.reshape(w_gate_up_scale[local_e, MOE_INTER + a0:MOE_INTER + a0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_fp32 = pl.cast(gate_up_i32[:, a0:a0 + ACT_INTER_TILE], target_type=pl.FP32, mode="none")
                up_fp32 = pl.cast(gate_up_i32[:, MOE_INTER + a0:MOE_INTER + a0 + ACT_INTER_TILE], target_type=pl.FP32, mode="none")
                gate_fp32 = pl.col_expand_mul(pl.row_expand_mul(gate_fp32, x_scale_col), gate_w_scale)
                up_fp32 = pl.col_expand_mul(pl.row_expand_mul(up_fp32, x_scale_col), up_w_scale)
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                gate_buffer = pl.create_tensor([RECV_TILE, ACT_INTER_TILE], dtype=pl.FP32)
                gate_buffer[:, :] = gate_fp32
                sigmoid = accurate_sigmoid(gate_buffer, RECV_TILE, ACT_INTER_TILE)
                gated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                h_fp32[:, a0 : a0 + ACT_INTER_TILE] = gated

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_h_quant"):
                row_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                    h_chunk = h_fp32[:, k0 : k0 + QUANT_TILE]
                    chunk_amax = pl.reshape(pl.row_max(pl.abs(h_chunk)), [1, RECV_TILE])
                    row_amax = pl.maximum(row_amax, chunk_amax)
                amax_buffer = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32)
                amax_buffer[:, :] = row_amax
                numerator = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32)
                numerator[:, :] = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                row_scale_q = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32)
                row_scale_q[:, :] = accurate_divide(numerator, amax_buffer)
                one = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32)
                one[:, :] = pl.full([1, RECV_TILE], dtype=pl.FP32, value=1.0)
                row_scale_dq = pl.create_tensor([1, RECV_TILE], dtype=pl.FP32)
                row_scale_dq[:, :] = accurate_divide(one, row_scale_q)
                h_scale_dq[:, :] = pl.reshape(row_scale_dq, [RECV_TILE, 1])
                row_scale_q_col = pl.reshape(row_scale_q, [RECV_TILE, 1])
                for q_idx in pl.pipeline(0, MOE_INTER // QUANT_TILE, stage=2):
                    k0 = q_idx * QUANT_TILE
                    h_scaled = pl.row_expand_mul(h_fp32[:, k0 : k0 + QUANT_TILE], row_scale_q_col)
                    h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                    h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                    h_i8[:, k0 : k0 + QUANT_TILE] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")

            # w2 (down) cube matmul: [rows, MOE_INTER] INT8 x [D, MOE_INTER] INT8 -> INT32.
            y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
            for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="exp_w2_mm"):
                d0 = db_idx * D_OUT_TILE
                y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, INTER_K_TILE, stage=2):
                    hs_k = h_i8[:, k0 : k0 + INTER_K_TILE]
                    w2_k = w_down[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K_TILE]
                    y_acc = pl.matmul_acc(y_acc, hs_k, w2_k, b_trans=True, init_cond=(k0 == 0))
                y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

            # Dequantize w2 before applying the routing weight. Combining the
            # scales first changes FP32 rounding before the owner sums slots.
            for db_idx in pl.spmd(D // (W2_GROUP_TILE * D_ACT_TILE), name_hint="exp_w2_act", allow_early_resolve=True):
                d_base = db_idx * (W2_GROUP_TILE * D_ACT_TILE)
                w_row_blk = pl.slice(route_weights, [RECV_TILE, 1], [flat_row, 0], valid_shape=[valid_rows, 1])
                for dg in pl.pipeline(W2_GROUP_TILE, stage=2):
                    d0 = d_base + dg * D_ACT_TILE
                    y_2d_i32 = y_i32[:, d0 : d0 + D_ACT_TILE]
                    w2_scale_chunk = pl.reshape(w_down_scale[local_e, d0:d0 + D_ACT_TILE], [1, D_ACT_TILE])
                    y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                    y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_scale_dq), w2_scale_chunk)
                    y_2d = pl.row_expand_mul(y_2d, w_row_blk)
                    y_valid = pl.set_validshape(y_2d, valid_rows, D_ACT_TILE)
                    output[flat_row:flat_row + RECV_TILE, d0:d0 + D_ACT_TILE] = y_valid

    return output


@pl.jit
def expert_routed_test(
    x_int8: pl.Tensor[[RECV_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    route_weights: pl.Tensor[[RECV_DYN, 1], pl.FP32],
    output: pl.InOut[pl.Tensor[[RECV_DYN, D], pl.FP32]],
):
    x_int8.bind_dynamic(0, RECV_DYN)
    x_scale.bind_dynamic(0, RECV_DYN)
    route_weights.bind_dynamic(0, RECV_DYN)
    output.bind_dynamic(0, RECV_DYN)

    expert_routed(
        x_int8, x_scale, w_gate_up, w_gate_up_scale,
        w_down, w_down_scale, expert_offsets, route_weights, output,
    )
    return output


class ExpertProvider:
    """Routed-expert weight source: one expert's FP32 matrices at a time.

    The streaming contract that bounds host memory: the provider materializes a
    single expert per call and the caller releases it before the next. The
    synthetic provider keeps one shared base and re-scales it per expert, so
    every expert is distinct without 15 GB of random tensors.
    """

    def gate_up(self, expert: int) -> torch.Tensor:
        """``[2 * moe_inter, D]`` with gate rows first, then up rows."""
        raise NotImplementedError

    def down(self, expert: int) -> torch.Tensor:
        """``[D, moe_inter]``."""
        raise NotImplementedError


class CheckpointExpertProvider(ExpertProvider):
    """Lazy per-expert dequant reads straight from the checkpoint."""

    def __init__(self, checkpoint: Checkpoint, layer: int):
        self._checkpoint = checkpoint
        self._layer = layer
        self._names: dict[int, tuple[str, str | None, str]] = {}

    def _resolve(self, expert: int) -> tuple[str, str | None, str]:
        if expert not in self._names:
            base = f"mlp.experts.{expert}."
            # Fused spelling first: _resolve_name raises when a spelling is
            # absent, so a fused-only checkpoint would fail on gate_proj
            # before the fused branch is ever considered.
            fused = next(
                (f"{root}{self._layer}.{base}{s}" for root in TEXT_LAYER_ROOTS for s in _FF_FUSED if f"{root}{self._layer}.{base}{s}" in self._checkpoint),
                None,
            )
            if fused is None:
                gate = _resolve_name(self._checkpoint, self._layer, base, _FF_GATE)
                up = _resolve_name(self._checkpoint, self._layer, base, _FF_UP)
            else:
                gate, up = fused, None
            down = _resolve_name(self._checkpoint, self._layer, base, _FF_DOWN)
            self._names[expert] = (gate, up, down)
        return self._names[expert]

    def gate_up(self, expert: int) -> torch.Tensor:
        gate_or_fused, up, _ = self._resolve(expert)
        if up is None:
            return read_reference_float(self._checkpoint, gate_or_fused).contiguous()
        gate = read_reference_float(self._checkpoint, gate_or_fused)
        up_w = read_reference_float(self._checkpoint, up)
        return torch.cat([gate, up_w], dim=0)

    def down(self, expert: int) -> torch.Tensor:
        return read_reference_float(self._checkpoint, self._resolve(expert)[2]).contiguous()


class SyntheticExpertProvider(ExpertProvider):
    """Deterministic synthetic experts: shared base scaled per expert id."""

    def __init__(self, model: Glm53FlashConfig, seed: int):
        generator = torch.Generator().manual_seed(seed)
        inter, hidden = model.moe_intermediate_size, model.hidden_size
        self._base_gate_up = torch.randn(2 * inter, hidden, generator=generator) * 0.02
        self._base_down = torch.randn(hidden, inter, generator=generator) * 0.02
        self._gate_up_buf = torch.empty_like(self._base_gate_up)
        self._down_buf = torch.empty_like(self._base_down)

    def gate_up(self, expert: int) -> torch.Tensor:
        torch.mul(self._base_gate_up, 1.0 + expert * 1e-3, out=self._gate_up_buf)
        return self._gate_up_buf

    def down(self, expert: int) -> torch.Tensor:
        torch.mul(self._base_down, 1.0 + expert * 1e-3, out=self._down_buf)
        return self._down_buf


def moe_forward(
    x: torch.Tensor,
    router: RouterTensors,
    experts: ExpertProvider,
    shared: DenseTensors,
    model: Glm53FlashConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sparse MoE (ports of ``Glm5NextTextMoE``/``TextExperts``) with expert_id streaming.

    Only experts the router actually hit are read from the provider, one at a
    time; shared experts run on the un-routed input. Returns
    ``(output [B, S, D], internals)``.
    """
    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    topk_weights, topk_indices = router_forward(flat, router, model)

    routed = torch.zeros_like(flat)
    hit = torch.unique(topk_indices)
    for expert_id in hit.tolist():
        rows, positions = torch.where(topk_indices == expert_id)
        gate_up = experts.gate_up(expert_id)
        gate, up = (flat[rows] @ gate_up.t()).chunk(2, dim=-1)
        hidden_act = swiglu_limited(gate, up, model.swiglu_limit)
        del gate_up
        down = experts.down(expert_id)
        contribution = (hidden_act @ down.t()) * topk_weights[rows, positions][:, None]
        del down
        routed.index_add_(0, rows, contribution)

    shared_out = dense_mlp_forward(x, shared, model)
    output = routed.view(*orig_shape) + shared_out
    internals = {
        "router_topk_weights": topk_weights,
        "router_topk_indices": topk_indices,
        "moe_routed": routed.view(*orig_shape),
        "moe_shared": shared_out,
    }
    return output, internals


def golden_expert_routed(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    route_weight: torch.Tensor,
) -> torch.Tensor:
    """``route_weight`` is ``[tokens, 1]``, matching the kernel ABI."""
    return expert(x, w_gate, w_up, w_down, route_weight)


def golden_expert_routed_fn(tensors):
    """Torch reference for the routed experts over the flat/offsets layout.

    Per-expert rows ``[offsets[e], offsets[e + 1])`` are dequantized, run through
    the clamped SwiGLU with per-token x scale, requantized per row, then the w2
    output is scaled by the per-token route weight (folded, matching the kernel).
    """

    x_int8 = tensors["x_int8"].to(torch.int32)
    x_scale = tensors["x_scale"].float()
    gate_up_i8 = tensors["w_gate_up"].to(torch.int32)
    gate_up_scale = tensors["w_gate_up_scale"].float()
    w_down_i8 = tensors["w_down"].to(torch.int32)
    w_down_scale = tensors["w_down_scale"].float()
    expert_offsets = tensors["expert_offsets"].to(torch.int64)
    route_weights = tensors["route_weights"].float()

    output = torch.zeros_like(tensors["output"])
    for e in range(N_LOCAL_EXPERTS):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end <= start:
            continue
        x_i = x_int8[start:end]
        xs = x_scale[start:end]
        gate_i = x_i @ gate_up_i8[e, :MOE_INTER].T
        up_i = x_i @ gate_up_i8[e, MOE_INTER:].T
        gate_v = gate_i.float() * xs * gate_up_scale[e, :MOE_INTER].view(1, -1)
        up_v = up_i.float() * xs * gate_up_scale[e, MOE_INTER:].view(1, -1)
        if SWIGLU_LIMIT > 0:
            gate_v = gate_v.clamp(max=SWIGLU_LIMIT)
            up_v = up_v.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        sigmoid = (1.0 / (torch.exp(-gate_v.double()).float() + 1.0).double()).float()
        h = (gate_v * sigmoid) * up_v
        h_i8, h_sd = quantize_per_token_int8(h)
        out_i = h_i8.to(torch.int32) @ w_down_i8[e].T
        out = out_i.float() * h_sd * w_down_scale[e].view(1, -1)
        output[start:end] = out * route_weights[start:end]

    tensors["output"][:] = output.to(torch.float32)


def gen_routed_weight(shape, dequant_std):
    """Synthesize an INT8 weight + per-channel FP32 scale with a realistic
    per-channel magnitude spread (see the DeepSeek-V4 sibling for the grid).

    The routed weights are the bulk of the model (18 experts x 4096 x 4096 at
    EP16), so the per-output-channel amax/round pass runs in bounded chunks
    instead of materialising the whole FP32 tensor family at once.
    """
    import torch

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim
    WEIGHT_ELEMENT_TILE = 1 << 25  # elements per pass, matching the DeepSeek-V4 sibling

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)
    step = max(1, WEIGHT_ELEMENT_TILE // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0 : i0 + step]
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        s = amax / INT8_SCALE_MAX
        w_i8[i0 : i0 + step] = torch.round(w / s).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0 : i0 + step] = s
    del W
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import TensorSpec

    # Distribute the test rows evenly across local experts: base rows each with
    # the remainder spread over the first experts, offsets cumulative.
    total = num_tokens
    base = total // N_LOCAL_EXPERTS
    rem = total - base * N_LOCAL_EXPERTS
    per_expert = torch.full((N_LOCAL_EXPERTS,), base, dtype=torch.int64)
    if base > 0:
        per_expert[:rem] += 1
    else:
        per_expert[:rem] = 1
    offsets = torch.zeros(N_LOCAL_EXPERTS + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(per_expert, dim=0).to(torch.int32)

    x_bf16 = torch.randn(total, D, dtype=torch.bfloat16)

    def init_x_int8():

        x_i8, _ = quantize_per_token_int8(x_bf16)
        return x_i8

    def init_x_scale():

        _, x_sd = quantize_per_token_int8(x_bf16)
        return x_sd

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}
    w_gu_i8, w_gu_s = gen_routed_weight((N_LOCAL_EXPERTS, 2 * MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w2_i8, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("x_int8", [total, D], torch.int8, init_value=init_x_int8),
        TensorSpec("x_scale", [total, 1], torch.float32, init_value=init_x_scale),
        TensorSpec("w_gate_up", [N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8, init_value=lambda: w_gu_i8),
        TensorSpec("w_gate_up_scale", [N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32, init_value=lambda : w_gu_s),
        TensorSpec("w_down", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("w_down_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("expert_offsets", [N_LOCAL_EXPERTS + 1], torch.int32, init_value=lambda: offsets),
        TensorSpec("route_weights", [total, 1], torch.float32, init_value=lambda: torch.rand(total, 1)),
        TensorSpec("output", [total, D], torch.float32, init_value=torch.zeros),
    ]


def build_checkpoint_moe_specs(checkpoint, layer_id, *, capacity, num_tokens, num_ranks, hidden=None):
    """Stage real FFN weights and synthetic or supplied pre-norm MoE activations.

    ``hidden`` is BF16 ``[num_ranks, num_tokens, D]`` after mHC pre mixing.
    Without it, random embeddings are folded with the checkpoint's FFN mHC
    weights. This fixture does not execute the preceding Transformer layers.
    """
    from golden import ScalarSpec, TensorSpec

    if num_ranks != EP_SIZE:
        raise ValueError(f"checkpoint fixture requires the compiled EP size {EP_SIZE}, got {num_ranks}")
    if not 0 <= num_tokens <= capacity:
        raise ValueError(f"num_tokens must be between 0 and capacity {capacity}, got {num_tokens}")
    if hidden is not None:
        expected = (num_ranks, num_tokens, D)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != expected
        ):
            raise ValueError(f"hidden must be a BF16 tensor with shape {expected}")
        if hidden.device.type != "cpu":
            raise ValueError("hidden must be a CPU tensor")

    checkpoint = Checkpoint(checkpoint) if not isinstance(checkpoint, Checkpoint) else checkpoint
    if not FLASH.first_k_dense_replace <= layer_id < FLASH.num_hidden_layers:
        raise ValueError("the layer entries require a sparse FFN layer in the text backbone")

    x_mixed = torch.zeros(num_ranks, capacity, D, dtype=torch.bfloat16)
    if hidden is None:

        generator = torch.Generator().manual_seed(7)
        embeddings = torch.randn(num_ranks, capacity, D, generator=generator).bfloat16()
        mhc = checkpoint.mhc(layer_id, "ffn")
        with torch.no_grad():
            for rank in range(num_ranks):
                stream = hc_seed(embeddings[rank])
                mix = hc_mixes(stream, mhc["function"], mhc["scale"], mhc["base"])[0]
                x_mixed[rank, :num_tokens] = hc_pre(stream, mix)[:num_tokens]
    else:
        x_mixed[:, :num_tokens] = hidden

    # Ordered local shapes match the L3 decode/prefill MoE ABI.
    layout = {
        "norm_weight": ((D,), torch.bfloat16),
        "gate_weight": ((N_EXPERTS, D), torch.bfloat16),
        "correction_bias": ((N_EXPERTS,), torch.float32),
        "w_gate_up": ((N_LOCAL_EXPERTS, 2 * MOE_INTER, D), torch.int8),
        "w_gate_up_scale": ((N_LOCAL_EXPERTS, 2 * MOE_INTER), torch.float32),
        "w_down": ((N_LOCAL_EXPERTS, D, MOE_INTER), torch.int8),
        "w_down_scale": ((N_LOCAL_EXPERTS, D), torch.float32),
        "shared_w_gate_up": ((2 * MOE_INTER, D), torch.int8),
        "shared_w_gate_up_scale": ((2 * MOE_INTER,), torch.float32),
        "shared_w_down": ((D, MOE_INTER), torch.int8),
        "shared_w_down_scale": ((D,), torch.float32),
    }
    weights = {}
    for rank in range(num_ranks):
        local = checkpoint.ffn(layer_id, ep_rank=rank, ep=num_ranks)
        if local.keys() != layout.keys():
            raise ValueError("checkpoint FFN tensors differ from the composed MoE input names")
        for name, (shape, dtype) in layout.items():
            value = local[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{name}: expected {shape} {dtype}, got {tuple(value.shape)} {value.dtype}")
            if rank == 0:
                weights[name] = torch.empty((num_ranks, *shape), dtype=dtype)
            weights[name][rank].copy_(value)
        del local, value

    for name in ("shared_w_gate_up", "shared_w_down"):
        weights[name] = pack_int8_nz(weights[name])

    specs = [TensorSpec("x_mixed", list(x_mixed.shape), x_mixed.dtype, init_value=x_mixed)]
    specs += [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in weights.items()
    ]
    specs += [
        TensorSpec("y", [num_ranks, capacity, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, num_ranks),
    ]
    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T_DYN_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_expert_routed_fn,
        config=dict(dump_passes=args.dump_passes, backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane),
        rtol=0.001,
        atol=0.001,
        compare_fn={"output": ratio_reldiff(diff_thd=0.003, pct_thd=0.01)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
