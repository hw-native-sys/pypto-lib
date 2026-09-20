# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Expert-parallel MoE dispatch, local expert compute, and routed-output combine."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
import torch
import torch.nn.functional as F

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import FLASH

# The PyPTO specializer resolves static extents outside the function body, so the
# kernel cannot read C.MOE_TOKENS directly.
MOE_TOKENS = C.MOE_TOKENS
D = C.D
MX_GROUP = C.MX_GROUP
MOE_INTER = C.MOE_INTER
TOPK = C.TOPK
N_LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
RECV_MAX = C.RECV_MAX
EP_SIZE = C.EP_SIZE
AUX_WIDTH = C.AUX_WIDTH
ROUTE_WIDTH = C.ROUTE_WIDTH
SKIP_SHARED_TEST = "--skip-shared" in __import__("sys").argv
SKIP_TRANSPORT_TEST = "--skip-transport" in __import__("sys").argv
from models.deepseek_v4_1_flash.golden import gate, rms_norm
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp4
from models.deepseek_v4_1_flash.quantization import dequantize_mxfp8
from models.deepseek_v4_1_flash.quantization import unpack_mx_b_scale

from models.deepseek_v4_1_flash.gate import gate as npu_gate
from models.deepseek_v4_1_flash.expert_shared import expert_shared
from models.deepseek_v4_1_flash.expert_routed import expert_routed
from models.deepseek_v4_1_flash.ep_transport import dispatch, combine


def _golden_expert(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    gate_value = F.linear(x.float(), w1.float()).clamp(max=FLASH.swiglu_limit)
    up_value = F.linear(x.float(), w3.float()).clamp(-FLASH.swiglu_limit, FLASH.swiglu_limit)
    hidden = F.silu(gate_value) * up_value
    output = F.linear(hidden, w2.float())
    if weights is not None:
        output = output * weights
    return output


def golden_moe(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    routed_w3: torch.Tensor,
    routed_w3_scale: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w1_scale: torch.Tensor,
    shared_w2: torch.Tensor,
    shared_w2_scale: torch.Tensor,
    shared_w3: torch.Tensor,
    shared_w3_scale: torch.Tensor,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """CPU reference for the V4.1 MoE layer, over rank-local tokens."""
    shape = x.shape
    normalized = rms_norm(x.reshape(-1, shape[-1]), norm_weight)
    active_tokens = (
        normalized.shape[0] if num_tokens is None else min(max(num_tokens, 0), normalized.shape[0])
    )
    normalized = normalized[:active_tokens]
    routed_w1 = dequantize_mxfp4(routed_w1, routed_w1_scale)
    routed_w2 = dequantize_mxfp4(routed_w2, routed_w2_scale)
    routed_w3 = dequantize_mxfp4(routed_w3, routed_w3_scale)
    shared_w1 = dequantize_mxfp8(shared_w1, unpack_mx_b_scale(shared_w1_scale)).transpose(-2, -1)
    shared_w2 = dequantize_mxfp8(shared_w2, unpack_mx_b_scale(shared_w2_scale)).transpose(-2, -1)
    shared_w3 = dequantize_mxfp8(shared_w3, unpack_mx_b_scale(shared_w3_scale)).transpose(-2, -1)
    route_weights, expert_indices = gate(normalized, gate_weight, correction_bias)
    output = _golden_expert(normalized, shared_w1, shared_w2, shared_w3)
    for expert_id in range(routed_w1.shape[0]):
        token_rows, route_columns = torch.where(expert_indices == expert_id)
        if token_rows.numel() == 0:
            continue
        routed = _golden_expert(
            normalized[token_rows],
            routed_w1[expert_id],
            routed_w2[expert_id],
            routed_w3[expert_id],
        )
        routed = routed * route_weights[token_rows, route_columns].unsqueeze(-1)
        output[token_rows] += routed
    result = x.reshape(-1, shape[-1]).clone()
    result[:active_tokens] = output.to(x.dtype)
    return result.reshape(shape)


@pl.jit.inline
def moe_core(
    x: pl.Tensor[[C.T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    # Device ABI: the checkpoint [expert,out,in] FP4 weights are converted offline to
    # [expert,in,out] FP8.
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, C.MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[C.T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    t = MOE_TOKENS
    with pl.spmd(t, name_hint="moe_output_zero") as _output_zero_tid:
        zero_t = pl.tile.get_block_idx()
        zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        output = pl.store(zero_row, [zero_t, 0], output)

    x_norm_mx = pl.create_tensor([t, D], dtype=pl.FP8E4M3FN)
    x_norm_scale = pl.create_tensor(
        [1, t * (D // MX_GROUP)], dtype=pl.FP8E8M0
    )
    indices = pl.create_tensor([t, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([t, TOPK], dtype=pl.FP32)
    npu_gate(x, norm_weight, gate_weight, correction_bias, num_tokens,
             x_norm_mx, x_norm_scale, indices, weights)

    shared_output = pl.create_tensor([t, D], dtype=pl.BF16)
    if SKIP_SHARED_TEST:
        with pl.spmd(t, name_hint="shared_zero"):
            shared_t = pl.tile.get_block_idx()
            zero_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            shared_output = pl.store(zero_row, [shared_t, 0], shared_output)
    else:
        expert_shared(x_norm_mx, x_norm_scale, shared_w1, shared_w1_scale,
                      shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                      shared_output)

    if SKIP_TRANSPORT_TEST:
        with pl.spmd(t, name_hint="moe_skip_transport_output", deps=[_output_zero_tid]):
            out_t = pl.tile.get_block_idx()
            if out_t < num_tokens:
                out_row = pl.load(shared_output, [out_t, 0], [1, D])
                output = pl.store(out_row, [out_t, 0], output)
    else:
        recv_x_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.FP8E4M3FN)
        # The transport side needs contiguous ND backing; expert_routed then creates its
        # static MX_A_ZZ view over that same backing.
        recv_scale_local_backing = pl.create_tensor(
            [1, N_LOCAL_EXPERTS * RECV_MAX * (D // MX_GROUP)],
            dtype=pl.FP8E8M0,
        )
        recv_weight_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
        recv_route_local = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
        recv_count_local = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
        recv_meta_local = pl.create_tensor([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        dispatch(indices, x_norm_mx, x_norm_scale, weights, recv_x_local, recv_scale_local_backing,
                 recv_weight_local, recv_route_local, recv_count_local, recv_meta_local,
                 recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived,
                 data_arrived, combine_arrived, moe_epoch - 1, num_tokens, ep_rank, moe_epoch)

        routed_y = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
        # dispatch already filled this backing; expert_routed views it as MX_A_ZZ.
        expert_routed(recv_x_local, recv_scale_local_backing, recv_weight_local, recv_count_local,
                      routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                      routed_w2, routed_w2_scale, routed_y)
        # combine writes the final output directly: a dynamically shaped intermediate
        # would escape its defining scope during PTOAS SSA conversion.
        combine(routed_y, recv_route_local, shared_output, output, recv_meta_local,
                routed_output, combine_arrived, num_tokens, ep_rank, moe_epoch)

    return output


@pl.jit
def moe(
    x: pl.Tensor[[C.T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    # Device ABI: the checkpoint [expert,out,in] FP4 weights are converted offline to
    # [expert,in,out] FP8.
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, C.MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[C.T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    return moe_core(
        x, norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale, shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        recv_meta, recv_x, recv_scale, recv_weights,
        recv_routes, arrived, data_arrived, routed_output,
        combine_arrived, output, num_tokens, ep_rank,
        moe_epoch,
    )


@pl.jit.host
def l3_moe(
    x: pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16],
    norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, C.MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS, D, C.MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    output: pl.Out[pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """EP host driver matching the V4 ``l3_moe`` validation structure."""
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    # Counter views remain packed, but each allocation owns its cache line.
    arrived_buf = pld.alloc_window_buffer(64)
    data_arrived_buf = pld.alloc_window_buffer(64)
    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer(64)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        routed_w1_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w1_scale[r]
        routed_w2_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w2_scale[r]
        routed_w3_scale_r: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = routed_w3_scale[r]
        shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[r]
        shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[r]
        shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[r]
        moe(
            x[r], norm_weight[r], gate_weight[r], correction_bias[r],
            routed_w1[r], routed_w1_scale_r, routed_w2[r], routed_w2_scale_r,
            routed_w3[r], routed_w3_scale_r, shared_w1[r], shared_w1_scale_r,
            shared_w2[r], shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived, output[r],
            num_tokens, r, moe_epoch, device=r,
        )


# ---------------------------------------------------------------------------
# Standalone full-MoE validation harness
# ---------------------------------------------------------------------------


def _fp8_dtype():
    return torch.float8_e4m3fn


def _e8m0_dtype():
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def _route_bias() -> torch.Tensor:
    # The selected ids are spread across the global expert id space so dispatch
    # and combine exercise cross-rank traffic.  The bias offset dominates the
    # O(1) score, which keeps top-k off the boundary where an FP32
    # reduction-order difference could swap an expert and change the output.
    selected = (torch.arange(TOPK, dtype=torch.int64) * max(1, C.N_EXPERTS // TOPK))
    selected = torch.remainder(selected, C.N_EXPERTS)
    bias = torch.full((C.N_EXPERTS,), -1.0, dtype=torch.float32)
    bias[selected] = 4.0
    return bias


def build_tensor_specs(num_tokens: int = MOE_TOKENS):
    import torch

    from golden.spec import ScalarSpec, TensorSpec
    from models.deepseek_v4_1_flash.expert_routed import (
        ROUTED_DEQUANT_STD,
        gen_routed_mx_weights,
    )
    from models.deepseek_v4_1_flash.quantization import gen_mxfp8_weight_kn_v41

    active = max(0, min(MOE_TOKENS, int(num_tokens)))
    torch.manual_seed(41)

    x = (torch.randn(EP_SIZE, MOE_TOKENS, D) * 0.25).to(torch.bfloat16)
    norm_weight = torch.ones(EP_SIZE, D, dtype=torch.bfloat16)
    gate_weight = (torch.randn(C.N_EXPERTS, D) / D ** 0.5)
    gate_weight = gate_weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    correction_bias = _route_bias().unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    routed_w1_shape = (EP_SIZE, N_LOCAL_EXPERTS, D, C.MOE_INTER)
    routed_w1_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER)
    routed_w2_shape = (EP_SIZE, N_LOCAL_EXPERTS, C.MOE_INTER, D)
    routed_w2_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D)
    routed_w3_shape = routed_w1_shape
    routed_w3_scale_shape = routed_w1_scale_shape

    # Real routed shards at the deployment magnitudes: the checkpoint MXFP4
    # [expert, out, in] weights go through the same offline conversion the
    # device ABI expects, and every rank draws its own seed.
    routed_w1_list, routed_w1_scale_list = [], []
    routed_w3_list, routed_w3_scale_list = [], []
    routed_w2_list, routed_w2_scale_list = [], []
    for rank in range(EP_SIZE):
        rw1, rw1_s, rw3, rw3_s, rw2, rw2_s = gen_routed_mx_weights(
            N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3
        )
        routed_w1_list.append(rw1)
        routed_w1_scale_list.append(rw1_s)
        routed_w3_list.append(rw3)
        routed_w3_scale_list.append(rw3_s)
        routed_w2_list.append(rw2)
        routed_w2_scale_list.append(rw2_s)
    routed_w1 = torch.stack(routed_w1_list)
    routed_w1_scale = torch.stack(routed_w1_scale_list)
    routed_w3 = torch.stack(routed_w3_list)
    routed_w3_scale = torch.stack(routed_w3_scale_list)
    routed_w2 = torch.stack(routed_w2_list)
    routed_w2_scale = torch.stack(routed_w2_scale_list)

    shared_std = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1, sw1_s = gen_mxfp8_weight_kn_v41(C.MOE_INTER, D, shared_std["w1"], chan_cv=0.50, seed=101)
    sw3, sw3_s = gen_mxfp8_weight_kn_v41(C.MOE_INTER, D, shared_std["w3"], chan_cv=0.50, seed=102)
    sw2, sw2_s = gen_mxfp8_weight_kn_v41(D, C.MOE_INTER, shared_std["w2"], chan_cv=0.33, seed=103)
    shared_w1 = sw1.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    shared_w1_scale = sw1_s.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    shared_w3 = sw3.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    shared_w3_scale = sw3_s.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    shared_w2 = sw2.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    shared_w2_scale = sw2_s.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()

    fp8 = _fp8_dtype()
    e8m0 = _e8m0_dtype()
    specs = [
        TensorSpec("x", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16, init_value=lambda: x),
        TensorSpec("norm_weight", [EP_SIZE, D], torch.bfloat16, init_value=lambda: norm_weight),
        TensorSpec("gate_weight", [EP_SIZE, C.N_EXPERTS, D], torch.float32, init_value=lambda: gate_weight),
        TensorSpec("correction_bias", [EP_SIZE, C.N_EXPERTS], torch.float32, init_value=lambda: correction_bias),
        TensorSpec("routed_w1", list(routed_w1_shape), fp8, init_value=lambda: routed_w1),
        TensorSpec("routed_w1_scale", list(routed_w1_scale_shape), e8m0, init_value=lambda: routed_w1_scale),
        TensorSpec("routed_w2", list(routed_w2_shape), fp8, init_value=lambda: routed_w2),
        TensorSpec("routed_w2_scale", list(routed_w2_scale_shape), e8m0, init_value=lambda: routed_w2_scale),
        TensorSpec("routed_w3", list(routed_w3_shape), fp8, init_value=lambda: routed_w3),
        TensorSpec("routed_w3_scale", list(routed_w3_scale_shape), e8m0, init_value=lambda: routed_w3_scale),
        TensorSpec("shared_w1", [EP_SIZE, D, C.MOE_INTER], fp8, init_value=lambda: shared_w1),
        TensorSpec("shared_w1_scale", [EP_SIZE, D // MX_GROUP, C.MOE_INTER], e8m0, init_value=lambda: shared_w1_scale),
        TensorSpec("shared_w2", [EP_SIZE, C.MOE_INTER, D], fp8, init_value=lambda: shared_w2),
        TensorSpec("shared_w2_scale", [EP_SIZE, C.MOE_INTER // MX_GROUP, D], e8m0, init_value=lambda: shared_w2_scale),
        TensorSpec("shared_w3", [EP_SIZE, D, C.MOE_INTER], fp8, init_value=lambda: shared_w3),
        TensorSpec("shared_w3_scale", [EP_SIZE, D // MX_GROUP, C.MOE_INTER], e8m0, init_value=lambda: shared_w3_scale),

        TensorSpec("output", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, active),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=1),
    ]

    for spec in specs:
        if spec.name not in {"x", "output", "num_tokens", "moe_epoch"}:
            spec.resident = "stacked"
    return specs


def golden_l3_moe(tensors):
    import torch

    from models.deepseek_v4_1_flash.gate import golden_gate_core
    from models.deepseek_v4_1_flash.expert_shared import golden_expert_shared
    from models.deepseek_v4_1_flash.expert_routed import golden_expert_routed
    from models.deepseek_v4_1_flash.quantization import pack_mx_a_scale, unpack_mx_a_scale

    active = max(0, min(MOE_TOKENS, int(tensors.get("num_tokens", MOE_TOKENS))))
    fp8 = _fp8_dtype()
    e8m0 = _e8m0_dtype()

    all_indices = []
    all_weights = []
    all_x_mx = []
    all_scale_packed = []
    all_scale_logical = []
    all_shared = []

    dummy_tid2eid = torch.zeros(FLASH.vocab_size, TOPK, dtype=torch.int32)
    dummy_input_ids = torch.zeros(MOE_TOKENS, dtype=torch.int64)

    for src in range(EP_SIZE):
        x_norm_mx = torch.zeros(MOE_TOKENS, D, dtype=torch.uint8).view(fp8)
        x_norm_scale = torch.zeros(1, MOE_TOKENS * (D // MX_GROUP), dtype=torch.uint8).view(e8m0)
        indices = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.int32)
        weights = torch.zeros(MOE_TOKENS, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed": tensors["x"][src],
            "norm_w": tensors["norm_weight"][src],
            "gate_w": tensors["gate_weight"][src],
            "gate_bias": tensors["correction_bias"][src],
            "layer_id": 0,
            "num_tokens": active,
            "tid2eid": dummy_tid2eid,
            "input_ids": dummy_input_ids,
            "x_norm_mx": x_norm_mx,
            "x_norm_scale": x_norm_scale,
            "indices": indices,
            "weights": weights,
        })
        shared = torch.zeros(MOE_TOKENS, D, dtype=torch.bfloat16)
        if not SKIP_SHARED_TEST:
            golden_expert_shared({
                "x_local": x_norm_mx,
                "x_local_scale": x_norm_scale,
                "shared_w1": tensors["shared_w1"][src],
                "shared_w1_scale": tensors["shared_w1_scale"][src],
                "shared_w3": tensors["shared_w3"][src],
                "shared_w3_scale": tensors["shared_w3_scale"][src],
                "shared_w2": tensors["shared_w2"][src],
                "shared_w2_scale": tensors["shared_w2_scale"][src],
                "sh": shared,
            })
        all_indices.append(indices)
        all_weights.append(weights)
        all_x_mx.append(x_norm_mx)
        scale_bytes = x_norm_scale.view(torch.uint8).reshape(MOE_TOKENS, D // MX_GROUP)
        all_scale_logical.append(unpack_mx_a_scale(scale_bytes))
        all_scale_packed.append(x_norm_scale)
        all_shared.append(shared)

    if SKIP_TRANSPORT_TEST:
        output = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
        for src in range(EP_SIZE):
            for t in range(active):
                output[src, t] = all_shared[src][t]
        tensors["output"][:] = output
        return

    send_counts = torch.zeros(EP_SIZE, EP_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
    for src in range(EP_SIZE):
        for t in range(active):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                send_counts[src, dst, local_e] += 1

    dst_recv_y = []
    for dst in range(EP_SIZE):
        recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.uint8).view(fp8)
        recv_scale_logical = torch.zeros(N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP, dtype=torch.uint8)
        recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
        recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
        slot_offsets = torch.zeros(EP_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
        running = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
        for src in range(EP_SIZE):
            slot_offsets[src] = running.clone()
            running = running + send_counts[src, dst]
        recv_count[:, 0] = running

        for src in range(EP_SIZE):
            cursors = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
            for t in range(active):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    route_dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                    if route_dst != dst:
                        continue
                    slot = int(slot_offsets[src, local_e].item() + cursors[local_e].item())
                    cursors[local_e] += 1
                    recv_x[local_e, slot] = all_x_mx[src][t]
                    recv_scale_logical[local_e * RECV_MAX + slot] = all_scale_logical[src][t]
                    recv_weights[local_e, slot] = all_weights[src][t, k]

        recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x": recv_x,
            "recv_mx_scale": pack_mx_a_scale(recv_scale_logical).view(e8m0),
            "recv_weights": recv_weights,
            "recv_expert_count": recv_count,
            "routed_w1": tensors["routed_w1"][dst],
            "routed_w1_scale": tensors["routed_w1_scale"][dst],
            "routed_w3": tensors["routed_w3"][dst],
            "routed_w3_scale": tensors["routed_w3_scale"][dst],
            "routed_w2": tensors["routed_w2"][dst],
            "routed_w2_scale": tensors["routed_w2_scale"][dst],
            "recv_y": recv_y,
        })
        dst_recv_y.append(recv_y)

    output = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    for src in range(EP_SIZE):
        routed = torch.zeros(MOE_TOKENS * TOPK, D, dtype=torch.bfloat16)
        cursors = {}
        for t in range(active):
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                src_off = int(send_counts[:src, dst, local_e].sum().item())
                cursor = cursors.get((dst, local_e), 0)
                cursors[(dst, local_e)] = cursor + 1
                routed[t * TOPK + k] = dst_recv_y[dst][local_e, src_off + cursor]
        for t in range(active):
            acc = all_shared[src][t].float()
            for k in range(TOPK):
                acc = acc + routed[t * TOPK + k].float()
            output[src, t] = acc.to(torch.bfloat16)

    tensors["output"][:] = output


def moe_output_compare(num_tokens: int):
    import torch

    from golden.validation import ratio_reldiff

    active = max(0, min(MOE_TOKENS, int(num_tokens)))
    active_compare = ratio_reldiff(diff_thd=3e-3, pct_thd=0.02)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, f"    output shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        if not torch.isfinite(actual.float()).all().item():
            return False, "    output contains NaN or Inf"

        rows = [(rank, t) for rank in range(actual.shape[0]) for t in range(active)]
        if rows:
            rank_idx = torch.tensor([r for r, _ in rows], dtype=torch.long)
            token_idx = torch.tensor([t for _, t in rows], dtype=torch.long)
            ok, detail = active_compare(actual[rank_idx, token_idx], expected[rank_idx, token_idx], **kwargs)
            if not ok:
                return False, f"    active token rows:\n{detail}"
        mask = torch.ones(actual.shape[:2], dtype=torch.bool)
        for rank, t in rows:
            mask[rank, t] = False
        inactive_actual = actual[mask]
        inactive_expected = expected[mask]
        if inactive_actual.numel() and not torch.equal(inactive_actual, inactive_expected):
            diff = (inactive_actual.float() - inactive_expected.float()).abs()
            return False, (
                f"    inactive rows changed: values={int((inactive_actual != inactive_expected).sum().item())}, "
                f"max_abs_diff={float(diff.max().item()):.6g}"
            )
        return True, ""

    compare.__name__ = f"moe_output_compare(num_tokens={active})"
    return compare


__all__ = [
    "golden_moe", "golden_l3_moe", "build_tensor_specs",
    "moe", "l3_moe",
]


if __name__ == "__main__":
    import argparse
    import pathlib
    import sys

    _model_dir = pathlib.Path(__file__).resolve().parent
    sys.path = [item for item in sys.path if pathlib.Path(item or ".").resolve() != _model_dir]

    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=list(C.SUPPORTED_EP_SIZES))
    parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=list(C.SUPPORTED_TP_SIZES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument("--num-tokens", type=int, default=MOE_TOKENS)
    parser.add_argument("--moe-epoch", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-shared", action="store_true", help="diagnostic only: zero shared expert output and still run gate/dispatch/routed/combine")
    parser.add_argument("--skip-transport", action="store_true", help="diagnostic only: bypass dispatch/routed/combine and write shared output on active rows")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device_ids = [int(d) for d in args.device.split(",") if d != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    if args.ep != EP_SIZE or args.tp != C.TP_SIZE:
        raise SystemExit(
            "--ep/--tp are parsed by config.py before argparse; run the module "
            "with the desired values, for example `python -m models.deepseek_v4_1_flash.moe --ep 8 --tp 2 ...`"
        )

    result = run(
        fn=l3_moe,
        specs=build_tensor_specs(num_tokens=args.num_tokens),
        golden_fn=golden_l3_moe,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            # Full MoE keeps dispatch buffers, shared matmul, routed matmul and
            # distributed windows live together; use the same 1 GiB ring heap as
            # the standalone routed expert regression.
            ring_heap=1_073_741_824,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "output": moe_output_compare(args.num_tokens),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
