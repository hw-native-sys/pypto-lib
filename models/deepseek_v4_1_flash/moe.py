# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: a5
"""Expert-parallel MoE dispatch, local expert compute, and routed-output combine."""

import math
import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import FLASH, HC_DIM, HC_MULT, MIX_HC

C.RECV_MAX = C.EP_SIZE * C.MOE_TOKENS

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

from models.deepseek_v4_1_flash.gate import gate_normalized as npu_gate
from models.deepseek_v4_1_flash.expert_shared import expert_shared
from models.deepseek_v4_1_flash.expert_routed import (
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
    expert_routed,
)
from models.deepseek_v4_1_flash.ep_transport import dispatch, combine
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.rmsnorm import golden_rms_norm, rms_norm


def _gen_routed_mx_weights_fixture(n_experts, dequant_std, seed_base=0):
    """Generate the packed FP4 device ABI used by the routed kernel."""
    from models.deepseek_v4_1_flash.expert_routed import (
        gen_routed_mxfp4_weights,
    )
    return gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base)


@pl.jit.inline(auto_scope=False)
def _moe_core(
    x_normed: pl.Tensor[[C.T_DYN, D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    # Device ABI: checkpoint [expert,out,in] FP4 weights stay packed in HBM and
    # are expanded to the matmul FP8 staging layout through the on-device LUT.
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
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
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
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
    npu_gate(x_normed, gate_weight, correction_bias, num_tokens,
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
            if out_t < num_tokens and pl.read(token_owners, [out_t]) == ep_rank:
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
                 data_arrived, token_owners, num_tokens, ep_rank, moe_epoch)

        routed_y = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
        # dispatch already filled this backing; expert_routed views it as MX_A_ZZ.
        expert_routed(recv_x_local, recv_scale_local_backing, recv_weight_local, recv_count_local,
                      routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                      routed_w2, routed_w2_scale, mxfp4_pair_lut, routed_y)
        # combine writes the final output directly: a dynamically shaped intermediate
        # would escape its defining scope during PTOAS SSA conversion.
        combine(routed_y, recv_route_local, shared_output, output, recv_meta_local,
                routed_output, combine_arrived, token_owners, num_tokens, ep_rank, moe_epoch)


@pl.jit.inline(auto_scope=False)
def moe(
    x_hc: pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[
        [D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[
        [C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[
        [D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
    next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[C.T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8
    ],
    recv_weights: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32
    ],
    recv_routes: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32
    ],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32]:
    """Run delayed mHC pre-mix, MoE, and residual expansion.

    The input pre_mix is produced by the preceding mHC sublayer. This
    sublayer's coefficient generation is returned as next_pre_mix for the
    following sublayer.
    """
    t = MOE_TOKENS
    post_mix = pl.create_tensor([t, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor(
        [t, HC_MULT, HC_MULT], dtype=pl.FP32
    )
    ffn_input = pl.create_tensor([t, D], dtype=pl.BF16)
    mhc_mixes(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        next_pre_mix, post_mix, residual_mix,
    )
    mhc_pre(x_hc, pre_mix, x_mixed)
    rms_norm(x_mixed, norm_weight, ffn_input)

    # Keep the transport windows and routed result alive through combine.
    with pl.scope():
        # combine writes owner rows only, and _moe_core zero-initialises the
        # whole buffer as its first action, so non-owner/inactive rows reach
        # mhc_post as the pure residual expansion.
        #
        # This must stay a pl.create_tensor(): pl.full() produces a fill value
        # with no inferable tensor metadata, and the specializer resolves
        # combine's `ffn_out` parameter through this definition.  Swapping in
        # pl.full() fails to compile with
        #   "missing inferred tensor metadata for parameter 'ffn_out' of 'combine'"
        sublayer = pl.create_tensor([t, D], dtype=pl.BF16)
        _moe_core(
            ffn_input, gate_weight, correction_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
            routed_w3, routed_w3_scale, mxfp4_pair_lut, shared_w1, shared_w1_scale,
            shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
            token_owners, recv_meta, recv_x, recv_scale, recv_weights,
            recv_routes, arrived, data_arrived, routed_output,
            combine_arrived, sublayer,
            num_tokens, ep_rank, group_base, tp_rank, moe_epoch,
        )
        mhc_post(sublayer, x_hc, post_mix, residual_mix, x_next)
    return x_next


@pl.jit
def moe_test(
    x_hc: pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[
        [D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w2: pl.Tensor[[C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[
        [C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w3: pl.Tensor[[D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[
        [D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
    next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, HC_MULT], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[C.T_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8
    ],
    recv_weights: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32
    ],
    recv_routes: pld.DistributedTensor[
        [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32
    ],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[C.T_DYN, HC_MULT, D], pl.FP32]:
    x_hc.bind_dynamic(0, C.T_DYN)
    pre_mix.bind_dynamic(0, C.T_DYN)
    next_pre_mix.bind_dynamic(0, C.T_DYN)
    x_mixed.bind_dynamic(0, C.T_DYN)
    x_next.bind_dynamic(0, C.T_DYN)
    return moe(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale, mxfp4_pair_lut, shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        token_owners, next_pre_mix, x_mixed, x_next,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        num_tokens, ep_rank, group_base, tp_rank, moe_epoch,
    )


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, C.N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        pl.UINT8,
    ],
    routed_w1_scale: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
    ],
    routed_w2: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        pl.UINT8,
    ],
    routed_w2_scale: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D],
        pl.FP8E8M0,
    ],
    routed_w3: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        pl.UINT8,
    ],
    routed_w3_scale: pl.Tensor[
        [EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER],
        pl.FP8E8M0,
    ],
    mxfp4_pair_lut: pl.Tensor[[EP_SIZE, 2, 256], pl.INT16],
    shared_w1: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[
        [EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0
    ],
    shared_w2: pl.Tensor[[EP_SIZE, C.MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[
        [EP_SIZE, C.MOE_INTER // MX_GROUP, D], pl.FP8E8M0
    ],
    shared_w3: pl.Tensor[[EP_SIZE, D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[
        [EP_SIZE, D // MX_GROUP, C.MOE_INTER], pl.FP8E8M0
    ],
    token_owners: pl.Tensor[[EP_SIZE, MOE_TOKENS], pl.INT32],
    next_pre_mix: pl.Out[
        pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32]
    ],
    x_mixed: pl.Out[
        pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16]
    ],
    x_next: pl.Out[
        pl.Tensor[[EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32]
    ],
    num_tokens: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """EP host driver for the mHC-wrapped Flash MoE block."""
    recv_meta_buf = pld.alloc_window_buffer(
        [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32
    )
    recv_x_buf = pld.alloc_window_buffer(
        [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8
    )
    recv_scale_buf = pld.alloc_window_buffer(
        [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8
    )
    recv_weights_buf = pld.alloc_window_buffer(
        [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32
    )
    recv_routes_buf = pld.alloc_window_buffer(
        [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer(
        [MOE_TOKENS * TOPK, D], dtype=pl.BF16
    )
    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(
            recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32
        )
        recv_x = pld.window(
            recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8
        )
        recv_scale = pld.window(
            recv_scale_buf,
            [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP],
            dtype=pl.UINT8,
        )
        recv_weights = pld.window(
            recv_weights_buf,
            [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH],
            dtype=pl.FP32,
        )
        recv_routes = pld.window(
            recv_routes_buf,
            [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH],
            dtype=pl.INT32,
        )
        arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32
        )
        routed_output = pld.window(
            routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16
        )
        combine_arrived = pld.window(
            combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32
        )
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
        moe_test(
            x_hc[r], pre_mix[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_weight[r], gate_weight[r], correction_bias[r],
            routed_w1[r], routed_w1_scale_r, routed_w2[r], routed_w2_scale_r,
            routed_w3[r], routed_w3_scale_r, mxfp4_pair_lut[r], shared_w1[r], shared_w1_scale_r,
            shared_w2[r], shared_w2_scale_r, shared_w3[r], shared_w3_scale_r,
            token_owners[r], next_pre_mix[r], x_mixed[r], x_next[r],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            num_tokens, r, pl.const(0, pl.INT32), r, moe_epoch, device=r,
        )


# ---------------------------------------------------------------------------
# Full mHC+MoE validation harness
# ---------------------------------------------------------------------------


def _fp8_dtype():
    return torch.float8_e4m3fn


def _e8m0_dtype():
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def _owner_pattern() -> torch.Tensor:
    # Every rank receives the same TP-owner row map; rank r only writes rows it owns.
    return torch.arange(MOE_TOKENS, dtype=torch.int32).remainder(EP_SIZE)


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


def _build_moe_tensor_specs(num_tokens: int = MOE_TOKENS):
    import torch

    from golden.spec import ScalarSpec, TensorSpec
    from models.deepseek_v4_1_flash.expert_routed import (
        ROUTED_DEQUANT_STD,
    )
    from models.deepseek_v4_1_flash.quantization import (
        build_mxfp4_pair_lut,
        gen_mxfp8_weight_kn_v41,
    )

    active = max(0, min(MOE_TOKENS, int(num_tokens)))
    torch.manual_seed(41)

    x = (torch.randn(EP_SIZE, MOE_TOKENS, D) * 0.25).to(torch.bfloat16)
    norm_weight = torch.ones(EP_SIZE, D, dtype=torch.bfloat16)
    gate_weight = (torch.randn(C.N_EXPERTS, D) / D ** 0.5)
    gate_weight = gate_weight.unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
    correction_bias = _route_bias().unsqueeze(0).expand(EP_SIZE, -1).contiguous()
    token_owners = _owner_pattern().unsqueeze(0).expand(EP_SIZE, -1).contiguous()

    routed_w1_shape = (EP_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
    routed_w1_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (D // MX_GROUP), C.MOE_INTER)
    routed_w2_shape = (EP_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
    routed_w2_scale_shape = (EP_SIZE, N_LOCAL_EXPERTS * (C.MOE_INTER // MX_GROUP), D)
    routed_w3_shape = routed_w1_shape
    routed_w3_scale_shape = routed_w1_scale_shape

    # Real routed shards at deployment magnitudes: checkpoint MXFP4
    # [expert, out, in] weights follow the packed device ABI, and every rank
    # draws its own seed.
    routed_w1_list, routed_w1_scale_list = [], []
    routed_w3_list, routed_w3_scale_list = [], []
    routed_w2_list, routed_w2_scale_list = [], []
    for rank in range(EP_SIZE):
        rw1, rw1_s, rw3, rw3_s, rw2, rw2_s = _gen_routed_mx_weights_fixture(
            N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3
        )
        routed_w1_list.append(rw1)
        routed_w1_scale_list.append(rw1_s)
        routed_w3_list.append(rw3)
        routed_w3_scale_list.append(rw3_s)
        routed_w2_list.append(rw2)
        routed_w2_scale_list.append(rw2_s)
    routed_w1 = torch.stack(routed_w1_list)
    mxfp4_pair_lut = build_mxfp4_pair_lut().unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous()
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
        TensorSpec("routed_w1", list(routed_w1_shape), torch.uint8, init_value=lambda: routed_w1),
        TensorSpec("routed_w1_scale", list(routed_w1_scale_shape), e8m0, init_value=lambda: routed_w1_scale),
        TensorSpec("routed_w2", list(routed_w2_shape), torch.uint8, init_value=lambda: routed_w2),
        TensorSpec("routed_w2_scale", list(routed_w2_scale_shape), e8m0, init_value=lambda: routed_w2_scale),
        TensorSpec("routed_w3", list(routed_w3_shape), torch.uint8, init_value=lambda: routed_w3),
        TensorSpec("routed_w3_scale", list(routed_w3_scale_shape), e8m0, init_value=lambda: routed_w3_scale),
        TensorSpec(
            "mxfp4_pair_lut", [EP_SIZE, 2, 256], torch.int16,
            init_value=lambda: mxfp4_pair_lut,
        ),
        TensorSpec("shared_w1", [EP_SIZE, D, C.MOE_INTER], fp8, init_value=lambda: shared_w1),
        TensorSpec("shared_w1_scale", [EP_SIZE, D // MX_GROUP, C.MOE_INTER], e8m0, init_value=lambda: shared_w1_scale),
        TensorSpec("shared_w2", [EP_SIZE, C.MOE_INTER, D], fp8, init_value=lambda: shared_w2),
        TensorSpec("shared_w2_scale", [EP_SIZE, C.MOE_INTER // MX_GROUP, D], e8m0, init_value=lambda: shared_w2_scale),
        TensorSpec("shared_w3", [EP_SIZE, D, C.MOE_INTER], fp8, init_value=lambda: shared_w3),
        TensorSpec("shared_w3_scale", [EP_SIZE, D // MX_GROUP, C.MOE_INTER], e8m0, init_value=lambda: shared_w3_scale),
        TensorSpec("token_owners", [EP_SIZE, MOE_TOKENS], torch.int32, init_value=lambda: token_owners),
        TensorSpec("output", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, active),
        ScalarSpec("moe_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=1),
    ]

    for spec in specs:
        if spec.name not in {"x", "token_owners", "output", "num_tokens", "moe_epoch"}:
            spec.resident = "stacked"
    return specs



def build_tensor_specs(num_tokens: int = MOE_TOKENS):
    """Build the default mHC+MoE integration fixture."""
    import torch
    from golden.spec import TensorSpec

    base_specs = _build_moe_tensor_specs(num_tokens)
    common_specs = [
        spec for spec in base_specs
        if spec.name not in {"x", "output", "num_tokens", "moe_epoch"}
    ]
    scalar_specs = [
        spec for spec in base_specs
        if spec.name in {"num_tokens", "moe_epoch"}
    ]

    generator = torch.Generator().manual_seed(3)
    magnitudes = (0.5 + torch.arange(MOE_TOKENS) % 4).reshape(1, -1, 1, 1)
    # The coefficient RMS path must observe distinct token scales. The residual
    # stream stores BF16 model values in FP32 so mHC post performs one rounding.
    x_hc = (torch.randn(
        EP_SIZE, MOE_TOKENS, HC_MULT, D, generator=generator
    ) * magnitudes).bfloat16().float()
    pre_mix = torch.sigmoid(torch.randn(
        EP_SIZE, MOE_TOKENS, HC_MULT, generator=generator
    )) + FLASH.hc_eps
    hc_ffn_fn = torch.randn(
        MIX_HC, HC_DIM, generator=generator
    ) / math.sqrt(HC_DIM)
    hc_ffn_scale = torch.randn(3, generator=generator)
    hc_ffn_base = torch.randn(MIX_HC, generator=generator)
    hc_ffn_fn = hc_ffn_fn.unsqueeze(0).expand(
        EP_SIZE, -1, -1
    ).contiguous()
    hc_ffn_scale = hc_ffn_scale.unsqueeze(0).expand(
        EP_SIZE, -1
    ).contiguous()
    hc_ffn_base = hc_ffn_base.unsqueeze(0).expand(
        EP_SIZE, -1
    ).contiguous()

    mhc_specs = [
        TensorSpec(
            "x_hc", [EP_SIZE, MOE_TOKENS, HC_MULT, D],
            torch.float32, init_value=lambda: x_hc,
        ),
        TensorSpec(
            "pre_mix", [EP_SIZE, MOE_TOKENS, HC_MULT],
            torch.float32, init_value=lambda: pre_mix,
        ),
        TensorSpec(
            "hc_ffn_fn", [EP_SIZE, MIX_HC, HC_DIM],
            torch.float32, init_value=lambda: hc_ffn_fn,
        ),
        TensorSpec(
            "hc_ffn_scale", [EP_SIZE, 3],
            torch.float32, init_value=lambda: hc_ffn_scale,
        ),
        TensorSpec(
            "hc_ffn_base", [EP_SIZE, MIX_HC],
            torch.float32, init_value=lambda: hc_ffn_base,
        ),
    ]
    next_pre_mix_spec = TensorSpec(
        "next_pre_mix", [EP_SIZE, MOE_TOKENS, HC_MULT], torch.float32
    )
    x_mixed_spec = TensorSpec(
        "x_mixed", [EP_SIZE, MOE_TOKENS, D], torch.bfloat16
    )
    x_next_spec = TensorSpec(
        "x_next", [EP_SIZE, MOE_TOKENS, HC_MULT, D], torch.float32
    )
    # Match l3_moe's ABI: mHC inputs, MoE weights/route inputs, outputs,
    # then runtime scalars.
    output_specs = [next_pre_mix_spec, x_mixed_spec, x_next_spec]
    specs = mhc_specs + common_specs + output_specs + scalar_specs
    for spec in specs:
        if spec.name in {
            "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        }:
            spec.resident = "stacked"
    return specs


def _golden_moe_core(tensors):
    import torch

    from models.deepseek_v4_1_flash.gate import golden_gate_normalized_core
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
        golden_gate_normalized_core({
            "x_normed": golden_rms_norm(
                tensors["x"][src], tensors["norm_weight"][src]
            ),
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
            owners = tensors["token_owners"][src].to(torch.int64)
            for t in range(active):
                if int(owners[t]) == src:
                    output[src, t] = all_shared[src][t]
        tensors["output"][:] = output
        return

    send_counts = torch.zeros(EP_SIZE, EP_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
    for src in range(EP_SIZE):
        owners = tensors["token_owners"][src].to(torch.int64)
        for t in range(active):
            if int(owners[t]) != src:
                continue
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
            owners = tensors["token_owners"][src].to(torch.int64)
            cursors = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
            for t in range(active):
                if int(owners[t]) != src:
                    continue
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
            "routed_w1_packed": tensors["routed_w1"][dst],
            "routed_w1_scale": tensors["routed_w1_scale"][dst],
            "routed_w3_packed": tensors["routed_w3"][dst],
            "routed_w3_scale": tensors["routed_w3_scale"][dst],
            "routed_w2_packed": tensors["routed_w2"][dst],
            "routed_w2_scale": tensors["routed_w2_scale"][dst],
            "mxfp4_pair_lut": tensors["mxfp4_pair_lut"][dst],
            "recv_y": recv_y,
        })
        dst_recv_y.append(recv_y)

    output = torch.zeros(EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16)
    for src in range(EP_SIZE):
        owners = tensors["token_owners"][src].to(torch.int64)
        routed = torch.zeros(MOE_TOKENS * TOPK, D, dtype=torch.bfloat16)
        cursors = {}
        for t in range(active):
            if int(owners[t]) != src:
                continue
            for k in range(TOPK):
                eid = int(all_indices[src][t, k].item())
                dst, local_e = divmod(eid, N_LOCAL_EXPERTS)
                src_off = int(send_counts[:src, dst, local_e].sum().item())
                cursor = cursors.get((dst, local_e), 0)
                cursors[(dst, local_e)] = cursor + 1
                routed[t * TOPK + k] = dst_recv_y[dst][local_e, src_off + cursor]
        for t in range(active):
            if int(owners[t]) != src:
                continue
            acc = all_shared[src][t].float()
            for k in range(TOPK):
                acc = acc + routed[t * TOPK + k].float()
            output[src, t] = acc.to(torch.bfloat16)

    tensors["output"][:] = output


def golden_moe(tensors):
    """Compose the validated mHC references around the MoE reference."""
    import torch

    x_hc = tensors["x_hc"]
    next_pre_mix = torch.zeros(
        EP_SIZE, MOE_TOKENS, HC_MULT, dtype=torch.float32
    )
    post_mix = torch.zeros_like(next_pre_mix)
    residual_mix = torch.zeros(
        EP_SIZE, MOE_TOKENS, HC_MULT, HC_MULT, dtype=torch.float32
    )
    x_mixed = torch.zeros(
        EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16
    )
    for rank in range(EP_SIZE):
        pre, post, residual = golden_mhc_mixes(
            x_hc[rank],
            tensors["hc_ffn_fn"][rank],
            tensors["hc_ffn_scale"][rank],
            tensors["hc_ffn_base"][rank],
        )
        next_pre_mix[rank] = pre
        post_mix[rank] = post
        residual_mix[rank] = residual
        x_mixed[rank] = golden_mhc_pre(
            x_hc[rank], tensors["pre_mix"][rank]
        )

    tensors["next_pre_mix"][:] = next_pre_mix
    tensors["x_mixed"][:] = x_mixed

    # Reuse the MoE reference exactly; only its input/output tensors change.
    # This keeps the composition test from growing a second transport
    # reference that could drift from the validated one.
    moe_tensors = dict(tensors)
    moe_tensors["x"] = x_mixed
    sublayer = torch.zeros(
        EP_SIZE, MOE_TOKENS, D, dtype=torch.bfloat16
    )
    moe_tensors["output"] = sublayer
    _golden_moe_core(moe_tensors)

    x_next = torch.zeros(
        EP_SIZE, MOE_TOKENS, HC_MULT, D, dtype=torch.float32
    )
    for rank in range(EP_SIZE):
        x_next[rank] = golden_mhc_post(
            sublayer[rank], x_hc[rank],
            post_mix[rank], residual_mix[rank],
        )
    tensors["x_next"][:] = x_next


def _token_owner_mhc_compare(num_tokens: int):
    """Apply the MoE budget to owner and non-owner rows alike.

    ``combine`` writes owner rows only, so a non-owner row arriving here is the
    pure residual expansion produced by ``mhc_post``.  Its coefficients come
    from ``mhc_mixes``, which the sibling entry validates on device at
    ``rtol=5e-3`` / ``atol=2.5e-5``; demanding bitwise equality here would
    instead assert that torch reproduces the AI core's ``exp``/``rsqrt``
    exactly, which it cannot.  The BF16 rounding in ``mhc_post`` turns a
    sub-ulp coefficient difference into a whole-ulp output difference, so an
    exact gate fails on rounding-boundary flips alone (~0.1% of the elements,
    while the relative budget passes with ~150x headroom).

    Ownership is still enforced: ``token_owners`` must select exactly one rank
    per active token, and the owner and non-owner groups are compared
    independently, so a rank that wrote rows it does not own still fails.
    """
    import torch

    from golden.validation import ratio_reldiff

    active = max(0, min(MOE_TOKENS, int(num_tokens)))
    budget = dict(
        diff_thd=3e-3,
        pct_thd=0.02,
        max_diff_hd=1.0,
    )
    owner_compare = ratio_reldiff(**budget)
    other_compare = ratio_reldiff(**budget)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    output shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        owners = kwargs["inputs"].get("token_owners")
        if owners is None or tuple(owners.shape) != tuple(actual.shape[:2]):
            return False, "    token_owners is missing or has the wrong shape"

        ranks = torch.arange(actual.shape[0], dtype=torch.int64).reshape(-1, 1)
        owner_active = owners[:, :active].to(torch.int64).cpu() == ranks
        if not (owner_active.sum(dim=0) == 1).all().item():
            return False, "    token_owners must select one rank per active token"

        other_mask = torch.ones(actual.shape[:2], dtype=torch.bool)
        other_mask[:, :active] = ~owner_active
        bitwise = int((actual[other_mask] != expected[other_mask]).sum())
        other_ok, other_message = other_compare(
            actual[other_mask].unsqueeze(0),
            expected[other_mask].unsqueeze(0),
            **kwargs,
        )
        if not other_ok:
            return False, (
                "    non-owner or inactive rows left the MoE budget "
                f"({bitwise} elements differ bitwise)\n{other_message}"
            )

        return owner_compare(
            actual[:, :active][owner_active].unsqueeze(0),
            expected[:, :active][owner_active].unsqueeze(0),
            **kwargs,
        )

    compare.__name__ = f"token_owner_mhc_compare(num_tokens={active})"
    return compare


__all__ = [
    "golden_moe", "build_tensor_specs", "moe", "moe_test", "l3_moe",
]


def validate(argv=None):
    """Validate the complete mHC and MoE block against its golden reference."""
    import argparse
    import pathlib

    _model_dir = pathlib.Path(__file__).resolve().parent
    sys.path = [item for item in sys.path if pathlib.Path(item or ".").resolve() != _model_dir]

    from golden import ratio_allclose
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
    parser.add_argument("--skip-transport", action="store_true", help="diagnostic only: bypass dispatch/routed/combine and write shared output on owner rows")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device_ids = [int(d) for d in args.device.split(",") if d != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    if args.ep != EP_SIZE or args.tp != C.TP_SIZE:
        raise SystemExit(
            "--ep/--tp are parsed by config.py before argparse; run the module "
            "with the desired values, for example `python -m models.deepseek_v4_1_flash.moe --ep 8 --tp 2 ...`"
        )

    # The public validation entry is the complete mHC+MoE block.  The plain
    # MoE kernel remains an internal subroutine of ``moe`` and is not exposed
    # as a second standalone runner.
    entry = l3_moe
    specs = build_tensor_specs(args.num_tokens)
    golden_fn = golden_moe
    compare_fn = {
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "x_next": _token_owner_mhc_compare(args.num_tokens),
    }

    result = run(
        fn=entry,
        specs=specs,
        golden_fn=golden_fn,
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
            # The complete mHC+MoE block keeps dispatch buffers, FP8 routed
            # staging, shared matmul, and distributed windows live together.
            # The packed routed path exceeded the previous 1 GiB ring on A5.
            ring_heap=1073741824,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    return result


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,ep", [(2, 4), (4, 4)])
    def test_precision(tp, ep, a5_args):
        """Validate TP2/DP2 and TP4/DP1 with EP-sized device allocations."""
        result = validate(a5_args(tp=tp, ep=ep))
        assert result.passed, result.error


if __name__ == "__main__":
    main()
