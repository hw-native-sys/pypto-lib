# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek V4.1 DSpark projection, metadata, and three-layer drafter."""

import argparse
import inspect

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import config as C
from config import (
    BLOCK_SIZE, B_DYN, EP_SIZE, FLASH, MOE_TOKENS, TABLE_DYN, WINDOW_CACHE_BYTES,
    DSPARK_DRAFT_LAYERS, DSPARK_N_ROUTED_EXPERTS, DSPARK_NUM_EXPERTS_PER_TOK,
    DSPARK_NOISE_TOKEN_ID, DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH,
)
from rmsnorm import rms_norm
from expert_routed import (
    MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS,
)
from input_pack import pack_x_hc
from decode_moe import DISPATCH_PACKET_WIDTH, RECV_MAX, ROUTE_WIDTH, SIGNAL_PAD, moe

from dspark_attn import dspark_attn, dspark_context_kv
from qkv_proj_rope import quantize_projection_input


# Dynamic shape variables.
T_DYN = pl.dynamic("DSPARK_CONTEXT_T_DYN")
DSPARK_QUERY_T_DYN = pl.dynamic("DSPARK_QUERY_T_DYN")

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--draft-experts-per-rank", type=int, default=DSPARK_N_ROUTED_EXPERTS // EP_SIZE)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.draft_experts_per_rank <= 0 or EP_SIZE * _shape_options.draft_experts_per_rank % 16:
    _shape_parser.error("--draft-experts-per-rank must be positive and give a total expert count aligned to 16")


# model config
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
HEAD_DIM = FLASH.head_dim
ROPE_DIM = FLASH.qk_rope_head_dim
Q_LORA = FLASH.q_lora_rank
LOCAL_H = FLASH.num_attention_heads
LOCAL_O_GROUPS = FLASH.o_groups
O_LORA = FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
VOCAB = FLASH.vocab_size
N_LOCAL_EXPERTS = _shape_options.draft_experts_per_rank
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
MOE_INTER = FLASH.moe_intermediate_size
MX_GROUP = FLASH.weight_block_size[0]
TOPK = DSPARK_NUM_EXPERTS_PER_TOK

MAIN_IN = DSPARK_DRAFT_LAYERS * D
DSPARK_WINDOW = FLASH.sliding_window

# tiling
PROJ_M_TILE = 16
PROJ_N_TILE = 160
PROJ_K_TILE = 128


@pl.jit.inline(auto_scope=False)
def dspark_proj(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    projected: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project active target-tap rows to BF16 with FP32 accumulation."""
    row_tiles = (num_tokens + PROJ_M_TILE - 1) // PROJ_M_TILE
    for block in pl.spmd(row_tiles * (D // PROJ_N_TILE), name_hint="dspark_proj"):
        row = block // (D // PROJ_N_TILE) * PROJ_M_TILE
        column = block % (D // PROJ_N_TILE) * PROJ_N_TILE
        valid_rows = pl.min(PROJ_M_TILE, num_tokens - row)
        acc = pl.tile.create([PROJ_M_TILE, PROJ_N_TILE], pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        acc = pl.set_validshape(acc, valid_rows, PROJ_N_TILE)
        for k in pl.pipeline(0, MAIN_IN, PROJ_K_TILE, stage=3):
            hidden = pl.load(target_hidden, [row, k], [PROJ_M_TILE, PROJ_K_TILE], [valid_rows, PROJ_K_TILE])
            weight = pl.load(main_proj_weight, [k, column], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
            acc = pl.matmul_acc(acc, hidden, weight, init_cond=(k == 0))
        pl.store(pl.set_validshape(acc, valid_rows, PROJ_N_TILE), [row, column], projected)
    return projected


@pl.jit.inline(auto_scope=False)
def dspark_metadata(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    padded_ids: pl.Out[pl.Tensor[[MOE_TOKENS], pl.INT64]],
    padded_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_TOKENS], pl.INT64]],
    padded_indices: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32]],
    rope_cos: pl.Out[pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32]],
    rope_sin: pl.Out[pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
):
    """Build padded query IDs, paged visibility, RoPE rows, and initial HC coefficients."""
    batch = pl.tensor.dim(anchor_ids, 0)
    table_blocks = pl.tensor.dim(block_tables, 2)
    for _ in pl.spmd(1, name_hint="dspark_metadata"):
        for row in pl.range(MOE_TOKENS):
            pl.write(padded_ids, [row], pl.cast(0, pl.INT64))
            rope_cos[row : row + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
            rope_sin[row : row + 1, :] = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
            if row < batch * DSPARK_QUERY_WIDTH:
                rope_cos[row : row + 1, :] = query_rope_cos[row : row + 1, :]
                rope_sin[row : row + 1, :] = query_rope_sin[row : row + 1, :]
            for lane in pl.range(HC_MULT):
                coefficient = pl.cast(0.0, pl.FP32)
                if lane == 0:
                    coefficient = pl.cast(1.0, pl.FP32)
                pl.write(pre_mix, [row, lane], coefficient)
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(padded_slots, [layer, row], pl.cast(-1, pl.INT64))
                empty_indices = pl.full([1, 1, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32, value=-1)
                padded_indices[layer : layer + 1, row : row + 1, :] = empty_indices
        for request in pl.range(batch):
            first = pl.read(first_query_positions, [request])
            if first > 0 and table_blocks > 0:
                prefix = pl.max(first - DSPARK_WINDOW, 0)
                visible = first - prefix + DSPARK_QUERY_WIDTH
                for offset in pl.range(DSPARK_QUERY_WIDTH):
                    row = request * DSPARK_QUERY_WIDTH + offset
                    token = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                    if offset == 0:
                        token = pl.read(anchor_ids, [request])
                    pl.write(padded_ids, [row], token)
                for layer in pl.range(DSPARK_DRAFT_LAYERS):
                    columns = pl.arange(0, [1, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32)
                    positions = pl.add(columns, pl.cast(prefix, pl.INT32))
                    indices = pl.full([1, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32, value=-1)
                    for page in pl.range(prefix // BLOCK_SIZE, (first + DSPARK_QUERY_WIDTH - 1) // BLOCK_SIZE + 1):
                        block = pl.read(block_tables, [layer, request, page % table_blocks])
                        lower = pl.cmp(positions, pl.cast(page * BLOCK_SIZE, pl.INT32), cmp_type=5)
                        upper = pl.cmp(positions, pl.cast((page + 1) * BLOCK_SIZE, pl.INT32), cmp_type=2)
                        in_page = pl.mul(lower, upper)
                        visible_mask = pl.cmp(columns, pl.cast(visible, pl.INT32), cmp_type=2)
                        valid = pl.mul(in_page, visible_mask)
                        slots = pl.add(positions, pl.cast((block - page) * BLOCK_SIZE, pl.INT32))
                        delta = pl.sub(slots, indices)
                        update = pl.mul(valid, delta)
                        indices = pl.add(indices, update)
                    index_row = pl.reshape(indices, [1, 1, DSPARK_SWA_INDEX_WIDTH])
                    for offset in pl.range(DSPARK_QUERY_WIDTH):
                        row = request * DSPARK_QUERY_WIDTH + offset
                        position = first + offset
                        page = position // BLOCK_SIZE % table_blocks
                        block = pl.read(block_tables, [layer, request, page])
                        slot = pl.cast(block * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
                        pl.write(padded_slots, [layer, row], slot)
                        padded_indices[layer : layer + 1, row : row + 1, :] = index_row


def _dspark_drafter(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    recv_meta: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], pl.INT8],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    data_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    routed_output: pld.DistributedTensor,
    combine_arrived: pld.DistributedTensor[[C.EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Compose three draft layers and publish their final HC state and coefficients."""
    target_hidden.bind_dynamic(0, T_DYN)
    context_slots.bind_dynamic(1, T_DYN)
    context_rope_cos.bind_dynamic(0, T_DYN)
    context_rope_sin.bind_dynamic(0, T_DYN)
    query_rope_cos.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    anchor_ids.bind_dynamic(0, B_DYN)
    first_query_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    window_cache.bind_dynamic(1, C.ORI_BLOCKS_DYN)
    context_tokens = pl.tensor.dim(target_hidden, 0)
    main_x = pl.create_tensor([context_tokens, D], dtype=pl.BF16)
    projected = pl.create_tensor([context_tokens, D], dtype=pl.BF16)
    dspark_proj(target_hidden, main_proj_weight, projected, context_tokens)
    rms_norm(projected, main_norm_weight, main_x, context_tokens, width=D, zero_tail=False)

    context_payload, context_scales, context_padded_rows = quantize_projection_input(main_x, context_tokens, row_tile=80)
    for layer in pl.range(DSPARK_DRAFT_LAYERS):
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[layer]
        dspark_context_kv(
            context_payload, context_scales,
            wkv[layer], layer_wkv_scale, kv_norm_weight[layer],
            context_rope_cos, context_rope_sin,
            context_slots[layer], window_cache[layer],
            context_tokens,
        )

    # MOE_TOKENS must cover batch * DSPARK_QUERY_WIDTH query rows.
    padded_ids = pl.create_tensor([MOE_TOKENS], dtype=pl.INT64)
    padded_slots = pl.create_tensor([DSPARK_DRAFT_LAYERS, MOE_TOKENS], dtype=pl.INT64)
    padded_indices = pl.create_tensor([DSPARK_DRAFT_LAYERS, MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32)
    rope_cos = pl.create_tensor([MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    rope_sin = pl.create_tensor([MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    pre_mix = pl.reshape(draft_pre_mix, [MOE_TOKENS, HC_MULT])
    dspark_metadata(
        anchor_ids, first_query_positions, block_tables, query_rope_cos, query_rope_sin,
        padded_ids, padded_slots, padded_indices, rope_cos, rope_sin, pre_mix,
    )
    x_hc = pl.reshape(draft_x_hc, [MOE_TOKENS, HC_MULT, D])
    pack_x_hc(padded_ids, embedding_weight, x_hc, hidden_tile=512)

    batch = pl.tensor.dim(num_sampled, 0)
    query_tokens = batch * DSPARK_QUERY_WIDTH
    for layer in pl.range(DSPARK_DRAFT_LAYERS):
        layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[layer]
        layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[layer]
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[layer]
        layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[layer]
        layer_routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[layer]
        layer_routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[layer]
        layer_routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[layer]
        layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[layer]
        layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[layer]
        layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[layer]
        x_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
        attention_pre_mix = pl.create_tensor([MOE_TOKENS, HC_MULT], dtype=pl.FP32)
        attention_hidden = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        dspark_attn(
            x_hc, pre_mix,
            hc_attn_fn[layer], hc_attn_scale[layer], hc_attn_base[layer], attn_norm_weight[layer],
            wq_a[layer], layer_wq_a_scale, q_norm_weight[layer], wq_b[layer], layer_wq_b_scale,
            wkv[layer], layer_wkv_scale, kv_norm_weight[layer], attn_sink[layer],
            wo_a[layer], wo_a_scale[layer], wo_b[layer], layer_wo_b_scale,
            rope_cos, rope_sin, padded_slots[layer], padded_indices[layer], window_cache[layer],
            attention_pre_mix, attention_hidden,
            query_tokens,
        )

        force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
        moe(
            attention_hidden, attention_pre_mix,
            hc_ffn_fn[layer], hc_ffn_scale[layer], hc_ffn_base[layer],
            ffn_norm_weight[layer],
            gate_weight[layer], correction_bias[layer],
            routed_w1[layer], layer_routed_w1_scale,
            routed_w2[layer], layer_routed_w2_scale,
            routed_w3[layer], layer_routed_w3_scale,
            shared_w1[layer], layer_shared_w1_scale,
            shared_w2[layer], layer_shared_w2_scale,
            shared_w3[layer], layer_shared_w3_scale,
            pre_mix, x_mixed, x_hc,
            recv_meta, recv_x, recv_routes,
            data_arrived, routed_output, combine_arrived,
            query_tokens, rank, moe_epoch + layer,
            force_topk,
            local_experts=N_LOCAL_EXPERTS, topk=TOPK, dispatch_worker_limit=64,
        )



dspark_drafter = pl.jit.inline(auto_scope=False)(_dspark_drafter)
l2_dspark_drafter = pl.jit(auto_scope=False)(_dspark_drafter)


@pl.jit.host
def l3_dspark_drafter(
    target_hidden: pl.Tensor[[C.EP_SIZE, T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[C.EP_SIZE, MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[C.EP_SIZE, D], pl.BF16],
    num_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[C.EP_SIZE, VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[C.EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[C.EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32]],
    moe_epoch: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(1, T_DYN)
    context_slots.bind_dynamic(2, T_DYN)
    context_rope_cos.bind_dynamic(1, T_DYN)
    context_rope_sin.bind_dynamic(1, T_DYN)
    query_rope_cos.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_ids.bind_dynamic(1, B_DYN)
    first_query_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    block_tables.bind_dynamic(3, TABLE_DYN)
    window_cache.bind_dynamic(2, C.ORI_BLOCKS_DYN)

    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        recv_routes = pld.window(recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32)
        l2_dspark_drafter(
            target_hidden[rank], main_proj_weight[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank],
            anchor_ids[rank], first_query_positions[rank],
            block_tables[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank],
            query_rope_cos[rank], query_rope_sin[rank],
            embedding_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank],
            q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale[rank],
            window_cache[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            draft_x_hc[rank], draft_pre_mix[rank],
            recv_meta, recv_x, recv_routes,
            data_arrived, routed_output, combine_arrived,
            rank, moe_epoch,
            device=rank,
        )


def build_drafter_specs():
    """One request on every DP rank, with nonzero embeddings and two context writes."""
    from golden import ScalarSpec, TensorSpec

    dynamic_sizes = {
        "DSPARK_CONTEXT_T_DYN": 6,
        "B_DYN": 1,
        "DSPARK_QUERY_T_DYN": DSPARK_QUERY_WIDTH,
        "TABLE_DYN": 1,
        "ORI_BLOCKS_DYN": 1,
    }
    dtypes = {
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8e4m3fn": torch.float8_e4m3fn,
        "fp8e8m0": torch.float8_e8m0fnu,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }

    initial_values = {
        "embedding_weight": 0.25,
        "num_sampled": 1, "last_sampled": 7, "anchor_ids": 7,
        "first_query_positions": 2,
        "context_rope_cos": 1, "query_rope_cos": 1,
        "target_hidden": 0.25, "main_norm_weight": 1,
    }

    def initial_value(name, shape, dtype):
        if name == "context_slots":
            value = torch.full(shape, -1, dtype=dtype)
            value[:, :, 0] = 32
            value[:, :, 1] = 33
            return value
        if name == "window_cache":
            value = torch.zeros(shape, dtype=torch.uint8)
            value[..., HEAD_DIM:].view(torch.bfloat16).fill_(1.0)
            return value
        if dtype == torch.float8_e8m0fnu:
            return lambda: torch.full(shape, 127, dtype=torch.uint8).view(dtype)
        return initial_values.get(name)

    specs = []
    for parameter in inspect.signature(l3_dspark_drafter._func).parameters.values():
        annotation = parameter.annotation
        if hasattr(annotation, "shape"):
            shape = [size if isinstance(size, int) else dynamic_sizes[size.name] for size in annotation.shape]
            dtype = dtypes[str(annotation.dtype)]
            init = initial_value(parameter.name, shape, dtype)
            specs.append(TensorSpec(parameter.name, shape, dtype, init_value=init))
        else:
            specs.append(ScalarSpec(parameter.name, torch.int32, 1, compile_runtime=True))
    return specs


if __name__ == "__main__":
    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="V4.1 DSpark drafter validation")
    parser.add_argument("--ep", type=int, default=C.EP_SIZE, choices=C.SUPPORTED_EP_SIZES)
    parser.add_argument("--draft-experts-per-rank", type=int, default=N_LOCAL_EXPERTS)
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(EP_SIZE)))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != EP_SIZE:
        parser.error(f"drafter needs {EP_SIZE} devices")

    result = run(
        fn=l3_dspark_drafter,
        specs=build_drafter_specs(),
        golden_fn=None,
        config={
            "backend_type": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "execution_mode": False,
            "ring_heap": (2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2),
        },
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
