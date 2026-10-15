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
"""DeepSeek-V4.1-Flash 40-layer decode backbone with a TP1/EP2, 128-expert CI smoke case."""

import argparse
import inspect

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import config as C

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--experts-per-rank", type=int, default=64)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.experts_per_rank <= 0 or C.EP_SIZE * _shape_options.experts_per_rank % 16:
    _shape_parser.error("--experts-per-rank must be positive and give a total expert count aligned to 16")
C.N_LOCAL_EXPERTS = _shape_options.experts_per_rank


from config import (
    FLASH as M,
    B_DYN,
    LOCAL_T_DYN,
    Q_START_DYN,
    ROUTE_T_DYN,
    T_DYN,
    BLOCK_SIZE,
    COMPRESSED_CACHE_GROUP,
    DECODE_INPUT_MAX_TOKENS,
    DECODE_MAX_TOKENS,
    INDEX_CACHE_GROUP,
    STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)

from config import MOE_TOKENS
from compressor import STATE_WIDTH
from config import TP_SIZE


from decode_c1a_full import decode_c1a_full_sharded
from decode_c1a_reindex import decode_c1a_reindex_sharded
from decode_c1a_reuse import decode_c1a_reuse_sharded
from decode_c2a_full import decode_c2a_full_sharded
from decode_c2a_reuse import decode_c2a_reuse_sharded
from decode_swa import decode_swa_sharded
from rope_tables import precompute_rope_tables
from utils import build_forward_metadata
from expert_routed import (
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
)
from moe import SIGNAL_PAD
from moe import (
    AUX_WIDTH,
    EP_SIZE,
    N_LOCAL_EXPERTS,
    RECV_MAX,
    ROUTE_WIDTH,
    moe,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
CANDIDATE_BLOCK_SIZE = M.candidate_block_size
CANDIDATE_TOPK_BLOCKS = M.candidate_topk_blocks
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size
MX_GROUP = M.weight_block_size[0]


N_LAYERS = M.num_hidden_layers
C2A_SOURCE_COUNT = 3
KV_SOURCE_COUNT = 4
INDEX_SOURCE_COUNT = 8

BLOCK_SIZE = BLOCK_SIZE
COMPRESSED_CACHE_GROUP = COMPRESSED_CACHE_GROUP
DECODE_MAX_TOKENS = DECODE_MAX_TOKENS
INDEX_CACHE_GROUP = INDEX_CACHE_GROUP
STATE_CAPACITY = STATE_CAPACITY
WINDOW_CACHE_GROUP = WINDOW_CACHE_GROUP

BACKBONE_SCHEDULE = tuple(M.layer_config(layer_id) for layer_id in range(N_LAYERS))
FIXTURE_BATCH = 12
FIXTURE_SEQUENCE = 6
FIXTURE_TOKENS = FIXTURE_BATCH * FIXTURE_SEQUENCE


def cache_source_ordinals(layer_id: int) -> tuple[int | None, int | None, int | None]:
    """Return packed KV, index, and C2A-state source ordinals for one layer."""
    layer = M.layer_config(layer_id)
    if layer.compression_ratio == 0:
        return None, None, None
    kv_sources = M.kv_source_layer_ids
    index_sources = M.index_source_layer_ids
    kv_ordinal = kv_sources.index(layer.kv_source_layer_id)
    index_ordinal = index_sources.index(layer.index_source_layer_id)
    state_ordinal = kv_ordinal if layer.compression_ratio == 2 else None
    return kv_ordinal, index_ordinal, state_ordinal


FWD_ORI_BLOCKS_DYN = pl.dynamic("V41_FWD_ORI_BLOCKS_DYN")
FWD_CMP_BLOCKS_DYN = pl.dynamic("V41_FWD_CMP_BLOCKS_DYN")
FWD_INDEX_BLOCKS_DYN = pl.dynamic("V41_FWD_INDEX_BLOCKS_DYN")
FWD_STATE_BLOCKS_DYN = pl.dynamic("V41_FWD_STATE_BLOCKS_DYN")
FWD_C2A_TABLE_DYN = pl.dynamic("V41_FWD_C2A_TABLE_DYN")
FWD_C1A_TABLE_DYN = pl.dynamic("V41_FWD_C1A_TABLE_DYN")


def _decode_encoder(
    x_hc: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    base_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    base_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c2a_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT * HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    attention_input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    attention_input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    attention_output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    base_rope_cos.bind_dynamic(0, T_DYN)
    base_rope_sin.bind_dynamic(0, T_DYN)
    compressed_token_rope_cos.bind_dynamic(0, T_DYN)
    compressed_token_rope_sin.bind_dynamic(0, T_DYN)
    c2a_rope_cos.bind_dynamic(0, T_DYN)
    c2a_rope_sin.bind_dynamic(0, T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c2a_compressed_lens.bind_dynamic(0, T_DYN)
    c2a_compressed_slots.bind_dynamic(0, T_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    global_tokens = pl.tensor.dim(base_rope_cos, 0)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // KV_SOURCE_COUNT
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // INDEX_SOURCE_COUNT
    state_blocks_per_source = pl.tensor.dim(state_cache_pool, 0) // C2A_SOURCE_COUNT
    x_pong = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    pre_mix_pong = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    gathered = pl.create_tensor([global_tokens, D], dtype=pl.BF16)
    topk_indices = pl.create_tensor([global_tokens, INDEX_TOPK], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    fwd_attention_layer_1_window_cache_l0 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [0 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_1_window_cache_scale_l0 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [0 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_swa_layer_2_hc_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0]
    )
    fwd_swa_layer_2_hc_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [0 * 3])
    fwd_swa_layer_2_hc_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    fwd_swa_layer_2_norm_weight_l0: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [0 * D])
    fwd_swa_layer_2_wq_a_layer_l0: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[0]
    fwd_swa_layer_2_wq_a_scale_layer_l0: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [0 * (D // MX_GROUP), 0])
    )
    fwd_swa_layer_2_q_norm_layer_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [0 * Q_LORA]
    )
    fwd_swa_layer_2_wq_b_layer_l0: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[0]
    fwd_swa_layer_2_wq_b_scale_layer_l0: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [0 * (Q_LORA // MX_GROUP), 0])
    fwd_swa_layer_2_wkv_layer_l0: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[0]
    fwd_swa_layer_2_wkv_scale_layer_l0: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [0 * (D // MX_GROUP), 0])
    )
    fwd_swa_layer_2_kv_norm_layer_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [0 * HEAD_DIM]
    )
    fwd_swa_layer_2_sink_layer_l0: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [0 * LOCAL_H]
    )
    fwd_swa_layer_2_wo_a_layer_l0: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [0 * LOCAL_O_GROUPS, 0, 0]
    )
    fwd_swa_layer_2_wo_b_layer_l0: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[0]
    fwd_swa_layer_2_wo_b_scale_layer_l0: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [0 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    )
    fwd_swa_layer_2_attention_output_l0 = pl.create_tensor([pl.tensor.dim(x_hc, 0), D], dtype=pl.BF16)
    decode_swa_sharded(
        x_hc,
        pre_mix,
        fwd_swa_layer_2_hc_fn_l0,
        fwd_swa_layer_2_hc_scale_l0,
        fwd_swa_layer_2_hc_base_l0,
        fwd_swa_layer_2_norm_weight_l0,
        fwd_swa_layer_2_wq_a_layer_l0,
        fwd_swa_layer_2_wq_a_scale_layer_l0,
        fwd_swa_layer_2_q_norm_layer_l0,
        fwd_swa_layer_2_wq_b_layer_l0,
        fwd_swa_layer_2_wq_b_scale_layer_l0,
        fwd_swa_layer_2_wkv_layer_l0,
        fwd_swa_layer_2_wkv_scale_layer_l0,
        fwd_swa_layer_2_kv_norm_layer_l0,
        fwd_swa_layer_2_sink_layer_l0,
        fwd_swa_layer_2_wo_a_layer_l0,
        fwd_swa_layer_2_wo_b_layer_l0,
        fwd_swa_layer_2_wo_b_scale_layer_l0,
        base_rope_cos,
        base_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_1_window_cache_l0,
        fwd_attention_layer_1_window_cache_scale_l0,
        fwd_swa_layer_2_attention_output_l0,
        attention_hidden,
        attention_pre_mix,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        rank,
        attention_num_tokens,
        0 + 1,
    )
    fwd_moe_layer_8_routed_base_l0 = 0 * N_LOCAL_EXPERTS
    fwd_moe_layer_8_hc_ffn_fn_layer_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0]
    )
    fwd_moe_layer_8_hc_ffn_scale_layer_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [0 * 3])
    fwd_moe_layer_8_hc_ffn_base_layer_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [0 * MIX_HC]
    )
    fwd_moe_layer_8_ffn_norm_weight_layer_l0: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [0 * D]
    )
    fwd_moe_layer_8_gate_weight_layer_l0: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [0 * N_EXPERTS, 0]
    )
    fwd_moe_layer_8_correction_bias_layer_l0: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [0 * N_EXPERTS]
    )
    fwd_moe_layer_8_routed_w1_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l0, 0, 0],
    )
    fwd_moe_layer_8_routed_w1_scale_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [0 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w2_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l0, 0, 0],
    )
    fwd_moe_layer_8_routed_w2_scale_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [0 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w3_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l0, 0, 0],
    )
    fwd_moe_layer_8_routed_w3_scale_layer_l0: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [0 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_shared_w1_layer_l0: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[0]
    fwd_moe_layer_8_shared_w1_scale_layer_l0: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [0 * (D // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w2_layer_l0: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[0]
    fwd_moe_layer_8_shared_w2_scale_layer_l0: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [0 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w3_layer_l0: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[0]
    fwd_moe_layer_8_shared_w3_scale_layer_l0: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [0 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_8_hc_ffn_fn_layer_l0,
        fwd_moe_layer_8_hc_ffn_scale_layer_l0,
        fwd_moe_layer_8_hc_ffn_base_layer_l0,
        fwd_moe_layer_8_ffn_norm_weight_layer_l0,
        fwd_moe_layer_8_gate_weight_layer_l0,
        fwd_moe_layer_8_correction_bias_layer_l0,
        fwd_moe_layer_8_routed_w1_layer_l0,
        fwd_moe_layer_8_routed_w1_scale_layer_l0,
        fwd_moe_layer_8_routed_w2_layer_l0,
        fwd_moe_layer_8_routed_w2_scale_layer_l0,
        fwd_moe_layer_8_routed_w3_layer_l0,
        fwd_moe_layer_8_routed_w3_scale_layer_l0,
        mxfp4_pair_lut,
        fwd_moe_layer_8_shared_w1_layer_l0,
        fwd_moe_layer_8_shared_w1_scale_layer_l0,
        fwd_moe_layer_8_shared_w2_layer_l0,
        fwd_moe_layer_8_shared_w2_scale_layer_l0,
        fwd_moe_layer_8_shared_w3_layer_l0,
        fwd_moe_layer_8_shared_w3_scale_layer_l0,
        pre_mix_pong,
        x_mixed,
        x_pong,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        0 + 1,
    )
    fwd_attention_layer_9_window_cache_l1 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [1 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_9_window_cache_scale_l1 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [1 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_swa_layer_10_hc_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0]
    )
    fwd_swa_layer_10_hc_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [1 * 3])
    fwd_swa_layer_10_hc_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    fwd_swa_layer_10_norm_weight_l1: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [1 * D])
    fwd_swa_layer_10_wq_a_layer_l1: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[1]
    fwd_swa_layer_10_wq_a_scale_layer_l1: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [1 * (D // MX_GROUP), 0])
    )
    fwd_swa_layer_10_q_norm_layer_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [1 * Q_LORA]
    )
    fwd_swa_layer_10_wq_b_layer_l1: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[1]
    fwd_swa_layer_10_wq_b_scale_layer_l1: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [1 * (Q_LORA // MX_GROUP), 0])
    fwd_swa_layer_10_wkv_layer_l1: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[1]
    fwd_swa_layer_10_wkv_scale_layer_l1: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [1 * (D // MX_GROUP), 0])
    )
    fwd_swa_layer_10_kv_norm_layer_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [1 * HEAD_DIM]
    )
    fwd_swa_layer_10_sink_layer_l1: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [1 * LOCAL_H]
    )
    fwd_swa_layer_10_wo_a_layer_l1: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [1 * LOCAL_O_GROUPS, 0, 0]
    )
    fwd_swa_layer_10_wo_b_layer_l1: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[1]
    fwd_swa_layer_10_wo_b_scale_layer_l1: pl.Tensor[
        [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [1 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    fwd_swa_layer_10_attention_output_l1 = pl.create_tensor([pl.tensor.dim(x_pong, 0), D], dtype=pl.BF16)
    decode_swa_sharded(
        x_pong,
        pre_mix_pong,
        fwd_swa_layer_10_hc_fn_l1,
        fwd_swa_layer_10_hc_scale_l1,
        fwd_swa_layer_10_hc_base_l1,
        fwd_swa_layer_10_norm_weight_l1,
        fwd_swa_layer_10_wq_a_layer_l1,
        fwd_swa_layer_10_wq_a_scale_layer_l1,
        fwd_swa_layer_10_q_norm_layer_l1,
        fwd_swa_layer_10_wq_b_layer_l1,
        fwd_swa_layer_10_wq_b_scale_layer_l1,
        fwd_swa_layer_10_wkv_layer_l1,
        fwd_swa_layer_10_wkv_scale_layer_l1,
        fwd_swa_layer_10_kv_norm_layer_l1,
        fwd_swa_layer_10_sink_layer_l1,
        fwd_swa_layer_10_wo_a_layer_l1,
        fwd_swa_layer_10_wo_b_layer_l1,
        fwd_swa_layer_10_wo_b_scale_layer_l1,
        base_rope_cos,
        base_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_9_window_cache_l1,
        fwd_attention_layer_9_window_cache_scale_l1,
        fwd_swa_layer_10_attention_output_l1,
        attention_hidden,
        attention_pre_mix,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        rank,
        attention_num_tokens,
        1 + 1,
    )
    fwd_moe_layer_16_routed_base_l1 = 1 * N_LOCAL_EXPERTS
    fwd_moe_layer_16_hc_ffn_fn_layer_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0]
    )
    fwd_moe_layer_16_hc_ffn_scale_layer_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [1 * 3])
    fwd_moe_layer_16_hc_ffn_base_layer_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [1 * MIX_HC]
    )
    fwd_moe_layer_16_ffn_norm_weight_layer_l1: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [1 * D]
    )
    fwd_moe_layer_16_gate_weight_layer_l1: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [1 * N_EXPERTS, 0]
    )
    fwd_moe_layer_16_correction_bias_layer_l1: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [1 * N_EXPERTS]
    )
    fwd_moe_layer_16_routed_w1_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l1, 0, 0],
    )
    fwd_moe_layer_16_routed_w1_scale_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [1 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w2_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l1, 0, 0],
    )
    fwd_moe_layer_16_routed_w2_scale_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [1 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w3_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l1, 0, 0],
    )
    fwd_moe_layer_16_routed_w3_scale_layer_l1: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [1 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_shared_w1_layer_l1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[1]
    fwd_moe_layer_16_shared_w1_scale_layer_l1: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [1 * (D // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w2_layer_l1: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[1]
    fwd_moe_layer_16_shared_w2_scale_layer_l1: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [1 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w3_layer_l1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[1]
    fwd_moe_layer_16_shared_w3_scale_layer_l1: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [1 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_16_hc_ffn_fn_layer_l1,
        fwd_moe_layer_16_hc_ffn_scale_layer_l1,
        fwd_moe_layer_16_hc_ffn_base_layer_l1,
        fwd_moe_layer_16_ffn_norm_weight_layer_l1,
        fwd_moe_layer_16_gate_weight_layer_l1,
        fwd_moe_layer_16_correction_bias_layer_l1,
        fwd_moe_layer_16_routed_w1_layer_l1,
        fwd_moe_layer_16_routed_w1_scale_layer_l1,
        fwd_moe_layer_16_routed_w2_layer_l1,
        fwd_moe_layer_16_routed_w2_scale_layer_l1,
        fwd_moe_layer_16_routed_w3_layer_l1,
        fwd_moe_layer_16_routed_w3_scale_layer_l1,
        mxfp4_pair_lut,
        fwd_moe_layer_16_shared_w1_layer_l1,
        fwd_moe_layer_16_shared_w1_scale_layer_l1,
        fwd_moe_layer_16_shared_w2_layer_l1,
        fwd_moe_layer_16_shared_w2_scale_layer_l1,
        fwd_moe_layer_16_shared_w3_layer_l1,
        fwd_moe_layer_16_shared_w3_scale_layer_l1,
        pre_mix,
        x_mixed,
        x_hc,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        1 + 1,
    )
    for c2a_block in pl.range(C2A_SOURCE_COUNT):
        fwd_attention_layer_1_window_cache_l2 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 2) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_window_cache_scale_l2 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 2) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_source_id_l2 = c2a_block
        fwd_attention_layer_1_compressed_cache_l2 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_1_source_id_l2 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_compressed_cache_scale_l2 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l2 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_index_cache_l2 = pl.slice(
            index_cache_pool,
            [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2],
            [fwd_attention_layer_1_source_id_l2 * index_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_index_cache_scale_l2 = pl.slice(
            index_cache_scale_pool,
            [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l2 * index_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_state_cache_l2 = pl.slice(
            state_cache_pool,
            [state_blocks_per_source, STATE_CAPACITY, STATE_WIDTH],
            [fwd_attention_layer_1_source_id_l2 * state_blocks_per_source, 0, 0],
        )
        fwd_attention_layer_1_index_wq_b_scale_source_l2: pl.Tensor[
            [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            index_wq_b_scale,
            [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM],
            [c2a_block * (Q_LORA // MX_GROUP), 0],
        )
        fwd_attention_layer_1_c2a_compressor_wkv_source_l2: pl.Tensor[[D, HEAD_DIM], pl.FP32] = (
            c2a_compressor_wkv[fwd_attention_layer_1_source_id_l2]
        )
        fwd_attention_layer_1_c2a_compressor_wgate_source_l2: pl.Tensor[[D, HEAD_DIM], pl.FP32] = (
            c2a_compressor_wgate[fwd_attention_layer_1_source_id_l2]
        )
        fwd_attention_layer_1_compressor_norm_source_l2: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            compressor_norm_weight, [HEAD_DIM], [fwd_attention_layer_1_source_id_l2 * HEAD_DIM]
        )
        fwd_attention_layer_1_index_wk_source_l2: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16] = pl.slice(
            index_wk, [HEAD_DIM, INDEX_DIM], [fwd_attention_layer_1_source_id_l2 * HEAD_DIM, 0]
        )
        fwd_attention_layer_1_index_norm_source_l2: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(
            index_norm_weight, [INDEX_DIM], [fwd_attention_layer_1_source_id_l2 * INDEX_DIM]
        )
        fwd_attention_layer_1_index_wq_b_source_l2: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = (
            index_wq_b[fwd_attention_layer_1_source_id_l2]
        )
        fwd_attention_layer_1_index_weights_proj_source_l2: pl.Tensor[[D, INDEX_H], pl.BF16] = (
            index_weights_proj[fwd_attention_layer_1_source_id_l2]
        )
        fwd_c2a_full_layer_3_hc_fn_l2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 2) * MIX_HC, 0]
        )
        fwd_c2a_full_layer_3_hc_scale_l2: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 2) * 3]
        )
        fwd_c2a_full_layer_3_hc_base_l2: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 2) * MIX_HC]
        )
        fwd_c2a_full_layer_3_norm_weight_l2: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 2) * D]
        )
        fwd_c2a_full_layer_3_wq_a_layer_l2: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 2]
        fwd_c2a_full_layer_3_wq_a_scale_layer_l2: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 2) * (D // MX_GROUP), 0])
        fwd_c2a_full_layer_3_q_norm_layer_l2: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 2) * Q_LORA]
        )
        fwd_c2a_full_layer_3_wq_b_layer_l2: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 2
        ]
        fwd_c2a_full_layer_3_wq_b_scale_layer_l2: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 2) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_full_layer_3_wkv_layer_l2: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 2]
        fwd_c2a_full_layer_3_wkv_scale_layer_l2: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 2) * (D // MX_GROUP), 0])
        fwd_c2a_full_layer_3_kv_norm_layer_l2: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 2) * HEAD_DIM]
        )
        fwd_c2a_full_layer_3_sink_layer_l2: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 2) * LOCAL_H]
        )
        fwd_c2a_full_layer_3_wo_a_layer_l2: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 2) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_full_layer_3_wo_b_layer_l2: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 2
        ]
        fwd_c2a_full_layer_3_wo_b_scale_layer_l2: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 2) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_full_layer_3_attention_output_l2 = pl.create_tensor(
            [pl.tensor.dim(x_hc, 0), D], dtype=pl.BF16
        )
        decode_c2a_full_sharded(
            x_hc,
            pre_mix,
            fwd_c2a_full_layer_3_hc_fn_l2,
            fwd_c2a_full_layer_3_hc_scale_l2,
            fwd_c2a_full_layer_3_hc_base_l2,
            fwd_c2a_full_layer_3_norm_weight_l2,
            fwd_c2a_full_layer_3_wq_a_layer_l2,
            fwd_c2a_full_layer_3_wq_a_scale_layer_l2,
            fwd_c2a_full_layer_3_q_norm_layer_l2,
            fwd_c2a_full_layer_3_wq_b_layer_l2,
            fwd_c2a_full_layer_3_wq_b_scale_layer_l2,
            fwd_c2a_full_layer_3_wkv_layer_l2,
            fwd_c2a_full_layer_3_wkv_scale_layer_l2,
            fwd_c2a_full_layer_3_kv_norm_layer_l2,
            fwd_c2a_full_layer_3_sink_layer_l2,
            fwd_c2a_full_layer_3_wo_a_layer_l2,
            fwd_c2a_full_layer_3_wo_b_layer_l2,
            fwd_c2a_full_layer_3_wo_b_scale_layer_l2,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_1_window_cache_l2,
            fwd_attention_layer_1_window_cache_scale_l2,
            fwd_attention_layer_1_compressed_cache_l2,
            fwd_attention_layer_1_compressed_cache_scale_l2,
            token_to_req_indices,
            c2a_compressed_lens,
            fwd_attention_layer_1_index_cache_l2,
            fwd_attention_layer_1_index_cache_scale_l2,
            c2a_index_block_table,
            position_ids,
            c2a_rope_cos,
            c2a_rope_sin,
            fwd_attention_layer_1_c2a_compressor_wkv_source_l2,
            fwd_attention_layer_1_c2a_compressor_wgate_source_l2,
            query_start_loc,
            state_block_table,
            fwd_attention_layer_1_state_cache_l2,
            fwd_attention_layer_1_compressor_norm_source_l2,
            c2a_compressed_slots,
            fwd_attention_layer_1_index_wk_source_l2,
            fwd_attention_layer_1_index_norm_source_l2,
            fwd_attention_layer_1_index_wq_b_source_l2,
            fwd_attention_layer_1_index_wq_b_scale_source_l2,
            fwd_attention_layer_1_index_weights_proj_source_l2,
            topk_indices,
            fwd_c2a_full_layer_3_attention_output_l2,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 2 + 1,
        )
        fwd_moe_layer_8_routed_base_l2 = (c2a_block * 6 + 2) * N_LOCAL_EXPERTS
        fwd_moe_layer_8_hc_ffn_fn_layer_l2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 2) * MIX_HC, 0]
        )
        fwd_moe_layer_8_hc_ffn_scale_layer_l2: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 2) * 3]
        )
        fwd_moe_layer_8_hc_ffn_base_layer_l2: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 2) * MIX_HC]
        )
        fwd_moe_layer_8_ffn_norm_weight_layer_l2: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 2) * D]
        )
        fwd_moe_layer_8_gate_weight_layer_l2: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 2) * N_EXPERTS, 0]
        )
        fwd_moe_layer_8_correction_bias_layer_l2: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 2) * N_EXPERTS]
        )
        fwd_moe_layer_8_routed_w1_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l2, 0, 0],
        )
        fwd_moe_layer_8_routed_w1_scale_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 2) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w2_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l2, 0, 0],
        )
        fwd_moe_layer_8_routed_w2_scale_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 2) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w3_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l2, 0, 0],
        )
        fwd_moe_layer_8_routed_w3_scale_layer_l2: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 2) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_shared_w1_layer_l2: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 2
        ]
        fwd_moe_layer_8_shared_w1_scale_layer_l2: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 2) * (D // MX_GROUP), 0])
        fwd_moe_layer_8_shared_w2_layer_l2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 2
        ]
        fwd_moe_layer_8_shared_w2_scale_layer_l2: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 2) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_8_shared_w3_layer_l2: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 2
        ]
        fwd_moe_layer_8_shared_w3_scale_layer_l2: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 2) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_8_hc_ffn_fn_layer_l2,
            fwd_moe_layer_8_hc_ffn_scale_layer_l2,
            fwd_moe_layer_8_hc_ffn_base_layer_l2,
            fwd_moe_layer_8_ffn_norm_weight_layer_l2,
            fwd_moe_layer_8_gate_weight_layer_l2,
            fwd_moe_layer_8_correction_bias_layer_l2,
            fwd_moe_layer_8_routed_w1_layer_l2,
            fwd_moe_layer_8_routed_w1_scale_layer_l2,
            fwd_moe_layer_8_routed_w2_layer_l2,
            fwd_moe_layer_8_routed_w2_scale_layer_l2,
            fwd_moe_layer_8_routed_w3_layer_l2,
            fwd_moe_layer_8_routed_w3_scale_layer_l2,
            mxfp4_pair_lut,
            fwd_moe_layer_8_shared_w1_layer_l2,
            fwd_moe_layer_8_shared_w1_scale_layer_l2,
            fwd_moe_layer_8_shared_w2_layer_l2,
            fwd_moe_layer_8_shared_w2_scale_layer_l2,
            fwd_moe_layer_8_shared_w3_layer_l2,
            fwd_moe_layer_8_shared_w3_scale_layer_l2,
            pre_mix_pong,
            x_mixed,
            x_pong,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 2 + 1,
        )
        fwd_attention_layer_9_window_cache_l3 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 3) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_window_cache_scale_l3 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 3) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_source_id_l3 = c2a_block
        fwd_attention_layer_9_compressed_cache_l3 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_9_source_id_l3 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_9_compressed_cache_scale_l3 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_9_source_id_l3 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c2a_reuse_layer_12_hc_fn_l3: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 3) * MIX_HC, 0]
        )
        fwd_c2a_reuse_layer_12_hc_scale_l3: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 3) * 3]
        )
        fwd_c2a_reuse_layer_12_hc_base_l3: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 3) * MIX_HC]
        )
        fwd_c2a_reuse_layer_12_norm_weight_l3: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 3) * D]
        )
        fwd_c2a_reuse_layer_12_wq_a_layer_l3: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 3]
        fwd_c2a_reuse_layer_12_wq_a_scale_layer_l3: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 3) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_q_norm_layer_l3: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 3) * Q_LORA]
        )
        fwd_c2a_reuse_layer_12_wq_b_layer_l3: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 3
        ]
        fwd_c2a_reuse_layer_12_wq_b_scale_layer_l3: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 3) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_reuse_layer_12_wkv_layer_l3: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 3]
        fwd_c2a_reuse_layer_12_wkv_scale_layer_l3: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 3) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_kv_norm_layer_l3: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 3) * HEAD_DIM]
        )
        fwd_c2a_reuse_layer_12_sink_layer_l3: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 3) * LOCAL_H]
        )
        fwd_c2a_reuse_layer_12_wo_a_layer_l3: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 3) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_reuse_layer_12_wo_b_layer_l3: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 3
        ]
        fwd_c2a_reuse_layer_12_wo_b_scale_layer_l3: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 3) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_reuse_layer_12_attention_output_l3 = pl.create_tensor(
            [pl.tensor.dim(x_pong, 0), D], dtype=pl.BF16
        )
        decode_c2a_reuse_sharded(
            x_pong,
            pre_mix_pong,
            fwd_c2a_reuse_layer_12_hc_fn_l3,
            fwd_c2a_reuse_layer_12_hc_scale_l3,
            fwd_c2a_reuse_layer_12_hc_base_l3,
            fwd_c2a_reuse_layer_12_norm_weight_l3,
            fwd_c2a_reuse_layer_12_wq_a_layer_l3,
            fwd_c2a_reuse_layer_12_wq_a_scale_layer_l3,
            fwd_c2a_reuse_layer_12_q_norm_layer_l3,
            fwd_c2a_reuse_layer_12_wq_b_layer_l3,
            fwd_c2a_reuse_layer_12_wq_b_scale_layer_l3,
            fwd_c2a_reuse_layer_12_wkv_layer_l3,
            fwd_c2a_reuse_layer_12_wkv_scale_layer_l3,
            fwd_c2a_reuse_layer_12_kv_norm_layer_l3,
            fwd_c2a_reuse_layer_12_sink_layer_l3,
            fwd_c2a_reuse_layer_12_wo_a_layer_l3,
            fwd_c2a_reuse_layer_12_wo_b_layer_l3,
            fwd_c2a_reuse_layer_12_wo_b_scale_layer_l3,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_9_window_cache_l3,
            fwd_attention_layer_9_window_cache_scale_l3,
            fwd_attention_layer_9_compressed_cache_l3,
            fwd_attention_layer_9_compressed_cache_scale_l3,
            topk_indices,
            fwd_c2a_reuse_layer_12_attention_output_l3,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 3 + 1,
        )
        fwd_moe_layer_16_routed_base_l3 = (c2a_block * 6 + 3) * N_LOCAL_EXPERTS
        fwd_moe_layer_16_hc_ffn_fn_layer_l3: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 3) * MIX_HC, 0]
        )
        fwd_moe_layer_16_hc_ffn_scale_layer_l3: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 3) * 3]
        )
        fwd_moe_layer_16_hc_ffn_base_layer_l3: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 3) * MIX_HC]
        )
        fwd_moe_layer_16_ffn_norm_weight_layer_l3: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 3) * D]
        )
        fwd_moe_layer_16_gate_weight_layer_l3: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 3) * N_EXPERTS, 0]
        )
        fwd_moe_layer_16_correction_bias_layer_l3: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 3) * N_EXPERTS]
        )
        fwd_moe_layer_16_routed_w1_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l3, 0, 0],
        )
        fwd_moe_layer_16_routed_w1_scale_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 3) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w2_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l3, 0, 0],
        )
        fwd_moe_layer_16_routed_w2_scale_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 3) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w3_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l3, 0, 0],
        )
        fwd_moe_layer_16_routed_w3_scale_layer_l3: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 3) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_shared_w1_layer_l3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 3
        ]
        fwd_moe_layer_16_shared_w1_scale_layer_l3: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 3) * (D // MX_GROUP), 0])
        fwd_moe_layer_16_shared_w2_layer_l3: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 3
        ]
        fwd_moe_layer_16_shared_w2_scale_layer_l3: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 3) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_16_shared_w3_layer_l3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 3
        ]
        fwd_moe_layer_16_shared_w3_scale_layer_l3: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 3) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_16_hc_ffn_fn_layer_l3,
            fwd_moe_layer_16_hc_ffn_scale_layer_l3,
            fwd_moe_layer_16_hc_ffn_base_layer_l3,
            fwd_moe_layer_16_ffn_norm_weight_layer_l3,
            fwd_moe_layer_16_gate_weight_layer_l3,
            fwd_moe_layer_16_correction_bias_layer_l3,
            fwd_moe_layer_16_routed_w1_layer_l3,
            fwd_moe_layer_16_routed_w1_scale_layer_l3,
            fwd_moe_layer_16_routed_w2_layer_l3,
            fwd_moe_layer_16_routed_w2_scale_layer_l3,
            fwd_moe_layer_16_routed_w3_layer_l3,
            fwd_moe_layer_16_routed_w3_scale_layer_l3,
            mxfp4_pair_lut,
            fwd_moe_layer_16_shared_w1_layer_l3,
            fwd_moe_layer_16_shared_w1_scale_layer_l3,
            fwd_moe_layer_16_shared_w2_layer_l3,
            fwd_moe_layer_16_shared_w2_scale_layer_l3,
            fwd_moe_layer_16_shared_w3_layer_l3,
            fwd_moe_layer_16_shared_w3_scale_layer_l3,
            pre_mix,
            x_mixed,
            x_hc,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 3 + 1,
        )
        fwd_attention_layer_1_window_cache_l4 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 4) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_window_cache_scale_l4 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 4) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_source_id_l4 = c2a_block
        fwd_attention_layer_1_compressed_cache_l4 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_1_source_id_l4 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_compressed_cache_scale_l4 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l4 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c2a_reuse_layer_4_hc_fn_l4: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 4) * MIX_HC, 0]
        )
        fwd_c2a_reuse_layer_4_hc_scale_l4: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 4) * 3]
        )
        fwd_c2a_reuse_layer_4_hc_base_l4: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 4) * MIX_HC]
        )
        fwd_c2a_reuse_layer_4_norm_weight_l4: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 4) * D]
        )
        fwd_c2a_reuse_layer_4_wq_a_layer_l4: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 4]
        fwd_c2a_reuse_layer_4_wq_a_scale_layer_l4: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 4) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_4_q_norm_layer_l4: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 4) * Q_LORA]
        )
        fwd_c2a_reuse_layer_4_wq_b_layer_l4: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 4
        ]
        fwd_c2a_reuse_layer_4_wq_b_scale_layer_l4: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 4) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_reuse_layer_4_wkv_layer_l4: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 4]
        fwd_c2a_reuse_layer_4_wkv_scale_layer_l4: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 4) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_4_kv_norm_layer_l4: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 4) * HEAD_DIM]
        )
        fwd_c2a_reuse_layer_4_sink_layer_l4: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 4) * LOCAL_H]
        )
        fwd_c2a_reuse_layer_4_wo_a_layer_l4: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 4) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_reuse_layer_4_wo_b_layer_l4: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 4
        ]
        fwd_c2a_reuse_layer_4_wo_b_scale_layer_l4: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 4) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_reuse_layer_4_attention_output_l4 = pl.create_tensor(
            [pl.tensor.dim(x_hc, 0), D], dtype=pl.BF16
        )
        decode_c2a_reuse_sharded(
            x_hc,
            pre_mix,
            fwd_c2a_reuse_layer_4_hc_fn_l4,
            fwd_c2a_reuse_layer_4_hc_scale_l4,
            fwd_c2a_reuse_layer_4_hc_base_l4,
            fwd_c2a_reuse_layer_4_norm_weight_l4,
            fwd_c2a_reuse_layer_4_wq_a_layer_l4,
            fwd_c2a_reuse_layer_4_wq_a_scale_layer_l4,
            fwd_c2a_reuse_layer_4_q_norm_layer_l4,
            fwd_c2a_reuse_layer_4_wq_b_layer_l4,
            fwd_c2a_reuse_layer_4_wq_b_scale_layer_l4,
            fwd_c2a_reuse_layer_4_wkv_layer_l4,
            fwd_c2a_reuse_layer_4_wkv_scale_layer_l4,
            fwd_c2a_reuse_layer_4_kv_norm_layer_l4,
            fwd_c2a_reuse_layer_4_sink_layer_l4,
            fwd_c2a_reuse_layer_4_wo_a_layer_l4,
            fwd_c2a_reuse_layer_4_wo_b_layer_l4,
            fwd_c2a_reuse_layer_4_wo_b_scale_layer_l4,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_1_window_cache_l4,
            fwd_attention_layer_1_window_cache_scale_l4,
            fwd_attention_layer_1_compressed_cache_l4,
            fwd_attention_layer_1_compressed_cache_scale_l4,
            topk_indices,
            fwd_c2a_reuse_layer_4_attention_output_l4,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 4 + 1,
        )
        fwd_moe_layer_8_routed_base_l4 = (c2a_block * 6 + 4) * N_LOCAL_EXPERTS
        fwd_moe_layer_8_hc_ffn_fn_layer_l4: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 4) * MIX_HC, 0]
        )
        fwd_moe_layer_8_hc_ffn_scale_layer_l4: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 4) * 3]
        )
        fwd_moe_layer_8_hc_ffn_base_layer_l4: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 4) * MIX_HC]
        )
        fwd_moe_layer_8_ffn_norm_weight_layer_l4: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 4) * D]
        )
        fwd_moe_layer_8_gate_weight_layer_l4: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 4) * N_EXPERTS, 0]
        )
        fwd_moe_layer_8_correction_bias_layer_l4: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 4) * N_EXPERTS]
        )
        fwd_moe_layer_8_routed_w1_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l4, 0, 0],
        )
        fwd_moe_layer_8_routed_w1_scale_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 4) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w2_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l4, 0, 0],
        )
        fwd_moe_layer_8_routed_w2_scale_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 4) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w3_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l4, 0, 0],
        )
        fwd_moe_layer_8_routed_w3_scale_layer_l4: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 4) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_shared_w1_layer_l4: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 4
        ]
        fwd_moe_layer_8_shared_w1_scale_layer_l4: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 4) * (D // MX_GROUP), 0])
        fwd_moe_layer_8_shared_w2_layer_l4: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 4
        ]
        fwd_moe_layer_8_shared_w2_scale_layer_l4: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 4) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_8_shared_w3_layer_l4: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 4
        ]
        fwd_moe_layer_8_shared_w3_scale_layer_l4: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 4) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_8_hc_ffn_fn_layer_l4,
            fwd_moe_layer_8_hc_ffn_scale_layer_l4,
            fwd_moe_layer_8_hc_ffn_base_layer_l4,
            fwd_moe_layer_8_ffn_norm_weight_layer_l4,
            fwd_moe_layer_8_gate_weight_layer_l4,
            fwd_moe_layer_8_correction_bias_layer_l4,
            fwd_moe_layer_8_routed_w1_layer_l4,
            fwd_moe_layer_8_routed_w1_scale_layer_l4,
            fwd_moe_layer_8_routed_w2_layer_l4,
            fwd_moe_layer_8_routed_w2_scale_layer_l4,
            fwd_moe_layer_8_routed_w3_layer_l4,
            fwd_moe_layer_8_routed_w3_scale_layer_l4,
            mxfp4_pair_lut,
            fwd_moe_layer_8_shared_w1_layer_l4,
            fwd_moe_layer_8_shared_w1_scale_layer_l4,
            fwd_moe_layer_8_shared_w2_layer_l4,
            fwd_moe_layer_8_shared_w2_scale_layer_l4,
            fwd_moe_layer_8_shared_w3_layer_l4,
            fwd_moe_layer_8_shared_w3_scale_layer_l4,
            pre_mix_pong,
            x_mixed,
            x_pong,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 4 + 1,
        )
        fwd_attention_layer_9_window_cache_l5 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 5) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_window_cache_scale_l5 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 5) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_source_id_l5 = c2a_block
        fwd_attention_layer_9_compressed_cache_l5 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_9_source_id_l5 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_9_compressed_cache_scale_l5 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_9_source_id_l5 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c2a_reuse_layer_12_hc_fn_l5: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 5) * MIX_HC, 0]
        )
        fwd_c2a_reuse_layer_12_hc_scale_l5: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 5) * 3]
        )
        fwd_c2a_reuse_layer_12_hc_base_l5: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 5) * MIX_HC]
        )
        fwd_c2a_reuse_layer_12_norm_weight_l5: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 5) * D]
        )
        fwd_c2a_reuse_layer_12_wq_a_layer_l5: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 5]
        fwd_c2a_reuse_layer_12_wq_a_scale_layer_l5: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 5) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_q_norm_layer_l5: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 5) * Q_LORA]
        )
        fwd_c2a_reuse_layer_12_wq_b_layer_l5: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 5
        ]
        fwd_c2a_reuse_layer_12_wq_b_scale_layer_l5: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 5) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_reuse_layer_12_wkv_layer_l5: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 5]
        fwd_c2a_reuse_layer_12_wkv_scale_layer_l5: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 5) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_kv_norm_layer_l5: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 5) * HEAD_DIM]
        )
        fwd_c2a_reuse_layer_12_sink_layer_l5: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 5) * LOCAL_H]
        )
        fwd_c2a_reuse_layer_12_wo_a_layer_l5: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 5) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_reuse_layer_12_wo_b_layer_l5: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 5
        ]
        fwd_c2a_reuse_layer_12_wo_b_scale_layer_l5: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 5) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_reuse_layer_12_attention_output_l5 = pl.create_tensor(
            [pl.tensor.dim(x_pong, 0), D], dtype=pl.BF16
        )
        decode_c2a_reuse_sharded(
            x_pong,
            pre_mix_pong,
            fwd_c2a_reuse_layer_12_hc_fn_l5,
            fwd_c2a_reuse_layer_12_hc_scale_l5,
            fwd_c2a_reuse_layer_12_hc_base_l5,
            fwd_c2a_reuse_layer_12_norm_weight_l5,
            fwd_c2a_reuse_layer_12_wq_a_layer_l5,
            fwd_c2a_reuse_layer_12_wq_a_scale_layer_l5,
            fwd_c2a_reuse_layer_12_q_norm_layer_l5,
            fwd_c2a_reuse_layer_12_wq_b_layer_l5,
            fwd_c2a_reuse_layer_12_wq_b_scale_layer_l5,
            fwd_c2a_reuse_layer_12_wkv_layer_l5,
            fwd_c2a_reuse_layer_12_wkv_scale_layer_l5,
            fwd_c2a_reuse_layer_12_kv_norm_layer_l5,
            fwd_c2a_reuse_layer_12_sink_layer_l5,
            fwd_c2a_reuse_layer_12_wo_a_layer_l5,
            fwd_c2a_reuse_layer_12_wo_b_layer_l5,
            fwd_c2a_reuse_layer_12_wo_b_scale_layer_l5,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_9_window_cache_l5,
            fwd_attention_layer_9_window_cache_scale_l5,
            fwd_attention_layer_9_compressed_cache_l5,
            fwd_attention_layer_9_compressed_cache_scale_l5,
            topk_indices,
            fwd_c2a_reuse_layer_12_attention_output_l5,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 5 + 1,
        )
        fwd_moe_layer_16_routed_base_l5 = (c2a_block * 6 + 5) * N_LOCAL_EXPERTS
        fwd_moe_layer_16_hc_ffn_fn_layer_l5: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 5) * MIX_HC, 0]
        )
        fwd_moe_layer_16_hc_ffn_scale_layer_l5: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 5) * 3]
        )
        fwd_moe_layer_16_hc_ffn_base_layer_l5: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 5) * MIX_HC]
        )
        fwd_moe_layer_16_ffn_norm_weight_layer_l5: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 5) * D]
        )
        fwd_moe_layer_16_gate_weight_layer_l5: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 5) * N_EXPERTS, 0]
        )
        fwd_moe_layer_16_correction_bias_layer_l5: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 5) * N_EXPERTS]
        )
        fwd_moe_layer_16_routed_w1_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l5, 0, 0],
        )
        fwd_moe_layer_16_routed_w1_scale_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 5) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w2_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l5, 0, 0],
        )
        fwd_moe_layer_16_routed_w2_scale_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 5) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w3_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l5, 0, 0],
        )
        fwd_moe_layer_16_routed_w3_scale_layer_l5: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 5) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_shared_w1_layer_l5: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 5
        ]
        fwd_moe_layer_16_shared_w1_scale_layer_l5: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 5) * (D // MX_GROUP), 0])
        fwd_moe_layer_16_shared_w2_layer_l5: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 5
        ]
        fwd_moe_layer_16_shared_w2_scale_layer_l5: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 5) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_16_shared_w3_layer_l5: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 5
        ]
        fwd_moe_layer_16_shared_w3_scale_layer_l5: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 5) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_16_hc_ffn_fn_layer_l5,
            fwd_moe_layer_16_hc_ffn_scale_layer_l5,
            fwd_moe_layer_16_hc_ffn_base_layer_l5,
            fwd_moe_layer_16_ffn_norm_weight_layer_l5,
            fwd_moe_layer_16_gate_weight_layer_l5,
            fwd_moe_layer_16_correction_bias_layer_l5,
            fwd_moe_layer_16_routed_w1_layer_l5,
            fwd_moe_layer_16_routed_w1_scale_layer_l5,
            fwd_moe_layer_16_routed_w2_layer_l5,
            fwd_moe_layer_16_routed_w2_scale_layer_l5,
            fwd_moe_layer_16_routed_w3_layer_l5,
            fwd_moe_layer_16_routed_w3_scale_layer_l5,
            mxfp4_pair_lut,
            fwd_moe_layer_16_shared_w1_layer_l5,
            fwd_moe_layer_16_shared_w1_scale_layer_l5,
            fwd_moe_layer_16_shared_w2_layer_l5,
            fwd_moe_layer_16_shared_w2_scale_layer_l5,
            fwd_moe_layer_16_shared_w3_layer_l5,
            fwd_moe_layer_16_shared_w3_scale_layer_l5,
            pre_mix,
            x_mixed,
            x_hc,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 5 + 1,
        )
        fwd_attention_layer_1_window_cache_l6 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 6) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_window_cache_scale_l6 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 6) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_source_id_l6 = c2a_block
        fwd_attention_layer_1_compressed_cache_l6 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_1_source_id_l6 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_compressed_cache_scale_l6 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l6 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c2a_reuse_layer_4_hc_fn_l6: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 6) * MIX_HC, 0]
        )
        fwd_c2a_reuse_layer_4_hc_scale_l6: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 6) * 3]
        )
        fwd_c2a_reuse_layer_4_hc_base_l6: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 6) * MIX_HC]
        )
        fwd_c2a_reuse_layer_4_norm_weight_l6: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 6) * D]
        )
        fwd_c2a_reuse_layer_4_wq_a_layer_l6: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 6]
        fwd_c2a_reuse_layer_4_wq_a_scale_layer_l6: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 6) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_4_q_norm_layer_l6: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 6) * Q_LORA]
        )
        fwd_c2a_reuse_layer_4_wq_b_layer_l6: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 6
        ]
        fwd_c2a_reuse_layer_4_wq_b_scale_layer_l6: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 6) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_reuse_layer_4_wkv_layer_l6: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 6]
        fwd_c2a_reuse_layer_4_wkv_scale_layer_l6: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 6) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_4_kv_norm_layer_l6: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 6) * HEAD_DIM]
        )
        fwd_c2a_reuse_layer_4_sink_layer_l6: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 6) * LOCAL_H]
        )
        fwd_c2a_reuse_layer_4_wo_a_layer_l6: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 6) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_reuse_layer_4_wo_b_layer_l6: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 6
        ]
        fwd_c2a_reuse_layer_4_wo_b_scale_layer_l6: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 6) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_reuse_layer_4_attention_output_l6 = pl.create_tensor(
            [pl.tensor.dim(x_hc, 0), D], dtype=pl.BF16
        )
        decode_c2a_reuse_sharded(
            x_hc,
            pre_mix,
            fwd_c2a_reuse_layer_4_hc_fn_l6,
            fwd_c2a_reuse_layer_4_hc_scale_l6,
            fwd_c2a_reuse_layer_4_hc_base_l6,
            fwd_c2a_reuse_layer_4_norm_weight_l6,
            fwd_c2a_reuse_layer_4_wq_a_layer_l6,
            fwd_c2a_reuse_layer_4_wq_a_scale_layer_l6,
            fwd_c2a_reuse_layer_4_q_norm_layer_l6,
            fwd_c2a_reuse_layer_4_wq_b_layer_l6,
            fwd_c2a_reuse_layer_4_wq_b_scale_layer_l6,
            fwd_c2a_reuse_layer_4_wkv_layer_l6,
            fwd_c2a_reuse_layer_4_wkv_scale_layer_l6,
            fwd_c2a_reuse_layer_4_kv_norm_layer_l6,
            fwd_c2a_reuse_layer_4_sink_layer_l6,
            fwd_c2a_reuse_layer_4_wo_a_layer_l6,
            fwd_c2a_reuse_layer_4_wo_b_layer_l6,
            fwd_c2a_reuse_layer_4_wo_b_scale_layer_l6,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_1_window_cache_l6,
            fwd_attention_layer_1_window_cache_scale_l6,
            fwd_attention_layer_1_compressed_cache_l6,
            fwd_attention_layer_1_compressed_cache_scale_l6,
            topk_indices,
            fwd_c2a_reuse_layer_4_attention_output_l6,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 6 + 1,
        )
        fwd_moe_layer_8_routed_base_l6 = (c2a_block * 6 + 6) * N_LOCAL_EXPERTS
        fwd_moe_layer_8_hc_ffn_fn_layer_l6: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 6) * MIX_HC, 0]
        )
        fwd_moe_layer_8_hc_ffn_scale_layer_l6: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 6) * 3]
        )
        fwd_moe_layer_8_hc_ffn_base_layer_l6: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 6) * MIX_HC]
        )
        fwd_moe_layer_8_ffn_norm_weight_layer_l6: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 6) * D]
        )
        fwd_moe_layer_8_gate_weight_layer_l6: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 6) * N_EXPERTS, 0]
        )
        fwd_moe_layer_8_correction_bias_layer_l6: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 6) * N_EXPERTS]
        )
        fwd_moe_layer_8_routed_w1_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l6, 0, 0],
        )
        fwd_moe_layer_8_routed_w1_scale_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 6) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w2_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l6, 0, 0],
        )
        fwd_moe_layer_8_routed_w2_scale_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 6) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w3_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l6, 0, 0],
        )
        fwd_moe_layer_8_routed_w3_scale_layer_l6: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 6) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_shared_w1_layer_l6: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 6
        ]
        fwd_moe_layer_8_shared_w1_scale_layer_l6: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 6) * (D // MX_GROUP), 0])
        fwd_moe_layer_8_shared_w2_layer_l6: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 6
        ]
        fwd_moe_layer_8_shared_w2_scale_layer_l6: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 6) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_8_shared_w3_layer_l6: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 6
        ]
        fwd_moe_layer_8_shared_w3_scale_layer_l6: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 6) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_8_hc_ffn_fn_layer_l6,
            fwd_moe_layer_8_hc_ffn_scale_layer_l6,
            fwd_moe_layer_8_hc_ffn_base_layer_l6,
            fwd_moe_layer_8_ffn_norm_weight_layer_l6,
            fwd_moe_layer_8_gate_weight_layer_l6,
            fwd_moe_layer_8_correction_bias_layer_l6,
            fwd_moe_layer_8_routed_w1_layer_l6,
            fwd_moe_layer_8_routed_w1_scale_layer_l6,
            fwd_moe_layer_8_routed_w2_layer_l6,
            fwd_moe_layer_8_routed_w2_scale_layer_l6,
            fwd_moe_layer_8_routed_w3_layer_l6,
            fwd_moe_layer_8_routed_w3_scale_layer_l6,
            mxfp4_pair_lut,
            fwd_moe_layer_8_shared_w1_layer_l6,
            fwd_moe_layer_8_shared_w1_scale_layer_l6,
            fwd_moe_layer_8_shared_w2_layer_l6,
            fwd_moe_layer_8_shared_w2_scale_layer_l6,
            fwd_moe_layer_8_shared_w3_layer_l6,
            fwd_moe_layer_8_shared_w3_scale_layer_l6,
            pre_mix_pong,
            x_mixed,
            x_pong,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 6 + 1,
        )
        fwd_attention_layer_9_window_cache_l7 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c2a_block * 6 + 7) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_window_cache_scale_l7 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c2a_block * 6 + 7) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_source_id_l7 = c2a_block
        fwd_attention_layer_9_compressed_cache_l7 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_9_source_id_l7 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_9_compressed_cache_scale_l7 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_9_source_id_l7 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c2a_reuse_layer_12_hc_fn_l7: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 7) * MIX_HC, 0]
        )
        fwd_c2a_reuse_layer_12_hc_scale_l7: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c2a_block * 6 + 7) * 3]
        )
        fwd_c2a_reuse_layer_12_hc_base_l7: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c2a_block * 6 + 7) * MIX_HC]
        )
        fwd_c2a_reuse_layer_12_norm_weight_l7: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c2a_block * 6 + 7) * D]
        )
        fwd_c2a_reuse_layer_12_wq_a_layer_l7: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c2a_block * 6 + 7]
        fwd_c2a_reuse_layer_12_wq_a_scale_layer_l7: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c2a_block * 6 + 7) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_q_norm_layer_l7: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c2a_block * 6 + 7) * Q_LORA]
        )
        fwd_c2a_reuse_layer_12_wq_b_layer_l7: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c2a_block * 6 + 7
        ]
        fwd_c2a_reuse_layer_12_wq_b_scale_layer_l7: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c2a_block * 6 + 7) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c2a_reuse_layer_12_wkv_layer_l7: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c2a_block * 6 + 7]
        fwd_c2a_reuse_layer_12_wkv_scale_layer_l7: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c2a_block * 6 + 7) * (D // MX_GROUP), 0])
        fwd_c2a_reuse_layer_12_kv_norm_layer_l7: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c2a_block * 6 + 7) * HEAD_DIM]
        )
        fwd_c2a_reuse_layer_12_sink_layer_l7: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c2a_block * 6 + 7) * LOCAL_H]
        )
        fwd_c2a_reuse_layer_12_wo_a_layer_l7: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c2a_block * 6 + 7) * LOCAL_O_GROUPS, 0, 0])
        )
        fwd_c2a_reuse_layer_12_wo_b_layer_l7: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c2a_block * 6 + 7
        ]
        fwd_c2a_reuse_layer_12_wo_b_scale_layer_l7: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [(c2a_block * 6 + 7) * (LOCAL_O_WIDTH // MX_GROUP), 0]
        )
        fwd_c2a_reuse_layer_12_attention_output_l7 = pl.create_tensor(
            [pl.tensor.dim(x_pong, 0), D], dtype=pl.BF16
        )
        decode_c2a_reuse_sharded(
            x_pong,
            pre_mix_pong,
            fwd_c2a_reuse_layer_12_hc_fn_l7,
            fwd_c2a_reuse_layer_12_hc_scale_l7,
            fwd_c2a_reuse_layer_12_hc_base_l7,
            fwd_c2a_reuse_layer_12_norm_weight_l7,
            fwd_c2a_reuse_layer_12_wq_a_layer_l7,
            fwd_c2a_reuse_layer_12_wq_a_scale_layer_l7,
            fwd_c2a_reuse_layer_12_q_norm_layer_l7,
            fwd_c2a_reuse_layer_12_wq_b_layer_l7,
            fwd_c2a_reuse_layer_12_wq_b_scale_layer_l7,
            fwd_c2a_reuse_layer_12_wkv_layer_l7,
            fwd_c2a_reuse_layer_12_wkv_scale_layer_l7,
            fwd_c2a_reuse_layer_12_kv_norm_layer_l7,
            fwd_c2a_reuse_layer_12_sink_layer_l7,
            fwd_c2a_reuse_layer_12_wo_a_layer_l7,
            fwd_c2a_reuse_layer_12_wo_b_layer_l7,
            fwd_c2a_reuse_layer_12_wo_b_scale_layer_l7,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_9_window_cache_l7,
            fwd_attention_layer_9_window_cache_scale_l7,
            fwd_attention_layer_9_compressed_cache_l7,
            fwd_attention_layer_9_compressed_cache_scale_l7,
            topk_indices,
            fwd_c2a_reuse_layer_12_attention_output_l7,
            attention_hidden,
            attention_pre_mix,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            rank,
            attention_num_tokens,
            c2a_block * 6 + 7 + 1,
        )
        fwd_moe_layer_16_routed_base_l7 = (c2a_block * 6 + 7) * N_LOCAL_EXPERTS
        fwd_moe_layer_16_hc_ffn_fn_layer_l7: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c2a_block * 6 + 7) * MIX_HC, 0]
        )
        fwd_moe_layer_16_hc_ffn_scale_layer_l7: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c2a_block * 6 + 7) * 3]
        )
        fwd_moe_layer_16_hc_ffn_base_layer_l7: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c2a_block * 6 + 7) * MIX_HC]
        )
        fwd_moe_layer_16_ffn_norm_weight_layer_l7: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c2a_block * 6 + 7) * D]
        )
        fwd_moe_layer_16_gate_weight_layer_l7: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c2a_block * 6 + 7) * N_EXPERTS, 0]
        )
        fwd_moe_layer_16_correction_bias_layer_l7: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c2a_block * 6 + 7) * N_EXPERTS]
        )
        fwd_moe_layer_16_routed_w1_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l7, 0, 0],
        )
        fwd_moe_layer_16_routed_w1_scale_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 7) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w2_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l7, 0, 0],
        )
        fwd_moe_layer_16_routed_w2_scale_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c2a_block * 6 + 7) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w3_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l7, 0, 0],
        )
        fwd_moe_layer_16_routed_w3_scale_layer_l7: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c2a_block * 6 + 7) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_shared_w1_layer_l7: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c2a_block * 6 + 7
        ]
        fwd_moe_layer_16_shared_w1_scale_layer_l7: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 7) * (D // MX_GROUP), 0])
        fwd_moe_layer_16_shared_w2_layer_l7: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c2a_block * 6 + 7
        ]
        fwd_moe_layer_16_shared_w2_scale_layer_l7: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c2a_block * 6 + 7) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_16_shared_w3_layer_l7: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c2a_block * 6 + 7
        ]
        fwd_moe_layer_16_shared_w3_scale_layer_l7: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c2a_block * 6 + 7) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_16_hc_ffn_fn_layer_l7,
            fwd_moe_layer_16_hc_ffn_scale_layer_l7,
            fwd_moe_layer_16_hc_ffn_base_layer_l7,
            fwd_moe_layer_16_ffn_norm_weight_layer_l7,
            fwd_moe_layer_16_gate_weight_layer_l7,
            fwd_moe_layer_16_correction_bias_layer_l7,
            fwd_moe_layer_16_routed_w1_layer_l7,
            fwd_moe_layer_16_routed_w1_scale_layer_l7,
            fwd_moe_layer_16_routed_w2_layer_l7,
            fwd_moe_layer_16_routed_w2_scale_layer_l7,
            fwd_moe_layer_16_routed_w3_layer_l7,
            fwd_moe_layer_16_routed_w3_scale_layer_l7,
            mxfp4_pair_lut,
            fwd_moe_layer_16_shared_w1_layer_l7,
            fwd_moe_layer_16_shared_w1_scale_layer_l7,
            fwd_moe_layer_16_shared_w2_layer_l7,
            fwd_moe_layer_16_shared_w2_scale_layer_l7,
            fwd_moe_layer_16_shared_w3_layer_l7,
            fwd_moe_layer_16_shared_w3_scale_layer_l7,
            pre_mix,
            x_mixed,
            x_hc,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c2a_block * 6 + 7 + 1,
        )
    return x_hc


def _decode_decoder(
    x_hc: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    base_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    base_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT * HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    attention_input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    attention_input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    attention_output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    base_rope_cos.bind_dynamic(0, T_DYN)
    base_rope_sin.bind_dynamic(0, T_DYN)
    compressed_token_rope_cos.bind_dynamic(0, T_DYN)
    compressed_token_rope_sin.bind_dynamic(0, T_DYN)
    c2a_rope_cos.bind_dynamic(0, T_DYN)
    c2a_rope_sin.bind_dynamic(0, T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c1a_compressed_lens.bind_dynamic(0, T_DYN)
    c1a_compressed_slots.bind_dynamic(0, T_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    global_tokens = pl.tensor.dim(base_rope_cos, 0)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // KV_SOURCE_COUNT
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // INDEX_SOURCE_COUNT
    x_pong = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    pre_mix_pong = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    gathered = pl.create_tensor([global_tokens, D], dtype=pl.BF16)
    topk_indices = pl.create_tensor([global_tokens, INDEX_TOPK], dtype=pl.INT32)
    # Sorted eight-token blocks shared by the four Decode Reindex layers.
    pool_blocks_raw = pl.tensor.dim(c1a_index_block_table, 1) * (BLOCK_SIZE // CANDIDATE_BLOCK_SIZE)
    pool_block_capacity = pl.min(CANDIDATE_TOPK_BLOCKS, pl.max(32, (pool_blocks_raw + 31) // 32 * 32))
    pool_block_ids = pl.create_tensor([global_tokens, pool_block_capacity], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([global_tokens], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    fwd_attention_layer_1_window_cache_l20 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [20 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_1_window_cache_scale_l20 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [20 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_1_source_id_l20 = 3
    fwd_attention_layer_1_index_source_id_l20 = 3
    fwd_attention_layer_1_compressed_cache_l20 = pl.slice(
        compressed_cache_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
        [fwd_attention_layer_1_source_id_l20 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_1_compressed_cache_scale_l20 = pl.slice(
        compressed_cache_scale_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        [fwd_attention_layer_1_source_id_l20 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_1_index_cache_l20 = pl.slice(
        index_cache_pool,
        [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2],
        [fwd_attention_layer_1_index_source_id_l20 * index_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_1_index_cache_scale_l20 = pl.slice(
        index_cache_scale_pool,
        [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP],
        [fwd_attention_layer_1_index_source_id_l20 * index_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_1_index_wq_b_scale_source_l20: pl.Tensor[
        [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        index_wq_b_scale,
        [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM],
        [3 * (Q_LORA // MX_GROUP), 0],
    )
    fwd_attention_layer_1_compressor_norm_source_l20: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        compressor_norm_weight, [HEAD_DIM], [fwd_attention_layer_1_source_id_l20 * HEAD_DIM]
    )
    fwd_attention_layer_1_index_wk_source_l20: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16] = pl.slice(
        index_wk, [HEAD_DIM, INDEX_DIM], [fwd_attention_layer_1_index_source_id_l20 * HEAD_DIM, 0]
    )
    fwd_attention_layer_1_index_norm_source_l20: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(
        index_norm_weight, [INDEX_DIM], [fwd_attention_layer_1_index_source_id_l20 * INDEX_DIM]
    )
    fwd_attention_layer_1_index_wq_b_source_l20: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = (
        index_wq_b[fwd_attention_layer_1_index_source_id_l20]
    )
    fwd_attention_layer_1_index_weights_proj_source_l20: pl.Tensor[[D, INDEX_H], pl.BF16] = (
        index_weights_proj[fwd_attention_layer_1_index_source_id_l20]
    )
    fwd_c1a_full_layer_5_hc_fn_l20: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [20 * MIX_HC, 0]
    )
    fwd_c1a_full_layer_5_hc_scale_l20: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [20 * 3])
    fwd_c1a_full_layer_5_hc_base_l20: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [20 * MIX_HC]
    )
    fwd_c1a_full_layer_5_norm_weight_l20: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [20 * D])
    fwd_c1a_full_layer_5_wq_a_layer_l20: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[20]
    fwd_c1a_full_layer_5_wq_a_scale_layer_l20: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [20 * (D // MX_GROUP), 0])
    )
    fwd_c1a_full_layer_5_q_norm_layer_l20: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [20 * Q_LORA]
    )
    fwd_c1a_full_layer_5_wq_b_layer_l20: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[20]
    fwd_c1a_full_layer_5_wq_b_scale_layer_l20: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [20 * (Q_LORA // MX_GROUP), 0])
    fwd_c1a_full_layer_5_wkv_layer_l20: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[20]
    fwd_c1a_full_layer_5_wkv_scale_layer_l20: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [20 * (D // MX_GROUP), 0])
    )
    fwd_c1a_full_layer_5_kv_norm_layer_l20: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [20 * HEAD_DIM]
    )
    fwd_c1a_full_layer_5_sink_layer_l20: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [20 * LOCAL_H]
    )
    fwd_c1a_full_layer_5_wo_a_layer_l20: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [20 * LOCAL_O_GROUPS, 0, 0]
    )
    fwd_c1a_full_layer_5_wo_b_layer_l20: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[20]
    fwd_c1a_full_layer_5_wo_b_scale_layer_l20: pl.Tensor[
        [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [20 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    decode_c1a_full_sharded(
        x_hc,
        pre_mix,
        fwd_c1a_full_layer_5_hc_fn_l20,
        fwd_c1a_full_layer_5_hc_scale_l20,
        fwd_c1a_full_layer_5_hc_base_l20,
        fwd_c1a_full_layer_5_norm_weight_l20,
        fwd_c1a_full_layer_5_wq_a_layer_l20,
        fwd_c1a_full_layer_5_wq_a_scale_layer_l20,
        fwd_c1a_full_layer_5_q_norm_layer_l20,
        fwd_c1a_full_layer_5_wq_b_layer_l20,
        fwd_c1a_full_layer_5_wq_b_scale_layer_l20,
        fwd_c1a_full_layer_5_wkv_layer_l20,
        fwd_c1a_full_layer_5_wkv_scale_layer_l20,
        fwd_c1a_full_layer_5_kv_norm_layer_l20,
        fwd_c1a_full_layer_5_sink_layer_l20,
        fwd_c1a_full_layer_5_wo_a_layer_l20,
        fwd_c1a_full_layer_5_wo_b_layer_l20,
        fwd_c1a_full_layer_5_wo_b_scale_layer_l20,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_1_window_cache_l20,
        fwd_attention_layer_1_window_cache_scale_l20,
        fwd_attention_layer_1_compressed_cache_l20,
        fwd_attention_layer_1_compressed_cache_scale_l20,
        request_ids,
        c1a_compressed_lens,
        fwd_attention_layer_1_index_cache_l20,
        fwd_attention_layer_1_index_cache_scale_l20,
        c1a_index_block_table,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        c1a_compressor_wkv,
        fwd_attention_layer_1_compressor_norm_source_l20,
        c1a_compressed_slots,
        fwd_attention_layer_1_index_wk_source_l20,
        fwd_attention_layer_1_index_norm_source_l20,
        fwd_attention_layer_1_index_wq_b_source_l20,
        fwd_attention_layer_1_index_wq_b_scale_source_l20,
        fwd_attention_layer_1_index_weights_proj_source_l20,
        topk_indices,
        pool_block_ids,
        pool_block_counts,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        attention_hidden,
        attention_pre_mix,
        rank // TP_SIZE * TP_SIZE,
        rank % TP_SIZE,
        attention_num_tokens,
        20 + 1,
    )
    fwd_moe_layer_8_routed_base_l20 = 20 * N_LOCAL_EXPERTS
    fwd_moe_layer_8_hc_ffn_fn_layer_l20: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [20 * MIX_HC, 0]
    )
    fwd_moe_layer_8_hc_ffn_scale_layer_l20: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [20 * 3])
    fwd_moe_layer_8_hc_ffn_base_layer_l20: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [20 * MIX_HC]
    )
    fwd_moe_layer_8_ffn_norm_weight_layer_l20: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [20 * D]
    )
    fwd_moe_layer_8_gate_weight_layer_l20: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [20 * N_EXPERTS, 0]
    )
    fwd_moe_layer_8_correction_bias_layer_l20: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [20 * N_EXPERTS]
    )
    fwd_moe_layer_8_routed_w1_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l20, 0, 0],
    )
    fwd_moe_layer_8_routed_w1_scale_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [20 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w2_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l20, 0, 0],
    )
    fwd_moe_layer_8_routed_w2_scale_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [20 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w3_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l20, 0, 0],
    )
    fwd_moe_layer_8_routed_w3_scale_layer_l20: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [20 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_shared_w1_layer_l20: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[20]
    fwd_moe_layer_8_shared_w1_scale_layer_l20: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [20 * (D // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w2_layer_l20: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[20]
    fwd_moe_layer_8_shared_w2_scale_layer_l20: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [20 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w3_layer_l20: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[20]
    fwd_moe_layer_8_shared_w3_scale_layer_l20: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [20 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_8_hc_ffn_fn_layer_l20,
        fwd_moe_layer_8_hc_ffn_scale_layer_l20,
        fwd_moe_layer_8_hc_ffn_base_layer_l20,
        fwd_moe_layer_8_ffn_norm_weight_layer_l20,
        fwd_moe_layer_8_gate_weight_layer_l20,
        fwd_moe_layer_8_correction_bias_layer_l20,
        fwd_moe_layer_8_routed_w1_layer_l20,
        fwd_moe_layer_8_routed_w1_scale_layer_l20,
        fwd_moe_layer_8_routed_w2_layer_l20,
        fwd_moe_layer_8_routed_w2_scale_layer_l20,
        fwd_moe_layer_8_routed_w3_layer_l20,
        fwd_moe_layer_8_routed_w3_scale_layer_l20,
        mxfp4_pair_lut,
        fwd_moe_layer_8_shared_w1_layer_l20,
        fwd_moe_layer_8_shared_w1_scale_layer_l20,
        fwd_moe_layer_8_shared_w2_layer_l20,
        fwd_moe_layer_8_shared_w2_scale_layer_l20,
        fwd_moe_layer_8_shared_w3_layer_l20,
        fwd_moe_layer_8_shared_w3_scale_layer_l20,
        pre_mix_pong,
        x_mixed,
        x_pong,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        20 + 1,
    )
    fwd_attention_layer_9_window_cache_l21 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [21 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_9_window_cache_scale_l21 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [21 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_9_source_id_l21 = 3
    fwd_attention_layer_9_compressed_cache_l21 = pl.slice(
        compressed_cache_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
        [fwd_attention_layer_9_source_id_l21 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_9_compressed_cache_scale_l21 = pl.slice(
        compressed_cache_scale_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        [fwd_attention_layer_9_source_id_l21 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_c1a_reuse_layer_15_hc_fn_l21: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [21 * MIX_HC, 0]
    )
    fwd_c1a_reuse_layer_15_hc_scale_l21: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [21 * 3])
    fwd_c1a_reuse_layer_15_hc_base_l21: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [21 * MIX_HC]
    )
    fwd_c1a_reuse_layer_15_norm_weight_l21: pl.Tensor[[D], pl.BF16] = pl.slice(
        attn_norm_weight, [D], [21 * D]
    )
    fwd_c1a_reuse_layer_15_wq_a_layer_l21: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[21]
    fwd_c1a_reuse_layer_15_wq_a_scale_layer_l21: pl.Tensor[
        [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [21 * (D // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_q_norm_layer_l21: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [21 * Q_LORA]
    )
    fwd_c1a_reuse_layer_15_wq_b_layer_l21: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[21]
    fwd_c1a_reuse_layer_15_wq_b_scale_layer_l21: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [21 * (Q_LORA // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_wkv_layer_l21: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[21]
    fwd_c1a_reuse_layer_15_wkv_scale_layer_l21: pl.Tensor[
        [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [21 * (D // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_kv_norm_layer_l21: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [21 * HEAD_DIM]
    )
    fwd_c1a_reuse_layer_15_sink_layer_l21: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [21 * LOCAL_H]
    )
    fwd_c1a_reuse_layer_15_wo_a_layer_l21: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
        pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [21 * LOCAL_O_GROUPS, 0, 0])
    )
    fwd_c1a_reuse_layer_15_wo_b_layer_l21: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[21]
    fwd_c1a_reuse_layer_15_wo_b_scale_layer_l21: pl.Tensor[
        [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [21 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    decode_c1a_reuse_sharded(
        x_pong,
        pre_mix_pong,
        fwd_c1a_reuse_layer_15_hc_fn_l21,
        fwd_c1a_reuse_layer_15_hc_scale_l21,
        fwd_c1a_reuse_layer_15_hc_base_l21,
        fwd_c1a_reuse_layer_15_norm_weight_l21,
        fwd_c1a_reuse_layer_15_wq_a_layer_l21,
        fwd_c1a_reuse_layer_15_wq_a_scale_layer_l21,
        fwd_c1a_reuse_layer_15_q_norm_layer_l21,
        fwd_c1a_reuse_layer_15_wq_b_layer_l21,
        fwd_c1a_reuse_layer_15_wq_b_scale_layer_l21,
        fwd_c1a_reuse_layer_15_wkv_layer_l21,
        fwd_c1a_reuse_layer_15_wkv_scale_layer_l21,
        fwd_c1a_reuse_layer_15_kv_norm_layer_l21,
        fwd_c1a_reuse_layer_15_sink_layer_l21,
        fwd_c1a_reuse_layer_15_wo_a_layer_l21,
        fwd_c1a_reuse_layer_15_wo_b_layer_l21,
        fwd_c1a_reuse_layer_15_wo_b_scale_layer_l21,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_9_window_cache_l21,
        fwd_attention_layer_9_window_cache_scale_l21,
        fwd_attention_layer_9_compressed_cache_l21,
        fwd_attention_layer_9_compressed_cache_scale_l21,
        topk_indices,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        attention_hidden,
        attention_pre_mix,
        rank // TP_SIZE * TP_SIZE,
        rank % TP_SIZE,
        attention_num_tokens,
        21 + 1,
    )
    fwd_moe_layer_16_routed_base_l21 = 21 * N_LOCAL_EXPERTS
    fwd_moe_layer_16_hc_ffn_fn_layer_l21: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [21 * MIX_HC, 0]
    )
    fwd_moe_layer_16_hc_ffn_scale_layer_l21: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [21 * 3])
    fwd_moe_layer_16_hc_ffn_base_layer_l21: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [21 * MIX_HC]
    )
    fwd_moe_layer_16_ffn_norm_weight_layer_l21: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [21 * D]
    )
    fwd_moe_layer_16_gate_weight_layer_l21: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [21 * N_EXPERTS, 0]
    )
    fwd_moe_layer_16_correction_bias_layer_l21: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [21 * N_EXPERTS]
    )
    fwd_moe_layer_16_routed_w1_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l21, 0, 0],
    )
    fwd_moe_layer_16_routed_w1_scale_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [21 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w2_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l21, 0, 0],
    )
    fwd_moe_layer_16_routed_w2_scale_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [21 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w3_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l21, 0, 0],
    )
    fwd_moe_layer_16_routed_w3_scale_layer_l21: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [21 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_shared_w1_layer_l21: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[21]
    fwd_moe_layer_16_shared_w1_scale_layer_l21: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [21 * (D // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w2_layer_l21: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[21]
    fwd_moe_layer_16_shared_w2_scale_layer_l21: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [21 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w3_layer_l21: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[21]
    fwd_moe_layer_16_shared_w3_scale_layer_l21: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [21 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_16_hc_ffn_fn_layer_l21,
        fwd_moe_layer_16_hc_ffn_scale_layer_l21,
        fwd_moe_layer_16_hc_ffn_base_layer_l21,
        fwd_moe_layer_16_ffn_norm_weight_layer_l21,
        fwd_moe_layer_16_gate_weight_layer_l21,
        fwd_moe_layer_16_correction_bias_layer_l21,
        fwd_moe_layer_16_routed_w1_layer_l21,
        fwd_moe_layer_16_routed_w1_scale_layer_l21,
        fwd_moe_layer_16_routed_w2_layer_l21,
        fwd_moe_layer_16_routed_w2_scale_layer_l21,
        fwd_moe_layer_16_routed_w3_layer_l21,
        fwd_moe_layer_16_routed_w3_scale_layer_l21,
        mxfp4_pair_lut,
        fwd_moe_layer_16_shared_w1_layer_l21,
        fwd_moe_layer_16_shared_w1_scale_layer_l21,
        fwd_moe_layer_16_shared_w2_layer_l21,
        fwd_moe_layer_16_shared_w2_scale_layer_l21,
        fwd_moe_layer_16_shared_w3_layer_l21,
        fwd_moe_layer_16_shared_w3_scale_layer_l21,
        pre_mix,
        x_mixed,
        x_hc,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        21 + 1,
    )
    fwd_attention_layer_1_window_cache_l22 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [22 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_1_window_cache_scale_l22 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [22 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_1_source_id_l22 = 3
    fwd_attention_layer_1_compressed_cache_l22 = pl.slice(
        compressed_cache_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
        [fwd_attention_layer_1_source_id_l22 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_1_compressed_cache_scale_l22 = pl.slice(
        compressed_cache_scale_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        [fwd_attention_layer_1_source_id_l22 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_c1a_reuse_layer_7_hc_fn_l22: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [22 * MIX_HC, 0]
    )
    fwd_c1a_reuse_layer_7_hc_scale_l22: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [22 * 3])
    fwd_c1a_reuse_layer_7_hc_base_l22: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [22 * MIX_HC]
    )
    fwd_c1a_reuse_layer_7_norm_weight_l22: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [22 * D])
    fwd_c1a_reuse_layer_7_wq_a_layer_l22: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[22]
    fwd_c1a_reuse_layer_7_wq_a_scale_layer_l22: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = (
        pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [22 * (D // MX_GROUP), 0])
    )
    fwd_c1a_reuse_layer_7_q_norm_layer_l22: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [22 * Q_LORA]
    )
    fwd_c1a_reuse_layer_7_wq_b_layer_l22: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[22]
    fwd_c1a_reuse_layer_7_wq_b_scale_layer_l22: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [22 * (Q_LORA // MX_GROUP), 0])
    fwd_c1a_reuse_layer_7_wkv_layer_l22: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[22]
    fwd_c1a_reuse_layer_7_wkv_scale_layer_l22: pl.Tensor[
        [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [22 * (D // MX_GROUP), 0])
    fwd_c1a_reuse_layer_7_kv_norm_layer_l22: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [22 * HEAD_DIM]
    )
    fwd_c1a_reuse_layer_7_sink_layer_l22: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [22 * LOCAL_H]
    )
    fwd_c1a_reuse_layer_7_wo_a_layer_l22: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [22 * LOCAL_O_GROUPS, 0, 0]
    )
    fwd_c1a_reuse_layer_7_wo_b_layer_l22: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[22]
    fwd_c1a_reuse_layer_7_wo_b_scale_layer_l22: pl.Tensor[
        [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [22 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    decode_c1a_reuse_sharded(
        x_hc,
        pre_mix,
        fwd_c1a_reuse_layer_7_hc_fn_l22,
        fwd_c1a_reuse_layer_7_hc_scale_l22,
        fwd_c1a_reuse_layer_7_hc_base_l22,
        fwd_c1a_reuse_layer_7_norm_weight_l22,
        fwd_c1a_reuse_layer_7_wq_a_layer_l22,
        fwd_c1a_reuse_layer_7_wq_a_scale_layer_l22,
        fwd_c1a_reuse_layer_7_q_norm_layer_l22,
        fwd_c1a_reuse_layer_7_wq_b_layer_l22,
        fwd_c1a_reuse_layer_7_wq_b_scale_layer_l22,
        fwd_c1a_reuse_layer_7_wkv_layer_l22,
        fwd_c1a_reuse_layer_7_wkv_scale_layer_l22,
        fwd_c1a_reuse_layer_7_kv_norm_layer_l22,
        fwd_c1a_reuse_layer_7_sink_layer_l22,
        fwd_c1a_reuse_layer_7_wo_a_layer_l22,
        fwd_c1a_reuse_layer_7_wo_b_layer_l22,
        fwd_c1a_reuse_layer_7_wo_b_scale_layer_l22,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_1_window_cache_l22,
        fwd_attention_layer_1_window_cache_scale_l22,
        fwd_attention_layer_1_compressed_cache_l22,
        fwd_attention_layer_1_compressed_cache_scale_l22,
        topk_indices,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        attention_hidden,
        attention_pre_mix,
        rank // TP_SIZE * TP_SIZE,
        rank % TP_SIZE,
        attention_num_tokens,
        22 + 1,
    )
    fwd_moe_layer_8_routed_base_l22 = 22 * N_LOCAL_EXPERTS
    fwd_moe_layer_8_hc_ffn_fn_layer_l22: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [22 * MIX_HC, 0]
    )
    fwd_moe_layer_8_hc_ffn_scale_layer_l22: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [22 * 3])
    fwd_moe_layer_8_hc_ffn_base_layer_l22: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [22 * MIX_HC]
    )
    fwd_moe_layer_8_ffn_norm_weight_layer_l22: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [22 * D]
    )
    fwd_moe_layer_8_gate_weight_layer_l22: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [22 * N_EXPERTS, 0]
    )
    fwd_moe_layer_8_correction_bias_layer_l22: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [22 * N_EXPERTS]
    )
    fwd_moe_layer_8_routed_w1_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l22, 0, 0],
    )
    fwd_moe_layer_8_routed_w1_scale_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [22 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w2_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l22, 0, 0],
    )
    fwd_moe_layer_8_routed_w2_scale_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [22 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_routed_w3_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_8_routed_base_l22, 0, 0],
    )
    fwd_moe_layer_8_routed_w3_scale_layer_l22: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [22 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_8_shared_w1_layer_l22: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[22]
    fwd_moe_layer_8_shared_w1_scale_layer_l22: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [22 * (D // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w2_layer_l22: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[22]
    fwd_moe_layer_8_shared_w2_scale_layer_l22: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [22 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_8_shared_w3_layer_l22: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[22]
    fwd_moe_layer_8_shared_w3_scale_layer_l22: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [22 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_8_hc_ffn_fn_layer_l22,
        fwd_moe_layer_8_hc_ffn_scale_layer_l22,
        fwd_moe_layer_8_hc_ffn_base_layer_l22,
        fwd_moe_layer_8_ffn_norm_weight_layer_l22,
        fwd_moe_layer_8_gate_weight_layer_l22,
        fwd_moe_layer_8_correction_bias_layer_l22,
        fwd_moe_layer_8_routed_w1_layer_l22,
        fwd_moe_layer_8_routed_w1_scale_layer_l22,
        fwd_moe_layer_8_routed_w2_layer_l22,
        fwd_moe_layer_8_routed_w2_scale_layer_l22,
        fwd_moe_layer_8_routed_w3_layer_l22,
        fwd_moe_layer_8_routed_w3_scale_layer_l22,
        mxfp4_pair_lut,
        fwd_moe_layer_8_shared_w1_layer_l22,
        fwd_moe_layer_8_shared_w1_scale_layer_l22,
        fwd_moe_layer_8_shared_w2_layer_l22,
        fwd_moe_layer_8_shared_w2_scale_layer_l22,
        fwd_moe_layer_8_shared_w3_layer_l22,
        fwd_moe_layer_8_shared_w3_scale_layer_l22,
        pre_mix_pong,
        x_mixed,
        x_pong,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        22 + 1,
    )
    fwd_attention_layer_9_window_cache_l23 = pl.slice(
        window_cache_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
        [23 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_9_window_cache_scale_l23 = pl.slice(
        window_cache_scale_pool,
        [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
        [23 * window_blocks_per_layer, 0, 0, 0],
    )
    fwd_attention_layer_9_source_id_l23 = 3
    fwd_attention_layer_9_compressed_cache_l23 = pl.slice(
        compressed_cache_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
        [fwd_attention_layer_9_source_id_l23 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_attention_layer_9_compressed_cache_scale_l23 = pl.slice(
        compressed_cache_scale_pool,
        [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
        [fwd_attention_layer_9_source_id_l23 * compressed_blocks_per_source, 0, 0, 0],
    )
    fwd_c1a_reuse_layer_15_hc_fn_l23: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [23 * MIX_HC, 0]
    )
    fwd_c1a_reuse_layer_15_hc_scale_l23: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [23 * 3])
    fwd_c1a_reuse_layer_15_hc_base_l23: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [23 * MIX_HC]
    )
    fwd_c1a_reuse_layer_15_norm_weight_l23: pl.Tensor[[D], pl.BF16] = pl.slice(
        attn_norm_weight, [D], [23 * D]
    )
    fwd_c1a_reuse_layer_15_wq_a_layer_l23: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[23]
    fwd_c1a_reuse_layer_15_wq_a_scale_layer_l23: pl.Tensor[
        [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [23 * (D // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_q_norm_layer_l23: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
        q_norm_weight, [Q_LORA], [23 * Q_LORA]
    )
    fwd_c1a_reuse_layer_15_wq_b_layer_l23: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[23]
    fwd_c1a_reuse_layer_15_wq_b_scale_layer_l23: pl.Tensor[
        [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [23 * (Q_LORA // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_wkv_layer_l23: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[23]
    fwd_c1a_reuse_layer_15_wkv_scale_layer_l23: pl.Tensor[
        [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [23 * (D // MX_GROUP), 0])
    fwd_c1a_reuse_layer_15_kv_norm_layer_l23: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        kv_norm_weight, [HEAD_DIM], [23 * HEAD_DIM]
    )
    fwd_c1a_reuse_layer_15_sink_layer_l23: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
        attn_sink, [LOCAL_H], [23 * LOCAL_H]
    )
    fwd_c1a_reuse_layer_15_wo_a_layer_l23: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
        pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [23 * LOCAL_O_GROUPS, 0, 0])
    )
    fwd_c1a_reuse_layer_15_wo_b_layer_l23: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[23]
    fwd_c1a_reuse_layer_15_wo_b_scale_layer_l23: pl.Tensor[
        [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [23 * (LOCAL_O_WIDTH // MX_GROUP), 0])
    decode_c1a_reuse_sharded(
        x_pong,
        pre_mix_pong,
        fwd_c1a_reuse_layer_15_hc_fn_l23,
        fwd_c1a_reuse_layer_15_hc_scale_l23,
        fwd_c1a_reuse_layer_15_hc_base_l23,
        fwd_c1a_reuse_layer_15_norm_weight_l23,
        fwd_c1a_reuse_layer_15_wq_a_layer_l23,
        fwd_c1a_reuse_layer_15_wq_a_scale_layer_l23,
        fwd_c1a_reuse_layer_15_q_norm_layer_l23,
        fwd_c1a_reuse_layer_15_wq_b_layer_l23,
        fwd_c1a_reuse_layer_15_wq_b_scale_layer_l23,
        fwd_c1a_reuse_layer_15_wkv_layer_l23,
        fwd_c1a_reuse_layer_15_wkv_scale_layer_l23,
        fwd_c1a_reuse_layer_15_kv_norm_layer_l23,
        fwd_c1a_reuse_layer_15_sink_layer_l23,
        fwd_c1a_reuse_layer_15_wo_a_layer_l23,
        fwd_c1a_reuse_layer_15_wo_b_layer_l23,
        fwd_c1a_reuse_layer_15_wo_b_scale_layer_l23,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        fwd_attention_layer_9_window_cache_l23,
        fwd_attention_layer_9_window_cache_scale_l23,
        fwd_attention_layer_9_compressed_cache_l23,
        fwd_attention_layer_9_compressed_cache_scale_l23,
        topk_indices,
        gathered,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        attention_hidden,
        attention_pre_mix,
        rank // TP_SIZE * TP_SIZE,
        rank % TP_SIZE,
        attention_num_tokens,
        23 + 1,
    )
    fwd_moe_layer_16_routed_base_l23 = 23 * N_LOCAL_EXPERTS
    fwd_moe_layer_16_hc_ffn_fn_layer_l23: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [23 * MIX_HC, 0]
    )
    fwd_moe_layer_16_hc_ffn_scale_layer_l23: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [23 * 3])
    fwd_moe_layer_16_hc_ffn_base_layer_l23: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [23 * MIX_HC]
    )
    fwd_moe_layer_16_ffn_norm_weight_layer_l23: pl.Tensor[[D], pl.BF16] = pl.slice(
        ffn_norm_weight, [D], [23 * D]
    )
    fwd_moe_layer_16_gate_weight_layer_l23: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_weight, [N_EXPERTS, D], [23 * N_EXPERTS, 0]
    )
    fwd_moe_layer_16_correction_bias_layer_l23: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        correction_bias, [N_EXPERTS], [23 * N_EXPERTS]
    )
    fwd_moe_layer_16_routed_w1_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w1,
        [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l23, 0, 0],
    )
    fwd_moe_layer_16_routed_w1_scale_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w1_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [23 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w2_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w2,
        [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l23, 0, 0],
    )
    fwd_moe_layer_16_routed_w2_scale_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w2_scale,
        [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
        [23 * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_routed_w3_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ] = pl.slice(
        routed_w3,
        [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
        [fwd_moe_layer_16_routed_base_l23, 0, 0],
    )
    fwd_moe_layer_16_routed_w3_scale_layer_l23: pl.Tensor[
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(
        routed_w3_scale,
        [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
        [23 * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
    )
    fwd_moe_layer_16_shared_w1_layer_l23: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[23]
    fwd_moe_layer_16_shared_w1_scale_layer_l23: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [23 * (D // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w2_layer_l23: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[23]
    fwd_moe_layer_16_shared_w2_scale_layer_l23: pl.Tensor[
        [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [23 * (MOE_INTER // MX_GROUP), 0])
    fwd_moe_layer_16_shared_w3_layer_l23: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[23]
    fwd_moe_layer_16_shared_w3_scale_layer_l23: pl.Tensor[
        [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [23 * (D // MX_GROUP), 0])
    moe(
        attention_hidden,
        attention_pre_mix,
        fwd_moe_layer_16_hc_ffn_fn_layer_l23,
        fwd_moe_layer_16_hc_ffn_scale_layer_l23,
        fwd_moe_layer_16_hc_ffn_base_layer_l23,
        fwd_moe_layer_16_ffn_norm_weight_layer_l23,
        fwd_moe_layer_16_gate_weight_layer_l23,
        fwd_moe_layer_16_correction_bias_layer_l23,
        fwd_moe_layer_16_routed_w1_layer_l23,
        fwd_moe_layer_16_routed_w1_scale_layer_l23,
        fwd_moe_layer_16_routed_w2_layer_l23,
        fwd_moe_layer_16_routed_w2_scale_layer_l23,
        fwd_moe_layer_16_routed_w3_layer_l23,
        fwd_moe_layer_16_routed_w3_scale_layer_l23,
        mxfp4_pair_lut,
        fwd_moe_layer_16_shared_w1_layer_l23,
        fwd_moe_layer_16_shared_w1_scale_layer_l23,
        fwd_moe_layer_16_shared_w2_layer_l23,
        fwd_moe_layer_16_shared_w2_scale_layer_l23,
        fwd_moe_layer_16_shared_w3_layer_l23,
        fwd_moe_layer_16_shared_w3_scale_layer_l23,
        pre_mix,
        x_mixed,
        x_hc,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        local_moe_tokens,
        rank,
        23 + 1,
    )
    for c1a_block in pl.range(4):
        fwd_attention_layer_1_window_cache_l24 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c1a_block * 4 + 24) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_window_cache_scale_l24 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c1a_block * 4 + 24) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_source_id_l24 = 3
        fwd_attention_layer_1_index_source_id_l24 = c1a_block + 4
        fwd_attention_layer_1_compressed_cache_l24 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_1_source_id_l24 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_compressed_cache_scale_l24 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l24 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_index_cache_l24 = pl.slice(
            index_cache_pool,
            [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2],
            [fwd_attention_layer_1_index_source_id_l24 * index_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_index_cache_scale_l24 = pl.slice(
            index_cache_scale_pool,
            [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP],
            [fwd_attention_layer_1_index_source_id_l24 * index_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_index_wq_b_scale_source_l24: pl.Tensor[
            [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            index_wq_b_scale,
            [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM],
            [(c1a_block + 4) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_attention_layer_1_index_wq_b_source_l24: pl.Tensor[
            [Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN
        ] = index_wq_b[fwd_attention_layer_1_index_source_id_l24]
        fwd_attention_layer_1_index_weights_proj_source_l24: pl.Tensor[[D, INDEX_H], pl.BF16] = (
            index_weights_proj[fwd_attention_layer_1_index_source_id_l24]
        )
        fwd_c1a_reindex_layer_6_hc_fn_l24: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 24) * MIX_HC, 0]
        )
        fwd_c1a_reindex_layer_6_hc_scale_l24: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c1a_block * 4 + 24) * 3]
        )
        fwd_c1a_reindex_layer_6_hc_base_l24: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c1a_block * 4 + 24) * MIX_HC]
        )
        fwd_c1a_reindex_layer_6_norm_weight_l24: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c1a_block * 4 + 24) * D]
        )
        fwd_c1a_reindex_layer_6_wq_a_layer_l24: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[
            c1a_block * 4 + 24
        ]
        fwd_c1a_reindex_layer_6_wq_a_scale_layer_l24: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c1a_block * 4 + 24) * (D // MX_GROUP), 0])
        fwd_c1a_reindex_layer_6_q_norm_layer_l24: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c1a_block * 4 + 24) * Q_LORA]
        )
        fwd_c1a_reindex_layer_6_wq_b_layer_l24: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c1a_block * 4 + 24
        ]
        fwd_c1a_reindex_layer_6_wq_b_scale_layer_l24: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c1a_block * 4 + 24) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c1a_reindex_layer_6_wkv_layer_l24: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[
            c1a_block * 4 + 24
        ]
        fwd_c1a_reindex_layer_6_wkv_scale_layer_l24: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c1a_block * 4 + 24) * (D // MX_GROUP), 0])
        fwd_c1a_reindex_layer_6_kv_norm_layer_l24: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c1a_block * 4 + 24) * HEAD_DIM]
        )
        fwd_c1a_reindex_layer_6_sink_layer_l24: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c1a_block * 4 + 24) * LOCAL_H]
        )
        fwd_c1a_reindex_layer_6_wo_a_layer_l24: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(
                wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c1a_block * 4 + 24) * LOCAL_O_GROUPS, 0, 0]
            )
        )
        fwd_c1a_reindex_layer_6_wo_b_layer_l24: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c1a_block * 4 + 24
        ]
        fwd_c1a_reindex_layer_6_wo_b_scale_layer_l24: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale,
            [LOCAL_O_WIDTH // MX_GROUP, D],
            [(c1a_block * 4 + 24) * (LOCAL_O_WIDTH // MX_GROUP), 0],
        )
        decode_c1a_reindex_sharded(
            x_hc,
            pre_mix,
            fwd_c1a_reindex_layer_6_hc_fn_l24,
            fwd_c1a_reindex_layer_6_hc_scale_l24,
            fwd_c1a_reindex_layer_6_hc_base_l24,
            fwd_c1a_reindex_layer_6_norm_weight_l24,
            fwd_c1a_reindex_layer_6_wq_a_layer_l24,
            fwd_c1a_reindex_layer_6_wq_a_scale_layer_l24,
            fwd_c1a_reindex_layer_6_q_norm_layer_l24,
            fwd_c1a_reindex_layer_6_wq_b_layer_l24,
            fwd_c1a_reindex_layer_6_wq_b_scale_layer_l24,
            fwd_c1a_reindex_layer_6_wkv_layer_l24,
            fwd_c1a_reindex_layer_6_wkv_scale_layer_l24,
            fwd_c1a_reindex_layer_6_kv_norm_layer_l24,
            fwd_c1a_reindex_layer_6_sink_layer_l24,
            fwd_c1a_reindex_layer_6_wo_a_layer_l24,
            fwd_c1a_reindex_layer_6_wo_b_layer_l24,
            fwd_c1a_reindex_layer_6_wo_b_scale_layer_l24,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_1_window_cache_l24,
            fwd_attention_layer_1_window_cache_scale_l24,
            fwd_attention_layer_1_compressed_cache_l24,
            fwd_attention_layer_1_compressed_cache_scale_l24,
            request_ids,
            c1a_compressed_lens,
            fwd_attention_layer_1_index_cache_l24,
            fwd_attention_layer_1_index_cache_scale_l24,
            c1a_index_block_table,
            pool_block_ids,
            pool_block_counts,
            fwd_attention_layer_1_index_wq_b_source_l24,
            fwd_attention_layer_1_index_wq_b_scale_source_l24,
            fwd_attention_layer_1_index_weights_proj_source_l24,
            topk_indices,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            attention_hidden,
            attention_pre_mix,
            rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE,
            attention_num_tokens,
            c1a_block * 4 + 24 + 1,
        )
        fwd_moe_layer_8_routed_base_l24 = (c1a_block * 4 + 24) * N_LOCAL_EXPERTS
        fwd_moe_layer_8_hc_ffn_fn_layer_l24: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 24) * MIX_HC, 0]
        )
        fwd_moe_layer_8_hc_ffn_scale_layer_l24: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c1a_block * 4 + 24) * 3]
        )
        fwd_moe_layer_8_hc_ffn_base_layer_l24: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c1a_block * 4 + 24) * MIX_HC]
        )
        fwd_moe_layer_8_ffn_norm_weight_layer_l24: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c1a_block * 4 + 24) * D]
        )
        fwd_moe_layer_8_gate_weight_layer_l24: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c1a_block * 4 + 24) * N_EXPERTS, 0]
        )
        fwd_moe_layer_8_correction_bias_layer_l24: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c1a_block * 4 + 24) * N_EXPERTS]
        )
        fwd_moe_layer_8_routed_w1_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l24, 0, 0],
        )
        fwd_moe_layer_8_routed_w1_scale_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 24) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w2_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l24, 0, 0],
        )
        fwd_moe_layer_8_routed_w2_scale_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c1a_block * 4 + 24) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w3_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l24, 0, 0],
        )
        fwd_moe_layer_8_routed_w3_scale_layer_l24: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 24) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_shared_w1_layer_l24: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c1a_block * 4 + 24
        ]
        fwd_moe_layer_8_shared_w1_scale_layer_l24: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 24) * (D // MX_GROUP), 0])
        fwd_moe_layer_8_shared_w2_layer_l24: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c1a_block * 4 + 24
        ]
        fwd_moe_layer_8_shared_w2_scale_layer_l24: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c1a_block * 4 + 24) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_8_shared_w3_layer_l24: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c1a_block * 4 + 24
        ]
        fwd_moe_layer_8_shared_w3_scale_layer_l24: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 24) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_8_hc_ffn_fn_layer_l24,
            fwd_moe_layer_8_hc_ffn_scale_layer_l24,
            fwd_moe_layer_8_hc_ffn_base_layer_l24,
            fwd_moe_layer_8_ffn_norm_weight_layer_l24,
            fwd_moe_layer_8_gate_weight_layer_l24,
            fwd_moe_layer_8_correction_bias_layer_l24,
            fwd_moe_layer_8_routed_w1_layer_l24,
            fwd_moe_layer_8_routed_w1_scale_layer_l24,
            fwd_moe_layer_8_routed_w2_layer_l24,
            fwd_moe_layer_8_routed_w2_scale_layer_l24,
            fwd_moe_layer_8_routed_w3_layer_l24,
            fwd_moe_layer_8_routed_w3_scale_layer_l24,
            mxfp4_pair_lut,
            fwd_moe_layer_8_shared_w1_layer_l24,
            fwd_moe_layer_8_shared_w1_scale_layer_l24,
            fwd_moe_layer_8_shared_w2_layer_l24,
            fwd_moe_layer_8_shared_w2_scale_layer_l24,
            fwd_moe_layer_8_shared_w3_layer_l24,
            fwd_moe_layer_8_shared_w3_scale_layer_l24,
            pre_mix_pong,
            x_mixed,
            x_pong,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c1a_block * 4 + 24 + 1,
        )
        fwd_attention_layer_9_window_cache_l25 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c1a_block * 4 + 25) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_window_cache_scale_l25 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c1a_block * 4 + 25) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_source_id_l25 = 3
        fwd_attention_layer_9_compressed_cache_l25 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_9_source_id_l25 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_9_compressed_cache_scale_l25 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_9_source_id_l25 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c1a_reuse_layer_15_hc_fn_l25: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 25) * MIX_HC, 0]
        )
        fwd_c1a_reuse_layer_15_hc_scale_l25: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c1a_block * 4 + 25) * 3]
        )
        fwd_c1a_reuse_layer_15_hc_base_l25: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c1a_block * 4 + 25) * MIX_HC]
        )
        fwd_c1a_reuse_layer_15_norm_weight_l25: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c1a_block * 4 + 25) * D]
        )
        fwd_c1a_reuse_layer_15_wq_a_layer_l25: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c1a_block * 4 + 25]
        fwd_c1a_reuse_layer_15_wq_a_scale_layer_l25: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c1a_block * 4 + 25) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_15_q_norm_layer_l25: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c1a_block * 4 + 25) * Q_LORA]
        )
        fwd_c1a_reuse_layer_15_wq_b_layer_l25: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c1a_block * 4 + 25
        ]
        fwd_c1a_reuse_layer_15_wq_b_scale_layer_l25: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c1a_block * 4 + 25) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c1a_reuse_layer_15_wkv_layer_l25: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c1a_block * 4 + 25]
        fwd_c1a_reuse_layer_15_wkv_scale_layer_l25: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c1a_block * 4 + 25) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_15_kv_norm_layer_l25: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c1a_block * 4 + 25) * HEAD_DIM]
        )
        fwd_c1a_reuse_layer_15_sink_layer_l25: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c1a_block * 4 + 25) * LOCAL_H]
        )
        fwd_c1a_reuse_layer_15_wo_a_layer_l25: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(
                wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c1a_block * 4 + 25) * LOCAL_O_GROUPS, 0, 0]
            )
        )
        fwd_c1a_reuse_layer_15_wo_b_layer_l25: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c1a_block * 4 + 25
        ]
        fwd_c1a_reuse_layer_15_wo_b_scale_layer_l25: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale,
            [LOCAL_O_WIDTH // MX_GROUP, D],
            [(c1a_block * 4 + 25) * (LOCAL_O_WIDTH // MX_GROUP), 0],
        )
        decode_c1a_reuse_sharded(
            x_pong,
            pre_mix_pong,
            fwd_c1a_reuse_layer_15_hc_fn_l25,
            fwd_c1a_reuse_layer_15_hc_scale_l25,
            fwd_c1a_reuse_layer_15_hc_base_l25,
            fwd_c1a_reuse_layer_15_norm_weight_l25,
            fwd_c1a_reuse_layer_15_wq_a_layer_l25,
            fwd_c1a_reuse_layer_15_wq_a_scale_layer_l25,
            fwd_c1a_reuse_layer_15_q_norm_layer_l25,
            fwd_c1a_reuse_layer_15_wq_b_layer_l25,
            fwd_c1a_reuse_layer_15_wq_b_scale_layer_l25,
            fwd_c1a_reuse_layer_15_wkv_layer_l25,
            fwd_c1a_reuse_layer_15_wkv_scale_layer_l25,
            fwd_c1a_reuse_layer_15_kv_norm_layer_l25,
            fwd_c1a_reuse_layer_15_sink_layer_l25,
            fwd_c1a_reuse_layer_15_wo_a_layer_l25,
            fwd_c1a_reuse_layer_15_wo_b_layer_l25,
            fwd_c1a_reuse_layer_15_wo_b_scale_layer_l25,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_9_window_cache_l25,
            fwd_attention_layer_9_window_cache_scale_l25,
            fwd_attention_layer_9_compressed_cache_l25,
            fwd_attention_layer_9_compressed_cache_scale_l25,
            topk_indices,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            attention_hidden,
            attention_pre_mix,
            rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE,
            attention_num_tokens,
            c1a_block * 4 + 25 + 1,
        )
        fwd_moe_layer_16_routed_base_l25 = (c1a_block * 4 + 25) * N_LOCAL_EXPERTS
        fwd_moe_layer_16_hc_ffn_fn_layer_l25: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 25) * MIX_HC, 0]
        )
        fwd_moe_layer_16_hc_ffn_scale_layer_l25: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c1a_block * 4 + 25) * 3]
        )
        fwd_moe_layer_16_hc_ffn_base_layer_l25: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c1a_block * 4 + 25) * MIX_HC]
        )
        fwd_moe_layer_16_ffn_norm_weight_layer_l25: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c1a_block * 4 + 25) * D]
        )
        fwd_moe_layer_16_gate_weight_layer_l25: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c1a_block * 4 + 25) * N_EXPERTS, 0]
        )
        fwd_moe_layer_16_correction_bias_layer_l25: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c1a_block * 4 + 25) * N_EXPERTS]
        )
        fwd_moe_layer_16_routed_w1_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l25, 0, 0],
        )
        fwd_moe_layer_16_routed_w1_scale_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 25) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w2_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l25, 0, 0],
        )
        fwd_moe_layer_16_routed_w2_scale_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c1a_block * 4 + 25) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w3_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l25, 0, 0],
        )
        fwd_moe_layer_16_routed_w3_scale_layer_l25: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 25) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_shared_w1_layer_l25: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c1a_block * 4 + 25
        ]
        fwd_moe_layer_16_shared_w1_scale_layer_l25: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 25) * (D // MX_GROUP), 0])
        fwd_moe_layer_16_shared_w2_layer_l25: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c1a_block * 4 + 25
        ]
        fwd_moe_layer_16_shared_w2_scale_layer_l25: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c1a_block * 4 + 25) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_16_shared_w3_layer_l25: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c1a_block * 4 + 25
        ]
        fwd_moe_layer_16_shared_w3_scale_layer_l25: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 25) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_16_hc_ffn_fn_layer_l25,
            fwd_moe_layer_16_hc_ffn_scale_layer_l25,
            fwd_moe_layer_16_hc_ffn_base_layer_l25,
            fwd_moe_layer_16_ffn_norm_weight_layer_l25,
            fwd_moe_layer_16_gate_weight_layer_l25,
            fwd_moe_layer_16_correction_bias_layer_l25,
            fwd_moe_layer_16_routed_w1_layer_l25,
            fwd_moe_layer_16_routed_w1_scale_layer_l25,
            fwd_moe_layer_16_routed_w2_layer_l25,
            fwd_moe_layer_16_routed_w2_scale_layer_l25,
            fwd_moe_layer_16_routed_w3_layer_l25,
            fwd_moe_layer_16_routed_w3_scale_layer_l25,
            mxfp4_pair_lut,
            fwd_moe_layer_16_shared_w1_layer_l25,
            fwd_moe_layer_16_shared_w1_scale_layer_l25,
            fwd_moe_layer_16_shared_w2_layer_l25,
            fwd_moe_layer_16_shared_w2_scale_layer_l25,
            fwd_moe_layer_16_shared_w3_layer_l25,
            fwd_moe_layer_16_shared_w3_scale_layer_l25,
            pre_mix,
            x_mixed,
            x_hc,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c1a_block * 4 + 25 + 1,
        )
        fwd_attention_layer_1_window_cache_l26 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c1a_block * 4 + 26) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_window_cache_scale_l26 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c1a_block * 4 + 26) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_1_source_id_l26 = 3
        fwd_attention_layer_1_compressed_cache_l26 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_1_source_id_l26 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_1_compressed_cache_scale_l26 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_1_source_id_l26 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c1a_reuse_layer_7_hc_fn_l26: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 26) * MIX_HC, 0]
        )
        fwd_c1a_reuse_layer_7_hc_scale_l26: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c1a_block * 4 + 26) * 3]
        )
        fwd_c1a_reuse_layer_7_hc_base_l26: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c1a_block * 4 + 26) * MIX_HC]
        )
        fwd_c1a_reuse_layer_7_norm_weight_l26: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c1a_block * 4 + 26) * D]
        )
        fwd_c1a_reuse_layer_7_wq_a_layer_l26: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c1a_block * 4 + 26]
        fwd_c1a_reuse_layer_7_wq_a_scale_layer_l26: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c1a_block * 4 + 26) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_7_q_norm_layer_l26: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c1a_block * 4 + 26) * Q_LORA]
        )
        fwd_c1a_reuse_layer_7_wq_b_layer_l26: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c1a_block * 4 + 26
        ]
        fwd_c1a_reuse_layer_7_wq_b_scale_layer_l26: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c1a_block * 4 + 26) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c1a_reuse_layer_7_wkv_layer_l26: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c1a_block * 4 + 26]
        fwd_c1a_reuse_layer_7_wkv_scale_layer_l26: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c1a_block * 4 + 26) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_7_kv_norm_layer_l26: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c1a_block * 4 + 26) * HEAD_DIM]
        )
        fwd_c1a_reuse_layer_7_sink_layer_l26: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c1a_block * 4 + 26) * LOCAL_H]
        )
        fwd_c1a_reuse_layer_7_wo_a_layer_l26: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(
                wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c1a_block * 4 + 26) * LOCAL_O_GROUPS, 0, 0]
            )
        )
        fwd_c1a_reuse_layer_7_wo_b_layer_l26: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c1a_block * 4 + 26
        ]
        fwd_c1a_reuse_layer_7_wo_b_scale_layer_l26: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale,
            [LOCAL_O_WIDTH // MX_GROUP, D],
            [(c1a_block * 4 + 26) * (LOCAL_O_WIDTH // MX_GROUP), 0],
        )
        decode_c1a_reuse_sharded(
            x_hc,
            pre_mix,
            fwd_c1a_reuse_layer_7_hc_fn_l26,
            fwd_c1a_reuse_layer_7_hc_scale_l26,
            fwd_c1a_reuse_layer_7_hc_base_l26,
            fwd_c1a_reuse_layer_7_norm_weight_l26,
            fwd_c1a_reuse_layer_7_wq_a_layer_l26,
            fwd_c1a_reuse_layer_7_wq_a_scale_layer_l26,
            fwd_c1a_reuse_layer_7_q_norm_layer_l26,
            fwd_c1a_reuse_layer_7_wq_b_layer_l26,
            fwd_c1a_reuse_layer_7_wq_b_scale_layer_l26,
            fwd_c1a_reuse_layer_7_wkv_layer_l26,
            fwd_c1a_reuse_layer_7_wkv_scale_layer_l26,
            fwd_c1a_reuse_layer_7_kv_norm_layer_l26,
            fwd_c1a_reuse_layer_7_sink_layer_l26,
            fwd_c1a_reuse_layer_7_wo_a_layer_l26,
            fwd_c1a_reuse_layer_7_wo_b_layer_l26,
            fwd_c1a_reuse_layer_7_wo_b_scale_layer_l26,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_1_window_cache_l26,
            fwd_attention_layer_1_window_cache_scale_l26,
            fwd_attention_layer_1_compressed_cache_l26,
            fwd_attention_layer_1_compressed_cache_scale_l26,
            topk_indices,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            attention_hidden,
            attention_pre_mix,
            rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE,
            attention_num_tokens,
            c1a_block * 4 + 26 + 1,
        )
        fwd_moe_layer_8_routed_base_l26 = (c1a_block * 4 + 26) * N_LOCAL_EXPERTS
        fwd_moe_layer_8_hc_ffn_fn_layer_l26: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 26) * MIX_HC, 0]
        )
        fwd_moe_layer_8_hc_ffn_scale_layer_l26: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c1a_block * 4 + 26) * 3]
        )
        fwd_moe_layer_8_hc_ffn_base_layer_l26: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c1a_block * 4 + 26) * MIX_HC]
        )
        fwd_moe_layer_8_ffn_norm_weight_layer_l26: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c1a_block * 4 + 26) * D]
        )
        fwd_moe_layer_8_gate_weight_layer_l26: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c1a_block * 4 + 26) * N_EXPERTS, 0]
        )
        fwd_moe_layer_8_correction_bias_layer_l26: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c1a_block * 4 + 26) * N_EXPERTS]
        )
        fwd_moe_layer_8_routed_w1_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l26, 0, 0],
        )
        fwd_moe_layer_8_routed_w1_scale_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 26) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w2_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l26, 0, 0],
        )
        fwd_moe_layer_8_routed_w2_scale_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c1a_block * 4 + 26) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_routed_w3_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_8_routed_base_l26, 0, 0],
        )
        fwd_moe_layer_8_routed_w3_scale_layer_l26: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 26) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_8_shared_w1_layer_l26: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c1a_block * 4 + 26
        ]
        fwd_moe_layer_8_shared_w1_scale_layer_l26: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 26) * (D // MX_GROUP), 0])
        fwd_moe_layer_8_shared_w2_layer_l26: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c1a_block * 4 + 26
        ]
        fwd_moe_layer_8_shared_w2_scale_layer_l26: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c1a_block * 4 + 26) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_8_shared_w3_layer_l26: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c1a_block * 4 + 26
        ]
        fwd_moe_layer_8_shared_w3_scale_layer_l26: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 26) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_8_hc_ffn_fn_layer_l26,
            fwd_moe_layer_8_hc_ffn_scale_layer_l26,
            fwd_moe_layer_8_hc_ffn_base_layer_l26,
            fwd_moe_layer_8_ffn_norm_weight_layer_l26,
            fwd_moe_layer_8_gate_weight_layer_l26,
            fwd_moe_layer_8_correction_bias_layer_l26,
            fwd_moe_layer_8_routed_w1_layer_l26,
            fwd_moe_layer_8_routed_w1_scale_layer_l26,
            fwd_moe_layer_8_routed_w2_layer_l26,
            fwd_moe_layer_8_routed_w2_scale_layer_l26,
            fwd_moe_layer_8_routed_w3_layer_l26,
            fwd_moe_layer_8_routed_w3_scale_layer_l26,
            mxfp4_pair_lut,
            fwd_moe_layer_8_shared_w1_layer_l26,
            fwd_moe_layer_8_shared_w1_scale_layer_l26,
            fwd_moe_layer_8_shared_w2_layer_l26,
            fwd_moe_layer_8_shared_w2_scale_layer_l26,
            fwd_moe_layer_8_shared_w3_layer_l26,
            fwd_moe_layer_8_shared_w3_scale_layer_l26,
            pre_mix_pong,
            x_mixed,
            x_pong,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c1a_block * 4 + 26 + 1,
        )
        fwd_attention_layer_9_window_cache_l27 = pl.slice(
            window_cache_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [(c1a_block * 4 + 27) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_window_cache_scale_l27 = pl.slice(
            window_cache_scale_pool,
            [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP],
            [(c1a_block * 4 + 27) * window_blocks_per_layer, 0, 0, 0],
        )
        fwd_attention_layer_9_source_id_l27 = 3
        fwd_attention_layer_9_compressed_cache_l27 = pl.slice(
            compressed_cache_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2],
            [fwd_attention_layer_9_source_id_l27 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_attention_layer_9_compressed_cache_scale_l27 = pl.slice(
            compressed_cache_scale_pool,
            [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP],
            [fwd_attention_layer_9_source_id_l27 * compressed_blocks_per_source, 0, 0, 0],
        )
        fwd_c1a_reuse_layer_15_hc_fn_l27: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 27) * MIX_HC, 0]
        )
        fwd_c1a_reuse_layer_15_hc_scale_l27: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [(c1a_block * 4 + 27) * 3]
        )
        fwd_c1a_reuse_layer_15_hc_base_l27: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [(c1a_block * 4 + 27) * MIX_HC]
        )
        fwd_c1a_reuse_layer_15_norm_weight_l27: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_weight, [D], [(c1a_block * 4 + 27) * D]
        )
        fwd_c1a_reuse_layer_15_wq_a_layer_l27: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[c1a_block * 4 + 27]
        fwd_c1a_reuse_layer_15_wq_a_scale_layer_l27: pl.Tensor[
            [D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [(c1a_block * 4 + 27) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_15_q_norm_layer_l27: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            q_norm_weight, [Q_LORA], [(c1a_block * 4 + 27) * Q_LORA]
        )
        fwd_c1a_reuse_layer_15_wq_b_layer_l27: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[
            c1a_block * 4 + 27
        ]
        fwd_c1a_reuse_layer_15_wq_b_scale_layer_l27: pl.Tensor[
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wq_b_scale,
            [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM],
            [(c1a_block * 4 + 27) * (Q_LORA // MX_GROUP), 0],
        )
        fwd_c1a_reuse_layer_15_wkv_layer_l27: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[c1a_block * 4 + 27]
        fwd_c1a_reuse_layer_15_wkv_scale_layer_l27: pl.Tensor[
            [D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [(c1a_block * 4 + 27) * (D // MX_GROUP), 0])
        fwd_c1a_reuse_layer_15_kv_norm_layer_l27: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            kv_norm_weight, [HEAD_DIM], [(c1a_block * 4 + 27) * HEAD_DIM]
        )
        fwd_c1a_reuse_layer_15_sink_layer_l27: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(
            attn_sink, [LOCAL_H], [(c1a_block * 4 + 27) * LOCAL_H]
        )
        fwd_c1a_reuse_layer_15_wo_a_layer_l27: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = (
            pl.slice(
                wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [(c1a_block * 4 + 27) * LOCAL_O_GROUPS, 0, 0]
            )
        )
        fwd_c1a_reuse_layer_15_wo_b_layer_l27: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[
            c1a_block * 4 + 27
        ]
        fwd_c1a_reuse_layer_15_wo_b_scale_layer_l27: pl.Tensor[
            [LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            wo_b_scale,
            [LOCAL_O_WIDTH // MX_GROUP, D],
            [(c1a_block * 4 + 27) * (LOCAL_O_WIDTH // MX_GROUP), 0],
        )
        decode_c1a_reuse_sharded(
            x_pong,
            pre_mix_pong,
            fwd_c1a_reuse_layer_15_hc_fn_l27,
            fwd_c1a_reuse_layer_15_hc_scale_l27,
            fwd_c1a_reuse_layer_15_hc_base_l27,
            fwd_c1a_reuse_layer_15_norm_weight_l27,
            fwd_c1a_reuse_layer_15_wq_a_layer_l27,
            fwd_c1a_reuse_layer_15_wq_a_scale_layer_l27,
            fwd_c1a_reuse_layer_15_q_norm_layer_l27,
            fwd_c1a_reuse_layer_15_wq_b_layer_l27,
            fwd_c1a_reuse_layer_15_wq_b_scale_layer_l27,
            fwd_c1a_reuse_layer_15_wkv_layer_l27,
            fwd_c1a_reuse_layer_15_wkv_scale_layer_l27,
            fwd_c1a_reuse_layer_15_kv_norm_layer_l27,
            fwd_c1a_reuse_layer_15_sink_layer_l27,
            fwd_c1a_reuse_layer_15_wo_a_layer_l27,
            fwd_c1a_reuse_layer_15_wo_b_layer_l27,
            fwd_c1a_reuse_layer_15_wo_b_scale_layer_l27,
            compressed_token_rope_cos,
            compressed_token_rope_sin,
            window_slots,
            window_indices,
            fwd_attention_layer_9_window_cache_l27,
            fwd_attention_layer_9_window_cache_scale_l27,
            fwd_attention_layer_9_compressed_cache_l27,
            fwd_attention_layer_9_compressed_cache_scale_l27,
            topk_indices,
            gathered,
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            attention_hidden,
            attention_pre_mix,
            rank // TP_SIZE * TP_SIZE,
            rank % TP_SIZE,
            attention_num_tokens,
            c1a_block * 4 + 27 + 1,
        )
        fwd_moe_layer_16_routed_base_l27 = (c1a_block * 4 + 27) * N_LOCAL_EXPERTS
        fwd_moe_layer_16_hc_ffn_fn_layer_l27: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [(c1a_block * 4 + 27) * MIX_HC, 0]
        )
        fwd_moe_layer_16_hc_ffn_scale_layer_l27: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [(c1a_block * 4 + 27) * 3]
        )
        fwd_moe_layer_16_hc_ffn_base_layer_l27: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [(c1a_block * 4 + 27) * MIX_HC]
        )
        fwd_moe_layer_16_ffn_norm_weight_layer_l27: pl.Tensor[[D], pl.BF16] = pl.slice(
            ffn_norm_weight, [D], [(c1a_block * 4 + 27) * D]
        )
        fwd_moe_layer_16_gate_weight_layer_l27: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_weight, [N_EXPERTS, D], [(c1a_block * 4 + 27) * N_EXPERTS, 0]
        )
        fwd_moe_layer_16_correction_bias_layer_l27: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            correction_bias, [N_EXPERTS], [(c1a_block * 4 + 27) * N_EXPERTS]
        )
        fwd_moe_layer_16_routed_w1_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l27, 0, 0],
        )
        fwd_moe_layer_16_routed_w1_scale_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w1_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 27) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w2_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l27, 0, 0],
        )
        fwd_moe_layer_16_routed_w2_scale_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w2_scale,
            [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D],
            [(c1a_block * 4 + 27) * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_routed_w3_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS],
            [fwd_moe_layer_16_routed_base_l27, 0, 0],
        )
        fwd_moe_layer_16_routed_w3_scale_layer_l27: pl.Tensor[
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            routed_w3_scale,
            [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER],
            [(c1a_block * 4 + 27) * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0],
        )
        fwd_moe_layer_16_shared_w1_layer_l27: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[
            c1a_block * 4 + 27
        ]
        fwd_moe_layer_16_shared_w1_scale_layer_l27: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 27) * (D // MX_GROUP), 0])
        fwd_moe_layer_16_shared_w2_layer_l27: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[
            c1a_block * 4 + 27
        ]
        fwd_moe_layer_16_shared_w2_scale_layer_l27: pl.Tensor[
            [MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(
            shared_w2_scale, [MOE_INTER // MX_GROUP, D], [(c1a_block * 4 + 27) * (MOE_INTER // MX_GROUP), 0]
        )
        fwd_moe_layer_16_shared_w3_layer_l27: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[
            c1a_block * 4 + 27
        ]
        fwd_moe_layer_16_shared_w3_scale_layer_l27: pl.Tensor[
            [D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
        ] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [(c1a_block * 4 + 27) * (D // MX_GROUP), 0])
        moe(
            attention_hidden,
            attention_pre_mix,
            fwd_moe_layer_16_hc_ffn_fn_layer_l27,
            fwd_moe_layer_16_hc_ffn_scale_layer_l27,
            fwd_moe_layer_16_hc_ffn_base_layer_l27,
            fwd_moe_layer_16_ffn_norm_weight_layer_l27,
            fwd_moe_layer_16_gate_weight_layer_l27,
            fwd_moe_layer_16_correction_bias_layer_l27,
            fwd_moe_layer_16_routed_w1_layer_l27,
            fwd_moe_layer_16_routed_w1_scale_layer_l27,
            fwd_moe_layer_16_routed_w2_layer_l27,
            fwd_moe_layer_16_routed_w2_scale_layer_l27,
            fwd_moe_layer_16_routed_w3_layer_l27,
            fwd_moe_layer_16_routed_w3_scale_layer_l27,
            mxfp4_pair_lut,
            fwd_moe_layer_16_shared_w1_layer_l27,
            fwd_moe_layer_16_shared_w1_scale_layer_l27,
            fwd_moe_layer_16_shared_w2_layer_l27,
            fwd_moe_layer_16_shared_w2_scale_layer_l27,
            fwd_moe_layer_16_shared_w3_layer_l27,
            fwd_moe_layer_16_shared_w3_scale_layer_l27,
            pre_mix,
            x_mixed,
            x_hc,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            local_moe_tokens,
            rank,
            c1a_block * 4 + 27 + 1,
        )
    return x_hc


def _decode_fwd(
    x_hc: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    base_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    base_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache_pool: pl.InOut[pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT * HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3: pl.Tensor[[N_LAYERS * N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w3: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    attention_input_window: pld.DistributedTensor[[DECODE_INPUT_MAX_TOKENS, D], pl.BF16],
    attention_input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
    attention_output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    wq_a_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    wq_b_scale: pl.Tensor[[N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    wo_b_scale: pl.Tensor[[N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    routed_w1_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3_scale: pl.Tensor[[N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
):
    decode_encoder(
        x_hc,
        pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        wq_a,
        q_norm_weight,
        wq_b,
        wkv,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        base_rope_cos,
        base_rope_sin,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        window_cache_pool,
        window_cache_scale_pool,
        compressed_cache_pool,
        compressed_cache_scale_pool,
        request_ids,
        token_to_req_indices,
        position_ids,
        c2a_compressed_lens,
        index_cache_pool,
        index_cache_scale_pool,
        c2a_index_block_table,
        c2a_rope_cos,
        c2a_rope_sin,
        c2a_compressor_wkv,
        c2a_compressor_wgate,
        c1a_compressor_wkv,
        query_start_loc,
        state_block_table,
        state_cache_pool,
        compressor_norm_weight,
        c2a_compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_weights_proj,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_weight,
        gate_weight,
        correction_bias,
        routed_w1,
        routed_w2,
        routed_w3,
        mxfp4_pair_lut,
        shared_w1,
        shared_w2,
        shared_w3,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        moe_num_tokens,
        attention_num_tokens,
        rank,
        wq_a_scale,
        wq_b_scale,
        wkv_scale,
        wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale,
        routed_w2_scale,
        routed_w3_scale,
        shared_w1_scale,
        shared_w2_scale,
        shared_w3_scale,
    )
    decode_decoder(
        x_hc,
        pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        wq_a,
        q_norm_weight,
        wq_b,
        wkv,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        base_rope_cos,
        base_rope_sin,
        compressed_token_rope_cos,
        compressed_token_rope_sin,
        window_slots,
        window_indices,
        window_cache_pool,
        window_cache_scale_pool,
        compressed_cache_pool,
        compressed_cache_scale_pool,
        request_ids,
        token_to_req_indices,
        position_ids,
        c1a_compressed_lens,
        index_cache_pool,
        index_cache_scale_pool,
        c1a_index_block_table,
        c2a_rope_cos,
        c2a_rope_sin,
        c2a_compressor_wkv,
        c2a_compressor_wgate,
        c1a_compressor_wkv,
        query_start_loc,
        state_block_table,
        state_cache_pool,
        compressor_norm_weight,
        c1a_compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_weights_proj,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        ffn_norm_weight,
        gate_weight,
        correction_bias,
        routed_w1,
        routed_w2,
        routed_w3,
        mxfp4_pair_lut,
        shared_w1,
        shared_w2,
        shared_w3,
        attention_input_window,
        attention_input_arrived,
        attention_output_window,
        attention_output_arrived,
        recv_meta,
        recv_x,
        recv_scale,
        recv_weights,
        recv_routes,
        arrived,
        data_arrived,
        routed_output,
        combine_arrived,
        moe_num_tokens,
        attention_num_tokens,
        rank,
        wq_a_scale,
        wq_b_scale,
        wkv_scale,
        wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale,
        routed_w2_scale,
        routed_w3_scale,
        shared_w1_scale,
        shared_w2_scale,
        shared_w3_scale,
    )
    return x_hc


decode_encoder = pl.jit.inline(auto_scope=False)(_decode_encoder)
decode_decoder = pl.jit.inline(auto_scope=False)(_decode_decoder)
l2_decode_encoder = pl.jit(auto_scope=False)(_decode_encoder)
l2_decode_decoder = pl.jit(auto_scope=False)(_decode_decoder)
decode_fwd = pl.jit.inline(auto_scope=False)(_decode_fwd)
l2_decode_fwd = pl.jit(auto_scope=False)(_decode_fwd)


@pl.jit.host
def l3_decode_fwd(
    x_hc: pl.InOut[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[EP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    hc_attn_fn: pl.Tensor[[EP_SIZE, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, N_LAYERS * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[EP_SIZE, N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    base_rope_cos: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    base_rope_sin: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_cos: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_token_rope_sin: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[EP_SIZE, T_DYN, BLOCK_SIZE], pl.INT32],
    window_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale_pool: pl.InOut[
        pl.Tensor[
            [EP_SIZE, FWD_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ]
    ],
    request_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_rope_cos: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_rope_sin: pl.Tensor[[EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.FP32],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[EP_SIZE, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
    ],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT * HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [EP_SIZE, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0
    ],
    index_weights_proj: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[EP_SIZE, N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[
        [EP_SIZE, N_LAYERS * N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w1_scale: pl.Tensor[
        [EP_SIZE, N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0
    ],
    routed_w2: pl.Tensor[
        [EP_SIZE, N_LAYERS * N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w2_scale: pl.Tensor[
        [EP_SIZE, N_LAYERS * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0
    ],
    routed_w3: pl.Tensor[
        [EP_SIZE, N_LAYERS * N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8
    ],
    routed_w3_scale: pl.Tensor[
        [EP_SIZE, N_LAYERS * (N_LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0
    ],
    mxfp4_pair_lut: pl.Tensor[[EP_SIZE, 2, 256], pl.INT16],
    shared_w1: pl.Tensor[[EP_SIZE, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Scalar[pl.INT32],
):
    """Run SWA with base rows and C2A/C1A with compressed-profile token rows.

    All rows use absolute original-token positions. C2A publication rows use
    p-1 for completed pairs and identity rotation otherwise. C1A publication
    uses compressed_token_rope_cos/sin at p, including its index key.

    The c2a_* and c1a_* metadata address separate compression families. At
    position p, their visible lengths include the current publication and
    equal (p+1)//2 and p+1. Incomplete C2A pairs use write slot -1.
    Page ids and slots are relative to each source's slice of the packed
    cache pools, with BLOCK_SIZE compressed rows per page. Sources within
    one family share a page mapping; the two families may use different
    page ids and table widths. Index and main KV caches share that mapping.
    build_forward_metadata lowers request positions and allocator tables
    into these lengths and slots. It does not allocate physical pages.
    """
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    pre_mix.bind_dynamic(1, LOCAL_T_DYN)
    base_rope_cos.bind_dynamic(1, T_DYN)
    base_rope_sin.bind_dynamic(1, T_DYN)
    compressed_token_rope_cos.bind_dynamic(1, T_DYN)
    compressed_token_rope_sin.bind_dynamic(1, T_DYN)
    c2a_rope_cos.bind_dynamic(1, T_DYN)
    c2a_rope_sin.bind_dynamic(1, T_DYN)
    window_cache_pool.bind_dynamic(1, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(1, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(1, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    state_cache_pool.bind_dynamic(1, FWD_STATE_BLOCKS_DYN)
    c2a_compressed_lens.bind_dynamic(1, T_DYN)
    c2a_compressed_slots.bind_dynamic(1, T_DYN)
    c1a_compressed_lens.bind_dynamic(1, T_DYN)
    c1a_compressed_slots.bind_dynamic(1, T_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    attention_input_buffer = pld.alloc_window_buffer([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
    attention_input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_output_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
    attention_output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer(
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32
    )
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer(
        [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD], dtype=pl.INT32
    )
    for rank in pl.range(pld.world_size()):
        attention_input_window = pld.window(attention_input_buffer, [DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
        attention_input_arrived = pld.window(attention_input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        attention_output_window = pld.window(attention_output_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        attention_output_arrived = pld.window(attention_output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(
            recv_scale_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8
        )
        recv_weights = pld.window(recv_weights_buffer, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(
            recv_routes_buffer, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32
        )
        arrived = pld.window(arrived_buffer, [EP_SIZE, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buffer,
            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buffer,
            [EP_SIZE, N_LOCAL_EXPERTS, SIGNAL_PAD],
            dtype=pl.INT32,
        )
        l2_decode_encoder(
            x_hc[rank],
            pre_mix[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            q_norm_weight[rank],
            wq_b[rank],
            wkv[rank],
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            base_rope_cos[rank],
            base_rope_sin[rank],
            compressed_token_rope_cos[rank],
            compressed_token_rope_sin[rank],
            window_slots[rank],
            window_indices[rank],
            window_cache_pool[rank],
            window_cache_scale_pool[rank],
            compressed_cache_pool[rank],
            compressed_cache_scale_pool[rank],
            request_ids[rank],
            token_to_req_indices[rank],
            position_ids[rank],
            c2a_compressed_lens[rank],
            index_cache_pool[rank],
            index_cache_scale_pool[rank],
            c2a_index_block_table[rank],
            c2a_rope_cos[rank],
            c2a_rope_sin[rank],
            c2a_compressor_wkv[rank],
            c2a_compressor_wgate[rank],
            c1a_compressor_wkv[rank],
            query_start_loc[rank],
            state_block_table[rank],
            state_cache_pool[rank],
            compressor_norm_weight[rank],
            c2a_compressed_slots[rank],
            index_wk[rank],
            index_norm_weight[rank],
            index_wq_b[rank],
            index_weights_proj[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank],
            correction_bias[rank],
            routed_w1[rank],
            routed_w2[rank],
            routed_w3[rank],
            mxfp4_pair_lut[rank],
            shared_w1[rank],
            shared_w2[rank],
            shared_w3[rank],
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            moe_num_tokens,
            attention_num_tokens,
            rank,
            wq_a_scale[rank],
            wq_b_scale[rank],
            wkv_scale[rank],
            wo_b_scale[rank],
            index_wq_b_scale[rank],
            routed_w1_scale[rank],
            routed_w2_scale[rank],
            routed_w3_scale[rank],
            shared_w1_scale[rank],
            shared_w2_scale[rank],
            shared_w3_scale[rank],
            device=rank,
        )
        l2_decode_decoder(
            x_hc[rank],
            pre_mix[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank],
            q_norm_weight[rank],
            wq_b[rank],
            wkv[rank],
            kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            base_rope_cos[rank],
            base_rope_sin[rank],
            compressed_token_rope_cos[rank],
            compressed_token_rope_sin[rank],
            window_slots[rank],
            window_indices[rank],
            window_cache_pool[rank],
            window_cache_scale_pool[rank],
            compressed_cache_pool[rank],
            compressed_cache_scale_pool[rank],
            request_ids[rank],
            token_to_req_indices[rank],
            position_ids[rank],
            c1a_compressed_lens[rank],
            index_cache_pool[rank],
            index_cache_scale_pool[rank],
            c1a_index_block_table[rank],
            c2a_rope_cos[rank],
            c2a_rope_sin[rank],
            c2a_compressor_wkv[rank],
            c2a_compressor_wgate[rank],
            c1a_compressor_wkv[rank],
            query_start_loc[rank],
            state_block_table[rank],
            state_cache_pool[rank],
            compressor_norm_weight[rank],
            c1a_compressed_slots[rank],
            index_wk[rank],
            index_norm_weight[rank],
            index_wq_b[rank],
            index_weights_proj[rank],
            hc_ffn_fn[rank],
            hc_ffn_scale[rank],
            hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank],
            correction_bias[rank],
            routed_w1[rank],
            routed_w2[rank],
            routed_w3[rank],
            mxfp4_pair_lut[rank],
            shared_w1[rank],
            shared_w2[rank],
            shared_w3[rank],
            attention_input_window,
            attention_input_arrived,
            attention_output_window,
            attention_output_arrived,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            moe_num_tokens,
            attention_num_tokens,
            rank,
            wq_a_scale[rank],
            wq_b_scale[rank],
            wkv_scale[rank],
            wo_b_scale[rank],
            index_wq_b_scale[rank],
            routed_w1_scale[rank],
            routed_w2_scale[rank],
            routed_w3_scale[rank],
            shared_w1_scale[rank],
            shared_w2_scale[rank],
            shared_w3_scale[rank],
            device=rank,
        )


def _torch_dtype(dtype):
    values = {
        pl.BF16: torch.bfloat16,
        pl.FP32: torch.float32,
        pl.FP8E4M3FN: torch.float8_e4m3fn,
        pl.FP8E8M0: torch.float8_e8m0fnu,
        pl.INT16: torch.int16,
        pl.INT32: torch.int32,
        pl.INT64: torch.int64,
        pl.UINT8: torch.uint8,
    }
    return values[dtype]


def _fixture_extent(dim):
    if isinstance(dim, int):
        return dim
    name = repr(dim)
    extents = {
        "LOCAL_T_DYN": MOE_TOKENS,
        "Q_START_DYN": FIXTURE_BATCH + 1,
        "T_DYN": FIXTURE_TOKENS,
        "FWD_ORI_BLOCKS_DYN": N_LAYERS * FIXTURE_BATCH,
        "FWD_CMP_BLOCKS_DYN": KV_SOURCE_COUNT * FIXTURE_BATCH,
        "FWD_INDEX_BLOCKS_DYN": INDEX_SOURCE_COUNT * FIXTURE_BATCH,
        "FWD_STATE_BLOCKS_DYN": C2A_SOURCE_COUNT * FIXTURE_BATCH,
        "FWD_C2A_TABLE_DYN": 1,
        "FWD_C1A_TABLE_DYN": 2,
        "B_DYN": FIXTURE_BATCH,
    }
    for marker, extent in extents.items():
        if marker in name:
            return extent
    raise ValueError(f"no decode_fwd fixture extent for {dim!r}")


def build_tensor_specs():
    """Build the TP1 twelve-request decode smoke fixture."""
    from golden import ScalarSpec, TensorSpec
    from moe import _route_bias
    from quant import build_mxfp4_pair_lut

    if TP_SIZE != 1:
        raise ValueError("decode_fwd fixture requires --tp 1")

    signature = inspect.signature(l3_decode_fwd._func)
    specs = []
    request_ids = torch.arange(FIXTURE_BATCH, dtype=torch.int32).repeat_interleave(FIXTURE_SEQUENCE)
    position_ids = torch.arange(FIXTURE_SEQUENCE, dtype=torch.int32).repeat(FIXTURE_BATCH)
    query_starts = torch.arange(0, FIXTURE_TOKENS + 1, FIXTURE_SEQUENCE, dtype=torch.int32)
    base_cos, base_sin = precompute_rope_tables(FIXTURE_SEQUENCE, compressed_attention=False)
    token_cos, token_sin = precompute_rope_tables(FIXTURE_SEQUENCE, compressed_attention=True)
    complete = (position_ids + 1).remainder(2) == 0
    pair_start = (position_ids - 1).clamp_min(0)
    rope_rows = {
        "base_rope_cos": base_cos[position_ids], "base_rope_sin": base_sin[position_ids],
        "compressed_token_rope_cos": token_cos[position_ids], "compressed_token_rope_sin": token_sin[position_ids],
        "c2a_rope_cos": token_cos[pair_start].masked_fill(~complete[:, None], 1.0),
        "c2a_rope_sin": token_sin[pair_start].masked_fill(~complete[:, None], 0.0),
    }
    window_table = torch.arange(FIXTURE_BATCH, dtype=torch.int32).reshape(FIXTURE_BATCH, 1)
    c2a_table = window_table.flip(0)
    c1a_table = torch.cat((window_table, torch.full_like(window_table, -1)), dim=1)
    compressed_tables = {
        source: c2a_table if M.compress_ratios[source] == 2 else c1a_table for source in M.kv_source_layer_ids
    }
    state_tables = {source: window_table for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2}
    prefix_lens = torch.zeros(FIXTURE_BATCH, dtype=torch.int32)
    metadata = build_forward_metadata(query_starts, prefix_lens, window_table, compressed_tables, state_tables)
    metadata_rows = {"window_slots": metadata.window_slots, "window_indices": metadata.window_indices}
    for family, ratio in (("c2a", 2), ("c1a", 1)):
        source = next(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == ratio)
        metadata_rows[family + "_compressed_lens"] = metadata.compressed_lens[source]
        metadata_rows[family + "_compressed_slots"] = metadata.compressed_slots[source]
        metadata_rows[family + "_index_block_table"] = compressed_tables[source]
    for name in l3_decode_fwd.param_names:
        if name == "attention_num_tokens":
            specs.append(
                ScalarSpec(
                    name,
                    torch.int32,
                    FIXTURE_TOKENS,
                    compile_runtime=True,
                )
            )
            continue
        annotation = signature.parameters[name].annotation
        shape = [_fixture_extent(dim) for dim in annotation.shape]
        dtype = _torch_dtype(annotation.dtype)
        init_value = None
        if name in ("request_ids", "token_to_req_indices"):
            init_value = request_ids.repeat(EP_SIZE, 1)
        elif name == "position_ids":
            init_value = position_ids.repeat(EP_SIZE, 1)
        elif name in rope_rows:
            init_value = rope_rows[name].unsqueeze(0).repeat(EP_SIZE, 1, 1)
        elif name in metadata_rows:
            row = metadata_rows[name]
            init_value = row.unsqueeze(0).repeat(EP_SIZE, *([1] * row.ndim))
        elif name == "query_start_loc":
            init_value = query_starts.repeat(EP_SIZE, 1)
        elif name == "state_block_table":
            init_value = (
                torch.arange(FIXTURE_BATCH, dtype=torch.int32)
                .reshape(1, FIXTURE_BATCH, 1)
                .repeat(EP_SIZE, 1, 1)
            )
        elif name == "moe_num_tokens":
            init_value = torch.full((EP_SIZE,), FIXTURE_TOKENS, dtype=torch.int32)
        elif name == "correction_bias":
            init_value = _route_bias().repeat(EP_SIZE, N_LAYERS)
        elif name == "mxfp4_pair_lut":
            init_value = build_mxfp4_pair_lut().repeat(EP_SIZE, 1, 1)
        elif name.endswith("_scale"):
            init_value = 1.0
        specs.append(TensorSpec(name, shape, dtype, init_value=init_value))
    return specs


if __name__ == "__main__":
    from utils import upload_forward_inputs
    from golden import run
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto.runtime import RunConfig
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1])
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--experts-per-rank", type=int, default=N_LOCAL_EXPERTS, help="routed experts on each EP rank")
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(EP_SIZE)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-scope-stats", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    if (args.tp, args.ep, args.experts_per_rank) != (TP_SIZE, EP_SIZE, N_LOCAL_EXPERTS):
        parser.error("--tp/--ep/--experts-per-rank are import-time configuration; pass them on the Python command line")
    if args.tp != 1:
        parser.error("the decode_fwd smoke fixture requires --tp 1")
    devices = [int(device) for device in args.device.split(",") if device]
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE:
        parser.error(f"need exactly {EP_SIZE} distinct devices, got {devices}")
    specs = build_tensor_specs()
    config = dict(
        platform=args.platform,
        dump_passes=args.dump_passes,
        enable_scope_stats=args.enable_scope_stats,
        log_level=args.log_level,
        ring_heap=(2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2),
        ring_dep_pool=131072,
        distributed_config=DistributedConfig(
            device_ids=devices,
            num_sub_workers=0,
        ),
    )
    result = run(
        fn=l3_decode_fwd,
        specs=specs,
        config=config,
        compile_only=True,
    )
    if result.passed and not args.compile_only:
        cfg = RunConfig(**{name: value for name, value in config.items() if name != "log_level"})
        compiled = DistributedCompiledProgram.from_dir(result.work_dir, platform=cfg.platform)
        replicated = ('moe_num_tokens',)
        host_inputs = {spec.name: spec.create_tensor().share_memory_() for spec in specs if spec.name in replicated}
        with compiled.prepare(cfg, persistent=False, inherited_host_tensors=list(host_inputs.values())) as runtime:
            arguments = upload_forward_inputs(runtime, specs, compiled._get_metadata()[0], len(devices), host_inputs)
            runtime(*arguments, config=cfg)
        print("[FWD] Execution passed; numerical accuracy was not validated.", flush=True)
    if not result.passed:
        raise SystemExit(result.error or 1)
