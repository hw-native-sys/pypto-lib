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
"""DeepSeek-V4.1-Flash 40-layer decode with final HC, RMSNorm, logits and greedy sampling."""

import argparse
import pypto.language as pl
import pypto.language.distributed as pld
import config as C

_shape_parser = argparse.ArgumentParser(add_help=False)
_default_experts = C.FLASH.n_routed_experts // C.EP_SIZE if C.EP_SIZE in (4, 8) else 64
_shape_parser.add_argument("--experts-per-rank", type=int, default=_default_experts)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.experts_per_rank <= 0 or C.EP_SIZE * _shape_options.experts_per_rank % 16:
    _shape_parser.error("--experts-per-rank must be positive and give a total expert count aligned to 16")
C.N_LOCAL_EXPERTS = _shape_options.experts_per_rank


from config import (
    FLASH as M, B_DYN, LOCAL_T_DYN, Q_START_DYN, ROUTE_T_DYN, T_DYN, BLOCK_SIZE, COMPRESSED_CACHE_GROUP,
    DECODE_MAX_TOKENS, DECODE_INPUT_MAX_TOKENS, INDEX_CACHE_GROUP, STATE_CAPACITY, WINDOW_CACHE_GROUP,
    MOE_TOKENS,
)
from compressor import STATE_WIDTH
from decode_c1a_full import decode_c1a_full
from decode_c1a_reindex import decode_c1a_reindex
from decode_c1a_reuse import decode_c1a_reuse
from decode_c2a_full import decode_c2a_full
from decode_c2a_reuse import decode_c2a_reuse
from decode_swa import decode_swa
from input_pack import prepare_compressed_lens
from lm_head import GROUP_LOGIT_ROWS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD, VOCAB, VOCAB_PER_TP, final_output
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows
from engram import ENGRAM_K, KV_OUT, engram_from_rows
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS
from decode_moe import (
    moe, SIGNAL_PAD, clear_moe_signals, AUX_WIDTH, EP_SIZE, N_LOCAL_EXPERTS, RECV_MAX, ROUTE_WIDTH,
)


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
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
DECODE_MAX_TOKENS = DECODE_MAX_TOKENS
STATE_CAPACITY = STATE_CAPACITY

FIXTURE_BATCH = 12
FIXTURE_SEQUENCE = 6
FIXTURE_TOKENS = FIXTURE_BATCH * FIXTURE_SEQUENCE


FWD_ORI_BLOCKS_DYN = pl.dynamic("V41_FWD_ORI_BLOCKS_DYN")
FWD_CMP_BLOCKS_DYN = pl.dynamic("V41_FWD_CMP_BLOCKS_DYN")
FWD_INDEX_BLOCKS_DYN = pl.dynamic("V41_FWD_INDEX_BLOCKS_DYN")
FWD_STATE_BLOCKS_DYN = pl.dynamic("V41_FWD_STATE_BLOCKS_DYN")
FWD_C2A_TABLE_DYN = pl.dynamic("V41_FWD_C2A_TABLE_DYN")
FWD_C1A_TABLE_DYN = pl.dynamic("V41_FWD_C1A_TABLE_DYN")


@pl.jit.inline(auto_scope=False)
def decode_encoder(
    engram_lookup: pl.Tensor[[2, LOCAL_T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
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
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    token_rope_cos: pl.Out[pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32]],
    token_rope_sin: pl.Out[pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32]],
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
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
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
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
) -> tuple[
    pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
    pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
]:
    attention_count = pl.read(attention_num_tokens, [rank])
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c2a_compressed_slots.bind_dynamic(0, T_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    position_ids.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, ROPE_ROWS_DYN)
    global_tokens = pl.tensor.dim(position_ids, 0)
    compressed_token_rope_cos = pl.slice(token_rope_cos, [global_tokens, ROPE_DIM // 2], [0, 0])
    compressed_token_rope_sin = pl.slice(token_rope_sin, [global_tokens, ROPE_DIM // 2], [0, 0])
    base_rope_cos = pl.create_tensor([global_tokens, ROPE_DIM // 2], dtype=pl.FP32)
    base_rope_sin = pl.create_tensor([global_tokens, ROPE_DIM // 2], dtype=pl.FP32)
    c2a_rope_cos = pl.create_tensor([global_tokens, ROPE_DIM // 2], dtype=pl.FP32)
    c2a_rope_sin = pl.create_tensor([global_tokens, ROPE_DIM // 2], dtype=pl.FP32)
    materialize_rope_rows(
        freqs_cos[0], freqs_sin[0], position_ids, attention_count, base_rope_cos, base_rope_sin, False,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids, attention_count,
        compressed_token_rope_cos, compressed_token_rope_sin, False,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids, attention_count, c2a_rope_cos, c2a_rope_sin, True,
    )
    c2a_compressed_lens = pl.create_tensor([global_tokens], dtype=pl.INT32)
    prepare_compressed_lens(position_ids, c2a_compressed_lens, 1)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // KV_SOURCE_COUNT
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // KV_SOURCE_COUNT
    state_blocks_per_source = pl.tensor.dim(state_cache_pool, 0) // C2A_SOURCE_COUNT
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    topk_indices = pl.create_tensor([global_tokens, INDEX_TOPK], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    for layer in pl.range(0, 20):
        if layer == 1 or layer == 14:
            with pl.scope():
                engram_slot = layer // 14
                layer_engram_scale = pl.slice(engram_wkv_scale, [ENGRAM_K // 32, KV_OUT], [engram_slot * (ENGRAM_K // 32), 0])
                engram_from_rows(
                    engram_lookup[engram_slot], engram_wkv[engram_slot], layer_engram_scale,
                    engram_weight[engram_slot], x_hc, x_hc,
                )
        with pl.scope():
            layer_window_cache = pl.slice(window_cache_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_window_cache_scale = pl.slice(window_cache_scale_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer * 3])
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer * MIX_HC])
            layer_norm_weight: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [layer * D])
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[layer]
            layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [layer * (D // MX_GROUP), 0])
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [layer * Q_LORA])
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[layer]
            layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [layer * (Q_LORA // MX_GROUP), 0])
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[layer]
            layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [layer * (D // MX_GROUP), 0])
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [layer * HEAD_DIM])
            layer_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [layer * LOCAL_H])
            layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[layer]
            layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
            if layer < 2:
                swa_attention_output = pl.create_tensor([tokens, D], dtype=pl.BF16)
                decode_swa(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    base_rope_cos, base_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    swa_attention_output,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            elif (layer - 2) % 6 == 0:
                full_source_id = (layer - 2) // 6
                full_compressed_cache = pl.slice(compressed_cache_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2], [full_source_id * compressed_blocks_per_source, 0, 0, 0])
                full_compressed_cache_scale = pl.slice(compressed_cache_scale_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], [full_source_id * compressed_blocks_per_source, 0, 0, 0])
                full_index_cache = pl.slice(index_cache_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2], [full_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_cache_scale = pl.slice(index_cache_scale_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], [full_source_id * index_blocks_per_source, 0, 0, 0])
                full_state_cache = pl.slice(state_cache_pool, [state_blocks_per_source, STATE_CAPACITY, STATE_WIDTH], [full_source_id * state_blocks_per_source, 0, 0])
                full_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [(layer - 2) // 6 * (Q_LORA // MX_GROUP), 0])
                full_c2a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wkv[full_source_id]
                full_c2a_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wgate[full_source_id]
                full_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [full_source_id * HEAD_DIM])
                full_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[full_source_id]
                full_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [full_source_id * INDEX_DIM])
                full_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[full_source_id]
                full_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[full_source_id]
                decode_c2a_full(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    full_compressed_cache, full_compressed_cache_scale,
                    token_to_req_indices, host_token_to_req_indices, c2a_compressed_lens,
                    full_index_cache, full_index_cache_scale,
                    c2a_index_block_table, position_ids,
                    c2a_rope_cos, c2a_rope_sin,
                    full_c2a_compressor_wkv, full_c2a_compressor_wgate,
                    query_start_loc, state_block_table,
                    full_state_cache, full_compressor_norm,
                    c2a_compressed_slots,
                    full_index_wk, full_index_norm,
                    full_index_wq_b, full_index_wq_b_scale, full_index_weights_proj,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            else:
                reuse_source_id = (layer - 2) // 6
                reuse_compressed_cache = pl.slice(compressed_cache_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2], [reuse_source_id * compressed_blocks_per_source, 0, 0, 0])
                reuse_compressed_cache_scale = pl.slice(compressed_cache_scale_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], [reuse_source_id * compressed_blocks_per_source, 0, 0, 0])
                decode_c2a_reuse(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    reuse_compressed_cache, reuse_compressed_cache_scale,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            ffn_routed_base = layer * N_LOCAL_EXPERTS
            ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
            ffn_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer * 3])
            ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer * MIX_HC])
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [layer * D])
            ffn_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_weight, [N_EXPERTS, D], [layer * N_EXPERTS, 0])
            ffn_correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [layer * N_EXPERTS])
            ffn_routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w1_scale, [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0])
            ffn_routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w2_scale, [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], [layer * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0])
            ffn_routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w3_scale, [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0])
            ffn_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[layer]
            ffn_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
            ffn_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[layer]
            ffn_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [layer * (MOE_INTER // MX_GROUP), 0])
            ffn_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[layer]
            ffn_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
            force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
            moe(
                attention_hidden, attention_pre_mix,
                ffn_hc_fn, ffn_hc_scale, ffn_hc_base,
                layer_ffn_norm,
                ffn_gate_weight, ffn_correction_bias,
                ffn_routed_w1, ffn_routed_w1_scale,
                ffn_routed_w2, ffn_routed_w2_scale,
                ffn_routed_w3, ffn_routed_w3_scale,
                mxfp4_pair_lut,
                ffn_shared_w1, ffn_shared_w1_scale,
                ffn_shared_w2, ffn_shared_w2_scale,
                ffn_shared_w3, ffn_shared_w3_scale,
                pre_mix, x_mixed, x_hc,
                recv_meta, recv_x, recv_scale,
                recv_weights, recv_routes,
                arrived, data_arrived,
                routed_output, combine_arrived,
                local_moe_tokens, rank, layer + 1,
                force_topk,
            )
    return x_hc, pre_mix, token_rope_cos, token_rope_sin


@pl.jit.inline(auto_scope=False)
def decode_decoder(
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
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    token_rope_cos: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
    token_rope_sin: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], pl.FP32],
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
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
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
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
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
    attention_count = pl.read(attention_num_tokens, [rank])
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    pre_mix.bind_dynamic(0, LOCAL_T_DYN)
    window_cache_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(0, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(0, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, FWD_INDEX_BLOCKS_DYN)
    c1a_compressed_slots.bind_dynamic(0, T_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    state_cache_pool.bind_dynamic(0, FWD_STATE_BLOCKS_DYN)
    tokens = pl.tensor.dim(x_hc, 0)
    position_ids.bind_dynamic(0, T_DYN)
    global_tokens = pl.tensor.dim(position_ids, 0)
    compressed_token_rope_cos = pl.slice(token_rope_cos, [global_tokens, ROPE_DIM // 2], [0, 0])
    compressed_token_rope_sin = pl.slice(token_rope_sin, [global_tokens, ROPE_DIM // 2], [0, 0])
    c1a_compressed_lens = pl.create_tensor([global_tokens], dtype=pl.INT32)
    prepare_compressed_lens(position_ids, c1a_compressed_lens, 0)
    window_blocks_per_layer = pl.tensor.dim(window_cache_pool, 0) // N_LAYERS
    compressed_blocks_per_source = pl.tensor.dim(compressed_cache_pool, 0) // KV_SOURCE_COUNT
    index_blocks_per_source = pl.tensor.dim(index_cache_pool, 0) // KV_SOURCE_COUNT
    attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attention_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    topk_indices = pl.create_tensor([global_tokens, INDEX_TOPK], dtype=pl.INT32)
    pool_blocks_raw = pl.tensor.dim(c1a_index_block_table, 1) * (BLOCK_SIZE // CANDIDATE_BLOCK_SIZE)
    pool_block_capacity = pl.min(CANDIDATE_TOPK_BLOCKS, pl.max(32, (pool_blocks_raw + 31) // 32 * 32))
    pool_block_ids = pl.create_tensor([global_tokens, pool_block_capacity], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([global_tokens], dtype=pl.INT32)
    x_mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_moe_tokens = pl.read(moe_num_tokens, [rank])
    for layer in pl.range(20, 40):
        with pl.scope():
            layer_window_cache = pl.slice(window_cache_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_window_cache_scale = pl.slice(window_cache_scale_pool, [window_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM // WINDOW_CACHE_GROUP], [layer * window_blocks_per_layer, 0, 0, 0])
            layer_source_id = 3
            layer_compressed_cache = pl.slice(compressed_cache_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // 2], [layer_source_id * compressed_blocks_per_source, 0, 0, 0])
            layer_compressed_cache_scale = pl.slice(compressed_cache_scale_pool, [compressed_blocks_per_source, BLOCK_SIZE, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], [layer_source_id * compressed_blocks_per_source, 0, 0, 0])
            layer_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer * 3])
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer * MIX_HC])
            layer_norm_weight: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [layer * D])
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[layer]
            layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [layer * (D // MX_GROUP), 0])
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [layer * Q_LORA])
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[layer]
            layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [layer * (Q_LORA // MX_GROUP), 0])
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[layer]
            layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [layer * (D // MX_GROUP), 0])
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [layer * HEAD_DIM])
            layer_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [layer * LOCAL_H])
            layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
            layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[layer]
            layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
            if layer == 20:
                full_index_source_id = 3
                full_index_cache = pl.slice(index_cache_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2], [full_index_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_cache_scale = pl.slice(index_cache_scale_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], [full_index_source_id * index_blocks_per_source, 0, 0, 0])
                full_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [3 * (Q_LORA // MX_GROUP), 0])
                full_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [layer_source_id * HEAD_DIM])
                full_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[full_index_source_id]
                full_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [full_index_source_id * INDEX_DIM])
                full_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[full_index_source_id]
                full_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[full_index_source_id]
                decode_c1a_full(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    layer_compressed_cache, layer_compressed_cache_scale,
                    host_token_to_req_indices, token_to_req_indices, c1a_compressed_lens,
                    full_index_cache, full_index_cache_scale,
                    c1a_index_block_table,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    c1a_compressor_wkv, full_compressor_norm,
                    c1a_compressed_slots,
                    full_index_wk, full_index_norm,
                    full_index_wq_b, full_index_wq_b_scale, full_index_weights_proj,
                    topk_indices, pool_block_ids, pool_block_counts,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            elif (layer - 20) % 4 == 0:
                reindex_index_source_id = (layer - 24) // 4 + 4
                reindex_index_cache = pl.slice(index_cache_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // 2], [layer_source_id * index_blocks_per_source, 0, 0, 0])
                reindex_index_cache_scale = pl.slice(index_cache_scale_pool, [index_blocks_per_source, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], [layer_source_id * index_blocks_per_source, 0, 0, 0])
                reindex_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [((layer - 24) // 4 + 4) * (Q_LORA // MX_GROUP), 0])
                reindex_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN] = index_wq_b[reindex_index_source_id]
                reindex_index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[reindex_index_source_id]
                decode_c1a_reindex(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    layer_compressed_cache, layer_compressed_cache_scale,
                    token_to_req_indices, c1a_compressed_lens,
                    reindex_index_cache, reindex_index_cache_scale,
                    c1a_index_block_table,
                    pool_block_ids, pool_block_counts,
                    reindex_index_wq_b, reindex_index_wq_b_scale, reindex_index_weights_proj,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            else:
                decode_c1a_reuse(
                    x_hc, pre_mix,
                    layer_hc_fn, layer_hc_scale, layer_hc_base,
                    layer_norm_weight,
                    layer_wq_a, layer_wq_a_scale,
                    layer_q_norm,
                    layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale,
                    layer_kv_norm,
                    layer_sink,
                    layer_wo_a, layer_wo_a_scale,
                    layer_wo_b, layer_wo_b_scale,
                    compressed_token_rope_cos, compressed_token_rope_sin,
                    window_slots, window_indices,
                    layer_window_cache, layer_window_cache_scale,
                    layer_compressed_cache, layer_compressed_cache_scale,
                    topk_indices,
                    attention_hidden, attention_pre_mix, attention_count,
                )
            ffn_routed_base = layer * N_LOCAL_EXPERTS
            ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
            ffn_hc_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer * 3])
            ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer * MIX_HC])
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [layer * D])
            ffn_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_weight, [N_EXPERTS, D], [layer * N_EXPERTS, 0])
            ffn_correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [layer * N_EXPERTS])
            ffn_routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w1_scale, [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0])
            ffn_routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w2_scale, [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], [layer * (N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0])
            ffn_routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [ffn_routed_base, 0, 0])
            ffn_routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w3_scale, [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (N_LOCAL_EXPERTS * (D // MX_GROUP)), 0])
            ffn_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[layer]
            ffn_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
            ffn_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[layer]
            ffn_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [layer * (MOE_INTER // MX_GROUP), 0])
            ffn_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[layer]
            ffn_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
            force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
            moe(
                attention_hidden, attention_pre_mix,
                ffn_hc_fn, ffn_hc_scale, ffn_hc_base,
                layer_ffn_norm,
                ffn_gate_weight, ffn_correction_bias,
                ffn_routed_w1, ffn_routed_w1_scale,
                ffn_routed_w2, ffn_routed_w2_scale,
                ffn_routed_w3, ffn_routed_w3_scale,
                mxfp4_pair_lut,
                ffn_shared_w1, ffn_shared_w1_scale,
                ffn_shared_w2, ffn_shared_w2_scale,
                ffn_shared_w3, ffn_shared_w3_scale,
                pre_mix, x_mixed, x_hc,
                recv_meta, recv_x, recv_scale,
                recv_weights, recv_routes,
                arrived, data_arrived,
                routed_output, combine_arrived,
                local_moe_tokens, rank, layer + 1,
                force_topk,
            )
    clear_moe_signals(x_hc, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    return x_hc


@pl.jit(auto_scope=False)
def l2_decode_fwd(
    engram_lookup: pl.Tensor[[2, LOCAL_T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
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
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
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
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
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
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
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
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[1, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[1, 1], pl.INT32],
):
    compressed_token_rope_cos = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_token_rope_sin = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    encoder_hidden, encoder_pre_mix, token_cos, token_sin = decode_encoder(
        engram_lookup, engram_wkv, engram_wkv_scale, engram_weight,
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, q_norm_weight, wq_b,
        wkv, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b,
        freqs_cos, freqs_sin,
        compressed_token_rope_cos, compressed_token_rope_sin,
        window_slots, window_indices,
        window_cache_pool, window_cache_scale_pool,
        compressed_cache_pool, compressed_cache_scale_pool,
        token_to_req_indices, host_token_to_req_indices, position_ids,
        index_cache_pool, index_cache_scale_pool,
        c2a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        query_start_loc, state_block_table, state_cache_pool,
        compressor_norm_weight,
        c2a_compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w2, routed_w3, mxfp4_pair_lut,
        shared_w1, shared_w2, shared_w3,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
    )
    result = decode_decoder(
        encoder_hidden, encoder_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, q_norm_weight, wq_b,
        wkv, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b,
        token_cos, token_sin,
        window_slots, window_indices,
        window_cache_pool, window_cache_scale_pool,
        compressed_cache_pool, compressed_cache_scale_pool,
        host_token_to_req_indices, token_to_req_indices, position_ids,
        index_cache_pool, index_cache_scale_pool,
        c1a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        query_start_loc, state_block_table, state_cache_pool,
        compressor_norm_weight,
        c1a_compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w2, routed_w3, mxfp4_pair_lut,
        shared_w1, shared_w2, shared_w3,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale,
        routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
    )
    final_logits, final_sampled_ids = final_output(
        result, encoder_pre_mix, final_norm_weight, lm_head_weight,
        logit_row_indices, logits, sampled_ids,
        hidden_window, hidden_done, logits_window, logits_done,
        rank, 0, 1, 1,
    )
    return final_logits, final_sampled_ids


@pl.jit.host
def l3_decode_fwd(
    engram_lookup: pl.Tensor[[EP_SIZE, 2, LOCAL_T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[EP_SIZE, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[EP_SIZE, 2, HC_MULT, D], pl.FP32],
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
    wo_a: pl.Tensor[[EP_SIZE, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[EP_SIZE, 2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[EP_SIZE, 2, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
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
    host_token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
    index_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, B_DYN, FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[EP_SIZE, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[
        pl.Tensor[[EP_SIZE, FWD_STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
    ],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[EP_SIZE, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
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
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    """Run SWA with base RoPE and C2A/C1A with compressed-profile RoPE.

    freqs_cos/sin are resident full FP32 half-width tables with profile order
    base, compressed. Their row capacity is independent of T and must cover
    all active absolute positions. The encoder gathers token rows at p and
    C2A publication rows at p-1 for completed pairs, identity otherwise, once.
    C1A attention and publication reuse compressed token rows at p, including
    index keys. Shared row buffers remain live through decoder completion.

    The c2a_* and c1a_* metadata address separate compression families. At
    position p, their visible lengths include the current publication and
    equal (p+1)//2 and p+1. Incomplete C2A pairs use write slot -1.
    Page ids and slots are relative to each source's slice of the packed
    cache pools, with BLOCK_SIZE compressed rows per page. Sources within
    one family share a page mapping; the two families may use different
    page ids and table widths. Index and main KV caches share that mapping.
    Visible lengths are derived once on device from position_ids and reused
    across layers. Host paging helpers lower allocator tables into slots
    without allocating physical pages.

    token_to_req_indices is the device token-to-request map for both families.
    host_token_to_req_indices is its separate CPU INT32 snapshot for C1A graph construction.
    Keep both snapshots unchanged until the call completes. Rebuild the HBG graph
    when request-run boundaries change; captured graphs cannot update that topology.
    query_start_loc contains packed request offsets. Padding uses -1 for
    positions, request IDs and physical rows, and zero for visible lengths.
    Prefill uses the same named family tables and slots and derives lengths
    from separate encoder/replay token sequences. state_block_table contains
    source-local block IDs shared by the three C2A sources; block zero is valid,
    -1 is inactive.

    Window pools contain 40 equal layer slices. Compressed and index-key pools
    contain four M.kv_source_layer_ids slices; index-query weights retain eight
    M.index_source_layer_ids slices. C1A Reindex reads layer-20 keys. C2A state
    has three source slices, each with caller-owned request slots. Native
    prefill caches alias these pools by flattening their source/page dimensions.

    attention_num_tokens is per rank; TP peers supply equal group counts.
    TP1 uses padded 80-row local x_hc/pre_mix slabs, with up to 72 active rows.
    Idle ranks use zero counts, -1 publication slots and empty query ranges;
    they still participate in EP collectives. Each new token starts from its
    embedding and identity pre-mix. After layer 39, final HC collapse and
    RMSNorm feed the TP-sharded LM head and greedy sampler. logit_row_indices
    selects up to MAX_LOGIT_ROWS rank-local rows; -1 produces zero logits and
    sampled ID zero, which the caller ignores, including on idle ranks.
    x_hc/pre_mix retain the final layer state; logits and sampled_ids are outputs.

    Retain cache allocations and page ownership until runtime completion, then
    commit lengths before the next call. Completed forwards clear MoE signals;
    use persistent=True with reset_persistent_windows=False for serialized calls.
    Reset a reused request slot in all C2A sources after its users complete. Restore or reset mutated state before retrying failed
    calls. Engram consumes host lookup rows; vision and DSpark require no inputs here.
    """
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    pre_mix.bind_dynamic(1, LOCAL_T_DYN)
    window_cache_pool.bind_dynamic(1, FWD_ORI_BLOCKS_DYN)
    window_cache_scale_pool.bind_dynamic(1, FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, FWD_CMP_BLOCKS_DYN)
    compressed_cache_scale_pool.bind_dynamic(1, FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, FWD_INDEX_BLOCKS_DYN)
    state_cache_pool.bind_dynamic(1, FWD_STATE_BLOCKS_DYN)
    c2a_compressed_slots.bind_dynamic(1, T_DYN)
    c1a_compressed_slots.bind_dynamic(1, T_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
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
    head_hidden_buffer = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    head_logits_done_buffer = pld.alloc_window_buffer([1, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
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
        head_hidden = pld.window(head_hidden_buffer, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [1, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        head_logits_done = pld.window(head_logits_done_buffer, [1, 1], dtype=pl.INT32)
        l2_decode_fwd(
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            x_hc[rank], pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], q_norm_weight[rank], wq_b[rank],
            wkv[rank], kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank],
            freqs_cos[rank], freqs_sin[rank],
            window_slots[rank], window_indices[rank],
            window_cache_pool[rank], window_cache_scale_pool[rank],
            compressed_cache_pool[rank], compressed_cache_scale_pool[rank],
            host_token_to_req_indices[rank], token_to_req_indices[rank], position_ids[rank],
            index_cache_pool[rank], index_cache_scale_pool[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank],
            c2a_compressor_wkv[rank], c2a_compressor_wgate[rank], c1a_compressor_wkv[rank],
            query_start_loc[rank], state_block_table[rank], state_cache_pool[rank],
            compressor_norm_weight[rank],
            c2a_compressed_slots[rank], c1a_compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_weights_proj[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w2[rank], routed_w3[rank], mxfp4_pair_lut[rank],
            shared_w1[rank], shared_w2[rank], shared_w3[rank],
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            moe_num_tokens, attention_num_tokens, rank,
            wq_a_scale[rank], wq_b_scale[rank], wkv_scale[rank], wo_b_scale[rank],
            index_wq_b_scale[rank],
            routed_w1_scale[rank], routed_w2_scale[rank], routed_w3_scale[rank],
            shared_w1_scale[rank], shared_w2_scale[rank], shared_w3_scale[rank],
            final_norm_weight[rank], lm_head_weight[rank],
            logit_row_indices[rank],
            logits[rank], sampled_ids[rank],
            head_hidden, head_hidden_done, head_logits, head_logits_done,
            device=rank,
        )
    return logits, sampled_ids


import inspect
import torch
from rope_tables import prepare_forward_rope_tables
from utils import paged_slots, window_metadata


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
        "ROPE_ROWS_DYN": FIXTURE_SEQUENCE,
        "LOCAL_T_DYN": MOE_TOKENS,
        "Q_START_DYN": FIXTURE_BATCH + 1,
        "T_DYN": FIXTURE_TOKENS,
        "FWD_ORI_BLOCKS_DYN": N_LAYERS * FIXTURE_BATCH,
        "FWD_CMP_BLOCKS_DYN": KV_SOURCE_COUNT * FIXTURE_BATCH,
        "FWD_INDEX_BLOCKS_DYN": KV_SOURCE_COUNT * FIXTURE_BATCH,
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
    from golden import TensorSpec
    from decode_moe import _route_bias
    from expert_routed import build_mxfp4_pair_lut

    signature = inspect.signature(l3_decode_fwd._func)
    specs = []
    token_to_req_indices = torch.arange(FIXTURE_BATCH, dtype=torch.int32).repeat_interleave(FIXTURE_SEQUENCE)
    position_ids = torch.arange(FIXTURE_SEQUENCE, dtype=torch.int32).repeat(FIXTURE_BATCH)
    query_starts = torch.arange(0, FIXTURE_TOKENS + 1, FIXTURE_SEQUENCE, dtype=torch.int32)
    freqs_cos, freqs_sin = prepare_forward_rope_tables(FIXTURE_SEQUENCE)
    rope_tables = {"freqs_cos": freqs_cos, "freqs_sin": freqs_sin}
    window_table = torch.arange(FIXTURE_BATCH, dtype=torch.int32).reshape(FIXTURE_BATCH, 1)
    c2a_table = window_table.flip(0)
    c1a_table = torch.cat((window_table, torch.full_like(window_table, -1)), dim=1)
    window_slots, window_indices, _ = window_metadata(position_ids, token_to_req_indices, window_table)
    metadata_rows = {"window_slots": window_slots, "window_indices": window_indices}
    for family, ratio, table in (("c2a", 2, c2a_table), ("c1a", 1, c1a_table)):
        slots = paged_slots(position_ids, token_to_req_indices, table, BLOCK_SIZE, ratio, True)
        metadata_rows[family + "_compressed_slots"] = slots
        metadata_rows[family + "_index_block_table"] = table
    for name in l3_decode_fwd.param_names:
        annotation = signature.parameters[name].annotation
        shape = [_fixture_extent(dim) for dim in annotation.shape]
        dtype = _torch_dtype(annotation.dtype)
        init_value = None
        if name in ("host_token_to_req_indices", "token_to_req_indices"):
            init_value = token_to_req_indices.repeat(EP_SIZE, 1)
        elif name == "position_ids":
            init_value = position_ids.repeat(EP_SIZE, 1)
        elif name in rope_tables:
            init_value = rope_tables[name].unsqueeze(0).repeat(EP_SIZE, 1, 1, 1)
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
        elif name in ("moe_num_tokens", "attention_num_tokens"):
            init_value = torch.full(shape, FIXTURE_TOKENS, dtype=torch.int32)
        elif name == "logit_row_indices":
            init_value = torch.full(shape, -1, dtype=torch.int32)
            init_value[:, :FIXTURE_BATCH] = query_starts[1:] - 1
        elif name == "final_norm_weight":
            init_value = 1.0
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
    from golden import normalize_config, run
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=[2, 4, 8])
    parser.add_argument("--experts-per-rank", type=int, default=N_LOCAL_EXPERTS, help="routed experts on each EP rank")
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(EP_SIZE)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-scope-stats", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("this forward entry requires hardware; --sim is unsupported")
    if (args.ep, args.experts_per_rank) != (EP_SIZE, N_LOCAL_EXPERTS):
        parser.error("--ep/--experts-per-rank are import-time configuration; pass them on the Python command line")
    devices = [int(device) for device in args.device.split(",") if device]
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE:
        parser.error(f"need exactly {EP_SIZE} distinct devices, got {devices}")
    specs = build_tensor_specs()
    config = dict(
        backend_type=args.backend_type,
        execution_mode=args.execution_mode,
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
        cfg = normalize_config(config)
        compiled = DistributedCompiledProgram.from_dir(
            result.work_dir, backend_type=cfg.backend_type, execution_mode=cfg.execution_mode,
            distributed_config=cfg.distributed_config,
        )
        replicated = ('moe_num_tokens', 'attention_num_tokens', 'host_token_to_req_indices')
        host_inputs = {spec.name: spec.create_tensor().share_memory_() for spec in specs if spec.name in replicated}
        output_names = ("logits", "sampled_ids")
        host_outputs = {spec.name: spec.create_tensor().share_memory_() for spec in specs if spec.name in output_names}
        upload_buffer = torch.empty(64 * 1024**2, dtype=torch.uint8).share_memory_()
        inherited = [*host_inputs.values(), *host_outputs.values(), upload_buffer]
        with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False, inherited_host_tensors=inherited) as runtime:
            parameters = compiled._get_metadata()[0]
            arguments = upload_forward_inputs(runtime, specs, parameters, len(devices), host_inputs, upload_buffer)
            runtime(*arguments, config=cfg)
            stacked = dict(zip((spec.name for spec in specs), arguments, strict=True))
            for name, tensor in host_outputs.items():
                runtime.copy_stacked_from(stacked[name], tensor)
        row_spec = next(spec for spec in specs if spec.name == "logit_row_indices")
        inactive_rows = row_spec.create_tensor() < 0
        if torch.count_nonzero(host_outputs["logits"][inactive_rows]).item():
            raise RuntimeError("inactive decode logit rows must be zero")
        if torch.count_nonzero(host_outputs["sampled_ids"][inactive_rows]).item():
            raise RuntimeError("inactive decode sampled-id rows must be zero")
        if torch.count_nonzero(host_outputs["sampled_ids"][:, :, 1:]).item():
            raise RuntimeError("decode sampled-id padding must be zero")
        print("[FWD] Execution and head output checks passed; numerical accuracy was not validated.", flush=True)
    if not result.passed:
        raise SystemExit(result.error or 1)
