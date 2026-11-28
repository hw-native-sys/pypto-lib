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
"""Run DeepSeek V4.1 Flash CED prefill with data, context, and expert parallelism."""

import argparse
from numbers import Integral
from typing import Mapping, Sequence

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
import torch

import config as C

# Set the forward shapes before importing operators that capture them.
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--compile-only", action="store_true")
_shape_parser.add_argument("--weight-bank-size", type=int, choices=(1, C.FLASH.num_hidden_layers))
_shape_parser.add_argument("--ep", type=int, default=2, choices=[2, 4, 8])
_shape_parser.add_argument("--capacity", type=int, default=128)
_shape_parser.add_argument("--moe-tokens", type=int, default=1024)
_shape_parser.add_argument("--dp", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
C.EP_SIZE = _shape_options.ep
_MAX_CHUNK_TOKENS = 4096
CAPACITY = _shape_options.capacity
CP_SIZE = C.CP_SIZE
LOCAL_CAPACITY = 2 * ((CAPACITY + 2 * CP_SIZE - 1) // (2 * CP_SIZE))
C.DP_SIZE = C.EP_SIZE // CP_SIZE
DP_SIZE = C.DP_SIZE
if C.EP_SIZE % CP_SIZE or _shape_options.dp not in (None, DP_SIZE):
    _shape_parser.error("prefill requires EP = CP * DP")
if C.EP_SIZE % C.LM_HEAD_TP_SIZE:
    _shape_parser.error("EP must be divisible by LM head TP")
if not 16 <= CAPACITY <= _MAX_CHUNK_TOKENS or CAPACITY % 16:
    _shape_parser.error(f"--capacity must be between 16 and {_MAX_CHUNK_TOKENS} and a multiple of 16")
C.set_moe_tokens(min((LOCAL_CAPACITY + 15) // 16 * 16, _shape_options.moe_tokens))

from config import (
    FLASH as M,
    B_DYN, CACHE_ROWS_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, ORI_BLOCKS_DYN,
    Q_START_DYN, STATE_BLOCKS_DYN,
    COMPRESSED_CACHE_BYTES, INDEX_CACHE_GROUP, STATE_CAPACITY, WINDOW_CACHE_BYTES,
    DECODE_INPUT_MAX_TOKENS, LM_HEAD_TP_SIZE, MOE_TOKENS, PREFILL_MAX_TOKENS,
    DSPARK_DRAFT_LAYERS, DSPARK_LAST_TARGET_LAYER, DSPARK_TARGET_LAYER_IDS,
)
from prefill_cp_allgather import prefill_cp_gather_hc
from hc import hc_head
from input_pack import gather_rows, pack_x_hc
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows, prepare_forward_rope_tables
from engram import (
    ENGRAM_K, ENGRAM_ROWS, ENGRAM_SHARD_ROWS, KV_OUT, N_HASH_COLS,
    engram, zero_engram_window_sources,
)
from utils import paged_slots, window_metadata
from lm_head import MAX_LOGIT_ROWS, clear_lm_head_signals, lm_head
from gate import gather_routing_bias
from sample import sample
from mega_moe import workspace_bytes
from prefill_moe import FORWARD_LOCAL_EXPERTS as LOCAL_EXPERTS, make_prefill_moe
from compressor import STATE_WIDTH
from prefill_c1a_full import prefill_publish_decoder, prefill_c1a_full, prefill_c1a_full_cp
from prefill_swa import prefill_swa, prefill_swa_cp
from o_proj import retire_prefill_signals
from prefill_c2a_full import prefill_c2a_full, prefill_c2a_full_cp
from prefill_c1a_reindex import prefill_c1a_reindex, prefill_c1a_reindex_cp
from prefill_c1a_reuse import prefill_c1a_reuse, prefill_c1a_reuse_cp
from rmsnorm import rms_norm
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS

# Dynamic shape variables.
BIAS_DYN = pl.dynamic("PREFILL_FORWARD_BIAS_DYN")
PAGES = pl.dynamic("PREFILL_HISTORY_PAGES")
WINDOW_PAGES = pl.dynamic("PREFILL_WINDOW_PAGES")
C2A_TABLE_DYN = pl.dynamic("PREFILL_C2A_TABLE")
C1A_TABLE_DYN = pl.dynamic("PREFILL_C1A_TABLE")

# Model config
RANKS = C.EP_SIZE
D = M.hidden_size
DSPARK_MAIN_IN = DSPARK_DRAFT_LAYERS * D
HC_MULT = M.hc_mult
VOCAB = M.vocab_size
N_LAYERS = M.num_hidden_layers
C2A_SOURCE_COUNT = sum(M.compress_ratios[source] == 2 for source in M.kv_source_layer_ids)
KV_SOURCE_COUNT = len(M.kv_source_layer_ids)
INDEX_SOURCE_COUNT = len(M.index_source_layer_ids)
MX_GROUP = M.weight_block_size[0]
RECV_MAX = C.MOE_RECV_MAX
HC_DIM = M.hc_dim
HEAD_DIM = M.head_dim
INDEX_DIM = M.index_head_dim
INDEX_H = M.index_n_heads
INDEX_TOPK = M.index_topk
INDEX_POOL_BLOCKS = M.candidate_topk_blocks
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * M.o_lora_rank
MIX_HC = M.mix_hc
MOE_INTER = M.moe_intermediate_size
N_EXPERTS = RANKS * LOCAL_EXPERTS
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
TOPK = M.num_experts_per_tok
ROPE_DIM = M.qk_rope_head_dim

# Forward configuration
if N_EXPERTS % 16:
    _shape_parser.error("--experts-per-rank must give a total expert count aligned to 16")
_standalone_smoke = __name__ == "__main__" and not _shape_options.compile_only
WEIGHT_BANK_SIZE = _shape_options.weight_bank_size or (1 if _standalone_smoke else N_LAYERS)
FINAL_SLOT = 2 * CAPACITY
_, prefill_moe_slabs = make_prefill_moe(LOCAL_EXPERTS)
MOE_WORKSPACE_BYTES = workspace_bytes(RANKS, LOCAL_EXPERTS, MOE_TOKENS, TOPK, D, MOE_INTER)

SMOKE_RING_HEAP_BYTES = 4 * 1024**3


@pl.jit.inline(auto_scope=False)
def restore_window_tail(
    cache: pl.Tensor,
    source_slots: pl.Tensor,
    destination_slots: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
):
    """Copy final prefill rows from transient scratch into private window rings."""
    pages = pl.tensor.dim(cache, 1)
    rows = pages * 128
    flat = pl.reshape(cache, [40 * rows, WINDOW_CACHE_BYTES])
    tokens = pl.tensor.dim(destination_slots, 0)
    with pl.spmd(32, name_hint="prefill_window_tail", deps=[ready]) as restored:
        worker = pl.tile.get_block_idx()
        for work in pl.range(worker, 40 * tokens, 32):
            layer = work // tokens
            token = work % tokens
            destination = pl.read(destination_slots, [token])
            if destination >= 0:
                source = pl.read(source_slots, [token])
                payload = pl.load(flat, [layer * rows + source, 0], [1, WINDOW_CACHE_BYTES], clamp=True)
                pl.store(payload, [layer * rows + destination, 0], flat)
    return restored


@pl.jit.inline(auto_scope=False)
def prefill_attention(
    hc_input: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[CAPACITY], pl.INT64],
    q_window_indices: pl.Tensor[[LOCAL_CAPACITY, 128], pl.INT32],
    q_rope_cos: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_request_ids: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    kv_restore_indices: pl.Tensor[[CAPACITY], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, CACHE_ROWS_DYN, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    token_to_req_indices: pl.Tensor[[CAPACITY], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[CAPACITY], pl.INT32],
    compressed_rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    pool_block_ids: pl.Tensor[[LOCAL_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    topk_indices: pl.Tensor[[LOCAL_CAPACITY, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_hidden: pl.Tensor[[CAPACITY, D], pl.BF16],
    hc_output: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    local_tokens: pl.Scalar[pl.INT32],
    cp_group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
    c1a_phase: pl.constexpr,
):
    if not c1a_phase:
        if mode == 0:
            prefill_swa(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_window_indices,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
        elif mode == 1:
            prefill_c2a_full(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                compressor_wkv, compressor_wgate, compressor_norm_weight,
                index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
                q_window_indices, q_request_ids, q_compressed_lens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots,
                token_to_req_indices, position_ids, query_start_loc,
                state_block_table, state_cache,
                compressed_cache, index_cache, index_cache_scale, c2a_index_block_table,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 2:
            prefill_c1a_reuse(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_window_indices, topk_indices,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
    else:
        if mode == 3:
            prefill_c1a_full(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                compressor_wkv, compressor_norm_weight,
                index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
                q_window_indices, q_request_ids, q_compressed_lens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices, pool_block_ids, pool_block_counts,
                publish_compressed=False,
            )
        elif mode == 4:
            prefill_c1a_reindex(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                index_wq_b, index_wq_b_scale, index_weights_proj,
                q_window_indices, q_request_ids, q_compressed_lens,
                pool_block_ids, pool_block_counts,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 5:
            prefill_c1a_reuse(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_window_indices, topk_indices,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
    return hc_output, hc_next_pre_mix


@pl.jit.inline(auto_scope=False)
def prefill_attention_cp(
    hc_input: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[CAPACITY], pl.INT64],
    q_window_indices: pl.Tensor[[LOCAL_CAPACITY, 128], pl.INT32],
    q_rope_cos: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_CAPACITY, ROPE_DIM // 2], pl.FP32],
    q_request_ids: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    kv_restore_indices: pl.Tensor[[CAPACITY], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, CACHE_ROWS_DYN, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    token_to_req_indices: pl.Tensor[[CAPACITY], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, CACHE_ROWS_DYN, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[CAPACITY], pl.INT32],
    compressed_rope_cos: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    pool_block_ids: pl.Tensor[[LOCAL_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[LOCAL_CAPACITY], pl.INT32],
    topk_indices: pl.Tensor[[LOCAL_CAPACITY, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_hidden: pl.Tensor[[CAPACITY, D], pl.BF16],
    hc_output: pl.Tensor[[LOCAL_CAPACITY, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_CAPACITY, HC_MULT], pl.FP32],
    local_tokens: pl.Scalar[pl.INT32],
    cp_group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
    c1a_phase: pl.constexpr,
):
    if not c1a_phase:
        if mode == 0:
            prefill_swa_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_rope_cos, q_rope_sin, q_window_indices, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
        elif mode == 1:
            prefill_c2a_full_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                compressor_wkv, compressor_wgate, compressor_norm_weight,
                index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
                q_rope_cos, q_rope_sin, q_window_indices, q_request_ids, q_compressed_lens, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots,
                token_to_req_indices, position_ids, query_start_loc,
                state_block_table, state_cache,
                compressed_cache, index_cache, index_cache_scale, c2a_index_block_table,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 2:
            prefill_c1a_reuse_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_rope_cos, q_rope_sin, q_window_indices, topk_indices, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
    else:
        if mode == 3:
            prefill_c1a_full_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                compressor_wkv, compressor_norm_weight,
                index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
                q_rope_cos, q_rope_sin, q_window_indices, q_request_ids, q_compressed_lens, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_rope_cos, compressed_rope_sin, c2a_compressed_slots,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices, pool_block_ids, pool_block_counts,
                publish_compressed=False,
            )
        elif mode == 4:
            prefill_c1a_reindex_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                index_wq_b, index_wq_b_scale, index_weights_proj,
                q_rope_cos, q_rope_sin, q_window_indices, q_request_ids, q_compressed_lens,
                pool_block_ids, pool_block_counts,
                local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache, index_cache, index_cache_scale, c1a_index_block_table,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
                topk_indices,
            )
        elif mode == 5:
            prefill_c1a_reuse_cp(
                hc_input, hc_pre_mix,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                w_attn_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
                wkv, wkv_scale, kv_norm_weight, attn_sink,
                wo_a, wo_a_scale, wo_b, wo_b_scale,
                q_rope_cos, q_rope_sin, q_window_indices, topk_indices, local_tokens,
                rope_cos, rope_sin, window_slots, num_tokens, window_cache,
                compressed_cache,
                output_window, output_arrived,
                kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
                kv_hidden, hc_output, hc_next_pre_mix,
            )
    return hc_output, hc_next_pre_mix


prefill_attention_entry = prefill_attention_cp if CP_SIZE > 1 else prefill_attention


@pl.jit.inline(auto_scope=False)
def prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    engram_ids: pl.Tensor[[2 * CAPACITY * N_HASH_COLS, 1], pl.INT32],
    engram_scales: pld.DistributedTensor[[2 * ENGRAM_SHARD_ROWS, 8], pl.UINT8],
    engram_payload: pld.DistributedTensor[[2 * ENGRAM_SHARD_ROWS, 256], pl.UINT8],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    moe_workspace: pld.DistributedTensor,
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hc_window: pld.DistributedTensor[[LOCAL_CAPACITY, HC_DIM + 16], pl.FP32],
    hc_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run both CED stages and retire communication signals after the decoder."""
    correction_bias.bind_dynamic(0, BIAS_DYN)
    compressed_cache.bind_dynamic(1, PAGES)
    index_cache.bind_dynamic(1, PAGES)
    index_scales.bind_dynamic(1, PAGES)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, ROPE_ROWS_DYN)

    rope_cos = pl.create_tensor([3, CAPACITY, 32], dtype=pl.FP32)
    rope_sin = pl.create_tensor([3, CAPACITY, 32], dtype=pl.FP32)
    encoder_count = pl.read(counts, [0])
    decoder_count = pl.read(counts, [1])
    materialize_rope_rows(
        freqs_cos[0], freqs_sin[0], position_ids[0], encoder_count, rope_cos[0], rope_sin[0], False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, rope_cos[1], rope_sin[1], False,
        worker_limit=32,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[1], decoder_count, rope_cos[2], rope_sin[2], False,
        worker_limit=32,
    )

    compressed_cos = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    compressed_sin = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, compressed_cos, compressed_sin, True,
        worker_limit=32,
    )

    q_rows = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    restore_indices = pl.create_tensor([2, CAPACITY], dtype=pl.INT32)
    q_rope_cos = pl.create_tensor([3, LOCAL_CAPACITY, 32], dtype=pl.FP32)
    q_rope_sin = pl.create_tensor([3, LOCAL_CAPACITY, 32], dtype=pl.FP32)
    q_window_indices = pl.create_tensor([2, LOCAL_CAPACITY, 128], dtype=pl.INT32)
    q_request_ids = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    q_compressed_lens = pl.create_tensor([2, LOCAL_CAPACITY], dtype=pl.INT32)
    for stage_index in pl.range(2):
        count = encoder_count
        if stage_index == 1:
            count = decoder_count
        chunk = (count + 2 * CP_SIZE - 1) // (2 * CP_SIZE)
        cp_rank = rank % CP_SIZE
        head_start = cp_rank * chunk
        tail_start = (2 * CP_SIZE - 1 - cp_rank) * chunk
        head_count = pl.max(0, pl.min(chunk, count - head_start))
        tail_count = pl.max(0, pl.min(chunk, count - tail_start))
        with pl.spmd(1, name_hint="prefill_cp_rows") as rows_ready:
            writer = pl.tile.get_block_idx()
            for row in pl.range(writer, LOCAL_CAPACITY):
                source = pl.cast(-1, pl.INT32)
                if row < head_count:
                    source = pl.cast(head_start + row, pl.INT32)
                elif row < head_count + tail_count:
                    source = pl.cast(tail_start + row - head_count, pl.INT32)
                pl.write(q_rows, [stage_index, row], source)
                request = pl.cast(0, pl.INT32)
                length = pl.cast(0, pl.INT32)
                if source >= 0:
                    request = pl.read(token_to_req_indices, [stage_index, source])
                    position = pl.read(position_ids, [stage_index, source])
                    length = pl.cast(position + 1, pl.INT32)
                    if stage_index == 0:
                        length = pl.cast(length // 2, pl.INT32)
                pl.write(q_request_ids, [stage_index, row], request)
                pl.write(q_compressed_lens, [stage_index, row], length)
            for row in pl.range(CAPACITY):
                restored = pl.cast(-1, pl.INT32)
                if row < count:
                    block = row // chunk
                    owner = block
                    local_offset = row % chunk
                    if block >= CP_SIZE:
                        owner = 2 * CP_SIZE - 1 - block
                        local_offset = chunk + local_offset
                    restored = pl.cast(owner * LOCAL_CAPACITY + local_offset, pl.INT32)
                pl.write(restore_indices, [stage_index, row], restored)
        with pl.spmd(32, name_hint="prefill_cp_windows", deps=[rows_ready]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, LOCAL_CAPACITY, 32):
                source = pl.read(q_rows, [stage_index, row])
                if source >= 0:
                    q_window_indices[stage_index, row:row + 1, :] = window_indices[stage_index, source:source + 1, :]
                else:
                    q_window_indices[stage_index, row:row + 1, :] = pl.full([1, 128], dtype=pl.INT32, value=-1)
        rope_slot = stage_index + 1
        gather_rows(
            rope_cos[rope_slot], q_rows[stage_index], q_rope_cos[rope_slot],
            rows_ready, column_tile=32, refresh_row_ids=False,
        )
        gather_rows(
            rope_sin[rope_slot], q_rows[stage_index], q_rope_sin[rope_slot],
            rows_ready, column_tile=32, refresh_row_ids=False,
        )
        if stage_index == 0:
            gather_rows(rope_cos[0], q_rows[0], q_rope_cos[0], rows_ready, column_tile=32, refresh_row_ids=False)
            gather_rows(rope_sin[0], q_rows[0], q_rope_sin[0], rows_ready, column_tile=32, refresh_row_ids=False)

    hidden_pair = pl.create_tensor([4 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    initial_hidden = hidden_pair[:CAPACITY]
    gather_ready = pl.system.task_dummy(deps=[])
    pack_x_hc(embedding_ids, embedding, initial_hidden, hidden_tile=D)

    pre_pair = pl.create_tensor([4 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_mix = pre_pair[:CAPACITY]
    gather_rows(initial_pre, identity_rows, initial_mix, gather_ready, column_tile=512, refresh_row_ids=False)

    topk_indices = pl.create_tensor([LOCAL_CAPACITY, INDEX_TOPK], dtype=pl.INT32)
    history_positions = pl.tensor.dim(compressed_cache, 1) * 128
    pool_blocks = pl.min(INDEX_POOL_BLOCKS, (history_positions // 8 + 31) // 32 * 32)
    pool_block_ids = pl.create_tensor([LOCAL_CAPACITY, pool_blocks], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([LOCAL_CAPACITY], dtype=pl.INT32)

    attn_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    attn_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_mixed = pl.create_tensor([LOCAL_CAPACITY, D], dtype=pl.BF16)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    local_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    local_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    local_next_hidden = pl.create_tensor([LOCAL_CAPACITY, HC_MULT, D], dtype=pl.FP32)
    local_next_pre = pl.create_tensor([LOCAL_CAPACITY, HC_MULT], dtype=pl.FP32)
    moe_ready = pl.system.task_dummy(deps=[])
    history_pages = pl.tensor.dim(compressed_cache, 1)
    c1a_cache_units = compressed_cache[3:5, :history_pages]
    c1a_cache = pl.reshape(c1a_cache_units, [history_pages, 128, 1, COMPRESSED_CACHE_BYTES])
    c1a_index_units = index_cache[3:5]
    c1a_index = pl.reshape(c1a_index_units, [history_pages, 128, 1, 64])
    c1a_scale_units = index_scales[3:5]
    c1a_scale = pl.reshape(c1a_scale_units, [history_pages, 128, 1, 4])

    for layer in pl.range(N_LAYERS):
        stage = layer // 20
        active = encoder_count
        if stage == 1:
            active = decoder_count
        cp_chunk = (active + 2 * CP_SIZE - 1) // (2 * CP_SIZE)
        cp_rank = rank % CP_SIZE
        cp_head = pl.max(0, pl.min(cp_chunk, active - cp_rank * cp_chunk))
        cp_tail = pl.max(0, pl.min(cp_chunk, active - (2 * CP_SIZE - 1 - cp_rank) * cp_chunk))
        local_active = pl.cast(cp_head + cp_tail, pl.INT32)
        current_slot = (2 * stage + layer % 2) * CAPACITY
        next_slot = (2 * stage + (layer + 1) % 2) * CAPACITY
        hidden = hidden_pair[current_slot : current_slot + CAPACITY]
        delayed_pre = pre_pair[current_slot : current_slot + CAPACITY]
        next_hidden = hidden_pair[next_slot : next_slot + CAPACITY]
        next_pre = pre_pair[next_slot : next_slot + CAPACITY]
        kv_source = pl.min(pl.max((layer - 2) // 6, 0), 2)
        if layer >= 20:
            kv_source = 3
        index_source = kv_source
        if layer >= 24:
            index_source = (layer - 24) // 4 + 4
        c2a_source = pl.min(kv_source, 2)
        weight_layer = layer % WEIGHT_BANK_SIZE
        mix_offset = weight_layer * MIX_HC
        layer_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = hc_attn_fn[mix_offset : mix_offset + MIX_HC]
        hc_scale_offset = weight_layer * 3
        layer_attn_scale: pl.Tensor[[3], pl.FP32] = hc_attn_scale[hc_scale_offset : hc_scale_offset + 3]
        layer_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_attn_base[mix_offset : mix_offset + MIX_HC]
        layer_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = hc_ffn_fn[mix_offset : mix_offset + MIX_HC]
        layer_ffn_scale: pl.Tensor[[3], pl.FP32] = hc_ffn_scale[hc_scale_offset : hc_scale_offset + 3]
        layer_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_ffn_base[mix_offset : mix_offset + MIX_HC]
        norm_offset = weight_layer * D
        layer_attn_norm: pl.Tensor[[D], pl.BF16] = attn_norm_weight[norm_offset : norm_offset + D]
        layer_ffn_norm: pl.Tensor[[D], pl.BF16] = ffn_norm_weight[norm_offset : norm_offset + D]
        q_norm_offset = weight_layer * Q_LORA
        layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = q_norm_weight[q_norm_offset : q_norm_offset + Q_LORA]
        kv_norm_offset = weight_layer * HEAD_DIM
        layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = kv_norm_weight[kv_norm_offset : kv_norm_offset + HEAD_DIM]
        layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[weight_layer]
        hidden_scale_offset = weight_layer * (D // MX_GROUP)
        layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[hidden_scale_offset : hidden_scale_offset + D // MX_GROUP]
        layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[weight_layer]
        query_scale_offset = weight_layer * (Q_LORA // MX_GROUP)
        layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[query_scale_offset : query_scale_offset + Q_LORA // MX_GROUP]
        layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[weight_layer]
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[hidden_scale_offset : hidden_scale_offset + D // MX_GROUP]
        layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[weight_layer]
        output_scale_offset = weight_layer * (LOCAL_O_WIDTH // MX_GROUP)
        layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[output_scale_offset : output_scale_offset + LOCAL_O_WIDTH // MX_GROUP]
        layer_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[weight_layer]
        layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[hidden_scale_offset : hidden_scale_offset + D // MX_GROUP]
        layer_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[weight_layer]
        ffn_scale_offset = weight_layer * (MOE_INTER // MX_GROUP)
        layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[ffn_scale_offset : ffn_scale_offset + MOE_INTER // MX_GROUP]
        layer_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[weight_layer]
        layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[hidden_scale_offset : hidden_scale_offset + D // MX_GROUP]
        sink_offset = weight_layer * LOCAL_H
        layer_attn_sink: pl.Tensor[[LOCAL_H], pl.FP32] = attn_sink[sink_offset : sink_offset + LOCAL_H]
        group_offset = weight_layer * LOCAL_O_GROUPS
        layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = wo_a[group_offset : group_offset + LOCAL_O_GROUPS]
        layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = wo_a_scale[group_offset : group_offset + LOCAL_O_GROUPS]
        layer_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wkv[c2a_source]
        layer_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wgate[c2a_source]
        compressor_norm_offset = kv_source * HEAD_DIM
        layer_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = compressor_norm_weight[compressor_norm_offset : compressor_norm_offset + HEAD_DIM]
        layer_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[index_source]
        index_norm_offset = index_source * INDEX_DIM
        layer_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = index_norm_weight[index_norm_offset : index_norm_offset + INDEX_DIM]
        layer_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = index_wq_b[index_source]
        index_scale_offset = index_source * (Q_LORA // MX_GROUP)
        layer_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[index_scale_offset : index_scale_offset + Q_LORA // MX_GROUP]
        layer_index_weights: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[index_source]
        gate_offset = weight_layer * N_EXPERTS
        layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16] = gate_weight[gate_offset : gate_offset + N_EXPERTS]
        bias_rows = pl.tensor.dim(correction_bias, 0) // (WEIGHT_BANK_SIZE * N_EXPERTS)
        bias_length = bias_rows * N_EXPERTS
        bias_offset = weight_layer * bias_length
        source_bias = correction_bias[bias_offset : bias_offset + bias_length]
        stage_gate_bias = pl.create_tensor([bias_length], dtype=pl.FP32)
        if stage == 0:
            gather_routing_bias(
                source_bias, identity_rows, stage_gate_bias,
                moe_ready, n_experts=N_EXPERTS, refresh_row_ids=False,
            )
        else:
            gather_routing_bias(
                source_bias, replay_rows, stage_gate_bias,
                moe_ready, n_experts=N_EXPERTS, refresh_row_ids=False,
            )
        local_bias_rows = pl.min(bias_rows, LOCAL_CAPACITY)
        layer_gate_bias = pl.create_tensor([local_bias_rows * N_EXPERTS], dtype=pl.FP32)
        gather_routing_bias(
            stage_gate_bias, q_rows[stage], layer_gate_bias,
            moe_ready, n_experts=N_EXPERTS, refresh_row_ids=False,
        )
        expert_offset = weight_layer * LOCAL_EXPERTS
        layer_routed_w1: pl.Tensor[[LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = routed_w1[expert_offset : expert_offset + LOCAL_EXPERTS]
        expert_up_scale_offset = weight_layer * (LOCAL_EXPERTS * (D // MX_GROUP))
        layer_routed_w1_scale: pl.Tensor[[LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[expert_up_scale_offset : expert_up_scale_offset + LOCAL_EXPERTS * (D // MX_GROUP)]
        layer_routed_w2: pl.Tensor[[LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = routed_w2[expert_offset : expert_offset + LOCAL_EXPERTS]
        expert_down_scale_offset = weight_layer * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP))
        layer_routed_w2_scale: pl.Tensor[[LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[expert_down_scale_offset : expert_down_scale_offset + LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)]
        layer_routed_w3: pl.Tensor[[LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = routed_w3[expert_offset : expert_offset + LOCAL_EXPERTS]
        layer_routed_w3_scale: pl.Tensor[[LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[expert_up_scale_offset : expert_up_scale_offset + LOCAL_EXPERTS * (D // MX_GROUP)]
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for all encoder rows before gathering replay rows.
                encoder_hidden = hidden_pair[:CAPACITY]
                encoder_pre = pre_pair[:CAPACITY]
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_head(encoder_hidden, encoder_pre, encoder_collapsed, worker_limit=64)

                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                norm_tokens = pl.tensor.dim(encoder_collapsed, 0)
                rms_norm(encoder_collapsed, layer_attn_norm, encoder_normalized, norm_tokens, width=D, zero_tail=False)
                publisher_ready = prefill_publish_decoder(
                    encoder_normalized, c1a_compressor_wkv, layer_compressor_norm, layer_index_wk, layer_index_norm,
                    rope_cos[1], rope_sin[1], c1a_compressed_slots,
                    c1a_cache, c1a_index, c1a_scale,
                    encoder_count,
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(
                    source_hidden, replay_rows, gathered_hidden,
                    publisher_ready, column_tile=D, refresh_row_ids=False,
                )
                gather_rows(
                    encoder_pre, replay_rows, delayed_pre,
                    publisher_ready, column_tile=512, refresh_row_ids=False,
                )
        if layer == 1 or layer == 14:
            with pl.scope():
                engram_slot = layer // 14
                engram_scale_offset = engram_slot * (ENGRAM_K // 32)
                layer_engram_scale = engram_wkv_scale[engram_scale_offset : engram_scale_offset + ENGRAM_K // 32]
                engram(
                    engram_ids, engram_payload, engram_scales, engram_slot, ENGRAM_SHARD_ROWS,
                    engram_wkv[engram_slot], layer_engram_scale, engram_weight[engram_slot],
                    hidden, hidden, mask_zero_rows=True,
                )
        with pl.scope():
            global_hc = pl.reshape(hidden, [CAPACITY, HC_DIM])
            local_hc = pl.reshape(local_hidden, [LOCAL_CAPACITY, HC_DIM])
            gather_rows(global_hc, q_rows[stage], local_hc, moe_ready, column_tile=D, refresh_row_ids=False)
            gather_rows(delayed_pre, q_rows[stage], local_pre, moe_ready, column_tile=16, refresh_row_ids=False)
            mode = 0
            cache_slot = 0
            rope_slot = 0
            if layer >= 20:
                cache_slot = 3
                rope_slot = 2
                mode = 5
                if layer == 20:
                    mode = 3
                elif layer % 4 == 0:
                    mode = 4
            elif layer >= 2:
                cache_slot = (layer - 2) // 6
                rope_slot = 1
                mode = 2
                if (layer - 2) % 6 == 0:
                    mode = 1
            state_slot = pl.min(cache_slot, 2)

            normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            if layer >= 20:
                prefill_attention_entry(
                    local_hidden, local_pre, layer_attn_fn, layer_attn_scale, layer_attn_base, layer_attn_norm,
                    layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale, layer_kv_norm, layer_attn_sink,
                    layer_wo_a, layer_wo_a_scale, layer_wo_b, layer_wo_b_scale,
                    rope_cos[rope_slot], rope_sin[rope_slot], window_slots[stage], q_window_indices[stage],
                    q_rope_cos[rope_slot], q_rope_sin[rope_slot], q_request_ids[stage], q_compressed_lens[stage],
                    restore_indices[stage], window_cache[layer], c1a_cache, token_to_req_indices[stage],
                    c1a_index, c1a_scale, c2a_index_block_table, c1a_index_block_table,
                    position_ids[stage], compressed_cos, compressed_sin,
                    layer_compressor_wkv, layer_compressor_wgate,
                    query_start_loc[stage], state_block_table, state_cache[state_slot], layer_compressor_norm,
                    c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale,
                    layer_index_weights, pool_block_ids, pool_block_counts, topk_indices,
                    attention_window, attention_arrived,
                    normalized, attn_hidden, attn_pre,
                    local_active, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, active, layer + 1, mode,
                    c1a_phase=True,
                )
            else:
                prefill_attention_entry(
                    local_hidden, local_pre, layer_attn_fn, layer_attn_scale, layer_attn_base, layer_attn_norm,
                    layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                    layer_wkv, layer_wkv_scale, layer_kv_norm, layer_attn_sink,
                    layer_wo_a, layer_wo_a_scale, layer_wo_b, layer_wo_b_scale,
                    rope_cos[rope_slot], rope_sin[rope_slot], window_slots[stage], q_window_indices[stage],
                    q_rope_cos[rope_slot], q_rope_sin[rope_slot], q_request_ids[stage], q_compressed_lens[stage],
                    restore_indices[stage], window_cache[layer], compressed_cache[cache_slot],
                    token_to_req_indices[stage],
                    index_cache[cache_slot], index_scales[cache_slot], c2a_index_block_table, c1a_index_block_table,
                    position_ids[stage], compressed_cos, compressed_sin,
                    layer_compressor_wkv, layer_compressor_wgate,
                    query_start_loc[stage], state_block_table, state_cache[state_slot], layer_compressor_norm,
                    c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale,
                    layer_index_weights, pool_block_ids, pool_block_counts, topk_indices,
                    attention_window, attention_arrived,
                    normalized, attn_hidden, attn_pre,
                    local_active, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, active, layer + 1, mode,
                    c1a_phase=False,
                )
        moe_ready = prefill_moe_slabs(
            attn_hidden, attn_pre,
            layer_ffn_fn, layer_ffn_scale, layer_ffn_base, layer_ffn_norm,
            layer_gate_weight, layer_gate_bias,
            layer_routed_w1, layer_routed_w1_scale, layer_routed_w2, layer_routed_w2_scale,
            layer_routed_w3, layer_routed_w3_scale,
            layer_shared_w1, layer_shared_w1_scale, layer_shared_w2, layer_shared_w2_scale,
            layer_shared_w3, layer_shared_w3_scale,
            local_next_pre, ffn_mixed, local_next_hidden,
            moe_workspace,
            slab_hidden, slab_pre, slab_mixed, slab_rows,
            moe_ready, local_active, capacity=LOCAL_CAPACITY,
        )
        if CP_SIZE > 1:
            moe_ready = prefill_cp_gather_hc(
                local_next_hidden, local_next_pre, restore_indices[stage], next_hidden, next_pre,
                hc_window, hc_arrived, rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, layer + 1, moe_ready,
                group_size=CP_SIZE,
            )
        else:
            with pl.spmd(32, name_hint="prefill_local_hc", deps=[moe_ready]) as copied:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, CAPACITY, 32):
                    next_hidden[row:row + 1, :, :] = local_next_hidden[row:row + 1, :, :]
            with pl.spmd(1, name_hint="prefill_local_pre", deps=[moe_ready]) as pre_copied:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, CAPACITY):
                    for lane in pl.range(HC_MULT):
                        value = pl.read(local_next_pre, [row, lane])
                        pl.write(next_pre, [row, lane], value)
            moe_ready = pl.system.task_dummy(deps=[copied, pre_copied])
        target_layer = layer + 1
        if target_layer >= DSPARK_TARGET_LAYER_IDS[0] and target_layer <= DSPARK_LAST_TARGET_LAYER:
            target_column = (target_layer - DSPARK_TARGET_LAYER_IDS[0]) * D
            for core in pl.spmd(48, name_hint="prefill_dspark_target_hidden", deps=[moe_ready]):
                for token in pl.range(core, CAPACITY, 48):
                    if token < active:
                        hc_row = next_hidden[token : token + 1, 0:HC_MULT, 0:D]
                        hc_streams = pl.reshape(hc_row, [HC_MULT, D])
                        hc_sum = pl.col_sum(hc_streams)
                        hc_mean = pl.mul(hc_sum, 1.0 / HC_MULT)
                        target_bf16 = pl.cast(hc_mean, pl.BF16, mode="rint")
                    else:
                        target_bf16 = pl.full([1, D], dtype=pl.BF16, value=0.0)
                    dspark_target_hidden[token : token + 1, target_column : target_column + D] = target_bf16

    _window_ready = restore_window_tail(window_cache, window_slots[0], window_restore_slots, moe_ready)
    final_hidden = hidden_pair[FINAL_SLOT : FINAL_SLOT + CAPACITY]
    final_pre = pre_pair[FINAL_SLOT : FINAL_SLOT + CAPACITY]
    source_hidden = pl.reshape(final_hidden, [CAPACITY, HC_DIM])
    output_hidden = pl.reshape(x_hc_out, [CAPACITY, HC_DIM])
    gather_rows(source_hidden, identity_rows, output_hidden, gather_ready, column_tile=D, refresh_row_ids=False)
    gather_rows(final_pre, identity_rows, pre_mix_out, gather_ready, column_tile=512, refresh_row_ids=False)
    if CP_SIZE > 1:
        retire_prefill_signals(
            x_hc_out, attention_arrived,
            rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, group_size=CP_SIZE,
        )
        retire_prefill_signals(
            x_hc_out, hc_arrived,
            rank // CP_SIZE * CP_SIZE, rank % CP_SIZE, group_size=CP_SIZE,
        )

    owner_logit_rows = pl.create_tensor([16], dtype=pl.INT32)
    with pl.spmd(1, name_hint="prefill_logit_owners"):
        writer = pl.tile.get_block_idx()
        for row in pl.range(writer, 16):
            source = pl.read(logit_rows, [row])
            owner_source = pl.cast(-1, pl.INT32)
            if source >= 0 and source < CAPACITY:
                owner_source = source
            pl.write(owner_logit_rows, [row], owner_source)

    head_rows = pl.tensor.dim(final_hidden, 0)
    head_collapsed = pl.create_tensor([head_rows, D], dtype=pl.BF16)
    hc_head(final_hidden, final_pre, head_collapsed, worker_limit=64)

    head_normalized = pl.create_tensor([head_rows, D], dtype=pl.BF16)
    rms_norm(head_collapsed, final_norm_weight, head_normalized, head_rows, width=D, zero_tail=False)
    lm_head(
        head_normalized, lm_head_weight, owner_logit_rows, logits,
        head_hidden, head_hidden_done, head_logits, head_logits_done,
        rank, MAX_LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, head_hidden_done, head_logits_done)
    sample(
        logits, sampling_temperatures, sampling_top_ks, sampling_top_ps,
        sampling_seeds, sampling_positions, sampled_ids,
    )
    return logits, sampled_ids


@pl.jit(auto_scope=False)
def l2_prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    engram_ids: pl.Tensor[[2 * CAPACITY * N_HASH_COLS, 1], pl.INT32],
    engram_scales: pld.DistributedTensor[[2 * ENGRAM_SHARD_ROWS, 8], pl.UINT8],
    engram_payload: pld.DistributedTensor[[2 * ENGRAM_SHARD_ROWS, 256], pl.UINT8],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[WEIGHT_BANK_SIZE * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    moe_workspace: pld.DistributedTensor,
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hc_window: pld.DistributedTensor[[LOCAL_CAPACITY, HC_DIM + 16], pl.FP32],
    hc_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[C.LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    return prefill_fwd(
        embedding,
        engram_ids, engram_scales, engram_payload,
        engram_wkv, engram_wkv_scale, engram_weight,
        embedding_ids, initial_pre, identity_rows,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight, wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv,
        compressor_norm_weight,
        index_wk, index_norm_weight,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale,
        routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale,
        shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale,
        shared_w3, shared_w3_scale,
        counts, replay_rows,
        position_ids, freqs_cos, freqs_sin,
        window_slots, window_restore_slots, window_indices,
        c2a_compressed_slots, c1a_compressed_slots,
        token_to_req_indices, query_start_loc,
        c2a_index_block_table, c1a_index_block_table, state_block_table,
        window_cache, compressed_cache, index_cache, index_scales, state_cache,
        final_norm_weight, lm_head_weight, logit_rows,
        sampling_temperatures, sampling_top_ks, sampling_top_ps,
        sampling_seeds, sampling_positions,
        x_hc_out, pre_mix_out, dspark_target_hidden,
        logits, sampled_ids,
        moe_workspace,
        attention_window, attention_arrived, hc_window, hc_arrived,
        head_hidden, head_hidden_done, head_logits, head_logits_done,
        rank,
    )


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_ids: pl.Tensor[[RANKS, 2 * CAPACITY * N_HASH_COLS, 1], pl.INT32],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[RANKS, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS, BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[RANKS, 2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[RANKS, B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[RANKS, B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS, B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_temperatures: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_top_ks: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_top_ps: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_seeds: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_positions: pl.Tensor[[RANKS, 16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[RANKS, CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
):
    """Append a packed CED chunk. Each DP group owns its requests and caches across calls.

    Pass encoder then decoder metadata with absolute positions and source-local cache slots.
    For full replay, use identical encoder and decoder counts and identity replay rows padded with -1.
    Supply base then compressed RoPE tables covering the retained history. Logit rows index decoder replay.
    Keep cache pages and request slots until completion. Restore or reset cache state before retrying.
    """
    correction_bias.bind_dynamic(1, BIAS_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    compressed_cache.bind_dynamic(2, PAGES)
    index_cache.bind_dynamic(2, PAGES)
    index_scales.bind_dynamic(2, PAGES)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    engram_scale_buffer = pld.alloc_window_buffer(
        [2 * ENGRAM_SHARD_ROWS, 8], dtype=pl.UINT8,
        memory_source="host", init="engram_scale",
    )
    engram_payload_buffer = pld.alloc_window_buffer(
        [2 * ENGRAM_SHARD_ROWS, 256], dtype=pl.UINT8,
        memory_source="host", init="engram_payload",
    )
    moe_workspace_buffer = pld.alloc_window_buffer([MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hc_window_buffer = pld.alloc_window_buffer([LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
    hc_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    head_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        engram_scales = pld.window(engram_scale_buffer, [2 * ENGRAM_SHARD_ROWS, 8], dtype=pl.UINT8)
        engram_payload = pld.window(engram_payload_buffer, [2 * ENGRAM_SHARD_ROWS, 256], dtype=pl.UINT8)
        moe_workspace = pld.window(moe_workspace_buffer, [MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        hc_window = pld.window(hc_window_buffer, [LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
        hc_arrived = pld.window(hc_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.BF16)
        head_logits_done = pld.window(head_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_fwd(
            embedding[rank],
            engram_ids[rank], engram_scales, engram_payload,
            engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            embedding_ids[rank], initial_pre[rank], identity_rows[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale[rank],
            c2a_compressor_wkv[rank], c2a_compressor_wgate[rank], c1a_compressor_wkv[rank],
            compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank],
            index_wq_b[rank], index_wq_b_scale[rank], index_weights_proj[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            counts[rank], replay_rows[rank],
            position_ids[rank], freqs_cos[rank], freqs_sin[rank],
            window_slots[rank], window_restore_slots[rank], window_indices[rank],
            c2a_compressed_slots[rank], c1a_compressed_slots[rank],
            token_to_req_indices[rank], query_start_loc[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank], state_block_table[rank],
            window_cache[rank], compressed_cache[rank],
            index_cache[rank], index_scales[rank], state_cache[rank],
            final_norm_weight[rank], lm_head_weight[rank], logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank],
            x_hc_out[rank], pre_mix_out[rank], dspark_target_hidden[rank],
            logits[rank], sampled_ids[rank],
            moe_workspace,
            attention_window, attention_arrived, hc_window, hc_arrived,
            head_hidden, head_hidden_done, head_logits, head_logits_done,
            rank, device=rank,
        )


@pl.jit(auto_scope=False)
def l2_generation_head(
    x_hc: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_INPUT_MAX_TOKENS, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16, pl.NZ],
    logit_rows: pl.Tensor[[16], pl.INT32],
    sampling_temperatures: pl.Tensor[[16], pl.FP32],
    sampling_top_ks: pl.Tensor[[16], pl.INT32],
    sampling_top_ps: pl.Tensor[[16], pl.FP32],
    sampling_seeds: pl.Tensor[[16], pl.INT32],
    sampling_positions: pl.Tensor[[16], pl.INT32],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    head_hidden: pld.DistributedTensor[[LM_HEAD_TP_SIZE * 16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.BF16],
    head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    collapsed = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
    hc_head(x_hc, pre_mix, collapsed, worker_limit=64)
    normalized = pl.create_tensor([DECODE_INPUT_MAX_TOKENS, D], dtype=pl.BF16)
    rms_norm(collapsed, final_norm_weight, normalized, DECODE_INPUT_MAX_TOKENS, width=D, zero_tail=False)
    lm_head(
        normalized, lm_head_weight, logit_rows, logits,
        head_hidden, head_hidden_done, head_logits, head_logits_done, rank, MAX_LOGIT_ROWS,
    )
    clear_lm_head_signals(logits, head_hidden_done, head_logits_done)
    sample(
        logits, sampling_temperatures, sampling_top_ks, sampling_top_ps,
        sampling_seeds, sampling_positions, sampled_ids,
    )


@pl.jit.host
def l3_generation_head(
    x_hc: pl.Tensor[[RANKS, DECODE_INPUT_MAX_TOKENS, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS, DECODE_INPUT_MAX_TOKENS, HC_MULT], pl.FP32],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_temperatures: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_top_ks: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_top_ps: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_seeds: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_positions: pl.Tensor[[RANKS, 16], pl.INT32],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
):
    hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        head_hidden = pld.window(hidden_buffer, [LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(logits_buffer, [16, VOCAB], dtype=pl.BF16)
        head_logits_done = pld.window(logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_generation_head(
            x_hc[rank], pre_mix[rank], final_norm_weight[rank], lm_head_weight[rank], logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank],
            logits[rank], sampled_ids[rank],
            head_hidden, head_hidden_done, head_logits, head_logits_done,
            rank, device=rank,
        )


PREFILL_TENSORS = (
    "embedding",
    "engram_ids", "engram_wkv", "engram_wkv_scale", "engram_weight",
    "embedding_ids", "initial_pre", "identity_rows",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale",
    "wkv", "wkv_scale", "kv_norm_weight",
    "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale",
    "c2a_compressor_wkv", "c2a_compressor_wgate", "c1a_compressor_wkv", "compressor_norm_weight",
    "index_wk", "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_weight",
    "gate_weight", "correction_bias",
    "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale",
    "shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
    "counts", "replay_rows", "position_ids", "freqs_cos", "freqs_sin",
    "window_slots", "window_restore_slots", "window_indices",
    "c2a_compressed_slots", "c1a_compressed_slots", "token_to_req_indices", "query_start_loc",
    "c2a_index_block_table", "c1a_index_block_table", "state_block_table",
    "window_cache", "compressed_cache", "index_cache", "index_scales", "state_cache",
    "final_norm_weight", "lm_head_weight", "logit_rows",
    "sampling_temperatures", "sampling_top_ks", "sampling_top_ps", "sampling_seeds", "sampling_positions",
    "x_hc_out", "pre_mix_out", "dspark_target_hidden", "logits", "sampled_ids",
)


def forward_weight_layouts(local_experts: int, *, layers: int):
    """Return full attention weights and rank-local expert layouts."""
    d = M.hidden_size
    inter = M.moe_intermediate_size
    local_h = M.num_attention_heads
    local_groups = M.o_groups
    o_width = local_groups * M.o_lora_rank
    o_group_in = M.num_attention_heads * M.head_dim // M.o_groups
    experts = C.EP_SIZE * local_experts
    packed_rows = d * inter // 256
    kv_sources = len(M.kv_source_layer_ids)
    index_sources = len(M.index_source_layer_ids)
    index_width = M.index_n_heads * M.index_head_dim
    c2a_sources = sum(M.compress_ratios[source] == 2 for source in M.kv_source_layer_ids)
    return {
        "engram_wkv": ((2, ENGRAM_K, KV_OUT), torch.float8_e4m3fn),
        "engram_wkv_scale": ((2 * (ENGRAM_K // 32), KV_OUT), torch.float8_e8m0fnu),
        "engram_weight": ((2, M.hc_mult, d), torch.float32),
        "hc_attn_fn": ((layers * M.mix_hc, M.hc_dim), torch.float32),
        "hc_attn_scale": ((layers * 3,), torch.float32),
        "hc_attn_base": ((layers * M.mix_hc,), torch.float32),
        "attn_norm_weight": ((layers * d,), torch.bfloat16),
        "wq_a": ((layers, d, M.q_lora_rank), torch.float8_e4m3fn),
        "wq_a_scale": ((layers * (d // 32), M.q_lora_rank), torch.float8_e8m0fnu),
        "q_norm_weight": ((layers * M.q_lora_rank,), torch.bfloat16),
        "wq_b": ((layers, M.q_lora_rank, local_h * M.head_dim), torch.float8_e4m3fn),
        "wq_b_scale": ((layers * (M.q_lora_rank // 32), local_h * M.head_dim), torch.float8_e8m0fnu),
        "wkv": ((layers, d, M.head_dim), torch.float8_e4m3fn),
        "wkv_scale": ((layers * (d // 32), M.head_dim), torch.float8_e8m0fnu),
        "kv_norm_weight": ((layers * M.head_dim,), torch.bfloat16),
        "attn_sink": ((layers * local_h,), torch.float32),
        "wo_a": ((layers * local_groups, o_group_in, M.o_lora_rank), torch.float8_e4m3fn),
        "wo_a_scale": ((layers * local_groups, o_group_in // 32, M.o_lora_rank), torch.float8_e8m0fnu),
        "wo_b": ((layers, o_width, d), torch.float8_e4m3fn),
        "wo_b_scale": ((layers * (o_width // 32), d), torch.float8_e8m0fnu),
        "c2a_compressor_wkv": ((c2a_sources, d, M.head_dim), torch.bfloat16),
        "c2a_compressor_wgate": ((c2a_sources, d, M.head_dim), torch.bfloat16),
        "c1a_compressor_wkv": ((d, M.head_dim), torch.bfloat16),
        "compressor_norm_weight": ((kv_sources * M.head_dim,), torch.bfloat16),
        "index_wk": ((index_sources, M.head_dim, M.index_head_dim), torch.bfloat16),
        "index_norm_weight": ((index_sources * M.index_head_dim,), torch.bfloat16),
        "index_wq_b": ((index_sources, M.q_lora_rank, index_width), torch.float8_e4m3fn),
        "index_wq_b_scale": ((index_sources * (M.q_lora_rank // 32), index_width), torch.float8_e8m0fnu),
        "index_weights_proj": ((index_sources, d, M.index_n_heads), torch.bfloat16),
        "hc_ffn_fn": ((layers * M.mix_hc, M.hc_dim), torch.float32),
        "hc_ffn_scale": ((layers * 3,), torch.float32),
        "hc_ffn_base": ((layers * M.mix_hc,), torch.float32),
        "ffn_norm_weight": ((layers * d,), torch.bfloat16),
        "gate_weight": ((layers * experts, d), torch.bfloat16),
        "correction_bias": ((layers * experts,), torch.float32),
        "routed_w1": ((layers * local_experts, packed_rows, 128), torch.uint8),
        "routed_w1_scale": ((layers * (local_experts * (d // 32)), inter), torch.float8_e8m0fnu),
        "routed_w2": ((layers * local_experts, packed_rows, 128), torch.uint8),
        "routed_w2_scale": ((layers * (local_experts * (inter // 32)), d), torch.float8_e8m0fnu),
        "routed_w3": ((layers * local_experts, packed_rows, 128), torch.uint8),
        "routed_w3_scale": ((layers * (local_experts * (d // 32)), inter), torch.float8_e8m0fnu),
        "shared_w1": ((layers, d, inter), torch.float8_e4m3fn),
        "shared_w1_scale": ((layers * (d // 32), inter), torch.float8_e8m0fnu),
        "shared_w2": ((layers, inter, d), torch.float8_e4m3fn),
        "shared_w2_scale": ((layers * (inter // 32), d), torch.float8_e8m0fnu),
        "shared_w3": ((layers, d, inter), torch.float8_e4m3fn),
        "shared_w3_scale": ((layers * (d // 32), inter), torch.float8_e8m0fnu),
        "final_norm_weight": ((d,), torch.bfloat16),
        "lm_head_weight": ((d, M.vocab_size // LM_HEAD_TP_SIZE), torch.bfloat16),
    }


def prepare_prefill_inputs(
    embedding: torch.Tensor,
    capacity: int,
    *,
    start_pos: int | Sequence[int] = 0,
    query_lens: Sequence[int] | None = None,
    request_capacity: int | None = None,
    max_seq_len: int | None = None,
    cache_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Pack one DP group, preserving request order and cache slots across chunks."""
    if embedding.ndim != 2 or embedding.shape[1] != M.hidden_size:
        raise ValueError("embedding must have shape [current tokens, hidden size]")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 16 or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = embedding.shape[0]
    if not 0 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("current rows must fit the configured prefill capacity")
    lengths = [count] if query_lens is None else list(query_lens)
    if not lengths or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in lengths):
        raise ValueError("query_lens must contain nonnegative integer lengths")
    if sum(lengths) != count:
        raise ValueError("query_lens must partition the embedding rows")
    batch = len(lengths)
    request_capacity = batch if request_capacity is None else request_capacity
    integer_capacity = not isinstance(request_capacity, bool) and isinstance(request_capacity, Integral)
    if not integer_capacity or not batch <= request_capacity <= 16:
        raise ValueError("request_capacity must cover the batch and fit the 16 output slots")
    if CP_SIZE > 1 and request_capacity != 1:
        raise ValueError("CP > 1 requires one prefill request per DP group")
    starts = [start_pos] * batch if isinstance(start_pos, Integral) else list(start_pos)
    if len(starts) != batch or any(isinstance(p, bool) or not isinstance(p, Integral) or p < 0 for p in starts):
        raise ValueError("start_pos must provide one nonnegative integer per request")
    max_seq_len = capacity if max_seq_len is None else max_seq_len
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, Integral):
        raise ValueError("max_seq_len must be a positive integer")
    valid_seq_limit = 1 <= max_seq_len <= M.max_position_embeddings
    if not valid_seq_limit or any(p + n > max_seq_len for p, n in zip(starts, lengths)):
        raise ValueError("a request exceeds the retained history capacity")
    if any(p and n for p, n in zip(starts, lengths)) and cache_state is None:
        raise ValueError("continuation requires the preceding call's retained cache state")

    data_pages = (max_seq_len + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    pages = request_capacity * data_pages + 1
    page_ids = torch.arange(request_capacity * data_pages, dtype=torch.int32)
    table = page_ids.reshape(request_capacity, data_pages)
    table[batch:] = -1
    c2a_pages = data_pages
    c2a_page_ids = torch.arange(request_capacity * c2a_pages, dtype=torch.int32)
    c2a_table = c2a_page_ids.reshape(request_capacity, c2a_pages)
    c2a_table[batch:] = -1
    compressed_tables = {
        source: c2a_table if M.compress_ratios[source] == 2 else table for source in M.kv_source_layer_ids
    }
    state_table = torch.arange(request_capacity, dtype=torch.int32).reshape(request_capacity, 1)
    state_table[batch:] = -1
    state_tables = {source: state_table for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2}
    query_starts = torch.full((request_capacity + 1,), count, dtype=torch.int32)
    query_starts[0] = 0
    query_starts[1:batch + 1] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    request_ids = torch.repeat_interleave(torch.arange(batch, dtype=torch.int32), torch.tensor(lengths))
    positions = torch.cat([torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(starts, lengths)])
    window_slots, window_indices, _ = window_metadata(positions, request_ids, table)
    c2a_slots = paged_slots(positions, request_ids, c2a_table, 64, 2, True)
    c1a_slots = paged_slots(positions, request_ids, table)
    freqs_cos, freqs_sin = prepare_forward_rope_tables(max_seq_len)

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[: value.shape[0]] = value
        return result

    initial_pre = torch.zeros((capacity, M.hc_mult), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    logit_rows = torch.full((16,), -1, dtype=torch.int32)
    for request, length in enumerate(lengths):
        if length:
            logit_rows[request] = query_starts[request + 1] - 1
    window_shape = (M.num_hidden_layers, pages, C.BLOCK_SIZE, 1)
    source_shape = (5, pages, 64, 1)
    scale_shape = (*source_shape, M.index_head_dim // C.INDEX_CACHE_GROUP)
    state_shape = (len(state_tables), request_capacity + 1, C.STATE_CAPACITY, 2 * M.head_dim)

    def native_cache(name, shape, dtype, fill=0):
        if cache_state is not None:
            value = cache_state[name]
            if value.device.type != "cpu" or value.dtype != dtype:
                raise ValueError(f"retained {name} must be a CPU tensor with the native cache dtype")
            if tuple(value.shape) != shape or not value.is_contiguous():
                raise ValueError(f"retained {name} must preserve the history layout and capacity")
            return value
        if dtype in (torch.float32, torch.bfloat16):
            return torch.full(shape, float(fill), dtype=dtype)
        value = torch.full(shape, fill, dtype=torch.uint8).view(dtype)
        if name in ("window_cache", "compressed_cache"):
            from quant import unpack_quantized_cache

            payload, factors = unpack_quantized_cache(value, name == "compressed_cache")
            factors.fill_(1.0)
        return value

    values = {
        "embedding": pad(embedding),
        "engram_ids": torch.full((2 * capacity * N_HASH_COLS, 1), -1, dtype=torch.int32),
        "embedding_ids": torch.arange(capacity, dtype=torch.int64),
        "initial_pre": initial_pre,
        "identity_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "counts": torch.tensor([count, count], dtype=torch.int32),
        "replay_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "position_ids": pad(positions, -1).repeat(2, 1),
        "freqs_cos": freqs_cos, "freqs_sin": freqs_sin,
        "window_slots": pad(window_slots, -1).repeat(2, 1),
        "window_restore_slots": torch.full((capacity,), -1, dtype=torch.int64),
        "window_indices": pad(window_indices, -1).repeat(2, 1, 1),
        "c2a_compressed_slots": pad(c2a_slots, -1),
        "c1a_compressed_slots": pad(c1a_slots, -1),
        "token_to_req_indices": pad(request_ids, -1).repeat(2, 1),
        "query_start_loc": query_starts.repeat(2, 1),
        "c2a_index_block_table": c2a_table,
        "c1a_index_block_table": table,
        "state_block_table": state_table,
        "window_cache": native_cache("window_cache", (*window_shape, WINDOW_CACHE_BYTES), torch.uint8),
        "compressed_cache": native_cache("compressed_cache", (*source_shape, COMPRESSED_CACHE_BYTES), torch.uint8),
        "index_cache": native_cache("index_cache", (*source_shape, M.index_head_dim // 2), torch.uint8),
        "index_scales": native_cache("index_scales", scale_shape, torch.float8_e8m0fnu, 127),
        "state_cache": native_cache("state_cache", state_shape, torch.float32),
        "logit_rows": logit_rows,
        "sampling_temperatures": torch.zeros(16, dtype=torch.float32),
        "sampling_top_ks": torch.zeros(16, dtype=torch.int32),
        "sampling_top_ps": torch.zeros(16, dtype=torch.float32),
        "sampling_seeds": torch.zeros(16, dtype=torch.int32),
        "sampling_positions": torch.zeros(16, dtype=torch.int32),
    }
    validate_prefill_caches(values, table, compressed_tables, state_tables)
    return values


def build_tensor_specs(args=None):
    """Build independent per-DP batches for a checkpoint-free, 40-layer CED smoke fixture."""
    from mega_moe import forward_routed_weight_spec
    from prefill_moe import _route_bias

    weight_layouts = forward_weight_layouts(LOCAL_EXPERTS, layers=WEIGHT_BANK_SIZE)
    outputs = {
        "x_hc_out": ([RANKS, CAPACITY, HC_MULT, D], torch.float32),
        "pre_mix_out": ([RANKS, CAPACITY, HC_MULT], torch.float32),
        "dspark_target_hidden": ([RANKS, CAPACITY, DSPARK_MAIN_IN], torch.bfloat16),
        "logits": ([RANKS, 16, VOCAB], torch.float32),
        "sampled_ids": ([RANKS, 16, 8], torch.int32),
    }
    from utils import prefill_batch_layout

    workload = argparse.Namespace(tokens=CAPACITY, dp=DP_SIZE, requests=1)
    if args is not None:
        workload.requests = args.requests
        workload.query_lens = args.query_lens
        workload.start_pos = args.start_pos
    if getattr(workload, "query_lens", None) is None:
        workload.dp_tokens = [CAPACITY, CAPACITY // 2, *([0] * max(0, DP_SIZE - 2))][:DP_SIZE]
    lengths, starts = prefill_batch_layout(workload)
    request_capacity = max(map(len, lengths))
    max_seq_len = CAPACITY + max(p for group in starts for p in group)
    rank_values = []
    for dp_rank, (group, prefixes) in enumerate(zip(lengths, starts)):
        count = sum(group)
        embedding = torch.full((count, D), 0.125 * (dp_rank + 1), dtype=torch.bfloat16)
        previous = None
        if any(prefixes):
            previous = prepare_prefill_inputs(
                embedding[:0], CAPACITY, query_lens=[0] * len(group),
                request_capacity=request_capacity, max_seq_len=max_seq_len,
            )
        values = prepare_prefill_inputs(
            embedding, CAPACITY, query_lens=group, start_pos=prefixes,
            request_capacity=request_capacity, max_seq_len=max_seq_len, cache_state=previous,
        )
        values["engram_ids"][:2 * N_HASH_COLS, 0] = 0
        values["sampling_temperatures"][0] = 0.7 if dp_rank % 2 == 0 else 0.0
        values["sampling_seeds"] = torch.arange(16, dtype=torch.int32)
        rank_values.extend([values] * CP_SIZE)
    specs = []
    for name in PREFILL_TENSORS:
        if name in weight_layouts:
            rank_shape, dtype = weight_layouts[name]
            shape = [RANKS, *rank_shape]
            value = 1.0 if dtype == torch.float8_e8m0fnu or name == "final_norm_weight" else 0.0
            if name in ("routed_w1", "routed_w2", "routed_w3"):
                value = None
            elif name == "correction_bias":
                value = _route_bias(N_EXPERTS).repeat(RANKS, WEIGHT_BANK_SIZE)
        elif name in outputs:
            shape, dtype = outputs[name]
            value = None
        else:
            value = torch.stack([values[name] for values in rank_values])
            shape, dtype = list(value.shape), value.dtype
        spec = forward_routed_weight_spec(
            name, shape, dtype, value, False,
            local_experts=LOCAL_EXPERTS, sparse_zero_routed=True,
        )
        specs.append(spec)
    return specs


def validate_prefill_caches(
    cache_state: Mapping[str, torch.Tensor],
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
    state_block_tables: Mapping[int, torch.Tensor],
) -> None:
    """Validate native prefill cache layouts and source-local page/state ownership."""
    cache_names = ("window_cache", "compressed_cache", "index_cache", "index_scales")
    for name in cache_names:
        value = cache_state[name]
        if value.ndim not in (5, 6) or not value.is_contiguous():
            raise ValueError(f"{name} must preserve the contiguous native source/page layout")
        sources = M.num_hidden_layers if name.startswith("window") else 5
        rows = C.BLOCK_SIZE if name.startswith("window") else 64
        if value.shape[-5] != sources or value.shape[-4] < 1 or tuple(value.shape[-3:-1]) != (rows, 1):
            raise ValueError(f"{name} has an incompatible source or page geometry")
        expected_dtype = {
            "window_cache": torch.uint8, "compressed_cache": torch.uint8,
            "index_cache": torch.uint8, "index_scales": torch.float8_e8m0fnu,
        }[name]
        if value.dtype != expected_dtype:
            raise ValueError(f"{name} must retain its native payload or scale dtype")
        expected_width = {
            "window_cache": WINDOW_CACHE_BYTES, "compressed_cache": COMPRESSED_CACHE_BYTES,
            "index_cache": M.index_head_dim // 2, "index_scales": M.index_head_dim // C.INDEX_CACHE_GROUP,
        }[name]
        if value.shape[-1] != expected_width:
            raise ValueError(f"{name} has an incompatible native row width")
        reference = cache_state["window_cache"]
        if value.shape[:-5] != reference.shape[:-5]:
            raise ValueError("all prefill cache pools must preserve the same rank extents")
        if value.device != reference.device:
            raise ValueError("all cache payloads and scales must reside on the same device")
    history_pages = cache_state["compressed_cache"].shape[-4]
    for name in ("index_cache", "index_scales"):
        if cache_state[name].shape[-4] != history_pages:
            raise ValueError(f"{name} must share the compressed-cache page extent")
    state = cache_state["state_cache"]
    if state.ndim not in (4, 5) or not state.is_contiguous():
        raise ValueError("state_cache must preserve its contiguous source/request layout")
    state_geometry = (C.STATE_CAPACITY, 2 * M.head_dim)
    if state.shape[-4] != 3 or state.shape[-3] < 1 or tuple(state.shape[-2:]) != state_geometry:
        raise ValueError("state_cache has an incompatible C2A source or state geometry")
    if state.shape[:-4] != cache_state["window_cache"].shape[:-5]:
        raise ValueError("state_cache must preserve the cache pools' rank extent")
    if state.dtype != torch.float32:
        raise ValueError("state_cache must retain FP32 compressor state")
    if state.device != cache_state["window_cache"].device:
        raise ValueError("state_cache must reside on the cache pools' device")

    if cache_state["window_cache"].ndim != 5:
        raise ValueError("validate_prefill_caches requires one rank's native cache tensors")
    if window_block_table.ndim != 2:
        raise ValueError("window block table must be a matrix")
    requests = window_block_table.shape[0]
    if requests < 1 or (CP_SIZE > 1 and requests != 1):
        raise ValueError("prefill requires positive request capacity and one request per DP when CP > 1")
    pages = cache_state["compressed_cache"].shape[1]
    state_blocks = cache_state["state_cache"].shape[1]

    def validate_table(name, table, capacity, state=False):
        if table.device.type != "cpu" or table.dtype != torch.int32:
            raise ValueError(f"{name} must be a CPU INT32 tensor")
        if table.ndim != 2 or table.shape[0] != requests or (state and table.shape[1] != 1):
            raise ValueError(f"{name} has incompatible request or table dimensions")
        if bool(((table < -1) | (table >= capacity)).any()):
            raise ValueError(f"{name} IDs must be -1 or within the source-local capacity {capacity}")
        if state:
            allocated = table[table >= 0]
            if allocated.unique().numel() != allocated.numel():
                raise ValueError("live requests must own distinct state blocks")

    validate_table("window block table", window_block_table, cache_state["window_cache"].shape[-4])
    family_tables = {}
    shared_state = None
    for source in M.kv_source_layer_ids:
        if source not in compressed_block_tables:
            raise ValueError(f"missing compressed block table for source layer {source}")
        table = compressed_block_tables[source]
        validate_table(f"source {source} compressed block table", table, pages)
        ratio = M.compress_ratios[source]
        family_table = family_tables.setdefault(ratio, table)
        if not torch.equal(table, family_table):
            raise ValueError("sources in one family must share local physical page IDs")
        if ratio > 1:
            if source not in state_block_tables:
                raise ValueError(f"missing state block table for source layer {source}")
            state_table = state_block_tables[source]
            validate_table(f"source {source} state block table", state_table, state_blocks, state=True)
            if shared_state is None:
                shared_state = state_table
            elif not torch.equal(state_table, shared_state):
                raise ValueError("C2A sources must share request slots within their separate state pools")


if __name__ == "__main__":
    from golden import run
    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--lm-head-tp", type=int, default=C.LM_HEAD_TP_SIZE, choices=C.SUPPORTED_LM_HEAD_TP_SIZES)
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=[1, 2, 4, 8])
    parser.add_argument("--ep", type=int, default=RANKS, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=[1, 2, 4, 8])
    parser.add_argument("--experts-per-rank", type=int, default=LOCAL_EXPERTS, help="routed experts on each EP rank")
    bank_help = "layer weight sets (default: 1 for smoke, full layer count for --compile-only)"
    parser.add_argument("--weight-bank-size", type=int, choices=(1, N_LAYERS), default=WEIGHT_BANK_SIZE, help=bank_help)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--query-lens", type=parse_prefill_groups, help="request lengths per DP, e.g. '17,15;32'")
    parser.add_argument("--start-pos", type=parse_prefill_groups, help="request prefix lengths per DP")
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--moe-tokens", type=int, default=MOE_TOKENS)
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(RANKS)))
    parser.add_argument("--engram-table-rows", type=int, default=ENGRAM_ROWS)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-scope-stats", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("this forward entry requires hardware; --sim is unsupported")
    requested_layout = (args.lm_head_tp, args.ep, args.dp, args.experts_per_rank, args.capacity)
    compiled_layout = (C.LM_HEAD_TP_SIZE, RANKS, DP_SIZE, LOCAL_EXPERTS, CAPACITY)
    if requested_layout != compiled_layout:
        parser.error("parallel sizes, expert count and capacity must match the import-time configuration")
    devices = [int(device) for device in args.device.split(",") if device]
    if len(devices) != RANKS or len(set(devices)) != RANKS:
        parser.error(f"need exactly {RANKS} distinct devices, got {devices}")
    specs = build_tensor_specs(args)
    config = dict(
        backend_type=args.backend_type, execution_mode=args.execution_mode,
        dump_passes=args.dump_passes, enable_scope_stats=args.enable_scope_stats,
        enable_chip_swimlane=args.enable_chip_swimlane, log_level=args.log_level,
        ring_heap=(SMOKE_RING_HEAP_BYTES,) * 4, ring_dep_pool=131072,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    with zero_engram_window_sources(len(devices), ENGRAM_SHARD_ROWS, compile_only=args.compile_only) as sources:
        result = run(
            a5_default_runtime="host_build_graph",
            fn=l3_prefill_fwd,
            specs=specs,
            golden_fn=None,
            config=config,
            compile_only=args.compile_only,
            window_sources=sources,
            persistent=True,
            reset_persistent_windows=False,
            return_tensors=True,
        )
    if result.passed and not args.compile_only:
        target_hidden = result.tensors["dspark_target_hidden"]
        for name in ("x_hc_out", "pre_mix_out", "logits"):
            if not torch.isfinite(result.tensors[name]).all():
                raise AssertionError(f"prefill {name} contains non-finite values")
        if not torch.isfinite(target_hidden.float()).all():
            raise AssertionError("DSpark prefill target hidden contains non-finite values")
        expected_tap = torch.zeros((RANKS, CAPACITY, D), dtype=torch.bfloat16)
        active_rows = specs[PREFILL_TENSORS.index("counts")].init_value[:, 1].tolist()
        for rank, count in enumerate(active_rows):
            expected_tap[rank, :count] = specs[PREFILL_TENSORS.index("embedding")].init_value[rank, :count]
        for tap, layer in enumerate(DSPARK_TARGET_LAYER_IDS):
            tap_rows = target_hidden[:, :, tap * D : (tap + 1) * D]
            if not torch.equal(tap_rows, expected_tap):
                raise AssertionError(f"DSpark prefill layer {layer} tap differs from the zero-weight reference")
        validation = "[FWD] Execution and all DSpark taps passed the zero-weight reference"
        print(f"{validation}; full numerical accuracy was not validated.", flush=True)
    if not result.passed:
        raise SystemExit(result.error or 1)
