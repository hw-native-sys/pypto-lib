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
"""Compose native V4.1 operators into one resident CED prefill per A5 rank.

CI uses TP1/EP2/DP2 with 128 experts. Checkpoint generation uses 384 experts
with EP4/DP4 (default) or EP8/DP8. Each rank owns one prompt and a shard of
routed experts; attention weights are replicated. Original MXFP8/MXFP4
weights remain resident. Calls append current rows to caller-owned native caches.
Engram consumes host-gathered rows at layers 1 and 14.
"""

import struct
import math
from numbers import Integral
from typing import Mapping, Sequence
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sibling operators share the same configuration module for script and -m entry points.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig, StackedDeviceTensor
import torch

import config as C

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--checkpoint")
_shape_parser.add_argument("--ep", type=int, choices=[2, 4, 8])
_shape_parser.add_argument("--experts-per-rank", type=int)
_shape_parser.add_argument("--capacity", type=int, default=128)
_shape_parser.add_argument("--dp", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
C.EP_SIZE = _shape_options.ep or (4 if _shape_options.checkpoint else 2)
_default_experts = C.FLASH.n_routed_experts // C.EP_SIZE if C.EP_SIZE in (4, 8) else 64
C.N_LOCAL_EXPERTS = _default_experts if _shape_options.experts_per_rank is None else _shape_options.experts_per_rank
if C.N_LOCAL_EXPERTS <= 0 or C.EP_SIZE * C.N_LOCAL_EXPERTS % 16:
    _shape_parser.error("--experts-per-rank must be positive and give a total expert count aligned to 16")
_MAX_CHUNK_TOKENS = 4096
CAPACITY = _shape_options.capacity
if C.TP_SIZE != 1 or _shape_options.dp not in (None, C.EP_SIZE):
    _shape_parser.error("prefill requires TP1 with DP equal to EP")
if not 16 <= CAPACITY <= _MAX_CHUNK_TOKENS or CAPACITY % 16:
    _shape_parser.error("--capacity must be between 16 and 4096 and a multiple of 16")
# Keep MoE scratch bounded independently of the current-row capacity.
C.set_moe_tokens(min(CAPACITY, C.MOE_TOKENS))

from config import (
    FLASH as M,
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
    B_DYN,
    INDEX_POOL_BLOCKS_DYN,
    PREFILL_MAX_TOKENS,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    STATE_CAPACITY,
    MOE_TOKENS,
    ROUTE_T_DYN,
)

from hc import hc_post
from hc import hc_head, hc_pre, zero_bf16_padding
from input_pack import gather_rows, pack_x_hc, prepare_compressed_lens
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows, prepare_forward_rope_tables
from utils import paged_slots, window_metadata
from engram import ENGRAM_K, KV_OUT, HostEngram, engram_from_rows
from lm_head import final_output
from prefill_moe import AUX_WIDTH, ROUTE_WIDTH, SIGNAL_PAD, clear_moe_signals
from prefill_moe import prefill_moe
from compressor import STATE_WIDTH
from prefill_c1a_full import prefill_publish_decoder
from prefill_swa import prefill_swa_attention
from o_proj import retire_prefill_signals, prefill_tp_output_all_reduce
from prefill_c2a_full import prefill_c2a_full_attention
from prefill_c2a_reuse import prefill_c2a_reuse_attention
from prefill_c1a_reindex import prefill_c1a_reindex_attention
from prefill_c1a_reuse import prefill_c1a_reuse_attention
from expert_routed import build_mxfp4_pair_lut
from rmsnorm import rms_norm_tiled
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS

RANKS = C.EP_SIZE
D = M.hidden_size
HC_MULT = M.hc_mult
VOCAB = M.vocab_size
PAGES = pl.dynamic("V41_PREFILL_HISTORY_PAGES")
C2A_TABLE_DYN = pl.dynamic("V41_PREFILL_C2A_TABLE")
C1A_TABLE_DYN = pl.dynamic("V41_PREFILL_C1A_TABLE")
LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
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
LOCAL_H = M.num_attention_heads // C.TP_SIZE
LOCAL_O_GROUPS = M.o_groups // C.TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * M.o_lora_rank
MIX_HC = M.mix_hc
MOE_INTER = M.moe_intermediate_size
N_EXPERTS = RANKS * LOCAL_EXPERTS
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
TOPK = M.num_experts_per_tok

ROPE_DIM = M.qk_rope_head_dim


@pl.jit.inline(auto_scope=False)
def prefill_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
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
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    mode: pl.Scalar[pl.INT32],
):
    if mode == 0:
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
        prefill_swa_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
            output_arrived, partial, num_tokens, attention_epoch,
        )
        prefill_tp_output_all_reduce(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == 1:
        attention_tokens = pl.tensor.dim(x, 0)
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.BF16)
        prefill_c2a_full_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, token_to_req_indices, c2a_compressed_lens, index_cache, index_cache_scale,
            c2a_index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
            compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight,
            c2a_compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
            index_weights_proj, topk_indices, output_arrived, attention_partial, num_tokens,
            attention_epoch,
        )
        prefill_tp_output_all_reduce(
            attention_partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == 2:
        attention_tokens = pl.tensor.dim(x, 0)
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.BF16)
        prefill_c2a_reuse_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, topk_indices, output_arrived, attention_partial, num_tokens,
            attention_epoch,
        )
        prefill_tp_output_all_reduce(
            attention_partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif mode == 3:
        prefill_c1a_reindex_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, token_to_req_indices, c1a_compressed_lens, index_cache, index_cache_scale,
            c1a_index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, output_window, output_arrived, output, group_base,
            tp_rank, num_tokens, attention_epoch,
            use_candidates=False, scatter_output=False,
        )
    elif mode == 4:
        prefill_c1a_reindex_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, token_to_req_indices, c1a_compressed_lens, index_cache, index_cache_scale,
            c1a_index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, output_window, output_arrived, output, group_base,
            tp_rank, num_tokens, attention_epoch,
            scatter_output=False, use_candidates=True,
        )
    elif mode == 5:
        prefill_c1a_reuse_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, topk_indices, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
            scatter_output=False,
        )
    return output


@pl.jit.inline(auto_scope=False)
def _prefill_moe(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[
        [LOCAL_EXPERTS * (D // 32), MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w2: pl.Tensor[[LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[
        [LOCAL_EXPERTS * (MOE_INTER // 32), D],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    routed_w3: pl.Tensor[[LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[
        [LOCAL_EXPERTS * (D // 32), MOE_INTER],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[
        [D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[
        [MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN
    ],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[
        [D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
    ],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    x_next: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[
        [LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8
    ],
    recv_weights: pld.DistributedTensor[
        [LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32
    ],
    recv_routes: pld.DistributedTensor[
        [LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32
    ],
    arrived: pld.DistributedTensor[[RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [RANKS, LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [RANKS, LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    slab_hidden: pl.Tensor[[2 * MOE_TOKENS, HC_MULT, D], pl.FP32],
    slab_pre: pl.Tensor[[2 * MOE_TOKENS, HC_MULT], pl.FP32],
    slab_mixed: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
    slab_rows: pl.Tensor[[MOE_TOKENS], pl.INT32],
    reuse_ready: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run bounded MoE slabs and return completion of every scratch reader."""
    capacity = pl.tensor.dim(x_hc, 0)
    source_hidden = pl.reshape(x_hc, [capacity, HC_DIM])
    round_completion = pl.array.create(1, pl.TASK_ID)
    round_completion[0] = reuse_ready
    rounds = (CAPACITY + MOE_TOKENS - 1) // MOE_TOKENS
    for round_id in pl.range(rounds):
        begin = round_id * MOE_TOKENS
        rows = pl.min(MOE_TOKENS, capacity - begin)
        # Keep caller-owned destinations outside the slab allocation scope.
        hidden_destination = pl.slice(x_next, [rows, HC_MULT, D], [begin, 0, 0])
        mixed_destination = pl.slice(x_mixed, [rows, D], [begin, 0])
        pre_destination = pl.slice(next_pre_mix, [rows, HC_MULT], [begin, 0])
        # Carry completion explicitly before overwriting shared scratch.
        row_ids = pl.slice(slab_rows, [MOE_TOKENS], [0])
        round_hidden = pl.slice(slab_hidden, [MOE_TOKENS, HC_MULT, D], [0, 0, 0])
        round_output = pl.slice(slab_hidden, [MOE_TOKENS, HC_MULT, D], [MOE_TOKENS, 0, 0])
        round_pre = pl.slice(slab_pre, [MOE_TOKENS, HC_MULT], [0, 0])
        round_next = pl.slice(slab_pre, [MOE_TOKENS, HC_MULT], [MOE_TOKENS, 0])
        round_mixed = pl.slice(slab_mixed, [MOE_TOKENS, D], [0, 0])
        with pl.scope():
            active = pl.max(0, pl.min(MOE_TOKENS, num_tokens - begin))
            with pl.spmd(1, name_hint="prefill_moe_rows", deps=[round_completion[0]]) as rows_ready:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    source_row = pl.cast(-1, pl.INT32)
                    if row < active:
                        source_row = pl.cast(begin + row, pl.INT32)
                    pl.write(row_ids, [row], source_row)
            packed_hidden = pl.reshape(round_hidden, [MOE_TOKENS, HC_DIM])
            gather_rows(source_hidden, row_ids, packed_hidden, rows_ready)
            gather_rows(pre_mix, row_ids, round_pre, rows_ready)
            epoch = (moe_epoch - 1) * rounds + round_id + 1
            prefill_moe(
                round_hidden, round_pre, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_weight,
                gate_weight, correction_bias,
                routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
                mxfp4_pair_lut,
                shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
                round_next, round_mixed, round_output,
                recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
                routed_output, combine_arrived, active, ep_rank, epoch,
            )
            with pl.spmd(32, name_hint="prefill_moe_results") as hidden_copied:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, MOE_TOKENS, 32):
                    if begin + row < capacity:
                        for lane in pl.range(HC_MULT):
                            for column in pl.range(0, D, 512):
                                hidden = round_output[row:row + 1, lane:lane + 1, column:column + 512]
                                hidden_destination[row:row + 1, lane:lane + 1, column:column + 512] = hidden
                        mixed = round_mixed[row:row + 1, :]
                        mixed_destination[row:row + 1, :] = mixed
            with pl.spmd(1, name_hint="prefill_moe_pre_mix") as pre_copied:
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    if begin + row < capacity:
                        for lane in pl.range(HC_MULT):
                            value = pl.read(round_next, [row, lane])
                            pl.write(pre_destination, [row, lane], value)
            round_completion[0] = pl.system.task_dummy(deps=[hidden_copied, pre_copied])
    return round_completion[0]


@pl.jit(auto_scope=False)
def l2_prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    counts: pl.Tensor[[2], pl.INT32],
    host_counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 512], pl.FP8E4M3FN]],
    window_scales: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 16], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 32], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    logit_rows: pl.Tensor[[16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8],
    recv_weights: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    data_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    routed_output: pld.DistributedTensor[[MOE_TOKENS * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.FP32],
    head_logits_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run both CED stages and retire communication signals after the decoder."""
    compressed_cache.bind_dynamic(1, PAGES)
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
    compressed_cos = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    compressed_sin = pl.create_tensor([CAPACITY, 32], dtype=pl.FP32)
    encoder_count = pl.read(host_counts, [0])
    decoder_count = pl.read(host_counts, [1])
    materialize_rope_rows(
        freqs_cos[0], freqs_sin[0], position_ids[0], encoder_count, rope_cos[0], rope_sin[0], False,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, rope_cos[1], rope_sin[1], False,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[1], decoder_count, rope_cos[2], rope_sin[2], False,
    )
    materialize_rope_rows(
        freqs_cos[1], freqs_sin[1], position_ids[0], encoder_count, compressed_cos, compressed_sin, True,
    )
    c2a_compressed_lens = pl.create_tensor([CAPACITY], dtype=pl.INT32)
    c1a_compressed_lens = pl.create_tensor([CAPACITY], dtype=pl.INT32)
    prepare_compressed_lens(position_ids[0], c2a_compressed_lens, 1)
    prepare_compressed_lens(position_ids[1], c1a_compressed_lens, 0)
    hidden_pair = pl.create_tensor([4 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([4 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    initial_mix = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
    gather_ready = pl.system.task_dummy(deps=[])
    pack_x_hc(embedding_ids, embedding, initial_hidden)
    gather_rows(initial_pre, identity_rows, initial_mix, gather_ready)
    topk_indices = pl.create_tensor([CAPACITY, INDEX_TOPK], dtype=pl.INT32)
    history_positions = pl.tensor.dim(compressed_cache, 1) * 128
    pool_blocks = pl.min(INDEX_POOL_BLOCKS, (history_positions // 8 + 31) // 32 * 32)
    pool_block_ids = pl.create_tensor([CAPACITY, pool_blocks], dtype=pl.INT32)
    pool_block_counts = pl.create_tensor([CAPACITY], dtype=pl.INT32)
    # Reuse bounded MoE slabs across all layers and rounds.
    attn_hidden = pl.create_tensor([CAPACITY, HC_MULT, D], dtype=pl.FP32)
    attn_pre = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_mixed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    moe_ready = pl.system.task_dummy(deps=[])
    for layer in pl.range(N_LAYERS):
        stage = layer // 20
        active = encoder_count
        if stage != 0:
            active = decoder_count
        current_slot = (2 * stage + layer % 2) * CAPACITY
        next_slot = (2 * stage + (layer + 1) % 2) * CAPACITY
        hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [current_slot, 0, 0])
        delayed_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [current_slot, 0])
        next_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [next_slot, 0, 0])
        next_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [next_slot, 0])
        kv_source = pl.min(pl.max((layer - 2) // 6, 0), 2)
        if layer >= 20:
            kv_source = 3
        index_source = kv_source
        if layer >= 24:
            index_source = (layer - 24) // 4 + 4
        c2a_source = pl.min(kv_source, 2)
        layer_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
        layer_attn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer * 3])
        layer_attn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer * MIX_HC])
        layer_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [layer * MIX_HC, 0])
        layer_ffn_scale: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer * 3])
        layer_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer * MIX_HC])
        layer_attn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_weight, [D], [layer * D])
        layer_ffn_norm: pl.Tensor[[D], pl.BF16] = pl.slice(ffn_norm_weight, [D], [layer * D])
        layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(q_norm_weight, [Q_LORA], [layer * Q_LORA])
        layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(kv_norm_weight, [HEAD_DIM], [layer * HEAD_DIM])
        layer_wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN] = wq_a[layer]
        layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_a_scale, [D // MX_GROUP, Q_LORA], [layer * (D // MX_GROUP), 0])
        layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN] = wq_b[layer]
        layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wq_b_scale, [Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], [layer * (Q_LORA // MX_GROUP), 0])
        layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN] = wkv[layer]
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wkv_scale, [D // MX_GROUP, HEAD_DIM], [layer * (D // MX_GROUP), 0])
        layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN] = wo_b[layer]
        layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(wo_b_scale, [LOCAL_O_WIDTH // MX_GROUP, D], [layer * (LOCAL_O_WIDTH // MX_GROUP), 0])
        layer_shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w1[layer]
        layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w1_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
        layer_shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN] = shared_w2[layer]
        layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w2_scale, [MOE_INTER // MX_GROUP, D], [layer * (MOE_INTER // MX_GROUP), 0])
        layer_shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN] = shared_w3[layer]
        layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(shared_w3_scale, [D // MX_GROUP, MOE_INTER], [layer * (D // MX_GROUP), 0])
        layer_attn_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [layer * LOCAL_H])
        layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
        layer_wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0] = pl.slice(wo_a_scale, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
        layer_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wkv[c2a_source]
        layer_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = c2a_compressor_wgate[c2a_source]
        layer_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [kv_source * HEAD_DIM])
        layer_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = index_wk[index_source]
        layer_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [index_source * INDEX_DIM])
        layer_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = index_wq_b[index_source]
        layer_index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(index_wq_b_scale, [Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], [index_source * (Q_LORA // MX_GROUP), 0])
        layer_index_weights: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = index_weights_proj[index_source]
        layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_weight, [N_EXPERTS, D], [layer * N_EXPERTS, 0])
        layer_gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [layer * N_EXPERTS])
        layer_routed_w1: pl.Tensor[[LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w1, [LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w1_scale: pl.Tensor[[LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w1_scale, [LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (LOCAL_EXPERTS * (D // MX_GROUP)), 0])
        layer_routed_w2: pl.Tensor[[LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w2, [LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w2_scale: pl.Tensor[[LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w2_scale, [LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], [layer * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), 0])
        layer_routed_w3: pl.Tensor[[LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8] = pl.slice(routed_w3, [LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w3_scale: pl.Tensor[[LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.slice(routed_w3_scale, [LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], [layer * (LOCAL_EXPERTS * (D // MX_GROUP)), 0])
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for ALL encoder rows before selecting replay.
                encoder_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
                encoder_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_head(encoder_hidden, encoder_pre, encoder_collapsed)
                norm_tokens = pl.tensor.dim(encoder_collapsed, 0)
                for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
                    rms_norm_tiled(
                        encoder_collapsed, layer_attn_norm, encoder_normalized, norm_block, norm_tokens,
                        output_fp32=False,
                    )
                publisher_ready = prefill_publish_decoder(
                    encoder_normalized, c1a_compressor_wkv, layer_compressor_norm, layer_index_wk, layer_index_norm,
                    rope_cos[1], rope_sin[1], c1a_compressed_slots,
                    compressed_cache[3], compressed_scales[3], index_cache[3], index_scales[3],
                    encoder_count,
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(source_hidden, replay_rows, gathered_hidden, publisher_ready)
                gather_rows(encoder_pre, replay_rows, delayed_pre, publisher_ready)
        if layer == 1 or layer == 14:
            with pl.scope():
                engram_slot = layer // 14
                layer_engram_scale = pl.slice(engram_wkv_scale, [ENGRAM_K // 32, KV_OUT], [engram_slot * (ENGRAM_K // 32), 0])
                engram_from_rows(
                    engram_lookup[engram_slot], engram_wkv[engram_slot], layer_engram_scale,
                    engram_weight[engram_slot], hidden, hidden,
                )
        with pl.scope():
            attn_post = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
            attn_residual = pl.create_tensor([CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
            collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            attn_output = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            zero_bf16_padding(attn_output, pl.cast(0, pl.INT32))
            hc_pre(
                hidden, delayed_pre, layer_attn_fn, layer_attn_scale, layer_attn_base,
                collapsed, attn_pre, attn_post, attn_residual,
            )
            norm_tokens = pl.tensor.dim(collapsed, 0)
            for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
                rms_norm_tiled(collapsed, layer_attn_norm, normalized, norm_block, norm_tokens, output_fp32=False)
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
            if active > 0:
                prefill_attention(
                    normalized, layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale, layer_wkv, layer_wkv_scale, layer_kv_norm,
                    layer_attn_sink, layer_wo_a, layer_wo_a_scale, layer_wo_b, layer_wo_b_scale, rope_cos[rope_slot], rope_sin[rope_slot],
                    window_slots[stage], window_indices[stage], window_cache[layer], window_scales[layer],
                    compressed_cache[cache_slot], compressed_scales[cache_slot], token_to_req_indices[stage],
                    c2a_compressed_lens, c1a_compressed_lens, index_cache[cache_slot], index_scales[cache_slot],
                    c2a_index_block_table, c1a_index_block_table, position_ids[stage], compressed_cos, compressed_sin,
                    layer_compressor_wkv, layer_compressor_wgate, query_start_loc[stage], state_block_table, state_cache[state_slot],
                    layer_compressor_norm, c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale,
                    layer_index_weights, pool_block_ids, pool_block_counts, topk_indices, attention_window, attention_arrived,
                    attn_output, rank, 0, active, layer + 1, mode,
                )
            hc_post_deps = pl.array.create(2, pl.TASK_ID)
            hc_post(attn_output, hidden, attn_post, attn_residual, attn_hidden, deps=hc_post_deps)
        moe_ready = _prefill_moe(
            attn_hidden, attn_pre, layer_ffn_fn, layer_ffn_scale, layer_ffn_base, layer_ffn_norm, layer_gate_weight, layer_gate_bias,
            layer_routed_w1, layer_routed_w1_scale, layer_routed_w2, layer_routed_w2_scale, layer_routed_w3, layer_routed_w3_scale,
            mxfp4_pair_lut, layer_shared_w1, layer_shared_w1_scale, layer_shared_w2, layer_shared_w2_scale,
            layer_shared_w3, layer_shared_w3_scale, next_pre, ffn_mixed, next_hidden,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
            routed_output, combine_arrived, slab_hidden, slab_pre, slab_mixed, slab_rows,
            moe_ready, active, rank, layer + 1,
        )
    final_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [2 * CAPACITY, 0, 0])
    final_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [2 * CAPACITY, 0])
    source_hidden = pl.reshape(final_hidden, [CAPACITY, HC_DIM])
    output_hidden = pl.reshape(x_hc_out, [CAPACITY, HC_DIM])
    gather_rows(source_hidden, identity_rows, output_hidden, gather_ready)
    gather_rows(final_pre, identity_rows, pre_mix_out, gather_ready)
    clear_moe_signals(x_hc_out, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    retire_prefill_signals(x_hc_out, attention_arrived, rank, 0)
    final_logits, final_sampled_ids = final_output(
        final_hidden, final_pre, final_norm_weight, lm_head_weight,
        logit_rows, logits, sampled_ids,
        head_hidden, head_hidden_done, head_logits, head_logits_done,
        rank, 0, 1, 1,
    )
    return final_logits, final_sampled_ids


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[RANKS, 2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[RANKS, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS, N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS, N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS, N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS, N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS, N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS, N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    mxfp4_pair_lut: pl.Tensor[[RANKS, 2, 256], pl.INT16],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    host_counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[RANKS, 2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[RANKS, B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[RANKS, B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS, B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 512], pl.FP8E4M3FN]],
    window_scales: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 16], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 32], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, VOCAB, D], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
):
    """Append one TP1/DP4/EP4 or TP1/DP8/EP8 chunk and return its final state/logits.

    CAPACITY bounds current rows to at most 4096; the dynamic page extent
    independently sizes retained history. For full causal continuation replay
    all current rows:
    counts[:, 0] == counts[:, 1], replay_rows and identity_rows are current-row
    identities, padded with -1. Positions, lengths and physical slots are
    absolute within the request. Both stages' window indices include committed
    history; the compressed families have independent local page tables.

    embedding holds caller-gathered BF16 rows; embedding_ids indexes that
    rank-local slab, not the model vocabulary. initial_pre is the current
    rows' delayed mix (identity for new tokens). position_ids, window_slots,
    window_indices, token_to_req_indices and query_start_loc have stage order
    encoder, decoder. Each stage has packed query_start_loc and absolute
    position_ids; token_to_req_indices indexes request-table rows.
    C2A/C1A index_block_table arguments have independent widths and source-local
    physical page IDs. KV and index keys share their family's mapping. Pages
    contain 128 stored rows: 256 original tokens for C2A, 128 for C1A/SWA.
    Visible lengths are derived once as floor((position+1)/ratio), including
    completed current groups: C2A follows encoder rows and C1A decoder replay rows.
    freqs_cos/sin are resident full FP32 half-width tables with profile order
    base, compressed. Their row capacity is independent of current CAPACITY
    and covers all active absolute positions. The graph gathers encoder base,
    encoder compressed, decoder compressed and C2A group-first rows once.
    C1A publication reuses encoder compressed rows before replay selection.
    c2a_compressed_slots and c1a_compressed_slots both follow encoder rows;
    C1A publishes all encoder rows before decoder replay selection. Slots are
    source-local physical rows; -1 disables publication. C2A publishes only at
    completed pair positions. state_block_table maps requests to source-local
    state blocks shared across three independent C2A source slices; -1 is inactive.
    logit_rows indexes the decoder replay slab; -1 disables that output row.

    Preserve all seven InOut cache tensors across chunks and decode. Their
    source/page dimensions can be flattened into decode pool views.
    Residual/pre-mix outputs describe these rows after layer 39. New tokens
    start from their own embedding and identity pre-mix; historical mHC state
    is represented by the committed KV/index/compressor caches.

    Zero-count ranks participate in EP communication without publishing cache
    rows. Wait for runtime completion before advancing lengths, launching a
    dependent call or reusing buffers. Use persistent=True with
    reset_persistent_windows=False; signals are retired at forward completion.
    After request completion, release pages and reset the request slot in all C2A sources before
    assigning them to another request; failed calls require cache restoration
    or reset before retry. Engram consumes host lookup rows; vision and DSpark have no inputs here.
    """
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    compressed_cache.bind_dynamic(2, PAGES)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.FP32)
    head_logits_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buffer, [LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buffer, [LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buffer, [LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buffer, [RANKS, 128], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.FP32)
        head_logits_done = pld.window(head_logits_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_fwd(
            embedding[rank],
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            embedding_ids[rank],
            initial_pre[rank],
            identity_rows[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale[rank],
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
            mxfp4_pair_lut[rank],
            counts[rank],
            host_counts[rank],
            replay_rows[rank],
            position_ids[rank],
            freqs_cos[rank], freqs_sin[rank],
            window_slots[rank],
            window_indices[rank],
            c2a_compressed_slots[rank],
            c1a_compressed_slots[rank],
            token_to_req_indices[rank],
            query_start_loc[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank],
            state_block_table[rank],
            window_cache[rank],
            window_scales[rank],
            compressed_cache[rank],
            compressed_scales[rank],
            index_cache[rank],
            index_scales[rank],
            state_cache[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            logit_rows[rank],
            x_hc_out[rank], pre_mix_out[rank],
            logits[rank],
            sampled_ids[rank],
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            attention_window,
            attention_arrived,
            head_hidden,
            head_hidden_done,
            head_logits,
            head_logits_done,
            rank, device=rank,
        )


PREFILL_TENSORS = (
    "embedding",
    "engram_lookup", "engram_wkv", "engram_wkv_scale", "engram_weight",
    "embedding_ids",
    "initial_pre",
    "identity_rows",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight",
    "wq_b", "wq_b_scale",
    "wkv", "wkv_scale", "kv_norm_weight",
    "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale",
    "c2a_compressor_wkv", "c2a_compressor_wgate", "c1a_compressor_wkv",
    "compressor_norm_weight",
    "index_wk", "index_norm_weight",
    "index_wq_b", "index_wq_b_scale", "index_weights_proj",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "ffn_norm_weight", "gate_weight", "correction_bias",
    "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale",
    "shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
    "mxfp4_pair_lut",
    "counts",
    "host_counts",
    "replay_rows",
    "position_ids",
    "freqs_cos", "freqs_sin",
    "window_slots",
    "window_indices",
    "c2a_compressed_slots",
    "c1a_compressed_slots",
    "token_to_req_indices",
    "query_start_loc",
    "c2a_index_block_table", "c1a_index_block_table",
    "state_block_table",
    "window_cache",
    "window_scales",
    "compressed_cache",
    "compressed_scales",
    "index_cache",
    "index_scales",
    "state_cache",
    "final_norm_weight",
    "lm_head_weight",
    "logit_rows",
    "x_hc_out", "pre_mix_out",
    "logits",
    "sampled_ids",
)


_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}


class PrefillCheckpoint:
    """Read requested original checkpoint tensors and selected embedding rows."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
            raise ValueError("checkpoint must use the official block-32 FP8/FP4 format")
        required = {
            "hidden_size": M.hidden_size,
            "num_hidden_layers": M.num_hidden_layers,
            "num_attention_heads": M.num_attention_heads,
            "head_dim": M.head_dim,
            "sliding_window": M.sliding_window,
            "kv_source_layer_ids": list(M.kv_source_layer_ids),
            "index_source_layer_ids": list(M.index_source_layer_ids),
            "engram_layer_ids": list(M.engram_layer_ids),
            "engram_num_embeddings": list(M.engram_num_embeddings),
            "engram_max_ngram_size": M.engram_max_ngram_size,
            "engram_n_heads": M.engram_n_heads,
            "engram_head_dim": M.engram_head_dim,
            "engram_compressed_vocab_size": M.engram_compressed_vocab_size,
            "engram_vocab_size": M.engram_vocab_size,
            "engram_pad_token_id": M.engram_pad_token_id,
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype = _DTYPES[entry["dtype"]]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if start < 0 or end - start != expected_bytes or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"invalid safetensors range for {name}")
            file.seek(8 + header_size + start)
            payload = bytearray(file.read(expected_bytes))
        if len(payload) != expected_bytes:
            raise ValueError(f"truncated safetensors payload for {name}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def tensor_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Read selected table rows without loading the complete tensor."""
        if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("table row ids must have an integer dtype")
        ids = rows.to(dtype=torch.int64, device="cpu").flatten()
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if len(shape) != 2 or bool(((ids < 0) | (ids >= shape[0])).any()):
                raise ValueError(f"invalid table rows for {name}")
            row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
            start, end = entry["data_offsets"]
            if (
                start < 0
                or end - start != shape[0] * row_bytes
                or 8 + header_size + end > path.stat().st_size
            ):
                raise ValueError(f"invalid safetensors range for {name}")
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            payload = bytearray(unique.numel() * row_bytes)
            for index, row in enumerate(unique.tolist()):
                file.seek(8 + header_size + start + row * row_bytes)
                data = file.read(row_bytes)
                if len(data) != row_bytes:
                    raise ValueError(f"truncated safetensors row for {name}")
                payload[index * row_bytes : (index + 1) * row_bytes] = data
        if not ids.numel():
            return torch.empty(*rows.shape, shape[1], dtype=dtype)
        table = torch.frombuffer(payload, dtype=dtype).reshape(-1, shape[1])
        # CPU indexing is unavailable for FP8; index its byte representation.
        return (
            table.view(torch.uint8)
            .reshape(unique.numel(), row_bytes)[inverse]
            .contiguous()
            .view(dtype)
            .reshape(*rows.shape, shape[1])
        )


def forward_weight_layouts(local_experts: int):
    """Return TP1 rank-local weight layouts shared by prefill and decode."""
    layers = M.num_hidden_layers
    d = M.hidden_size
    inter = M.moe_intermediate_size
    local_h = M.num_attention_heads // TP_SIZE
    local_groups = M.o_groups // TP_SIZE
    o_width = local_groups * M.o_lora_rank
    o_group_in = M.num_attention_heads * M.head_dim // M.o_groups
    experts = C.EP_SIZE * local_experts
    packed_rows = d * inter // 256
    kv_sources = len(M.kv_source_layer_ids)
    index_sources = len(M.index_source_layer_ids)
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
        "index_wq_b": ((index_sources, M.q_lora_rank, M.index_n_heads * M.index_head_dim), torch.float8_e4m3fn),
        "index_wq_b_scale": ((index_sources * (M.q_lora_rank // 32), M.index_n_heads * M.index_head_dim), torch.float8_e8m0fnu),
        "index_weights_proj": ((index_sources, d, M.index_n_heads), torch.bfloat16),
        "hc_ffn_fn": ((layers * M.mix_hc, M.hc_dim), torch.float32),
        "hc_ffn_scale": ((layers * 3,), torch.float32),
        "hc_ffn_base": ((layers * M.mix_hc,), torch.float32),
        "ffn_norm_weight": ((layers * d,), torch.bfloat16),
        "gate_weight": ((layers * experts, d), torch.float32),
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
        "lm_head_weight": ((M.vocab_size // TP_SIZE, d), torch.bfloat16),
    }


class ResidentPrefillWeights:
    """Stream native checkpoint weights into named TP1 forward tensors."""

    fp8_projections = (
        ("wq_a", "attn.wq_a"), ("wq_b", "attn.wq_b"), ("wkv", "attn.wkv"),
        ("wo_a", "attn.wo_a"), ("wo_b", "attn.wo_b"),
        ("shared_w1", "ffn.shared_experts.w1"), ("shared_w2", "ffn.shared_experts.w2"),
        ("shared_w3", "ffn.shared_experts.w3"),
    )
    layer_projections = (
        ("hc_attn_fn", "hc_attn_fn"), ("hc_attn_scale", "hc_attn_scale"),
        ("hc_attn_base", "hc_attn_base"), ("attn_norm_weight", "attn_norm.weight"),
        ("q_norm_weight", "attn.q_norm.weight"), ("kv_norm_weight", "attn.kv_norm.weight"),
        ("attn_sink", "attn.attn_sink"), ("hc_ffn_fn", "hc_ffn_fn"),
        ("hc_ffn_scale", "hc_ffn_scale"), ("hc_ffn_base", "hc_ffn_base"),
        ("ffn_norm_weight", "ffn_norm.weight"), ("gate_weight", "ffn.gate.weight"),
        ("correction_bias", "ffn.gate.bias"),
    )

    def __init__(self, checkpoint: PrefillCheckpoint, *, ep_size: int = 4):
        if isinstance(ep_size, bool) or not isinstance(ep_size, Integral) or ep_size not in (4, 8):
            raise ValueError("resident prefill requires EP4 or EP8")
        if TP_SIZE != 1 or ep_size != C.EP_SIZE:
            raise ValueError("resident weights require TP1 and the configured EP size")
        self.ranks = int(ep_size)
        self.local_experts = M.n_routed_experts // self.ranks
        self.checkpoint = checkpoint
        self.tensor_layouts = forward_weight_layouts(self.local_experts)
        self.entries = []
        for layer in range(M.num_hidden_layers):
            for target, suffix in self.fp8_projections:
                self.entries.append((target, f"layers.{layer}.{suffix}", layer, M.num_hidden_layers, "fp8"))
            for target, suffix in self.layer_projections:
                self.entries.append((target, f"layers.{layer}.{suffix}", layer, M.num_hidden_layers, "plain"))
        for slot, layer in enumerate(M.kv_source_layer_ids):
            self.entries.append(("compressor_norm_weight", f"layers.{layer}.attn.compressor.norm.weight",
                                 slot, len(M.kv_source_layer_ids), "plain"))
        for slot, layer in enumerate(M.index_source_layer_ids):
            prefix = f"layers.{layer}.attn.indexer."
            self.entries.extend((
                ("index_wq_b", prefix + "wq_b", slot, len(M.index_source_layer_ids), "fp8"),
                ("index_wk", prefix + "wk.weight", slot, len(M.index_source_layer_ids), "transpose"),
                ("index_norm_weight", prefix + "k_norm.weight", slot, len(M.index_source_layer_ids), "plain"),
                ("index_weights_proj", prefix + "weights_proj.weight", slot, len(M.index_source_layer_ids), "transpose"),
            ))
        c2a_sources = [source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2]
        for slot, layer in enumerate(c2a_sources):
            for projection in ("wkv", "wgate"):
                name = f"layers.{layer}.attn.compressor.{projection}.weight"
                self.entries.append(("c2a_compressor_" + projection, name, slot, len(c2a_sources), "bf16_nz"))
        c1a_source = next(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 1)
        self.entries.extend((
            ("c1a_compressor_wkv", f"layers.{c1a_source}.attn.compressor.wkv.weight", 0, 1, "bf16_nz"),
            ("final_norm_weight", "norm.weight", 0, 1, "plain"),
            ("lm_head_weight", "head.weight", 0, 1, "plain"),
        ))
        for slot, layer in enumerate(M.engram_layer_ids):
            self.entries.append(("engram_wkv", f"layers.{layer}.engram.wkv", slot, 2, "fp8"))
            self.entries.append(("engram_weight", f"layers.{layer}.engram.q_weight", slot, 2, "engram_gate"))
            if f"layers.{layer}.engram.k_weight" not in checkpoint.weight_map:
                raise ValueError(f"checkpoint is missing layer {layer} Engram gate weights")
        for target, name, slot, slots, kind in self.entries:
            optional = (target in ("index_wk", "index_norm_weight")
                        and M.index_source_layer_ids[slot] not in M.kv_source_layer_ids)
            required = (name + ".weight", name + ".scale") if kind == "fp8" else (name,)
            for key in required:
                if key not in checkpoint.weight_map and not optional:
                    raise ValueError(f"checkpoint is missing required forward weight: {key}")

    @property
    def bytes_per_rank(self):
        return sum(math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                   for shape, dtype in self.tensor_layouts.values())

    def upload(self, runtime, rank: int, handles):
        """Upload one projection at a time into persistent named allocations."""
        from quant import pack_bf16_nz, pack_fp8_nz, pack_mx_b_scale
        from expert_routed import pack_mxfp4_weight_tiles

        if rank not in range(self.ranks):
            raise ValueError(f"expert rank must be in [0, {self.ranks})")
        checkpoint = self.checkpoint

        def copy(target, data, slot=0, slots=1):
            shape, dtype = self.tensor_layouts[target]
            elements = math.prod(shape) // slots
            data = data.contiguous()
            if data.dtype != dtype or data.numel() != elements:
                raise ValueError(f"checkpoint tensor does not match {target}: {data.shape}, {data.dtype}")
            runtime.copy_to(handles[target].data_ptr, data.data_ptr(), data.numel() * data.element_size(),
                            dst_offset=slot * elements * data.element_size(), worker_id=rank)

        for target, name, slot, slots, kind in self.entries:
            if kind == "fp8":
                shape, dtype = self.tensor_layouts[target]
                weight = checkpoint.tensor(name + ".weight")
                scale = checkpoint.tensor(name + ".scale")
                n, k = weight.shape
                expected = ((M.o_groups * M.o_lora_rank, shape[-2]) if target == "wo_a"
                            else (shape[-1], shape[-2]))
                valid_payload = weight.dtype == torch.float8_e4m3fn and (n, k) == expected
                valid_scale = scale.dtype == torch.float8_e8m0fnu and tuple(scale.shape) == (n // 32, k // 32)
                if not valid_payload or not valid_scale or n % 32 or k % 64:
                    raise ValueError(f"unsupported native MXFP8 weight layout: {name}")
                logical_scale = scale.view(torch.uint8).repeat_interleave(32, dim=0).t().contiguous()
                if target == "wo_a":
                    weight = weight.reshape(M.o_groups, M.o_lora_rank, k).transpose(-1, -2)
                    logical_scale = logical_scale.reshape(k // 32, M.o_groups, M.o_lora_rank).permute(1, 0, 2)
                else:
                    weight = weight.t()
                if target in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "index_wq_b"):
                    weight = pack_fp8_nz(weight)
                copy(target, weight, slot, slots)
                packed_scale = pack_mx_b_scale(logical_scale).view(torch.float8_e8m0fnu)
                copy(target + "_scale", packed_scale, slot, slots)
            else:
                shape, dtype = self.tensor_layouts[target]
                if name not in checkpoint.weight_map:
                    value = torch.zeros(math.prod(shape) // slots, dtype=dtype)
                else:
                    value = checkpoint.tensor(name)
                    if value.dtype not in (torch.float32, torch.bfloat16):
                        raise ValueError(f"unsupported inference weight dtype: {name}")
                    valid_head = value.dtype == torch.bfloat16 and tuple(value.shape) == shape
                    if target == "lm_head_weight" and not valid_head:
                        raise ValueError("the native LM head requires the released BF16 checkpoint matrix")
                    value = value.to(dtype)
                    if kind in ("transpose", "bf16_nz"):
                        value = value.t()
                    elif kind == "engram_gate":
                        value = value * checkpoint.tensor(name.replace("q_weight", "k_weight")).float()
                if target in ("index_wk", "index_weights_proj"):
                    value = pack_bf16_nz(value.reshape(-1, shape[-1]))
                if kind == "bf16_nz":
                    value = pack_bf16_nz(value.reshape(shape[-2:]))
                copy(target, value, slot, slots)
        for layer in range(M.num_hidden_layers):
            for projection in ("w1", "w2", "w3"):
                k, n = ((M.moe_intermediate_size, M.hidden_size) if projection == "w2"
                        else (M.hidden_size, M.moe_intermediate_size))
                logical_scales = torch.empty((self.local_experts * (k // 32), n), dtype=torch.uint8)
                for local in range(self.local_experts):
                    expert = rank * self.local_experts + local
                    stem = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                    weight = checkpoint.tensor(stem + ".weight")
                    scale = checkpoint.tensor(stem + ".scale")
                    valid_payload = weight.dtype == torch.int8 and tuple(weight.shape) == (n, k // 2)
                    valid_scale = scale.dtype == torch.float8_e8m0fnu and tuple(scale.shape) == (n, k // 32)
                    if not valid_payload or not valid_scale:
                        raise ValueError(f"unsupported native MXFP4 weight layout: {stem}")
                    packed = pack_mxfp4_weight_tiles(weight, 256, 256)
                    copy("routed_" + projection, packed, layer * self.local_experts + local,
                         M.num_hidden_layers * self.local_experts)
                    logical_scales[local * (k // 32) : (local + 1) * (k // 32)] = scale.view(torch.uint8).t()
                packed_scale = pack_mx_b_scale(logical_scales).view(torch.float8_e8m0fnu)
                copy("routed_" + projection + "_scale", packed_scale, layer, M.num_hidden_layers)

    def lookup_rows(self, token_ids: torch.Tensor):
        """Read only the prompt's BF16 token embeddings."""
        return self.checkpoint.tensor_rows("embed.weight", token_ids).to(torch.bfloat16)


def prepare_prefill_inputs(
    embedding: torch.Tensor,
    capacity: int,
    *,
    start_pos: int = 0,
    max_seq_len: int | None = None,
    cache_state: Mapping[str, torch.Tensor] | None = None,
    rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare one TP1 rank's next embedding chunk and native cache metadata.

    ``capacity`` bounds current rows, independently of ``max_seq_len``, which
    sizes the retained history. Reuse rope_tables across calls; their full
    table capacity may exceed capacity. Pass the previous call's cache tensors after
    its completion and advance ``start_pos`` by its valid count. An empty
    sequence has zero work and preserves every cache byte. This convenience
    helper owns one request at physical slot zero per DP rank; serving may
    supply its own paged metadata directly to the forward entry instead.
    """
    if embedding.ndim != 2 or embedding.shape[1] != M.hidden_size:
        raise ValueError("embedding must have shape [current tokens, hidden size]")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity < 16 or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = embedding.shape[0]
    if not 0 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("current rows must fit the configured prefill capacity")
    if isinstance(start_pos, bool) or not isinstance(start_pos, Integral) or start_pos < 0:
        raise ValueError("start_pos must be a non-negative integer")
    max_seq_len = capacity if max_seq_len is None else max_seq_len
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, Integral):
        raise ValueError("max_seq_len must be a positive integer")
    if not 1 <= max_seq_len <= M.max_position_embeddings or start_pos + count > max_seq_len:
        raise ValueError("current rows exceed the retained history capacity")
    if start_pos and cache_state is None:
        raise ValueError("continuation requires the preceding call's retained cache state")

    data_pages = (max_seq_len + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    pages = data_pages + 1
    table = torch.arange(data_pages, dtype=torch.int32).reshape(1, -1)
    c2a_pages = max(1, (max_seq_len // 2 + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE)
    c2a_table = torch.arange(c2a_pages, dtype=torch.int32).reshape(1, -1)
    compressed_tables = {
        source: c2a_table if M.compress_ratios[source] == 2 else table
        for source in M.kv_source_layer_ids
    }
    state_table = torch.zeros((1, 1), dtype=torch.int32)
    state_tables = {
        source: state_table
        for source in M.kv_source_layer_ids
        if M.compress_ratios[source] == 2
    }
    query_starts = torch.tensor([0, count], dtype=torch.int32)
    token_to_req_indices = torch.zeros(count, dtype=torch.int32)
    positions = torch.arange(start_pos, start_pos + count, dtype=torch.int32)
    window_slots, window_indices, _ = window_metadata(positions, token_to_req_indices, table)
    c2a_slots = paged_slots(positions, token_to_req_indices, c2a_table, C.BLOCK_SIZE, 2, True)
    c1a_slots = paged_slots(positions, token_to_req_indices, table)
    freqs_cos, freqs_sin = prepare_forward_rope_tables(max_seq_len, rope_tables)

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[: value.shape[0]] = value
        return result

    initial_pre = torch.zeros((capacity, M.hc_mult), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    logit_rows = torch.full((16,), -1, dtype=torch.int32)
    logit_rows[0] = count - 1
    window_shape = (M.num_hidden_layers, pages, C.BLOCK_SIZE, 1)
    source_shape = (len(M.kv_source_layer_ids), pages, C.BLOCK_SIZE, 1)

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
        return torch.full(shape, fill, dtype=torch.uint8).view(dtype)

    values = {
        "embedding": pad(embedding),
        "engram_lookup": torch.zeros((2, capacity, ENGRAM_K), dtype=torch.bfloat16),
        "embedding_ids": torch.arange(capacity, dtype=torch.int64),
        "initial_pre": initial_pre,
        "identity_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "counts": torch.tensor([count, count], dtype=torch.int32),
        "host_counts": torch.tensor([count, count], dtype=torch.int32),
        "replay_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "position_ids": pad(positions, -1).repeat(2, 1),
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
        "window_slots": pad(window_slots, -1).repeat(2, 1),
        "window_indices": pad(window_indices, -1).repeat(2, 1, 1),
        "c2a_compressed_slots": pad(c2a_slots, -1),
        "c1a_compressed_slots": pad(c1a_slots, -1),
        "token_to_req_indices": pad(token_to_req_indices, -1).repeat(2, 1),
        "query_start_loc": query_starts.repeat(2, 1),
        "c2a_index_block_table": c2a_table,
        "c1a_index_block_table": table,
        "state_block_table": state_table,
        "window_cache": native_cache("window_cache", (*window_shape, M.head_dim), torch.float8_e4m3fn),
        "window_scales": native_cache(
            "window_scales", (*window_shape, M.head_dim // C.WINDOW_CACHE_GROUP), torch.bfloat16, 1.0,
        ),
        "compressed_cache": native_cache("compressed_cache", (*source_shape, M.head_dim // 2), torch.uint8),
        "compressed_scales": native_cache(
            "compressed_scales", (*source_shape, M.head_dim // C.COMPRESSED_CACHE_GROUP), torch.bfloat16, 1.0,
        ),
        "index_cache": native_cache("index_cache", (*source_shape, M.index_head_dim // 2), torch.uint8),
        "index_scales": native_cache(
            "index_scales", (*source_shape, M.index_head_dim // C.INDEX_CACHE_GROUP), torch.float8_e8m0fnu, 127,
        ),
        "state_cache": native_cache(
            "state_cache", (len(state_tables), 2, C.STATE_CAPACITY, 2 * M.head_dim), torch.float32,
        ),
        "logit_rows": logit_rows,
    }
    validate_prefill_caches(values, table, compressed_tables, state_tables)
    return values


def prepare_resident_inputs(
    weights: ResidentPrefillWeights,
    token_ids: torch.Tensor | Sequence[int],
    capacity: int,
    *,
    start_pos: int = 0,
    max_seq_len: int | None = None,
    cache_state: Mapping[str, torch.Tensor] | None = None,
    rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
    engram: HostEngram,
    request_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Prepare checkpoint embeddings and causal replay for one prefill rank."""
    ids = torch.as_tensor(token_ids, device="cpu")
    if ids.ndim != 1 or ids.numel() and ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a one-dimensional integer sequence")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = ids.numel()
    if not 0 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("current rows must fit the configured prefill capacity")
    if bool(((ids < 0) | (ids >= M.vocab_size)).any()):
        raise ValueError("token id is outside the checkpoint vocabulary")
    ids = ids.long()
    embedding = weights.lookup_rows(ids)

    values = prepare_prefill_inputs(
        embedding, capacity, start_pos=start_pos, max_seq_len=max_seq_len,
        cache_state=cache_state, rope_tables=rope_tables,
    )
    values["engram_lookup"] = engram.lookup(ids, capacity, start_pos, request_id=request_id)
    return values


def generate(checkpoint, compiled, device_ids, prompts, max_new_tokens, *, chunk_size=None, decode_compiled=None):
    """Generate one independent continuation per rank using resident checkpoint weights.

    Prefill and subsequent one-token calls retain native device caches. Only
    current-token metadata is uploaded; completed history is never recomputed.
    With decode_compiled, continuation uses decode_fwd with shared resident
    weights and cache pools. Otherwise, one-token calls use the prefill composition.
    """
    from tokenizers import Tokenizer
    from pypto.runtime import DeviceTensor
    from utils import DECODE_CACHE_NAMES, decode_cache_views, prepare_decode_inputs

    # Validate the host-row Engram ABI and compilation shapes.
    parameters = compiled._get_metadata()[0]
    if (len(parameters) != len(PREFILL_TENSORS)
            or parameters[0].shape != [RANKS, CAPACITY, D]):
        raise ValueError("--runtime-dir must contain prefill compiled for the same EP and capacity with host Engram rows")
    decode_names = []
    if decode_compiled is not None:
        decode_parameters = decode_compiled._get_metadata()[0]
        decode_names = [parameter.name.removesuffix("__ssa_v0") for parameter in decode_parameters]
        if decode_names[:4] != ["engram_lookup", "engram_wkv", "engram_wkv_scale", "engram_weight"]:
            raise ValueError("decode artifact must use the host-row Engram ABI")
        decode_shape = decode_parameters[0].shape
        decode_rows = C.DECODE_INPUT_MAX_TOKENS
        if (decode_shape[:2] != [RANKS, 2] or decode_shape[2] not in (-1, decode_rows)
                or decode_shape[3] != ENGRAM_K):
            raise ValueError("decode artifact must match EP and the configured decode capacity")
    if len(device_ids) != RANKS or len(set(device_ids)) != RANKS or len(prompts) != RANKS:
        raise ValueError(f"prefill requires {RANKS} distinct devices and one prompt per rank")

    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    spec = importlib.util.spec_from_file_location(
        "v41_checkpoint_encoding", checkpoint.root / "encoding/encoding.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("checkpoint is missing the official prompt encoder")
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    sequences = [
        tokenizer.encode(
            encoding.encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat"),
            add_special_tokens=False,
        ).ids
        for prompt in prompts
    ]
    chunk_size = CAPACITY if chunk_size is None else chunk_size
    if not 1 <= chunk_size <= CAPACITY:
        raise ValueError("chunk_size must fit CAPACITY and be at most 4096")
    max_seq_len = max(map(len, sequences)) + max_new_tokens
    if max_seq_len > M.max_position_embeddings:
        raise ValueError("prompt plus generation exceeds the model position bound")
    tokenizer_config = json.loads((checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    if eos_id is None:
        raise ValueError("checkpoint EOS token is missing from its tokenizer")
    weights = ResidentPrefillWeights(checkpoint, ep_size=RANKS)
    engram = HostEngram(checkpoint, tokenizer)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, ring_heap=2 * 1024**3,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    outputs = {
        "x_hc_out": ((CAPACITY, HC_MULT, D), torch.float32),
        "pre_mix_out": ((CAPACITY, HC_MULT), torch.float32),
        "logits": ((16, VOCAB), torch.float32),
        "sampled_ids": ((16, 8), torch.int32),
    }
    constants = {*weights.tensor_layouts, "mxfp4_pair_lut"}
    cache_names = {
        "window_cache", "window_scales", "compressed_cache", "compressed_scales",
        "index_cache", "index_scales", "state_cache",
    }
    constants.update(cache_names)
    constants.update(("freqs_cos", "freqs_sin"))
    generated = [[] for _ in prompts]
    finished = [False] * RANKS
    # Host control tensors must be shared and registered before the chip-worker fork.
    decode_counts = {
        name: torch.zeros(RANKS, dtype=torch.int32).share_memory_()
        for name in ("moe_num_tokens", "attention_num_tokens")
    }
    prefill_host_counts = torch.zeros((RANKS, 2), dtype=torch.int32).share_memory_()
    print("CED_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
    start = time.monotonic()
    # Completed forwards retire signals; native cache allocations persist.
    extra = [] if decode_compiled is None else [decode_compiled]
    with compiled.prepare(
        config, extra_compiled=extra, persistent=True, reset_persistent_windows=False,
        inherited_host_tensors=list(decode_counts.values()),
    ) as runtime:
        per_rank = [{} for _ in device_ids]
        host_cache = []
        rope_tables = None
        for rank, tensors in enumerate(per_rank):
            inputs = prepare_resident_inputs(
                weights, sequences[rank][:chunk_size], CAPACITY,
                max_seq_len=max_seq_len, rope_tables=rope_tables,
                engram=engram, request_id=rank,
            )
            rope_tables = inputs["freqs_cos"], inputs["freqs_sin"]
            host_cache.append({name: inputs[name] for name in cache_names})
            inputs["mxfp4_pair_lut"] = build_mxfp4_pair_lut()
            for name in PREFILL_TENSORS:
                if name == "host_counts":
                    continue
                if name in weights.tensor_layouts:
                    shape, dtype = weights.tensor_layouts[name]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                elif name in outputs:
                    shape, dtype = outputs[name]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                else:
                    value = inputs[name]
                    tensors[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def upload(rank):
            weights.upload(runtime, rank, per_rank[rank])
            print("CED_RANK_WEIGHTS_READY", rank, flush=True)

        with ThreadPoolExecutor(max_workers=RANKS) as executor:
            list(executor.map(upload, range(RANKS)))
        stacked = {
            name: StackedDeviceTensor(
                [rank[name] for rank in per_rank], (RANKS, *per_rank[0][name].shape), tuple(range(RANKS))
            )
            for name in PREFILL_TENSORS if name != "host_counts"
        }
        stacked["host_counts"] = prefill_host_counts
        logits = torch.empty((RANKS, 16, VOCAB), dtype=torch.float32)
        sampled_ids = torch.empty((RANKS, 16, 8), dtype=torch.int32)
        committed = [0] * RANKS
        last_logits = torch.empty((RANKS, VOCAB), dtype=torch.float32)
        last_ids = [0] * RANKS

        decode_rank = [{} for _ in device_ids]
        decode_stacked = {}
        if decode_compiled is not None:
            for rank, tensors in enumerate(decode_rank):
                values = prepare_resident_inputs(
                    weights, [], decode_rows, start_pos=0, max_seq_len=max_seq_len,
                    cache_state=host_cache[rank], rope_tables=rope_tables, engram=engram, request_id=rank,
                )
                values = prepare_decode_inputs(values)
                cache_views = decode_cache_views(host_cache[rank])
                for name in decode_names:
                    if name in decode_counts:
                        continue
                    if name in DECODE_CACHE_NAMES:
                        source = per_rank[rank][DECODE_CACHE_NAMES[name]]
                        tensors[name] = DeviceTensor(source.data_ptr, cache_views[name].shape, source.dtype,
                                                     buffer=source.buffer)
                    elif name in constants or name in ("logits", "sampled_ids"):
                        tensors[name] = per_rank[rank][name]
                    else:
                        value = values[name]
                        tensors[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
            decode_stacked = {
                name: StackedDeviceTensor(
                    [rank[name] for rank in decode_rank], (RANKS, *decode_rank[0][name].shape), tuple(range(RANKS)),
                )
                for name in decode_names if name not in decode_counts
            }
            decode_stacked.update(decode_counts)

        def run_rows(rows, *, decode=False):
            rank_tensors = decode_rank if decode else per_rank
            capacity = decode_rows if decode else CAPACITY
            for rank, (tensors, token_ids) in enumerate(zip(rank_tensors, rows)):
                inputs = prepare_resident_inputs(
                    weights, token_ids, capacity,
                    start_pos=committed[rank], max_seq_len=max_seq_len,
                    cache_state=host_cache[rank], rope_tables=rope_tables,
                    engram=engram, request_id=rank,
                )
                if decode:
                    inputs = prepare_decode_inputs(inputs)
                    for counts in decode_counts.values():
                        counts[rank] = len(token_ids)
                for name, value in inputs.items():
                    if not decode and name == "host_counts":
                        prefill_host_counts[rank].copy_(value)
                        continue
                    if name not in constants:
                        runtime.copy_to(
                            tensors[name].data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                            worker_id=rank,
                        )
            program = decode_compiled if decode else compiled
            arguments = decode_stacked if decode else stacked
            names = decode_names if decode else PREFILL_TENSORS
            runtime.run(program, *[arguments[name] for name in names], config=config)
            runtime.copy_stacked_from(arguments["logits"], logits)
            runtime.copy_stacked_from(arguments["sampled_ids"], sampled_ids)
            for rank, token_ids in enumerate(rows):
                if not token_ids:
                    continue
                if not bool(torch.isfinite(logits[rank, 0]).all()):
                    raise FloatingPointError(f"nonfinite logits on rank {rank}")
                token = int(sampled_ids[rank, 0, 0])
                if token != int(logits[rank, 0].argmax()):
                    raise AssertionError("device greedy sampling disagrees with copied logits")
                last_logits[rank].copy_(logits[rank, 0])
                last_ids[rank] = token
                committed[rank] += len(token_ids)

        while any(committed[rank] < len(sequence) for rank, sequence in enumerate(sequences)):
            rows = [sequence[committed[rank]:committed[rank] + chunk_size]
                    for rank, sequence in enumerate(sequences)]
            run_rows(rows)
            print("PREFILL_CHUNK_COMPLETE", committed, flush=True)
        for step in range(max_new_tokens):
            tick = time.monotonic()
            if step:
                run_rows([[tokens[-1]] if not finished[rank] else []
                          for rank, tokens in enumerate(generated)], decode=decode_compiled is not None)
            for rank, token in enumerate(last_ids):
                if not finished[rank]:
                    generated[rank].append(token)
                    finished[rank] = token == eos_id
            print("CED_GENERATION_STEP", json.dumps({
                "step": step, "seconds": time.monotonic() - tick,
                "token_ids": last_ids,
                "text": [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in generated],
            }, ensure_ascii=False), flush=True)
            if all(finished):
                break
    return {
        "platform": "a5", "devices": device_ids, "tp": 1, "ep": RANKS, "dp": RANKS,
        "precision": "official_bf16_mxfp8_mxfp4", "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY, "chunk_size": chunk_size, "history_capacity": max_seq_len,
        "full_prefix_recomputation": False, "engram_enabled": True, "engram_lookup_device": "host",
        "continuation_forward": "decode_fwd" if decode_compiled is not None else "prefill_fwd",
        "shared_resident_weights": decode_compiled is not None,
        "layers_per_forward": 40, "l2_invocations_per_rank_per_forward": 1,
        "seconds": time.monotonic() - start,
        "cases": [
            {"prompt": prompt, "token_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=False),
             "eos_reached": done}
            for prompt, tokens, done in zip(prompts, generated, finished)
        ],
    }


def build_tensor_specs():
    """Build a checkpoint-free, 40-layer CED prefill smoke fixture."""
    from golden import TensorSpec
    from prefill_moe import _route_bias

    weight_layouts = forward_weight_layouts(LOCAL_EXPERTS)
    outputs = {
        "x_hc_out": ([RANKS, CAPACITY, HC_MULT, D], torch.float32),
        "pre_mix_out": ([RANKS, CAPACITY, HC_MULT], torch.float32),
        "logits": ([RANKS, 16, VOCAB], torch.float32),
        "sampled_ids": ([RANKS, 16, 8], torch.int32),
    }
    values = prepare_prefill_inputs(torch.zeros((CAPACITY, D), dtype=torch.bfloat16), CAPACITY)
    values["mxfp4_pair_lut"] = build_mxfp4_pair_lut()
    specs = []
    for name in PREFILL_TENSORS:
        if name in weight_layouts:
            rank_shape, dtype = weight_layouts[name]
            shape = [RANKS, *rank_shape]
            value = 1.0 if dtype == torch.float8_e8m0fnu or name == "final_norm_weight" else 0.0
            if name == "correction_bias":
                value = _route_bias().repeat(RANKS, N_LAYERS)
        elif name in outputs:
            shape, dtype = outputs[name]
            value = None
        else:
            tensor = values[name]
            value = tensor.unsqueeze(0).repeat(RANKS, *([1] * tensor.ndim))
            shape, dtype = list(value.shape), value.dtype
        specs.append(TensorSpec(name, shape, dtype, init_value=value))
    return specs


def validate_prefill_caches(
    cache_state: Mapping[str, torch.Tensor],
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
    state_block_tables: Mapping[int, torch.Tensor],
) -> None:
    """Validate native prefill cache layouts and source-local page/state ownership."""
    cache_names = (
        "window_cache", "window_scales", "compressed_cache", "compressed_scales", "index_cache", "index_scales",
    )
    for name in cache_names:
        value = cache_state[name]
        if value.ndim not in (5, 6) or not value.is_contiguous():
            raise ValueError(f"{name} must preserve the contiguous native source/page layout")
        sources = M.num_hidden_layers if name.startswith("window") else len(M.kv_source_layer_ids)
        if value.shape[-5] != sources or value.shape[-4] < 1 or tuple(value.shape[-3:-1]) != (C.BLOCK_SIZE, 1):
            raise ValueError(f"{name} has an incompatible source or page geometry")
        expected_dtype = {
            "window_cache": torch.float8_e4m3fn, "window_scales": torch.bfloat16,
            "compressed_cache": torch.uint8, "compressed_scales": torch.bfloat16,
            "index_cache": torch.uint8, "index_scales": torch.float8_e8m0fnu,
        }[name]
        if value.dtype != expected_dtype:
            raise ValueError(f"{name} must retain its native payload or scale dtype")
        expected_width = {
            "window_cache": M.head_dim, "window_scales": M.head_dim // C.WINDOW_CACHE_GROUP,
            "compressed_cache": M.head_dim // 2,
            "compressed_scales": M.head_dim // C.COMPRESSED_CACHE_GROUP,
            "index_cache": M.index_head_dim // 2, "index_scales": M.index_head_dim // C.INDEX_CACHE_GROUP,
        }[name]
        if value.shape[-1] != expected_width:
            raise ValueError(f"{name} has an incompatible native row width")
        reference = cache_state["window_cache"]
        if value.shape[:-5] != reference.shape[:-5] or value.shape[-4] != reference.shape[-4]:
            raise ValueError("all prefill cache pools must preserve the same rank and page extents")
        if value.device != reference.device:
            raise ValueError("all cache payloads and scales must reside on the same device")
    state = cache_state["state_cache"]
    if state.ndim not in (4, 5) or not state.is_contiguous():
        raise ValueError("state_cache must preserve its contiguous source/request layout")
    if state.shape[-4] != 3 or state.shape[-3] < 1 or tuple(state.shape[-2:]) != (C.STATE_CAPACITY, 2 * M.head_dim):
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
    pages = cache_state["window_cache"].shape[1]
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

    validate_table("window block table", window_block_table, pages)
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
    if _shape_options.checkpoint:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--runtime-dir", type=Path, required=True, help="compiled host-row Engram prefill")
        parser.add_argument("--decode-runtime-dir", type=Path, help="compiled host-row Engram decode for continuation")
        parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE"))
        parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=[1])
        parser.add_argument("--ep", type=int, default=RANKS, choices=[4, 8])
        parser.add_argument("--dp", type=int, default=RANKS, choices=[4, 8])
        parser.add_argument("--experts-per-rank", type=int, default=LOCAL_EXPERTS)
        parser.add_argument("--capacity", type=int, default=CAPACITY)
        parser.add_argument("--prompt", action="append", help="one prompt per rank; provide exactly --ep prompts")
        parser.add_argument("--max-new-tokens", type=int, default=16)
        parser.add_argument("--chunk-size", type=int, default=CAPACITY)
        parser.add_argument("--report", type=Path)
        options = parser.parse_args()
        if options.experts_per_rank * options.ep != M.n_routed_experts or N_EXPERTS != M.n_routed_experts:
            parser.error("checkpoint generation requires all 384 routed experts")
        if (options.tp, options.ep, options.dp) != (C.TP_SIZE, RANKS, RANKS):
            raise ValueError("parallel sizes must match compilation: TP1 with EP4/DP4 or EP8/DP8")
        if options.capacity != CAPACITY or not 16 <= CAPACITY <= _MAX_CHUNK_TOKENS or CAPACITY % 16:
            raise ValueError("--capacity must match compilation and be a positive multiple of 16")
        device_ids = [int(value) for value in (options.devices or "").split(",") if value]
        if len(device_ids) != RANKS or len(set(device_ids)) != RANKS:
            raise ValueError(f"provide {RANKS} distinct task-allocated A5 devices")
        default_prompts = [
            "What is 2 + 3? Reply with just the number.",
            "What is the capital of France? Reply with just the city name.",
            "Translate 'hello' into Chinese. Reply with only the translation.",
            "Continue the sequence with three numbers: 1, 2, 3,",
        ]
        prompts = options.prompt or default_prompts * (RANKS // len(default_prompts))
        if len(prompts) != RANKS or options.max_new_tokens <= 0:
            raise ValueError(f"provide {RANKS} prompts and a positive --max-new-tokens")
        torch.set_num_threads(8)
        checkpoint = PrefillCheckpoint(options.checkpoint)
        compiled = DistributedCompiledProgram.from_dir(
            options.runtime_dir, backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )
        decode_compiled = None
        if options.decode_runtime_dir:
            decode_compiled = DistributedCompiledProgram.from_dir(
                options.decode_runtime_dir, backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            )
        report = generate(
            checkpoint, compiled, device_ids, prompts, options.max_new_tokens,
            chunk_size=options.chunk_size, decode_compiled=decode_compiled,
        )
        if options.report:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("CED_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    else:
        from utils import upload_forward_inputs
        from golden import normalize_config, run
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
        parser.add_argument("--sim", dest="execution_mode", action="store_true")
        parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=[1])
        parser.add_argument("--ep", type=int, default=RANKS, choices=[2, 4, 8])
        parser.add_argument("--dp", type=int, default=RANKS, choices=[2, 4, 8])
        parser.add_argument("--experts-per-rank", type=int, default=LOCAL_EXPERTS, help="routed experts on each EP rank")
        parser.add_argument("--capacity", type=int, default=CAPACITY)
        parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(RANKS)))
        parser.add_argument("--compile-only", action="store_true")
        parser.add_argument("--dump-passes", action="store_true")
        parser.add_argument("--enable-scope-stats", action="store_true")
        parser.add_argument("--log-level", default=None)
        args = parser.parse_args()
        if args.execution_mode:
            parser.error("this forward entry requires hardware; --sim is unsupported")
        if (args.tp, args.ep, args.dp, args.experts_per_rank, args.capacity) != (1, RANKS, RANKS, LOCAL_EXPERTS, CAPACITY):
            parser.error("parallel sizes, expert count and capacity must match the import-time configuration")
        devices = [int(device) for device in args.device.split(",") if device]
        if len(devices) != RANKS or len(set(devices)) != RANKS:
            parser.error(f"need exactly {RANKS} distinct devices, got {devices}")
        specs = build_tensor_specs()
        config = dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            dump_passes=args.dump_passes,
            enable_scope_stats=args.enable_scope_stats,
            log_level=args.log_level,
            ring_heap=(2 * 1024**3,) * 4,
            ring_dep_pool=131072,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        )
        result = run(
            fn=l3_prefill_fwd,
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
            host_inputs = {}
            upload_buffer = torch.empty(64 * 1024**2, dtype=torch.uint8).share_memory_()
            inherited = [*host_inputs.values(), upload_buffer]
            with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False, inherited_host_tensors=inherited) as runtime:
                parameters = compiled._get_metadata()[0]
                arguments = upload_forward_inputs(runtime, specs, parameters, len(devices), host_inputs, upload_buffer)
                runtime(*arguments, config=cfg)
            print("[FWD] Execution passed; numerical accuracy was not validated.", flush=True)
        if not result.passed:
            raise SystemExit(result.error or 1)
