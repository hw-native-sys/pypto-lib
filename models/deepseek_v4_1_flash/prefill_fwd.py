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
Engram is disabled, including host hashes, table lookup and weight preparation.
"""

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
from pypto.runtime import RunConfig, StackedDeviceTensor
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
    CMP_POSITIONS_DYN,
    PREFILL_MAX_TOKENS,
    Q_START_DYN,
    STATE_BLOCKS_DYN,
    STATE_CAPACITY,
    MOE_TOKENS,
    ROUTE_T_DYN,
)

from hc import hc_head
from hc import hc_mixes
from hc import hc_post
from hc import hc_pre, zero_bf16_padding
from input_pack import gather_rows, pack_x_hc
from lm_head import greedy_sample, lm_head
from utils import (
    PrefillCheckpoint, ResidentPrefillWeights, prepare_prefill_inputs, prepare_resident_inputs,
)
from moe import AUX_WIDTH, ROUTE_WIDTH, SIGNAL_PAD, moe
from compressor import STATE_WIDTH
from prefill_c1a_full import prefill_publish_decoder
from prefill_swa import prefill_swa_attention
from o_proj import prefill_tp_output_all_reduce
from prefill_c2a_full import prefill_c2a_full_attention
from prefill_c2a_reuse import prefill_c2a_reuse_attention
from prefill_c1a_reindex import prefill_c1a_reindex_attention
from prefill_c1a_reuse import prefill_c1a_reuse_attention
from quant import build_mxfp4_pair_lut
from rmsnorm import rms_norm_tiled, rms_norm_tiled_compensated
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS

RANKS = C.EP_SIZE
D = M.hidden_size
HC_MULT = M.hc_mult
VOCAB = M.vocab_size
PAGES = pl.dynamic("V41_PREFILL_HISTORY_PAGES")
C2A_TABLE_DYN = pl.dynamic("V41_PREFILL_C2A_TABLE")
C1A_TABLE_DYN = pl.dynamic("V41_PREFILL_C1A_TABLE")
LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
RECV_MAX = C.MOE_RECV_MAX
HC_DIM = M.hc_dim
HEAD_DIM = M.head_dim
INDEX_DIM = M.index_head_dim
INDEX_H = M.index_n_heads
INDEX_TOPK = M.index_topk
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
B8 = pl.dynamic("PREFILL_FP8_BLOCKS")
B4 = pl.dynamic("PREFILL_FP4_BLOCKS")
S8 = pl.dynamic("PREFILL_SCALE_BLOCKS")
B16 = pl.dynamic("PREFILL_BF16_VALUES")
B32 = pl.dynamic("PREFILL_FP32_VALUES")

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
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
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
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
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
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.FP32)
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
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.FP32)
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
            c1a_index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj,
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
            c1a_index_block_table, candidate_mask, index_wq_b, index_wq_b_scale, index_weights_proj,
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
            moe(
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
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    bank8: pl.Tensor[[B8, 128], pl.FP8E4M3FN],
    bank4: pl.Tensor[[B4, 128], pl.UINT8],
    scales8: pl.Tensor[[S8, 128], pl.FP8E8M0],
    bf16: pl.Tensor[[B16], pl.BF16],
    dense: pl.Tensor[[B32], pl.FP32],
    fp8_offsets: pl.Tensor[[40, 9], pl.INT64],
    fp8_scale_offsets: pl.Tensor[[40, 9], pl.INT64],
    bf16_offsets: pl.Tensor[[40, 9], pl.INT64],
    dense_offsets: pl.Tensor[[40, 11], pl.INT64],
    expert_offsets: pl.Tensor[[40, 3], pl.INT64],
    expert_scale_offsets: pl.Tensor[[40, 3], pl.INT64],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    rope_cos: pl.Tensor[[3, CAPACITY, 32], pl.FP32],
    rope_sin: pl.Tensor[[3, CAPACITY, 32], pl.FP32],
    compressed_cos: pl.Tensor[[CAPACITY, 32], pl.FP32],
    compressed_sin: pl.Tensor[[CAPACITY, 32], pl.FP32],
    publisher_cos: pl.Tensor[[CAPACITY, 32], pl.FP32],
    publisher_sin: pl.Tensor[[CAPACITY, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[2, CAPACITY], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[CAPACITY], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[CAPACITY], pl.INT32],
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
    final_norm: pl.Tensor[[D], pl.BF16],
    head: pl.Tensor[[VOCAB, D], pl.BF16],
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
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.FP32],
    attention_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.FP32],
    head_logits_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run both CED stages and the head with fresh, zeroed communication windows."""
    compressed_cache.bind_dynamic(1, PAGES)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(0, B_DYN)
    c2a_index_block_table.bind_dynamic(1, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(0, B_DYN)
    c1a_index_block_table.bind_dynamic(1, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    bank8.bind_dynamic(0, B8)
    bank4.bind_dynamic(0, B4)
    scales8.bind_dynamic(0, S8)
    bf16.bind_dynamic(0, B16)
    dense.bind_dynamic(0, B32)
    hidden_pair = pl.create_tensor([4 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([4 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    initial_mix = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
    gather_ready = pl.system.task_dummy(deps=[])
    pack_x_hc(embedding_ids, embedding, initial_hidden)
    gather_rows(initial_pre, identity_rows, initial_mix, gather_ready)
    topk_indices = pl.create_tensor([CAPACITY, INDEX_TOPK], dtype=pl.INT32)
    candidate_mask = pl.create_tensor([CAPACITY, pl.tensor.dim(compressed_cache, 1) * 128], dtype=pl.UINT8)
    # Keep stage handoff buffers outside attention scopes and reuse bounded
    # MoE slabs across all layers and rounds.
    attn_hidden = pl.create_tensor([CAPACITY, HC_MULT, D], dtype=pl.FP32)
    attn_pre = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_mixed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    slab_hidden = pl.create_tensor([2 * MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([2 * MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    slab_rows = pl.create_tensor([MOE_TOKENS], dtype=pl.INT32)
    moe_ready = pl.system.task_dummy(deps=[])
    for layer in pl.range(40):
        stage = layer // 20
        active = pl.read(counts, [stage])
        current_slot = (2 * stage + layer % 2) * CAPACITY
        next_slot = (2 * stage + (layer + 1) % 2) * CAPACITY
        hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [current_slot, 0, 0])
        delayed_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [current_slot, 0])
        next_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [next_slot, 0, 0])
        next_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [next_slot, 0])
        # Use 128-element rows: runtime view coordinates are uint32, while
        # their row-stride product and storage offsets are uint64.
        wq_a_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 0]), 0), pl.INDEX)
        wq_a_storage = pl.slice(bank8, [(D * Q_LORA) // 128, 128], [wq_a_offset // 128, 0])
        wq_a = pl.reshape(wq_a_storage, [D, Q_LORA])
        wq_a_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 0]), 0), pl.INDEX)
        wq_a_scale_storage = pl.slice(scales8, [(D // 32 * Q_LORA) // 128, 128], [wq_a_scale_offset // 128, 0])
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wq_a_scale_storage, [D // 32, Q_LORA])
        wq_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 1]), 0), pl.INDEX)
        wq_b_storage = pl.slice(bank8, [(Q_LORA * (LOCAL_H * HEAD_DIM)) // 128, 128], [wq_b_offset // 128, 0])
        wq_b = pl.reshape(wq_b_storage, [Q_LORA, LOCAL_H * HEAD_DIM])
        wq_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 1]), 0), pl.INDEX)
        wq_b_scale_storage = pl.slice(scales8, [(Q_LORA // 32 * (LOCAL_H * HEAD_DIM)) // 128, 128], [wq_b_scale_offset // 128, 0])
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wq_b_scale_storage, [Q_LORA // 32, LOCAL_H * HEAD_DIM])
        wkv_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 2]), 0), pl.INDEX)
        wkv_storage = pl.slice(bank8, [(D * HEAD_DIM) // 128, 128], [wkv_offset // 128, 0])
        wkv = pl.reshape(wkv_storage, [D, HEAD_DIM])
        wkv_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 2]), 0), pl.INDEX)
        wkv_scale_storage = pl.slice(scales8, [(D // 32 * HEAD_DIM) // 128, 128], [wkv_scale_offset // 128, 0])
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wkv_scale_storage, [D // 32, HEAD_DIM])
        wo_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 3]), 0), pl.INDEX)
        wo_b_storage = pl.slice(bank8, [(LOCAL_O_WIDTH * D) // 128, 128], [wo_b_offset // 128, 0])
        wo_b = pl.reshape(wo_b_storage, [LOCAL_O_WIDTH, D])
        wo_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 3]), 0), pl.INDEX)
        wo_b_scale_storage = pl.slice(scales8, [(LOCAL_O_WIDTH // 32 * D) // 128, 128], [wo_b_scale_offset // 128, 0])
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wo_b_scale_storage, [LOCAL_O_WIDTH // 32, D])
        index_wq_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 4]), 0), pl.INDEX)
        index_wq_b_storage = pl.slice(bank8, [(Q_LORA * (INDEX_H * INDEX_DIM)) // 128, 128], [index_wq_b_offset // 128, 0])
        index_wq_b = pl.reshape(index_wq_b_storage, [Q_LORA, INDEX_H * INDEX_DIM])
        index_wq_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 4]), 0), pl.INDEX)
        index_wq_b_scale_storage = pl.slice(scales8, [(Q_LORA // 32 * (INDEX_H * INDEX_DIM)) // 128, 128], [index_wq_b_scale_offset // 128, 0])
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(index_wq_b_scale_storage, [Q_LORA // 32, INDEX_H * INDEX_DIM])
        shared_w1_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 5]), 0), pl.INDEX)
        shared_w1_storage = pl.slice(bank8, [(D * MOE_INTER) // 128, 128], [shared_w1_offset // 128, 0])
        shared_w1 = pl.reshape(shared_w1_storage, [D, MOE_INTER])
        shared_w1_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 5]), 0), pl.INDEX)
        shared_w1_scale_storage = pl.slice(scales8, [(D // 32 * MOE_INTER) // 128, 128], [shared_w1_scale_offset // 128, 0])
        shared_w1_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w1_scale_storage, [D // 32, MOE_INTER])
        shared_w3_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 6]), 0), pl.INDEX)
        shared_w3_storage = pl.slice(bank8, [(D * MOE_INTER) // 128, 128], [shared_w3_offset // 128, 0])
        shared_w3 = pl.reshape(shared_w3_storage, [D, MOE_INTER])
        shared_w3_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 6]), 0), pl.INDEX)
        shared_w3_scale_storage = pl.slice(scales8, [(D // 32 * MOE_INTER) // 128, 128], [shared_w3_scale_offset // 128, 0])
        shared_w3_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w3_scale_storage, [D // 32, MOE_INTER])
        shared_w2_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 7]), 0), pl.INDEX)
        shared_w2_storage = pl.slice(bank8, [(MOE_INTER * D) // 128, 128], [shared_w2_offset // 128, 0])
        shared_w2 = pl.reshape(shared_w2_storage, [MOE_INTER, D])
        shared_w2_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 7]), 0), pl.INDEX)
        shared_w2_scale_storage = pl.slice(scales8, [(MOE_INTER // 32 * D) // 128, 128], [shared_w2_scale_offset // 128, 0])
        shared_w2_scale: pl.Tensor[[MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w2_scale_storage, [MOE_INTER // 32, D])
        attn_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 0]), 0), pl.INDEX)
        attn_norm_storage = pl.slice(bf16, [D], [attn_norm_offset])
        attn_norm = pl.reshape(attn_norm_storage, [D])
        ffn_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 1]), 0), pl.INDEX)
        ffn_norm_storage = pl.slice(bf16, [D], [ffn_norm_offset])
        ffn_norm = pl.reshape(ffn_norm_storage, [D])
        q_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 2]), 0), pl.INDEX)
        q_norm_storage = pl.slice(bf16, [Q_LORA], [q_norm_offset])
        q_norm = pl.reshape(q_norm_storage, [Q_LORA])
        kv_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 3]), 0), pl.INDEX)
        kv_norm_storage = pl.slice(bf16, [HEAD_DIM], [kv_norm_offset])
        kv_norm = pl.reshape(kv_norm_storage, [HEAD_DIM])
        wo_a_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 8]), 0), pl.INDEX)
        wo_a_storage = pl.slice(bank8, [(LOCAL_O_GROUPS * O_GROUP_IN * O_LORA) // 128, 128], [wo_a_offset // 128, 0])
        wo_a = pl.reshape(wo_a_storage, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA])
        wo_a_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 8]), 0), pl.INDEX)
        wo_a_scale_storage = pl.slice(scales8, [(LOCAL_O_GROUPS * (O_GROUP_IN // 32) * O_LORA) // 128, 128], [wo_a_scale_offset // 128, 0])
        wo_a_scale = pl.reshape(wo_a_scale_storage, [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA])
        publisher_wkv_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 4]), 0), pl.INDEX)
        publisher_wkv_storage = pl.slice(bf16, [D * HEAD_DIM], [publisher_wkv_offset])
        publisher_wkv = pl.reshape(publisher_wkv_storage, [D, HEAD_DIM])
        compressor_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 5]), 0), pl.INDEX)
        compressor_norm_storage = pl.slice(bf16, [HEAD_DIM], [compressor_norm_offset])
        compressor_norm = pl.reshape(compressor_norm_storage, [HEAD_DIM])
        index_wk_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 6]), 0), pl.INDEX)
        index_wk_storage = pl.slice(bf16, [HEAD_DIM * INDEX_DIM], [index_wk_offset])
        index_wk = pl.reshape(index_wk_storage, [HEAD_DIM, INDEX_DIM])
        index_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 7]), 0), pl.INDEX)
        index_norm_storage = pl.slice(bf16, [INDEX_DIM], [index_norm_offset])
        index_norm = pl.reshape(index_norm_storage, [INDEX_DIM])
        index_weights_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 8]), 0), pl.INDEX)
        index_weights_storage = pl.slice(bf16, [D * INDEX_H], [index_weights_offset])
        index_weights = pl.reshape(index_weights_storage, [D, INDEX_H])
        attn_fn_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 0]), 0), pl.INDEX)
        attn_fn_storage = pl.slice(dense, [MIX_HC * HC_DIM], [attn_fn_offset])
        attn_fn = pl.reshape(attn_fn_storage, [MIX_HC, HC_DIM])
        attn_scale_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 1]), 0), pl.INDEX)
        attn_scale_storage = pl.slice(dense, [3], [attn_scale_offset])
        attn_scale = pl.reshape(attn_scale_storage, [3])
        attn_base_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 2]), 0), pl.INDEX)
        attn_base_storage = pl.slice(dense, [MIX_HC], [attn_base_offset])
        attn_base = pl.reshape(attn_base_storage, [MIX_HC])
        ffn_fn_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 3]), 0), pl.INDEX)
        ffn_fn_storage = pl.slice(dense, [MIX_HC * HC_DIM], [ffn_fn_offset])
        ffn_fn = pl.reshape(ffn_fn_storage, [MIX_HC, HC_DIM])
        ffn_scale_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 4]), 0), pl.INDEX)
        ffn_scale_storage = pl.slice(dense, [3], [ffn_scale_offset])
        ffn_scale = pl.reshape(ffn_scale_storage, [3])
        ffn_base_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 5]), 0), pl.INDEX)
        ffn_base_storage = pl.slice(dense, [MIX_HC], [ffn_base_offset])
        ffn_base = pl.reshape(ffn_base_storage, [MIX_HC])
        attn_sink_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 6]), 0), pl.INDEX)
        attn_sink_storage = pl.slice(dense, [LOCAL_H], [attn_sink_offset])
        attn_sink = pl.reshape(attn_sink_storage, [LOCAL_H])
        gate_weight_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 7]), 0), pl.INDEX)
        gate_weight_storage = pl.slice(dense, [N_EXPERTS * D], [gate_weight_offset])
        gate_weight = pl.reshape(gate_weight_storage, [N_EXPERTS, D])
        gate_bias_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 8]), 0), pl.INDEX)
        gate_bias_storage = pl.slice(dense, [N_EXPERTS], [gate_bias_offset])
        gate_bias = pl.reshape(gate_bias_storage, [N_EXPERTS])
        compressor_wkv_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 9]), 0), pl.INDEX)
        compressor_wkv_storage = pl.slice(dense, [D * HEAD_DIM], [compressor_wkv_offset])
        compressor_wkv = pl.reshape(compressor_wkv_storage, [D, HEAD_DIM])
        compressor_wgate_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 10]), 0), pl.INDEX)
        compressor_wgate_storage = pl.slice(dense, [D * HEAD_DIM], [compressor_wgate_offset])
        compressor_wgate = pl.reshape(compressor_wgate_storage, [D, HEAD_DIM])
        routed_w1_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 0]), 0), pl.INDEX)
        routed_w1_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w1_offset // 128, 0])
        routed_w1 = pl.reshape(routed_w1_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w1_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 0]), 0), pl.INDEX)
        routed_w1_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (D // 32) * MOE_INTER) // 128, 128], [routed_w1_scale_offset // 128, 0])
        routed_w1_scale: pl.Tensor[[LOCAL_EXPERTS * (D // 32), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w1_scale_storage, [LOCAL_EXPERTS * (D // 32), MOE_INTER])
        routed_w3_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 1]), 0), pl.INDEX)
        routed_w3_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w3_offset // 128, 0])
        routed_w3 = pl.reshape(routed_w3_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w3_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 1]), 0), pl.INDEX)
        routed_w3_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (D // 32) * MOE_INTER) // 128, 128], [routed_w3_scale_offset // 128, 0])
        routed_w3_scale: pl.Tensor[[LOCAL_EXPERTS * (D // 32), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w3_scale_storage, [LOCAL_EXPERTS * (D // 32), MOE_INTER])
        routed_w2_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 2]), 0), pl.INDEX)
        routed_w2_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w2_offset // 128, 0])
        routed_w2 = pl.reshape(routed_w2_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w2_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 2]), 0), pl.INDEX)
        routed_w2_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (MOE_INTER // 32) * D) // 128, 128], [routed_w2_scale_offset // 128, 0])
        routed_w2_scale: pl.Tensor[[LOCAL_EXPERTS * (MOE_INTER // 32), D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w2_scale_storage, [LOCAL_EXPERTS * (MOE_INTER // 32), D])
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for ALL encoder rows before selecting replay.
                encoder_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
                encoder_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_pre(encoder_hidden, encoder_pre, encoder_collapsed)
                norm_tokens = pl.tensor.dim(encoder_collapsed, 0)
                for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
                    rms_norm_tiled(encoder_collapsed, attn_norm, encoder_normalized, norm_block, norm_tokens)
                publisher_ready = prefill_publish_decoder(
                    encoder_normalized, publisher_wkv, compressor_norm, index_wk, index_norm,
                    publisher_cos, publisher_sin, c1a_compressed_slots,
                    compressed_cache[3], compressed_scales[3], index_cache[3], index_scales[3],
                    pl.read(counts, [0]),
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(source_hidden, replay_rows, gathered_hidden, publisher_ready)
                gather_rows(encoder_pre, replay_rows, delayed_pre, publisher_ready)
        with pl.scope():
            attn_post = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
            attn_residual = pl.create_tensor([CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
            collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            attn_output = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            zero_bf16_padding(attn_output, pl.cast(0, pl.INT32))
            hc_mixes(hidden, attn_fn, attn_scale, attn_base, attn_pre, attn_post, attn_residual)
            hc_pre(hidden, delayed_pre, collapsed)
            norm_tokens = pl.tensor.dim(collapsed, 0)
            for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
                rms_norm_tiled(collapsed, attn_norm, normalized, norm_block, norm_tokens)
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
            prefill_attention(
                normalized, wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm,
                attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos[rope_slot], rope_sin[rope_slot],
                window_slots[stage], window_indices[stage], window_cache[layer], window_scales[layer],
                compressed_cache[cache_slot], compressed_scales[cache_slot], token_to_req_indices[stage],
                c2a_compressed_lens, c1a_compressed_lens, index_cache[cache_slot], index_scales[cache_slot],
                c2a_index_block_table, c1a_index_block_table, position_ids[stage], compressed_cos, compressed_sin,
                compressor_wkv, compressor_wgate, query_start_loc[stage], state_block_table, state_cache[state_slot],
                compressor_norm, c2a_compressed_slots, index_wk, index_norm, index_wq_b, index_wq_b_scale,
                index_weights, candidate_mask, topk_indices, attention_window, attention_arrived,
                attn_output, rank, 0, active, layer + 1, mode,
            )
            hc_post(attn_output, hidden, attn_post, attn_residual, attn_hidden)
        moe_ready = _prefill_moe(
            attn_hidden, attn_pre, ffn_fn, ffn_scale, ffn_base, ffn_norm, gate_weight, gate_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
            mxfp4_pair_lut, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale,
            shared_w3, shared_w3_scale, next_pre, ffn_mixed, next_hidden,
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
    collapsed_head = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    normalized_head = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    hc_head(final_hidden, final_pre, collapsed_head)
    norm_tokens = pl.tensor.dim(collapsed_head, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled_compensated(collapsed_head, final_norm, normalized_head, norm_block, norm_tokens)
    lm_head(normalized_head, head, logit_rows, logits, head_hidden, head_hidden_done,
            head_logits, head_logits_done, rank, 0, 1)
    greedy_sample(logits, sampled_ids)
    return logits, sampled_ids


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    bank8: pl.Tensor[[RANKS, B8, 128], pl.FP8E4M3FN],
    bank4: pl.Tensor[[RANKS, B4, 128], pl.UINT8],
    scales8: pl.Tensor[[RANKS, S8, 128], pl.FP8E8M0],
    bf16: pl.Tensor[[RANKS, B16], pl.BF16],
    dense: pl.Tensor[[RANKS, B32], pl.FP32],
    fp8_offsets: pl.Tensor[[RANKS, 40, 9], pl.INT64],
    fp8_scale_offsets: pl.Tensor[[RANKS, 40, 9], pl.INT64],
    bf16_offsets: pl.Tensor[[RANKS, 40, 9], pl.INT64],
    dense_offsets: pl.Tensor[[RANKS, 40, 11], pl.INT64],
    expert_offsets: pl.Tensor[[RANKS, 40, 3], pl.INT64],
    expert_scale_offsets: pl.Tensor[[RANKS, 40, 3], pl.INT64],
    mxfp4_pair_lut: pl.Tensor[[RANKS, 2, 256], pl.INT16],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    rope_cos: pl.Tensor[[RANKS, 3, CAPACITY, 32], pl.FP32],
    rope_sin: pl.Tensor[[RANKS, 3, CAPACITY, 32], pl.FP32],
    compressed_cos: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    compressed_sin: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    publisher_cos: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    publisher_sin: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    c2a_compressed_lens: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    c1a_compressed_lens: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
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
    final_norm: pl.Tensor[[RANKS, D], pl.BF16],
    head: pl.Tensor[[RANKS, VOCAB, D], pl.BF16],
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
    c2a_compressed_lens follows encoder rows; c1a_compressed_lens follows decoder
    replay rows. Each is floor((position+1)/ratio), including completed current
    groups. Token RoPE profiles are encoder base, encoder compressed, decoder compressed.
    compressed_cos/sin rotates completed C2A pairs at their group-first
    position; publisher_cos/sin rotates C1A keys at their token position.
    c2a_compressed_slots and c1a_compressed_slots both follow encoder rows;
    C1A publishes all encoder rows before decoder replay selection. Slots are
    source-local physical rows; -1 disables publication. C2A publishes only at
    completed pair positions. state_block_table maps requests to source-local
    state blocks shared across three independent C2A source slices; -1 is inactive.
    logit_rows indexes the decoder replay slab; -1 disables that output row.

    Preserve all seven InOut cache tensors across chunks and decode. Their
    source/page layout aliases decode pools through utils.decode_cache_views.
    Residual/pre-mix outputs describe these rows after layer 39. New tokens
    start from their own embedding and identity pre-mix; historical mHC state
    is represented by the committed KV/index/compressor caches.

    Zero-count ranks participate in EP communication without publishing cache
    rows. Wait for runtime completion before advancing lengths, launching a
    dependent call or reusing buffers. Communication windows are fresh per
    invocation (prepare with persistent=False). After request completion, the
    caller releases its pages and resets its request slot in all C2A sources before
    assigning them to another request; failed calls require cache restoration
    or reset before retry. Engram, vision and DSpark have no inputs here.
    """
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
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
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
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        attention_arrived = pld.window(attention_arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.FP32)
        head_logits_done = pld.window(head_logits_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_fwd(
            embedding[rank],
            embedding_ids[rank],
            initial_pre[rank],
            identity_rows[rank],
            bank8[rank],
            bank4[rank],
            scales8[rank],
            bf16[rank],
            dense[rank],
            fp8_offsets[rank],
            fp8_scale_offsets[rank],
            bf16_offsets[rank],
            dense_offsets[rank],
            expert_offsets[rank],
            expert_scale_offsets[rank],
            mxfp4_pair_lut[rank],
            counts[rank],
            replay_rows[rank],
            position_ids[rank],
            rope_cos[rank],
            rope_sin[rank],
            compressed_cos[rank],
            compressed_sin[rank],
            publisher_cos[rank],
            publisher_sin[rank],
            window_slots[rank],
            window_indices[rank],
            c2a_compressed_slots[rank],
            c1a_compressed_slots[rank],
            token_to_req_indices[rank],
            c2a_compressed_lens[rank], c1a_compressed_lens[rank],
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
            final_norm[rank],
            head[rank],
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
    "embedding_ids",
    "initial_pre",
    "identity_rows",
    "bank8",
    "bank4",
    "scales8",
    "bf16",
    "dense",
    "fp8_offsets",
    "fp8_scale_offsets",
    "bf16_offsets",
    "dense_offsets",
    "expert_offsets",
    "expert_scale_offsets",
    "mxfp4_pair_lut",
    "counts",
    "replay_rows",
    "position_ids",
    "rope_cos",
    "rope_sin",
    "compressed_cos",
    "compressed_sin",
    "publisher_cos",
    "publisher_sin",
    "window_slots",
    "window_indices",
    "c2a_compressed_slots",
    "c1a_compressed_slots",
    "token_to_req_indices",
    "c2a_compressed_lens", "c1a_compressed_lens",
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
    "final_norm",
    "head",
    "logit_rows",
    "x_hc_out", "pre_mix_out",
    "logits",
    "sampled_ids",
)


def generate(checkpoint, compiled, device_ids, prompts, max_new_tokens, *, chunk_size=None):
    """Generate one independent continuation per rank using resident checkpoint weights.

    Prefill and subsequent one-token calls retain native device caches. Only
    current-token metadata is uploaded; completed history is never recomputed.
    This example uses the prefill composition for one-token continuation too;
    l3_decode_fwd can consume the same cache storage through decode_cache_views.
    """
    from tokenizers import Tokenizer

    # Reject a different rank/capacity or the old Engram ABI before loading weights.
    parameters = compiled._get_metadata()[0]
    if (len(parameters) != len(PREFILL_TENSORS)
            or parameters[0].shape != [RANKS, CAPACITY, D]):
        raise ValueError("--runtime-dir must contain prefill compiled for the same EP and capacity with Engram disabled")
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
    config = RunConfig(
        platform="a5", ring_heap=2 * 1024**3,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    outputs = {
        "x_hc_out": ((CAPACITY, HC_MULT, D), torch.float32),
        "pre_mix_out": ((CAPACITY, HC_MULT), torch.float32),
        "logits": ((16, VOCAB), torch.float32),
        "sampled_ids": ((16, 8), torch.int32),
    }
    constants = {*weights.tensor_layouts, *weights.offset_tensors, "head", "final_norm", "mxfp4_pair_lut"}
    cache_names = {
        "window_cache", "window_scales", "compressed_cache", "compressed_scales",
        "index_cache", "index_scales", "state_cache",
    }
    constants.update(cache_names)
    generated = [[] for _ in prompts]
    finished = [False] * RANKS
    print("CED_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
    start = time.monotonic()
    # Fresh communication windows start epoch 1; native cache allocations persist.
    with compiled.prepare(config, persistent=False) as runtime:
        per_rank = [{} for _ in device_ids]
        host_cache = []
        for rank, tensors in enumerate(per_rank):
            inputs = prepare_resident_inputs(
                checkpoint, weights, sequences[rank][:chunk_size], CAPACITY, max_seq_len=max_seq_len,
            )
            host_cache.append({name: inputs[name] for name in cache_names})
            inputs["mxfp4_pair_lut"] = build_mxfp4_pair_lut()
            for name in PREFILL_TENSORS:
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
            for name in PREFILL_TENSORS
        }
        logits = torch.empty((RANKS, 16, VOCAB), dtype=torch.float32)
        sampled_ids = torch.empty((RANKS, 16, 8), dtype=torch.int32)
        committed = [0] * RANKS
        last_logits = torch.empty((RANKS, VOCAB), dtype=torch.float32)
        last_ids = [0] * RANKS

        def run_rows(rows):
            for rank, (tensors, token_ids) in enumerate(zip(per_rank, rows)):
                inputs = prepare_resident_inputs(
                    checkpoint, weights, token_ids, CAPACITY,
                    start_pos=committed[rank], max_seq_len=max_seq_len, cache_state=host_cache[rank],
                )
                for name, value in inputs.items():
                    if name not in constants:
                        runtime.copy_to(
                            tensors[name].data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                            worker_id=rank,
                        )
            runtime(*[stacked[name] for name in PREFILL_TENSORS], config=config)
            runtime.copy_stacked_from(stacked["logits"], logits)
            runtime.copy_stacked_from(stacked["sampled_ids"], sampled_ids)
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
                          for rank, tokens in enumerate(generated)])
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
        "platform": "a5", "tp": 1, "ep": RANKS, "dp": RANKS,
        "precision": "official_bf16_mxfp8_mxfp4", "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY, "chunk_size": chunk_size, "history_capacity": max_seq_len,
        "full_prefix_recomputation": False, "engram_enabled": False,
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
    from moe import _route_bias

    fp8_sizes = (
        D * Q_LORA, Q_LORA * LOCAL_H * HEAD_DIM, D * HEAD_DIM, LOCAL_O_WIDTH * D,
        Q_LORA * INDEX_H * INDEX_DIM, D * MOE_INTER, D * MOE_INTER, MOE_INTER * D,
        LOCAL_O_GROUPS * O_GROUP_IN * O_LORA,
    )
    bf16_sizes = (
        D, D, Q_LORA, HEAD_DIM,
        D * HEAD_DIM, HEAD_DIM, HEAD_DIM * INDEX_DIM, INDEX_DIM, D * INDEX_H,
    )
    dense_sizes = (
        MIX_HC * HC_DIM, 3, MIX_HC, MIX_HC * HC_DIM, 3, MIX_HC,
        LOCAL_H, N_EXPERTS * D, N_EXPERTS, D * HEAD_DIM, D * HEAD_DIM,
    )

    def offsets(sizes, start=0, alignment=128):
        rows = []
        for layer in range(M.num_hidden_layers):
            row = []
            for size in sizes:
                start = (start + alignment - 1) // alignment * alignment
                row.append(start)
                start += size
            rows.append(row)
        return torch.tensor(rows, dtype=torch.int64), start

    fp8_offsets, fp8_values = offsets(fp8_sizes)
    fp8_scale_offsets, scale_values = offsets(tuple(size // 32 for size in fp8_sizes))
    bf16_offsets, bf16_values = offsets(bf16_sizes, alignment=1)
    dense_offsets, dense_values = offsets(dense_sizes)
    expert_sizes = (LOCAL_EXPERTS * D * MOE_INTER // 2,) * 3
    expert_offsets, fp4_values = offsets(expert_sizes)
    expert_scale_sizes = (LOCAL_EXPERTS * D * MOE_INTER // 32,) * 3
    expert_scale_offsets, scale_values = offsets(expert_scale_sizes, start=scale_values)
    layouts = {
        "bank8": ([RANKS, fp8_values // 128, 128], torch.float8_e4m3fn, 0.0),
        "bank4": ([RANKS, fp4_values // 128, 128], torch.uint8, 0),
        "scales8": ([RANKS, scale_values // 128, 128], torch.float8_e8m0fnu, 1.0),
        "bf16": ([RANKS, bf16_values], torch.bfloat16, 0.0),
        "head": ([RANKS, VOCAB, D], torch.bfloat16, None),
        "x_hc_out": ([RANKS, CAPACITY, HC_MULT, D], torch.float32, None),
        "pre_mix_out": ([RANKS, CAPACITY, HC_MULT], torch.float32, None),
        "logits": ([RANKS, 16, VOCAB], torch.float32, None),
        "sampled_ids": ([RANKS, 16, 8], torch.int32, None),
    }
    values = prepare_prefill_inputs(torch.zeros((CAPACITY, D), dtype=torch.bfloat16), CAPACITY)
    values.update({
        "fp8_offsets": fp8_offsets, "fp8_scale_offsets": fp8_scale_offsets,
        "bf16_offsets": bf16_offsets, "dense_offsets": dense_offsets,
        "expert_offsets": expert_offsets, "expert_scale_offsets": expert_scale_offsets,
        "mxfp4_pair_lut": build_mxfp4_pair_lut(), "final_norm": torch.ones(D, dtype=torch.bfloat16),
    })
    dense = torch.zeros((RANKS, dense_values), dtype=torch.float32)
    for layer in range(M.num_hidden_layers):
        bias_offset = int(dense_offsets[layer, 8])
        dense[:, bias_offset : bias_offset + N_EXPERTS] = _route_bias()
    specs = []
    for name in PREFILL_TENSORS:
        if name in layouts:
            shape, dtype, value = layouts[name]
        elif name == "dense":
            shape, dtype, value = list(dense.shape), dense.dtype, dense
        else:
            tensor = values[name]
            value = tensor.unsqueeze(0).repeat(RANKS, *([1] * tensor.ndim))
            shape, dtype = list(value.shape), value.dtype
        specs.append(TensorSpec(name, shape, dtype, init_value=value))
    return specs


if __name__ == "__main__":
    if _shape_options.checkpoint:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--runtime-dir", type=Path, required=True, help="previously compiled L3 artifact")
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
            options.runtime_dir, platform="a5",
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )
        report = generate(
            checkpoint, compiled, device_ids, prompts, options.max_new_tokens, chunk_size=options.chunk_size,
        )
        if options.report:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("CED_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    else:
        from utils import upload_forward_inputs
        from golden import run
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
        from pypto.runtime import RunConfig

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
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
        if (args.tp, args.ep, args.dp, args.experts_per_rank, args.capacity) != (1, RANKS, RANKS, LOCAL_EXPERTS, CAPACITY):
            parser.error("parallel sizes, expert count and capacity must match the import-time configuration")
        devices = [int(device) for device in args.device.split(",") if device]
        if len(devices) != RANKS or len(set(devices)) != RANKS:
            parser.error(f"need exactly {RANKS} distinct devices, got {devices}")
        specs = build_tensor_specs()
        config = dict(
            platform=args.platform,
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
            cfg = RunConfig(**{name: value for name, value in config.items() if name != "log_level"})
            compiled = DistributedCompiledProgram.from_dir(result.work_dir, platform=cfg.platform)
            host_inputs = {}
            with compiled.prepare(cfg, persistent=False, inherited_host_tensors=list(host_inputs.values())) as runtime:
                arguments = upload_forward_inputs(runtime, specs, compiled._get_metadata()[0], len(devices), host_inputs)
                runtime(*arguments, config=cfg)
            print("[FWD] Execution passed; numerical accuracy was not validated.", flush=True)
        if not result.passed:
            raise SystemExit(result.error or 1)
