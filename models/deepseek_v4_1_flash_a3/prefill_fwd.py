# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=16
# ci: no-sim
"""Compose the forty-layer W8A8 A3 prefill with TP4 heads and EP16 experts."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse


import pypto.language as pl
import pypto.language.distributed as pld

from .sparse_attention import CACHE_MAX_ROWS, SELECTED_KEYS
from . import config as C
from .driver import PREFILL_MOE_SLAB_ROWS, default_prefill_slab_rows, main

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--checkpoint")
_shape_parser.add_argument("--prefill-moe-slab-rows", type=int, choices=PREFILL_MOE_SLAB_ROWS)
_shape_parser.add_argument("--capacity", type=int, default=8192)
_shape_parser.add_argument("--experts-per-rank", type=int, default=C.N_LOCAL_EXPERTS)
_shape_parser.add_argument("--prepare-windows", action="store_true")
_shape_options = _shape_parser.parse_known_args()[0]
# Direct tensor-only callers specialize this flag away. Generation can reserve
# the program's communication windows before the first request arrives.
ExecutionFlag = pl.Scalar[pl.INT32] if _shape_options.prepare_windows else pl.constexpr
CAPACITY = _shape_options.capacity
if not 16 <= CAPACITY <= C.PREFILL_MAX_TOKENS or CAPACITY % 16:
    _shape_parser.error("--capacity must be a multiple of 16 between 16 and 8192")
if _shape_options.experts_per_rank != C.N_LOCAL_EXPERTS:
    _shape_parser.error("EP16 requires 24 experts per rank")
MOE_TOKENS = _shape_options.prefill_moe_slab_rows or default_prefill_slab_rows(CAPACITY)

from .config import (
    FLASH as M, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, WINDOW_CACHE_GROUP, TP_SIZE, PREFILL_MAX_TOKENS,
    STATE_CAPACITY,
)
from .dsl_types import (
    CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, B_DYN, INDEX_POOL_BLOCKS_DYN, Q_START_DYN,
    STATE_BLOCKS_DYN, ROUTE_T_DYN,
)

from .hc import hc_post
from .hc import hc_head, hc_pre, hc_pre_prefill, zero_bf16_padding
from .input_pack import HIDDEN_TILE, gather_rows, pack_x_hc, prepare_compressed_lens
from .rope import ROPE_ROWS_DYN, materialize_rope_rows
from .engram import ENGRAM_K, KV_OUT, prefill_engram_from_rows
from .lm_head import GROUP_LOGIT_ROWS, final_output
from .moe import moe_sublayer, prefill_ffn_restore
from .moe import AUX_WIDTH, SIGNAL_PAD, clear_moe_signals
from .moe import prepare_prefill_routes
from .compressor import STATE_WIDTH
from .prefill_c1a_full import prefill_publish_decoder
from .prefill_swa import prefill_swa_attention
from .o_proj import retire_prefill_signals, prefill_tp_output_all_reduce
from .prefill_c2a_full import prefill_c2a_full_attention
from .prefill_c2a_reuse import prefill_c2a_reuse_attention
from .prefill_c1a_reindex import prefill_c1a_reindex_attention
from .prefill_c1a_reuse import prefill_c1a_reuse_attention
from .rmsnorm import prefill_rms_norm

RANKS = C.EP_SIZE
D = M.hidden_size
HC_MULT = M.hc_mult
VOCAB = M.vocab_size
VOCAB_PER_TP = VOCAB // C.TP_SIZE
PAGES = pl.dynamic("V41_PREFILL_HISTORY_PAGES")
C2A_TABLE_DYN = pl.dynamic("V41_PREFILL_C2A_TABLE")
C1A_TABLE_DYN = pl.dynamic("V41_PREFILL_C1A_TABLE")
LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
N_LAYERS = M.num_hidden_layers
C2A_SOURCE_COUNT = sum(M.compress_ratios[source] == 2 for source in M.kv_source_layer_ids)
KV_SOURCE_COUNT = len(M.kv_source_layer_ids)
INDEX_SOURCE_COUNT = len(M.index_source_layer_ids)
MX_GROUP = M.weight_block_size[0]
HC_DIM = M.hc_dim
HEAD_DIM = M.head_dim
INDEX_DIM = M.index_head_dim
INDEX_H = M.index_n_heads
INDEX_TOPK = M.index_topk
INDEX_POOL_BLOCKS = M.candidate_topk_blocks
POOL_BLOCK_SIZE = M.candidate_block_size
MOE_ROUNDS = (CAPACITY + MOE_TOKENS * TP_SIZE - 1) // (MOE_TOKENS * TP_SIZE)
LOCAL_MOE_ROWS = MOE_ROUNDS * MOE_TOKENS
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
ROUTED_RECV_ROWS = RANKS * MOE_TOKENS * TOPK

ROPE_DIM = M.qk_rope_head_dim


@pl.jit.inline(auto_scope=False)
def _shared_prefill_prefix(starts: pl.Tensor[[Q_START_DYN], pl.INT32],
                           logical_rows: pl.Scalar[pl.INDEX], pool_rows: pl.Scalar[pl.INDEX]):
    # Single-slot callers retain the original whole-batch path. Packed request
    # chunks arrive in decreasing length, so the streaming rows form one prefix.
    slots = pl.tensor.dim(starts, 0) - 1
    prefix = pl.cast(-1, pl.INT32)
    if slots > 1:
        prefix = pl.cast(0, pl.INT32)
        if pool_rows <= CACHE_MAX_ROWS:
            for slot in pl.range(slots):
                start = pl.read(starts, [slot])
                end = pl.read(starts, [slot + 1])
                if logical_rows < (end - start) * SELECTED_KEYS:
                    prefix = end
    return prefix


@pl.jit.inline(auto_scope=False)
def prefill_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
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
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    c2a_index_block_table: pl.Tensor[[B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[B_DYN, C1A_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
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
    shared_prefix: pl.Scalar[pl.INT32],
):
    if mode == 0:
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
        prefill_swa_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, output_arrived, partial, num_tokens, attention_epoch,
        )
        prefill_tp_output_all_reduce(
            partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, owner_rows=False,
        )
    elif mode == 1:
        attention_tokens = pl.tensor.dim(x, 0)
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.BF16)
        prefill_c2a_full_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, c2a_compressed_lens, index_cache, index_cache_scale, c2a_index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight, c2a_compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_arrived, attention_partial, num_tokens, attention_epoch, shared_prefix=shared_prefix,
        )
        prefill_tp_output_all_reduce(
            attention_partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, owner_rows=False,
        )
    elif mode == 2:
        attention_tokens = pl.tensor.dim(x, 0)
        attention_partial = pl.create_tensor([attention_tokens, D], dtype=pl.BF16)
        prefill_c2a_reuse_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, topk_indices, output_arrived, attention_partial, num_tokens, attention_epoch, shared_prefix=shared_prefix,
        )
        prefill_tp_output_all_reduce(
            attention_partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, owner_rows=False,
        )
    elif mode == 3:
        prefill_c1a_reindex_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, c1a_compressed_lens, index_cache, index_cache_scale, c1a_index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, use_candidates=False, scatter_output=False, shared_prefix=shared_prefix,
        )
    elif mode == 4:
        prefill_c1a_reindex_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, c1a_compressed_lens, index_cache, index_cache_scale, c1a_index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, scatter_output=False, use_candidates=True, shared_prefix=shared_prefix,
        )
    elif mode == 5:
        prefill_c1a_reuse_attention(
            x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, topk_indices, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch, scatter_output=False, shared_prefix=shared_prefix,
        )
    return output


@pl.jit.inline(auto_scope=False)
def _pack_moe_round_inputs(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    round_hidden: pl.Tensor,
    round_pre: pl.Tensor,
    begin: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    active: pl.Scalar[pl.INDEX],
    reuse_ready: pl.Scalar[pl.TASK_ID],
):
    """Materialize a zero-padded input slab."""
    capacity = pl.tensor.dim(x_hc, 0)
    source_hidden = pl.reshape(x_hc, [capacity, HC_DIM])
    source_pre = pl.reshape(pre_mix, [1, capacity * HC_MULT])
    packed = pl.reshape(round_hidden, [rows, HC_DIM])
    packed_pre = pl.reshape(round_pre, [1, rows * HC_MULT])
    pack_workers = pl.max(1, pl.min(rows // 4 * (HC_DIM // 1024), pl.system.available_aiv_count()))
    with pl.spmd(pack_workers, name_hint="moe_round_pack", deps=[reuse_ready]) as packed_ready:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, rows // 4 * (HC_DIM // 1024), pack_workers):
            row = block // (HC_DIM // 1024) * 4
            column = block % (HC_DIM // 1024) * 1024
            live = pl.max(0, pl.min(4, active - row))
            if live > 0:
                source = pl.load(source_hidden, [begin + row, column], [4, 1024], valid_shape=[live, 1024])
                values = pl.fillpad(source, pad_value=pl.PadValue.zero)
                pl.store(pl.set_validshape(values, 4, 1024), [row, column], packed)
            else:
                empty = pl.tile.full([4, 1024], dtype=pl.FP32, value=0.0)
                pl.store(empty, [row, column], packed)
        for tile in pl.range(worker, rows // 16, pack_workers):
            offset = tile * 16
            live_pre = pl.max(0, pl.min(16, active - offset)) * HC_MULT
            if live_pre > 0:
                start = (begin + offset) * HC_MULT
                factors = pl.load(source_pre, [0, start], [1, 16 * HC_MULT], valid_shape=[1, live_pre])
                complete_pre = pl.fillpad(factors, pad_value=pl.PadValue.zero)
                pl.store(pl.set_validshape(complete_pre, 1, 16 * HC_MULT), [0, offset * HC_MULT], packed_pre)
            else:
                empty = pl.tile.full([1, 16 * HC_MULT], dtype=pl.FP32, value=0.0)
                pl.store(empty, [0, offset * HC_MULT], packed_pre)
    return packed_ready


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
    routed_w1: pl.Tensor[[LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[
        [LOCAL_EXPERTS * (1), MOE_INTER],
        pl.FP32,
        pl.ND,
    ],
    routed_w2: pl.Tensor[[LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[
        [LOCAL_EXPERTS * (1), D],
        pl.FP32,
        pl.ND,
    ],
    routed_w3: pl.Tensor[[LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[
        [LOCAL_EXPERTS * (1), MOE_INTER],
        pl.FP32,
        pl.ND,
    ],

    shared_w13: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[
        [1, MOE_INTER], pl.FP32, pl.ND
    ],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[
        [1, D], pl.FP32, pl.ND
    ],
    shared_w3_scale: pl.Tensor[
        [1, MOE_INTER], pl.FP32, pl.ND
    ],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_next: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[ROUTED_RECV_ROWS, D], pl.INT8],
    recv_weights: pld.DistributedTensor[
        [ROUTED_RECV_ROWS, AUX_WIDTH], pl.FP32
    ],
    arrived: pld.DistributedTensor[[RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[
        [RANKS, LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[
        [RANKS, LOCAL_EXPERTS, SIGNAL_PAD], pl.INT32
    ],
    slab_hidden: pl.Tensor[[MOE_TOKENS * TP_SIZE, HC_MULT, D], pl.FP32],
    slab_pre: pl.Tensor[[MOE_TOKENS * TP_SIZE, HC_MULT], pl.FP32],
    slab_mixed: pl.Tensor[[MOE_TOKENS * TP_SIZE, D], pl.BF16],
    ffn_input_rows: pl.Tensor[[T_DYN, D], pl.BF16],
    ffn_post_rows: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    ffn_residual_rows: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    statistics: pl.Tensor[[MOE_ROUNDS, 3], pl.INT32],
    gate_inputs: pl.Tensor[[LOCAL_MOE_ROWS, D], pl.BF16],
    owner_payload: pl.Tensor[[LOCAL_MOE_ROWS, D], pl.INT8],
    owner_scales: pl.Tensor[[LOCAL_MOE_ROWS, 1], pl.FP32],
    gate_indices: pl.Tensor[[LOCAL_MOE_ROWS, TOPK], pl.INT32],
    gate_weights: pl.Tensor[[LOCAL_MOE_ROWS, TOPK], pl.FP32],
    reuse_ready: pl.Scalar[pl.TASK_ID],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    ffn_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    ffn_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
):
    """Execute one interleaved TP slab per EP round."""
    capacity = pl.tensor.dim(x_hc, 0)
    output_flat = pl.reshape(x_next, [capacity, HC_DIM])
    # The previous layer's final MoE round fences reuse of the shared mHC rows.
    completion = pl.array.create(1, pl.TASK_ID)
    completion[0] = reuse_ready
    hc_pre_prefill(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_input_rows, next_pre_mix, ffn_post_rows, ffn_residual_rows, completion,
    )
    round_tokens = MOE_TOKENS * TP_SIZE
    rounds = (CAPACITY + round_tokens - 1) // round_tokens
    with pl.scope():
        prepare_prefill_routes(
            ffn_input_rows, norm_weight, gate_weight, correction_bias,
            gate_inputs, owner_payload, owner_scales, gate_indices, gate_weights,
            num_tokens, ep_rank % TP_SIZE, LOCAL_MOE_ROWS,
        )
    for round_id in pl.range(rounds):
        with pl.scope():
            begin = round_id * round_tokens
            physical_rows = pl.min(round_tokens, capacity - begin)
            active = pl.max(0, pl.min(round_tokens, num_tokens - begin))
            rows = physical_rows
            if active == 0:
                rows = pl.min(physical_rows, 16)
            round_ready = completion[0]
            if active == round_tokens and rows == round_tokens:
                hidden_view = pl.slice(x_hc, [round_tokens, HC_MULT, D], [begin, 0, 0])
                pre_view = pl.slice(pre_mix, [round_tokens, HC_MULT], [begin, 0])
                round_hidden = pl.slice(hidden_view, [rows, HC_MULT, D], [0, 0, 0])
                round_pre = pl.slice(pre_view, [rows, HC_MULT], [0, 0])
            else:
                round_hidden = pl.slice(slab_hidden, [rows, HC_MULT, D], [0, 0, 0])
                round_pre = pl.slice(slab_pre, [rows, HC_MULT], [0, 0])
                _pack_moe_round_inputs(x_hc, pre_mix, round_hidden, round_pre, begin, rows, active, round_ready)
            round_output = pl.slice(x_next, [rows, HC_MULT, D], [begin, 0, 0])
            round_next = pl.slice(next_pre_mix, [rows, HC_MULT], [begin, 0])
            round_mixed = pl.slice(slab_mixed, [rows, D], [0, 0])
            post = pl.slice(ffn_post_rows, [rows, HC_MULT], [begin, 0])
            residual = pl.slice(ffn_residual_rows, [rows, HC_MULT, HC_MULT], [begin, 0, 0])
            ffn_input = pl.slice(ffn_input_rows, [rows, D], [begin, 0])
            owned = pl.create_tensor([rows, D], dtype=pl.BF16)
            epoch = (moe_epoch - 1) * rounds + round_id + 1
            route_offset = round_id * MOE_TOKENS
            local_gate_inputs = pl.slice(gate_inputs, [MOE_TOKENS, D], [route_offset, 0])
            local_gate_indices = pl.slice(gate_indices, [MOE_TOKENS, TOPK], [route_offset, 0])
            local_gate_weights = pl.slice(gate_weights, [MOE_TOKENS, TOPK], [route_offset, 0])
            local_payload = pl.slice(owner_payload, [MOE_TOKENS, D], [route_offset, 0])
            scale_rows = owner_scales[route_offset : route_offset + MOE_TOKENS, :]
            local_scales = pl.reshape(scale_rows, [1, MOE_TOKENS])
            owned_result, round_counts = moe_sublayer(
                round_hidden, round_pre, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, norm_weight,
                gate_weight, correction_bias,
                routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
                 shared_w13, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3_scale,
                recv_meta, recv_x, recv_weights, arrived, data_arrived,
                routed_output, combine_arrived, ffn_input, owned, round_next, post, residual,
                local_gate_inputs, local_gate_indices, local_gate_weights,
                active, ep_rank, ep_rank % TP_SIZE, epoch, completion, True, True, MOE_TOKENS,
                local_payload, local_scales, pl.const(0, pl.INDEX),
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_counts"):
                for counter in pl.range(3):
                    value = pl.read(round_counts, [counter])
                    pl.write(statistics, [round_id, counter], value)
            prefill_ffn_restore(
                round_hidden, owned, post, residual, ffn_window, ffn_arrived,
                round_mixed, round_output, ep_rank // TP_SIZE * TP_SIZE, ep_rank % TP_SIZE, active, epoch,
                prefill=True,
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_round_complete") as completed:
                hidden_anchor = pl.read(round_output, [0, 0, 0])
                pl.write(round_output, [0, 0, 0], hidden_anchor)
                pre_anchor = pl.read(round_next, [0, 0])
                pl.write(round_next, [0, 0], pre_anchor)
            completion[0] = completed
            if rows < physical_rows:
                padding_workers = pl.max(1, pl.min((physical_rows - rows) // 4 * (HC_DIM // 1024), pl.system.available_aiv_count()))
                with pl.spmd(padding_workers, name_hint="moe_empty_round_padding", deps=[completed]) as padded:
                    worker = pl.tile.get_block_idx()
                    for block in pl.range(worker, (physical_rows - rows) // 4 * (HC_DIM // 1024), padding_workers):
                        row = rows + block // (HC_DIM // 1024) * 4
                        column = block % (HC_DIM // 1024) * 1024
                        zero_row = pl.load(output_flat, [begin, column], [1, 1024])
                        target = pl.tile.create([4, 1024], dtype=pl.FP32)
                        repeated = pl.col_expand(target, zero_row)
                        pl.store(repeated, [begin + row, column], output_flat)
                    for tile in pl.range(worker, (physical_rows - rows) // 16, padding_workers):
                        mix = pl.load(next_pre_mix, [begin, 0], [1, 8], valid_shape=[1, HC_MULT])
                        mix = pl.set_validshape(mix, 1, 8)
                        target_mix = pl.tile.create([16, 8], dtype=pl.FP32)
                        mixes = pl.col_expand(target_mix, mix)
                        pl.store(pl.set_validshape(mixes, 16, HC_MULT), [begin + rows + tile * 16, 0], next_pre_mix)
                completion[0] = padded
    return completion[0]


@pl.jit(auto_scope=False)
def l2_prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    engram_rotation: pl.Tensor[[D // 32, 32, 32], pl.FP32],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[N_LAYERS * (1), Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[N_LAYERS * (1), LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[N_LAYERS * (1), HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[N_LAYERS, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    c2a_compressor_wkv: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.ND],
    c2a_compressor_wgate: pl.Tensor[[C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16, pl.ND],
    c1a_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[INDEX_SOURCE_COUNT * (1), INDEX_H * INDEX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16, pl.ND],
    hc_ffn_fn: pl.Tensor[[N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (1)), MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (1)), D], pl.FP32],
    routed_w3: pl.Tensor[[N_LAYERS * LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LAYERS * (LOCAL_EXPERTS * (1)), MOE_INTER], pl.FP32],
    shared_w13: pl.Tensor[[N_LAYERS, D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[N_LAYERS * (1), MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[N_LAYERS * (1), D], pl.FP32],
    shared_w3_scale: pl.Tensor[[N_LAYERS * (1), MOE_INTER], pl.FP32],

    layer_range: pl.Tensor[[2], pl.INT32],
    topk_indices: pl.Tensor[[CAPACITY, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[CAPACITY], pl.INT32],
    moe_statistics: pl.Tensor[[N_LAYERS, MOE_ROUNDS, 3], pl.INT32],
    moe_gate_inputs: pl.Tensor[[LOCAL_MOE_ROWS, D], pl.BF16],
    moe_route_indices: pl.Tensor[[N_LAYERS, LOCAL_MOE_ROWS, TOPK], pl.INT32],
    moe_route_weights: pl.Tensor[[N_LAYERS, LOCAL_MOE_ROWS, TOPK], pl.FP32],
    counts: pl.Tensor[[2], pl.INT32],
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
    window_cache: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 512], pl.UINT8]],
    window_scales: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 16], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 32], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 4], pl.UINT8]],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_rows: pl.Tensor[[16], pl.INT32],
    x_hc_out: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    attn_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[ROUTED_RECV_ROWS, D], pl.INT8],
    recv_weights: pld.DistributedTensor[[ROUTED_RECV_ROWS, AUX_WIDTH], pl.FP32],
    arrived: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    data_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    routed_output: pld.DistributedTensor[[MOE_TOKENS * TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    attention_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.FP32],
    head_logits_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    ffn_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.BF16],
    ffn_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run both CED stages and retire communication signals after the decoder."""
    compressed_cache.bind_dynamic(1, PAGES)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
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
    encoder_count = pl.read(counts, [0])
    decoder_count = pl.read(counts, [1])
    logical_cache_rows = (pl.tensor.dim(c1a_index_block_table, 1) + 1) * 2 * 128
    pool_cache_rows = pl.tensor.dim(window_cache, 1) * 2 * 128
    encoder_shared = _shared_prefill_prefix(query_start_loc[0], logical_cache_rows, pool_cache_rows)
    decoder_shared = _shared_prefill_prefix(query_start_loc[1], logical_cache_rows, pool_cache_rows)
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
    hidden_pair = pl.create_tensor([3 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([3 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    initial_mix = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
    gather_ready = pl.system.task_dummy(deps=[])
    first_layer = pl.read(layer_range, [0])
    last_layer = pl.read(layer_range, [1])
    if first_layer == 0:
        pack_x_hc(embedding_ids, embedding, initial_hidden)
    else:
        replay_hidden = pl.reshape(x_hc_out, [CAPACITY, HC_DIM])
        replay_initial = pl.reshape(initial_hidden, [CAPACITY, HC_DIM])
        gather_rows(replay_hidden, identity_rows, replay_initial, gather_ready, D)
    gather_rows(initial_pre, identity_rows, initial_mix, gather_ready, HIDDEN_TILE)
    # Reuse mHC rows across layers and bounded MoE slabs across rounds.
    attn_hidden = attn_hc_out
    attn_pre = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
    slab_hidden = pl.create_tensor([MOE_TOKENS * TP_SIZE, HC_MULT, D], dtype=pl.FP32)
    slab_pre = pl.create_tensor([MOE_TOKENS * TP_SIZE, HC_MULT], dtype=pl.FP32)
    slab_mixed = pl.create_tensor([MOE_TOKENS * TP_SIZE, D], dtype=pl.BF16)
    owner_payload = pl.create_tensor([LOCAL_MOE_ROWS, D], dtype=pl.INT8)
    owner_scales = pl.create_tensor([LOCAL_MOE_ROWS, 1], dtype=pl.FP32)
    ffn_input_rows = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    ffn_post_rows = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_residual_rows = pl.create_tensor([CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
    moe_ready = pl.system.task_dummy(deps=[])
    for layer in pl.range(first_layer, last_layer):
        stage = layer // 20
        active = pl.read(counts, [stage])
        shared_prefix = encoder_shared
        if stage == 1:
            shared_prefix = decoder_shared
        current_slot = (2 * (stage - first_layer // 20) + (layer - first_layer) % 2) * CAPACITY
        next_slot = (2 * (stage - first_layer // 20) + (layer - first_layer + 1) % 2) * CAPACITY
        if current_slot == 3 * CAPACITY:
            current_slot = CAPACITY
        if next_slot == 3 * CAPACITY:
            next_slot = CAPACITY
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
        layer_wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ] = wq_a[layer]
        layer_wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND] = pl.slice(wq_a_scale, [1, Q_LORA], [layer * (1), 0])
        layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ] = wq_b[layer]
        layer_wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND] = pl.slice(wq_b_scale, [1, LOCAL_H * HEAD_DIM], [layer * (1), 0])
        layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ] = wkv[layer]
        layer_wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND] = pl.slice(wkv_scale, [1, HEAD_DIM], [layer * (1), 0])
        layer_wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ] = wo_b[layer]
        layer_shared_w13: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ] = shared_w13[layer]
        layer_shared_w1_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND] = pl.slice(shared_w1_scale, [1, MOE_INTER], [layer * (1), 0])
        layer_shared_w2: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ] = shared_w2[layer]
        layer_shared_w2_scale: pl.Tensor[[1, D], pl.FP32, pl.ND] = pl.slice(shared_w2_scale, [1, D], [layer * (1), 0])
        layer_shared_w3_scale: pl.Tensor[[1, MOE_INTER], pl.FP32, pl.ND] = pl.slice(shared_w3_scale, [1, MOE_INTER], [layer * (1), 0])
        layer_attn_sink: pl.Tensor[[LOCAL_H], pl.FP32] = pl.slice(attn_sink, [LOCAL_H], [layer * LOCAL_H])
        layer_wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [layer * LOCAL_O_GROUPS, 0, 0])
        layer_compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND] = c2a_compressor_wkv[c2a_source]
        layer_compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND] = c2a_compressor_wgate[c2a_source]
        layer_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(compressor_norm_weight, [HEAD_DIM], [kv_source * HEAD_DIM])
        layer_index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND] = index_wk[index_source]
        layer_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16] = pl.slice(index_norm_weight, [INDEX_DIM], [index_source * INDEX_DIM])
        layer_index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND] = index_wq_b[index_source]
        layer_index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND] = pl.slice(index_wq_b_scale, [1, INDEX_H * INDEX_DIM], [index_source * (1), 0])
        layer_index_weights: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND] = index_weights_proj[index_source]
        layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_weight, [N_EXPERTS, D], [layer * N_EXPERTS, 0])
        layer_gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(correction_bias, [N_EXPERTS], [layer * N_EXPERTS])
        layer_routed_w1: pl.Tensor[[LOCAL_EXPERTS, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [LOCAL_EXPERTS, MOE_INTER, D], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w1_scale: pl.Tensor[[LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND] = pl.slice(routed_w1_scale, [LOCAL_EXPERTS * (1), MOE_INTER], [layer * (LOCAL_EXPERTS * (1)), 0])
        layer_routed_w2: pl.Tensor[[LOCAL_EXPERTS, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [LOCAL_EXPERTS, D, MOE_INTER], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w2_scale: pl.Tensor[[LOCAL_EXPERTS * (1), D], pl.FP32, pl.ND] = pl.slice(routed_w2_scale, [LOCAL_EXPERTS * (1), D], [layer * (LOCAL_EXPERTS * (1)), 0])
        layer_routed_w3: pl.Tensor[[LOCAL_EXPERTS, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [LOCAL_EXPERTS, MOE_INTER, D], [layer * LOCAL_EXPERTS, 0, 0])
        layer_routed_w3_scale: pl.Tensor[[LOCAL_EXPERTS * (1), MOE_INTER], pl.FP32, pl.ND] = pl.slice(routed_w3_scale, [LOCAL_EXPERTS * (1), MOE_INTER], [layer * (LOCAL_EXPERTS * (1)), 0])
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for ALL encoder rows before selecting replay.
                encoder_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
                encoder_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_head(encoder_hidden, encoder_pre, encoder_collapsed)
                norm_tokens = pl.tensor.dim(encoder_collapsed, 0)
                prefill_rms_norm(encoder_collapsed, layer_attn_norm, encoder_normalized, norm_tokens)
                publisher_ready = prefill_publish_decoder(
                    encoder_normalized, c1a_compressor_wkv, layer_compressor_norm, layer_index_wk, layer_index_norm,
                    rope_cos[1], rope_sin[1], c1a_compressed_slots,
                    compressed_cache[3], compressed_scales[3], index_cache[3], index_scales[3],
                    pl.read(counts, [0]),
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(source_hidden, replay_rows, gathered_hidden, publisher_ready, D)
                gather_rows(encoder_pre, replay_rows, delayed_pre, publisher_ready, HIDDEN_TILE)
        # Empty source rounds consume zeros. Materialize the observable attention output on the final layer.
        if active > 0 or layer + 1 == last_layer:
            if layer == 1 or layer == 14:
                with pl.scope():
                    engram_slot = layer // 14
                    prefill_engram_from_rows(
                        engram_lookup[engram_slot], engram_wkv[engram_slot], engram_weight[engram_slot], engram_rotation, hidden, hidden,
                    )
            with pl.scope():
                attn_post = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
                attn_residual = pl.create_tensor([CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
                collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                attn_output = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                hc_pre(
                    hidden, delayed_pre, layer_attn_fn, layer_attn_scale, layer_attn_base,
                    collapsed, attn_pre, attn_post, attn_residual,
                    prefill=True,
                )
                norm_tokens = pl.tensor.dim(collapsed, 0)
                prefill_rms_norm(collapsed, layer_attn_norm, normalized, norm_tokens)
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
                        # A pool published in this replay contains every causal block when all fit.
                        history_blocks = pl.tensor.dim(c1a_index_block_table, 1) * 128 // POOL_BLOCK_SIZE
                        if first_layer <= 20 and history_blocks <= pl.tensor.dim(pool_block_ids, 1) and history_blocks <= INDEX_POOL_BLOCKS:
                            mode = 3
                elif layer >= 2:
                    cache_slot = (layer - 2) // 6
                    rope_slot = 1
                    mode = 2
                    if (layer - 2) % 6 == 0:
                        mode = 1
                state_slot = pl.min(cache_slot, 2)
                if active > 0:
                    prefill_attention(
                        normalized, layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale, layer_wkv, layer_wkv_scale, layer_kv_norm, layer_attn_sink, layer_wo_a, layer_wo_b, rope_cos[rope_slot], rope_sin[rope_slot], window_slots[stage], window_indices[stage], window_cache[layer], window_scales[layer], compressed_cache[cache_slot], compressed_scales[cache_slot], token_to_req_indices[stage], c2a_compressed_lens, c1a_compressed_lens, index_cache[cache_slot], index_scales[cache_slot], c2a_index_block_table, c1a_index_block_table, position_ids[stage], compressed_cos, compressed_sin, layer_compressor_wkv, layer_compressor_wgate, query_start_loc[stage], state_block_table, state_cache[state_slot], layer_compressor_norm, c2a_compressed_slots, layer_index_wk, layer_index_norm, layer_index_wq_b, layer_index_wq_b_scale, layer_index_weights, pool_block_ids, pool_block_counts, topk_indices, attention_window, attention_arrived, attn_output, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, layer - first_layer + 1, mode, shared_prefix,
                    )
                else:
                    zero_bf16_padding(attn_output, pl.cast(0, pl.INT32))
                hc_post_deps = pl.array.create(2, pl.TASK_ID)
                hc_post_deps[0] = moe_ready
                hc_post(
                    attn_output, hidden, attn_post, attn_residual, attn_hidden,
                    deps=hc_post_deps, prefill=True,
                )
        moe_ready = _prefill_moe(
            attn_hidden, attn_pre, layer_ffn_fn, layer_ffn_scale, layer_ffn_base, layer_ffn_norm, layer_gate_weight, layer_gate_bias,
            layer_routed_w1, layer_routed_w1_scale, layer_routed_w2, layer_routed_w2_scale, layer_routed_w3, layer_routed_w3_scale,
             layer_shared_w13, layer_shared_w1_scale, layer_shared_w2, layer_shared_w2_scale,
            layer_shared_w3_scale, next_pre, next_hidden,
            recv_meta, recv_x, recv_weights, arrived, data_arrived,
            routed_output, combine_arrived, slab_hidden, slab_pre, slab_mixed,
            ffn_input_rows, ffn_post_rows, ffn_residual_rows, moe_statistics[layer],
            moe_gate_inputs, owner_payload, owner_scales, moe_route_indices[layer], moe_route_weights[layer],
            moe_ready, active, rank, layer - first_layer + 1, ffn_window, ffn_arrived,
        )
    final_slot = (2 * ((last_layer - 1) // 20 - first_layer // 20) + (last_layer - first_layer) % 2) * CAPACITY
    if final_slot == 3 * CAPACITY:
        final_slot = CAPACITY
    final_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [final_slot, 0, 0])
    final_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [final_slot, 0])
    source_hidden = pl.reshape(final_hidden, [CAPACITY, HC_DIM])
    output_hidden = pl.reshape(x_hc_out, [CAPACITY, HC_DIM])
    gather_rows(source_hidden, identity_rows, output_hidden, gather_ready, D)
    gather_rows(final_pre, identity_rows, pre_mix_out, gather_ready, HIDDEN_TILE)
    retire_prefill_signals(x_hc_out, ffn_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    clear_moe_signals(x_hc_out, arrived, data_arrived, combine_arrived, anchor_indices=(0, 0, 0))
    retire_prefill_signals(x_hc_out, attention_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    final_logits, final_sampled_ids = final_output(
        final_hidden, final_pre, final_norm_weight, lm_head_weight,
        logit_rows, logits, sampled_ids,
        head_hidden, head_hidden_done, head_logits, head_logits_done,
        rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, 1, 1,
    )
    return final_logits, final_sampled_ids


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[RANKS, 2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.BF16, pl.NZ],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    engram_rotation: pl.Tensor[[RANKS, D // 32, 32, 32], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, N_LAYERS, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[RANKS, N_LAYERS * (1), Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[RANKS, N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[RANKS, N_LAYERS * (1), LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[RANKS, N_LAYERS, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[RANKS, N_LAYERS * (1), HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[RANKS, N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[RANKS, N_LAYERS, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (1), INDEX_H * INDEX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (1)), MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (1)), D], pl.FP32],
    routed_w3: pl.Tensor[[RANKS, N_LAYERS * LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_EXPERTS * (1)), MOE_INTER], pl.FP32],
    shared_w13: pl.Tensor[[RANKS, N_LAYERS, D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (1), MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[RANKS, N_LAYERS, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (1), D], pl.FP32],
    shared_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (1), MOE_INTER], pl.FP32],

    layer_range: pl.Tensor[[RANKS, 2], pl.INT32],
    topk_indices: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[RANKS, CAPACITY], pl.INT32]],
    moe_statistics: pl.InOut[pl.Tensor[[RANKS, N_LAYERS, MOE_ROUNDS, 3], pl.INT32]],
    moe_gate_inputs: pl.InOut[pl.Tensor[[RANKS, LOCAL_MOE_ROWS, D], pl.BF16]],
    moe_route_indices: pl.InOut[pl.Tensor[[RANKS, N_LAYERS, LOCAL_MOE_ROWS, TOPK], pl.INT32]],
    moe_route_weights: pl.InOut[pl.Tensor[[RANKS, N_LAYERS, LOCAL_MOE_ROWS, TOPK], pl.FP32]],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
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
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 512], pl.UINT8]],
    window_scales: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 16], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 32], pl.BF16]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 4], pl.UINT8]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    x_hc_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    attn_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
    execute_layers: ExecutionFlag = 1,
):
    """Append one DP4/TP4/EP16 chunk and return its final state/logits.

    CAPACITY bounds current rows to at most 8192; the dynamic page extent
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
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([ROUTED_RECV_ROWS, D], dtype=pl.INT8)
    recv_weights_buffer = pld.alloc_window_buffer([ROUTED_RECV_ROWS, AUX_WIDTH], dtype=pl.FP32)
    arrived_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    ffn_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    ffn_arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.FP32)
    head_logits_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [ROUTED_RECV_ROWS, D], dtype=pl.INT8)
        recv_weights = pld.window(recv_weights_buffer, [ROUTED_RECV_ROWS, AUX_WIDTH], dtype=pl.FP32)
        arrived = pld.window(arrived_buffer, [RANKS, 128], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        ffn_window = pld.window(ffn_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        ffn_arrived = pld.window(ffn_arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.FP32)
        head_logits_done = pld.window(head_logits_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        if execute_layers != 0:
            l2_prefill_fwd(
                embedding[rank], engram_lookup[rank], engram_wkv[rank], engram_weight[rank], engram_rotation[rank],
                embedding_ids[rank], initial_pre[rank], identity_rows[rank],
                hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
                wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank],
                wq_b_scale[rank], wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
                attn_sink[rank], wo_a[rank], wo_b[rank],
                c2a_compressor_wkv[rank], c2a_compressor_wgate[rank],
                c1a_compressor_wkv[rank], compressor_norm_weight[rank],
                index_wk[rank], index_norm_weight[rank], index_wq_b[rank],
                index_wq_b_scale[rank], index_weights_proj[rank],
                hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
                ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
                routed_w1[rank], routed_w1_scale[rank], routed_w2[rank],
                routed_w2_scale[rank], routed_w3[rank], routed_w3_scale[rank],
                shared_w13[rank], shared_w1_scale[rank], shared_w2[rank],
                shared_w2_scale[rank], shared_w3_scale[rank],
                layer_range[rank], topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank],
                moe_statistics[rank], moe_gate_inputs[rank], moe_route_indices[rank], moe_route_weights[rank],
                counts[rank], replay_rows[rank], position_ids[rank],
                freqs_cos[rank], freqs_sin[rank],
                window_slots[rank], window_indices[rank], c2a_compressed_slots[rank],
                c1a_compressed_slots[rank], token_to_req_indices[rank],
                query_start_loc[rank], c2a_index_block_table[rank],
                c1a_index_block_table[rank], state_block_table[rank],
                window_cache[rank], window_scales[rank], compressed_cache[rank], compressed_scales[rank],
                index_cache[rank], index_scales[rank], state_cache[rank],
                final_norm_weight[rank], lm_head_weight[rank], logit_rows[rank],
                x_hc_out[rank], attn_hc_out[rank], pre_mix_out[rank],
                logits[rank], sampled_ids[rank],
                recv_meta, recv_x, recv_weights, arrived, data_arrived, routed_output, combine_arrived,
                attention_window, attention_arrived,
                head_hidden, head_hidden_done, head_logits, head_logits_done,
                ffn_window, ffn_arrived,
                rank, device=rank,
            )


if __name__ == "__main__":
    main(l3_prefill_fwd, mode="prefill", capacity=CAPACITY, slab_rows=MOE_TOKENS)
