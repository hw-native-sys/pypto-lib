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
"""V4.1 DSpark recurrent decode stages for DP=EP."""


import pypto.language as pl
import pypto.language.distributed as pld

if __name__ == "__main__":
    import sys

    # Select the bounded smoke shapes before decode_fwd reads its import-time options.
    sys.argv[1:1] = ["--weight-bank-size", "1"]
    if not any(arg == "--experts-per-rank" or arg.startswith("--experts-per-rank=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--experts-per-rank", "16"]

from config import (
    BLOCK_SIZE,
    COMPRESSED_POOL_UNIT_ROWS,
    DECODE_SEQ,
    DSPARK_DRAFT_LAYERS,
    EP_SIZE,
    FLASH,
    DECODE_BATCH,
    LM_HEAD_TP_SIZE, SUPPORTED_LM_HEAD_TP_SIZES, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)
from quant import unpack_quantized_cache
from decode_fwd import FWD_C1A_TABLE_DYN, FWD_C2A_TABLE_DYN
import decode_fwd as target_fwd
import dspark_drafter as draft
import markov
import decode_metadata as state_ops
from decode_fwd import decode_fwd
from hc import hc_head
from rmsnorm import rms_norm
from lm_head import VOCAB, lm_head, greedy_sample
import lm_head as head
from dspark_drafter import dspark_drafter
from markov import markov_sample
from input_pack import VOCAB_DYN
from decode_metadata import WINDOW_TABLE_DYN, decode_prepare, decode_accept, decode_publish, commit_drafts


ROWS = DECODE_SEQ
TARGET_ROWS = DECODE_BATCH * ROWS
PADDED_ROWS = (TARGET_ROWS + 15) // 16 * 16
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
DRAFT_MOE_TOKENS = draft.MOE_TOKENS
WINDOW = FLASH.sliding_window



STATE_OPS_STATE_TOKEN_WIDTH = state_ops.STATE_TOKEN_WIDTH
STATE_OPS_STATE_META_WIDTH = state_ops.STATE_META_WIDTH
STATE_OPS_ROWS = state_ops.ROWS
TARGET_FWD_T_DYN = target_fwd.T_DYN
TARGET_FWD_ENGRAM_K = target_fwd.ENGRAM_K
TARGET_FWD_KV_OUT = target_fwd.KV_OUT
TARGET_FWD_DSPARK_MAIN_IN = target_fwd.DSPARK_MAIN_IN
TARGET_FWD_WEIGHT_BANK_SIZE = target_fwd.WEIGHT_BANK_SIZE
TARGET_FWD_MIX_HC = target_fwd.MIX_HC
TARGET_FWD_HC_DIM = target_fwd.HC_DIM
TARGET_FWD_Q_LORA = target_fwd.Q_LORA
TARGET_FWD_MX_GROUP = target_fwd.MX_GROUP
TARGET_FWD_LOCAL_H = target_fwd.LOCAL_H
TARGET_FWD_HEAD_DIM = target_fwd.HEAD_DIM
TARGET_FWD_LOCAL_O_GROUPS = target_fwd.LOCAL_O_GROUPS
TARGET_FWD_O_GROUP_IN = target_fwd.O_GROUP_IN
TARGET_FWD_O_LORA = target_fwd.O_LORA
TARGET_FWD_LOCAL_O_WIDTH = target_fwd.LOCAL_O_WIDTH
TARGET_FWD_ROPE_ROWS_DYN = target_fwd.ROPE_ROWS_DYN
TARGET_FWD_ROPE_DIM = target_fwd.ROPE_DIM
TARGET_FWD_FWD_ORI_BLOCKS_DYN = target_fwd.FWD_ORI_BLOCKS_DYN
TARGET_FWD_FWD_CMP_BLOCKS_DYN = target_fwd.FWD_CMP_BLOCKS_DYN
TARGET_FWD_FWD_INDEX_BLOCKS_DYN = target_fwd.FWD_INDEX_BLOCKS_DYN
TARGET_FWD_INDEX_DIM = target_fwd.INDEX_DIM
TARGET_FWD_INDEX_CACHE_GROUP = target_fwd.INDEX_CACHE_GROUP
TARGET_FWD_B_DYN = target_fwd.B_DYN
TARGET_FWD_FWD_C2A_TABLE_DYN = target_fwd.FWD_C2A_TABLE_DYN
TARGET_FWD_FWD_C1A_TABLE_DYN = target_fwd.FWD_C1A_TABLE_DYN
TARGET_FWD_C2A_SOURCE_COUNT = target_fwd.C2A_SOURCE_COUNT
TARGET_FWD_FWD_STATE_BLOCKS_DYN = target_fwd.FWD_STATE_BLOCKS_DYN
TARGET_FWD_STATE_CAPACITY = target_fwd.STATE_CAPACITY
TARGET_FWD_STATE_WIDTH = target_fwd.STATE_WIDTH
TARGET_FWD_KV_SOURCE_COUNT = target_fwd.KV_SOURCE_COUNT
TARGET_FWD_INDEX_SOURCE_COUNT = target_fwd.INDEX_SOURCE_COUNT
TARGET_FWD_INDEX_H = target_fwd.INDEX_H
TARGET_FWD_N_EXPERTS = target_fwd.N_EXPERTS
TARGET_FWD_N_LOCAL_EXPERTS = target_fwd.N_LOCAL_EXPERTS
TARGET_FWD_MX_W1_PACKED_ROWS = target_fwd.MX_W1_PACKED_ROWS
TARGET_FWD_MX_PACKED_LANE_COLS = target_fwd.MX_PACKED_LANE_COLS
TARGET_FWD_MOE_INTER = target_fwd.MOE_INTER
TARGET_FWD_MX_W2_PACKED_ROWS = target_fwd.MX_W2_PACKED_ROWS
TARGET_FWD_MX_W3_PACKED_ROWS = target_fwd.MX_W3_PACKED_ROWS
TARGET_FWD_VOCAB_PER_TP = head.VOCAB_PER_TP
TARGET_FWD_MAX_LOGIT_ROWS = PADDED_ROWS
TARGET_FWD_VOCAB = head.VOCAB
TARGET_FWD_SAMPLED_IDS_PAD = head.SAMPLED_IDS_PAD
STATE_OPS_TABLE_DYN = state_ops.TABLE_DYN
STATE_OPS_ROPE_ROWS_DYN = state_ops.ROPE_ROWS_DYN
STATE_OPS_ROPE_HALF = state_ops.ROPE_HALF
STATE_OPS_DSPARK_QUERY_WIDTH = state_ops.DSPARK_QUERY_WIDTH
DRAFT_EP_SIZE = draft.C.EP_SIZE
DRAFT_MAIN_IN = draft.MAIN_IN
DRAFT_B_DYN = draft.B_DYN
DRAFT_VOCAB = draft.VOCAB
DRAFT_MIX_HC = draft.MIX_HC
DRAFT_HC_DIM = draft.HC_DIM
DRAFT_Q_LORA = draft.Q_LORA
DRAFT_MX_GROUP = draft.MX_GROUP
DRAFT_LOCAL_H = draft.LOCAL_H
DRAFT_HEAD_DIM = draft.HEAD_DIM
DRAFT_LOCAL_O_GROUPS = draft.LOCAL_O_GROUPS
DRAFT_O_GROUP_IN = draft.O_GROUP_IN
DRAFT_O_LORA = draft.O_LORA
DRAFT_LOCAL_O_WIDTH = draft.LOCAL_O_WIDTH
DRAFT_ORI_BLOCKS_DYN = draft.C.ORI_BLOCKS_DYN
DRAFT_N_EXPERTS = draft.N_EXPERTS
DRAFT_N_LOCAL_EXPERTS = draft.N_LOCAL_EXPERTS
DRAFT_MX_W1_PACKED_ROWS = draft.MX_W1_PACKED_ROWS
DRAFT_MX_PACKED_LANE_COLS = draft.MX_PACKED_LANE_COLS
DRAFT_MOE_INTER = draft.MOE_INTER
DRAFT_MX_W2_PACKED_ROWS = draft.MX_W2_PACKED_ROWS
DRAFT_MX_W3_PACKED_ROWS = draft.MX_W3_PACKED_ROWS
DRAFT_DSPARK_QUERY_WIDTH = draft.DSPARK_QUERY_WIDTH
MARKOV_VOCAB_PER_TP = markov.VOCAB_PER_TP
MARKOV_VOCAB = markov.VOCAB
MARKOV_MARKOV_RANK = markov.MARKOV_RANK
MARKOV_B_DYN = markov.B_DYN
MARKOV_DSPARK_QUERY_WIDTH = markov.DSPARK_QUERY_WIDTH
TARGET_FWD_RECV_MAX = target_fwd.RECV_MAX
TARGET_FWD_DISPATCH_PACKET_WIDTH = target_fwd.DISPATCH_PACKET_WIDTH
TARGET_FWD_ROUTE_WIDTH = target_fwd.ROUTE_WIDTH
TARGET_FWD_SIGNAL_PAD = target_fwd.SIGNAL_PAD
TARGET_FWD_MOE_TOKENS = target_fwd.MOE_TOKENS
TARGET_FWD_TOPK = target_fwd.TOPK
TARGET_FWD_GROUP_LOGIT_ROWS = LM_HEAD_TP_SIZE * PADDED_ROWS
DRAFT_RECV_MAX = draft.RECV_MAX
DRAFT_DISPATCH_PACKET_WIDTH = draft.DISPATCH_PACKET_WIDTH
DRAFT_ROUTE_WIDTH = draft.ROUTE_WIDTH
DRAFT_SIGNAL_PAD = draft.SIGNAL_PAD
DRAFT_MOE_TOKENS = draft.MOE_TOKENS
DRAFT_TOPK = draft.TOPK
MARKOV_GROUP_LOGIT_ROWS = markov.GROUP_LOGIT_ROWS
MARKOV_MAX_LOGIT_ROWS = markov.LOGIT_ROWS


@pl.jit(auto_scope=False)
def l2_decode_fwd_dspark(
    slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_OPS_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[DECODE_BATCH, STATE_OPS_STATE_META_WIDTH], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    target_embedding_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    target_window_table: pl.Tensor[[DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    target_c2a_table: pl.Tensor[[DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    target_c1a_table: pl.Tensor[[DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[DECODE_BATCH + 1], pl.INT32]],
    engram_lookup: pl.Tensor[[2, PADDED_ROWS, TARGET_FWD_ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[2, TARGET_FWD_ENGRAM_K, TARGET_FWD_KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[2 * (TARGET_FWD_ENGRAM_K // 32), TARGET_FWD_KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[2, HC_MULT, D], pl.FP32],
    target_hidden_padded: pl.Out[pl.Tensor[[PADDED_ROWS, TARGET_FWD_DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_Q_LORA, TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN, TARGET_FWD_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN // 32, TARGET_FWD_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_LOCAL_O_WIDTH // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    window_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // TARGET_FWD_INDEX_CACHE_GROUP], pl.FP8E8M0]],
    c2a_index_block_table: pl.Tensor[[TARGET_FWD_B_DYN, TARGET_FWD_FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[TARGET_FWD_B_DYN, TARGET_FWD_FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    c2a_compressor_wgate: pl.Tensor[[TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    c1a_compressor_wkv: pl.Tensor[[D, TARGET_FWD_HEAD_DIM], pl.BF16, pl.NZ],
    state_block_table: pl.Tensor[[TARGET_FWD_B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[TARGET_FWD_FWD_STATE_BLOCKS_DYN, TARGET_FWD_STATE_CAPACITY, TARGET_FWD_STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[TARGET_FWD_KV_SOURCE_COUNT * TARGET_FWD_HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_HEAD_DIM, TARGET_FWD_INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT * TARGET_FWD_INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_Q_LORA, TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TARGET_FWD_INDEX_SOURCE_COUNT, D, TARGET_FWD_INDEX_H], pl.BF16, pl.NZ],
    hc_ffn_fn: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W1_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W2_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W3_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[D, TARGET_FWD_VOCAB_PER_TP], pl.BF16, pl.NZ],
    target_sampled_ids: pl.Out[pl.Tensor[[TARGET_ROWS, TARGET_FWD_SAMPLED_IDS_PAD], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    draft_table: pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH, STATE_OPS_TABLE_DYN], pl.INT32],
    draft_cos: pl.Tensor[[STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    draft_sin: pl.Tensor[[STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, DECODE_BATCH * STATE_OPS_ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_MAIN_IN, D], pl.FP8E4M3FN],
    draft_main_proj_scale: pl.Tensor[[DRAFT_MAIN_IN // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_main_norm_weight: pl.Tensor[[D], pl.BF16],
    zero_prefill_tokens: pl.Tensor[[DRAFT_B_DYN], pl.INT64],
    draft_embedding_weight: pl.Tensor[[DRAFT_VOCAB, D], pl.BF16],
    draft_hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    draft_wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    draft_q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    draft_attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    draft_wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    draft_wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    draft_wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_window_cache: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    draft_hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, D], pl.BF16],
    draft_correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    draft_routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), D], pl.FP8E8M0],
    draft_routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, D], pl.FP8E4M3FN],
    draft_shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_B_DYN, DRAFT_DSPARK_QUERY_WIDTH, D], pl.BF16]],
    moe_epoch: pl.Scalar[pl.INT32],
    markov_final_norm_weight: pl.Tensor[[D], pl.BF16],
    markov_lm_head_weight: pl.Tensor[[D, MARKOV_VOCAB_PER_TP], pl.BF16, pl.NZ],
    markov_markov_w1: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_markov_w2: pl.Tensor[[MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_confidence_weight: pl.Tensor[[1, D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    committed: pl.Out[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    target_fwd_recv_meta: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], pl.INT32],
    target_fwd_recv_x: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_DISPATCH_PACKET_WIDTH], pl.INT8],
    target_fwd_recv_routes: pld.DistributedTensor[[TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], pl.INT32],
    target_fwd_data_arrived: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], pl.INT32],
    target_fwd_routed_output: pld.DistributedTensor[[TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], pl.BF16],
    target_fwd_combine_arrived: pld.DistributedTensor[[EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], pl.INT32],
    draft_recv_meta: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS], pl.INT32],
    draft_recv_x: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], pl.INT8],
    draft_recv_routes: pld.DistributedTensor[[DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], pl.INT32],
    draft_data_arrived: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    draft_routed_output: pld.DistributedTensor[[DRAFT_MOE_TOKENS * DRAFT_TOPK, D], pl.BF16],
    draft_combine_arrived: pld.DistributedTensor[[EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], pl.INT32],
    markov_hidden_window: pld.DistributedTensor[[MARKOV_GROUP_LOGIT_ROWS, D], pl.BF16],
    markov_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    markov_logits_window: pld.DistributedTensor[[MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], pl.BF16],
    markov_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    sampling_hidden_window: pld.DistributedTensor[[TARGET_FWD_GROUP_LOGIT_ROWS, D], pl.BF16],
    sampling_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    sampling_logits_window: pld.DistributedTensor[[TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], pl.BF16],
    sampling_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    host_control: pl.Tensor,
    prepared_runs: pl.Tensor,
    use_host_plan: pl.constexpr,
):
    """Verify one rank's draft block and publish its next proposals."""
    local_x_hc = pl.create_tensor([PADDED_ROWS, HC_MULT, D], dtype=pl.FP32)
    local_pre_mix = pl.create_tensor([PADDED_ROWS, HC_MULT], dtype=pl.FP32)
    local_position_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT32)
    local_request_ids = pl.create_tensor([PADDED_ROWS], dtype=pl.INT32)
    local_window_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_window_indices = pl.create_tensor([PADDED_ROWS, WINDOW], dtype=pl.INT32)
    local_c2a_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_c1a_slots = pl.create_tensor([PADDED_ROWS], dtype=pl.INT64)
    local_query_starts = pl.create_tensor([DECODE_BATCH + 1], dtype=pl.INT32)
    local_target_hidden_padded = pl.create_tensor([PADDED_ROWS, TARGET_FWD_DSPARK_MAIN_IN], dtype=pl.BF16)
    target_embedding_weight.bind_dynamic(0, VOCAB_DYN)
    target_window_table.bind_dynamic(1, WINDOW_TABLE_DYN)
    target_c2a_table.bind_dynamic(1, FWD_C2A_TABLE_DYN)
    target_c1a_table.bind_dynamic(1, FWD_C1A_TABLE_DYN)
    freqs_cos.bind_dynamic(1, TARGET_FWD_ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(1, TARGET_FWD_ROPE_ROWS_DYN)
    window_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(0, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    c2a_index_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    c2a_index_block_table.bind_dynamic(1, TARGET_FWD_FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    c1a_index_block_table.bind_dynamic(1, TARGET_FWD_FWD_C1A_TABLE_DYN)
    state_block_table.bind_dynamic(0, TARGET_FWD_B_DYN)
    state_cache_pool.bind_dynamic(0, TARGET_FWD_FWD_STATE_BLOCKS_DYN)
    draft_table.bind_dynamic(2, STATE_OPS_TABLE_DYN)
    draft_cos.bind_dynamic(0, STATE_OPS_ROPE_ROWS_DYN)
    draft_sin.bind_dynamic(0, STATE_OPS_ROPE_ROWS_DYN)
    zero_prefill_tokens.bind_dynamic(0, DRAFT_B_DYN)
    draft_window_cache.bind_dynamic(1, DRAFT_ORI_BLOCKS_DYN)
    draft_head_hidden.bind_dynamic(0, DRAFT_B_DYN)
    draft_ids.bind_dynamic(0, MARKOV_B_DYN)
    confidences.bind_dynamic(0, MARKOV_B_DYN)
    sample_row_indices = pl.create_tensor([PADDED_ROWS], dtype=pl.INT32)
    decode_prepare(
        slot_ids, generations, state_tokens, state_meta,
        active_widths, target_embedding_weight,
        target_window_table, target_c2a_table, target_c1a_table, local_x_hc,
        local_pre_mix, local_position_ids, local_request_ids, local_window_slots,
        local_window_indices, local_c2a_slots, local_c1a_slots, local_query_starts, sample_row_indices,
    )
    decode_fwd(
        engram_lookup, engram_wkv, engram_wkv_scale, engram_weight,
        local_x_hc, local_pre_mix, local_target_hidden_padded, hc_attn_fn,
        hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a,
        q_norm_weight, wq_b, wkv, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b,
        freqs_cos, freqs_sin, local_window_slots, local_window_indices,
        window_cache_pool, compressed_cache_pool, local_request_ids, local_position_ids,
        index_cache_pool, index_cache_scale_pool, c2a_index_block_table, c1a_index_block_table,
        c2a_compressor_wkv, c2a_compressor_wgate, c1a_compressor_wkv, local_query_starts,
        state_block_table, state_cache_pool, compressor_norm_weight, local_c2a_slots,
        local_c1a_slots, index_wk, index_norm_weight, index_wq_b,
        index_weights_proj, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight, gate_weight, correction_bias, routed_w1,
        routed_w2, routed_w3, shared_w1, shared_w2,
        shared_w3, target_fwd_recv_meta, target_fwd_recv_x, target_fwd_recv_routes, target_fwd_data_arrived, target_fwd_routed_output,
        target_fwd_combine_arrived, moe_num_tokens, attention_num_tokens, rank,
        wq_a_scale, wq_b_scale, wkv_scale, wo_b_scale,
        index_wq_b_scale, routed_w1_scale, routed_w2_scale, routed_w3_scale,
        shared_w1_scale, shared_w2_scale, shared_w3_scale,
        host_control, prepared_runs, use_host_plan,
    )
    decode_publish(
        local_x_hc, local_pre_mix, local_target_hidden_padded,
        local_position_ids, local_request_ids, local_window_slots, local_window_indices,
        local_c2a_slots, local_c1a_slots, local_query_starts,
        x_hc, pre_mix, target_hidden_padded,
        position_ids, request_ids, window_slots, window_indices, c2a_slots, c1a_slots, query_starts,
    )
    sample_hidden = pl.reshape(local_x_hc, [PADDED_ROWS, HC_MULT, D])
    sample_pre_mix = pl.reshape(local_pre_mix, [PADDED_ROWS, HC_MULT])
    sample_tokens = pl.tensor.dim(sample_hidden, 0)
    sample_collapsed = pl.create_tensor([PADDED_ROWS, D], dtype=pl.BF16)
    hc_head(sample_hidden, sample_pre_mix, sample_collapsed, worker_limit=64)
    sample_normalized = pl.create_tensor([PADDED_ROWS, D], dtype=pl.BF16)
    rms_norm(
        sample_collapsed, final_norm_weight, sample_normalized,
        sample_tokens, width=D, zero_tail=False,
    )
    sample_logits = pl.create_tensor([PADDED_ROWS, VOCAB], dtype=pl.FP32)
    lm_head(
        sample_normalized, lm_head_weight, sample_row_indices, sample_logits,
        sampling_hidden_window, sampling_hidden_done, sampling_logits_window, sampling_logits_done,
        rank, PADDED_ROWS,
    )
    greedy_sample(sample_logits[:TARGET_ROWS, :], target_sampled_ids)
    markov_row_indices = pl.create_tensor([MARKOV_MAX_LOGIT_ROWS], dtype=pl.INT32)
    target_hidden = decode_accept(
        slot_ids, generations, active_widths, target_sampled_ids,
        state_tokens, state_meta, accepted_ids, accepted_counts,
        num_sampled, last_sampled, anchor_ids, first_query_positions,
        local_target_hidden_padded, draft_table, draft_cos, draft_sin,
        context_slots, context_rope_cos, context_rope_sin, query_rope_cos, query_rope_sin,
        markov_row_indices,
    )
    draft_x_hc = pl.create_tensor([DRAFT_MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
    draft_pre_mix = pl.create_tensor([DRAFT_MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    dspark_drafter(
        target_hidden, draft_main_proj_weight, draft_main_proj_scale, draft_main_norm_weight, num_sampled,
        last_sampled, zero_prefill_tokens, anchor_ids, first_query_positions,
        draft_table, context_slots, context_rope_cos, context_rope_sin,
        query_rope_cos, query_rope_sin, draft_embedding_weight, draft_hc_attn_fn,
        draft_hc_attn_scale, draft_hc_attn_base, draft_attn_norm_weight, draft_wq_a,
        draft_wq_a_scale, draft_q_norm_weight, draft_wq_b, draft_wq_b_scale,
        draft_wkv, draft_wkv_scale, draft_kv_norm_weight, draft_attn_sink,
        draft_wo_a, draft_wo_a_scale, draft_wo_b, draft_wo_b_scale,
        draft_window_cache, draft_hc_ffn_fn, draft_hc_ffn_scale, draft_hc_ffn_base,
        draft_ffn_norm_weight, draft_gate_weight, draft_correction_bias, draft_routed_w1,
        draft_routed_w1_scale, draft_routed_w2, draft_routed_w2_scale, draft_routed_w3,
        draft_routed_w3_scale, draft_shared_w1, draft_shared_w1_scale, draft_shared_w2,
        draft_shared_w2_scale, draft_shared_w3, draft_shared_w3_scale, draft_x_hc, draft_pre_mix,
        draft_recv_meta, draft_recv_x, draft_recv_routes, draft_data_arrived, draft_routed_output, draft_combine_arrived,
        rank, moe_epoch,
    )
    markov_batch = pl.tensor.dim(num_sampled, 0)
    markov_rows = markov_batch * MARKOV_DSPARK_QUERY_WIDTH
    markov_flat_hidden = pl.reshape(draft_head_hidden, [markov_rows, D])
    hc_head(draft_x_hc[:markov_rows, :, :], draft_pre_mix[:markov_rows, :], markov_flat_hidden, worker_limit=64)
    markov_normalized = pl.create_tensor([markov_rows, D], dtype=pl.BF16)
    rms_norm(
        markov_flat_hidden, markov_final_norm_weight, markov_normalized,
        markov_rows, width=D, zero_tail=False,
    )
    markov_logits = pl.create_tensor([MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.FP32)
    lm_head(
        markov_normalized, markov_lm_head_weight, markov_row_indices, markov_logits,
        markov_hidden_window, markov_hidden_done, markov_logits_window, markov_logits_done,
        rank, MARKOV_MAX_LOGIT_ROWS,
    )
    markov_base_logits = pl.reshape(markov_logits[:markov_rows, :], [markov_batch, MARKOV_DSPARK_QUERY_WIDTH, MARKOV_VOCAB])
    markov_sample(
        markov_base_logits, draft_head_hidden, anchor_ids, markov_markov_w1, markov_markov_w2,
        markov_confidence_weight, draft_ids, confidences,
    )
    commit_drafts(
        slot_ids, generations, draft_ids, state_tokens,
        state_meta, committed,
    )



@pl.jit.host
def l3_decode_fwd_dspark(
    slot_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_META_WIDTH], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    target_embedding_weight: pl.Tensor[[EP_SIZE, VOCAB_DYN, D], pl.BF16],
    target_window_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    target_c2a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    target_c1a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH + 1], pl.INT32]],
    engram_lookup: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_T_DYN, TARGET_FWD_ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ENGRAM_K, TARGET_FWD_KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[EP_SIZE, 2 * (TARGET_FWD_ENGRAM_K // 32), TARGET_FWD_KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[EP_SIZE, 2, HC_MULT, D], pl.FP32],
    target_hidden_padded: pl.Out[pl.Tensor[[EP_SIZE, TARGET_FWD_T_DYN, TARGET_FWD_DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_Q_LORA, TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN, TARGET_FWD_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN // 32, TARGET_FWD_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_LOCAL_O_WIDTH // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    window_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // TARGET_FWD_INDEX_CACHE_GROUP], pl.FP8E8M0]],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    state_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_STATE_BLOCKS_DYN, TARGET_FWD_STATE_CAPACITY, TARGET_FWD_STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_KV_SOURCE_COUNT * TARGET_FWD_HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_HEAD_DIM, TARGET_FWD_INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * TARGET_FWD_INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_Q_LORA, TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, D, TARGET_FWD_INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W1_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W2_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W3_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, D, TARGET_FWD_VOCAB_PER_TP], pl.BF16],
    target_sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, TARGET_ROWS, TARGET_FWD_SAMPLED_IDS_PAD], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    draft_table: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH, STATE_OPS_TABLE_DYN], pl.INT32],
    draft_cos: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    draft_sin: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH * STATE_OPS_ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_MAIN_IN, D], pl.FP8E4M3FN],
    draft_main_proj_scale: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_MAIN_IN // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_main_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, D], pl.BF16],
    zero_prefill_tokens: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN], pl.INT64],
    draft_embedding_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_VOCAB, D], pl.BF16],
    draft_hc_attn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_wq_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    draft_wq_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    draft_q_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wq_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_wkv: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    draft_attn_sink: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    draft_wo_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    draft_wo_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    draft_wo_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_window_cache: pl.InOut[pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    draft_hc_ffn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_gate_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, D], pl.BF16],
    draft_correction_bias: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    draft_routed_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_routed_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), D], pl.FP8E8M0],
    draft_routed_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, D], pl.FP8E4M3FN],
    draft_shared_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_shared_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN, DRAFT_DSPARK_QUERY_WIDTH, D], pl.BF16]],
    moe_epoch: pl.Scalar[pl.INT32],
    markov_final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    markov_lm_head_weight: pl.Tensor[[EP_SIZE, D, MARKOV_VOCAB_PER_TP], pl.BF16],
    markov_markov_w1: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_markov_w2: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_confidence_weight: pl.Tensor[[EP_SIZE, 1, D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    committed: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
):
    """Run one complete recurrent DSpark L2 per rank."""
    target_embedding_weight.bind_dynamic(1, VOCAB_DYN)
    target_window_table.bind_dynamic(2, WINDOW_TABLE_DYN)
    target_c2a_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    target_c1a_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    engram_lookup.bind_dynamic(2, TARGET_FWD_T_DYN)
    target_hidden_padded.bind_dynamic(1, TARGET_FWD_T_DYN)
    freqs_cos.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    window_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    c2a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c2a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c1a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C1A_TABLE_DYN)
    state_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    state_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_STATE_BLOCKS_DYN)
    draft_table.bind_dynamic(3, STATE_OPS_TABLE_DYN)
    draft_cos.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    draft_sin.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    zero_prefill_tokens.bind_dynamic(1, DRAFT_B_DYN)
    draft_window_cache.bind_dynamic(2, DRAFT_ORI_BLOCKS_DYN)
    draft_head_hidden.bind_dynamic(1, DRAFT_B_DYN)
    draft_ids.bind_dynamic(1, MARKOV_B_DYN)
    confidences.bind_dynamic(1, MARKOV_B_DYN)
    target_fwd_recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
    target_fwd_recv_x_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    target_fwd_recv_routes_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
    target_fwd_data_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    target_fwd_routed_output_buffer = pld.alloc_window_buffer([TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
    target_fwd_combine_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    sample_hidden_buffer = pld.alloc_window_buffer([TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    sample_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    sample_logits_buffer = pld.alloc_window_buffer([TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.BF16)
    sample_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    draft_recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
    draft_recv_x_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    draft_recv_routes_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
    draft_data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    draft_routed_output_buf = pld.alloc_window_buffer([DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
    draft_combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    markov_hidden_window_buf = pld.alloc_window_buffer([MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    markov_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    markov_logits_window_buf = pld.alloc_window_buffer([MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
    markov_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        target_fwd_recv_meta = pld.window(target_fwd_recv_meta_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
        target_fwd_recv_x = pld.window(target_fwd_recv_x_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        target_fwd_recv_routes = pld.window(target_fwd_recv_routes_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
        target_fwd_data_arrived = pld.window(target_fwd_data_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        target_fwd_routed_output = pld.window(target_fwd_routed_output_buffer, [TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
        target_fwd_combine_arrived = pld.window(target_fwd_combine_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        sampling_hidden_window = pld.window(sample_hidden_buffer, [TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        sampling_hidden_done = pld.window(sample_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        sampling_logits_window = pld.window(sample_logits_buffer, [TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.BF16)
        sampling_logits_done = pld.window(sample_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        draft_recv_meta = pld.window(draft_recv_meta_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
        draft_recv_x = pld.window(draft_recv_x_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        draft_recv_routes = pld.window(draft_recv_routes_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
        draft_data_arrived = pld.window(draft_data_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        draft_routed_output = pld.window(draft_routed_output_buf, [DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
        draft_combine_arrived = pld.window(draft_combine_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        markov_hidden_window = pld.window(markov_hidden_window_buf, [MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        markov_hidden_done = pld.window(markov_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        markov_logits_window = pld.window(markov_logits_window_buf, [MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
        markov_logits_done = pld.window(markov_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_fwd_dspark(
            slot_ids[rank], generations[rank], state_tokens[rank], state_meta[rank],
            active_widths[rank], target_embedding_weight[rank],
            target_window_table[rank], target_c2a_table[rank], target_c1a_table[rank], x_hc[rank],
            pre_mix[rank], position_ids[rank], request_ids[rank], window_slots[rank],
            window_indices[rank], c2a_slots[rank], c1a_slots[rank], query_starts[rank],
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            target_hidden_padded[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank],
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale[rank], freqs_cos[rank], freqs_sin[rank],
            window_cache_pool[rank], compressed_cache_pool[rank], index_cache_pool[rank], index_cache_scale_pool[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank], c2a_compressor_wkv[rank], c2a_compressor_wgate[rank],
            c1a_compressor_wkv[rank], state_block_table[rank], state_cache_pool[rank], compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale[rank],
            index_weights_proj[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank], routed_w3[rank],
            routed_w3_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w2[rank],
            shared_w2_scale[rank], shared_w3[rank], shared_w3_scale[rank], moe_num_tokens,
            attention_num_tokens, final_norm_weight[rank], lm_head_weight[rank],
            target_sampled_ids[rank], accepted_ids[rank],
            accepted_counts[rank], num_sampled[rank],
            last_sampled[rank], anchor_ids[rank], first_query_positions[rank],
            draft_table[rank], draft_cos[rank], draft_sin[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank], query_rope_cos[rank], query_rope_sin[rank],
            draft_main_proj_weight[rank], draft_main_proj_scale[rank], draft_main_norm_weight[rank], zero_prefill_tokens[rank], draft_embedding_weight[rank],
            draft_hc_attn_fn[rank], draft_hc_attn_scale[rank], draft_hc_attn_base[rank], draft_attn_norm_weight[rank],
            draft_wq_a[rank], draft_wq_a_scale[rank], draft_q_norm_weight[rank], draft_wq_b[rank],
            draft_wq_b_scale[rank], draft_wkv[rank], draft_wkv_scale[rank], draft_kv_norm_weight[rank],
            draft_attn_sink[rank], draft_wo_a[rank], draft_wo_a_scale[rank], draft_wo_b[rank],
            draft_wo_b_scale[rank], draft_window_cache[rank], draft_hc_ffn_fn[rank], draft_hc_ffn_scale[rank],
            draft_hc_ffn_base[rank], draft_ffn_norm_weight[rank], draft_gate_weight[rank], draft_correction_bias[rank],
            draft_routed_w1[rank], draft_routed_w1_scale[rank], draft_routed_w2[rank], draft_routed_w2_scale[rank],
            draft_routed_w3[rank], draft_routed_w3_scale[rank], draft_shared_w1[rank], draft_shared_w1_scale[rank],
            draft_shared_w2[rank], draft_shared_w2_scale[rank], draft_shared_w3[rank], draft_shared_w3_scale[rank],
            draft_head_hidden[rank], moe_epoch, markov_final_norm_weight[rank], markov_lm_head_weight[rank],
            markov_markov_w1[rank], markov_markov_w2[rank], markov_confidence_weight[rank], draft_ids[rank],
            confidences[rank], logits_epoch, committed[rank], target_fwd_recv_meta,
            target_fwd_recv_x, target_fwd_recv_routes,
            target_fwd_data_arrived, target_fwd_routed_output, target_fwd_combine_arrived, draft_recv_meta,
            draft_recv_x, draft_recv_routes,
            draft_data_arrived, draft_routed_output, draft_combine_arrived, markov_hidden_window,
            markov_hidden_done, markov_logits_window, markov_logits_done, sampling_hidden_window,
            sampling_hidden_done, sampling_logits_window, sampling_logits_done, rank,
            attention_num_tokens, c1a_index_block_table[rank], False,
            device=rank,
        )


@pl.jit.host
def l3_decode_fwd_dspark_prepared(
    slot_ids: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    generations: pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_STATE_META_WIDTH], pl.INT32]],
    active_widths: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    target_embedding_weight: pl.Tensor[[EP_SIZE, VOCAB_DYN, D], pl.BF16],
    target_window_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, WINDOW_TABLE_DYN], pl.INT32],
    target_c2a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C2A_TABLE_DYN], pl.INT32],
    target_c1a_table: pl.Tensor[[EP_SIZE, DECODE_BATCH, FWD_C1A_TABLE_DYN], pl.INT32],
    x_hc: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT, D], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, HC_MULT], pl.FP32]],
    position_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    request_ids: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT32]],
    window_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS, WINDOW], pl.INT32]],
    c2a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    c1a_slots: pl.Out[pl.Tensor[[EP_SIZE, PADDED_ROWS], pl.INT64]],
    query_starts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH + 1], pl.INT32]],
    engram_lookup: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_T_DYN, TARGET_FWD_ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ENGRAM_K, TARGET_FWD_KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[EP_SIZE, 2 * (TARGET_FWD_ENGRAM_K // 32), TARGET_FWD_KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[EP_SIZE, 2, HC_MULT, D], pl.FP32],
    target_hidden_padded: pl.Out[pl.Tensor[[EP_SIZE, TARGET_FWD_T_DYN, TARGET_FWD_DSPARK_MAIN_IN], pl.BF16]],
    hc_attn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_Q_LORA, TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_LOCAL_H * TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN, TARGET_FWD_O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_LOCAL_O_GROUPS, TARGET_FWD_O_GROUP_IN // 32, TARGET_FWD_O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_LOCAL_O_WIDTH // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[EP_SIZE, 2, TARGET_FWD_ROPE_ROWS_DYN, TARGET_FWD_ROPE_DIM // 2], pl.FP32],
    window_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_CMP_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_INDEX_BLOCKS_DYN, COMPRESSED_POOL_UNIT_ROWS, 1, TARGET_FWD_INDEX_DIM // TARGET_FWD_INDEX_CACHE_GROUP], pl.FP8E8M0]],
    c2a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, TARGET_FWD_FWD_C1A_TABLE_DYN], pl.INT32],
    c2a_compressor_wkv: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[EP_SIZE, TARGET_FWD_C2A_SOURCE_COUNT, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[EP_SIZE, D, TARGET_FWD_HEAD_DIM], pl.BF16],
    state_block_table: pl.Tensor[[EP_SIZE, TARGET_FWD_B_DYN, 1], pl.INT32],
    state_cache_pool: pl.InOut[pl.Tensor[[EP_SIZE, TARGET_FWD_FWD_STATE_BLOCKS_DYN, TARGET_FWD_STATE_CAPACITY, TARGET_FWD_STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_KV_SOURCE_COUNT * TARGET_FWD_HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_HEAD_DIM, TARGET_FWD_INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * TARGET_FWD_INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, TARGET_FWD_Q_LORA, TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT * (TARGET_FWD_Q_LORA // TARGET_FWD_MX_GROUP), TARGET_FWD_INDEX_H * TARGET_FWD_INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[EP_SIZE, TARGET_FWD_INDEX_SOURCE_COUNT, D, TARGET_FWD_INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC, TARGET_FWD_HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W1_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W2_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_MX_W3_PACKED_ROWS, TARGET_FWD_MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_N_LOCAL_EXPERTS * (D // TARGET_FWD_MX_GROUP)), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, TARGET_FWD_MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (TARGET_FWD_MOE_INTER // TARGET_FWD_MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE, D, TARGET_FWD_MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_SIZE, TARGET_FWD_WEIGHT_BANK_SIZE * (D // TARGET_FWD_MX_GROUP), TARGET_FWD_MOE_INTER], pl.FP8E8M0],
    moe_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    attention_num_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[EP_SIZE, D, TARGET_FWD_VOCAB_PER_TP], pl.BF16],
    target_sampled_ids: pl.Out[pl.Tensor[[EP_SIZE, TARGET_ROWS, TARGET_FWD_SAMPLED_IDS_PAD], pl.INT32]],
    accepted_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH, STATE_OPS_ROWS], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    num_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    last_sampled: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    anchor_ids: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT64]],
    first_query_positions: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    draft_table: pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH, STATE_OPS_TABLE_DYN], pl.INT32],
    draft_cos: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    draft_sin: pl.Tensor[[EP_SIZE, STATE_OPS_ROPE_ROWS_DYN, STATE_OPS_ROPE_HALF], pl.FP32],
    context_slots: pl.Out[pl.Tensor[[EP_SIZE, DSPARK_DRAFT_LAYERS, DECODE_BATCH * STATE_OPS_ROWS], pl.INT64]],
    context_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    context_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_ROWS, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_cos: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    query_rope_sin: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH * STATE_OPS_DSPARK_QUERY_WIDTH, STATE_OPS_ROPE_HALF], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_MAIN_IN, D], pl.FP8E4M3FN],
    draft_main_proj_scale: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_MAIN_IN // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_main_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, D], pl.BF16],
    zero_prefill_tokens: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN], pl.INT64],
    draft_embedding_weight: pl.Tensor[[DRAFT_EP_SIZE, DRAFT_VOCAB, D], pl.BF16],
    draft_hc_attn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_wq_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_Q_LORA], pl.FP8E4M3FN],
    draft_wq_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_Q_LORA], pl.FP8E8M0],
    draft_q_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wq_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA // DRAFT_MX_GROUP, DRAFT_LOCAL_H * DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_wkv: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_HEAD_DIM], pl.BF16],
    draft_attn_sink: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN, DRAFT_O_LORA], pl.FP8E4M3FN],
    draft_wo_a_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_GROUPS, DRAFT_O_GROUP_IN // DRAFT_MX_GROUP, DRAFT_O_LORA], pl.FP8E8M0],
    draft_wo_b: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    draft_wo_b_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_LOCAL_O_WIDTH // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_window_cache: pl.InOut[pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    draft_hc_ffn_fn: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    draft_gate_weight: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS, D], pl.BF16],
    draft_correction_bias: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_EXPERTS], pl.FP32],
    draft_routed_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W1_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_routed_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W2_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (DRAFT_MOE_INTER // DRAFT_MX_GROUP), D], pl.FP8E8M0],
    draft_routed_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS, DRAFT_MX_W3_PACKED_ROWS, DRAFT_MX_PACKED_LANE_COLS], pl.UINT8],
    draft_routed_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_N_LOCAL_EXPERTS * (D // DRAFT_MX_GROUP), DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w1: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w1_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_shared_w2: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, D], pl.FP8E4M3FN],
    draft_shared_w2_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER // DRAFT_MX_GROUP, D], pl.FP8E8M0],
    draft_shared_w3: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D, DRAFT_MOE_INTER], pl.FP8E4M3FN],
    draft_shared_w3_scale: pl.Tensor[[DRAFT_EP_SIZE, DSPARK_DRAFT_LAYERS, D // DRAFT_MX_GROUP, DRAFT_MOE_INTER], pl.FP8E8M0],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_EP_SIZE, DRAFT_B_DYN, DRAFT_DSPARK_QUERY_WIDTH, D], pl.BF16]],
    moe_epoch: pl.Scalar[pl.INT32],
    markov_final_norm_weight: pl.Tensor[[EP_SIZE, D], pl.BF16],
    markov_lm_head_weight: pl.Tensor[[EP_SIZE, D, MARKOV_VOCAB_PER_TP], pl.BF16],
    markov_markov_w1: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_markov_w2: pl.Tensor[[EP_SIZE, MARKOV_VOCAB, MARKOV_MARKOV_RANK], pl.BF16],
    markov_confidence_weight: pl.Tensor[[EP_SIZE, 1, D + MARKOV_MARKOV_RANK], pl.FP32],
    draft_ids: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.INT32]],
    confidences: pl.Out[pl.Tensor[[EP_SIZE, MARKOV_B_DYN, MARKOV_DSPARK_QUERY_WIDTH], pl.FP32]],
    logits_epoch: pl.Scalar[pl.INT32],
    committed: pl.Out[pl.Tensor[[EP_SIZE, DECODE_BATCH], pl.INT32]],
    host_control: pl.Tensor[[EP_SIZE, 2], pl.INT32],
    prepared_runs: pl.Tensor[[EP_SIZE, 4, PADDED_ROWS + 1], pl.INT32],
):
    """Run a recurrent DSpark step with fresh Host routing metadata."""
    target_embedding_weight.bind_dynamic(1, VOCAB_DYN)
    target_window_table.bind_dynamic(2, WINDOW_TABLE_DYN)
    target_c2a_table.bind_dynamic(2, FWD_C2A_TABLE_DYN)
    target_c1a_table.bind_dynamic(2, FWD_C1A_TABLE_DYN)
    engram_lookup.bind_dynamic(2, TARGET_FWD_T_DYN)
    target_hidden_padded.bind_dynamic(1, TARGET_FWD_T_DYN)
    freqs_cos.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, TARGET_FWD_ROPE_ROWS_DYN)
    window_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_ORI_BLOCKS_DYN)
    compressed_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_CMP_BLOCKS_DYN)
    index_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    index_cache_scale_pool.bind_dynamic(1, TARGET_FWD_FWD_INDEX_BLOCKS_DYN)
    c2a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c2a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    c1a_index_block_table.bind_dynamic(2, TARGET_FWD_FWD_C1A_TABLE_DYN)
    state_block_table.bind_dynamic(1, TARGET_FWD_B_DYN)
    state_cache_pool.bind_dynamic(1, TARGET_FWD_FWD_STATE_BLOCKS_DYN)
    draft_table.bind_dynamic(3, STATE_OPS_TABLE_DYN)
    draft_cos.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    draft_sin.bind_dynamic(1, STATE_OPS_ROPE_ROWS_DYN)
    zero_prefill_tokens.bind_dynamic(1, DRAFT_B_DYN)
    draft_window_cache.bind_dynamic(2, DRAFT_ORI_BLOCKS_DYN)
    draft_head_hidden.bind_dynamic(1, DRAFT_B_DYN)
    draft_ids.bind_dynamic(1, MARKOV_B_DYN)
    confidences.bind_dynamic(1, MARKOV_B_DYN)
    target_fwd_recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
    target_fwd_recv_x_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    target_fwd_recv_routes_buffer = pld.alloc_window_buffer([TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
    target_fwd_data_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    target_fwd_routed_output_buffer = pld.alloc_window_buffer([TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
    target_fwd_combine_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
    sample_hidden_buffer = pld.alloc_window_buffer([TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    sample_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    sample_logits_buffer = pld.alloc_window_buffer([TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.BF16)
    sample_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    draft_recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
    draft_recv_x_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
    draft_recv_routes_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
    draft_data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    draft_routed_output_buf = pld.alloc_window_buffer([DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
    draft_combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
    markov_hidden_window_buf = pld.alloc_window_buffer([MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    markov_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    markov_logits_window_buf = pld.alloc_window_buffer([MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
    markov_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        target_fwd_recv_meta = pld.window(target_fwd_recv_meta_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS], dtype=pl.INT32)
        target_fwd_recv_x = pld.window(target_fwd_recv_x_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        target_fwd_recv_routes = pld.window(target_fwd_recv_routes_buffer, [TARGET_FWD_N_LOCAL_EXPERTS * TARGET_FWD_RECV_MAX, TARGET_FWD_ROUTE_WIDTH], dtype=pl.INT32)
        target_fwd_data_arrived = pld.window(target_fwd_data_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        target_fwd_routed_output = pld.window(target_fwd_routed_output_buffer, [TARGET_FWD_MOE_TOKENS * TARGET_FWD_TOPK, D], dtype=pl.BF16)
        target_fwd_combine_arrived = pld.window(target_fwd_combine_arrived_buffer, [EP_SIZE, TARGET_FWD_N_LOCAL_EXPERTS, TARGET_FWD_SIGNAL_PAD], dtype=pl.INT32)
        sampling_hidden_window = pld.window(sample_hidden_buffer, [TARGET_FWD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        sampling_hidden_done = pld.window(sample_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        sampling_logits_window = pld.window(sample_logits_buffer, [TARGET_FWD_MAX_LOGIT_ROWS, TARGET_FWD_VOCAB], dtype=pl.BF16)
        sampling_logits_done = pld.window(sample_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        draft_recv_meta = pld.window(draft_recv_meta_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS], dtype=pl.INT32)
        draft_recv_x = pld.window(draft_recv_x_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_DISPATCH_PACKET_WIDTH], dtype=pl.INT8)
        draft_recv_routes = pld.window(draft_recv_routes_buf, [DRAFT_N_LOCAL_EXPERTS * DRAFT_RECV_MAX, DRAFT_ROUTE_WIDTH], dtype=pl.INT32)
        draft_data_arrived = pld.window(draft_data_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        draft_routed_output = pld.window(draft_routed_output_buf, [DRAFT_MOE_TOKENS * DRAFT_TOPK, D], dtype=pl.BF16)
        draft_combine_arrived = pld.window(draft_combine_arrived_buf, [EP_SIZE, DRAFT_N_LOCAL_EXPERTS, DRAFT_SIGNAL_PAD], dtype=pl.INT32)
        markov_hidden_window = pld.window(markov_hidden_window_buf, [MARKOV_GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        markov_hidden_done = pld.window(markov_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        markov_logits_window = pld.window(markov_logits_window_buf, [MARKOV_MAX_LOGIT_ROWS, MARKOV_VOCAB], dtype=pl.BF16)
        markov_logits_done = pld.window(markov_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_fwd_dspark(
            slot_ids[rank], generations[rank], state_tokens[rank], state_meta[rank],
            active_widths[rank], target_embedding_weight[rank],
            target_window_table[rank], target_c2a_table[rank], target_c1a_table[rank], x_hc[rank],
            pre_mix[rank], position_ids[rank], request_ids[rank], window_slots[rank],
            window_indices[rank], c2a_slots[rank], c1a_slots[rank], query_starts[rank],
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            target_hidden_padded[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank],
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale[rank], freqs_cos[rank], freqs_sin[rank],
            window_cache_pool[rank], compressed_cache_pool[rank], index_cache_pool[rank], index_cache_scale_pool[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank], c2a_compressor_wkv[rank], c2a_compressor_wgate[rank],
            c1a_compressor_wkv[rank], state_block_table[rank], state_cache_pool[rank], compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale[rank],
            index_weights_proj[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank], routed_w3[rank],
            routed_w3_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w2[rank],
            shared_w2_scale[rank], shared_w3[rank], shared_w3_scale[rank], moe_num_tokens,
            attention_num_tokens, final_norm_weight[rank], lm_head_weight[rank],
            target_sampled_ids[rank], accepted_ids[rank],
            accepted_counts[rank], num_sampled[rank],
            last_sampled[rank], anchor_ids[rank], first_query_positions[rank],
            draft_table[rank], draft_cos[rank], draft_sin[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank], query_rope_cos[rank], query_rope_sin[rank],
            draft_main_proj_weight[rank], draft_main_proj_scale[rank], draft_main_norm_weight[rank], zero_prefill_tokens[rank], draft_embedding_weight[rank],
            draft_hc_attn_fn[rank], draft_hc_attn_scale[rank], draft_hc_attn_base[rank], draft_attn_norm_weight[rank],
            draft_wq_a[rank], draft_wq_a_scale[rank], draft_q_norm_weight[rank], draft_wq_b[rank],
            draft_wq_b_scale[rank], draft_wkv[rank], draft_wkv_scale[rank], draft_kv_norm_weight[rank],
            draft_attn_sink[rank], draft_wo_a[rank], draft_wo_a_scale[rank], draft_wo_b[rank],
            draft_wo_b_scale[rank], draft_window_cache[rank], draft_hc_ffn_fn[rank], draft_hc_ffn_scale[rank],
            draft_hc_ffn_base[rank], draft_ffn_norm_weight[rank], draft_gate_weight[rank], draft_correction_bias[rank],
            draft_routed_w1[rank], draft_routed_w1_scale[rank], draft_routed_w2[rank], draft_routed_w2_scale[rank],
            draft_routed_w3[rank], draft_routed_w3_scale[rank], draft_shared_w1[rank], draft_shared_w1_scale[rank],
            draft_shared_w2[rank], draft_shared_w2_scale[rank], draft_shared_w3[rank], draft_shared_w3_scale[rank],
            draft_head_hidden[rank], moe_epoch, markov_final_norm_weight[rank], markov_lm_head_weight[rank],
            markov_markov_w1[rank], markov_markov_w2[rank], markov_confidence_weight[rank], draft_ids[rank],
            confidences[rank], logits_epoch, committed[rank], target_fwd_recv_meta,
            target_fwd_recv_x, target_fwd_recv_routes,
            target_fwd_data_arrived, target_fwd_routed_output, target_fwd_combine_arrived, draft_recv_meta,
            draft_recv_x, draft_recv_routes,
            draft_data_arrived, draft_routed_output, draft_combine_arrived, markov_hidden_window,
            markov_hidden_done, markov_logits_window, markov_logits_done, sampling_hidden_window,
            sampling_hidden_done, sampling_logits_window, sampling_logits_done, rank,
            host_control[rank], prepared_runs[rank], True,
            device=rank,
        )


def build_tensor_specs():
    """Build a bounded single-dispatch DSpark decode fixture."""
    import inspect
    import torch

    from golden import ScalarSpec, TensorSpec

    limit = 64
    extents = {
        VOCAB_DYN.name: 32,
        WINDOW_TABLE_DYN.name: 1,
        state_ops.TABLE_DYN.name: 1,
        DRAFT_ORI_BLOCKS_DYN.name: DECODE_BATCH,
        TARGET_FWD_ROPE_ROWS_DYN.name: limit,
        STATE_OPS_ROPE_ROWS_DYN.name: limit,
    }
    entry = l3_decode_fwd_dspark
    target_specs = {s.name: s for s in target_fwd.build_tensor_specs()}
    table = torch.arange(DECODE_BATCH, dtype=torch.int32).reshape(1, DECODE_BATCH, 1).repeat(EP_SIZE, 1, 1)
    target_cos, target_sin = target_fwd.prepare_forward_rope_tables(limit)
    initial = {
        "slot_ids": -1,
        "target_window_table": table,
        "target_c2a_table": target_specs["c2a_index_block_table"].init_value,
        "target_c1a_table": target_specs["c1a_index_block_table"].init_value,
        "draft_table": table[:, None].repeat(1, DSPARK_DRAFT_LAYERS, 1, 1),
        "freqs_cos": target_cos.unsqueeze(0).repeat(EP_SIZE, 1, 1, 1),
        "freqs_sin": target_sin.unsqueeze(0).repeat(EP_SIZE, 1, 1, 1),
        "draft_cos": 1.0,
        "draft_main_norm_weight": 1.0,
        "markov_final_norm_weight": 1.0,
        "final_norm_weight": 1.0,
        "attention_num_tokens": ROWS,
        "moe_num_tokens": ROWS,
    }
    aliases = {
        "markov_lm_head_weight": "lm_head_weight",
    }
    specs = {}
    scalars = {}
    for name, param in inspect.signature(entry._func).parameters.items():
        ann = param.annotation
        if not hasattr(ann, "shape"):
            scalars[name] = 1
            continue
        if name in aliases:
            source = specs[aliases[name]]
            specs[name] = TensorSpec(name, source.shape, source.dtype, init_value=source.init_value)
            continue
        shape = [
            extents[d.name] if hasattr(d, "name") and d.name in extents else target_fwd._fixture_extent(d)
            for d in ann.shape
        ]
        dtype = target_fwd._torch_dtype(ann.dtype)
        base = target_specs.get(name)
        value = initial.get(name, base.init_value if base is not None else None)
        if name == "draft_window_cache":
            value = torch.zeros(shape, dtype=torch.uint8)
            _, factors = unpack_quantized_cache(value, False)
            factors.fill_(1.0)
        if value is None and dtype == torch.float8_e8m0fnu:
            value = 1.0
        specs[name] = TensorSpec(name, shape, dtype, init_value=value)

    for name in ("slot_ids", "generations", "state_meta", "state_tokens"):
        specs[name].init_value = specs[name].create_tensor()
    specs["slot_ids"].init_value[:, 0] = 0
    specs["generations"].init_value[:, 0] = 1
    meta = specs["state_meta"].init_value[:, 0]
    meta[:, state_ops.STATE_VALID] = 1
    meta[:, state_ops.STATE_GENERATION] = 1
    meta[:, state_ops.STATE_ANCHOR] = 2
    meta[:, state_ops.STATE_COMMITTED] = 2
    meta[:, state_ops.STATE_DRAFT_COUNT] = ROWS - 1
    meta[:, state_ops.STATE_LIMIT] = limit
    specs["state_tokens"].init_value[:, 0, 1] = 1

    arguments = dict(specs)
    for name, value in scalars.items():
        arguments[name] = ScalarSpec(name, torch.int32, value, compile_runtime=True)
    return [arguments[name] for name in entry.param_names]



def prepare_dspark_reindex_inputs(specs):
    """Project the submitted DSpark state to the target forward's reindex rows."""
    import torch
    from golden import TensorSpec

    by_name = {spec.name: spec for spec in specs}
    slots = by_name["slot_ids"].init_value
    generations = by_name["generations"].init_value
    metadata = by_name["state_meta"].init_value
    tables = by_name["c1a_index_block_table"]
    ranks = torch.arange(EP_SIZE)[:, None]
    selected = metadata[ranks, slots.clamp(0, DECODE_BATCH - 1).long()]
    anchors = selected[..., state_ops.STATE_ANCHOR]
    limits = selected[..., state_ops.STATE_LIMIT]
    eligible = (
        (slots >= 0) & (slots < DECODE_BATCH)
        & (selected[..., state_ops.STATE_VALID] == 1)
        & (selected[..., state_ops.STATE_GENERATION] == generations)
        & (anchors < limits)
    )
    draft_counts = selected[..., state_ops.STATE_DRAFT_COUNT].clamp(0, STATE_OPS_DSPARK_QUERY_WIDTH)
    widths = torch.where(eligible, torch.where(anchors + draft_counts < limits, draft_counts + 1, 1), 0)
    offsets = torch.arange(ROWS, dtype=torch.int32)
    active = offsets < widths[..., None]
    positions = torch.full((EP_SIZE, PADDED_ROWS), -1, dtype=torch.int32)
    requests = torch.full_like(positions, -1)
    positions[:, :TARGET_ROWS] = torch.where(active, anchors[..., None] + offsets, -1).flatten(1)
    request_ids = torch.arange(DECODE_BATCH, dtype=torch.int32)[None, :, None]
    requests[:, :TARGET_ROWS] = torch.where(active, request_ids, -1).flatten(1)

    # The scorer reads c1a_index_block_table. target_c1a_table is for decode_prepare.
    return target_fwd.prepare_forward_reindex_inputs([
        TensorSpec("position_ids", list(positions.shape), positions.dtype, init_value=positions),
        TensorSpec("token_to_req_indices", list(requests.shape), requests.dtype, init_value=requests),
        tables,
        by_name["attention_num_tokens"],
        by_name["index_cache_pool"],
    ])


if __name__ == "__main__":
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--lm-head-tp", type=int, default=LM_HEAD_TP_SIZE, choices=SUPPORTED_LM_HEAD_TP_SIZES)
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--dp", type=int, default=EP_SIZE)
    parser.add_argument("--weight-bank-size", type=int, choices=[1], default=1)
    parser.add_argument("--experts-per-rank", type=int, default=target_fwd.N_LOCAL_EXPERTS)
    parser.add_argument("--draft-experts-per-rank", type=int, default=DRAFT_N_LOCAL_EXPERTS)
    parser.add_argument("-d", "--device", default=",".join(map(str, range(EP_SIZE))))
    parser.add_argument("--balanced-routing", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    args = parser.parse_args()
    if (args.ep, args.dp) != (EP_SIZE, EP_SIZE):
        parser.error("--ep/--dp must match the import-time DP=EP configuration")
    devices = tuple(int(value) for value in args.device.split(","))
    if args.execution_mode:
        parser.error("DSpark decode requires A5 hardware")
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE or min(devices) < 0:
        parser.error(f"need {EP_SIZE} distinct non-negative device IDs")
    from golden import run
    from pypto.ir import DistributedConfig

    specs = build_tensor_specs()
    if args.runtime_dir:
        metadata = json.loads((Path(args.runtime_dir) / "distributed_meta.json").read_text())
        prepared = any(item["name"].partition("__ssa_v")[0] == "host_control" for item in metadata["params"])
    else:
        prepared = os.environ.get("PYPTO_LIB_A5_RUNTIME", "host_build_graph") != "tensormap_and_ringbuffer"
    if prepared:
        specs.extend(prepare_dspark_reindex_inputs(specs))
    result = run(
        a5_default_runtime="host_build_graph",
        fn=l3_decode_fwd_dspark_prepared if prepared else l3_decode_fwd_dspark,
        specs=specs,
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type="a5", execution_mode=False,
            enable_chip_swimlane=args.enable_chip_swimlane,
            distributed_config=DistributedConfig(device_ids=list(devices), num_sub_workers=0),
            ring_heap=(2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2), ring_dep_pool=131072,
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
