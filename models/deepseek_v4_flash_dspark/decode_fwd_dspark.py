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
"""One-L2 DSpark target decode, drafter, Markov sampler, and state commit."""

import pypto.language as pl
import pypto.language.distributed as pld

import decode_fwd as decode

if decode.TP_SIZE not in (2, 4):
    raise ValueError(f"the fused DSpark one-L2 decode path supports --tp=2 or 4, got --tp={decode.TP_SIZE}")

import decode_prepare as prepare
import dspark_drafter as dspark
import dspark_markov as markov
import lm_head as lmhead
from decode_prepare import (
    CSA_GROUP_STATE_BLOCKS_DYN,
    HCA_GROUP_STATE_BLOCKS_DYN,
    accept_target_into_device_state,
    build_group_decode_metadata,
    commit_drafts_to_device_state,
    fence_drafter_head_hidden,
    gather_group_decode_rope_rows,
    prepare_drafter_after_target,
    prepare_target_group_from_device_state,
)
from decode_fwd import decode_fwd
from dspark_drafter import dspark_drafter
from dspark_markov import distributed_markov_sample
ATTENTION_WINDOW_ROWS = decode.ATTENTION_WINDOW_ROWS
AUX_PAD = decode.AUX_PAD
BLOCK_SIZE = decode.BLOCK_SIZE
CSA_B_DYN = decode.CSA_B_DYN
CSA_CMP_MAX_BLOCKS = decode.CSA_CMP_MAX_BLOCKS
CSA_COMPRESS_RATIO = decode.CSA_COMPRESS_RATIO
CSA_IDX_HEAD_DIM = decode.CSA_IDX_HEAD_DIM
CSA_IDX_MAX_BLOCKS = decode.CSA_IDX_MAX_BLOCKS
CSA_IDX_N_HEADS = decode.CSA_IDX_N_HEADS
CSA_INNER_OUT_DIM = decode.CSA_INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = decode.CSA_INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_DIM = decode.CSA_INNER_STATE_DIM
CSA_INNER_STATE_MAX_BLOCKS = decode.CSA_INNER_STATE_MAX_BLOCKS
CSA_MAIN_OUT_DIM = decode.CSA_MAIN_OUT_DIM
CSA_MAIN_STATE_BLOCK_SIZE = decode.CSA_MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_DIM = decode.CSA_MAIN_STATE_DIM
CSA_MAIN_STATE_MAX_BLOCKS = decode.CSA_MAIN_STATE_MAX_BLOCKS
D = decode.D
DECODE_GROUP_CAP = decode.DECODE_GROUP_CAP
DSPARK_STATE_CAPACITY = prepare.STATE_CAPACITY
DSPARK_STATE_LOCAL_BATCH = prepare.LOCAL_BATCH
DSPARK_STATE_META_WIDTH = prepare.STATE_META_WIDTH
DSPARK_STATE_TOKEN_WIDTH = prepare.STATE_TOKEN_WIDTH
EMBED_VOCAB_DYN = decode.EMBED_VOCAB_DYN
EP_SIZE = decode.EP_SIZE
FWD_CSA_CMP_BLOCKS_DYN = decode.FWD_CSA_CMP_BLOCKS_DYN
FWD_CSA_IDX_BLOCKS_DYN = decode.FWD_CSA_IDX_BLOCKS_DYN
FWD_CSA_INNER_STATE_BLOCKS_DYN = decode.FWD_CSA_INNER_STATE_BLOCKS_DYN
FWD_CSA_MAIN_STATE_BLOCKS_DYN = decode.FWD_CSA_MAIN_STATE_BLOCKS_DYN
FWD_CSA_WEIGHT_BANK_SIZE = decode.FWD_CSA_WEIGHT_BANK_SIZE
FWD_HCA_CMP_BLOCKS_DYN = decode.FWD_HCA_CMP_BLOCKS_DYN
FWD_HCA_STATE_BLOCKS_DYN = decode.FWD_HCA_STATE_BLOCKS_DYN
FWD_HCA_WEIGHT_BANK_SIZE = decode.FWD_HCA_WEIGHT_BANK_SIZE
FWD_PACKED_RAW_BLOCKS_DYN = decode.FWD_PACKED_RAW_BLOCKS_DYN
FWD_WEIGHT_BANK_SIZE = decode.FWD_WEIGHT_BANK_SIZE
GROUP_LOGIT_ROWS = decode.GROUP_LOGIT_ROWS
H = decode.H
HCA_B_DYN = decode.HCA_B_DYN
HCA_CMP_STORAGE_BLOCK_SIZE = decode.HCA_CMP_STORAGE_BLOCK_SIZE
HCA_CMP_TABLE_BLOCKS_DYN = decode.HCA_CMP_TABLE_BLOCKS_DYN
HCA_COMPRESS_RATIO = decode.HCA_COMPRESS_RATIO
HCA_COMPRESS_STATE_BLOCK_SIZE = decode.HCA_COMPRESS_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_DIM = decode.HCA_COMPRESS_STATE_DIM
HCA_COMPRESS_STATE_MAX_BLOCKS = decode.HCA_COMPRESS_STATE_MAX_BLOCKS
HCA_MAIN_OUT_DIM = decode.HCA_MAIN_OUT_DIM
HC_DIM = decode.HC_DIM
HC_FN_STORAGE_ROWS = decode.HC_FN_STORAGE_ROWS
HC_MULT = decode.HC_MULT
HEAD_DIM = decode.HEAD_DIM
IDX_PAD = decode.IDX_PAD
KV_B_DYN = decode.KV_B_DYN
KV_T_DYN = decode.KV_T_DYN
LM_HEAD_TP_SIZE = decode.LM_HEAD_TP_SIZE
LM_HEAD_VOCAB = decode.LM_HEAD_VOCAB
LOCAL_O_GROUPS = decode.LOCAL_O_GROUPS
MAIN_HIDDEN_DIM = decode.MAIN_HIDDEN_DIM
MAX_LOGIT_ROWS = decode.MAX_LOGIT_ROWS
MIX_HC = decode.MIX_HC
MOE_INTER = decode.MOE_INTER
MOE_TOKENS = decode.MOE_TOKENS
N_EXPERTS_GLOBAL = decode.N_EXPERTS_GLOBAL
N_LOCAL = decode.N_LOCAL
N_RANKS = decode.N_RANKS
N_ROUTES = decode.N_ROUTES
O_GROUP_IN = decode.O_GROUP_IN
O_LORA = decode.O_LORA
O_WINDOW_ROWS = decode.O_WINDOW_ROWS
Q_LORA = decode.Q_LORA
RECV_MAX = decode.RECV_MAX
ROPE_HEAD_DIM = decode.ROPE_HEAD_DIM
SAMPLED_IDS_PAD = decode.SAMPLED_IDS_PAD
TOPK = decode.TOPK
TP_SIZE = decode.TP_SIZE
T_DYN = decode.T_DYN
VOCAB = decode.VOCAB
VOCAB_PER_TP = decode.VOCAB_PER_TP
WIN = decode.WIN
DECODE_BATCH = prepare.B
GROUP_DECODE_TOKENS = prepare.T
LOCAL_DECODE_BATCH = prepare.LOCAL_BATCH
LOCAL_DECODE_TOKENS = prepare.LOCAL_T
HCA_STATE_TABLE_BLOCKS = prepare.HCA_STATE_TABLE_BLOCKS
CSA_STATE_TABLE_BLOCKS = prepare.CSA_STATE_TABLE_BLOCKS
ORI_TABLE_BLOCKS_DYN = prepare.ORI_TABLE_BLOCKS_DYN
ROPE_ROWS_DYN = prepare.ROPE_ROWS_DYN
DRAFT_ATTENTION_WINDOW_ROWS = dspark.ATTENTION_WINDOW_ROWS
DRAFT_AUX_PAD = dspark.AUX_PAD

DRAFT_BLOCK_SIZE = dspark.BLOCK_SIZE
B_DYN = dspark.B_DYN
CP_CONTEXT_T_DYN = dspark.CP_CONTEXT_T_DYN
DRAFT_D = dspark.D
DRAFT_DSPARK_CP_SIZE = dspark.DSPARK_CP_SIZE
DRAFT_DSPARK_DRAFT_LAYERS = dspark.DSPARK_DRAFT_LAYERS
DRAFT_DSPARK_MARKOV_RANK = markov.DSPARK_MARKOV_RANK
DRAFT_DSPARK_QUERY_WIDTH = dspark.DSPARK_QUERY_WIDTH
DRAFT_GROUP_LOGIT_ROWS = lmhead.GROUP_LOGIT_ROWS
DRAFT_H = dspark.H
DRAFT_HC_DIM = dspark.HC_DIM
DRAFT_HC_MULT = dspark.HC_MULT
DRAFT_HEAD_DIM = dspark.HEAD_DIM
DRAFT_IDX_PAD = dspark.IDX_PAD
DRAFT_LOCAL_O_GROUPS = dspark.LOCAL_O_GROUPS
DRAFT_MAIN_IN = dspark.MAIN_IN
DRAFT_MAX_LOGIT_ROWS = lmhead.MAX_LOGIT_ROWS
DRAFT_MIX_HC = dspark.MIX_HC
DRAFT_MOE_INTER = dspark.MOE_INTER
DRAFT_N_EXPERTS_GLOBAL = dspark.N_EXPERTS_GLOBAL
DRAFT_N_LOCAL = dspark.N_LOCAL
DRAFT_N_RANKS = dspark.N_RANKS
DRAFT_N_ROUTES = dspark.N_ROUTES
DRAFT_ORI_BLOCK_NUM = dspark.ORI_BLOCK_NUM
DRAFT_ORI_MAX_BLOCKS = dspark.ORI_MAX_BLOCKS
DRAFT_O_GROUP_IN = dspark.O_GROUP_IN
DRAFT_O_LORA = dspark.O_LORA
DRAFT_O_WINDOW_ROWS = dspark.O_WINDOW_ROWS
DRAFT_PREFILL_GROUP_CAP = dspark.PREFILL_GROUP_CAP
DRAFT_Q_LORA = dspark.Q_LORA
DRAFT_RECV_MAX = dspark.RECV_MAX
DRAFT_ROPE_DIM = dspark.ROPE_DIM
DRAFT_T = dspark.T
DRAFT_TOPK = dspark.TOPK
DRAFT_TP_SIZE = dspark.TP_SIZE
DRAFT_T_QUERY = dspark.T_QUERY
DRAFT_VOCAB = dspark.VOCAB
DRAFT_VOCAB_PER_TP = lmhead.VOCAB_PER_TP
BRIDGE_GROUP_METADATA_ROWS = prepare.GROUP_METADATA_ROWS
BRIDGE_METADATA_WIDTH = prepare.METADATA_WIDTH
BRIDGE_ROPE_CANDIDATE_ROWS = prepare.ROPE_CANDIDATE_ROWS

# Per-ring heap for the target forward, three draft layers, and their MoE graphs.
DSPARK_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4


@pl.jit(auto_scope=False)
def l2_decode_fwd_dspark(
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_WEIGHT_BANK_SIZE, D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[FWD_WEIGHT_BANK_SIZE, Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    raw_kv_pool: pl.InOut[pl.Tensor[[FWD_PACKED_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_wkv: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_MAIN_STATE_BLOCKS_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    csa_idx_wq_b: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_INNER_STATE_BLOCKS_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    csa_idx_kv_cache: pl.InOut[pl.Tensor[[FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    csa_idx_kv_scale: pl.InOut[pl.Tensor[[FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    csa_idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    hca_cmp_wkv: pl.Tensor[[FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[FWD_HCA_WEIGHT_BANK_SIZE * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[FWD_HCA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCKS_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCKS_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * VOCAB, TOPK], pl.INT32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, pl.NZ],
    routed_w1: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[FWD_WEIGHT_BANK_SIZE, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_WEIGHT_BANK_SIZE, MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_WEIGHT_BANK_SIZE, D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hidden_workspace: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_ping: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_attn_active: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.InOut[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    dspark_target_hidden: pl.InOut[pl.Tensor[[T_DYN, MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, decode.GREEDY_GRID_ROWS, decode.GRAMMAR_SEGMENT_WORDS], pl.INT16],
    valid_draft_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    sampled_ids: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    state_slot_ids: pl.Tensor[[DSPARK_STATE_LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[DSPARK_STATE_LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[DSPARK_STATE_CAPACITY, DSPARK_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[DSPARK_STATE_CAPACITY, DSPARK_STATE_META_WIDTH], pl.INT32]],
    accepted_token_ids: pl.Out[pl.Tensor[[DSPARK_STATE_LOCAL_BATCH, SAMPLED_IDS_PAD], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[DSPARK_STATE_LOCAL_BATCH], pl.INT32]],
    group_state_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    group_state_generations: pl.Tensor[[DECODE_BATCH], pl.INT32],
    group_ori_block_table: pl.Tensor[[DECODE_BATCH, ORI_TABLE_BLOCKS_DYN], pl.INT32],
    group_hca_cmp_block_table: pl.Tensor[[DECODE_BATCH, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    group_csa_cmp_block_table: pl.Tensor[[DECODE_BATCH, CSA_CMP_MAX_BLOCKS], pl.INT32],
    group_idx_block_table: pl.Tensor[[DECODE_BATCH, CSA_IDX_MAX_BLOCKS], pl.INT32],
    group_hca_state_block_table: pl.Tensor[[DECODE_BATCH, HCA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    group_csa_state_block_table: pl.Tensor[[DECODE_BATCH, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    group_csa_inner_state_block_table: pl.Tensor[[DECODE_BATCH, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    swa_rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio4_rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio4_rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio128_rope_cos_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio128_rope_sin_table: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    draft_main_proj_weight: pl.Tensor[[DRAFT_D, DRAFT_MAIN_IN], pl.BF16],
    draft_main_norm_weight: pl.Tensor[[DRAFT_D], pl.BF16],
    draft_embedding_weight: pl.Tensor[[DRAFT_VOCAB, DRAFT_D], pl.BF16],
    draft_block_tables: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, B_DYN, DRAFT_ORI_MAX_BLOCKS], pl.INT32],
    draft_hc_attn_fn: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_w: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.BF16],
    draft_wq_a: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_Q_LORA], pl.BF16, pl.NZ],
    draft_wq_b: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_H * DRAFT_HEAD_DIM], pl.INT8, pl.NZ],
    draft_wq_b_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_H * DRAFT_HEAD_DIM], pl.FP32],
    draft_wkv: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D, DRAFT_HEAD_DIM], pl.BF16],
    draft_gamma_cq: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_Q_LORA], pl.BF16],
    draft_gamma_ckv: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_HEAD_DIM], pl.BF16],
    draft_kv_caches: pl.InOut[pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCK_NUM, DRAFT_BLOCK_SIZE, 1, DRAFT_HEAD_DIM], pl.BF16]],
    draft_attn_sink: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_LOCAL_O_GROUPS, DRAFT_O_LORA, DRAFT_O_GROUP_IN], pl.BF16, pl.NZ],
    draft_wo_b: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_LOCAL_O_GROUPS, DRAFT_D, DRAFT_O_LORA], pl.INT8, pl.NZ],
    draft_wo_b_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.FP32],
    draft_hc_ffn_fn: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_w: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.BF16],
    draft_gate_w: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_EXPERTS_GLOBAL, DRAFT_D], pl.FP32],
    draft_gate_bias: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_EXPERTS_GLOBAL], pl.FP32],
    draft_tid2eid: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_VOCAB, DRAFT_TOPK], pl.INT32],
    draft_routed_w1: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER, DRAFT_D], pl.INT8, pl.NZ],
    draft_routed_w1_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER], pl.FP32],
    draft_routed_w3: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER, DRAFT_D], pl.INT8, pl.NZ],
    draft_routed_w3_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER], pl.FP32],
    draft_routed_w2: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_D, DRAFT_MOE_INTER], pl.INT8, pl.NZ],
    draft_routed_w2_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_D], pl.FP32],
    draft_shared_w1: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.INT8, pl.NZ],
    draft_shared_w1_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MOE_INTER], pl.FP32],
    draft_shared_w3: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.INT8, pl.NZ],
    draft_shared_w3_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MOE_INTER], pl.FP32],
    draft_shared_w2: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.INT8, pl.NZ],
    draft_shared_w2_scale: pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.FP32],
    draft_hc_head_fn: pl.Tensor[[DRAFT_HC_MULT, DRAFT_HC_DIM], pl.FP32],
    draft_hc_head_scale: pl.Tensor[[1], pl.FP32],
    draft_hc_head_base: pl.Tensor[[DRAFT_HC_MULT], pl.FP32],
    draft_initial_hidden: pl.Out[pl.Tensor[[DRAFT_T, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_intermediate_hidden: pl.Out[pl.Tensor[[DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_T, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_head_hidden: pl.Out[pl.Tensor[[B_DYN, DRAFT_DSPARK_QUERY_WIDTH, DRAFT_D], pl.BF16]],
    draft_hidden_gather_window: pld.DistributedTensor[[DRAFT_PREFILL_GROUP_CAP, DRAFT_D], pl.BF16],
    draft_hidden_gather_signal: pld.DistributedTensor[[DRAFT_DSPARK_CP_SIZE, 1], pl.INT32],
    draft_attention_window: pld.DistributedTensor[[DRAFT_ATTENTION_WINDOW_ROWS, DRAFT_O_GROUP_IN], pl.BF16],
    draft_attention_signal: pld.DistributedTensor[[DRAFT_TP_SIZE, 1], pl.INT32],
    draft_o_window: pld.DistributedTensor[[DRAFT_O_WINDOW_ROWS, DRAFT_D], pl.BF16],
    draft_o_signal: pld.DistributedTensor[[DRAFT_TP_SIZE, 1], pl.INT32],
    draft_recv_meta: pld.DistributedTensor[[DRAFT_N_RANKS, DRAFT_N_LOCAL], pl.INT32],
    draft_recv_x: pld.DistributedTensor[[DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_D], pl.INT8],
    draft_recv_aux: pld.DistributedTensor[[DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_AUX_PAD], pl.FP32],
    draft_recv_route: pld.DistributedTensor[[DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_IDX_PAD], pl.INT32],
    draft_arrived: pld.DistributedTensor[[DRAFT_N_RANKS, 1], pl.INT32],
    draft_data_arrived: pld.DistributedTensor[[DRAFT_N_RANKS, 1], pl.INT32],
    draft_routed_y_buf: pld.DistributedTensor[[DRAFT_N_ROUTES, DRAFT_D], pl.BF16],
    draft_combine_arrived: pld.DistributedTensor[[DRAFT_N_RANKS, 1], pl.INT32],
    draft_final_norm_weight: pl.Tensor[[DRAFT_D], pl.BF16],
    draft_lm_head_weight: pl.Tensor[[DRAFT_VOCAB_PER_TP, DRAFT_D], pl.BF16, pl.NZ],
    draft_markov_w1: pl.Tensor[[DRAFT_VOCAB, DRAFT_DSPARK_MARKOV_RANK], pl.BF16],
    draft_markov_w2: pl.Tensor[[DRAFT_VOCAB, DRAFT_DSPARK_MARKOV_RANK], pl.BF16],
    draft_confidence_head_weight: pl.Tensor[[1, DRAFT_D + DRAFT_DSPARK_MARKOV_RANK], pl.FP32],
    draft_draft_token_ids: pl.Out[pl.Tensor[[B_DYN, DRAFT_DSPARK_QUERY_WIDTH], pl.INT32]],
    draft_confidence_probs: pl.Out[pl.Tensor[[B_DYN, DRAFT_DSPARK_QUERY_WIDTH], pl.FP32]],
    draft_markov_hidden_window: pld.DistributedTensor[[DRAFT_GROUP_LOGIT_ROWS, DRAFT_D], pl.BF16],
    draft_markov_hidden_done: pld.DistributedTensor[[DRAFT_TP_SIZE, 1], pl.INT32],
    draft_markov_logits_window: pld.DistributedTensor[[DRAFT_MAX_LOGIT_ROWS, DRAFT_VOCAB], pl.FP32],
    draft_markov_logits_done: pld.DistributedTensor[[DRAFT_TP_SIZE, 1], pl.INT32],
    draft_bridge_metadata_window: pld.DistributedTensor[[BRIDGE_GROUP_METADATA_ROWS, BRIDGE_METADATA_WIDTH], pl.INT64],
    draft_bridge_metadata_signal: pld.DistributedTensor[[DRAFT_DSPARK_CP_SIZE, 1], pl.INT32],
    draft_bridge_rope_cos_window: pld.DistributedTensor[[BRIDGE_GROUP_METADATA_ROWS, DRAFT_ROPE_DIM], pl.BF16],
    draft_bridge_rope_sin_window: pld.DistributedTensor[[BRIDGE_GROUP_METADATA_ROWS, DRAFT_ROPE_DIM], pl.BF16],
    draft_bridge_rope_cos_signal: pld.DistributedTensor[[DRAFT_DSPARK_CP_SIZE, 1], pl.INT32],
    draft_bridge_rope_sin_signal: pld.DistributedTensor[[DRAFT_DSPARK_CP_SIZE, 1], pl.INT32],
):
    """Run the complete recurrent DSpark decode step in one rank-local L2."""
    group_ori_block_table.bind_dynamic(1, ORI_TABLE_BLOCKS_DYN)
    group_hca_cmp_block_table.bind_dynamic(1, HCA_CMP_TABLE_BLOCKS_DYN)
    group_hca_state_block_table.bind_dynamic(1, HCA_GROUP_STATE_BLOCKS_DYN)
    group_csa_state_block_table.bind_dynamic(1, CSA_GROUP_STATE_BLOCKS_DYN)
    group_csa_inner_state_block_table.bind_dynamic(1, CSA_GROUP_STATE_BLOCKS_DYN)
    swa_rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    swa_rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio4_rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio4_rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio128_rope_cos_table.bind_dynamic(0, ROPE_ROWS_DYN)
    ratio128_rope_sin_table.bind_dynamic(0, ROPE_ROWS_DYN)
    # The drafter bridge outlives the target scope: the Markov sampler and the
    # state commit read it from their own scopes, so it is built at frame level.
    draft_batch = pl.tensor.dim(draft_head_hidden, 0)
    draft_group_context_tokens = DRAFT_DSPARK_CP_SIZE * draft_batch * SAMPLED_IDS_PAD
    bridge_num_sampled = pl.create_tensor([draft_batch], dtype=pl.INT32)
    bridge_last_sampled = pl.create_tensor([draft_batch], dtype=pl.INT64)
    bridge_next_prefill_tokens = pl.create_tensor([draft_batch], dtype=pl.INT64)
    bridge_anchor_positions = pl.create_tensor([draft_batch], dtype=pl.INT32)
    bridge_state_slot_ids = pl.create_tensor([draft_batch], dtype=pl.INT32)
    bridge_state_generations = pl.create_tensor([draft_batch], dtype=pl.INT32)
    bridge_logit_row_indices = pl.create_tensor([DRAFT_MAX_LOGIT_ROWS], dtype=pl.INT32)
    bridge_context_group_position_ids = pl.create_tensor([draft_group_context_tokens], dtype=pl.INT32)
    bridge_context_group_slot_mapping = pl.create_tensor(
        [DRAFT_DSPARK_DRAFT_LAYERS, draft_group_context_tokens],
        dtype=pl.INT64,
    )
    bridge_query_group_position_ids = pl.create_tensor([DRAFT_DSPARK_CP_SIZE * DRAFT_T_QUERY], dtype=pl.INT32)
    bridge_query_group_slot_mapping = pl.create_tensor(
        [DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_DSPARK_CP_SIZE * DRAFT_T_QUERY],
        dtype=pl.INT64,
    )
    bridge_context_group_freqs_cos = pl.create_tensor([draft_group_context_tokens, DRAFT_ROPE_DIM], dtype=pl.BF16)
    bridge_context_group_freqs_sin = pl.create_tensor([draft_group_context_tokens, DRAFT_ROPE_DIM], dtype=pl.BF16)
    bridge_query_freqs_cos = pl.create_tensor([DRAFT_T_QUERY, DRAFT_ROPE_DIM], dtype=pl.BF16)
    bridge_query_freqs_sin = pl.create_tensor([DRAFT_T_QUERY, DRAFT_ROPE_DIM], dtype=pl.BF16)
    bridge_query_group_freqs_cos = pl.create_tensor(
        [DRAFT_DSPARK_CP_SIZE * DRAFT_T_QUERY, DRAFT_ROPE_DIM],
        dtype=pl.BF16,
    )
    bridge_query_group_freqs_sin = pl.create_tensor(
        [DRAFT_DSPARK_CP_SIZE * DRAFT_T_QUERY, DRAFT_ROPE_DIM],
        dtype=pl.BF16,
    )
    with pl.scope():
        # Match fused MTP's ownership model: position-dependent decode
        # metadata is invocation-local scratch, not a serving-owned InOut ABI.
        # The legacy parameters remain in the transitional outer signature
        # until the compact L3 ABI is validated, but are deliberately shadowed
        # here so no producer/fanout lifetime can cross decode invocations.
        prepared_input_ids = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT64)
        prepared_position_ids_local = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_position_ids = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT32)
        prepared_logit_row_indices = pl.create_tensor([MAX_LOGIT_ROWS], dtype=pl.INT32)
        prepared_sampled_row_offsets = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT32)
        prepared_drafter_target_hidden = pl.create_tensor([LOCAL_DECODE_TOKENS, MAIN_HIDDEN_DIM], dtype=pl.BF16)
        prepared_drafter_context_positions = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_drafter_context_valid = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_drafter_last_sampled = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT64)
        prepared_drafter_anchor_positions = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT32)
        prepared_drafter_row_offsets = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT32)
        prepared_swa_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_swa_indices = pl.create_tensor([LOCAL_DECODE_TOKENS, WIN], dtype=pl.INT32)
        prepared_swa_lens = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_csa_ori_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_csa_window_swa_indices = pl.create_tensor([LOCAL_DECODE_TOKENS, WIN], dtype=pl.INT32)
        prepared_csa_window_swa_lens = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_csa_cmp_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_csa_idx_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_csa_state_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_csa_inner_state_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_csa_kv_seq_lens = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT32)
        prepared_hca_ori_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_hca_window_swa_indices = pl.create_tensor([LOCAL_DECODE_TOKENS, WIN], dtype=pl.INT32)
        prepared_hca_window_swa_lens = pl.create_tensor([LOCAL_DECODE_TOKENS], dtype=pl.INT32)
        prepared_hca_cmp_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_hca_state_slot_mapping = pl.create_tensor([GROUP_DECODE_TOKENS], dtype=pl.INT64)
        prepared_hca_kv_seq_lens = pl.create_tensor([LOCAL_DECODE_BATCH], dtype=pl.INT32)
        prepared_freqs_cos = pl.create_tensor([LOCAL_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_freqs_sin = pl.create_tensor([LOCAL_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_compressed_freqs_cos = pl.create_tensor([LOCAL_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_compressed_freqs_sin = pl.create_tensor([LOCAL_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_csa_cmp_freqs_cos = pl.create_tensor([GROUP_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_csa_cmp_freqs_sin = pl.create_tensor([GROUP_DECODE_TOKENS, ROPE_HEAD_DIM], dtype=pl.BF16)
        prepared_hca_cmp_freqs_cos = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        prepared_hca_cmp_freqs_sin = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
        prepared_hca_compress_state_block_table = pl.create_tensor([DECODE_BATCH, HCA_STATE_TABLE_BLOCKS], dtype=pl.INT32)
        prepared_csa_compress_state_block_table = pl.create_tensor([DECODE_BATCH, CSA_STATE_TABLE_BLOCKS], dtype=pl.INT32)
        prepared_csa_inner_compress_state_block_table = pl.create_tensor(
            [DECODE_BATCH, CSA_STATE_TABLE_BLOCKS], dtype=pl.INT32
        )
        local_active_widths = pl.create_tensor([DSPARK_STATE_LOCAL_BATCH], dtype=pl.INT32)
        group_active_widths = pl.create_tensor([DECODE_BATCH], dtype=pl.INT32)
        prepared_draft_rope_cos_candidates = pl.create_tensor(
            [draft_batch, BRIDGE_ROPE_CANDIDATE_ROWS, DRAFT_ROPE_DIM],
            dtype=pl.BF16,
        )
        prepared_draft_rope_sin_candidates = pl.create_tensor(
            [draft_batch, BRIDGE_ROPE_CANDIDATE_ROWS, DRAFT_ROPE_DIM],
            dtype=pl.BF16,
        )
        prepare_target_group_from_device_state(
            group_state_slot_ids, group_state_generations, valid_draft_counts,
            state_tokens, state_meta,
            prepared_input_ids, prepared_position_ids_local, prepared_position_ids,
            prepared_csa_kv_seq_lens, prepared_hca_kv_seq_lens,
            prepared_logit_row_indices, prepared_sampled_row_offsets,
            local_active_widths, group_active_widths,
            tp_rank,
        )
        build_group_decode_metadata(
            prepared_position_ids, group_active_widths,
            group_ori_block_table, group_hca_cmp_block_table, group_csa_cmp_block_table,
            group_idx_block_table, group_hca_state_block_table, group_csa_state_block_table,
            group_csa_inner_state_block_table, prepared_hca_compress_state_block_table,
            prepared_csa_compress_state_block_table,
            prepared_csa_inner_compress_state_block_table, prepared_swa_slot_mapping, prepared_swa_indices,
            prepared_swa_lens, prepared_hca_ori_slot_mapping, prepared_hca_window_swa_indices,
            prepared_hca_window_swa_lens, prepared_hca_cmp_slot_mapping,
            prepared_hca_state_slot_mapping, prepared_csa_ori_slot_mapping, prepared_csa_window_swa_indices,
            prepared_csa_window_swa_lens, prepared_csa_cmp_slot_mapping, prepared_csa_idx_slot_mapping,
            prepared_csa_state_slot_mapping,
            prepared_csa_inner_state_slot_mapping,
            tp_rank,
        )
        gather_group_decode_rope_rows(
            swa_rope_cos_table, swa_rope_sin_table, ratio4_rope_cos_table, ratio4_rope_sin_table,
            ratio128_rope_cos_table, ratio128_rope_sin_table,
            prepared_position_ids, group_active_widths,
            prepared_freqs_cos, prepared_freqs_sin,
            prepared_compressed_freqs_cos, prepared_compressed_freqs_sin,
            prepared_csa_cmp_freqs_cos, prepared_csa_cmp_freqs_sin,
            prepared_hca_cmp_freqs_cos, prepared_hca_cmp_freqs_sin,
            prepared_draft_rope_cos_candidates, prepared_draft_rope_sin_candidates,
            tp_rank,
        )
        target_accept_ready = pl.create_tensor([1], dtype=pl.INT32)
        decode_fwd(
            embed_weight, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b,
            wq_b_scale, wkv, gamma_cq, gamma_ckv, raw_kv_pool,
            prepared_freqs_cos, prepared_freqs_sin,
            prepared_compressed_freqs_cos, prepared_compressed_freqs_sin,
            prepared_swa_slot_mapping, prepared_swa_indices, prepared_swa_lens,
            prepared_position_ids_local, prepared_position_ids,
            prepared_csa_cmp_freqs_cos, prepared_csa_cmp_freqs_sin, csa_cmp_wkv,
            csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w, csa_compress_state,
            prepared_csa_compress_state_block_table,
            csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
            csa_hadamard_idx, csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
            csa_inner_compress_state, prepared_csa_inner_compress_state_block_table, csa_cmp_kv,
            csa_cmp_block_table, csa_idx_kv_cache, csa_idx_kv_scale, csa_idx_block_table,
            prepared_csa_ori_slot_mapping,
            prepared_csa_window_swa_indices, prepared_csa_window_swa_lens,
            prepared_csa_cmp_slot_mapping, prepared_csa_idx_slot_mapping,
            prepared_csa_state_slot_mapping, prepared_csa_inner_state_slot_mapping,
            prepared_csa_kv_seq_lens,
            prepared_hca_cmp_freqs_cos, prepared_hca_cmp_freqs_sin,
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w, hca_compress_state,
            prepared_hca_compress_state_block_table,
            hca_cmp_kv, hca_cmp_block_table, prepared_hca_ori_slot_mapping,
            prepared_hca_window_swa_indices, prepared_hca_window_swa_lens,
            prepared_hca_cmp_slot_mapping, prepared_hca_state_slot_mapping,
            prepared_hca_kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, hc_ffn_fn,
            hc_ffn_scale, hc_ffn_base, norm_w, gate_w, gate_bias, tid2eid,
            prepared_input_ids,
            num_tokens_per_owner, hc_head_fn, hc_head_scale, hc_head_base, final_norm_w,
            lm_head_weight, prepared_logit_row_indices, routed_w1, routed_w1_scale, routed_w3,
            routed_w3_scale, routed_w2, routed_w2_scale, shared_w1, shared_w1_scale, shared_w3,
            shared_w3_scale, shared_w2, shared_w2_scale, hidden_workspace, x_ping, x_pong,
            x_attn_active, x_moe_next, pre_hc_hidden_out, dspark_target_hidden, x_out, logits,
            grammar_mask, sampled_ids, gather_window,
            gather_signal, attention_window, attention_signal, o_window, o_signal, recv_meta,
            recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf, combine_arrived,
            lm_head_hidden_window, lm_head_hidden_done, lm_head_logits_window, lm_head_logits_done,
            group_base, tp_rank, my_rank
        )
        accept_target_into_device_state(
            state_slot_ids, state_generations, valid_draft_counts, tp_rank,
            prepared_sampled_row_offsets, prepared_sampled_row_offsets,
            state_tokens, state_meta,
            sampled_ids, dspark_target_hidden,
            accepted_token_ids, accepted_counts,
            prepared_drafter_target_hidden, prepared_drafter_context_positions,
            prepared_drafter_context_valid, prepared_drafter_last_sampled, prepared_drafter_anchor_positions,
            prepared_drafter_row_offsets,
            target_accept_ready,
        )
        draft_context_tokens = draft_group_context_tokens // DRAFT_DSPARK_CP_SIZE
        draft_target_hidden = pl.create_tensor([draft_context_tokens, DRAFT_MAIN_IN], dtype=pl.BF16)
        with pl.spmd(draft_context_tokens, name_hint="dspark_accept_hidden_compact"):
            hidden_row = pl.tile.get_block_idx()
            source_row = hidden_row + pl.cast(pl.read(target_accept_ready, [0]), pl.INDEX)
            draft_target_hidden[
                hidden_row : hidden_row + 1, 0:DRAFT_MAIN_IN
            ] = prepared_drafter_target_hidden[
                source_row : source_row + 1, 0:DRAFT_MAIN_IN
            ]
        # Bare call: rebinding the returned outputs would version them inside
        # this scope, and the later scopes could no longer name them.
        prepare_drafter_after_target(
            target_accept_ready,
            state_slot_ids, state_generations,
            accepted_counts,
            prepared_drafter_context_positions, prepared_drafter_context_valid,
            prepared_drafter_last_sampled, prepared_drafter_anchor_positions,
            prepared_drafter_row_offsets, draft_block_tables,
            prepared_draft_rope_cos_candidates, prepared_draft_rope_sin_candidates,
            bridge_num_sampled,
            bridge_last_sampled, bridge_next_prefill_tokens, bridge_anchor_positions,
            bridge_state_slot_ids, bridge_state_generations,
            bridge_logit_row_indices,
            bridge_context_group_position_ids, bridge_context_group_slot_mapping,
            bridge_query_group_position_ids, bridge_query_group_slot_mapping,
            bridge_context_group_freqs_cos, bridge_context_group_freqs_sin,
            bridge_query_freqs_cos, bridge_query_freqs_sin,
            bridge_query_group_freqs_cos, bridge_query_group_freqs_sin,
            draft_bridge_metadata_window, draft_bridge_metadata_signal,
            draft_bridge_rope_cos_window, draft_bridge_rope_sin_window,
            draft_bridge_rope_cos_signal, draft_bridge_rope_sin_signal,
            group_base, tp_rank,
        )
        draft_head_hidden, draft_head_hidden_ready_tid = dspark_drafter(
            draft_target_hidden, draft_main_proj_weight, draft_main_norm_weight,
            bridge_num_sampled, bridge_last_sampled, bridge_next_prefill_tokens,
            draft_embedding_weight, bridge_context_group_position_ids,
            bridge_context_group_slot_mapping, bridge_anchor_positions, draft_block_tables,
            bridge_query_group_position_ids, bridge_query_group_slot_mapping,
            bridge_context_group_freqs_cos, bridge_context_group_freqs_sin,
            bridge_query_freqs_cos, bridge_query_freqs_sin,
            bridge_query_group_freqs_cos, bridge_query_group_freqs_sin,
            draft_hc_attn_fn, draft_hc_attn_scale, draft_hc_attn_base, draft_attn_norm_w,
            draft_wq_a, draft_wq_b, draft_wq_b_scale, draft_wkv, draft_gamma_cq, draft_gamma_ckv,
            draft_kv_caches, draft_attn_sink, draft_wo_a, draft_wo_b, draft_wo_b_scale,
            draft_hc_ffn_fn, draft_hc_ffn_scale, draft_hc_ffn_base, draft_ffn_norm_w, draft_gate_w,
            draft_gate_bias, draft_tid2eid, draft_routed_w1, draft_routed_w1_scale,
            draft_routed_w3, draft_routed_w3_scale, draft_routed_w2, draft_routed_w2_scale,
            draft_shared_w1, draft_shared_w1_scale, draft_shared_w3, draft_shared_w3_scale,
            draft_shared_w2, draft_shared_w2_scale, draft_hc_head_fn, draft_hc_head_scale,
            draft_hc_head_base, draft_initial_hidden, draft_intermediate_hidden, draft_head_hidden,
            draft_hidden_gather_window, draft_hidden_gather_signal, draft_attention_window,
            draft_attention_signal, draft_o_window, draft_o_signal, draft_recv_meta, draft_recv_x,
            draft_recv_aux, draft_recv_route, draft_arrived, draft_data_arrived,
            draft_routed_y_buf, draft_combine_arrived, group_base, tp_rank, my_rank,
        )
        fence_drafter_head_hidden(draft_head_hidden, draft_head_hidden_ready_tid)
    with pl.scope():
        draft_draft_token_ids, draft_confidence_probs = distributed_markov_sample(
            draft_head_hidden, draft_final_norm_weight, draft_lm_head_weight,
            bridge_logit_row_indices, bridge_num_sampled, bridge_last_sampled,
            bridge_next_prefill_tokens, draft_markov_w1, draft_markov_w2,
            draft_confidence_head_weight, draft_draft_token_ids, draft_confidence_probs,
            draft_markov_hidden_window, draft_markov_hidden_done,
            draft_markov_logits_window, draft_markov_logits_done,
            group_base, tp_rank,
        )
    with pl.scope():
        state_tokens, state_meta = commit_drafts_to_device_state(
            bridge_state_slot_ids, bridge_state_generations,
            state_tokens, state_meta, draft_draft_token_ids,
        )
    return (
        x_out, sampled_ids,
        accepted_token_ids, accepted_counts,
        draft_draft_token_ids, draft_confidence_probs,
        state_tokens, state_meta,
    )


@pl.jit.host
def l3_decode_fwd_dspark(
    embed_weight: pl.Tensor[[N_RANKS, EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    raw_kv_pool: pl.InOut[pl.Tensor[[N_RANKS, FWD_PACKED_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_MAIN_STATE_BLOCKS_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_INNER_STATE_BLOCKS_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    csa_idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    csa_idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    csa_idx_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_STATE_BLOCKS_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_CMP_BLOCKS_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * VOCAB, TOPK], pl.INT32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    routed_w1: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hidden_workspace: pl.Out[pl.Tensor[[N_RANKS, T_DYN, D], pl.BF16]],
    x_ping: pl.InOut[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_attn_active: pl.InOut[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.InOut[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    dspark_target_hidden: pl.InOut[pl.Tensor[[N_RANKS, T_DYN, MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[N_RANKS, T_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, decode.GREEDY_GRID_ROWS, decode.GRAMMAR_SEGMENT_WORDS], pl.INT16],
    valid_draft_counts: pl.Tensor[[N_RANKS, DECODE_BATCH], pl.INT32],
    sampled_ids: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    state_slot_ids: pl.Tensor[[N_RANKS, DSPARK_STATE_LOCAL_BATCH], pl.INT32],
    state_generations: pl.Tensor[[N_RANKS, DSPARK_STATE_LOCAL_BATCH], pl.INT32],
    state_tokens: pl.InOut[pl.Tensor[[N_RANKS, DSPARK_STATE_CAPACITY, DSPARK_STATE_TOKEN_WIDTH], pl.INT64]],
    state_meta: pl.InOut[pl.Tensor[[N_RANKS, DSPARK_STATE_CAPACITY, DSPARK_STATE_META_WIDTH], pl.INT32]],
    accepted_token_ids: pl.Out[pl.Tensor[[N_RANKS, DSPARK_STATE_LOCAL_BATCH, SAMPLED_IDS_PAD], pl.INT32]],
    accepted_counts: pl.Out[pl.Tensor[[N_RANKS, DSPARK_STATE_LOCAL_BATCH], pl.INT32]],
    group_state_slot_ids: pl.Tensor[[N_RANKS, DECODE_BATCH], pl.INT32],
    group_state_generations: pl.Tensor[[N_RANKS, DECODE_BATCH], pl.INT32],
    group_ori_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, ORI_TABLE_BLOCKS_DYN], pl.INT32],
    group_hca_cmp_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    group_csa_cmp_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, CSA_CMP_MAX_BLOCKS], pl.INT32],
    group_idx_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, CSA_IDX_MAX_BLOCKS], pl.INT32],
    group_hca_state_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, HCA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    group_csa_state_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    group_csa_inner_state_block_table: pl.Tensor[[N_RANKS, DECODE_BATCH, CSA_GROUP_STATE_BLOCKS_DYN], pl.INT32],
    swa_rope_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_rope_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio4_rope_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio4_rope_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio128_rope_cos_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ratio128_rope_sin_table: pl.Tensor[[N_RANKS, ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.BF16],
    draft_initial_hidden: pl.Out[pl.Tensor[[DRAFT_N_RANKS, DRAFT_T, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_intermediate_hidden: pl.Out[pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_T, DRAFT_HC_MULT, DRAFT_D], pl.FP32]],
    draft_main_proj_weight: pl.Tensor[[DRAFT_N_RANKS, DRAFT_D, DRAFT_MAIN_IN], pl.BF16],
    draft_main_norm_weight: pl.Tensor[[DRAFT_N_RANKS, DRAFT_D], pl.BF16],
    draft_embedding_weight: pl.Tensor[[DRAFT_N_RANKS, DRAFT_VOCAB, DRAFT_D], pl.BF16],
    draft_block_tables: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, B_DYN, DRAFT_ORI_MAX_BLOCKS], pl.INT32],
    draft_hc_attn_fn: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_attn_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    draft_hc_attn_base: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC], pl.FP32],
    draft_attn_norm_w: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.BF16],
    draft_wq_a: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_Q_LORA], pl.BF16],
    draft_wq_b: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_Q_LORA, DRAFT_H * DRAFT_HEAD_DIM], pl.INT8],
    draft_wq_b_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_H * DRAFT_HEAD_DIM], pl.FP32],
    draft_wkv: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D, DRAFT_HEAD_DIM], pl.BF16],
    draft_gamma_cq: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_Q_LORA], pl.BF16],
    draft_gamma_ckv: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_HEAD_DIM], pl.BF16],
    draft_kv_caches: pl.InOut[pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_ORI_BLOCK_NUM, DRAFT_BLOCK_SIZE, 1, DRAFT_HEAD_DIM], pl.BF16]],
    draft_attn_sink: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_H], pl.FP32],
    draft_wo_a: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_LOCAL_O_GROUPS, DRAFT_O_LORA, DRAFT_O_GROUP_IN], pl.BF16],
    draft_wo_b: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_LOCAL_O_GROUPS, DRAFT_D, DRAFT_O_LORA], pl.INT8],
    draft_wo_b_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.FP32],
    draft_hc_ffn_fn: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC, DRAFT_HC_DIM], pl.FP32],
    draft_hc_ffn_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * 3], pl.FP32],
    draft_hc_ffn_base: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MIX_HC], pl.FP32],
    draft_ffn_norm_w: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.BF16],
    draft_gate_w: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_EXPERTS_GLOBAL, DRAFT_D], pl.FP32],
    draft_gate_bias: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_EXPERTS_GLOBAL], pl.FP32],
    draft_tid2eid: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_VOCAB, DRAFT_TOPK], pl.INT32],
    draft_routed_w1: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER, DRAFT_D], pl.INT8],
    draft_routed_w1_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER], pl.FP32],
    draft_routed_w3: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER, DRAFT_D], pl.INT8],
    draft_routed_w3_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_MOE_INTER], pl.FP32],
    draft_routed_w2: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_D, DRAFT_MOE_INTER], pl.INT8],
    draft_routed_w2_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_N_LOCAL, DRAFT_D], pl.FP32],
    draft_shared_w1: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.INT8],
    draft_shared_w1_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MOE_INTER], pl.FP32],
    draft_shared_w3: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_MOE_INTER, DRAFT_D], pl.INT8],
    draft_shared_w3_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_MOE_INTER], pl.FP32],
    draft_shared_w2: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS, DRAFT_D, DRAFT_MOE_INTER], pl.INT8],
    draft_shared_w2_scale: pl.Tensor[[DRAFT_N_RANKS, DRAFT_DSPARK_DRAFT_LAYERS * DRAFT_D], pl.FP32],
    draft_hc_head_fn: pl.Tensor[[DRAFT_N_RANKS, DRAFT_HC_MULT, DRAFT_HC_DIM], pl.FP32],
    draft_hc_head_scale: pl.Tensor[[DRAFT_N_RANKS, 1], pl.FP32],
    draft_hc_head_base: pl.Tensor[[DRAFT_N_RANKS, DRAFT_HC_MULT], pl.FP32],
    draft_head_hidden: pl.Out[pl.Tensor[[DRAFT_N_RANKS, B_DYN, DRAFT_DSPARK_QUERY_WIDTH, DRAFT_D], pl.BF16]],
    draft_final_norm_weight: pl.Tensor[[DRAFT_N_RANKS, DRAFT_D], pl.BF16],
    draft_lm_head_weight: pl.Tensor[[DRAFT_N_RANKS, DRAFT_VOCAB_PER_TP, DRAFT_D], pl.BF16],
    draft_markov_w1: pl.Tensor[[DRAFT_N_RANKS, DRAFT_VOCAB, DRAFT_DSPARK_MARKOV_RANK], pl.BF16],
    draft_markov_w2: pl.Tensor[[DRAFT_N_RANKS, DRAFT_VOCAB, DRAFT_DSPARK_MARKOV_RANK], pl.BF16],
    draft_confidence_head_weight: pl.Tensor[[DRAFT_N_RANKS, 1, DRAFT_D + DRAFT_DSPARK_MARKOV_RANK], pl.FP32],
    draft_draft_token_ids: pl.Out[pl.Tensor[[DRAFT_N_RANKS, B_DYN, DRAFT_DSPARK_QUERY_WIDTH], pl.INT32]],
    draft_confidence_probs: pl.Out[pl.Tensor[[DRAFT_N_RANKS, B_DYN, DRAFT_DSPARK_QUERY_WIDTH], pl.FP32]],
):
    """Launch one complete DSpark decode L2 on every rank."""
    'Prepare, run target decode, and accept K=7 output in one L3 graph.'
    embed_weight.bind_dynamic(1, EMBED_VOCAB_DYN)
    hidden_workspace.bind_dynamic(1, T_DYN)
    dspark_target_hidden.bind_dynamic(1, T_DYN)
    x_ping.bind_dynamic(1, T_DYN)
    raw_kv_pool.bind_dynamic(1, FWD_PACKED_RAW_BLOCKS_DYN)
    csa_compress_state.bind_dynamic(1, FWD_CSA_MAIN_STATE_BLOCKS_DYN)
    csa_inner_compress_state.bind_dynamic(1, FWD_CSA_INNER_STATE_BLOCKS_DYN)
    csa_cmp_kv.bind_dynamic(1, FWD_CSA_CMP_BLOCKS_DYN)
    csa_cmp_block_table.bind_dynamic(1, CSA_B_DYN)
    csa_idx_kv_cache.bind_dynamic(1, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_kv_scale.bind_dynamic(1, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_block_table.bind_dynamic(1, CSA_B_DYN)
    hca_compress_state.bind_dynamic(1, FWD_HCA_STATE_BLOCKS_DYN)
    hca_cmp_kv.bind_dynamic(1, FWD_HCA_CMP_BLOCKS_DYN)
    hca_cmp_block_table.bind_dynamic(1, HCA_B_DYN)
    hca_cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    x_pong.bind_dynamic(1, T_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    pre_hc_hidden_out.bind_dynamic(1, T_DYN)
    x_out.bind_dynamic(1, T_DYN)
    group_ori_block_table.bind_dynamic(2, ORI_TABLE_BLOCKS_DYN)
    group_hca_cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    group_hca_state_block_table.bind_dynamic(2, HCA_GROUP_STATE_BLOCKS_DYN)
    group_csa_state_block_table.bind_dynamic(2, CSA_GROUP_STATE_BLOCKS_DYN)
    group_csa_inner_state_block_table.bind_dynamic(2, CSA_GROUP_STATE_BLOCKS_DYN)
    swa_rope_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    swa_rope_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio4_rope_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio4_rope_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio128_rope_cos_table.bind_dynamic(1, ROPE_ROWS_DYN)
    ratio128_rope_sin_table.bind_dynamic(1, ROPE_ROWS_DYN)
    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    draft_block_tables.bind_dynamic(2, B_DYN)
    draft_head_hidden.bind_dynamic(1, B_DYN)
    draft_draft_token_ids.bind_dynamic(1, B_DYN)
    draft_confidence_probs.bind_dynamic(1, B_DYN)
    draft_hidden_gather_buf = pld.alloc_window_buffer([DRAFT_PREFILL_GROUP_CAP, DRAFT_D], dtype=pl.BF16)
    draft_hidden_signal_buf = pld.alloc_window_buffer([DRAFT_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
    draft_attention_buf = pld.alloc_window_buffer([DRAFT_ATTENTION_WINDOW_ROWS, DRAFT_O_GROUP_IN], dtype=pl.BF16)
    draft_attention_signal_buf = pld.alloc_window_buffer([DRAFT_TP_SIZE, 1], dtype=pl.INT32)
    draft_o_buf = pld.alloc_window_buffer([DRAFT_O_WINDOW_ROWS, DRAFT_D], dtype=pl.BF16)
    draft_o_signal_buf = pld.alloc_window_buffer([DRAFT_TP_SIZE, 1], dtype=pl.INT32)
    draft_recv_meta_buf = pld.alloc_window_buffer([DRAFT_N_RANKS, DRAFT_N_LOCAL], dtype=pl.INT32)
    draft_recv_x_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_D], dtype=pl.INT8)
    draft_recv_aux_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_AUX_PAD], dtype=pl.FP32)
    draft_recv_route_buf = pld.alloc_window_buffer([DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_IDX_PAD], dtype=pl.INT32)
    draft_arrived_buf = pld.alloc_window_buffer([DRAFT_N_RANKS, 1], dtype=pl.INT32)
    draft_data_arrived_buf = pld.alloc_window_buffer([DRAFT_N_RANKS, 1], dtype=pl.INT32)
    draft_routed_y_buf = pld.alloc_window_buffer([DRAFT_N_ROUTES, DRAFT_D], dtype=pl.BF16)
    draft_combine_buf = pld.alloc_window_buffer([DRAFT_N_RANKS, 1], dtype=pl.INT32)
    draft_markov_hidden_buf = pld.alloc_window_buffer(DRAFT_GROUP_LOGIT_ROWS * DRAFT_D * 2)
    draft_markov_logits_buf = pld.alloc_window_buffer(DRAFT_MAX_LOGIT_ROWS * DRAFT_VOCAB * 4)
    draft_markov_hidden_done_buf = pld.alloc_window_buffer(DRAFT_TP_SIZE * 4)
    draft_markov_logits_done_buf = pld.alloc_window_buffer(DRAFT_TP_SIZE * 4)
    draft_bridge_metadata_buf = pld.alloc_window_buffer(BRIDGE_GROUP_METADATA_ROWS * BRIDGE_METADATA_WIDTH * 8)
    draft_bridge_metadata_signal_buf = pld.alloc_window_buffer(DRAFT_DSPARK_CP_SIZE * 4)
    draft_bridge_rope_cos_buf = pld.alloc_window_buffer(BRIDGE_GROUP_METADATA_ROWS * DRAFT_ROPE_DIM * 2)
    draft_bridge_rope_sin_buf = pld.alloc_window_buffer(BRIDGE_GROUP_METADATA_ROWS * DRAFT_ROPE_DIM * 2)
    draft_bridge_rope_cos_signal_buf = pld.alloc_window_buffer(DRAFT_DSPARK_CP_SIZE * 4)
    draft_bridge_rope_sin_signal_buf = pld.alloc_window_buffer(DRAFT_DSPARK_CP_SIZE * 4)
    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        draft_hidden_gather_window = pld.window(draft_hidden_gather_buf, [DRAFT_PREFILL_GROUP_CAP, DRAFT_D], dtype=pl.BF16)
        draft_hidden_gather_signal = pld.window(draft_hidden_signal_buf, [DRAFT_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
        draft_attention_window = pld.window(draft_attention_buf, [DRAFT_ATTENTION_WINDOW_ROWS, DRAFT_O_GROUP_IN], dtype=pl.BF16)
        draft_attention_signal = pld.window(draft_attention_signal_buf, [DRAFT_TP_SIZE, 1], dtype=pl.INT32)
        draft_o_window = pld.window(draft_o_buf, [DRAFT_O_WINDOW_ROWS, DRAFT_D], dtype=pl.BF16)
        draft_o_signal = pld.window(draft_o_signal_buf, [DRAFT_TP_SIZE, 1], dtype=pl.INT32)
        draft_recv_meta_window = pld.window(draft_recv_meta_buf, [DRAFT_N_RANKS, DRAFT_N_LOCAL], dtype=pl.INT32)
        draft_recv_x_window = pld.window(draft_recv_x_buf, [DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_D], dtype=pl.INT8)
        draft_recv_aux_window = pld.window(draft_recv_aux_buf, [DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_AUX_PAD], dtype=pl.FP32)
        draft_recv_route_window = pld.window(draft_recv_route_buf, [DRAFT_N_LOCAL * DRAFT_RECV_MAX, DRAFT_IDX_PAD], dtype=pl.INT32)
        draft_arrived_window = pld.window(draft_arrived_buf, [DRAFT_N_RANKS, 1], dtype=pl.INT32)
        draft_data_arrived_window = pld.window(draft_data_arrived_buf, [DRAFT_N_RANKS, 1], dtype=pl.INT32)
        draft_routed_y_window = pld.window(draft_routed_y_buf, [DRAFT_N_ROUTES, DRAFT_D], dtype=pl.BF16)
        draft_combine_window = pld.window(draft_combine_buf, [DRAFT_N_RANKS, 1], dtype=pl.INT32)
        draft_markov_hidden_window = pld.window(draft_markov_hidden_buf, [DRAFT_GROUP_LOGIT_ROWS, DRAFT_D], dtype=pl.BF16)
        draft_markov_hidden_done = pld.window(draft_markov_hidden_done_buf, [DRAFT_TP_SIZE, 1], dtype=pl.INT32)
        draft_markov_logits_window = pld.window(draft_markov_logits_buf, [DRAFT_MAX_LOGIT_ROWS, DRAFT_VOCAB], dtype=pl.FP32)
        draft_markov_logits_done = pld.window(draft_markov_logits_done_buf, [DRAFT_TP_SIZE, 1], dtype=pl.INT32)
        draft_bridge_metadata_window = pld.window(draft_bridge_metadata_buf, [BRIDGE_GROUP_METADATA_ROWS, BRIDGE_METADATA_WIDTH], dtype=pl.INT64)
        draft_bridge_metadata_signal = pld.window(draft_bridge_metadata_signal_buf, [DRAFT_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
        draft_bridge_rope_cos_window = pld.window(draft_bridge_rope_cos_buf, [BRIDGE_GROUP_METADATA_ROWS, DRAFT_ROPE_DIM], dtype=pl.BF16)
        draft_bridge_rope_sin_window = pld.window(draft_bridge_rope_sin_buf, [BRIDGE_GROUP_METADATA_ROWS, DRAFT_ROPE_DIM], dtype=pl.BF16)
        draft_bridge_rope_cos_signal = pld.window(draft_bridge_rope_cos_signal_buf, [DRAFT_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
        draft_bridge_rope_sin_signal = pld.window(draft_bridge_rope_sin_signal_buf, [DRAFT_DSPARK_CP_SIZE, 1], dtype=pl.INT32)
        l2_decode_fwd_dspark(
            embed_weight[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank], gamma_cq[rank],
            gamma_ckv[rank], raw_kv_pool[rank], csa_cmp_wkv[rank],
            csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank], csa_compress_state[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank],
            csa_weights_proj[rank], csa_hadamard_idx[rank], csa_inner_wkv[rank],
            csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            csa_cmp_kv[rank], csa_cmp_block_table[rank], csa_idx_kv_cache[rank],
            csa_idx_kv_scale[rank], csa_idx_block_table[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank],
            hca_cmp_norm_w[rank], hca_compress_state[rank],
            hca_cmp_kv[rank], hca_cmp_block_table[rank], attn_sink[rank], wo_a[rank],
            wo_b[rank], wo_b_scale[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            num_tokens_per_owner, hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
            final_norm_w[rank], lm_head_weight[rank], routed_w1[rank],
            routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank], routed_w2[rank],
            routed_w2_scale[rank], shared_w1[rank], shared_w1_scale[rank], shared_w3[rank],
            shared_w3_scale[rank], shared_w2[rank], shared_w2_scale[rank], hidden_workspace[rank],
            x_ping[rank], x_pong[rank], x_attn_active[rank], x_moe_next[rank],
            pre_hc_hidden_out[rank], dspark_target_hidden[rank], x_out[rank], logits[rank],
            grammar_mask[rank], valid_draft_counts[rank], sampled_ids[rank],
            state_slot_ids[rank], state_generations[rank], state_tokens[rank],
            state_meta[rank], accepted_token_ids[rank], accepted_counts[rank],
            group_state_slot_ids[rank], group_state_generations[rank],
            group_ori_block_table[rank], group_hca_cmp_block_table[rank],
            group_csa_cmp_block_table[rank], group_idx_block_table[rank],
            group_hca_state_block_table[rank], group_csa_state_block_table[rank],
            group_csa_inner_state_block_table[rank],
            swa_rope_cos_table[rank],
            swa_rope_sin_table[rank], ratio4_rope_cos_table[rank],
            ratio4_rope_sin_table[rank], ratio128_rope_cos_table[rank],
            ratio128_rope_sin_table[rank],
            gather_window, gather_signal, attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived, routed_y_buf,
            combine_arrived, lm_head_hidden_window, lm_head_hidden_done, lm_head_logits_window,
            lm_head_logits_done, group_base, tp_rank, rank, draft_main_proj_weight[rank],
            draft_main_norm_weight[rank], draft_embedding_weight[rank],
            draft_block_tables[rank],
            draft_hc_attn_fn[rank], draft_hc_attn_scale[rank],
            draft_hc_attn_base[rank], draft_attn_norm_w[rank], draft_wq_a[rank],
            draft_wq_b[rank], draft_wq_b_scale[rank], draft_wkv[rank],
            draft_gamma_cq[rank], draft_gamma_ckv[rank], draft_kv_caches[rank],
            draft_attn_sink[rank], draft_wo_a[rank], draft_wo_b[rank],
            draft_wo_b_scale[rank], draft_hc_ffn_fn[rank],
            draft_hc_ffn_scale[rank], draft_hc_ffn_base[rank],
            draft_ffn_norm_w[rank], draft_gate_w[rank], draft_gate_bias[rank],
            draft_tid2eid[rank], draft_routed_w1[rank],
            draft_routed_w1_scale[rank], draft_routed_w3[rank],
            draft_routed_w3_scale[rank], draft_routed_w2[rank],
            draft_routed_w2_scale[rank], draft_shared_w1[rank],
            draft_shared_w1_scale[rank], draft_shared_w3[rank],
            draft_shared_w3_scale[rank], draft_shared_w2[rank],
            draft_shared_w2_scale[rank], draft_hc_head_fn[rank],
            draft_hc_head_scale[rank], draft_hc_head_base[rank],
            draft_initial_hidden[rank], draft_intermediate_hidden[rank],
            draft_head_hidden[rank], draft_hidden_gather_window, draft_hidden_gather_signal,
            draft_attention_window, draft_attention_signal, draft_o_window, draft_o_signal,
            draft_recv_meta_window, draft_recv_x_window, draft_recv_aux_window,
            draft_recv_route_window, draft_arrived_window, draft_data_arrived_window,
            draft_routed_y_window, draft_combine_window, draft_final_norm_weight[rank],
            draft_lm_head_weight[rank],
            draft_markov_w1[rank], draft_markov_w2[rank],
            draft_confidence_head_weight[rank], draft_draft_token_ids[rank],
            draft_confidence_probs[rank], draft_markov_hidden_window,
            draft_markov_hidden_done, draft_markov_logits_window, draft_markov_logits_done,
            draft_bridge_metadata_window, draft_bridge_metadata_signal,
            draft_bridge_rope_cos_window, draft_bridge_rope_sin_window,
            draft_bridge_rope_cos_signal, draft_bridge_rope_sin_signal,
            device=rank,
        )


# Fixture geometry.  The group tables are indexed by logical page, so their
# widths follow decode_prepare's own bounds; the RoPE tables only need to cover
# the fixture's position limit.
FIXTURE_ACTIVE_REQUESTS = 2
FIXTURE_POSITION_LIMIT = 512
FIXTURE_ROPE_ROWS = 2048
FIXTURE_ORI_TABLE_BLOCKS = (FIXTURE_POSITION_LIMIT + SAMPLED_IDS_PAD + BLOCK_SIZE - 1) // BLOCK_SIZE


def _fixture_dyn_bindings(param_specs):
    """Read every dynamic extent off the specs the component builders supply."""
    import inspect

    bindings = {}
    for param in inspect.signature(l3_decode_fwd_dspark._func).parameters.values():
        spec = param_specs.get(param.name)
        if spec is None:
            continue
        for annotated, concrete in zip(param.annotation.shape, spec.shape):
            if type(annotated).__name__ == "DynVar":
                bindings.setdefault(str(annotated), int(concrete))
    bindings.setdefault("DynVar('DSPARK_PREPARE_ORI_TABLE_BLOCKS_DYN')", FIXTURE_ORI_TABLE_BLOCKS)
    bindings.setdefault("DynVar('DSPARK_PREPARE_HCA_GROUP_STATE_BLOCKS_DYN')", prepare.HCA_STATE_TABLE_BLOCKS)
    bindings.setdefault("DynVar('DSPARK_PREPARE_CSA_GROUP_STATE_BLOCKS_DYN')", prepare.CSA_STATE_TABLE_BLOCKS)
    bindings.setdefault("DynVar('DSPARK_PREPARE_ROPE_ROWS_DYN')", FIXTURE_ROPE_ROWS)
    return bindings


def _fixture_device_state():
    """Seed FIXTURE_ACTIVE_REQUESTS verified requests per rank, rest padding."""
    import torch

    active = FIXTURE_ACTIVE_REQUESTS
    slot_ids = torch.full((N_RANKS, DSPARK_STATE_LOCAL_BATCH), -1, dtype=torch.int32)
    generations = torch.zeros((N_RANKS, DSPARK_STATE_LOCAL_BATCH), dtype=torch.int32)
    group_slot_ids = torch.full((N_RANKS, DECODE_BATCH), -1, dtype=torch.int32)
    group_generations = torch.zeros((N_RANKS, DECODE_BATCH), dtype=torch.int32)
    tokens = torch.zeros((N_RANKS, DSPARK_STATE_CAPACITY, DSPARK_STATE_TOKEN_WIDTH), dtype=torch.int64)
    meta = torch.zeros((N_RANKS, DSPARK_STATE_CAPACITY, DSPARK_STATE_META_WIDTH), dtype=torch.int32)
    for slot in range(active):
        slot_ids[:, slot] = slot
        generations[:, slot] = 1
        group_slot_ids[:, slot] = slot
        group_generations[:, slot] = 1
        tokens[:, slot, :] = torch.arange(100 * (slot + 1), 100 * (slot + 1) + DSPARK_STATE_TOKEN_WIDTH)
        meta[:, slot, prepare.STATE_VALID] = 1
        meta[:, slot, prepare.STATE_GENERATION] = 1
        meta[:, slot, prepare.STATE_ANCHOR_POSITION] = 64 * (slot + 1)
        meta[:, slot, prepare.STATE_DRAFT_COUNT] = DRAFT_DSPARK_QUERY_WIDTH
        meta[:, slot, prepare.STATE_POSITION_LIMIT] = FIXTURE_POSITION_LIMIT
    return slot_ids, generations, group_slot_ids, group_generations, tokens, meta


def build_tensor_specs():
    """Assemble the fused L2's fixture from the component builders it inlines."""
    import inspect

    import torch
    from golden import TensorSpec

    def rename(name, source):
        copied = TensorSpec(name, list(source.shape), source.dtype, init_value=source.init_value)
        copied.resident = source.resident
        return copied

    specs = {spec.name: spec for spec in decode.build_tensor_specs()}
    for spec in dspark.build_tensor_specs(dspark.DSPARK_MAX_BATCH):
        specs.setdefault("draft_" + spec.name, rename("draft_" + spec.name, spec))
    for spec in markov.build_tensor_specs(dspark.DSPARK_MAX_BATCH, distributed=True):
        specs.setdefault("draft_" + spec.name, rename("draft_" + spec.name, spec))

    bindings = _fixture_dyn_bindings(specs)
    slot_ids, generations, group_slot_ids, group_generations, tokens, meta = _fixture_device_state()
    seeded = {
        "state_slot_ids": slot_ids,
        "state_generations": generations,
        "group_state_slot_ids": group_slot_ids,
        "group_state_generations": group_generations,
        "state_tokens": tokens,
        "state_meta": meta,
        "valid_draft_counts": torch.full((N_RANKS, DECODE_BATCH), DRAFT_DSPARK_QUERY_WIDTH, dtype=torch.int32),
    }

    def resolve(shape):
        return [bindings[str(dim)] if type(dim).__name__ == "DynVar" else int(dim) for dim in shape]

    torch_dtype = {
        "bfloat16": torch.bfloat16, "fp32": torch.float32, "float32": torch.float32,
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    }
    ordered = []
    for param in inspect.signature(l3_decode_fwd_dspark._func).parameters.values():
        spec = specs.get(param.name)
        if spec is None:
            shape = resolve(param.annotation.shape)
            dtype = torch_dtype[str(param.annotation.dtype)]
            value = seeded.get(param.name)
            if value is None and "block_table" in param.name:
                # every request maps its own pages onto the same physical window: the
                # caches are sized for one request's pages, and this harness has no
                # golden, so the overlap costs nothing and stays in range
                pages = torch.arange(shape[1] * shape[2], dtype=torch.int32) % shape[2]
                value = pages.reshape(1, shape[1], shape[2]).expand(shape[0], -1, -1).contiguous()
            elif value is None and "rope" in param.name:
                angle = torch.arange(shape[1], dtype=torch.float32).unsqueeze(1) * 1e-4
                trig = torch.cos(angle) if "cos" in param.name else torch.sin(angle)
                value = trig.expand(-1, shape[2]).to(torch.bfloat16).expand(shape[0], -1, -1).contiguous()
            elif param.name == "grammar_mask":
                value = torch.full(shape, -1, dtype=dtype)
            elif value is None:
                value = torch.full(shape, -1 if "token_ids" in param.name else 0, dtype=dtype)
            spec = TensorSpec(param.name, shape, dtype, init_value=value)
        ordered.append(spec)
    return ordered


def main():
    import argparse

    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek-V4 D-Spark fused decode, drafter, and sampler")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=(2, 4))
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=(2, 4, 8, 16))
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; EP={EP_SIZE} needs {EP_SIZE}",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    if args.tp != TP_SIZE or args.ep != EP_SIZE:
        parser.error(f"parallel sizes froze at import as TP={TP_SIZE}, EP={EP_SIZE}")

    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(EP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != EP_SIZE:
        parser.error(f"EP={EP_SIZE} needs exactly {EP_SIZE} devices, got {device_ids}")
    if len(set(device_ids)) != len(device_ids) or any(device < 0 for device in device_ids):
        parser.error(f"device IDs must be distinct and non-negative: {device_ids}")

    result = run(
        fn=l3_decode_fwd_dspark,
        specs=build_tensor_specs(),
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            platform=args.platform,
            enable_scope_stats=args.enable_scope_stats,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
            ring_heap=DSPARK_RING_HEAP,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
