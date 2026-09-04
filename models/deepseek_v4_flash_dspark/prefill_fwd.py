# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 Flash DSpark 43-layer layer-major DSA-CP prefill forward with LM head and greedy sampling."""

import argparse
import os

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run
from pypto.ir.distributed_compiled_program import DistributedConfig

from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_prefill_moe_signals,
    prefill_moe,
)
from config import FLASH as MODEL_CONFIG
from prefill_swa import (
    build_cp_tensor_specs as build_swa_attention_tensor_specs,
    build_ragged2_cp_tensor_specs as build_swa_ragged2_tensor_specs,
    prefill_attention_swa_cp,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_CMP_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    SPARSE_CMP_MAX_BLOCKS as HCA_CMP_MAX_BLOCKS,
    build_cp_tensor_specs as build_hca_attention_tensor_specs,
    build_ragged2_cp_tensor_specs as build_hca_ragged2_tensor_specs,
    prefill_attention_hca_cp,
)
from prefill_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS as CSA_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    START_POS,
    build_cp_tensor_specs as build_csa_attention_tensor_specs,
    build_ragged2_cp_tensor_specs as build_csa_ragged2_tensor_specs,
    prefill_attention_csa_cp,
)
from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
)
from prefill_metadata import QUERY_START_LOC_DYN, REQUESTS_DYN, lower_local_request_ids
from prefill_o_proj import (
    O_PROJ_LOCAL_COLS,
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_COLS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    retire_o_proj_weight_signals,
)
from hc_head import hc_head
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    greedy_sample,
    lm_head,
)
from rmsnorm import rms_norm


# Dynamic shape variables.
FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
FWD_GROUP_TOKENS_DYN = pl.dynamic("PREFILL_FWD_GROUP_TOKENS_DYN")
FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
# Layer schedule: SWA 0-1, CSA even 2-42, and HCA odd 3-41.

# Runtime ring heaps by scope depth.
PREFILL_RING_HEAP = (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
LM_HEAD_COMM_EPOCH = 1

if MODEL_NUM_LAYERS != FWD_NUM_LAYERS:
    raise ValueError("DeepSeek-V4 Flash hidden layer count changed")
if N_RANKS % TP_SIZE:
    raise ValueError(f"EP world size {N_RANKS} must be divisible by CP group size {TP_SIZE}")
if LM_HEAD_TP_SIZE != TP_SIZE:
    raise ValueError(f"LM-head TP={LM_HEAD_TP_SIZE} does not match prefill TP={TP_SIZE}")
if LM_HEAD_VOCAB != MODEL_CONFIG.vocab_size:
    raise ValueError(f"LM-head vocab={LM_HEAD_VOCAB} does not match model vocab={MODEL_CONFIG.vocab_size}")
if MODEL_CONFIG.vocab_size % TP_SIZE:
    raise ValueError(f"vocab size {MODEL_CONFIG.vocab_size} must be divisible by TP={TP_SIZE}")

# FWD-layer stacked tensors, indexed by layer 0-42.
FWD_LAYER_STACKED_NAMES = [
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "hca_cmp_kv", "csa_cmp_kv",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
]
# MoE tensors flattened along the layer-major first axis.
MOE_LAYER_STACKED_NAMES = [
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
]
# CSA tensors indexed by CSA order 0-20.
CSA_LAYER_STACKED_NAMES = [
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
]
# HCA tensors indexed by HCA order 0-19.
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_cmp_kv",
]
# Per-rank tensors shared by every layer.
SHARED_NAMES = [
    "swa_freqs_cos", "swa_freqs_sin",
    "compressed_freqs_cos", "compressed_freqs_sin",
    "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
    "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
    "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "ori_slot_mapping_full", "position_ids_local", "position_ids_full", "input_ids",
    "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
    "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
    "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
]

# Mutable KV and compressor-state pools.
CACHE_NAMES = {
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
}

# Rank-sharded resident weights.
RESIDENT_WEIGHT_NAMES = frozenset(
    [
        n
        for n in (*FWD_LAYER_STACKED_NAMES, *CSA_LAYER_STACKED_NAMES, *HCA_LAYER_STACKED_NAMES)
        if n not in CACHE_NAMES
    ]
)

# Resident attention tensors selected by layer inside each child.
ATTENTION_RESIDENT_NAMES = RESIDENT_WEIGHT_NAMES.difference(MOE_LAYER_STACKED_NAMES)
ATTENTION_LAYER_STACKED_NAMES = ATTENTION_RESIDENT_NAMES
FLATTENED_LAYER_STACKED_NAMES = frozenset(MOE_LAYER_STACKED_NAMES).union(ATTENTION_LAYER_STACKED_NAMES, CACHE_NAMES)

# Rank-sharded resident caches.
RESIDENT_CACHE_NAMES = frozenset(CACHE_NAMES)

# Caches returned to the following decode invocation.
RESIDENT_CACHE_OUTPUT_NAMES = RESIDENT_CACHE_NAMES


@pl.jit.inline
def mask_inactive_sample_rows(
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Mark sampled rows without a live logit row with -1."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="prefill_fwd_sample_mask"):
        if pl.read(logit_row_indices, [row]) < 0:
            sampled_ids[row : row + 1, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=-1)
    return sampled_ids


@pl.jit(auto_scope=False)
def prefill_fwd(
    x_hc: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    swa_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[REQUESTS_DYN, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[REQUESTS_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    o_proj_wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    o_proj_wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    attn_stage: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    hidden_workspace: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    x_out: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS * LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the DeepSeek-V4 prefill backbone, LM head, and sampling."""
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    hca_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    idx_block_table.bind_dynamic(0, REQUESTS_DYN)
    swa_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE
    local_tokens = pl.tensor.dim(position_ids_local, 0)
    local_request_ids = pl.create_tensor([local_tokens], dtype=pl.INT32)
    lower_local_request_ids(query_start_loc, local_request_ids, tp_rank * local_tokens)
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_cmp_block_num = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    csa_cmp_block_num = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    hca_state_block_num = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_state_block_num = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    inner_state_block_num = pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS

    stage_token = pl.create_tensor([1], dtype=pl.INT32)
    layer_completion = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_order_state_init"):
        pl.write(stage_token, [0], pl.cast(0, pl.INT32))
        pl.write(layer_completion, [0], pl.cast(0, pl.INT32))

    # Layer 0: SWA.
    with pl.scope():
        layer_l0 = pl.const(0, pl.INT32)
        mix_start_l0 = layer_l0 * MIX_HC
        scale_start_l0 = layer_l0 * 3
        d_start_l0 = layer_l0 * D
        q_lora_start_l0 = layer_l0 * Q_LORA
        q_head_start_l0 = layer_l0 * H * HEAD_DIM
        head_dim_start_l0 = layer_l0 * HEAD_DIM
        attn_head_start_l0 = layer_l0 * H
        o_proj_group_start_l0 = layer_l0 * O_PROJ_LOCAL_GROUPS
        expert_start_l0 = layer_l0 * N_EXPERTS_GLOBAL
        vocab_start_l0 = layer_l0 * VOCAB
        local_expert_start_l0 = layer_l0 * N_LOCAL
        moe_start_l0 = layer_l0 * MOE_INTER

        hc_attn_fn_l0 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_l0, 0])
        hc_attn_scale_l0 = pl.slice(hc_attn_scale, [3], [scale_start_l0])
        hc_attn_base_l0 = pl.slice(hc_attn_base, [MIX_HC], [mix_start_l0])
        attn_norm_w_l0 = pl.slice(attn_norm_w, [D], [d_start_l0])
        wq_a_l0 = pl.slice(wq_a, [D, Q_LORA], [d_start_l0, 0])
        wq_b_l0 = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start_l0, 0])
        wq_b_scale_l0 = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_l0])
        wkv_l0 = pl.slice(wkv, [D, HEAD_DIM], [d_start_l0, 0])
        gamma_cq_l0 = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_l0])
        gamma_ckv_l0 = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_l0])
        kv_cache_l0 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_l0 * ori_block_num, 0, 0, 0])
        attn_sink_l0 = pl.slice(attn_sink, [H], [attn_head_start_l0])
        wo_a_l0 = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_l0, 0, 0])
        wo_b_l0 = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start_l0, 0])
        wo_b_scale_l0 = pl.slice(wo_b_scale, [D], [d_start_l0])

        hc_ffn_fn_l0 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_l0, 0])
        hc_ffn_scale_l0 = pl.slice(hc_ffn_scale, [3], [scale_start_l0])
        hc_ffn_base_l0 = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_l0])
        norm_w_l0 = pl.slice(norm_w, [D], [d_start_l0])
        gate_w_l0 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_l0, 0])
        gate_bias_l0 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_l0])
        tid2eid_l0 = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_l0, 0])
        routed_w1_l0 = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_l0, 0, 0])
        routed_w1_scale_l0 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l0, 0])
        routed_w3_l0 = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_l0, 0, 0])
        routed_w3_scale_l0 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l0, 0])
        routed_w2_l0 = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_l0, 0, 0])
        routed_w2_scale_l0 = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_l0, 0])
        shared_w1_l0 = pl.slice(shared_w1, [MOE_INTER, D], [moe_start_l0, 0])
        shared_w1_scale_l0 = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start_l0])
        shared_w3_l0 = pl.slice(shared_w3, [MOE_INTER, D], [moe_start_l0, 0])
        shared_w3_scale_l0 = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start_l0])
        shared_w2_l0 = pl.slice(shared_w2, [D, MOE_INTER], [d_start_l0, 0])
        shared_w2_scale_l0 = pl.slice(shared_w2_scale, [D], [d_start_l0])

        with pl.scope():
            kv_cache_l0, attn_stage, gather_signal = prefill_attention_swa_cp(
                x_hc,
                hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0,
                attn_norm_w_l0, wq_a_l0, wq_b_l0, wq_b_scale_l0,
                wkv_l0, gamma_cq_l0, gamma_ckv_l0,
                swa_freqs_cos, swa_freqs_sin,
                kv_cache_l0, ori_block_table, ori_slot_mapping_full,
                position_ids_local, position_ids_full, local_request_ids,
                attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
                o_proj_wo_a_full, o_proj_wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                layer_completion,
                group_base, tp_rank, layer_l0 + pl.const(1, pl.INT32),
            )

        with pl.scope():
            prefill_moe(
                attn_stage,
                hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
                norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0, input_ids,
                routed_w1_l0, routed_w1_scale_l0, routed_w3_l0, routed_w3_scale_l0,
                routed_w2_l0, routed_w2_scale_l0,
                shared_w1_l0, shared_w1_scale_l0, shared_w3_l0, shared_w3_scale_l0,
                shared_w2_l0, shared_w2_scale_l0,
                x_hc, x_mixed, post_ffn, comb_ffn, ffn_out,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                stage_done, stage_token, layer_completion,
                gather_window, gather_signal,
                group_base, tp_rank, layer_l0, my_rank,
            )

    # Layer 1: SWA.
    with pl.scope():
        layer_l1 = pl.const(1, pl.INT32)
        mix_start_l1 = layer_l1 * MIX_HC
        scale_start_l1 = layer_l1 * 3
        d_start_l1 = layer_l1 * D
        q_lora_start_l1 = layer_l1 * Q_LORA
        q_head_start_l1 = layer_l1 * H * HEAD_DIM
        head_dim_start_l1 = layer_l1 * HEAD_DIM
        attn_head_start_l1 = layer_l1 * H
        o_proj_group_start_l1 = layer_l1 * O_PROJ_LOCAL_GROUPS
        expert_start_l1 = layer_l1 * N_EXPERTS_GLOBAL
        vocab_start_l1 = layer_l1 * VOCAB
        local_expert_start_l1 = layer_l1 * N_LOCAL
        moe_start_l1 = layer_l1 * MOE_INTER

        hc_attn_fn_l1 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_l1, 0])
        hc_attn_scale_l1 = pl.slice(hc_attn_scale, [3], [scale_start_l1])
        hc_attn_base_l1 = pl.slice(hc_attn_base, [MIX_HC], [mix_start_l1])
        attn_norm_w_l1 = pl.slice(attn_norm_w, [D], [d_start_l1])
        wq_a_l1 = pl.slice(wq_a, [D, Q_LORA], [d_start_l1, 0])
        wq_b_l1 = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start_l1, 0])
        wq_b_scale_l1 = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_l1])
        wkv_l1 = pl.slice(wkv, [D, HEAD_DIM], [d_start_l1, 0])
        gamma_cq_l1 = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_l1])
        gamma_ckv_l1 = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_l1])
        kv_cache_l1 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_l1 * ori_block_num, 0, 0, 0])
        attn_sink_l1 = pl.slice(attn_sink, [H], [attn_head_start_l1])
        wo_a_l1 = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_l1, 0, 0])
        wo_b_l1 = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start_l1, 0])
        wo_b_scale_l1 = pl.slice(wo_b_scale, [D], [d_start_l1])

        hc_ffn_fn_l1 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_l1, 0])
        hc_ffn_scale_l1 = pl.slice(hc_ffn_scale, [3], [scale_start_l1])
        hc_ffn_base_l1 = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_l1])
        norm_w_l1 = pl.slice(norm_w, [D], [d_start_l1])
        gate_w_l1 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_l1, 0])
        gate_bias_l1 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_l1])
        tid2eid_l1 = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_l1, 0])
        routed_w1_l1 = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_l1, 0, 0])
        routed_w1_scale_l1 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l1, 0])
        routed_w3_l1 = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_l1, 0, 0])
        routed_w3_scale_l1 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l1, 0])
        routed_w2_l1 = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_l1, 0, 0])
        routed_w2_scale_l1 = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_l1, 0])
        shared_w1_l1 = pl.slice(shared_w1, [MOE_INTER, D], [moe_start_l1, 0])
        shared_w1_scale_l1 = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start_l1])
        shared_w3_l1 = pl.slice(shared_w3, [MOE_INTER, D], [moe_start_l1, 0])
        shared_w3_scale_l1 = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start_l1])
        shared_w2_l1 = pl.slice(shared_w2, [D, MOE_INTER], [d_start_l1, 0])
        shared_w2_scale_l1 = pl.slice(shared_w2_scale, [D], [d_start_l1])

        with pl.scope():
            kv_cache_l1, attn_stage, gather_signal = prefill_attention_swa_cp(
                x_hc,
                hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1,
                attn_norm_w_l1, wq_a_l1, wq_b_l1, wq_b_scale_l1,
                wkv_l1, gamma_cq_l1, gamma_ckv_l1,
                swa_freqs_cos, swa_freqs_sin,
                kv_cache_l1, ori_block_table, ori_slot_mapping_full,
                position_ids_local, position_ids_full, local_request_ids,
                attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
                o_proj_wo_a_full, o_proj_wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                layer_completion,
                group_base, tp_rank, layer_l1 + pl.const(1, pl.INT32),
            )

        with pl.scope():
            prefill_moe(
                attn_stage,
                hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
                norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1, input_ids,
                routed_w1_l1, routed_w1_scale_l1, routed_w3_l1, routed_w3_scale_l1,
                routed_w2_l1, routed_w2_scale_l1,
                shared_w1_l1, shared_w1_scale_l1, shared_w3_l1, shared_w3_scale_l1,
                shared_w2_l1, shared_w2_scale_l1,
                x_hc, x_mixed, post_ffn, comb_ffn, ffn_out,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                stage_done, stage_token, layer_completion,
                gather_window, gather_signal,
                group_base, tp_rank, layer_l1, my_rank,
            )

    # Layers 2-41: CSA/HCA pairs.
    for pair_order in pl.range(HCA_NUM_LAYERS):
        attention_order = pl.cast(pair_order, pl.INT32)

        with pl.scope():
            csa_layer = attention_order * 2 + pl.const(2, pl.INT32)
            mix_start_csa = csa_layer * MIX_HC
            scale_start_csa = csa_layer * 3
            d_start_csa = csa_layer * D
            q_lora_start_csa = csa_layer * Q_LORA
            q_head_start_csa = csa_layer * H * HEAD_DIM
            head_dim_start_csa = csa_layer * HEAD_DIM
            attn_head_start_csa = csa_layer * H
            o_proj_group_start_csa = csa_layer * O_PROJ_LOCAL_GROUPS
            expert_start_csa = csa_layer * N_EXPERTS_GLOBAL
            vocab_start_csa = csa_layer * VOCAB
            local_expert_start_csa = csa_layer * N_LOCAL
            moe_start_csa = csa_layer * MOE_INTER

            hc_attn_fn_csa = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_csa, 0])
            hc_attn_scale_csa = pl.slice(hc_attn_scale, [3], [scale_start_csa])
            hc_attn_base_csa = pl.slice(hc_attn_base, [MIX_HC], [mix_start_csa])
            attn_norm_w_csa = pl.slice(attn_norm_w, [D], [d_start_csa])
            wq_a_csa = pl.slice(wq_a, [D, Q_LORA], [d_start_csa, 0])
            wq_b_csa = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start_csa, 0])
            wq_b_scale_csa = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_csa])
            wkv_csa = pl.slice(wkv, [D, HEAD_DIM], [d_start_csa, 0])
            gamma_cq_csa = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_csa])
            gamma_ckv_csa = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_csa])
            kv_cache_csa = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * ori_block_num, 0, 0, 0])
            attn_sink_csa = pl.slice(attn_sink, [H], [attn_head_start_csa])
            wo_a_csa = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_csa, 0, 0])
            wo_b_csa = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start_csa, 0])
            wo_b_scale_csa = pl.slice(wo_b_scale, [D], [d_start_csa])

            hc_ffn_fn_csa = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_csa, 0])
            hc_ffn_scale_csa = pl.slice(hc_ffn_scale, [3], [scale_start_csa])
            hc_ffn_base_csa = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_csa])
            norm_w_csa = pl.slice(norm_w, [D], [d_start_csa])
            gate_w_csa = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_csa, 0])
            gate_bias_csa = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_csa])
            tid2eid_csa = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_csa, 0])
            routed_w1_csa = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_csa, 0, 0])
            routed_w1_scale_csa = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_csa, 0])
            routed_w3_csa = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_csa, 0, 0])
            routed_w3_scale_csa = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_csa, 0])
            routed_w2_csa = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_csa, 0, 0])
            routed_w2_scale_csa = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_csa, 0])
            shared_w1_csa = pl.slice(shared_w1, [MOE_INTER, D], [moe_start_csa, 0])
            shared_w1_scale_csa = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start_csa])
            shared_w3_csa = pl.slice(shared_w3, [MOE_INTER, D], [moe_start_csa, 0])
            shared_w3_scale_csa = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start_csa])
            shared_w2_csa = pl.slice(shared_w2, [D, MOE_INTER], [d_start_csa, 0])
            shared_w2_scale_csa = pl.slice(shared_w2_scale, [D], [d_start_csa])

            csa_compress_start_csa = attention_order * CSA_MAIN_OUT_DIM
            csa_ape_start_csa = attention_order * CSA_COMPRESS_RATIO
            csa_norm_start_csa = attention_order * HEAD_DIM
            csa_idx_head_start_csa = attention_order * IDX_HEAD_DIM
            csa_idx_q_lora_start_csa = attention_order * Q_LORA
            csa_idx_q_head_start_csa = attention_order * IDX_N_HEADS * IDX_HEAD_DIM
            csa_idx_proj_start_csa = attention_order * D
            csa_inner_start_csa = attention_order * INNER_OUT_DIM
            csa_state_start_csa = attention_order * csa_state_block_num
            csa_inner_state_start_csa = attention_order * inner_state_block_num
            csa_cmp_start_csa = attention_order * csa_cmp_block_num
            csa_idx_start_csa = attention_order * idx_block_num

            csa_cmp_wkv_csa = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_csa, 0])
            csa_cmp_wgate_csa = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_csa, 0])
            csa_cmp_ape_csa = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_ape_start_csa, 0])
            csa_cmp_norm_w_csa = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_norm_start_csa])
            csa_compress_state_csa = pl.slice(
                csa_compress_state,
                [csa_state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
                [csa_state_start_csa, 0, 0],
            )
            csa_hadamard_idx_csa = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_idx_head_start_csa, 0])
            csa_idx_wq_b_csa = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_lora_start_csa, 0])
            csa_idx_wq_b_scale_csa = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_head_start_csa])
            csa_weights_proj_csa = pl.slice(csa_weights_proj, [D, IDX_N_HEADS], [csa_idx_proj_start_csa, 0])
            csa_inner_wkv_csa = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [csa_inner_start_csa, 0])
            csa_inner_wgate_csa = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [csa_inner_start_csa, 0])
            csa_inner_ape_csa = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_ape_start_csa, 0])
            csa_inner_norm_w_csa = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_idx_head_start_csa])
            csa_inner_compress_state_csa = pl.slice(
                csa_inner_compress_state,
                [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
                [csa_inner_state_start_csa, 0, 0],
            )
            csa_cmp_kv_csa = pl.slice(
                csa_cmp_kv,
                [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [csa_cmp_start_csa, 0, 0, 0],
            )
            idx_kv_cache_csa = pl.slice(
                idx_kv_cache,
                [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
                [csa_idx_start_csa, 0, 0, 0],
            )
            idx_kv_scale_csa = pl.slice(
                idx_kv_scale,
                [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
                [csa_idx_start_csa, 0, 0, 0],
            )

            with pl.scope():
                attn_stage, gather_signal = prefill_attention_csa_cp(
                    x_hc,
                    query_start_loc,
                    hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa,
                    attn_norm_w_csa, wq_a_csa, wq_b_csa, wq_b_scale_csa,
                    wkv_csa, gamma_cq_csa, gamma_ckv_csa,
                    compressed_freqs_cos, compressed_freqs_sin,
                    csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                    csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                    csa_compress_state_csa, csa_compress_state_block_table,
                    csa_hadamard_idx_csa,
                    csa_idx_wq_b_csa, csa_idx_wq_b_scale_csa, csa_weights_proj_csa,
                    csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                    csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                    kv_cache_csa, ori_block_table, ori_slot_mapping_full,
                    csa_cmp_kv_csa, csa_cmp_block_table,
                    idx_kv_cache_csa, idx_kv_scale_csa, idx_block_table,
                    position_ids_local, position_ids_full, local_request_ids,
                    csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full,
                    csa_state_slot_mapping_full, csa_inner_state_slot_mapping_full,
                    attn_sink_csa, wo_a_csa, wo_b_csa, wo_b_scale_csa,
                    o_proj_wo_a_full, o_proj_wo_b_full,
                    attn_stage,
                    gather_window, gather_signal,
                    o_proj_wo_a_window, o_proj_wo_b_window,
                    o_proj_weight_ready, o_proj_weight_consumed,
                    layer_completion,
                    group_base, tp_rank, csa_layer + pl.const(1, pl.INT32),
                )

            with pl.scope():
                prefill_moe(
                    attn_stage,
                    hc_ffn_fn_csa, hc_ffn_scale_csa, hc_ffn_base_csa,
                    norm_w_csa, gate_w_csa, gate_bias_csa, tid2eid_csa, input_ids,
                    routed_w1_csa, routed_w1_scale_csa, routed_w3_csa, routed_w3_scale_csa,
                    routed_w2_csa, routed_w2_scale_csa,
                    shared_w1_csa, shared_w1_scale_csa, shared_w3_csa, shared_w3_scale_csa,
                    shared_w2_csa, shared_w2_scale_csa,
                    x_hc, x_mixed, post_ffn, comb_ffn, ffn_out,
                    recv_meta, recv_x, recv_aux, recv_route,
                    arrived, data_arrived, routed_y_buf, combine_arrived,
                    stage_done, stage_token, layer_completion,
                    gather_window, gather_signal,
                    group_base, tp_rank, csa_layer, my_rank,
                )

        with pl.scope():
            hca_layer = attention_order * 2 + pl.const(3, pl.INT32)
            mix_start_hca = hca_layer * MIX_HC
            scale_start_hca = hca_layer * 3
            d_start_hca = hca_layer * D
            q_lora_start_hca = hca_layer * Q_LORA
            q_head_start_hca = hca_layer * H * HEAD_DIM
            head_dim_start_hca = hca_layer * HEAD_DIM
            attn_head_start_hca = hca_layer * H
            o_proj_group_start_hca = hca_layer * O_PROJ_LOCAL_GROUPS
            expert_start_hca = hca_layer * N_EXPERTS_GLOBAL
            vocab_start_hca = hca_layer * VOCAB
            local_expert_start_hca = hca_layer * N_LOCAL
            moe_start_hca = hca_layer * MOE_INTER

            hc_attn_fn_hca = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_hca, 0])
            hc_attn_scale_hca = pl.slice(hc_attn_scale, [3], [scale_start_hca])
            hc_attn_base_hca = pl.slice(hc_attn_base, [MIX_HC], [mix_start_hca])
            attn_norm_w_hca = pl.slice(attn_norm_w, [D], [d_start_hca])
            wq_a_hca = pl.slice(wq_a, [D, Q_LORA], [d_start_hca, 0])
            wq_b_hca = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start_hca, 0])
            wq_b_scale_hca = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_hca])
            wkv_hca = pl.slice(wkv, [D, HEAD_DIM], [d_start_hca, 0])
            gamma_cq_hca = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_hca])
            gamma_ckv_hca = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_hca])
            kv_cache_hca = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * ori_block_num, 0, 0, 0])
            attn_sink_hca = pl.slice(attn_sink, [H], [attn_head_start_hca])
            wo_a_hca = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_hca, 0, 0])
            wo_b_hca = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start_hca, 0])
            wo_b_scale_hca = pl.slice(wo_b_scale, [D], [d_start_hca])

            hc_ffn_fn_hca = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_hca, 0])
            hc_ffn_scale_hca = pl.slice(hc_ffn_scale, [3], [scale_start_hca])
            hc_ffn_base_hca = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_hca])
            norm_w_hca = pl.slice(norm_w, [D], [d_start_hca])
            gate_w_hca = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_hca, 0])
            gate_bias_hca = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_hca])
            tid2eid_hca = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_hca, 0])
            routed_w1_hca = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_hca, 0, 0])
            routed_w1_scale_hca = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_hca, 0])
            routed_w3_hca = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_hca, 0, 0])
            routed_w3_scale_hca = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_hca, 0])
            routed_w2_hca = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_hca, 0, 0])
            routed_w2_scale_hca = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_hca, 0])
            shared_w1_hca = pl.slice(shared_w1, [MOE_INTER, D], [moe_start_hca, 0])
            shared_w1_scale_hca = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start_hca])
            shared_w3_hca = pl.slice(shared_w3, [MOE_INTER, D], [moe_start_hca, 0])
            shared_w3_scale_hca = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start_hca])
            shared_w2_hca = pl.slice(shared_w2, [D, MOE_INTER], [d_start_hca, 0])
            shared_w2_scale_hca = pl.slice(shared_w2_scale, [D], [d_start_hca])

            hca_compress_start_hca = attention_order * HCA_MAIN_OUT_DIM
            hca_ape_start_hca = attention_order * HCA_COMPRESS_RATIO
            hca_norm_start_hca = attention_order * HEAD_DIM
            hca_state_start_hca = attention_order * hca_state_block_num
            hca_cmp_start_hca = attention_order * hca_cmp_block_num
            hca_cmp_wkv_hca = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [hca_compress_start_hca, 0])
            hca_cmp_wgate_hca = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [hca_compress_start_hca, 0])
            hca_cmp_ape_hca = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [hca_ape_start_hca, 0])
            hca_cmp_norm_w_hca = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [hca_norm_start_hca])
            hca_compress_state_hca = pl.slice(
                hca_compress_state,
                [hca_state_block_num, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
                [hca_state_start_hca, 0, 0],
            )
            hca_cmp_kv_hca = pl.slice(
                hca_cmp_kv,
                [hca_cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [hca_cmp_start_hca, 0, 0, 0],
            )

            with pl.scope():
                attn_stage, gather_signal = prefill_attention_hca_cp(
                    x_hc,
                    query_start_loc, local_request_ids,
                    hc_attn_fn_hca, hc_attn_scale_hca, hc_attn_base_hca,
                    attn_norm_w_hca, wq_a_hca, wq_b_hca, wq_b_scale_hca,
                    wkv_hca, gamma_cq_hca, gamma_ckv_hca,
                    compressed_freqs_cos, compressed_freqs_sin,
                    hca_cmp_freqs_cos, hca_cmp_freqs_sin,
                    hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                    hca_compress_state_hca, hca_compress_state_block_table,
                    kv_cache_hca, ori_slot_mapping_full, ori_block_table,
                    hca_cmp_kv_hca, hca_cmp_block_table,
                    position_ids_local, position_ids_full,
                    hca_cmp_slot_mapping_full, hca_state_slot_mapping_full,
                    attn_sink_hca, wo_a_hca, wo_b_hca, wo_b_scale_hca,
                    o_proj_wo_a_full, o_proj_wo_b_full,
                    attn_stage,
                    gather_window, gather_signal,
                    o_proj_wo_a_window, o_proj_wo_b_window,
                    o_proj_weight_ready, o_proj_weight_consumed,
                    layer_completion,
                    group_base, tp_rank, hca_layer + pl.const(1, pl.INT32),
                )

            with pl.scope():
                prefill_moe(
                    attn_stage,
                    hc_ffn_fn_hca, hc_ffn_scale_hca, hc_ffn_base_hca,
                    norm_w_hca, gate_w_hca, gate_bias_hca, tid2eid_hca, input_ids,
                    routed_w1_hca, routed_w1_scale_hca, routed_w3_hca, routed_w3_scale_hca,
                    routed_w2_hca, routed_w2_scale_hca,
                    shared_w1_hca, shared_w1_scale_hca, shared_w3_hca, shared_w3_scale_hca,
                    shared_w2_hca, shared_w2_scale_hca,
                    x_hc, x_mixed, post_ffn, comb_ffn, ffn_out,
                    recv_meta, recv_x, recv_aux, recv_route,
                    arrived, data_arrived, routed_y_buf, combine_arrived,
                    stage_done, stage_token, layer_completion,
                    gather_window, gather_signal,
                    group_base, tp_rank, hca_layer, my_rank,
                )

    # Layer 42: CSA order 20.
    with pl.scope():
        layer_last = pl.const(42, pl.INT32)
        csa_order_last = pl.const(20, pl.INT32)
        mix_start_last = layer_last * MIX_HC
        scale_start_last = layer_last * 3
        d_start_last = layer_last * D
        q_lora_start_last = layer_last * Q_LORA
        q_head_start_last = layer_last * H * HEAD_DIM
        head_dim_start_last = layer_last * HEAD_DIM
        attn_head_start_last = layer_last * H
        o_proj_group_start_last = layer_last * O_PROJ_LOCAL_GROUPS
        expert_start_last = layer_last * N_EXPERTS_GLOBAL
        vocab_start_last = layer_last * VOCAB
        local_expert_start_last = layer_last * N_LOCAL
        moe_start_last = layer_last * MOE_INTER

        hc_attn_fn_last = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_last, 0])
        hc_attn_scale_last = pl.slice(hc_attn_scale, [3], [scale_start_last])
        hc_attn_base_last = pl.slice(hc_attn_base, [MIX_HC], [mix_start_last])
        attn_norm_w_last = pl.slice(attn_norm_w, [D], [d_start_last])
        wq_a_last = pl.slice(wq_a, [D, Q_LORA], [d_start_last, 0])
        wq_b_last = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start_last, 0])
        wq_b_scale_last = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_last])
        wkv_last = pl.slice(wkv, [D, HEAD_DIM], [d_start_last, 0])
        gamma_cq_last = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_last])
        gamma_ckv_last = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_last])
        kv_cache_last = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_last * ori_block_num, 0, 0, 0])
        attn_sink_last = pl.slice(attn_sink, [H], [attn_head_start_last])
        wo_a_last = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_last, 0, 0])
        wo_b_last = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start_last, 0])
        wo_b_scale_last = pl.slice(wo_b_scale, [D], [d_start_last])

        hc_ffn_fn_last = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_last, 0])
        hc_ffn_scale_last = pl.slice(hc_ffn_scale, [3], [scale_start_last])
        hc_ffn_base_last = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_last])
        norm_w_last = pl.slice(norm_w, [D], [d_start_last])
        gate_w_last = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_last, 0])
        gate_bias_last = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_last])
        tid2eid_last = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_last, 0])
        routed_w1_last = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_last, 0, 0])
        routed_w1_scale_last = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_last, 0])
        routed_w3_last = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_last, 0, 0])
        routed_w3_scale_last = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_last, 0])
        routed_w2_last = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_last, 0, 0])
        routed_w2_scale_last = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_last, 0])
        shared_w1_last = pl.slice(shared_w1, [MOE_INTER, D], [moe_start_last, 0])
        shared_w1_scale_last = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start_last])
        shared_w3_last = pl.slice(shared_w3, [MOE_INTER, D], [moe_start_last, 0])
        shared_w3_scale_last = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start_last])
        shared_w2_last = pl.slice(shared_w2, [D, MOE_INTER], [d_start_last, 0])
        shared_w2_scale_last = pl.slice(shared_w2_scale, [D], [d_start_last])

        csa_compress_start_last = csa_order_last * CSA_MAIN_OUT_DIM
        csa_ape_start_last = csa_order_last * CSA_COMPRESS_RATIO
        csa_norm_start_last = csa_order_last * HEAD_DIM
        csa_idx_head_start_last = csa_order_last * IDX_HEAD_DIM
        csa_idx_q_lora_start_last = csa_order_last * Q_LORA
        csa_idx_q_head_start_last = csa_order_last * IDX_N_HEADS * IDX_HEAD_DIM
        csa_idx_proj_start_last = csa_order_last * D
        csa_inner_start_last = csa_order_last * INNER_OUT_DIM
        csa_state_start_last = csa_order_last * csa_state_block_num
        csa_inner_state_start_last = csa_order_last * inner_state_block_num
        csa_cmp_start_last = csa_order_last * csa_cmp_block_num
        csa_idx_start_last = csa_order_last * idx_block_num

        csa_cmp_wkv_last = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_last, 0])
        csa_cmp_wgate_last = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_last, 0])
        csa_cmp_ape_last = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_ape_start_last, 0])
        csa_cmp_norm_w_last = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_norm_start_last])
        csa_compress_state_last = pl.slice(
            csa_compress_state,
            [csa_state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
            [csa_state_start_last, 0, 0],
        )
        csa_hadamard_idx_last = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_idx_head_start_last, 0])
        csa_idx_wq_b_last = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_lora_start_last, 0])
        csa_idx_wq_b_scale_last = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_head_start_last])
        csa_weights_proj_last = pl.slice(csa_weights_proj, [D, IDX_N_HEADS], [csa_idx_proj_start_last, 0])
        csa_inner_wkv_last = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [csa_inner_start_last, 0])
        csa_inner_wgate_last = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [csa_inner_start_last, 0])
        csa_inner_ape_last = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_ape_start_last, 0])
        csa_inner_norm_w_last = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_idx_head_start_last])
        csa_inner_compress_state_last = pl.slice(
            csa_inner_compress_state,
            [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
            [csa_inner_state_start_last, 0, 0],
        )
        csa_cmp_kv_last = pl.slice(
            csa_cmp_kv,
            [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            [csa_cmp_start_last, 0, 0, 0],
        )
        idx_kv_cache_last = pl.slice(
            idx_kv_cache,
            [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            [csa_idx_start_last, 0, 0, 0],
        )
        idx_kv_scale_last = pl.slice(
            idx_kv_scale,
            [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
            [csa_idx_start_last, 0, 0, 0],
        )

        with pl.scope():
            attn_stage, gather_signal = prefill_attention_csa_cp(
                x_hc,
                query_start_loc,
                hc_attn_fn_last, hc_attn_scale_last, hc_attn_base_last,
                attn_norm_w_last, wq_a_last, wq_b_last, wq_b_scale_last,
                wkv_last, gamma_cq_last, gamma_ckv_last,
                compressed_freqs_cos, compressed_freqs_sin,
                csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                csa_cmp_wkv_last, csa_cmp_wgate_last, csa_cmp_ape_last, csa_cmp_norm_w_last,
                csa_compress_state_last, csa_compress_state_block_table,
                csa_hadamard_idx_last,
                csa_idx_wq_b_last, csa_idx_wq_b_scale_last, csa_weights_proj_last,
                csa_inner_wkv_last, csa_inner_wgate_last, csa_inner_ape_last, csa_inner_norm_w_last,
                csa_inner_compress_state_last, csa_inner_compress_state_block_table,
                kv_cache_last, ori_block_table, ori_slot_mapping_full,
                csa_cmp_kv_last, csa_cmp_block_table,
                idx_kv_cache_last, idx_kv_scale_last, idx_block_table,
                position_ids_local, position_ids_full, local_request_ids,
                csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full,
                csa_state_slot_mapping_full, csa_inner_state_slot_mapping_full,
                attn_sink_last, wo_a_last, wo_b_last, wo_b_scale_last,
                o_proj_wo_a_full, o_proj_wo_b_full,
                attn_stage,
                gather_window, gather_signal,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                layer_completion,
                group_base, tp_rank, layer_last + pl.const(1, pl.INT32),
            )

        with pl.scope():
            prefill_moe(
                attn_stage,
                hc_ffn_fn_last, hc_ffn_scale_last, hc_ffn_base_last,
                norm_w_last, gate_w_last, gate_bias_last, tid2eid_last, input_ids,
                routed_w1_last, routed_w1_scale_last, routed_w3_last, routed_w3_scale_last,
                routed_w2_last, routed_w2_scale_last,
                shared_w1_last, shared_w1_scale_last, shared_w3_last, shared_w3_scale_last,
                shared_w2_last, shared_w2_scale_last,
                x_hc, x_mixed, post_ffn, comb_ffn, ffn_out,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                stage_done, stage_token, layer_completion,
                gather_window, gather_signal,
                group_base, tp_rank, layer_last, my_rank,
            )

    with pl.scope():
        clear_prefill_moe_signals(stage_token, arrived, data_arrived, combine_arrived, stage_done)
        retire_o_proj_weight_signals(
            layer_completion,
            o_proj_weight_ready, o_proj_weight_consumed,
            group_base, tp_rank, pl.const(43, pl.INT32),
        )

    # Final head over the gathered TP-group tokens: after the layer-42 token
    # gather x_hc holds every group token on each rank, and logit_row_indices
    # index that group token space.
    with pl.scope():
        hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, hidden_workspace)
        final_norm_tid = rms_norm(hidden_workspace, final_norm_w, x_out)
        lm_head(
            x_out, lm_head_weight, logit_row_indices, logits,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            group_base, tp_rank,
            pl.const(LM_HEAD_COMM_EPOCH, pl.INT32), final_norm_tid,
        )
        greedy_sample(logits, sampled_ids)
        mask_inactive_sample_rows(logit_row_indices, sampled_ids)
    return x_out


# DSA-CP layer-major multi-wave forward.
@pl.jit.host
def l3_prefill_fwd(
    x_hc: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    query_start_loc: pl.Tensor[[N_RANKS, QUERY_START_LOC_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    swa_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    o_proj_wo_a_full: pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    o_proj_wo_b_full: pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    attn_stage: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    hidden_workspace: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    x_out: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    """Run layer-major DSA-CP over a caller-padded physical token extent.

    Packed request boundaries are provided by a monotonic ``query_start_loc``
    that starts at zero and ends at total logical length ``N``. Its request
    count must match the leading dimension of every request-indexed block table,
    and metadata must be identical within a TP group. Each TP group supplies
    ``P = align_up(N, TP_SIZE)`` full rows and each rank supplies
    ``L = P // TP_SIZE`` local rows. Callers zero padded ``x_hc`` and
    ``input_ids``, use non-aliasing synthetic positions and ``-1`` cache/state
    mappings for padding, and restrict live ``logit_row_indices`` to ``[0, N)``.
    The device schedule, including MoE collectives, runs over physical P/L.
    """
    x_hc.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_START_LOC_DYN)
    hca_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    idx_block_table.bind_dynamic(1, REQUESTS_DYN)
    hidden_workspace.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    x_out.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    attn_stage.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ffn_out.bind_dynamic(1, FWD_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, FWD_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    stage_done_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS * LM_HEAD_VOCAB], dtype=pl.FP32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        stage_done = pld.window(stage_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16,
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8,
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS * LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        prefill_fwd(
            x_hc[r],
            query_start_loc[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r], wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r], gamma_cq[r], gamma_ckv[r],
            kv_cache[r], attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hca_cmp_kv[r], csa_cmp_kv[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r],
            hca_cmp_norm_w[r], hca_compress_state[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r],
            csa_cmp_norm_w[r], csa_compress_state[r],
            csa_hadamard_idx[r], csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r], csa_weights_proj[r],
            csa_inner_wkv[r], csa_inner_wgate[r],
            csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_compress_state[r], idx_kv_cache[r], idx_kv_scale[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            swa_freqs_cos[r], swa_freqs_sin[r],
            compressed_freqs_cos[r], compressed_freqs_sin[r],
            hca_cmp_freqs_cos[r], hca_cmp_freqs_sin[r],
            csa_cmp_freqs_cos[r], csa_cmp_freqs_sin[r],
            ori_block_table[r], hca_cmp_block_table[r],
            csa_cmp_block_table[r], idx_block_table[r],
            ori_slot_mapping_full[r], position_ids_local[r],
            position_ids_full[r], input_ids[r],
            hca_cmp_slot_mapping_full[r], hca_state_slot_mapping_full[r],
            csa_cmp_slot_mapping_full[r], csa_idx_slot_mapping_full[r],
            csa_state_slot_mapping_full[r], csa_inner_state_slot_mapping_full[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r],
            routed_w1[r], routed_w1_scale[r],
            routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r],
            shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            o_proj_wo_a_full[r], o_proj_wo_b_full[r],
            attn_stage[r], x_mixed[r],
            post_ffn[r], comb_ffn[r], ffn_out[r],
            hc_head_fn[r], hc_head_scale[r], hc_head_base[r],
            final_norm_w[r], lm_head_weight[r], logit_row_indices[r],
            hidden_workspace[r], x_out[r], logits[r], sampled_ids[r],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            stage_done,
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            r,
            device=r,
        )
    return x_out, logits, sampled_ids


# Kernel-only smoke fixtures.
def _layer_count(name):
    if name in CSA_LAYER_STACKED_NAMES:
        return CSA_NUM_LAYERS
    if name in HCA_LAYER_STACKED_NAMES:
        return HCA_NUM_LAYERS
    if name in FWD_LAYER_STACKED_NAMES:
        return FWD_NUM_LAYERS
    return 1


def _expand_rank_axis(value, torch):
    """Replicate one TP-group fixture across all EP ranks."""
    rank_count = value.shape[0]
    if rank_count == N_RANKS:
        return value.contiguous()
    if rank_count != TP_SIZE:
        raise ValueError(f"fixture rank axis must be TP={TP_SIZE} or EP={N_RANKS}, got {rank_count}")
    repeats = [N_RANKS // TP_SIZE, *([1] * (value.ndim - 1))]
    return value.repeat(*repeats).contiguous()


def _make_stacked_spec(name, base_specs, cache_block_nums=None):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    count = _layer_count(name)
    unit_shape = list(spec.shape[1:])
    if cache_block_nums and name in cache_block_nums:
        unit_shape[0] = cache_block_nums[name]
    flatten_layers = name in FLATTENED_LAYER_STACKED_NAMES
    if flatten_layers:
        packed_shape = [N_RANKS, count * unit_shape[0], *unit_shape[1:]]
    else:
        packed_shape = [N_RANKS, count, *unit_shape]

    def init_value():
        if cache_block_nums and name in cache_block_nums:
            return torch.zeros(packed_shape, dtype=spec.dtype)
        if name == "tid2eid":
            token_ids = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
            topk_ids = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
            rows = []
            for layer in range(count):
                rows.append((token_ids * TOPK + topk_ids + layer * TOPK) % N_EXPERTS_GLOBAL)
            packed = torch.cat(rows, dim=0)
            return packed.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
        layer_values = [_expand_rank_axis(_spec_value(spec, torch), torch) for _ in range(count)]
        if flatten_layers:
            return torch.cat(layer_values, dim=1)
        return torch.stack(layer_values, dim=1)

    # Mutable caches are fixture outputs.
    return TensorSpec(name, packed_shape, spec.dtype, init_value=init_value)


def _make_o_proj_tp_stacked_spec(name, base_specs):
    """Pack 43 resident output-projection layers in TP-sharded layout."""
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    if name == "wo_a":
        packed_shape = [N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN]
    elif name == "wo_b":
        packed_shape = [N_RANKS, FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS]
    else:
        raise ValueError(f"unsupported o-projection TP weight {name!r}")

    def init_value():
        packed = torch.empty(packed_shape, dtype=spec.dtype)
        for layer in range(FWD_NUM_LAYERS):
            full = _spec_value(spec, torch)
            source_ranks = full.shape[0]
            if source_ranks not in {TP_SIZE, N_RANKS}:
                raise ValueError(f"{name} fixture rank axis must be TP={TP_SIZE} or EP={N_RANKS}, got {source_ranks}")
            for rank in range(N_RANKS):
                source_rank = rank if source_ranks == N_RANKS else rank % TP_SIZE
                if name == "wo_a":
                    layer_start = layer * O_PROJ_LOCAL_GROUPS
                    target = packed[rank, layer_start : layer_start + O_PROJ_LOCAL_GROUPS]
                    source = full[source_rank]
                    target.copy_(source)
                else:
                    row_start = layer * D
                    target = packed[rank, row_start : row_start + D]
                    source = full[source_rank]
                    target.copy_(source)
        return packed

    return TensorSpec(name, packed_shape, spec.dtype, init_value=init_value)


def _make_shared_spec(name, base_specs):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]

    def init_value():
        return _expand_rank_axis(_spec_value(spec, torch), torch)

    return TensorSpec(name, [N_RANKS, *spec.shape[1:]], spec.dtype, init_value=init_value)


def _align_up(value, alignment):
    """Round a positive host-side extent up to the CP alignment."""
    return (value + alignment - 1) // alignment * alignment


def _global_token_index_map(local_tokens, torch):
    """Map each rank-local row to its TP-group prompt token."""
    local_row = torch.arange(local_tokens, dtype=torch.int64)
    rank_rows = [(rank % TP_SIZE) * local_tokens + local_row for rank in range(N_RANKS)]
    return torch.stack(rank_rows, dim=0).contiguous()


# Canonical host-tensor order for a single unified prefill layer.
HOST_TENSOR_ORDER = (
    "x_hc", "query_start_loc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "swa_freqs_cos", "swa_freqs_sin",
    "compressed_freqs_cos", "compressed_freqs_sin",
    "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
    "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w", "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w", "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache", "ori_block_table", "ori_slot_mapping_full",
    "hca_cmp_kv", "csa_cmp_kv", "hca_cmp_block_table", "csa_cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "position_ids_local", "position_ids_full",
    "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
    "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full", "csa_state_slot_mapping_full",
    "csa_inner_state_slot_mapping_full",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "x_next",
)


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def build_single_layer_tensor_specs(
    start_pos=START_POS,
    token_count=TP_SIZE * T,
    layer_id=2,
    fixture_case="b1",
):
    """Build the single-layer tensor specs used by the stacked forward fixtures."""
    import torch
    from golden import ScalarSpec, TensorSpec

    if fixture_case not in {"b1", "ragged2"}:
        raise ValueError(f"unsupported full-forward fixture case {fixture_case!r}")
    if fixture_case == "ragged2" and (TP_SIZE != 2 or token_count != 8):
        raise ValueError(f"ragged2 requires TP=2 and physical token_count=8, got TP={TP_SIZE}, tokens={token_count}")

    def kind_specs(build_fn, build_ragged_fn):
        source_specs = (
            build_ragged_fn(tp_size=TP_SIZE)
            if fixture_case == "ragged2"
            else build_fn(start_pos=start_pos, token_count=token_count, tp_size=TP_SIZE)
        )
        return {
            s.name: s
            for s in source_specs
            if isinstance(s, TensorSpec)
        }

    swa = kind_specs(build_swa_attention_tensor_specs, build_swa_ragged2_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs, build_hca_ragged2_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs, build_csa_ragged2_tensor_specs)
    active_kind = _attention_kind_for_layer(layer_id)
    active = {"swa": swa, "hca": hca, "csa": csa}[active_kind]
    active_tokens = token_count // TP_SIZE

    # Unified names and source specs for the selected attention kind.
    attention_specs = [
        ("x_hc", active["x_hc_full"]),
        ("query_start_loc", csa["query_start_loc"]),
        ("hc_attn_fn", active["hc_attn_fn"]), ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]), ("wq_b", active["wq_b"]), ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]), ("gamma_cq", active["gamma_cq"]), ("gamma_ckv", active["gamma_ckv"]),
        ("swa_freqs_cos", swa["freqs_cos"]), ("swa_freqs_sin", swa["freqs_sin"]),
        ("compressed_freqs_cos", hca["freqs_cos"]), ("compressed_freqs_sin", hca["freqs_sin"]),
        ("hca_cmp_freqs_cos", hca["cmp_freqs_cos"]), ("hca_cmp_freqs_sin", hca["cmp_freqs_sin"]),
        ("csa_cmp_freqs_cos", csa["cmp_freqs_cos"]), ("csa_cmp_freqs_sin", csa["cmp_freqs_sin"]),
        ("hca_cmp_wkv", hca["cmp_wkv"]), ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]), ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("hca_compress_state", hca["compress_state"]),
        ("hca_compress_state_block_table", hca["compress_state_block_table"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]), ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]), ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_compress_state", csa["compress_state"]),
        ("csa_compress_state_block_table", csa["compress_state_block_table"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]), ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]), ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]), ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]), ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("csa_inner_compress_state", csa["inner_compress_state"]),
        ("csa_inner_compress_state_block_table", csa["inner_compress_state_block_table"]),
        ("kv_cache", active["kv_cache"]),
        ("ori_block_table", active.get("ori_block_table", swa.get("block_table"))),
        ("ori_slot_mapping_full", active["ori_slot_mapping_full"]),
        ("hca_cmp_kv", hca["cmp_kv"]), ("csa_cmp_kv", csa["cmp_kv"]),
        ("hca_cmp_block_table", hca["cmp_block_table"]), ("csa_cmp_block_table", csa["cmp_block_table"]),
        ("idx_kv_cache", csa["idx_kv_cache"]), ("idx_kv_scale", csa["idx_kv_scale"]),
        ("idx_block_table", csa["idx_block_table"]),
        ("position_ids_local", active["position_ids_local"]), ("position_ids_full", active["position_ids_full"]),
        ("hca_cmp_slot_mapping_full", hca["cmp_slot_mapping_full"]),
        ("hca_state_slot_mapping_full", hca["state_slot_mapping_full"]),
        ("csa_cmp_slot_mapping_full", csa["cmp_slot_mapping_full"]),
        ("csa_idx_slot_mapping_full", csa["idx_slot_mapping_full"]),
        ("csa_state_slot_mapping_full", csa["state_slot_mapping_full"]),
        ("csa_inner_state_slot_mapping_full", csa["inner_state_slot_mapping_full"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]), ("wo_b", active["wo_b"]), ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [
        TensorSpec(name, list(src.shape), src.dtype, init_value=src.init_value)
        for name, src in attention_specs
    ]

    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        elif spec.name == "input_ids":
            input_ids_shape = list(spec.shape)
            if len(input_ids_shape) != 2 or input_ids_shape[0] != N_RANKS:
                raise ValueError(f"MoE input_ids must be [EP, T], got shape {spec.shape}")

            def init_input_ids(spec=spec):
                tokens = spec.shape[-1]
                active = min(active_tokens, tokens)
                rows = []
                for rank in range(N_RANKS):
                    row = torch.roll(torch.arange(tokens, dtype=spec.dtype), shifts=rank)
                    if layer_id >= 3 and active < tokens:
                        row[active:] = -1
                    rows.append(row)
                return torch.stack(rows, dim=0).contiguous()

            tensor_specs.append(TensorSpec(spec.name, input_ids_shape, spec.dtype, init_value=init_input_ids))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.float32))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("num_tokens", torch.int32, active_tokens),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def build_tensor_specs(
    start_pos=0,
    num_tokens=TP_SIZE * T,
    ori_block_num=CSA_ORI_BLOCK_NUM,
    hca_cmp_block_num=HCA_CMP_BLOCK_NUM,
    csa_cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=IDX_CACHE_BLOCK_NUM,
    hca_state_block_num=HCA_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_STATE_BLOCK_NUM,
    inner_state_block_num=INNER_STATE_BLOCK_NUM,
    fixture_case="b1",
):
    """Build CP-padded full-forward fixtures from a logical prompt length."""
    import torch
    from golden import TensorSpec

    if fixture_case not in {"b1", "ragged2"}:
        raise ValueError(f"unsupported full-forward fixture case {fixture_case!r}")
    if fixture_case == "ragged2" and TP_SIZE != 2:
        raise ValueError(f"ragged2 requires TP=2, got TP={TP_SIZE}")
    if fixture_case == "ragged2" and (start_pos != 0 or num_tokens != 7):
        raise ValueError(f"ragged2 uses fixed start_pos=0 and logical num_tokens=7, got {start_pos} and {num_tokens}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    capacities = {
        "ori_block_num": (ori_block_num, CSA_ORI_BLOCK_NUM),
        "hca_cmp_block_num": (hca_cmp_block_num, HCA_CMP_BLOCK_NUM),
        "csa_cmp_block_num": (csa_cmp_block_num, CSA_CMP_BLOCK_NUM),
        "idx_block_num": (idx_block_num, IDX_CACHE_BLOCK_NUM),
        "hca_state_block_num": (hca_state_block_num, HCA_STATE_BLOCK_NUM),
        "csa_state_block_num": (csa_state_block_num, CSA_STATE_BLOCK_NUM),
        "inner_state_block_num": (inner_state_block_num, INNER_STATE_BLOCK_NUM),
    }
    undersized = [
        f"{name}={value} (minimum {minimum})"
        for name, (value, minimum) in capacities.items()
        if value < minimum
    ]
    if undersized:
        raise ValueError(
            "custom cache/state pools cannot be smaller than the canonical "
            f"physical layout: {', '.join(undersized)}"
        )
    if num_tokens < 1 or num_tokens > PREFILL_GROUP_CAP:
        raise ValueError(f"num_tokens must be in [1, {PREFILL_GROUP_CAP}], got {num_tokens}")

    physical_tokens = _align_up(num_tokens, TP_SIZE)
    if start_pos + physical_tokens > MAX_SEQ_LEN:
        raise ValueError(
            "start_pos plus the CP-padded current chunk must fit the model context: "
            f"{start_pos} + {physical_tokens} > {MAX_SEQ_LEN}"
        )
    local_tokens = physical_tokens // TP_SIZE
    global_token_indices = _global_token_index_map(local_tokens, torch)
    expected_global_indices = torch.arange(physical_tokens, dtype=torch.int64)
    for group_base in range(0, N_RANKS, TP_SIZE):
        group_indices = global_token_indices[group_base : group_base + TP_SIZE]
        flat_group_indices = group_indices.reshape(-1)
        sorted_group_indices = torch.sort(flat_group_indices).values
        if not torch.equal(sorted_group_indices, expected_global_indices):
            raise ValueError(f"TP group at rank {group_base} is not a permutation of 0..{physical_tokens - 1}")
    fixture_seed = torch.initial_seed()

    base_specs = {
        spec.name: spec
        for spec in build_single_layer_tensor_specs(
            start_pos=start_pos,
            token_count=physical_tokens,
            layer_id=0,
            fixture_case=fixture_case,
        )
        if isinstance(spec, TensorSpec)
    }

    ordered_names = [
        "x_hc", "query_start_loc",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        "hca_cmp_kv", "csa_cmp_kv",
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "hca_compress_state",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_compress_state",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "swa_freqs_cos", "swa_freqs_sin",
        "compressed_freqs_cos", "compressed_freqs_sin",
        "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
        "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
        "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
        "ori_slot_mapping_full", "position_ids_local", "position_ids_full", "input_ids",
        "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
        "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
        "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ]

    cache_block_nums = {
        "kv_cache": ori_block_num,
        "hca_cmp_kv": hca_cmp_block_num,
        "csa_cmp_kv": csa_cmp_block_num,
        "idx_kv_cache": idx_block_num,
        "idx_kv_scale": idx_block_num,
        "hca_compress_state": hca_state_block_num,
        "csa_compress_state": csa_state_block_num,
        "csa_inner_compress_state": inner_state_block_num,
    }
    padded_mapping_names = {
        "ori_slot_mapping_full",
        "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
        "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
        "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
    }
    specs = []
    for name in ordered_names:
        if name == "query_start_loc":
            base = base_specs[name]
            if fixture_case == "ragged2":
                def init_query_start_loc(spec=base):
                    return _expand_rank_axis(_spec_value(spec, torch), torch)

                query_start_loc_shape = [N_RANKS, *base.shape[1:]]
            else:
                def init_query_start_loc(active_tokens=num_tokens, dtype=base.dtype):
                    boundaries = torch.tensor([0, active_tokens], dtype=dtype)
                    return boundaries.view(1, 2).expand(N_RANKS, -1).contiguous()

                query_start_loc_shape = [N_RANKS, 2]
            specs.append(TensorSpec(name, query_start_loc_shape, base.dtype, init_value=init_query_start_loc))
        elif name == "x_hc":
            base = base_specs[name]
            x_hc_shape = list(base.shape)
            x_hc_shape[0] = N_RANKS
            x_hc_shape[1] = physical_tokens

            def init_x_hc(tokens=physical_tokens, active_tokens=num_tokens, dtype=base.dtype, seed=fixture_seed):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                global_x = torch.randn(N_RANKS // TP_SIZE, tokens, HC_MULT, D, generator=generator, dtype=torch.float32)
                global_x[:, active_tokens:] = 0
                group_ids = torch.arange(N_RANKS, dtype=torch.int64) // TP_SIZE
                return (global_x[group_ids] * 0.05).to(dtype).contiguous()

            specs.append(TensorSpec(name, x_hc_shape, base.dtype, init_value=init_x_hc))
        elif name == "position_ids_local":
            if fixture_case == "ragged2":
                specs.append(_make_shared_spec(name, base_specs))
            else:
                dtype = base_specs[name].dtype

                def init_position_ids_local(indices=global_token_indices, dtype=dtype):
                    return (start_pos + indices).to(dtype).contiguous()

                specs.append(TensorSpec(name, [N_RANKS, local_tokens], dtype, init_value=init_position_ids_local))
        elif name == "input_ids":
            dtype = base_specs[name].dtype

            def init_input_ids(indices=global_token_indices, active_tokens=num_tokens, dtype=dtype):
                group_ids = torch.arange(N_RANKS, dtype=torch.int64) // TP_SIZE
                token_ids = group_ids[:, None] * physical_tokens + indices
                active_rows = indices < active_tokens
                return torch.where(active_rows, token_ids % VOCAB, 0).to(dtype).contiguous()

            specs.append(TensorSpec(name, [N_RANKS, local_tokens], dtype, init_value=init_input_ids))
        elif name in {"wo_a", "wo_b"}:
            specs.append(_make_o_proj_tp_stacked_spec(name, base_specs))
        elif name in padded_mapping_names:
            base_spec = _make_shared_spec(name, base_specs)

            def init_padded_mapping(spec=base_spec, active_tokens=num_tokens):
                value = _spec_value(spec, torch)
                value[:, active_tokens:] = -1
                return value.contiguous()

            specs.append(TensorSpec(name, list(base_spec.shape), base_spec.dtype, init_value=init_padded_mapping))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs))
        else:
            specs.append(_make_stacked_spec(name, base_specs, cache_block_nums))

    # Resident rank shards for weights and persistent KV/state pools.
    for spec in specs:
        if spec.name == "x_hc" or spec.name in RESIDENT_WEIGHT_NAMES or spec.name in RESIDENT_CACHE_NAMES:
            spec.resident = "stacked"

    stage_tokens = physical_tokens
    o_proj_scratch_specs = [
        TensorSpec(
            "o_proj_wo_a_full", [N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT],
            torch.bfloat16,
            init_value=lambda: torch.zeros(
                N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT,
                dtype=torch.bfloat16,
            ),
        ),
        TensorSpec(
            "o_proj_wo_b_full", [N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], torch.int8,
            init_value=lambda: torch.zeros(N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS, dtype=torch.int8),
        ),
    ]
    for spec in o_proj_scratch_specs:
        spec.resident = "stacked"
        specs.append(spec)
    attn_stage = TensorSpec(
        "attn_stage", [N_RANKS, stage_tokens, HC_MULT, D], torch.float32,
        init_value=lambda: torch.zeros(N_RANKS, stage_tokens, HC_MULT, D, dtype=torch.float32),
    )
    attn_stage.resident = "stacked"
    specs.append(attn_stage)
    moe_stage_specs = [
        TensorSpec(
            "x_mixed", [N_RANKS, stage_tokens, D], torch.bfloat16,
            init_value=lambda: torch.zeros(N_RANKS, stage_tokens, D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "post_ffn", [N_RANKS, stage_tokens, HC_MULT], torch.float32,
            init_value=lambda: torch.zeros(N_RANKS, stage_tokens, HC_MULT, dtype=torch.float32),
        ),
        TensorSpec(
            "comb_ffn", [N_RANKS, stage_tokens, HC_MULT * HC_MULT], torch.float32,
            init_value=lambda: torch.zeros(N_RANKS, stage_tokens, HC_MULT * HC_MULT, dtype=torch.float32),
        ),
        TensorSpec(
            "ffn_out", [N_RANKS, local_tokens, D], torch.bfloat16,
            init_value=lambda: torch.zeros(N_RANKS, local_tokens, D, dtype=torch.bfloat16),
        ),
    ]
    for spec in moe_stage_specs:
        spec.resident = "stacked"
        specs.append(spec)

    # Head and LM-head fixtures: one replicated draw per model weight, per-rank
    # LM-head vocabulary shards.
    def init_hc_head_fn():
        head_fn = torch.randn(HC_MULT, HC_DIM) * 0.0519
        return head_fn.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_hc_head_scale():
        return torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32)

    def init_hc_head_base():
        base = torch.tensor([5.9166, -3.6223, -2.9324, -3.3124], dtype=torch.float32)
        return base.view(1, HC_MULT).expand(N_RANKS, -1).contiguous()

    def init_final_norm_w():
        norm = (torch.randn(D) * 0.1 + 1.0).to(torch.bfloat16)
        return norm.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_lm_head_weight():
        shards = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D**0.5).to(torch.bfloat16)
        return torch.stack([shards[rank % TP_SIZE] for rank in range(N_RANKS)], dim=0)

    # Group leaders publish one last-token row per packed request.
    request_last_rows = (2, 6) if fixture_case == "ragged2" else (num_tokens - 1,)

    def init_logit_row_indices(last_rows=request_last_rows):
        indices = torch.full((N_RANKS, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[::TP_SIZE, : len(last_rows)] = torch.tensor(last_rows, dtype=torch.int32)
        return indices

    def init_hidden_workspace():
        return torch.zeros(N_RANKS, stage_tokens, D, dtype=torch.bfloat16)

    head_specs = [
        TensorSpec("hc_head_fn", [N_RANKS, HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [N_RANKS, 1], torch.float32, init_value=init_hc_head_scale),
        TensorSpec("hc_head_base", [N_RANKS, HC_MULT], torch.float32, init_value=init_hc_head_base),
        TensorSpec("final_norm_w", [N_RANKS, D], torch.bfloat16, init_value=init_final_norm_w),
        TensorSpec("lm_head_weight", [N_RANKS, VOCAB_PER_TP, D], torch.bfloat16, init_value=init_lm_head_weight),
        TensorSpec("logit_row_indices", [N_RANKS, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
        TensorSpec("hidden_workspace", [N_RANKS, stage_tokens, D], torch.bfloat16, init_value=init_hidden_workspace),
        TensorSpec("x_out", [N_RANKS, stage_tokens, D], torch.bfloat16),
        TensorSpec("logits", [N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], torch.float32),
        TensorSpec("sampled_ids", [N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32),
    ]
    for spec in head_specs:
        spec.resident = "stacked"
        specs.append(spec)

    spec_by_name = {spec.name: spec for spec in specs}
    request_count = spec_by_name["query_start_loc"].shape[1] - 1
    request_table_names = (
        "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
    )
    mismatched = [
        f"{name}={spec_by_name[name].shape[1]}"
        for name in request_table_names
        if spec_by_name[name].shape[1] != request_count
    ]
    if mismatched:
        raise ValueError(
            f"request-indexed table rows must match query_start_loc request count {request_count}: "
            f"{', '.join(mismatched)}"
        )
    return specs


def golden_prefill_fwd(_tensors):
    """Prefill forward is a topology/liveness witness; layer math is gated by prefill_layer."""


def finite_tensor_compare(actual, _expected, **_kwargs):
    """Require a completed finite device result without duplicating 43 goldens."""
    import torch

    if actual.numel() == 0:
        return False, "    prefill forward output is empty"
    if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
        return False, "    prefill forward output contains NaN or Inf"
    return True, ""


def x_out_compare(actual, _expected, **kwargs):
    """Recompute hc_head plus the final norm from the device x_hc and compare."""
    import torch
    from hc_head import golden_hc_head
    from rmsnorm import golden_rms_norm

    inputs = kwargs.get("inputs", {})
    device_x_hc = kwargs.get("actual_outputs", {}).get("x_hc")
    if device_x_hc is None:
        return False, "    missing device x_hc output"
    for rank in range(actual.shape[0]):
        head_out = torch.empty(device_x_hc.shape[1], D, dtype=torch.bfloat16)
        golden_hc_head({
            "x_hc": device_x_hc[rank].cpu(),
            "hc_head_fn": inputs["hc_head_fn"][rank],
            "hc_head_scale": inputs["hc_head_scale"][rank],
            "hc_head_base": inputs["hc_head_base"][rank],
            "y": head_out,
        })
        expected_rank = golden_rms_norm(head_out, inputs["final_norm_w"][rank]).float()
        actual_rank = actual[rank].float()
        if not bool(torch.isfinite(actual_rank).all()):
            return False, f"    rank {rank} x_out contains NaN or Inf"
        # bf16 point tolerance plus a per-token outlier bound: a fully corrupted
        # token row stays under the tensor-wide 0.5% budget.
        tolerance = 1e-4 + (1.0 / 128) * expected_rank.abs()
        bad = (actual_rank - expected_rank).abs() > tolerance
        ratio = float(bad.float().mean())
        if ratio > 0.005:
            return False, f"    rank {rank} head oracle mismatch: {ratio:.2%} points out of tolerance"
        row_bad = bad.float().mean(dim=1)
        worst_row = int(torch.argmax(row_bad))
        if float(row_bad[worst_row]) > 0.05:
            return False, f"    rank {rank} token {worst_row}: {float(row_bad[worst_row]):.1%} points out of tolerance"
    return True, ""


def logits_compare(actual, _expected, **kwargs):
    """Recompute every active logit row from the device x_out and the TP vocab shards."""
    import torch

    inputs = kwargs.get("inputs", {})
    device_x_out = kwargs.get("actual_outputs", {}).get("x_out")
    row_indices = inputs.get("logit_row_indices")
    weight = inputs.get("lm_head_weight")
    if device_x_out is None or row_indices is None or weight is None:
        return False, "    missing device x_out or LM-head inputs"
    if not bool(torch.isfinite(actual).all()):
        return False, "    logits contain NaN or Inf"
    for rank in range(actual.shape[0]):
        group_base = rank // TP_SIZE * TP_SIZE
        for row in range(MAX_LOGIT_ROWS):
            source = int(row_indices[rank, row])
            if source < 0:
                continue
            hidden_row = device_x_out[rank][source].float()
            # Shard tp owns vocabulary rows [tp * VOCAB_PER_TP, (tp + 1) * VOCAB_PER_TP).
            expected_row = torch.cat([weight[group_base + tp].float() @ hidden_row for tp in range(TP_SIZE)])
            actual_row = actual[rank, row].cpu()
            if not torch.allclose(actual_row, expected_row, rtol=1e-3, atol=1e-3):
                worst = float((actual_row - expected_row).abs().max())
                return False, f"    rank {rank} row {row} (token {source}) logits mismatch, max |err|={worst:.3e}"
    return True, ""


def sampled_ids_compare(actual, _expected, **kwargs):
    """Validate greedy ids against the device logits and the inactive-row -1 contract."""
    import torch

    row_indices = kwargs.get("inputs", {}).get("logit_row_indices")
    device_logits = kwargs.get("actual_outputs", {}).get("logits")
    if row_indices is None or device_logits is None:
        return False, "    missing logit_row_indices input or device logits output"
    active = row_indices >= 0
    inactive_values = actual.masked_select((~active).unsqueeze(-1).expand_as(actual))
    if inactive_values.numel() and not bool(torch.all(inactive_values == -1)):
        return False, "    inactive sampled-id rows are not -1"
    for rank in range(actual.shape[0]):
        for row in range(MAX_LOGIT_ROWS):
            if int(row_indices[rank, row]) < 0:
                continue
            sampled = int(actual[rank, row, 0])
            expected = int(torch.argmax(device_logits[rank][row]))
            # Both scans keep the first maximum, over identical fp32 data.
            if sampled != expected:
                return False, f"    rank {rank} row {row}: sampled id {sampled} != device logits argmax {expected}"
    return True, ""


def compare_functions():
    """Return the head oracles and finite-completion comparators for every output."""
    finite_names = {
        "x_hc", "attn_stage",
        "kv_cache", "hca_cmp_kv", "csa_cmp_kv",
        "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
        "idx_kv_cache", "idx_kv_scale",
    }
    compare = {name: finite_tensor_compare for name in finite_names}
    compare["x_out"] = x_out_compare
    compare["logits"] = logits_compare
    compare["sampled_ids"] = sampled_ids_compare
    return compare


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash DSA-CP 43-layer prefill-forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument(
        "--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16],
        help="EP world size / rank count (parsed at import by moe).",
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, choices=[1, 2, 4],
        help="CP group size (parsed at import by the CP attention leaves).",
    )
    default_devices = os.environ.get("TASK_DEVICE", ",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument(
        "-d", "--device", type=str, default=default_devices,
        help=f"comma-separated device ids; need at least {N_RANKS}",
    )
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument(
        "--num-tokens", type=int, default=None,
        help=f"B1 prompt tokens per TP/CP group; defaults to {TP_SIZE * T}. ragged2 is fixed at 7.",
    )
    parser.add_argument(
        "--case", choices=["b1", "ragged2"], default="b1",
        help="Fixture case; ragged2 is the fixed two-request TP2 boundary case.",
    )
    parser.add_argument("--ori-block-num", type=int, default=CSA_ORI_BLOCK_NUM)
    parser.add_argument("--hca-cmp-block-num", type=int, default=HCA_CMP_BLOCK_NUM)
    parser.add_argument("--csa-cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=IDX_CACHE_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=INNER_STATE_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=20260824, help="Torch seed for reproducible runner inputs and weights.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        parser.error(f"need at least {N_RANKS} devices, got {device_ids}")
    if TP_SIZE != args.tp:
        parser.error(f"import-time TP_SIZE must match --tp, got {TP_SIZE} vs {args.tp}")
    if N_RANKS != args.ep:
        parser.error(f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}")
    if args.ep % args.tp != 0:
        parser.error(f"EP must be divisible by TP/CP, got --ep {args.ep} and --tp {args.tp}")
    if args.case == "ragged2" and TP_SIZE != 2:
        parser.error("--case ragged2 requires --tp 2")
    if args.case == "ragged2" and args.start_pos != 0:
        parser.error("--case ragged2 has fixed request starts and requires --start-pos 0")
    if args.num_tokens is None:
        args.num_tokens = 7 if args.case == "ragged2" else TP_SIZE * T
    if args.case == "ragged2" and args.num_tokens != 7:
        parser.error("--case ragged2 has a fixed logical extent and requires --num-tokens 7")
    if args.num_tokens < 1 or args.num_tokens > PREFILL_GROUP_CAP:
        parser.error(f"--num-tokens must be in [1, {PREFILL_GROUP_CAP}]")

    import torch

    torch.manual_seed(args.seed)
    specs = build_tensor_specs(
        start_pos=args.start_pos, num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num,
        hca_cmp_block_num=args.hca_cmp_block_num, csa_cmp_block_num=args.csa_cmp_block_num,
        idx_block_num=args.idx_block_num,
        hca_state_block_num=args.hca_state_block_num, csa_state_block_num=args.csa_state_block_num,
        inner_state_block_num=args.inner_state_block_num,
        fixture_case=args.case,
    )

    result = run(
        fn=l3_prefill_fwd,
        specs=specs,
        golden_fn=golden_prefill_fwd,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            ring_heap=PREFILL_RING_HEAP,
        ),
        compare_fn=compare_functions(),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
