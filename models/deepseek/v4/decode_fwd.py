# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 Flash decode forward experiment with looped CSA/HCA layers inside a pl.jit function."""
# ruff: noqa: F403,F405

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run_jit
from hc_head import hc_head
from pypto.ir.distributed_compiled_program import DistributedConfig
from rmsnorm import rms_norm

# decode_fwd is self-contained: it imports kernels, constants, and per-kind
# spec builders directly from the leaf modules (no dependency on decode_layer).
# Import order matches decode_layer: attention kinds, then config, then moe.
from decode_attention_swa import (
    B,
    BLOCK_SIZE,
    D,
    HC_DIM,
    HC_MULT,
    H,
    HEAD_DIM,
    MAX_SEQ_LEN,
    MIX_HC,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    attention_swa,
    build_tensor_specs as build_attention_tensor_specs,
)
from decode_attention_hca import (
    CMP_BLOCK_NUM as HCA_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as HCA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_BLOCK_NUM as HCA_COMPRESS_STATE_BLOCK_NUM,
    COMPRESS_STATE_BLOCK_SIZE as HCA_COMPRESS_STATE_BLOCK_SIZE,
    COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM,
    COMPRESS_STATE_MAX_BLOCKS as HCA_COMPRESS_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    attention_hca,
    build_tensor_specs as build_hca_tensor_specs,
)
from decode_attention_csa import (
    CMP_BLOCK_NUM as CSA_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as CSA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    IDX_CACHE_BLOCK_NUM as CSA_IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS as CSA_IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM as CSA_IDX_HEAD_DIM,
    IDX_N_HEADS as CSA_IDX_N_HEADS,
    INNER_OUT_DIM as CSA_INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM as CSA_INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM as CSA_INNER_STATE_DIM,
    INNER_STATE_MAX_BLOCKS as CSA_INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_NUM as CSA_MAIN_STATE_BLOCK_NUM,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_DIM as CSA_MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS as CSA_MAIN_STATE_MAX_BLOCKS,
    attention_csa,
    build_tensor_specs as build_csa_tensor_specs,
)
from config import FLASH as MODEL_CONFIG
from moe import (
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
    W_PAD,
    build_tensor_specs as build_moe_tensor_specs,
    moe,
)

assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM, "unified host shares cmp_kv between HCA and CSA"
assert HCA_CMP_MAX_BLOCKS == CSA_CMP_MAX_BLOCKS, "unified host shares cmp_block_table between HCA and CSA"
from lm_head import (
    TP_SIZE as LM_HEAD_ACTIVE_TP_SIZE,
    T_MAX as LM_HEAD_T_MAX,
    VOCAB_PER_TP,
    lm_head_tp,
)

MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
# FWD index of the last layer (indexes per-FWD-layer stacked weights, not csa order).
FWD_LAST_LAYER = FWD_NUM_LAYERS - 1
assert MODEL_NUM_LAYERS == 43, "DeepSeek-V4 Flash hidden layer count changed"
assert N_RANKS == LM_HEAD_ACTIVE_TP_SIZE, "decode_fwd with lm_head currently supports --ep 2 / TP=2 only"

CSA_LAYER_STACKED_NAMES = [
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w", "csa_compress_state",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj", "csa_hadamard_idx",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "idx_kv_cache",
]
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w", "hca_compress_state",
]

LAYER_STACKED_NAMES = ['attn_norm_w', 'attn_sink', 'cmp_kv', 'csa_cmp_ape', 'csa_cmp_norm_w', 'csa_cmp_wgate', 'csa_cmp_wkv', 'csa_compress_state', 'csa_hadamard_idx', 'csa_idx_wq_b', 'csa_idx_wq_b_scale', 'csa_inner_ape', 'csa_inner_compress_state', 'csa_inner_norm_w', 'csa_inner_wgate', 'csa_inner_wkv', 'csa_weights_proj', 'gamma_ckv', 'gamma_cq', 'gate_bias', 'gate_w', 'hc_attn_base', 'hc_attn_fn', 'hc_attn_scale', 'hc_ffn_base', 'hc_ffn_fn', 'hc_ffn_scale', 'hca_cmp_ape', 'hca_cmp_norm_w', 'hca_cmp_wgate', 'hca_cmp_wkv', 'hca_compress_state', 'idx_kv_cache', 'kv_cache', 'norm_w', 'routed_w1', 'routed_w1_scale', 'routed_w2', 'routed_w2_scale', 'routed_w3', 'routed_w3_scale', 'shared_w1', 'shared_w1_scale', 'shared_w2', 'shared_w2_scale', 'shared_w3', 'shared_w3_scale', 'tid2eid', 'wkv', 'wo_a', 'wo_b', 'wo_b_scale', 'wq_a', 'wq_b', 'wq_b_scale']
SHARED_NAMES = ['x_hc', 'block_table', 'cmp_block_table', 'csa_cmp_slot_mapping', 'csa_compress_state_block_table', 'csa_idx_slot_mapping', 'csa_inner_compress_state_block_table', 'csa_inner_state_slot_mapping', 'csa_state_slot_mapping', 'freqs_cos', 'freqs_sin', 'hca_cmp_slot_mapping', 'hca_compress_state_block_table', 'hca_state_slot_mapping', 'idx_block_table', 'input_ids', 'kv_seq_lens', 'ori_slot_mapping', 'position_ids']
HC_HEAD_NAMES = ["hc_head_fn", "hc_head_scale", "hc_head_base"]
FINAL_NORM_NAMES = ["final_norm_w"]
LM_HEAD_NAMES = ["lm_head_weight"]

# Paged KV / compressor-state pools. These are sized for the full decode context
# (B x per-request-max-blocks per FWD layer, tens of thousands of blocks), so
# randn-ing every layer's pool independently dominates the "generate inputs"
# stage. Their content is smoke-only (decode_fwd has no golden_fn), so we randn a
# single layer's pool and tile it across layers instead: each layer keeps the
# full per-block diversity its standalone attention init produces (the indexer's
# top-k block selection still sees varied blocks) while the randn work drops
# ~layer_count x. Weights / gate / routing metadata are unaffected.
CACHE_POOL_NAMES = frozenset({
    "kv_cache", "cmp_kv", "idx_kv_cache",
    "csa_compress_state", "csa_inner_compress_state", "hca_compress_state",
})


@pl.jit(auto_scope=False)
def decode_fwd(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.InOut[pl.Tensor[[FWD_NUM_LAYERS * B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32],
    cmp_kv: pl.Tensor[[FWD_NUM_LAYERS * CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    x_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]:
    hc_attn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_attn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [0 * 3])
    hc_attn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    attn_norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [0 * D])
    wq_a_l0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [0 * D, 0])
    wq_b_l0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [0 * Q_LORA, 0])
    wq_b_scale_l0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [0 * H * HEAD_DIM])
    wkv_l0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [0 * D, 0])
    gamma_cq_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [0 * Q_LORA])
    gamma_ckv_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [0 * HEAD_DIM])
    kv_cache_l0: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [0 * B * ORI_MAX_BLOCKS, 0, 0, 0])
    attn_sink_l0: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [0 * H])
    wo_a_l0: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [0 * O_GROUPS, 0, 0])
    wo_b_l0: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [0 * D, 0])
    wo_b_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [0 * D])
    cmp_kv_l0: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [0 * CSA_CMP_BLOCK_NUM, 0, 0, 0])
    hc_ffn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_ffn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [0 * 3])
    hc_ffn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [0 * MIX_HC])
    norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [0 * D])
    gate_w_l0: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [0 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l0: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [0 * N_EXPERTS_GLOBAL])
    tid2eid_l0: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [0 * VOCAB, 0])
    routed_w1_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w1_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w3_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w3_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w2_l0: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [0 * N_LOCAL, 0, 0])
    routed_w2_scale_l0: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [0 * N_LOCAL, 0])
    shared_w1_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w1_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w3_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w3_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w2_l0: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [0 * D, 0])
    shared_w2_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [0 * D])
    hc_attn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_attn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [1 * 3])
    hc_attn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    attn_norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [1 * D])
    wq_a_l1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [1 * D, 0])
    wq_b_l1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [1 * Q_LORA, 0])
    wq_b_scale_l1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [1 * H * HEAD_DIM])
    wkv_l1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [1 * D, 0])
    gamma_cq_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [1 * Q_LORA])
    gamma_ckv_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [1 * HEAD_DIM])
    kv_cache_l1: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [1 * B * ORI_MAX_BLOCKS, 0, 0, 0])
    attn_sink_l1: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [1 * H])
    wo_a_l1: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [1 * O_GROUPS, 0, 0])
    wo_b_l1: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [1 * D, 0])
    wo_b_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [1 * D])
    cmp_kv_l1: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [1 * CSA_CMP_BLOCK_NUM, 0, 0, 0])
    hc_ffn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_ffn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [1 * 3])
    hc_ffn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [1 * MIX_HC])
    norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [1 * D])
    gate_w_l1: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [1 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l1: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [1 * N_EXPERTS_GLOBAL])
    tid2eid_l1: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [1 * VOCAB, 0])
    routed_w1_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w1_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w3_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w3_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w2_l1: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [1 * N_LOCAL, 0, 0])
    routed_w2_scale_l1: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [1 * N_LOCAL, 0])
    shared_w1_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w1_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w3_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w3_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w2_l1: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [1 * D, 0])
    shared_w2_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [1 * D])
    x_attn0: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_attn1: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    hidden: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    with pl.scope():
        attention_swa(
            x_hc,
            hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0,
            attn_norm_w_l0, wq_a_l0, wq_b_l0, wq_b_scale_l0,
            wkv_l0, gamma_cq_l0, gamma_ckv_l0, freqs_cos, freqs_sin,
            kv_cache_l0, block_table,
            ori_slot_mapping, position_ids,
            cmp_kv_l0, cmp_block_table,
            attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
            x_attn0,
        )
    with pl.scope():
        moe(
            x_attn0,
            hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
            norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0, input_ids,
            routed_w1_l0, routed_w1_scale_l0, routed_w3_l0, routed_w3_scale_l0,
            routed_w2_l0, routed_w2_scale_l0,
            shared_w1_l0, shared_w1_scale_l0, shared_w3_l0, shared_w3_scale_l0,
            shared_w2_l0, shared_w2_scale_l0,
            hidden,
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            pl.cast(0, pl.INT32), num_tokens, my_rank, pl.cast(1, pl.INT32),
        )
    with pl.scope():
        attention_swa(
            hidden,
            hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1,
            attn_norm_w_l1, wq_a_l1, wq_b_l1, wq_b_scale_l1,
            wkv_l1, gamma_cq_l1, gamma_ckv_l1, freqs_cos, freqs_sin,
            kv_cache_l1, block_table,
            ori_slot_mapping, position_ids,
            cmp_kv_l1, cmp_block_table,
            attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
            x_attn1,
        )
    with pl.scope():
        moe(
            x_attn1,
            hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
            norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1, input_ids,
            routed_w1_l1, routed_w1_scale_l1, routed_w3_l1, routed_w3_scale_l1,
            routed_w2_l1, routed_w2_scale_l1,
            shared_w1_l1, shared_w1_scale_l1, shared_w3_l1, shared_w3_scale_l1,
            shared_w2_l1, shared_w2_scale_l1,
            hidden,
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            pl.cast(1, pl.INT32), num_tokens, my_rank, pl.cast(2, pl.INT32),
        )
    for loop_i in pl.range(HCA_NUM_LAYERS):
        csa_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 2, pl.INT32)
        hca_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        csa_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        hca_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 4, pl.INT32)
        x_attn_csa: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        x_attn_hca: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        hidden_mid: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        hc_attn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_attn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer * 3])
        hc_attn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [csa_layer * MIX_HC])
        attn_norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer * D])
        wq_a_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer * D, 0])
        wq_b_csa: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer * Q_LORA, 0])
        wq_b_scale_csa: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_layer * H * HEAD_DIM])
        wkv_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer * D, 0])
        gamma_cq_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer * Q_LORA])
        gamma_ckv_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_layer * HEAD_DIM])
        kv_cache_csa: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * B * ORI_MAX_BLOCKS, 0, 0, 0])
        attn_sink_csa: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer * H])
        wo_a_csa: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer * O_GROUPS, 0, 0])
        wo_b_csa: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [csa_layer * D, 0])
        wo_b_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer * D])
        csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0])
        csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM])
        csa_compress_state_csa: pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32] = pl.slice(csa_compress_state, [CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [loop_i * CSA_MAIN_STATE_BLOCK_NUM, 0, 0])
        csa_idx_wq_b_csa: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [loop_i * Q_LORA, 0])
        csa_idx_wq_b_scale_csa: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale, [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM])
        csa_weights_proj_csa: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj, [D, CSA_IDX_N_HEADS], [loop_i * D, 0])
        csa_hadamard_idx_csa: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_HEAD_DIM, 0])
        csa_inner_wkv_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [loop_i * CSA_INNER_OUT_DIM, 0])
        csa_inner_wgate_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [loop_i * CSA_INNER_OUT_DIM, 0])
        csa_inner_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0])
        csa_inner_norm_w_csa: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_HEAD_DIM])
        csa_inner_compress_state_csa: pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32] = pl.slice(csa_inner_compress_state, [CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [loop_i * CSA_INNER_STATE_BLOCK_NUM, 0, 0])
        cmp_kv_csa: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * CSA_CMP_BLOCK_NUM, 0, 0, 0])
        idx_kv_cache_csa: pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(idx_kv_cache, [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_CACHE_BLOCK_NUM, 0, 0, 0])
        hc_ffn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_ffn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer * 3])
        hc_ffn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer * MIX_HC])
        norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer * D])
        gate_w_csa: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_csa: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_layer * N_EXPERTS_GLOBAL])
        tid2eid_csa: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [csa_layer * VOCAB, 0])
        routed_w1_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w1_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w3_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w3_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w2_csa: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_layer * N_LOCAL, 0, 0])
        routed_w2_scale_csa: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_layer * N_LOCAL, 0])
        shared_w1_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w1_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w3_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w3_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w2_csa: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [csa_layer * D, 0])
        shared_w2_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [csa_layer * D])
        with pl.scope():
            attention_csa(
                hidden,
                hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa,
                attn_norm_w_csa, wq_a_csa, wq_b_csa, wq_b_scale_csa,
                wkv_csa, gamma_cq_csa, gamma_ckv_csa, freqs_cos, freqs_sin,
                csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                csa_compress_state_csa, csa_compress_state_block_table,
                csa_idx_wq_b_csa, csa_idx_wq_b_scale_csa, csa_weights_proj_csa, csa_hadamard_idx_csa,
                csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                kv_cache_csa, block_table, cmp_kv_csa, cmp_block_table,
                idx_kv_cache_csa, idx_block_table,
                ori_slot_mapping, csa_cmp_slot_mapping, csa_idx_slot_mapping,
                csa_state_slot_mapping, csa_inner_state_slot_mapping,
                position_ids, kv_seq_lens,
                attn_sink_csa, wo_a_csa, wo_b_csa, wo_b_scale_csa,
                x_attn_csa,
            )
        with pl.scope():
            moe(
                x_attn_csa,
                hc_ffn_fn_csa, hc_ffn_scale_csa, hc_ffn_base_csa,
                norm_w_csa, gate_w_csa, gate_bias_csa, tid2eid_csa, input_ids,
                routed_w1_csa, routed_w1_scale_csa, routed_w3_csa, routed_w3_scale_csa,
                routed_w2_csa, routed_w2_scale_csa,
                shared_w1_csa, shared_w1_scale_csa, shared_w3_csa, shared_w3_scale_csa,
                shared_w2_csa, shared_w2_scale_csa,
                hidden_mid,
                pub_counts, count_done, data_done,
                recv_x, recv_scale, recv_w, recv_r_route,
                routed_y_buf, combine_done,
                csa_layer, num_tokens, my_rank, csa_moe_epoch,
            )
        hc_attn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_attn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [hca_layer * 3])
        hc_attn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [hca_layer * MIX_HC])
        attn_norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [hca_layer * D])
        wq_a_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [hca_layer * D, 0])
        wq_b_hca: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [hca_layer * Q_LORA, 0])
        wq_b_scale_hca: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [hca_layer * H * HEAD_DIM])
        wkv_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [hca_layer * D, 0])
        gamma_cq_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [hca_layer * Q_LORA])
        gamma_ckv_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [hca_layer * HEAD_DIM])
        kv_cache_hca: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * B * ORI_MAX_BLOCKS, 0, 0, 0])
        attn_sink_hca: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [hca_layer * H])
        wo_a_hca: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [hca_layer * O_GROUPS, 0, 0])
        wo_b_hca: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [hca_layer * D, 0])
        wo_b_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [hca_layer * D])
        hca_cmp_wkv_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0])
        hca_cmp_wgate_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0])
        hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [loop_i * HCA_COMPRESS_RATIO, 0])
        hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM])
        hca_compress_state_hca: pl.Tensor[[HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32] = pl.slice(hca_compress_state, [HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], [loop_i * HCA_COMPRESS_STATE_BLOCK_NUM, 0, 0])
        cmp_kv_hca: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * CSA_CMP_BLOCK_NUM, 0, 0, 0])
        hc_ffn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_ffn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [hca_layer * 3])
        hc_ffn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [hca_layer * MIX_HC])
        norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [hca_layer * D])
        gate_w_hca: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [hca_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_hca: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [hca_layer * N_EXPERTS_GLOBAL])
        tid2eid_hca: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [hca_layer * VOCAB, 0])
        routed_w1_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w1_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w3_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w3_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w2_hca: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [hca_layer * N_LOCAL, 0, 0])
        routed_w2_scale_hca: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [hca_layer * N_LOCAL, 0])
        shared_w1_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w1_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w3_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w3_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w2_hca: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [hca_layer * D, 0])
        shared_w2_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [hca_layer * D])
        with pl.scope():
            attention_hca(
                hidden_mid,
                hc_attn_fn_hca, hc_attn_scale_hca, hc_attn_base_hca,
                attn_norm_w_hca, wq_a_hca, wq_b_hca, wq_b_scale_hca,
                wkv_hca, gamma_cq_hca, gamma_ckv_hca, freqs_cos, freqs_sin,
                hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                hca_compress_state_hca, hca_compress_state_block_table,
                kv_cache_hca, block_table, cmp_kv_hca, cmp_block_table,
                ori_slot_mapping, hca_cmp_slot_mapping, hca_state_slot_mapping,
                position_ids, kv_seq_lens,
                attn_sink_hca, wo_a_hca, wo_b_hca, wo_b_scale_hca,
                x_attn_hca,
            )
        with pl.scope():
            moe(
                x_attn_hca,
                hc_ffn_fn_hca, hc_ffn_scale_hca, hc_ffn_base_hca,
                norm_w_hca, gate_w_hca, gate_bias_hca, tid2eid_hca, input_ids,
                routed_w1_hca, routed_w1_scale_hca, routed_w3_hca, routed_w3_scale_hca,
                routed_w2_hca, routed_w2_scale_hca,
                shared_w1_hca, shared_w1_scale_hca, shared_w3_hca, shared_w3_scale_hca,
                shared_w2_hca, shared_w2_scale_hca,
                hidden,
                pub_counts, count_done, data_done,
                recv_x, recv_scale, recv_w, recv_r_route,
                routed_y_buf, combine_done,
                hca_layer, num_tokens, my_rank, hca_moe_epoch,
            )
    # FWD index (14), not CSA_LAST_LAYER (6): the *_last slices below index per-FWD-layer
    # stacked weights; csa-stacked weights use the literal (CSA_NUM_LAYERS-1).
    csa_layer_last: pl.Scalar[pl.INT32] = pl.cast(FWD_LAST_LAYER, pl.INT32)
    last_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(2 * HCA_NUM_LAYERS + 3, pl.INT32)
    x_attn_last: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_next_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    hc_attn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0])
    hc_attn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer_last * 3])
    hc_attn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [csa_layer_last * MIX_HC])
    attn_norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer_last * D])
    wq_a_last: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer_last * D, 0])
    wq_b_last: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer_last * Q_LORA, 0])
    wq_b_scale_last: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_layer_last * H * HEAD_DIM])
    wkv_last: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer_last * D, 0])
    gamma_cq_last: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer_last * Q_LORA])
    gamma_ckv_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_layer_last * HEAD_DIM])
    kv_cache_last: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer_last * B * ORI_MAX_BLOCKS, 0, 0, 0])
    attn_sink_last: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer_last * H])
    wo_a_last: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer_last * O_GROUPS, 0, 0])
    wo_b_last: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [csa_layer_last * D, 0])
    wo_b_scale_last: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer_last * D])
    csa_cmp_wkv_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_MAIN_OUT_DIM, 0])
    csa_cmp_wgate_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_MAIN_OUT_DIM, 0])
    csa_cmp_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [(CSA_NUM_LAYERS - 1) * CSA_COMPRESS_RATIO, 0])
    csa_cmp_norm_w_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [(CSA_NUM_LAYERS - 1) * HEAD_DIM])
    csa_compress_state_last: pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32] = pl.slice(csa_compress_state, [CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [(CSA_NUM_LAYERS - 1) * CSA_MAIN_STATE_BLOCK_NUM, 0, 0])
    csa_idx_wq_b_last: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * Q_LORA, 0])
    csa_idx_wq_b_scale_last: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale, [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM])
    csa_weights_proj_last: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj, [D, CSA_IDX_N_HEADS], [(CSA_NUM_LAYERS - 1) * D, 0])
    csa_hadamard_idx_last: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_HEAD_DIM, 0])
    csa_inner_wkv_last: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_INNER_OUT_DIM, 0])
    csa_inner_wgate_last: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_INNER_OUT_DIM, 0])
    csa_inner_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [(CSA_NUM_LAYERS - 1) * CSA_COMPRESS_RATIO, 0])
    csa_inner_norm_w_last: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_HEAD_DIM])
    csa_inner_compress_state_last: pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32] = pl.slice(csa_inner_compress_state, [CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [(CSA_NUM_LAYERS - 1) * CSA_INNER_STATE_BLOCK_NUM, 0, 0])
    cmp_kv_last: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer_last * CSA_CMP_BLOCK_NUM, 0, 0, 0])
    idx_kv_cache_last: pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(idx_kv_cache, [CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_CACHE_BLOCK_NUM, 0, 0, 0])
    hc_ffn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0])
    hc_ffn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer_last * 3])
    hc_ffn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer_last * MIX_HC])
    norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer_last * D])
    gate_w_last: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_layer_last * N_EXPERTS_GLOBAL, 0])
    gate_bias_last: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_layer_last * N_EXPERTS_GLOBAL])
    tid2eid_last: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [csa_layer_last * VOCAB, 0])
    routed_w1_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w1_scale_last: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_layer_last * N_LOCAL, 0])
    routed_w3_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w3_scale_last: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_layer_last * N_LOCAL, 0])
    routed_w2_last: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w2_scale_last: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_layer_last * N_LOCAL, 0])
    shared_w1_last: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [csa_layer_last * MOE_INTER, 0])
    shared_w1_scale_last: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [csa_layer_last * MOE_INTER])
    shared_w3_last: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [csa_layer_last * MOE_INTER, 0])
    shared_w3_scale_last: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [csa_layer_last * MOE_INTER])
    shared_w2_last: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [csa_layer_last * D, 0])
    shared_w2_scale_last: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [csa_layer_last * D])
    with pl.scope():
        attention_csa(
            hidden,
            hc_attn_fn_last, hc_attn_scale_last, hc_attn_base_last,
            attn_norm_w_last, wq_a_last, wq_b_last, wq_b_scale_last,
            wkv_last, gamma_cq_last, gamma_ckv_last, freqs_cos, freqs_sin,
            csa_cmp_wkv_last, csa_cmp_wgate_last, csa_cmp_ape_last, csa_cmp_norm_w_last,
            csa_compress_state_last, csa_compress_state_block_table,
            csa_idx_wq_b_last, csa_idx_wq_b_scale_last, csa_weights_proj_last, csa_hadamard_idx_last,
            csa_inner_wkv_last, csa_inner_wgate_last, csa_inner_ape_last, csa_inner_norm_w_last,
            csa_inner_compress_state_last, csa_inner_compress_state_block_table,
            kv_cache_last, block_table, cmp_kv_last, cmp_block_table,
            idx_kv_cache_last, idx_block_table,
            ori_slot_mapping, csa_cmp_slot_mapping, csa_idx_slot_mapping,
            csa_state_slot_mapping, csa_inner_state_slot_mapping,
            position_ids, kv_seq_lens,
            attn_sink_last, wo_a_last, wo_b_last, wo_b_scale_last,
            x_attn_last,
        )
    with pl.scope():
        moe(
            x_attn_last,
            hc_ffn_fn_last, hc_ffn_scale_last, hc_ffn_base_last,
            norm_w_last, gate_w_last, gate_bias_last, tid2eid_last, input_ids,
            routed_w1_last, routed_w1_scale_last, routed_w3_last, routed_w3_scale_last,
            routed_w2_last, routed_w2_scale_last,
            shared_w1_last, shared_w1_scale_last, shared_w3_last, shared_w3_scale_last,
            shared_w2_last, shared_w2_scale_last,
            x_next_hc,
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            csa_layer_last, num_tokens, my_rank, last_moe_epoch,
        )
    x_head: pl.Tensor[[T, D], pl.BF16] = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.scope():
        hc_head(x_next_hc, hc_head_fn, hc_head_scale, hc_head_base, x_head)
        rms_norm(x_head, final_norm_w, x_out)
    return x_out



@pl.jit.host
def l3_decode_fwd(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32],
    cmp_kv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16],
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
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[N_RANKS, B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    kv_seq_lens: pl.Tensor[[N_RANKS, B], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[N_RANKS, B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[N_RANKS, T, VOCAB], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    hidden_norm = pl.create_tensor([N_RANKS, T, D], dtype=pl.BF16)

    for r in pl.range(pld.world_size()):
        pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32] = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32] = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32] = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        decode_fwd(
            x_hc[r],
            hc_attn_fn[r],
            hc_attn_scale[r],
            hc_attn_base[r],
            attn_norm_w[r],
            wq_a[r],
            wq_b[r],
            wq_b_scale[r],
            wkv[r],
            gamma_cq[r],
            gamma_ckv[r],
            kv_cache[r],
            attn_sink[r],
            wo_a[r],
            wo_b[r],
            wo_b_scale[r],
            hca_cmp_wkv[r],
            hca_cmp_wgate[r],
            hca_cmp_ape[r],
            hca_cmp_norm_w[r],
            hca_compress_state[r],
            csa_cmp_wkv[r],
            csa_cmp_wgate[r],
            csa_cmp_ape[r],
            csa_cmp_norm_w[r],
            csa_compress_state[r],
            csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r],
            csa_weights_proj[r],
            csa_hadamard_idx[r],
            csa_inner_wkv[r],
            csa_inner_wgate[r],
            csa_inner_ape[r],
            csa_inner_norm_w[r],
            csa_inner_compress_state[r],
            cmp_kv[r],
            idx_kv_cache[r],
            hc_ffn_fn[r],
            hc_ffn_scale[r],
            hc_ffn_base[r],
            norm_w[r],
            gate_w[r],
            gate_bias[r],
            tid2eid[r],
            routed_w1[r],
            routed_w1_scale[r],
            routed_w3[r],
            routed_w3_scale[r],
            routed_w2[r],
            routed_w2_scale[r],
            shared_w1[r],
            shared_w1_scale[r],
            shared_w3[r],
            shared_w3_scale[r],
            shared_w2[r],
            shared_w2_scale[r],
            freqs_cos[r],
            freqs_sin[r],
            block_table[r],
            ori_slot_mapping[r],
            hca_cmp_slot_mapping[r],
            hca_state_slot_mapping[r],
            csa_cmp_slot_mapping[r],
            csa_idx_slot_mapping[r],
            csa_state_slot_mapping[r],
            csa_inner_state_slot_mapping[r],
            position_ids[r],
            kv_seq_lens[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            cmp_block_table[r],
            idx_block_table[r],
            input_ids[r],
            hc_head_fn[r],
            hc_head_scale[r],
            hc_head_base[r],
            final_norm_w[r],
            hidden_norm[r],
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            r, num_tokens,
            device=r,
        )

    lm_hidden_window_buf = pld.alloc_window_buffer(LM_HEAD_ACTIVE_TP_SIZE * LM_HEAD_T_MAX * D * 2)
    lm_hidden_done_buf = pld.alloc_window_buffer(LM_HEAD_ACTIVE_TP_SIZE * 4)
    lm_logits_window_buf = pld.alloc_window_buffer(LM_HEAD_T_MAX * VOCAB * 4)
    lm_logits_done_buf = pld.alloc_window_buffer(LM_HEAD_ACTIVE_TP_SIZE * 4)
    for r in pl.range(pld.world_size()):
        lm_hidden_window: pld.DistributedTensor[[LM_HEAD_ACTIVE_TP_SIZE * LM_HEAD_T_MAX, D], pl.BF16] = pld.window(
            lm_hidden_window_buf, [LM_HEAD_ACTIVE_TP_SIZE * LM_HEAD_T_MAX, D], dtype=pl.BF16
        )
        lm_hidden_done: pld.DistributedTensor[[LM_HEAD_ACTIVE_TP_SIZE, 1], pl.INT32] = pld.window(
            lm_hidden_done_buf, [LM_HEAD_ACTIVE_TP_SIZE, 1], dtype=pl.INT32
        )
        lm_logits_window: pld.DistributedTensor[[LM_HEAD_T_MAX, VOCAB], pl.FP32] = pld.window(
            lm_logits_window_buf, [LM_HEAD_T_MAX, VOCAB], dtype=pl.FP32
        )
        lm_logits_done: pld.DistributedTensor[[LM_HEAD_ACTIVE_TP_SIZE, 1], pl.INT32] = pld.window(
            lm_logits_done_buf, [LM_HEAD_ACTIVE_TP_SIZE, 1], dtype=pl.INT32
        )
        lm_head_tp(
            hidden_norm[r],
            lm_head_weight[r],
            logits[r],
            lm_hidden_window,
            lm_hidden_done,
            lm_logits_window,
            lm_logits_done,
            r,
            device=r,
        )


# ---------------------------------------------------------------------------
# Fixtures (kernel-only smoke path: no golden).  Stacked weights reuse each
# layer's standalone attention/moe init; routing metadata, slot mappings and
# tid2eid carry meaningful values.
# ---------------------------------------------------------------------------
def _make_layer_stacked_spec(name, base_specs, layer_count=FWD_NUM_LAYERS):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    packed_shape = [spec.shape[0], layer_count * spec.shape[1], *spec.shape[2:]]

    def init_value():
        if name == "tid2eid":
            token_ids = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
            topk_ids = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
            rows = []
            for layer in range(layer_count):
                layer_eids = (token_ids * TOPK + topk_ids + layer * TOPK) % N_EXPERTS_GLOBAL
                rows.append(layer_eids)
            packed = torch.cat(rows, dim=0)
            return packed.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

        base_init = spec.init_value
        if name in CACHE_POOL_NAMES:
            # Randn one layer's pool, then tile across layers along the block dim.
            one_layer = base_init()
            reps = [1, layer_count] + [1] * (one_layer.dim() - 2)
            return one_layer.repeat(*reps)
        return torch.cat([base_init() for _ in range(layer_count)], dim=1)

    return TensorSpec(
        name,
        packed_shape,
        spec.dtype,
        init_value=init_value,
        is_output=False,
    )


def _make_shared_spec(name, base_spec, out_name=None):
    from golden import TensorSpec
    return TensorSpec(out_name or name, list(base_spec.shape), base_spec.dtype, init_value=base_spec.init_value if out_name is None else None, is_output=out_name is not None)


def _make_hc_head_spec(name):
    import torch
    from golden import TensorSpec

    if name == "hc_head_fn":
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519,
        )
    if name == "hc_head_scale":
        return TensorSpec(
            name,
            [N_RANKS, 1],
            torch.float32,
            init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32),
        )
    if name == "hc_head_base":
        base = [5.9166, -3.6223, -2.9324, -3.3124]
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor(base, dtype=torch.float32).view(1, HC_MULT).expand(N_RANKS, -1).contiguous(),
        )
    raise ValueError(f"unclassified hc_head spec: {name}")


def _make_final_norm_spec(name):
    import torch
    from golden import TensorSpec

    if name == "final_norm_w":
        return TensorSpec(
            name,
            [N_RANKS, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16),
        )
    raise ValueError(f"unclassified final norm spec: {name}")


def _make_lm_head_spec(name):
    import torch
    from golden import TensorSpec

    if name == "lm_head_weight":
        def init_lm_head_weight():
            return (torch.randn(N_RANKS, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)

        return TensorSpec(
            name,
            [N_RANKS, VOCAB_PER_TP, D],
            torch.bfloat16,
            init_value=init_lm_head_weight,
        )
    raise ValueError(f"unclassified lm_head spec: {name}")


def _make_forward_metadata_specs(base_specs, start_pos=None):
    import torch
    from golden import TensorSpec

    seq_per_batch = T // B
    win = MODEL_CONFIG.sliding_window

    def ranked(init_single):
        return torch.stack([init_single() for _ in range(N_RANKS)], dim=0)

    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        pattern = torch.tensor([
            10,
            CSA_COMPRESS_RATIO - seq_per_batch,
            CSA_COMPRESS_RATIO - 1,
            CSA_COMPRESS_RATIO,
            CSA_COMPRESS_RATIO * 2 - seq_per_batch,
            CSA_COMPRESS_RATIO * 3 - 1,
            win - seq_per_batch,
            win - 1,
            win,
        ], dtype=torch.int32)
        return pattern.repeat((B + pattern.numel() - 1) // pattern.numel())[:B].clone()

    def init_position_ids_single():
        starts = init_start_pos().to(torch.int64)
        positions = torch.empty((T,), dtype=torch.int32)
        for t in range(T):
            b = t // seq_per_batch
            s = t - b * seq_per_batch
            positions[t] = starts[b] + s
        return positions

    def init_kv_seq_lens_single():
        return (init_start_pos().to(torch.int64) + seq_per_batch).to(torch.int32)

    def init_block_table_single(max_blocks):
        tbl = torch.full((B, max_blocks), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(max_blocks):
                tbl[b, j] = b * max_blocks + j
        return tbl

    def init_ori_slot_mapping_single():
        positions = init_position_ids_single().to(torch.int64)
        block_table = init_block_table_single(ORI_MAX_BLOCKS).to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // seq_per_batch
            pos = int(positions[t].item())
            slot = pos % win
            blk = int(block_table[b, slot // BLOCK_SIZE].item())
            mapping[t] = blk * BLOCK_SIZE + slot % BLOCK_SIZE
        return mapping

    def init_compressed_slot_mapping_single(compress_ratio, max_blocks):
        positions = init_position_ids_single().to(torch.int64)
        block_table = init_block_table_single(max_blocks).to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // seq_per_batch
            pos = int(positions[t].item())
            if (pos + 1) % compress_ratio == 0:
                cache_col = pos // compress_ratio
                logical_blk = cache_col // BLOCK_SIZE
                intra = cache_col % BLOCK_SIZE
                blk = int(block_table[b, logical_blk].item())
                mapping[t] = blk * BLOCK_SIZE + intra
        return mapping

    def init_state_slot_mapping_single(max_blocks, block_size):
        positions = init_position_ids_single().to(torch.int64)
        block_table = init_block_table_single(max_blocks).to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // seq_per_batch
            pos = int(positions[t].item())
            logical_blk = pos // block_size
            intra = pos % block_size
            blk = int(block_table[b, logical_blk].item())
            mapping[t] = blk * block_size + intra
        return mapping

    init_by_name = {
        "block_table": lambda: ranked(lambda: init_block_table_single(ORI_MAX_BLOCKS)),
        "cmp_block_table": lambda: ranked(lambda: init_block_table_single(CSA_CMP_MAX_BLOCKS)),
        "idx_block_table": lambda: ranked(lambda: init_block_table_single(CSA_IDX_CACHE_MAX_BLOCKS)),
        "hca_compress_state_block_table": lambda: ranked(lambda: init_block_table_single(HCA_COMPRESS_STATE_MAX_BLOCKS)),
        "csa_compress_state_block_table": lambda: ranked(lambda: init_block_table_single(CSA_MAIN_STATE_MAX_BLOCKS)),
        "csa_inner_compress_state_block_table": lambda: ranked(lambda: init_block_table_single(CSA_INNER_STATE_MAX_BLOCKS)),
        "ori_slot_mapping": lambda: ranked(init_ori_slot_mapping_single),
        "hca_cmp_slot_mapping": lambda: ranked(lambda: init_compressed_slot_mapping_single(HCA_COMPRESS_RATIO, CSA_CMP_MAX_BLOCKS)),
        "csa_cmp_slot_mapping": lambda: ranked(lambda: init_compressed_slot_mapping_single(CSA_COMPRESS_RATIO, CSA_CMP_MAX_BLOCKS)),
        "csa_idx_slot_mapping": lambda: ranked(lambda: init_compressed_slot_mapping_single(CSA_COMPRESS_RATIO, CSA_IDX_CACHE_MAX_BLOCKS)),
        "hca_state_slot_mapping": lambda: ranked(lambda: init_state_slot_mapping_single(HCA_COMPRESS_STATE_MAX_BLOCKS, HCA_COMPRESS_STATE_BLOCK_SIZE)),
        "csa_state_slot_mapping": lambda: ranked(lambda: init_state_slot_mapping_single(CSA_MAIN_STATE_MAX_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE)),
        "csa_inner_state_slot_mapping": lambda: ranked(lambda: init_state_slot_mapping_single(CSA_INNER_STATE_MAX_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE)),
        "position_ids": lambda: ranked(init_position_ids_single),
        "kv_seq_lens": lambda: ranked(init_kv_seq_lens_single),
    }

    return {
        name: TensorSpec(name, list(base_specs[name].shape), base_specs[name].dtype, init_value=init_value)
        for name, init_value in init_by_name.items()
    }


def _ranked_init(single_spec, *, replicated=False):
    import torch

    def init():
        if replicated:
            value = single_spec.create_tensor()
            return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()
        return torch.stack([single_spec.create_tensor() for _ in range(N_RANKS)], dim=0)

    return init


def _validate_layer_id(layer_id):
    if not 0 <= layer_id < MODEL_CONFIG.num_hidden_layers:
        raise ValueError(
            f"layer_id must be in [0, {MODEL_CONFIG.num_hidden_layers - 1}] "
            f"for {MODEL_CONFIG.name} hidden layers, got {layer_id}"
        )


def _attention_kind_for_layer(layer_id):
    _validate_layer_id(layer_id)
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported compress ratio {ratio} for layer_id={layer_id}")


def build_single_layer_tensor_specs(start_pos=None, layer_id=10):
    """Per-layer single-rank tensor specs: the base shapes/dtypes/inits that
    build_tensor_specs restacks across the 43 forward layers."""
    import torch
    from golden import ScalarSpec, TensorSpec

    _validate_layer_id(layer_id)

    swa_specs = {
        spec.name: spec
        for spec in build_attention_tensor_specs(start_pos)
        if isinstance(spec, TensorSpec)
    }
    hca_specs = {
        spec.name: spec
        for spec in build_hca_tensor_specs(start_pos)
        if isinstance(spec, TensorSpec)
    }
    csa_specs = {
        spec.name: spec
        for spec in build_csa_tensor_specs(start_pos)
        if isinstance(spec, TensorSpec)
    }
    moe_specs = build_moe_tensor_specs(layer_id)
    moe_tensor_specs = {spec.name: spec for spec in moe_specs if isinstance(spec, TensorSpec)}
    attention_kind = _attention_kind_for_layer(layer_id)
    active_specs = {
        "swa": swa_specs,
        "hca": hca_specs,
        "csa": csa_specs,
    }[attention_kind]

    replicated_attention = {
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "freqs_cos",
        "freqs_sin",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
        "hca_cmp_wkv",
        "hca_cmp_wgate",
        "hca_cmp_ape",
        "hca_cmp_norm_w",
        "csa_cmp_wkv",
        "csa_cmp_wgate",
        "csa_cmp_ape",
        "csa_cmp_norm_w",
        "csa_idx_wq_b",
        "csa_idx_wq_b_scale",
        "csa_weights_proj",
        "csa_hadamard_idx",
        "csa_inner_wkv",
        "csa_inner_wgate",
        "csa_inner_ape",
        "csa_inner_norm_w",
    }
    attention_specs = [
        ("x_hc", swa_specs["x_hc"]),
        ("hc_attn_fn", swa_specs["hc_attn_fn"]),
        ("hc_attn_scale", swa_specs["hc_attn_scale"]),
        ("hc_attn_base", swa_specs["hc_attn_base"]),
        ("attn_norm_w", swa_specs["attn_norm_w"]),
        ("wq_a", swa_specs["wq_a"]),
        ("wq_b", swa_specs["wq_b"]),
        ("wq_b_scale", swa_specs["wq_b_scale"]),
        ("wkv", swa_specs["wkv"]),
        ("gamma_cq", swa_specs["gamma_cq"]),
        ("gamma_ckv", swa_specs["gamma_ckv"]),
        ("freqs_cos", swa_specs["freqs_cos"]),
        ("freqs_sin", swa_specs["freqs_sin"]),
        ("kv_cache", swa_specs["kv_cache"]),
        ("block_table", swa_specs["block_table"]),
        ("ori_slot_mapping", active_specs["ori_slot_mapping"]),
        ("hca_cmp_slot_mapping", hca_specs["cmp_slot_mapping"]),
        ("hca_state_slot_mapping", hca_specs["state_slot_mapping"]),
        ("csa_cmp_slot_mapping", csa_specs["cmp_slot_mapping"]),
        ("csa_idx_slot_mapping", csa_specs["idx_slot_mapping"]),
        ("csa_state_slot_mapping", csa_specs["state_slot_mapping"]),
        ("csa_inner_state_slot_mapping", csa_specs["inner_state_slot_mapping"]),
        ("position_ids", active_specs["position_ids"]),
        ("kv_seq_lens", active_specs.get("kv_seq_lens", hca_specs["kv_seq_lens"])),
        ("attn_sink", swa_specs["attn_sink"]),
        ("wo_a", swa_specs["wo_a"]),
        ("wo_b", swa_specs["wo_b"]),
        ("wo_b_scale", swa_specs["wo_b_scale"]),
        ("hca_cmp_wkv", hca_specs["cmp_wkv"]),
        ("hca_cmp_wgate", hca_specs["cmp_wgate"]),
        ("hca_cmp_ape", hca_specs["cmp_ape"]),
        ("hca_cmp_norm_w", hca_specs["cmp_norm_w"]),
        ("hca_compress_state", hca_specs["compress_state"]),
        ("hca_compress_state_block_table", hca_specs["compress_state_block_table"]),
        ("csa_cmp_wkv", csa_specs["cmp_wkv"]),
        ("csa_cmp_wgate", csa_specs["cmp_wgate"]),
        ("csa_cmp_ape", csa_specs["cmp_ape"]),
        ("csa_cmp_norm_w", csa_specs["cmp_norm_w"]),
        ("csa_compress_state", csa_specs["compress_state"]),
        ("csa_compress_state_block_table", csa_specs["compress_state_block_table"]),
        ("csa_idx_wq_b", csa_specs["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa_specs["idx_wq_b_scale"]),
        ("csa_weights_proj", csa_specs["weights_proj"]),
        ("csa_hadamard_idx", csa_specs["hadamard_idx"]),
        ("csa_inner_wkv", csa_specs["inner_wkv"]),
        ("csa_inner_wgate", csa_specs["inner_wgate"]),
        ("csa_inner_ape", csa_specs["inner_ape"]),
        ("csa_inner_norm_w", csa_specs["inner_norm_w"]),
        ("csa_inner_compress_state", csa_specs["inner_compress_state"]),
        ("csa_inner_compress_state_block_table", csa_specs["inner_compress_state_block_table"]),
        ("cmp_kv", csa_specs["cmp_kv"]),
        ("cmp_block_table", csa_specs["cmp_block_table"]),
        ("idx_kv_cache", csa_specs["idx_kv_cache"]),
        ("idx_block_table", csa_specs["idx_block_table"]),
    ]

    specs = [
        TensorSpec(
            name,
            [N_RANKS, *spec.shape],
            spec.dtype,
            init_value=_ranked_init(spec, replicated=name in replicated_attention),
            is_output=name == "kv_cache",
        )
        for name, spec in attention_specs
    ]

    for spec in moe_specs:
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid():
                base = torch.arange(VOCAB, dtype=torch.int32).reshape(VOCAB, 1) * TOPK
                offs = torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK)
                table = (base + offs) % N_EXPERTS_GLOBAL
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            specs.append(TensorSpec("tid2eid", spec.shape, spec.dtype, init_value=init_tid2eid))
        elif spec.name == "input_ids":
            def init_input_ids():
                ids = torch.arange(T, dtype=torch.int64)
                return ids.unsqueeze(0).expand(N_RANKS, -1).contiguous()

            specs.append(TensorSpec("input_ids", spec.shape, spec.dtype, init_value=init_input_ids))
        else:
            specs.append(moe_tensor_specs[spec.name])

    specs.extend([
        TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ])
    return specs


def build_tensor_specs(start_pos=None, num_tokens=T):
    import torch
    from golden import ScalarSpec, TensorSpec
    base_specs = {spec.name: spec for spec in build_single_layer_tensor_specs(start_pos=start_pos, layer_id=0) if isinstance(spec, TensorSpec)}
    metadata_specs = _make_forward_metadata_specs(base_specs, start_pos=start_pos)
    ordered_names = ['x_hc', 'hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'attn_norm_w', 'wq_a', 'wq_b', 'wq_b_scale', 'wkv', 'gamma_cq', 'gamma_ckv', 'kv_cache', 'attn_sink', 'wo_a', 'wo_b', 'wo_b_scale', 'hca_cmp_wkv', 'hca_cmp_wgate', 'hca_cmp_ape', 'hca_cmp_norm_w', 'hca_compress_state', 'csa_cmp_wkv', 'csa_cmp_wgate', 'csa_cmp_ape', 'csa_cmp_norm_w', 'csa_compress_state', 'csa_idx_wq_b', 'csa_idx_wq_b_scale', 'csa_weights_proj', 'csa_hadamard_idx', 'csa_inner_wkv', 'csa_inner_wgate', 'csa_inner_ape', 'csa_inner_norm_w', 'csa_inner_compress_state', 'cmp_kv', 'idx_kv_cache', 'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base', 'norm_w', 'gate_w', 'gate_bias', 'tid2eid', 'routed_w1', 'routed_w1_scale', 'routed_w3', 'routed_w3_scale', 'routed_w2', 'routed_w2_scale', 'shared_w1', 'shared_w1_scale', 'shared_w3', 'shared_w3_scale', 'shared_w2', 'shared_w2_scale', 'freqs_cos', 'freqs_sin', 'block_table', 'ori_slot_mapping', 'hca_cmp_slot_mapping', 'hca_state_slot_mapping', 'csa_cmp_slot_mapping', 'csa_idx_slot_mapping', 'csa_state_slot_mapping', 'csa_inner_state_slot_mapping', 'position_ids', 'kv_seq_lens', 'hca_compress_state_block_table', 'csa_compress_state_block_table', 'csa_inner_compress_state_block_table', 'cmp_block_table', 'idx_block_table', 'input_ids', 'hc_head_fn', 'hc_head_scale', 'hc_head_base', 'final_norm_w', 'lm_head_weight']
    specs = []
    for name in ordered_names:
        if name in metadata_specs:
            specs.append(metadata_specs[name])
        elif name in CSA_LAYER_STACKED_NAMES:
            specs.append(_make_layer_stacked_spec(name, base_specs, CSA_NUM_LAYERS))
        elif name in HCA_LAYER_STACKED_NAMES:
            specs.append(_make_layer_stacked_spec(name, base_specs, HCA_NUM_LAYERS))
        elif name in LAYER_STACKED_NAMES:
            specs.append(_make_layer_stacked_spec(name, base_specs))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs[name]))
        elif name in HC_HEAD_NAMES:
            specs.append(_make_hc_head_spec(name))
        elif name in FINAL_NORM_NAMES:
            specs.append(_make_final_norm_spec(name))
        elif name in LM_HEAD_NAMES:
            specs.append(_make_lm_head_spec(name))
        else:
            raise ValueError(f"unclassified decode_fwd spec: {name}")
    specs.append(TensorSpec("logits", [N_RANKS, T, VOCAB], torch.float32, is_output=True))
    specs.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    return specs


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash packed single-token decode forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8], help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)), help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=None, help="If set, use this single start_pos for all batches.")
    parser.add_argument("--num-tokens", type=int, default=T, help=f"Active token rows for MoE routing/combine; default is T={T}.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    specs = build_tensor_specs(start_pos=args.start_pos, num_tokens=args.num_tokens)

    result = run_jit(
        fn=l3_decode_fwd,
        specs=specs,
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_scope_stats=args.enable_scope_stats,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
