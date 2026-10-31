# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash forward kernels, tensor contracts and CPU weight validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import dataclasses
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from math import gcd
from typing import Mapping, Sequence

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import models.glm5_3_flash.config as C
from models.glm5_3_flash.layer import (
    AUX_PAD, BATCH, CACHE_LAYOUTS as CACHE_LAYOUTS, CACHE_TOKENS, CAPACITY, COUNT_PAD,
    DENSE_NUM_LAYERS, DSA_NUM_LAYERS,
    FORWARD_ABI_VERSION as FORWARD_ABI_VERSION, FORWARD_PROGRAMS as FORWARD_PROGRAMS, HC_DIM, KDA_CONV_K,
    KDA_NUM_LAYERS, KDA_ROWS, KDA_STATE_BLOCKS as KDA_STATE_BLOCKS, KDA_STATE_COLUMNS,
    LATENT_ROWS, LOCAL_DENSE_INTER, LOGIT_ROWS, LayerRefWeights,
    MAX_SEQ_LEN, MIX_HC, MLA_STACK_LAYERS as MLA_STACK_LAYERS, MOE_INTER,
    MOE_STACK_LAYERS as MOE_STACK_LAYERS, NORM_STACK_LAYERS, NUM_LAYERS as NUM_LAYERS, N_LOCAL_EXPERTS,
    N_RANKS, POOL_AXIS, POOL_STRIDE, POOL_TABLE_MAX, POOL_TABLE_ROWS, PREFILL_MAX_TOKENS, QUERY_ROWS_DYN,
    Q_LORA, RANKS, RESIDENT_ROUTES_PER_SRC, RESIDENT_TOTAL_CAP, ROUTER_STACK_LAYERS, TABLE_MAX, T_MOE,
    _complete_forward_stage, _shape_options, _test_mla, _test_router_and_moe,
    forward_epoch_guard as forward_epoch_guard, forward_geometry as forward_geometry,
    forward_parameter_names as forward_parameter_names,
    forward_query_tensor_names as forward_query_tensor_names,
    forward_tensor_layouts as forward_tensor_layouts, l3_prefill_head_rows,
    l3_prefill_lm_head_logits as l3_prefill_lm_head_logits, l3_prefill_moe_layer,
    l3_prefill_mtp_head as l3_prefill_mtp_head, l3_prefill_mtp_prepare as l3_prefill_mtp_prepare,
    l3_prefill_sparse_attention, prefill_dense_layer, prefill_head_rows,
    prefill_kda_attention, prefill_lm_head_logits, prefill_moe_layer, prefill_seed, prefill_sparse_attention,
    reference_layer_forward, synthetic_layer,
)
from models.glm5_3_flash.attention import MlaTensors
from models.glm5_3_flash.config import (
    BLOCK_SIZE, DENSE_INTER, DP_SIZE as DP_SIZE, EP_SIZE as EP_SIZE, FLASH, HC_MULT, IMAGE_TOKEN_ID, INDEX_DIM, INDEX_H,
    INDEX_KPOOL, INDEX_STATE_WIDTH, KDA_DIM, KPOOL_SELECT_K, KV_LORA, LOCAL_H, LOCAL_KDA_H,
    LOCAL_KDA_QKV_DIM, LOCAL_VOCAB as LOCAL_VOCAB, N_EXPERTS, QK_DIM, SUPPORTED_EP_SIZES, SUPPORTED_TP_SIZES, TP_SIZE,
    V_DIM, AttentionKind, D, Glm53FlashConfig, MlpKind,
)
from models.glm5_3_flash.dense_mlp import DenseTensors, _test_dense_mlp
from models.glm5_3_flash.expert_routed import CheckpointExpertProvider
from models.glm5_3_flash.gate import RouterTensors
from models.glm5_3_flash.indexer import POOL_SCALE_WIDTH, IndexerTensors, _test_indexer
from models.glm5_3_flash.kda import (
    BETA_PAD, CHUNK_TILE as CHUNK_TILE, CONV_DIM, STATE_LEN, KdaTensors, _test_kda_cores, kda_host_constants,
)
from models.glm5_3_flash.lm_head import FP32_HEAD_VOCAB_PAD as FP32_HEAD_VOCAB_PAD
from models.glm5_3_flash.mhc import MhcTensors, _test_mhc, combine_streams
from models.glm5_3_flash.rmsnorm import reference_rms_norm
from models.glm5_3_flash.utils import (
    _FF_DOWN, _FF_FUSED, _FF_GATE, _FF_UP, ROLES, TEXT_LAYER_ROOTS, Checkpoint, QuantDescription,
    CacheTables, _as_batch_seqs, _assert_close, _find, _pad_to, _prefill_model as _prefill_model, _prefixed,
    _read_quant, _resolve_name, _resolve_top_name, _scale_of, _try_find, _weight_shape,
    build_forward_metadata, build_indexer_chunk_metadata, check_ffn_scales, fuse_gate_up, gather_rows,
    golden_split_kv_b, layer_prefix, layer_prefixes, layer_router, load_shared_expert, load_tensors,
    pack_int8_nz, read_reference_float, scaled_model_from_config, shard_dense_mlp, shard_routed_experts,
    split_kv_b, sylvester_hadamard, unpack_int8_nz,
)

_inline_prefill_seed = pl.jit.inline(auto_scope=False)(prefill_seed._func)

_inline_prefill_dense_layer = pl.jit.inline(auto_scope=False)(prefill_dense_layer._func)

_inline_prefill_kda_attention = pl.jit.inline(auto_scope=False)(prefill_kda_attention._func)

_inline_prefill_sparse_attention = pl.jit.inline(auto_scope=False)(prefill_sparse_attention._func)

_inline_prefill_moe_layer = pl.jit.inline(auto_scope=False)(prefill_moe_layer._func)

_inline_prefill_head_rows = pl.jit.inline(auto_scope=False)(prefill_head_rows._func)

_inline_prefill_lm_head_logits = pl.jit.inline(auto_scope=False)(prefill_lm_head_logits._func)


@pl.jit(auto_scope=False)
def prefill_hidden_fwd(
    embedding_rows: pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH, KDA_STATE_COLUMNS], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    kda_w_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    mhc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    norm_post_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    dense_w_gate_up: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D], pl.INT8],
    dense_w_gate_up_scale: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_w_down: pl.Tensor[[DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER], pl.INT8],
    dense_w_down_scale: pl.Tensor[[DENSE_NUM_LAYERS, D], pl.FP32],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    ws_attn_pre: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT], pl.FP32]],
    ws_attn_post: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT], pl.FP32]],
    ws_attn_res: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT, HC_MULT], pl.FP32]],
    ws_collapsed: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16]],
    ws_normed: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16]],
    ws_attn_partial: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.FP32]],
    ws_sublayer_out: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16]],
    ws_x_hc_mid: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT, D], pl.BF16]],
    ws_ffn_pre: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT], pl.FP32]],
    ws_ffn_post: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT], pl.FP32]],
    ws_ffn_res: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, HC_MULT, HC_MULT], pl.FP32]],
    ws_ffn_input: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16]],
    ws_ffn_partial: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.FP32]],
    ws_ffn_out: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    dense_x_int8: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, D], pl.INT8]],
    request_ids: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    mla_slots: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    index_slots: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    seg_start: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    pool_count_tok: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    tail_start: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    tail_count: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    kv_len_tok: pl.Tensor[[QUERY_ROWS_DYN], pl.INT32],
    pool_token_slots: pl.Tensor[[POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_AXIS], pl.INT32],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    mla_w_q_a: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, D], pl.INT8],
    mla_w_q_a_scale: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.FP32],
    mla_q_a_norm: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    mla_w_q_b_scale: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM], pl.FP32],
    mla_w_kv_a: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA, D], pl.INT8],
    mla_w_kv_a_scale: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.FP32],
    mla_kv_a_norm: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_v: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[MLA_STACK_LAYERS, D, LOCAL_H * V_DIM], pl.INT8],
    mla_w_o_scale: pl.Tensor[[MLA_STACK_LAYERS, D], pl.FP32],
    idx_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_compress_ape: pl.Tensor[[MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    dsa_query: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, LOCAL_H, V_DIM], pl.BF16]],
    index_q: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, INDEX_DIM], pl.BF16]],
    index_scores: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, POOL_AXIS], pl.FP32]],
    selected_pools: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[QUERY_ROWS_DYN, KPOOL_SELECT_K], pl.INT32]],
    router_gate_w: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS, D], pl.BF16],
    router_gate_bias: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS], pl.FP32],
    moe_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    moe_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    moe_w_down: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    moe_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    moe_shared_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER, D], pl.INT8, pl.NZ],
    moe_shared_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER], pl.FP32],
    moe_shared_w_down: pl.Tensor[[MOE_STACK_LAYERS, D, MOE_INTER], pl.INT8, pl.NZ],
    moe_shared_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, D], pl.FP32],
    logit_row_indices: pl.Tensor[[LOGIT_ROWS], pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[LOGIT_ROWS, D], pl.BF16]],
    ws_x_hc_pair: pl.InOut[pl.Tensor[[2 * CAPACITY, HC_MULT, D], pl.BF16]],
    tp_partial: pld.DistributedTensor[[QUERY_ROWS_DYN, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    lane_x: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[RESIDENT_ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    forward_epoch: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run the complete prefill backbone and gather request-final hidden rows."""
    current = pl.slice(ws_x_hc_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    _inline_prefill_seed(
        embedding_rows, current,
    )
    for layer in pl.range(NUM_LAYERS):
        current_offset = (layer % 2) * CAPACITY
        following_offset = ((layer + 1) % 2) * CAPACITY
        current = pl.slice(ws_x_hc_pair, [CAPACITY, HC_MULT, D], [current_offset, 0, 0])
        following = pl.slice(ws_x_hc_pair, [CAPACITY, HC_MULT, D], [following_offset, 0, 0])
        bb_ord = pl.cast(layer, pl.INT32)
        kda_ord = pl.cast(layer - layer // 4, pl.INT32)
        dsa_ord = pl.cast(layer // 4, pl.INT32)
        dense_ord = pl.cast(layer, pl.INT32)
        moe_ord = pl.cast(layer - DENSE_NUM_LAYERS, pl.INT32)
        attn_epoch = forward_epoch * (2 * NUM_LAYERS) + 2 * bb_ord + 1
        ffn_epoch = attn_epoch + 1
        moe_epoch = forward_epoch * (NUM_LAYERS - DENSE_NUM_LAYERS) + moe_ord + 1
        skip_topk = pl.cast(0, pl.INT32)
        if layer < DENSE_NUM_LAYERS:
            _inline_prefill_dense_layer(
                current, query_start_loc, kda_state_rows, active_tokens, kda_tri_incl,
                kda_tri_strict, kda_head_onehot, kda_eye_wide, kda_w_q, kda_w_k,
                kda_w_v, kda_conv_q, kda_conv_k, kda_conv_v, kda_w_f_a,
                kda_w_f_b, kda_w_g_a, kda_w_g_b, kda_w_b, kda_dt_bias,
                kda_a_log, kda_o_norm, kda_w_o, mhc_attn_fn, mhc_attn_scale,
                mhc_attn_base, mhc_ffn_fn, mhc_ffn_scale, mhc_ffn_base, norm_input_norm,
                norm_post_norm, dense_w_gate_up, dense_w_gate_up_scale, dense_w_down, dense_w_down_scale,
                kda_conv_state, kda_recurrent_state, tp_partial, tp_arrived, ws_attn_pre,
                ws_attn_post, ws_attn_res, ws_collapsed, ws_normed, ws_attn_partial,
                ws_sublayer_out, ws_x_hc_mid, ws_ffn_pre, ws_ffn_post, ws_ffn_res,
                ws_ffn_input, ws_ffn_partial, ws_ffn_out, following, kda_out,
                mixed_qkv, decay, beta, kda_q, kda_k,
                kda_v, dense_x_int8, bb_ord, kda_ord, dense_ord,
                rank, attn_epoch, ffn_epoch, CAPACITY,
            )
        else:
            if layer % 4 == 3:
                _inline_prefill_sparse_attention(
                    current, request_ids, mla_slots, index_slots, seg_start,
                    pool_count_tok, tail_start, tail_count, kv_len_tok, pool_token_slots,
                    pool_slots, pool_blocks, block_table, active_tokens, mla_w_q_a,
                    mla_w_q_a_scale, mla_q_a_norm, mla_w_q_b, mla_w_q_b_scale, mla_w_kv_a,
                    mla_w_kv_a_scale, mla_kv_a_norm, mla_w_k, mla_w_v, mla_w_o,
                    mla_w_o_scale, idx_w_q_b, idx_w_k, idx_k_norm_w, idx_k_norm_b,
                    idx_w_weights, idx_w_compress_gate, idx_compress_ape, mhc_attn_fn, mhc_attn_scale,
                    mhc_attn_base, norm_input_norm, hadamard, mla_latent_cache, index_raw_cache,
                    index_pool_cache, index_pool_scale, tp_partial, tp_arrived, ws_attn_pre,
                    ws_attn_post, ws_attn_res, ws_collapsed, ws_normed, ws_attn_partial,
                    ws_sublayer_out, ws_x_hc_mid, dsa_query, attn_out, index_q,
                    index_k, index_scores, selected_pools, selected_valid, bb_ord,
                    dsa_ord, rank, attn_epoch, skip_topk, CAPACITY,
                )
            else:
                _inline_prefill_kda_attention(
                    current, query_start_loc, kda_state_rows, active_tokens, kda_tri_incl,
                    kda_tri_strict, kda_head_onehot, kda_eye_wide, kda_w_q, kda_w_k,
                    kda_w_v, kda_conv_q, kda_conv_k, kda_conv_v, kda_w_f_a,
                    kda_w_f_b, kda_w_g_a, kda_w_g_b, kda_w_b, kda_dt_bias,
                    kda_a_log, kda_o_norm, kda_w_o, mhc_attn_fn, mhc_attn_scale,
                    mhc_attn_base, norm_input_norm, kda_conv_state, kda_recurrent_state, tp_partial,
                    tp_arrived, ws_attn_pre, ws_attn_post, ws_attn_res, ws_collapsed,
                    ws_normed, ws_attn_partial, ws_sublayer_out, ws_x_hc_mid, kda_out,
                    mixed_qkv, decay, beta, kda_q, kda_k,
                    kda_v, bb_ord, kda_ord, rank, attn_epoch,
                    CAPACITY,
                )
            _complete_forward_stage(ws_x_hc_mid)
            _inline_prefill_moe_layer(
                ws_x_hc_mid, active_tokens, mhc_ffn_fn, mhc_ffn_scale, mhc_ffn_base,
                norm_post_norm, router_gate_w, router_gate_bias, moe_w_gate_up, moe_w_gate_up_scale,
                moe_w_down, moe_w_down_scale, moe_shared_w_gate_up, moe_shared_w_gate_up_scale, moe_shared_w_down,
                moe_shared_w_down_scale, lane_x, lane_aux, lane_count, count_ready,
                x_ready, slot_out, return_ready, tp_partial, tp_arrived,
                ws_ffn_pre, ws_ffn_post, ws_ffn_res, ws_ffn_input, ws_ffn_partial,
                ws_ffn_out, following, bb_ord, moe_ord, rank,
                moe_epoch, ffn_epoch,
            )
        _complete_forward_stage(following)
    current = pl.slice(ws_x_hc_pair, [CAPACITY, HC_MULT, D], [(NUM_LAYERS % 2) * CAPACITY, 0, 0])
    _inline_prefill_head_rows(
        current, logit_row_indices, head_hidden,
    )
    return head_hidden


_inline_prefill_hidden_fwd = pl.jit.inline(auto_scope=False)(prefill_hidden_fwd._func)


@pl.jit(auto_scope=False)
def prefill_fwd(
    embedding_rows: pl.Tensor[[CAPACITY, D], pl.BF16],
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH, KDA_STATE_COLUMNS], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    kda_w_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    mhc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    norm_post_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    dense_w_gate_up: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D], pl.INT8],
    dense_w_gate_up_scale: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_w_down: pl.Tensor[[DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER], pl.INT8],
    dense_w_down_scale: pl.Tensor[[DENSE_NUM_LAYERS, D], pl.FP32],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    ws_attn_pre: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    ws_attn_post: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    ws_attn_res: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    ws_collapsed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    ws_normed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    ws_attn_partial: pl.InOut[pl.Tensor[[CAPACITY, D], pl.FP32]],
    ws_sublayer_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    ws_x_hc_mid: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    ws_ffn_pre: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    ws_ffn_post: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    ws_ffn_res: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    ws_ffn_input: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    ws_ffn_partial: pl.InOut[pl.Tensor[[CAPACITY, D], pl.FP32]],
    ws_ffn_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    dense_x_int8: pl.InOut[pl.Tensor[[CAPACITY, D], pl.INT8]],
    request_ids: pl.Tensor[[CAPACITY], pl.INT32],
    mla_slots: pl.Tensor[[CAPACITY], pl.INT32],
    index_slots: pl.Tensor[[CAPACITY], pl.INT32],
    seg_start: pl.Tensor[[CAPACITY], pl.INT32],
    pool_count_tok: pl.Tensor[[CAPACITY], pl.INT32],
    tail_start: pl.Tensor[[CAPACITY], pl.INT32],
    tail_count: pl.Tensor[[CAPACITY], pl.INT32],
    kv_len_tok: pl.Tensor[[CAPACITY], pl.INT32],
    pool_token_slots: pl.Tensor[[POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_AXIS], pl.INT32],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    mla_w_q_a: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, D], pl.INT8],
    mla_w_q_a_scale: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.FP32],
    mla_q_a_norm: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    mla_w_q_b_scale: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM], pl.FP32],
    mla_w_kv_a: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA, D], pl.INT8],
    mla_w_kv_a_scale: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.FP32],
    mla_kv_a_norm: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_v: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[MLA_STACK_LAYERS, D, LOCAL_H * V_DIM], pl.INT8],
    mla_w_o_scale: pl.Tensor[[MLA_STACK_LAYERS, D], pl.FP32],
    idx_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_compress_ape: pl.Tensor[[MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    dsa_query: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_H, QK_DIM], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_H, V_DIM], pl.BF16]],
    index_q: pl.InOut[pl.Tensor[[CAPACITY, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.InOut[pl.Tensor[[CAPACITY, INDEX_DIM], pl.BF16]],
    index_scores: pl.InOut[pl.Tensor[[CAPACITY, POOL_AXIS], pl.FP32]],
    selected_pools: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    router_gate_w: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS, D], pl.BF16],
    router_gate_bias: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS], pl.FP32],
    moe_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    moe_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    moe_w_down: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    moe_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    moe_shared_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER, D], pl.INT8, pl.NZ],
    moe_shared_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER], pl.FP32],
    moe_shared_w_down: pl.Tensor[[MOE_STACK_LAYERS, D, MOE_INTER], pl.INT8, pl.NZ],
    moe_shared_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, D], pl.FP32],
    logit_row_indices: pl.Tensor[[LOGIT_ROWS], pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[LOGIT_ROWS, D], pl.BF16]],
    ws_x_hc_pair: pl.InOut[pl.Tensor[[2 * CAPACITY, HC_MULT, D], pl.BF16]],
    final_norm: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
    tp_partial: pld.DistributedTensor[[QUERY_ROWS_DYN, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    lane_x: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[RESIDENT_ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    forward_epoch: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run a complete prefill step with device logits."""
    _inline_prefill_hidden_fwd(
        embedding_rows, query_start_loc, kda_state_rows, active_tokens, kda_tri_incl,
        kda_tri_strict, kda_head_onehot, kda_eye_wide, kda_w_q, kda_w_k,
        kda_w_v, kda_conv_q, kda_conv_k, kda_conv_v, kda_w_f_a,
        kda_w_f_b, kda_w_g_a, kda_w_g_b, kda_w_b, kda_dt_bias,
        kda_a_log, kda_o_norm, kda_w_o, mhc_attn_fn, mhc_attn_scale,
        mhc_attn_base, mhc_ffn_fn, mhc_ffn_scale, mhc_ffn_base, norm_input_norm,
        norm_post_norm, dense_w_gate_up, dense_w_gate_up_scale, dense_w_down, dense_w_down_scale,
        kda_conv_state, kda_recurrent_state, ws_attn_pre, ws_attn_post, ws_attn_res,
        ws_collapsed, ws_normed, ws_attn_partial, ws_sublayer_out, ws_x_hc_mid,
        ws_ffn_pre, ws_ffn_post, ws_ffn_res, ws_ffn_input, ws_ffn_partial,
        ws_ffn_out, kda_out, mixed_qkv, decay, beta,
        kda_q, kda_k, kda_v, dense_x_int8, request_ids,
        mla_slots, index_slots, seg_start, pool_count_tok, tail_start,
        tail_count, kv_len_tok, pool_token_slots, pool_slots, pool_blocks,
        block_table, mla_w_q_a, mla_w_q_a_scale, mla_q_a_norm, mla_w_q_b,
        mla_w_q_b_scale, mla_w_kv_a, mla_w_kv_a_scale, mla_kv_a_norm, mla_w_k,
        mla_w_v, mla_w_o, mla_w_o_scale, idx_w_q_b, idx_w_k,
        idx_k_norm_w, idx_k_norm_b, idx_w_weights, idx_w_compress_gate, idx_compress_ape,
        hadamard, mla_latent_cache, index_raw_cache, index_pool_cache, index_pool_scale,
        dsa_query, attn_out, index_q, index_k, index_scores,
        selected_pools, selected_valid, router_gate_w, router_gate_bias, moe_w_gate_up,
        moe_w_gate_up_scale, moe_w_down, moe_w_down_scale, moe_shared_w_gate_up, moe_shared_w_gate_up_scale,
        moe_shared_w_down, moe_shared_w_down_scale, logit_row_indices, head_hidden, ws_x_hc_pair,
        tp_partial, tp_arrived, lane_x, lane_aux, lane_count,
        count_ready, x_ready, slot_out, return_ready, forward_epoch,
        rank,
    )
    _inline_prefill_lm_head_logits(
        head_hidden, final_norm, lm_head_weight, logits,
    )
    return logits


@pl.jit.host
def l3_prefill_fwd(
    embedding_rows: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    query_start_loc: pl.Tensor[[RANKS, BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[RANKS, BATCH, KDA_STATE_COLUMNS], pl.INT32],
    active_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    kda_tri_incl: pl.Tensor[[RANKS, CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_tri_strict: pl.Tensor[[RANKS, CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_head_onehot: pl.Tensor[[RANKS, LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[RANKS, CHUNK_TILE, KDA_DIM], pl.FP32],
    kda_w_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[RANKS, KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    mhc_ffn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    norm_post_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    dense_w_gate_up: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D], pl.INT8],
    dense_w_gate_up_scale: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_w_down: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER], pl.INT8],
    dense_w_down_scale: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, D], pl.FP32],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    ws_attn_pre: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    ws_attn_post: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    ws_attn_res: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    ws_collapsed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    ws_normed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    ws_attn_partial: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    ws_sublayer_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    ws_x_hc_mid: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    ws_ffn_pre: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    ws_ffn_post: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    ws_ffn_res: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    ws_ffn_input: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    ws_ffn_partial: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    ws_ffn_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[RANKS, CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[RANKS, CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    dense_x_int8: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.INT8]],
    request_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    mla_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    index_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    seg_start: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    pool_count_tok: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    tail_start: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    tail_count: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    kv_len_tok: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    pool_token_slots: pl.Tensor[[RANKS, POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[RANKS, POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[RANKS, POOL_AXIS], pl.INT32],
    block_table: pl.Tensor[[RANKS, BATCH, TABLE_MAX], pl.INT32],
    mla_w_q_a: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA, D], pl.INT8],
    mla_w_q_a_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA], pl.FP32],
    mla_q_a_norm: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    mla_w_q_b_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H * QK_DIM], pl.FP32],
    mla_w_kv_a: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA, D], pl.INT8],
    mla_w_kv_a_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA], pl.FP32],
    mla_kv_a_norm: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_v: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, LOCAL_H * V_DIM], pl.INT8],
    mla_w_o_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D], pl.FP32],
    idx_w_q_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_compress_ape: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[RANKS, INDEX_DIM, INDEX_DIM], pl.BF16],
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    dsa_query: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_H, QK_DIM], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_H, V_DIM], pl.BF16]],
    index_q: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_DIM], pl.BF16]],
    index_scores: pl.InOut[pl.Tensor[[RANKS, CAPACITY, POOL_AXIS], pl.FP32]],
    selected_pools: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    router_gate_w: pl.Tensor[[RANKS, ROUTER_STACK_LAYERS, N_EXPERTS, D], pl.BF16],
    router_gate_bias: pl.Tensor[[RANKS, ROUTER_STACK_LAYERS, N_EXPERTS], pl.FP32],
    moe_w_gate_up: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    moe_w_gate_up_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    moe_w_down: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    moe_w_down_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    moe_shared_w_gate_up: pl.Tensor[[RANKS, MOE_STACK_LAYERS, 2 * MOE_INTER, D], pl.INT8],
    moe_shared_w_gate_up_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, 2 * MOE_INTER], pl.FP32],
    moe_shared_w_down: pl.Tensor[[RANKS, MOE_STACK_LAYERS, D, MOE_INTER], pl.INT8],
    moe_shared_w_down_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, D], pl.FP32],
    logit_row_indices: pl.Tensor[[RANKS, LOGIT_ROWS], pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[RANKS, LOGIT_ROWS, D], pl.BF16]],
    ws_x_hc_pair: pl.InOut[pl.Tensor[[RANKS, 2 * CAPACITY, HC_MULT, D], pl.BF16]],
    final_norm: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[RANKS, BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
    forward_epoch: pl.Scalar[pl.INT32],
):
    """Launch one complete prefill orchestration per rank."""
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    lane_x_buffer = pld.alloc_window_buffer([RESIDENT_TOTAL_CAP, D], dtype=pl.INT8)
    lane_aux_buffer = pld.alloc_window_buffer([RESIDENT_TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
    lane_count_buffer = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    count_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buffer = pld.alloc_window_buffer([RESIDENT_ROUTES_PER_SRC, D], dtype=pl.FP32)
    return_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        lane_x = pld.window(lane_x_buffer, [RESIDENT_TOTAL_CAP, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buffer, [RESIDENT_TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buffer, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
        count_ready = pld.window(count_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        x_ready = pld.window(x_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buffer, [RESIDENT_ROUTES_PER_SRC, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        prefill_fwd(
            embedding_rows[rank], query_start_loc[rank], kda_state_rows[rank], active_tokens[rank], kda_tri_incl[rank],
            kda_tri_strict[rank], kda_head_onehot[rank], kda_eye_wide[rank], kda_w_q[rank], kda_w_k[rank],
            kda_w_v[rank], kda_conv_q[rank], kda_conv_k[rank], kda_conv_v[rank], kda_w_f_a[rank],
            kda_w_f_b[rank], kda_w_g_a[rank], kda_w_g_b[rank], kda_w_b[rank], kda_dt_bias[rank],
            kda_a_log[rank], kda_o_norm[rank], kda_w_o[rank], mhc_attn_fn[rank], mhc_attn_scale[rank],
            mhc_attn_base[rank], mhc_ffn_fn[rank], mhc_ffn_scale[rank], mhc_ffn_base[rank], norm_input_norm[rank],
            norm_post_norm[rank], dense_w_gate_up[rank], dense_w_gate_up_scale[rank], dense_w_down[rank], dense_w_down_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank], ws_attn_pre[rank], ws_attn_post[rank], ws_attn_res[rank],
            ws_collapsed[rank], ws_normed[rank], ws_attn_partial[rank], ws_sublayer_out[rank], ws_x_hc_mid[rank],
            ws_ffn_pre[rank], ws_ffn_post[rank], ws_ffn_res[rank], ws_ffn_input[rank], ws_ffn_partial[rank],
            ws_ffn_out[rank], kda_out[rank], mixed_qkv[rank], decay[rank], beta[rank],
            kda_q[rank], kda_k[rank], kda_v[rank], dense_x_int8[rank], request_ids[rank],
            mla_slots[rank], index_slots[rank], seg_start[rank], pool_count_tok[rank], tail_start[rank],
            tail_count[rank], kv_len_tok[rank], pool_token_slots[rank], pool_slots[rank], pool_blocks[rank],
            block_table[rank], mla_w_q_a[rank], mla_w_q_a_scale[rank], mla_q_a_norm[rank], mla_w_q_b[rank],
            mla_w_q_b_scale[rank], mla_w_kv_a[rank], mla_w_kv_a_scale[rank], mla_kv_a_norm[rank], mla_w_k[rank],
            mla_w_v[rank], mla_w_o[rank], mla_w_o_scale[rank], idx_w_q_b[rank], idx_w_k[rank],
            idx_k_norm_w[rank], idx_k_norm_b[rank], idx_w_weights[rank], idx_w_compress_gate[rank], idx_compress_ape[rank],
            hadamard[rank], mla_latent_cache[rank], index_raw_cache[rank], index_pool_cache[rank], index_pool_scale[rank],
            dsa_query[rank], attn_out[rank], index_q[rank], index_k[rank], index_scores[rank],
            selected_pools[rank], selected_valid[rank], router_gate_w[rank], router_gate_bias[rank], moe_w_gate_up[rank],
            moe_w_gate_up_scale[rank], moe_w_down[rank], moe_w_down_scale[rank], moe_shared_w_gate_up[rank], moe_shared_w_gate_up_scale[rank],
            moe_shared_w_down[rank], moe_shared_w_down_scale[rank], logit_row_indices[rank], head_hidden[rank], ws_x_hc_pair[rank],
            final_norm[rank], lm_head_weight[rank], logits[rank], tp_partial, tp_arrived,
            lane_x, lane_aux, lane_count, count_ready, x_ready,
            slot_out, return_ready, forward_epoch, rank, device=rank,
        )


if CACHE_TOKENS <= 0 or CACHE_TOKENS % C.BLOCK_SIZE:
    raise ValueError(f"--cache-tokens must be positive and aligned to {C.BLOCK_SIZE}")
if _shape_options.tokens is not None and _shape_options.tokens != T_MOE:
    raise ValueError(f"--tokens must match the resident owner capacity {T_MOE}")
_ci_rung_pinned = os.environ.get("GLM53_PREFILL_CI_PIN") == "2"
if (C.TP_SIZE, C.EP_SIZE, C.DP_SIZE) == (2, 2, 1) and _ci_rung_pinned:
    pass
elif (C.TP_SIZE, C.EP_SIZE, C.DP_SIZE) not in ((8, 8, 1), (16, 16, 1)):
    raise ValueError(
        "glm5_3_flash prefill_fwd requires TP8/EP8/DP1 or TP16/EP16/DP1; "
        f"got TP{C.TP_SIZE}/EP{C.EP_SIZE}/DP{C.DP_SIZE}. TP2/EP2 is accepted "
        "only for the marker-pinned CI subset (GLM53_PREFILL_CI_PIN)"
    )
if not 16 <= CAPACITY <= PREFILL_MAX_TOKENS or CAPACITY % 16:
    raise ValueError(f"--capacity must be 16..{PREFILL_MAX_TOKENS} and a multiple of 16, got {CAPACITY}")
if CAPACITY % 64:
    raise ValueError(f"--capacity must be a multiple of 64 for indexer_proj, got {CAPACITY}")
if not 1 <= MAX_SEQ_LEN <= FLASH.max_position_embeddings:
    raise ValueError(f"--max-seq-len must be 1..{FLASH.max_position_embeddings}, got {MAX_SEQ_LEN}")
_RESET_MLA_ELEMENTS = TABLE_MAX * BLOCK_SIZE * KV_LORA
_RESET_RAW_ELEMENTS = TABLE_MAX * BLOCK_SIZE * INDEX_STATE_WIDTH
_RESET_POOL_ELEMENTS = POOL_STRIDE * INDEX_DIM
_RESET_SCALE_ELEMENTS = POOL_STRIDE * POOL_SCALE_WIDTH
_RESET_CONV_ELEMENTS = STATE_LEN * CONV_DIM
_RESET_RECURRENT_ELEMENTS = LOCAL_KDA_H * KDA_DIM * KDA_DIM
_RESET_MLA_TILE = gcd(4096, _RESET_MLA_ELEMENTS)
_RESET_RAW_TILE = gcd(4096, _RESET_RAW_ELEMENTS)
_RESET_POOL_TILE = gcd(4096, _RESET_POOL_ELEMENTS)
_RESET_SCALE_TILE = gcd(BLOCK_SIZE // INDEX_KPOOL * POOL_SCALE_WIDTH, _RESET_SCALE_ELEMENTS)
_RESET_CONV_TILE = gcd(4096, _RESET_CONV_ELEMENTS)
_RESET_CONV_LANE_TILE = min(4, _RESET_CONV_ELEMENTS // _RESET_CONV_TILE)
_RESET_RECURRENT_TILE = gcd(4096, _RESET_RECURRENT_ELEMENTS)
_RESET_RECURRENT_LANE_TILE = min(4, _RESET_RECURRENT_ELEMENTS // _RESET_RECURRENT_TILE)
CONSTANT_STEP_INPUTS = {"kda_tri_incl", "kda_tri_strict", "kda_head_onehot", "kda_eye_wide"}

_FAMILY_PREFIX = {
    "kda": "self_attn.",
    "mla": "self_attn.",
    "indexer": "self_attn.indexer.",
    "mhc": "",
    "norm": "",
    "router": "mlp.",
}

GLM_WEIGHT_STACKS = (
    ("kda_w_q", "kda.w_q"),
    ("kda_w_k", "kda.w_k"),
    ("kda_w_v", "kda.w_v"),
    ("kda_conv_q", "kda.conv_q"),
    ("kda_conv_k", "kda.conv_k"),
    ("kda_conv_v", "kda.conv_v"),
    ("kda_w_f_a", "kda.w_f_a"),
    ("kda_w_f_b", "kda.w_f_b"),
    ("kda_w_g_a", "kda.w_g_a"),
    ("kda_w_g_b", "kda.w_g_b"),
    ("kda_w_b", "kda.w_b"),
    ("kda_dt_bias", "kda.dt_bias"),
    ("kda_a_log", "kda.a_log"),
    ("kda_o_norm", "kda.o_norm"),
    ("kda_w_o", "kda.w_o"),
    ("mla_w_q_a", "mla.w_q_a"),
    ("mla_w_q_a_scale", "mla.w_q_a_scale"),
    ("mla_q_a_norm", "mla.q_a_norm"),
    ("mla_w_q_b", "mla.w_q_b"),
    ("mla_w_q_b_scale", "mla.w_q_b_scale"),
    ("mla_w_kv_a", "mla.w_kv_a"),
    ("mla_w_kv_a_scale", "mla.w_kv_a_scale"),
    ("mla_kv_a_norm", "mla.kv_a_norm"),
    ("mla_w_k", "mla.w_k"),
    ("mla_w_v", "mla.w_v"),
    ("mla_w_o", "mla.w_o"),
    ("mla_w_o_scale", "mla.w_o_scale"),
    ("idx_w_q_b", "indexer.w_q_b"),
    ("idx_w_k", "indexer.w_k"),
    ("idx_k_norm_w", "indexer.k_norm_w"),
    ("idx_k_norm_b", "indexer.k_norm_b"),
    ("idx_w_weights", "indexer.w_weights"),
    ("idx_w_compress_gate", "indexer.w_compress_gate"),
    ("idx_compress_ape", "indexer.compress_ape"),
    ("mhc_attn_fn", "mhc.attn_fn"),
    ("mhc_attn_scale", "mhc.attn_scale"),
    ("mhc_attn_base", "mhc.attn_base"),
    ("mhc_ffn_fn", "mhc.ffn_fn"),
    ("mhc_ffn_scale", "mhc.ffn_scale"),
    ("mhc_ffn_base", "mhc.ffn_base"),
    ("norm_input_norm", "norm.input_norm"),
    ("norm_post_norm", "norm.post_norm"),
    ("router_gate_w", "router.gate_w"),
    ("router_gate_bias", "router.gate_bias"),
    ("moe_w_gate_up", "moe.w_gate_up"),
    ("moe_w_gate_up_scale", "moe.w_gate_up_scale"),
    ("moe_w_down", "moe.w_down"),
    ("moe_w_down_scale", "moe.w_down_scale"),
    ("moe_shared_w_gate_up", "moe.shared_w_gate_up"),
    ("moe_shared_w_gate_up_scale", "moe.shared_w_gate_up_scale"),
    ("moe_shared_w_down", "moe.shared_w_down"),
    ("moe_shared_w_down_scale", "moe.shared_w_down_scale"),
    ("dense_w_gate_up", "dense.w_gate_up"),
    ("dense_w_gate_up_scale", "dense.w_gate_up_scale"),
    ("dense_w_down", "dense.w_down"),
    ("dense_w_down_scale", "dense.w_down_scale"),
    ("hadamard", "hadamard"),
)
_PARAM_TO_STACK = dict(GLM_WEIGHT_STACKS)
_STACK_TO_PARAM = {stack: param for param, stack in GLM_WEIGHT_STACKS}
WORKSPACES = {
    "ws_attn_pre": ((CAPACITY, HC_MULT), torch.float32),
    "ws_attn_post": ((CAPACITY, HC_MULT), torch.float32),
    "ws_attn_res": ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    "ws_collapsed": ((CAPACITY, D), torch.bfloat16),
    "ws_normed": ((CAPACITY, D), torch.bfloat16),
    "ws_attn_partial": ((CAPACITY, D), torch.float32),
    "ws_sublayer_out": ((CAPACITY, D), torch.bfloat16),
    "ws_ffn_pre": ((CAPACITY, HC_MULT), torch.float32),
    "ws_ffn_post": ((CAPACITY, HC_MULT), torch.float32),
    "ws_ffn_res": ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    "ws_ffn_input": ((CAPACITY, D), torch.bfloat16),
    "ws_ffn_partial": ((CAPACITY, D), torch.float32),
    "ws_ffn_out": ((CAPACITY, D), torch.bfloat16),
    "ws_x_hc_a": ((CAPACITY, HC_MULT, D), torch.bfloat16),
    "ws_x_hc_b": ((CAPACITY, HC_MULT, D), torch.bfloat16),
    "kda_out": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "dsa_query": ((CAPACITY, LOCAL_H, QK_DIM), torch.bfloat16),
    "attn_out": ((CAPACITY, LOCAL_H, V_DIM), torch.bfloat16),
    "index_q": ((CAPACITY, INDEX_H, INDEX_DIM), torch.bfloat16),
    "index_k": ((CAPACITY, INDEX_DIM), torch.bfloat16),
    "index_scores": ((CAPACITY, POOL_AXIS), torch.float32),
    "selected_pools": ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    "selected_valid": ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    "mixed_qkv": ((CAPACITY, 3 * LOCAL_KDA_QKV_DIM), torch.bfloat16),
    "decay": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.float32),
    "beta": ((CAPACITY, BETA_PAD), torch.float32),
    "kda_q": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "kda_k": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "kda_v": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "dense_x_int8": ((CAPACITY, D), torch.int8),
}
OUTPUTS = {"head_hidden": ((LOGIT_ROWS, D), torch.bfloat16)}

_KDA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q", "self_attn.", ("q_proj.weight",)),
    ("w_k", "self_attn.", ("k_proj.weight",)),
    ("w_v", "self_attn.", ("v_proj.weight",)),
    ("conv_q", "self_attn.", ("q_conv1d.weight",)),
    ("conv_k", "self_attn.", ("k_conv1d.weight",)),
    ("conv_v", "self_attn.", ("v_conv1d.weight",)),
    ("w_f_a", "self_attn.", ("forget_gate.f_a_proj.weight", "f_a_proj.weight")),
    ("w_f_b", "self_attn.", ("forget_gate.f_b_proj.weight", "f_b_proj.weight")),
    ("dt_bias", "self_attn.", ("forget_gate.dt_bias", "dt_bias")),
    ("a_log", "self_attn.", ("forget_gate.A_log", "A_log")),
    ("w_g_a", "self_attn.", ("g_a_proj.weight",)),
    ("w_g_b", "self_attn.", ("g_b_proj.weight",)),
    ("w_b", "self_attn.", ("b_proj.weight",)),
    ("o_norm", "self_attn.", ("o_norm.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)
_MLA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q_a", "self_attn.", ("q_a_proj.weight",)),
    ("q_a_norm", "self_attn.", ("q_a_layernorm.weight",)),
    ("w_q_b", "self_attn.", ("q_b_proj.weight",)),
    ("w_kv_a", "self_attn.", ("kv_a_proj_with_mqa.weight",)),
    ("kv_a_norm", "self_attn.", ("kv_a_layernorm.weight",)),
    ("kv_b", "self_attn.", ("kv_b_proj.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)
_INDEXER_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("wq_b", "self_attn.indexer.", ("wq_b.weight",)),
    ("wk", "self_attn.indexer.", ("wk.weight",)),
    ("k_norm_w", "self_attn.indexer.", ("k_norm.weight",)),
    ("k_norm_b", "self_attn.indexer.", ("k_norm.bias",)),
    ("w_weights", "self_attn.indexer.", ("weights_proj.weight",)),
    ("w_compress_gate", "self_attn.indexer.", ("index_kpool_compress_gate",)),
    ("ape", "self_attn.indexer.", ("index_kpool_compress_ape",)),
)
_MHC_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attn_fn", "", ("attn_hc.fn", "hc_attn_fn")),
    ("attn_scale", "", ("attn_hc.scale", "hc_attn_scale")),
    ("attn_base", "", ("attn_hc.base", "hc_attn_base")),
    ("ffn_fn", "", ("ffn_hc.fn", "hc_ffn_fn")),
    ("ffn_scale", "", ("ffn_hc.scale", "hc_ffn_scale")),
    ("ffn_base", "", ("ffn_hc.base", "hc_ffn_base")),
)
_NORM_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("input_norm", "", ("input_layernorm.weight",)),
    ("post_norm", "", ("post_attention_layernorm.weight",)),
)
DEFAULT_DUMP_KEYS = (
    "x_hc_in", "attn_norm_in", "attn_out", "x_hc_mid",
    "ffn_norm_in", "ffn_out", "x_hc_out",
)
INTERNAL_DUMP_KEYS = (
    "attn_hc_pre", "attn_hc_post", "attn_hc_comb",
    "ffn_hc_pre", "ffn_hc_post", "ffn_hc_comb", "kda_q", "kda_k", "kda_v", "kda_g", "kda_beta", "kda_core",
    "kda_gate", "kda_o_normed", "mla_q_resid", "mla_k_latent", "mla_topk_indices",
    "router_topk_weights", "router_topk_indices", "moe_routed", "moe_shared",
)

MTP_SPEC_TOKENS = 3
VERIFY_X_H_C = os.environ.get("GLM53_PREFILL_VERIFY_XHC", "") == "1"
TRACE_DISPATCH = os.environ.get("GLM53_PREFILL_TRACE_DISPATCH", "") == "1"


@pl.jit(auto_scope=False)
def reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    slot_mask: pl.Scalar[pl.INT32],
):
    """Clear private working rows; mask -1 initializes all physical storage."""
    if slot_mask == -1:
        mla_flat = pl.reshape(mla_latent_cache, [DSA_NUM_LAYERS, LATENT_ROWS * KV_LORA])
        for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_mla"):
            layer = block // 4
            lane = block % 4
            mla_zero = pl.tile.full([1, _RESET_MLA_TILE], dtype=pl.BF16, value=0)
            for offset in pl.range(lane * _RESET_MLA_TILE, LATENT_ROWS * KV_LORA, 4 * _RESET_MLA_TILE):
                pl.tile.store(mla_zero, [layer, offset], mla_flat)
        raw_flat = pl.reshape(index_raw_cache, [DSA_NUM_LAYERS, LATENT_ROWS * INDEX_STATE_WIDTH])
        for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_raw"):
            layer = block // 4
            lane = block % 4
            raw_zero = pl.tile.full([1, _RESET_RAW_TILE], dtype=pl.FP32, value=0)
            for offset in pl.range(lane * _RESET_RAW_TILE, LATENT_ROWS * INDEX_STATE_WIDTH, 4 * _RESET_RAW_TILE):
                pl.tile.store(raw_zero, [layer, offset], raw_flat)
        pool_flat = pl.reshape(index_pool_cache, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * INDEX_DIM])
        for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_pool"):
            layer = block // 4
            lane = block % 4
            zero_i16 = pl.tile.full([1, _RESET_POOL_TILE], dtype=pl.INT16, value=0)
            pool_zero = pl.cast(zero_i16, pl.INT8)
            for offset in pl.range(lane * _RESET_POOL_TILE, POOL_TABLE_ROWS * INDEX_DIM, 4 * _RESET_POOL_TILE):
                pl.tile.store(pool_zero, [layer, offset], pool_flat)
        scale_flat = pl.reshape(index_pool_scale, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * POOL_SCALE_WIDTH])
        for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_scale"):
            layer = block // 4
            lane = block % 4
            scale_zero = pl.tile.full([1, _RESET_SCALE_TILE], dtype=pl.FP32, value=0)
            scale_elements = POOL_TABLE_ROWS * POOL_SCALE_WIDTH
            for offset in pl.range(lane * _RESET_SCALE_TILE, scale_elements, 4 * _RESET_SCALE_TILE):
                pl.tile.store(scale_zero, [layer, offset], scale_flat)
    conv_layer_tiles = KDA_ROWS * STATE_LEN * CONV_DIM // _RESET_CONV_TILE
    conv_slot_tiles = _RESET_CONV_ELEMENTS // _RESET_CONV_TILE
    conv_flat_rows = KDA_NUM_LAYERS * conv_layer_tiles
    conv_flat = pl.reshape(kda_conv_state, [conv_flat_rows, _RESET_CONV_TILE])
    for block in pl.spmd(KDA_NUM_LAYERS * _RESET_CONV_LANE_TILE, name_hint="reset_conv"):
        layer = block // _RESET_CONV_LANE_TILE
        lane = block % _RESET_CONV_LANE_TILE
        conv_zero = pl.tile.full([1, _RESET_CONV_TILE], dtype=pl.BF16, value=0)
        if slot_mask == -1:
            for chunk in pl.range(lane, conv_layer_tiles, _RESET_CONV_LANE_TILE):
                offset = layer * conv_layer_tiles + chunk
                pl.tile.store(conv_zero, [offset, 0], conv_flat)
        else:
            for slot in pl.range(BATCH):
                selected = (slot_mask >> pl.cast(slot, pl.INT32)) & 1
                if selected != 0:
                    for chunk in pl.range(lane, conv_slot_tiles, _RESET_CONV_LANE_TILE):
                        offset = layer * conv_layer_tiles + slot * conv_slot_tiles + chunk
                        pl.tile.store(conv_zero, [offset, 0], conv_flat)
    recurrent_layer_tiles = KDA_ROWS * LOCAL_KDA_H * KDA_DIM * KDA_DIM // _RESET_RECURRENT_TILE
    recurrent_slot_tiles = _RESET_RECURRENT_ELEMENTS // _RESET_RECURRENT_TILE
    recurrent_flat_rows = KDA_NUM_LAYERS * recurrent_layer_tiles
    recurrent_flat = pl.reshape(kda_recurrent_state, [recurrent_flat_rows, _RESET_RECURRENT_TILE])
    for block in pl.spmd(KDA_NUM_LAYERS * _RESET_RECURRENT_LANE_TILE, name_hint="reset_recurrent"):
        layer = block // _RESET_RECURRENT_LANE_TILE
        lane = block % _RESET_RECURRENT_LANE_TILE
        recurrent_zero = pl.tile.full([1, _RESET_RECURRENT_TILE], dtype=pl.FP32, value=0)
        if slot_mask == -1:
            for chunk in pl.range(lane, recurrent_layer_tiles, _RESET_RECURRENT_LANE_TILE):
                offset = layer * recurrent_layer_tiles + chunk
                pl.tile.store(recurrent_zero, [offset, 0], recurrent_flat)
        else:
            for slot in pl.range(BATCH):
                selected = (slot_mask >> pl.cast(slot, pl.INT32)) & 1
                if selected != 0:
                    for chunk in pl.range(lane, recurrent_slot_tiles, _RESET_RECURRENT_LANE_TILE):
                        offset = layer * recurrent_layer_tiles + slot * recurrent_slot_tiles + chunk
                        pl.tile.store(recurrent_zero, [offset, 0], recurrent_flat)
    return (mla_latent_cache, index_raw_cache, index_pool_cache, index_pool_scale, kda_conv_state, kda_recurrent_state)


@pl.jit.host
def l3_reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[
        pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]
    ],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    slot_mask: pl.Scalar[pl.INT32],
):
    """Clear the same selected request slots on every resident rank."""
    for rank in pl.unroll(RANKS):
        reset_request_caches(
            mla_latent_cache[rank],
            index_raw_cache[rank], index_pool_cache[rank], index_pool_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank],
            slot_mask, device=rank,
        )


@dataclass(frozen=True)
class _StackMember:
    """One kernel-side stack member: checkpoint spellings, view dtype and shard."""

    member: str
    suffixes: tuple[str, ...]
    dtype: torch.dtype
    transform: str = "identity"  # identity | transpose | split_k | split_v
    shard: str = "replicate"  # replicate | rows | columns


_KDA_MEMBERS = (
    _StackMember("w_q", ("q_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_k", ("k_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_v", ("v_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_q", ("q_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_k", ("k_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_v", ("v_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_f_a", ("forget_gate.f_a_proj.weight", "f_a_proj.weight"), torch.bfloat16),
    _StackMember("w_f_b", ("forget_gate.f_b_proj.weight", "f_b_proj.weight"), torch.bfloat16, shard="rows"),
    _StackMember("w_g_a", ("g_a_proj.weight",), torch.bfloat16),
    _StackMember("w_g_b", ("g_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_b", ("b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("dt_bias", ("forget_gate.dt_bias", "dt_bias"), torch.float32, shard="rows"),
    _StackMember("a_log", ("forget_gate.A_log", "A_log"), torch.float32, shard="rows"),
    _StackMember("o_norm", ("o_norm.weight",), torch.bfloat16),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)
_MLA_MEMBERS = (
    _StackMember("w_q_a", ("q_a_proj.weight",), torch.bfloat16),
    _StackMember("q_a_norm", ("q_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_q_b", ("q_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_kv_a", ("kv_a_proj_with_mqa.weight",), torch.bfloat16),
    _StackMember("kv_a_norm", ("kv_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_k", ("kv_b_proj.weight",), torch.bfloat16, transform="split_k"),
    _StackMember("w_v", ("kv_b_proj.weight",), torch.bfloat16, transform="split_v"),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)
_INDEXER_MEMBERS = (
    _StackMember("w_q_b", ("wq_b.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_k", ("wk.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("k_norm_w", ("k_norm.weight",), torch.bfloat16),
    _StackMember("k_norm_b", ("k_norm.bias",), torch.bfloat16),
    _StackMember("w_weights", ("weights_proj.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_compress_gate", ("index_kpool_compress_gate",), torch.bfloat16, transform="transpose"),
    _StackMember("compress_ape", ("index_kpool_compress_ape",), torch.bfloat16),
)
_MHC_MEMBERS = (
    _StackMember("attn_fn", ("attn_hc.fn", "hc_attn_fn"), torch.bfloat16),
    _StackMember("attn_scale", ("attn_hc.scale", "hc_attn_scale"), torch.float32),
    _StackMember("attn_base", ("attn_hc.base", "hc_attn_base"), torch.float32),
    _StackMember("ffn_fn", ("ffn_hc.fn", "hc_ffn_fn"), torch.bfloat16),
    _StackMember("ffn_scale", ("ffn_hc.scale", "hc_ffn_scale"), torch.float32),
    _StackMember("ffn_base", ("ffn_hc.base", "hc_ffn_base"), torch.float32),
)
_NORM_MEMBERS = (
    _StackMember("input_norm", ("input_layernorm.weight",), torch.bfloat16),
    _StackMember("post_norm", ("post_attention_layernorm.weight",), torch.bfloat16),
)
_ROUTER_MEMBERS = (
    _StackMember("gate_w", ("gate.weight",), torch.bfloat16),
    _StackMember("gate_bias", ("gate.e_score_correction_bias", "e_score_correction_bias"), torch.float32),
)
_FAMILY_MEMBERS = {
    "kda": _KDA_MEMBERS,
    "mla": _MLA_MEMBERS,
    "indexer": _INDEXER_MEMBERS,
    "mhc": _MHC_MEMBERS,
    "norm": _NORM_MEMBERS,
    "router": _ROUTER_MEMBERS,
}
_TEXT_MODEL_ROOTS = ("model.language_model.", "model.")


def _source_shape(model: Glm53FlashConfig, family: str, member: str) -> tuple[int, ...]:
    """The checkpoint-side shape a member is read from (before any transform)."""
    hidden = model.hidden_size
    qkv = model.linear_num_heads * model.linear_head_dim
    if family == "kda":
        shapes = {
            "w_q": (qkv, hidden),
            "w_k": (qkv, hidden),
            "w_v": (qkv, hidden),
            "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
            "w_f_a": (model.linear_head_dim, hidden),
            "w_f_b": (qkv, model.linear_head_dim),
            "w_g_a": (model.linear_head_dim, hidden),
            "w_g_b": (qkv, model.linear_head_dim),
            "w_b": (model.linear_num_heads, hidden),
            "dt_bias": (qkv,),
            "a_log": (model.linear_num_heads,),
            "o_norm": (model.linear_head_dim,),
            "w_o": (hidden, qkv),
        }
    elif family == "mla":
        shapes = {
            "w_q_a": (model.q_lora_rank, hidden),
            "q_a_norm": (model.q_lora_rank,),
            "w_q_b": (model.num_attention_heads * model.qk_head_dim, model.q_lora_rank),
            "w_kv_a": (model.kv_lora_rank, hidden),
            "kv_a_norm": (model.kv_lora_rank,),
            "w_k": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_v": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_o": (hidden, model.num_attention_heads * model.v_head_dim),
        }
    elif family == "indexer":
        shapes = {
            "w_q_b": (model.index_n_heads * model.index_head_dim, model.q_lora_rank),
            "w_k": (model.index_head_dim, hidden),
            "k_norm_w": (model.index_head_dim,),
            "k_norm_b": (model.index_head_dim,),
            "w_weights": (model.index_n_heads, hidden),
            "w_compress_gate": (model.index_head_dim, hidden),
            "compress_ape": (model.index_kpool, model.index_head_dim),
        }
    elif family == "mhc":
        shapes = {
            "attn_fn": (model.mix_hc, model.hc_dim),
            "attn_scale": (3,),
            "attn_base": (model.mix_hc,),
            "ffn_fn": (model.mix_hc, model.hc_dim),
            "ffn_scale": (3,),
            "ffn_base": (model.mix_hc,),
        }
    elif family == "norm":
        shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
    else:
        shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
    return shapes[member]


@dataclass(frozen=True)
class StackPlan:
    """One device-resident stacked tensor as the runner must allocate it."""

    name: str  # e.g. "kda.w_q"
    family: str  # kda | mla | indexer | mhc | norm | router | moe | dense | mtp | single
    layers: tuple[int, ...]  # source layer ids on the stack axis, ascending
    shape: tuple[int, ...]  # per-rank stacked shape (leading axis = family ordinal)
    dtype: torch.dtype

    @property
    def bytes(self) -> int:
        count = math.prod(self.shape)
        return count * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class RankPlan:
    """Pure-metadata view of one rank's resident weights (no payload reads)."""

    rank: int
    tp_rank: int
    ep_rank: int
    local_expert_ids: tuple[int, ...]
    stacks: tuple[StackPlan, ...]
    device_bytes: int
    host_tables: tuple[str, ...]  # ("embed_tokens", "lm_head"), host-side aliases included


def _fused_ffn_at(
    tensors: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    roles: tuple[str, str, str, str, str],
    *,
    expert: int | None,
    intermediate: int,
    hidden: int,
    channels: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_fused_ffn`` carried over an explicit geometry: hidden and intermediate
    come from the staging model instead of the module constants, so scaled
    fixtures load through the same read/fuse pipeline as the deployment."""
    fmt = {"e": expert} if expert is not None else {}
    fused_role, fused_down_role, gate_role, up_role, down_role = roles
    fused_name = _try_find(tensors, _prefixed(prefixes, ROLES[fused_role], **fmt))
    down_name = _find(tensors, _prefixed(prefixes, ROLES[fused_down_role] + ROLES[down_role], **fmt))
    if fused_name is not None:
        if _weight_shape(tensors, fused_name) != (2 * intermediate, hidden):
            raise ValueError(f"{fused_name} must have shape {(2 * intermediate, hidden)}")
        if channels == slice(None):
            gate_up, gate_up_scale = _read_quant(tensors, fused_name)
        else:
            gate, gate_scale = _read_quant(tensors, fused_name, channels)
            up_rows = slice(intermediate + channels.start, intermediate + channels.stop)
            up, up_scale = _read_quant(tensors, fused_name, up_rows)
            gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    else:
        gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
        up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
        if _weight_shape(tensors, gate_name) != (intermediate, hidden) or _weight_shape(tensors, up_name) != (
            intermediate,
            hidden,
        ):
            raise ValueError(f"{gate_name} and {up_name} must have shape {(intermediate, hidden)}")
        gate, gate_scale = _read_quant(tensors, gate_name, channels)
        up, up_scale = _read_quant(tensors, up_name, channels)
        gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    if _weight_shape(tensors, down_name) != (hidden, intermediate):
        raise ValueError(f"{down_name} must have shape {(hidden, intermediate)}")
    down, down_scale = _read_quant(tensors, down_name, cols=channels)
    return gate_up, gate_up_scale, down, down_scale


def _ffn_roles(kind: str) -> tuple[str, str, str, str, str]:
    return {
        "routed": ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"),
        "shared": ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down"),
        "dense": ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down"),
    }[kind]


class ResidentPrefillWeights:
    """Plan and stage the resident prefill weights for the TP/EP ranks.

    Construction goes through :meth:`load`, which reads headers only: it
    resolves every stack member's source name (both layer-root spellings, fused
    or separate gate/up, nested or flat mHC/KDA-gate spellings), validates the
    header shapes against the geometry, and builds the full stack plan. Payloads
    are yielded by :meth:`iter_rank_uploads`: layered stacks arrive one layer
    slice at a time, while dense stacks and unstacked residents arrive whole.

    Rank mapping: ranks are EP ranks; the TP rank is ``rank % tp_size``. In the
    deployment topology (TP16/EP16/DP1) the two coincide with the rank id, and
    TP1/EP16 collapses to tp_rank 0 for every rank.
    """

    def __init__(
        self,
        checkpoint,
        *,
        tp_size: int,
        ep_size: int,
        model: Glm53FlashConfig | None = None,
        mla_mode: str = "bf16",
    ):
        model = FLASH if model is None else model
        if isinstance(checkpoint, Checkpoint) and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; pass model only for fixtures")
        if tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
        if ep_size not in SUPPORTED_EP_SIZES:
            raise ValueError(f"ep_size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
        if ep_size % tp_size:
            raise ValueError(f"EP{ep_size} must be divisible by TP{tp_size}")
        if model.num_attention_heads % tp_size:
            raise ValueError(f"{model.num_attention_heads} MLA heads do not divide TP{tp_size}")
        if model.linear_num_heads % tp_size:
            raise ValueError(f"{model.linear_num_heads} KDA heads do not divide TP{tp_size}")
        if model.intermediate_size % tp_size:
            raise ValueError(f"{model.intermediate_size} dense channels do not divide TP{tp_size}")
        if model.n_routed_experts % ep_size:
            raise ValueError(f"{model.n_routed_experts} routed experts do not divide EP{ep_size}")
        if mla_mode not in ("bf16", "native"):
            raise ValueError("mla_mode must be bf16 or native")
        self.mla_mode = mla_mode
        self._family_members = dict(_FAMILY_MEMBERS)
        if mla_mode == "native":
            native = []
            for spec in _MLA_MEMBERS:
                if spec.member in ("w_q_a", "w_q_b", "w_kv_a", "w_o"):
                    native.append(dataclasses.replace(spec, dtype=torch.int8))
                    native.append(dataclasses.replace(spec, member=spec.member + "_scale", dtype=torch.float32))
                else:
                    native.append(spec)
            self._family_members["mla"] = tuple(native)
        self.checkpoint = checkpoint
        self.model = model
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ranks = ep_size
        self.local_h = model.num_attention_heads // tp_size
        self.local_kda_h = model.linear_num_heads // tp_size
        self.local_kda_qkv = model.linear_num_heads * model.linear_head_dim // tp_size
        self.n_local_experts = model.n_routed_experts // ep_size
        self.local_dense_inter = model.intermediate_size // tp_size
        mtp = model.mtp_layer_id
        self._family_layers = {
            "kda": tuple(model.kda_layer_ids),
            "mla": tuple(model.indexer_layer_ids),
            "indexer": tuple(model.indexer_layer_ids),
            "mhc": tuple(range(model.num_hidden_layers)),
            "norm": tuple(range(model.num_checkpoint_layers)),
            "router": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "moe": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "dense": tuple(range(model.first_k_dense_replace)),
        }
        self._mtp_layer = mtp
        self._lm_head_cache: torch.Tensor | None = None
        self._source: dict[tuple[str, int], str] = {}
        self.stacks: dict[str, StackPlan] = {}

    # -- construction ----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        checkpoint,
        *,
        tp_size: int = 16,
        ep_size: int = 16,
        model: Glm53FlashConfig | None = None,
        mla_mode: str = "bf16",
    ) -> "ResidentPrefillWeights":
        """Build the full stack plan from checkpoint headers, reading no payloads."""
        self = cls(checkpoint, tp_size=tp_size, ep_size=ep_size, model=model, mla_mode=mla_mode)
        self._resolve_sources()
        self._build_stacks()
        return self

    def _resolve(self, candidates: Sequence[str], what: str) -> str:
        for name in candidates:
            if name in self.checkpoint:
                return name
        raise ValueError(f"checkpoint is missing {what} (tried {list(candidates)})")

    def _label(self, name: str) -> str:
        description = getattr(self.checkpoint, "description", None)
        label = None if description is None else description.get(name)
        if label not in ("FLOAT", "W8A8_DYNAMIC"):
            raise ValueError(f"{name} has unsupported quantization label {label!r}")
        return label

    def _resolve_sources(self) -> None:
        """Resolve and header-validate every stack member's source tensor."""
        model = self.model
        for family, members in self._family_members.items():
            prefix = _FAMILY_PREFIX[family]
            for layer in self._family_layers[family]:
                for spec in members:
                    candidates = [
                        root + f"{layer}." + prefix + suffix
                        for root in TEXT_LAYER_ROOTS
                        for suffix in spec.suffixes
                    ]
                    name = self._resolve(candidates, f"{family}.{spec.member} on layer {layer}")
                    self._source[(f"{family}.{spec.member}", layer)] = name
                    member = spec.member.removesuffix("_scale") if family == "mla" else spec.member
                    native = (
                        family == "mla"
                        and self.mla_mode == "native"
                        and member in ("w_q_a", "w_q_b", "w_kv_a", "w_o")
                    )
                    if native and self._label(name) != "W8A8_DYNAMIC":
                        raise ValueError(f"native MLA projection must be W8A8_DYNAMIC: {name}")
                    dtype = torch.bfloat16 if native else spec.dtype
                    self._validate_member_header(name, _source_shape(model, family, member), dtype)
        hidden = model.hidden_size
        for member, suffix, shape in (
            ("eh_proj", "eh_proj.weight", (hidden, 2 * hidden)),
            ("enorm", "enorm.weight", (hidden,)),
            ("hnorm", "hnorm.weight", (hidden,)),
            ("final_norm", "shared_head.norm.weight", (hidden,)),
        ):
            candidates = [root + f"{self._mtp_layer}." + suffix for root in TEXT_LAYER_ROOTS]
            name = self._resolve(candidates, f"mtp.{member}")
            self._source[(f"mtp.{member}", self._mtp_layer)] = name
            self._validate_member_header(name, shape, torch.bfloat16)
        self._source["norm.final_norm", -1] = self._resolve(
            [root + "norm.weight" for root in _TEXT_MODEL_ROOTS], "norm.final_norm",
        )
        self._validate_member_header(self._source[("norm.final_norm", -1)], (hidden,), torch.bfloat16)
        self._embed_name = self._resolve([root + "embed_tokens.weight" for root in _TEXT_MODEL_ROOTS], "embed_tokens")
        self._lm_head_name = self._resolve(("lm_head.weight",), "lm_head")

    def _validate_member_header(self, name: str, expected_shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Header-only shape and label gate; skipped for header-less fixture views.

        ``Checkpoint`` validates dtype at read time (``_float`` accepts only
        BF16/FP32 storage, ``quant`` only INT8); here the shape is checked from
        the shard header so a layout mismatch fails before any payload moves.
        """
        shape_of = getattr(self.checkpoint, "shape", None)
        if shape_of is None:
            return
        shape = shape_of(name)
        if tuple(shape) != expected_shape:
            raise ValueError(f"{name} has shape {tuple(shape)}, expected {expected_shape}")
        label = self._label(name)
        if label == "W8A8_DYNAMIC" and dtype != torch.bfloat16:
            # Only the MLA projections are INT8 in storage; every other member
            # must be FLOAT, and dequantizing one would silently change its ABI.
            raise ValueError(f"{name} is {label}; only BF16 members may dequantize, view is {dtype}")

    def _build_stacks(self) -> None:
        """Compute every stack's per-rank shape from the geometry alone."""
        model = self.model
        hidden = model.hidden_size
        stacks: dict[str, StackPlan] = {}
        for family, members in self._family_members.items():
            layers = self._family_layers[family]
            for spec in members:
                name = f"{family}.{spec.member}"
                shape = (len(layers), *self._member_shape(family, spec))
                stacks[name] = StackPlan(name, family, layers, shape, spec.dtype)
        moe_layers = self._family_layers["moe"]
        inter, local_experts = model.moe_intermediate_size, self.n_local_experts
        for member, shape, dtype in (
            ("w_gate_up", (local_experts, 2 * inter, hidden), torch.int8),
            ("w_gate_up_scale", (local_experts, 2 * inter), torch.float32),
            ("w_down", (local_experts, hidden, inter), torch.int8),
            ("w_down_scale", (local_experts, hidden), torch.float32),
            ("shared_w_gate_up", (2 * inter, hidden), torch.int8),
            ("shared_w_gate_up_scale", (2 * inter,), torch.float32),
            ("shared_w_down", (hidden, inter), torch.int8),
            ("shared_w_down_scale", (hidden,), torch.float32),
        ):
            name = f"moe.{member}"
            stacks[name] = StackPlan(name, "moe", moe_layers, (len(moe_layers), *shape), dtype)
        dense_layers = self._family_layers["dense"]
        local = self.local_dense_inter
        for member, shape, dtype in (
            ("w_gate_up", (2 * local, hidden), torch.int8),
            ("w_gate_up_scale", (2 * local,), torch.float32),
            ("w_down", (hidden, local), torch.int8),
            ("w_down_scale", (hidden,), torch.float32),
        ):
            name = f"dense.{member}"
            stacks[name] = StackPlan(name, "dense", dense_layers, (len(dense_layers), *shape), dtype)
        for name, family, shape in (
            ("norm.final_norm", "single", (hidden,)),
            ("mtp.eh_proj", "mtp", (hidden, 2 * hidden)),
            ("mtp.enorm", "mtp", (hidden,)),
            ("mtp.hnorm", "mtp", (hidden,)),
            ("mtp.final_norm", "mtp", (hidden,)),
            ("hadamard", "single", (model.index_head_dim, model.index_head_dim)),
        ):
            layers = (self._mtp_layer,) if family == "mtp" else ()
            stacks[name] = StackPlan(name, family, layers, shape, torch.bfloat16)
        self.stacks = stacks

    def _member_shape(self, family: str, spec: _StackMember) -> tuple[int, ...]:
        """One layer's rank-local shape for a stack member."""
        model = self.model
        hidden, qkv = model.hidden_size, self.local_kda_qkv
        if family == "kda":
            shapes = {
                "w_q": (qkv, hidden),
                "w_k": (qkv, hidden),
                "w_v": (qkv, hidden),
                "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
                "w_f_a": (model.linear_head_dim, hidden),
                "w_f_b": (qkv, model.linear_head_dim),
                "w_g_a": (model.linear_head_dim, hidden),
                "w_g_b": (qkv, model.linear_head_dim),
                "w_b": (self.local_kda_h, hidden),
                "dt_bias": (qkv,),
                "a_log": (self.local_kda_h,),
                "o_norm": (model.linear_head_dim,),
                "w_o": (hidden, qkv),
            }
        elif family == "mla":
            shapes = {
                "w_q_a": (model.q_lora_rank, hidden),
                "q_a_norm": (model.q_lora_rank,),
                "w_q_b": (self.local_h * model.qk_head_dim, model.q_lora_rank),
                "w_kv_a": (model.kv_lora_rank, hidden),
                "kv_a_norm": (model.kv_lora_rank,),
                "w_k": (self.local_h, model.qk_head_dim, model.kv_lora_rank),
                "w_v": (self.local_h, model.v_head_dim, model.kv_lora_rank),
                "w_o": (hidden, self.local_h * model.v_head_dim),
            }
        elif family == "indexer":
            shapes = {
                "w_q_b": (model.q_lora_rank, model.index_n_heads * model.index_head_dim),
                "w_k": (hidden, model.index_head_dim),
                "k_norm_w": (model.index_head_dim,),
                "k_norm_b": (model.index_head_dim,),
                "w_weights": (hidden, model.index_n_heads),
                "w_compress_gate": (hidden, model.index_head_dim),
                "compress_ape": (model.index_kpool, model.index_head_dim),
            }
        elif family == "mhc":
            shapes = {
                "attn_fn": (model.mix_hc, model.hc_dim),
                "attn_scale": (3,),
                "attn_base": (model.mix_hc,),
                "ffn_fn": (model.mix_hc, model.hc_dim),
                "ffn_scale": (3,),
                "ffn_base": (model.mix_hc,),
            }
        elif family == "norm":
            shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
        else:
            shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
        if family == "mla" and spec.member.endswith("_scale"):
            return (shapes[spec.member.removesuffix("_scale")][0],)
        return shapes[spec.member]

    # -- public interface --------------------------------------------------------------

    @property
    def tensor_layouts(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Device stack name -> stacked shape/dtype, for ``runtime.alloc_tensor``."""
        return {plan.name: (plan.shape, plan.dtype) for plan in self.stacks.values()}

    @property
    def bytes_per_rank(self) -> int:
        """Device-resident bytes every rank allocates (identical across ranks)."""
        return sum(plan.bytes for plan in self.stacks.values())

    @property
    def host_bytes(self) -> int:
        """Host-side table bytes: the full LM head plus the embedding table.

        The backbone head is cached and embedding rows are gathered per prompt.
        MTP rows and head shards are read separately without retaining full tables.
        """
        return 2 * self.model.vocab_size * self.model.hidden_size * 2

    def plan(self, rank: int) -> RankPlan:
        """The rank's slice of every device stack plus byte totals; no payload reads."""
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        first = rank * self.n_local_experts
        return RankPlan(
            rank=rank, tp_rank=rank % self.tp_size, ep_rank=rank,
            local_expert_ids=tuple(range(first, first + self.n_local_experts)),
            stacks=tuple(self.stacks.values()), device_bytes=self.bytes_per_rank,
            host_tables=("embed_tokens", "lm_head"),
        )

    def lookup_rows(self, token_ids: torch.Tensor, *, mtp: bool = False) -> torch.Tensor:
        """Gather one prompt's BF16 embedding rows from the host table."""
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("token ids must have an integer dtype")
        if bool(((ids < 0) | (ids >= self.model.vocab_size)).any()):
            raise ValueError("token id is outside the checkpoint vocabulary")
        name = self._embed_name
        if mtp:
            name = self._resolve(
                [root + f"{self._mtp_layer}.embed_tokens.weight" for root in TEXT_LAYER_ROOTS],
                "MTP embedding",
            )
        return gather_rows(self.checkpoint, name, ids).to(torch.bfloat16)

    def mtp_head_shard(self, rank: int) -> torch.Tensor:
        """Read the MTP-owned vocabulary shard without assuming tied tables."""
        if rank not in range(self.tp_size):
            raise ValueError("MTP head rank is outside the TP group")
        name = self._resolve(
            [root + f"{self._mtp_layer}.shared_head.head.weight" for root in TEXT_LAYER_ROOTS],
            "MTP shared head",
        )
        width = self.model.vocab_size // self.tp_size
        return self.checkpoint.read(name, slice(rank * width, (rank + 1) * width)).bfloat16()

    def lm_head(self) -> torch.Tensor:
        """The full ``[vocab, hidden]`` BF16 head matrix, read once and cached."""
        if self._lm_head_cache is None:
            head = self.checkpoint.read(self._lm_head_name)
            expected = (self.model.vocab_size, self.model.hidden_size)
            if tuple(head.shape) != expected:
                raise ValueError(f"lm_head has shape {tuple(head.shape)}, expected {expected}")
            self._lm_head_cache = head.to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    # -- materialization ----------------------------------------------------------------

    def _read_source(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Authority-driven read: FLOAT casts, W8A8_DYNAMIC dequantizes to BF16.

        Dequant is one FP32 scale multiply and one BF16 rounding, so the INT8
        storage of the four MLA projections never crosses the host/device
        boundary; a non-zero ``weight_offset`` would invalidate the symmetric
        assumption and is rejected instead of silently dropped.
        """
        label = self._label(name)
        if label == "FLOAT":
            return self.checkpoint.read(name).to(dtype)
        if dtype != torch.bfloat16:
            raise ValueError(f"{name} is {label}; only BF16 consumers may dequantize, asked for {dtype}")
        weight, scale = _read_quant(self.checkpoint, name)
        product = weight.to(torch.float32) * scale.reshape(-1, *([1] * (weight.dim() - 1)))
        return product.to(torch.bfloat16)

    def _shard(self, tensor: torch.Tensor, shard: str, tp_rank: int) -> torch.Tensor:
        """Keep one TP rank's row/column block (heads sit contiguously)."""
        if shard == "replicate":
            return tensor
        axis = 0 if shard == "rows" else 1
        size = tensor.shape[axis]
        if size % self.tp_size:
            raise ValueError(f"{shard} axis {size} does not divide TP{self.tp_size}")
        width, start = size // self.tp_size, tp_rank * (size // self.tp_size)
        return tensor[start : start + width] if axis == 0 else tensor[:, start : start + width]

    def _member_tensor(self, family: str, spec: _StackMember, layer: int, tp_rank: int) -> torch.Tensor:
        name = self._source[(f"{family}.{spec.member}", layer)]
        value = self._read_source(name, spec.dtype)
        if spec.transform == "transpose":
            value = value.t()
        return self._shard(value, spec.shard, tp_rank).contiguous()

    def _mla_layer(self, layer: int, tp_rank: int) -> dict[str, torch.Tensor]:
        """One MLA layer's eight rank slices; ``kv_b_proj`` is read once."""
        out: dict[str, torch.Tensor] = {}
        kv_b = None
        for spec in self._family_members["mla"]:
            if spec.member.endswith("_scale"):
                continue
            if spec.dtype == torch.int8:
                name = self._source[(f"mla.{spec.member}", layer)]
                weight, scale = _read_quant(self.checkpoint, name)
                out[spec.member] = self._shard(weight, spec.shard, tp_rank).contiguous()
                if spec.shard == "rows":
                    scale = self._shard(scale, "rows", tp_rank)
                out[spec.member + "_scale"] = scale.contiguous()
            elif spec.transform in ("split_k", "split_v"):
                if kv_b is None:
                    kv_b = self._read_source(self._source[("mla.w_k", layer)], torch.bfloat16)
                (key, value) = split_kv_b(
                    kv_b, tp_rank, self.tp_size, qk_dim=self.model.qk_head_dim, v_dim=self.model.v_head_dim,
                    kv_lora=self.model.kv_lora_rank,
                )
                out[spec.member] = key if spec.transform == "split_k" else value
            else:
                out[spec.member] = self._member_tensor("mla", spec, layer, tp_rank)
        return out

    def _stack_family(self, family: str, tp_rank: int) -> dict[str, torch.Tensor]:
        """Materialize one non-FFN family as its stacked rank tensor per member."""
        collected: dict[str, list[torch.Tensor]] = {spec.member: [] for spec in self._family_members[family]}
        for layer in self._family_layers[family]:
            per_layer = (
                self._mla_layer(layer, tp_rank)
                if family == "mla"
                else {
                    spec.member: self._member_tensor(family, spec, layer, tp_rank)
                    for spec in self._family_members[family]
                }
            )
            for member, tensor in per_layer.items():
                collected[member].append(tensor)
        return {member: torch.stack(slices) for member, slices in collected.items()}

    def _stack_dense(self, tp_rank: int) -> dict[str, torch.Tensor]:
        """The dense layers' TP-sliced MLPs, stacked on the family axis."""
        model = self.model
        channels = slice(tp_rank * self.local_dense_inter, (tp_rank + 1) * self.local_dense_inter)
        shards = [
            _fused_ffn_at(
                self.checkpoint,
                layer_prefixes(layer),
                _ffn_roles("dense"),
                expert=None,
                intermediate=model.intermediate_size,
                hidden=model.hidden_size,
                channels=channels,
            )
            for layer in self._family_layers["dense"]
        ]
        return {
            "w_gate_up": torch.stack([shard[0] for shard in shards]),
            "w_gate_up_scale": torch.stack([shard[1] for shard in shards]),
            "w_down": torch.stack([shard[2] for shard in shards]),
            "w_down_scale": torch.stack([shard[3] for shard in shards]),
        }

    def _moe_layer(self, layer: int, ep_rank: int) -> dict[str, torch.Tensor]:
        """One MoE layer's local experts and replicated shared expert."""
        model = self.model
        prefixes = layer_prefixes(layer)
        first = ep_rank * self.n_local_experts
        stacked = None
        for expert in range(first, first + self.n_local_experts):
            values = _fused_ffn_at(
                self.checkpoint,
                prefixes, _ffn_roles("routed"), expert=expert, intermediate=model.moe_intermediate_size,
                hidden=model.hidden_size,
            )
            if stacked is None:
                stacked = [torch.empty((self.n_local_experts, *value.shape), dtype=value.dtype) for value in values]
            for target, value in zip(stacked, values):
                target[expert - first].copy_(value)
        shared = _fused_ffn_at(
            self.checkpoint,
            prefixes, _ffn_roles("shared"), expert=None, intermediate=model.moe_intermediate_size,
            hidden=model.hidden_size,
        )
        return {
            "w_gate_up": stacked[0],
            "w_gate_up_scale": stacked[1],
            "w_down": stacked[2],
            "w_down_scale": stacked[3],
            "shared_w_gate_up": shared[0],
            "shared_w_gate_up_scale": shared[1],
            "shared_w_down": shared[2],
            "shared_w_down_scale": shared[3],
        }

    def _single_tensors(self) -> dict[str, torch.Tensor]:
        """The unstacked residents: final norm, MTP extras, the Hadamard table."""

        model = self.model
        hadamard = sylvester_hadamard(model.index_head_dim)
        return {
            "norm.final_norm": self._read_source(self._source[("norm.final_norm", -1)], torch.bfloat16),
            "mtp.eh_proj": self._read_source(self._source[("mtp.eh_proj", self._mtp_layer)], torch.bfloat16),
            "mtp.enorm": self._read_source(self._source[("mtp.enorm", self._mtp_layer)], torch.bfloat16),
            "mtp.hnorm": self._read_source(self._source[("mtp.hnorm", self._mtp_layer)], torch.bfloat16),
            "mtp.final_norm": self._read_source(
                self._source[("mtp.final_norm", self._mtp_layer)], torch.bfloat16
            ),
            "hadamard": hadamard.contiguous(),
        }

    def iter_rank_uploads(self, rank: int):
        """Yield ``(stack_name, cpu_tensor, byte_offset)`` upload payloads.

        Layered norm, mHC, router, KDA, MLA, Indexer and MoE stacks arrive one
        layer slice at a time. Dense stacks and unstacked residents each arrive
        as a whole payload at byte offset zero.
        Consumers own device allocation and transfers. Shared INT8 matrices use
        NZ bytes; the other stacks retain their logical row-major layout.
        """
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        tp_rank = rank % self.tp_size

        def payload(name: str, data: torch.Tensor, layer_index: int | None = None):
            plan = self.stacks[name]
            if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
                data = pack_int8_nz(data)
            data = data.contiguous()
            expected = plan.shape if layer_index is None else plan.shape[1:]
            if tuple(data.shape) != expected or data.dtype != plan.dtype:
                raise ValueError(
                    f"staged {name} is {tuple(data.shape)}/{data.dtype}, planned {expected}/{plan.dtype}"
                )
            offset = 0 if layer_index is None else layer_index * data.numel() * data.element_size()
            return name, data, offset

        for name, value in self._single_tensors().items():
            yield payload(name, value)
        for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
            for index, layer in enumerate(self._family_layers[family]):
                per_layer = (
                    self._mla_layer(layer, tp_rank)
                    if family == "mla"
                    else {
                        spec.member: self._member_tensor(family, spec, layer, tp_rank)
                        for spec in self._family_members[family]
                    }
                )
                for member, value in per_layer.items():
                    yield payload(f"{family}.{member}", value, layer_index=index)
        for member, stack in self._stack_dense(tp_rank).items():
            yield payload(f"dense.{member}", stack)
        for index, layer in enumerate(self._family_layers["moe"]):
            for member, value in self._moe_layer(layer, rank).items():
                yield payload(f"moe.{member}", value, layer_index=index)


def prepare_step_inputs(
    weights: ResidentPrefillWeights,
    sequences: list[list[int]],
    *,
    kv_seq_lens: torch.Tensor | None = None,
    visual_features: dict[int, torch.Tensor] | None = None,
    cache_pages: CacheTables | None = None,
    mtp: bool = False,
    kda_state_rows: torch.Tensor | None = None,
) -> dict:
    """Pack only new query tokens into stable request-owned cache slots."""
    if len(sequences) != BATCH:
        raise ValueError(f"the packed batch has {BATCH} request slots, got {len(sequences)}")
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)
    previous = torch.zeros(BATCH, dtype=torch.int32) if kv_seq_lens is None else kv_seq_lens.to(torch.int32)
    if previous.shape != (BATCH,) or bool((previous < 0).any()):
        raise ValueError("kv_seq_lens must contain one nonnegative prefix length per request slot")
    if bool((previous.to(torch.int64) + lengths > MAX_SEQ_LEN).any()):
        raise ValueError(f"request context exceeds the compiled maximum {MAX_SEQ_LEN}")
    total = int(lengths.sum().item())
    if total > CAPACITY:
        raise ValueError(f"{total} new query tokens exceed the compiled capacity {CAPACITY}")

    embedding_rows = torch.zeros(CAPACITY, D, dtype=torch.bfloat16)
    if total:
        ids = torch.tensor([token for seq in sequences for token in seq], dtype=torch.int64)
        embedding_rows[:total] = weights.lookup_rows(ids, mtp=mtp)
    if visual_features:

        offset = 0
        if any(type(slot) is not int or slot not in range(BATCH) for slot in visual_features):
            raise ValueError("visual feature keys must be request slots")
        for slot, sequence in enumerate(sequences):
            if slot in visual_features:
                features = visual_features[slot]
                indices = torch.tensor(
                    [row + offset for (row, token) in enumerate(sequence) if token == IMAGE_TOKEN_ID],
                )
                if features.dtype != torch.bfloat16 or tuple(features.shape) != (indices.numel(), D):
                    raise ValueError("visual features must be BF16 [image_placeholder_count, hidden_size]")
                if not torch.isfinite(features).all():
                    raise ValueError("visual features must be finite")
                embedding_rows[indices.long()] = features.cpu()
            offset += len(sequence)
    query_start_loc = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), lengths]), 0).to(torch.int32)
    if cache_pages is None:
        if CACHE_TOKENS < BATCH * TABLE_MAX * BLOCK_SIZE:
            raise ValueError("a bounded shared cache requires caller-owned CacheTables")
        block_table = torch.arange(BATCH * TABLE_MAX, dtype=torch.int32).reshape(BATCH, TABLE_MAX)
        pool_table = torch.arange(BATCH * POOL_TABLE_MAX, dtype=torch.int32).reshape(BATCH, POOL_TABLE_MAX)
        pool_offsets = torch.arange(BATCH, dtype=torch.int32) * POOL_STRIDE
    else:
        block_table = cache_pages.table.clone()
        pool_table = cache_pages.pool_table()
        pool_offsets = cache_pages.pool_offsets()
    metadata = build_forward_metadata(
        query_start_loc, kv_seq_lens=previous, mla_block_table=block_table,
        index_block_table=block_table,
        pool_block_table=pool_table,
        kda_state_rows=(
            torch.tensor([[row, row, *[-1] * (KDA_STATE_COLUMNS - 2)] for row in range(BATCH)], dtype=torch.int32)
            if kda_state_rows is None else kda_state_rows
        ), tp_size=TP_SIZE,
    )
    step = build_indexer_chunk_metadata(
        metadata, block_table, pool_offsets, pools_cap=POOL_AXIS, pool_rows=POOL_TABLE_ROWS,
        pool_block_table=pool_table,
    )
    for name in ("seg_start", "pool_count_tok", "tail_start", "tail_count", "kv_len_tok"):
        step[name] = _pad_to(step[name], CAPACITY, 0)
    logit_rows = torch.where(lengths > 0, metadata.logit_row_indices, torch.zeros(BATCH, dtype=torch.int32))
    return {
        "embedding_rows": embedding_rows,
        "query_start_loc": query_start_loc,
        "request_ids": _pad_to(metadata.request_ids, CAPACITY, -1),
        "mla_slots": _pad_to(metadata.mla_slots, CAPACITY, -1),
        "index_slots": _pad_to(metadata.index_slots, CAPACITY, -1),
        "kda_state_rows": metadata.kda_state_rows,
        "block_table": block_table,
        "logit_row_indices": _pad_to(logit_rows, LOGIT_ROWS, 0),
        "active_tokens": torch.tensor([total], dtype=torch.int32),
        **step,
        **kda_host_constants(),
    }


def production_program_entry(program: str):
    """Select an auxiliary single-stage L3 entry for MTP execution."""
    if program == "dsa_attn":
        return l3_prefill_sparse_attention
    if program == "moe_ffn":
        return l3_prefill_moe_layer
    if program == "head":
        return l3_prefill_head_rows
    raise ValueError(f"unknown consolidated program {program!r}")


def _l3_arg_arity(entry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """One entry's ``(tensor names, scalar names)``, both in signature order."""
    tensors, scalars = [], []
    for name, param in inspect.signature(entry._func).parameters.items():
        if param.annotation is pl.constexpr:
            continue
        if getattr(param.annotation, "shape", None) is None:
            scalars.append(name)
        else:
            tensors.append(name)
    return tuple(tensors), tuple(scalars)


def _alloc_name_of(param: str, overrides: dict[str, str]) -> str:
    """One signature parameter's allocation name in the runner's tensor table.

    Stream names answer the step's ping-pong override; every other ``out_*``
    boundary name answers its ``ws_*`` workspace; the rest (step inputs, cache
    pools, resident stacks, ``head_hidden``) allocate under their own names.
    """
    if param in overrides:
        return overrides[param]
    if param.startswith("out_"):
        workspace = "ws_" + param[len("out_") :]
        if workspace in WORKSPACES:
            return workspace
    return param


def _read_ffn(
    checkpoint: Checkpoint, layer: int, mlp_base: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FFN as FP32 ``(gate, up, down)``, fused or separate gate/up spelling."""

    def names(suffixes: Sequence[str]) -> list[str]:
        return [f"{root}{layer}.{mlp_base}{s}" for root in TEXT_LAYER_ROOTS for s in suffixes]

    fused = next((n for n in names(_FF_FUSED) if n in checkpoint), None)
    if fused is not None:
        gate_up = read_reference_float(checkpoint, fused)
        gate, up = gate_up.chunk(2, dim=0)
    else:
        gate = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_GATE))
        up = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_UP))
    down = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_DOWN))
    return gate.contiguous(), up.contiguous(), down.contiguous()


@dataclass
class RefRunResult:
    """What ``PrefillReferenceDriver.run`` returns."""

    final_hidden: torch.Tensor | None  # [B, S, D] post final-norm (full range only)
    streams_out: torch.Tensor  # [B, S, hc_mult, D] leaving the last layer
    layer_summaries: list[dict[str, float]] = field(default_factory=list)
    dump_paths: dict[int, Path] = field(default_factory=dict)
    final_dump_path: Path | None = None


class PrefillReferenceDriver:
    """Host-side reference prefill driver over real or synthetic weights.

    Real mode (``checkpoint_root`` given) wraps the deployment W8A8 checkpoint
    through :class:`Checkpoint` and streams one layer at a time; the
    routed experts of a MoE layer stream one expert at a time inside the layer.
    Synthetic mode (``checkpoint_root=None``) serves deterministic synthetic
    weights at the true ``config.py`` geometry for the no-checkpoint self-test.
    """

    def __init__(self, checkpoint_root: str | Path | None = None, model: Glm53FlashConfig = FLASH):
        if checkpoint_root is not None and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; scaled models run checkpoint-less")
        self.model = model
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self.checkpoint = Checkpoint(self.checkpoint_root) if self.checkpoint_root is not None else None
        self._lm_head_cache: torch.Tensor | None = None

    # -- weight loading -----------------------------------------------------------------

    def load_layer(self, layer_id: int, *, seed: int = 0) -> LayerRefWeights:
        """Load (and dequantize) one layer, or synthesize it in checkpoint-less mode.

        Backbone layers are ``[0, num_hidden_layers)``; the MTP layer
        (``model.mtp_layer_id``) also loads when present — it has no mHC
        tensors (``has_hyper_connections=False``), so its ``attn_hc``/``ffn_hc``
        are ``None`` and its sub-layer weights feed :func:`mla_forward` /
        :func:`moe_forward` directly for the MTP diff harness.
        """
        model = self.model
        if not 0 <= layer_id <= model.mtp_layer_id:
            raise ValueError(
                f"layer_id must be in [0, {model.mtp_layer_id}] (backbone layers plus the "
                f"MTP layer), got {layer_id}"
            )
        if self.checkpoint is None:
            return synthetic_layer(model, layer_id, seed)
        checkpoint = self.checkpoint
        config = model.layer_config(layer_id)

        def read(sub_prefix: str, suffixes: Sequence[str]) -> torch.Tensor:
            return read_reference_float(checkpoint, _resolve_name(checkpoint, layer_id, sub_prefix, suffixes))

        norms = {member: read(prefix, suffixes) for member, prefix, suffixes in _NORM_TENSORS}

        def read_mhc(prefix_name: str) -> MhcTensors:
            picked = {
                member: read(prefix, suffixes)
                for member, prefix, suffixes in _MHC_TENSORS
                if member.startswith(prefix_name)
            }
            return MhcTensors(
                fn=picked[f"{prefix_name}fn"], base=picked[f"{prefix_name}base"],
                scale=picked[f"{prefix_name}scale"],
            )

        layer = LayerRefWeights(
            layer_id=layer_id,
            config=config,
            model=model,
            input_norm=norms["input_norm"],
            post_norm=norms["post_norm"],
            # The MTP layer carries no hc_* tensors: mHC reads are gated on the
            # layer's has_hyper_connections flag, never unconditional.
            attn_hc=read_mhc("attn_") if config.has_hyper_connections else None,
            ffn_hc=read_mhc("ffn_") if config.has_hyper_connections else None,
        )
        if config.attention_kind is AttentionKind.LINEAR:
            kda = {member: read(prefix, suffixes) for member, prefix, suffixes in _KDA_TENSORS}
            layer.kda = KdaTensors(**kda)
        else:
            mla = {member: read(prefix, suffixes) for member, prefix, suffixes in _MLA_TENSORS}
            layer.mla = MlaTensors(**mla)
            indexer = {member: read(prefix, suffixes) for member, prefix, suffixes in _INDEXER_TENSORS}
            layer.indexer = IndexerTensors(**indexer)
        if config.mlp_kind is MlpKind.DENSE:
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.")
            layer.dense = DenseTensors(gate, up, down)
        else:
            router_weight = read("mlp.", ("gate.weight",))
            bias_name = _resolve_name(checkpoint, layer_id, "mlp.", ("gate.e_score_correction_bias",))
            layer.router = RouterTensors(router_weight, read_reference_float(checkpoint, bias_name))
            layer.experts = CheckpointExpertProvider(checkpoint, layer_id)
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.shared_experts.")
            layer.shared = DenseTensors(gate, up, down)
        return layer

    def missing_layer_tensors(self, layer_id: int) -> list[str]:
        """Checkpoint tensors this layer still needs but no present shard carries.

        A non-empty answer means the download is incomplete and ``load_layer``
        will raise; empty means the layer can run against true weights today.
        Names are symbolic (canonical root spelling): a tensor counts as present
        when any accepted spelling resolves, mirroring :func:`_resolve_name`.
        mHC tensors are requested only when the layer has hyper-connections —
        the MTP layer carries none (``has_hyper_connections=False``).
        """
        if self.checkpoint is None:
            return []
        if not 0 <= layer_id <= self.model.mtp_layer_id:
            raise ValueError(f"layer_id must be in [0, {self.model.mtp_layer_id}], got {layer_id}")
        model = self.model
        config = model.layer_config(layer_id)
        checkpoint = self.checkpoint

        def present(sub_prefix: str, suffixes: Sequence[str]) -> bool:
            return any(
                (f"{root}{layer_id}.{sub_prefix}{suffix}" in checkpoint for root in TEXT_LAYER_ROOTS for suffix in suffixes),
            )

        def gap(sub_prefix: str, suffixes: Sequence[str]) -> list[str]:
            if present(sub_prefix, suffixes):
                return []
            return [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{sub_prefix}{suffixes[0]}"]

        def ffn_gaps(base: str) -> list[str]:
            gate_up_ok = present(base, _FF_FUSED) or (present(base, _FF_GATE) and present(base, _FF_UP))
            gaps = [] if gate_up_ok else [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_FUSED[0]}"]
            if not present(base, _FF_DOWN):
                gaps.append(f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_DOWN[0]}")
            return gaps

        gaps: list[str] = []
        for _, prefix, suffixes in _NORM_TENSORS:
            gaps += gap(prefix, suffixes)
        if config.has_hyper_connections:
            for _, prefix, suffixes in _MHC_TENSORS:
                gaps += gap(prefix, suffixes)
        family = _KDA_TENSORS if config.attention_kind is AttentionKind.LINEAR else _MLA_TENSORS + _INDEXER_TENSORS
        for _, prefix, suffixes in family:
            gaps += gap(prefix, suffixes)
        if config.mlp_kind is MlpKind.DENSE:
            gaps += ffn_gaps("mlp.")
        else:
            gaps += gap("mlp.", ("gate.weight",))
            gaps += gap("mlp.", ("gate.e_score_correction_bias",))
            for expert in range(model.n_routed_experts):
                gaps += ffn_gaps(f"mlp.experts.{expert}.")
            gaps += ffn_gaps("mlp.shared_experts.")
        return gaps

    # -- host tables --------------------------------------------------------------------

    def embed_rows(
        self, token_ids: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Embedding rows ``[B, S, D]`` FP32: checkpoint row-gather or synthetic."""
        ids = _as_batch_seqs(token_ids)
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("embed_tokens.weight",))
            return gather_rows(self.checkpoint, name, ids).to(torch.float32)
        gen = generator if generator is not None else torch.Generator().manual_seed(1234)
        rows = torch.randn(ids.numel(), self.model.hidden_size, generator=gen) * 0.02
        return rows.view(*ids.shape, self.model.hidden_size)

    def final_norm_weight(self) -> torch.Tensor:
        """The final RMSNorm weight ``[D]`` (ones in synthetic mode)."""
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("norm.weight",))
            return read_reference_float(self.checkpoint, name)
        return torch.ones(self.model.hidden_size)

    def _lm_head(self) -> torch.Tensor:
        if self._lm_head_cache is None:
            if self.checkpoint is None:
                raise RuntimeError("lm_head top-1 needs a real checkpoint")
            name = _resolve_top_name(self.checkpoint, ("lm_head.weight",))
            self._lm_head_cache = self.checkpoint.read(name).to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    def lm_head_top1(
        self, hidden: torch.Tensor, k: int = 1, *, chunk: int = 8192
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host LM-head top-k: chunked FP32 matmul over the full ``[vocab, D]`` table.

        Returns ``(values, indices)``, each ``[N, k]``, rows of ``hidden`` ``[N, D]``.
        """
        head = self._lm_head()
        rows = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        best_values = torch.full((rows.shape[0], k), -torch.inf)
        best_indices = torch.zeros((rows.shape[0], k), dtype=torch.int64)
        for start in range(0, head.shape[0], chunk):
            logits = (head[start : start + chunk].to(torch.float32)) @ rows.t()  # [chunk, N]
            candidates = torch.cat([best_values, logits.t()], dim=1)
            indices = torch.cat(
                [best_indices, torch.arange(start, start + logits.shape[0])[None].expand(rows.shape[0], -1)],
                dim=1,
            )
            order = candidates.argsort(dim=1, descending=True)[:, :k]
            best_values = candidates.gather(1, order)
            best_indices = indices.gather(1, order)
        return best_values.contiguous(), best_indices.contiguous()

    # -- the streaming run ----------------------------------------------------------------

    def run(
        self,
        token_ids: torch.Tensor,
        *,
        layers: tuple[int, int] | None = None,
        attention_mask: torch.Tensor | None = None,
        initial_streams: torch.Tensor | None = None,
        dump_dir: Path | str | None = None,
        dump_layers: set[int] | None = None,
        dump_keys: Sequence[str] | None = None,
        dump_internals: bool = False,
        dump_dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> RefRunResult:
        """Stream the requested backbone layers on a prompt.

        ``layers`` is an inclusive-exclusive ``(start, stop)`` over
        ``[0, num_hidden_layers)``; default runs the whole backbone. Entering
        streams come from the embedding (all ``hc_mult`` streams equal) unless
        ``initial_streams [B, S, hc_mult, D]`` is given. When ``stop`` reaches
        the backbone depth the final collapse (mean over streams) and final
        RMSNorm run, and ``final_hidden`` is set. Layer ``L`` is dumped to
        ``dump_dir/layer_{L:02d}.pt`` when ``dump_layers`` selects it (default:
        every executed layer); ``dump_keys`` filters the payload (default:
        :data:`DEFAULT_DUMP_KEYS`, plus internals with ``dump_internals``).
        """
        model = self.model
        ids = _as_batch_seqs(token_ids)
        batch_size, seq_len = ids.shape
        start, stop = layers if layers is not None else (0, model.num_hidden_layers)
        if not 0 <= start < stop <= model.num_hidden_layers:
            raise ValueError(f"layers must satisfy 0 <= start < stop <= {model.num_hidden_layers}")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask = attention_mask.to(torch.bool)

        if initial_streams is None:
            embedded = self.embed_rows(ids, generator=torch.Generator().manual_seed(seed))
            streams = embedded.unsqueeze(2).expand(-1, -1, model.hc_mult, -1).contiguous()
        else:
            streams = initial_streams.to(torch.float32).contiguous()

        keys = tuple(dump_keys) if dump_keys is not None else DEFAULT_DUMP_KEYS
        if dump_internals:
            keys = tuple(dict.fromkeys((*keys, *INTERNAL_DUMP_KEYS)))
        dump_dir_path = Path(dump_dir) if dump_dir is not None else None
        if dump_dir_path is not None:
            dump_dir_path.mkdir(parents=True, exist_ok=True)

        result = RefRunResult(final_hidden=None, streams_out=streams)
        for layer_id in range(start, stop):
            weights = self.load_layer(layer_id, seed=seed)
            streams, boundaries = reference_layer_forward(streams, weights, attention_mask)
            del weights
            result.layer_summaries.append(
                {"layer": layer_id, **{key: float(boundaries[key].abs().max()) for key in ("x_hc_in", "attn_out", "x_hc_mid", "ffn_out", "x_hc_out")}},
            )
            if dump_dir_path is not None and (dump_layers is None or layer_id in dump_layers):
                layer_config = model.layer_config(layer_id)
                payload: dict[str, object] = {
                    "layer_id": layer_id,
                    "attention_kind": layer_config.attention_kind.value,
                    "mlp_kind": layer_config.mlp_kind.value,
                }
                payload.update({key: boundaries[key].to(dump_dtype) for key in keys if key in boundaries})
                path = dump_dir_path / f"layer_{layer_id:02d}.pt"
                torch.save(payload, path)
                result.dump_paths[layer_id] = path
            del boundaries

        result.streams_out = streams
        if stop == model.num_hidden_layers:
            collapsed = streams.mean(dim=2)
            result.final_hidden = reference_rms_norm(collapsed, self.final_norm_weight(), model.rms_norm_eps)
            if dump_dir_path is not None:
                result.final_dump_path = dump_dir_path / "final.pt"
                torch.save(
                    {"final_hidden": result.final_hidden.to(dump_dtype), "streams_mean_pre_norm": collapsed.to(dump_dtype), "token_ids": ids, "layer_range": (start, stop), "final_norm_eps": model.rms_norm_eps},
                    result.final_dump_path,
                )
        return result


def _test_layers_end_to_end(model: Glm53FlashConfig, tmp_root: Path) -> None:
    torch.manual_seed(31)
    b, s = 2, 17
    driver = PrefillReferenceDriver()  # synthetic mode
    mask = torch.ones(b, s, dtype=torch.bool)

    for layer_id, attention, mlp in (
        (0, "linear_attention", "dense"),
        (4, "linear_attention", "sparse"),
        (3, "deepseek_sparse_attention", "sparse"),
    ):
        weights = driver.load_layer(layer_id)
        assert weights.config.attention_kind.value == attention
        assert weights.config.mlp_kind.value == mlp
        embedded = torch.randn(b, s, model.hc_mult, model.hidden_size) * 0.5
        streams_a, bounds_a = reference_layer_forward(embedded, weights, mask)
        streams_b, _ = reference_layer_forward(embedded, weights, mask)
        assert streams_a.shape == (b, s, model.hc_mult, model.hidden_size)
        assert torch.equal(streams_a, streams_b), f"layer {layer_id} is nondeterministic"
        assert torch.isfinite(streams_a).all()
        for key in DEFAULT_DUMP_KEYS:
            assert key in bounds_a, f"layer {layer_id} dump key {key} missing"
        # Wiring: recompute the final combine from the dumped boundaries.
        recomputed = combine_streams(
            bounds_a["ffn_hc_post"], bounds_a["ffn_hc_comb"], bounds_a["ffn_out"], bounds_a["x_hc_mid"],
        )
        _assert_close(streams_a, recomputed, f"layer {layer_id} combine wiring", atol=1e-4, rtol=1e-4)
        print(f"  layer {layer_id:2d} ({attention}/{mlp}): deterministic, wired, finite")

    # Driver run over layers 0..4 with internals: dump-chain continuity and shapes.
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        token_ids = torch.randint(0, 1000, (b, s))
        result = driver.run(token_ids, layers=(0, 5), dump_dir=tmp, dump_internals=True)
        assert result.final_hidden is None, "mid-range stop must not produce final hidden"
        assert set(result.dump_paths) == {0, 1, 2, 3, 4}
        payloads = {L: torch.load(path, weights_only=False) for L, path in result.dump_paths.items()}
        for layer_id, payload in payloads.items():
            assert payload["layer_id"] == layer_id
            assert payload["x_hc_in"].shape == (b, s, model.hc_mult, model.hidden_size)
            assert payload["x_hc_out"].shape == (b, s, model.hc_mult, model.hidden_size)
        # Stream continuity: each layer's dumped input is the previous output.
        for layer_id in range(4):
            assert torch.equal(payloads[layer_id]["x_hc_out"], payloads[layer_id + 1]["x_hc_in"])
        assert payloads[3]["mla_topk_indices"].shape[-1] == model.index_topk + model.index_kpool - 1
        assert payloads[0]["kda_core"].shape == (
            b,
            s,
            model.linear_num_heads,
            model.linear_head_dim,
        )
        assert payloads[4]["router_topk_weights"].shape == (b * s, model.num_experts_per_tok)
        assert "router_topk_weights" not in payloads[1]  # layer 1 is a dense-MLP layer

    # Whole-backbone synthetic run (final-norm path + final.pt roundtrip).
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp, dump_layers=set())
        assert result.final_hidden.shape == (1, 8, model.hidden_size)
        assert torch.isfinite(result.final_hidden).all()
        final_dump = torch.load(result.final_dump_path, weights_only=False)
        _assert_close(
            final_dump["final_hidden"].to(torch.float32), result.final_hidden, "final dump roundtrip",
            atol=0.0001,
        )
        manual = reference_rms_norm(result.streams_out.mean(dim=2), driver.final_norm_weight(), model.rms_norm_eps)
        _assert_close(result.final_hidden, manual, "final norm recompute", atol=1e-4)
    print("  driver.run: dump-chain continuity, internal keys, final norm + final.pt")


def _test_scaled_model(model: Glm53FlashConfig) -> None:
    """The mini-checkpoint consumption path: a scaled 4-layer schedule.

    ``layer_ckpt_l0l3`` (layers 0-3, num_hidden_layers=4, first_k_dense_replace=3)
    is the first real single-layer-diff weight source; this exercises the same
    model-override mechanics without needing its files on disk yet.
    """
    scaled = replace(model, num_hidden_layers=4, first_k_dense_replace=3)
    assert scaled.kda_layer_ids == (0, 1, 2) and scaled.dsa_layer_ids == (3,)
    assert scaled.layer_config(3).mlp_kind is MlpKind.SPARSE

    driver = PrefillReferenceDriver(model=scaled)  # synthetic weights, scaled schedule
    with tempfile.TemporaryDirectory() as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp)
        assert len(result.layer_summaries) == 4
        assert result.final_hidden.shape == (1, 8, scaled.hidden_size)
        assert result.dump_paths[3].name == "layer_03.pt"
        payload = torch.load(result.dump_paths[3], weights_only=False)
        assert payload["attention_kind"] == "deepseek_sparse_attention"
        assert payload["mlp_kind"] == "sparse"
    # Layers past the MTP ordinal stay rejected under the scaled model.
    try:
        driver.load_layer(scaled.mtp_layer_id + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("scaled model must reject layers past the MTP ordinal")
    # ...and scaled_model_from_config round-trips a deployment-style config.json.
    with tempfile.TemporaryDirectory() as tmp:
        text = {
            "num_hidden_layers": 4,
            "first_k_dense_replace": 3,
            "hidden_size": model.hidden_size,
            "linear_attn_config": {"num_heads": model.linear_num_heads},
        }
        (Path(tmp) / "config.json").write_text(json.dumps({"text_config": text}))
        derived = scaled_model_from_config(tmp)
        assert derived.num_hidden_layers == 4 and derived.first_k_dense_replace == 3
        assert derived.hidden_size == model.hidden_size and derived.vocab_size == model.vocab_size

    # The MTP layer (layer 4 of the scaled schedule) loads without mHC tensors
    # and refuses the backbone wiring instead of demanding absent hc_* names.
    mtp = driver.load_layer(scaled.mtp_layer_id)
    assert mtp.attn_hc is None and mtp.ffn_hc is None
    assert mtp.config.has_hyper_connections is False
    assert driver.missing_layer_tensors(scaled.mtp_layer_id) == []
    try:
        reference_layer_forward(torch.randn(1, 4, scaled.hc_mult, scaled.hidden_size), mtp)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("MTP layer must reject the hyper-connection wiring")
    print("  scaled model: 4-layer schedule, DSA/MoE layer 3, config.json override, MTP gating")


def _test_lm_head_top1(model: Glm53FlashConfig) -> None:
    """Chunked top-k merge against one direct matmul (fake 20000-row head)."""
    torch.manual_seed(41)
    driver = PrefillReferenceDriver()
    driver._lm_head_cache = torch.randn(20000, model.hidden_size, dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(7, model.hidden_size)
    values, indices = driver.lm_head_top1(hidden, k=3)  # 20000 rows -> 3 merge chunks
    reference = driver._lm_head().to(torch.float32) @ hidden.t()
    top = reference.topk(3, dim=0)
    assert indices.shape == (7, 3)
    assert torch.equal(indices, top.indices.t()), "top-k indices disagree with direct matmul"
    _assert_close(values, top.values.t(), "lm_head top-k merge values", atol=1e-3, rtol=1e-4)
    print("  lm_head top1: chunked merge == direct matmul (k=3, 3 chunks)")


def _real_checkpoint_smoke(root: str) -> None:
    """Against the real (possibly still downloading) W8A8 snapshot."""
    driver = PrefillReferenceDriver(root)
    print(f"  checkpoint opened: {len(driver.checkpoint)} indexed tensors")
    # Dequant one real W8A8 tensor through this module's reader and verify the
    # scale multiply against the raw INT8 payload directly.
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    if name in driver.checkpoint:
        dequant = read_reference_float(driver.checkpoint, name)
        raw = driver.checkpoint.read(name).to(torch.float32)
        scale = driver.checkpoint.read(name[: -len(".weight")] + ".weight_scale").float().reshape(-1)
        _assert_close(dequant, raw * scale[:, None], "real W8A8 dequant", atol=0, rtol=0)
        print(f"  real dequant exact: {name} {tuple(dequant.shape)}")
    for layer_id in (0, 3):
        missing = driver.missing_layer_tensors(layer_id)
        verdict = "loadable" if not missing else f"{len(missing)} tensors pending"
        print(f"  layer {layer_id} real-weight coverage: {verdict}")


def run_self_tests(model: Glm53FlashConfig = FLASH, checkpoint_root: str | None = None) -> None:
    """CPU self-test battery (synthetic weights at the true geometry)."""
    print(
        f"prefill reference self-test: hidden={model.hidden_size} layers={model.num_hidden_layers} heads={model.num_attention_heads}/{model.linear_num_heads} experts={model.n_routed_experts}",
    )
    _test_mhc(model)
    _test_kda_cores(model)
    _test_indexer(model)
    _test_mla(model)
    _test_router_and_moe(model)
    _test_dense_mlp(model)
    _test_lm_head_top1(model)
    _test_layers_end_to_end(model, Path("."))
    _test_scaled_model(model)
    if checkpoint_root is not None:
        print(f"real-checkpoint smoke: {checkpoint_root}")
        _real_checkpoint_smoke(checkpoint_root)
    print("prefill reference self-test: ALL PASS")


def run_weight_checks(argv=None):
    """CPU golden: layout, fusion and sharding against direct slicing.

    Host-side only, but the a2a3 sweep invokes every model file as
    ``python <file> -p a2a3 -d <ids>``, so the same switches are accepted and
    ignored. ``--tp`` / ``--ep`` are read by ``config.py`` at import time.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="0")
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--ep", type=int, default=16)
    parser.parse_args(argv)
    torch.manual_seed(23)

    def pieces(rows: int, width: int) -> tuple[torch.Tensor, ...]:
        """One FFN's canonical INT8 weights and per-channel scales."""
        return (
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (width, rows), dtype=torch.int8),
            torch.rand(rows, 1) * 1e-2,
            torch.rand(rows, 1) * 1e-2,
            torch.rand(width, 1) * 1e-2,
        )

    routed_global, routed_ep, routed_local = 32, 2, 16
    routed_pieces = {expert: pieces(2048, D) for expert in range(routed_global)}
    shared_pieces = pieces(2048, D)
    dense_pieces = pieces(DENSE_INTER, D)
    router_weight = torch.randn(N_EXPERTS, D, dtype=torch.bfloat16)
    router_bias = torch.randn(N_EXPERTS)

    def materialize(fused: bool, root: str = TEXT_LAYER_ROOTS[0]) -> dict[str, torch.Tensor]:
        """Write one canonical weight set out in either checkpoint spelling."""

        def prefix(layer: int) -> str:
            return f"{root}{layer}.mlp."

        out: dict[str, torch.Tensor] = {
            prefix(0) + "gate.weight": router_weight,
            prefix(0) + "gate.e_score_correction_bias": router_bias,
            prefix(3) + "gate.weight": router_weight,
            prefix(3) + "gate.e_score_correction_bias": router_bias,
        }

        def put(prefix: str, base: str, piece: tuple[torch.Tensor, ...]) -> None:
            gate, up, down, gate_scale, up_scale, down_scale = piece
            if fused:
                out[prefix + base + "w_gate_up"] = torch.cat([gate, up], dim=0)
                out[prefix + base + "w_gate_up.weight_scale"] = torch.cat([gate_scale, up_scale], dim=0)
                out[prefix + base + "w_down"] = down
                out[prefix + base + "w_down.weight_scale"] = down_scale
            else:
                out[prefix + base + "gate_proj.weight"] = gate
                out[prefix + base + "gate_proj.weight_scale"] = gate_scale
                out[prefix + base + "up_proj.weight"] = up
                out[prefix + base + "up_proj.weight_scale"] = up_scale
                out[prefix + base + "down_proj.weight"] = down
                out[prefix + base + "down_proj.weight_scale"] = down_scale

        for expert, piece in routed_pieces.items():
            put(prefix(3), f"experts.{expert}.", piece)
        put(prefix(3), "shared_experts.", shared_pieces)
        put(prefix(0), "", dense_pieces)
        return out

    sparse_fused = materialize(fused=True)
    sparse_separate = materialize(fused=False)

    def assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
        if a.shape != b.shape:
            raise AssertionError(f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        if not torch.equal(a, b):
            raise AssertionError(f"{what}: values differ")

    # Fusing separate gate/up must reproduce the fused checkpoint bit-exactly,
    # and EP must pick whole experts in the kernel's [local, 2*inter, D] layout.
    for ep_rank in (0, 1):
        shard_args = dict(local_experts=routed_local, global_experts=routed_global)
        separate = shard_routed_experts(sparse_separate, 3, ep_rank, routed_ep, **shard_args)
        fused = shard_routed_experts(sparse_fused, 3, ep_rank, routed_ep, **shard_args)
        assert_close(separate.w_gate_up, fused.w_gate_up, f"ep{ep_rank} routed gate_up")
        assert_close(separate.w_gate_up_scale, fused.w_gate_up_scale, f"ep{ep_rank} routed gate_up_scale")
        assert_close(separate.w_down, fused.w_down, f"ep{ep_rank} routed down")
        assert_close(separate.w_down_scale, fused.w_down_scale, f"ep{ep_rank} routed down_scale")
        expected = torch.stack(
            [torch.cat([routed_pieces[e][0], routed_pieces[e][1]], dim=0) for e in range(ep_rank * routed_local, (ep_rank + 1) * routed_local)],
        )
        assert_close(separate.w_gate_up, expected, f"ep{ep_rank} routed layout")
        assert separate.w_gate_up.dtype == torch.int8
        assert separate.w_gate_up_scale.dtype == torch.float32
        assert separate.w_gate_up_scale.shape == (routed_local, 4096)

    # Shared expert: every EP rank keeps complete projections and scales.
    shared = load_shared_expert(sparse_fused, 3)
    shared_separate = load_shared_expert(sparse_separate, 3)
    inter = FLASH.moe_intermediate_size
    assert_close(shared.gate_up[:inter], shared_pieces[0], "shared gate")
    assert_close(shared.gate_up[inter:], shared_pieces[1], "shared up")
    assert_close(shared.down, shared_pieces[2], "shared down")
    assert_close(
        shared.gate_up_scale, torch.cat([shared_pieces[3].reshape(-1), shared_pieces[4].reshape(-1)]),
        "shared gate/up scale",
    )
    assert_close(shared.down_scale, shared_pieces[5].reshape(-1), "shared down scale")
    for field_name in ("gate_up", "gate_up_scale", "down", "down_scale"):
        assert_close(getattr(shared_separate, field_name), getattr(shared, field_name), f"shared fused {field_name}")

    # Dense layer: TP slices the intermediate width, and the router passes through
    # with the kernels' dtypes.
    dense = shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE)
    local_dense = DENSE_INTER // TP_SIZE
    assert dense.gate_up.shape == (2 * local_dense, D)
    assert dense.down.shape == (D, local_dense)
    assert dense.gate_up_scale.shape == (2 * local_dense,)
    assert dense.down_scale.shape == (D,)
    assert_close(dense.gate_up[:local_dense], dense_pieces[0][:local_dense], "dense gate slice")
    router = layer_router(sparse_fused, 0)
    assert router.weight.dtype == torch.bfloat16 and router.weight.shape == (N_EXPERTS, D)
    assert router.correction_bias.dtype == torch.float32 and router.correction_bias.shape == (N_EXPERTS,)
    assert_close(router.correction_bias, router_bias, "router bias")

    # The released checkpoint nests the text backbone under ``language_model``
    # (``model.language_model.layers.*``); both roots load the same tensors.
    nested = materialize(fused=True, root="model.language_model.layers.")
    ep_rank = 1
    nested_routed = shard_routed_experts(
        nested, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global,
    )
    canonical_routed = shard_routed_experts(
        sparse_fused, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global,
    )
    for field_name in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"):
        assert_close(getattr(nested_routed, field_name), getattr(canonical_routed, field_name), f"nested routed {field_name}")
    nested_shared = load_shared_expert(nested, 3)
    assert_close(nested_shared.gate_up, shared.gate_up, "nested shared gate_up")
    assert_close(nested_shared.down, shared.down, "nested shared down")
    nested_dense = shard_dense_mlp(nested, 0, 0, TP_SIZE)
    assert_close(nested_dense.gate_up, dense.gate_up, "nested dense gate_up")
    nested_router = layer_router(nested, 3)
    canonical_router = layer_router(sparse_fused, 3)
    assert_close(nested_router.weight, canonical_router.weight, "nested router weight")
    assert_close(nested_router.correction_bias, canonical_router.correction_bias, "nested router bias")
    try:
        load_shared_expert(materialize(fused=True, root="model.text."), 3)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown layer root should have been rejected")

    # Boundary validation: the one shipped scheme, and a scale is mandatory.
    QuantDescription.parse({"model_quant_type": "W8A8_DYNAMIC", "group_size": 0, "is_rot_used": True}).require_dynamic_symmetric(
    )
    for bad in (
        {"model_quant_type": "FP8", "group_size": 0},
        {"model_quant_type": "W8A8_DYNAMIC", "group_size": 128},
    ):
        try:
            QuantDescription.parse(bad).require_dynamic_symmetric()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")
    try:
        check_ffn_scales(sparse_fused, [layer_prefix(0) + "missing.weight"])
    except KeyError:
        pass
    else:
        raise AssertionError("a missing scale should have been rejected")

    # Shard geometry is validated rather than silently slicing short.
    for bad_call in (
        lambda: shard_dense_mlp(sparse_fused, 0, -1, TP_SIZE),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, intermediate=DENSE_INTER + 32),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, local_inter=DENSE_INTER // TP_SIZE + 1),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shard geometry should have been rejected")

    # The .safetensors branch of ``load_tensors`` is the one loader path with no
    # coverage when the optional package is absent; round-trip a small shard
    # through it whenever the package is present.
    try:
        from safetensors.torch import save_file  # noqa: PLC0415
    except ImportError:
        print("[GOLDEN] skip  weights safetensors reader (safetensors not installed)")
    else:
        import tempfile  # noqa: PLC0415

        probe = {
            "probe.weight": torch.arange(24, dtype=torch.int8).reshape(4, 6),
            "probe.weight_scale": torch.linspace(0.1, 1.0, 4).reshape(4, 1),
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_file(probe, str(Path(tmp) / "shard-0.safetensors"))
            loaded = load_tensors(tmp)
            if not torch.equal(loaded["probe.weight"], probe["probe.weight"]):
                raise AssertionError("safetensors reader lost the weight payload")
            if not torch.equal(_scale_of(loaded, "probe.weight"), probe["probe.weight_scale"].reshape(-1)):
                raise AssertionError("safetensors reader did not flatten the [out, 1] scale")
        print("[GOLDEN] PASS weights (safetensors round-trip)")

    print(f"[GOLDEN] PASS weights (routed 32 experts over EP{routed_ep}, shared replicated, dense TP{TP_SIZE})")

    # ------------------- resident prefill staging (in-memory fixture) -------------------
    # Everything below runs against a scaled-geometry synthetic checkpoint held
    # in memory through the staging read protocol: never /data/models, never a
    # device, and never the optional safetensors package. ``Checkpoint`` parsing
    # and metadata validation are covered by the loader tests above; these cases
    # cover the staging itself - resolution, fusion, sharding, splitting,
    # stacking and the staged upload.

    TINY = Glm53FlashConfig(
        name="glm5_3_flash_tiny", vocab_size=128, hidden_size=64, num_hidden_layers=4,
        first_k_dense_replace=1, intermediate_size=32, rms_norm_eps=1e-05, max_position_embeddings=1024,
        swiglu_limit=10.0, tie_word_embeddings=False, num_attention_heads=4, q_lora_rank=12, kv_lora_rank=6,
        qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8, index_n_heads=2, index_head_dim=8, index_topk=4,
        index_kpool=2, index_kpool_compress=True, index_kpool_always_select_tail=True,
        index_share_for_mtp_iteration=True, linear_num_heads=4, linear_head_dim=8, linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=32, scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=2.5, n_group=1, topk_group=1, mhc=True, hc_mult=2, hc_sinkhorn_iters=20,
        hc_eps=1e-06, num_nextn_predict_layers=1, dtype="bfloat16", quant_method="w8a8",
        activation_scheme="dynamic",
    )

    def tiny_root(layer: int) -> str:
        """The fixture's dual-spelling rule: layers 0-1 canonical, 2-4 nested."""
        return "model.layers." if layer <= 1 else "model.language_model.layers."

    class _FixtureWeights(Mapping[str, torch.Tensor]):
        """In-memory stand-in for :class:`Checkpoint` (the staging read protocol)."""

        def __init__(self, tensors: dict[str, torch.Tensor], labels: dict[str, str]):
            self.tensors, self.description = tensors, labels

        def __contains__(self, name: object) -> bool:
            return name in self.tensors

        def __iter__(self):
            return iter(self.tensors)

        def __len__(self) -> int:
            return len(self.tensors)

        def __getitem__(self, name: str) -> torch.Tensor:
            return self.tensors[name]

        def read(self, name: str, selection=None) -> torch.Tensor:
            return self.tensors[name] if selection is None else self.tensors[name][selection]

        def shape(self, name: str) -> tuple[int, ...]:
            return tuple(self.tensors[name].shape)

        def gather(self, name: str, rows: torch.Tensor) -> torch.Tensor:
            return self.tensors[name][rows]

    def build_tiny_fixture(mutate=None) -> _FixtureWeights:
        """Materialize the scaled-geometry deployment checkpoint the tests run against.

        Covers both layer-root spellings, nested and flat KDA-gate/mHC spellings,
        fused and separate gate/up layouts, the quantized MLA projections with
        scales and zero offsets, and the MTP duplicates of the host tables.
        ``mutate(tensors, labels)`` exists for the negative tests.
        """
        generator = torch.Generator().manual_seed(101)
        hidden = TINY.hidden_size
        qkv = TINY.linear_num_heads * TINY.linear_head_dim
        tensors: dict[str, torch.Tensor] = {}
        labels: dict[str, str] = {}

        def put(name: str, tensor: torch.Tensor, label: str = "FLOAT") -> None:
            tensors[name] = tensor
            labels[name] = label

        def bf16(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator).bfloat16()

        def int8_linear(stem: str, out: int, inn: int, *, bare: bool = False) -> None:
            name = stem if bare else stem + ".weight"
            weight = torch.randint(-127, 128, (out, inn), dtype=torch.int8, generator=generator)
            scale = torch.rand(out, 1, generator=generator).float() * 1e-2
            put(name, weight, "W8A8_DYNAMIC")
            put(stem + ".weight_scale", scale, "W8A8_DYNAMIC")
            put(stem + ".weight_offset", torch.zeros(out, 1), "W8A8_DYNAMIC")

        def ffn(base: str, inter: int, *, fused: bool, dotted: bool = False) -> None:
            # The canonical fused spelling is bare (``w_gate_up``); only the
            # routed expert entries also accept the ``.weight`` variant.
            if fused:
                int8_linear(base + "w_gate_up", 2 * inter, hidden, bare=not dotted)
                int8_linear(base + "w_down", hidden, inter, bare=not dotted)
            else:
                for part in ("gate", "up"):
                    int8_linear(base + part + "_proj", inter, hidden)
                int8_linear(base + "down_proj", hidden, inter)

        for layer in range(TINY.num_checkpoint_layers):
            base = tiny_root(layer) + f"{layer}."
            if layer in TINY.kda_layer_ids:
                attn = base + "self_attn."
                for part in ("q", "k", "v"):
                    put(attn + f"{part}_proj.weight", bf16(qkv, hidden))
                    put(attn + f"{part}_conv1d.weight", bf16(qkv, 1, TINY.linear_conv_kernel_dim))
                gates = attn if layer == 0 else attn + "forget_gate."  # flat vs nested spelling
                put(gates + "dt_bias", torch.randn(qkv, generator=generator).float())
                put(gates + "A_log", torch.randn(TINY.linear_num_heads, generator=generator).float())
                put(gates + "f_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(gates + "f_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "g_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(attn + "g_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "b_proj.weight", bf16(TINY.linear_num_heads, hidden))
                put(attn + "o_norm.weight", bf16(TINY.linear_head_dim))
                put(attn + "o_proj.weight", bf16(hidden, qkv))
            else:  # MLA + indexer (layers 3 and 4, including the MTP layer)
                attn = base + "self_attn."
                int8_linear(attn + "q_a_proj", TINY.q_lora_rank, hidden)
                put(attn + "q_a_layernorm.weight", bf16(TINY.q_lora_rank))
                int8_linear(attn + "q_b_proj", TINY.num_attention_heads * TINY.qk_head_dim, TINY.q_lora_rank)
                int8_linear(attn + "kv_a_proj_with_mqa", TINY.kv_lora_rank, hidden)
                put(attn + "kv_a_layernorm.weight", bf16(TINY.kv_lora_rank))
                put(
                    attn + "kv_b_proj.weight",
                    bf16(TINY.num_attention_heads * (TINY.qk_head_dim + TINY.v_head_dim), TINY.kv_lora_rank),
                )
                int8_linear(attn + "o_proj", hidden, TINY.num_attention_heads * TINY.v_head_dim)
                indexer = attn + "indexer."
                put(indexer + "wq_b.weight", bf16(TINY.index_n_heads * TINY.index_head_dim, TINY.q_lora_rank))
                put(indexer + "wk.weight", bf16(TINY.index_head_dim, hidden))
                put(indexer + "k_norm.weight", bf16(TINY.index_head_dim))
                put(indexer + "k_norm.bias", bf16(TINY.index_head_dim))
                put(indexer + "weights_proj.weight", bf16(TINY.index_n_heads, hidden))
                put(indexer + "index_kpool_compress_gate", bf16(TINY.index_head_dim, hidden))
                put(indexer + "index_kpool_compress_ape", bf16(TINY.index_kpool, TINY.index_head_dim))
            if layer < TINY.num_hidden_layers:  # mHC is backbone-only
                for kind in ("attn", "ffn"):
                    if layer == 0:  # flat spelling on one layer, canonical elsewhere
                        put(base + f"hc_{kind}_fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"hc_{kind}_scale", torch.rand(3, generator=generator).float())
                        put(base + f"hc_{kind}_base", torch.rand(TINY.mix_hc, generator=generator).float())
                    else:
                        put(base + f"{kind}_hc.fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"{kind}_hc.scale", torch.rand(3, generator=generator).float())
                        put(base + f"{kind}_hc.base", torch.rand(TINY.mix_hc, generator=generator).float())
            put(base + "input_layernorm.weight", bf16(hidden))
            put(base + "post_attention_layernorm.weight", bf16(hidden))
            if layer >= TINY.first_k_dense_replace:
                put(base + "mlp.gate.weight", bf16(TINY.n_routed_experts, hidden))
                flat_bias = layer == 1  # flat spelling on one layer, canonical elsewhere
                bias_stem = "mlp.e_score_correction_bias" if flat_bias else "mlp.gate.e_score_correction_bias"
                put(base + bias_stem, torch.randn(TINY.n_routed_experts, generator=generator).float())
                for expert in range(TINY.n_routed_experts):
                    dotted = layer == 4 and expert % 2 == 0  # the routed ``.weight`` variant
                    fused = layer == 2 or dotted
                    ffn(base + f"mlp.experts.{expert}.", TINY.moe_intermediate_size, fused=fused, dotted=dotted)
                ffn(base + "mlp.shared_experts.", TINY.moe_intermediate_size, fused=layer >= 3)
            else:
                ffn(base + "mlp.", TINY.intermediate_size, fused=False)
        embed = bf16(TINY.vocab_size, hidden)
        head = bf16(TINY.vocab_size, hidden)
        put("model.language_model.embed_tokens.weight", embed)
        put("lm_head.weight", head)
        put("model.language_model.norm.weight", bf16(hidden))
        mtp = tiny_root(TINY.mtp_layer_id) + f"{TINY.mtp_layer_id}."
        put(mtp + "eh_proj.weight", bf16(hidden, 2 * hidden))
        put(mtp + "enorm.weight", bf16(hidden))
        put(mtp + "hnorm.weight", bf16(hidden))
        put(mtp + "embed_tokens.weight", embed.clone())  # MTP duplicates alias the
        put(mtp + "shared_head.head.weight", head.clone())  # top-level host tables
        put(mtp + "shared_head.norm.weight", bf16(hidden))  # ignored, like rot.weight
        if mutate is not None:
            mutate(tensors, labels)
        return _FixtureWeights(tensors, labels)

    # Resident test 1: the fixture loads through the staging protocol and the
    # plan arithmetic holds.
    checkpoint = build_tiny_fixture()
    assert checkpoint.description["model.language_model.layers.3.self_attn.q_b_proj.weight"] == "W8A8_DYNAMIC"
    assert checkpoint.description["model.language_model.layers.3.self_attn.kv_b_proj.weight"] == "FLOAT"
    weights = ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=2, model=TINY)
    assert weights.bytes_per_rank == sum(
        math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in weights.tensor_layouts.values()
    )
    assert weights.plan(1).device_bytes == weights.bytes_per_rank
    assert weights.plan(1).tp_rank == 1 and weights.plan(1).local_expert_ids == tuple(range(8, 16))
    try:
        weights.plan(2)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range rank should have been rejected")

    # Resident test 3: authority-driven dequant of the quantized MLA
    # projections equals int8 * scale with exactly one BF16 rounding, and
    # survives a matmul within that rounding.
    q_b_name = "model.language_model.layers.3.self_attn.q_b_proj.weight"
    q_b_stem = "model.language_model.layers.3.self_attn.q_b_proj"
    raw, scale = checkpoint[q_b_name], checkpoint[q_b_stem + ".weight_scale"].reshape(-1)
    dequant = weights._read_source(q_b_name, torch.bfloat16)
    assert dequant.dtype == torch.bfloat16
    assert_close(dequant, (raw.float() * scale.reshape(-1, 1)).to(torch.bfloat16), "mla q_b dequant")
    mla_layer = weights._mla_layer(3, tp_rank=1)
    assert_close(mla_layer["w_q_b"], dequant[16:], "mla q_b tp slice")  # 4 heads x 8 rows, head-major
    probe = torch.randn(6, TINY.q_lora_rank).bfloat16()
    got = (probe.float() @ mla_layer["w_q_b"].float().t()).to(torch.bfloat16)
    reference = (probe.float() @ (raw[16:].float() * scale[16:].reshape(-1, 1)).t()).to(torch.bfloat16)
    # One BF16 rounding of the weight matrix: the error scales with the
    # products' magnitude, so bound it by the reference's span, not an absolute.
    bound = reference.abs().max().item() / 64
    if not torch.allclose(got.float(), reference.float(), rtol=2.0 / 128, atol=bound):
        raise AssertionError("dequantized q_b disagrees with the reference within one BF16 rounding")
    kv_b = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    kv_b_passthrough = weights._read_source("model.language_model.layers.3.self_attn.kv_b_proj.weight", torch.bfloat16)
    assert_close(kv_b_passthrough, kv_b, "float passthrough")
    bad_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update({"model.language_model.layers.3.self_attn.q_b_proj.weight_offset": torch.ones(32, 1)}),
    )
    bad_offset_weights = ResidentPrefillWeights.load(bad_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_offset_weights._read_source(q_b_name, torch.bfloat16)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero MLA weight_offset should have been rejected")
    bad_ffn_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update({"model.layers.1.mlp.experts.0.gate_proj.weight_offset": torch.ones(16, 1)}),
    )
    bad_ffn_weights = ResidentPrefillWeights.load(bad_ffn_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_ffn_weights._moe_layer(1, ep_rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero FFN weight_offset should have been rejected")

    # Resident test 4: every family's rank slices are disjoint and cover the
    # source exactly once, across the bring-up geometries.
    for tp_size, ep_size in ((1, 1), (2, 2), (4, 16)):
        sharded = ResidentPrefillWeights.load(checkpoint, tp_size=tp_size, ep_size=ep_size, model=TINY)
        kda_stacks = [sharded._stack_family("kda", rank % tp_size) for rank in range(tp_size)]
        for stack in kda_stacks:
            if not all(tensor.is_contiguous() for tensor in stack.values()):
                raise AssertionError("every staged family tensor must be contiguous")
        for index, layer in enumerate(TINY.kda_layer_ids):
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.q_proj.weight"]
            rebuilt = torch.cat([stack["w_q"][index] for stack in kda_stacks], dim=0)
            assert_close(rebuilt, source, f"kda w_q coverage tp{tp_size} layer {layer}")
        mla_stacks = [sharded._stack_family("mla", rank % tp_size) for rank in range(tp_size)]
        heads = TINY.num_attention_heads
        for index, layer in enumerate(TINY.indexer_layer_ids):
            prefix = tiny_root(layer) + f"{layer}.self_attn."
            kv_source = checkpoint[prefix + "kv_b_proj.weight"]
            reshaped = kv_source.reshape(heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank)
            key, value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
            assert_close(torch.cat([s["w_k"][index] for s in mla_stacks], dim=0), key, f"kv_b key coverage tp{tp_size}")
            assert_close(
                torch.cat([s["w_v"][index] for s in mla_stacks], dim=0), value,
                f"kv_b value coverage tp{tp_size}",
            )
            # o_proj is W8A8 on the MLA layers: coverage is against the
            # dequantized source, which is what the ranks actually hold.
            o_source = sharded._read_source(prefix + "o_proj.weight", torch.bfloat16)
            assert_close(
                torch.cat([s["w_o"][index] for s in mla_stacks], dim=1), o_source,
                f"mla o_proj column coverage tp{tp_size}",
            )
        expert_ids = [expert for rank in range(ep_size) for expert in sharded.plan(rank).local_expert_ids]
        assert sorted(expert_ids) == list(range(TINY.n_routed_experts))
        layer_one = 1
        moe_per_rank = [sharded._moe_layer(layer_one, rank) for rank in range(ep_size)]
        for piece in moe_per_rank:
            if not all(tensor.is_contiguous() for tensor in piece.values()):
                raise AssertionError("every staged moe tensor must be contiguous")
        rebuilt_experts = torch.cat([piece["w_gate_up"] for piece in moe_per_rank], dim=0)

        def expert_tensor(expert: int, part: str) -> torch.Tensor:
            stem = tiny_root(layer_one) + f"{layer_one}.mlp.experts.{expert}.{part}_proj.weight"
            return checkpoint[stem]

        expected_experts = torch.stack(
            [torch.cat([expert_tensor(expert, "gate"), expert_tensor(expert, "up")], dim=0) for expert in range(TINY.n_routed_experts)],
        )
        assert_close(rebuilt_experts, expected_experts, f"routed expert coverage ep{ep_size}")
    for bad_call in (
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=8, ep_size=8, model=TINY),  # 4 MLA heads
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=3, ep_size=3, model=TINY),  # unsupported
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=32, model=TINY),  # unsupported EP
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=4, ep_size=2, model=TINY),  # EP % TP
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid TP/EP geometry should have been rejected")

    # Resident test 5: the kv_b halves land reduction-first ([QK, KV_LORA]),
    # match the inline split sliced per head, and the full-geometry helper is
    # bit-equal to the shared projection golden with complete head coverage.
    kv_source = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    reshaped = kv_source.reshape(TINY.num_attention_heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank)
    expected_key, expected_value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
    assert mla_layer["w_k"].shape == (2, TINY.qk_head_dim, TINY.kv_lora_rank)  # local_h = 4/2
    assert mla_layer["w_v"].shape == (2, TINY.v_head_dim, TINY.kv_lora_rank)
    assert_close(mla_layer["w_k"], expected_key[2:], "reduction-first w_k head slice")
    assert_close(mla_layer["w_v"], expected_value[2:], "reduction-first w_v head slice")
    generator = torch.Generator().manual_seed(202)
    full_kv_b = torch.randn(FLASH.num_attention_heads * (FLASH.qk_head_dim + FLASH.v_head_dim), FLASH.kv_lora_rank, generator=generator).bfloat16(
    )
    golden_key, golden_value = golden_split_kv_b(full_kv_b)
    rank_keys = [split_kv_b(full_kv_b, rank, 16)[0] for rank in range(16)]
    rank_values = [split_kv_b(full_kv_b, rank, 16)[1] for rank in range(16)]
    assert_close(torch.cat(rank_keys, dim=0), golden_key, "split_kv_b covers every head exactly once")
    assert_close(torch.cat(rank_values, dim=0), golden_value, "split_kv_b value coverage")
    assert rank_keys[3].shape == (FLASH.num_attention_heads // 16, FLASH.qk_head_dim, FLASH.kv_lora_rank)

    # Resident test 6: the Hadamard table is the scaled Sylvester form and
    # orthonormal at BF16 tolerance.
    singles = weights._single_tensors()
    hadamard = singles["hadamard"]
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < TINY.index_head_dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    local_build = (matrix * TINY.index_head_dim**-0.5).to(torch.bfloat16)
    assert hadamard.dtype == torch.bfloat16 and torch.equal(hadamard, local_build)
    # The full-width table is the sylvester_hadamard the loader imports,
    # checked orthonormal at INDEX_DIM = 128.
    big = sylvester_hadamard(128)
    if not torch.allclose(big.float() @ big.float().t(), torch.eye(128), atol=2e-2):
        raise AssertionError("the Hadamard table is not orthonormal at BF16 tolerance")

    # Resident test 7: the indexer projections are transposed once into the
    # kernels' K-major [in, out] layout.
    indexer_stack = weights._stack_family("indexer", tp_rank=0)
    transposed = (
        ("w_q_b", "wq_b.weight"),
        ("w_k", "wk.weight"),
        ("w_weights", "weights_proj.weight"),
        ("w_compress_gate", "index_kpool_compress_gate"),
    )
    passthrough = (
        ("k_norm_w", "k_norm.weight"),
        ("k_norm_b", "k_norm.bias"),
        ("compress_ape", "index_kpool_compress_ape"),
    )
    for index, layer in enumerate(TINY.indexer_layer_ids):
        for member, stem in transposed:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(indexer_stack[member][index], source.t(), f"indexer {member} transpose layer {layer}")
        for member, stem in passthrough:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(indexer_stack[member][index], source, f"indexer {member} passthrough layer {layer}")

    # Resident test 8: backbone tables and independently addressed MTP tables.
    assert weights.host_bytes == 2 * TINY.vocab_size * TINY.hidden_size * 2
    assert not any("embed" in name or "head" in name for name in weights.tensor_layouts)

    def nudge_mtp_embed(tensors, labels):
        tensors["model.language_model.layers.4.embed_tokens.weight"][0].add_(1.0)
        tensors["model.language_model.layers.4.shared_head.head.weight"][0].add_(2.0)

    untied_weights = ResidentPrefillWeights.load(
        build_tiny_fixture(mutate=nudge_mtp_embed), tp_size=2, ep_size=2, model=TINY,
    )
    mtp_rows = untied_weights.lookup_rows(torch.tensor([0]), mtp=True)
    backbone_rows = untied_weights.lookup_rows(torch.tensor([0]))
    assert_close(mtp_rows, backbone_rows + 1, "MTP keeps its untied embedding values")
    mtp_head = untied_weights.mtp_head_shard(0)
    assert_close(mtp_head[0], untied_weights.lm_head()[0] + 2, "MTP keeps its untied head values")
    rows = weights.lookup_rows(torch.tensor([[3, 0], [127, 5]]))
    assert rows.shape == (2, 2, TINY.hidden_size) and rows.dtype == torch.bfloat16
    embed_table = checkpoint["model.language_model.embed_tokens.weight"]
    assert_close(rows[1, 0], embed_table[127], "lookup_rows gathers the host embed table")
    assert_close(rows[0, 1], embed_table[0], "lookup_rows keeps the request shape")
    assert weights.lm_head().shape == (TINY.vocab_size, TINY.hidden_size)
    try:
        weights.lookup_rows(torch.tensor([TINY.vocab_size]))
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-vocabulary token id should have been rejected")

    # Rank payload order, layer offsets and byte coverage are pure CPU contracts.
    def raw_bytes(tensor: torch.Tensor) -> bytes:
        return bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())

    per_name: dict[str, list[tuple[int, int]]] = {}
    blobs: dict[str, bytearray] = {}
    upload_order = []
    for name, data, offset in weights.iter_rank_uploads(rank=1):
        nbytes = data.numel() * data.element_size()
        upload_order.append(name)
        per_name.setdefault(name, []).append((offset, nbytes))
        blob = blobs.setdefault(name, bytearray(weights.stacks[name].bytes))
        blob[offset:offset + nbytes] = raw_bytes(data)
    for name, plan in weights.stacks.items():
        copies = per_name.get(name, [])
        if not copies:
            raise AssertionError(f"stack {name} was never staged")
        if sum(nbytes for _, nbytes in copies) != plan.bytes:
            raise AssertionError(f"stack {name} staged {sum(n for _, n in copies)} of {plan.bytes} bytes")
        if plan.family in ("norm", "mhc", "router", "kda", "mla", "indexer", "moe"):
            layer_bytes = math.prod(plan.shape[1:]) * torch.empty((), dtype=plan.dtype).element_size()
            offsets = [offset for offset, _ in copies]
            if offsets != [index * layer_bytes for index in range(len(plan.layers))]:
                raise AssertionError(f"stack {name} was not staged one layer at a time in order")
            if any(nbytes != layer_bytes for _, nbytes in copies):
                raise AssertionError(f"stack {name} staged a partial layer slice")
        elif copies != [(0, plan.bytes)]:
            raise AssertionError(f"unstacked resident {name} must be one whole payload")
    first_seen = list(dict.fromkeys(name.split(".")[0] for name in upload_order))
    expected_order = ["norm", "mtp", "hadamard", "mhc", "router", "kda", "mla", "indexer", "dense", "moe"]
    if first_seen != expected_order:
        raise AssertionError(f"staging order {first_seen} != {expected_order}")

    # Compare streamed bytes with independently materialized whole stacks.
    reference: dict[str, torch.Tensor] = {}
    for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
        for member, stack in weights._stack_family(family, tp_rank=1).items():
            reference[f"{family}.{member}"] = stack
    for member, stack in weights._stack_dense(tp_rank=1).items():
        reference[f"dense.{member}"] = stack
    for member in (
        "w_gate_up",
        "w_gate_up_scale",
        "w_down",
        "w_down_scale",
        "shared_w_gate_up",
        "shared_w_gate_up_scale",
        "shared_w_down",
        "shared_w_down_scale",
    ):
        reference[f"moe.{member}"] = torch.stack(
            [weights._moe_layer(layer, 1)[member] for layer in weights._family_layers["moe"]],
        )
    for name, tensor in weights._single_tensors().items():
        reference[name] = tensor
    if set(reference) != set(blobs):
        raise AssertionError("the parity reference and the staged stacks disagree on names")
    for name, tensor in reference.items():
        staged = bytes(blobs[name])
        if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
            physical = torch.frombuffer(blobs[name], dtype=tensor.dtype).reshape(tensor.shape)
            staged = raw_bytes(unpack_int8_nz(physical))
        if staged != raw_bytes(tensor):
            raise AssertionError(f"per-layer staging of {name} is not byte-identical to the stack")
    print(f"[GOLDEN] PASS weights upload byte parity ({len(reference)} stacks, rank 1 of TP2/EP2)")

    try:
        list(weights.iter_rank_uploads(rank=2))
    except ValueError:
        pass
    else:
        raise AssertionError("payload iteration with an out-of-range rank should have been rejected")
    copies = len(upload_order)
    message = f"[GOLDEN] PASS weights resident loader (tiny checkpoint, TP1-4/EP1-16, {copies} payloads)"
    print(message)

    class _FullGeometryCheckpoint(Mapping[str, torch.Tensor]):
        """Geometry-only checkpoint view: canonical names, no files, no payloads."""

        def __init__(self):
            model = FLASH
            names: dict[str, str] = {}
            for layer in range(model.num_checkpoint_layers):
                base = f"model.layers.{layer}."
                if layer in model.kda_layer_ids:
                    for part in ("q", "k", "v"):
                        names[base + f"self_attn.{part}_proj.weight"] = "FLOAT"
                        names[base + f"self_attn.{part}_conv1d.weight"] = "FLOAT"
                    for stem in ("dt_bias", "A_log", "f_a_proj.weight", "f_b_proj.weight"):
                        names[base + "self_attn.forget_gate." + stem] = "FLOAT"
                    kda_stems = (
                        "g_a_proj.weight",
                        "g_b_proj.weight",
                        "b_proj.weight",
                        "o_norm.weight",
                        "o_proj.weight",
                    )
                    for stem in kda_stems:
                        names[base + "self_attn." + stem] = "FLOAT"
                else:
                    quant_stems = ("q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight", "o_proj.weight")
                    for stem in quant_stems:
                        names[base + "self_attn." + stem] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_offset"] = "W8A8_DYNAMIC"
                    for stem in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "kv_b_proj.weight"):
                        names[base + "self_attn." + stem] = "FLOAT"
                    for stem in (
                        "wq_b.weight",
                        "wk.weight",
                        "k_norm.weight",
                        "k_norm.bias",
                        "weights_proj.weight",
                        "index_kpool_compress_gate",
                        "index_kpool_compress_ape",
                    ):
                        names[base + "self_attn.indexer." + stem] = "FLOAT"
                if layer < model.num_hidden_layers:
                    for kind in ("attn", "ffn"):
                        names[base + f"{kind}_hc.fn"] = "FLOAT"
                        names[base + f"{kind}_hc.scale"] = "FLOAT"
                        names[base + f"{kind}_hc.base"] = "FLOAT"
                for stem in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    names[base + stem] = "FLOAT"
                if layer >= model.first_k_dense_replace:
                    names[base + "mlp.gate.weight"] = "FLOAT"
                    names[base + "mlp.gate.e_score_correction_bias"] = "FLOAT"
                    bases = [f"mlp.experts.{expert}." for expert in range(model.n_routed_experts)]
                    bases.append("mlp.shared_experts.")
                else:
                    bases = ["mlp."]
                for expert_base in bases:
                    for part in ("gate_proj", "up_proj", "down_proj"):
                        names[base + expert_base + part + ".weight"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_offset"] = "W8A8_DYNAMIC"
            mtp = f"model.layers.{model.mtp_layer_id}."
            for stem in (
                "eh_proj.weight",
                "enorm.weight",
                "hnorm.weight",
                "embed_tokens.weight",
                "shared_head.head.weight",
                "shared_head.norm.weight",
            ):
                names[mtp + stem] = "FLOAT"
            names["model.language_model.norm.weight"] = "FLOAT"
            names["model.language_model.embed_tokens.weight"] = "FLOAT"
            names["lm_head.weight"] = "FLOAT"
            self._names = names

        def __contains__(self, name: object) -> bool:
            return name in self._names

        def __iter__(self):
            return iter(self._names)

        def __len__(self) -> int:
            return len(self._names)

        def __getitem__(self, name: str) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

        @property
        def description(self) -> dict[str, str]:
            return self._names

        def read(self, name: str, selection=None) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

    def family_bytes(resident, family: str, *, shared: bool | None = None) -> int:
        total = 0
        for plan in resident.stacks.values():
            if plan.family != family:
                continue
            if shared is True and not plan.name.startswith("moe.shared"):
                continue
            if shared is False and plan.name.startswith("moe.shared"):
                continue
            total += plan.bytes
        return total

    TP16_COMPONENTS = {
        "kda": 652_252_192,
        "mla": 364_953_600,
        "indexer": 179_324_928,
        "mhc": 70_788_600,
        "norm": 753_664,
        "router": 101_499_264,
        "routed": 19_503_710_208,
        "shared": 1_083_539_456,
        "dense": 28_379_136,
        "mtp": 67_133_440,
        "single": 40_960,
    }
    TP16_TOTAL = 22_052_375_448  # 22.05 GB per rank, the design's headline
    TP1_OVERRIDES = {"kda": 9_366_356_992, "mla": 2_818_621_440, "dense": 453_328_896}
    TP1_TOTAL = 33_645_097_848
    for tp_size, overrides, total in ((16, {}, TP16_TOTAL), (1, TP1_OVERRIDES, TP1_TOTAL)):
        resident = ResidentPrefillWeights.load(_FullGeometryCheckpoint(), tp_size=tp_size, ep_size=16)
        components = {**TP16_COMPONENTS, **overrides}
        for family in ("kda", "mla", "indexer", "mhc", "norm", "router", "dense", "mtp", "single"):
            got = family_bytes(resident, family)
            if got != components[family]:
                raise AssertionError(f"TP{tp_size} {family} budget {got} != design value {components[family]}")
        routed = family_bytes(resident, "moe", shared=False)
        shared = family_bytes(resident, "moe", shared=True)
        if (routed, shared) != (components["routed"], components["shared"]):
            raise AssertionError(f"TP{tp_size} moe budget {(routed, shared)} != design values")
        if resident.bytes_per_rank != total:
            raise AssertionError(f"TP{tp_size}/EP16 budget {resident.bytes_per_rank} != design total {total}")
        if resident.host_bytes != 2 * FLASH.vocab_size * FLASH.hidden_size * 2:
            raise AssertionError("host bytes must be the head plus the embedding table")
        assert resident.plan(0).device_bytes == resident.bytes_per_rank
    if abs(TP16_TOTAL / 1e9 - 22.05) > 0.01:
        raise AssertionError("the TP16 budget drifted off the design's 22.05 GB headline")
    print("[GOLDEN] PASS weights resident budget (TP16/EP16 22.05 GB, TP1/EP16 33.65 GB, headers only)")


def run_reference_checks_cli(argv=None) -> None:
    """Run the CPU self-test and optionally probe a real W8A8 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="optional real W8A8 checkpoint root for a read/dequant/coverage smoke")
    args = parser.parse_args(argv)
    # Tiny-tensor CPU math parallelizes badly past a few dozen threads (this
    # host defaults to 320); cap the self-test only, library imports stay pure.
    torch.set_num_threads(min(32, torch.get_num_threads()))
    run_self_tests(checkpoint_root=args.checkpoint)


def main(argv=None):
    """Run bounded CPU contracts; generation is owned by pypto-serving."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check", choices=["reference", "weights"], default="weights")
    selected, remaining = parser.parse_known_args(argv)
    if selected.check == "reference":
        return run_reference_checks_cli(remaining)
    return run_weight_checks(remaining)


if __name__ == "__main__":
    main()
