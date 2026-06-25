# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 packed prefill single layer with MoE EP2."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# Import moe first. It applies the EP2 FLASH override before dependent
# modules bake config-derived MoE shapes.
from moe import (
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
    W_PAD,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from config import (
    FLASH as MODEL_CONFIG,
    PREFILL_BATCH,
    PREFILL_SEQ,
    PREFILL_CHUNK_TOKENS,
    USER_BATCH_DYN,
    PREFILL_TOKENS_DYN,
    PREFILL_ORI_BLOCK_TABLE_DYN,
    PREFILL_CMP_BLOCK_TABLE_DYN,
    PREFILL_IDX_BLOCK_TABLE_DYN,
    PREFILL_ORI_CACHE_BLOCKS_DYN,
    PREFILL_CMP_CACHE_BLOCKS_DYN,
    PREFILL_IDX_CACHE_BLOCKS_DYN,
    PREFILL_HCA_STATE_BLOCKS_DYN,
    PREFILL_CSA_STATE_BLOCKS_DYN,
    PREFILL_INNER_STATE_BLOCKS_DYN,
    PREFILL_HCA_STATE_BLOCK_TABLE_DYN,
    PREFILL_CSA_STATE_BLOCK_TABLE_DYN,
    PREFILL_INNER_STATE_BLOCK_TABLE_DYN,
)
from prefill_attention_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    build_tensor_specs as build_swa_attention_tensor_specs,
    golden_prefill_attention_swa,
    prefill_attention_swa,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    golden_prefill_attention_hca,
    prefill_attention_hca,
)
from prefill_attention_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_TOPK,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    START_POS,
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_prefill_attention_csa,
    prefill_attention_csa,
)


assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM
assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM
assert T == PREFILL_CHUNK_TOKENS

@pl.jit
def prefill_layer(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[
        [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    if layer_id < 2:
        prefill_attention_swa(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            kv_cache, ori_block_table, ori_slot_mapping,
            cmp_sparse_indices, cmp_sparse_lens, position_ids,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn, num_tokens,
        )
    elif layer_id % 2 == 1:
        prefill_attention_hca(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
            hca_cmp_kv_state, hca_cmp_score_state, hca_compress_state_block_table,
            kv_cache, ori_slot_mapping, ori_block_table,
            cmp_kv, cmp_block_table, cmp_sparse_indices, cmp_sparse_lens,
            position_ids, hca_cmp_slot_mapping, hca_state_slot_mapping,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn, num_tokens,
        )
    else:
        prefill_attention_csa(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
            csa_cmp_kv_state, csa_cmp_score_state, csa_compress_state_block_table,
            csa_hadamard_idx,
            csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
            csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
            csa_inner_kv_state, csa_inner_score_state, csa_inner_compress_state_block_table,
            kv_cache, ori_block_table, ori_slot_mapping,
            cmp_kv, cmp_block_table, idx_kv_cache, idx_block_table,
            position_ids, csa_cmp_slot_mapping, csa_idx_slot_mapping,
            csa_state_slot_mapping, csa_inner_state_slot_mapping,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn, num_tokens,
        )
    moe(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        pub_counts, count_done, data_done,
        recv_x, recv_scale, recv_w, recv_r_route,
        routed_y_buf, combine_done,
        layer_id, num_tokens, my_rank, moe_epoch,
    )
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16],
    seq_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_HCA_STATE_BLOCK_TABLE_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCK_TABLE_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCK_TABLE_DYN], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[N_RANKS, PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[N_RANKS, PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[N_RANKS, PREFILL_IDX_BLOCK_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(1, USER_BATCH_DYN)
    chunk_lens.bind_dynamic(1, USER_BATCH_DYN)
    chunk_offsets.bind_dynamic(1, USER_BATCH_DYN)
    hca_cmp_kv_state.bind_dynamic(1, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_cmp_score_state.bind_dynamic(1, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(1, PREFILL_HCA_STATE_BLOCK_TABLE_DYN)
    csa_cmp_kv_state.bind_dynamic(1, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_cmp_score_state.bind_dynamic(1, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(1, PREFILL_CSA_STATE_BLOCK_TABLE_DYN)
    csa_inner_kv_state.bind_dynamic(1, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_score_state.bind_dynamic(1, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, PREFILL_INNER_STATE_BLOCK_TABLE_DYN)
    kv_cache.bind_dynamic(1, PREFILL_ORI_CACHE_BLOCKS_DYN)
    ori_block_table.bind_dynamic(1, PREFILL_ORI_BLOCK_TABLE_DYN)
    ori_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    cmp_kv.bind_dynamic(1, PREFILL_CMP_CACHE_BLOCKS_DYN)
    cmp_block_table.bind_dynamic(1, PREFILL_CMP_BLOCK_TABLE_DYN)
    cmp_sparse_indices.bind_dynamic(1, PREFILL_TOKENS_DYN)
    cmp_sparse_lens.bind_dynamic(1, PREFILL_TOKENS_DYN)
    idx_kv_cache.bind_dynamic(1, PREFILL_IDX_CACHE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(1, PREFILL_IDX_BLOCK_TABLE_DYN)
    position_ids.bind_dynamic(1, PREFILL_TOKENS_DYN)
    hca_cmp_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    hca_state_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    csa_cmp_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    csa_idx_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    csa_state_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(1, PREFILL_TOKENS_DYN)
    input_ids.bind_dynamic(1, PREFILL_TOKENS_DYN)
    x_next.bind_dynamic(1, PREFILL_TOKENS_DYN)

    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for rank in pl.range(pld.world_size()):
        pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        for request_id in pl.range(PREFILL_BATCH):
            moe_epoch = pl.cast(request_id + 1, pl.INT32)
            token_base = request_id * T

            prefill_layer(
                x_hc[rank, token_base:token_base + T],
                hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
                wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
                hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
                hca_cmp_kv_state[rank, request_id * HCA_STATE_BLOCK_NUM:(request_id + 1) * HCA_STATE_BLOCK_NUM],
                hca_cmp_score_state[rank, request_id * HCA_STATE_BLOCK_NUM:(request_id + 1) * HCA_STATE_BLOCK_NUM],
                hca_compress_state_block_table[rank, request_id * HCA_STATE_MAX_BLOCKS:(request_id + 1) * HCA_STATE_MAX_BLOCKS],
                csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
                csa_cmp_kv_state[rank, request_id * CSA_STATE_BLOCK_NUM:(request_id + 1) * CSA_STATE_BLOCK_NUM],
                csa_cmp_score_state[rank, request_id * CSA_STATE_BLOCK_NUM:(request_id + 1) * CSA_STATE_BLOCK_NUM],
                csa_compress_state_block_table[rank, request_id * CSA_STATE_MAX_BLOCKS:(request_id + 1) * CSA_STATE_MAX_BLOCKS],
                csa_hadamard_idx[rank],
                csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
                csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
                csa_inner_kv_state[rank, request_id * INNER_STATE_BLOCK_NUM:(request_id + 1) * INNER_STATE_BLOCK_NUM],
                csa_inner_score_state[rank, request_id * INNER_STATE_BLOCK_NUM:(request_id + 1) * INNER_STATE_BLOCK_NUM],
                csa_inner_compress_state_block_table[rank, request_id * INNER_STATE_MAX_BLOCKS:(request_id + 1) * INNER_STATE_MAX_BLOCKS],
                kv_cache[rank, request_id * CSA_ORI_BLOCK_NUM:(request_id + 1) * CSA_ORI_BLOCK_NUM],
                ori_block_table[rank, request_id * SPARSE_ORI_MAX_BLOCKS:(request_id + 1) * SPARSE_ORI_MAX_BLOCKS],
                ori_slot_mapping[rank, token_base:token_base + T],
                cmp_kv[rank, request_id * CSA_CMP_BLOCK_NUM:(request_id + 1) * CSA_CMP_BLOCK_NUM],
                cmp_block_table[rank, request_id * SPARSE_CMP_MAX_BLOCKS:(request_id + 1) * SPARSE_CMP_MAX_BLOCKS],
                cmp_sparse_indices[rank, token_base:token_base + T],
                cmp_sparse_lens[rank, token_base:token_base + T],
                idx_kv_cache[rank, request_id * CSA_CMP_BLOCK_NUM:(request_id + 1) * CSA_CMP_BLOCK_NUM],
                idx_block_table[rank, request_id * IDX_CACHE_MAX_BLOCKS:(request_id + 1) * IDX_CACHE_MAX_BLOCKS],
                position_ids[rank, token_base:token_base + T],
                hca_cmp_slot_mapping[rank, token_base:token_base + T],
                hca_state_slot_mapping[rank, token_base:token_base + T],
                csa_cmp_slot_mapping[rank, token_base:token_base + T],
                csa_idx_slot_mapping[rank, token_base:token_base + T],
                csa_state_slot_mapping[rank, token_base:token_base + T],
                csa_inner_state_slot_mapping[rank, token_base:token_base + T],
                attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
                hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
                norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
                input_ids[rank, token_base:token_base + T],
                routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
                routed_w2[rank], routed_w2_scale[rank],
                shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
                shared_w2[rank], shared_w2_scale[rank],
                x_next[rank, token_base:token_base + T],
                pub_counts, count_done, data_done,
                recv_x, recv_scale, recv_w, recv_r_route,
                routed_y_buf, combine_done,
                num_tokens, layer_id, rank, moe_epoch,
                device=rank,
            )

HOST_TENSOR_ORDER = (
    "x_hc",
    "seq_lens",
    "chunk_lens",
    "chunk_offsets",
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
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_cmp_kv_state",
    "hca_cmp_score_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_cmp_kv_state",
    "csa_cmp_score_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_kv_state",
    "csa_inner_score_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "cmp_kv",
    "cmp_block_table",
    "cmp_sparse_indices",
    "cmp_sparse_lens",
    "idx_kv_cache",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_next",
)


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def ranked_init(spec, n_ranks, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        return torch.stack(values, dim=0).contiguous()

    return init


def ranked_x_hc_init(spec, n_ranks, active_tokens, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        stacked = torch.stack(values, dim=0).contiguous()
        active = min(active_tokens, stacked.shape[1])
        if active < stacked.shape[1]:
            inactive = torch.randn(stacked[:, active:].shape, dtype=torch.float32).to(stacked.dtype)
            stacked[:, active:] = inactive / 10.0
        return stacked

    return init


def ranked_pack_token_init(spec, n_ranks, chunk_lens, chunk_offsets, torch):
    def init():
        total_tokens = int(chunk_offsets[-1].item()) + spec.shape[0]
        out = torch.zeros([n_ranks, total_tokens, *spec.shape[1:]], dtype=spec.dtype)
        for rank in range(n_ranks):
            for request_id, chunk_len_t in enumerate(chunk_lens):
                chunk_len = int(chunk_len_t.item())
                token_base = int(chunk_offsets[request_id].item())
                if chunk_len <= 0:
                    continue
                value = _spec_value(spec, torch)
                out[rank, token_base:token_base + chunk_len] = value[:chunk_len]
        return out.contiguous()

    return init


def ranked_pack_request_init(spec, n_ranks, batch, torch):
    def init():
        per_request_shape = spec.shape
        out = torch.empty([n_ranks, batch * per_request_shape[0], *per_request_shape[1:]], dtype=spec.dtype)
        for rank in range(n_ranks):
            for request_id in range(batch):
                value = _spec_value(spec, torch)
                base = request_id * per_request_shape[0]
                out[rank, base:base + per_request_shape[0]] = value
        return out.contiguous()

    return init


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def build_tensor_specs(start_pos=START_POS, num_tokens=T, layer_id=2, batch=PREFILL_BATCH, prefill_seq=PREFILL_SEQ):
    import torch
    from golden import ScalarSpec, TensorSpec

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if batch != PREFILL_BATCH:
        raise ValueError(
            f"batch must match config.PREFILL_BATCH for l3 host orchestration: "
            f"batch={batch}, PREFILL_BATCH={PREFILL_BATCH}"
        )
    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if prefill_seq < start_pos + num_tokens:
        raise ValueError(
            f"prefill_seq must cover the requested chunk: prefill_seq={prefill_seq}, "
            f"start_pos={start_pos}, num_tokens={num_tokens}"
        )

    seq_lens_values = torch.full((batch,), start_pos + num_tokens, dtype=torch.int32)
    chunk_lens_values = torch.full((batch,), num_tokens, dtype=torch.int32)
    chunk_offsets_values = torch.arange(batch, dtype=torch.int32) * T
    total_tokens = batch * T

    def kind_specs(build_fn):
        return {s.name: s for s in build_fn(start_pos=start_pos, num_tokens=num_tokens) if isinstance(s, TensorSpec)}

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active_kind = _attention_kind_for_layer(layer_id)
    active = {"swa": swa, "hca": hca, "csa": csa}[active_kind]

    # (layer_name, source_spec). Shared state is taken from the active kind (its
    # init is what the active attention + its golden both consume). The hca_/csa_
    # compressor + indexer params are namespaced from their own kind; cmp_kv and
    # the idx caches fall back to the owning kind when the active kind lacks them.
    attention_specs = [
        ("x_hc", active["x_hc"]),
        ("hc_attn_fn", active["hc_attn_fn"]),
        ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]),
        ("wq_b", active["wq_b"]),
        ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]),
        ("gamma_cq", active["gamma_cq"]),
        ("gamma_ckv", active["gamma_ckv"]),
        ("freqs_cos", active["freqs_cos"]),
        ("freqs_sin", active["freqs_sin"]),
        ("hca_cmp_wkv", hca["cmp_wkv"]),
        ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]),
        ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("hca_cmp_kv_state", hca["cmp_kv_state"]),
        ("hca_cmp_score_state", hca["cmp_score_state"]),
        ("hca_compress_state_block_table", hca["compress_state_block_table"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_cmp_kv_state", csa["cmp_kv_state"]),
        ("csa_cmp_score_state", csa["cmp_score_state"]),
        ("csa_compress_state_block_table", csa["compress_state_block_table"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("csa_inner_kv_state", csa["inner_kv_state"]),
        ("csa_inner_score_state", csa["inner_score_state"]),
        ("csa_inner_compress_state_block_table", csa["inner_compress_state_block_table"]),
        ("kv_cache", active["kv_cache"]),
        ("ori_block_table", active.get("ori_block_table", swa.get("block_table"))),
        ("ori_slot_mapping", active["ori_slot_mapping"]),
        ("cmp_kv", active.get("cmp_kv", hca["cmp_kv"])),
        ("cmp_block_table", active.get("cmp_block_table", hca["cmp_block_table"])),
        ("cmp_sparse_indices", active.get("cmp_sparse_indices", swa["cmp_sparse_indices"])),
        ("cmp_sparse_lens", active.get("cmp_sparse_lens", swa["cmp_sparse_lens"])),
        ("idx_kv_cache", csa["idx_kv_cache"]),
        ("idx_block_table", csa["idx_block_table"]),
        ("position_ids", active["position_ids"]),
        ("hca_cmp_slot_mapping", hca["cmp_slot_mapping"]),
        ("hca_state_slot_mapping", hca["state_slot_mapping"]),
        ("csa_cmp_slot_mapping", csa["cmp_slot_mapping"]),
        ("csa_idx_slot_mapping", csa["idx_slot_mapping"]),
        ("csa_state_slot_mapping", csa["state_slot_mapping"]),
        ("csa_inner_state_slot_mapping", csa["inner_state_slot_mapping"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    token_spec_names = {
        "x_hc",
        "ori_slot_mapping",
        "cmp_sparse_indices",
        "cmp_sparse_lens",
        "position_ids",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    }
    request_spec_names = {
        "kv_cache",
        "ori_block_table",
        "cmp_kv",
        "cmp_block_table",
        "idx_kv_cache",
        "idx_block_table",
        "hca_cmp_kv_state",
        "hca_cmp_score_state",
        "hca_compress_state_block_table",
        "csa_cmp_kv_state",
        "csa_cmp_score_state",
        "csa_compress_state_block_table",
        "csa_inner_kv_state",
        "csa_inner_score_state",
        "csa_inner_compress_state_block_table",
    }

    tensor_specs = [
        TensorSpec("x_hc", [N_RANKS, total_tokens, HC_MULT, D], torch.bfloat16,
                   init_value=ranked_pack_token_init(active["x_hc"], N_RANKS, chunk_lens_values, chunk_offsets_values, torch)),
        TensorSpec("seq_lens", [N_RANKS, batch], torch.int32,
                   init_value=lambda: seq_lens_values.unsqueeze(0).expand(N_RANKS, -1).contiguous()),
        TensorSpec("chunk_lens", [N_RANKS, batch], torch.int32,
                   init_value=lambda: chunk_lens_values.unsqueeze(0).expand(N_RANKS, -1).contiguous()),
        TensorSpec("chunk_offsets", [N_RANKS, batch], torch.int32,
                   init_value=lambda: chunk_offsets_values.unsqueeze(0).expand(N_RANKS, -1).contiguous()),
    ]

    for name, src in attention_specs:
        if name == "x_hc":
            continue
        if name in token_spec_names:
            tensor_specs.append(TensorSpec(
                name,
                [N_RANKS, total_tokens, *src.shape[1:]],
                src.dtype,
                init_value=ranked_pack_token_init(src, N_RANKS, chunk_lens_values, chunk_offsets_values, torch),
                is_output=src.is_output,
            ))
        elif name in request_spec_names:
            tensor_specs.append(TensorSpec(
                name,
                [N_RANKS, batch * src.shape[0], *src.shape[1:]],
                src.dtype,
                init_value=ranked_pack_request_init(src, N_RANKS, batch, torch),
                is_output=src.is_output,
            ))
        else:
            tensor_specs.append(TensorSpec(
                name,
                [N_RANKS, *src.shape],
                src.dtype,
                init_value=ranked_init(src, N_RANKS, torch),
                is_output=src.is_output,
            ))

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
            def init_input_ids(spec=spec):
                rows = []
                base_ids = torch.arange(total_tokens, dtype=spec.dtype)
                for rank in range(N_RANKS):
                    rows.append(torch.roll(base_ids, shifts=rank))
                return torch.stack(rows, dim=0).contiguous()

            tensor_specs.append(TensorSpec(spec.name, [N_RANKS, total_tokens], spec.dtype, init_value=init_input_ids))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, total_tokens, HC_MULT, D], torch.bfloat16, is_output=True))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def golden_prefill_layer(tensors):
    import torch
    from golden import TensorSpec

    kind = _attention_kind_for_layer(int(tensors["layer_id"]))
    seq_lens = tensors["seq_lens"]
    chunk_lens = tensors["chunk_lens"]
    chunk_offsets = tensors["chunk_offsets"]
    batch = seq_lens.shape[1]

    # Un-namespace the active kind's params back to the attention-local names its
    # golden expects (decode-style mapped dict), then build a per-rank view.
    mapped = dict(tensors)
    if kind == "swa":
        mapped["block_table"] = tensors["ori_block_table"]
        attention_golden = golden_prefill_attention_swa
    elif kind == "hca":
        mapped.update({
            "cmp_wkv": tensors["hca_cmp_wkv"],
            "cmp_wgate": tensors["hca_cmp_wgate"],
            "cmp_ape": tensors["hca_cmp_ape"],
            "cmp_norm_w": tensors["hca_cmp_norm_w"],
            "cmp_kv_state": tensors["hca_cmp_kv_state"],
            "cmp_score_state": tensors["hca_cmp_score_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"],
            "state_slot_mapping": tensors["hca_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_hca
    else:
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"],
            "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"],
            "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "cmp_kv_state": tensors["csa_cmp_kv_state"],
            "cmp_score_state": tensors["csa_cmp_score_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "hadamard_idx": tensors["csa_hadamard_idx"],
            "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"],
            "idx_weights_proj": tensors["csa_weights_proj"],
            "inner_wkv": tensors["csa_inner_wkv"],
            "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"],
            "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_kv_state": tensors["csa_inner_kv_state"],
            "inner_score_state": tensors["csa_inner_score_state"],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"],
            "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
            "state_slot_mapping": tensors["csa_state_slot_mapping"],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_csa

    token_names = {
        "x_hc",
        "ori_slot_mapping",
        "cmp_sparse_indices",
        "cmp_sparse_lens",
        "position_ids",
        "cmp_slot_mapping",
        "idx_slot_mapping",
        "state_slot_mapping",
        "inner_state_slot_mapping",
    }
    request_slices = {
        "kv_cache": CSA_ORI_BLOCK_NUM,
        "block_table": SPARSE_ORI_MAX_BLOCKS,
        "ori_block_table": SPARSE_ORI_MAX_BLOCKS,
        "cmp_kv": CSA_CMP_BLOCK_NUM,
        "cmp_block_table": SPARSE_CMP_MAX_BLOCKS,
        "idx_kv_cache": CSA_CMP_BLOCK_NUM,
        "idx_block_table": IDX_CACHE_MAX_BLOCKS,
        "cmp_kv_state": HCA_STATE_BLOCK_NUM if kind == "hca" else CSA_STATE_BLOCK_NUM,
        "cmp_score_state": HCA_STATE_BLOCK_NUM if kind == "hca" else CSA_STATE_BLOCK_NUM,
        "compress_state_block_table": HCA_STATE_MAX_BLOCKS if kind == "hca" else CSA_STATE_MAX_BLOCKS,
        "inner_kv_state": INNER_STATE_BLOCK_NUM,
        "inner_score_state": INNER_STATE_BLOCK_NUM,
        "inner_compress_state_block_table": INNER_STATE_MAX_BLOCKS,
    }

    def token_chunk(value, rank, token_base, chunk_len):
        out = torch.zeros([T, *value.shape[2:]], dtype=value.dtype)
        if chunk_len > 0:
            out[:chunk_len] = value[rank, token_base:token_base + chunk_len]
        return out

    def request_chunk(value, rank, request_id, rows):
        base = request_id * rows
        return value[rank, base:base + rows]

    for request_id in range(batch):
        chunk_len = int(chunk_lens[0, request_id].item())
        token_base = int(chunk_offsets[0, request_id].item())
        if chunk_len <= 0:
            continue

        if kind == "swa":
            specs = build_swa_attention_tensor_specs(start_pos=int(seq_lens[0, request_id].item()) - chunk_len, num_tokens=chunk_len)
        elif kind == "hca":
            specs = build_hca_attention_tensor_specs(start_pos=int(seq_lens[0, request_id].item()) - chunk_len, num_tokens=chunk_len)
        else:
            specs = build_csa_attention_tensor_specs(start_pos=int(seq_lens[0, request_id].item()) - chunk_len, num_tokens=chunk_len)

        x_attn_req = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=tensors["x_hc"].dtype)
        for rank in range(N_RANKS):
            attn_tensors = {}
            rank_token_base = int(chunk_offsets[rank, request_id].item())
            rank_chunk_len = int(chunk_lens[rank, request_id].item())
            for spec in specs:
                if isinstance(spec, TensorSpec):
                    if spec.name == "x_out":
                        attn_tensors[spec.name] = x_attn_req[rank]
                    elif spec.name in token_names:
                        attn_tensors[spec.name] = token_chunk(mapped[spec.name], rank, rank_token_base, rank_chunk_len)
                    elif spec.name in request_slices:
                        attn_tensors[spec.name] = request_chunk(mapped[spec.name], rank, request_id, request_slices[spec.name])
                    else:
                        attn_tensors[spec.name] = mapped[spec.name][rank]
                else:
                    attn_tensors[spec.name] = rank_chunk_len if spec.name == "num_tokens" else mapped[spec.name]
            attention_golden(attn_tensors)

        input_ids_req = torch.zeros(N_RANKS, T, dtype=tensors["input_ids"].dtype)
        for rank in range(N_RANKS):
            rank_token_base = int(chunk_offsets[rank, request_id].item())
            rank_chunk_len = int(chunk_lens[rank, request_id].item())
            input_ids_req[rank, :rank_chunk_len] = tensors["input_ids"][rank, rank_token_base:rank_token_base + rank_chunk_len]

        x_next_req = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=tensors["x_next"].dtype)
        moe_tensors = dict(tensors)
        moe_tensors["x_hc"] = x_attn_req
        moe_tensors["input_ids"] = input_ids_req
        moe_tensors["x_next"] = x_next_req
        moe_tensors["num_tokens"] = chunk_len
        golden_moe(moe_tensors)

        for rank in range(N_RANKS):
            rank_token_base = int(chunk_offsets[rank, request_id].item())
            rank_chunk_len = int(chunk_lens[rank, request_id].item())
            tensors["x_next"][rank, rank_token_base:rank_token_base + rank_chunk_len] = x_next_req[rank, :rank_chunk_len]


def packed_valid_ratio_reldiff(batch, num_tokens, diff_thd, pct_thd):
    """Relative-diff comparator over the valid prefix of each packed request."""
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def cmp(actual, expected, **kwargs):
        import torch

        actual_parts = []
        expected_parts = []
        for request_id in range(batch):
            token_base = request_id * T
            actual_parts.append(actual[:, token_base:token_base + num_tokens])
            expected_parts.append(expected[:, token_base:token_base + num_tokens])
        return base_cmp(torch.cat(actual_parts, dim=1), torch.cat(expected_parts, dim=1), **kwargs)

    cmp.__name__ = f"packed_valid_ratio_reldiff(batch={batch}, num_tokens={num_tokens})"
    return cmp


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--layer-id", type=int, default=2,
                        help="Layer id selects attention by MODEL_CONFIG.compress_ratios[layer_id].")
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only context_len (multiple of S=WIN).")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Fixture active token count (q_len), capped by T.")
    parser.add_argument("--batch", type=int, default=PREFILL_BATCH,
                        help="Request count for packed prefill fixtures.")
    parser.add_argument("--prefill-seq", type=int, default=PREFILL_SEQ,
                        help="Per-request sequence length/upper bound for packed prefill fixtures.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(
            start_pos=args.start_pos,
            num_tokens=args.num_tokens,
            layer_id=args.layer_id,
            batch=args.batch,
            prefill_seq=args.prefill_seq,
        ),
        golden_fn=golden_prefill_layer,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Real-weight x_next over-thd fractions (frac>5e-3 / frac>1e-2):
            # swa(L0) 0.3% / 0.0%, hca(L9) 1.7% / 0.6%, csa(L8) 5.4% / 0.7%.
            "x_next": packed_valid_ratio_reldiff(args.batch, args.num_tokens, diff_thd=0.01, pct_thd=0.05),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
