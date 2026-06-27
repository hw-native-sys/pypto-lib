# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""DeepSeek-V4 decode layer smoke: attention DP-N followed by MoE EP-N.

Each rank owns a local decode micro-batch for the selected attention stage.
The resulting per-rank hidden states feed the N-rank EP MoE path. The EP world
size is chosen with --ep (2/4/8, default 2), inherited from moe; see __main__.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

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
    golden_attention_swa,
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
    golden_attention_hca,
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
    golden_attention_csa,
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
    golden_moe,
    moe,
)

assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM, "unified host shares cmp_kv between HCA and CSA"
assert HCA_CMP_MAX_BLOCKS == CSA_CMP_MAX_BLOCKS, "unified host shares cmp_block_table between HCA and CSA"


@pl.jit
def decode_layer(
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
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
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
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.Tensor[
        [HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[D, CSA_INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[D, CSA_INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.Tensor[
        [CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
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
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    if layer_id < 2:
        attention_swa(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale,
            wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
            kv_cache, block_table,
            ori_slot_mapping, position_ids,
            cmp_kv, cmp_block_table,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn,
        )
    elif layer_id % 2 == 1:
        attention_hca(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale,
            wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
            hca_compress_state, hca_compress_state_block_table,
            kv_cache, block_table, cmp_kv, cmp_block_table,
            ori_slot_mapping, hca_cmp_slot_mapping, hca_state_slot_mapping,
            position_ids, kv_seq_lens,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn,
        )
    else:
        attention_csa(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale,
            wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin,
            csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
            csa_compress_state, csa_compress_state_block_table,
            csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj, csa_hadamard_idx,
            csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
            csa_inner_compress_state, csa_inner_compress_state_block_table,
            kv_cache, block_table, cmp_kv, cmp_block_table,
            idx_kv_cache, idx_block_table,
            ori_slot_mapping, csa_cmp_slot_mapping, csa_idx_slot_mapping,
            csa_state_slot_mapping, csa_inner_state_slot_mapping,
            position_ids, kv_seq_lens,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn,
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
        layer_id, pl.const(T, pl.INT32), my_rank, pl.const(1, pl.INT32),
    )
    return x_next



@pl.jit.host
def l3_decode_layer(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.Tensor[[N_RANKS, B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
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
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.Tensor[
        [N_RANKS, HCA_COMPRESS_STATE_BLOCK_NUM, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.Tensor[
        [N_RANKS, CSA_MAIN_STATE_BLOCK_NUM, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, D, CSA_INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, D, CSA_INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.Tensor[
        [N_RANKS, CSA_INNER_STATE_BLOCK_NUM, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[N_RANKS, B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[N_RANKS, B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
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
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]],
    layer_id: pl.Scalar[pl.INT32],
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

    for r in pl.range(pld.world_size()):
        pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        decode_layer(
            x_hc[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r], wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r], gamma_cq[r], gamma_ckv[r], freqs_cos[r], freqs_sin[r],
            kv_cache[r], block_table[r],
            ori_slot_mapping[r],
            hca_cmp_slot_mapping[r], hca_state_slot_mapping[r],
            csa_cmp_slot_mapping[r], csa_idx_slot_mapping[r],
            csa_state_slot_mapping[r], csa_inner_state_slot_mapping[r],
            position_ids[r], kv_seq_lens[r],
            attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r], hca_cmp_norm_w[r],
            hca_compress_state[r], hca_compress_state_block_table[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r], csa_cmp_norm_w[r],
            csa_compress_state[r], csa_compress_state_block_table[r],
            csa_idx_wq_b[r], csa_idx_wq_b_scale[r], csa_weights_proj[r], csa_hadamard_idx[r],
            csa_inner_wkv[r], csa_inner_wgate[r], csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_compress_state[r], csa_inner_compress_state_block_table[r],
            cmp_kv[r], cmp_block_table[r], idx_kv_cache[r], idx_block_table[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            layer_id, r,
            device=r,
        )


def golden_decode_layer(tensors):
    import torch

    x_attn = torch.empty_like(tensors["x_hc"])
    for r in range(N_RANKS):
        golden_attention_swa({
            "x_hc": tensors["x_hc"][r],
            "hc_attn_fn": tensors["hc_attn_fn"][r],
            "hc_attn_scale": tensors["hc_attn_scale"][r],
            "hc_attn_base": tensors["hc_attn_base"][r],
            "attn_norm_w": tensors["attn_norm_w"][r],
            "wq_a": tensors["wq_a"][r],
            "wq_b": tensors["wq_b"][r],
            "wq_b_scale": tensors["wq_b_scale"][r],
            "wkv": tensors["wkv"][r],
            "gamma_cq": tensors["gamma_cq"][r],
            "gamma_ckv": tensors["gamma_ckv"][r],
            "freqs_cos": tensors["freqs_cos"][r],
            "freqs_sin": tensors["freqs_sin"][r],
            "kv_cache": tensors["kv_cache"][r],
            "block_table": tensors["block_table"][r],
            "ori_slot_mapping": tensors["ori_slot_mapping"][r],
            "position_ids": tensors["position_ids"][r],
            "cmp_kv": tensors["cmp_kv"][r],
            "cmp_block_table": tensors["cmp_block_table"][r],
            "attn_sink": tensors["attn_sink"][r],
            "wo_a": tensors["wo_a"][r],
            "wo_b": tensors["wo_b"][r],
            "wo_b_scale": tensors["wo_b_scale"][r],
            "x_out": x_attn[r],
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["num_tokens"] = T
    golden_moe(moe_tensors)


def golden_decode_layer_hca(tensors):
    import torch

    x_attn = torch.empty_like(tensors["x_hc"])
    for r in range(N_RANKS):
        golden_attention_hca({
            "x_hc": tensors["x_hc"][r],
            "hc_attn_fn": tensors["hc_attn_fn"][r],
            "hc_attn_scale": tensors["hc_attn_scale"][r],
            "hc_attn_base": tensors["hc_attn_base"][r],
            "attn_norm_w": tensors["attn_norm_w"][r],
            "wq_a": tensors["wq_a"][r],
            "wq_b": tensors["wq_b"][r],
            "wq_b_scale": tensors["wq_b_scale"][r],
            "wkv": tensors["wkv"][r],
            "gamma_cq": tensors["gamma_cq"][r],
            "gamma_ckv": tensors["gamma_ckv"][r],
            "freqs_cos": tensors["freqs_cos"][r],
            "freqs_sin": tensors["freqs_sin"][r],
            "cmp_wkv": tensors["cmp_wkv"][r],
            "cmp_wgate": tensors["cmp_wgate"][r],
            "cmp_ape": tensors["cmp_ape"][r],
            "cmp_norm_w": tensors["cmp_norm_w"][r],
            "compress_state": tensors["compress_state"][r],
            "compress_state_block_table": tensors["compress_state_block_table"][r],
            "kv_cache": tensors["kv_cache"][r],
            "ori_block_table": tensors["ori_block_table"][r],
            "cmp_kv": tensors["cmp_kv"][r],
            "cmp_block_table": tensors["cmp_block_table"][r],
            "ori_slot_mapping": tensors["ori_slot_mapping"][r],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"][r],
            "state_slot_mapping": tensors["state_slot_mapping"][r],
            "position_ids": tensors["position_ids"][r],
            "kv_seq_lens": tensors["kv_seq_lens"][r],
            "attn_sink": tensors["attn_sink"][r],
            "wo_a": tensors["wo_a"][r],
            "wo_b": tensors["wo_b"][r],
            "wo_b_scale": tensors["wo_b_scale"][r],
            "x_out": x_attn[r],
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["num_tokens"] = T
    golden_moe(moe_tensors)


def golden_decode_layer_csa(tensors):
    import torch

    x_attn = torch.empty_like(tensors["x_hc"])
    for r in range(N_RANKS):
        golden_attention_csa({
            "x_hc": tensors["x_hc"][r],
            "hc_attn_fn": tensors["hc_attn_fn"][r],
            "hc_attn_scale": tensors["hc_attn_scale"][r],
            "hc_attn_base": tensors["hc_attn_base"][r],
            "attn_norm_w": tensors["attn_norm_w"][r],
            "wq_a": tensors["wq_a"][r],
            "wq_b": tensors["wq_b"][r],
            "wq_b_scale": tensors["wq_b_scale"][r],
            "wkv": tensors["wkv"][r],
            "gamma_cq": tensors["gamma_cq"][r],
            "gamma_ckv": tensors["gamma_ckv"][r],
            "freqs_cos": tensors["freqs_cos"][r],
            "freqs_sin": tensors["freqs_sin"][r],
            "cmp_wkv": tensors["cmp_wkv"][r],
            "cmp_wgate": tensors["cmp_wgate"][r],
            "cmp_ape": tensors["cmp_ape"][r],
            "cmp_norm_w": tensors["cmp_norm_w"][r],
            "compress_state": tensors["compress_state"][r],
            "compress_state_block_table": tensors["compress_state_block_table"][r],
            "idx_wq_b": tensors["idx_wq_b"][r],
            "idx_wq_b_scale": tensors["idx_wq_b_scale"][r],
            "weights_proj": tensors["weights_proj"][r],
            "hadamard_idx": tensors["hadamard_idx"][r],
            "inner_wkv": tensors["inner_wkv"][r],
            "inner_wgate": tensors["inner_wgate"][r],
            "inner_ape": tensors["inner_ape"][r],
            "inner_norm_w": tensors["inner_norm_w"][r],
            "inner_compress_state": tensors["inner_compress_state"][r],
            "inner_compress_state_block_table": tensors["inner_compress_state_block_table"][r],
            "kv_cache": tensors["kv_cache"][r],
            "ori_block_table": tensors["ori_block_table"][r],
            "cmp_kv": tensors["cmp_kv"][r],
            "cmp_block_table": tensors["cmp_block_table"][r],
            "idx_kv_cache": tensors["idx_kv_cache"][r],
            "idx_block_table": tensors["idx_block_table"][r],
            "ori_slot_mapping": tensors["ori_slot_mapping"][r],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"][r],
            "idx_slot_mapping": tensors["idx_slot_mapping"][r],
            "state_slot_mapping": tensors["state_slot_mapping"][r],
            "inner_state_slot_mapping": tensors["inner_state_slot_mapping"][r],
            "position_ids": tensors["position_ids"][r],
            "kv_seq_lens": tensors["kv_seq_lens"][r],
            "attn_sink": tensors["attn_sink"][r],
            "wo_a": tensors["wo_a"][r],
            "wo_b": tensors["wo_b"][r],
            "wo_b_scale": tensors["wo_b_scale"][r],
            "x_out": x_attn[r],
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    moe_tensors["num_tokens"] = T
    golden_moe(moe_tensors)


def golden_decode_layer_auto(tensors):
    attention_mode = _attention_kind_for_layer(int(tensors["layer_id"]))
    if attention_mode == "swa":
        golden_decode_layer(tensors)
    elif attention_mode == "hca":
        mapped = dict(tensors)
        mapped.update({
            "cmp_wkv": tensors["hca_cmp_wkv"],
            "cmp_wgate": tensors["hca_cmp_wgate"],
            "cmp_ape": tensors["hca_cmp_ape"],
            "cmp_norm_w": tensors["hca_cmp_norm_w"],
            "compress_state": tensors["hca_compress_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"],
            "state_slot_mapping": tensors["hca_state_slot_mapping"],
            "ori_block_table": tensors["block_table"],
        })
        golden_decode_layer_hca(mapped)
    else:
        mapped = dict(tensors)
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"],
            "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"],
            "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "compress_state": tensors["csa_compress_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"],
            "weights_proj": tensors["csa_weights_proj"],
            "hadamard_idx": tensors["csa_hadamard_idx"],
            "inner_wkv": tensors["csa_inner_wkv"],
            "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"],
            "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_compress_state": tensors["csa_inner_compress_state"],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"],
            "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
            "state_slot_mapping": tensors["csa_state_slot_mapping"],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
            "ori_block_table": tensors["block_table"],
        })
        golden_decode_layer_csa(mapped)


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


def build_tensor_specs(start_pos=None, layer_id=10):
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


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="If set, use this single start_pos for all batches.")
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    host_fn = l3_decode_layer
    golden_fn = golden_decode_layer_auto

    result = run_jit(
        fn=host_fn,
        specs=build_tensor_specs(
            start_pos=args.start_pos,
            layer_id=args.layer_id,
        ),
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
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
            # swa(L0) 0.4% / 0.003%, hca(L9) 1.9% / 1.0%, csa(L8) 3.8% / 0.6%.
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
