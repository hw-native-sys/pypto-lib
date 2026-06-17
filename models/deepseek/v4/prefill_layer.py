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

# Import moe_ep first. It applies the EP2 FLASH override before dependent
# modules bake config-derived MoE shapes. The stress case
# `--case basic128 --layer-id 3` was validated with
# `DSV4_MOE_EP_RECV_MAX=128`; default RECV_MAX=96 is not its pass criterion.
from moe_ep import (
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
    golden_moe_ep,
    moe_ep,
)
from config import FLASH as MODEL_CONFIG
from prefill_attention_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    _resolve_swa_case,
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
    _resolve_hca_case,
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
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_REQS,
    MAX_SEQ_LEN,
    MAX_TOKENS,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_TOPK,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    START_POS,
    _resolve_csa_case,
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_prefill_attention_csa,
    prefill_attention_csa,
)


assert MAX_TOKENS == T, "prefill attention and MoE EP must agree on token-major T"
assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM
assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM

ACTIVE_TOKEN_TILE = 16
ACTIVE_D_TILE = 512


@pl.jit
def prefill_layer(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
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
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    hca_cmp_kv_state: pl.Tensor[
        [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[MAX_REQS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    csa_cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    csa_inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[MAX_TOKENS, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[MAX_TOKENS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.FP32],
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
    x_attn: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
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
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    if layer_id < 2:
        kv_cache, x_attn = prefill_attention_swa(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            freqs_cos,
            freqs_sin,
            kv_cache,
            ori_block_table,
            ori_slot_mapping,
            cmp_sparse_indices,
            cmp_sparse_lens,
            token_to_request,
            position_ids,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_attn,
            num_tokens,
        )
    elif layer_id % 2 == 1:
        x_attn = prefill_attention_hca(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            freqs_cos,
            freqs_sin,
            hca_cmp_wkv,
            hca_cmp_wgate,
            hca_cmp_ape,
            hca_cmp_norm_w,
            hca_cmp_kv_state,
            hca_cmp_score_state,
            hca_compress_state_block_table,
            kv_cache,
            ori_slot_mapping,
            ori_block_table,
            cmp_kv,
            cmp_block_table,
            cmp_sparse_indices,
            cmp_sparse_lens,
            token_to_request,
            position_ids,
            hca_cmp_slot_mapping,
            hca_state_slot_mapping,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_attn,
            num_tokens,
        )
    else:
        (
            kv_cache,
            cmp_kv,
            csa_cmp_kv_state,
            csa_cmp_score_state,
            idx_kv_cache,
            csa_inner_kv_state,
            csa_inner_score_state,
            x_attn,
        ) = prefill_attention_csa(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            freqs_cos,
            freqs_sin,
            csa_cmp_wkv,
            csa_cmp_wgate,
            csa_cmp_ape,
            csa_cmp_norm_w,
            csa_cmp_kv_state,
            csa_cmp_score_state,
            csa_compress_state_block_table,
            csa_hadamard_idx,
            csa_inner_wkv,
            csa_inner_wgate,
            csa_inner_ape,
            csa_inner_norm_w,
            csa_inner_kv_state,
            csa_inner_score_state,
            csa_inner_compress_state_block_table,
            kv_cache,
            ori_block_table,
            ori_slot_mapping,
            cmp_kv,
            cmp_block_table,
            idx_kv_cache,
            idx_block_table,
            token_to_request,
            position_ids,
            csa_cmp_slot_mapping,
            csa_idx_slot_mapping,
            csa_state_slot_mapping,
            csa_inner_state_slot_mapping,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_attn,
            num_tokens,
        )
    x_moe = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_attn_flat = pl.reshape(x_attn, [T * HC_MULT, D])
    x_hc_flat = pl.reshape(x_hc, [T * HC_MULT, D])
    x_moe_flat = pl.reshape(x_moe, [T * HC_MULT, D])
    for act_t0 in pl.parallel(0, T, ACTIVE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_prepare_x_moe"):
            for act_dt in pl.range(ACTIVE_TOKEN_TILE):
                act_t = act_t0 + act_dt
                if act_t < T:
                    for act_h in pl.range(HC_MULT):
                        act_row = (act_t * HC_MULT) + act_h
                        for act_d0 in pl.range(0, D, ACTIVE_D_TILE):
                            if act_t < num_tokens:
                                act_tile = pl.load(
                                    x_attn_flat,
                                    [act_row, act_d0],
                                    [1, ACTIVE_D_TILE],
                                    target_memory=pl.MemorySpace.Vec,
                                )
                            else:
                                act_tile = pl.load(
                                    x_hc_flat,
                                    [act_row, act_d0],
                                    [1, ACTIVE_D_TILE],
                                    target_memory=pl.MemorySpace.Vec,
                                )
                            x_moe_flat = pl.store(act_tile, [act_row, act_d0], x_moe_flat)
    x_moe = pl.reshape(x_moe_flat, [T, HC_MULT, D])
    x_next = moe_ep(
        x_moe,
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
        layer_id, my_rank,
    )
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
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
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.FP32],
    hca_cmp_kv_state: pl.Tensor[
        [N_RANKS, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [N_RANKS, HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, MAX_REQS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.FP32],
    csa_cmp_kv_state: pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.FP32],
    csa_inner_kv_state: pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[N_RANKS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[N_RANKS, MAX_TOKENS, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[N_RANKS, MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
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
    x_attn: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
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
        prefill_layer(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_cmp_kv_state[rank], hca_cmp_score_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_cmp_kv_state[rank], csa_cmp_score_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_kv_state[rank], csa_inner_score_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            cmp_sparse_indices[rank], cmp_sparse_lens[rank],
            idx_kv_cache[rank], idx_block_table[rank],
            token_to_request[rank], position_ids[rank],
            hca_cmp_slot_mapping[rank], hca_state_slot_mapping[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank], csa_inner_state_slot_mapping[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_attn[rank],
            x_next[rank],
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            num_tokens, layer_id, rank,
            device=rank,
        )

HCA_NAME_MAP = {
    "cmp_wkv": "hca_cmp_wkv",
    "cmp_wgate": "hca_cmp_wgate",
    "cmp_ape": "hca_cmp_ape",
    "cmp_norm_w": "hca_cmp_norm_w",
    "cmp_kv_state": "hca_cmp_kv_state",
    "cmp_score_state": "hca_cmp_score_state",
    "compress_state_block_table": "hca_compress_state_block_table",
    "cmp_slot_mapping": "hca_cmp_slot_mapping",
    "state_slot_mapping": "hca_state_slot_mapping",
}

CSA_NAME_MAP = {
    "cmp_wkv": "csa_cmp_wkv",
    "cmp_wgate": "csa_cmp_wgate",
    "cmp_ape": "csa_cmp_ape",
    "cmp_norm_w": "csa_cmp_norm_w",
    "cmp_kv_state": "csa_cmp_kv_state",
    "cmp_score_state": "csa_cmp_score_state",
    "compress_state_block_table": "csa_compress_state_block_table",
    "hadamard_idx": "csa_hadamard_idx",
    "inner_wkv": "csa_inner_wkv",
    "inner_wgate": "csa_inner_wgate",
    "inner_ape": "csa_inner_ape",
    "inner_norm_w": "csa_inner_norm_w",
    "inner_kv_state": "csa_inner_kv_state",
    "inner_score_state": "csa_inner_score_state",
    "inner_compress_state_block_table": "csa_inner_compress_state_block_table",
    "cmp_slot_mapping": "csa_cmp_slot_mapping",
    "idx_slot_mapping": "csa_idx_slot_mapping",
    "state_slot_mapping": "csa_state_slot_mapping",
    "inner_state_slot_mapping": "csa_inner_state_slot_mapping",
}

SWA_NAME_MAP = {
    "block_table": "ori_block_table",
}

HOST_TENSOR_ORDER = (
    "x_hc",
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
    "token_to_request",
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
    "x_attn",
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
            gen = torch.Generator(device="cpu")
            gen.manual_seed(20260616 + active)
            inactive = torch.randn(
                stacked[:, active:].shape,
                generator=gen,
                dtype=torch.float32,
            ).to(stacked.dtype)
            stacked[:, active:] = inactive / 10.0
        return stacked

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


def _attention_specs(kind, start_pos=START_POS, num_tokens=MAX_TOKENS, case="custom"):
    if kind == "swa":
        return build_swa_attention_tensor_specs(
            start_pos=start_pos,
            num_tokens=num_tokens,
            swa_case=case,
        ), SWA_NAME_MAP, golden_prefill_attention_swa
    if kind == "hca":
        return build_hca_attention_tensor_specs(
            start_pos=start_pos,
            num_tokens=num_tokens,
            hca_case=case,
        ), HCA_NAME_MAP, golden_prefill_attention_hca
    if kind == "csa":
        return build_csa_attention_tensor_specs(
            start_pos=start_pos,
            num_tokens=num_tokens,
            csa_case=case,
        ), CSA_NAME_MAP, golden_prefill_attention_csa
    raise ValueError(f"unknown attention kind {kind}")


def _add_ranked_attention_specs(specs, name_map, tensor_specs, scalar_specs, tensor_names,
                                scalar_names, active_tokens, torch, TensorSpec):
    for spec in specs:
        if isinstance(spec, TensorSpec):
            if spec.name == "x_out":
                continue
            spec_name = name_map.get(spec.name, spec.name)
            if spec_name in tensor_names:
                continue
            tensor_specs.append(
                TensorSpec(
                    spec_name,
                    [N_RANKS, *spec.shape],
                    spec.dtype,
                    init_value=(
                        ranked_x_hc_init(spec, N_RANKS, active_tokens, torch)
                        if spec_name == "x_hc" else ranked_init(spec, N_RANKS, torch)
                    ),
                    is_output=spec.is_output,
                )
            )
            tensor_names.add(spec_name)
        else:
            if spec.name in scalar_names:
                continue
            scalar_specs.append(spec)
            scalar_names.add(spec.name)


def _add_moe_specs(moe_specs, tensor_specs, scalar_specs, tensor_names, scalar_names,
                   active_tokens, layer_id_value, torch, TensorSpec):
    for spec in moe_specs:
        if isinstance(spec, TensorSpec):
            if spec.name in {"x_hc", "x_next"} or spec.name in tensor_names:
                continue
            if spec.name == "tid2eid":
                def init_tid2eid(spec=spec):
                    _, vocab, topk = spec.shape
                    ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                    ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                    table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                    return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

                tensor_specs.append(
                    TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid, is_output=spec.is_output)
                )
            elif spec.name == "input_ids":
                def init_input_ids(spec=spec):
                    _, tokens = spec.shape
                    active = min(active_tokens, tokens)
                    rows = []
                    for rank in range(N_RANKS):
                        ids = torch.arange(tokens, dtype=spec.dtype)
                        row = torch.roll(ids, shifts=rank)
                        if layer_id_value >= 3 and active < tokens:
                            row[active:] = -1
                        rows.append(row)
                    return torch.stack(rows, dim=0).contiguous()

                tensor_specs.append(
                    TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_input_ids, is_output=spec.is_output)
                )
            else:
                tensor_specs.append(spec)
            tensor_names.add(spec.name)
        else:
            if spec.name in scalar_names:
                continue
            scalar_specs.append(spec)
            scalar_names.add(spec.name)


def build_tensor_specs(start_pos=START_POS, num_tokens=MAX_TOKENS, case="custom", layer_id=2):
    import torch
    from golden import TensorSpec

    active_kind = _attention_kind_for_layer(layer_id)
    ordered_kinds = [active_kind] + [kind for kind in ("swa", "hca", "csa") if kind != active_kind]
    tensor_specs = []
    scalar_specs = []
    tensor_names = set()
    scalar_names = set()
    active_tokens = num_tokens
    for kind in ordered_kinds:
        specs, name_map, _ = _attention_specs(
            kind,
            start_pos=start_pos,
            num_tokens=num_tokens,
            case=case if kind == active_kind else "custom",
        )
        if kind == active_kind:
            for spec in specs:
                if not isinstance(spec, TensorSpec) and spec.name == "num_tokens":
                    active_tokens = int(spec.value.item())
                    break
        _add_ranked_attention_specs(
            specs,
            name_map,
            tensor_specs,
            scalar_specs,
            tensor_names,
            scalar_names,
            active_tokens,
            torch,
            TensorSpec,
        )

    moe_specs = build_moe_tensor_specs(layer_id=layer_id)
    _add_moe_specs(
        moe_specs,
        tensor_specs,
        scalar_specs,
        tensor_names,
        scalar_names,
        active_tokens,
        layer_id,
        torch,
        TensorSpec,
    )
    tensor_specs.append(TensorSpec("x_attn", [N_RANKS, T, HC_MULT, D], torch.bfloat16, is_output=True))
    tensor_specs.append(TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.bfloat16, is_output=True))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    scalar_by_name = {spec.name: spec for spec in scalar_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    ordered_scalars = [scalar_by_name[name] for name in ("num_tokens", "layer_id") if name in scalar_by_name]
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + ordered_scalars


def golden_prefill_layer(tensors):
    import torch
    from golden import TensorSpec

    layer_id = int(tensors["layer_id"])
    kind = _attention_kind_for_layer(layer_id)
    attention_specs, name_map, attention_golden = _attention_specs(
        kind,
        num_tokens=int(tensors["num_tokens"]),
    )
    x_attn = torch.zeros_like(tensors["x_hc"])
    for rank in range(N_RANKS):
        attn_tensors = {}
        for spec in attention_specs:
            if isinstance(spec, TensorSpec):
                if spec.name == "x_out":
                    attn_tensors[spec.name] = x_attn[rank]
                else:
                    source_name = name_map.get(spec.name, spec.name)
                    attn_tensors[spec.name] = tensors[source_name][rank]
            else:
                attn_tensors[spec.name] = tensors[spec.name]
        attention_golden(attn_tensors)

    tensors["x_attn"][:] = x_attn
    moe_x = x_attn.clone()
    num_tokens = int(tensors.get("num_tokens", moe_x.shape[1]))
    if num_tokens < moe_x.shape[1]:
        moe_x[:, num_tokens:] = tensors["x_hc"][:, num_tokens:]
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = moe_x
    golden_moe_ep(moe_tensors)


def active_ranked_x_next_compare(num_tokens):
    from golden import ratio_reldiff

    base_compare = ratio_reldiff(diff_thd=0.1, pct_thd=0.05)

    def compare(actual, expected, **kwargs):
        return base_compare(actual[:, :num_tokens], expected[:, :num_tokens], **kwargs)

    compare.__name__ = f"active_ranked_x_next_compare(num_tokens={num_tokens})"
    return compare


def _resolve_compare_tokens(args):
    kind = _attention_kind_for_layer(args.layer_id)
    if kind == "swa":
        q_lens_values, _, num_tokens, _ = _resolve_swa_case(
            args.start_pos,
            args.num_tokens,
            args.case,
            False,
            False,
        )
    elif kind == "hca":
        _, q_lens_values, _, num_tokens = _resolve_hca_case(
            args.start_pos,
            args.num_tokens,
            args.case,
            False,
            False,
            False,
        )
    else:
        _, q_lens_values, _, num_tokens = _resolve_csa_case(
            args.start_pos,
            args.num_tokens,
            args.case,
            False,
            False,
        )
    return min(num_tokens, sum(q_lens_values))


def _compare_fns(kind, compare_tokens):
    from golden import ratio_allclose

    compare_fn = {
        "x_attn": active_ranked_x_next_compare(compare_tokens),
        "x_next": active_ranked_x_next_compare(compare_tokens),
        "kv_cache": ratio_allclose(atol=3e-3 if kind == "hca" else 1e-4, rtol=1e-2),
    }
    if kind in {"hca", "csa"}:
        compare_fn["cmp_kv"] = ratio_allclose(atol=5e-3, rtol=1e-2)
    if kind == "csa":
        compare_fn["idx_kv_cache"] = ratio_allclose(atol=5e-3, rtol=1e-2)
    return compare_fn


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help="comma-separated device ids; need at least 2")
    parser.add_argument("--layer-id", type=int, default=2,
                        help="Layer id selects attention by MODEL_CONFIG.compress_ratios[layer_id].")
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only prefix length for --case=custom.")
    parser.add_argument("--num-tokens", type=int, default=MAX_TOKENS,
                        help="Fixture active token count for --case=custom.")
    parser.add_argument("--case", type=str, default="custom",
                        help="Attention fixture case for the selected layer type.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    kind = _attention_kind_for_layer(args.layer_id)
    compare_tokens = _resolve_compare_tokens(args)
    result = run_jit(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(
            start_pos=args.start_pos,
            num_tokens=args.num_tokens,
            case=args.case,
            layer_id=args.layer_id,
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
        compare_fn=_compare_fns(kind, compare_tokens),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
