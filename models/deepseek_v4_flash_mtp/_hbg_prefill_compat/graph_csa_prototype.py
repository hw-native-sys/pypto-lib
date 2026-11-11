# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode Indexer: q projection, Hadamard INT8 quant, inner compressor, paged C8 score and Top-K."""

import pypto.language as pl
from graph_csa_core import (
    M as M,
    DECODE_BATCH as DECODE_BATCH,
    DECODE_SEQ as DECODE_SEQ,
    BLOCK_SIZE as BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE as C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM as DECODE_CMP_BLOCK_NUM,
    DECODE_IDX_BLOCK_NUM as DECODE_IDX_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM as DECODE_ORI_BLOCK_NUM,
    KV_ORI_TABLE_MAX_BLOCKS as KV_ORI_TABLE_MAX_BLOCKS,
    INT8_SCALE_MAX as INT8_SCALE_MAX,
    INT8_AMAX_EPS as INT8_AMAX_EPS,
    MAIN_STATE_TABLE_BLOCKS_DYN as MAIN_STATE_TABLE_BLOCKS_DYN,
    compressor_ratio4 as compressor_ratio4,
    hc_post as hc_post,
    hc_pre as hc_pre,
    IDX_TABLE_BLOCKS_DYN as IDX_TABLE_BLOCKS_DYN,
    INNER_STATE_TABLE_BLOCKS_DYN as INNER_STATE_TABLE_BLOCKS_DYN,
    IDX_TOPK_LEAF as IDX_TOPK_LEAF,
    indexer as indexer,
    qkv_proj_rope as qkv_proj_rope,
    rms_norm as rms_norm,
    rope_interleave as rope_interleave,
    CMP_TABLE_BLOCKS_DYN as CMP_TABLE_BLOCKS_DYN,
    sparse_attn_csa as sparse_attn_csa,
    B as B,
    S as S,
    T as T,
    EPS as EPS,
    D as D,
    H as H,
    HEAD_DIM as HEAD_DIM,
    ROPE_HEAD_DIM as ROPE_HEAD_DIM,
    HALF_ROPE as HALF_ROPE,
    Q_LORA as Q_LORA,
    WIN as WIN,
    MAX_SEQ_LEN as MAX_SEQ_LEN,
    HC_MULT as HC_MULT,
    MIX_HC as MIX_HC,
    HC_DIM as HC_DIM,
    IDX_N_HEADS as IDX_N_HEADS,
    IDX_HEAD_DIM as IDX_HEAD_DIM,
    IDX_TOPK as IDX_TOPK,
    O_LORA as O_LORA,
    O_GROUPS as O_GROUPS,
    O_GROUP_IN as O_GROUP_IN,
    COMPRESS_RATIO as COMPRESS_RATIO,
    CMP_STORAGE_BLOCK_SIZE as CMP_STORAGE_BLOCK_SIZE,
    OVERLAP as OVERLAP,
    COFF as COFF,
    MAIN_OUT_DIM as MAIN_OUT_DIM,
    MAIN_STATE_DIM as MAIN_STATE_DIM,
    MAIN_STATE_BLOCK_SIZE as MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_PHYSICAL_BLOCKS as MAIN_STATE_PHYSICAL_BLOCKS,
    MAIN_STATE_BLOCK_NUM as MAIN_STATE_BLOCK_NUM,
    MAIN_STATE_BLOCK_NUM_DYN as MAIN_STATE_BLOCK_NUM_DYN,
    INNER_OUT_DIM as INNER_OUT_DIM,
    INNER_STATE_DIM as INNER_STATE_DIM,
    INNER_STATE_BLOCK_SIZE as INNER_STATE_BLOCK_SIZE,
    INNER_STATE_PHYSICAL_BLOCKS as INNER_STATE_PHYSICAL_BLOCKS,
    INNER_STATE_BLOCK_NUM as INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_NUM_DYN as INNER_STATE_BLOCK_NUM_DYN,
    IDX_CACHE_BLOCK_NUM as IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_BLOCK_NUM_DYN as IDX_CACHE_BLOCK_NUM_DYN,
    ORI_TABLE_MAX_BLOCKS as ORI_TABLE_MAX_BLOCKS,
    ORI_BLOCK_NUM as ORI_BLOCK_NUM,
    ORI_BLOCK_NUM_DYN as ORI_BLOCK_NUM_DYN,
    CMP_BLOCK_NUM as CMP_BLOCK_NUM,
    CMP_BLOCK_NUM_DYN as CMP_BLOCK_NUM_DYN,
    CSA_WB_TOKEN_TILE as CSA_WB_TOKEN_TILE,
    IDX_SCORE_PAGE_ROWS as IDX_SCORE_PAGE_ROWS,
    hc_pre_shape_core as hc_pre_shape_core,
    hc_post_shape_core as hc_post_shape_core,
    rms_norm_shape_core as rms_norm_shape_core,
    qkv_proj_rope_shape_core as qkv_proj_rope_shape_core,
    attention_csa_shape_core as attention_csa_shape_core,
    attention_csa as attention_csa,
    attention_csa_test as attention_csa_test,
    golden_attention_csa as golden_attention_csa,
    build_tensor_specs as build_tensor_specs,
    attention_csa_recorded as attention_csa_recorded,
    attention_csa_fixed_eager as attention_csa_fixed_eager,
    attention_csa_fixed_graph as attention_csa_fixed_graph,
)

def make_csa_whole_component(*,idx_table_blocks:int):
    if type(idx_table_blocks) is not int or idx_table_blocks<=0:
        raise ValueError("idx_table_blocks must come from actual layout.idx_max_blocks")

    @pl.jit.graph
    def csa_recorded(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[B, HALF_ROPE], pl.FP32], cmp_freqs_sin: pl.Tensor[[B, HALF_ROPE], pl.FP32], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.InOut[pl.Tensor], compress_state_block_table: pl.Tensor[[B, MAIN_STATE_TABLE_BLOCKS_DYN], pl.INT32], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8, pl.NZ], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16, pl.NZ], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor], inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_TABLE_BLOCKS_DYN], pl.INT32], kv_cache: pl.InOut[pl.Tensor], cmp_kv: pl.InOut[pl.Tensor], cmp_block_table: pl.Tensor[[B, CMP_TABLE_BLOCKS_DYN], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor], idx_kv_scale: pl.InOut[pl.Tensor], idx_block_table: pl.Tensor[[B, idx_table_blocks], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], window_swa_indices: pl.Tensor[[T, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], idx_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64], position_ids: pl.Tensor[[T], pl.INT32], kv_seq_lens: pl.Tensor[[B], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ], wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]], boundary_scale0: pl.Scalar[pl.FP32], boundary_scale1: pl.Scalar[pl.FP32], boundary_scale2: pl.Scalar[pl.FP32]) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        pl.static_assert(idx_block_table.shape[1] == idx_table_blocks, 'actual IDX table metadata mismatch')
        (kv_cache, compress_state, cmp_kv, inner_compress_state, idx_kv_cache, idx_kv_scale, x_out) = attention_csa_shape_core(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, x_out, True, T, boundary_scale0, boundary_scale1, boundary_scale2)
        return (kv_cache, compress_state, cmp_kv, inner_compress_state, idx_kv_cache, idx_kv_scale, x_out)

    @pl.jit(auto_scope=False)
    def csa_whole_component(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], cmp_freqs_cos: pl.Tensor[[B, HALF_ROPE], pl.FP32], cmp_freqs_sin: pl.Tensor[[B, HALF_ROPE], pl.FP32], cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16], cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32], cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32], compress_state_block_table: pl.Tensor[[B, MAIN_STATE_TABLE_BLOCKS_DYN], pl.INT32], idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8, pl.NZ], idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32], weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16, pl.NZ], hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16], inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16], inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32], inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32], inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_TABLE_BLOCKS_DYN], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16], cmp_block_table: pl.Tensor[[B, CMP_TABLE_BLOCKS_DYN], pl.INT32], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B, idx_table_blocks], pl.INT32], ori_slot_mapping: pl.Tensor[[T], pl.INT64], window_swa_indices: pl.Tensor[[T, WIN], pl.INT32], window_swa_lens: pl.Tensor[[T], pl.INT32], cmp_slot_mapping: pl.Tensor[[T], pl.INT64], idx_slot_mapping: pl.Tensor[[T], pl.INT64], state_slot_mapping: pl.Tensor[[T], pl.INT64], inner_state_slot_mapping: pl.Tensor[[T], pl.INT64], position_ids: pl.Tensor[[T], pl.INT32], kv_seq_lens: pl.Tensor[[B], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ], wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]]) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor, pl.Tensor]:
        pl.static_assert(idx_block_table.shape[1] == idx_table_blocks, 'actual IDX table metadata mismatch')
        pl.static_assert(x_hc.shape[0] == T, 'fixed decode token contract x_hc')
        pl.static_assert(freqs_cos.shape[0] == T, 'fixed decode token contract freqs_cos')
        pl.static_assert(freqs_sin.shape[0] == T, 'fixed decode token contract freqs_sin')
        pl.static_assert(ori_slot_mapping.shape[0] == T, 'fixed decode token contract ori_slot_mapping')
        pl.static_assert(window_swa_indices.shape[0] == T, 'fixed decode token contract window_swa_indices')
        pl.static_assert(window_swa_lens.shape[0] == T, 'fixed decode token contract window_swa_lens')
        pl.static_assert(cmp_slot_mapping.shape[0] == T, 'fixed decode token contract cmp_slot_mapping')
        pl.static_assert(idx_slot_mapping.shape[0] == T, 'fixed decode token contract idx_slot_mapping')
        pl.static_assert(state_slot_mapping.shape[0] == T, 'fixed decode token contract state_slot_mapping')
        pl.static_assert(inner_state_slot_mapping.shape[0] == T, 'fixed decode token contract inner_state_slot_mapping')
        pl.static_assert(position_ids.shape[0] == T, 'fixed decode token contract position_ids')
        pl.static_assert(x_out.shape[0] == T, 'fixed decode token contract x_out')
        boundary_scale0 = pl.read(hc_attn_scale, [0])
        boundary_scale1 = pl.read(hc_attn_scale, [1])
        boundary_scale2 = pl.read(hc_attn_scale, [2])
        (kv_cache, compress_state, cmp_kv, inner_compress_state, idx_kv_cache, idx_kv_scale, x_out) = csa_recorded(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, x_out, boundary_scale0, boundary_scale1, boundary_scale2)
        return (kv_cache, compress_state, cmp_kv, inner_compress_state, idx_kv_cache, idx_kv_scale, x_out)

    return csa_whole_component
