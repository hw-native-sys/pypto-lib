# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import pypto.language as pl
from graph_decode_swa import (
    M as M,
    DECODE_BATCH as DECODE_BATCH,
    DECODE_ORI_BLOCK_NUM as DECODE_ORI_BLOCK_NUM,
    KV_ORI_TABLE_MAX_BLOCKS as KV_ORI_TABLE_MAX_BLOCKS,
    DECODE_SEQ as DECODE_SEQ,
    BLOCK_SIZE as BLOCK_SIZE,
    INT8_SCALE_MAX as INT8_SCALE_MAX,
    INT8_AMAX_EPS as INT8_AMAX_EPS,
    hc_pre as hc_pre,
    hc_post as hc_post,
    qkv_proj_rope as qkv_proj_rope,
    rms_norm as rms_norm,
    sparse_attn_swa as sparse_attn_swa,
    B as B,
    S as S,
    T as T,
    EPS as EPS,
    D as D,
    H as H,
    HEAD_DIM as HEAD_DIM,
    ROPE_HEAD_DIM as ROPE_HEAD_DIM,
    NOPE_HEAD_DIM as NOPE_HEAD_DIM,
    Q_LORA as Q_LORA,
    WIN as WIN,
    SOFTMAX_SCALE as SOFTMAX_SCALE,
    HC_MULT as HC_MULT,
    MIX_HC as MIX_HC,
    HC_DIM as HC_DIM,
    HC_SINKHORN_ITER as HC_SINKHORN_ITER,
    HC_EPS as HC_EPS,
    MAX_SEQ_LEN as MAX_SEQ_LEN,
    O_LORA as O_LORA,
    O_GROUPS as O_GROUPS,
    O_GROUP_IN as O_GROUP_IN,
    ORI_TABLE_MAX_BLOCKS as ORI_TABLE_MAX_BLOCKS,
    ORI_BLOCK_NUM as ORI_BLOCK_NUM,
    ORI_BLOCK_NUM_DYN as ORI_BLOCK_NUM_DYN,
    TOPK as TOPK,
    SPARSE_IDX_TOPK as SPARSE_IDX_TOPK,
    SPARSE_TOPK as SPARSE_TOPK,
    SPARSE_ROPE_TILE as SPARSE_ROPE_TILE,
    SPARSE_ROPE_INTERLEAVE_TILE as SPARSE_ROPE_INTERLEAVE_TILE,
    NEG_INF as NEG_INF,
    hc_pre_shape_core as hc_pre_shape_core,
    rms_norm_shape_core as rms_norm_shape_core,
    qkv_proj_rope_shape_core as qkv_proj_rope_shape_core,
    hc_post_shape_core as hc_post_shape_core,
    attention_swa_shape_core as attention_swa_shape_core,
    attention_swa as attention_swa,
    attention_swa_test as attention_swa_test,
    golden_attention_swa as golden_attention_swa,
    build_tensor_specs as build_tensor_specs,
    attention_swa_recorded as attention_swa_recorded,
    attention_swa_fixed_eager as attention_swa_fixed_eager,
    attention_swa_fixed_graph as attention_swa_fixed_graph,
    attention_swa_graph_test as attention_swa_graph_test,
)


def _make_swa_main_graph(physical_tokens):
    @pl.jit.graph
    def swa_main_recorded(
        x_hc: pl.Tensor[[physical_tokens, HC_MULT, D], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
        wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
        wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
        gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
        gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
        freqs_cos: pl.Tensor[[physical_tokens, ROPE_HEAD_DIM], pl.BF16],
        freqs_sin: pl.Tensor[[physical_tokens, ROPE_HEAD_DIM], pl.BF16],
        kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        swa_slot_mapping: pl.Tensor[[physical_tokens], pl.INT64],
        swa_indices: pl.Tensor[[physical_tokens, WIN], pl.INT32],
        swa_lens: pl.Tensor[[physical_tokens], pl.INT32],
        position_ids: pl.Tensor[[physical_tokens], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        x_out: pl.Out[pl.Tensor[[physical_tokens, HC_MULT, D], pl.FP32]],
        boundary_scale0: pl.Scalar[pl.FP32],
        boundary_scale1: pl.Scalar[pl.FP32],
        boundary_scale2: pl.Scalar[pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        pl.static_assert(x_hc.shape[0] == physical_tokens, "physical token metadata mismatch")
        pl.static_assert(x_out.shape[0] == physical_tokens, "physical output metadata mismatch")
        x_out = attention_swa_shape_core(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, x_out, True, physical_tokens, boundary_scale0, boundary_scale1, boundary_scale2)
        return kv_cache, x_out
    return swa_main_recorded


def _make_swa_mtp_graph(physical_tokens):
    @pl.jit.graph
    def swa_mtp_recorded(
        x_hc: pl.Tensor[[physical_tokens, HC_MULT, D], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
        wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
        wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
        gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
        gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
        freqs_cos: pl.Tensor[[physical_tokens, ROPE_HEAD_DIM], pl.BF16],
        freqs_sin: pl.Tensor[[physical_tokens, ROPE_HEAD_DIM], pl.BF16],
        kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        swa_slot_mapping: pl.Tensor[[physical_tokens], pl.INT64],
        swa_indices: pl.Tensor[[physical_tokens, WIN], pl.INT32],
        swa_lens: pl.Tensor[[physical_tokens], pl.INT32],
        position_ids: pl.Tensor[[physical_tokens], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        x_out: pl.Out[pl.Tensor[[physical_tokens, HC_MULT, D], pl.FP32]],
        boundary_scale0: pl.Scalar[pl.FP32],
        boundary_scale1: pl.Scalar[pl.FP32],
        boundary_scale2: pl.Scalar[pl.FP32],
    ) -> tuple[pl.Tensor, pl.Tensor]:
        pl.static_assert(x_hc.shape[0] == physical_tokens, "physical token metadata mismatch")
        pl.static_assert(x_out.shape[0] == physical_tokens, "physical output metadata mismatch")
        x_out = attention_swa_shape_core(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, x_out, True, physical_tokens, boundary_scale0, boundary_scale1, boundary_scale2)
        return kv_cache, x_out
    return swa_mtp_recorded


# This module is loaded only after the existing Serving layout sets B/S/T.
# The body references make these real captured constexpr values, not annotation aliases.
swa_main_recorded = _make_swa_main_graph(T)
swa_mtp_recorded = _make_swa_mtp_graph(T)


@pl.jit.inline
def attention_swa_main_graph(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale_host: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], swa_slot_mapping: pl.Tensor[[T], pl.INT64], swa_indices: pl.Tensor[[T, WIN], pl.INT32], swa_lens: pl.Tensor[[T], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ], wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]]) -> tuple[pl.Tensor, pl.Tensor]:
    pl.static_assert(x_hc.shape[0] == T, 'fixed decode token contract: x_hc axis 0')
    pl.static_assert(freqs_cos.shape[0] == T, 'fixed decode token contract: freqs_cos axis 0')
    pl.static_assert(freqs_sin.shape[0] == T, 'fixed decode token contract: freqs_sin axis 0')
    pl.static_assert(swa_slot_mapping.shape[0] == T, 'fixed decode token contract: swa_slot_mapping axis 0')
    pl.static_assert(swa_indices.shape[0] == T, 'fixed decode token contract: swa_indices axis 0')
    pl.static_assert(swa_lens.shape[0] == T, 'fixed decode token contract: swa_lens axis 0')
    pl.static_assert(position_ids.shape[0] == T, 'fixed decode token contract: position_ids axis 0')
    pl.static_assert(x_out.shape[0] == T, 'fixed decode token contract: x_out axis 0')
    scale0 = pl.read(hc_attn_scale_host, [0])
    scale1 = pl.read(hc_attn_scale_host, [1])
    scale2 = pl.read(hc_attn_scale_host, [2])
    (kv_cache, x_out) = swa_main_recorded(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, x_out, scale0, scale1, scale2)
    return (kv_cache, x_out)


@pl.jit.inline
def attention_swa_mtp_graph(x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32], hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32], hc_attn_scale_host: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32], attn_norm_w: pl.Tensor[[D], pl.BF16], wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ], wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ], wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32], wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ], gamma_cq: pl.Tensor[[Q_LORA], pl.BF16], gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16], freqs_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], freqs_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]], swa_slot_mapping: pl.Tensor[[T], pl.INT64], swa_indices: pl.Tensor[[T, WIN], pl.INT32], swa_lens: pl.Tensor[[T], pl.INT32], position_ids: pl.Tensor[[T], pl.INT32], attn_sink: pl.Tensor[[H], pl.FP32], wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ], wo_b: pl.Tensor[[O_GROUPS, D, O_LORA], pl.INT8, pl.NZ], wo_b_scale: pl.Tensor[[D], pl.FP32], x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]]) -> tuple[pl.Tensor, pl.Tensor]:
    pl.static_assert(x_hc.shape[0] == T, 'fixed decode token contract: x_hc axis 0')
    pl.static_assert(freqs_cos.shape[0] == T, 'fixed decode token contract: freqs_cos axis 0')
    pl.static_assert(freqs_sin.shape[0] == T, 'fixed decode token contract: freqs_sin axis 0')
    pl.static_assert(swa_slot_mapping.shape[0] == T, 'fixed decode token contract: swa_slot_mapping axis 0')
    pl.static_assert(swa_indices.shape[0] == T, 'fixed decode token contract: swa_indices axis 0')
    pl.static_assert(swa_lens.shape[0] == T, 'fixed decode token contract: swa_lens axis 0')
    pl.static_assert(position_ids.shape[0] == T, 'fixed decode token contract: position_ids axis 0')
    pl.static_assert(x_out.shape[0] == T, 'fixed decode token contract: x_out axis 0')
    scale0 = pl.read(hc_attn_scale_host, [0])
    scale1 = pl.read(hc_attn_scale_host, [1])
    scale2 = pl.read(hc_attn_scale_host, [2])
    (kv_cache, x_out) = swa_mtp_recorded(x_hc, hc_attn_fn, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids, attn_sink, wo_a, wo_b, wo_b_scale, x_out, scale0, scale1, scale2)
    return (kv_cache, x_out)
