# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 packed prefill CSA single layer with MoE EP2."""

import os

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# Keep the default aligned with standalone moe_ep. The stress case
# `--csa-case basic128 --layer-id 3` was validated with
# `DSV4_MOE_EP_RECV_MAX=128`; default RECV_MAX=96 is not its pass criterion.
os.environ.setdefault("DSV4_MOE_EP_RECV_MAX", "96")
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
    prefill_layer_select_active_x_attn,
)
from hc_pre import hc_pre
from hc_post import hc_post
from prefill_attention_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO,
    CSA_CASES,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_TOPK,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM,
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
    _prefill_csa_assemble_sparse_indices,
    _prefill_csa_cache_writeback_overlay,
    _resolve_csa_case,
    build_tensor_specs as build_attention_tensor_specs,
    golden_prefill_attention_csa,
)
from prefill_compressor_ratio4 import prefill_compressor_ratio4
from prefill_indexer import prefill_indexer
from prefill_layer_ep_common import (
    active_ranked_x_next_compare,
    build_ranked_layer_specs,
    golden_prefill_layer_ep,
)
from qkv_proj_rope import materialize_rope_rows, qkv_proj_rope
from rmsnorm import attn_norm
from prefill_sparse_attn import prefill_sparse_attn_padded_indices


assert MAX_TOKENS == T, "prefill CSA and MoE EP must agree on token-major T"


@pl.jit
def prefill_layer_csa_moe_ep(
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
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    idx_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
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
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        x_mixed, post, comb,
    )

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed = attn_norm(x_mixed, attn_norm_w, x_normed)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos_t, rope_sin_t)
    q = qkv_proj_rope(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    cmp_kv, cmp_kv_state, cmp_score_state = prefill_compressor_ratio4(
        x_normed,
        cmp_kv_state,
        cmp_score_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        freqs_cos,
        freqs_sin,
        cmp_kv,
        token_to_request,
        position_ids,
        num_tokens,
        cmp_slot_mapping,
        state_slot_mapping,
    )
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_kv_cache, inner_kv_state, inner_score_state, cmp_topk_indices = prefill_indexer(
        x_normed,
        freqs_cos,
        freqs_sin,
        hadamard_idx,
        inner_kv_state,
        inner_score_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_block_table,
        cmp_topk_indices,
        token_to_request,
        position_ids,
        num_tokens,
        idx_slot_mapping,
        inner_state_slot_mapping,
    )

    cmp_sparse_work = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    cmp_sparse_work = _prefill_csa_assemble_sparse_indices(
        cmp_topk_indices,
        token_to_request,
        position_ids,
        num_tokens,
        cmp_sparse_work,
    )
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn_padded_indices(
        q,
        kv_cache,
        ori_block_table,
        kv,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_work,
        attn_sink,
        token_to_request,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    _prefill_csa_cache_writeback_overlay(kv, kv_cache, ori_slot_mapping, attn_out, num_tokens)
    x_attn = hc_post(attn_out, x_hc, post, comb, x_attn)
    x_moe = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_moe = prefill_layer_select_active_x_attn(x_attn, x_hc, x_moe, num_tokens)
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
def l3_prefill_layer_csa_moe_ep(
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
    cmp_wkv: pl.Tensor[[N_RANKS, D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[N_RANKS, CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[N_RANKS, MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[N_RANKS, D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[N_RANKS, COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[N_RANKS, INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[N_RANKS, MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[N_RANKS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[N_RANKS, MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    idx_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[N_RANKS, MAX_TOKENS], pl.INT64],
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
        prefill_layer_csa_moe_ep(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            cmp_kv_state[rank], cmp_score_state[rank], compress_state_block_table[rank],
            hadamard_idx[rank],
            inner_wkv[rank], inner_wgate[rank], inner_ape[rank], inner_norm_w[rank],
            inner_kv_state[rank], inner_score_state[rank], inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_block_table[rank],
            token_to_request[rank], position_ids[rank],
            cmp_slot_mapping[rank], idx_slot_mapping[rank],
            state_slot_mapping[rank], inner_state_slot_mapping[rank],
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

def build_tensor_specs(start_pos=START_POS, num_tokens=MAX_TOKENS, csa_case="custom",
                       hetero_smoke=False, hetero_boundary=False, layer_id=2):
    import torch
    from golden import TensorSpec

    attention_specs = build_attention_tensor_specs(
        start_pos=start_pos,
        num_tokens=num_tokens,
        csa_case=csa_case,
        hetero_smoke=hetero_smoke,
        hetero_boundary=hetero_boundary,
    )
    moe_specs = build_moe_tensor_specs(layer_id=layer_id)
    return build_ranked_layer_specs(
        attention_specs,
        moe_specs,
        N_RANKS,
        [N_RANKS, T, HC_MULT, D],
        torch,
        TensorSpec,
        n_experts_global=N_EXPERTS_GLOBAL,
    )

def golden_prefill_layer_csa_moe_ep(tensors):
    import torch
    from golden import TensorSpec

    attention_specs = build_attention_tensor_specs(num_tokens=int(tensors["num_tokens"]))
    golden_prefill_layer_ep(
        tensors,
        attention_specs,
        golden_prefill_attention_csa,
        golden_moe_ep,
        N_RANKS,
        torch,
        TensorSpec,
    )


def _resolve_compare_tokens(args):
    _, q_lens_values, _, num_tokens = _resolve_csa_case(
        args.start_pos,
        args.num_tokens,
        args.csa_case,
        args.hetero_smoke,
        args.hetero_boundary,
    )
    return min(num_tokens, sum(q_lens_values))


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help="comma-separated device ids; need at least 2")
    parser.add_argument("--layer-id", type=int, default=2)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only prefix length for --csa-case=custom.")
    parser.add_argument("--num-tokens", type=int, default=MAX_TOKENS,
                        help="Fixture active token count for --csa-case=custom.")
    parser.add_argument("--csa-case", type=str, default="custom", choices=CSA_CASES)
    parser.add_argument("--hetero-smoke", action="store_true", default=False)
    parser.add_argument("--hetero-boundary", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    compare_tokens = _resolve_compare_tokens(args)
    result = run_jit(
        fn=l3_prefill_layer_csa_moe_ep,
        specs=build_tensor_specs(
            start_pos=args.start_pos,
            num_tokens=args.num_tokens,
            csa_case=args.csa_case,
            hetero_smoke=args.hetero_smoke,
            hetero_boundary=args.hetero_boundary,
            layer_id=args.layer_id,
        ),
        golden_fn=golden_prefill_layer_csa_moe_ep,
        compile_only=args.platform.endswith("sim"),
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
            "x_attn": active_ranked_x_next_compare(compare_tokens),
            "x_next": active_ranked_x_next_compare(compare_tokens),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
            "cmp_kv": ratio_allclose(atol=5e-3, rtol=1e-2),
            "idx_kv_cache": ratio_allclose(atol=5e-3, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
